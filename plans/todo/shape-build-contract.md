# `panelary/shape/` — build contract

> **Status (2026-09-28): partially implemented.** Wave 1 (M1–M3) is built and
> tested. The pieces are the spine (`_axes`, `_tensor`, `_window`, `_array`), all
> §4 primitives (`_rsvd`, `_project`, `_sketch`, `_id`, `_paa`, `_spectral`,
> `_delay`, `_tucker`), `_explain` (`explain()`, `stability`, `procrustes`), the
> registry extension, the `reduce/` options (`PanelPCA`/`PanelSVD`
> `backend="sklearn"|"numpy"|"auto"`, `PanelRandomProjection`
> `method="gaussian"|"sparse"|"srht"`), the `pn.reduce` / `pn.features` method
> wiring, and `docs/user-guide/shape.md` + `docs/api-reference/shape.md`.
> The §9 tests `tests/test_shape_{prefix_invariance,leak_safety,stateless,roundtrip,agreement,window,verbs}.py`
> pass, and so does the conformance suite; the tensorly cross-check skips
> because tensorly is not installed.
>
> **Not done:**
> - M4 (Wave 2) has not started.
> - The full `make check` was not run.
> - `llms.txt`, `mkdocs.yml`, `CHANGELOG.md` and the top-level `pn.shape`
>   export are orchestrator-owned.
> - `Pipeline` does not call `plan()` (§3.4). `plan_chain()` exists instead.
>
> **Deviations:**
> - `shape/__init__.py` is lazy (PEP 562). `import panelary` loads only
>   `panelary.shape` + `_tensor` (the `cluster` re-export makes §9.7's "does
>   not pull `shape/`" impossible verbatim). The catalogue registers on first
>   access.
> - Procrustes alignment lives in `shape/_explain.py`, because `shape` must not
>   import `embed`.
> - `PartialTucker` and `CrossSectionalRandomizedPCA` stay out of the registry.
>
> **§9 targets, measured (Accelerate BLAS; thread caps had no measurable
> effect):**
> - `RandomizedPCA(k=16)` on the wide panel fits in 1.18 s, against 2.45 s for
>   `sklearn` `PCA(svd_solver="randomized")`. That meets the target, but only
>   after switching to CholeskyQR2; with Householder QR it was 3.7 s.
>   `sklearn`'s `"auto"` solver (`covariance_eigh`, 0.74 s) is faster still,
>   so the `backend` default stays `"sklearn"`.
> - Trailing PAA / Spectral have no per-entity Python loop at all. On a
>   1M-row panel the kernel is 64% (PAA, 122 ms total) and 97% (Spectral) of
>   the time.
>
> The full table is in `docs/user-guide/shape.md` under "Measured performance".

A **shape algebra** for panels: transforms that change the *shape* of a panel
(the width of a named axis, the order of the tensor, or the numerical rank of a
factorization) while leaving the meaning of `entity` / `time` / `feature`
intact.

Pure **numpy + polars** in the default path. No Rust, no compiled extension, no
Polars expression plugin, no scipy/sklearn/tensorly in any import that the
default path reaches. Optional backends go behind `panelary._internal._deps.require`.

Reuse, do not reinvent:
`panelary.core.protocol.PanelTransformer` (the `panel_safe` / `leakage_safe`
contract), `panelary.registry` (`FeatureSpec`, `register_feature`, `audit`),
`panelary.cluster._tensor.build_tensor` (the existing, already-correct long→tensor
builder — §3.1 moves it here), `panelary.reduce._base._PanelReducer` (the
fit/sign-fix/column-emit machinery), `panelary.reduce._n_factors` (`bai_ng`,
`eigenvalue_ratio`), `panelary.core.model_selection` (`PurgedKFold`),
`panelary.core.pipeline.Pipeline`.

Issued 2026-09-09, in response to the external report
`PanelKit_Dimensional_Rank_Transforms_Research.pdf` (filed under its
pre-rename name). That report is a good survey and its **central idea is
adopted**: model transform *intent* — compress an axis, lift into a basis,
factorize numerical rank — rather than shipping a bag of "dimensionality
reducers". §1.2 below records the four places where its recommendations are
**not** adopted, because they are incompatible with constraints this repo has
already committed to.

---

## 0. TL;DR — the decision

**Build the materialization boundary first, and the linear algebra second.**

Every method in the report's top ten assumes it is handed a dense rectangular
`X[entity, time, feature]`. Panels are long-format, ragged, misaligned and
missing-valued. Getting from long Polars to that array — and back — with an
explicit, declared policy for raggedness *is the whole problem*; randomized SVD
is forty lines of numpy underneath it. The report never mentions this step. It
is §3.1–§3.2 here, and it blocks everything else.

**And: the report's own shape algebra leaks.** Its worked example

```python
paa(axis="time", segments=32)      # X[E,T,F] -> Z[E,32,F]
```

pools each entity's **entire** series into 32 segments. Segment 0 of that
output is a function of the whole series, including its future. Used as a row
feature at time `t`, that is textbook look-ahead. The same defect applies to its
`fft(axis="time", bins=32)`, `partial_tucker`, `signature(level=3)` and `stft()`
examples: all are whole-series operations presented as if they were features.

So every time-axis transform in this module ships in **two flavours, and the
causal one is the default**:

| flavour | contract | what it is for |
|---|---|---|
| `trailing` (default, `window=W`) | one output row per input row, from that row's own trailing `W` observations within its entity. `leakage_safe = True`. | row features — the thing you put in a model |
| `whole_series` (explicit opt-in) | one output row per **entity**, from all of that entity's data. `leakage_safe = False`, enforced by `_check_leakage`. | clustering series, describing a fixed historical sample, plotting |

There is no third flavour, and `whole_series` never silently becomes a row
feature: it returns a frame keyed by `entity` alone, so the shape of the return
value makes the misuse a join error rather than a silent leak.

That distinction is the module's reason to exist. Everything else is
implementation.

---

## 1. Evidence base and where it is overridden

### 1.1 Adopted from the external report

| Idea | Where it lands |
|---|---|
| Three separate meanings of "rank": tensor **order**, **axis width**, **numerical rank**. Do not conflate. | §2 — the `Intent` enum, and why this module is not called `reduce`. |
| Name the semantic axis, never the position. `3_to_2` is unreadable; `axis="feature"` is not. | §2.2 axis vocabulary. |
| Lift → compress is the interesting pipeline, and the two halves must stay separate primitives. | §4.6, and it is also the `embed/` contract's §0 four-layer architecture — the two agree. |
| Interpolative decomposition / CUR return *real columns*, so the compression is readable. | §4.5, promoted from "nice property" to **the explainability story** (§5). |
| Partial Tucker preserves the entity axis while compressing time and feature. | §4.7. Built in numpy; no TensorLy dependency (§1.2.4). |
| Fixed-width embeddings belong in one `pl.Array(Float32, D)` column, not `D` loose columns. | §3.3. |
| Cost hints so a planner can refuse an accidental O(T²) expansion. | §3.4 `plan()`. |
| GAF / recurrence plots / MTF are a deliberate rank *expansion*, valuable but O(T²). | §6 — not shipped, and the reason is recorded. |

### 1.2 Not adopted, with reasons

**1. The entire Rust / Polars-plugin strategy (report §7, §7.1, §7.3, and the
`faer` / `rsvd-faer` / `rustfft` / `petal-decomposition` / `TenRSo` column of the
top-40 table).**

`AGENTS.md`: *"Do not reintroduce Rust / a compiled extension. 0.4.0 is
deliberately pure-Python and `tests/test_wheel_guardrails.py` asserts the wheel
is `py3-none-any` with no compiled artifacts."* The Rust extension was removed
**in this release cycle**, on purpose. A plan whose performance story is "write a
Polars expression plugin" is not implementable here, and the report's §7.4 ("avoid
Python UDFs in the hot path") is advice we cannot take.

This is less costly than it sounds, because the primitives that matter are
BLAS-bound, not loop-bound. `numpy.linalg.qr`, `numpy.linalg.svd` and `X @ Omega`
already run in compiled, multi-threaded LAPACK/BLAS; `numpy.fft.rfft` is
compiled pocketfft. Randomized SVD on an `n x d` matrix is two `@` products, a
`qr` and a small `svd` — **all** of that time is inside numpy. The Python cost is
per-*call*, not per-element, so the design rule is:

> **Batch to amortize the Python frame.** One `(n_windows, W)` array through one
> numpy call, never one call per window. This is the same lesson `embed/` §3.1
> records for catch22, and the same reason `AGENTS.md` bans `rolling_map`.

Where that is not enough, the escape hatch is `numba` behind the existing `fast`
extra with a pure-numpy default — not a new language.

**2. MiniROCKET / MultiROCKET / HYDRA / QUANT / TensorSketch / RFF / sparse-RP
as deliverables of this module.**

`plans/todo/embed-build-contract.md` was issued 2026-09-09 and already owns the
convolutional and kernel feature families, with a decision (ship QUANT before
MiniRocket, on leak-surface grounds) that this plan does not reopen. Two modules
implementing sparse random projection is exactly the drift `AGENTS.md` warns
about. §7 fixes the boundary: **`shape/` owns the projection and sketching
primitives; `embed/` imports them.**

**3. `TransformSpec` as a new 15-field metadata class (report §6.1).**

`panelary/registry.py` already has `FeatureSpec` — frozen, slotted, carrying
`input_shape` / `output_shape` / `tier` / `panel_safe` / `leakage_safe` /
`source` / `license`, with a working `registry.audit()` that enforces permissive
licensing and a `to_llms_txt()` renderer the agent layer consumes. A second,
parallel spec class would fork the catalogue. **Extend the one that exists**
(§2.3), and every transform here registers into it.

**4. "Partial Tucker API, initially via a TensorLy adapter" (report §9, Phase 1).**

Partial Tucker via HOSVD/HOOI is mode-unfold → thin SVD → mode-product: roughly
sixty lines of numpy over `np.moveaxis` / `np.tensordot`, reusing the `_rsvd.py`
we are already building for the matrix case. Taking a mandatory-in-practice
dependency for that trades a small amount of code for a large amount of
dependency surface, in a library whose required footprint is `numpy` + `polars`.
Ship numpy; keep `tensorly` as a **test-only** cross-check (§8) and as an
optional backend for the Wave-3 formats (CP, PARAFAC2, TT) where the algorithms
are genuinely iterative and fiddly.

### 1.3 Facts established by inspection of this repo (2026-09-09)

Verified, not assumed:

- `panelary/cluster/_tensor.py::build_tensor` already produces
  `(n_entities, n_times, n_values)` with **exactly** the policy this module
  needs: dense grid by cross join, `forward_fill().over(entity)` only (never
  backward), optional per-entity *expanding* z-normalisation, NaN where history
  is insufficient, plus the `entities` / `times` / `keys` indices needed to
  realign. It is written as a k-Shape helper but nothing about it is
  k-Shape-specific.
- Nothing named `hankel`, `minirocket`, `tucker`, `tensorly`, `countsketch`,
  `frequent_direction`, `SRHT` or `johnson` exists anywhere in `panelary/`.
  `np.fft.rfft` appears only inside `feature_extractors.py` and
  `econ/features/_longmemory.py` as an implementation detail of specific
  features — there is no spectral *axis transform*.
- `panelary/reduce/` covers the row-wise feature-reduction case (`PanelPCA`,
  `PanelSVD`, `PanelRandomProjection`, `PanelKernelPCA`, `PanelNMF`, `PanelUMAP`)
  and latent-factor extraction (`PCAFactors`, `HFAFactors`, `ICAFactors`,
  `RobustPCAFactors`), all sklearn-backed through `_PanelReducer`. There is no
  pure-numpy randomized SVD core, and no time-axis or tensor-mode transform.
- `panelary/transform/` is **value** transforms (`frac_diff`, `neutralize`,
  `rank`, `scaling`) — it changes what numbers mean, not what shape they are in.
  The name is taken and the meaning is different.
- `pl.Array` is used once in the codebase (`forecasting/lance.py`), so the
  fixed-width-embedding column convention has precedent but no shared helper.
- `_deps._MODULE_TO_EXTRA` has no entry for `pywt`, `tensorly` or `iisignature`.
  Adding one means adding the matching extra to `pyproject.toml`;
  `tests/test_dependency_drift.py` enforces that they stay in step.

**No benchmark numbers appear in this plan.** The house style records measured
facts; none have been measured for this module yet. §9 lists the numbers that
must be produced before Wave 1 is called done, and they are targets to *measure*,
not to assume.

---

## 2. The model

### 2.1 Intent, not "dimensionality reduction"

Four intents, and a transform declares exactly one:

| `Intent` | does | output contract | examples |
|---|---|---|---|
| `COMPRESS` | narrows one named axis | same axes, one narrower | `rsvd`, `sparse_rp`, `srht`, `paa`, `spectral`, `id` |
| `LIFT` | adds width or an axis | new basis axis of declared width | `delay`, `spectral(mode="band")`, `stft` (W2) |
| `FACTORIZE` | returns factors, not a frame | factor state + optional reconstruction | `partial_tucker`, `id`, `cur`, `tt` (W3) |
| `SKETCH` | maintains bounded state over a stream | fixed-size state, mergeable | `count_sketch`, `frequent_directions` |

`id` appears under two intents because it genuinely has two return modes
(`.transform()` → narrowed frame; `.factors()` → skeleton + interpolation
matrix). That is allowed; a transform declares a primary intent and may offer
the second as a named method.

### 2.2 Axis vocabulary — and why axis choice *is* the leak contract

Exactly four axis names. The label is not decoration; it determines the safety
contract, and `_axes.py` enforces the mapping:

| `axis=` | means | implies | `panel_safe` | `leakage_safe` |
|---|---|---|---|---|
| `"feature"` | across columns within a row | a row-wise map; no `.over()` at all | `True` | `True` **iff** fit on training rows only |
| `"time"` | along one entity's history | `.over(entity_col)`, time-ordered | `True` | `True` **iff** flavour is `trailing` |
| `"entity"` | across entities at one date | `.over(time_col)` | `False` (by design) | `True` — all of date *t* is observable at *t* |
| `"lag"` | the synthetic axis a `LIFT` creates | — | inherited | inherited |

Two consequences the report misses:

1. **`axis="entity"` is the *safe* direction and `axis="time"` is the dangerous
   one**, which is the opposite of the intuition most users bring. A
   cross-sectional PCA refit each date sees only that date; a whole-series PCA
   over time sees the future. `reduce/xs.py::CrossSectionalPCA` already
   establishes this pattern — follow it.
2. **Per-date output has an unidentified rotation across dates.** The
   `embed/` contract records this (Gabaix et al. App. C) and requires Procrustes
   alignment before per-date components may be used as a time-series feature.
   Same rule here, same implementation: `axis="entity"` transforms return
   date-local components and **refuse** to emit them as a panel feature column
   unless `align="procrustes"` is passed. Import the aligner from `embed/_xs.py`
   when that lands; until then, refuse with an error naming the reason.

### 2.3 Shape contracts live in the existing registry

Extend `panelary/registry.py::FeatureSpec` — do not create a sibling class.
Additive, all with defaults, so the 56 existing specs keep validating unchanged:

```python
# new, optional fields on FeatureSpec
intent: str | None = None            # "compress" | "lift" | "factorize" | "sketch"
axis: str | None = None              # "feature" | "time" | "entity" | "lag"
flavour: str | None = None           # "trailing" | "whole_series" | None
width_rule: str | None = None        # "exact" | "rank_dependent" | "data_dependent"
invertible: str = "none"             # "exact" | "approximate" | "from_factors" | "none"
streaming: str = "batch"             # "batch" | "partial_fit" | "mergeable"
cost_hint: str | None = None         # "O(n d k)", "O(n T log T)", "O(n T^2)" ...
```

Widen the `input_shape` / `output_shape` vocabulary from `{"series", "frame"}` to
also accept `{"tensor", "factors", "state"}`, and extend `registry.audit()` to
fail on: an `intent` that is not in the enum; `flavour="whole_series"` combined
with `leakage_safe=True`; a `cost_hint` containing `T^2` on a spec whose `tier`
is `A` or `B`. Those three checks are the machine-readable form of this
document's rules, and they are cheap.

---

## 3. Wave 1, part A — the spine (blocks everything else)

### 3.1 `_tensor.py` — the materialization boundary

Move `panelary/cluster/_tensor.py` here as `panelary/shape/_tensor.py`, keeping
`PanelTensor` and `build_tensor` byte-identical in behaviour, and leave
`from panelary.shape._tensor import PanelTensor, build_tensor` behind in
`cluster/_tensor.py` so `cluster/` is untouched and `tests/test_cluster*.py`
keeps passing. This is a move, not a rewrite: the policy it already implements
(forward-fill only, expanding z-norm, NaN never fabricated) is the policy we
want, and it was written against a documented leak in SovAI's `pandas_to_array`.

Then add, in the same file:

```python
def to_long(t: PanelTensor, *, prefix: str, dtype=pl.Float32) -> pl.DataFrame
    # (E, T, K) -> long frame keyed (entity, time) with K columns `{prefix}_0..K-1`,
    # rows aligned to `t.keys`. The exact inverse of build_tensor's indexing.

class Ragged(StrEnum):
    REFUSE   = "refuse"    # default: raise, naming the offending entities
    PAD      = "pad"       # right-pad with NaN to max length; never a value
    TRUNCATE = "truncate"  # keep each entity's most recent `T` observations
    NATIVE   = "native"    # hand the list-of-matrices through unpadded
```

`REFUSE` is the default because silently padding a 12-observation entity out to
3,000 and then running SVD on it produces a number, and the number is garbage.
`NATIVE` exists so PARAFAC2 (Wave 3) — the one method in the survey that
genuinely handles varying row counts — has somewhere to receive its input.

Hard rules for this file:

- The tensor's time axis is **always** ascending and **always** dense on the
  union grid. No transform downstream is permitted to re-sort it.
- NaN means "no observation". Every consumer must state its NaN policy in its
  docstring and implement it; none may call `np.nan_to_num` silently.
- `build_tensor` and `to_long` must round-trip: for a complete panel,
  `to_long(build_tensor(df)) == df` on the value columns, up to dtype.

### 3.2 `_window.py` — the causal spine for every time-axis transform

The single piece of machinery that makes `flavour="trailing"` cheap, and the
reason a `paa` here is not the `paa` in the report.

```python
def trailing_windows(
    values: NDArray,          # (n_rows,) or (n_rows, k), one entity's history, time-ordered
    window: int,
    *, min_periods: int | None = None,
) -> NDArray                  # (n_rows, window[, k]) view; rows < min_periods are NaN
```

Implemented with `numpy.lib.stride_tricks.sliding_window_view` — a **view**, so
framing a 3,000-point series into 3,000 windows of 64 allocates nothing. The
per-entity loop builds views; the numeric kernel then runs **once** over the
stacked `(n_rows_total, window)` array. That is the batching rule from §1.2.1,
and it is why this file must exist before any of §4.

Invariants, tested in `tests/test_shape_window.py`:

- **Prefix invariance.** `f(x[:T])[t] == f(x[:T+k])[t]` for every `t <= T`, every
  `k`, every transform built on this. This is `AGENTS.md` invariant 1 and it is
  the one test that would catch a regression to whole-series behaviour.
- Row `t` reads `values[t-window+1 : t+1]` and nothing else. Never centred —
  `embed/` §1.1 records that centred windows inflate results more than
  full-sample scalers do, so `center=` is not a parameter, it is absent.
- Rows with fewer than `min_periods` observations are **NaN**, never
  back-filled, never computed on a short window as if it were full.
- Windows never cross an entity boundary. The loop is per entity; there is no
  code path in which a single window spans two.

### 3.3 `_array.py` — fixed-width output

```python
def as_embedding(df, cols, *, name: str, dtype=pl.Float32) -> pl.DataFrame  # -> Array(dtype, D)
def explode_embedding(df, name: str, *, prefix: str | None = None) -> pl.DataFrame
```

Adopted from report §7.2. Default `Float32` for `LIFT` output (a 10,000-wide
embedding at Float64 is 80 KB per row and nothing in it is precise to 1e-16),
**Float64 for every `FACTORIZE` and `COMPRESS` result** — `AGENTS.md` invariant 3
is float64 in accumulation, and reconstruction error is the thing users check.
The dtype rule is: accumulate in float64 always; store in float32 only at the
final `LIFT` boundary, and say so in the docstring.

### 3.4 `_axes.py` — spec, plan, budget

```python
@dataclass(frozen=True, slots=True)
class ShapeSpec:                       # the per-instance, resolved form of the FeatureSpec fields
    intent: Intent; axis: Axis; flavour: Flavour | None
    in_width: int | None; out_width: int | None; width_rule: str
    invertible: str; streaming: str; cost_hint: str

class ShapeTransform(PanelTransformer):
    spec: ClassVar[ShapeSpec]
    def plan(self, X) -> Plan: ...     # output shape + estimated peak bytes, WITHOUT executing
    def explain(self) -> pl.DataFrame  # §5
```

`plan()` is the usability feature the report asks for in its §10 and never
specifies. It returns predicted output `(rows, width)`, predicted peak scratch
bytes, and the `cost_hint` string, computed from input shape and parameters
alone. `Pipeline` calls `plan()` on every step before executing any of them, so
a `lift → compress` chain reports "this will materialize 41 GB" **before** it
allocates the first byte. A `max_bytes=` budget (default: 25% of
`psutil`-free-memory if available, else a fixed 8 GB) raises `ShapeBudgetError`
naming the offending step and the parameter to lower.

This is also how the O(T²) methods in §6 stay out of the default path without
being banned outright: they are reachable, and they refuse loudly.

---

## 4. Wave 1, part B — the primitives

All pure numpy. All seeded via an explicit `seed: int` (`AGENTS.md` invariant 2);
`np.random.default_rng(seed)` — never a global, never unseeded. All accumulate in
float64. Every one registers a `FeatureSpec`.

### 4.1 `_rsvd.py` — randomized SVD / PCA (report rank #1)

```python
def randomized_svd(A, k, *, n_oversamples=10, n_iter="auto", seed=0)  -> (U, s, Vt)
class RandomizedPCA(ShapeTransform)   # axis="feature", COMPRESS, invertible="approximate"
```

Halko–Martinsson–Tropp: draw `Omega (d, k+p)`, `Y = A @ Omega`, `q` power
iterations with re-orthonormalization between each (LU or QR — without it, `q>1`
silently loses the small singular values to rounding), `Q = qr(Y)`,
`B = Q.T @ A`, small `svd(B)`. `n_iter="auto"` → 7 when `k < 0.1 * min(A.shape)`,
else 4, matching the established heuristic.

Sign-fix components deterministically using the existing
`panelary.reduce._base._sign_of_max_abs`, so signs are stable across refits —
`reduce/` already does this and the two must not disagree.

`inverse_transform` is mandatory here, and `reconstruction_error(X)` returns the
relative Frobenius error so "approximate" is a number rather than an adjective.

**Integration:** give `reduce.PanelPCA` / `PanelSVD` a `backend="numpy"` option
routed here, and make it the default when `k <= 0.1 * n_features`, where the
randomized path wins and sklearn is a hard dependency for no gain. Do **not**
change their public API or default output; this is a backend swap, tested by
asserting the two agree to `1e-6` on a fixed panel.

### 4.2 `_project.py` — sparse JL and SRHT (report ranks #2, #3)

```python
def sparse_rp_matrix(d, k, *, density="auto", seed)   # Li et al.; density = 1/sqrt(d)
def srht(X, k, *, seed)                                # subsampled randomized Hadamard
def fwht(x)                                            # in-place radix-2 FWHT, no scipy
class SparseRandomProjection(ShapeTransform)           # COMPRESS, stateless given seed
class SRHT(ShapeTransform)
```

Both are **stateless given the seed** — `fit()` stores the seed and the input
width and nothing else, so the `embed/` contract's `fit_is_empty = True` claim
holds and is testable the same way (fit on two disjoint datasets → byte-identical
state). That property is why these outrank PCA for the leak-conscious default,
and it should be said in the docstring.

`fwht` is a bit-reversal-free iterative butterfly; pad `d` to the next power of
two. SRHT is `sample(FWHT(D @ x))` with `D` a random sign diagonal.

Carry the report's honesty note through into the docstring: **the JL bound does
not license `k=64`.** At n = 1e6 and ε = 0.1 the bound wants ~1e4 dimensions.
Small outputs rest on empirical performance, not theory. `embed/` §3.4 says the
same thing; the two docstrings should not contradict each other.

### 4.3 `_sketch.py` — CountSketch and Frequent Directions (ranks #6, #7)

```python
class CountSketch(ShapeTransform)         # SKETCH, streaming="mergeable"
class FrequentDirections(ShapeTransform)  # SKETCH, streaming="mergeable", deterministic
```

Frequent Directions is the interesting one and the report undersells it: it is
**deterministic** (no seed, so nothing to leak through), mergeable (sketch each
entity independently, combine), and has a proven `||A^T A - B^T B|| <= ||A||_F^2 / l`
bound — an error bar rather than a hope. `partial_fit(rows)` and
`merge(other)` are the API; `merge` must be associative and tested as such.

`CountSketch` is `O(nnz)`: one hash and one sign per column. It is also the
first half of TensorSketch, which `embed/_sketch.py` builds — see §7.

### 4.4 `_paa.py` — piecewise aggregate approximation (rank #9)

The exemplar for the two-flavour rule, and the file to read first when writing
any other time-axis transform.

```python
class PAA(ShapeTransform)   # axis="time", COMPRESS, invertible="approximate"
# flavour="trailing" (default): window=W, segments=S -> S columns per ROW
# flavour="whole_series":       segments=S           -> S columns per ENTITY
```

Trailing PAA is `trailing_windows(x, W).reshape(n, S, W//S).mean(axis=2)` — one
reshape and one mean over the whole panel at once. `W % S == 0` is required, and
the error message says to change `W` or `S` rather than silently ragged-splitting.
`inverse_transform` is the step function (each segment mean repeated `W//S`
times), and `reconstruction_error` reports what that cost.

Add `pool=` ∈ `{mean, max, min, std, last}`, because on a returns panel `std` over
a segment is realized volatility and `last` is a subsample — both more useful than
the mean, and all four are the same reshape.

### 4.5 `_id.py` — pivoted QR, interpolative decomposition, CUR (ranks #5, #21)

```python
def pivoted_qr(A, k)                  # greedy column-pivoted modified Gram-Schmidt
def interpolative(A, k)               # -> (cols: list[int], Z)  with A ~= A[:, cols] @ Z
class ColumnSubset(ShapeTransform)    # COMPRESS via ID -- output columns are REAL features
class CUR(ShapeTransform)             # FACTORIZE
```

`numpy.linalg.qr` has no pivoting and `scipy.linalg.qr(pivoting=True)` is a
non-default dependency, so hand-roll it: modified Gram-Schmidt, at each step pick
the column of largest residual norm, downdate the norms rather than recomputing
them. For `k << d` this is `O(n d k)` and entirely BLAS-bound.

**This is the explainability primitive** (§5). `ColumnSubset(k=8)` on a
200-feature panel returns eight of your actual columns, under their actual
names, plus `Z` saying how the other 192 are reconstructed from them. No user
has ever asked what `pca_component_3` means and been satisfied by the answer.

### 4.6 `_spectral.py` and `_delay.py` — the temporal basis (ranks #8, #10)

```python
class Spectral(ShapeTransform)   # axis="time"; mode="truncate"|"band"|"power"
class Delay(ShapeTransform)      # axis="time" -> axis="lag"; LIFT
```

`Spectral` is `np.fft.rfft` over the trailing window, then one of: keep the
lowest `k` bins (COMPRESS, `invertible="approximate"` via `irfft`); sum energy
into `k` log-spaced bands (LIFT, real-valued, scale-invariant, and the one most
likely to be useful on a returns panel); total power. Magnitude-only by default —
phase on a returns window is close to noise, and complex columns do not survive
a Polars round-trip cleanly. `phase=True` opts in and emits `2k` columns.

`Delay` is `sliding_window_view` again with a `stride`/`dilation` parameter, so
`Delay(lags=16, dilation=4)` reaches back 64 observations for 16 columns. It is
causal **by construction** — lag `j` of row `t` is `x[t-j]` and there is no
parameter that could make it otherwise — which makes it the cheapest legitimate
LIFT in the module and the natural input to `_rsvd` (that composition is SSA,
which is then nearly free in Wave 2).

### 4.7 `_tucker.py` — partial Tucker (rank #4)

```python
def unfold(T, mode); def mode_dot(T, M, mode)          # np.moveaxis + np.tensordot
def hosvd(T, ranks, *, modes)                          # truncated HOSVD (the initializer)
def partial_tucker(T, ranks: dict[str, int], *, modes, n_iter=10, seed=0)
class PartialTucker(ShapeTransform)                    # FACTORIZE
```

The call the report is right about, and the one that justifies having a tensor
representation at all:

```python
PartialTucker(rank={"time": 32, "feature": 8})   # X[E,T,F] -> core[E,32,8] + factors
```

Entity is **never** a compressed mode by default — compressing across entities
mixes AAPL into MSFT, and `panel_safe` would have to be `False`. Passing
`rank={"entity": k}` is permitted but flips `panel_safe=False` on the instance
and is documented as a research tool, not a feature path.

HOOI is: HOSVD init, then alternate — for each mode in `modes`, project out the
other modes and take the leading `rank[mode]` left singular vectors of the
unfolding, using `_rsvd.randomized_svd` for the thin SVD. Converges in a handful
of iterations; `n_iter=10` with a relative-change tolerance.

**Leak contract.** Compressing the `time` mode of a whole panel is a whole-series
operation: `leakage_safe = False`, full stop. There is no trailing flavour of
Tucker in Wave 1 — a per-window Tucker is `n_rows` decompositions, which is not
affordable and should not be pretended into existence. The honest positioning is
that `PartialTucker` describes a **fixed historical sample** (regime analysis,
factor structure over a training window, compressing a panel for storage), and
the docstring says exactly that, next to a pointer to `Delay` + `RandomizedPCA`
for the row-feature case.

---

## 5. Explainability — the part the report has none of

Every fitted `ShapeTransform` implements one method with one return type:

```python
def explain(self) -> pl.DataFrame
    # columns: component, source_feature, loading, abs_loading, rank
    # plus, as frame-level metadata via .attrs-equivalent columns:
    #   explained_variance_ratio (where defined), reconstruction_error
```

A tidy frame, not a numpy array and not a plot, so it composes with everything
else in the library — `.filter(pl.col("component") == "pc_1").sort("abs_loading", descending=True).head(10)`
is the whole story of what a component is, in one line the user already knows how
to write.

Three rules that make the answers honest:

1. **Loadings are reported in the user's units.** If the transform standardized
   internally, `explain()` un-scales the loadings before reporting them, so a
   loading is "per unit of the original column" and two features measured on
   different scales are comparable. Reporting post-standardization loadings as if
   they were raw is the most common way a PCA explanation misleads.
2. **Sign is fixed deterministically** (`_sign_of_max_abs`), so the same
   component does not flip between folds and make a stability plot look like
   noise.
3. **For `ColumnSubset` / `CUR`, `explain()` is exact, not approximate** —
   `source_feature` is a real column name and `loading` is 1.0. That asymmetry is
   the argument for preferring them, and it should be visible in the output
   rather than buried in prose.

Add `stability(cv)` on top: refit across `PurgedKFold` folds, align components
across folds by Procrustes, and report per-feature loading dispersion. A
component whose loadings reshuffle every fold is not a factor, it is an artifact,
and the library should be able to say so. `panelary.validation` already has
`_selection_stats` / `_bootstrap` to build this on.

---

## 6. What we deliberately do not build

Recorded as decisions, so nobody re-litigates them in six months.

| Not shipping | Why |
|---|---|
| Rust, compiled extensions, Polars expression plugins | `AGENTS.md` + `tests/test_wheel_guardrails.py`. §1.2.1. |
| MiniROCKET, MultiROCKET, HYDRA, QUANT, RFF, TensorSketch | Owned by `plans/todo/embed-build-contract.md`. §7. |
| Anything copied from a GPL-3.0 upstream (MultiROCKET, HYDRA, QUANT reference repos) | `AGENTS.md`: contributions are clean-room. Implement from the paper, record `source` + `license` on the `FeatureSpec`. The report's §8 reaches the same conclusion. |
| GAF / recurrence plots / MTF | O(T²) memory: a 3,000-point series becomes a 9M-cell image *per entity per feature*. The report itself ranks them last (#40) and says keep them out of the default path. `plan()` (§3.4) means they can be added later behind a budget refusal rather than being unavailable — but nobody needs them in Wave 1. |
| Mantis / MantisV2 / any frozen neural encoder | Requires torch. Contradicts the CPU-only, numpy+polars footprint. The report's own benchmark note (§55) is that the classical transforms remain the faster CPU default. |
| Robust PCA / PCP, diffusion maps | Iterative, `C+` cost, and `reduce.RobustPCAFactors` already covers the heavy-tails case for the factor path. |
| A ninth top-level verb (`pn.lift`) | The golden path is eight verbs for the common case. Compressors reach the surface through the existing `pn.reduce(method=...)`; lifts that produce row features go through `pn.features`; lifts that produce a new tensor axis are advanced API and stay a deep import. Adding a verb is a public-API decision, not a refactor. §7.3. |
| `TransformSpec` as a new class | Extend `FeatureSpec`. §1.2.3. |

---

## 7. Integration points — the boundaries that prevent drift

### 7.1 With `embed/` (plan issued, not yet built)

`shape/` is the **primitive layer**; `embed/` is the **feature layer** that
composes it. Concretely, when `embed/` is implemented:

- `embed/_compress.py` must **not** implement sparse random projection. It calls
  `shape._project.SparseRandomProjection` and `shape._rsvd.RandomizedPCA`. Its
  `srp | pca | svd | none` switch becomes a thin dispatch.
- `embed/_sketch.py`'s TensorSketch is `shape._sketch.CountSketch` composed with
  `np.fft.rfft`; it owns the polynomial-kernel composition, not the sketch.
- `embed/_rff.py` keeps RFF/ORF/RVFL entirely — bandwidth heuristics and the
  Nagel (2025) guardrails are feature-layer concerns and do not belong here.
- Both modules use `shape._tensor.build_tensor` and `shape._window.trailing_windows`.
  Neither writes its own windower.
- `fit_is_empty` / `is_cross_sectional`, introduced by the `embed/` contract as
  enforced class attributes, are declared on `ShapeTransform` **here** so there is
  one definition. `embed/` imports them.

Whichever module is built second must not duplicate the first. If `embed/` lands
first, this plan's §4.2 and §4.3 become "move it to `shape/`, leave an import".

### 7.2 With `reduce/`

`reduce/` stays the user-facing home of row-wise feature reduction and factor
extraction; `shape/` supplies backends and adds the axes `reduce/` cannot express
(time, tensor modes). Do not move or rename anything in `reduce/`. Two additions
only: the `backend="numpy"` option on `PanelPCA` / `PanelSVD` (§4.1) and
`PanelRandomProjection` gaining a `method="sparse"|"srht"` routed to `_project.py`.

### 7.3 With the golden path

Extend the `method=` registry on `pn.reduce` (`_verbs.py:1126`) with the
compress-intent transforms — `"rsvd"`, `"sparse_rp"`, `"srht"`, `"paa"`,
`"spectral"`, `"id"`, `"cur"` — keeping the existing signature and defaults
exactly as they are. The docstring's `Leak-safety` section gains one paragraph:
time-axis methods take `window=`, and passing `flavour="whole_series"` moves the
call into the fit-on-what-you-pass regime.

`pn.features` gains `method="delay"` and `method="spectral"` for the LIFT
direction. `PartialTucker` and the sketches stay deep imports — they do not have
a natural verb and forcing one would make the verb worse.

### 7.4 With the registry and `llms.txt`

Every public transform registers a `FeatureSpec` with the §2.3 fields populated,
`source` naming the paper (Halko et al. 2011; Li et al. 2006; Ailon–Chazelle 2009;
Ghashami et al. 2016; Keogh et al. 2001; De Lathauwer et al. 2000) and `license`
set to `Apache-2.0` for clean-room work. `registry.audit()` must stay green.
Regenerate `llms.txt` after Wave 1; the shape algebra is exactly the kind of
thing an agent consumer needs the catalogue for.

---

## 8. File ownership — touch ONLY your file(s)

| File | Agent | Wave | Depends on |
|---|---|---|---|
| `shape/_axes.py` — `Intent`, `Axis`, `Flavour`, `ShapeSpec`, `ShapeTransform`, `Plan`, `ShapeBudgetError` | A | 1 | `core/protocol.py`, `registry.py` |
| `registry.py` — additive `FeatureSpec` fields + 3 new `audit()` checks | A | 1 | — |
| `shape/_tensor.py` — moved `build_tensor`, new `to_long`, `Ragged` | B | 1 | — |
| `shape/_window.py` — `trailing_windows`, the causal spine | B | 1 | — |
| `shape/_array.py` — `as_embedding`, `explode_embedding` | B | 1 | — |
| `shape/_rsvd.py` — `randomized_svd`, `RandomizedPCA` | C | 1 | A |
| `shape/_project.py` — sparse JL, `fwht`, SRHT | D | 1 | A |
| `shape/_sketch.py` — `CountSketch`, `FrequentDirections` | D | 1 | A |
| `shape/_id.py` — `pivoted_qr`, `interpolative`, `ColumnSubset`, `CUR` | E | 1 | A, C |
| `shape/_paa.py` — `PAA` (both flavours) | F | 1 | A, B |
| `shape/_spectral.py` — `Spectral` | F | 1 | A, B |
| `shape/_delay.py` — `Delay` | G | 1 | A, B |
| `shape/_tucker.py` — `unfold`, `mode_dot`, `hosvd`, `partial_tucker` | H | 1 | A, B, C |
| `shape/_explain.py` — `explain()` helpers, `stability()` | I | 1 | A, C, E |
| `shape/__init__.py` — public API, docstring, `__all__` | A | 1 | all |
| `reduce/pca.py` — `backend=` only; no API change | C | 1 | C |
| `_verbs.py` — `pn.reduce` / `pn.features` method registry only | A | 1 | all |
| `tests/test_shape_*.py` | J | 1 | all |
| `shape/_dwt.py`, `_ipca.py`, `_ssa.py`, `_dmd.py`, `_stft.py` | — | 2 | Wave 1 |
| `shape/_cp.py`, `_parafac2.py`, `_tt.py` (optional `tensorly` backend) | — | 3 | Wave 2 |

Agent B's files block C–I. Agent A's `_axes.py` blocks everything. Build those
two first, in that order, and hand the signatures out before the rest start.

---

## 9. Acceptance — Wave 1 is done when

**Correctness (blocking).**

1. `tests/test_shape_prefix_invariance.py` — for every transform declaring
   `flavour="trailing"`, `f(x[:T])[t] == f(x[:T+k])[t]` exactly (not
   `allclose` — bit-identical, since the computation reads identical inputs) for
   a grid of `T`, `k`, `t`, on a hypothesis-generated ragged panel.
2. `tests/test_shape_leak_safety.py` — every `whole_series` transform raises
   through `_check_leakage` when applied across a train/test boundary; every
   `axis="entity"` transform refuses to emit unaligned per-date components as a
   panel column. Modelled on `tests/test_reduce_factor_leakage.py`.
3. `tests/test_shape_stateless.py` — `SparseRandomProjection`, `SRHT`,
   `CountSketch`, `PAA`, `Spectral`, `Delay` fitted on two **disjoint** datasets
   with the same seed produce byte-identical state. Mirrors the `embed/` contract's
   `fit_is_empty` test.
4. `tests/test_shape_roundtrip.py` — `to_long(build_tensor(df))` reproduces `df`;
   `inverse_transform` reconstruction error is below a stated bound for a
   synthetic exactly-rank-`k` matrix (`< 1e-10` for `rsvd` at `k = true rank`);
   `FrequentDirections.merge` is associative.
5. `tests/test_shape_agreement.py` — `randomized_svd` agrees with
   `np.linalg.svd` to `1e-6` on the leading `k`; `hosvd` / `partial_tucker` agree
   with `tensorly` where installed (skipped otherwise — test-only dependency).
6. `registry.audit()` green; `make check` green (`ruff`, `mypy`, full pytest).
7. `tests/test_import_hygiene.py` still passes: `import panelary` must not pull
   `shape/` transitively, and no module here may import scipy/sklearn/tensorly at
   top level.
8. `tests/test_wheel_guardrails.py` still passes — `py3-none-any`, no compiled
   artifacts. Nothing in this plan changes that and the test proves it.

**Performance (measure, then record here; do not assume).**

Numbers to produce on the report's own workload grid (§11 of the PDF), which is
a good grid: small `E=1k, T=64, F=8`; medium `E=10k, T=256, F=16`; long
`E=1k, T=4096, F=8`; wide `E=10k, T=64, F=256`; ragged; streaming. For each:
`fit` ms, `transform` ms, peak RSS, and — for the approximate methods — the
reconstruction error at matched output width. One-core and all-core.

Two specific targets worth stating up front, both falsifiable:

- `RandomizedPCA(k=16)` on the wide panel should beat
  `sklearn.PCA(svd_solver="randomized")` on wall clock, or the `backend="numpy"`
  default in §4.1 is not justified and should not be made the default.
- Trailing `PAA` / `Spectral` over a 1M-row panel should be dominated by the
  numpy kernel, not the per-entity Python loop. If the loop is more than ~20% of
  the time, the batching in §3.2 is wrong and needs fixing before Wave 2.

---

## 10. Open questions to resolve with a measurement, not an opinion

1. **Is the `sliding_window_view` view actually free at panel scale?** It is a
   view, but the first numpy kernel that touches it materializes a
   `(n_rows, W)` copy. For `n_rows = 1e6, W = 256` that is 2 GB in float64.
   Measure; if it bites, chunk the row axis and stream, and put the chunk size in
   `plan()`'s byte estimate.
2. **Where exactly does the randomized path beat the exact path?** The `k <= 0.1 *
   min(n, d)` heuristic in §4.1 is folklore. Produce the crossover curve on this
   repo's panels and set the default from it.
3. **Does `ColumnSubset` (ID) beat `RandomizedPCA` on downstream task
   performance at matched width?** If interpretable compression costs nothing
   measurable, it should be the *default* `pn.reduce` method for wide panels, and
   that is a much stronger product claim than anything in the report. If it costs
   materially, say so in the docstring and keep PCA the default.
4. **Is trailing Tucker affordable at all** with an incremental/warm-started HOOI
   (reuse the previous window's factors as the init)? If yes, it is a genuinely
   novel row-feature path and belongs in Wave 2. If no, §4.7's positioning stands
   and should stop being revisited.
5. **`Float32` output width tipping point** — at what `D` does storing a `LIFT` as
   `Array(Float32, D)` beat `D` loose `Float64` columns on join, groupby and
   `to_numpy` time? The report asserts `Array` is better; measure it on a real
   panel before making it the convention.

---

## 11. Milestones

- **M1 — the spine.** `_axes.py`, `_tensor.py`, `_window.py`, `_array.py`,
  registry extension, `__init__.py` skeleton, and tests 1/3/4 (prefix invariance,
  statelessness, round-trip) passing against `PAA` alone as the reference
  transform. Nothing else starts before this merges.
- **M2 — linear core.** `_rsvd.py`, `_project.py`, `_sketch.py`, `_id.py`,
  `_explain.py`, the `reduce/` backend swap. Report the §9 performance numbers.
- **M3 — temporal + tensor.** `_paa.py` (full), `_spectral.py`, `_delay.py`,
  `_tucker.py`, verb wiring, docs page, `llms.txt` regeneration.
- **M4 — Wave 2.** DWT, incremental PCA, SSA (which is `Delay` + `_rsvd` and is
  nearly free once M3 lands), DMD, STFT. Optional-extra plumbing for `pywt`.

---

## 12. Honest gaps

- **The `whole_series` flavour is a real capability with a real footgun**, and no
  amount of return-type design fully closes it. A determined user can
  `join` a per-entity frame back onto rows and leak. The mitigation is the
  entity-keyed return shape plus a loud docstring; it is not a proof.
- **`plan()`'s byte estimates will be wrong at the margins.** They are computed
  from shapes and parameters, and numpy's actual scratch depends on LAPACK
  workspace queries we do not model. Treat the budget as a guardrail against
  order-of-magnitude mistakes, not as an accounting system, and say so.
- **Nothing here helps with genuinely ragged tensors until PARAFAC2 (Wave 3).**
  `Ragged.REFUSE` is the honest default, but it means a user with a realistically
  unbalanced panel gets an error rather than a result, and `PAD` is a
  statistically questionable escape hatch. This is the single largest known gap.
- **No supervised transform is in scope.** Everything here is unsupervised, which
  keeps the leak surface small but means the module cannot compete with a
  supervised feature extractor on accuracy. That is the right trade for a
  primitive layer, but it should not be oversold.
- **The report's accuracy claims for the convolutional family are not
  re-verified here** — `embed/` §1.1 already tempers them substantially
  (regression reverses the ordering; ROCKET dominance is a univariate-UCR
  artifact). Do not repeat the PDF's rankings in this module's docs.
