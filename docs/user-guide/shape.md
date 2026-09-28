# Shape algebra: compress, lift, factorise

`panelary.shape` holds transforms that change the **shape** of a panel: the
width of one named axis, the order of the tensor, or the numerical rank of a
factorisation. What `entity`, `time` and `feature` mean stays the same. It is
pure numpy + polars. No scipy, scikit-learn or tensorly is on any default
import path.

The module models **intent**, not "dimensionality reduction". Every transform
declares one:

| `Intent` | What it does | Examples |
| --- | --- | --- |
| `COMPRESS` | narrows one named axis | `RandomizedPCA`, `SparseRandomProjection`, `SRHT`, `ColumnSubset`, `PAA`, `Spectral(mode="truncate")` |
| `LIFT` | adds width or a new axis | `Delay`, `Spectral(mode="band" \| "power")` |
| `FACTORIZE` | returns factors, not just a frame | `PartialTucker`, `CUR` |
| `SKETCH` | keeps bounded, mergeable state | `CountSketch`, `FrequentDirections` |

## The axis is the leak contract

Every transform also names the **axis** it acts along. The axis label is not
decoration. It determines the safety contract, and `ShapeTransform` refuses a
class whose declaration contradicts it:

| `axis=` | Means | `panel_safe` | `leakage_safe` |
| --- | --- | --- | --- |
| `"feature"` | across columns within a row | yes | only if fit on training rows |
| `"time"` | along one entity's history | yes | only if the flavour is `trailing` |
| `"entity"` | across entities at one date | **no** (by design) | yes, because all of date *t* is observable at *t* |
| `"lag"` | the axis a `LIFT` creates | inherited | inherited |

This runs against the usual intuition: `axis="entity"` is the *safe*
direction and `axis="time"` is the dangerous one. A reduction refit on each
date sees only that date. A reduction over the whole time axis sees the future.

### Two flavours along time, and the causal one is the default

The obvious `PAA(segments=32)` pools each entity's **entire** series into 32
segments. Segment 0 of that output is then a function of the whole series,
future included, which is a textbook look-ahead when it is used as a row
feature. So every time-axis transform comes in two flavours:

| Flavour | Contract | Use it for |
| --- | --- | --- |
| `trailing` (default, `window=W`) | one output row per input row, from that row's own last `W` observations within its entity | row features: the thing you put in a model |
| `whole_series` (explicit opt-in) | one output row per **entity**, from all of that entity's data; `leakage_safe = False` | clustering series, describing a fixed historical sample |

The `whole_series` result is keyed by entity alone. Its time column holds the
constant `"whole_series"`, so joining it back onto rows fails loudly instead of
broadcasting a summary that saw the future. `Pipeline` refuses a whole-series
step across a train/test boundary.

```python
import numpy as np
import polars as pl
import panelary as pn
from panelary.shape import PAA, Spectral, Delay

rng = np.random.default_rng(0)
df = pl.DataFrame({
    "ticker": np.repeat(["A", "B", "C"], 120),
    "date": np.tile(np.arange(120), 3),
    "ret": rng.standard_normal(360) * 0.01,
})

# Trailing (default): 4 segment volatilities of each row's last 32 returns.
vol = PAA(window=32, segments=4, pool="std").fit_transform(df, entity="ticker", time="date")

# Energy share in 4 log-spaced frequency bands of each row's last 64 returns.
bands = Spectral(window=64, k=4, mode="band").fit_transform(df, entity="ticker", time="date")

# Delay embedding: ret[t], ret[t-2], ret[t-4], ret[t-6].
lags = Delay(lags=4, dilation=2).fit_transform(df, entity="ticker", time="date")

# Whole-series (explicit): one row per ticker. Not a row feature.
shape_of_series = PAA(segments=8, flavour="whole_series").fit_transform(
    df, entity="ticker", time="date"
)
```

Every trailing transform is **bit-identical under truncation**:
`f(x[:T])[t] == f(x[:T+k])[t]` exactly, not approximately.
`tests/test_shape_prefix_invariance.py` checks this on hypothesis-generated
ragged panels.

## The spine

Two pieces of machinery sit under everything else.

**The materialization boundary** (`build_tensor` / `to_long`). This is how a
long, ragged polars panel becomes a dense `(entity, time, feature)` array, and
how it gets back. `build_tensor` forward-fills within an entity only (never
backward) and marks a missing observation as NaN. It never fabricates a value.
`to_long(build_tensor(df))` reproduces `df`. When entities have different
lengths, `build_sequences` needs an explicit `Ragged` policy:

- `"refuse"` is the default. It raises and names the offending entities.
- `"pad"` pads with NaN.
- `"truncate"` keeps the most recent `length` observations.
- `"native"` returns unpadded per-entity arrays.

Silently padding a 12-observation entity out to 3,000 and running an SVD on
it produces a number, and the number is garbage.

**The causal windower** (`trailing_windows`, `PanelWindows`). Row `t` reads
`values[t - W + 1 : t + 1]` and nothing else. There is no `center=` parameter:
it is absent, not defaulted. The whole panel is laid out in one buffer with a
NaN gap between entities, so a window can never span two entities. Each kernel
runs **once** over every row of every entity, with no per-entity Python loop.

## Feature-axis compressors

| Transform | What you get | Fit |
| --- | --- | --- |
| `RandomizedPCA(n_components=k)` | principal components (Halko–Martinsson–Tropp randomized SVD, sign-fixed like `panelary.reduce`) | training rows (scaler + rotation) |
| `SparseRandomProjection(n_components=k, seed=)` | Li et al. very sparse JL projection | **nothing**: a function of width and seed |
| `SRHT(n_components=k, seed=)` | subsampled randomized Hadamard transform | **nothing** |
| `ColumnSubset(k=)` | `k` of **your real columns**, chosen by pivoted QR (interpolative decomposition) | training rows (column choice) |
| `CUR(k=, k_rows=)` | real columns, real training observations, and a linking matrix | training rows |
| `CountSketch(n_components=, seed=)` | signed feature hashing | **nothing** |
| `FrequentDirections(ell=, n_components=)` | deterministic, mergeable Gram sketch with a proven error bar | training rows (streamable) |

The seeded projections declare `fit_is_empty = True`, and the base class
enforces it: `fit` hands them a zero-row frame. Their state therefore provably
depends on nothing in the data, and fitting on two disjoint datasets gives
byte-identical state. That is why they are the leak-conscious default
compressor.

!!! warning "The Johnson–Lindenstrauss bound does not license small outputs"
    At `n = 1e6` and `eps = 0.1` the JL bound wants on the order of 1e4
    dimensions. `n_components=64` rests on empirical performance, not on the
    theorem.

`ColumnSubset` is the explainability primitive. It returns eight of your actual
columns, under their actual names, plus `Z_`, which says how every other column
is reconstructed from them:

```python
from panelary.shape import ColumnSubset

wide = pl.DataFrame({
    "ticker": np.repeat(["A", "B"], 200),
    "date": np.tile(np.arange(200), 2),
    **{f"f{j}": rng.standard_normal(400) for j in range(20)},
})
cs = ColumnSubset(k=5).fit(wide, entity="ticker", time="date")
cs.selected_                 # e.g. ['f7', 'f0', ...] -- real column names
cs.interpolation()           # kept_feature, reconstructed_feature, coefficient
cs.reconstruction_error_     # a number, not an adjective
```

## Tensor modes: `PartialTucker`

`PartialTucker(rank={"time": 32, "feature": 8})` factorises the panel tensor
`X[E, T, F]` into a per-entity core `[E, 32, 8]` plus time and feature factors.
It uses a HOSVD start followed by HOOI sweeps, all in numpy, with
`randomized_svd` for the thin SVDs.

It is `leakage_safe = False`, full stop. Compressing the time mode of a whole
panel sees every date. There is no trailing flavour, because a per-window
Tucker would be `n_rows` decompositions. Use it to describe a **fixed
historical sample**: regime analysis, factor structure over a training window,
or compressing a panel for storage. For a row feature, use `Delay` followed by
`RandomizedPCA` fit on training rows. That combination is SSA.

Entity is never compressed by default. `rank={"entity": k}` is permitted, but
it narrows the instance to `panel_safe = False` and is a research tool.

## Entity axis: `CrossSectionalRandomizedPCA`

This transform fits a separate PCA on each date's cross-section. It is leak-safe
in time by construction, and it is never panel-safe. Per-date components are
identified only up to rotation across dates, so a raw per-date score is not a
time series. The transform **refuses** to emit one unless you pass
`align="procrustes"`. With that option, each date's loadings are rotated onto
the previous date's aligned loadings (orthogonal Procrustes to `t - 1`). The
alignment reads only the past, so the output is causal and prefix-invariant.
`date_components(X)` returns the unaligned date-local loadings as a
description.

## Explainability: `explain()` and `stability()`

Every fitted transform implements `explain() -> pl.DataFrame` with the columns
`component, source_feature, loading, abs_loading, rank,
explained_variance_ratio, reconstruction_error`. It is a tidy frame, so one
line answers what a component is:

```python
from panelary.shape import RandomizedPCA

p = RandomizedPCA(n_components=3).fit(wide, entity="ticker", time="date")
p.explain().filter(pl.col("component") == "rpc_1").sort("abs_loading", descending=True).head(5)
```

Loadings are reported in the user's units: internal standardisation is undone.
Signs are fixed deterministically. For `ColumnSubset` / `CUR` the answer is
exact: `source_feature` is a real column and `loading` is `1.0`.

`stability(transform, X)` refits the transform on each `PurgedKFold` training
fold and aligns the loadings across folds by Procrustes. It then reports the
per-feature loading dispersion and a per-component congruence. A component
whose loadings reshuffle every fold is an artifact, not a factor.

## Refusing before allocating: `plan()`

`t.plan(X)` predicts the output shape and peak scratch bytes from the input
shape and the parameters alone, without executing anything. `plan_chain(steps,
X)` feeds each step's predicted output to the next. A `Delay(lags=512)` feeding
a PCA therefore reports its size before the first byte is allocated.
Transforms check their plan against `max_bytes=` (default: 25% of available
memory if `psutil` is installed, else 8 GiB). When a step is over budget,
`ShapeBudgetError` names the step and the parameter to lower. The estimate is a
guardrail against order-of-magnitude mistakes, not an accounting system: LAPACK
workspace is not modelled.

## From the golden path

```python
pn.reduce(df, method="rsvd", n_components=8)                # also "sparse_rp", "srht"
pn.reduce(df, method="id", n_components=8)                  # ColumnSubset: real columns
pn.reduce(df, method="paa", n_components=4, window=32)      # trailing, point-in-time
pn.features(df, method="delay", lags=8, dilation=2)         # trailing LIFT, one row per row
pn.features(df, method="spectral", window=64, k=4)          # band shares by default
```

`PartialTucker` and the sketches stay deep imports. They do not have a natural
verb.

`reduce.PanelPCA` / `PanelSVD` also accept `backend="numpy"`, which routes them
through `randomized_svd` with no scikit-learn, and `backend="auto"`.
`PanelRandomProjection` accepts `method="sparse" | "srht"`. The defaults are
unchanged.

## Measured performance

These numbers come from one Apple-silicon machine using numpy's Accelerate
BLAS, measured 2026-09-28. Each figure is from its own process, with peak RSS
reported per process. They are measurements, not guarantees.

Feature axis, `k = 16` on the "wide" panel (640,000 rows × 256 features) and
`k = 4` on "medium" (2.56M rows × 16), on the same standardised float64 matrix:

| Method | wide fit | medium fit | Reconstruction error (wide) |
| --- | --- | --- | --- |
| `randomized_svd` (this module) | 1.18 s | 0.92 s | 0.0381 |
| `sklearn` `PCA(svd_solver="randomized")` | 2.45 s | 2.04 s | 0.0381 |
| `sklearn` `PCA(svd_solver="auto")` (`covariance_eigh` here) | 0.74–0.82 s | 0.58–0.70 s | 0.0381 |
| `ColumnSubset` (pivoted QR) | 2.96 s | 0.53 s | 0.0542 |
| `SparseRandomProjection` transform | 0.04 s | 0.05 s | — |
| `SRHT` transform | 2.2 s | 0.46 s | — |

Time axis, `window = 32`, transform only:

| Panel | `PAA(segments=8)` | `Spectral(k=8)` | `Delay(lags=8, dilation=4)` |
| --- | --- | --- | --- |
| small (64k rows × 8) | 16 ms | 118 ms | 21 ms |
| medium (2.56M rows × 16) | 0.80 s | 5.7 s | 1.2 s |
| long (4.1M rows × 8) | 0.48 s | 4.9 s | 0.83 s |

On a 1M-row panel there is no per-entity Python loop at all. For `PAA`, 64% of
the 122 ms goes to the numeric kernel and the rest to the polars sort/collect
and the padded buffer. For `Spectral`, 97% goes to `numpy.fft.rfft`.

## What is not built

These are deliberate decisions, recorded so they are not re-litigated. See
`plans/todo/shape-build-contract.md` §6.

- Rust or a Polars plugin: the wheel is pure Python.
- MiniRocket / QUANT / RFF / TensorSketch: `embed/` owns those.
- GAF / recurrence plots: these cost O(T²) and are kept behind the budget.
- Robust PCA: `reduce.RobustPCAFactors` covers it.
- A ninth top-level verb.
