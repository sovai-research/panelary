# `panelary/embed/` — build contract

> **Status (2026-09-28): partially implemented.** Every module in section 2 exists and
> is tested, and all are exported from `panelary.embed`. Wave 1: `_intervals`,
> `_contract`, `_quant`, `_c22i`, `_rff`, `_compress`. Wave 2: `_probe`, `_xs`;
> `_rocket` and `_cross_rocket` were built by their own owner. Wave 3: `_hydra`,
> `_sketch`, `_diagnostics`. The section 3.1 catch22 batch rewrite has landed: measured
> ~12,000 windows/s at L=128. The earlier agent measured the scalar path before it at
> ~240 windows/s, not the 10.2 recorded in section 1.1. The stateless, leakage and prefix-invariance suites
> of section 8 are green, and a deliberately leaky variant of every transform fails
> them. **Outstanding:**
> - Section 8's accuracy benchmarks are unmeasured: linear-probe AUROC, k-NN recall,
>   robustness and out-of-domain transfer. So is RSS at 10^6 windows.
> - No `FeatureSpec` is registered, because the conformance suite only knows
>   `ts` / `panel` / `xs`.
> - The per-method `NOTICE` paragraphs, the top-level export, the CHANGELOG entry and
>   the mkdocs nav are with the orchestrator.
>
> **Deliberate deviations:**
> - `svd` in `_compress` is `RandomizedPCA(standardize=False)`: centred, unscaled.
> - Hydra runs per trailing window (valid convolutions, `np.bincount` tallies)
>   rather than the cumsum trick.
> - The RFF scaler and bandwidth are expanding by-date schedules over the fitted rows.
> - The `Embedder` protocol does not require `get_state()`, because the two rocket
>   classes do not provide it.

Fast, CPU-only, **leak-safe numerical embeddings** for panel data — the numerical
analogue of model2vec static text embeddings. Issued to the implementation
agents 2026-09-09.

Pure **numpy + polars**. No scipy, sklearn, statsmodels, numba, torch or Rust in
the import path. Optional accelerators (numba) go behind
`panelary._internal._deps.require` under the existing `fast` extra and must have a
pure-numpy fallback that is the **default**.

Reuse, do not reinvent:
`panelary.core.protocol.PanelTransformer` (the `panel_safe` /
`leakage_safe` contract), `panelary.core.model_selection`
(`PurgedKFold`, `CombinatorialPurgedCV`), `panelary.validation`
(`_selection_stats`, `_bootstrap`), `panelary.reduce`
(`PanelPCA`, `PanelRandomProjection`, `n_factors`), `panelary.catch22`
(the 22 feature functions), `panelary.reduce.xs.CrossSectionalPCA`.

---

## 0. TL;DR — the decision

**Ship QUANT first, not MiniRocket.** Every external ranking puts MiniRocket at
the top; for a library whose entire pitch is leak-safety, the ordering must
follow the **fitted-state surface**, not the UCR leaderboard.

| | MiniRocket | QUANT |
|---|---|---|
| fitted state | biases = quantiles of conv output over ~4K training examples | **none — `fit()` is `pass`** |
| leak surface | real; must be fit inside the fold | **structurally zero** |
| measured cost | 3.69 ms/series | **0.34 ms/series** |
| output dim | 9,744 | **1,042** |
| permissive numpy impl exists? | no (GPL ref; aeon = numba) | **no (GPL ref; aeon requires PyTorch)** |

QUANT's intervals are deterministic dyadic functions of *series length alone*.
Fitting it on wildly different data yields byte-identical state. That is
leak-safe **by construction rather than by discipline** — the only method in the
entire survey with that property, and the exact thing this library sells.

The four-layer architecture:

```
[1 causal window]  →  [2 nonlinear expansion]  →  [3 compaction]  →  [4 probe]
 trailing / per-date    QUANT · RandIntC22 ·        SRP · PCA · SVD    PreValidated
 (never centred)        MiniRocket-PPV · RFF                            Ridge
```

Layers 2 and 3 **must stay separate**. The best transforms expand (QUANT ~10n,
MiniRocket ~10k); users want 128–512 dims. Do not distort the expansion to
force compactness — compose it with a compressor.

---

## 1. Evidence base

Eight parallel research lanes (2026-09-09) plus an external report
(`panelkit_fast_numerical_embeddings_deep_research.pdf`, filed alongside this
plan under its pre-rename filename). Where they disagree, this contract follows the lanes, because the lanes
verified licences and benchmarks against primary sources and measured
implementations locally.

### 1.1 Findings that constrain the design

| Finding | Source | Consequence |
|---|---|---|
| **Nagel (2025)**: RFF + ridgeless with P>T degenerates *mechanically* into volatility-timed momentum, and **mis-learns** on mean-reverting data. Explicitly extends to the cross-sectional panel case with RFF of firm characteristics. | NBER w34104 | §5 guardrails are **non-negotiable**. Never ship ridgeless. |
| **Deflated Sharpe and PBO do not detect leakage.** A leaky oracle at Sharpe 35 passes both. | arXiv:2608.27734 | Leakage must be excluded *structurally*, at the API. DSR/PBO are search-intensity corrections only — say so in the docstring. |
| **Centred windows inflate far more than full-sample scalers.** | arXiv:2605.23959 | Ban `center=True` before worrying about scaler placement. |
| **Per-date cross-section is the leak-safe direction** — all of date *t* is observable at *t*. Every replicated finance result uses "rank within date, refit each period". | GKX, IPCA, Gabaix et al. | Cross-sectional mode is a **first-class API**, not an afterthought. |
| **Per-date embeddings have unidentified rotation across dates.** | Gabaix et al. App. C | `refuse` to emit unaligned per-date components as a time-series feature. Ship Procrustes alignment. |
| **On regression, hand-crafted beats ROCKET.** TSER 62 datasets: DrCIF 2.97, FreshPRINCE 3.19 > MultiRocket 4.68, ROCKET 5.94. Ordering *reverses* vs classification. | arXiv:2305.01429 | Panel finance is regression. Temper expectations for the conv family; keep the feature-set path first-class. |
| **ROCKET dominance is a univariate-UCR artefact.** Multiverse 2026, 133 multivariate datasets: MR-Hydra 7.30, ROCKET 7.90, QUANT 8.45 — all significantly beaten by HC2. | arXiv:2603.20352 | Panels are multivariate. Do not over-promise. |
| **Archive overfitting is real.** On 30 held-out datasets the 3–4 point gaps shrink to ~1 point; QUANT vs HC2 is 14W/16L. | recomputed from `tsml-eval/table_c4.csv` | The cheap methods are closer than the headline tables suggest. |
| **R-Clustering** (the only published *unsupervised* ROCKET result): **500 kernels beat 10,000** for clustering; PCA to 10–20 dims; **MiniRocket's bias ordering injects artificial autocorrelation** — must permute. | DMKD 2024 | Wave-2 defaults: 500 kernels, permuted biases. |
| **Whole-series catch22 is weak; catch22-over-intervals is strong.** Whole-series ranks ~6th of 9 and barely beats 1NN-DTW; the same 22 features over random intervals jump to ~3rd, level with tsfresh's 780. | arXiv:2201.12048, arXiv:2308.01071 | **Highest-ROI item on code we already own.** |
| **ROCKET-family per-instance z-norm destroys amplitude.** In finance amplitude *is* volatility. | sktime#4374 | Any conv port needs `normalise=False` **plus** retained explicit scale features. |
| **Measured locally**: `catch22_all` runs at **10.2 windows/s** (~33 h for a 1.2M-window panel). 89.6% of cost is in 3 functions. | this repo, 2026-09-09 | See §3.1. Blocking prerequisite. |

### 1.2 Where the external PDF is wrong for this product

The PDF is a good survey and §9 (`compress=`), §10 (API surface) and §12
(benchmark-as-embedding-library) are adopted below. Four of its rankings are
**not** adopted:

1. **MiniRocket as #1 default (score 95 vs QUANT 94).** Inverted here, for the
   reason in §0. The PDF ranks on accuracy-per-CPU-second; we rank on leak
   surface first.
2. **HDC-ROCKET at #5 (score 90), "the most exciting new addition."** The
   primary source (arXiv:2202.08055) reports that an *oracle*-selected scale
   parameter improves 81/128 datasets by mean 3.1% — but **honest CV selection
   improves only 17/128 and can be 0.5% worse**, and the authors state "there is
   no single value of the scale parameter that works for all datasets." The PDF
   cites a paywalled DMKD 2025 follow-up we could not verify. **Demoted to an
   experimental flag (§6), not a core module.**
3. **RFF at #3 (score 92) with no finance guardrail.** See Nagel above. Also
   Grinsztajn et al. (NeurIPS 2022): tabular data is **not** rotation-invariant
   — randomly rotating features *reverses* model rankings — and dense-Gaussian
   RFF *is* rotation-invariant. RVFL direct links are mandatory, not optional.
4. **SORF-DCT as core.** Structured transforms only pay when `d` is in the
   thousands; post-summary-stat `d` here is 20–500, where the GEMM is already
   negligible. And the TPAMI survey's own conclusion: *"better kernel
   approximation does not directly translate to lower generalization errors."*
   Deferred to the watchlist.

The PDF's largest omission is that **it treats Panelary as a time-series
library**. Its entire API is trailing windows per entity; the cross-sectional
axis never appears. §4 fixes that.

### 1.3 What the PDF adds that the lanes missed — adopt

- **PreValidated Ridge** (Dempster, Webb & Schmidt, *Machine Learning* 2026) —
  10×–1000× faster than tuned logistic regression, median ~140× in the
  MiniRocket setting. Directly solves "we cannot ship sklearn." **Wave 2.**
- **TSPulse-R1** — 1.08M params, 4.37 MB float32, Apache-2.0, GPU-free, ICLR
  2026, disentangled temporal/spectral/semantic views. The most literal tiny
  pretrained numeric encoder found by either effort. **Watchlist**, tempered by
  the lanes' general finding that frozen FM embeddings lose to conv transforms.
- **wildboar** (BSD-3) — `CastorTransform`, `QuantTransform`, `HydraTransform`.
  Useful comparator. ⚠️ Same provenance caveat as aeon (see §7).
- The explicit `compress=` layer, and benchmarking as an *embedding* library.

---

## 2. Module layout

```
panelary/
    embed/
        __init__.py           # public API + the Embedder protocol
        _contract.py          # EmbeddingState: serialise/deserialise, seeds, schema
        _quant.py             # QUANT — clean-room, stateless           [Wave 1]
        _intervals.py         # random dilated interval sampler (shared) [Wave 1]
        _c22i.py              # RandIntC22: catch24 x intervals          [Wave 1]
        _compress.py          # srp | pca | svd | none                   [Wave 1]
        _rff.py               # RFF/ORF + arc-cosine + RVFL direct links [Wave 1]
        _rocket.py            # causal rolling MiniRocket-PPV            [Wave 2]
        _probe.py             # PreValidated Ridge                       [Wave 2]
        _xs.py                # per-date cross-sectional mode + Procrustes [Wave 2]
        _hydra.py             # competing kernels                        [Wave 3]
        _sketch.py            # TensorSketch (CountSketch + rFFT)        [Wave 3]
        _diagnostics.py       # reversal-synthetic, null panel, baselines [Wave 3]
```

Every public class subclasses `PanelTransformer` and sets `panel_safe` /
`leakage_safe`. Two **new** class attributes are introduced and enforced at
runtime, not by docstring convention:

```python
fit_is_empty: bool      # True => transform() is a pure function of the window
is_cross_sectional: bool  # True => fits per-date, never across dates
```

`fit_is_empty = True` is a **testable claim**: `tests/test_embed_stateless.py`
must assert that fitting on two disjoint datasets yields byte-identical state.

---

## 3. Wave 1 — native primitives (zero new dependencies)

### 3.1 Blocking prerequisite: batch the existing catch22

Measured in this repo on 2026-09-09:

```
catch22_all: 10.2 windows/s (19.6 s for 200 windows of L=128)
             → ~33 hours for a 1.2M-window panel
total 99.4 ms/window:
  46.9 ms  47.2%  SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
  21.3 ms  21.4%  SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
  20.8 ms  21.0%  IN_AutoMutualInfoStats_40_gaussian_fmmi
  10.4 ms  10.4%  the other 19 features combined
```

**89.6% of the cost sits in three functions.** The cause is structural: every
helper takes a single 1-D array via `_as_1d`, so nothing batches. Note this
contradicts the profile reported for the reference C implementation, where
`PD_PeriodicityWang` is a third of runtime — here it is 1.3%. Different
implementation, different bottleneck; do not port the upstream optimisation.

Task: rewrite the 22 features to accept a `(n_windows, L)` batch axis. The two
`SC_FluctAnal` variants share a fluctuation-analysis inner loop
(`catch22.py:761`), so one batched rewrite fixes 68% of the cost. Target
**≥1,000 windows/s** at L=128. Existing scalar entry points stay as thin
wrappers; `tests/test_catch22.py` must pass unchanged.

### 3.2 `QuantEmbedder` — the flagship

Deterministic dyadic intervals over four representations: raw `X`; smoothed
first difference (5-tap moving average of `diff(X)`, replicate padding); second
difference; `|rfft(X)|`. Depth `d = min(6, floor(log2(n)) + 1)`; at each depth
`>1` add a half-shifted copy. Per interval take `m/4` quantiles
(`m` = interval length), then **subtract the interval mean from every second
quantile**. Output ≈ `10n` features per representation-set.

- `fit_is_empty = True`, `panel_safe = True`, `leakage_safe = True`.
- Pure numpy: `np.diff`, `np.fft.rfft`, `np.quantile`/`np.sort` over slices.
- Cap `interval_depth` for short windows; expose it.
- The reference paper validates these features with **ExtraTrees**, not a linear
  head. Ship the transform; document that the linear-probe number is unverified.

**⚠️ Clean-room requirement.** `angus924/quant` is GPL-3.0. A working numpy
port exists in this session's scratchpad, but it was written **while reading the
GPL source** — it is a *test oracle only*, never shippable source. Implementation
must be done from arXiv:2308.00928 alone, by an agent that has not read the
reference. Extend the existing `NOTICE` clean-room paragraph (which already does
this correctly for catch22) to cover it.

### 3.3 `RandIntC22` — highest ROI on code we already own

catch24 (catch22 + mean + std — the two features catch22 deliberately excludes,
which in a cross-section *are* the signal) computed over `k` random dilated
intervals rather than the whole window. Two independent studies put this ~3 rank
positions above whole-series catch22, level with tsfresh's 780 features.

- Intervals sampled from a **seeded** RNG that depends only on window length →
  `fit_is_empty = True`.
- Depends on §3.1 landing first. At `k=45` this is 45× the per-window cost.
- Share `_intervals.py` with Wave 3 so the dilation formula has one definition.

### 3.4 `_compress.py`

`srp` (seeded sparse random projection, density `1/sqrt(d)`), `pca` /
`svd` (fit on training rows only, reusing `reduce/`), `none`.

Document honestly that **JL bounds do not license `dim=64`**: at n=1e6, ε=0.1
needs ~10⁴ dimensions. Small outputs rest on empirical performance, not theory.
R-Clustering's evidence says 10–20 dims is the right target *for distance-based
use*; default `dim=512` for feature use, `dim=16` for the clustering path.

### 3.5 `_rff.py`

`Z = sqrt(2/D) * cos(X @ W + b)`, plus optional orthogonal `W` (QR of Gaussian
blocks, rescaled to χ_d norms — provably lower variance for the Gaussian
kernel), plus arc-cosine/ReLU features `max(0, X @ W)` (one line, no bandwidth).

**Mandatory, not optional:**
- **RVFL direct links** — always concatenate the raw standardised features with
  the random features. This is the older and better-performing design and it
  partially breaks the rotation invariance Grinsztajn shows is harmful.
- Bandwidth via median heuristic computed on an **expanding past window only** —
  the median heuristic on the full sample is a leak.
- Multi-scale bandwidth bank (0.25×, 0.5×, 1×, 2×, 4× σ) rather than one σ.

Never use the term "ELM" anywhere in code, docs or docstrings — documented
duplication of Broomhead–Lowe (1988) and Pao's RVFL (1994); reviewers reject on
the label alone. Ship the mechanism as RVFL direct links.

---

## 4. The panel contract (what the external report misses)

Two orthogonal modes, both first-class:

**Temporal mode** (`by=entity, time=date`): one embedding per `(entity, date)`
from a **strictly trailing** window. `center=True` must not exist as an
argument. Warm-up rows are dropped with the affected entities named in the
warning (mlforecast's pattern).

**Cross-sectional mode** (`is_cross_sectional = True`): fit the transform on
date *t*'s cross-section, apply to date *t*'s rows. Leak-safe by construction —
all of date *t* is observable at *t*. This is what every replicated finance
result actually does.

**Cross-date identification is a correctness bug waiting to happen.** Per-date
PCA/embedding components are unidentified up to rotation across dates. The API
must **refuse** to emit unaligned per-date components as a time-series feature.
Ship Procrustes alignment to `t-1` (default) or the `‖x_t − x_{t−1}‖²` link
penalty; require the user to pick one explicitly.

### Serialisation (`_contract.py`)

Serialise: transform version, method, seed, input-column schema, scaler state,
any fitted biases/thresholds, window and resampling policy, compression state.
Output `Float32` fixed-size Arrow arrays by default — **never** materialise
10,000 top-level Polars columns unless the user calls an explicit `unnest()`.
Deterministic inference is a compatibility promise: identical state + identical
Float32 input gives numerically stable output within a documented tolerance.

Memory: default `float32`; offer `float16` (measured median rel err 1.7e-4, p99
4.3e-4, no overflow — QUANT features max at |44| vs f16's 65504); chunk over
entities (measured *faster* than monolithic at 2k-row chunks — cache locality,
so chunking is free). The Polars↔numpy boundary is zero-copy; 100% of cost is
the numeric kernel. This retroactively validates dropping the Rust extension.

---

## 5. Guardrails — the differentiator

None of the surveyed libraries ship these. All are cheap.

1. **Never ridgeless.** The probe requires an explicit, cross-validated ridge λ.
   Passing `alpha=0` raises.
2. **Warn hard when P > T** in the readout, with a pointer to Nagel (2025).
3. **Reversal-synthetic diagnostic** (~60 LOC). Inject a strongly mean-reverting
   MA(2) into the target and re-run the pipeline. If the embedding still
   produces the same positive inverse-vol weights on recent returns, it is
   mis-learning. This is the test that busted the flagship RFF-in-finance paper.
4. **Mechanical baseline.** Recency-weighted, inverse-vol-scaled average of the
   target over the training window. If the embedding does not beat it, it
   learned nothing. Report both, always.
5. **Null-panel falsification harness.** Phase-randomised / martingale-difference
   panel with identical shape and missingness; report the null distribution.
6. **Naive baselines inside the evaluation surface** — `y_t = y_{t-1}` and the
   seasonal naive. The single most common dismissal of this class of work is
   "did you beat the random walk?"
7. **Amplitude retention.** Any conv path must offer `normalise=False` *and*
   emit explicit scale features (window σ, MAD, realised vol) alongside.
8. **Do not ship DSR/PBO as a leakage gate** — docstring must say they are
   search-intensity corrections and that a leaky oracle at Sharpe 35 passes both.

---

## 6. Waves 2–3

**Wave 2 — temporal default and the panel axis.**
Causal rolling MiniRocket-PPV: 84 fixed length-9 kernels with weights in
{−1, 2}, **500 kernels not 10,000** (R-Clustering), **permuted bias values**
(fixes the documented artificial-autocorrelation bug), left-only padding so
`C[t]` depends on `x[t−8d…t]`, then PPV/MPV as a **cumsum-based rolling mean of
an indicator** — O(1) per timestep, giving window-level causality at O(N·T)
cost. Measured 33 µs/panel-row for a 1,008-dim embedding, single-threaded pure
numpy. Drop `max` pooling (it does not cumsum; MiniRocket already dropped it).
Biases are fitted state → `fit_is_empty = False`, fit inside the fold.
Optional numba path behind the existing `fast` extra (measured ~20× gap).
Plus PreValidated Ridge (`_probe.py`) and cross-sectional mode (`_xs.py`).

**Wave 3 — breadth.** Hydra (119-line reference, easiest port in the family;
use `np.bincount` on flattened indices, **never** `np.add.at`); TensorSketch
(`np.add.at` scatter + `np.fft.rfft`, ~30 LOC); the full diagnostics harness.

**Watchlist — do not build yet.**
`SOCK` (arXiv:2606.05138, Imperial + JPMorgan) — differentiable Hydra with
softmax competition and soft-deviation pooling, explicitly designed for the
single-path small-sample financial regime; beats Hydra, matches MultiRocket with
an order of magnitude fewer features. Full pseudocode in the appendix, no public
code, single unreplicated preprint. **Highest upside item on the list.**
`TSPulse-R1` behind a `foundation` extra. `HDC` binding as an experimental
`temporal_position=` flag with the honest 17/128 number in the docstring.
`SORF-DCT`, `RQMC`, randomised signatures.

**Explicitly do not build:** MultiRocket (50k features is the wrong shape for a
panel); Detach-ROCKET/POCKET/S-ROCKET as embeddings (supervised selectors — a
leak hazard); UMAP as a feature source (`transform()` is batch-dependent —
[umap#1224](https://github.com/lmcinnes/umap/issues/1224) — so a row's embedding
depends on what else is scored alongside it, which means backtest and live
values differ; keep `PanelUMAP` for visualisation only, and say so); tabular
foundation models; GPU ports (measured slower per watt than CPU numba).

---

## 7. Licensing

**All Dempster reference implementations are GPL-3.0** — ROCKET, MiniRocket,
Hydra, QUANT — as are MultiRocket, pycatch22, HDC-MiniROCKET and MrSQM
(verified by fetching raw LICENSE files). The algorithms are published and
reimplementable; the *code* is not vendorable.

⚠️ **A BSD-3 label on aeon/sktime/wildboar does not launder GPL provenance.**
[sktime#7368](https://github.com/sktime/sktime/issues/7368), filed by sktime's
founder, states that where methods "were forked by a non-owner from GPL
repositories… there is a license violation that we need to resolve" — naming
QUANT, Hydra, ROCKET and MultiRocket. Open and unanswered for 22 months. aeon's
only permission claim is a docstring covering **MiniRocket alone**. Treat these
as *behavioural oracles for tests*, never as source to read while implementing.

Clean-room procedure: implement from the paper; the implementing agent must not
have read the reference; verify numerically against aeon/wildboar as a black
box; extend `NOTICE` with a per-method paragraph mirroring the existing catch22
one.

---

## 8. Acceptance

- `make lint`, `make typecheck` clean; `tests/test_embed_*.py` green.
- `tests/test_embed_stateless.py` — `fit_is_empty` classes produce byte-identical
  state across disjoint fit sets.
- `tests/test_embed_leakage.py` — models the existing leakage suites; each
  transform must **fail** under a deliberately leaky implementation.
- `tests/test_embed_prefix_invariance.py` — `f(x[:T])[t] == f(x[:T+k])[t]`.
- Benchmarks recorded as an *embedding* library, not a classifier library:
  throughput (windows/s, 1 thread and all), cold start, peak RSS at 10k/100k/1M
  windows, linear-probe AUROC, k-NN recall@k, robustness (noise/scale/shift/
  missingness), out-of-domain transfer, strict temporal holdout, determinism.
- catch22 batch rewrite ≥1,000 windows/s at L=128 (from 10.2).

## 9. Honest gaps

- **No published head-to-head of MiniRocket-style features vs hand-crafted
  rolling features on a financial cross-section exists.** Searching arXiv for
  `catch22 AND (financial OR stock)` and for `tsfresh` returns zero finance
  hits; searching r/quant, r/algotrading and HN returns **not one report** of
  ROCKET or MiniRocket on financial data, positive or negative. Every ranking in
  §1 comes from UCR/UEA sensor and ECG data with high SNR and phase-aligned
  shapes — financial panels have neither. **Treat all of it as priors, not
  predictions, and benchmark on our own panels before claiming anything.** This
  gap is also the opportunity: it is the most differentiating thing Panelary
  could publish.
- QUANT under a *linear* head is unmeasured; every published result uses
  ExtraTrees.
- The causal rolling-conv construction in Wave 2 has **no prior art** — the
  cumsum trick is sound but its accuracy retention is unvalidated.
- Two research lanes (non-linear DR benchmarks, NLDR/tabular GitHub sweep) were
  lost to a session restart. The external PDF partially covers that ground and
  independently reaches the same verdict (UMAP/PaCMAP/TriMap ranked last,
  "viz only"), so the gap is not blocking — but the parametric-DR question is
  unaudited.
