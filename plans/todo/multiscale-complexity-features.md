# `panelary/complexity/` — causal multiscale, scaling and complexity features: build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.**
>
> **Gate (read first).** `AGENTS.md`: *"No new public surface without a named caller in
> the engine, and a test in the engine that exercises it."* No such caller exists today
> (verified: `truepoint/src/truepoint/` has no `quant/` package; `diagnose/` is
> forbidden from importing `panelary` by `truepoint/_firewall.py::FORBIDDEN_PRODUCT`).
> This plan is therefore **M0-gated**: nothing below M0 starts until the truepoint owner
> accepts (or replaces) the caller proposed in §12. It also ranks behind the AGENTS.md
> "what to build first" items (`CompileResult.to_json`, pipeline `audit()`, as-of join).

Pure **numpy + polars** in every default path. No scipy, no PyWavelets, no sklearn in
any import that a default path reaches. `numba` only through the existing `fast` extra,
lazily, with a numpy path that is always present. Oracles are **test-only**, behind
`pytest.importorskip`.

Reuse, do not reinvent (all verified on disk 2026-09-29):
`panelary.shape._window` (`trailing_windows`, `PanelWindows`, `SortedPanel`,
`sort_panel`), `panelary.detect._moments` (`_blocked_prefix`, `_window_sums` — the
block-reset prefix sums), `panelary.depend._rolling` (`window_statistic`, `_pair_expr`,
`_check_window` — the struct + `map_batches` idiom CI already proves on polars 1.35 and
1.42), `panelary.depend._null` (`phase_randomise`, `iaaft`, `_rng`, `SurrogateResult`),
`panelary._internal._numpy_stats` (`norm_ppf`), `panelary.feature_extractors._kernels`
(`_get_cusum_numba`, the lazy-numba pattern), `panelary.synth._generate`
(`SeedSequence(seed, spawn_key=...)`), `panelary.evolve._genome` (stable `blake2b`
hashing), `panelary.registry` (`FeatureSpec`, `registry`), `panelary.testing`
(`assert_no_lookahead`, `assert_prefix_invariant`).

---

## 1. Why this exists

### 1.1 The gap, verified in the code

Panelary has a handful of complexity statistics, and **none of them can be used as a
causal row feature**:

| What exists | Why it is not a causal feature |
|---|---|
| `feature_extractors/_stats.py::permutation_entropy` (`.ts.permutation_entropy`) | Whole-series `series -> scalar`. Not in `_TS_SCALAR_AGG_SPECS`, so it is not even registered. |
| `_stats.py::sample_entropy`, `approximate_entropy` | Series-only; the `Expr` path logs and returns `NotImplemented`. `r = ratio * x.std()` over the **whole** series. |
| `_stats.py::cwt_coefficients`, `number_cwt_peaks` | Ricker CWT via `np.convolve(..., mode="same")`, which is **centred**, and stored as `float32`. As a window summary this is harmless; broadcast per row it is a look-ahead. |
| `catch22/_batch.py::_fluctuation_rows` (`SC_FluctAnal_2_dfa…`, `…rsrangefit…`) | A fixed published estimand (the proportion of scales before a breakpoint), not α. Boxes are tiled from the window **start** (`profile[:, : n_win * tau]`), so the newest `L mod τ` observations are dropped at every scale. |
| `econ/features/_longmemory.py::gph`, `local_whittle`, `rolling_long_memory_features` | Correct and causal, but they estimate `d` in the frequency domain, and the rolling path is a per-row Python loop (`econ/features/_common.py::rolling_apply`). |

No wavelet filter bank, Hurst estimator, fractal dimension, ordinal-pattern statistic
beyond plain PE, dispersion entropy or visibility graph exists anywhere in `panelary/`
(grep for `wavelet|haar|hurst|higuchi|katz|petrosian|multifractal|ordinal|visibility|lehmer`,
2026-09-29).

### 1.2 Evidence, skeptical first

- **Long memory is mostly short memory in disguise.** Lo (1991) showed that classical
  R/S cannot tell short-range dependence from long memory. With the modified R/S
  statistic, US stock returns show no long memory. *Consequence:* every scaling output
  here is a **descriptor of the window's scaling**, never a test. No p-values are
  shipped. R/S ships only with the Anis–Lloyd–Peters correction and its measured bias
  (§1.3).
- **Multifractality in returns is mostly fat tails.** Barunik, Aste, Di Matteo & Liu
  (2012) found that shuffled returns (correlations destroyed, marginal kept) look *more*
  multifractal than the originals. Most measured multifractality comes from the fat-tailed
  distribution. *Measured here* (§1.3): i.i.d. Student-t₃ noise, which has no temporal
  structure at all, gives a GHE multiscaling gap `H(1) − H(2) ≈ +0.053` at W=256.
  *Consequence:* no raw multifractal width ships as a default feature. Widths are
  reported only net of a Gaussian null table (every row) or of within-window shuffles
  (opt-in, at stride).
- **These estimators are noisy at feature-sized windows.** At W=256 the best Hurst
  estimator we measured has a standard deviation of 0.046, and the classical ones reach
  0.06–0.09. A per-row Hurst is a noisy state variable, not a point estimate. Its use is
  cross-sectional ranking and regime conditioning.
- **Positive evidence.**
  - Faria & Verona (2018, *JEF* 45:228–242): summing wavelet frequency-decomposed parts
    gives a monthly out-of-sample R² of 2.60% and a utility gain of 558 bp/yr.
  - Zunino et al. (2010, *Physica A* 389:1891): the complexity–entropy causality plane
    separates developed from emerging equity markets, i.e. it orders them by
    inefficiency.
  - Di Matteo, Aste & Dacorogna (2005): generalized Hurst exponents track a market's
    stage of development.
  - Gençay, Selçuk & Whitcher (2005): systematic risk (beta) is scale-dependent, and
    the beta–return relation strengthens at longer wavelet scales†.
  - Gneiting & Schlather (2004): fractal dimension (local roughness) and the Hurst
    effect (global persistence) are **separate** parameters. That is the reason we ship
    both families instead of treating `D = 2 − H` as redundant.

### 1.3 Measured facts that dictate the design

All measurements are scratch runs on an Apple M5 Pro (numpy 2.5.3, polars 1.44.2,
CPython 3.13, single process, 2026-09-29). Exact fGn comes from Davies–Harte. Each cell
is **bias/sd** over 400–600 seeded replicates, so the Monte-Carlo SE of a bias is about
0.003. None of this is package code. The reproduction script lands as
`benchmarks/bench_complexity_estimators.py` in M3 and must reproduce these numbers to
within MC noise.

**Hurst estimators on one window, W = 256** (fGn with the stated H; the last column is
i.i.d. t₃, true H = 0.5):

| Estimator | H=0.3 | H=0.5 | H=0.7 | t₃ |
|---|---|---|---|---|
| DFA-1, maximal-overlap boxes, OLS (scales 8..64, 13) | +0.017/0.052 | −0.005/0.073 | −0.001/0.085 | +0.007/0.073 |
| DFA-1, non-overlapping, anchored at window end | +0.016/0.052 | −0.006/0.075 | −0.004/0.089 | +0.004/0.075 |
| DFA-1, maximal overlap, WLS (w_s ∝ (W−s+1)/s) | +0.026/0.049 | +0.006/0.064 | +0.009/0.079 | — |
| DFA-2, maximal overlap (scales 12..64) | +0.029/0.058 | +0.007/0.074 | +0.011/0.095 | — |
| GHE q=1 (τ = 1..19) | −0.005/0.051 | −0.009/0.066 | −0.004/0.067 | **+0.047**/0.065 |
| GHE q=2 (τ = 1..19) | −0.007/0.048 | −0.013/0.063 | −0.011/0.063 | −0.006/0.060 |
| R/S with Anis–Lloyd–Peters (8..128) | +0.030/0.048 | −0.029/0.057 | **−0.082**/0.061 | −0.030/0.054 |
| Haar à trous wavelet variance, OLS (J=5) | −0.001/0.048 | −0.008/0.055 | −0.009/0.057 | −0.006/0.056 |
| **Haar à trous, WLS η_j = M_j/2^j, χ²-log offset** | +0.014/0.042 | +0.010/0.046 | +0.011/0.046 | — |

At W=512 the Haar-WLS standard deviation is 0.027–0.036, against 0.042–0.047 for GHE
q=2 and 0.039–0.064 for DFA-1. At H=0.9, W=256, Haar-WLS scores −0.003/0.051.

**Drift** (fGn H=0.5 plus a constant 0.2σ per step, W=256):

| Estimator | Bias |
|---|---|
| DFA-1 | +0.002 |
| GHE q=2 | **+0.088** |
| GHE q=2, demeaned in-window | −0.022 (−0.048 at H=0.7) |
| Haar | −0.007 |

**Kernels and throughput** (500 entities × 5,000 dates = 2.5M rows, W=256; multiply by
about 10 for 5,000 × 5,000):

| Fact | Measurement | Consequence |
|---|---|---|
| Haar à trous, J=6, plus windowed energies | 0.11 s | The wavelet family costs about 1 s on the full panel. |
| Lehmer codes (m=4) | 0.02 s | Ordinal coding is free. |
| Cumulative one-hot histogram (m=4) + exact entropy | 0.22 s | Every-row ordinal features are cheap. |
| GHE q=1, τ ≤ 19, window-local | 0.46 s | Cheap. |
| DFA-1, box moments from **global prefix sums** | 0.65 s, but box-RSS relative error has median **2e-5** and max **1.6e-3** (T=20k, drift 0.3σ) | **Rejected.** Cancellation grows with the level of the profile. |
| DFA-1, box statistics **direct, box-local centring** | 1.32 s, max relative error **6.6e-13** vs `lstsq` | **Adopted.** Exact and window-local, at 2× the cost. |
| SampEn, W=256, batched boolean (R, W, W) | **189 µs/window** | Every row on 25M rows would take about 79 min. SampEn runs **at stride**. |

**What the measurements decide:**

1. The default local Hurst is the **Haar à trous wavelet-variance estimator (WLS)**. It
   has the lowest variance, no drift bias and no fat-tail bias, and it comes free with
   the Haar energy features.
2. **GHE q=2** is the default for the generalized-Hurst machinery. q=1 is exposed, but it
   carries a **measured +0.047 fat-tail bias**.
3. **DFA-1** ships for familiarity and oracle parity, with maximal-overlap boxes. The gain
   over the non-overlapping version is 2–5% in sd, which is small. The real benefit is
   that it removes any box-grid anchoring choice.
4. **R/S-AL** ships as a legacy estimator with its compression bias documented. It is
   never a default.
5. The analytic Abry–Veitch χ² log-bias correction **over-corrects**: +0.015 at W=256,
   because the overlapping à trous coefficients have more degrees of freedom than
   M_j/2^j. It is replaced by a per-(W, j) offset table fitted by simulation (§6.1).

---

## 2. What already exists, and what this plan must NOT duplicate

| Existing (file::symbol) | Verdict |
|---|---|
| `feature_extractors/_stats.py::permutation_entropy` + `.ts.permutation_entropy` | **Reuse as the scalar twin.** Register it in the window-scope catalogue (§5.3) unchanged. The only new code is the rolling kernel. One test pins `rolling_permutation_entropy(W)[t] == permutation_entropy(x[t−W+1..t]) / ln(m!)`: the existing function returns raw Shannon entropy in nats (polars `entropy(normalize=True)` normalises the counts, not the value), and ours is ln(m!)-normalised. |
| `_stats.py::sample_entropy`, `approximate_entropy`, `_chebyshev_counter` | **Do not extend.** The KD-tree/sorted-sweep counter is one call per series, which becomes the banned per-window UDF when used rolling. The windowed kernel (§6.8) is a batched boolean band count. Documented discrepancy: the existing SampEn uses N−m+1 templates at length m, where Richman–Moorman use N−m. The new kernel follows Richman–Moorman, and the existing function is left alone (changing it silently would move users' numbers). |
| `_stats.py::lempel_ziv_complexity`, `binned_entropy`, `fourier_entropy`, `time_reversal_asymmetry_statistic` | Out of scope, not duplicated. The ordinal and HVG irreversibility measures (§6.6, §6.9) are rank- and graph-based and are a different estimand from the amplitude lag statistic. |
| `_stats.py::cwt_coefficients`, `number_cwt_peaks` | CWT is excluded as a feature (§4). No change here; they are not registered and stay window-scope summaries. |
| `catch22/_batch.py::_fluctuation_rows` | Not reused: it has a different estimand and start-anchored boxes. Not changed: it must reproduce the published catch22 definition. |
| `catch22/_helpers.py::_pair_counts`, `_coarsegrain_quantile` | Not reused. They are single-series helpers; the ordinal transition histogram here is batched and uses the m!·m successor code (§6.6). |
| `econ/features/_longmemory.py::gph`, `local_whittle` | Not duplicated. They are the **frequency-domain comparators** in the M3 Hurst benchmark (`H = d + ½`). Batching their rolling path belongs to econ. |
| `shape/_window.py::PanelWindows`, `trailing_windows` | **Reused** for every per-window kernel (dispersion entropy, SampEn, numpy HVG). Imported **inside functions**: `import panelary` must not load `panelary.shape._window` (`tests/test_shape_leak_safety.py::test_import_panelary_loads_no_shape_transform_and_shape_is_light`). |
| `shape/_spectral.py::Spectral`, `_paa.py`, `_delay.py` | Not duplicated. Spectral band power is the Fourier sibling of wavelet energy; §6.1 records why both exist. |
| `detect/_moments.py::_blocked_prefix`, `_window_sums` | **Reused** for every rolling sum. M1 **moves** them to `_internal/_blocked.py` and leaves an import behind in `detect/_moments.py`, the same pattern shape used for `build_tensor`. |
| `depend/_rolling.py::_pair_expr`, `window_statistic`, `_check_window` | **Reused** for two-series ops (wavelet beta and correlation, DCCA). |
| `depend/_null.py::phase_randomise`, `iaaft`, `_rng` | **Reused** for surrogate null bands (M5) and in tests. A batched per-window variant, if needed, is **added next to them** in `depend/_null.py` as a depend-owner change, never forked here. |

---

## 3. Scope, non-goals, and boundaries with sibling plans

### 3.1 In scope

1. **Causal multiresolution wavelets.** À trous Haar (causal, additive); one-sided MODWT
   coefficients for Haar, D4 and LA8 with a NaN warm-up; per-level energies, relative
   energies and Percival–Walden wavelet variance; the Haar-WLS Hurst; per-level beta and
   correlation against a reference series.
2. **Scaling.**
   - DFA-n (n = 1, 2) with maximal-overlap boxes and optional non-overlapping boxes
     anchored at the window end.
   - DMA, centred-in-window (θ = ½, default) or backward (θ = 0, Carbone).
   - Generalized Hurst H(q) (Di Matteo) and R/S-AL.
   - MF-DFA h(q), Δh, and Δα via Legendre, each net of a null.
   - ρ_DCCA(n) against a reference series.
   - Local (time-varying) Hurst, which is simply every rolling estimator above.
3. **Fractal dimension.** Higuchi, Katz, Petrosian.
4. **Ordinal family.**
   - Permutation entropy (rolling) and weighted PE.
   - Jensen–Shannon statistical complexity, which gives the complexity–entropy plane.
   - Ordinal Fisher information and missing-pattern counts.
   - Ordinal transition network: self-loop rate and transition entropy.
   - Ordinal time irreversibility and composite multiscale PE.
5. **Other entropies.** Dispersion entropy (window-local normalisation), SampEn and
   refined-composite MSE at stride, and HVG time irreversibility.

### 3.2 Non-goals, with reasons

| Not shipping | Why |
|---|---|
| EMD / EEMD / CEEMDAN / VMD | Adaptive. An IMF at `t` changes when data is appended: sifting interpolates spline envelopes through extrema that include future extrema, and end effects move with the series end. VMD solves a global Fourier-domain optimisation, which is two-sided. There is no stable causal definition, so `assert_prefix_invariant` could not certify one. |
| CWT features, wavelet coherence, MODWT MRA synthesis (D_j, S_J for L > 2), SSA reconstruction, Hilbert/analytic signal via FFT, `filtfilt` | Two-sided by construction. SSA belongs to shape W2 anyway. |
| **Aligning** MODWT coefficients by their phase delay ν_j | Percival–Walden shift coefficients by \|ν_j\| for plotting. That is a **forward shift**, which is a look-ahead. The delay is reported as metadata (§6.1) and never applied. |
| Sevcik FD | Its window-range normalisation enters a nonlinear sum, so it costs O(W) per row. Esteller et al. (2001) found it tracks Katz. |
| Correlation dimension, Lyapunov exponents, RQA | Unreliable at W ≤ 512: the Eckmann–Ruelle bound caps a reliable D at about 2·log10 N. RQA needs O(W²) memory, and shape §6 already excludes recurrence plots. |
| ApEn windowed, fuzzy / bubble / slope entropies | ApEn's self-matching bias is the reason SampEn exists. The rest is an EntropyHub zoo with no finance evidence. Keep the surface small. |
| Long-memory tests (Lo's modified R/S, GPH significance) | Tests, not features. Econ owns them (`econ/features/_longmemory.py`). |
| Lempel–Ziv / Kontoyiannis entropy rate (AFML ch. 18) | `lempel_ziv_complexity` exists. A windowed Kontoyiannis estimator is a possible follow-up, not this plan. |

### 3.3 Boundaries with sibling plans (written 2026-09-29, in parallel)

| Sibling | Boundary |
|---|---|
| **5 ohlc-volatility-and-liquidity** | **They own** rough-volatility H of log-vol: realized-variance construction, measurement-error bias corrections, and the H≈0.1 interpretation. **We own** the generic estimators. The Gatheral–Jaisson–Rosenbaum smoothness estimator is GHE `m(q, Δ) ∝ Δ^{qH}`, so they call `complexity._scaling.ghe_rows(X, q=..., taus=...)` on log-RV. We keep that signature stable and do not ship a "rough vol" feature. |
| **7 causal-state-space-and-regimes** | **They own** causal IIR filters (EWMA, Kalman, one-sided HP) and regime models. **We own** FIR dyadic filter banks (à trous / MODWT). Neither ships the other's filter. They may feed our features into regime models. |
| **3 covariance-and-market-state** | **They own** N×N matrices and market-state indices, including any multiscale covariance matrix. **We own** per-entity, per-level beta and correlation against one reference column (a row feature). They may call `_internal/_wavelets.py` for per-level returns. |
| **2 drift-monitoring-and-sequential-inference** | They own monitors and sequential tests. Our features are inputs to them, never tests. |
| **6 network-and-spatial-panel** | They own cross-entity graphs. Our ordinal transition networks and HVGs are **within-series** graphs; no overlap. |
| **9 tail-risk-and-self-excitation** | They own tail indices, EVT, Hawkes, and the Markov-switching multifractal (MSM) volatility model. We own multifractal spectra and DFA (their plan states the same split). We touch tails only through the shuffle/null correction of multiscaling (§6.4). We estimate no tail index. |
| **3 / 7 (cross-check)** | Their plans already state the matching split: sibling 3 uses `xs_`-prefixed names (`xs_entropy`, eigen-entropy) and has no per-entity entropies; sibling 7 has "no wavelets here, no IIR there". None of our registry names is a bare `entropy`. |
| **1, 4, 10** | No overlap. |
| **shape W2 (`shape/_dwt.py`, not started)** | This row is the boundary: the causal filter-bank **kernel** lives in `_internal/_wavelets.py`. Shape W2 owns window→coefficient-vector LIFT/COMPRESS transforms (decimated DWT of a trailing window, the `whole_series` flavour, `plan()` budgets, and an optional `pywt` backend). Shape imports filter constants and the pyramid from `_internal/_wavelets.py` and must not redefine them. We never emit a coefficient *vector* per row. Whichever lands first creates `_internal/_wavelets.py`, and the second imports it. |

---

## 4. Leak-safety design: the traps and how each is designed out

Hard invariants, inherited from `detect` and `depend` and applying to every function:

1. **Prefix invariance, bitwise.** `f(x[:T])[t] == f(x[:T+k])[t]`. No quantity may depend
   on `len(x)`: not scale grids, not `k_max`, not `m`, not the q-grid, not tolerances, not
   null tables, not seeds, not strides.
2. **Window closure.** For every rolling op, `rolling_f(W)[t]` equals the scalar twin
   `f(x[t−W+1 .. t])`. It is bitwise equal where the kernel is exact-integer or
   shifted-slice. Where a blocked rolling sum is used it agrees to ≤ 1e-12 relative. One
   oracle test therefore covers both scopes.
3. **Determinism.** No unseeded RNG. Surrogate seeds derive from `(seed, entity, time)`
   (§4.2).
4. **float64** in every accumulator; Float32 inputs are upcast.
5. **Row independence.** No BLAS matmul in a bitwise path (the `catch22/_batch.py` rule),
   because a GEMM's blocking may depend on how many rows share the call. Regressions on
   fixed designs are `(logF * w).sum(-1)` along a contiguous last axis. Chunk size and
   thread count never change a bit of output.
6. **No `rolling_map`, no per-window Python UDF.** A stride grid is not a loophole: its
   evaluated windows are still batched.
7. **NaN policy.** A window containing NaN yields null, never a statistic on a silently
   shortened sample (the `depend/_rolling.py` precedent). For streams, NaN propagates for
   exactly the filter support. Rolling sums replace NaN by 0 and carry a rolling NaN count.
8. **Backend invariance.** Under `backend="auto"`, numba is used only for kernels whose
   outputs are **integers or exact comparisons**: pattern counts, SampEn match counts, HVG
   degrees, running max/min. Floats are then computed from those integers by shared numpy
   code, so `"auto"` and `"numpy"` agree bit for bit. The one float numba kernel (the DFA
   box loop, §6.2) runs only under an explicit `backend="numba"`. The backend is chosen by
   availability and an explicit parameter, **never by data size**, because a size-based
   switch would break prefix invariance.

### 4.1 Trap table

Every trap has a **deliberately leaky twin** in the tests (the `depend` precedent). The
verifier must catch the twin, which proves the test has teeth.

| Trap | Where it bites | Design | Leaky twin that must FAIL |
|---|---|---|---|
| Centred / two-sided transforms | CWT `mode="same"`, `filtfilt`, FFT Hilbert, MODWT MRA synthesis | Only one-sided FIR recursions on `t, t−2^{j−1}, …`. No synthesis for L > 2. | Circular (periodised) Haar in numpy: `assert_no_lookahead` fails at the series start. |
| PyWavelets periodisation | `pywt.swt` / `wavedec(mode="periodization")` wrap the **end** into the first coefficients | No boundary extension at all: NaN until `L_j − 1` rows exist in the entity. | `pywt.swt` energies (importorskip), expected to fail `assert_no_lookahead`. |
| Box anchoring | DFA / MF-DFA / DCCA / R/S boxes tiled from the series or window **start** | Default: **maximal-overlap** boxes (every box wholly inside the window, so there is no grid). Non-overlapping mode anchors the last box at `t`. | DFA on an expanding window with boxes from the entity start and scales from `T`: `assert_prefix_invariant` fails. |
| Global normalisation or tolerance | SampEn `r` from the full series; dispersion entropy μ, σ from the full series; z-scoring before any feature | μ, σ and `r` are recomputed **inside each window**. | SampEn with `r = 0.2·std(full series)`: `assert_no_lookahead` fails. Dispersion entropy with global μσ: fails. |
| "Centred" confusion (DMA) | A centred MA on the **stream** reads `t + n/2` | Centred **inside the window** is legal. The residual at box end `e` is `Y_{e−(n−1)/2} − MA_n(e)`, which is causal at `e`. | Centred DMA on the stream: `assert_no_lookahead` fails. |
| Length-derived parameters | `k_max = N//4`, `scales = f(T)`, `m = f(T)`, null table indexed by T | Every grid is a pure function of `(W, params)`. The scalar twin uses `len(window)` as W, which is legitimate at window scope. | Scale grid from T: `assert_prefix_invariant` fails. |
| Adaptive decompositions | EMD / VMD | Excluded (§3.2). | — |
| Surrogate seeds | Shuffles seeded by row count, `hash()`, or global RNG state | `window_rng(seed, entity, time)` (§4.2). | Seed = `hash(len(x))`: `assert_prefix_invariant` fails. |
| Stride grid anchored at the end | Evaluating at `T − 1, T − 1 − s, …` | The grid is anchored at the **entity's first row**: evaluate where `(pos − (W−1)) % stride == 0`. Rows in between take an **as-of forward fill** from the latest evaluated row ≤ t, within the entity. | Grid anchored at `T`: `assert_prefix_invariant` fails. |
| Phase alignment | Shifting LA8 coefficients by ν_j | Never shifted. The delay is reported. | Aligned LA8: `assert_no_lookahead` fails. |
| Reference series built with look-ahead | "Market" column computed from a future-listed universe | Not ours to fix, but the docstrings require the reference column to be point-in-time (e.g. `pl.col("ret").mean().over("date")`). | — |

### 4.2 Seeds

```python
def window_rng(seed: int, entity_key: object, time_ordinal: int) -> np.random.Generator:
    key = int.from_bytes(hashlib.blake2b(repr(entity_key).encode(), digest_size=8).digest(), "big")
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(entropy=int(seed), spawn_key=(key, int(time_ordinal)))))
```

- `time_ordinal` is the row's **time value**, as integers for Int columns, days since the
  epoch for Date, and µs since the epoch for Datetime after unit normalisation. It is
  never the row index. Both are prefix-invariant, but only the time value also survives
  left-truncation and entity reordering.
- Python `hash()` is salted per process, so it is banned.
- The `blake2b` precedent is `evolve/_genome.py`; the `SeedSequence` precedent is
  `synth/_generate.py`.
- `B` shuffles for one window come from one generator, drawn in fixed order.

---

## 5. Module placement and public API

### 5.1 Layout

```
panelary/_internal/_wavelets.py   # filter constants (haar, d4, la8), a-trous Haar, causal MODWT pyramid,
                                  #   L_j, phase delays nu_j. numpy only. Shared with shape W2.
panelary/_internal/_blocked.py    # MOVED from detect/_moments.py: _blocked_prefix, _window_sums (+ alias left)
panelary/complexity/__init__.py   # public API; imports _ts for its side effect; kernels imported lazily
panelary/complexity/_stream.py    # rolling sums over blocked prefixes, NaN counts, van Herk/Gil-Werman
                                  #   rolling max/min, stride grid + as-of ffill, entity chunking, n_jobs pool
panelary/complexity/_seeds.py     # window_rng
panelary/complexity/_design.py    # fixed log-log designs, WLS weights, calibrated offset tables,
                                  #   Anis-Lloyd table, Gaussian-null tables (checked-in constants)
panelary/complexity/_wavelet.py   # energies, wavelet variance, haar_hurst, beta/correlation
panelary/complexity/_scaling.py   # box stats (direct), DFA-n, DMA, GHE, R/S-AL, DCCA, MF-DFA
panelary/complexity/_fractal.py   # Higuchi, Katz, Petrosian
panelary/complexity/_ordinal.py   # Lehmer codes, LEX/REV tables, histograms, PE/WPE/C_JS/Fisher/
                                  #   missing/OTN/irreversibility/composite MPE
panelary/complexity/_entropy.py   # dispersion entropy, SampEn, RCMSE (stride)
panelary/complexity/_hvg.py       # HVG degrees: numpy binary-lifting PGE/NGE; numba exact incremental
panelary/complexity/_numba.py     # lazy numba kernels (integer-statistic kernels only), _get_cusum_numba pattern
panelary/complexity/_ts.py        # additive .ts methods + FeatureSpec registration (mirrors namespaces/ts.py)
panelary/complexity/_features.py  # ComplexityFeatures (PanelTransformer), features(), presets, haar_decompose, modwt
panelary/complexity/_scalar.py    # scalar twins (series -> scalar map_batches, returns_scalar=True)
```

`panelary.complexity` is imported **eagerly** by `panelary/__init__.py`, as `depend` is,
so that `.ts` ops resolve after `import panelary`. The import cost must stay inside
`tests/test_import_hygiene.py::test_import_time_within_budget`. That is possible because
`__init__` loads only `_ts.py`, and every kernel and `shape._window` import is
function-local.

### 5.2 Expression ops (`safe_scope="rowwise"`, `axis="time"`, `flavour="trailing"`)

The brief asked for `safe_scope="window"`. **That label is wrong for these ops: it
under-claims them, and the conformance suite would count them against the registry.**

- In `registry.py`, `"window"` means *a summary of an already-delimited window,
  broadcasting it per row is a look-ahead*. A correct rolling op is not a look-ahead per
  row, so the label would be false.
- `tests/test_registry_conformance.py::test_window_spec_is_not_prefix_invariant_per_row`
  expects a window spec to *fail* prefix invariance when broadcast. A correct rolling op
  passes, so each one would be tallied as "degenerate" and drag
  `test_window_evidence_coverage_is_accounted_for` toward its 80% floor (§8.6).
- A trailing-window statistic evaluated at every row is `"rowwise"`, which is the
  `depend/_rolling.py` precedent. `"window"` belongs to `output_shape="scalar"` specs
  (`test_scalar_output_is_never_rowwise`).

So each windowed feature ships as **two registered specs**:

- a `rolling_<name>` op that is `rowwise` and verified by both instruments;
- a scalar twin `<name>` that is `window` scope and joins `extract_features`.

Core rolling ops, registered from M1–M4 at tier B once their gates pass:

```python
pl.col("ret").ts.haar_detail(level=3)                          # stream coefficient w_j(t); NaN for pos < 2^j-1
pl.col("ret").ts.rolling_haar_energy(level=3, window=256, relative=True)
pl.col("ret").ts.rolling_haar_hurst(window=256, levels=None)  # levels=None -> floor(log2(W/8))
pl.col("ret").ts.rolling_wavelet_beta("mkt_ret", level=4, window=256)   # also rolling_wavelet_corr
pl.col("ret").ts.rolling_dfa(window=256, order=1)             # alpha
pl.col("ret").ts.rolling_ghe(window=256, q=2.0)               # H(q)
pl.col("ret").ts.rolling_higuchi_fd(window=256, k_max=10)
pl.col("ret").ts.rolling_permutation_entropy(window=256, m=None, tau=1, weighted=False)
pl.col("ret").ts.rolling_ordinal_complexity(window=256, m=None, tau=1)  # C_JS
pl.col("ret").ts.rolling_ordinal_irreversibility(window=256, m=None, tau=1)
pl.col("ret").ts.rolling_dispersion_entropy(window=256, c=6, m=2)
```

- `m=None` resolves to the largest m with `5·m! ≤ W − (m−1)τ` (Riedl et al. 2013†).
  That gives m=4 at W=256 and m=3 at W=64, and it depends on W only.
- Two-series ops (`other=`) use the `_pair_expr` struct idiom.
- Every op validates `window` with `_check_window` (an int, never derived from `len`).

### 5.3 Scalar twins (`safe_scope="window"`, `output_shape="scalar"`)

The scalar twins are:

- `haar_hurst`, `dfa_alpha`, `ghe_hurst`, `hurst_rs`;
- `higuchi_fd`, `katz_fd`, `petrosian_fd`;
- `weighted_permutation_entropy`, `ordinal_complexity`, `ordinal_fisher`,
  `missing_patterns`, `ordinal_irreversibility`;
- `dispersion_entropy`, `hvg_irreversibility`;
- plus the **existing** `permutation_entropy`, registered as-is.

They join `extract_features` through an **opt-in** path: a small additive hook in
`feature_extractors/_catalogue.py` (orchestrator-owned).

- `_TS_OPTIN_AGGS`, plus a `register_optin_scalar(name, params, builder)` function.
- `features="all"` stays **byte-identical**. Adding columns to the default would be a
  silent breaking change to everyone's `extract_features(df)`.
- Named lists resolve against base ∪ opt-in.
- New preset strings: `features="complexity"` and `"complexity_full"`.

Only the rolling ops need the M3/M4 accuracy gates. The scalar twins are thin, because
they call the same kernel on one window.

### 5.4 Frame-level entry points (throughput path)

```python
pn.complexity.features(panel, "ret", *, window=256, preset="core" | "full",
                       features=None, other=None, stride=None, seed=0,
                       backend="auto", n_jobs=1, max_bytes=512 * 2**20) -> PanelFrame
pn.complexity.haar_decompose(panel, "ret", *, levels=6) -> PanelFrame   # ret_w1..w6, ret_c6; x == c6 + sum w
pn.complexity.modwt(panel, "ret", *, wavelet="haar" | "d4" | "la8", levels=4) -> PanelFrame
```

**`features()`** is `ComplexityFeatures(PanelTransformer)`, with
`panel_safe = leakage_safe = True` and an empty fit.

- It sorts **once** (`shape._window.sort_panel`, imported lazily).
- It lays the values out per entity chunk. The chunk size comes from a
  bytes-per-row table and `max_bytes`, not from data.
- It runs each kernel **once** per chunk, and each output column is written exactly
  once.
- `n_jobs > 1` runs entity chunks in a `ThreadPoolExecutor`. numpy ufuncs release the
  GIL, and chunks are row-independent, so the output does not depend on `n_jobs`.
- It returns keys plus feature columns, one row per input row.

`pn.features(panel, method="complexity", window=...)` routes here, the way `"delay"` and
`"spectral"` route to shape in `_internal/_verbs.py::features` (orchestrator-owned edit).

**Registry.** Each frame function registers one spec: namespace `"complexity"`,
`output_shape="frame"`, `rowwise`. The conformance suite gets a `_FRAME_OPS` adapter row
for each.

**Presets:**

- `core` (every row, exact, about 10 s per 25M rows by estimate):
  `haar_hurst`, `haar_energy_rel_{1..J}`, `higuchi_fd`, `permutation_entropy`,
  `ordinal_complexity`, `ordinal_irreversibility`.
- `full` adds:
  - `dfa_alpha`, `ghe_hurst(q=2)`, `ghe_multiscaling_excess`, `hurst_rs`;
  - `katz_fd`, `petrosian_fd`;
  - `wpe`, `ordinal_fisher`, `missing_patterns_excess`, `otn_self_loop`,
    `otn_transition_entropy`, `mpe_{τ}`;
  - `dispersion_entropy`, `mfdfa_dh_excess`;
  - `wavelet_beta_{j}` / `wavelet_corr_{j}` and `dcca_rho_{n}`, both only if `other=`
    is given.
- Stride / shuffle / numba-preferred features are **never** in a preset. They must be
  named: `sampen`, `rcmse_{τ}`, `hvg_irreversibility`,
  `ghe_multiscaling_shuffle_corrected`, `mfdfa_dh_shuffle_corrected`.

**`.ts` ops versus `features()`.** The `.ts` ops are for composability and are
per-entity `map_batches` under `.over`. `features()` is for throughput. **Both call the
same kernel**, and the window-closure test (§8.2) pins them together.

---

## 6. Algorithms: the chosen method per feature, and why

Conventions:

- `x` is the **increments** (returns) series. The profile `Y_k = Σ_{i≤k} x_i` (cumulative
  from the entity start) is built internally.
- DFA, DCCA and MF-DFA with detrending order ≥ 1 are invariant to adding constants and
  linear terms to Y. So the window-mean subtraction and the profile origin are
  irrelevant, and **box statistics are computed once per (box end e, scale s) on the
  stream and reused by every window containing that box**. This is the central speed
  idea.
- `input="levels"` is available; it skips the cumsum. A cheap warning fires when the
  lag-1 autocorrelation of the input exceeds 0.95, which usually means prices were
  passed where returns were expected.
- Rolling means of nonnegative stream quantities (f², w², |S_τ|^q) use `_window_sums`
  over `_blocked_prefix`. The error bound is then independent of T.
- Log-log slopes use a **fixed weight vector** per `(W, params)`, precomputed in
  `_design.py`, computed as `(L * w).sum(-1)`.

### 6.1 Wavelets (`_internal/_wavelets.py`, `complexity/_wavelet.py`)

**Causal MODWT pyramid.**

- `V_0 = x`
- `W_j(t) = Σ_{l<L} h̃_l V_{j−1}(t − 2^{j−1} l)`
- `V_j(t) = Σ_{l<L} g̃_l V_{j−1}(t − 2^{j−1} l)`
- with `h̃ = h/√2` and `g̃ = g/√2`.

The coefficient at t is valid iff the entity has at least `L_j = (2^j − 1)(L − 1) + 1`
observations up to t. For `t ≥ L_j − 1` it is **identical** to the Percival–Walden
circular MODWT (no wrap is reached), so non-boundary coefficients match
waveslim/wmtsa-style outputs exactly. Cost is O(L·J) vector ops over the whole (N, T)
block, with no per-row work.

**À trous Haar is the L=2 case.**

- `c_j(t) = ½(c_{j−1}(t) + c_{j−1}(t − 2^{j−1}))`
- `w_j(t) = c_{j−1}(t) − c_j(t)`

This equals the Haar MODWT `W_j(t)` exactly, and **only for Haar** is the decomposition
additive: `x = c_J + Σ_j w_j`. That additivity is why `haar_decompose` is the only
multiresolution decomposition shipped. It is exactly causal (Renaud, Starck & Murtagh
2005; Murtagh, Starck & Renaud 2004†).

**Filters.**

- Haar; D4 (db2); LA8 (sym4).
- Constants are transcribed from Daubechies (1992, Table 6.3) with a test of
  orthonormality, double-shift orthogonality, `Σg = √2`, and the vanishing moments.
- pywt parity of the constants only is test-only.
- The phase delay ν_j of LA8 (Percival & Walden §4.8) is exposed as metadata and is never
  applied (§3.2).

**Features over a window of W.** The window uses only coefficients whose support lies
inside it, `s ∈ [t − W + L_j, t]`, so `M_j = W − L_j + 1` of them. That is a rolling mean
over `M_j` of the causal coefficient stream, which gives window closure for free.

- Wavelet variance: `ν̂²_j = mean(W_j²)` (the unbiased MODWT estimator, Percival & Walden
  ch. 8).
- Relative energy: `ν̂²_j / Σ_k ν̂²_k`.
- Per-level beta: `β_j = Σ W_j^y W_j^m / Σ (W_j^m)²`, and correlation `ρ_j`, both O(1) per
  row (Gençay et al. 2005).

**Haar Hurst (the default local Hurst).**

- `Ĥ = 1 + ½ · Σ_j ω_j (log2 ν̂²_j − δ_{W,j})` for `j = 1..J`, with
  `J = floor(log2(W/8))`.
- WLS weights come from `η_j = M_j/2^j` (effective degrees of freedom).
- `δ_{W,j}` is a **simulation-calibrated** offset table. It is generated once by a
  seeded script on Gaussian white noise, checked in with the script, and indexed by
  (W, j). It is never computed at run time.
- Target: |bias| ≤ 0.01 for H ∈ [0.3, 0.9] at W ∈ {128, 256, 512}.
- Cost: O(J) per row.

**Why Haar is the default and LA8 is an option.** Haar has the shortest warm-up and the
smallest delay, and it has the measured best bias and variance (§1.3). It has only one
vanishing moment, so for strongly persistent input (H → 1, or levels) low-frequency
leakage biases ν̂²_j. `wavelet="la8"` (four vanishing moments) is the documented remedy,
at a warm-up of 106 rows for j = 4.

**Why wavelet energy and not shape's Spectral bands.** A trailing FFT band power is
O(W log W) per row and has no time localisation inside the window. Wavelet energy is
O(J) per row, and it localises scale energy at the newest end of the window.

### 6.2 DFA-n, DMA, DCCA (`_scaling.py`)

**Box statistic.**

- `f²(e, s) = (1/s)·[Σ_u (Y_u − Ȳ)² − Σ_{q=1}^{n} (φ_q · (Y − Ȳ))²]` over
  `u = e−s+1 .. e`.
- `φ_q` are orthonormal discrete (Gram) polynomials on `0..s−1`, precomputed per s.
- It is computed **directly** by shifted-slice accumulation, O(s·(n+2)) vector ops per
  scale.
- Measured max relative error is 6.6e-13, against 1.6e-3 for prefix-sum moments. That
  is why the prefix-sum formulation is rejected (§1.3).
- The numba kernel computes the same box in a tight loop. It is a float path, so it is
  **not** bit-identical to numpy (invariant 8), and it is off unless `backend="numba"`
  is explicit.

**Window aggregation.**

- Maximal overlap: `F²(t, s) = mean_{e ∈ [t−W+s, t]} f²(e, s)`, a rolling mean over
  `W−s+1` box ends.
- Non-overlap: the strided subset `e = t, t−s, …`, a rolling sum on residue classes.
- Both reuse the same `f²` array, so parity tests against any oracle's segmentation
  select the matching subset of box ends. No extra public mode is needed.

**Estimate.** `α = Σ_s ω_s · ½ log F²(t, s)`.

- Scales are geometric, 2^{1/4} apart, rounded and made unique, over
  `[s_min, floor(W/4)]`, with `s_min = 8` for n=1 and 12 for n=2.
- Weights are OLS by default. WLS (`w_s ∝ (W−s+1)/s`) is opt-in: −10% sd, +0.006 bias,
  measured.
- `order=2` is available. It measured worse at W=256, so it is not the default.
- Cost is about 15 s per 25M rows for 13 scales (from the 1.32 s measurement).

**DMA (Alessio et al. 2002; Carbone et al. 2004).**

- The residual stream is `r_θ(e, n) = Y_{e − ⌊(n−1)θ⌋} − MA_n(Y)(e)`, with n odd.
- `σ²_DMA(t, n) = mean_{e ∈ [t−W+n, t]} r²`.
- For θ = ½ (default), a centred MA preserves linear trends, so r is invariant to the
  window mean.
- For θ = 0 (Carbone's backward DMA) it is not invariant. The window-demeaned value is
  recovered exactly as `mean(r²) − 2c·x̄_w·mean(r) + c²·x̄_w²` with `c = (n−1)/2`, still
  O(1) per row.
- Shao et al. (2012) rank centred DMA and DFA as "the methods of choice" and backward DMA
  as inferior. The brief's "backward DMA" is therefore available as `theta=0`, not the
  default.

**DCCA (Podobnik & Stanley 2008; Zebende 2011).**

- The box covariance `f²_xy(e, n)` is the same direct projection applied to both
  profiles.
- `ρ_DCCA(t, n) = mean_e f²_xy / sqrt(mean_e f²_xx · mean_e f²_yy)` over the same
  maximal-overlap box ends.
- `|ρ| ≤ 1` holds exactly by Cauchy–Schwarz on the pooled residuals, and it is tested.
- Kristoufek (2014) supports ρ_DCCA as a correlation estimator for non-stationary series.
- It is opt-in in `full`, because the Haar `wavelet_corr_j` gives a scale-resolved
  correlation at O(1).

### 6.3 GHE and R/S (`_scaling.py`)

**GHE (Di Matteo et al. 2005; Di Matteo 2007).**

- `S_τ(e) = Σ_{i=e−τ+1}^{e} x_i` uses **direct** shifted sums (τ ≤ 19 adds; exact, with
  no cumsum cancellation).
- `K_q(t, τ) = mean_{e ∈ [t−W+1+τ, t]} |S_τ(e)|^q`.
- `H(q) = (1/q) Σ_τ ω_τ log K_q(τ)`, where `ω` is Di Matteo's average of the OLS slopes
  over `τ_max ∈ [5, min(19, W/10)]`. Averaging slopes is linear, so it **collapses to one
  precomputed weight vector** at no extra cost.
- `demean=True` (q=2 only) removes drift in O(1) from the rolling first and second
  moments of `S_τ`. It trades bias at high H (§1.3), so it is off by default, and
  `haar_hurst` is recommended for drifting series.
- `ghe_multiscaling = H(1) − H(2)` (Morales, Di Matteo & Aste 2013†) is shipped only as
  `…_excess` (minus the Gaussian-null median for (W, τ-grid) from `_design.py`) or as
  `…_shuffle_corrected` (§6.4).

**R/S-AL (Anis & Lloyd 1976, corrected by Peters 1994).**

- Per box: `(max_u Z_u − min_u Z_u)/sd` with `Z_u = Σ(x − x̄_box)`, computed directly in
  O(n) per box. Max and min are exact, and sd uses ddof=0.
- Boxes are maximal-overlap.
- `H_AL = ½ + Σ_n ω_n [log RS(n) − log E_AL(n)]`.
- `E_AL(n) = ((n−½)/n) · Γ((n−1)/2)/(√π Γ(n/2)) · Σ_{i<n} √((n−i)/i)`, via `math.lgamma`.
  For n > 340 the Γ-ratio is `(nπ/2)^{−½}`. It is tabulated per n.
- The measured compression bias (−0.08 at H=0.7) goes in the docstring.

### 6.4 MF-DFA and the multiscaling nulls (`_scaling.py`)

**MF-DFA (Kantelhardt et al. 2002).**

- `F_q(t, s) = [mean_e (f²(e,s))^{q/2}]^{1/q}`; `F_0 = exp(½ mean_e log f²)`.
- It reuses the DFA `f²` array, costing O(|S|·|Q|) per row.
- `h(q) = Σ_s ω_s log F_q`.
- Default q-grid `{−3, −2, −1, 0, 1, 2, 3}`.
- Headline `Δh = h(−2) − h(2)`, which is more stable than the extreme q.
- `Δα` comes from the Legendre transform (`τ(q) = q·h(q) − 1`, `α = dτ/dq` by central
  differences, `f = qα − τ`). It is opt-in as the noisiest output.
- Boxes with `f² = 0` make negative q infinite. The row is null, never floored, because
  a floor would be a silent distortion.

**Nulls.**

- **`*_excess` (every row, deterministic).** Subtract the median width of Gaussian
  white noise at the same `(W, grids)`. The table is generated by a seeded script and
  checked in, as a fitted constant in the `detect._critvals.mc_table` style.
- **`*_shuffle_corrected` (opt-in, at stride).** Subtract `mean_b width(shuffle_b(window))`
  for `B = 8` within-window permutations drawn from `window_rng`. Shuffling keeps the
  window's marginal and removes its time structure, so the difference isolates
  correlation-driven multiscaling from fat-tail-driven multiscaling (Barunik et al.
  2012). Default stride `W//4`.
- Budget, estimated: 25M rows / 64 × 8 shuffles × about 9e4 ops ≈ 3e11 ops. That is
  roughly 5 min single-threaded and under 1 min with `n_jobs=8`, and it is measured in M5.

### 6.5 Fractal dimension (`_fractal.py`)

**Higuchi (1988).**

- `d_k(e) = |x_e − x_{e−k}|` for `k = 1..k_max`.
- Residue-class sums inside the window come from **strided cumulative sums**
  (`C_k(e) = d_k(e) + C_k(e−k)`, a cumsum on each stride-k subsequence). Each class sum
  is then O(1) per window, so the cost is O(k_max²/2) per row, 55 terms at k_max = 10.
- Higuchi's exact per-class normalisation `(N−1)/(⌊(N−m)/k⌋ k)` is kept, for antropy
  parity.
- `FD = −Σ_k ω_k log L(k)`.
- Default `k_max = 10`, and it requires `W ≥ 8·k_max`. Esteller et al. (2001) rank
  Higuchi the most accurate of the classical waveform estimators, which is why it is in
  `core`.

**Katz (1988), antropy's definition.**

- `L = Σ|Δx|` (rolling sum), `n = W − 1`, `d = max_i |x_i − x_a|` with `a` the window's
  first index. So `d = max(rollmax − x_a, x_a − rollmin)`.
- The rolling max and min use **van Herk / Gil-Werman**: block prefix-max and suffix-max
  with blocks anchored at the entity start. Three vectorised passes, O(1) per element,
  exact, and bitwise prefix-invariant. The suffix max only reads up to t, which the tests
  verify.
- `KFD = log10(n) / (log10(n) + log10(d/L))`.

**Petrosian (1995).**

- `N_Δ` = number of sign changes of Δx in the window, a rolling integer count.
- `PFD = log10 W / (log10 W + log10(W/(W + 0.4 N_Δ)))`.
- O(1) per row.

### 6.6 Ordinal family (`_ordinal.py`)

**Codes.** The pattern ending at `e` uses `v_i = x_{e−(m−1−i)τ}`.

- Lehmer code: `L(e) = Σ_{i<m−1} c_i (m−1−i)!`, where `c_i = #{j > i : v_j < v_i}`.
- That is `m(m−1)/2` comparisons of shifted (N, T) arrays, stored as int16, fully
  vectorised. It measured 0.02 s per 2.5M rows at m = 4.
- Ties: a strict `<` gives the Bandt–Pompe order-of-occurrence rule (earlier counts as
  smaller).
- A `tie_fraction` diagnostic column is emitted by `features()`. Discretised prices with
  many zero returns bias PE (Zunino et al. 2017†), and the docs say so.

**Tables.** These are fixed per m and built once:

- `LEX[m]`: Lehmer → the lexicographic index of the `argsort` permutation. It is needed
  for ordpy's Fisher ordering.
- `REV[m]`: code → the code of the time-reversed pattern.
- `G[c] = c·ln c` for `c ≤ W`.

**Histograms.**

- The patterns inside a window are those with `e ∈ [t−W+1+(m−1)τ, t]`, so
  `W_p = W − (m−1)τ` of them.
- Default: a **cumulative one-hot** `(G, T+1, m!)` int32 per entity chunk, with
  `hist_t = C[t] − C[t−W_p]`. The chunk size G comes from `max_bytes`: 96 B/row at m=4,
  480 B/row at m=5.
- For m ≥ 6: an entity-vectorised **time loop** keeps a `(N, m!)` histogram with two
  fancy-index increments per step. That is T Python iterations regardless of N, and
  bounded memory.
- Measured at m=4: 0.22 s per 2.5M rows, including entropy.

**Features from `hist` (integers, so every feature is bitwise-exact given the counts):**

| Feature | Definition |
|---|---|
| PE | `[ln W_p − (1/W_p) Σ_k G[h_k]] / ln m!`, summed over k in fixed order. |
| WPE (Fadlallah et al. 2013) | Weights = population variance of `v`, as rolling weighted counts per code via blocked sums. Scale-invariant within the window. |
| C_JS (Rosso et al. 2007) | `Q_0 · J(P, U) · H_S`, with `J = S((P+U)/2) − S(P)/2 − S(U)/2` and `Q_0 = −2 / [((N+1)/N) ln(N+1) − 2 ln 2N + ln N]`, `N = m!`. Empty bins add a closed-form constant. Output pair `(H_S, C_JS)` is the complexity–entropy plane (Zunino et al. 2010). |
| Fisher (Olivares, Plastino & Rosso 2012†) | `F_0 Σ_i (√p_{i+1} − √p_i)²` in `LEX` order. `F_0 = 1` if all mass sits on the first or last pattern, else ½. |
| Missing patterns (Amigó et al. 2007; Zanin 2008) | `#{k: h_k = 0}/m!`. Headline `missing_patterns_excess` subtracts the i.i.d. expectation for `(m, W_p)` from a seeded table. Meaningful only when `W_p ≫ m!`. |
| OTN (Small 2013; McCullough et al. 2015) | Patterns at `e−τ` and `e` share m−1 values, so the successor is `r_e = #{shared v_j < x_e}` ∈ `0..m−1`. The transition code `L(e−τ)·m + r_e` needs only **m!·m bins**, not (m!)². Self-loop rate is `mean 1[L(e) = L(e−τ)]`. Transition entropy is `[H(pairs) − H(sources)]/ln m`. |
| Irreversibility (Zanin et al. 2018†) | `JSD_2(P, P∘REV) ∈ [0, 1]`. It is bounded and needs no smoothing, unlike KLD. |
| Composite multiscale PE (after Aziz & Arif 2005†) | Coarse stream `y_τ(e)` = trailing τ-mean (direct sums). Patterns use delay τ on `y_τ`. Using every position pools all τ offsets. `W_p = W − mτ + 1`. |

### 6.7 Dispersion entropy (`_entropy.py`)

Rostaghi & Azami (2016), with window-local μ_w and σ_w (ddof=0).

- `z_i = round(c·Φ((x_i−μ_w)/σ_w) + ½)` equals `1 + #{k: x_i ≥ μ_w + σ_w θ_k}`, with
  `θ_k = Φ^{-1}(k/c)` precomputed once by `_numpy_stats.norm_ppf` (≤ 1e-14 vs scipy).
- So the kernel needs **no erf at run time**. This matters because
  `_numpy_stats.norm_cdf` is `np.vectorize(math.erf)`, a Python-level loop.
- Patterns are `c^m` codes, with defaults c=6, m=2 (36 bins, `c^m ≪ W`). Entropy is
  normalised by `ln c^m`.
- Window-local normalisation makes this O(W) per row, an inherent cost of about 20–30 s
  per 25M rows (estimated).
- Windows are batched through `PanelWindows.iter_windows`. The numba kernel emits
  integer codes.

### 6.8 SampEn and RCMSE, at stride (`_entropy.py`)

**SampEn (Richman & Moorman 2000).**

- m=2, `r = 0.2·σ_w` (ddof=0, as antropy does), Chebyshev distance, N−m templates.
- `B1 = |x_i − x_j| ≤ r` is a `(R, W, W)` bool.
- `B_m` is the AND of m shifted diagonal bands. The counts are the upper triangle.
- Measured at 189 µs/window.
- The chunk R comes from `max_bytes` (65 KB per window per bool plane at W=256).

**RCMSE (Costa et al. 2002; Wu et al. 2014).**

- For τ = 1..5, pool A and B counts over the τ offsets of the coarse-grained window.
- `r = 0.15·σ_w` of the scale-1 window.
- Cost ≈ 2.3× SampEn.
- Output is per-scale entropies plus their sum (the complexity index).

**Stride.**

- Default `stride = W//16` (16 at W=256), with the as-of forward fill of §4.1.
- `stride=1` reproduces every-row bitwise (tested).
- Estimated budget: about 5 min single-threaded for SampEn on 25M rows.
- The numba backend emits integer match counts with an early exit and is bit-identical.

### 6.9 HVG time irreversibility (`_hvg.py`)

**Graph.** Horizontal visibility (Luque et al. 2009): `i<j` are linked iff
`x_k < min(x_i, x_j)` for every k between them. With `PGE(j)` the nearest earlier index
with `x ≥ x_j` and `NGE(i)` the next later index with `x ≥ x_i`, the degrees are:

- `out(i) = #{j: PGE(j)=i, x_j<x_i} + 1[NGE(i) ≤ t]`
- `in(j) = #{i: NGE(i)=j, x_i<x_j} + 1[PGE(j) ≥ a]`

Derivation:

- The HVG of a window is the **induced subgraph** of the series' HVG, because visibility
  depends only on intermediate points.
- PGE and NGE come from **binary lifting over a sparse table of range maxima**, with
  log2 W levels, restricted to spans < W. That is O(T log W) vectorised, entity-chunked
  for memory.
- Ties: an equal height blocks visibility. The strict criterion is tested against ts2vg.

**Feature.** Irreversibility is the divergence between `P_out` and `P_in` in the window.
Lacasa et al. (2012) use `KLD(P_out‖P_in)`, available as `measure="kld"` with degrees
truncated at `k_max=W/4`. The default is JSD, which is bounded. See Lacasa & Flanagan
(2015) for nonstationary series.

**Backends.**

- numba: exact incremental edge-set maintenance per entity, every row, O(1) amortised,
  with integer degree histograms, so it is bit-identical.
- numpy: O(W log W) per evaluated window, at stride `W//16`.

**What not to use.** Mean degree carries no information. For any series without ties,
`⟨k⟩ = 4(1 − 1/(2N))` (Núñez et al. 2012†). A geometric-tail λ fitted from the mean
degree would be constant. This identity is tested exactly on tie-free data; if it fails,
the reference is wrong and the assertion is dropped.

---

## 7. Dependencies

- **Runtime: zero new.**
  - No change to `[project.dependencies]`.
  - No new extra.
  - No new `_deps._MODULE_TO_EXTRA` row, so `tests/test_dependency_drift.py` is
    untouched.
  - numba stays behind `fast` via `have("numba")`, and it is never imported at module
    scope (`tests/test_import_hygiene.py`).
  - scipy is not used.
- **Test-only oracles** (`pytest.importorskip`, never imported by the package):

| Oracle | Licence | Used for |
|---|---|---|
| nolds | MIT† | `dfa` (`fit_exp="poly"`, `overlap=False` on the **reversed** window, since reversal maps start-anchored to end-anchored and leaves F unchanged), `hurst_rs(corrected=True, fit="poly")`, `sampen` with explicit tolerance |
| antropy | BSD-3-Clause | `perm_entropy`, `higuchi_fd`, `katz_fd`, `petrosian_fd`, `sample_entropy`, `detrended_fluctuation` |
| ordpy | MIT | `permutation_entropy` (incl. weighted), `complexity_entropy`, `fisher_shannon`, `missing_patterns` |
| MFDFA (Rydin Gorjão) | MIT† | `F_q(s)` per box segmentation (select the matching box ends) |
| PyWavelets | MIT | filter constants; interior `swt` coefficients after a pinned shift |
| EntropyHub | Apache-2.0† | `DispEn(Typex="ncdf")`, `cMSEn(Refined=True)` |
| ts2vg | MIT† | HVG directed degrees |
| **fathon** | **GPL-3.0†** | **Reference only, never imported**, even in tests. This follows `depend`'s GPL policy. |

- **CI.** The oracles are not in `dev`. Add an **advisory** `oracle-parity` CI job that
  installs them (orchestrator-owned `ci.yml`). The numpy oracles (direct `lstsq` per box,
  direct convolution with the level-j equivalent filter, brute-force window loops) run in
  every job.
- **Polars.** Only the idioms already green on 1.35 and 1.42 are used:
  - `struct(...).map_batches(fn, return_dtype=pl.Float64)` under `.over(entity)`, from
    `depend/_rolling.py`;
  - `map_batches(..., returns_scalar=True)` for scalar twins, from `.ts.lempel_ziv_complexity`.
- **Python 3.10.** No `match` on types, no PEP 695. mypy ratchet: new modules are fully
  annotated with `NDArray[np.float64]`. Pass `ruff check`, `ruff format --check`,
  `mypy panelary`.

---

## 8. Tests (`tests/test_complexity_*.py`)

`--strict-markers` is on. Only `slow` and `benchmark` are used, and both already exist.

### 8.1 Leak safety

**`test_complexity_leak_safety.py`**

- `assert_no_lookahead` and `assert_prefix_invariant(tol=0.0)` (bitwise) for every
  rolling op and frame function.
- Run on a hypothesis-generated ragged panel: 1–40 entities, lengths 0–600, interior
  NaNs, ties, a constant stretch.
- Cut points hit `L_j − 1`, `W − 1`, stride boundaries, and a block boundary of
  `_blocked_prefix`.

**`test_complexity_leaky_twins.py`**

- Every twin in the §4.1 table must **fail** the named verifier.
- Each is marked with the invariant it breaks, and each must never be "fixed" by
  loosening the test.

### 8.2 Closure, backends and chunking

**`test_complexity_closure.py`**

- `rolling_f(W)[t]` vs scalar twin on `x[t−W+1..t]`. Bitwise for the integer and
  shifted-slice paths; ≤ 1e-12 relative for blocked-sum paths.
- `features()` vs `.ts` ops, and `stride=1` vs the every-row path: bitwise.
- `n_jobs ∈ {1, 4}`, `chunk ∈ {1 entity, all}`: bitwise.
- `backend="numba"` vs `"numpy"`: bitwise for integer-statistic kernels. Skipped without
  numba.

### 8.3 Oracle parity

**`test_complexity_oracles.py`** runs the table in §7, with the tolerance stated per
statistic:

| Group | Tolerance |
|---|---|
| Pattern / count statistics (PE, WPE, C_JS, Fisher, missing, DispEn, HVG degrees, SampEn counts) | ≤ 1e-12, or exact integers |
| Higuchi / Katz / Petrosian | ≤ 1e-10 |
| DFA / R/S F-values | ≤ 1e-9 |
| Wavelet coefficients vs direct numpy convolution | ≤ 1e-12 |

Two further checks run in the same file:

- LA8/D4 constants: orthonormality to ≤ 1e-14.
- The existing `permutation_entropy` vs `rolling_permutation_entropy · ln m!`: ≤ 1e-12
  on tie-free data.

### 8.4 Known answers (`test_complexity_known_answers.py`)

Exact fGn comes from a test-local Davies–Harte generator. Monte-Carlo assertions are
`slow`.

**Hurst.** At W=512 and 300 replicates, for H ∈ {0.3, 0.5, 0.7, 0.9}, the mean estimate
lies within the tolerance (the true H inside an interval of stated half-width around the
mean):

| Estimator | Tolerance | Also asserted |
|---|---|---|
| Haar-WLS | ±0.015 | sd ≤ 0.04 |
| GHE q=2 | ±0.02 | — |
| DFA-1 | ±0.03 | — |
| R/S-AL | ±0.10 | Pins the **documented** bias sign: `mean(H_AL) < 0.7` at H=0.7. |

**Pinned skeptical findings** (these tests exist to keep the evidence, not to be
loosened):

- i.i.d. t₃ gives GHE q=1 bias > +0.03 and `ghe_multiscaling` (raw) > +0.03, while
  `ghe_multiscaling_excess` lies within ±0.02 of zero.
- Drift of 0.2σ gives a GHE q=2 bias > +0.05 and |Haar bias| < 0.02.

**Wavelets.**

- White noise: `E[ν²_j] = σ²/2^j`, within 3 SE.
- `x == c_J + Σ w_j` to ≤ 1e-12·|x|.
- Haar à trous `w_j` equals Haar MODWT `W_j` exactly.
- `y = β·m + ε` gives `β_j ≈ β` at every level.

**DCCA.** `y = x` gives ρ = 1 (≤ 1e-12); `y = −x` gives −1; `|ρ| ≤ 1` always;
independent series give `|mean ρ| < 0.05`.

**Fractal dimension.** fBm gives Higuchi FD ≈ 2 − H (±0.05 at W=512). Petrosian and
Katz are pinned to hand-computed values on small arrays.

**Ordinal.**

- White noise: PE ≥ 0.99 (m=3, W=512), missing-patterns excess ≈ 0, Fisher → 0, and the
  irreversibility is inside the IAAFT band.
- A monotone ramp gives PE = 0, C = 0, Fisher = 1.
- Sine: irreversibility ≈ 0. An asymmetric sawtooth has irreversibility near its maximum.
- Logistic map r=4 (10⁵ points):
  - exactly one missing pattern at m=3 (Amigó, Kocarev & Szczepanski 2006†; the
    implementer confirms the count by enumeration before pinning it);
  - C_JS above the fGn curve at matched entropy (Rosso et al. 2007);
  - irreversibility ≫ 0.

**HVG.** i.i.d. data gives `P_out(k) ≈ P_in(k) ≈ 2^{−k}` and irreversibility within the
surrogate band. Mean-degree identity: see §6.9.

**Dispersion entropy.** i.i.d. Gaussian gives normalised DispEn ≈ 1. A constant window
gives null (σ_w = 0), not NaN arithmetic.

### 8.5 Seeds (`test_complexity_seeds.py`)

- The same `(seed, entity, time)` gives identical shuffles across processes (checked by
  subprocess, because `hash()` salting must not matter).
- Reordering entities in the frame gives identical outputs.
- Truncating the panel gives identical outputs.
- A different entity gives a different permutation.
- `seed` changes the outputs.

### 8.6 Registry conformance (edits to `tests/test_registry_conformance.py`, orchestrator-owned)

- Add `_SYNTHESISED_ARGS` rows with window ≤ 16 (the probe entities have 12–24 rows),
  e.g. `"rolling_haar_hurst": ((), {"window": 16, "levels": 2})`.
- Add `_FRAME_OPS` adapters for the frame functions.
- Add `_NOT_EXERCISABLE` entries, with the minimum length stated, for any scalar twin that
  cannot produce a value below 24 observations.
- **Gate:** `test_window_evidence_coverage_is_accounted_for` must stay ≥ 80%
  demonstrated. Fifteen new window specs that are all-null on the probe panel would sink
  the whole registry's tally. So every scalar twin either works at n ≤ 18 (its grid is a
  function of n) or is registered as not exercisable, with its reason.
- `registry.audit()` stays clean (source, licence Apache-2.0 for clean-room work, no
  `T^2` cost hint in tier A/B).

### 8.7 Performance (`test_complexity_perf.py`, marker `benchmark`)

The §9 budgets, with a 2× regression allowance, on a 200 × 2,500 panel.

---

## 9. Benchmarks and performance budgets

`benchmarks/bench_complexity.py` (throughput) and `benchmarks/bench_complexity_estimators.py`
(the §1.3 bias/sd table, plus GPH / local-Whittle comparators from econ).

The target workload is 5,000 entities × 5,000 dates (25M rows), W = 256, float64, on a
laptop CPU (the §1.3 machine). "Measured" means ×10 from the 2.5M-row scratch runs.
"Estimate" means a count of operations, which must be measured before its milestone
closes.

| Family | Single-thread budget | Basis |
|---|---|---|
| Haar à trous J=6 + energies + Haar-WLS H + betas | ≤ 3 s | measured 0.11 s / 2.5M |
| Ordinal m=4: PE + C_JS + Fisher + missing + irreversibility | ≤ 6 s | measured 0.24 s / 2.5M for codes + PE |
| OTN m=4 (96 bins) | ≤ 15 s | estimate |
| GHE, τ ≤ 19, one q | ≤ 6 s | measured 0.46 s / 2.5M |
| DFA-1, 13 scales, exact | ≤ 20 s | measured 1.32 s / 2.5M (box stats) + rolling means |
| MF-DFA, 7 q (reusing DFA f²) | ≤ 30 s incl. DFA | estimate |
| R/S-AL, maximal overlap | ≤ 30 s | estimate (Σn ≈ 300 vector passes) |
| Higuchi k_max=10, Katz, Petrosian | ≤ 8 s | estimate |
| Dispersion entropy c=6, m=2 | ≤ 40 s | estimate (O(W) per row) |
| **`core` preset** | **≤ 15 s** | sum of the above |
| SampEn at stride 16 | ≤ 6 min (≤ 1 min at `n_jobs=8`) | measured 189 µs/window |
| HVG, numba every row / numpy stride 16 | ≤ 20 s / ≤ 2 min | estimate |
| Shuffle-corrected widths, stride 64, B=8 | ≤ 6 min (≤ 1 min at `n_jobs=8`) | estimate |

**Parallelism.** Target ≥ 4× on 8 performance cores for `n_jobs=8`. That must be
measured; if numpy's GIL release does not deliver it, fall back to `n_jobs` documented as
advisory.

**Memory.**

- Input and each output column are 200 MB at 25M rows. A 10-column `core` output is
  about 2 GB, which is inherent in float64 and stated in the docstring.
- Kernel scratch is bounded by `max_bytes` (default 512 MB) through entity chunking. The
  per-row costs:
  - DFA: 4 scratch streams per scale.
  - Ordinal cumulative histogram: 96 B/row at m=4, 480 B/row at m=5.
  - OTN: 384 B/row at m=4.
  - SampEn: 3 bool planes × W² per window in flight.
- `features()` raises before allocating when a single entity cannot fit (the shape
  `plan()` philosophy, without importing shape's classes).

**The `.ts` path under `.over`.** One `map_batches` call per entity. Measure the per-group
overhead at 5,000 entities in M1. If it is more than 20% of kernel time, the docstrings
point to `features()` as the throughput path.

---

## 10. Milestones

| M | Contents | Exit gate |
|---|---|---|
| **M0 — caller** | Agree the §12 caller with the truepoint owner. Engine test stub written on the truepoint side. | A named caller and a failing engine test exist. **Nothing else starts before this.** |
| **M1 — spine + wavelets** | Create `_internal/_wavelets.py` and `_internal/_blocked.py` (the move plus the `detect` alias). `_stream`, `_seeds`, `_design` (Haar offset table + generator script). Haar/D4/LA8 streams, `haar_decompose`, `modwt`, energies, wavelet variance, `haar_hurst`, `wavelet_beta`/`corr`. `.ts` ops + specs, `features()` skeleton with `preset="core"` (wavelet part), eager import. | §8.1–8.3 for wavelets; the Haar known-answer rows; the import-time budget; §9 wavelet budget; conformance ≥ 80%. |
| **M2 — ordinal** | Codes, LEX/REV tables, cumulative histograms plus the time-loop fallback, PE (rolling + registration of the existing scalar), WPE, C_JS, Fisher, missing (+ i.i.d. table), OTN, irreversibility, composite MPE. The `extract_features` opt-in hook + `features="complexity"` preset; `pn.features(method="complexity")`. | ordpy/antropy parity; logistic/sine/ramp known answers; `core` preset complete and ≤ 15 s. |
| **M3 — scaling** | Direct box statistics, DFA-1/2, DMA θ∈{0,½}, GHE (+ τ_max-averaged weights, `demean`), R/S-AL, DCCA, MF-DFA + Gaussian-null tables. `bench_complexity_estimators.py`. | The §1.3 table reproduced within MC noise; nolds/MFDFA parity; fGn known answers; pinned skeptical tests green. |
| **M4 — fractal + entropies** | Higuchi, Katz, Petrosian (van Herk/Gil-Werman), dispersion entropy (threshold form), SampEn/RCMSE at stride + as-of ffill. | antropy/EntropyHub parity; `stride=1` ≡ every-row; SampEn budget. |
| **M5 — opt-in heavy** | Shuffle-corrected widths (seeded), HVG (numpy stride + numba incremental), numba integer-statistic kernels (SampEn counts, HVG degrees, ordinal codes, dispersion codes), IAAFT/phase-randomised null bands as a frame-level diagnostic using `depend._null`. | Backend bitwise parity; seed tests; HVG vs ts2vg; §9 budgets for stride features. |
| **M6 — integration** | Docs page `docs/user-guide/complexity.md` (with the §1.3 table and the skeptical findings up front), `llms.txt` regeneration, CHANGELOG, mkdocs nav, advisory oracle CI job. Promote tier C → B for ops whose gates passed. | `make check` green; `registry.audit()` clean. |

M1 + M2 alone are shippable. They deliver the whole `core` preset and have the lowest
leak surface: integer or exact kernels, no RNG, no stride.

**File ownership.** Agents touch only their files.

| Owner | Files |
|---|---|
| A | `_internal/_wavelets.py`, `complexity/_wavelet.py` |
| B | `_internal/_blocked.py` (move), `_stream.py`, `_seeds.py`, `_design.py` |
| C | `_ordinal.py` |
| D | `_scaling.py` |
| E | `_fractal.py`, `_entropy.py` |
| F | `_hvg.py`, `_numba.py` |
| G | `_ts.py`, `_scalar.py`, `_features.py` |
| H | `tests/test_complexity_*.py` |
| Orchestrator only | `panelary/__init__.py`, `feature_extractors/_catalogue.py`, `_internal/_verbs.py`, `tests/test_registry_conformance.py`, `detect/_moments.py` (alias line), `ci.yml`, docs |

Sequencing: B blocks all others, and A blocks D's DCCA-vs-wavelet comparison only.

---

## 11. Risks and open questions (resolve with a measurement, not an opinion)

1. **Caller risk (the big one).** If the engine never needs series descriptors, this
   plan should stay in `todo/`. M0 exists so that the rule is not bypassed.
2. **Is any of this predictive?** The positive evidence (§1.2) is for wavelet
   decompositions of predictors and for market-level efficiency rankings, not for per-stock
   rolling Hurst as an alpha. Measure information content (rank IC vs forward realized
   volatility and |returns|, purged CV) in M3 before promoting anything beyond tier C in
   the docs' "recommended" list.
3. **Calibrated tables are fitted constants.** These are the Haar offsets, the Gaussian
   nulls, and the i.i.d. missing-pattern expectations. Regenerating them changes
   outputs, so they are versioned, generated by a checked-in seeded script, and any
   change needs a CHANGELOG line.
4. **Irregular sampling.** Every feature is defined on the entity's **row sequence**:
   lag `2^{j−1}` means rows, not days. Gaps are the user's to regularise, for example
   with `shape._tensor.build_tensor`'s forward-fill grid. Should `features()` refuse
   entities with irregular spacing, or warn? The proposal is to warn, with the fraction
   of irregular steps.
5. **Ties.** How much do zero returns in tick-discretised series bias PE and the HVG, on
   real data? Measure it with `tie_fraction`. If it is material, add `ties="jitter"`,
   seeded through `window_rng`.
6. **numba float paths (DFA box loop).** They are not bitwise-equal to numpy. Is the
   speed worth a second numerical path? Measure first; the default is numpy.
7. **`.ts.permutation_entropy` semantics.** It is raw nats, while the new rolling op is
   normalised. Adding a `normalize=` flag to the existing method is an API change and the
   feature_extractors owner's decision. Until then, document the difference.
8. **Existing `sample_entropy` template count (N−m+1).** Fixing it silently moves users'
   numbers. The owner decides whether a fix plus a CHANGELOG entry is warranted. This
   plan does not touch it.
9. **Shape W2 timing.** If shape's `_dwt.py` lands first and puts filters in `shape/`,
   M1 moves them to `_internal/_wavelets.py` and leaves an import behind, rather than
   duplicating them.
10. **MF-DFA negative q with W ≤ 512.** Even with nulls, is `Δh` informative at all
    beyond `ghe_multiscaling_excess`? If M3 shows no incremental information, demote
    MF-DFA to frame-level-only.
11. **Registry size.** This plan adds 12 rowwise, 15 window and 3 frame specs. If the
    conformance runtime or the `llms.txt` catalogue becomes unwieldy, keep only the
    `core` ops as `.ts` methods and reach the rest through `features()`.

---

## 12. Caller

**Status: none exists** (see the gate at the top of this file). Proposed, for the truepoint owner to accept or reject:

- **Where:** `truepoint/src/truepoint/quant/series_descriptors.py` (new). Not in
  `diagnose/`, which is firewalled from panelary.
- **What:** for each generated item in the computation and temporal families
  (`generate/family_computation.py`, e.g. the `return_window` template), compute
  point-in-time descriptors of the underlying public series as of the item's date:
  `haar_hurst`, `permutation_entropy`, `ordinal_complexity`, `higuchi_fd`.
- **Why:** they become stratification covariates for `inference/errorbars.py`. That
  answers a question the product leads with: *is the assessed AI silently wrong more
  often on high-entropy, anti-persistent windows?*
- **Engine test:** `truepoint/tests/quant/test_series_descriptors.py`. It asserts
  bitwise-identical descriptors across two runs (deterministic re-runs), and that a
  descriptor for as-of date d is unchanged when data after d is appended
  (`panelary.testing.assert_prefix_invariant`).
- **Sealing:** descriptors are computed inside the sealed pipeline and never published
  per item. A per-item descriptor could fingerprint which series and windows are in a
  sealed set (STRATEGY §21). Aggregates only.

---

## 13. References

Entries marked † were not verified in this session: bibliographic details or the precise
claim should be confirmed before they are cited in docs. Unmarked entries were verified
2026-09-29 or are standard.

- Alessio, Carbone, Castelli & Frappietro (2002). Second-order moving average and scaling of stochastic time series. *EPJ B* 27:197–200.†
- Amigó, Kocarev & Szczepanski (2006). Order patterns and chaos. *Phys. Lett. A* 355:27–31.†
- Amigó, Zambrano & Sanjuán (2007). True and false forbidden patterns in deterministic and random dynamics. *EPL* 79:50001.†
- Anis & Lloyd (1976). The expected value of the adjusted rescaled Hurst range of independent normal summands. *Biometrika* 63:111–116.
- Aziz & Arif (2005). Multiscale permutation entropy of physiological time series. *IEEE INMIC*.†
- Bandt & Pompe (2002). Permutation entropy. *PRL* 88:174102.
- Barunik & Kristoufek (2010). On Hurst exponent estimation under heavy-tailed distributions. *Physica A* 389(18):3844–3855.
- Barunik, Aste, Di Matteo & Liu (2012). Understanding the source of multifractality in financial markets. *Physica A* 391(17):4234–4251.
- Carbone, Castelli & Stanley (2004). Time-dependent Hurst exponent in financial time series. *Physica A* 344:267–271.†
- Costa, Goldberger & Peng (2002). Multiscale entropy analysis of complex physiologic time series. *PRL* 89:068102.
- Daubechies (1992). *Ten Lectures on Wavelets*. SIAM.
- Davies & Harte (1987). Tests for Hurst effect. *Biometrika* 74:95–101.
- Di Matteo, Aste & Dacorogna (2005). Long-term memories of developed and emerging markets. *J. Banking & Finance* 29:827–851.
- Di Matteo (2007). Multi-scaling in finance. *Quantitative Finance* 7:21–36.
- Esteller, Vachtsevanos, Echauz & Litt (2001). A comparison of waveform fractal dimension algorithms. *IEEE TCAS-I* 48:177–183.†
- Fadlallah, Chen, Keil & Príncipe (2013). Weighted-permutation entropy. *PRE* 87:022911.
- Faria & Verona (2018). Forecasting stock market returns by summing the frequency-decomposed parts. *J. Empirical Finance* 45:228–242. (The Haar-MODWT and recursive-estimation details are unverified.†)
- Gençay, Selçuk & Whitcher (2005). Multiscale systematic risk. *J. Int. Money & Finance* 24:55–70.†
- Gneiting & Schlather (2004). Stochastic models that separate fractal dimension and the Hurst effect. *SIAM Review* 46:269–282.
- Higuchi (1988). Approach to an irregular time series on the basis of the fractal theory. *Physica D* 31:277–283.
- Kantelhardt et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. *Physica A* 316:87–114.
- Katz (1988). Fractals and the analysis of waveforms. *Comput. Biol. Med.* 18:145–156.
- Kristoufek (2014). Measuring correlations between non-stationary series with DCCA coefficient. *Physica A* 402:291–298.†
- Lacasa, Nuñez, Roldán, Parrondo & Luque (2012). Time series irreversibility: a visibility graph approach. *EPJ B* 85:217.
- Lacasa & Flanagan (2015). Time reversibility from visibility graphs of nonstationary processes. *PRE* 92:022817. (The brief's "2016" is corrected here.)
- Lo (1991). Long-term memory in stock market prices. *Econometrica* 59:1279–1313.
- Luque, Lacasa, Ballesteros & Luque (2009). Horizontal visibility graphs: exact results for random time series. *PRE* 80:046103.
- McCullough, Small, Stemler & Iu (2015). Time lagged ordinal partition networks. *Chaos* 25:053101.†
- Morales, Di Matteo & Aste (2013). Non-stationary multifractality in stock returns. *Physica A* 392 (arXiv:1212.3195).†
- Murtagh, Starck & Renaud (2004). On neuro-wavelet modeling. *Decision Support Systems* 37:475–484.†
- Núñez, Lacasa, Luque & Gómez (2012). Visibility algorithms: a short review. InTech.†
- Olivares, Plastino & Rosso (2012). Contrasting chaos with noise via local versus global information quantifiers. *Phys. Lett. A* 376:1577–1583.†
- Peng et al. (1994). Mosaic organization of DNA nucleotides. *PRE* 49:1685.
- Percival & Walden (2000). *Wavelet Methods for Time Series Analysis*. CUP.
- Peters (1994). *Fractal Market Analysis*. Wiley.
- Petrosian (1995). Kolmogorov complexity of finite sequences and recognition of different preictal EEG patterns. *IEEE CBMS*.†
- Podobnik & Stanley (2008). Detrended cross-correlation analysis. *PRL* 100:084102.
- Renaud, Starck & Murtagh (2005). Wavelet-based combined signal filtering and prediction. *IEEE Trans. SMC-B* 35:1241–1251.†
- Richman & Moorman (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am. J. Physiol.* 278:H2039.
- Riedl, Müller & Wessel (2013). Practical considerations of permutation entropy. *EPJ ST* 222:249–262.†
- Rosso, Larrondo, Martín, Plastino & Fuentes (2007). Distinguishing noise from chaos. *PRL* 99:154102.
- Rostaghi & Azami (2016). Dispersion entropy. *IEEE Signal Proc. Lett.* 23:610–614.
- Shao, Gu, Jiang, Zhou & Sornette (2012). Comparing the performance of FA, DFA and DMA using different synthetic long-range correlated time series. *Sci. Rep.* 2:835.
- Small (2013). Complex networks from time series: capturing dynamics. *IEEE ISCAS*.†
- Weron (2002). Estimating long-range dependence: finite sample properties and confidence intervals. *Physica A* 312:285–299.†
- Wu, Wu, Lin, Wang & Lee (2014). Analysis of complex time series using refined composite multiscale entropy. *Phys. Lett. A* 378:1369–1374.†
- Zanin (2008). Forbidden patterns in financial time series. *Chaos* 18:013119.†
- Zanin, Rodríguez-González, Menasalvas Ruiz & Papo (2018). Assessing time series reversibility through permutation patterns. *Entropy* 20:665.†
- Zebende (2011). DCCA cross-correlation coefficient. *Physica A* 390:614–618.
- Zunino, Zanin, Tabak, Pérez & Rosso (2010). Complexity-entropy causality plane: a useful approach to quantify the stock market inefficiency. *Physica A* 389:1891–1901.
- Zunino, Olivares, Scholkmann & Rosso (2017). Permutation entropy based time series analysis: equalities in the input signal can lead to false conclusions. *Phys. Lett. A* 381:1883–1892.†
