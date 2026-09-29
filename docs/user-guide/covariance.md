# Covariance and market state, as of every date

`panelary.covariance` estimates the entity × entity covariance (or correlation)
matrix of a return panel **as of each date**, from a trailing window, for
panels of 50 to 5,000 assets. On top of it sit the per-date market-state
features that are functions of that matrix: absorption ratio, eigenvalue
shares, effective rank, Marchenko–Pastur signal count, market-mode
localisation, average correlation, market loadings and turbulence.

Pure `numpy` + `polars`. scikit-learn and the authors' reference code appear
only as test oracles. The package is lazy: `import panelary` does not load it,
`pn.covariance` does.

## Why it exists

The textbook estimator is silently wrong exactly where equity panels live.
On a one-market, eight-sector factor population with Student-t(5) noise and a
252-day window, the true variance of the global-minimum-variance portfolio
built from each estimator, over the oracle minimum (median of 30 draws,
`tests/test_covariance_accuracy.py`):

| Estimator | N = 100 (q = 0.4) | N = 500 (q ≈ 2) |
|---|---|---|
| sample covariance, `np.linalg.solve` (raises nothing) | 1.6 | **~970** |
| sample covariance, `pinv` | 1.6 | 5.6 |
| Ledoit–Wolf, identity target | 1.46 | 2.46 |
| **QIS, correlation space (the default)** | **1.23** | **1.37** |

`estimate(..., method="sample").solve(...)` **raises** when the matrix is
singular instead of pseudo-inverting, and reports `cond() == inf`.

## The 60-second version

```python
import panelary as pn

cov = pn.covariance

# 1. Functional core: a (W, N) window in, an estimate out. No panel, no time.
est = cov.estimate(X, method="qis", space="correlation")
w = est.gmv_weights()          # Sigma^-1 1 / 1' Sigma^-1 1, O(N r), never N x N
est.inv_quad(x); est.logdet(); est.cond(); est.corr(); est.subset(idx)

# 2. As-of series over a panel: lazy, computed on demand, bit-for-bit prefix-safe.
series = cov.rolling(panel, returns="ret", window=252, method="qis",
                     schedule=cov.Schedule(every="1mo"))
series.at(date)                # the estimate from the last scheduled date <= date

# 3. Per-date market state (one row per date; broadcast=True joins onto rows).
state = cov.market_state(panel, returns="ret", window=252, stride=5,
                         features=("absorption_ratio", "ar_shift", "lambda1_share",
                                   "effective_rank", "mp_signal_count",
                                   "market_ipr", "avg_corr"),
                         group=None, broadcast=False)
turb = cov.turbulence(panel, returns="ret", window=252, refit="1mo", lag=1)
beta = cov.market_loading(panel, returns="ret", window=252)   # (entity, time)

# 4. Pipeline transformers (fit learns nothing; every value is as of its date)
cov.MarketState(returns="ret", window=252).fit_transform(panel)
cov.Turbulence(returns="ret", window=252).fit_transform(panel)
```

## Estimators

Every estimate is a `CovEstimate` in **diagonal-plus-low-rank** form,
`Σ = diag(s) (B diag(g) Bᵀ + E) diag(s)`, built from the `min(N, W)`-side Gram
matrix of the window: at `N = 3,000, W = 252` the 252 × 252 dual
eigendecomposition replaces a 3,000 × 3,000 one. Inverse, log-determinant,
condition number and Mahalanobis distances are exact in `O(N r)` (isotropic
`E`, orthonormal `B`) or `O(N r² + r³)` (Woodbury).

| `method=` | What it is | Oracle in the tests |
|---|---|---|
| `"qis"` (default) | Quadratic-Inverse Shrinkage, Ledoit & Wolf (2022) | authors' `QIS.py` fixtures, 1e-10 |
| `"lw2020"` | analytical nonlinear shrinkage, Ledoit & Wolf (2020) | dense port of the formulas, 1e-6 |
| `"lw_identity"` (`"lw"`) | linear shrinkage to `μI`, Ledoit & Wolf (2004a) | scikit-learn `LedoitWolf`, 1e-12 |
| `"oas"` | Chen et al. (2010), eq. 23 as published | scikit-learn `OAS` (which drops both `2/p` terms) |
| `"lw_diagonal"`, `"lw_constant_correlation"`, `"lw_single_index"` | linear shrinkage to a diagonal, constant-correlation or single-index target | authors' `covDiag` / `covCor` / `covMarket` fixtures, 1e-10 |
| `"mp_clip"`, `"mp_targeted"`, `"detone"` | Marchenko–Pastur denoising, targeted shrinkage, detoning (López de Prado 2020) | spiked-model properties |
| `"factor"` | `k` principal components + diagonal; `k` fixed, by eigenvalue ratio, Bai–Ng or MP count, **per window** | — |
| `"sample"` | the honest baseline; singular when `N > W` | — |

`space="correlation"` (the default) shrinks the correlation matrix and puts the
window standard deviations on the diagonal: at `q ≈ 2` the same QIS map scored
1.37 in correlation space and 1.87 in covariance space.

Numerical notes, all measured and tested:

- **LW 2020's Hilbert transform cancels catastrophically at large `|x|`.**
  Evaluated literally, a 1e-14 relative change in one eigenvalue moved the
  largest shrunk eigenvalue by up to 8.6e-7. Summing the transform as an exact
  series in `√5 / x` beyond `|x| = 5` brings that to ~7e-14, on par with QIS.
- **Tracy–Widom edge.** Johnstone's centring with `n − 1`: measured size 4.2%
  at the nominal 5% (`p = 400`, `n = 999`, 1,000 white-noise windows).
- **Corrected two-pass centring.** Exact to 1e-12 against rational arithmetic
  at a 1e8 offset.
- **No estimator dominates.** On a smooth log-uniform spectrum at `q = 2` the
  linear LW estimator tracked the oracle eigenvalues slightly better than QIS
  (0.085 vs 0.111 mean relative error); on the clustered spectrum of Ledoit &
  Wolf's own designs QIS wins at both `q = 0.5` and `q = 2`.

## The as-of engine and its schedule

For each evaluated date `t`:

1. The **universe** is the entities observed at `t` with at least
   `ceil(min_coverage · W)` observations in `(t − W, t]` (default 95%).
   Nothing about an entity's future is consulted.
2. The window is **compacted** to those entities before any matrix product, so
   an entity that lists after `t` cannot change a single bit at `t`.
3. Missing cells inside the window are zero after demeaning (the estimate stays
   PSD; correlations attenuate by about `√(fᵢ fⱼ)`, at most 5% at the default
   coverage).

`Schedule` (shared with the rest of the package, `panelary.core._schedule`)
says which dates are evaluated:

| `every=` | Grid |
|---|---|
| `"every"` / `1` | every date |
| `k` | positions `i ≡ 0 (mod k)` counted from the panel's first date |
| `"1w"`, `"1mo"`, `"1q"`, … | the **first** date of each calendar period |
| `"last"`, `"end"` | **refused** (trap T6): knowing `t` is last needs `t + 1` |

Values are carried forward to the dates between grid points (as-of forward
fill); every output row carries `asof_date` so staleness is visible.

## Market-state features

| Feature | Definition |
|---|---|
| `absorption_ratio` | `Σ_{i≤k} λᵢ / Σ λᵢ`, `k = ⌈0.2 · min(N_t, n_eff)⌉` or a fixed `ar_k` (recommended when `N_t` varies); emits `ar_k` |
| `ar_shift` | `(mean₁₅(AR) − mean₂₅₂(AR)) / sd₂₅₂(AR)`, trailing windows on the AR series |
| `lambda1_share` | `λ₁ / tr` |
| `effective_rank`, `eigen_entropy` | `exp(H)` and `H / log m`, `H` the entropy of `λ / Σλ` |
| `participation_ratio` | `tr² / ‖S‖²_F`, from the Gram (no eigensolve) |
| `mp_signal_count` (`mp_sigma2`, `mp_edge`) | eigenvalues above the Tracy–Widom edge, noise level by fixed point |
| `market_ipr` | `Σ v₁ᵢ⁴`; null when `λ₁ / λ₂ < min_gap` (1.05) |
| `avg_corr` | mean off-diagonal correlation, by the identity `(‖Z1‖²/n_eff − tr) / (N(N−1))` |

Diagnostics on every row: `n_entities`, `coverage`, `q`, `n_eff`,
`lambda_gap`, `asof_date`. `group="sector"` runs each group on its own members,
partitioned by the **date-t** group value (a later reclassification never
changes the past, trap T12).

`turbulence` is `(r_t − μ_s)ᵀ Σ_s⁻¹ (r_t − μ_s) / N_t` with `s` the latest refit
`≤ t − lag`, `lag ≥ 1`: today's return never sits inside its own covariance
(trap T2), and there is no full-sample mean or covariance (trap T1, the
published definition). Entities unobserved at `t` are marginalised out
exactly.

## Leak safety

Every function is bit-for-bit prefix-invariant (`assert_prefix_invariant`
with `tol=0`) and passes `assert_no_lookahead`, for daily, strided, weekly and
monthly grids, with and without groups, on `synth` panels with entities that
list and delist mid-sample (`tests/test_covariance_prefix.py`). Each named
trap has a test that builds the leaky variant and shows the instrument
catching it (`tests/test_covariance_traps.py`): full-sample z-scores (T3),
panel-sized parameters (T4), survivorship and forward-filled delisted returns
(T5), end-anchored grids (T6), full-sample intensities (T9), snapshot group
labels (T12), eigenvector sign flips (T13), pseudo-inverted singular
estimates (T14), full-sample turbulence (T1) and `r_t` inside its own
covariance (T2).

## Measured performance

`benchmarks/bench_covariance.py`, `T = 5,000` dates, `W = 252`, 4 BLAS threads
(`VECLIB_MAXIMUM_THREADS=4`), Apple M5 Pro with Accelerate. **Measured while
eight other build agents shared the 15-core machine** (load average 25–35;
calibration GEMM 2,000 × 2,000: 97 ms), so every number is inflated relative
to an idle machine -- the same 252 × 252 `eigvalsh` took 2.2–6.3 ms here
against 1.5 ms in the plan's idle measurement. Budgets are the plan's
section 8.1 figures.

| Workload | Measured | Budget |
|---|---|---|
| `estimate(qis)`, N = 3,000 | 14.2 ms | 8 ms |
| `estimate(qis)`, N = 500 | 6.0 ms | — |
| `market_state` spectrum set, daily, N = 500 | 19.0 s | 12 s |
| same, stride 5 | 4.8 s (25% of daily) | ≤ 30% of daily |
| `market_state` spectrum set, daily, N = 3,000 | 41.8 s | 20 s |
| same, stride 5 | 9.5 s (23% of daily) | ≤ 30% of daily |
| `turbulence`, refit every 21, daily scoring, N = 500 | 2.8 s | 5 s |
| `turbulence`, refit every 21, daily scoring, N = 3,000 | 6.8 s | 5 s |
| `build_tensor` pivot, 15M cells (N = 3,000) | 1.1–3.1 s | 2 s |
| peak numpy memory, N = 3,000 | 0.35 GiB | 1 GiB |
| `estimate(lw_identity)` vs scikit-learn `LedoitWolf().fit`, N = 3,000 | 12.5 ms vs 12.2 s (×981) | ≥ ×1,000 |

Per evaluated date at N = 3,000 the time is LAPACK/BLAS: window gather and
standardisation, the 252 × 3,000 Gram product and the 252 × 252 `eigvalsh`.
The pivot's cross-join grid is the one piece outside this package; if it stays
above 2 s at 15M cells, the fix is a scatter path inside `build_tensor`
(plan question Q11), not a second pivot here.

## Not in this release

Cross-sectional distribution features (`.xs.dispersion`, `.xs.tail_index`,
Kelly–Jiang, W₁ between cross-sections, average skewness) and the
`avg_correlation` / `common_idio_vol` frame ops are built separately. EWMA,
pairwise covariance with PSD repair, Gerber, the online co-moment API and the
walk-forward accuracy harness are milestone M5; RIE, regularised Tyler and a
scipy top-k backend are gated extras (M6).
