# Nonlinear dependence, with a p-value you can trust

`panelary.depend` measures, tests and screens **nonlinear dependence** in panel
data. Every dependence library ships a coefficient; the coefficient is the easy
half. This module is built around the other half: **the p-value being real**
on the data panels actually contain — persistent series, common shocks,
volatility clustering, unbalanced entities.

Pure `numpy` + `polars`. No SciPy, scikit-learn or compiled code in the import
path; external packages appear only as test oracles.

## The 60-second version

```python
import panelary as pn
import panelary.depend as dep

# One pair, with the null chosen for you (and reported).
dep.dependence(panel, "x", "y", method="xi")               # by entity, common-time null

# Rank features against a target -- leak-safely inside CV via ScreenSelector.
dep.feature_screen(panel, target="fwd_ret_20d", features=cols, method="dcor",
                   correction="benjamini_hochberg")

# Lag structure, family-wise error controlled across lags (Romano-Wolf).
dep.nonlinear_ccf(panel, "x", "y", max_lag=24, method="xi")

# p x p, batched, directed for xi.
dep.dependence_matrix(panel, cols, method="xi")

# Dependence as a feature: one value per row, trailing window, prefix-invariant.
panel.with_columns(pl.col("ret").ts.rolling_xi("mkt", window=60).over("ticker"))
```

Every analysis function returns a `pl.DataFrame` on **one fixed schema** —
`estimate, p_value, p_value_adj, method, estimator, null_method, n_resamples,
block_length, direction, lag, n_obs, n_entities, coverage, heterogeneity,
transform, approximate, seed, warnings` — so `pl.concat` across methods, nulls
and transforms just works, and no p-value ever travels without the null that
produced it.

## Why the null matters: measured, not asserted

Two **independent** AR(1) series, `n = 500`, tested against each other with
Chatterjee's ξ under its textbook i.i.d. asymptotic null `N(0, 2/5n)`
(2000 replications, nominal 5%; `tests/test_depend_calibration.py`):

| Persistence φ | Type-I error, i.i.d. null | Build-contract measurement |
| --- | --- | --- |
| 0.00 | 5.4% | 5.2% |
| 0.70 | 7.6% | 6.9% |
| 0.95 | **55.4%** | 56.3% |

φ = 0.95 is what every price level, valuation ratio and macro series looks
like. The test asserting the 55% exists to prove the bug is real and must never
be "fixed" by loosening it.

Two further facts from the same experiments shape the design:

- **Persistence biases the estimate, not only its variance.** For independent
  AR(1) series with φ = 0.95 and `n = 250`, the *mean* ξ is 0.089 and the mean
  bias-corrected distance correlation is 0.081 (both ≈ 0 for i.i.d. data;
  Spearman stays unbiased). Resampling nulls carry the same bias, so p-values
  stay honest — but raw estimates of features with different persistence are
  not comparable. `feature_screen` therefore ranks by adjusted p-value first.
- **An explicit i.i.d. null on such data warns.** `null="iid"` /
  `"asymptotic"` emits a `SerialDependenceWarning` (a `LeakageWarning`) when
  the lag-1 rank-autocorrelation pre-check exceeds 0.2 on both series.

## The null policy

`null="auto"` implements this table exactly (it is `depend.NULL_POLICY`, and
a test pins it):

| Statistic | serially independent | serially dependent | panel |
| --- | --- | --- | --- |
| `xi`, `spearman`, `pearson`, `kendall`, `hoeffding`, `gcmi` | asymptotic | block | common-time |
| `dcor` (p < 10) | gamma | block | common-time |
| `dcor` (p ≥ 10) | t | block | common-time |
| `hsic` | gamma | shift | common-time |
| `mi_ksg` | permutation | block | common-time |
| `transfer_entropy` | asymptotic (χ²) | block | common-time |
| `gcm` | asymptotic (N(0,1)) | HAC-studentised | common-time |

The serial branch is chosen by the lag-1 rank-autocorrelation pre-check at
`|ρ| > 0.2` — the minimum over the two series, because if either is serially
independent the i.i.d. null is exact. The choice is always reported in
`null_method`.

### Measured calibration of the serial nulls

Same design (independent AR(1), `n = 500`, nominal 5%, `B = 199`):

| Null | φ = 0 | φ = 0.7 | φ = 0.95 |
| --- | --- | --- | --- |
| `block` (ξ) | 4.5% | 4.8% | 6.0% |
| `shift` (ξ) | 4.4% | 4.8% | 5.8% |
| `block` (Spearman) | 4.0% | 4.7% | 5.4% |
| `shift` (Spearman) | 4.5% | 4.4% | 5.3% |
| `iaaft` (ξ, 200 reps, `B = 99`) | | | 6.0% |

(1000 replications; the slow calibration suite re-runs every cell at 2000 and
asserts `[3%, 8%]`.) Two design choices were forced by measurements, not taste:

- **The block length is sized for a dependence test, not for a variance.**
  Politis–White (`auto_block_length`) optimises the MSE of a long-run variance
  estimate; used for a block *permutation* test of ξ at φ = 0.95 its length (38)
  rejected at **8.2%**. Every block junction removes a fraction `k/L` of each
  cross term `ρx(k)ρy(k)` of the statistic's long-run variance, so the null is
  too narrow by `G/(L·D)`. `pair_block_length` returns the smallest `L` with
  that shortfall below 5% (never below Politis–White, never above `n/4`).
  Measured: `L = 60, 100, 125` → 7.2%, 6.6%, 6.2%.
- **A rotation null uses every rotation.** A randomisation test is exact over a
  *group*; excluding short shifts ("near-copies") makes the draw set a
  non-group and the test anti-conservative — 7.4% with shifts ≥ 38, **9.5%**
  with shifts ≥ 150. `null="shift"` draws from all `n − 1` rotations (and
  enumerates them when `n_resamples` exceeds that).

## Panels: within entity by default, common shocks in the null

Pooling a panel lets the longest entity dominate and mixes within-entity with
cross-sectional variation — Simpson's paradox with extra steps
(`tests/test_depend_api.py::test_simpson_pooled_vs_within` builds one where
pooled-raw Spearman is +0.8 and within-entity is −0.8). So:

- `by="entity"` (default): per-entity estimates on the shared date axis,
  aggregated — Fisher `arctanh` weighting by default (`how="precision"` for
  DerSimonian–Laird random effects, `"mean"`, `"median"`) — and **always** with
  a heterogeneity statistic (Higgins' `I²` in `heterogeneity`;
  `aggregate()` also returns Cochran's `Q`).
- `by="pooled"` with `demean in {"none", "entity", "time", "two-way"}`; the
  demeaning is always written into `transform`.
- `min_obs_for(method)` gates each entity (ξ 50, dCor 30, Hoeffding 50, GCMI
  `10·dim`, KSG `max(100, 10k + 1)`, tail `20/q`): below it, NaN and a warning,
  never a noisy number.

**The panel null is `"common-time"`**: whole date columns of `y` are permuted
jointly for every entity (in blocks when the series are persistent). Every
date's cross-section — every common shock — survives in the null. Measured on
20 entities × 250 dates whose `x` and `y` load on *independent* common factors
(Spearman, 300 replications, nominal 5%):

| Null | i.i.d. factors | factors with φ = 0.9 |
| --- | --- | --- |
| per-entity permutation | **55%** | **86%** |
| per-entity block | **55%** | **70%** |
| `common-time` | 6.7% | 6.7% |

A subtlety worth knowing: **ξ is naturally robust to this failure** — per-entity
permutation on the same i.i.d.-factor panel rejected at 5.0% — because ξ's
null fluctuation is driven by local neighbour pairs within each entity's own
`x`-order, which barely correlate across entities. Correlation-type statistics
(a global cross-product) inherit the common factor fully. `common-time` is the
default for every method because it is valid for all of them.

`null="entity"` re-pairs one entity's `y` with another's `x` on the same dates:
a different hypothesis (cross-sectional matching), offered side by side so the
choice is deliberate.

## Volatility clustering is not dependence

`devol=` reports every statistic twice — raw and after dividing both series by
a **causal** rolling standard deviation over the previous `devol` dates — as two
rows with different `transform`. What it does, measured on simulated
GARCH(1,1) returns (`a = 0.2`, `b = 0.75`, `n = 5000`, lag-1 dCor of `r_t` on
`r_{t-1}`):

- a 20-date window removes **75–85%** of the estimate; a 60-date window only
  **~30%** (the rolling SD lags the true volatility). Neither removes all of it.
- lag-1 ξ is a *weak* detector of volatility clustering in the first place: at
  `a = 0.1, b = 0.85` its raw z-statistic is indistinguishable from noise for
  `n ≤ 6000`; at `a = 0.2, n = 20 000` it is 1.6–3.3 raw and 0.2–1.2
  devolatilised — the build contract's "+1.91 raw, +0.21 devolatilised" is
  reproduced in direction, but it is not a robust single number.

## Lag scans

`lag_dependence`, `nonlinear_ccf` (lags `−max_lag..max_lag`; `k > 0` means `x`
leads) and `nonlinear_acf` (lags `1..max_lag`) draw the resamples **once** and
evaluate every lag on the same draw, so `p_value_adj` (mandatory) is
Romano–Wolf over the joint `(B × n_lags)` null, studentised per lag. Lags are
in units of the shared date axis. `nonlinear_acf` tests serial independence
(permutation null); `null="iaaft"` or `"phase"` keeps the linear
autocorrelation and asks whether there is nonlinear structure beyond it — the
diagnostic to run on model residuals. Block and shift nulls preserve exactly
what an ACF tests for and are refused.

## Screening features without leaking

A dependence **estimate** is fit-free. A dependence-based **selection** is not:
screening the full panel and then cross-validating on the survivors leaks the
test folds into the selection. `tests/test_depend_leakage.py` measures it on
pure noise (300 features, 5 entities × 200 dates, top-10 by Spearman): the
out-of-fold signed correlation of the selected features is **0.008** for the
honest selector and **0.082** for one that screened the full panel.

`ScreenSelector` is the leak-safe form (`panel_safe = leakage_safe = True`,
because it re-runs the screen inside `fit` on the training fold only):

```python
from panelary.depend import ScreenSelector
sel = ScreenSelector("fwd_ret_20d", features=cols, method="xi", k=20)
pipeline = pn.Pipeline([..., ("screen", sel), ...])
```

`feature_screen` uses the same resample indices and block length for every
feature, so `correction="romano_wolf"` gets a genuinely joint null
(`common-time` / `entity` nulls, or one complete series). It warns when
`n_resamples` is too small for any feature to survive the correction
(`n_resamples ≥ n_features / alpha`).

## Which statistic

| Want | Use | Note |
| --- | --- | --- |
| "Is `y` a function of `x`?" (directional) | `xi` | ξ(x, y) ≠ ξ(y, x), never symmetrised; weak on non-functional shapes (circles) |
| Any dependence, one pair, cheap | `hoeffding`, `dcor` | Hoeffding's D is consistent against all continuous alternatives |
| Multivariate `x` or `y` | `dcor` (`dcov2`), `hsic` | dCor tiled `O(n²)`, subsampled above `max_n` and recorded |
| Kernel dependence at scale | `hsic` | random Fourier features; see the `D` note below |
| MI you can condition on | `gcmi`, `gcmi_conditional` | **for one pair, GCMI is Spearman in a hat** — it cannot see a U-shape |
| Genuinely nonlinear MI | `mutual_information(estimator="ksg")` | `O(n²)`, capped |
| Conditional independence | `conditional_dependence(method="gcm")`, `foci` | GCM: two OLS fits + N(0,1); HAC when persistent |
| Directed information flow | `transfer_entropy` | **not causation**: a common driver inflates both directions |
| Tails | `tail_dependence`, `exceedance_corr` | ≈ `q` under independence, not 0; decays for a Gaussian copula |

With distance-induced kernels, HSIC **is** distance covariance (Sejdinovic et
al., 2013): one engine, two names — not two independent pieces of evidence.

`foci`'s stopping rule (stop when no candidate has a positive CODEC increment)
is a point-estimate rule without an error-rate guarantee at finite `n`; on a
test panel it admitted a pure-noise feature after the true one. Cap it with
`k`, and run it — like any selection — inside the training fold.

## Not shipped, on purpose

MGC/MGCX (class-D cost, narrow power gain over dCor; use `hyppo`), HHG and Ball
covariance (`O(n²)`, no fast path, thin evidence), MIC/TIC (the equitability
claim did not survive Simon–Tibshirani and Kinney–Atwal; lower power than dCor
at comparable cost), exact KCI (`O(n³)`), neural MI (needs torch), ppscore (its
CV is not panel-safe) and any Polars plugin (compiled; forbidden).

## Performance, measured

Apple Silicon, NumPy + Accelerate:

| Operation | Time | Contract |
| --- | --- | --- |
| rolling ξ, 200 entities × 2500 dates, `w = 60` | 0.3 s | 1.97 s |
| rolling dCor / tail (`w = 250`) / GCMI, same panel | 1.15 s / 0.62 s / 0.47 s | — |
| `sliding_ranks`, `T = 2500`, `w = 60` | 3.4 ms | 5.5 ms |
| RFF-HSIC, `n = 100 000`, `D = 256` / `D = 64` | 0.34 s / 0.09 s | 0.31 s / 0.08 s |
| `xi_matrix`, `n = 5000`, `p = 200` / `p = 500` | 0.37 s / 2.4 s | < 30 s |
| blocked dominance counts, `n = 100 000` | 0.12 s | — |
| exact univariate dCov, `n = 100 000` | 0.43 s | — |

## The build contract's open questions, answered by measurement

1. **Is the blocked `O(n^1.5)` dominance count fast enough at `n = 100 000`?**
   Yes: 0.10–0.12 s. The merge-sort upgrade was not written.
2. **What `D` does RFF-HSIC need for its p-value to be within 10% of exact
   HSIC at `n = 5000`?** More than the `D = 256` default. Against the exact
   Gaussian-kernel statistic with the same null-moment formula (13 weak-signal
   cases × 10 frequency draws), the median relative p-value error is 47% at
   `D = 16`, 30% at 64, 17% at 256, 12% at 512 and 7% at 1024; only `D = 1024`
   is within 10% for most draws (65%). This matters only for the gamma closed
   form: under a permutation / block / common-time null the RFF statistic is
   itself exactly calibrated, and `D` trades power for time.
3. **Does devolatilising change the *ranking* of features?** On
   `data/sp500.parquet` (503 tickers × 251 days; ten causal price features
   against the 5-day forward return), for ξ the rankings are nearly unchanged
   (Kendall τ between raw and devolatilised rankings 0.87, top-3 identical) and
   for dCor unchanged (τ 0.96) — the values move, not the order. For Spearman
   the ranking is reshuffled (τ −0.07; `vol20` falls from 1st to 10th). So
   `devol=` stays a documented option rather than a headline parameter, with
   the caveat that it matters for linear/rank statistics and for
   volatility-level features.
4. **Is `xi_matrix` at `p = 500`, `n = 5000` tolerable?** Yes, 2.4 s; the
   5000-row subsample default stays.
