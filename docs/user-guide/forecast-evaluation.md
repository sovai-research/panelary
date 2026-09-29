# Forecast evaluation & Sharpe inference

Statistics that decide whether a performance claim survives serial dependence, fat
tails, nested models and a search over many candidates. Everything is numpy + polars
(scipy appears only as a test oracle), vectorised over `M` models, and every test
returns the same [`EvaluationResult`](#one-result-type) so results from different
tests stack into one evidence table.

## One result type

```python
import panelary.validation as pv

res = pv.sharpe_ratio_test(strategy_returns, benchmark_returns)
res.to_frame()          # one row per model, fixed EVALUATION_SCHEMA
res.to_dict()           # JSON-safe payload
res["momentum"]         # the result restricted to one model
pv.evaluation_table([res, other_res])   # pl.concat of several tests
```

No test returns a bare p-value: `reference` always names the null distribution that
was actually used (`"N(0,1)"`, `"cbb-studentized(b=4,B=4999)"`, `"nct(T-1) [assumes
i.i.d. normal returns]"`, ...).

## Sharpe-ratio inference

### Why the textbook tests are not enough

Measured on Ledoit & Wolf's (2008, §4.2) data-generating processes at `T = 120`,
2,000 replications, nominal 5 % (`tests/test_validation_sharpe_calibration.py`):

| DGP (equal Sharpe ratios) | JKM | HAC (QS) | HAC (QS, prewhitened) | Boot-TS (`"auto"`) | Boot-TS (`"calibrate"`)† |
|---|---|---|---|---|---|
| Normal-IID | 5.7 % | 6.1 % | 6.4 % | 4.9 % | — |
| t6-IID | **8.6 %** | 6.5 % | 6.9 % | 4.9 % | 4.4 % |
| Normal-GARCH | 6.3 % | 6.7 % | 7.0 % | 5.4 % | — |
| t6-GARCH | 6.4 % | 6.2 % | 6.5 % | 5.1 % | — |
| Normal-VAR (AR φ = 0.2) | **10.1 %** | 8.6 % | 8.0 % | 6.3 % | 5.9 % |
| t6-VAR | **12.8 %** | 7.7 % | 7.1 % | 5.1 % | — |

† 1,000 replications, 100 pseudo-sequences and 199 bootstrap replicates per
calibration interval (the defaults are 1,000 and 499).

Jobson–Korkie with Memmel's correction (JKM) is valid only for i.i.d. normal returns;
it is kept as a labelled baseline and always carries a warning. HAC inference is
consistent but liberal in small samples. The studentized circular-block bootstrap
("Boot-TS") holds its size throughout. With `M = 50` null strategies against one
benchmark, the Romano–Wolf family-wise error rate of `sharpe_ratio_test` (default
settings) was 2 % at nominal 5 %; power at a per-period Sharpe difference of 0.3 was
58 %.

### API

```python
pv.sharpe_ratio_inference(r, method="hac")            # SE / CI / test of SR = SR0
pv.sharpe_ratio_test(r_strategies, r_benchmark)        # SR_i - SR_b, Boot-TS + Romano-Wolf
pv.sharpe_ratio_test(r, b, statistic="log_variance")   # equal volatility (LW 2011)
pv.sharpe_block_length(r, b)                           # LW Algorithm 3.1 calibration
pv.annualize_sharpe(sr, 12, autocorrelations=rho)      # Lo (2002) eta(q)
```

`method` selects the estimator of `Var(sqrt(T) SR)`; every one is the variance of the
influence series `psi_t = z_t - SR/2 (z_t^2 - 1)`:

| method | estimator | reference |
|---|---|---|
| `iid_normal` | `1 + SR^2 / 2` (Lo "IID"; JKM for differences) | `N(0,1)` |
| `iid` | `1 - g3 SR + (g4 - 1)/4 SR^2` (Mertens/Opdyke) | `N(0,1)` |
| `hac` | long-run variance of `psi`: prewhitened QS (default), QS, Parzen, or Bartlett with fixed lags (Lo "non-IID") | `N(0,1)` |
| `bootstrap` | studentized circular-block bootstrap (LW 2008 Boot-TS) | resampling |
| `exact_normal` | `sqrt(T) SR_hat ~ nct(T-1, sqrt(T) SR)`, CI by inversion | exact under i.i.d. normal |

Conventions: Sharpe ratios are per period with `ddof = 1`. `hac="qs_pw_vector"` in
`sharpe_ratio_test` reproduces LW's 4-vector prewhitened-QS `Psi` with the `T/(T-4)`
factor; the default scalar influence path applies `T/(T-2)`.

### How the bootstrap is computed

One matrix of block-start counts `N (B, T)` is drawn per call (one generator, before
any chunking). With circular block sums `Q` of the centred returns and centred squared
returns, every replicate moment and every entry of the Götze–Künsch matrix
`Psi* = (1/l) sum_j zeta_j zeta_j'` is a BLAS product `N @ (Q_a * Q_c)`, so no
resample is ever materialised. This is algebraically LW's procedure (agreement with the
literal construction: ~1e-15 relative on `s(Delta*)`).

- All `M` strategies share `N`, which is what Romano–Wolf needs for its joint null.
- A model's statistic and p-value are bitwise unchanged when other models are added or
  removed (for a fixed integer `block_length`; `"auto"` takes the maximum Politis–White
  length over columns and can change with the set of models).
- Models with different missing-value patterns are grouped by availability and tested
  on their own common dates; Romano–Wolf runs within the largest group, with a warning.
- `boundaries=` keeps blocks inside CV folds.

### Block length

`"auto"` (default) is Politis–White with the Patton–Politis–White correction on the
difference-influence series, maximised over columns. `"calibrate"` runs LW's Algorithm
3.1: fit a VAR(1), simulate `K` pseudo-sequences with stationary-bootstrapped residuals
(mean block 5), and pick the block length whose bootstrap-interval coverage is closest
to the nominal level. The `K` pseudo-sequences are `K` extra columns in the same
BLAS products. The default grid is LW's `(1, 2, 4, 6, 8, 10)` at `T = 120`, scaled by
`(T/120)^(1/3)`.

### Annualisation

`annualize_sharpe(sr, q, autocorrelations=rho)` applies Lo's
`eta(q) = q / sqrt(q + 2 sum_{k<q} (q - k) rho_k)`. Without `autocorrelations` it
returns `sqrt(q) * sr` and warns, because `sqrt(q)` overstates the annual Sharpe ratio
of positively autocorrelated returns.

## Forecast comparison beyond Diebold–Mariano

Every test takes a `(T, M)` matrix of forecasts or losses against one `(T,)` benchmark
and runs in one vectorised pass. The default HAC is Bartlett with `horizon - 1` lags
(the MA order of an optimal `h`-step error), never a function of `T`.

| Function | Question | Null distribution |
|---|---|---|
| `clark_west(y, bench, fc)` | does a model that *nests* the benchmark beat it? | N(0,1), one-sided |
| `oos_r2(y, fc)` | Campbell–Thompson R²_OS against the expanding historical mean (CW p-value) | N(0,1), one-sided |
| `mincer_zarnowitz(y, fc)` | unbiased and efficient forecasts, `(alpha, beta) = (0, 1)` | chi2(2) (HAC) or F(2, T-2) |
| `pesaran_timmermann(y, fc)` | is the direction right more often than chance? | N(0,1); hypergeometric with `exact=True`; HAC regression for `h > 1` |
| `encompassing_test(y, fa, fb)` | does forecast A encompass B? (HLN 1998) | t(T-1), one-sided |
| `giacomini_white(la, lb)` | conditional predictive ability | chi2(q) |
| `fluctuation_test(la, lb)` | relative performance over rolling windows (GR 2010) | shipped GR table |
| `one_time_reversal_test(la, lb)` | a single break in relative performance (GR 2010) | shipped GR table |
| `loss_panel(frame, ...)` | panel → `(T, M)` date series of cross-sectional mean losses | — |

**Nested models need Clark–West.** Under the null, the larger model estimates
parameters that are truly zero, so its squared error is inflated and DM is centred
below zero. In a rolling-OLS design (R = 120, P = 240, 1,000 replications) DM rejected
at most 2 % of the time at nominal 10 %, while CW stayed inside [3 %, 12 %]
(`tests/test_validation_forecast_compare.py`).

**R²_OS benchmark.** The default is the expanding mean of `y` known at `t - h`, with a
fixed `min_periods`; the full-sample mean uses future returns (a pinned test shows it
changes R²_OS and fails `assert_no_lookahead`). `benchmark="zero"` reproduces the
Gu–Kelly–Xiu convention of `embed.baseline_report`. The expanding R²_OS path and the
Goyal–Welch cumulative SSE-difference path are returned in `details` and are
prefix-invariant.

**Giacomini–White** is valid for forecasts from a rolling (fixed-size) estimation
window; `estimation_scheme="expanding"` attaches a warning.

### Giacomini–Rossi tests and their critical values

`fluctuation_test(mode="test")` standardises by the full-sample HAC (GR's test; its path
is not prefix-invariant and says so). `mode="monitor"` uses an expanding Bartlett HAC
with a fixed lag and a fixed integer window, so `path[t]` uses only rows up to `t`;
fractional windows and data-driven lags are rejected there. It is a diagnostic and makes
no anytime-validity claim; sequential monitoring belongs to the drift-monitoring
package.

The critical values are **simulated here, not copied**: 200,000 Brownian paths of
20,000 steps (seed 20260929), plus the same paths subsampled 4×, with a Richardson step
that removes the `O(1/sqrt(n))` bias of a discretely monitored supremum (Monte Carlo
SE ≈ 0.004 at 5 %). The generator is `benchmarks/gr_critical_values.py` and
`verify_gr_tables()` re-simulates a coarse table. Cross-checks:

- The Brownian-bridge part of the reversal statistic reproduces Andrews' sup-LM
  values (8.88 against 8.85 at 5 %, 15 % trimming).
- GR's published Table 1 could not be consulted. Rossi's Stata note quotes 2.89 at 5 %
  near `mu = 0.4`, while the continuous limit here is 2.955. That gap is what a supremum
  over a few hundred grid points gives: simulated values are 2.875 at P = 200 and 2.91
  at P = 500.
- The shipped values are continuous-limit values, so finite-P tests are slightly
  conservative. Measured sizes at P = 500 and nominal 5 %: 4.7 % (test mode), 4.9 %
  (monitor mode), and 4.6 % for the one-time reversal at P = 400.

| `mu` | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| `k_0.05` (two-sided) | 3.529 | 3.276 | 3.100 | 2.955 | 2.824 | 2.708 | 2.582 | 2.458 | 2.302 |
| `k_0.10` | 3.298 | 3.029 | 2.840 | 2.684 | 2.546 | 2.416 | 2.289 | 2.149 | 1.993 |

One-time reversal, 5 %: 10.50 (trim 0.15), 10.08 (trim 0.20).

## Performance

Reference machine: Apple Silicon, Python 3.13, NumPy 2.5 + Accelerate, single process,
shared with other jobs (so timings are upper bounds).

| Operation | Size | Budget | Measured |
|---|---|---|---|
| bootstrap means, count matrix | B=1000, T=5000, M=1000 | ≤ 50 ms | 43 ms |
| `sharpe_ratio_test`, Boot-TS, M vs benchmark | B=1000, T=5000, M=1000 | ≤ 1.0 s | 0.68–0.87 s |
| pairwise Boot-TS + calibration | T=120, K=1000, grid 6, B=499 | ≤ 1 s | 0.16 s |
| same | T=5000 | ≤ 10 s | 2.8–3.7 s |
| QS-PW column LRV | T=5000, M=1000 | ≤ 0.3 s | 0.19–0.23 s |
| Bartlett column LRV, L=9 | T=5000, M=1000 | ≤ 20 ms | 14–29 ms (load-dependent) |
| Romano–Wolf stepdown | S=1000, B=1000 | ≤ 20 ms | 5 ms (was 0.81 s) |
| CW / R²_OS / MZ / PT / ENC / GW | T=5000, M=1000 | ≤ 0.3 s each | 0.10 / 0.19 / 0.08 / 0.04 / 0.11 / 0.17 s |
| fluctuation (test / monitor), one-time reversal | P=5000, M=1000 | ≤ 0.3 s | 0.09 / 0.12 / 0.07 s |

## References

- Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe
  ratio. *J. Empirical Finance* 15, 850–859.
- Ledoit, O. & Wolf, M. (2011). Robust performances hypothesis testing with the
  variance. *Wilmott* 55, 86–89.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4).
- Andrews, D. W. K. (1991); Andrews, D. W. K. & Monahan, J. C. (1992). HAC estimation
  and prewhitening. *Econometrica* 59, 60.
- Romano, J. P. & Wolf, M. (2005). Stepwise multiple testing as formalized data
  snooping. *Econometrica* 73(4).
- Clark, T. E. & West, K. D. (2007). *J. Econometrics* 138(1); Campbell, J. Y. &
  Thompson, S. B. (2008). *RFS* 21(4); Harvey, Leybourne & Newbold (1998). *JBES*
  16(2); Pesaran & Timmermann (1992). *JBES* 10(4); Giacomini & White (2006).
  *Econometrica* 74(6); Giacomini & Rossi (2010). *J. Applied Econometrics* 25(4);
  Andrews, D. W. K. (1993). Tests for parameter instability and structural change
  with unknown change point. *Econometrica* 61(4).
