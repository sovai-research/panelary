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
