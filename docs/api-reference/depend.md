# Dependence

Nonlinear dependence **measurement, screening and inference** for panel data,
written clean-room from the published papers and depending on nothing beyond
`numpy` and `polars`. The user guide ([Dependence](../user-guide/dependence.md))
has the measured calibration tables and the answers to the build contract's
open questions; this page is the reference.

Every analysis function returns a `pl.DataFrame` on the fixed schema
`RESULT_SCHEMA` (`estimate, p_value, p_value_adj, method, estimator,
null_method, n_resamples, block_length, direction, lag, n_obs, n_entities,
coverage, heterogeneity, transform, approximate, seed, warnings`), with
irrelevant fields null. No function returns a bare p-value.

## What's here

| Problem | Entry point |
| --- | --- |
| Test one pair on a panel, with the null chosen by policy and reported | `dependence` |
| The same for two numpy arrays | `independence_test` |
| Per-entity estimates; their aggregate with `I²` / Cochran's `Q` | `by_entity`, `aggregate` |
| One statistic on the pooled panel, with explicit demeaning | `pooled` |
| Lag structure with Romano–Wolf FWER across lags | `lag_dependence`, `nonlinear_ccf`, `nonlinear_acf`, `optimal_lag` |
| `p × p` matrix (directed for ξ), batched and subsampled | `dependence_matrix`, `matrix_values` |
| Distances for clustering; PSD repair | `to_distance`, `psd_repair` |
| Rank features against a target; leak-safe selection step | `feature_screen`, `ScreenSelector` |
| Mutual information (Gaussian copula or KSG) | `mutual_information`, `gcmi`, `mi_ksg`, `cmi_ksg` |
| Transfer entropy (not causation) | `transfer_entropy`, `transfer_entropy_array` |
| Conditional independence; conditional feature ordering | `conditional_dependence`, `gcm`, `codec`, `foci` |
| Kernel dependence with a frozen feature map | `hsic`, `RFFMap`, `rff`, `hsic_matrix` |
| Rolling dependence **features** (`.ts` namespace) | `rolling_xi`, `rolling_dcor`, `rolling_tail_dep`, `rolling_gcmi` |
| Null machinery | `NULL_POLICY`, `pair_block_length`, `auto_block_length`, `block_permutation_indices`, `circular_shift`, `iaaft`, `phase_randomise`, `common_time_indices`, `entity_permutation_indices`, `pvalue`, `gamma_pvalue`, `serial_dependence` |
| Rank primitives | `ranks`, `normal_scores`, `sliding_ranks`, `dominance_counts`, `pairwise_complete` |

## Statistics

| Function | Estimator | i.i.d. null | Reference |
| --- | --- | --- | --- |
| `xi` | Chatterjee's ξ, **directional**; ties in `x` by exact expectation (deterministic), ties in `y` by the ties-corrected formula | `N(0, 2/5n)`, refused (NaN) above 5% ties | Chatterjee (2021), *JASA* |
| `hoeffding_d` | Hoeffding's D, Hollander–Wolfe scale, ties weighted ½ / ¼ | Imhof inversion of the limit law, Chernoff bound in the far tail | Hoeffding (1948); Blum, Kiefer & Rosenblatt (1961) |
| `dcov2` / `dcor` | bias-corrected (U-centred) distance covariance; exact `O(n^1.5)` univariate path, tiled multivariate path | gamma matched to the exact permutation moments (`t` only for dim ≥ 10) | Székely & Rizzo (2014); Huo & Székely (2016) |
| `gcmi` | Gaussian-copula MI with Wishart bias correction, nats | χ² (Bartlett-corrected) | Ince et al. (2017) |
| `mi_ksg` / `cmi_ksg` | KSG algorithm 1; Frenzel–Pompe conditional | permutation | Kraskov et al. (2004); Frenzel & Pompe (2007) |
| `hsic` | random-Fourier-feature HSIC on normal scores | gamma (permutation mean, Gaussian-limit variance) | Gretton et al. (2008); Rahimi & Recht (2007) |
| `gcm` | generalised covariance measure, OLS residuals | `N(0,1)`, Newey–West when persistent | Shah & Peters (2020) |
| `codec` / `foci` | Azadkia–Chatterjee `T(Y, Z ∣ X)` (conditional on `X`), exact nearest neighbours | — | Azadkia & Chatterjee (2021) |
| `transfer_entropy_array` | Gaussian (Granger log-likelihood ratio), copula, or KSG | χ²(lag); source rotations for KSG | Barnett, Barrett & Seth (2009) |
| `tail_dependence` | co-exceedance count / `floor(q n)` | exact hypergeometric | — |

## Features in the `.ts` namespace

Importing `panelary.depend` adds four trailing-window operators to the `.ts`
expression namespace (additively — no existing method is replaced) and
registers a `FeatureSpec` for each (`safe_scope="rowwise"`, `panel_safe=True`,
`leakage_safe=True`, `license="Apache-2.0"`):

```python
pl.col("ret").ts.rolling_xi("mkt", window=60).over("ticker")      # xi(mkt -> ret)
pl.col("ret").ts.rolling_dcor("mkt", window=60).over("ticker")
pl.col("ret").ts.rolling_tail_dep("mkt", window=250, q=0.1).over("ticker")
pl.col("ret").ts.rolling_gcmi("vol", window=60).over("ticker")
pl.col("ret").ts.rolling_xi(window=60).over("ticker")             # xi(ret[t-1] -> ret[t])
```

Each is prefix-invariant by construction: the window is a fixed integer, ranks
are taken inside the window, and the tail threshold is a within-window rank
count. `tests/test_depend_prefix.py` checks all three bitwise, alongside a
deliberately leaky twin of each (window `n // 6`, global ranks, global
quantile) that must fail the same check.

::: panelary.depend
