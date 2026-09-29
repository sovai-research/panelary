# Calibration diagnostics, risk backtests and multivariate scores

This page covers three parts of `panelary.validation` that judge a *forecast* rather than
a strategy:

| Question | Entry point |
| --- | --- |
| Are these probabilities (or mean forecasts) calibrated, with no bins to tune? | `corp_reliability` → `CORPResult` |
| Does one forecast beat another for *every* consistent score? | `murphy_diagram` |
| Is this VaR / ES forecast calibrated, with exact small-sample nulls? | `var_backtest`, `kupiec_test`, `christoffersen_test`, `dynamic_quantile_test`, `acerbi_szekely_test` |
| Which risk forecast is better? | `fz0_loss` (VaR+ES), `qlike_loss` (variance), fed to `diebold_mariano` / `model_confidence_set` |

Everything here is numpy + polars. The PAV kernel behind CORP has an optional numba
twin (the `fast` extra) that returns bitwise-identical results. Every test returns an
`EvaluationResult`, which is vectorised over models and stacks into one evidence table
with a fixed schema.

## CORP reliability diagrams

A binned reliability diagram (and the binned ECE computed from it) depends on the bin
count and the bin edges. Dimitriadis, Gneiting & Jordan (2021) show that this choice can
change the verdict. CORP replaces the bins with the pool-adjacent-violators (PAV) fit: the
isotonic regression of the outcome on the forecast. That fit has no tuning parameter.

```python
import numpy as np
from panelary.validation import corp_reliability

rng = np.random.default_rng(7)
p = rng.beta(0.7, 0.7, 2000)                 # forecast probabilities
y = (rng.random(2000) < p).astype(float)     # outcomes (calibrated by construction)

res = corp_reliability(p, y)                 # Brier score, consistency band at 90%
res.mcb, res.dsc, res.unc                    # miscalibration, discrimination, uncertainty
res.to_frame()                               # x, recalibrated, weight, band_low, band_high
```

The same data under binned ECE (equal-width bins, weighted mean |gap|). The forecast is
calibrated by construction, yet the verdict more than doubles as the bin count changes:

| bins | 5 | 10 | 15 | 20 | 50 |
| --- | --- | --- | --- | --- | --- |
| ECE | 0.0152 | 0.0170 | 0.0213 | 0.0207 | 0.0381 |

CORP gives one answer: `mcb = 0.00236`, `dsc = 0.1041`, `unc = 0.2497` (Brier). An
overconfident version of the same forecast, `0.5 + 1.3 (p − 0.5)` clipped to [0, 1], has
`mcb = 0.00629`, about 2.7× higher. Its ECE ranges from 0.0575 to 0.0740 depending on the
bins.

### The decomposition

`mean_score = MCB − DSC + UNC` holds exactly (to rounding):

- **MCB** is the mean score of the forecasts minus that of their PAV recalibration. It is
  ≥ 0, and it is 0 exactly when the forecasts are already isotonic-calibrated.
- **DSC** is the score of climatology (the constant `mean(y)`) minus that of the
  recalibration. It is ≥ 0.
- **UNC** is the score of climatology.

Both inequalities hold because PAV is optimal for every Bregman loss at once. Supported
scores are `score="brier"` or `score="log"` for binary outcomes, and squared error for
`functional="mean"`, the Gneiting–Resin extension to real-valued outcomes. A log score
is infinite when a forecast of exactly 0 or 1 meets the opposite outcome. It is reported
as `inf` with a warning, never clipped.

### Bands

- `band="consistency"` (binary outcomes) resamples `y* ~ Bernoulli(x)`, so it assumes the
  forecasts *are* calibrated. It refits PAV and takes pointwise quantiles. A CORP curve that
  leaves the band is evidence of miscalibration at that forecast value.
- `band="confidence"` resamples (forecast, outcome) pairs. It is the default for
  `functional="mean"`.
- `band="auto"` computes the band only while the `n_boot × G` replicate matrix has at most
  5·10⁷ cells, which is about 400 MB. Beyond that it returns no band and records a warning.
  The band is never silently subsampled.

!!! warning "The recalibrated values are a diagnostic, not a forecast"
    PAV is fit in-sample on the outcomes it is judged against. Feeding the recalibrated
    values back in as forecasts is look-ahead. `CORPResult` therefore has no `fit`,
    `predict` or `transform` method.

### Replacing a binned ECE

A caller that reports binned ECE today (for example a calibration metric over a model's
stated confidence and its correctness) swaps in one call. It passes the confidences as
`forecast` and the 0/1 correctness as `outcome`, then reads `res.mcb` in place of ECE. It
reports `dsc` and `unc` alongside, and `res.to_dict()` gives the JSON-safe evidence payload.
The consistency band replaces the per-bin error bars. MCB is on the scale of the chosen
score (Brier by default), not the |gap| scale of ECE, so thresholds must be re-derived,
not copied.

## Murphy diagrams

Every consistent scoring function for a quantile, an expectile (the mean is the ½-expectile)
or an event probability is a mixture of *elementary* scores `S_θ` (Ehm, Gneiting, Jordan &
Krüger, 2016). If model A's mean elementary score is at or below model B's at every
threshold θ, then A beats B under **every** consistent score, and no choice of loss can
reverse the ranking.

```python
from panelary.validation import murphy_diagram

md = murphy_diagram(np.column_stack([f_a, f_b]), y, functional="mean", names=["a", "b"])
wide = md.pivot(on="model", index="theta", values="mean_elementary_score")
dominates = bool((wide["a"] <= wide["b"]).all())
```

The curves are exact. Quantile curves are piecewise constant and the others piecewise
linear. The default grid is the set of breakpoints, and any user grid is exact at its
points. The integrals are the familiar scores: the pinball loss for quantiles, a quarter
of the squared error for the mean, and half the Brier score for probabilities.

## VaR and ES backtests

Every risk backtest here is *statistical evidence about the calibration of a risk
forecast; not a regulatory determination*. No traffic-light zone is emitted. Exact tail
probabilities are reported instead.

```python
import numpy as np
from panelary.validation import PredictiveSpec, var_backtest

# forecasts for row t were computed from data up to t - 1:
#   var_t = var_model(returns[:t]), e.g. a polars rolling op followed by .shift(1).over(entity)
table = var_backtest(
    returns, var99, es975,             # (T,) or (T, M); one column per model or entity
    level=0.01,                        # the tail probability alpha, not the confidence level
    var_convention="loss",             # REQUIRED: "loss" (VaR > 0) or "quantile" (VaR < 0)
    predictive=PredictiveSpec("normal", loc=mu_hat, scale=sigma_hat),  # enables Z2 p-values
)
```

`var_convention` has no default because sign bugs are the most common VaR-backtest error.
Under `"loss"`, VaR and ES are positive loss amounts and a hit is `y < −VaR`. Under
`"quantile"`, they are the negative return quantile and tail mean and a hit is `y < VaR`.
If most VaR values have the wrong sign for the declared convention, the call raises.

### Which null, and why (measured at T = 250)

| Test | Reference | Size at nominal 5 % |
| --- | --- | --- |
| Kupiec, asymptotic χ²₁ (VaR 1 %) | kept in `details` only | **0.096**: over-rejects about 2× |
| Kupiec, **exact binomial** (default) | `binomial-exact` | 0.040 (conservative, because the count is discrete) |
| Christoffersen `LR_ind`, asymptotic χ²₁ (VaR 1 %) | kept in `details` only | **0.014**: almost no power |
| Christoffersen, **exact Monte Carlo** (default, Dufour 2006) | `mc-exact(B=9999)` | 0.036 (independence), 0.029 (conditional coverage) |
| DQ, asymptotic χ²₆ (VaR 1 %) | `chi2(6)` + a warning | **0.093** (0.108 at T = 1000) |
| DQ, Monte Carlo null | `mc-exact(B=…)` | 0.051 |

The Monte Carlo nulls are exact because H0 (i.i.d. Bernoulli(α) hits) is fully
specified. The Christoffersen null is simulated once for each missing-value pattern and
shared by every model with that pattern. Each null has its own seeded generator keyed by
content, so adding or reordering models never changes another model's p-value. Ties in
the discrete statistics count against the model (conservative). `var_backtest` runs DQ
with the Monte Carlo null. The asymptotic version remains available from
`dynamic_quantile_test`, and every asymptotic result carries the oversize warning.

### Expected shortfall

`acerbi_szekely_test` computes Z1 (ES given the exceptions) and Z2 (VaR and ES jointly).
Both have expectation 0 under a correct forecast. Negative values mean risk was
underestimated. A p-value needs the forecast's predictive distribution, supplied as a
`PredictiveSpec`: `"normal"`, unit-variance `"student_t"`, or filtered historical
simulation (`"fhs"`, a pool of standardised residuals). Without one the result carries
the statistic and the published Z2 5 % threshold range (about −0.70 to −0.82; Acerbi &
Székely 2014), and no p-value.

`fz0_loss` is the Patton–Ziegel–Chen (2019) joint VaR/ES loss. Its differences are
homogeneous of degree 0, so comparisons are unit-free. `qlike_loss` is Patton's (2011)
volatility loss, and it is robust to noise in the variance proxy. Both return per-period
losses, which feed `diebold_mariano` or `model_confidence_set` for comparative
backtesting.

!!! warning "Alignment (trap T1)"
    The forecast for row `t` must use data up to `t − h`. A VaR computed with `y_t` in
    its window hides its own exceptions. The symptom is an implausibly **low** hit rate,
    which the exact Kupiec test flags as strongly as a high one.

**Panels.** Every test is column-wise, so N entities are N columns. Pass the per-entity
p-values to `benjamini_yekutieli`, which is valid under the arbitrary cross-sectional
dependence of a panel.

## Performance (measured)

Apple Silicon, Python 3.13, NumPy 2.5, polars 1.44, numba 0.67, single process. The
machine was shared with other jobs, so these are upper bounds.

| Operation | Size | Measured | Plan budget |
| --- | --- | --- | --- |
| `corp_reliability`, no band | n = 10⁶ binary, numba | 0.09 s | ≤ 0.3 s |
| same | pure Python PAV | 0.32 s | ≤ 0.3 s |
| consistency band | G = 10⁴, B = 1000, numba | 0.19 s | ≤ 0.1 s |
| same | pure Python PAV | 2.3 s | ≤ 2 s |
| `murphy_diagram` | n = 10⁶, M = 10, 1001-point grid | 1.2 s | ≤ 2 s |
| same | n = 10⁵, M = 10, default breakpoints (1.1·10⁷ rows) | 0.45 s | — |
| `kupiec_test`, exact | M = 1000, T = 5000 | 17 ms | ≤ 50 ms |
| `christoffersen_test`, MC null | B = 9999, T = 1000 | 70 ms | ≤ 0.1 s |
| `dynamic_quantile_test`, asymptotic | M = 1000, T = 5000 | 0.13 s | ≤ 0.2 s |
| same, MC null | M = 10, T = 1000, B = 999 | 0.36 s | — |
| `acerbi_szekely_test`, Z2 MC | M = 100, T = 1000, B = 2000 | 1.4 s | ≤ 3 s |
| `var_backtest` (all four tests) | M = 10, T = 1000 | 0.79 s | — |

## References

- Acerbi, C. & Székely, B. (2014). Back-testing expected shortfall. *Risk* 27(11), 76–81.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *IER* 39(4), 841–862.
- Dimitriadis, T., Gneiting, T. & Jordan, A. I. (2021). Stable reliability diagrams for
  probabilistic classifiers. *PNAS* 118(8), e2016191118.
- Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters. *J. Econometrics*
  133(2), 443–477.
- Ehm, W., Gneiting, T., Jordan, A. & Krüger, F. (2016). Of quantiles and expectiles:
  consistent scoring functions, Choquet representations and forecast rankings. *JRSS-B*
  78(3), 505–562.
- Engle, R. F. & Manganelli, S. (2004). CAViaR. *JBES* 22(4), 367–381.
- Gneiting, T. & Resin, J. (2023). Regression diagnostics meets forecast evaluation.
  *Electron. J. Statist.* 17, 3226–3286.
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models.
  *J. Derivatives* 3(2), 73–84.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies.
  *J. Econometrics* 160(1), 246–256.
- Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic semiparametric models for
  expected shortfall (and Value-at-Risk). *J. Econometrics* 211(2), 388–413.
