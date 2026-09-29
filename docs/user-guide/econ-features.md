# Econometric Features

Econometrics has a large catalogue of statistics that make excellent panel
features — persistence, long memory, realized volatility, curve shape, liquidity,
tail risk — and almost all of them are computed, in the textbooks, on the *whole
sample*. That is exactly what you must not do inside a backtest.

`panelary.econ.features` reworks the useful ones so that every value is
either **trailing** (row `t` uses only rows `≤ t` of its own entity) or
**contemporaneous within a date** (a cross-section fitted at one date only), with
the handful of genuinely fitted parameters pushed into `PanelTransformer`s that
learn on train and freeze.

```python
from panelary.econ.features import (
    # per-row feature generators (trailing, causal)
    har_features, liquidity_features, evt_features,
    rolling_unit_root_features, rolling_long_memory_features,
    decompose_features, daily_realized_measures, nelson_siegel_factors,
    # single-series estimators
    zivot_andrews, local_whittle, gph, estimate_fractional_order,
    hill_index, gpd_fit, pot_var_es,
    # fitted transformers (train-only decisions, frozen at transform time)
    StationarityDifferencer, AutoFracDiff, HARModel, NelsonSiegel,
    CausalSeasonalDecomposer,
)
```

## The two surfaces of every estimator

Take ADF. There are two honest ways to use it and one dishonest one.

```python
# 1. As a per-row FEATURE: trailing window, causal, safe anywhere.
feats = rolling_unit_root_features(
    df, "px", entity="ticker", time="date", window=252, tests=("adf", "kpss")
)
# -> px_adf_stat, px_adf_pvalue, px_adf_lags, px_kpss_stat, px_kpss_pvalue

# 2. As a train-only PREPROCESSING decision, frozen into a transformer.
diff = StationarityDifferencer(["px"], max_order=1).fit(train_panel)
train_out = diff.transform(train_panel)
test_out = diff.transform(test_panel)      # same orders, no re-estimation
```

The dishonest third way — running ADF once on the full sample to decide how much
to difference, then cross-validating — is precisely what `StationarityDifferencer`
exists to prevent. The order it picks is a *fitted parameter*; it is chosen on the
rows handed to `fit` and reused verbatim.

!!! warning "`*_table` functions fit on everything you give them"
    `unit_root_table` and `long_memory_table` run on each entity's **entire**
    series. They are fine for exploration and for train-only preprocessing, but
    do not compute them over a panel that spans a fold boundary and then feed the
    result in as a feature. Use `rolling_unit_root_features` /
    `rolling_long_memory_features` for that.

## Persistence and regime features

Six tests, all pure NumPy, all returning a statistic, an approximate p-value and
the critical values used:

| Test | Null hypothesis | Rejects when | Use it for |
| --- | --- | --- | --- |
| `adf` | unit root | statistic is very negative | the default screen |
| `kpss` | **stationarity** | statistic is large | confirming ADF, not replacing it |
| `phillips_perron` | unit root | very negative | serial correlation without adding lags |
| `dfgls` | unit root | very negative | much more power near a unit root |
| `ng_perron` | unit root | `MZa`/`MZt` very negative | series with a large negative MA root |
| `zivot_andrews` | unit root, no break | very negative | series with a structural break |

ADF and KPSS point in opposite directions on purpose. A series that rejects ADF's
unit root *and* fails to reject KPSS's stationarity is stationary on both counts;
one that does neither is simply uninformative at that sample size.

`zivot_andrews` additionally returns the **break location**, which is a feature in
its own right:

```python
res = zivot_andrews(prices, regression="intercept")
res.stat, res.pvalue, res.break_index, res.break_fraction
```

## Long memory: making frac-diff data driven

The fixed-width fractional-differencing filter (`panelary._internal._ffd`) needs an
order `d`. Picking one number for a whole panel is arbitrary; tuning it by eye on
the full sample is a leak. Estimate it instead:

```python
from panelary.econ.features import local_whittle, gph

local_whittle(x).d     # Robinson (1995): lower variance, the default
gph(x).d               # Geweke-Porter-Hudak log-periodogram regression
```

Both are semiparametric estimators derived for stationary `d ∈ (-0.5, 0.5)`;
`estimate_fractional_order` handles the integrated region by differencing once and
adding one back, so a random walk correctly comes back as `d ≈ 1`.

`AutoFracDiff` wires the whole thing together:

```python
ffd = AutoFracDiff(["px"], d_min=0.0, d_max=1.0, round_to=0.05).fit(train_panel)
ffd.orders_          # {(entity, "px"): d} learned on train
ffd.transform(test_panel)   # frozen kernels applied causally per entity
```

### The divergence guard

`ffd_weights` **rejects `d < 0`**. Negative `d` is fractional *integration*:
its weights decay like `k**(d-1)`, which is not summable, so a truncated
fixed-width filter is dominated by its own truncation point and grows without
bound as the window lengthens. It also raises when a legal `d` is paired with a
threshold so small the kernel never decays within the 100 000-term safety cap —
without the guard that silently produces a kernel longer than any realistic
per-entity series, nulling every row. Passing an explicit `max_width` is an
informed opt-in to truncation and never raises.

## Realized volatility and HAR

```python
feats = har_features(df, entity="ticker", time="date", returns="ret", window=22)
# -> rv, bv, jump, rel_jump, har_d, har_w, har_m
```

`rv` is the trailing sum of squared returns; `bv` is bipower variation, which
multiplies *consecutive* absolute returns so a single jump enters linearly rather
than quadratically. Their difference isolates the jump component, and `rel_jump =
jump / rv` is the scale-free version.

`har_d` / `har_w` / `har_m` are Corsi's heterogeneous cascade — trailing means of
`RV` over 1, 5 and 22 periods. `HARModel` fits one OLS of `RV_{t+h}` on those
terms per entity on the training rows and emits the frozen-coefficient forecast:

```python
har = HARModel(returns="ret", horizon=1, window=22, log_target=True).fit(train_panel)
har.transform(test_panel)   # adds `har_forecast`
```

Entities with too few training rows, and entities never seen at fit time, fall
back to pooled coefficients estimated across all training entities.

If you have intraday data, aggregate it first — within-day aggregation is
contemporaneous, not forward looking:

```python
daily = daily_realized_measures(intraday, entity="ticker", date="date", returns="ret")
```

## Yield-curve factors

Per date, across maturities — so the factors at date `t` use only date `t`'s
yields:

```python
factors = nelson_siegel_factors(
    curves, date="date", maturity="months", yield_col="yield"
)
# -> ns_level, ns_slope, ns_curvature, ns_rmse, ns_nobs
```

`level` is `b0`, the asymptotic long rate; the reported `slope` is `-b1`, i.e. the
long-minus-short spread (the negative of Diebold & Li's slope factor, signed so an
upward-sloping curve is positive); `curvature` is `b2`.

The decay `λ` defaults to Diebold & Li's `0.0609` for maturities in **months**. If
you want it fitted, `NelsonSiegel(..., fit_lambda=True)` grid-searches it over the
**training dates** and freezes the result:

```python
ns = NelsonSiegel("yield", fit_lambda=True).fit(train_panel)  # entity=maturity, time=date
ns.lambda_
ns.transform(test_panel)   # broadcasts each date's factors onto its rows
```

## Liquidity

```python
liq = liquidity_features(
    df, entity="ticker", time="date",
    returns="ret", price="px", dollar_volume="dv", volume="vol",
    windows=(21, 63),
)
```

- **Amihud** `mean(|r| / dollar_volume)` — average price impact per unit traded.
- **Roll** `2 * sqrt(-Cov(Δp_t, Δp_{t-1}))` — the effective spread implied by
  bid-ask bounce. Only defined when that autocovariance is negative; positive
  autocovariance gives null (or `0.0` with `clip_positive=True`) rather than a
  fabricated number.
- **Amivest** `mean(volume / |r|)` and **turnover** round out the block.

Each is emitted for every window, alongside a trailing return volatility as the
natural control.

Range (OHLC) volatility, bid–ask spreads from OHLC bars (EDGE, Corwin–Schultz,
Abdi–Ranaldo) and the low-frequency proxies `price_impact`,
`pastor_stambaugh_gamma`, `zero_return_share` and `fht_spread` have their own page:
[OHLC Volatility & Liquidity](ohlc-volatility-and-liquidity.md).

## Tail risk

```python
evt = evt_features(df, "ret", entity="ticker", time="date", window=252, q=0.99)
# -> ret_hill_alpha, ret_gpd_xi, ret_gpd_sigma, ret_evt_var, ret_evt_es, ret_evt_n_exceed
```

`hill_index` returns the Pareto tail exponent `α` (larger = thinner tail);
`gpd_fit` fits a generalised Pareto to peaks over a threshold by
probability-weighted moments, and `pot_var_es` turns that into VaR and expected
shortfall. `tail="lower"` (the default) reads the loss tail of a returns series.

!!! note "Which tail estimator to trust"
    PWM is consistent and efficient for `xi < 0.5`, which covers essentially all
    financial return tails, and its shape estimate is bounded above by 1 by
    construction. For a genuinely infinite-mean tail (`α < 1`) the GPD shape will
    saturate near 1 — read `hill_index`, which has no such bound.

## Causal seasonal decomposition

```python
parts = decompose_features(
    df, "sales", entity="store", time="week", period=52, strength_window=104
)
# -> sales_trend, sales_seasonal, sales_remainder,
#    sales_strength_trend, sales_strength_seasonal
```

This is **not** STL. A classical STL smooths two-sidedly: its trend at time `t`
averages observations after `t`, so every historical value is rewritten when new
data arrives — one of the most common silent look-ahead leaks in time-series
feature engineering. Here the trend is a trailing moving average, the seasonal is
a trailing mean over the same seasonal phase (`t`, `t - period`, `t - 2·period`,
…), and the strengths are Wang-Smith-Hyndman variance ratios over a trailing
window. The cost is a phase lag of roughly half the trend window. That is a real
trade-off, and it is the honest one.

`CausalSeasonalDecomposer` is the same thing as a pipeline step.

## Proving it

`tests/test_econ_features_leakage.py` runs the same experiment on every generator
in this module: compute the features, append future rows to every entity,
recompute, and require that **no pre-existing row moved by a single ulp**. The
same file includes a control that computes a *centred* moving average on the same
data and asserts that it **fails** — so the passing tests are not vacuous.

To do the same on your own features:

```python
short = panel.filter(pl.col("date") < cutoff)
a = my_feature_fn(short)
b = my_feature_fn(panel)          # same rows plus future ones
# every value of `a` must appear unchanged in `b`
```
