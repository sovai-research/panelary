# Realized Measures and Rough Volatility

Intraday prices carry far more information about a day's variance than the
daily close does. `panelary.econ.features` turns them into daily features in
three layers:

1. `intraday_realized_measures`: one row per `(entity, session)`, holding
   variance, semivariance, quarticity, and jump-robust and noise-robust
   measures, all computed from that session's observations only.
2. `HARModel(spec=...)`: HAR forecasts built on those measures (HAR, HARQ,
   SHAR, HAR-CJ), fitted on training rows and frozen.
3. `rough_hurst` and `RFSVForecaster`: the roughness of the log-variance
   path, corrected for the measurement error of the proxy, and the
   rough-volatility forecast built on it.

```python
from panelary.econ.features import (
    intraday_realized_measures, HARModel, rough_hurst, RFSVForecaster,
)
```

## The one rule: a day's value exists from its close

Every measure of session `d` uses only session `d`'s ticks or bars. The output
row is stamped at the session close (column `time`), which is the earliest
moment the value is known. Use it at or after that stamp, for example with an
as-of join onto decision times. **Never join it back onto the same session's
intraday rows.** A 10:00 row that carries the day's realized variance is
carrying information from the rest of the day. For that reason the registry
scope is `"window"`, and the conformance suite checks that the per-row
broadcast is length-dependent.

Two stamping details are handled for you:

- **Open-stamped bars.** With `label="left"`, the stamp is moved to the close
  of the last bar: the last open plus the session's median bar spacing.
- **Overnight gaps.** The first return of a session is taken from the session's
  own first price, never from the previous close. Overnight variance is a
  separate quantity.

```python
daily = intraday_realized_measures(
    bars, entity="ticker", session="date", time="ts", price="close",
    measures=["rv_ss", "bv", "rs_pos", "rs_neg", "rq", "jump_sig", "log_rv_var"],
)
# one row per (ticker, date); `ts` is that session's close
```

## The measures

`M` is the number of returns in the session, and `r_i` is the i-th return.

| Measure | What it is | Robust to | Asymptotic variance / IQ |
|---|---|---|---|
| `rv` | `sum r^2` | nothing | 2 |
| `rv_ss` | mean of the `K` offset-subsampled RVs (`subsample=K`) | mild noise | none |
| `bv` | bipower variation | jumps | 2.61 |
| `medrv` | median RV (ADS 2012) | jumps, zero returns | 2.96 |
| `minrv` | minimum RV (ADS 2012) | jumps | 3.81 |
| `rs_pos`, `rs_neg` | realized semivariances | none | none |
| `sjv` | signed jump variation `rs_pos - rs_neg` | none | none |
| `rq`, `tpq`, `medrq` | realized, tri-power and median quarticity | TPQ and MedRQ: jumps | none |
| `jump_z` | Huang–Tauchen ratio-max jump statistic | none | none |
| `jump_sig`, `cont` | significant jump at `jump_alpha`, and `rv - jump_sig` | none | none |
| `log_rv_var` | delta-method variance of `log rv`: `(2/3) sum r^4 / rv^2` | none | none |
| `rk`, `tsrv`, `pav` | realized kernel, two-scales RV, pre-averaged RV | microstructure noise | none |

The asymptotic-variance column was **measured**, not copied. With 20 000
simulated sessions of 390 returns, `M Var / IQ` came out at 1.99, 2.62, 2.97
and 3.83. The MedRV, MinRV and MedRQ constants and `mu_{4/3}` were checked
against the order statistics of `|Z|` by quadrature to 1e-15.

`rv_ss` covers the `M - K + 1` complete `K`-step returns, so its expectation
is `(M - K + 1)/M` of the integrated variance: −1.0% for `K = 5` on
1-minute bars. `jump_sig` uses `jump_alpha = 0.999`. The simulated size of
the test at 0.95, 0.99 and 0.999 was 5.1%, 1.1% and 0.12%.

### When the prices are noisy

At high frequency the observed price is the efficient price plus
bid–ask bounce. RV then measures the noise: with 5-second returns and a
noise-to-signal ratio `xi^2 = omega^2 / IV` of 1e-3, RV is **ten times** the
integrated variance. Three estimators remove the noise:

- **`rk`**: the non-flat-top Parzen realized kernel (non-negative by
  construction) on end-point-jittered returns. The bandwidth
  `H = ceil(c* xi^{4/5} n^{3/5})` is chosen per session from that session's own
  noise and variance estimates (BNHLS 2009), or fixed with
  `kernel_bandwidth=H`. Diagnostics: `rk_h`, and `rk_capped` when the rule
  asks for more than `kernel_max_lags`.
- **`tsrv`**: Zhang, Mykland & Aït-Sahalia's two-scales RV, with the per-session
  optimal slow scale `K` (`tsrv_k`) unless you fix it.
- **`pav`**: pre-averaged RV (JLMPV 2009) with `g(x) = min(x, 1 - x)` and
  window `k = 2 ceil(theta sqrt(M) / 2)`, reported in `pav_k`.

Measured on constant volatility plus i.i.d. noise (seeded Monte Carlo in
`tests/test_realized_accuracy.py`), as the mean of each estimate divided by IV,
minus 1:

| Sampling | `xi^2` | `rv` | `rk` | `tsrv` | `pav` |
|---|---|---|---|---|---|
| 5 s (`M = 4680`) | 1e-4 | +94% | −0.3% | −0.5% | −0.1% |
| 5 s (`M = 4680`) | 1e-3 | +935% | +0.8% | −0.5% | −0.4% |
| 1 min (`M = 390`) | 1e-3 | +78% | +0.2% | −2.3% | −1.7% |

Three small biases remain, and all of them are largest on 1-minute bars:

- `pav` also removes `psi_1 / (2 k^2 psi_2)` of the IV along with the noise,
  about `6 / (theta^2 M)`.
- `tsrv` misses the `K - 1` edge returns, so its expectation without noise is
  `(M - K + 1)/(M + 1)` of the IV.
- A bandwidth estimated from the same session is correlated with that
  session's autocovariances. This costs −1.4% on clean 1-minute bars, and
  nothing with a fixed `kernel_bandwidth`.

!!! note "Raise `kernel_max_lags` for second or tick data"
    The BNHLS bandwidth grows like `n^{3/5}`. It is about 12–14 on 1-minute
    bars, which the default of 30 covers, and about 55–60 at 5 seconds. A capped
    bandwidth leaves noise in `rk` (+5.7% at 5 seconds with `xi^2 = 1e-3`),
    and `rk_capped` flags every session where that happened.

The BNHLS recipe estimates the noise variance on a grid of returns about 2
minutes apart and the integrated variance on a grid about 20 minutes apart.
Here those grids are defined as about 195 and 19.5 returns per session, their
values for a 390-minute session, so the rule needs no clock units.

## HAR, HARQ, SHAR and HAR-CJ

`HARModel` regresses `RV_{t+h}` on trailing daily, weekly and monthly terms,
with one OLS per entity fitted on the training rows only. New specifications
take the daily measures above:

```python
harq = HARModel(rv="rv", rq="rq", spec="harq", insanity_filter=True)
shar = HARModel(rv="rv", rs_pos="rs_pos", rs_neg="rs_neg", spec="shar")
cj = HARModel(rv="rv", jump="jump_sig", spec="har_cj")
forecasts = harq.fit(train).transform(test)
```

| `spec` | Regressors |
|---|---|
| `"har"` (default, Corsi 2009) | `1, RV_d, RV_w, RV_m` |
| `"harq"` (BPQ 2016) | adds `(RQ_t^{1/2} - m) RV_d`, so the daily coefficient shrinks when RV was measured imprecisely |
| `"shar"` (Patton & Sheppard 2015) | `RV_d` is split into `RS+_d, RS-_d` |
| `"har_cj"` (ABD 2007) | `C_d, C_w, C_m, J_d, J_w, J_m`, with `C = RV - J` |

Every constant that is fitted is fitted on the training fold and then frozen:

- HARQ's centre `m` is the training mean of `RQ^{1/2}` (`rq_center_`).
- The insanity filter replaces a forecast that falls outside the training
  target's `[min, max]` with the training mean (`bounds_`).

`spec="har"` is pinned bitwise to the pre-extension model.

## Rough volatility

Gatheral, Jaisson & Rosenbaum (2018) found that log volatility is *rough*: its
variogram `m(D) = E(x_{t+D} - x_t)^2` grows like `D^{2H}` with `H ~ 0.1`,
far below Brownian motion's 0.5. `rough_hurst` estimates `H` over a
trailing window from the OLS slope of `log m(D)` on `log D`.

A daily variance proxy is the true log variance plus measurement error, and
that error flattens the variogram. Uncorrected, the estimate is biased towards
roughness. `noise_var` is therefore **required**:

```python
daily = daily.with_columns(log_rv=pl.col("rv").log())
feats = rough_hurst(
    daily, entity="ticker", time="ts", log_variance="log_rv",
    noise_var="log_rv_var",      # per-day variance of log rv
    window=500,
)
```

- For `log rv`, pass the `log_rv_var` measure (or `log_rv_var_tpq` if the
  data has jumps).
- For the log of a range proxy, pass the variance of the log range term as a
  constant. For daily Parkinson ranges from 390-step paths it is 0.360.
- `0.0` states explicitly that the proxy is noise-free.

The table below is the plan's §1e table, reproduced by
`tests/test_rough_accuracy.py`. The underlying process is a fractional-Brownian
log variance with `nu = 0.3`, 1000 days and 200 paths; each cell is the mean of
`H` estimated over the whole window:

| true `H` | oracle | `log rv` (78 returns), raw | `log rv`, `noise_var="log_rv_var"` | Parkinson, raw | Parkinson, `noise_var=0.36` |
|---|---|---|---|---|---|
| 0.1 | 0.101 | 0.090 | 0.100 | **0.038** | 0.100 |
| 0.3 | 0.299 | 0.279 | 0.299 | **0.152** | 0.303 |
| 0.5 | 0.496 | 0.473 | 0.496 | **0.306** | 0.500 |

A smooth volatility read through daily ranges without the correction looks
rough.

!!! warning "A descriptive feature, not a verdict on the model"
    `rough_hurst` measures the roughness of the proxy path. Cont & Das (2024)
    show that roughness can be an estimation artefact even when volatility is
    smooth. Fukasawa, Takabatake & Westphal (2022) still find `H < 0.5`
    after correcting for measurement error. The correction here addresses
    independent proxy errors only.

### RFSV forecasts

`RFSVForecaster` estimates `H` on the training fold, pooled across entities
by default or per entity with `pooled=False`, and freezes it. At each row `t`
it emits the GJR forecast of `log sigma^2_{t+horizon}`, a normalised causal
weighted average of the last `n_lags` proxy values. The weights come from
`(u + D)^{-1} u^{-(H + 1/2)}`, integrated over each day. Rows with fewer than
`n_lags` days of history are null.

```python
rfsv = RFSVForecaster(log_variance="log_rv", noise_var="log_rv_var", horizon=1)
out = rfsv.fit(train).transform(panel)       # rfsv_forecast = E[log sigma^2_{t+1}]
```

In simulation (true `H = 0.1`, 40 paths of 3000 days), the fitted `H` was
0.102. Out of sample, the forecast error variance was 27% below the
last-value forecast and 12% below a 22-day mean.

## Performance

Measured 2026-09-29 on an Apple M5 Pro (15 threads, polars 1.44.2), with
eight other jobs sharing the machine. Scripts are in
`benchmarks/ohlc_vol/bench_intraday.py`.

| Workload | Measured | Budget |
|---|---|---|
| core battery, 19.7M one-minute rows | 0.96 s (≈4.8 s at 98M) | 8 s at 98M |
| full battery incl. `rk` (30 lags), `tsrv`, `pav`, 19.7M rows | 5.6 s (≈28 s at 98M) | 40 s at 98M |
| `rough_hurst`, 10 lags, 25M daily rows, constant noise | 4.0 s | 4 s (target) |
| `rough_hurst`, 10 lags, 25M daily rows, noise column | 7.0 s | none |

## References

Andersen & Bollerslev (1998); Barndorff-Nielsen & Shephard (2004, 2006);
Huang & Tauchen (2005); Andersen, Bollerslev & Diebold (2007);
Barndorff-Nielsen, Kinnebrock & Shephard (2010); Andersen, Dobrev &
Schaumburg (2012); Patton & Sheppard (2015); Liu, Patton & Sheppard (2015);
Bollerslev, Patton & Quaedvlieg (2016); Corsi (2009); Zhang, Mykland &
Aït-Sahalia (2005); Barndorff-Nielsen, Hansen, Lunde & Shephard (2008, 2009);
Jacod, Li, Mykland, Podolskij & Vetter (2009); Gatheral, Jaisson & Rosenbaum
(2018); Fukasawa, Takabatake & Westphal (2022); Cont & Das (2024).
