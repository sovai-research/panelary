# OHLC Volatility and Liquidity

A daily bar's open, high, low and close carry far more information than its close
alone. `panelary.econ.features` turns them into **volatility** estimates that are
five to eight times as precise as close-to-close, **bid–ask spread** estimates for
when quotes are absent, and **low-frequency liquidity proxies** — every one a
trailing-window statistic computed per entity with native Polars rolling moments,
so the value at row `t` uses only bars `≤ t` of its own entity.

```python
from panelary.econ.features import range_volatility, ohlc_variance_terms

vol = range_volatility(
    bars, entity="ticker", time="date",          # open/high/low/close columns
    method="yang_zhang", window=21, periods_per_year=252,
)
# -> vol_yang_zhang_21, vol_yang_zhang_21_n_valid
```

All functions take a `DataFrame` or `LazyFrame` and return a `DataFrame` sorted by
`(entity, time)` with the new columns appended. Windows count rows (bars), not
calendar time.

## Range volatility

Natural-log prices; `u = H − O`, `d = L − O`, `c = C − O`, overnight
`o = O_t − C_{t−1}`.

| `method` | per-bar term / window formula | efficiency vs close-to-close | drift | opening jump |
| --- | --- | --- | --- | --- |
| `close_to_close` | sample variance of `C_t − C_{t−1}` | 1 | demeaned | included |
| `parkinson` | `(H − L)² / (4 ln 2)` | 5.2 | biased up | missed |
| `garman_klass` | `½(H − L)² − (2 ln 2 − 1) c²` | 7.4 | biased up | missed |
| `rogers_satchell` | `u(u − c) + d(d − c)` | ≈ 6 | **unbiased** | missed |
| `gk_overnight` | window mean of `o²` + GK term | ≈ 8 (n = 21) | biased | included |
| `yang_zhang` (default) | `V_o + k·V_c + (1 − k)·V_RS` | ≈ 7–8 (n = 21) | **unbiased** | **consistent** |

Yang–Zhang's weight is `k = (α − 1) / (α + (n + 1)/(n − 1))` with `α = 1.34` and `n`
the row's **own trailing count of valid bars** — never the series length, which
would make every value depend on how much data follows it.

`ohlc_variance_terms` returns the per-bar pieces (`ohlc_o`, `ohlc_u`, `ohlc_d`,
`ohlc_c`, `ohlc_parkinson`, `ohlc_garman_klass`, `ohlc_rogers_satchell`,
`ohlc_gk_overnight`), so an exponentially weighted estimator is one line:

```python
terms = ohlc_variance_terms(bars, entity="ticker", time="date")
ewm_rs = terms.with_columns(
    pl.col("ohlc_rogers_satchell").ewm_mean(half_life=10).over("ticker").sqrt()
)
```

### What each choice costs you — measured

Monte Carlo through the public function, 3000 independent 21-bar windows per row,
bar variance 1; `M` intrabar returns form each high and low; `f` is the overnight
share of the variance; `μ` the drift in bar standard deviations; "SV" a
Heston-type stochastic variance. Cells are **bias / MSE-efficiency** relative to
the 21-bar close-to-close sample variance (`benchmarks/ohlc_vol/range_mc.py`,
2026-09-29):

| M | f | μ | SV | yang_zhang | gk_overnight | rogers_satchell | garman_klass | parkinson |
| ---: | ---: | ---: | :-: | --- | --- | --- | --- | --- |
| 78 | 0 | 0 | n | −16.8% / 2.47 | −19.2% / 2.08 | −19.4% / 1.91 | −19.2% / 2.08 | −14.1% / 2.67 |
| 78 | 0 | 0.1 | n | −16.8% / 2.57 | −19.1% / 2.17 | −19.5% / 1.98 | −19.1% / 2.17 | −13.6% / 2.82 |
| 78 | 0.25 | 0 | n | −12.5% / 3.64 | −14.4% / 3.17 | −39.6% / 0.62 | −39.4% / 0.64 | −35.3% / 0.76 |
| 78 | 0.25 | 0.1 | n | −12.3% / 3.64 | −14.1% / 3.21 | −39.3% / 0.62 | −39.1% / 0.64 | −34.8% / 0.77 |
| 390 | 0 | 0 | n | −7.4% / 5.32 | −8.8% / 5.03 | −8.8% / 4.27 | −8.8% / 5.03 | −6.1% / 4.45 |
| 390 | 0 | 0.1 | n | −7.8% / 4.94 | −8.8% / 4.76 | −9.2% / 3.99 | −8.8% / 4.76 | −5.9% / 4.37 |
| 390 | 0.25 | 0 | n | −5.6% / 6.06 | −6.5% / 5.98 | −31.8% / 0.91 | −31.6% / 0.93 | −29.5% / 1.02 |
| 390 | 0.25 | 0.1 | n | −6.1% / 5.78 | −6.8% / 5.77 | −31.9% / 0.90 | −31.7% / 0.92 | −29.7% / 1.00 |
| 5000 | 0 | 0 | n | −2.1% / 7.29 | −2.5% / 7.68 | −2.5% / 6.17 | −2.5% / 7.68 | −1.8% / 5.19 |
| 5000 | 0 | 0.1 | n | −2.2% / 6.87 | −2.4% / 7.30 | −2.6% / 5.96 | −2.4% / 7.30 | −1.4% / 4.95 |
| 5000 | 0.25 | 0 | n | −1.9% / 6.54 | −2.2% / 6.87 | −27.1% / 1.16 | −27.1% / 1.18 | −26.5% / 1.17 |
| 5000 | 0.25 | 0.1 | n | −1.5% / 7.14 | −1.6% / 7.42 | −27.0% / 1.22 | −26.7% / 1.27 | −25.8% / 1.28 |
| 390 | 0.25 | 0.05 | y | −5.8% / 6.40 | −6.6% / 6.29 | −32.2% / 1.02 | −32.0% / 1.05 | −29.9% / 1.13 |

Three things to take from it:

* **Discrete monitoring biases every range estimator down**, by roughly
  `1.7–1.9/√M` (GK, RS) and `1.2–1.3/√M` (Parkinson): −9% on one-minute highs and
  lows, −20% on five-minute ones.
* **An intraday-only estimator silently drops the overnight variance.** With a
  quarter of it overnight, RS/GK/Parkinson read 25–40% low and lose to
  close-to-close outright. Use `yang_zhang` or `gk_overnight` on daily bars.
* **The default is `yang_zhang`, decided by this table.** The plan's rule was to
  switch to `gk_overnight` only if its MSE were lower by more than 5% in *every*
  regime. It is lower by more than 5% in 2 of the 13 (fine sampling, no overnight
  variance) and higher or within 5% in the other 11 — because Yang–Zhang's
  sample-variance components carry no discreteness bias. The default stays
  `yang_zhang`, and will not change after release.

The often-quoted "Yang–Zhang is 14× close-to-close" is an upper bound from the
paper's setting; at `n = 21` it is 7–8×.

### Discrete-monitoring correction

`discrete_bars=M` divides each Parkinson / Garman–Klass / Rogers–Satchell term by
its expected bias factor `b(M)` for a Gaussian random walk sampled at `M` intrabar
returns (in Yang–Zhang only the RS component, in `gk_overnight` only the GK part):

| M | 26 | 78 | 390 | 2340 | 23 400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parkinson | 0.773 | 0.860 | 0.935 | 0.973 | 0.991 |
| Garman–Klass | 0.685 | 0.807 | 0.909 | 0.962 | 0.988 |
| Rogers–Satchell | 0.678 | 0.804 | 0.909 | 0.962 | 0.988 |

The Rogers–Satchell factor is **exact**: a Baxter–Spitzer identity gives
`b_RS(M) = (1/(πM)) Σ_{j+k≤M} (jk)^{−1/2}`. Parkinson and Garman–Klass depend on
`E[(H − L)²]`, which has no closed form for a discrete walk, and are control-variate
Monte Carlo at 10⁶ bars per `M` (standard error ≤ 1e-4). The table covers
`M ∈ [2, 23 400]` on a log grid and is interpolated in `1/√M`
(`benchmarks/ohlc_vol/discreteness_table.py` regenerates it).

!!! warning "Off by default"
    On real data `M` — the number of trades that formed the high and low — is not
    known, and microstructure noise *widens* the observed range, partly offsetting
    the bias. Treat `discrete_bars` as an option for data you sampled yourself
    (e.g. one-minute bars aggregated to daily), not as a default.

### Invalid bars

A bar is valid when every price the method reads is finite and positive and
`L ≤ min(O, C) ≤ max(O, C) ≤ H`. `invalid="null"` (default) drops an invalid bar,
which then does not count towards `min_periods` (the `*_n_valid` column reports how
many valid bars each window held); `"clip"` repairs the ordering row-locally
(`H = max(H, O, C)`, `L = min(L, O, C)`); `"raise"` names the first offender;
`"keep"` exists for parity tests. Zero-range bars (`H = L`) are legitimate and kept.
Missing or invalid bars are never filled from a neighbour. CRSP-style negative
prices (the bid/ask-midpoint flag) must be `abs()`-ed by the caller.

### Annualisation

`periods_per_year` (252, 365, `252 * 390`, …) multiplies the variance after
averaging. It is a constant you supply; it is never inferred from the data, because
an inferred frequency is a function of the whole sample.

## `.panel.rolling_vol` and the `rs_vol` rename

`.panel.rs_vol` was documented as "Rogers–Satchell-style volatility" but has always
been a trailing rolling standard deviation of one column. It is now
`.panel.rolling_vol` (expression and frame forms, bitwise-identical output), and
`rs_vol` is a deprecated alias that emits a `FutureWarning` once per process. The
name `rs_vol` will never be reused for the real Rogers–Satchell estimator — that
would change existing users' numbers without an error. The OHLC estimator is
`range_volatility(method="rogers_satchell")`. See `MIGRATING.md`.

## Leak safety

Every estimator here is registered with `safe_scope="rowwise"` and verified by the
registry conformance suite with both instruments of `panelary.testing`:
`assert_no_lookahead` (perturb the future, the past must not move) and
`assert_prefix_invariant` (truncate the panel, no surviving value may move). The
named traps are tested with a deliberately leaky variant that the checks must
catch — for example, Yang–Zhang's `k` computed from the series length passes the
perturbation test and fails prefix invariance.

Range estimators use log *ratios* of prices at most one bar apart, so they are
unchanged by a back-adjustment factor applied to all earlier prices (tested with a
synthetic split); measures on price *levels* are not.

## Performance

Measured with `benchmarks/ohlc_vol/bench_daily.py` on 5000 entities × 5000 bars
(25M rows; Apple M5 Pro, 15 threads, polars 1.44.2, 2026-09-29, on a machine shared
with other jobs — load average ≈ 14 — so treat these as upper bounds):

| Workload | Measured | Budget (plan §11) |
| --- | ---: | ---: |
| Parkinson + GK + RS, w = 21 (three calls) | 3.2 s under load; 1.7 s idle | ≤ 2.5 s (fail at 5 s) |
| Yang–Zhang, w = 21 | 1.4 s | ≤ 3 s |
| close-to-close, w = 21 | 0.6 s | — |

Each call sorts, validates and logs its own copy of the prices, so three separate
calls cost more than the plan's single-pass prototype (1.21 s); the budget's 2×
failure threshold still holds.

## References

Parkinson (1980), *J. Business* 53(1); Garman & Klass (1980), *J. Business* 53(1);
Rogers & Satchell (1991), *Ann. Appl. Probab.* 1(4); Yang & Zhang (2000),
*J. Business* 73(3); Broadie, Glasserman & Kou (1997), *Math. Finance* 7(4);
Asmussen, Glynn & Pitman (1995), *Ann. Appl. Probab.* 5(4).
