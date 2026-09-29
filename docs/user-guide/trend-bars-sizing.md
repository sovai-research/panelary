# Trend labels, bars and bet sizing

Three pieces of the López de Prado toolkit, each built so that it cannot leak:

| Module | What it gives you |
| --- | --- |
| [`pn.label`](#trend-scanning) | `trend_scanning` (plus the causal `.ts.trend_scan` feature), `excess_over_median`, `quantile_label` |
| [`pn.sample`](#bars) | `bars` (tick / volume / dollar, fixed or adaptive) and `imbalance_bars` (imbalance and run bars) |
| [`pn.sizing`](#bet-sizing) | `bet_size`, `average_active`, `discretize`, and sigmoid sizing with limit prices |

Every label here emits `t1` (in your time column's dtype) and a `censored`
flag, the same span contract as [`triple_barrier`](labeling.md#the-t1-contract).
Every bar is stamped at its **last tick**. numba (the `fast` extra) speeds up
the sequential kernels when installed; the numpy / pure-Python twins give
bit-identical results without it.

```python
import numpy as np
import polars as pl
import panelary as pn

rng = np.random.default_rng(0)
prices = pl.concat(
    pl.DataFrame({
        "ticker": [t] * 250,
        "date": list(range(250)),
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 250))),
    })
    for t in ["AAA", "BBB", "CCC", "DDD"]
)
```

## Trend scanning

For every row `t`, `trend_scanning` fits an OLS line to the log price over
`y[t .. t+L-1]` for each window length `L` in a grid and keeps the `L` with
the largest slope t-statistic. The label is the sign of that t-statistic
(or `0` below `t_threshold`).

```python
out = pn.label.trend_scanning(prices, price="close", min_window=5, max_window=20)
out.select("ticker", "date", "label", "t_value", "horizon", "t1", "t1_trend", "censored")
```

**Which end time to purge on.** The label was decided by scanning *every*
horizon up to `L_max`, so its information set runs to row `t + L_max - 1`:
that is `t1`. `t1_trend` (the end of the chosen horizon) is descriptive only;
purging on it would leave training labels that read test-period prices.

**The tail.** Rows without `L_max` rows of future data are `censored` and left
unlabelled (`allow_partial=False`, the default). A label computed from fewer
horizons at the end of the sample would change as data arrives.

**Accuracy.** The t-statistics come from a recursive-residual sweep over the
horizons (Brown, Durbin & Evans 1975), not from prefix sums of `y`, `x·y` and
`y²`. The prefix-sum shortcut cancels catastrophically: it is wrong in the second
decimal on log prices, and by up to 99 % on near-perfect trends, which are
exactly the rows the label is about. The sweep matches an exact rational
evaluation to about 1e-11 on random walks, at one pass over the horizons.

The look-back twin is a causal feature: the best-|t| trend over the windows
**ending** at each row.

```python
prices.with_columns(
    pl.col("close").log().ts.trend_scan(min_window=5, max_window=60).over("ticker").alias("trend_t"),
    pl.col("close").log().ts.trend_scan(max_window=60, output="horizon").over("ticker").alias("trend_len"),
)
```

## Cross-sectional labels

Both read the forward return from `pn.factor.forward_return`, the library's
single audited negative-shift site with its gap guard.

```python
pn.label.excess_over_median(prices, price="close", horizon=21, min_count=3)
pn.label.quantile_label(prices, price="close", horizon=21, q=3)                  # per-date buckets
pn.label.quantile_label(prices, price="close", horizon=21, q=3,
                        mode="trailing", lookback=60)                          # own history
```

* `excess_over_median`: forward return minus the date's median; `binary=True`
  gives its sign. Dates with fewer than `min_count` resolved names are null.
* `quantile_label`: buckets on a symmetric scale (`q=3` gives `-1, 0, +1`;
  even `q` has no zero bucket). In `mode="trailing"` the bucket edges are
  quantiles of the entity's previous `lookback` **resolved** forward returns:
  at `t` only returns that ended by `t` are used. Edges from the full-sample
  distribution would be a look-ahead.

A cross-sectional label reads every constituent's forward return, so its `t1`
is the latest constituent end time on that date (equal to the row's own on a
regular grid). **Survivorship:** a name that delists inside the horizon has no
forward return, so the median or rank is over survivors. That row is flagged
`censored`; the selection effect is yours to judge.

## Bars

Tick, volume and dollar bars close on the tick whose cumulative amount first
reaches the next multiple of the threshold. The output is a long panel, one row
per **completed** bar, stamped at its last tick:

```python
ticks = pl.DataFrame({
    "sym": ["X"] * 10_000,
    "ts": pl.datetime_range(pl.datetime(2024, 1, 2, 9, 30), pl.datetime(2024, 1, 2, 9, 30)
                            + pl.duration(seconds=9_999), "1s", eager=True),
    "px": 100 + 0.01 * np.cumsum(rng.integers(-2, 3, 10_000)),
    "qty": rng.integers(1, 500, 10_000),
})
bars = pn.sample.bars(ticks, entity="sym", time="ts", price="px", size="qty",
                      kind="dollar", threshold=2e6)
# sym, ts (last tick), t_open, open, high, low, close, volume, dollar_volume,
# vwap, n_ticks, buy_volume, bar_index
```

* **Lattice.** A tick belongs to bar `floor(C_{t-1} / θ)`, with `C` the
  cumulative amount *before* the tick, so the crossing tick closes its own bar.
  Overshoot carries into the next bar and boundaries sit on a fixed lattice:
  a bar's boundaries never depend on later ticks. `overshoot="reset"` restarts
  the count after each close instead.
* **Adaptive threshold.** `bars_per_day=50, lookback_days=20` sets each day's
  threshold from the entity's previous 20 **completed** days. The first 20 days
  give no bars unless you pass `init_threshold`. A threshold from the
  full-sample average would move every boundary when the sample grows.
* **Duplicate close times.** Two bars can close on the same timestamp;
  `on_duplicate_time="nudge"` moves the later one forward by one time unit
  (never earlier), `"error"` raises, `"keep"` keeps both.
* **Sides.** `buy_volume` uses the tick rule. Ticks before an entity's first
  price change have no side.

Information-driven bars close when the order-flow imbalance (or the longest
one-sided run) exceeds its expected value:

```python
imb = pn.sample.imbalance_bars(ticks, entity="sym", time="ts", price="px", size="qty",
                               kind="volume", init_expected_ticks=200, span_bars=20)
run = pn.sample.imbalance_bars(ticks, entity="sym", time="ts", price="px", size="qty",
                               kind="tick", run=True, init_expected_ticks=200)
# ... plus threshold, imbalance, clamped
```

The expectations are EWMAs over **completed** bars only, never over the bar in
progress or a full-sample mean. The expected bar length and the threshold feed
back on each other and can collapse or explode (a known property of the
definition), so `E[T]` is clamped to `bounds × init_expected_ticks`
(default 0.1 to 10) and each bar reports whether its threshold used a clamped
value. The default bounds are a guess, not a calibration: check bars-per-day
stability on your own data.

!!! note "Time bars"
    Calendar resampling stays in `pn.preprocessing.resample`. Pass
    `label="right"` to stamp each window at its close. The historical default
    stamps it at the open, which is a look-ahead of up to one `freq`; see
    [MIGRATING](https://github.com/sovai-research/panelary/blob/main/MIGRATING.md).

## Bet sizing

```python
oof = pl.DataFrame({"prob": [0.55, 0.7, 0.9, None], "side": [1, -1, 1, 1]})
pn.sizing.bet_size(oof, prob="prob", side="side")          # m = side · (2Φ(z) − 1)
```

`z = (p − 1/K) / sqrt(p(1 − p))` tests the predicted class's probability against
the uniform `1/K`. **Use out-of-fold probabilities** (from
`pn.cross_validate`). In-sample probabilities inherit the fit's overconfidence.

`average_active` averages the sizes of all bets active at each row
(`t0 ≤ t < exit`), with `0.0` where none is active:

```python
pn.sizing.average_active(frame, size="size", exit_time="exit", entity="ticker", time="date")
```

It takes a causal `exit_time`, not a label's `t1`. A trend-scanning `t1` is an
information end, not the moment a position is closed.

Dynamic sizing maps a forecast-price divergence `x` to `m = x / sqrt(w + x²)`:

```python
w = pn.sizing.sigmoid_w(divergence=10.0, size=0.95)      # calibrate
pn.sizing.target_position(w, forecast=115.0, price=100.0, max_pos=100)
pn.sizing.limit_price(target_pos=60, pos=10, forecast=115.0, w=w, max_pos=100)
pn.sizing.discretize(np.array([0.23, 0.26]), step=0.1)    # half-to-even
```

`limit_price` averages the inverse price over the signed path from `pos` to
`target_pos`. It agrees with AFML's printed loop where that loop is valid
(`0 ≤ pos < target`) and handles sells and short positions too.
