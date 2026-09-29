# Labeling

Supervised learning on panels starts with a **target**. Panelary ships the
leak-safe labeling primitives from López de Prado, *Advances in Financial
Machine Learning* (AFML, Ch. 3), rewritten Polars-native for long-format panel
data:

| Function | What it produces |
| --- | --- |
| [`triple_barrier`](#triple-barrier) | Sign of the first profit-take / stop-loss / vertical barrier touched. |
| [`fixed_horizon`](#fixed-horizon) | Forward return over a fixed number of steps (continuous or ternary sign). |
| [`meta_label`](#meta-labeling) | Binary *act / pass* on a primary model's side signal. |

Every labeler operates **forward-only** over an event span `[t, t1]` and returns
three columns — `label`, `ret`, and `t1` — on top of your input rows (sorted by
`[entity, time]`). The `t1` column is the shared contract that drives
[label-driven purge in cross-validation](#the-t1-contract): it is always emitted
in the *same dtype* as your time column so a training row whose span overlaps a
test set can be purged.

```python
from panelary.label import triple_barrier, fixed_horizon, meta_label
```

## A worked panel

All examples below use a small synthetic panel of daily closes for three
tickers. An integer `date` axis is used for brevity — a `Date`/`Datetime` column
works identically.

```python
import numpy as np
import polars as pl

def make_prices(n_per=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for eid in ["AAA", "BBB", "CCC"]:
        steps = rng.normal(0.0005, 0.02, size=n_per)
        price = 100.0 * np.exp(np.cumsum(steps))
        frames.append(pl.DataFrame({
            "ticker": [eid] * n_per,
            "date": list(range(n_per)),
            "close": price,
        }))
    return pl.concat(frames)

prices = make_prices()
```

The entity and time columns default to the first two columns; pass `entity=` and
`time=` to be explicit. `triple_barrier` accepts either a `polars.DataFrame` or
a `polars.LazyFrame` and always returns an eager `DataFrame`.

## Triple barrier

The triple-barrier method looks forward over `[t, t + max_holding]` (in time
order, within each entity) and places a profit-take and a stop-loss barrier,
both scaled by a **trailing** volatility estimate. The label is the sign of the
first barrier touched; if neither horizontal barrier is hit before the vertical
barrier (`t + max_holding` steps), the label is `0`.

```python
labeled = triple_barrier(
    prices,
    entity="ticker",
    time="date",
    price="close",
    pt=2.0,            # profit-take multiple of trailing vol (0 disables it)
    sl=1.0,            # stop-loss multiple of trailing vol (0 disables it)
    max_holding=10,    # vertical barrier: forward steps per entity
    vol_lookback=20,   # trailing look-back for the vol estimate
)

labeled.select("ticker", "date", "close", "label", "ret", "t1").head(6)
```

```text
shape: (6, 6)
┌────────┬──────┬────────────┬───────┬───────────┬─────┐
│ ticker ┆ date ┆ close      ┆ label ┆ ret       ┆ t1  │
│ ---    ┆ ---  ┆ ---        ┆ ---   ┆ ---       ┆ --- │
│ str    ┆ i64  ┆ f64        ┆ i64   ┆ f64       ┆ i64 │
╞════════╪══════╪════════════╪═══════╪═══════════╪═════╡
│ AAA    ┆ 0    ┆ 100.301915 ┆ 0     ┆ 0.00698   ┆ 10  │
│ AAA    ┆ 1    ┆ 100.087289 ┆ 0     ┆ 0.010479  ┆ 11  │
│ AAA    ┆ 2    ┆ 101.428199 ┆ 1     ┆ 0.027056  ┆ 6   │
│ AAA    ┆ 3    ┆ 101.692052 ┆ -1    ┆ -0.010161 ┆ 4   │
│ AAA    ┆ 4    ┆ 100.658718 ┆ 1     ┆ 0.034907  ┆ 6   │
│ AAA    ┆ 5    ┆ 101.440017 ┆ 1     ┆ 0.026936  ┆ 6   │
└────────┴──────┴────────────┴───────┴───────────┴─────┘
```

The added columns are:

- **`label`** (`Int64`) — `+1` (profit-take), `-1` (stop-loss), or `0` (vertical
  barrier).
- **`ret`** (`Float64`) — realized return from `t` to the touched barrier.
- **`t1`** (same dtype as `time`) — the timestamp at which the label resolves.
  Always `t <= t1`, and within `max_holding` steps of `t`.
- **`censored`** (`Boolean`) — `True` when no barrier was touched and the
  vertical barrier lies beyond the entity's last row: the horizon was
  *truncated*, so the label is not resolved yet (the last row is `label=0,
  t1=t`). Drop these rows before training; see
  [Label weights](labels-and-weights.md).

A causal volatility column can replace the built-in trailing standard deviation
for the barrier widths: `triple_barrier(df, ..., vol="sigma")`.

Notice `t1` varies row-by-row: an early barrier touch (row `date=2` resolves at
`t1=6`) gives a short span, while an untouched observation runs to the full
vertical barrier (`date=0` → `t1=10`).

Inspect the class balance:

```python
labeled.get_column("label").value_counts().sort("label")
```

```text
shape: (3, 2)
┌───────┬───────┐
│ label ┆ count │
│ ---   ┆ ---   │
│ i64   ┆ u32   │
╞═══════╪═══════╡
│ -1    ┆ 216   │
│ 0     ┆ 32    │
│ 1     ┆ 112   │
└───────┴───────┘
```

!!! note "Leak-safety"
    The volatility estimate is strictly trailing (past-and-present only) and the
    forward scan stops at the first touch, so nothing beyond `[t, t1]` can
    influence a row's label. Set `pt=0` or `sl=0` to disable a horizontal
    barrier (e.g. a stop-loss-only label).

## Fixed horizon

`fixed_horizon` labels each observation by its forward return over a fixed
number of steps. With `threshold=None` the `label` column is the continuous
forward return:

```python
fh = fixed_horizon(
    prices, entity="ticker", time="date", price="close", horizon=5
)
fh.select("ticker", "date", "label", "ret", "t1").head(4)
```

```text
shape: (4, 5)
┌────────┬──────┬──────────┬──────────┬─────┐
│ ticker ┆ date ┆ label    ┆ ret      ┆ t1  │
│ ---    ┆ ---  ┆ ---      ┆ ---      ┆ --- │
│ str    ┆ i64  ┆ f64      ┆ f64      ┆ i64 │
╞════════╪══════╪══════════╪══════════╪═════╡
│ AAA    ┆ 0    ┆ 0.011347 ┆ 0.011347 ┆ 5   │
│ AAA    ┆ 1    ┆ 0.040816 ┆ 0.040816 ┆ 6   │
│ AAA    ┆ 2    ┆ 0.047219 ┆ 0.047219 ┆ 7   │
│ AAA    ┆ 3    ┆ 0.030419 ┆ 0.030419 ┆ 8   │
└────────┴──────┴──────────┴──────────┴─────┘
```

Pass a `threshold` to get a ternary sign label (`+1` above `threshold`, `-1`
below `-threshold`, else `0`) — useful for classification:

```python
fh_sign = fixed_horizon(
    prices, entity="ticker", time="date", price="close",
    horizon=5, threshold=0.02,
)
fh_sign.get_column("label").value_counts().sort("label")
```

```text
shape: (3, 2)
┌───────┬───────┐
│ label ┆ count │
│ ---   ┆ ---   │
│ i64   ┆ u32   │
╞═══════╪═══════╡
│ -1    ┆ 110   │
│ 0     ┆ 149   │
│ 1     ┆ 101   │
└───────┴───────┘
```

Here `t1` is simply the timestamp `horizon` steps ahead, and is `null` at the
tail of each entity where the full horizon is unavailable. `ret` is always the
continuous forward return; only `label` is discretized when `threshold` is set.
The forward return comes from `factor.forward_return`, so an irregular time
grid raises unless you pass `allow_gaps=True`.

## Meta-labeling

Meta-labeling (AFML Ch. 3.6) trains a secondary model to decide *whether to act*
on a primary model that has already picked a *side*. Given a primary side signal
and a realized label, `meta_label` emits a binary `meta_label` column: `1` (act)
when the primary side agrees with a non-zero realized outcome, `0` (pass)
otherwise.

Start from the triple-barrier `label` above and add a simple primary signal
(here, price above/below its 10-step moving average):

```python
primary = labeled.with_columns(
    (pl.col("close") - pl.col("close").rolling_mean(10).over("ticker"))
    .sign().alias("signal")
)

meta = meta_label(primary, primary_signal="signal", label="label")
meta.get_column("meta_label").value_counts().sort("meta_label")
```

```text
shape: (2, 2)
┌────────────┬───────┐
│ meta_label ┆ count │
│ ---        ┆ ---   │
│ i64        ┆ u32   │
╞════════════╪═══════╡
│ 0          ┆ 210   │
│ 1          ┆ 150   │
└────────────┴───────┘
```

Only the **signs** of `primary_signal` and `label` are used. Train the secondary
model on `meta_label` to size or gate the primary bets; keep the primary side
model separate.

## The `t1` contract

The single most important output for leak-safe evaluation is `t1`. A label at
time `t` "uses up" information over the whole span `[t, t1]`. If a training row's
span overlaps a test row's span, the two share information and the training row
must be **purged** — otherwise the backtest leaks.

Because every labeler emits `t1` in the same dtype as `time`, you can hand the
column name straight to the purged cross-validators:

```python
from panelary import PurgedKFold, cross_validate

cv = PurgedKFold(n_splits=5, embargo=2, t1="t1")   # label-driven purge
```

When `t1=` is supplied it **supersedes** the scalar `horizon`: each observation
is purged against the *explicit* label interval rather than a fixed-width
window. For a shared-time-axis panel, `t1` is aggregated to the **max** end time
per unique time (the most conservative purge). See the
[Validation guide](validation.md) for the full end-to-end pipeline.

## API reference

::: panelary.label.triple_barrier
::: panelary.label.fixed_horizon
::: panelary.label.meta_label
