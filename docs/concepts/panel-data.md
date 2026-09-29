# Panel data

**Panelary** is built for one shape of data: a **panel** — many *entities* observed
over *time*. A ticker's price every day, a customer's spend every month, a country's
GDP every quarter. Every feature, transformer, and cross-validator in the library
assumes this shape and protects its two axes from leaking into each other.

## Long format: `(entity, time, *features)`

A panel is stored **long** (also called "tidy" or "stacked"): one row per
`(entity, time)` observation, with the measured quantities in the remaining
columns.

```python
import polars as pl

df = pl.DataFrame(
    {
        "ticker": ["A", "A", "A", "B", "B", "B"],  # entity
        "date":   [1, 2, 3, 1, 2, 3],              # time
        "ret":    [0.10, -0.05, 0.02, 0.03, 0.04, -0.01],  # feature
    }
).sort(["ticker", "date"])
print(df)
```

```text
shape: (6, 3)
┌────────┬──────┬───────┐
│ ticker ┆ date ┆ ret   │
│ ---    ┆ ---  ┆ ---   │
│ str    ┆ i64  ┆ f64   │
╞════════╪══════╪═══════╡
│ A      ┆ 1    ┆ 0.1   │
│ A      ┆ 2    ┆ -0.05 │
│ A      ┆ 3    ┆ 0.02  │
│ B      ┆ 1    ┆ 0.03  │
│ B      ┆ 2    ┆ 0.04  │
│ B      ┆ 3    ┆ -0.01 │
└────────┴──────┴───────┘
```

Three roles of columns:

- **entity** — the panel id (`ticker`). Which series a row belongs to.
- **time** — the ordering axis (`date`). When the observation happened.
- **features** — everything else (`ret`). What was measured.

This is the same layout as pandas' *long* DataFrame, not the *wide* one-column-per-series
layout. Long format scales to millions of entities and lets Polars express every
operation as a grouped expression.

## The entity × time grid

Conceptually a panel is a grid: entities down one axis, time across the other, with
a feature value in each cell.

|            | date=1 | date=2 | date=3 |
| ---------- | ------ | ------ | ------ |
| **A**      | 0.10   | -0.05  | 0.02   |
| **B**      | 0.03   | 0.04   | -0.01  |

The long table above is just this grid unrolled row by row. Two directions of movement
in the grid give the **two axes of operations** (see below):

- **along a row** (fixing an entity, moving through time) → *per-entity* / time-series work.
- **down a column** (fixing a date, moving across entities) → *cross-sectional* work.

## Why panels differ from a single time series and from a cross-section

A **single time series** has time but only one entity. You can compute a rolling
mean over the whole column and never worry about mixing series.

A **cross-section** has entities but only one time point. You can rank or standardize
the column and never worry about the future.

A **panel has both at once**, and that is exactly where leakage hides:

- A rolling / lagged / cumulative feature computed over the *whole* long column will
  happily use entity A's last row as if it were entity B's first — a silent
  **cross-entity leak**. Per-entity operations must be scoped `.over(entity)`.
- A cross-sectional rank or z-score is only meaningful *within one date*; computed
  over the whole column it mixes dates. Cross-sectional operations must be scoped
  `.over(time)`.

Panelary makes these scopes first-class so you cannot forget them (the frame-level
namespaces even *require* an `over=` key).

## Gaps and the `(entity, time)` grid

Real panels are rarely a full rectangle: entities enter and leave, dates are missing,
holidays punch holes. That is fine — the panel stays long and simply has fewer rows
for some entities. What matters is:

- **Uniqueness.** Each `(entity, time)` pair should appear at most once. Panelary can
  check this for you (it materialises, so it is opt-in):

  ```python
  import panelary as pn
  pn.PanelFrame(df, entity="ticker", time="date").assert_unique_keys()
  ```

- **Gaps are not zeros.** A missing row means "not observed", not "value 0". Trailing
  windows count *rows*, so gaps affect what a rolling window spans. Impute or resample
  first if you need calendar-regular windows.

## Sorting requirements

Any per-entity feature that looks backwards — lags, rolling windows, cumulative sums,
fractional differencing — is only correct if each entity's rows are in ascending time
order. Sort **by `(entity, time)`** once, near the top of a pipeline:

```python
import panelary as pn

panel = pn.PanelFrame(df, entity="ticker", time="date").sort_panel()
print(panel.is_sorted_per_entity())  # True
```

`sort_panel()` orders by entity then ascending time; `is_sorted_per_entity()` verifies
the current row order without reordering it. Cross-sectional operations do not depend
on within-entity ordering (they group by date), but sorting first never hurts.

## The two axes of operations

Every panel feature falls on one of two axes. Panelary gives each its own namespace so
your intent — and the leak-safety scope — is explicit in the call.

### Per-entity — `.panel` (over each entity's own history)

Time-series transforms that walk *forward through one entity's history*, using only
the past. Scope them with `.over(entity)`.

```python
import panelary  # registers the namespaces (side effect)

# Causal rolling z-score of each ticker's returns, per entity.
out = df.with_columns(
    ret_z=pl.col("ret").panel.zscore(2).over("ticker"),
)
```

```text
shape: (6, 4)
┌────────┬──────┬───────┬───────────┐
│ ticker ┆ date ┆ ret   ┆ ret_z     │
│ ---    ┆ ---  ┆ ---   ┆ ---       │
│ str    ┆ i64  ┆ f64   ┆ f64       │
╞════════╪══════╪═══════╪═══════════╡
│ A      ┆ 1    ┆ 0.1   ┆ null      │
│ A      ┆ 2    ┆ -0.05 ┆ -0.707107 │
│ A      ┆ 3    ┆ 0.02  ┆ 0.707107  │
│ B      ┆ 1    ┆ 0.03  ┆ null      │
│ B      ┆ 2    ┆ 0.04  ┆ 0.707107  │
│ B      ┆ 3    ┆ -0.01 ┆ -0.707107 │
└────────┴──────┴───────┴───────────┘
```

`.panel` operators are **causal** (leak-safe in time) and, once scoped `.over(entity)`,
**panel-safe** (no cross-entity bleed). Shipped operators: `frac_diff`, `zscore`,
`rolling_vol` (a rolling standard deviation; `rs_vol` is its deprecated alias).

### Cross-sectional — `.xs` (within each date, across entities)

Comparisons *between entities at the same instant*: rank returns within the day,
demean, standardize, winsorize, quantile-bin, or factor-neutralize. Scope them with
`.over(time)`.

```python
out = df.with_columns(
    ret_rank=pl.col("ret").xs.rank(normalize=True).over("date"),
    ret_csz=pl.col("ret").xs.zscore().over("date"),
)
```

```text
shape: (6, 5)
┌────────┬──────┬───────┬──────────┬─────────┐
│ ticker ┆ date ┆ ret   ┆ ret_rank ┆ ret_csz │
│ ---    ┆ ---  ┆ ---   ┆ ---      ┆ ---     │
│ str    ┆ i64  ┆ f64   ┆ f64      ┆ f64     │
╞════════╪══════╪═══════╪══════════╪═════════╡
│ A      ┆ 1    ┆ 0.1   ┆ 0.666667 ┆ 1.0     │
│ A      ┆ 2    ┆ -0.05 ┆ 0.333333 ┆ -1.0    │
│ A      ┆ 3    ┆ 0.02  ┆ 0.666667 ┆ 1.0     │
│ B      ┆ 1    ┆ 0.03  ┆ 0.333333 ┆ -1.0    │
│ B      ┆ 2    ┆ 0.04  ┆ 0.666667 ┆ 1.0     │
│ B      ┆ 3    ┆ -0.01 ┆ 0.333333 ┆ -1.0    │
└────────┴──────┴───────┴──────────┴─────────┘
```

`.xs` operators are inherently **leak-safe in time** (they use only same-date data) but
are *not* panel-safe in the per-entity sense — by design they mix entities within a
date. Shipped operators: `rank`, `demean`, `zscore`, `standardize`, `winsorize`,
`quantile_bin`, `neutralize`.

!!! tip "Which axis?"
    "Compared to its own past" → **`.panel`**, scope `.over(entity)`.
    "Compared to its peers today" → **`.xs`**, scope `.over(time)`.

## The `entity=` / `time=` convention

Across the whole library the panel keys are named the same way: **`entity`** for the
id column and **`time`** for the ordering column. You supply them once and everything
downstream trusts them.

```python
import panelary as pn

# On the PanelFrame view:
panel = pn.PanelFrame(df, entity="ticker", time="date")

# Or on an estimator, taking a bare polars frame:
scaler = pn.transform.TimeSeriesScaler(columns="ret")
scaled = scaler.fit_transform(df, entity="ticker", time="date")
```

`entity` and `time` must name real, distinct columns; `time` must have an orderable
dtype (numeric or temporal). When keys are omitted where a default is allowed (e.g.
`pn.as_panel(df)`), the convention is **column 0 = entity, column 1 = time** — but
passing them explicitly is always clearer.

In the frame-level namespaces the *same idea* appears as the `over=` argument:
`df.panel.zscore("ret", window=2, over="ticker")` scopes per entity;
`df.xs.rank("ret", over="date")` scopes per cross-section. These `over` keys are
**required** — omitting them raises, precisely to stop an accidental cross-boundary
computation.

## Where to go next

- **[The two-tier API](two-tier-api.md)** — when to reach for the bare-frame
  namespaces versus the `PanelFrame` / `Pipeline` objects, and how they interoperate.
