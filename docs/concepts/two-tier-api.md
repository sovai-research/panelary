# The two-tier API

Panelary exposes the same leak-safe machinery at two levels of abstraction. You can
stay in plain Polars and reach for the panel operators as **namespaces on a frame**
(Tier 1), or you can wrap your data in a **`PanelFrame`** and compose **sklearn-shaped
estimators and pipelines** (Tier 2). Both tiers share the same expression kernel and
the same leak-safety guarantees; they interoperate freely.

## Tier 1 — bare-frame namespaces (`.panel` / `.xs` / `.ts`)

Importing `panelary` registers custom Polars namespaces (a side effect). After
that, panel operators are available directly on any `DataFrame` / `LazyFrame` /
`Expr` — no wrapper, no fitting, no objects to manage.

```python
import polars as pl
import panelary as pn  # registers .panel / .xs / .ts namespaces

df = pl.DataFrame(
    {
        "ticker": ["A", "A", "A", "B", "B", "B"],
        "date":   [1, 2, 3, 1, 2, 3],
        "ret":    [0.10, -0.05, 0.02, 0.03, 0.04, -0.01],
    }
).sort(["ticker", "date"])

# Per-entity, causal — frame-level form (over= is REQUIRED):
df = df.panel.zscore("ret", window=2, over="ticker", suffix="_z")

# Cross-sectional, same-date — frame-level form (over= is REQUIRED):
df = df.xs.rank("ret", over="date", normalize=True, suffix="_rank")
```

There are three registered namespaces:

| Namespace | Axis | Scope | Shipped operators |
| --------- | ---- | ----- | ----------------- |
| `.panel`  | per-entity, causal | `over=<entity>` | `frac_diff`, `zscore`, `rolling_vol` |
| `.xs`     | cross-sectional, same-date | `over=<time>` | `rank`, `demean`, `zscore`, `standardize`, `winsorize`, `quantile_bin`, `neutralize` |
| `.ts`     | single-series feature extraction | your own `group_by`/`over` | tsfresh-style features (`absolute_energy`, `autocorrelation`, …) |

Each of `.panel` and `.xs` comes in **two callable forms** built from one shared
implementation:

- **Frame-level** — `df.panel.zscore("ret", window=2, over="ticker")`. Names the
  column(s), takes a **required** `over=` key, and controls output naming with
  `alias=` (single column) or `suffix=` (`f"{col}{suffix}"` per column; default is
  in-place). The required `over=` is a deliberate guardrail: a per-entity op computed
  across the whole frame would bleed one entity into the next, and a cross-sectional
  op is undefined without a cross-section.
- **Expression-level** — `pl.col("ret").panel.zscore(2).over("ticker")`. Returns a
  bare `pl.Expr` with *no* grouping of its own; you compose `.over(...)` yourself. Use
  it inside `with_columns`, alongside other expressions, or when you need full control
  of grouping.

```python
# Expression form — you own the .over(...) scope.
out = df.with_columns(
    ret_fd=pl.col("ret").panel.frac_diff(0.4).over("ticker"),
    ret_cs=pl.col("ret").xs.zscore().over("date"),
)
```

**Reach for Tier 1 when** you are doing ad-hoc feature engineering, exploring in a
notebook, or adding a couple of columns to an existing Polars pipeline and you do not
need fitted state carried across a train/test boundary.

## Tier 2 — `PanelFrame` + estimators + `Pipeline`

Tier 2 adds the objects you need for *modelling*: a typed panel view, sklearn-shaped
transformers/estimators that **learn parameters on training rows only**, and a
`Pipeline` that threads them together without leaking.

```python
import panelary as pn

# A typed, lazy view that validates and remembers the (entity, time) keys.
panel = pn.PanelFrame(df, entity="ticker", time="date").sort_panel()

# A transformer learns statistics in fit(), applies them in transform().
scaler = pn.transform.TimeSeriesScaler(columns="ret", mode="zscore")
scaled = scaler.fit(panel).transform(panel)   # -> a PanelFrame
print(scaled.collect())
```

The pieces:

- **`PanelFrame`** — a thin, lazy *view* over a `LazyFrame` that validates the
  `(entity, time)` contract once and exposes panel-aware helpers (`sort_panel`,
  `feature_cols`, `over_entity`, `entities`, `assert_unique_keys`, …). It never copies
  data and stays lazy until you call `.collect()`.
- **`PanelTransformer` / `PanelEstimator`** — abstract bases with sklearn ergonomics:
  `fit` / `transform` / `fit_transform` (and `predict` for estimators). Every concrete
  subclass declares two machine-checkable booleans, `panel_safe` and `leakage_safe`,
  so callers (and cross-validators) can *refuse* an unsafe transform across a fold
  boundary. Shipped transformers live under `pn.transform`: `TimeSeriesScaler`,
  `CrossSectionalScaler`, `CrossSectionalRank`, `Neutralize`, `FracDiff`.
- **`Pipeline`** — chains named steps and guarantees leak-safety end to end: `fit`
  fits each step **only on the train-fold output of its predecessors**, and
  `transform` / `predict` apply the already-fitted steps with no re-fitting, so
  applying the pipeline to a test fold cannot leak. It mirrors sklearn's `named_steps`,
  integer/string indexing, and slicing.

```python
pipe = pn.Pipeline([
    ("scale", pn.transform.TimeSeriesScaler(columns="ret", mode="zscore", suffix="_z")),
    ("xsrank", pn.transform.CrossSectionalScaler(columns="ret_z", suffix="_r")),
])
result = pipe.fit_transform(df, entity="ticker", time="date")
print(result.collect())
```

**Reach for Tier 2 when** you are building a model: you need fit-on-train /
apply-on-test semantics, want the leak-safety contract enforced, or are feeding the
result into `PurgedKFold` / `CombinatorialPurgedCV` and the `cross_validate` /
`validate` runner.

## `PanelFrame` is optional sugar

A key design point: **the estimator layer does not require a `PanelFrame`.** Every
`fit` / `transform` / `predict` / `fit_transform` accepts *either* a `PanelFrame`
*or* a bare `pl.DataFrame` / `pl.LazyFrame` plus `entity=` / `time=` keys. The bare
frame is wrapped on the fly.

```python
# Way A — pass a PanelFrame (its keys win):
panel = pn.PanelFrame(df, entity="ticker", time="date")
pn.transform.TimeSeriesScaler(columns="ret").fit_transform(panel)

# Way B — pass a bare frame + keys per call:
pn.transform.TimeSeriesScaler(columns="ret").fit_transform(
    df, entity="ticker", time="date"
)

# Way C — configure the keys once on the constructor, then pass bare frames:
scaler = pn.transform.TimeSeriesScaler(columns="ret", entity="ticker", time="date")
scaler.fit(df).transform(df)
```

All three are equivalent. If you pass a `PanelFrame` *and* conflicting `entity=` /
`time=` keys, that raises — the `PanelFrame`'s validated contract is not silently
overridden. Transforms always **return a `PanelFrame`** (lazy where possible); call
`.collect()` for a `pl.DataFrame` or `.to_native()` to drop the wrapper.

## How the tiers interoperate

The two tiers are the same kernel wearing different clothes, so you mix them freely:

- A `PanelFrame` wraps a `LazyFrame`, and `panel.lazy()` / `panel.to_native()` hand
  you a plain Polars frame — on which the Tier-1 `.panel` / `.xs` / `.ts` namespaces
  are available again.
- Tier-1 columns you engineer with `df.panel.*` / `df.xs.*` become ordinary columns a
  Tier-2 estimator can then `fit` on.
- The Tier-2 `CrossSectionalScaler` and the Tier-1 `.xs.zscore` compute the *same*
  expression — Tier 2 just adds fit/transform bookkeeping and the safety contract.

## The same task, both ways

Cross-sectional (same-date) z-score of returns, done in each tier:

=== "Tier 1 — bare-frame namespace"

    ```python
    import polars as pl
    import panelary as pn

    result = df.xs.zscore("ret", over="date", suffix="_z")
    print(result.sort(["ticker", "date"]))
    ```

    ```text
    shape: (6, 4)
    ┌────────┬──────┬───────┬───────┐
    │ ticker ┆ date ┆ ret   ┆ ret_z │
    ╞════════╪══════╪═══════╪═══════╡
    │ A      ┆ 1    ┆ 0.1   ┆ 1.0   │
    │ A      ┆ 2    ┆ -0.05 ┆ -1.0  │
    │ A      ┆ 3    ┆ 0.02  ┆ 1.0   │
    │ B      ┆ 1    ┆ 0.03  ┆ -1.0  │
    │ B      ┆ 2    ┆ 0.04  ┆ 1.0   │
    │ B      ┆ 3    ┆ -0.01 ┆ -1.0  │
    └────────┴──────┴───────┴───────┘
    ```

=== "Tier 2 — estimator + PanelFrame"

    ```python
    import panelary as pn

    panel = pn.PanelFrame(df, entity="ticker", time="date")
    scaler = pn.transform.CrossSectionalScaler(columns="ret")
    result = scaler.fit_transform(panel).collect()
    print(result.sort(["ticker", "date"]))
    ```

    ```text
    shape: (6, 3)
    ┌────────┬──────┬──────┐
    │ ticker ┆ date ┆ ret  │
    ╞════════╪══════╪══════╡
    │ A      ┆ 1    ┆ 1.0  │
    │ A      ┆ 2    ┆ -1.0 │
    │ A      ┆ 3    ┆ 1.0  │
    │ B      ┆ 1    ┆ -1.0 │
    │ B      ┆ 2    ┆ 1.0  │
    │ B      ┆ 3    ┆ -1.0 │
    └────────┴──────┴──────┘
    ```

Same numbers, same leak-safety. Tier 1 writes a new `ret_z` column in place on a plain
frame; Tier 2 returns a `PanelFrame` (here scaling in place) and carries the fit/apply
contract you need for cross-validation.

!!! tip "Rule of thumb"
    Exploring or adding a few features → **Tier 1**. Building a model with
    train/test discipline and cross-validation → **Tier 2**. They share the same
    kernel, so start in Tier 1 and graduate to Tier 2 without rewriting your feature
    logic.

## Where to go next

- **[Panel data](panel-data.md)** — the data shape both tiers assume, and the
  per-entity vs cross-sectional axes.
