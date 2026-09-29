# Feature Engineering

**Panelary** turns a long-format `(entity, time, *values)` panel into wide feature
frames using pure, leak-safe [Polars](https://pola.rs/) expressions. Every feature is
computed **per entity** (and, where relevant, per date), so no information ever leaks
across entities or from the future.

There are three families of feature operators, each exposed as a Polars expression
namespace that is registered the moment you `import panelary`:

| Namespace | Scope | Combine with | Examples |
| --------- | ----- | ------------ | -------- |
| `.ts` | per-entity time-series extractors (tsfresh-style) | `group_by(entity)` / `.over(entity)` | `absolute_energy`, `longest_streak_above_mean`, `cid_ce`, `max_drawdown` |
| `.panel` | per-entity transforms | `.over(entity)` | `frac_diff`, `zscore`, `rolling_vol` |
| `.xs` | cross-sectional (per-date) transforms | `.over(time)` | `rank`, `demean`, `zscore`, `standardize`, `winsorize`, `quantile_bin`, `neutralize` |

A fourth namespace, `.factor`, holds the cross-sectional factor-research
operators (`forward_return`, `ic`, `orthogonalize`, `portfolio_sort`); see the
[factor API reference](../api-reference/factor.md).

For batch time-series featurisation there is also a one-call
[`extract_features`](#bulk-extraction-with-extract_features) helper and a clean-room
[catch22](#catch22) implementation.

---

## The `.ts` scalar extractors

Importing Panelary registers a `ts` (time-series) namespace on every Polars expression.
Each extractor consumes one series and reduces it to a scalar (or, for a handful, a
list/struct). Because they are ordinary Polars expressions, they run eagerly on a
`Series`, lazily on a `LazyFrame`, and — crucially — **per group** inside a `group_by`.

```python
import numpy as np
import polars as pl
import panelary as pn  # registers the .ts / .panel / .xs namespaces

rng = np.random.default_rng(0)

# A single series, three features at once
out = (
    pl.LazyFrame({"value": rng.normal(0, 1, 20)})
    .select(
        energy=pl.col("value").ts.absolute_energy(),
        longest_up=pl.col("value").ts.longest_streak_above_mean(),
        n_above_mean=pl.col("value").ts.count_above_mean(),
    )
    .collect()
)
print(out)
# shape: (1, 3)
# ┌───────────┬────────────┬──────────────┐
# │ energy    ┆ longest_up ┆ n_above_mean │
# │ f64       ┆ u32        ┆ u32          │
# ╞═══════════╪════════════╪══════════════╡
# │ 15.150875 ┆ 4          ┆ 10           │
# └───────────┴────────────┴──────────────┘
```

!!! warning "Use the extractor names, not raw reducers"
    The `.ts` namespace exposes named time-series features such as `absolute_energy`,
    `mean_abs_change`, `variation_coefficient`, `root_mean_square`, `cid_ce`,
    `max_drawdown`, `return_skew`, and `longest_streak_above_mean`. It is **not** a bag
    of plain reducers — there is no `.ts.mean()` or `.ts.std()`; use Polars' own
    `pl.col(...).mean()` for those.

### Per-entity extraction

On a panel, always group (or window) by the entity so each series is featurised
independently — this is what makes the result leak-safe.

```python
panel = pl.DataFrame({
    "ticker": ["A"] * 6 + ["B"] * 6,
    "date":   list(range(6)) * 2,
    "ret":    rng.normal(0, 1, 12),
})

feats = (
    panel.group_by("ticker", maintain_order=True)
    .agg(
        rms=pl.col("ret").ts.root_mean_square(),
        skew=pl.col("ret").ts.return_skew(),
        max_drawdown=pl.col("ret").ts.max_drawdown(),
    )
)
print(feats)
# shape: (2, 4)
# ┌────────┬──────────┬───────────┬──────────────┐
# │ ticker ┆ rms      ┆ skew      ┆ max_drawdown │
# │ ---    ┆ ---      ┆ ---       ┆ ---          │
# │ str    ┆ f64      ┆ f64       ┆ f64          │
# ╞════════╪══════════╪═══════════╪══════════════╡
# │ A      ┆ 0.738792 ┆ -1.255106 ┆ -1.4868      │
# │ B      ┆ 0.673727 ┆ -1.235255 ┆ -5.585107    │
# └────────┴──────────┴───────────┴──────────────┘
```

For **rolling** windows over each series, swap `group_by` for `group_by_dynamic`:

```python
rolling = (
    panel.sort("ticker", "date")
    .group_by_dynamic("date", every="3i", group_by="ticker")
    .agg(rms=pl.col("ret").ts.root_mean_square())
)
```

### Parametrised and list-valued extractors

Some `.ts` methods take arguments (`autocorrelation(n_lags)`, `binned_entropy(bin_count)`,
`c3(n_lags)`, `count_above(threshold)`, `cid_ce(normalize=...)`), and a few return a
list or struct rather than a scalar (`linear_trend`, `fft_coefficients`, `energy_ratios`,
`streak_length_stats`, `cwt_coefficients`). Those work inside `group_by(...).agg(...)`
too, but they are excluded from the argument-free
[bulk path](#bulk-extraction-with-extract_features) because they do not reduce to a single
column.

```python
detail = (
    panel.group_by("ticker", maintain_order=True)
    .agg(
        acf1=pl.col("ret").ts.autocorrelation(1),
        entropy=pl.col("ret").ts.binned_entropy(bin_count=5),
        trend=pl.col("ret").ts.linear_trend(),  # struct: slope / intercept / rss
    )
    .unnest("trend")
)
```

---

## Discovering features programmatically

Panelary ships a **feature registry** — the single source of truth for every registered
operator, its namespace, its parameters, and its safety guarantees. Use it instead of
hard-coding names.

```python
from panelary.registry import registry

# What namespaces exist?
print(registry.namespaces())
# ['factor', 'panel', 'ts', 'xs']

# Every registered ts scalar extractor (usable in extract_features)
ts_features = [spec.name for spec in registry.by_namespace("ts")]
print(len(ts_features), ts_features[:6])
# 42 ['absolute_energy', 'absolute_maximum', 'absolute_sum_of_changes',
#     'benford_correlation', 'cid_ce', 'count_above']

# Inspect one operator's contract
spec = registry.get("frac_diff")
print(spec.namespace, spec.params, spec.panel_safe, spec.leakage_safe)
# panel {'d': <class 'float'>, 'threshold': <class 'float'>} True True
```

Each entry is a `FeatureSpec` carrying `namespace`, `input_shape`, `output_shape`,
`params`, `tier`, `panel_safe`, `leakage_safe`, `source`, and `license`. Convenience views:

```python
registry.all()            # every spec, sorted by qualified name
registry.by_namespace("xs")   # just the cross-sectional operators
registry.to_records()     # JSON-friendly list of dicts (build a catalogue frame)
registry.audit()          # {'missing_panel_safe': [...], 'missing_leakage_safe': [...], ...}
```

The registry can also render a machine-readable catalogue for agents / `llms.txt`:

```python
print(registry.to_llms_txt())   # deterministic, line-oriented operator listing
```

!!! note "Registry vs. the full `.ts` namespace"
    The `.ts` namespace exposes **63 extractors** in total, but the registry catalogues
    the **42** that are argument-free scalar reductions — precisely the ones the bulk
    `extract_features` path can compute. Parametrised or list-valued extractors
    (e.g. `autocorrelation`, `linear_trend`, `fft_coefficients`) live on the namespace
    but are not in the bulk catalogue.

---

## Bulk extraction with `extract_features`

`extract_features` is a tsfresh-style batch extractor that is **lazy and panel-safe**: it
runs a single `group_by(entity).agg(...)` over the requested value column(s) and returns
exactly one row per entity, one column per feature.

```python
import panelary as pn

wide = pn.extract_features(
    panel,
    entity="ticker",
    time="date",
    features=["absolute_energy", "mean_abs_change", "variation_coefficient", "cid_ce"],
    column="ret",
)
print(wide)
# shape: (2, 5)
# ┌────────┬─────────────────┬─────────────────┬───────────────────────┬──────────┐
# │ ticker ┆ absolute_energy ┆ mean_abs_change ┆ variation_coefficient ┆ cid_ce   │
# │ ---    ┆ ---             ┆ ---             ┆ ---                   ┆ ---      │
# │ str    ┆ f64             ┆ f64             ┆ f64                   ┆ f64      │
# ╞════════╪═════════════════╪═════════════════╪═══════════════════════╪══════════╡
# │ A      ┆ 3.274884        ┆ 1.180956        ┆ 2.078615              ┆ 2.890714 │
# │ B      ┆ 2.723451        ┆ 0.670081        ┆ -0.822794             ┆ 1.69109  │
# └────────┴─────────────────┴─────────────────┴───────────────────────┴──────────┘
```

Key behaviours:

- **`features="all"`** (the default) computes every registered `ts` scalar feature (42
  columns). Pass a list to subset.
- **`column`** selects the value column(s). If omitted, every numeric column that is
  neither `entity` nor `time` is featurised.
- **Column naming.** With a single value column, output columns are named `<feature>`.
  With several value columns they are namespaced `<column>__<feature>`.
- **Lazy in, lazy out.** Pass a `LazyFrame` and the whole pipeline stays lazy; pass a
  `DataFrame` and it is collected on return.
- **Validated names.** An unknown feature name raises `ValueError` listing the valid ones.

Always pull the valid names from the registry rather than guessing:

```python
valid = [s.name for s in registry.by_namespace("ts")]
wide_all = pn.extract_features(panel, entity="ticker", time="date",
                              column="ret", features=valid)
```

Multiple value columns produce a `<column>__<feature>` grid:

```python
panel2 = panel.with_columns(vol=pl.col("ret").abs())
multi = pn.extract_features(
    panel2, entity="ticker", time="date",
    features=["absolute_energy", "root_mean_square"],
    column=["ret", "vol"],
)
print(multi.columns)
# ['ticker', 'ret__absolute_energy', 'ret__root_mean_square',
#  'vol__absolute_energy', 'vol__root_mean_square']
```

---

## catch22

Panelary includes a **clean-room, Polars-native** implementation of the *catch22* feature
set (Lubba et al., 2019) — 22 canonical, low-redundancy time-series features. It is
written from the published algorithm descriptions and does **not** vendor the
GPL-licensed `pycatch22` / `hctsa` sources.

### Per-entity catch22

`catch22_features` computes the set independently for each entity's sorted series (again,
leak-safe by construction):

```python
from panelary import catch22

c22 = catch22.catch22_features(
    panel,
    entity="ticker",
    time="date",     # sorts each entity's series ascending before extraction
    column="ret",
)
print(c22.shape)          # (2, 23) -> entity col + 22 features
print(c22.columns[:4])
# ['ticker', 'DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac']

# The 22 canonical names are a public constant
print(len(catch22.CATCH22_NAMES))   # 22
```

Set `catch24=True` to additionally emit the raw mean and standard deviation
(`DN_Mean`, `DN_Spread_Std`), for 24 features in total:

```python
c24 = catch22.catch22_features(panel, entity="ticker", time="date",
                              column="ret", catch24=True)
print(c24.shape)   # (2, 25)
print(catch22.CATCH24_EXTRA_NAMES)   # ('DN_Mean', 'DN_Spread_Std')
```

### Single series and expression forms

For one array there is `catch22_all`; for use inside your own `group_by(...).agg(...)`
there is the `.catch22` expression namespace (remember to sort by time first):

```python
# One 1-D series -> dict[str, float]
single = catch22.catch22_all(rng.normal(0, 1, 100))

# Expression form: one struct column per group, then unnest
c22_expr = (
    panel.sort("ticker", "date")
    .group_by("ticker", maintain_order=True)
    .agg(pl.col("ret").catch22.all())
    .unnest("catch22")
)
```

---

## Per-entity vs. cross-sectional features

Panel data has two axes, and Panelary gives each its own namespace so you never
accidentally mix them.

### `.panel` — down the time axis (per entity)

`.panel` operators transform each entity's series over time. Combine them with
`.over(entity)`:

```python
transformed = panel.with_columns(
    ret_z=pl.col("ret").panel.zscore(window=3).over("ticker"),
)
```

Registered `.panel` operators include `frac_diff` (fractional differencing), `zscore`
(rolling z-score), and `rolling_vol` (the trailing rolling standard deviation) — all
`panel_safe` and `leakage_safe`. `rs_vol` is a deprecated alias of `rolling_vol`: it was
never the Rogers-Satchell estimator, only a rolling standard deviation. Range (OHLC)
volatility — Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang — lives in
[`range_volatility`](ohlc-volatility-and-liquidity.md).

### `.xs` — across the cross-section (per date)

`.xs` operators compare entities *within the same timestamp*. Combine them with
`.over(time)`:

```python
cross = panel.with_columns(
    ret_rank=pl.col("ret").xs.rank(normalize=True).over("date"),
    ret_demean=pl.col("ret").xs.demean().over("date"),
)
```

The `.xs` operators are `rank`, `demean`, `zscore`, `standardize`, `winsorize`,
`quantile_bin`, and `neutralize`. These are leakage-safe (they use no future
information) but are *not* `panel_safe` — by design they read across entities within a
date, which is exactly the cross-sectional comparison you want.

---

## Feature-name provenance and safety

Every registered feature records where it came from and whether it is safe in a leak-free
panel pipeline. This is how Panelary proves it is an independent, permissively licensed
implementation:

```python
for spec in registry.by_namespace("ts")[:3]:
    print(spec.name, "|", spec.source, "|", spec.license,
          "| panel_safe:", spec.panel_safe, "| leakage_safe:", spec.leakage_safe)
# absolute_energy | Panelary | Apache-2.0 | panel_safe: True | leakage_safe: True
# ...

# Provenance / license guard — empty lists mean the catalogue is clean
audit = registry.audit()
print({k: len(v) for k, v in audit.items()})
# {'missing_panel_safe': 6, 'missing_leakage_safe': 0,
#  'missing_provenance': 0, 'non_permissive_license': 0}

print(audit["missing_panel_safe"])
# ['ic', 'orthogonalize', 'portfolio_sort', 'demean', 'rank', 'standardize']
```

`missing_panel_safe` is the only non-empty list, and every entry is there **by
design**: the cross-sectional `.xs` operators and the `.factor` operators read
across entities within a date, which is the whole point of them. Everything in
the catalogue is `leakage_safe`, has recorded provenance, and carries a
permissive license.

---

## Worked example: a wide feature frame

Putting it together — a realistic panel featurised down two axes and enriched with
catch22, producing one row per entity:

```python
import numpy as np
import polars as pl
import panelary as pn
from panelary import catch22

rng = np.random.default_rng(42)
n_dates = 40
tickers = ["AAA", "BBB", "CCC"]

panel = pl.DataFrame({
    "ticker": np.repeat(tickers, n_dates),
    "date":   np.tile(np.arange(n_dates), len(tickers)),
    "ret":    rng.normal(0, 0.02, n_dates * len(tickers)),
}).sort("ticker", "date")

# 1) Add per-entity (.panel) and cross-sectional (.xs) transforms
panel = panel.with_columns(
    ret_z=pl.col("ret").panel.zscore(window=10).over("ticker"),
    ret_rank=pl.col("ret").xs.rank(normalize=True).over("date"),
)

# 2) Bulk-extract a curated set of ts scalar features (one row per ticker)
ts_wide = pn.extract_features(
    panel,
    entity="ticker",
    time="date",
    column="ret",
    features=[
        "absolute_energy", "mean_abs_change", "root_mean_square",
        "variation_coefficient", "cid_ce", "max_drawdown", "return_skew",
        "longest_streak_above_mean",
    ],
)

# 3) Add catch22 features, then join into one wide feature frame
c22_wide = catch22.catch22_features(panel, entity="ticker", time="date", column="ret")

features = ts_wide.join(c22_wide, on="ticker")
print(features.shape)          # (3, 31) -> ticker + 8 ts + 22 catch22
print(features.columns[:5])
# ['ticker', 'absolute_energy', 'mean_abs_change', 'root_mean_square',
#  'variation_coefficient']
```

The result is a model-ready feature matrix: one row per entity, every column produced by
a leak-safe, per-entity computation.

---

## See also

- [Preprocessing](preprocessing.md) — imputation and scaling.
- API reference: [Feature extractors](../api-reference/feature-extractors.md).
