# Quickstart

**Panelary** is leak-safe, fast feature engineering and ML for panel data, built on
[Polars](https://pola.rs/). This is a runnable 5-minute tour: build a panel, engineer
leak-safe features, extract a feature matrix, label events, impute gaps, and validate a model
with combinatorial purged cross-validation — proving no lookahead along the way.

The canonical import is `import panelary as pn`. Every code block below runs against the
synthetic panel we build in step 1 — copy them in order.

## Install

```bash
pip install panelary           # core: numpy + polars only
pip install 'panelary[cafe]'   # + CAFE imputation (step 5)
```

See [Installation](./installation.md) for all extras and version notes.

## 1. Make a panel

A **panel** is long-format data keyed by `(entity, time)`: many entities observed over time.
Wrap it in a [`PanelFrame`](./index.md) — a typed, lazy view that validates the keys but
computes nothing until you `.collect()`.

```python
import numpy as np
import polars as pl
import panelary as pn

rng = np.random.default_rng(0)
rows = []
for ticker, base in [("AAA", 100.0), ("BBB", 50.0)]:
    px = base
    for day in range(60):
        px *= 1 + rng.normal(0, 0.02)
        rows.append((ticker, day, px, float(rng.integers(1_000, 5_000))))

prices = pl.DataFrame(rows, schema=["ticker", "day", "close", "volume"], orient="row")
panel = pn.PanelFrame(prices, entity="ticker", time="day")

print(panel.feature_cols)          # ['close', 'volume']
print(panel.collect().shape)       # (120, 4)
```

## 2. Engineer leak-safe features (`.panel` / `.xs`)

Panelary registers two Polars expression namespaces:

- **`.panel`** — *within-entity, causal* transforms (use `.over(entity)`): `frac_diff`,
  `zscore` (trailing rolling), `rolling_vol` (trailing rolling standard deviation).
- **`.xs`** — *cross-sectional* transforms (use `.over(time)`): `rank`, `demean`, `zscore`,
  `winsorize`, `quantile_bin`, `neutralize`.

Because they are ordinary Polars expressions, you compose them straight into
`PanelFrame.with_columns`:

```python
feats = panel.with_columns(
    # within-entity, causal: fractional differencing per ticker, in time order
    pl.col("close").panel.frac_diff(d=0.4).over("ticker").alias("close_fd"),
    # within-entity, causal: trailing rolling z-score
    pl.col("close").panel.zscore(window=10).over("ticker").alias("close_z"),
    # cross-sectional: rank each day across all tickers, rescaled to (0, 1)
    pl.col("close").xs.rank(normalize=True).over("day").alias("xs_rank"),
    # cross-sectional: z-score each day across the cross-section
    pl.col("close").xs.zscore().over("day").alias("xs_z"),
)
print(feats.collect().columns)
# ['ticker', 'day', 'close', 'volume', 'close_fd', 'close_z', 'xs_rank', 'xs_z']
```

!!! tip "Frame-level namespaces"
    The same operators exist on a bare `DataFrame` / `LazyFrame`, where the entity/time key is
    a **required** keyword (omitting it raises — it would leak one entity's history into the
    next):

    ```python
    prices.panel.frac_diff("close", d=0.4, over="ticker", suffix="_fd")
    prices.xs.rank("close", over="day", normalize=True, suffix="_rank")
    ```

## 3. Extract a feature matrix (`extract_features`)

`extract_features` is a lazy, tsfresh-style bulk extractor: one `group_by(entity).agg(...)`
producing exactly one row per entity. Every feature is a pure per-entity aggregation, so
nothing leaks across entities or from the future.

```python
summary = pn.extract_features(
    prices,
    entity="ticker",
    time="day",
    column="close",
    features=["mean_change", "absolute_energy", "longest_streak_above_mean"],
)
print(summary)
# shape: (2, 4) — one row per ticker, one column per feature
```

Pass `features="all"` for every registered `ts` scalar aggregation, or several `column=`
names to featurise them all (output columns become `<column>__<feature>`).

## 4. Label events (`triple_barrier`)

The triple-barrier method (López de Prado, *AFML* Ch. 3) labels each observation by which
barrier it hits first: a profit-take (`+1`) or stop-loss (`-1`) scaled by *trailing*
volatility, or the vertical time barrier (`0`). The volatility estimate is trailing and the
forward scan stops at the first touch, so labels are leak-safe.

```python
labels = pn.triple_barrier(
    prices,
    entity="ticker",
    time="day",
    price="close",
    pt=2.0,            # profit-take = 2× trailing vol
    sl=1.0,            # stop-loss  = 1× trailing vol
    max_holding=5,     # vertical barrier: 5 steps per entity
    vol_lookback=10,   # trailing window for the vol estimate
)
print(labels.select("ticker", "day", "label", "ret", "t1").head())
```

The result adds `label` (`+1` / `-1` / `0`), `ret` (realized return to the touched barrier),
and `t1` (the time the label resolves) — `t1` is exactly what the purged CV splitters below
use to purge overlapping events.

## 5. Impute gaps leak-safely (CAFE)

CAFE is a strictly point-in-time (no look-ahead) imputer: each filled cell uses only past and
contemporaneous information within its entity, so it is safe inside walk-forward CV. Use the
`CafeImputer` transformer (requires the `cafe` extra):

```python
# introduce some gaps, then fill them
gappy = prices.with_columns(
    pl.when(pl.arange(0, pl.len()) % 7 == 0)
    .then(None)
    .otherwise(pl.col("close"))
    .alias("close")
)
gp = pn.PanelFrame(gappy, entity="ticker", time="day")

imputer = pn.CafeImputer(engine="joint")
filled = imputer.fit(gp).transform(gp).collect()

print(gappy["close"].null_count(), "->", filled["close"].null_count())   # 18 -> 0
```

!!! note "Function form"
    A pipe-friendly transformer factory is also available:
    `gappy.pipe(pn.cafe_impute(engine="joint")).collect()`. Both share the same leak-safe
    engine; `CafeImputer` additionally slots into a Panelary `Pipeline`.

## 6. Prove no lookahead (`assert_no_lookahead`)

`assert_no_lookahead` splits the time axis, corrupts *every* future value, and asserts that
every past output cell is bit-identical — a direct future-perturbation test of leak-safety.

```python
# A causal op passes silently.
pn.assert_no_lookahead(
    pl.col("close").panel.zscore(window=5).over("ticker").alias("z"),
    panel,
)

# A forward-looking op is caught.
try:
    pn.assert_no_lookahead(
        pl.col("close").shift(-1).over("ticker").alias("lead"),
        panel,
    )
except AssertionError as exc:
    print("caught leak:", str(exc).splitlines()[0])
```

## 7. Validate leak-safely (CPCV)

Combinatorial Purged Cross-Validation (CPCV) trains and tests over multiple group
combinations, **purges** training observations whose events overlap the test window, and
applies an **embargo** around fold boundaries — then reconstructs backtest paths and reports
overfitting diagnostics (Deflated Sharpe, PBO).

```python
# Build a small supervised panel: two features + a forward-return target.
train = panel.with_columns(
    pl.col("close").panel.zscore(window=5).over("ticker").alias("x1"),
    pl.col("close").pct_change().over("ticker").alias("x2"),
    pl.col("close").pct_change().shift(-1).over("ticker").alias("target"),
).collect().drop_nulls()

from panelary.models import PanelSklearnRegressor

est = PanelSklearnRegressor(target="target", features=["x1", "x2"])

report = pn.validate.cpcv(
    est, train, "target",
    n_groups=6, n_test_groups=2, embargo=1,
    entity="ticker", time="day",
)
print(report.summary())
# {'metric': 'neg_mean_squared_error', 'n_splits': 15, ...,
#  'n_paths': 5, 'deflated_sharpe': ..., 'pbo': ...}
```

Prefer a plain purged K-fold? Build the splitter and pass it to `cross_validate`:

```python
from panelary import PurgedKFold

cv = PurgedKFold(n_splits=4, embargo=1)
report = pn.cross_validate(est, train, "target", cv, entity="ticker", time="day")
print(report.summary())
```

Both accept any sklearn-shaped estimator (`fit(X, y)` / `predict(X)`) as well as Panelary
estimators and `Pipeline`s; the estimator is deep-copied per fold so folds stay independent.

## Where to next

- **[Leakage & correctness-by-construction](./leakage.md)** — the `panel_safe` /
  `leakage_safe` contracts in depth.
- **[Benchmarks](./benchmarks/vs_pandas.md)** — the 10–19× vs pandas numbers and methodology.
- **[The golden path](./index.md#the-golden-path)** — the top-level verbs (`pn.impute`,
  `pn.features`, `pn.select`, `pn.reduce`, `pn.cluster`, `pn.regression`, `pn.causal`,
  `pn.bubbles`) that front each stage of the workflow.
- Feature selection (`pn.select.mrmr` / `mda` / `mdi`) and panel models
  (`pn.models.PanelLGBMClassifier`, …) for building the rest of the pipeline.
