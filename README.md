<div align="center">

# Panelary

**The best methods for panel data.**

*Leak-safe, fast feature engineering and ML for panel data, built on Polars.
Panel-native data science: impute → engineer → label → select → model → validate,
with no lookahead, ever.*

[![PyPI](https://img.shields.io/pypi/v/panelary.svg)](https://pypi.org/project/panelary/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Polars](https://img.shields.io/badge/built%20on-Polars-CD792C.svg)](https://pola.rs/)
[![Docs](https://img.shields.io/badge/docs-sovai--research.github.io-informational.svg)](https://sovai-research.github.io/panelary/)

[Documentation](https://sovai-research.github.io/panelary/) ·
[Quickstart](./docs/quickstart.md) ·
[Leakage & correctness](./docs/leakage.md) ·
[Contributing](./CONTRIBUTING.md)

</div>

---

> **Your backtest is lying to you.** Most panel-data feature engineering leaks the future into the past — a single careless `.shift`, a global `StandardScaler`, a cross-sectional rank computed over the whole sample — and your beautiful Sharpe ratio evaporates in production. Panelary makes leak-safe feature engineering the *default*, not an afterthought.

## What is Panelary?

Panelary is a feature-engineering and machine-learning toolkit for **panel data** — many entities observed over time (stocks, customers, sensors, regions). It is:

- **Panel-first** — the panel (entity × time) is the unit of work, not a bag of rows.
- **Polars-native & lazy** — every transform is a Polars expression; nothing computes until you `.collect()`.
- **sklearn-familiar** — `fit` / `transform` / `Pipeline`, the API you already know.
- **Correct by construction** — every operation is panel-aware and leak-safe. No lookahead, ever.
- **Light to install** — the required footprint is `numpy` + `polars`. Everything heavier is an optional extra, imported lazily at first use.

## Installation

```bash
pip install panelary
```

Or, with [uv](https://docs.astral.sh/uv/) — the same package, much faster:

```bash
uv pip install panelary       # into an existing environment
uv add panelary               # into a uv-managed project
```

Panelary requires **Python 3.10+**. The import name matches the package name:

```python
import panelary as pn
```

### Optional extras

The base install carries only `numpy` and `polars`. Heavier capabilities live behind extras, and a missing one raises an actionable error naming the extra to install — never a bare `ImportError`.

```bash
pip install "panelary[recommended]"   # batteries included: ml, scipy, seasonality, cafe
pip install "panelary[cafe,explain]"  # pick exactly what you need
pip install "panelary[all]"           # every optional feature set
```

| Extra | Unlocks |
| --- | --- |
| `ml` | scikit-learn-backed estimators, selection, reduction, clustering |
| `scipy` | scientific kernels used by feature extraction, catch22, metrics |
| `cafe` | CAFE point-in-time imputation (`CafeImputer` / `cafe_impute`) |
| `explain` | SHAP / Shapley-interaction feature attribution |
| `dimreduce` | UMAP non-linear dimensionality reduction |
| `lightgbm`, `xgboost`, `catboost` | gradient-boosting back-ends |
| `forecasting`, `automl` | the functime-derived forecasting and AutoML paths |
| `seasonality` | holiday/calendar effects (Fourier terms need nothing) |
| `llm` | LLM-assisted panel analysis |
| `viz` | Plotly plotting helpers |
| `gpu` | the Polars GPU engine (`.collect(engine="gpu")`) |
| `ann`, `fast`, `progress` | approximate nearest neighbours, numba JIT, progress bars |
| `recommended`, `all` | curated bundles |

Development and documentation toolchains are `dev` and `docs`; see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## The golden path

Each stage of the workflow has one obvious entry point on the top-level `pn` namespace, so you rarely have to remember which submodule a method lives in. Drop to the underlying classes when you want the full parameter surface.

| Stage | Verb | What it covers |
| --- | --- | --- |
| Fill gaps | `pn.impute` | Point-in-time imputation: CAFE and simpler baselines |
| Engineer | `pn.features` | Bulk and per-column leak-safe feature generation |
| Narrow | `pn.select` | Feature selection under purged cross-validation |
| Compress | `pn.reduce` | Latent factors and dimensionality reduction |
| Group | `pn.cluster` | Time-series and cross-sectional clustering |
| Fit | `pn.regression` | Panel regression and panel-aware ML estimators |
| Explain | `pn.causal` | Causal and econometric panel estimators |
| Detect | `pn.bubbles` | Explosive-behaviour and change-point detection |

Full signatures live in the [documentation](https://sovai-research.github.io/panelary/).

## 60-second quickstart

```python
import polars as pl
import panelary as pn

# A panel is long-format data keyed by (entity, time).
prices = pl.read_parquet("prices.parquet")   # ticker, date, close, volume, ...
panel = pn.PanelFrame(prices, entity="ticker", time="date")

# --- Feature engineering: leak-safe by construction ------------------------
# `.panel` = within-entity, causal (use .over(entity)).
# `.xs`    = cross-sectional, per-date (use .over(time)).
feats = panel.with_columns(
    pl.col("close").panel.frac_diff(d=0.4).over("ticker").alias("close_fd"),
    pl.col("close").panel.zscore(window=21).over("ticker").alias("close_z"),
    pl.col("close").xs.rank(normalize=True).over("date").alias("xs_rank"),
    pl.col("close").xs.zscore().over("date").alias("xs_z"),
)

# --- Labels: triple-barrier (Lopez de Prado, AFML Ch. 3) -------------------
# Trailing volatility, forward scan stops at the first touch: leak-safe.
labels = pn.triple_barrier(
    prices,
    entity="ticker", time="date", price="close",
    pt=2.0, sl=1.0, max_holding=10, vol_lookback=20,
)

# --- Prove it: perturb every future value, assert the past is unchanged ----
pn.assert_no_lookahead(
    pl.col("close").panel.zscore(window=21).over("ticker").alias("z"),
    panel,
)
```

Every step respects two contracts: **`panel_safe`** (within-entity operations stay inside their entity and run in time order) and **`leakage_safe`** (cross-sectional and `fit`-based operations never see the future or the test fold).

The [Quickstart](./docs/quickstart.md) continues from here — bulk feature extraction, CAFE imputation, purged and combinatorial-purged cross-validation, and backtest-overfitting diagnostics — with every block runnable.

## What's inside

| Area | Module |
| --- | --- |
| `PanelFrame`, leak-safe `Pipeline`, `PanelTransformer` / `PanelEstimator` protocols | `pn.core` |
| `.panel` / `.xs` expression and frame namespaces (typed `.pyi` stubs ship in the wheel) | `panelary.namespaces` |
| Panel-native preprocessing: scaling, ranking, neutralization, `frac_diff` | `pn.transform` |
| Point-in-time imputation (CAFE and baselines) | `pn.imputation` |
| Triple-barrier and related leak-safe labeling | `pn.label` |
| Feature selection: MRMR, PFA, correlation, variance, MDI/MDA | `pn.select` |
| Dimensionality reduction: PCA, ICA, hierarchical and robust factors | `pn.reduce` |
| Panel clustering, including k-Shape and cross-sectional | `pn.cluster` |
| Cross-sectional factor evaluation: ICs, neutralization, portfolios | `pn.factor` |
| Econometrics: HDFE, Fama–MacBeth, IVX, DML, connectedness, EVT, HAR-RV | `pn.econ` |
| Leak-safe, panel-aware SHAP attribution | `pn.explain` |
| Explosive-regime, bubble and change-point detection (GSADF/BSADF, CUSUM/FOCuS) | `pn.detect` |
| Purged and combinatorial-purged CV, deflated Sharpe, PBO | `pn.core`, `pn.validation` |
| 100+ Polars-native feature extractors (`ts` namespace), catch22, `extract_features` | `pn.feature_extractors`, `pn.catch22` |
| Forecasting (linear, kNN, GBM, conformal, censored, AutoML) | `pn.forecasting`, `pn.models` |
| Metrics, evaluation, backtesting, seasonality, LLM analysis | `pn.metrics`, `pn.evaluation`, `pn.backtesting`, `pn.seasonality`, `pn.llm` |
| Point-in-time compilation (`causalize`) and borrowed accuracy | `pn.leakage` |
| The leakage verifier and test helpers | `pn.assert_no_lookahead`, `pn.testing` |

Every operator also registers a machine-readable `FeatureSpec` in `pn.registry` carrying its safety contract, provenance and license (79 registered by `import panelary`: `ts`=47, `xs`=11, `econ`=8, `factor`=4, `panel`=4, `label`=3, `sample`=2; 149 once the lazy `covariance` (8), `evolve` (52) and `shape` (10) packages load).

See [`CHANGELOG.md`](./CHANGELOG.md) for release history and [the roadmap](./docs/roadmap.md) for what is next.

## How Panelary fits in the ecosystem

Panelary is opinionated about **panel-native correctness and ML workflow**, and deliberately *depends on* the ecosystem rather than competing with it:

- **`functime`** — Panelary is built on functime's Polars-native feature-extraction and forecasting engine (Apache-2.0), which we reuse and credit. We add the panel object, leakage safety, labeling, and validation on top; we interoperate with functime, we don't replace it.
- **`polars-ds` / `polars_ta`** — we build on and recommend these for general Polars-native data-science and technical-analysis primitives. Panelary focuses on the panel object, leakage safety, labeling and validation that sit *above* them.
- **Nixtla (`statsforecast`, `mlforecast`, …)** — Nixtla owns forecasting. Panelary's centre of gravity is leak-safe feature engineering, labeling, selection, attribution and cross-sectional ML for panels. We interoperate; we don't reinvent forecasting.
- **`tsfresh` / `pycatch22`** — Panelary's extractors are Polars-native and far faster. catch22 features are clean-room reimplemented from Lubba et al. (2019), not vendored from the GPL `pycatch22`.

## Documentation

- [Documentation site](https://sovai-research.github.io/panelary/)
- [Quickstart](./docs/quickstart.md) · [Installation](./docs/installation.md) · [Roadmap](./docs/roadmap.md)
- [Leakage & correctness-by-construction](./docs/leakage.md)
- [Contributing](./CONTRIBUTING.md) · [`AGENTS.md`](./AGENTS.md) (the contract for coding agents)
- [`llms.txt`](./llms.txt) (a compact machine-readable overview)

## License

Panelary is distributed under the **Apache License 2.0**, retained from [functime](https://github.com/functime-org/functime), the upstream this work is derived from — see [`NOTICE`](./NOTICE) and [`LICENSE`](./LICENSE).

CAFE imputation (`cafe_impute` / `CafeImputer`) is powered by [`cafe-impute`](https://pypi.org/project/cafe-impute/) (MIT, Sov.ai), an optional dependency installed via the `cafe` extra.
