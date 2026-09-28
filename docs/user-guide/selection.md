# Leak-Safe Feature Selection

Feature selection is a place leakage sneaks in quietly: rank your features on the
whole dataset, keep the top-k, and every fold's "test" rows have already voted on
which columns survive. Panelary's selectors in `panelary.select` are built
to be run **inside a fold / on a training set only**, so the choice of features
never benefits from data you are about to score on.

!!! warning "Run selection inside the fold"
    To stay leak-safe, call these selectors on the **training block of each CV
    fold** (or on your single train split), then apply the resulting feature set
    to the matching test block. Selecting once on the full panel and re-using
    that set across folds leaks test-fold structure into training. The
    `MRMRSelector` transformer and `mda` are designed around this: they compute
    only from the rows handed in.

Four complementary functions plus one pipeline transformer:

| Tool | What it does | When to use |
| --- | --- | --- |
| `mrmr(X, y, k, ...)` | Greedy minimum-Redundancy-Maximum-Relevance: features that track the target but not each other. | Fast, model-free shortlist; drops redundant/collinear columns. |
| `mda(estimator, X, y, cv, ...)` | Mean-Decrease-Accuracy (permutation importance) evaluated **through a purged CV splitter**. | Leak-free, model-based importance you can trust for ranking. |
| `mdi(estimator, features)` | Mean-Decrease-Impurity read from a fitted tree ensemble. | Cheap in-sample importance; not leak-free — use for speed/exploration. |
| `refit_shapley(estimator, X, y, cv, groups, ...)` | Shapley value of each feature **group** (or data source) to a model **refit** on it, through a purged CV splitter. | "What is this feature group, or this dataset, worth to my pipeline?" — when inputs are redundant or interact. |
| `MRMRSelector(k, ...)` | `mrmr` wrapped as a pipeline `"select"` step. | Drop selection into a `Pipeline` so it re-fits per fold automatically. |

All of them accept a `PanelFrame` or a bare polars `DataFrame` / `LazyFrame`; for
bare frames pass `entity=` / `time=`.

## Example data

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(0)
rows = []
for e in range(6):
    level = rng.normal()
    for t in range(40):
        mom = rng.normal()
        vol = rng.normal()
        mom_lag = mom * 0.9 + rng.normal() * 0.1   # redundant with `mom`
        noise = rng.normal()                        # irrelevant
        ret_fwd = 2.0 * mom - 1.0 * vol + level + rng.normal() * 0.1
        rows.append(
            {"ticker": f"stock_{e}", "date": t,
             "mom": mom, "vol": vol, "mom_lag": mom_lag,
             "noise": noise, "ret_fwd": ret_fwd}
        )
df = pl.DataFrame(rows)
train = df.filter(pl.col("date") < 30)   # select on the training block only
```

`mom` and `vol` genuinely drive `ret_fwd`; `mom_lag` is a near-duplicate of
`mom`; `noise` is irrelevant. A good selector keeps `mom` and `vol` and discards
the other two.

## mRMR — relevance without redundancy

`mrmr` greedily builds a feature set that is maximally correlated with the target
yet minimally redundant with the features already chosen (Ding & Peng, 2005).
Relevance is the absolute Pearson correlation with the target, computed
Polars-natively; redundancy is the mean absolute correlation with the picks so
far. It returns the selected names **in selection order** (most informative
first). Everything is computed only from the rows in `X`, so there is no global
fit to leak.

```python
from panelary.select import mrmr

selected = mrmr(
    train, "ret_fwd", k=2,
    features=["mom", "vol", "mom_lag", "noise"],
    entity="ticker", time="date",
)
print(selected)   # ['mom', 'vol']
```

`mom_lag` is dropped even though it correlates strongly with the target, because
it is redundant with the already-selected `mom` — exactly the behaviour that
makes mRMR a good de-correlating shortlist. Pass `k` in `1..n_features`; omit
`features=` to consider every numeric non-key, non-target column.

## MDA — leak-free permutation importance

`mda` gives a model-based importance you can trust. For each `(train, test)` fold
from a **purged** splitter, it fits a fresh clone of your estimator on the train
block, scores the held-out block (accuracy for classifiers, R² for regressors),
then permutes each feature in the test block and records the drop in score
(López de Prado, 2018, Ch. 8). Because the splitter purges overlapping label
windows, training and test never share information, so the importances are
leak-free.

```python
from sklearn.ensemble import RandomForestRegressor
from panelary.core.model_selection import PurgedKFold
from panelary.select import mda

cv = PurgedKFold(n_splits=4, embargo=1)
importance = mda(
    RandomForestRegressor(n_estimators=100, random_state=0),
    train, "ret_fwd", cv,
    features=["mom", "vol", "mom_lag", "noise"],
    entity="ticker", time="date",
)
print(importance)
```

```text
shape: (4, 3)
┌─────────┬────────────┬────────────────┐
│ feature ┆ importance ┆ importance_std │
│ ---     ┆ ---        ┆ ---            │
│ str     ┆ f64        ┆ f64            │
╞═════════╪════════════╪════════════════╡
│ mom_lag ┆ …          ┆ …              │
│ vol     ┆ …          ┆ …              │
│ mom     ┆ …          ┆ …              │
│ noise   ┆ …          ┆ …              │
└─────────┴────────────┴────────────────┘
```

The result is a polars `DataFrame` of `feature`, `importance` (mean decrease in
score across folds), and `importance_std` (per-fold stability), sorted by
importance. `noise` lands near zero. The estimator is cloned per fold, so the
instance you pass is left unfitted. Tune with `n_repeats=` (permutation repeats)
and `random_state=`.

!!! note "Redundant features and permutation importance"
    When two features are near-duplicates (`mom` and `mom_lag` here), permutation
    importance can split or inflate their scores — permuting one still leaves its
    twin to carry the signal. This is a known property of MDA on collinear
    inputs, and a reason to run a de-correlating step like `mrmr` first — or to
    ask the question with [`refit_shapley`](#refit-shapley-what-a-feature-group-is-worth),
    which refits instead of permuting.

## Refit Shapley — what a feature group is worth

`mda` asks what one fitted model loses when a column is scrambled. Quants usually
want something else: *what is this feature group — or this dataset — worth to
the pipeline?* Two limits of the permutation answer get in the way:

* **Substitutes mask each other.** Permute `mom`, and `mom_lag` still carries the
  signal, so near-duplicates can both look cheap — or the split between them
  follows whatever the fit happened to do with two collinear columns.
* **Interactions are invisible.** A group that only pays off alongside another
  gets no credit from a one-at-a-time test.

`refit_shapley` plays a cooperative game whose players are **groups of
columns** and whose value `v(S)` is the purged-CV score of the model **refit**
on the columns in `S`. Each group's number is its exact Shapley value in that
game, so three axioms hold for what you care about:

* **Efficiency.** The values add up to `v(all groups) − v(no groups)`.
* **Null player.** A group the learner cannot use gets zero.
* **Symmetry.** Two interchangeable groups get the same value. They share their
  joint worth rather than both scoring zero.

```python
from sklearn.linear_model import LinearRegression
from panelary.core.model_selection import PurgedKFold
from panelary.select import refit_shapley

cv = PurgedKFold(n_splits=4, embargo=1)
report = refit_shapley(
    LinearRegression(), train, "ret_fwd", cv,
    ["mom", "mom_lag", "vol", "noise"],          # one player per column
    entity="ticker", time="date",
)
print(report)
```

```text
Refit Shapley (r2): total +0.789397 over the baseline  (4 groups, exact, 16 coalitions, 60 fits, 4 folds)
  all groups      +0.78033
  baseline   -0.00906688  (train_mean)
    mom          +0.319844   40.5%
    mom_lag       +0.30954   39.2%
    vol          +0.158923   20.1%
    noise      +0.00109034    0.1%
```

The redundant pair splits its worth almost evenly: 0.320 against 0.310. `mda`
on the same model, data and folds gives `mom` 1.87, `mom_lag` 0.16 and `vol`
0.26 in its own units, so its ranking puts `vol` above `mom_lag`. `noise` is
worth nothing under either method. `report.to_frame()` returns the same numbers
as a polars frame (`group`, `n_features`, `phi`, `share`).

**Players are groups.** Pass `{name: [columns]}` to value a factor family or a
vendor's dataset as one player:

```python
report = refit_shapley(
    LinearRegression(), train, "ret_fwd", cv,
    {"momentum": ["mom", "mom_lag"], "vol": ["vol"], "noise": ["noise"]},
    entity="ticker", time="date",
)
# momentum +0.641  vol +0.145  noise +0.003
```

Groups must be disjoint. Choose them on purpose: Shapley values depend on how
the players are defined. Splitting one signal into two duplicate players moves
credit towards the other players. The same data gives `vol` 0.159 as one of
four players and 0.145 as one of three.

### The baseline, and what "worth" is measured against

`v({})` is defined explicitly. With no `base_features`, it predicts each
**training fold's** mean of `y` (the majority class, for a classifier) and fits
nothing. `report.baseline_score` is that score, and every value is measured from
it. To ask what a group adds **on top of** features you already have, pass
`base_features=[...]`. Those columns are in every coalition, including the
baseline, which becomes the model refit on them alone.

A **negative** value is a result, not a bug. Shapley averages a group's marginal
contribution over *every* coalition, including ones you would never deploy. A
fully grown random forest (`RandomForestRegressor(n_estimators=100)`, five
seeds) on these 180 rows scores R² −0.68 on `vol` alone, against a baseline of
−0.01. `vol`'s Shapley value is negative (about −0.08),
even though adding it to `momentum` improves the forest by +0.26. If that
marginal question is the one you mean, ask it directly with
`base_features=["mom", "mom_lag"]`.

### Seeds, periods and many groups

* **`seeds=[0, 1, 2, …]`** reruns the whole game once per seed and sets the
  estimator's `random_state` (or calls a factory `make(seed)`). The seed is the
  **same for every coalition and fold**. These are common random numbers, so fit
  noise cancels in the differences Shapley is built from. The report gives the
  mean, plus `attribution_median` and `attribution_range` across seeds. Each
  seed's values sum to that seed's total.
* **`per_period=True`** also decomposes each fold's own score (`report.per_period`,
  one entry per fold with its test-time span). Shapley is linear, so each fold's
  values sum to that fold's total, and the headline is exactly the mean of the
  per-fold values. A group whose share decays shows up here.
* **More than 12 groups** is refused by default. `2**k` coalitions, each a full
  purged-CV refit, is the cost. First try to use fewer, coarser groups. If you
  cannot, opt in to `method="permutation", n_permutations=M`, which samples
  orderings in antithetic pairs. That costs at most `M(k−1)+2` coalitions, keeps
  efficiency, and reports a Monte-Carlo standard error per group
  (`report.standard_error`). It is never chosen for you.

**Cost.** Coalitions × folds × seeds estimator fits: `2**k` coalitions exactly
(16 for four groups), less the fit-free baseline. Features are never recomputed.
Each fold is collected once and a coalition just selects columns.

**Leak-safety.** Every fit sees one fold's training rows only, through your
purged splitter. The baseline mean is the training fold's. Rows missing any
group's column are dropped from **every** coalition, so all coalitions score the
same rows. The features themselves are taken as given, so they must already be
point-in-time (see [`panelary.leakage`](../api-reference/leakage.md)).

!!! note "What kind of Shapley value this is"
    This is an interventional (do-)Shapley value on the *training procedure*,
    in the sense of Jung et al. (ICML 2022). Every coalition is *executed* by
    refitting, so nothing has to be identified or estimated. It measures what
    the data offers this learner under this CV. It is **not** a causal effect
    of a feature on the target. `explain.joint_group_shap` is the model-fixed
    counterpart: it tells you what *one fitted model* uses. See
    [Feature Attribution](attribution.md#choosing-a-value-function).

## MDI — cheap in-sample importance

`mdi` reads `feature_importances_` off an already-fitted tree ensemble (sklearn
forests / gradient boosters, LightGBM) and pairs each value with its feature
name. It is fast but **in-sample** — not leak-free the way `mda` is. Use it for
quick exploration; prefer `mda` when leak-safety matters.

```python
from panelary.select import mdi

features = ["mom", "vol", "mom_lag", "noise"]
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(train.select(features).to_numpy(), train.get_column("ret_fwd").to_numpy())

print(mdi(rf, features))
```

```text
shape: (4, 2)
┌─────────┬────────────┐
│ feature ┆ importance │
│ ---     ┆ ---        │
│ str     ┆ f64        │
╞═════════╪════════════╡
│ mom_lag ┆ …          │
│ mom     ┆ …          │
│ vol     ┆ …          │
│ noise   ┆ …          │
└─────────┴────────────┘
```

Pass the `features` list in the **same order** it was fed to `estimator.fit`. It
raises `AttributeError` if the estimator has no `feature_importances_`, and
`ValueError` on a length mismatch.

## MRMRSelector — selection as a pipeline step

`MRMRSelector` wraps `mrmr` as a `PanelTransformer` so selection lives inside a
`Pipeline` and re-fits on each fold's training panel automatically — the
leak-safe way to select. It fits by running `mrmr` on the training panel and
remembering the chosen names; it transforms by projecting any panel onto
`(entity, time)` + the selected features (+ the target, when `keep_target=True`,
so a downstream estimator can still see it).

```python
from panelary.select import MRMRSelector

selector = MRMRSelector(
    k=2,
    target="ret_fwd",
    features=["mom", "vol", "mom_lag", "noise"],
    entity="ticker", time="date",
).fit(train)

print(selector.selected_)                 # ['mom', 'vol']

reduced = selector.transform(train).collect()
print(reduced.columns)   # ['ticker', 'date', 'mom', 'vol', 'ret_fwd']
```

Because the feature set is fixed at `fit` time and `transform` never re-selects,
`MRMRSelector` is `panel_safe` and `leakage_safe`. Set `keep_target=False` to
drop the target from the transformed output. Feeding the reduced panel into a
`PanelSklearnRegressor` (see [Panel ML Models](models.md)) gives a compact,
de-correlated model that fits and predicts on the same `(entity, time)` keys.

## References

- Ding, C., & Peng, H. (2005). *Minimum redundancy feature selection from
  microarray gene expression data.* Journal of Bioinformatics and Computational
  Biology, 3(2).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 8
  ("Feature Importance").
- Jung, Y., Kasiviswanathan, S., Tian, J., Janzing, D., Blöbaum, P., &
  Bareinboim, E. (2022). *On measuring causal contributions via
  do-interventions.* ICML 2022, PMLR 162:10476–10501.
