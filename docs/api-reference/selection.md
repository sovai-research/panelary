# Feature selection

Leak-safe feature selection for panels. Every routine is computed only on the
rows it is passed (in-fold), so importances and selections are free of
look-ahead when driven through a purged CV splitter.

## What's here

- `mrmr` — minimum-Redundancy-Maximum-Relevance selection, computed in-fold.
- `mda` — Mean-Decrease-Accuracy (permutation) importance evaluated through a
  purged CV splitter.
- `mdi` — Mean-Decrease-Impurity importance from a fitted tree ensemble.
- `refit_shapley` — the exact Shapley value of each feature group (or data
  source) to a model **refit** on it through a purged CV splitter, with an
  explicit training-mean baseline, common random numbers across coalitions,
  optional seed replicates and per-fold decomposition, and opt-in permutation
  sampling beyond 12 groups. Returns a `RefitShapleyReport` (`to_frame()`,
  `to_dict()`, `to_json()`).
- `MRMRSelector` — a `PanelTransformer` wrapping `mrmr` for use as a `"select"`
  step in a `Pipeline`.

## See also

- [Feature Selection guide](../user-guide/selection.md) — the narrative walkthrough.
- [`validation`](validation.md) — the purged splitter `mda` evaluates through.
- [`explain`](explain.md) — attribution, once a model is fitted.
  `refit_shapley` refits per coalition; `explain.joint_group_shap` holds one
  fitted model fixed.
- [`leakage`](leakage.md) — `borrowed_accuracy`, the same Shapley core applied
  to pipeline stages run permissively or point-in-time.

## API

::: panelary.select
