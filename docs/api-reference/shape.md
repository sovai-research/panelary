# Shape algebra

`panelary.shape` holds transforms that change the *shape* of a panel: the width
of a named axis, the order of the tensor, or the numerical rank of a
factorisation. Each one declares an **intent** (`COMPRESS`, `LIFT`,
`FACTORIZE` or `SKETCH`) and an **axis** (`feature`, `time`, `entity` or
`lag`). The axis is the leak contract. Time-axis transforms are **trailing** by
default and one row per input row. `whole_series` is an explicit opt-in that is
`leakage_safe = False`. The module is pure numpy + polars.

The package initialiser is lazy. `import panelary` loads no transform module,
and the `shape` operator catalogue (in `panelary.registry`) is registered the
first time a public name is read from `panelary.shape`.

See the [Shape algebra guide](../user-guide/shape.md) for the narrative.

## What's here

| Your problem | Entry point |
| --- | --- |
| Long panel ↔ dense `(entity, time, feature)` tensor | `build_tensor`, `to_long`, `build_sequences`, `Ragged` |
| Strictly trailing windows, batched over the panel | `trailing_windows`, `PanelWindows` |
| Many loose columns ↔ one `pl.Array` column | `as_embedding`, `explode_embedding` |
| PCA without scikit-learn | `RandomizedPCA`, `randomized_svd` |
| Seeded, data-independent compression | `SparseRandomProjection`, `SRHT`, `CountSketch` |
| Compress onto **real** columns | `ColumnSubset`, `CUR`, `interpolative`, `pivoted_qr` |
| Streaming / mergeable Gram sketch | `FrequentDirections` |
| Segment statistics of each row's trailing window | `PAA` |
| Frequency content of each row's trailing window | `Spectral` |
| Lagged copies (delay / Hankel embedding) | `Delay` |
| Compress time and feature modes of a fixed sample | `PartialTucker`, `partial_tucker`, `hosvd` |
| Per-date PCA across entities, aligned across dates | `CrossSectionalRandomizedPCA` |
| What a component is made of; is it stable across folds? | `explain()`, `stability`, `procrustes` |
| Refuse an oversized step before it allocates | `plan()`, `plan_chain`, `ShapeBudgetError` |

## See also

- [`reduce`](reduce.md): row-wise reducers and latent factors. `PanelPCA` /
  `PanelSVD` accept `backend="numpy"`, and `PanelRandomProjection` accepts
  `method="sparse" | "srht"`, all routed here.
- [`registry`](registry.md): every catalogued shape transform carries `intent`,
  `axis`, `flavour`, `invertible`, `streaming` and `cost_hint`.
- [Leak-safety](../concepts/leak-safety.md).

## API

### Vocabulary and base class

::: panelary.shape.Intent

::: panelary.shape.Axis

::: panelary.shape.Flavour

::: panelary.shape.ShapeSpec

::: panelary.shape.ShapeTransform

::: panelary.shape.Plan

::: panelary.shape.ShapeBudgetError

::: panelary.shape.plan_chain

### The spine

::: panelary.shape.build_tensor

::: panelary.shape.to_long

::: panelary.shape.build_sequences

::: panelary.shape.Ragged

::: panelary.shape.trailing_windows

::: panelary.shape.PanelWindows

::: panelary.shape.as_embedding

::: panelary.shape.explode_embedding

### Feature axis

::: panelary.shape.randomized_svd

::: panelary.shape.RandomizedPCA

::: panelary.shape.SparseRandomProjection

::: panelary.shape.SRHT

::: panelary.shape.CountSketch

::: panelary.shape.FrequentDirections

::: panelary.shape.pivoted_qr

::: panelary.shape.interpolative

::: panelary.shape.ColumnSubset

::: panelary.shape.CUR

### Time axis

::: panelary.shape.PAA

::: panelary.shape.Spectral

::: panelary.shape.Delay

### Tensor modes

::: panelary.shape.PartialTucker

::: panelary.shape.partial_tucker

::: panelary.shape.hosvd

### Entity axis

::: panelary.shape.CrossSectionalRandomizedPCA

### Explainability

::: panelary.shape.stability

::: panelary.shape.procrustes
