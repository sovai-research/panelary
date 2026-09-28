# embed

Leak-safe numerical embeddings for panel data: fast, CPU-only, pure numpy +
polars. Each public transform subclasses `PanelTransformer` and declares
`panel_safe` / `leakage_safe`, plus the two enforced attributes `fit_is_empty`
(`fit` is handed a zero-row frame) and `is_cross_sectional` (fits per date,
never across dates). Outputs are fixed-size `pl.Array` columns (`Float32` by
default).

Clean-room note: QUANT, Hydra and MiniRocket-PPV were implemented from the
papers. The GPL-3.0 reference implementations and the aeon / sktime / wildboar
ports were not read (see `NOTICE`).

## What's here

| Entry point | Layer | Purpose |
| --- | --- | --- |
| `QuantEmbedder` | expansion | QUANT dyadic-interval quantiles over a trailing window; no fitted state |
| `RandIntC22` | expansion | catch24 over seeded random dilated intervals; no fitted state |
| `HydraEmbedder` | expansion | Hydra competing kernels; no fitted state; amplitude retained |
| `CausalMiniRocket` | expansion | causal rolling MiniRocket-PPV; fitted biases |
| `RandomFourierFeatures` | expansion | RFF / ORF / arc-cosine with mandatory direct links and a past-only bandwidth |
| `TensorSketch` | expansion | polynomial-kernel sketch (CountSketch ⊛ FFT) |
| `CrossRocket` | expansion (per date) | random permutation-equivariant operators across entities |
| `CrossSectionalEmbedder` | per-date mode | per-date PCA with required cross-date alignment |
| `procrustes_rotation` | per-date mode | the orthogonal Procrustes aligner |
| `EmbeddingCompressor` | compaction | `srp \| pca \| svd \| none`, dispatching to `panelary.shape` |
| `PreValidatedRidge` | probe | ridge with exact leave-one-out / purged-fold penalty choice; refuses `alpha=0` |
| `reversal_check`, `mechanical_baseline`, `naive_baselines`, `null_panel`, `null_distribution`, `baseline_report` | guardrails | the section-5 diagnostics |
| `EmbeddingState`, `Embedder` | contract | serialisable state; structural protocol |
| `random_dilated_intervals`, `dyadic_intervals`, `dilation_ladder` | shared | interval and dilation samplers |

## See also

- [Numerical Embeddings guide](../user-guide/embeddings.md): the narrative walkthrough.
- [`shape`](shape.md): the primitive layer that `EmbeddingCompressor` and `TensorSketch` compose.
- [`catch22`](catch22.md): the batched feature kernels behind `RandIntC22`.
- [Leak verifiers](testing.md): the instruments that `tests/test_embed_leakage.py` runs.

## API

::: panelary.embed
