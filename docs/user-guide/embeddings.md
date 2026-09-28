# Numerical Embeddings

`panelary.embed` turns every `(entity, date)` row of a panel into a fixed-width
vector, using only information available at that date. It is CPU-only and pure
numpy + polars: the numerical analogue of static text embeddings. Its selling
point is not a leaderboard rank. Every transform declares, and a test enforces,
**what it is allowed to learn and from which rows**.

```python
import panelary.embed as pe
```

## The four layers

```text
[1 causal window] -> [2 nonlinear expansion] -> [3 compaction] -> [4 probe]
 trailing / per-date  QUANT · RandIntC22 ·       srp · pca · svd    PreValidatedRidge
 (never centred)      Hydra · MiniRocket-PPV ·
                      random features · TensorSketch
```

Layers 2 and 3 are kept separate on purpose. The best expansions are wide
(QUANT ~10 features per window cell), and you usually want 16–512 dimensions.
Compose an expansion with a compressor rather than distorting the expansion.

| Layer | Class | Fitted state | `fit_is_empty` |
| --- | --- | --- | --- |
| 2 | `QuantEmbedder` | none: dyadic intervals are a function of the window length | **True** |
| 2 | `RandIntC22` | none: seeded random intervals | **True** |
| 2 | `HydraEmbedder` | none: seeded random kernels | **True** |
| 2 | `CausalMiniRocket` | biases (quantiles of training responses) | False |
| 2 | `RandomFourierFeatures` | expanding scaler + bandwidth schedule | False |
| 2 | `TensorSketch` | none: seeded CountSketches | **True** |
| 2 (per date) | `CrossRocket`, `CrossSectionalEmbedder` | none; everything is fitted per date at transform time | **True** |
| 3 | `EmbeddingCompressor("srp")` | none (seeded projection) | (True in effect) |
| 3 | `EmbeddingCompressor("pca" / "svd")` | components, centre, scale | False |
| 4 | `PreValidatedRidge` | coefficients, penalty | n/a |

`fit_is_empty = True` is enforced, not documented. `fit` is handed a zero-row
frame, so the transform cannot read a data row. `tests/test_embed_stateless.py`
also checks that fitting on two disjoint datasets gives byte-identical
`get_state()`. Start with those classes: they have no leak surface to get wrong.

## Temporal mode: one embedding per `(entity, date)`

Every temporal embedder reads a **strictly trailing** window of `window`
observations per entity. There is no `center=` argument. Centred windows
inflate results even more than full-sample scalers do.

```python
q = pe.QuantEmbedder(window=64, columns=["ret"], entity="ticker", time="date")
emb = q.fit(train).transform(panel).collect()     # fit reads the schema only
emb.schema["ret_quant"]                            # Array(Float32, shape=(607,))
```

- **Warm-up.** Rows before an entity's first full window are dropped, with an
  `EmbeddingWarmupWarning` naming the entities that had no full window at all.
  Pass `warmup="null"` to keep them as null rows instead.
- **Missing values** inside a window null that row, for that column only.
  Nothing is imputed.
- **Output** is one fixed-size `pl.Array` column per input column (`Float32` by
  default; `dtype="float16"` or `"float64"` on request). All arithmetic is
  float64. Pass `output="columns"` only if you really want thousands of
  top-level columns.
- **Amplitude is retained.** No transform z-normalises windows by default,
  because in finance amplitude *is* volatility. `HydraEmbedder` also appends
  window standard deviation, MAD and realised volatility, and its
  `normalise=True` is exactly per-window z-normalisation.

`RandIntC22` computes catch24 (catch22 plus mean and standard deviation) over
seeded random dilated intervals. The batched catch22 kernels make it
affordable. Features computed over intervals rank well above whole-series
catch22 in the literature.

`CausalMiniRocket` pools a fixed length-9 kernel bank with a per-entity
cumulative sum. Its biases are fitted state, so fit it on the training fold.

## Cross-sectional mode: fit on date *t*, apply to date *t*

All of date *t* is observable at *t*, so a per-date fit is leak-safe by
construction. `CrossSectionalEmbedder` takes the principal directions of each
date's cross-section. Its inputs can be characteristics or a temporal
embedding.

Per-date components are only identified up to rotation (and sign, and order)
across dates. Emitted unaligned, they would be a time series whose axes spin
at random. So `align` has **no default**:

```python
xs = pe.CrossSectionalEmbedder(align="procrustes", n_components=3,
                               columns=["value", "momentum", "quality"],
                               entity="ticker", time="date")
```

- `align="procrustes"` rotates date *t*'s loadings onto date *t−1*'s aligned
  loadings (`pe.procrustes_rotation`).
- `align="link"` does the same and adds the `‖z_t − z_{t−1}‖²` link penalty
  (`link_penalty=λ`).

Both run forward through the frame, so transform the whole panel once and
split afterwards. `CrossRocket` is the random-operator counterpart:
permutation-equivariant operators across entities at each date.

## Compaction

```python
comp = pe.EmbeddingCompressor("srp", dim=128)       # stateless given the seed
comp = pe.EmbeddingCompressor("pca", dim=16)        # fit on the training fold
```

The compressor owns no linear algebra: `srp` is
`panelary.shape`'s sparse random projection, and `pca` / `svd` are its
randomized PCA (standardised or centred-only). Be honest about the dimension.
**The Johnson–Lindenstrauss bound does not license `dim=64`**: at *n* = 10⁶
and ε = 0.1 it asks for about 10⁴ dimensions. Small outputs rest on empirical
performance, not on the theorem. Use `dim≈512` for features and `dim≈16` for
distance-based use such as clustering.

## Random features, with the finance guardrails built in

`RandomFourierFeatures` (Gaussian or arc-cosine, optionally orthogonal)
always prepends the standardised inputs. These are the RVFL direct links, and
there is no switch to remove them. The scaler and the median-heuristic
bandwidth are **expanding** statistics: a row at date *t* uses fitted rows at
dates ≤ *t* only. Even a full-sample fit therefore leaks nothing into row *t*.
It uses a bank of bandwidths (0.25×–4× the heuristic), not one σ.

## The readout: PreValidated Ridge

```python
probe = pe.PreValidatedRidge()                    # alpha grid 1e-3 .. 1e4
probe.fit_frame(train_emb, features=["ret_quant"], target="fwd_ret", time="date")
```

One SVD gives the exact leave-one-out prediction for every penalty on the
grid, so selecting the penalty costs no refits. For classification, the
leave-one-out scores also calibrate the probabilities.

- `alpha = 0` is **refused**. More random features than observations, with no
  penalty, degenerates mechanically into volatility-timed momentum
  (Nagel, 2025).
- An `OverparameterisedReadoutWarning` fires when features outnumber the
  training **dates**.
- Leave-one-out is optimistic when rows are dependent. For panels, pass
  `cv=5, purge=…` (contiguous date blocks) or a `PurgedKFold`.

## Guardrails: run them before believing a number

| Function | Question it answers |
| --- | --- |
| `reversal_check(fit_predict, panel, return_col=…)` | Does the pipeline learn a strongly mean-reverting target, or time momentum mechanically? |
| `mechanical_baseline(panel, target)` | Does the model beat a recency-weighted, inverse-vol average of past targets? |
| `naive_baselines(panel, target, season=…)` | Does it beat the random walk, and the seasonal naive? |
| `null_panel` / `null_distribution` | How does the score look on sign-flipped (martingale-difference) or phase-randomised panels of the same shape and missingness? |
| `baseline_report(frame, target=…, predictions=…)` | All of the above, side by side: IC, IC t-stat and OOS R². |

Deflated Sharpe ratios and PBO are **search-intensity corrections, not leakage
detectors**: a leaky oracle at Sharpe 35 passes both. Leakage is excluded
structurally here, by the contracts above and by
`tests/test_embed_leakage.py` and `tests/test_embed_prefix_invariance.py`.
Each of those files pairs every transform with a deliberately leaky variant
that must fail.

## Reproducibility

```python
state = q.get_state()           # EmbeddingState: method, version, params, columns, arrays
state.save("quant.npz")         # no pickle
q2 = pe.QuantEmbedder.from_state(pe.EmbeddingState.load("quant.npz"))
state.fingerprint()             # SHA-256 over all of it
```

The same state and the same input give the same output: bit-for-bit at
`float64` on one platform, and to storage precision at `float32` / `float16`.
Every temporal kernel computes a row from that row's window alone, never
through a BLAS call whose blocking could depend on the batch. So a window's
embedding does not depend on what else is scored alongside it.

## Measured cost (this repository, 2026-09-28)

These were measured single-threaded on the development machine with a
`window=64` univariate panel, including sorting and output assembly.

| Transform | Width | Windows/s | Peak RSS |
| --- | --- | --- | --- |
| `QuantEmbedder(window=64)` | 607 | ~62,000 | 138 MB @ 10k, 380 MB @ 100k windows |
| `QuantEmbedder(window=128)` | 1,130 | ~27,000 | 230 MB @ 10k |
| `HydraEmbedder(window=64)` | 771 | ~8,000 | 541 MB @ 100k |
| `RandIntC22(window=64, n_intervals=16)` | 384 | ~2,900 | 265 MB @ 10k |
| `catch22_batch`, L=128 | 22 | ~12,000 | — |

Not yet measured: throughput at 10⁶ windows, linear-probe AUROC, k-NN recall,
robustness and out-of-domain transfer. No published head-to-head of these
embeddings against hand-crafted rolling features on a financial cross-section
exists. Every ranking in the literature comes from sensor and ECG archives.
Treat the methods as priors and benchmark them on your own panel.
