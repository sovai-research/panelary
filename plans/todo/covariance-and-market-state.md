# `panelary/covariance/` — build contract: as-of covariance estimators, RMT cleaning, eigen-fragility and cross-sectional market state

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): M1–M4 implemented; M5–M7 not started.** Shipped in
> `panelary/covariance/` (lazy): the estimators and `CovEstimate` (M1), the as-of engine
> `rolling` / `market_state` / `market_loading` and `core/_schedule.py` (M2), `turbulence`,
> the transformers, `avg_correlation` and `common_idio_vol` (M3), and the `.xs`
> distribution ops, Kelly–Jiang, `xs_wasserstein` and `avg_skewness` (M4); `psd_repair`
> moved to `_internal/_linalg.py`. Measured findings: exact LW2020 series removes most of
> its numerical sensitivity (8.6e-7 → 7e-14), and on a smooth spectrum plain Ledoit–Wolf
> tracks the oracle better than QIS, so the QIS default needs the M5 harness before it is
> locked. `market_state` daily at N = 3,000 measured over budget under machine load.
> Remaining: M5 (Gerber, pairwise repair, EWMA, `OnlineCovariance`, accuracy harness;
> traps T7, T8, T15), M6 (gated extras), M7 (multivariate realized kernel, per note D7).

As-of covariance and correlation estimation on trailing windows, per date, for
panels of N = 50 to 5,000 assets. On top of that engine sit the per-date market-state
features that are functions of the cross-asset second-moment matrix (absorption
ratio, turbulence, effective rank, Marchenko–Pastur signal count, market-mode
localisation, average correlation, common idiosyncratic volatility). Alongside it
sit the single-date cross-sectional distribution features (dispersion, tail
index, up-share, entropy), which live in the `.xs` namespace.

Pure **numpy + polars** on every default path. No scipy, sklearn, numba or
compiled code in any import the default path reaches. Oracles (`sklearn`,
`skfolio`, `pyRMT`†, fixtures from the Ledoit–Wolf reference code) are
**test-only**, behind `pytest.importorskip` or committed as fixtures with provenance.

Reuse; do not reinvent (all verified by reading the code on 2026-09-29):
`panelary.shape._tensor.build_tensor` (the long→dense boundary),
`panelary.shape._rsvd.randomized_svd` / `_orthonormalize` (top-k, CholeskyQR2),
`panelary.reduce._common` (`top_eigenvectors`, `fix_signs`, `sign_of_max_abs`),
`panelary.reduce._n_factors` (`bai_ng`, `eigenvalue_ratio`),
`panelary.depend._matrix.psd_repair`, `panelary.detect._panel.residualise`,
`panelary.econ.features._evt.hill_index`, `panelary.econ.features._common.rolling_beta`,
`panelary.namespaces.xs` (`_expr_*` builders, `.over(time)` contract),
`panelary.testing` (`assert_prefix_invariant`, `assert_no_lookahead`),
`panelary.synth.generate_panel` (planted factors, Markov regimes, entity entry and exit),
`panelary.registry` (`FeatureSpec`, `safe_scope`),
`panelary._internal._deps.require`.

---

## 1. Why this exists

Three claims, each measured on this machine (Apple M5 Pro, 15 cores, NumPy 2.5.3 +
Accelerate, polars 1.44.2, scikit-learn 1.9.0; scripts in the session scratchpad,
to be promoted to `benchmarks/bench_covariance.py` in M2).

### 1.1 The textbook estimator is silently wrong in exactly the regime equity panels live in

Out-of-sample global-minimum-variance (GMV) portfolio. The metric is the true variance
of the GMV weights built from each estimator, divided by the oracle minimum
(1.0 = perfect). The population has 1 market factor, 8 sector factors and
heterogeneous idiosyncratic variance. Innovations are Student-t(5). W = 252 daily
observations, median of 30 draws.

| Estimator | N=100 (q=0.40) | N=500 (q=1.98) |
|---|---|---|
| sample covariance (`np.cov`; `pinv` when singular) | 1.69 | **2,095** |
| Ledoit–Wolf 2004, identity target (sklearn) | 1.41 | 2.52 |
| OAS (sklearn) | 1.43 | 2.54 |
| Ledoit–Wolf 2004, constant-correlation target | 1.45 | 2.28 |
| Marchenko–Pastur clipping, correlation space | 1.21–1.23 | 1.32–1.34 |
| LW 2020 analytical nonlinear, covariance space | 1.35 | 2.05–2.21 |
| **LW 2020 analytical nonlinear, correlation space** | **1.19** | **1.28–1.31** |
| **QIS (LW 2022), correlation space** | **1.19** | **1.31** |
| PCA factor model, k=9, + diagonal | 1.20 | 1.30 |
| PCA factor model, k=1, + diagonal | 1.13 | 1.30 |

Nothing in the sample-covariance row raises an error. At N > W the matrix is
singular. `pinv` returns weights, and those weights carry **2,095×** the minimum
variance. That is the "silently wrong" failure mode the AgenticFinance engine tests for.

No single estimator dominates: a 1-factor model wins at N=100 *in this
factor-structured population*. The product decision therefore has two parts:
- ship the suite with one minimax-robust default (QIS in correlation space);
- ship a walk-forward OOS harness that makes the choice evidence-based (§8.3).

### 1.2 Per-date use is infeasible the way existing libraries are built

| Operation | Measured | Consequence |
|---|---|---|
| `np.linalg.eigh`, N×N | N=500: 10.7 ms · N=1,000: 52 ms · **N=3,000: 1,854 ms** | Daily primal spectra at N=3,000 × 5,000 dates = **2.6 h**. |
| **dual** W×W Gram + `eigh`, W=252 | N=500: 2.8 ms · **N=3,000: 3.3 ms** | Same spectrum (nonzero part) **560× cheaper**. The whole design turns on this. |
| `eigvalsh` vs `eigh`, 252×252 | 1.5 ms vs 2.5 ms | Use `eigvalsh` whenever vectors are not needed. |
| batched `eigh` 64×(252×252) vs a Python loop | 169.6 ms vs 161.2 ms | **Batching buys nothing** on Accelerate. The per-date loop is not the cost. |
| thread pool over dates (2/4/8 threads) | ×1.1 / ×1.2 / ×1.3 | Not worth a scheduler in M1 (Accelerate already threads GEMM). |
| `sklearn.covariance.LedoitWolf().fit`, N=3,000, W=252 | **4,796 ms** | Per-date loop = 6.7 h. |
| LW intensity from the dual Gram, N=3,000 | **1.36 ms**, equal to sklearn's to 2.7e-16 | Closed-form intensities never need the N×N matrix. |
| LW 2020 in dual (diagonal-plus-low-rank) form, N=3,000 | **5.2 ms** / estimate; primal–dual max rel. dev. 3.6e-9 at N=500 | Full nonlinear-shrinkage estimate without an N×N eigendecomposition. |
| Mahalanobis via Woodbury on that form vs dense solve | rel. err 2.4e-14 | Turbulence in O(N·r), never O(N³). |
| rank-2 update vs full recompute of XᵀX, N=3,000, W=252 | 2.6 ms vs 9.5 ms | Rank-1 updates save <10% of a per-date cost dominated by the eigensolve. They are **not** the lever for the batch path. |
| W=1,260 (5y), N=3,000: Gram / `eigvalsh` / rSVD k=10 | 18 / 110 / 33 ms | Long windows need stride or top-k-only feature sets. |
| W=63, N=3,000: Gram + `eigvalsh` | 0.20 ms | Short windows are nearly free. |

Memory makes the same point. A T×N×N tensor is 10 GB at N=500 and **360 GB** at
N=3,000 (T=5,000). Even the (T, W, N) window tensor is 30 GB at N=3,000. Nothing in
this module may materialise either.

### 1.3 The published recipes leak, and the common implementations inherit it

Kritzman & Li's turbulence is defined with the **full-sample** mean and covariance.
The absorption-ratio shift is routinely z-scored over the full sample. "Month-end
refits" need to know tomorrow's date. The Kelly–Jiang threshold is estimated over the
*calendar* month that contains t. The existing `build_tensor` default
`forward_fill=True` would repeat a delisted stock's last return forever. §6 names 16
such traps; each gets a test that must fail when the trap is re-introduced.

### 1.4 Numerical findings that constrain the design

- The LW 2020 kernel map amplifies a 1e-14 relative perturbation of the eigenvalues
  to **2.5e-8** in the shrunk eigenvalues. The cause is the log singularity of the
  Epanechnikov Hilbert transform at |x|=√5. **QIS amplifies the same perturbation
  only to 1.4e-13** (a rational kernel), with equal accuracy (§1.1). QIS is therefore
  the default nonlinear engine.
- Polars `rolling_var` has max relative error 1.4e-15 on zero-mean data, **1.8e-10**
  at a 1e4 offset and **1.8e-6** at 1e8. Any co-moment path must use shifted data
  (Chan–Golub–LeVeque) or a two-pass window.
- Raw RIE (η = N^-½, no regularisation) scored 1.50 / **23.2** on §1.1, against 1.19 /
  1.31 for QIS. RIE ships only behind a measured gate (M6).

---

## 2. What already exists — and what this must NOT duplicate

| Where (verified) | What it is | Relationship |
|---|---|---|
| `shape/_tensor.py::build_tensor` | long panel → dense (entity, time, value) tensor, causal ffill + expanding z-norm | **Reuse** as the only long→wide path, called with `forward_fill=False, z_normalize=False` (trap T5). Do not write a fourth pivot. `StatisticalFactors._wide_matrix` and `econ/_connectedness.py::_pivot` already duplicate it. If the cross-join grid costs >2 s at 15M cells, the fix is a scatter path *inside* `build_tensor` (shape owner), not here. |
| `shape/_rsvd.py::randomized_svd`, `_orthonormalize` | HMT 2011 rSVD with CholeskyQR2 | **Reuse** for top-k-only features at W ≥ 1,000 (measured 3.4× over the full `eigvalsh`). Reuse `_orthonormalize` to re-orthonormalise dual eigenvectors if ‖VᵀV−I‖ > 1e-10 (measured 1.3e-14, so a guard only). |
| `reduce/_common.py::top_eigenvectors`, `fix_signs`, `sign_of_max_abs`, `covariance` | eigh helpers, sign convention | **Reuse** the sign convention for components ≥2. The market mode is oriented by Σv₁>0 instead (documented deviation: "market up" must be the positive direction). |
| `reduce/_n_factors.py::bai_ng`, `eigenvalue_ratio` | factor-count selectors on a matrix / a spectrum | **Reuse** `eigenvalue_ratio` on the per-window spectrum directly. Add a private `_bai_ng_from_spectrum(sv2, n, p, kmax, criterion)` in `_n_factors.py` so `bai_ng` does not redo an SVD we already have (additive refactor; `bai_ng` delegates to it). |
| `reduce/factors.py::StatisticalFactors`, `reduce/xs.py::CrossSectionalPCA`, `shape/_rsvd.py::CrossSectionalRandomizedPCA` | train-fit global loadings / per-date feature-axis PCA | **Different object.** They reduce *features*, or fit loadings once. We estimate the *entity×entity* second-moment matrix per date on a trailing window. No duplication; the docs cross-link them. |
| `detect/_panel.py::residualise` | trailing PCA residuals, betas frozen on an index-anchored refit schedule, prefix-invariant | **Reuse** for common idiosyncratic volatility and Kelly–Jiang residual mode. Its refit-schedule convention is the one we adopt (§5.3). |
| `depend/_matrix.py::psd_repair`, `to_distance`, `dependence_matrix` | clip-to-PSD + unit diagonal; distances; **feature×feature** dependence matrices pooled over entities | **Reuse** `psd_repair`. M1 moves it to `panelary/_internal/_linalg.py` with a re-export from `depend._matrix` (one-line change; the depend tests are the regression). `dependence_matrix` is a feature-axis object, not an as-of asset covariance, so there is no overlap. |
| `econ/_panel.py::CrossSectionalAverages`, `_cross_sectional_means`, `pesaran_cd` | per-date means via `.over(time)`; whole-sample CD test (O(N²) Python double loop) | Our xs ops follow the `CrossSectionalAverages` pattern. `pesaran_cd` is a whole-sample *test*, while `avg_correlation` is a rolling *feature*. Do not re-implement CD. Follow-on note for the econ owner: on balanced panels, CD = √(2T/(N(N−1)))·Σρᵢⱼ, which is O(NT) via the identity in §5.14. |
| `econ/features/_evt.py::hill_index` | Hill α on one sample, k = max(10, ⌊0.1n⌋) | **Parity oracle** for `xs.tail_index` (matched k). The Polars-native path is for speed. |
| `econ/features/_common.py::rolling_beta` | per-entity trailing univariate beta, native rolling moments | **Reuse** for Kelly–Jiang tail betas (regressor = lagged tail index). |
| `feature_extractors` `return_skew` | whole-series **scalar** skew (`safe_scope="window"`) | Not usable per row. Average skewness (JZZ 2019) is composed from Polars `rolling_skew(...).over(entity)` then `.mean().over(time)`. |
| `namespaces/xs.py` | `rank`, `standardize`, `demean`, `zscore`, `winsorize`, `quantile_bin`, `neutralize` | **Extend** additively with 4 ops (§4.4). `xs.winsorize` is the pre-winsorisation robust option. |
| `econ/_connectedness.py::rolling_connectedness` | per-date VAR/FEVD on a trailing window (Python loop, re-pivots) | Different statistic. Its per-date loop is the pattern we improve on (compact universe, reuse the window, no per-date DataFrame). |
| `shape/_sketch.py` Frequent Directions | mergeable sketch of a feature Gram | Not a covariance estimator for assets. No overlap. |
| anywhere: Ledoit–Wolf, OAS, Marchenko–Pastur, absorption ratio, turbulence, Mahalanobis, denoise, Gerber, effective rank | — | **grep: none exist** (the only hits are prose in `reduce/_hfa.py`). |

---

## 3. Scope and non-goals

**In scope:**
- **Estimators:** sample; EWMA (finite-window and recursive); linear shrinkage to identity / diagonal / constant-correlation / single-index (LW 2003, 2004a, 2004b); OAS (Chen et al. 2010); nonlinear shrinkage: QIS (default) and LW 2020 analytical; Marchenko–Pastur clipping (constant residual eigenvalue, targeted shrinkage) and detoning (López de Prado 2020); PCA factor model + diagonal (Woodbury); Gerber statistic; RIE (gated, M6).
- **Missing-data policies:** listwise-by-entity, zero-after-demean, pairwise + repair.
- **Estimate object:** one as-of covariance object with `solve`, `inv_quad`, `logdet`, `cond`, `gmv_weights`, `to_dense`, `corr`.
- **Market-state features:** absorption ratio and ΔAR, λ₁ share, effective rank / eigen-entropy / participation ratio, MP signal count, market-mode IPR, turbulence, average correlation, common idiosyncratic volatility, per-entity market loading.
- **Cross-sectional features:** dispersion (sd, MAD, IQR, interdecile), tail index, up-share, entropy (the `.xs` ops); Kelly–Jiang pooled tail index + tail betas, W₁ between consecutive cross-sections, average skewness (frame ops).
- **Variants:** group (sector) variants of all of the above.

**Non-goals, as decisions:**
- **QuEST** (LW 2012/2015/2017) — numerical inversion of the MP equation by optimisation per window. It is orders of magnitude slower†, and LW 2020/2022 match its accuracy by the authors' own comparisons†.
- **NERCOME**† (split-sample, B repeated eigendecompositions) — B× cost for a loss that QIS already attains.
- **EM for missing data** (Stambaugh 1997; Little–Rubin) — O(iter·N³) per window. Heavy missingness should be imputed upstream with `pn.CafeImputer` (point-in-time, exists).
- **POET** (Fan–Liao–Mincheva 2013) and **graphical lasso** — thresholded / sparse precision. It could be future work; the sklearn graphical lasso is O(N³) per iteration.
- **Kendall/Spearman-to-Pearson** (sin(πτ/2)) — O(N²·W log W) per window and not PSD.
- **DCC, GARCH, BEKK, DCC-NL** — sibling plan 9. We expose `nonlinear_shrinkage` for their correlation target.
- **Realised covariance from intraday data** — out of scope; HAR-RV exists in `econ.features`.
- **Portfolio allocation** (HRP, NCO, ERC, GMV as a product) — future work (§13). Every output here is allocation-ready.
- **Regime models, graphs and monitoring statistics** — sibling plans 7, 6 and 2 (§12).

---

## 4. Module placement and public API

### 4.1 Layout

```
panelary/covariance/
  __init__.py      # PEP 562 lazy exports; registers frame-op FeatureSpecs on first access
  _types.py        # CovEstimate (diagonal-plus-low-rank), Spectrum, WindowStats, Schedule
  _gram.py         # min-side Gram, two-pass centring/standardising, eigen (+cache), clip
  _update.py       # Welford/Chan-Golub-LeVeque shifted co-moments; EWMA recursion (online API)
  _linear.py       # LW identity / diagonal / constant-corr / single-index, OAS -- dual intensities
  _nonlinear.py    # Ledoit-Peche map family: QIS (default), LW2020; RIE (M6, gated)
  _rmt.py          # MP law, sigma^2 fixed point, Tracy-Widom edge, clip/denoise, detone
  _factor.py       # PCA factor model + diagonal; Woodbury
  _robust.py       # Gerber; window-local winsorisation; (regularised Tyler, M6 gated)
  _missing.py      # listwise-by-entity, zero-after-demean, pairwise + repair (clip | Higham)
  _window.py       # (T, N) matrix, as-of universe, schedule/anchors, per-date driver, as-of ffill
  _state.py        # market-state features from a Spectrum / CovEstimate
  _xsdist.py       # cross-date frame ops: Kelly-Jiang, W1, tail betas, avg skewness, CIV
  _estimator.py    # MarketState / Turbulence PanelTransformers (fit is empty)
panelary/namespaces/xs.py   # + dispersion, tail_index, up_share, entropy (additive)
panelary/_internal/_linalg.py  # psd_repair (moved), higham_nearest_corr, dplr_* kernels
```

`covariance` joins `_LAZY_SUBMODULES` in `panelary/__init__.py` (next to `embed`). It
is numpy + polars only, but no caller needs it at import time, and the import budget
(`tests/test_import_hygiene.py`, 300 ms) is shared. The four `.xs` ops are eager
because `namespaces` is eager. They are pure Polars expressions with no import cost.

### 4.2 The estimate object and estimator protocol

```python
@dataclass(frozen=True, slots=True)
class CovEstimate:
    """Sigma = diag(scale) @ (B diag(g) B^T + E) @ diag(scale), as of `asof`.

    E is either e0 * I (isotropic, B orthonormal: the spectral fast path) or diag(e).
    Never stores N x N unless the estimator is intrinsically dense (Gerber, pairwise).
    """
    asof: Any                   # the date whose window produced it
    entities: tuple[str, ...]   # the as-of universe, sorted
    scale: NDArray              # (N,) window sds (ones in covariance space)
    B: NDArray                  # (N, r) basis (orthonormal when `orthonormal`)
    g: NDArray                  # (r,) gains
    e: float | NDArray          # isotropic residual or (N,) diagonal
    orthonormal: bool
    method: str; space: str     # "qis", "lw_identity", ...; "correlation" | "covariance"
    n_eff: float; q: float      # effective sample size, N / n_eff
    shrinkage: float | None     # linear intensity, when defined
    dense: NDArray | None       # only Gerber / pairwise
    def solve(self, b) -> NDArray          # Sigma^{-1} b, O(N r) or O(N r^2 + r^3)
    def inv_quad(self, x) -> float         # x^T Sigma^{-1} x (Mahalanobis)
    def logdet(self) -> float              # matrix determinant lemma
    def cond(self, *, exact: bool = False) -> float
    def gmv_weights(self) -> NDArray       # Sigma^{-1} 1 / 1^T Sigma^{-1} 1
    def risk(self, w) -> float             # sqrt(w^T Sigma w), O(N r)
    def corr(self) -> "CovEstimate"        # same object with scale = 1
    def subset(self, idx) -> "CovEstimate" # marginal on a subset (turbulence with missing r_t)
    def to_dense(self, *, max_bytes: int = 512 * 2**20) -> NDArray  # refuses above budget

class CovarianceEstimator(Protocol):
    name: ClassVar[str]
    needs_vectors: ClassVar[bool]          # False -> eigvalsh path
    def __call__(self, w: WindowStats) -> CovEstimate: ...
```

`WindowStats` holds the compacted, centred (and in correlation space, standardised)
window `X` (n×N, C-contiguous), `scale`, `n_eff`, and a lazily computed, **cached**
`Spectrum` (eigenvalues descending, dual or primal vectors). Every estimator and every
market-state feature on the same date share one eigendecomposition.

### 4.3 Public functions

```python
import panelary as pn
cov = pn.covariance          # lazy

# (1) Functional core -- matrix in, estimate out. What drift monitoring (sibling 2),
#     graphs (sibling 6) and DCC-NL (sibling 9) call. No panel, no time.
est = cov.estimate(X, method="qis", space="correlation", assume_centered=False)

# (2) As-of series over a panel. Lazy: holds the (T, N) matrix + schedule, computes on demand.
series = cov.rolling(panel, returns="ret", window=252, method="qis",
                     schedule=cov.Schedule(every="1mo"),   # or every=21, or "every"
                     min_coverage=0.95, missing="zero_after_demean")
series.at(date)            # CovEstimate from the last scheduled date <= date
series.universe(date); series.dates

# (3) Per-date market state. Frame keyed by time (and group); broadcast=True joins to rows.
state = cov.market_state(panel, returns="ret", window=252, stride=1,
                         features=("absorption_ratio", "ar_shift", "lambda1_share",
                                   "effective_rank", "participation_ratio",
                                   "mp_signal_count", "market_ipr", "avg_corr"),
                         group=None, broadcast=False)
turb = cov.turbulence(panel, returns="ret", window=252, method="qis", refit="1mo")
civ  = cov.common_idio_vol(panel, returns="ret", window=63, n_factors=1, refit_every=21)
avgc = cov.avg_correlation(panel, returns="ret", window=63, kind="equal")  # | "pollet_wilson"
kj   = cov.tail_index(panel, returns="ret", window=21, q=0.05, residual=False)
w1   = cov.xs_wasserstein(panel, value="ret", standardize=False)

# (4) Transformers for Pipeline use. fit only records columns; every value is as-of.
cov.MarketState(returns="ret", window=252).fit_transform(panel)   # panel_safe=False, leakage_safe=True

# (5) Single-date cross-sectional ops (Polars-native, .over is load-bearing)
pl.col("ret").xs.dispersion(kind="mad").over("date")
pl.col("ret").xs.dispersion(kind="iqr").over(["date", "sector"])     # group variant
pl.col("ret").xs.tail_index(q=0.05, tail="lower").over("date")
pl.col("ret").xs.up_share().over("date")
pl.col("ret").xs.entropy(kind="share").over("date")
```

**Defaults and why.** `method="qis"` is the best or near-best in both regimes of §1.1
and the best-conditioned numerically. `space="correlation"`: at q≈2 the gap is 1.31
vs 2.21 for the same estimator in covariance space. `min_coverage=0.95` bounds
zero-fill attenuation to ≤5% (§5.11). `stride=1`.

### 4.4 Registry entries (names are globally unique — the registry keys by bare name)

| name | namespace | in → out | safe_scope | tier | cost_hint |
|---|---|---|---|---|---|
| `xs_dispersion` (method `.xs.dispersion`) | xs | series → series | rowwise | B | O(N log N) per date |
| `xs_tail_index` (`.xs.tail_index`) | xs | series → series | rowwise | B | O(N) per date |
| `xs_up_share` (`.xs.up_share`) | xs | series → series | rowwise | B | O(N) per date |
| `xs_entropy` (`.xs.entropy`) | xs | series → series | rowwise | B | O(N) per date |
| `market_state` | covariance | frame → frame | rowwise | B | O(D·(min(N,W)²·max(N,W) + min(N,W)³)) for D evaluated dates |
| `turbulence` | covariance | frame → frame | rowwise | B | O(R·estimate + T·N·r) |
| `avg_correlation` | covariance | frame → frame | rowwise | B | O(T·N·W) exact; O(T·N) Pollet–Wilson |
| `common_idio_vol` | covariance | frame → frame | rowwise | B | residualise + O(T·N) |
| `kelly_jiang_tail` | covariance | frame → frame | rowwise | B | O(T·D·N) (partition) |
| `xs_wasserstein` | covariance | frame → frame | rowwise | B | O(T·N log N) |

All ten: `panel_safe=False` (they mix entities by design), `leakage_safe=True`,
`source=<paper>`, `license="Apache-2.0"` (clean-room). The `xs_*` names need
`_METHOD_ALIASES` rows in `tests/test_registry_conformance.py`. The `sibling-8`
entropy plan will want `entropy`, so we avoid the bare name. The `covariance` frame
ops need `_FRAME_OPS` rows there with `window ≤ 5`, because the probe panel's shortest
entity has 12 rows. None contains the substring `T^2`.

### 4.5 Broadcasting and group variants

- **Single-date ops** (`.xs.*`) broadcast by construction: `.over("date")` gives every
  row its date's value. `.over(["date", "sector"])` gives the within-sector value.
  Evaluated bare they pool the whole panel, which is a leak (trap T11). The registry
  scope is claimed for the `.over(time)` composition only, as for the existing xs ops.
- **Covariance-derived features** return a per-date frame `(time[, group], features…,
  diagnostics…)`. `broadcast=True` performs `panel.join(state, on=[time(, group)],
  how="left")`. Per-entity outputs (`market_loading`, `idio_vol`) are keyed by
  `(entity, time)` directly.
- **Group variants** (`group="sector"`): for date t, the universe is partitioned by the
  **date-t value** of the group column. Each group with ≥ `min_entities` (default 10)
  runs its own window kernel on its members' trailing returns. A member's *historical*
  group labels inside the window are ignored, because membership is as-of t (trap T12).
  Cost is Σ_g of per-group costs. Groups are small, so this is usually cheaper than the
  pooled run.

---

## 5. Algorithms — the most accurate and fastest known CPU formulation for each

### 5.1 Data path

1. `build_tensor(panel, [returns], forward_fill=False, z_normalize=False)` gives
   (N, T, 1). Transpose it to a C-contiguous time-major `R` (T, N) float64 (120 MB at
   N=3,000, T=5,000). Float32 input is upcast. `NaN` means not observed.
2. `F = isfinite(R)`. Take **integer** prefix counts `C = cumsum(F, axis=0, dtype=int32)`,
   so window counts are `C[t] − C[t−W]`. This is exact, with no float drift.
3. The as-of universe `U[t] = F[t] & (count ≥ ceil(min_coverage·W))`, fully
   vectorised (T, N). An entity enters once it has coverage and leaves after its last
   observation. Nothing about its future is consulted. A later listing cannot enter an
   earlier window.
4. Evaluated dates `E` come from the schedule (§5.3).
5. **Per evaluated date** (the only Python loop): `idx = flatnonzero(U[t])` →
   `X = R[t−W+1 : t+1, idx]` (a compact contiguous copy, W×N_t; 6 MB at N=3,000) →
   §5.2. Compaction is load-bearing for bitwise prefix invariance: masking by zero
   instead would change GEMM blocking and summation order when an entity that appears
   *after* t widens the matrix.

Loop cost is justified by measurement. Each iteration is 0.2 ms (W=63) to 110 ms
(W=1,260) of LAPACK, against roughly 20 µs of interpreter overhead. Batching the
eigensolves gave 0.95× (§1.2). A batched path for W ≤ 128 is added only if M2
measures ≥1.5×, since tiny matrices are where call overhead matters.

### 5.2 The window kernel (min-side Gram)

- **Centring** is two-pass per window: `μ = nanmean(X, 0)`, `X −= μ`, then `X[~finite] = 0`
  (zero-after-demean). No raw-sum cancellation (§1.4). In correlation space,
  `σ_i = sqrt(Σ_obs x²/(n_i−1))` and `Z = X/σ`. `n_eff = W − 1`, or W with
  `assume_centered=True`, matching sklearn's MLE divisor when running parity tests.
- **Min-side Gram.** If `N_t > n`: `G = Z Zᵀ / n_eff` (W×W). Otherwise
  `S = Zᵀ Z / n_eff` (N_t×N_t). Symmetrise `½(G+Gᵀ)`. The two share the nonzero
  spectrum exactly. Dual eigenvectors lift as `V = Zᵀ U Λ^{-½} / √n_eff`: measured
  ‖VᵀV−I‖ = 1.3e-14, and eigenvalue parity with the primal is 1.4e-14.
- **Eigensolver choice.** `eigvalsh` when no feature or estimator on the date needs
  vectors. `eigh` otherwise. `randomized_svd(Z, k)` when only top-k is needed and
  `min(N,W) ≥ 1,000`. An optional `scipy.linalg.eigh(subset_by_index=…)` backend via
  `require("scipy")` is M6, used only for top-k and parity-tested at 1e-10.
- **Clipping.** Eigenvalues below `max(N,W)·eps·λ_max` are set to exactly 0 and
  counted as null directions (m = rank). Demeaning removes one direction, so
  m ≤ min(N_t, W−1).
- **Cost per date.** O(min(N,W)²·max(N,W)) for the Gram plus O(min(N,W)³) for the
  eigensolve. Memory is O(W·N + min(N,W)²).

### 5.3 Evaluation schedule, stride and as-of forward fill

- `Schedule(every="every" | int | "1w" | "1mo" | "1q")`.
  - An integer k evaluates at time-axis index ≡ 0 (mod k), anchored at index 0. This
    is the same convention as `detect.residualise`. Appending never moves the grid.
  - Calendar strings mean **the first trading date of each period**, which is causally
    decidable because it needs only the previous date. These anchors are invariant to
    both appending and prepending history, and are the default for temporal time
    columns.
  - **"Last trading date of the period" is refused** with a `ValueError` naming trap
    T6: knowing that t is last requires t+1.
- **Stride with as-of forward fill** evaluates only on the grid, then gives each row t
  the value from the latest grid date ≤ t (`np.maximum.accumulate` over indices, or
  `join_asof(strategy="backward")`). This is prefix-invariant because the grid is, and
  each filled value uses a window ending at or before t. Output always carries
  `asof_date` so staleness is visible.
- Estimator **refits** (turbulence, `rolling`) use the same `Schedule` object.
  Features and refits may use different schedules.

### 5.4 Rolling moments and co-moments (the online path only)

The batch path recomputes each window (§1.2 shows why). Rank-1 add/remove exists for:
- `cov.OnlineCovariance.update(row)`, the streaming API (primal, N ≤ ~1,000);
- the recursive EWMA;
- the O(N) per-date Pollet–Wilson path, which uses Polars rolling moments.

Formulations:
- **Add** (Welford 1962): `δ = x − μ; μ += δ/n; C += δ (x − μ)ᵀ`.
- **Remove** (reverse update, Pébay 2008): `δ = x − μ; μ −= δ/(n−1); C −= δ (x − μ)ᵀ`.
  In code the updates are rank-2 GEMMs (measured 2.7× cheaper than two `np.outer`
  updates at N=3,000).
- **Shifted data** (Chan–Golub–LeVeque 1983): subtract a causal per-entity reference
  (its first finite value in the current recompute block) before accumulating. This
  fixes the 1.8e-6 drift of §1.4.
- **Exact recompute** every `recompute_every` steps (default W), anchored at the stream
  origin (step ≡ 0 mod R). An end-anchored recompute breaks bitwise prefix invariance
  (trap T7). Drift between recomputes is bounded and tested (≤1e-12 relative at
  R=W=252 on returns).

### 5.5 EWMA

- **Finite-window EWMA (default).** `w_s = (1−λ)λ^s/(1−λ^W)`, s = 0..W−1, with
  `λ = 2^{−1/halflife}`. It enters the same dual kernel through row scaling
  `√w_s · x_{t−s}` with a weighted mean. `n_eff = 1/Σw_s²` (Kish) is passed to the
  shrinkage maps. This is a documented approximation, since shrinkage theory assumes
  equal weights; M5 benchmarks it (open question Q2).
- **Recursive EWMA** (`Σ_t = λΣ_{t−1} + (1−λ)x_t x_tᵀ`, RiskMetrics 1996) is only
  for the online API and N ≤ ~1,000. Measured 0.19 ms/step at N=500 and 5.6 ms/step at
  N=3,000. It starts at the stream's first row and emits NaN until `min_periods`. It
  never initialises from a full-sample covariance (trap T8).
- The two agree to 1e-12 when W ≥ log(1e-16)/log λ. That agreement is a test.

### 5.6 Linear shrinkage — closed-form intensities from the dual Gram

Let `G = X_c X_cᵀ` (n×n), `d_t = G_tt = ‖x_t‖²` and `S = X_cᵀX_c/n`. Then
`tr S = Σd_t/n`, `‖S‖²_F = ‖G‖²_F/n²` and `Σ_t‖x_t‖⁴ = Σ d_t²`. Every sum is O(n²)
given G, never O(N²).

- **LW 2004a, identity target.**
  - `μ = tr S / p`
  - `δ² = (‖S‖²_F − 2μ tr S + pμ²)/p`
  - `β̄² = (Σd_t²/n − ‖S‖²_F)/(p n)`
  - `ρ = min(β̄², δ²)/δ²`
  - `Σ̂ = (1−ρ)S + ρμI`

  This is exactly sklearn's `ledoit_wolf_shrinkage` algebra. Measured equal to
  3.7e-16 and 22× faster at N=3,000. The spectrum is `(1−ρ)λ_i + ρμ`, so a
  `CovEstimate` with orthonormal B = dual vectors has `g = (1−ρ)λ` and `e0 = ρμ`.
- **Diagonal target** (covShrinkage `covDiag`). Same pattern; the residual is `diag(e)`,
  non-isotropic, and takes the Woodbury path.
- **Constant correlation (LW 2004b).** In correlation space the target is
  `(1−r̄)I + r̄11ᵀ`, with `r̄ = (‖Z1‖²/n − p)/(p(p−1))`. That is O(nN) via the §5.14
  identity. The intensity's π, ρ and γ terms reduce to Σd_t², matvecs of `Z³ᵀ(Z·1)`,
  and `‖S‖²_F` (O(n²+nN)). Build: augment B with the normalised residual of **1**
  against U, eigendecompose the (r+1)×(r+1) projected matrix (tiny), rotate, and the
  isotropic complement is `ρ(1−r̄)`. Exact, and it stays on the spectral fast path.
- **Single index (LW 2003).** The market proxy is the equal-weight mean of the date-t
  universe. Target `s_m²ββᵀ + D`. All intensity terms are matvecs against the market
  vector, O(nN). The estimate is DPLR with `B = [dual vectors, β]` and `E = ρD` on the
  Woodbury path.
- **OAS (Chen et al. 2010, eq. 23), exact:**
  `ρ = min(1, ((1−2/p)tr(S²) + tr²S) / ((n+1−2/p)(tr(S²) − tr²S/p)))`,
  with `tr(S²) = ‖G‖²_F/n²`. **sklearn omits both 2/p terms** (verified in the
  installed source). We implement the paper; parity is handled in §7.3.

Measured: LW and OAS are indistinguishable under t(5) tails (1.41 vs 1.43). LW's π̂ is
distribution-free, while OAS's derivation is Gaussian. LW-identity is the linear default.

### 5.7 Nonlinear shrinkage — one Ledoit–Péché engine, three smoothers

All three keep the sample eigenvectors and map the sample eigenvalues `λ_i` to `d_i`,
estimating the Ledoit–Péché (2011) oracle `u_iᵀΣu_i`. They differ in how they smooth
the Stieltjes transform. One `_nonlinear.py` engine takes `(λ (m,), n, p)` and returns
`(d_0 for null directions, d (m,))`. That is O(m²) time and memory, chunked above
m=2,048 (32 MB blocks). Output is always a spectral `CovEstimate`
(B = V, g = d − d_0, e0 = d_0), so inverse, log-det and cond are exact in O(N·m).

| Candidate | Accuracy (§1.1, corr space, q=0.4 / 2.0) | Cost per window | Numerical sensitivity | Decision |
|---|---|---|---|---|
| **QIS** (LW 2022, Bernoulli 28(3)) | 1.19 / 1.31 | O(m³)+O(m²) | **1.4e-13** | **default** |
| LW 2020 analytical (AoS 48(5)) | 1.19 / 1.28–1.31 | O(m³)+O(m²) | 2.5e-8 | `method="lw2020"`; parity rtol 1e-6 |
| RIE (Bun–Bouchaud–Potters 2016/2017), raw η=N^-½ | 1.50 / 23.2 | O(m³)+O(m²) | — | M6 gate: needs the IW regularisation†; ship only within 5% of QIS |
| QuEST (LW 2012/2015) | reference-grade | optimisation per window† | — | excluded |

**QIS**, clean-room from Ledoit & Wolf (2022). m = min(p, n) non-null eigenvalues,
ascending; `c = p/n`; `h = min(c², c⁻²)^{0.35} / p^{0.35}`; `ι_i = 1/λ_i`.

```
θ_i  = mean_j [ ι_j (ι_j − ι_i) / ((ι_j − ι_i)² + h² ι_j²) ]
Hθ_i = mean_j [ h ι_j² / ((ι_j − ι_i)² + h² ι_j²) ]
A_i  = θ_i² + Hθ_i²
p ≤ n:  d_i = 1 / ((1−c)² ι_i + 2c(1−c) ι_i θ_i + c² ι_i A_i)
p > n:  d_0 = 1 / ((c−1) mean(ι)),   d_i = 1 / (ι_i A_i)
then    d ← d · tr(S) / Σd   (trace preservation over all p directions)
```

With demeaning, n = T−1 (the reference code's `k=1` convention).

**LW 2020.** Epanechnikov kernel with local bandwidth `h·λ_j`, `h = n^{-1/3}`, and its
Hilbert transform (eqs. 4.3, 4.7, 4.9). For p > n it uses eqs. C.4, C.5 and C.8 for
`d_0`. The `|x| = √5` edge is evaluated at the closed-form limit. The log term is
computed as `log|√5−x| − log|√5+x|` with `log1p` near the edge, and M1 measures
whether that lowers the 2.5e-8 sensitivity.

**Why correlation space by default.** Rotation-equivariant shrinkage of a covariance
matrix with heterogeneous variances wastes its budget on the variance spread. Measured:
2.21 → 1.31 at q≈2. This is also the construction of DCC-NL (Engle, Ledoit & Wolf
2019). The window sds are the diagonal scale.

### 5.8 Random-matrix cleaning

- **MP edge.** `q = N_t/n_eff`, `λ± = σ²(1 ± √q)²`.
- **σ² by fixed point**, which is deterministic, numpy-only and needs no KDE or
  optimiser. Start from `σ² = 1 − λ₁/N_t` in correlation space (Laloux et al. 1999).
  Iterate `σ² ← Σ_{λ_i ≤ edge(σ²)} λ_i / (p − #spikes)`. The denominator counts the
  p−n null eigenvalues, because MP's mean over *all* p eigenvalues is σ² while the
  mean over the nonzero ones is qσ² when q>1. Stop when the spike set is unchanged,
  which takes ≤10 iterations. This deliberately replaces López de Prado's scipy KDE
  fit; parity is on the signal *count*, not bitwise (§7.3).
- **Finite-size edge.** `edge_α = σ²[(√n+√p)² + t_α(√n+√p)(1/√n+1/√p)^{1/3}]/n` with
  the Tracy–Widom-1 quantile `t_0.95 ≈ 0.98`† (Johnstone 2001 centring and scaling).
  The bare edge over-counts signal eigenvalues at small N. `edge="bare"` is offered.
- **Denoise, constant residual (LdP 2020 §2.5).** Replace λ_i ≤ edge by their mean
  (trace-preserving), then restore the unit diagonal. In DPLR form the diagonal
  renormalisation folds into `scale` (`scale ← scale/√diag(C)`), so the inner matrix
  stays orthonormal + isotropic: exact O(N·k) inverse. `diag(C)` itself is
  `Σ_k g_k B_ik² + e0`, which is O(N·k).
- **Targeted shrinkage (LdP 2020 §2.6).** `C = C_signal + α C_noise + (1−α) diag(C_noise)`.
  The residual is a non-isotropic diagonal and takes the Woodbury path.
- **Detone (LdP 2020 §2.8).** Remove the top `k_m` (default 1) market components and
  renormalise. The result is singular by construction: `CovEstimate(invertible=False)`
  raises on `solve` with a message pointing to clustering use (sibling 6), not
  allocation.

### 5.9 Statistical factor model (k PCs + diagonal)

`Σ̂ = diag(σ)(B_k B_kᵀ + diag(ψ))diag(σ)` with `B_k = V_k Λ_k^{½}` and
`ψ_i = max(1 − Σ_k B_ik², ψ_floor)` (correlation space). Choosing k:
- fixed (default 1);
- `eigenvalue_ratio(spectrum)`;
- `_bai_ng_from_spectrum` (§2);
- the MP signal count.

All are per window. A full-sample k is trap T9. The Woodbury inverse is
`E⁻¹ − E⁻¹B(I_k + BᵀE⁻¹B)⁻¹BᵀE⁻¹`. Mahalanobis costs O(Nk), log-det uses
`Σlog ψ + logdet(I_k + BᵀE⁻¹B)`, and cond is bounded as in §5.13.

### 5.10 Gerber statistic (Gerber et al. 2022)

Thresholds are `c·σ_i` (window sd, default c=0.5). `U = 1[x > cσ]`, `D = 1[x < −cσ]`,
`H = U − D`, `A = U + D`, `n_i = Σ_t A_ti`.

`g_ij = (HᵀH)_ij / (n_i + n_j − (AᵀA)_ij)`, i.e.
`(n_UU + n_DD − n_UD − n_DU)/(T − n_NN)`.

That is two GEMMs, O(N²W), with **proved PSD** (Gerber et al. 2023,
arXiv:2305.05663); `psd_repair` is a guard, not a step. The matrix is dense and not
low-rank, so it is stored as `dense`. Inverting it is O(N³): refit dates only, and
combined with LW-identity shrinkage because rank ≤ W. The expected cost at N=3,000 is
~1.3 s per refit (§8), so Gerber is off by default in `market_state`.

### 5.11 Missing data

Admission is listwise by entity: `min_coverage` of W (§5.1). Within admitted entities
there are three policies.

- **`zero_after_demean`** (default). The estimate stays PSD and on the dual fast path.
  With independent missingness the correlation attenuates by ≈√(f_i f_j) (f = coverage
  fraction), so min_coverage=0.95 bounds it at 5%. Per-date `coverage` is reported.
- **`pairwise`.** `C_ij = Σ m_i m_j x_i x_j / √(Σ m_i m_j x_i² · Σ m_i m_j x_j²)`, via
  three GEMMs of the masked matrices, O(N²W). It is primal and dense, and not PSD, so
  it is followed by a repair:
  - `repair="clip"`, the reused `psd_repair`, reporting λ_min *before* repair;
  - `repair="higham"`, alternating projections with Dykstra's correction (Higham
    2002), O(iter·N³), refused above N=1,000.

  Refit dates only.
- **EM** is not offered (§3). Use `pn.CafeImputer` upstream.

### 5.12 Robust options

- **Window-local winsorisation (cheap, default off).** Per entity, clip at
  `median ± c·1.4826·MAD` of its own window before centring. This is O(W·N) per date
  and window-local, hence prefix-safe. A cross-sectional variant composes the
  existing `xs.winsorize` before the call.
- **Regularised Tyler** (Tyler 1987; Chen–Wiesel–Hero 2011). Each fixed-point step is
  O(W²N + W³) in dual Woodbury form, and 20–50 iterations make it 20–50× an estimate.
  **M6 gate:** ship only if it improves OOS GMV by ≥10% over QIS-corr on winsorised
  t(3) data.

### 5.13 CovEstimate operations (allocation-ready)

- **Spectral (orthonormal B, isotropic e0).**
  - `Σ⁻¹b = S⁻¹[B((g+e0)⁻¹ ⊙ Bᵀb') + (b' − BBᵀb')/e0]` with `b' = S⁻¹b` and S = diag(scale).
  - `logdet = 2Σlog scale + Σlog(g+e0) + (N−r)log e0`.
  - `cond_corr = max(g+e0, e0)/min(g+e0, e0)` exactly.
- **Woodbury (general B, diagonal E).**
  `(E + BGBᵀ)⁻¹ = E⁻¹ − E⁻¹B(I + GBᵀE⁻¹B)⁻¹GBᵀE⁻¹`. This form never inverts G, which
  may be singular, and costs O(Nr² + r³). The capacitance matrix is solved by Cholesky
  with a fallback to `eigh`. Use `np.linalg.solve(A, b[..., None])[..., 0]`, never
  `solve(A, b)` (the NumPy 2 batch pitfall, detect contract invariant 4).
- **`cond`.**
  - `exact=False`: `cond_corr` (exact on the spectral path; on the Woodbury path the
    bounds `λ_min ≥ min e`, `λ_max ≤ max e + ‖G‖`), plus
    `cond_cov ≤ cond_corr·(max σ/min σ)²`.
  - `exact=True`: dense `eigvalsh` if N ≤ 2,000; otherwise 50 power iterations on Σ
    and on Σ⁻¹ (O(Nr) matvecs, deterministic start vector 1).
  - A singular sample estimate reports `cond=inf`, and `solve` **raises** rather than
    silently using `pinv` (§1.1: `pinv` GMV was 2,095× oracle).
- **`to_dense`** refuses above `max_bytes`, following `shape`'s `_enforce_budget`
  idiom.

### 5.14 Market-state features (per evaluated date, from the shared Spectrum)

| Feature | Definition | Extra cost given the spectrum |
|---|---|---|
| `absorption_ratio` | `Σ_{i≤k}λ_i / Σλ_i` on the sample correlation spectrum (Kritzman et al. 2011); `k = ceil(0.2·min(N_t, n_eff))` by default (their "one fifth", capped at rank); fixed `k=` recommended when N_t varies; emits `ar_k` | O(1) |
| `ar_shift` | `(mean_{15}(AR) − mean_{252 preceding}(AR)) / sd_{252 preceding}(AR)`, trailing rolling on the AR series | O(1) per date |
| `lambda1_share` | `λ₁/tr` (= λ₁/N_t in correlation space); `lambda_gap = λ₁/λ₂` as a stability diagnostic | O(1) |
| `effective_rank`, `eigen_entropy` | `exp(H)`, `H/log m`, `H = −Σp_i log p_i`, `p = λ/Σλ` (Roy–Vetterli 2007) | O(m) |
| `participation_ratio` | `tr²/‖C‖²_F = N_t²·n_eff²/‖G‖²_F`, **no eigensolve needed** | O(n²) |
| `mp_signal_count`, `mp_sigma2`, `mp_edge` | §5.8 | O(m·iters) |
| `market_ipr` | `Σ_i v₁ᵢ⁴` ∈ [1/N, 1] (Plerou et al. 2002); `v₁ = Zᵀu₁/√(n λ₁)`; on the `eigvalsh` path u₁ comes from ≤30 power iterations on G seeded with `Z·1` (deterministic; the gap λ₁/λ₂ is reported) | O(n²·iters + nN) |
| `market_loading` (per entity) | v₁ᵢ oriented so Σv₁ > 0 | O(nN) |
| `avg_corr` | **exact identity**: `ρ̄ = (‖Z·1‖²/n_eff − N_t)/(N_t(N_t−1))`, i.e. `Var(z̄) = 1/N + (N−1)ρ̄/N` | O(nN), one matvec |
| `turbulence` | `d_t = (r_t − μ̂_s)ᵀΣ̂_s⁻¹(r_t − μ̂_s)/N_t` with s = the latest refit ≤ **t−1** (Chow et al. 1999; Kritzman & Li 2010, made causal); missing r_{i,t} → `est.subset(observed)`; plus a trailing percentile of d over 252 dates | O(N·r) per date |
| `common_idio_vol`, `idio_vol` | `ε = residualise(R, n_factors=k, min_periods, refit_every, window)` (reused); `idio_vol_i` = trailing sd of ε_i (Polars `rolling_std().over(entity)`; ε is mean≈0, so §1.4 drift does not apply); CIV = xs mean (Herskovic et al. 2016, with statistical factors instead of FF3, documented) | O(T·N) after residualise |
| diagnostics | `n_entities`, `coverage`, `q`, `n_eff`, `cond`, `shrinkage`, `asof_date` | — |

**`avg_correlation(kind="pollet_wilson")`** is the O(N)-per-date path, with no window
materialisation. `ρ̄_PW = (σ_p² − Σw_i²σ_i²)/((Σw_iσ_i)² − Σw_i²σ_i²)`, which is
*exactly* the σ_iσ_j-weighted mean of ρ_ij (Pollet & Wilson 2010). σ_p² comes from
the equal-weight portfolio series and σ_i² from per-entity rolling variances, all
Polars-native. It is exact only when the universe is constant across the window.
Dates where the universe changed are flagged (`universe_stable=False`) and
recomputed on the exact kernel path. A test covers this.

### 5.15 Cross-sectional distribution features

**Single-date `.xs` ops** are Polars expressions:
- `dispersion`: kind ∈ {sd, mad (×1.4826), iqr, idr (q90−q10)}. Quantiles always use
  `interpolation="linear"`, fixed explicitly because the Polars default differs from
  numpy.
- `tail_index`: Hill on one cross-section. `u` is the q-quantile of losses,
  `ξ = mean(log(L/u) | L > u)`, return `α = 1/ξ` to match `hill_index`. NaN when fewer
  than 10 exceedances.
- `up_share`: `#(r>0)/#non-null`.
- `entropy`: kind="share" is `−Σp log p / log N` with `p_i = |r_i|/Σ|r|`, which
  measures how concentrated the moves are. kind="hist" uses same-date
  Freedman–Diaconis bins.

**Frame ops** (need more than one date):
- **Kelly–Jiang** (2014). Pool the trailing D dates (default 21) × universe returns,
  optionally residuals via `residualise`. `u_t` is the pooled 5% quantile of returns;
  `λ_t = mean(log(R/u_t) | R < u_t)` (their ξ convention, documented against
  `hill_index`'s α). Implementation: `sliding_window_view` over time, then
  `np.partition` per date, O(D·N) per date (~1.5 s at N=3,000, T=5,000). **Tail beta**
  per entity is `rolling_beta(y=r_i, x=λ_{t−1}, window)`, reused.
- **W₁ between consecutive cross-sections** (t versus the previous date on the time
  axis). The default is the exact merge formula `∫|F_t(x) − F_{t−1}(x)|dx` over the
  two sorted samples, O(N log N) (Vallender 1973). `grid=K` gives the quantile-grid
  approximation `mean_k|F_t⁻¹(u_k) − F_{t−1}⁻¹(u_k)|`, fully vectorised.
  `standardize=True` computes on same-date z-scores, isolating shape change from
  location and scale.
- **Average skewness** (Jondeau–Zhang–Zhu 2019) is
  `pl.col(r).rolling_skew(D).over(entity)` → `.mean().over(time)`. It is a composition,
  exposed via `market_state(features=("avg_skew",))`. Verify `rolling_skew`'s signature
  on polars 1.35 (Q5).

---

## 6. Leak-safety design and named traps

**Invariants (every function):**
1. **Prefix invariance, bitwise.** `f(x[:T])[t] == f(x[:T+k])[t]`, tested with `tol=0.0`.
2. Every parameter is window-local (`N_t`, `n_eff`, σ², h, k, ρ). **None may depend on
   `len(panel)`, the panel's total entity count, or its last date.**
3. Deterministic: no RNG except `randomized_svd(seed=)` (explicit; M2 records it).
4. float64 throughout.
5. Never `rolling_map`.
6. The universe is compacted before any GEMM (§5.1).

**Safe defaults.** The as-of estimate at t uses rows (t−W, t]. Turbulence enforces
`estimate_lag ≥ 1`. Stride and refit grids are origin- or calendar-anchored. Hidden
global state does not exist, because there is no cross-date `fit`.

| # | Trap | Where it appears in the wild | Our guard | Test that must fail if re-introduced |
|---|---|---|---|---|
| T1 | Turbulence with **full-sample** μ, Σ | Kritzman & Li (2010) as published | not offered | `assert_no_lookahead` on `turbulence` |
| T2 | r_t inside its own Σ (window ends at t) | common rolling code | `estimate_lag ≥ 1`, validated | exact equality with Mahalanobis vs the estimate at t−1; lag=0 raises |
| T3 | ΔAR or returns z-scored with full-sample mean and sd | common | trailing windows only | perturbation test |
| T4 | Length/universe-dependent parameters: k=N/5 with the panel's N, q with total entities, coverage from T | common | N_t, n_eff per window | prefix test on `synth` with entity entry and exit |
| T5 | Survivorship / ffill: universe = survivors; delisted returns forward-filled (`build_tensor` default) | common | `forward_fill=False`, as-of universe | dropping entities after their exit, or ones listed after t, leaves values at t bit-identical |
| T6 | End-anchored or "last trading day" grids | common | index/epoch/first-of-period anchors; "last" raises | prefix test at every cut; `ValueError` test |
| T7 | Recompute of rank-1 / EWMA state anchored at the end | subtle | origin-anchored recompute | bitwise prefix test on `OnlineCovariance` |
| T8 | EWMA initialised with full-sample Σ | pandas-style code | warm-up NaN | perturbation test |
| T9 | Intensity, σ², k, bandwidth fit on the full sample and reused per date | common | no such parameter exists | `market_state` intensity at t == `estimate()` on the window, bitwise |
| T10 | Kelly–Jiang threshold over the *calendar month containing t* | paper's monthly design reused daily | trailing D dates | perturbation test |
| T11 | `.xs` op evaluated bare (pools all dates) | user error | docs + scope claim | explicit failing example, as for existing xs ops |
| T12 | Group labels from the latest classification | GICS snapshot joins | date-t group value | changing a future row's sector leaves the past unchanged |
| T13 | Eigenvector sign/order flips give spurious jumps in loadings (not a leak, but fake signal) | every per-date PCA | Σv₁>0; `_sign_of_max_abs` for others; λ-gap reported | determinism test; sign continuity on synth |
| T14 | `pinv` of a singular sample covariance used silently | `np.cov` + `pinv` | `solve` raises; cond=inf | test asserts the raise |
| T15 | Estimator/hyperparameter chosen by full-sample OOS comparison | backtest overfitting | walk-forward harness only | harness test: selection at s uses ≤ s |
| T16 | Timestamp semantics: r_t (close-to-close) used for a decision at open t | label misalignment | outputs documented "as of close t"; `lag=` helper | doc test |

---

## 7. Tests

`--strict-markers` is on; no new marker (`slow` and `benchmark` exist). Fixtures are
synthetic only (`pn.synth` or local generators). **Never** real sealed data.

### 7.1 `tests/test_covariance_kernels.py`

- **Primal = dual.** Spectra: rtol 1e-12. Vectors via projectors: 1e-10. QIS maps:
  1e-12. LW 2020 maps: **1e-8** (measured 3.6e-9).
- **Two-pass centring** at offsets 0 / 1e4 / 1e8: ≤1e-12 relative versus a float128
  (`np.longdouble`) reference where available.
- **PSD** of every estimator: `λ_min ≥ −N·eps·λ_max`.
- **`cond`** is exact on the spectral path against dense `eigvalsh`.
- **Woodbury** `solve`, `inv_quad` and `logdet` against dense (rtol 1e-10). Include
  the `p == batch` `solve` shape case.
- **`subset()`** marginal against dense slicing.
- **Float32 inputs** are upcast.
- **Singular sample** gives `solve` raising and cond=inf (T14).
- **Online co-moments** against recompute: ≤1e-12 at R=W; origin-anchored recompute
  is bitwise prefix-invariant (T7).

### 7.2 `tests/test_covariance_rmt.py` (properties on known populations)

- **White Wishart** N=400, n=1,000. The spectral CDF has KS distance < 0.03 from MP
  (density integrated numerically in numpy). Over 200 reps, the share of λ_max above
  the TW-95% edge is 5% ± 3%.
- **Spiked model** (BBP 2005; Paul 2007†), q=0.25, spikes ℓ ∈ {1.4, 3, 10} (threshold
  1+√q = 1.5):
  - top sample eigenvalues ≈ `ℓ(1 + q/(ℓ−1))` within 3%;
  - `mp_signal_count == 2` in ≥90% of reps;
  - squared overlap ≈ `(1 − q/(ℓ−1)²)/(1 + q/(ℓ−1))` within 0.05.
- **Oracle tracking.** Mean relative error of `d_i` against the oracle `u_iᵀΣu_i`:
  QIS < LW-identity < sample, at q ∈ {0.5, 2}.
- **Exact identities (1e-12).** `avg_corr` equals the mean of the off-diagonal of
  `np.corrcoef`. `participation_ratio` from the Gram equals `tr²/‖C‖²_F` from dense.
  AR on an exact 1-factor population equals its analytic share. Effective rank:
  identity → N, rank-1 → 1. IPR: uniform v₁ → 1/N.
- **Pollet–Wilson path** equals the exact path on a constant universe, and is flagged
  and recomputed when the universe changes.

### 7.3 `tests/test_covariance_oracle.py` (each behind `importorskip` or a committed fixture)

| Ours | Oracle | Licence | Tolerance |
|---|---|---|---|
| LW identity (ρ and Σ̂) | `sklearn.covariance.LedoitWolf` (MLE divisor n) | BSD-3 | rtol 1e-12 (measured 3.7e-16) |
| OAS | sklearn `OAS`: recompute *its* 2/p-free formula from our traces → 1e-12; paper eq. 23 differs by ≤4ρ/p | BSD-3 | as stated |
| diagonal, constant-corr, single-index, **QIS** | fixtures generated once from `pald22/covShrinkage` (MIT repo, BSD-2 files) on 60×20 and 20×60 inputs, committed with provenance; never a runtime or test dependency. Note: the reference uses `np.linalg.eig` and sorts by pandas column label, which misorders tied eigenvalues, so fixtures use distinct-spectrum inputs | MIT / BSD-2 | 1e-10 |
| LW 2020 | fixture from the authors' `analytical_shrinkage` code† | † | rtol 1e-6 |
| denoise, detone, Gerber | `skfolio` `DenoiseCovariance`, `DetoneCovariance`, `GerberCovariance`† | BSD-3† | Gerber 1e-12; denoise: **equal signal count** on spiked data, matrix rtol 1e-2 (σ² estimator differs by design, §5.8) |
| RIE (M6) | `pyRMT`† | licence† | 1e-8 |
| W₁ | `scipy.stats.wasserstein_distance` | BSD-3 | 1e-12 |
| `xs.tail_index` | `econ.features._evt.hill_index`, matched k | ours | 1e-12 |
| finite vs recursive EWMA | each other | — | 1e-12 |

### 7.4 `tests/test_covariance_prefix.py` and `tests/test_covariance_traps.py`

- **Prefix and look-ahead.** Every frame op and xs op runs through
  `assert_prefix_invariant(tol=0.0)` and `assert_no_lookahead(tol=0.0)` at 3 cuts, for
  `stride ∈ {1, 5}` and `every ∈ {5, "1w", "1mo"}`, with and without `group=`, on a
  `synth` panel with entity entry and exit.
- **Traps.** One test per trap T1–T16 (§6). Each constructs the leaky variant inline
  and asserts the verifier **catches** it, so the instruments are proven sharp.

### 7.5 `tests/test_covariance_accuracy.py` (`slow`)

- **§1.1 as assertions**, on the same generator with a fixed seed:
  - QIS-corr ≤ 1.40 at q≈2 and ≤ 1.25 at q=0.4;
  - QIS-corr < LW-identity in both;
  - **sample-`pinv` ≥ 100× at q≈2.** This asserts the silently-wrong baseline stays
    wrong. Never loosen it.
- **Regimes on `synth`** (n_regimes=2, regime_vol_ratio=2): mean `avg_corr` and
  `absorption_ratio` are higher in the high-vol regime, using planted regimes from
  `GroundTruth`.
- **Walk-forward OOS GMV harness** (§8.3): runs end to end and never reads beyond s
  when deciding at s (T15).

### 7.6 `tests/test_xs_distribution.py`

- Each `.xs` op against a numpy reference per date.
- `.over(["date", "sector"])` against a manual group loop.
- NaN and null policy.
- Bare-evaluation failure example (T11).

### 7.7 Standing guardrails

- `tests/test_registry_conformance.py`: add `_METHOD_ALIASES` and `_FRAME_OPS` rows
  and import `panelary.covariance` like `_shape`.
- `test_import_hygiene.py`: `covariance` not imported by `import panelary`; no
  top-level optional imports.
- `test_dependency_drift.py`: **zero** new mandatory dependencies.
- mypy ratchet (no new errors).
- `ruff`.
- CI legs: polars 1.35 / py3.10 and 1.42 / py3.11+.

---

## 8. Benchmarks and performance budgets

### 8.1 Budgets

Measured basis is §1.2; perf tests assert ≤2× these. Laptop CPU (Apple M5 Pro,
Accelerate), T = 5,000 dates.

| Workload | Budget | Basis |
|---|---|---|
| `market_state` spectrum set, N=500, W=252, daily | **≤ 12 s** (≤2.4 ms/date) | 1.74 ms/date measured (`eigvalsh` + Gram) |
| same, **N=3,000** | **≤ 20 s** (≤4 ms/date) | 2.6 ms/date measured |
| same, stride 5 / `"1w"` | ≤ 30% of daily | grid is 20% of dates + ffill |
| W=1,260, N=3,000, top-k set (AR, λ₁, IPR) daily | ≤ 4 min (rSVD, 33 ms/date); full set needs `"1mo"`: ≤ 40 s | measured 33 / 110 ms/date |
| `estimate(method="qis", space="correlation")`, N=3,000, W=252 | ≤ 8 ms | LW 2020 dual 5.2 ms measured; QIS same order |
| `turbulence`, N=3,000, monthly refit, daily scoring | ≤ 5 s | 238 × ≤8 ms + 5,000 × O(N·r) (~0.3 ms) |
| `avg_correlation` exact (kernel path), N=3,000, W=63 | ≤ 3 s | one matvec per date + window copy |
| `avg_correlation` Pollet–Wilson, 15M rows | ≤ 2 s | Polars rolling |
| four `.xs` ops together, 15M rows (N=3,000) | ≤ 3 s | Polars `.over` |
| `kelly_jiang_tail` D=21, N=3,000 | ≤ 3 s | partition 63k × 5,000 |
| `xs_wasserstein` exact, N=3,000 | ≤ 2 s | merge per date |
| Gerber per refit, N=3,000 | ≤ 2 s (two GEMMs + dense solve) | M5 measures; off by default |
| peak memory, N=3,000, T=5,000, W=252 | **≤ 1 GB** | R (120 MB) + `build_tensor` transient + O(W·N + W²) scratch; never T·N·N or T·W·N |
| scaling vs sklearn per-date `LedoitWolf().fit` | ≥ 1,000× at N=3,000 | 4.8 s vs ≤ 5 ms |

### 8.2 Benchmark design (`benchmarks/bench_covariance.py`, M2; results to `docs/user-guide/covariance.md` "Measured performance")

- **Grid:** N ∈ {100, 500, 3,000} × W ∈ {63, 252, 1,260} × stride ∈ {1, 5, 21} ×
  method ∈ {sample, lw, oas, qis, lw2020, mp_clip, factor-k, gerber}.
- **Per cell:** wall time split into `build_tensor` / universe / Gram / eigensolve /
  maps / features; ms per evaluated date; peak RSS (tracemalloc for numpy
  allocations); and the Python-overhead fraction (loop time minus kernel time).
- **Baseline column:** sklearn per-date `LedoitWolf`/`OAS` loop, subsampled to 50
  dates and extrapolated.
- **Machine calibration:** a 2,000×2,000 GEMM timing is stored with results. Perf tests
  scale budgets by it, as `test_import_hygiene` scales by `import polars`, so Linux
  OpenBLAS CI is not failed by Apple numbers.
- **Parallelism experiment** (records a decision; builds nothing): a thread pool over
  date blocks on Linux/OpenBLAS. Adopt only if ≥2× (Accelerate gave 1.3×).

### 8.3 Accuracy harness (benchmark, not public API until a caller exists)

Walk-forward GMV. At each refit s (`"1mo"`), estimate from (s−W, s], form
unconstrained GMV weights, hold for the next period, and record realised daily
returns. Report realised annualised vol per method and its ratio to LW-identity.
- **On synthetic data** also report true `wᵀΣw/oracle` and Frobenius PRIAL (LW 2020
  §6 design).
- **Populations:** the §1.1 factor population; a non-factor population (smooth
  eigenvalue spectrum, the LW papers' design); t(3) tails; q ∈ {0.2, 0.5, 1, 2, 5}.
- **Output:** a table in the docs. Its verdict can change the default only through a
  CHANGELOG entry.

---

## 9. Dependencies

- **Mandatory:** numpy, polars. **No change.**
- **Optional, acceleration:** `scipy` via `require("scipy", feature="covariance top-k eigensolver")`,
  for `eigh(subset_by_index)`. M6 only, with parity 1e-10. The default path never needs it.
- **numba / `fast`:** **not used.** We checked the three candidate loops.
  - The per-date loop is ≥97% LAPACK.
  - The EWMA recursion is an N² BLAS update.
  - The Gerber counts are GEMMs.

  None is interpreter-bound, so JIT buys nothing.
- **Test-only:** `scikit-learn` (`ml`), `scipy`, `skfolio`†, `pyRMT`†, all behind
  `pytest.importorskip`. The covShrinkage-derived fixtures are committed data, not
  dependencies.
- No new extra. `all` is unchanged. `tests/test_dependency_drift.py` stays green.

---

## 10. Milestones

| M | Contents | Exit criteria |
|---|---|---|
| **M1 — Kernels & estimators (private functional core)** | `_types`, `_gram`, `_linear` (LW identity, diag, const-corr, single-index, OAS), `_nonlinear` (QIS, LW 2020), `_rmt` (MP, σ² fixed point, TW edge, clip, targeted, detone), `_factor`, `_missing` (zero_after_demean), `cov.estimate()`. Move `psd_repair` to `_internal/_linalg.py` (re-export from `depend._matrix`). Add `_bai_ng_from_spectrum`. | §7.1, §7.2 and the sklearn/covShrinkage rows of §7.3 green; primal = dual; §7.5 assertions at the two §1.1 configs green. No public export yet. |
| **M2 — As-of engine** | `_window` (build_tensor path, integer-count universe, `Schedule` incl. calendar anchors, compaction, as-of ffill), `cov.rolling` → lazy series, `market_state` spectrum features + diagnostics, `benchmarks/bench_covariance.py` | `test_covariance_prefix` (tol=0) and traps T3–T9, T13 green; §8.1 budgets for N=500 and 3,000 met (2× slack) on this machine; measured table in docs. |
| **M3 — Market state completion** | `turbulence` (T1/T2), `avg_correlation` (exact + Pollet–Wilson), `common_idio_vol` (reuse `residualise`), `ar_shift`, `market_loading`, group variants, `MarketState`/`Turbulence` transformers, frame-op `FeatureSpec`s + conformance wiring, docs page | Traps T1, T2, T12, T14, T15 green; registry conformance green; **engine caller test exists (§12) before `pn.covariance` is documented as public.** |
| **M4 — Cross-sectional distribution** | `.xs.dispersion/tail_index/up_share/entropy` (+ specs, aliases), `kelly_jiang_tail` + tail betas (reuse `rolling_beta`), `xs_wasserstein`, avg skewness composition | §7.6 green; T10, T11 green; W₁ and Hill oracles green; polars 1.35 leg green. |
| **M5 — Robust, missing, EWMA, harness** | Gerber, window-local winsorisation, `pairwise` + clip/Higham repair, finite-window EWMA in the dual kernel, `OnlineCovariance` (Welford/CGL + recursive EWMA), §8.3 accuracy harness with its docs table | skfolio Gerber parity; online parity and T7/T8; harness results published; the default method re-confirmed or changed via CHANGELOG. |
| **M6 — Gated extras (each ships only if its gate passes)** | RIE with IW regularisation (gate: within 5% of QIS-corr on the §8.3 harness); regularised Tyler (gate: ≥10% OOS gain on winsorised t(3)); scipy top-k backend (gate: ≥1.5× at W ≥ 1,000); row-shift Gram update for covariance space (gate: Gram > 40% of runtime); thread pool (gate: ≥2× on Linux) | Each gate's measurement is recorded in docs, whether it passes or fails. |

M1 alone is useful to the siblings (drift monitoring, graphs, DCC-NL) as a matrix-in,
estimate-out core. M2 is where the as-of promise is proven.

---

## 11. Risks and open questions (resolve with a measurement, not an opinion)

1. **Q1 — Default estimator.** QIS-corr is best or tied on *one* factor-structured
   population. The M5 harness must cover smooth-spectrum and heavy-tail populations
   and q from 0.2 to 5 before the default is locked.
2. **Q2 — EW weights with nonlinear shrinkage.** The Kish `n_eff` substitution is a
   heuristic with no theory we know of. Measure the PRIAL loss against equal weights
   of the same n_eff.
3. **Q3 — Absorption ratio under a changing universe.** AR drifts mechanically with
   N_t and with k=⌈0.2·min(N_t, n)⌉. Measure on `synth` with entry and exit whether a
   fixed k or a rank-normalised AR is the more stable default.
4. **Q4 — Correlation-space scale.** Window sds are noisy at W=63. Sibling 5's range
   estimators (Parkinson, Garman–Klass, Yang–Zhang) may be better diagonal scales.
   `CovEstimate` accepts an external `scale=` so the composition is a parameter, not
   a fork.
5. **Q5 — polars 1.35.** Confirm `rolling_skew`, `rolling_std(min_samples=)`,
   `Expr.quantile(interpolation="linear")` and `join_asof` behave identically on 1.35
   and 1.42 (CI matrix).
6. **Q6 — Tracy–Widom constants.** The real-case centring and scaling and the 95%
   quantile (≈0.98)† need verifying against Johnstone (2001) before M1 closes. The
   §7.2 size test is the check.
7. **Q7 — BLAS determinism.** Bitwise prefix invariance holds within a platform and
   thread count. Eigen outputs differ across BLAS builds, so tests compare within one
   run, never against stored bits.
8. **Q8 — Degenerate spectra.** When λ₁/λ₂ ≈ 1, v₁ and hence IPR and market loading
   are ill-defined. Emit NaN below a gap threshold (default 1.05), or report the gap
   only? Decide on `synth` data.
9. **Q9 — Gerber at N=3,000.** It needs a dense N³ solve per refit. If M5 measures
   above 2 s, restrict Gerber to N ≤ 1,500 or to correlation output only (no `solve`).
10. **Q10 — Naming.** `panelary.covariance` avoids `risk`, which sibling 9 is likely
    to want. Confirm no sibling claims `covariance` before M1.
11. **Q11 — `build_tensor` cost.** Its cross-join grid is N·T rows. Measure at 15M
    cells in M2. If it exceeds 2 s, the shape owner adds a scatter path; we do not
    fork.

---

## 12. Caller, and boundaries with sibling plans

### 12.1 Caller (AGENTS.md rule — flagged)

> AGENTS.md: *"No new public surface without a named caller in the engine, and a test
> in the engine that exercises it."* **No engine caller exists today.** The user asked
> for this plan explicitly, so the rule is honoured this way: M1–M2 land as private
> modules. `pn.covariance` is not documented, not in `mkdocs` nav, and not advertised
> in `llms.txt` until the caller test below exists (M3 exit criterion).

Most plausible caller: **`truepoint/src/truepoint/generate/family_computation.py`**
(the deterministic, tolerance-graded computation family, metric `COMPUTATION_FIDELITY`).
A risk-computation template kind (trailing correlation or volatility, shrunk-covariance
GMV variance, absorption ratio as of a date) needs a reference calculator that is
as-of, deterministic and prefix-invariant. That is exactly `cov.estimate` /
`cov.rolling(...).at(date)`. Grading stays in `score/m02_computation.py`, which must not
import `generate` and needs nothing from panelary. Generated answer keys are sealed
ground truth. They are never written into panelary tests, fixtures or docs.

Second candidate: **`truepoint/src/truepoint/validity/`**, stratifying reliability
results by as-of market state (turbulence, absorption-ratio decile) to test whether
agents are more often silently wrong in stressed markets.

**Firewall note:** `truepoint/_firewall.py` forbids `diagnose/` from importing panelary.
Cause assignment must never call this module. It may appear only later, in
`recommend/`, as a disclosed affiliated option alongside alternatives (STRATEGY §3).

Consumed by: `truepoint/generate/` (reference values), then `truepoint/validity/`
(regime stratification).
Caller: `truepoint/src/truepoint/generate/family_computation.py`.

### 12.2 Boundaries with the nine sibling plans

| Sibling | They own | We own | Interface |
|---|---|---|---|
| 1 forecast-evaluation-and-sharpe-inference | Sharpe inference, forecast tests | covariance of *asset* returns | none required. Our §8.3 harness reports realised vol, not Sharpe inference. |
| 2 drift-monitoring-and-sequential-inference | Hotelling T², MEWMA, control limits, in-control fitting discipline | the shrunk covariance and its inverse | they call `cov.estimate(X_train, method="lw"|"oas"|"qis")` and use `.inv_quad()` / `.logdet()`. They fit it on in-control training rows only; that train/test discipline is theirs. |
| 3 (this plan) | — | — | — |
| 4 label-weights-and-event-sampling | sample weights, concurrency | — | no overlap. |
| 5 ohlc-volatility-and-liquidity | range-based per-entity volatility, liquidity | correlation structure | `CovEstimate(scale=...)` accepts their vol as the diagonal (Q4). We never estimate from OHLC. |
| 6 network-and-spatial-panel | graphs (TMFG, MST, PMFG), centralities, spatial weights | correlation matrices (denoised, detoned, Gerber) | they consume `est.corr().to_dense()` or `depend.to_distance`. Graph eigenvector centrality is theirs; the spectral market-mode loading and IPR of the correlation matrix are ours. |
| 7 causal-state-space-and-regimes | HMM, Markov switching, Kalman | market-state *observables* | they may use `market_state` columns as observations. We fit no regime model. |
| 8 multiscale-complexity-features | per-entity entropies (sample, permutation), multiscale measures | per-date cross-sectional entropy (`xs_entropy`) and eigen-entropy | registry names are `xs_`-prefixed to avoid a bare `entropy` collision. |
| 9 tail-risk-and-self-excitation | DCC, GARCH, BEKK, DCC-NL dynamics, CoVaR, MES, Hawkes, per-entity EVT | static-window estimators incl. EWMA; Kelly–Jiang cross-sectional tail index | they call `cov._nonlinear` (QIS) for a DCC-NL correlation target. We do not model conditional dynamics. Per-entity POT/EVT stays in `econ.features._evt` and theirs. |
| 10 panel-causal-inference | synthetic control, interactive fixed effects | — | none. They may use `_factor` for a PCA factor covariance. |

---

## 13. Future work (not in this plan)

- Portfolio construction consuming `CovEstimate`: HRP (López de Prado 2016), NCO
  (2020, which uses our denoised correlation), ERC. Every needed primitive (`corr`,
  `to_dense`, `solve`, `risk`, `gmv_weights`) is already on the object.
- POET and graphical lasso.
- DCC-NL jointly with sibling 9.
- A realised-covariance input path for intraday data.
- `pesaran_cd` O(NT) rewrite via the §5.14 identity (econ owner).

---

## 14. References

Unverified details are marked †. Items marked "verified" were checked against source
code or the publisher page this session.

- Baik, J., Ben Arous, G., Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *Annals of Probability* 33(5), 1643–1697.
- Bai, J., Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica* 70(1), 191–221.
- Bun, J., Allez, R., Bouchaud, J.-P., Potters, M. (2016). Rotational invariant estimator for general noisy matrices. *IEEE Trans. Inf. Theory* 62(12)†.
- Bun, J., Bouchaud, J.-P., Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. *Physics Reports* 666, 1–109 (IW regularisation details†).
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. *American Statistician* 37(3), 242–247.
- Chen, Y., Wiesel, A., Eldar, Y. C., Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Trans. Signal Processing* 58(10), 5016–5029 (verified via the sklearn source citation, eq. 23).
- Chen, Y., Wiesel, A., Hero, A. O. (2011). Robust shrinkage estimation of high-dimensional covariance matrices. *IEEE Trans. Signal Processing* 59(9)†.
- Chow, G., Jacquier, E., Kritzman, M., Lowry, K. (1999). Optimal portfolios in good times and bad. *Financial Analysts Journal* 55(3), 65–73.
- Engle, R. F., Ledoit, O., Wolf, M. (2019). Large dynamic covariance matrices. *J. Business & Economic Statistics* 37(2), 363–375.
- Fan, J., Liao, Y., Mincheva, M. (2013). Large covariance estimation by thresholding principal orthogonal complements. *JRSS-B* 75(4), 603–680.
- Gerber, S., Markowitz, H. M., Ernst, P. A., Miao, Y., Javid, B., Sargen, P. (2022). The Gerber statistic: a robust co-movement measure for portfolio optimization. *J. Portfolio Management* 48(3), 87–102 (pages†). Formula (n_UU+n_DD−n_UD−n_DU)/(T−n_NN) verified via portfoliooptimizer.io.
- Gerber, S., et al. (2023). Proofs that the Gerber statistic is positive semidefinite. arXiv:2305.05663 (verified).
- Halko, N., Martinsson, P.-G., Tropp, J. A. (2011). Finding structure with randomness. *SIAM Review* 53(2), 217–288.
- Herskovic, B., Kelly, B., Lustig, H., Van Nieuwerburgh, S. (2016). The common factor in idiosyncratic volatility: quantitative asset pricing implications. *J. Financial Economics* 119(2), 249–283.
- Higham, N. J. (2002). Computing the nearest correlation matrix — a problem from finance. *IMA J. Numerical Analysis* 22(3), 329–343.
- Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics* 3(5), 1163–1174.
- Johnstone, I. M. (2001). On the distribution of the largest eigenvalue in principal components analysis. *Annals of Statistics* 29(2), 295–327 (TW constants for §5.8†).
- Jondeau, E., Zhang, Q., Zhu, X. (2019). Average skewness matters. *J. Financial Economics* 134(1), 29–47.
- Kelly, B., Jiang, H. (2014). Tail risk and asset prices. *Review of Financial Studies* 27(10), 2841–2871 (verified: Hill on the monthly cross-section of daily returns, 5% threshold, FF3 residuals).
- Kritzman, M., Li, Y. (2010). Skulls, financial turbulence, and risk management. *Financial Analysts Journal* 66(5), 30–41.
- Kritzman, M., Li, Y., Page, S., Rigobon, R. (2011). Principal components as a measure of systemic risk. *J. Portfolio Management* 37(4), 112–126 (verified: k = N/5, 500-day window, 1-year half-life; ΔAR = 15-day mean minus preceding 1-year mean, standardised).
- Laloux, L., Cizeau, P., Bouchaud, J.-P., Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters* 83(7), 1467–1470.
- Ledoit, O., Péché, S. (2011). Eigenvectors of some large sample covariance matrix ensembles. *Probability Theory and Related Fields* 151, 233–264†.
- Ledoit, O., Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *J. Empirical Finance* 10(5), 603–621.
- Ledoit, O., Wolf, M. (2004a). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivariate Analysis* 88(2), 365–411.
- Ledoit, O., Wolf, M. (2004b). Honey, I shrunk the sample covariance matrix. *J. Portfolio Management* 30(4), 110–119.
- Ledoit, O., Wolf, M. (2012). Nonlinear shrinkage estimation of large-dimensional covariance matrices. *Annals of Statistics* 40(2)†; (2015) Spectrum estimation (QuEST), *J. Multivariate Analysis* 139†; (2017) Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks, *Review of Financial Studies* 30(12)†.
- Ledoit, O., Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics* 48(5), 3043–3065 (equation numbers in §5.7†).
- Ledoit, O., Wolf, M. (2022). Quadratic shrinkage for large covariance matrices. *Bernoulli* 28(3), 1519–1547 (verified; formula verified against the authors' BSD-2 reference `QIS.py`).
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press, ch. 2 (section numbers†).
- Marčenko, V. A., Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Math. USSR-Sbornik* 1(4), 457–483.
- Paul, D. (2007). Asymptotics of sample eigenstructure for a large dimensional spiked covariance model. *Statistica Sinica* 17, 1617–1642†.
- Pébay, P. (2008). Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments. Sandia Report SAND2008-6212.
- Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T., Stanley, H. E. (2002). Random matrix approach to cross correlations in financial data. *Physical Review E* 65, 066126.
- Pollet, J. M., Wilson, M. (2010). Average correlation and stock market returns. *J. Financial Economics* 96(3), 364–380.
- RiskMetrics (1996). *RiskMetrics — Technical Document*, 4th ed. J.P. Morgan / Reuters.
- Roy, O., Vetterli, M. (2007). The effective rank: a measure of effective dimensionality. *EUSIPCO 2007*.
- Stambaugh, R. F. (1997). Analyzing investments whose histories differ in length. *J. Financial Economics* 45(3), 285–331.
- Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter. *Annals of Statistics* 15(1), 234–251.
- Vallender, S. S. (1973). Calculation of the Wasserstein distance between probability distributions on the line. *Theory of Probability & Its Applications* 18(4), 784–786.
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics* 4(3), 419–420.

Software (test oracles only):
- scikit-learn `sklearn.covariance` (BSD-3; LW and OAS formulas verified in the installed 1.9.0 source).
- [pald22/covShrinkage](https://github.com/pald22/covShrinkage) (MIT repository, BSD-2 file headers; verified).
- skfolio (BSD-3†).
- pyRMT (licence†).
- SciPy `scipy.stats.wasserstein_distance` (BSD-3).
