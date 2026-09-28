# `panelary/depend/` — build contract

> **Status (2026-09-28): partially implemented.** M1–M4 are in `panelary/depend/`
> (+ additive `.ts` ops in `panelary/namespaces/ts.py`), with 16 `tests/test_depend_*.py`
> files; `test_depend_calibration.py` is green, slow suite included. Not done: the
> `select.mrmr` / `select.correlation` upgrade (out of scope this round), `tau_star`,
> `hsic_lasso` and `rcit`. Wired in as `pn.depend` (top-level export, mkdocs nav,
> conformance entries, CHANGELOG). Measured deviations from this contract
> (Politis–White block length over-rejects, `min_shift` breaks the rotation null,
> D=256 is not enough for RFF p-values within 10%, `chebyshev_neighbour_counts`
> cannot serve KSG, the GARCH claim holds only in direction) are recorded in
> `docs/user-guide/dependence.md`.

Nonlinear **dependence measurement, screening and inference** for panel data.
Issued to the implementation agents 2026-09-09.

Pure **numpy + polars**. No scipy, sklearn, statsmodels, numba, Rust in the
import path. Oracles (`dcor`, `hyppo`, `scipy.stats.chatterjeexi`) are
**test-only**, imported inside tests behind `pytest.importorskip`.

Reuse, do not reinvent:
`panelary.econ._common` (`ols`, `pinv_sym`, `norm_cdf`, `norm_ppf`, `t_sf`,
`chi2_sf`, `factorize`, `group_mean`, `group_sum`, `newey_west_lrv`,
`auto_bandwidth`, `winsorize`), `panelary._internal._numpy_stats`
(`chebyshev_neighbour_counts`, `lstsq`), `panelary.validation._bootstrap`
(`moving_block_bootstrap`, `circular_block_bootstrap`, `stationary_bootstrap`,
`wild_bootstrap`, `sieve_bootstrap`, `block_bootstrap_indices`,
`resolve_segments`), `panelary.validation._selection_stats`
(`romano_wolf`, `holm_bonferroni`, `benjamini_hochberg`,
`benjamini_yekutieli`, `MultipleTestResult`), `panelary.core.model_selection`
(`PurgedKFold`, `CombinatorialPurgedCV`).

---

## Why this module exists

Every dependence library on the market ships a **coefficient** and calls the
job done. The coefficient is the easy half. Three things are missing everywhere,
and all three are exactly what a leak-safe panel library is for.

**1. The null is wrong, and it is wrong by an order of magnitude.** Measured on
this machine (3000 replications, n=500, two *independent* AR(1) series, testing
them against each other with Chatterjee's ξ under the standard i.i.d.
asymptotic null):

| Serial dependence | Type-I error at nominal 5% | Inflation |
|---|---|---|
| φ = 0.00 | 5.2 % | 1.0× |
| φ = 0.70 | 6.9 % | 1.4× |
| φ = 0.95 | **56.3 %** | **11.3×** |

φ = 0.95 is not a corner case — it is what every price level, valuation ratio,
accrual, sentiment index and macro series in a real panel looks like. A library
that reports i.i.d. permutation or i.i.d. asymptotic p-values on that data
reports noise as discovery *more than half the time*. This is the same class of
error as unpurged cross-validation, and it is the reason this module exists.
Panelary already owns the fix — five bootstraps and four multiplicity
corrections are already in `panelary/validation/`. We are wiring them to
dependence statistics, not building them.

**2. Nobody separates dependence from volatility clustering.** In finance most
apparent "nonlinear dependence" is conditional heteroskedasticity, not
structure a feature can exploit. On simulated GARCH(1,1) returns the lag-1
ξ z-statistic is +1.91 raw and +0.21 after dividing by a causal rolling
volatility. The library must make devolatilising a **first-class, labelled
parameter** and report both numbers, so a user cannot mistake vol clustering
for signal. See `benchmarks` §M5 — this claim is a required experiment, not an
assumption.

**3. Nobody is panel-aware.** Pooling a panel and computing one dependence
number lets the longest entity dominate and mixes within-entity with
cross-sectional variation (Simpson's paradox with extra steps). And when
entities share macro shocks, per-entity independent shuffles are not a valid
null at all. `depend` computes within-entity by default and ships a
**common-time** null that permutes whole date columns jointly across entities.

We are not competing on having the most statistics. We are competing on the
p-value being real.

---

## Hard invariants (every function)

1. **Prefix invariance.** `f(x[:T])[t] == f(x[:T+k])[t]` for all `t <= T`. No
   quantity may depend on `len(x)` — not window sizes, not thresholds, not
   normalisation constants, and **not kernel bandwidths**. See invariant 7.
2. **Determinism without a seed.** Ties are broken **deterministically** by
   default (`tie_break="average"` / the ties-corrected estimator), *not* by the
   uniform-random tie-breaking that SciPy and XICOR use. `tie_break="random"`
   is offered and then requires an explicit `seed: int`. Rationale: a library
   whose ξ changes between two calls on the same frame is unusable in a test
   suite and in a research log.
3. **Determinism where RNG is unavoidable.** Surrogates, permutations,
   subsampling and random Fourier features take an explicit `seed: int` and
   build `np.random.default_rng(seed)` internally. Never module-level RNG.
4. **float64 everywhere.** Upcast Float32 polars columns before accumulating.
   Rank arrays are `int64`; ξ/Hoeffding accumulators are float64 (`n**2` at
   n = 100 000 is 1e10 — it fits int64, but the divisions do not stay exact).
5. `np.linalg.solve(A, b[..., None])[..., 0]` — never `solve(A, b)` (NumPy 2.0
   silently mis-solves when `p == batch size`). Relevant in `_cond.gcm` and
   `_kernel.hsic_lasso`.
6. Never `rolling_map` (measured 249× penalty). Rolling dependence uses
   `sliding_window_view` + batched argsort (§ "Measured facts"); `map_batches`
   per group is the only escape hatch.
7. **A bandwidth is a fit.** The median heuristic for an RBF kernel, the
   normalising constant of a rank transform computed over the pooled sample, a
   quantile threshold for tail dependence, the random frequencies of an RFF
   map — all of these are *fitted parameters*. They must be estimated **in-fold
   only** and frozen into the transformer at `fit` time. A module-level default
   bandwidth computed from the full frame is a leak and will be caught by
   `tests/test_depend_leakage.py`.
8. **A dependence estimate is fit-free; a dependence-based *selection* is not.**
   `depend.xi(x, y)` on a training fold cannot leak. `depend.feature_screen(...)`
   run on the full panel and then used inside CV **is** a leak, and is the
   single most common way this module will be misused. Every screening and
   matrix function therefore returns a frozen artefact and every
   `PanelTransformer` wrapper sets `leakage_safe = True` only because it
   re-screens per fold.
9. **Every result carries its null.** No function returns a bare p-value. The
   result struct always carries `null_method`, `n_resamples`, `block_length`,
   and `n_obs`, and `null="iid"` emits a `LeakageWarning`-class warning when the
   input fails a cheap serial-dependence pre-check (lag-1 |ρ| of ranks > 0.2).

---

## Measured facts that dictate the design

Verified on this machine (Apple Silicon, NumPy 2.5.2 + Accelerate BLAS,
polars 1.44.1). These are measurements, not estimates.

| Fact | Measurement | Consequence |
|---|---|---|
| `sliding_window_view` + batched argsort | rolling ranks, T=2500, w=60: **5.5 ms** for one series | Rolling nonlinear dependence is *cheap*. This is the whole reason `_rolling.py` exists. |
| single `argsort` + `put_along_axis` scatter vs double `argsort` | 720 ms vs 1176 ms on (200, 2441, 60) | **1.63× — mandated idiom.** Never write `argsort(argsort(...))`. |
| full rolling ξ, 200 entities × 2500 dates, w=60 | **1.97 s** → ~49 s extrapolated to 5000 entities, single-threaded | Rolling ξ over a real panel is a coffee break, not a weekend. Ship it. |
| RFF-HSIC, n=100 000, D=256 | **312 ms** (two gemms, Accelerate) | RFF-HSIC is the matrix-scan workhorse. D=64 → 76 ms. |
| dense `|x_i - x_j|`, n=10 000 | 169 ms, **800 MB** | Hard cap. n=20 000 is 3.2 GB. |
| dense `|x_i - x_j|`, n=20 000 | 1.14 s, **3.2 GB** | O(n²) dCor must be blocked, and needs an O(n log n) univariate path. |
| √n·ξ null variance, 2000 reps | n=200 → 0.4151; n=1000 → 0.3972; n=5000 → 0.3943 | The 2/5 asymptotic null is **real and usable from n≈200**. No permutation needed for i.i.d. data. |
| ξ type-I error, independent AR(1) φ=0.95, i.i.d. null | **56.3 %** at nominal 5 % | See "Why this module exists". |

---

## What we build, and what we deliberately do not

The published survey that prompted this work assumes a Rust kernel crate and a
Polars plugin. **That is not available to us** — `AGENTS.md` forbids
reintroducing a compiled extension and `tests/test_wheel_guardrails.py` enforces
a `py3-none-any` wheel. So the method ranking must be re-derived from the numpy
cost model, not inherited. In numpy, three shapes are fast and one is not:

* **A. sort / cumsum shaped** — O(n log n), ~10⁸ elements/s. ξ, Spearman,
  Hoeffding D, normal-score transforms, tail dependence, fast univariate dCov.
* **B. matmul shaped** — Accelerate/OpenBLAS, ~10²  GFLOPS. RFF-HSIC, GCMI,
  GCM, RCIT, correlation matrices, tail-dependence matrices.
* **C. blocked O(n²) reductions** — fine to n ≈ 10 000, dead beyond. dCor
  (multivariate), exact HSIC.
* **D. per-element tree traversal** — *slow, and the reason the survey wanted
  Rust.* KSG k-NN, CODEC nearest neighbours, MGC.

Everything in class A and B ships. Class C ships with a hard cap and automatic
subsampling. Class D ships only where an existing sort-windowed primitive
already exists in this repo (`chebyshev_neighbour_counts`) and is capped.

### Shipping

| # | Statistic | Class | Null | Milestone |
|---|---|---|---|---|
| 1 | `spearman`, `kendall`, `pearson` | A | closed form | M1 |
| 2 | `xi` — Chatterjee, **directional** (X→Y and Y→X separately) | A | **N(0, 2/5) closed form** | M1 |
| 3 | `hoeffding_d` — general rank test, catches what ξ misses | A | closed form (tabulated) | M1 |
| 4 | `dcor` — bias-corrected distance correlation | A (univariate) / C (multivariate) | gamma / permutation | M1 |
| 5 | `gcmi` — Gaussian-copula MI, **multivariate & conditional** | B | χ² closed form | M1 |
| 6 | `tail_dependence` (λ_L, λ_U), `exceedance_corr` | A | block bootstrap | M1 |
| 7 | `hsic` — random-Fourier-feature HSIC | B | gamma moment-matched | M3 |
| 8 | `mi` — KSG, capped n | D | block permutation | M3 |
| 9 | `transfer_entropy` — Gaussian/copula default, KSG opt-in | B / D | block permutation | M3 |
| 10 | `gcm` — generalised covariance measure (conditional independence) | B | **N(0,1) closed form** | M3 |
| 11 | `codec` / `foci` — conditional screening | D | — | M3 |
| 12 | `nonlinear_acf` / `nonlinear_ccf` / `lag_dependence` | any backend | max-stat FWER over lags | M2 |
| 13 | `.ts.rolling_xi` / `rolling_dcor` / `rolling_tail_dep` — **features** | A | — | M4 |

### Not shipping — and this is a decision, not an omission

* **MGC / MGCX** — O(n² log n) plus a permutation loop, class D. The power gain
  over dCor is real but narrow (multi-scale local alternatives); the cost in a
  numpy-only library is 100–1000×. Document `hyppo` as the escape hatch.
* **HHG, Ball covariance** — O(n²) with no fast path and thin evidence of
  beating dCor/HSIC on the alternatives panels actually contain.
* **MIC / TIC** — the equitability claim did not survive the Simon–Tibshirani
  and Kinney–Atwal critiques; MIC has *lower* power than dCor on most
  alternatives at comparable cost. Do not ship it, and say why in the docs.
* **Exact KCI** — O(n³) eigendecomposition. `gcm` and `rcit` cover the use case.
* **Neural / diffusion MI** — needs torch. Out of scope by dependency policy.
* **ppscore** — its CV is not panel-safe and cannot be made so without becoming
  a different thing. `feature_screen` is the honest replacement.
* **`polars-ds` / any Polars plugin** — compiled extension, forbidden.

### The honesty trap in GCMI — read this before implementing `_info.py`

Bivariate Gaussian-copula MI is `-½ log(1 - ρ_z²)` where `ρ_z` is the Pearson
correlation of normal scores. That is a **strictly monotone function of the
van der Waerden rank correlation**. Shipping it as a "nonlinear MI" for a pair
of columns is selling Spearman in a hat. Its genuine value is (a) multivariate
`I(X; Y₁…Y_k)`, (b) conditional `I(X; Y | Z)`, (c) producing a full p×p matrix
from one covariance. The docstring must say this in the first paragraph, and
`dependence(method="gcmi")` on a bivariate pair must emit a note in `warnings`
pointing at `xi` / `dcor` / `hsic` for genuine nonlinearity.

---

## File ownership — touch ONLY your file(s)

| File | Owner | Milestone |
|---|---|---|
| `_ranks.py` | Agent A | M1 |
| `_coef.py` | Agent B | M1 |
| `_energy.py` | Agent C | M1 |
| `_info.py` | Agent D | M1 (gcmi) / M3 (ksg, te) |
| `_null.py` | Agent E | M2 |
| `_lag.py` | Agent F | M2 |
| `_panel.py` | Agent G | M2 |
| `_matrix.py` | Agent H | M2 |
| `_kernel.py` | Agent I | M3 |
| `_cond.py` | Agent J | M3 |
| `_rolling.py` + `namespaces/ts.py` additions | Agent K | M4 |
| `tests/test_depend_*.py` | Agent L | all |
| `__init__.py`, registry specs, docs | orchestrator (do not create) | — |

---

## Interfaces you may assume exist

### `_ranks.py` (Agent A) — the shared primitive layer

Everything in M1 bottoms out here. Get this right and the rest is arithmetic.

```python
def ranks(x: np.ndarray, *, axis: int = -1, method: str = "average") -> np.ndarray:
    # method in {"average", "ordinal", "dense"}. Uses argsort + put_along_axis
    # scatter -- NEVER argsort(argsort(x)) (measured 1.63x penalty).
    # Batched over leading axes so (n_entities, n_windows, w) works in one call.

def normal_scores(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    # Phi^{-1}(rank / (n+1)) via econ._common.norm_ppf. The copula transform.

def sliding_ranks(x: np.ndarray, window: int) -> np.ndarray:
    # sliding_window_view + batched ranks. Returns (..., n - window + 1, window).
    # Measured 5.5 ms for T=2500, w=60. NaN-aware: a window containing NaN
    # yields an all-NaN row rather than silently ranking the NaN last.

def dominance_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Q_i = #{j != i : x_j < x_i and y_j < y_i}, the bivariate rank.
    # This one primitive powers hoeffding_d, bkr, tau_star AND the O(n log n)
    # univariate distance covariance. Implement the BLOCKED sqrt(n) version
    # first: sort by x, process in blocks of b = ceil(sqrt(n)), count
    # within-block by direct comparison and cross-block by a cumulative
    # histogram over y-ranks -> O(n^1.5) with no Python loop over n.
    # A vectorised bottom-up merge-sort (log n numpy passes) is the O(n log n)
    # upgrade; do NOT attempt it until a benchmark shows the blocked version is
    # the bottleneck. Correctness first.

def pairwise_complete(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    # Drop rows where either is NaN/inf. Returns (x, y, n_obs). Every public
    # statistic goes through this and reports the surviving n_obs.
```

### `_coef.py` (Agent B)

```python
def xi(x, y, *, tie_break="average", ties="auto") -> float:
    # Chatterjee (2021) JASA. Sort by x (stable); r_i = rank of y in that order.
    #   no ties in y:  xi = 1 - 3 * sum|r_{i+1} - r_i| / (n^2 - 1)
    #   ties in y:     xi = 1 - n * sum|r_{i+1} - r_i| / (2 * sum l_i (n - l_i))
    #                  with l_i = #{j : y_j >= y_(i)}
    # ASYMMETRIC by construction: xi(x, y) != xi(y, x). Both are meaningful --
    # xi(x, y) ~ "is y a function of x". Never silently symmetrise.

def xi_null_sd(n: int, *, ties: np.ndarray | None = None) -> float:
    # sqrt(2/5 / n) for continuous y. VERIFIED empirically in this contract:
    # n=200 -> 0.4151, n=1000 -> 0.3972, n=5000 -> 0.3943 against theory 0.4.
    # With ties the variance is larger; when `ties` is supplied and the tie
    # fraction exceeds 5%, return NaN and force the caller onto a permutation
    # null rather than reporting a p-value known to be wrong.

def xi_matrix(X: np.ndarray) -> np.ndarray:
    # (p, p) DIRECTED matrix, M[i, j] = xi(X[:, i] -> X[:, j]).
    # MUST be O(p) argsorts, not O(p^2): for each column i, reorder ALL other
    # columns' ranks by i's argsort in one take_along_axis, then one vectorised
    # diff/abs/sum over axis 0. p=200, n=5000 must complete in seconds.

def hoeffding_d(x, y) -> float          # via dominance_counts
def tau_star(x, y) -> float             # Bergsma-Dassios-Yanagimoto; optional, M2
def tail_dependence(x, y, *, q=0.05) -> tuple[float, float]
    # Nonparametric lambda_L = P(F_y(Y) < q | F_x(X) < q) estimated as
    # #{both below q} / (n q). Return (lower, upper). q is a FITTED threshold
    # under invariant 7 when used inside a transformer.
def exceedance_corr(x, y, *, q=0.05) -> tuple[float, float]
def tail_dependence_matrix(X, *, q=0.05) -> np.ndarray
    # ONE matmul: I = (ranks(X) < q*n).astype(float64); M = I.T @ I / (n*q).
```

### `_energy.py` (Agent C)

```python
def dcov2(x, y, *, bias_corrected=True, max_n=10_000, seed=0) -> DcorResult:
    # Distance covariance. Two paths, chosen automatically:
    #  * univariate x and y -> O(n log n) exact via dominance_counts and sorted
    #    prefix sums (Huo & Szekely 2016). No n x n matrix is ever formed.
    #  * multivariate      -> BLOCKED O(n^2): accumulate row sums and the grand
    #    sum over 2000 x 2000 tiles, never materialising the full matrix.
    #    MEASURED: full n=10_000 is 800 MB, n=20_000 is 3.2 GB -- so above
    #    `max_n`, subsample WITH the supplied seed and set approximate=True and
    #    n_subsample in the result. Never silently OOM, never silently truncate.
    # bias_corrected=True uses U-centering (Szekely & Rizzo 2014):
    #   A~_ij = a_ij - (1/(n-2)) sum_k a_ik - (1/(n-2)) sum_k a_kj
    #           + (1/((n-1)(n-2))) sum_kl a_kl,  diag 0
    #   Omega = (1/(n(n-3))) sum_{i != j} A~_ij B~_ij

def dcor_pvalue(result: DcorResult, *, method="auto") -> float:
    # CAREFUL -- this is the subtlety most implementations get wrong.
    # The Szekely-Rizzo t-test (T = sqrt(v-1) R*/sqrt(1-R*^2), v = n(n-3)/2,
    # t_{v-1}) is derived for the HIGH-DIMENSIONAL limit p -> infinity. It is
    # NOT a valid null for univariate x and y, which is the panel case.
    # method="auto" therefore selects:
    #   dim(x) >= 10 and dim(y) >= 10 -> "t"
    #   otherwise                     -> "gamma" (moment-matched to the exact
    #                                     permutation mean/variance of n*dCov^2)
    # and any serially dependent input is routed to _null regardless.
    # Write the test that shows the t-test is miscalibrated at p=1.

def partial_dcor(x, y, z) -> float     # Szekely & Rizzo (2014), M3
```

### `_info.py` (Agent D)

```python
def gcmi(x, y, *, biascorrect=True) -> float
    # Gaussian-copula MI in NATS. z = normal_scores; then
    #   I = 0.5 * (logdet(S_xx) + logdet(S_yy) - logdet(S))
    # with the Ince et al. (2017) finite-sample entropy bias correction
    # (the digamma/psi terms). gcmi upstream is GPL -- CLEAN-ROOM from the
    # paper, and validate against an analytic Gaussian ground truth where the
    # true MI is known in closed form, not against the GPL code.
    # READ THE "honesty trap" SECTION ABOVE before writing the docstring.

def gcmi_matrix(X) -> np.ndarray        # one covariance of normal scores
def gcmi_conditional(x, y, z) -> float  # I(X;Y|Z) from four log-dets

def mi_ksg(x, y, *, k=5, max_n=20_000, seed=0) -> float   # M3
    # Kraskov-Stogbauer-Grassberger algorithm 1. Needs (a) the k-th nearest
    # neighbour Chebyshev distance per point and (b) marginal counts strictly
    # inside that radius. (b) is EXACTLY what
    # panelary._internal._numpy_stats.chebyshev_neighbour_counts already provides
    # (sort-windowed, blocked, exact parity with KDTree p=inf) -- reuse it,
    # do not write a second neighbour search. For (a) use the same sorted
    # window with a per-block partition. Digamma: implement psi() in
    # econ._common via the standard Lanczos/asymptotic recurrence and test it
    # against known values (psi(1) = -gamma, psi(1/2) = -gamma - 2 ln 2).
    # Returns raw nats. Do NOT normalise silently; see `mi_to_r`.

def mi_to_r(mi: float) -> float         # sqrt(1 - exp(-2 * mi)), documented as
                                        # "the Gaussian-equivalent correlation",
                                        # never presented as MI itself.
def variation_of_information(x, y) -> float   # a true metric; feeds cluster/

def transfer_entropy(source, target, *, lag=1, estimator="gaussian") -> TEResult
    # estimator="gaussian" is the default and is CHEAP: Gaussian TE equals
    # half the Granger-causality F-statistic's log-likelihood ratio
    # (Barnett, Barrett & Seth 2009), so it is two OLS fits via econ._common.ols
    # and a closed-form chi^2 null. estimator="copula" applies normal_scores
    # first. estimator="ksg" is opt-in and capped.
    # The docstring MUST state that TE is not evidence of causation and that
    # under a common latent driver both directions inflate.
```

### `_null.py` (Agent E) — the differentiator

Do not write a bootstrap. Five already exist in
`panelary/validation/_bootstrap.py`. This file adds only what is missing and
wires the rest to dependence statistics.

```python
def circular_shift(n: int, *, n_resamples: int, seed: int) -> np.ndarray
    # (B, n) index matrix of cyclic rotations. O(n) per draw and preserves the
    # shifted series' autocorrelation EXACTLY. Caveat that must be enforced in
    # code, not just documented: there are only n distinct rotations, so the
    # smallest achievable p-value is 1/n. If n_resamples > n, cap it, set
    # exact=True, and enumerate all n rotations instead of sampling.

def phase_randomise(x, *, n_resamples, seed) -> np.ndarray
    # FT surrogate: rfft, randomise phases (keep the DC and Nyquist terms real),
    # irfft. Preserves the power spectrum, hence the linear autocorrelation
    # exactly; destroys nonlinear structure. Gaussianises the marginal.

def iaaft(x, *, n_resamples, seed, max_iter=200, tol=1e-8) -> np.ndarray
    # Schreiber & Schmitz (1996). Alternate: (1) impose the target amplitude
    # spectrum in Fourier space, (2) impose the target rank order in the time
    # domain. Preserves BOTH the spectrum and the exact empirical marginal --
    # which is what makes it the right null for a rank statistic like xi on a
    # fat-tailed financial series. Converge on rank-order change, report
    # n_iter and converged in the result.

def auto_block_length(x, *, kind="circular") -> int
    # Politis & White (2004) with the Patton, Politis & White (2009) correction.
    # Fallback ceil(n ** (1/3)) when the flat-top spectral estimate degenerates.
    # Prefix invariance note: this depends on n by construction, so it is a
    # FITTED parameter -- freeze it at fit time, never recompute per fold-end.

def common_time_indices(dates: np.ndarray, *, n_resamples, seed, block: int) -> np.ndarray
    # THE PANEL NULL. Given the sorted unique date axis, return (B, T) index
    # matrices that permute whole DATE COLUMNS in blocks, applied IDENTICALLY
    # to every entity. This breaks the x-y temporal alignment while preserving
    # (a) each date's full cross-section, hence every common macro shock, and
    # (b) serial structure up to the block length. Per-entity independent
    # shuffles do neither and are invalid whenever entities load on a common
    # factor -- which, in an equity panel, is always.

def entity_permutation_indices(...)   # the OTHER panel null: reassign which
    # entity's y pairs with which entity's x. Tests a different hypothesis
    # (cross-sectional matching) -- documented side by side so users pick
    # deliberately.

def pvalue(observed: float, null_draws: np.ndarray, *, alternative="greater") -> float
    # (1 + #{null >= observed}) / (1 + B). The +1 is not optional; without it a
    # p-value of exactly 0 is reported and every downstream FDR breaks.

def gamma_pvalue(observed: float, mean: float, var: float) -> float
    # Moment-matched gamma tail via econ._common. Used by dcor (p small) and
    # hsic. Cheaper than B resamples by ~3 orders of magnitude.
```

**Null policy table — this is the contract, encode it as a dict and test it.**

| Statistic | serially independent | serially dependent | panel, common shocks |
|---|---|---|---|
| `xi` | `asymptotic` (N(0, 2/5)) | `block` or `iaaft` | `common-time` |
| `spearman` / `pearson` | `asymptotic` | `block` | `common-time` |
| `hoeffding_d` | `asymptotic` | `block` | `common-time` |
| `dcor` (p ≥ 10) | `t` | `block` | `common-time` |
| `dcor` (p < 10) | `gamma` | `block` | `common-time` |
| `gcmi` | `asymptotic` (χ²) | `block` | `common-time` |
| `hsic` | `gamma` | `shift` or `iaaft` | `common-time` |
| `mi_ksg` | `permutation` | `block` | `common-time` |
| `transfer_entropy` | `asymptotic` (χ²) | `block` | `common-time` |
| `gcm` | `asymptotic` (N(0,1)) | HAC-studentised | `common-time` |

`null="auto"` implements exactly this table. The serial-dependence branch is
chosen by a lag-1 rank-autocorrelation pre-check at |ρ| > 0.2, and the choice is
reported in `null_method` — never silent.

### `_lag.py` (Agent F)

```python
def lag_dependence(x, y, *, max_lag=24, method="xi", null="auto", ...) -> pl.DataFrame
    # One row per lag in [-max_lag, max_lag] (or [0, max_lag] for acf).
    # Columns: lag, estimate, p_value, p_value_adj, null_method, n_obs.
def nonlinear_acf(x, *, max_lag=24, method="xi", ...) -> pl.DataFrame
def nonlinear_ccf(x, y, *, max_lag=24, method="dcor", ...) -> pl.DataFrame
def optimal_lag(profile: pl.DataFrame) -> int
```

Lag scans are where naive libraries manufacture false discoveries: 49 lags × 200
features is 9800 tests. **`p_value_adj` is mandatory, not optional**, and the
default is `romano_wolf` over the shared (B × n_lags) resample matrix, which is
already implemented in `validation._selection_stats.romano_wolf` and exploits
the enormous correlation between adjacent lags far better than Holm. Draw the
resamples ONCE and reuse them across all lags — a fresh resample per lag is both
wasteful and destroys the joint structure Romano-Wolf needs.

### `_panel.py` (Agent G)

```python
def by_entity(df, x, y, *, entity, time, statistic, min_obs=50) -> pl.DataFrame
    # Per-entity estimates. Extract the frame ONCE as a time-sorted numpy array
    # and slice by group offsets computed from the entity run-lengths; do NOT
    # partition_by into thousands of small frames.

def aggregate(per_entity: pl.DataFrame, *, how="fisher") -> AggregateResult
    # how in {"fisher", "median", "precision", "mean"}.
    # "fisher"    : arctanh transform, inverse-variance weight by (n_i - 3),
    #               back-transform. The default for correlation-like statistics.
    # "precision" : random-effects (DerSimonian-Laird) weights.
    # ALWAYS return a heterogeneity statistic (I^2 or Cochran's Q). A pooled
    # dependence number without heterogeneity is the panel equivalent of a
    # Sharpe ratio without a t-stat.

def pooled(df, ..., demean="none") -> Result
    # demean in {"none", "entity", "time", "two-way"}. Reuse econ._hdfe.
    # The transform applied MUST appear in the result's `transform` field --
    # pooled-raw and pooled-two-way-demeaned are different statistics and
    # conflating them is how Simpson's paradox gets into a research log.

def min_obs_for(method: str) -> int
    # xi: 50, dcor: 30, hoeffding_d: 50, gcmi: 10 * dim, mi_ksg: 100 and k < n/10,
    # tail_dependence: 20 / q. Below the threshold: return NaN and a warning,
    # never a noisy number. Unbalanced panels are the norm; this gate is what
    # stops a 12-observation entity from entering a Fisher average.
```

### `_matrix.py` (Agent H)

```python
def dependence_matrix(df, cols, *, method="xi", by="entity", subsample=5000,
                      seed=0, symmetrise=None) -> pl.DataFrame
    # p x p. Directed for xi/transfer_entropy, symmetric otherwise; `symmetrise`
    # in {None, "max", "mean"} and the default None PRESERVES direction.
    # Automatic subsampling to `subsample` rows for the scan (a matrix does not
    # need n=100_000 to rank features) with approximate=True recorded.
    # Must dispatch to the batched *_matrix kernels, never a p^2 Python loop.

def to_distance(M, *, kind="angular") -> np.ndarray
    # sqrt(0.5 * (1 - M)) for correlation-like input; variation-of-information
    # for MI input. Feeds panelary.cluster and select.pfa directly.
    # A pairwise-complete matrix need not be PSD -- offer eigenvalue clipping
    # and report the smallest eigenvalue before repair.

def feature_screen(df, target, features, *, method="xi", by="entity",
                   null="auto", correction="benjamini_hochberg", ...) -> pl.DataFrame
    # The headline user-facing function. Ranked table: feature, estimate,
    # p_value, p_value_adj, n_obs, n_entities, heterogeneity, null_method,
    # warnings. Wrapped as ScreenSelector(PanelTransformer) with
    # panel_safe=True, leakage_safe=True -- the latter ONLY because the
    # transformer re-runs the screen inside fit() on the training fold.
```

Also in M3, **upgrade `select.mrmr`**: it currently uses absolute *Pearson* for
both relevance and redundancy ([`panelary/select/_methods.py`](../../panelary/select/_methods.py)).
Add `relevance=` / `redundancy=` parameters accepting any `depend` method,
defaulting to `"pearson"` for exact backward compatibility. Same for
`select.correlation`, which should gain a nonlinear pruning mode.

### `_kernel.py` (Agent I) — M3

```python
def rff(x, *, n_features=256, bandwidth=None, seed=0) -> np.ndarray
    # z(x) = sqrt(2/D) * cos(w x + b), w ~ N(0, 1/sigma^2), b ~ U(0, 2 pi).
    # bandwidth=None -> median heuristic. THIS IS A FIT (invariant 7): the
    # bandwidth AND the drawn (w, b) must be frozen at fit time and reused for
    # every pair in a matrix, or the entries are not comparable.

def hsic(x, y, *, n_features=256, ...) -> HSICResult
    # HSIC ~= || Zx^T Zy / n ||_F^2 with Zx, Zy column-centred. Two gemms.
    # MEASURED: n=100_000, D=256 -> 312 ms; D=64 -> 76 ms. Cost is O(n D^2),
    # so D is the knob: D=256 pairwise, D=32-64 for full matrices.
    # Null: gamma moment-matching of the biased HSIC statistic.
    # Note in the docstring: with distance-induced kernels HSIC and distance
    # covariance are the SAME statistic (Sejdinovic et al. 2013) -- one engine,
    # two names, and the user should be told rather than sold both.

def hsic_lasso(X, y, *, ...) -> np.ndarray    # non-negative least squares on
    # kernel-derived features; coordinate descent in numpy. Non-redundant
    # nonlinear screening. Only if M3 has room -- feature_screen + foci covers
    # most of the need.
```

### `_cond.py` (Agent J) — M3

```python
def gcm(x, y, z, *, regressor="ols", hac_lags=None) -> GCMResult
    # Generalised covariance measure (Shah & Peters 2020). Regress x ~ z and
    # y ~ z, take residuals rx, ry, and test whether E[rx * ry] = 0:
    #   T = sqrt(n) * mean(rx ry) / sd(rx ry)  ->  N(0, 1) under H0.
    # Cheapest defensible conditional-independence test in existence: two OLS
    # fits (econ._common.ols) and a closed-form null. For serially dependent
    # data replace sd() with a Newey-West long-run variance --
    # econ._common.newey_west_scalar with auto_bandwidth. This single
    # substitution is what makes it panel-valid, and it is why gcm beats KCI
    # here despite KCI having more power on paper.

def codec(y, z, x) -> float     # Azadkia & Chatterjee (2021) T(Y | Z, X);
                                # nearest neighbours in the conditioning space
                                # via chebyshev_neighbour_counts. Cap dim(Z).
def foci(df, target, features, *, k=None) -> list[str]   # greedy forward
                                # selection on codec. Stop when the increment
                                # turns non-positive.
def rcit(x, y, z, *, n_features=128, seed=0)             # RFF version of KCI,
                                # if time allows.
```

### `_rolling.py` + `.ts` namespace (Agent K) — M4

The feature-engineering payoff. These emit **columns**, not tables.

```python
# pl.col("ret").ts.rolling_xi("mkt_ret", window=60)
# pl.col("ret").ts.rolling_dcor("mkt_ret", window=60)
# pl.col("ret").ts.rolling_tail_dep("mkt_ret", window=250, q=0.10)
# pl.col("ret").ts.rolling_gcmi("vol", window=60)
```

Implementation: `map_batches` over the two-column struct per entity, then
`sliding_ranks` + one batched vectorised reduction. **Measured: 1.97 s for 200
entities × 2500 dates at w=60; ~49 s for 5000 entities single-threaded.**

Prefix invariance is the whole game here and is easy to break in three ways —
all three get a test:

1. The window must be fixed, never `n // k`.
2. Ranks must be computed **inside** the window, never globally then sliced.
3. A tail-dependence quantile or kernel bandwidth must come from the window (or
   be a user-supplied constant), never from the full series.

Every op registers a `FeatureSpec` with `panel_safe=True`, `leakage_safe=True`,
`source` = the paper, `license` = the clean-room provenance. `registry.audit()`
will reject a missing or non-permissive licence.

---

## Public API

```python
import panelary as pk

# pairwise, with an honest null
pk.depend.dependence(df, "x", "y", method="xi", by="entity", null="auto")

# p x p, batched, subsampled, directed
pk.depend.dependence_matrix(df, cols, method="xi", by="entity")

# the headline: rank features against a target, leak-safely
pk.depend.feature_screen(df, target="fwd_ret_20d", features=cols,
                         method="dcor", by="entity", null="common-time",
                         correction="benjamini_hochberg")

# lag structure with FWER across lags
pk.depend.nonlinear_ccf(df, "x", "y", max_lag=24, method="xi", null="iaaft")

# information theory
pk.depend.mutual_information(df, "x", "y", estimator="gcmi")
pk.depend.transfer_entropy(df, "src", "tgt", lag=1, estimator="gaussian")

# conditional
pk.depend.conditional_dependence(df, "x", "y", given=["z"], method="gcm")
pk.depend.foci(df, target="fwd_ret_20d", features=cols)

# features
df.with_columns(pl.col("ret").ts.rolling_xi("mkt", window=60).over("ticker"))
```

Every analysis function returns a `pl.DataFrame` with a **fixed schema** —
`estimate, p_value, p_value_adj, method, estimator, null_method, n_resamples,
block_length, direction, lag, n_obs, n_entities, coverage, heterogeneity,
transform, approximate, seed, warnings` — with irrelevant fields null. Fixed
schema means `pl.concat` across methods just works, which is what makes a
screening report composable.

---

## Tests — `tests/test_depend_*.py` (Agent L)

`--strict-markers` is on; register any new marker in `pyproject.toml` first.

**`test_depend_calibration.py` — the most important file in this module.**
Reproduce the table in "Why this module exists" as an assertion:

* two independent AR(1) series, φ ∈ {0, 0.7, 0.95}, n=500, ≥2000 reps;
* assert the **i.i.d. null is broken** at φ=0.95 (type-I error > 0.30) — this
  test exists to prove the bug is real and must never be "fixed" by loosening it;
* assert `null="block"`, `"shift"` and `"iaaft"` are calibrated
  (type-I error ∈ [0.03, 0.08] at nominal 5%);
* the same for a panel with a common factor: per-entity shuffles must fail and
  `null="common-time"` must pass. Mark `slow`.

**`test_depend_leakage.py`** — the module's leakage regression suite, modelled
on `tests/test_detect_leak_safety.py`:

* a deliberately leaky `ScreenSelector` that screens on the full frame must be
  detected (out-of-fold score materially better than the honest version);
* fitting an RFF bandwidth or a tail quantile globally must be detected;
* `assert_no_lookahead` over every `.ts.rolling_*` op.

**`test_depend_prefix.py`** — bitwise `f(x[:T])[t] == f(x[:T+k])[t]` for every
rolling op and every lag profile, at several `T`.

**`test_depend_oracle.py`** — parity against external references, each behind
`pytest.importorskip`, tolerance stated per statistic:

| Ours | Oracle | Licence | Note |
|---|---|---|---|
| `xi` | `scipy.stats.chatterjeexi` | BSD-3 | verify the SciPy version that has it before relying on it; skip if absent |
| `dcor`, `partial_dcor` | `dcor` package | MIT | the primary numerical oracle |
| `dcor`, `hsic` | `hyppo` | permissive | secondary |
| `hoeffding_d` | R `energy`/hand-computed | — | small-n hand cases |
| `gcmi` | analytic Gaussian MI | — | closed-form ground truth, **not** the GPL `gcmi` package |
| `mi_ksg` | `ennemi` / `NPEET` | MIT | |
| `psi` (digamma) | known values | — | ψ(1)=−γ, ψ(½)=−γ−2ln2 |

`gcmi`, `minepy`, `Tigramite`, `IDTxl`, `JIDT` are **GPL — reference only**.
Clean-room from the papers; record `source` and `license` on every `FeatureSpec`
and let `registry.audit()` enforce it.

**`test_depend_zoo.py`** — a relation zoo asserting each statistic detects what
it should and, just as importantly, **fails where it is documented to fail**:

* linear, monotone-exponential, quadratic/U, sinusoid, circle, checkerboard/XOR,
  heteroskedastic-only (zero mean dependence, strong variance dependence),
  heavy-tailed (t₃), discrete/mixed with heavy ties;
* pin the documented weaknesses: ξ on the circle, GCMI on anything
  non-monotone (see the honesty trap), tail dependence on a Gaussian copula
  (λ = 0 in the limit and the estimator must not manufacture one).

**`test_depend_perf.py`** (marked `benchmark`) — assert the measured budgets
above are not regressed by more than 2×: rolling ξ on 200 × 2500 × w=60 under
5 s; RFF-HSIC n=100 000 D=256 under 1 s; `xi_matrix` p=200 n=5000 under 30 s;
`dcov2` raising rather than allocating above `max_n`.

Plus the standing packaging guardrails: `tests/test_import_hygiene.py` (no
top-level optional imports), `tests/test_dependency_drift.py` (no new mandatory
dependency — `depend` adds **zero**), `tests/test_wheel_guardrails.py`.

---

## Milestones

| M | Contents | Ships |
|---|---|---|
| **M1** | `_ranks`, `_coef`, `_energy`, `_info` (gcmi), `dependence`, `dependence_matrix`, closed-form nulls | A real nonlinear dependence battery with correct i.i.d. inference. Zero new dependencies. |
| **M2** | `_null`, `_lag`, `_panel`, `_matrix`, `feature_screen` | The differentiator: valid inference under serial and cross-sectional dependence, FWER over lags, panel aggregation. |
| **M3** | `_kernel`, `_cond`, KSG MI, transfer entropy, `mrmr` upgrade, `foci` | Kernel + conditional layer; nonlinear redundancy pruning. |
| **M4** | `_rolling`, `.ts` namespace ops, `FeatureSpec` registration, docs | Dependence as *features*, not just analysis. |

M1 alone is shippable and useful. M2 is what makes it defensible. Do not start
M3 before `test_depend_calibration.py` is green.

---

## Integration points

* `panelary.cluster` — `to_distance()` output is a drop-in codependence matrix
  for hierarchical/ONC clustering.
* `panelary.select` — `mrmr` relevance/redundancy upgrade; `correlation` gains a
  nonlinear pruning mode; `pfa` can consume a nonlinear matrix.
* `panelary.reduce` — nonlinear dependence matrices are noisier than linear
  ones; if a Marchenko-Pastur denoiser lands in `reduce`, route matrices
  through it before clustering.
* `panelary.evolve` — `Descriptors.max_corr` is currently linear; a nonlinear
  redundancy descriptor is a natural follow-on (out of scope here).
* `panelary.detect` — `nonlinear_acf` on residuals is a nonlinearity diagnostic
  complementing the bubble monitors.

## Open questions for the implementer to resolve with a benchmark, not an opinion

1. Is the blocked O(n^1.5) `dominance_counts` fast enough at n = 100 000, or is
   the vectorised merge-sort upgrade actually needed? Measure before writing it.
2. What D does RFF-HSIC need for the p-value to be within 10% of exact HSIC at
   n = 5000? The D=256 default is a guess; replace it with a measured number.
3. Does devolatilising materially change the *ranking* of features in
   `feature_screen` on real data, or only the absolute values? If only the
   latter, demote `devol=` from a headline parameter to a documented option.
4. Is `xi_matrix` at p=500, n=5000 tolerable, or does the matrix path need
   subsampling below the current 5000 default?
