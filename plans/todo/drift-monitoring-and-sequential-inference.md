# `panelary/monitor/` — drift monitoring, SPC and anytime-valid sequential inference: build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.**

This plan covers distribution **drift** between a reference and a current window
(univariate and multivariate), **shift correction** (label shift, covariate-shift
density ratios feeding weighted conformal), **statistical process control** with an
honest Phase I / Phase II split, and **anytime-valid sequential inference**
(confidence sequences, e-processes, mSPRT, e-detectors, e-BH). It also covers
prefix-invariant **rolling drift features** for panels.

The core install stays at **numpy + polars**. SciPy, scikit-learn and numba are
optional accelerators or test oracles only (see §10). Every capability has a pure
numpy/polars path.

Reuse, do not reinvent (all verified by reading the code on 2026-09-29):
`panelary.depend._null` (`pvalue`, `gamma_pvalue`, `block_permutation_indices`,
`common_time_indices`, `entity_permutation_indices`, `pair_block_length`,
`serial_dependence`, `SerialDependenceWarning`), `panelary.depend._ranks` (`ranks`,
`normal_scores`, `lag1_rank_autocorr`), `panelary.depend._kernel` (`RFFMap`,
`median_heuristic`), `panelary.depend._energy` (`_row_dist_sums`,
`_pairwise_dist_tile`, `_blocked_terms` tiling), `panelary.detect._monitors`
(`page_cusum`, `page_cusum_expr`, `shiryaev_roberts`, `focus`),
`panelary.validation._selection_stats` (`benjamini_hochberg`, `benjamini_yekutieli`,
`holm_bonferroni`, `romano_wolf`, `_bh_family`, `MultipleTestResult`),
`panelary.validation._bootstrap` (`block_bootstrap_indices`, `resolve_segments`),
`panelary.validation._cv.purged_calibration_split`,
`panelary.core.model_selection.PurgedKFold`, `panelary.conformal._intervals`
(`conformal_quantile`, `nexcp_quantile`), `panelary.econ._common` (`chi2_sf`, `f_sf`,
`t_sf`, `_betainc`, `norm_cdf`, `norm_ppf`), `panelary.quality._common`
(`CheckResult`, `ValidationReport`, `jsonable`, `canonical_json`),
`panelary.quality._leakage.check_fitted_state`, `panelary.embed._rff.orthogonal_gaussian`
(lazy import), `panelary.testing` (`assert_no_lookahead`, `assert_prefix_invariant`,
`assert_no_train_test_leak`), `panelary._internal._deps` (`require`, `have`).

From sibling plans, once they land: `panelary.covariance` (`estimate`, `CovEstimate`,
`Schedule`; sibling 3) and `validation/_hac.py` (sibling 1).

---

## 1. Why this exists — measured evidence

Everything in this section was measured on 2026-09-29 on an Apple M5 Pro (15 cores)
with NumPy 2.5.3 + Accelerate, polars 1.44.2 and Python 3.13.12. SciPy 1.18.1 was used
**only as an oracle**. The scripts are listed in §9.4 and must be checked in as
`benchmarks/monitor/` at M6. These are measurements, not estimates. Every estimate
elsewhere in this plan is labelled as one.

Most drift tooling ships a divergence number and a fixed threshold, and leaves it
there. Three failures follow, and each one is large.

### 1.1 The PSI rule of thumb is a sample-size accident

We drew both samples from **the same** N(0,1), used 10 quantile bins frozen on the
reference, applied Jeffreys +0.5 smoothing, and ran 4000 replications.

| n = m | P(PSI > 0.10) | P(PSI > 0.25) | Empirical 95% quantile | Theory 95%: (1/n+1/m)·χ²₉(0.95) | Size of the χ² rule at 5% |
|---|---|---|---|---|---|
| 50 | **96.4%** | **60.6%** | 0.588 | 0.677 | 2.0% |
| 100 | **79.3%** | **15.4%** | 0.323 | 0.338 | 4.2% |
| 250 | 17.2% | 0.02% | 0.133 | 0.135 | 4.6% |
| 500 | 0.4% | 0 | 0.0698 | 0.0677 | 5.8% |
| 1 000 | 0 | 0 | 0.0331 | 0.0338 | 4.5% |
| 20 000 | 0 | 0 | 0.00172 | 0.00169 | 5.3% |
| 5 000 vs 200 | 1.9% | 0 | 0.0850 | 0.0880 | 4.0% |

- **At n = 100, the "0.1 = investigate" rule fires on pure noise 79% of the time.**
- **At n = 20 000, the same rule sits 58× above the noise floor.** Real but modest
  drift goes unflagged.
- The asymptotic null PSI ≈ (1/n + 1/m)·χ²_{B−1} (Yurdakul & Naranjo 2020) is
  calibrated from n ≈ 250. At n = 50 it is conservative (2%) because of the
  smoothing.

The library must therefore report a p-value **and** an effect size. It must also say
"insufficient n" when the smallest detectable PSI exceeds the effect floor (§6.2).

### 1.2 Two-window tests on persistent features are wrong by an order of magnitude

We took adjacent windows (n = m = 250) of **one stationary** AR(1) series, so there is
no drift. We ran 2000 replications of the i.i.d. KS test. For a date-block
permutation null (block length 25, B = 199), we ran 400 replications.

| φ | i.i.d. KS type-I error at nominal 5% | Block-permutation null (L = 25) |
|---|---|---|
| 0.00 | 5.5% | 5.2% |
| 0.50 | **23.4%** | 5.8% |
| 0.90 | **72.6%** | 10.2% |
| 0.98 | **95.0%** | **31.0%** |

This is the drift-monitoring version of the finding in the `depend` contract. A
persistent feature's window distribution **moves under the null**, so an i.i.d. drift
test on price levels, valuation ratios or slow macro series reports noise as drift
almost every time. Block permutation repairs moderate persistence. It **cannot**
repair near-unit-root persistence at these window lengths, because adjacent windows
are genuinely non-exchangeable. The contract response is a **persistence refusal**
(§5, invariant 9). It is not a bigger block length.

### 1.3 Peeking at fixed-n intervals destroys their guarantee

This is the monitoring failure mode. We took a Bernoulli(0.10) stream, computed a
95% Wilson interval (the interval `truepoint/inference/errorbars.py:wilson_interval`
uses), and checked it **at every step** from t = 30 to t = 10⁴. We ran 2000 paths.

| Procedure | P(truth ever excluded) | Width at t = 10⁴ | Cost, 2000 streams × 10⁴ steps |
|---|---|---|---|
| Wilson 95% interval, re-checked every step | **53.5%** | 0.0118 | — |
| Predictable plug-in empirical-Bernstein CS (Waudby-Smith & Ramdas 2024) | **1.9%** | 0.0254 (2.16×) | **559 ms**, cumsums only |
| Hedged-capital betting CS, grid G = 1000 (100 paths) | 4.0% | 0.0257 | 239 ms **per stream** |

Anytime validity costs about 2.2× in width at t = 10⁴. That is the honest price of
being allowed to look whenever you like, and it is what a monitoring product has to
pay. At this horizon the betting CS is **no tighter** than PrPl-EB and is about 400×
slower per stream. That fixes the default (§6.7).

### 1.4 Measured facts that dictate the design

| Fact | Measurement | Consequence |
|---|---|---|
| One pooled stable argsort per feature row yields KS, CvM, AD, W1 and energy together | F = 500, n = m = 5000: **312 ms** for all 5 statistics. SciPy per-feature loop: 2 578 ms (8.3×) | The univariate engine is a **single sort**. Everything else is cumsum and arithmetic (§6.1). |
| Parity with SciPy | KS 5.6e-17, W1 4.5e-17, energy 3.1e-17, CvM 1.8e-13, AD (unstandardised A²ₖN) ≤ 1.1e-16 | Exact reimplementation. Statistic tolerance is 1e-12 in tests. |
| Presorted frozen reference + `argsort(kind="stable")` of two sorted runs (timsort merges the runs in O(N)) | Merge step 24 ms vs 217 ms for a full argsort. End-to-end 81 ms vs 246 ms | When the reference is frozen, sort it **once**. Each new current window costs O(m log m + N). |
| Thread-parallel feature chunks (numpy sort/cumsum release the GIL) | 291 ms → 78 ms on 8 threads (3.7×), 66 ms on 12 threads. **Bitwise identical** | Deterministic parallelism with no extra dependency (§6.9). |
| Shared permutation plan: sort once, then gather the permuted labels | B = 200, F = 500, N = 10⁴: 3.83 s, i.e. **19 ms per permutation for all features** | Permutation nulls are affordable. The shared plan keeps the joint null that Romano–Wolf needs. |
| Binning against frozen edges: numpy `searchsorted` + `bincount` | 5·10⁷ elements in 0.99 s (51 M/s) | Numpy fallback path. |
| Binning with polars `cut` (native, parallel over columns) + `bincount` | 0.10 s + 0.09 s (≈ 260 M elements/s end to end). Codes equal numpy's | Default binning path for panels (§6.2). |
| Trailing-reference PSI from date prefix sums of the histogram cube | 135 dates × 50 features in **0.9 ms** | Every trailing reference window costs O(B) (§6.8). |
| EWMA ARL₀ by Brook–Evans Markov chain (m = 301 states, λ = 0.1) | L = 2.814 → **499.4** (published value: 500). L = 2.7 → 368.9. 8 ms per limit | The Markov chain is the default ARL engine for EWMA and CUSUM. |
| A single simulation yields the whole ARL(L) curve (survival counting) | R = 2·10⁴, 281 limits in 5.1 s. Agrees with the Markov chain to 0.3–1.2% | Simulation calibrates charts that have no chain (runs rules, MCUSUM, shrunk T²) in **one pass** (§6.6). |

---

## 2. What already exists — and what this must NOT duplicate

| Existing code (verified) | What it does | Relationship |
|---|---|---|
| `quality/_report.py:quality_report(reference=)` | **dtype/schema** drift only: changed, added and removed columns | Untouched. Distribution drift is new. Its findings reuse the same `CheckResult` path. |
| `quality/_common.py:CheckResult.to_evidence`, `ValidationReport` | Byte-deterministic evidence records (`kind, locator, observed, expected, produced_by`) | **Reused.** `DriftReport.to_evidence()` emits `CheckResult`s and has no serialiser of its own. |
| `quality/_leakage.py:check_fitted_state` | Proves that a fitted object's state derives only from training rows | **Reused** in tests to prove that Phase-I chart state, frozen bin edges and density-ratio models derive only from their declared rows. |
| `explain/_stability.py:attribution_drift` | Descriptive spread of the mean SHAP value over time. `first_last_shift` uses whole-sample thirds, so it is analysis, not rowwise | Untouched. Running `drift_report` over SHAP columns gives the inferential version. This is documented, not coded. |
| `conformal/_intervals.py`: `conformal_quantile`, `nexcp_quantile` (fixed geometric weights, test weight ≡ 1), ACI, conformal PID, CQR; `_enbpi.py` | Conformal quantiles and online intervals | **Extended additively.** Add `weighted_conformal_quantile(scores, weights, test_weights, alpha)` (Tibshirani et al. 2019) and re-express `nexcp_quantile` through it, with a bitwise parity test. There is no second conformal module. |
| `detect/_monitors.py`: `page_cusum(_expr)`, `shiryaev_roberts`, `focus`, `hb_cusum`, `end_of_sample_S` | Closed-form univariate mean-shift and bubble monitors. **No calibration layer, no Phase I, no multivariate charts** | **Reused, not duplicated.** The CUSUM chart and the CUSUM e-detector call `page_cusum`. Page–Hinkley is a recipe over it (§3.3). The only additive edit is an `axis=` argument on `page_cusum` (1-D behaviour unchanged). |
| `detect/_critvals.py:mc_table` | One-pass nested-path Monte Carlo for BSADF | Only the pattern is reused: one simulation, every threshold (§6.6). No code is shared. |
| `depend/_null.py` | Resampling p-value with +1, gamma tail, block/date/entity permutation index generators, serial-dependence pre-check, `SerialDependenceWarning(LeakageWarning)` | **Reused** as the permutation-plan backend. No new generator is written. |
| `depend/_kernel.py:RFFMap` (bandwidth and frequencies frozen; copula option) | Random Fourier features | **Reused** for MMD features. `embed/_rff.py:orthogonal_gaussian` is imported lazily for orthogonal frequencies. |
| `depend/_energy.py`: tiled distance sums | O(n²) tiles for dCov | **Reused** for the multivariate energy statistic. Energy distance and MMD with a distance kernel are one statistic (Sejdinovic et al. 2013). |
| `depend/_rolling.py:window_statistic` | `sliding_window_view` + a row-batched statistic over two aligned windows | Reused **only** when reference length equals window length (b = x lagged). Otherwise a dedicated kernel is needed (§6.8). |
| `validation/_selection_stats.py`: BH, BY, Holm, Romano–Wolf, `_bh_family` | Multiplicity control | **Reused.** Add `e_benjamini_hochberg`, which is `_bh_family` applied to `min(1, 1/e)` (Wang & Ramdas 2022) — a one-function additive edit. |
| `embed/_probe.py:PreValidatedRidge` | Ridge readout with LOO/purged penalty choice | **Considered and not reused** for the classifier two-sample test. Its label-dependent penalty choice invalidates a permutation null (trap T4), and it has no batched-RHS path. The classifier test's ridge is a ~20-line closed-form solve (§6.4). |
| `core/model_selection.py:expanding_window_split / sliding_window_split` | Walk-forward splitters **anchored at the end of the sample** | **Must not be used as refit schedules.** Every boundary moves when a date is appended (trap T7). |
| `shape/_sketch.py` (CountSketch, FrequentDirections) | Feature-axis sketches | No quantile sketch is needed. With frozen edges, the histogram cube is already exactly mergeable (sum), so it registers `streaming="mergeable"`. |

A grep for PSI, KS, Wasserstein, Jensen–Shannon, Anderson–Darling, Cramér–von Mises,
MMD, Hotelling, EWMA, Shewhart, ADWIN, confidence sequences, e-values, mSPRT, KLIEP,
uLSIF, BBSE and label/covariate shift found **none of them** anywhere in `panelary/`.

---

## 3. Scope and non-goals

### 3.1 Shipping

| # | Capability | Default algorithm | Null / guarantee | M |
|---|---|---|---|---|
| 1 | KS, CvM, AD (k = 2), W1, energy (1-D) | one-sort engine (§6.1) | KS exact/asymptotic, CvM and AD standardised + asymptotic, W1 and energy by permutation | M1 |
| 2 | PSI/CSI, Jensen–Shannon, χ²/G on frozen-edge histograms | one-pass binning (§6.2) | (1/n+1/m)·χ²_{B−1} and the G-test | M1 |
| 3 | Multiplicity across features | BY (panel default), BH, Holm, Romano–Wolf max-T | — | M1 |
| 4 | Confidence sequences: bounded mean, asymptotic, normal mixture | PrPl-EB, WS-A-S-K-R 2024, Robbins/Howard | time-uniform | M2 |
| 5 | e-processes (mean, proportion, paired two-sample), mSPRT, e-detector, e-BH | betting capital, closed forms | Ville; ARL ≥ 1/α | M2 |
| 6 | Dependence-robust and panel nulls, persistence refusal | date-block / entity-paired permutation | — | M3 |
| 7 | Rolling drift features `.ts.rolling_drift`, date-level `xs_drift` | prefix sums, epoch-frozen edges | prefix-invariant | M3 |
| 8 | MMD (RFF-Hotelling, quadratic, block), multivariate energy, classifier two-sample test | §6.3–6.4 | χ²/F or permutation | M4 |
| 9 | Label shift (MLLS+BCTS, BBSE, RLLS); density ratio (logistic, RuLSIF); weighted conformal | §6.5 | — | M4 |
| 10 | SPC: Shewhart + WE/Nelson rules, EWMA, CUSUM (via detect), Hotelling T², MEWMA, MCUSUM; ARL engines; guaranteed-performance limits | §6.6 | ARL₀ (conditional) | M5 |
| 11 | numba kernels (`fast`), pruned betting CS, perf suite, docs | — | — | M6 |

### 3.2 Not shipping — decisions, not omissions

- **KLIEP**. Iterative projected gradient. uLSIF/RuLSIF are closed form with analytic
  LOO and comparable accuracy (Kanamori et al. 2009).
- **MMDAgg / MMD-FUSE**. These have the best power, but cost O(N² × #kernels × B).
  They are documented as the escape hatch (`hyppo`/reference code). The quadratic path
  plus RFF-Hotelling covers the need.
- **KDE-based KL/JS on continuous data**. Bandwidth-fragile and not needed: the
  frozen-bin JS has an exact G-test null.
- **Neural C2ST / deep kernels**. They need torch, which the dependency policy rules
  out. An optional sklearn HistGradientBoosting discriminator is the nonlinear option.
- **Sequential kernel two-sample by betting** (Shekhar & Ramdas 2024) and **sequential
  C2ST** (Podkopaev & Ramdas 2023). Deferred to a follow-up. M2 ships the mean and
  paired-difference e-processes that they generalise.
- **BOCPD, HMM, Kalman**. Owned by sibling plan 7 (§12).

### 3.3 ADWIN and Page–Hinkley — decided

- **Page–Hinkley is redundant and is not shipped.** PH_T = m_T − min_{t≤T} m_t with
  m_T = Σ_{t≤T}(x_t − x̄_t − δ) is exactly Page's CUSUM applied to the input minus its
  expanding mean. It is one line over existing code:
  `detect.page_cusum_expr(pl.col(x) - pl.col(x).cum_sum() / pl.col(x).cum_count(), drift=δ)`.
  It ships as a documented recipe, plus a test that pins the identity against `river`
  (oracle, `importorskip`).
- **ADWIN is not shipped**, for four reasons:
  1. Its guarantee is a per-check union bound, not ARL control (Bifet & Gavaldà 2007).
  2. The exact form scans O(W) cuts per step. ADWIN2's exponential histograms are a
     branchy per-stream structure that does not vectorise across streams in numpy.
  3. The **e-detector** (§6.7, Shin, Ramdas & Rinaldo 2023) covers the same use case
     (bounded streams such as 0/1 error marks) with a non-asymptotic ARL ≥ 1/α and a
     closed-form, stream-vectorised computation.
  4. `detect.focus` already gives CUSUM at every window length for mean shifts.

  `river.ADWIN` appears only in `benchmarks/monitor/` as a detection-delay baseline at
  a matched false-alarm rate.

---

## 4. Module placement and public API

### 4.1 Decision: a new `panelary/monitor/`, not an extension of `quality/`

`quality` is the **data-integrity** gate. It is deterministic, pure Polars, uses no
RNG, "profiles without judging", and its checks are keyed to rows and columns. Drift
monitoring is **statistical inference**: seeded permutations, nulls, multiplicity,
ARL calibration and sequential guarantees. Putting it in `quality` would break that
package's contract (no RNG, no fitted state beyond schemas).

`validation` is the statistical layer, but it is organised around **model selection
and backtests** (CPCV, DSR, SPA/MCS). Monitoring is a separate stage of the life
cycle. So:

- A new subpackage **`panelary/monitor/`**, loaded lazily (added to
  `_LAZY_SUBMODULES` in `panelary/__init__.py`, costing 0 ms against the 300 ms budget
  in `tests/test_import_hygiene.py`).
- Findings serialise through `quality.CheckResult`, giving one evidence path for the
  engine.
- Multiplicity primitives stay in `validation` (e-BH goes next to BH/BY).
- Weighted conformal stays in `conformal`.

### 4.2 Layout and file ownership (touch ONLY your files)

| File | Contents | Owner | M |
|---|---|---|---|
| `monitor/_twosample.py` | one-sort engine, tie/NaN handling, exact-integer KS, p-values | Agent A | M1 |
| `monitor/_hist.py` | `BinEdges` (frozen artefact), polars/numpy binning, histogram cube, prefix sums | Agent B | M1 |
| `monitor/_divergence.py` | PSI, JS, χ²/G nulls, `psi_threshold`, detectable-effect gate | Agent B | M1 |
| `monitor/_tables.py` | CvM asymptotic CDF knots, Marsaglia `adinf` coefficients, AD k > 2 table | Agent A | M1 |
| `monitor/_report.py` | `drift_report`, `DriftReport`, fixed schema, `to_evidence` | Agent C | M1 |
| `monitor/_sequential.py` | CS, e-processes, mSPRT, e-detector | Agent D | M2 |
| `monitor/_null.py` | permutation plans (iid / date-block / entity-paired), persistence gate, max-T wiring | Agent E | M3 |
| `monitor/_rolling.py` | `rolling_drift` kernel, `xs_drift`, epoch schedule | Agent F | M3 |
| `monitor/_multivariate.py` | MMD (rff / quadratic / block), energy, classifier test | Agent G | M4 |
| `monitor/_shift.py` | label shift, `DensityRatio`, ESS, `shift_aware_conformal` | Agent H | M4 |
| `monitor/_spc.py`, `monitor/_arl.py` | Phase I/II charts; Markov/simulation ARL; guaranteed limits | Agent I | M5 |
| `monitor/_fast.py` | numba kernels behind `require("numba")`, each with a numpy twin | Agent J | M6 |
| `tests/test_monitor_*.py` | all tests (§8) | Agent K | all |
| `monitor/__init__.py`, registry specs, `namespaces/ts.py` stub, docs, CHANGELOG | wiring | orchestrator | — |

**Additive edits outside `monitor/`** — these are the only ones, each with a parity test:

1. `validation/_selection_stats.py`: `e_benjamini_hochberg(e_values, *, alpha)`.
2. `conformal/_intervals.py`: `weighted_conformal_quantile`, with `nexcp_quantile`
   delegating to it.
3. `detect/_monitors.py`: an optional `axis=` on `page_cusum`.
4. `namespaces/ts.py`: a `rolling_drift` method stub (the kernel is imported inside
   the method) and its `FeatureSpec`.
5. `panelary/__init__.py`: `"monitor"` added to `_LAZY_SUBMODULES`.
6. A new private file `_internal/_glm.py::logistic_irls` (L2 Newton/IRLS with
   step-halving), shared with any later propensity-score caller (§12).

### 4.3 Public API

```python
import panelary as pn
m = pn.monitor                                  # lazy (PEP 562)

# ---- one-shot, fixed-n drift (scheduled comparisons; NOT for continuous peeking) ----
edges = m.BinEdges.fit(reference, features=cols, bins=10, binning="quantile")  # frozen artefact
rep = m.drift_report(
    reference, current, features=cols,
    stats=("psi", "ks", "w1"),        # psi | js | chi2 | ks | cvm | ad | w1 | energy
    edges=edges,                       # default: BinEdges.fit(reference, ...)
    null="auto",                       # closed-form | permutation | date-block | entity-paired | auto
    entity=None, time=None, pairing=None,   # pairing="entity" when both windows share entities
    correction="benjamini_yekutieli",  # | benjamini_hochberg | holm | romano_wolf | None
    alpha=0.05, effect_floor={"psi": 0.10, "js": 0.01, "w1": None},
    n_resamples=999, seed=0,
)
rep.frame                     # fixed schema (§4.4); pl.concat across reports just works
rep.to_evidence(include_edges=False)    # list[dict] via quality.CheckResult

# array level: rows are features, batched, no per-feature Python loop
out = m.two_sample(ref, cur, stats=("ks", "cvm", "ad", "w1", "energy"), presorted_ref=False)
p = m.ks_pvalue(d_num, n, m_, method="auto")          # exact integer numerator D*n*m
thr = m.psi_threshold(n, m_, bins=10, alpha=0.01)      # sample-size-aware critical PSI

# ---- anytime-valid (continuous monitoring; peeking is free) ----
cs = m.confidence_sequence(x, bounds=(0.0, 1.0), method="eb", alpha=0.05)   # x: (T,) or (T, N)
cs.lower, cs.upper                                   # running-intersection, prefix-invariant
e = m.e_process_mean(x, null=0.10, alternative="greater", bounds=(0, 1))    # log-e, (T, N)
e2 = m.e_process_paired(x, y, bounds=(0, 1))          # H0: E[x - y] = 0
ab = m.msprt(x_a, x_b, tau2=..., sigma2="plugin")     # always-valid p and CI
det = m.e_detector(x, pre_mean=0.10, bounds=(0, 1), lambdas="grid", kind="cusum", alpha=1/1000)
res = pn.validation.e_benjamini_hochberg(e_values_at_stop, alpha=0.1)

# ---- multivariate / shift ----
m.mmd_test(X_ref, X_cur, method="rff", n_frequencies=64, seed=0)       # | "quadratic" | "block"
m.energy_test(X_ref, X_cur, n_resamples=199, seed=0)
m.classifier_test(frame_ref, frame_cur, features=cols, time="date",
                  folds=5, purge=5, model="ridge", n_resamples=199, seed=0)
w = m.label_shift_weights(val_probs, val_labels, target_probs, method="mlls", calibrate="bcts")
dr = m.DensityRatio(method="logistic", features="linear").fit(X_ref, X_cur, groups=dates)
pn.conformal.weighted_conformal_quantile(scores, dr.weights(X_cal), dr.weights(X_test), alpha=0.1)

# ---- SPC (PanelTransformer: fit = Phase I, transform = Phase II) ----
ch = m.EWMAChart(lam=0.1, arl0=370, limits="exact", guarantee=0.90)
ch.fit(phase1, value="err", entity="model", time="date")
out = ch.transform(phase2)        # statistic, lcl, ucl, alarm, rule, phase
m.ShewhartChart(rules="western_electric"); m.CUSUMChart(k=0.5, arl0=370)
m.HotellingT2Chart(cov="lw", arl0=200)   # cov -> panelary.covariance.estimate(method=...)
m.MEWMAChart(lam=0.1, arl0=200); m.MCUSUMChart(k=0.5, arl0=200)
m.arl(ch, L=2.814, method="markov"); m.calibrate_limit(ch, arl0=370, method="markov")

# ---- features ----
df.with_columns(pl.col("ret").ts.rolling_drift(
    stat="psi", window=21, reference=252, gap=0, bins=10,
    refit_every=21, reference_mode="epoch", stride=1).over("ticker"))
m.xs_drift(df, features=cols, time="date", reference=63, gap=1, stat="psi")  # date-level long frame
```

### 4.4 Result schemas

`DriftReport.frame` has a fixed schema. Irrelevant fields are null:

`feature, stat, estimate, p_value, p_value_adj, correction, threshold, effect_floor,
detectable_effect, verdict, null_method, sequential_valid, n_ref, n_cur, n_eff_ref,
n_eff_cur, n_bins, n_resamples, block_length, seed, reference_span, current_span,
approximate, warnings`

- `verdict` ∈ {`drift`, `no-drift`, `insufficient-n`, `refused-persistent`}.
- `sequential_valid` is always `False` for this report (trap T9).

`ChartResult` (from `transform`) has the schema:
`entity, time, stream, statistic, lcl, ucl, alarm, rule, phase, run_length`.

`CSResult` holds: `lower, upper, center, method, alpha, t_first_exclusion`.

---

## 5. Hard invariants (every function)

1. **Prefix invariance.** `f(x[:T])[t] == f(x[:T+k])[t]` bitwise for every rowwise
   output. No window, threshold, bin edge, bandwidth, epoch boundary or limit may
   depend on `len(x)`. Epoch and stride schedules are anchored at a **fixed origin**,
   never at the end of the sample. The schedule object is sibling 3's
   `panelary.covariance.Schedule`: an integer k means index ≡ 0 (mod k) anchored at
   index 0 (the `detect.residualise` convention), and a calendar string means the
   **first** trading date of each period. "Last date of the period" is refused,
   because knowing t is last requires t + 1. If sibling 3 has not landed, the same two
   rules are implemented privately and swapped out later.
2. **Frozen artefacts are fits.** Bin edges, category vocabularies, kernel bandwidths,
   RFF frequencies, Phase-I parameters, control limits, density-ratio models and
   calibration temperatures are all **fitted**. Each is estimated once, on declared
   rows, and serialisable (`to_dict`). None is silently recomputed on the current
   window.
3. **Label-invariant tuning for permutation tests.** Any quantity chosen before a
   permutation null — bandwidth, ridge penalty, normal-score transform — is a function
   of the **pooled** sample (label-invariant) or of data strictly before both windows.
   A quantity chosen from reference-only data, or tuned on the observed labels,
   invalidates the permutation p-value (trap T4).
4. **Determinism.** Every RNG takes `seed: int`. Per-stream generators come from
   `np.random.SeedSequence([seed, stable_key])`, where `stable_key` is a **content
   hash** of the entity/feature key (`blake2b`, 8 bytes). A position code is not stable
   when a new entity is appended. Shared permutation plans use `SeedSequence([seed, 0])`
   and are shared across features by design.
5. **float64 accumulation; integer counts.** Counts, cumulative counts and KS
   numerators are `int64`. The KS statistic is carried as the **exact integer**
   `max|m·cₐ − n·c_b|` so exact tables are indexed without float fuzz. Sums of products
   use `np.sum` (pairwise summation), not `einsum`, except for GEMMs, which use `@`.
6. **`np.linalg.solve(A, b[..., None])[..., 0]`** for batched solves. Triangular
   whitening uses a Cholesky factor computed once per Phase I. The condition number is
   reported, and the call refuses above 1e10 unless shrinkage is on.
7. **No `rolling_map`, no per-window Python UDFs.** The allowed patterns are listed in
   §6.9.
8. **Every result carries its null.** `null_method, n_resamples, block_length, n_eff,
   seed` are always present. A bare p-value is never returned.
9. **Persistence gate.** When `null="auto"`, a lag-1 rank-autocorrelation pre-check
   (`depend._ranks.lag1_rank_autocorr`, pooled over the window) runs:
   - |ρ| ≤ 0.2: closed form.
   - 0.2 < |ρ| < 0.95 and n_eff = n(1−ρ)/(1+ρ) ≥ 30: date-block permutation.
   - Otherwise: `p_value = NaN`, `verdict = "refused-persistent"`, plus a
     `SerialDependenceWarning` naming the fix (test the first difference or the
     cross-sectional rank instead).

   §1.2 shows why this is a refusal rather than a bigger block.
10. **One-shot vs sequential is explicit.** Fixed-n tests carry `sequential_valid=False`.
    Anything intended for continuous looking lives in `_sequential.py` and satisfies
    Ville's inequality.
11. **No sealed data.** No function persists raw reference values. `to_evidence`
    omits bin edges and quantiles unless `include_edges=True`, because edges are
    quantiles of the reference and can reveal the distribution of a sensitive column
    (STRATEGY §21).

---

## 6. Algorithms — candidates, choice, complexity, numerics

### 6.1 The one-sort univariate engine (KS, CvM, AD, W1, energy)

**Candidates.** Per-feature SciPy calls (5 sorts per feature, measured 8.3× slower).
Per-statistic ECDF evaluation on a grid (approximate). A single merge of the two
samples. **Choice: the single merge.**

**Algorithm** (batched over the F feature rows):

1. `Z = concat(ref, cur)` → `o = argsort(Z, axis=1, kind="stable")` gives O(F N log N).
   **Fast path when the reference is frozen:** sort the reference once, sort each
   current window (O(m log m)), then `argsort(kind="stable")` of the two sorted runs.
   Timsort detects the two runs and gallop-merges them in O(N). Measured: 24 ms vs
   217 ms for the sort step.
2. `isA = o < n`, then `cₐ = cumsum(isA)` and `c_b = k − cₐ` (int64). Tie runs use
   `last = z[k+1] ≠ z[k]`, and the ECDFs are evaluated only at the last element of
   each tie run.
3. **KS:** `D·n·m = max|m·cₐ − n·c_b|` over tie-run ends. This is an exact integer.
4. **CvM** (Anderson 1962): `T = (nm/N²) Σ (Fₐ − F_b)²` over pooled points.
5. **AD, k = 2** (Scholz & Stephens 1987): `A² = (1/N) Σᵢ (1/nᵢ) Σⱼ (N·Mᵢⱼ − j·nᵢ)² / (j(N − j))`.
   The numerators are exact int64. With ties, the midrank variant A²_akN is used, with
   parity against SciPy `midrank=True`.
6. **W1:** `Σ |Fₐ − F_b|·Δz`. **Cramér ℓ²:** `Σ (Fₐ − F_b)²·Δz`. **Energy distance
   (1-D):** `2·ℓ²` (Baringhaus & Franz 2004; Székely & Rizzo 2013), so it is free.
7. **NaN:** NaNs sort last. Per-row finite counts `n_f, m_f` bound the evaluation
   index, and no row is ever silently shortened.

**p-values:**
- **KS, n = m:** the exact Gnedenko–Korolyuk alternating sum
  `P(D ≥ k/n) = 2 Σⱼ (−1)^{j+1} C(2n, n−jk)/C(2n, n)`. The binomial ratio is taken
  from one log-prefix `L[s] = Σ_{i≤s} log1p(−(2i−1)/(n+i))`, computed once per n and
  indexed by `j·k` for all features at once.
- **KS, n ≠ m with n·m ≤ 10⁶:** the Hodges (1958) lattice-path DP, O(nm) per
  **distinct** D (deduplicated across features).
- **KS, otherwise:** Kolmogorov asymptotic with the Stephens (1970) effective n
  (`(√nₑ + 0.12 + 0.11/√nₑ)·D`). For λ < 1.18 it uses the Jacobi-theta dual series
  `(√(2π)/λ) Σ exp(−(2j−1)²π²/(8λ²))`, so p near 1 stays accurate.
- **CvM:** Anderson's exact null mean and variance, standardisation, then the
  asymptotic ω² CDF from a 512-knot log-tail table in `_tables.py` (monotone PCHIP).
  The table is generated offline by `benchmarks/monitor/gen_tables.py` with SciPy as
  the oracle.
- **AD:** standardise with the exact σ_N² (formula verified to 1e-16 against SciPy),
  map to A²∞ (mean 1, variance 2(π² − 9)/3), and evaluate with the Marsaglia &
  Marsaglia (2004) `adinf` closed form. For k > 2 (comparing several epochs), use the
  Scholz–Stephens interpolation, re-tabulated by seeded Monte Carlo.
- **W1 and energy:** shared permutation plan (§6.3.1). Optional Besag–Clifford (1991)
  sequential stopping skips features whose p-value is already decided.

**Cost.**
- O(F N log N) time, or O(F N) with a presorted reference.
- Memory ≈ 4 × F × N × 8 B at peak. Features are chunked to a 64 MB working set, and
  chunks are thread-parallel (§6.9).
- Measured: 312 ms for F = 500, n = m = 5000, all 5 statistics; 81 ms on the
  presorted path.

### 6.2 Frozen-edge histograms: PSI/CSI, Jensen–Shannon, χ²/G

**`BinEdges.fit(reference)`**:
- `binning="quantile"`: B−1 interior reference quantiles, deduplicated under ties.
  `n_bins_effective` is recorded.
- `"uniform"`: fixed width on [q₀.₀₀₁, q₀.₉₉₉] of the reference, with open tails.
- `"given"`: user-supplied edges.
- Categoricals: a frozen vocabulary plus an `__unseen__` bucket.

All bins are left-closed: `[eᵦ₋₁, eᵦ)`.

**Binning.** Polars `pl.col(f).cut(edges, left_closed=True)` codes (parallel across
columns, measured ≈ 500 M elements/s), then one `bincount` of the composite key
`(date·F + f)·B + bin` into an int32 **histogram cube** (T, F, B). The numpy fallback
is per-row `searchsorted` + `bincount` (51 M/s). A parity test must show the polars
codes equal numpy's on both CI polars legs (1.35 and 1.42). Polars reworked
Categorical in 1.32, so `to_physical()` ordering is verified, not assumed. If it is
not guaranteed, fall back to `search_sorted` on an edges literal.

**Statistics** (smoothed proportions p̂, q̂ with Jeffreys +0.5):
- `PSI = Σ (q̂ − p̂)·ln(q̂/p̂)`. Null: `PSI / (1/n + 1/m) → χ²_{B_eff−1}` (Yurdakul &
  Naranjo 2020; calibration in §1.1).
- `JS_π` with weights π = (n/N, m/N). **`G = 2N·JS_π` is exactly the G-test of
  homogeneity**, so it has a χ²_{B−1} null. √JS is reported as the metric.
- Plain Pearson χ² on the same 2 × B table.

**Sample-size-aware verdict:**
- `detectable_effect = (1/n + 1/m)·χ²_{B−1}(1 − α_adj)` is the smallest PSI that can
  reach significance at this n.
- `drift` iff `p_adj ≤ α` **and** `estimate ≥ effect_floor`.
- `insufficient-n` iff `detectable_effect > effect_floor` and not `drift`.
- Otherwise `no-drift`.

This fixes both failure modes in §1.1: noise at small n, and trivial drift at large n.

**Cost.** Binning is O(F N log B). The statistics are O(F B). Every trailing reference
is a difference of date prefix sums of the cube, O(F B) per date (measured 0.9 ms for
135 × 50). Frozen edges make the cube exactly mergeable (streaming, sharded, or
re-run on appended dates without touching old ones).

### 6.3 Nulls, multiplicity, and the multivariate kernel tests

#### 6.3.1 Permutation plans (M3; the i.i.d. plan ships in M1)

- **i.i.d.:** a (B, N) index plan from `SeedSequence([seed, 0])`.
- **date-block:** `depend._null.block_permutation_indices` over the ordered union of
  window **dates**, whole cross-sections moved together. The block length comes from
  `pair_block_length`, frozen per report.
- **entity-paired** (`pairing="entity"`, when both windows hold the same entities):
  each entity's (ref, cur) values swap with probability ½. This is exact under
  within-pair exchangeability. Independent-sample tests are **conservative and
  low-power** on paired panels, and this is the correct null for "same universe, two
  dates".

**Evaluation.** Sort each feature once; each permutation is a **gather of labels**
through the fixed order, then cumsum (19 ms per permutation for 500 features at
N = 10⁴).

**Multiplicity.**
- Closed-form p-values → BY (the panel default, valid under arbitrary dependence) or BH.
- Permutation p-values → **Romano–Wolf max-T** (`validation.romano_wolf`) over the
  shared (B × F) null matrix. It needs B ≈ 999 regardless of F. BH over permutation
  p-values needs `1/(B+1) ≤ α/F`, i.e. B ≥ 10⁴ at F = 500. The report checks this and
  raises instead of returning an unrejectable screen.

#### 6.3.2 MMD — candidates and choice

| Estimator | Cost | Null | Verdict |
|---|---|---|---|
| Quadratic unbiased MMD²ᵤ (Gretton et al. 2012) | O(N²d) + O(N²B) | permutation | ships (`"quadratic"`, N ≤ 2·10⁴) |
| Linear-time MMD (Gretton et al. 2012 §6) | O(N) | normal | not shipped: variance ∝ 1/N, dominated by the block test |
| B-test (Zaremba, Gretton & Blaschko 2013) | O(N b) | normal, closed form | ships (`"block"`, streaming) |
| Random-feature / characteristic-function Hotelling (Chwialkowski et al. 2015) | O(N D (d + D)) | χ²_D, or F_{D, N−D−1} | **default** (`"rff"`) |
| ME with optimised locations (Jitkrittum et al. 2016) | O(N J d) + optimisation | χ²_J on a held-out split | not shipped: needs a split and an optimiser |
| MMDAgg / MMD-FUSE (Schrab et al. 2023; Biggs et al. 2023†) | O(N² K B) | permutation | not shipped (§3.2) |

**Default `"rff"`:**
- Features: `RFFMap` (reused) with **pooled** normal scores and bandwidth 1, or a
  pooled median heuristic. Both are label-invariant (invariant 3). Orthogonal
  frequencies come from `embed._rff.orthogonal_gaussian` (Yu et al. 2016).
- Test: Hotelling two-sample T² on the D features, `δ = φ̄_ref − φ̄_cur` and
  `S = pooled cov + εI`, then `F = (N−D−1)/(D(N−2))·T² ~ F_{D, N−D−1}` via
  `econ._common.f_sf`.
- D = min(128, N/10). Estimated cost at N = 10⁴, d = 50, D = 128: ≈ 20 ms (to be
  measured in M4).

**`"quadratic"` with batched permutations.** The biased and unbiased statistics
differ by a permutation-invariant affine map for fixed group sizes, so the p-value is
unchanged. Build a weight matrix `W` (N × (1+B)) with entries 1/n and −1/m, and
compute all statistics as `colsum(W ⊙ K W)`, accumulated over 2000² kernel tiles
(`K_ij @ W_j`, one GEMM per tile). Cost is O(N²(d + B)) flops: an estimated ≈ 0.3 s at
N = 10⁴, B = 200. Memory is one 32 MB tile plus W.

**Energy (multivariate).** The same tiled engine with `_pairwise_dist_tile`, the
distance kernel. Inputs are centred on the pooled mean before the Gram trick, to limit
cancellation.

### 6.4 Classifier two-sample test / adversarial validation

**Discriminator.**
- Default: **ridge-LDA in closed form** on pooled-standardised features. With `"rff"`
  features it is a kernel classifier.
- Opt-in: sklearn `HistGradientBoostingClassifier` via
  `require("sklearn", feature="classifier_test")`, with B ≤ 99 permutations.

**Folds.** Date-block folds built by applying `PurgedKFold` separately to the
reference dates and the current dates and zipping fold k. Folds are purged/embargoed
by `purge` dates. A random K-fold on temporally dependent rows inflates accuracy under
H0 (trap T3).

**Batched permutation null.**
- Per fold: one Cholesky of `X_trᵀX_tr + λI`. The label matrix `Y` (n_tr × (1+B))
  holds the observed ±1 labels plus B **date-block-permuted** label vectors. Then
  `β = solve(G, X_trᵀY)` is a single multi-RHS solve.
- Statistic: out-of-fold **AUC** (Mann–Whitney via `depend._ranks.ranks`), which is
  robust to class imbalance.
- λ = 10⁻³·tr(XᵀX)/p is label-free. Tuning λ on the observed labels and then permuting
  only the final fit is invalid (trap T4).
- Estimated cost: N = 10⁴, p = 500, B = 199, 5 folds ≈ 0.5 s.
- Theory: Lopez-Paz & Oquab (2017); Kim, Ramdas, Singh & Wasserman (2021). With λ → 0
  the test is a cross-fitted Hotelling T² two-sample test, and the docstring says so.

**Output.** AUC, p-value, and the out-of-fold probabilities. The same fit feeds
`DensityRatio(method="logistic")` (§6.5), so detection and correction share one model.

### 6.5 Shift correction

**Label shift.** Candidates:
- BBSE (Lipton et al. 2018): solve `Ĉw = μ̂_T`, O(K³).
- RLLS (Azizzadenesheli et al. 2019): regularised `min ‖Ĉθ − b̂‖₂ + Δ‖θ‖₂`, solved via
  an SVD of Ĉ and a 1-D search on the multiplier.
- MLLS/EM (Saerens et al. 2002) with bias-corrected temperature scaling (BCTS).
  Alexandari, Kundaje & Shrikumar (2020) and Garg et al. (2020) show that calibrated
  MLLS dominates BBSE/RLLS.

**Choice:**
- `method="mlls", calibrate="bcts"` when probabilities are available. BCTS is a
  (T, b) Newton fit on **held-out source** rows.
- `"bbse"` for hard predictions, with `cond(Ĉ)` reported and a warning above 1e3.
- `"rlls"` for small target n.

Cost is O(iter·N·K). The weights are clipped at ≥ 0, and the implied target prior is
reported.

**Covariate shift density ratio `w(x) = p_cur(x)/p_ref(x)`.**
- **Default `"logistic"`** (Qin 1998; Bickel, Brückner & Scheffer 2009):
  `w = (n_ref/n_cur)·exp(logit p(cur | x))`, from an L2 logistic fitted by Newton/IRLS
  with step-halving, O(N p²) per iteration. It is cross-fitted on the same date-block
  folds as §6.4.
- **`"rulsif"`** (Yamada et al. 2013; uLSIF when α = 0, Kanamori et al. 2009):
  `θ = (Ĥ + λI)⁻¹ĥ` over b ≤ 200 Gaussian centres. Analytic LOO for all λ comes from
  one eigendecomposition of Ĥ, O(N b² + b³). It also gives a Pearson-divergence drift
  magnitude (Liu et al. 2013).

**Diagnostics.** Kish ESS = (Σw)²/Σw², maximum weight share, and optional clipping at
a weight quantile. These are recorded, never silent.

**Weighted conformal** (`conformal.weighted_conformal_quantile`, Tibshirani et al.
2019). Sort the calibration scores once and take `cumW`. For test weights w_t, the
thresholds are `searchsorted(cumW, (1−α)(W + w_t))`, vectorised over all m test
points. Cost: O((n + m) log n). `nexcp_quantile` becomes the special case with
geometric weights and w_t = 1.

The three-way split — model / ratio-fit / calibration — comes from
`validation.purged_calibration_split`. With estimated weights, the coverage gap is
bounded by the ratio error (Lei & Candès 2021), and that bound is documented.

### 6.6 SPC with Phase I / Phase II and ARL calibration

**Phase I (`fit`)** estimates, per stream:
- μ₀;
- σ₀, by moving range `MR̄/1.128` by default (robust to slow Phase-I drift; the pooled
  SD is optional);
- Σ₀ for multivariate charts, from sibling 3's
  `panelary.covariance.estimate(X_phase1, method="sample"|"lw"|"oas"|"qis")`. The
  charts use its `CovEstimate.solve` / `inv_quad`, and no covariance code is written
  here;
- the Phase-I end date.

For p ≤ 50, one dense Cholesky factor is materialised from the `CovEstimate` so that
whitening is stream-vectorised. `transform` **refuses** rows at or before
`phase1_end`.

**Charts** (all vectorised across streams):
- **Shewhart individuals** with Western Electric rules 1–4 and Nelson rules 1–8.
  Indicator series go through **rolling sums as cumsum differences**, and run lengths
  use the reset trick `(c != c.shift()).cum_sum()` → `cum_count().over(...)`. This is
  native polars, `.over(entity)`.
- **EWMA** (Roberts 1959): `pl.col(x).ewm_mean(alpha=λ, adjust=False)` with the Z₀ = μ₀
  correction `Z_t = y_t − (1−λ)^{t+1}(x₀ − μ₀)`. Exact time-varying limits use
  `−expm1(2t·log1p(−λ))` for accuracy at small λt.
- **CUSUM**: `detect.page_cusum(z, drift=k, side=…)` on Phase-I-standardised data.
  This is a reuse, not a reimplementation.
- **Hotelling T²** (individuals, Tracy, Young & Mason 1992). Phase I:
  `((m−1)²/m)·Beta(p/2, (m−p−1)/2)`. Phase II:
  `UCL = p(m+1)(m−1)/(m(m−p))·F_{p, m−p}(1−α)`. Quantiles come from bisection on
  `econ._common._betainc` / `f_sf`. **With a shrunk Σ the F law no longer holds**, so
  limits are simulated.
- **MEWMA** (Lowry et al. 1992) and **MCUSUM** (Crosier 1988). Whiten once by the
  Phase-I Cholesky factor. The recursions are then elementwise:
  - MEWMA: `Z̃_t = λx̃_t + (1−λ)Z̃_{t−1}` and `T²_t = ‖Z̃_t‖²/c_t`, with
    `c_t = λ/(2−λ)·(1−(1−λ)^{2t})`.
  - MCUSUM: `C_t = ‖S_{t−1} + x̃_t‖`, and `S_t = 0` if `C_t ≤ k`, else
    `(S_{t−1} + x̃_t)(1 − k/C_t)`.

  Each is a time loop over T with (N, p) vector operations. Estimated: T = N = 5000,
  p = 10 in ≈ 1 s numpy, and faster with numba.

**ARL engines:**
- **Markov chain** (Brook & Evans 1972; Lucas & Saccucci 1990) for EWMA and CUSUM:
  m = 201–301 states, solve `(I − Q)·ARL = 1`, Richardson extrapolation over (m, 2m).
  Measured: 499.4 vs the published 500, 8 ms per limit. The MEWMA in-control chain is
  1-D in ‖Z‖ (Runger & Prabhu 1996). The non-central χ² transitions use a Poisson
  mixture of `chi2_sf`, which is an M5 stretch goal; simulation is the default for
  MEWMA.
- **One-pass simulation for every limit at once.** The chart statistic does not
  depend on L until alarm, so `ARL(L) = Σ_t P(max_{s≤t}|stat_s| ≤ L)`. Track each
  replication's running max. At each step, add `searchsorted(sort(M), grid)` to a
  survival counter, then truncate at H ≈ 10·ARL₀ with a geometric tail correction.
  Measured: 281 limits from one 5.1 s run. `calibrate_limit` interpolates the
  monotone curve. This covers runs rules, MCUSUM, shrunk T² and non-Gaussian
  (bootstrap-innovation) charts.
- **Guaranteed conditional performance** (Gandy & Kvaløy 2013). Estimated Phase-I
  parameters make the *conditional* in-control ARL highly variable (Jones, Champ &
  Rigdon 2001; Jensen et al. 2006). `guarantee=0.90` picks L so that
  P(conditional ARL₀ ≥ target) ≥ 0.90. For Gaussian Phase I the estimation error is
  pivotal: a ~ N(0, 1/m) and b ~ √(χ²_{m−1}/(m−1)). One batched Markov solve over
  (B_boot, m, m) per bisection step therefore gives a single adjusted L, **shared by
  every stream with the same Phase-I length** (estimated ≈ 1.5 s total). A
  nonparametric option uses `validation.block_bootstrap_indices`.
- **Alarm budget across streams.** With S streams, per-stream ARL₀ = S / budget,
  where budget is the expected number of false alarms per date. A per-stream ARL₀ of
  370 with S = 2.5·10⁶ streams gives ~6 700 false alarms a day. The docs must say this
  in the first paragraph.

### 6.7 Anytime-valid inference

**Bounded-mean CS — default `method="eb"`** (predictable plug-in empirical Bernstein,
Waudby-Smith & Ramdas 2024, Thm 2):
- `λ_t = min(√(2 log(2/α) / (σ̂²_{t−1}·t·log(1+t))), c)`, with
  `μ̂_t = (½ + Σx)/(t+1)` and `σ̂²_t = (¼ + Σ(x − μ̂)²)/(t+1)`.
- CS: `(Σλx)/(Σλ) ± (log(2/α) + Σ 4(x − μ̂_{i−1})²·ψ_E(λ)) / Σλ`, with
  `ψ_E(λ) = (−log(1−λ) − λ)/4`. The running intersection is `maximum.accumulate` /
  `minimum.accumulate`.

Everything is a cumsum over (T, N). Measured: 2000 × 10⁴ in 559 ms, ever-miss 1.9%.

**Betting CS — opt-in `method="betting"`** (hedged capital, WSR 2024 §4):
`log K_t(m) = logaddexp(Σ log1p(λ⁺(m)(x−m)), Σ log1p(−λ⁻(m)(x−m))) − log 2` on a grid
of m. It is tighter mainly at small t (WSR 2024, figures). At t = 10⁴ it measured no
tighter than EB and cost 239 ms per stream.

Fast approximation: **alive-set pruning**. Under the running intersection a grid point
that leaves the CS never returns. Only live points need updating, and the live set
shrinks like t^{−1/2}. Target (to be verified): ≥ 20× faster at G = 10⁴. There is a
numba kernel under `fast`.

**Unbounded metrics:**
- `method="asymptotic"` (Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas 2024):
  `μ̂_t ± σ̂_t·√(2(tρ²+1)/(t²ρ²)·log(√(tρ²+1)/α))`, with ρ tuned to a user
  `t_opt`. Valid for i.i.d. data with finite 2+δ moments.
- `method="normal-mixture"` (Robbins 1970; Howard et al. 2021) for sub-Gaussian data
  with known σ.

**e-processes:**
- `e_process_mean`: H₀: μ ≤ m₀. The betting capital `K_t(m₀) = Π(1 + λ_i(x_i − m₀))`
  with predictable λ (PrPl or aGRAPA), in log domain. The anytime p-value is
  `1/max_{s≤t} K_s`, which is valid by Ville.
- Proportions are the {0,1} case.
- `e_process_paired`: the same machinery on `(x − y + 1)/2 ∈ [0,1]` with m₀ = ½.

**mSPRT** (Johari et al. 2022). Normal mixture:
`Λ_n = √(σ²/(σ²+nτ²))·exp(n²τ²(x̄ − θ₀)²/(2σ²(σ²+nτ²)))`, in log domain. The
always-valid p is `p_n = min(p_{n−1}, 1/Λ_n)` (a running min, so prefix-invariant). The
CI comes by inversion. σ² defaults to a plug-in (asymptotically valid), and the
fixed-σ variant is exact. τ² is chosen from the minimum effect size of interest, and
the docs show its effect.

**e-detector** (Shin, Ramdas & Rinaldo 2023), replacing ADWIN:
- Default `kind="cusum"`: `log M_t = logsumexp_k(log w_k + page_cusum(log e^{(k)})_t)`
  over a grid of constant bets λ_k. Each `page_cusum` is the closed-form reflection, so
  there is **no time loop**, using the additive `axis=` on `detect.page_cusum`.
- Alarm at `log M_t ≥ log(1/α)`, which guarantees ARL ≥ 1/α.
- `kind="sr"` uses `detect.shiryaev_roberts` with `llr = log e_t`. It is 1-D only and
  documented as the reference implementation. A stream-batched SR twin in `detect` is
  a possible follow-up owned by `detect`.

**e-BH** (Wang & Ramdas 2022) is `_bh_family(min(1, 1/e))`, valid under arbitrary
dependence with no BY harmonic penalty. The input **must** be e-process values at a
stopping time, **never** `max_{s≤t} K_s` (trap T8).

### 6.8 Rolling drift features (prefix-invariant)

**Per-entity `.ts.rolling_drift`** (`over(entity)`):
- The current window is `[t−w+1, t]`.
- `reference_mode="epoch"` (default): edges **and** the reference histogram are fitted
  on `[s_e − r, s_e − 1]` at the start `s_e` of epoch e. Epochs follow the
  `Schedule` of invariant 1: every `refit_every` rows (index ≡ 0 mod k), or the first
  date of each calendar period when a duration is given with `by=`. The current-window
  counts come from one-hot prefix sums under epoch-e binning: O(T·B·(1 + w/E)) per
  series.
- `reference_mode="sliding"` (opt-in): reference `[t−w−g−r+1, t−w−g]`, costing
  O(T·B·(1 + (w+g+r)/E)).
- `stat ∈ {psi, js, chi2, ks_binned, w1_binned}`. The binned KS/W1 are exact on
  B = 64 reference-quantile bins, with error ≤ the maximum bin mass, and are O(B) per
  evaluation.
- Exact `ks`/`w1` use `sliding_window_view` + a batched sort with `stride` evaluation
  and as-of forward fill. The stride schedule is anchored like the epochs, so it stays
  prefix-invariant.

**`xs_drift` (date-level):**
- Current = the cross-section at date t (same-date data only). Reference = pooled
  cross-sections of dates `[t−g−r+1, t−g]`, counted on the panel's own date axis.
  Appending later dates never re-indexes earlier ones.
- Implementation: the histogram cube (§6.2) streamed over date chunks with
  `pl.scan_*`, plus date prefix sums.
- Output: `(time, feature, stat, estimate, p_value)`, joined back on `time` by the
  user. This is a frame-level analysis function, not a registered expression, because
  an `.over(time)` expression cannot see trailing dates.

**Registry.** `FeatureSpec(name="rolling_drift", namespace="ts", input_shape="series",
output_shape="series", tier="B", panel_safe=True, leakage_safe=True,
safe_scope="rowwise", axis="time", flavour="trailing", streaming="mergeable",
cost_hint="O(T B)", source="Panelary", license="Apache-2.0")`. Default parameters must
produce non-degenerate output on the conformance probe panel of
`tests/test_registry_conformance.py`, or an exercise entry is added there.

### 6.9 Vectorisation and parallelism patterns (which one, where)

| Pattern | Used by |
|---|---|
| (a) time loop, stream-vectorised recursion on padded (T, N) | MEWMA, MCUSUM, betting CS grid, one-pass ARL simulation |
| (b) prefix sums / cumulative counts → O(1) per trailing window | PSI/JS cube, runs rules, EB-CS, mSPRT, e-process, CUSUM reflection, rolling_drift |
| (c) `sliding_window_view` + batched sort / BLAS | exact rolling KS/W1 (strided), tiled MMD/energy GEMMs, C2ST multi-RHS |
| (d) native polars `.over()` | `cut` binning, EWMA `ewm_mean`, Shewhart rules, e-detector expressions |
| (e) numba `@njit(cache=True, parallel=True)` behind `fast`, with a numpy twin and a ≤ 1e-12 / identical-alarm parity test | per-row linear merge (presorted path), MCUSUM, pruned betting CS, two-bin O(1) rolling-PSI update |
| (f) stride evaluation + as-of forward fill (anchored schedule) | exact rolling KS/AD/W1, rolling MMD |
| (g) `ThreadPoolExecutor` over feature chunks (numpy sort, cumsum and GEMM release the GIL; results placed by index) | the one-sort engine, permutation nulls, binning fallback. Measured 3.7× on 8 threads, bitwise identical |

---

## 7. Leak-safety design and traps

| # | Trap | Symptom | Safe default | Test |
|---|---|---|---|---|
| T1 | Bin edges fitted on pooled ref+cur, on the current window, or on the full series in a rolling feature | the reference absorbs the drift; the rolling value at t moves when data is appended | `BinEdges.fit(reference)`; rolling edges per anchored epoch from pre-epoch data | leaky twin caught by `assert_prefix_invariant` |
| T2 | Bandwidth, RFF frequencies or normal scores taken from the full series | rolling MMD not prefix-invariant | pooled-window or pre-window quantities only | prefix test on rolling MMD |
| T3 | C2ST / density ratio with random K-fold on dependent rows | AUC > 0.5 under H0 → false drift | date-block folds, purged | AR panel with no drift: random K-fold rejects far above 5%, date-block ∈ [3%, 7%] |
| T4 | Tuning the discriminator (λ, depth) or the bandwidth on the observed labels, then permuting only the final fit | anti-conservative permutation p | label-free λ; pooled bandwidth | size test with a deliberately tuned twin |
| T5 | Phase-I parameters estimated on data that includes Phase II ("fit on everything, then monitor") | limits leak; alarms suppressed | `transform` refuses `time ≤ phase1_end`; `check_fitted_state` | quality check + perturbation of Phase II leaves Phase I state unchanged |
| T6 | Performance drift on **unmatured labels**: the error at t uses outcomes observed at t + h | lookahead in every performance monitor | `label_horizon=h`: date t sees only predictions made ≤ t − h | `assert_no_lookahead` on the performance-drift feature |
| T7 | Refit/epoch schedule from `expanding_window_split` / `sliding_window_split` (end-anchored), or a "last date of the month" anchor | every boundary moves on append; the last-date anchor needs t + 1 | `Schedule` (index ≡ 0 mod k, or the first date of each period) | leaky twins must fail `assert_prefix_invariant` |
| T8 | e-BH fed `max_{s≤t} K_s`, or an anytime p fed into BH as an e-value | FDR above α | API accepts e-process values at a stop time only | FDR simulation with a deliberately wrong twin |
| T9 | Re-running a fixed-n test (KS, PSI, Wilson) every period and acting on the first rejection | error grows toward 1 (§1.3: 53.5%) | `sequential_valid=False` in the schema; `_sequential` for continuous monitoring | peeking test asserts the bug is real |
| T10 | Adversarial-validation **feature pruning** using the whole test fold's covariates | lookahead in walk-forward: future X selects features | not offered. An optional `DriftScreen(PanelTransformer)` uses the early vs late halves of the **training fold** only | `assert_no_train_test_leak` |
| T11 | Density-ratio model fitted on the calibration rows used by weighted conformal | coverage loss | three-way purged split | coverage test |
| T12 | Label-shift confusion matrix or BCTS on the classifier's training rows | overconfident Ĉ → wrong weights | held-out source rows only | recovery test under known priors |
| T13 | Persistent feature tested i.i.d. | §1.2: 73–95% false drift | persistence gate (invariant 9) | calibration test pins the refusal |
| T14 | Serialising reference quantiles of a sensitive/sealed column | leaks its distribution | `include_edges=False` | evidence snapshot contains no edges |

---

## 8. Tests — `tests/test_monitor_*.py`

`--strict-markers` is on. Reuse `slow` and `benchmark`, and add no marker without
registering it in `pyproject.toml` first.

**`test_monitor_calibration.py` (slow). The most important file.**
- Size under H0 for every test, from seeded Monte Carlo (R ≥ 4000, so
  SE = 0.0034; acceptance [0.040, 0.060] at nominal 5%):
  - n ∈ {50, 250, 1000}, equal and unequal.
  - Exact nulls must also pass at n = 50.
  - Asymptotic nulls may be conservative below n = 250: assert ≤ 0.06 only.
  - Covers PSI-χ², JS-G, χ², KS (exact and asymptotic), CvM, AD, W1/energy
    (permutation), MMD rff/quadratic/block, C2ST.
- **Pin §1.1** as assertions: P(PSI > 0.1 | n = 100) > 0.7, and the χ² rule within
  [0.035, 0.065] at n = 1000.
- **Pin §1.2:** i.i.d. KS at φ = 0.9 has size > 0.5. **This test proves the bug is
  real and must never be "fixed" by loosening it.** Date-block null at φ = 0.5 is in
  [0.03, 0.08]. At φ = 0.98, `verdict == "refused-persistent"`.
- Paired panel: `pairing="entity"` in [0.03, 0.08], and its power beats the unpaired
  test on a common-factor panel.
- Power sanity: location, scale and tail shifts are detected at n = 1000 with power
  > 0.8.

**`test_monitor_sequential.py` (slow). Anytime validity.**
- R = 2000 paths, T = 10⁴. Adversarial optional stopping (stop at the first exclusion
  or rejection, or at a random data-dependent time). Assert
  P(ever wrong) ≤ α + 3·SE (0.0646) for EB, betting, asymptotic (with a Gaussian check
  only), normal-mixture CS, the e-processes, mSPRT and paired two-sample.
- Assert peeked Wilson > 0.3 (the bug is real).
- e-BH: FDR ≤ α + 3·SE with strongly correlated e-processes stopped at a common
  stopping time. The wrong twin (T8) must exceed α.
- e-detector: empirical ARL₀ ≥ 1/α on bounded i.i.d. streams, plus finite detection
  delay after a mean shift.

**`test_monitor_oracle.py`** — behind `pytest.importorskip`, with tolerances per
statistic:

| Ours | Oracle | Tolerance |
|---|---|---|
| KS stat / p | `scipy.stats.ks_2samp` (exact, asymp) | 1e-12 / 1e-9 exact, 1e-4 asymptotic |
| CvM | `scipy.stats.cramervonmises_2samp` | 1e-12 / 1e-6 |
| AD | `scipy.stats.anderson_ksamp` (unstandardise via σ_N) | 1e-12 stat |
| W1, energy | `scipy.stats.wasserstein_distance`, `energy_distance` | 1e-12 |
| PSI χ² quantiles | `scipy.stats.chi2.ppf` | 1e-10 |
| MMD, energy (multivariate) | `hyppo` (MIT†) | 1e-10 stat |
| Page–Hinkley recipe, ADWIN baseline | `river` (BSD-3) | identical alarms |
| CS bounds | `confseq` (Howard†) | 1e-8 |
| weighted conformal | brute-force per-test-point quantile | exact |
| Hotelling F/Beta quantiles | `scipy.stats.f/beta.ppf` | 1e-8 |

`alibi-detect` is **reference only** unless its licence is verified permissive (†;
believed to be BSL since 2024).

**`test_monitor_spc.py`:**
- Markov ARL₀ for EWMA λ = 0.1, L = 2.814 is within 1% of 500. Shewhart 3σ gives
  370.4 ± 0.5%. WE rules 1–4 give ARL₀ ≈ 91.75† (Champ & Woodall 1987), checked by
  simulation within its MC error.
- Simulation matches the chain within 3 MC SE.
- T² Phase-I Beta and Phase-II F coverage by Monte Carlo.
- Guaranteed performance: over 500 Phase-I resamples, P(conditional ARL₀ ≥ target)
  ≥ 0.9 − 3·SE.

**`test_monitor_shift.py`:**
- MLLS/BBSE/RLLS recover known class priors (error ↓ as n ↑; MLLS ≤ BBSE with
  calibrated probabilities).
- Logistic and RuLSIF ratios match the closed-form Gaussian-shift ratio (correlation
  > 0.95).
- Weighted conformal coverage under covariate shift is ≥ 1 − α − 0.02, while
  unweighted coverage fails.
- `nexcp_quantile` is bitwise unchanged.

**`test_monitor_prefix.py`:**
- `assert_prefix_invariant` **and** `assert_no_lookahead` (neither subsumes the other)
  for: every `rolling_drift` stat and mode, `xs_drift`, every chart's `transform`
  output, CS/e-process/mSPRT/e-detector sequences, strided + forward-filled
  variants.
- Leaky twins for T1, T2 and T7 must be caught.

**`test_monitor_leakage.py`:** T3, T4, T5, T6, T10, T11 and T12 as executable
regressions, modelled on `tests/test_depend_leakage.py`. Phase-I state is checked with
`quality.check_fitted_state`.

**`test_monitor_perf.py` (`benchmark`):** assert no more than a 2× regression against
§9.2.

**Packaging guardrails:**
- `tests/test_import_hygiene.py`: `monitor` is lazy, so the cold import does not move.
- `tests/test_dependency_drift.py`: **zero** new mandatory dependencies.
- `tests/test_wheel_guardrails.py`: `py3-none-any`.
- `tests/test_registry_conformance.py`: `ts.rolling_drift` passes.
- polars 1.35 and 1.42 CI legs: `cut` codes, `ewm_mean(adjust=False, min_samples=)`
  naming, `search_sorted` with an expression argument.

---

## 9. Benchmarks and performance budgets

### 9.1 Realistic sizes

The reference size is **F = 500 features × N = 5000 entities × T = 5000 dates**
(1.25·10¹⁰ cells, 100 GB at float64). It never fits in memory, so every panel path
**streams over date chunks** (`pl.scan_parquet(...)` → chunks of dates).
- Resident state for the PSI cube: (T, F, B) int32 = **100 MB** at B = 10.
- The one-sort engine is chunked to a ≤ 64 MB working set per thread.
- Peak RSS target: **≤ 2 GB**.

### 9.2 Budgets (measured → asserted at ≤ 2×; estimates marked)

| Operation | Size | Measured / target |
|---|---|---|
| `two_sample`, 5 stats | F = 500, n = m = 5000 | measured 312 ms → budget 0.65 s; 8 threads measured 78 ms (KS) |
| presorted-reference path | same | measured 81 ms → budget 0.2 s |
| shared-permutation null, KS | B = 200, F = 500, N = 10⁴ | measured 3.8 s → budget 8 s |
| PSI cube binning | 5·10⁷ cells | measured ≈ 0.19 s (polars) → budget 0.5 s |
| `xs_drift` PSI, full panel | 1.25·10¹⁰ cells | **estimate** ≈ 50 s at 260 M/s → target ≤ 90 s, ≤ 2 GB |
| per-entity `rolling_drift` PSI, full panel | 2.5·10⁶ series × 5000 | **estimate** numpy (b) ≈ 150 s; numba O(1) two-bin update ≈ 20–30 s → target ≤ 60 s with `fast` |
| exact rolling KS | 200 entities × 50 features × 5000 dates, w + r = 315, stride 5 | **estimate** ≈ 50 s → target ≤ 60 s. At full-panel scale, exact rolling KS is **not** a target — use `ks_binned` |
| EB confidence sequence | 2000 streams × 10⁴ | measured 559 ms → budget 1.2 s |
| betting CS | 1 stream, T = 10⁴, G = 10⁴ | **target** ≤ 20 ms with alive-set pruning (unpruned G = 10³: measured 239 ms) |
| EWMA ARL, Markov (m = 301) | per limit | measured 8 ms → budget 20 ms |
| one-pass ARL curve | R = 2·10⁴, 281 limits | measured 5.1 s → budget 10 s |
| MMD rff | N = 10⁴, d = 50, D = 128 | **estimate** ≈ 20 ms → budget 100 ms |
| MMD quadratic + 200 permutations | N = 10⁴, d = 50 | **estimate** ≈ 0.3 s (GEMM-bound) → budget 2 s |
| C2ST ridge, 5 folds, B = 199 | N = 10⁴, p = 500 | **estimate** ≈ 0.5 s → budget 2 s |
| MEWMA/MCUSUM Phase II | N = 5000 streams, p = 10, T = 5000 | **estimate** ≈ 1 s numpy → budget 3 s |

### 9.3 Numerical-accuracy budget

- Statistics against SciPy: ≤ 1e-12.
- Exact KS p against SciPy exact: ≤ 1e-9.
- Asymptotic p-values: ≤ 1e-4 absolute in [1e-6, 1].
- Markov ARL against published tables: ≤ 1% (Richardson ≤ 0.2% target).
- EB-CS closed form against a direct loop: bitwise.
- numba kernels against the numpy twins: ≤ 1e-12 with identical alarms.

### 9.4 Benchmark design

`benchmarks/monitor/` contains:
- the eight scripts behind §1 (PSI MC, KS under AR(1), engine vs SciPy, presorted
  merge, threads, binning, ARL, CS);
- `gen_tables.py` (offline, needs SciPy);
- `bench_scale.py`, which streams a synthetic F × N × T panel from
  `panelary.synth` (planted drifts at known dates) and reports wall time, peak RSS
  (`resource.getrusage`), detection delay and false-alarm rate;
- `bench_adwin_vs_edetector.py` (river baseline).

Each script prints the machine, versions and seed, and the results go into
`docs/user-guide/monitoring.md` as measured tables.

---

## 10. Dependencies

- **Mandatory:** none added (numpy + polars).
- **`fast` (numba, existing extra):** the kernels listed in §6.9(e). Each has a numpy
  twin, and the import goes through `require("numba", feature=...)`.
- **`ml` (scikit-learn, existing extra):** the optional HistGradientBoosting
  discriminator only.
- **`scipy`:** none at runtime. It is an oracle, and it is used offline to generate
  `_tables.py`.
- **Test-only oracles** (`importorskip`): scipy, hyppo, river, confseq†.
- **Clean-room:** every method is implemented from its paper. `source` and `license`
  on the `FeatureSpec` are enforced by `registry.audit()`.

---

## 11. Milestones

The order follows the caller (§14): comparability checks first, then continuous
monitoring. **Do not start M3 before `test_monitor_calibration.py` is green for M1's
statistics.**

| M | Contents | Ships |
|---|---|---|
| **M1** | `_twosample`, `_hist`, `_divergence`, `_tables`, `_report` (`drift_report`, `BinEdges`, `DriftReport.to_evidence`), i.i.d. closed-form nulls, BY/BH/Holm wiring, persistence pre-check (warn only) | Batched KS/CvM/AD/W1/energy/PSI/JS/χ² with sample-size-aware verdicts and evidence records. Zero new dependencies. |
| **M2** | `_sequential` (EB, asymptotic and normal-mixture CS; e-processes mean/proportion/paired; mSPRT; e-detector `kind="cusum"`), `validation.e_benjamini_hochberg`, `detect.page_cusum(axis=)` | Anytime-valid monitoring of error rates and A/B comparisons; peeking is free. |
| **M3** | `_null` (date-block, entity-paired, shared plans, Romano–Wolf max-T, persistence **refusal**), `_rolling` (`ts.rolling_drift`, `xs_drift`), registry spec, prefix and leakage suites | Honest nulls on panels; drift as prefix-invariant **features**. |
| **M4** | `_multivariate` (MMD rff/quadratic/block, energy, C2ST), `_shift` (MLLS/BBSE/RLLS, `DensityRatio`), `conformal.weighted_conformal_quantile` | Multivariate detection and shift correction integrated with conformal. |
| **M5** | `_spc`, `_arl` (Markov, one-pass simulation, guaranteed performance, alarm budget) | Phase I/II SPC across many streams with calibrated ARL. |
| **M6** | `_fast` numba kernels, pruned betting CS, `benchmarks/monitor/`, docs page, mkdocs nav, CHANGELOG, conformance entries | Measured budgets in the docs. |

M1 + M2 alone is shippable and covers the named caller. M3 is what makes the panel
claims defensible.

---

## 12. Boundaries with sibling plans

| Sibling | Boundary |
|---|---|
| 1 forecast-evaluation-and-sharpe-inference (plan landed; read) | It owns Sharpe tests, CW/GW, VaR/ES backtests (Kupiec/Christoffersen), luck-vs-skill and CORP. Its Giacomini–Rossi `mode="monitor"` is a fixed-sample diagnostic with no anytime claim. **It assigns anytime-valid monitoring of Sharpe/loss (CS, e-values, CUSUM on loss differentials) to this plan.** It calls `monitor.e_process_mean` / `confidence_sequence` on the loss differential. We write no forecast-comparison statistic. It proposes `validation/_hac.py` and `_resample.py` as single homes, and we **import** them if an HAC variance is ever needed (e.g. mSPRT on autocorrelated metrics). |
| 3 covariance-and-market-state (plan landed; read) | It owns `panelary/covariance/`: `estimate(...)` → `CovEstimate` (`solve`, `inv_quad`, `logdet`, `cond`), shrinkage (LW, OAS, QIS), **turbulence** (the Mahalanobis market-state feature), and the `Schedule` object. Its plan names this one as a caller: Hotelling T², MEWMA and MCUSUM call `cov.estimate` on **Phase-I rows only**, and that train/monitor discipline is ours. We reuse `Schedule` for every epoch and stride grid. If it has not landed by M5, a private ≤ 15-line OAS (Chen et al. 2010) plus the two schedule rules are allowed, with a TODO and parity tests, and are deleted on landing. **Request to sibling 3:** a row-batched `inv_quad(X: (k, p)) -> (k,)`, since charts evaluate thousands of streams per date. |
| 4 label-weights-and-event-sampling | It owns uniqueness/concurrency/time-decay sample weights (and the CUSUM event filter already in `feature_extractors`). We own **density-ratio** and **label-shift** importance weights. They compose multiplicatively, and neither computes the other's. |
| 7 causal-state-space-and-regimes | It owns Kalman, HMM and **BOCPD** (Bayesian changepoint posteriors). We own frequentist e-detectors and SPC. `detect` keeps CUSUM/SR/FOCuS. Nobody reimplements Page's CUSUM. |
| 10 panel-causal-inference (plan landed; read) | It owns SC/DiD/IFE and CWZ conformal counterfactual inference. It ships **no sequential test**, and it leaves `conformal/` unmodified, so our additive `weighted_conformal_quantile` edit does not conflict. It exposes prefix-invariant gap/CAR series, and a sequential test on them is a **call** into `monitor._sequential`. Its plan has no propensity-score logistic, so M4 creates the L2 logistic IRLS as a shared private primitive, `panelary/_internal/_glm.py::logistic_irls`, rather than burying it in `_shift.py`. |
| 5, 6, 8, 9 | No overlap. Tail-risk (9) may monitor exceedance rates with our CS; that would be a call, not a copy. |

---

## 13. Risks and open questions — resolve with a benchmark, not an opinion

1. **Evidence kind.** The engine's `EvidenceKind` (truepoint `schema/diagnosis.py`)
   has no "statistical-test" member. The proposal emits `"arithmetic-recheck"` (a
   deterministic, seeded recomputation the reader can re-run) and asks the engine
   owner. `CONTRACT.md` is frozen, and this plan must not change it.
2. **Rolling-PSI numpy cost at full panel scale** (estimated at 150 s) may force
   `fast` as the recommended path. Measure in M3 before promising more.
3. **The AD `adinf` mapping for k = 2 at small N**: verify calibration at n = 20–50.
   If it is off, fall back to a Monte Carlo table.
4. **Is D = 128 enough for RFF-Hotelling power** against quadratic MMD at N = 2000?
   The `depend` contract measured that D = 256 was not enough for RFF-HSIC p-values
   within 10%. Measure the power gap and set D from the measurement.
5. **Date-block null at 0.9 ≤ φ < 0.95** measured 10% at L = 25. Find the block rule
   (`pair_block_length` vs a fixed L ∝ 1/(1−φ)) that brings it into [3%, 8%], or lower
   the refusal threshold.
6. **Betting CS alive-set pruning** is an estimate (≥ 20×). If it misses, keep
   `"betting"` opt-in and numba-only.
7. **The Polars Categorical rework (1.32)**: `cut().to_physical()` code order on 1.35
   is unverified. The fallback is ready.
8. **The mSPRT plug-in variance** is only asymptotically valid. Show its size at small
   n and default to `sigma2="plugin"` only if the size is ≤ α + 1%.
9. **Guaranteed-performance limits** assume Gaussian Phase I. Measure how badly heavy
   tails (t₃) break the guarantee, and when the nonparametric bootstrap is needed.

---

## 14. Caller

> **AGENTS.md rule:** *no new public surface without a named caller in the engine and
> a test in the engine that exercises it.*

The user explicitly asked for this plan. The rule still applies to **what gets
built**, and M1–M2 are scoped to it.

**Most plausible caller: `truepoint/src/truepoint/quant/`.** It is named in
`AGENTS.md` as panelary's consumer and **does not exist yet**.

Concrete call sites:

1. **Re-test comparability.** STRATEGY requires re-tests to use *newly generated*
   sealed sets. `drift_report` on **non-secret question metadata** (category mix,
   difficulty score, as-of-date distribution) between the original and the re-test
   set certifies that the two sets are comparable. Each finding becomes
   `Evidence(**check.to_evidence())`. The input never includes question text,
   answers, seeds or answer-key columns (invariant 11).
2. **Continuous monitoring of a client system's error rate.** Anytime-valid CS on 0/1
   marks, `e_process_paired` for "did the remediation help?" on matched items, and an
   e-detector on the mark stream. Today `truepoint/inference/errorbars.py` computes a
   fixed-n Wilson interval. §1.3 shows what happens when that interval is re-checked
   continuously.

**Constraints and honest risk:**
- truepoint's `_firewall.py` forbids `diagnose/` from importing panelary, so these
  calls must live in `quant/` or `report/`.
- truepoint does not depend on panelary today.
- If the engine owner prefers a ~60-line in-repo CS in `truepoint/inference/`, then M2
  has **no caller** and should be kept private (`_sequential.py`, not exported) until
  one exists.

M3–M6 are research-workflow features (panel drift, SPC, shift correction) with **no
engine caller yet**. Build them only when one appears, or keep them behind
`panelary.monitor` without top-level verbs.

---

## 15. References

† = not verified this session.

- Alexandari, Kundaje & Shrikumar (2020). Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. ICML.
- Anderson, T. W. (1962). On the distribution of the two-sample Cramér–von Mises criterion. Ann. Math. Stat. 33.
- Azizzadenesheli, Liu, Yang & Anandkumar (2019). Regularized learning for domain adaptation under label shifts (RLLS). ICLR.
- Baringhaus & Franz (2004). On a new multivariate two-sample test. J. Multivariate Anal. 88.
- Besag & Clifford (1991). Sequential Monte Carlo p-values. Biometrika 78.
- Bickel, Brückner & Scheffer (2009). Discriminative learning under covariate shift. JMLR 10.
- Bifet & Gavaldà (2007). Learning from time-changing data with adaptive windowing (ADWIN). SDM.
- Biggs, Schrab & Gretton (2023). MMD-FUSE. NeurIPS†.
- Brook & Evans (1972). An approach to the probability distribution of CUSUM run length. Biometrika 59.
- Champ & Woodall (1987). Exact results for Shewhart control charts with supplementary runs rules. Technometrics 29 (ARL₀ 91.75 for WE 1–4†).
- Chen, Wiesel, Eldar & Hero (2010). Shrinkage algorithms for MMSE covariance estimation (OAS). IEEE TSP 58.
- Chwialkowski, Ramdas, Sejdinovic & Gretton (2015). Fast two-sample testing with analytic representations of probability measures. NeurIPS.
- Crosier (1988). Multivariate generalizations of cumulative sum quality-control schemes. Technometrics 30.
- Gandy & Kvaløy (2013). Guaranteed conditional performance of control charts via bootstrap methods. Scand. J. Stat. 40.
- Garg, Wu, Balakrishnan & Lipton (2020). A unified view of label shift estimation. NeurIPS.
- Gnedenko & Korolyuk (1951). On the maximum discrepancy between two empirical distributions†.
- Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012). A kernel two-sample test. JMLR 13.
- Henzi & Ziegel (2022). Valid sequential inference on probability forecast performance. Biometrika 109.
- Hodges (1958). The significance probability of the Smirnov two-sample test. Ark. Mat. 3.
- Howard, Ramdas, McAuliffe & Sekhon (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann. Stat. 49.
- Jensen, Jones-Farmer, Champ & Woodall (2006). Effects of parameter estimation on control chart properties. JQT 38.
- Jitkrittum, Szabó, Chwialkowski & Gretton (2016). Interpretable distribution features with maximum testing power. NeurIPS.
- Johari, Koomen, Pekelis & Walsh (2022). Always valid inference: continuous monitoring of A/B tests. Oper. Res. 70.
- Jones, Champ & Rigdon (2001). The performance of EWMA charts with estimated parameters. Technometrics 43.
- Kanamori, Hido & Sugiyama (2009). A least-squares approach to direct importance estimation (uLSIF). JMLR 10.
- Kim, Ramdas, Singh & Wasserman (2021). Classification accuracy as a proxy for two-sample testing. Ann. Stat. 49.
- Kritzman & Li (2010). Skulls, financial turbulence, and risk management. FAJ 66.
- Lei & Candès (2021). Conformal inference of counterfactuals and individual treatment effects. JRSS-B 83.
- Lipton, Wang & Smola (2018). Detecting and correcting for label shift with black box predictors (BBSE). ICML.
- Liu, Yamada, Collier & Sugiyama (2013). Change-point detection in time-series data by relative density-ratio estimation. Neural Networks 43.
- Lopez-Paz & Oquab (2017). Revisiting classifier two-sample tests. ICLR.
- Lowry, Woodall, Champ & Rigdon (1992). A multivariate exponentially weighted moving average control chart. Technometrics 34.
- Lucas & Saccucci (1990). Exponentially weighted moving average control schemes. Technometrics 32.
- Marsaglia & Marsaglia (2004). Evaluating the Anderson–Darling distribution. J. Stat. Softw. 9 (accuracy of `adinf`†).
- Nelson (1984). The Shewhart control chart — tests for special causes. JQT 16.
- Page (1954). Continuous inspection schemes. Biometrika 41. Hinkley (1971). Inference about the change-point from cumulative sum tests. Biometrika 58.
- Podkopaev & Ramdas (2023). Sequential predictive two-sample and independence testing. NeurIPS.
- Qin (1998). Inferences for case-control and semiparametric two-sample density ratio models. Biometrika 85.
- Rabanser, Günnemann & Lipton (2019). Failing loudly: an empirical study of methods for detecting dataset shift. NeurIPS.
- Ramdas, Grünwald, Vovk & Shafer (2023). Game-theoretic statistics and safe anytime-valid inference. Stat. Sci. 38.
- Robbins (1970). Statistical methods related to the law of the iterated logarithm. Ann. Math. Stat. 41.
- Roberts (1959). Control chart tests based on geometric moving averages. Technometrics 1.
- Runger & Prabhu (1996). A Markov chain model for the multivariate EWMA control chart. JASA 91.
- Saerens, Latinne & Decaestecker (2002). Adjusting the outputs of a classifier to new a priori probabilities. Neural Comput. 14.
- Schrab, Kim, Albert, Laurent, Guedj & Gretton (2023). MMD aggregated two-sample test. JMLR 24.
- Scholz & Stephens (1987). K-sample Anderson–Darling tests. JASA 82.
- Sejdinovic, Sriperumbudur, Gretton & Fukumizu (2013). Equivalence of distance-based and RKHS-based statistics. Ann. Stat. 41.
- Shekhar & Ramdas (2024). Nonparametric two-sample testing by betting. IEEE Trans. Inf. Theory 70(2), 1178–1203.
- Shin, Ramdas & Rinaldo (2023). E-detectors: a nonparametric framework for sequential change detection. New England J. Stat. Data Sci. 2(2), 229–260.
- Siddiqi (2006). Credit Risk Scorecards. Wiley (origin of the 0.1/0.25 PSI rule).
- Stephens (1970). Use of the Kolmogorov–Smirnov, Cramér–von Mises and related statistics without extensive tables. JRSS-B 32.
- Sugiyama, Suzuki, Nakajima, Kashima, von Bünau & Kawanabe (2008). Direct importance estimation for covariate shift adaptation (KLIEP). AISM 60.
- Székely & Rizzo (2013). Energy statistics: a class of statistics based on distances. JSPI 143.
- Tibshirani, Barber, Candès & Ramdas (2019). Conformal prediction under covariate shift. NeurIPS.
- Tracy, Young & Mason (1992). Multivariate control charts for individual observations. JQT 24.
- Ville (1939). Étude critique de la notion de collectif.
- Wang & Ramdas (2022). False discovery rate control with e-values. JRSS-B 84.
- Waudby-Smith & Ramdas (2024). Estimating means of bounded random variables by betting. JRSS-B 86.
- Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas (2024). Time-uniform central limit theory and asymptotic confidence sequences. Ann. Stat. 52(6), 2613–2640.
- Yamada, Suzuki, Kanamori, Hachiya & Sugiyama (2013). Relative density-ratio estimation for robust distribution comparison (RuLSIF). Neural Comput. 25.
- Yu, Suresh, Choromanski, Holtmann-Rice & Kumar (2016). Orthogonal random features. NeurIPS.
- Yurdakul & Naranjo (2020). Statistical properties of the population stability index. J. Risk Model Validation 14 (PSI ≈ (1/n+1/m)·χ²_{B−1}; confirmed via the WMU dissertation and the paper abstract).
- Zaremba, Gretton & Blaschko (2013). B-test: a non-parametric, low variance kernel two-sample test. NeurIPS.
- Licences to verify before use as oracles: `hyppo` (MIT†), `confseq` (MIT†), `alibi-detect` (BSL†).
