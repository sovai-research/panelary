# `panelary/counterfactual/`: build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started. This is a plan only.** No package code exists
> for anything below. Every number in "Why this exists" and "Measured facts" was
> measured on 2026-09-29 on an Apple M5 Pro (15 cores, 48 GB) with NumPy 2.5.3 on
> Accelerate BLAS, polars 1.44.2 and CPython 3.13, using scratch scripts that call
> the existing `panelary.econ` code. They are measurements, not estimates.

This contract covers **staggered difference-in-differences, finance event studies,
synthetic control, interactive fixed effects and counterfactual inference** for long
panels. It is one of ten sibling plans written in parallel (§13 lists them). It
deliberately reverses one line of `plans/done/econometric-integration-plan.md` §10,
which parked "modern DiD (Callaway-Sant'Anna, synthetic DiD)" until "Panelary
deliberately moves there". This is that move. The justification is the named caller
in §14, and nothing here ships publicly without it (AGENTS.md).

The core stays **numpy + polars**. scipy, sklearn and cvxpy never appear on an import
path. numba is optional behind the existing `fast` extra and gets a single kernel. The
oracles (R `did`/`synthdid`/`fixest`, python `csdid`/`pyfixest`, `scipy.optimize`) are
test-only, behind `pytest.importorskip` or stored reference values.

---

## 1. Why this exists

### 1.1 The estimator everyone runs is wrong under staggered adoption, and this repo makes it easy to run

`pn.regression(df, y=..., x=["treated"], absorb=[entity, time])`, which is
`econ.hdfe`, is exactly the static two-way fixed-effects (TWFE) regression that
Goodman-Bacon (2021), de Chaisemartin–D'Haultfœuille (2020), Sun–Abraham (2021) and
Borusyak–Jaravel–Spiess (2024) proved is biased under staggered timing with
heterogeneous or dynamic effects. Baker, Larcker & Wang (2022, *JFE*) replicated
published finance and accounting studies and found the TWFE conclusions can flip
sign. We reproduced the mechanism with the repo's own code. `econ.hdfe` gives the TWFE
estimate. `econ._hdfe.demean` fit on the untreated cells gives the imputation
estimator. The design is N=3000, T=40, three cohorts adopting at t=10/20/30 with an
effect of 0.2·(e+1) that grows with event time e, and unit FE + time FE + N(0,1) noise:

| Design | True ATT | TWFE (`econ.hdfe`) | Imputation (BJS) |
|---|---|---|---|
| no never-treated units, identifiable cells | **1.759** | **0.098** (-94%) | 1.754 |
| 30% never-treated | **2.412** | **1.226** (-49%) | 2.420 |

### 1.2 The obvious port of the fix has its own silent bug

With no never-treated units, every unit is treated in periods t ≥ 30. So the time
effect λ_t is **not identified** there. The group-mean sweep in `econ._hdfe.demean`
treats an empty group as "offset 0" (`group_mean`: "empty groups -> 0"). A naive
imputation then fills 30,000 treated cells with λ_t := 0 and reports **2.625 against
a truth of 2.411 (+8.9%)**. It raises no warning and produces no NaN. Hence invariant
6 (§5): unidentified cells **fail closed**.

### 1.3 Finance event studies over-reject when event dates cluster

All events here fall on one date and the abnormal returns share a common shock with
cross-correlation r̄. We ran 2,000 Monte Carlo replications at a nominal 5% level:

| n events, r̄ | BMP size | Kolari–Pynnönen-adjusted BMP size |
|---|---|---|
| 50, 0.02 | **16.6%** | 5.4% |
| 100, 0.05 | **43.9%** | 5.8% |

This is the event-study version of the i.i.d.-null error that `depend` exists to fix.
Earnings seasons, index rebalances, macro announcements and regulatory dates all cluster.

### 1.4 The honest estimators are cheap if written as sufficient statistics and BLAS calls

A per-cell loop of regressions (R `did` ≤ 2.1) or a general QP solver is slow. The
Stata `csdid` 2.0 release notes report a 10–308× speedup from "the shared structure
across cohort-period comparisons". We measured the same effect in numpy (§1.5).

### 1.5 Measured facts that dictate the design

| Fact | Measurement | Consequence |
|---|---|---|
| Two-way FE on the untreated mask (N=10k, T=200, 20 cohorts, 1.23M rows, 4 columns): `econ._hdfe.demean` alternating projections | 18 sweeps, **0.92 s** | correct, but not the fast path for 2 FE dimensions |
| The same system solved directly through the T×T Schur complement (one GEMM plus a Cholesky) | **0.026 s** (35×); max \|fit diff\| vs AP **5.9e-13** | `_fe.py` uses the direct solve for unit+time FE and falls back to `demean` for ≥3 dimensions |
| All 4,000 CS ATT(g,t) cells (20 cohorts × 200 periods) from cohort means (one `H @ Y` GEMM) | **0.7 ms**; per-cell numpy loop ≈ 90 ms; agreement 7e-15 | no loop over cells, ever |
| Multiplier bootstrap as V(B×N) @ Ψ(N×J), B=999, N=10k | J=400: GEMM **19 ms** + draws 29 ms; J=4,000: **202 ms**, Ψ = 320 MB | uniform bands in one GEMM; chunk Ψ above ~500 columns |
| SC simplex QP, J=500 donors, T0=100: **primal active set** | **1.7 ms**, 28 iterations, support 22, KKT residual **3.8e-13** | the default ADH solver |
| the same problem, FISTA + exact simplex projection, 5,000 iterations | 223 ms and **not converged** (KKT 1.5e-6): the Gram is rank ≤ 100 | FISTA is not the unpenalised SC solver |
| the same problem, `scipy.optimize` SLSQP | 3.9 s, objective **worse** by 1.5e-9 | no QP dependency is needed or wanted |
| SDID unit weights (ridge ζ²T0·I, J=500): FISTA with adaptive restart | **12 ms**, 158 iterations; active set 946 ms (support 406); objectives equal to 1.4e-13 | the solver follows the structure: ridge → FISTA, no ridge → active set |
| 200 ridge-penalised weight problems batched through one shared Gram | **217 ms** | SDID placebo and bootstrap replicates are batched |
| Simplex projection of a 1000×500 batch | Michelot (vectorised) **4.3 ms**; Duchi sort 7.9 ms; equal to 8.9e-16 | Michelot in the loop, Duchi as the test oracle |
| 50,000 events, 250-day estimation window, 4 factors + constant, gather + batched normal equations + ARs | **0.37 s** | no per-event loop; chunk by memory budget |
| Thin SVD of 10k×200 / `shape.randomized_svd` k=20 | 46 ms / **8 ms** | soft-impute switches to rSVD while rank < 0.1·min(N,T) |
| Masked ALS for IFE, rank 4, 10k×200, 30% of cells masked | 16 iterations, 1.5 s with einsum Grams | M5 must form Grams as `(W∘Y) @ (F⊗F)` GEMMs (invariant 4) |
| Soft-impute at a too-small λ | 122 iterations, 6.0 s, rank 156 (overfit) | λ is chosen by a warm-started path with CV. A fixed default is refused |

---

## 2. What already exists, and must NOT be duplicated

Everything below was verified by reading the file on 2026-09-29. **Reuse it. Never
copy it.**

| Need | Existing code (file:function) | How it is used here |
|---|---|---|
| N-way FE absorption | `econ/_hdfe.py:demean(values, codes, n_levels, tol, max_iter)` returns `(resid, offsets, n_iter, max_dev)` | the fallback for ≥3 FE dimensions and the oracle for `_fe.py`. It has **no acceleration** today (see §12 Q2) |
| TWFE / dense oracle regressions | `econ/_hdfe.py:hdfe(..., absorb, cluster, vcov)` (classical/HC1/1- and 2-way cluster/Driscoll–Kraay) | TWFE for diagnostics. It is also the dense saturated-dummy oracle for Sun–Abraham and the Bacon identity |
| Distributions, OLS, grouping, HAC | `econ/_common.py`: `norm_cdf`, `norm_ppf`, `t_sf`, `chi2_sf`, `f_sf`, `ols`, `pinv_sym`, `factorize`, `group_sum`, `group_mean`, `newey_west_lrv`, `newey_west_scalar`, `auto_bandwidth`, `winsorize` | every p-value, every Wald test, calendar-time NW SEs |
| DML / PDS (the current `pn.causal`) | `econ/_dml.py:dml_partial_linear`, `post_double_selection`, `rigorous_lasso` | **untouched.** DML answers a different question (one continuous treatment with unconfoundedness). This plan adds design-based counterfactual estimators |
| CCE / mean group | `econ/_panel.py:cce_mg`, `cce_pooled`, `mean_group`, `pesaran_cd` | CCE is **not** a counterfactual estimator. `pesaran_cd` is reused as a diagnostic on IFE residuals |
| Fama–MacBeth | `econ/_famamacbeth.py:fama_macbeth` | not reused. Calendar-time portfolios run a time-series regression, not a cross-sectional one |
| Factor count | `reduce/_n_factors.py:bai_ng(X, max_r, criterion)`, `eigenvalue_ratio`, `n_factors` | the IFE rank default on the complete control-only block |
| Randomized SVD | `shape/_rsvd.py:randomized_svd(A, k, n_oversamples, n_iter, seed)` (HMT 2011, CholeskyQR2) | SVT in soft-impute and IFE initialisation |
| Long → dense tensor | `shape/_tensor.py:build_tensor(panel, value_cols, forward_fill, z_normalize)` returns `PanelTensor` (float64, NaN = missing) and `to_long` | the one materialisation boundary. Always called with `forward_fill=False, z_normalize=False`. **Trap:** `to_long` defaults to `dtype=pl.Float32`, so always pass `pl.Float64` |
| Memory guard | `shape/_axes.py:ShapeBudgetError`, `default_max_bytes()` | refuse to allocate before any Ψ, gather or tensor exceeds budget |
| Wild/multiplier draws | `validation/_bootstrap.py:wild_bootstrap(residuals, n_boot, distribution, seed)` | the multiplier **law** (rademacher/mammen/normal). It is reused as the law but not as the draw loop: we need per-replicate streams (§5 inv. 2) and a GEMM, not `v * e` broadcasting |
| Block resampling | `validation/_bootstrap.py:block_bootstrap_indices`, `stationary_bootstrap` | the IFE/MC bootstrap over units and CWZ iid permutations |
| Multiplicity | `validation/_selection_stats.py:romano_wolf`, `holm_bonferroni`, `benjamini_hochberg` | event-test families across windows and horizons (sup-t bands handle event time) |
| Conformal quantile | `conformal/_intervals.py:conformal_quantile(scores, alpha)` | CWZ confidence-set inversion uses the same finite-sample convention. `conformal/` itself is **not modified** |
| As-of join, business calendars | `core/asof.py:asof_join`; `core/_calendar.py:BusinessDays`, `shift_forward` | treatment timing from vintage tables. Event day-0 mapping and the estimation gap in trading days |
| Leak verifiers | `testing.py:assert_no_lookahead`, `assert_prefix_invariant`, `assert_no_train_test_leak` | every as-of variant is tested with **both** |
| Synthetic panels | `synth/_generate.py:generate_panel`, `synth/_truth.py:GroundTruth` | **cannot plant a treatment today** (verified: the value is intercept + factors + cluster + lagged signal + AR(1) idio). §8.1 specifies the additive extension |
| Imputation | `imputation.py:CafeImputer` (causal, point-in-time) | **not** a counterfactual imputer: it fills missing *observations* from the past, whereas IFE/MC impute *untreated potential outcomes* from both sides in time. §6 keeps them apart in the docs |
| Verb facade | `_internal/_verbs.py:causal(data, *, method="dml", y, treatment, controls, ...)` (dml/pds only) | extended with new `method=` values **only in M8**, after the caller exists |
| Top-level name | `panelary/__init__.py`: "The package is `leakage`, not `causal`: `causal` is already one of the eight verbs." | the new package is **`counterfactual`**, never `causal` (§4) |

Grep on 2026-09-29 found no DiD, SC, event-study, soft-impute or IFE code anywhere in
`panelary/`.

---

## 3. Scope and non-goals

**In scope:**
- Staggered DiD: BJS imputation (conservative variance, pre-trend test); Callaway–Sant'Anna
  ATT(g,t) with OR/IPW/DR, never- and not-yet-treated controls, four aggregations, and
  multiplier-bootstrap uniform bands; Sun–Abraham IW.
- TWFE diagnostics: the Goodman-Bacon decomposition and the dCDH negative-weight report.
- Synthetic control family: ADH SC, augmented SC (ridge), synthetic DiD with
  placebo/jackknife/bootstrap variance, in-space and in-time placebos, and
  Chernozhukov–Wüthrich–Zhu conformal inference.
- Latent-factor counterfactuals: IFE (Bai 2009) by ALS with rank selection, and
  MC-NNM (Athey et al. 2021) by soft-impute, each with an as-of variant.
- Finance event studies: mean-, market- and factor-model abnormal returns, CAR, BHAR,
  and the Patell, BMP, Kolari–Pynnönen, Corrado/GRANK, generalised sign and
  skewness-adjusted tests; clustered-date handling; calendar-time portfolios.
- Optional later milestones (M7): dCDH `DID_M`, HonestDiD-style sensitivity, and
  spillover-aware DiD that consumes sibling 6's exposures.

**Non-goals (decisions, not omissions):**
- **Continuous or multi-valued treatment DiD** (Callaway, Goodman-Bacon & Sant'Anna 2024).
  It needs a different identification argument. Revisit only on caller demand.
- **Synthetic-control V by global optimisation.** Nested V is non-robust and
  optimiser-dependent (Klößner et al. 2018†). It ships as an opt-in with multi-start
  and a warning, and is never the default.
- **Bayesian structural time series / CausalImpact.** These belong to sibling 7
  (causal-state-space-and-regimes). §6.10 defines the protocol that lets their
  counterfactual path use our placebo and conformal inference.
- **Unknown-date interventions.** Break detection is `panelary.detect`'s job. Every
  estimator here needs a known, as-of adoption date.
- **Repeated cross-sections** for CS. Panel data only through M2; the RCS estimator is a
  follow-up.
- **Causal forests / heterogeneous-effect ML.** Out of scope, as already stated in the
  econ plan.
- **Any QP/LP/convex solver dependency** (cvxpy, osqp, quadprog). §6 shows it is
  unnecessary and slower.

---

## 4. Module placement and public API

### 4.1 Placement

We add a new top-level subpackage, **`panelary/counterfactual/`**, reached as
`pn.counterfactual`. It is lazy-initialised through PEP 562 like `shape`, so the
cold-import budget in `tests/test_import_hygiene.py` is untouched. Reasons:

1. **The name `causal` is taken** by the golden-path verb (`_internal/_verbs.py:causal`),
   and `panelary/__init__.py` records that `leakage` avoided it for this reason. A
   `causal` subpackage would force a fourth `_SubpackageVerb` collision for no gain.
2. **Every estimator here is a counterfactual estimator.** DiD imputation, IFE, MC and
   SC impute Y(0) for treated cells. An event study's "normal return" is Y(0) and the
   abnormal return is Y − Y(0). One result schema (`effects`: entity, time, observed,
   counterfactual, effect) and one aggregation layer serve all of them (Liu, Wang & Xu
   2024).
3. **Not inside `econ/`.** `econ/__init__.py` imports eagerly. The name `econ.effects`
   reads as "fixed effects". `synth/` already means the synthetic *panel generator*, so
   SC files are named `_sc*`, never `synth*`.

The verb `pn.causal` gains `method="did" | "sc" | "sdid"` in **M8 only**, dispatching
here. It keeps `method="dml"` as the default, so no current caller changes.

### 4.2 Layout and file ownership (touch ONLY your files)

| File | Contents | Owner | M |
|---|---|---|---|
| `_design.py` | cohort derivation (as-of), absorbing check, anticipation, identifiability masks, dense layout via `build_tensor`, event-time grid | A | M0 |
| `_fe.py` | direct two-way FE Schur solver (multi-RHS, weights, iterative refinement, components); ≥3-dim fallback to `econ._hdfe.demean` | A | M0 |
| `_influence.py` | IF assembly, per-replicate seeded multipliers, cluster summation, IQR-SE, sup-t bands, chunked GEMM | B | M1 |
| `_imputation.py` | BJS estimator, implied weights, conservative variance, pre-trend test | A | M1 |
| `_cs.py` | CS ATT(g,t): unconditional (M1); OR/IPW/DR with IPT (M2); aggregations | B | M1–M2 |
| `_sa.py` | Sun–Abraham IW (CS-equivalence path and regression path) | B | M1 |
| `_twfe_diag.py` | Goodman-Bacon decomposition; dCDH weights, σ_fe | C | M1 |
| `_simplex.py` | Michelot / Duchi projection, primal active set, batched FISTA-restart, KKT certificate, synthdid-compatible Frank–Wolfe | D | M3 |
| `_sc.py`, `_sdid.py`, `_placebo.py` | ADH SC + V options + ASCM; SDID + variance; space/time placebos | D | M3 |
| `_conformal.py` | CWZ permutation inference over any `CounterfactualModel` | E | M4 |
| `_factor.py`, `_mc.py` | IFE masked ALS + rank selection; MC-NNM soft-impute + λ path + CV | F | M5 |
| `_events.py`, `_event_tests.py`, `_calendar_time.py` | event alignment, gather, batched OLS, AR/CAR/BHAR; tests; calendar-time portfolios | G | M6 |
| `_asof.py` | as-of monitors: `sc_gap_asof`, `event_car_asof`, `factor_counterfactual_asof`, `cs_att_asof` | E | M3–M6 |
| `_results.py` | `DiDResult`, `SCResult`, `CounterfactualResult`, `EventStudyResult` (fixed schemas, `to_json`) | B | M1 |
| `_fast.py` | the optional numba kernel (batched active set), lazily imported via `_deps.have("numba")` | D | M3 |
| `synth/_config.py`, `synth/_generate.py`, `synth/_truth.py` | the additive treatment-planting dials (§8.1) | H | M0 |
| `tests/test_counterfactual_*.py`, `tests/data/counterfactual_reference.json` | all tests and stored oracles | I | all |
| `__init__.py`, verb wiring, registry specs, docs, CHANGELOG | orchestrator. Do not create | — | M8 |

### 4.3 Public API (names final; exports gated on §14)

```python
import panelary as pn
cf = pn.counterfactual

# --- staggered DiD --------------------------------------------------------------
cf.staggered_did(data, *, y, cohort=None, treated=None, method="imputation",  # |"cs"|"sa"
                 anticipation,                    # REQUIRED keyword: no default (trap T3)
                 control="not_yet_treated",       # |"never_treated"|"last_treated"
                 covariates=None, covariate_timing="base",   # trap T5
                 est_method="dr",                 # CS only: "or"|"ipw"|"dr"
                 base_period="varying",           # CS only: |"universal"
                 unbalanced="balance",            # |"pairwise"  (CS)
                 horizons=None, pre_periods=None, balance_e=None,
                 weights=None, cluster=None,
                 inference="analytic",            # |"bootstrap"|"randomization"
                 n_boot=999, multiplier="rademacher", seed=0, alpha=0.05,
                 entity=None, time=None) -> DiDResult
cf.imputation_did(...); cf.callaway_santanna(...); cf.sun_abraham(...)   # same kwargs, one method each
cf.pretrend_test(data, *, y, cohort|treated, anticipation, n_leads, ...) -> TestResult   # BJS Test 1
cf.bacon_decomposition(data, *, y, treated, entity=None, time=None) -> pl.DataFrame
cf.twfe_weights(data, *, treated, ...) -> TWFEWeights     # dCDH: weights, n_negative, sigma_fe

# --- synthetic control family ---------------------------------------------------
cf.synthetic_control(data, *, y, treated_unit, treatment_time, method="sc",  # |"ascm"|"sdid"|"did"
                     donors=None, exclude=None, predictors=None, v="identity",  # |"regression"|"nested"
                     penalty=0.0, ridge="auto", solver="exact",   # |"synthdid" (parity mode)
                     pre_start=None, gap=0,
                     inference="placebo",   # |"jackknife"|"bootstrap"|"conformal"|"none"
                     n_boot=200, seed=0, alpha=0.05, entity=None, time=None) -> SCResult
cf.synthetic_did(...)                                     # = synthetic_control(method="sdid")
cf.placebo_test(result, *, kind="space"|"time", fake_times=None, max_pre_rmspe_ratio=None)
cf.conformal_counterfactual(result_or_model, *, null=0.0, permutations="moving_block"|"iid",
                            q=1, grid=None, n_perm=1000, seed=0) -> ConformalResult

# --- latent-factor counterfactuals -----------------------------------------------
cf.interactive_fixed_effects(data, *, y, cohort|treated, anticipation, n_factors="cv",
                             max_factors=8, covariates=None, fixed_effects="two-way",
                             tol=1e-10, max_iter=2000, inference="parametric", n_boot=200, seed=0)
cf.matrix_completion(data, *, y, cohort|treated, anticipation, lam="cv", n_lambda=20,
                     n_folds=5, cv="random_cells"|"last_periods", fixed_effects=True, seed=0)

# --- finance event studies --------------------------------------------------------
cf.abnormal_returns(returns, events, *, ret="ret", model="market",  # |"factor"|"mean"|"market_adjusted"
                    market=None, factors=None,
                    event_window,                  # REQUIRED, e.g. (-1, 1)
                    estimation_window=250, gap=10, min_estimation_obs=200,
                    knowledge_time=None, close_cutoff=None,   # trap T11
                    exclude_other_events=True, calendar=None,
                    entity=None, time=None) -> EventStudyResult
cf.event_tests(result, *, windows=((0, 0), (-1, 1)),
               tests=("patell", "bmp", "kp", "grank", "gsign"), cluster_adjust="auto")
cf.bhar(result, *, horizon, benchmark="model"|"reference", ...) -> pl.DataFrame
cf.calendar_time_portfolio(returns, events, *, holding=(1, 250), factors, weighting="ew"|"vw",
                           nw_lags=-1) -> CalendarTimeResult

# --- as-of monitors (prefix-invariant; the only outputs registered as features) ---
cf.sc_gap_asof(data, *, y, treated_unit, treatment_time, ...) -> pl.DataFrame   # weights frozen at T0
cf.event_car_asof(result) -> pl.DataFrame                   # CAR through t with beta frozen
cf.factor_counterfactual_asof(data, ...) -> pl.DataFrame    # loadings frozen at T0, F_t by XS-OLS
cf.cs_att_asof(data, ...) -> pl.DataFrame                   # not-yet-treated controls, known-at-t cohorts
```

**Result contract.** Every result carries a fixed-schema `pl.DataFrame` `.estimates`.
Columns: `estimand, cohort, time, event_time, estimate, std_error, ci_lower,
ci_upper, band_lower, band_upper, p_value, n_treated, n_control, weight`. Irrelevant
fields are null, so `pl.concat` across methods works. A cell-level `.effects` frame
holds `entity, time, observed, counterfactual, effect, treated, identified`. The design
record is `method, control, anticipation, base_period, covariate_timing, timing
("ex_post"|"as_of"), n_boot, seed, multiplier, solver, converged, n_iter,
kkt_residual, critical_value, dropped_units, unidentified_cells, warnings`. Each result
also has `.summary()`, `.aggregate(kind)` and `.to_json()`. `to_json()` includes
`produced_by="panelary.counterfactual.<fn>@<version>"`, which mirrors truepoint's
`Evidence.produced_by` convention (`truepoint/schema/diagnosis.py`).

---

## 5. Hard invariants (every function)

1. **float64 everywhere.** Upcast Float32 columns. `to_long(..., dtype=pl.Float64)`.
2. **Deterministic seeded randomness, prefix-consistent in B.** Replicate `b` draws
   from `np.random.Generator(PCG64(SeedSequence(seed, spawn_key=(component, b))))`,
   the `synth/_generate.py:_stream` pattern. Replicate b is then identical whatever
   `n_boot` or chunk size is chosen, and adding a component never shifts another.
   Components: 1 = multipliers, 2 = placebo assignment, 3 = CV folds, 4 = unit
   bootstrap, 5 = CWZ permutations, 6 = randomization inference.
3. `np.linalg.solve(A, b[..., None])[..., 0]`. Never `solve(A, b)` on batches
   (NumPy 2 mis-solves when p == batch). Every batched solve gets a test with p == batch.
4. Batched Grams use `@`/`matmul` GEMMs. Never bare `einsum` (the measured ALS in §1.5
   used einsum; the shipped one must not).
5. **No Python loop over cells, units, events or bootstrap draws.** `rolling_map` is
   banned. The permitted loops are over cohorts (≤ ~50), (cohort, control-set) pairs
   (≤ G(G+1)), placebo problems in the active set, λ-path steps and ALS/FISTA
   iterations. Each permitted loop is named in its docstring.
6. **Fail closed on identification.** A treated cell whose unit has no untreated
   observation, whose time has no untreated observation, or whose FE graph component
   contains no treated-and-untreated overlap is `identified=False`, excluded from every
   aggregate and listed in `unidentified_cells`. It is **never** imputed with an
   implicit zero offset (§1.2).
7. **Every result names its design** (§4.3). A p-value or band without `seed`,
   `n_boot`, `multiplier` and `control` is a bug.
8. **Timing is declared.** `timing="ex_post"` estimators are never registered as
   `leakage_safe` features. Only `_asof.py` outputs are registered, with
   `safe_scope="rowwise"`.
9. **Convergence is certified, not assumed.** Iterative solvers return `converged`,
   `n_iter` and a residual certificate: a KKT residual for simplex QPs, a
   normal-equation residual for FE, a relative change for ALS/soft-impute. A
   non-converged fit raises unless `allow_nonconvergence=True`, and even then it warns.

---

## 6. Algorithms: candidates, choice and justification

### 6.1 Design layer (`_design.py`)

- **Cohort.** Either `cohort` (adoption time per entity, null = never within the data)
  or `treated` (0/1 per row). In the second case the cohort is the first time with
  treated == 1, computed on the time-sorted entity. We validate that treatment is
  absorbing; if it is not, we raise and name `method="dcdh"` (M7).
- **Anticipation δ.** Cells in `[G_i − δ, G_i)` are neither untreated (they are
  excluded from Ω0) nor estimands. The base period becomes `g − 1 − δ`.
- **Dense layout** via `shape.build_tensor(..., forward_fill=False, z_normalize=False)`.
  This gives the observation mask W (N×T), Y and X as float64. `default_max_bytes()`
  guards allocation. At 10k×200 the layout costs 16 MB per column.
- **Identifiability** (invariant 6): `n_i = Σ_t W0_it > 0`, `m_t = Σ_i W0_it > 0`, and
  the bipartite unit–time graph of Ω0 is connected per component (see §6.2).

### 6.2 Two-way FE on an arbitrary mask (`_fe.py`), the shared kernel of BJS, IFE-FE, dCDH weights and the pre-trend test

The candidates were:

- (a) `econ._hdfe.demean` alternating projections: 18 sweeps, 0.92 s;
- (b) AP with Irons–Tuck acceleration (fixest);
- (c) CG on the Laplacian (reghdfe);
- (d) a **direct Schur complement solve**.

**Choice: (d) for unit+time FE, (a) for three or more dimensions.**

Normal equations on the mask W: `n_i α_i + Σ_t W_it λ_t = r_i` and `Σ_i W_it α_i +
m_t λ_t = c_t`. Eliminating α gives the T×T system
`S λ = c − Wᵀ(r/n)` with `S = diag(m) − Wᵀ diag(1/n) W`. S is one `(T×N)(N×T)` GEMM,
symmetric PSD, and its null space is spanned by the component indicator vectors. Then
`α = (r − W λ)/n`.

- Eliminate onto the **smaller** of N and T.
- For min(N,T) ≤ 2000 use `eigh(S)`. It is exact and robust, the nullity counts the
  components, and it reports the condition number. Above that size use Cholesky of
  `S + 11ᵀ/T` plus a BFS component check.
- Always do one step of iterative refinement. The measured normal-equation residual
  was 7e-11 before refinement; the target is below 1e-12 after it.
- Multi-RHS: y and all covariates share one factorisation.
- Covariates enter by FWL: residualise [y, X], take β from the residual Gram, then
  recover α and λ from y − Xβ.
- Unit weights w_i scale rows.

Measured at 10k×200, 1.23M rows, 4 columns: **26 ms vs 916 ms**, agreeing to 6e-13.

### 6.3 BJS imputation (`_imputation.py`)

1. Fit `Y_it = α_i + λ_t + X_itᵀβ + ε` on Ω0 with the §6.2 solver.
2. Set `Ŷ_it(0) = α̂_i + λ̂_t + X_itᵀβ̂` and `τ̂_it = Y_it − Ŷ_it(0)` on identified
   treated cells.
3. For estimand weights w on Ω1 (overall, per horizon h, per cohort, or user-supplied),
   the estimate is `τ̂_w = Σ w τ̂`.

**Variance** (BJS Thm 3, conservative, clustered by unit by default):

- Implied weights on untreated cells: `v0 = −Z0(Z0ᵀZ0)⁻¹Z1ᵀw1`. That is the FE system
  solved again with the right-hand side `(Σ_t w_it, Σ_i w_it, Σ w X)`, so it reuses
  the §6.2 factorisation. All H estimands form one multi-RHS solve.
- `σ̂² = Σ_clusters (Σ_{it} v_it ε̃_it)²`, where ε̃ is the Ω0 residual on untreated cells
  and `τ̂_it − τ̄_{g(it)}` on treated cells. τ̄ is the v²-weighted mean within
  cohort×horizon groups, which is BJS's conservative auxiliary model.
- The per-cluster sums are formed **without an N×T×H tensor**:
  `−(ã_iᵀΣ_tε̃_it + ε̃ @ l̃ + (Σ_t X_itε̃_it) b̃)`, which is GEMMs.
- A leave-one-out residual option (`leave_out=True`) is available for units with few
  untreated periods.

**Pre-trend test** (BJS Test 1):

- Use **untreated observations only**. Add K lead dummies `1{t − G_i = −k}` for
  eventually-treated units as extra FWL columns.
- Report a clustered Wald χ²_K via `econ._common.chi2_sf`.
- Testing never feeds back into estimation, which is BJS's separation.

**Complexity:** O(N·T·(k+H)) + O(T³). Budget: < 0.5 s at 10k×200, 20 cohorts, 3
covariates, 40 horizons.

### 6.4 Callaway–Sant'Anna (`_cs.py`, `_influence.py`)

**Unconditional case (M1).**

- Cohort sums come from one GEMM `H @ Y` (H is the (G+1)×N cohort one-hot); the
  second-moment blocks are `Ỹ_kᵀỸ_k` (T×T) per cohort.
- `ATT(g,t) = [Ȳ_g(t) − Ȳ_g(b)] − [Ȳ_C(t) − Ȳ_C(b)]`, where b = t−1 for pre-periods
  under the "varying" base and g−1−δ otherwise.
- **Not-yet-treated control** `C(g,t) = {G > max(t, g−1) + δ} ∪ never, minus cohort g`
  is a cohort **suffix set**, so control sums come from suffix cumulative sums over
  cohorts sorted by G. That is O(1) per cell and O(G·T) for all of them (measured 0.7 ms).
- Analytic SEs come from the same sufficient statistics:
  `Σ_i(ΔY_i − ΔȲ)² = Q[t,t] + Q[b,b] − 2Q[t,b] − n·(ΔȲ)²`.
- **Unbalanced panels.** The default is `"balance"`: drop units with any missing
  outcome and report the count. This matches R `did`'s default. `"pairwise"` uses, per
  (t,b), only units observed at both, through masked GEMMs `(W∘Y)ᵀW`.

**Covariates (M2)**: OR (Heckman–Ichimura–Todd 1997), IPW (Hájek; Abadie 2005), and
the default **improved DR** (Sant'Anna–Zhao 2020: IPT propensity (Graham, Pinto & Egel
2012) plus WLS outcome regression with weights p/(1−p)).

- **Covariates are measured at the base period** and held fixed (trap T5).
- **The key structural fact:** the propensity and OR models depend on (g, control set)
  but **not on t**. The control set only changes when the threshold crosses an adoption
  date. So there are at most G fits (never-treated) or G(G+1) fits (not-yet-treated),
  never G·T.
- Per (g, C), the OR coefficients for **all** periods come from one solve
  `(X_Cᵀ W X_C)⁻¹ X_Cᵀ W Y_C` → p×T. `β(t,b)` is a column difference.
- Each (g, C) Newton solve for IPT/logit is vectorised within the problem. Stopping
  rule: ‖grad‖∞ < 1e-10, with backtracking. Propensity trimming at 0.995 matches DRDID
  and is reported.
- Budget: < 2 s at G=20, N=10k, p=5, not-yet-treated.

**Aggregations** (CS §4): simple, dynamic θ(e) (with `balance_e`), group θ(g) and
calendar θ(t). The IFs include the weight-estimation term.

**Influence functions without forming Ψ(N×K).**

- Ψ·A for any aggregation matrix A (K×J) is per-cohort `Ỹ_k @ M_k`. M_k (T×J) collects
  `a_{(g,t),j}(e_t − e_b)` scaled by N/n_g for treated units, and the negated
  control-side weights for each cohort's control memberships.
- Cost is N·T·J (~8e8 flops at J=400) and memory is N·J.
- The all-cells band (J = K = 4000) is chunked in 512 columns.

**Multiplier bootstrap.**

- Draw V (B×N) row by row with per-replicate streams (inv. 2): Rademacher by default,
  as in R `did`'s `mboot`; Mammen and normal are options.
- Clusters: sum IF rows within a cluster first, then draw one multiplier per cluster.
- `Z = V @ (ΨA)/N` is **one GEMM**. The SE is `bSigma = IQR(Z)/(z.75 − z.25)`, as R
  `did` computes it.
- The sup-t critical value is the (1−α) quantile of `max_j |Z_j|/bSigma_j`, giving
  uniform bands.
- Budget: < 0.3 s (J ≈ 400) and < 1.5 s (J = 4000) at N=10k, B=999.

**Small-N inference** (the caller has tens of clients): `inference="randomization"`
permutes adoption dates across units (seeded, exact enumeration when ≤ `n_perm`) and
recomputes the statistic. That is feasible only because a full refit costs ~1 ms.

### 6.5 Sun–Abraham (`_sa.py`)

Sun–Abraham with never-treated (or last-treated) controls, a universal base g−1 and no
covariates is numerically CS with those settings on a balanced panel†. So the default
path is the §6.4 engine with the IW cohort-share weights and SA Prop. 6 variance (share
estimation included).

Tests still verify it against a **dense saturated regression**:
`econ.hdfe(y ~ Σ_g Σ_e 1{G=g}1{t−g=e} | unit + time)` on a small panel, to 1e-10, and
against `pyfixest` `event_study(estimator="saturated")`.

Unbalanced panels use the regression path, capped at 2,000 interaction columns
(`ShapeBudgetError` above that).

### 6.6 TWFE diagnostics (`_twfe_diag.py`)

- **Bacon decomposition** (balanced panels): the timing groups include never-treated and
  always-treated. Each 2×2 DD is built from the cohort-mean matrix already computed for
  CS, using time cumulative sums, so it is O(1) per pair (≤ 210 pairs at G=20). Weights
  follow Goodman-Bacon eqs. 10a–c.
- **Identity test:** the weights sum to 1 and `Σ s·β̂ = econ.hdfe` TWFE β to 1e-10.
- **dCDH weights:** regress D on two-way FE with the §6.2 solver. The residuals give
  `w_gt ∝ ε̃_gt` on treated cells. Report `n_negative`, the sum of negative weights, and
  `σ_fe = |β_fe|/σ(w)`†.

### 6.7 Simplex-constrained least squares (`_simplex.py`), the kernel of SC, SDID and the CWZ refits

Problem: `min ½wᵀGw − cᵀw` over the simplex, with the Gram G = Y0ᵀY0 (+ ridge·I)
precomputed.

| Candidate | Result (§1.5) | Verdict |
|---|---|---|
| General QP (SLSQP, cvxpy/osqp) | 3.9 s, worse objective, adds a dependency | rejected |
| FISTA + exact projection, no ridge | not converged after 5,000 iterations (rank-deficient G, sublinear rate) | rejected for ridge = 0 |
| **Primal active set** (Lawson–Hanson adapted to Σw = 1; KKT system `[G_SS 1; 1ᵀ 0]` solved by lstsq; exact after finitely many steps) | 1.7 ms, KKT 4e-13 | **default when ridge = 0** (ADH SC, penalised SC, CWZ refits) |
| **FISTA + adaptive restart** (Beck–Teboulle 2009; O'Donoghue–Candès 2015 function restart; step 1/λ_max(G)) + **Michelot projection** | 12 ms, linear rate once strongly convex | **default when ridge > 0** (SDID, ASCM warm start); batched across problems with one shared G |
| Frank–Wolfe (R `synthdid`) | FW with R's stopping rule | only in `solver="synthdid"` parity mode |

**Projection.**

- Vectorised Michelot (1986): iterate `τ = (Σ_active v − 1)/|active|`, drop v ≤ τ. It
  terminates finitely, is exact, and warm-starts from the previous support.
- A per-row mask supports excluded donors (in-space placebos: the diagonal). Duchi et
  al. (2008) is the test oracle. Condat (2016) is noted as sequential and therefore not
  vectorisable in numpy.
- **Hybrid fallback:** if FISTA misses its tolerance, run the active set on the detected
  support.
- **Certificate** (invariant 9):
  `kkt = max(max_S |g_j − μ|, max(0, −min_{j∉S}(g_j − μ))) / max(1, ‖g‖∞)`.

**Non-uniqueness** (ADH with J > T0): the fitted pre-period path is unique, as a
projection onto a convex set, but **w and hence the post-period counterfactual may not
be**. The active set returns a vertex solution (≤ rank+1 donors) with donors ordered by
sorted entity id, so it is deterministic. `SCResult.nonunique` is set when
|S| > rank(Y0_S). `penalty=λ` gives Abadie–L'Hour (2021) penalised SC, which is unique
for λ > 0. It adds `λ‖x1 − x_j‖²` as a **linear** term in c, so the same solver applies.

**numba** (`fast` extra): `_fast.py` holds one `prange` kernel that runs the active set
over a batch of placebo problems. Parity with numpy is 1e-12. Target: 1,000 refits in
< 1 s instead of < 3 s.

### 6.8 SC, ASCM, SDID (`_sc.py`, `_sdid.py`, `_placebo.py`)

**SC.**

- The default predictors are **all pre-period outcomes with V = I** (Doudchenko–Imbens;
  Kaul et al. 2022† show covariates become irrelevant once all lags are included).
- With covariates: `v="regression"` (Synth's regression-based diagonal V) or
  `v="nested"`. Nested uses deterministic multi-start (identity, regression, equal) plus
  softmax-parametrised projected coordinate search, with an optional `scipy` minimiser.
  It always warns and reports `v_objective`.
- `v_validation="split"` fits V on the early pre-period and w on the late pre-period
  (ADH 2015).

**ASCM** (Ben-Michael–Feller–Rothstein 2021): ridge augmentation
`w_aug = w_sc + X0ᵀ(X0X0ᵀ + λI)⁻¹(x1 − X0w_sc)`. The weights may be negative, which is
reported. λ comes from one SVD of X0, with every λ in closed form, selected by
pre-period time-series CV (holding out the last pre-periods)†.

**SDID** (Arkhangelsky et al. 2021).

- Unit weights: simplex plus intercept (eliminated by time-demeaning), with ridge
  `ζ²T_pre`, where `ζ = (N_tr T_post)^{1/4}·σ̂` and σ̂ is the SD of first differences of
  control pre-outcomes.
- Time weights: the same with ζ_λ = 1e-6·σ̂.
- τ̂ is the doubly weighted DiD.
- **Variance:**
  - placebo (Alg. 4; single treated unit): controls are reassigned, and the batched
    FISTA uses a mask and a shared Gram;
  - jackknife (Alg. 3; ≥ 2 treated, fixed weights);
  - bootstrap (Alg. 2; units resampled with per-replicate streams).
- `solver="synthdid"` replicates R's Frank–Wolfe: `max.iter=1e4`,
  `min.decrease=1e-5·σ̂`, and `sparsify` (zero weights ≤ max/4, renormalise, re-run)†.
  It exists for parity only.

**Placebos.**

- In-space: every donor becomes "treated" against the rest, in one batch. Report the
  post/pre RMSPE ratio and the permutation p-value = rank/(J+1). An optional exclusion
  rule drops placebos whose pre-RMSPE exceeds m× the treated unit's.
- In-time: a fake date T0' < T0, fitted on data before T0' **only**.

### 6.9 Conformal counterfactual inference (`_conformal.py`, CWZ 2021)

- Under H0 (θ_t = θ0 on the post period), refit the counterfactual model on **all** T
  periods with `y1,post − θ0` and take the residuals û.
- The statistic is `S_q = (T*)^{-1/2}‖û_post‖_q`, with q=1 as the default.
- The p-value is the share of permutations (moving block: all T cyclic shifts, exact;
  or iid: B seeded draws) whose S is ≥ the observed S, with the identity included.
- Per-period confidence sets come from test inversion over a grid (default: 200 points
  around the point estimate ± 6σ̂). Because `c(θ) = c0 − θ·Y0ᵀ1_post` is affine and the
  Gram is shared, grid refits are warm-started active sets (≈ 2 ms each).
- Any model implementing the `CounterfactualModel` protocol
  (`fit(Y, mask) -> Ŷ(0)`) plugs in: SC, SDID, DiD, IFE, MC, or sibling 7's state-space
  path.
- Finite-sample note in the docstring: the moving-block p-value cannot go below 1/T.

### 6.10 Latent-factor counterfactuals (`_factor.py`, `_mc.py`)

**IFE** (Bai 2009; Gobillon–Magnac 2016; Xu 2017):
`Y_it = X_itᵀβ + α_i + ξ_t + λ_iᵀF_t + ε` on Ω0.

- The candidates were EM (impute, then SVD) and **masked ALS**. EM slows as the masked
  fraction grows. ALS converged in 16 iterations at a 30% mask (§1.5). **Choice: masked
  ALS.**
- Per iteration:
  - loadings: batched r×r normal equations, with Grams formed as one GEMM
    `W @ (F⊙F).reshape(T, r²)`;
  - factors: likewise;
  - β: pooled FWL on `Y − λF`;
  - FE: the §6.2 solver.
- Initialisation is the rSVD of the complete control-only block (seeded,
  sign-fixed). Stop when the relative change in fitted values is < 1e-10.
- Rank: `n_factors="cv"` (Xu 2017 leave-one-pre-period-out MSPE on treated units,
  r = 0..8, warm-started) or `"bai_ng"` (the existing `bai_ng` on the balanced control
  block).
- Identification: a treated unit needs ≥ r + 1 (+FE) pre-periods, else it is flagged
  unidentified.
- Inference: parametric bootstrap (Xu 2017 Alg. 2, pseudo-treated controls,
  warm-started, capped); `pesaran_cd` on residuals as a diagnostic.

**MC-NNM** (Athey et al. 2021):

- Minimise `‖P_Ω0(Y − L − α1ᵀ − 1ξᵀ)‖²/|Ω0| + λ‖L‖_*` by soft-impute (Mazumder,
  Hastie & Tibshirani 2010).
- Unpenalised FE are updated exactly on the completed matrix (row and column means).
- The λ path is `λ_k = λ_max·0.75^k` with k < 20, warm-started.
- SVT uses a thin SVD, or `randomized_svd` with **certified truncation** (grow k until
  σ_{k+1} < λ) while rank < 0.1·min(N,T).
- CV is K-fold over held-out observed control cells: `"random_cells"` (the paper) or
  `"last_periods"` (time-aware).
- Stop when the relative Frobenius change is < 1e-8.

**PIT caveat** (in both docstrings and as `timing="ex_post"`):

- F_t is estimated from control units at every t, and the rank and λ are chosen on the
  full sample. **Both methods are two-sided in time and length-dependent.**
- They are valid ex post and invalid inside a backtest.
- The **as-of variant** `factor_counterfactual_asof` freezes the rank, β, α, the
  control loadings Λ_c and the treated loadings at T0 using data ≤ T0 only. For each
  t > T0 it sets `F̂_t = argmin‖Y_c,t − α_c − Λ_c F_t‖` (cross-sectional OLS on the
  controls at t) and `Ŷ_it(0) = α_i + λ_iᵀF̂_t`. This is prefix-invariant by
  construction.
- `refit="expanding"` offers a costlier alternative that refits at every date.

### 6.11 Finance event studies (`_events.py`, `_event_tests.py`, `_calendar_time.py`)

**Alignment.**

- The trading calendar is the sorted unique times of `returns`, or an explicit
  calendar.
- Day 0 is the first session at which the event is **known**. With `knowledge_time`
  (Datetime) and `close_cutoff`, an after-close announcement moves day 0 to the next
  session. Mapping uses `core.asof` semantics, so a Datetime is never truncated to a
  Date.
- When only a Date is supplied, the result records `event_time_resolution="date"` and
  warns if `event_window[0] > -1`.

**Windows (trap T10).**

- The event window is `[τ1, τ2]` (required).
- The estimation window is `[τ1 − gap − L, τ1 − gap − 1]`, which **ends strictly before
  the event window, with a gap**.
- `min_estimation_obs` applies. `exclude_other_events=True` masks the same firm's other
  event windows out of its estimation window.

**Computation.**

- Returns are stored CSR-like: entity offsets plus positions on the calendar.
- Blocks of `(n_events_chunk × L)` are gathered; the factor rows are gathered once.
- Batched normal equations `XᵀX`, `Xᵀy` are `matmul` over `(n, L, k+1)`, then a
  batched solve. Windows with `cond(XᵀX) > 1e10` fall back to batched QR.
- Outputs:
  - ARs over the event window and CAR;
  - the prediction-error factor `C_it = 1 + x_tᵀ(XᵀX)⁻¹x_t` as a batched quadratic form;
  - BHAR = `Π(1+R) − Π(1+R̂)`.
- The chunk size comes from `max_bytes`. Measured: 50k events in 0.37 s.
- **Prefix sums were considered and rejected**: `XᵀX` windows as cumulative differences
  are O(1) per event but break under missing returns and lose precision to cancellation.
  The gather is exact and already fast.

**Tests.**

- **Patell** (1976) with Mikkelson–Partch multi-day standardisation.
- **BMP** (1991).
- **KP-adjusted BMP/Patell** (2010): `t·√((1−r̄)/(1+(n−1)r̄))`. r̄ is computed in
  **one pass**: `[Σ_d (Σ_i z_id)² − Σ_d Σ_i z_id²] / Σ_d n_d(n_d−1)` over
  estimation-window standardised residuals, bincounted on calendar dates, so it is
  O(total rows), not O(n²) pairs.
- `cluster_adjust="auto"` applies KP when event windows overlap in calendar time.
  Partial overlap follows Kolari–Pape–Pynnönen (2018)†.
- **Corrado** (1989) and **GRANK** (Kolari–Pynnönen 2011) for CAR windows: ranks over
  the combined estimation and event window by one batched `argsort` + `put_along_axis`
  scatter, never `argsort(argsort)`.
- **Generalised sign** (Cowan 1992).
- **BHAR:** Lyon–Barber–Tsai (1999) skewness-adjusted t, plus a seeded bootstrap.
- **Calendar-time portfolios** (Jaffe 1974; Fama 1998; Mitchell–Stafford 2000):
  membership is `(event, relative day) → calendar index`. EW/VW portfolio returns come
  from `bincount` on dates, followed by a time-series regression on factors. The
  intercept α gets a Newey–West SE via `econ._common.newey_west_lrv`. This is the
  robust choice for partial clustering and long horizons.

### 6.12 Numerical accuracy targets

| Quantity | Target | Checked against |
|---|---|---|
| FE normal-equation residual (§6.2) | < 1e-12 relative after refinement | direct evaluation; `econ._hdfe.demean` at tol 1e-14 |
| BJS point estimates | 1e-8 relative | `pyfixest` did2s (numerically equal to imputation, Gardner 2022) |
| CS ATT(g,t), unconditional | 1e-12 vs brute-force per-cell 2×2; 4 d.p. vs R `did` on `mpdta` | stored reference (§8.3) |
| SA | 1e-10 | dense `econ.hdfe` saturated regression |
| Bacon | Σ s·β = TWFE β to 1e-10 | `econ.hdfe` |
| SC / SDID weights | KKT < 1e-10; objective ≤ SLSQP + 1e-12 | `scipy.optimize` (importorskip) |
| California Prop 99 | DID **−27.34911** exactly (1e-8); SC −19.61966 and SDID −15.60383 to 1e-4 in `solver="synthdid"` mode | synthdid vignette (§8.3) |
| Event statistics | 1e-12 | hand-computed small cases |

---

## 7. Leak-safety design and traps

These are mostly **ex-post estimators**. Honesty means that every fitted weight uses
only the data its identification argument allows, and that every as-of variant is
prefix-invariant. Each trap below has a safe default and a test.

| # | Trap | Safe default | Test |
|---|---|---|---|
| T1 | Static TWFE under staggered timing (§1.1) | not offered as an estimator. `bacon_decomposition` and `twfe_weights` exist to **diagnose** a user's TWFE | `test_counterfactual_twfe_bias.py` asserts the bias. It must never be "fixed" by loosening |
| T2 | Imputing unidentified cells with a zero FE (§1.2) | fail closed (inv. 6) | the measured +8.9% case must produce `identified=False`, not a number |
| T3 | Silent anticipation contaminating the base period | `anticipation` is a **required** keyword: `anticipation=0` is one explicit token. The pre-trend test reports leads | a planted δ=2 must bias the δ=0 fit and not the δ=2 fit |
| T4 | "Never-treated" is a statement about the future: a unit untreated through s but treated after s changes group when the sample grows | ex post, `control="not_yet_treated"` is the default. `cs_att_asof` **refuses** `never_treated` | `assert_prefix_invariant` must **fail** for never-treated (proving the trap is real) and **pass** for not-yet-treated |
| T5 | Post-treatment covariates as controls ("bad controls") | `covariate_timing="base"`: X is taken at each unit's base period (g−1−δ) and held fixed. Time-varying X needs `covariate_timing="contemporaneous"` plus a warning | a covariate affected by treatment must bias the contemporaneous fit and not the base fit |
| T6 | SC / SDID weights or donor screens computed on post-period data (e.g. selecting donors by full-sample correlation) | weights, V, the ridge ζ, σ̂ and donor filters use `t < T0 − gap` only | perturb every post-T0 outcome, and the weights must be bit-identical (`assert_no_lookahead` on the weight table) |
| T7 | Contaminated donors: a donor that adopts treatment inside the post window | donors are filtered **as-of** each post date. A donor treated at s is dropped from the pool for its own placebo and reported | the synth panel with late-adopting donors |
| T8 | Treatment timing taken from a revised or restated source | `cohort` may be supplied as a vintage table and joined via `core.asof.asof_join`. `knowledge_time` wins over `event_time` | revised adoption dates in `synth` vintages |
| T9 | In-time placebo fitted with data after the fake date | the fit window ends at T0' − gap | perturbation test at T0' |
| T10 | Event-study estimation window overlapping the event window or other events | the estimation window ends at `τ1 − gap − 1`, with `gap ≥ 1` enforced (default 10) and `exclude_other_events=True` | perturb event-window returns, and β̂, σ̂ must be unchanged |
| T11 | After-close announcements attributed to the same day | `knowledge_time` + `close_cutoff`; date-only inputs are flagged | a planted after-close effect lands on day +1 |
| T12 | Survivorship: requiring complete post-event data drops delisted firms | incomplete windows are kept and flagged, and a user-supplied delisting return column (Shumway 1997) is honoured | planted exits in `synth` |
| T13 | IFE/MC used in a backtest: two-sided F_t, full-sample rank and λ | `timing="ex_post"`, never registered as a feature. `factor_counterfactual_asof` freezes everything at T0 | prefix invariance of the as-of variant; perturbation shows the ex-post one **does** move |
| T14 | CV folds leaking time within MC-NNM | `cv="last_periods"` available; the docstring states that `"random_cells"` is ex-post only | — |
| T15 | Reusing an i.i.d. null under event-date clustering (§1.3) | `cluster_adjust="auto"` | KP size table as an assertion |

**As-of monitors** (`_asof.py`): outputs are long frames keyed by `(entity|event_id,
time)`. Each is prefix-invariant and passes both `assert_no_lookahead` and
`assert_prefix_invariant`, which catch different defects (the `testing.py` docstring).

- `sc_gap_asof`: weights frozen at T0, `gap_t = y1_t − Y0,t·w`.
- `event_car_asof`: CAR through t with frozen β.
- `factor_counterfactual_asof`: §6.10.
- `cs_att_asof`: the estimate at date s uses cells with t ≤ s, not-yet-treated controls
  known at s, and base periods < t.

---

## 8. Tests (`tests/test_counterfactual_*.py`)

`--strict-markers` is on. Only the existing `slow` and `benchmark` markers are used.

### 8.1 The synth extension (M0, `synth/`, additive, byte-identical by default)

**New dials**, each neutral by default:

- `treatment_share = 0.0`;
- `adoption_times: tuple[int, ...] | None`;
- `adoption_rate` (per-step hazard);
- `anticipation_steps = 0`;
- `effect_level`, `effect_slope` (per step since adoption), `effect_duration: int | None`
  (None means permanent; a finite value gives an event-study pulse);
- `effect_cohort_gradient` (multiplies by the adoption *step*, never by a fraction of T);
- `effect_heterogeneity` (entity log-normal SD);
- `selection_on_loadings` (the logit of adoption loads on `loadings[:, 0]`: this
  violates parallel trends while IFE/SC still identify);
- `differential_trend` (a linear trend for treated units: the HonestDiD case).

**Streams.** All treatment randomness comes from **one new entity stream
`_ENTITY_TREATMENT = 25`** (u_treated, u_cohort, z_heterogeneity). No existing
component ID changes, so `test_synth_generate.py` hashes stay byte-identical when
`treatment_share = 0`. A new test asserts exactly that.

**Ground truth.** `GroundTruth.entities` gains `adoption_step`. `latent` gains
`treated` and `effect`. `GroundTruth.att(by="overall"|"event_time"|"cohort"|"calendar",
cells=...)` returns the true estimand over the observed, identified cells, which is
what every recovery test compares against.

**Prefix consistency** is preserved: adoption is a per-entity draw that never depends
on T. Sibling plans extending `synth` must use other component IDs (§13).

### 8.2 Test files

| File | Asserts |
|---|---|
| `_twfe_bias` | reproduces the §1.1 table: TWFE bias > 40% in both designs; BJS/CS/SA within 3 SE of `truth.att()`; the Bacon identity holds |
| `_fe` | Schur vs `demean` (tol 1e-14) ≤ 1e-11 on random masks, disconnected components (nullity = #components), weights, p == batch solves |
| `_identification` | the §1.2 case gives `identified=False` for exactly the 30,000 cells; no aggregate includes them |
| `_imputation`, `_cs`, `_sa` | recovery of planted heterogeneous and dynamic effects (`effect_slope`, `effect_cohort_gradient`); never vs not-yet; varying vs universal base; DR = OR = IPW without covariates; DR consistency when either nuisance is misspecified; balance vs pairwise |
| `_coverage` (**slow**) | 500 seeded panels: 95% CI coverage ∈ [0.92, 0.98] for BJS, CS and SDID (several treated); sup-t band coverage ∈ [0.92, 0.98]; BJS conservative variance coverage ≥ 0.93 |
| `_placebo_size` (**slow**) | SC in-space permutation p-values uniform under a planted null (5% size ∈ [0.02, 0.09], J=40, 500 reps); CWZ size ≤ α + 0.02; randomization inference exact under the null |
| `_simplex` | active set vs Duchi-projected FISTA vs SLSQP (importorskip): KKT < 1e-10, objective parity; Michelot = Duchi to 1e-15; masked projection; non-uniqueness flag; the numba kernel (importorskip) matches numpy to 1e-12 |
| `_sc`, `_sdid` | California: DID −27.34911 (exact); SC −19.61966 / SDID −15.60383 in `solver="synthdid"` mode to 1e-4; the exact solver's objective ≤ FW's; ASCM reduces pre-fit imbalance; SDID with a single treated unit refuses the jackknife |
| `_conformal` | CWZ on planted effects: the confidence set covers the truth at ≥ 1 − α − 0.02 (slow); moving-block exactness (T permutations) |
| `_factor`, `_mc` | IFE/MC recover planted `selection_on_loadings` effects where DiD is biased (asserted: DiD bias > 3 SE, IFE within 3 SE); rank selection recovers `n_factors`; certified-SVT parity with the thin SVD |
| `_events` | hand-computed AR/CAR/BHAR/Patell/BMP/Corrado/sign cases to 1e-12; the §1.3 KP size table (slow); calendar-time α equals a hand OLS |
| `_leakage` | T3–T13: perturbation of the post period leaves weights, β̂ and FE bit-identical; `assert_no_lookahead` **and** `assert_prefix_invariant` pass on every `_asof` function and **fail** on the deliberately leaky variants (never-treated as-of, ex-post IFE) |
| `_oracle` | `pyfixest` (did2s, `event_study` saturated, feols TWFE), `csdid` python (ATT(g,t) with analytic SEs), `scipy.optimize`, all `importorskip`; stored R `did` / `synthdid` numbers from `tests/data/counterfactual_reference.json` |
| `_perf` (**benchmark**) | §9 budgets at 3× slack |

Plus the standing guardrails: `test_import_hygiene.py` (no top-level optional
imports; cold-import budget), `test_dependency_drift.py` (**zero** new mandatory
dependencies), `test_wheel_guardrails.py`, `test_registry_conformance.py` (only
`_asof` specs are registered; the orchestrator adds their harness).

### 8.3 Stored reference values (numbers only, never third-party code)

- **R `did`, `mpdta`**, never-treated control, varying base, no covariates, from the
  package README:
  - ATT(2004, 2004…2007) = −0.0105, −0.0704, −0.1373, −0.1008;
  - ATT(2006, 2004…2007) = 0.0065, −0.0028, −0.0046, −0.0412;
  - ATT(2007, 2004…2007) = 0.0305, −0.0027, −0.0311, −0.0261;
  - dynamic e = −3…3: 0.0305, −0.0006, −0.0245, −0.0199, −0.0510, −0.1373, −0.1008;
  - dynamic overall −0.0772.
  - SEs in the README are bootstrap SEs, so they are compared only in distribution.
  - `mpdta` ships with GPL R `did`, so it is **not vendored**. The test loads it from
    python `csdid` if installed (licence†), else it skips.
- **`synthdid` `california_prop99`** (39 states × 31 years; N0 = 38, T0 = 19): the repo is
  BSD-3, so the CSV **may be vendored** under `tests/data/` with NOTICE attribution
  (the data's ADH 2010 provenance†). Values: DID −27.34911, SC −19.61966, SDID −15.60383.
- **Implementation defaults differ across ports.** Never-treated vs not-yet-treated and
  varying vs universal base differ between R `did`, Stata `csdid` and python `csdid`
  (per the Stata `csdid` page). Every oracle test pins every setting explicitly.

---

## 9. Benchmarks and performance budgets

`benchmarks/bench_counterfactual.py` prints a table against naive baselines: the
per-cell loop, AP `demean`, SLSQP, per-event Python loops. `tests/test_counterfactual_perf.py`
(marked `benchmark`) asserts the budgets at 3× slack. The reference is the M5 Pro with
Accelerate, single process, BLAS threads at default.

| Workload | Budget | Peak memory |
|---|---|---|
| BJS: N=10k, T=200, 20 cohorts, 3 covariates, 40 horizons + conservative SE + pre-trend test | **< 0.5 s** (FE solve measured 26 ms) | < 250 MB |
| CS unconditional: all 4,000 cells + analytic SE + 4 aggregations | **< 0.2 s** | < 100 MB |
| CS multiplier bootstrap B=999: event-time bands (J≈400) / all cells (J=4000, chunked) | **< 0.3 s / < 1.5 s** | < 150 MB (chunks of 512) |
| CS DR, p=5, not-yet-treated (≤ 420 propensity/OR problems) | **< 2 s** | < 300 MB |
| Bacon + dCDH weights | **< 0.1 s** | — |
| SC fit, J=500, T0=100 | **< 5 ms** (1.7 ms measured) | < 10 MB |
| 1,000 SC refits (500 in-space + in-time + CWZ grid) | **< 3 s** numpy / **< 1 s** with `fast` | < 50 MB |
| SDID, J=500, T_pre=100, placebo B=200 (batched FISTA) | **< 1.5 s** | < 100 MB |
| IFE ALS rank ≤ 8 + CV r=0..8, 10k×200 | **< 10 s** (**slow**) | < 400 MB |
| MC-NNM 20-λ path + 5-fold CV, 10k×200 | **< 60 s** (**slow**) | < 400 MB |
| Event study: 50k events, L=250, 4 factors, window 26 + all tests | **< 1.5 s** (0.37 s measured for ARs) | ≤ `max_bytes` (default 256 MB chunks) |
| Calendar-time portfolios: 50k events, 5,000 dates | **< 0.5 s** | < 100 MB |

Parallelism comes from vectorisation and BLAS threading across cohorts, cells, events,
placebos and bootstrap draws. There is no multiprocessing. The single numba kernel
(§6.7) is the only JIT.

---

## 10. Dependencies

- **Mandatory: none added** (numpy + polars).
- **Optional:** `fast` (numba), for one kernel. Update the `pyproject.toml` comment on
  `fast` ("CUSUM kernel") in M8. `scipy` is used only for `v="nested"`'s optional
  minimiser, via `_deps.require("scipy", feature="nested V")`, and a pure-numpy
  coordinate search is the default.
- **Test-only oracles:** `pyfixest` (MIT†), `csdid` (licence†), `scipy`, all through
  `pytest.importorskip`. R, `did`, `synthdid`, `fixest` and `HonestDiD` supply **stored
  numbers only**. R is not installed on the reference machine, and GPL code is never
  read or copied. Everything is clean-room from the papers. Specs record `source` and
  `license="Apache-2.0"`.

---

## 11. Milestones

| M | Contents | Ships | Caller gate |
|---|---|---|---|
| **M0** | synth treatment dials (§8.1); `_design.py`; `_fe.py`; `test_counterfactual_twfe_bias.py` + `_fe` + `_identification` | the fixture, the kernel, and the bug proven. Zero new deps | internal |
| **M1** | BJS imputation (+ variance, pre-trend test); CS unconditional (never/not-yet, varying/universal, 4 aggregations, IF, multiplier bootstrap, sup-t bands); SA; Bacon + dCDH weights; `_results.py`; oracle + coverage tests | honest staggered DiD at BLAS speed | **§14 caller required before public export** |
| **M2** | CS covariates (OR/IPW/DR-IPT, batched per (g, C)); pairwise unbalanced; clustering and sampling weights; randomization inference; `cs_att_asof` | the full CS surface; small-N inference for the caller | §14 |
| **M3** | `_simplex.py` (active set, FISTA-restart, Michelot, KKT, synthdid-FW); SC (+V options, penalised), ASCM, SDID (+placebo/jackknife/bootstrap); space/time placebos; `sc_gap_asof`; `_fast.py`; California parity | the SC family | §14 (single-client intervention) |
| **M4** | CWZ conformal over the `CounterfactualModel` protocol | exact finite-sample permutation inference | with M3 |
| **M5** | IFE ALS + rank selection; MC-NNM + λ path + CV; `factor_counterfactual_asof` | counterfactuals robust to selection on loadings | **no caller yet. Build only when one is named** |
| **M6** | event studies: alignment, gather + batched OLS, AR/CAR/BHAR, Patell/BMP/KP/GRANK/sign/skew-t, calendar-time; `event_car_asof` | leak-safe event studies that do not over-reject | §14 (reference answers) |
| **M7** (optional) | dCDH `DID_M` (non-absorbing); HonestDiD RM/SD bounds (tiny LPs by a pure-numpy dense simplex, with scipy `linprog` optional); spillover-aware DiD consuming sibling 6's exposure mappings | robustness layer | only on demand |
| **M8** | `counterfactual/__init__.py` exports; `pn.causal(method="did"/"sc"/"sdid")`; registry specs for `_asof`; docs page; CHANGELOG; mkdocs nav | public surface | **only with the engine test green** |

M0 + M1 alone is shippable and is what the caller needs. Do not start M3 before
`_coverage` is green for M1.

---

## 12. Risks and open questions (resolve with a measurement)

1. **Schur at large T.** At daily T = 5,000 with N ≥ T, `eigh` on 5000² costs ~0.3–0.5
   s, which is fine. At T = 20,000 eliminate onto units if N is smaller, else fall back
   to AP. Measure the crossover.
2. **AP acceleration** for ≥3 FE dimensions. Irons–Tuck needs an opt-in
   `accelerate=` keyword on `econ._hdfe.demean`, which is a change to econ's file.
   Default behaviour must stay bit-identical and `test_econ_hdfe.py` must stay green.
   Propose it only if a weakly connected mask benchmark shows > 100 sweeps.
3. **SA ≡ CS** on unbalanced panels and with last-treated control: confirm numerically
   before relying on it (†).
4. **IPT non-convergence and propensities near 1.** Match DRDID's trimming and report
   it. Should IPT fall back to logit MLE automatically, or raise?
5. **SC non-uniqueness** changes post-period counterfactuals. Is the vertex tie-break
   enough, or should `penalty` default to a tiny Abadie–L'Hour λ? Decide from how often
   `nonunique` fires on California and the synth panels.
6. **synthdid parity** depends on reproducing FW + sparsify exactly. If 1e-4 cannot be
   reached, document the gap. The exact solver remains the default.
7. **KP under partial overlap.** Validate the bincount r̄ estimator against
   Kolari–Pape–Pynnönen (2018)† on synth events with staggered dates.
8. **Memory of the all-cells IF** at N = 100k. The chunking policy must keep peak memory
   ≤ `default_max_bytes()`.
9. **Small-N honesty for the caller.** With ~20 clients, are randomization inference and
   CS analytic SEs adequate, or is a wild cluster bootstrap (Cameron–Gelbach–Miller
   2008) needed? Measure coverage at N ∈ {10, 20, 40} in `_coverage`.
10. **polars 1.35.** All heavy lifting is numpy. Polars use is limited to
    `group_by/agg/join/sort/with_columns/filter`, and `build_tensor` owns the pivot.
    CI covers 1.35 and 1.42.

---

## 13. Boundaries with sibling plans

| Sibling | Theirs | Ours | Interface |
|---|---|---|---|
| 1 forecast-evaluation-and-sharpe-inference | DM/SPA/MCS, Sharpe inference | none of it | — |
| 2 drift-monitoring-and-sequential-inference | sequential and anytime-valid tests, drift alarms | the **gap / CAR series** from `_asof` | they may run a sequential test on our prefix-invariant gap series. We ship no sequential test |
| 3 covariance-and-market-state | shrinkage and market covariance | the KP r̄ (a scalar, computed in one pass) | `event_tests` may accept their covariance estimate later; no dependency |
| 4 label-weights-and-event-sampling | event *sampling* (CUSUM filter, uniqueness weights, barriers) for ML labels | event *studies* on a given event table | their event table is a valid `events` input; no overlap in code |
| 5 ohlc-volatility-and-liquidity | vol and liquidity estimators | none | a user may pass their vol for BHAR scaling |
| 6 network-and-spatial-panel | exposure mappings, Conley HAC | spillover-aware DiD (M7) **consumes** their exposures to drop exposed controls or estimate exposure effects | we never implement exposure mappings or Conley HAC |
| 7 causal-state-space-and-regimes | BSTS / CausalImpact / Kalman counterfactuals, regimes | the SC/DiD/IFE family and CWZ inference | the `CounterfactualModel` protocol (§6.9): their path can use our placebo and conformal inference |
| 8 multiscale-complexity-features | — | — | — |
| 9 tail-risk-and-self-excitation | Hawkes / event clustering in time | — | — |

**Shared-file conflicts to resolve in the orchestrator:**

- `synth/` stream component IDs: this plan claims **25**.
- `pyproject.toml` `fast` comment.
- `_internal/_verbs.py:causal` (M8).
- `test_registry_conformance.py` harness entries.

---

## 14. Caller

> AGENTS.md: **"No new public surface without a named caller in the engine, and a test
> in the engine that exercises it."** This plan is bound by that rule. M0–M1 internals
> may be built and tested inside panelary. **Public exports (M8) wait for the engine
> test.**

**Primary caller: staggered re-test effect measurement.**

- AgenticFinance's journey is Assess → Diagnose → Recommend → Improve → Re-test.
  Clients adopt a remediation at different dates. Legitimate remediations include
  changing the model, using deterministic code, adding human review, or not automating.
  Each client is re-assessed on **newly generated sealed test sets**, never a replay.
- "Did the remediation reduce the silent-error rate, and by how much?" is a staggered
  adoption design. The panel is client × assessment round, the outcome is the error
  rate, and not-yet-treated clients are the controls.
- Static TWFE is exactly the estimator §1.1 shows to be wrong here. The caller needs M1
  plus M2's small-N randomization inference.
- A single client's intervention, with the other clients as donors, needs M3 (SDID/SC).

**Where it lives.**

- In `truepoint/src/truepoint/quant/`, the caller location AGENTS.md names. It does not
  exist yet.
- **Not** in `diagnose/`: truepoint's `_firewall.py` forbids `diagnose/` from importing
  `panelary`, because cause assignment must not see what we sell.
- Results enter reports through `to_json()` with `produced_by`.
- The estimator does not choose the report label. Whether an affiliated service
  materially helped ("Performance Evaluation" vs "Independent Assurance", STRATEGY) is
  the engine's call, recorded next to the estimate.

**Secondary caller (M6).** Deterministic reference values for event-type assessment
items: an abnormal return or CAR computed point-in-time with an audited estimation
window. This is the kind of number an assessed AI is often silently wrong about. No
sealed item appears in any panelary test, fixture or doc.

**M5 has no named caller.** IFE/MC stays unbuilt until one exists.

---

## 15. References

(† = not verified against the source during this session.)

- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *ReStud* 72(1), 1–19.
- Abadie, A. (2021). Using synthetic controls. *JEL* 59(2), 391–425.
- Abadie, A., Diamond, A. & Hainmueller, J. (2010). *JASA* 105(490), 493–505; (2015) *AJPS* 59(2), 495–510.
- Abadie, A. & Gardeazabal, J. (2003). *AER* 93(1), 113–132.
- Abadie, A. & L'Hour, J. (2021). A penalized synthetic control estimator. *JASA* 116(536), 1817–1834.
- Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G. & Wager, S. (2021). Synthetic difference-in-differences. *AER* 111(12), 4088–4118. R `synthdid` (BSD-3), vignette values: DID −27.34911, SC −19.61966, SDID −15.60383.
- Athey, S., Bayati, M., Doudchenko, N., Imbens, G. & Khosravi, K. (2021). Matrix completion methods for causal panel data models. *JASA* 116(536), 1716–1730.
- Bai, J. (2009). Panel data models with interactive fixed effects. *Econometrica* 77(4), 1229–1279. Bai, J. & Ng, S. (2002). *Econometrica* 70(1), 191–221.
- Baker, A., Larcker, D. & Wang, C. (2022). How much should we trust staggered difference-in-differences estimates? *JFE* 144(2), 370–395.
- Barber, B. & Lyon, J. (1997). *JFE* 43(3), 341–372. Lyon, J., Barber, B. & Tsai, C.-L. (1999). *JF* 54(1), 165–201.
- Beck, A. & Teboulle, M. (2009). FISTA. *SIAM J. Imaging Sci.* 2(1), 183–202.
- Ben-Michael, E., Feller, A. & Rothstein, J. (2021). The augmented synthetic control method. *JASA* 116(536), 1789–1803.
- Boehmer, E., Musumeci, J. & Poulsen, A. (1991). *JFE* 30(2), 253–272.
- Borusyak, K., Jaravel, X. & Spiess, J. (2024). Revisiting event-study designs: robust and efficient estimation. *ReStud* 91(6), 3253–3285, doi:10.1093/restud/rdae007.
- Callaway, B. & Sant'Anna, P. (2021). Difference-in-differences with multiple time periods. *JoE* 225(2), 200–230. R `did` (`mboot`: Rademacher multipliers, IQR-based bSigma, sup-t bands). Stata `csdid` 2.0 (10–308× speedup from shared structure).
- Cameron, A. C., Gelbach, J. & Miller, D. (2008). Bootstrap-based improvements for inference with clustered errors. *ReStat* 90(3), 414–427.
- Chernozhukov, V., Wüthrich, K. & Zhu, Y. (2021). An exact and robust conformal inference method for counterfactual and synthetic controls. *JASA* 116(536), 1849–1864.
- Condat, L. (2016). Fast projection onto the simplex and the ℓ1 ball. *Math. Programming* 158, 575–585. Duchi, J. et al. (2008), *ICML*. Michelot, C. (1986), *JOTA* 50, 195–200.
- Corrado, C. (1989). *JFE* 23(2), 385–395. Corrado, C. & Zivney, T. (1992). *JFQA* 27(3), 465–478.
- Cowan, A. (1992). Nonparametric event study tests. *RQFA* 2, 343–358†.
- de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *AER* 110(9), 2964–2996 (σ_fe formula†).
- Doudchenko, N. & Imbens, G. (2016). NBER WP 22791.
- Fama, E. (1998). *JFE* 49(3), 283–306. Mitchell, M. & Stafford, E. (2000). *JB* 73(3), 287–329.
- Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.
- Gobillon, L. & Magnac, T. (2016). *ReStat* 98(3), 535–551.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *JoE* 225(2), 254–277.
- Graham, B., Pinto, C. & Egel, D. (2012). Inverse probability tilting. *ReStud* 79(3), 1053–1079.
- Heckman, J., Ichimura, H. & Todd, P. (1997). *ReStud* 64(4), 605–654.
- Kaul, A., Klößner, S., Pfeifer, G. & Schieler, M. (2022). Standard synthetic control methods: the case of using all preintervention outcomes together with covariates. *JBES*†.
- Klößner, S., Kaul, A., Pfeifer, G. & Schieler, M. (2018). *Swiss J. Econ. Stat.* 154†.
- Kolari, J. & Pynnönen, S. (2010). Event study testing with cross-sectional correlation of abnormal returns. *RFS* 23(11), 3996–4025. (2011) Nonparametric rank tests for event studies. *J. Empirical Finance* 18(5), 953–971.
- Kolari, J., Pape, B. & Pynnönen, S. (2018). Event study testing with cross-sectional correlation due to partially overlapping event windows. SSRN 3167271 (publication venue†).
- Liu, L., Wang, Y. & Xu, Y. (2024). A practical guide to counterfactual estimators. *AJPS* 68(1), 160–176.
- Mazumder, R., Hastie, T. & Tibshirani, R. (2010). Spectral regularization algorithms. *JMLR* 11, 2287–2322.
- O'Donoghue, B. & Candès, E. (2015). Adaptive restart for accelerated gradient schemes. *FoCM* 15(3), 715–732.
- Patell, J. (1976). *JAR* 14(2), 246–276.
- Rambachan, A. & Roth, J. (2023). A more credible approach to parallel trends. *ReStud* 90(5), 2555–2591.
- Roth, J., Sant'Anna, P., Bilinski, A. & Poe, J. (2023). What's trending in difference-in-differences? *JoE* 235(2), 2218–2244 (SA ≡ CS equivalence statement†).
- Sant'Anna, P. & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *JoE* 219(1), 101–122.
- Shumway, T. (1997). The delisting bias in CRSP data. *JF* 52(1), 327–340.
- Sun, L. & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *JoE* 225(2), 175–199.
- Xu, Y. (2017). Generalized synthetic control method. *Political Analysis* 25(1), 57–76.
- `pyfixest` (did2s, lpdid, `event_study` saturated; licence MIT†); python `csdid` (d2cml-ai; bundles `mpdta`; licence†).
