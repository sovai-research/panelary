# `panelary/statespace/` — build contract: causal state-space filtering, IIR filters, filtered regimes and changepoints

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.**

This plan covers five capabilities, and for each it states which outputs are point-in-time
(PIT) and which are not:
- causal Kalman filtering across a whole panel at once;
- a causal IIR filter bank;
- regime models whose features are **filtered** probabilities;
- Bayesian online changepoint detection;
- changepoint segmentation.

**Dependencies.** The import path is pure **numpy + polars**.
- `numba` is optional, via the existing `fast` extra. It is imported lazily through
  `_internal._deps.have/require` and must reproduce the numpy fallback's output (§5.0).
- `scipy` is **never** imported, because every design here is closed-form.
- Oracles (`statsmodels`, `hmmlearn`, `filterpy`, `ruptures`, `jumpmodels`,
  `scipy.signal`) are **test-only**, behind `pytest.importorskip`.

**Reuse, do not reinvent:**
- `detect`: `focus`, `page_cusum`, `shiryaev_roberts`, `volatility_rescale`,
  `spot_variance`, and `mc_table` as the calibration pattern.
- `econ._common`: `pinv_sym`, `ols`, `chi2_sf`, `winsorize`.
- `core._calendar`: `validate_duration`, `shift_forward`, `BusinessDays` (refit lags).
- `core.asof.asof_join`, for knowledge-time panels.
- `testing`: `assert_no_lookahead`, `assert_prefix_invariant`.
- `registry`: `FeatureSpec`, `register_feature`.
- `synth`: `SynthConfig`, `generate_panel`, `GroundTruth`.
- The `SeedSequence(seed, spawn_key=…)` idiom from `synth/_generate.py::_stream`. Copy it;
  do not import synth privates.
- The lazy-numba idiom from `feature_extractors/_kernels.py::_get_cusum_numba`, keeping
  `error_model="numpy"`, which is load-bearing there.

---

## 1. Why this exists

### 1.1 The look-ahead that every regime tutorial ships

The standard libraries put the leaky call next to the safe one, and name it like the
default:

| Library | Leaky (uses `y[t+1:]`) | Point-in-time |
|---|---|---|
| `hmmlearn.GaussianHMM` | `predict` (Viterbi over the whole sequence), `predict_proba` (forward–backward) | none public (the forward pass is private) |
| `statsmodels` `MarkovRegression` | `smoothed_marginal_probabilities` (Kim smoother) | `filtered_marginal_probabilities` |
| `statsmodels` `UnobservedComponents` | `smoothed_state`, `level.smoothed` | `filtered_state`, `level.filtered` |
| `jumpmodels` (Shu, Apache-2.0) | `.predict()`, `.predict_proba()` | `.predict_online()`, `.predict_proba_online()` |
| `scipy.signal` | `filtfilt`, `sosfiltfilt`, centred `savgol_filter` | `lfilter`, `sosfilt` |

The `jumpmodels` example (verified, `examples/nasdaq/example.py`) says it outright:
*"the `.predict()` method performs state decoding using all test data (i.e., from 2022 to
2024) at once"*. As a result it *"exits the bear signal in 2023 slightly earlier than with
online inference"*. That earlier exit is look-ahead, presented as performance.

**Measured here.** Setup: one entity, a 2-state Gaussian HMM with the true parameters,
T = 5000.
- The full-sample smoothed label differs from the as-of smoothed label on **11.1%** of
  dates (sampled every 50 steps from t = 500).
- Accuracy is **94.2% smoothed versus 90.7% filtered**, so the 3.5-point gap is pure
  look-ahead.
- The filtered probabilities are bitwise prefix-invariant.

**The second trap is the parameters.** Filtered probabilities computed with θ fitted on
the full sample still leak, through θ. Shu, Yu & Mulvey (2024) (verified, arXiv
2402.05272) avoid this, and their scheme is this plan's default contract (§6):
- refit every six months on a 3000-day window;
- choose the jump penalty monthly by time-series CV;
- take the regime at t as the last state of a DP over a lookback ending at t;
- trade with a one-day delay.

### 1.2 Measured facts that dictate the design

Measured 2026-09-29 on this machine: arm64 Apple Silicon, numpy 2.5.3, polars 1.44.2,
numba 0.67.0, scipy 1.18.1. Scripts in the session scratchpad. M6 re-measures them as
committed benchmarks.

| Fact | Measurement | Consequence |
|---|---|---|
| Local-level Kalman filter, time loop over a padded `(T,N)` matrix, all entities as `(N,)` arrays, 5% missing | T=N=5000 (25M cells): **0.16 s** numpy (31 µs/step) vs **0.14 s** numba; outputs **bitwise equal** | The canonical pattern (§5.0) is T numpy steps, not N·T Python steps. numba is not needed when N is large. It pays off when N is small and T is long. |
| Steady-state local level (q=0.01) vs `pl.Expr.ewm_mean(alpha=K̄, adjust=False)` | abs diff 2.7e-2 at t=10, 5.2e-7 at t=100, 2.4e-11 at t=200, **0.0** at t=500 | The converged filter *is* an EWMA (Muth 1960). Once P reaches a bitwise fixed point, a gain shortcut is exact (§5.1.4). |
| 2-state Hamilton filter, log-shifted scaling | T=N=5000: **1.09 s** numpy (218 µs/step) | Regime probabilities for a full panel cost about one second. |
| TVP regression k=3, Joseph update, batched `@` | T=2000, N=5000: **2.33 s** (1.16 ms/step). Symmetric rank-1 form: 1.16 s | Joseph costs 2×. Chunking across entities is **bitwise invariant** (measured). |
| Same, approximate-diffuse start P₀ = 10⁶·I | Joseph and rank-1 disagree by **1.2e-7** in β | Big-κ starts lose about 7 digits. Use an **exact diffuse** start (§5.1.3). |
| BOCPD, NIG model, top-R pruning (capacity 2R, amortised) | R=100, T=2000, N=1000: **6.8 s** numpy (34 ns/hypothesis-step), **4.8 s** numba (16 ns) | O(T·R), about 1.4× from numba. The cost is transcendentals, not interpretation. |
| BOCPD in the Adams–MacKay convention (x_t scored under the r_{t−1} predictive) | P(r_t=0 \| x_{1:t}) ∈ [0.004000, 0.0040066] with H=0.004. It is **constant**. | P(r_t=0) there is an uninformative feature. Use the convention in §5.7. |
| Butterworth order 8, cutoff 1/500 cycles/sample, `(b, a)` polynomial vs second-order sections (SOS) | `lfilter(b, a)` output reaches **2.6e22**; SOS max \|y\| = 0.14 | Only SOS cascades are allowed. Never build high-order polynomials. |
| Own bilinear + prewarp Butterworth design vs `scipy.signal.butter` | max \|ΔH(e^{iω})\| ≤ **3.8e-15**, orders 2–8, low and high | Closed-form design matches scipy, so scipy is not needed. |
| Vectorised DF2T biquad cascade vs `scipy.signal.sosfilt` | 5000×5000, order 4: **0.15 s** vs 0.12 s; parity **1.25e-14** (not bitwise) | numpy is within 25% of C. scipy is an oracle only. |
| PELT forward pass, L2 cost, T=5000 | numpy 41 ms; numba **9 ms** (3 changes) / 20 ms (no change) per entity. `last_cp[t]` is bitwise prefix-invariant. A change at 1500 is detected by t=1520 (`last_cp=1502`). | The PELT forward DP is *already* a point-in-time "last changepoint as of t" (§5.8). |
| Long → dense `(T,N)` scatter; dense → long gather | 23.8M rows, string entity: **0.81 s** and **0.15 s** | End-to-end panel cost is dominated by the pivot, not the filter. |
| `polars ewm_mean` with nulls | `ignore_nulls=False` renormalises after a gap (the gain rises to α/((1−α)^k+α)). `ignore_nulls=True` skips the gap. A null row outputs null. | The EWMA's missing-data semantics differ from the Kalman filter. This is documented and not papered over. |

### 1.3 What is missing today

`grep` over `panelary/` finds **none** of these: a Kalman filter, HMM, Baum–Welch, Hamilton
filter, jump model, BOCPD, PELT/BinSeg, Butterworth or Ehlers filter. `forecasting/` has
no state-space or exponential-smoothing model either. The gaps that matter:
- `docs/roadmap.md` §Imputation asks for a **`KalmanImputer` (filter-only, with Gaussian
  posterior variance)**.
- `detect/` monitors mean shifts in *standardised increments*, but nothing produces
  model-based standardised innovations to feed it.
- Nothing handles a variance change or unknown pre-change parameters with a posterior.
- Nothing emits regime probabilities.

---

## 2. What already exists, and what this must NOT duplicate (verified)

| Existing (file:function) | What it does | Relationship |
|---|---|---|
| `detect/_monitors.py:focus` | GLR mean-shift monitor over all magnitudes and windows, O(log n); standardised input, known pre-change mean | **Complementary.** BOCPD adds a variance change, unknown pre-change mean and variance, and a posterior. `focus` consumes our `zinnov`. Do not add another CUSUM or GLR. |
| `detect/_monitors.py:page_cusum`, `page_cusum_expr`, `shiryaev_roberts`, `hb_cusum` | Closed-form sequential monitors | Same as above. The detection study (§7) compares against them and never re-implements them. |
| `detect/_monitors.py:volatility_rescale`, `spot_variance` | One-sided spot variance | Used to standardise causally before BOCPD. Do not re-derive. |
| `detect/_bsadf.py`, `_critvals.py:mc_table` | Explosive-root tests; data-free MC calibration | The pattern for PIT-safe penalty and threshold calibration (§5.8) is copied from `mc_table`. |
| `econ/features/_common.py:rolling_beta` | Trailing-window OLS beta via polars rolling moments | TVP / RLS are the *recursive* alternative. The numbers differ by design. Do not route one through the other. |
| `econ/features/_decompose.py:causal_seasonal_decompose` | Trailing moving-average seasonal-trend decomposition | The model-based seasonal state (M7) is an alternative, not a rewrite. The trailing MA stays the default. |
| `evolve/_ops.py:_b_ts_ewm_mean`; `pl.Expr.ewm_mean` | EWMA (`adjust=False`, `ignore_nulls=False`) | **The 1-pole IIR path is polars itself.** We register no EWMA operator. We ship only `steady_state_gain(q)`, the bridge from local-level SNR to α. |
| `econ/features/_common.py:per_entity_apply`, `entity_arrays` | A per-entity Python loop over numpy kernels | **Not used for recursions.** It gives N·T Python steps, which is exactly what §5.0 avoids. It is fine for per-entity *fits* in retrospective code. |
| `core/model_selection.py:expanding_window_split` | Walk-forward folds counted **back from the end** of the sample | **Must not** define refit dates: its fold dates move when data is appended. The schedule is calendar-anchored (§6.1). |
| `imputation.py:CafeImputer` | PIT imputation via the optional `cafe` package | `KalmanImputer` is the CAFE-free fallback that the roadmap names. It has the same `fit`-records-columns contract. |
| `synth/`: `GroundTruth.factors["regime"]`, `break_steps`, `latent` | Planted common Markov regimes and structural breaks | Ground truth for the recovery tests. Do not write another regime simulator. |

---

## 3. Scope and non-goals

**In scope (v1).**
1. **Linear-Gaussian state space, vectorised across entities.** Models: local level (LL),
   local linear trend (LLT), and time-varying-coefficient regression (TVP, random-walk β),
   with RLS under a forgetting factor as a special case. Several observation columns of one
   latent are processed sequentially. A missing observation triggers a predict-only step,
   and initialisation is exact diffuse. Outputs are the filtered state and its s.d., the
   innovation, the standardised innovation, and the log predictive density.
2. **Walk-forward MLE** on a date-anchored refit schedule: profile likelihood for LL and EM
   with SQUAREM for LLT and TVP. Also `KalmanImputer`.
3. **Causal IIR bank.** Butterworth low-, high- and band-pass filters designed as SOS by
   the closed-form bilinear transform with prewarping. A DF2T kernel runs them vectorised
   across entities with deterministic initial conditions. Also included: the Ehlers super
   smoother and roofing filter, the Laguerre filter, frequency response and group delay,
   and an explicit refusal of `filtfilt`.
4. **Regimes.** Gaussian HMM (vectorised Hamilton filter, Baum–Welch on training windows;
   per-entity, pooled and common modes) with deterministic label alignment,
   Markov-switching mean/variance regression, and statistical jump models whose features
   come from **online inference only**.
5. **BOCPD** (Normal-inverse-gamma, top-R pruning). Outputs are cp probability, P(short
   run), E and MAP run length, and pruned mass.
6. **Changepoint segmentation.** The PIT-safe `pelt_online` gives the last changepoint of
   the optimal segmentation of the prefix. Offline PELT, BinSeg and optimal partitioning
   exist **only** in `retrospective` (§5.8).

**Non-goals, each with its reason.**
- **Leak-bearing outputs.** Smoothed states, posteriors and Viterbi paths are never
  features; they exist only in `retrospective`, behind `acknowledge_leak=True`. `filtfilt`
  and zero-phase filtering exist nowhere.
- **Outside this plan's model class.**
  - Nonlinear filters (EKF, UKF, particle filters): v2 at the earliest.
  - Stochastic volatility.
  - GARCH and MSM belong to sibling 9, which reuses our `hamilton_filter` for MSM's 2^k̄
    states.
  - Hamilton's mean-adjusted MS-AR needs K^(p+1) states. MS regression with lagged-y
    regressors covers the need instead.
  - HDP-HMM and automatic choice of K.
  - MCMC and BSTS belong to sibling 10, which calls our Kalman kernel.
- **Supervised selection.** Choosing the jump penalty by strategy Sharpe is supervised
  model selection and belongs in purged CV (`PurgedKFold`), run by the user.
- **Deferred to M7.**
  - Robust (β-divergence) BOCPD and Student-t observation models, which are
    non-conjugate.
  - WBS and FPOP.
  - `.ts` expression operators (§5.0 explains why).

**Boundaries with sibling plans.**

| Sibling | They own | We own | Interface |
|---|---|---|---|
| 2 drift-monitoring | SPC, ADWIN, Page–Hinkley on model or prediction streams, e-processes | BOCPD and PELT-online on panel series | They may *call* `statespace.bocpd`, and they consume our `zinnov` / `logpdf` streams. No second BOCPD. |
| 3 covariance-and-market-state | EWMA / shrinkage covariance, DCC, turbulence, absorption ratio, dynamic factor models | The Kalman and Hamilton kernels | A dynamic factor model or regime-conditional covariance is built on `_kalman` / `_hmm`, not re-implemented. |
| 4 label-weights | Labels and targets | `retrospective.*` segment and smoothed-label producers, flagged `leakage_safe=False` | They consume retrospective outputs as **targets only**. |
| 5 ohlc-volatility | Realised-vol estimators | — | Their vol is a valid causal standardiser for BOCPD and HMM inputs. |
| 8 multiscale-complexity | Wavelet filters, multiscale entropy | The IIR bank | No wavelets here. No IIR there. |
| 9 tail-risk | GARCH, MSM, Hawkes | Markov-switching mean/variance regression; `hamilton_filter` | MSM uses our forward kernel with a Kronecker-structured transition. |
| 10 panel-causal-inference | Synthetic control, BSTS-style counterfactuals | The Kalman filter and RTS smoother (training window) | They call the kernel. Counterfactual forecasting stays theirs. |

---

## 4. Module placement and public API

### 4.1 Decision: one new subpackage, `panelary/statespace/`

**Rejected: putting HMM and BOCPD in `detect/`.**
- `detect/`'s contract says "no numba", which the `fast` path would break.
- Its entry points take one 1-D series; everything here is a `(T,N)` lane kernel.
- Its build contract is already closed.

**Rejected: a separate `regime/` package.**
- A Hamilton filter and a Kalman filter share one predict/update skeleton; only the state
  differs (discrete versus Gaussian).
- BOCPD is a Hamilton filter on the run-length chain.
- The jump-model online DP and PELT's forward DP are both min-plus recursions.
- All of them need the same four pieces of infrastructure: the long↔dense adaptor, the
  date-anchored refit scheduler with label alignment, numba dispatch, and log-space
  numerics. Splitting them would duplicate that infrastructure or create circular imports.

"Regime" names a *use*; the package is named for the *method*. `detect/__init__`'s
docstring gets a "See also: `statespace.bocpd`, `statespace.pelt_online`" line, added by
the orchestrator. There are no re-exports.

### 4.2 Layout and file ownership (touch ONLY your files)

| File | Contents | Owner | M |
|---|---|---|---|
| `_dense.py` | long↔dense adaptor, `align="entity"\|"date"`, entity chunking, row-order restore | A | M1 |
| `_fast.py` | backend resolution, lazy numba kernel cache | A | M1 |
| `_logspace.py` | row log-sum-exp, Gaussian / Student-t log-pdf, `lgamma` table via `math.lgamma` | A | M1 |
| `_kalman.py` | LL / LLT / TVP / RLS filter kernels, exact diffuse, sequential multi-column, private RTS and disturbance smoother (training only) | B | M1 |
| `_schedule.py` | `Refit`, refit dates, training slices, `rerun` / `carry`, seeded streams, label alignment | C | M2 |
| `_estimate.py` | LL profile likelihood; EM + SQUAREM for LLT and TVP | B | M2 |
| `_impute.py` | `KalmanImputer` | B | M2 |
| `_hmm.py` | Hamilton filter, forward–backward (private), Baum–Welch, MS regression | D | M3 |
| `_bocpd.py` | BOCPD | E | M4 |
| `_segment.py` | PELT forward (online), optimal partitioning, BinSeg, costs, penalties | E | M4 |
| `_iir.py` | SOS design, DF2T kernel, Ehlers filters, Laguerre, `freqz` / group delay | F | M5 |
| `_jump.py` | Jump model: coordinate-descent fit plus online DP | D | M6 |
| `_transformers.py` | `KalmanFeatures`, `GaussianHMM`, `MarkovSwitchingRegression`, `JumpModel` | C | M2–M6 |
| `retrospective.py` | Non-PIT namespace with the `acknowledge_leak` gate | E | M1/M3/M4 |
| `_specs.py` | `FeatureSpec` registrations | orchestrator | each M |
| `__init__.py` | PEP 562 lazy exports (the `shape` pattern) | orchestrator | — |
| `tests/test_statespace_*.py` | §7 | G | all |

### 4.3 Public API

Everything below is built first in underscore modules. It is exported only under the
AGENTS.md rule (§12).

```python
import panelary as pn
ss = pn.statespace

# ---- linear Gaussian: fit-free with fixed params, or walk-forward with refit= --------
ss.local_level(df, *, entity, time, y: str | list[str], q=None, r=None, refit=None,
               outputs=("level", "level_sd", "innov", "zinnov"), align="entity",
               backend="auto", prefix=None) -> pl.DataFrame
ss.local_linear_trend(df, *, entity, time, y, ..., outputs=("level", "slope", "level_sd",
               "slope_sd", "innov", "zinnov"))
ss.tvp_regression(df, *, entity, time, y, x: list[str], intercept=True, q=None, r=None,
               refit=None, outputs=("beta", "beta_sd", "innov", "zinnov"))
ss.rls(df, *, entity, time, y, x, intercept=True, forgetting=0.99)   # TVP special case
ss.steady_state_gain(q: float) -> float     # alpha such that ewm_mean(alpha) == converged LL
ss.Refit(every="1y", window=None, min_train=252, lag=0, on_refit="rerun", burn=252,
         warm_start=True, restart_every=5, anchor=None)               # frozen dataclass
ss.KalmanFeatures(model="local_level", columns=..., refit=ss.Refit(), ...)   # PanelTransformer
ss.KalmanImputer(model="local_level", columns=None, refit=ss.Refit(), sigma=True)

# ---- IIR bank (fit-free) ----------------------------------------------------------
ss.butterworth_sos(order: int, period: float | tuple[float, float], btype="low") -> np.ndarray
ss.super_smoother_sos(period); ss.highpass2_sos(period)                   # Ehlers biquads
ss.sos_filter(df, *, entity, time, columns, sos, init="step", on_missing="skip")
ss.butterworth(df, *, entity, time, columns, order=2, period=20, btype="low")
ss.super_smoother(df, ..., period=10); ss.roofing(df, ..., hp_period=48, lp_period=10)
ss.laguerre(df, ..., gamma=0.8)
ss.frequency_response(sos, n=512) -> tuple[np.ndarray, np.ndarray]
ss.group_delay(sos, freqs) -> np.ndarray                                  # the lag you pay

# ---- regimes (PanelTransformers; emitted features are FILTERED) -------------------
ss.GaussianHMM(n_states=2, columns=..., pooling="entity" | "pooled" | "common",
               refit=ss.Refit(), order_by="variance", align="canonical" | "previous",
               n_init=4, sticky=10.0, seed=0,
               outputs=("prob", "pred_prob", "regime", "logpdf"))
ss.MarkovSwitchingRegression(y=..., x=[...], k_regimes=2, switching=("beta", "variance"), ...)
ss.JumpModel(features=[...], n_states=2, jump_penalty=50.0,
             refit=ss.Refit(every="6mo", window=3000), n_init=10, seed=0, order_by=None)
ss.hamilton_filter(loglik: np.ndarray, transition: np.ndarray, init: np.ndarray)
    -> FilterResult          # array kernel (T,N,K); public for siblings 3 and 9

# ---- changepoints -----------------------------------------------------------------
ss.bocpd(df, *, entity, time, y, expected_run_length=250, prior=ss.NIGPrior(...) | "refit",
         refit=None, max_hypotheses=100, short_run=5,
         outputs=("cp_prob", "p_short", "run_mean", "run_map", "pruned_mass"))
ss.pelt_online(df, *, entity, time, y, cost="normal_mean" | "normal_var" | "normal_meanvar",
               penalty: float, min_seg=5, outputs=("since_cp", "seg_mean", "seg_sd"))

# ---- retrospective: NOT point-in-time. Every call raises unless acknowledge_leak=True --
ss.retrospective.pelt(df, ..., cost=..., penalty="mbic", acknowledge_leak=True) -> segments
ss.retrospective.binseg(...); ss.retrospective.optimal_partitioning(...)
ss.retrospective.smooth(...)            # RTS-smoothed states (columns prefixed "retro_")
ss.retrospective.hmm_posterior(...); ss.retrospective.viterbi(...)
ss.retrospective.jump_labels(...)       # in-sample DP labels (jumpmodels .predict())
```

**Frame contract.**
- **Input** is a long `pl.DataFrame`, `LazyFrame` or `PanelFrame`. `(entity, time)` keys
  must be unique; a duplicate raises and names the key. Float32 is upcast before
  accumulating.
- **Output** is the input in its **original row order** with float64 columns
  `{prefix or col}_{model}_{output}` appended (for example `ret_ll_level`, `ret_hmm_p1`,
  `ret_bocpd_p_short`). Nulls are real nulls: kernel NaN is mapped through
  `fill_nan(None)`.

**`PanelTransformer` wrappers.**
- They declare `panel_safe=True, leakage_safe=True`. The flags are honest because θ is
  fitted *inside* the operator on the date-anchored walk-forward schedule (§6.1): every θ
  used at row t was fitted on data dated before t.
- `fit()` only records the columns, as `CafeImputer` does.
- `refit_log_` is exposed for audit: refit date, entity, parameters, loglik, n_iter,
  converged, permutation and alignment cost.
- `params="fit"` (fit θ on the training fold, then freeze it) is an M7 open question.

**Registry.**
- Every fit-free frame op registers a `FeatureSpec` with namespace `"statespace"`,
  `safe_scope="rowwise"`, `panel_safe=True`, `leakage_safe=True`, `source=<paper>`,
  `license="Apache-2.0"` and a `cost_hint`. It is tier B until M6's parity suite is green,
  and is exercised through `_FRAME_OPS` adapters in `tests/test_registry_conformance.py`.
- `retrospective.*` specs declare `leakage_safe=False`, `flavour="whole_series"` and
  `safe_scope="window"`. Conformance requires them to be *demonstrably not*
  prefix-invariant.

---

## 5. Algorithms

For each method we compare candidates, pick one, and justify the choice with complexity
and numerical stability.

### 5.0 The dense panel kernel pattern (applies to every recursion)

1. **Layout.** `_dense.to_dense(df, entity, time, cols, align)` returns `Y: (T_max, N, C)`
   float64, with NaN for missing values, and `rows: (T_max, N) int64 → input row`, with
   −1 marking padding.
   - Entity codes come from `pl.Enum` / `to_physical` over the **sorted unique** keys; time
     codes come from `searchsorted`.
   - Measured: 0.81 s to scatter and 0.15 s to gather 23.8M rows. The gather restores the
     original row order.
   - `align="entity"` is the default for per-entity models. Each entity's rows are
     left-aligned, so step k is its k-th row: the entity keeps its own clock, and a ragged
     panel wastes no steps.
   - `align="date"` makes step k the k-th shared unique time. It is required by
     `pooling="common"` / `"pooled"`.
2. **The loop.** `for t in range(T): state = update(predict(state), Y[t], obs[t]); out[t] = …`
   - State arrays are `(N,)`, `(N,k)` or `(N,k,k)`.
   - A missing value means **predict-only**, applied lane-wise with `np.where`. There is
     never per-entity Python branching.
   - The result is T numpy calls, not N·T Python steps.
3. **Chunking.** Lanes are independent. Entity chunks keep the working set at 512 MB or
   less: `chunk = max(1, 512MB // (8·T·(C_in+C_out)))`. Chunking is **bitwise invisible**
   (measured for batched `@` at TVP k=3) and has a test (§7).
4. **Backends.** `backend="auto" | "numpy" | "numba"`.
   - `"auto"` selects numba if it is installed **and** either N < 256 or the kernel is
     per-lane sequential (PELT, jump backtracking). Otherwise it selects numpy, which is
     equally fast at N=5000.
   - The numba kernels use `njit(cache=True, error_model="numpy")` and never `fastmath`.
     They perform the *same IEEE operations in the same order* as numpy, including scalar
     reductions in index order.
   - Target: bitwise-equal output (measured equal for LL).
   - Kernels whose numpy reduction is pairwise (BOCPD's log-sum-exp over R) get a
     documented tolerance of 1e-12 relative. For those, `"auto"` picks the backend by
     fixed rule, never by timing.
5. **No `.ts` expression ops in v1.** An expression evaluated `.over(entity)` sees one
   entity per call, which rules out the cross-entity vectorisation.
   - Measured: 5000 × 5000 per-entity Python runs in ~75 s, versus 0.15 s dense.
   - The single-series alternative is a blocked linear-recurrence scan (complex-pole
     partial fractions, block-local `cumsum` anchored at index 0, O(T/L) Python steps).
     It agrees with DF2T only to ~1e-13, which would make two numerically different
     implementations of one op. M7, on measured need only.

### 5.1 Linear-Gaussian filtering

**Models.** All are time-invariant except TVP, where Z_t = x_t'.

| Model | Observation | State transition | Noise |
|---|---|---|---|
| **LL** | y_t = μ_t + ε_t | μ_{t+1} = μ_t + η_t | ε ~ N(0, r); η ~ N(0, q·r), where q is the signal-to-noise ratio |
| **LLT** | Z = [1, 0] | state (μ, ν), T = [[1,1],[0,1]] | Q = diag(q_μ, q_ν)·r |
| **TVP** | y_t = x_t'β_t + ε_t | β_{t+1} = β_t + η_t | η ~ N(0, diag(q)·r) |
| **RLS** (special case of TVP) | same measurement update | P ← P/λ; r = 1 | Exactly exponentially-weighted LS with weights λ^{t−s} (Ljung & Söderström†) |

**5.1.1 Covariance update: candidates and decision.**

| Form | Cost per scalar observation | PSD guarantee | Verdict |
|---|---|---|---|
| Conventional P − K h'P | O(k²) | No: asymmetric, and can go negative | ✗ |
| Symmetric rank-1 P − F·KK', then symmetrise | O(k²) | Symmetric only | Fallback, `update="rank1"` |
| **Joseph** (I−Kh')P(I−Kh')' + rKK' | O(k³) via batched `@` | A sum of PSD terms; robust to gain error | **Default for k ≥ 2** (measured 2× rank-1 cost) |
| Potter / Carlson square root | O(k²) + re-triangularisation after P+Q | Yes; roughly double precision | Adopt only if the M1 stress suite finds λ_min < 0 under Joseph |
| Bierman–Thornton UD | O(k²), but k² Python-level ops per step | Yes | ✗: the Thornton time update costs as much as Joseph here |
| Information filter | O(k³) | — | Used only for the exact diffuse start |

- **LL uses the harmonic form**, P_{t|t} = P_{t|t−1}·r/F_t. It is strictly positive and
  contains no subtraction.
- k ≥ 2 models are symmetrised after every update: P ← ½(P+P').

**5.1.2 Several observation columns of one latent** (for example, the same quantity from
two vendors).
- With diagonal H, use **univariate sequential processing** (Koopman & Durbin 2000): p
  scalar updates in column order, skipping missing columns. This is exact, inverts no p×p
  matrix, and handles partial missingness per element.
- With non-diagonal H, decorrelate once by Cholesky (their LDL transform). Since H is
  fitted, the decorrelation is frozen per refit.

**5.1.3 Exact diffuse initialisation.**
- **LL:** a = y₁ and P = r after the first observation.
- **LLT:** closed form after two observations (Harvey 1989 §3.3).
- **TVP / RLS:** start in **information form** with Ω₀ = 0 and ω₀ = 0, which is exactly the
  diffuse prior.
  - Measurement update: Ω += hh'/r, ω += hy/r.
  - Random-walk time update: Ω ← Ω − Ω(Ω + Q⁻¹)⁻¹Ω, which is valid even when Ω is
    singular. For RLS, the time update is Ω ← λΩ.
  - Each lane switches to covariance form at its first well-conditioned step: the first
    Cholesky with min pivot / max pivot > 1e-10. This depends only on the lane's own
    history, so it is prefix-invariant.
  - β is null until the switch, because it is not identified before then.
- The likelihood excludes the d diffuse observations. This removes the P₀ = 10⁶ start and
  the ~7 digits it cost (§1.2).

**5.1.4 Steady-state gain shortcut.**
- On complete data, the LL and LLT recursions for P converge to the discrete algebraic
  Riccati equation (DARE) solution. For LL: P̄ = (q + √(q²+4q))/2·r and K̄ = P̄/(P̄+r).
- **Rule:** once a lane's P_{t|t} equals P_{t−1|t−1} **bitwise**, freeze its gain and skip
  the P recursion.
  - This happens by t = 500 at q = 0.01 (measured).
  - The shortcut is exact, because the full recursion would reproduce the same P.
  - A missing value knocks the lane out of steady state until P reconverges.
  - A lane whose P 2-cycles in the last bit never switches, which is harmless.
- TVP has no steady state.
- `steady_state_gain(q)` exposes K̄ for use with polars `ewm_mean(alpha=K̄, adjust=False)`.
  The docs state the transient (2.7e-2 at t = 10) and the different handling of nulls
  (§1.2).

**5.1.5 Outputs and cost.**
- Outputs are a_{t|t}, √diag(P_{t|t}), v_t, z_t = v_t/√F_t and the log predictive
  density.
- z_t is i.i.d. N(0,1) under a correct model, which is exactly the input `detect.focus`
  expects.
- A null y_t produces the predicted state and a null innovation.
- Cost is O(T·N·k³) with Joseph updates and O(T·N) for LL.
- Memory is dominated by outputs at T·N·8 bytes per column, which is why chunking exists.

### 5.2 Estimation (training windows only; smoothing inside a fit is legal)

| Candidate | Per-iteration cost | Robustness | Decision |
|---|---|---|---|
| **LL profile likelihood** in log q | 1 filter pass per evaluation | 1-D, bounded, deterministic | **Default for LL** |
| EM with the disturbance smoother (Shumway–Stoffer 1982; Koopman 1993†) | filter + smoother | Monotone; variances stay positive with no bounds; slow near the optimum | **Default for LLT / TVP, with SQUAREM** |
| Quasi-Newton on the exact score (Koopman & Shephard 1992) | filter + smoother gives the full gradient | Fast final convergence, but branchy across lanes | Opt-in `optimizer="lbfgsb"` polish, scipy-only, per-lane loop |
| Numeric-gradient BFGS | (1 + #θ) passes | Fine for one series; poor across lanes | ✗ |

**LL profile likelihood** (Durbin & Koopman 2012 §2.10.2).
- Concentrate σ²_ε out:
  - σ̂²(q) = (n−1)⁻¹ Σ_{t≥2} v_t²/F*_t
  - log L_c(q) = −(n−1)/2·(log 2π + 1 + log σ̂²) − ½ Σ log F*_t
- **Grid.** 33 points in log₁₀ q ∈ [−8, 4], stacked as lanes: one pass over N·33 lanes.
- **Refine.** Vectorised golden section inside each lane's bracket to |Δ log q| < 1e-6,
  about 33 passes. Deterministic, no RNG.
- **Boundary.** When the maximum is at the boundary q̂ = 0 (pile-up; Shephard & Harvey
  1990†), report it as such.
- **Cost.** About 5 s per refit at N=5000 and T_train=2500, from the measured 31 µs/step.

**EM for LLT / TVP.**
- **M-step.** It needs the lag-one smoothed covariance (de Jong 1989†):
  - r = n⁻¹ Σ[(y_t − Z_t â_{t|n})² + Z_t P_{t|n} Z_t']
  - q_j = (n−1)⁻¹ Σ[(Δâ_j)² + P_{t|n,jj} + P_{t−1|n,jj} − 2P_{t,t−1|n,jj}]
- **SQUAREM** (Varadhan & Roland 2008) extrapolates per lane in log-parameter space. If
  the likelihood falls, it falls back to the plain EM step.
- **Convergence** per lane: |Δℓ| ≤ 1e-9·n **and** max|Δ log θ| ≤ 1e-7, or
  `max_iter=500`. Both are logged in `refit_log_`.
- **Warm start** from the previous refit (§6.1).

### 5.3 `KalmanImputer`

A missing cell is filled with the **one-step-ahead predicted observation** Z a_{t|t−1},
never a smoothed value.
- **Extra columns.** `__sigma` = √F_t is the Gaussian predictive s.d., which widens through
  a gap. `__was_imputed` flags filled cells.
- **Parameters** come from the walk-forward `Refit`.
- **Versus forward-fill,** it carries the *filtered level*, not the last noisy print.

This is the roadmap's "CAFE-free fallback and principled single-series UQ source". It is
not a rival to CAFE.

### 5.4 IIR filter bank

**One-pole.** This is polars `ewm_mean(..., adjust=False)` itself (§2). The only thing we
build is `steady_state_gain`.

**Butterworth design: closed form, no scipy.** Sample rate 1; f_c = 1/period.
- **Prewarp:** Ω_c = 2·tan(πf_c).
- **Analog poles:** p_k = Ω_c·exp(iπ(2k+n−1)/(2n)).
- **High-pass:** p_k → Ω_c/p_k.
- **Band-pass:** the exact transform s → (s²+Ω₀²)/(sB), with Ω₀ = √(Ω₁Ω₂) from prewarped
  edges. The order doubles.
- **Bilinear map** z = (2+s)/(2−s). Zeros sit at −1 (LP), +1 (HP) and ±1 (BP).
- **SOS pairing** is trivial: conjugate pairs (k, n+1−k), plus one first-order section if
  n is odd.
- **Gain normalisation:** each section gets unit gain at DC, Nyquist or band centre, which
  spreads the gain and prevents overflow.
- **Parity** with `scipy.signal.butter`: ≤ 3.8e-15 (measured).

**Realisation: candidates and decision.**

| Form | Verdict |
|---|---|
| Direct-form `(b, a)`, order > 4 | ✗. Measured blow-up to 2.6e22 at order 8, f_c = 1/500. |
| DF-I per section | Works, but needs 4 states per section. |
| **DF-II transposed per section** | **Chosen.** 2 states and the best floating-point behaviour of the canonical forms; `sosfilt` uses it too. The step is y = b₀x+z₁, z₁ = b₁x−a₁y+z₂, z₂ = b₂x−a₂y, on `(N,)` arrays. Measured 0.15 s at 5000², order 4. |
| FFT filtering | ✗. Circular convolution; non-causal at block edges. |

**Initial conditions.**
- `init="step"` sets each section to its steady state for a constant input equal to the
  lane's **first observation** x₀: y_ss = x₀Σb/Σa, z₂ = b₂x₀ − a₂y_ss, z₁ = y_ss − b₀x₀.
  This is scipy's `lfilter_zi` in closed form, and it is prefix-invariant.
- `init="zero"` is also available.
- Initialising from the whole-series mean (`zi·x.mean()`) is refused.

**Missing values.** `on_missing="skip"` holds the state and emits null; `"hold"` emits the
last output. NaN never enters the state; in `lfilter` a NaN would be permanent.

**Ehlers filters** are all SOS instances, so they share one kernel.
- **Super smoother** (Ehlers 2013; radian form, since EasyLanguage uses degrees):
  - a₁ = e^{−√2π/P}, b₁ = 2a₁cos(√2π/P), c₂ = b₁, c₃ = −a₁², c₁ = 1−c₂−c₃
  - b = [c₁/2, c₁/2, 0], a = [1, −c₂, −c₃]
- **2-pole high-pass:**
  - α₁ = (cos(√2π/P) + sin(√2π/P) − 1)/cos(√2π/P)
  - b = (1−α₁/2)²[1,−2,1], a = [1, −2(1−α₁), (1−α₁)²]
- **Roofing** = HP(48) ∘ super smoother(10).

**Laguerre filter** (Ehlers 2002†) runs on a dedicated `(N,4)` kernel:
- L0 = (1−γ)x + γL0₋₁
- L_i = −γL_{i−1} + L_{i−1,−1} + γL_{i,−1}
- output = (L0 + 2L1 + 2L2 + L3)/6

**Diagnostics.**
- `frequency_response` evaluates H(e^{iω}) directly.
- `group_delay` sums the analytic biquad group delays.
- Every filter's docstring states its DC group delay, the lag price of causality.

**Refusal.** No `filtfilt`, `sosfiltfilt` or centred smoother exists in any namespace,
including `retrospective`: a zero-phase filter is necessarily two-sided. A test asserts
the symbols are absent.

### 5.5 Gaussian HMM and Markov switching

**Hamilton filter (scaled, log-shifted).** One step, vectorised over entities:
1. Predict: p̃_t = p_{t−1} @ A (batched).
2. Shift: m_t = max_k ℓ_t(k), where ℓ_t(k) is the log emission density.
3. Weight: w = p̃_t ⊙ exp(ℓ_t − m_t).
4. Update: p_t = w/Σw, and log c_t = m_t + log Σw.

Why this form:
- Plain scaling (Rabiner 1989) underflows when every emission density underflows, for
  example on an extreme outlier.
- A full log-space pass does K× more `exp`/`log` work.
- The log-shifted hybrid is as stable as log-space at plain-scaling cost. **Chosen.**

Outputs are all point-in-time:
- **filtered** p(s_t | y_{1:t});
- **predicted** p(s_{t+1} | y_{1:t}), the tradeable forecast;
- `regime` (argmax of filtered);
- `logpdf` (the predictive surprise).

A missing observation gets the predict step only. Cost: O(T·N·K²).

**Baum–Welch (training window only).**
- **E-step:** scaled forward–backward.
- **M-step:**
  - μ_k = Σγy/Σγ; σ²_k is a **two-pass** weighted variance.
  - A_ij ∝ Σξ_ij + α₀ + κδ_ij. The sticky Dirichlet pseudo-count (`sticky`) prevents
    degenerate switching.
  - Variance floor: 1e-4 × the training-window variance.
- **Convergence:** |Δℓ| ≤ 1e-8·(1+|ℓ|), or `max_iter=200`.
- **Restarts:** restart 0 is deterministic (states at variance quantiles, no RNG).
  Restarts 1..n_init−1 use seeded k-means++ (§6.1). All restarts run as extra lanes for 10
  iterations; only the best lane continues.

**Pooling modes.**

| Mode | Parameters | State path | Cost per iteration |
|---|---|---|---|
| `"entity"` | N sets | per entity | ≈ 3× filter cost on the training window |
| `"pooled"` | one set; M-step sums over entities (as `hmmlearn` `lengths=`) | per entity | same |
| `"common"` | one set | **one per date**: ℓ_t(k) = Σ_i log N(y_it; μ_ik, σ²_ik) | emission sum + one (T,K) pass |

- `"common"` is the market-regime model. The synth ground truth is a common regime, so the
  recovery test uses this mode.
- `"pooled"` and `"common"` are cross-sectional models. Under entity-held-out CV, use
  `"entity"` or refit per fold.
- `"common"` also needs the cross-section at t to be known at t; use `asof_join`.

**Markov-switching regression.** y_t = x_t'β_{s_t} + σ_{s_t}ε_t.
- Emission: Gaussian on the residual.
- M-step: weighted least squares per state and entity, via
  `np.linalg.solve(G, b[..., None])[..., 0]` over (N,K,p,p). This follows detect
  invariant 4, with a p == batch-size test.
- Lagged y is allowed as a regressor (MS-ARX).

### 5.6 Statistical jump models

**Fit (training window).** Coordinate descent (Nystrup et al. 2020; Bemporad et al. 2018†)
on Σ_t ½‖x_t − θ_{s_t}‖² + λΣ1[s_t ≠ s_{t−1}], alternating:
1. **States:** solved by DP. Backtracking is allowed because this is training.
2. **Centroids:** each θ_k is the mean of its assigned points. An empty state keeps its θ.

- **Stops** when labels are unchanged, or at `max_iter=50`.
- **Features** are standardised with training-window moments, frozen per refit.
- **Initialisation:** `n_init` seeded k-means++ starts; the lowest objective wins.

Because λ is constant, the DP is **O(K) per step, not O(K²)**:
V_t(k) = ℓ_t(k) + min(V_{t−1}(k), min_j V_{t−1}(j) + λ).

**Online inference is the only feature.**
- Carry the forward DP. The label is argmin_k V_t(k), with ties broken by the lowest
  canonical index.
- Subtract min V_t each step, so values stay within [0, λ + max ℓ].
- Shu et al. rerun a 3000-row DP every day: O(l·K) per day. We rerun once per refit, from
  `r − burn`, then step forward: O(K) per day.
- The two agree once the recursion couples, and λ-bounded relative values make coupling
  fast. Parity is therefore label agreement ≥ 99% after burn, not bitwise (§7).

Continuous (Aydınhan et al. 2024) and sparse (Nystrup, Kolm & Lindström 2021) jump models
are M7.

### 5.7 BOCPD (Adams & MacKay 2007)

**Convention: a measured trap.** Let r_t count the observations in the current run
*including* x_t. The recursion is:
- P(r_t=1, x_{1:t}) = H·π₀(x_t)·P(x_{1:t−1})
- P(r_t=r+1, x_{1:t}) = (1−H)·π_r(x_t)·P(r_{t−1}=r, x_{1:t−1})

The original convention scores x_t under the r_{t−1} predictive instead, which forces
P(r_t=0 | x_{1:t}) ≡ H (measured: 0.00400 for H = 0.004).

Our `cp_prob` = P(x_t starts a segment) is informative, but a single observation is weak
evidence: measured median 0.0041 at a 1.5σ break, against a 0.0027 baseline. The
**headline features** are `p_short` = P(r_t ≤ `short_run`), `run_mean` = E[r_t], and
`run_map`.

Measured E[r] medians for a break at t = 1000: 926 at t = 999, 348 at 1005, 20.2 at 1020,
and 97.7 at 1100.

**Model.** NIG(μ₀, κ₀, α₀, β₀) prior, giving a Student-t predictive with df = 2α_n and
scale² = β_n(κ_n+1)/(α_nκ_n).
- **Recursive (Welford-like) updates**, with no S₂ − S₁²/n cancellation:
  - μ_{n+1} = (κ_nμ_n + x)/(κ_n+1)
  - β_{n+1} = β_n + κ_n(x−μ_n)²/(2(κ_n+1))
- **Anchoring:** inputs are anchored at μ₀, never centred on the sample mean (the detect
  rule).
- **Log-gamma term:** lgamma(α₀+(n+1)/2) − lgamma(α₀+n/2) is read from an append-only table
  indexed by run length. The table is built with stdlib `math.lgamma`, so it needs no scipy
  and is deterministic.

**Pruning: candidates.**

| Scheme | Cost | Deterministic | Fixed width (vectorises over N) |
|---|---|---|---|
| None (exact) | O(T²) | yes | no |
| Threshold p < ε | O(T·R̄) | yes | no |
| Fearnhead & Liu (2007) resampling | O(T·R) | RNG | yes |
| **Top-R, capacity 2R, amortised** | **O(T·R)**, one `argpartition` per R steps | **yes**: ties broken by (−log p, run length) | **yes** |

- **Chosen:** top-R. `pruned_mass` is emitted so an undersized R is visible. The unpruned
  recursion is the test oracle.
- **Hazard:** H = 1/`expected_run_length`, or a user-supplied h(r).
- **Prior:** either an explicit `NIGPrior` on causally standardised input
  (`detect.volatility_rescale` or sibling 5's volatility), or `prior="refit"`, which uses
  training-window moments. Full-sample moments are trap 7.
- **Missing value:** predict only. All runs grow with probability 1−H, and an empty new run
  starts with probability H.
- **Cost:** O(T·N·R). State is 32 MB at N = 5000, R = 100.

### 5.8 Changepoint segmentation, and the PIT decision

**The fork.**
- **(a) Offline segmentation** (PELT, BinSeg, optimal partitioning) is retrospective by
  construction.
  - New data revises earlier changepoints.
  - BIC (p·log n) and MBIC (Zhang & Siegmund 2007) depend on n, the length of the data
    currently held.
- **(b) A PIT derivative comes for free.** It is exact, unlike expanding refits at a
  stride.
  - PELT computes F(t) = min_{τ∈R_t}[F(τ) + C(y_{τ+1:t}) + β] **for every prefix t**. Its
    argmin τ*(t) is the last changepoint of the optimal segmentation of y_{1:t}.
  - With a **fixed β**, τ*(t) depends only on y_{1:t} (measured bitwise
    prefix-invariant).
  - Pruning (Killick et al. 2012) discards only candidates that can never be optimal for
    any future t. So the pruned τ*(t) equals the one from optimal partitioning exactly.
    This is tested.

**Decision.**
- **`pelt_online`** (PIT, rowwise) emits `since_cp` = t − τ*(t), and `seg_mean` / `seg_sd`
  over y_{τ*+1:t}, a causal current-regime level.
- `penalty` must be a number. `"bic"` / `"mbic"` raise, with a message pointing to
  `retrospective`.
- `calibrate_penalty(arl0=…)` would follow the data-free MC pattern of `detect.mc_table`.
  M4 stretch goal.
- `retrospective` holds everything that returns a segmentation:
  - `pelt` (MBIC by default);
  - `binseg`: O(T log T) and greedy, with no causal analogue;
  - `optimal_partitioning`: O(T²) and exact; it is the oracle.

**Costs.**
- `normal_mean`: S₂ − S₁²/n, on standardised input.
- `normal_var`: known mean.
- `normal_meanvar`: n·log σ̂², with a variance floor and `min_seg ≥ 2`.

Prefix sums are anchored at y_first.

**FPOP** (Maidstone et al. 2017) is the offline twin of `detect.focus`'s functional pruning,
with O(log T) candidates. M7, and only if `pelt_online` misses its budget.

---

## 6. Refit schedule and leak-safety design

### 6.1 The refit schedule

`Refit(every, window, min_train, lag, on_refit, burn, warm_start, restart_every, anchor)`:

- **Refit dates.** The first shared-index time in each calendar period: the value of
  `time.dt.truncate(every)` (for example `"6mo"`), or `t % every == anchor` on an integer
  axis. Whether t is a refit date depends only on t and its predecessor, so appending data
  **never moves an earlier refit date**. This is why `expanding_window_split` is not used.
- **Training set for refit r.** Rows with `time < shift_back(r, lag)` (via
  `core._calendar`), cut to the trailing `window` if one is set; otherwise the window
  expands.
  - Each lane needs at least `min_train` observations. For pooled and common modes, the
    threshold applies to the whole panel.
  - Output is null before the first refit.
  - θ_r is frozen for t ∈ [r, r_next).
- **`on_refit`.** Both options are PIT.
  - `"rerun"` (default) re-filters from `r − burn` (absolute rows) with θ_r, starting from
    the stationary or diffuse state. Output at t is a function of
    (θ_{r(t)}, y_{r−burn..t}).
  - `"carry"` continues the old filter state after label alignment.
- **`warm_start`.** EM and golden-section search start from θ_{r−1}. This is deterministic
  and PIT. Every `restart_every`-th refit, fresh restarts compete against the warm start.
- **Seeding.** `PCG64(SeedSequence(seed, spawn_key=(refit_ordinal, restart, entity_hash)))`,
  where `entity_hash` is an 8-byte blake2b of the entity key (the `evolve/_genome.py`
  idiom). Streams are keyed by **entity**, never by lane position, so permuting entities
  permutes the outputs and changes nothing else. Pooled and common modes drop
  `entity_hash`.

### 6.2 Label alignment across refits

States are identified only up to permutation (Stephens 2000†).
- **`align="canonical"`** (the default) sorts states by `order_by`.
  - `"variance"` is the default: return means are near zero and noisy, while variances are
    well identified.
  - Other keys: `"mean"`, or `("feature", name)` for jump models.
- **`align="previous"`** picks the permutation of θ_r that minimises
  Σ|μ_k−μ'_k|/σ'_k + |log σ_k − log σ'_k| against θ_{r−1}. Brute force over K! (≤ 720 for
  K ≤ 6) avoids needing a Hungarian solver or scipy.
- Ties are broken by lexicographic parameter order.
- The permutation and its cost go into `refit_log_`. A high `alignment_cost` means the
  regimes changed character.

### 6.3 The traps: every one named, with the default that prevents it and the test that pins it

| # | Trap | Default that prevents it | Pinned by |
|---|---|---|---|
| 1 | RTS-smoothed state or `smoothed_state` as a feature | Features are filtered only. Smoothing is private, or behind `retrospective` + `acknowledge_leak`. | leaky twin `_leaky_rts` must fail prefix and poison checks |
| 2 | HMM posterior (`predict_proba`) or Viterbi (`predict`) as a feature | Same | leaky twin `_leaky_fb`; the 94.2% vs 90.7% gap as an assertion |
| 3 | jumpmodels `.predict()` in-sample DP labels | Online forward DP only | leaky twin `_leaky_jump_insample` |
| 4 | Filtered output with **full-sample θ** | Walk-forward `Refit`. A fit-free mode requires explicit params. | leaky twin `_leaky_full_theta` must fail prefix invariance |
| 5 | Refit dates counted from the sample end | Calendar-anchored schedule | `test_statespace_schedule.py`: appending data never moves a refit date |
| 6 | Label switching across refits | Canonical ordering or previous-alignment | Planted swap: recovery accuracy is unchanged across a refit boundary |
| 7 | Prior, standardisation or variance floor from full-sample moments | Training-window statistics or explicit constants | leaky twin `_leaky_prior` |
| 8 | BIC/MBIC with n = full length; segmentation broadcast per row | `pelt_online` takes numeric β only; segmentations live only in `retrospective` | `_leaky_bic_pelt` twin; string penalty raises |
| 9 | `filtfilt` / centred smoothers | Not provided anywhere | Symbol-absence test |
| 10 | Initial conditions from whole-series statistics (`zi·mean`, HMM π learned on the full sample) | Step init from the first observation; stationary π from A | poison test with a huge constant injected after t |
| 11 | Hyperparameters (K, λ, hazard, q grid) chosen on the full sample | Fixed arguments or `Refit`. Selection belongs to purged CV. | documented; no automatic selection shipped |
| 12 | RNG keyed by lane position | Entity-hash streams | entity-shuffle invariance test |
| 13 | numba `fastmath` or default `error_model` changing results | `error_model="numpy"`, no fastmath | backend parity test |
| 14 | A&M convention: P(r_t=0) ≡ H, which is uninformative | §5.7 convention | test: `cp_prob` is not constant |
| 15 | Pooled or common θ under entity-held-out CV | Documented. `pooling="entity"` recommended for grouped CV. | docs + test for a `pooling` warning under a grouped splitter (M3) |

---

## 7. Tests (`tests/test_statespace_*.py`)

`--strict-markers` is on. No new marker is planned; use `slow` and `benchmark`.

**`test_statespace_leak_safety.py`** applies the five `detect` invariants mechanically to
every public PIT entry point through one parametrised harness:
1. **Prefix invariance**, **tol = 0.0**, at 15 cuts including r−1, r and r+1 around every
   refit date.
2. **Future poison** (noise, then NaN, then 1e300 after t): the output at ≤ t is
   bit-identical.
3. **Determinism**: twice in-process and once in a fresh subprocess, byte for byte.
4. **Entity-shuffle invariance.**
5. **Affine equivariance** where it applies: for linear filters and LL / LLT / TVP states,
   f(a·y+b) = a·f(y) + b·gain to 1e-12 relative. Regime labels are invariant.

It also runs `panelary.testing.assert_no_lookahead` and `assert_prefix_invariant` with
**`tol=0.0`**, plus **one leaky twin per §6.3 trap, each of which must fail**. The twins
prove the harness has teeth, mirroring `test_depend_prefix.py`.

**`test_statespace_numerics.py`**
- PSD stress for TVP (cond(X'X) up to 1e10, q → 1e-12, scales of 1e±8): Joseph keeps
  λ_min(P) ≥ 0. After an exact diffuse start, Joseph and rank-1 agree to ≤ 1e-10; this
  is recorded against the 1.2e-7 measured under big-κ.
- An HMM emission outlier that underflows every density gives no NaN.
- BOCPD against the unpruned O(T²) oracle with R ≥ T agrees to ≤ 1e-10. `pruned_mass`
  at R=100 on break-bearing series is reported (target < 1e-6).
- Butterworth order 8, f_c = 1/500: the SOS output is bounded, and the `(b, a)` blow-up is
  asserted as the documented reason.
- The `solve(G, b[..., None])[..., 0]` idiom is tested with p == batch size.
- PELT and optimal partitioning give identical F(t) and τ*(t) for every t.
- The steady-state shortcut is bitwise identical on vs off.

**`test_statespace_backends.py`**: numba == numpy, bitwise for LL, IIR, HMM (K ≤ 3), the jump
DP and PELT, and ≤ 1e-12 relative for BOCPD (skipped without numba). Chunked == unchunked,
bitwise, for every kernel. `import panelary.statespace` loads neither numba nor scipy.

**`test_statespace_schedule.py`**:
- refit dates do not move when data is appended;
- θ_r is bitwise identical under truncation after r;
- `rerun` and `carry` are both prefix-invariant;
- both alignment modes repair a planted label swap;
- `refit_log_` has a fixed schema.

**`test_statespace_oracle.py`**: parity, each oracle behind `importorskip`.

| Ours | Oracle (licence) | Tolerance |
|---|---|---|
| LL / LLT filtered state; `llf` | `statsmodels` `UnobservedComponents(..., use_exact_diffuse=True)` (BSD-3; its default is approximate diffuse, verified) | ≤ 1e-9 rel after d; `llf` ≤ 1e-8 abs |
| LL MLE | same `.fit()` | \|Δ log q\| ≤ 1e-3 **and** ℓ_ours ≥ ℓ_sm − 1e-6 |
| TVP step-by-step | `filterpy.kalman.KalmanFilter` (MIT) | ≤ 1e-10 rel |
| MS regression, fixed θ | `statsmodels` `MarkovRegression` `filtered_marginal_probabilities`, `llf` | ≤ 1e-10 |
| HMM ℓ, fixed θ; `retrospective.hmm_posterior` | `hmmlearn` `score`; `predict_proba` (BSD-3) | ≤ 1e-9 |
| Butterworth SOS; `sos_filter` | `scipy.signal.butter(output="sos")`; `sosfilt(zi=…)` | ≤ 1e-12 (measured 3.8e-15; 1.25e-14) |
| `retrospective.pelt` | `ruptures.Pelt(model="l2")` (BSD-2), same penalty | identical changepoints |
| Jump-model online labels | `jumpmodels` `predict_online` (Apache-2.0) | agreement ≥ 99% after burn |

The implementation is clean-room from the papers. Every `FeatureSpec` records `source` and
`license="Apache-2.0"`. The R `changepoint` package (GPL) is reference-only.

**`test_statespace_recovery.py`**: planted truth from `panelary.synth`.
- **Regimes.** Data: `SynthConfig.plain(n_regimes=2, regime_persistence=0.98,
  regime_vol_ratio=3.0, n_factors=1, factor_persistence=0.0, n_entities=50,
  n_periods=2000)`. Fit `GaussianHMM(pooling="common", refit=Refit(every=250))` and
  compare with `truth.factors["regime"]`.
  - Filtered accuracy ≥ a floor measured at M3 and then frozen.
  - **smoothed accuracy − filtered accuracy > 0.** This pins the direction of the
    look-ahead gain; it must never be "fixed".
- **Breaks.** Data: `SynthConfig.plain(break_times=(600, 1200), break_size=2.0, ...)`,
  on entities with |shift| ≥ 2·idio s.d.
  - BOCPD `p_short` > 0.5 within 20 steps in ≥ 90% of those entities.
  - `pelt_online` τ*(break+30) within ±3 of the break in ≥ 90%.
  - The pre-break false-alarm rate is recorded.
- **Kalman.**
  - Planted LL, q ∈ {1e-3, 1e-2, 1e-1}, n=2000: q̂ within 2× in ≥ 90% of lanes.
  - Planted random-walk β: RMSE(MLE θ) / RMSE(true θ) ≤ 1.1.

**`test_statespace_detection.py`** (`slow`) compares BOCPD with `detect.focus`,
`detect.page_cusum` and `pelt_online`.
- Thresholds are calibrated to ARL₀ = 500 on i.i.d. N(0,1) by seeded, data-free MC.
- Scenarios: mean shift δ ∈ {0.5, 1, 2} with a known pre-change mean; δ=1 with an unknown
  pre-change mean; variance ×2 and ×3.
- The table of detection rate within 200 steps and conditional delay goes into the docs.
- **Asserted directions, pending measurement**, following the depend precedent (if the
  measurement disagrees, the docs report it and the assertion pins the measured order):
  - FOCuS delay ≤ BOCPD delay on a known-mean shift, because FOCuS is the exact GLR there.
  - BOCPD (`normal_meanvar`) detection rate ≥ FOCuS's on a variance-only shift.

**`test_statespace_perf.py`** (`benchmark`; wall-clock checks skipped on GitHub Actions
unless `PANELARY_STRICT_TIMING=1`, as in `test_depend_perf.py`): the §8 budgets with 2×
slack, and `tracemalloc` peak ≤ 1.5× the chunk budget.

**Standing guardrails.**
- `test_import_hygiene.py` (budget unchanged; lazy load) and `test_dependency_drift.py`
  (zero new mandatory deps).
- `test_wheel_guardrails.py`.
- `test_registry_conformance.py`, via `_FRAME_OPS` adapters; window specs are
  demonstrably not prefix-invariant.
- mypy ratchet (**no increase** in `MYPY_BASELINE`) and ruff.
- Polars **1.35 / 1.42** CI legs, using only 1.35 APIs (`pl.Enum`, `to_physical`,
  `dt.truncate`, `ewm_mean(min_samples=)`, `search_sorted`, `with_row_index`).

---

## 8. Benchmarks and performance budgets

**Design.**
- **Where:** full size in `benchmarks/bench_statespace.py`; scaled down in
  `test_statespace_perf.py`.
- **Panels:** N ∈ {100, 1000, 5000} × T ∈ {1000, 5000}, 5% missing at random, 10%
  staggered entry, string keys, seeded.
- **Reported:** kernel time, end-to-end time including pivots, µs/step, ns per lane-step,
  `tracemalloc` peak, backend, and machine metadata.
- **Threads:** single-threaded throughout (`OMP_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`).

| Case | Measured (§1.2) | Budget (single thread, laptop) |
|---|---|---|
| LL filter, 5000×5000 | 0.16 s kernel | ≤ 0.5 s kernel; ≤ 3 s end-to-end |
| LL + walk-forward profile MLE, yearly refits, 5000×5000 | ≈ 5 s per refit (derived) | ≤ 2 min |
| LLT filter; TVP k=3 Joseph, 5000×5000 | TVP 2.33 s at T=2000 | LLT ≤ 3 s; TVP ≤ 8 s |
| HMM K=2 filter, 5000×5000 | 1.09 s | ≤ 2 s (K=3: ≤ 3 s) |
| HMM per-entity Baum–Welch, yearly refits, 10-year window, 5000×5000 | — | ≤ 10 min numpy / ≤ 3 min numba (M3 gate) |
| HMM common mode, same | — | ≤ 30 s |
| BOCPD R=100, 1000×2000 | 6.8 s numpy / 4.8 s numba | ≤ 10 s numpy (5000² ≈ 2 min; R=30 ≈ 3× faster) |
| Butterworth order 4, 5000×5000 | 0.15 s | ≤ 0.5 s |
| `pelt_online`, T=5000, per entity | 41 ms numpy / 9–20 ms numba | ≤ 60 ms numpy / ≤ 25 ms numba |
| Jump online DP K=2, 5000×5000 | — | ≤ 1.5 s |

**Memory.**
- Each output column takes 8·T·N bytes (200 MB at 5000²).
- Chunking caps the working set at 512 MB.
- BOCPD state is 32 MB at N=5000, R=100.
- No kernel allocates a (T, N, R) array.

---

## 9. Dependencies

- **Mandatory:** none new.
- **Optional:** `numba`, via the existing `fast` extra. It is already mapped, so `_deps.py`
  is unchanged.
- **scipy:** not used, not even as an optional fast path; that would produce two
  numerically distinct answers.
- **Test-only oracles:** `statsmodels`, `hmmlearn`, `filterpy`, `ruptures`, `jumpmodels`,
  `scipy`, all behind `importorskip`.
  - No `pyproject` change is needed.
  - CI does not install them, so each milestone PR attaches a local
    `test_statespace_oracle.py` run (open question 6).

---

## 10. Milestones

| M | Contents | Ships | Gate to start the next |
|---|---|---|---|
| **M1** | `_dense`, `_fast`, `_logspace`; `_kalman` fixed-θ LL / LLT / TVP / RLS (exact diffuse, Joseph, sequential multi-column, bitwise steady-state switch); private RTS smoother plus `retrospective.smooth`; the leak-safety harness with leaky twins | The dense-kernel pattern and PIT trend/level features | Harness green with **every twin failing**; statsmodels / filterpy parity |
| **M2** | `_schedule` (`Refit`, alignment, streams); `_estimate` (LL profile, EM + SQUAREM); `KalmanFeatures`; `KalmanImputer` | Walk-forward state-space features; the roadmap's `KalmanImputer` | Schedule tests green; MLE parity |
| **M3** | `_hmm`: Hamilton filter, Baum–Welch (entity / pooled / common), `GaussianHMM`, `MarkovSwitchingRegression`, `hamilton_filter` kernel; `retrospective.hmm_posterior` / `viterbi`; synth recovery | Filtered regime probabilities | Recovery floors frozen; the smoothed > filtered gap pinned; the Baum–Welch budget measured |
| **M4** | `_bocpd`; `_segment` (`pelt_online`, OP, BinSeg, retrospective PELT); the detection study vs `detect` | Causal changepoint features | Detection table in the docs; ruptures parity |
| **M5** | `_iir`: Butterworth SOS design, DF2T, super smoother, roofing, Laguerre, `frequency_response`, `group_delay`, `steady_state_gain` | Causal filter bank | scipy parity |
| **M6** | `_jump` (online); numba kernels for EM / BOCPD / PELT; `test_statespace_perf`; `_specs` registration with conformance adapters; `docs/user-guide/statespace.md` (with the §1.1 trap table and measured numbers); CHANGELOG | Jump-model regimes; the full catalogue | — |
| M7 (conditional) | Seasonal state; continuous and sparse jump models; FPOP; `.ts` expression ops via the blocked scan; continuous-time (Δt-scaled) Kalman; Potter square-root; `params="fit"` mode; robust BOCPD | Only on a measured need or a named caller | — |

**Public exports are gated by §12, not by milestone.** Every milestone builds in
underscore modules. A name enters `panelary.statespace.__all__` only in the same change
set as its engine caller and engine test.

---

## 11. Risks and open questions

Resolve each with a benchmark, not an opinion.

1. **numba vs numpy on x86.** Outputs were bitwise-identical on arm64 (LL). x86 FMA
   contraction is not expected without fastmath, but CI must confirm it. If they differ,
   relax the tolerance to 1e-13 and pin the backend in the determinism tests.
2. **Batched `@` chunk invariance under OpenBLAS / MKL.** It holds on Accelerate
   (measured). If it fails elsewhere, use explicit (i,j) loops for k ≤ 8.
3. **Per-entity Baum–Welch at 5000² may take more than 10 minutes.** Mitigations, in order:
   lane pruning, warm starts, numba EM, then recommending `pooled`.
4. **EM local optima and regime drift inside a window.** `alignment_cost` detects this but
   does not fix it. Should a persistently high cost trigger a larger K, or just be reported?
5. **BOCPD is sensitive to outliers.** Is causal winsorisation (`econ._common.winsorize` on
   training quantiles) enough, or does it need a Student-t model (non-conjugate, so M7)?
6. **Oracle tests don't run in CI.** Options are a nightly oracle job or attaching a local
   log to each PR. This is a CI decision for the maintainer.
7. **Default clock for Kalman on ragged panels.** `align="entity"` ignores calendar gaps;
   Δt-scaled Q is M7. Measure how often gaps matter on real equity panels.
8. **`pelt_online` penalty.** Is data-free ARL₀ calibration worth shipping, or is
   β = c·log(n_ref) with an explicit `n_ref` enough?
9. **`params="fit"`.** Does any caller need a fit-then-freeze mode, given that
   walk-forward fitting is already leakage-safe?
10. **Jump-model parity** is label agreement, because the lookback definitions differ
    (§5.6). If agreement falls below 99%, document the coupling gap. Do not copy their
    O(l) daily rerun.

---

## 12. Caller

> **AGENTS.md: "No new public surface without a named caller in the engine, and a test in
> the engine that exercises it."** Today this plan does **not** satisfy that rule.

Truepoint does not import panelary anywhere (verified 2026-09-29). The only mentions are a
firewall rule in `truepoint/src/truepoint/_firewall.py` that forbids `diagnose/` from
importing it, and a `produced_by` example string in `schema/diagnosis.py`. No engine
module needs a Kalman filter, HMM or BOCPD yet. Hence §10: build privately, and export
only alongside the caller.

**Consumed by:** truepoint M03 temporal integrity (as-of items whose leak signature is
*hindsight labelling*).
**Caller:** `truepoint/src/truepoint/generate/family_asof.py`

**Mechanism.** `score/m03_temporal.py` already flags a response that reads as the gold's
`post_cutoff_value`. A new as-of template kind asks "which volatility regime (or trend
state) was X in as of the cutoff":
- **gold** is the *filtered* quantity (`GaussianHMM` / `local_level`, with θ fitted only
  on data before `knowledge_time`);
- **`post_cutoff_value`** is the *retrospective* smoothed quantity
  (`retrospective.hmm_posterior` / `smooth`).

This is the regime-dating leak: the machine version of quoting recession start dates that
were only announced months later. The template is generic, and this plan names no item,
sealed question, seed or answer key. Gold is computed at generation time, so `score/`
never imports panelary and the firewall stays untouched.

**Engine test**, written with the caller:
- (a) gold is prefix-invariant to as-of store data after `knowledge_time`
  (`panelary.testing.assert_prefix_invariant`);
- (b) an item is minted only when |gold − post_cutoff_value| exceeds the verifier
  tolerance, so every item carries a leak signal;
- (c) regeneration from the same seed is byte-identical.

**Other notes.**
- **Secondary candidate (not a gate):** regime-conditional checks in
  `truepoint/src/truepoint/validity/`.
- **Firewall:** BOCPD-based model-degradation diagnosis must never live in `diagnose/`.
- **Risk:** truepoint is local-only, with no remote or backup, so the caller's test is not
  backed up.
- **Library-internal consumers** shape the API but do **not** satisfy the rule:
  - sibling 4: targets from `retrospective`;
  - sibling 9: `hamilton_filter` for MSM;
  - sibling 2: the `zinnov` / `logpdf` streams;
  - sibling 3: dynamic-factor Kalman;
  - sibling 10: structural counterfactuals.

---

## 13. References

† = not re-verified in this session (page numbers and details from memory). Check before
citing in the docs.

**Verified this session (web):** Nystrup, Lindström & Madsen (2020), *Expert Syst. Appl.*
150:113307 · Nystrup, Kolm & Lindström (2021), *Expert Syst. Appl.* 184:115558 ·
Aydınhan, Kolm, Mulvey & Shu (2024), *Annals of OR* (to appear) · Shu, Yu & Mulvey
(2024), *J. Asset Management* 25:493–507, arXiv:2402.05272 (online inference, refit and
lookback details) · `jumpmodels` (Apache-2.0) README and `examples/nasdaq/example.py`
(predict vs predict_online) · statsmodels `UnobservedComponents` (`use_exact_diffuse`
defaults to False) · Ehlers super smoother and roofing coefficients (Linn Software and
LuxAlgo summaries of Ehlers 2013).

**Filtering and estimation:** Kalman (1960), *J. Basic Eng.* 82:35–45 · Muth (1960),
*JASA* 55:299–306† · Harvey (1989), *Forecasting, Structural Time Series Models and the
Kalman Filter*, CUP · Durbin & Koopman (2012), *Time Series Analysis by State Space
Methods*, 2nd ed., OUP · Koopman & Durbin (2000), *JTSA* 21(3):281–296 · Koopman &
Shephard (1992), exact score, *Biometrika* 79(4)† · Koopman (1993), disturbance smoother,
*Biometrika* 80† · Shumway & Stoffer (1982), *JTSA* 3(4):253–264 · de Jong (1989), *JASA*
84† · Shephard & Harvey (1990), *JTSA* 11(4)† · Varadhan & Roland (2008), SQUAREM,
*Scand. J. Statist.* 35(2)† · Bierman (1977)† · Bucy & Joseph (1968)† · Ljung &
Söderström (1983)†.

**Regimes:** Hamilton (1989), *Econometrica* 57(2):357–384 · Kim (1994), *J. Econometrics*
60† · Rabiner (1989), *Proc. IEEE* 77(2):257–286 · Stephens (2000), label switching,
*JRSSB* 62(4)† · Bemporad et al. (2018), fitting jump models, *Automatica* 96,
arXiv:1711.09220†.

**Changepoints:** Adams & MacKay (2007), arXiv:0710.3742 · Fearnhead & Liu (2007), *JRSSB*
69(4):589–605† · Murphy (2007), conjugate Gaussian (tech. note)† · Killick, Fearnhead &
Eckley (2012), PELT, *JASA* 107(500):1590–1598 · Zhang & Siegmund (2007), MBIC,
*Biometrics* 63(1)† · Maidstone et al. (2017), FPOP, *Stat. Comput.* 27† · Scott & Knott
(1974), *Biometrics* 30† · Romano et al. (2023), FOCuS, *JMLR* 24(81), as cited in
`detect` · Knoblauch, Jewson & Damoulas (2018), *NeurIPS*†.

**Filters:** Butterworth (1930)† · Oppenheim & Schafer, *Discrete-Time Signal Processing*
(bilinear transform and prewarping)† · Ehlers (2013), *Cycle Analytics for Traders*,
Wiley† · Ehlers (2002), "Time warp — without space travel" (Laguerre)†.
