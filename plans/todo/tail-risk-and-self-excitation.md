# `panelary/risk/` — tail risk and self-excitation: build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.**
> Sibling plan 9 of 10 (sibling boundaries are listed below). M1 (drawdowns) has a
> plausible engine caller. **M2 and later do not have one yet.** AGENTS.md's rule
> ("no new public surface without a named caller in the engine and a test in the
> engine") therefore blocks them until the engine names one. See **Caller**.

Five capabilities for a panel of entities over time, all leak-safe by construction:

1. **Conditional volatility with honest refits.** GARCH / GJR / EGARCH with Gaussian,
   Student-t and Hansen skew-t innovations. Parameters are fitted only on data before
   each refit date and frozen until the next one.
2. **Conditional EVT.** McNeil–Frey GARCH-EVT, which fits a GPD to standardised
   residuals above a threshold taken from the fit window.
3. **Self-exciting tail events.** Hawkes–POT: univariate and two-tailed (2T-POT),
   marked, market→stock, with a power-law kernel built from a sum of exponentials.
4. **Systemic co-risk.** ΔCoVaR, MES, LRMES and SRISK, with liabilities taken
   point-in-time.
5. **Drawdown analytics** as causal features and as metrics.

Every capability has a pure numpy + polars path. It has optional numba kernels
behind the existing `fast` extra, and oracles that are used only in tests.

Reuse, do not reinvent:

- `panelary.econ.features._evt`: `gpd_fit`, `pot_var_es`, `hill_index`, `GPDFit`.
- `panelary.econ.features._common`: `norm_cdf`, `norm_ppf`, `sorted_panel`.
- `panelary.econ._common`: `t_cdf`, `t_sf`, `chi2_sf`, `pinv_sym`, `factorize`,
  `newey_west_lrv`, `auto_bandwidth`.
- `panelary.depend._info.psi` (digamma).
- `panelary.shape._tensor.build_tensor(forward_fill=False, z_normalize=False)`: the
  one long→wide path. Sibling 3 names it the single pivot, so we do not write another.
- `panelary.core.asof.asof_join` (point-in-time liabilities).
- `panelary.core._calendar`: `validate_duration`, `shift_forward`, `BusinessDays`.
- `panelary.feature_extractors._kernels._get_cusum_numba`, the lazy-numba pattern
  to copy.
- `panelary.validation._forecast_tests.pinball_loss`.
- `panelary.testing`: `assert_no_lookahead`, `assert_prefix_invariant`.
- `panelary.registry`: `FeatureSpec` with `safe_scope`.

---

## Why this exists

### Measured on this machine

Apple M5 Pro, 15 cores, 48 GiB; Python 3.13.12, numpy 2.5.3, polars 1.44.2, numba 0.67.0.
The scratch scripts are not committed; `benchmarks/bench_risk.py` reproduces them in M2.

| Fact | Measurement | Consequence |
|---|---|---|
| **Full-sample GARCH fit versus a prefix fit.** 30 simulated GARCH-t(6) entities, T=3000, a volatility regime break at t=2000; compare the fit on `[0, 3000)` with the fit on `[0, 1500)`. | On rows 500–1500, σ_t moves by a **median 4.5% and a p90 6.6%** just because 1500 future rows were appended. | The classic trap is not a rounding error. Any row-level feature built from in-sample σ_t fails `assert_prefix_invariant` by a wide margin. |
| **`validation._cv.walk_forward_splits` geometry.** | Test-block starts are [170, 336, 502, 668, 834] at T=1000 and [185, 368, 551, 734, 917] at T=1100. | Its splits are anchored to **the end** of the sample. It cannot be used as a refit schedule: appending data moves every refit date. We need a calendar-anchored schedule (§A). |
| **Existing `evt_features`, which fits once per row in a Python loop.** | 0.23 s per entity × 5000 rows at w=252, so **~19 min** for 5000 entities. | Conditional EVT must refit on a schedule, batched across entities. We do not add a second per-row loop. |
| **GARCH(1,1) Gaussian log-likelihood plus analytic gradient**, 5000 lanes × 1000 obs. | numpy lane loop **78–103 ms**; numba serial **22 ms**; numba `prange` with 15 threads **3.2 ms**. numba and numpy agree to 4e-15. | The pure-numpy path is viable. numba is 25–30× faster, not a hard requirement. |
| **Geometric block scan versus lane loop** (same likelihood). | 1 lane × 5000 obs: **0.5 ms vs 9.8 ms**. 50 lanes: 4.6 vs 15.1 ms. 500 lanes: 21 vs 16 ms. 2000 lanes: 47 vs 37 ms. σ² agrees to 9e-16 and stays exact at β=1e-8. | Use the scan below about **256 lanes**; use the lane loop above that (§C.2). |
| **Hawkes exponential-kernel log-likelihood**, 5000 lanes × 2500 days, 5% event rate. | Day-by-day **165 ms**, event-time with closed-form gaps **99 ms**. The two agree to 5e-15. | The O(n_events) event-time form (Ozaki 1979) is exact. In numpy the gain is 1.7×; in numba it is about 20× fewer steps. |
| **Rolling max-drawdown via prefix/suffix blocks** (van Herk / Gil-Werman generalised). | **Bitwise equal** to brute force. 5000 × 5000 × w=252 takes **1.3 s** even with a per-entity loop. | An O(T) trailing max drawdown exists in pure numpy. Ship it (§H). |
| **GPD estimators**, relative RMSE of the 98% excess quantile, 1000 reps per cell. | See the table in §E. **PWM has the lowest RMSE for ξ ≤ 0.3 at n_exc ≤ 100.** Zhang–Stephens and MLE are nearly unbiased at n_exc = 250. PWM under-states the quantile by −14% at ξ=0.8. Cost per fit (n=100): PWM 8 µs, ZS 37 µs, `scipy genpareto.fit` 3.5 ms. | PWM stays the default. We do **not** switch to ZS on reputation. The choice is decided by a coverage benchmark (M4). |

### What the literature says, and why a panel library should own it

- **McNeil & Frey (2000).** GARCH filtering followed by a GPD on standardised residuals
  gives the best-calibrated conditional VaR/ES at 99% and above among the methods they
  compare. Their design is inherently a refit design: estimate on a window of 1000
  observations and forecast the next day.
- **Chavez-Demoulin, Davison & McNeil (2005).** A POT point process whose intensity is
  self-exciting beats the unconditional POT model for VaR.
- **Tomlinson, Greenwood & Mucha-Kruczyński (2024, IJF; arXiv 2202.01043).** The 2T-POT
  Hawkes model is "more reliably accurate than the GARCH-EVT model" for VaR/ES at 5%
  coverage and below, out of sample (2015–2022) on six large-cap indices. The paper
  depends on a reparameterisation (eq. 17: the mean intensity a_λ replaces μ, with
  a_λ = 2a_u) that cut optimisation time by 53% and reduced failed calibrations. We
  adopt it. Note that the paper calibrates **once** on 1975–2015. It never refits, so
  its protocol cannot be run on a live panel as published.
- **Hansen & Lunde (2005).** GARCH(1,1) is hard to beat out of sample. Accuracy therefore
  comes from **honest estimation and refits**, not from exotic variants.
- **Adrian & Brunnermeier (2016)** (ΔCoVaR, quantile regressions) and **Brownlees & Engle
  (2017)** (SRISK = k·D − (1−k)·W·(1−LRMES), from GJR-DCC) are the standard systemic
  co-risk measures. The published estimates are full-sample. In a feature pipeline that
  is a leak.
- **Filimonov & Sornette (2012)** estimated a rising Hawkes branching ratio as a
  "reflexivity" index. **Hardiman, Bercot & Bouchaud (2013)** found that with power-law
  kernels and long windows markets are "and have always been" close to criticality.
  **Filimonov & Sornette (2015)** and **Wheatley, Wehrli & Sornette (2019)** attribute
  apparent criticality to non-stationary baselines and kernel misspecification. The
  branching ratio is **fragile**, so we emit it as a parameter with a standard error and
  a documented caveat, never as a headline signal.

**The gap.** `arch` (Sheppard) fits one series at a time with scipy's SLSQP. rugarch is R
and GPL. `tick` is unmaintained on current Pythons†. Nobody ships panel-batched refits
with a leak-safety contract. A 5000-entity panel refitted monthly over 20 years is about
**1.07 million independent fits**. At an assumed 30 ms per `arch` fit† that is roughly
9 CPU-hours. Our target is minutes in pure numpy and seconds with numba.

---

## Hard invariants (every function)

1. **Prefix invariance, bitwise within a backend.**
   `f(x[:T])[t] == f(x[:T+k])[t]` for all `t ≤ T`, **including across refit
   boundaries**. Nothing may depend on `len(x)`: not refit dates, windows, thresholds,
   scan block anchors or normalisations.
2. **Batch-composition invariance.** A lane's output is bitwise independent of three
   things: which other entities share its batch or chunk, `chunk_entities`, and the
   numba thread count. There are no cross-lane reductions and no global early exit that
   changes a live lane.
3. **A fit sees only its fit window.** The fit window is the entity's observations with
   `time < refit_date − embargo`. Every fitted quantity comes from it and nothing else:
   parameters, the variance-targeting level s², the mean, the rescaling constant, the
   backcast, POT thresholds, GPD, DCC Q̄, kernel bandwidths and QR coefficients.
4. **Determinism.** There is no unseeded RNG anywhere. Multi-start is a deterministic
   grid. Simulation seeds are `np.random.SeedSequence([seed, stable_key, refit_ordinal])`
   with `stable_key = int.from_bytes(blake2b(repr(entity).encode(), digest_size=8))`.
   Never use Python's `hash()`, and never use one RNG stream whose consumption depends
   on the number of entities.
5. **float64 everywhere.** Float32 polars columns are upcast on extraction.
6. **Small linear algebra.** For k ≤ 8 parameters, use a **hand-rolled vectorised batched
   Cholesky** with triangular solves. Never call `np.linalg.solve(A, b)` on a stack:
   NumPy 2 mis-solves when p equals the batch size (detect contract invariant 4). Use
   `@` for batched Gram matrices, never bare `einsum`.
7. **Where loops are allowed.** Never `rolling_map`, never a per-window Python UDF, and
   never a per-entity Python loop on a hot path. Loops are allowed over time steps (with
   lanes vectorised), refit waves, optimiser iterations, event index and interior-point
   iterations.
8. **Prefix-stable sums.** A quantity emitted per row uses sequential accumulation
   (`np.cumsum`, `ufunc.accumulate`). Never use `np.sum` over a span whose length grows
   with T: pairwise summation is not prefix-stable bitwise. A sum over a fixed fit
   window is fine.
9. **numba kernels** use `error_model="numpy"`, `fastmath=False` (fastmath reorders
   reductions, which breaks invariant 2) and `cache=True`. They are lazily compiled
   (copy `feature_extractors/_kernels.py`). numba and numpy agree to 1e-12 relative, not
   bitwise. The backend is recorded in provenance.
10. **No silent failures.** Every fit reports `converged`, `n_iter`, the Newton
    decrement and `at_boundary`. A non-converged lane follows `on_fail` (default
    `"previous"`: keep the previous refit's parameters, which are strictly older).
11. **Output conventions.** VaR/ES are **positive loss magnitudes** for the lower tail,
    matching `pot_var_es`. A `*_next` column at row t is a forecast for t+1 made with
    information up to and including t.

---

## What already exists and must NOT be duplicated

Everything in this table was checked by reading the file.

| File : symbol | What it is | How this plan uses it |
|---|---|---|
| `econ/features/_evt.py:gpd_fit` | PWM GPD fit, threshold from `np.quantile`, strict `>` | Reused as the scalar reference. The batch kernel in §E must equal `gpd_fit(x, threshold=u)` to 1e-14. M4 adds `method={"pwm","zs","mle"}` with **default `"pwm"` unchanged**. |
| `econ/features/_evt.py:pot_var_es`, `hill_index`, `GPDFit` | POT VaR/ES formulas; Hill α | The same formulas, applied to standardised residuals. No second copy: the batch kernel imports the formula helpers. |
| `econ/features/_evt.py:evt_features` | Trailing-window **unconditional** EVT with a per-row Python loop (19 min for 5000 entities) | Left untouched. Its vectorisation is a separate follow-up, with the bitwise-parity requirement stated in Risks. |
| `econ/features/_harrv.py` (`realized_variance`, `HARModel`) | RV/BV/jumps; HAR fitted on the train fold | We do not re-implement realised measures. An optional GARCH-X term takes an RV column from here or from sibling 5. |
| `econ/features/_common.py:rolling_apply`, `per_entity_apply` | Per-row / per-entity Python loops | **Not used on hot paths** (invariant 7). |
| `depend/_coef.py:tail_dependence`, `exceedance_corr`, `tail_dependence_matrix`; `depend/_rolling.py:rolling_tail_dep` | Nonparametric tail dependence | ΔCoVaR, MES and SRISK are **conditional-quantile / expected-shortfall** measures, not copula λ. No overlap. The docs cross-link them. |
| `depend/_engine.py:devolatilise` | Divides by a causal rolling standard deviation | This is the model-free sibling of our `z`. Our GARCH `z` is documented as the model-based alternative. |
| `detect/_monitors.py:spot_variance`, `volatility_rescale` | One-sided kernel spot variance; `include_current=False` rationale | We cite the same self-normalisation argument for strictly-past POT thresholds (trap T12). No reuse of code. |
| `feature_extractors/_finance.py:max_drawdown`; `feature_extractors/_namespace.py:FeatureExtractor.max_drawdown` (`.ts.max_drawdown`) | Whole-series scalar MDD (window scope) | Kept. The new operators are **row-wise** and get distinct names. |
| `evolve/_ops.py:_b_ts_drawdown` | `y / y.rolling_max(d) − 1` (evolve op) | `.ts.drawdown(window=d)` **must be bitwise equal** to it (test). evolve is not edited. |
| `depend/_info.py:psi` | Vectorised digamma | Reused for the Student-t gradients. M2 hoists it to `_internal/_special.py` and leaves a one-line re-export in `depend._info`. That is the only edit to `depend/`. |
| `shape/_tensor.py:build_tensor` | Causal dense `(entity, time, value)` tensor; ffill and z-norm switchable | **The only pivot we use**, called with `forward_fill=False, z_normalize=False`. Sibling 3 declares it the single long→wide path, and `depend/_frame.py:PanelArrays.dense`, `econ/_connectedness.py:_pivot` and `StatisticalFactors._wide_matrix` already duplicate it. We add no fourth. |
| `detect/_panel.py:residualise` | PCA residuals with betas frozen on an **index-anchored** `refit_every` schedule (refits at `min_periods + k·refit_every`); prefix-invariant | This is the existing precedent for fit-and-freeze. `RefitSchedule(every=int, anchor="position")` must reproduce its refit positions exactly (test), so the two conventions cannot drift apart. |
| `core/asof.py:asof_join` | Bitemporal join with `knowledge_time` and `lag` | **The only way liabilities enter SRISK** (trap T8). |
| `validation/_cv.py:walk_forward_splits`, `core/model_selection.py:expanding_window_split` | End-anchored walk-forward splits | **Not reusable as a refit schedule** (measured above). A test pins the difference. |
| `validation/_forecast_tests.py:pinball_loss` | Quantile loss | Used in tests and benchmarks. |
| `econ/_common.py:t_cdf`, `t_sf` | Scalar-loop incomplete beta | Too slow for per-lane quantiles, so `_internal/_special.py` gains a vectorised version. The scalar ones become test oracles. |

A grep found no GARCH, Hawkes, quantile-regression, CoVaR, MES or SRISK code anywhere
in `panelary/`. The only existing drawdown code is the whole-series `max_drawdown` and
the evolve op. `conformal/_intervals.py:conformalized_quantile_regression` is a conformal
wrapper, not a QR solver.

---

## Scope, non-goals and sibling boundaries

**In scope.** Everything in the five capabilities above, plus pairwise (entity, market)
DCC because dynamic MES and LRMES need it.

Optional late milestones:

- N-dimensional DCC by composite likelihood.
- APARCH.
- GARCH-X with an exogenous realised measure.
- CoES.

**Non-goals, as decisions:**

- **GARCH(p, q) with p or q greater than 1**, and ARMA mean models. The mean is zero, or
  the fit-window mean estimated in two steps (§C.5). Joint μ is a documented non-goal.
- **FIGARCH / HARCH / MIDAS.** No evidence we can cite that they beat GARCH(1,1) plus
  refits in a panel. `arch` covers them.
- **An N×N multivariate Hawkes contagion network** (Aït-Sahalia et al. 2015) is not
  scheduled. If it is ever built, sibling 6 expects the **estimation to be ours**, with
  the matrix handed over through their `EdgeTable.from_matrix`. The same applies to a
  pairwise CoVaR network. This plan stops at univariate, 2T and bivariate market→stock.
- **BEKK.** It has O(N²) parameters with no panel-scale estimator. Sibling 3 lists it
  under this plan; we decline it.
- **Markov-switching multifractal (Calvet & Fisher 2004).** The out-of-sample evidence is
  strong, but the filter is an HMM forward filter over 2^k̄ states with Kronecker-
  structured transitions, which is sibling 7's machinery. Decision: this plan owns the
  volatility-model API and refit schedule. MSM is optional **M10**, built only on sibling
  7's forward-filter primitive. We never write a second HMM.
- **Performance inference.** VaR/ES backtests, FZ loss, QLIKE and Sharpe or drawdown
  inference belong to sibling 1. We use them; we do not ship them.
- **Intraday Hawkes on order flow**, and continuous-time event data. Daily bars only.

| Sibling | Boundary |
|---|---|
| 1 forecast-evaluation-and-sharpe-inference | Kupiec, Christoffersen, DQ, Acerbi–Székely, FZ0 and QLIKE are **theirs**. Our coverage tests call their backtests when present, and otherwise a test-local binomial interval (no public surface). They consume our `var_next_*` / `es_next_*` columns, aligned per invariant 11. Their backtests require an explicit `var_convention`; ours is always the positive-loss convention, and the docs say so. Inference on Calmar or drawdowns is theirs; the drawdown statistics are ours. |
| 2 drift-monitoring | We emit `pit` (the PIT u_t = F(z_t) under frozen parameters) and `z`. Their sequential monitors consume them. We build no monitors. |
| 3 covariance-and-market-state | Static and shrinkage covariance, EWMA, factor covariance and the Kelly–Jiang cross-sectional tail index (`xs.tail_index`) are **theirs**. Their plan assigns "DCC, GARCH, BEKK, DCC-NL dynamics, CoVaR, MES, Hawkes, per-entity EVT" to us. DCC is **ours**: pairwise in M8; N-dimensional composite likelihood and DCC-NL in M9, using their `nonlinear_shrinkage` (QIS) for the correlation target and their matrix container. |
| 4 label-weights-and-event-sampling | Event **samplers** are theirs. Our trailing-threshold exceedance indicator is a column they may consume. |
| 5 ohlc-volatility-and-liquidity | Range-based and realised volatility **inputs** are theirs. GARCH-X accepts their column. We compute no range estimators. |
| 6 network-and-spatial-panel | The graph container is theirs. Any excitation or CoVaR matrix we estimate enters through their `EdgeTable.from_matrix`; the estimation is ours. |
| 7 causal-state-space-and-regimes | Kalman, HMM and `hamilton_filter` are theirs. Their plan agrees that GARCH and MSM are ours and that MSM reuses their forward kernel with a Kronecker-structured transition (our M10). |
| 8 multiscale-complexity | Multifractal spectra and DFA are theirs. MSM is a volatility model, not a spectrum. |
| 10 panel-causal-inference | No overlap. |

**Shared primitives: a known conflict.** Three landed plans each specify a
date-anchored refit scheduler:

- ours, `RefitSchedule` (§A);
- sibling 7's `ss.Refit(every, window, min_train, lag, on_refit="rerun"|"carry", burn)` in
  `statespace/_schedule.py`, with a `refit_log_`;
- sibling 3's evaluation schedule (their §5.3), which adopts `residualise`'s
  index-anchored convention.

The semantics are compatible: our `embargo` is their `lag`, our terminal-state
continuity (§A.5) is `on_refit="carry"`, and both reject the end-anchored splitters.
**One primitive must ship, in `panelary/core/_refit.py`**, owned by whichever plan
reaches M2 first. The rest import it, and the field names follow sibling 7's (`lag`, not
`embargo`) if theirs lands first. The same applies to the dense adaptor: sibling 3 names
`build_tensor`, while sibling 7 proposes `statespace/_dense.py`. We use `build_tensor`.
numba dispatch is copied per package (`risk/_numba.py`), as every landed sibling does,
pending the consolidation into `_internal/_jit.py` that sibling 4 recommends. **The
orchestrator must resolve all three before any sibling's M2.**

---

## Module placement and file ownership

```
panelary/risk/
  __init__.py      # public API; the risk FeatureSpecs are registered here
  _schedule.py     # re-exports core/_refit.RefitSchedule; wave planning      (M2)
  _optim.py        # batched LM-Newton, grid starts, batched Cholesky         (M2)
  _dists.py        # normal / t / Hansen skew-t: logpdf, derivatives,
                   #   cdf, ppf, ES                                           (M2, M3)
  _garch.py        # GARCH / GJR / EGARCH recursions, VT, forecasts;
                   #   GARCH, garch                                           (M2, M3)
  _scan.py         # geometric block scan for linear first-order recursions   (M2)
  _numba.py        # all numba kernels, lazily compiled                       (M3)
  _cevt.py         # GARCH-EVT, batch PWM/ZS/MLE GPD                          (M4)
  _hawkes.py       # POT events, 1T/2T Hawkes, marks, SOE kernels             (M5, M6)
  _qr.py           # batched Frisch-Newton QR with vertex crossover           (M7)
  _systemic.py     # delta_covar, mes, dcc_pairwise, lrmes, srisk             (M7, M8)
  _drawdown.py     # drawdown kernels, CDaR, E[MDD] table                     (M1)
  _emdd_table.py   # generated constant table for Q_p / Q_n                   (M1)
panelary/core/_refit.py        # RefitSchedule (shared; see above)                   (M2)
panelary/_internal/_special.py # lgamma_vec, psi (moved), trigamma, betainc_vec,
                               #   t_cdf/t_ppf vectorised                             (M2)
panelary/namespaces/ts_risk.py # additive .ts drawdown operators + FeatureSpecs (the
                               #   depend precedent in namespaces/ts.py)             (M1)
```

**Edits outside `risk/`, all additive:**

- `panelary/__init__.py`: a guarded `risk` import.
- `econ/features/_evt.py`: `method=` on `gpd_fit`, default unchanged (M4).
- `depend/_info.py`: a psi re-export.
- `tests/test_registry_conformance.py`: `_FRAME_OPS` builders.
- `docs/user-guide/tail-risk.md`, the mkdocs nav and CHANGELOG.

There are **no new** `_MODULE_TO_EXTRA` rows, because `numba → fast` already exists.

---

## Public API

```python
import panelary as pn
from panelary.risk import RefitSchedule, GARCHSpec

sched = RefitSchedule(
    every="1mo",          # Date axis: polars duration for dt.truncate. Int axis: k (bucket = time // k).
    scheme="expanding",   # or "rolling"
    window=2500,          # cap on fit-window observations (rolling length, or the expanding cap)
    min_train=500,        # entity observations required before its first fit
    embargo=0,            # int steps or duration: fit data must satisfy time < refit - embargo
    warm_start=True,
)

# 1. Conditional volatility: a self-refitting causal feature (frame -> frame, row-wise safe)
out = pn.risk.garch(
    panel, returns="ret", entity="ticker", time="date",
    spec=GARCHSpec(kind="gjr", dist="skewt", mean="constant", variance_targeting=True),
    refit=sched,
    levels=(0.99, 0.975),     # emits var_next_99, es_next_99, ...
    horizons=(1, 5, 22),      # emits cumvar_h5, cumvar_h22 (closed form for GARCH/GJR)
    backend="auto",           # "numpy" | "numba" | "auto" (numba if the `fast` extra is installed)
    chunk_entities=1024, prefix="g_", return_log=False,
)
# columns: g_sigma, g_sigma_next, g_z, g_pit, g_var_next_99, g_es_next_99, g_cumvar_h5,
#          g_refit_id, g_param_age
# return_log=True -> (frame, refit_log): one row per (entity, refit) with params,
#   robust (sandwich) SEs, loglik, n_obs, window bounds, converged, n_iter, newton_decrement,
#   at_boundary, start in {"grid", "warm"}, backend, schedule_hash

# The PanelTransformer form (fit on the train fold, apply frozen; or keep refitting)
m = pn.risk.GARCH(returns="ret", spec=..., refit="train" | sched)
m.fit(train).transform(test)    # refit="train": frozen params, filter state carried from
                                # train's terminal state (trap T5); refit=sched: the same
                                # self-refitting behaviour as garch()

# 2. Conditional EVT (McNeil-Frey)
pn.risk.garch_evt(panel, returns="ret", spec=..., refit=sched, tail_fraction=0.10,
                  gpd="pwm", levels=(0.99, 0.995))
# -> ..._var_next_99, _es_next_99, _gpd_xi, _gpd_sigma, _gpd_u

# 3. Hawkes-POT
pn.risk.hawkes_pot(panel, returns="ret", tails="both",     # "lower" | "upper" | "both" (2T-POT)
                   a_u=0.025, kernel="exp",                 # "exp" | "powerlaw" (sum of exponentials, K=8)
                   marks=True, scale_on_intensity=True,     # alpha (mark impact), eta (GPD scale on lambda)
                   excite_by=None,                          # or "mkt_ret": market->stock excitation
                   on="returns",                            # or "residuals" (chained after garch)
                   refit=RefitSchedule(every="3mo", window=5000, min_train=2500),
                   levels=(0.99,))
# -> p_lower_next, p_upper_next, intensity_next, var_next_99, es_next_99, var_censored,
#    branching_ratio, u_lower, u_upper
pn.risk.HawkesPOT(...)            # PanelTransformer form

# 4. Systemic co-risk
pn.risk.delta_covar(panel, returns="ret", system="fin_ret", state=["vix", "term"],
                    q=0.05, exclude_self=True, weights="mcap", refit=sched)
                    # state columns lagged one date internally
pn.risk.mes(panel, returns="ret", market="mkt_ret", method="static" | "dynamic",
            q=0.05, window=252, c=-0.02)
pn.risk.dcc_pairwise(panel, returns="ret", market="mkt_ret", spec=..., refit=sched,
                     variant="dcc" | "cdcc")
pn.risk.srisk(panel, liabilities=vintages, knowledge_time="filed", lag=None,
              equity="mcap", lrmes="beta" | "mes18" | "simulate", k=0.08, seed=0)
pn.risk.srisk_aggregate(srisk_frame)   # per date: SRISK total and shares (cross-sectional)

# 5. Drawdowns, as .ts expressions (row-wise) plus one frame function
pl.col("px").ts.drawdown()                          # P / cum_max(P) - 1
pl.col("px").ts.drawdown(window=63)                 # bitwise equal to evolve's ts_drawdown
pl.col("px").ts.drawdown_duration(unit="rows")      # time under water
pl.col("px").ts.rolling_max_drawdown(window=252)    # O(T) block algorithm, §H
pl.col("px").ts.ulcer_index(window=14)
pn.risk.drawdown_features(panel, price="px", window=756, cdar_alpha=0.95)
    # -> rolling_mdd, cdar, calmar, emdd_bm, mdd_surprise
pn.risk.expected_max_drawdown(mu, sigma, horizon)   # Magdon-Ismail et al. 2004, vectorised
```

### Registry specs

Every spec has `panel_safe=True`, `leakage_safe=True`, `safe_scope="rowwise"`,
`axis="time"`, `flavour="trailing"`, `license="Apache-2.0"` and `source="Panelary"`
(papers are cited in docstrings, following the `.ts` convention).

| name | namespace | shape | tier | cost_hint |
|---|---|---|---|---|
| `garch`, `garch_evt` | `risk` | frame→frame | B | `O(N T iters)` |
| `hawkes_pot` | `risk` | frame | C | `O(N n_events K iters)` |
| `delta_covar` | `risk` | frame | C | `O(N R W p^2 iters)` |
| `mes` | `risk` | frame | B | `O(N T k)` |
| `dcc_pairwise` | `risk` | frame | C | `O(N T iters)` |
| `srisk` | `risk` | frame | C | `O(N T)` |
| `drawdown`, `drawdown_duration`, `ulcer_index` | `ts` | series | A | `O(T)` |
| `rolling_max_drawdown` | `ts` | series | B | `O(T)` |

`risk` is not a Polars expression namespace. Every `risk` spec therefore gets a
`_FRAME_OPS` builder in `tests/test_registry_conformance.py`, with probe-sized settings
(`min_train=8, window=10, refit=RefitSchedule(every=4)` on the probe's `Int64` time axis).
With those settings the probe panel of 12–24 rows exercises real refits instead of
passing vacuously on all-NaN output. `srisk` has a builder that synthesises a vintage
table.

---

## Algorithms

### A. Refit schedule (calendar-anchored, prefix-invariant)

1. **Refit dates.**
   - Build the panel's global sorted unique time axis τ.
   - The bucket of each time is `dt.truncate(every)` on a Date/Datetime axis, or
     `time // every` on an Int axis (anchored at absolute 0, not at the sample start).
   - Refit dates are D = the first τ in each bucket after the first.
   - D depends only on dates up to and including each refit date, so truncating the
     panel never moves a surviving refit date.
   - `anchor="position"` instead buckets global row positions `min_train + k·every`. This
     reproduces `detect/_panel.py:residualise` exactly, and a test pins it.
   - Never derive refit dates from `n_times`, `n_splits`, or the position of the last
     row (walk-forward geometry is end-anchored; measured above).
2. **Fit set for (entity i, refit d).** The last `min(c_i(d), window)` finite
   observations of i with `time < d − embargo`, where c_i is the count of such
   observations. Missing days are **compacted**: the recursion runs over consecutive
   observations. A lane is active at d only if `c_i(d) ≥ min_train`.
3. **In-force parameters.** The parameters used at row t come from the latest refit d with
   `d ≤ t`. They apply to rows `d ≤ t < d_next`.
4. **Waves.** Entities are processed in lock-step: wave k fits every active lane at
   `d_k`, warm-started from wave k−1. This loops over refit dates (about 214 for monthly
   refits over 20 years), never over entities.
5. **Filter continuity (the arch-compatible convention).** At refit d the live filter
   state is **the terminal state of the final likelihood pass over the fit window**. It
   is then advanced with the new parameters through any embargo gap and forward. This
   costs nothing extra, and it is exactly what
   `arch_model(r[:d]).fit().forecast(horizon=1)` produces, which makes parity exact up
   to optimiser tolerance.
6. `schedule_hash` (blake2b of the schedule's parameters) goes into every refit-log row.

### B. Batched optimiser: damped Newton / Levenberg–Marquardt with masks

We compared three candidates:

- **(i) scipy SLSQP** (what `arch` uses). One problem per call, a Python loop over lanes,
  and a scipy dependency. Rejected.
- **(ii) Batched BFGS.** No Hessian, but 40–80 iterations at k=12 on poorly scaled
  problems.
- **(iii) Batched damped Newton with an analytic Hessian from the same recursion pass.**
  Quadratic convergence from a warm start, typically 2–4 iterations. **Chosen**, with a
  pluggable Hessian:
  - `"exact"` (GARCH/GJR, DCC);
  - `"bhhh"`, the outer product of per-observation scores (EGARCH, Hawkes);
  - `"bfgs"`, a fallback update.

**Why the exact Hessian matters.** Under QMLE with non-Gaussian innovations the
information-matrix equality fails. BHHH is then a poor Hessian and converges slowly.
M2's benchmark measures iterations and wall time for exact versus BHHH on 1000 simulated
and 500 real lanes; the default follows the measurement.

**One iteration, all lanes at once:**

```
f, g, H = loglik(theta)                     # (L,), (L,k), (L,k,k); one recursion pass
A = -H + mu_i * diag(max(diag(-H), 1e-8))   # Marquardt scaling; Cholesky fails -> mu_i *= 4, retry
step = chol_solve(A, g)                     # hand-rolled batched Cholesky, k <= 8
f_new = loglik_value(theta + step)          # value-only pass, about 0.4x the cost
rho = (f_new - f) / (g.step - 0.5 step'A step)   # gain ratio (Nielsen 1999 update of mu)
accept = rho > 1e-4;  mu *= where(rho > 0.75, 1/3, where(rho < 0.25, 2, 1))
converged |= (0.5 * g' A^{-1} g / n_obs < 1e-12) | (max|step| < 1e-10)   # Newton decrement
```

- **Masks.** Converged or failed lanes are frozen: they are excluded from the active index
  set, and their values never change again (invariant 2).
- **Limits.** `max_iter=100` (200 for Hawkes).
- **Unconstrained reparameterisation**, so the constraints hold without projection:

  | Model | Reparameterisation |
  |---|---|
  | GARCH | `p = α+β = p_max·σ(θ₁)`, `α = p·σ(θ₂)`, `β = p − α`, with p_max = 1 − 1e-6 |
  | GJR | persistence `α + γ·m₋ + β` with `m₋ = E[z²·1(z<0)]` (½ if symmetric; skew-t by closed form); `α + γ ≥ 0` |
  | EGARCH | `β = tanh θ` |
  | Student-t | `ν = 2 + exp θ`, capped at 500 |
  | Skew-t | `λ = tanh θ` |
  | Hawkes | `n = (γ←+γ→)/2 = n_max·σ(θ)`; decays and scales use `exp` |

- **Boundaries.** θ is clipped to [−30, 30]. A lane that pins there sets `at_boundary`:
  α → 0 is a real MLE boundary that `arch` can reach exactly and we cannot, so parity
  there is judged on log-likelihood (Tests).
- **Starts.** A deterministic grid at the first refit, and for any lane that failed to
  converge. The GARCH grid is p ∈ {0.90, 0.95, 0.98, 0.995} × α-share ∈ {0.03, 0.08, 0.15},
  × ν ∈ {5, 10} for t, × λ ∈ {−0.1, 0.1} for skew-t. It is evaluated as extra lanes in one
  value-only pass. Later refits warm-start from the previous refit.
- **Standard errors.** Robust sandwich H⁻¹ S H⁻¹ (Bollerslev & Wooldridge 1992), where S is
  the per-observation score outer product. It comes free from the final pass.

### C. GARCH family

1. **Recursions**, each run on fit-window data rescaled by the fit-window standard
   deviation s_w (trap T15). GARCH is scale-equivariant: α and β are invariant and ω
   scales by s_w². This conditions the problem and makes the tolerances scale-free.
   - GARCH / GJR: `σ²_t = ω + (α + γ·1[e_{t−1}<0])·e²_{t−1} + β·σ²_{t−1}`.
   - Variance targeting (Engle & Mezrich 1996): `ω = (1 − persistence)·s²`, with s²
     from the fit window. In scaled units s² = 1.
   - Backcast: `σ²_0 = Σ_i w_i e²_i` over the first min(75, W) fit-window observations,
     with `w_i ∝ 0.94^i` (arch's convention†, verified by the parity test). Its
     derivatives are 0.
   - EGARCH (Nelson 1991): `ln σ²_t = ω + α(|z_{t−1}| − E|z|) + γ z_{t−1} + β ln σ²_{t−1}`,
     where E|z| comes from the innovation distribution (closed form for normal and t).
2. **Two numpy implementations of the same math.** Each tangent and second-order tangent
   is itself a linear first-order recursion **with the same coefficient β**:
   - `∂α σ²_t = (e²_{t−1} − s²) + β ∂α σ²_{t−1}`;
   - `∂β σ²_t = (σ²_{t−1} − s²) + β ∂β σ²_{t−1}`;
   - `∂αβ σ²_t = ∂α σ²_{t−1} + β ∂αβ σ²_{t−1}`;
   - `∂ββ σ²_t = 2∂β σ²_{t−1} + β ∂ββ σ²_{t−1}`;
   - `∂αα σ²_t = 0` under variance targeting.

   The two implementations are:
   - **Lane loop.** Loop over t with `(L,)` arrays; all recursions advance together.
   - **Geometric block scan** for `y_t = β y_{t−1} + x_t`. For block size B and in-block
     step s, `y_s = β^s (y_prev + cumsum_j β^{−j} x_j)`.
     - `P = β^{1..B}` is computed once per evaluation.
     - Each block costs four vectorised ops over `(L, B)`.
     - B is chosen per call as `clip(floor(690 / −ln β_min), 8, 128)`, so `β^{−B}` never
       overflows. B=128 when β ≥ 0.005, and B=37 at β=1e-8.
     - All summands are non-negative, so there is no cancellation (measured agreement
       9e-16).
     - Blocks are anchored at the start of each fit window or live segment, which is
       fixed, and `np.cumsum` is sequential. The scan is therefore prefix-stable
       (invariant 8).

   **Dispatch:** the scan when `L < 256`, the lane loop otherwise (measured crossover;
   re-measured on x86 in M2). GJR's indicator only changes the source term x_t, so the
   scan applies. APARCH with δ fixed is linear in σ^δ, so the scan applies there too.
   EGARCH is nonlinear (z_{t−1} depends on σ_{t−1}), so it always uses the lane loop and
   the Hessian is BHHH.
3. **numba** (`fast`). One kernel per model family: a `prange` over lanes, a scalar loop
   over t, and value, gradient and exact Hessian accumulated in registers. The measured
   likelihood-plus-gradient cost is 22 ms serial and 3.2 ms on 15 threads for 5000×1000.
   This is the only path where warm-started, sequential-over-refits per-lane
   optimisation is cheap.
4. **Distributions** (`_dists.py`, pure numpy, vectorised over lanes).
   - Per-lane constants (lgamma, psi, trigamma of ν) are computed once per evaluation,
     never per observation.
   - Student-t is standardised to unit variance.
   - Hansen (1994) skew-t uses `c = Γ((ν+1)/2) / (√(π(ν−2)) Γ(ν/2))`,
     `a = 4λc(ν−2)/(ν−1)` and `b = √(1 + 3λ² − a²)`, split at −a/b.
   - Analytic ∂/∂ν and ∂/∂λ, plus the exact Hessian entries.
   - **Quantiles** are needed only once per (lane, refit), so about 1.07M evaluations.
     The t ppf uses Hill's (1970) initial approximation plus ≤ 3 Halley steps on a
     vectorised incomplete-beta t cdf (Lentz continued fraction, masked, ≤ 300
     iterations). The skew-t ppf follows in closed form from the t ppf.
   - **ES** in closed form for normal and t. For skew-t, Gauss–Legendre quadrature (64
     nodes) of the quantile function after the substitution `u = 1 − (1−q)s²`, which
     removes the endpoint singularity.
   - Targets against scipy: logpdf 1e-13, ppf 1e-10, ES 1e-9.
5. **Mean.** `mean="constant"` subtracts the fit-window sample mean (two-step). This is
   **not** a joint MLE parameter; parity runs `arch` with `mean="Zero"` on the same
   demeaned window. `mean="zero"` is also offered. Estimating the mean jointly would add
   a non-scan recursion (∂e/∂μ) for an effect that M2 will measure at daily horizons
   before anyone revisits it.
6. **Forecasts.**
   - GARCH / GJR: `E_t σ²_{t+h} = σ̄² + P^{h−1}(σ²_{t+1} − σ̄²)`, where P is the
     persistence. `cumvar_h` is the closed-form geometric sum.
   - EGARCH has no closed form for arithmetic variance. `horizons` > 1 is refused unless
     `simulate=True`, which uses seeded paths with per-(entity, refit) seeds.
   - A one-step VaR at level q is `μ + σ_{t+1} F⁻¹(1 − q)`, reported as a positive loss.
   - The multi-horizon VaR of **aggregated** returns is not the scaled one-step quantile.
     It is available only through seeded filtered historical simulation (Barone-Adesi
     et al. 1999†) in M3, and documented as such.
7. **Budget for 5000 × 5000, monthly refits, window cap 1000, GARCH(1,1)-t with VT.**
   About 214 waves at about 4 Newton iterations each, with a measured Gaussian
   likelihood-plus-gradient cost scaled ×1.5 for t and ×1.4 for the Hessian recursions:
   - numpy: roughly 3–4 min;
   - numba serial: roughly 40 s;
   - numba on 15 threads: roughly 6 s;
   - the live filter pass: under 1 s.

   Memory: `chunk_entities=1024` gives about 8 MB per `(L, W)` array, with about 20 live
   arrays, so ≤ 200 MB, plus the 200 MB dense `(N, T)` input.

### D. Special functions (`_internal/_special.py`)

- `lgamma_vec` uses `np.vectorize(math.lgamma)`. It is called on per-lane constants only.
- `psi` moves here from `depend._info`.
- `trigamma` uses a recurrence to x ≥ 10 and then an asymptotic series; the target is
  1e-13.
- `betainc_vec` and vectorised `t_cdf` / `t_ppf`.

Each is tested against known values, with `scipy.special` as an `importorskip` oracle.
**scipy is on no production path.** There is nothing for it to accelerate that the
per-lane-constant design has not already removed.

### E. Conditional EVT (McNeil–Frey), and the GPD estimator decision

At each refit, the GARCH fit gives in-window standardised residuals z (fit window only).

- **Threshold.** The **(k+1)-th largest order statistic** of the loss-oriented z, with
  `k = round(tail_fraction · W)` (McNeil & Frey used k=100 of 1000). This gives exactly k
  exceedances per lane, so batches are rectangular, and uses no interpolation.
- **GPD.** Fit (ξ, σ) to the k excesses and get `z_q = u + (σ/ξ)·[((W/k)(1−q))^{−ξ} − 1]`
  and `ES_z` from the existing `pot_var_es` formulas.
- **Forecast.** `VaR_{t+1} = μ + σ_{t+1|t} z_q` and `ES_{t+1} = μ + σ_{t+1|t} ES_z`.

The batch kernel is **new code for batching only**. Its PWM arithmetic must equal
`gpd_fit(x, threshold=u)` to 1e-14 on every lane. The existing scalar function is the
oracle, and its semantics (the `np.quantile` threshold) are untouched.

**Estimator comparison.** Relative RMSE of the 98% excess quantile, 1000 reps, measured.
Bias is shown in brackets where it was recorded.

| ξ | n_exc | PWM | Zhang–Stephens 2009 | MLE (scipy) |
|---|---|---|---|---|
| 0.1 | 25 / 50 / 100 / 250 | **0.319 / 0.230 / 0.162 / 0.106** | 0.499 / 0.269 / 0.174 / 0.108 | 0.401 / 0.248 / 0.167 / 0.106 |
| 0.3 | 25 / 50 / 100 / 250 | **0.405 / 0.307 / 0.240 / 0.157** | 0.696 / 0.374 / 0.267 / 0.163 | 0.584 / 0.352 / 0.257 / 0.160 |
| 0.5 | 50 / 250 | **0.390** (−7.5%) / 0.212 (−2.6%) | 0.555 (+10.2%) / 0.204 (+1.9%) | 0.543 (+4.8%) / **0.202** (+0.7%) |
| 0.8 | 50 / 250 | **0.533** (−19%) / **0.276** (−14%) | 0.895 (+19%) / 0.317 (+3.7%) | 0.941 (+18%) / 0.319 (+3.2%) |
| −0.1 | 250 | 0.081 | **0.070** | **0.070** |

**Decision.** `gpd="pwm"` is the default. It has the lowest RMSE in the operating regime:
the tails of standardised residuals typically have ξ ≈ 0–0.3, and a 1000-observation
window at 10% gives about 100 exceedances.

- **`"zs"` (Zhang & Stephens 2009).** Offered: it always exists, has no iterations, and
  is nearly unbiased from n_exc ≈ 250.
  - Implementation: m = 20 + ⌊√n⌋ grid points θ_j = 1/x_(n) + (1 − √(m/(j−½)))/(3x*),
    with profile weights and the posterior mean of θ.
  - Vectorised over `(L, m, k)`.
- **`"mle"`.** Grimshaw's (1993) one-dimensional profile root, offered for oracle parity
  only.
- **Known weakness, stated in the docs.** PWM's **downward** quantile bias for ξ ≥ 0.5
  under-states risk, and PWM's ξ is bounded above by 1 (already documented in `gpd_fit`).
  `gpd_xi` is emitted so a monitor can flag ξ̂ > 0.4.
- **Open question.** Should an `"auto"` rule ship, switching to ZS when ξ̂_PWM > 0.4? M4
  answers this with a VaR-coverage benchmark (99% and 99.5%, Kupiec and DQ from sibling
  1), not by opinion.

### F. Hawkes–POT (1T, 2T-POT, market→stock, power-law kernels)

**Events.**

- The thresholds `u← = Q̂_{a_u}` and `u→ = Q̂_{1−a_u}` are mirrored quantiles **of the
  fit window**. They use order statistics and are frozen for the whole refit segment.
  This is the 2T-POT construction, with "in-sample" replaced by "fit window".
- Exceedance at t: `M_t = u← − r_t > 0` or `r_t − u→ > 0`.
- The option `on="residuals"` defines events on GARCH z instead (a hybrid; documented as
  not the paper's model).
- A trailing-threshold feature must exclude row t from its own threshold window (trap T12).

**2T-POT with a common intensity** (Tomlinson et al., eqs. 9–17; notation from the paper):

```
chi_j(t)     = sum_{k: t_k^j < t} beta_j exp(-beta_j (t - t_k^j)) kappa_j(M_k),   j in {<-, ->}
kappa_j(M|t) = (1 - alpha_j ln[1 - F_GPD(M; xi_j, sigma_t^j)]) / (1 + alpha_j)
             = (1 + (alpha_j / xi_j) ln(1 + xi_j M / sigma_t^j)) / (1 + alpha_j)       [E kappa = 1]
lambda(t)    = mu + gamma_<- chi_<-(t) + gamma_-> chi_->(t)
sigma_t^j    = s_j + eta_j (lambda(t) - mu) / 2
p_t^j        = 0.5 (1 - exp(-Lambda_t)),    Lambda_t = integral_{t-1}^{t} lambda = mu + sum_j gamma_j (1 - e^{-beta_j}) S_j(t)
S_j(t+1)     = e^{-beta_j} S_j(t) + kappa_j(M_t) 1[event_t = j]
mu           = [2 - (gamma_<- + gamma_->)] a_lambda,  a_lambda = 2 a_u   (eq. 17; one fewer free parameter)
branching    n = (gamma_<- + gamma_->)/2 < 1
```

**Discrete-day likelihood.**

```
LL = sum_t  1[E_t = j] (ln p_t^j + ln f_GPD(M_t; xi_j, sigma_t^j))  +  1[E_t = none] (-Lambda_t)
```

With a common intensity, `P(no event) = 1 − p← − p→ = e^{−Λ_t}` exactly.

**Event-time evaluation, O(n_events) per lane** (Ozaki 1979):

- Between events S decays geometrically.
- The no-event days in a gap of length g contribute
  `−[g·μ + Σ_j γ_j(1−e^{−β_j}) S_j · (1−e^{−β_j g})/(1−e^{−β_j})]`, which is closed form.
- Measured: exact to 5e-15 against the day-by-day loop.

When `alpha=0` or `eta=0` the source term of S is independent of S, so the day-by-day
form is a linear recursion and §C.2's scan applies. With both active, κ depends on S
through σ_t, so the recursion is nonlinear: the lane loop runs over event index,
recommended with numba.

**Gradient.** Forward-mode tangents carried along the event recursion as a `(k+1, L)`
dual array (k=12 for the full 2T-POT). The Hessian is BHHH, with per-event plus per-gap
scores. The final sandwich standard errors use per-day scores, recomputed once.

**Starts.** Grid over β ∈ {0.02, 0.05, 0.1, 0.3} × n ∈ {0.3, 0.6} (8 lanes) at the first
refit, warm starts later.

**Predictions.** Tomlinson et al. eqs. 19–20:

- `VaR_{q,t+1} = u ∓ [(a_q/p^j_{t+1})^{−ξ} − 1] σ^j_{t+1}/ξ` when `p^j_{t+1} ≥ a_q`.
- ES follows eq. 20.
- When `p < a_q` the quantile lies inside the bulk. M5 emits NaN and sets
  `var_censored = True`. M6 adds the paper's subordinate Student-t bulk (eqs. 21–22),
  giving full support.

**Variants.**

- **Univariate 1T.** One tail, with `p_t = 1 − e^{−Λ_t}`.
- **Symmetric H₁.** Parameters tied across the two tails. It is the cheap default when
  there are few events.
- **Market→stock (M6).** `λ_i = μ_i + γ_ii χ_i + γ_im χ_m`. The market's events and
  states come from a single market series and are computed once, then broadcast to the
  lanes. The event recursion runs over the union of entity and market events.

**Power-law (Omori) kernel via sums of exponentials** (Bochud & Challet 2007†):

- `φ(t) ≈ Σ_{m=1}^{K} w_m β_m e^{−β_m t}`, with `β_m = β_0 c^{−(m−1)}` (c=5, K=8) and
  `w_m ∝ β_m^{θ}` normalised to Σw = 1.
- This gives `φ ~ t^{−1−θ}` between 1/β_0 and 1/β_K.
- The free parameters are (θ, β_0), not K rates. It keeps O(K·n_events) and all of the
  above.
- Test: the maximum relative error against the exact power law is ≤ 1e-2 over
  [1, 1000] days.

**Branching ratio.** Emitted per refit, with its sandwich SE. The docstring states four
things:

- It depends on the threshold and on the window.
- A non-stationary baseline biases it towards 1 (Filimonov & Sornette 2015; Wheatley et
  al. 2019).
- Exponential and power-law kernels give materially different values (Hardiman et al.
  2013 versus Filimonov & Sornette 2012).
- Daily bars merge intraday clusters.

It is never labelled "reflexivity" or "criticality" in any output.

**Estimation noise is the real constraint.** At `a_u=0.025` a 10-year window holds about
125 events per tail for 12 parameters. That is why the defaults are an expanding window
capped at 5000, `min_train=2500`, quarterly refits and the symmetric variant.
Panel-pooled kernels shared across entities are an open question (Risks).

**Budget, 5000 entities, quarterly refits, window 5000.** numpy ≤ 10 min, numba ≤ 1 min.
To be measured in M5; the only number measured so far is the 99 ms pass for an unmarked
kernel at 5000×2500.

### G. Systemic co-risk

**ΔCoVaR** (Adrian & Brunnermeier 2016).

- Per (entity, refit), two quantile regressions on the fit window:
  - `X^i_t = a^i + c^i M_{t−1}`;
  - `X^sys_t = a + c M_{t−1} + b X^i_t`.
- Then:
  - `VaR^i_{q,t} = â^i + ĉ^i M_{t−1}`;
  - `CoVaR = â + ĉ M_{t−1} + b̂ VaR^i_q`;
  - `ΔCoVaR = b̂ (VaR^i_q − VaR^i_{0.5})`.
- The state variables are lagged one date internally (trap T9).
- `exclude_self=True` builds a leave-one-out system return,
  `(Σ_j w_j r_j − w_i r_i)/(Σ_j w_j − w_i)`, with weights lagged one date. This is O(NT)
  vectorised and avoids the mechanical correlation of an entity with an index that
  contains it.
- CoES (optional) averages CoVaR over a grid of 5 levels ≤ q.

**QR solver** (`_qr.py`). We compared four:

- **(i) Barrodale–Roberts / Koenker–d'Orey AS 229 simplex.** The fastest exact method for
  n ≲ 5000 in compiled code, but inherently sequential pivoting with no batch form.
- **(ii) Frisch–Newton interior point** (Portnoy & Koenker 1997; quantreg `fn`) with the
  Mehrotra predictor–corrector. Each iteration is a p×p weighted Gram plus a Cholesky,
  so it **batches across lanes** as `(L, n, p)` with the `@` Gram.
- **(iii) Smoothed "conquer" QR** (He et al. 2023†). It changes the estimator (O(h²)
  bias) and only pays off at n ≥ 10⁴.
- **(iv) IRLS** (statsmodels). Approximate.

**Chosen: (ii)** in both backends, which gives one algorithm and exact numba/numpy
parity. It stops at a duality gap ≤ 1e-10·(1 + |obj|), then does a **vertex crossover**:

1. take the p observations with the smallest |residual|;
2. solve `X_h b = y_h` exactly;
3. accept the vertex if its check loss is ≤ the interior-point loss + 1e-12.

This returns the exact LP vertex, which is what BR returns when the solution is unique.
QR solutions can be non-unique, so parity is asserted on the **objective** everywhere and
on coefficients only for continuous simulated data.

Sizes: weekly data, n = 260–520 and p ≤ 9. The budget for 5000 entities × 80 quarterly
refits × 2 regressions is numpy ≤ 10 min and numba ≤ 1 min. To be measured in M7.

**MES, static** (Acharya et al. 2017).

- For each date t, the tail set is the `k = ⌊qW⌋` worst **market** days in
  `[t−W+1, t]`, found by order statistic with ties broken by earliest date.
- `MES_{i,t} = −mean_{s∈S_t} r_{i,s}`, averaged over available observations with a
  `min_count` gate.
- The tail sets come from one series (a `(T, W)` argpartition). The entity gather is
  chunked over T. Target ≤ 5 s for 5000×5000.

**Pairwise DCC** (Engle 2002), for each (entity, market) pair.

- Step 1 is the §C GARCH for both series, fitted **inside** the DCC wave on the same fit
  window. Live `z` columns built from mixed refits must not be reused as DCC fit input
  (trap T18).
- Step 2 runs `Q_t = (1−a−b)Q̄ + a z_{t−1}z'_{t−1} + b Q_{t−1}` with Q̄ equal to the
  fit-window correlation of z. The three entries of Q follow linear recursions with
  coefficient b, so the §C.2 scan applies.
- `ρ_t = q12/√(q11 q22)`, and the correlation log-likelihood has an exact gradient and
  Hessian by the chain rule.
- `variant="cdcc"` (Aielli 2013†) is consistent but nonlinear, so it uses the lane loop
  or numba. The default is `"dcc"` for rmgarch parity. `(a, b)` is reparameterised like
  GARCH.

**MES, dynamic** (Brownlees & Engle 2017, C = −2% daily):

```
MES_{i,t+1} = σ_i ρ E[ε_m | ε_m < C/σ_m] + σ_i √(1−ρ²) E[ξ_i | ε_m < C/σ_m]
ξ_i = (ε_i − ρ ε_m) / √(1−ρ²)
```

- The tail expectations are kernel-weighted means over fit-window residuals, with
  weights `w_s(κ_t) = Φ((κ_t − ε_{m,s})/h)` and h frozen per refit (a fit).
- The key saving is that **the weights depend only on the market**. The entity side is
  therefore one masked matmul per refit segment,
  `(N×W)@(W×T_seg) / (mask@w)`, about 1e8 flops per refit. A naive N×T×W loop would be
  2.5e10.

**LRMES.**

- `"beta"` (the V-Lab form†): `1 − exp(ln(1−d)·β_{i,t})` with `β = ρσ_i/σ_m` and d = 40%.
  This is the default.
- `"mes18"` (Acharya, Engle & Richardson 2012†): `1 − exp(−18·MES)`.
- `"simulate"` (Brownlees & Engle): seeded FHS of the bivariate GJR-DCC, h = 22 days,
  C = −10%, 10⁴ paths. Computed only at refit dates or on request; numba recommended.

**SRISK.**

- `SRISK_{i,t} = k·D_{i,t} − (1−k)·W_{i,t}·(1 − LRMES_{i,t})`, with k = 0.08.
- W is the market capitalisation at t.
- D is book liabilities **as known at t**, via `asof_join` on `knowledge_time` (for
  example a filing date) or an explicit `lag`. With neither, the call **raises** (trap T8).
- `srisk_aggregate` works per date: `Σ_i max(SRISK_i, 0)` and shares. It is
  cross-sectional, so it is safe.

### H. Drawdowns

All of these are per entity in time order. The log-price is `x = ln P`, or
`cum_sum(ln(1+r))`.

- **`drawdown`** is `P / cum_max(P) − 1`, a polars expression. With `window=d` it is
  `P / rolling_max(P, d) − 1`, identical to evolve's `ts_drawdown`.
- **`drawdown_duration`** is the row index minus the forward-filled index of the last row
  where `P == cum_max(P)`. It uses `int_range` positions, which are prefix-stable, or
  time differences when `unit="time"`.
- **`rolling_max_drawdown`** is `min over u ≤ s in [t−w+1, t] of (x_s − x_u)`. It uses the
  associative (max, min, drop) monoid in the style of van Herk / Gil-Werman:
  - Split the series into blocks of w anchored at position 0.
  - Within each block compute
    - the prefix running max, running min and drop, via `cummin(x − cummax x)`;
    - the suffix min, suffix max and drop, via a reverse `cummin(sufmin − x)`.
  - Answer each window as `min(sufdrop[i], predrop[j], premin[j] − sufmax[i])`, or
    `predrop[j]` when the window is block-aligned.
  - It is O(T), fully vectorised over `(N, n_blocks, w)`. Min and max are exact, and the
    single subtraction is the same one brute force makes, so it is **bitwise equal to
    brute force** (measured). It is prefix-stable because the only values truncation
    changes belong to windows that have not ended yet.
- **CDaR_α** (Chekhlov, Uryasev & Zabarankin 2005) is the CVaR of the **window-anchored**
  drawdown path over the trailing window: the drawdowns are relative to the running max
  since the window start, which is their definition applied to each window. It is O(T·w)
  via `sliding_window_view`, a running max along the window axis and `np.partition`.
  It is chunked over T. Target ≤ 60 s for 5000×5000 at w=252.
  `basis="expanding"` (drawdowns against the all-time high, then a trailing tail mean)
  is offered as an O(T) alternative, documented as a different quantity.
- **Ulcer index** (Martin 1987†) is
  `sqrt(rolling_mean((100·(P/rolling_max(P, w) − 1))², w))`, pure polars.
- **Calmar** (Young 1991†) is `annualised mean log return over the trailing window /
  |rolling_max_drawdown|`, with NaN when MDD = 0.
- **Expected MDD of Brownian motion** (Magdon-Ismail, Atiya, Pratap & Abu-Mostafa 2004):
  - μ = 0: `E[D] = 2γσ√T = √(π/2)·σ√T`, with γ = √(π/8).
  - μ > 0: `E[D] = (2σ²/μ) Q_p(α²)`.
  - μ < 0: `E[D] = −(2σ²/μ) Q_n(α²)`.
  - Here `α = μ√(T/(2σ²))`, with the asymptotes `Q_p → ¼ ln x + 0.49088` and
    `Q_n → x + ½`.
  - Q_p and Q_n are **tabulated once by a committed generator script**
    (`benchmarks/gen_emdd_table.py`). The script evaluates the paper's eigen-series (eq.
    17–19) in high precision, validates against the paper's Appendix B at 1e-4, and emits
    the constant array in `_emdd_table.py`.
  - Lookup is monotone cubic interpolation in `ln x`, with the asymptotes outside the
    table.
  - `mdd_surprise = realised rolling MDD / E[MDD | μ̂, σ̂, w]`, where μ̂ and σ̂ come from the
    same trailing window. `mu="zero"` is the robust default, because μ̂ is noise at w=252.

### I. Optional late milestones

**N-dimensional DCC by composite likelihood** (Pakel, Shephard, Sheppard & Engle 2021,
JBES 39(3)).

- Sum the pair log-likelihoods over contiguous pairs (i, i+1) with a common (a, b). This
  reuses the pairwise kernel. Estimation is O(N·T) instead of O(N²T) plus N×N inversions.
- R_t is materialised only at requested dates; at N=5000 the full path is 25M entries per
  date.
- The Q̄ target and the output container come from sibling 3.
- **DCC-NL** (Engle, Ledoit & Wolf 2019†): the same recursion with sibling 3's
  `nonlinear_shrinkage` (QIS) for the correlation target, built jointly with them.

**APARCH** (δ free; parity with arch's APARCH), and **GARCH-X** (an exogenous realised
measure term, still linear, so the scan applies).

**MSM** (Calvet & Fisher 2004): M10, gated on sibling 7's filter.

---

## Leak-safety design and traps

The safe path is the default everywhere. Each trap gets a deliberately leaky reference
implementation in `tests/test_risk_leakage.py`. **The verifiers must catch it.** That is
what proves the tests have power.

| # | Trap | Safe default in this plan |
|---|---|---|
| T1 | GARCH, Hawkes or QR fitted on the full sample; row-level σ_t read in-sample (measured: σ_t moves 4.5% median) | Refits on data `< d` only; in-force parameters by date |
| T2 | Refit dates from end-anchored walk-forward geometry (measured) | Calendar-anchored `RefitSchedule`; a test pins `walk_forward_splits` as unusable |
| T3 | POT thresholds or quantiles over the full sample | Fit-window order statistics, frozen per segment |
| T4 | Variance targeting with the full-sample variance: leaks the unconditional level | s² from the fit window |
| T5 | Backcast or filter initialised from full-sample or future rows. This includes `transform` on a test fold that backcasts from **the test fold's first 75 rows** | Terminal state of the fit pass. `refit="train"` carries train's terminal state, else ω/(1−P) with no data |
| T6 | Evaluation misalignment: `sigma_next` or `var_next` at row t compared with r_t, not r_{t+1}. The hit rate falls and coverage looks too good | Invariant 11 naming. Sibling 1's backtests take `shift(1)` explicitly; a test pins it |
| T7 | McNeil–Frey GPD on residuals from full-sample parameters | GPD refitted inside each wave on in-window z |
| T8 | SRISK liabilities keyed by period end, not filing: 40–90 days of look-ahead | `asof_join` on `knowledge_time` or `lag=`; raise otherwise |
| T9 | ΔCoVaR state M_t used contemporaneously; a system index that contains the entity | M lagged one date internally; `exclude_self=True` |
| T10 | MES tail days picked by a full-sample market quantile | Trailing window per date |
| T11 | Parameters pooled across entities over the whole sample | Any pooling (a future option) happens inside the wave's fit window |
| T12 | An exceedance threshold that includes the current row, so r_t normalises its own event. This is the `spot_variance(include_current=False)` argument | Thresholds come from `time < d`. Trailing variants exclude t |
| T13 | Mixing backends between runs: numba and numpy differ at 1e-13 | `backend` recorded in the refit log; prefix invariance is asserted per backend |
| T14 | `np.sum` over a span that grows with T (pairwise summation) | Invariant 8 |
| T15 | Returns rescaled or winsorised by full-sample statistics (`arch`'s rescale advice uses the whole y) | Fit-window s_w only |
| T16 | Returns demeaned by the full-sample mean | Fit-window mean |
| T17 | Simulation RNG consumption depends on the number of entities, so batch composition changes results | Per-(seed, entity, refit) `SeedSequence` |
| T18 | DCC or dynamic MES fitted on live `z` columns stitched from different refits (inconsistent two-step; not a leak but silently wrong) | Both steps are refitted inside the same wave |
| T19 | A refit that fails to converge silently emits garbage | `on_fail="previous"`; flagged in the log |

---

## Tests

Every file uses `--strict-markers`. `slow` and `benchmark` are already registered.

- **`test_risk_schedule.py`.**
  - Refit dates are invariant under truncation on Date and Int axes; embargo and
    `min_train` are respected.
  - A **regression test showing `walk_forward_splits` moves under appended rows**,
    documenting why it is not reused.
  - `schedule_hash` is stable.
- **`test_risk_optim.py`.**
  - Batched LM on known problems (a batched Rosenbrock, GARCH at known optima).
  - The hand-rolled Cholesky against `np.linalg` for k = 1..8, including **p == batch
    size**.
  - **Batch-composition invariance**: one entity alone equals the same entity inside 50,
    and `chunk_entities` 1 equals 1024, both bitwise.
  - Frozen lanes never change after convergence.
- **`test_risk_garch.py`.**
  - Scan against lane loop, relative 1e-13, including at β = 1e-8 and 0.9999.
  - Analytic gradient and Hessian against central differences (1e-6) and against the
    numba kernel (1e-12).
  - `_dists` logpdf, ppf and ES against `scipy.stats` (importorskip) at the stated
    tolerances.
  - Closed-form h-step forecasts against a 200k-path seeded Monte Carlo.
  - **Simulation recovery**: GARCH(1,1)-t(6) with (0.02, 0.08, 0.90), T = 5000, 200 seeds.
    The mean estimates are within 2 Monte Carlo SE of the truth, and the empirical 95%
    coverage of the sandwich CIs is in [0.92, 0.98]. Marked `slow`.
  - Recovery repeated for GJR skew-t and EGARCH.
- **`test_risk_oracle.py`** (all behind `pytest.importorskip`).
  - **arch** (NCSA; permissive): GARCH, GJR, EGARCH × {Normal, StudentsT, SkewStudent}
    with `mean="Zero"` on demeaned data and variance targeting off.
    - With identical parameters, the σ_t filter matches at relative 1e-12. This pins
      the backcast convention.
    - At the optimum our LL is ≥ arch's LL − 1e-6·|LL|.
    - Parameters agree within 1e-3 relative on n ≥ 2000 away from boundaries.
    - arch's `VarianceTargetingGARCH`† is used when present.
  - **rugarch** (GPL: outputs only, never read its source): JSON fixtures in
    `tests/data/risk/` store the simulated series, R and rugarch versions, parameters and
    LL for sGARCH / gjrGARCH / eGARCH with `variance.targeting=TRUE` and FALSE. Tolerance
    1e-4 on parameters and 1e-6 relative on LL. The generator script is committed, but R
    is not needed in CI.
  - **tick** (BSD†; often uninstallable): exponential-Hawkes LL parity when present.
    The **always-on oracle** is our own O(n²) brute-force continuous-sum likelihood.
  - **QR**: `scipy.optimize.linprog` (HiGHS) objective at 1e-9, statsmodels QuantReg at
    1e-4 (it is IRLS), and quantreg `rq(method="br")` fixtures (GPL, outputs only).
  - **GPD**: `scipy.stats.genpareto.fit` for `method="mle"`.
- **`test_risk_leakage.py` — the contract.**
  - For every public function: `assert_no_lookahead` and `assert_prefix_invariant`, with
    cuts **exactly at a refit date, one row before and one row after, and mid-segment**.
  - Append 120 future rows per entity and compare bitwise, following
    `test_econ_features_leakage.py`.
  - Run under each available backend.
  - Leaky references T1, T3, T4, T5, T7, T8, T9 and T10 must be **detected**.
  - Numba thread count 1 against default, bitwise.
- **`test_risk_evt.py`.**
  - The batch PWM kernel equals `gpd_fit(threshold=u)` at 1e-14.
  - The order-statistic threshold gives exactly k exceedances.
  - The ZS implementation reproduces the posterior-mean identity on hand cases.
  - `gpd_fit`'s default output is **unchanged** (a pinned golden value).
- **`test_risk_hawkes.py`.**
  - Event-time LL equals the day-by-day LL (1e-12) and the O(n²) brute force.
  - Closed-form gap sums.
  - Eq. 17 reparameterisation: the empirical event rate equals a_u per tail in the fit
    window.
  - Eq. 19–20 quantile and ES against numeric integration of the model's conditional law.
  - **Simulation recovery** via exact discrete-day simulation (draw events from p_t, marks
    from the GPD, update S; seeded). Recovery of known (γ, β, α, ξ, ς, η) within 2 MC SE
    at W = 10000. Marked `slow`.
  - Sum-of-exponentials against the exact power law (≤ 1e-2 on [1, 1000]).
  - Branching ratio stays < 1 by construction.
- **`test_risk_systemic.py`.**
  - QR vertex crossover gives the exact vertex (objective against linprog).
  - `exclude_self` algebra.
  - MES static against brute force.
  - Dynamic MES matmul form against the naive triple loop on a small panel.
  - DCC gradient against finite differences.
  - SRISK arithmetic.
  - SRISK **raises** without `knowledge_time` or `lag`.
  - LRMES `"simulate"` is reproducible under a permuted entity order (T17).
- **`test_risk_drawdown.py`.**
  - Expanding drawdown and duration against brute force.
  - `rolling_max_drawdown` **bitwise** equal to brute force on random walks, constant
    series, monotone series and NaN gaps.
  - `.ts.drawdown(window=d)` bitwise equal to evolve's `_b_ts_drawdown`.
  - Ulcer index against the definition.
  - E[MDD]: μ = 0 closed form; table against the paper's Appendix B (1e-4); against a
    seeded Monte Carlo (≤ 1% after the discrete-monitoring correction, which is an open
    question).
- **`test_risk_coverage.py`** (`slow`).
  - VaR/ES coverage sanity with sibling 1's Kupiec, Christoffersen and DQ if importable,
    else a test-local binomial check.
  - A correctly specified GARCH-t on 50 × 4000 simulated rows: the 1% hit rate is within
    ±3 binomial SE.
  - **A misspecified Gaussian on t(4) data must fail**, which proves power.
  - The same for GARCH-EVT and 2T-POT at 1% and 0.5%.
- **Registry conformance**: `_FRAME_OPS` builders for every `risk` spec; the `ts`
  drawdown ops are exercised automatically.
- **Guardrails**: `test_import_hygiene.py` (no top-level numba or scipy),
  `test_dependency_drift.py` (**zero** new mandatory dependencies) and
  `test_wheel_guardrails.py`, all unchanged.

---

## Benchmarks and performance budgets

`benchmarks/bench_risk.py` is seeded and reports numpy and numba separately.
`tests/test_risk_perf.py` (`benchmark`) asserts no regression beyond 2×.

| Workload (5000 entities × 5000 days unless stated) | Measured now | Budget: numpy | Budget: numba, 15 threads |
|---|---|---|---|
| GARCH(1,1) LL + gradient, 5000 lanes × 1000 obs | 78–103 ms / 22 ms serial / 3.2 ms | — | — |
| `garch`, t, variance targeting, monthly refits, cap 1000, incl. filter and VaR | — | ≤ 5 min | ≤ 30 s |
| `garch`, EGARCH-t, same | — | ≤ 10 min | ≤ 60 s |
| `garch_evt`, including the GPD per wave | — | + ≤ 20 s over `garch` | + ≤ 5 s |
| `hawkes_pot` 2T, quarterly, window 5000 | unmarked LL pass 99 ms (5000×2500) | ≤ 10 min | ≤ 60 s |
| `delta_covar`, weekly T=1000, quarterly, W=260, p=4 | — | ≤ 10 min | ≤ 60 s |
| `mes` static, W=252 | — | ≤ 5 s | — |
| `mes` dynamic plus `dcc_pairwise`, monthly | — | ≤ 8 min | ≤ 45 s |
| `rolling_max_drawdown`, w=252 | 1.3 s (per-entity loop) | ≤ 1.5 s vectorised | — |
| `cdar`, window-anchored, w=252 | — | ≤ 60 s | — |
| Peak memory, `chunk_entities=1024` | — | ≤ 1 GB beyond the input | same |

Every "—" becomes a measured number in its milestone. The budgets were derived from the
measured kernels above. Where a measurement misses its budget by more than 2×, the plan
is amended with the measured number rather than quietly relaxed.

---

## Dependencies

- **Mandatory:** none new (numpy and polars).
- **Optional:** `fast` (numba) for kernels, which already exists. scipy is **not** used
  on any production path.
- **Test-only**, behind `importorskip`: `arch` (NCSA), `scipy`, `statsmodels` (BSD-3)
  and `tick` (BSD†).
- **Offline fixtures** from R `rugarch` and `quantreg`, both GPL. We store their outputs
  only, never code, and we write clean-room from the papers.
- `registry.audit()` enforces `license="Apache-2.0"` on every spec.

---

## Milestones

Each milestone is shippable alone. **Gate: AGENTS.md. M2 and later start only once the
engine names a caller with a test** (see Caller). Within that gate the order is fixed.

| M | Contents | Ships | Exit criterion |
|---|---|---|---|
| **M1** | `_drawdown.py`, `namespaces/ts_risk.py` (`drawdown`, `drawdown_duration`, `rolling_max_drawdown`, `ulcer_index`), `drawdown_features` (CDaR, Calmar, E[MDD] with the generated table), registry specs, docs | Causal drawdown features and metrics; zero dependencies | Bitwise brute-force parity; conformance green; the engine caller's golden test passes |
| **M2** | `core/_refit.py` `RefitSchedule` (coordinated with siblings 3 and 7), `_optim.py`, `_scan.py`, `_internal/_special.py`, `_garch.py` GARCH and GJR × normal and t, variance targeting, pure numpy; `garch()` and `GARCH` | Honest-refit conditional volatility, σ, z, PIT, one-step VaR/ES | Leak suite, arch parity and recovery green; decide variance-targeting default and exact Hessian against BHHH by benchmark |
| **M3** | Skew-t, EGARCH, h-step forecasts and seeded FHS, `_numba.py` kernels for everything so far, rugarch fixtures | Full univariate family; numba backend | numba/numpy ≤ 1e-12; budgets met or plan amended |
| **M4** | `_cevt.py` GARCH-EVT, batch GPD kernel, `gpd_fit(method=)` with the default unchanged | Conditional EVT | Batch equals `gpd_fit`; coverage benchmark decides the `"auto"` GPD question |
| **M5** | `_hawkes.py` 1T and 2T-POT with exponential kernel, marks, η, eq. 17 reparameterisation, event-time LL; `hawkes_pot` and `HawkesPOT` | Self-exciting tail probabilities, VaR/ES (censored), branching ratio with caveats | Brute-force LL parity, recovery, leak suite |
| **M6** | 2T-POT subordinate Student-t bulk (full support), market→stock excitation, power-law sum-of-exponentials kernel | Uncensored VaR/ES; contagion from the market | Sum-of-exponentials accuracy; coverage against GARCH-EVT reproduces the paper's direction on index data (documented, not asserted) |
| **M7** | `_qr.py` batched Frisch–Newton with crossover; `delta_covar` (+CoES); `mes` static | Adrian–Brunnermeier co-risk, leak-safe | linprog objective parity; T9 and T10 detection |
| **M8** | `dcc_pairwise` (DCC and cDCC), dynamic `mes`, LRMES (beta, mes18, simulate), `srisk`, `srisk_aggregate` | Brownlees–Engle SRISK with point-in-time liabilities | T8 raises; seeded simulation reproducible under entity permutation |
| **M9** (optional) | N-dimensional DCC by composite likelihood; DCC-NL (with sibling 3's QIS); APARCH; GARCH-X | Coordinated with sibling 3 before starting | — |
| **M10** (optional) | MSM volatility on sibling 7's forward filter | Only if sibling 7 ships a structured-transition filter hook | — |

---

## Risks and open questions

Each of these is resolved by a benchmark, not an opinion.

1. **Variance-targeting default.** Targeting drops a parameter and is robust on short
   windows; full QMLE is more efficient asymptotically (Francq, Horváth & Zakoïan 2011†).
   M2 compares out-of-sample QLIKE (sibling 1) on simulated and real panels; the default
   follows.
2. **Exact Hessian or BHHH.** Iteration counts and wall time, measured in M2.
3. **Scan/lane-loop crossover** was measured only on Apple silicon (about 256 lanes).
   Re-measure on x86 CI hardware; the threshold is a constant in `_scan.py`, not
   hard-coded at call sites.
4. **`"auto"` GPD rule** (M4 coverage benchmark).
5. **Hawkes estimation noise.** About 125 events per tail for 12 parameters. Measure
   parameter RMSE by simulation at W ∈ {2500, 5000, 10000}. If it is unusable, add
   **panel-pooled kernels** (shared β, α, η within a group, entity-specific μ). Pooling
   is leak-safe only inside the wave's fit window (T11).
6. **Discrete-day 2T-POT versus the paper's continuous λ(t).** We use the left-limit λ(t)
   in eq. 16 and Λ_t for probabilities. Measure the effect against a continuous-time
   refit on index data.
7. **arch parity near α = 0** is judged on log-likelihood only, since our reparameterised
   boundary is not reachable exactly.
8. **Discrete-monitoring bias** of realised MDD against E[MDD] of continuous Brownian
   motion. The Broadie–Glasserman–Kou-style shift of 0.5826·σ√Δt† is a candidate
   correction, to be validated by Monte Carlo before `mdd_surprise` ships with it.
9. **`evt_features` vectorisation** (19 min for 5000 entities) is tempting but out of
   scope. It is allowed only with bitwise parity to the current `np.quantile` threshold
   semantics.
10. **Naming.** `pn.risk` is claimed here. If sibling 1 wants `risk` for backtests,
    resolve before M1 (proposal: backtests live in `pn.validation`).
11. **Ownership of the refit scheduler.** Sibling 7's `ss.Refit` and sibling 3's §5.3
    schedule overlap with ours. The orchestrator resolves this before any M2 (see
    "Shared primitives"). The dense adaptor has the same problem: `build_tensor` against
    sibling 7's `_dense.py`.
12. **Dense `(N, T)` pivot** is 200 MB per column at 5000×5000. Larger panels chunk over
    entities; the pivot is not chunked over time, because waves need the whole history
    before d.
13. **numba on new Pythons.** The `fast` extra already carries this risk. The numpy path
    is complete, so a numba gap is a speed regression, never a feature gap.

---

## Caller

**AGENTS.md: "No new public surface without a named caller in the engine, and a test in
the engine that exercises it."** This plan is honest about where it stands.

- **M1 (drawdowns): candidate caller
  `truepoint/src/truepoint/generate/family_computation.py`.**
  - The COMPUTATION family (metric `COMPUTATION_FIDELITY`) has three templates today:
    `return_window`, `agency_fy_obligation` and `weekly_sum`.
  - Templates such as `max_drawdown_window` or `time_under_water` need **one fixed,
    documented definition** so a gold answer is reproducible. The definition has to pin
    down the ratio convention, whether the window is inclusive, and the tie at the peak.
  - `pn.risk` supplies that definition, and the engine test is a golden-answer test for
    the new template.
  - The engine's firewall forbids product imports only in `diagnose/` (and LLM or
    generate imports in `score/`), so `generate/` may import it. The engine owner must
    confirm.
  - Gold answers are computed from ledger facts. Nothing here touches sealed questions.
- **M2–M4 (GARCH, GARCH-EVT): no caller today.** The natural home is
  `truepoint/src/truepoint/quant/`, which AGENTS.md names but which **does not exist
  yet**. A plausible use is a check that recomputes a system's volatility or VaR claim
  under the causal reference and the leaky full-sample reference, and reports which one
  it matches, as `Evidence(kind=STATIC_ANALYSIS)`. **Do not start M2 until that check is
  specified in the engine with a test.**
- **M5–M8 (Hawkes, co-risk, SRISK): no caller.** These are research-grade and blocked by
  the rule. They are listed so that the design (the shared schedule, optimiser and scan)
  does not have to be reinvented when a caller appears.

---

## References

A † marks a reference that was not re-verified in this session.

- Adrian, T. & Brunnermeier, M. K. (2016). CoVaR. *American Economic Review* 106(7), 1705–1741.
- Acharya, V., Pedersen, L., Philippon, T. & Richardson, M. (2017). Measuring systemic risk. *Review of Financial Studies* 30(1), 2–47.
- Acharya, V., Engle, R. & Richardson, M. (2012). Capital shortfall: a new approach to ranking and regulating systemic risks. *AER Papers & Proceedings* 102(3), 59–64. †
- Aielli, G. P. (2013). Dynamic conditional correlation: on properties and estimation. *JBES* 31(3), 282–299. †
- Aït-Sahalia, Y., Cacho-Diaz, J. & Laeven, R. (2015). Modeling financial contagion using mutually exciting jump processes. *JFE* 117(3), 585–606. †
- Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *J. Futures Markets* 19(5). †
- Bochud, T. & Challet, D. (2007). Optimal approximations of power laws with exponentials. *Quantitative Finance* 7(6), 585–589. †
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31, 307–327.
- Bollerslev, T. & Wooldridge, J. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews* 11(2), 143–172.
- Brownlees, C. & Engle, R. (2017). SRISK: a conditional capital shortfall measure of systemic risk. *Review of Financial Studies* 30(1), 48–79.
- Calvet, L. & Fisher, A. (2004). How to forecast long-run volatility: regime switching and the estimation of multifractal processes. *J. Financial Econometrics* 2(1), 49–83.
- Chavez-Demoulin, V., Davison, A. C. & McNeil, A. J. (2005). Estimating value-at-risk: a point process approach. *Quantitative Finance* 5(2), 227–234.
- Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005). Drawdown measure in portfolio optimization. *IJTAF* 8(1), 13–58.
- Ding, Z., Granger, C. & Engle, R. (1993). A long memory property of stock market returns and a new model. *J. Empirical Finance* 1, 83–106.
- Engle, R. (2002). Dynamic conditional correlation. *JBES* 20(3), 339–350.
- Engle, R. & Mezrich, J. (1996). GARCH for groups. *Risk* 9(8). †
- Filimonov, V. & Sornette, D. (2012). Quantifying reflexivity in financial markets: toward a prediction of flash crashes. *Physical Review E* 85, 056108.
- Filimonov, V. & Sornette, D. (2015). Apparent criticality and calibration issues in the Hawkes self-excited point process model. *Quantitative Finance* 15(8), 1293–1314. †
- Francq, C., Horváth, L. & Zakoïan, J.-M. (2011). Merits and drawbacks of variance targeting in GARCH models. *J. Financial Econometrics* 9(4), 619–656. †
- Glosten, L., Jagannathan, R. & Runkle, D. (1993). *Journal of Finance* 48(5), 1779–1801.
- Grimshaw, S. (1993). Computing maximum likelihood estimates for the generalized Pareto distribution. *Technometrics* 35(2), 185–191. †
- Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review* 35(3), 705–730.
- Hansen, P. R. & Lunde, A. (2005). A forecast comparison of volatility models: does anything beat a GARCH(1,1)? *J. Applied Econometrics* 20(7), 873–889.
- Hardiman, S., Bercot, N. & Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. *EPJ B* 86, 442 (arXiv 1302.1405).
- He, X., Pan, X., Tan, K. M. & Zhou, W.-X. (2023). Smoothed quantile regression with large-scale inference. *J. Econometrics* 232(2), 367–388. †
- Hill, G. W. (1970). Algorithm 396: Student's t-quantiles. *CACM* 13(10). †
- Hosking, J. & Wallis, J. (1987). Parameter and quantile estimation for the generalized Pareto distribution. *Technometrics* 29(3), 339–349.
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1), 33–50.
- Koenker, R. & d'Orey, V. (1987). Algorithm AS 229. *Applied Statistics* 36(3), 383–393. †
- Magdon-Ismail, M., Atiya, A., Pratap, A. & Abu-Mostafa, Y. (2004). On the maximum drawdown of a Brownian motion. *J. Applied Probability* 41(1), 147–161.
- Martin, P. & McCann, B. (1989). *The Investor's Guide to Fidelity Funds* (Ulcer Index). †
- McNeil, A. J. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *J. Empirical Finance* 7(3–4), 271–300.
- Mehrotra, S. (1992). On the implementation of a primal-dual interior point method. *SIAM J. Optimization* 2(4), 575–601.
- Nelson, D. (1991). Conditional heteroskedasticity in asset returns: a new approach. *Econometrica* 59(2), 347–370.
- Nielsen, H. B. (1999). Damping parameter in Marquardt's method. IMM-REP-1999-05, DTU. †
- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Trans. Information Theory* 27(1), 23–31.
- Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Annals of the Institute of Statistical Mathematics* 31(1), 145–155.
- Pakel, C., Shephard, N., Sheppard, K. & Engle, R. (2021). Fitting vast dimensional time-varying covariance models. *JBES* 39(3), 652–668.
- Portnoy, S. & Koenker, R. (1997). The Gaussian hare and the Laplacian tortoise. *Statistical Science* 12(4), 279–300.
- Tomlinson, M. F., Greenwood, D. & Mucha-Kruczyński, M. (2024). 2T-POT Hawkes model for left- and right-tail conditional quantile forecasts of financial log returns: out-of-sample comparison of conditional EVT models. *International Journal of Forecasting* (arXiv 2202.01043; eqs. 9–22 read from the arXiv PDF).
- van Herk, M. (1992); Gil, J. & Werman, M. (1993). O(1)-per-element running max/min filters. †
- Wheatley, S., Wehrli, A. & Sornette, D. (2019). The endo–exo problem in high frequency financial price fluctuations and rejecting criticality. *Quantitative Finance* 19(7), 1165–1178. †
- Young, T. W. (1991). Calmar ratio: a smoother tool. *Futures* 20(1). †
- Zhang, J. & Stephens, M. A. (2009). A new and efficient estimation method for the generalized Pareto distribution. *Technometrics* 51(3), 316–325.
- V-Lab SRISK documentation and the `frds` SRISK page: the LRMES "beta" form and the parameter conventions. †
