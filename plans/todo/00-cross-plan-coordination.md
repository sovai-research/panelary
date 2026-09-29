# Cross-plan coordination: the ten build contracts of 2026-09-29

> **Build status (2026-09-29, end of day):** plans 1, 4 and 5 are implemented (optional
> M7s aside) and moved to `plans/done/`; plan 3 has M1–M4. Shared plumbing shipped:
> D1 `core/_schedule.py` (by plan 3, with a bit-exact `anchor="position"` mode for
> `residualise`), D3 `_internal/_jit.py` (the CUSUM kernel migrated onto it), D4
> `_internal/_special.py`, D5 `validation/_hac.py` and `_resample.py`, D6 `_internal/_linalg.py`,
> `_ohlc.py`, `_realized_kernel.py` and `_variogram.py` (built by plan 5 because plan 8 has not
> started; plan 8 reuses it). Bugs B1–B6 and B11 are fixed; B7–B10 remain. New issues found
> during the build are listed under "Bugs found in shipped code" (B12–B19).
>
> **Status (2026-09-29): decisions only, no code.** Ten plans were written in parallel on
> 2026-09-29, each without seeing most of the others. They propose overlapping shared
> infrastructure and a few competing edits to the same files. **Where this note and a plan
> disagree, this note wins.** Each plan links here under its title.

## The ten plans

| # | Plan | New home | Engine caller today | Headline finding (measured in the plan) |
|---|---|---|---|---|
| 1 | [forecast-evaluation-and-sharpe-inference](../done/forecast-evaluation-and-sharpe-inference.md) | `validation/` (new private modules) | **Yes, for M2**: truepoint's M07 calibration uses binned ECE; CORP replaces it | Jobson–Korkie over-rejects 8.5–8.9% at 5%; asymptotic Kupiec 9.7% at T=250; DM on nested models 0.3% vs Clark–West 5.5% |
| 2 | [drift-monitoring-and-sequential-inference](drift-monitoring-and-sequential-inference.md) | `monitor/` | No | PSI > 0.1 fires 79% under no drift at n=100; KS flags 73% on a φ=0.9 feature; peeking at a Wilson CI misses 53.5% |
| 3 | [covariance-and-market-state](covariance-and-market-state.md) | `covariance/` | No (proposed: `generate/family_computation.py`) | Sample covariance min-variance risk 2,095× optimal at N=500, T=252; dual-Gram path 3.3 ms/date at N=3,000 vs 1.85 s |
| 4 | [label-weights-and-event-sampling](../done/label-weights-and-event-sampling.md) | `label/`, `core/_spans.py`, `sample/` | No (M1 hardens existing purge) | Vectorised purge identical and up to 800× faster; exact sequential bootstrap 90 ms (numba) at 100k labels |
| 5 | [ohlc-volatility-and-liquidity](../done/ohlc-volatility-and-liquidity.md) | `econ/features/` | No (proposed: `generate/family_computation.py`) | Yang–Zhang ≈7–8× close-to-close efficiency; EDGE matches `bidask` to 4e-15; staged polars 554 s → 6.2 s |
| 6 | [network-and-spatial-panel](network-and-spatial-panel.md) | `network/` | No (M1 has a proposed caller) | Own-lag residualising leaves the Burt–Hrdlicka commonality (t 4.6); peer-trailing-mean control cuts it to 1.6 |
| 7 | [causal-state-space-and-regimes](causal-state-space-and-regimes.md) | `statespace/` | No (proposed: `generate/family_asof.py`) | Smoothed HMM labels "beat" filtered 94.2% vs 90.7% — all look-ahead; direct-form Butterworth order 8 blows up to 2.6e22 |
| 8 | [multiscale-complexity-features](multiscale-complexity-features.md) | `complexity/` | No (M0 gate) | Haar wavelet-variance Hurst: sd 0.042–0.046 at W=256, unbiased under drift/fat tails; DFA prefix-sum shortcut errs 1.6e-3 |
| 9 | [tail-risk-and-self-excitation](tail-risk-and-self-excitation.md) | `risk/` | M1 (drawdowns) only | Full-sample GARCH moves earlier σ_t by a median 4.5% on append; batched GARCH 22 ms (numba, 1 thread) for 5,000×1,000 |
| 10 | [panel-causal-inference](panel-causal-inference.md) | `counterfactual/` | No | TWFE recovers 6% of a planted staggered effect (0.098 vs 1.759); BMP rejects 43.9% under clustered events, KP 5.8% |

All seven new packages are **lazy** (added to `_LAZY_SUBMODULES` in `panelary/__init__.py`),
so none of them touches the cold-import budget.

## Shared-infrastructure decisions

**D1. One refit/rebuild schedule, in `panelary/core/_schedule.py`.** Plans 3 (`cov.Schedule`),
7 (`ss.Refit`), 9 (`RefitSchedule`, proposed at `core/_refit.py`), 2 (reuses plan 3's) and 6
(`rebuild=` durations) all need the same primitive. Ship **one**:

- `Schedule(every=...)`: *which* dates are refit/rebuild dates. An integer `k` means index ≡ 0 (mod k)
  counted from the panel's first date. A calendar period means the **first** date of each period.
  End-anchored schedules and "last date of the month" are forbidden, because both need t+1.
- `Refit(schedule, window=None, min_train=..., lag=0, burn=...)`: the fitting policy layered on a
  `Schedule` (plans 7 and 9 need window, min_train and lag).
- Durations are parsed only by `core/_calendar.py` (`validate_duration`, `BusinessDays`). There is no
  second parser.
- `detect/_panel.py::residualise` keeps its positional refits. An `anchor="position"` mode must
  reproduce them bit for bit.
- `expanding_window_split` / `sliding_window_split` are **not** schedules: they are anchored at the
  end of the sample (plans 2, 7 and 9 all found this independently).

Whichever plan reaches its schedule milestone first creates the module. The others import it and
drop their local names; plan-local aliases are allowed only as thin re-exports.

**D2. One long↔dense pivot: `shape/_tensor.py::build_tensor`.**
- Plans 3, 8, 9 and 10 already use it. Plan 7's `statespace/_dense.py` becomes a thin adaptor over
  it (entity chunking and row-order restore), not a fourth pivot.
- New callers must pass `forward_fill=False`, since the default repeats a delisted entity's last value
  (plan 3, trap list).
- New callers must also request Float64, because `shape.to_long` defaults to Float32 (plan 10).

**D3. One numba dispatch helper: `panelary/_internal/_jit.py`.**
- Five plans copy the `_get_cusum_numba` lazy-import pattern into per-package modules. The kernels
  may live per package, but the lazy import, flags (`cache=True`, no fastmath) and the
  numpy-parity test fixture live in `_internal/_jit.py`.
- The first plan that ships a numba kernel creates it and migrates `detect/_monitors.py` and
  `feature_extractors/_kernels.py` onto it, with identical output.
- Every kernel must match its numpy fallback **bitwise** unless its plan states a tolerance and the
  reason.

**D4. One home for special functions: `panelary/_internal/_special.py`.**
- It gathers `norm_cdf` / `norm_sf` / `norm_ppf` (from `depend/_special.py` and `econ/_common.py`),
  `psi` (from `depend/_info.py`), trigamma, vectorised `lgamma`, and vectorised `betainc` / `t_cdf` /
  `t_sf`.
- Old locations keep one-line re-exports.
- The private `_betainc` / `_t_sf` copies in `validation/_forecast_tests.py` are replaced, not
  joined by a third copy. This supersedes plan 1's "promote into `_numpy_stats.py`" and plan 9's
  `_special.py` proposal (the same idea).

**D5. HAC and resampling are owned by plan 1.** `validation/_hac.py` (column long-run variances,
Newey–West / Andrews / prewhitening) and `validation/_resample.py` (count-matrix block bootstraps)
ship in plan 1 M1. Plans 2, 9 and 10 import them and write no local HAC or block bootstrap.

**D6. Other new `_internal` leaves, one owner each:**

| Module | Owner (creates it) | Also used by |
|---|---|---|
| `_internal/_linalg.py` (`psd_repair` moved from `depend/_matrix.py`, with a re-export) | 3 | 2, 6, 9 |
| `_internal/_glm.py` (`logistic_irls`) | 2 | 10 (propensity scores, IPW/DR), 4 (hazard labels, if M7 ships) |
| `_internal/_kmeans.py` (moved from `evolve/_select.py`) | 6 | 3 (market states, if added) |
| `_internal/_wavelets.py` (filter banks) | 8 | shape Wave 2 DWT (`plans/todo/shape-build-contract.md` M4) |
| `_internal/_blocked.py` (block-reset sums moved from `detect/_moments.py`, alias left) | 8 | 5 |
| `_internal/_variogram.py` (trailing multi-lag variogram) | **8** (M3) | 5 (rough-vol H, M6) |
| `_internal/_ohlc.py`, `_internal/_realized_kernel.py` | 5 | 3 (kernel weights and bandwidth rule for the multivariate kernel) |

**D7. Multivariate intraday covariance is assigned to plan 3.** Plan 5 hands refresh-time sampling,
Hayashi–Yoshida and the multivariate realized kernel to plan 3, but plan 3 does not mention them.
They become **plan 3, milestone M7 (new, optional)**, built on plan 5's `_internal/_realized_kernel.py`.
Plan 5 keeps all single-series measures.

**D8. Edits to `econ/_panel.py::pesaran_cd` are owned by plan 6 (M6).** That covers the balanced-panel
O(NT) identity from plan 3 §5.14, masked matrix products when unbalanced, and the bias-corrected CD\*.
Plan 3 does not edit it. **PMFG is not shipped** (O(N³)); plan 3's boundary table listing PMFG under
plan 6 is superseded.

**D9. Registry naming and scope.**
- The registry keys on the bare name, so every new spec name must be globally unique. Use a prefix
  where a collision is plausible (`xs_…` in plan 3; plan 5's `econ` namespace). Plan 8 checks its
  entropy names against plan 3's `xs_entropy`.
- `safe_scope` convention, as `depend`'s rolling ops already use it:
  - **"rowwise"**: trailing-window, per-row operators;
  - **"window"**: whole-window summaries, plus forward-looking labels (the `forward_return` precedent).
- Every new spec gets an adapter or a justified `_NOT_EXERCISABLE` entry in
  `tests/test_registry_conformance.py` in the same commit.

**D10. `synth/` extensions.**
- New dials are **neutral by default**: existing seeds must produce byte-identical panels, checked by
  a golden hash test.
- Each dial claims a unique `_stream` spawn key, recorded in one table at the top of
  `synth/_generate.py`. Plan 10 claims key 25 (treatment planting).
- Plan 6's planted-spillover panel is test-only unless it moves into `synth/` under this rule.

**D11. Shared files that several plans edit.** Land these in separate, small commits, and re-read the
file before each edit:

| File | Plans |
|---|---|
| `panelary/__init__.py` (`_LAZY_SUBMODULES`) | 2, 3, 6, 7, 8, 9, 10 |
| `tests/test_registry_conformance.py` | 3, 5, 6, 8, 9, (4 bars) |
| `panelary/validation/_cv.py` / `core/model_selection.py` (`cross_validate` gains `sample_weight=` / `score_weight=`; purge rewrite) | 4 only. Plan 1 does not touch `cross_validate`. |
| `panelary/_internal/_verbs.py` (`causal`, `complexity` verbs) | 8, 10 |
| `pyproject.toml` (comment on the `fast` extra only; **no new dependencies in any plan**) | 3, 6, 7, 9, 10 |
| `.github/workflows/ci.yml` (advisory oracle job) | 8 |

## Bugs found in shipped code

These came out of the planning work. They should be fixed on their own, ahead of any new feature,
because several are silent-wrong-answer bugs. Rows marked **verified** were confirmed by reading the
code on 2026-09-29. The rest are as reported by the plan named, so reproduce them before fixing.

| # | Issue | Where | Reported by | Changes outputs? |
|---|---|---|---|---|
| B1 | **verified.** Horizon truncated at the tail (`end = min(i + max_holding, n - 1)`) with no censoring flag, so the last `max_holding` rows get labels the data cannot support, and the last row gets `label=0, t1=t` | `label/_barriers.py` (`triple_barrier` kernel) | 4 (M1: `censored` column) | Yes (additive flag; default can stay) |
| B2 | **verified.** A second negative-shift site: `fixed_horizon` uses `shift(-horizon)` directly instead of the audited `factor.forward_return` (no gap guard) | `label/_barriers.py:fixed_horizon` | 4 (M1) | Only on gapped panels |
| B3 | **verified.** `resample` bars are stamped at the window's left edge (the polars `group_by_dynamic` default), so a value timestamped t contains data up to t + freq. That is a look-ahead for features. | `preprocessing/_frame.py:resample` | 4 (out of scope; unowned) | Yes, so it needs a `label=` option and a deprecation |
| B4 | `econ._hdfe.demean` treats empty groups as offset 0. A naive imputation estimator on the untreated mask silently fills unidentified cells (+8.9% bias in the plan's test) | `econ/_hdfe.py` | 10 | Yes (should raise or return NaN) |
| B5 | Purge is O(times × test size), about 12 s per fold at 500k times. A null `t1` at a test time never purges. | `core/model_selection.py:_purge_embargo_positions_t1` | 4 (M1, identical folds) | No (speed); edge case guarded |
| B6 | `.panel.rs_vol` is documented as Rogers–Satchell but is a rolling std | `namespaces/panel.py` | 5 (M1: rename with a warning alias) | No |
| B7 | `sample_entropy` uses N−m+1 templates (standard is N−m) and a whole-series tolerance; `cwt_coefficients` is a centred convolution; catch22 `SC_FluctAnal` boxes anchor at the window start | `feature_extractors/_stats.py`, `catch22/` | 8 | Yes, so they need doc notes or versioned fixes |
| B8 | `_harrv._rolling_sum` drifts (4.8e-11 at 10⁶ rows); `rolling_beta` uses the cancellation-prone covariance formula; `pl.rolling_cov` is off by 5.6e-4 at large means unless series are first-value-centred; `roll_spread` defaults to price levels | `econ/features/_harrv.py`, `_common.py`, `_liquidity.py` | 5 | Small numeric changes |
| B9 | `ConnectednessFeatures.panel_safe=True` although it mixes entities; `econ/__init__` promises per-entity centrality that is not emitted | `econ/_connectedness.py` | 6 | Flag or doc fix |
| B10 | `evt_features` measured at about 19 min for 5,000 entities (per-row Python loop); the econ rolling long-memory estimator is also a per-row loop | `econ/features/_evt.py`, `_longmemory.py` | 9, 8 | No (speed only) |
| B11 | `pn.evolve` raises `AttributeError` until `import panelary.evolve` (in neither the eager nor the lazy list); README says 56 registered specs (there are 60, and 122 with shape and evolve) | `panelary/__init__.py`, `README.md` | inventory | No |
| B12 | `econ.features._common.ols` solves by the pseudo-inverse of X'X, silently zeroing badly scaled columns (a HARQ coefficient came out 0.00028 instead of −209.6; new code works around it) | `econ/features/_common.py` | 5 (build) | Yes |
| B13 | `roll_spread` and `liquidity_features` fail on polars 1.35, the CI floor ("window expression not allowed in aggregation", nested `.over`); four tests in `test_econ_features_liquidity.py` | `econ/features/_liquidity.py` | 5 (build) | No (crash) |
| B14 | `testing._first_mismatch` reports an identical `±inf` cell as a leak (inf − inf is NaN) | `panelary/testing.py` | 5 (build) | No (false alarm) |
| B15 | `core.model_selection._norm_ppf` is off by ~1.6e-9 at 0.975 (feeds DSR, haircut, `expected_maximum_sharpe`); `diebold_mariano`'s `"less"` p-value is `1 - t_sf` | `core/model_selection.py`, `validation/_forecast_tests.py` | 1 (build) | Small numeric changes |
| B16 | `residualise` takes an SVD of the full W×N block (42 ms at 252×3,000) where the dual Gram takes 6 ms, and is prefix-invariant only to ~5.6e-15 when names list later | `detect/_panel.py` | 3 (build) | No (speed); tiny numeric |
| B17 | `cs_zscore`, `winsorize`, `quantile_bin` and `neutralize` are flagged `panel_safe=True` though they mix entities | `namespaces/xs.py` specs | 3 (build) | Flag only |
| B18 | `preprocessing/_features.py` still calls `group_by_dynamic(by=...)` (deprecated); stub drift in `panel.pyi` (`frac_diff`, `zscore`, `rs_vol` frame forms) | `preprocessing/_features.py`, `namespaces/panel.pyi` | 4, 5 (build) | No |
| B19 | `evolve_features` never passes the label horizon to the evaluator (embargo always 1), and there is no embargo between the search window and the holdout; `evaluate()` keeps a broad `except` | `evolve/` | fixes (build) | Yes |

## Recommended build order

- **Wave 0, fixes and shared plumbing** (no new public surface, so the AGENTS.md gate does not apply):
  - B1, B2, B5 (plan 4 M1), B3, B4, B6 and B11.
  - The D1, D3 and D4 modules.
- **Wave 1, what the engine can call soonest:**
  - Plan 1 M1–M2: Sharpe inference and CORP, which replaces truepoint's binned ECE.
  - Plan 9 M1 (drawdowns).
  - Plan 5 M1–M2 (range volatility, EDGE).
  - Plan 3 M1–M2 (estimators, as-of engine).
- **Wave 2:**
  - Plan 2 M1–M2 (drift engine, confidence sequences).
  - Plan 4 M2–M4 (weights, trend scanning, sequential bootstrap).
  - Plan 7 M1–M4 (Kalman, HMM, BOCPD).
  - Plan 8 M1–M2 (wavelets, ordinal family).
  - Plan 6 M1 (as-of `EdgeTable`, `network_lag`).
- **Wave 3:** the rest, in each plan's own milestone order. Plan 10 M5 (IFE / matrix completion)
  waits until a caller exists.

## The AGENTS.md gate

AGENTS.md says: *"No new public surface without a named caller in the engine, and a test in the
engine that exercises it."*

- `truepoint/src/truepoint/quant/` does not exist.
- `truepoint/_firewall.py` forbids `diagnose/` from importing panelary.
- Today truepoint imports only `panelary.leakage.audit`.

The plans name these candidate callers:
- **plan 1:** `inference/calibration.py`
- **plans 3, 5 and 9:** `generate/family_computation.py`
- **plan 7:** `generate/family_asof.py`
- **plan 8:** `inference/errorbars.py`

Every plan builds privately and exports nothing until its caller and engine test land together. The
user explicitly asked for these plans. Whether to relax the rule for research-side features with no
engine caller is the user's decision, not a plan's.
