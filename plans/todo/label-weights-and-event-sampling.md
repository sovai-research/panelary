# Label weights, sample weighting, sequential bootstrap, trend scanning, bet sizing and information-driven bars — build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.** Nothing below exists in
> `panelary/` except what §2 lists as existing (verified by reading the code on
> 2026-09-29). This plan realises the **"Labeling & Weights"** section of
> [`docs/roadmap.md`](../../docs/roadmap.md) and the label-horizon half of
> "Validation & Leak-Audit". It was written in parallel with nine sibling plans;
> §3.3 states every boundary explicitly rather than assuming one.

Pure **numpy + polars** in the import path. numba is used only through the
existing `fast` extra (`panelary._internal._deps.have("numba")`, imported lazily
at first kernel call, never at `import panelary`), and **every numba kernel has a
numpy / pure-Python twin with bitwise-identical output**. scipy, scikit-learn and
statsmodels are **test-only oracles** behind `pytest.importorskip`. Zero new
mandatory dependencies, zero new extras.

Reuse, do not reinvent:
`panelary.core.model_selection` (`PurgedKFold`, `CombinatorialPurgedCV`,
`_train_positions`, `_resolve_t1`, `_contiguous_blocks`, `cross_validate`,
`_fit_predict_fold`), `panelary.core._calendar` (`shift_forward`,
`validate_duration`, `BusinessDays`), `panelary.factor._align.forward_return`
(the single audited negative-shift site), `panelary.namespaces.xs` (`.xs.rank`),
`panelary.models` (`_PanelSupervised(sample_weight=...)`, `_clone_estimator`),
`panelary._internal._numpy_stats` (`norm_cdf`, erf-exact to 1e-16),
`panelary.feature_extractors._kernels` (the numba-dispatch pattern of
`_get_cusum_numba`), `panelary.testing` (`assert_no_lookahead`,
`assert_prefix_invariant`, `assert_no_train_test_leak`), `panelary.registry`
(`FeatureSpec`, `safe_scope`).

---

## 1. Why this exists

The roadmap's one-line brief for this area is: *"Unify labels, sample weights,
and CV purge under one invariant so they can never disagree."* Panelary already
emits `t1` (`label/_barriers.py`) and already purges on it
(`core/model_selection.py`). Everything between those two — concurrency,
uniqueness, sample weights, sequential bootstrap, the modern labels, bet sizing,
and the bar constructions that produce the rows labels are defined on — is
missing, and the part that exists does not scale.

### Measured facts that dictate the design

Measured on this machine on 2026-09-29 (Apple M5 Pro, 15 cores, CPython 3.13.12,
NumPy 2.5.3, polars 1.44.2, numba 0.67.0) with throwaway prototypes. These are
measurements, not estimates; "extrap." marks a linear extrapolation.

| # | What | Measurement | Consequence |
|---|---|---|---|
| 1 | existing `_purge_embargo_positions_t1` (Python loop × `np.any`) | 5.3 ms/fold at 5k unique times; **119 ms** at 50k; O(n·m) → **~12 s/fold** at 500k (intraday bars) | Rewrite the body on the span table. |
| 2 | vectorised span-table purge prototype | **4.0 ms** at 50k (30×), **14.4 ms** at 500k (~800×); **byte-identical** to #1 incl. `NaT` tails | A1/A2 below. |
| 3 | concurrency + average uniqueness, per-event loop (AFML 4.1/4.2 shape, numpy slices — pandas `.loc` is slower still) vs event-stream `bincount`+`cumsum` | R=2·10⁵ rows, N=10⁵ labels: 0.14 s vs **2.1 ms** (67×); 25M rows × 25M spans: **0.28 s** | Event stream is the only path. |
| 4 | same, accuracy of a single global prefix sum of 1/c | max \|Δū\| = **2.9e-12** already at R=2·10⁵ and growing with R | Block-local prefix sums (A3). |
| 5 | dense indicator matrix (mlfinlab `get_ind_matrix` shape) | T=N=5000: **200 MB** | Never materialise it outside tests. |
| 6 | AFML snippet 4.5 sequential bootstrap on the dense matrix | N=120, T=400: **0.30 s**; cost ∝ N³·T → days at N=T=5000 | Test oracle only. |
| 7 | "global recompute" bootstrap, O(T+N) per draw, numpy | N=T=5000: 0.17 s; N=10⁵: **~60 s** extrap. | Rejected. |
| 8 | **exact local-recompute** bootstrap (A8), numba | N=5000: **2 ms**; N=10⁵: **90 ms**. Draws **identical** to the AFML dense reference (N=120) given the same uniforms | Chosen algorithm. |
| 9 | same algorithm, numpy port | N=5000: 0.11 s (22 µs/draw); N=10⁵: **3.37 s** (34 µs/draw); **bitwise-identical draws to numba** | Acceptable fallback. |
| 10 | trend-scan t-stat by global prefix sums of (y, t·y, y²) | max rel. error vs two-pass OLS oracle **1.2e-2** on log prices; **99 %** on near-perfect lines | The "obvious O(T·\|L\|)" method is wrong. |
| 11 | same with block-anchored prefix sums (B=1024) | 4e-7; near-perfect lines 2.8e-2; 2.87 s per 1M rows × 46 horizons | Still not good enough. |
| 12 | **recursive-residual horizon sweep** (A9), numpy | max rel. error **4.0e-12** (log prices), **2.3e-13** (raw prices); near-perfect lines 1e-9 median / 2.7e-8 max; **0.36 s per 1M rows × all horizons 5..50** → ~9 s at 25M | Chosen: 8× faster *and* 10⁹× more accurate than #10. |
| 13 | same, numba `prange` | **27 ms** per 1M rows (272 ms 1 thread) → ~0.7 s at 25M; **bitwise-equal to numpy** | Fast path. |
| 14 | fixed dollar bars: `cum_sum().over(entity)` → lattice id → `group_by` | 10⁷ ticks → 113k bars in **0.17 s** → ~2 s per 10⁸ ticks | Fully vectorised; no kernel needed. |
| 15 | tick-imbalance bars with EWMA thresholds | pure Python over `list` floats **0.06 s/10⁶ ticks**; numba **0.7 ms/10⁶**; identical | Sequential but cheap. |
| 16 | AFML symmetric CUSUM filter | pure Python 0.11 s/10⁶; numba 2.1 ms/10⁶; identical. (Existing SPC `_cusum_events_py`: 0.46 s/10⁶) | Same pattern as #15. |
| 17 | `_numpy_stats.norm_cdf` (`np.vectorize(math.erf)`) | 57 ms/10⁶ → 1.4 s at 25M | Good enough for bet sizing; no new erf. |
| 18 | existing `triple_barrier` (per-entity group_by + double Python loop + O(n·w) `_trailing_std`) | 4.4 µs/row → ~1.9 min at 25M rows | Out of scope to rewrite; §2 F1/F7. |

### Three consequences

1. **Overlapping labels silently inflate the sample size, and nobody reports it.**
   A 20-day fixed-horizon label on every row has average uniqueness ≈ 1/21: a
   "5,000-day backtest" carries ~240 independent labels. `weights.effective_n`
   (Σ ū) is the single most honest number an assessment can print about a
   financial-ML backtest, and it costs 0.3 s on 25M rows (#3).
2. **The obvious way to add weights is a leak.** AFML computes concurrency on
   the whole sample and then cross-validates. Near a fold boundary, concurrency
   counts purged and test labels whose `t1` (a triple-barrier touch time) is a
   function of **test-period prices**, so test-fold information shapes training
   weights. Weights must be recomputed **fold-locally** from training labels only
   (§6 A7) — which is also the conceptually correct definition, since only
   training labels create redundancy in the training set.
3. **The textbook algorithms are the slow ones, and the naive fast ones are
   inaccurate.** mlfinlab-style concurrency and sequential bootstrap are O(N²)
   to O(N³T) (#5, #6); the naive vectorised trend scan is wrong in the second
   decimal (#10). This contract picks, for every method, the fastest *exact*
   formulation we could find and measures it against a brute-force oracle.

---

## 2. What already exists — and what this plan must not duplicate

Verified by reading on 2026-09-29. `grep` for `uniqueness|concurren|sequential_bootstrap|trend_scan|bet_size|imbalance|dollar_bar|time_decay|return_attribution|excess_over_median|class_weight` over `panelary/` returns **nothing** outside unrelated docstrings.

| Where | What it does | This plan |
|---|---|---|
| `label/_barriers.py:triple_barrier` | AFML triple barrier, trailing-σ barriers, emits `label, ret, t1` (t1 same dtype as time) | **Reuse**; additive `censored` column + optional `vol=` column (F1, F7). No rewrite. |
| `label/_barriers.py:fixed_horizon` | forward return over `horizon` rows, null `t1` tail | Route through `forward_return` (F2), byte-identical. |
| `label/_barriers.py:meta_label` | side × outcome → {0,1} | Reuse; feeds `sizing.bet_size`. |
| `core/model_selection.py:_purge_embargo_positions_t1` | closed-interval purge `tj ≤ ei ∧ ti ≤ ej`, O(n·m) loop | **Replace body** with span-table overlap (A2), byte-identical. |
| `core/model_selection.py:_purge_embargo_positions` | integer-horizon purge via Python sets | Untouched (0.9–9 ms/fold, not a bottleneck). |
| `core/model_selection.py:_resolve_t1` | per-unique-time **max** of `t1` (conservative panel purge) | Route through `SpanTable.time_projection()`; identical output. |
| `core/model_selection.py:_train_positions`, `_calendar_embargo_positions` | calendar horizon/embargo dispatch | Untouched; reused. |
| `core/model_selection.py:PurgedKFold`, `CombinatorialPurgedCV` | `t1=` already supported | No split-logic change. |
| `core/model_selection.py:cross_validate`, `_fit_predict_fold` | fits per fold; **no sample-weight path**; `metric(y_true, y_pred)` only | Add `sample_weight=` / `score_weight=` (A7). |
| `validation/_cv.py:cpcv_splits`, `walk_forward_splits` | positional splitters; purge delegated to core; **no `t1=`** | Additive `t1=` pass-through. |
| `factor/_align.py:forward_return` | the single audited `shift(-h)` site + gap guard; spec `safe_scope="window"` | **Reuse** for `excess_over_median` / `quantile_label`; additive `end_time=` kwarg writes `t1` (keeps one negative-shift site). |
| `namespaces/xs.py:.xs.rank` | per-date rank with `normalize=` presets | Reuse for cross-sectional quantile labels. |
| `feature_extractors/_kernels.py:_cusum_events*` → `.ts.cusum` | **SPC** CUSUM: warm-up-standardised, resets both sides and restarts warm-up | Not the AFML filter. Keep; add `sample.cusum_filter` (A11) with a docstring cross-reference. Copy its numba-dispatch pattern (`have("numba")`, `error_model="numpy"`, `cache=True`). |
| `detect/_monitors.py:page_cusum_expr` | Lindley closed form, **no reset** — a monitor | Not an event sampler; §6 A11 explains why the closed form does not apply. |
| `models.py:_PanelSupervised` | `sample_weight=<column>` forwarded to `fit`, retries without it on `TypeError` | Reuse; `cross_validate` attaches a fold-local `__w__` column. |
| `preprocessing/_frame.py:resample` | time bars via `group_by_dynamic` (**default `label="left"`**, verified) | Not duplicated; F6. |
| `validation/_bootstrap.py` | block / stationary / wild / sieve bootstraps of *time* | Different object (resample time blocks, not labels by uniqueness). Not duplicated. |
| `_internal/_numpy_stats.py:norm_cdf` | Φ via `math.erf`, ≤1e-16 vs scipy | Reuse for bet sizing. |

**Defects found while reading (reported, fixed only where in scope):**

- **F1 — triple-barrier tail labels are censored but reported.**
  `_triple_barrier_entity` uses `end = min(i + max_holding, n - 1)`: the last
  `max_holding` rows of every entity get a *truncated* vertical barrier; the last
  row gets `label=0, t1=t`. These are fabricated "no-trend" labels that enter
  training and are not prefix-stable. Fix (M1, additive, non-breaking): emit
  `censored: Boolean` (true iff no touch and `i + max_holding > n - 1`);
  `FoldWeights` and `SpanTable` exclude censored rows by default.
- **F2 — `fixed_horizon` is a second negative-shift site without the gap guard**
  (`pl.col(price).shift(-horizon).over(entity_col)`), contradicting
  `factor/_align.py`'s "single audited negative-shift site". Fix (M1): route
  through `forward_return(..., allow_gaps=True, end_time="t1")` — byte-identical.
- **F3 — purge is O(n·m)** (#1). Fix: A2.
- **F4 — null `t1` at a test time is never purged against.** The comparison
  `tj <= ei` with `ei = NaT` is `False`, so a training span covering a test time
  whose (per-time max) `t1` is null is kept. Rare (needs a test time where every
  entity's label is unresolved — e.g. a tail-only final fold), but real. The
  rewrite preserves current semantics byte-for-byte (a behaviour change needs its
  own decision, §12 Q3); `FoldWeights` adds a guard that raises if any training
  span covers a test time.
- **F5 — sample weights cannot reach a model inside `cross_validate`.** The only
  way today is a pre-computed weight column, which is a full-sample weight —
  exactly trap T1 below.
- **F6 (out of scope, reported to the owner) — `preprocessing.resample` stamps
  each bin at its *left* edge** (`group_by_dynamic` defaults `label="left"`,
  `closed="left"`), so a bar aggregating `[t, t+freq)` is visible at `t`. Used as
  a feature, that is a within-bin look-ahead. Our bars stamp the **last tick**.
- **F7 — `triple_barrier` σ is an O(n·w) Python loop** (4.4 µs/row). Out of
  scope to rewrite; add an optional `vol: str | None` column parameter so a
  causal estimator from the sibling OHLC-volatility plan can set barrier widths.

---

## 3. Scope, non-goals, sibling boundaries

### 3.1 In scope (roadmap bullet → milestone)

| Roadmap "Labeling & Weights" bullet | Here | M |
|---|---|---|
| `_spans_from_t1` canonical span table | `core/_spans.py` (A1) | M1 |
| Event-stream concurrency, O(N log N) average uniqueness | A3 | M1 |
| `weights.return_attribution` + `attach` | A4 | M2 |
| `weights.time_decay`, `class_weights` | A5, A6 | M2 |
| CV weight plumbing, fold-local, weighted scoring | A7 | M2 |
| `label.trend_scanning` (+ look-back feature) | A9 | M3 |
| `label.excess_over_median` (+ quantile labels) | A10 | M3 |
| `sample.cusum` event sampler, `sequential_bootstrap` | A11, A8 | M4 |
| `sizing.bet_size` (2Φ(z)−1, average-active, discretised; dynamic/limit) | A14 | M6 |
| (brief) information-driven bars | A12, A13 | M5 |
| (brief, optional) discrete-time hazard / competing-risk labels | A15 | M7 |

### 3.2 Non-goals — decisions, not omissions

- **Time bars.** `preprocessing.resample` exists (see F6 for its labelling bug).
- **Microstructure features** (VPIN, Kyle's λ, Amihud, Roll, bulk-volume
  classification) — sibling *ohlc-volatility-and-liquidity*. We emit bars with a
  tick-rule `buy_volume`; they compute features from bars.
- **EF3M mixture ("reserve") bet sizing** (AFML ch. 10) — iterative moment
  matching, weak evidence, heavy; not shipped.
- **Power-function dynamic sizing** — sigmoid only in v1.
- **Out-of-bag scores** for `SequentialBagging` — OOB is inflated under label
  overlap (AFML ch. 6 †); not provided, and the docstring says why.
- **A new `safe_scope="target"`** in the registry (see `factor/__init__.py`
  note). Labels follow the `forward_return` precedent (`"window"`); §12 Q5.
- **Rewriting `triple_barrier`'s first-touch scan** — F7 only adds `vol=`.
- **DSR / PBO / PSR / `_sharpe` / `_reconstruct_paths`** — sibling 1.
- **`pn.audit()` label-horizon-overlap check** — the Validation roadmap; it
  should *consume* `SpanTable`, not re-derive spans.

### 3.3 Sibling boundaries

| Sibling plan | Boundary |
|---|---|
| 1 forecast-evaluation-and-sharpe-inference | They own scoring rules, DSR/PBO/PSR, any one-call CV runner. **We** add only `sample_weight=`/`score_weight=` kwargs to `cross_validate` and the two call sites (`estimator.fit`, `metric`). We do not touch `_reconstruct_paths`, `CVReport` fields or `_sharpe`. Whoever lands second rebases; a new runner must call `FoldWeights.compute`. |
| 2 drift-monitoring-and-sequential-inference | They own detection statistics, ARL calibration, monitors, and **density-ratio / label-shift importance weights** (their §, verified in their plan). `sample.cusum_filter` is an **event sampler** with no calibration claim; if they add a shared CUSUM kernel, ours stays separate (different recursion: one-sided resets, no drift term). Their importance weights and our uniqueness weights **compose multiplicatively**; neither computes the other's, and theirs must also be fitted in-fold before being multiplied into `FoldWeights` output. |
| 3 covariance-and-market-state | None. |
| 5 ohlc-volatility-and-liquidity | They supply causal σ estimators we **consume** for barrier widths (`triple_barrier(vol=)`) and CUSUM thresholds (`cusum_filter(threshold=<column>)`). We own bar construction and the tick rule; they own everything computed *from* bars. |
| 6 network-and-spatial-panel | None. |
| 7 causal-state-space-and-regimes | None (regime labels, if any, can be fed to `first_event`). |
| 8 multiscale-complexity-features | Possible overlap: rolling OLS-slope/t features across scales. `.ts.trend_scan` (look-back, max-\|t\| over a window grid) is ours; if they need multi-scale slopes, they should reuse `label/_trend.py:_sweep` rather than write a second regression kernel. |
| 9 tail-risk-and-self-excitation | They may consume our event timestamps / bars (Hawkes). No shared code. |
| 10 panel-causal-inference | Cross-fitting with purged folds may want fold-local weights: consume `FoldWeights`. |
| numba dispatch (all siblings) | Verified in the landed sibling plans: drift (`monitor/_fast.py`), causal-state-space (`_fast.py`), network (`_kernels_numba.py`) and panel-causal (`_fast.py`) each copy the `_get_cusum_numba` pattern into a **per-package** module; none proposes a shared helper. We do the same (`sample/_kernels.py`, `label/_trend.py`) with identical flags, and recommend the orchestrator later consolidate all of them into one `_internal/_jit.py`. We do not create a sixth variant of the helper. |

---

## 4. Hard invariants (every function)

1. **One span table.** Purge, concurrency, uniqueness, return attribution,
   time decay, the sequential bootstrap and fold-local reweighting all read the
   same `SpanTable` built by `_spans_from_t1` from the same `t1` column. A test
   asserts `_resolve_t1(col) == SpanTable.time_projection()` elementwise.
2. **Features are prefix-invariant; labels are resolved-prefix-invariant.**
   Features (`.ts.trend_scan`, `cusum_filter`, `tick_rule`, completed bars):
   `f(x[:T])[t] == f(x[:T+k])[t]` bitwise. Labels are forward-looking by
   definition; their contract is: for every row with `t1 ≤ τ` and not censored,
   `label(prefix ≤ τ)[row] == label(full)[row]` bitwise. Every label emits `t1`
   in the time column's dtype and a `censored` flag.
3. **Anything fit-like is fold-local.** Concurrency, uniqueness, weight
   normalisation, time-decay anchoring, class frequencies, bootstrap draws:
   computed from training labels of the fold only, never globally, never on the
   test fold. Global versions (`weights.attach`) exist for the final refit and
   say so in the first line of their docstring.
4. **Determinism.** No unseeded RNG. Every RNG takes `seed: int` and builds
   `np.random.default_rng(np.random.SeedSequence(...))`. Per-entity streams use
   `SeedSequence([seed, stable_hash(entity)])` with
   `stable_hash = int.from_bytes(blake2b(repr(key).encode(), digest_size=8).digest())`
   — **never Python `hash()`** (salted per process). The uniform stream for a
   sequential bootstrap is generated once up front and consumed identically by
   both backends.
5. **Backends are bitwise-identical.** numba kernels: `fastmath=False`,
   `error_model="numpy"`, no reductions across threads (`prange` only over
   independent rows/entities). numpy twins use the same operation order:
   sequential sums via `np.cumsum` (never `np.sum`, which is pairwise), explicit
   NaN skipping (Python `max(0.0, nan) == 0.0` but `np.maximum(0.0, nan)` is
   `nan`). Every kernel has a `numba ≡ numpy` test and a thread-count test.
6. **float64 everywhere; exact integers where possible.** Upcast Float32.
   Concurrency, tick counts and share volumes accumulate in `int64` (exact).
   Stable algorithms (recursive residuals, block-local prefix sums, busy-period
   resets) are mandatory where cancellation was measured (#4, #10).
7. **No per-entity Python loops, no `rolling_map`.** Panels are processed on the
   **flat row axis** of the `(entity, time)`-sorted long frame: spans never cross
   an entity boundary, so one flat array of row indices *is* the per-entity
   computation. Inherently sequential recursions run as numba kernels or as a
   flat pure-Python pass with state resets at entity starts, or — for wide
   panels — **lockstep** numpy over entities (one vectorised step per time
   index). `.over(entity)` + `map_batches` is the accepted house pattern only for
   `.ts` expressions.
8. **No quantity depends on `len(x)`** — not thresholds, not EWMA inits, not
   normalisers — except explicitly fold-local fits (invariant 3).
9. **polars ≥ 1.35 API only.** `cum_sum`, `over`, `rle_id`, `forward_fill`,
   `backward_fill`, `replace`, `group_by(...).agg`, `map_batches(return_dtype=)`.
   Do not rely on `Expr.round(mode=)` (not guaranteed on 1.35 †) — discretisation
   uses `np.round` (half-to-even, AFML/pandas parity).

---

## 5. Module placement and public API

### 5.1 Layout

```
panelary/core/_spans.py             NEW  SpanTable, _spans_from_t1, _span_sums, _concurrency, _overlaps_any (M1)
panelary/core/model_selection.py    EDIT purge body (A2), _resolve_t1 via spans, cross_validate weights (M1/M2)
panelary/validation/_cv.py          EDIT t1= on cpcv_splits / walk_forward_splits                 (M2)
panelary/factor/_align.py           EDIT forward_return(end_time=None) additive                   (M1)
panelary/label/_barriers.py         EDIT censored column, vol=, fixed_horizon via forward_return   (M1)
panelary/label/_trend.py            NEW  _sweep kernel (numpy+numba), trend_scanning              (M3)
panelary/label/_xsection.py         NEW  excess_over_median, quantile_label                       (M3)
panelary/label/_events.py           NEW  first_event (optional)                                   (M7)
panelary/namespaces/ts.py           EDIT additive .ts.trend_scan (+ FeatureSpec)                  (M3)
panelary/weights/{__init__,_concurrency,_weights,_fold}.py        NEW                              (M1/M2)
panelary/sample/{__init__,_cusum,_bootstrap,_bagging,_bars,_imbalance,_kernels}.py  NEW           (M4/M5)
panelary/sizing/{__init__,_bet,_dynamic}.py                       NEW                              (M6)
panelary/__init__.py                EDIT eager import of weights / sample / sizing (numpy+polars only)
```

`weights`, `sample`, `sizing` are eager subpackages (≤ 3 ms each on the cold
import; `tests/test_import_hygiene.py` enforces the 300 ms budget and that numba
never enters `sys.modules` at import). No name collides with the eight verbs.

### 5.2 Public API

```python
import panelary as pn

# ---- spans (power users; the rest of the API builds them internally)
spans = pn.weights.spans(df, t1="t1", entity="id", time="date")      # -> SpanTable (frozen)

# ---- weights (global: exploratory / final refit only — docstrings say so)
pn.weights.concurrency(df, t1="t1")          # + "concurrency" Int64 on every grid row
pn.weights.average_uniqueness(df, t1="t1")   # + "uniqueness" Float64 on label rows
pn.weights.effective_n(df, t1="t1")          # float: Σ ū  (the honest sample size)
pn.weights.return_attribution(df, t1="t1", price="close", afml_compat=False)  # + "w_ret"
pn.weights.time_decay(df, t1="t1", c=0.5)    # + "w_decay" on cumulative uniqueness
pn.weights.class_weights(labels, effective=None)   # dict class -> weight ("balanced")
pn.weights.attach(df, t1="t1", kind="return", price="close", decay=0.5,
                  balance="label", out="w")  # product, normalised Σw = n

# ---- fold-local (CV) weights: a frozen config, evaluated per fold
fw = pn.weights.FoldWeights(t1="t1", kind="return", price="close", decay=0.5,
                            balance="label", normalize="sum_n")
report = pn.cross_validate(est, df, y="label",
                           cv=pn.PurgedKFold(5, t1="t1", embargo="5bd"),
                           sample_weight=fw,        # fitted per fold on train labels only
                           score_weight=fw)         # per fold on test labels only
w_train = fw.compute(panel, train_positions)        # for hand-written CV loops

# ---- labels (all emit t1 in the time dtype + censored)
pn.label.trend_scanning(df, price="close", min_window=5, max_window=20, step=1,
                        log=True, t_threshold=0.0, allow_partial=False)
    # + label {-1,0,1}, t_value, horizon, slope, t1 (information end), t1_trend
pn.label.excess_over_median(df, entity="id", time="date", price="close", horizon=21,
                            binary=False, min_count=20, allow_gaps=False)
pn.label.quantile_label(df, entity="id", time="date", price="close", horizon=21, q=3,
                        mode="cross_section", lookback=252)   # or mode="trailing"
pn.label.first_event(df, entity="id", time="date", events=["default", "delist"],
                     horizon=252)                             # M7, optional

# ---- causal feature (registered, safe_scope="rowwise")
df.with_columns(pl.col("close").log().ts.trend_scan(min_window=5, max_window=60,
                                                    output="t").over("id"))

# ---- sampling
pl.col("ret").pipe(pn.sample.cusum_filter, threshold="h")    # Boolean Expr; .over(entity)
pn.sample.tick_rule(pl.col("price"))                         # {-1,+1} Expr; .over(entity)
pn.sample.bars(ticks, entity="sym", time="ts", price="px", size="qty",
               kind="dollar", threshold=5e6)                 # or bars_per_day=50, lookback_days=20
pn.sample.imbalance_bars(ticks, entity="sym", time="ts", price="px", size="qty",
                         kind="volume", run=False, init_expected_ticks=2000,
                         span_bars=20, bounds=(0.1, 10.0))
idx = pn.sample.sequential_bootstrap(df, t1="t1", n_draws=None, seed=0,
                                     method="exact", stratify=None)  # int64 row indices
bag = pn.sample.SequentialBagging(estimator, n_estimators=100, max_samples=None,
                                  t1="t1", seed=0, target="label")   # PanelEstimator

# ---- bet sizing
pn.sizing.bet_size(df, prob="prob", side="side", pred=None, n_classes=2, out="size")
pn.sizing.average_active(df, size="size", exit_time="exit", entity="id", time="date")
pn.sizing.discretize(x, step=0.1)
w = pn.sizing.sigmoid_w(divergence=10.0, size=0.95)
pn.sizing.sigmoid_size(w, forecast - price); pn.sizing.target_position(w, forecast, price, max_pos=100)
pn.sizing.inverse_price(forecast, w, m); pn.sizing.limit_price(target_pos, pos, forecast, w, max_pos=100)
```

Frame-level functions accept `DataFrame | LazyFrame`, return a `DataFrame`
sorted by `(entity, time)`, and default `entity/time` to columns 0/1 like
`label/_barriers.py`.

### 5.3 Output schemas (fixed)

- **Label frames**: input columns + `label`, `t1` (time dtype), `censored`
  (Boolean) + method extras (`ret`; `t_value, horizon, slope, t1_trend`;
  `fwd_ret`).
- **Bar frames**: `entity, time (= last tick), t_open, open, high, low, close,
  volume, dollar_volume, vwap, n_ticks, buy_volume, bar_index` (+ `threshold`,
  `imbalance` for information-driven bars). `(entity, time)` must be unique;
  duplicate last-tick timestamps are resolved by `on_duplicate_time="nudge"`
  (+1 time unit per duplicate — conservative, never earlier), `"error"` or
  `"keep"`.
- **Bootstrap**: `int64` row indices into the `(entity, time)`-sorted frame,
  with multiplicity, in draw order.

### 5.4 Registry

| Operator | `safe_scope` | Notes |
|---|---|---|
| `ts.trend_scan` | `rowwise` | `axis="time", flavour="trailing"`, `cost_hint="O(T L_max)"`; picked up by `test_registry_conformance.py` (both instruments). |
| `sample.cusum_filter`, `sample.tick_rule` | `rowwise` | expression factories, like `detect.page_cusum_expr`. |
| `label.trend_scanning`, `label.excess_over_median`, `label.quantile_label`, `label.first_event` | `window` | the `forward_return` precedent: labels read forward by design. |
| `sample.bars`, `sample.imbalance_bars` | `window` | `intent="compress", axis="time", flavour="trailing", width_rule="data_dependent"` — a bar is safe as the summary of its delimited window stamped at its end. |

`weights.*`, `sizing.*`, `sequential_bootstrap` are frame utilities, not
catalogue operators, and are not registered (like `model_selection`).
Licence `Apache-2.0`, source `"Panelary"` (clean-room from the papers; see §10
on AFML snippets).

---

## 6. Algorithms

Notation: R rows in the `(entity, time)`-sorted grid, N labels, E entities,
T_e rows of entity e, h label length in rows, L_max the largest horizon.

### A1. `_spans_from_t1` — the canonical span table

```python
@dataclass(frozen=True, slots=True)
class SpanTable:
    start: NDArray[np.int64]      # grid row of the label (t0), sorted ascending
    end: NDArray[np.int64]        # last grid row of the same entity with time <= t1 (>= start)
    seg_start: NDArray[np.int64]  # first grid row of the label's entity
    seg_end: NDArray[np.int64]    # last grid row of the label's entity
    label_row: NDArray[np.int64]  # row in the *labels* frame (events may be a subset of the grid)
    n_rows: int
    n_dropped: int                # null t1 / censored / t1 beyond the entity's data
    def subset(self, mask: NDArray[np.bool_]) -> "SpanTable": ...
    def time_projection(self, time_pos: NDArray[np.int64], n_times: int) -> NDArray[np.int64]
        # per unique time: max end-time position over labels starting there (== _resolve_t1)
```

Build (no per-entity loop): sort grid by `(entity, time)`; `seg = pl.col(entity).rle_id()`;
`tpos = searchsorted(unique_times, time)`; composite key
`k = seg · (n_times + 1) + tpos` is strictly increasing on the grid; for each
label `k1 = seg · (n_times + 1) + (searchsorted(unique_times, t1, "right") - 1)`;
`end = searchsorted(k, k1, "right") - 1`. **O(R + N log R)**. Validation:
`t1 < t0` raises; `t1` null or `censored` → dropped and counted; `t1` beyond the
entity's last row → dropped as censored (never clamped: clamping is AFML's
`t1.fillna(closeIdx[-1])`, trap T2). Events may be a subset of the grid (CUSUM
sampling): the grid defines the row axis, the labels frame defines the spans.
Integer overflow: `E · (n_times + 1) ≤ 10⁶ · 10⁷` fits int64; assert it.

### A2. Purge on the span table (replaces the body of `_purge_embargo_positions_t1`)

Candidates: (a) current per-train-position `np.any` over test intervals, O(n·m);
(b) interval tree, O((n+m) log m) but pointer-heavy; (c) **sorted starts + running
max of ends + `searchsorted`**, O((n+m) log m), three numpy calls. Chosen: (c).

Encode everything as int64 time positions (`end_pos = searchsorted(times, t1,
"right") - 1`; null → `-1`). Test intervals are sorted by start (they are
positions). `run_end = np.maximum.accumulate(test_end)`; for each position j,
`k = searchsorted(test_start, end_pos[j], "right") - 1`; purge iff `k ≥ 0` and
`run_end[k] ≥ j`. Correctness: among intervals with start ≤ e_j, the one with the
largest end decides overlap. Position encoding preserves `tj ≤ ei ⟺ j ≤ end_pos_i`
because starts lie on the grid. The `-1` sentinel reproduces today's `NaT`
semantics exactly (never overlaps) — **do not** use `np.maximum` on raw `NaT`
values (it propagates and would silently under-purge every later interval; use
the sentinel or `np.fmax`). Embargo code path unchanged. Verified byte-identical
at 50k positions including `NaT` tails (#2).

### A3. Concurrency and average uniqueness (AFML 4.1–4.4)

Candidates: per-event slice loop O(Σh) Python-bound (#3); dense indicator matrix
O(T·N) memory (#5); **event stream** O(R + N). Chosen: event stream.

```
d = bincount(start, minlength=R+1) - bincount(end+1, minlength=R+1)   # int64
c = cumsum(d[:R])                                                     # concurrency, exact
ū_i = (1/len_i) Σ_{r=start_i}^{end_i} 1/c_r        via _span_sums(1/c, spans)
```

`_span_sums(v, spans, offset=0)` returns `Σ_{r=start+offset}^{end} v_r` for every
span using **block-local prefix sums**: rows are cut into blocks of ≤ B = 4096
rows that restart at every entity start; within-block inclusive prefixes are one
`np.cumsum(axis=1)` over a zero-padded `(n_blocks, B)` view (sequential order,
length-independent); a second-level prefix over block totals serves spans
crossing blocks. Error ≲ (B + R/len)·ε·max|v| instead of R·ε·max|v| for the
single global prefix that measured 2.9e-12 at R = 2·10⁵ (#4). Target: rtol 1e-12
vs brute force at R = 10⁷ (benchmark verifies). Closed intervals `[t0, t1]` match
AFML and the purge; `c_r ≥ 1` on every summed row. `effective_n = Σ ū_i`.

### A4. Return attribution (AFML 4.10)

`w̃_i = |Σ_{r ∈ (t0, t1]} r_r / c_r|` with `r_r = log(p_r / p_{r-1})` within entity
(null at entity start), via `_span_sums(r/c, spans, offset=1)`. **Default
excludes the return at t0** (it is earned *before* the event); `afml_compat=True`
uses AFML's inclusive `[t0, t1]` for parity (AFML snippet 4.10 sums
`ret.loc[tIn:tOut]` †). Return attribution already embeds concurrency, so
`kind="return"` and `kind="uniqueness"` are mutually exclusive (multiplying both
double-counts).

### A5. Time decay on cumulative uniqueness (AFML 4.11)

Sort training labels by `(t0, entity code)` (stable, deterministic); `x = cumsum(ū)`
(sequential), `X = x[-1]`; `c ∈ (-1, 1]`: if `c ≥ 0`, `slope = (1-c)/X`, else
`slope = 1/((c+1)X)`; `d = max(0, (1 - slope·X) + slope·x)`. Newest label → 1,
oldest → c (or 0 for the oldest `|c|` fraction when `c < 0`). **X is a fold-level
fit** (trap T4). In CPCV with training blocks on both sides of the test block the
decay runs across the gap toward the most recent *training* label; documented,
and `FoldWeights` warns once per CV.

### A6. Class weights and normalisation

`w_k = n / (K · n_k)` over classes present in the training fold (scikit-learn
"balanced" parity). `effective=True` uses uniqueness mass: `n_k = Σ_{i∈k} ū_i`,
`n = Σ ū`. Final weight = base (`ū` or `w̃`) × decay × class, normalised
`w ← w · n_train / Σw` (AFML's `out.shape[0]/out['w'].sum()`), **per fold**.

### A7. Fold-local reweighting and CV plumbing

`FoldWeights.compute(panel, train_positions)`:

1. `train_labels` = resolved, uncensored label rows whose time position ∈ `train_positions`.
2. `spans_tr = spans.subset(train_labels)`; concurrency on the **full grid row
   axis** (prices inside a training span are part of that label's information;
   rows of purged/test labels are simply not counted).
3. A3–A6 on `spans_tr`; normalise over the fold.
4. **Guard (F4):** if any training span covers a test time position, raise
   `ValueError` naming the entity/time (an under-purge that would otherwise leak
   test prices into both the label and its weight).
5. Walk-forward folds additionally assert `t1 < first test time` for every
   training label (resolved before the fit time).

Cost per fold O(R + N_tr log R): ~0.5 s at 25M rows; 15 CPCV splits ≈ 8 s.

`cross_validate(..., sample_weight: str | FoldWeights | None = None,
score_weight: ... = None)`: a `str` is a pre-computed column (global — emits a
`UserWarning` pointing at T1); a `FoldWeights` is evaluated per fold. Weights reach
sklearn-shaped estimators as `fit(X, y, sample_weight=w)` and Panelary
estimators by attaching `__w__` to the train `PanelFrame` and setting
`sample_weight="__w__"` on the per-fold deep copy (existing `_PanelSupervised`
plumbing). Score weights call `metric(y_true, y_pred, sample_weight=w_test)`; a
metric without that parameter raises (no silent unweighted fallback). With both
`None`, `cross_validate` is byte-identical to today. `validate.purged_kfold/cpcv`
pass through; `cpcv_splits` / `walk_forward_splits` gain `t1=` (per-time end
times, forwarded to `_iter_folds(t1_arr=...)`).

### A8. Sequential bootstrap (AFML 4.5) — exact and fast

AFML's rule: draw i with probability ∝ `ū_i(φ) = mean_{r ∈ span_i} 1/(c_φ(r) + 1)`,
where `c_φ` counts already-drawn labels (with multiplicity).

| Candidate | Per draw | Total | Exact? |
|---|---|---|---|
| AFML snippet 4.5, dense matrix | O(N·T·\|φ\|) | O(N³T) | reference |
| mlfinlab-style numba over the matrix (recursive mean †) | O(N·h) | O(N²h) | yes |
| global recompute: prefix sums of 1/(c+1), all ū in O(N) | O(T+N) | O(N(T+N)) | ~1e-12 |
| **local recompute + fresh partial sums** | **O(K·h + √N)** or O(K·h + K log N) | **O(N(Kh + √N))** | **yes** |

Chosen: **local recompute**. After drawing j, `c[start_j..end_j] += 1`; only
candidates overlapping span j change (`K ≈ 2h · labels-per-row`, ~40 for dense
daily labels). They are found by `searchsorted` on the sorted starts over
`[start_j - (h_max - 1), end_j]` then `end ≥ start_j`, and each affected `ū_i`
is **recomputed from scratch** (sequential sum over its span) — so every `ū_i`
is a deterministic function of the current `c`, independent of history: no
drift, no Fenwick-style accumulated rounding. Sampling uses inverse CDF with the
pre-generated uniform `u_k`: two-level √N blocks whose sums are recomputed fresh
(sequentially) when a member changes; for N > 2²⁰ an implicit pairwise segment
tree (node = left + right, recomputed along changed paths) replaces blocks to
remove the √N term. Both are history-independent, so numba and numpy agree
bitwise (#8, #9). Draws match the dense AFML reference except when a uniform
lands within ~1e-13 of a CDF boundary (tests skip such seeds).

```
c[:] = 0; ū[:] = 1.0; fresh block sums
for k in range(n_draws):
    j = inverse_cdf(u[k] * total)            # block scan, then within-block scan
    out[k] = j; c[start_j:end_j+1] += 1
    for i in candidates(start_j - h_max + 1 .. end_j) with end_i >= start_j:
        ū[i] = seqsum(1/(c[start_i:end_i+1] + 1)) / len_i
    recompute sums of the touched blocks
```

numpy twin: the same steps with `np.cumsum` along a padded `(K, h_max)` gather
(34 µs/draw). Panels: the flat row axis makes a pooled bootstrap exact with no
extra code. `stratify="entity"` runs independent per-entity bootstraps (seeds per
invariant 4; `n_draws_e = N_e` or the `max_samples` share) — `prange` over
entities in numba, lockstep across entities in numpy when E ≥ 32. **Bound**:
the pooled draw loop is inherently serial — N = 25M draws ≈ 25M × ~1 µs; the
docstring says to stratify or cap `n_draws` at that scale.

**Documented fast approximation**, `method="uniqueness_iid"`: i.i.d. draws with
`p_i ∝ ū_i` (fold-local), O(N + n_draws log N). It matches the exact method's
first-draw distribution and ignores redundancy *among drawn* labels. AFML ch. 6's
alternative (`BaggingClassifier(max_samples=mean(ū))`) needs no code — the
docstring shows it.

`SequentialBagging(PanelEstimator)`: `fit(panel)` builds spans from the **fit
panel's** `t1` (fold-local by construction inside `cross_validate`), draws
`n_estimators` index sets with `SeedSequence(seed).spawn(n_estimators)`, fits
`_clone_estimator` copies; `predict` averages predictions / `predict_proba`.
`panel_safe = leakage_safe = True`; `t1` and weight columns are excluded from
features. No OOB (§3.2).

### A9. Trend scanning — recursive-residual horizon sweep

Label (López de Prado 2020, MLAM §5.4 †): for each row t, OLS of
`y_{t..t+L-1}` on `x = 0..L-1` for every L in the grid; `L* = argmax |t̂_β(L)|`;
`label = sign(t̂(L*))` if `|t̂| ≥ t_threshold` else 0.

| Candidate | Cost | Accuracy (measured vs two-pass OLS) |
|---|---|---|
| per-window OLS (mlfinlab/statsmodels shape) | O(R·Σ L) + per-call overhead | exact, slow |
| sliding-window two-pass, centred | O(R·Σ L) vectorised; 7–8 µs/row for L=5..50 | exact (the oracle) |
| prefix sums of y, t·y, y² | O(R·\|L\|) | **1.2e-2** — cancellation (#10) |
| block-anchored prefix sums | O(R·\|L\|) | 4e-7; near-perfect lines 2.8e-2 (#11) |
| **recursive residuals, one sweep over L** | **O(R·L_max)**, 0.36 µs/row numpy, 0.03 µs/row numba | **4e-12**; near-perfect lines 1e-9 (#12) |

Chosen: grow every window by one point per step, all rows at once. With `n`
points and the new point at `x_new = n` (forward; `x̄_n = (n-1)/2`,
`S_xx(n) = n(n²-1)/12` are exact scalars):

```
dx = (n+1)/2                      # x_new - x̄_n  (backward sweep: -(n+1)/2)
e  = y_new - ȳ - β·dx             # one-step-ahead prediction error (n >= 2)
h  = 1/n + dx²/S_xx(n)            # leverage of the new point
SSE += e² / (1 + h)               # Brown–Durbin–Evans recursive residual²: no cancellation
ȳ  += (y_new - ȳ)/(n+1);  S_xy += dx·(y_new - ȳ);  β = S_xy / S_xx(n+1)
t(L=n+1) = β · sqrt(S_xx(n+1)·(n-1) / SSE)
```

SSE is a sum of non-negative terms, so its relative accuracy does not degrade as
R² → 1 (the failure of every prefix-sum method). Streaming argmax keeps memory
O(R) (no `(R, |L|)` matrix: 4 GB at 25M × 20). Ties → smallest L (strict `>`).
A NaN inside a window makes that and every longer window NaN (correct: a window
with a missing price is invalid); all-NaN → null label. `SSE == 0` with β ≠ 0 →
±inf (label = sign); constant windows → NaN, label 0; weight helpers clip |t| at
a user cap. Ragged panels: flat axis, a window is valid iff
`row + L - 1 ≤ seg_end` (monotone in L, so a mask per step suffices). numpy: loop
over L (≤ L_max iterations), ~12 elementwise ops over the row axis, chunked in
2¹⁸-row slices for cache; numba: `prange` over rows, scalar state in registers.
Bitwise-identical (#13).

**Outputs.** `t_value`, `horizon = L*`, `slope = β(L*)`, `t1_trend` = time at row
`t + L* - 1` (descriptive), and **`t1` = time at row `t + L_max - 1`, the
information end** — the label was decided by scanning every horizon up to
`L_max`, so that is its information set. `t1` is what the span table, purge and
weights consume (trap T6). `allow_partial=False` (default) makes rows without a
full `L_max` window null + `censored` (resolved-prefix invariance, T7).

**Look-back feature** `.ts.trend_scan(min_window, max_window, step, output)`:
the same sweep with `x_new = -n` over `y_{t-L+1..t}`; causal and bitwise
prefix-invariant (each row's arithmetic depends only on its own window; no
length-dependent constant). `output ∈ {"t", "horizon", "slope"}`.

### A10. Cross-sectional and quantile labels

- `excess_over_median`: `forward_return(..., end_time="t1")` → `fwd`;
  `label = fwd - fwd.median().over(time)` (non-null entities only), null when
  `pl.len().over(time) < min_count`; `binary=True` → sign. One expression; no
  second negative shift. Survivorship (delisted entities have null `fwd`, so the
  median is over survivors) is documented, not silently "fixed".
- `quantile_label(mode="cross_section")`: `.xs.rank(normalize="uniform")` per
  date → bucket `min(q-1, floor(pct·q))` mapped symmetrically for odd q
  (q=3 → {-1, 0, +1}).
- `quantile_label(mode="trailing")`: thresholds from **resolved** past forward
  returns only: `fwd.shift(horizon).rolling_quantile(lookback, ...)` per entity
  (a *positive* shift — at t, the forward return of `t - h` has just resolved),
  `min_samples = lookback`. Using the full-sample distribution is trap T13.

### A11. AFML symmetric CUSUM event filter

`S⁺ = max(0, S⁺ + y_t)`, `S⁻ = min(0, S⁻ + y_t)`; if `S⁻ < -h_t`: event, `S⁻ = 0`;
elif `S⁺ > h_t`: event, `S⁺ = 0` (only the triggered side resets; AFML
snippet 2.4 †). `h_t` scalar or a **causal** column. NaN `y_t` → skipped (state
unchanged, no event) in every backend. Page's closed form
(`detect.page_cusum_expr`, `S - min(S.cum_min(), 0)`) does not apply: with resets
the next event depends on the previous event time. Backends: numba (2.1 ms/10⁶);
pure Python over `.tolist()` with state reset at entity starts (0.11 s/10⁶ →
2.75 s at 25M); lockstep numpy over a dense `(E, T_max)` view when E ≥ 32 and
`E·T_max ≤ 2R` (one vectorised step per time index; ~0.1 s at 5000 × 5000).
Output Boolean per row; prefix-invariant.

### A12. Fixed tick / volume / dollar bars

**Lattice ("carry") variant, default.** `C_t = cum_sum(v).over(entity)` (int64
for ticks/shares, float64 for dollars). A bar closes on the tick whose cumulative
amount first reaches a multiple of θ; tick t belongs to bar
`b_t = floor(C_{t-1}/θ)` (the *exclusive* cumsum). The tempting
`floor(C_t/θ)` puts the crossing tick into the *next* bar — an off-by-one that
the brute-force test pins. A trade crossing several multiples closes one bar
(ids skip; `rle_id` re-densifies). Then one `group_by([entity, bar])`: `time =
last`, `t_open = first`, OHLC, `volume`, `dollar_volume`, `vwap`, `n_ticks`,
`buy_volume` (tick rule). O(n), polars-parallel, ~2 s per 10⁸ ticks (#14).
**Reset variant** (`overshoot="reset"`, mlfinlab parity): next close =
`searchsorted(C, C_open + θ, "left")`, O(n_bars log n); numba or a Python loop
over bars.

**Adaptive threshold** (`bars_per_day=`, `lookback_days=`):
`θ_d = mean(dollar volume of completed days d-W..d-1) / bars_per_day`; the lattice
runs on normalised units `u_t = v_t / θ_{day(t)}`, so no daily reset is needed and
bar boundaries stay prefix-invariant. First W days: null bars unless
`init_threshold=` is given. Full-sample θ is trap T10.

`tick_rule`: `sign(Δp)`, zeros carried forward
(`.replace(0, None).forward_fill()` over entity). The head of each entity is
**null** until the first non-zero move — no fabricated side (mlfinlab-style
implementations seed it with `+1` †); imbalance/run bars skip null-side ticks.

### A13. Imbalance and run bars (AFML 2.3.2 †)

Per tick `b_t` (tick rule), `v_t` ∈ {1, volume, dollar volume}. Since the bar opened:
imbalance `θ = Σ b v`; run `θ = max(Σ_{b=+1} v, Σ_{b=-1} v)`. Close when
`|θ| ≥ E[T] · |E[b v]|` (imbalance) or
`θ ≥ E[T] · max(P⁺ E[v|+], (1-P⁺) E[v|-])` (run). Expectations are per-bar EWMAs
(`α = 2/(span_bars+1)`) of completed bars' length `T_k`, mean imbalance `θ_k/T_k`,
buy fraction and side-conditional mean size — **updated only when a bar
completes** (T9). Init: `init_expected_ticks` (required) and `E[b v]` from the
first `warmup_ticks` ticks (no bar may close during warm-up) or a user constant.
**Known instability:** `E[T]` and the threshold feed back on each other and can
explode (widely reported with AFML's definition †); `bounds=(lo, hi)` clamps
`E[T]` to `[lo, hi] × init_expected_ticks` (default 0.1–10) and the clamp events
are counted in the output. Backends as A11 (numba 0.7 ms/10⁶; Python 0.06 s/10⁶;
lockstep for wide panels); all bitwise-identical because θ accumulates in the
same order and the EWMA update is one fixed expression. `prange` over entities.

### A14. Bet sizing (AFML ch. 10 †)

- **From probabilities**: `p` = predicted-class probability clipped to
  `[ε, 1-ε]`; `z = (p - 1/K) / sqrt(p(1-p))`; `m = pred · (2Φ(z) - 1)`;
  meta-labelling multiplies by the primary `side`. Φ = `_numpy_stats.norm_cdf`
  (1.4 s at 25M, #17). Probabilities must be **out-of-fold** (T16).
- **Average active bets** (AFML 10.2 evaluates only at change points, O(points·N)):
  event stream on the grid. A bet is active on rows `[start, exit_row - 1]`
  (AFML's `t0 ≤ t < t1`; open bets with null exit run to the entity end).
  `S = cumsum(bincount(start, m) - bincount(exit, m))`,
  `C = cumsum(±1)`; **busy-period reset**: rows with `C = 0` start a new segment
  and `S` is a segmented cumsum, so float drift cannot survive an idle period and
  idle rows are exactly 0. `avg = S/C`. O(R + N). The parameter is named
  `exit_time`, not `t1`, because a scanning label's `t1` is not a causal exit (T15).
- **Discretise**: `clip(np.round(m/step)·step, -1, 1)` (half-to-even).
- **Dynamic sizing and limit price** (sigmoid): `m(ω, x) = x / sqrt(ω + x²)`;
  calibrate `ω = x²(m⁻² - 1)`; `target_position = trunc(m · Q)`;
  `inverse_price(f, ω, m) = f - m·sqrt(ω / (1 - m²))`;
  `limit_price` = mean of `inverse_price(f, ω, j/Q)` over the signed path
  `j = pos + sgn, …, target_pos`. Closed form: `sqrt(ω)` factors out, so a
  per-Q table `G[j] = Σ_{i≤j} (i/Q)/sqrt(1-(i/Q)²)` gives every row in O(1):
  O(N + Q). `|target_pos| = Q` would require `m = 1` (infinite price) and raises.
  AFML's printed loop bounds use `abs(pos+1)` and are only right for
  `0 ≤ pos < target` (†); we implement the signed path and pin parity on that
  case.

### A15 (optional, M7). First-event / competing-risk labels

For event columns `E_1..E_K` (Boolean, e.g. default, delisting, regime switch):
next-event row per column via
`pl.when(ev).then(row).otherwise(None).shift(-1).backward_fill().over(entity)`
(label-side look-forward, O(R·K)); first event = argmin over k. Within
`horizon`: `label = k`, `t1` = time of that row. No event and horizon available:
`label = 0`, `t1 = t + horizon` rows. Entity ends first: `censored = True`, label
null. `time_to_event` in rows. This is the label side of a Shumway (2001)
discrete-time hazard model; fitting the logit is the user's model, purged on `t1`
like any other label. Ships only if M1–M6 are green.

---

## 7. Leak-safety design and named traps

Each trap has a test that **fails on the naive implementation** — a leak test
without power is a rubber stamp.

| # | Trap | Design answer | Test that catches it |
|---|---|---|---|
| T1 | Uniqueness / return weights computed on the full sample, then used inside CV: concurrency near the boundary counts purged and test labels whose `t1` depends on test prices. | `FoldWeights` (A7). | Perturb test-fold prices (moving triple-barrier `t1`s): fold-local train weights bitwise unchanged; the global `attach` weights change. |
| T2 | AFML's `t1.fillna(last_index)` — unresolved labels stretched to the data end: length-dependent. | Unresolved/censored dropped (A1). | Append rows: weights of labels resolved before the cut are unchanged. |
| T3 | Normalising weights to sum to N over the full sample. | Per-fold normalisation. | Fold weights independent of rows outside the fold. |
| T4 | Time decay anchored to the full-sample Σū. | Fold-local anchor. | As T3. |
| T5 | Class weights from full-sample frequencies. | Fold-local. | Flip test-fold labels: train class weights unchanged. |
| T6 | Trend-scanning `t1` = end of the *chosen* horizon: under-purges (the label saw data to `t + L_max - 1`). | `t1` = information end; `t1_trend` descriptive. | Perturb prices in `(t1_trend, t1]`: label changes (so `t1_trend` is not a valid purge end); perturb after `t1`: never changes. |
| T7 | Trend scanning with partial windows at the sample end. | `allow_partial=False`. | Resolved-prefix invariance of labels. |
| T8 | Triple-barrier tail truncation reported as vertical-barrier zeros (F1). | `censored` column; dropped by weights. | Censored rows are exactly the non-resolved-prefix-stable ones. |
| T9 | Imbalance/run thresholds from future bars, the full-sample mean bar size, or the in-progress bar. | EWMA updated on completed bars only; causal init. | Append/perturb ticks after bar k closes: bars ≤ k and their thresholds bitwise unchanged; a full-sample-init variant fails. |
| T10 | Dollar-bar θ from full-sample average daily dollar volume. | Trailing completed days (A12). | Bar boundaries prefix-invariant; full-sample θ variant fails. |
| T11 | Bars stamped at open / bin-left (`group_by_dynamic` default, F6). | `time = last tick`. | `assert_no_lookahead` on a feature computed from bars. |
| T12 | CUSUM threshold from full-sample σ. | Threshold must be a causal column or constant. | Prefix invariance passes with trailing σ, fails with full-sample σ (demonstration test). |
| T13 | Trailing quantile thresholds from forward returns not yet resolved. | `fwd.shift(h)` before `rolling_quantile`. | Perturb returns resolving after t: threshold at t unchanged. |
| T14 | Cross-sectional median over survivors only. | Documented; not a lookahead. | — (doc test only) |
| T15 | Averaging active bets with a non-causal exit (a scanning label's `t1`). | `exit_time` parameter; docs. | Moving an exit to any later time > t leaves the size at t unchanged. |
| T16 | Bet sizes from in-sample probabilities. | Docstring + example uses OOF predictions from `cross_validate`. | — |
| T17 | Bootstrap / bagging spans from the full sample used inside CV; OOB under overlap. | Spans from the fit panel; no OOB. | Draws inside a fold depend only on training rows. |
| T18 | Per-entity seeds from Python `hash()`. | `blake2b` stable hash. | Same draws across two subprocesses and after adding an unrelated entity. |
| T19 | `NaT` handling in the vectorised purge (`np.maximum` propagates `NaT`). | int64 positions with a `-1` sentinel. | Parity with a `NaT` in the *middle* of the test block. |
| T20 | numba `fastmath` / FMA contraction breaking parity and prefix invariance. | `fastmath=False`; parity tests. | `numba ≡ numpy` bitwise on every kernel. |
| T21 | `Expr.round` mode differences across polars versions. | `np.round`. | Half-way cases on both CI polars versions. |
| T22 | `max/min` vs `np.maximum/np.minimum` on NaN. | Explicit NaN skipping. | NaN-laden inputs in the parity tests. |

---

## 8. Tests

`--strict-markers` is on; only the existing `slow` and `benchmark` markers are
used. Brute-force references are **clean-room from the definitions** (not
transcribed AFML book code — its licence is not Apache-compatible †), live in
`tests/_labelweights_reference.py`, and are O(N²)/O(N³) by design.

| File | Contents |
|---|---|
| `test_spans.py` | construction on ragged panels, event subsets, off-grid calendar `t1`, null / censored / `t1 < t` (raises), overflow assert; `time_projection == _resolve_t1`; purge **byte-identical** to a frozen copy of today's `_purge_embargo_positions_t1` and to a pairwise brute force on random spans incl. equal endpoints and `NaT` at the start, middle and end of test blocks; PurgedKFold / CPCV fold sets unchanged on every existing fixture. |
| `test_weights_concurrency.py` | `c` equals the brute-force loop exactly (int); `ū`, `w̃`: rtol 1e-12 vs brute force (R ≤ 2000) and vs the dense indicator matrix; properties: `ū ∈ (0,1]`, `ū = 1` without overlap, `k` identical spans → `1/k` each, `Σ c = Σ len`, `effective_n ≤ N`, fixed-horizon h on T rows → closed-form Σū; `afml_compat` parity. |
| `test_weights_decay_class.py` | `c = 1` → all ones; `c = 0` → linear to 0; `c = -0.5` → oldest half 0; newest = 1 within 1e-15; class weights ≡ `sklearn.utils.class_weight.compute_class_weight("balanced")` (importorskip); `effective=True` mass identity. |
| `test_weights_fold_local.py` | T1, T3, T4, T5 with power (the global variant must fail); spy estimator receives `sample_weight` of the fold's length; spy metric receives `sample_weight`; metric without it raises; `None` → `CVReport` byte-identical to today; F4 guard raises on a constructed under-purge; `t1=` on `cpcv_splits` / `walk_forward_splits`. |
| `test_label_trend.py` | sweep vs sliding-window two-pass OLS (rtol 1e-10) and `numpy.linalg.lstsq` per window; vs `statsmodels` OLS t-values if installed; near-perfect lines within the documented bound; ties → smallest L; NaN windows; ±inf / constant windows; `numba ≡ numpy` bitwise, 1 vs N threads; T6 (both directions), T7; `.ts.trend_scan` through `assert_no_lookahead` and `assert_prefix_invariant` (and the registry conformance suite). |
| `test_label_xsection.py` | `excess_over_median` and `quantile_label` vs hand-computed panels; `min_count`; T13; `fixed_horizon` byte-identical after routing through `forward_return`. |
| `test_label_censoring.py` | F1 `censored` flag; resolved-prefix invariance for `triple_barrier`, `fixed_horizon`, `trend_scanning`, `excess_over_median` at several cuts. |
| `test_sample_bootstrap.py` | draws ≡ dense AFML-definition reference given the same uniforms (N ≤ 150, 20 seeds, skipping seeds with a uniform within 1e-9 of a boundary); `numba ≡ numpy` bitwise at N = 10⁴ (blocks) and N = 2²⁰+1 (tree, `slow`); stratified seeds stable under entity reordering/addition and across subprocesses (T18); MC property: mean uniqueness of sequential samples > i.i.d. (200 reps, `slow`); `uniqueness_iid` first-draw distribution; `SequentialBagging` inside `cross_validate` uses only training spans (T17). |
| `test_sample_cusum.py` | vs the definition; Python ≡ lockstep ≡ numba; time-varying threshold; NaN skipping (T22); prefix invariance; T12 demonstration. |
| `test_sample_bars.py` | lattice and reset variants vs brute-force loops (incl. the crossing-tick off-by-one and multi-multiple trades); integer exactness for tick/volume; `time = last tick` (T11); adaptive threshold prefix invariance (T10); imbalance/run: Python ≡ lockstep ≡ numba, T9, clamp counting, duplicate-timestamp policies; `tick_rule` null head. |
| `test_sizing.py` | `bet_size` vs `scipy.stats.norm.cdf` closed form (importorskip) and `norm_cdf`; multi-class; clipping; `average_active` vs brute force (AFML 10.2 definition) at change points and on the full grid, idle rows exactly 0; T15; `discretize` half-to-even (T21); `sigmoid_size(sigmoid_w(x, m), x) == m`, `inverse_price` round trip, `limit_price` vs explicit signed loop; `|target| = Q` raises. |
| `test_label_first_event.py` | (M7) vs brute force; competing events on the same row (lowest column index wins, documented); censoring; purge on `t1`. |
| `test_labelweights_perf.py` (`benchmark`) | §9 budgets at reduced sizes with 2× slack. |

Standing guardrails that must stay green: `test_import_hygiene.py` (no numba /
scipy / sklearn at import; cold-import budget), `test_dependency_drift.py` (no
new mandatory dependency), `test_wheel_guardrails.py` (`py3-none-any`),
`test_registry_conformance.py`, mypy ratchet (`MYPY_BASELINE` must not rise; new
modules type-clean), ruff.

---

## 9. Benchmarks and performance budgets

`benchmarks/bench_labelweights.py` (house style of `bench_hotspots.py`): prints
one table, records machine, versions, thread count and numba compile time
**separately** (first-call JIT is excluded from throughput and reported).
Datasets are generated in-script, seeded: (a) daily panel 5,000 entities × 5,000
dates of GBM log-prices, fixed-horizon h = 20 labels on every row plus
triple-barrier-like random spans (h ∈ [1, 20]); (b) CUSUM-sampled events at ~5 %
of rows; (c) ticks: 10⁷ by default, 10⁸ opt-in (`--ticks 1e8`; needs ~16 GB RAM:
int64 ts + f64 price + f64 size ≈ 2.4 GB input), 1-cent price grid, lognormal
sizes, 1–5,000 symbols.

Budgets on the reference machine (Apple M5 Pro); the perf test enforces 2× at
reduced sizes. "Measured" cites §1; the rest are targets.

| Operation | Size | numpy / polars path | numba path |
|---|---|---|---|
| `_spans_from_t1` | 25M rows | ≤ 1.5 s | — |
| purge (A2) per fold | 500k times | ≤ 30 ms (14 ms measured) | — |
| concurrency + ū | 25M spans | ≤ 1.0 s (0.28 s measured, global prefix) | — |
| `FoldWeights.compute` | 25M rows, one fold | ≤ 2 s | — |
| sequential bootstrap | N = 10⁵ | ≤ 6 s (3.4 s measured) | ≤ 0.2 s (90 ms measured) |
| sequential bootstrap | N = 10⁶, stratified, E = 5000 | ≤ 60 s | ≤ 2 s |
| trend scan, L = 5..50 | 25M rows | ≤ 15 s (9 s extrap.) | ≤ 2 s (0.7 s extrap.) |
| `.ts.trend_scan` via `.over()` | 5000 entities | per-group overhead ≤ 0.5 s total | — |
| fixed bars | 10⁸ ticks | ≤ 5 s (~2 s extrap.) | — |
| imbalance / run bars | 10⁸ ticks, 1 symbol | ≤ 15 s (Python, 6 s extrap.) | ≤ 0.5 s (70 ms extrap.) |
| CUSUM filter | 25M rows | ≤ 5 s flat Python / ≤ 0.5 s lockstep (E = 5000) | ≤ 0.2 s |
| `bet_size` + `average_active` | 25M rows | ≤ 3 s | — |

Accuracy budgets (asserted in tests, not benchmarks): trend t-values rtol ≤ 1e-10
vs two-pass OLS on random-walk data (4e-12 measured); uniqueness / attribution
rtol ≤ 1e-12 vs brute force; everything else exact or bitwise.

---

## 10. Dependencies

- **Import path:** numpy, polars. Nothing else.
- **Optional:** numba via the existing `fast` extra, following the
  `_get_cusum_numba` pattern in per-package kernel modules (`sample/_kernels.py`,
  `label/_trend.py`): lazy `have("numba")`, module-level compiled-kernel cache,
  `njit(cache=True, error_model="numpy", fastmath=False, nogil=True)`,
  `parallel=True` only for row/entity-parallel kernels (§3.3 on consolidation). A private
  `_backend: Literal["auto", "numpy", "numba"]` argument on internal dispatchers
  lets tests force either path; no public env var in v1 (§12 Q6).
- **Test-only oracles** (`importorskip`): scipy (`norm.cdf`), scikit-learn
  (`compute_class_weight`), statsmodels (OLS t-values, if installed).
- **Not used:** mlfinlab — current releases ship stubs (the fetched
  `bet_sizing/ch10_snippets.py` has `pass` bodies) and its licence is
  proprietary; AFML book snippets — copyrighted, so references are clean-room
  from the definitions.
- `tests/test_dependency_drift.py` must show **zero** new mandatory deps.

---

## 11. Milestones

| M | Contents | Ships | Gate |
|---|---|---|---|
| **M1** | `core/_spans.py`; A2 purge rewrite (byte-identical); `_resolve_t1` via spans; `weights.spans / concurrency / average_uniqueness / effective_n`; F1 `censored`, F7 `vol=`; F2 `fixed_horizon` via `forward_return(end_time=)`. | A 30–800× faster purge with no behaviour change, and the honest sample size (Σū) for any labelled panel. | `test_spans`, `test_weights_concurrency`, `test_label_censoring` green; every existing CV test unchanged. |
| **M2** | A4–A7: `return_attribution`, `time_decay`, `class_weights`, `attach`, `FoldWeights`; `cross_validate(sample_weight=, score_weight=)`; `t1=` on positional splitters. | Leak-safe sample weighting end-to-end. | T1–T5 tests have power (global variant fails). |
| **M3** | A9 sweep kernel (numpy + numba); `label.trend_scanning`; `.ts.trend_scan` + spec; A10 `excess_over_median`, `quantile_label`. | The modern labels with correct purge spans. | Accuracy budget; T6, T7, T13. |
| **M4** | A8 sequential bootstrap (exact + `uniqueness_iid`), `SequentialBagging`; A11 `cusum_filter`, `tick_rule`. | Max-uniqueness bagging and AFML event sampling. | Oracle-identical draws; bitwise backend parity. |
| **M5** | A12 fixed + adaptive bars; A13 imbalance / run bars with clamps. | Information-driven bars as long panel frames. | T9–T11; 10⁸-tick benchmark run once and recorded. |
| **M6** | A14 `sizing.*`. | Side → size, completing meta-labelling. | T15; closed-form parity. |
| **M7** (optional) | A15 `first_event`. | Competing-risk labels with censoring. | Only after M1–M6 green. |

Orchestrator (not a milestone agent): `__init__` exports, registry specs, docs
page `docs/user-guide/labels-and-weights.md`, mkdocs nav, CHANGELOG, marking the
roadmap bullets shipped. M1 alone is shippable; M2 is what makes it defensible.
M5 is independent of M2–M4 and can run in parallel.

---

## 12. Risks and open questions (resolve with a benchmark, not an opinion)

1. **Block-local prefix accuracy at R = 25M** — does B = 4096 meet rtol 1e-12?
   If not, drop to B = 1024 or sum spans directly in numba (exact sequential).
2. **Segment tree vs √N blocks** — measured numbers are for the block sampler;
   confirm the tree is faster above 2²⁰ before shipping it, else keep blocks.
3. **F4 behaviour change** — should a null `t1` at a test time purge
   conservatively (treat as `+∞`)? Safer, but changes folds; needs an owner
   decision and a CHANGELOG entry. Until then: byte-identical + the guard.
4. **Imbalance-bar clamp defaults** (0.1–10 × init) are a guess; calibrate on
   real tick data (bars/day stability over a year) before documenting them as
   defaults.
5. **Registry scope for labels** — `"window"` follows `forward_return`, but the
   conformance suite requires window specs to be demonstrably *not*
   prefix-invariant per row; bars are prefix-stable for completed bars and may
   need a conformance adapter or exemption. A `"target"` scope would be cleaner
   and is out of this plan's remit.
6. **A public `PANELARY_NUMBA=0` switch** for reproducibility audits (force the
   numpy twin) — useful for the engine, but new public surface; defer.
7. **Pooled vs per-entity concurrency** — v1 is per entity (AFML: overlap of the
   *same instrument's* returns). A cross-sectional "time" scope for pooled models
   has no agreed definition; leave out until a caller needs it.
8. **`.ts.trend_scan` per-group dispatch** — if the `.over()` overhead exceeds
   budget at 5000 entities, offer the flat frame-level path as the documented
   bulk route.
9. **Scratchpad hygiene (process note)** — the prototypes behind §1 were run in
   a session scratchpad shared with sibling agents; re-run them from
   `benchmarks/bench_labelweights.py` before quoting numbers in docs.

---

## 13. Caller

`AGENTS.md`: *"No new public surface without a named caller in the engine, and a
test in the engine that exercises it."* **This rule is not yet satisfied.** The
engine (`truepoint`) does not import Panelary, `truepoint/src/truepoint/quant/`
does not exist, and `truepoint/_firewall.py` forbids `diagnose/` from importing
`panelary` at all. The user explicitly asked for this plan; the rule still
applies to what ships.

The honest candidate caller, if the engine grows the `quant/` package AGENTS.md
names: a financial-ML backtest assessment that reports (a) `weights.effective_n`
/ N — how many independent labels a submitted backtest really has — and (b)
re-scores the client's model under `PurgedKFold` with `FoldWeights`, both as
`Evidence(kind=STATIC_ANALYSIS)`. That is directly "how often is this financial
AI silently wrong?". Recommendation: **M1 proceeds** (it hardens existing surface
— the purge — and adds three read-only weight functions); **M2–M7 wait for the
engine to name the caller and add its test.**

---

## 14. References

(† = recalled from memory or a secondary source, not re-verified against the
primary text in this session.)

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  Ch. 2 (standard and information-driven bars; symmetric CUSUM filter, snippet
  2.4 †), ch. 3 (triple barrier, meta-labelling), ch. 4 (concurrency, uniqueness,
  sequential bootstrap, return attribution, time decay; snippets 4.1–4.5, 4.10,
  4.11 †), ch. 6 (bagging, `max_samples` = average uniqueness, OOB under
  overlap †), ch. 7 (purged k-fold), ch. 10 (bet sizing, snippets 10.1–10.4 †).
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge
  Elements in Quantitative Finance. §5.4 trend-scanning labels †.
- Brown, R. L., Durbin, J., Evans, J. M. (1975). Techniques for testing the
  constancy of regression relationships over time. *JRSS B* 37(2):149–163 —
  recursive residuals.
  <https://academic.oup.com/jrsssb/article/37/2/149/7027284>
- Welford, B. P. (1962). Note on a method for calculating corrected sums of
  squares and products. *Technometrics* 4(3):419–420.
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). Algorithms for computing the
  sample variance: analysis and recommendations. *Am. Statistician* 37(3):242–247.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd
  ed., SIAM — ch. 4, summation error bounds.
- Easley, D., López de Prado, M., O'Hara, M. (2012). The volume clock: insights
  into the high-frequency paradigm. *J. Portfolio Management* 39(1):19–29 †.
- Lee, C. M. C., Ready, M. J. (1991). Inferring trade direction from intraday
  data. *J. Finance* 46(2):733–746 — tick rule.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41:100–115.
- Shumway, T. (2001). Forecasting bankruptcy more accurately: a simple hazard
  model. *Journal of Business* 74(1):101–124.
- Martínez, G. (2019 †). Information-driven bars for financial machine learning:
  imbalance bars — the exploding expected-bar-size feedback loop.
  <https://medium.com/data-science/information-driven-bars-for-financial-machine-learning-imbalance-bars-dda9233058f0>
- Hudson & Thames (†). Bagging in financial machine learning: sequential
  bootstrapping — standard implementation O(n³); numba + recursive mean.
  <https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/>
- mlfinpy documentation, Data Labelling — trend-scanning parameters
  (`look_forward_window`, `min_sample_length`, `step`) and outputs (`t1`,
  `t_value`, `ret`, `bin`). <https://mlfinpy.readthedocs.io/en/latest/Labelling.html>
- mlfinlab bet sizing module (now stub-only upstream).
  <https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/bet_sizing/bet_sizing.py>
- scikit-learn, `utils.class_weight.compute_class_weight("balanced")`:
  `n_samples / (n_classes · bincount(y))`.
