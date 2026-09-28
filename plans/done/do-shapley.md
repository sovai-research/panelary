# do-Shapley: attribution by intervention, not association

> **Status (2026-09-28): implemented.** C0–C3 all shipped. The C3 build
> contract is written below, under C3.
>
> - **Core.** `panelary/_internal/_shapley.py` is new: the exact solver, the
>   antithetic permutation solver with Monte-Carlo SE, the coalition memo, and
>   canonical JSON helpers. Both callers now use it.
> - **C0.** `leakage/_borrowed.py` names the attribution an executed
>   interventional (do-)Shapley value, citing Jung et al. (ICML 2022) and Heskes
>   et al. (2020). `docs/benchmarks/leakage-table.md` says the same, and
>   `docs/user-guide/attribution.md` gains a causal-reading column, plus a
>   warning that covers Corollary 2 and the direct-cause assumption.
> - **C1.** `evaluate` may return a vector of replicate scores. The headline is
>   the replicate mean, which keeps efficiency. The report adds per-replicate
>   values, `attribution_median` / `attribution_range` and
>   `total_median` / `total_range`. A scalar `evaluate` is unchanged. The
>   benchmark's Shapley section now runs over all seeds (`--long`: 9 seeds, 144
>   backtests, about 10 s), and the doc table reports median, min, max and mean.
>   The earlier single-seed numbers were seed 0, the minimum of the nine.
> - **C2.** `method="permutation", n_permutations=M, seed=` is opt-in only. The
>   refusal still leads with "Group components".
> - **C3.** `panelary/select/_refit_shapley.py` adds `refit_shapley` and
>   `RefitShapleyReport`, exported from `panelary.select`. It supports groups,
>   `base_features`, a training-mean baseline, CRN folds and seeds, seed
>   replicates, `per_period`, and permutation beyond 12 groups. The selection
>   guide and API reference document it.
> - **Serialisation.** `to_dict()` / `to_json()` on both reports follows the
>   shared `schema` / `produced_by` convention.
> - **Tests.** `tests/test_shapley_core.py` (37), `tests/test_refit_shapley.py`
>   (36) and `tests/test_leakage_borrowed.py` (56, of which 18 are new).
> - **Deferred.** The per-fold decomposition is per *fold*; sub-fold periods
>   (e.g. per month inside a test block) are not offered. The out-of-scope items
>   below stay out of scope.

**Stage:** todo — C0–C2 are small enough to build directly; C3 needs a build contract · **Priority:** C1 first (it fixes a reporting inconsistency), C3 has the most user value · **Home:** Panelary — `leakage/_borrowed.py`, `explain`, `select`

## Pitch

Jung, Kasiviswanathan, Tian, Janzing, Blöbaum & Bareinboim, *On Measuring
Causal Contributions via do-interventions* (ICML 2022, PMLR 162:10476–10501).
Compute Shapley over `v(S) = E[Y | do(V_S = v_S)]` instead of
`E[Y | V_S = v_S]`. Four axioms (perfect assignment, causal irrelevance, causal
symmetry, causal approximation) make it unique (Thm 1). Thm 2 is a sufficient,
O(n³) graphical test for when all 2ⁿ interventional quantities are identifiable
from observational data. §5 gives IPW, outcome-regression and DML estimators
for the Markovian and direct-cause cases, with permutation sampling
(Algorithm 1); the DML one is doubly robust. Heskes et al. (2020) proposed the
same formula for models you can query ("causal Shapley"). do-Shapley extends it
to outcomes produced by nature.

## Assessment

**Most of the paper solves a problem Panelary does not have.** §4–5 exist
because the authors cannot run the intervention: `Y` comes from nature, so every
`E[Y | do(v_S)]` has to be identified from a causal graph and then estimated.
When the thing being explained is a *pipeline*, the program is the causal model
and an intervention is a rerun. Identification is trivial, and none of the
estimators are needed.

**We already ship one, in that easy regime.** `leakage.borrowed_accuracy` is a
do-Shapley. The players are pipeline stages. Each coalition sets its stages'
mode by intervention (permissive or point-in-time), and each coalition is
*executed*, not estimated. The axioms carry over one for one: efficiency is
`sum(phi) == B`, and causal irrelevance is the null-player test. Strictly, it is
the baseline form: both modes are interventions, and `v({})` is the
all-point-in-time run rather than a natural, un-intervened regime.

**Where the paper sharpens `explain`.** Its Table 1 and Corollary 2 say exactly
what our mode table in `docs/user-guide/attribution.md` says only loosely:

- `mode="interventional"` (marginal Shapley) intervenes on the *model's inputs*.
  It equals the do-Shapley on the real target only under the *direct-cause*
  graph, where no feature causes another and nothing confounds a feature with
  `Y`. (Cor. 2: `E[Y|do(v_S)] = Σ E[Y | v_S, v_S̄] P(v_S̄)`, which is marginal
  Shapley applied to the regression function.)
- `mode="conditional"` fails causal irrelevance. It can credit a feature the
  outcome does not depend on, because that feature is correlated with one that
  it does.
- Engineered panel features break the direct-cause assumption by construction:
  momentum, volatility and drawdown are all functions of the same price path.
  No `explain` output is a causal contribution of a feature *to returns*.

**Why not port the estimators.** Each of these reasons is sufficient on its own:

1. The setup assumes discrete `V` ("V is a set of discrete variables"). The IPW
   and DML weights are indicators, `1{v_S}(V_S) / P(V_S | ...)`, which are zero
   almost surely for continuous features.
2. It needs a causal graph over features that users cannot justify. That was
   already ruled out of scope in `plans/done/shap-attribution-plan.md`.
3. Cross-fitting is "split D randomly into two halves". On a panel, random halves
   are serially and cross-sectionally dependent, which breaks the independence
   the error bounds rest on. A panel port would need purged, time-blocked
   cross-fitting. That is the one piece that would be Panelary's to contribute,
   if this is ever pursued.

## Builds, in order

### C0: say it in the docs (tiny)

- `leakage/_borrowed.py` and `docs/benchmarks/leakage-table.md`: name the
  decomposition as an interventional (do-)Shapley whose coalitions are executed,
  and cite the paper. That answers "Shapley isn't causal" for this particular
  number.
- `docs/user-guide/attribution.md`: add the causal reading to the mode table,
  plus the Cor. 2 caveat above.

### C1: replicate-aware `borrowed_accuracy` (small)

The leakage table's honesty notes say "A single seed is not evidence … Report
ranges", yet its Shapley section reports a single seed. Let `evaluate` return a
vector of replicate scores, one per seed. Shapley is linear in `v`, so the
per-replicate φ are exact and efficiency holds within each replicate. Report the
median and range per stage.

Replicates must be independent reruns. Per-fold values of a *pooled* metric
(pooled R²) do not qualify, because pooling is nonlinear. Rerun the benchmark
and update `docs/benchmarks/leakage-table.md`.

### C2: sampled mode beyond `max_components` (small)

Today `k > 12` is refused. The permutation estimator in Algorithm 1 lifts that
limit, and it keeps two properties:

- **Cost.** Memoise coalitions. `v({})` and `v(N)` are shared, so each
  permutation adds at most `k − 1` new evaluations, and M permutations cost at
  most `M(k−1) + 2`, against `2^k` for exact. At k = 20 and M = 50 that is 952
  against 1,048,576.
- **Efficiency stays exact.** Each permutation's marginal contributions
  telescope to `v(N) − v({})`, so the sum is preserved. Report a Monte-Carlo
  standard error per player, and use antithetic (reversed) permutation pairs to
  cut variance.

Make it opt-in (`method="permutation", n_permutations=...`), never a silent
fallback. The refusal message should keep "group your components" as its first
suggestion.

### C3: refit Shapley for feature groups and data sources (medium; needs a contract)

Quants most often ask "what is this feature group, or this dataset, worth to my
pipeline?" Panelary currently has two answers, and neither can split credit
between redundant inputs:

- `select.mda` permutes one feature at a time against a fixed fitted model.
  Substitutes mask each other, so two near-duplicates can both look worthless,
  and interactions are invisible.
- `explain.joint_group_shap` intervenes on inputs while holding the model fixed.
  It says what *this model* uses, not what the data offers the learner.

Refit Shapley plays a game whose players are feature groups (or raw data sources)
and whose `v(S)` is the purged-CV score of the pipeline *refit* on `S`. It is a
do-Shapley on the training procedure, exact and executable. By symmetry, a
redundant pair splits the credit instead of both scoring zero. Combined with C1
it gives a spread per player. Decomposing per-period scores gives attribution
over time, e.g. a feature group whose share decays.

Design constraints:

- Players are groups: up to 12 exactly, C2 beyond that.
- Compute features once; coalitions select columns, so only the model is refit.
- Use **common random numbers**: the same folds and seeds for every coalition,
  so noise cancels in the differences (the leakage table's paired design, and
  `_borrowed`'s determinism requirement).
- Score through the same purged splitter as `select.mda`.
- Define `v({})` explicitly, e.g. as predicting the training mean. It is the
  baseline every φ is measured against.
- Extract the combinatorics from `_borrowed.py` into a neutral core rather than
  calling a function named "borrowed accuracy" for a different question.

#### C3 build contract (written 2026-09-28, from the constraints above)

**Surface.** `panelary.select.refit_shapley(estimator, X, y, cv, groups, *,
base_features=(), scoring=None, seeds=None, per_period=False, method="exact",
n_permutations=None, seed=0, max_players=12, entity=None, time=None)
-> RefitShapleyReport`. Reachable as `pn.select.refit_shapley`.

**Players.** `groups` maps a player name to a non-empty list of numeric feature
columns; a plain list of columns means one player per column. Groups are
pairwise disjoint and may not contain the target or a panel key.
`base_features` (disjoint from every group) are in *every* coalition, including
the empty one: they answer "what is this dataset worth on top of what I
already have?". `k >= 1`.

**Value function.** `v(S)` is the mean, over the folds `cv.split(panel)`
yields (the same `split(panel) -> (train, test)` protocol `select.mda` scores
through), of the out-of-fold score of a **fresh** estimator fitted on the
training block using the columns `base_features + union(groups[g] for g in S)`
and scored on the test block. Scores are higher-is-better: `"r2"` (default for
regressors; 1 − SSE/SST about the test-fold mean, numpy, identical to
sklearn's `r2_score`), `"neg_mse"`, `"accuracy"` (default for classifiers), or
a callable `(y_true, y_pred) -> float`. A non-finite score is an error.

**`v({})`, explicitly.** With no `base_features`: predict the **training-fold**
mean of `y` (majority class for a classifier), scored like any other
coalition; nothing is fitted. With `base_features`: the estimator refit on
those alone. Every φ is measured against this baseline, and `v(N) − v({})` is
the report's `total`.

**Features computed once.** The function never computes a feature. Each fold is
collected to numpy once; a coalition slices columns. Only the model is refit.

**Common random numbers.** Folds are materialised once and shared by every
coalition. Rows: a row enters a fold's train or test block iff the target and
every group and base column are finite, so every coalition sees the same rows.
A fold with fewer than 2 usable training rows or no usable test rows is dropped
for all coalitions. Replicate `r` uses `seeds[r]` as the estimator's
`random_state` (via `set_params`, else the attribute; an estimator with
neither is refused) for **every** coalition and fold. `estimator` may instead
be a factory `make(seed: int | None) -> estimator`.

**Solve.** Through the neutral core `panelary/_internal/_shapley.py` (the
combinatorics extracted from `_borrowed.py`, which now uses it too). Exact up
to `max_players` (default 12); beyond, refuse with "group your features" as the
first suggestion, unless `method="permutation", n_permutations=M` is passed (C2
semantics: antithetic pairs, memoised coalitions, efficiency to floating-point
tolerance, a Monte-Carlo SE per player). Never a silent fallback.

**Replicates (C1) and periods.** The game's value is an array `(R, F)`
(replicates × folds); Shapley is linear, so each entry is solved exactly and
efficiency holds per replicate and per fold. Headline φ = mean over replicates
of the fold-mean φ (so it sums to `total`); with `seeds`, the per-replicate φ
and their median and range are reported. `per_period=True` adds the per-fold
decomposition (each fold's test-time span, φ averaged over replicates, and its
own total); because `v` is the fold mean, the headline equals the mean of the
per-fold φ exactly.

**Leak-safety (the contract it touches: `leakage_safe`).** Every fit sees
training rows only, per fold, through the caller's purged splitter; the
baseline mean is the training fold's; nothing is fitted on the whole panel.
Refit Shapley does **not** audit the features themselves — they must already
be point-in-time (`panelary.leakage.audit`, `assert_no_lookahead`).

**Cost.** (coalitions evaluated) × (folds) × (replicates) fits, minus the
fit-free `v({})`: `2**k` coalitions exact, at most `M(k−1)+2` sampled.

**Report.** `RefitShapleyReport`, frozen: `attribution`, `total`,
`full_score`, `baseline_score`, `groups`, `group_features`, `base_features`,
`n_evaluations`, `n_fits`, `n_folds`, `scoring`, `baseline`, `method`,
`n_permutations`, `seed`, `standard_error`, `seeds`, `replicate_attribution`,
`replicate_totals`, `per_period`, `coalition_values`; `ranked`, `share`,
`attribution_median` / `attribution_range`, `to_frame()`, `to_dict()` /
`to_json()` (shared convention: `schema = "panelary.RefitShapleyReport/1"`,
`produced_by = "panelary.select.refit_shapley@<version>"`, byte-deterministic).

**Files.**

| File | Owner | Change |
| --- | --- | --- |
| `panelary/_internal/_shapley.py` | do-shapley | new: exact + permutation solvers, memo, JSON helpers |
| `panelary/leakage/_borrowed.py` | do-shapley | C0 docs, C1 replicates, C2 permutation, `to_dict`/`to_json`, uses the core |
| `panelary/select/_refit_shapley.py` | do-shapley | new: `refit_shapley`, `RefitShapleyReport` |
| `panelary/select/__init__.py` | do-shapley | additive export |
| `tests/test_shapley_core.py`, `tests/test_refit_shapley.py`, `tests/test_leakage_borrowed.py` | do-shapley | axioms, CRN, baseline, leak and determinism tests |
| `panelary/__init__.py`, `leakage/__init__.py`, `CHANGELOG.md`, `mkdocs.yml` | orchestrator | none needed beyond what the report lists |

**Dependencies.** numpy + polars only. sklearn is used if present (for
`clone`), never required; the default scorers are numpy.

**Out of scope.**

- Observational do-Shapley estimators (see above).
- Asymmetric, causal-order Shapley. For a chain of pipeline stages it reduces to
  cumulative ablation in pipeline order, which hands a joint leak entirely to
  whichever stage comes second: the order-dependent answer that exact
  decomposition exists to avoid.
