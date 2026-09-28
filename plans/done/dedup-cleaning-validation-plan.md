# Implementation Plan — Deduplication, Data Cleaning & Data Validation (the fifth pillar)

> **Status — quality/ (2026-09-28): M2 and M6 implemented; zero new mandatory dependencies.**
> `panelary/quality/` exports `PanelValidator`, `validate_panel()`, `ColumnContract`, `check_fitted_state()`, `check_near_duplicate_straddle()`, `quality_report()`, and the result types `CheckResult` / `ValidationReport` / `QualityReport`.
> **Panel invariants:** usable keys, null keys, unique `(entity, time)`, per-entity time order (measured via `PanelFrame.is_sorted_per_entity`, warned as `PanelOrderWarning`), gaps against the panel's own calendar or an explicit `frequency`, `min_obs`, `coverage`.
> **Contracts:** dtype, nullability, NaN, unique, allowed values, range, named rules.
> **§6 leak-safety invariants:** fitted-state provenance, checked two ways (the `_fit_panel` keys, plus a counterfactual refit on train compared against learned state); and near-duplicate straddle over `panelary.clean.near_duplicate_clusters`, against an `IndexSplit`, a `(train, test)` pair, a fold-label column or a list of folds, warned as `LeakageWarning`.
> **Behaviour:** dataframely-style `valid` / `invalid` split with `__failed_checks__`; fail / warn / off impact levels; `PanelValidationError` carries the report.
> **Serialisation:** deterministic `to_dict()` / `to_json()` / `to_evidence()`. The evidence output was checked against the engine's real `Evidence` model.
> **M6:** `quality_report()` covers null %, NaN %, duplicate rows and keys, constant columns (global and within-entity), entities per time, and dtype drift against a reference. The optional `dataframely` backend takes a user `dy.Schema` or `backend="dataframely"`. The adapter is tested only against a stand-in module, because dataframely is not installed here; the real-library test is `importorskip`.
> **Tier-1 surface:** `.panel.validate()` and `.panel.quality_report()` on DataFrame and LazyFrame, imported lazily.
> **Tests:** `tests/test_quality_{panel,schema,leakage,report}.py`.
> **Docs:** `docs/user-guide/data-quality.md`, `docs/api-reference/quality.md`.
> **Deferred:**
> - `PanelValidator` is deliberately not a `PanelTransformer`. Validation learns nothing (§5, step 1).
> - The `panel.pyi` stub entries, the extras / `_MODULE_TO_EXTRA` entry for `schema`, and the top-level export are pending the orchestrator.

> **Status — clean/ (2026-09-28): M1, M3, M4 and M5 implemented; zero new mandatory dependencies.**
> **M1:** `Deduplicator` / `dedup` (exact, MinHash, C-MinHash, SimHash; LSH banding, exact verification, b-bit), `near_duplicate_clusters` / `near_duplicate_pairs`, and the split-aware `straddling_pairs`, `assert_no_straddle`, `straddling_clusters`, `purge_near_duplicates`, `SplitAwareCV`. `preprocessing.reindex(drop_duplicates=True)` now routes through `clean.exact_unique`, with unchanged behaviour.
> **Design decision (refines §2 step 4):** connected-component dedup reads the future, because a later row can bridge two earlier clusters. The default therefore drops a row only if it near-duplicates an *earlier* row: pairwise, prefix-invariant, `leakage_safe`. `linkage="component"`, `keep="last"` and `survivorship=` set the instance's `leakage_safe = False`.
> **M3:** `normalize_text`, `fingerprint`, `ngram_fingerprint`, `fingerprint_clusters`, and `Canonicalizer`, whose learned `vocab_` is fit on train only. `Survivorship` / `golden_records` (ten rules plus callables), wired into `Deduplicator(linkage="component", survivorship=...)`.
> **M4:** `OutlierCleaner`: MAD / IQR / z-score / quantile bounds fit on train (per entity with a pooled fallback, pooled, or per-date cross-section), plus causal Hampel and rolling filters.
> **M5:** `EntityResolver` / `resolve_entities` (exact-key, sorted-neighbourhood and `LSHBlocking` OR-blocking; weighted field comparison; union-find), and `string_similarity`, which uses `rapidfuzz` when installed and a NumPy / pure-Python fallback otherwise. At transform, ids unseen at fit are matched only against the training records and are never merged with each other.
> **Plan correction (§4, M4):** `preprocessing.scale` (per-entity mean/std) and `preprocessing.trim` (time-range alignment) compute no robust statistic, so there is nothing to share with `OutlierCleaner`. Both are unchanged.
> **Tests:** `tests/test_clean_{dedup,outliers,canonicalize,resolve}.py`. Every fitted state has a deliberately leaky variant that must fail. The datasketch oracle and rapidfuzz parity tests are `importorskip`.
> **Docs:** `docs/user-guide/cleaning.md`, `docs/api-reference/clean.md`.
> **Deferred:**
> - The `splink` (`er`) backend, semantic dedup and suffix arrays (M7).
> - §4's `_estimators.py` was not needed; each transformer lives beside its kernels.
> - Pending the orchestrator: the `fuzzy` extra and `rapidfuzz → fuzzy` in `_MODULE_TO_EXTRA`. Until they land, `tests/test_dependency_drift.py::test_every_require_call_maps_to_a_declared_extra` flags `_strsim.py`. Also pending: the top-level `clean` export, `.panel.dedup()`, the mkdocs nav and the CHANGELOG.

**Status:** implemented 2026-09-28 (M1–M6; M7 deferred) — see the two status blocks above
**Author of plan:** research + design session, 2026-09-08 (5-agent sweep: 60+ repos/methods across Rust/Polars dedup, Python ER, dedup academia, Polars validation libs, cleaning frameworks)
**Branch context:** `panelkit-roadmap-impl` (branch name predates the PanelKit → Panelary rename; left as-is)
**Siblings (already implemented — in `plans/done/`):** [factor-extraction-plan.md](../done/factor-extraction-plan.md) · [shap-attribution-plan.md](../done/shap-attribution-plan.md) · [interactions-theme-plan.md](../done/interactions-theme-plan.md) · [econometric-integration-plan.md](../done/econometric-integration-plan.md)
**This plan lives in `plans/done/`.** M7 (semantic dedup, Splink backend, suffix arrays) remains optional future work.

---

## 0. TL;DR — the decision

Add two new leak-safe subpackages that ride the existing `PanelTransformer` contract:

- **`clean/`** — panel-aware **deduplication**, **entity resolution**, **canonicalization**, **survivorship**, and **robust outlier cleaning**.
- **`quality/`** — panel-aware **data validation**: schema/type contracts, **panel invariants** (unique `(entity,time)`, monotone time, no gaps, coverage, min-obs), and **leak-safety invariants**. (Named `quality/`, **not** `validation/` — that name is already the *statistical* honesty layer: CPCV, Deflated Sharpe, bootstrap.)

**The dependency decision (the load-bearing one).** Panelary's mandatory footprint is exactly `numpy + polars`, the wheel is pure-Python (hatchling), and there is **no Rust extension** today. Therefore **everything here ships pure Polars + numpy, adding zero mandatory dependencies.** The one genuinely worth-it fast kernel (`rapidfuzz`) and the one nice-to-have declarative backend (`dataframely`) are **optional extras**, lazy-imported through the existing `require()` / `_MODULE_TO_EXTRA` machinery, each with a pure-Polars/numpy fallback so the capability is never *gated* on the extra.

**The moat.** Every dedup engine and validation library surveyed leaves the train/test boundary to the caller, and none is panel-native. Panelary's differentiator is to make **"dedup and validation happen across the whole panel, and dedup collapses near-duplicate clusters *before* any split"** a first-class, `PanelTransformer`-enforced guarantee — wired into the existing `validation/_cv.py` purged/embargoed splitters. **No competitor does panel-global, split-aware dedup, and none validates ML leak-safety invariants on Polars.** That is unowned territory.

Must-ship: **M1** (pure-Polars `Deduplicator`: exact + MinHash-LSH near-dup, panel-global & split-aware) and **M2** (`quality.PanelValidator`: panel + leak-safety invariants).

---

## 1. Why this is a natural fifth pillar

The four existing plans form an arc: **features → factors (`reduce`) → attribution (`explain`) → honest validation (`validation`)**. This plan supplies the **input-integrity** stage that logically precedes all of them:

> **clean & validated inputs → features → factors → attribution → honest validation**

It is the same leak-safety throughline, one stage earlier: a fitted statistic downstream is only as honest as the rows it was fit on. Duplicate rows silently corrupt every train-only estimate (means, quantiles, MAD, imputation regressions, factor covariances); near-duplicate rows straddling a CV split leak the target. Cleaning and validating *on the whole panel, before the split* is the precondition that makes the other four pillars' leak-safety claims true.

---

## 2. Dependency posture — how we implement this "for us"

Ranked by preference, the implementation strategy is **pure-Polars-first, native fallback always, extra only for speed**:

| Tier | Mechanism | What lives here |
|---|---|---|
| **0 — pure Polars expressions (zero deps)** | `pl.Expr`, `.over(entity)`, `.str.*`, `group_by/agg`, `.hash(seed=…)`, `.unique/is_duplicated` | exact dedup; MinHash signatures + LSH banding; canonicalization; survivorship; panel & leak-safety validation; data-quality report |
| **0 — numpy (already a core dep)** | small vectorized kernels | union-find over candidate edges; SimHash random-projection + popcount; C-MinHash permutation; edit-distance fallback |
| **1 — optional extra, lazy + fallback** | `require("rapidfuzz", extra="fuzzy")` etc. | `rapidfuzz` fast fuzzy scoring (`fuzzy`); `dataframely` declarative schema backend (`schema`); `splink` heavy probabilistic linkage (`er`) |
| **2 — deferred / moonshot** | future extra | semantic dedup embeddings+ANN (`semantic`); suffix-array long-text dedup; re-home hot kernels to a Rust ext *iff one is ever reintroduced* |

**The central technical insight — near-dup dedup with zero dependencies.** MinHash-LSH is fully expressible in Polars:

1. **Shingle/tokenize** the dedup key into a `List` column (Polars `.str.split` / n-gram expr; for row dedup the key is the value tuple).
2. **MinHash signature:** for each of `k` seeds `s`, `pl.col("shingles").list.eval(pl.element().hash(seed=s)).list.min()` → signature slot `s`. Loop is over `k` seeds (~64–128), never over rows. (Or one hash + a numpy permutation table for the C-MinHash/OPH speedup — still zero-dep.)
3. **LSH banding:** split the `k` slots into `b` bands of `r`; hash each band tuple to a bucket id; `group_by(band, bucket).agg(row_ids)` → any bucket with >1 row is a candidate group. The S-curve threshold `(1/b)^(1/r)` is the tunable knob.
4. **Cluster:** feed candidate edges to a numpy union-find → connected components → one canonical row per component (survivorship rules).
5. **Verify (optional):** exact Jaccard on candidate pairs to prune LSH false positives.

Steps 1–3 and survivorship are pure Polars; step 4 is ~30 lines of numpy. **No rensa, no datasketch, no pyo3-polars required** — those become *idea/oracle* references (datasketch as a correctness oracle in tests), not dependencies. If a Rust extension is ever reintroduced, steps 2 and 4 are the hot kernels to move there behind the same Python API.

---

## 3. Ranked technique shortlist → what we actually build

From the 5-agent sweep, filtered by the zero-dep constraint (full ranking lives in [[panelkit-dedup-validation-research]] — memo filed under its pre-rename name):

| Rank | Technique | How we ship it |
|---|---|---|
| 1 | Exact-row dedup (`.unique`/`is_duplicated`) | pure Polars; `Deduplicator(method="exact")`. Supersede/wrap `preprocessing.reindex(drop_duplicates=)`. |
| 2 | MinHash + LSH banding + union-find | pure Polars + numpy (§2); `Deduplicator(method="minhash")`. **Panel-global, split-aware.** |
| 3 | C-MinHash / OPH + optimal densification | numpy permutation kernel; better accuracy-per-hash, same API. |
| 4 | b-bit signature compression | numpy dtype narrowing on signatures; memory knob. |
| 5 | SimHash / SRP + Hamming-LSH | numpy sign-of-projection + popcount; `Deduplicator(method="simhash")` for **numeric/embedding feature rows**. |
| 6 | Robust outlier cleaning (MAD/IQR/Hampel/rolling) | pure Polars, panel-aware, fit-on-train; `OutlierCleaner`. Reconcile with `preprocessing.scale/trim`. |
| 7 | Canonicalization (NFC/casefold/whitespace/regex) + fingerprint clustering | pure Polars `.str`; `Canonicalizer`. Must precede dedup. |
| 8 | Survivorship / golden-record | pure Polars `group_by/agg` (recency/completeness/source-priority). |
| 9 | Blocking / sorted-neighborhood | pure Polars keys (entity prefix, time bucket, phonetic); feeds ER. |
| 10 | Fuzzy scoring for ER | **optional** `rapidfuzz` (`fuzzy` extra) + numpy edit-distance fallback; `EntityResolver`. |
| 11 | Panel + leak-safety validation | pure Polars `pl.Expr.over(entity)`; `quality.PanelValidator`. |
| 12 | Declarative schema contracts | pure-Polars default; **optional** `dataframely` (`schema` extra) backend. |
| — | Splink (probabilistic linkage), SemDeDup/SemHash (semantic), suffix arrays (long text) | deferred to `er`/`semantic` extras / M7. |

**Copyleft to avoid depending on (concepts only):** Zingg, cleanlab (AGPL); legacy fuzzywuzzy (GPL). Verify SemDeDup/polars-strsim licenses before any adoption.

---

## 4. Module layout

```
panelary/clean/
  __init__.py        # public: Deduplicator, EntityResolver, Canonicalizer, OutlierCleaner, dedup(), ...
  _dedup.py          # exact + near-dup (MinHash-LSH) row dedup; panel-global, split-aware
  _sketch.py         # MinHash / C-MinHash / OPH / SimHash / b-bit kernels (Polars + numpy)
  _cluster.py        # numpy union-find over candidate edges → components
  _canonicalize.py   # unicode NFC, casefold, whitespace, regex; fingerprint/n-gram clustering
  _resolve.py        # entity resolution: blocking + compare (rapidfuzz optional) + survivorship
  _survivorship.py   # golden-record merge rules (recency/completeness/source-priority)
  _outliers.py       # robust MAD/IQR/Hampel/rolling — panel-aware, fit-on-train
  _estimators.py     # PanelTransformer wrappers (fit-on-train, split-aware)
  _common.py

panelary/quality/            # DATA validation — distinct from statistical validation/
  __init__.py        # public: PanelValidator, validate_panel(), quality_report()
  _panel.py          # panel invariants: unique (entity,time), monotone time, gaps, coverage, min-obs
  _schema.py         # dtype/key/nullability contracts; optional dataframely backend
  _leakage.py        # leak-safety invariants (reconcile with testing.assert_no_lookahead + LeakageWarning)
  _report.py         # lightweight data-quality report (null%, dup%, constant cols, dtype drift)
  _common.py
```

**Tier-1 namespace surface** (mirroring `.panel`/`.xs`, which must not import the estimator layer):
- `df.panel.dedup(...)`, `df.panel.validate(...)`, `df.panel.quality_report()`
- Estimator layer (`clean.Deduplicator(...)`, `quality.PanelValidator(...)`) for the fit-on-train contract.

**Reconciliation (do not duplicate):**
- `preprocessing.reindex(drop_duplicates=)` → becomes a thin call into `clean.Deduplicator(method="exact")`; keep the old signature working.
- `preprocessing.scale/trim` overlap `OutlierCleaner` → share the robust-stat helpers; `OutlierCleaner` adds the fit-on-train + panel-aware layer.
- `preprocessing.LeakageWarning` and `testing.assert_no_lookahead` → the canonical leak primitives `quality/_leakage.py` builds on; do not invent parallel ones.
- `validation/_cv.py` purged/embargoed splitters → `Deduplicator` integrates so cluster-collapse happens **before** fold assignment (the split-aware guarantee).
- `imputation.py` (CAFE) stays the **last** cleaning stage (§5).

---

## 5. The leak-safe cleaning pipeline (ordering doctrine)

The organizing rule the sweep converged on: **deterministic/identity work first (leak-safe on all rows); anything that *learns* a parameter last, fit on the train partition only, never using an entity's future.** Dedup sits deliberately in the middle.

1. **Schema / integrity gate** — `quality`: dtypes, unique `(entity,time)`, monotone time. Deterministic → all rows.
2. **Canonicalization** — `clean.Canonicalizer`: NFC/casefold/whitespace/regex, category crosswalks. Deterministic → all rows. **Must precede dedup** so variants collapse to one key.
3. **Dedup + survivorship** — `clean.Deduplicator` / `EntityResolver`: collapse exact/near-dup rows, resolve entities, pick surviving values. Panel-global; **before any split**. (Any *learned* matcher/TF table fit on train only.)
4. **Constraint / FD checks** — `quality`: flag rows breaking hard rules. Deterministic → all rows (mined rules on train only).
5. **Outlier treatment** — `clean.OutlierCleaner`: caps/flags with thresholds **fit on train only** (per-entity or pooled); causal rolling windows for time series.
6. **Imputation (last)** — `imputation.py` / CAFE: **fit on train only**, applied downstream, benefiting from cleaned, deduped, outlier-treated inputs.

Why dedup before the fitted stats: duplicate rows inflate an entity's effective sample and skew every train-only estimate; deduping first makes those fits correct and avoids baking duplicate bias into stored parameters.

---

## 6. Leak-safety guardrails (the invariants `quality/_leakage.py` enforces)

- **Panel-global dedup, split-aware collapse.** Clusters are computed over all entities × all time and collapsed **before** fold assignment; a near-dup pair may never land in different folds. Assert against `validation/_cv.py` fold boundaries.
- **Fit-on-train only.** Every stateful step (LSH index, EM/TF tables, TF-IDF vocab, robust thresholds, embeddings) stores parameters from `_fit(train)` and only reads them in `_transform`. Stateless kernels (rapidfuzz, hashing, Polars `.unique`) carry no risk.
- **No future leakage.** Per-entity rolling/expanding windows are causal (past-only); no cross-entity or forward information in any fitted statistic.
- **The novel checks** (no competitor offers these on Polars): "does a fitted transform's stored state derive only from train rows?" and "do any near-duplicate rows straddle the train/test boundary?" — surface as `PanelValidator` assertions and a `LeakageWarning`.

---

## 7. Optional extras (added to pyproject + `_MODULE_TO_EXTRA`)

```
fuzzy  = ["rapidfuzz"]      # fast stateless fuzzy scoring for EntityResolver; numpy fallback otherwise
schema = ["dataframely"]    # declarative schema backend for quality; pure-Polars default otherwise
er     = ["splink"]         # heavy probabilistic linkage (DuckDB/Arrow); deferred/optional
# semantic = [...]          # M7: embeddings + ANN semantic dedup
```
Map `rapidfuzz→fuzzy`, `dataframely→schema`, `splink→er` in `_MODULE_TO_EXTRA`. Each import routed through `require(...)` so a missing extra yields the actionable `pip install 'panelary[fuzzy]'` hint — never a hard failure of the pure-Polars path.

---

## 8. Milestones

- **M1 (must-ship) — `clean.Deduplicator`, zero deps.** Exact dedup (wrap Polars `.unique`) + MinHash-LSH near-dup (§2) + numpy union-find + optional exact-Jaccard verify. `PanelTransformer`, panel-global, split-aware (integrate `validation/_cv.py`). datasketch as test oracle.
- **M2 (must-ship) — `quality.PanelValidator`, zero deps.** Panel invariants + leak-safety invariants as `pl.Expr.over(entity)`; dataframely-style valid/invalid split; Validoopsie-style warn-vs-fail impact levels; Tier-1 `.panel.validate()`.
- **M3 — `clean.Canonicalizer` + survivorship, zero deps.** Polars `.str` normalizers + fingerprint/n-gram clustering; golden-record merge rules.
- **M4 — `clean.OutlierCleaner`, zero deps.** Robust MAD/IQR/Hampel/rolling, panel-aware, fit-on-train; reconcile `preprocessing.scale/trim`.
- **M5 — `clean.EntityResolver`.** Pure-Polars blocking + compare with optional `rapidfuzz` (`fuzzy`) and numpy fallback; SimHash/C-MinHash sketch upgrades in `_sketch.py`.
- **M6 — declarative + reporting.** Optional `dataframely` (`schema`) backend for `quality`; `quality.quality_report()` (pointblank-style HTML/summary).
- **M7 (optional/moonshot).** Semantic dedup (`semantic` extra: embeddings + ANN); Splink (`er`) backend; suffix-array long-text dedup; re-home hot MinHash/union-find kernels to a Rust ext **iff** one is reintroduced.

M1+M2 deliver the differentiated core with **no new dependencies**.

---

## 9. Out of scope
- GPU dedup (NeMo Curator/RAPIDS), Spark/cluster pipelines (datatrove) — wrong footprint.
- Deep entity matching (Ditto/DeepMatcher) — accuracy-first, too heavy.
- A general-purpose validation framework (great_expectations/soda/dqx) — we build a thin panel-aware layer, not a platform.
- Interactive active-learning ER UX (dedupe/zingg) — batch pipeline only.

## 10. Licensing / clean-room
- Ship pure Polars+numpy; no code copied from surveyed repos. datasketch (MIT) usable as a test-time correctness oracle only.
- **Never depend on** AGPL (Zingg, cleanlab) or GPL (legacy fuzzywuzzy) — concepts only.
- Optional extras are all permissive (rapidfuzz MIT, dataframely BSD-3, splink MIT). Verify SemDeDup / polars-strsim before any M7 adoption.

## 11. Cross-references
- Reuses `validation/_cv.py` (split-aware dedup), `testing.assert_no_lookahead` + `preprocessing.LeakageWarning` (leak primitives), `imputation.py` (pipeline tail), `namespaces/{panel,xs}.py` (Tier-1 surface), `select/_unsupervised.py` (estimator pattern to mirror).
- Positioned as the input-integrity stage ahead of the other four plans.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
