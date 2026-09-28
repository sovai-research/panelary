# Point-in-time evidence: as-of join, calendar embargo, serialisable audits

> **Status (2026-09-28): implemented.** Shipped: `panelary/core/asof.py`
> (`asof_join`, plus additive `PanelFrame.asof_join`), `panelary/core/_calendar.py`
> (duration vocabulary, `BusinessDays`, `shift_forward`), calendar
> `horizon`/`embargo` in `core/model_selection.py` (`PurgedKFold`,
> `CombinatorialPurgedCV`, `_train_positions`) and `validation/_cv.py`
> (`times=` on the positional helpers), `to_dict()`/`to_json()` on
> `leakage.Finding` / `CompileResult` / new `FeatureSetAudit` and on
> `core.pipeline.PipelineAudit` / `StepAudit`, and `leakage.audit_features`
> (one-call sweep of a named feature set). Tests: `tests/test_asof_join.py`,
> `tests/test_embargo_calendar.py`, `tests/test_leakage_json.py`. Docs:
> `docs/api-reference/{leakage,cross-validation,panel-frame}.md`. Deferred: a
> sweep over a *registry namespace* (specs are operators with parameters, not
> expressions -- there is nothing to audit until someone binds arguments);
> top-level `pn.*` exports (orchestrator-owned); the engine-side adapter in
> `truepoint/quant/` (not this repo).

**Stage:** build contract, written after the fact (the work had no plan file).
**Customer:** the AgenticFinance assessment engine (`AGENTS.md`), caller
`truepoint/src/truepoint/quant/` -- leakage findings become
`Evidence(kind=STATIC_ANALYSIS)` in an assessment report.

## What the engine consumes

| Engine need | Panelary surface |
| --- | --- |
| A leakage finding as report evidence | `CompileResult.to_json()`, `FeatureSetAudit.to_json()`, `PipelineAudit.to_json()` |
| A whole feature set checked in one call | `panelary.leakage.audit_features({name: expr}, time=, entity=)` |
| "What was known at t" for restated data | `panelary.core.asof.asof_join(panel, vintages, ...)` |
| An embargo in business days, not rows | `PurgedKFold(embargo="5bd")`, `BusinessDays(5, holidays=...)` |

Evidence mapping (documented in `docs/api-reference/leakage.md`, exercised in
`tests/test_leakage_json.py`): `locator = "feature:<name>#<finding.locator>"`,
`observed = {expr: source, node: kind, classification, reason}`,
`expected = {expr: compiled-or-null, rewrote_to}`, `produced_by =
"panelary.leakage.audit@<version>"`.

## API

```python
# serialisation convention (shared with BorrowedAccuracyReport)
obj.to_dict() -> dict            # JSON-safe; "schema": "panelary.<Type>/1",
                                 # "produced_by": "panelary.<module>.<fn>@<version>"
obj.to_json(*, indent=None) -> str  # json.dumps(..., sort_keys=True, separators=(",", ":"))

audit_features(features, *, time=None, entity=None, allow_approximate=False, trust=()) -> FeatureSetAudit

asof_join(panel, vintages, *, entity=None, time=None, event_time="event_time",
          knowledge_time="knowledge_time", vintage_entity=None, values=None,
          lag=None, suffix="", provenance=False)

PurgedKFold(n_splits, *, horizon: int | Duration = 0, embargo: int | Duration = 0, ...)
Duration = int | timedelta | np.timedelta64 | "5d"/"1mo"/... | "5bd" | BusinessDays
```

## Invariants (each has a test)

1. **as-of is strictly causal**: the value at `(e, t)` depends only on vintages
   with `max(knowledge_time + lag, event_time) <= t` -- `assert_no_lookahead`
   and `assert_prefix_invariant` run over the vintage table keyed by knowledge
   time, and over the panel; a latest-vintage join fails both.
2. **as-of is order independent** and returns rows in the caller's order.
3. **as-of fails closed**: null vintage keys, conflicting duplicate vintages,
   Date-vs-Datetime and time-zone mismatches raise.
4. **Integer embargo/horizon are byte-identical** to the pre-change algorithm
   (compared against a frozen copy in the test file).
5. **Calendar embargo** blocks exactly `(block_end, block_end + duration]`,
   checked against hand-written windows and an independent oracle on
   irregular axes (weekend rows, holidays, gaps, intraday across DST).
6. **Serialised evidence is byte-deterministic** across processes and hash
   seeds; no memory address survives.
