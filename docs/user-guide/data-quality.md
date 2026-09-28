# Data Quality

Every leak-safety guarantee in Panelary assumes the panel it runs on is what it
claims to be: one row per `(entity, time)`, each entity's rows in time order,
no silent holes, and fitted statistics that saw only training rows.
`panelary.quality` checks those assumptions **before** anything is fitted.

It is the *data* integrity gate, and it is distinct from
[`panelary.validation`](honest-validation.md), the *statistical* honesty layer
(purged CV, Deflated Sharpe, bootstraps). Run this first.

```python
from panelary.quality import (
    PanelValidator, ColumnContract, validate_panel, quality_report,
    check_fitted_state, check_near_duplicate_straddle,
)
```

## One call

```python
import polars as pl
from panelary.quality import ColumnContract, PanelValidator

df = pl.DataFrame({
    "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
    "date":   [1, 2, 4, 1, 2, 2, 3],
    "ret":    [0.01, None, 0.03, -0.02, 0.00, 0.00, 0.01],
})

validator = PanelValidator(
    entity="ticker",
    time="date",
    schema={"ret": ColumnContract(dtype=pl.Float64, nullable=False)},
)
valid, report = validator.filter(df)
print(report.summary())
```

```text
ValidationReport[fail] 7 rows, 2 entities, 4 times ('ticker', 'date'): 8 checks, 2 failed, 1 warned
  FAIL unique_keys: 1 (ticker, date) key(s) occur more than once, covering 2 row(s). ...
  WARN gaps: 1 entity has 1 internal gap(s) totalling 1 missing time step(s) against the panel calendar. ...
  FAIL nullability:ret: 1 null value(s) in non-nullable 'ret'.
```

`filter` splits the rows the way `dataframely` does: `valid` holds the four
clean rows, and `report.invalid` holds the other three with the ids of the
checks each one failed:

```text
┌────────┬──────┬──────┬─────────────────────┐
│ ticker ┆ date ┆ ret  ┆ __failed_checks__   │
╞════════╪══════╪══════╪═════════════════════╡
│ AAA    ┆ 2    ┆ null ┆ ["nullability:ret"] │
│ BBB    ┆ 2    ┆ 0.0  ┆ ["unique_keys"]     │
│ BBB    ┆ 2    ┆ 0.0  ┆ ["unique_keys"]     │
└────────┴──────┴──────┴─────────────────────┘
```

Both copies of a duplicated key are rejected: which copy is right is a
survivorship decision for `panelary.clean`, not for a validator.

`validate_panel(df, entity=..., time=..., ...)` is the same thing as a function,
and it is on every frame too: `df.panel.validate(entity="ticker", time="date")`,
`lf.panel.validate(...)`.

## What is checked

| Check | Default impact | What fails it |
| --- | --- | --- |
| `key_columns` | fail (always) | entity/time missing, equal, or time not orderable |
| `null_keys` | fail | a null entity or time |
| `unique_keys` | fail | an `(entity, time)` pair that occurs twice |
| `monotone_time` | warn | time going backwards within an entity, in frame order |
| `gaps` | warn | an entity skipping a time step between its first and last row |
| `min_obs` | fail | an entity with fewer than `min_obs` distinct times (opt-in) |
| `coverage` | warn | an entity on less than `min_coverage` of the calendar (opt-in) |
| `dtype`, `nullability`, `nan`, `unique`, `allowed_values`, `range` | fail | a `ColumnContract` violated |
| `columns_present`, `extra_columns` | fail | a required column missing; with `strict=True`, an unexpected one |
| `rule` | fail | a named boolean expression false on a row (null counts as valid) |
| `fitted_state` | fail | a fitted transform whose state did not come from train rows alone |
| `near_duplicate_straddle` | fail | a near-duplicate cluster with rows on both sides of a split |

Override any level with `impact={"monotone_time": "fail", "gaps": "off"}`.

- **`fail`** makes the report fail. `validate` raises
  `PanelValidationError` (the report rides on `err.report`) unless
  `raise_on_fail=False`; a *row-level* failure also sends the offending rows to
  `report.invalid`.
- **`warn`** records the finding and emits a warning, but rejects nothing.
  Row-order findings are raised as `PanelOrderWarning` and leak-safety findings
  as `LeakageWarning`, so existing filters for those keep working; the rest are
  `QualityWarning`.
- **`off`** does not run the check.

### Row order is measured, not trusted

`monotone_time` asks `PanelFrame.is_sorted_per_entity()` on a fresh frame, so a
`mark_sorted()` promise does not satisfy it. The observed
`PanelFrame.sortedness` (`"panel"`, `"time"` or `"unsorted"`) is part of the
finding. It warns by default because `.over(entity)` on an unsorted panel
returns a plausible wrong number rather than an error. Sort it, or write the
expression with `PanelFrame.within_entity`.

### Gaps without a calendar library

By default a gap is measured against the **panel calendar**, meaning every time
at which *any* entity is observed. Business-day and exchange calendars need no
configuration, because the panel is its own calendar. Pass `frequency="1d"`,
`"1mo"`, a `timedelta`, or a number for a numeric time axis to check against a
regular grid instead. `"1mo"` respects month ends. A late listing or an early
delisting is not a gap; `min_coverage` is the check for that.

## Leak-safety invariants

These two checks need a split, so they run only when you pass one:

```python
report = validator.validate(
    df,
    split=(train, test),           # or an IndexSplit, a fold-label column, or a list of folds
    fitted={"scaler": scaler},     # fitted transforms to audit
)
```

**Does a fitted transform's stored state come only from train rows?**
`check_fitted_state` answers this in two independent ways, and fails if either
one fails:

1. **Provenance.** A `PanelTransformer` records the panel it was fitted on. If
   any of that panel's `(entity, time)` keys lies outside `train`, you have a
   direct witness, and those keys become the locator.
2. **Counterfactual refit.** A deep copy is refitted on `train` alone, and its
   learned state (the sklearn-convention trailing-underscore attributes) must
   equal the original's. This works for any object with a `fit` method. It
   catches a scaler fitted on the whole panel, and also a transformer that
   quietly re-learns its state inside `transform`, where provenance is clean.

State tables are compared as sets of rows, because polars does not guarantee
the output order of `group_by`. The refit assumes the fit is deterministic,
which the library's seeded-RNG contract already requires. A fit that reads
data it was never given (a frame captured in a closure, say) reproduces the
same leak on refit, so this check cannot see it.

**Do near-duplicate rows straddle the train/test boundary?**
`check_near_duplicate_straddle` clusters rows across the whole panel with
`panelary.clean.near_duplicate_clusters`. It then fails if a cluster has rows
on both sides of the split. A row that sits in both train and test counts as
well. Purged or embargoed rows, which belong to neither side, are ignored. If
you already have cluster ids, pass them with `clusters=`. The fix is the one
`panelary.clean` is built for: collapse near-duplicates across the whole panel
*before* you assign folds.

## Evidence

Every result follows the same serialisation convention:

- `to_dict()` returns a plain dict that is safe to serialise as JSON.
- `to_json()` is byte-deterministic: `json.dumps(to_dict(), sort_keys=True,
  separators=(",", ":"))`. It contains no timestamps, and each offending-key
  sample is sorted before it is truncated.
- The top-level keys include `"schema"` (`"panelary.ValidationReport/1"`) and
  `"produced_by"` (`"panelary.quality.validate_panel@0.5.0"`).
- Each finding carries a `locator`, plus `observed` and `expected`.

```python
report.check("unique_keys").locator
# 'panelary.quality/unique_keys#[{"count":2,"keys":{"date":2,"ticker":"BBB"}}]'

report.to_evidence()[0]
# {'kind': 'schema-validation',
#  'locator': 'panelary.quality/unique_keys#[{"count":2,"keys":{"date":2,"ticker":"BBB"}}]',
#  'observed': {'duplicated_keys': 1, 'rows': 2},
#  'expected': {'duplicated_keys': 0},
#  'produced_by': 'panelary.quality.PanelValidator@0.5.0'}
```

The keys of each evidence record are the fields of an assessment-engine
evidence record. `kind` is `schema-validation` for panel and schema checks,
`counterfactual-run` for the fitted-state refit, and `static-analysis` for the
straddle check.

## Profiling: `quality_report`

`quality_report` describes a frame without passing judgement on it:

```python
rep = quality_report(df, entity="ticker", time="date", reference=train_df, max_null_pct=0.2)
rep.to_frame()          # per column: dtype, null %, NaN %, distinct, constant, ...
rep.duplicates          # exact duplicate rows and duplicate (entity, time) keys, with %
rep.constant_columns    # at most one distinct non-null value
rep.dtype_drift         # changed / added / removed columns vs `reference`
rep.findings            # warn-level CheckResults, with to_evidence()
```

When you give it panel keys, the profile also covers:

- entities per time;
- `constant_within_entity`, which flags a static attribute such as a sector
  code: it never varies inside an entity, so it adds nothing to a per-entity
  model;
- the worst null rate of any single entity.

`df.panel.quality_report(...)` is the frame-level spelling.

## Backends

The default is pure Polars and adds no dependencies. With the `schema` extra
(`pip install 'panelary[schema]'`) you can pass a `dataframely.Schema` subclass
as `schema`. Its rule counts land in one `dataframely` check, and the rows it
rejects join `report.invalid`. Alternatively, keep a mapping of contracts and
set `backend="dataframely"`. In that mode dtype and nullability go to
dataframely, while uniqueness, allowed values, ranges and rules stay in Polars.
Panel and leak-safety checks are pure Polars whichever backend you choose.

## Where it sits

The cleaning doctrine orders the stages like this:

1. Schema and integrity gate (`quality`), on all rows.
2. Canonicalise.
3. Deduplicate (`clean`), across the whole panel and before any split.
4. Constraint checks (`quality`).
5. Outlier caps, fit on train.
6. Imputation, fit on train.

Validation learns nothing, so it has no `fit` and runs on every row. The one
learned quantity it looks at is a transform's fitted state, and auditing that
state is exactly what `fitted_state` does.
