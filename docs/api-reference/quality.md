# quality

Data validation for panels. It checks panel invariants, column contracts and
leak-safety invariants, then splits rows into valid and invalid. Separately, it
can produce a data-quality profile. See the
[Data Quality guide](../user-guide/data-quality.md) for the walkthrough.

## What's here

- `PanelValidator`: configure the checks once, then call `.validate(df)` for a
  report or `.filter(df)` for `(valid_rows, report)`.
- `validate_panel`: the same checks as a single function call.
- `ColumnContract`: the contract for one column, covering dtype, nullability,
  NaN, uniqueness, allowed values and range.
- `check_fitted_state`: tests whether a fitted transform's stored state derives
  only from train rows, using provenance and a counterfactual refit.
- `check_near_duplicate_straddle`: tests whether a near-duplicate cluster has
  rows on both sides of a split.
- `quality_report`: reports null %, duplicate %, constant columns and dtype
  drift against a reference.
- Result types:
    - `CheckResult`: one finding.
    - `ValidationReport`: all findings, plus `valid` and `invalid` rows.
    - `QualityReport`: the profile.

  Each has `to_dict()`, a byte-deterministic `to_json()`, and `to_evidence()`.
- `PanelValidationError`: raised when a check with `impact="fail"` fails. The
  report is on `.report`.
- `QualityWarning`: the category for warn-level findings. Row-order findings use
  `PanelOrderWarning` and leak-safety findings use `LeakageWarning` instead.
- `DEFAULT_IMPACT`: the default impact level of each check.

## JSON convention

`to_json(indent=None)` returns
`json.dumps(to_dict(), sort_keys=True, separators=(",", ":"))`.

The top level of the output carries two keys:

- `"schema"`, for example `"panelary.ValidationReport/1"`;
- `"produced_by"`, for example `"panelary.quality.validate_panel@0.5.0"`.

Each check carries its `locator`, `observed` and `expected` values. The locator
has the form `panelary.quality/<check>[/<target>][#<offending keys as canonical
JSON>]`.

## API

::: panelary.quality
