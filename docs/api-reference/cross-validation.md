# Cross-validation splitters

`panelary.cross_validation` holds the **simple, chronological** panel splitters: a single
train/test cut and the two classic rolling-origin schemes. Each returns a callable that takes a
panel frame and yields `(train, test)` frames, split **within each entity** in time order, so a
test fold is always strictly later than its training fold for every entity.

These splitters purge nothing. They are the right tool when observations are point-in-time and
non-overlapping. The moment labels *span* time — a triple-barrier label, a multi-period forward
return — an adjacent train row overlaps its test fold and the split leaks; use the purged and
embargoed splitters in [`validation`](validation.md) instead.

## What's here

| Splitter | Shape |
| --- | --- |
| `train_test_split` | One chronological cut per entity, by count or fraction |
| `expanding_window_split` | Growing train window, fixed test window (rolling origin) |
| `sliding_window_split` | Fixed-length train window, fixed test window |

## Calendar-aware embargo and purge

The purged splitters — `PurgedKFold` and `CombinatorialPurgedCV` in
`panelary.core.model_selection`, and the positional `cpcv_splits`,
`walk_forward_splits` and `purged_calibration_split` in `validation` — take
`embargo` and `horizon` either as an **`int`** (time-steps: positions in the
sorted unique-time index, exactly as before, byte-identical folds) or as a
**calendar duration** applied to the actual time values:

| Spec | Meaning |
| --- | --- |
| `5` | five positions (rows of the unique-time index) |
| `"5d"`, `"2w"`, `"1mo"`, `"1q"`, `"1y"`, `"6h"` | Polars duration string via `dt.offset_by`: calendar days / months in the column's time zone |
| `datetime.timedelta(hours=36)`, `numpy.timedelta64` | exact elapsed time |
| `"5bd"`, `BusinessDays(5, holidays=[...], weekmask=...)` | business days via `numpy.busday_offset`; a non-business date rolls *forward* first |

A calendar **embargo** removes every training time `t` with
`block_end < t <= block_end + embargo` after each contiguous test block — on an
irregular axis (weekends, holidays, a data outage) that is a different set of
rows from "the next five", and it is the one the serial-correlation argument is
about. A calendar **horizon** gives each label the span `[t, t + horizon]` and
purges on interval overlap (the `t1` machinery). For the walk-forward and
calibration helpers the embargo is a gap: training ends more than `embargo`
of real time before the test block starts.

```python
from panelary.core.model_selection import BusinessDays, PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    horizon="5d",                                  # 5-calendar-day forward label
    embargo=BusinessDays(5, holidays=nyse_holidays),
)
```

Calendar specs need a `Date` / `Datetime` time column (a `TypeError` says so on
an integer axis); a `Date` axis refuses sub-day durations rather than letting
Polars floor them; negative durations are refused. The positional helpers need
the time values as `times=` when a spec is calendar-valued.

## See also

- [`validation`](validation.md) — purged / embargoed CV, CPCV, and backtest paths.
- [Validation guide](../user-guide/validation.md) — choosing between the two families.

## API

::: panelary.cross_validation
