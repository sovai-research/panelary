"""Panel invariants and the PanelValidator / validate_panel surface.

Covers every structural check (keys, null keys, unique keys, per-entity time
order, gaps, min history, coverage), impact levels (fail / warn / off), the
dataframely-style valid / invalid split, and the warning categories each
warn-level finding is raised as.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame, PanelOrderWarning
from panelary.quality import (
    ColumnContract,
    PanelValidationError,
    PanelValidator,
    QualityWarning,
    validate_panel,
)
from panelary.quality._common import FAILED_CHECKS_COL


def _panel(n_entities: int = 3, n_times: int = 8, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [chr(ord("a") + i) for i in range(n_entities)]
    return pl.DataFrame(
        {
            "id": np.repeat(ids, n_times),
            "t": np.tile(np.arange(n_times, dtype=np.int64), n_entities),
            "x": rng.normal(size=n_entities * n_times),
        }
    )


def _quiet(**kw):
    return validate_panel(raise_on_fail=False, emit_warnings=False, **kw)


# --------------------------------------------------------------------------- #
# A clean panel
# --------------------------------------------------------------------------- #
def test_clean_panel_passes_every_default_check():
    df = _panel()
    report = validate_panel(df, entity="id", time="t")
    assert report.status == "pass" and report.ok
    assert [c.name for c in report.checks] == [
        "key_columns",
        "null_keys",
        "unique_keys",
        "monotone_time",
        "gaps",
    ]
    assert report.valid.equals(df)
    assert report.invalid.height == 0
    assert FAILED_CHECKS_COL in report.invalid.columns
    assert (report.n_rows, report.n_entities, report.n_times) == (24, 3, 8)
    assert report.check("monotone_time").observed["sortedness"] == "panel"


def test_keys_default_to_first_two_columns():
    report = validate_panel(_panel())
    assert (report.entity, report.time) == ("id", "t")


def test_lazyframe_and_panelframe_inputs():
    df = _panel()
    assert validate_panel(df.lazy(), entity="id", time="t").ok
    pf = PanelFrame(df, entity="id", time="t")
    report = validate_panel(pf)
    assert (report.entity, report.time) == ("id", "t")
    with pytest.raises(ValueError, match="conflicts"):
        validate_panel(pf, entity="x")


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def test_missing_key_column_fails_and_stops_panel_checks():
    df = _panel().drop("t")
    report = _quiet(data=df, entity="id", time="t")
    assert [c.name for c in report.checks] == ["key_columns"]
    assert report.status == "fail"
    assert report.n_entities is None and report.n_times is None
    with pytest.raises(PanelValidationError) as info:
        validate_panel(df, entity="id", time="t", emit_warnings=False)
    assert info.value.report.check("key_columns").passed is False


def test_unorderable_time_dtype_fails_key_columns():
    df = _panel().with_columns(pl.col("t").cast(pl.String))
    report = _quiet(data=df, entity="id", time="t")
    assert not report.check("key_columns").passed
    assert "orderable" in report.check("key_columns").message


def test_null_keys_are_flagged_rowwise():
    df = _panel().with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(None)
        .otherwise(pl.col("id"))
        .alias("id")
    )
    report = _quiet(data=df, entity="id", time="t")
    check = report.check("null_keys")
    assert check.status == "fail" and check.n_failing == 1
    assert check.observed == {"null_entity": 1, "null_time": 0}
    assert report.invalid.height == 1
    assert report.invalid[FAILED_CHECKS_COL].to_list() == [["null_keys"]]


def test_duplicate_keys_flag_every_copy():
    df = pl.concat(
        [_panel(), _panel().filter((pl.col("id") == "b") & (pl.col("t") < 2))]
    )
    report = _quiet(data=df, entity="id", time="t")
    check = report.check("unique_keys")
    assert check.status == "fail"
    assert check.observed == {"duplicated_keys": 2, "rows": 4}
    assert [r["keys"] for r in check.offending] == [
        {"id": "b", "t": 0},
        {"id": "b", "t": 1},
    ]
    assert all(r["count"] == 2 for r in check.offending)
    assert report.invalid.height == 4
    assert report.valid.height == df.height - 4


# --------------------------------------------------------------------------- #
# Row order: reuses PanelFrame's measurement, raises PanelOrderWarning
# --------------------------------------------------------------------------- #
def test_shuffled_panel_warns_with_panel_order_warning():
    df = _panel().sample(fraction=1.0, shuffle=True, seed=3)
    with pytest.warns(PanelOrderWarning, match="monotone_time"):
        report = validate_panel(df, entity="id", time="t")
    check = report.check("monotone_time")
    assert check.status == "warn" and report.ok  # warn does not fail the report
    assert check.observed["sortedness"] == "unsorted"
    prev = pl.col("t").shift(1).over("id")
    expected = int(df.select((pl.col("t") < prev).fill_null(False).sum()).item())
    assert check.n_failing == expected > 0
    assert report.invalid.height == 0  # warn-level: nothing is rejected


def test_monotone_time_as_fail_rejects_the_backward_rows():
    df = pl.DataFrame({"id": ["a", "a", "a", "b"], "t": [1, 3, 2, 1], "x": [0.0] * 4})
    report = _quiet(data=df, entity="id", time="t", impact={"monotone_time": "fail"})
    assert report.status == "fail"
    assert report.invalid.select("id", "t").rows() == [("a", 2)]
    assert report.check("monotone_time").offending[0]["previous_time"] == 3


def test_interleaved_but_time_sorted_is_reported_as_time():
    df = _panel().sort("t", "id")
    report = validate_panel(df, entity="id", time="t")
    assert report.check("monotone_time").observed["sortedness"] == "time"


def test_validator_measures_rather_than_trusting_a_promise():
    shuffled = _panel().sample(fraction=1.0, shuffle=True, seed=1)
    promised = PanelFrame(shuffled, entity="id", time="t").mark_sorted()
    report = _quiet(data=promised)
    assert report.check("monotone_time").status == "warn"


# --------------------------------------------------------------------------- #
# Gaps
# --------------------------------------------------------------------------- #
def test_gap_against_panel_calendar():
    df = _panel().filter(~((pl.col("id") == "b") & pl.col("t").is_in([3, 4])))
    with pytest.warns(QualityWarning, match="gaps"):
        report = validate_panel(df, entity="id", time="t")
    check = report.check("gaps")
    assert check.status == "warn" and check.n_failing == 1
    assert check.observed["missing_steps"] == 2
    assert check.offending == (
        {
            "keys": {"id": "b"},
            "n_gaps": 1,
            "n_missing": 2,
            "first_gap_after": 2,
            "first_gap_before": 5,
        },
    )


def test_late_listing_and_delisting_are_not_gaps():
    df = _panel().filter(
        ~((pl.col("id") == "a") & (pl.col("t") < 3))
        & ~((pl.col("id") == "c") & (pl.col("t") > 5))
    )
    assert validate_panel(df, entity="id", time="t").check("gaps").passed


def _business_day_panel() -> pl.DataFrame:
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(21)]
    bdays = [d for d in days if d.weekday() < 5]
    return pl.DataFrame(
        {
            "id": ["a"] * len(bdays) + ["b"] * len(bdays),
            "date": bdays * 2,
            "px": np.arange(2 * len(bdays), dtype=np.float64),
        }
    )


def test_panel_calendar_needs_no_business_day_configuration():
    df = _business_day_panel()
    assert validate_panel(df, entity="id", time="date").check("gaps").passed
    # a daily grid, by contrast, sees every weekend as a two-day gap
    report = _quiet(data=df, entity="id", time="date", frequency="1d")
    gaps = report.check("gaps")
    assert gaps.status == "warn"
    # two internal weekends x two entities (the trailing weekend is no gap)
    assert gaps.observed["gap_intervals"] == 4
    assert gaps.observed["missing_steps"] == 8


def test_monthly_frequency_respects_month_ends():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a"],
            "date": [dt.date(2024, 1, 31), dt.date(2024, 3, 31), dt.date(2024, 4, 30)],
            "x": [1.0, 2.0, 3.0],
        }
    )
    gaps = _quiet(data=df, entity="id", time="date", frequency="1mo").check("gaps")
    assert gaps.observed["missing_steps"] == 1  # 2024-02-29
    assert gaps.offending[0]["first_gap_after"] == "2024-01-31"


def test_numeric_frequency_and_timedelta_frequency():
    df = pl.DataFrame({"id": ["a"] * 3, "t": [0, 2, 8], "x": [0.0] * 3})
    gaps = _quiet(data=df, entity="id", time="t", frequency=2).check("gaps")
    assert gaps.observed["missing_steps"] == 2  # 4 and 6
    dts = [dt.datetime(2024, 1, 1, h) for h in (0, 1, 4)]
    df2 = pl.DataFrame({"id": ["a"] * 3, "ts": dts, "x": [0.0] * 3})
    gaps2 = _quiet(data=df2, entity="id", time="ts", frequency=dt.timedelta(hours=1))
    assert gaps2.check("gaps").observed["missing_steps"] == 2


@pytest.mark.parametrize(
    ("time_values", "frequency"),
    [([1, 2, 3], "1d"), ([1, 2, 3], 0), ([1, 2, 3], True)],
)
def test_frequency_must_match_the_time_axis(time_values, frequency):
    df = pl.DataFrame({"id": ["a"] * 3, "t": time_values, "x": [0.0] * 3})
    with pytest.raises(ValueError, match="frequency"):
        validate_panel(df, entity="id", time="t", frequency=frequency)


# --------------------------------------------------------------------------- #
# History and coverage
# --------------------------------------------------------------------------- #
def test_min_obs_rejects_short_entities_and_filter_drops_them():
    df = pl.concat(
        [_panel(), pl.DataFrame({"id": ["z", "z"], "t": [0, 1], "x": [0.0, 0.0]})]
    )
    validator = PanelValidator(entity="id", time="t", min_obs=5)
    report = validator.validate(df, raise_on_fail=False, emit_warnings=False)
    check = report.check("min_obs")
    assert check.status == "fail" and check.n_failing == 1 and check.unit == "entities"
    assert check.offending == ({"keys": {"id": "z"}, "n_obs": 2},)
    valid, report2 = validator.filter(df, emit_warnings=False)
    assert "z" not in valid["id"].to_list()
    assert report2.invalid["id"].unique().to_list() == ["z"]


def test_coverage_uses_the_whole_calendar():
    df = _panel(n_times=10).filter(~((pl.col("id") == "c") & (pl.col("t") < 6)))
    with pytest.warns(QualityWarning, match="coverage"):
        report = validate_panel(df, entity="id", time="t", min_coverage=0.8)
    check = report.check("coverage")
    assert check.n_failing == 1
    assert check.offending == ({"keys": {"id": "c"}, "n_obs": 4, "coverage": 0.4},)
    assert check.observed["calendar_size"] == 10


@pytest.mark.parametrize(
    "bad", [{"min_obs": 0}, {"min_obs": 2.5}, {"min_coverage": 0.0}]
)
def test_bad_thresholds_raise(bad):
    with pytest.raises(ValueError):
        PanelValidator(entity="id", time="t", **bad)


# --------------------------------------------------------------------------- #
# Impact levels, filter, warnings
# --------------------------------------------------------------------------- #
def test_impact_overrides_and_off():
    df = _panel().filter(~((pl.col("id") == "b") & (pl.col("t") == 3)))
    report = _quiet(data=df, entity="id", time="t", impact={"gaps": "fail"})
    assert report.status == "fail"
    assert set(report.invalid["id"].unique().to_list()) == {"b"}
    report_off = _quiet(data=df, entity="id", time="t", impact={"gaps": "off"})
    assert "gaps" not in [c.name for c in report_off.checks]
    with pytest.raises(ValueError, match="unknown check"):
        PanelValidator(impact={"not_a_check": "warn"})
    with pytest.raises(ValueError, match="must be one of"):
        PanelValidator(impact={"gaps": "loud"})


def test_failed_checks_column_lists_every_failure_sorted():
    df = pl.concat(
        [_panel(), _panel().filter((pl.col("id") == "a") & (pl.col("t") == 0))]
    )
    df = df.with_columns(
        pl.when(pl.col("id") == "a").then(None).otherwise(pl.col("x")).alias("x")
    )
    report = _quiet(
        data=df,
        entity="id",
        time="t",
        schema={"x": ColumnContract(nullable=False)},
    )
    both = report.invalid.filter(pl.col("t") == 0)
    assert both[FAILED_CHECKS_COL].to_list() == [["nullability:x", "unique_keys"]] * 2


def test_filter_raises_only_for_frame_level_failures():
    df = _panel()
    with pytest.raises(PanelValidationError, match="frame-level"):
        PanelValidator(entity="id", time="t", schema={"missing": pl.Int64}).filter(df)
    valid, report = PanelValidator(entity="id", time="t").filter(
        pl.concat([df, df.head(1)]), emit_warnings=False
    )
    assert valid.height == df.height - 1 and report.status == "fail"


def test_emit_warnings_false_is_silent():
    df = _panel().sample(fraction=1.0, shuffle=True, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = validate_panel(df, entity="id", time="t", emit_warnings=False)
    assert report.status == "warn"


def test_raise_for_status_returns_self_when_ok_and_report_rides_the_error():
    df = _panel()
    report = validate_panel(df, entity="id", time="t")
    assert report.raise_for_status() is report
    with pytest.raises(PanelValidationError) as info:
        validate_panel(
            pl.concat([df, df.head(2)]), entity="id", time="t", emit_warnings=False
        )
    assert info.value.report.failures[0].name == "unique_keys"
    assert "unique_keys" in str(info.value)


def test_empty_panel_passes():
    df = _panel().clear()
    report = validate_panel(df, entity="id", time="t", min_obs=3, min_coverage=0.5)
    assert report.ok and report.n_rows == 0 and report.n_entities == 0


def test_repr_and_to_frame():
    validator = PanelValidator(entity="id", time="t", min_obs=3)
    assert "min_obs=3" in repr(validator)
    report = validator.validate(_panel())
    frame = report.to_frame()
    assert frame.height == len(report.checks)
    assert frame["status"].to_list() == ["pass"] * len(report.checks)
    assert "ValidationReport[pass]" in repr(report)
    with pytest.raises(KeyError, match="no check"):
        report.check("coverage")
