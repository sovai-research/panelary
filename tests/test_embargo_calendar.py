"""Calendar-aware embargo and purge horizon for the purged splitters.

An embargo exists because of serial correlation, and serial correlation decays
in *time*, not in rows. On an irregular axis -- weekends, exchange holidays, a
data outage, a listing that starts late -- "five rows after the test block" and
"five business days after the test block" are different spans, and only the
second is what the argument is about. These tests pin:

* **exact windows** on a hand-built daily axis with two holidays and a
  two-day outage, for ``int``, ``"5d"``, ``timedelta``, ``"5bd"`` and
  ``BusinessDays(..., holidays=...)``;
* the calendar **purge horizon** (a label spans ``[tau, tau + horizon]``);
* **irregular axes** -- weekend rows, intraday bars across a DST change --
  against an independent oracle;
* **backward compatibility** -- every integer spec produces byte-identical
  folds to a frozen copy of the pre-change algorithm;
* the positional helpers in :mod:`panelary.validation._cv`.
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import polars as pl
import pytest

from panelary.core._calendar import BusinessDays, shift_forward, validate_duration
from panelary.core.model_selection import (
    CombinatorialPurgedCV,
    PurgedKFold,
    _contiguous_blocks,
)
from panelary.core.panel_frame import PanelFrame
from panelary.validation._cv import (
    cpcv_splits,
    purged_calibration_split,
    walk_forward_splits,
)

D = dt.date

# --------------------------------------------------------------------------- #
# A hand-built daily axis
# --------------------------------------------------------------------------- #
#: First half: two clean business weeks ending Friday 2024-12-20.
FIRST = [D(2024, 12, d) for d in (9, 10, 11, 12, 13, 16, 17, 18, 19, 20)]
#: Second half: 12-25 and 01-01 are holidays (no rows); 12-26 and 12-27 are a
#: data outage (business days with no rows).
SECOND = [
    D(2024, 12, 23),
    D(2024, 12, 24),
    D(2024, 12, 30),
    D(2024, 12, 31),
    D(2025, 1, 2),
    D(2025, 1, 3),
    D(2025, 1, 6),
    D(2025, 1, 7),
    D(2025, 1, 8),
    D(2025, 1, 9),
]
DATES = FIRST + SECOND
HOLIDAYS = ["2024-12-25", "2025-01-01"]


def _panel(dates: list, n_entities: int = 3) -> PanelFrame:
    rows = [
        (f"E{e}", d, float(i + e))
        for e in range(n_entities)
        for i, d in enumerate(dates)
    ]
    df = pl.DataFrame(rows, schema=["entity", "date", "x"], orient="row")
    return PanelFrame(df, entity="entity", time="date")


def _fold0_train_dates(**kwargs: object) -> list:
    """Training dates of fold 0 (test = FIRST) of a 2-fold PurgedKFold."""
    cv = PurgedKFold(n_splits=2, return_indices=True, **kwargs)  # type: ignore[arg-type]
    (train, test), _ = list(cv.split(_panel(DATES)))
    assert [DATES[i] for i in test] == FIRST
    return [DATES[i] for i in train]


def _fold1_train_dates(**kwargs: object) -> list:
    cv = PurgedKFold(n_splits=2, return_indices=True, **kwargs)  # type: ignore[arg-type]
    _, (train, test) = list(cv.split(_panel(DATES)))
    assert [DATES[i] for i in test] == SECOND
    return [DATES[i] for i in train]


# --------------------------------------------------------------------------- #
# Exact windows
# --------------------------------------------------------------------------- #
def test_integer_embargo_counts_rows() -> None:
    """The historical meaning: the next five *rows*, however far apart."""
    assert _fold0_train_dates(embargo=5) == SECOND[5:]  # drops 23,24,30,31,01-02


def test_business_day_embargo_with_holidays() -> None:
    """12-20 + 5bd skipping 12-25 = 23, 24, 26, 27, 30 -> embargo through 12-30.

    The outage (26, 27) has no rows, so only 23, 24 and 30 are removed -- three
    rows, not five. Five rows would have eaten into January.
    """
    expected = SECOND[3:]  # 12-31 onwards
    assert _fold0_train_dates(embargo=BusinessDays(5, holidays=HOLIDAYS)) == expected


def test_business_day_shorthand_ignores_holidays() -> None:
    """``"5bd"`` is BusinessDays(5): no holiday list, so 12-25 counts."""
    assert _fold0_train_dates(embargo="5bd") == SECOND[2:]  # through 12-27
    assert PurgedKFold(embargo="5bd").embargo == BusinessDays(5)


@pytest.mark.parametrize(
    ("embargo", "n_dropped"),
    [
        ("5d", 2),  # through 12-25: 23, 24
        (dt.timedelta(days=10), 3),  # through 12-30: 23, 24, 30
        (np.timedelta64(11, "D"), 4),  # through 12-31
        ("1w", 2),  # through 12-27
        ("2w", 6),  # through 01-03
        ("0d", 0),
        (BusinessDays(0), 0),  # 12-20 is a business day: no roll, no embargo
    ],
)
def test_calendar_embargo_windows(embargo: object, n_dropped: int) -> None:
    assert _fold0_train_dates(embargo=embargo) == SECOND[n_dropped:]


def test_month_embargo_is_a_calendar_month() -> None:
    """12-20 + 1mo = 01-20: every second-half row is embargoed."""
    cv = PurgedKFold(n_splits=2, embargo="1mo", return_indices=True)
    (train, _test), _ = list(cv.split(_panel(DATES)))
    assert train.size == 0


def test_embargo_after_the_last_block_does_nothing() -> None:
    assert _fold1_train_dates(embargo="5bd") == FIRST


def test_calendar_purge_horizon() -> None:
    """A label at tau spans [tau, tau + 3d].

    Fold 0: the test labels reach 12-23, so only the 12-23 training label
    overlaps (12-24's starts after 12-23). Fold 1: the first test label starts
    12-23, so a training label must end before it: 12-20 + 3d = 12-23 overlaps,
    12-19 + 3d = 12-22 does not.
    """
    assert _fold0_train_dates(horizon="3d") == SECOND[1:]
    assert _fold1_train_dates(horizon="3d") == FIRST[:-1]
    # The integer horizon counts rows instead: 3 on each side.
    assert _fold0_train_dates(horizon=3) == SECOND[3:]


def test_calendar_horizon_and_embargo_compose() -> None:
    got = _fold0_train_dates(horizon="3d", embargo=BusinessDays(5, holidays=HOLIDAYS))
    assert got == SECOND[3:]


def test_t1_supersedes_a_calendar_horizon_but_embargo_still_applies() -> None:
    t1 = np.array(DATES, dtype="datetime64[D]")  # point-in-time labels
    got = _fold0_train_dates(horizon="30d", t1=t1, embargo="5d")
    assert got == SECOND[2:]


# --------------------------------------------------------------------------- #
# Irregular axes against an independent oracle
# --------------------------------------------------------------------------- #
def _oracle_train(times: list, test_pos: np.ndarray, limit_of: object) -> np.ndarray:
    """Train positions = not test, and not in (block_end, limit_of(block_end)]."""
    blocked = set(test_pos.tolist())
    for _s, end in _contiguous_blocks(np.sort(test_pos)):
        limit = limit_of(times[end])  # type: ignore[operator]
        blocked.update(j for j in range(end + 1, len(times)) if times[j] <= limit)
    return np.array([j for j in range(len(times)) if j not in blocked], dtype=np.int64)


def test_weekend_rows_inside_a_business_day_embargo() -> None:
    """A 7-day-a-week axis (crypto-like): Saturday/Sunday rows that fall inside
    the business-day window are embargoed too."""
    times = [D(2025, 3, 3) + dt.timedelta(days=i) for i in range(42)]
    spec = BusinessDays(3)

    def limit(d: dt.date) -> dt.date:
        rolled = np.busday_offset(np.datetime64(d, "D"), 3, roll="forward")
        return rolled.astype(object)

    cv = CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, embargo=spec, return_indices=True
    )
    for train, test in cv.split(_panel(times, n_entities=1)):
        np.testing.assert_array_equal(train, _oracle_train(times, test, limit))


def test_intraday_embargo_across_dst() -> None:
    """Hourly bars through the 2024-03-10 US DST change.

    ``"1d"`` is a calendar day in the column's zone (wall clock: 24 rows on an
    ordinary day, 23 across spring-forward); ``timedelta(hours=24)`` is exact
    elapsed time. Both are checked against Python's own zone arithmetic.
    """
    start = dt.datetime(2024, 3, 8, 0, 0)
    naive = [start + dt.timedelta(hours=h) for h in range(24 * 5)]
    times = (
        pl.Series("t", naive)
        .dt.replace_time_zone(
            "America/New_York", ambiguous="earliest", non_existent="null"
        )
        .drop_nulls()
        .to_list()
    )
    panel = _panel(times, n_entities=1)
    zone = times[0].tzinfo

    def wall_day(t: dt.datetime) -> dt.datetime:
        return (t.replace(tzinfo=None) + dt.timedelta(days=1)).replace(tzinfo=zone)

    def elapsed(t: dt.datetime) -> dt.datetime:
        return (t.astimezone(dt.timezone.utc) + dt.timedelta(hours=24)).astimezone(zone)

    for spec, limit in (("1d", wall_day), (dt.timedelta(hours=24), elapsed)):
        cv = PurgedKFold(n_splits=5, embargo=spec, return_indices=True)
        for train, test in cv.split(panel):
            np.testing.assert_array_equal(train, _oracle_train(times, test, limit))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_irregular_daily_axis_matches_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    all_days = [D(2023, 1, 2) + dt.timedelta(days=i) for i in range(200)]
    keep = rng.random(200) < 0.55  # gaps everywhere, weekends included
    times = [d for d, k in zip(all_days, keep, strict=True) if k]
    holidays = [str(all_days[i]) for i in rng.choice(200, size=8, replace=False)]
    spec = BusinessDays(4, holidays=holidays)

    def limit(d: dt.date) -> dt.date:
        return spec.offset_dates(np.array([d], dtype="datetime64[D]"))[0].astype(object)

    for n_groups, k in ((5, 2), (6, 1)):
        cv = CombinatorialPurgedCV(
            n_groups=n_groups, n_test_groups=k, embargo=spec, return_indices=True
        )
        for train, test in cv.split(_panel(times, n_entities=2)):
            np.testing.assert_array_equal(train, _oracle_train(times, test, limit))


# --------------------------------------------------------------------------- #
# Backward compatibility: integer specs are byte-identical to the old code
# --------------------------------------------------------------------------- #
def _frozen_purge_embargo(
    n_times: int, test_positions: np.ndarray, horizon: int, embargo: int
) -> np.ndarray:
    """Verbatim copy of ``_purge_embargo_positions`` before calendar support."""
    test_set = {int(p) for p in test_positions}
    blocked = set(test_set)
    for i in test_set:
        lo = max(0, i - horizon)
        hi = min(n_times - 1, i + horizon)
        for j in range(lo, hi + 1):
            blocked.add(j)
    if embargo > 0:
        for _start, end in _contiguous_blocks(np.array(sorted(test_set))):
            lo = end + 1
            hi = min(n_times - 1, end + embargo)
            for j in range(lo, hi + 1):
                blocked.add(j)
    return np.array([p for p in range(n_times) if p not in blocked], dtype=np.int64)


def _frozen_walk_forward(
    n_times, n_splits, horizon, embargo, expanding, window_size=None
):  # type: ignore[no-untyped-def]
    test_size = n_times // (n_splits + 1)
    if window_size is None:
        window_size = test_size * n_splits
    first_test = n_times - n_splits * test_size
    out = []
    for s in range(n_splits):
        start = first_test + s * test_size
        test = np.arange(start, start + test_size, dtype=np.int64)
        allowed = _frozen_purge_embargo(n_times, test, horizon, embargo)
        lo = 0 if expanding else max(0, start - window_size)
        train = allowed[(allowed < start) & (allowed >= lo)]
        if embargo > 0 and train.size:
            train = train[train < start - embargo]
        if train.size:
            out.append((train, test))
    return out


GRID = list(itertools.product([0, 1, 3], [0, 1, 2, 5]))


@pytest.mark.parametrize(("horizon", "embargo"), GRID)
@pytest.mark.parametrize("axis", ["int", "date"])
def test_integer_specs_are_byte_identical_purged_kfold(
    horizon: int, embargo: int, axis: str
) -> None:
    times = (
        list(range(37))
        if axis == "int"
        else DATES + [D(2025, 2, d) for d in range(1, 18)]
    )
    cv = PurgedKFold(n_splits=4, horizon=horizon, embargo=embargo, return_indices=True)
    folds = np.array_split(np.arange(len(times), dtype=np.int64), 4)
    for (train, test), expected_test in zip(
        cv.split(_panel(times)), folds, strict=True
    ):
        np.testing.assert_array_equal(test, expected_test)
        ref = _frozen_purge_embargo(len(times), expected_test, horizon, embargo)
        assert train.dtype == ref.dtype and train.tobytes() == ref.tobytes()
    assert cv.horizon == horizon and cv.embargo == embargo
    assert type(cv.embargo) is int


@pytest.mark.parametrize(("horizon", "embargo"), GRID)
def test_integer_specs_are_byte_identical_cpcv(horizon: int, embargo: int) -> None:
    n = 30
    cv = CombinatorialPurgedCV(
        n_groups=5, n_test_groups=2, horizon=horizon, embargo=embargo
    )
    for split in cpcv_splits(
        n, n_groups=5, n_test_groups=2, horizon=horizon, embargo=embargo
    ):
        ref = _frozen_purge_embargo(n, split.test, horizon, embargo)
        assert split.train.tobytes() == ref.tobytes()
    # and through the panel path
    cv.return_indices = True
    for train, test in cv.split(_panel(list(range(n)))):
        assert (
            train.tobytes()
            == _frozen_purge_embargo(n, test, horizon, embargo).tobytes()
        )


@pytest.mark.parametrize(("horizon", "embargo"), GRID)
@pytest.mark.parametrize("expanding", [True, False])
def test_integer_specs_are_byte_identical_walk_forward(
    horizon: int, embargo: int, expanding: bool
) -> None:
    got = walk_forward_splits(
        40, n_splits=4, horizon=horizon, embargo=embargo, expanding=expanding
    )
    ref = _frozen_walk_forward(40, 4, horizon, embargo, expanding)
    assert len(got) == len(ref)
    for split, (train, test) in zip(got, ref, strict=True):
        assert split.train.tobytes() == train.tobytes()
        assert split.test.tobytes() == test.tobytes()


def test_integer_calibration_split_unchanged() -> None:
    # purge (horizon 2) drops 13, 14; the gap (embargo 3) keeps train < 15 - 3.
    train, calib = purged_calibration_split(
        20, calibration_size=5, horizon=2, embargo=3
    )
    assert (int(train.max()), int(calib.min())) == (11, 15)
    train, calib = purged_calibration_split(20, calibration_size=5, horizon=2)
    assert (int(train.max()), int(calib.min())) == (12, 15)  # the doctest's value


def test_integral_float_is_still_accepted() -> None:
    """``embargo=0.0`` used to work; it still does, as the int 0."""
    assert PurgedKFold(embargo=0.0).embargo == 0


# --------------------------------------------------------------------------- #
# Positional helpers with `times=`
# --------------------------------------------------------------------------- #
def test_walk_forward_calendar_gap() -> None:
    """Training must end more than 5 business days before each test block."""
    n = len(DATES)
    splits = walk_forward_splits(n, n_splits=2, embargo="5bd", times=DATES)
    assert splits
    for sp in splits:
        start = DATES[int(sp.test[0])]
        for j in sp.train.tolist():
            gap_end = np.busday_offset(np.datetime64(DATES[j], "D"), 5, roll="forward")
            assert gap_end < np.datetime64(start, "D")
        # and nothing that satisfies the gap was dropped
        eligible = [
            j
            for j in range(int(sp.test[0]))
            if np.busday_offset(np.datetime64(DATES[j], "D"), 5, roll="forward")
            < np.datetime64(start, "D")
        ]
        assert sp.train.tolist() == eligible


def test_cpcv_splits_calendar_matches_panel_splitter() -> None:
    cv = CombinatorialPurgedCV(
        n_groups=4, n_test_groups=2, embargo="3bd", return_indices=True
    )
    via_panel = list(cv.split(_panel(DATES)))
    via_positions = cpcv_splits(
        len(DATES), n_groups=4, n_test_groups=2, embargo="3bd", times=DATES
    )
    assert len(via_panel) == len(via_positions)
    for (train, test), split in zip(via_panel, via_positions, strict=True):
        np.testing.assert_array_equal(train, split.train)
        np.testing.assert_array_equal(test, split.test)


def test_calibration_split_calendar() -> None:
    train, calib = purged_calibration_split(
        len(DATES),
        calibration_size=10,
        embargo=BusinessDays(5, holidays=HOLIDAYS),
        times=DATES,
    )
    assert calib.tolist() == list(range(10, 20))
    # 12-20 + 5bd (skipping 12-25) = 12-30 >= 12-23: dropped. 12-13 + 5bd = 12-20 < 12-23: kept.
    assert [DATES[i] for i in train] == FIRST[:5]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        -1,
        "-1d",
        "5x",
        "3i",
        "1bd2d",
        np.timedelta64(1, "M"),
        1.5,
        dt.timedelta(days=-1),
    ],
)
def test_bad_specs_rejected_at_construction(bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        PurgedKFold(embargo=bad)  # type: ignore[arg-type]
    with pytest.raises((ValueError, TypeError)):
        CombinatorialPurgedCV(horizon=bad)  # type: ignore[arg-type]


def test_negative_int_keeps_its_old_message() -> None:
    with pytest.raises(ValueError, match=r"`embargo` must be >= 0, got -1\."):
        PurgedKFold(embargo=-1)
    with pytest.raises(ValueError, match=r"`horizon` must be >= 0, got -2\."):
        CombinatorialPurgedCV(horizon=-2)


def test_calendar_spec_on_integer_axis_is_a_type_error() -> None:
    cv = PurgedKFold(n_splits=2, embargo="5d")
    with pytest.raises(TypeError, match="calendar duration"):
        list(cv.split(_panel(list(range(10)))))


def test_positional_calendar_needs_times() -> None:
    with pytest.raises(TypeError, match="times="):
        walk_forward_splits(20, n_splits=2, embargo="5d")
    with pytest.raises(TypeError, match="times="):
        cpcv_splits(20, n_groups=4, embargo="1bd")
    with pytest.raises(ValueError, match="strictly increasing"):
        walk_forward_splits(
            3,
            n_splits=1,
            embargo="1d",
            times=[D(2024, 1, 2), D(2024, 1, 1), D(2024, 1, 3)],
        )
    with pytest.raises(ValueError, match="values but"):
        cpcv_splits(5, n_groups=2, n_test_groups=1, embargo="1d", times=DATES)
    with pytest.raises(TypeError, match="Date or Datetime"):
        cpcv_splits(4, n_groups=2, n_test_groups=1, embargo="1d", times=[1, 2, 3, 4])


def test_date_axis_refuses_sub_day_durations() -> None:
    cv = PurgedKFold(n_splits=2, embargo="36h")
    with pytest.raises(ValueError, match="sub-day"):
        list(cv.split(_panel(DATES)))


def test_business_days_validation() -> None:
    with pytest.raises(ValueError):
        BusinessDays(-1)
    with pytest.raises(TypeError):
        BusinessDays(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BusinessDays(1, weekmask="nonsense")
    assert validate_duration("7bd", name="x") == BusinessDays(7)
    s = shift_forward(pl.Series([D(2024, 12, 21)]), BusinessDays(0))  # Saturday
    assert s.to_list() == [D(2024, 12, 23)]  # rolled forward to Monday
