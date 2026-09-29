"""The canonical span table and the vectorised purge built on it.

The purge rewrite must be **byte-identical** to the historical O(n x m) loop
(frozen verbatim in ``tests/_spans_reference.py``) -- including its handling of
null end times, ties and inverted spans -- and the span table must agree with
brute-force loops.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest

from panelary.core._spans import (
    _overlaps_any,
    _spans_from_t1,
    _time_positions,
)
from panelary.core.model_selection import (
    CombinatorialPurgedCV,
    PurgedKFold,
    _purge_embargo_positions_t1,
    _resolve_t1,
)
from panelary.core.panel_frame import as_panel
from panelary.label import fixed_horizon, triple_barrier
from tests import _spans_reference as ref


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Random inputs for the golden purge comparison
# --------------------------------------------------------------------------- #
def _random_axis(rng: np.random.Generator, n: int, kind: int):
    base = np.sort(rng.choice(np.arange(3 * n + 5), size=n, replace=False))
    if kind == 0:  # int axis, float t1 with NaN, some inverted spans
        times = base.astype(np.int64)
        t1 = (times + rng.integers(-3, 12, n)).astype(np.float64)
        t1[rng.random(n) < 0.15] = np.nan
    elif kind == 1:  # Date axis, off-grid Datetime t1 with NaT
        times = (np.datetime64("2020-01-01") + base).astype("datetime64[D]")
        t1 = (times + rng.integers(-2, 15, n)).astype("datetime64[us]")
        t1 = t1 + rng.integers(0, 3, n).astype("timedelta64[h]")
        t1[rng.random(n) < 0.15] = np.datetime64("NaT", "us")
    elif kind == 2:  # float axis, continuous t1
        times = base.astype(np.float64)
        t1 = times + rng.random(n) * 10 - 1
    else:  # on-grid t1 with many ties (equal endpoints)
        times = base.astype(np.int64)
        t1 = times[np.minimum(np.arange(n) + rng.integers(0, 6, n), n - 1)]
    return times, t1


def _random_test(rng: np.random.Generator, n: int) -> np.ndarray:
    mode = int(rng.integers(0, 3))
    m = int(rng.integers(0, n + 1))
    if mode == 0 and m:
        s0 = int(rng.integers(0, n - m + 1))
        return np.arange(s0, s0 + m)
    if mode == 1:
        return rng.choice(n, size=m, replace=False)
    k = max(1, n // 5)
    groups = rng.choice(5, size=2, replace=False)
    return np.concatenate([np.arange(g * k, min(n, (g + 1) * k)) for g in groups])


def test_purge_byte_identical_to_frozen_loop() -> None:
    rng = np.random.default_rng(20260929)
    for trial in range(1500):
        n = int(rng.integers(1, 70))
        times, t1 = _random_axis(rng, n, trial % 4)
        test = _random_test(rng, n)
        embargo = int(rng.integers(0, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _purge_embargo_positions_t1(n, test, times, t1, embargo)
        want = ref.frozen_purge_embargo_positions_t1(n, test, times, t1, embargo)
        assert _same(got, want), (trial, n, test, times, t1, embargo)


def test_purge_matches_pairwise_brute_force_on_positions() -> None:
    rng = np.random.default_rng(7)
    for _ in range(400):
        n = int(rng.integers(1, 50))
        times, t1 = _random_axis(rng, n, int(rng.integers(0, 4)))
        test = _random_test(rng, n)
        embargo = int(rng.integers(0, 4))
        end_pos = _time_positions(times, t1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _purge_embargo_positions_t1(n, test, times, t1, embargo)
        assert _same(got, ref.pairwise_purge(n, test, end_pos, embargo))


@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_nat_inside_test_block_matches_frozen_loop(where: str) -> None:
    """T19: a NaT anywhere in a test block must not poison the running max."""
    n = 40
    times = (np.datetime64("2021-01-01") + np.arange(n)).astype("datetime64[D]")
    t1 = times + np.int64(3)
    test = np.arange(15, 25)
    idx = {"start": 15, "middle": 20, "end": 24}[where]
    t1 = t1.astype("datetime64[D]")
    t1[idx] = np.datetime64("NaT", "D")
    for embargo in (0, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _purge_embargo_positions_t1(n, test, times, t1, embargo)
        want = ref.frozen_purge_embargo_positions_t1(n, test, times, t1, embargo)
        assert _same(got, want)


def test_overlaps_any_matches_pairwise() -> None:
    rng = np.random.default_rng(3)
    for _ in range(300):
        nq, nr = int(rng.integers(0, 30)), int(rng.integers(0, 30))
        qs = rng.integers(0, 40, nq)
        qe = qs + rng.integers(-3, 8, nq)
        rs = np.sort(rng.integers(0, 40, nr))
        re_ = rs + rng.integers(-3, 8, nr)
        qe[rng.random(nq) < 0.1] = -1
        re_[rng.random(nr) < 0.1] = -1
        got = _overlaps_any(qs, qe, rs, re_)
        want = np.array(
            [
                any(r0 <= b and a <= r1 for r0, r1 in zip(rs, re_))
                for a, b in zip(qs, qe)
            ],
            dtype=bool,
        )
        assert np.array_equal(got, want)


def test_time_positions_object_arrays_and_nulls() -> None:
    times = np.array(["2020-01-01", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")
    vals = np.array([dt.date(2020, 1, 2), None, dt.date(2020, 1, 6)], dtype=object)
    assert _time_positions(times, vals).tolist() == [0, -1, 2]
    ints = np.array([0, 2, 4], dtype=np.int64)
    assert _time_positions(ints, np.array([3, None, -1], dtype=object)).tolist() == [
        1,
        -1,
        -1,
    ]


# --------------------------------------------------------------------------- #
# Splitters: the folds are unchanged end to end
# --------------------------------------------------------------------------- #
def _labelled_panel(seed: int = 0, n_ent: int = 4, n_t: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_ent):
        keep = np.sort(
            rng.choice(n_t, size=n_t - int(rng.integers(0, 8)), replace=False)
        )
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, keep.size)))
        frames.append(
            pl.DataFrame(
                {
                    "id": [f"e{e}"] * keep.size,
                    "date": [
                        dt.date(2020, 1, 1) + dt.timedelta(days=int(d)) for d in keep
                    ],
                    "close": price,
                }
            )
        )
    return triple_barrier(
        pl.concat(frames), entity="id", time="date", max_holding=5, vol_lookback=5
    )


@pytest.mark.parametrize("seed", range(6))
def test_splitter_folds_unchanged(seed: int) -> None:
    df = _labelled_panel(seed)
    pf = as_panel(df, "id", "date")
    times = pf.time_index().to_numpy()
    t1_arr = _resolve_t1("t1", pf, times)
    assert t1_arr is not None
    for embargo in (0, 3):
        cv = PurgedKFold(n_splits=5, t1="t1", embargo=embargo, return_indices=True)
        for train, test in cv.split(df):
            want = ref.frozen_purge_embargo_positions_t1(
                times.shape[0], test, times, t1_arr, embargo
            )
            assert _same(train, want)
        cpcv = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, t1="t1", embargo=embargo, return_indices=True
        )
        for train, test in cpcv.split(df):
            want = ref.frozen_purge_embargo_positions_t1(
                times.shape[0], test, times, t1_arr, embargo
            )
            assert _same(train, want)


# --------------------------------------------------------------------------- #
# The null-t1 guard (B5 / F4): folds unchanged, a warning when at risk
# --------------------------------------------------------------------------- #
def test_null_test_t1_warns_but_keeps_folds() -> None:
    n = 30
    times = np.arange(n, dtype=np.int64)
    t1 = (times + 2).astype(np.float64)
    t1[10] = np.nan  # a test time whose span is unknown
    test = np.arange(10, 15)
    with pytest.warns(UserWarning, match="null label end time"):
        got = _purge_embargo_positions_t1(n, test, times, t1, 0)
    want = ref.frozen_purge_embargo_positions_t1(n, test, times, t1, 0)
    assert _same(got, want)
    # Position 8 ends at 10 -- certainly overlaps the null test label at 10 --
    # and is kept, exactly as before (the owner decision is recorded in F4).
    assert 8 in got.tolist()


def test_null_t1_in_tail_fold_does_not_warn() -> None:
    """fixed_horizon's null tail in the final fold puts no training label at risk."""
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "id": np.repeat(["a", "b"], 50),
            "t": np.tile(np.arange(50), 2),
            "close": 100 + rng.normal(0, 1, 100).cumsum(),
        }
    )
    lab = fixed_horizon(df, entity="id", time="t", horizon=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folds = list(PurgedKFold(n_splits=5, t1="t1", return_indices=True).split(lab))
    assert len(folds) == 5


# --------------------------------------------------------------------------- #
# The span table itself
# --------------------------------------------------------------------------- #
def _ragged(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(int(rng.integers(1, 5))):
        ts = np.sort(rng.choice(40, size=int(rng.integers(1, 25)), replace=False))
        for k, t in enumerate(ts):
            if rng.random() < 0.25:
                end = None  # unlabelled / unresolved
            else:
                end = float(t + rng.integers(0, 9) + (0.5 if rng.random() < 0.3 else 0))
            rows.append((f"e{e}", int(t), end, bool(rng.random() < 0.1), k))
    df = pl.DataFrame(
        rows,
        schema={
            "id": pl.String,
            "t": pl.Int64,
            "t1": pl.Float64,
            "censored": pl.Boolean,
            "k": pl.Int64,
        },
        orient="row",
    )
    return df.sample(fraction=1.0, shuffle=True, seed=seed)  # unsorted input


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("censored", [None, "censored"])
def test_span_table_matches_brute_force(seed: int, censored: str | None) -> None:
    df = _ragged(seed)
    frame, table = _spans_from_t1(df, t1="t1", entity="id", time="t", censored=censored)
    want = ref.ref_spans(df, entity="id", time="t", t1="t1", censored=censored)
    got = list(zip(table.start.tolist(), table.end.tolist(), strict=True))
    assert got == want
    assert table.n_dropped == frame.height - len(want)
    assert frame.equals(df.sort(["id", "t"]))
    assert np.all(table.end >= table.start)
    assert np.all(table.seg_start <= table.start) and np.all(table.end <= table.seg_end)


def test_span_table_raises_on_inverted_and_duplicates() -> None:
    df = pl.DataFrame({"id": ["a", "a"], "t": [1, 2], "t1": [0, 3]})
    with pytest.raises(ValueError, match="ends before it starts"):
        _spans_from_t1(df, t1="t1")
    dup = pl.DataFrame({"id": ["a", "a"], "t": [1, 1], "t1": [1, 1]})
    with pytest.raises(ValueError, match="not unique"):
        _spans_from_t1(dup, t1="t1")


def test_t1_beyond_entity_data_is_dropped_not_clamped() -> None:
    df = pl.DataFrame(
        {"id": ["a"] * 4 + ["b"] * 6, "t": [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]}
    ).with_columns((pl.col("t") + 2).alias("t1"))
    _, table = _spans_from_t1(df, t1="t1")
    # a: rows 0,1 resolve (t1 <= 3); b: rows 4..7 resolve (t1 <= 5).
    assert table.start.tolist() == [0, 1, 4, 5, 6, 7]
    assert table.n_dropped == 4


@pytest.mark.parametrize("seed", range(6))
def test_time_projection_equals_resolve_t1(seed: int) -> None:
    """Invariant 1: the span table and the panel purge read the same end times."""
    df = _labelled_panel(seed)
    pf = as_panel(df, "id", "date")
    times = pf.time_index().to_numpy()
    resolved = _resolve_t1("t1", pf, times)
    assert resolved is not None
    want = _time_positions(times, resolved)
    # Without dropping censored rows every label is kept (triple-barrier t1 is
    # always a row of its own entity), so the projection is exactly _resolve_t1.
    _, table = _spans_from_t1(df, t1="t1", entity="id", time="date", censored=None)
    assert np.array_equal(table.time_projection(), want)
    # Dropping censored labels can only make the projection less conservative.
    _, kept = _spans_from_t1(df, t1="t1", entity="id", time="date")
    assert np.all(kept.time_projection() <= want)


def test_subset_and_covers() -> None:
    df = pl.DataFrame({"id": ["a"] * 6, "t": list(range(6)), "t1": [1, 2, 3, 4, 5, 5]})
    _, table = _spans_from_t1(df, t1="t1")
    sub = table.subset(np.array([True, False, True, False, True, False]))
    assert sub.start.tolist() == [0, 2, 4] and sub.end.tolist() == [1, 3, 5]
    assert sub.covers(np.array([3])).tolist() == [False, True, False]
    with pytest.raises(ValueError):
        table.subset(np.array([True]))
