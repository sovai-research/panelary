"""The causal spine: ``trailing_windows`` and ``PanelWindows`` (plan section 3.2).

* Row ``t`` reads ``values[t - W + 1 : t + 1]`` and nothing else.
* Never centred: ``center=`` is absent, not defaulted.
* Rows with fewer than ``min_periods`` observations are NaN -- never
  back-filled, never computed on a short window.
* Windows never cross an entity boundary.
* The single-entity primitive returns a view (framing allocates nothing).
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.shape import PanelWindows, trailing_windows
from panelary.shape._window import sort_panel


def test_row_t_reads_exactly_its_trailing_window() -> None:
    x = np.arange(10, dtype=float)
    w = trailing_windows(x, 4)
    assert w.shape == (10, 4)
    for t in range(10):
        lo = t - 3
        expected = [np.nan] * max(0, -lo) + list(x[max(lo, 0) : t + 1])
        assert np.array_equal(w[t], expected, equal_nan=True)


def test_two_dimensional_values_keep_the_window_axis_second() -> None:
    x = np.arange(12, dtype=float).reshape(6, 2)
    w = trailing_windows(x, 3)
    assert w.shape == (6, 3, 2)
    assert np.array_equal(w[4], x[2:5])


def test_trailing_windows_is_a_view_when_min_periods_is_the_window() -> None:
    x = np.random.default_rng(0).standard_normal(3000)
    w = trailing_windows(x, 64)
    assert not w.flags.owndata and w.base is not None
    assert w.nbytes == 3000 * 64 * 8  # logical size; nothing materialised


def test_min_periods_blanks_short_rows_and_never_backfills() -> None:
    x = np.arange(1.0, 9.0)
    w = trailing_windows(x, 4, min_periods=2)
    assert np.isnan(w[0]).all()  # one observation < min_periods=2
    assert np.array_equal(w[1], [np.nan, np.nan, 1.0, 2.0], equal_nan=True)
    with pytest.raises(ValueError, match="cannot exceed"):
        trailing_windows(x, 3, min_periods=4)
    with pytest.raises(ValueError, match="positive int"):
        trailing_windows(x, 0)


def test_prefix_invariance_of_the_primitive() -> None:
    x = np.random.default_rng(1).standard_normal(50)
    full = trailing_windows(x, 7)
    for T in (1, 6, 7, 20, 49):
        assert np.array_equal(trailing_windows(x[:T], 7), full[:T], equal_nan=True)


def test_there_is_no_center_parameter() -> None:
    for fn in (
        trailing_windows,
        PanelWindows.__init__,
        PanelWindows.windows,
        PanelWindows.rolling,
    ):
        assert "center" not in inspect.signature(fn).parameters


def test_panel_windows_never_cross_an_entity_boundary() -> None:
    lengths = np.array([3, 5, 2], dtype=np.int64)
    values = np.arange(10, dtype=float)[:, None] + 100.0
    pw = PanelWindows(values, lengths, reach=4)
    w = pw.windows(4)[:, :, 0]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    for e, (s, n) in enumerate(zip(starts, lengths, strict=True)):
        for i in range(n):
            row = s + i
            own = values[s : row + 1, 0][-4:]
            expected = np.concatenate([np.full(4 - own.size, np.nan), own])
            assert np.array_equal(w[row], expected, equal_nan=True), (e, i)
    assert np.array_equal(pw.position, [0, 1, 2, 0, 1, 2, 3, 4, 0, 1])
    assert np.isnan(pw.lagged(3)[3, 0])  # entity 1's first row: nothing 3 back


def test_segment_statistics_match_a_naive_per_window_computation() -> None:
    rng = np.random.default_rng(2)
    lengths = np.array([9, 4, 12], dtype=np.int64)
    values = rng.standard_normal((25, 2))
    pw = PanelWindows(values, lengths, reach=6)
    wins = pw.windows(3)
    naive = {
        "mean": np.mean(wins, axis=1),
        "max": np.max(wins, axis=1),
        "min": np.min(wins, axis=1),
        "std": np.std(wins, axis=1),
        "sum": np.sum(wins, axis=1),
        "last": wins[:, -1],
    }
    for stat, ref in naive.items():
        got = pw.rolling(3, stat)
        assert np.allclose(got, ref, equal_nan=True, rtol=1e-12, atol=1e-12), stat
    with pytest.raises(ValueError, match="unknown statistic"):
        pw.rolling(3, "median")


def test_iter_windows_chunks_equal_the_one_shot_windows() -> None:
    rng = np.random.default_rng(3)
    lengths = np.array([7, 3, 11], dtype=np.int64)
    pw = PanelWindows(rng.standard_normal((21, 1)), lengths, reach=5)
    whole = pw.windows(5)
    parts = np.concatenate([w for _sl, w in pw.iter_windows(5, chunk_rows=4)])
    assert np.array_equal(whole, parts, equal_nan=True)


def test_sort_panel_refuses_duplicate_keys_and_round_trips_row_order() -> None:
    df = pl.DataFrame(
        {"id": ["b", "a", "b", "a"], "t": [1, 0, 0, 1], "x": [4.0, 1.0, 3.0, 2.0]}
    )
    sp = sort_panel(PanelFrame(df, entity="id", time="t"), ["x"], owner="T")
    assert sp.values[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(sp.unsort(sp.values)[:, 0], df["x"].to_numpy())
    dup = pl.concat([df, df.head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        sort_panel(
            PanelFrame(dup, entity="id", time="t", validate=False), ["x"], owner="T"
        )
