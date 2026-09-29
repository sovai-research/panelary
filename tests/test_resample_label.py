"""Bug B3: ``preprocessing.resample`` stamped windows at their left edge.

A window ``[s, s + freq)`` stamped ``s`` contains data from after ``s``. As a
feature that is a look-ahead; ``label="right"`` stamps it at ``s + freq``. The
default keeps the old stamping (it would silently move every output) and warns.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.preprocessing import resample
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

_T0 = datetime(2024, 1, 1)


def _hourly(n_hours: int = 24) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    frames = [
        pl.DataFrame(
            {
                "id": [name] * n_hours,
                "t": [_T0 + timedelta(hours=i) for i in range(n_hours)],
                "x": rng.standard_normal(n_hours),
            }
        )
        for name in ("a", "b")
    ]
    return pl.concat(frames)


def _run(df: pl.DataFrame, **kw: object) -> pl.DataFrame:
    out = resample("2h", "sum", 0, **kw)(df)  # type: ignore[arg-type]
    return out.collect().sort("id", "t")


def test_default_warns_and_keeps_the_left_edge() -> None:
    df = _hourly()
    with pytest.warns(FutureWarning, match="label='right'"):
        default = _run(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # explicit label: no FutureWarning
        left = _run(df, label="left")
    assert default.equals(left)
    assert default["t"][0] == _T0
    first = df.filter(pl.col("id") == "a")["x"].head(2).sum()
    assert default["x"][0] == pytest.approx(first)


def test_right_label_stamps_the_window_close() -> None:
    df = _hourly()
    left = _run(df, label="left")
    right = _run(df, label="right")
    assert right["x"].equals(left["x"])
    assert (right["t"] - left["t"] == timedelta(hours=2)).all()


def test_no_polars_deprecation_or_performance_warning() -> None:
    """``group_by=`` rather than the deprecated ``by=``; schema via collect_schema."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(_hourly(), label="right")


def test_invalid_label_raises() -> None:
    with pytest.raises(ValueError, match="label"):
        resample("2h", "sum", 0, label="center").func  # noqa: B018


@pytest.mark.parametrize("cut_hour", [9, 10, 11, 14])
def test_right_label_is_leak_safe(cut_hour: int) -> None:
    panel = PanelFrame(_hourly(), entity="id", time="t")
    op = resample("2h", "sum", 0, label="right")
    assert_no_lookahead(op, panel, cut=_T0 + timedelta(hours=cut_hour))
    assert_prefix_invariant(op, panel, cut=_T0 + timedelta(hours=cut_hour))


@pytest.mark.parametrize("cut_hour", [10, 14])
def test_left_label_looks_ahead(cut_hour: int) -> None:
    """At an even-hour cut the window stamped ``cut`` holds ``cut + 1h``."""
    panel = PanelFrame(_hourly(), entity="id", time="t")
    op = resample("2h", "sum", 0, label="left")
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(op, panel, cut=_T0 + timedelta(hours=cut_hour))
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(op, panel, cut=_T0 + timedelta(hours=cut_hour))
