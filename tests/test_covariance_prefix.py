"""Bitwise prefix invariance and no look-ahead for the covariance frame ops.

Plan section 7.4: every frame op runs through ``assert_prefix_invariant`` and
``assert_no_lookahead`` with ``tol=0.0`` at three cuts, for ``stride`` in
{1, 5} and ``every`` in {5, "1w", "1mo"}, with and without ``group=``, on a
``synth`` panel with entity entry and exit.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core.panel_frame import PanelFrame
from panelary.covariance._state import market_loading, market_state
from panelary.covariance._window import rolling
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

W = 20
COV = 0.9


def _panel() -> PanelFrame:
    d = pn.synth.generate_panel(
        seed=3,
        n_entities=28,
        n_periods=150,
        start=dt.date(2021, 1, 4),
        every="1d",
        missing_rate=0.01,
        entry_rate=0.03,
        exit_rate=0.01,
        initial_fraction=0.6,
    )
    df = d.panel.with_columns(
        pl.col("entity")
        .str.slice(-1)
        .cast(pl.Int64)
        .mod(3)
        .cast(pl.String)
        .alias("sector")
    )
    return PanelFrame(df, entity="entity", time="time")


PANEL = _panel()
TIMES = PANEL.time_index().to_list()
CUTS = [TIMES[len(TIMES) * k // 4] for k in (1, 2, 3)]

SCHEDULES = [
    pytest.param({"stride": 1}, id="stride1"),
    pytest.param({"stride": 5}, id="stride5"),
    pytest.param({"schedule": 5}, id="every5"),
    pytest.param({"schedule": "1w"}, id="every1w"),
    pytest.param({"schedule": "1mo"}, id="every1mo"),
]
FEATURES = (
    "absorption_ratio",
    "ar_shift",
    "lambda1_share",
    "effective_rank",
    "eigen_entropy",
    "participation_ratio",
    "mp_signal_count",
    "market_ipr",
    "avg_corr",
)


def _state_op(sched: dict, group: str | None):
    def op(frame):
        return market_state(
            frame,
            returns="value",
            window=W,
            min_coverage=COV,
            features=FEATURES,
            group=group,
            min_entities=3,
            ar_short=3,
            ar_long=10,
            broadcast=True,
            **sched,
        )

    return op


def _nonvacuous(out: pl.DataFrame) -> None:
    assert out.get_column("absorption_ratio").drop_nulls().len() > 50


@pytest.mark.parametrize("sched", SCHEDULES)
@pytest.mark.parametrize("group", [None, "sector"])
def test_market_state_is_prefix_invariant_bitwise(
    sched: dict, group: str | None
) -> None:
    op = _state_op(sched, group)
    _nonvacuous(op(PANEL))
    for cut in CUTS:
        assert_prefix_invariant(op, PANEL, tol=0.0, cut=cut)


@pytest.mark.parametrize("sched", SCHEDULES)
@pytest.mark.parametrize("group", [None, "sector"])
def test_market_state_has_no_lookahead(sched: dict, group: str | None) -> None:
    op = _state_op(sched, group)
    for cut in CUTS:
        assert_no_lookahead(op, PANEL, tol=0.0, cut=cut)


@pytest.mark.parametrize("sched", SCHEDULES)
def test_market_loading_is_prefix_invariant_and_causal(sched: dict) -> None:
    def op(frame):
        return market_loading(
            frame, returns="value", window=W, min_coverage=COV, **sched
        )

    assert op(PANEL).get_column("market_loading").drop_nulls().len() > 200
    for cut in CUTS:
        assert_prefix_invariant(op, PANEL, tol=0.0, cut=cut)
        assert_no_lookahead(op, PANEL, tol=0.0, cut=cut)


@pytest.mark.parametrize("method", ["qis", "lw_identity", "mp_clip", "factor"])
def test_rolling_estimates_are_prefix_invariant_bitwise(method: str) -> None:
    full = rolling(PANEL, returns="value", window=W, min_coverage=COV, method=method)
    df = PANEL.collect()
    for cut in CUTS:
        pref = rolling(
            PanelFrame(df.filter(pl.col("time") <= cut), entity="entity", time="time"),
            returns="value",
            window=W,
            min_coverage=COV,
            method=method,
        )
        for date in [d for d in TIMES if d <= cut][W:]:
            try:
                a = full.at(date)
            except LookupError:
                with pytest.raises(LookupError):
                    pref.at(date)
                continue
            b = pref.at(date)
            assert a.entities == b.entities
            assert np.array_equal(a.to_dense(), b.to_dense())
            assert a.shrinkage == b.shrinkage


def test_every_date_is_covered_by_the_output() -> None:
    out = market_state(PANEL, returns="value", window=W, min_coverage=COV, stride=5)
    assert out.get_column("time").to_list() == TIMES
    grid = out.filter(pl.col("time") == pl.col("asof_date"))
    assert grid.height == len(range(0, len(TIMES), 5))


@pytest.mark.parametrize("refit", [5, "1w", "1mo"])
@pytest.mark.parametrize("group", [None, "sector"])
def test_turbulence_is_prefix_invariant_and_causal(refit, group: str | None) -> None:
    from panelary.covariance._turbulence import turbulence

    def op(frame):
        return turbulence(
            frame, returns="value", window=W, min_coverage=COV, refit=refit,
            group=group, min_entities=3, pct_window=10, broadcast=True,
        )  # fmt: skip

    assert op(PANEL).get_column("turbulence").drop_nulls().len() > 500
    for cut in CUTS:
        assert_prefix_invariant(op, PANEL, tol=0.0, cut=cut)
        assert_no_lookahead(op, PANEL, tol=0.0, cut=cut)


def test_group_market_loading_is_prefix_invariant() -> None:
    def op(frame):
        return market_loading(
            frame, returns="value", window=W, min_coverage=COV, group="sector",
            min_entities=3, schedule="1w",
        )  # fmt: skip

    for cut in CUTS:
        assert_prefix_invariant(op, PANEL, tol=0.0, cut=cut)
        assert_no_lookahead(op, PANEL, tol=0.0, cut=cut)
