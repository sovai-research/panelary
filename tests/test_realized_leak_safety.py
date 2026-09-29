"""Leak safety of the intraday -> daily realized measures (plan 5, section 9).

The daily output is keyed by ``(entity, close stamp)``, so the panel verifiers
apply to it directly once the cut sits at a session boundary: every session at
or before the cut is complete in both runs. Both instruments run bitwise
(``tol=0``) with every measure switched on, including the per-session realized
kernel bandwidth and the automatic TSRV scale (trap 3).

Each check is shown to have teeth: a deliberately leaky variant -- a kernel
bandwidth pooled over all sessions, and the per-row broadcast of trap 5 -- is
caught by the same instrument.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from panelary.core import PanelFrame
from panelary.econ.features import intraday_realized_measures
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

_MEASURES = [
    "rv",
    "rv_ss",
    "bv",
    "medrv",
    "minrv",
    "rs_pos",
    "rs_neg",
    "sjv",
    "rq",
    "tpq",
    "jump_z",
    "n_obs",
    "jump",
    "rel_jump",
    "jump_sig",
    "cont",
    "medrq",
    "log_rv_var",
    "log_rv_var_tpq",
    "rk",
    "tsrv",
    "pav",
]
_SESSIONS = 8
_DAY0 = dt.date(2024, 1, 1)


def _intraday_panel() -> pl.DataFrame:
    """Three entities, ragged sessions, noisy prices; time = session * 1000 + tick.

    The session key is a Date (not numeric), so the verifiers perturb only the
    price column.
    """
    rng = np.random.default_rng(20260929)
    rows: list[tuple[str, dt.date, int, float]] = []
    for ent in ("A", "B", "C"):
        for sess in range(_SESSIONS):
            m = int(rng.integers(60, 300))
            efficient = np.cumsum(rng.standard_normal(m) * 4e-4)
            logp = np.log(40.0) + efficient + rng.standard_normal(m) * 2e-4
            day = _DAY0 + dt.timedelta(days=sess)
            rows.extend(
                (ent, day, sess * 1000 + i, float(np.exp(v)))
                for i, v in enumerate(logp)
            )
    return pl.DataFrame(rows, schema=["e", "s", "t", "px"], orient="row")


_PANEL = PanelFrame(_intraday_panel(), entity="e", time="t")
_BOUNDARY_CUTS = [1999, 3999, 6999]  # after sessions 1, 3 and 6


def _daily(frame: Any, **extra: Any) -> pl.DataFrame:
    return intraday_realized_measures(
        frame,
        entity="e",
        session="s",
        time="t",
        price="px",
        measures=_MEASURES,
        min_obs=10,
        **extra,
    )


@pytest.mark.parametrize("cut", _BOUNDARY_CUTS)
def test_daily_measures_never_look_ahead(cut: int) -> None:
    """Trap 3: perturbing later sessions leaves earlier days bit-identical."""
    assert_no_lookahead(_daily, _PANEL, cut=cut, tol=0.0)


@pytest.mark.parametrize("cut", _BOUNDARY_CUTS)
def test_daily_measures_are_prefix_invariant(cut: int) -> None:
    assert_prefix_invariant(_daily, _PANEL, cut=cut, tol=0.0)


@pytest.mark.parametrize("omega2", ["bnhls", "debiased"])
def test_fixed_settings_are_prefix_invariant_too(omega2: str) -> None:
    def op(frame: Any) -> pl.DataFrame:
        return _daily(frame, omega2=omega2, kernel_bandwidth=4, tsrv_k=5)

    for cut in _BOUNDARY_CUTS:
        assert_no_lookahead(op, _PANEL, cut=cut, tol=0.0)
        assert_prefix_invariant(op, _PANEL, cut=cut, tol=0.0)


def test_a_pooled_bandwidth_is_caught() -> None:
    """The leaky alternative to a day-t bandwidth: one H for the whole sample."""

    def leaky(frame: Any) -> pl.DataFrame:
        per_day = _daily(frame)
        pooled_h = int(per_day["rk_h"].median())  # uses every session, future included
        return _daily(frame, kernel_bandwidth=min(pooled_h + 1, 30))

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        for cut in _BOUNDARY_CUTS:
            assert_no_lookahead(leaky, _PANEL, cut=cut, tol=0.0)


def test_broadcasting_onto_intraday_rows_is_caught() -> None:
    """Trap 5: joining a day's value back onto that day's ticks leaks the close."""

    def broadcast(frame: Any) -> pl.DataFrame:
        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        return df.join(_daily(df).drop("t"), on=["e", "s"], how="left")

    # Cuts in the middle of sessions: the broadcast value for an early tick
    # depends on the ticks after it.
    with pytest.raises(AssertionError):
        assert_prefix_invariant(broadcast, _PANEL, cut=2100, tol=0.0)


def test_label_left_stamps_after_the_last_open() -> None:
    """Trap 10: an open-stamped bar is known only at its close."""
    out = intraday_realized_measures(
        _PANEL.collect(), entity="e", session="s", time="t", price="px", label="left"
    )
    right = intraday_realized_measures(
        _PANEL.collect(), entity="e", session="s", time="t", price="px"
    )
    assert (out["t"] - right["t"] == 1).all()
