"""The look-ahead audits must not hinge on where one cut happens to fall.

Regression for an external report: a *period-average* leak -- every row gets
the mean of its calendar period, which includes the later rows of the same
period -- passed ``assert_no_lookahead`` at its single default cut (the median
time, which on that panel was the last day of a period) and failed at 200 of
the other 249 cuts. ``assert_prefix_invariant`` and ``evolve.assert_causal``
used three fixed fractions of the axis, which a period length can divide just
as well. The defaults now test a deterministic spread of *consecutive pairs*
of cuts, so at least one cut of each pair falls strictly inside any period
longer than one step.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

E, T = "entity", "time"


def _panel(n_times: int, *, n_entities: int = 4, seed: int = 0) -> PanelFrame:
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {
            E: np.repeat(np.arange(n_entities), n_times),
            T: np.tile(np.arange(n_times), n_entities),
            "x": rng.standard_normal(n_entities * n_times),
        }
    )
    return PanelFrame(df, entity=E, time=T)


def _period_average(period: int):
    """Each row gets the mean of ``x`` over its entity's calendar period."""

    def op(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("x").mean().over(E, pl.col(T) // period).alias("period_mean")
        )

    return op


def _trailing_mean(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("x").rolling_mean(5, min_samples=1).over(E).alias("trailing_mean")
    )


class TestAssertNoLookahead:
    def test_the_reported_case_fails_at_most_cuts_but_not_the_old_default(
        self,
    ) -> None:
        panel, op = _panel(250), _period_average(5)
        caught = 0
        for cut in range(249):
            try:
                assert_no_lookahead(op, panel, cut=cut)
            except AssertionError:
                caught += 1
        assert caught == 200  # every cut that is not a period's last day
        assert_no_lookahead(op, panel, cut=124)  # the old median default

    @pytest.mark.parametrize(("n_times", "period"), [(250, 5), (240, 5), (240, 12)])
    def test_the_default_catches_a_period_average_leak(
        self, n_times: int, period: int
    ) -> None:
        with pytest.raises(AssertionError, match="LOOK-AHEAD"):
            assert_no_lookahead(_period_average(period), _panel(n_times))

    def test_the_default_still_passes_a_causal_op(self) -> None:
        assert_no_lookahead(_trailing_mean, _panel(240), tol=0.0)

    def test_cuts_can_be_chosen_explicitly(self) -> None:
        panel, op = _panel(240), _period_average(5)
        assert_no_lookahead(op, panel, cuts=[4, 119, 234])  # period ends only
        with pytest.raises(AssertionError, match=r"time <= 120"):
            assert_no_lookahead(op, panel, cuts=[119, 120])

    def test_cut_and_cuts_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            assert_no_lookahead(_trailing_mean, _panel(20), cut=5, cuts=[5])

    def test_explicit_cuts_are_validated(self) -> None:
        with pytest.raises(ValueError, match="no future rows"):
            assert_no_lookahead(_trailing_mean, _panel(20), cuts=[3, 19])
        with pytest.raises(ValueError, match="at least one"):
            assert_no_lookahead(_trailing_mean, _panel(20), cuts=[])


class TestAssertPrefixInvariant:
    @pytest.mark.parametrize(("n_times", "period"), [(240, 5), (240, 12)])
    def test_the_default_catches_a_period_average_leak(
        self, n_times: int, period: int
    ) -> None:
        # On 240 dates the old quarter / half / three-quarter cuts (59, 119,
        # 179) are all period ends for a 5- or 12-step period.
        panel, op = _panel(n_times), _period_average(period)
        for old_cut in (59, 119, 179):
            assert_prefix_invariant(op, panel, cut=old_cut)
        with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
            assert_prefix_invariant(op, panel)

    def test_the_default_still_passes_a_causal_op(self) -> None:
        assert_prefix_invariant(_trailing_mean, _panel(240), tol=0.0)

    def test_cuts_can_be_chosen_explicitly(self) -> None:
        panel, op = _panel(240), _period_average(5)
        assert_prefix_invariant(op, panel, cuts=[4, 119])
        with pytest.raises(AssertionError, match=r"time <= 120"):
            assert_prefix_invariant(op, panel, cuts=[120])
        with pytest.raises(ValueError, match="not both"):
            assert_prefix_invariant(op, panel, cut=5, cuts=[5])


class TestEvolveAssertCausal:
    def test_the_default_catches_a_period_average_leak(self) -> None:
        evolve = pytest.importorskip("panelary.evolve")
        df = _panel(240).collect()
        leak = pl.col("x").mean().over(E, pl.col(T) // 12)
        # The old default fractions (0.4, 0.6, 0.8) cut after dates 95, 143
        # and 191: all ends of a 12-step period.
        assert evolve.assert_causal(
            leak, df, entity=E, time=T, cut_fractions=(0.4, 0.6, 0.8)
        )
        assert not evolve.assert_causal(leak, df, entity=E, time=T)
        assert evolve.assert_causal(
            pl.col("x").rolling_mean(5).over(E), df, entity=E, time=T
        )
