"""The shared refit / evaluation schedule (coordination note D1).

The property every consumer relies on: **appending data never moves a grid
point**, so ``mask(times[:k]) == mask(times)[:k]`` for every ``k``. Plus trap
T6 (end-anchored grids are refused) and the ``anchor="position"`` mode that
reproduces ``detect._panel.residualise``'s refits exactly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from panelary.core._schedule import Refit, Schedule, as_schedule


def _dates(n: int, start: dt.date = dt.date(2023, 12, 27)) -> pl.Series:
    days = [start + dt.timedelta(days=i) for i in range(int(n * 1.5))]
    days = [d for d in days if d.weekday() < 5][:n]
    return pl.Series("date", days, dtype=pl.Date)


@pytest.mark.parametrize(
    "sched",
    [
        Schedule(),
        Schedule(every=5),
        Schedule(every=7, anchor="position"),
        Schedule(every="1w"),
        Schedule(every="1mo"),
        Schedule(every="1q"),
    ],
    ids=str,
)
def test_appending_never_moves_the_grid(sched: Schedule) -> None:
    times = _dates(300)
    full = sched.mask(times, start=3)
    for k in (1, 2, 17, 60, 151, 299):
        assert np.array_equal(sched.mask(times[:k], start=3), full[:k])
    fill = sched.asof_positions(times, start=3)
    for k in (5, 100, 250):
        assert np.array_equal(sched.asof_positions(times[:k], start=3), fill[:k])


def test_integer_grid_is_anchored_at_the_first_date() -> None:
    assert Schedule(every=3).positions(range(10)).tolist() == [0, 3, 6, 9]
    assert Schedule(every=3).positions(range(10), start=4).tolist() == [6, 9]
    pos = Schedule(every=3, anchor="position").positions(range(10), start=4)
    assert pos.tolist() == [4, 7]
    assert Schedule("every").positions(range(4)).tolist() == [0, 1, 2, 3]
    assert Schedule(1) == Schedule("every")


def test_calendar_grid_is_the_first_date_of_each_period() -> None:
    times = _dates(70)
    pos = Schedule(every="1mo").positions(times)
    got = [times[int(i)] for i in pos]
    months = times.dt.truncate("1mo")
    firsts = [
        times[int(i)]
        for i in range(times.len())
        if i == 0 or months[i] != months[i - 1]
    ]
    assert got == firsts
    assert all(d.day <= 3 or i == 0 for i, d in enumerate(got))
    weeks = [times[int(i)] for i in Schedule(every="1w").positions(times)]
    assert all(d.weekday() == 0 for d in weeks[1:])  # Mondays (no holidays here)


def test_calendar_grid_works_on_datetime_axes() -> None:
    t = pl.Series(
        "ts",
        [
            dt.datetime(2024, 1, 31, 16),
            dt.datetime(2024, 2, 1, 9),
            dt.datetime(2024, 2, 1, 16),
        ],
    )
    assert Schedule(every="1mo").mask(t).tolist() == [True, True, False]


@pytest.mark.parametrize(
    "kwargs",
    [{"every": "last"}, {"every": "end"}, {"anchor": "end"}, {"anchor": "last"}],
)
def test_T6_end_anchored_schedules_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="T6"):
        Schedule(**kwargs)


def test_bad_specs() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        Schedule(every=0)
    with pytest.raises(TypeError):
        Schedule(every=True)
    with pytest.raises(ValueError, match="business-day"):
        Schedule(every="5bd")
    with pytest.raises(ValueError, match="duration"):
        Schedule(every="monthly")
    with pytest.raises(ValueError, match="integer schedules only"):
        Schedule(every="1mo", anchor="position")
    with pytest.raises(TypeError, match="Date or Datetime"):
        Schedule(every="1mo").mask(pl.Series([1, 2, 3]))
    with pytest.raises(ValueError, match="sub-day"):
        Schedule(every="6h").mask(_dates(5))
    assert as_schedule(None) == Schedule()
    assert as_schedule(5) == Schedule(5)
    assert as_schedule("1mo") == Schedule("1mo")


def test_refit_positions_slices_and_in_force() -> None:
    r = Refit(Schedule(every=5), window=10, min_train=4, lag=2)
    assert r.first == 6
    pos = r.positions(range(30))
    assert pos.tolist() == [10, 15, 20, 25]
    assert r.train_slice(10) == slice(0, 8)
    assert r.train_slice(25) == slice(13, 23)
    force = r.in_force(range(30))
    assert force[:10].tolist() == [-1] * 10
    assert force[10:16].tolist() == [10, 10, 10, 10, 10, 15]
    expanding = Refit(Schedule(every=5), min_train=3)
    assert expanding.train_slice(20) == slice(0, 20)
    with pytest.raises(ValueError, match="fewer than min_train"):
        expanding.train_slice(2)
    with pytest.raises(ValueError, match="window"):
        Refit(window=2, min_train=5)
    with pytest.raises(ValueError, match="lag"):
        Refit(lag=-1)


@pytest.mark.parametrize(
    ("min_periods", "refit_every", "window"),
    [(30, 7, None), (25, 10, 40), (60, 1, 60), (40, 50, None)],
)
def test_position_anchor_reproduces_residualise_refits(
    monkeypatch: pytest.MonkeyPatch,
    min_periods: int,
    refit_every: int,
    window: int | None,
) -> None:
    """Every training block residualise fits on is exactly Refit's slice."""
    import panelary.detect._panel as dpanel

    rng = np.random.default_rng(0)
    r = rng.standard_normal((12, 173))
    blocks: list[np.ndarray] = []
    real = dpanel._fit_loadings

    def spy(block, k, floor):  # noqa: ANN001, ANN202 - mirrors the private signature
        blocks.append(np.array(block, copy=True))
        return real(block, k, floor)

    monkeypatch.setattr(dpanel, "_fit_loadings", spy)
    out = dpanel.residualise(
        r, n_factors=1, min_periods=min_periods, refit_every=refit_every, window=window
    )
    refit = Refit(
        Schedule(every=refit_every, anchor="position"),
        window=window,
        min_train=min_periods,
    )
    pos = refit.positions(range(r.shape[1]))
    assert len(blocks) == len(pos)
    for block, p in zip(blocks, pos, strict=True):
        assert np.array_equal(block, r[:, refit.train_slice(int(p))])
    # and the frozen betas are in force exactly where Refit says
    force = refit.in_force(range(r.shape[1]))
    assert np.isnan(out[:, force < 0]).all()
    assert np.isfinite(out[:, force >= 0]).all()
