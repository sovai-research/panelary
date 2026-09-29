"""One refit / evaluation schedule for the whole package (coordination note D1).

Several subpackages evaluate or refit something on a grid of dates -- the
as-of covariance engine, state-space refits, conditional-volatility refits,
drift-monitor baselines, graph rebuilds. They share this module instead of
each growing its own, because the subtle part is the same everywhere: **the
grid must never move when data is appended.**

* :class:`Schedule` says *which* dates are on the grid.

  - An integer ``every=k`` selects time-axis positions ``i`` with
    ``i % k == 0``, counted from the panel's **first** date
    (``anchor="origin"``); ``anchor="position"`` instead counts
    ``start, start + k, ...`` from a caller-supplied start (the convention of
    :func:`panelary.detect._panel.residualise`, whose refits are at
    ``min_periods + j * refit_every``).
  - A calendar duration (``"1w"``, ``"1mo"``, ``"1q"``, ``"1y"``, ``"5d"``)
    selects the **first** date of each period, i.e. where
    ``time.dt.truncate(every)`` changes. Whether ``t`` is on the grid depends
    only on ``t`` and the date before it.
  - ``"every"`` (or ``1``) is every date.
  - End-anchored grids -- "the last trading day of the month", or positions
    counted back from the end of the sample -- are **refused**: knowing that
    ``t`` is the last date of its period requires ``t + 1`` (trap T6), and a
    grid counted from the end moves every point when a row is appended.
    ``expanding_window_split`` / ``sliding_window_split`` are not schedules
    for exactly this reason.

* :class:`Refit` layers a fitting policy on a schedule: the training window,
  the minimum training length, an embargo ``lag`` and a ``burn`` hint for
  recursive consumers. The training rows for a refit at position ``r`` are
  ``[r - lag - window, r - lag)`` (all of ``[0, r - lag)`` when ``window`` is
  ``None``), so a refit never trains on its own date.

Durations are parsed only by :mod:`panelary.core._calendar`; there is no
second parser. This module imports numpy, polars and the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.core._calendar import (
    _SUB_DAY_UNITS,
    _TERM_RE,
    BusinessDays,
    is_steps,
    validate_duration,
)

__all__ = ["Refit", "Schedule", "as_schedule"]

_ANCHORS = ("origin", "position")
_END_ANCHORS = ("end", "last", "last_of_period", "period_end")


def _refuse_end_anchor(what: str) -> None:
    raise ValueError(
        f"{what}: end-anchored schedules are refused (trap T6). Knowing that t "
        "is the last date of its period requires t + 1, and a grid counted "
        "back from the end of the sample moves every point when data is "
        "appended. Use a calendar string such as '1mo' (the FIRST date of "
        "each period) or an integer step anchored at the panel's first date."
    )


def _as_series(times: Any) -> pl.Series:
    if isinstance(times, pl.Series):
        return times
    if isinstance(times, pl.DataFrame):
        if times.width != 1:
            raise ValueError("pass a single time column, not a frame")
        return times.to_series()
    return pl.Series(
        "time", list(times) if not isinstance(times, np.ndarray) else times
    )


@dataclass(frozen=True)
class Schedule:
    """Which dates of a sorted time axis are evaluation / refit dates.

    Parameters
    ----------
    every : int or str, default "every"
        ``"every"`` or ``1``: every date. An integer ``k >= 1``: every
        ``k``-th position. A Polars calendar duration (``"1w"``, ``"1mo"``,
        ``"1q"``, ``"1y"``, ``"3d"``): the first date of each period, on a
        ``Date`` / ``Datetime`` axis.
    anchor : {"origin", "position"}, default "origin"
        For an integer ``every``: ``"origin"`` selects ``i % k == 0`` counted
        from the first date on the axis; ``"position"`` selects
        ``start + j * k`` for the ``start`` passed to :meth:`positions` (see
        :class:`Refit`). Calendar schedules are always start-of-period.
        ``"end"`` / ``"last"`` raise (trap T6).

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.core._schedule import Schedule
    >>> Schedule(every=3).positions(pl.Series(range(10))).tolist()
    [0, 3, 6, 9]
    >>> Schedule(every=4, anchor="position").positions(range(10), start=2).tolist()
    [2, 6]
    """

    every: int | str = "every"
    anchor: str = "origin"

    def __post_init__(self) -> None:
        every = self.every
        if isinstance(every, str) and every.strip().lower() in _END_ANCHORS:
            _refuse_end_anchor(f"Schedule(every={every!r})")
        if self.anchor in _END_ANCHORS:
            _refuse_end_anchor(f"Schedule(anchor={self.anchor!r})")
        if self.anchor not in _ANCHORS:
            raise ValueError(
                f"`anchor` must be one of {_ANCHORS}, got {self.anchor!r}."
            )
        if isinstance(every, bool):
            raise TypeError("`every` must be an int or a duration string, not a bool.")
        if isinstance(every, str) and every.strip().lower() == "every":
            object.__setattr__(self, "every", 1)
            return
        if is_steps(every):
            if int(every) < 1:
                raise ValueError(f"`every` must be >= 1, got {every}.")
            object.__setattr__(self, "every", int(every))
            return
        if not isinstance(every, str):
            raise TypeError(
                "`every` must be an int (steps), 'every', or a calendar duration "
                f"string such as '1mo'; got {type(every).__name__!r}."
            )
        spec = validate_duration(every, name="every")
        if isinstance(spec, BusinessDays):
            raise ValueError(
                f"`every={every!r}`: business-day counts are not calendar periods; "
                "pass an int number of steps instead."
            )
        if self.anchor != "origin":
            raise ValueError(
                "`anchor` applies to integer schedules only; a calendar "
                "schedule is always the first date of each period."
            )
        object.__setattr__(self, "every", str(spec))

    @property
    def calendar(self) -> bool:
        """Whether this is a calendar-period schedule."""
        return isinstance(self.every, str)

    def mask(self, times: Any, *, start: int = 0) -> NDArray[np.bool_]:
        """Boolean mask over the sorted unique time axis ``times``.

        Parameters
        ----------
        times : polars.Series or sequence
            The **sorted, unique** time axis (only its length is used by an
            integer schedule).
        start : int, default 0
            Positions before ``start`` are never on the grid; with
            ``anchor="position"`` the grid is ``start, start + k, ...``.
        """
        if start < 0:
            raise ValueError(f"`start` must be >= 0, got {start}.")
        if self.calendar:
            s = _as_series(times)
            dtype = s.dtype
            if not (dtype == pl.Date or isinstance(dtype, pl.Datetime)):
                raise TypeError(
                    f"calendar schedule every={self.every!r} needs a Date or "
                    f"Datetime time axis, got {dtype}; pass an int step instead."
                )
            spec = str(self.every)
            units = {u for _, u in _TERM_RE.findall(spec)}
            if dtype == pl.Date and units & _SUB_DAY_UNITS:
                raise ValueError(f"every={spec!r} is a sub-day period on a Date axis.")
            key = s.dt.truncate(spec)
            first = (key != key.shift(1)).fill_null(value=True).to_numpy()
            out = np.asarray(first, dtype=bool)
        else:
            n = len(times) if not isinstance(times, pl.Series) else times.len()
            k = int(self.every)
            idx = np.arange(n)
            if self.anchor == "position":
                out = (idx >= start) & ((idx - start) % k == 0)
            else:
                out = idx % k == 0
        if start:
            out[: min(start, out.size)] = False
        return out

    def positions(self, times: Any, *, start: int = 0) -> NDArray[np.int64]:
        """Grid positions (indices into ``times``), ascending."""
        return np.flatnonzero(self.mask(times, start=start)).astype(np.int64)

    def asof_positions(self, times: Any, *, start: int = 0) -> NDArray[np.int64]:
        """For every position ``t``: the latest grid position ``<= t`` (-1 if none).

        This is the as-of forward fill: a value evaluated on the grid is
        carried to every later date until the next grid date. It is
        prefix-invariant because the grid is.
        """
        m = self.mask(times, start=start)
        idx = np.where(m, np.arange(m.size), -1)
        return np.maximum.accumulate(idx).astype(np.int64) if m.size else idx

    def __str__(self) -> str:
        if self.calendar:
            return f"first date of every {self.every}"
        if self.every == 1:
            return "every date"
        return f"every {self.every} steps ({self.anchor}-anchored)"


def as_schedule(spec: Schedule | int | str | None) -> Schedule:
    """Coerce ``None`` / an int / a string / a :class:`Schedule` to a Schedule."""
    if spec is None:
        return Schedule()
    if isinstance(spec, Schedule):
        return spec
    return Schedule(every=spec)


@dataclass(frozen=True)
class Refit:
    """A fitting policy on top of a :class:`Schedule`.

    Parameters
    ----------
    schedule : Schedule, int or str, default "every"
        Which dates are refit dates (coerced with :func:`as_schedule`).
    window : int, optional
        Training rows per refit; ``None`` uses an expanding window from the
        first date.
    min_train : int, default 1
        Minimum training rows on the time axis. Refits whose training window
        would be shorter are skipped (with ``anchor="position"`` the grid
        starts at ``min_train + lag``).
    lag : int, default 0
        Embargo in steps between the end of the training window and the
        refit date: training rows are ``[r - lag - window, r - lag)``.
    burn : int, default 0
        For recursive consumers (filters): how many rows before a refit to
        re-run from. The schedule itself does not use it.

    Examples
    --------
    ``detect.residualise(r, min_periods=100, refit_every=21, window=250)``
    refits at the positions of::

        Refit(Schedule(21, anchor="position"), window=250, min_train=100)
    """

    schedule: Schedule | int | str = "every"
    window: int | None = None
    min_train: int = 1
    lag: int = 0
    burn: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", as_schedule(self.schedule))
        for name in ("min_train", "lag", "burn"):
            v = getattr(self, name)
            if isinstance(v, bool) or not is_steps(v) or int(v) < 0:
                raise ValueError(f"`{name}` must be an int >= 0, got {v!r}.")
            object.__setattr__(self, name, int(v))
        if self.min_train < 1:
            raise ValueError(f"`min_train` must be >= 1, got {self.min_train}.")
        if self.window is not None:
            w = self.window
            if isinstance(w, bool) or not is_steps(w) or int(w) < 1:
                raise ValueError(f"`window` must be an int >= 1 or None, got {w!r}.")
            if int(w) < self.min_train:
                raise ValueError(
                    f"`window` ({w}) must be >= `min_train` ({self.min_train})."
                )
            object.__setattr__(self, "window", int(w))

    @property
    def first(self) -> int:
        """The earliest position with ``min_train`` training rows before it."""
        return self.min_train + self.lag

    def positions(self, times: Any) -> NDArray[np.int64]:
        """Refit positions on the sorted unique time axis ``times``."""
        sched = self.schedule
        assert isinstance(sched, Schedule)
        return sched.positions(times, start=self.first)

    def train_slice(self, r: int) -> slice:
        """Training rows ``[lo, hi)`` for a refit at position ``r``."""
        hi = int(r) - self.lag
        if hi < self.min_train:
            raise ValueError(
                f"a refit at position {r} has {max(hi, 0)} training rows, fewer "
                f"than min_train={self.min_train}."
            )
        lo = 0 if self.window is None else max(0, hi - self.window)
        return slice(lo, hi)

    def in_force(self, times: Any) -> NDArray[np.int64]:
        """For every position ``t``: the refit position in force (-1 if none)."""
        sched = self.schedule
        assert isinstance(sched, Schedule)
        return sched.asof_positions(times, start=self.first)
