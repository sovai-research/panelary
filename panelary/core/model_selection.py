"""Leak-safe, finance-grade cross-validation for panels.

This module implements the cross-validation machinery from Marcos Lopez de
Prado, *Advances in Financial Machine Learning* (Wiley, 2018), adapted to
long-format Polars panels:

* :class:`PurgedKFold` — k-fold CV that **purges** training observations whose
  labels overlap the test window and applies an **embargo** after each test
  fold (de Prado, Ch. 7).
* :class:`CombinatorialPurgedCV` — the Combinatorial Purged Cross-Validation
  (CPCV) scheme that tests every combination of ``k`` of ``N`` groups, producing
  many backtest paths (de Prado, Ch. 12).
* :func:`expanding_window_split` / :func:`sliding_window_split` — walk-forward
  splitters on the shared unique-time axis. They take their fold schedule from
  the one kernel the library has for it,
  :func:`panelary.cross_validation._walk_forward_cutoffs`, but they are *not*
  the row-based splitters of the same name in :mod:`panelary.cross_validation`:
  those slice each entity's own rows, these slice the common time index.
* :func:`deflated_sharpe_ratio` — the Deflated Sharpe Ratio (Bailey & de Prado,
  2014) correcting for multiple testing, non-normality and sample length.
* :func:`probability_of_backtest_overfitting` — the PBO via Combinatorially
  Symmetric Cross-Validation (Bailey, Borwein, de Prado, Zhu, 2017).

Leakage model
-------------
Time alignment is along the **time axis**, shared across entities (a panel
backtest rebalances on common dates). Splitters therefore operate on the sorted
unique time index; purging and embargo are applied in time units, and the
resulting train/test time sets are mapped back to *all* entities, so per-entity
grouping is respected automatically (an entity contributes its rows for the
selected times only).

Each splitter is *event-based*: a "label" for an observation at time ``t`` is
assumed to span ``[t, t + horizon]`` (the prediction horizon). An observation is
**purged** from the training set if its label window overlaps any test label
window; an additional **embargo** of ``embargo`` time-steps after each test
block is removed from training to handle serial correlation.

Both ``horizon`` and ``embargo`` may instead be a **calendar duration** -- a
``datetime.timedelta``, a Polars duration string (``"5d"``, ``"1mo"``), or
:class:`BusinessDays` (``"5bd"`` for short) -- applied to the actual time
values rather than to positions. On an irregular axis (weekends, holidays, a
gap in the data) "five rows" and "five business days" are different amounts of
time, and only the second is what a serial-correlation argument is about. An
``int`` keeps its historical meaning (positions in the sorted unique-time
index) and produces byte-identical folds.

All splitters yield ``(train, test)`` as :class:`PanelFrame` pairs by default,
or as integer time-index positions when ``return_indices=True``.
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core._calendar import (
    BusinessDays,
    Duration,
    as_steps,
    is_steps,
    shift_forward,
    validate_duration,
)
from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.core.protocol import PanelTransformer
from panelary.cross_validation import _walk_forward_cutoffs

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BusinessDays",
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "expanding_window_split",
    "sliding_window_split",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "CVReport",
    "cross_validate",
    "validate",
]

# A fold is (train_panel, test_panel) or (train_time_idx, test_time_idx).
PanelFold = tuple[PanelFrame, PanelFrame]
IndexFold = tuple["NDArray[np.int64]", "NDArray[np.int64]"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _unique_times(panel: PanelFrame) -> np.ndarray:
    """Return the sorted unique time values of ``panel`` as a numpy array."""
    return panel.time_index().to_numpy()


def _sharpe(
    returns: np.ndarray | Sequence[float],
    *,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> float:
    """Sharpe of a return series. NaN (not 0) on degenerate/empty input.

    Naive ``mu / sd``. If ``periods_per_year`` is given the result is annualised
    by ``sqrt(periods_per_year)``; otherwise it is per-observation (the frequency
    DSR/PSR expect). Returns ``nan`` when fewer than 2 finite points or ``sd == 0``
    so downstream stats can drop it rather than treat luck as skill.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if not (sd > 0):
        return float("nan")
    sr = r.mean() / sd
    if periods_per_year is not None:
        sr *= math.sqrt(periods_per_year)
    return float(sr)


def _perf_stat(
    rows: np.ndarray,
    statistic: str | Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Column-wise performance statistic for a ``(T, S)`` slice.

    ``"mean"`` -> per-column mean; ``"sharpe"`` -> per-column :func:`_sharpe`
    (per-observation, degenerate columns become ``nan``); a callable is applied
    to the ``(T, S)`` matrix and must return an ``(S,)`` vector.
    """
    if callable(statistic):
        out = np.asarray(statistic(rows), dtype=float)
        if out.shape != (rows.shape[1],):
            raise ValueError(
                "custom `statistic` must return one value per strategy column "
                f"(expected shape {(rows.shape[1],)}, got {out.shape})."
            )
        return out
    if statistic == "mean":
        return rows.mean(axis=0)
    if statistic == "sharpe":
        return np.array([_sharpe(rows[:, j]) for j in range(rows.shape[1])])
    raise ValueError(
        f"unknown `statistic` {statistic!r}; use 'mean', 'sharpe', or a callable."
    )


def _subset_by_times(panel: PanelFrame, times: Sequence) -> PanelFrame:
    """Return a PanelFrame containing only rows whose time is in ``times``.

    Stays lazy; uses ``is_in`` against the selected time values, so every entity
    contributes its rows for those times (panel grouping respected).
    """
    # `.implode()` gives the unambiguous set-membership form of `is_in`
    # (a plain same-dtype collection is deprecated in recent Polars).
    time_vals = pl.Series(values=list(times)).implode()
    lf = panel.lazy().filter(pl.col(panel.time_col).is_in(time_vals))
    return PanelFrame(lf, entity=panel.entity_col, time=panel.time_col, validate=False)


def _contiguous_blocks(positions: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted positions into contiguous ``[start, end]`` index blocks."""
    if positions.size == 0:
        return []
    blocks: list[tuple[int, int]] = []
    start = prev = int(positions[0])
    for p in positions[1:]:
        p = int(p)
        if p == prev + 1:
            prev = p
        else:
            blocks.append((start, prev))
            start = prev = p
    blocks.append((start, prev))
    return blocks


def _purge_embargo_positions(
    n_times: int,
    test_positions: np.ndarray,
    horizon: int,
    embargo: int,
) -> np.ndarray:
    """Return the *train* time-index positions after purge + embargo.

    Parameters
    ----------
    n_times : int
        Number of unique time steps.
    test_positions : ndarray of int
        Positions (into the sorted unique-time index) belonging to the test set.
    horizon : int
        Label horizon in time-steps. An observation at position ``i`` has a label
        spanning positions ``[i, i + horizon]``. Training observations whose
        label window overlaps any test position are purged.
    embargo : int
        Number of time-steps after each contiguous test block to additionally
        remove from training.

    Returns
    -------
    ndarray of int
        Sorted training positions.

    Notes
    -----
    Purge logic (de Prado, 7.4.1): a train observation at position ``j`` overlaps
    the test set if ``[j, j + horizon]`` intersects ``[i, i + horizon]`` for some
    test position ``i``. Equivalently ``j`` is purged when it lies within
    ``horizon`` of any test position on either side. The embargo (7.4.3) removes
    a further ``embargo`` positions immediately *after* each test block.
    """
    test_set = {int(p) for p in test_positions}
    blocked = set(test_set)

    # Purge: any train position whose label window [j, j+horizon] overlaps a test
    # label window [i, i+horizon]. Overlap <=> |j - i| <= horizon for the closed
    # windows of equal length. We expand each test position by `horizon` on both
    # sides.
    for i in test_set:
        lo = max(0, i - horizon)
        hi = min(n_times - 1, i + horizon)
        for j in range(lo, hi + 1):
            blocked.add(j)

    # Embargo: remove `embargo` positions after each contiguous test block.
    if embargo > 0:
        for _start, end in _contiguous_blocks(np.array(sorted(test_set))):
            lo = end + 1
            hi = min(n_times - 1, end + embargo)
            for j in range(lo, hi + 1):
                blocked.add(j)

    train = np.array([p for p in range(n_times) if p not in blocked], dtype=np.int64)
    return train


def _purge_embargo_positions_t1(
    n_times: int,
    test_positions: np.ndarray,
    times: np.ndarray,
    t1: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """Return train positions after **label-driven** purge + embargo.

    Instead of a scalar ``horizon``, each observation carries an explicit label
    *end* time ``t1``: the label for the observation at time ``times[j]`` spans
    the closed interval ``[times[j], t1[j]]`` (de Prado, 7.4.1). A training
    observation is purged when its label interval overlaps **any** test
    observation's label interval, i.e.

    .. math:: t_j \\le t1_i \\quad\\text{and}\\quad t_i \\le t1_j

    for some test observation ``i`` (standard closed-interval intersection). The
    embargo is applied on the position axis exactly as in
    :func:`_purge_embargo_positions`.

    Parameters
    ----------
    n_times : int
        Number of unique time steps.
    test_positions : ndarray of int
        Positions (into the sorted unique-time index) belonging to the test set.
    times : ndarray
        The sorted unique time *values* (label start times), aligned to
        positions ``0..n_times-1``.
    t1 : ndarray
        Label *end* time values, aligned to the same positions. Must be
        comparable to ``times`` (same dtype / axis).
    embargo : int
        Embargo in time-steps applied after each contiguous test block.

    Returns
    -------
    ndarray of int
        Sorted training positions.
    """
    test_set = {int(p) for p in test_positions}
    blocked = set(test_set)

    test_arr = np.array(sorted(test_set), dtype=np.int64)
    ti = times[test_arr]  # test label start times
    ei = t1[test_arr]  # test label end times

    for j in range(n_times):
        if j in test_set:
            continue
        tj = times[j]
        ej = t1[j]
        # Overlap with any test interval [ti_k, ei_k]: tj <= ei_k and ti_k <= ej.
        if bool(np.any((tj <= ei) & (ti <= ej))):
            blocked.add(j)

    if embargo > 0:
        for _start, end in _contiguous_blocks(test_arr):
            lo = end + 1
            hi = min(n_times - 1, end + embargo)
            for j in range(lo, hi + 1):
                blocked.add(j)

    return np.array([p for p in range(n_times) if p not in blocked], dtype=np.int64)


def _calendar_embargo_positions(
    times: pl.Series,
    test_positions: np.ndarray,
    embargo: Duration,
) -> np.ndarray:
    """Positions embargoed by a **calendar** duration after each test block.

    For every contiguous block of test positions ending at time ``e``, every
    later position whose time is ``<= shift_forward(e, embargo)`` is blocked --
    the calendar analogue of "the next ``embargo`` positions". On an irregular
    axis the two differ: five business days after a Friday block end blocks
    through the following Friday whether the data has five rows in that span,
    three (a holiday and a gap), or ten (intraday bars).

    Parameters
    ----------
    times : polars.Series
        Sorted unique time values (``Date`` or ``Datetime``), aligned to
        positions ``0..n_times-1``.
    test_positions : ndarray of int
        Test positions into ``times``.
    embargo : Duration
        A calendar duration (not an ``int``); see
        :mod:`panelary.core._calendar`.

    Returns
    -------
    ndarray of int
        Sorted positions to remove from training (never a test position).
    """
    test_arr = np.array(sorted({int(p) for p in test_positions}), dtype=np.int64)
    blocks = _contiguous_blocks(test_arr)
    if not blocks:
        return np.array([], dtype=np.int64)
    ends = times.gather([end for _start, end in blocks])
    limits = shift_forward(ends, embargo).to_numpy()
    values = times.to_numpy()
    test_set = set(test_arr.tolist())
    blocked: set[int] = set()
    for (_start, end), limit in zip(blocks, limits, strict=True):
        hi = int(np.searchsorted(values, limit, side="right"))  # times[:hi] <= limit
        blocked.update(j for j in range(end + 1, hi) if j not in test_set)
    return np.array(sorted(blocked), dtype=np.int64)


def _train_positions(
    n_times: int,
    test_positions: np.ndarray,
    *,
    horizon: int | Duration,
    embargo: int | Duration,
    times: pl.Series | None = None,
    t1: np.ndarray | None = None,
) -> np.ndarray:
    """Train positions after purge + embargo, for integer **or** calendar specs.

    The single dispatcher every splitter goes through:

    * integer ``horizon`` and ``embargo`` with no ``t1`` -- exactly
      :func:`_purge_embargo_positions`, the historical path (byte-identical);
    * an explicit ``t1`` -- exactly :func:`_purge_embargo_positions_t1`;
    * a calendar ``horizon`` -- the label of the observation at ``tau`` spans
      ``[tau, shift_forward(tau, horizon)]``, so ``t1`` is derived from the
      time values and the ``t1`` path does the (closed-interval) purge;
    * a calendar ``embargo`` -- :func:`_calendar_embargo_positions` is removed
      from whatever the purge left.

    ``times`` (the sorted unique time values, as a Polars series so dtype and
    time zone survive) is required whenever a spec is calendar-valued.
    """
    int_embargo = as_steps(embargo) if is_steps(embargo) else 0
    calendar = not (is_steps(horizon) and is_steps(embargo))
    if (calendar or t1 is not None) and times is None:
        raise TypeError(
            "a calendar `horizon`/`embargo` (or a `t1`) is measured on the time "
            "values, so the time index is required; pass `times=` (the sorted "
            "unique times)."
        )
    if t1 is None and is_steps(horizon):
        train = _purge_embargo_positions(
            n_times, test_positions, as_steps(horizon), int_embargo
        )
    else:
        assert times is not None  # guaranteed by the check above
        if t1 is None:
            t1 = shift_forward(times, horizon).to_numpy()
        train = _purge_embargo_positions_t1(
            n_times, test_positions, times.to_numpy(), t1, int_embargo
        )
    if not is_steps(embargo):
        assert times is not None
        blocked = _calendar_embargo_positions(times, test_positions, embargo)
        if blocked.size:
            train = train[~np.isin(train, blocked)]
    return train


def _check_time_axis(times: pl.Series, *specs: tuple[str, object]) -> None:
    """Fail early, with the parameter name, if a calendar spec meets a non-temporal axis."""
    dtype = times.dtype
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return
    for name, spec in specs:
        if not is_steps(spec):
            raise TypeError(
                f"`{name}`={spec!r} is a calendar duration, but the panel's time "
                f"axis is {dtype}. Calendar embargo/purge needs a Date or Datetime "
                f"time column; on an integer axis pass `{name}` as an int number "
                "of time-steps."
            )


def _resolve_t1(t1: object, pf: PanelFrame, times: np.ndarray) -> np.ndarray | None:
    """Resolve a ``t1`` spec into an end-time array aligned to ``times``.

    ``t1`` may be:

    * ``None`` — return ``None`` (the scalar-``horizon`` path is used instead).
    * a column name (``str``) present in ``pf`` — the label end time per row; it
      is aggregated to the **max** end time per unique time (the most
      conservative purge for a shared-time-axis panel).
    * an array-like / :class:`polars.Series` of length ``len(times)`` — used
      directly as the per-time end times.
    """
    if t1 is None:
        return None
    n_times = times.shape[0]
    if isinstance(t1, str):
        agg = (
            pf.lazy()
            .group_by(pf.time_col)
            .agg(pl.col(t1).max().alias("__t1_end__"))
            .sort(pf.time_col)
            .collect()
        )
        arr = agg["__t1_end__"].to_numpy()
        if arr.shape[0] != n_times:
            raise ValueError(
                f"resolving `t1={t1!r}` produced {arr.shape[0]} end-times but "
                f"there are {n_times} unique times; every time must have a label "
                "end time."
            )
        return arr
    arr = t1.to_numpy() if isinstance(t1, pl.Series) else np.asarray(t1)
    if arr.shape[0] != n_times:
        raise ValueError(
            f"`t1` has length {arr.shape[0]} but there are {n_times} unique "
            "times; `t1` must align with the sorted unique-time index (pass a "
            "column name to derive it per-time instead)."
        )
    return arr


# --------------------------------------------------------------------------- #
# PurgedKFold
# --------------------------------------------------------------------------- #
class PurgedKFold:
    """K-fold cross-validation with purging and embargo for panels.

    Splits the **sorted unique time index** into ``n_splits`` contiguous folds.
    Each fold is used once as the test set; the corresponding training set is the
    remaining times with (a) any observation whose label window overlaps the test
    window **purged**, and (b) an **embargo** of ``embargo`` time-steps after the
    test block removed.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be >= 2.
    horizon : int or duration, default=0
        Label horizon used for purging. An ``int`` counts time-steps (``0``
        means each label is point-in-time: only the exact test times are
        purged from train). A calendar duration -- ``datetime.timedelta``,
        ``"5d"``, ``"1mo"``, :class:`BusinessDays` / ``"5bd"`` -- means the
        label of the observation at ``tau`` spans ``[tau, tau + horizon]`` on
        the actual time values. Ignored when ``t1`` is given.
    embargo : int or duration, default=0
        What to embargo from training after each test block. An ``int`` counts
        time-steps (positions), exactly as before. A calendar duration removes
        every later time ``<= block_end + embargo`` -- e.g. ``"5bd"`` or
        ``BusinessDays(5, holidays=...)`` embargoes five *business days*
        however many rows the data happens to have in that span. Calendar
        specs need a ``Date`` / ``Datetime`` time column.
    t1 : str or array-like, optional
        Per-time label end times (event-based purge); supersedes ``horizon``.
    return_indices : bool, default=False
        If True, :meth:`split` yields ``(train_positions, test_positions)`` as
        integer numpy arrays into the sorted unique time index. If False
        (default), it yields ``(train_panel, test_panel)`` as PanelFrames.

    Raises
    ------
    ValueError
        If ``n_splits < 2``, ``horizon < 0``, or ``embargo < 0`` (or a
        negative / malformed duration).
    TypeError
        If ``horizon`` / ``embargo`` is neither an ``int`` nor a duration, or
        (at :meth:`split`) a calendar duration meets a non-temporal time axis.

    References
    ----------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Chapter 7 ("Cross-Validation in Finance").

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=4, horizon=1, embargo=1)
    >>> for train, test in cv.split(panel):  # doctest: +SKIP
    ...     model.fit(train)
    ...     preds = model.predict(test)

    A calendar embargo on a daily panel -- five business days, skipping a
    holiday -- whatever the number of rows in that span:

    >>> cv = PurgedKFold(  # doctest: +SKIP
    ...     n_splits=5, horizon="1d", embargo=BusinessDays(5, holidays=["2024-12-25"])
    ... )
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        horizon: int | Duration = 0,
        embargo: int | Duration = 0,
        t1: object | None = None,
        return_indices: bool = False,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"`n_splits` must be >= 2, got {n_splits}.")
        # An int keeps the historical check and message; a duration is
        # validated (and "5bd" normalised to BusinessDays) once, here.
        horizon = validate_duration(horizon, name="horizon")
        embargo = validate_duration(embargo, name="embargo")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        # Optional per-time label end times (event-based purge). When given it
        # supersedes the scalar ``horizon`` (see :func:`_purge_embargo_positions_t1`).
        self.t1 = t1
        self.return_indices = return_indices

    def get_n_splits(self) -> int:
        """Return the number of folds (sklearn-compatible)."""
        return self.n_splits

    def _test_position_folds(self, n_times: int) -> list[np.ndarray]:
        """Partition ``range(n_times)`` into ``n_splits`` contiguous test folds."""
        if n_times < self.n_splits:
            raise ValueError(
                f"cannot make {self.n_splits} folds from only {n_times} unique "
                "time steps; reduce `n_splits` or supply more history."
            )
        return [
            np.array(part, dtype=np.int64)
            for part in np.array_split(
                np.arange(n_times, dtype=np.int64), self.n_splits
            )
        ]

    def split(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[PanelFold | IndexFold]:
        """Generate purged, embargoed train/test folds.

        Parameters
        ----------
        panel : PanelFrame | polars.DataFrame | polars.LazyFrame
            Long-format panel. Frames are coerced via
            :func:`~panelary.core.panel_frame.as_panel` (entity = col 0,
            time = col 1 by convention).

        Yields
        ------
        (train, test) : tuple
            PanelFrames, or integer position arrays if ``return_indices=True``.
        """
        pf = as_panel(panel)
        time_index = pf.time_index()
        times = time_index.to_numpy()
        n_times = times.shape[0]
        t1_arr = _resolve_t1(self.t1, pf, times)
        _check_time_axis(
            time_index, ("horizon", self.horizon), ("embargo", self.embargo)
        )
        for test_pos in self._test_position_folds(n_times):
            if t1_arr is None and is_steps(self.horizon) and is_steps(self.embargo):
                # The historical integer path, untouched.
                train_pos = _purge_embargo_positions(
                    n_times, test_pos, as_steps(self.horizon), as_steps(self.embargo)
                )
            elif is_steps(self.embargo) and t1_arr is not None:
                train_pos = _purge_embargo_positions_t1(
                    n_times, test_pos, times, t1_arr, as_steps(self.embargo)
                )
            else:
                train_pos = _train_positions(
                    n_times,
                    test_pos,
                    horizon=self.horizon,
                    embargo=self.embargo,
                    times=time_index,
                    t1=t1_arr,
                )
            if self.return_indices:
                yield train_pos, test_pos
            else:
                yield (
                    _subset_by_times(pf, times[train_pos]),
                    _subset_by_times(pf, times[test_pos]),
                )


# --------------------------------------------------------------------------- #
# Combinatorial Purged Cross-Validation
# --------------------------------------------------------------------------- #
@dataclass
class _CPCVFold:
    """A single CPCV split: which groups are test, plus position arrays."""

    test_groups: tuple[int, ...]
    train_positions: NDArray[np.int64]
    test_positions: NDArray[np.int64]


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV) for panels.

    The sorted unique time index is partitioned into ``n_groups`` contiguous
    groups. Every combination of ``n_test_groups`` groups is used as the test
    set (so there are ``C(n_groups, n_test_groups)`` splits), with the remaining
    groups as training — purged and embargoed as in :class:`PurgedKFold`.

    Because each group appears in many test combinations, CPCV yields a number of
    distinct **backtest paths** rather than a single one, giving a distribution
    of out-of-sample performance instead of a point estimate.

    Parameters
    ----------
    n_groups : int, default=6
        Number of contiguous time groups ``N``. Must be >= 2.
    n_test_groups : int, default=2
        Number of groups per test set ``k`` (``1 <= k < N``).
    horizon : int or duration, default=0
        Label horizon used for purging: time-steps (``int``) or a calendar
        duration, exactly as in :class:`PurgedKFold`.
    embargo : int or duration, default=0
        Embargo after each contiguous test block: time-steps (``int``) or a
        calendar duration (``"5d"``, ``"5bd"``, :class:`BusinessDays`,
        ``datetime.timedelta``), exactly as in :class:`PurgedKFold`.
    t1 : str or array-like, optional
        Per-time label end times (event-based purge); supersedes ``horizon``.
    return_indices : bool, default=False
        If True, yield integer position arrays; else PanelFrames.

    Attributes
    ----------
    n_splits : int
        ``C(n_groups, n_test_groups)``.
    n_paths : int
        Number of backtest paths, ``C(n_groups, n_test_groups) * n_test_groups
        / n_groups`` (de Prado, 12.4).

    Raises
    ------
    ValueError
        If parameters are out of range.

    References
    ----------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Chapter 12 ("Backtesting through Cross-Validation").
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        horizon: int | Duration = 0,
        embargo: int | Duration = 0,
        t1: object | None = None,
        return_indices: bool = False,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"`n_groups` must be >= 2, got {n_groups}.")
        if not (1 <= n_test_groups < n_groups):
            raise ValueError(
                f"`n_test_groups` must satisfy 1 <= k < n_groups "
                f"({n_groups}), got {n_test_groups}."
            )
        horizon = validate_duration(horizon, name="horizon")
        embargo = validate_duration(embargo, name="embargo")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.horizon = horizon
        self.embargo = embargo
        # Optional per-time label end times (event-based purge), overriding the
        # scalar ``horizon`` when supplied.
        self.t1 = t1
        self.return_indices = return_indices

    @property
    def n_splits(self) -> int:
        """Number of train/test combinations, ``C(n_groups, n_test_groups)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of distinct backtest paths reconstructable from the splits.

        Equals ``C(N, k) * k / N`` (each of the ``N`` groups is tested in
        ``C(N-1, k-1)`` splits, and the paths recombine non-overlapping test
        groups).
        """
        return self.n_splits * self.n_test_groups // self.n_groups

    def get_n_splits(self) -> int:
        """Return :attr:`n_splits` (sklearn-compatible)."""
        return self.n_splits

    def _group_positions(self, n_times: int) -> list[np.ndarray]:
        """Partition ``range(n_times)`` into ``n_groups`` contiguous groups."""
        if n_times < self.n_groups:
            raise ValueError(
                f"cannot make {self.n_groups} groups from only {n_times} unique "
                "time steps; reduce `n_groups` or supply more history."
            )
        return [
            np.array(part, dtype=np.int64)
            for part in np.array_split(
                np.arange(n_times, dtype=np.int64), self.n_groups
            )
        ]

    def _iter_folds(
        self,
        n_times: int,
        times: np.ndarray | None = None,
        t1_arr: np.ndarray | None = None,
        time_index: pl.Series | None = None,
    ) -> Iterator[_CPCVFold]:
        groups = self._group_positions(n_times)
        integer = is_steps(self.horizon) and is_steps(self.embargo)
        if not integer and time_index is None:
            raise TypeError(
                "this CombinatorialPurgedCV has a calendar `horizon`/`embargo`, "
                "which is measured on time values; the time index is required "
                "(split a panel, or pass `times=` to the positional helper)."
            )
        for test_combo in itertools.combinations(
            range(self.n_groups), self.n_test_groups
        ):
            test_pos = np.sort(np.concatenate([groups[g] for g in test_combo])).astype(
                np.int64
            )
            if t1_arr is None and integer:
                # The historical integer path, untouched.
                train_pos = _purge_embargo_positions(
                    n_times, test_pos, as_steps(self.horizon), as_steps(self.embargo)
                )
            elif t1_arr is not None and is_steps(self.embargo):
                assert times is not None  # a t1 is only resolved against times
                train_pos = _purge_embargo_positions_t1(
                    n_times, test_pos, times, t1_arr, as_steps(self.embargo)
                )
            else:
                train_pos = _train_positions(
                    n_times,
                    test_pos,
                    horizon=self.horizon,
                    embargo=self.embargo,
                    times=time_index,
                    t1=t1_arr,
                )
            yield _CPCVFold(
                test_groups=test_combo,
                train_positions=train_pos,
                test_positions=test_pos,
            )

    def split(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[PanelFold | IndexFold]:
        """Generate all CPCV train/test folds.

        Parameters
        ----------
        panel : PanelFrame | polars.DataFrame | polars.LazyFrame
            Long-format panel.

        Yields
        ------
        (train, test) : tuple
            PanelFrames, or integer position arrays if ``return_indices=True``.
            Folds are emitted in lexicographic order of the test-group
            combination; use :meth:`split_with_groups` to also obtain the test
            group ids for path reconstruction.
        """
        for train, test, _ in self.split_with_groups(panel):
            yield train, test

    def split_with_groups(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[
        tuple[
            PanelFrame | NDArray[np.int64],
            PanelFrame | NDArray[np.int64],
            tuple[int, ...],
        ]
    ]:
        """Like :meth:`split` but also yields the test-group tuple per fold.

        Yields
        ------
        (train, test, test_groups) : tuple
            ``test_groups`` is the tuple of group indices used as test, enabling
            reconstruction of the :attr:`n_paths` backtest paths.
        """
        pf = as_panel(panel)
        time_index = pf.time_index()
        times = time_index.to_numpy()
        n_times = times.shape[0]
        t1_arr = _resolve_t1(self.t1, pf, times)
        _check_time_axis(
            time_index, ("horizon", self.horizon), ("embargo", self.embargo)
        )
        for fold in self._iter_folds(n_times, times, t1_arr, time_index):
            if self.return_indices:
                yield fold.train_positions, fold.test_positions, fold.test_groups
            else:
                yield (
                    _subset_by_times(pf, times[fold.train_positions]),
                    _subset_by_times(pf, times[fold.test_positions]),
                    fold.test_groups,
                )

    def backtest_paths(self) -> list[list[tuple[int, int]]]:
        """Return the assignment of (split, group) to each backtest path.

        Each path is a list of ``(split_index, group_index)`` pairs covering all
        ``n_groups`` groups exactly once, in time order. Combine the test
        predictions of those (split, group) cells to obtain one full backtest
        path. There are :attr:`n_paths` such paths (de Prado, 12.4).

        Returns
        -------
        list of list of (int, int)
            ``paths[p][g]`` gives the ``(split_index, group_index)`` providing
            the test prediction for group ``g`` on path ``p``.
        """
        combos = list(itertools.combinations(range(self.n_groups), self.n_test_groups))
        # For each group, list the (split_index, position-within-combo) where it
        # is tested. Each group is tested in exactly C(N-1, k-1) splits == n_paths.
        per_group: list[list[int]] = [[] for _ in range(self.n_groups)]
        for s_idx, combo in enumerate(combos):
            for g in combo:
                per_group[g].append(s_idx)
        n_paths = self.n_paths
        paths: list[list[tuple[int, int]]] = []
        for p in range(n_paths):
            path: list[tuple[int, int]] = []
            for g in range(self.n_groups):
                split_idx = per_group[g][p]
                path.append((split_idx, g))
            paths.append(path)
        return paths


# --------------------------------------------------------------------------- #
# Walk-forward splitters (panel-aware re-exposure)
# --------------------------------------------------------------------------- #
def _window_panel_split(
    pf: PanelFrame,
    test_size: int,
    n_splits: int,
    step_size: int,
    window_size: int | None,
) -> list[PanelFold]:
    """Shared walk-forward split logic on the unique-time axis.

    The fold schedule is :func:`panelary.cross_validation._walk_forward_cutoffs`
    — the same kernel the row-based splitters in
    :mod:`panelary.cross_validation` use, so the two families place folds
    identically. The difference is the axis: this slices the **shared
    unique-time index** (so all entities are aligned on the same train/test
    dates) and returns PanelFrames, whereas ``_window_split`` slices each
    entity's own rows inside ``group_by(entity)``.
    """
    times = _unique_times(pf)
    n_times = times.shape[0]

    cutoffs = _walk_forward_cutoffs(test_size, n_splits, step_size)

    folds: list[PanelFold] = []
    for i in range(n_splits):
        cutoff = cutoffs[i]
        test_start = n_times - cutoff
        test_end = test_start + test_size
        if test_start < 0:
            raise ValueError(
                f"split {i}: not enough history ({n_times} time steps) for "
                f"test_size={test_size}, n_splits={n_splits}, step_size={step_size}"
                + (f", window_size={window_size}" if window_size else "")
                + ". Reduce these parameters or supply more data."
            )
        test_pos = np.arange(test_start, min(test_end, n_times), dtype=np.int64)

        if window_size is not None:
            train_start = max(0, test_start - window_size)
            train_pos = np.arange(train_start, test_start, dtype=np.int64)
        else:
            train_pos = np.arange(0, test_start, dtype=np.int64)

        folds.append(
            (
                _subset_by_times(pf, times[train_pos]),
                _subset_by_times(pf, times[test_pos]),
            )
        )
    return folds


def expanding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
):
    """Return a panel-aware expanding-window walk-forward splitter.

    Each split grows the training window from the start of history up to the test
    block; the test block is a fixed-size window that slides forward by
    ``step_size`` across splits. Splitting happens on the **shared unique-time
    index**, so all entities are aligned on the same train/test dates.

    For ``test_size=3, n_splits=5, step_size=1`` the folds are::

        | o o o x x x - - - - |
        | o o o o x x x - - - |
        | o o o o o x x x - - |
        | o o o o o o x x x - |
        | o o o o o o o x x x |

    Parameters
    ----------
    test_size : int
        Number of *time steps* per test block.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Forward step (in time steps) between consecutive test blocks.

    Returns
    -------
    callable
        ``split(panel) -> list[(train_panel, test_panel)]`` where ``panel`` is a
        :class:`PanelFrame` or a coercible frame.

    See Also
    --------
    panelary.cross_validation.expanding_window_split : the row-based functime
        original. This function does **not** wrap it — it is a second splitter
        with the same fold schedule (both call
        :func:`panelary.cross_validation._walk_forward_cutoffs`) and a different
        axis. The original slices each entity's own rows and returns
        ``(train, test)`` LazyFrames, so on a ragged panel entities are tested on
        different dates; this one slices the shared unique-time index and returns
        ``PanelFrame`` folds, so they are tested on the same dates.
    """

    def split(panel: PanelFrame | pl.DataFrame | pl.LazyFrame) -> list[PanelFold]:
        pf = as_panel(panel)
        return _window_panel_split(pf, test_size, n_splits, step_size, None)

    return split


def sliding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
    window_size: int = 10,
):
    """Return a panel-aware sliding-window walk-forward splitter.

    Like :func:`expanding_window_split` but the training window is a fixed length
    of ``window_size`` time steps that slides forward with the test block.

    For ``test_size=3, n_splits=5, step_size=1, window_size=5`` the folds are::

        | o o o o o x x x - - - - |
        | - o o o o o x x x - - - |
        | - - o o o o o x x x - - |
        | - - - o o o o o x x x - |
        | - - - - o o o o o x x x |

    Parameters
    ----------
    test_size : int
        Number of time steps per test block.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Forward step (in time steps) between consecutive test blocks.
    window_size : int, default=10
        Number of time steps in each (fixed-length) training window.

    Returns
    -------
    callable
        ``split(panel) -> list[(train_panel, test_panel)]``.

    Notes
    -----
    Splits with less than ``window_size`` of history get a training window
    truncated at the start of the index, never one that reaches past the test
    block.

    See Also
    --------
    panelary.cross_validation.sliding_window_split : the row-based functime
        original. Same fold schedule (both call
        :func:`panelary.cross_validation._walk_forward_cutoffs`), different
        axis and different return type; this function does not wrap it.
    """

    def split(panel: PanelFrame | pl.DataFrame | pl.LazyFrame) -> list[PanelFold]:
        pf = as_panel(panel)
        return _window_panel_split(pf, test_size, n_splits, step_size, window_size)

    return split


# --------------------------------------------------------------------------- #
# de Prado performance / overfitting metrics
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over the open interval (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"`p` must be in the open interval (0, 1), got {p}.")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    sharpe_std: float | None = None,
    sharpe_variance_across_trials: float | None = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (DSR): probability the strategy is truly skilled.

    Implements Bailey & Lopez de Prado (2014). The DSR is the probability that
    the observed (non-annualised) Sharpe ratio exceeds an *expected maximum
    Sharpe* due to multiple testing, while accounting for the non-normality
    (skewness, kurtosis) and length of the return series.

    Parameters
    ----------
    observed_sharpe : float
        The Sharpe ratio actually observed for the selected strategy, expressed
        per observation (same frequency as ``n_observations``; do **not**
        annualise).
    n_trials : int
        Number ``N`` of independent strategy configurations tried (the multiple
        testing count).
    n_observations : int
        Number ``T`` of return observations the Sharpe was estimated from.
    sharpe_std : float, optional
        Standard deviation of the Sharpe estimator. If omitted it is computed
        from ``n_observations``, ``skewness`` and ``kurtosis`` via the standard
        formula (Bailey & de Prado eq. for ``sigma(SR_hat)``).
    sharpe_variance_across_trials : float, optional
        Variance of the Sharpe ratios *across the N trials*. Used to estimate the
        expected maximum Sharpe under the null. If omitted, defaults to
        ``sharpe_std ** 2`` (a conservative fallback assuming trials are as noisy
        as the estimator).
    skewness : float, default=0.0
        Skewness of the return series.
    kurtosis : float, default=3.0
        Kurtosis of the return series (3.0 == normal).
    benchmark_sharpe : float, optional
        The expected maximum Sharpe under the null. If omitted it is computed
        from ``n_trials`` and ``sharpe_variance_across_trials`` using the
        expected-maximum-of-Gaussians approximation (eq. for ``SR_0``).

    Returns
    -------
    float
        The DSR in ``[0, 1]``. Values near 1 indicate the strategy is unlikely to
        be a false positive of the selection process.

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    *Journal of Portfolio Management*, 40(5).

    Notes
    -----
    The estimator standard deviation is

    .. math::

        \\hat\\sigma(\\widehat{SR}) = \\sqrt{\\frac{1 - \\gamma_3\\,SR
        + \\frac{\\gamma_4 - 1}{4} SR^2}{T - 1}}

    where :math:`\\gamma_3` is skewness and :math:`\\gamma_4` is kurtosis. The
    expected maximum Sharpe across :math:`N` trials uses

    .. math::

        SR_0 = \\sqrt{V}\\left[(1-\\gamma)\\,Z^{-1}\\!\\left(1-\\tfrac1N\\right)
        + \\gamma\\,Z^{-1}\\!\\left(1-\\tfrac1N e^{-1}\\right)\\right]

    with :math:`V` the variance of Sharpes across trials and
    :math:`\\gamma\\approx 0.5772` the Euler–Mascheroni constant.
    """
    if n_observations < 2:
        raise ValueError("`n_observations` must be >= 2.")
    if n_trials < 1:
        raise ValueError("`n_trials` must be >= 1.")

    sr = float(observed_sharpe)

    if sharpe_std is None:
        var = (1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr) / (
            n_observations - 1
        )
        if var <= 0:
            raise ValueError(
                "computed Sharpe-estimator variance is non-positive; check "
                "`skewness`/`kurtosis`/`observed_sharpe` inputs."
            )
        sharpe_std = math.sqrt(var)

    if benchmark_sharpe is None:
        v = (
            sharpe_variance_across_trials
            if sharpe_variance_across_trials is not None
            else sharpe_std * sharpe_std
        )
        if v < 0:
            raise ValueError("`sharpe_variance_across_trials` must be >= 0.")
        gamma = 0.5772156649015329  # Euler-Mascheroni
        if n_trials == 1:
            benchmark_sharpe = 0.0
        else:
            z1 = _norm_ppf(1.0 - 1.0 / n_trials)
            z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
            benchmark_sharpe = math.sqrt(v) * ((1.0 - gamma) * z1 + gamma * z2)

    dsr = _norm_cdf((sr - benchmark_sharpe) / sharpe_std)
    return float(dsr)


def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    n_partitions: int = 16,
    higher_is_better: bool = True,
    statistic: str | Callable[[np.ndarray], np.ndarray] = "mean",
) -> float:
    """Probability of Backtest Overfitting (PBO) via CSCV.

    Implements the Combinatorially Symmetric Cross-Validation estimator of
    Bailey, Borwein, Lopez de Prado & Zhu (2017). Given a matrix ``M`` of
    out-of-sample-style performance for ``S`` strategies over ``T`` time slices,
    PBO estimates the probability that the strategy selected as best in-sample
    underperforms the median strategy out-of-sample.

    Parameters
    ----------
    performance_matrix : ndarray of shape (T, S) or nested sequence
        ``T`` rows of per-period performance (e.g. returns) for ``S`` strategy
        configurations. Each column is one strategy's track record.
    n_partitions : int, default=16
        Even number ``S_partitions`` of contiguous, equal-size sub-matrices the
        rows are split into; CSCV evaluates every way of choosing half as the
        in-sample set (``C(n_partitions, n_partitions/2)`` combinations). Must be
        even and >= 2. Reduce if ``T`` is small.
    higher_is_better : bool, default=True
        Whether larger performance values are better (e.g. Sharpe, returns). If
        False (e.g. loss), the sign of the selection is flipped.
    statistic : {"mean", "sharpe"} or callable, default="mean"
        Column-wise performance statistic used to rank strategies within each IS
        / OS slice. ``"mean"`` reproduces the block-mean behaviour; ``"sharpe"``
        ranks by per-observation Sharpe (recommended for return matrices, since
        two strategies with equal means but different vols are then separated). A
        callable receives the ``(T_slice, S)`` slice and must return an ``(S,)``
        vector.

    Returns
    -------
    float
        PBO in ``[0, 1]``. Values near 0 indicate the in-sample-best strategy
        tends to remain good out-of-sample (low overfitting); values near or
        above 0.5 indicate selection is no better than chance.

    Raises
    ------
    ValueError
        If the matrix is degenerate (fewer than 2 strategies), ``n_partitions``
        is not a valid even number, or ``T`` is too small to partition.

    References
    ----------
    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2017). "The
    Probability of Backtest Overfitting." *Journal of Computational Finance*,
    20(4).

    Notes
    -----
    Algorithm (CSCV):

    1. Split the ``T`` rows into ``S`` (``=n_partitions``) disjoint, equal
       sub-matrices.
    2. For each combination ``c`` of ``S/2`` sub-matrices forming the in-sample
       (IS) set, the complement is out-of-sample (OS).
    3. Find ``n*``, the strategy best on IS. Compute its OS rank as a fraction
       ``omega = rank / (n_strategies + 1)`` and the logit
       ``lambda = ln(omega / (1 - omega))``.
    4. PBO is the fraction of combinations with ``lambda <= 0`` (i.e. the IS-best
       strategy lands in the bottom half OS).
    """
    M = np.asarray(performance_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(
            f"`performance_matrix` must be 2-D (T x S), got shape {M.shape}."
        )
    T, S = M.shape
    if S < 2:
        raise ValueError(f"need at least 2 strategies (columns), got {S}.")
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError(
            f"`n_partitions` must be an even integer >= 2, got {n_partitions}."
        )
    if n_partitions > T:
        raise ValueError(
            f"need at least n_partitions={n_partitions} rows, got T={T}. "
            "Reduce `n_partitions` or supply a longer track record."
        )
    if not higher_is_better:
        M = -M

    # Trim rows so they divide evenly into n_partitions contiguous blocks.
    block = T // n_partitions
    block * n_partitions
    blocks = [M[i * block : (i + 1) * block, :] for i in range(n_partitions)]

    half = n_partitions // 2
    logits: list[float] = []
    for is_combo in itertools.combinations(range(n_partitions), half):
        is_set = set(is_combo)
        is_rows = np.vstack([blocks[i] for i in range(n_partitions) if i in is_set])
        os_rows = np.vstack([blocks[i] for i in range(n_partitions) if i not in is_set])

        # Performance per strategy from the chosen statistic (mean by default,
        # Sharpe recommended for return matrices).
        is_perf = _perf_stat(is_rows, statistic)
        os_perf = _perf_stat(os_rows, statistic)

        n_star = int(np.nanargmax(is_perf))
        # Rank of the IS-best strategy OS (1 = worst .. S = best).
        order = np.argsort(np.argsort(os_perf))  # 0-based ranks
        rank = int(order[n_star]) + 1
        omega = rank / (S + 1)
        omega = min(max(omega, 1e-12), 1 - 1e-12)
        logits.append(math.log(omega / (1.0 - omega)))

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0))
    return pbo


# --------------------------------------------------------------------------- #
# Cross-validation runner (CPCV / PurgedKFold driver)
# --------------------------------------------------------------------------- #
def _neg_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default fold metric: negative MSE (higher is better)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return -float(np.mean((yt - yp) ** 2))


@dataclass
class CVReport:
    """Result of :func:`cross_validate`: per-fold and (for CPCV) per-path scores.

    Attributes
    ----------
    metric_name : str
        Name of the scoring function used for :attr:`fold_scores`.
    fold_scores : list of float
        One score per CV split, in split order.
    fold_test_groups : list
        For CPCV, the tuple of test-group indices per split; ``None`` for
        non-combinatorial splitters.
    n_splits : int
        Number of splits actually evaluated.
    n_paths : int, optional
        Number of reconstructed backtest paths (CPCV only).
    path_scores : list of float, optional
        Per-path aggregate performance (mean per-period strategy return).
    path_sharpes : list of float, optional
        Per-path Sharpe ratio of the reconstructed return series.
    performance_matrix : numpy.ndarray, optional
        ``(n_periods, n_paths)`` matrix of per-period strategy returns, one
        column per backtest path — the input to
        :func:`probability_of_backtest_overfitting`.
    deflated_sharpe : float, optional
        :func:`deflated_sharpe_ratio` of the best path. Deflated against the
        honest number of configurations searched (``n_trials``, supplied by the
        caller) and the *empirical* variance of the path Sharpes
        (``sharpe_variance``). When ``n_trials`` is not supplied the CPCV
        ``n_paths`` is used only as a (usually too-small) floor and a warning is
        recorded in :attr:`extra`.
    n_trials : int, optional
        The multiple-testing count actually used to deflate
        :attr:`deflated_sharpe` (the user's ``n_trials``, or the ``n_paths``
        floor when omitted).
    sharpe_variance : float, optional
        Empirical variance of the path Sharpes (``var(path_sharpes, ddof=1)``)
        passed to :func:`deflated_sharpe_ratio` as ``V``.
    pbo : float, optional
        :func:`probability_of_backtest_overfitting` across the paths.
    """

    metric_name: str
    fold_scores: list[float]
    fold_test_groups: list[tuple[int, ...] | None]
    n_splits: int
    n_paths: int | None = None
    path_scores: list[float] | None = None
    path_sharpes: list[float] | None = None
    performance_matrix: NDArray[np.float64] | None = None
    deflated_sharpe: float | None = None
    n_trials: int | None = None
    sharpe_variance: float | None = None
    pbo: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a compact dict of aggregate CV statistics."""
        fs = np.asarray(self.fold_scores, dtype=float)
        out: dict[str, Any] = {
            "metric": self.metric_name,
            "n_splits": self.n_splits,
            "mean_score": float(np.nanmean(fs)) if fs.size else float("nan"),
            "std_score": float(np.nanstd(fs)) if fs.size else float("nan"),
            "min_score": float(np.nanmin(fs)) if fs.size else float("nan"),
            "max_score": float(np.nanmax(fs)) if fs.size else float("nan"),
        }
        if self.n_paths is not None:
            ps = np.asarray(self.path_sharpes or [], dtype=float)
            out.update(
                {
                    "n_paths": self.n_paths,
                    "mean_path_sharpe": (
                        float(np.nanmean(ps)) if ps.size else float("nan")
                    ),
                    "deflated_sharpe": self.deflated_sharpe,
                    "n_trials": self.n_trials,
                    "V": self.sharpe_variance,
                    "pbo": self.pbo,
                }
            )
        return out


def _prepare_panel(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    y: object,
    entity: str | None,
    time: str | None,
) -> tuple[PanelFrame, str | None]:
    """Coerce ``X`` to a sorted PanelFrame and resolve the target column.

    Returns the panel and the name of the target column (or ``None`` if ``y`` is
    ``None``). Array-like ``y`` is attached as a ``__target__`` column, aligned
    to the panel's sorted ``(entity, time)`` row order.
    """
    pf = X if isinstance(X, PanelFrame) else as_panel(X, entity, time)
    pf = pf.sort_panel()
    entity_col, time_col = pf.entity_col, pf.time_col

    if y is None:
        return pf, None
    if isinstance(y, str):
        return pf, y

    base = pf.collect()
    yv = y.to_numpy() if isinstance(y, pl.Series) else np.asarray(y)
    if yv.shape[0] != base.height:
        raise ValueError(
            f"`y` has length {yv.shape[0]} but the panel has {base.height} rows; "
            "pass a target column name or an array aligned to the panel rows."
        )
    base = base.with_columns(pl.Series("__target__", yv))
    return PanelFrame(
        base, entity=entity_col, time=time_col, validate=False
    ), "__target__"


def _fit_predict_fold(
    estimator: Any,
    train: PanelFrame,
    test: PanelFrame,
    *,
    feature_cols: list[str],
    target_col: str | None,
    entity_col: str,
    time_col: str,
    is_panel: bool,
) -> tuple[np.ndarray, pl.DataFrame, np.ndarray | None]:
    """Fit ``estimator`` on ``train`` and predict ``test``.

    Returns ``(y_pred, pred_frame, y_true)`` where ``pred_frame`` carries
    ``entity``, ``time``, ``__pred__`` and (if available) ``__target__``.
    """
    test_df = test.collect()
    if is_panel:
        estimator.fit(train)
        pred_df = estimator.predict(test).collect()
        candidates = [c for c in pred_df.columns if c not in (entity_col, time_col)]
        pred_col = next(
            (c for c in candidates if "pred" in c.lower()),
            None,
        )
        if pred_col is None:
            input_cols = set(test.columns)
            extra = [c for c in candidates if c not in input_cols]
            pred_col = extra[0] if extra else candidates[-1]
        y_pred = pred_df.get_column(pred_col).to_numpy().ravel()
        pred_frame = pred_df.select(entity_col, time_col).with_columns(
            pl.Series("__pred__", y_pred)
        )
    else:
        x_train = train.collect().select(feature_cols).to_numpy()
        x_test = test_df.select(feature_cols).to_numpy()
        if target_col is not None:
            y_train = train.collect().get_column(target_col).to_numpy().ravel()
        else:
            y_train = None
        estimator.fit(x_train, y_train)
        y_pred = np.asarray(estimator.predict(x_test)).ravel()
        pred_frame = test_df.select(entity_col, time_col).with_columns(
            pl.Series("__pred__", y_pred)
        )

    if target_col is not None:
        y_true = test_df.get_column(target_col).to_numpy().ravel()
        pred_frame = pred_frame.with_columns(pl.Series("__target__", y_true))
    else:
        y_true = None
    return y_pred, pred_frame, y_true


def _reconstruct_paths(
    report: CVReport,
    cv: CombinatorialPurgedCV,
    pf: PanelFrame,
    fold_preds: list[pl.DataFrame],
    entity_col: str,
    time_col: str,
    *,
    n_trials: int | None = None,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: float | None = None,
) -> None:
    """Reconstruct CPCV backtest paths and fill overfitting diagnostics.

    For each of the :attr:`CombinatorialPurgedCV.n_paths` paths, the per-group
    test predictions (one split per group) are stitched into a full-coverage
    out-of-sample series; a per-period strategy return ``mean_entities(pred *
    target)`` is formed and fed into :func:`deflated_sharpe_ratio` and
    :func:`probability_of_backtest_overfitting`.
    """
    times = _unique_times(pf)
    n_times = times.shape[0]
    group_positions = cv._group_positions(n_times)
    group_times = {
        g: pl.Series(values=times[pos].tolist()).implode()
        for g, pos in enumerate(group_positions)
    }

    paths = cv.backtest_paths()
    perf_cols: list[np.ndarray] = []
    path_sharpes: list[float] = []
    path_scores: list[float] = []
    for path in paths:
        frames: list[pl.DataFrame] = []
        for split_idx, g in path:
            frame = fold_preds[split_idx]
            frames.append(frame.filter(pl.col(time_col).is_in(group_times[g])))
        full = pl.concat(frames)
        rets = (
            full.with_columns(
                (pl.col("__pred__") * pl.col("__target__")).alias("__ret__")
            )
            .group_by(time_col)
            .agg(pl.col("__ret__").mean())
            .sort(time_col)
        )
        r = rets.get_column("__ret__").to_numpy().astype(float)
        perf_cols.append(r)
        path_scores.append(float(np.nanmean(r)) if r.size else float("nan"))
        path_sharpes.append(_sharpe(r, periods_per_year=periods_per_year))

    report.n_paths = cv.n_paths
    report.path_scores = path_scores
    report.path_sharpes = path_sharpes

    lengths = {c.shape[0] for c in perf_cols}
    matrix = np.column_stack(perf_cols) if len(lengths) == 1 else None
    report.performance_matrix = matrix

    finite = [s for s in path_sharpes if math.isfinite(s)]
    if finite:
        # (N, V) must be a matched pair: N is the number of configurations
        # searched and V the variance of *those* configurations' Sharpes.
        # When the caller supplies `trial_sharpes` (the Sharpe of every
        # configuration they tried) both come from it, which is the estimator
        # Bailey & de Prado actually define. Otherwise V falls back to the
        # variance across CPCV paths of the single fitted strategy -- a proxy
        # for trial dispersion, not the same quantity -- and N to the user's
        # honest search count.
        trial_sample = (
            [float(s) for s in trial_sharpes if math.isfinite(float(s))]
            if trial_sharpes is not None
            else None
        )
        if trial_sample and len(trial_sample) > 1:
            v_trials = float(np.var(trial_sample, ddof=1))
        else:
            v_trials = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
        # n_paths is a geometry constant, never the search count. Use the honest
        # user N; fall back to n_paths only as a floor, and say so loudly.
        if n_trials is not None:
            eff_trials = n_trials
        elif trial_sample:
            eff_trials = len(trial_sample)
        else:
            eff_trials = cv.n_paths
        if n_trials is None and not trial_sample:
            msg = (
                "n_trials not supplied; DSR deflated against n_paths="
                f"{cv.n_paths} (a CV-geometry floor, likely too small). Pass "
                "cross_validate(..., n_trials=<configs you searched>) for an "
                "honest Deflated Sharpe."
            )
            report.extra["dsr_n_trials_warning"] = msg
            warnings.warn(msg, stacklevel=2)
        report.n_trials = int(eff_trials)
        report.sharpe_variance = v_trials
        # For DSR the observed SR must be per-observation; strip annualisation.
        best = max(finite)
        if periods_per_year is not None:
            best = best / math.sqrt(periods_per_year)
        try:
            report.deflated_sharpe = deflated_sharpe_ratio(
                best,
                n_trials=max(int(eff_trials), 1),
                n_observations=max(n_times, 2),
                sharpe_variance_across_trials=v_trials if v_trials > 0 else None,
            )
        except (ValueError, ZeroDivisionError):
            report.deflated_sharpe = None

    if matrix is not None and matrix.shape[1] >= 2:
        n_part = min(10, matrix.shape[0])
        if n_part % 2 == 1:
            n_part -= 1
        if n_part >= 2:
            try:
                report.pbo = probability_of_backtest_overfitting(
                    matrix, n_partitions=n_part, statistic="sharpe"
                )
            except ValueError:
                report.pbo = None


def cross_validate(
    estimator: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    y: object = None,
    cv: Any = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_trials: int | None = None,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: float | None = None,
) -> CVReport:
    """Run leak-safe cross-validation of ``estimator`` over ``cv``.

    Fits ``estimator`` on each training fold and predicts the test fold. With a
    :class:`CombinatorialPurgedCV`, the per-split test predictions are recombined
    into the :attr:`~CombinatorialPurgedCV.n_paths` backtest paths (via
    :meth:`~CombinatorialPurgedCV.backtest_paths`) and fed into the
    Deflated-Sharpe / PBO overfitting diagnostics.

    Parameters
    ----------
    estimator : object
        Either a Panelary estimator/:class:`~panelary.core.pipeline.Pipeline`
        (``fit(panel)`` / ``predict(panel) -> PanelFrame``) or any sklearn-shaped
        object with ``fit(X, y)`` / ``predict(X)`` over numpy arrays. The object
        is deep-copied per fold so folds are independent.
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        The feature panel (may also carry the target column).
    y : str | polars.Series | array-like, optional
        The target: a column name in ``X``, or an array aligned to ``X`` rows.
        Required for scoring and for CPCV path reconstruction.
    cv : splitter
        A :class:`PurgedKFold` or :class:`CombinatorialPurgedCV` (or any object
        with a compatible ``split``). Must have ``return_indices=False``.
    entity, time : str, optional
        Panel keys when ``X`` is a bare frame.
    metric : callable, optional
        ``metric(y_true, y_pred) -> float`` for the per-fold score. Defaults to
        negative mean squared error (higher is better).
    n_trials : int, optional
        For CPCV only: the honest number of strategy configurations you searched
        (hyper-parameter grid, feature sets, …). This is the multiple-testing
        count ``N`` used to deflate the Sharpe. If omitted, the CPCV ``n_paths``
        (a CV-geometry constant, **not** a search count) is used only as a floor
        and a warning is emitted — pass ``n_trials`` for an honest DSR.
    trial_sharpes : sequence of float, optional
        For CPCV only: the per-observation Sharpe of **every configuration you
        searched**. This is the sample Bailey & de Prado's ``V`` is defined on,
        so supplying it makes the ``(N, V)`` pair used by the Deflated Sharpe
        mutually consistent (``N`` defaults to ``len(trial_sharpes)`` when
        ``n_trials`` is omitted). Without it, ``V`` falls back to the variance
        of this strategy's CPCV path Sharpes, which is a proxy for trial
        dispersion rather than the quantity itself.
    periods_per_year : float, optional
        For CPCV only: annualisation factor for the reported path Sharpes. The
        DSR itself always strips annualisation (it needs the per-observation
        Sharpe), so this only affects the annualised path Sharpe values.

    Returns
    -------
    CVReport
    """
    if cv is None:
        raise ValueError(
            "`cv` is required (e.g. a PurgedKFold or CombinatorialPurgedCV)."
        )
    if getattr(cv, "return_indices", False):
        raise ValueError(
            "cross_validate needs a splitter that yields PanelFrames; construct "
            "`cv` with `return_indices=False`."
        )

    pf, target_col = _prepare_panel(X, y, entity, time)
    entity_col, time_col = pf.entity_col, pf.time_col
    feature_cols = [c for c in pf.feature_cols if c != target_col]
    is_panel = isinstance(estimator, PanelTransformer)

    if metric is None:
        metric = _neg_mean_squared_error
        metric_name = "neg_mean_squared_error"
    else:
        metric_name = getattr(metric, "__name__", "metric")

    is_cpcv = hasattr(cv, "backtest_paths") and hasattr(cv, "split_with_groups")
    if is_cpcv:
        fold_iter: Iterator[tuple[Any, Any, Any]] = cv.split_with_groups(pf)
    else:
        fold_iter = ((tr, te, None) for tr, te in cv.split(pf))

    fold_scores: list[float] = []
    fold_groups: list[tuple[int, ...] | None] = []
    fold_preds: list[pl.DataFrame] = []
    for train, test, groups in fold_iter:
        est = copy.deepcopy(estimator)
        y_pred, pred_frame, y_true = _fit_predict_fold(
            est,
            train,
            test,
            feature_cols=feature_cols,
            target_col=target_col,
            entity_col=entity_col,
            time_col=time_col,
            is_panel=is_panel,
        )
        if y_true is not None:
            fold_scores.append(float(metric(y_true, y_pred)))
        else:
            fold_scores.append(float("nan"))
        fold_groups.append(groups)
        fold_preds.append(pred_frame)

    report = CVReport(
        metric_name=metric_name,
        fold_scores=fold_scores,
        fold_test_groups=fold_groups,
        n_splits=len(fold_scores),
    )
    if is_cpcv and target_col is not None:
        _reconstruct_paths(
            report,
            cv,
            pf,
            fold_preds,
            entity_col,
            time_col,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            periods_per_year=periods_per_year,
        )
    return report


class _Validate:
    """Convenience namespace for common leak-safe validation recipes."""

    @staticmethod
    def cpcv(
        estimator: Any,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        y: object = None,
        *,
        n_groups: int = 6,
        n_test_groups: int = 2,
        horizon: int | Duration = 0,
        embargo: int | Duration = 0,
        t1: object | None = None,
        entity: str | None = None,
        time: str | None = None,
        metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
        n_trials: int | None = None,
        trial_sharpes: Sequence[float] | None = None,
        periods_per_year: float | None = None,
    ) -> CVReport:
        """Combinatorial Purged CV of ``estimator`` -> :class:`CVReport`."""
        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            horizon=horizon,
            embargo=embargo,
            t1=t1,
        )
        return cross_validate(
            estimator,
            X,
            y,
            cv,
            entity=entity,
            time=time,
            metric=metric,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            periods_per_year=periods_per_year,
        )

    @staticmethod
    def purged_kfold(
        estimator: Any,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        y: object = None,
        *,
        n_splits: int = 5,
        horizon: int | Duration = 0,
        embargo: int | Duration = 0,
        t1: object | None = None,
        entity: str | None = None,
        time: str | None = None,
        metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> CVReport:
        """Purged K-Fold CV of ``estimator`` -> :class:`CVReport`."""
        cv = PurgedKFold(n_splits=n_splits, horizon=horizon, embargo=embargo, t1=t1)
        return cross_validate(
            estimator, X, y, cv, entity=entity, time=time, metric=metric
        )


validate = _Validate()
