"""The as-of engine: a (T, N) matrix, the as-of universe, one window per date.

Data path (plan section 5.1):

1. :func:`panel_matrix` pivots the long panel with
   :func:`panelary.shape._tensor.build_tensor` -- the package's one long-to-dense
   boundary -- with ``forward_fill=False`` (a delisted entity's last return
   must not repeat forever, trap T5) and ``z_normalize=False``, into a
   time-major C-contiguous float64 ``R`` of shape ``(T, N)``. NaN means "not
   observed".
2. Integer prefix counts of observed cells, so the observation count of any
   trailing window is one subtraction -- exact, no float drift.
3. The as-of universe at ``t``: entities observed **at** ``t`` with at least
   ``ceil(min_coverage * W)`` observations in ``(t - W, t]``. Nothing about an
   entity's future is consulted, so an entity listed after ``t`` cannot enter
   an earlier window and dropping it changes nothing at ``t``.
4. The window is **compacted** before any matrix product: ``R[t-W+1:t+1,
   idx]`` is copied to a fresh contiguous ``(W, N_t)`` array. Masking absent
   entities with zeros instead would change the GEMM's blocking and summation
   order when an entity that appears after ``t`` widens the matrix, and
   bitwise prefix invariance would be lost.

:func:`rolling` exposes the engine as a lazy :class:`CovarianceSeries`:
``series.at(date)`` computes (and caches) the estimate from the last
scheduled date ``<= date``.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.core._schedule import Schedule, as_schedule
from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.covariance._estimate import METHODS, resolve_method
from panelary.covariance._gram import SPACES, window_stats
from panelary.covariance._missing import MISSING_POLICIES
from panelary.covariance._types import CovEstimate, WindowStats

__all__ = [
    "CovarianceSeries",
    "PanelMatrix",
    "panel_matrix",
    "rolling",
    "window_at",
]

#: The smallest universe for which a window kernel is run.
DEFAULT_MIN_ENTITIES = 2


def _as_pf(panel: Any, entity: str | None, time: str | None) -> PanelFrame:
    if isinstance(panel, PanelFrame):
        return panel
    return as_panel(panel, entity=entity, time=time)


@dataclass(eq=False)
class PanelMatrix:
    """A time-major ``(T, N)`` view of one panel column, plus prefix counts.

    Attributes
    ----------
    R : ndarray, shape (T, N)
        Float64, C-contiguous, NaN where unobserved.
    times, entities : polars.Series
        The sorted unique time axis and entity ids.
    entity_col, time_col : str
    finite : ndarray of bool, shape (T, N)
    counts : ndarray of int32, shape (T + 1, N)
        ``counts[t + 1] - counts[t + 1 - W]`` is the number of observations
        in ``(t - W, t]``.
    groups : ndarray of int64, shape (T, N), optional
        Dense group codes (``-1`` where unobserved or null) when a group
        column was requested; ``group_values`` maps code -> label.
    """

    R: NDArray[np.float64]
    times: pl.Series
    entities: pl.Series
    entity_col: str
    time_col: str
    finite: NDArray[np.bool_]
    counts: NDArray[np.int32]
    groups: NDArray[np.int64] | None = None
    group_values: list[Any] = field(default_factory=list)
    group_col: str | None = None
    _entity_list: list[Any] | None = field(default=None, repr=False)

    @property
    def n_times(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_entities(self) -> int:
        return int(self.R.shape[1])

    def entity_labels(self, idx: NDArray[np.intp]) -> tuple[Any, ...]:
        if self._entity_list is None:
            self._entity_list = self.entities.to_list()
        lst = self._entity_list
        return tuple(lst[int(i)] for i in idx)

    def window_count(self, t: int, window: int) -> NDArray[np.int32]:
        """Observations per entity in ``(t - window, t]`` (clipped at 0)."""
        lo = max(0, t + 1 - window)
        return self.counts[t + 1] - self.counts[lo]


def panel_matrix(
    panel: Any,
    returns: str,
    *,
    entity: str | None = None,
    time: str | None = None,
    group: str | None = None,
) -> PanelMatrix:
    """Pivot ``returns`` (and optionally a group column) to a ``(T, N)`` matrix.

    Uses :func:`~panelary.shape._tensor.build_tensor` with
    ``forward_fill=False, z_normalize=False`` (coordination note D2); values
    are cast to Float64 there.
    """
    from panelary.shape._tensor import build_tensor

    pf = _as_pf(panel, entity, time)
    if returns not in pf.columns:
        raise ValueError(
            f"returns column {returns!r} not found. Available: {pf.columns}."
        )
    cols = [returns]
    codes_col = None
    values: list[Any] = []
    if group is not None:
        if group not in pf.columns:
            raise ValueError(
                f"group column {group!r} not found. Available: {pf.columns}."
            )
        codes_col = "__cov_group_code__"
        # Dense codes are only used to test equality within one date, so their
        # numbering (which depends on the whole panel) never reaches an output.
        lf = pf.lazy().with_columns(
            pl.col(group).rank("dense").cast(pl.Float64).alias(codes_col)
        )
        mapping = (
            lf.select(group, codes_col).unique().drop_nulls().sort(codes_col).collect()
        )
        values = mapping.get_column(group).to_list()
        pf = PanelFrame(lf, entity=pf.entity_col, time=pf.time_col, validate=False)
        cols.append(codes_col)
    pt = build_tensor(pf, cols, forward_fill=False, z_normalize=False)
    R = np.ascontiguousarray(pt.tensor[:, :, 0].T, dtype=np.float64)
    finite = np.isfinite(R)
    T, N = R.shape
    counts = np.zeros((T + 1, N), dtype=np.int32)
    np.cumsum(finite, axis=0, dtype=np.int32, out=counts[1:])
    groups = None
    if codes_col is not None:
        g = pt.tensor[:, :, 1].T
        groups = np.where(np.isfinite(g), g - 1.0, -1.0).astype(np.int64)
    return PanelMatrix(
        R=R,
        times=pt.times,
        entities=pt.entities,
        entity_col=pf.entity_col,
        time_col=pf.time_col,
        finite=finite,
        counts=counts,
        groups=groups,
        group_values=values,
        group_col=group,
    )


def _need(window: int, min_coverage: float) -> int:
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(f"`min_coverage` must be in (0, 1], got {min_coverage}.")
    # round first so 0.95 * 100 is 95, not 96
    return max(2, math.ceil(round(min_coverage * window, 9)))


def universe(
    pm: PanelMatrix, t: int, window: int, min_coverage: float
) -> NDArray[np.intp]:
    """Positions of the as-of universe at ``t`` (empty until a full window)."""
    if t + 1 < window:
        return np.empty(0, dtype=np.intp)
    cnt = pm.window_count(t, window)
    ok = pm.finite[t] & (cnt >= _need(window, min_coverage))
    return np.flatnonzero(ok)


def _nonconstant(X: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Columns with at least two distinct observed values in the window.

    ``fmin`` / ``fmax`` skip NaN, so this is two reductions and no copy.
    """
    with np.errstate(invalid="ignore"):
        return np.fmax.reduce(X, axis=0) > np.fmin.reduce(X, axis=0)


def window_at(
    pm: PanelMatrix,
    t: int,
    *,
    window: int,
    min_coverage: float = 0.95,
    space: str = "correlation",
    idx: NDArray[np.intp] | None = None,
    min_entities: int = DEFAULT_MIN_ENTITIES,
    labels: bool = False,
) -> tuple[WindowStats | None, NDArray[np.intp]]:
    """The compacted window ending at position ``t`` and its universe.

    ``idx`` restricts the universe (a group's members). Entities that are
    constant over the window are dropped (they have no correlation). Returns
    ``(None, idx)`` when fewer than ``min_entities`` remain.
    """
    if idx is None:
        idx = universe(pm, t, window, min_coverage)
    if idx.size < min_entities:
        return None, idx
    X = np.ascontiguousarray(pm.R[t + 1 - window : t + 1, idx])
    keep = _nonconstant(X)
    if not keep.all():
        idx = idx[keep]
        if idx.size < min_entities:
            return None, idx
        X = np.ascontiguousarray(X[:, keep])
    ents = pm.entity_labels(idx) if labels else None
    ws = window_stats(X, space=space, entities=ents, asof=pm.times[t])
    return ws, idx


@dataclass(eq=False)
class CovarianceSeries:
    """A lazy as-of series of covariance estimates (from :func:`rolling`).

    Nothing is computed until :meth:`at` (or iteration) asks for a date; the
    last ``cache_size`` estimates are kept.

    Attributes
    ----------
    dates : polars.Series
        The scheduled (evaluation) dates.
    """

    matrix: PanelMatrix
    window: int
    method: str
    space: str
    min_coverage: float
    schedule: Schedule
    options: dict[str, Any]
    min_entities: int = DEFAULT_MIN_ENTITIES
    cache_size: int = 16
    _positions: NDArray[np.int64] = field(init=False, repr=False)
    _asof: NDArray[np.int64] = field(init=False, repr=False)
    _cache: OrderedDict[int, CovEstimate | None] = field(
        init=False, repr=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        self._positions = self.schedule.positions(self.matrix.times)
        self._asof = self.schedule.asof_positions(self.matrix.times)

    @property
    def dates(self) -> pl.Series:
        """The evaluation dates on the schedule."""
        return self.matrix.times.gather(self._positions)

    def _position_of(self, date: Any) -> int:
        """Position of the latest time ``<= date`` (-1 if before the axis)."""
        times = self.matrix.times
        pos = int(times.search_sorted(date, side="right")) - 1
        return pos

    def asof_position(self, date: Any) -> int:
        """Position of the scheduled date whose estimate is in force at ``date``."""
        pos = self._position_of(date)
        return -1 if pos < 0 else int(self._asof[pos])

    def estimate_at_position(self, s: int) -> CovEstimate | None:
        """The estimate from the window ending at time position ``s``."""
        if s in self._cache:
            self._cache.move_to_end(s)
            return self._cache[s]
        ws, _ = window_at(
            self.matrix,
            s,
            window=self.window,
            min_coverage=self.min_coverage,
            space=self.space,
            min_entities=self.min_entities,
            labels=True,
        )
        est = None if ws is None else METHODS[self.method](ws, **self.options)
        self._cache[s] = est
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return est

    def at(self, date: Any) -> CovEstimate:
        """The estimate from the last scheduled date ``<= date``.

        Raises
        ------
        LookupError
            If no scheduled date ``<= date`` has a full window with at least
            ``min_entities`` admitted entities.
        """
        s = self.asof_position(date)
        est = None if s < 0 else self.estimate_at_position(s)
        if est is None:
            raise LookupError(
                f"no covariance estimate is available as of {date!r}: the latest "
                "scheduled date on or before it has no full window with enough "
                "covered entities (window="
                f"{self.window}, min_coverage={self.min_coverage})."
            )
        return est

    def universe(self, date: Any) -> tuple[Any, ...]:
        """The entities of the estimate in force at ``date`` (empty if none)."""
        s = self.asof_position(date)
        if s < 0:
            return ()
        _, idx = window_at(
            self.matrix,
            s,
            window=self.window,
            min_coverage=self.min_coverage,
            space=self.space,
            min_entities=self.min_entities,
        )
        return self.matrix.entity_labels(idx) if idx.size >= self.min_entities else ()

    def __iter__(self) -> Iterator[tuple[Any, CovEstimate | None]]:
        """Iterate ``(date, estimate or None)`` over the scheduled dates."""
        for s in self._positions:
            yield self.matrix.times[int(s)], self.estimate_at_position(int(s))

    def __len__(self) -> int:
        return int(self._positions.size)


def rolling(
    panel: Any,
    *,
    returns: str,
    window: int = 252,
    method: str = "qis",
    schedule: Schedule | int | str | None = None,
    space: str = "correlation",
    min_coverage: float = 0.95,
    missing: str = "zero_after_demean",
    min_entities: int = DEFAULT_MIN_ENTITIES,
    entity: str | None = None,
    time: str | None = None,
    **options: Any,
) -> CovarianceSeries:
    """As-of covariance estimates over a panel, as a lazy series.

    Parameters
    ----------
    panel : PanelFrame or polars.DataFrame / LazyFrame
        Long panel; ``entity`` / ``time`` name the keys of a bare frame.
    returns : str
        The return column (returns, not prices).
    window : int, default 252
        Trailing window ``W``: the estimate as of ``t`` uses rows ``(t-W, t]``.
    method : str, default "qis"
        Any :func:`estimate` method.
    schedule : Schedule, int or str, optional
        Evaluation dates (default: every date). ``series.at(t)`` uses the last
        scheduled date ``<= t``.
    space : {"correlation", "covariance"}, default "correlation"
    min_coverage : float, default 0.95
        An entity enters the universe at ``t`` only if observed at ``t`` and on
        at least ``ceil(min_coverage * W)`` of the window's dates.
    missing : {"zero_after_demean"}
    min_entities : int, default 2
    **options
        Passed to the estimator (see :func:`estimate`).

    Returns
    -------
    CovarianceSeries
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(
            f"`missing` must be one of {MISSING_POLICIES}, got {missing!r}."
        )
    if space not in SPACES:
        raise ValueError(f"`space` must be one of {SPACES}, got {space!r}.")
    if int(window) < 2:
        raise ValueError(f"`window` must be >= 2, got {window}.")
    _need(int(window), min_coverage)
    pm = panel_matrix(panel, returns, entity=entity, time=time)
    return CovarianceSeries(
        matrix=pm,
        window=int(window),
        method=resolve_method(method),
        space=space,
        min_coverage=float(min_coverage),
        schedule=as_schedule(schedule),
        options=dict(options),
        min_entities=int(min_entities),
    )
