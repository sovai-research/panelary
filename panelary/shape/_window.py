"""The causal spine: strictly trailing windows, per entity, batched over the panel.

Every ``flavour="trailing"`` transform in :mod:`panelary.shape` (and every
windowed ``embed/`` transform) reads its input through this module. Two entry
points:

* :func:`trailing_windows` -- the contract primitive: one entity's
  time-ordered history in, an ``(n_rows, window[, k])`` **view** out, where row
  ``t`` reads ``values[t - window + 1 : t + 1]`` and nothing else.
* :class:`PanelWindows` -- the same windows for a whole panel at once, so a
  numeric kernel runs **once** over every row of every entity instead of once
  per entity (the batching rule: amortise the Python frame, never loop per
  window). Each entity's rows are laid out in one buffer preceded by
  ``window - 1`` NaN cells, so a window that starts before its entity's first
  observation reads NaN -- never the previous entity's tail. There is no code
  path in which a single window spans two entities.

Invariants (``tests/test_shape_window.py``):

* **Prefix invariance** -- ``f(x[:T])[t] == f(x[:T+k])[t]``, bit for bit.
  Every quantity here is a function of the row's own trailing cells; nothing
  depends on ``len(x)``.
* **Never centred.** There is no ``center=`` parameter; it is absent, not
  defaulted.
* Rows with fewer than ``min_periods`` observations are NaN -- never
  back-filled, never computed on a short window as if it were full.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from panelary.core.panel_frame import PanelFrame

__all__ = [
    "PanelWindows",
    "SortedPanel",
    "sort_panel",
    "trailing_windows",
]


def _check_window(window: object, name: str = "window") -> int:
    if (
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or window < 1
    ):
        raise ValueError(f"`{name}` must be a positive int, got {window!r}.")
    return int(window)


def trailing_windows(
    values: NDArray[Any],
    window: int,
    *,
    min_periods: int | None = None,
) -> NDArray[np.float64]:
    """Frame one entity's history into strictly trailing windows.

    Row ``t`` of the result is ``values[t - window + 1 : t + 1]``; positions
    before the first observation are NaN. Nothing ahead of ``t`` is ever read.

    Parameters
    ----------
    values : numpy.ndarray
        ``(n_rows,)`` or ``(n_rows, k)``: one entity's history, time-ordered.
    window : int
        Window length ``W`` (number of observations, not a time span).
    min_periods : int, optional
        Rows with fewer than ``min_periods`` observations (``t + 1 <
        min_periods``) are entirely NaN. Default ``window``: every row before
        the first full window carries NaN padding, so a NaN-propagating kernel
        emits NaN there.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, window)`` or ``(n_rows, window, k)`` float64. A **view** onto
        one padded copy of the input (``n + window - 1`` rows) when
        ``min_periods == window``; a copy when ``min_periods < window`` (rows
        before ``min_periods`` must be blanked, which a view cannot express).
        Treat it as read-only either way.

    Raises
    ------
    ValueError
        If ``window`` / ``min_periods`` are not positive, or
        ``min_periods > window``.
    """
    w = _check_window(window)
    mp = w if min_periods is None else _check_window(min_periods, "min_periods")
    if mp > w:
        raise ValueError(
            f"`min_periods` ({mp}) cannot exceed `window` ({w}): a window never "
            "holds more than `window` observations."
        )
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"`values` must be 1-D or 2-D, got shape {x.shape}.")
    pad = np.full((w - 1, *x.shape[1:]), np.nan, dtype=np.float64)
    buf = np.concatenate([pad, x], axis=0)
    view = sliding_window_view(buf, w, axis=0)  # (n, [k,] w)
    if x.ndim == 2:
        view = np.moveaxis(view, -1, 1)  # (n, w, k), still a view
    if mp < w:
        out = np.array(view, dtype=np.float64)
        out[: mp - 1] = np.nan
        return out
    return view


# --------------------------------------------------------------------------- #
# Panel layout: sort once, remember how to put rows back
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SortedPanel:
    """A collected panel, sorted ``(entity, time)``, plus the way back.

    Attributes
    ----------
    frame : polars.DataFrame
        The collected input, in the caller's original row order.
    sorted_keys : polars.DataFrame
        ``(entity, time)`` in sorted order.
    order : numpy.ndarray
        ``order[i]`` is the original row index of sorted row ``i``.
    values : numpy.ndarray
        ``(n, k)`` float64 value columns in sorted order (null -> NaN).
    lengths : numpy.ndarray
        Observations per entity, in sorted entity order.
    entities : polars.Series
        Sorted unique entity ids.
    entity_col, time_col : str
    value_cols : list of str
    """

    frame: pl.DataFrame
    sorted_keys: pl.DataFrame
    order: NDArray[np.int64]
    values: NDArray[np.float64]
    lengths: NDArray[np.int64]
    entities: pl.Series
    entity_col: str
    time_col: str
    value_cols: list[str]

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def starts(self) -> NDArray[np.int64]:
        """Sorted-row index of each entity's first observation."""
        return np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)

    def unsort(self, out_sorted: NDArray[Any]) -> NDArray[Any]:
        """Scatter a sorted-order ``(n, ...)`` array back to the original row order."""
        res = np.empty_like(out_sorted)
        res[self.order] = out_sorted
        return res

    @property
    def is_sorted(self) -> bool:
        """Whether the input already arrived in ``(entity, time)`` order."""
        return bool(np.array_equal(self.order, np.arange(self.order.size)))

    def unsort_last(self, out_sorted: NDArray[Any]) -> NDArray[Any]:
        """Put a ``(..., n)`` column-major array back into the original row order.

        A gather along the last axis (cheaper than a scatter), skipped entirely
        when the input was already sorted.
        """
        if self.is_sorted:
            return out_sorted
        inv = np.empty_like(self.order)
        inv[self.order] = np.arange(self.order.size, dtype=self.order.dtype)
        return out_sorted[..., inv]


def sort_panel(
    panel: PanelFrame, value_cols: Sequence[str], *, owner: str
) -> SortedPanel:
    """Collect ``panel`` and sort it ``(entity, time)``, keeping the inverse order.

    Values are cast to float64 before anything touches them (``AGENTS.md``
    invariant 3); nulls become NaN, which every consumer treats as "no
    observation".

    Raises
    ------
    ValueError
        On duplicate ``(entity, time)`` keys -- a trailing window over a
        duplicated date has no defined order.
    """
    ent, tim = panel.entity_col, panel.time_col
    cols = list(value_cols)
    frame = panel.collect()
    work = frame.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("__row__"),
        pl.col(ent),
        pl.col(tim),
        *[pl.col(c).cast(pl.Float64) for c in cols],
    ).sort([ent, tim], maintain_order=True)
    if work.height and work.select(pl.struct(ent, tim).is_duplicated().any()).item():
        dup = (
            work.filter(pl.struct(ent, tim).is_duplicated())
            .select(ent, tim)
            .head(3)
            .rows()
        )
        raise ValueError(
            f"{owner}: duplicate (entity, time) keys, e.g. {dup}. A trailing window "
            "needs each entity observed at most once per time."
        )
    runs = work.group_by(ent, maintain_order=True).len()
    values = (
        work.select(cols).to_numpy().astype(np.float64, copy=False)
        if cols
        else np.empty((work.height, 0), dtype=np.float64)
    )
    return SortedPanel(
        frame=frame,
        sorted_keys=work.select(ent, tim),
        order=work["__row__"].to_numpy().astype(np.int64),
        values=np.ascontiguousarray(values),
        lengths=runs["len"].to_numpy().astype(np.int64),
        entities=runs[ent],
        entity_col=ent,
        time_col=tim,
        value_cols=cols,
    )


# --------------------------------------------------------------------------- #
# The batched spine
# --------------------------------------------------------------------------- #
class PanelWindows:
    """Trailing windows for every row of a sorted panel, without a per-entity loop.

    The ``(n, k)`` value block is copied once into a buffer in which each
    entity's rows are preceded by ``reach - 1`` NaN rows. Row ``i``'s trailing
    window of length ``W <= reach`` is then the ``W`` buffer cells ending at
    ``end[i]`` -- which by construction lie inside row ``i``'s own entity or its
    NaN gap. All index arithmetic is vectorised; the only Python loops in the
    kernels below run over *window offsets* (at most ``W`` iterations), never
    over rows or entities.

    Parameters
    ----------
    values : numpy.ndarray
        ``(n, k)`` float64 values, sorted ``(entity, time)``.
    lengths : numpy.ndarray
        Observations per entity, in the same order.
    reach : int
        The longest look-back any kernel will need (``W`` for a window of ``W``
        observations; ``(lags - 1) * dilation + 1`` for a delay embedding).
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        lengths: NDArray[np.int64],
        reach: int,
    ) -> None:
        self.reach = _check_window(reach, "reach")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        lengths = np.asarray(lengths, dtype=np.int64)
        if int(lengths.sum()) != x.shape[0]:
            raise ValueError(
                f"PanelWindows: entity lengths sum to {int(lengths.sum())} but "
                f"values has {x.shape[0]} rows."
            )
        n, k = x.shape
        gap = self.reach - 1
        ent_ord = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        self.n, self.k = n, k
        self.position: NDArray[np.int64] = (
            np.arange(n, dtype=np.int64) - starts[ent_ord]
        )
        self.end: NDArray[np.int64] = np.arange(n, dtype=np.int64) + (ent_ord + 1) * gap
        self.buf: NDArray[np.float64] = np.full((n + lengths.size * gap, k), np.nan)
        self.buf[self.end] = x

    # ------------------------------------------------------------------ #
    @property
    def nbytes(self) -> int:
        return int(self.buf.nbytes)

    def lagged(self, lag: int) -> NDArray[np.float64]:
        """``(n, k)``: each row's value ``lag`` observations back (NaN before start)."""
        if not 0 <= lag < self.reach:
            raise ValueError(f"lag {lag} outside [0, reach={self.reach}).")
        return self.buf[self.end - lag]

    def windows(
        self, window: int, rows: slice | NDArray[Any] | None = None
    ) -> NDArray[np.float64]:
        """Materialise ``(c, window, k)`` trailing windows for ``rows`` (all by default).

        A copy of ``c * window * k`` float64s -- use :meth:`iter_windows` to
        bound the memory on large panels.
        """
        w = _check_window(window)
        if w > self.reach:
            raise ValueError(f"window {w} exceeds the buffer's reach {self.reach}.")
        view = sliding_window_view(self.buf, w, axis=0)  # (N-w+1, k, w)
        idx = self.end if rows is None else self.end[rows]
        return np.moveaxis(view[idx - w + 1], -1, 1)

    def iter_windows(
        self, window: int, chunk_rows: int
    ) -> Iterator[tuple[slice, NDArray[np.float64]]]:
        """Yield ``(row_slice, windows)`` chunks of at most ``chunk_rows`` rows."""
        step = max(1, int(chunk_rows))
        for lo in range(0, self.n, step):
            sl = slice(lo, min(self.n, lo + step))
            yield sl, self.windows(window, sl)

    def windows_last(
        self, window: int, rows: slice | NDArray[Any] | None = None
    ) -> NDArray[np.float64]:
        """As :meth:`windows`, but ``(c, k, window)`` -- the window on the last axis.

        The gather produces this layout contiguously, so a kernel that works
        along the window (an FFT) needs no transpose.
        """
        w = _check_window(window)
        if w > self.reach:
            raise ValueError(f"window {w} exceeds the buffer's reach {self.reach}.")
        view = sliding_window_view(self.buf, w, axis=0)  # (N-w+1, k, w)
        idx = self.end if rows is None else self.end[rows]
        return view[idx - w + 1]

    @property
    def buf_t(self) -> NDArray[np.float64]:
        """The padded buffer transposed to ``(k, N)``, contiguous (cached)."""
        cached = getattr(self, "_buf_t", None)
        if cached is None:
            cached = np.ascontiguousarray(self.buf.T)
            self._buf_t = cached
        return cached

    def rolling(
        self,
        length: int,
        stat: str,
        *,
        lag: int = 0,
    ) -> NDArray[np.float64]:
        """``(n, k)``: a statistic over the ``length`` observations ending ``lag`` back.

        Computed for every buffer cell with elementwise operations over shifted
        slices, so the value for a row depends on exactly its own ``length``
        cells, in a fixed order -- bit-identical under truncation of the panel.
        NaN anywhere in the segment propagates (a segment with a missing
        observation is missing).

        Parameters
        ----------
        length : int
            Segment length ``L``.
        stat : {"mean", "max", "min", "std", "last", "sum"}
            ``std`` is the population standard deviation (``ddof=0``).
        lag : int, default 0
            Offset of the segment's last cell behind the row (``lag + length <=
            reach`` is required).
        """
        seg = self.segment_stat(length, stat)
        return seg[self.end - lag]

    def segment_stat(self, length: int, stat: str) -> NDArray[np.float64]:
        """The statistic of :meth:`rolling` for *every* buffer cell (``(N, k)``)."""
        L = _check_window(length, "length")
        if self.reach < L:
            raise ValueError(
                f"segment length {L} exceeds the buffer's reach {self.reach}."
            )
        buf = self.buf
        N = buf.shape[0]
        out = np.full_like(buf, np.nan)
        if N < L:
            return out
        # shifted[j] is the cell j steps back from each target cell q >= L-1.
        shifted: Callable[[int], NDArray[np.float64]] = lambda j: buf[L - 1 - j : N - j]  # noqa: E731
        if stat == "last":
            out[L - 1 :] = shifted(0)
            return out
        if stat in ("mean", "sum", "std"):
            acc = shifted(L - 1).copy()
            for j in range(L - 2, -1, -1):
                acc += shifted(j)
            if stat == "sum":
                out[L - 1 :] = acc
                return out
            mean = acc / L
            if stat == "mean":
                out[L - 1 :] = mean
                return out
            ss = (shifted(L - 1) - mean) ** 2
            for j in range(L - 2, -1, -1):
                ss += (shifted(j) - mean) ** 2
            out[L - 1 :] = np.sqrt(ss / L)
            return out
        if stat in ("max", "min"):
            op = np.maximum if stat == "max" else np.minimum  # both propagate NaN
            acc = shifted(L - 1).copy()
            for j in range(L - 2, -1, -1):
                op(acc, shifted(j), out=acc)
            out[L - 1 :] = acc
            return out
        raise ValueError(
            f"unknown statistic {stat!r}; choose one of "
            "['last', 'max', 'mean', 'min', 'std', 'sum']."
        )
