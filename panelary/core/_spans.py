"""The canonical label-span table: one ``t1`` column, one set of spans.

Purge, concurrency, uniqueness, return attribution, time decay, the sequential
bootstrap and fold-local reweighting all read the spans built here, from the
same ``t1`` column, so they can never disagree about which rows a label covers.

A label at grid row ``r`` (time ``t0``) with end time ``t1`` covers the closed
interval ``[t0, t1]`` -- the rows of *its own entity* whose time lies in that
interval (de Prado, AFML ch. 4 and 7). Everything is done on the **flat row
axis** of the ``(entity, time)``-sorted frame: a span never crosses an entity
boundary, so one flat array of row indices *is* the per-entity computation and
no per-entity Python loop is needed.

Two coordinate systems are used and must not be confused:

* **grid rows** (``start``, ``end``) -- rows of the sorted frame, restricted to
  the label's entity. Concurrency and uniqueness live here.
* **time positions** (``start_tpos``, ``end_tpos``) -- positions in the sorted
  *unique* time index shared by every entity. The purge lives here, because
  the splitters cut the shared time axis. ``end_tpos`` is the last unique time
  ``<= t1`` over the whole panel, not just the label's entity.

Leaf module: numpy and polars only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "SpanTable",
    "_concurrency",
    "_null_mask",
    "_overlaps_any",
    "_span_sums",
    "_spans_from_t1",
    "_time_positions",
    "_uniqueness",
]

#: Fan-out of the block-local prefix-sum tree in :func:`_span_sums`: every
#: rounded partial sum covers at most this many terms.
_BLOCK = 64

#: Largest composite key ``n_entities * (n_times + 1)`` that fits int64 with room.
_KEY_LIMIT = 2**62


# --------------------------------------------------------------------------- #
# Time-position encoding (shared with the purge)
# --------------------------------------------------------------------------- #
def _null_mask(values: np.ndarray) -> NDArray[np.bool_]:
    """Boolean mask of missing values (``None``, ``NaN``, ``NaT``) in ``values``."""
    kind = values.dtype.kind
    if kind in "mM":
        return np.isnat(values)
    if kind in "fc":
        return np.isnan(values)
    if kind == "O":
        # `v != v` is True for float NaN and for numpy NaT objects.
        return np.fromiter(
            (v is None or v != v for v in values.tolist()),
            dtype=bool,
            count=values.shape[0],
        )
    return np.zeros(values.shape[0], dtype=bool)


def _coerce_like(values: np.ndarray, like: np.dtype) -> np.ndarray:
    """Coerce an ``object`` array of end times onto the time axis' dtype family."""
    if values.dtype.kind != "O":
        return values
    target = like if like.kind in "mM" else np.dtype(np.float64)
    items = [None if (v is None or v != v) else v for v in values.tolist()]
    if target.kind in "mM":
        nat = np.array("NaT", dtype=target)[()]
        return np.array([nat if v is None else v for v in items], dtype=target)
    return np.array([np.nan if v is None else v for v in items], dtype=target)


def _time_positions(times: np.ndarray, values: np.ndarray) -> NDArray[np.int64]:
    """Map end-time *values* onto positions of the sorted unique time axis.

    ``pos[k]`` is the last position ``p`` with ``times[p] <= values[k]``, so
    ``times[j] <= values[k]  <=>  j <= pos[k]`` for every grid position ``j``.
    Comparisons are numpy's (the same promotion as the elementwise ``<=`` the
    purge historically used). A missing value (``None`` / ``NaN`` / ``NaT``)
    maps to the sentinel ``-1``, which -- like a ``NaT`` comparison -- never
    satisfies ``j <= pos``.

    Never apply ``np.maximum`` to raw ``NaT`` values: it propagates and would
    silently under-purge; the ``-1`` sentinel is what makes the vectorised
    purge safe.
    """
    arr = _coerce_like(np.asarray(values), np.asarray(times).dtype)
    null = _null_mask(arr)
    pos = np.searchsorted(times, arr, side="right").astype(np.int64) - 1
    pos[null] = -1
    return pos


def _overlaps_any(
    q_start: NDArray[np.int64],
    q_end: NDArray[np.int64],
    ref_start: NDArray[np.int64],
    ref_end: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Does each closed query interval intersect any closed reference interval?

    ``[qs, qe]`` and ``[rs, re]`` intersect iff ``rs <= qe`` and ``qs <= re``.
    ``ref_start`` must be sorted ascending. Among the references starting at or
    before ``qe``, the one with the largest end decides the answer, so the test
    is a running maximum plus one ``searchsorted``: O((n + m) log m) instead of
    the O(n * m) pairwise loop. Inverted intervals (end < start) and the ``-1``
    null sentinel are handled by exactly the same inequality.
    """
    if ref_start.size == 0 or q_start.size == 0:
        return np.zeros(q_start.shape[0], dtype=bool)
    run_end = np.maximum.accumulate(ref_end)
    k = np.searchsorted(ref_start, q_end, side="right") - 1
    hit = k >= 0
    out = np.zeros(q_start.shape[0], dtype=bool)
    out[hit] = run_end[k[hit]] >= q_start[hit]
    return out


# --------------------------------------------------------------------------- #
# The span table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SpanTable:
    """Label spans on the ``(entity, time)``-sorted grid (frozen).

    Every array has one entry per *kept* label, ordered by ``start``.

    Attributes
    ----------
    start : ndarray of int64
        Grid row of the label (its ``t0``), sorted ascending.
    end : ndarray of int64
        Last grid row of the *same entity* with ``time <= t1`` (``>= start``).
    seg_start, seg_end : ndarray of int64
        First / last grid row of the label's entity.
    label_row : ndarray of int64
        Row of the label in the sorted frame (equal to ``start`` here; kept
        separate so an event-subset table can point elsewhere).
    entity_code : ndarray of int64
        Dense entity index (``0..E-1`` in sorted entity order).
    start_tpos, end_tpos : ndarray of int64
        Positions of ``t0`` and ``t1`` on the panel's sorted unique-time axis
        (``end_tpos`` = last unique time ``<= t1``, over every entity).
    n_rows : int
        Number of grid rows ``R``.
    n_times : int
        Number of unique times.
    n_dropped : int
        Rows not turned into spans: null ``t1``, flagged ``censored``, or
        ``t1`` beyond the entity's last row (never clamped -- clamping is
        AFML's length-dependent ``t1.fillna(last)``).
    """

    start: NDArray[np.int64]
    end: NDArray[np.int64]
    seg_start: NDArray[np.int64]
    seg_end: NDArray[np.int64]
    label_row: NDArray[np.int64]
    entity_code: NDArray[np.int64]
    start_tpos: NDArray[np.int64]
    end_tpos: NDArray[np.int64]
    n_rows: int
    n_times: int
    n_dropped: int

    def __len__(self) -> int:
        return int(self.start.shape[0])

    @property
    def length(self) -> NDArray[np.int64]:
        """Rows covered by each span (``end - start + 1``)."""
        return self.end - self.start + 1

    def subset(self, mask: NDArray[np.bool_]) -> SpanTable:
        """Keep the spans where ``mask`` is True (order preserved)."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != self.start.shape:
            raise ValueError(
                f"mask has shape {m.shape}, expected {self.start.shape} (one "
                "entry per span)."
            )
        return SpanTable(
            start=self.start[m],
            end=self.end[m],
            seg_start=self.seg_start[m],
            seg_end=self.seg_end[m],
            label_row=self.label_row[m],
            entity_code=self.entity_code[m],
            start_tpos=self.start_tpos[m],
            end_tpos=self.end_tpos[m],
            n_rows=self.n_rows,
            n_times=self.n_times,
            n_dropped=self.n_dropped,
        )

    def time_projection(self) -> NDArray[np.int64]:
        """Per unique time, the largest ``end_tpos`` of the spans starting there.

        ``-1`` where no span starts. This is the position form of the
        conservative per-time ``max(t1)`` the panel purge uses
        (``model_selection._resolve_t1``): the two agree elementwise on every
        label this table keeps.
        """
        out = np.full(self.n_times, -1, dtype=np.int64)
        np.maximum.at(out, self.start_tpos, self.end_tpos)
        return out

    def covers(self, positions: NDArray[np.int64]) -> NDArray[np.bool_]:
        """For each span, does ``[start_tpos, end_tpos]`` contain any of ``positions``?"""
        pos = np.unique(np.asarray(positions, dtype=np.int64))
        lo = np.searchsorted(pos, self.start_tpos, side="left")
        hi = np.searchsorted(pos, self.end_tpos, side="right")
        return hi > lo


def _as_frame(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _resolve_keys(
    frame: pl.DataFrame, entity: str | None, time: str | None
) -> tuple[str, str]:
    cols = frame.columns
    if len(cols) < 2 and (entity is None or time is None):
        raise ValueError("need an entity and a time column (got fewer than two).")
    return (
        entity if entity is not None else cols[0],
        time if time is not None else cols[1],
    )


def _grid_arrays(
    frame: pl.DataFrame, entity: str, time: str
) -> tuple[NDArray[np.int64], NDArray[np.int64], np.ndarray, np.ndarray]:
    """``(seg, tpos, time_values, unique_times)`` for an already-sorted grid."""
    if frame.get_column(time).null_count():
        raise ValueError(f"time column {time!r} contains nulls.")
    seg = frame.get_column(entity).rle_id().to_numpy().astype(np.int64, copy=False)
    tvals = frame.get_column(time).to_numpy()
    utimes = frame.get_column(time).unique().sort().to_numpy()
    tpos = np.searchsorted(utimes, tvals, side="left").astype(np.int64)
    return seg, tpos, tvals, utimes


def _spans_from_t1(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    t1: str,
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = None,
) -> tuple[pl.DataFrame, SpanTable]:
    """Build the canonical :class:`SpanTable` from a ``t1`` column.

    The grid is every row of ``frame`` (sorted by ``(entity, time)``); a label is
    every row with a non-null ``t1`` that is not flagged ``censored``. Returns
    the sorted frame together with the table, so callers can align outputs.

    O(R + N log R) with no per-entity loop: with ``seg`` the dense entity id and
    ``tpos`` the unique-time position, the composite key
    ``seg * (n_times + 1) + tpos`` is strictly increasing on the sorted grid,
    and one ``searchsorted`` of ``seg * (n_times + 1) + end_tpos`` finds each
    span's last row inside its own entity.

    Raises
    ------
    ValueError
        If ``(entity, time)`` is not unique, the time column has nulls, or a
        label ends before it starts (``t1 < t0``).
    """
    df = _as_frame(frame)
    entity_col, time_col = _resolve_keys(df, entity, time)
    if t1 not in df.columns:
        raise ValueError(f"t1 column {t1!r} not found; columns are {df.columns}.")
    df = df.sort([entity_col, time_col], maintain_order=True)
    n_rows = df.height

    seg, tpos, tvals, utimes = _grid_arrays(df, entity_col, time_col)
    n_times = int(utimes.shape[0])
    n_ent = int(seg[-1]) + 1 if n_rows else 0
    if n_ent * (n_times + 1) >= _KEY_LIMIT:  # pragma: no cover - 10^6 x 10^12
        raise OverflowError("panel too large for the int64 span key.")
    key = seg * (n_times + 1) + tpos
    if n_rows > 1 and not bool(np.all(key[1:] > key[:-1])):
        raise ValueError(
            f"(entity, time) = ({entity_col!r}, {time_col!r}) is not unique; a "
            "span table needs one row per entity and time."
        )

    rows = np.arange(n_rows, dtype=np.int64)
    first = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]]) if n_rows else rows
    last = np.r_[first[1:] - 1, n_rows - 1] if n_rows else rows
    seg_first = first[seg]
    seg_last = last[seg]

    t1_vals = df.get_column(t1).to_numpy()
    t1_vals = _coerce_like(np.asarray(t1_vals), tvals.dtype)
    keep = ~_null_mask(t1_vals)
    if censored is not None and censored in df.columns:
        flag = df.get_column(censored).fill_null(False).to_numpy().astype(bool)
        keep &= ~flag

    idx = rows[keep]
    t1_kept = t1_vals[keep]
    t0_kept = tvals[keep]
    if idx.size and bool(np.any(t1_kept < t0_kept)):
        bad = int(idx[np.argmax(t1_kept < t0_kept)])
        raise ValueError(
            f"label at row {bad} ends before it starts ({t1!r} < {time_col!r}); "
            "every label must satisfy t0 <= t1."
        )
    # Beyond the entity's last observed time: the span cannot be observed in
    # full, so the label is unresolved -- dropped, never clamped to the end.
    inside = t1_kept <= tvals[seg_last[idx]]
    idx = idx[inside]
    t1_kept = t1_kept[inside]

    end_tpos = np.searchsorted(utimes, t1_kept, side="right").astype(np.int64) - 1
    k1 = seg[idx] * (n_times + 1) + end_tpos
    end = np.searchsorted(key, k1, side="right").astype(np.int64) - 1

    table = SpanTable(
        start=idx,
        end=end,
        seg_start=seg_first[idx],
        seg_end=seg_last[idx],
        label_row=idx.copy(),
        entity_code=seg[idx],
        start_tpos=tpos[idx],
        end_tpos=end_tpos,
        n_rows=n_rows,
        n_times=n_times,
        n_dropped=int(n_rows - idx.shape[0]),
    )
    return df, table


# --------------------------------------------------------------------------- #
# Kernels on the span table
# --------------------------------------------------------------------------- #
def _concurrency(
    start: NDArray[np.int64], end: NDArray[np.int64], n_rows: int
) -> NDArray[np.int64]:
    """Number of spans covering each grid row (AFML 4.1), exact in int64.

    Event stream: ``+1`` at each start, ``-1`` after each end, one cumulative
    sum. O(R + N); spans never cross an entity boundary, so no reset is needed.
    """
    d = np.bincount(start, minlength=n_rows + 1)[: n_rows + 1].astype(np.int64)
    d -= np.bincount(end + 1, minlength=n_rows + 1)[: n_rows + 1]
    return np.cumsum(d[:n_rows])


def _span_sums(
    v: np.ndarray,
    start: NDArray[np.int64],
    end: NDArray[np.int64],
    *,
    offset: int = 0,
    block: int = _BLOCK,
) -> NDArray[np.float64]:
    """``sum(v[start + offset : end + 1])`` for every span, accurately.

    A single global prefix sum loses accuracy as the grid grows: its magnitude,
    and so the rounding of every difference of two prefixes, scales with ``R``
    (measured 2.9e-12 relative at ``R = 2e5``, growing). Here the row axis is
    cut into fixed blocks of ``block`` rows with inclusive prefixes *inside*
    each block (one sequential ``np.cumsum`` along a padded
    ``(n_blocks, block)`` view). A span inside one block is a difference of two
    in-block prefixes; a span crossing blocks is the tail of its first block,
    plus the head of its last block, plus the sum of the whole blocks in between
    -- which is itself a span sum over the array of block totals, computed by
    the same function (a ``block``-ary tree, ``log_block(R)`` levels deep).
    Every difference therefore involves partial sums of at most ``block``
    terms, so the absolute error is about ``block * eps * max|v|`` whatever the
    size of the grid. Empty spans (``start + offset > end``) sum to 0.
    """
    x = np.asarray(v, dtype=np.float64)
    s = np.asarray(start, dtype=np.int64) + offset
    e = np.asarray(end, dtype=np.int64)
    out = np.zeros(s.shape[0], dtype=np.float64)
    ok = s <= e
    if x.shape[0] == 0 or not bool(ok.any()):
        return out
    out[ok] = _span_sums_nonempty(x, s[ok], e[ok], block)
    return out


def _span_sums_nonempty(
    x: NDArray[np.float64],
    s: NDArray[np.int64],
    e: NDArray[np.int64],
    block: int,
) -> NDArray[np.float64]:
    """:func:`_span_sums` for spans known to satisfy ``s <= e`` (recursive)."""
    n = x.shape[0]
    n_blocks = -(-n // block)
    padded = np.zeros(n_blocks * block, dtype=np.float64)
    padded[:n] = x
    prefix = np.cumsum(padded.reshape(n_blocks, block), axis=1)
    flat = prefix.ravel()
    head = np.where(s % block != 0, flat[np.maximum(s - 1, 0)], 0.0)
    res = flat[e] - head
    bs = s // block
    be = e // block
    cross = bs != be
    if bool(cross.any()):
        totals = prefix[:, -1]
        c_bs = bs[cross]
        c_be = be[cross]
        middle = np.zeros(c_bs.shape[0], dtype=np.float64)
        inner = c_be > c_bs + 1
        if bool(inner.any()):
            middle[inner] = _span_sums_nonempty(
                totals, c_bs[inner] + 1, c_be[inner] - 1, block
            )
        res[cross] = (totals[c_bs] - head[cross]) + middle + flat[e[cross]]
    return res


def _uniqueness(
    spans: SpanTable, concurrency: NDArray[np.int64] | None = None
) -> NDArray[np.float64]:
    """Average uniqueness ``mean_{r in span} 1 / c_r`` of every span (AFML 4.2)."""
    c = (
        _concurrency(spans.start, spans.end, spans.n_rows)
        if concurrency is None
        else concurrency
    )
    inv = np.zeros(c.shape[0], dtype=np.float64)
    pos = c > 0
    inv[pos] = 1.0 / c[pos]
    return _span_sums(inv, spans.start, spans.end) / spans.length
