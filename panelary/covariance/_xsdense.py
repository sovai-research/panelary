"""Dense ``(time, entity)`` plumbing for the co-movement and cross-sectional frame ops.

:mod:`panelary.covariance._comovement` (average correlation, common
idiosyncratic volatility) and :mod:`panelary.covariance._xsdist` (Kelly-Jiang
tail risk, cross-sectional Wasserstein distance, average skewness) share one way
of crossing from a long panel to a dense matrix and back. It lives here, so
there is exactly one.

The rules this module enforces
------------------------------
* **One pivot.** Long -> dense goes through
  :func:`panelary.shape._tensor.build_tensor` with ``forward_fill=False`` and
  ``z_normalize=False`` (coordination note D2): its default forward fill would
  repeat a delisted name's last return forever (plan trap T5). Dense -> long goes
  back through :func:`panelary.shape._tensor.to_long` with ``pl.Float64``.
* **NaN means "not observed".** Nulls, NaNs and infinities in the value column
  all become NaN in the matrix and are never imputed.
* **Compact before you reduce.** A reduction across entities runs over the
  compacted members of one date (or one date and group), never over the full
  entity axis with a mask. The full axis grows when a name lists *after* the
  date being computed, and a masked reduction over it changes the summation
  tree -- so the value at ``t`` would move, in the last bit, when later data
  arrives. :func:`segment_sums` is the one place the per-date reductions happen.
* **Group labels are as of the date.** A group variant partitions date ``t``'s
  members by their label *on date t* (plan trap T12). Labels are carried as
  integer codes whose numbering is arbitrary: only equality within a date is
  used, and outputs are keyed by the original label.

Everything here is numpy + polars and internal: no public name, no registry
entry.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.detect._panel import residualise
from panelary.shape._tensor import PanelTensor, build_tensor, to_long

__all__ = [
    "Dense",
    "broadcast_rows",
    "coerce_panel",
    "coverage_count",
    "dense_matrix",
    "long_from_dense",
    "masked_row_sums",
    "segment_residuals",
    "segment_sums",
    "validate_coverage",
    "validate_int",
    "window_counts",
]

#: Temporary column that carries a group's integer code through the pivot.
_GROUP_CODE = "__panelary_xs_group_code__"


class Dense(NamedTuple):
    """A long panel pivoted to a time-major matrix, with its indices.

    Attributes
    ----------
    values : numpy.ndarray
        ``(T, N)`` float64, C-contiguous, time-major; NaN where not observed.
    times : polars.Series
        The panel's sorted unique times (row order of ``values``).
    entities : polars.Series
        The panel's sorted unique entities (column order of ``values``).
    keys : polars.DataFrame
        The dense ``(entity, time)`` grid from ``build_tensor``, entity-major,
        for :func:`long_from_dense`.
    entity_col, time_col : str
        The panel's key columns.
    group_codes : numpy.ndarray or None
        ``(T, N)`` int64 code of each cell's group label **on that date**; ``-1``
        where the row is missing or its label is null.
    group_labels : polars.Series or None
        ``group_labels[c]`` is the label behind code ``c``.
    """

    values: NDArray[np.float64]
    times: pl.Series
    entities: pl.Series
    keys: pl.DataFrame
    entity_col: str
    time_col: str
    group_codes: NDArray[np.int64] | None
    group_labels: pl.Series | None


def coerce_panel(panel: Any, entity: str | None, time: str | None) -> PanelFrame:
    """Return ``panel`` as a :class:`PanelFrame` (an existing one's keys win)."""
    if isinstance(panel, PanelFrame):
        return panel
    if not isinstance(panel, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            "`panel` must be a PanelFrame, polars.DataFrame or polars.LazyFrame, "
            f"got {type(panel).__name__!r}."
        )
    return as_panel(panel, entity=entity, time=time)


def validate_int(name: str, value: Any, *, minimum: int) -> int:
    """Return ``value`` as an int, refusing bools, floats and small values."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"`{name}` must be an int, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value!r}.")
    return int(value)


def validate_coverage(min_coverage: Any) -> float:
    """Return ``min_coverage`` as a float in ``(0, 1]``."""
    value = float(min_coverage)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"`min_coverage` must lie in (0, 1], got {min_coverage!r}.")
    return value


def dense_matrix(pf: PanelFrame, value: str, *, group: str | None = None) -> Dense:
    """Pivot ``value`` (and optionally ``group``) of ``pf`` to a time-major matrix.

    Raises :class:`ValueError` on a missing column or a duplicated
    ``(entity, time)`` key, which the pivot cannot represent.
    """
    ent, tim = pf.entity_col, pf.time_col
    cols = [value] if group is None else [value, group]
    missing = [c for c in cols if c not in pf.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} not found in the panel; available: {pf.columns}."
        )
    if group is not None and group in (ent, tim, value):
        raise ValueError(
            f"`group` ({group!r}) must be a separate column, not the entity, "
            "time or value column."
        )
    df = pf.lazy().select([ent, tim, *cols]).collect()
    n_dupes = df.height - df.select(pl.struct(ent, tim).n_unique()).item()
    if n_dupes:
        raise ValueError(
            f"the panel has {n_dupes} duplicated ({ent!r}, {tim!r}) row(s); a "
            "cross-sectional covariance feature needs one value per entity and "
            "date. Deduplicate first."
        )

    labels: pl.Series | None = None
    value_cols = [value]
    if group is not None:
        labels = df.get_column(group).drop_nulls().unique().sort()
        lut = pl.DataFrame(
            [labels, pl.Series(_GROUP_CODE, np.arange(labels.len(), dtype=np.float64))]
        )
        df = df.join(lut, on=group, how="left", maintain_order="left")
        value_cols.append(_GROUP_CODE)

    tensor = build_tensor(
        PanelFrame(df, entity=ent, time=tim, validate=False),
        value_cols,
        forward_fill=False,
        z_normalize=False,
    )
    values = np.ascontiguousarray(tensor.tensor[:, :, 0].T, dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    codes: NDArray[np.int64] | None = None
    if group is not None:
        raw = tensor.tensor[:, :, 1].T
        codes = np.where(np.isfinite(raw), raw, -1.0).astype(np.int64)
    return Dense(
        values=values,
        times=tensor.times,
        entities=tensor.entities,
        keys=tensor.keys,
        entity_col=ent,
        time_col=tim,
        group_codes=codes,
        group_labels=labels,
    )


def long_from_dense(dense: Dense, arrays: dict[str, NDArray[Any]]) -> pl.DataFrame:
    """Flatten ``(T, N)`` arrays back to a long ``(entity, time, ...)`` frame.

    NaN becomes null, so "undefined" reads the same in every output column.
    """
    names = list(arrays)
    stacked = np.stack([np.asarray(arrays[n], dtype=np.float64).T for n in names], -1)
    tensor = PanelTensor(
        tensor=stacked,
        entities=dense.entities,
        times=dense.times,
        keys=dense.keys,
        value_cols=names,
    )
    out = to_long(tensor, dtype=pl.Float64)
    return out.with_columns([pl.col(n).fill_nan(None) for n in names])


def window_counts(finite: NDArray[np.bool_], window: int) -> NDArray[np.int64]:
    """Observations of each entity in the trailing ``window`` rows ending at ``t``.

    Integer prefix sums, so exact: ``counts[t] = C[t + 1] - C[max(0, t + 1 - W)]``.
    Rows before the panel's first date count as unobserved.
    """
    csum = np.zeros((finite.shape[0] + 1, finite.shape[1]), dtype=np.int64)
    np.cumsum(finite, axis=0, dtype=np.int64, out=csum[1:])
    ends = np.arange(1, finite.shape[0] + 1)
    starts = np.maximum(ends - window, 0)
    counts: NDArray[np.int64] = csum[ends] - csum[starts]
    return counts


def segment_sums(values: NDArray[np.float64], counts: NDArray[Any]) -> NDArray[Any]:
    """Sum consecutive segments of ``values`` with lengths ``counts`` (0 if empty).

    Each segment is reduced on its own (``np.add.reduceat``), so a segment's sum
    depends only on its own elements, in their own order -- the property that
    keeps per-date reductions bit-identical when the panel grows.
    """
    counts = np.asarray(counts, dtype=np.int64)
    out = np.zeros(counts.shape[0], dtype=np.float64)
    nonempty = counts > 0
    if nonempty.any():
        starts = (np.cumsum(counts) - counts)[nonempty]
        out[nonempty] = np.add.reduceat(values, starts)
    return out


def masked_row_sums(
    matrix: NDArray[np.float64], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Per-row sums of ``matrix`` over the compacted ``mask`` members, and counts.

    ``matrix[mask]`` is row-major, so each row's members come out contiguous and
    in entity order; :func:`segment_sums` then reduces each row on its own.
    """
    counts = mask.sum(axis=1).astype(np.int64)
    return segment_sums(matrix[mask], counts), counts


def segment_residuals(
    values: NDArray[np.float64],
    *,
    n_factors: int,
    refit_every: int,
    fit_window: int,
    min_obs: int,
) -> NDArray[np.float64]:
    """Statistical-factor residuals of a ``(T, N)`` matrix, one refit segment at a time.

    This is :func:`panelary.detect._panel.residualise`, reused unchanged: the
    loadings are fit by PCA on the trailing ``fit_window`` rows before each
    refit date ``s = fit_window + k * refit_every`` (anchored at the panel's
    first date, so appending data never moves the grid), frozen, and applied to
    the dates ``[s, s + refit_every)`` through each date's own cross-section.

    What this wrapper adds is **compaction**. ``residualise`` fits on its whole
    entity axis, and a name that lists after the refit date enters that fit as
    an all-missing row: harmless in exact arithmetic, but it changes the eigen
    solver's rounding (measured: 5.6e-15 in the residuals of the other names).
    So each segment is handed only the names with at least ``min_obs``
    observations in its training block -- exactly the set ``residualise`` itself
    would give a non-zero loading -- which makes every residual a function of
    data at or before its own date, bit for bit. Names without enough history
    get no residual (NaN), rather than ``residualise``'s own-demeaned-return
    fallback, which still contains the common factor.
    """
    n_time, _n_ent = values.shape
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if n_time <= fit_window:
        return out
    csum = np.zeros((n_time + 1, values.shape[1]), dtype=np.int64)
    np.cumsum(np.isfinite(values), axis=0, dtype=np.int64, out=csum[1:])
    for start in range(fit_window, n_time, refit_every):
        lo = start - fit_window
        stop = min(start + refit_every, n_time)
        members = np.flatnonzero(csum[start] - csum[lo] >= min_obs)
        if members.size == 0:
            continue
        block = np.ascontiguousarray(values[lo:stop, members].T)
        resid = residualise(
            block,
            n_factors=n_factors,
            min_periods=fit_window,
            refit_every=refit_every,
            window=fit_window,
            min_obs=min_obs,
        )
        out[start:stop, members] = resid[:, fit_window:].T
    return out


def coverage_count(window: int, min_coverage: float) -> int:
    """Observations a window of ``window`` rows needs at ``min_coverage``."""
    return max(1, math.ceil(min_coverage * window - 1e-12))


def broadcast_rows(pf: PanelFrame, state: pl.DataFrame, on: list[str]) -> pl.DataFrame:
    """Left-join a per-date (or per-date-and-group) ``state`` onto the panel rows.

    Keeps the panel's row order. Refuses to shadow an existing panel column, so
    a broadcast can never silently rename one to ``*_right``.
    """
    df = pf.collect()
    clash = sorted((set(state.columns) - set(on)) & set(df.columns))
    if clash:
        raise ValueError(
            f"broadcast would overwrite panel column(s) {clash}; rename them "
            "first or call with broadcast=False and join yourself."
        )
    return df.join(state, on=on, how="left", maintain_order="left")
