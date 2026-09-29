"""Concurrency, average uniqueness and the effective sample size (AFML 4.1-4.4).

Every function here builds the one canonical span table
(:func:`panelary.core._spans._spans_from_t1`) from the frame's ``t1`` column,
so the rows a label covers are exactly the rows the purge treats as that
label's information set.

These functions are **global**: they count every label in the frame. That is
the right thing for describing a labelled panel (how many independent labels
does it hold?) and for the final refit on all data. Inside cross-validation use
:class:`panelary.weights.FoldWeights`, which recomputes everything from the
training fold's labels only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from panelary.core._spans import (
    SpanTable,
    _concurrency,
    _spans_from_t1,
    _uniqueness,
)

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["average_uniqueness", "concurrency", "effective_n", "spans"]


def _label_column(
    n_rows: int, rows: np.ndarray, values: np.ndarray, name: str
) -> pl.Series:
    """A Float64 column holding ``values`` at ``rows`` and null elsewhere."""
    series = pl.repeat(None, n_rows, dtype=pl.Float64, eager=True).alias(name)
    return series.scatter(rows, np.asarray(values, dtype=np.float64))


def spans(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
) -> SpanTable:
    """The canonical :class:`~panelary.core._spans.SpanTable` of a labelled panel.

    A label is every row with a non-null ``t1`` that is not flagged in the
    ``censored`` column (used only if the frame has it). It covers the rows of
    its own entity whose time lies in ``[t, t1]``. A label whose ``t1`` lies
    beyond its entity's last row is unresolved and is dropped, never clamped.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel, one row per ``(entity, time)``.
    t1 : str, default "t1"
        Label end-time column (same dtype as the time column).
    entity, time : str, optional
        Key columns; default to the first and second columns.
    censored : str or None, default "censored"
        Boolean column of unresolved labels to drop (e.g. from
        :func:`panelary.label.triple_barrier`). Ignored if absent.

    Returns
    -------
    SpanTable
        Frozen table of grid-row spans on the ``(entity, time)``-sorted frame.
    """
    _, table = _spans_from_t1(df, t1=t1, entity=entity, time=time, censored=censored)
    return table


def concurrency(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
    out: str = "concurrency",
) -> pl.DataFrame:
    """Attach the number of labels covering each row (AFML snippet 4.1 idea).

    Global: counts every label in the frame (see the module note). An event
    stream -- ``+1`` at each label start, ``-1`` after each end, one cumulative
    sum -- in exact int64, O(R + N) with no per-event loop.

    Parameters
    ----------
    df, t1, entity, time, censored
        As in :func:`spans`.
    out : str, default "concurrency"
        Name of the Int64 column written on **every** row (0 where no label
        is active).

    Returns
    -------
    polars.DataFrame
        The frame sorted by ``(entity, time)`` with the ``out`` column.
    """
    frame, table = _spans_from_t1(
        df, t1=t1, entity=entity, time=time, censored=censored
    )
    c = _concurrency(table.start, table.end, table.n_rows)
    return frame.with_columns(pl.Series(out, c, dtype=pl.Int64))


def average_uniqueness(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
    out: str = "uniqueness",
) -> pl.DataFrame:
    """Attach each label's average uniqueness ``mean_{r in span} 1 / c_r`` (AFML 4.2).

    Global: concurrency ``c`` counts every label in the frame. ``1`` means no
    other label overlaps this one; ``k`` identical labels get ``1/k`` each.
    Computed as block-local prefix sums of ``1/c`` over the span table, so the
    cost is O(R + N) and the rounding does not grow with the panel.

    Parameters
    ----------
    df, t1, entity, time, censored
        As in :func:`spans`.
    out : str, default "uniqueness"
        Name of the Float64 column; set on label rows, null elsewhere
        (unresolved, censored or unlabelled rows).

    Returns
    -------
    polars.DataFrame
        The frame sorted by ``(entity, time)`` with the ``out`` column.
    """
    frame, table = _spans_from_t1(
        df, t1=t1, entity=entity, time=time, censored=censored
    )
    u = _uniqueness(table)
    return frame.with_columns(_label_column(table.n_rows, table.label_row, u, out))


def effective_n(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
) -> float:
    """The effective number of independent labels, ``sum_i ubar_i``.

    Overlapping labels share information, so a panel's label count overstates
    its sample size: a 20-row fixed-horizon label on every row has average
    uniqueness close to ``1/21``, so a "5,000-day backtest" holds about 240
    independent labels. This is the honest number to report next to any
    financial-ML backtest. Always ``<=`` the number of resolved labels.

    Parameters
    ----------
    df, t1, entity, time, censored
        As in :func:`spans`.

    Returns
    -------
    float
        ``sum`` of the average uniqueness over resolved, uncensored labels
        (``0.0`` when there are none).
    """
    _, table = _spans_from_t1(df, t1=t1, entity=entity, time=time, censored=censored)
    if len(table) == 0:
        return 0.0
    return float(np.sum(_uniqueness(table)))
