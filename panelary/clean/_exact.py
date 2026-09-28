"""The exact-duplicate path: the one place Panelary spells "drop duplicates".

Exact deduplication is a thin layer over Polars' own ``unique`` /
``over``-grouping (null-equal: two nulls, or two NaNs, are the same value).
:func:`panelary.preprocessing.reindex` routes its ``drop_duplicates=True``
branch through :func:`exact_unique` so there is a single implementation, and
:class:`~panelary.clean.Deduplicator` uses :func:`exact_groups` as the
pre-pass of every method.

This module imports nothing from the rest of :mod:`panelary.clean`, so the
preprocessing layer can depend on it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar

import numpy as np
import polars as pl

__all__ = ["exact_groups", "exact_unique"]

_F = TypeVar("_F", pl.DataFrame, pl.LazyFrame)


def exact_unique(
    frame: _F,
    subset: str | Sequence[str] | None = None,
    *,
    keep: Literal["first", "last", "any", "none"] = "any",
    maintain_order: bool = False,
) -> _F:
    """Drop exact duplicate rows (optionally judged on ``subset`` only).

    A direct pass-through to :meth:`polars.DataFrame.unique` with the same
    defaults, so callers that switch to it keep bit-identical behaviour.

    Parameters
    ----------
    frame : polars.DataFrame | polars.LazyFrame
        Input frame; the output has the same type.
    subset : str | sequence of str, optional
        Columns that define a duplicate. Defaults to all columns.
    keep : {"first", "last", "any", "none"}, default="any"
        Which member of each duplicate group survives (Polars semantics).
    maintain_order : bool, default=False
        Keep the input order of the surviving rows.

    Returns
    -------
    polars.DataFrame | polars.LazyFrame
    """
    return frame.unique(subset=subset, keep=keep, maintain_order=maintain_order)


def exact_groups(
    frame: pl.DataFrame,
    cols: Sequence[str],
    *,
    row_col: str,
    rank: np.ndarray,
) -> np.ndarray:
    """Representative row of every exact-duplicate group.

    Parameters
    ----------
    frame : polars.DataFrame
        Carries ``row_col`` holding ``0 .. n - 1``.
    cols : sequence of str
        Columns whose (null-equal) values define a group.
    row_col : str
        The row-id column.
    rank : numpy.ndarray
        Point-in-time rank per row id; the representative is the member with
        the lowest rank (the earliest occurrence).

    Returns
    -------
    numpy.ndarray
        ``int64`` array: ``rep[row]`` is the row id of its group's earliest
        member (``rep[row] == row`` for representatives and singletons).
    """
    n = frame.height
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rows = frame.get_column(row_col).to_numpy().astype(np.int64)
    ranked = frame.select(list(cols)).with_columns(
        pl.Series("__r", rank[rows]), pl.Series("__row", rows)
    )
    rep = (
        ranked.select(pl.col("__row").sort_by("__r").first().over(list(cols)))
        .to_series()
        .to_numpy()
        .astype(np.int64)
    )
    out = np.empty(n, dtype=np.int64)
    out[rows] = rep
    return out
