"""Fixed-width output: ``D`` loose columns <-> one ``pl.Array(dtype, D)`` column.

A 10,000-wide embedding should not be 10,000 top-level Polars columns. These two
helpers move between the two representations.

The dtype rule for the whole shape algebra: **accumulate in float64 always;
store in float32 only at the final** ``LIFT`` **boundary**. ``as_embedding``
defaults to ``Float32`` because a ``LIFT`` output (a basis expansion, thousands
wide) is the case it exists for -- nothing in it is precise to 1e-16, and at
float64 a 10,000-wide row is 80 KB. Every ``COMPRESS`` and ``FACTORIZE`` result
in :mod:`panelary.shape` is emitted as float64, because reconstruction error is
the number users check; pass ``dtype=pl.Float64`` to keep it that way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

__all__ = ["as_embedding", "explode_embedding"]


def as_embedding(
    df: pl.DataFrame,
    cols: Sequence[str],
    *,
    name: str,
    dtype: Any = pl.Float32,
) -> pl.DataFrame:
    """Pack ``cols`` into a single fixed-width ``pl.Array(dtype, len(cols))`` column.

    Parameters
    ----------
    df : polars.DataFrame
        The frame holding the loose columns.
    cols : sequence of str
        Columns to pack, in order. They are dropped from the result.
    name : str
        Name of the new array column (appended last).
    dtype : polars float dtype, default ``pl.Float32``
        Storage dtype. Values are gathered in float64 and cast once, here.

    Returns
    -------
    polars.DataFrame
        ``df`` without ``cols``, plus ``name``. Nulls become NaN (an array cell
        is a float, and NaN is this module's "no observation").

    Raises
    ------
    ValueError
        If ``cols`` is empty, names a missing column, or ``name`` collides with
        a column that is kept.
    """
    cols = list(cols)
    if not cols:
        raise ValueError("as_embedding: `cols` must name at least one column.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"as_embedding: column(s) {missing} not found.")
    if name in df.columns and name not in cols:
        raise ValueError(f"as_embedding: a column named {name!r} already exists.")
    block = (
        df.select([pl.col(c).cast(pl.Float64) for c in cols])
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    arr = pl.Series(name, block, dtype=pl.Array(pl.Float64, len(cols))).cast(
        pl.Array(dtype, len(cols))
    )
    return df.drop(cols).with_columns(arr)


def explode_embedding(
    df: pl.DataFrame,
    name: str,
    *,
    prefix: str | None = None,
) -> pl.DataFrame:
    """Unpack a fixed-width array column into ``D`` loose columns.

    Parameters
    ----------
    df : polars.DataFrame
    name : str
        The ``pl.Array`` column to unpack. It is dropped from the result.
    prefix : str, optional
        New columns are ``{prefix}_0 .. {prefix}_{D-1}``; default ``name``.

    Returns
    -------
    polars.DataFrame
        The inner dtype is kept (a ``Float32`` embedding explodes to ``Float32``
        columns).

    Raises
    ------
    ValueError
        If ``name`` is missing or is not a fixed-width ``pl.Array`` column.
    """
    if name not in df.columns:
        raise ValueError(f"explode_embedding: column {name!r} not found.")
    dt = df.schema[name]
    if not isinstance(dt, pl.Array):
        raise ValueError(
            f"explode_embedding: column {name!r} has dtype {dt}, not a fixed-width "
            "pl.Array; use `.list` operations for variable-width lists."
        )
    width = int(dt.size)
    pre = name if prefix is None else prefix
    return df.with_columns(
        [pl.col(name).arr.get(i).alias(f"{pre}_{i}") for i in range(width)]
    ).drop(name)
