"""Frame plumbing: extract a panel **once** as time-sorted numpy arrays, and the
fixed result schema every analysis function returns.

Extraction sorts by ``(entity, time)`` a single time and slices entities by the
offsets of their runs -- it never ``partition_by``\\ s into thousands of small
frames. Float32 columns are upcast to float64; nulls become NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame

__all__ = ["RESULT_SCHEMA", "PanelArrays", "extract", "result_frame"]

#: The fixed schema of every analysis result (irrelevant fields are null), so
#: ``pl.concat`` across methods, nulls and transforms just works.
RESULT_SCHEMA: dict[str, Any] = {
    "estimate": pl.Float64,
    "p_value": pl.Float64,
    "p_value_adj": pl.Float64,
    "method": pl.Utf8,
    "estimator": pl.Utf8,
    "null_method": pl.Utf8,
    "n_resamples": pl.Int64,
    "block_length": pl.Int64,
    "direction": pl.Utf8,
    "lag": pl.Int64,
    "n_obs": pl.Int64,
    "n_entities": pl.Int64,
    "coverage": pl.Float64,
    "heterogeneity": pl.Float64,
    "transform": pl.Utf8,
    "approximate": pl.Boolean,
    "seed": pl.Int64,
    "warnings": pl.List(pl.Utf8),
}


def result_frame(
    rows: list[dict[str, Any]], keys: dict[str, Any] | None = None
) -> pl.DataFrame:
    """Build a result frame: ``keys`` columns first, then :data:`RESULT_SCHEMA`."""
    schema = {**(keys or {}), **RESULT_SCHEMA}
    data: dict[str, list[Any]] = {name: [] for name in schema}
    for row in rows:
        for name in schema:
            val = row.get(name)
            if name == "warnings":
                val = list(val) if val else []
            elif (
                isinstance(val, float)
                and not np.isfinite(val)
                and schema[name] != pl.Float64
            ):
                val = None
            elif isinstance(val, (np.floating, np.integer, np.bool_)):
                val = val.item()
            data[name].append(val)
    return pl.DataFrame(data, schema=schema)


@dataclass
class PanelArrays:
    """A panel extracted once as numpy arrays, sorted by ``(entity, time)``.

    Attributes
    ----------
    entities : list
        Entity labels in sorted order (``[None]`` for a single series).
    offsets : numpy.ndarray
        ``(N + 1,)``; entity ``i`` occupies rows ``offsets[i]:offsets[i+1]``.
    tcode : numpy.ndarray
        ``(n,)`` dense index into the shared sorted date axis. Without a time
        column it is the **within-entity row position** (``has_time`` is then
        False and a common-time null is refused).
    n_times : int
        Length of the shared date axis.
    values : dict of str -> numpy.ndarray
        float64 columns (NaN for null).
    entity_col, time_col : str or None
    """

    entities: list[Any]
    offsets: np.ndarray
    tcode: np.ndarray
    n_times: int
    values: dict[str, np.ndarray] = field(default_factory=dict)
    entity_col: str | None = None
    time_col: str | None = None
    times: list[Any] = field(default_factory=list)

    @property
    def has_time(self) -> bool:
        return self.time_col is not None

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    def slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def dense(self, col: str) -> np.ndarray:
        """``(N, T)`` layout of ``col`` on the shared date axis (NaN if absent)."""
        out = np.full((self.n_entities, self.n_times), np.nan)
        ecode = np.repeat(np.arange(self.n_entities), np.diff(self.offsets))
        out[ecode, self.tcode] = self.values[col]
        return out


def _resolve(
    df: Any, entity: str | None, time: str | None
) -> tuple[pl.DataFrame, str | None, str | None]:
    if isinstance(df, PanelFrame):
        if entity is not None and entity != df.entity_col:
            raise ValueError(
                f"`entity={entity!r}` conflicts with the PanelFrame's keys."
            )
        if time is not None and time != df.time_col:
            raise ValueError(f"`time={time!r}` conflicts with the PanelFrame's keys.")
        return df.collect(), df.entity_col, df.time_col
    if isinstance(df, pl.LazyFrame):
        return df.collect(), entity, time
    if isinstance(df, pl.DataFrame):
        return df, entity, time
    raise TypeError(
        f"expected a polars DataFrame/LazyFrame or a PanelFrame, got {type(df).__name__!r}."
    )


def extract(
    df: Any,
    cols: list[str],
    *,
    entity: str | None = None,
    time: str | None = None,
) -> PanelArrays:
    """Extract ``cols`` as float64 arrays sorted by ``(entity, time)``.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
        The data. A PanelFrame supplies its own keys.
    cols : list of str
        Numeric columns to extract.
    entity, time : str, optional
        Keys for a bare frame. Without ``entity`` the frame is one series;
        without ``time`` its row order is taken as the time order.

    Returns
    -------
    PanelArrays

    Raises
    ------
    ValueError
        On a missing column or duplicate ``(entity, time)`` keys.
    """
    frame, ent, tme = _resolve(df, entity, time)
    missing = [
        c for c in [*cols, *(k for k in (ent, tme) if k)] if c not in frame.columns
    ]
    if missing:
        raise ValueError(f"column(s) {missing} not found; available: {frame.columns}.")
    uniq = list(dict.fromkeys(cols))
    sort_keys = [k for k in (ent, tme) if k is not None]
    work = frame.select([*(k for k in (ent, tme) if k is not None), *uniq])
    if sort_keys:
        work = work.sort(sort_keys, maintain_order=True)
    if tme is not None:
        keys = [k for k in (ent, tme) if k is not None]
        if work.select(pl.struct(keys).is_duplicated().any()).item():
            raise ValueError(f"duplicate {tuple(keys)!r} keys in the panel.")
    values = {
        c: work.get_column(c)
        .cast(pl.Float64)
        .fill_null(float("nan"))
        .to_numpy()
        .astype(np.float64, copy=True)
        for c in uniq
    }
    n = work.height
    if ent is None:
        entities: list[Any] = [None]
        offsets = np.array([0, n], dtype=np.int64)
    else:
        e = work.get_column(ent)
        change = (e != e.shift(1)).fill_null(True).to_numpy()
        starts = np.flatnonzero(change)
        entities = e.gather(starts).to_list()
        offsets = np.append(starts, n).astype(np.int64)
    times: list[Any] = []
    if tme is not None:
        tser = work.get_column(tme)
        tcode = tser.rank("dense").to_numpy().astype(np.int64) - 1
        n_times = int(tcode.max()) + 1 if n else 0
        times = tser.unique(maintain_order=False).sort().to_list()
    else:
        sizes = np.diff(offsets)
        tcode = np.arange(n, dtype=np.int64) - np.repeat(offsets[:-1], sizes)
        n_times = int(sizes.max()) if n else 0
    return PanelArrays(entities, offsets, tcode, n_times, values, ent, tme, times)
