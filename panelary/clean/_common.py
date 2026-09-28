"""Shared plumbing for :mod:`panelary.clean`: column resolution, scopes, order.

Kept free of any heavy import; everything here is polars + numpy.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel

#: Temporary row-id column added while a frame is being cleaned. The name is
#: deliberately unlikely to collide with user columns.
ROW = "__panelary_row"

#: Polars >= 1.3x warns unless ``explode`` is told how to treat empty lists;
#: older releases do not know the keyword. Every caller here drops the
#: resulting nulls, so the old behaviour (empty -> null) is requested.
_EXPLODE_KW: dict[str, Any] = (
    {"empty_as_null": True}
    if "empty_as_null" in inspect.signature(pl.DataFrame.explode).parameters
    else {}
)

Scope = Literal["global", "entity", "time", "key"]
_SCOPES: tuple[str, ...] = ("global", "entity", "time", "key")


def explode(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """``frame.explode(column)`` with empty lists mapped to null, warning-free."""
    return frame.explode(column, **_EXPLODE_KW)


def check_choice(name: str, value: object, choices: Sequence[object]) -> None:
    """Raise a uniform ``ValueError`` when ``value`` is not one of ``choices``."""
    if value not in choices:
        raise ValueError(f"`{name}` must be one of {list(choices)}, got {value!r}.")


def check_unit_interval(name: str, value: float, *, closed_low: bool = False) -> None:
    """Raise unless ``value`` lies in ``(0, 1]`` (or ``[0, 1]``)."""
    ok = (0.0 <= value <= 1.0) if closed_low else (0.0 < value <= 1.0)
    if not ok:
        span = "[0, 1]" if closed_low else "(0, 1]"
        raise ValueError(f"`{name}` must be in {span}, got {value!r}.")


def scope_columns(scope: str, entity: str, time: str) -> list[str]:
    """The key columns two rows must share to be compared under ``scope``.

    ``"global"`` compares every row with every other (panel-global);
    ``"entity"`` only within an entity; ``"time"`` only within a date (a
    cross-section); ``"key"`` only within one ``(entity, time)`` cell.
    """
    check_choice("scope", scope, _SCOPES)
    return {
        "global": [],
        "entity": [entity],
        "time": [time],
        "key": [entity, time],
    }[scope]


def resolve_columns(
    names: Sequence[str],
    entity: str,
    time: str,
    columns: Sequence[str] | str | None,
    *,
    what: str = "content",
) -> list[str]:
    """Resolve the content columns: explicit list, or every non-key column."""
    if columns is None:
        cols = [c for c in names if c not in (entity, time) and c != ROW]
    else:
        cols = [columns] if isinstance(columns, str) else list(columns)
        missing = [c for c in cols if c not in names]
        if missing:
            raise ValueError(
                f"{what} column(s) {missing} not found. Available columns: "
                f"{list(names)}."
            )
    return cols


def to_panel(
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    entity: str | None,
    time: str | None,
) -> PanelFrame:
    """Wrap ``X`` as a :class:`PanelFrame` (column 0/1 fallback, as elsewhere)."""
    return as_panel(X, entity, time)


def rewrap(template: Any, out: pl.DataFrame | pl.LazyFrame, panel: PanelFrame) -> Any:
    """Return ``out`` in the same container type the caller passed in."""
    if isinstance(template, PanelFrame):
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )
    if isinstance(template, pl.LazyFrame):
        return out.lazy()
    return out.collect() if isinstance(out, pl.LazyFrame) else out


def with_row_ids(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the :data:`ROW` id column (``Int64``, ``0 .. n - 1``) at the front."""
    if ROW in frame.columns:
        raise ValueError(f"column name {ROW!r} is reserved by panelary.clean.")
    return frame.with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64))


def order_ranks(frame: pl.DataFrame, time: str, *, reverse: bool = False) -> np.ndarray:
    """Rank of every row in point-in-time order: by ``time``, then input order.

    ``frame`` must carry :data:`ROW` as ``0 .. n - 1``. Returns an ``int64``
    array ``rank`` with ``rank[row]`` the row's position; ``reverse=True``
    ranks latest-first (used by ``keep="last"``).
    """
    by = frame.select(ROW, time).sort(
        [time, ROW], descending=[reverse, reverse], nulls_last=not reverse
    )
    rows = by.get_column(ROW).to_numpy().astype(np.int64)
    rank = np.empty(rows.size, dtype=np.int64)
    rank[rows] = np.arange(rows.size, dtype=np.int64)
    return rank


def group_codes(frame: pl.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Integer code per row identifying its group over ``cols`` (null-equal).

    The code is the smallest :data:`ROW` id in the group; with no ``cols``
    every row gets code 0.
    """
    if not cols:
        return np.zeros(frame.height, dtype=np.int64)
    return (
        frame.select(pl.col(ROW).min().over(list(cols)))
        .to_series()
        .to_numpy()
        .astype(np.int64)
    )
