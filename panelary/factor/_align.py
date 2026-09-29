"""Leak-safe forward-return alignment — the single audited negative-shift site.

Forward-return alignment is the #1 caller-side leakage trap in cross-sectional
backtests: a signal observed at time ``t`` must be paired with the return
*earned after* ``t`` (over ``(t, t+h]``), and that pairing is the one place a
**backward** per-entity shift (``shift(-h)``) is legitimate. Isolating it here
means every downstream evaluator (:mod:`~panelary.factor.ic`,
:mod:`~panelary.factor.portfolio`) consumes an already-aligned frame and
never re-derives a forward return — so there is exactly one line to audit.

Two safety properties are enforced:

* **backward shift only** — the forward return at row ``t`` is
  ``col.shift(-horizon).over(entity)``; the last ``horizon`` rows of each entity
  become null (no fabricated future), never a positive shift.
* **gap guard** — unless ``allow_gaps=True``, the per-entity time grid must be
  regular, so a shift can never silently jump across a missing period (which
  would align ``t`` with a return from ``t + h + gap``).
"""

from __future__ import annotations

import polars as pl

__all__ = ["forward_return"]

FrameT = pl.LazyFrame | pl.DataFrame


def _check_no_gaps(frame: FrameT, *, entity: str, time: str) -> None:
    """Raise if the per-entity ``time`` grid is irregular (has gaps).

    Computes the per-entity consecutive time delta; a regular grid has exactly
    one distinct positive delta. More than one distinct delta means at least one
    entity spans a missing period, so a backward shift would leak a return from
    beyond the intended horizon.
    """
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    deltas = (
        lf.sort([entity, time])
        .select(pl.col(time).diff().over(entity).alias("__delta__"))
        .drop_nulls()
        .unique()
        .collect()
        .get_column("__delta__")
    )
    if deltas.len() > 1:
        raise ValueError(
            "forward_return: irregular per-entity time grid detected "
            f"(distinct time deltas: {sorted(deltas.to_list())}). A backward "
            "shift across a missing period would align a signal with a return "
            "from beyond the intended horizon (a leak). Fill the panel to a "
            "regular grid, or pass allow_gaps=True to bypass this guard."
        )


def forward_return(
    frame: FrameT,
    *,
    entity: str,
    time: str,
    price: str | None = None,
    ret: str | None = None,
    horizon: int = 1,
    out: str = "fwd_ret",
    allow_gaps: bool = False,
    end_time: str | None = None,
) -> FrameT:
    """Attach the leak-safe forward return over ``(t, t+horizon]`` to each row.

    Exactly one of ``price`` or ``ret`` must be given:

    * ``ret`` (a same-period return): the forward return at ``t`` is
      ``col(ret).shift(-horizon).over(entity)`` — the return realised at
      ``t + horizon``.
    * ``price``: the forward return is
      ``price.shift(-horizon) / price - 1`` per entity — the holding-period
      return from ``t`` to ``t + horizon``.

    The frame is sorted by ``[entity, time]`` (returned sorted). The last
    ``horizon`` rows of each entity carry null (no future exists to align).

    Parameters
    ----------
    frame : LazyFrame | DataFrame
        Long-format panel. Returned as the same type.
    entity, time : str, keyword-only
        Entity and time key columns.
    price, ret : str | None, keyword-only
        Source column; pass exactly one.
    horizon : int, keyword-only, default 1
        Forward horizon (in periods of the regular grid); integer ``>= 1``.
    out : str, keyword-only, default "fwd_ret"
        Name of the forward-return column to write.
    allow_gaps : bool, keyword-only, default False
        If ``False`` (default), raise when the per-entity time grid is irregular
        (a shift could jump a gap and leak). Set ``True`` only when the grid is
        deliberately irregular and you accept the risk.
    end_time : str | None, keyword-only, default None
        If given, also write the time ``horizon`` rows ahead -- the label end
        time ``t1`` that purged cross-validation consumes -- under this name,
        in the time column's dtype (null where ``out`` is null). It comes
        from the same backward shift, so the library keeps one audited
        negative-shift site.

    Returns
    -------
    LazyFrame | DataFrame
        The input frame, sorted by ``[entity, time]``, with the ``out`` column
        (and ``end_time`` when requested).
    """
    if (price is None) == (ret is None):
        raise ValueError(
            "forward_return: pass exactly one of `price=` or `ret=` "
            f"(got price={price!r}, ret={ret!r})."
        )
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(
            f"forward_return: `horizon` must be an integer >= 1, got {horizon!r}."
        )

    if not allow_gaps:
        _check_no_gaps(frame, entity=entity, time=time)

    ordered = frame.sort([entity, time])
    if ret is not None:
        fwd = pl.col(ret).shift(-horizon).over(entity)
    else:
        fwd = (pl.col(price).shift(-horizon) / pl.col(price) - 1.0).over(entity)
    if end_time is None:
        return ordered.with_columns(fwd.alias(out))
    end = pl.col(time).shift(-horizon).over(entity)
    return ordered.with_columns(fwd.alias(out), end.alias(end_time))
