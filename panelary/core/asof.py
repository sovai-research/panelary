"""Point-in-time (bitemporal) as-of join: what was *known* at each panel date.

A fundamentals table, an estimates table or a macro series is not one value per
period. It is a history of **vintages**: the figure for a period is first
published some time after the period ends, and may later be revised or
restated. A vintage table therefore has two clocks:

``event_time``
    What the value is *about* -- the fiscal period, the reference month.
``knowledge_time``
    When the value became *known* -- the filing, the release, the restatement.

A backtest that joins such a table on ``event_time`` alone ("the Q1 value") or
takes the latest row per period ("the restated Q1 value") uses figures nobody
had on the date being simulated. That is look-ahead, and it is the most common
kind in practice because the join *looks* correct. :func:`asof_join` is the
point-in-time version:

    each panel row ``(entity, t)`` receives, among the vintages of that entity
    with ``knowledge_time + lag <= t`` **and** ``event_time <= t``, the one with
    the latest ``event_time`` -- and, for that ``event_time``, the latest
    ``knowledge_time``: the most recent period, as it was known at ``t``.

Restatements are handled by construction: a restated Q1 figure published in
August is invisible to every panel date before August, where the original is
returned, and is returned from August on. A later *period* always beats an
earlier one, so a Q1 restatement filed after Q2 was published does not displace
Q2 as "the latest figure" -- it only replaces Q1's value for callers who ask
about Q1.

Correctness contract
--------------------
* **Strictly causal.** The value at ``t`` is a function of the vintages with
  ``max(knowledge_time + lag, event_time) <= t`` only. Perturbing or deleting
  any vintage that becomes eligible after ``t`` cannot change it.
* **Prefix invariant** (hard invariant 1). The value at ``(entity, t)`` does not
  depend on how many panel rows follow it, nor on vintages known after ``t``.
* **Order independent.** Neither the panel nor the vintage table needs to be
  sorted; rows are returned in the panel's original order. The join sorts
  internally on explicit keys, never on frame position, so no
  :class:`~panelary.core.panel_frame.PanelOrderWarning` applies.
* **Fail closed.** A vintage with a null ``entity`` / ``event_time`` /
  ``knowledge_time`` cannot be placed in time and is rejected, not guessed.
  Two different vintages with the same ``(entity, event_time, knowledge_time)``
  are ambiguous and rejected (exact duplicate rows are collapsed). Time columns
  must share a dtype family with the panel's time column: a ``Datetime``
  knowledge time is *not* silently truncated to a ``Date`` panel (that would
  make a 16:30 filing usable at that day's open), and a ``Date`` knowledge time
  is not silently promoted to midnight on a ``Datetime`` panel.

Column roles default to ``(entity, event_time, knowledge_time, value...)``,
the shape of the ``vintages`` table :func:`panelary.synth.generate_panel`
emits (its extra ``revision`` column comes along as a value unless ``values=``
narrows the selection).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import polars as pl

from panelary.core._calendar import (
    BusinessDays,
    Duration,
    as_steps,
    is_steps,
    shift_forward,
    validate_duration,
)
from panelary.core.panel_frame import PanelFrame

__all__ = ["BusinessDays", "asof_join"]

_Frame = TypeVar("_Frame", PanelFrame, pl.DataFrame, pl.LazyFrame)

# Internal column names. Double underscores keep them out of user namespaces;
# every one is dropped before returning.
_KEY = "__asof_key__"
_KIND = "__asof_kind__"
_RANK = "__asof_rank__"
_ROW = "__asof_row__"
_BEST = "__asof_best__"
_EV = "__asof_event__"

_UNIT_ORDER = {"ms": 0, "us": 1, "ns": 2}


def _time_family(dtype: pl.DataType) -> str:
    if dtype.is_integer():
        return "integer"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "datetime"
    return "other"


def _common_time_dtype(panel_dtype: pl.DataType, **others: pl.DataType) -> pl.DataType:
    """The dtype every time key is compared in, or a TypeError naming the clash.

    Integers compare as ``Int64``. ``Date`` only with ``Date``. ``Datetime``
    only with ``Datetime`` of the *same* time zone, compared in the finest unit
    present -- casting to a coarser unit would floor a knowledge time and could
    make a value usable a fraction of a second early.
    """
    fam = _time_family(panel_dtype)
    if fam == "other":
        raise TypeError(
            f"asof_join: the panel's time column has dtype {panel_dtype}; an "
            "as-of join needs an integer, Date or Datetime time axis."
        )
    for name, dtype in others.items():
        if _time_family(dtype) != fam:
            hint = ""
            if {fam, _time_family(dtype)} == {"date", "datetime"}:
                hint = (
                    " A Date and a Datetime are not silently reconciled: "
                    "truncating a Datetime knowledge time to a Date makes a "
                    "value usable before the moment it was published, and "
                    "promoting a Date to midnight claims a precision the data "
                    "does not have. Convert explicitly (and consider `lag=`)."
                )
            raise TypeError(
                f"asof_join: `{name}` has dtype {dtype} but the panel's time "
                f"column is {panel_dtype}; they must be the same kind of time.{hint}"
            )
    if fam == "integer":
        return pl.Int64()
    if fam == "date":
        return pl.Date()
    dtypes = [panel_dtype, *others.values()]
    zones = {getattr(d, "time_zone", None) for d in dtypes}
    if len(zones) > 1:
        raise TypeError(
            f"asof_join: time columns carry different time zones {sorted(map(str, zones))}; "
            "convert them to one zone first (`.dt.convert_time_zone`)."
        )
    unit = max(
        (getattr(d, "time_unit", "us") or "us" for d in dtypes),
        key=_UNIT_ORDER.__getitem__,
    )
    return pl.Datetime(time_unit=unit, time_zone=zones.pop())  # type: ignore[arg-type]


def _as_lazy(frame: Any, what: str) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    if isinstance(frame, PanelFrame):
        return frame.lazy()
    raise TypeError(
        f"asof_join: `{what}` must be a polars DataFrame/LazyFrame"
        + (" or PanelFrame" if what == "panel" else "")
        + f", got {type(frame).__name__!r}."
    )


def asof_join(
    panel: _Frame,
    vintages: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    event_time: str = "event_time",
    knowledge_time: str = "knowledge_time",
    vintage_entity: str | None = None,
    values: str | Sequence[str] | None = None,
    lag: int | Duration | None = None,
    suffix: str = "",
    provenance: bool = False,
) -> _Frame:
    """Attach, to every panel row, the vintage values that were known at its time.

    For each panel row ``(entity, t)`` the candidates are the vintages of that
    entity that were **available** (``knowledge_time + lag <= t``) and are
    **about the past** (``event_time <= t``). The row receives the candidate
    with the latest ``event_time``, and among vintages of that ``event_time``
    the latest ``knowledge_time`` -- the newest period, as known at ``t``. With
    no candidate the values are null.

    Parameters
    ----------
    panel : PanelFrame, polars.DataFrame or polars.LazyFrame
        The panel to enrich. A bare frame needs ``entity`` and ``time`` (or
        follows the column-0 / column-1 convention of
        :func:`~panelary.core.panel_frame.as_panel`). The result has the same
        type, the same rows in the same order, and the value columns appended.
    vintages : polars.DataFrame or polars.LazyFrame
        The bitemporal table: one row per ``(entity, event_time,
        knowledge_time)`` with the value columns alongside. Need not be sorted.
    entity, time : str, optional
        Panel key columns, when ``panel`` is a bare frame.
    event_time : str, default="event_time"
        Column in ``vintages``: what period the value is about.
    knowledge_time : str, default="knowledge_time"
        Column in ``vintages``: when the value became known (the vintage).
    vintage_entity : str, optional
        Entity column in ``vintages``; defaults to the panel's entity column
        name.
    values : str or sequence of str, optional
        Value columns to attach; default every ``vintages`` column that is not
        a key.
    lag : int or duration, optional
        Publication lag added to ``knowledge_time`` before comparing with
        ``t``: a vintage known at ``k`` is usable from ``k + lag``. An ``int``
        on an integer time axis; on a ``Date``/``Datetime`` axis a
        ``datetime.timedelta``, a Polars duration string (``"1d"``,
        ``"1mo"``) or business days (``"2bd"``,
        :class:`~panelary.core._calendar.BusinessDays`). Use it when
        ``knowledge_time`` is the event of publication but the panel's ``t``
        is, say, the market open of the same day.
    suffix : str, default=""
        Appended to every attached column name (values and provenance), to
        avoid collisions with panel columns.
    provenance : bool, default=False
        Also attach the ``event_time`` and ``knowledge_time`` of the vintage
        each value came from (named after those columns, plus ``suffix``), so
        every value carries the identity of its source vintage.

    Returns
    -------
    PanelFrame, polars.DataFrame or polars.LazyFrame
        Same type as ``panel``; its rows and columns, in order, plus the
        attached columns. A PanelFrame keeps its keys and what it knew about
        its row order.

    Raises
    ------
    TypeError
        If a frame has the wrong type, the time columns are of incompatible
        kinds (see the module notes), or ``lag`` does not fit the time axis.
    ValueError
        If a column is missing, a key column of ``vintages`` has nulls, two
        different vintages share ``(entity, event_time, knowledge_time)``, an
        attached column would overwrite a panel column, or ``lag`` is negative.

    Notes
    -----
    Implementation: each vintage becomes *eligible* at ``max(knowledge_time +
    lag, event_time)``, which turns the two-sided condition into a single
    prefix condition. Vintages are ranked by ``(event_time, knowledge_time)``;
    panel rows and eligibility events are merged per entity in time order
    (events first on ties, so eligibility at exactly ``t`` counts), and a
    running maximum of the rank selects the vintage. Everything is keyed on
    explicit columns, so input row order is irrelevant. ``O((n + m) log(n +
    m))``; the vintage key columns are collected once, the panel side stays
    lazy if it was lazy.

    Examples
    --------
    >>> import datetime as dt
    >>> import polars as pl
    >>> from panelary.core.asof import asof_join
    >>> panel = pl.DataFrame(
    ...     {
    ...         "ticker": ["A"] * 4,
    ...         "date": [dt.date(2024, m, 15) for m in (4, 6, 8, 9)],
    ...     }
    ... )
    >>> vintages = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "A", "A"],
    ...         "event_time": [dt.date(2024, 3, 31), dt.date(2024, 3, 31), dt.date(2024, 6, 30)],
    ...         "knowledge_time": [dt.date(2024, 5, 1), dt.date(2024, 9, 1), dt.date(2024, 8, 1)],
    ...         "eps": [1.00, 0.90, 1.10],  # Q1, Q1 restated in September, Q2
    ...     }
    ... )
    >>> asof_join(panel, vintages, entity="ticker", time="date")["eps"].to_list()
    [None, 1.0, 1.1, 1.1]
    """
    # ---- the panel side -------------------------------------------------- #
    if isinstance(panel, PanelFrame):
        pf = panel
    else:
        from panelary.core.panel_frame import as_panel

        pf = as_panel(_as_lazy(panel, "panel"), entity=entity, time=time)
    ecol, tcol = pf.entity_col, pf.time_col
    left = pf.lazy()
    left_schema = left.collect_schema()

    # ---- the vintage side: columns, roles, dtypes ------------------------ #
    right = _as_lazy(vintages, "vintages")
    vschema = right.collect_schema()
    ventity = vintage_entity if vintage_entity is not None else ecol
    for role, col in (
        ("vintage_entity", ventity),
        ("event_time", event_time),
        ("knowledge_time", knowledge_time),
    ):
        if col not in vschema:
            raise ValueError(
                f"asof_join: `{role}` column {col!r} is not in the vintage table "
                f"(columns: {vschema.names()})."
            )
    if len({ventity, event_time, knowledge_time}) != 3:
        raise ValueError(
            "asof_join: `vintage_entity`, `event_time` and `knowledge_time` must "
            "be three different columns."
        )
    keys = {ventity, event_time, knowledge_time}
    if values is None:
        value_cols = [c for c in vschema.names() if c not in keys]
    else:
        value_cols = [values] if isinstance(values, str) else list(values)
        missing = [c for c in value_cols if c not in vschema]
        if missing:
            raise ValueError(
                f"asof_join: value column(s) {missing} not in the vintage table."
            )
        if keys & set(value_cols):
            raise ValueError(
                f"asof_join: {sorted(keys & set(value_cols))} are key columns and "
                "cannot also be value columns; use `provenance=True` to attach "
                "the vintage's times."
            )
    if not value_cols and not provenance:
        raise ValueError("asof_join: the vintage table has no value columns to attach.")

    out_names = {c: f"{c}{suffix}" for c in value_cols}
    if provenance:
        out_names[_EV] = f"{event_time}{suffix}"
        out_names["__asof_known__"] = f"{knowledge_time}{suffix}"
    clash = sorted(set(out_names.values()) & set(left_schema.names()))
    if clash or len(set(out_names.values())) != len(out_names):
        raise ValueError(
            f"asof_join: attaching {sorted(out_names.values())} would overwrite "
            f"panel column(s) {clash or '(duplicate output names)'}; pass "
            "`suffix=` (e.g. suffix='_pit')."
        )

    key_dtype = _common_time_dtype(
        left_schema[tcol],
        event_time=vschema[event_time],
        knowledge_time=vschema[knowledge_time],
    )
    lag_spec = None if lag is None else validate_duration(lag, name="lag")
    if lag_spec is not None and is_steps(lag_spec) and as_steps(lag_spec) == 0:
        lag_spec = None

    # ---- vintages: validate, dedupe, rank, eligibility ------------------- #
    ent_dtype = left_schema[ecol]
    vt = (
        right.select(
            pl.col(ventity).cast(ent_dtype, strict=True).alias(ecol),
            pl.col(event_time),
            pl.col(knowledge_time),
            *[pl.col(c) for c in value_cols],
        )
        .unique(maintain_order=False)
        .collect()
    )
    nulls = {
        c: int(n)
        for c, n in zip(
            (ventity, event_time, knowledge_time),
            vt.select(ecol, event_time, knowledge_time).null_count().row(0),
            strict=True,
        )
        if n
    }
    if nulls:
        raise ValueError(
            f"asof_join: vintage key column(s) contain nulls {nulls}. A vintage "
            "with no entity, event time or knowledge time cannot be placed in "
            "time, and treating it as 'always known' would leak; drop or repair "
            "those rows first."
        )
    dupes = (
        vt.group_by(ecol, event_time, knowledge_time).len().filter(pl.col("len") > 1)
    )
    if dupes.height:
        sample = dupes.head(3).drop("len").rows()
        raise ValueError(
            f"asof_join: {dupes.height} (entity, event_time, knowledge_time) "
            "key(s) carry different values, so which one was known is "
            f"ambiguous. First offenders: {sample}. De-duplicate the vintage "
            "table (exact duplicate rows are collapsed automatically)."
        )

    known = vt.get_column(knowledge_time)
    available = shift_forward(known, lag_spec) if lag_spec is not None else known
    vt = (
        vt.with_columns(
            available.cast(key_dtype).alias("__asof_available__"),
            pl.col(event_time).cast(key_dtype).alias(_EV + "_key"),
        )
        # Rank = lexicographic (event_time, knowledge_time) order, which is the
        # "newest period, then newest vintage of it" preference. Keys are unique
        # after the duplicate check, so the rank is a total order.
        .sort(ecol, event_time, knowledge_time)
        .with_row_index(_RANK)
        .with_columns(
            pl.max_horizontal("__asof_available__", _EV + "_key").alias(_KEY),
            pl.lit(0, dtype=pl.Int8).alias(_KIND),
        )
    )
    payload = vt.select(
        pl.col(_RANK),
        *[pl.col(c).alias(out_names[c]) for c in value_cols],
        *(
            [
                pl.col(event_time).alias(out_names[_EV]),
                pl.col(knowledge_time).alias(out_names["__asof_known__"]),
            ]
            if provenance
            else []
        ),
    ).lazy()
    # ---- merge: eligibility events and panel rows, per entity, in time ----- #
    indexed = left.with_row_index(_ROW)
    # The index dtype is UInt32, or UInt64 on a big-index Polars build; read it
    # rather than assume it, so the two halves of the merge share a schema.
    idx_dtype = vt.schema[_RANK]
    events = vt.lazy().select(
        pl.col(ecol),
        pl.col(_KEY),
        pl.col(_KIND),
        pl.col(_RANK),
        pl.lit(None, dtype=idx_dtype).alias(_ROW),
    )
    rows = indexed.select(
        pl.col(ecol),
        pl.col(tcol).cast(key_dtype).alias(_KEY),
        pl.lit(1, dtype=pl.Int8).alias(_KIND),
        pl.lit(None, dtype=idx_dtype).alias(_RANK),
        pl.col(_ROW).cast(idx_dtype),
    )
    chosen = (
        pl.concat(
            [events, rows.select(events.collect_schema().names())], how="vertical"
        )
        # Events sort before panel rows at the same key, so "eligible at t" is
        # included at t. Rank is unique, so the order is total and the result
        # never depends on input row order.
        .sort(ecol, _KEY, _KIND, _RANK, _ROW, nulls_last=True)
        .with_columns(pl.col(_RANK).cum_max().forward_fill().over(ecol).alias(_BEST))
        .filter(pl.col(_KIND) == 1)
        # A panel row with no time sorts last within its entity; it is at no
        # point in time, so it must not inherit the entity's latest vintage.
        .select(
            _ROW,
            pl.when(pl.col(_KEY).is_null())
            .then(None)
            .otherwise(pl.col(_BEST))
            .alias(_BEST),
        )
    )
    result = (
        indexed.with_columns(pl.col(_ROW).cast(idx_dtype))
        .join(chosen, on=_ROW, how="left")
        .join(payload, left_on=_BEST, right_on=_RANK, how="left")
        .sort(_ROW)
        .drop(_ROW, _BEST)
    )

    if isinstance(panel, PanelFrame):
        # Row order is the panel's own, so what it knew about that order holds.
        return panel._rewrap(result, order=panel.sortedness)
    if isinstance(panel, pl.LazyFrame):
        return result
    return result.collect()
