"""Panel invariants: the structural contract every panel operation assumes.

Each function takes an eager :class:`polars.DataFrame` plus the panel keys and
returns one :class:`~panelary.quality._common.CheckResult`. They are
deterministic identity checks -- nothing is learned -- so, per the cleaning
doctrine, they run on **all** rows, before any split.

* :func:`check_key_columns` -- entity and time exist, differ, and time is
  orderable (the :class:`~panelary.core.panel_frame.PanelFrame` schema
  contract, reused rather than restated).
* :func:`check_null_keys` -- no null entity or time.
* :func:`check_unique_keys` -- every ``(entity, time)`` pair occurs once.
* :func:`check_monotone_time` -- time is non-decreasing within every entity *in
  frame order* (the ``panel_safe`` precondition; measured by
  :meth:`PanelFrame.is_sorted_per_entity`).
* :func:`check_gaps` -- no internal holes in an entity's time axis, against the
  panel's own calendar or an explicit ``frequency``.
* :func:`check_min_obs` / :func:`check_coverage` -- enough history per entity.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import polars as pl

from panelary.quality._common import (
    EVIDENCE_SCHEMA,
    CheckResult,
    Impact,
    as_panel_frame,
    key_records,
    ratio,
)

__all__ = [
    "check_coverage",
    "check_gaps",
    "check_key_columns",
    "check_min_obs",
    "check_monotone_time",
    "check_null_keys",
    "check_unique_keys",
]

#: Frequency accepted by :func:`check_gaps`: a polars duration string or a
#: :class:`datetime.timedelta` for a temporal time axis, a positive number for
#: a numeric one.
Frequency = str | int | float | _dt.timedelta


def _entities(n: int) -> str:
    return f"{n} entity" if n == 1 else f"{n} entities"


def _have(n: int) -> str:
    return "has" if n == 1 else "have"


def _cover(n: int) -> str:
    return "covers" if n == 1 else "cover"


def _entity_mask(df: pl.DataFrame, entity: str, offenders: pl.Series) -> pl.Series:
    """Per-row flag: does the row belong to one of ``offenders``?"""
    return df.select(
        pl.col(entity).is_in(offenders.implode()).fill_null(False)
    ).to_series()


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def check_key_columns(df: pl.DataFrame, entity: str, time: str) -> CheckResult:
    """The entity / time schema contract, via :class:`PanelFrame` itself.

    Always ``impact="fail"``: without usable keys no other panel check means
    anything.
    """
    from panelary.core.panel_frame import PanelFrame

    try:
        PanelFrame(df, entity=entity, time=time, validate=True)
    except (ValueError, TypeError) as exc:
        schema = df.schema
        observed = {
            "columns": df.columns,
            "time_dtype": str(schema[time]) if time in schema else None,
        }
        return CheckResult(
            name="key_columns",
            category="panel",
            impact="fail",
            passed=False,
            message=str(exc),
            n_failing=1,
            unit="columns",
            observed=observed,
            expected={"entity": entity, "time": time, "time_orderable": True},
        )
    return CheckResult(
        name="key_columns",
        category="panel",
        impact="fail",
        passed=True,
        message=f"entity {entity!r} and time {time!r} are present and usable.",
        observed={"entity": entity, "time": time, "time_dtype": str(df.schema[time])},
        expected={"entity": entity, "time": time, "time_orderable": True},
    )


def check_null_keys(
    df: pl.DataFrame, entity: str, time: str, *, impact: Impact, limit: int
) -> CheckResult:
    """No row may have a null entity or a null time."""
    mask = df.select(
        (pl.col(entity).is_null() | pl.col(time).is_null()).alias("m")
    ).to_series()
    n = int(mask.sum())
    offending = key_records(
        df.with_row_index("row").filter(mask),
        [entity, time],
        extra=("row",),
        limit=limit,
    )
    counts = df.select(
        pl.col(entity).null_count().alias("e"), pl.col(time).null_count().alias("t")
    ).row(0)
    return CheckResult(
        name="null_keys",
        category="panel",
        impact=impact,
        passed=n == 0,
        message=(
            "no null entity or time keys."
            if n == 0
            else f"{n} row(s) have a null key ({counts[0]} null {entity!r}, "
            f"{counts[1]} null {time!r}); a row without a key cannot be placed "
            "in the panel."
        ),
        n_failing=n,
        observed={"null_entity": int(counts[0]), "null_time": int(counts[1])},
        expected={"null_entity": 0, "null_time": 0},
        offending=tuple(offending),
        evidence_kind=EVIDENCE_SCHEMA,
        row_mask=mask,
    )


def check_unique_keys(
    df: pl.DataFrame, entity: str, time: str, *, impact: Impact, limit: int
) -> CheckResult:
    """Every ``(entity, time)`` pair must occur exactly once.

    Every row of a duplicated key is flagged (as ``dataframely`` flags a
    primary-key violation): which copy is "right" is a survivorship decision
    for :mod:`panelary.clean`, not for a validator.
    """
    mask = df.select(pl.struct(entity, time).is_duplicated().alias("m")).to_series()
    n_rows = int(mask.sum())
    dup_keys = (
        df.filter(mask)
        .group_by(entity, time)
        .agg(pl.len().cast(pl.Int64).alias("count"))
    )
    n_keys = dup_keys.height
    return CheckResult(
        name="unique_keys",
        category="panel",
        impact=impact,
        passed=n_rows == 0,
        message=(
            f"every ({entity}, {time}) key is unique."
            if n_rows == 0
            else f"{n_keys} ({entity}, {time}) key(s) occur more than once, "
            f"covering {n_rows} row(s). Each key must be unique; a duplicated "
            "row double-counts in every fitted statistic downstream."
        ),
        n_failing=n_rows,
        observed={"duplicated_keys": n_keys, "rows": n_rows},
        expected={"duplicated_keys": 0},
        offending=tuple(
            key_records(dup_keys, [entity, time], extra=("count",), limit=limit)
        ),
        row_mask=mask,
    )


def check_monotone_time(
    df: pl.DataFrame, entity: str, time: str, *, impact: Impact, limit: int
) -> CheckResult:
    """Time must be non-decreasing within every entity, in frame order.

    The verdict comes from :meth:`PanelFrame.is_sorted_per_entity` on a fresh
    frame of unknown order -- a validator measures, it does not trust a
    ``mark_sorted()`` promise -- and :attr:`PanelFrame.sortedness` is reported
    as observed (``"panel"``, ``"time"`` or ``"unsorted"``). The row mask flags
    each row whose time is earlier than the previous row of the same entity.
    """
    pf = as_panel_frame(df, entity, time)
    ordered = pf.is_sorted_per_entity()
    state = pf.sortedness
    if ordered:
        return CheckResult(
            name="monotone_time",
            category="panel",
            impact=impact,
            passed=True,
            message=(
                f"time is non-decreasing within every {entity!r} "
                f"(sortedness={state!r})."
            ),
            observed={"backward_steps": 0, "sortedness": state},
            expected={"backward_steps": 0},
        )
    prev = pl.col(time).shift(1).over(entity)
    flagged = df.with_row_index("row").with_columns(
        prev.alias("previous_time"),
        (pl.col(time) < prev).fill_null(False).alias("__back__"),
    )
    mask = flagged.get_column("__back__")
    n = int(mask.sum())
    offenders = flagged.filter(pl.col("__back__"))
    return CheckResult(
        name="monotone_time",
        category="panel",
        impact=impact,
        passed=False,
        message=(
            f"{n} row(s) go backwards in time relative to the previous row of "
            f"the same {entity!r}. `.over({entity!r})` walks rows in frame "
            "order, so a shift or rolling window on this panel returns a wrong "
            "number rather than an error; sort with `PanelFrame.sort_panel()` "
            "or use `PanelFrame.within_entity`."
        ),
        n_failing=n,
        observed={"backward_steps": n, "sortedness": state},
        expected={"backward_steps": 0},
        offending=tuple(
            key_records(
                offenders, [entity, time], extra=("row", "previous_time"), limit=limit
            )
        ),
        row_mask=mask,
    )


# --------------------------------------------------------------------------- #
# Gaps, history, coverage
# --------------------------------------------------------------------------- #
def _distinct_keys(df: pl.DataFrame, entity: str, time: str) -> pl.DataFrame:
    return df.select(entity, time).drop_nulls().unique().sort(entity, time)


def _duration_string(freq: _dt.timedelta) -> str:
    us = (freq.days * 86_400 + freq.seconds) * 1_000_000 + freq.microseconds
    if us <= 0:
        raise ValueError(f"`frequency` must be a positive duration, got {freq!r}.")
    return f"{us}us"


def _normalise_frequency(freq: Frequency, dtype: pl.DataType) -> str | float:
    """Validate ``freq`` against the time dtype; return its canonical form."""
    if dtype in (pl.Date, pl.Datetime):
        if isinstance(freq, _dt.timedelta):
            return _duration_string(freq)
        if isinstance(freq, str) and freq:
            return freq
        raise ValueError(
            f"a {dtype} time axis needs a polars duration string (e.g. '1d', "
            f"'1mo') or a datetime.timedelta as `frequency`, got {freq!r}."
        )
    if dtype.is_numeric():
        if isinstance(freq, bool) or not isinstance(freq, (int, float)) or freq <= 0:
            raise ValueError(
                f"a numeric time axis needs a positive number as `frequency`, "
                f"got {freq!r}."
            )
        return float(freq)
    raise ValueError(
        f"`frequency` is supported for Date, Datetime and numeric time axes, "
        f"not {dtype}; omit it to check gaps against the panel's own calendar."
    )


def check_gaps(
    df: pl.DataFrame,
    entity: str,
    time: str,
    *,
    frequency: Frequency | None = None,
    impact: Impact,
    limit: int,
) -> CheckResult:
    """No entity may skip a time step between its first and last observation.

    Two calendars:

    * ``frequency=None`` (default) -- the **panel calendar**: the sorted set of
      times observed for *any* entity. An entity has a gap where the calendar
      has a time inside the entity's ``[first, last]`` span that the entity
      lacks. Business-day or exchange calendars need no configuration: the
      panel is its own calendar.
    * ``frequency`` given -- a regular grid: a gap is a step between two
      consecutive observed times longer than ``frequency`` (``offset_by`` for
      temporal axes, so ``"1mo"`` respects month ends).

    Leading and trailing absence (late listing, delisting) is not a gap; see
    :func:`check_coverage` for that. Entity-level: every row of an entity with
    a gap is flagged.
    """
    keys = _distinct_keys(df, entity, time)
    dtype = df.schema[time]
    prev = pl.col(time).shift(1).over(entity)
    if frequency is None:
        keys = keys.with_columns(
            pl.col(time).rank("dense").cast(pl.Int64).alias("__pos__")
        )
        steps = keys.with_columns(
            prev.alias("after"),
            (pl.col("__pos__") - pl.col("__pos__").shift(1).over(entity) - 1).alias(
                "missing"
            ),
        ).filter(pl.col("missing") > 0)
        basis: dict[str, Any] = {"calendar": "panel"}
    else:
        step = _normalise_frequency(frequency, dtype)
        with_prev = keys.with_columns(prev.alias("after"))
        if isinstance(step, str):
            expected_next = pl.col("after").dt.offset_by(step)
            gaps = with_prev.filter(pl.col(time) > expected_next)
            # The grid restarts at each observation: the skipped steps are the
            # grid points strictly between `after` and the next observed time,
            # which is exactly an open (`closed="none"`) range between them.
            ranges = (pl.date_ranges if dtype == pl.Date else pl.datetime_ranges)(
                pl.col("after"), pl.col(time), interval=step, closed="none"
            )
            steps = gaps.with_columns(
                ranges.list.len().cast(pl.Int64).clip(lower_bound=1).alias("missing")
            )
        else:
            diff = (pl.col(time) - pl.col("after")).cast(pl.Float64)
            steps = with_prev.with_columns(
                ((diff / step).ceil() - 1).cast(pl.Int64).alias("missing")
            ).filter(pl.col("missing") > 0)
        basis = {"calendar": "frequency", "frequency": frequency}

    per_entity = (
        steps.group_by(entity)
        .agg(
            pl.col("missing").sum().cast(pl.Int64).alias("n_missing"),
            pl.len().cast(pl.Int64).alias("n_gaps"),
            pl.col("after").sort_by(time).first().alias("first_gap_after"),
            pl.col(time).sort().first().alias("first_gap_before"),
        )
        .sort(entity)
    )
    n_entities = per_entity.height
    n_missing = int(per_entity.get_column("n_missing").sum()) if n_entities else 0
    n_gaps = int(per_entity.get_column("n_gaps").sum()) if n_entities else 0
    mask = _entity_mask(df, entity, per_entity.get_column(entity))
    return CheckResult(
        name="gaps",
        category="panel",
        impact=impact,
        passed=n_entities == 0,
        message=(
            f"no entity skips a time step ({basis['calendar']} calendar)."
            if n_entities == 0
            else f"{_entities(n_entities)} {_have(n_entities)} {n_gaps} internal gap(s) "
            f"totalling {n_missing} missing time step(s) against the "
            f"{basis['calendar']} calendar. A lag or rolling window across a "
            "gap silently spans more time than it says."
        ),
        n_failing=n_entities,
        unit="entities",
        observed={
            "entities_with_gaps": n_entities,
            "gap_intervals": n_gaps,
            "missing_steps": n_missing,
            **basis,
        },
        expected={"entities_with_gaps": 0},
        offending=tuple(
            key_records(
                per_entity,
                [entity],
                extra=("n_gaps", "n_missing", "first_gap_after", "first_gap_before"),
                limit=limit,
            )
        ),
        row_mask=mask,
    )


def check_min_obs(
    df: pl.DataFrame,
    entity: str,
    time: str,
    *,
    min_obs: int,
    impact: Impact,
    limit: int,
) -> CheckResult:
    """Every entity needs at least ``min_obs`` distinct observed times.

    Entity-level: every row of a short entity is flagged, so ``filter`` drops
    entities without enough history to fit a window on.
    """
    counts = (
        _distinct_keys(df, entity, time)
        .group_by(entity)
        .agg(pl.len().cast(pl.Int64).alias("n_obs"))
    )
    short = counts.filter(pl.col("n_obs") < min_obs).sort(entity)
    n = short.height
    lowest = int(counts.get_column("n_obs").min()) if counts.height else 0  # type: ignore[arg-type]
    return CheckResult(
        name="min_obs",
        category="panel",
        impact=impact,
        passed=n == 0,
        message=(
            f"every entity has at least {min_obs} observation(s)."
            if n == 0
            else f"{_entities(n)} {_have(n)} fewer than {min_obs} distinct "
            f"observation(s) (fewest: {lowest})."
        ),
        n_failing=n,
        unit="entities",
        observed={"entities_below": n, "fewest_obs": lowest},
        expected={"min_obs": int(min_obs)},
        offending=tuple(key_records(short, [entity], extra=("n_obs",), limit=limit)),
        row_mask=_entity_mask(df, entity, short.get_column(entity)),
    )


def _calendar_size(
    df: pl.DataFrame, time: str, frequency: Frequency | None
) -> tuple[int, dict[str, Any]]:
    times = df.get_column(time).drop_nulls()
    if times.len() == 0:
        return 0, {"calendar": "panel"}
    if frequency is None:
        return int(times.n_unique()), {"calendar": "panel"}
    dtype = df.schema[time]
    step = _normalise_frequency(frequency, dtype)
    lo: Any = times.min()
    hi: Any = times.max()
    if isinstance(step, str):
        if dtype == pl.Date:
            n = pl.date_range(lo, hi, interval=step, eager=True).len()
        else:
            n = pl.datetime_range(lo, hi, interval=step, eager=True).len()
    else:
        n = int((float(hi) - float(lo)) // step) + 1
    return int(n), {"calendar": "frequency", "frequency": frequency}


def check_coverage(
    df: pl.DataFrame,
    entity: str,
    time: str,
    *,
    min_coverage: float,
    frequency: Frequency | None = None,
    impact: Impact,
    limit: int,
) -> CheckResult:
    """Every entity must be observed on at least ``min_coverage`` of the calendar.

    Coverage is ``distinct observed times / calendar size`` where the calendar
    spans the **whole panel** (so, unlike :func:`check_gaps`, a late listing or
    an early delisting lowers it). The calendar is the panel's own set of
    observed times, or the ``frequency`` grid over the panel's time range.
    """
    n_cal, basis = _calendar_size(df, time, frequency)
    per = (
        _distinct_keys(df, entity, time)
        .group_by(entity)
        .agg(pl.len().cast(pl.Int64).alias("n_obs"))
        .with_columns((pl.col("n_obs") / max(n_cal, 1)).round(12).alias("coverage"))
    )
    low = per.filter(pl.col("coverage") < min_coverage).sort(entity)
    n = low.height
    worst = float(per.get_column("coverage").min()) if per.height else 1.0  # type: ignore[arg-type]
    return CheckResult(
        name="coverage",
        category="panel",
        impact=impact,
        passed=n == 0,
        message=(
            f"every entity covers at least {min_coverage:.0%} of the "
            f"{n_cal}-step calendar."
            if n == 0
            else f"{_entities(n)} {_cover(n)} less than {min_coverage:.0%} of the "
            f"{n_cal}-step {basis['calendar']} calendar (lowest: {worst:.1%})."
        ),
        n_failing=n,
        unit="entities",
        observed={
            "entities_below": n,
            "lowest_coverage": ratio(worst, 1.0),
            "calendar_size": n_cal,
            **basis,
        },
        expected={"min_coverage": float(min_coverage)},
        offending=tuple(
            key_records(low, [entity], extra=("n_obs", "coverage"), limit=limit)
        ),
        row_mask=_entity_mask(df, entity, low.get_column(entity)),
    )
