"""Tick, volume and dollar bars (AFML ch. 2), fixed or adaptive, as a long panel.

A bar closes on the tick whose cumulative amount (ticks, shares or dollars)
first reaches the next multiple of the threshold ``theta``. Every bar is
emitted as one row of a long ``(entity, time)`` panel, stamped at its **last
tick** -- the first instant at which everything in it is known -- so a feature
computed from bars is causal (trap T11; contrast ``preprocessing.resample``'s
historical left-edge stamp, bug B3).

The lattice ("carry") construction
----------------------------------
With ``C_t`` the per-entity cumulative amount, tick ``t`` belongs to bar
``floor(C_{t-1} / theta)``: the *exclusive* cumulative sum. The tempting
``floor(C_t / theta)`` puts the tick that crosses a multiple into the *next*
bar, an off-by-one the brute-force tests pin. A trade crossing several
multiples closes one bar (the lattice id skips, and bar ids are re-densified).
The overshoot carries into the next bar, so boundaries sit on a fixed lattice,
depend only on the running sum, and are **prefix-invariant**. Everything is one
``cum_sum().over(entity)`` and one ``group_by``: no per-entity Python loop.

``overshoot="reset"`` instead restarts the count at zero after each close (the
mlfinlab convention): a sequential scan, numba via
:mod:`panelary._internal._jit` with a pure-Python twin (bitwise identical).

Adaptive thresholds
-------------------
``bars_per_day=`` sets ``theta_d = mean(amount of the entity's previous
lookback_days completed days) / bars_per_day``. The lattice then runs on
normalised units ``u_t = v_t / theta_{day(t)}`` with threshold 1, so no daily
reset is needed and boundaries stay prefix-invariant. The first
``lookback_days`` days of each entity produce no bars unless ``init_threshold``
is given. A threshold from the full-sample average daily amount would be a
look-ahead (trap T10).

Only **completed** bars are emitted. The trailing, still-open bar of each
entity is dropped: it would change as ticks arrive.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from panelary._internal._jit import lazy_njit
from panelary.registry import FeatureSpec, registry

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["bars"]

_KINDS = ("tick", "volume", "dollar")
_DUP_POLICIES = ("nudge", "error", "keep")

# Private working columns.
_SIDE = "__side__"
_AMOUNT = "__amount__"
_CLOSES = "__closes__"
_BAR = "__bar__"
_SIZE = "__size__"

#: The fixed output schema, in order (information-driven bars append theirs).
_BAR_COLUMNS = (
    "t_open",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "vwap",
    "n_ticks",
    "buy_volume",
    "bar_index",
)


# --------------------------------------------------------------------------- #
# Shared plumbing (also used by panelary.sample._imbalance)
# --------------------------------------------------------------------------- #
def _prepare_ticks(
    ticks: PolarsFrame,
    *,
    entity: str | None,
    time: str | None,
    price: str,
    size: str | None,
    kind: str,
    name: str,
) -> tuple[pl.DataFrame, str, str]:
    """Validate, clean and sort the ticks; add the side and the bar amount.

    Returns the frame with columns ``entity, time, price, __size__, __side__,
    __amount__`` sorted by ``(entity, time)`` with ties kept in input order (the
    tick sequence), plus the resolved key names.
    """
    # Imported here, not at module scope: _imbalance imports this module.
    from panelary.sample._imbalance import _tick_sign  # noqa: PLC0415

    if kind not in _KINDS:
        raise ValueError(f"{name}: `kind` must be one of {list(_KINDS)}, got {kind!r}.")
    frame = ticks.collect() if isinstance(ticks, pl.LazyFrame) else ticks
    cols = frame.columns
    entity_col = entity if entity is not None else cols[0]
    time_col = time if time is not None else cols[1]
    for col in (entity_col, time_col, price, *([size] if size else [])):
        if col not in cols:
            raise ValueError(f"{name}: column {col!r} is not in the tick frame.")
    if kind in ("volume", "dollar") and size is None:
        raise ValueError(f"{name}: kind={kind!r} needs a `size` column.")

    size_expr = pl.col(size) if size is not None else pl.lit(1, dtype=pl.Int64)
    frame = frame.select(
        pl.col(entity_col),
        pl.col(time_col),
        pl.col(price).cast(pl.Float64).fill_nan(None),
        (
            size_expr.cast(pl.Float64).fill_nan(None)
            if size is not None and not frame.schema[size].is_integer()
            else size_expr.cast(pl.Int64)
        ).alias(_SIZE),
    ).drop_nulls([price, _SIZE])
    if frame.height and frame.get_column(_SIZE).min() < 0:  # type: ignore[operator]
        raise ValueError(f"{name}: `size` must be non-negative.")
    frame = frame.sort([entity_col, time_col], maintain_order=True)

    if kind == "tick":
        amount = pl.lit(1, dtype=pl.Int64)
    elif kind == "volume":
        amount = pl.col(_SIZE)
    else:
        amount = pl.col(price) * pl.col(_SIZE).cast(pl.Float64)
    frame = frame.with_columns(
        _tick_sign(pl.col(price)).over(entity_col).alias(_SIDE),
        amount.alias(_AMOUNT),
    )
    return frame, entity_col, time_col


def _entity_starts(frame: pl.DataFrame, entity_col: str) -> np.ndarray:
    """Boolean array: row is the first tick of its entity (frame sorted by entity)."""
    seg = frame.get_column(entity_col).rle_id().to_numpy()
    starts = np.ones(seg.shape[0], dtype=np.bool_)
    if seg.shape[0]:
        starts[1:] = seg[1:] != seg[:-1]
    return starts


def _aggregate(
    frame: pl.DataFrame,
    *,
    entity: str,
    time: str,
    price: str,
    extra: list[pl.Expr] | None = None,
    on_duplicate_time: str = "nudge",
    name: str = "bars",
) -> pl.DataFrame:
    """Turn ticks carrying a Boolean ``__closes__`` column into completed bars.

    Tick ``t`` belongs to bar ``#closes strictly before t`` within its entity;
    only bars whose last tick closes them are kept.
    """
    if on_duplicate_time not in _DUP_POLICIES:
        raise ValueError(
            f"{name}: `on_duplicate_time` must be one of {list(_DUP_POLICIES)}, "
            f"got {on_duplicate_time!r}."
        )
    bar = pl.col(_CLOSES).cast(pl.Int64).cum_sum().shift(1, fill_value=0).over(entity)
    frame = frame.with_columns(bar.alias(_BAR)).filter(
        pl.col(_CLOSES).any().over([entity, _BAR])
    )
    size = pl.col(_SIZE)
    out = frame.group_by([entity, _BAR], maintain_order=True).agg(
        pl.col(time).last().alias(time),
        pl.col(time).first().alias("t_open"),
        pl.col(price).first().alias("open"),
        pl.col(price).max().alias("high"),
        pl.col(price).min().alias("low"),
        pl.col(price).last().alias("close"),
        size.sum().alias("volume"),
        (pl.col(price) * size.cast(pl.Float64)).sum().alias("dollar_volume"),
        pl.len().cast(pl.Int64).alias("n_ticks"),
        size.filter(pl.col(_SIDE) == 1).sum().alias("buy_volume"),
        *(extra or []),
    )
    out = out.with_columns(
        (pl.col("dollar_volume") / pl.col("volume").cast(pl.Float64)).alias("vwap"),
        pl.col(_BAR).cast(pl.Int64).alias("bar_index"),
    )
    extra_names = [
        c for c in out.columns if c not in (entity, _BAR, time, *_BAR_COLUMNS)
    ]
    out = out.select(entity, time, *_BAR_COLUMNS, *extra_names)
    return _resolve_duplicate_times(
        out, entity=entity, time=time, policy=on_duplicate_time, name=name
    )


def _resolve_duplicate_times(
    out: pl.DataFrame, *, entity: str, time: str, policy: str, name: str
) -> pl.DataFrame:
    """Make ``(entity, time)`` unique when two bars close on the same timestamp.

    ``"nudge"`` moves each duplicate **later** by one unit of the time column's
    physical resolution (1 us for ``Datetime("us")``, 1 day for ``Date``, 1 for
    integers): ``time'_k = max(time_k, time'_{k-1} + 1)``, computed as
    ``cum_max(time - k) + k``. Never earlier, so still causal.
    """
    if policy == "keep" or out.height == 0:
        return out
    dup = out.select(pl.struct(entity, time).is_duplicated().any()).item()
    if not dup:
        return out
    if policy == "error":
        raise ValueError(
            f"{name}: two bars of one entity close on the same timestamp; pass "
            "on_duplicate_time='nudge' (move later by one time unit) or 'keep'."
        )
    dtype = out.schema[time]
    physical = out.select(pl.col(time).to_physical()).schema[time]
    if not physical.is_integer():
        raise ValueError(
            f"{name}: on_duplicate_time='nudge' needs an integer-backed time "
            f"column (integer, Date, Datetime), got {dtype}."
        )
    phys = pl.col(time).to_physical().cast(pl.Int64)
    k = pl.col("bar_index")
    nudged = ((phys - k).cum_max().over(entity) + k).cast(dtype)
    return out.with_columns(nudged.alias(time))


# --------------------------------------------------------------------------- #
# Reset ("no carry") scan
# --------------------------------------------------------------------------- #
@lazy_njit(nogil=True)
def _reset_kernel(amount: Any, starts: Any, theta: float, closes: Any) -> None:
    """Running sum restarted at 0 after each close and at each entity start.

    Plain Python on lists is the twin; numba compiles it for arrays. The same
    IEEE additions in the same order, so both agree bitwise.
    """
    acc = 0.0
    for i in range(len(amount)):
        if starts[i]:
            acc = 0.0
        acc = acc + amount[i]
        if acc >= theta:
            closes[i] = True
            acc = 0.0
        else:
            closes[i] = False


def _reset_closes(
    amount: np.ndarray,
    starts: np.ndarray,
    theta: float,
    *,
    _backend: Literal["auto", "numpy", "numba"] = "auto",
) -> np.ndarray:
    """Dispatch :func:`_reset_kernel` (``"numpy"`` = the pure-Python twin)."""
    amount = np.ascontiguousarray(amount, dtype=np.float64)
    kernel = None if _backend == "numpy" else _reset_kernel.compiled()
    if _backend == "numba" and kernel is None:
        raise RuntimeError("the numba backend was requested but is unavailable")
    if kernel is not None:
        closes = np.zeros(amount.shape[0], dtype=np.bool_)
        kernel(amount, starts, float(theta), closes)
        return closes
    out: list[bool] = [False] * amount.shape[0]
    _reset_kernel.py_func(amount.tolist(), starts.tolist(), float(theta), out)
    return np.asarray(out, dtype=np.bool_)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _check_threshold(theta: Any, *, integral: bool, name: str) -> int | float:
    if isinstance(theta, bool) or not isinstance(theta, (int, float, np.number)):
        raise TypeError(f"{name}: `threshold` must be a number, got {theta!r}.")
    value = float(theta)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: `threshold` must be finite and > 0, got {theta!r}.")
    if integral:
        if value != int(value):
            raise ValueError(
                f"{name}: an integer amount (ticks, integer sizes) needs an "
                f"integer `threshold`, got {theta!r}."
            )
        return int(value)
    return value


def _adaptive_theta(
    frame: pl.DataFrame,
    *,
    entity: str,
    time: str,
    bars_per_day: float,
    lookback_days: int,
    init_threshold: float | None,
    name: str,
) -> pl.DataFrame:
    """Attach ``__theta__``: the causal per-day threshold (trailing completed days)."""
    dtype = frame.schema[time]
    if dtype == pl.Date:
        day = pl.col(time)
    elif isinstance(dtype, pl.Datetime):
        day = pl.col(time).dt.date()
    else:
        raise ValueError(
            f"{name}: bars_per_day= needs a Date or Datetime time column to "
            f"define days, got {dtype}."
        )
    frame = frame.with_columns(day.alias("__day__"))
    daily = frame.group_by([entity, "__day__"], maintain_order=True).agg(
        pl.col(_AMOUNT).cast(pl.Float64).sum().alias("__daily__")
    )
    theta = (
        pl.col("__daily__")
        .shift(1)
        .rolling_sum(window_size=lookback_days, min_samples=lookback_days)
        .over(entity)
        / lookback_days
        / bars_per_day
    )
    if init_threshold is not None:
        theta = theta.fill_null(float(init_threshold))
    daily = daily.with_columns(theta.alias("__theta__")).select(
        entity, "__day__", "__theta__"
    )
    return (
        frame.join(daily, on=[entity, "__day__"], how="left", maintain_order="left")
        .drop("__day__")
        .filter(pl.col("__theta__").is_not_null())
    )


def bars(
    ticks: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "price",
    size: str | None = None,
    kind: Literal["tick", "volume", "dollar"] = "dollar",
    threshold: float | None = None,
    bars_per_day: float | None = None,
    lookback_days: int = 20,
    init_threshold: float | None = None,
    overshoot: Literal["carry", "reset"] = "carry",
    on_duplicate_time: Literal["nudge", "error", "keep"] = "nudge",
) -> pl.DataFrame:
    """Tick, volume or dollar bars from a long tick panel (AFML ch. 2).

    Parameters
    ----------
    ticks : polars.DataFrame or polars.LazyFrame
        One row per trade: entity, time, ``price`` and (for volume / dollar
        bars) ``size``. Ticks with a missing price or size are dropped; ticks
        sharing a timestamp keep their input order.
    entity, time : str, optional
        Key columns. Default to the first and second column.
    price : str, default "price"
        Trade price.
    size : str, optional
        Trade size. Required for ``kind="volume"`` / ``"dollar"``; without it
        every tick counts as one unit in ``volume`` / ``buy_volume``.
    kind : {"tick", "volume", "dollar"}, default "dollar"
        What accumulates towards the threshold: 1 per tick, the size, or
        ``price * size``. Tick and integer-size volume bars are exact integer
        arithmetic.
    threshold : float, optional
        Fixed bar size in ``kind`` units. Exactly one of ``threshold`` and
        ``bars_per_day`` must be given.
    bars_per_day : float, optional
        Adaptive threshold: ``theta_d = mean(daily amount over the entity's
        previous lookback_days days) / bars_per_day`` (see the module notes).
        Needs a Date / Datetime time column.
    lookback_days : int, default 20
        Trailing completed days behind the adaptive threshold.
    init_threshold : float, optional
        Adaptive only: threshold for days without ``lookback_days`` of history
        (otherwise those days produce no bars).
    overshoot : {"carry", "reset"}, default "carry"
        ``"carry"``: fixed lattice, overshoot counts towards the next bar.
        ``"reset"``: the count restarts at 0 after each close.
    on_duplicate_time : {"nudge", "error", "keep"}, default "nudge"
        Two bars of an entity can close on the same timestamp; ``"nudge"``
        moves the later one forward by one time unit (never earlier),
        ``"error"`` raises, ``"keep"`` leaves duplicate keys.

    Returns
    -------
    polars.DataFrame
        One row per **completed** bar, sorted by ``(entity, time)``: ``entity,
        time`` (the bar's last tick), ``t_open`` (first tick), ``open, high,
        low, close``, ``volume`` (Int64 for integer sizes), ``dollar_volume``,
        ``vwap``, ``n_ticks``, ``buy_volume`` (tick-rule buys; ticks before an
        entity's first price change have no side) and ``bar_index`` (0-based
        within the entity).
    """
    fname = "bars"
    if overshoot not in ("carry", "reset"):
        raise ValueError(
            f"{fname}: `overshoot` must be 'carry' or 'reset', got {overshoot!r}."
        )
    if (threshold is None) == (bars_per_day is None):
        raise ValueError(
            f"{fname}: pass exactly one of `threshold` and `bars_per_day`."
        )
    frame, entity_col, time_col = _prepare_ticks(
        ticks, entity=entity, time=time, price=price, size=size, kind=kind, name=fname
    )

    if bars_per_day is not None:
        bpd = float(bars_per_day)
        if not math.isfinite(bpd) or bpd <= 0:
            raise ValueError(f"{fname}: `bars_per_day` must be finite and > 0.")
        if (
            isinstance(lookback_days, bool)
            or not isinstance(lookback_days, int)
            or lookback_days < 1
        ):
            raise ValueError(f"{fname}: `lookback_days` must be an integer >= 1.")
        if init_threshold is not None:
            _check_threshold(init_threshold, integral=False, name=fname)
        frame = _adaptive_theta(
            frame,
            entity=entity_col,
            time=time_col,
            bars_per_day=bpd,
            lookback_days=lookback_days,
            init_threshold=init_threshold,
            name=fname,
        )
        frame = frame.with_columns(
            (pl.col(_AMOUNT).cast(pl.Float64) / pl.col("__theta__")).alias(_AMOUNT)
        ).drop("__theta__")
        theta: int | float = 1.0
    else:
        integral = frame.schema[_AMOUNT].is_integer()
        theta = _check_threshold(threshold, integral=kind == "tick", name=fname)
        if integral and float(theta).is_integer():
            theta = int(theta)
        elif integral:
            frame = frame.with_columns(pl.col(_AMOUNT).cast(pl.Float64))

    if overshoot == "carry":
        c_in = "__c_in__"
        frame = frame.with_columns(
            pl.col(_AMOUNT).cum_sum().over(entity_col).alias(c_in)
        )
        c_ex = pl.col(c_in).shift(1, fill_value=0).over(entity_col)
        if isinstance(theta, int):
            closes = (pl.col(c_in) // theta) > (c_ex // theta)
        else:
            closes = (pl.col(c_in) / theta).floor() > (c_ex / theta).floor()
        frame = frame.with_columns(closes.alias(_CLOSES)).drop(c_in)
    else:
        flags = _reset_closes(
            frame.get_column(_AMOUNT).to_numpy(),
            _entity_starts(frame, entity_col),
            float(theta),
        )
        frame = frame.with_columns(pl.Series(_CLOSES, flags, dtype=pl.Boolean))

    return _aggregate(
        frame,
        entity=entity_col,
        time=time_col,
        price=price,
        on_duplicate_time=on_duplicate_time,
        name=fname,
    )


registry.register(
    FeatureSpec(
        name="bars",
        namespace="sample",
        input_shape="frame",
        output_shape="frame",
        params={
            "price": str,
            "size": str,
            "kind": str,
            "threshold": float,
            "bars_per_day": float,
            "lookback_days": int,
            "overshoot": str,
        },
        tier="B",
        panel_safe=True,  # cumulative sums restart at every entity
        leakage_safe=True,
        # Like `factor.ic` (one output row per date): one row per completed bar,
        # stamped at its last tick and built from ticks at or before it, so the
        # value at every emitted time uses only data at <= that time -- and the
        # conformance suite verifies exactly that with both instruments. The
        # plan proposed "window"; "rowwise" is the claim the tests can check.
        safe_scope="rowwise",
        source="Panelary",
        license="Apache-2.0",
        backend_fn=bars,
        intent="compress",
        axis="time",
        flavour="trailing",
        width_rule="data_dependent",
        cost_hint="O(n log n)",
    )
)
