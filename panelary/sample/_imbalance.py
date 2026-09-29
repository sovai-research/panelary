"""Information-driven bars: tick / volume / dollar imbalance and run bars (AFML 2.3.2).

Each tick carries a side ``b_t`` (the tick rule) and an amount ``v_t`` (1, the
size, or ``price * size``). Since the current bar opened:

* **imbalance** bars accumulate ``theta = sum(b v)`` and close when
  ``|theta| >= E[T] * |E[b v]|``;
* **run** bars accumulate ``theta = max(sum_{b=+1} v, sum_{b=-1} v)`` and close
  when ``theta >= E[T] * max(P+ E[v|+], (1 - P+) E[v|-])``.

The expectations are per-bar EWMAs (``alpha = 2 / (span_bars + 1)``) of the
**completed** bars' length ``T_k``, mean imbalance ``theta_k / T_k``, buy
fraction and side-conditional mean size. They are updated only when a bar
closes (trap T9): never from the bar in progress, never from later bars, never
from a full-sample mean. The initial ``E[T]`` is ``init_expected_ticks``; the
initial ``E[b v]`` (or ``P+``, ``E[v|+-]``) comes from the entity's first
``warmup_ticks`` ticks, during which no bar may close, or from
``init_imbalance``.

Known instability, clamped
--------------------------
``E[T]`` and the threshold feed back on each other: a small ``|E[b v]|``
closes short bars, which shrinks ``E[T]``, which closes shorter bars (and the
mirror image explodes). This is widely reported for AFML's definition. ``E[T]``
is therefore clamped to ``[lo, hi] * init_expected_ticks`` (``bounds``,
default 0.1 - 10); each bar reports in ``clamped`` whether the ``E[T]`` behind
its threshold had hit a bound. The default bounds are a guess, not a
calibration.

Backends: one sequential scan over the flat, ``(entity, time)``-sorted tick axis
with state reset at each entity start. numba (via
:mod:`panelary._internal._jit`) when available, else the same function run as
plain Python over lists: identical IEEE operations in the same order, so the
two agree bitwise.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from panelary._internal._jit import lazy_njit
from panelary.registry import FeatureSpec, registry
from panelary.sample._bars import (
    _AMOUNT,
    _CLOSES,
    _aggregate,
    _entity_starts,
    _prepare_ticks,
)

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["imbalance_bars"]

_THR = "__threshold__"
_THETA = "__theta__"
_CLAMP = "__clamped__"


def _tick_sign(price: pl.Expr) -> pl.Expr:
    """Tick rule: ``sign(price_t - price_{t-1})``, zero moves carry the last side.

    Null until the first non-zero price change (no fabricated side for the head
    of the series). Apply ``.over(entity)``. Returns Int8 in ``{-1, +1}``.

    # TODO: use sample.tick_rule once merged (plan 4 M4, A11; label-spans branch).
    """
    move = price.diff().sign()
    return pl.when(move == 0).then(None).otherwise(move).forward_fill().cast(pl.Int8)


@lazy_njit(nogil=True)
def _info_bars_kernel(
    side: Any,
    amount: Any,
    starts: Any,
    run: bool,
    init_t: float,
    lo_t: float,
    hi_t: float,
    alpha: float,
    warmup: int,
    has_init: bool,
    init_imb: float,
    closes: Any,
    thr_out: Any,
    theta_out: Any,
    clamped_out: Any,
) -> None:
    """One sequential pass; state resets at each entity start (``starts``).

    ``side`` is -1 / 0 / +1 (0 = no side yet). Written in the numba-compatible
    subset so that it runs unchanged as Python over lists (the twin).
    """
    n = len(side)
    k = 0
    e_t = init_t
    e_bv = 0.0
    p_buy = 0.5
    e_vb = 0.0
    e_vs = 0.0
    ready = False
    was_clamped = False
    w_bv = 0.0
    w_v = 0.0
    w_nb = 0
    w_ns = 0
    w_sb = 0.0
    w_ss = 0.0
    t_len = 0
    th = 0.0
    sb = 0.0
    ss = 0.0
    nb = 0
    ns = 0
    for i in range(n):
        if starts[i]:
            k = 0
            e_t = init_t
            was_clamped = False
            w_bv = 0.0
            w_v = 0.0
            w_nb = 0
            w_ns = 0
            w_sb = 0.0
            w_ss = 0.0
            t_len = 0
            th = 0.0
            sb = 0.0
            ss = 0.0
            nb = 0
            ns = 0
            ready = warmup == 0
            e_bv = init_imb
            p_buy = 0.5
            e_vb = 0.0
            e_vs = 0.0
        b = side[i]
        v = amount[i]
        t_len += 1
        if b > 0:
            th = th + v
            sb = sb + v
            nb += 1
        elif b < 0:
            th = th - v
            ss = ss + v
            ns += 1
        closes[i] = False
        thr_out[i] = np.nan
        theta_out[i] = np.nan
        clamped_out[i] = False
        if k < warmup:
            w_v = w_v + v
            if b > 0:
                w_bv = w_bv + v
                w_nb += 1
                w_sb = w_sb + v
            elif b < 0:
                w_bv = w_bv - v
                w_ns += 1
                w_ss = w_ss + v
            if k == warmup - 1:
                ready = True
                e_bv = init_imb if has_init else w_bv / warmup
                mean_v = w_v / warmup
                p_buy = w_nb / (w_nb + w_ns) if w_nb + w_ns > 0 else 0.5
                e_vb = w_sb / w_nb if w_nb > 0 else mean_v
                e_vs = w_ss / w_ns if w_ns > 0 else mean_v
        elif ready:
            if run:
                a_buy = p_buy * e_vb
                a_sell = (1.0 - p_buy) * e_vs
                thr = e_t * (a_buy if a_buy >= a_sell else a_sell)
                theta = sb if sb >= ss else ss
                hit = theta >= thr
            else:
                thr = e_t * abs(e_bv)
                theta = th
                hit = abs(theta) >= thr
            if hit:
                closes[i] = True
                thr_out[i] = thr
                theta_out[i] = theta
                clamped_out[i] = was_clamped
                new_t = alpha * t_len + (1.0 - alpha) * e_t
                was_clamped = False
                if new_t < lo_t:
                    new_t = lo_t
                    was_clamped = True
                elif new_t > hi_t:
                    new_t = hi_t
                    was_clamped = True
                e_t = new_t
                if run:
                    if nb + ns > 0:
                        p_buy = alpha * (nb / (nb + ns)) + (1.0 - alpha) * p_buy
                    if nb > 0:
                        e_vb = alpha * (sb / nb) + (1.0 - alpha) * e_vb
                    if ns > 0:
                        e_vs = alpha * (ss / ns) + (1.0 - alpha) * e_vs
                else:
                    e_bv = alpha * (th / t_len) + (1.0 - alpha) * e_bv
                t_len = 0
                th = 0.0
                sb = 0.0
                ss = 0.0
                nb = 0
                ns = 0
        k += 1


def _info_bars_scan(
    side: np.ndarray,
    amount: np.ndarray,
    starts: np.ndarray,
    *,
    run: bool,
    init_t: float,
    lo_t: float,
    hi_t: float,
    alpha: float,
    warmup: int,
    init_imb: float | None,
    _backend: Literal["auto", "numpy", "numba"] = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the scan; returns ``(closes, threshold, theta, clamped)`` per tick.

    ``_backend="numpy"`` runs the pure-Python twin (the kernel over lists).
    """
    n = side.shape[0]
    args = (
        bool(run),
        float(init_t),
        float(lo_t),
        float(hi_t),
        float(alpha),
        int(warmup),
        init_imb is not None,
        float(init_imb) if init_imb is not None else 0.0,
    )
    kernel = None if _backend == "numpy" else _info_bars_kernel.compiled()
    if _backend == "numba" and kernel is None:
        raise RuntimeError("the numba backend was requested but is unavailable")
    if kernel is not None:
        closes = np.zeros(n, dtype=np.bool_)
        thr = np.empty(n, dtype=np.float64)
        theta = np.empty(n, dtype=np.float64)
        clamped = np.zeros(n, dtype=np.bool_)
        kernel(
            np.ascontiguousarray(side, dtype=np.int8),
            np.ascontiguousarray(amount, dtype=np.float64),
            np.ascontiguousarray(starts, dtype=np.bool_),
            *args,
            closes,
            thr,
            theta,
            clamped,
        )
        return closes, thr, theta, clamped
    c_l: list[bool] = [False] * n
    t_l: list[float] = [0.0] * n
    h_l: list[float] = [0.0] * n
    k_l: list[bool] = [False] * n
    _info_bars_kernel.py_func(
        side.astype(np.int64).tolist(),
        np.asarray(amount, dtype=np.float64).tolist(),
        starts.tolist(),
        *args,
        c_l,
        t_l,
        h_l,
        k_l,
    )
    return (
        np.asarray(c_l, dtype=np.bool_),
        np.asarray(t_l, dtype=np.float64),
        np.asarray(h_l, dtype=np.float64),
        np.asarray(k_l, dtype=np.bool_),
    )


def _positive_int(value: Any, name: str, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name}: `{label}` must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name}: `{label}` must be >= {minimum}, got {value!r}.")
    return int(value)


def imbalance_bars(
    ticks: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "price",
    size: str | None = None,
    kind: Literal["tick", "volume", "dollar"] = "tick",
    run: bool = False,
    init_expected_ticks: int,
    span_bars: int = 20,
    warmup_ticks: int | None = None,
    init_imbalance: float | None = None,
    bounds: tuple[float, float] = (0.1, 10.0),
    on_duplicate_time: Literal["nudge", "error", "keep"] = "nudge",
) -> pl.DataFrame:
    """Imbalance or run bars (AFML 2.3.2) from a long tick panel.

    Parameters
    ----------
    ticks : polars.DataFrame or polars.LazyFrame
        One row per trade (see :func:`panelary.sample.bars`).
    entity, time, price, size, kind
        As in :func:`panelary.sample.bars`; ``kind`` sets ``v_t``.
    run : bool, default False
        Run bars instead of imbalance bars.
    init_expected_ticks : int
        Initial ``E[T]``, and the anchor of ``bounds``. Required: there is no
        honest data-free default.
    span_bars : int, default 20
        EWMA span over completed bars, ``alpha = 2 / (span_bars + 1)``.
    warmup_ticks : int, optional
        Ticks at the start of each entity used to estimate the initial
        ``E[b v]`` (run bars: ``P+``, ``E[v|+]``, ``E[v|-]``); no bar closes
        during them. Defaults to ``init_expected_ticks``, or to 0 when
        ``init_imbalance`` is given.
    init_imbalance : float, optional
        Imbalance bars only: a fixed initial ``E[b v]`` instead of the warm-up
        estimate.
    bounds : (float, float), default (0.1, 10.0)
        ``E[T]`` is clamped to ``[lo, hi] * init_expected_ticks``.
    on_duplicate_time : {"nudge", "error", "keep"}, default "nudge"
        As in :func:`panelary.sample.bars`.

    Returns
    -------
    polars.DataFrame
        The :func:`panelary.sample.bars` schema plus ``threshold`` (the
        threshold in force when the bar closed), ``imbalance`` (``theta`` at the
        close: signed for imbalance bars, the larger side's run for run bars)
        and ``clamped`` (the ``E[T]`` behind the threshold was at a bound).
        Only completed bars; ticks before an entity's first price change have
        no side and add nothing to ``theta``.
    """
    fname = "imbalance_bars"
    init_t = _positive_int(init_expected_ticks, fname, "init_expected_ticks")
    span = _positive_int(span_bars, fname, "span_bars")
    if init_imbalance is not None:
        if run:
            raise ValueError(
                f"{fname}: `init_imbalance` applies to imbalance bars only."
            )
        if not math.isfinite(float(init_imbalance)) or float(init_imbalance) == 0.0:
            raise ValueError(f"{fname}: `init_imbalance` must be finite and non-zero.")
    if warmup_ticks is None:
        warmup = 0 if init_imbalance is not None else init_t
    else:
        warmup = _positive_int(warmup_ticks, fname, "warmup_ticks", minimum=0)
    if warmup == 0 and init_imbalance is None:
        raise ValueError(
            f"{fname}: with warmup_ticks=0 the initial expectation must come from "
            "`init_imbalance` (imbalance bars); run bars need warmup_ticks >= 1."
        )
    lo, hi = (float(b) for b in bounds)
    if not (0.0 < lo <= 1.0 <= hi < math.inf):
        raise ValueError(
            f"{fname}: `bounds` must satisfy 0 < lo <= 1 <= hi, got {bounds!r}."
        )

    frame, entity_col, time_col = _prepare_ticks(
        ticks, entity=entity, time=time, price=price, size=size, kind=kind, name=fname
    )
    closes, thr, theta, clamped = _info_bars_scan(
        frame.get_column("__side__").fill_null(0).to_numpy(),
        frame.get_column(_AMOUNT).cast(pl.Float64).to_numpy(),
        _entity_starts(frame, entity_col),
        run=run,
        init_t=float(init_t),
        lo_t=lo * init_t,
        hi_t=hi * init_t,
        alpha=2.0 / (span + 1.0),
        warmup=warmup,
        init_imb=init_imbalance,
    )
    frame = frame.with_columns(
        pl.Series(_CLOSES, closes, dtype=pl.Boolean),
        pl.Series(_THR, thr, dtype=pl.Float64),
        pl.Series(_THETA, theta, dtype=pl.Float64),
        pl.Series(_CLAMP, clamped, dtype=pl.Boolean),
    )
    return _aggregate(
        frame,
        entity=entity_col,
        time=time_col,
        price=price,
        extra=[
            pl.col(_THR).last().alias("threshold"),
            pl.col(_THETA).last().alias("imbalance"),
            pl.col(_CLAMP).last().alias("clamped"),
        ],
        on_duplicate_time=on_duplicate_time,
        name=fname,
    )


registry.register(
    FeatureSpec(
        name="imbalance_bars",
        namespace="sample",
        input_shape="frame",
        output_shape="frame",
        params={
            "price": str,
            "size": str,
            "kind": str,
            "run": bool,
            "init_expected_ticks": int,
            "span_bars": int,
            "warmup_ticks": int,
            "bounds": tuple,
        },
        tier="C",
        panel_safe=True,  # the scan resets its state at every entity start
        leakage_safe=True,
        # See `bars`: one row per completed bar at its last tick; thresholds
        # from completed bars only (T9). Verified per row by both instruments.
        safe_scope="rowwise",
        source="Panelary",
        license="Apache-2.0",
        backend_fn=imbalance_bars,
        intent="compress",
        axis="time",
        flavour="trailing",
        width_rule="data_dependent",
        cost_hint="O(n log n)",
    )
)
