"""Trend scanning: the recursive-residual horizon sweep, a label and a feature.

Trend scanning (Lopez de Prado 2020, *Machine Learning for Asset Managers*
section 5.4) fits, at every row ``t``, an OLS line ``y ~ a + b x`` over a grid
of window lengths ``L`` and keeps the horizon whose slope t-statistic is
largest in absolute value. The label reads ``y[t .. t+L-1]`` (forward); the
feature :meth:`.ts.trend_scan` reads ``y[t-L+1 .. t]`` (trailing).

Why a sweep, and not prefix sums
--------------------------------
The obvious O(T |L|) method keeps prefix sums of ``y``, ``x y`` and ``y^2`` and
forms each window's residual sum of squares as a difference. That difference
cancels catastrophically: on log prices it was measured wrong in the **second
decimal** of the t-statistic, and on near-perfect lines (R^2 -> 1, exactly where
the label is decided) it was wrong by 99 %. This module instead grows every
window by one point per step, for all rows at once, and accumulates the
residual sum of squares with Brown, Durbin & Evans' (1975) recursive residuals::

    dx   = x_new - xbar_n                 # +-(n+1)/2 on the integer grid
    e    = y_new - ybar - beta * dx       # one-step-ahead prediction error
    h    = 1/n + dx^2 / Sxx(n)            # leverage of the new point
    SSE += e^2 / (1 + h)
    ybar += (y_new - ybar) / (n+1);  Sxy += dx * (y_new - ybar);  beta = Sxy / Sxx(n+1)
    t(L = n+1) = beta * sqrt(Sxx(n+1) * (n-1) / SSE)

``SSE`` is a sum of non-negative terms, so its relative accuracy does not
degrade as the fit becomes perfect. ``Sxx(n) = n(n^2-1)/12`` and the leverage
are exact scalars. Measured: max relative error ~4e-12 against a two-pass
centred OLS oracle on log-price random walks.

Backends
--------
:func:`_sweep` dispatches to a numba kernel (a scalar recursion per row) when
the ``fast`` extra is installed, and otherwise to a numpy twin that runs the
same recursion over a chunk of rows per step. The two perform the same IEEE
operations in the same order and agree **bitwise**
(``tests/test_label_trend.py``); numba is a speed-up, never a feature.

The kernel is compiled ``nogil=True`` and **not** ``parallel=True``: the
``.ts`` feature runs inside polars' ``map_batches``, which polars calls from
several threads at once under ``.over()``, and numba's default ``workqueue``
threading layer aborts the process on concurrent parallel launches (measured).
The frame-level label instead splits the row axis into disjoint chunks and runs
them on a thread pool; rows are independent, so the result is bitwise the same
for any thread count.

Leak-safety
-----------
* :func:`trend_scanning` is a *label*: it reads forward by design. Its ``t1``
  is the **information end** ``t + L_max - 1`` -- every horizon up to ``L_max``
  was scanned to pick ``L*`` -- not the end of the chosen horizon
  (``t1_trend``). Purging on ``t1_trend`` would under-purge (trap T6).
* Rows without a complete ``L_max`` window are ``censored`` and unlabelled by
  default (``allow_partial=False``), so every emitted label is invariant to data
  arriving after its ``t1`` (trap T7).
* :meth:`.ts.trend_scan` is causal and bitwise prefix-invariant: the value at
  ``t`` depends on ``y[t-L_max+1 .. t]`` only, with no length-dependent constant.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from panelary._internal._jit import lazy_njit
from panelary.registry import FeatureSpec, registry

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["trend_scan_expr", "trend_scanning"]

#: Rows per numpy chunk: keeps the ~12 working arrays of one step in cache.
_CHUNK = 1 << 18

#: Below this many rows per thread the numba path runs on the calling thread.
_MIN_ROWS_PER_THREAD = 1 << 16

_OUTPUTS = ("t", "horizon", "slope")


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #
@lazy_njit(nogil=True)
def _sweep_kernel(
    y: np.ndarray,
    avail: np.ndarray,
    direction: int,
    min_w: int,
    l_max: int,
    step: int,
    lo: int,
    hi: int,
    out_t: np.ndarray,
    out_h: np.ndarray,
    out_b: np.ndarray,
    out_ok: np.ndarray,
) -> None:  # pragma: no cover - exercised only when numba is installed
    """Scalar recursion for each row in ``[lo, hi)``; rows are independent."""
    for i in range(lo, hi):
        top = avail[i]
        if top > l_max:
            top = l_max
        best_abs = -1.0
        best_t = np.nan
        best_h = 0
        best_b = np.nan
        ok = False
        ybar = y[i]
        sxy = 0.0
        sse = 0.0
        beta = 0.0
        for m in range(1, top):
            ynew = y[i + direction * m]
            dx = direction * (m + 1) / 2.0
            if m >= 2:
                h = 1.0 / m + dx * dx / (m * (m * m - 1) / 12.0)
                e = ynew - ybar - beta * dx
                sse = sse + e * e / (1.0 + h)
            ybar = ybar + (ynew - ybar) / (m + 1)
            sxy = sxy + dx * (ynew - ybar)
            n_pts = m + 1
            sxx = n_pts * (n_pts * n_pts - 1) / 12.0
            beta = sxy / sxx
            if n_pts >= min_w and (n_pts - min_w) % step == 0:
                if n_pts == min_w:
                    ok = np.isfinite(ybar)
                t = beta * np.sqrt(sxx * (n_pts - 2) / sse)
                a = abs(t)
                if a > best_abs:
                    best_abs = a
                    best_t = t
                    best_h = n_pts
                    best_b = beta
        out_t[i] = best_t
        out_h[i] = best_h
        out_b[i] = best_b
        out_ok[i] = ok


def _sweep_numpy(
    y: np.ndarray,
    avail: np.ndarray,
    direction: int,
    min_w: int,
    l_max: int,
    step: int,
    out_t: np.ndarray,
    out_h: np.ndarray,
    out_b: np.ndarray,
    out_ok: np.ndarray,
) -> None:
    """The numpy twin: the same recursion, one vectorised step per window size.

    Every expression mirrors :func:`_sweep_kernel` operation for operation, so
    the results agree bitwise. Rows whose window has run out of data keep being
    updated with clipped (garbage) inputs, but ``active`` masks them out of the
    best-horizon update, and they can never become active again (availability
    is monotone in the window length).
    """
    n = y.shape[0]
    for lo in range(0, n, _CHUNK):
        hi = min(n, lo + _CHUNK)
        k = hi - lo
        rows = np.arange(lo, hi, dtype=np.int64)
        av = avail[lo:hi]
        ybar = y[lo:hi].copy()
        sxy = np.zeros(k)
        sse = np.zeros(k)
        beta = np.zeros(k)
        best_abs = np.full(k, -1.0)
        best_t = np.full(k, np.nan)
        best_h = np.zeros(k, dtype=np.int64)
        best_b = np.full(k, np.nan)
        ok = np.zeros(k, dtype=np.bool_)
        top = min(l_max, int(av.max())) if k else 0
        src = np.empty(k, dtype=np.int64)
        with np.errstate(all="ignore"):
            for m in range(1, top):
                active = av >= m + 1
                np.add(rows, direction * m, out=src)
                np.clip(src, 0, n - 1, out=src)
                ynew = y[src]
                dx = direction * (m + 1) / 2.0
                if m >= 2:
                    h = 1.0 / m + dx * dx / (m * (m * m - 1) / 12.0)
                    e = ynew - ybar - beta * dx
                    sse = sse + e * e / (1.0 + h)
                ybar = ybar + (ynew - ybar) / (m + 1)
                sxy = sxy + dx * (ynew - ybar)
                n_pts = m + 1
                sxx = n_pts * (n_pts * n_pts - 1) / 12.0
                beta = sxy / sxx
                if n_pts >= min_w and (n_pts - min_w) % step == 0:
                    if n_pts == min_w:
                        ok = active & np.isfinite(ybar)
                    t = beta * np.sqrt(sxx * (n_pts - 2) / sse)
                    a = np.abs(t)
                    upd = active & (a > best_abs)
                    best_abs[upd] = a[upd]
                    best_t[upd] = t[upd]
                    best_h[upd] = n_pts
                    best_b[upd] = beta[upd]
        out_t[lo:hi] = best_t
        out_h[lo:hi] = best_h
        out_b[lo:hi] = best_b
        out_ok[lo:hi] = ok


def _n_threads() -> int:
    return max(1, os.cpu_count() or 1)


def _sweep(
    y: np.ndarray,
    avail: np.ndarray,
    *,
    direction: int,
    min_w: int,
    l_max: int,
    step: int,
    threads: int = 1,
    _backend: Literal["auto", "numpy", "numba"] = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-|t| horizon per row over the grid ``min_w, min_w+step, ..., l_max``.

    Parameters
    ----------
    y : numpy.ndarray
        Float64 values on the flat row axis (NaN = missing).
    avail : numpy.ndarray
        Int64, rows available to row ``i``'s window in ``direction`` including
        ``i`` itself (``seg_end - i + 1`` forward, ``i - seg_start + 1``
        backward). Windows never cross an entity boundary.
    direction : {+1, -1}
        ``+1`` scans ``y[i .. i+L-1]`` (label), ``-1`` scans ``y[i-L+1 .. i]``
        (feature). The slope is always per step *forward in time*.
    min_w, l_max, step : int
        The horizon grid (validated by :func:`_check_grid`).
    threads : int, default 1
        numba path only: split the rows into this many disjoint chunks and run
        them on a thread pool (the kernel releases the GIL). Never set it from
        inside a polars UDF.
    _backend : {"auto", "numpy", "numba"}
        Private: force one backend (tests). ``"numba"`` raises if unavailable.

    Returns
    -------
    (t, horizon, slope, ok)
        ``t``: best t-statistic (NaN when no horizon produced a comparable
        value); ``horizon``: its window length (0 if none); ``slope``: the OLS
        slope at that horizon; ``ok``: the shortest window is complete and
        finite (``False`` -> no statistic can be formed at all).
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    avail = np.ascontiguousarray(avail, dtype=np.int64)
    n = y.shape[0]
    out_t = np.empty(n, dtype=np.float64)
    out_h = np.empty(n, dtype=np.int64)
    out_b = np.empty(n, dtype=np.float64)
    out_ok = np.empty(n, dtype=np.bool_)
    outs = (out_t, out_h, out_b, out_ok)
    grid = (int(direction), int(min_w), int(l_max), int(step))
    kernel = None if _backend == "numpy" else _sweep_kernel.compiled()
    if _backend == "numba" and kernel is None:
        raise RuntimeError("the numba backend was requested but is unavailable")
    if kernel is None:
        _sweep_numpy(y, avail, *grid, *outs)
        return outs
    k = max(1, min(int(threads), n // _MIN_ROWS_PER_THREAD))
    if k == 1:
        kernel(y, avail, *grid, 0, n, *outs)
        return outs
    bounds = np.linspace(0, n, k + 1).astype(np.int64).tolist()

    def run_chunk(lo: int, hi: int) -> None:
        kernel(y, avail, *grid, lo, hi, *outs)

    with ThreadPoolExecutor(max_workers=k) as pool:
        jobs = [
            pool.submit(run_chunk, lo, hi)
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        for job in jobs:
            job.result()
    return outs


def _check_grid(
    min_window: int, max_window: int, step: int, *, name: str
) -> tuple[int, int, int]:
    """Validate the horizon grid; return ``(min_w, l_max, step)``.

    ``l_max`` is the largest grid value actually scanned
    (``min_window + k * step <= max_window``), which is what defines a label's
    information end.
    """
    for label, val in (
        ("min_window", min_window),
        ("max_window", max_window),
        ("step", step),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError(
                f"{name}: `{label}` must be a fixed integer (never derived from "
                f"the series length), got {val!r}."
            )
    min_w, max_w, stp = int(min_window), int(max_window), int(step)
    if min_w < 3:
        raise ValueError(
            f"{name}: `min_window` must be >= 3 (a slope t-statistic needs at "
            f"least one residual degree of freedom), got {min_w}."
        )
    if max_w < min_w:
        raise ValueError(
            f"{name}: `max_window` ({max_w}) must be >= `min_window` ({min_w})."
        )
    if stp < 1:
        raise ValueError(f"{name}: `step` must be >= 1, got {stp}.")
    l_max = min_w + ((max_w - min_w) // stp) * stp
    return min_w, l_max, stp


# --------------------------------------------------------------------------- #
# The look-back feature: .ts.trend_scan
# --------------------------------------------------------------------------- #
def trend_scan_expr(
    expr: pl.Expr,
    *,
    min_window: int = 5,
    max_window: int = 20,
    step: int = 1,
    output: str = "t",
) -> pl.Expr:
    """Trailing trend scan of ``expr``: the best-|t| OLS trend over a window grid.

    At row ``t``, fits ``y[t-L+1 .. t] ~ a + b x`` for every ``L`` in
    ``min_window, min_window+step, ..., <= max_window`` and keeps the ``L`` with
    the largest ``|t(b)|`` (ties -> the shortest). ``output`` selects what is
    emitted: ``"t"`` (the signed t-statistic, Float64), ``"horizon"`` (the
    chosen ``L``, Int64) or ``"slope"`` (``b`` at that ``L``, per row, Float64).

    Windows are *complete or absent*: a window that would reach before the
    entity's first row, or that contains a missing value, is not scanned, and
    every longer window is not scanned either. The output is null when no
    window qualifies (the first ``min_window - 1`` rows, or a missing value
    within the shortest window) and when every qualifying window is degenerate
    (a constant window: ``t = 0/0``).

    Use ``.over(entity)``; without it the whole column is one series.
    Causal and bitwise prefix-invariant (``safe_scope="rowwise"``).
    """
    min_w, l_max, stp = _check_grid(min_window, max_window, step, name="ts.trend_scan")
    if output not in _OUTPUTS:
        raise ValueError(
            f"ts.trend_scan: `output` must be one of {list(_OUTPUTS)}, got {output!r}."
        )
    dtype = pl.Int64 if output == "horizon" else pl.Float64

    def run(s: pl.Series) -> pl.Series:
        y = s.cast(pl.Float64).to_numpy()
        n = y.shape[0]
        avail = np.arange(1, n + 1, dtype=np.int64)
        t, h, b, _ok = _sweep(
            y, avail, direction=-1, min_w=min_w, l_max=l_max, step=stp
        )
        chosen = h > 0
        if output == "horizon":
            return pl.Series(s.name, h, dtype=pl.Int64).scatter(
                np.flatnonzero(~chosen), None
            )
        vals = t if output == "t" else b
        vals = np.where(chosen, vals, np.nan)
        return pl.Series(s.name, vals, dtype=pl.Float64, nan_to_null=True)

    return expr.map_batches(run, return_dtype=dtype)


# --------------------------------------------------------------------------- #
# The forward label: label.trend_scanning
# --------------------------------------------------------------------------- #
def _as_dataframe(df: PolarsFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _resolve_cols(
    df: pl.DataFrame, entity: str | None, time: str | None
) -> tuple[str, str]:
    cols = df.columns
    return (
        entity if entity is not None else cols[0],
        time if time is not None else cols[1],
    )


def _segment_bounds(entity: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """First and last flat row of each row's entity (frame sorted by entity)."""
    seg = entity.rle_id().to_numpy().astype(np.int64)
    n = seg.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:] - 1, n - 1]
    return starts[seg], ends[seg]


def _gather(
    times: pl.Series, idx: np.ndarray, valid: np.ndarray, name: str
) -> pl.Series:
    """``times[idx]`` where ``valid``, null elsewhere; keeps the time dtype."""
    safe = np.where(valid, idx, 0).astype(np.int64)
    out = times.gather(safe).alias(name)
    missing = np.flatnonzero(~valid)
    if missing.size:
        out = out.scatter(missing, None)
    return out


def trend_scanning(
    df: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "close",
    min_window: int = 5,
    max_window: int = 20,
    step: int = 1,
    log: bool = True,
    t_threshold: float = 0.0,
    allow_partial: bool = False,
) -> pl.DataFrame:
    """Trend-scanning labels (Lopez de Prado 2020, MLAM section 5.4).

    For each row ``t`` (within its entity, in time order) fit
    ``y[t .. t+L-1] ~ a + b x`` with ``x = 0 .. L-1`` for every ``L`` in the
    grid ``min_window, min_window+step, ..., <= max_window``; pick
    ``L* = argmax |t(b)|`` (ties -> the shortest ``L``) and label the row
    ``sign(t(L*))`` if ``|t(L*)| >= t_threshold``, else ``0``. ``y`` is the log
    price when ``log=True``.

    The t-statistics come from a recursive-residual sweep over the horizons
    (see the module docstring); they match a two-pass OLS to ~1e-12 relative,
    where the textbook prefix-sum shortcut is wrong in the second decimal.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Panel with an entity, a time and a ``price`` column.
    entity, time : str, optional
        Key columns. Default to the first and second column.
    price : str, default "close"
        Price column.
    min_window, max_window, step : int
        The horizon grid, in rows. ``min_window >= 3``. The largest grid value
        ``L_max`` (``<= max_window``) sets the information end.
    log : bool, default True
        Regress log prices (requires strictly positive prices).
    t_threshold : float, default 0.0
        Minimum ``|t|`` for a non-zero label.
    allow_partial : bool, default False
        If ``False``, rows without ``L_max`` rows of forward data are left
        unlabelled (null) and ``censored``. If ``True`` they are labelled from
        the horizons that fit, still flagged ``censored`` because their label can
        change as data arrives.

    Returns
    -------
    polars.DataFrame
        The input rows, sorted by ``[entity, time]``, plus:

        ``label`` : Int64
            ``-1`` / ``0`` / ``+1``; null when censored (and not
            ``allow_partial``) or when the shortest window has a missing value.
            A window whose every horizon is degenerate (constant) is ``0``.
        ``t_value`` : Float64
            ``t(L*)``; ``+-inf`` for a perfect line; null when no horizon gives
            a comparable statistic.
        ``horizon`` : Int64
            ``L*``, in rows.
        ``slope`` : Float64
            OLS slope at ``L*``, per row (log units when ``log=True``).
        ``t1`` : time dtype
            **The information end**, the time at row ``t + L_max - 1`` (the last
            row the scan read). This -- not ``t1_trend`` -- is the span to purge
            and weight on.
        ``t1_trend`` : time dtype
            Time at row ``t + L* - 1``, the end of the chosen trend. Descriptive
            only: the label depends on data after it.
        ``censored`` : Boolean
            ``True`` iff fewer than ``L_max`` rows of forward data exist.

    Notes
    -----
    A missing price inside a window invalidates that window and every longer
    one; the label is then decided on the shorter complete windows.
    """
    min_w, l_max, stp = _check_grid(min_window, max_window, step, name="trend_scanning")
    if not np.isfinite(t_threshold) or t_threshold < 0:
        raise ValueError(
            f"trend_scanning: `t_threshold` must be finite and >= 0, got {t_threshold!r}."
        )
    frame = _as_dataframe(df)
    entity_col, time_col = _resolve_cols(frame, entity, time)
    frame = frame.sort([entity_col, time_col])

    y = frame.get_column(price).cast(pl.Float64).to_numpy().astype(np.float64)
    if log:
        finite = y[np.isfinite(y)]
        if finite.size and np.any(finite <= 0.0):
            raise ValueError(
                "trend_scanning: log=True needs strictly positive prices; pass "
                "log=False to regress price levels."
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log(y)

    n = frame.height
    rows = np.arange(n, dtype=np.int64)
    _starts, ends = _segment_bounds(frame.get_column(entity_col))
    avail = ends - rows + 1
    full = avail >= l_max
    censored = ~full

    t, h, b, ok = _sweep(
        y,
        avail,
        direction=1,
        min_w=min_w,
        l_max=l_max,
        step=stp,
        threads=_n_threads(),
    )

    scanned = full | allow_partial
    labelled = scanned & ok
    chosen = labelled & (h > 0)
    with np.errstate(invalid="ignore"):
        sign = np.where(np.abs(t) >= t_threshold, np.sign(t), 0.0)
    label = np.where(chosen, sign, 0.0).astype(np.int64)

    times = frame.get_column(time_col)
    info_end = np.minimum(rows + l_max - 1, ends)
    t1 = _gather(times, info_end, labelled, "t1")
    t1_trend = _gather(times, rows + h - 1, chosen, "t1_trend")

    return frame.with_columns(
        pl.Series("label", label, dtype=pl.Int64).scatter(
            np.flatnonzero(~labelled), None
        ),
        pl.Series("t_value", np.where(chosen, t, np.nan), dtype=pl.Float64).scatter(
            np.flatnonzero(~chosen), None
        ),
        pl.Series("horizon", h, dtype=pl.Int64).scatter(np.flatnonzero(~chosen), None),
        pl.Series("slope", np.where(chosen, b, np.nan), dtype=pl.Float64).scatter(
            np.flatnonzero(~chosen), None
        ),
        t1,
        t1_trend,
        pl.Series("censored", censored, dtype=pl.Boolean),
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
registry.register(
    FeatureSpec(
        name="trend_scanning",
        namespace="label",
        input_shape="frame",
        output_shape="frame",
        params={
            "price": str,
            "min_window": int,
            "max_window": int,
            "step": int,
            "log": bool,
            "t_threshold": float,
            "allow_partial": bool,
        },
        tier="B",
        panel_safe=True,  # windows never cross an entity boundary
        leakage_safe=True,
        # A forward label, so the `forward_return` precedent: it reads
        # `t + L_max - 1` by definition and must never be read as a row-safe
        # feature. Its purge contract is the emitted `t1` (information end).
        safe_scope="window",
        source="Panelary",
        license="Apache-2.0",
        axis="time",
        cost_hint="O(T L_max)",
    )
)
