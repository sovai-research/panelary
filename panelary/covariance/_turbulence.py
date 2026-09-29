"""Financial turbulence, made causal (plan section 5.14, traps T1 and T2).

Chow, Jacquier, Kritzman & Lowry (1999) and Kritzman & Li (2010) define
turbulence as the Mahalanobis distance of today's return vector from the
historical mean under the historical covariance -- as published, with the
**full-sample** mean and covariance (trap T1). Here::

    d_t = (r_t - mu_s)^T Sigma_s^{-1} (r_t - mu_s) / N_t

where ``s`` is the latest refit date ``<= t - lag`` with ``lag >= 1``, and
``mu_s`` / ``Sigma_s`` come from the trailing window ``(s - W, s]``. So
``r_t`` never enters its own covariance (trap T2): ``lag=0`` raises.

Entities of the estimate's universe that are unobserved at ``t`` are
marginalised out exactly: with ``P = Sigma^{-1}`` and the missing set ``B``,
``x_A^T (Sigma_AA)^{-1} x_A = x~^T P x~ - y_B^T (P_BB)^{-1} y_B`` where ``x~``
is ``x`` zero-padded and ``y = P x~`` -- ``1 + |B|`` solves of ``O(N r)``
instead of a new ``O(N r^2)`` factorisation (the latter is used when many are
missing). ``N_t`` is the number of entities scored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.core._schedule import Schedule, as_schedule
from panelary.covariance._estimate import METHODS, resolve_method
from panelary.covariance._state import _broadcast, _validate_common
from panelary.covariance._types import CovEstimate
from panelary.covariance._window import (
    DEFAULT_MIN_ENTITIES,
    PanelMatrix,
    panel_matrix,
    window_at,
)

__all__ = ["inv_quad_observed", "turbulence"]


def inv_quad_observed(est: CovEstimate, x: NDArray[np.float64]) -> tuple[float, int]:
    """``x_A^T (Sigma_AA)^{-1} x_A`` over the finite entries ``A`` of ``x``.

    Returns ``(value, |A|)``. Exact for the marginal of the estimate on ``A``.
    """
    obs = np.isfinite(x)
    n_obs = int(obs.sum())
    if n_obs == x.size:
        return est.inv_quad(x), n_obs
    if n_obs == 0:
        return math.nan, 0
    miss = np.flatnonzero(~obs)
    m = miss.size
    if m > max(8, est.rank // 4):
        sub = est.subset(obs)
        return sub.inv_quad(x[obs]), n_obs
    x0 = np.where(obs, x, 0.0)
    y = est.solve(x0)
    E = np.zeros((x.size, m))
    E[miss, np.arange(m)] = 1.0
    PB = est.solve(E)
    PBB = PB[miss]
    yB = y[miss]
    corr = float(yB @ np.linalg.solve(PBB, yB[:, None])[:, 0])
    return float(x0 @ y) - corr, n_obs


def _default_refit(pm: PanelMatrix) -> Schedule:
    dtype = pm.times.dtype
    temporal = dtype == pl.Date or isinstance(dtype, pl.Datetime)
    return as_schedule("1mo" if temporal else 21)


def _trailing_percentile(
    d: NDArray[np.float64], window: int, min_periods: int
) -> NDArray[np.float64]:
    """Share of the trailing ``window`` values (finite, incl. ``t``) ``<= d_t``."""
    out = np.full(d.size, np.nan)
    for t in range(d.size):
        if not np.isfinite(d[t]):
            continue
        w = d[max(0, t + 1 - window) : t + 1]
        w = w[np.isfinite(w)]
        if w.size >= min_periods:
            out[t] = float(np.count_nonzero(w <= d[t])) / w.size
    return out


def turbulence(
    panel: Any,
    *,
    returns: str,
    window: int = 252,
    method: str = "qis",
    refit: Schedule | int | str | None = None,
    lag: int = 1,
    space: str = "correlation",
    min_coverage: float = 0.95,
    min_entities: int = DEFAULT_MIN_ENTITIES,
    pct_window: int = 252,
    broadcast: bool = False,
    entity: str | None = None,
    time: str | None = None,
    **options: Any,
) -> pl.DataFrame:
    """Causal financial turbulence per date.

    Parameters
    ----------
    panel : PanelFrame or polars frame
        Long panel of returns.
    returns : str
        Return column.
    window : int, default 252
        Estimation window ``W`` of each refit.
    method : str, default "qis"
        Covariance estimator (see :func:`estimate`). Shrinkage matters here:
        the sample covariance is singular when ``N > W`` and would raise.
    refit : Schedule, int or str, optional
        Refit grid; default ``"1mo"`` (first date of each month) on a
        temporal axis and every 21 dates on an integer axis.
    lag : int, default 1
        The estimate used at ``t`` is the one from the latest refit date
        ``<= t - lag``. Must be >= 1: ``lag=0`` would put ``r_t`` inside its
        own covariance (trap T2).
    space, min_coverage, min_entities
        As for :func:`rolling`.
    pct_window : int, default 252
        Window of the trailing percentile ``turbulence_pct``.
    broadcast : bool, default False
        Join the per-date values onto the panel's rows.
    **options
        Estimator options.

    Returns
    -------
    polars.DataFrame
        ``time, turbulence, turbulence_pct, n_scored, cond, shrinkage,
        asof_date`` -- ``asof_date`` is the refit date whose estimate scored
        the row.
    """
    if isinstance(lag, bool) or int(lag) < 1:
        raise ValueError(
            f"`lag` must be >= 1, got {lag!r}: with lag=0 the return being scored "
            "sits inside the window its covariance is estimated from (trap T2)."
        )
    _validate_common(window, min_coverage, space)
    window = int(window)
    lag = int(lag)
    name = resolve_method(method)
    pm = panel_matrix(panel, returns, entity=entity, time=time)
    sched = _default_refit(pm) if refit is None else as_schedule(refit)
    asof = sched.asof_positions(pm.times)
    T = pm.n_times
    d = np.full(T, np.nan)
    n_scored = np.zeros(T, dtype=np.int64)
    cond = np.full(T, np.nan)
    shrink = np.full(T, np.nan)
    src = np.full(T, -1, dtype=np.int64)
    cur_s = -2
    est: CovEstimate | None = None
    idx = np.empty(0, dtype=np.intp)
    est_cond = math.nan
    est_shrink = math.nan
    for t in range(lag, T):
        s = int(asof[t - lag])
        if s < 0:
            continue
        if s != cur_s:
            cur_s = s
            ws, idx = window_at(
                pm, s, window=window, min_coverage=min_coverage, space=space,
                min_entities=min_entities,
            )  # fmt: skip
            est = None if ws is None else METHODS[name](ws, **options)
            if est is not None:
                est_cond = est.cond()
                est_shrink = math.nan if est.shrinkage is None else est.shrinkage
        if est is None:
            continue
        assert est.location is not None
        x = pm.R[t, idx] - est.location
        q, n_obs = inv_quad_observed(est, x)
        if n_obs < min_entities:
            continue
        d[t] = q / n_obs
        n_scored[t] = n_obs
        cond[t] = est_cond
        shrink[t] = est_shrink
        src[t] = s
    pct = _trailing_percentile(d, int(pct_window), min(int(pct_window), 21))
    tcol = pm.time_col
    out = pl.DataFrame(
        {
            tcol: pm.times,
            "turbulence": d,
            "turbulence_pct": pct,
            "n_scored": n_scored,
            "cond": cond,
            "shrinkage": shrink,
            "__src__": src,
        }
    )
    out = (
        out.with_columns(
            pm.times.gather(np.clip(src, 0, None)).alias("asof_date"),
            pl.col("turbulence", "turbulence_pct", "cond", "shrinkage").fill_nan(None),
        )
        .with_columns(
            pl.when(pl.col("__src__") >= 0).then(pl.col("asof_date")).otherwise(None),
            pl.when(pl.col("__src__") >= 0)
            .then(pl.col("n_scored"))
            .otherwise(None)
            .alias("n_scored"),
        )
        .drop("__src__")
    )
    if broadcast:
        return _broadcast(panel, out, entity=entity, time=time, on=[tcol])
    return out
