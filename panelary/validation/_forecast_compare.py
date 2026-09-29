"""Forecast comparison beyond Diebold-Mariano: nested, conditional, unstable, rational.

Every test takes a ``(T, M)`` matrix of forecasts (or losses) against one
``(T,)`` benchmark and runs a single vectorised pass, returning an
:class:`~panelary.validation.EvaluationResult` with one row per model.

=========================== ================================================ ========================
test                        question                                         null distribution
=========================== ================================================ ========================
:func:`clark_west`          does a model that *nests* the benchmark beat it? N(0,1), one-sided
:func:`oos_r2`              Campbell-Thompson R^2_OS against the expanding   N(0,1) one-sided (CW)
                            historical mean
:func:`mincer_zarnowitz`    are forecasts unbiased and efficient?            chi2(2) (HAC) / F(2,T-2)
:func:`pesaran_timmermann`  is the direction predicted better than chance?   N(0,1) / hypergeometric
:func:`encompassing_test`   does forecast A encompass forecast B? (HLN)      t(T-1), one-sided
:func:`giacomini_white`     conditional predictive ability                   chi2(q)
:func:`fluctuation_test`    relative performance over rolling windows (GR)   shipped GR table
:func:`one_time_reversal_test` a single break in relative performance (GR)   shipped GR table
=========================== ================================================ ========================

Serial correlation: the default HAC is Bartlett with ``horizon - 1`` lags (the
MA order of an optimal ``h``-step forecast error), never a function of ``T``;
``hac_lags`` overrides it.

Why Clark-West and not DM for nested models: under the null the larger model
estimates parameters that are truly zero, which inflates its squared error, so
the DM statistic is centred below zero and the test has almost no power
(measured size 0.3 % at nominal 10 % in the plan's rolling-OLS design). CW adds
back the ``(yhat_0 - yhat_1)^2`` term that the noise contributes.

Panels enter through :func:`loss_panel`, which averages per-row losses within
each date (same-date rows only) so that the date series absorbs the
contemporaneous cross-entity dependence that per-entity tests pooled naively
ignore.

Prefix invariance: the time-indexed outputs (the expanding historical-mean
benchmark, the expanding R^2_OS path, the cumulative SSE-difference path and
the fluctuation *monitor* path) are running sums with fixed windows, lags and
``min_periods``, so appending rows never changes an earlier value. The
full-sample statistics are, by design, functions of the whole record.

References
----------
Clark, T. E. & West, K. D. (2007). *J. Econometrics* 138(1), 291-311.
Campbell, J. Y. & Thompson, S. B. (2008). *RFS* 21(4), 1509-1531.
Mincer, J. & Zarnowitz, V. (1969). The evaluation of economic forecasts. NBER.
Pesaran, M. H. & Timmermann, A. (1992). *JBES* 10(4), 461-465; (2009) *JASA*
104(485), 325-337.
Harvey, D., Leybourne, S. & Newbold, P. (1998). *JBES* 16(2), 254-259.
Giacomini, R. & White, H. (2006). *Econometrica* 74(6), 1545-1578.
Giacomini, R. & Rossi, B. (2010). *J. Applied Econometrics* 25(4), 595-620.
Goyal, A. & Welch, I. (2008). *RFS* 21(4), 1455-1508.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from panelary._internal import _special
from panelary.econ._common import chi2_sf
from panelary.validation import _gr_tables, _hac
from panelary.validation._arrays import (
    as_matrix,
    as_vector,
    normal_pvalue,
    resolve_names,
)
from panelary.validation._results import EvaluationResult

__all__ = [
    "FluctuationResult",
    "clark_west",
    "encompassing_test",
    "fluctuation_test",
    "giacomini_white",
    "loss_panel",
    "mincer_zarnowitz",
    "one_time_reversal_test",
    "oos_r2",
    "pesaran_timmermann",
]

_MIN_OBS = 5


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _lags(horizon: int, hac_lags: int | None) -> int:
    if horizon < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    if hac_lags is None:
        return horizon - 1
    if not isinstance(hac_lags, (int, np.integer)) or hac_lags < 0:
        raise ValueError(
            "`hac_lags` must be a non-negative integer (a fixed lag, never T-based)."
        )
    return int(hac_lags)


def _forecasts(
    y: Any, forecasts: Any, names: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    mat, colnames = as_matrix(forecasts, "forecasts")
    yy = as_vector(y, "y", mat.shape[0])
    return yy, mat, resolve_names(names, mat.shape[1], colnames)


def _losses(
    loss_a: Any, loss_b: Any, names: Sequence[str] | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """``loss_a - loss_b`` as ``(T, M)``; either side may be ``(T,)``."""
    a, ca = as_matrix(loss_a, "loss_a")
    b, cb = as_matrix(loss_b, "loss_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"`loss_a` has {a.shape[0]} rows, `loss_b` {b.shape[0]}.")
    if a.shape[1] != b.shape[1] and 1 not in (a.shape[1], b.shape[1]):
        raise ValueError(f"loss widths {a.shape[1]} and {b.shape[1]} do not broadcast.")
    d = a - b
    cols = cb if b.shape[1] == d.shape[1] and cb is not None else ca
    return d, resolve_names(names, d.shape[1], cols)


def _groups(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return _hac.mask_groups(mask)


def _t_pvalue(stat: np.ndarray, df: float, alternative: str) -> np.ndarray:
    s = np.asarray(stat, dtype=np.float64)
    out = np.full(s.shape, np.nan)
    ok = np.isfinite(s)
    if not ok.any():
        return out
    v = s[ok]
    if alternative == "greater":
        out[ok] = np.asarray(_special.t_sf(v, df), dtype=np.float64)
    elif alternative == "less":
        out[ok] = np.asarray(_special.t_cdf(v, df), dtype=np.float64)
    else:
        out[ok] = np.minimum(2.0 * np.asarray(_special.t_sf(np.abs(v), df), float), 1.0)
    return out


def _dm_columns(
    d: np.ndarray,
    *,
    horizon: int = 1,
    lags: int | None = None,
    harvey: bool = True,
    alternative: str = "two-sided",
) -> dict[str, np.ndarray]:
    """Vectorised Diebold-Mariano (HLN-corrected, ``t(T-1)``) on columns of ``d``.

    Each column uses its own finite rows (compressed, as
    :func:`~panelary.validation.diebold_mariano` does). Returns per-column
    ``statistic``, ``pvalue``, ``mean``, ``lrv`` and ``n``. Degenerate columns
    (fewer than 3 observations, zero long-run variance, an HLN factor <= 0) are
    ``nan``.
    """
    arr = np.asarray(d, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    m = arr.shape[1]
    out = {k: np.full(m, np.nan) for k in ("statistic", "pvalue", "mean", "lrv")}
    out["n"] = np.zeros(m)
    lag = _lags(horizon, lags)
    for rows, cols in _groups(np.isfinite(arr)):
        n = rows.size
        out["n"][cols] = n
        if n < 3:
            continue
        x = arr[rows][:, cols]
        mean = _hac.colmean(x)
        lrv = np.atleast_1d(_hac.bartlett_lrv(x, min(lag, n - 1)))
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = np.where(lrv > 0, mean / np.sqrt(lrv / n), np.nan)
        if harvey:
            h = horizon
            factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
            stat = (
                stat * math.sqrt(factor) if factor > 0 else np.full_like(stat, np.nan)
            )
        out["statistic"][cols] = stat
        out["pvalue"][cols] = _t_pvalue(stat, n - 1.0, alternative)
        out["mean"][cols] = mean
        out["lrv"][cols] = lrv
    return out


def _column_pass(
    data: np.ndarray, fn: Any, m: int, keys: Sequence[str], min_obs: int = _MIN_OBS
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Run ``fn(rows, cols) -> dict`` on each availability group of ``data``."""
    out = {k: np.full(m, np.nan) for k in keys}
    nobs = np.zeros(m, dtype=np.int64)
    notes: list[str] = []
    for rows, cols in _groups(
        np.all(np.isfinite(data), axis=0) if data.ndim == 3 else np.isfinite(data)
    ):
        nobs[cols] = rows.size
        if rows.size < min_obs:
            notes.append(
                f"{len(cols)} model(s) have fewer than {min_obs} observations; nan."
            )
            continue
        for k, v in fn(rows, cols).items():
            out[k][cols] = v
    return out, nobs, notes


# --------------------------------------------------------------------------- #
# Clark-West
# --------------------------------------------------------------------------- #
def _cw_core(
    y: np.ndarray, bench: np.ndarray, fc: np.ndarray, lag: int
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    e0 = y[:, None] - bench[:, None]
    e1 = y[:, None] - fc
    f = e0**2 - (e1**2 - (bench[:, None] - fc) ** 2)
    stack = np.stack([f, e0 * np.ones_like(e1), e1])

    def fn(rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
        n = rows.size
        ff = f[rows][:, cols]
        mean = _hac.colmean(ff)
        lrv = np.atleast_1d(_hac.bartlett_lrv(ff, min(lag, n - 1)))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(lrv > 0, mean / np.sqrt(lrv / n), np.nan)
        s0 = _hac.coldot(
            e0[rows][:, np.zeros(len(cols), dtype=np.int64)],
            e0[rows][:, np.zeros(len(cols), dtype=np.int64)],
        )
        s1 = _hac.coldot(e1[rows][:, cols], e1[rows][:, cols])
        return {
            "adj": mean,
            "t": t,
            "mspe_benchmark": s0 / n,
            "mspe_model": s1 / n,
            "sse_benchmark": s0,
            "sse_model": s1,
        }

    keys = ("adj", "t", "mspe_benchmark", "mspe_model", "sse_benchmark", "sse_model")
    return _column_pass(stack, fn, fc.shape[1], keys)


def clark_west(
    y: Any,
    benchmark_forecast: Any,
    forecasts: Any,
    *,
    horizon: int = 1,
    hac_lags: int | None = None,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Clark-West (2007) test of equal predictive accuracy for **nested** models.

    ``f_t = e_0^2 - [e_i^2 - (yhat_0 - yhat_i)^2]`` and
    ``t = sqrt(P) mean(f) / sqrt(LRV(f))``, one-sided ``N(0,1)``: a small
    p-value says the larger model (which must nest the benchmark; this cannot
    be checked from forecasts) predicts better.

    Parameters
    ----------
    y : array-like of shape (T,)
        Realisations.
    benchmark_forecast : array-like of shape (T,)
        Forecasts of the nested (smaller) model, e.g. the historical mean.
    forecasts : array-like of shape (T,) or (T, M)
        Forecasts of the larger models.
    horizon : int, default 1
        Forecast horizon; Bartlett HAC with ``horizon - 1`` lags by default.
    hac_lags : int, optional
        Fixed Bartlett lag.
    names : sequence of str, optional

    Returns
    -------
    EvaluationResult
        ``estimate`` is the MSPE-adjusted mean ``mean(f)``; ``details`` holds
        ``mspe_benchmark``, ``mspe_model`` and ``mspe_difference``.
    """
    yy, fc, labels = _forecasts(y, forecasts, names)
    bench = as_vector(benchmark_forecast, "benchmark_forecast", yy.shape[0])
    lag = _lags(horizon, hac_lags)
    out, nobs, notes = _cw_core(yy, bench, fc, lag)
    return EvaluationResult(
        test="clark_west",
        names=labels,
        estimate=out["adj"],
        statistic=out["t"],
        pvalue=normal_pvalue(out["t"], "greater"),
        reference="N(0,1) one-sided",
        alternative="greater",
        n_obs=nobs,
        horizon=int(horizon),
        hac=f"bartlett(L={lag})",
        details={
            "mspe_benchmark": out["mspe_benchmark"],
            "mspe_model": out["mspe_model"],
            "mspe_difference": out["mspe_benchmark"] - out["mspe_model"],
        },
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Out-of-sample R^2 (Campbell-Thompson)
# --------------------------------------------------------------------------- #
def historical_mean_benchmark(
    y: Any, *, horizon: int = 1, min_periods: int = 60
) -> np.ndarray:
    """Expanding historical mean known at ``t``: ``mean(y_0 .. y_{t-h})``.

    ``nan`` until ``min_periods`` finite observations are available (a fixed
    count, so the series is prefix-invariant); missing values are skipped.
    """
    yy = as_vector(y, "y")
    if horizon < 1 or min_periods < 1:
        raise ValueError("`horizon` and `min_periods` must be >= 1.")
    fin = np.isfinite(yy)
    s = np.cumsum(np.where(fin, yy, 0.0))
    c = np.cumsum(fin)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(c >= min_periods, s / np.maximum(c, 1), np.nan)
    out = np.full(yy.shape, np.nan)
    out[horizon:] = mean[: yy.size - horizon]
    return out


def historical_mean_expr(
    target: str,
    *,
    entity: str | None = None,
    horizon: int = 1,
    min_periods: int = 60,
) -> pl.Expr:
    """Polars twin of :func:`historical_mean_benchmark`, per entity.

    Expects rows sorted by time within each entity; ``NaN`` and null are
    skipped. Evaluated ``.over(entity)`` when ``entity`` is given.
    """
    if horizon < 1 or min_periods < 1:
        raise ValueError("`horizon` and `min_periods` must be >= 1.")
    y = pl.col(target).cast(pl.Float64).fill_nan(None)
    s = y.fill_null(0.0).cum_sum()
    c = y.is_not_null().cast(pl.Int64).cum_sum()
    mean = pl.when(c >= min_periods).then(s / c).otherwise(None)
    out = mean.shift(horizon)
    if entity is not None:
        out = out.over(entity)
    return out.alias(f"{target}_hist_mean")


def oos_r2(
    y: Any,
    forecasts: Any,
    *,
    benchmark: str | Any = "historical_mean",
    min_periods: int = 60,
    horizon: int = 1,
    truncate_at: float | None = None,
    hac_lags: int | None = None,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Campbell-Thompson out-of-sample R^2 with a Clark-West p-value.

    ``R^2_OS = 1 - sum e_i^2 / sum e_0^2`` over the dates where the benchmark
    and the model both exist.

    Parameters
    ----------
    y : array-like of shape (T,)
        Realisations.
    forecasts : array-like of shape (T,) or (T, M)
        Model forecasts, each made with information up to ``t - horizon``.
    benchmark : {"historical_mean", "zero"} or array-like of shape (T,)
        ``"historical_mean"`` (default) is the expanding mean of ``y`` known at
        ``t - horizon`` with at least ``min_periods`` observations -- never the
        full-sample mean, which uses future data. ``"zero"`` reproduces the
        Gu-Kelly-Xiu convention of :func:`panelary.embed.baseline_report`.
    min_periods : int, default 60
        Fixed warm-up for the historical mean.
    horizon : int, default 1
    truncate_at : float, optional
        Campbell-Thompson restriction: floor the model forecasts at this value
        (``0.0`` for a non-negative equity premium).
    hac_lags : int, optional
        Bartlett lag for the CW statistic (default ``horizon - 1``).
    names : sequence of str, optional

    Returns
    -------
    EvaluationResult
        ``estimate`` is ``R^2_OS``; ``statistic``/``pvalue`` are Clark-West's
        (valid when the model nests the benchmark, as any model with an
        intercept nests the historical mean). ``details``: ``r2_path`` and
        ``cumulative_sse_difference`` (Goyal-Welch), both ``(M, T)`` and
        prefix-invariant, and ``benchmark_forecast`` ``(1, T)``.
    """
    yy, fc, labels = _forecasts(y, forecasts, names)
    n = yy.shape[0]
    if isinstance(benchmark, str):
        if benchmark == "historical_mean":
            bench = historical_mean_benchmark(
                yy, horizon=horizon, min_periods=min_periods
            )
        elif benchmark == "zero":
            bench = np.zeros(n)
        else:
            raise ValueError(
                "`benchmark` must be 'historical_mean', 'zero' or an array."
            )
    else:
        bench = as_vector(benchmark, "benchmark", n)
    if truncate_at is not None:
        fc = np.where(np.isfinite(fc), np.maximum(fc, float(truncate_at)), fc)
    lag = _lags(horizon, hac_lags)
    out, nobs, notes = _cw_core(yy, bench, fc, lag)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - out["sse_model"] / out["sse_benchmark"]
    valid = np.isfinite(yy)[:, None] & np.isfinite(bench)[:, None] & np.isfinite(fc)
    e0 = np.where(valid, (yy - bench)[:, None] ** 2, 0.0)
    e1 = np.where(valid, (yy[:, None] - fc) ** 2, 0.0)
    c0, c1 = np.cumsum(e0, axis=0), np.cumsum(e1, axis=0)
    seen = np.cumsum(valid, axis=0) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.where(seen & (c0 > 0), 1.0 - c1 / c0, np.nan)
    gw = np.where(seen, c0 - c1, np.nan)
    return EvaluationResult(
        test="oos_r2",
        names=labels,
        estimate=r2,
        statistic=out["t"],
        pvalue=normal_pvalue(out["t"], "greater"),
        reference="N(0,1) one-sided (Clark-West)",
        alternative="greater",
        n_obs=nobs,
        horizon=int(horizon),
        hac=f"bartlett(L={lag})",
        details={
            "r2_path": path.T.copy(),
            "cumulative_sse_difference": gw.T.copy(),
            "benchmark_forecast": bench[None, :],
        },
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Mincer-Zarnowitz
# --------------------------------------------------------------------------- #
def _lagged_cross(a: np.ndarray, b: np.ndarray, lag: int) -> np.ndarray:
    """``sum_j w_j (sum_t a_t b_{t-j} + a_{t-j} b_t)`` plus ``sum a b`` (Bartlett), per column."""
    n = a.shape[0]
    out = _hac.coldot(a, b)
    for j in range(1, min(lag, n - 1) + 1):
        w = 1.0 - j / (lag + 1.0)
        out = out + w * (_hac.coldot(a[j:], b[:-j]) + _hac.coldot(a[:-j], b[j:]))
    return out


def mincer_zarnowitz(
    y: Any,
    forecasts: Any,
    *,
    horizon: int = 1,
    cov: str = "hac",
    hac_lags: int | None = None,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Mincer-Zarnowitz rationality test ``H0: (alpha, beta) = (0, 1)``.

    Regression ``y = alpha + beta yhat + u`` in the centred form
    ``y = a + beta (yhat - mean(yhat)) + u`` (diagonal ``X'X``), with the Wald
    statistic of ``(a, beta) = (mean(yhat), 1)`` -- identical to the uncentred
    test because the reparametrisation is fixed.

    Parameters
    ----------
    cov : {"hac", "ols"}, default "hac"
        ``"hac"``: Bartlett sandwich (``horizon - 1`` lags; White at ``h = 1``),
        ``chi2(2)``. ``"ols"``: exact ``F(2, T-2)`` under Gaussian i.i.d. errors.

    Returns
    -------
    EvaluationResult
        ``estimate`` is the slope ``beta``; ``details`` has ``intercept``
        (``alpha``), ``intercept_se`` and ``slope_se``.
    """
    if cov not in ("hac", "ols"):
        raise ValueError("`cov` must be 'hac' or 'ols'.")
    yy, fc, labels = _forecasts(y, forecasts, names)
    lag = _lags(horizon, hac_lags)
    stack = np.stack([np.broadcast_to(yy[:, None], fc.shape), fc])

    def fn(rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
        n = rows.size
        yv = yy[rows]
        f = fc[rows][:, cols]
        mf = _hac.colmean(f)
        xt = f - mf
        sxx = _hac.coldot(xt, xt)
        ybar = float(_hac.colmean(yv[:, None])[0])
        yc = np.broadcast_to(yv[:, None], xt.shape)
        beta = _hac.coldot(xt, yc) / sxx
        u = yc - ybar - xt * beta
        if cov == "hac":
            s11 = _lagged_cross(u, u, lag) / n
            s22 = _lagged_cross(xt * u, xt * u, lag) / n
            s12 = _lagged_cross(u, xt * u, lag) / n
            v11 = s11 / n
            v22 = n * s22 / (sxx * sxx)
            v12 = s12 / sxx
        else:
            s2 = _hac.coldot(u, u) / (n - 2.0)
            v11 = s2 / n
            v22 = s2 / sxx
            v12 = np.zeros_like(s2)
        d1 = ybar - mf
        d2 = beta - 1.0
        det = v11 * v22 - v12 * v12
        with np.errstate(divide="ignore", invalid="ignore"):
            wald = (v22 * d1 * d1 - 2.0 * v12 * d1 * d2 + v11 * d2 * d2) / det
            if cov == "hac":
                p = np.exp(-0.5 * wald)
            else:
                fstat = wald / 2.0
                x = (n - 2.0) / (n - 2.0 + 2.0 * fstat)
                p = np.array(
                    [
                        float(_special.betainc(0.5 * (n - 2.0), 1.0, xi))
                        if np.isfinite(xi)
                        else np.nan
                        for xi in np.atleast_1d(x)
                    ]
                )
        # uncentred alpha = a - beta * mean(yhat); its variance by the delta map
        alpha = ybar - beta * mf
        var_alpha = v11 - 2.0 * mf * v12 + mf * mf * v22
        return {
            "beta": beta,
            "wald": np.where(det > 0, wald, np.nan),
            "p": np.where(det > 0, p, np.nan),
            "alpha": alpha,
            "alpha_se": np.sqrt(np.maximum(var_alpha, 0.0)),
            "beta_se": np.sqrt(np.maximum(v22, 0.0)),
        }

    out, nobs, notes = _column_pass(
        stack, fn, fc.shape[1], ("beta", "wald", "p", "alpha", "alpha_se", "beta_se")
    )
    return EvaluationResult(
        test="mincer_zarnowitz",
        names=labels,
        estimate=out["beta"],
        statistic=out["wald"],
        pvalue=out["p"],
        reference="chi2(2)" if cov == "hac" else "F(2,T-2)",
        alternative="two-sided",
        n_obs=nobs,
        std_error=out["beta_se"],
        horizon=int(horizon),
        hac=f"bartlett(L={lag})" if cov == "hac" else None,
        details={
            "intercept": out["alpha"],
            "intercept_se": out["alpha_se"],
            "slope_se": out["beta_se"],
        },
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Pesaran-Timmermann
# --------------------------------------------------------------------------- #
def _fisher_greater(
    a: np.ndarray, row1: np.ndarray, col1: np.ndarray, n: int
) -> np.ndarray:
    """One-sided Fisher exact p ``P(A >= a)``, hypergeometric via log-factorials."""
    lf = np.asarray(
        _special.lgamma(np.arange(n + 2, dtype=np.float64) + 1.0), dtype=np.float64
    )
    out = np.empty(a.size)
    for i in range(a.size):
        k_lo = max(0, int(row1[i] + col1[i] - n))
        k_hi = int(min(row1[i], col1[i]))
        k = np.arange(k_lo, k_hi + 1)
        r1, c1 = int(row1[i]), int(col1[i])
        logp = (
            lf[c1]
            - lf[k]
            - lf[c1 - k]
            + lf[n - c1]
            - lf[r1 - k]
            - lf[n - c1 - r1 + k]
            - (lf[n] - lf[r1] - lf[n - r1])
        )
        top = logp.max()
        weights = np.exp(logp - top)
        tail = weights[k >= a[i]].sum()
        out[i] = min(1.0, tail / weights.sum())
    return out


def pesaran_timmermann(
    y: Any,
    forecasts: Any,
    *,
    horizon: int = 1,
    exact: bool = False,
    threshold: float = 0.0,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Pesaran-Timmermann test of directional accuracy.

    Directions are ``y > threshold`` and ``yhat > threshold``.

    * ``horizon == 1`` (default): PT (1992),
      ``S = (P - P*) / sqrt(V(P) - V(P*))`` with ``P*`` the hit rate under
      independence; one-sided ``N(0,1)``.
    * ``exact=True``: Fisher's exact test on the 2x2 table (one-sided,
      hypergeometric); valid for i.i.d. directions.
    * ``horizon > 1``: PT (2009)-style HAC t-statistic of the slope in the
      regression of the realised direction on the forecast direction (Bartlett
      with ``horizon - 1`` lags), because PT (1992) assumes independence.

    Returns
    -------
    EvaluationResult
        ``estimate`` is the hit rate; ``details`` has ``hit_rate_independence``
        (``P*``), ``share_up_realised`` and ``share_up_forecast``.
    """
    yy, fc, labels = _forecasts(y, forecasts, names)
    if horizon < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    notes: list[str] = []
    if exact and horizon > 1:
        notes.append(
            "the exact test assumes independent directions; with horizon > 1 it is approximate."
        )
    mode = "exact" if exact else ("hac" if horizon > 1 else "pt92")
    lag = horizon - 1
    stack = np.stack([np.broadcast_to(yy[:, None], fc.shape), fc])

    def fn(rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
        n = rows.size
        yd = (yy[rows] > threshold).astype(np.float64)[:, None]
        xd = (fc[rows][:, cols] > threshold).astype(np.float64)
        yb = np.broadcast_to(yd, xd.shape)
        hit = _hac.colmean((yb == xd).astype(np.float64))
        py = float(yd.mean())
        px = _hac.colmean(xd)
        pstar = py * px + (1.0 - py) * (1.0 - px)
        out: dict[str, np.ndarray] = {
            "hit": hit,
            "pstar": pstar,
            "py": np.full(len(cols), py),
            "px": px,
        }
        degenerate = (px <= 0) | (px >= 1) | (py <= 0) | (py >= 1)
        if mode == "pt92":
            v_p = pstar * (1.0 - pstar) / n
            v_ps = (
                (2 * py - 1) ** 2 * px * (1 - px) / n
                + (2 * px - 1) ** 2 * py * (1 - py) / n
                + 4 * py * px * (1 - py) * (1 - px) / n**2
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (hit - pstar) / np.sqrt(v_p - v_ps)
            s = np.where(degenerate | ~(v_p - v_ps > 0), np.nan, s)
            out["stat"] = s
            out["p"] = normal_pvalue(s, "greater")
        elif mode == "exact":
            a = _hac.coldot(yb, xd)
            row1 = np.round(px * n)
            col1 = np.full(len(cols), round(py * n))
            p = _fisher_greater(np.round(a), row1, col1, n)
            out["stat"] = a
            out["p"] = np.where(degenerate, np.nan, p)
        else:
            xc = xd - px
            sxx = _hac.coldot(xc, xc)
            with np.errstate(divide="ignore", invalid="ignore"):
                b = _hac.coldot(xc, yb) / sxx
                u = yb - float(yd.mean()) - xc * b
                s22 = _lagged_cross(xc * u, xc * u, lag) / n
                se = np.sqrt(n * s22) / sxx
                t = b / se
            t = np.where(degenerate, np.nan, t)
            out["stat"] = t
            out["p"] = normal_pvalue(t, "greater")
        return out

    out, nobs, gnotes = _column_pass(
        stack, fn, fc.shape[1], ("hit", "pstar", "py", "px", "stat", "p")
    )
    if np.any(np.isnan(out["p"]) & (nobs >= _MIN_OBS)):
        notes.append("a series with no variation in direction gives nan.")
    reference = {
        "pt92": "N(0,1) one-sided",
        "exact": "hypergeometric-exact one-sided",
        "hac": f"N(0,1) one-sided (HAC regression, bartlett(L={lag}))",
    }[mode]
    return EvaluationResult(
        test="pesaran_timmermann",
        names=labels,
        estimate=out["hit"],
        statistic=out["stat"],
        pvalue=out["p"],
        reference=reference,
        alternative="greater",
        n_obs=nobs,
        horizon=int(horizon),
        hac=f"bartlett(L={lag})" if mode == "hac" else None,
        details={
            "hit_rate_independence": out["pstar"],
            "share_up_realised": out["py"],
            "share_up_forecast": out["px"],
        },
        warnings=tuple(dict.fromkeys(notes + gnotes)),
    )


# --------------------------------------------------------------------------- #
# HLN encompassing
# --------------------------------------------------------------------------- #
def encompassing_test(
    y: Any,
    forecast_a: Any,
    forecasts_b: Any,
    *,
    horizon: int = 1,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Harvey-Leybourne-Newbold (1998) forecast-encompassing test.

    ``H0``: forecast A encompasses forecast B (B adds nothing). With
    ``d_t = (e_A - e_B) e_A``, the HLN-corrected DM statistic on ``d`` is
    referred to ``t(T-1)``, one-sided (``E d > 0`` rejects). The reverse
    direction (B encompasses A) is in ``details``.

    Returns
    -------
    EvaluationResult
        ``estimate`` is the optimal combination weight on B,
        ``sum d / sum (e_A - e_B)^2``; ``details``: ``mean_d``,
        ``reverse_statistic``, ``reverse_pvalue``.
    """
    yy, fb, labels = _forecasts(y, forecasts_b, names)
    fa = as_vector(forecast_a, "forecast_a", yy.shape[0])
    ea = (yy - fa)[:, None]
    eb = yy[:, None] - fb
    d = (ea - eb) * ea
    rev = (eb - ea) * eb
    fwd = _dm_columns(d, horizon=horizon, alternative="greater")
    bwd = _dm_columns(
        np.where(np.isfinite(d), rev, np.nan), horizon=horizon, alternative="greater"
    )
    diff2 = np.where(np.isfinite(d), (ea - eb) ** 2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.nansum(np.where(np.isfinite(d), d, 0.0), axis=0) / np.nansum(
            diff2, axis=0
        )
    weight = np.where(fwd["n"] >= 3, weight, np.nan)
    notes = []
    if np.any((fwd["n"] >= 3) & ~np.isfinite(fwd["statistic"])):
        notes.append("zero long-run variance of d (identical forecasts?) gives nan.")
    return EvaluationResult(
        test="encompassing",
        names=labels,
        estimate=weight,
        statistic=fwd["statistic"],
        pvalue=fwd["pvalue"],
        reference="t(T-1) one-sided (HLN)",
        alternative="greater",
        n_obs=fwd["n"].astype(np.int64),
        horizon=int(horizon),
        hac=f"bartlett(L={horizon - 1})",
        details={
            "mean_d": fwd["mean"],
            "reverse_statistic": bwd["statistic"],
            "reverse_pvalue": bwd["pvalue"],
        },
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Giacomini-White
# --------------------------------------------------------------------------- #
def giacomini_white(
    loss_a: Any,
    loss_b: Any,
    *,
    instruments: str | Any = "lagged",
    horizon: int = 1,
    estimation_scheme: str = "rolling",
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Giacomini-White (2006) test of conditional predictive ability.

    ``Z_t = h_t Delta L_{t+tau}`` with ``tau = horizon`` and test function
    ``h_t`` = ``(1, Delta L_t)`` (``instruments="lagged"``), ``(1,)``
    (``"constant"``: unconditional) or a user ``(T, q)`` array.
    ``W = n Zbar' Omega^{-1} Zbar ~ chi2(q)`` with ``Omega`` the outer-product
    mean (``tau = 1``) or Bartlett(``tau - 1``).

    Valid for forecasts from a **rolling** (fixed-size) estimation window only;
    ``estimation_scheme="expanding"`` returns the statistic with a warning.

    Returns
    -------
    EvaluationResult
        ``estimate`` is ``mean(Delta L)``; ``details['decision_share']`` is the
        share of dates where the fitted rule ``delta' h_t`` predicts model A's
        loss to exceed B's (i.e. prefers B).
    """
    if estimation_scheme not in ("rolling", "fixed", "expanding"):
        raise ValueError(
            "`estimation_scheme` must be 'rolling', 'fixed' or 'expanding'."
        )
    d, labels = _losses(loss_a, loss_b, names)
    tau = int(horizon)
    if tau < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    notes: list[str] = []
    if estimation_scheme == "expanding":
        notes.append(
            "Giacomini-White asymptotics need a rolling (fixed-size) estimation window; "
            "with an expanding window the test is not valid."
        )
    user = None
    if not isinstance(instruments, str):
        user = np.asarray(instruments, dtype=np.float64)
        if user.ndim == 1:
            user = user[:, None]
        if user.shape[0] != d.shape[0]:
            raise ValueError("`instruments` must have one row per date.")
    elif instruments not in ("lagged", "constant"):
        raise ValueError("`instruments` must be 'lagged', 'constant' or an array.")
    q = user.shape[1] if user is not None else (2 if instruments == "lagged" else 1)
    m = d.shape[1]
    stat = np.full(m, np.nan)
    pval = np.full(m, np.nan)
    share = np.full(m, np.nan)
    nobs = np.zeros(m, dtype=np.int64)
    mask = np.isfinite(d)
    if user is not None:
        mask &= np.all(np.isfinite(user), axis=1)[:, None]
    for rows, cols in _groups(mask):
        x = d[rows][:, cols]
        n_all = rows.size
        n = n_all - tau
        nobs[cols] = max(n, 0)
        if n < max(_MIN_OBS, q + 2):
            notes.append(f"{len(cols)} model(s) have too few observations; nan.")
            continue
        lead = x[tau:]
        if user is not None:
            h = [np.broadcast_to(user[rows][:-tau, [k]], lead.shape) for k in range(q)]
        else:
            h = [np.ones_like(lead)]
            if q == 2:
                h.append(x[:-tau])
        z = [hk * lead for hk in h]
        zbar = np.stack([_hac.colmean(zk) for zk in z], axis=1)  # (Mc, q)
        omega = np.empty((len(cols), q, q))
        for k in range(q):
            for k2 in range(k, q):
                # (k, k2) entry of Gamma_0 + sum_j w_j (Gamma_j + Gamma_j'), uncentred
                val = _lagged_cross(z[k], z[k2], tau - 1) / n
                omega[:, k, k2] = val
                omega[:, k2, k] = val
        try:
            sol = np.linalg.solve(omega, zbar[..., None])[..., 0]
        except np.linalg.LinAlgError:
            notes.append("singular instrument covariance; nan.")
            continue
        w = n * np.einsum("mq,mq->m", zbar, sol)
        stat[cols] = w
        pval[cols] = np.asarray(chi2_sf(w, float(q)), dtype=np.float64)
        # decision rule: regress Delta L_{t+tau} on h_t
        gram = np.empty((len(cols), q, q))
        rhs = np.empty((len(cols), q))
        for k in range(q):
            rhs[:, k] = _hac.coldot(h[k], lead)
            for k2 in range(q):
                gram[:, k, k2] = _hac.coldot(h[k], h[k2])
        try:
            delta = np.linalg.solve(gram, rhs[..., None])[..., 0]
            fitted = np.zeros_like(lead)
            for k in range(q):
                fitted += h[k] * delta[:, k]
            share[cols] = _hac.colmean(np.greater(fitted, 0.0).astype(np.float64))
        except np.linalg.LinAlgError:
            pass
    est = np.full(m, np.nan)
    for rows, cols in _groups(np.isfinite(d)):
        if rows.size:
            est[cols] = _hac.colmean(d[rows][:, cols])
    return EvaluationResult(
        test="giacomini_white",
        names=labels,
        estimate=est,
        statistic=stat,
        pvalue=pval,
        reference=f"chi2({q})",
        alternative="two-sided",
        n_obs=nobs,
        horizon=tau,
        hac=f"bartlett(L={tau - 1})" if tau > 1 else "outer-product",
        details={"decision_share": share},
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Giacomini-Rossi fluctuation and one-time reversal
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class FluctuationResult:
    """Giacomini-Rossi (2010) fluctuation test / monitor.

    Attributes
    ----------
    path : numpy.ndarray
        ``(P, M)`` standardised rolling mean loss differential ``F_t``,
        labelled at the **end** of its window (``nan`` before the first full
        window). GR label at the centre; the set of windows, and hence the
        supremum, is the same.
    critical_value : float
        Two-sided ``k_alpha(mu)`` from the shipped table (``nan`` in monitor
        mode when ``mu = m / P`` is outside ``[0.1, 0.9]``).
    max_stat, argmax_index, reject, pvalue : numpy.ndarray
        ``(M,)``: ``max_t |F_t|``, its row, ``max > critical_value`` and the
        table-interpolated p-value.
    mode : str
        ``"test"`` (full-sample HAC; not prefix-invariant) or ``"monitor"``
        (expanding HAC with a fixed lag; the path is prefix-invariant).
    window : int
        Window length ``m``; ``mu = window / P``.
    """

    names: tuple[str, ...]
    path: np.ndarray
    critical_value: float
    max_stat: np.ndarray
    argmax_index: np.ndarray
    reject: np.ndarray
    pvalue: np.ndarray
    mode: str
    window: int
    mu: float
    alpha: float
    horizon: int
    hac: str
    n_obs: np.ndarray
    warnings: tuple[str, ...] = field(default=())

    def to_evaluation(self) -> EvaluationResult:
        """The test as an :class:`EvaluationResult` (``statistic = max |F_t|``)."""
        m = len(self.names)
        return EvaluationResult(
            test="fluctuation",
            names=self.names,
            estimate=self.max_stat,
            statistic=self.max_stat,
            pvalue=self.pvalue,
            reference=f"gr-table(mu={self.mu:.3g})",
            alternative="two-sided",
            n_obs=self.n_obs,
            horizon=self.horizon,
            hac=self.hac,
            details={
                "critical_value": np.full(m, self.critical_value),
                "argmax_index": self.argmax_index.astype(np.float64),
                "reject": self.reject.astype(np.float64),
                "prefix_invariant": np.full(m, float(self.mode == "monitor")),
            },
            warnings=self.warnings,
        )

    def to_frame(self) -> pl.DataFrame:
        """Long frame of the path: ``index, model, statistic``."""
        p, m = self.path.shape
        return pl.DataFrame(
            {
                "index": np.tile(np.arange(p), m),
                "model": np.repeat(np.asarray(self.names, dtype=object), p).tolist(),
                "statistic": self.path.T.ravel(),
            }
        )


def fluctuation_test(
    loss_a: Any,
    loss_b: Any,
    *,
    window: float | int = 0.3,
    mode: str = "test",
    horizon: int = 1,
    hac_lags: int | None = None,
    alpha: float = 0.05,
    names: Sequence[str] | None = None,
) -> FluctuationResult:
    """Giacomini-Rossi (2010) fluctuation test of equal predictive ability over time.

    ``F_t = m^{-1/2} sum_{j=t-m+1}^{t} Delta L_j / sigma_hat``; the null of equal
    accuracy at every date is rejected when ``max_t |F_t| > k_alpha(mu)``.

    Parameters
    ----------
    loss_a, loss_b : array-like of shape (P,) or (P, M)
        Out-of-sample losses; ``Delta L = loss_a - loss_b``.
    window : float or int, default 0.3
        ``mode="test"``: the window share ``mu``, one of 0.1, ..., 0.9 (or an
        integer ``m`` with ``m / P`` equal to one of them). ``mode="monitor"``:
        a fixed integer ``m`` (a share would make the path depend on ``P``).
    mode : {"test", "monitor"}, default "test"
        ``"test"``: sigma from the full-sample Bartlett LRV (GR's test; the path
        is **not** prefix-invariant). ``"monitor"``: sigma_t from an expanding
        Bartlett LRV with a fixed lag, so ``path[t]`` uses rows ``<= t`` only;
        a diagnostic that makes no anytime-validity claim.
    horizon : int, default 1
    hac_lags : int, optional
        Fixed Bartlett lag (default ``horizon - 1``); data-driven bandwidths are
        rejected.
    alpha : float, default 0.05
    names : sequence of str, optional

    Returns
    -------
    FluctuationResult
    """
    if mode not in ("test", "monitor"):
        raise ValueError("`mode` must be 'test' or 'monitor'.")
    lag = _lags(horizon, hac_lags)
    d, labels = _losses(loss_a, loss_b, names)
    p_total, m = d.shape
    if mode == "monitor":
        if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
            raise ValueError(
                "mode='monitor' needs a fixed integer `window`; a share of P would make "
                "the path depend on the sample length."
            )
        width = int(window)
    elif isinstance(window, (int, np.integer)) and not isinstance(window, bool):
        width = int(window)
    else:
        width = round(float(window) * p_total)
    if width < 2 or width > p_total:
        raise ValueError(f"window of {width} observations does not fit P={p_total}.")
    mu = width / p_total
    notes: list[str] = []
    if mode == "test" and not any(abs(mu - g) < 1e-9 for g in _gr_tables.MU_GRID):
        if isinstance(window, float) and any(
            abs(window - g) < 1e-9 for g in _gr_tables.MU_GRID
        ):
            mu = float(window)  # rounding of mu * P; keep the tabulated share
        else:
            raise ValueError(
                f"mode='test' needs a tabulated window share {_gr_tables.MU_GRID}; got {mu:.4g}."
            )
    path = np.full((p_total, m), np.nan)
    nobs = np.zeros(m, dtype=np.int64)
    for rows, cols in _groups(np.isfinite(d)):
        n = rows.size
        nobs[cols] = n
        if n < width:
            notes.append(
                f"{len(cols)} model(s) have fewer observations than the window; nan."
            )
            continue
        x = d[rows][:, cols]
        s = np.vstack([np.zeros((1, len(cols))), np.cumsum(x, axis=0)])
        roll = s[width:] - s[:-width]  # windows ending at positions width-1 .. n-1
        if mode == "test":
            sig = np.sqrt(np.atleast_1d(_hac.bartlett_lrv(x, min(lag, n - 1))))[None, :]
            sig_rows = np.broadcast_to(sig, roll.shape)
        else:
            sig_rows = np.sqrt(_hac.expanding_bartlett_lrv(x, lag))[width - 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = roll / (math.sqrt(width) * sig_rows)
        f = np.where(np.isfinite(f), f, np.nan)
        block = np.full((n, len(cols)), np.nan)
        block[width - 1 :] = f
        path[np.ix_(rows, cols)] = block
    absval = np.abs(path)
    has = np.any(np.isfinite(absval), axis=0)
    max_stat = np.where(
        has, np.nanmax(np.where(np.isfinite(absval), absval, -np.inf), axis=0), np.nan
    )
    argmax = np.where(
        has, np.argmax(np.where(np.isfinite(absval), absval, -np.inf), axis=0), -1
    )
    if _gr_tables.MU_GRID[0] - 1e-9 <= mu <= _gr_tables.MU_GRID[-1] + 1e-9:
        mu_c = min(max(mu, _gr_tables.MU_GRID[0]), _gr_tables.MU_GRID[-1])
        cv = _gr_tables.fluctuation_critical_value(mu_c, alpha)
        pval, clipped = _gr_tables.fluctuation_pvalue(max_stat, mu_c)
        if mode == "monitor" and not any(
            abs(mu - g) < 1e-9 for g in _gr_tables.MU_GRID
        ):
            notes.append(
                f"critical value interpolated in mu between tabulated shares (mu={mu:.3g})."
            )
    else:
        cv = float("nan")
        pval, clipped = np.full(m, np.nan), False
        notes.append(
            f"mu = m/P = {mu:.3g} is outside the tabulated [0.1, 0.9]; no critical value."
        )
    if clipped:
        notes.append("p-values beyond the table range are clipped to [0.005, 0.95].")
    reject = np.where(np.isfinite(max_stat), max_stat > cv, False)
    return FluctuationResult(
        names=labels,
        path=path,
        critical_value=float(cv),
        max_stat=max_stat,
        argmax_index=argmax.astype(np.int64),
        reject=reject,
        pvalue=pval,
        mode=mode,
        window=width,
        mu=float(mu),
        alpha=float(alpha),
        horizon=int(horizon),
        hac=f"bartlett(L={lag})" + (", expanding" if mode == "monitor" else ""),
        n_obs=nobs,
        warnings=tuple(dict.fromkeys(notes)),
    )


def one_time_reversal_test(
    loss_a: Any,
    loss_b: Any,
    *,
    trim: float = 0.15,
    horizon: int = 1,
    hac_lags: int | None = None,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Giacomini-Rossi (2010) one-time-reversal test.

    ``Phi = sup_{tau in [trim P, (1 - trim) P]} [LM_1 + LM_2(tau)]`` with
    ``LM_1 = P mean(Delta L)^2 / sigma^2`` and
    ``LM_2(tau) = [S_tau - (tau/P) S_P]^2 / (sigma^2 P (tau/P)(1 - tau/P))``.
    The limit is ``W(1)^2 + sup_r BB(r)^2 / (r (1 - r))``, from the shipped
    table (``trim`` in 0.15, 0.20).

    Returns
    -------
    EvaluationResult
        ``estimate`` is ``mean(Delta L)``; ``details``: ``break_index`` (the
        maximising ``tau``, as a row position), ``lm1``, ``lm2``,
        ``mean_before`` and ``mean_after``.
    """
    if not any(abs(trim - g) < 1e-9 for g in _gr_tables.TRIM_GRID):
        raise ValueError(f"`trim` must be one of {_gr_tables.TRIM_GRID}.")
    lag = _lags(horizon, hac_lags)
    d, labels = _losses(loss_a, loss_b, names)
    m = d.shape[1]
    keys = ("mean", "phi", "lm1", "lm2", "brk", "before", "after")
    out = {k: np.full(m, np.nan) for k in keys}
    nobs = np.zeros(m, dtype=np.int64)
    notes: list[str] = []
    for rows, cols in _groups(np.isfinite(d)):
        n = rows.size
        nobs[cols] = n
        lo, hi = round(trim * n), round((1.0 - trim) * n)
        if n < 20 or hi <= lo:
            notes.append(f"{len(cols)} model(s) have too few observations; nan.")
            continue
        x = d[rows][:, cols]
        sig2 = np.atleast_1d(_hac.bartlett_lrv(x, min(lag, n - 1)))
        s = np.cumsum(x, axis=0)
        total = s[-1]
        tau = np.arange(lo, hi + 1, dtype=np.float64)  # number of obs in the first part
        r = (tau / n)[:, None]
        part = s[lo - 1 : hi]  # S_tau for tau = lo .. hi
        with np.errstate(divide="ignore", invalid="ignore"):
            lm1 = n * (total / n) ** 2 / sig2
            lm2 = (part - r * total) ** 2 / (sig2 * n * r * (1.0 - r))
        best = np.argmax(np.where(np.isfinite(lm2), lm2, -np.inf), axis=0)
        k = np.arange(len(cols))
        t_star = (lo + best).astype(np.int64)
        out["mean"][cols] = total / n
        out["lm1"][cols] = lm1
        out["lm2"][cols] = lm2[best, k]
        out["phi"][cols] = lm1 + lm2[best, k]
        out["brk"][cols] = rows[t_star]  # first row of the second regime
        out["before"][cols] = part[best, k] / t_star
        out["after"][cols] = (total - part[best, k]) / (n - t_star)
    pval, clipped = _gr_tables.reversal_pvalue(out["phi"], trim)
    if clipped:
        notes.append("p-values beyond the table range are clipped to [0.005, 0.95].")
    return EvaluationResult(
        test="one_time_reversal",
        names=labels,
        estimate=out["mean"],
        statistic=out["phi"],
        pvalue=pval,
        reference=f"gr-table(trim={trim:g})",
        alternative="two-sided",
        n_obs=nobs,
        horizon=int(horizon),
        hac=f"bartlett(L={lag})",
        details={
            "break_index": out["brk"],
            "lm1": out["lm1"],
            "lm2": out["lm2"],
            "mean_before": out["before"],
            "mean_after": out["after"],
        },
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Panel adaptor
# --------------------------------------------------------------------------- #
def loss_panel(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    target: str,
    forecasts: str | Sequence[str],
    loss: str = "squared",
    quantile: float | None = None,
    aggregate: str = "date_mean",
    min_entities: int = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Turn a forecast panel into a ``(T, M)`` date series of losses.

    1. Keep rows where the target and **every** forecast are present (a common
       sample, so models are compared on the same entity-dates), sorted by
       ``(entity, time)``.
    2. Per-row losses in polars: ``"squared"`` ``(y - f)^2``, ``"absolute"``
       ``|y - f|``, or ``"quantile"`` (pinball at ``quantile``).
    3. ``aggregate="date_mean"``: the cross-sectional mean loss per date (same-date
       rows only), dropping dates with fewer than ``min_entities`` entities.
       ``aggregate="none"``: the per-row losses, ``(N_rows, M)``, with their dates
       (for per-entity tests and an FDR correction).

    Every array-level test in this module consumes the result.

    Returns
    -------
    dates : numpy.ndarray
        ``(T,)`` dates (sorted) or per-row dates for ``aggregate="none"``.
    losses : numpy.ndarray
        ``(T, M)`` float64.
    names : tuple of str
        The forecast column names.
    """
    cols = [forecasts] if isinstance(forecasts, str) else list(forecasts)
    if not cols:
        raise ValueError("`forecasts` must name at least one column.")
    if loss not in ("squared", "absolute", "quantile"):
        raise ValueError("`loss` must be 'squared', 'absolute' or 'quantile'.")
    if loss == "quantile" and (quantile is None or not 0.0 < quantile < 1.0):
        raise ValueError("loss='quantile' needs `quantile` in (0, 1).")
    if aggregate not in ("date_mean", "none"):
        raise ValueError("`aggregate` must be 'date_mean' or 'none'.")
    if min_entities < 1:
        raise ValueError("`min_entities` must be >= 1.")
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    num = [target, *cols]
    lf = (
        lf.select([entity, time, *num])
        .with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in num])
        .drop_nulls(subset=num)
        .sort([entity, time])
    )
    y = pl.col(target)

    def row_loss(c: str) -> pl.Expr:
        e = y - pl.col(c)
        if loss == "squared":
            return (e * e).alias(c)
        if loss == "absolute":
            return e.abs().alias(c)
        q = float(quantile)  # type: ignore[arg-type]
        return (pl.when(e >= 0).then(q * e).otherwise((q - 1.0) * e)).alias(c)

    rows = lf.select([pl.col(time), *[row_loss(c) for c in cols]])
    if aggregate == "none":
        df = rows.collect()
        return (
            df[time].to_numpy(),
            df.select(cols).to_numpy().astype(np.float64),
            tuple(cols),
        )
    agg = (
        rows.group_by(time)
        .agg([pl.col(c).mean() for c in cols] + [pl.len().alias("__n")])
        .filter(pl.col("__n") >= min_entities)
        .sort(time)
        .collect()
    )
    return (
        agg[time].to_numpy(),
        agg.select(cols).to_numpy().astype(np.float64),
        tuple(cols),
    )
