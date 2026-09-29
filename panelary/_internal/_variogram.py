"""Trailing multi-lag variograms and their log-log scaling slope.

The q-th order variogram of a series ``x`` at lag ``D`` over the trailing window
of ``W`` rows ending at ``t`` is

    m_q(D, t) = mean over s in [t - W + 1 + D, t] of |x_s - x_{s-D}|^q,

the ``W - D`` increments whose two ends both lie in the window. Its growth in
``D`` measures roughness: for a process with Hurst exponent ``H``,
``m_q(D) ~ D^(qH)``, so ``H = slope(log m_q, log D) / q``. That one scaling law
is behind the rough-volatility estimator of Gatheral, Jaisson & Rosenbaum (2018)
(``q = 2`` on a log-variance proxy) and the generalized Hurst exponent of Di
Matteo et al. (2005) (``q = 1, 2`` on a price or cumulative-return level).

Two forms, one definition:

* :func:`variogram_exprs` -- one polars expression per lag, each a native
  ``rolling_mean`` over ``W - D`` rows (no cumsum-difference; see the numerics
  rules of ``plans/todo/ohlc-volatility-and-liquidity.md`` section 7.7). The
  expressions are entity-agnostic: wrap each in ``.over(entity)``.
* :func:`trailing_variogram` -- the numpy twin, used as the test oracle and for
  dense arrays. :func:`sample_variogram` is the whole-array version, for
  estimators that fit on a training fold.

A measured-with-noise series ``x_s = y_s + e_s`` with independent errors of
variance ``nu_s`` has ``E(x_s - x_{s-D})^2 = E(y_s - y_{s-D})^2 + nu_s + nu_{s-D}``,
so passing ``noise`` subtracts the trailing mean of ``nu_s + nu_{s-D}`` over the
same pairs (``q = 2`` only). Without it, measurement noise flattens the
variogram at short lags and the fitted ``H`` is biased towards zero.

The OLS slope of ``log m`` on ``log D`` is a fixed linear combination of the
``log m`` values, so :func:`log_slope_weights` precomputes it once per lag set.

Leaf module: numpy and polars only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

__all__ = [
    "check_lags",
    "fit_power_law",
    "log_slope_weights",
    "sample_variogram",
    "trailing_variogram",
    "variogram_exprs",
]


def check_lags(lags: Sequence[int], window: int | None = None) -> tuple[int, ...]:
    """Validate a lag set: at least two distinct positive integers, ascending.

    With ``window`` given, the largest lag must leave at least two increments
    in the window (``max(lags) <= window - 2``).
    """
    out = tuple(int(d) for d in lags)
    if len(out) < 2:
        raise ValueError(f"need at least two lags to fit a slope, got {lags!r}.")
    if any(d < 1 for d in out):
        raise ValueError(f"lags must be positive integers, got {lags!r}.")
    if len(set(out)) != len(out) or list(out) != sorted(out):
        raise ValueError(f"lags must be distinct and ascending, got {lags!r}.")
    if window is not None and out[-1] > int(window) - 2:
        raise ValueError(
            f"the largest lag ({out[-1]}) leaves fewer than two increments in a "
            f"window of {window} rows; use a longer window or shorter lags."
        )
    return out


def log_slope_weights(lags: Sequence[int]) -> np.ndarray:
    """OLS slope weights ``w_D = (l_D - mean l) / sum (l - mean l)^2``, ``l = log D``.

    ``sum_D w_D * log m(D)`` is the least-squares slope of ``log m`` on
    ``log D``; the weights sum to zero, so a constant factor in ``m`` drops out.
    """
    ell = np.log(np.asarray(check_lags(lags), dtype=np.float64))
    centred = ell - ell.mean()
    return centred / float(centred @ centred)


def fit_power_law(m: np.ndarray, lags: Sequence[int]) -> tuple[float, float]:
    """``(slope, intercept)`` of the OLS fit ``log m(D) = a + b log D``.

    Returns ``(nan, nan)`` unless every ``m(D)`` is finite and positive.
    """
    vals = np.asarray(m, dtype=np.float64)
    ell = np.log(np.asarray(check_lags(lags), dtype=np.float64))
    if vals.shape != ell.shape or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        return float("nan"), float("nan")
    y = np.log(vals)
    slope = float(log_slope_weights(lags) @ y)
    return slope, float(y.mean() - slope * ell.mean())


def _min_pairs(lag: int, window: int, min_periods: int | None) -> int:
    """Minimum valid increments for ``lag``: ``min_periods - lag``, at least 1."""
    if min_periods is None:
        return window - lag
    return max(1, min(window - lag, int(min_periods) - lag))


def variogram_exprs(
    x: pl.Expr,
    lags: Sequence[int],
    window: int,
    *,
    q: float = 2.0,
    noise: pl.Expr | None = None,
    min_periods: int | None = None,
) -> list[pl.Expr]:
    """Trailing variogram ``m_q(D, t)`` per lag, as polars expressions.

    Each expression is ``rolling_mean(|x - x.shift(D)|^q, W - D)`` over the
    increments that are valid (both ends non-null, and both noise values
    non-null when ``noise`` is given). Nulls mark missing values; a NaN is a
    value to polars, so callers turn NaN into null first. Wrap each expression
    in ``.over(entity)``: the shift and the rolling mean must both stay inside
    the entity.

    Parameters
    ----------
    x : polars.Expr
        The series (Float64).
    lags : sequence of int
        Distinct ascending positive lags.
    window : int
        Trailing window ``W`` in rows; lag ``D`` averages ``W - D`` increments.
    q : float, default 2.0
        Moment order.
    noise : polars.Expr, optional
        Per-row measurement-error variance ``nu_s``; subtracts the trailing mean
        of ``nu_s + nu_{s-D}`` over the same increments. Requires ``q == 2``.
    min_periods : int, optional
        Minimum valid rows; lag ``D`` needs ``max(1, min_periods - D)`` valid
        increments. ``None`` requires all ``W - D``.

    Returns
    -------
    list of polars.Expr
        One Float64 expression per lag, null where too few increments are valid.
    """
    window = int(window)
    checked = check_lags(lags, window)
    if noise is not None and q != 2.0:
        raise ValueError("the noise correction is only valid for q == 2.")
    out: list[pl.Expr] = []
    for lag in checked:
        inc = x - x.shift(lag)
        valid = inc.is_not_null()
        if noise is not None:
            pair_noise = noise + noise.shift(lag)
            valid = valid & pair_noise.is_not_null()
        moment = inc.abs().pow(q) if q != 2.0 else inc * inc
        span = window - lag
        mp = _min_pairs(lag, window, min_periods)
        m = pl.when(valid).then(moment).rolling_mean(span, min_samples=mp)
        if noise is not None:
            m = m - pl.when(valid).then(pair_noise).rolling_mean(span, min_samples=mp)
        out.append(m)
    return out


def _increments(
    x: np.ndarray, lag: int, q: float, noise: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """``|x_s - x_{s-D}|^q`` (and ``nu_s + nu_{s-D}``) aligned at ``s``; NaN if invalid."""
    n = x.shape[0]
    inc = np.full(n, np.nan)
    if lag < n:
        d = x[lag:] - x[:-lag]
        inc[lag:] = d * d if q == 2.0 else np.abs(d) ** q
    pair: np.ndarray | None = None
    if noise is not None:
        pair = np.full(n, np.nan)
        if lag < n:
            pair[lag:] = noise[lag:] + noise[:-lag]
        bad = ~(np.isfinite(inc) & np.isfinite(pair))
        inc[bad] = np.nan
        pair[bad] = np.nan
    return inc, pair


def trailing_variogram(
    x: np.ndarray,
    lags: Sequence[int],
    window: int,
    *,
    q: float = 2.0,
    noise: np.ndarray | None = None,
    min_periods: int | None = None,
) -> np.ndarray:
    """Numpy twin of :func:`variogram_exprs` for one series: shape ``(n, len(lags))``.

    Row ``t`` averages the valid increments ending at ``t - W + 1 + D .. t``
    (fewer at the start of the series, exactly as polars' ``rolling_mean``
    with ``min_samples``); ``nan`` where fewer than the required number are
    valid.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    nu = None if noise is None else np.asarray(noise, dtype=np.float64).ravel()
    window = int(window)
    checked = check_lags(lags, window)
    if nu is not None and q != 2.0:
        raise ValueError("the noise correction is only valid for q == 2.")
    n = arr.shape[0]
    out = np.full((n, len(checked)), np.nan)
    for col, lag in enumerate(checked):
        inc, pair = _increments(arr, lag, q, nu)
        span = window - lag
        mp = _min_pairs(lag, window, min_periods)
        pad = np.full(span - 1, np.nan)
        win = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, inc]), span)
        ok = np.isfinite(win)
        cnt = ok.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(ok, win, 0.0).sum(axis=1) / cnt
            if pair is not None:
                pwin = np.lib.stride_tricks.sliding_window_view(
                    np.concatenate([pad, pair]), span
                )
                mean = mean - np.where(ok, pwin, 0.0).sum(axis=1) / cnt
        out[:, col] = np.where(cnt >= mp, mean, np.nan)
    return out


def sample_variogram(
    x: np.ndarray,
    lags: Sequence[int],
    *,
    q: float = 2.0,
    noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-array variogram: ``(sums, counts)`` of the valid increments per lag.

    ``sums[i] / counts[i]`` is ``m_q(D_i)`` over every valid increment of ``x``
    (noise-corrected when ``noise`` is given). Sums and counts, rather than
    means, so several series can be pooled by adding them.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    nu = None if noise is None else np.asarray(noise, dtype=np.float64).ravel()
    checked = check_lags(lags)
    if nu is not None and q != 2.0:
        raise ValueError("the noise correction is only valid for q == 2.")
    sums = np.zeros(len(checked))
    counts = np.zeros(len(checked))
    for col, lag in enumerate(checked):
        inc, pair = _increments(arr, lag, q, nu)
        ok = np.isfinite(inc)
        counts[col] = float(ok.sum())
        total = float(inc[ok].sum())
        if pair is not None:
            total -= float(pair[ok].sum())
        sums[col] = total
    return sums, counts
