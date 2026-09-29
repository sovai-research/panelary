"""Microstructure liquidity features: price impact, spreads and zero returns.

Every feature is a trailing rolling statistic computed per entity with native
Polars rolling expressions, so it is cheap and causal: row ``t`` uses only rows
``t - window + 1 .. t`` of the same entity.

* **Amihud (2002) illiquidity** -- ``mean_t(|r| / dollar_volume)``, the average
  price impact per unit of traded value. Usually reported scaled by ``1e6``.
* **Roll (1984) effective spread** -- ``2 * sqrt(-Cov(dp_t, dp_{t-1}))`` when
  that autocovariance is negative (bid-ask bounce); undefined otherwise, which
  is reported as null or, optionally, floored at zero.
* **Amivest liquidity** -- ``mean_t(volume / |r|)``, the reciprocal-flavoured
  companion to Amihud.
* **Turnover** -- ``mean_t(volume / shares_outstanding)``.
* **Price impact** (:func:`price_impact`) -- a Kyle-lambda-style slope of
  returns on signed square-root dollar volume.
* **Pastor-Stambaugh gamma** (:func:`pastor_stambaugh_gamma`) -- the
  return-reversal coefficient on lagged signed dollar volume.
* **Zero-return share** (:func:`zero_return_share`) -- Lesmond, Ogden &
  Trzcinka's ``Zeros`` and Goyenko-Holden-Trzcinka's ``Zeros2``.
* **FHT spread** (:func:`fht_spread`) -- Fong, Holden & Trzcinka's closed-form
  descendant of the LOT estimator.

The regression proxies centre every variable on the entity's **first valid
value** before forming second moments: a constant per entity, so it is causal
and prefix-invariant, and it removes the cancellation of the naive
``E[xy] - E[x]E[y]`` (relative error 5.6e-4 at a mean/sd of 1e6, 6e-16 after
anchoring -- measured in the plan).

References
----------
Amihud (2002), *Journal of Financial Markets*; Roll (1984), *Journal of
Finance*; Goyenko, Holden & Trzcinka (2009) for the low-frequency-proxy survey;
Kyle (1985), *Econometrica*; Hasbrouck (2009), *J. Finance* 64(3); Pastor &
Stambaugh (2003), *J. Political Economy* 111(3); Lesmond, Ogden & Trzcinka
(1999), *Rev. Financial Studies* 12(5); Fong, Holden & Trzcinka (2017),
*Rev. Finance* 21(4).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from panelary._internal._special import norm_ppf
from panelary.econ.features._common import sorted_panel
from panelary.registry import FeatureSpec

__all__ = [
    "amihud_illiquidity",
    "roll_spread",
    "amivest_liquidity",
    "liquidity_features",
    "price_impact",
    "pastor_stambaugh_gamma",
    "zero_return_share",
    "fht_spread",
]

#: Conventional Amihud scaling (returns per million units of traded value).
AMIHUD_SCALE: float = 1e6


def amihud_illiquidity(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    dollar_volume: str,
    window: int = 21,
    min_periods: int | None = None,
    scale: float = AMIHUD_SCALE,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Amihud illiquidity ratio per entity.

    ``ILLIQ_t = scale * mean(|r_s| / dollar_volume_s)`` over the trailing
    ``window``. Rows with non-positive traded value contribute nothing (they are
    treated as missing) rather than producing an infinite ratio.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns : str
        Period return column.
    dollar_volume : str
        Traded value (price times volume) column.
    window : int, default=21
        Trailing window length.
    min_periods : int, optional
        Minimum observations in the window (defaults to ``window``).
    scale : float, default=1e6
        Multiplicative scaling of the raw ratio.
    alias : str, optional
        Output column name (defaults to ``"amihud_{window}"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the feature column appended.
    """
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"amihud_{window}"
    ratio = (
        pl.when(pl.col(dollar_volume) > 0)
        .then(pl.col(returns).abs() / pl.col(dollar_volume))
        .otherwise(None)
    )
    return frame.with_columns(
        (scale * ratio.rolling_mean(window, min_samples=mp).over(entity)).alias(name)
    )


def roll_spread(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    price: str | None = None,
    price_change: str | None = None,
    window: int = 21,
    min_periods: int | None = None,
    clip_positive: bool = False,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Roll (1984) effective-spread estimator per entity.

    ``S_t = 2 * sqrt(-Cov(dp_s, dp_{s-1}))`` over the trailing window, where
    ``dp`` is the price change. The estimator is only defined when the
    autocovariance is negative (the bid-ask-bounce case); positive
    autocovariance yields null, or ``0.0`` with ``clip_positive=True``.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    price : str, optional
        Price column, differenced internally. Mutually exclusive with
        ``price_change``.
    price_change : str, optional
        Pre-computed price change column.
    window : int, default=21
        Trailing window length.
    min_periods : int, optional
        Minimum observations (defaults to ``window``).
    clip_positive : bool, default=False
        Emit ``0.0`` instead of null where the autocovariance is non-negative.
    alias : str, optional
        Output column name (defaults to ``"roll_spread_{window}"``).

    Returns
    -------
    polars.DataFrame

    Notes
    -----
    ``price=`` differences price *levels*, so the estimate is in currency units
    and moves when a back-adjustment factor (a later split or dividend)
    rescales earlier prices. Passing log returns as ``price_change=`` gives a
    proportional spread that is invariant to that rescaling, and is the
    recommended input. The default is left unchanged so existing values do not
    move. For bid-ask spreads from OHLC bars see
    :func:`~panelary.econ.features.ohlc_spread`.
    """
    if (price is None) == (price_change is None):
        raise ValueError("pass exactly one of `price` or `price_change`.")
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"roll_spread_{window}"
    dp = (
        pl.col(price).diff().over(entity) if price is not None else pl.col(price_change)
    )
    dp_lag = dp.shift(1).over(entity)
    cov = (
        (dp * dp_lag).rolling_mean(window, min_samples=mp)
        - dp.rolling_mean(window, min_samples=mp)
        * dp_lag.rolling_mean(window, min_samples=mp)
    ).over(entity)
    spread = 2.0 * (-cov).sqrt()
    fallback = pl.lit(0.0) if clip_positive else pl.lit(None, dtype=pl.Float64)
    return frame.with_columns(
        pl.when(cov < 0).then(spread).otherwise(fallback).alias(name)
    )


def amivest_liquidity(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    volume: str,
    window: int = 21,
    min_periods: int | None = None,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Amivest liquidity ratio ``mean(volume / |r|)`` per entity.

    Rows with a zero return are treated as missing (the ratio is undefined).
    """
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"amivest_{window}"
    ratio = (
        pl.when(pl.col(returns).abs() > 0)
        .then(pl.col(volume) / pl.col(returns).abs())
        .otherwise(None)
    )
    return frame.with_columns(
        ratio.rolling_mean(window, min_samples=mp).over(entity).alias(name)
    )


def liquidity_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    price: str | None = None,
    dollar_volume: str | None = None,
    volume: str | None = None,
    shares_outstanding: str | None = None,
    windows: tuple[int, ...] = (21, 63),
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Compute the whole liquidity block over one or more trailing windows.

    Emits whichever measures the supplied columns allow: Amihud (needs
    ``dollar_volume``), Roll (needs ``price``), Amivest (needs ``volume``) and
    turnover (needs ``volume`` and ``shares_outstanding``), for each window in
    ``windows``.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the feature columns appended.
    """
    out = sorted_panel(df, entity, time)
    for window in windows:
        mp = window if min_periods is None else int(min_periods)
        if dollar_volume is not None:
            out = amihud_illiquidity(
                out,
                entity=entity,
                time=time,
                returns=returns,
                dollar_volume=dollar_volume,
                window=window,
                min_periods=mp,
            )
        if price is not None:
            out = roll_spread(
                out,
                entity=entity,
                time=time,
                price=price,
                window=window,
                min_periods=mp,
            )
        if volume is not None:
            out = amivest_liquidity(
                out,
                entity=entity,
                time=time,
                returns=returns,
                volume=volume,
                window=window,
                min_periods=mp,
            )
            if shares_outstanding is not None:
                out = out.with_columns(
                    (
                        pl.when(pl.col(shares_outstanding) > 0)
                        .then(pl.col(volume) / pl.col(shares_outstanding))
                        .otherwise(None)
                    )
                    .rolling_mean(window, min_samples=mp)
                    .over(entity)
                    .alias(f"turnover_{window}")
                )
        # Realized volatility of returns is the natural companion control.
        out = out.with_columns(
            pl.col(returns)
            .rolling_std(window, min_samples=mp)
            .over(entity)
            .alias(f"ret_vol_{window}")
        )
    return out


# --------------------------------------------------------------------------- #
# Low-frequency price-impact and spread proxies (plan 5, M3)
# --------------------------------------------------------------------------- #
_P = "__liq_"


def _check_int(name: str, value: object, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"`{name}` must be an integer >= {low}, got {value!r}.")
    return value


def _periods(window: int, min_periods: int | None, low: int) -> int:
    mp = window if min_periods is None else _check_int("min_periods", min_periods, low)
    if mp > window:
        raise ValueError(f"`min_periods` ({mp}) cannot exceed `window` ({window}).")
    return mp


def _check_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    ok = ok and math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    if not ok:
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"`{name}` must be a finite number {bound}, got {value!r}.")
    return float(value)


def _finite(col: str) -> pl.Expr:
    """``col`` as float64, with NaN and +-inf treated as missing."""
    x = pl.col(col).cast(pl.Float64)
    return pl.when(x.is_finite()).then(x).otherwise(None)


def _require(frame: pl.DataFrame, *cols: str | None) -> None:
    missing = [c for c in cols if c is not None and c not in frame.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} not found in frame; available: {frame.columns}."
        )


def _anchor(col: str, entity: str) -> pl.Expr:
    """``col`` minus its entity's first valid value: causal, prefix-invariant.

    Every row that has a value comes at or after that first value, so the
    anchor never reaches into the future.
    """
    return (pl.col(col) - pl.col(col).drop_nulls().first().over(entity)).alias(col)


def price_impact(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    dollar_volume: str | None = None,
    signed_volume: str | None = None,
    window: int = 63,
    min_periods: int | None = None,
    intercept: bool = True,
    volume_scale: float = 1e-6,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing price-impact slope (a Kyle-lambda proxy), per entity.

    ``lambda_t = Cov_w(r, x) / Var_w(x)`` over the trailing window (or
    ``sum(r x) / sum(x^2)`` with ``intercept=False``), where ``x`` is the signed
    order flow: ``signed_volume`` when you have it, else
    ``sign(r_t) * sqrt(dollar_volume_t * volume_scale)`` (Hasbrouck 2009's
    square-root form with the return sign standing in for the trade sign).

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns : str
        Period return column (log returns recommended).
    dollar_volume : str, optional
        Traded value; used to build ``x`` when ``signed_volume`` is not given.
        Negative values are treated as missing.
    signed_volume : str, optional
        A signed order-flow measure to use as ``x`` directly (e.g. buy minus
        sell dollar volume from TAQ). Pass exactly one of the two.
    window : int, default 63
        Trailing window length (rows).
    min_periods : int, optional
        Rows with both a return and an order flow required (default
        ``window``).
    intercept : bool, default True
        Regress with an intercept (covariance over variance) or through the
        origin.
    volume_scale : float, default 1e-6
        Multiplies dollar volume before the square root (``1e-6``: millions).
    alias : str, optional
        Output column name (defaults to ``f"price_impact_{window}"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the slope appended; null where
        the window's order flow has no variance.

    Notes
    -----
    **Read this before using it as a Kyle lambda.** With ``sign(r)`` standing in
    for the order-flow sign, ``Cov(r, x) = E[|r| sqrt(DV)] > 0`` mechanically,
    so the slope is a price-impact *magnitude* measure akin to Amihud rather
    than an estimate of Kyle's lambda. Goyenko, Holden & Trzcinka (2009) found
    Amihud's ratio the best low-frequency price-impact proxy against
    high-frequency benchmarks; this slope is offered as a complement, and as the
    right estimator when a genuine ``signed_volume`` is available. Second
    moments are anchored at each entity's first valid value (see the module
    notes).
    """
    if (dollar_volume is None) == (signed_volume is None):
        raise ValueError("pass exactly one of `dollar_volume` or `signed_volume`.")
    window = _check_int("window", window, 2 if intercept else 1)
    mp = _periods(window, min_periods, 2 if intercept else 1)
    scale = _check_positive("volume_scale", volume_scale)
    frame = sorted_panel(df, entity, time)
    _require(frame, returns, dollar_volume, signed_volume)

    r = _finite(returns)
    if signed_volume is not None:
        x = _finite(signed_volume)
    else:
        assert dollar_volume is not None
        dv = _finite(dollar_volume)
        x = pl.when(dv >= 0).then(r.sign() * (dv * scale).sqrt())
    ok = r.is_not_null() & x.is_not_null()
    rc, xc = f"{_P}r", f"{_P}x"
    work = frame.with_columns(
        pl.when(ok).then(r).alias(rc), pl.when(ok).then(x).alias(xc)
    )
    if intercept:
        work = work.with_columns(_anchor(rc, entity), _anchor(xc, entity))
    R, X = pl.col(rc), pl.col(xc)

    def mean(e: pl.Expr) -> pl.Expr:
        return e.rolling_mean(window, min_samples=mp).over(entity)

    work = work.with_columns(
        mean(R).alias(f"{_P}mr"),
        mean(X).alias(f"{_P}mx"),
        mean(R * X).alias(f"{_P}mrx"),
        mean(X * X).alias(f"{_P}mxx"),
    )
    mr, mx, mrx, mxx = (pl.col(f"{_P}{n}") for n in ("mr", "mx", "mrx", "mxx"))
    if intercept:
        var = mxx - mx * mx
        slope = pl.when(var > 1e-12 * mxx).then((mrx - mr * mx) / var)
    else:
        slope = pl.when(mxx > 0).then(mrx / mxx)
    return work.select(*frame.columns, slope.alias(alias or f"price_impact_{window}"))


def pastor_stambaugh_gamma(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    market_returns: str,
    dollar_volume: str,
    window: int = 21,
    min_periods: int | None = None,
    volume_scale: float = 1e-6,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Pastor-Stambaugh (2003) liquidity gamma, per entity.

    Over the window ending at ``t`` (all inputs ``<= t``), the OLS coefficient
    ``gamma`` of

        r^e_s = theta + phi r_{s-1} + gamma sign(r^e_{s-1}) v_{s-1} + e_s,

    with ``r^e = r - r^m`` the excess return over the market and ``v`` the
    dollar volume times ``volume_scale``. Illiquidity shows up as a *negative*
    gamma: order flow is followed by a partial reversal.

    Computed by Frisch-Waugh from window-centred moments of the anchored
    variables -- nine trailing means -- so no per-window regression runs:
    ``gamma = (S11 S2y - S12 S1y) / (S11 S22 - S12^2)``; null when that
    determinant is ``<= 1e-12 S11 S22`` (a collinear window).

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns, market_returns : str
        The asset's and the market's period returns (aligned per row).
    dollar_volume : str
        Traded value (negative values are treated as missing).
    window : int, default 21
        Trailing window length (rows, ``>= 3``). PS use a month of daily data.
    min_periods : int, optional
        Complete regression rows required (default ``window``, ``>= 3``). A row
        needs its own excess return and the previous row's return, excess
        return and volume.
    volume_scale : float, default 1e-6
        Multiplies dollar volume (``1e-6``: millions, as in PS).
    alias : str, optional
        Output column name (defaults to ``f"ps_gamma_{window}"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with ``gamma`` appended.

    Notes
    -----
    Pastor & Stambaugh themselves note that individual-stock gammas are very
    noisy; their liquidity factor averages them across stocks. Treat a single
    entity's value accordingly.
    """
    window = _check_int("window", window, 3)
    mp = _periods(window, min_periods, 3)
    scale = _check_positive("volume_scale", volume_scale)
    frame = sorted_panel(df, entity, time)
    _require(frame, returns, market_returns, dollar_volume)

    r = _finite(returns)
    dv = _finite(dollar_volume)
    ex, sv = f"{_P}ex", f"{_P}sv"
    work = frame.with_columns(
        (r - _finite(market_returns)).alias(ex), r.alias(f"{_P}r")
    ).with_columns((pl.col(ex).sign() * pl.when(dv >= 0).then(dv * scale)).alias(sv))
    work = work.with_columns(
        pl.col(f"{_P}r").shift(1).over(entity).alias(f"{_P}x1"),
        pl.col(sv).shift(1).over(entity).alias(f"{_P}x2"),
    )
    y, x1, x2 = f"{_P}y", f"{_P}x1", f"{_P}x2"
    ok = pl.all_horizontal(
        pl.col(ex).is_not_null(), pl.col(x1).is_not_null(), pl.col(x2).is_not_null()
    )
    work = work.with_columns(
        pl.when(ok).then(pl.col(ex)).alias(y),
        pl.when(ok).then(pl.col(x1)).alias(x1),
        pl.when(ok).then(pl.col(x2)).alias(x2),
    ).with_columns(_anchor(y, entity), _anchor(x1, entity), _anchor(x2, entity))
    Y, X1, X2 = pl.col(y), pl.col(x1), pl.col(x2)

    def mean(e: pl.Expr) -> pl.Expr:
        return e.rolling_mean(window, min_samples=mp).over(entity)

    names = ("m1", "m2", "my", "m11", "m22", "m12", "m1y", "m2y")
    exprs = (X1, X2, Y, X1 * X1, X2 * X2, X1 * X2, X1 * Y, X2 * Y)
    work = work.with_columns(
        mean(e).alias(f"{_P}{n}") for n, e in zip(names, exprs, strict=True)
    )
    m1, m2, my, m11, m22, m12, m1y, m2y = (pl.col(f"{_P}{n}") for n in names)
    work = work.with_columns(
        (m11 - m1 * m1).alias(f"{_P}s11"),
        (m22 - m2 * m2).alias(f"{_P}s22"),
        (m12 - m1 * m2).alias(f"{_P}s12"),
        (m1y - m1 * my).alias(f"{_P}s1y"),
        (m2y - m2 * my).alias(f"{_P}s2y"),
    )
    s11, s22, s12, s1y, s2y = (
        pl.col(f"{_P}{n}") for n in ("s11", "s22", "s12", "s1y", "s2y")
    )
    det = s11 * s22 - s12 * s12
    gamma = pl.when(det > 1e-12 * s11 * s22).then((s11 * s2y - s12 * s1y) / det)
    return work.select(*frame.columns, gamma.alias(alias or f"ps_gamma_{window}"))


def zero_return_share(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    volume: str | None = None,
    window: int = 21,
    min_periods: int | None = None,
    tol: float = 0.0,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing share of zero-return days, per entity (``Zeros``, ``Zeros2``).

    ``zeros_t`` is the mean over the window of ``1{|r| <= tol}``, over rows with
    a return (Lesmond, Ogden & Trzcinka 1999). With ``volume``, ``zeros2_t`` is
    the share of rows with a zero return **and positive volume**, over rows with
    both a return and a volume (Goyenko, Holden & Trzcinka 2009): a stale
    vendor fill -- the close carried forward on a no-trade day, volume 0 -- is a
    zero return that says nothing about transaction costs, and ``Zeros2``
    excludes it.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns : str
        Period return column. NaN / inf count as missing.
    volume : str, optional
        Share or dollar volume; adds the ``Zeros2`` column when given.
    window : int, default 21
        Trailing window length (rows).
    min_periods : int, optional
        Rows with a return required (default ``window``).
    tol : float, default 0.0
        ``|r| <= tol`` counts as zero (for returns rounded from prices on a
        tick grid, a tolerance below one tick).
    alias : str, optional
        Name of the ``Zeros`` column; ``Zeros2`` becomes ``f"{alias}2"``.
        Defaults to ``f"zeros_{window}"`` and ``f"zeros2_{window}"``.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the share(s) appended.
    """
    window = _check_int("window", window, 1)
    mp = _periods(window, min_periods, 1)
    tol = _check_positive("tol", tol, allow_zero=True)
    frame = sorted_panel(df, entity, time)
    _require(frame, returns, volume)

    r = _finite(returns)
    zero = (r.abs() <= tol).cast(pl.Float64)
    name = alias or f"zeros_{window}"
    out = [zero.rolling_mean(window, min_samples=mp).over(entity).alias(name)]
    if volume is not None:
        v = _finite(volume)
        both = r.is_not_null() & v.is_not_null()
        z2 = pl.when(both).then(((r.abs() <= tol) & (v > 0)).cast(pl.Float64))
        name2 = f"{alias}2" if alias else f"zeros2_{window}"
        out.append(z2.rolling_mean(window, min_samples=mp).over(entity).alias(name2))
    return frame.with_columns(out)


def _fht_quantiles(window: int) -> np.ndarray:
    """``norm_ppf((1 + k / n) / 2)`` on the ``(n, k)`` grid, flat by ``n (w + 1) + k``.

    The zero share is a ratio of two integer counts, so the inverse normal CDF
    is evaluated once per ``(n, k)`` pair here -- ``O(w^2)`` scalar calls at
    construction -- and gathered per row, never evaluated per row. Entries with
    ``k >= n`` (an all-zero window, where FHT is undefined) are NaN.
    """
    table = np.full((window + 1) * (window + 1), np.nan)
    for n in range(1, window + 1):
        k = np.arange(n)
        table[n * (window + 1) + k] = norm_ppf((1.0 + k / n) / 2.0)
    return table


def fht_spread(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    returns: str,
    window: int = 21,
    min_periods: int | None = None,
    tol: float = 0.0,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Fong-Holden-Trzcinka (2017) effective spread, per entity.

    ``FHT = 2 sigma Phi^{-1}((1 + z) / 2)``, with ``sigma`` the trailing sample
    standard deviation of returns (``ddof=1``) and ``z`` the trailing share of
    zero returns (``|r| <= tol``). It is the closed-form descendant of the LOT
    model: a zero return is read as a day on which the value change did not
    cover the transaction cost. Null when every return in the window is zero.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel.
    entity, time : str
        Panel keys.
    returns : str
        Period return column. NaN / inf count as missing.
    window : int, default 21
        Trailing window length (rows, ``>= 2``).
    min_periods : int, optional
        Rows with a return required (default ``window``, ``>= 2``).
    tol : float, default 0.0
        Zero-return tolerance, as in :func:`zero_return_share`.
    alias : str, optional
        Output column name (defaults to ``f"fht_{window}"``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the spread appended.

    Notes
    -----
    ``Phi^{-1}`` comes from a table of the ``(window + 1)^2`` possible count
    pairs, built once per call with
    :func:`panelary._internal._special.norm_ppf` (``O(window^2)``), so the
    per-row cost is a gather.
    """
    window = _check_int("window", window, 2)
    mp = _periods(window, min_periods, 2)
    tol = _check_positive("tol", tol, allow_zero=True)
    frame = sorted_panel(df, entity, time)
    _require(frame, returns)

    r = _finite(returns)
    n, k, sd = f"{_P}n", f"{_P}k", f"{_P}sd"
    work = frame.with_columns(
        r.is_not_null()
        .cast(pl.Int64)
        .rolling_sum(window, min_samples=1)
        .over(entity)
        .alias(n),
        (r.abs() <= tol)
        .fill_null(value=False)
        .cast(pl.Int64)
        .rolling_sum(window, min_samples=1)
        .over(entity)
        .alias(k),
        r.rolling_std(window, min_samples=mp).over(entity).alias(sd),
    )
    table = pl.Series(f"{_P}q", _fht_quantiles(window))
    q = pl.lit(table).gather(pl.col(n) * (window + 1) + pl.col(k))
    ok = (pl.col(n) >= mp) & (pl.col(k) < pl.col(n))
    spread = pl.when(ok).then(2.0 * pl.col(sd) * q)
    return work.select(*frame.columns, spread.alias(alias or f"fht_{window}"))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _spec(
    name: str,
    fn: Callable[..., Any],
    params: dict[str, type],
    tier: str,
    cost: str,
    source: str,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        namespace="econ",
        input_shape="frame",
        output_shape="frame",
        params=params,
        tier=tier,
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=f"Panelary (clean-room; {source})",
        license="Apache-2.0",
        backend_fn=fn,
        axis="time",
        flavour="trailing",
        cost_hint=cost,
    )


_SPECS: tuple[FeatureSpec, ...] = (
    _spec(
        "price_impact",
        price_impact,
        {"window": int, "min_periods": int, "intercept": bool, "volume_scale": float},
        "C",
        "O(N)",
        "Kyle 1985, Hasbrouck 2009",
    ),
    _spec(
        "pastor_stambaugh_gamma",
        pastor_stambaugh_gamma,
        {"window": int, "min_periods": int, "volume_scale": float},
        "C",
        "O(N)",
        "Pastor & Stambaugh 2003",
    ),
    _spec(
        "zero_return_share",
        zero_return_share,
        {"window": int, "min_periods": int, "tol": float},
        "B",
        "O(N)",
        "Lesmond, Ogden & Trzcinka 1999; Goyenko, Holden & Trzcinka 2009",
    ),
    _spec(
        "fht_spread",
        fht_spread,
        {"window": int, "min_periods": int, "tol": float},
        "B",
        "O(N + w^2)",
        "Fong, Holden & Trzcinka 2017",
    ),
)
