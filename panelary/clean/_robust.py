"""Robust location/scale statistics as Polars expressions (and numpy windows).

These are the building blocks of :class:`~panelary.clean.OutlierCleaner`:

* :func:`robust_bounds_exprs` -- per-column ``(lo, hi)`` aggregation
  expressions for the MAD, IQR, z-score and quantile rules, usable in any
  ``group_by(...).agg(...)`` (per entity) or ``select`` (pooled).
* :func:`trailing_median_sigma` -- exact rolling median and robust sigma
  (MAD-based) over the
  *previous* ``window`` observations of each entity (numpy sliding windows;
  Polars has no native rolling MAD, and ``rolling_map`` is off-limits).

Constants: ``1.4826 * MAD`` estimates a normal standard deviation
(``1 / Phi^-1(3/4)``); when MAD is zero -- more than half the values tie --
the scale falls back to ``1.2533 * mean absolute deviation``
(``sqrt(pi / 2)``), the Iglewicz-Hoaglin convention.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

__all__ = ["robust_bounds_exprs", "robust_scale_expr", "trailing_median_sigma"]

MAD_TO_SIGMA = 1.4826
MEANAD_TO_SIGMA = 1.2533

BoundsMethod = Literal["mad", "iqr", "zscore", "quantile"]


def _clean(col: str) -> pl.Expr:
    """Float64 view of ``col`` with NaN treated as missing."""
    return pl.col(col).cast(pl.Float64).fill_nan(None)


def robust_scale_expr(col: str) -> pl.Expr:
    """Robust sigma of ``col``: ``1.4826 * MAD``, or ``1.2533 * MeanAD`` if 0."""
    x = _clean(col)
    med = x.median()
    mad = (x - med).abs().median()
    meanad = (x - x.mean()).abs().mean()
    return pl.when(mad > 0).then(MAD_TO_SIGMA * mad).otherwise(MEANAD_TO_SIGMA * meanad)


def robust_bounds_exprs(
    col: str,
    *,
    method: BoundsMethod,
    k: float,
    quantiles: tuple[float, float] = (0.01, 0.99),
    lo_name: str,
    hi_name: str,
    n_name: str,
) -> list[pl.Expr]:
    """Aggregation expressions for ``col``'s outlier bounds.

    Parameters
    ----------
    col : str
        Numeric column.
    method : {"mad", "iqr", "zscore", "quantile"}
        ``"mad"``: median +/- k * robust sigma. ``"iqr"``: Tukey fences
        ``[Q1 - k*IQR, Q3 + k*IQR]``. ``"zscore"``: mean +/- k * std.
        ``"quantile"``: the ``quantiles`` themselves (winsorisation limits;
        ``k`` unused).
    k : float
        Width multiplier.
    quantiles : (float, float), default=(0.01, 0.99)
        Limits for ``"quantile"``.
    lo_name, hi_name, n_name : str
        Output names of the lower bound, upper bound and non-missing count.

    Returns
    -------
    list of polars.Expr
        Three aggregation expressions.
    """
    x = _clean(col)
    if method == "mad":
        med = x.median()
        s = robust_scale_expr(col)
        lo, hi = med - k * s, med + k * s
    elif method == "iqr":
        q1 = x.quantile(0.25, interpolation="linear")
        q3 = x.quantile(0.75, interpolation="linear")
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    elif method == "zscore":
        m, s = x.mean(), x.std()
        lo, hi = m - k * s, m + k * s
    elif method == "quantile":
        lo = x.quantile(quantiles[0], interpolation="linear")
        hi = x.quantile(quantiles[1], interpolation="linear")
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown bounds method {method!r}.")
    return [
        lo.cast(pl.Float64).alias(lo_name),
        hi.cast(pl.Float64).alias(hi_name),
        x.count().cast(pl.Int64).alias(n_name),
    ]


def trailing_median_sigma(
    values: np.ndarray, groups: np.ndarray, window: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Median and robust sigma of each row's previous ``window`` group values.

    Parameters
    ----------
    values : numpy.ndarray
        Float values, **sorted by (group, time)**; NaN marks missing.
    groups : numpy.ndarray
        Integer group code per row (same order); contiguous runs.
    window : int
        Number of previous observations in the window (the current row is
        excluded, so a spike cannot mask itself).
    min_periods : int
        Minimum non-missing values in the window; fewer gives NaN.

    Returns
    -------
    (median, sigma) : tuple of numpy.ndarray
        ``sigma`` is ``1.4826 * MAD`` (``1.2533 * MeanAD`` when the MAD is
        zero). NaN where the window is too short.
    """
    x = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups).astype(np.int64)
    n = x.size
    med = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    if n == 0:
        return med, sigma
    pad_x = np.concatenate([np.full(window, np.nan), x])
    # A sentinel group that never matches a real code pads the front.
    pad_g = np.concatenate([np.full(window, np.iinfo(np.int64).min), g])
    wx = np.lib.stride_tricks.sliding_window_view(pad_x, window)[:n]
    wg = np.lib.stride_tricks.sliding_window_view(pad_g, window)[:n]
    win = np.where(wg == g[:, None], wx, np.nan)
    enough = np.sum(~np.isnan(win), axis=1) >= min_periods
    if enough.any():
        sub = win[enough]
        with np.errstate(all="ignore"):
            m = np.nanmedian(sub, axis=1)
            mad = np.nanmedian(np.abs(sub - m[:, None]), axis=1)
            meanad = np.nanmean(np.abs(sub - np.nanmean(sub, axis=1)[:, None]), axis=1)
        med[enough] = m
        sigma[enough] = np.where(mad > 0, MAD_TO_SIGMA * mad, MEANAD_TO_SIGMA * meanad)
    return med, sigma
