"""Shared, dependency-free numerics for the econometric feature generators.

Everything in :mod:`panelary.econ.features` is built on this module:
scipy-free normal-distribution helpers, a small OLS kernel with HAC standard
errors, lag/deterministic-term construction, critical-value interpolation, and
the per-entity / rolling-window plumbing that turns a 1-D NumPy kernel into a
leak-safe panel feature.

Only :mod:`numpy` and :mod:`polars` are imported at module level -- there is no
``scipy`` dependency anywhere in this subpackage. Where a heavier optional
dependency would help it must be routed through
:func:`panelary._internal._deps.require` inside the calling function.

Causality convention
--------------------
Every rolling helper here is **trailing and right-aligned**: the value emitted at
row ``t`` is a function of ``x[t - window + 1 : t + 1]`` only. Nothing in this
module ever looks at a row after ``t``, which is what makes the generators built
on top of it invariant to appended future rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from panelary._internal._special import (
    norm_cdf as norm_cdf,
)
from panelary._internal._special import (
    norm_ppf as norm_ppf,
)
from panelary._internal._special import (
    norm_sf as norm_sf,
)

__all__ = [
    "OLSResult",
    "add_deterministic",
    "auto_hac_lags",
    "bartlett_lrvar",
    "demean_by_group",
    "entity_arrays",
    "interp_pvalue",
    "lagmat",
    "norm_cdf",
    "norm_ppf",
    "norm_sf",
    "ols",
    "per_entity_apply",
    "per_entity_reduce",
    "rolling_apply",
    "rolling_beta",
    "sorted_panel",
]


# --------------------------------------------------------------------------- #
# Normal distribution (scipy-free)
# --------------------------------------------------------------------------- #
# norm_cdf / norm_sf / norm_ppf live in `panelary._internal._special`
# (imported above and re-exported here for existing callers).


# --------------------------------------------------------------------------- #
# OLS + HAC
# --------------------------------------------------------------------------- #
class OLSResult(NamedTuple):
    """Result of a plain least-squares fit.

    Attributes
    ----------
    beta : numpy.ndarray
        Coefficients, one per column of ``X``.
    resid : numpy.ndarray
        Residuals ``y - X @ beta``.
    se : numpy.ndarray
        Homoskedastic standard errors.
    tstat : numpy.ndarray
        ``beta / se``.
    sigma2 : float
        Residual variance ``SSR / (n - k)``.
    nobs : int
        Number of observations used.
    df_resid : int
        Residual degrees of freedom ``n - k``.
    xtx_inv : numpy.ndarray
        ``(X'X)^-1`` (pseudo-inverse if ``X`` is rank deficient).
    """

    beta: np.ndarray
    resid: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    sigma2: float
    nobs: int
    df_resid: int
    xtx_inv: np.ndarray


def ols(X: np.ndarray, y: np.ndarray) -> OLSResult:
    """Least-squares fit of ``y`` on ``X`` with homoskedastic standard errors.

    Uses a pseudo-inverse so a rank-deficient design degrades gracefully (the
    minimum-norm solution) rather than raising -- important when a rolling
    window happens to contain a constant series.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    df_resid = max(n - k, 1)
    sigma2 = float(resid @ resid) / df_resid
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        tstat = np.where(se > 0, beta / np.where(se > 0, se, 1.0), np.nan)
    return OLSResult(beta, resid, se, tstat, sigma2, n, df_resid, xtx_inv)


def auto_hac_lags(nobs: int) -> int:
    """Newey-West automatic bandwidth ``floor(4 * (n / 100) ** (2 / 9))``."""
    if nobs <= 1:
        return 0
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def bartlett_lrvar(u: np.ndarray, lags: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of ``u``.

    ``gamma_0 + 2 * sum_{j=1..L} (1 - j / (L + 1)) * gamma_j``, with the
    autocovariances scaled by ``1 / n`` (not demeaned -- callers pass residuals).
    """
    u = np.asarray(u, dtype=float).ravel()
    n = u.shape[0]
    if n == 0:
        return float("nan")
    lrv = float(u @ u) / n
    for j in range(1, int(lags) + 1):
        if j >= n:
            break
        w = 1.0 - j / (lags + 1.0)
        lrv += 2.0 * w * float(u[j:] @ u[:-j]) / n
    return lrv


def newey_west_se(
    X: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray, lags: int
) -> np.ndarray:
    """HAC (Newey-West) standard errors for an OLS fit."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    u = np.asarray(resid, dtype=float).ravel()
    n = X.shape[0]
    h = X * u[:, None]
    s = h.T @ h
    for j in range(1, int(lags) + 1):
        if j >= n:
            break
        w = 1.0 - j / (lags + 1.0)
        g = h[j:].T @ h[:-j]
        s = s + w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# --------------------------------------------------------------------------- #
# Design-matrix helpers
# --------------------------------------------------------------------------- #
def lagmat(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return the ``(n - nlags, nlags)`` matrix of lags ``x_{t-1} .. x_{t-nlags}``.

    Row ``i`` corresponds to observation ``t = i + nlags`` of ``x``; column ``j``
    holds ``x[t - 1 - j]``. Strictly backward looking.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if nlags <= 0:
        return np.empty((n, 0), dtype=float)
    rows = n - nlags
    if rows <= 0:
        return np.empty((0, nlags), dtype=float)
    return np.column_stack([x[nlags - 1 - j : n - 1 - j] for j in range(nlags)])


def add_deterministic(nobs: int, trend: str, *, start: int = 1) -> np.ndarray:
    """Build the deterministic regressor block.

    Parameters
    ----------
    nobs : int
        Number of rows.
    trend : {"n", "c", "ct"}
        ``"n"`` no terms, ``"c"`` intercept, ``"ct"`` intercept + linear trend.
    start : int, default=1
        Value of the trend at the first row.
    """
    if trend == "n":
        return np.empty((nobs, 0), dtype=float)
    if trend == "c":
        return np.ones((nobs, 1), dtype=float)
    if trend == "ct":
        t = np.arange(start, start + nobs, dtype=float)
        return np.column_stack([np.ones(nobs), t])
    raise ValueError(f"`trend` must be one of 'n', 'c', 'ct'; got {trend!r}.")


def demean_by_group(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Subtract the within-group mean of ``values`` for each code in ``codes``."""
    values = np.asarray(values, dtype=float)
    codes = np.asarray(codes)
    uniq, inv = np.unique(codes, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=uniq.shape[0])
    counts = np.bincount(inv, minlength=uniq.shape[0])
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return values - means[inv]


# --------------------------------------------------------------------------- #
# Critical-value interpolation
# --------------------------------------------------------------------------- #
def interp_pvalue(
    stat: float,
    crit_values: Sequence[float],
    levels: Sequence[float],
    *,
    clip: tuple[float, float] = (1e-4, 0.9999),
) -> float:
    """Approximate a p-value by interpolating a published critical-value table.

    The interpolation happens in the *normal-quantile* domain
    (``Phi^-1(level)``), which linearises the tail far better than interpolating
    the levels directly, and is extrapolated (rather than clamped) outside the
    tabulated range so that a decisively-rejecting statistic still gets a small
    p-value. The result is an **approximation**, not an exact asymptotic p-value:
    it is exact at the tabulated points and monotone in ``stat`` everywhere.

    Parameters
    ----------
    stat : float
        The realised test statistic.
    crit_values : sequence of float
        Critical values, in the same order as ``levels``.
    levels : sequence of float
        Significance levels in ``(0, 1)`` matching ``crit_values``.
    clip : (float, float)
        Bounds applied to the returned p-value.

    Notes
    -----
    The tail direction is implied by the table itself and needs no flag: for a
    lower-tail test (ADF, DF-GLS, Phillips-Perron, Zivot-Andrews) the critical
    values *increase* with the significance level, while for an upper-tail test
    (KPSS) they *decrease*, so mapping ``stat -> level`` through the sorted table
    gives the correctly-oriented p-value in both cases.
    """
    stat = float(stat)
    if not np.isfinite(stat):
        return float("nan")
    crit = np.asarray(crit_values, dtype=float)
    lev = np.asarray(levels, dtype=float)
    order = np.argsort(crit)
    crit, lev = crit[order], lev[order]
    z = np.asarray(norm_ppf(lev), dtype=float)
    if crit.shape[0] == 1:
        return float(np.clip(lev[0], *clip))
    if stat <= crit[0]:
        slope = (z[1] - z[0]) / (crit[1] - crit[0])
        z_hat = z[0] + slope * (stat - crit[0])
    elif stat >= crit[-1]:
        slope = (z[-1] - z[-2]) / (crit[-1] - crit[-2])
        z_hat = z[-1] + slope * (stat - crit[-1])
    else:
        z_hat = float(np.interp(stat, crit, z))
    return float(np.clip(float(norm_cdf(z_hat)), *clip))


# --------------------------------------------------------------------------- #
# Rolling / per-entity plumbing
# --------------------------------------------------------------------------- #
def rolling_apply(
    x: np.ndarray,
    window: int,
    func: Callable[[np.ndarray], float],
    *,
    min_periods: int | None = None,
) -> np.ndarray:
    """Apply ``func`` to every **trailing** window of ``x``.

    Output ``out[t] = func(x[max(0, t - window + 1) : t + 1])`` whenever that
    slice has at least ``min_periods`` finite observations, else ``nan``. This is
    the causal primitive under every rolling feature in this subpackage: row
    ``t`` never sees ``x[t + 1]``, so appending future rows cannot change it.

    Notes
    -----
    ``func`` is called once per row in Python. That is fine for the
    hundreds-of-rows-per-entity windows these estimators target, but it is not a
    vectorised kernel -- prefer a Polars ``rolling_*`` expression when one
    exists (see :func:`rolling_beta`).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    window = int(window)
    if window < 1:
        raise ValueError(f"`window` must be >= 1, got {window!r}.")
    mp = window if min_periods is None else int(min_periods)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - window + 1)
        chunk = x[lo : t + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.shape[0] < mp:
            continue
        try:
            out[t] = float(func(chunk))
        except (
            ValueError,
            ZeroDivisionError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            out[t] = np.nan
    return out


def sorted_panel(
    df: pl.DataFrame | pl.LazyFrame, entity: str, time: str
) -> pl.DataFrame:
    """Collect ``df`` and sort it by ``(entity, time)``.

    Every panel-level function here returns rows in this canonical order, so the
    caller can align outputs positionally.
    """
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    for col in (entity, time):
        if col not in frame.columns:
            raise ValueError(
                f"column {col!r} not found in frame; available: {frame.columns}."
            )
    return frame.sort([entity, time])


def entity_arrays(
    df: pl.DataFrame, entity: str, columns: Sequence[str]
) -> list[tuple[object, np.ndarray, dict[str, np.ndarray]]]:
    """Split a ``(entity, time)``-sorted frame into per-entity NumPy arrays.

    Returns one ``(entity_key, row_positions, {column: values})`` tuple per
    entity, where ``row_positions`` indexes back into ``df``.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} not found in frame; available: {df.columns}."
        )
    ent = df.get_column(entity).to_numpy()
    order = np.arange(ent.shape[0])
    data = {c: df.get_column(c).cast(pl.Float64).to_numpy() for c in columns}
    out: list[tuple[object, np.ndarray, dict[str, np.ndarray]]] = []
    if ent.shape[0] == 0:
        return out
    # `df` is sorted by entity, so groups are contiguous.
    boundaries = np.flatnonzero(ent[1:] != ent[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [ent.shape[0]]])
    for lo, hi in zip(starts, stops, strict=True):
        idx = order[lo:hi]
        out.append((ent[lo], idx, {c: v[lo:hi] for c, v in data.items()}))
    return out


def per_entity_apply(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    columns: Sequence[str],
    func: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
) -> pl.DataFrame:
    """Run a per-entity, row-aligned NumPy kernel and attach its output columns.

    ``func`` receives ``{column: values}`` for one entity (in time order) and must
    return ``{new_column: values}`` arrays of the *same length*. The result is
    ``df`` sorted by ``(entity, time)`` with the new columns appended.
    """
    frame = sorted_panel(df, entity, time)
    groups = entity_arrays(frame, entity, columns)
    n = frame.height
    buffers: dict[str, np.ndarray] = {}
    for _key, idx, data in groups:
        produced = func(data)
        for name, values in produced.items():
            arr = np.asarray(values, dtype=float).ravel()
            if arr.shape[0] != idx.shape[0]:
                raise ValueError(
                    f"per-entity kernel returned {arr.shape[0]} values for "
                    f"{idx.shape[0]} rows of column {name!r}."
                )
            if name not in buffers:
                buffers[name] = np.full(n, np.nan, dtype=float)
            buffers[name][idx] = arr
    if not buffers:
        return frame
    # NaN is the natural in-kernel "no value" marker, but Polars treats NaN as a
    # valid float; convert to real nulls so `is_null()` / `drop_nulls()` behave.
    return frame.with_columns(
        [pl.Series(name, values).fill_nan(None) for name, values in buffers.items()]
    )


def per_entity_reduce(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    columns: Sequence[str],
    func: Callable[[dict[str, np.ndarray]], dict[str, float]],
) -> pl.DataFrame:
    """Run a per-entity NumPy kernel that collapses to one row per entity.

    ``func`` receives ``{column: values}`` for one entity (in time order) and
    returns a flat ``{name: scalar}`` mapping. The result has one row per entity
    with the entity key plus those scalars.
    """
    frame = sorted_panel(df, entity, time)
    groups = entity_arrays(frame, entity, columns)
    keys: list[object] = []
    records: list[dict[str, float]] = []
    for key, _idx, data in groups:
        keys.append(key)
        records.append(func(data))
    if not records:
        return pl.DataFrame({entity: []})
    names: list[str] = []
    for rec in records:
        for name in rec:
            if name not in names:
                names.append(name)
    out: dict[str, object] = {entity: keys}
    for name in names:
        values = [rec.get(name, None) for rec in records]
        if all(v is None or isinstance(v, (int, float, np.floating)) for v in values):
            out[name] = pl.Series(
                name,
                [None if v is None else float(v) for v in values],
                dtype=pl.Float64,
            )
        else:
            out[name] = pl.Series(name, values)
    return pl.DataFrame(out)


def rolling_beta(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    y: str,
    x: str,
    window: int,
    min_periods: int | None = None,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing rolling univariate beta of ``y`` on ``x``, per entity.

    ``beta_t = Cov_t(y, x) / Var_t(x)`` over the trailing ``window`` rows within
    each entity, computed with Polars' native rolling moments (no Python loop).
    Row ``t`` uses only rows ``t - window + 1 .. t`` of its own entity, so it is
    leak-safe by construction.
    """
    frame = sorted_panel(df, entity, time)
    mp = window if min_periods is None else int(min_periods)
    name = alias or f"{y}_beta_{x}_{window}"
    cov = (
        (pl.col(y) * pl.col(x)).rolling_mean(window, min_samples=mp)
        - pl.col(y).rolling_mean(window, min_samples=mp)
        * pl.col(x).rolling_mean(window, min_samples=mp)
    ).over(entity)
    var = (
        (pl.col(x) ** 2).rolling_mean(window, min_samples=mp)
        - pl.col(x).rolling_mean(window, min_samples=mp) ** 2
    ).over(entity)
    return frame.with_columns(
        pl.when(var > 0).then(cov / var).otherwise(None).alias(name)
    )
