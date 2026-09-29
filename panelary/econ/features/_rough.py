"""Rough volatility: a trailing Hurst feature and the RFSV forecaster.

* :func:`rough_hurst` -- the trailing scaling exponent ``H`` of a daily
  log-variance proxy, from the noise-corrected variogram of Gatheral, Jaisson &
  Rosenbaum (2018): ``m(D) = E(x_{t+D} - x_t)^2 ~ D^{2H}``.
* :class:`RFSVForecaster` -- the rough fractional stochastic volatility
  forecast of ``log sigma^2_{t+D}``, with ``H`` fitted on the training fold.

Why the noise correction is not optional
----------------------------------------
A daily log-variance proxy is the true log variance plus measurement error. The
error adds ``nu_s + nu_{s-D}`` to every squared increment, which flattens the
variogram at short lags, so the fitted ``H`` is biased towards zero: a smooth
(``H = 0.5``) volatility read through daily Parkinson ranges comes out at
``H ~ 0.31`` (plan section 1e, reproduced in ``tests/test_rough_accuracy.py``).
Both functions therefore take ``noise_var`` as a required argument: a column of
per-day error variances (for ``log rv``, the ``log_rv_var`` measure of
:func:`~panelary.econ.features.intraday_realized_measures`), a constant (for a
range proxy, the variance of the log range term at your bar count), or ``0.0``
to state explicitly that the proxy is noise-free.

A descriptive feature, not a verdict
------------------------------------
``rough_hurst`` measures the roughness of the proxy path. Cont & Das (2024)
show that roughness can be an estimation artefact even when volatility is
smooth, and Fukasawa, Takabatake & Westphal (2022) still find ``H < 0.5`` after
correcting for measurement error. The correction here addresses independent
proxy errors only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import polars as pl

from panelary._internal._variogram import (
    check_lags,
    fit_power_law,
    log_slope_weights,
    sample_variogram,
    variogram_exprs,
)
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.econ.features._common import entity_arrays, sorted_panel

__all__ = ["RFSVForecaster", "rfsv_weights", "rough_hurst"]

_DEFAULT_LAGS: tuple[int, ...] = tuple(range(1, 11))

#: Fitted ``H`` is clipped into this interval: the RFSV kernel
#: ``(t - s)^{-(H + 1/2)}`` is integrable only for ``H < 1/2``.
_H_BOUNDS: tuple[float, float] = (0.01, 0.49)


def _noise_arg(noise_var: str | float, columns: Sequence[str]) -> str | float:
    """Validate ``noise_var``: a column name, or a finite non-negative constant."""
    if isinstance(noise_var, str):
        if noise_var not in columns:
            raise ValueError(
                f"noise_var column {noise_var!r} not found in frame; available: "
                f"{list(columns)}."
            )
        return noise_var
    if isinstance(noise_var, bool) or not isinstance(noise_var, (int, float)):
        raise TypeError(
            "`noise_var` must be a column name or a float (0.0 for a noise-free "
            f"proxy), got {type(noise_var).__name__}."
        )
    value = float(noise_var)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"a constant `noise_var` must be finite and >= 0, got {noise_var!r}."
        )
    return value


def rough_hurst(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    log_variance: str,
    noise_var: str | float,
    window: int = 500,
    lags: Sequence[int] = _DEFAULT_LAGS,
    min_periods: int | None = None,
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing Hurst exponent of a log-variance proxy, noise-corrected.

    For each lag ``D`` the trailing variogram ``m(D, t)`` is the mean of
    ``(x_s - x_{s-D})^2`` over the ``W - D`` increments inside the last ``W``
    rows, minus the mean of ``nu_s + nu_{s-D}`` over the same increments. Then
    ``H_t = (1/2) sum_D w_D log m(D, t)`` with the fixed OLS slope weights
    ``w_D`` of ``log m`` on ``log D``. Every term is a native trailing rolling
    mean within the entity, so row ``t`` uses rows ``t - W + 1 .. t`` only.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long daily panel.
    entity, time : str
        Panel keys.
    log_variance : str
        Daily log-variance proxy ``x`` (for example ``log rv`` or
        ``log rv_ss`` from :func:`intraday_realized_measures`).
    noise_var : str or float
        Measurement-error variance of ``x``: a column (per day; for ``log rv``
        use the ``log_rv_var`` measure, or ``log_rv_var_tpq`` under jumps), a
        constant (for a range proxy), or ``0.0`` for a noise-free proxy.
        Required, because without it the estimate is biased towards roughness.
    window : int, default 500
        Trailing window ``W`` in rows.
    lags : sequence of int, default 1..10
        Variogram lags, ascending; the largest must be ``<= W - 2``.
    min_periods : int, optional
        Minimum valid rows; lag ``D`` needs ``min_periods - D`` valid
        increments. ``None`` requires the full window.
    alias : str, optional
        Output name; default ``f"rough_hurst_{window}"``.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the Hurst column appended; null
        until enough history exists, or where a corrected ``m(D)`` is not
        positive (the noise swamps the signal at that lag).

    Notes
    -----
    Measured on a 1000-day fBM log variance with ``nu = 0.3`` (200 paths,
    ``tests/test_rough_accuracy.py``): the corrected estimate from ``log rv``
    of 78 returns a day is within 0.01 of the truth at ``H`` = 0.1, 0.3 and
    0.5, and from daily Parkinson ranges within 0.02, where the uncorrected
    Parkinson estimate reads 0.31 for a true 0.5.
    """
    window = int(window)
    checked = check_lags(lags, window)
    if min_periods is not None and not 1 <= int(min_periods) <= window:
        raise ValueError(f"`min_periods` must be in [1, window], got {min_periods!r}.")
    frame = sorted_panel(df, entity, time)
    if log_variance not in frame.columns:
        raise ValueError(
            f"column {log_variance!r} not found in frame; available: {frame.columns}."
        )
    noise = _noise_arg(noise_var, frame.columns)
    name = alias or f"rough_hurst_{window}"

    # Non-finite values (log of a zero variance, NaN) are missing, not data.
    x = pl.col(log_variance).cast(pl.Float64)
    x = pl.when(x.is_finite()).then(x)
    nu: pl.Expr | None = None
    if isinstance(noise, str):
        nu = pl.col(noise).cast(pl.Float64)
        nu = pl.when(nu.is_finite()).then(nu)
    moments = variogram_exprs(x, checked, window, noise=nu, min_periods=min_periods)
    tmp = [f"__rh_m{lag}" for lag in checked]
    # A constant noise variance contributes exactly 2 nu to every increment.
    offset = 2.0 * noise if isinstance(noise, float) else 0.0
    staged = frame.with_columns(
        [(m.over(entity) - offset).alias(c) for m, c in zip(moments, tmp, strict=True)]
    )
    weights = log_slope_weights(checked)
    positive = pl.all_horizontal([pl.col(c) > 0 for c in tmp])
    # A left-to-right chain, not `pl.sum_horizontal`: the horizontal sum's
    # reduction order depends on the thread count and frame size, which moves
    # the last bit and breaks bitwise prefix invariance (seen on 3 threads).
    terms = [float(w) * pl.col(c).log() for w, c in zip(weights, tmp, strict=True)]
    slope = terms[0]
    for term in terms[1:]:
        slope = slope + term
    return staged.with_columns(
        pl.when(positive).then(0.5 * slope).otherwise(None).alias(name)
    ).drop(tmp)


# --------------------------------------------------------------------------- #
# RFSV forecast
# --------------------------------------------------------------------------- #
def rfsv_weights(hurst: float, horizon: int, n_lags: int) -> np.ndarray:
    """Normalised RFSV forecast weights for lags ``k = 0 .. n_lags - 1``.

    Gatheral, Jaisson & Rosenbaum (2018): ``E[log sigma^2_{t+D} | F_t]`` is
    proportional to ``int_0^inf log sigma^2_{t-u} / ((u + D) u^{H + 1/2}) du``.
    Day ``t - k`` stands for ``u`` in ``[k, k + 1)``, so its weight is that
    integral over the interval (Gauss-Legendre; the ``u^{-(H+1/2)}``
    singularity at ``k = 0`` is removed by substituting
    ``u = v^{1 / (1/2 - H)}``). The weights are truncated at ``n_lags`` and
    normalised to sum to one, the finite-sample form GJR use.
    """
    h = float(hurst)
    if not 0.0 < h < 0.5:
        raise ValueError(f"`hurst` must be in (0, 0.5), got {hurst!r}.")
    d = float(horizon)
    if d <= 0:
        raise ValueError(f"`horizon` must be positive, got {horizon!r}.")
    n = int(n_lags)
    if n < 1:
        raise ValueError(f"`n_lags` must be >= 1, got {n_lags!r}.")
    a = h + 0.5
    nodes, wts = np.polynomial.legendre.leggauss(64)
    v = 0.5 * (nodes + 1.0)  # nodes on [0, 1]
    half = 0.5 * wts
    out = np.empty(n)
    # k = 0: int_0^1 u^{-a} / (u + D) du = (1/(1-a)) int_0^1 dv / (v^{1/(1-a)} + D).
    out[0] = float(half @ (1.0 / (v ** (1.0 / (1.0 - a)) + d))) / (1.0 - a)
    if n > 1:
        k = np.arange(1, n, dtype=np.float64)[:, None]
        u = k + v[None, :]
        out[1:] = (1.0 / ((u + d) * u**a)) @ half
    return out / out.sum()


def _rfsv_apply(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Causal normalised FIR: ``sum_k w_k x_{t-k} / sum_k w_k [x_{t-k} valid]``.

    The ``_ffd`` shape (``numpy.convolve`` with the newest-first kernel): row
    ``t`` reads ``x[t - L + 1 .. t]`` only, and is ``nan`` for ``t < L - 1``,
    a rule that depends on the row's position, never on the series length.
    Missing days drop out of both sums, so the weights renormalise over the
    days that exist.
    """
    n = x.shape[0]
    width = weights.shape[0]
    valid = np.isfinite(x)
    num = np.convolve(np.where(valid, x, 0.0), weights)[:n]
    den = np.convolve(valid.astype(np.float64), weights)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    out[: min(width - 1, n)] = np.nan
    return out


class RFSVForecaster(PanelTransformer):
    """Rough fractional stochastic volatility forecast of the log variance.

    ``fit`` estimates ``H`` from the noise-corrected variogram of the training
    rows (pooled across entities, or per entity with a pooled fallback) and
    freezes it. ``transform`` emits, at each row ``t``, the RFSV forecast of
    ``log sigma^2_{t + horizon}``: a normalised causal weighted average of the
    trailing ``n_lags`` values of the proxy (:func:`rfsv_weights`).

    Parameters
    ----------
    log_variance : str
        Daily log-variance proxy.
    noise_var : str or float
        Measurement-error variance of the proxy, as in :func:`rough_hurst`
        (used for the ``H`` fit only).
    horizon : int, default 1
        Forecast horizon ``D`` in rows.
    lags : sequence of int, default 1..10
        Variogram lags for the ``H`` fit.
    n_lags : int, default 250
        FIR truncation ``L``: rows before an entity's ``L``-th are null.
    pooled : bool, default True
        One ``H`` for all entities (the rough-volatility literature finds it
        close to universal). ``False`` fits one per entity; entities with fewer
        than ``min_train_rows`` valid rows, or whose fit fails, use the pooled
        value.
    min_train_rows : int, default 250
        Minimum valid proxy rows for a per-entity fit.
    output : str, default "rfsv_forecast"
        Output column name.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    pooled_hurst_ : float
        The pooled ``H`` (clipped into ``[0.01, 0.49]``).
    hurst_ : dict
        ``{entity: H}`` for entities with their own fit (``pooled=False``).
    raw_hurst_ : float
        The pooled ``H`` before clipping.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        log_variance: str,
        noise_var: str | float,
        horizon: int = 1,
        lags: Sequence[int] = _DEFAULT_LAGS,
        n_lags: int = 250,
        pooled: bool = True,
        min_train_rows: int = 250,
        output: str = "rfsv_forecast",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if int(horizon) < 1:
            raise ValueError(f"`horizon` must be >= 1, got {horizon!r}.")
        if int(n_lags) < 1:
            raise ValueError(f"`n_lags` must be >= 1, got {n_lags!r}.")
        if isinstance(noise_var, str):
            pass
        elif isinstance(noise_var, bool) or not isinstance(noise_var, (int, float)):
            raise TypeError("`noise_var` must be a column name or a float.")
        self.log_variance = log_variance
        self.noise_var = noise_var
        self.horizon = int(horizon)
        self.lags = check_lags(lags)
        self.n_lags = int(n_lags)
        self.pooled = bool(pooled)
        self.min_train_rows = int(min_train_rows)
        self.output = output
        self.pooled_hurst_: float | None = None
        self.raw_hurst_: float | None = None
        self.hurst_: dict[object, float] = {}

    def _columns(self, frame: pl.DataFrame) -> list[str]:
        noise = _noise_arg(self.noise_var, frame.columns)
        cols = [self.log_variance]
        if isinstance(noise, str) and noise != self.log_variance:
            cols.append(noise)
        return cols

    def _variogram(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = data[self.log_variance]
        if isinstance(self.noise_var, str):
            return sample_variogram(x, self.lags, noise=data[self.noise_var])
        sums, counts = sample_variogram(x, self.lags)
        return sums - 2.0 * float(self.noise_var) * counts, counts

    def _fit(self, panel: PanelFrame) -> None:
        frame = sorted_panel(panel.collect(), panel.entity_col, panel.time_col)
        if self.log_variance not in frame.columns:
            raise ValueError(f"column {self.log_variance!r} not found in frame.")
        cols = self._columns(frame)
        total_sums = np.zeros(len(self.lags))
        total_counts = np.zeros(len(self.lags))
        own: dict[object, float] = {}
        for key, _idx, data in entity_arrays(frame, panel.entity_col, cols):
            sums, counts = self._variogram(data)
            total_sums += sums
            total_counts += counts
            if (
                not self.pooled
                and np.isfinite(data[self.log_variance]).sum() >= self.min_train_rows
            ):
                with np.errstate(invalid="ignore", divide="ignore"):
                    slope, _ = fit_power_law(sums / counts, self.lags)
                if math.isfinite(slope):
                    own[key] = slope / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            slope, _ = fit_power_law(total_sums / total_counts, self.lags)
        if not math.isfinite(slope):
            raise ValueError(
                "RFSVForecaster.fit: the noise-corrected variogram of the training "
                "rows is not positive at every lag, so H cannot be estimated. Check "
                "`noise_var` (too large?) or supply more history."
            )
        self.raw_hurst_ = slope / 2.0
        self.pooled_hurst_ = float(np.clip(self.raw_hurst_, *_H_BOUNDS))
        self.hurst_ = {k: float(np.clip(h, *_H_BOUNDS)) for k, h in own.items()}

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        assert self.pooled_hurst_ is not None  # set by fit
        frame = sorted_panel(panel.collect(), panel.entity_col, panel.time_col)
        if self.log_variance not in frame.columns:
            raise ValueError(f"column {self.log_variance!r} not found in frame.")
        preds = np.full(frame.height, np.nan)
        cache: dict[float, np.ndarray] = {}
        for key, idx, data in entity_arrays(
            frame, panel.entity_col, [self.log_variance]
        ):
            h = self.hurst_.get(key, self.pooled_hurst_)
            if h not in cache:
                cache[h] = rfsv_weights(h, self.horizon, self.n_lags)
            preds[idx] = _rfsv_apply(data[self.log_variance], cache[h])
        out = frame.with_columns(pl.Series(self.output, preds).fill_nan(None))
        return PanelFrame(out.lazy(), entity=panel.entity_col, time=panel.time_col)
