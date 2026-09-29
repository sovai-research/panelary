"""The trailing multi-lag variogram leaf: polars form, numpy twin, fit helpers.

Owned here for now (plan 8 was to create it and has not started; see
``plans/todo/00-cross-plan-coordination.md`` D6). The API is generic on
purpose: any moment order ``q`` (Di Matteo's generalized Hurst exponent uses
``q = 1, 2``), an optional measurement-noise correction, and ``min_periods``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary._internal._variogram import (
    check_lags,
    fit_power_law,
    log_slope_weights,
    sample_variogram,
    trailing_variogram,
    variogram_exprs,
)


def _polars(
    x: np.ndarray,
    lags: tuple[int, ...],
    window: int,
    *,
    q: float = 2.0,
    noise: np.ndarray | None = None,
    min_periods: int | None = None,
) -> np.ndarray:
    # NaN marks a missing value in the numpy twin; polars needs a real null.
    frame = pl.DataFrame(
        {"x": x, "nu": noise if noise is not None else np.zeros_like(x)}
    ).fill_nan(None)
    exprs = variogram_exprs(
        pl.col("x"),
        lags,
        window,
        q=q,
        noise=pl.col("nu") if noise is not None else None,
        min_periods=min_periods,
    )
    out = frame.select([e.alias(f"m{i}") for i, e in enumerate(exprs)])
    return out.to_numpy().astype(np.float64)


def _direct(x: np.ndarray, lag: int, window: int, t: int, q: float = 2.0) -> float:
    """``mean |x_s - x_{s-D}|^q`` over ``s`` in ``[t - W + 1 + D, t]``, by hand."""
    s = np.arange(max(lag, t - window + 1 + lag), t + 1)
    return float(np.mean(np.abs(x[s] - x[s - lag]) ** q))


@pytest.mark.parametrize("q", [1.0, 2.0])
def test_polars_numpy_and_direct_agree(q: float) -> None:
    x = np.cumsum(np.random.default_rng(0).standard_normal(300))
    lags, window = (1, 2, 5, 9), 40
    pol = _polars(x, lags, window, q=q)
    ref = trailing_variogram(x, lags, window, q=q)
    np.testing.assert_allclose(pol, ref, rtol=1e-12, equal_nan=True)
    for col, lag in enumerate(lags):
        assert np.isnan(ref[window - 2, col]) and np.isfinite(ref[window - 1, col])
        for t in (window - 1, 120, 299):
            assert ref[t, col] == pytest.approx(
                _direct(x, lag, window, t, q), rel=1e-12
            )


def test_noise_correction_and_missing_values() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(200))
    nu = rng.uniform(0.0, 0.2, 200)
    x[[30, 31, 100]] = np.nan
    nu[150] = np.nan
    lags, window = (1, 3, 6), 25
    pol = _polars(x, lags, window, noise=nu, min_periods=15)
    ref = trailing_variogram(x, lags, window, noise=nu, min_periods=15)
    np.testing.assert_allclose(pol, ref, rtol=1e-12, atol=1e-15, equal_nan=True)
    # By hand at one row: the increments valid at both ends and in nu.
    t, lag = 160, 3
    s = np.arange(t - window + 1 + lag, t + 1)
    d2 = (x[s] - x[s - lag]) ** 2
    pair = nu[s] + nu[s - lag]
    ok = np.isfinite(d2) & np.isfinite(pair)
    assert ref[t, 1] == pytest.approx(d2[ok].mean() - pair[ok].mean(), rel=1e-12)


def test_min_periods_allows_partial_windows() -> None:
    x = np.cumsum(np.random.default_rng(2).standard_normal(60))
    full = trailing_variogram(x, (1, 2), 20)
    partial = trailing_variogram(x, (1, 2), 20, min_periods=10)
    assert np.isnan(full[18, 0]) and np.isfinite(partial[10, 0])
    # lag D needs min_periods - D increments: lag 1 from row 9 (9 increments).
    assert np.isnan(partial[8, 0]) and np.isfinite(partial[9, 0])
    np.testing.assert_allclose(partial[19:], full[19:])
    np.testing.assert_allclose(
        _polars(x, (1, 2), 20, min_periods=10), partial, equal_nan=True
    )


def test_log_slope_weights_are_the_ols_slope() -> None:
    lags = (1, 2, 3, 5, 8, 13)
    w = log_slope_weights(lags)
    assert w.sum() == pytest.approx(0.0, abs=1e-15)
    y = np.random.default_rng(3).standard_normal(len(lags))
    slope = np.polyfit(np.log(lags), y, 1)[0]
    assert float(w @ y) == pytest.approx(slope, rel=1e-12)


def test_fit_power_law_recovers_an_exact_law() -> None:
    lags = tuple(range(1, 11))
    m = 0.36 * np.asarray(lags, dtype=float) ** 0.2
    slope, intercept = fit_power_law(m, lags)
    assert slope == pytest.approx(0.2, abs=1e-13)
    assert np.exp(intercept) == pytest.approx(0.36, rel=1e-13)
    bad = m.copy()
    bad[3] = -1.0
    assert all(np.isnan(v) for v in fit_power_law(bad, lags))


def test_sample_variogram_pools_by_adding() -> None:
    rng = np.random.default_rng(4)
    a, b = np.cumsum(rng.standard_normal(80)), np.cumsum(rng.standard_normal(50))
    lags = (1, 4)
    sa, ca = sample_variogram(a, lags)
    sb, cb = sample_variogram(b, lags)
    assert ca.tolist() == [79.0, 76.0] and cb.tolist() == [49.0, 46.0]
    assert sa[1] / ca[1] == pytest.approx(np.mean((a[4:] - a[:-4]) ** 2))
    whole = trailing_variogram(a, lags, 80)[-1]
    np.testing.assert_allclose(sa / ca, whole, rtol=1e-12)
    np.testing.assert_allclose((sa + sb) / (ca + cb), (sa + sb) / (ca + cb))


@pytest.mark.parametrize(
    ("lags", "window", "match"),
    [
        ((1,), None, "two lags"),
        ((0, 1), None, "positive"),
        ((2, 1), None, "ascending"),
        ((1, 1), None, "distinct"),
        ((1, 9), 10, "window"),
    ],
)
def test_lag_validation(lags: tuple[int, ...], window: int | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_lags(lags, window)


def test_noise_needs_q_two() -> None:
    with pytest.raises(ValueError, match="q == 2"):
        variogram_exprs(pl.col("x"), (1, 2), 10, q=1.0, noise=pl.col("nu"))
    with pytest.raises(ValueError, match="q == 2"):
        trailing_variogram(np.zeros(20), (1, 2), 10, q=1.0, noise=np.zeros(20))
