"""Rough-volatility Monte Carlo (``slow``, seeded): the plan's section 1e table.

A fractional-Brownian log variance (``nu = 0.3``, 1000 days, 200 paths) is
observed through two proxies: log RV from 78 intraday returns a day, and the
daily Parkinson range term from 390-step paths. ``rough_hurst`` over the whole
1000-day window must reproduce the table: the uncorrected Parkinson estimate
reads a smooth ``H = 0.5`` volatility as rough (< 0.35, the trap is real),
and the noise-corrected estimates are within 0.02 of the truth. The RFSV
forecaster must recover ``H`` from its training fold and beat the naive
last-value forecast out of sample.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core import PanelFrame
from panelary.econ.features import RFSVForecaster, rough_hurst

pytestmark = pytest.mark.slow

_T, _R, _NU = 1000, 200, 0.3


def _fgn(n: int, hurst: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Exact fractional Gaussian noise by circulant embedding (Davies-Harte)."""
    k = np.arange(n + 1)
    g = 0.5 * (
        np.abs(k + 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k - 1) ** (2 * hurst)
    )
    c = np.r_[g, g[-2:0:-1]]
    lam = np.maximum(np.fft.fft(c).real, 0.0)
    z = rng.standard_normal((reps, c.size)) + 1j * rng.standard_normal((reps, c.size))
    return np.fft.fft(np.sqrt(lam / c.size) * z, axis=1).real[:, :n]


def _hurst(x: np.ndarray, noise: float | np.ndarray) -> np.ndarray:
    reps, n = x.shape
    df = pl.DataFrame(
        {
            "e": np.repeat(np.arange(reps), n),
            "t": np.tile(np.arange(n), reps),
            "x": x.ravel(),
        }
    )
    noise_var: str | float
    if isinstance(noise, np.ndarray):
        df = df.with_columns(nv=pl.Series(noise.ravel()))
        noise_var = "nv"
    else:
        noise_var = noise
    out = rough_hurst(
        df, entity="e", time="t", log_variance="x", noise_var=noise_var, window=n
    )
    return out.filter(pl.col("t") == n - 1)[f"rough_hurst_{n}"].to_numpy()


@pytest.fixture(scope="module")
def parkinson_bank() -> tuple[np.ndarray, float]:
    """Log Parkinson terms of unit-variance 390-step days, and their variance."""
    rng = np.random.default_rng(1)
    z = np.cumsum(rng.standard_normal((40_000, 390)), axis=1) / np.sqrt(390)
    term = (np.maximum(z.max(1), 0) - np.minimum(z.min(1), 0)) ** 2 / (4 * np.log(2))
    logs = np.log(term)
    # Stored-constant route: the variance comes from paths the test never uses.
    return logs[:20_000], float(np.var(logs[20_000:]))


def test_parkinson_log_noise_variance_matches_the_plan(
    parkinson_bank: tuple[np.ndarray, float],
) -> None:
    _bank, var = parkinson_bank
    assert var == pytest.approx(0.360, abs=0.01)


@pytest.mark.parametrize("hurst", [0.1, 0.3, 0.5])
def test_the_section_1e_table(
    hurst: float, parkinson_bank: tuple[np.ndarray, float]
) -> None:
    rng = np.random.default_rng(int(hurst * 100))
    lv = 2 * _NU * np.cumsum(_fgn(_T, hurst, _R, rng), axis=1)
    oracle = _hurst(lv, 0.0)
    assert np.mean(oracle) == pytest.approx(hurst, abs=0.01)

    m = 78
    r = np.exp(lv[:, :, None] / 2) * rng.standard_normal((_R, _T, m)) / np.sqrt(m)
    rv = (r**2).sum(-1)
    log_rv_var = (2.0 / 3.0) * (r**4).sum(-1) / rv**2  # the `log_rv_var` measure
    raw_rv = _hurst(np.log(rv), 0.0)
    corrected_rv = _hurst(np.log(rv), log_rv_var)
    # Measurably biased towards roughness: -0.010 to -0.024 in the plan's table
    # (standard error of the mean over 200 paths: 0.001-0.002).
    assert np.mean(raw_rv) < hurst - 0.007
    assert np.mean(corrected_rv) == pytest.approx(hurst, abs=0.01)

    bank, pk_var = parkinson_bank
    proxy = lv + rng.choice(bank, size=lv.shape)
    raw_pk = _hurst(proxy, 0.0)
    corrected_pk = _hurst(proxy, pk_var)
    assert np.mean(corrected_pk) == pytest.approx(hurst, abs=0.02)
    if hurst == 0.5:
        assert np.mean(raw_pk) < 0.35  # smooth volatility reads as rough


def test_rfsv_recovers_h_and_beats_the_naive_forecast() -> None:
    rng = np.random.default_rng(21)
    reps, n, m = 40, 3000, 78
    lv = 2 * _NU * np.cumsum(_fgn(n, 0.1, reps, rng), axis=1)
    r = np.exp(lv[:, :, None] / 2) * rng.standard_normal((reps, n, m)) / np.sqrt(m)
    rv = (r**2).sum(-1)
    df = pl.DataFrame(
        {
            "e": np.repeat(np.arange(reps), n),
            "t": np.tile(np.arange(n), reps),
            "x": np.log(rv).ravel(),
            "nv": ((2.0 / 3.0) * (r**4).sum(-1) / rv**2).ravel(),
        }
    )
    panel = PanelFrame(df, entity="e", time="t")
    train = PanelFrame(df.filter(pl.col("t") < 2000), entity="e", time="t")
    model = RFSVForecaster(log_variance="x", noise_var="nv").fit(train)
    assert model.pooled_hurst_ == pytest.approx(0.1, abs=0.01)
    f = (
        model.transform(panel)
        .collect()
        .sort("e", "t")["rfsv_forecast"]
        .to_numpy()
        .reshape(reps, n)
    )
    x = np.log(rv)
    target = lv[:, 2001:]
    # Both forecasts inherit log RV's small negative bias; compare error spreads.
    rfsv_err = f[:, 2000:-1] - target
    naive_err = x[:, 2000:-1] - target
    rfsv_mse = np.var(rfsv_err)
    naive_mse = np.var(naive_err)
    assert rfsv_mse < 0.8 * naive_mse
