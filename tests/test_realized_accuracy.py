"""Monte Carlo accuracy of the realized measures (``slow``, seeded).

Every assertion reproduces a property the literature states and the plan
(``plans/todo/ohlc-volatility-and-liquidity.md`` section 10.4) requires, with a
tolerance of at least three standard errors at the replication count used:

* no noise, no jumps: every variance measure is unbiased for the integrated
  variance, the quarticities for the integrated quarticity, and the asymptotic
  variances ``M Var / IQ`` are the published 2 (RV), 2.61 (BV), 2.96 (MedRV)
  and 3.81 (MinRV);
* with jumps: RV absorbs them, the jump-robust measures do not;
* the jump test has its nominal size;
* under stochastic volatility the efficiency ordering survives.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary._internal._special import norm_ppf
from panelary.econ.features import intraday_realized_measures

pytestmark = pytest.mark.slow

_M = 390
_IV = 1e-4  # daily integrated variance under constant volatility


def _measures(returns: np.ndarray, **kwargs: object) -> pl.DataFrame:
    """Run the daily battery on a ``(sessions, M)`` array of returns."""
    s, m = returns.shape
    df = pl.DataFrame(
        {
            "e": np.zeros(s * m, dtype=np.int8),
            "s": np.repeat(np.arange(s), m),
            "t": np.tile(np.arange(m), s),
            "r": returns.ravel(),
        }
    )
    return intraday_realized_measures(
        df,
        entity="e",
        session="s",
        time="t",
        returns="r",
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def constant_vol() -> tuple[np.ndarray, pl.DataFrame]:
    rng = np.random.default_rng(7)
    r = rng.standard_normal((20_000, _M)) * np.sqrt(_IV / _M)
    out = _measures(
        r,
        measures=[
            "rv",
            "rv_ss",
            "bv",
            "medrv",
            "minrv",
            "rs_pos",
            "rs_neg",
            "rq",
            "tpq",
            "medrq",
            "jump_z",
            "log_rv_var",
        ],
    )
    return r, out


def test_variance_measures_are_unbiased_without_noise_or_jumps(
    constant_vol: tuple[np.ndarray, pl.DataFrame],
) -> None:
    _r, out = constant_vol
    for col in ("rv", "bv", "medrv", "minrv"):  # se of the mean ~ 0.07%
        assert out[col].mean() / _IV == pytest.approx(1.0, abs=3e-3), col
    for col in ("rs_pos", "rs_neg"):
        assert out[col].mean() / _IV == pytest.approx(0.5, abs=2e-3), col
    # rv_ss covers M - K + 1 complete K-step returns: expectation (M - K + 1)/M.
    assert out["rv_ss"].mean() / _IV == pytest.approx((_M - 4) / _M, abs=3e-3)
    iq = _IV**2
    for col in ("rq", "tpq", "medrq"):  # se ~ 0.13%
        assert out[col].mean() / iq == pytest.approx(1.0, abs=6e-3), col


def test_asymptotic_variances_match_the_published_constants(
    constant_vol: tuple[np.ndarray, pl.DataFrame],
) -> None:
    """``M Var(X) / IQ``: RV 2, BV pi^2/4 + pi - 3 = 2.61, MedRV 2.96, MinRV 3.81
    (Barndorff-Nielsen & Shephard 2006; ADS 2012). The relative standard error
    of a variance from 20 000 draws is 1%, so the tolerance is 5%."""
    _r, out = constant_vol
    published = {
        "rv": 2.0,
        "bv": np.pi**2 / 4 + np.pi - 3,
        "medrv": 2.96,
        "minrv": 3.81,
    }
    measured = {c: _M * float(np.var(out[c].to_numpy())) / _IV**2 for c in published}
    for col, value in published.items():
        assert measured[col] == pytest.approx(value, rel=0.05), (col, measured[col])
    assert measured["rv"] < measured["bv"] < measured["medrv"] < measured["minrv"]


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_the_jump_test_has_its_nominal_size(
    constant_vol: tuple[np.ndarray, pl.DataFrame], alpha: float
) -> None:
    _r, out = constant_vol
    rate = float((out["jump_z"].to_numpy() > float(norm_ppf(alpha))).mean())
    se = np.sqrt(alpha * (1 - alpha) / out.height)
    assert abs(rate - (1 - alpha)) < 4 * se, rate


def test_log_rv_var_is_the_sampling_variance_of_log_rv(
    constant_vol: tuple[np.ndarray, pl.DataFrame],
) -> None:
    """The delta-method noise variance that ``rough_hurst`` subtracts."""
    _r, out = constant_vol
    empirical = float(np.var(np.log(out["rv"].to_numpy())))
    assert out["log_rv_var"].mean() == pytest.approx(empirical, rel=0.03)
    assert out["log_rv_var"].mean() == pytest.approx(2 / _M, rel=0.03)


def test_jump_robust_measures_ignore_a_planted_jump(
    constant_vol: tuple[np.ndarray, pl.DataFrame],
) -> None:
    r, _out = constant_vol
    rng = np.random.default_rng(8)
    s = r.shape[0]
    jumpy = r.copy()
    where = rng.integers(0, _M, s)
    # A jump of half a daily standard deviation adds 0.25 IV to RV.
    jumpy[np.arange(s), where] += rng.choice([-1.0, 1.0], s) * 0.5 * np.sqrt(_IV)
    out = _measures(
        jumpy, measures=["rv", "bv", "medrv", "minrv", "jump_z", "jump_sig", "cont"]
    )
    assert out["rv"].mean() / _IV == pytest.approx(1.25, abs=5e-3)
    assert out["bv"].mean() / _IV - 1 < 0.08
    assert abs(out["medrv"].mean() / _IV - 1) < 0.02
    assert abs(out["minrv"].mean() / _IV - 1) < 0.02
    assert float((out["jump_z"].to_numpy() > float(norm_ppf(0.999))).mean()) > 0.5
    # The C/J split moves most of the jump out of the continuous part.
    assert out["cont"].mean() / _IV - 1 < 0.12


def test_efficiency_ordering_survives_stochastic_volatility() -> None:
    """Heston-type variance within the day (Euler, seeded): MSE ordering holds."""
    rng = np.random.default_rng(9)
    s = 6_000
    dt = 1.0 / _M
    kappa, theta, xi = 5.0, _IV, 0.5 * np.sqrt(_IV)
    v = np.full(s, theta)
    r = np.empty((s, _M))
    iv = np.zeros(s)
    for i in range(_M):
        z1, z2 = rng.standard_normal(s), rng.standard_normal(s)
        r[:, i] = np.sqrt(v * dt) * z1
        iv += v * dt
        v = np.abs(
            v
            + kappa * (theta - v) * dt
            + xi * np.sqrt(v * dt) * (-0.7 * z1 + 0.71 * z2)
        )
    out = _measures(r, measures=["rv", "bv", "medrv", "minrv"])
    mse = {
        c: float(np.mean((out[c].to_numpy() - iv) ** 2))
        for c in ("rv", "bv", "medrv", "minrv")
    }
    assert mse["rv"] < mse["bv"] < mse["medrv"] < mse["minrv"], mse
    for col in mse:
        assert out[col].mean() / iv.mean() == pytest.approx(1.0, abs=0.01), col
