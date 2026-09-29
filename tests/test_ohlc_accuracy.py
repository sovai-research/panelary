"""Monte Carlo accuracy of the OHLC estimators (plan 5 §10.4). Slow, seeded.

Each assertion reproduces a measured claim of the plan, with its tolerance at
three or more standard errors of the replication count used. Bars are a
Gaussian random walk of ``M`` intrabar steps (so the high and low carry the
discrete-monitoring bias real bars do), plus an overnight return and a drift
where stated; bar variance is 1.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary._internal._ohlc import discreteness_factor
from panelary.econ.features import range_volatility

pytestmark = pytest.mark.slow

LN2 = math.log(2.0)


def _bars(n, m, rng, *, f=0.0, mu=0.0, chunk=4000):
    """Per-bar log (o, h, l, c) relative to the previous close; variance 1."""
    o = np.empty(n)
    h = np.empty(n)
    lo = np.empty(n)
    c = np.empty(n)
    sd = math.sqrt((1.0 - f) / m)
    for s in range(0, n, chunk):
        k = min(chunk, n - s)
        path = np.cumsum(rng.standard_normal((k, m)) * sd + mu / m, axis=1)
        o[s : s + k] = math.sqrt(f) * rng.standard_normal(k)
        h[s : s + k] = np.maximum(path.max(axis=1), 0.0)
        lo[s : s + k] = np.minimum(path.min(axis=1), 0.0)
        c[s : s + k] = path[:, -1]
    return o, h, lo, c


def _panel(o, h, lo, c, n_bars):
    """Stack per-bar moves into entities of ``n_bars`` consecutive bars."""
    n_ent = len(o) // n_bars
    size = n_ent * n_bars
    ov, hi, low, cl = (x[:size].reshape(n_ent, n_bars) for x in (o, h, lo, c))
    opens = np.cumsum(ov + np.pad(cl[:, :-1], ((0, 0), (1, 0))), axis=1)
    return pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_ent), n_bars),
            "t": np.tile(np.arange(n_bars), n_ent),
            "open": np.exp(opens.ravel() + 3.0),
            "high": np.exp((opens + hi).ravel() + 3.0),
            "low": np.exp((opens + low).ravel() + 3.0),
            "close": np.exp((opens + cl).ravel() + 3.0),
        }
    )


def _terms(o, h, lo, c):
    u, d = h, lo
    return {
        "c2": c * c,
        "parkinson": (h - lo) ** 2 / (4 * LN2),
        "garman_klass": 0.5 * (h - lo) ** 2 - (2 * LN2 - 1) * c * c,
        "rogers_satchell": u * (u - c) + d * (d - c),
    }


def test_efficiency_ordering_gk_rs_parkinson_close() -> None:
    rng = np.random.default_rng(101)
    t = _terms(*_bars(60_000, 2000, rng))
    var = {k: float(np.var(v)) for k, v in t.items()}
    eff = {
        k: var["c2"] / var[k] for k in ("parkinson", "garman_klass", "rogers_satchell")
    }
    # theory 7.4 / ~6 / 5.2; measured in the plan 7.47 / 6.03 / 4.95
    assert eff["garman_klass"] > eff["rogers_satchell"] > eff["parkinson"] > 1.0
    assert eff["garman_klass"] == pytest.approx(7.4, rel=0.08)
    assert eff["parkinson"] == pytest.approx(5.0, rel=0.08)


def test_rogers_satchell_is_drift_free_and_parkinson_is_not() -> None:
    m = 1000
    rng = np.random.default_rng(102)
    t = _terms(*_bars(120_000, m, rng, mu=1.0))
    rs_bias = t["rogers_satchell"].mean() / discreteness_factor("rogers_satchell", m)
    pk_bias = t["parkinson"].mean() / discreteness_factor("parkinson", m)
    # RS at mu = sigma: no drift bias beyond discreteness (se ~0.2%)
    assert abs(rs_bias - 1.0) < 0.01
    # Parkinson at mu = sigma: +36% measured
    assert pk_bias - 1.0 > 0.30


@pytest.mark.parametrize("m", [26, 78, 390])
@pytest.mark.parametrize("method", ["parkinson", "garman_klass", "rogers_satchell"])
def test_discreteness_table_and_corrected_estimates_are_unbiased(m, method) -> None:
    rng = np.random.default_rng(1000 + m)
    n = 100_000
    df = _panel(*_bars(n, m, rng), n)
    raw = range_volatility(
        df, entity="e", time="t", method=method, window=n, output="var"
    )[f"var_{method}_{n}"][-1]
    cor = range_volatility(
        df,
        entity="e",
        time="t",
        method=method,
        window=n,
        output="var",
        discrete_bars=m,
    )[f"var_{method}_{n}"][-1]
    # the raw mean reproduces the stored factor, the corrected one is unbiased
    assert raw == pytest.approx(discreteness_factor(method, m), abs=0.01)
    assert cor == pytest.approx(1.0, abs=0.01)


def _window_estimates(method, n_windows, n, m, rng, *, f=0.0, mu=0.0):
    o, h, lo, c = _bars(n_windows * (n + 1), m, rng, f=f, mu=mu)
    df = _panel(o, h, lo, c, n + 1)
    out = range_volatility(
        df, entity="e", time="t", method=method, window=n, output="var"
    )
    return out.filter(pl.col("t") == n)[f"var_{method}_{n}"].to_numpy()


def test_intraday_only_estimators_miss_the_overnight_variance() -> None:
    # A quarter of the variance overnight. M = 5000 keeps the discreteness
    # bias of YZ's RS component near -1.7% (se of the mean ~0.2%).
    rng = np.random.default_rng(103)
    rs = _window_estimates("rogers_satchell", 3000, 21, 5000, rng, f=0.25)
    yz = _window_estimates("yang_zhang", 3000, 21, 5000, rng, f=0.25)
    assert rs.mean() - 1.0 < -0.18
    assert abs(yz.mean() - 1.0) < 0.03


def test_yang_zhang_and_gkyz_are_about_seven_times_close_to_close() -> None:
    n, m = 21, 2000
    rng = np.random.default_rng(104)
    eff = {}
    for method in ("close_to_close", "yang_zhang", "gk_overnight"):
        est = _window_estimates(method, 3000, n, m, rng)
        eff[method] = float(np.var(est))
    assert eff["close_to_close"] / eff["yang_zhang"] >= 7.0 * 0.9
    assert eff["close_to_close"] / eff["gk_overnight"] >= 7.0 * 0.9


def test_efficiency_ordering_survives_stochastic_variance() -> None:
    """Heston-type variance (Euler): the ordering holds, not the constants."""
    rng = np.random.default_rng(105)
    n_bars, m = 40_000, 500
    v = np.empty(n_bars)
    v[0] = 1.0
    for i in range(1, n_bars):
        v[i] = abs(
            v[i - 1]
            + 0.07 * (1.0 - v[i - 1])
            + 0.15 * math.sqrt(v[i - 1]) * rng.standard_normal()
        )
    o, h, lo, c = _bars(n_bars, m, rng)
    scale = np.sqrt(v)
    t = _terms(o * scale, h * scale, lo * scale, c * scale)
    var = {k: float(np.var(val / v)) for k, val in t.items()}
    assert var["garman_klass"] < var["rogers_satchell"] < var["parkinson"] < var["c2"]


def test_yang_zhang_default_gate_holds() -> None:
    """Plan §7.1: flip to gk_overnight only if its MSE wins by >5% everywhere.

    ``benchmarks/ohlc_vol/range_mc.py`` runs the full 13-regime grid (recorded
    in docs/user-guide/ohlc-volatility-and-liquidity.md). Here one regime where
    it does not -- 5-minute bars, a quarter of the variance overnight, a
    calibrated drift -- pins the decision: the default stays yang_zhang.
    """
    rng = np.random.default_rng(106)
    yz = _window_estimates("yang_zhang", 3000, 21, 78, rng, f=0.25, mu=0.1)
    rng = np.random.default_rng(106)
    gkyz = _window_estimates("gk_overnight", 3000, 21, 78, rng, f=0.25, mu=0.1)
    mse_yz = float(np.mean((yz - 1.0) ** 2))
    mse_gkyz = float(np.mean((gkyz - 1.0) ** 2))
    assert not mse_gkyz < 0.95 * mse_yz


# --------------------------------------------------------------------------- #
# Spreads (plan 5 §1c; benchmarks/ohlc_vol/spread_mc.py has the full table)
# --------------------------------------------------------------------------- #
from panelary.econ.features import ohlc_spread  # noqa: E402

_SPREAD_REGIMES = {
    "mid": {"spread": 0.005, "sigma": 0.02, "prob": 1.0},
    "thin": {"spread": 0.005, "sigma": 0.02, "prob": 0.1},
    "illiquid": {"spread": 0.02, "sigma": 0.03, "prob": 0.02},
    "liquid": {"spread": 0.001, "sigma": 0.015, "prob": 1.0},
}


def _spread_bars(n_ent, n_bars, *, spread, sigma, prob, rng, k=390, overnight=0.2):
    """Trades at the efficient log price +- S/2 (random side); bars from trades.

    ``k`` trade opportunities a day, each taken with probability ``prob``; 20% of
    the daily variance overnight; a day without a trade is a flat bar at the
    previous close.
    """
    days = n_ent * n_bars
    sd_in = sigma * math.sqrt((1.0 - overnight) / k)
    rel = np.full((days, 4), np.nan)
    move = np.empty(days)
    chunk = max(1, 4_000_000 // k)
    for s in range(0, days, chunk):
        d = min(chunk, days - s)
        path = np.cumsum(rng.standard_normal((d, k)) * sd_in, axis=1)
        hit = rng.random((d, k)) < prob
        side = rng.integers(0, 2, (d, k)) * 2.0 - 1.0
        trades = np.where(hit, path + side * spread / 2.0, np.nan)
        rows = np.arange(d)
        any_hit = hit.any(axis=1)
        filled = np.where(any_hit[:, None], np.where(hit, trades, -np.inf), 0.0)
        rel[s : s + d, 1] = np.where(any_hit, filled.max(axis=1), np.nan)
        filled = np.where(any_hit[:, None], np.where(hit, trades, np.inf), 0.0)
        rel[s : s + d, 2] = np.where(any_hit, filled.min(axis=1), np.nan)
        rel[s : s + d, 0] = trades[rows, np.argmax(hit, axis=1)]
        rel[s : s + d, 3] = trades[rows, k - 1 - np.argmax(hit[:, ::-1], axis=1)]
        move[s : s + d] = path[:, -1]
    inc = rng.standard_normal((n_ent, n_bars)) * sigma * math.sqrt(overnight)
    inc[:, 1:] += move.reshape(n_ent, n_bars)[:, :-1]
    start = np.cumsum(inc, axis=1).ravel()
    names = ("open", "high", "low", "close")
    frame = pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_ent), n_bars),
            "t": np.tile(np.arange(n_bars), n_ent),
            "start": start,
            **{n: rel[:, j] + start for j, n in enumerate(names)},
        }
    ).with_columns(pl.col(n).fill_nan(None) for n in names)
    stale = pl.coalesce(
        pl.col("close").forward_fill().shift(1).over("e"), pl.col("start")
    )
    return frame.with_columns(stale.alias("stale")).select(
        "e", "t", *((pl.coalesce(n, "stale").exp() * 50.0).alias(n) for n in names)
    )


def _spread_errors(regime, n, windows, seed):
    p = _SPREAD_REGIMES[regime]
    bars = _spread_bars(windows, n, rng=np.random.default_rng(seed), **p)
    out = {}
    for method in ("edge", "corwin_schultz", "abdi_ranaldo"):
        res = ohlc_spread(bars, entity="e", time="t", method=method, window=n)
        last = res.filter(pl.col("t") == n - 1)
        est = last[f"spread_{method}_{n}"].to_numpy().astype(float)
        out[method] = math.sqrt(np.mean((est - p["spread"]) ** 2)) / p["spread"]
        if method == "edge":
            m = float(np.mean(last[f"spread_edge_moment_{n}"].to_numpy()))
            out["edge_pooled"] = math.copysign(math.sqrt(abs(m)), m) / p["spread"]
    return out


@pytest.fixture(scope="module")
def spread_mc():
    """RMSE / S (and EDGE's pooled root moment) per regime and window."""
    windows = {21: 1500, 63: 800, 252: 300}
    return {
        (regime, n): _spread_errors(regime, n, windows[n], seed=7000 + i * 10 + j)
        for i, regime in enumerate(_SPREAD_REGIMES)
        for j, n in enumerate((21, 63, 252))
    }


@pytest.mark.parametrize("regime", ["mid", "illiquid"])
def test_edge_pooled_moment_is_unbiased(spread_mc, regime) -> None:
    for n in (21, 63, 252):
        assert spread_mc[(regime, n)]["edge_pooled"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("regime", ["mid", "thin", "illiquid", "liquid"])
def test_edge_error_falls_with_the_window(spread_mc, regime) -> None:
    rmse = [spread_mc[(regime, n)]["edge"] for n in (21, 63, 252)]
    assert rmse[0] > rmse[1] > rmse[2]


@pytest.mark.parametrize("regime", ["mid", "illiquid", "liquid"])
def test_edge_beats_corwin_schultz_on_longer_windows(spread_mc, regime) -> None:
    """CS's zero-clipping is a bias floor that more data cannot remove."""
    for n in (63, 252):
        assert spread_mc[(regime, n)]["edge"] < spread_mc[(regime, n)]["corwin_schultz"]
    cs = [spread_mc[(regime, n)]["corwin_schultz"] for n in (21, 63, 252)]
    assert cs[2] > 0.8 * cs[1]  # CS barely improves from 63 to 252 bars


def test_thin_trading_is_where_corwin_schultz_holds_out_longest(spread_mc) -> None:
    """Measured, and narrower than the plan's "EDGE wins at n >= 63 everywhere".

    With 10% of trade opportunities taken, CS beats EDGE at 21 bars (0.56 vs
    0.72 RMSE / S) and is level with it at 63 (0.46 vs 0.48); EDGE only pulls
    ahead by 252 bars, where CS's clipping floor binds.
    """
    thin = {n: spread_mc[("thin", n)] for n in (21, 63, 252)}
    assert thin[21]["corwin_schultz"] < thin[21]["edge"]
    assert thin[252]["edge"] < thin[252]["corwin_schultz"]


def test_no_ohlc_estimator_resolves_a_small_spread(spread_mc) -> None:
    """The honest caveat: S = 0.1% at 1.5% daily volatility is below resolution.

    This test exists to prove the caveat is real. Never "fix" it by loosening.
    """
    for n in (21, 63):
        for method in ("edge", "corwin_schultz", "abdi_ranaldo"):
            assert spread_mc[("liquid", n)][method] > 1.0
