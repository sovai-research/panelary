"""Monte Carlo accuracy of the OHLC spread estimators (plan 5 §1c).

A simulated efficient log price with ``K = 390`` trade opportunities a day and
20% of the daily variance overnight; at each opportunity a trade happens with
probability ``prob``, at the efficient price plus or minus half the spread
``S`` (random side). The day's bar is the first, highest, lowest and last
*trade*; a day with no trade is a flat bar at the previous close (a stale
vendor fill). Each regime draws ``windows`` independent entities of ``n`` bars
and scores the last row's ``n``-bar estimate of every estimator, through
:func:`panelary.econ.features.ohlc_spread` (``negative="literature"``), plus
the pooled root-mean signed square of EDGE's moment.

Run::

    python benchmarks/ohlc_vol/spread_mc.py                 # all regimes, ~2 min
    python benchmarks/ohlc_vol/spread_mc.py --windows 300
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
import polars as pl

from panelary.econ.features import ohlc_spread

REGIMES: dict[str, dict[str, float]] = {
    "mid": {"spread": 0.005, "sigma": 0.02, "prob": 1.0},
    "thin": {"spread": 0.005, "sigma": 0.02, "prob": 0.1},
    "illiquid": {"spread": 0.02, "sigma": 0.03, "prob": 0.02},
    "liquid": {"spread": 0.001, "sigma": 0.015, "prob": 1.0},
}
METHODS = ("edge", "corwin_schultz", "abdi_ranaldo")


def simulate_bars(
    n_entities: int,
    n_bars: int,
    *,
    spread: float,
    sigma: float,
    prob: float,
    rng: np.random.Generator,
    k: int = 390,
    overnight: float = 0.2,
) -> pl.DataFrame:
    """OHLC bars of ``n_entities`` independent entities (see the module doc)."""
    days = n_entities * n_bars
    sd_in = sigma * math.sqrt((1.0 - overnight) / k)
    sd_on = sigma * math.sqrt(overnight)
    rel = np.full((days, 4), np.nan)  # o, h, l, c relative to the day's start
    move = np.empty(days)  # efficient intraday move of each day
    chunk = max(1, 4_000_000 // k)
    for s in range(0, days, chunk):
        d = min(chunk, days - s)
        path = np.cumsum(rng.standard_normal((d, k)) * sd_in, axis=1)
        side = rng.integers(0, 2, (d, k)) * 2.0 - 1.0
        hit = rng.random((d, k)) < prob
        trades = np.where(hit, path + side * spread / 2.0, np.nan)
        rows = np.arange(d)
        first = np.argmax(hit, axis=1)
        last = k - 1 - np.argmax(hit[:, ::-1], axis=1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rel[s : s + d, 1] = np.nanmax(trades, axis=1)
            rel[s : s + d, 2] = np.nanmin(trades, axis=1)
        rel[s : s + d, 0] = trades[rows, first]
        rel[s : s + d, 3] = trades[rows, last]
        move[s : s + d] = path[:, -1]
    # day-start efficient level: the previous efficient close plus overnight
    inc = rng.standard_normal((n_entities, n_bars)) * sd_on
    inc[:, 1:] += move.reshape(n_entities, n_bars)[:, :-1]
    start = np.cumsum(inc, axis=1).ravel()
    frame = pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_entities), n_bars),
            "t": np.tile(np.arange(n_bars), n_entities),
            "start": start,
            **{name: rel[:, j] + start for j, name in enumerate("ohlc")},
        }
    ).with_columns(pl.col(x).fill_nan(None) for x in "ohlc")
    # A day without a trade is a flat bar at the previous observed close.
    stale = pl.coalesce(pl.col("c").forward_fill().shift(1).over("e"), pl.col("start"))
    frame = frame.with_columns(stale.alias("stale")).with_columns(
        pl.coalesce(pl.col(x), pl.col("stale")).alias(x) for x in "ohlc"
    )
    return frame.select(
        "e",
        "t",
        *(
            (pl.col(x).exp() * 50.0).alias(name)
            for x, name in zip("ohlc", ("open", "high", "low", "close"), strict=True)
        ),
    )


def run_regime(
    regime: str, n: int, windows: int, seed: int
) -> dict[str, tuple[float, float]]:
    """``{method: (bias / S, RMSE / S)}``, plus ``"edge_pooled"`` root-mean s^2."""
    params = REGIMES[regime]
    rng = np.random.default_rng(seed)
    bars = simulate_bars(windows, n, rng=rng, **params)
    s_true = params["spread"]
    out: dict[str, tuple[float, float]] = {}
    for method in METHODS:
        res = ohlc_spread(bars, entity="e", time="t", method=method, window=n)
        last = res.filter(pl.col("t") == n - 1)
        est = last[f"spread_{method}_{n}"].to_numpy().astype(float)
        err = est - s_true
        out[method] = (
            float(np.nanmean(err) / s_true),
            float(math.sqrt(np.nanmean(err**2)) / s_true),
        )
        if method == "edge":
            m = last[f"spread_edge_moment_{n}"].to_numpy().astype(float)
            pooled = math.copysign(math.sqrt(abs(np.nanmean(m))), np.nanmean(m))
            out["edge_pooled"] = (pooled / s_true - 1.0, float("nan"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--windows", type=int, default=1500)
    args = parser.parse_args()
    t0 = time.perf_counter()
    print(
        "bias / RMSE, both divided by the true spread S; EDGE pooled = sqrt(mean s^2)/S - 1"
    )
    for regime in REGIMES:
        for n in (21, 63, 252):
            windows = args.windows if n < 252 else max(200, args.windows // 3)
            res = run_regime(
                regime, n, windows, seed=1000 * (list(REGIMES).index(regime) + 1) + n
            )
            cells = "  ".join(
                f"{m}: {res[m][0]:+6.1%} / {res[m][1]:5.2f}" for m in METHODS
            )
            print(
                f"{regime:>8} n={n:>3} | {cells} | edge pooled {res['edge_pooled'][0]:+6.1%}"
                f"  [{time.perf_counter() - t0:.0f}s]",
                flush=True,
            )


if __name__ == "__main__":
    main()
