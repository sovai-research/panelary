"""Wall-clock budgets of the daily-bar OHLC estimators (plan 5 §11).

Builds a synthetic ``entities x bars`` panel of positive OHLC bars (default
5000 x 5000 = 25M rows, the plan's reference size) and times each estimator
through its public function, best of ``--repeat`` runs after one warm-up. Peak
RSS is reported for the process as a whole (``ru_maxrss``), so run one
``--only`` group at a time when the memory figure matters.

Budgets (plan §11, reference machine: Apple M5 Pro, 15 threads; fail at 2x):

=============================================  ========
P + GK + RS, w=21                              <= 2.5 s
Yang-Zhang, w=21                               <= 3 s
EDGE, w=21, batched 256                        <= 12 s, <= 6 GB
Corwin-Schultz / Abdi-Ranaldo                  <= 3 s (target)
price_impact / PS gamma / FHT                  <= 3 / 5 / 3 s (target)
=============================================  ========

Run::

    python benchmarks/ohlc_vol/bench_daily.py                  # everything
    python benchmarks/ohlc_vol/bench_daily.py --only range     # one group
    python benchmarks/ohlc_vol/bench_daily.py --entities 500 --bars 2000
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from collections.abc import Callable

import numpy as np
import polars as pl

from panelary.econ import features as F


def make_bars(n_entities: int, n_bars: int, seed: int = 7) -> pl.DataFrame:
    """Random-walk OHLC bars with overnight gaps, zero-range bars and volume."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_bars
    intraday = rng.standard_normal(n) * 0.015
    overnight = rng.standard_normal(n) * 0.006
    lc = np.cumsum((intraday + overnight).reshape(n_entities, n_bars), axis=1).ravel()
    lo_ = lc - intraday
    up = np.abs(rng.standard_normal(n)) * 0.008
    down = np.abs(rng.standard_normal(n)) * 0.008
    lh = np.maximum(lo_, lc) + up
    ll = np.minimum(lo_, lc) - down
    ret = np.diff(lc, prepend=0.0)
    ret[::n_bars] = np.nan
    ret = np.where(rng.random(n) < 0.03, 0.0, ret)  # stale days
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_entities, dtype=np.int32), n_bars),
            "t": np.tile(np.arange(n_bars, dtype=np.int32), n_entities),
            "open": np.exp(lo_ + 3.0),
            "high": np.exp(lh + 3.0),
            "low": np.exp(ll + 3.0),
            "close": np.exp(lc + 3.0),
            "ret": ret,
            "mkt": rng.standard_normal(n) * 0.01,
            "dv": np.exp(rng.standard_normal(n) + 15.0),
        }
    ).with_columns(pl.col("ret").fill_nan(None))


def _range(df: pl.DataFrame, methods: tuple[str, ...]) -> Callable[[], object]:
    def run() -> object:
        out = df
        for m in methods:
            out = F.range_volatility(out, entity="id", time="t", method=m, window=21)
        return out

    return run


def workloads(df: pl.DataFrame) -> dict[str, dict[str, Callable[[], object]]]:
    groups: dict[str, dict[str, Callable[[], object]]] = {
        "range": {
            "P + GK + RS, w=21": _range(
                df, ("parkinson", "garman_klass", "rogers_satchell")
            ),
            "Yang-Zhang, w=21": _range(df, ("yang_zhang",)),
            "close-to-close, w=21": _range(df, ("close_to_close",)),
        },
    }
    if hasattr(F, "ohlc_spread"):
        groups["spread"] = {
            "EDGE, w=21, batched 256": lambda: F.ohlc_spread(
                df, entity="id", time="t", method="edge", window=21
            ),
            "EDGE, w=21, unbatched": lambda: F.ohlc_spread(
                df, entity="id", time="t", method="edge", window=21, batch_entities=None
            ),
            "Corwin-Schultz, w=21": lambda: F.ohlc_spread(
                df, entity="id", time="t", method="corwin_schultz", window=21
            ),
            "Abdi-Ranaldo, w=21": lambda: F.ohlc_spread(
                df, entity="id", time="t", method="abdi_ranaldo", window=21
            ),
        }
    if hasattr(F, "price_impact"):
        groups["proxies"] = {
            "price_impact, w=63": lambda: F.price_impact(
                df, entity="id", time="t", returns="ret", dollar_volume="dv"
            ),
            "pastor_stambaugh_gamma, w=21": lambda: F.pastor_stambaugh_gamma(
                df,
                entity="id",
                time="t",
                returns="ret",
                market_returns="mkt",
                dollar_volume="dv",
            ),
            "zero_return_share, w=21": lambda: F.zero_return_share(
                df, entity="id", time="t", returns="ret", volume="dv"
            ),
            "fht_spread, w=21": lambda: F.fht_spread(
                df, entity="id", time="t", returns="ret"
            ),
        }
    return groups


def _rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--entities", type=int, default=5000)
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = make_bars(args.entities, args.bars)
    print(
        f"panel: {args.entities} x {args.bars} = {df.height:,} rows "
        f"(built in {time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    for group, runs in workloads(df).items():
        if args.only and group != args.only:
            continue
        for label, fn in runs.items():
            fn()  # warm-up
            best = float("inf")
            for _ in range(args.repeat):
                start = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - start)
            print(
                f"{group:>8} | {label:<32} {best:7.2f} s   peak RSS {_rss_gb():5.1f} GB",
                flush=True,
            )


if __name__ == "__main__":
    main()
