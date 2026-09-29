"""Wall-clock budgets for the intraday realized measures and ``rough_hurst``.

Plan ``plans/todo/ohlc-volatility-and-liquidity.md`` section 11:

* intraday core battery, 98M rows: budget 8 s;
* intraday full battery incl. RK (``H_max`` 30), PAV and TSRV, 98M rows:
  budget 40 s;
* ``rough_hurst``, 10 lags, 25M daily rows: target 4 s.

The intraday panel is ``entities x days x 390`` one-minute returns; the
default ``100 x 504`` (19.7M rows) keeps the run to about a minute, and the
budgets scale linearly (98M = 500 entities). Usage::

    python benchmarks/ohlc_vol/bench_intraday.py [entities] [days] [daily_entities]

Measured 2026-09-29 (Apple M5 Pro, 15 threads, polars 1.44.2, with eight other
jobs sharing the machine, load average 30-45), best of two:

=========================================  ========  ===================
workload                                   19.7M     98M (x5, linear)
=========================================  ========  ===================
core battery (12 measures)                 0.96 s    ~4.8 s (budget 8)
full battery (+ rk, tsrv, pav, medrq, C/J)  5.6 s    ~28 s (budget 40)
=========================================  ========  ===================

``rough_hurst`` (10 lags, W = 500): 5M rows 0.79 s (constant noise) / 1.14 s
(noise column); 25M rows 4.0 s / 7.0 s, peak RSS 8.4 / 10.6 GB.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from panelary.econ.features import intraday_realized_measures, rough_hurst
from panelary.econ.features._realized import _DEFAULT_MEASURES


def _best(fn, reps: int = 2) -> float:  # type: ignore[no-untyped-def]
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(entities: int = 100, days: int = 504, daily_entities: int = 1000) -> None:
    m = 390
    rng = np.random.default_rng(3)
    n = entities * days * m
    intraday = pl.DataFrame(
        {
            "id": np.repeat(np.arange(entities, dtype=np.int32), days * m),
            "day": np.tile(np.repeat(np.arange(days, dtype=np.int32), m), entities),
            "t": np.tile(np.arange(m, dtype=np.int32), entities * days),
            "r": rng.standard_normal(n) * 0.01 / np.sqrt(m),
        }
    )
    kw = {"entity": "id", "session": "day", "time": "t", "returns": "r"}
    full = [*_DEFAULT_MEASURES, "rk", "tsrv", "pav", "medrq", "jump_sig", "cont"]
    print(f"intraday: {n / 1e6:.1f}M rows ({entities} x {days} x {m})")
    for label, measures in (
        ("core battery", _DEFAULT_MEASURES),
        ("full battery", full),
    ):
        secs = _best(
            lambda ms=measures, frame=intraday: intraday_realized_measures(
                frame, measures=ms, **kw
            )
        )
        print(f"  {label:14s} {secs:6.2f} s  -> {secs * 98e6 / n:6.1f} s at 98M rows")
    del intraday

    t_len = 5000
    daily = pl.DataFrame(
        {
            "e": np.repeat(np.arange(daily_entities, dtype=np.int32), t_len),
            "t": np.tile(np.arange(t_len, dtype=np.int32), daily_entities),
            "x": np.cumsum(rng.standard_normal(daily_entities * t_len)) * 0.01,
            "nv": np.full(daily_entities * t_len, 0.02),
        }
    )
    rows = daily.height
    print(f"rough_hurst: {rows / 1e6:.1f}M rows ({daily_entities} x {t_len})")
    for label, noise in (("constant noise", 0.02), ("noise column", "nv")):
        secs = _best(
            lambda nv=noise: rough_hurst(
                daily, entity="e", time="t", log_variance="x", noise_var=nv
            )
        )
        print(
            f"  {label:14s} {secs:6.2f} s  -> {secs * 25e6 / rows:6.1f} s at 25M rows"
        )


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:4]))
