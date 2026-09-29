"""Benchmarks for ``panelary.covariance`` (plan section 8).

Times the section 8.1 budget workloads on a synthetic factor panel and prints
one row per workload with the measured time, the budget and the ratio. A
2,000 x 2,000 GEMM is timed first and printed with the results, so numbers
from different machines (Accelerate vs OpenBLAS) can be compared.

Thread count matters for BLAS-bound code: set it explicitly and record it::

    OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python benchmarks/bench_covariance.py
    python benchmarks/bench_covariance.py --n 500 --t 1000        # a quick run
    python benchmarks/bench_covariance.py --only estimate,turbulence

Only numpy and polars are required; the scikit-learn baseline row is skipped
when scikit-learn is absent.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import tracemalloc
from collections.abc import Callable

import numpy as np
import polars as pl

import panelary.covariance as cov
from panelary.covariance._state import DEFAULT_FEATURES
from panelary.covariance._window import panel_matrix, window_at

SPECTRUM_SET = tuple(f for f in DEFAULT_FEATURES if f != "ar_shift")


def make_panel(n: int, t: int, *, seed: int = 0) -> pl.DataFrame:
    """Long panel: one market factor, eight sectors, t(5) noise, no gaps."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n)
    sector = np.arange(n) % 8
    F = rng.standard_normal((t, 9)) * np.array([1.0] + [0.5] * 8)
    L = np.zeros((n, 9))
    L[:, 0] = beta
    L[np.arange(n), 1 + sector] = rng.uniform(0.3, 0.8, n)
    R = F @ L.T + rng.standard_t(5, (t, n)) / math.sqrt(5 / 3) * rng.uniform(0.5, 2, n)
    return pl.DataFrame(
        {
            "entity": np.repeat(np.arange(n, dtype=np.int32), t),
            "time": np.tile(np.arange(t, dtype=np.int32), n),
            "ret": R.T.reshape(-1),
        }
    )


def timed(fn: Callable[[], object], repeat: int = 1) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def gemm_calibration() -> float:
    A = np.random.default_rng(1).standard_normal((2000, 2000))
    A @ A
    return timed(lambda: A @ A, repeat=3)


def run(
    n: int, t: int, window: int, only: set[str] | None
) -> list[tuple[str, float, float | None, str]]:
    rows: list[tuple[str, float, float | None, str]] = []

    def want(name: str) -> bool:
        return only is None or name in only

    df = make_panel(n, t)
    if want("build_tensor"):
        sec = timed(lambda: panel_matrix(df, "ret", entity="entity", time="time"))
        rows.append(
            (f"panel_matrix / build_tensor, {n * t / 1e6:.1f}M cells", sec, 2.0, "Q11")
        )
    if want("estimate"):
        X = df.filter(pl.col("time") < window).sort(["time", "entity"])
        X = X.get_column("ret").to_numpy().reshape(window, n)
        cov.estimate(X, method="qis")
        sec = timed(lambda: cov.estimate(X, method="qis"), repeat=7)
        rows.append(
            (f"estimate(qis, correlation), N={n}, W={window}", sec, 0.008, "per call")
        )
    if want("breakdown"):
        pm = panel_matrix(df, "ret", entity="entity", time="time")
        dates = range(window, min(t, window + 50))
        t0 = time.perf_counter()
        wss = [window_at(pm, s, window=window)[0] for s in dates]
        t_win = (time.perf_counter() - t0) / len(dates)
        t0 = time.perf_counter()
        for ws in wss:
            ws.gram()
        t_gram = (time.perf_counter() - t0) / len(dates)
        t0 = time.perf_counter()
        for ws in wss:
            ws.spectrum()
        t_eig = (time.perf_counter() - t0) / len(dates)
        rows.append(
            ("  universe + window copy + standardise, per date", t_win, None, "")
        )
        rows.append(("  min-side Gram, per date", t_gram, None, ""))
        rows.append(("  eigvalsh, per date", t_eig, None, ""))
    if want("market_state"):
        budget = 12.0 * (t / 5000) if n <= 500 else 20.0 * (t / 5000)
        sec = timed(
            lambda: cov.market_state(
                df,
                returns="ret",
                window=window,
                features=SPECTRUM_SET,
                entity="entity",
                time="time",
            )  # fmt: skip
        )
        rows.append(
            (f"market_state spectrum set, daily, N={n}, T={t}", sec, budget, "8.1")
        )
        daily = sec
        # stride 5 on the integer axis; a "1w" grid (the same density on a
        # business-day Date axis) costs the same per evaluated date
        sec = timed(
            lambda: cov.market_state(
                df,
                returns="ret",
                window=window,
                features=SPECTRUM_SET,
                stride=5,
                entity="entity",
                time="time",
            )
        )
        rows.append(("  same, stride 5", sec, 0.3 * daily, "<= 30% of daily"))
    if want("turbulence") and hasattr(cov, "turbulence"):
        budget = 5.0 * (t / 5000)
        sec = timed(
            lambda: cov.turbulence(
                df, returns="ret", window=window, refit=21, entity="entity", time="time"
            )
        )
        rows.append(
            (
                f"turbulence, refit every 21, daily scoring, N={n}, T={t}",
                sec,
                budget,
                "8.1",
            )
        )
    if want("memory"):
        tracemalloc.start()
        cov.market_state(
            df, returns="ret", window=window, features=SPECTRUM_SET,
            stride=21, entity="entity", time="time",
        )  # fmt: skip
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            (f"peak numpy memory (tracemalloc), N={n}, T={t}", peak / 2**30, 1.0, "GiB")
        )
    if want("sklearn"):
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError:
            rows.append(("sklearn LedoitWolf per date", math.nan, None, "skipped"))
        else:
            X = df.filter(pl.col("time") < window).sort(["time", "entity"])
            X = X.get_column("ret").to_numpy().reshape(window, n)
            sk = timed(lambda: LedoitWolf().fit(X), repeat=2)
            ours = timed(
                lambda: cov.estimate(X, method="lw_identity", space="covariance"),
                repeat=5,
            )
            rows.append((f"sklearn LedoitWolf().fit, N={n}", sk, None, "baseline"))
            rows.append(
                (
                    f"  estimate(lw_identity), N={n}; speed-up x{sk / ours:,.0f}",
                    ours,
                    None,
                    "",
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, nargs="+", default=[500, 3000])
    ap.add_argument("--t", type=int, default=5000)
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    threads = os.environ.get("VECLIB_MAXIMUM_THREADS") or os.environ.get(
        "OMP_NUM_THREADS"
    )
    print(f"BLAS threads: {threads or 'default'}")
    print(f"calibration: 2000x2000 GEMM {gemm_calibration() * 1e3:.1f} ms")
    for n in args.n:
        print(f"\nN = {n}, T = {args.t}, W = {args.window}")
        for name, sec, budget, note in run(n, args.t, args.window, only):
            unit = "GiB" if note == "GiB" else "s"
            val = f"{sec:10.3f} {unit}" if unit == "GiB" else f"{sec * 1e3:10.1f} ms"
            b = (
                ""
                if budget is None
                else f"  budget {budget:.3g} {unit}  ({sec / budget:.2f}x)"
            )
            print(f"  {name:<62}{val}{b}  {note if note != 'GiB' else ''}")


if __name__ == "__main__":
    main()
