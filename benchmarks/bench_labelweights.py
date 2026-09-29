"""Benchmarks for label spans, sample weights, the purge and event sampling.

Times the operations behind ``panelary.weights`` / ``panelary.sample`` and the
span-table purge on seeded synthetic data, and prints one table. numba compile
time is measured separately (first call) and excluded from the throughput
rows; with numba missing, the numba rows read ``skipped``.

Run::

    python benchmarks/bench_labelweights.py            # default sizes
    python benchmarks/bench_labelweights.py --big      # 25M-row weight kernels

Only numpy and polars are required (numba is the optional ``fast`` extra).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import polars as pl

import panelary as pn
from panelary._internal._jit import force_numpy, numba_available
from panelary.core._spans import _concurrency, _span_sums, _spans_from_t1
from panelary.core.model_selection import (
    _contiguous_blocks,
    _purge_embargo_positions_t1,
)
from panelary.sample._kernels import _cusum_filter, _seq_boot

ROWS: list[tuple[str, str, str]] = []


def best_of(fn: Callable[[], object], repeat: int = 3) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def once(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def record(op: str, size: str, seconds: float | None, note: str = "") -> None:
    val = "skipped" if seconds is None else f"{seconds * 1e3:,.1f} ms"
    ROWS.append((op, size, val + (f"  ({note})" if note else "")))
    print(f"  {op:<42} {size:<22} {ROWS[-1][2]}", flush=True)


def _old_purge(n_times, test_positions, times, t1, embargo):
    """The pre-rewrite O(n x m) purge, verbatim (for the speed-up)."""
    test_set = {int(p) for p in test_positions}
    blocked = set(test_set)
    test_arr = np.array(sorted(test_set), dtype=np.int64)
    ti = times[test_arr]
    ei = t1[test_arr]
    for j in range(n_times):
        if j in test_set:
            continue
        if bool(np.any((times[j] <= ei) & (ti <= t1[j]))):
            blocked.add(j)
    if embargo > 0:
        for _s, end in _contiguous_blocks(test_arr):
            blocked.update(range(end + 1, min(n_times - 1, end + embargo) + 1))
    return np.array([p for p in range(n_times) if p not in blocked], dtype=np.int64)


def bench_purge(rng: np.random.Generator) -> None:
    for n in (5_000, 50_000, 500_000):
        times = np.arange(n, dtype=np.int64)
        t1 = (times + rng.integers(0, 20, n)).astype(np.float64)
        t1[-10:] = np.nan
        test = np.arange(n // 5, 2 * n // 5)
        new = best_of(partial(_purge_embargo_positions_t1, n, test, times, t1, 5))
        if n <= 50_000:
            old = once(partial(_old_purge, n, test, times, t1, 5))
            record("purge per fold (old loop)", f"{n:,} times", old)
            record(
                "purge per fold (span table)", f"{n:,} times", new, f"{old / new:,.0f}x"
            )
        else:
            record("purge per fold (span table)", f"{n:,} times", new)


def daily_panel(n_ent: int, n_t: int, rng: np.random.Generator) -> pl.DataFrame:
    """GBM log-prices with random triple-barrier-like spans h in [1, 20]."""
    n = n_ent * n_t
    t = np.tile(np.arange(n_t), n_ent)
    t1 = t + rng.integers(0, 20, n)
    day0 = np.datetime64("2000-01-03", "D")
    ends = pl.Series("t1", day0 + np.minimum(t1, n_t - 1).astype("timedelta64[D]"))
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_ent), n_t),
            "date": pl.Series(day0 + t.astype("timedelta64[D]")),
            "close": np.exp(rng.normal(0, 0.01, (n_ent, n_t)).cumsum(axis=1).ravel()),
        }
    ).with_columns(pl.when(pl.Series(t1 < n_t)).then(ends).alias("t1"))


def bench_weights(rng: np.random.Generator, n_ent: int, n_t: int) -> None:
    df = daily_panel(n_ent, n_t, rng)
    size = f"{n_ent:,} x {n_t:,} = {df.height / 1e6:.0f}M rows"
    record("_spans_from_t1", size, best_of(lambda: _spans_from_t1(df, t1="t1"), 2))
    _, table = _spans_from_t1(df, t1="t1")

    def conc_u() -> None:
        c = _concurrency(table.start, table.end, table.n_rows)
        inv = np.zeros(c.shape[0])
        inv[c > 0] = 1.0 / c[c > 0]
        _span_sums(inv, table.start, table.end, segments=table.segments)

    record(
        "concurrency + uniqueness (kernels)",
        f"{len(table) / 1e6:.0f}M spans",
        best_of(conc_u, 2),
    )
    record(
        "weights.effective_n (frame)",
        size,
        best_of(lambda: pn.weights.effective_n(df, t1="t1"), 2),
    )
    bound = pn.weights.FoldWeights(
        t1="t1", kind="return", price="close", decay=0.5
    ).bind(df)
    utimes = df.get_column("date").unique().sort()
    n_times = utimes.len()
    test = np.arange(2 * n_times // 5, 3 * n_times // 5)
    train = np.r_[np.arange(0, test[0] - 25), np.arange(test[-1] + 25, n_times)]
    record(
        "FoldWeights.compute (return+decay)",
        size + ", 1 fold",
        best_of(lambda: bound.compute(train, test_positions=test), 2),
    )


def bench_bootstrap(rng: np.random.Generator, big: bool) -> None:
    for n in (5_000, 100_000):
        start = np.sort(rng.integers(0, 2 * n, n)).astype(np.int64)
        end = start + rng.integers(0, 20, n)
        u = rng.random(n)
        if numba_available():
            compile_s = once(partial(_seq_boot, start[:10], end[:10], u[:10]))
            record(
                "sequential bootstrap (numba)",
                f"N = {n:,}",
                best_of(partial(_seq_boot, start, end, u), 2),
                f"first-call compile {compile_s:.2f} s" if n == 5_000 else "",
            )
        else:
            record("sequential bootstrap (numba)", f"N = {n:,}", None)
        with force_numpy():
            record(
                "sequential bootstrap (numpy)",
                f"N = {n:,}",
                once(partial(_seq_boot, start, end, u)),
            )
    # stratified, E = 5000 entities, 10^6 labels (numba only unless --big)
    df = daily_panel(5_000, 200, rng)
    if numba_available():
        record(
            "sequential_bootstrap(stratify='entity')",
            "N = 1M, E = 5,000",
            once(
                lambda: pn.sample.sequential_bootstrap(df, t1="t1", stratify="entity")
            ),
        )
    if big:
        with force_numpy():
            record(
                "sequential_bootstrap(stratify, numpy)",
                "N = 1M, E = 5,000",
                once(
                    lambda: pn.sample.sequential_bootstrap(
                        df, t1="t1", stratify="entity"
                    )
                ),
            )


def bench_cusum(rng: np.random.Generator) -> None:
    n = 1_000_000
    y = rng.normal(0, 0.01, n)
    h = np.full(n, 0.02)
    if numba_available():
        compile_s = once(lambda: _cusum_filter(y[:10], h[:10]))
        record(
            "cusum_filter kernel (numba)",
            "1M rows",
            best_of(lambda: _cusum_filter(y, h)),
            f"first-call compile {compile_s:.2f} s",
        )
    with force_numpy():
        record(
            "cusum_filter kernel (pure Python)",
            "1M rows",
            best_of(lambda: _cusum_filter(y, h), 1),
        )
    df = pl.DataFrame(
        {"id": np.repeat(np.arange(5_000), 200), "ret": rng.normal(0, 0.01, n)}
    )
    expr = pn.sample.cusum_filter("ret", threshold=0.02).over("id")
    record(
        "cusum_filter expr .over(id)",
        "1M rows, 5,000 entities",
        best_of(lambda: df.select(expr), 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--big",
        action="store_true",
        help="25M-row weight kernels, numpy stratified bootstrap",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(20260929)
    warnings.simplefilter("ignore", UserWarning)  # the decay-across-the-gap note
    print(
        f"machine: {platform.machine()} {platform.processor()}  cores: {os.cpu_count()}"
    )
    print(
        f"python {sys.version.split()[0]}, numpy {np.__version__}, polars {pl.__version__}, "
        f"numba {'yes' if numba_available() else 'no'}, POLARS_MAX_THREADS={os.environ.get('POLARS_MAX_THREADS', 'default')}"
    )
    bench_purge(rng)
    bench_weights(rng, 5_000, 1_000)
    if args.big:
        bench_weights(rng, 5_000, 5_000)
    bench_bootstrap(rng, args.big)
    bench_cusum(rng)


if __name__ == "__main__":
    main()
