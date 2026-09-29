"""Benchmarks for plan 4 M3 / M5 / M6: trend scanning, bars, bet sizing.

Prints one table: the operation, its size, the wall-clock seconds (best of
``--repeat`` after a warm-up) and a linear extrapolation to the plan's budget
size. numba's first-call compile time is measured and reported separately, never
folded into throughput. Datasets are generated in-script, seeded.

Run::

    python benchmarks/bench_trend_bars_sizing.py                  # defaults
    python benchmarks/bench_trend_bars_sizing.py --rows 5e6 --ticks 1e8

The defaults take a few minutes (most of it the numpy sweep and the
pure-Python scans). ``--ticks 1e8`` needs ~16 GB of RAM. Extrapolations are
linear and say so; re-measure at full size, on an idle machine, before quoting
a number as measured.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime

import numpy as np
import polars as pl

import panelary as pn
from panelary._internal._deps import have
from panelary.label import _trend
from panelary.sample import _imbalance


def best_of(fn: Callable[[], object], repeat: int) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def ticks_frame(n: int, n_sym: int, seed: int = 0) -> pl.DataFrame:
    """1-cent price grid, lognormal integer sizes, ~5 s between trades."""
    rng = np.random.default_rng(seed)
    px = 100.0 + 0.01 * np.cumsum(rng.integers(-2, 3, n))
    qty = np.maximum(1, rng.lognormal(4.0, 1.0, n)).astype(np.int64)
    sym = np.repeat(np.arange(n_sym), -(-n // n_sym))[:n]
    ts = np.datetime64("2024-01-02T09:30") + np.cumsum(
        rng.integers(1, 10_000, n)
    ).astype("timedelta64[ms]")
    return pl.DataFrame({"sym": sym, "ts": ts, "px": px, "qty": qty})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=float, default=2e6, help="rows for trend / sizing")
    ap.add_argument("--ticks", type=float, default=1e7, help="ticks for bars")
    ap.add_argument(
        "--py-ticks", type=float, default=1e6, help="ticks for the pure-Python scan"
    )
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    rows, ticks, py_ticks = int(args.rows), int(args.ticks), int(args.py_ticks)
    numba = have("numba")

    print(
        f"# {platform.platform()} | {os.cpu_count()} cores | Python "
        f"{platform.python_version()} | numpy {np.__version__} | polars "
        f"{pl.__version__} | numba {'yes' if numba else 'no'} | {datetime.now():%Y-%m-%d}"
    )
    table: list[tuple[str, str, float, str]] = []

    def row(name: str, size: int, secs: float, target: int, unit: str) -> None:
        extra = f"{secs * target / size:.2f} s @ {target:,} {unit} (linear extrap.)"
        table.append((name, f"{size:,} {unit}", secs, extra))

    # --- trend scanning: the sweep, L = 5..50 -------------------------------
    rng = np.random.default_rng(1)
    y = np.log(100.0) + np.cumsum(rng.standard_normal(rows) * 0.01)
    avail = rows - np.arange(rows)
    kw = {"direction": 1, "min_w": 5, "l_max": 50, "step": 1}
    row(
        "trend sweep numpy",
        rows,
        best_of(lambda: _trend._sweep(y, avail, _backend="numpy", **kw), args.repeat),
        25_000_000,
        "rows",
    )
    if numba:
        t0 = time.perf_counter()
        _trend._sweep(y[:100], avail[:100] - rows + 100, _backend="numba", **kw)
        compile_s = time.perf_counter() - t0
        for threads in (1, os.cpu_count() or 1):
            secs = best_of(
                lambda t=threads: _trend._sweep(
                    y, avail, _backend="numba", threads=t, **kw
                ),
                args.repeat,
            )
            row(f"trend sweep numba ({threads} thr)", rows, secs, 25_000_000, "rows")
        table.append(("  numba compile (first call)", "-", compile_s, "excluded above"))

    n_ent = 5000
    per = max(60, rows // n_ent)
    panel = pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_ent), per),
            "t": np.tile(np.arange(per), n_ent),
            "y": np.cumsum(rng.standard_normal(n_ent * per)) * 0.01,
        }
    )
    expr = pl.col("y").ts.trend_scan(min_window=5, max_window=50).over("id")
    row(
        ".ts.trend_scan .over(5000 ids)",
        panel.height,
        best_of(lambda: panel.with_columns(expr), args.repeat),
        25_000_000,
        "rows",
    )
    prices = panel.with_columns((pl.col("y") + 5.0).exp().alias("close"))
    row(
        "label.trend_scanning",
        prices.height,
        best_of(
            lambda: pn.label.trend_scanning(prices, min_window=5, max_window=50),
            args.repeat,
        ),
        25_000_000,
        "rows",
    )

    # --- bars ------------------------------------------------------------------
    tk = ticks_frame(ticks, n_sym=50)
    thr = float((tk["px"] * tk["qty"]).sum()) / (ticks / 100)
    row(
        "sample.bars dollar (lattice)",
        ticks,
        best_of(
            lambda: pn.sample.bars(
                tk,
                entity="sym",
                time="ts",
                price="px",
                size="qty",
                kind="dollar",
                threshold=thr,
            ),
            args.repeat,
        ),
        100_000_000,
        "ticks",
    )
    one = ticks_frame(ticks, n_sym=1, seed=2)
    side = (
        one.select(_imbalance._tick_sign(pl.col("px")).fill_null(0))
        .to_series()
        .to_numpy()
    )
    amt = one["qty"].cast(pl.Float64).to_numpy()
    starts = np.zeros(ticks, dtype=bool)
    starts[0] = True
    scan_kw = {
        "init_t": 2000.0,
        "lo_t": 200.0,
        "hi_t": 20000.0,
        "alpha": 2 / 21,
        "warmup": 2000,
        "init_imb": None,
    }
    py_side, py_amt, py_starts = side[:py_ticks], amt[:py_ticks], starts[:py_ticks]
    for run in (False, True):
        name = "run" if run else "imbalance"
        if numba:
            secs = best_of(
                lambda r=run: _imbalance._info_bars_scan(
                    side, amt, starts, run=r, _backend="numba", **scan_kw
                ),
                args.repeat,
            )
            row(f"{name} bars scan numba (1 symbol)", ticks, secs, 100_000_000, "ticks")
        secs = best_of(
            lambda r=run: _imbalance._info_bars_scan(
                py_side, py_amt, py_starts, run=r, _backend="numpy", **scan_kw
            ),
            1,
        )
        row(f"{name} bars scan pure Python", py_ticks, secs, 100_000_000, "ticks")
    row(
        "sample.imbalance_bars end-to-end",
        ticks,
        best_of(
            lambda: pn.sample.imbalance_bars(
                tk,
                entity="sym",
                time="ts",
                price="px",
                size="qty",
                kind="volume",
                init_expected_ticks=2000,
            ),
            1,
        ),
        100_000_000,
        "ticks",
    )

    # --- sizing -------------------------------------------------------------------
    t = np.tile(np.arange(rows // 1000), 1000)
    bets = rng.random(t.size) < 0.2
    sz = pl.DataFrame(
        {
            "id": np.repeat(np.arange(1000), rows // 1000),
            "t": t,
            "prob": np.where(bets, rng.uniform(0.4, 0.99, t.size), np.nan),
            "side": rng.choice([-1.0, 1.0], t.size),
            "exit": t + rng.integers(1, 30, t.size),
        }
    ).with_columns(pl.col("prob").fill_nan(None))
    row(
        "sizing.bet_size + average_active",
        sz.height,
        best_of(
            lambda: pn.sizing.average_active(
                pn.sizing.bet_size(sz), entity="id", time="t"
            ),
            args.repeat,
        ),
        25_000_000,
        "rows",
    )

    width = max(len(r[0]) for r in table)
    print(f"{'operation':<{width}}  {'size':>18}  {'seconds':>9}  note")
    for name, size, secs, note in table:
        print(f"{name:<{width}}  {size:>18}  {secs:9.3f}  {note}")


if __name__ == "__main__":
    main()
