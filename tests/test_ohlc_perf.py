"""Wall-clock budgets of the daily OHLC estimators (plan 5 §11), scaled down.

The plan's budgets are for 25M rows (5000 entities x 5000 bars) and fail at 2x;
``benchmarks/ohlc_vol/bench_daily.py`` measures that size. Here the same
workloads run on 500 x 2000 = 1M rows with each budget scaled linearly
(x 1/25) and the 2x failure threshold applied, floored at 0.25 s for the fixed
per-call cost (sorting, validation) that does not shrink with the panel. Each
timing is the best of three runs. Measured on the development machine at 1M rows
(2026-09-29, under load): P+GK+RS 0.11 s, YZ 0.06 s, EDGE 0.17 s, CS 0.06 s,
AR 0.05 s, price impact 0.04 s, PS gamma 0.06 s, FHT 0.04 s.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from panelary.econ import features as F

pytestmark = pytest.mark.benchmark

#: Hosted CI runners are shared and several times slower; the repo's stance
#: (tests/test_depend_perf.py) is to skip wall-clock budgets there unless
#: ``PANELARY_STRICT_TIMING=1``.
wall_clock = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true"
    and os.environ.get("PANELARY_STRICT_TIMING", "").lower()
    not in {"1", "true", "yes"},
    reason="wall-clock budget; hosted-runner timings are noisy (PANELARY_STRICT_TIMING=1 enforces)",
)

_SCALE = (500 * 2000) / (5000 * 5000)


def _budget(full_size_seconds: float) -> float:
    return max(2.0 * full_size_seconds * _SCALE, 0.25)


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n_ent, n_bars = 500, 2000
    n = n_ent * n_bars
    intraday = rng.standard_normal(n) * 0.015
    lc = np.cumsum(
        (intraday + rng.standard_normal(n) * 0.006).reshape(n_ent, n_bars), axis=1
    ).ravel()
    lo_ = lc - intraday
    lh = np.maximum(lo_, lc) + np.abs(rng.standard_normal(n)) * 0.008
    ll = np.minimum(lo_, lc) - np.abs(rng.standard_normal(n)) * 0.008
    ret = np.diff(lc, prepend=0.0)
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_ent, dtype=np.int32), n_bars),
            "t": np.tile(np.arange(n_bars, dtype=np.int32), n_ent),
            "open": np.exp(lo_ + 3.0),
            "high": np.exp(lh + 3.0),
            "low": np.exp(ll + 3.0),
            "close": np.exp(lc + 3.0),
            "ret": np.round(ret, 3),
            "mkt": rng.standard_normal(n) * 0.01,
            "dv": np.exp(rng.standard_normal(n) + 15.0),
        }
    )


def _best(fn: Callable[[], object], repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _range(df: pl.DataFrame, *methods: str) -> Callable[[], object]:
    def run() -> object:
        out = df
        for m in methods:
            out = F.range_volatility(out, entity="id", time="t", method=m)
        return out

    return run


_WORKLOADS: dict[str, tuple[float, Callable[[pl.DataFrame], Callable[[], object]]]] = {
    "P + GK + RS": (
        2.5,
        lambda df: _range(df, "parkinson", "garman_klass", "rogers_satchell"),
    ),
    "Yang-Zhang": (3.0, lambda df: _range(df, "yang_zhang")),
    "EDGE batched": (
        12.0,
        lambda df: lambda: F.ohlc_spread(df, entity="id", time="t", method="edge"),
    ),
    "Corwin-Schultz": (
        3.0,
        lambda df: (
            lambda: F.ohlc_spread(df, entity="id", time="t", method="corwin_schultz")
        ),
    ),
    "Abdi-Ranaldo": (
        3.0,
        lambda df: (
            lambda: F.ohlc_spread(df, entity="id", time="t", method="abdi_ranaldo")
        ),
    ),
    "price_impact": (
        3.0,
        lambda df: (
            lambda: F.price_impact(
                df, entity="id", time="t", returns="ret", dollar_volume="dv"
            )
        ),
    ),
    "pastor_stambaugh_gamma": (
        5.0,
        lambda df: (
            lambda: F.pastor_stambaugh_gamma(
                df,
                entity="id",
                time="t",
                returns="ret",
                market_returns="mkt",
                dollar_volume="dv",
            )
        ),
    ),
    "fht_spread": (
        3.0,
        lambda df: lambda: F.fht_spread(df, entity="id", time="t", returns="ret"),
    ),
}


@wall_clock
@pytest.mark.parametrize("name", list(_WORKLOADS))
def test_daily_ohlc_budget(bars: pl.DataFrame, name: str) -> None:
    full_size, make = _WORKLOADS[name]
    elapsed = _best(make(bars))
    budget = _budget(full_size)
    assert elapsed < budget, (
        f"{name}: {elapsed:.3f} s on 1M rows, budget {budget:.3f} s "
        f"(plan: {full_size} s at 25M rows, fail at 2x)"
    )
