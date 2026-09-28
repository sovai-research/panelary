"""Prefix invariance of the rolling dependence features: bitwise
``f(x[:T])[t] == f(x[:T+k])[t]`` at several ``T``.

The build contract names three ways to break it, and each gets a deliberately
leaky twin that must FAIL the same check -- otherwise the check proves nothing:

1. a window derived from the series length (``n // k``) instead of fixed;
2. ranks taken over the whole series and then sliced into windows;
3. a tail threshold (quantile) fitted on the whole series instead of the window.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp  # noqa: F401 -- registers the .ts rolling ops
from panelary.depend._coef import _xi_rows
from panelary.depend._ranks import ranks
from panelary.depend._rolling import window_statistic


def _series(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = np.sin(2 * a) + 0.4 * rng.standard_normal(n)
    return a, b


_OPS = {
    "rolling_xi": lambda: pl.col("y").ts.rolling_xi("x", window=40),
    "rolling_xi_self": lambda: pl.col("y").ts.rolling_xi(window=40),
    "rolling_dcor": lambda: pl.col("y").ts.rolling_dcor("x", window=40),
    "rolling_tail_dep": lambda: pl.col("y").ts.rolling_tail_dep("x", window=50, q=0.1),
    "rolling_tail_dep_upper": lambda: pl.col("y").ts.rolling_tail_dep(
        "x", window=50, q=0.2, side="upper"
    ),
    "rolling_gcmi": lambda: pl.col("y").ts.rolling_gcmi("x", window=40),
}


@pytest.mark.parametrize("name", sorted(_OPS))
def test_rolling_ops_are_bitwise_prefix_invariant(name: str) -> None:
    a, b = _series()
    df = pl.DataFrame({"x": a, "y": b})
    full = df.select(_OPS[name]().alias("f"))["f"].to_numpy()
    assert np.isfinite(full).sum() > 200  # non-vacuous
    for cut in (60, 137, 250):
        part = df.head(cut).select(_OPS[name]().alias("f"))["f"].to_numpy()
        np.testing.assert_array_equal(part, full[:cut])


def _leaky_window_len(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return window_statistic(a, b, max(5, a.size // 6), _xi_rows)  # rule 1 broken


def _global_rank_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rule 2 broken: rank the whole series, slice windows, correlate the
    global ranks (a rank correlation whose ranks saw the future)."""
    ra, rb = ranks(a) / a.size, ranks(b) / b.size

    def rows(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
        return (wa * wb).mean(axis=1)

    return window_statistic(ra, rb, 40, rows)


def _global_tail_threshold(a: np.ndarray, b: np.ndarray, q: float = 0.1) -> np.ndarray:
    """Rule 3 broken: the q-quantile is fitted on the full series."""
    ta, tb = np.quantile(a, q), np.quantile(b, q)

    def rows(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
        return ((wa <= ta) & (wb <= tb)).mean(axis=1) / q

    return window_statistic(a, b, 50, rows)


@pytest.mark.parametrize(
    "leaky",
    [_leaky_window_len, _global_rank_pearson, _global_tail_threshold],
    ids=["window=n//6", "global-ranks", "global-tail-quantile"],
)
def test_each_leak_breaks_prefix_invariance(leaky) -> None:  # type: ignore[no-untyped-def]
    a, b = _series()
    full = leaky(a, b)
    broken = False
    for cut in range(60, 300, 7):
        part = leaky(a[:cut], b[:cut])
        both = np.isfinite(part) & np.isfinite(full[:cut])
        if both.any() and not np.array_equal(part[both], full[:cut][both]):
            broken = True
    assert broken, "the leaky twin passed: the check would not have caught it"


def test_rolling_matches_direct_window_computation() -> None:
    a, b = _series(120, seed=3)
    df = pl.DataFrame({"x": a, "y": b})
    w = 30
    got = df.select(pl.col("y").ts.rolling_xi("x", window=w).alias("f"))["f"].to_numpy()
    assert np.isnan(got[: w - 1]).all()
    for t in (w - 1, 64, 119):
        assert got[t] == pytest.approx(
            dp.xi(a[t - w + 1 : t + 1], b[t - w + 1 : t + 1]), abs=1e-12
        )
    d = df.select(pl.col("y").ts.rolling_dcor("x", window=w).alias("f"))["f"].to_numpy()
    assert d[80] == pytest.approx(dp.dcor(a[51:81], b[51:81]), abs=1e-10)
    lo = df.select(pl.col("y").ts.rolling_tail_dep("x", window=60, q=0.1).alias("f"))[
        "f"
    ].to_numpy()
    assert lo[100] == pytest.approx(dp.tail_dependence(a[41:101], b[41:101], q=0.1)[0])
    g = df.select(pl.col("y").ts.rolling_gcmi("x", window=w).alias("f"))["f"].to_numpy()
    from panelary.depend._info import _gcmi_rows

    assert g[90] == pytest.approx(
        float(_gcmi_rows(a[61:91][None], b[61:91][None])[0]), abs=1e-12
    )
    # Self-lag default: xi(y[t-1] -> y[t]) over the window.
    s = df.select(pl.col("y").ts.rolling_xi(window=w).alias("f"))["f"].to_numpy()
    assert np.isnan(s[:w]).all()
    assert s[50] == pytest.approx(dp.xi(b[20:50], b[21:51]), abs=1e-12)


def test_missing_values_null_their_windows_only() -> None:
    a, b = _series(100, seed=4)
    b[40] = np.nan
    df = pl.DataFrame({"x": a, "y": b})
    out = df.select(pl.col("y").ts.rolling_xi("x", window=10).alias("f"))["f"]
    vals = out.to_numpy()
    assert out[45] is None and out[49] is None
    assert np.isfinite(vals[39]) and np.isfinite(vals[50])


def test_window_validation() -> None:
    with pytest.raises(TypeError, match="fixed integer"):
        pl.col("y").ts.rolling_xi("x", window=20.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pl.col("y").ts.rolling_xi("x", window=2)
    with pytest.raises(ValueError, match="floor"):
        pl.col("y").ts.rolling_tail_dep("x", window=5, q=0.1)
    with pytest.raises(ValueError):
        pl.col("y").ts.rolling_tail_dep("x", window=50, side="middle")


def test_lag_profile_pairs_by_date_not_by_position() -> None:
    """A lag profile is a whole-sample statistic, so 'prefix invariance' for it
    means: lag-k pairs are built on the date axis with no wrap-around, and the
    profile on a prefix equals a direct computation on that prefix."""
    a, b = _series(200, seed=5)
    df = pl.DataFrame({"t": np.arange(200), "x": a, "y": b})
    prof = dp.lag_dependence(
        df.head(150), "x", "y", time="t", lags=[2], method="xi", null="asymptotic"
    )
    assert prof["estimate"][0] == pytest.approx(dp.xi(a[:148], b[2:150]), abs=1e-12)
    assert prof["n_obs"][0] == 148
