"""The rank primitive layer of :mod:`panelary.depend`."""

from __future__ import annotations

import numpy as np
import pytest

from panelary.depend import (
    dominance_counts,
    normal_scores,
    pairwise_complete,
    ranks,
    sliding_ranks,
)
from panelary.depend._ranks import lag1_rank_autocorr


@pytest.mark.parametrize("method", ["average", "ordinal", "dense", "min", "max"])
def test_ranks_match_scipy(method: str) -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 12, size=(4, 7, 50)).astype(float)  # heavy ties, batched
    ours = ranks(x, method=method)
    ref = stats.rankdata(x, method=method, axis=-1)
    np.testing.assert_array_equal(ours, ref)
    assert ours.dtype == (np.float64 if method == "average" else np.int64)


def test_ranks_axis_and_empty() -> None:
    x = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    np.testing.assert_array_equal(
        ranks(x, axis=0), [[3.0, 2.0], [1.0, 3.0], [2.0, 1.0]]
    )
    assert ranks(np.empty(0)).shape == (0,)
    with pytest.raises(ValueError, match="unknown rank"):
        ranks(x, method="first")


def test_normal_scores_standard_marginal() -> None:
    rng = np.random.default_rng(1)
    z = normal_scores(rng.standard_t(2, size=5001))
    assert abs(z.mean()) < 1e-12
    assert abs(np.median(z)) < 1e-12
    assert z.std() == pytest.approx(1.0, abs=0.01)


def test_sliding_ranks_within_window_and_nan() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(40)
    x[17] = np.nan
    r = sliding_ranks(x, 5)
    assert r.shape == (36, 5)
    for k in range(36):
        win = x[k : k + 5]
        if np.isnan(win).any():
            assert np.isnan(r[k]).all()
        else:
            np.testing.assert_array_equal(r[k], ranks(win))
    with pytest.raises(ValueError):
        sliding_ranks(x, 0)
    with pytest.raises(ValueError):
        sliding_ranks(x[:3], 5)


def _brute_dominance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((x[None, :] < x[:, None]) & (y[None, :] < y[:, None])).sum(axis=1)


@pytest.mark.parametrize("n", [1, 2, 3, 17, 100, 1003])
def test_dominance_counts_bruteforce(n: int) -> None:
    rng = np.random.default_rng(n)
    x = rng.integers(0, max(2, n // 3), n).astype(float)  # ties in both
    y = rng.integers(0, max(2, n // 4), n).astype(float)
    np.testing.assert_array_equal(dominance_counts(x, y), _brute_dominance(x, y))
    xc, yc = rng.standard_normal(n), rng.standard_normal(n)
    np.testing.assert_array_equal(dominance_counts(xc, yc), _brute_dominance(xc, yc))


def test_pairwise_complete_drops_nonfinite() -> None:
    x = np.array([1.0, np.nan, 3.0, 4.0, np.inf])
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    xa, ya, n = pairwise_complete(x, y)
    assert n == 2
    np.testing.assert_array_equal(xa, [1.0, 4.0])
    np.testing.assert_array_equal(ya, [1.0, 4.0])
    X = np.column_stack([x, x])
    xa2, _, n2 = pairwise_complete(X, y)
    assert n2 == 2 and xa2.shape == (2, 2)
    with pytest.raises(ValueError):
        pairwise_complete(x, y[:3])


def test_lag1_rank_autocorr() -> None:
    rng = np.random.default_rng(3)
    e = rng.standard_normal(2000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, e.size):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    assert abs(lag1_rank_autocorr(e)) < 0.1
    assert lag1_rank_autocorr(ar) > 0.8
    assert np.isnan(lag1_rank_autocorr(np.array([1.0, 2.0])))
