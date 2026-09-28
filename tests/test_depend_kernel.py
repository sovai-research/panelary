"""Random-Fourier-feature HSIC and its frozen feature map."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp
from panelary.depend._kernel import median_heuristic


def _exact_hsic_b(x: np.ndarray, y: np.ndarray, s: float = 1.0) -> float:
    zx, zy = dp.normal_scores(x), dp.normal_scores(y)
    K = np.exp(-((zx[:, None] - zx[None, :]) ** 2) / (2 * s * s))
    L = np.exp(-((zy[:, None] - zy[None, :]) ** 2) / (2 * s * s))
    n = x.size
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H)) / n**2


def test_rff_hsic_approximates_exact_hsic() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(600)
    y = np.cos(2 * x) + 0.5 * rng.standard_normal(600)
    exact = _exact_hsic_b(x, y)
    approx = np.mean([dp.hsic(x, y, n_features=512, seed=2 * s).hsic for s in range(8)])
    assert approx == pytest.approx(exact, rel=0.1)


def test_hsic_detects_and_gamma_null_is_calibrated() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    res = dp.hsic(x, np.abs(x) + 0.3 * rng.standard_normal(500))
    assert res.p_value < 1e-8 and 0 < res.normalised <= 1
    pv = [
        dp.hsic(
            rng.standard_normal(300), rng.standard_normal(300), n_features=64, seed=s
        ).p_value
        for s in range(400)
    ]
    rate = float(np.mean(np.array(pv) <= 0.05))
    assert 0.02 <= rate <= 0.09, rate


def test_rff_map_is_frozen_and_bandwidth_is_a_fit() -> None:
    """Invariant 7: the bandwidth and frequencies are fitted once; transforming
    the training rows must not depend on data seen later."""
    rng = np.random.default_rng(2)
    train = rng.standard_normal((200, 2))
    test = 5.0 + 3.0 * rng.standard_normal((100, 2))  # a regime the train set never saw
    honest = dp.RFFMap.fit(train, n_features=32, copula=False, seed=3)
    np.testing.assert_array_equal(honest.transform(train), honest.transform(train))
    assert honest.bandwidth == pytest.approx(median_heuristic(train))
    # A map fitted on train + test (the leak) gives different training features.
    leaky = dp.RFFMap.fit(np.vstack([train, test]), n_features=32, copula=False, seed=3)
    assert leaky.bandwidth != pytest.approx(honest.bandwidth, rel=0.05)
    assert np.abs(leaky.transform(train) - honest.transform(train)).max() > 1e-3
    # The copula map estimates nothing from the data but its dimension.
    c1 = dp.RFFMap.fit(train, copula=True, seed=3)
    c2 = dp.RFFMap.fit(np.vstack([train, test]), copula=True, seed=3)
    assert c1.bandwidth == c2.bandwidth == 1.0
    np.testing.assert_array_equal(c1.W, c2.W)


def test_hsic_matrix_and_engine_kernel() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal(800)
    A = np.column_stack(
        [a, a**2 + 0.2 * rng.standard_normal(800), rng.standard_normal(800)]
    )
    M = dp.hsic_matrix(A, n_features=64)
    np.testing.assert_allclose(np.diag(M), 1.0)
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    assert M[0, 1] > 10 * M[0, 2]
    df = pl.DataFrame({"a": A[:, 0], "b": A[:, 1], "c": A[:, 2]})
    out = dp.dependence(df, "a", "b", method="hsic")
    assert out["null_method"][0] == "gamma" and out["p_value"][0] < 1e-8
    assert "D=64" in out["estimator"][0]
    nul = dp.dependence(df, "a", "c", method="hsic", null="permutation", n_resamples=99)
    assert nul["p_value"][0] > 0.01
