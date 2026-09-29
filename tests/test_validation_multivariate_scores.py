"""Energy and variogram scores (``validation._multivariate_scores``)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from panelary._internal._jit import assert_backend_parity, force_numpy
from panelary.validation import crps_ensemble
from panelary.validation._multivariate_scores import (
    _variogram_kernel,
    _variogram_numpy,
    energy_score,
    variogram_score,
)


def _ensemble(t: int, m: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, d)), rng.standard_normal((t, m, d))


def _energy_loop(y: np.ndarray, x: np.ndarray, fair: bool) -> np.ndarray:
    t, m, _ = x.shape
    out = np.empty(t)
    for i in range(t):
        term1 = np.mean([np.linalg.norm(x[i, k] - y[i]) for k in range(m)])
        pair = sum(
            np.linalg.norm(x[i, k] - x[i, j]) for k in range(m) for j in range(m)
        )
        out[i] = term1 - pair / (2 * m * (m - 1) if fair else 2 * m * m)
    return out


# --------------------------------------------------------------------------- #
# Energy score
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fair", [False, True])
def test_energy_exact_matches_direct_and_a_plain_loop(fair: bool) -> None:
    y, x = _ensemble(15, 40, 4, 0)
    exact = energy_score(y, x, fair=fair)
    direct = energy_score(y, x, method="direct", fair=fair)
    np.testing.assert_allclose(exact, direct, rtol=1e-10, atol=0)
    np.testing.assert_allclose(direct, _energy_loop(y, x, fair), rtol=1e-12)


def test_energy_exact_survives_offsets_and_duplicate_members() -> None:
    y, x = _ensemble(10, 60, 5, 1)
    x[:, 1] = x[:, 0]  # exact duplicates must contribute exactly 0
    x[:, 7] = x[:, 0]
    big_y, big_x = 1e6 * y + 1e9, 1e6 * x + 1e9  # far from the origin
    np.testing.assert_allclose(
        energy_score(big_y, big_x),
        energy_score(big_y, big_x, method="direct"),
        rtol=1e-10,
    )


def test_energy_d1_is_crps_ensemble() -> None:
    rng = np.random.default_rng(2)
    y = rng.standard_normal(30)
    x = rng.standard_normal((30, 25))
    np.testing.assert_allclose(
        energy_score(y[:, None], x[:, :, None]), crps_ensemble(y, x), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(energy_score(y, x), crps_ensemble(y, x))
    fair = energy_score(y, x, fair=True)
    ref = np.array(
        [
            np.mean(np.abs(x[i] - y[i]))
            - np.abs(x[i][:, None] - x[i][None, :]).sum() / (2 * 25 * 24)
            for i in range(30)
        ]
    )
    np.testing.assert_allclose(fair, ref, rtol=1e-12)


def test_energy_sliced_error_at_1024_projections() -> None:
    """The sliced score is unbiased; at K = 1024 its error is small (measured)."""
    y, x = _ensemble(20, 200, 8, 3)
    exact = energy_score(y, x)
    with pytest.warns(UserWarning, match="Monte Carlo approximation"):
        sliced = energy_score(y, x, method="sliced", n_projections=1024, seed=0)
    rel = np.abs(sliced / exact - 1.0)
    # measured on this fixture: K=1024 max 0.64%, mean 0.20%; K=16 mean 1.7%
    assert rel.max() < 0.01 and rel.mean() < 0.004
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        again = energy_score(y, x, method="sliced", n_projections=1024, seed=0)
        coarse = energy_score(y, x, method="sliced", n_projections=16, seed=0)
    np.testing.assert_array_equal(sliced, again)  # seeded
    assert np.abs(coarse / exact - 1.0).mean() > rel.mean()


def test_energy_nan_rows_and_validation() -> None:
    y, x = _ensemble(5, 10, 3, 4)
    x[2, 3, 1] = np.nan
    y[4, 0] = np.inf
    out = energy_score(y, x)
    assert np.isnan(out[[2, 4]]).all() and np.isfinite(out[[0, 1, 3]]).all()
    with pytest.raises(ValueError, match="method"):
        energy_score(y, x, method="fast")
    with pytest.raises(ValueError, match="does not match"):
        energy_score(y[:, :2], x)
    with pytest.raises(ValueError, match="m <= 256"):
        energy_score(np.zeros((1, 2)), np.zeros((1, 300, 2)), method="direct")
    with pytest.raises(ValueError, match="fair"):
        energy_score(np.zeros((1, 2)), np.zeros((1, 1, 2)), fair=True)


def test_energy_chunking_is_invisible() -> None:
    y, x = _ensemble(25, 30, 3, 5)
    np.testing.assert_array_equal(
        energy_score(y, x), energy_score(y, x, chunk_bytes=30 * 30 * 8 * 3)
    )


# --------------------------------------------------------------------------- #
# Variogram score
# --------------------------------------------------------------------------- #
def _variogram_loop(
    y: np.ndarray, x: np.ndarray, p: float, w: np.ndarray
) -> np.ndarray:
    t, _, d = x.shape
    out = np.zeros(t)
    for i in range(t):
        for a in range(d):
            for b in range(a + 1, d):
                obs = abs(y[i, a] - y[i, b]) ** p
                ens = np.mean(np.abs(x[i, :, a] - x[i, :, b]) ** p)
                out[i] += w[a, b] * (obs - ens) ** 2
    return out


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 0.8])
def test_variogram_matches_a_triple_loop(p: float) -> None:
    y, x = _ensemble(12, 30, 6, 6)
    w = np.random.default_rng(7).random((6, 6))
    np.testing.assert_allclose(
        variogram_score(y, x, p=p, weights=w), _variogram_loop(y, x, p, w), rtol=1e-12
    )
    np.testing.assert_allclose(
        variogram_score(y, x, p=p),
        _variogram_loop(y, x, p, np.ones((6, 6))),
        rtol=1e-12,
    )


def test_variogram_numba_matches_numpy_twin() -> None:
    pytest.importorskip("numba")
    y, x = _ensemble(9, 50, 12, 8)
    w = np.random.default_rng(9).random((12, 12))
    kernel = _variogram_kernel.compiled()
    assert kernel is not None
    for p, code in ((0.5, 0), (1.0, 1), (2.0, 2)):
        fast = np.empty(9)
        kernel(y, x, w, p, code, fast)
        assert_backend_parity(fast, _variogram_numpy(y, x, w, p, code, 2**20))
    # a general p goes through pow() on both sides; bitwise on this platform,
    # documented as last-bits only.
    fast = np.empty(9)
    kernel(y, x, w, 0.7, 3, fast)
    np.testing.assert_allclose(
        fast, _variogram_numpy(y, x, w, 0.7, 3, 2**20), rtol=1e-14
    )
    with force_numpy():
        assert_backend_parity(
            variogram_score(y, x, weights=w),
            _variogram_numpy(y, x, w, 0.5, 0, 256 * 2**20),
        )


def test_variogram_chunking_edge_shapes_and_validation() -> None:
    y, x = _ensemble(11, 7, 4, 10)
    with force_numpy():
        assert_backend_parity(
            variogram_score(y, x), variogram_score(y, x, chunk_bytes=100)
        )
    # d = 1 has no pairs: the score is 0
    np.testing.assert_array_equal(variogram_score(y[:, :1], x[:, :, :1]), 0.0)
    x[3, 0, 0] = np.nan
    assert np.isnan(variogram_score(y, x)[3])
    with pytest.raises(ValueError, match="positive"):
        variogram_score(y, x, p=0.0)
    with pytest.raises(ValueError, match="shape"):
        variogram_score(y, x, weights=np.ones((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        variogram_score(y, x, weights=-np.ones((4, 4)))


# --------------------------------------------------------------------------- #
# Propriety
# --------------------------------------------------------------------------- #
def _correlated(t: int, m: int, d: int, rho: float, seed: int):
    rng = np.random.default_rng(seed)
    cov = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    chol = np.linalg.cholesky(cov)
    y = rng.standard_normal((t, d)) @ chol.T
    truth = rng.standard_normal((t, m, d)) @ chol.T
    indep = rng.standard_normal((t, m, d))  # right marginals, wrong dependence
    return y, truth, indep, rng


def test_energy_score_propriety() -> None:
    y, truth, _, rng = _correlated(2000, 50, 4, 0.6, 11)
    shifted = truth + 0.3
    wide = truth * 1.4
    narrow = truth * 0.7
    scores = {
        name: float(np.mean(energy_score(y, ens, fair=True)))
        for name, ens in {
            "truth": truth,
            "shifted": shifted,
            "wide": wide,
            "narrow": narrow,
        }.items()
    }
    assert min(scores, key=scores.get) == "truth"  # type: ignore[arg-type]


def test_variogram_score_propriety_and_dependence_sensitivity() -> None:
    y, truth, indep, _ = _correlated(2000, 50, 4, 0.6, 12)
    vs_truth = float(np.mean(variogram_score(y, truth)))
    vs_indep = float(np.mean(variogram_score(y, indep)))
    vs_strong = float(
        np.mean(variogram_score(y, _correlated(2000, 50, 4, 0.95, 13)[1]))
    )
    assert vs_truth < vs_indep and vs_truth < vs_strong
    # the energy score barely separates the dependence error (Pinson & Tastu)
    es_gap = np.mean(energy_score(y, indep)) / np.mean(energy_score(y, truth)) - 1
    vs_gap = vs_indep / vs_truth - 1
    assert vs_gap > 5 * es_gap > 0
