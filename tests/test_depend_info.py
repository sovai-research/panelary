"""Information-theoretic dependence: digamma, GCMI against analytic Gaussian MI,
variation of information."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary.depend import (
    gcmi,
    gcmi_conditional,
    gcmi_matrix,
    gcmi_pvalue,
    mi_to_r,
    psi,
    variation_of_information,
)

EULER = 0.5772156649015329


def test_psi_known_values() -> None:
    assert psi(1.0) == pytest.approx(-EULER, abs=1e-12)
    assert psi(0.5) == pytest.approx(-EULER - 2 * math.log(2), abs=1e-12)
    # Recurrence psi(x + 1) = psi(x) + 1/x over a range.
    xs = np.linspace(0.05, 40, 400)
    np.testing.assert_allclose(psi(xs + 1), psi(xs) + 1 / xs, atol=1e-11)
    assert math.isnan(psi(-1.0))


def test_psi_matches_scipy() -> None:
    special = pytest.importorskip("scipy.special")
    xs = np.concatenate([np.linspace(1e-3, 1, 50), np.linspace(1, 1e4, 200)])
    np.testing.assert_allclose(psi(xs), special.digamma(xs), rtol=0, atol=1e-10)


def _gauss(rng: np.random.Generator, cov: np.ndarray, n: int) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.8])
def test_gcmi_bivariate_analytic(rho: float) -> None:
    rng = np.random.default_rng(0)
    z = _gauss(rng, np.array([[1, rho], [rho, 1]]), 20_000)
    truth = -0.5 * math.log(1 - rho * rho)
    # Monotone transforms leave the copula (and GCMI) unchanged.
    assert gcmi(np.exp(z[:, 0]), z[:, 1] ** 3) == pytest.approx(truth, abs=0.01)
    assert mi_to_r(truth) == pytest.approx(abs(rho), abs=1e-12)


def test_gcmi_multivariate_and_conditional_analytic() -> None:
    rng = np.random.default_rng(1)
    cov = np.array(
        [
            [1.0, 0.5, 0.3, 0.4],
            [0.5, 1.0, 0.2, 0.1],
            [0.3, 0.2, 1.0, 0.5],
            [0.4, 0.1, 0.5, 1.0],
        ]
    )
    d = _gauss(rng, cov, 40_000)
    x, y, z = d[:, :2], d[:, 2:3], d[:, 3:]
    ld = lambda idx: math.log(np.linalg.det(cov[np.ix_(idx, idx)]))  # noqa: E731
    truth = 0.5 * (ld([0, 1]) + ld([2]) - ld([0, 1, 2]))
    assert gcmi(x, y) == pytest.approx(truth, abs=0.01)
    truth_c = 0.5 * (ld([0, 1, 3]) + ld([2, 3]) - ld([3]) - ld([0, 1, 2, 3]))
    assert gcmi_conditional(x, y, z) == pytest.approx(truth_c, abs=0.01)


def test_gcmi_matrix_and_pvalue() -> None:
    rng = np.random.default_rng(2)
    d = _gauss(rng, np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]), 5000)
    M = gcmi_matrix(d)
    assert np.isinf(np.diag(M)).all()
    assert M[0, 1] == pytest.approx(gcmi(d[:, 0], d[:, 1]), abs=1e-3)
    assert gcmi_pvalue(d[:, 0], d[:, 1]) < 1e-10
    assert gcmi_pvalue(d[:, 0], d[:, 2]) > 1e-3


def test_gcmi_is_blind_to_nonmonotone_dependence() -> None:
    """The honesty trap: GCMI on one pair cannot see a U-shape."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000)
    assert abs(gcmi(x, x**2)) < 0.01


def test_variation_of_information_is_a_metric() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000)
    y = x + 0.5 * rng.standard_normal(3000)
    z = rng.standard_normal(3000)
    assert variation_of_information(x, x, bins=10) == pytest.approx(0.0, abs=1e-12)
    vxy = variation_of_information(x, y, bins=10, normalize=False)
    vyz = variation_of_information(y, z, bins=10, normalize=False)
    vxz = variation_of_information(x, z, bins=10, normalize=False)
    assert vxz <= vxy + vyz + 1e-12
    assert vxy == pytest.approx(
        variation_of_information(y, x, bins=10, normalize=False)
    )
    assert 0.0 <= variation_of_information(x, z) <= 1.0
