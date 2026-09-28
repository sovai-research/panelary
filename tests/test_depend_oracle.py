"""Parity of :mod:`panelary.depend` against external references.

Each oracle sits behind ``pytest.importorskip`` and states its tolerance.
Oracles are **test-only**: nothing here is imported by the library.

=========================  =====================================  ==========
Ours                       Oracle                                 Tolerance
=========================  =====================================  ==========
``xi``                     ``scipy.stats.chatterjeexi`` (BSD-3)   1e-12
``kendall`` / p-value      ``scipy.stats.kendalltau`` (BSD-3)     1e-12 / 1e-6 rel
``dcor``, ``partial_dcor`` ``dcor`` package (MIT)                 1e-8 / 1e-6 rel
``dcor``, ``hsic``         ``hyppo`` (MIT)                        stated below
``hoeffding_d``            hand-computed small-n cases            exact
``gcmi``                   analytic Gaussian MI (closed form)     see test_depend_info
``mi_ksg``                 ``sklearn`` KSG (BSD-3), ``ennemi``    1e-10
``psi``                    known values / ``scipy.special``       1e-12 / 1e-10
=========================  =====================================  ==========

``gcmi``, ``minepy``, ``Tigramite``, ``IDTxl`` and ``JIDT`` are GPL -- reference
only, never an oracle here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import panelary.depend as dp


def test_xi_matches_scipy_chatterjeexi() -> None:
    stats = pytest.importorskip("scipy.stats")
    if not hasattr(stats, "chatterjeexi"):
        pytest.skip("scipy.stats.chatterjeexi needs SciPy >= 1.15")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    y = np.sin(3 * x) + 0.3 * rng.standard_normal(500)
    res = stats.chatterjeexi(x, y)
    assert dp.xi(x, y) == pytest.approx(float(res.statistic), abs=1e-12)
    # Ties in y only: SciPy is deterministic there as well.
    yt = np.round(y, 1)
    assert dp.xi(x, yt) == pytest.approx(
        float(stats.chatterjeexi(x, yt).statistic), abs=1e-12
    )
    # Asymptotic p-value: ours is Theorem 2.1's N(0, 2/5), SciPy's with
    # y_continuous=True (its default uses the ties-aware tau^2 of eq. 2.2).
    ref_p = float(stats.chatterjeexi(x, y, y_continuous=True).pvalue)
    assert dp.xi_pvalue(dp.xi(x, y), 500) == pytest.approx(ref_p, rel=1e-6, abs=1e-300)
    z = rng.standard_normal(500)
    ref_null = float(stats.chatterjeexi(x, z, y_continuous=True).pvalue)
    assert dp.xi_pvalue(dp.xi(x, z), 500) == pytest.approx(ref_null, rel=1e-9)


def test_hoeffding_hand_cases() -> None:
    """Small-n values computed by hand from the definition (Hollander & Wolfe)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert dp.hoeffding_d(x, x) == pytest.approx(1.0)
    assert dp.hoeffding_d(x, -x) == pytest.approx(1.0)  # D is sign-blind
    # One adjacent swap, no ties: Q_i = 1 + #{x_j < x_i, y_j < y_i} = [1, 2, 2, 4, 5]
    # by direct count, and the ranks R, S are x and y themselves.
    y = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
    q = np.array([1, 2, 2, 4, 5], dtype=float)
    r, s = x, y
    d1 = ((q - 1) * (q - 2)).sum()
    d2 = ((r - 1) * (r - 2) * (s - 1) * (s - 2)).sum()
    d3 = ((r - 2) * (s - 2) * (q - 1)).sum()
    n = 5
    ref = (
        30
        * ((n - 2) * (n - 3) * d1 + d2 - 2 * (n - 2) * d3)
        / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    )
    assert dp.hoeffding_d(x, y) == pytest.approx(ref, abs=1e-12)


def test_dcor_package() -> None:
    dcor_pkg = pytest.importorskip("dcor")
    rng = np.random.default_rng(5)
    x = rng.standard_normal(200)
    y = x**2 + rng.standard_normal(200)
    assert dp.dcov2(x, y).dcor == pytest.approx(
        dcor_pkg.u_distance_correlation_sqr(x, y), rel=1e-8
    )
    z = rng.standard_normal(200)
    assert dp.partial_dcor(x, y, z) == pytest.approx(
        dcor_pkg.partial_distance_correlation(x, y, z), rel=1e-6
    )


def test_hyppo_dcorr_and_hsic() -> None:
    hyppo = pytest.importorskip("hyppo.independence")
    rng = np.random.default_rng(6)
    x = rng.standard_normal((300, 1))
    y = x**2 + 0.5 * rng.standard_normal((300, 1))
    ref = hyppo.Dcorr(bias=False).statistic(x, y)
    assert dp.dcov2(x, y).dcor == pytest.approx(ref, rel=1e-6)
    # RFF-HSIC is an approximation: agreement in direction and order of
    # magnitude only (hyppo uses the exact Gram matrix and its own bandwidth).
    assert dp.hsic(x, y).p_value < 1e-6 and hyppo.Hsic().test(x, y, reps=0)[1] < 1e-3


def test_mi_ksg_matches_sklearn() -> None:
    mi = pytest.importorskip("sklearn.feature_selection._mutual_info")
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1500)
    y = np.sin(x) + 0.5 * rng.standard_normal(1500)
    for k in (3, 5):
        assert dp.mi_ksg(x, y, k=k) == pytest.approx(
            mi._compute_mi_cc(x, y, k), abs=1e-10
        )


def test_mi_ksg_matches_ennemi() -> None:
    ennemi = pytest.importorskip("ennemi")
    rng = np.random.default_rng(8)
    x = rng.standard_normal(1000)
    y = 0.5 * x + rng.standard_normal(1000)
    assert dp.mi_ksg(x, y, k=3) == pytest.approx(
        float(ennemi.estimate_mi(y, x, k=3)), abs=1e-6
    )


def test_psi_known_values_and_scipy() -> None:
    euler = 0.5772156649015329
    assert dp.psi(1.0) == pytest.approx(-euler, abs=1e-12)
    assert dp.psi(0.5) == pytest.approx(-euler - 2 * math.log(2), abs=1e-12)
    special = pytest.importorskip("scipy.special")
    xs = np.geomspace(1e-3, 1e5, 300)
    np.testing.assert_allclose(dp.psi(xs), special.digamma(xs), rtol=0, atol=1e-10)
