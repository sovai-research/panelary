"""Shared plumbing: ``_internal._special`` (special functions) and ``_internal._jit``.

The special functions are checked against closed forms and, where scipy is
installed, against scipy. The old private locations must re-export the very
same objects so existing callers stay bit-identical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary._internal import _jit, _special


# --------------------------------------------------------------------------- #
# _special
# --------------------------------------------------------------------------- #
def test_old_locations_reexport_the_shared_objects() -> None:
    from panelary.depend import _info
    from panelary.depend import _special as dep_special
    from panelary.econ.features import _common
    from panelary.validation import _forecast_tests

    assert dep_special.norm_sf is _special.norm_sf
    assert _common.norm_cdf is _special.norm_cdf
    assert _common.norm_sf is _special.norm_sf
    assert _common.norm_ppf is _special.norm_ppf
    assert _info.psi is _special.psi
    assert _forecast_tests._betainc is _special._betainc_scalar
    assert _forecast_tests._t_sf is _special._t_sf_scalar


def test_normal_closed_forms_and_tails() -> None:
    assert _special.norm_cdf(0.0) == 0.5
    assert _special.norm_sf(0.0) == 0.5
    assert _special.norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    # erfc keeps the far tail: 1 - cdf would underflow to 0 here.
    assert 0.0 < _special.norm_sf(30.0) < 1e-190
    p = np.array([1e-12, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-12])
    x = np.asarray(_special.norm_ppf(p))
    np.testing.assert_allclose(np.asarray(_special.norm_cdf(x)), p, rtol=1e-12)
    assert _special.norm_ppf(0.0) == -np.inf
    assert _special.norm_ppf(1.0) == np.inf
    assert math.isnan(_special.norm_ppf(1.5))


def test_gamma_family_closed_forms() -> None:
    euler = 0.5772156649015329
    assert _special.psi(1.0) == pytest.approx(-euler, abs=1e-13)
    assert _special.psi(0.5) == pytest.approx(-euler - 2 * math.log(2), abs=1e-13)
    assert _special.trigamma(1.0) == pytest.approx(math.pi**2 / 6, rel=1e-14)
    assert _special.trigamma(0.5) == pytest.approx(math.pi**2 / 2, rel=1e-14)
    # recurrence psi_1(x) = psi_1(x + 1) + 1/x^2 across the switch at 10
    xs = np.array([0.1, 3.7, 9.5, 10.0, 25.0, 1e4])
    lhs = np.asarray(_special.trigamma(xs))
    rhs = np.asarray(_special.trigamma(xs + 1.0)) + 1.0 / xs**2
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13)
    assert math.isnan(_special.trigamma(-1.0))
    assert _special.lgamma(5.0) == pytest.approx(math.log(24.0))
    np.testing.assert_allclose(
        np.asarray(_special.lgamma(np.array([1.0, 5.0]))), [0.0, math.log(24.0)]
    )


def test_student_t_symmetry_and_broadcasting() -> None:
    assert _special.t_sf(0.0, 5.0) == 0.5
    assert _special.t_cdf(-2.0, 7.0) == _special.t_sf(2.0, 7.0)
    t = np.array([-3.0, -0.5, 0.0, 1.0, 40.0])
    out = np.asarray(_special.t_sf(t, 4.0))
    assert out.shape == t.shape
    np.testing.assert_allclose(
        out + np.asarray(_special.t_cdf(t, 4.0)), 1.0, rtol=1e-14
    )
    assert math.isnan(_special.t_sf(float("nan"), 3.0))
    with pytest.raises(ValueError):
        _special.t_sf(1.0, 0.0)
    # t with 1 d.o.f. is Cauchy: P(T > 1) = 1/4
    assert _special.t_sf(1.0, 1.0) == pytest.approx(0.25, rel=1e-13)
    # far lower tail keeps relative accuracy (no 1 - sf cancellation)
    assert _special.t_cdf(-50.0, 3.0) > 0.0


def test_betainc_closed_forms() -> None:
    # I_x(1, 1) = x ; I_x(a, 1) = x^a
    xs = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(
        np.asarray(_special.betainc(1.0, 1.0, xs)), xs, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(_special.betainc(3.0, 1.0, xs)), xs**3, rtol=1e-13, atol=1e-15
    )
    assert isinstance(_special.betainc(2.0, 3.0, 0.4), float)


def test_against_scipy_when_installed() -> None:
    sp = pytest.importorskip("scipy.special")
    st = pytest.importorskip("scipy.stats")
    x = np.linspace(0.05, 30.0, 97)
    np.testing.assert_allclose(np.asarray(_special.psi(x)), sp.digamma(x), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(_special.trigamma(x)), sp.polygamma(1, x), rtol=1e-12
    )
    t = np.linspace(-8.0, 8.0, 41)
    for df in (1.0, 2.5, 10.0, 200.0):
        np.testing.assert_allclose(
            np.asarray(_special.t_sf(t, df)), st.t.sf(t, df), rtol=1e-10, atol=1e-300
        )
    a, b = np.meshgrid([0.5, 2.0, 30.0], [0.5, 4.0, 50.0])
    np.testing.assert_allclose(
        np.asarray(_special.betainc(a, b, 0.37)), sp.betainc(a, b, 0.37), rtol=1e-11
    )


# --------------------------------------------------------------------------- #
# _jit
# --------------------------------------------------------------------------- #
def _cumsum_loop(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.shape[0]):
        acc += x[i]
        out[i] = acc
    return out


_CUMSUM = _jit.lazy_njit(_cumsum_loop)


def test_lazy_kernel_falls_back_without_numba() -> None:
    x = np.arange(5, dtype=np.float64)
    with _jit.force_numpy():
        assert _jit.numba_available() is False
        assert _CUMSUM.compiled() is None
        np.testing.assert_array_equal(_CUMSUM(x), np.cumsum(x))
        with _jit.force_numpy():  # nests
            assert _CUMSUM.compiled() is None
        assert _CUMSUM.compiled() is None


def test_lazy_kernel_bitwise_parity_with_numba() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000)
    kernel = _CUMSUM.compiled()
    assert kernel is not None
    assert _CUMSUM.compiled() is kernel  # compiled once, cached
    _jit.assert_backend_parity(kernel(x), _cumsum_loop(x))


def test_fastmath_is_rejected() -> None:
    with pytest.raises(ValueError, match="fastmath"):
        _jit.LazyKernel(_cumsum_loop, fastmath=True)


def test_assert_backend_parity() -> None:
    a = np.array([1.0, np.nan, 3.0])
    _jit.assert_backend_parity(a, a.copy())
    _jit.assert_backend_parity((a, np.arange(3)), (a.copy(), np.arange(3)))
    with pytest.raises(AssertionError, match="bitwise"):
        _jit.assert_backend_parity(a, a + 1e-16 * np.array([0.0, 0.0, 1e3]))
    with pytest.raises(AssertionError, match="dtype"):
        _jit.assert_backend_parity(np.arange(3), np.arange(3.0))
    with pytest.raises(AssertionError, match="bitwise"):
        _jit.assert_backend_parity(np.arange(3), np.array([0, 1, 5]))
    _jit.assert_backend_parity(a, a + 1e-14, rtol=1e-12)
