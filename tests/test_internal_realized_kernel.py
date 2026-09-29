"""The realized-kernel / pre-averaging leaf, and the constants it carries.

The plan marked ``c* = 3.5134`` and the pre-averaging normalisation as
not-yet-verified (†). This file is the verification record:

* ``c* = (k''(0)^2 / int_0^1 k^2)^(1/5)`` for the Parzen kernel, with
  ``int_0^1 k^2 = 151/560`` derived in closed form and confirmed by quadrature;
  the exact constant is 3.51168, and the published 3.5134 is the same formula
  with the integral rounded to 0.269.
* ``psi_1^k = 1`` and ``psi_2^k = 1/12 + 1/(6k^2)`` for ``g(x) = min(x, 1 - x)``;
  the pre-averaged return is a double box filter of the returns; and the
  normalisation ``E[Ybar^2] = k psi_2 IV/n + (psi_1/k) omega^2`` holds by
  simulation (the unbiasedness of the daily ``pav`` is in
  ``tests/test_realized_accuracy.py``).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary._internal._realized_kernel import (
    PARZEN_C_STAR,
    PARZEN_C_STAR_EXACT,
    bnhls_bandwidth,
    bnhls_bandwidth_expr,
    jitter_returns,
    parzen,
    parzen_expr,
    parzen_weights,
    pre_averaged_returns,
    preaverage_k,
    preaverage_psi,
    preaverage_psi_expr,
    realized_kernel,
)


def test_parzen_kernel_values() -> None:
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    want = [1.0, 1 - 6 / 16 + 6 / 64, 0.25, 2 * 0.25**3, 0.0, 0.0]
    np.testing.assert_allclose(parzen(x), want, atol=1e-15)
    # Both branches meet at 1/2 with value 1/4 and slope -3/2.
    eps = 1e-6
    left = (parzen(0.5) - parzen(0.5 - eps)) / eps
    right = (parzen(0.5 + eps) - parzen(0.5)) / eps
    assert left == pytest.approx(-1.5, abs=1e-5)
    assert right == pytest.approx(-1.5, abs=1e-5)
    grid = np.linspace(0.0, 1.2, 97)
    got = pl.select(parzen_expr(pl.lit(pl.Series(grid)))).to_series().to_numpy()
    np.testing.assert_allclose(got, parzen(grid), atol=0, rtol=1e-15)


def test_c_star_is_derived_not_copied() -> None:
    u = np.linspace(0.0, 1.0, 2_000_001)
    integral = np.trapezoid(parzen(u) ** 2, u)
    assert integral == pytest.approx(151.0 / 560.0, abs=1e-12)
    # k''(0) = -12 for k(x) = 1 - 6x^2 + 6x^3.
    assert pytest.approx((144.0 / integral) ** 0.2, rel=1e-10) == PARZEN_C_STAR_EXACT
    assert pytest.approx(3.51168, abs=1e-5) == PARZEN_C_STAR_EXACT
    # The published constant is the same formula with the integral rounded.
    assert round((144.0 / 0.269) ** 0.2, 4) == PARZEN_C_STAR


def test_parzen_weights_and_bandwidth() -> None:
    np.testing.assert_allclose(parzen_weights(3), parzen(np.array([1, 2, 3]) / 4.0))
    assert parzen_weights(0).shape == (0,)
    with pytest.raises(ValueError):
        parzen_weights(-1)
    h = bnhls_bandwidth(1e-3, 390)
    assert h == pytest.approx(3.5134 * 1e-3**0.4 * 390**0.6)
    got = pl.select(bnhls_bandwidth_expr(pl.lit(1e-3), pl.lit(390))).item()
    assert got == pytest.approx(float(h), rel=1e-15)


def test_jittering_averages_the_end_prices() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    got = jitter_returns(r, 2)
    np.testing.assert_allclose(got, [0.5 + 2.0, 3.0, 4.0, 5.0 + 3.0])
    # The jittered returns span from the mean of the first two prices to the
    # mean of the last two.
    p = np.r_[0.0, np.cumsum(r)]
    assert got.sum() == pytest.approx(p[-2:].mean() - p[:2].mean())
    assert jitter_returns(r[:2], 2).shape == (0,)
    np.testing.assert_array_equal(jitter_returns(r, 1), r)


def test_the_non_flat_top_parzen_kernel_is_never_negative() -> None:
    """Parzen's Fourier transform is non-negative, so the kernel is a PSD form,
    including on adversarial, strongly negatively autocorrelated returns."""
    rng = np.random.default_rng(0)
    worst = np.inf
    for _ in range(300):
        n = int(rng.integers(10, 200))
        r = rng.standard_normal(n)
        r = r - 0.95 * np.r_[0.0, r[:-1]] if rng.random() < 0.5 else r
        alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * np.abs(r)
        for series in (r, alternating):
            for h in (1, 3, 10, 40):
                worst = min(worst, realized_kernel(series, h) / float(series @ series))
    assert worst >= -1e-12


def test_realized_kernel_reduces_to_rv_without_jitter_or_lags() -> None:
    r = np.random.default_rng(1).standard_normal(50)
    assert realized_kernel(r, 0, jitter=False) == pytest.approx(float(r @ r))


@pytest.mark.parametrize("k", [2, 4, 10, 20, 70])
def test_preaveraging_psi_closed_forms(k: int) -> None:
    psi1, psi2 = preaverage_psi(k)
    assert psi1 == pytest.approx(1.0, abs=1e-14)
    assert psi2 == pytest.approx(1.0 / 12.0 + 1.0 / (6.0 * k * k), abs=1e-15)
    e1, e2 = preaverage_psi_expr(pl.lit(k))
    assert pl.select(e1).item() == 1.0
    assert pl.select(e2).item() == pytest.approx(psi2, abs=1e-15)


def test_preaverage_k_is_even() -> None:
    assert preaverage_k(390, 1.0) == 20
    assert preaverage_k(4680, 1.0) == 70
    assert all(preaverage_k(n, 0.7) % 2 == 0 for n in range(5, 500, 7))
    with pytest.raises(ValueError):
        preaverage_psi(1)


@pytest.mark.parametrize("k", [2, 4, 8, 20])
def test_preaveraged_return_is_a_double_box_filter(k: int) -> None:
    """``k Ybar = S_{k/2}(S_{k/2}(r))``, the O(n) form the daily measure uses."""
    r = np.random.default_rng(k).standard_normal(200) * 1e-3
    direct = pre_averaged_returns(r, k)
    boxed = pl.Series(r).rolling_sum(k // 2).rolling_sum(k // 2).to_numpy()[k - 2 :] / k
    assert direct.shape == boxed.shape == (200 - k + 2,)
    # Mixed-sign sums: the error is relative to the returns, not to Ybar.
    np.testing.assert_allclose(boxed, direct, rtol=0, atol=1e-15 * np.abs(r).max())


def test_preaveraging_normalisation_by_simulation() -> None:
    """``E[Ybar^2] = k psi_2 IV/n + (psi_1/k) omega^2`` (the derivation behind
    the daily ``pav``), checked on 4 000 simulated noisy sessions."""
    rng = np.random.default_rng(2)
    n, k, iv, omega2 = 400, 20, 1.0, 0.004
    x = np.cumsum(rng.standard_normal((4000, n + 1)) * np.sqrt(iv / n), axis=1)
    y = x + rng.standard_normal(x.shape) * np.sqrt(omega2)
    r = np.diff(y, axis=1)
    ybar2 = np.mean([np.mean(pre_averaged_returns(row, k) ** 2) for row in r])
    psi1, psi2 = preaverage_psi(k)
    want = k * psi2 * iv / n + psi1 / k * omega2
    assert ybar2 == pytest.approx(want, rel=0.02)
