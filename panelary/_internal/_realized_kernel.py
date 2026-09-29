"""Realized-kernel and pre-averaging building blocks for noisy intraday prices.

One home for the pieces of the noise-robust integrated-variance estimators that
more than one estimator needs: the Parzen kernel and its weights, the
Barndorff-Nielsen, Hansen, Lunde & Shephard (BNHLS) bandwidth rule, end-point
jittering, and the pre-averaging constants of Jacod, Li, Mykland, Podolskij &
Vetter (JLMPV 2009). :mod:`panelary.econ.features._realized` builds the
univariate daily measures on top of it; a multivariate realized kernel would
import the same weights and bandwidth rule rather than re-derive them.

Verified constants (2026-09-29; see ``tests/test_internal_realized_kernel.py``)
-------------------------------------------------------------------------------
* ``c* = (k''(0)^2 / k^{0,0})^(1/5)`` with ``k''(0) = -12`` and
  ``k^{0,0} = int_0^1 k(x)^2 dx = 151/560 = 0.269642...`` for the Parzen kernel
  (derived in closed form, confirmed by quadrature). That gives
  ``c* = 3.51168``. The published BNHLS (2009) value **3.5134** is the same
  formula evaluated with ``k^{0,0}`` rounded to 0.269; it is what the literature
  and reference code use, so it is the default here (:data:`PARZEN_C_STAR`),
  with the exact value kept as :data:`PARZEN_C_STAR_EXACT`. The two bandwidths
  differ by 0.05%, which the integer ceiling almost always absorbs.
* For ``g(x) = min(x, 1 - x)`` and even ``k``: ``psi_1^k = 1`` exactly and
  ``psi_2^k = 1/12 + 1/(6 k^2)`` (derived; matches the direct sums to 1e-16).

Leaf module: numpy and polars only.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

__all__ = [
    "BNHLS_DENSE_RETURNS",
    "BNHLS_SPARSE_RETURNS",
    "PARZEN_C_STAR",
    "PARZEN_C_STAR_EXACT",
    "bnhls_bandwidth",
    "bnhls_bandwidth_expr",
    "jitter_returns",
    "parzen",
    "parzen_expr",
    "parzen_weights",
    "pre_averaged_returns",
    "preaverage_k",
    "preaverage_psi",
    "preaverage_psi_expr",
    "realized_kernel",
]

#: BNHLS (2009) optimal-bandwidth constant for the Parzen kernel, as published:
#: ``(k''(0)^2 / 0.269)^(1/5)`` with ``k''(0)^2 = 144``.
PARZEN_C_STAR: float = 3.5134

#: The same constant with ``int_0^1 k^2 = 151/560`` unrounded.
PARZEN_C_STAR_EXACT: float = (144.0 * 560.0 / 151.0) ** 0.2

#: BNHLS (2009) estimate the noise variance from a "dense" grid of about
#: 2-minute returns and the integrated variance from a "sparse" 20-minute grid.
#: Expressed per session so the rule needs no clock units: a 390-minute equity
#: session has 195 two-minute and 19.5 twenty-minute returns. The grid step for
#: a session with ``n`` returns is ``max(1, round(n / BNHLS_*_RETURNS))``.
BNHLS_DENSE_RETURNS: float = 195.0
BNHLS_SPARSE_RETURNS: float = 19.5


# --------------------------------------------------------------------------- #
# Parzen kernel
# --------------------------------------------------------------------------- #
def parzen(x: np.ndarray | float) -> np.ndarray:
    """The Parzen kernel ``k(x)`` on ``x >= 0`` (zero beyond 1).

    ``k(x) = 1 - 6x^2 + 6x^3`` on ``[0, 1/2]``, ``2(1 - x)^3`` on ``[1/2, 1]``.
    Its Fourier transform is non-negative, which is what makes the
    non-flat-top realized kernel built on it non-negative.
    """
    u = np.abs(np.asarray(x, dtype=np.float64))
    inner = 1.0 - 6.0 * u**2 + 6.0 * u**3
    outer = 2.0 * (1.0 - u) ** 3
    return np.where(u <= 0.5, inner, np.where(u <= 1.0, outer, 0.0))


def parzen_expr(x: pl.Expr) -> pl.Expr:
    """:func:`parzen` as a polars expression (``x >= 0``; zero beyond 1)."""
    return (
        pl.when(x <= 0.5)
        .then(1.0 - 6.0 * x**2 + 6.0 * x**3)
        .when(x <= 1.0)
        .then(2.0 * (1.0 - x) ** 3)
        .otherwise(0.0)
    )


def parzen_weights(bandwidth: int) -> np.ndarray:
    """Non-flat-top kernel weights ``k(h / (H + 1))`` for lags ``h = 1..H``.

    The realized kernel is ``gamma_0 + 2 * sum_h w_h * gamma_h`` (BNHLS 2009,
    2011). ``H = 0`` returns an empty array, i.e. the kernel reduces to the
    realized variance of the (jittered) returns.
    """
    h_max = int(bandwidth)
    if h_max < 0:
        raise ValueError(f"`bandwidth` must be >= 0, got {bandwidth!r}.")
    h = np.arange(1, h_max + 1, dtype=np.float64)
    return parzen(h / (h_max + 1.0))


def bnhls_bandwidth(
    xi2: np.ndarray | float, n: np.ndarray | float, *, c_star: float = PARZEN_C_STAR
) -> np.ndarray:
    """Real-valued BNHLS (2009) bandwidth ``H* = c* xi^(4/5) n^(3/5)``.

    ``xi2`` is the noise-to-signal ratio ``omega^2 / IV`` and ``n`` the number of
    (jittered) returns. Callers take the integer bandwidth as ``ceil(H*)``.
    """
    xi2_arr = np.asarray(xi2, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    return c_star * xi2_arr**0.4 * n_arr**0.6


def bnhls_bandwidth_expr(
    xi2: pl.Expr, n: pl.Expr, *, c_star: float = PARZEN_C_STAR
) -> pl.Expr:
    """:func:`bnhls_bandwidth` as a polars expression (real-valued ``H*``)."""
    return c_star * xi2.pow(0.4) * n.cast(pl.Float64).pow(0.6)


def jitter_returns(returns: np.ndarray, m: int = 2) -> np.ndarray:
    """Returns after BNHLS (2009) end-point jittering over ``m`` prices.

    The first and last prices of the session are replaced by the average of the
    first and last ``m`` prices, which removes the end effects that would
    otherwise bias the kernel. From ``N`` returns this leaves ``N - 2(m - 1)``;
    with ``m = 2`` they are ``[r_1/2 + r_2, r_3, ..., r_{N-2}, r_{N-1} + r_N/2]``.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    m = int(m)
    if m < 1:
        raise ValueError(f"`m` must be >= 1, got {m!r}.")
    n_ret = r.shape[0]
    if n_ret < 2 * m - 1:
        return np.empty(0, dtype=np.float64)
    p = np.concatenate([[0.0], np.cumsum(r)])
    x0 = p[:m].mean()
    xn = p[n_ret - m + 1 :].mean()
    interior = p[m : n_ret - m + 1]
    levels = np.concatenate([[x0], interior, [xn]])
    return np.diff(levels)


def realized_kernel(
    returns: np.ndarray, bandwidth: int, *, jitter: bool = True
) -> float:
    """Univariate Parzen realized kernel of one session (numpy reference).

    ``K = gamma_0 + 2 * sum_{h=1}^{H} k(h / (H + 1)) gamma_h`` with
    ``gamma_h = sum_j x_j x_{j-h}`` over the (optionally jittered, ``m = 2``)
    returns ``x``. Used as the oracle for the vectorised daily measure.
    """
    x = jitter_returns(returns, 2) if jitter else np.asarray(returns, dtype=np.float64)
    if x.shape[0] == 0:
        return float("nan")
    total = float(x @ x)
    for h, w in enumerate(parzen_weights(bandwidth), start=1):
        if h >= x.shape[0]:
            break
        total += 2.0 * float(w) * float(x[h:] @ x[:-h])
    return total


# --------------------------------------------------------------------------- #
# Pre-averaging (JLMPV 2009) with g(x) = min(x, 1 - x)
# --------------------------------------------------------------------------- #
def preaverage_k(n: int, theta: float) -> int:
    """The even pre-averaging window ``k = 2 * ceil(theta * sqrt(n) / 2)``."""
    return int(2 * math.ceil(theta * math.sqrt(n) / 2.0))


def preaverage_psi(k: int) -> tuple[float, float]:
    """``(psi_1^k, psi_2^k)`` for ``g(x) = min(x, 1 - x)`` by their defining sums.

    ``psi_1^k = k * sum_{j=0}^{k-1} (g((j+1)/k) - g(j/k))^2`` and
    ``psi_2^k = (1/k) * sum_{j=1}^{k-1} g(j/k)^2``.
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"`k` must be >= 2, got {k!r}.")
    j = np.arange(k + 1, dtype=np.float64) / k
    g = np.minimum(j, 1.0 - j)
    psi1 = k * float(np.sum(np.diff(g) ** 2))
    psi2 = float(np.sum(g[1:k] ** 2)) / k
    return psi1, psi2


def preaverage_psi_expr(k: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Closed-form ``(psi_1^k, psi_2^k)`` for even ``k`` as polars expressions.

    ``psi_1^k = 1`` and ``psi_2^k = 1/12 + 1/(6 k^2)``; see :func:`preaverage_psi`
    for the defining sums they equal.
    """
    kf = k.cast(pl.Float64)
    return pl.lit(1.0), 1.0 / 12.0 + 1.0 / (6.0 * kf * kf)


def pre_averaged_returns(returns: np.ndarray, k: int) -> np.ndarray:
    """Pre-averaged returns ``Ybar_i = sum_{j=1}^{k-1} g(j/k) r_{i+j}`` (direct).

    The numpy oracle for the O(n) double-box-filter form used by the daily
    measure; returns the ``n - k + 2`` values ``i = 0 .. n - k + 1``.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    k = int(k)
    j = np.arange(1, k, dtype=np.float64) / k
    g = np.minimum(j, 1.0 - j)
    n_ret = r.shape[0]
    if n_ret < k - 1:
        return np.empty(0, dtype=np.float64)
    # window i covers r_{i+1} .. r_{i+k-1}, i.e. 0-based positions i .. i+k-2.
    windows = np.lib.stride_tricks.sliding_window_view(r, k - 1)
    return windows @ g
