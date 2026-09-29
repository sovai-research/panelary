"""Scipy-free special functions, in one place.

Panelary grew several private copies of the same few functions (normal tails,
digamma, the regularised incomplete beta, Student-t tails). New code imports
them from here; the old locations re-export these objects, so every existing
caller gets bit-identical results.

Normal tails go through ``math.erfc``: ``1 - Phi(z)`` underflows to exactly 0
near ``z ~ 8``, while ``erfc`` stays accurate to ``z ~ 38`` (``p ~ 1e-316``).
A p-value of 0 outranks every honest small p-value in a screen, so this is a
correctness issue, not a cosmetic one.

Known copy left in place: ``panelary.econ._common`` (``_betacf`` / ``t_sf`` /
``t_cdf``) differs in its continued-fraction stopping rule, so pointing it here
would move econ outputs in the last bits. Consolidate it only with a golden
test.

Leaf module: standard library and numpy only.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "betainc",
    "lgamma",
    "norm_cdf",
    "norm_pdf",
    "norm_ppf",
    "norm_sf",
    "psi",
    "t_cdf",
    "t_sf",
    "trigamma",
]

_ERFC = np.frompyfunc(math.erfc, 1, 1)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Normal distribution
# --------------------------------------------------------------------------- #
# Acklam's rational approximation to the inverse normal CDF (|eps| < 1.15e-9),
# refined by one Halley step against `norm_cdf` for full double precision.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal density."""
    arr = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * arr * arr) / _SQRT_2PI
    return float(out) if out.ndim == 0 else out


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal CDF, evaluated exactly via ``erfc`` (no scipy)."""
    arr = np.asarray(x, dtype=float)
    out = 0.5 * np.asarray(_ERFC(-arr / math.sqrt(2.0)), dtype=float)
    return float(out) if out.ndim == 0 else out


def norm_sf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal survival function ``P(Z >= x)``, accurate far into the tail."""
    arr = np.asarray(x, dtype=float)
    out = 0.5 * np.asarray(_ERFC(arr / math.sqrt(2.0)), dtype=float)
    return float(out) if out.ndim == 0 else out


def _norm_ppf_scalar(p: float) -> float:
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        return float("nan")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    # One Halley refinement step.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return float(x - u / (1.0 + x * u / 2.0))


_NORM_PPF = np.frompyfunc(_norm_ppf_scalar, 1, 1)


def norm_ppf(p: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal quantile function (inverse CDF), scipy-free."""
    arr = np.asarray(p, dtype=float)
    out = np.asarray(_NORM_PPF(arr), dtype=float)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# Gamma family
# --------------------------------------------------------------------------- #
_LGAMMA = np.vectorize(math.lgamma, otypes=[np.float64])


def lgamma(x: np.ndarray | float) -> np.ndarray | float:
    """``log|Gamma(x)|``, elementwise (``math.lgamma`` lifted to arrays)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return math.lgamma(float(arr))
    return _LGAMMA(arr)


def psi(x: float | np.ndarray) -> float | np.ndarray:
    """Digamma function ``psi(x) = d/dx log Gamma(x)`` for ``x > 0``.

    Upward recurrence ``psi(x) = psi(x + 1) - 1/x`` to ``x >= 10``, then the
    asymptotic Bernoulli series to ``x^-10`` (truncation error < 1e-13).
    Checked against ``psi(1) = -gamma`` and ``psi(1/2) = -gamma - 2 ln 2``.

    Parameters
    ----------
    x : float or ndarray
        Positive argument(s); non-positive values return ``nan``.

    Returns
    -------
    float or ndarray
    """
    arr = np.asarray(x, dtype=np.float64)
    xx = np.atleast_1d(arr).astype(np.float64).copy()
    bad = ~(xx > 0)
    xx[bad] = 1.0
    acc = np.zeros_like(xx)
    small = xx < 10.0
    while small.any():
        acc[small] -= 1.0 / xx[small]
        xx[small] += 1.0
        small = xx < 10.0
    f = 1.0 / (xx * xx)
    series = f * (
        1.0 / 12.0
        - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0 - f / 132.0)))
    )
    out = acc + np.log(xx) - 0.5 / xx - series
    out[bad] = np.nan
    if arr.ndim == 0:
        return float(out[0])
    return out.reshape(arr.shape)


def trigamma(x: float | np.ndarray) -> float | np.ndarray:
    """Trigamma function ``psi_1(x) = d^2/dx^2 log Gamma(x)`` for ``x > 0``.

    Upward recurrence ``psi_1(x) = psi_1(x + 1) + 1/x^2`` to ``x >= 10``, then
    the asymptotic series ``1/x + 1/(2x^2) + sum_k B_2k / x^(2k+1)`` through
    ``x^-13`` (truncation error below 1e-15 at ``x = 10``). Checked against
    ``psi_1(1) = pi^2/6`` and ``psi_1(1/2) = pi^2/2``. Non-positive values
    return ``nan``.
    """
    arr = np.asarray(x, dtype=np.float64)
    xx = np.atleast_1d(arr).astype(np.float64).copy()
    bad = ~(xx > 0)
    xx[bad] = 1.0
    acc = np.zeros_like(xx)
    small = xx < 10.0
    while small.any():
        acc[small] += 1.0 / (xx[small] * xx[small])
        xx[small] += 1.0
        small = xx < 10.0
    inv = 1.0 / xx
    f = inv * inv
    # B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730
    series = (
        inv
        + 0.5 * f
        + inv
        * f
        * (
            1.0 / 6.0
            - f
            * (
                1.0 / 30.0
                - f
                * (
                    1.0 / 42.0
                    - f * (1.0 / 30.0 - f * (5.0 / 66.0 - f * 691.0 / 2730.0))
                )
            )
        )
    )
    out = acc + series
    out[bad] = np.nan
    if arr.ndim == 0:
        return float(out[0])
    return out.reshape(arr.shape)


# --------------------------------------------------------------------------- #
# Regularised incomplete beta and Student-t tails
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc_scalar(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return (
        1.0
        - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x))
        * _betacf(b, a, 1.0 - x)
        / b
    )


def _t_sf_scalar(t: float, df: float) -> float:
    """Upper-tail probability ``P(T > t)`` for a Student-t with ``df`` d.o.f."""
    if df <= 0:
        raise ValueError(f"`df` must be positive, got {df}.")
    if not math.isfinite(t):
        if math.isnan(t):
            return float("nan")
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * _betainc_scalar(0.5 * df, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def _t_cdf_scalar(t: float, df: float) -> float:
    # Symmetry, not ``1 - sf``: keeps full relative accuracy in the lower tail.
    return _t_sf_scalar(-t, df)


_BETAINC_VEC = np.vectorize(_betainc_scalar, otypes=[np.float64])
_T_SF_VEC = np.vectorize(_t_sf_scalar, otypes=[np.float64])
_T_CDF_VEC = np.vectorize(_t_cdf_scalar, otypes=[np.float64])


def betainc(
    a: float | np.ndarray, b: float | np.ndarray, x: float | np.ndarray
) -> float | np.ndarray:
    """Regularised incomplete beta ``I_x(a, b)``, broadcasting over array inputs.

    Scalar inputs return a Python ``float``; array inputs return float64 arrays.
    """
    if np.ndim(a) == 0 and np.ndim(b) == 0 and np.ndim(x) == 0:
        return _betainc_scalar(float(a), float(b), float(x))
    return _BETAINC_VEC(a, b, x)


def t_sf(t: float | np.ndarray, df: float | np.ndarray) -> float | np.ndarray:
    """Upper tail ``P(T > t)`` of Student's t, broadcasting over array inputs.

    Raises ``ValueError`` for ``df <= 0``. ``nan`` in ``t`` gives ``nan``.
    """
    if np.ndim(t) == 0 and np.ndim(df) == 0:
        return _t_sf_scalar(float(t), float(df))
    return _T_SF_VEC(t, df)


def t_cdf(t: float | np.ndarray, df: float | np.ndarray) -> float | np.ndarray:
    """CDF ``P(T <= t)`` of Student's t, computed by symmetry (no cancellation)."""
    if np.ndim(t) == 0 and np.ndim(df) == 0:
        return _t_cdf_scalar(float(t), float(df))
    return _T_CDF_VEC(t, df)
