"""Nonlinear shrinkage: one Ledoit--Peche engine, two smoothers (plan 5.7).

Both estimators keep the sample eigenvectors and map the sample eigenvalues
``lambda_i`` to ``d_i``, estimating the Ledoit & Peche (2011) oracle
``u_i^T Sigma u_i``. A map takes the non-null eigenvalues, the effective
sample size ``n`` and the dimension ``p`` and returns ``(d0, d)``: the value
given to every null direction and the shrunk non-null eigenvalues. It is
``O(m^2)`` in ``m = min(p, n)`` and chunked so that no ``m x m`` temporary
exceeds ~32 MB.

* :func:`qis_map` -- Quadratic-Inverse Shrinkage, Ledoit & Wolf (2022),
  *Bernoulli* 28(3). A rational kernel: a 1e-14 relative perturbation of the
  eigenvalues moves the output by ~1e-13. **The default.**
* :func:`lw2020_map` -- analytical nonlinear shrinkage, Ledoit & Wolf (2020),
  *Annals of Statistics* 48(5): an Epanechnikov kernel and its Hilbert
  transform. Evaluated literally, the transform cancels catastrophically at
  large ``|x|`` and a 1e-14 perturbation of one eigenvalue moved the top
  shrunk eigenvalue by up to 8.6e-7; summing it as a series there (see
  :func:`_epanechnikov_hilbert`) brings that to ~7e-14, on par with QIS.

Both are clean-room from the papers (QIS checked against fixtures produced by
the authors' BSD-2 reference code; LW 2020 against a dense port of the
published formulas).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from panelary.covariance._types import CovEstimate, WindowStats

__all__ = ["lw2020", "lw2020_map", "nonlinear_estimate", "qis", "qis_map"]

#: Rows of the ``m x m`` kernel evaluated at once (``1024 x m`` doubles, a few
#: temporaries alive): bounds memory without changing a single bit of output,
#: because each row's mean is reduced independently.
_CHUNK = 1024


def _nonnull(values: NDArray[np.float64], p: int, n: float) -> NDArray[np.float64]:
    """The ``m = min(p, n, rank)`` largest eigenvalues, **ascending**."""
    m = min(p, int(math.floor(n + 1e-9)), int(np.count_nonzero(values > 0.0)))
    if m < 1:
        raise ValueError("nonlinear shrinkage needs at least one non-null eigenvalue.")
    return np.ascontiguousarray(values[:m][::-1])


def qis_map(
    values: NDArray[np.float64], n: float, p: int
) -> tuple[float, NDArray[np.float64]]:
    """Quadratic-Inverse Shrinkage of a spectrum (Ledoit & Wolf 2022).

    Parameters
    ----------
    values : ndarray
        Sample eigenvalues, **descending** (zeros allowed at the end).
    n : float
        Effective sample size (``T - 1`` after demeaning).
    p : int
        Dimension.

    Returns
    -------
    (d0, d)
        ``d0`` for each of the ``p - m`` null directions (``nan`` when there
        are none) and ``d`` (``m``,) for the non-null ones, **descending**,
        trace-preserving over all ``p`` directions.

    Notes
    -----
    With ``c = p / n``, ``h = min(c^2, c^-2)^0.35 / p^0.35`` and
    ``iota = 1 / lambda``::

        theta_i  = mean_j iota_j (iota_j - iota_i) / ((iota_j - iota_i)^2 + h^2 iota_j^2)
        Htheta_i = mean_j h iota_j^2 / ((iota_j - iota_i)^2 + h^2 iota_j^2)
        A_i      = theta_i^2 + Htheta_i^2
        p <= n:  d_i = 1 / ((1-c)^2 iota_i + 2c(1-c) iota_i theta_i + c^2 iota_i A_i)
        p >  n:  d0 = 1 / ((c-1) mean(iota)),  d_i = 1 / (iota_i A_i)

    A rank-deficient window with ``p <= n`` (duplicate columns) is mapped with
    the ``p > n`` branch at ``c = p / m``, so the null directions stay
    positive.
    """
    lam = _nonnull(values, p, n)
    m = lam.size
    c = p / n
    h = min(c * c, 1.0 / (c * c)) ** 0.35 / p**0.35
    iota = 1.0 / lam
    theta = np.empty(m)
    htheta = np.empty(m)
    h2 = h * h
    for lo in range(0, m, _CHUNK):
        hi = min(lo + _CHUNK, m)
        ii = iota[lo:hi, None]  # output index i (rows)
        diff = iota[None, :] - ii  # iota_j - iota_i, j along the row
        den = diff * diff + h2 * (iota * iota)[None, :]
        theta[lo:hi] = np.mean(iota[None, :] * diff / den, axis=1)
        htheta[lo:hi] = np.mean(h * (iota * iota)[None, :] / den, axis=1)
    A = theta * theta + htheta * htheta
    full_rank = m == p
    if full_rank and c <= 1.0:
        d = 1.0 / (
            (1 - c) ** 2 * iota + 2 * c * (1 - c) * iota * theta + c * c * iota * A
        )
        d0 = math.nan
        total = float(d.sum())
    else:
        c_null = c if c > 1.0 else p / m
        d0 = 1.0 / ((c_null - 1.0) * float(iota.mean()))
        d = 1.0 / (iota * A)
        total = float(d.sum()) + (p - m) * d0
    trace = float(values.sum())
    k = trace / total
    d = d * k
    if not math.isnan(d0):
        d0 *= k
    return float(d0), np.ascontiguousarray(d[::-1])


_SQRT5 = math.sqrt(5.0)

#: Beyond this ``|x|`` the Epanechnikov Hilbert transform is summed as a
#: series in ``u = sqrt(5) / x`` (``|u| <= 0.448``; 24 terms reach 1e-17).
_HILBERT_SERIES_X = 5.0
_HILBERT_SERIES_TERMS = 24


def _epanechnikov_hilbert(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hilbert transform of the Epanechnikov kernel (LW 2020 eq. 4.8).

    ``-3x/(10 pi) + 3/(4 sqrt5 pi) (1 - x^2/5) log|(sqrt5 - x)/(sqrt5 + x)|``.
    For large ``|x|`` the two terms are each ``~ 0.1 |x|`` and cancel to
    ``O(1/x)``: evaluated literally, a 1e-14 relative change in one eigenvalue
    moved the largest shrunk eigenvalue by 8.6e-7 (measured, ``n = 200``,
    ``p = 80``). There the exact series in ``u = sqrt5 / x`` is used instead::

        H(x) = -3/(sqrt5 pi) * sum_k u^(2k+1) / ((2k+1)(2k+3))

    obtained by expanding ``log((1-u)/(1+u)) = -2 artanh(u)``; the ``O(x)``
    terms cancel analytically. At ``|x| = sqrt5`` the closed-form limit
    ``-3x/(10 pi)`` is used.
    """
    out = np.empty_like(x)
    ax = np.abs(x)
    far = ax >= _HILBERT_SERIES_X
    near = ~far
    if near.any():
        xn = x[near]
        with np.errstate(divide="ignore", invalid="ignore"):
            logt = np.log(np.abs(_SQRT5 - xn)) - np.log(np.abs(_SQRT5 + xn))
            hn = (-3.0 / 10.0 / math.pi) * xn + (3.0 / 4.0 / _SQRT5 / math.pi) * (
                1.0 - xn * xn / 5.0
            ) * logt
        edge = np.abs(xn) == _SQRT5
        if edge.any():
            hn[edge] = (-3.0 / 10.0 / math.pi) * xn[edge]
        out[near] = hn
    if far.any():
        u = _SQRT5 / x[far]
        u2 = u * u
        term = u.copy()
        acc = np.zeros_like(u)
        for k in range(_HILBERT_SERIES_TERMS):
            acc += term / ((2 * k + 1) * (2 * k + 3))
            term = term * u2
        out[far] = (-3.0 / _SQRT5 / math.pi) * acc
    return out


def lw2020_map(
    values: NDArray[np.float64], n: float, p: int
) -> tuple[float, NDArray[np.float64]]:
    """Analytical nonlinear shrinkage of a spectrum (Ledoit & Wolf 2020).

    Epanechnikov kernel with local bandwidth ``h lambda_j``, ``h = n^{-1/3}``
    (their eq. 4.9), its density (4.7) and Hilbert transform (4.8), the
    shrinkage formula (4.3) for ``p <= n`` and (C.4), (C.5), (C.8) for
    ``p > n``. The ``|x| = sqrt(5)`` edge of the Hilbert transform is set to
    its closed-form limit, the log term is evaluated as
    ``log|sqrt5 - x| - log|sqrt5 + x|``, and beyond ``|x| = 5`` the transform
    is summed as an exact series (no cancellation).

    Returns
    -------
    (d0, d)
        As :func:`qis_map` (no trace renormalisation: LW 2020 has none).
    """
    lam = _nonnull(values, p, n)
    m = lam.size
    c = p / n
    h = n ** (-1.0 / 3.0)
    H = h * lam  # bandwidth at each kernel centre j
    f = np.empty(m)
    Hf = np.empty(m)
    k_f = 3.0 / 4.0 / _SQRT5
    for lo in range(0, m, _CHUNK):
        hi = min(lo + _CHUNK, m)
        x = (lam[lo:hi, None] - lam[None, :]) / H[None, :]
        f[lo:hi] = k_f * np.mean(
            np.maximum(1.0 - x * x / 5.0, 0.0) / H[None, :], axis=1
        )
        Hf[lo:hi] = np.mean(_epanechnikov_hilbert(x) / H[None, :], axis=1)
    if m == p and c <= 1.0:
        d = lam / (
            (math.pi * c * lam * f) ** 2 + (1.0 - c - math.pi * c * lam * Hf) ** 2
        )
        d0 = math.nan
    else:
        if not math.sqrt(5.0) * h < 1.0:
            raise ValueError(
                "lw2020 with p > n needs n >= 12 (sqrt(5) * n^(-1/3) < 1); use "
                "method='qis'."
            )
        c_null = c if c > 1.0 else p / m
        hf0 = (
            (1.0 / math.pi)
            * (
                3.0 / 10.0 / (h * h)
                + 3.0 / 4.0 / _SQRT5 / h * (1.0 - 1.0 / 5.0 / (h * h))
                * math.log((1.0 + _SQRT5 * h) / (1.0 - _SQRT5 * h))
            )
            * float(np.mean(1.0 / lam))
        )  # fmt: skip
        d0 = 1.0 / (math.pi * (c_null - 1.0) * hf0)
        d = lam / (math.pi**2 * lam * lam * (f * f + Hf * Hf))
    return float(d0), np.ascontiguousarray(d[::-1])


def nonlinear_estimate(ws: WindowStats, method: str) -> CovEstimate:
    """A spectral :class:`CovEstimate` from a nonlinear map of the spectrum.

    ``B = V`` (non-null sample eigenvectors), ``g = d - d0``, ``e0 = d0``;
    when there are no null directions ``e0 = 0`` and ``g = d``.
    """
    fn = {"qis": qis_map, "lw2020": lw2020_map}[method]
    spec = ws.spectrum(vectors=True)
    assert spec.vectors is not None
    d0, d = fn(spec.values, ws.n_eff, ws.p)
    m = d.size
    V = spec.vectors[:, :m]
    if math.isnan(d0):
        g, e0 = d, 0.0
    else:
        g, e0 = d - d0, d0
    return CovEstimate(
        asof=ws.asof,
        location=ws.mean.copy(),
        entities=ws.entities,
        scale=ws.scale.copy(),
        B=V,
        g=g,
        e=float(e0),
        orthonormal=True,
        method=method,
        space=ws.space,
        n_eff=ws.n_eff,
        q=ws.p / ws.n_eff,
        shrinkage=None,
        info={"null_value": float(e0)},
    )


def qis(ws: WindowStats) -> CovEstimate:
    """Quadratic-Inverse Shrinkage estimate (the package default)."""
    return nonlinear_estimate(ws, "qis")


def lw2020(ws: WindowStats) -> CovEstimate:
    """Analytical nonlinear shrinkage estimate (Ledoit & Wolf 2020)."""
    return nonlinear_estimate(ws, "lw2020")
