"""Linear shrinkage with closed-form intensities from the dual Gram (plan 5.6).

Let ``Z`` be the centred ``(n_rows, p)`` window, ``n = n_eff`` the divisor,
``S = Z^T Z / n`` and ``d_t = ||z_t||^2``. Then

* ``tr S = sum_t d_t / n``,
* ``||S||_F^2 = ||gram||_F^2`` (the min-side Gram, already divided by ``n``),
* ``sum_{ij} sum_t z_ti^2 z_tj^2 = sum_t d_t^2``,

so every intensity below costs ``O(n^2 + n p)``, never ``O(p^2)``. The
estimates are emitted in diagonal-plus-low-rank form on the sample
eigenvectors (plus one extra column for the constant-correlation and
single-index targets), so ``solve`` / ``logdet`` never touch ``p x p``.

Formulas, each clean-room from the paper and checked against the reference
implementation in the tests:

* ``lw_identity`` -- Ledoit & Wolf (2004a), target ``mu I``. The same algebra
  as ``sklearn.covariance.ledoit_wolf_shrinkage``.
* ``lw_diagonal`` -- target ``diag(S)`` (``covDiag`` of Ledoit & Wolf's
  covShrinkage).
* ``lw_constant_correlation`` -- Ledoit & Wolf (2004b), target with the sample
  variances and one average correlation ``r_bar``.
* ``lw_single_index`` -- Ledoit & Wolf (2003), target ``c c^T / v`` plus the
  residual diagonal, the market being the equal-weight mean of the window.
* ``oas`` -- Chen, Wiesel, Eldar & Hero (2010), eq. 23 **as published**.
  scikit-learn omits both ``2/p`` terms; the difference is at most ``4 rho/p``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from panelary.covariance._types import CovEstimate, WindowStats

__all__ = [
    "lw_constant_correlation",
    "lw_diagonal",
    "lw_identity",
    "lw_identity_intensity",
    "lw_single_index",
    "oas",
    "oas_intensity",
]


def _traces(ws: WindowStats) -> tuple[float, float, NDArray[np.float64]]:
    """``(tr S, ||S||_F^2, d)`` with ``d_t = ||z_t||^2``."""
    Z = ws.Z
    d = np.einsum("ij,ij->i", Z, Z)
    tr = float(d.sum()) / ws.n_eff
    G = ws.gram()
    f2 = float(np.einsum("ij,ij->", G, G))
    return tr, f2, d


def _base(
    ws: WindowStats,
    method: str,
    B: NDArray[np.float64],
    g: NDArray[np.float64],
    e: float | NDArray[np.float64],
    *,
    orthonormal: bool,
    shrinkage: float | None,
    info: dict[str, float] | None = None,
) -> CovEstimate:
    return CovEstimate(
        asof=ws.asof,
        location=ws.mean.copy(),
        entities=ws.entities,
        scale=ws.scale.copy(),
        B=B,
        g=g,
        e=e,
        orthonormal=orthonormal,
        method=method,
        space=ws.space,
        n_eff=ws.n_eff,
        q=ws.p / ws.n_eff,
        shrinkage=shrinkage,
        info=dict(info or {}),
    )


def _vectors(ws: WindowStats) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    spec = ws.spectrum(vectors=True)
    assert spec.vectors is not None
    r = spec.vectors.shape[1]
    return spec.vectors, spec.values[:r]


# --------------------------------------------------------------------------- #
# identity target
# --------------------------------------------------------------------------- #
def lw_identity_intensity(ws: WindowStats) -> tuple[float, float]:
    """``(rho, mu)`` for Ledoit--Wolf (2004a) shrinkage to ``mu I``.

    ``mu = tr S / p``; ``delta^2 = (||S||_F^2 - 2 mu tr S + p mu^2) / p``;
    ``beta^2 = (sum_t d_t^2 / n - ||S||_F^2) / (p n)``;
    ``rho = min(beta^2, delta^2) / delta^2``.
    """
    n, p = ws.n_eff, ws.p
    tr, f2, d = _traces(ws)
    mu = tr / p
    delta = (f2 - 2.0 * mu * tr + p * mu * mu) / p
    beta = (float(d @ d) / n - f2) / (p * n)
    beta = min(beta, delta)
    rho = 0.0 if beta == 0.0 else beta / delta
    return float(rho), float(mu)


def lw_identity(ws: WindowStats) -> CovEstimate:
    """Ledoit--Wolf (2004a): ``(1 - rho) S + rho mu I`` (spectral form)."""
    rho, mu = lw_identity_intensity(ws)
    V, lam = _vectors(ws)
    return _base(
        ws, "lw_identity", V, (1.0 - rho) * lam, rho * mu,
        orthonormal=True, shrinkage=rho, info={"target_mean": mu},
    )  # fmt: skip


# --------------------------------------------------------------------------- #
# OAS
# --------------------------------------------------------------------------- #
def oas_intensity(ws: WindowStats) -> tuple[float, float]:
    """``(rho, mu)`` for OAS, Chen et al. (2010) eq. 23 as published.

    ``rho = min(1, ((1 - 2/p) tr(S^2) + tr^2 S) /
    ((n + 1 - 2/p) (tr(S^2) - tr^2 S / p)))``.
    """
    n, p = ws.n_eff, ws.p
    tr, f2, _ = _traces(ws)
    num = (1.0 - 2.0 / p) * f2 + tr * tr
    den = (n + 1.0 - 2.0 / p) * (f2 - tr * tr / p)
    rho = 1.0 if den <= 0.0 else min(1.0, num / den)
    return float(rho), float(tr / p)


def oas(ws: WindowStats) -> CovEstimate:
    """Oracle approximating shrinkage to ``mu I`` (spectral form)."""
    rho, mu = oas_intensity(ws)
    V, lam = _vectors(ws)
    return _base(
        ws, "oas", V, (1.0 - rho) * lam, rho * mu,
        orthonormal=True, shrinkage=rho, info={"target_mean": mu},
    )  # fmt: skip


# --------------------------------------------------------------------------- #
# targets that keep the sample variances
# --------------------------------------------------------------------------- #
def _moments(
    ws: WindowStats,
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
]:
    """``(n, s, Z^2, d, ||S||_F^2, pi, rho_diag)`` shared by the three targets.

    ``pi = sum_t d_t^2 / n - ||S||_F^2`` and
    ``rho_diag = sum_i (sum_t z_ti^4 / n - s_ii^2)``.
    """
    n = ws.n_eff
    Z = ws.Z
    Z2 = Z * Z
    s = Z2.sum(axis=0) / n
    d = Z2.sum(axis=1)
    _, f2, _ = _traces(ws)
    pi = float(d @ d) / n - f2
    rho_diag = float(np.einsum("ij,ij->", Z2, Z2)) / n - float(s @ s)
    return n, s, Z2, d, f2, pi, rho_diag


def _intensity(num: float, gamma: float, n: float) -> float:
    if not gamma > 0.0:
        return 1.0
    return float(max(0.0, min(1.0, num / gamma / n)))


def lw_diagonal(ws: WindowStats) -> CovEstimate:
    """Shrinkage to ``diag(S)`` (covShrinkage ``covDiag``); Woodbury form."""
    n, s, _, _, f2, pi, rho_diag = _moments(ws)
    gamma = f2 - float(s @ s)
    delta = _intensity(pi - rho_diag, gamma, n)
    V, lam = _vectors(ws)
    return _base(
        ws, "lw_diagonal", V, (1.0 - delta) * lam, delta * s,
        orthonormal=True, shrinkage=delta,
    )  # fmt: skip


def lw_constant_correlation(ws: WindowStats) -> CovEstimate:
    """Ledoit--Wolf (2004b): shrink to a constant-correlation target.

    Target ``F = r_bar sigma sigma^T + diag((1 - r_bar) s)``, with the sample
    variances ``s`` on its diagonal and ``r_bar`` the mean off-diagonal sample
    correlation, ``(||Z sigma^{-1}||^2 / n - p) / (p (p - 1))`` -- one matvec.
    Emitted as ``B = [V, sigma]`` on the Woodbury path.
    """
    n, s, Z2, d, f2, pi, rho_diag = _moments(ws)
    Z = ws.Z
    p = ws.p
    if p < 2:
        raise ValueError("lw_constant_correlation needs at least 2 entities.")
    sig = np.sqrt(s)
    if np.any(sig <= 0.0):
        raise ValueError(
            "lw_constant_correlation: a column has zero variance in the window."
        )
    inv = 1.0 / sig
    u = Z @ inv
    r_bar = (float(u @ u) / n - p) / (p * (p - 1))
    b = Z @ sig
    sSs = float(b @ b) / n
    ss = float(s @ s)
    s_sum = float(s.sum())
    gamma = (f2 - ss) - 2.0 * r_bar * (sSs - ss) + r_bar * r_bar * (s_sum * s_sum - ss)
    a = (Z2 * Z) @ inv
    rho_off = r_bar * (float(a @ b) / n - sSs - rho_diag)
    delta = _intensity(pi - rho_diag - rho_off, gamma, n)
    V, lam = _vectors(ws)
    B = np.column_stack([V, sig])
    g = np.append((1.0 - delta) * lam, delta * r_bar)
    return _base(
        ws, "lw_constant_correlation", B, g, delta * (1.0 - r_bar) * s,
        orthonormal=False, shrinkage=delta, info={"r_bar": r_bar},
    )  # fmt: skip


def lw_single_index(ws: WindowStats) -> CovEstimate:
    """Ledoit--Wolf (2003): shrink to a single-index (market) target.

    The market is the equal-weight mean of the window's universe,
    ``m_t = mean_i z_ti``; ``c = Z^T m / n``, ``v = m^T m / n`` and the target
    is ``c c^T / v + diag(s - c^2 / v)``. Every intensity term is a matvec
    against ``m`` or ``c`` (``O(n p)``). Emitted as ``B = [V, c]`` on the
    Woodbury path.
    """
    n, s, Z2, d, f2, pi, rho_diag = _moments(ws)
    Z = ws.Z
    m = Z.mean(axis=1)
    c = (Z.T @ m) / n
    v = float(m @ m) / n
    if not v > 0.0:
        raise ValueError("lw_single_index: the market proxy has zero variance.")
    Zc = Z @ c
    cSc = float(Zc @ Zc) / n
    c2 = c * c
    sc2 = float(s @ c2)
    c2_sum = float(c2.sum())
    gamma = (f2 - float(s @ s)) - (2.0 / v) * (cSc - sc2)
    gamma += (c2_sum * c2_sum - float(c2 @ c2)) / (v * v)
    first = float(np.sum(m * d * Zc)) / n
    diag1 = float(m @ ((Z2 * Z) @ c)) / n - sc2
    roff1 = (first - cSc - diag1) / v
    m2 = m * m
    full3 = float(m2 @ (Zc * Zc)) / n - v * cSc
    diag3 = float(m2 @ (Z2 @ c2)) / n - v * sc2
    roff3 = (full3 - diag3) / (v * v)
    rho_off = 2.0 * roff1 - roff3
    delta = _intensity(pi - rho_diag - rho_off, gamma, n)
    V, lam = _vectors(ws)
    B = np.column_stack([V, c])
    g = np.append((1.0 - delta) * lam, delta / v)
    return _base(
        ws, "lw_single_index", B, g, delta * (s - c2 / v),
        orthonormal=False, shrinkage=delta, info={"market_var": v},
    )  # fmt: skip
