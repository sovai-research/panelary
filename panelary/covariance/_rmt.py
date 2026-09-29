"""Random-matrix cleaning: Marchenko--Pastur edge, signal count, denoising.

All deterministic and numpy-only -- no KDE, no optimiser (plan section 5.8).

* **MP edge.** ``lambda_pm = sigma^2 (1 +/- sqrt(q))^2`` with ``q = p / n``.
* **sigma^2 by fixed point.** Start from ``(tr S - lambda_1) / p`` (in
  correlation space ``1 - lambda_1 / p``, Laloux et al. 1999) and iterate
  ``sigma^2 <- sum_{lambda_i <= edge} lambda_i / (p - #spikes)``. The
  denominator counts the ``p - n`` null eigenvalues: MP's mean over *all*
  ``p`` eigenvalues is ``sigma^2``. Stops when the spike set is unchanged.
  This replaces Lopez de Prado's scipy KDE fit by design; parity with it is
  on the signal *count*.
* **Finite-size edge.** Johnstone's (2001) Tracy--Widom centring and scaling,
  ``edge = sigma^2 [(a + b)^2 + t (a + b)(1/a + 1/b)^{1/3}] / n`` with
  ``a = sqrt(n - 1)``, ``b = sqrt p`` and the TW1 95% quantile
  ``t = 0.9793``. The bare edge over-counts at small ``p``.
* **Denoise** (Lopez de Prado 2020, section 2.5): constant residual
  eigenvalue, unit diagonal restored through the estimate's ``scale``.
* **Targeted shrinkage** (section 2.6) and **detoning** (section 2.8).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from panelary.covariance._types import CovEstimate, Spectrum, WindowStats

__all__ = [
    "TW1_QUANTILES",
    "detone",
    "mp_clip",
    "mp_edges",
    "mp_fit",
    "mp_targeted",
    "tw_edge",
]

#: Quantiles of the Tracy--Widom law for beta = 1 (real case), from the
#: tabulated distribution (Tracy & Widom 1996; Johnstone 2001, table 1).
TW1_QUANTILES: dict[float, float] = {0.90: 0.4501, 0.95: 0.9793, 0.99: 2.0234}

_MAX_ITER = 50


def mp_edges(sigma2: float, q: float) -> tuple[float, float]:
    """Marchenko--Pastur support ``sigma^2 (1 -/+ sqrt q)^2``."""
    r = math.sqrt(q)
    return sigma2 * (1.0 - r) ** 2, sigma2 * (1.0 + r) ** 2


def tw_edge(sigma2: float, n: float, p: int, alpha: float = 0.95) -> float:
    """Tracy--Widom finite-size upper edge at level ``alpha``.

    ``sigma^2 [mu + t_alpha s] / n`` with Johnstone's (2001) centring and
    scaling, ``mu = (sqrt(n - 1) + sqrt p)^2`` and
    ``s = (sqrt(n - 1) + sqrt p)(1/sqrt(n - 1) + 1/sqrt p)^{1/3}``, where
    ``n`` is the effective sample size (``T - 1`` after demeaning). Measured
    size at ``p = 400``, ``n = 999``: 4.2% over 1,000 white-noise windows
    (3.6% with ``sqrt n`` in place of ``sqrt(n - 1)``).
    """
    if alpha not in TW1_QUANTILES:
        raise ValueError(
            f"`alpha` must be one of {sorted(TW1_QUANTILES)}, got {alpha}."
        )
    t = TW1_QUANTILES[alpha]
    a, b = math.sqrt(max(n - 1.0, 1.0)), math.sqrt(p)
    mu = (a + b) ** 2
    s = (a + b) * (1.0 / a + 1.0 / b) ** (1.0 / 3.0)
    return sigma2 * (mu + t * s) / n


def mp_fit(
    values: NDArray[np.float64],
    p: int,
    n: float,
    *,
    edge: str = "tw",
    alpha: float = 0.95,
) -> dict[str, float]:
    """Noise level, edge and signal count of one spectrum, by fixed point.

    Parameters
    ----------
    values : ndarray
        Sample eigenvalues, descending (``min(n, p)`` of them; the rest of the
        ``p`` are 0).
    p : int
        Dimension.
    n : float
        Effective sample size.
    edge : {"tw", "bare"}, default "tw"
        Tracy--Widom finite-size edge, or the bare MP edge.
    alpha : float, default 0.95
        TW level (``edge="tw"``).

    Returns
    -------
    dict
        ``sigma2``, ``edge`` (the upper edge used), ``n_signal`` (eigenvalues
        strictly above it) and ``n_iter``.
    """
    if edge not in ("tw", "bare"):
        raise ValueError(f"`edge` must be 'tw' or 'bare', got {edge!r}.")
    lam = np.asarray(values, dtype=np.float64)
    q = p / n
    tr = float(lam.sum())

    def upper(s2: float) -> float:
        return tw_edge(s2, n, p, alpha) if edge == "tw" else mp_edges(s2, q)[1]

    k_prev = -1
    sigma2 = (tr - float(lam[0])) / p if lam.size else 0.0
    it = 0
    k = 0
    for it in range(1, _MAX_ITER + 1):  # noqa: B007 - `it` is reported
        up = upper(sigma2)
        k = int(np.count_nonzero(lam > up))
        if k == k_prev or k >= p:
            break
        sigma2 = float(lam[k:].sum()) / (p - k)
        k_prev = k
    up = upper(sigma2)
    k = int(np.count_nonzero(lam > up))
    return {"sigma2": sigma2, "edge": up, "n_signal": float(k), "n_iter": float(it)}


def _spec_vectors(ws: WindowStats) -> tuple[Spectrum, NDArray[np.float64]]:
    spec = ws.spectrum(vectors=True)
    assert spec.vectors is not None
    return spec, spec.vectors


def _estimate(
    ws: WindowStats,
    method: str,
    *,
    scale: NDArray[np.float64],
    B: NDArray[np.float64],
    g: NDArray[np.float64],
    e: float | NDArray[np.float64],
    orthonormal: bool,
    info: dict[str, Any],
    invertible: bool = True,
) -> CovEstimate:
    return CovEstimate(
        asof=ws.asof,
        location=ws.mean.copy(),
        entities=ws.entities,
        scale=scale,
        B=B,
        g=g,
        e=e,
        orthonormal=orthonormal,
        method=method,
        space=ws.space,
        n_eff=ws.n_eff,
        q=ws.p / ws.n_eff,
        shrinkage=None,
        invertible=invertible,
        info=info,
    )


def _renormalised_scale(
    ws: WindowStats, B: NDArray[np.float64], g: NDArray[np.float64], e0: float
) -> NDArray[np.float64]:
    """In correlation space, fold the unit-diagonal restoration into ``scale``."""
    if ws.space != "correlation":
        return ws.scale.copy()
    diag = (B * B) @ g + e0
    return ws.scale / np.sqrt(diag)


def _signal_count(
    spec: Spectrum, ws: WindowStats, n_signal: int | None, edge: str, alpha: float
) -> tuple[int, dict[str, float]]:
    fit = mp_fit(spec.values, ws.p, ws.n_eff, edge=edge, alpha=alpha)
    k = int(fit["n_signal"]) if n_signal is None else int(n_signal)
    return k, fit


def mp_clip(
    ws: WindowStats,
    *,
    n_signal: int | None = None,
    edge: str = "tw",
    alpha: float = 0.95,
) -> CovEstimate:
    """Denoise by constant residual eigenvalue (Lopez de Prado 2020, s. 2.5).

    The ``k`` signal eigenvalues (above the MP edge, or ``n_signal`` if given)
    are kept; all other ``p - k`` eigenvalues -- including the null ones -- are
    replaced by their mean, which preserves the trace. In correlation space
    the unit diagonal is restored by rescaling, which stays exact in DPLR form
    (``scale <- scale / sqrt(diag C)``), so the estimate remains spectral.
    """
    spec, V = _spec_vectors(ws)
    k, fit = _signal_count(spec, ws, n_signal, edge, alpha)
    p = ws.p
    k = max(0, min(k, V.shape[1], p - 1))
    lam = spec.values
    resid = (spec.trace - float(lam[:k].sum())) / (p - k)
    B = V[:, :k]
    g = lam[:k] - resid
    info = {**fit, "n_signal": float(k), "residual": resid}
    return _estimate(
        ws, "mp_clip", scale=_renormalised_scale(ws, B, g, resid), B=B, g=g,
        e=float(resid), orthonormal=True, info=info,
    )  # fmt: skip


def mp_targeted(
    ws: WindowStats,
    *,
    alpha_shrink: float = 0.0,
    n_signal: int | None = None,
    edge: str = "tw",
    alpha: float = 0.95,
) -> CovEstimate:
    """Targeted shrinkage (Lopez de Prado 2020, s. 2.6).

    ``C = C_signal + a C_noise + (1 - a) diag(C_noise)`` with ``a =
    alpha_shrink``: the signal block is kept, the noise block is shrunk toward
    its own diagonal. Woodbury form (the residual is a non-isotropic
    diagonal). ``alpha_shrink=0`` is Lopez de Prado's default.
    """
    if not 0.0 <= alpha_shrink <= 1.0:
        raise ValueError(f"`alpha_shrink` must be in [0, 1], got {alpha_shrink}.")
    spec, V = _spec_vectors(ws)
    k, fit = _signal_count(spec, ws, n_signal, edge, alpha)
    r = V.shape[1]
    k = max(0, min(k, r))
    lam = spec.values[:r]
    s = np.einsum("ij,ij->j", ws.Z, ws.Z) / ws.n_eff  # diag(S)
    sig_diag = (V[:, :k] * V[:, :k]) @ lam[:k]
    noise_diag = np.clip(s - sig_diag, 0.0, None)
    g = lam.copy()
    g[k:] *= alpha_shrink
    keep = g != 0.0
    keep[:k] = True
    B = V[:, keep]
    g = g[keep]
    info = {**fit, "n_signal": float(k), "alpha_shrink": alpha_shrink}
    return _estimate(
        ws, "mp_targeted", scale=ws.scale.copy(), B=B, g=g,
        e=(1.0 - alpha_shrink) * noise_diag, orthonormal=True, info=info,
    )  # fmt: skip


def detone(
    ws: WindowStats,
    *,
    n_market: int = 1,
    n_signal: int | None = None,
    edge: str = "tw",
    alpha: float = 0.95,
) -> CovEstimate:
    """Denoise, then remove the top ``n_market`` components (LdP 2020, s. 2.8).

    The result is singular by construction (``invertible=False``): it is
    meant for clustering and graph construction, and :meth:`solve` raises.
    """
    if n_market < 1:
        raise ValueError(f"`n_market` must be >= 1, got {n_market}.")
    spec, V = _spec_vectors(ws)
    k, fit = _signal_count(spec, ws, n_signal, edge, alpha)
    p = ws.p
    r = V.shape[1]
    k = max(0, min(k, r, p - 1))
    lam = spec.values
    resid = (spec.trace - float(lam[:k].sum())) / (p - k)
    kk = min(max(k, n_market), r)
    B = V[:, :kk]
    g = np.where(np.arange(kk) < k, lam[:kk] - resid, 0.0)
    g[: min(n_market, kk)] = -resid
    info = {**fit, "n_signal": float(k), "residual": resid, "n_market": float(n_market)}
    return _estimate(
        ws, "detone", scale=_renormalised_scale(ws, B, g, resid), B=B, g=g,
        e=float(resid), orthonormal=True, info=info, invertible=False,
    )  # fmt: skip
