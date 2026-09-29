"""Statistical factor model: ``k`` principal components plus a diagonal (5.9).

``Sigma = diag(scale) (V_k diag(lambda_k) V_k^T + diag(psi)) diag(scale)``
with ``psi_i = max(s_ii - sum_k lambda_k V_ik^2, psi_floor * s_ii)``. The
number of factors is chosen **per window** -- a full-sample ``k`` is trap T9:

* an ``int`` (default 1);
* ``"eigenvalue_ratio"`` -- :func:`panelary.reduce._n_factors.eigenvalue_ratio`
  on the window spectrum;
* ``"bai_ng"`` -- :func:`panelary.reduce._n_factors._bai_ng_from_spectrum`,
  which reuses the spectrum instead of a second SVD;
* ``"mp"`` -- the Marchenko--Pastur signal count (:func:`_rmt.mp_fit`).

The inverse is Woodbury in ``O(N k)``.
"""

from __future__ import annotations

import numpy as np

from panelary.covariance._rmt import mp_fit
from panelary.covariance._types import CovEstimate, WindowStats
from panelary.reduce._n_factors import (
    DEFAULT_MAX_FACTORS,
    _bai_ng_from_spectrum,
    eigenvalue_ratio,
)

__all__ = ["FACTOR_RULES", "factor_model", "n_factors_from_spectrum"]

FACTOR_RULES = ("eigenvalue_ratio", "bai_ng", "mp")


def n_factors_from_spectrum(
    ws: WindowStats, rule: int | str, *, max_factors: int | None = None
) -> int:
    """Resolve the factor count on this window only."""
    spec = ws.spectrum(vectors=False)
    r = spec.rank
    if isinstance(rule, (int, np.integer)) and not isinstance(rule, bool):
        if int(rule) < 0:
            raise ValueError(f"`n_factors` must be >= 0, got {rule}.")
        return min(int(rule), r)
    kmax = DEFAULT_MAX_FACTORS if max_factors is None else int(max_factors)
    kmax = max(1, min(kmax, r - 1 if r > 1 else 1, ws.p - 1 if ws.p > 1 else 1))
    if rule == "eigenvalue_ratio":
        return min(eigenvalue_ratio(spec.values[:r], max_r=kmax), r)
    if rule == "bai_ng":
        sv2 = spec.values * ws.n_eff
        return min(_bai_ng_from_spectrum(sv2, ws.n, ws.p, kmax), r)
    if rule == "mp":
        return min(int(mp_fit(spec.values, ws.p, ws.n_eff)["n_signal"]), r)
    raise ValueError(
        f"`n_factors` must be an int or one of {FACTOR_RULES}, got {rule!r}."
    )


def factor_model(
    ws: WindowStats,
    *,
    n_factors: int | str = 1,
    psi_floor: float = 1e-3,
    max_factors: int | None = None,
) -> CovEstimate:
    """PCA factor model plus diagonal residual (Woodbury form)."""
    if not psi_floor > 0.0:
        raise ValueError(f"`psi_floor` must be > 0, got {psi_floor}.")
    # Vectors first, so the count is read off the same (eigh) spectrum.
    spec = ws.spectrum(vectors=True)
    assert spec.vectors is not None
    k = n_factors_from_spectrum(ws, n_factors, max_factors=max_factors)
    V = spec.vectors[:, :k]
    lam = spec.values[:k].copy()
    s = np.einsum("ij,ij->j", ws.Z, ws.Z) / ws.n_eff
    psi = np.maximum(s - (V * V) @ lam, psi_floor * s)
    return CovEstimate(
        asof=ws.asof,
        entities=ws.entities,
        scale=ws.scale.copy(),
        B=np.ascontiguousarray(V),
        g=lam,
        e=psi,
        orthonormal=True,
        method="factor",
        space=ws.space,
        n_eff=ws.n_eff,
        q=ws.p / ws.n_eff,
        shrinkage=None,
        info={"n_factors": float(k)},
    )
