"""The functional core: a window in, a :class:`CovEstimate` out.

``estimate(X, method=..., space=...)`` is what other subpackages call (drift
monitoring's Hotelling statistics, graph construction, DCC-NL targets): no
panel, no time axis, no schedule. The as-of machinery in
:mod:`panelary.covariance._window` calls the same estimators on each
scheduled window, so ``cov.rolling(...).at(t)`` and ``estimate(window_at_t)``
are bit-identical (trap T9's guard).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from panelary.covariance._factor import factor_model
from panelary.covariance._gram import window_stats
from panelary.covariance._linear import (
    lw_constant_correlation,
    lw_diagonal,
    lw_identity,
    lw_single_index,
    oas,
)
from panelary.covariance._missing import MISSING_POLICIES
from panelary.covariance._nonlinear import lw2020, qis
from panelary.covariance._rmt import detone, mp_clip, mp_targeted
from panelary.covariance._types import CovEstimate, WindowStats

__all__ = ["METHODS", "estimate", "estimate_window", "resolve_method", "sample"]


def sample(ws: WindowStats) -> CovEstimate:
    """The sample estimate ``S`` itself (spectral form, singular when ``p > n``).

    Kept as the honest baseline: when it is singular, ``solve`` raises and
    ``cond`` is ``inf`` -- it never pseudo-inverts (trap T14).
    """
    spec = ws.spectrum(vectors=True)
    assert spec.vectors is not None
    r = spec.vectors.shape[1]
    return CovEstimate(
        asof=ws.asof,
        entities=ws.entities,
        scale=ws.scale.copy(),
        B=spec.vectors,
        g=spec.values[:r].copy(),
        e=0.0,
        orthonormal=True,
        method="sample",
        space=ws.space,
        n_eff=ws.n_eff,
        q=ws.p / ws.n_eff,
        shrinkage=None,
    )


#: Canonical method name -> estimator. Every one takes a :class:`WindowStats`
#: (plus keyword options) and returns a :class:`CovEstimate`.
METHODS: dict[str, Callable[..., CovEstimate]] = {
    "sample": sample,
    "lw_identity": lw_identity,
    "lw_diagonal": lw_diagonal,
    "lw_constant_correlation": lw_constant_correlation,
    "lw_single_index": lw_single_index,
    "oas": oas,
    "qis": qis,
    "lw2020": lw2020,
    "mp_clip": mp_clip,
    "mp_targeted": mp_targeted,
    "detone": detone,
    "factor": factor_model,
}

_ALIASES = {
    "lw": "lw_identity",
    "ledoit_wolf": "lw_identity",
    "constant_correlation": "lw_constant_correlation",
    "single_index": "lw_single_index",
    "denoise": "mp_clip",
}


def resolve_method(method: str) -> str:
    """Canonical name for ``method`` (aliases resolved), or ``ValueError``."""
    name = _ALIASES.get(method, method)
    if name not in METHODS:
        raise ValueError(
            f"unknown covariance method {method!r}; expected one of "
            f"{sorted(METHODS)} (aliases: {sorted(_ALIASES)})."
        )
    return name


def estimate_window(
    ws: WindowStats, method: str = "qis", **options: Any
) -> CovEstimate:
    """Run one estimator on an already-built :class:`WindowStats`."""
    return METHODS[resolve_method(method)](ws, **options)


def estimate(
    X: Any,
    *,
    method: str = "qis",
    space: str = "correlation",
    assume_centered: bool = False,
    missing: str = "zero_after_demean",
    entities: tuple[Any, ...] | None = None,
    asof: Any = None,
    **options: Any,
) -> CovEstimate:
    """Estimate the covariance of the columns of one ``(n, p)`` window.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Rows are observations (dates), columns variables (entities). NaN
        marks a missing cell. Float32 is upcast to float64.
    method : str, default "qis"
        ``"qis"`` (Quadratic-Inverse Shrinkage, the default), ``"lw2020"``,
        ``"lw_identity"`` (alias ``"lw"``), ``"oas"``, ``"lw_diagonal"``,
        ``"lw_constant_correlation"``, ``"lw_single_index"``, ``"mp_clip"``,
        ``"mp_targeted"``, ``"detone"``, ``"factor"`` or ``"sample"``.
    space : {"correlation", "covariance"}, default "correlation"
        Where the estimator operates. In correlation space the window sds are
        the diagonal ``scale`` and the estimator shrinks the correlation
        matrix -- measured 1.31 vs 2.21 (out-of-sample GMV variance over the
        oracle) at ``N / W ~ 2`` for the same QIS map.
    assume_centered : bool, default False
        Skip demeaning; ``n_eff = n`` (sklearn's MLE divisor) instead of
        ``n - 1``.
    missing : {"zero_after_demean"}, default "zero_after_demean"
    entities : tuple, optional
        Column labels carried onto the estimate.
    asof : Any, optional
        The date the window ends on, carried onto the estimate.
    **options
        Estimator options: ``n_signal``, ``edge``, ``alpha`` (RMT methods),
        ``alpha_shrink`` (``mp_targeted``), ``n_market`` (``detone``),
        ``n_factors``, ``psi_floor``, ``max_factors`` (``factor``).

    Returns
    -------
    CovEstimate

    Examples
    --------
    >>> import numpy as np
    >>> from panelary.covariance import estimate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((60, 100))           # p > n: sample is singular
    >>> est = estimate(X, method="qis")
    >>> w = est.gmv_weights()
    >>> bool(np.isclose(w.sum(), 1.0)), est.cond() < np.inf
    (True, True)
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(
            f"`missing` must be one of {MISSING_POLICIES}, got {missing!r} "
            "(pairwise + repair ships in M5)."
        )
    name = resolve_method(method)
    ws = window_stats(
        X, space=space, assume_centered=assume_centered, entities=entities, asof=asof
    )
    return METHODS[name](ws, **options)
