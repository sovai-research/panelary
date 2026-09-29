"""How many factors? Bai--Ng information criteria and the eigenvalue ratio.

This is the connective tissue of the factor family: every extractor in
:mod:`panelary.reduce` accepts ``n_factors=None`` and resolves the count
here, on the **training rows only**, so the choice of ``r`` is as leak-safe as
the loadings themselves.

Two selectors, both pure NumPy (no ``scipy``, no ``sklearn``):

* :func:`bai_ng` -- Bai & Ng (2002) information criteria ``IC_p1`` / ``IC_p2``.
  Minimise ``ln(V_k) + k * g(N, T)`` where ``V_k`` is the mean squared residual
  of the rank-``k`` principal-component reconstruction. The default for the
  covariance-based methods (PCA, robust PCA, ICA).
* :func:`eigenvalue_ratio` -- the Ahn--Horenstein style eigenvalue-ratio test,
  ``r = argmax_k lambda_k / lambda_{k+1}``. It takes a *spectrum*, so it works
  equally well on a covariance spectrum and on HFA's higher-order **cumulant**
  spectrum -- which is exactly why HFA defaults to it: a cumulant matrix has no
  "explained variance" to plug into an information criterion.

References
----------
Bai, J. and Ng, S. (2002). "Determining the Number of Factors in Approximate
Factor Models." *Econometrica* 70(1), 191-221.

Ahn, S. C. and Horenstein, A. R. (2013). "Eigenvalue Ratio Test for the Number
of Factors." *Econometrica* 81(3), 1203-1227.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from panelary.reduce._common import covariance, prepare_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["n_factors", "bai_ng", "eigenvalue_ratio", "DEFAULT_MAX_FACTORS"]

#: Default cap on the searched factor count when ``max_r`` is not given. Bai &
#: Ng's own applied work fixes ``kmax = 8``; the criterion is only consistent for
#: ``kmax`` well below ``min(N, T)``, and letting ``k`` approach ``N`` drives the
#: residual ``V_k`` to zero and the criterion to nonsense.
DEFAULT_MAX_FACTORS = 8

_CRITERIA = ("IC_p1", "IC_p2")


def _resolve_max_r(n_rows: int, n_features: int, max_r: int | None) -> int:
    """Cap the search range at ``min(n_features, n_rows - 1, DEFAULT_MAX_FACTORS)``."""
    hard = max(1, min(int(n_features) - 1, int(n_rows) - 1))
    if max_r is None:
        return max(1, min(hard, DEFAULT_MAX_FACTORS))
    if int(max_r) < 1:
        raise ValueError(f"`max_r` must be a positive integer, got {max_r!r}.")
    return max(1, min(int(max_r), hard))


def bai_ng(
    X: NDArray[Any],
    *,
    max_r: int | None = None,
    criterion: str = "IC_p2",
    standardize: bool = True,
) -> int:
    """Number of factors by a Bai--Ng (2002) information criterion.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` observation matrix (``T x N`` in the paper's
        notation). NaNs are mean-filled and the matrix is centred (and scaled
        when ``standardize``).
    max_r : int, optional
        Largest factor count to consider. Defaults to
        ``min(n_features - 1, n_rows - 1, 8)`` -- the criterion is only
        well-behaved for ``max_r`` well below the number of features.
    criterion : {"IC_p1", "IC_p2"}, default "IC_p2"
        The penalty. ``IC_p2`` (``ln(min(N, T))``) is the more conservative of
        the two and is the library default.
    standardize : bool, default True
        Scale columns to unit variance before the decomposition.

    Returns
    -------
    int
        The minimiser of ``ln(V_k) + k * g(N, T)`` over ``k = 1 .. max_r``.

    Notes
    -----
    ``V_k``, the mean squared reconstruction residual, is read straight off the
    singular values: ``V_k = sum_{j>k} s_j^2 / (n_rows * n_features)``. It is
    floored at a tiny multiple of ``V_0`` so an exactly rank-deficient matrix
    does not produce ``log(0)``; with a floor in play the penalty term breaks the
    tie in favour of the smallest sufficient ``k``.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"`criterion` must be one of {_CRITERIA}, got {criterion!r}.")
    Z = prepare_matrix(X, standardize=standardize)
    n, p = Z.shape
    kmax = _resolve_max_r(n, p, max_r)

    svals = np.linalg.svd(Z, compute_uv=False)
    return _bai_ng_from_spectrum(svals**2, n, p, kmax, criterion)


def _bai_ng_from_spectrum(
    sv2: NDArray[Any], n: int, p: int, kmax: int, criterion: str = "IC_p2"
) -> int:
    """The Bai--Ng minimiser from squared singular values already in hand.

    :func:`bai_ng` without the SVD, for callers that have the spectrum already
    (``panelary.covariance`` computes one per window and must not pay for a
    second decomposition). ``sv2`` are the squared singular values of the
    centred ``(n, p)`` matrix -- equivalently ``n_eff`` times the eigenvalues
    of its sample covariance -- in any order that is descending; missing
    trailing zeros are fine. ``kmax`` is used as given (at least 1).
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"`criterion` must be one of {_CRITERIA}, got {criterion!r}.")
    kmax = max(1, int(kmax))
    sv2 = np.asarray(sv2, dtype=np.float64)
    if sv2.size < kmax + 1:
        sv2 = np.concatenate([sv2, np.zeros(kmax + 1 - sv2.size)])
    tail = np.concatenate([np.cumsum(sv2[::-1])[::-1], [0.0]])
    denom = float(n * p)
    v = tail / denom  # v[k] == V_k, the residual after keeping k components
    floor = max(float(v[0]) * 1e-12, np.finfo(float).tiny)
    v = np.maximum(v, floor)

    ratio = (n + p) / (n * p)
    if criterion == "IC_p1":
        g = ratio * np.log((n * p) / (n + p))
    else:
        g = ratio * np.log(min(n, p))

    ks = np.arange(1, kmax + 1)
    ic = np.log(v[1 : kmax + 1]) + ks * g
    return int(ks[int(np.argmin(ic))])


def eigenvalue_ratio(eigenvalues: NDArray[Any], *, max_r: int | None = None) -> int:
    """Number of factors by the eigenvalue-ratio rule.

    ``r = argmax_{k=1..max_r} lambda_k / lambda_{k+1}`` on the descending
    spectrum. Works on any non-negative spectrum, which is what lets HFA apply
    it to a higher-order **cumulant** spectrum rather than a covariance one.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Spectrum, in any order (it is sorted descending internally). Negative
        entries -- possible for a Gaussian-corrected order-4 cumulant matrix --
        are clipped to zero.
    max_r : int, optional
        Largest factor count to consider. Defaults to
        ``min(len(eigenvalues) - 1, 8)``.

    Returns
    -------
    int
        The selected factor count (at least 1).
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    lam = np.clip(lam, 0.0, None)
    m = lam.size
    if m < 2:
        return 1
    kmax = _resolve_max_r(m, m - 1, max_r)
    kmax = min(kmax, m - 1)
    floor = max(float(lam[0]) * 1e-12, np.finfo(float).tiny)
    denom = np.maximum(lam[1 : kmax + 1], floor)
    ratios = lam[:kmax] / denom
    return int(np.argmax(ratios)) + 1


def n_factors(
    X: NDArray[Any],
    *,
    method: str = "bai_ng",
    max_r: int | None = None,
    standardize: bool = True,
    criterion: str = "IC_p2",
) -> int:
    """Resolve the number of latent factors in ``X``.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` observation matrix.
    method : {"bai_ng", "eigenratio"}, default "bai_ng"
        ``"bai_ng"`` runs :func:`bai_ng`; ``"eigenratio"`` runs
        :func:`eigenvalue_ratio` on the covariance spectrum of ``X``.
    max_r : int, optional
        Largest factor count to consider.
    standardize : bool, default True
        Scale columns to unit variance first.
    criterion : {"IC_p1", "IC_p2"}, default "IC_p2"
        Only used by ``method="bai_ng"``.

    Returns
    -------
    int
        The selected factor count.

    Examples
    --------
    >>> import numpy as np
    >>> from panelary.reduce import n_factors
    >>> rng = np.random.default_rng(0)
    >>> F = rng.standard_normal((500, 3))
    >>> L = rng.standard_normal((3, 30))
    >>> X = F @ L + rng.standard_normal((500, 30))
    >>> n_factors(X)
    3
    >>> n_factors(X, method="eigenratio")
    3
    """
    if method == "bai_ng":
        return bai_ng(X, max_r=max_r, criterion=criterion, standardize=standardize)
    if method == "eigenratio":
        Z = prepare_matrix(X, standardize=standardize)
        eigvals = np.linalg.eigvalsh(covariance(Z))
        return eigenvalue_ratio(eigvals, max_r=max_r)
    raise ValueError(f"unknown `method={method!r}`; expected 'bai_ng' or 'eigenratio'.")
