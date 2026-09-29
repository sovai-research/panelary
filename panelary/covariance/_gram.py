"""The window kernel: two-pass centring, the min-side Gram, one eigensolve.

Everything in the package reads the second-moment structure of a window
through here (plan section 5.2):

* **Centring** is two-pass per window (``_missing.zero_after_demean``): the
  column mean is removed before any product is formed, so there is no raw-sum
  cancellation. Polars' ``rolling_var`` loses 1.8e-6 relative at a 1e8 offset;
  this path keeps ~1e-16.
* **Correlation space** divides each centred column by its window standard
  deviation ``sqrt(sum_obs x^2 / (n_i - 1))`` (``n_i`` when
  ``assume_centered``); the sds become the estimate's diagonal ``scale``.
* **Min-side Gram.** With ``p > n`` the ``n x n`` dual ``Z Z^T / n_eff`` has
  the same nonzero spectrum as the ``p x p`` primal ``Z^T Z / n_eff`` and is
  ~560x cheaper to decompose at ``p = 3000``. Dual eigenvectors lift to the
  primal as ``V = Z^T U diag(lambda * n_eff)^{-1/2}``.
* **Clipping.** Eigenvalues below ``max(n, p) * eps * lambda_max`` are set to
  exactly 0; demeaning removes one direction, so the rank is at most
  ``min(p, n - 1)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from panelary.covariance._missing import zero_after_demean
from panelary.covariance._types import Spectrum, WindowStats

__all__ = ["SPACES", "eigen", "min_side_gram", "orient_signs", "window_stats"]

SPACES = ("correlation", "covariance")

#: Columns of lifted dual eigenvectors whose norm is off by more than this are
#: re-orthonormalised (measured deviation 1.3e-14, so this is a guard only).
_ORTHO_TOL = 1e-10


def window_stats(
    X: Any,
    *,
    space: str = "correlation",
    assume_centered: bool = False,
    entities: tuple[Any, ...] | None = None,
    asof: Any = None,
    count: NDArray[np.int64] | None = None,
    copy: bool = True,
    drop_constant: bool = False,
) -> WindowStats:
    """Centre (and standardise) one ``(n, p)`` window.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Rows are dates, columns entities. Non-finite cells are missing and
        are zero-filled after demeaning. Float32 is upcast to float64.
    space : {"correlation", "covariance"}, default "correlation"
    assume_centered : bool, default False
        Treat the data as already centred: no mean is removed and
        ``n_eff = n`` (sklearn's maximum-likelihood divisor). Otherwise
        ``n_eff = n - 1``.
    entities : tuple, optional
        Column labels; defaults to ``0 .. p-1``.
    asof : Any, optional
        The date the window ends on, carried onto every estimate.
    count, copy
        Passed to :func:`~panelary.covariance._missing.zero_after_demean`
        (the as-of engine supplies exact counts and an owned array).
    drop_constant : bool, default False
        Drop columns with zero variance (or fewer than two observations)
        instead of raising, recording the kept ones in ``ws.kept``.

    Raises
    ------
    ValueError
        On a bad shape or space, fewer than 2 rows, or (correlation space) a
        column with zero variance over the window.
    """
    if space not in SPACES:
        raise ValueError(f"`space` must be one of {SPACES}, got {space!r}.")
    Z, mean, count = zero_after_demean(
        X, assume_centered=assume_centered, count=count, copy=copy
    )
    n, p = Z.shape
    if n < 2 or p < 1:
        raise ValueError(
            f"a window needs at least 2 rows and 1 column, got shape {(n, p)}."
        )
    n_eff = float(n if assume_centered else n - 1)
    ddof = 0 if assume_centered else 1
    kept = None
    var = None
    if space == "correlation" or drop_constant:
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.einsum("ij,ij->j", Z, Z) / (count - ddof)
        bad = ~(np.isfinite(var) & (var > 0.0))
        if bad.any():
            if not drop_constant:
                cols = np.flatnonzero(bad)[:8].tolist()
                raise ValueError(
                    "correlation space needs every column to vary over the "
                    f"window; column(s) {cols} have zero variance or fewer than "
                    f"{ddof + 1} observations. Drop them, or use "
                    "space='covariance'."
                )
            kept = ~bad
            Z = np.ascontiguousarray(Z[:, kept])
            mean, count, var = mean[kept], count[kept], var[kept]
            p = Z.shape[1]
            if entities is not None:
                entities = tuple(e for e, k in zip(entities, kept, strict=True) if k)
    if space == "correlation":
        assert var is not None
        scale = np.sqrt(var)
        Z /= scale
    else:
        scale = np.ones(p, dtype=np.float64)
    labels = tuple(range(p)) if entities is None else tuple(entities)
    if len(labels) != p:
        raise ValueError(f"`entities` has {len(labels)} labels for {p} columns.")
    return WindowStats(
        Z=Z,
        scale=scale,
        mean=mean,
        n_eff=n_eff,
        coverage=count / float(n),
        space=space,
        entities=labels,
        asof=asof,
        kept=kept,
    )


def min_side_gram(Z: NDArray[np.float64], n_eff: float) -> NDArray[np.float64]:
    """``Z Z^T / n_eff`` if ``p > n`` else ``Z^T Z / n_eff``, symmetrised."""
    n, p = Z.shape
    G = (Z @ Z.T) if p > n else (Z.T @ Z)
    G /= n_eff
    return 0.5 * (G + G.T)


def orient_signs(V: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fix eigenvector signs in place and return ``V``.

    Column 0 (the market mode) is oriented so that its entries sum to a
    positive number -- "market up" is the positive direction (a documented
    deviation from :func:`panelary.reduce._common.fix_signs`). Every other
    column, and column 0 when its sum is exactly 0, follows ``fix_signs``:
    the largest-magnitude entry is positive, ties to the first occurrence.
    """
    if V.shape[1] == 0:
        return V
    idx = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[idx, np.arange(V.shape[1])])
    s[s == 0.0] = 1.0
    total = float(V[:, 0].sum())
    if total != 0.0:
        s[0] = 1.0 if total > 0.0 else -1.0
    V *= s
    return V


def eigen(ws: WindowStats, *, vectors: bool = False) -> Spectrum:
    """Eigendecompose the window's min-side Gram (``eigvalsh`` unless vectors).

    Returns the descending spectrum of ``S = Z^T Z / n_eff`` (``min(n, p)``
    values, the rest are 0) and, with ``vectors=True``, orthonormal primal
    eigenvectors for its non-null part, sign-fixed by :func:`orient_signs`.
    """
    G = ws.gram()
    n, p = ws.n, ws.p
    if vectors:
        w, U = np.linalg.eigh(G)
        w = w[::-1].copy()
        U = U[:, ::-1]
    else:
        w = np.linalg.eigvalsh(G)[::-1].copy()
        U = None
    top = max(float(w[0]), 0.0) if w.size else 0.0
    tol = max(n, p) * np.finfo(np.float64).eps * top
    w[w <= tol] = 0.0
    V = None
    if U is not None:
        r = int(np.count_nonzero(w > 0.0))
        if ws.dual:
            V = ws.Z.T @ (U[:, :r] / np.sqrt(w[:r] * ws.n_eff))
            norms = np.sqrt(np.einsum("ij,ij->j", V, V))
            if r and np.abs(norms - 1.0).max() > _ORTHO_TOL:
                V = np.linalg.qr(V)[0]
        else:
            V = np.ascontiguousarray(U[:, :r])
        V = orient_signs(V)
    return Spectrum(values=w, vectors=V, p=p, n_rows=n, n_eff=ws.n_eff)
