"""Small dense linear-algebra helpers shared across subpackages.

Leaf module: standard library and numpy only (``_internal/README.md`` rule 1).

* :func:`psd_repair` -- nearest-by-clipping PSD correlation matrix. It lived in
  :mod:`panelary.depend._matrix`, which re-exports this object, so every
  existing caller gets bit-identical results (coordination note D6).
* :func:`solve_vec` -- ``A^{-1} b`` for a single right-hand side, written the
  way that is unambiguous under NumPy 2's batched ``solve`` semantics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["psd_repair", "solve_vec"]


def psd_repair(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest-by-clipping PSD correlation matrix and the smallest eigenvalue
    **before** repair.

    A pairwise-complete (or nonlinear) dependence matrix need not be positive
    semi-definite. Negative eigenvalues are clipped to zero and the result is
    rescaled to a unit diagonal.

    Returns
    -------
    (repaired, min_eigenvalue)
    """
    A = np.asarray(M, dtype=np.float64)
    A = 0.5 * (A + A.T)
    A = np.where(np.isfinite(A), A, 0.0)
    np.fill_diagonal(A, 1.0)
    w, v = np.linalg.eigh(A)
    lam_min = float(w.min())
    if lam_min >= 0:
        return A, lam_min
    R = (v * np.clip(w, 0.0, None)) @ v.T
    d = np.sqrt(np.clip(np.diag(R), 1e-300, None))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R, lam_min


def solve_vec(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``A x = b`` for a 1-D ``b`` (or column-wise for a 2-D ``b``).

    ``np.linalg.solve(A, b)`` with a 1-D ``b`` whose length happens to equal a
    batch dimension is ambiguous under NumPy 2; ``b[:, None]`` is not.
    """
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        return np.linalg.solve(A, b[:, None])[:, 0]
    return np.linalg.solve(A, b)
