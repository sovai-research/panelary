"""Missing-data policies for one window (plan section 5.11).

Admission to the as-of universe is listwise by entity (``min_coverage`` of the
window, decided in :mod:`panelary.covariance._window`). Within the admitted
entities this module decides what a missing cell contributes.

Only ``zero_after_demean`` ships in M1: centre each column on its own observed
mean (corrected two-pass, so no raw-sum cancellation), then set missing cells
to 0. The estimate stays positive semi-definite and on the dual fast path.
With independent missingness a correlation is attenuated by about
``sqrt(f_i f_j)`` (``f`` = coverage), so ``min_coverage=0.95`` bounds the
attenuation at 5%.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["MISSING_POLICIES", "zero_after_demean"]

MISSING_POLICIES = ("zero_after_demean",)


def zero_after_demean(
    X: Any,
    *,
    assume_centered: bool = False,
    count: NDArray[np.int64] | None = None,
    copy: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Centre each column on its observed mean, then zero the missing cells.

    Parameters
    ----------
    X : array_like, shape (n, p)
        One window; non-finite cells are missing. Float32 is upcast.
    assume_centered : bool, default False
        Skip the centring (the mean is taken to be 0).
    count : ndarray of int64, optional
        Observed cells per column, when the caller already knows them (the
        as-of engine reads them off its prefix counts). Must be exact.
    copy : bool, default True
        ``False`` works in place on ``X``, which must then be a C-contiguous
        float64 array the caller owns. The arithmetic is identical.

    Returns
    -------
    (Xc, mean, count)
        ``Xc`` is a fresh C-contiguous float64 copy with missing cells 0;
        ``mean`` the removed per-column means; ``count`` the observed cells
        per column.
    """
    if copy:
        A = np.array(X, dtype=np.float64, order="C", copy=True)
    else:
        A = X
        if not (A.dtype == np.float64 and A.flags.c_contiguous):
            raise ValueError("copy=False needs a C-contiguous float64 array.")
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D (n, p) window, got shape {A.shape}.")
    n = A.shape[0]
    finite = None
    if count is None:
        finite = np.isfinite(A)
        count = np.count_nonzero(finite, axis=0).astype(np.int64)
    else:
        count = np.asarray(count, dtype=np.int64)
    all_finite = bool(count.size == 0 or int(count.min()) == n)
    if not all_finite:
        if finite is None:
            finite = np.isfinite(A)
        A[~finite] = 0.0
    if assume_centered:
        mean: NDArray[np.float64] = np.zeros(A.shape[1], dtype=np.float64)
        return A, mean, count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = A.sum(axis=0) / count
    mean = np.where(count > 0, mean, 0.0)
    A -= mean
    if finite is not None and not all_finite:
        A[~finite] = 0.0
    # Corrected two-pass (Chan, Golub & LeVeque 1983): the first mean of data
    # at a large offset carries the summation error of numbers ~offset; the
    # residuals are small and exactly representable, so re-centring them
    # removes that error instead of rounding it back into an offset-sized mean.
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = A.sum(axis=0) / count
    resid = np.where(count > 0, resid, 0.0)
    A -= resid
    if finite is not None and not all_finite:
        A[~finite] = 0.0
    return A, mean + resid, count
