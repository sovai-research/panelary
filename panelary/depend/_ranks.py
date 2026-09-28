"""The shared rank primitive layer of :mod:`panelary.depend`.

Every rank-based statistic in the package bottoms out here. Four rules, each
with a measured reason:

* **One argsort, then a scatter.** :func:`ranks` sorts once and scatters the
  sorted positions back with :func:`numpy.put_along_axis`; it never writes
  ``argsort(argsort(x))``. On a ``(200, 2441, 60)`` stack the scatter took
  720 ms against 1176 ms for the double argsort (1.63x), and the rolling
  statistics call this in their inner loop.
* **Batched over leading axes.** ``ranks`` works on the last axis of any
  array, so ``(n_entities, n_windows, window)`` is one call.
* **Deterministic ties.** ``method="average"`` (the default) gives tied values
  their mean rank -- no uniform-random tie-breaking, so two calls on the same
  frame return the same bits.
* **float64 / int64.** Average ranks are float64; ordinal, dense, min and max
  ranks are int64.

:func:`dominance_counts` is the bivariate rank ``Q_i = #{j : x_j < x_i and
y_j < y_i}`` that powers Hoeffding's D, Kendall's tau and the ``O(n log n)``-
shaped univariate distance covariance. It is the *blocked* ``O(n^1.5)``
algorithm the build contract asks for first: sort by ``x``, cut the sorted
order into blocks of ``ceil(sqrt(n))``, count within a block by direct
comparison and across blocks from a cumulative histogram over ``y``-ranks. No
Python loop runs over ``n`` -- only over the ``sqrt(n)`` blocks.
"""

from __future__ import annotations

import math

import numpy as np

from panelary._internal._numpy_stats import norm_ppf

__all__ = [
    "dominance_counts",
    "lag1_rank_autocorr",
    "normal_scores",
    "pairwise_complete",
    "ranks",
    "sliding_ranks",
]

_RANK_METHODS = frozenset({"average", "ordinal", "dense", "min", "max"})


def _group_bounds(srt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tie-group start/end positions for an array already sorted on its last axis.

    Returns ``(new, start, end)``: ``new`` flags the first element of each run of
    equal values, ``start``/``end`` hold (0-based) the first and last position
    of the run each element belongs to.
    """
    n = srt.shape[-1]
    new = np.ones(srt.shape, dtype=bool)
    if n > 1:
        new[..., 1:] = srt[..., 1:] != srt[..., :-1]
    pos = np.broadcast_to(np.arange(n, dtype=np.int64), srt.shape)
    start = np.maximum.accumulate(np.where(new, pos, 0), axis=-1)
    last = np.ones(srt.shape, dtype=bool)
    if n > 1:
        last[..., :-1] = new[..., 1:]
    rev = np.where(last, pos, n - 1)[..., ::-1]
    end = np.minimum.accumulate(rev, axis=-1)[..., ::-1]
    return new, start, end


def ranks(x: np.ndarray, *, axis: int = -1, method: str = "average") -> np.ndarray:
    """Rank ``x`` along ``axis`` (1-based), batched over every other axis.

    Parameters
    ----------
    x : array_like
        Values to rank. Must be free of NaN on the ranked axis (drop them first
        with :func:`pairwise_complete`; :func:`sliding_ranks` handles NaN
        windows itself).
    axis : int, default=-1
        Axis to rank along.
    method : {"average", "ordinal", "dense", "min", "max"}, default="average"
        Tie handling, with SciPy's ``rankdata`` meanings. ``"average"`` is the
        deterministic default of the whole package.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``; float64 for ``"average"``, int64 otherwise.

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    Notes
    -----
    One stable argsort plus one :func:`numpy.put_along_axis` scatter -- never
    ``argsort(argsort(x))`` (measured 1.63x slower on the rolling workload).

    Examples
    --------
    >>> ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist()
    [3.5, 1.0, 3.5, 2.0]
    """
    if method not in _RANK_METHODS:
        raise ValueError(
            f"unknown rank `method` {method!r}; expected one of {sorted(_RANK_METHODS)}."
        )
    arr = np.asarray(x, dtype=np.float64)
    moved = np.moveaxis(arr, axis, -1)
    n = moved.shape[-1]
    out_dtype = np.float64 if method == "average" else np.int64
    if n == 0:
        return np.empty(arr.shape, dtype=out_dtype)
    order = np.argsort(moved, axis=-1, kind="stable")
    if method == "ordinal":
        ranked = np.broadcast_to(np.arange(1, n + 1, dtype=np.int64), moved.shape)
    else:
        srt = np.take_along_axis(moved, order, axis=-1)
        new, start, end = _group_bounds(srt)
        if method == "dense":
            ranked = np.cumsum(new, axis=-1, dtype=np.int64)
        elif method == "min":
            ranked = start + 1
        elif method == "max":
            ranked = end + 1
        else:
            ranked = 0.5 * (start + end).astype(np.float64) + 1.0
    out = np.empty(moved.shape, dtype=out_dtype)
    np.put_along_axis(out, order, ranked, axis=-1)
    return np.moveaxis(out, -1, axis)


def normal_scores(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Copula (van der Waerden) transform: ``Phi^{-1}(rank / (n + 1))``.

    Average ranks, so ties map to one score. The result has an exactly standard
    normal *marginal* whatever the input distribution, which is what
    :func:`~panelary.depend.gcmi` needs.

    Parameters
    ----------
    x : array_like
        Values, NaN-free along ``axis``.
    axis : int, default=-1
        Axis to transform along.

    Returns
    -------
    numpy.ndarray
        float64 normal scores, same shape as ``x``.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[axis] if arr.ndim else 1
    r = ranks(arr, axis=axis, method="average")
    if n > 0 and r.size > 4 * n:
        # Average ranks along an axis of length n are multiples of 1/2 in
        # [1, n]: evaluate Phi^{-1} once on that grid and gather (batched
        # windows would otherwise pay one norm_ppf per element).
        grid = np.arange(2, 2 * n + 1, dtype=np.float64) / 2.0
        table = np.asarray(norm_ppf(grid / (n + 1.0)), dtype=np.float64)
        return table[np.rint(2.0 * r).astype(np.int64) - 2]
    u = r / (n + 1.0)
    return np.asarray(norm_ppf(u.ravel()), dtype=np.float64).reshape(u.shape)


def sliding_ranks(x: np.ndarray, window: int) -> np.ndarray:
    """Ranks inside every trailing window of length ``window`` (last axis).

    Parameters
    ----------
    x : array_like
        ``(..., n)`` series.
    window : int
        Window length ``w >= 1``. A fixed integer -- never derived from ``n``.

    Returns
    -------
    numpy.ndarray
        ``(..., n - w + 1, w)`` float64 average ranks; row ``k`` ranks
        ``x[..., k : k + w]`` *within that window only*. A window containing a
        NaN yields an all-NaN row rather than silently ranking the NaN last.

    Raises
    ------
    ValueError
        If ``window`` is not a positive integer or exceeds ``n``.
    """
    w = int(window)
    if w < 1:
        raise ValueError(f"`window` must be a positive integer, got {window!r}.")
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[-1] < w:
        raise ValueError(f"`window`={w} exceeds the series length {arr.shape[-1]}.")
    view = np.lib.stride_tricks.sliding_window_view(arr, w, axis=-1)
    bad = np.isnan(view).any(axis=-1)
    r = ranks(np.where(np.isnan(view), 0.0, view), axis=-1, method="average")
    if bad.any():
        r[bad] = np.nan
    return r


def _dominance_sums(ry: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``D[b, i] = sum_{j < i, ry[b, j] < ry[b, i]} w[b, j]`` -- blocked, batched.

    ``ry`` is ``(B, m)`` integer ranks **in position order** (ties may share a
    rank; the comparison is strict); ``w`` is ``(B, m, k)`` float64 weights.
    Cost ``O(B k m^1.5)`` with a Python loop over ``ceil(sqrt(m))`` blocks only.
    """
    ry = np.asarray(ry, dtype=np.int64)
    bsz, m = ry.shape
    k = w.shape[-1]
    out = np.zeros((bsz, m, k), dtype=np.float64)
    if m < 2:
        return out
    block = max(1, math.isqrt(m - 1) + 1)
    # hist[b, r + 1] accumulates the weight of processed elements with rank r, so
    # cumsum(hist)[b, r] is the weight of processed elements with rank < r.
    hist = np.zeros((bsz, m + 1, k), dtype=np.float64)
    rows = np.arange(bsz)[:, None]
    for s in range(0, m, block):
        e = min(s + block, m)
        r_blk = ry[:, s:e]
        w_blk = w[:, s:e]
        if s > 0:
            cum = np.cumsum(hist, axis=1)
            out[:, s:e] = cum[rows, r_blk]
        nb = e - s
        if nb > 1:
            lower = r_blk[:, None, :] < r_blk[:, :, None]  # [b, i, j]: r_j < r_i
            lower &= np.tri(nb, nb, -1, dtype=bool)[None]  # j before i
            out[:, s:e] += np.einsum("bij,bjk->bik", lower.astype(np.float64), w_blk)
        np.add.at(hist, (rows, r_blk + 1), w_blk)
    return out


def dominance_counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bivariate rank ``Q_i = #{j != i : x_j < x_i and y_j < y_i}``.

    Both comparisons are **strict**, so tied pairs never count. Blocked
    ``O(n^1.5)`` (see module notes); no ``n x n`` matrix is formed.

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length, NaN-free.

    Returns
    -------
    numpy.ndarray
        int64 counts, in the input order.

    Raises
    ------
    ValueError
        If the shapes differ or the inputs are not 1-D.

    Examples
    --------
    >>> dominance_counts(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0])).tolist()
    [0, 1, 1]
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or xa.shape != ya.shape:
        raise ValueError(
            f"`x` and `y` must be 1-D and equal length, got {xa.shape} and {ya.shape}."
        )
    n = xa.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # Sort by x ascending and, within tied x, by y DESCENDING: a tied-x
    # predecessor then never has a strictly smaller y, so it is never counted.
    order = np.lexsort((-ya, xa))
    ry = ranks(ya, method="min")[order] - 1
    d = _dominance_sums(ry[None, :], np.ones((1, n, 1)))[0, :, 0]
    out = np.empty(n, dtype=np.int64)
    out[order] = np.rint(d).astype(np.int64)
    return out


def pairwise_complete(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop every row where ``x`` or ``y`` is NaN or infinite.

    Parameters
    ----------
    x, y : array_like
        ``(n,)`` or ``(n, d)`` arrays with the same number of rows. A 2-D row is
        dropped if *any* of its entries is non-finite.

    Returns
    -------
    x, y : numpy.ndarray
        float64 copies restricted to the complete rows.
    n_obs : int
        Number of surviving rows. Every public statistic reports it.

    Raises
    ------
    ValueError
        If the row counts differ.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape[0] != ya.shape[0]:
        raise ValueError(
            f"`x` and `y` must have the same number of rows, got {xa.shape[0]} "
            f"and {ya.shape[0]}."
        )
    fx = np.isfinite(xa) if xa.ndim == 1 else np.isfinite(xa).all(axis=1)
    fy = np.isfinite(ya) if ya.ndim == 1 else np.isfinite(ya).all(axis=1)
    keep = fx & fy
    n_obs = int(keep.sum())
    return xa[keep], ya[keep], n_obs


def lag1_rank_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of the ranks of ``x`` (NaN dropped first).

    The cheap serial-dependence pre-check behind ``null="auto"`` and the
    ``null="iid"`` warning: ``|rho| > 0.2`` routes a test to a
    serial-dependence-valid null. Returns ``nan`` below 4 finite values.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    xa = xa[np.isfinite(xa)]
    if xa.size < 4:
        return float("nan")
    r = ranks(xa)
    a = r[:-1] - r[:-1].mean()
    b = r[1:] - r[1:].mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else 0.0
