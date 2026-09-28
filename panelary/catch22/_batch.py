"""Batched catch22: every feature over an ``(n_windows, L)`` array at once.

The scalar feature functions in :mod:`panelary.catch22._features` are thin
wrappers over the kernels here, so there is one implementation of each feature
and the scalar and batched entry points agree by construction.

Design rules
------------
* **Row independence.** Every kernel computes row ``i`` from row ``i`` alone:
  per-row reductions along the contiguous last axis, elementwise arithmetic, and
  never a BLAS product whose blocking could depend on how many rows share the
  call. A window's features therefore do not depend on what else is in the
  batch (asserted in ``tests/test_catch22_batch.py``), which is what makes the
  embedding layer's "same window, same numbers" promise hold.
* **Decisions mirror NumPy exactly.** Features whose output is a discrete
  decision (a lag, a run length, a bin, a count) reproduce the exact comparisons
  the scalar code made -- histogram edges and bin assignment follow
  :func:`numpy.histogram`'s own algorithm, quantiles go through
  :func:`numpy.quantile` along the row axis, and compacted sums are summed in
  the same order as the 1-D call they replace.
* **Continuous features may differ in the last bits** from the pre-batch
  implementation where a BLAS/LAPACK call (``lstsq``, ``polyfit``,
  ``corrcoef``, ``cov``) was replaced by an explicit per-row reduction. The
  drift is bounded by the tolerances that ``tests/test_perf_parity_vectorization.py``
  already derives for exactly this situation.

Rows containing NaN are computed one at a time on the NaN-dropped series
(the scalar contract); NaN-free rows are computed together in chunks of
:data:`_CHUNK_ROWS`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._helpers import _bspline_design

__all__ = ["catch22_rows", "feature_rows"]

F64 = NDArray[np.float64]

#: Rows per kernel call. Row-independent kernels make chunking invisible in the
#: output; the chunk only bounds the scratch memory (~15 MB at L=128).
_CHUNK_ROWS = 2048

#: Below this many rows the per-row scan of ``DN_OutlierInclude`` is cheaper
#: than the vectorised linked-list sweep (which costs O(L) NumPy calls however
#: few rows there are). Both are exact, so the switch is invisible.
_OUTLIER_SWEEP_MIN_ROWS = 16

_OUTLIER_INC = 0.01


# ---------------------------------------------------------------------------
# Shared intermediates
# ---------------------------------------------------------------------------
def _zscore_rows(X: F64) -> F64:
    """Row-wise :func:`panelary.catch22._helpers._zscore` (``ddof=1``).

    Degenerate rows (non-finite or ``< 1e-12`` spread) become all zeros, as in
    the scalar helper.
    """
    n, L = X.shape
    if L == 0:
        return X.copy()
    mean = X.mean(axis=1)
    std = X.std(axis=1, ddof=1) if L > 1 else np.zeros(n)
    bad = ~np.isfinite(std) | (std < 1e-12)
    with np.errstate(invalid="ignore", over="ignore"):
        Y = (X - mean[:, None]) / np.where(bad, 1.0, std)[:, None]
    Y[bad] = 0.0
    return Y


def _power(z: NDArray[np.complex128]) -> F64:
    """``|z|**2`` as ``re**2 + im**2`` -- deliberately not ``(z * conj(z)).real``.

    NumPy's SIMD complex multiply fuses ``a*c - b*d`` into an FMA in the vector
    body but not in the scalar tail, so ``z * conj(z)`` rounds an element
    differently depending on *where in the array* it lands. That made a row's
    ACF depend on how many rows shared the call (measured: 10 of 400 rows at
    L=20 moved by one ulp). Two squares and an add have no fusion opportunity,
    so every element is rounded the same way wherever it sits, on every
    platform. The imaginary part of ``|z|**2`` is exactly zero by definition,
    rather than the FMA's rounding residue.
    """
    return z.real * z.real + z.imag * z.imag


def _acf_rows(Y: F64) -> F64:
    """Row-wise :func:`panelary.catch22._helpers._acf` (FFT, biased, ``acf[0]=1``)."""
    n, L = Y.shape
    if L == 0:
        return np.ones((n, 1))
    Yc = Y - Y.mean(axis=1)[:, None]
    nfft = int(2 ** np.ceil(np.log2(2 * L - 1))) if L > 1 else 1
    f = np.fft.fft(Yc, n=nfft, axis=1)
    acov = np.fft.ifft(_power(f), axis=1)[:, :L].real
    a0 = acov[:, 0]
    zero = a0 == 0
    out = acov / np.where(zero, 1.0, a0)[:, None]
    if zero.any():
        out[zero] = 0.0
        out[zero, 0] = 1.0
    return out


def _first_zero_rows(acf: F64) -> NDArray[np.int64]:
    """First lag ``>= 1`` with ``acf <= 0``; the ACF length if none (per row)."""
    m = acf.shape[1]
    le = acf[:, 1:] <= 0.0
    has = le.any(axis=1)
    return np.where(has, np.argmax(le, axis=1) + 1, m).astype(np.int64)


def _longest_run_rows(M: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Length of the longest run of ``True`` in each row (0 if none)."""
    n, L = M.shape
    if L == 0:
        return np.zeros(n, dtype=np.int64)
    pos = np.arange(1, L + 1)
    last_false = np.maximum.accumulate(np.where(M, 0, pos), axis=1)
    return (pos - last_false).max(axis=1).astype(np.int64)


def _rowsum_compact(T: F64, M: NDArray[np.bool_]) -> F64:
    """``np.sum(T[i][M[i]])`` for every row, summed in the 1-D call's order.

    The masked entries are moved (stably) to the front of each row and rows with
    the same count are summed together over exactly that many contiguous
    columns, which is the same reduction :func:`numpy.sum` performs on the
    compacted 1-D array -- so the result is bit-identical to the scalar code,
    not merely close.
    """
    n = T.shape[0]
    counts = M.sum(axis=1)
    order = np.argsort(~M, axis=1, kind="stable")
    C = np.take_along_axis(T, order, axis=1)
    out = np.zeros(n)
    for c in np.unique(counts):
        if c == 0:
            continue
        rows = np.flatnonzero(counts == c)
        out[rows] = C[rows, :c].sum(axis=1)
    return out


def _uniform_hist(V: F64, lo: F64, hi: F64, n_bins: int) -> tuple[F64, F64]:
    """Per-row equal-width histogram, exactly as ``np.histogram(v, bins=n_bins)``.

    ``lo``/``hi`` are each row's min/max (``lo < hi``). Edges replicate
    :func:`numpy.linspace` (``i * step + lo``, last edge pinned to ``hi``) and
    the bin assignment replicates NumPy's uniform-bin fast path, including its
    one-ULP boundary corrections, so the counts are identical.
    """
    g, m = V.shape
    step = (hi - lo) / n_bins
    edges = np.arange(n_bins + 1, dtype=np.float64)[None, :] * step[:, None]
    edges += lo[:, None]
    edges[:, -1] = hi
    idx = (((V - lo[:, None]) / (hi - lo)[:, None]) * n_bins).astype(np.intp)
    idx[idx == n_bins] -= 1
    dec = np.take_along_axis(edges, idx, axis=1) > V
    idx[dec] -= 1
    inc = (np.take_along_axis(edges, idx + 1, axis=1) <= V) & (idx != n_bins - 1)
    idx[inc] += 1
    flat = (np.arange(g)[:, None] * n_bins + idx).ravel()
    counts = np.bincount(flat, minlength=g * n_bins).reshape(g, n_bins)
    return counts.astype(np.float64), edges


def _coarsegrain3_rows(Y: F64) -> NDArray[np.intp]:
    """Row-wise :func:`panelary.catch22._helpers._coarsegrain_quantile` (3 groups)."""
    qs = np.quantile(Y, np.arange(1, 3) / 3, axis=1, method="linear")
    labels = (qs[0][:, None] <= Y).astype(np.intp) + (qs[1][:, None] <= Y)
    return np.clip(labels, 0, 2)


def _pair_counts3_rows(S: NDArray[np.intp]) -> F64:
    """Row-wise 3x3 consecutive-symbol pair counts, flattened to ``(n, 9)``."""
    n = S.shape[0]
    flat = (np.arange(n)[:, None] * 9 + S[:, :-1] * 3 + S[:, 1:]).ravel()
    return np.bincount(flat, minlength=n * 9).reshape(n, 9).astype(np.float64)


class _Batch:
    """Lazily computed intermediates shared by the kernels for one batch."""

    def __init__(self, X: F64) -> None:
        self.X = X
        self.n, self.L = X.shape

    @cached_property
    def Y(self) -> F64:
        return _zscore_rows(self.X)

    @cached_property
    def acf(self) -> F64:
        return _acf_rows(self.Y)

    @cached_property
    def first_zero(self) -> NDArray[np.int64]:
        return _first_zero_rows(self.acf)

    @cached_property
    def dY(self) -> F64:
        return np.diff(self.Y, axis=1)

    @cached_property
    def fluct(self) -> tuple[F64, F64]:
        return _fluctuation_rows(self.Y)

    @cached_property
    def welch(self) -> tuple[F64, F64, NDArray[np.bool_]] | None:
        return _welch_cumulative_rows(self.Y)


# ---------------------------------------------------------------------------
# Distribution features
# ---------------------------------------------------------------------------
def _k_histogram_mode(b: _Batch, n_bins: int) -> F64:
    return _histogram_mode_rows(b.Y, n_bins)


def _histogram_mode_rows(Y: F64, n_bins: int) -> F64:
    """Row-wise histogram mode of already z-scored rows (NaN if constant)."""
    n, L = Y.shape
    out = np.full(n, np.nan)
    if L == 0:
        return out
    lo = Y.min(axis=1)
    hi = Y.max(axis=1)
    rows = np.flatnonzero(hi != lo)
    if rows.size == 0:
        return out
    counts, edges = _uniform_hist(Y[rows], lo[rows], hi[rows], n_bins)
    centers = (edges[:, :-1] + edges[:, 1:]) / 2.0
    is_max = counts == counts.max(axis=1)[:, None]
    single = is_max.sum(axis=1) == 1
    res = np.empty(rows.size)
    res[single] = centers[single, np.argmax(is_max[single], axis=1)]
    for i in np.flatnonzero(~single):  # ties: the scalar mean, verbatim
        res[i] = float(centers[i][is_max[i]].mean())
    out[rows] = res
    return out


def _k_trev(b: _Batch) -> F64:
    if b.L < 2:
        return np.full(b.n, np.nan)
    return np.mean(b.dY**3, axis=1)


def _k_pnn40(b: _Batch) -> F64:
    if b.L < 2:
        return np.full(b.n, np.nan)
    return np.mean(np.abs(b.dY) > 0.04, axis=1)


# ---------------------------------------------------------------------------
# Autocorrelation-based features
# ---------------------------------------------------------------------------
def _k_f1ecac(b: _Batch) -> F64:
    acf = b.acf
    m = acf.shape[1]
    thresh = 1.0 / np.e
    below = acf[:, 1:] < thresh
    has = below.any(axis=1)
    out = np.full(b.n, float(m))
    rows = np.flatnonzero(has)
    if rows.size:
        i = np.argmax(below[rows], axis=1)
        a_i = acf[rows, i]
        slope = acf[rows, i + 1] - a_i
        flat = slope == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            interp = i + (thresh - a_i) / slope
        out[rows] = np.where(flat, i.astype(np.float64), interp)
    return out


def _k_firstmin_ac(b: _Batch) -> F64:
    acf = b.acf
    m = acf.shape[1]
    if m < 3:
        return np.full(b.n, float(m))
    mid = acf[:, 1:-1]
    minima = (mid < acf[:, :-2]) & (mid < acf[:, 2:])
    has = minima.any(axis=1)
    return np.where(has, np.argmax(minima, axis=1) + 1, m).astype(np.float64)


def _k_histogram_ami(b: _Batch) -> F64:
    Y, n, L = b.Y, b.n, b.L
    tau, n_bins = 2, 5
    out = np.full(n, np.nan)
    if tau + 1 >= L:
        return out
    fmin = Y.min(axis=1)
    fmax = Y.max(axis=1)
    step = (fmax - fmin + 0.2) / n_bins
    rows = np.flatnonzero(step > 0)
    if rows.size == 0:
        return out
    Yr = Y[rows]
    g = rows.size
    edges = (fmin[rows] - 0.1)[:, None] + np.arange(n_bins + 1)[None, :] * step[
        rows, None
    ]

    def _bin(v: F64) -> NDArray[np.intp]:
        # np.histogramdd with explicit edges: searchsorted(side="right"), with
        # values sitting exactly on the last edge moved into the last bin.
        idx = (v[:, :, None] >= edges[:, None, :]).sum(axis=2) - 1
        idx[v == edges[:, -1:]] -= 1
        return idx

    i1 = _bin(Yr[:, :-tau])
    i2 = _bin(Yr[:, tau:])
    keep = (i1 >= 0) & (i1 < n_bins) & (i2 >= 0) & (i2 < n_bins)
    cell = np.arange(g)[:, None] * (n_bins * n_bins) + i1 * n_bins + i2
    counts = np.bincount(cell[keep], minlength=g * n_bins * n_bins).astype(np.float64)
    joint = counts.reshape(g, n_bins, n_bins)
    p = joint / joint.reshape(g, -1).sum(axis=1)[:, None, None]
    pi = p.sum(axis=2, keepdims=True)
    pj = p.sum(axis=1, keepdims=True)
    denom = pi * pj
    mask = (p > 0) & (denom > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mask, p * np.log(np.where(mask, p / denom, 1.0)), 0.0)
    out[rows] = _rowsum_compact(terms.reshape(g, -1), mask.reshape(g, -1))
    return out


def _k_ami_gaussian(b: _Batch) -> F64:
    Y, n, L = b.Y, b.n, b.L
    out = np.full(n, np.nan)
    tau_max = min(40, math.ceil(L / 2))
    if tau_max < 2:
        return out
    ami = np.empty((n, tau_max))
    for k in range(1, tau_max + 1):
        m = L - k
        if m < 2:
            ami[:, k - 1] = 0.0
            continue
        a = Y[:, :-k]
        c = Y[:, k:]
        ac = a - a.mean(axis=1)[:, None]
        cc = c - c.mean(axis=1)[:, None]
        saa = (ac * ac).sum(axis=1)
        scc = (cc * cc).sum(axis=1)
        sac = (ac * cc).sum(axis=1)
        zero = (saa == 0) | (scc == 0)
        # np.corrcoef: cov scaled by 1/(m-1), then divided by each stddev.
        f = np.true_divide(1, m - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (sac * f) / np.sqrt(saa * f) / np.sqrt(scc * f)
        r = np.minimum(np.maximum(r, -0.9999999), 0.9999999)
        with np.errstate(invalid="ignore"):
            ami[:, k - 1] = np.where(zero, 0.0, -0.5 * np.log(1.0 - r * r))
    if tau_max < 3:
        return np.full(n, float(tau_max))
    mid = ami[:, 1:-1]
    minima = (mid < ami[:, :-2]) & (mid < ami[:, 2:])
    has = minima.any(axis=1)
    return np.where(has, np.argmax(minima, axis=1) + 1, tau_max).astype(np.float64)


def _k_embed2(b: _Batch) -> F64:
    Y, n, L = b.Y, b.n, b.L
    out = np.full(n, np.nan)
    tau = b.first_zero.copy()
    cap = L // 10
    tau[tau > cap] = cap
    ok = (tau >= 1) & (L - tau - 1 >= 2)
    for t in np.unique(tau[ok]):
        t = int(t)
        rows = np.flatnonzero(ok & (tau == t))
        Yr = Y[rows]
        dx = Yr[:, 1 : L - t] - Yr[:, 0 : L - t - 1]
        dy = Yr[:, 1 + t : L] - Yr[:, t : L - 1]
        d = np.sqrt(dx * dx + dy * dy)
        m = d.shape[1]
        if m < 2:
            continue
        lam = d.mean(axis=1)
        std = d.std(axis=1, ddof=1)
        res = np.full(rows.size, np.nan)
        positive = lam > 0
        res[positive & (std < 1e-3)] = 0.0
        calc = np.flatnonzero(positive & ~(std < 1e-3))
        if calc.size:
            span = d[calc].max(axis=1) - d[calc].min(axis=1)
            nb = np.ceil(span / (3.5 * std[calc] / m ** (1.0 / 3.0))).astype(np.int64)
            for nbv in np.unique(nb):
                sub = calc[nb == nbv]
                if nbv <= 0:
                    res[sub] = 0.0
                    continue
                D = d[sub]
                counts, edges = _uniform_hist(D, D.min(axis=1), D.max(axis=1), int(nbv))
                norm = counts / m
                centers = (edges[:, :-1] + edges[:, 1:]) / 2.0
                widths = np.diff(edges, axis=1)
                lam_s = lam[sub][:, None]
                exp_pdf = np.exp(-centers / lam_s) / lam_s
                exp_pdf[exp_pdf < 0] = 0.0
                res[sub] = np.abs(norm / widths - exp_pdf).mean(axis=1)
        out[rows] = res
    return out


# ---------------------------------------------------------------------------
# Binary / symbolic features
# ---------------------------------------------------------------------------
def _k_mean_longstretch1(b: _Batch) -> F64:
    if b.L == 0:
        return np.full(b.n, np.nan)
    Y = b.Y
    return _longest_run_rows(Y.mean(axis=1)[:, None] < Y).astype(np.float64)


def _k_diff_longstretch0(b: _Batch) -> F64:
    if b.L < 2:
        return np.full(b.n, np.nan)
    return _longest_run_rows(b.dY <= 0).astype(np.float64)


def _k_motif_three(b: _Batch) -> F64:
    if b.L < 3:
        return np.full(b.n, np.nan)
    counts = _pair_counts3_rows(_coarsegrain3_rows(b.Y))
    p = counts / counts.sum(axis=1)[:, None]
    nz = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(nz, p * np.log(np.where(nz, p, 1.0)), 0.0)
    return -_rowsum_compact(terms, nz)


def _k_transition_matrix(b: _Batch) -> F64:
    Y, n = b.Y, b.n
    out = np.full(n, np.nan)
    tau = np.maximum(b.first_zero, 1)
    for t in np.unique(tau):
        t = int(t)
        rows = np.flatnonzero(tau == t)
        yd = np.ascontiguousarray(Y[rows][:, ::t])
        m = yd.shape[1]
        if m < 4:
            continue
        T = _pair_counts3_rows(_coarsegrain3_rows(yd)).reshape(-1, 3, 3)
        T /= m - 1
        # trace(np.cov(T, rowvar=False)): the ddof=1 variance of each column.
        dev = T - T.mean(axis=1, keepdims=True)
        diag = (dev * dev).sum(axis=1) * np.true_divide(1, 2)
        out[rows] = diag.sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Periodicity / forecasting features
# ---------------------------------------------------------------------------
_SPLINE_BASIS: dict[int, F64] = {}


def _spline_basis(L: int) -> F64:
    """Orthonormal basis of the cubic-spline design's column space (cached per L).

    ``PD_PeriodicityWang`` detrends with a least-squares cubic spline on 3
    interior knots; the fitted values are the projection of ``y`` onto the
    design's column space, ``U_r U_r^T y``. Rank is truncated with LAPACK
    ``lstsq``'s default cut-off, so a rank-deficient design is handled the same
    way.
    """
    basis = _SPLINE_BASIS.get(L)
    if basis is None:
        t = np.arange(L, dtype=float)
        interior = np.linspace(0, L - 1, 5)[1:-1]
        k = 3
        knots = np.concatenate(
            [np.repeat(t[0], k + 1), interior, np.repeat(t[-1], k + 1)]
        )
        design = _bspline_design(t, knots, k)
        if design.shape[1] == 0:
            basis = np.zeros((L, 0))
        else:
            U, s, _ = np.linalg.svd(design, full_matrices=False)
            rcond = np.finfo(np.float64).eps * max(design.shape)
            basis = np.ascontiguousarray(U[:, s > rcond * s.max()])
        _SPLINE_BASIS[L] = basis
    return basis


def _k_periodicity(b: _Batch) -> F64:
    Y, n, L = b.Y, b.n, b.L
    if L < 8:
        return np.full(n, np.nan)
    th = 0.01
    U = _spline_basis(L)  # (L, r)
    # Explicit multiply-and-reduce rather than a BLAS product, so row i's fit
    # cannot depend on how many rows share the call.
    coef = (Y[:, None, :] * U.T[None, :, :]).sum(axis=2)  # (n, r)
    y_spline = (coef[:, None, :] * U[None, :, :]).sum(axis=2)  # (n, L)
    ac_max = math.ceil(L / 3)
    acf = _acf_rows(Y - y_spline)[:, :ac_max]
    m = acf.shape[1]
    if m < 3:
        return np.zeros(n)
    diffs = np.diff(acf, axis=1)
    slope_in, slope_out = diffs[:, :-1], diffs[:, 1:]
    trough = (slope_in < 0) & (slope_out > 0)
    peak = (slope_in > 0) & (slope_out < 0)
    pos = np.arange(1, m - 1)
    last_trough = np.maximum.accumulate(np.where(trough, pos, -1), axis=1)
    has_prior = last_trough >= 0
    a_peak = acf[:, 1:-1]
    a_trough = np.take_along_axis(acf, np.maximum(last_trough, 0), axis=1)
    accepted = peak & has_prior & (a_peak - a_trough >= th) & (a_peak >= 0.0)
    has = accepted.any(axis=1)
    return np.where(has, pos[np.argmax(accepted, axis=1)], 0).astype(np.float64)


def _k_localsimple_mean1(b: _Batch) -> F64:
    if b.L < 3:
        return np.full(b.n, np.nan)
    res = b.Y[:, 1:] - b.Y[:, :-1]
    num = _first_zero_rows(_acf_rows(res)).astype(np.float64)
    den = b.first_zero.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0, np.nan, num / den)


def _k_localsimple_mean3(b: _Batch) -> F64:
    Y, L = b.Y, b.L
    if L < 5:
        return np.full(b.n, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(Y[:, : L - 1], 3, axis=1)
    res = Y[:, 3:] - windows.mean(axis=-1)
    return res.std(axis=1, ddof=1)


# ---------------------------------------------------------------------------
# Outlier-timing features
# ---------------------------------------------------------------------------
def _outlier_include_1d(y: F64, sign: int) -> float:
    """The per-row ``DN_OutlierInclude`` scan (the original scalar algorithm).

    Kept verbatim as the small-batch path; :func:`_outlier_include_sweep` is the
    vectorised equivalent and the two are asserted equal.

    NEEDS REVIEW: the trimming rule (which thresholds contribute to the final
    median) is a faithful best-effort re-derivation; exact parity with the
    reference C implementation is not guaranteed.
    """
    inc = _OUTLIER_INC
    n = y.size
    yw = sign * y
    max_val = yw.max()
    if max_val < inc:
        return 0.0
    n_thresh = int(max_val / inc) + 1
    thresholds = np.arange(n_thresh, dtype=np.float64) * inc
    ascending = np.sort(yw)
    counts = n - np.searchsorted(ascending, thresholds, side="left")
    pct = 100.0 * counts / n
    valid = pct > 2.0
    if not valid.any():
        return np.nan
    kept_thresholds = thresholds[valid]
    _, first_of_count, inverse = np.unique(
        counts[valid], return_index=True, return_inverse=True
    )
    medians = np.fromiter(
        (np.median(np.flatnonzero(yw >= kept_thresholds[j])) for j in first_of_count),
        dtype=np.float64,
        count=first_of_count.size,
    )
    med_pos = medians[inverse.ravel()] / (n / 2.0) - 1.0
    return float(np.nanmedian(med_pos))


def _prefix_medians(order_desc: NDArray[np.intp]) -> F64:
    """Median index of the top-``c`` set, for every ``c = 1..L``, for every row.

    ``order_desc[r]`` lists row ``r``'s positions from largest to smallest value,
    so the top-``c`` set is its first ``c`` entries and ``out[r, c - 1]`` is
    ``np.median`` of those positions. Computed by the reverse sweep: start from
    the full set, whose median is known, and delete positions smallest-value
    first from a doubly linked list threaded through the positions in index
    order. Each deletion moves the lower median by at most one link, so the
    whole table costs O(L) vectorised steps instead of O(L^2) work per row.
    """
    R, L = order_desc.shape
    rows = np.arange(R)
    nodes = np.arange(L + 2)  # node v+1 is position v; 0 and L+1 are sentinels
    prev = np.broadcast_to(nodes - 1, (R, L + 2)).copy()
    nxt = np.broadcast_to(nodes + 1, (R, L + 2)).copy()
    lo_rank = np.full(R, (L - 1) // 2 + 1)
    hi_rank = np.full(R, L // 2 + 1)
    med_lo = np.empty((R, L), dtype=np.int64)
    med_hi = np.empty((R, L), dtype=np.int64)
    med_lo[:, L - 1] = lo_rank - 1
    med_hi[:, L - 1] = hi_rank - 1
    lo = lo_rank
    delete = order_desc[:, ::-1] + 1  # smallest value first
    for s in range(L - 1):
        c = L - s
        v = delete[:, s]
        if c % 2 == 1:
            lo = np.where(v >= lo, prev[rows, lo], lo)
        else:
            lo = np.where(v <= lo, nxt[rows, lo], lo)
        p = prev[rows, v]
        q = nxt[rows, v]
        nxt[rows, p] = q
        prev[rows, q] = p
        hi = lo if (c - 1) % 2 == 1 else nxt[rows, lo]
        med_lo[:, c - 2] = lo - 1
        med_hi[:, c - 2] = hi - 1
    return (med_lo.astype(np.float64) + med_hi.astype(np.float64)) / 2.0


def _outlier_include_sweep(Y: F64, sign: int) -> F64:
    """Vectorised ``DN_OutlierInclude`` over rows; equal to the per-row scan."""
    R, L = Y.shape
    inc = _OUTLIER_INC
    out = np.full(R, np.nan)
    yw = sign * Y
    max_val = yw.max(axis=1)
    small = max_val < inc
    out[small] = 0.0
    rows = np.flatnonzero(~small)
    if rows.size == 0:
        return out
    yw = yw[rows]
    n_thresh = (max_val[rows] / inc).astype(np.int64) + 1
    j_max = int(n_thresh.max())
    thresholds = np.arange(j_max, dtype=np.float64) * inc
    # kmax[i] = largest j with thresholds[j] <= yw[i] (-1 if none), found from a
    # division estimate and then settled by the exact comparisons themselves.
    est = np.floor(yw / inc).astype(np.int64)
    est = np.clip(est, -1, j_max - 1)
    for _ in range(2):
        up = (est + 1 < j_max) & (thresholds[np.minimum(est + 1, j_max - 1)] <= yw)
        est = np.where(up, est + 1, est)
        down = (est >= 0) & (thresholds[np.maximum(est, 0)] > yw)
        est = np.where(down, est - 1, est)
    # counts[r, j] = #{i : yw[r, i] >= thresholds[j]} = #{i : kmax[i] >= j}
    g = rows.size
    hist = np.bincount(
        (np.arange(g)[:, None] * (j_max + 1) + (est + 1)).ravel(),
        minlength=g * (j_max + 1),
    ).reshape(g, j_max + 1)
    counts = np.cumsum(hist[:, ::-1], axis=1)[:, ::-1][:, 1:]  # (g, j_max)
    pct = 100.0 * counts / L
    valid = (np.arange(j_max)[None, :] < n_thresh[:, None]) & (pct > 2.0)
    any_valid = valid.any(axis=1)
    order_desc = np.argsort(-yw, axis=1, kind="stable")
    med = _prefix_medians(order_desc)  # (g, L): median index of the top-c set
    c_idx = np.clip(counts - 1, 0, L - 1)
    med_pos = np.take_along_axis(med, c_idx, axis=1) / (L / 2.0) - 1.0
    med_pos = np.where(valid, med_pos, np.inf)
    med_pos.sort(axis=1)
    n_valid = valid.sum(axis=1)
    lo_i = np.maximum((n_valid - 1) // 2, 0)
    hi_i = np.maximum(n_valid // 2, 0)
    lo_v = np.take_along_axis(med_pos, lo_i[:, None], axis=1)[:, 0]
    hi_v = np.take_along_axis(med_pos, hi_i[:, None], axis=1)[:, 0]
    res = (lo_v + hi_v) / 2.0
    out[rows] = np.where(any_valid, res, np.nan)
    return out


def _k_outlier_include(b: _Batch, sign: int) -> F64:
    if b.L == 0:
        return np.full(b.n, np.nan)
    if b.n < _OUTLIER_SWEEP_MIN_ROWS:
        return np.array([_outlier_include_1d(y, sign) for y in b.Y])
    return _outlier_include_sweep(b.Y, sign)


# ---------------------------------------------------------------------------
# Power-spectrum features
# ---------------------------------------------------------------------------
def _welch_cumulative_rows(Y: F64) -> tuple[F64, F64, NDArray[np.bool_]] | None:
    """Row-wise ``_welch_cumulative``: rectangular one-segment Welch spectrum.

    Returns ``(w, cs, ok)`` -- the angular frequency grid, the per-row
    normalised cumulative spectrum, and the rows for which it is defined -- or
    ``None`` when no row is long enough.
    """
    n, L = Y.shape
    if L < 4:
        return None
    z = np.fft.rfft(Y, n=L, axis=1)
    scale = 1.0 / (1.0 * float(L))
    p = _power(z) * scale
    if L % 2 == 0:
        p[:, 1:-1] *= 2
    else:
        p[:, 1:] *= 2
    w = 2.0 * np.pi * np.fft.rfftfreq(L, 1.0)
    if w.size < 2:
        return None
    dw = w[1] - w[0]
    area = np.sum(p, axis=1) * dw
    ok = area > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        s_norm = p / np.where(ok, area, 1.0)[:, None]
    cs = np.cumsum(s_norm, axis=1) * dw
    return w, cs, ok


def _k_welch_area(b: _Batch) -> F64:
    out = np.full(b.n, np.nan)
    res = b.welch
    if res is None:
        return out
    w, cs, ok = res
    idx = min(w.size // 5, cs.shape[1] - 1)
    out[ok] = cs[ok, idx]
    return out


def _k_welch_centroid(b: _Batch) -> F64:
    out = np.full(b.n, np.nan)
    res = b.welch
    if res is None:
        return out
    w, cs, ok = res
    idx = np.argmax(cs >= 0.5, axis=1)
    out[ok] = w[idx[ok]]
    return out


# ---------------------------------------------------------------------------
# Fluctuation-scaling (DFA family) features
# ---------------------------------------------------------------------------
def _line_sse_rows(x: F64, Y: F64) -> F64:
    """Per-row SSE of the OLS line of ``Y[r]`` on ``x`` (centred, closed form)."""
    if x.size < 2:
        return np.zeros(Y.shape[0])
    xc = x - x.mean()
    sxx = (xc * xc).sum()
    slope = (Y * xc).sum(axis=1) / sxx
    resid = Y - Y.mean(axis=1)[:, None] - slope[:, None] * xc
    return (resid * resid).sum(axis=1)


def _best_breakpoint(log_tau: F64, log_f: F64) -> F64:
    """``best_br / ntt`` of the two-segment fit (first minimum, strict ``<``)."""
    ntt = log_tau.size
    best_err = np.full(log_f.shape[0], np.inf)
    best_br = np.full(log_f.shape[0], ntt // 2, dtype=np.int64)
    for br in range(2, ntt - 1):
        err = _line_sse_rows(log_tau[:br], log_f[:, :br]) + _line_sse_rows(
            log_tau[br - 1 :], log_f[:, br - 1 :]
        )
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_br = np.where(better, br, best_br)
    return best_br / ntt


def _fluctuation_rows(Y: F64) -> tuple[F64, F64]:
    """Both ``SC_FluctAnal`` variants over rows: ``(dfa, rsrangefit)``.

    The integrated profile is cut into non-overlapping windows at ~50
    log-spaced scales; each window is linearly detrended (closed-form OLS,
    replacing a per-scale ``lstsq``), and the RMS residual (``dfa``) or the RMS
    detrended range (``rsrangefit``) is recorded. A two-segment line fit to
    ``log F`` against ``log tau`` then picks the breakpoint.

    NEEDS REVIEW: the exact scale grid and breakpoint convention differ subtly
    between implementations; this is a documented best-effort re-derivation and
    may not match ``pycatch22`` to the last digit.  It returns values in
    ``[0, 1]``.
    """
    n, L = Y.shape
    dfa = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    if L < 20:
        return dfa, rs
    tau_min, tau_max = 5, L // 2
    if tau_max <= tau_min + 1:
        return dfa, rs
    taus = np.unique(
        np.round(np.exp(np.linspace(np.log(tau_min), np.log(tau_max), 50))).astype(int)
    )
    taus = taus[(taus >= tau_min) & (taus <= tau_max)]
    profile = np.cumsum(Y - Y.mean(axis=1)[:, None], axis=1)
    F = {"dfa": np.empty((n, taus.size)), "rs": np.empty((n, taus.size))}
    for j, tau in enumerate(taus):
        tau = int(tau)
        n_win = L // tau
        seg = profile[:, : n_win * tau].reshape(n, n_win, tau)
        tc = np.arange(tau, dtype=np.float64) - (tau - 1) / 2.0
        slope = (seg * tc).sum(axis=2) / (tc * tc).sum()
        resid = seg - (seg.mean(axis=2)[:, :, None] + slope[:, :, None] * tc)
        sq_dfa = np.mean(resid**2, axis=2)
        sq_rs = (resid.max(axis=2) - resid.min(axis=2)) ** 2
        F["dfa"][:, j] = np.sqrt(np.mean(sq_dfa, axis=1))
        F["rs"][:, j] = np.sqrt(np.mean(sq_rs, axis=1))
    log_tau_all = np.log(taus.astype(np.float64))
    results = {"dfa": dfa, "rs": rs}
    for how, Fh in F.items():
        out = results[how]
        pos = Fh > 0
        full = pos.all(axis=1)
        if full.any() and taus.size >= 5:
            with np.errstate(divide="ignore"):
                out[full] = _best_breakpoint(log_tau_all, np.log(Fh[full]))
        for r in np.flatnonzero(~full):  # some scales degenerate: ragged row
            keep = pos[r]
            if keep.sum() < 5:
                continue
            out[r] = _best_breakpoint(log_tau_all[keep], np.log(Fh[r, keep])[None, :])[
                0
            ]
    return dfa, rs


# ---------------------------------------------------------------------------
# Raw (catch24) features
# ---------------------------------------------------------------------------
def _k_mean(b: _Batch) -> F64:
    if b.L == 0:
        return np.full(b.n, np.nan)
    return b.X.mean(axis=1)


def _k_std(b: _Batch) -> F64:
    if b.L < 2:
        return np.full(b.n, np.nan)
    return b.X.std(axis=1, ddof=1)


# ---------------------------------------------------------------------------
# Registry and entry points
# ---------------------------------------------------------------------------
_KERNELS: dict[str, Callable[[_Batch], F64]] = {
    "DN_HistogramMode_5": lambda b: _k_histogram_mode(b, 5),
    "DN_HistogramMode_10": lambda b: _k_histogram_mode(b, 10),
    "CO_f1ecac": _k_f1ecac,
    "CO_FirstMin_ac": _k_firstmin_ac,
    "CO_HistogramAMI_even_2_5": _k_histogram_ami,
    "CO_trev_1_num": _k_trev,
    "CO_Embed2_Dist_tau_d_expfit_meandiff": _k_embed2,
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": _k_ami_gaussian,
    "MD_hrv_classic_pnn40": _k_pnn40,
    "SB_BinaryStats_mean_longstretch1": _k_mean_longstretch1,
    "SB_BinaryStats_diff_longstretch0": _k_diff_longstretch0,
    "SB_MotifThree_quantile_hh": _k_motif_three,
    "SB_TransitionMatrix_3ac_sumdiagcov": _k_transition_matrix,
    "PD_PeriodicityWang_th0_01": _k_periodicity,
    "FC_LocalSimple_mean1_tauresrat": _k_localsimple_mean1,
    "FC_LocalSimple_mean3_stderr": _k_localsimple_mean3,
    "DN_OutlierInclude_p_001_mdrmd": lambda b: _k_outlier_include(b, 1),
    "DN_OutlierInclude_n_001_mdrmd": lambda b: _k_outlier_include(b, -1),
    "SP_Summaries_welch_rect_area_5_1": _k_welch_area,
    "SP_Summaries_welch_rect_centroid": _k_welch_centroid,
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": lambda b: b.fluct[0],
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": lambda b: b.fluct[1],
    "DN_Mean": _k_mean,
    "DN_Spread_Std": _k_std,
}


def _run_kernel(name: str, b: _Batch) -> F64:
    """One kernel over one batch, with the scalar path's error contract.

    The scalar aggregate turned any exception inside a feature into NaN for
    that feature. A kernel that raises on a batch is re-run row by row so one
    pathological window cannot blank out its neighbours.
    """
    kernel = _KERNELS[name]
    try:
        with np.errstate(all="ignore"):
            return np.asarray(kernel(b), dtype=np.float64)
    except Exception:
        out = np.full(b.n, np.nan)
        for i in range(b.n):
            try:
                with np.errstate(all="ignore"):
                    out[i] = kernel(_Batch(b.X[i : i + 1]))[0]
            except Exception:
                pass
        return out


def _rows_equal_length(X: F64, names: Sequence[str]) -> F64:
    out = np.empty((X.shape[0], len(names)))
    for start in range(0, X.shape[0], _CHUNK_ROWS):
        b = _Batch(np.ascontiguousarray(X[start : start + _CHUNK_ROWS]))
        for j, name in enumerate(names):
            out[start : start + b.n, j] = _run_kernel(name, b)
    return out


def catch22_rows(X: Any, names: Sequence[str]) -> F64:
    """Compute the named catch22/catch24 features for every row of ``X``.

    Parameters
    ----------
    X : array-like
        A 2-D ``(n_windows, L)`` array (1-D input is treated as one row). Each
        row is one series; rows are z-scored internally where the feature
        requires it.
    names : sequence of str
        Feature names from :data:`panelary.catch22.CATCH22_NAMES` plus,
        optionally, ``"DN_Mean"`` / ``"DN_Spread_Std"``.

    Returns
    -------
    numpy.ndarray
        ``(n_windows, len(names))`` float64. Row ``i`` depends only on
        ``X[i]``. A row containing NaN is computed on its NaN-dropped values,
        exactly as the scalar :func:`~panelary.catch22.catch22_all` does.

    Raises
    ------
    KeyError
        If a name is not a known feature.
    """
    for name in names:
        if name not in _KERNELS:
            raise KeyError(f"unknown catch22 feature {name!r}")
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (n_windows, L) array, got ndim={arr.ndim}")
    n = arr.shape[0]
    out = np.empty((n, len(names)))
    if n == 0:
        return out
    has_nan = np.isnan(arr).any(axis=1)
    clean = np.flatnonzero(~has_nan)
    if clean.size:
        out[clean] = _rows_equal_length(arr[clean], names)
    for i in np.flatnonzero(has_nan):
        row = arr[i][~np.isnan(arr[i])]
        out[i] = _rows_equal_length(row[None, :], names)[0]
    return out


def feature_rows(name: str, X: Any) -> F64:
    """One feature over every row of ``X`` (see :func:`catch22_rows`)."""
    return catch22_rows(X, [name])[:, 0]
