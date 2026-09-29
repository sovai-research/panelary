"""Sequential kernels for event sampling: numba fast path + bitwise numpy twins.

Two recursions live here, each as a numba-compatible function compiled lazily
through :mod:`panelary._internal._jit` and a numpy / pure-Python twin that
produces **bitwise-identical** output (same operations, same order; every sum
is sequential -- ``np.cumsum``, never the pairwise ``np.sum``):

* the exact sequential bootstrap (AFML 4.5) by local recomputation;
* the AFML symmetric CUSUM event filter.

numba is never imported here; :meth:`LazyKernel.compiled` returns ``None``
without it (or inside :func:`panelary._internal._jit.force_numpy`), and the
dispatchers run the twin.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from panelary._internal._jit import lazy_njit

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "_cusum_filter",
    "_cusum_filter_kernel",
    "_cusum_filter_py",
    "_seq_boot",
    "_seq_boot_kernel",
    "_seq_boot_numpy",
]


# --------------------------------------------------------------------------- #
# Exact sequential bootstrap
# --------------------------------------------------------------------------- #
def _block_size(n: int) -> int:
    """Members per block of the two-level inverse-CDF sampler (about sqrt(N))."""
    return max(1, math.isqrt(max(n, 1)))


@lazy_njit
def _seq_boot_kernel(
    start: np.ndarray,
    end: np.ndarray,
    u: np.ndarray,
    block: int,
    out: np.ndarray,
) -> None:  # pragma: no cover - exercised only when numba is installed
    """Draw ``len(u)`` span indices by AFML's sequential bootstrap.

    ``start`` / ``end`` are sorted-by-start closed row spans on a 0-based grid.
    Draw ``k`` picks span ``i`` with probability proportional to its average
    uniqueness ``mean_{r in span_i} 1 / (c_r + 1)`` given the counts ``c`` of
    already-drawn spans, by inverse CDF of ``u[k]`` over two levels (blocks of
    ``block`` spans). After a draw only the spans overlapping it change; each is
    recomputed from scratch, so every weight is a function of ``c`` alone and no
    rounding accumulates across draws.
    """
    n = start.shape[0]
    n_rows = 0
    hmax = 1
    for i in range(n):
        if end[i] + 1 > n_rows:
            n_rows = end[i] + 1
        if end[i] - start[i] + 1 > hmax:
            hmax = end[i] - start[i] + 1
    c = np.zeros(n_rows, dtype=np.float64)
    ubar = np.ones(n, dtype=np.float64)
    n_blocks = (n + block - 1) // block
    bsum = np.zeros(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        acc = 0.0
        for i in range(b * block, min(n, (b + 1) * block)):
            acc += ubar[i]
        bsum[b] = acc
    touched = np.zeros(n_blocks, dtype=np.bool_)

    for k in range(u.shape[0]):
        total = 0.0
        for b in range(n_blocks):
            total += bsum[b]
        target = u[k] * total
        acc = 0.0
        bsel = -1
        for b in range(n_blocks):
            nxt = acc + bsum[b]
            if nxt > target:
                bsel = b
                break
            acc = nxt
        j = -1
        if bsel >= 0:
            for i in range(bsel * block, min(n, (bsel + 1) * block)):
                acc += ubar[i]
                if acc > target:
                    j = i
                    break
        if j < 0:  # rounding put the target at the very top of the CDF
            j = n - 1
        out[k] = j

        s0 = start[j]
        e0 = end[j]
        for r in range(s0, e0 + 1):
            c[r] += 1.0
        lo = np.searchsorted(start, s0 - hmax + 1, side="left")
        hi = np.searchsorted(start, e0, side="right")
        for i in range(lo, hi):
            if end[i] >= s0:
                acc2 = 0.0
                for r in range(start[i], end[i] + 1):
                    acc2 += 1.0 / (c[r] + 1.0)
                ubar[i] = acc2 / (end[i] - start[i] + 1)
                touched[i // block] = True
        if hi > lo:
            for b in range(lo // block, (hi - 1) // block + 1):
                if touched[b]:
                    acc3 = 0.0
                    for i in range(b * block, min(n, (b + 1) * block)):
                        acc3 += ubar[i]
                    bsum[b] = acc3
                    touched[b] = False


def _seq_boot_numpy(
    start: NDArray[np.int64],
    end: NDArray[np.int64],
    u: NDArray[np.float64],
    block: int,
    out: NDArray[np.int64],
) -> None:
    """The numpy twin of :func:`_seq_boot_kernel` (bitwise-identical draws).

    Same arithmetic in the same order: every running sum is ``np.cumsum``
    (sequential), the inverse CDF is ``searchsorted(..., side="right")`` on
    those running sums (the first partial sum strictly above the target, as the
    kernel's loop finds it), and the affected spans are recomputed from a
    padded gather whose padding (``+0.0``) cannot change a sequential sum.
    """
    n = start.shape[0]
    if n == 0 or u.shape[0] == 0:
        return
    n_rows = int(end.max()) + 1
    lengths = end - start + 1
    hmax = int(lengths.max())
    c = np.zeros(n_rows, dtype=np.float64)
    ubar = np.ones(n, dtype=np.float64)
    n_blocks = (n + block - 1) // block
    padded = np.zeros(n_blocks * block, dtype=np.float64)
    padded[:n] = ubar
    bsum = np.cumsum(padded.reshape(n_blocks, block), axis=1)[:, -1].copy()
    steps = np.arange(hmax, dtype=np.int64)

    for k in range(u.shape[0]):
        cs = np.cumsum(bsum)
        total = cs[-1]
        target = u[k] * total
        bsel = int(np.searchsorted(cs, target, side="right"))
        j = -1
        if bsel < n_blocks:
            acc0 = cs[bsel - 1] if bsel > 0 else 0.0
            lo_m = bsel * block
            members = ubar[lo_m : min(n, lo_m + block)]
            part = np.cumsum(np.concatenate((np.array([acc0]), members)))[1:]
            w = int(np.searchsorted(part, target, side="right"))
            if w < members.shape[0]:
                j = lo_m + w
        if j < 0:
            j = n - 1
        out[k] = j

        s0 = int(start[j])
        e0 = int(end[j])
        c[s0 : e0 + 1] += 1.0
        lo = int(np.searchsorted(start, s0 - hmax + 1, side="left"))
        hi = int(np.searchsorted(start, e0, side="right"))
        if hi <= lo:
            continue
        cand: NDArray[np.int64] = np.arange(lo, hi, dtype=np.int64)
        cand = cand[end[cand] >= s0]
        if cand.size == 0:
            continue
        lens = lengths[cand]
        rows = start[cand][:, None] + steps[None, :]
        valid = steps[None, :] < lens[:, None]
        vals = np.where(valid, 1.0 / (c[np.where(valid, rows, 0)] + 1.0), 0.0)
        sums = np.cumsum(vals, axis=1)[np.arange(cand.size), lens - 1]
        ubar[cand] = sums / lens
        for b in np.unique(cand // block).tolist():
            seg = ubar[b * block : min(n, (b + 1) * block)]
            bsum[b] = np.cumsum(seg)[-1]


def _seq_boot(
    start: NDArray[np.int64],
    end: NDArray[np.int64],
    u: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Sequential-bootstrap span indices for uniforms ``u`` (numba if available)."""
    start = np.ascontiguousarray(start, dtype=np.int64)
    end = np.ascontiguousarray(end, dtype=np.int64)
    u = np.ascontiguousarray(u, dtype=np.float64)
    out = np.zeros(u.shape[0], dtype=np.int64)
    if start.shape[0] == 0 or u.shape[0] == 0:
        return out
    # Rebase to a 0-based local grid so the count array covers only this range.
    base = int(start[0])
    s = start - base
    e = end - base
    block = _block_size(start.shape[0])
    kernel = _seq_boot_kernel.compiled()
    if kernel is not None:
        kernel(s, e, u, block, out)
    else:
        _seq_boot_numpy(s, e, u, block, out)
    return out


# --------------------------------------------------------------------------- #
# AFML symmetric CUSUM filter
# --------------------------------------------------------------------------- #
@lazy_njit
def _cusum_filter_kernel(
    y: np.ndarray, h: np.ndarray, out: np.ndarray
) -> None:  # pragma: no cover - exercised only when numba is installed
    """``S+ = max(0, S+ + y)``, ``S- = min(0, S- + y)``; fire and reset one side.

    A NaN ``y`` is skipped (state unchanged, no event). A NaN threshold fires
    nothing but the sums still accumulate.
    """
    sp = 0.0
    sn = 0.0
    for t in range(y.shape[0]):
        yt = y[t]
        if np.isnan(yt):
            out[t] = False
            continue
        sp = max(0.0, sp + yt)
        sn = min(0.0, sn + yt)
        ht = h[t]
        if sn < -ht:
            out[t] = True
            sn = 0.0
        elif sp > ht:
            out[t] = True
            sp = 0.0
        else:
            out[t] = False


def _cusum_filter_py(y: list[float], h: list[float]) -> list[bool]:
    """Pure-Python twin of :func:`_cusum_filter_kernel` over plain floats."""
    out = [False] * len(y)
    sp = 0.0
    sn = 0.0
    for t, yt in enumerate(y):
        if yt != yt:  # NaN
            continue
        sp = max(0.0, sp + yt)
        sn = min(0.0, sn + yt)
        ht = h[t]
        if sn < -ht:
            out[t] = True
            sn = 0.0
        elif sp > ht:
            out[t] = True
            sp = 0.0
    return out


def _cusum_filter(y: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Symmetric CUSUM events for one series (numba if available)."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    kernel = _cusum_filter_kernel.compiled()
    if kernel is not None:
        out = np.zeros(y.shape[0], dtype=np.bool_)
        kernel(y, h, out)
        return out
    return np.asarray(_cusum_filter_py(y.tolist(), h.tolist()), dtype=np.bool_)
