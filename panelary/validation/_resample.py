"""Vectorised resampling engine: count matrices and start-count block bootstraps.

Private engine for the forecast-evaluation family. The existing samplers in
:mod:`panelary.validation._bootstrap` return ``(B, T)`` index matrices, and
their callers then loop over replicates gathering ``x[idx[b]]``. For every
statistic that is a function of *sample moments* that loop is unnecessary:

**Count matrix (A1).** For an index matrix ``idx (B, T)``,
``C = bincount(b T + idx).reshape(B, T)`` counts how often each observation
appears in each replicate. Every replicate sum of every column is then one BLAS
product, ``C @ X``: ``O(B T M)`` flops, no Python loop, no ``(B, T, M)``
temporary. Counts are small integers, exact in float64.

**Start-count circular-block bootstrap (A2).** A circular-block bootstrap
(Politis & Romano 1992) with fixed block length ``b`` draws ``l = floor(T/b)``
start positions per replicate. With ``N = bincount(b T + starts) (B, T)`` and the
circular block sums ``Q_x[s] = sum_{k<b} x[(s + k) mod T]``:

* the replicate sum of ``x`` is ``N @ Q_x``;
* the sum over blocks of block-sum products, ``sum_j Q_{x,s_j} Q_{y,s_j}``, is
  ``N @ (Q_x * Q_y)``.

That is everything the Goetze-Kuensch studentization of Ledoit & Wolf (2008,
§3.2.2) needs, so no bootstrap sample is ever materialised.

Fold segments (``boundaries``) follow the guarantee of
:func:`~panelary.validation.block_bootstrap_indices`: a block never straddles a
boundary. Segment ``s`` of length ``T_s`` gets ``floor(T_s / b)`` blocks, started
uniformly inside it and wrapped circularly *inside* it; block sums use
segment-local circular padding.

Determinism: all random draws for a call are made up front by one generator, so
chunking the linear algebra over replicates never changes which indices are
used. BLAS products are bitwise reproducible for a fixed shape, but a different
shape (a different chunk of replicates, or more columns) may be blocked
differently and round differently -- CI's OpenBLAS and Apple M1 runners differ
by about 1 ulp -- so chunked results agree to rounding, not bitwise. The helpers
here keep the shapes that matter most stable (e.g. a single column is padded to
two, because ``gemv`` and ``gemm`` round differently).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from panelary.validation._bootstrap import resolve_segments

__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "CircularBlockDraw",
    "bootstrap_means",
    "circular_block_draw",
    "circular_block_sums",
    "count_matrix",
    "matmul_cols",
    "row_chunks",
    "start_counts",
    "stationary_indices",
]

#: Default working-set budget for one chunk of replicate rows.
DEFAULT_CHUNK_BYTES = 256 * 2**20


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    return (
        seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    )


def row_chunks(n_rows: int, bytes_per_row: int, chunk_bytes: int) -> list[slice]:
    """Split ``range(n_rows)`` into balanced contiguous chunks under ``chunk_bytes``.

    Balanced (``numpy.array_split`` sizes) rather than "full chunks plus a
    remainder", so no chunk degenerates to one row.
    """
    if n_rows <= 0:
        return []
    per = max(1, int(chunk_bytes // max(int(bytes_per_row), 1)))
    n_chunks = -(-n_rows // per)
    edges = np.linspace(0, n_rows, n_chunks + 1).round().astype(np.int64)
    return [
        slice(int(a), int(b))
        for a, b in zip(edges[:-1], edges[1:], strict=True)
        if b > a
    ]


def matmul_cols(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``left @ right`` for a 2-D ``right``, with a width-1 ``right`` padded to 2.

    BLAS rounds a one-column product (``gemv``) differently from the same column
    inside a wider product (``gemm``). Padding routes both through ``gemm``, so a
    model's result does not depend on how many other models share the call --
    bitwise on most BLAS builds, to within an ulp on the rest.
    """
    if right.ndim != 2:
        raise ValueError("`right` must be 2-D.")
    if right.shape[1] == 1:
        return (left @ np.concatenate([right, right], axis=1))[:, :1]
    return left @ right


# --------------------------------------------------------------------------- #
# A1: count matrix
# --------------------------------------------------------------------------- #
def count_matrix(idx: np.ndarray, n_obs: int) -> np.ndarray:
    """``(B, T)`` float64 counts of each observation in each replicate.

    ``C[b, t] = #{s : idx[b, s] == t}`` via one ``bincount`` on ``b * T + idx``.
    """
    ix = np.asarray(idx, dtype=np.int64)
    if ix.ndim != 2:
        raise ValueError(f"`idx` must be (B, n), got shape {ix.shape}.")
    n_boot = ix.shape[0]
    if ix.size and (ix.min() < 0 or ix.max() >= n_obs):
        raise ValueError("`idx` holds positions outside [0, n_obs).")
    flat = (np.arange(n_boot, dtype=np.int64)[:, None] * n_obs + ix).ravel()
    counts = np.bincount(flat, minlength=n_boot * n_obs)
    return counts.reshape(n_boot, n_obs).astype(np.float64)


def bootstrap_means(
    x: np.ndarray, idx: np.ndarray, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> np.ndarray:
    """Replicate column means ``C @ X / n`` for index matrix ``idx (B, n)``.

    Equivalent to ``np.stack([x[i].mean(0) for i in idx])`` (to ~1e-16) without
    the Python loop. Chunked over replicates when ``B * T * 8 > chunk_bytes``.
    Returns ``(B, M)`` (or ``(B,)`` for 1-D ``x``).
    """
    arr = np.asarray(x, dtype=np.float64)
    was_1d = arr.ndim == 1
    mat = arr[:, None] if was_1d else arr
    ix = np.asarray(idx, dtype=np.int64)
    n_obs = mat.shape[0]
    out = np.empty((ix.shape[0], mat.shape[1]), dtype=np.float64)
    for rows in row_chunks(ix.shape[0], n_obs * 8, chunk_bytes):
        out[rows] = matmul_cols(count_matrix(ix[rows], n_obs), mat)
    out /= ix.shape[1]
    return out[:, 0] if was_1d else out


# --------------------------------------------------------------------------- #
# A2: start-count circular-block bootstrap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CircularBlockDraw:
    """Start positions of a (segment-aware) circular-block bootstrap.

    Attributes
    ----------
    starts : ndarray of shape (B, l)
        Absolute start positions; columns are grouped by segment.
    block_length : int
        Fixed block length ``b``.
    n_obs : int
        Length ``T`` of the original sample.
    segments : list of (int, int)
        Half-open segments; blocks start and wrap inside their segment.
    blocks_per_segment : tuple of int
        ``floor(T_s / b)`` per segment.
    """

    starts: np.ndarray
    block_length: int
    n_obs: int
    segments: list[tuple[int, int]]
    blocks_per_segment: tuple[int, ...]

    @property
    def n_boot(self) -> int:
        """Number of replicates ``B``."""
        return int(self.starts.shape[0])

    @property
    def n_blocks(self) -> int:
        """Blocks per replicate ``l``."""
        return int(self.starts.shape[1])

    @property
    def n_sample(self) -> int:
        """Bootstrap sample size ``l * b`` (LW 2008: not padded up to ``T``)."""
        return self.n_blocks * self.block_length

    def indices(self) -> np.ndarray:
        """The literal ``(B, l*b)`` gather indices (for tests and small problems)."""
        b = self.block_length
        out = np.empty((self.n_boot, self.n_sample), dtype=np.int64)
        j = 0
        for (a, stop), nb in zip(self.segments, self.blocks_per_segment, strict=True):
            seg_len = stop - a
            for _ in range(nb):
                rel = self.starts[:, j] - a
                out[:, j * b : (j + 1) * b] = (
                    a + (rel[:, None] + np.arange(b)) % seg_len
                )
                j += 1
        return out


def circular_block_draw(
    n_obs: int,
    block_length: int,
    n_boot: int,
    *,
    boundaries: Sequence[int] | np.ndarray | None = None,
    seed: int | np.random.Generator | None = 0,
) -> CircularBlockDraw:
    """Draw ``(B, l)`` circular-block start positions in one generator call.

    Parameters
    ----------
    n_obs : int
        Sample length ``T``.
    block_length : int
        Block length ``b``; must not exceed the shortest segment.
    n_boot : int
        Replicates ``B``.
    boundaries : sequence of int, optional
        Segment start positions (fold boundaries); blocks never straddle them.
    seed : int | numpy.random.Generator
        Seed or generator; the draw is one ``integers`` call.

    Raises
    ------
    ValueError
        If ``block_length`` exceeds a segment, or no block fits.
    """
    b = int(block_length)
    if b < 1:
        raise ValueError(f"`block_length` must be >= 1, got {block_length}.")
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1, got {n_boot}.")
    segments = resolve_segments(int(n_obs), boundaries)
    lengths = [stop - a for a, stop in segments]
    if b > min(lengths):
        raise ValueError(
            f"`block_length`={b} exceeds the shortest segment ({min(lengths)} "
            "observations); blocks may not straddle a fold boundary."
        )
    per = tuple(length // b for length in lengths)
    highs = np.repeat(np.asarray(lengths, dtype=np.int64), per)
    offsets = np.repeat(np.asarray([a for a, _ in segments], dtype=np.int64), per)
    rng = _rng(seed)
    starts = rng.integers(0, highs, size=(int(n_boot), highs.size)) + offsets
    return CircularBlockDraw(starts, b, int(n_obs), segments, per)


def start_counts(draw: CircularBlockDraw, rows: slice | None = None) -> np.ndarray:
    """``N (B_c, T)``: how many blocks start at each position, per replicate."""
    st = draw.starts if rows is None else draw.starts[rows]
    return count_matrix(st, draw.n_obs)


def circular_block_sums(
    x: np.ndarray,
    block_length: int,
    segments: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Circular block sums ``Q[s] = sum_{k<b} x[(s + k) mod T_seg]`` per segment.

    Returns an array shaped like ``x``; row ``s`` is the sum of the block that
    starts at ``s`` and wraps inside ``s``'s segment.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[0]
    b = int(block_length)
    segs = [(0, n)] if segments is None else segments
    # C order whatever the input layout: these feed BLAS products, whose
    # rounding must not depend on the operand layout.
    out = np.empty(arr.shape, dtype=np.float64)
    for a, stop in segs:
        seg = arr[a:stop]
        length = stop - a
        if b > length:
            raise ValueError("`block_length` exceeds a segment.")
        pad = np.concatenate([seg, seg[: b - 1]], axis=0)
        acc = pad[:length].copy()
        for k in range(1, b):
            acc += pad[k : k + length]
        out[a:stop] = acc
    return out


# --------------------------------------------------------------------------- #
# Vectorised stationary bootstrap indices
# --------------------------------------------------------------------------- #
def stationary_indices(
    n_obs: int,
    mean_block: float,
    n_boot: int,
    *,
    length: int | None = None,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """``(B, length)`` stationary-bootstrap indices (Politis & Romano 1994), no loop.

    ``length`` defaults to ``n_obs`` (a resample of the same size); a longer
    ``length`` draws a longer path over the same ``n_obs`` positions.

    A new block starts at each position with probability ``1 / mean_block`` (and
    always at position 0) at a uniform random origin; within a block the index
    advances by one, wrapping circularly. Two generator calls, ``O(B T)``.

    This stream differs from :func:`~panelary.validation.block_bootstrap_indices`
    (which draws block by block); use that one where its exact stream matters.
    """
    if mean_block < 1:
        raise ValueError(f"`mean_block` must be >= 1, got {mean_block}.")
    rng = _rng(seed)
    n = int(n_obs)
    width = n if length is None else int(length)
    flags = rng.random((int(n_boot), width)) < (1.0 / float(mean_block))
    flags[:, 0] = True
    origin = rng.integers(0, n, size=(int(n_boot), width))
    t = np.arange(width, dtype=np.int64)
    last = np.maximum.accumulate(np.where(flags, t, 0), axis=1)
    start = np.take_along_axis(origin, last, axis=1)
    return (start + (t - last)) % n
