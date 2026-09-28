"""Similarity sketches for near-duplicate detection -- pure Polars + numpy.

Everything here is a clean-room implementation from the published algorithms;
no code is taken from datasketch, rensa or any other sketching library
(datasketch is used only as an optional test-time oracle).

Sketches
--------
* **Shingling** (:func:`tokenize`) -- a row becomes a *set* of tokens: one per
  ``column=value`` cell (``"cells"``), or character / word n-grams of the
  concatenated text (``"char"`` / ``"word"``). Built with Polars expressions.
* **MinHash** (:func:`minhash_signatures`, ``variant="minhash"``) -- Broder
  (1997). ``num_perm`` universal hash functions ``(a*x + b) mod p`` with
  ``p = 2**31 - 1`` over one Polars token hash; the fraction of equal
  signature slots is an unbiased estimate of Jaccard similarity.
* **C-MinHash** (``variant="cminhash"``) -- Li & Li (2022). Tokens are hashed
  into a universe of size ``D`` (playing the role of the first permutation
  sigma), and slot ``k`` is ``min pi[(sigma(x) + k) mod D]`` for a single random
  permutation ``pi`` -- one permutation, circulantly shifted, with a smaller
  estimator variance than ``num_perm`` independent ones.
* **b-bit compression** (:func:`bbit`, :func:`estimate_jaccard`) -- Li &
  Koenig (2010). Keep the lowest ``b`` bits of every slot and correct the
  collision rate ``2**-b`` in the estimator (the large-universe limit).
* **SimHash** (:func:`simhash_bits`) -- Charikar (2002). Signs of random
  Gaussian projections; the fraction of agreeing bits estimates
  ``1 - angle / pi``, i.e. cosine similarity for numeric feature rows.
* **LSH banding** (:func:`lsh_params`, :func:`lsh_candidate_pairs`) -- the
  ``b`` bands x ``r`` rows scheme of Leskovec, Rajaraman & Ullman (*Mining of
  Massive Datasets*, ch. 3). A pair becomes a candidate iff it agrees on every
  slot of at least one band; the S-curve ``1 - (1 - s**r)**b`` has its knee
  near ``(1/b)**(1/r)``.

Determinism
-----------
Every random quantity is drawn from ``np.random.default_rng(seed)``. Token
hashes come from Polars' ``Expr.hash(seed=...)``, which is deterministic for a
given seed **and Polars version**; Polars does not promise hash stability
across releases, so signatures should not be persisted across upgrades.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from panelary.clean._common import explode

__all__ = [
    "bbit",
    "estimate_jaccard",
    "exact_jaccard",
    "lsh_candidate_pairs",
    "lsh_params",
    "minhash_signatures",
    "simhash_bits",
    "tokenize",
]

#: Mersenne prime 2**31 - 1: universal-hash modulus. With ``a, x < 2**31`` the
#: product ``a * x`` stays below 2**62, so ``(a * x + b) % p`` never overflows
#: a uint64.
_P31 = np.uint64((1 << 31) - 1)
#: Sentinel slot value for a row with no tokens (never equal to a real slot,
#: whose values are all below 2**31).
EMPTY_SLOT = np.uint64(np.iinfo(np.uint64).max)
#: Universe size for C-MinHash's hashed first permutation.
_CMINHASH_UNIVERSE = 1 << 20
#: Soft cap on the ``(tokens x slots)`` scratch matrix, in elements.
_CHUNK_ELEMS = 4_000_000
#: Separator used when concatenating columns into one document / cell token.
_SEP = "\x1f"
_NULL = "\x00"

Tokenizer = Literal["cells", "char", "word"]


def _trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Trapezoid-rule integral of each row of ``y`` over the 1-D grid ``x``.

    Written out rather than taken from numpy: 2.0 renamed ``trapz`` to
    ``trapezoid`` and later releases drop the old name, so neither spelling
    exists on every supported numpy. Same operation order as numpy's.
    """
    d = np.diff(x)
    return np.add.reduce(d * (y[:, 1:] + y[:, :-1]) / 2.0, axis=1)


# --------------------------------------------------------------------------- #
# Shingling
# --------------------------------------------------------------------------- #
def tokenize(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    row_col: str,
    tokenizer: Tokenizer = "cells",
    ngram: int = 3,
) -> pl.DataFrame:
    """Turn each row into a set of string tokens (one output row per token).

    Parameters
    ----------
    frame : polars.DataFrame
        Must contain ``row_col`` and every name in ``columns``.
    columns : sequence of str
        The content columns that define a row's identity.
    row_col : str
        Row-id column carried through to the output.
    tokenizer : {"cells", "char", "word"}, default="cells"
        ``"cells"``: one token per ``column=value`` (nulls are a value, so two
        nulls agree). ``"char"`` / ``"word"``: the columns are joined into one
        document and shingled into character / whitespace-word n-grams.
    ngram : int, default=3
        Shingle length for ``"char"`` / ``"word"``. A document shorter than
        ``ngram`` becomes a single token.

    Returns
    -------
    polars.DataFrame
        Columns ``[row_col, "token"]``, unique per row, sorted by ``row_col``.
        Rows whose document is empty contribute no tokens.
    """
    if ngram < 1:
        raise ValueError(f"`ngram` must be >= 1, got {ngram}.")
    cols = list(columns)
    if not cols:
        raise ValueError("tokenize needs at least one content column.")
    if tokenizer == "cells":
        # A null is a value in a cell token, so two nulls agree.
        as_text = [pl.col(c).cast(pl.String).fill_null(_NULL) for c in cols]
        cells = [
            pl.concat_str([pl.lit(f"{i}{_SEP}"), expr])
            for i, expr in enumerate(as_text)
        ]
        tok = frame.select(pl.col(row_col), pl.concat_list(cells).alias("token"))
        out = explode(tok, "token")
    elif tokenizer == "char":
        # A null contributes no text to a document.
        as_text = [pl.col(c).cast(pl.String).fill_null("") for c in cols]
        doc = pl.concat_str(as_text, separator=_SEP) if len(cols) > 1 else as_text[0]
        base = frame.select(pl.col(row_col), doc.alias("__doc"))
        n_starts = (pl.col("__doc").str.len_chars().cast(pl.Int64) - ngram + 1).clip(
            lower_bound=1
        )
        out = explode(
            base.with_columns(pl.int_ranges(0, n_starts).alias("__start")), "__start"
        ).select(
            pl.col(row_col),
            pl.col("__doc").str.slice(pl.col("__start"), ngram).alias("token"),
        )
    elif tokenizer == "word":
        as_text = [pl.col(c).cast(pl.String).fill_null("") for c in cols]
        doc = pl.concat_str(as_text, separator=" ") if len(cols) > 1 else as_text[0]
        words = doc.str.split(" ").list.eval(
            pl.element().filter(pl.element().str.len_chars() > 0)
        )
        if ngram == 1:
            grams = words
        else:
            shifted = [pl.element().shift(-k) for k in range(ngram)]
            grams = words.list.eval(pl.concat_str(shifted, separator=" ").drop_nulls())
            # A document shorter than `ngram` words keeps its words as one token.
            grams = (
                pl.when(words.list.len() < ngram)
                .then(pl.concat_list(words.list.join(" ")))
                .otherwise(grams)
            )
        out = explode(frame.select(pl.col(row_col), grams.alias("token")), "token")
    else:
        raise ValueError(
            f"unknown tokenizer {tokenizer!r}; use 'cells', 'char' or 'word'."
        )
    return (
        out.filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
        .unique(maintain_order=True)
        .sort(row_col, maintain_order=True)
    )


def token_hashes(tokens: pl.DataFrame, *, row_col: str, seed: int) -> pl.DataFrame:
    """Hash every token once (``Expr.hash(seed=seed)``) and de-duplicate.

    Returns ``[row_col, "h"]`` (``UInt64``), unique per row, sorted by row.
    """
    return (
        tokens.select(pl.col(row_col), pl.col("token").hash(seed=seed).alias("h"))
        .unique(maintain_order=True)
        .sort(row_col, maintain_order=True)
    )


# --------------------------------------------------------------------------- #
# MinHash / C-MinHash
# --------------------------------------------------------------------------- #
def _segment_starts(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start offsets and ids of each run of equal values in sorted ``rows``."""
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(rows)) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    return starts, rows[starts]


def minhash_signatures(
    rows: np.ndarray,
    hashes: np.ndarray,
    n_rows: int,
    *,
    num_perm: int = 128,
    seed: int = 0,
    variant: Literal["minhash", "cminhash"] = "minhash",
) -> np.ndarray:
    """MinHash (or C-MinHash) signatures of token-hash sets.

    Parameters
    ----------
    rows : numpy.ndarray
        Row id of each token hash, **sorted ascending**, values in
        ``[0, n_rows)``.
    hashes : numpy.ndarray
        ``uint64`` token hashes aligned with ``rows``.
    n_rows : int
        Number of rows (rows with no token get :data:`EMPTY_SLOT` everywhere).
    num_perm : int, default=128
        Signature length.
    seed : int, default=0
        Seed of the hash-family parameters.
    variant : {"minhash", "cminhash"}, default="minhash"
        ``"minhash"``: ``num_perm`` independent universal hashes.
        ``"cminhash"``: one permutation of a ``2**20`` universe, circulantly
        shifted (Li & Li, 2022).

    Returns
    -------
    numpy.ndarray
        ``(n_rows, num_perm)`` ``uint64`` signature matrix.
    """
    if num_perm < 1:
        raise ValueError(f"`num_perm` must be >= 1, got {num_perm}.")
    rows = np.asarray(rows, dtype=np.int64)
    h = np.asarray(hashes, dtype=np.uint64)
    sig = np.full((n_rows, num_perm), EMPTY_SLOT, dtype=np.uint64)
    if h.size == 0:
        return sig
    if rows.size > 1 and bool((np.diff(rows) < 0).any()):
        raise ValueError("`rows` must be sorted ascending.")
    starts, ids = _segment_starts(rows)
    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_ELEMS // max(1, h.size))
    if variant == "minhash":
        a = rng.integers(1, int(_P31), size=num_perm, dtype=np.uint64)
        b = rng.integers(0, int(_P31), size=num_perm, dtype=np.uint64)
        x = (h % _P31)[:, None]
        for lo in range(0, num_perm, chunk):
            hi = min(num_perm, lo + chunk)
            vals = (a[None, lo:hi] * x + b[None, lo:hi]) % _P31
            sig[ids, lo:hi] = np.minimum.reduceat(vals, starts, axis=0)
    elif variant == "cminhash":
        universe = _CMINHASH_UNIVERSE
        pi = rng.permutation(universe).astype(np.uint64)
        sigma = (h % np.uint64(universe)).astype(np.int64)[:, None]
        shifts = np.arange(num_perm, dtype=np.int64)
        for lo in range(0, num_perm, chunk):
            hi = min(num_perm, lo + chunk)
            vals = pi[(sigma + shifts[None, lo:hi]) % universe]
            sig[ids, lo:hi] = np.minimum.reduceat(vals, starts, axis=0)
    else:
        raise ValueError(f"unknown variant {variant!r}; use 'minhash' or 'cminhash'.")
    return sig


def bbit(signatures: np.ndarray, b: int) -> np.ndarray:
    """Keep only the lowest ``b`` bits of every signature slot (b-bit MinHash).

    Parameters
    ----------
    signatures : numpy.ndarray
        ``uint64`` signature matrix.
    b : int
        Bits to keep, ``1 <= b <= 32``. The result uses the narrowest unsigned
        dtype that holds ``b`` bits -- the memory knob.

    Returns
    -------
    numpy.ndarray
    """
    if not 1 <= b <= 32:
        raise ValueError(f"`b` must be in [1, 32], got {b}.")
    dtype = np.uint8 if b <= 8 else np.uint16 if b <= 16 else np.uint32
    mask = np.uint64((1 << b) - 1)
    return (np.asarray(signatures, dtype=np.uint64) & mask).astype(dtype)


def estimate_jaccard(
    sig_a: np.ndarray, sig_b: np.ndarray, *, b_bits: int | None = None
) -> np.ndarray:
    """Estimate Jaccard similarity from two (stacks of) signatures.

    The fraction of equal slots is unbiased for full signatures. For b-bit
    signatures two different minima collide with probability ``2**-b``, which
    is removed with ``(P - 2**-b) / (1 - 2**-b)`` (clipped to ``[0, 1]``).

    Parameters
    ----------
    sig_a, sig_b : numpy.ndarray
        Broadcast-compatible signature arrays; the last axis is the slot axis.
    b_bits : int, optional
        Set when the signatures were compressed with :func:`bbit`.

    Returns
    -------
    numpy.ndarray
        Estimated similarity per leading index.
    """
    p = np.mean(np.asarray(sig_a) == np.asarray(sig_b), axis=-1)
    if b_bits is None:
        return p
    c = 2.0**-b_bits
    return np.clip((p - c) / (1.0 - c), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# SimHash
# --------------------------------------------------------------------------- #
def simhash_bits(X: np.ndarray, *, n_bits: int = 64, seed: int = 0) -> np.ndarray:
    """SimHash / signed-random-projection bits of numeric rows.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n_rows, n_features)`` float matrix, already centred/scaled as the
        caller intends; NaNs are treated as 0 (the centre).
    n_bits : int, default=64
        Number of hyperplanes.
    seed : int, default=0
        Seed of the Gaussian hyperplanes.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_bits)`` ``uint8`` 0/1 matrix. The fraction of agreeing
        bits between two rows estimates ``1 - angle / pi``.
    """
    if n_bits < 1:
        raise ValueError(f"`n_bits` must be >= 1, got {n_bits}.")
    Xf = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
    rng = np.random.default_rng(seed)
    planes = rng.standard_normal((Xf.shape[1], n_bits))
    return (Xf @ planes >= 0.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# LSH banding
# --------------------------------------------------------------------------- #
def lsh_params(
    threshold: float,
    num_perm: int,
    *,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> tuple[int, int]:
    """Choose ``(bands, rows)`` minimising weighted false-positive/negative mass.

    For every ``b * r <= num_perm`` the candidate probability of a pair with
    similarity ``s`` is ``1 - (1 - s**r)**b``. The false-positive mass is its
    integral over ``[0, threshold]`` and the false-negative mass the integral
    of the complement over ``[threshold, 1]``; the pair minimising the
    weighted sum is returned.

    Parameters
    ----------
    threshold : float
        Target similarity in ``(0, 1]``.
    num_perm : int
        Signature length available for banding.
    false_positive_weight, false_negative_weight : float
        Relative costs. Candidates are verified downstream, so the defaults
        weigh both equally.

    Returns
    -------
    (bands, rows) : tuple of int
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"`threshold` must be in (0, 1], got {threshold!r}.")
    grid_lo = np.linspace(0.0, threshold, 101)
    grid_hi = np.linspace(threshold, 1.0, 101)
    best: tuple[float, int, int] | None = None
    for b in range(1, num_perm + 1):
        r = np.arange(1, num_perm // b + 1)
        if r.size == 0:
            continue
        p_lo = 1.0 - (1.0 - grid_lo[None, :] ** r[:, None]) ** b
        p_hi = 1.0 - (1.0 - grid_hi[None, :] ** r[:, None]) ** b
        fp = _trapezoid(p_lo, grid_lo)
        fn = _trapezoid(1.0 - p_hi, grid_hi)
        err = false_positive_weight * fp + false_negative_weight * fn
        k = int(np.argmin(err))
        if best is None or err[k] < best[0] - 1e-15:
            best = (float(err[k]), b, int(r[k]))
    assert best is not None
    return best[1], best[2]


def lsh_candidate_pairs(
    signatures: np.ndarray,
    *,
    bands: int,
    rows: int,
    block: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    max_pairs: int = 50_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate pairs: rows that agree on every slot of at least one band.

    Parameters
    ----------
    signatures : numpy.ndarray
        ``(n, k)`` integer signature matrix with ``bands * rows <= k``.
    bands, rows : int
        Banding geometry (see :func:`lsh_params`).
    block : numpy.ndarray, optional
        Integer blocking code per row; pairs are only formed within a block.
    valid : numpy.ndarray, optional
        Boolean mask; rows where it is False never become candidates.
    max_pairs : int, default=50_000_000
        Safety valve on the number of within-bucket pairs.

    Returns
    -------
    (i, j) : tuple of numpy.ndarray
        Unique ``int64`` pairs with ``i < j``.

    Raises
    ------
    ValueError
        If the geometry does not fit the signature or the buckets would emit
        more than ``max_pairs`` pairs.
    """
    sig = np.asarray(signatures)
    n, k = sig.shape
    if bands < 1 or rows < 1 or bands * rows > k:
        raise ValueError(
            f"bands={bands} x rows={rows} does not fit a signature of length {k}."
        )
    idx = np.arange(n, dtype=np.int64)
    keep = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    blk = np.zeros(n, dtype=np.int64) if block is None else np.asarray(block)
    pieces: list[pl.DataFrame] = []
    total = 0
    for band in range(bands):
        cols = {f"s{j}": sig[keep, band * rows + j] for j in range(rows)}
        frame = pl.DataFrame({"row": idx[keep], "blk": blk[keep], **cols})
        buckets = (
            frame.group_by(["blk", *cols])
            .agg(pl.col("row"))
            .filter(pl.col("row").list.len() > 1)
        )
        if buckets.height == 0:
            continue
        sizes = buckets.get_column("row").list.len().to_numpy().astype(np.int64)
        total += int((sizes * (sizes - 1) // 2).sum())
        if total > max_pairs:
            raise ValueError(
                f"LSH buckets would emit more than max_pairs={max_pairs:,} "
                "candidate pairs. Raise `threshold`, block on a scope, or "
                "collapse exact duplicates first."
            )
        members = explode(
            buckets.with_row_index("bucket").select("bucket", "row"), "row"
        )
        pairs = (
            members.join(members, on="bucket", suffix="_r")
            .filter(pl.col("row") < pl.col("row_r"))
            .select(pl.col("row").alias("i"), pl.col("row_r").alias("j"))
        )
        pieces.append(pairs)
    if not pieces:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    allp = pl.concat(pieces).unique().sort(["i", "j"])
    return (
        allp.get_column("i").to_numpy().astype(np.int64),
        allp.get_column("j").to_numpy().astype(np.int64),
    )


def exact_jaccard(
    token_frame: pl.DataFrame, i: np.ndarray, j: np.ndarray, *, row_col: str
) -> np.ndarray:
    """Exact Jaccard similarity of the token-hash sets of each pair ``(i, j)``.

    Parameters
    ----------
    token_frame : polars.DataFrame
        ``[row_col, "h"]`` unique per row (see :func:`token_hashes`).
    i, j : numpy.ndarray
        Pair endpoints (row ids).
    row_col : str
        Row-id column name in ``token_frame``.

    Returns
    -------
    numpy.ndarray
        ``float64`` similarity per pair (0 for two empty sets).
    """
    ii = np.asarray(i, dtype=np.int64)
    jj = np.asarray(j, dtype=np.int64)
    if ii.size == 0:
        return np.zeros(0, dtype=np.float64)
    tok = token_frame.select(pl.col(row_col).cast(pl.Int64).alias("row"), pl.col("h"))
    sizes = tok.group_by("row").agg(pl.len().cast(pl.Int64).alias("n"))
    pairs = pl.DataFrame({"pid": np.arange(ii.size, dtype=np.int64), "i": ii, "j": jj})
    inter = (
        pairs.join(tok, left_on="i", right_on="row")
        .join(tok, left_on=["j", "h"], right_on=["row", "h"], how="semi")
        .group_by("pid")
        .agg(pl.len().cast(pl.Int64).alias("inter"))
    )
    scored = (
        pairs.join(inter, on="pid", how="left")
        .join(sizes.rename({"row": "i", "n": "n_i"}), on="i", how="left")
        .join(sizes.rename({"row": "j", "n": "n_j"}), on="j", how="left")
        .with_columns(pl.col("inter", "n_i", "n_j").fill_null(0))
        .sort("pid")
    )
    inter_v = scored.get_column("inter").to_numpy().astype(np.float64)
    union = (
        scored.get_column("n_i").to_numpy() + scored.get_column("n_j").to_numpy()
    ).astype(np.float64) - inter_v
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter_v / union, 0.0)
    return out
