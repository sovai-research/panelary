"""Pairwise string similarity with an optional ``rapidfuzz`` fast path.

:func:`string_similarity` scores aligned pairs ``(a[i], b[i])`` in ``[0, 1]``.
The default backend is ``"auto"``: ``rapidfuzz`` (MIT, ``pip install
'panelary[fuzzy]'``) when it is installed, otherwise the dependency-free
fallback here. The capability is never gated on the extra.

Metrics (fallback implementations are clean-room, from the definitions):

``"levenshtein"``
    ``1 - d / max(len(a), len(b))`` with ``d`` the edit distance
    (insert/delete/substitute, unit cost). Vectorised across pairs in numpy:
    one dynamic-programming row per character of the longer side.
``"jaro"`` / ``"jaro_winkler"``
    Jaro (1989) similarity; Winkler (1990) adds ``l * 0.1 * (1 - jaro)`` for a
    common prefix of ``l <= 4`` characters when ``jaro > 0.7`` (the original
    boost threshold). Pure Python per pair in the fallback.
``"token_jaccard"``
    Jaccard similarity of whitespace-token sets (Polars list set ops).
``"exact"``
    1.0 if equal else 0.0.

Missing values (``None``) score ``NaN`` so a caller can drop the field from a
weighted comparison instead of treating "unknown" as "different".
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from panelary.clean._common import check_choice

__all__ = ["string_similarity", "levenshtein_distance", "jaro_winkler"]

Metric = Literal["levenshtein", "jaro", "jaro_winkler", "token_jaccard", "exact"]
METRICS: tuple[str, ...] = (
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "token_jaccard",
    "exact",
)
Backend = Literal["auto", "numpy", "rapidfuzz"]


def _codes(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pad strings to a ``(n, L)`` int32 code-point matrix plus lengths."""
    lens = np.array([len(s) for s in strings], dtype=np.int64)
    width = int(lens.max()) if lens.size else 0
    mat = np.full((len(strings), max(width, 1)), -1, dtype=np.int32)
    for i, s in enumerate(strings):
        if s:
            mat[i, : len(s)] = np.frombuffer(s.encode("utf-32-le"), dtype=np.int32)
    return mat, lens


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> np.ndarray:
    """Edit distance of each aligned pair, vectorised across pairs.

    Parameters
    ----------
    a, b : sequence of str
        Equal-length sequences (no ``None``).

    Returns
    -------
    numpy.ndarray
        ``int64`` distances.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}.")
    n = len(a)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    A, la = _codes(a)
    B, lb = _codes(b)
    Lb = B.shape[1]
    prev = np.tile(np.arange(Lb + 1, dtype=np.int64), (n, 1))
    out = lb.copy()  # la == 0 -> distance is len(b)
    for i in range(1, int(la.max()) + 1):
        cur = np.empty_like(prev)
        cur[:, 0] = i
        ai = A[:, i - 1][:, None]
        sub = prev[:, :-1] + (ai != B).astype(np.int64)
        dele = prev[:, 1:] + 1
        best = np.minimum(sub, dele)
        # Insertions chain left-to-right: cur[j] = min(best[j], cur[j-1] + 1).
        # Equivalent closed form: min over k <= j of (best[k] - k) + j, with
        # cur[0] = i contributing (i - 0).
        j = np.arange(1, Lb + 1, dtype=np.int64)
        shifted = np.concatenate([np.full((n, 1), i, dtype=np.int64), best - j], axis=1)
        cur[:, 1:] = np.minimum.accumulate(shifted, axis=1)[:, 1:] + j
        done = la == i
        if done.any():
            out[done] = cur[done, lb[done]]
        prev = cur
    return out


def _jaro(s1: str, s2: str) -> float:
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    reach = max(max(l1, l2) // 2 - 1, 0)
    used = [False] * l2
    m1: list[str] = []
    for i, ch in enumerate(s1):
        lo, hi = max(0, i - reach), min(l2, i + reach + 1)
        for j in range(lo, hi):
            if not used[j] and s2[j] == ch:
                used[j] = True
                m1.append(ch)
                break
    m = len(m1)
    if m == 0:
        return 0.0
    m2 = [s2[j] for j in range(l2) if used[j]]
    out_of_order = sum(x != y for x, y in zip(m1, m2, strict=True))
    # Transpositions are counted whole (``out_of_order // 2``), as rapidfuzz,
    # jellyfish and Apache Commons do. Halving an odd count to a fraction is the
    # other reading of Jaro (1989); it disagrees with rapidfuzz on exactly those
    # pairs, which would make `backend="auto"` depend on what is installed.
    t = out_of_order // 2
    return (m / l1 + m / l2 + (m - t) / m) / 3.0


def jaro_winkler(
    s1: str, s2: str, *, prefix_weight: float = 0.1, winkler: bool = True
) -> float:
    """Jaro(-Winkler) similarity of two strings (pure Python).

    Parameters
    ----------
    s1, s2 : str
        The strings.
    prefix_weight : float, default=0.1
        Winkler's scaling factor ``p``.
    winkler : bool, default=True
        Apply the prefix boost (when Jaro exceeds 0.7).

    Returns
    -------
    float
    """
    j = _jaro(s1, s2)
    if not winkler or j <= 0.7:
        return j
    prefix = 0
    for x, y in zip(s1[:4], s2[:4], strict=False):
        if x != y:
            break
        prefix += 1
    return j + prefix * prefix_weight * (1.0 - j)


def _rapidfuzz_available() -> bool:
    return importlib.util.find_spec("rapidfuzz") is not None


def _rapidfuzz_scores(a: list[str], b: list[str], metric: str) -> np.ndarray:
    from panelary._internal._deps import require

    rf_distance = require("rapidfuzz.distance", extra="fuzzy", feature="fuzzy matching")
    scorer = {
        "levenshtein": rf_distance.Levenshtein.normalized_similarity,
        "jaro": rf_distance.Jaro.similarity,
        "jaro_winkler": rf_distance.JaroWinkler.similarity,
    }[metric]
    rf_process = require("rapidfuzz.process", extra="fuzzy", feature="fuzzy matching")
    if hasattr(rf_process, "cpdist"):
        return np.asarray(rf_process.cpdist(a, b, scorer=scorer), dtype=np.float64)
    return np.array([scorer(x, y) for x, y in zip(a, b, strict=True)], dtype=np.float64)


def string_similarity(
    a: Sequence[str | None] | pl.Series,
    b: Sequence[str | None] | pl.Series,
    *,
    metric: Metric = "levenshtein",
    backend: Backend = "auto",
) -> np.ndarray:
    """Similarity in ``[0, 1]`` of each aligned pair ``(a[i], b[i])``.

    Parameters
    ----------
    a, b : sequence of str or polars.Series
        Equal-length string sequences; ``None`` marks a missing value.
    metric : {"levenshtein", "jaro", "jaro_winkler", "token_jaccard", "exact"}
        See the module docs.
    backend : {"auto", "numpy", "rapidfuzz"}, default="auto"
        ``"rapidfuzz"`` requires the ``fuzzy`` extra; ``"auto"`` uses it
        when installed. ``token_jaccard`` / ``exact`` always use Polars.

    Returns
    -------
    numpy.ndarray
        ``float64`` scores; ``NaN`` where either side is missing.
    """
    check_choice("metric", metric, METRICS)
    check_choice("backend", backend, ("auto", "numpy", "rapidfuzz"))
    sa = a if isinstance(a, pl.Series) else pl.Series("a", list(a), dtype=pl.String)
    sb = b if isinstance(b, pl.Series) else pl.Series("b", list(b), dtype=pl.String)
    sa, sb = sa.cast(pl.String).rename("a"), sb.cast(pl.String).rename("b")
    if sa.len() != sb.len():
        raise ValueError(f"length mismatch: {sa.len()} vs {sb.len()}.")
    n = sa.len()
    out = np.full(n, np.nan, dtype=np.float64)
    ok = (sa.is_not_null() & sb.is_not_null()).to_numpy()
    if not ok.any():
        return out
    xa = sa.filter(pl.Series(ok)).to_list()
    xb = sb.filter(pl.Series(ok)).to_list()
    if metric == "exact":
        out[ok] = (np.array(xa, dtype=object) == np.array(xb, dtype=object)).astype(
            np.float64
        )
        return out
    if metric == "token_jaccard":
        frame = pl.DataFrame(
            {"a": xa, "b": xb}, schema={"a": pl.String, "b": pl.String}
        )
        ta = (
            pl.col("a")
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element() != ""))
        )
        tb = (
            pl.col("b")
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element() != ""))
        )
        scored = frame.select(
            ta.list.set_intersection(tb).list.len().alias("i"),
            ta.list.set_union(tb).list.len().alias("u"),
        )
        i = scored.get_column("i").to_numpy().astype(np.float64)
        u = scored.get_column("u").to_numpy().astype(np.float64)
        out[ok] = np.where(u > 0, i / np.where(u > 0, u, 1.0), 1.0)
        return out
    use_rf = backend == "rapidfuzz" or (backend == "auto" and _rapidfuzz_available())
    if use_rf:
        out[ok] = _rapidfuzz_scores(xa, xb, metric)
        return out
    if metric == "levenshtein":
        dist = levenshtein_distance(xa, xb).astype(np.float64)
        longest = np.maximum([len(s) for s in xa], [len(s) for s in xb]).astype(
            np.float64
        )
        out[ok] = np.where(
            longest > 0, 1.0 - dist / np.where(longest > 0, longest, 1.0), 1.0
        )
        return out
    wink = metric == "jaro_winkler"
    out[ok] = np.array(
        [jaro_winkler(x, y, winkler=wink) for x, y in zip(xa, xb, strict=True)],
        dtype=np.float64,
    )
    return out
