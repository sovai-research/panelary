"""Sequential bootstrap (AFML 4.5): resample labels in proportion to uniqueness.

An i.i.d. bootstrap of overlapping labels draws near-duplicates: two labels
sharing 19 of 20 days are nearly the same observation, so a bag built from them
is far less diverse than its size suggests. The sequential bootstrap draws one
label at a time with probability proportional to its average uniqueness *given
the labels already drawn*, so redundant labels become unlikely once their
neighbours are in the bag.

The textbook implementation rebuilds a dense ``rows x labels`` indicator matrix
on every draw (O(N^3 T)). Here a draw only changes the labels that overlap it;
those are recomputed from scratch (so every weight is a function of the current
counts, with no accumulated rounding) and sampled through a two-level
inverse-CDF. The numba kernel and its numpy twin make bitwise-identical draws.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from panelary.core._spans import SpanTable, _spans_from_t1, _uniqueness
from panelary.sample._kernels import _seq_boot

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["sequential_bootstrap"]

_METHODS = ("exact", "uniqueness_iid")


def _stable_hash(key: Any) -> int:
    """A process-independent 64-bit hash of ``key`` (Python's ``hash`` is salted)."""
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _entity_seed(base: np.random.SeedSequence, key: Any) -> np.random.SeedSequence:
    """Per-entity stream: depends only on the base seed and the entity key."""
    h = _stable_hash(key)
    words = [int(w) for w in base.generate_state(4)]
    return np.random.SeedSequence(words + [h & 0xFFFFFFFF, h >> 32])


def _resolve_n_draws(n_draws: int | float | None, n: int) -> int:
    if n_draws is None:
        return n
    if isinstance(n_draws, bool):
        raise TypeError("`n_draws` must be an int, a float in (0, 1], or None.")
    if isinstance(n_draws, float):
        if not 0.0 < n_draws <= 1.0:
            raise ValueError(f"a float `n_draws` is a share in (0, 1], got {n_draws}.")
        return max(1, int(round(n_draws * n))) if n else 0
    if n_draws < 0:
        raise ValueError(f"`n_draws` must be >= 0, got {n_draws}.")
    return int(n_draws)


def _iid_by_uniqueness(spans: SpanTable, u: NDArray[np.float64]) -> NDArray[np.int64]:
    """i.i.d. draws with ``p_i`` proportional to average uniqueness."""
    cs = np.cumsum(_uniqueness(spans))
    idx = np.searchsorted(cs, u * cs[-1], side="right").astype(np.int64)
    return np.minimum(idx, len(spans) - 1)


def _draw(
    spans: SpanTable, n_draws: int, ss: np.random.SeedSequence, method: str
) -> NDArray[np.int64]:
    """Span indices (into ``spans``) for one pooled bootstrap."""
    if len(spans) == 0 or n_draws == 0:
        return np.zeros(0, dtype=np.int64)
    u = np.random.default_rng(ss).random(n_draws)
    if method == "exact":
        return _seq_boot(spans.start, spans.end, u)
    return _iid_by_uniqueness(spans, u)


def _bootstrap_rows(
    spans: SpanTable,
    entity_keys: list[Any],
    *,
    n_draws: int | float | None,
    ss: np.random.SeedSequence,
    method: str,
    stratify: str | None,
) -> NDArray[np.int64]:
    """Row indices of the drawn labels (pooled or stratified by entity)."""
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {_METHODS}, got {method!r}.")
    if stratify is None:
        k = _resolve_n_draws(n_draws, len(spans))
        return spans.label_row[_draw(spans, k, ss, method)]
    if stratify != "entity":
        raise ValueError(f'`stratify` must be None or "entity", got {stratify!r}.')
    n_total = len(spans)
    k_total = _resolve_n_draws(n_draws, n_total)
    out: list[NDArray[np.int64]] = []
    codes = spans.entity_code
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
        sub = spans.slice(a, b)  # an entity's spans are contiguous in start order
        k = (b - a) if n_draws is None else int(round(k_total * (b - a) / n_total))
        key = entity_keys[int(codes[a])]
        out.append(sub.label_row[_draw(sub, k, _entity_seed(ss, key), method)])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def sequential_bootstrap(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    n_draws: int | float | None = None,
    seed: int = 0,
    method: str = "exact",
    stratify: str | None = None,
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
) -> NDArray[np.int64]:
    """Draw labels by AFML's sequential bootstrap; return their row indices.

    Draw ``k`` picks label ``i`` with probability proportional to
    ``mean_{r in span_i} 1 / (c_r + 1)``, where ``c_r`` counts the labels
    already drawn (with multiplicity) that cover row ``r``. A label whose rows
    are already well covered becomes unlikely, so a bag is as close to
    independent draws as the overlap allows.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Labelled panel; spans come from ``t1`` exactly as in
        :func:`panelary.weights.spans` (null, censored and past-the-data labels
        are never drawn). Inside cross-validation pass the **training** fold
        only.
    t1 : str, default "t1"
        Label end-time column.
    n_draws : int, float or None, default None
        Number of draws: ``None`` = the number of labels; a float in ``(0, 1]``
        is a share of it.
    seed : int, default 0
        Seed of the uniform stream (generated once, up front, and consumed
        identically by both backends).
    method : {"exact", "uniqueness_iid"}, default "exact"
        ``"exact"``: the sequential bootstrap. ``"uniqueness_iid"``: i.i.d.
        draws with ``p_i`` proportional to the labels' average uniqueness
        (among themselves) -- crowded labels are down-weighted, but redundancy
        among the *drawn* labels is ignored (the exact method's first draw is
        uniform; its later draws adapt to the bag); O(N + n_draws log N).
        AFML ch. 6's other alternative, a plain bagger with
        ``max_samples = mean uniqueness``, needs no code here.
    stratify : {None, "entity"}, default None
        ``None`` pools all labels (spans never cross entities, so the pooled
        draw is exact). ``"entity"`` runs an independent bootstrap per entity
        with ``N_e`` draws (or its share of ``n_draws``), each seeded from
        ``(seed, entity)`` by a stable hash -- unchanged when entities are
        added or reordered, and across processes.
    entity, time : str, optional
        Keys; default to the first and second columns.
    censored : str or None, default "censored"
        Boolean column of unresolved labels to exclude (ignored if absent).

    Returns
    -------
    numpy.ndarray of int64
        Row indices into the ``(entity, time)``-sorted frame, with
        multiplicity, in draw order (grouped by entity when stratified).

    Notes
    -----
    Cost per draw is O(K h + sqrt(N)) for K overlapping labels of length up to
    ``h``: about 1 us with the ``fast`` extra (numba) and 30 us with numpy.
    The pooled draw loop is inherently serial; at tens of millions of labels
    stratify by entity or cap ``n_draws``.

    Draws agree with the dense-matrix definition given the same uniforms,
    except when a uniform lands within rounding (~1e-13) of a CDF boundary.

    References
    ----------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    section 4.5.
    """
    frame, spans = _spans_from_t1(
        df, t1=t1, entity=entity, time=time, censored=censored
    )
    entity_col = entity if entity is not None else frame.columns[0]
    keys = frame.get_column(entity_col).unique(maintain_order=True).to_list()
    return _bootstrap_rows(
        spans,
        keys,
        n_draws=n_draws,
        ss=np.random.SeedSequence(seed),
        method=method,
        stratify=stratify,
    )
