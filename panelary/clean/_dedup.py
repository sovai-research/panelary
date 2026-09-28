"""Panel-global, point-in-time near-duplicate detection and removal.

The pipeline, for every method:

1. **Exact pre-pass** -- rows with identical (null-equal) content inside the
   same scope collapse onto their earliest member. Everything after this runs
   on those representatives only, which keeps LSH buckets small.
2. **Sketch** -- MinHash / C-MinHash signatures of each row's token set
   (``"minhash"`` / ``"cminhash"``), or SimHash bits of its standardised
   numeric vector (``"simhash"``). See :mod:`panelary.clean._sketch`.
3. **LSH banding** -- candidate pairs agree on a whole band.
4. **Verify** -- exact Jaccard (or exact cosine for SimHash) on the candidate
   pairs, keeping those at or above ``threshold``.
5. **Resolve** -- either the point-in-time rule (a row is a duplicate iff it
   near-duplicates an *earlier* row; pairwise, so it is prefix-invariant) or
   connected components via :func:`~panelary.clean._cluster.connected_components`.

"Earlier" is the panel's point-in-time order: by ``time``, then input order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from panelary.clean._cluster import connected_components, later_endpoints
from panelary.clean._common import (
    ROW,
    check_choice,
    check_unit_interval,
    group_codes,
    order_ranks,
    resolve_columns,
    rewrap,
    scope_columns,
    to_panel,
    with_row_ids,
)
from panelary.clean._exact import exact_groups
from panelary.clean._sketch import (
    estimate_jaccard,
    exact_jaccard,
    lsh_candidate_pairs,
    lsh_params,
    minhash_signatures,
    simhash_bits,
    token_hashes,
    tokenize,
)
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

if TYPE_CHECKING:
    from panelary.clean._survivorship import Survivorship

__all__ = [
    "Deduplicator",
    "dedup",
    "near_duplicate_clusters",
    "near_duplicate_pairs",
]

Method = Literal["exact", "minhash", "cminhash", "simhash"]
Verify = Literal["exact", "estimate", "none"]
_METHODS: tuple[str, ...] = ("exact", "minhash", "cminhash", "simhash")
_VERIFY: tuple[str, ...] = ("exact", "estimate", "none")
_TOKENIZERS: tuple[str, ...] = ("auto", "cells", "char", "word")


# --------------------------------------------------------------------------- #
# The similarity graph
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Graph:
    """Near-duplicate structure of one frame (row ids ``0 .. n - 1``).

    ``rep`` maps every row to its exact-group representative; ``ei``/``ej``
    are verified near-duplicate edges **between representatives** with their
    similarity ``sim``; ``rank`` is the point-in-time order used throughout.
    """

    n: int
    rep: np.ndarray
    ei: np.ndarray
    ej: np.ndarray
    sim: np.ndarray
    rank: np.ndarray

    def all_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Member->representative stars plus the representative edges."""
        members = np.flatnonzero(self.rep != np.arange(self.n))
        return (
            np.concatenate([members, self.ei]),
            np.concatenate([self.rep[members], self.ej]),
        )

    def point_in_time_duplicates(self) -> np.ndarray:
        """Rows that near-duplicate some row earlier in ``rank`` (pairwise)."""
        ei, ej = self.all_edges()
        return later_endpoints(self.n, ei, ej, self.rank)

    def components(self) -> np.ndarray:
        """Cluster id per row: the row id of the component's earliest member."""
        ei, ej = self.all_edges()
        # Union-find in rank space so the label (smallest node) is the earliest.
        label = connected_components(self.n, self.rank[ei], self.rank[ej])
        row_of_rank = np.empty(self.n, dtype=np.int64)
        row_of_rank[self.rank] = np.arange(self.n, dtype=np.int64)
        return row_of_rank[label[self.rank]]


def _resolve_tokenizer(tokenizer: str, frame: pl.DataFrame, cols: list[str]) -> str:
    check_choice("tokenizer", tokenizer, _TOKENIZERS)
    if tokenizer != "auto":
        return tokenizer
    if len(cols) == 1 and frame.schema[cols[0]] == pl.String:
        return "char"
    return "cells"


def _numeric_matrix(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    bad = [c for c in cols if not frame.schema[c].is_numeric()]
    if bad:
        raise ValueError(
            f"method='simhash' needs numeric content columns; {bad} are not. "
            "Pass `columns=` with numeric features, or use method='minhash'."
        )
    return (
        frame.select(pl.col(c).cast(pl.Float64) for c in cols)
        .to_numpy()
        .astype(np.float64, copy=False)
    )


def _standardise_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column centre / scale for SimHash (NaN-aware; zero scale -> 1)."""
    with np.errstate(invalid="ignore"):
        if X.shape[0] == 0:
            return np.zeros(X.shape[1]), np.ones(X.shape[1])
        centre = np.nanmean(X, axis=0) if np.isfinite(X).any() else np.zeros(X.shape[1])
        scale = np.nanstd(X, axis=0)
    centre = np.nan_to_num(centre, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0)
    scale[scale <= 0.0] = 1.0
    return centre, scale


def _lsh_weights(verify: str) -> dict[str, float]:
    """LSH geometry costs: when candidates are verified, a false positive
    only costs a comparison while a false negative is a missed duplicate, so
    the S-curve is pushed towards recall."""
    if verify == "none":
        return {"false_positive_weight": 0.5, "false_negative_weight": 0.5}
    return {"false_positive_weight": 0.1, "false_negative_weight": 0.9}


def _build_graph(
    frame: pl.DataFrame,
    *,
    time: str,
    cols: list[str],
    scope_cols: list[str],
    method: str,
    threshold: float,
    num_perm: int,
    seed: int,
    tokenizer: str,
    ngram: int,
    verify: str,
    b_bits: int | None,
    n_bits: int,
    reverse: bool = False,
    simhash_stats: tuple[np.ndarray, np.ndarray] | None = None,
    max_pairs: int = 50_000_000,
) -> _Graph:
    """Compute the near-duplicate graph of ``frame`` (which carries ``ROW``)."""
    n = frame.height
    rank = order_ranks(frame, time, reverse=reverse)
    empty = np.zeros(0, dtype=np.int64)
    if n == 0:
        return _Graph(0, empty, empty, empty, np.zeros(0), rank)
    key_cols = [*scope_cols, *cols]
    rep = exact_groups(frame, key_cols, row_col=ROW, rank=rank)
    if method == "exact":
        return _Graph(n, rep, empty, empty.copy(), np.zeros(0), rank)

    reps = np.flatnonzero(rep == np.arange(n))
    sub = frame.filter(pl.col(ROW).is_in(pl.Series(reps).implode())).sort(ROW)
    local = np.arange(reps.size, dtype=np.int64)
    sub = sub.with_columns(pl.Series("__local", local))
    block = group_codes(sub, scope_cols)

    if method in ("minhash", "cminhash"):
        tok = tokenize(
            sub,
            cols,
            row_col="__local",
            tokenizer=_resolve_tokenizer(tokenizer, frame, cols),  # type: ignore[arg-type]
            ngram=ngram,
        )
        th = token_hashes(tok, row_col="__local", seed=seed)
        rows_arr = th.get_column("__local").to_numpy().astype(np.int64)
        sig = minhash_signatures(
            rows_arr,
            th.get_column("h").to_numpy(),
            reps.size,
            num_perm=num_perm,
            seed=seed,
            variant=method,  # type: ignore[arg-type]
        )
        valid = np.zeros(reps.size, dtype=bool)
        valid[rows_arr] = True
        if b_bits is not None:
            from panelary.clean._sketch import bbit

            sig_used: np.ndarray = bbit(sig, b_bits)
        else:
            sig_used = sig
        bands, rows = lsh_params(threshold, num_perm, **_lsh_weights(verify))
        li, lj = lsh_candidate_pairs(
            sig_used,
            bands=bands,
            rows=rows,
            block=block,
            valid=valid,
            max_pairs=max_pairs,
        )
        if verify == "exact":
            sim = exact_jaccard(th, li, lj, row_col="__local")
        else:
            sim = estimate_jaccard(sig_used[li], sig_used[lj], b_bits=b_bits)
    else:  # simhash
        X = _numeric_matrix(sub, cols)
        centre, scale = (
            simhash_stats
            if simhash_stats is not None
            else (_standardise_stats(_numeric_matrix(frame, cols)))
        )
        Z = np.nan_to_num((X - centre) / scale, nan=0.0)
        bits = simhash_bits(Z, n_bits=n_bits, seed=seed)
        p_agree = 1.0 - float(np.arccos(np.clip(threshold, -1.0, 1.0))) / np.pi
        bands, rows = lsh_params(p_agree, n_bits, **_lsh_weights(verify))
        li, lj = lsh_candidate_pairs(
            bits, bands=bands, rows=rows, block=block, max_pairs=max_pairs
        )
        if verify == "exact":
            norms = np.linalg.norm(Z, axis=1)
            dots = np.einsum("ij,ij->i", Z[li], Z[lj])
            denom = norms[li] * norms[lj]
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = np.where(denom > 0, dots / denom, 0.0)
        else:
            agree = np.mean(bits[li] == bits[lj], axis=1)
            sim = np.cos(np.pi * (1.0 - agree))

    if verify != "none":
        ok = sim >= threshold - 1e-12
        li, lj, sim = li[ok], lj[ok], sim[ok]
    return _Graph(n, rep, reps[li], reps[lj], np.asarray(sim, dtype=np.float64), rank)


def _validate_common(
    method: str,
    threshold: float,
    num_perm: int,
    verify: str,
    b_bits: int | None,
    n_bits: int,
    ngram: int,
) -> None:
    check_choice("method", method, _METHODS)
    check_choice("verify", verify, _VERIFY)
    check_unit_interval("threshold", threshold)
    if not isinstance(num_perm, int) or num_perm < 1:
        raise ValueError(f"`num_perm` must be a positive integer, got {num_perm!r}.")
    if b_bits is not None and not (isinstance(b_bits, int) and 1 <= b_bits <= 32):
        raise ValueError(f"`b_bits` must be an int in [1, 32] or None, got {b_bits!r}.")
    if not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError(f"`n_bits` must be a positive integer, got {n_bits!r}.")
    if not isinstance(ngram, int) or ngram < 1:
        raise ValueError(f"`ngram` must be a positive integer, got {ngram!r}.")


def _graph_for(
    df: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None,
    time: str | None,
    columns: Sequence[str] | str | None,
    method: str,
    threshold: float,
    num_perm: int,
    seed: int,
    tokenizer: str,
    ngram: int,
    verify: str,
    scope: str,
    b_bits: int | None,
    n_bits: int,
) -> tuple[PanelFrame, pl.DataFrame, _Graph]:
    _validate_common(method, threshold, num_perm, verify, b_bits, n_bits, ngram)
    panel = to_panel(df, entity, time)
    frame = with_row_ids(panel.collect())
    cols = resolve_columns(frame.columns, panel.entity_col, panel.time_col, columns)
    if not cols:
        raise ValueError(
            "no content columns to compare: pass `columns=` or give the frame "
            "at least one non-key column."
        )
    graph = _build_graph(
        frame,
        time=panel.time_col,
        cols=cols,
        scope_cols=scope_columns(scope, panel.entity_col, panel.time_col),
        method=method,
        threshold=threshold,
        num_perm=num_perm,
        seed=seed,
        tokenizer=tokenizer,
        ngram=ngram,
        verify=verify,
        b_bits=b_bits,
        n_bits=n_bits,
    )
    return panel, frame, graph


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def near_duplicate_clusters(
    df: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    columns: Sequence[str] | str | None = None,
    method: Method = "minhash",
    threshold: float = 0.8,
    num_perm: int = 128,
    seed: int = 0,
    tokenizer: Literal["auto", "cells", "char", "word"] = "auto",
    ngram: int = 3,
    verify: Verify = "exact",
    scope: Literal["global", "entity", "time", "key"] = "global",
    b_bits: int | None = None,
    n_bits: int = 64,
) -> pl.DataFrame:
    """Cluster the rows of a panel into near-duplicate groups (panel-global).

    Every row gets a cluster id; rows that are (transitively) near-duplicates
    of one another share it. This is the **diagnostic** view -- connected
    components read the whole panel, so a later row can merge two earlier
    clusters. Use :class:`Deduplicator` (point-in-time by default) to actually
    remove duplicates, and the ``cluster_id`` column here to check or enforce
    that no cluster straddles a train/test split (see
    :func:`~panelary.clean.straddling_clusters`).

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel. Bare frames use ``entity`` / ``time`` (or columns
        0 / 1).
    entity, time : str, optional
        Panel keys for a bare frame.
    columns : str | sequence of str, optional
        Content columns that define a row. Defaults to every non-key column,
        so rows at different times or entities can be duplicates of each other.
    method : {"minhash", "cminhash", "simhash", "exact"}, default="minhash"
        ``"exact"``: identical content only. ``"minhash"`` / ``"cminhash"``:
        Jaccard similarity of token sets (see ``tokenizer``). ``"simhash"``:
        cosine similarity of standardised numeric rows.
    threshold : float, default=0.8
        Similarity at or above which two rows are near-duplicates (Jaccard for
        MinHash, cosine for SimHash).
    num_perm : int, default=128
        MinHash signature length.
    seed : int, default=0
        Seed for every hash family and projection (determinism).
    tokenizer : {"auto", "cells", "char", "word"}, default="auto"
        How a row becomes a token set. ``"auto"`` shingles a single string
        column into character n-grams and otherwise uses one token per cell.
    ngram : int, default=3
        Shingle length for ``"char"`` / ``"word"``.
    verify : {"exact", "estimate", "none"}, default="exact"
        How LSH candidates are confirmed: exact Jaccard / cosine, the
        sketch estimate, or not at all.
    scope : {"global", "entity", "time", "key"}, default="global"
        Which rows may be compared: all (panel-global), same entity, same
        date, or same ``(entity, time)`` cell.
    b_bits : int, optional
        Compress MinHash slots to ``b_bits`` bits (b-bit MinHash).
    n_bits : int, default=64
        SimHash signature length.

    Returns
    -------
    polars.DataFrame
        One row per input row, **in input order**, with columns
        ``row_index`` (``Int64``, the input position), the entity and time
        keys, ``cluster_id`` (``Int64``: the ``row_index`` of the cluster's
        earliest member, so singletons carry their own index) and
        ``cluster_size`` (``Int64``).

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "id": ["a", "b", "c"],
    ...         "t": [1, 2, 3],
    ...         "text": ["acme corp ltd", "acme corp ltd.", "zebra"],
    ...     }
    ... )
    >>> near_duplicate_clusters(
    ...     df, entity="id", time="t", threshold=0.7
    ... )["cluster_id"].to_list()
    [0, 0, 2]
    """
    panel, frame, graph = _graph_for(
        df,
        entity=entity,
        time=time,
        columns=columns,
        method=method,
        threshold=threshold,
        num_perm=num_perm,
        seed=seed,
        tokenizer=tokenizer,
        ngram=ngram,
        verify=verify,
        scope=scope,
        b_bits=b_bits,
        n_bits=n_bits,
    )
    cid = graph.components()
    sizes = np.bincount(cid, minlength=graph.n) if graph.n else np.zeros(0, np.int64)
    return frame.select(
        pl.col(ROW).alias("row_index"), pl.col(panel.entity_col), pl.col(panel.time_col)
    ).with_columns(
        pl.Series("cluster_id", cid, dtype=pl.Int64),
        pl.Series("cluster_size", sizes[cid] if graph.n else cid, dtype=pl.Int64),
    )


def near_duplicate_pairs(
    df: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    columns: Sequence[str] | str | None = None,
    method: Method = "minhash",
    threshold: float = 0.8,
    num_perm: int = 128,
    seed: int = 0,
    tokenizer: Literal["auto", "cells", "char", "word"] = "auto",
    ngram: int = 3,
    verify: Verify = "exact",
    scope: Literal["global", "entity", "time", "key"] = "global",
    b_bits: int | None = None,
    n_bits: int = 64,
) -> pl.DataFrame:
    """Every near-duplicate pair of rows, with its similarity.

    Same parameters as :func:`near_duplicate_clusters`. Exact duplicates are
    reported with similarity ``1.0``.

    Returns
    -------
    polars.DataFrame
        Columns ``row_i``, ``row_j`` (``Int64`` input positions, ``row_i <
        row_j``) and ``similarity`` (``Float64``), sorted by ``(row_i, row_j)``.
    """
    _panel, _frame, graph = _graph_for(
        df,
        entity=entity,
        time=time,
        columns=columns,
        method=method,
        threshold=threshold,
        num_perm=num_perm,
        seed=seed,
        tokenizer=tokenizer,
        ngram=ngram,
        verify=verify,
        scope=scope,
        b_bits=b_bits,
        n_bits=n_bits,
    )
    return _expand_pairs(graph)


def _expand_pairs(graph: _Graph) -> pl.DataFrame:
    """All row pairs implied by the graph: within exact groups, and across the
    member sets of every verified representative edge."""
    n = graph.n
    groups = pl.DataFrame({"row": np.arange(n, dtype=np.int64), "rep": graph.rep})
    within = (
        groups.join(groups, on="rep", suffix="_r")
        .filter(pl.col("row") < pl.col("row_r"))
        .select(
            pl.col("row").alias("row_i"),
            pl.col("row_r").alias("row_j"),
            pl.lit(1.0, dtype=pl.Float64).alias("similarity"),
        )
    )
    edges = pl.DataFrame(
        {"a": graph.ei, "b": graph.ej, "similarity": graph.sim},
        schema={"a": pl.Int64, "b": pl.Int64, "similarity": pl.Float64},
    )
    across = (
        edges.join(groups.rename({"row": "ra"}), left_on="a", right_on="rep")
        .join(groups.rename({"row": "rb"}), left_on="b", right_on="rep")
        .select(
            pl.min_horizontal("ra", "rb").alias("row_i"),
            pl.max_horizontal("ra", "rb").alias("row_j"),
            pl.col("similarity"),
        )
    )
    return (
        pl.concat([within, across])
        .group_by(["row_i", "row_j"])
        .agg(pl.col("similarity").max())
        .sort(["row_i", "row_j"])
    )


# --------------------------------------------------------------------------- #
# The transformer
# --------------------------------------------------------------------------- #
class Deduplicator(PanelTransformer):
    """Remove (or flag) exact and near-duplicate rows of a panel.

    Deduplication is panel-global by default: a row is compared with rows of
    every entity at every date, because a vendor re-sending yesterday's record
    or the same filing mapped to two tickers is exactly the duplication that
    corrupts train-only statistics and leaks targets across a split.

    **The point-in-time rule (default).** With ``keep="first"`` and
    ``linkage="pairwise"`` a row is dropped iff it near-duplicates some row
    that came *earlier* (by ``time``, then input order). The decision is
    pairwise, so it depends only on the row and its past: the transform is
    prefix-invariant and ``leakage_safe``. After it runs, **no near-duplicate
    pair survives anywhere in the panel**, so none can straddle any train/test
    split built afterwards -- the split-aware guarantee, by construction.

    ``linkage="component"`` instead keeps one row per connected component,
    and ``keep="last"`` keeps latest occurrences; both read the future, so
    either sets this instance's ``leakage_safe`` to ``False``.

    Parameters
    ----------
    method : {"minhash", "cminhash", "simhash", "exact"}, default="minhash"
        See :func:`near_duplicate_clusters`.
    columns : str | sequence of str, optional
        Content columns. Defaults to every non-key column seen at fit time.
    threshold, num_perm, seed, tokenizer, ngram, verify, b_bits, n_bits
        See :func:`near_duplicate_clusters`.
    scope : {"global", "entity", "time", "key"}, default="global"
        Which rows may be compared. ``"key"`` with ``method="exact"`` removes
        repeated ``(entity, time)`` rows with identical content.
    keep : {"first", "last"}, default="first"
        Which occurrence survives.
    linkage : {"pairwise", "component"}, default="pairwise"
        Point-in-time pairwise rule, or one survivor per connected component.
    action : {"drop", "flag"}, default="drop"
        Drop duplicates, or keep every row and add a boolean ``flag_col``
        (plus ``cluster_id`` under ``linkage="component"``).
    flag_col : str, default="is_duplicate"
        Name of the flag column for ``action="flag"``.
    survivorship : Survivorship, optional
        With ``linkage="component"`` and ``action="drop"``, merge each
        component into a golden record (see :class:`~panelary.clean.Survivorship`)
        instead of keeping one row verbatim. Reads the whole component, so it
        is not leakage-safe.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        ``False`` for ``scope`` ``"global"`` / ``"time"`` (rows of one entity
        are dropped because of another's); ``True`` for ``"entity"`` / ``"key"``.
    leakage_safe : bool
        ``True`` only for ``keep="first"``, ``linkage="pairwise"`` and no
        ``survivorship``.
    feature_names_in_ : list of str
        The content columns resolved at fit time.
    center_, scale_ : numpy.ndarray or None
        SimHash standardisation statistics, learned from the fit panel only.
    n_duplicates_ : int or None
        Rows dropped/flagged by the most recent :meth:`transform`.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {"id": ["a", "a", "b"], "t": [1, 2, 1], "x": [1.0, 1.0, 2.0]}
    ... )
    >>> Deduplicator(method="exact", entity="id", time="t").fit_transform(
    ...     df
    ... ).collect().height
    2
    """

    panel_safe = False
    leakage_safe = True

    def __init__(
        self,
        *,
        method: Method = "minhash",
        columns: Sequence[str] | str | None = None,
        threshold: float = 0.8,
        num_perm: int = 128,
        seed: int = 0,
        tokenizer: Literal["auto", "cells", "char", "word"] = "auto",
        ngram: int = 3,
        verify: Verify = "exact",
        scope: Literal["global", "entity", "time", "key"] = "global",
        keep: Literal["first", "last"] = "first",
        linkage: Literal["pairwise", "component"] = "pairwise",
        action: Literal["drop", "flag"] = "drop",
        flag_col: str = "is_duplicate",
        b_bits: int | None = None,
        n_bits: int = 64,
        survivorship: Survivorship | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        _validate_common(method, threshold, num_perm, verify, b_bits, n_bits, ngram)
        check_choice("tokenizer", tokenizer, _TOKENIZERS)
        check_choice("scope", scope, ("global", "entity", "time", "key"))
        check_choice("keep", keep, ("first", "last"))
        check_choice("linkage", linkage, ("pairwise", "component"))
        check_choice("action", action, ("drop", "flag"))
        if survivorship is not None and (linkage != "component" or action != "drop"):
            raise ValueError(
                "`survivorship` merges whole clusters, so it needs "
                "linkage='component' and action='drop'."
            )
        self.method = method
        self.columns = columns
        self.threshold = threshold
        self.num_perm = num_perm
        self.seed = seed
        self.tokenizer = tokenizer
        self.ngram = ngram
        self.verify = verify
        self.scope = scope
        self.keep = keep
        self.linkage = linkage
        self.action = action
        self.flag_col = flag_col
        self.b_bits = b_bits
        self.n_bits = n_bits
        self.survivorship = survivorship
        # Per-instance overrides of the class defaults (see reduce.PCAFactors).
        self.panel_safe = scope in ("entity", "key")
        self.leakage_safe = (
            keep == "first" and linkage == "pairwise" and survivorship is None
        )
        self.feature_names_in_: list[str] = []
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_duplicates_: int | None = None

    def _fit(self, panel: PanelFrame) -> None:
        cols = resolve_columns(
            panel.columns, panel.entity_col, panel.time_col, self.columns
        )
        if not cols:
            raise ValueError(
                "Deduplicator: no content columns to compare; pass `columns=`."
            )
        self.feature_names_in_ = cols
        if self.method == "simhash":
            X = _numeric_matrix(panel.lazy().select(cols).collect(), cols)
            self.center_, self.scale_ = _standardise_stats(X)
        else:
            self.center_ = self.scale_ = None

    def _graph(self, frame: pl.DataFrame, panel: PanelFrame) -> _Graph:
        missing = [c for c in self.feature_names_in_ if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Deduplicator.transform: fitted content column(s) {missing} are "
                "missing from the frame."
            )
        stats = None
        if self.center_ is not None and self.scale_ is not None:
            stats = (self.center_, self.scale_)
        return _build_graph(
            frame,
            time=panel.time_col,
            cols=list(self.feature_names_in_),
            scope_cols=scope_columns(self.scope, panel.entity_col, panel.time_col),
            method=self.method,
            threshold=self.threshold,
            num_perm=self.num_perm,
            seed=self.seed,
            tokenizer=self.tokenizer,
            ngram=self.ngram,
            verify=self.verify,
            b_bits=self.b_bits,
            n_bits=self.n_bits,
            reverse=self.keep == "last",
            simhash_stats=stats,
        )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        frame = with_row_ids(panel.collect())
        graph = self._graph(frame, panel)
        cluster: np.ndarray | None = None
        if self.linkage == "pairwise":
            dup = graph.point_in_time_duplicates()
        else:
            cluster = graph.components()
            dup = cluster != np.arange(graph.n)
        self.n_duplicates_ = int(dup.sum())

        if self.action == "flag":
            extra = [pl.Series(self.flag_col, dup, dtype=pl.Boolean)]
            if cluster is not None:
                extra.append(pl.Series("cluster_id", cluster, dtype=pl.Int64))
            out = frame.with_columns(extra).drop(ROW)
        elif self.survivorship is not None and cluster is not None:
            out = self.survivorship.merge(
                frame.with_columns(pl.Series("__cluster", cluster)),
                cluster_col="__cluster",
                entity=panel.entity_col,
                time=panel.time_col,
                order_col=ROW,
                rank=graph.rank,
            ).drop(["__cluster", ROW], strict=False)
        else:
            out = frame.filter(pl.Series(~dup)).drop(ROW)
        return PanelFrame(
            out.lazy(), entity=panel.entity_col, time=panel.time_col, validate=False
        )


def dedup(
    df: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> Any:
    """Functional form of :class:`Deduplicator` (``fit_transform`` in one call).

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel; the result has the same container type.
    entity, time : str, optional
        Panel keys for a bare frame (defaults: columns 0 / 1).
    **kwargs
        Any :class:`Deduplicator` parameter.

    Returns
    -------
    PanelFrame | polars.DataFrame | polars.LazyFrame
        The deduplicated (or flagged) panel.
    """
    panel = to_panel(df, entity, time)
    out = Deduplicator(**kwargs).fit_transform(panel)
    return rewrap(df, out.lazy(), panel)
