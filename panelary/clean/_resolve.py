"""Entity resolution: which entity ids name the same real-world thing?

"AAPL US Equity", "Apple Inc" and "APPLE INC." arriving from three vendors
under three ids is the panel version of a duplicate: the same entity's history
split across rows that ``.over(entity)`` treats as strangers. This module
resolves ids to a canonical one with a deterministic, dependency-light
pipeline:

1. **Normalise** compared string fields (:func:`~panelary.clean.normalize_text`).
2. **Block** -- only records that share a blocking key are compared: exact
   keys (a column or any Polars expression, e.g. a name prefix), a
   sorted-neighbourhood window, or MinHash-LSH on character shingles
   (:class:`LSHBlocking`). Blocks are unioned (OR-blocking).
3. **Compare** -- per-field similarity (:func:`~panelary.clean.string_similarity`;
   ``rapidfuzz`` when installed, numpy/pure-Python otherwise; numeric fields
   by relative tolerance), combined as a weighted mean over the fields both
   records actually have.
4. **Cluster** -- matched pairs (score >= ``threshold``) feed the union-find;
   each component's canonical id is chosen by a survivorship rule.

Probabilistic (Fellegi-Sunter/EM) linkage is deliberately out of scope here;
it is the job of the optional ``splink`` backend (``er`` extra, deferred).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from panelary.clean._canonicalize import normalize_text
from panelary.clean._cluster import connected_components
from panelary.clean._common import check_choice, check_unit_interval
from panelary.clean._sketch import (
    lsh_candidate_pairs,
    lsh_params,
    minhash_signatures,
    token_hashes,
    tokenize,
)
from panelary.clean._strsim import METRICS, string_similarity
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

__all__ = ["EntityResolver", "LSHBlocking", "resolve_entities"]

_REC = "__panelary_rec"
FieldSpec = str | tuple[str, float]


@dataclass(frozen=True)
class LSHBlocking:
    """Block records by MinHash-LSH on the character shingles of one field.

    Catches typo'd and re-ordered names that exact blocking keys miss, at a
    cost linear in the number of records.

    Parameters
    ----------
    field : str
        String field to shingle.
    threshold : float, default=0.5
        Approximate Jaccard similarity at which pairs become candidates.
    num_perm : int, default=64
        Signature length.
    ngram : int, default=3
        Character shingle length.
    seed : int, default=0
        Hash seed.
    """

    field: str
    threshold: float = 0.5
    num_perm: int = 64
    ngram: int = 3
    seed: int = 0


BlockKey = str | pl.Expr | LSHBlocking


def _pairs_from_keys(keys: pl.Series) -> pl.DataFrame:
    """Record pairs (i < j) sharing a non-null key value."""
    frame = pl.DataFrame({"k": keys, "i": np.arange(keys.len(), dtype=np.int64)})
    frame = frame.filter(pl.col("k").is_not_null())
    return (
        frame.join(frame, on="k", suffix="_r")
        .filter(pl.col("i") < pl.col("i_r"))
        .select(pl.col("i"), pl.col("i_r").alias("j"))
    )


def _lsh_pairs(records: pl.DataFrame, spec: LSHBlocking) -> pl.DataFrame:
    check_unit_interval("LSHBlocking.threshold", spec.threshold)
    base = records.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias(_REC), pl.col(spec.field)
    )
    tok = tokenize(base, [spec.field], row_col=_REC, tokenizer="char", ngram=spec.ngram)
    th = token_hashes(tok, row_col=_REC, seed=spec.seed)
    rows = th.get_column(_REC).to_numpy().astype(np.int64)
    sig = minhash_signatures(
        rows,
        th.get_column("h").to_numpy(),
        records.height,
        num_perm=spec.num_perm,
        seed=spec.seed,
    )
    valid = np.zeros(records.height, dtype=bool)
    valid[rows] = True
    bands, r = lsh_params(
        spec.threshold,
        spec.num_perm,
        false_positive_weight=0.1,
        false_negative_weight=0.9,
    )
    i, j = lsh_candidate_pairs(sig, bands=bands, rows=r, valid=valid)
    return pl.DataFrame({"i": i, "j": j}, schema={"i": pl.Int64, "j": pl.Int64})


def _candidate_pairs(
    records: pl.DataFrame,
    blocking: Sequence[BlockKey] | None,
    neighbourhood: tuple[str | pl.Expr, int] | None,
    max_pairs: int,
    *,
    split: int | None = None,
) -> pl.DataFrame:
    """Candidate pairs ``(i, j)``, ``i < j``, from the union of the blockers.

    With ``split``, only *bipartite* pairs ``i < split <= j`` are kept (the
    reference-vs-new comparison of :meth:`EntityResolver.transform`), and the
    ``max_pairs`` guard counts only those.
    """
    n = records.height
    pieces: list[pl.DataFrame] = []
    if not blocking and neighbourhood is None:
        total = n * (n - 1) // 2 if split is None else split * (n - split)
        if total > max_pairs:
            raise ValueError(
                f"{n} records give {total:,} pairs without blocking "
                f"(max_pairs={max_pairs:,}); pass `blocking=` or `neighbourhood=`."
            )
        if split is None:
            idx = pl.DataFrame({"i": np.arange(n, dtype=np.int64)})
            pieces.append(
                idx.join(idx.rename({"i": "j"}), how="cross").filter(
                    pl.col("i") < pl.col("j")
                )
            )
        else:
            left = pl.DataFrame({"i": np.arange(split, dtype=np.int64)})
            right = pl.DataFrame({"j": np.arange(split, n, dtype=np.int64)})
            pieces.append(left.join(right, how="cross"))
    for key in blocking or []:
        if isinstance(key, LSHBlocking):
            pieces.append(_lsh_pairs(records, key))
            continue
        expr = pl.col(key) if isinstance(key, str) else key
        pieces.append(_pairs_from_keys(records.select(expr.alias("k")).to_series()))
    if neighbourhood is not None:
        key, w = neighbourhood
        if w < 2:
            raise ValueError(f"neighbourhood window must be >= 2, got {w}.")
        expr = pl.col(key) if isinstance(key, str) else key
        order = (
            records.select(expr.alias("k"))
            .with_row_index("i")
            .sort("k", nulls_last=True, maintain_order=True)
            .get_column("i")
            .to_numpy()
            .astype(np.int64)
        )
        a, b = [], []
        for d in range(1, w):
            a.append(order[:-d])
            b.append(order[d:])
        if a:
            ai, bj = np.concatenate(a), np.concatenate(b)
            pieces.append(
                pl.DataFrame(
                    {"i": np.minimum(ai, bj), "j": np.maximum(ai, bj)},
                    schema={"i": pl.Int64, "j": pl.Int64},
                )
            )
    if not pieces:
        return pl.DataFrame(schema={"i": pl.Int64, "j": pl.Int64})
    allp = pl.concat([p.cast(pl.Int64) for p in pieces]).unique().sort(["i", "j"])
    if split is not None:
        allp = allp.filter((pl.col("i") < split) & (pl.col("j") >= split))
    if allp.height > max_pairs:
        raise ValueError(
            f"blocking produced {allp.height:,} candidate pairs (max_pairs="
            f"{max_pairs:,}); use tighter blocking keys."
        )
    return allp


def _key_columns(
    blocking: Sequence[BlockKey] | None,
    neighbourhood: tuple[str | pl.Expr, int] | None,
) -> list[str]:
    """Columns the blocking / neighbourhood keys read (in first-use order)."""
    keys: list[BlockKey] = list(blocking or [])
    if neighbourhood is not None:
        keys.append(neighbourhood[0])
    cols: list[str] = []
    for key in keys:
        if isinstance(key, LSHBlocking):
            names = [key.field]
        elif isinstance(key, str):
            names = [key]
        else:
            names = key.meta.root_names()
        cols.extend(c for c in names if c not in cols)
    return cols


def _normalise_fields(fields: Mapping[str, Any]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for col, spec in fields.items():
        metric, weight = (
            (spec, 1.0) if isinstance(spec, str) else (spec[0], float(spec[1]))
        )
        check_choice(f"fields[{col!r}]", metric, (*METRICS, "numeric"))
        if weight <= 0:
            raise ValueError(f"field weight for {col!r} must be > 0, got {weight}.")
        out[col] = (metric, weight)
    return out


def _score(
    records: pl.DataFrame,
    pairs: pl.DataFrame,
    fields: dict[str, tuple[str, float]],
    *,
    numeric_tolerance: float,
    backend: str,
    right: pl.DataFrame | None = None,
) -> np.ndarray:
    """Weighted mean field similarity per pair (fields missing on either side
    are skipped; a pair with no comparable field scores 0)."""
    other = records if right is None else right
    i = pairs.get_column("i").to_numpy()
    j = pairs.get_column("j").to_numpy()
    num = np.zeros(i.size)
    den = np.zeros(i.size)
    for col, (metric, weight) in fields.items():
        a, b = records.get_column(col)[i], other.get_column(col)[j]
        if metric == "numeric":
            x = a.cast(pl.Float64).to_numpy()
            y = b.cast(pl.Float64).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(x - y) / np.maximum(
                    np.maximum(np.abs(x), np.abs(y)), 1e-12
                )
                sim = np.where(
                    np.isnan(rel),
                    np.nan,
                    np.clip(1.0 - rel / numeric_tolerance, 0.0, 1.0),
                )
            sim = np.where((x == y) & ~np.isnan(x), 1.0, sim)
        else:
            sim = string_similarity(a, b, metric=metric, backend=backend)  # type: ignore[arg-type]
        have = ~np.isnan(sim)
        num[have] += weight * sim[have]
        den[have] += weight
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _prepare(
    records: pl.DataFrame, fields: dict[str, tuple[str, float]], normalize: bool
) -> pl.DataFrame:
    if not normalize:
        return records
    exprs = [
        normalize_text(c, strip_accents=True, strip_punctuation=True).alias(c)
        for c, (metric, _w) in fields.items()
        if metric != "numeric" and records.schema[c] == pl.String
    ]
    return records.with_columns(exprs) if exprs else records


def resolve_entities(
    records: pl.DataFrame | pl.LazyFrame,
    *,
    id_col: str,
    fields: Mapping[str, str | tuple[str, float]],
    blocking: Sequence[BlockKey] | None = None,
    neighbourhood: tuple[str | pl.Expr, int] | None = None,
    threshold: float = 0.85,
    normalize: bool = True,
    numeric_tolerance: float = 0.05,
    canonical: Literal["first", "most_frequent", "min"] = "first",
    order_by: str | None = None,
    backend: Literal["auto", "numpy", "rapidfuzz"] = "auto",
    max_pairs: int = 20_000_000,
) -> pl.DataFrame:
    """Map every entity id to a resolved (canonical) id.

    Parameters
    ----------
    records : polars.DataFrame | polars.LazyFrame
        Entity records: an id column plus descriptive fields. An id may have
        several records (e.g. a name that changed); any matching record links it.
    id_col : str
        Entity id column.
    fields : mapping of str -> metric or (metric, weight)
        Fields to compare and how: ``"levenshtein"``, ``"jaro"``,
        ``"jaro_winkler"``, ``"token_jaccard"``, ``"exact"``, or ``"numeric"``
        (relative difference scaled by ``numeric_tolerance``).
    blocking : sequence of str | polars.Expr | LSHBlocking, optional
        Blocking keys (OR-ed). With neither ``blocking`` nor ``neighbourhood``
        every pair is compared (guarded by ``max_pairs``).
    neighbourhood : (key, window), optional
        Sorted-neighbourhood blocking: compare each record with the next
        ``window - 1`` records in ``key`` order.
    threshold : float, default=0.85
        Minimum weighted similarity for a match.
    normalize : bool, default=True
        Normalise string fields (case, accents, punctuation, whitespace)
        before blocking and comparison.
    numeric_tolerance : float, default=0.05
        Relative difference at which a numeric field's similarity reaches 0.
    canonical : {"first", "most_frequent", "min"}, default="first"
        Which id represents a cluster: the first seen (by ``order_by``, else
        input order), the one with the most records, or the smallest.
    order_by : str, optional
        Column ordering records for ``canonical="first"`` (e.g. the time).
    backend : {"auto", "numpy", "rapidfuzz"}, default="auto"
        String-similarity backend (see :func:`~panelary.clean.string_similarity`).
    max_pairs : int, default=20_000_000
        Safety valve on candidate pairs.

    Returns
    -------
    polars.DataFrame
        One row per distinct id: ``id_col``, ``resolved_id`` and
        ``cluster_size`` (number of ids resolved together), in first-seen order.
    """
    check_unit_interval("threshold", threshold)
    check_choice("canonical", canonical, ("first", "most_frequent", "min"))
    spec = _normalise_fields(fields)
    frame = records.collect() if isinstance(records, pl.LazyFrame) else records
    needed = [id_col, *spec, *_key_columns(blocking, neighbourhood)]
    missing = [c for c in dict.fromkeys(needed) if c not in frame.columns]
    if missing:
        raise ValueError(f"resolve_entities: column(s) {missing} not found.")
    if order_by is not None:
        frame = frame.sort(order_by, maintain_order=True, nulls_last=True)
    frame = _prepare(frame, spec, normalize)
    ids = frame.get_column(id_col)
    uniq = ids.unique(maintain_order=True)
    code = {v: k for k, v in enumerate(uniq.to_list())}
    rec_id = np.array([code[v] for v in ids.to_list()], dtype=np.int64)

    pairs = _candidate_pairs(frame, blocking, neighbourhood, max_pairs)
    if pairs.height:
        pi = pairs.get_column("i").to_numpy()
        pj = pairs.get_column("j").to_numpy()
        diff = rec_id[pi] != rec_id[pj]
        pairs = pairs.filter(pl.Series(diff))
    ei = np.zeros(0, dtype=np.int64)
    ej = np.zeros(0, dtype=np.int64)
    if pairs.height:
        score = _score(
            frame, pairs, spec, numeric_tolerance=numeric_tolerance, backend=backend
        )
        hit = score >= threshold - 1e-12
        ei = rec_id[pairs.get_column("i").to_numpy()[hit]]
        ej = rec_id[pairs.get_column("j").to_numpy()[hit]]
    comp = connected_components(uniq.len(), ei, ej)
    if canonical == "first":
        rep = comp
    else:
        counts = np.bincount(rec_id, minlength=uniq.len())
        table = pl.DataFrame(
            {"node": np.arange(uniq.len()), "comp": comp, "n": counts, "id": uniq}
        )
        if canonical == "most_frequent":
            ranked = table.sort(["comp", "n", "node"], descending=[False, True, False])
        else:
            ranked = table.sort(["comp", "id", "node"])
        best = ranked.group_by("comp", maintain_order=True).agg(pl.col("node").first())
        lookup = dict(zip(best["comp"].to_list(), best["node"].to_list(), strict=True))
        rep = np.array([lookup[c] for c in comp.tolist()], dtype=np.int64)
    sizes = np.bincount(comp, minlength=uniq.len())
    return pl.DataFrame(
        {
            id_col: uniq,
            "resolved_id": uniq.gather(rep),
            "cluster_size": sizes[comp].astype(np.int64),
        }
    )


class EntityResolver(PanelTransformer):
    """Re-key a panel's entity column onto resolved (canonical) entity ids.

    At :meth:`fit` the distinct ``(entity, fields...)`` records of the training
    panel are resolved with :func:`resolve_entities`; the id -> resolved-id map
    and the training records are stored. :meth:`transform` rewrites known ids
    through the map, and matches an id **unseen at fit** against the stored
    training records only (same blocking and threshold): if its best match
    scores at least ``threshold`` it adopts that record's resolved id,
    otherwise it keeps its own. Unseen ids are never merged with each other --
    that would be learning at transform time.

    Parameters
    ----------
    fields, blocking, neighbourhood, threshold, normalize, numeric_tolerance, canonical, backend, max_pairs
        See :func:`resolve_entities`.
    output : str, optional
        Write the resolved id to this new column instead of overwriting the
        entity column.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        ``False`` -- merging ids deliberately combines entities' rows.
    leakage_safe : bool
        ``True`` -- the map and reference records come from the fit panel only.
    mapping_ : polars.DataFrame
        ``entity -> resolved_id`` learned at fit.
    records_ : polars.DataFrame
        The (normalised) training records used to match unseen ids.

    Notes
    -----
    The records compared are the *distinct* ``(entity, fields...)`` tuples of
    the panel, in time order, so ``canonical="first"`` picks the id seen
    earliest and ``canonical="most_frequent"`` the id with the most distinct
    records (not the most rows).

    Re-keying can create repeated ``(entity, time)`` rows where two resolved
    ids overlapped in time; follow it with
    :class:`~panelary.clean.Deduplicator` (``scope="key"``) or a
    :class:`~panelary.clean.Survivorship` merge.
    """

    panel_safe = False
    leakage_safe = True

    def __init__(
        self,
        *,
        fields: Mapping[str, str | tuple[str, float]],
        blocking: Sequence[BlockKey] | None = None,
        neighbourhood: tuple[str | pl.Expr, int] | None = None,
        threshold: float = 0.85,
        normalize: bool = True,
        numeric_tolerance: float = 0.05,
        canonical: Literal["first", "most_frequent", "min"] = "first",
        backend: Literal["auto", "numpy", "rapidfuzz"] = "auto",
        max_pairs: int = 20_000_000,
        output: str | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        check_unit_interval("threshold", threshold)
        check_choice("canonical", canonical, ("first", "most_frequent", "min"))
        check_choice("backend", backend, ("auto", "numpy", "rapidfuzz"))
        self.fields = _normalise_fields(fields)
        self.blocking = list(blocking) if blocking else None
        self.neighbourhood = neighbourhood
        self.threshold = threshold
        self.normalize = normalize
        self.numeric_tolerance = numeric_tolerance
        self.canonical = canonical
        self.backend = backend
        self.max_pairs = max_pairs
        self.output = output
        self.mapping_: pl.DataFrame | None = None
        self.records_: pl.DataFrame | None = None

    def _records(self, panel: PanelFrame) -> pl.DataFrame:
        # Blocking keys may read columns that are not compared (e.g. a
        # country or exchange code); they have to travel with the records.
        keyed = _key_columns(self.blocking, self.neighbourhood)
        cols = list(dict.fromkeys([panel.entity_col, *self.fields, *keyed]))
        missing = [c for c in cols if c not in panel]
        if missing:
            raise ValueError(f"EntityResolver: column(s) {missing} not found.")
        return (
            panel.lazy()
            .sort(panel.time_col, maintain_order=True)
            .select(cols)
            .unique(maintain_order=True)
            .collect()
        )

    def _fit(self, panel: PanelFrame) -> None:
        recs = self._records(panel)
        self.mapping_ = resolve_entities(
            recs,
            id_col=panel.entity_col,
            fields=self.fields,
            blocking=self.blocking,
            neighbourhood=self.neighbourhood,
            threshold=self.threshold,
            normalize=self.normalize,
            numeric_tolerance=self.numeric_tolerance,
            canonical=self.canonical,
            backend=self.backend,
            max_pairs=self.max_pairs,
        )
        prepared = _prepare(recs, self.fields, self.normalize)
        self.records_ = prepared.join(
            self.mapping_.select(panel.entity_col, "resolved_id"),
            on=panel.entity_col,
            how="left",
        )

    def _match_unseen(self, unseen: pl.DataFrame, entity: str) -> pl.DataFrame:
        """Best training match (>= threshold) for each unseen id's records."""
        assert self.records_ is not None
        ref = self.records_
        new = _prepare(unseen, self.fields, self.normalize)
        both = pl.concat([ref.select(new.columns), new], how="vertical_relaxed")
        # Only reference x new pairs are formed: nothing is learned among new ids.
        pairs = _candidate_pairs(
            both, self.blocking, self.neighbourhood, self.max_pairs, split=ref.height
        )
        empty = pl.DataFrame(
            schema={
                entity: new.schema[entity],
                "resolved_id": ref.schema["resolved_id"],
            }
        )
        if pairs.height == 0:
            return empty
        score = _score(
            both,
            pairs,
            self.fields,
            numeric_tolerance=self.numeric_tolerance,
            backend=self.backend,
        )
        scored = pairs.with_columns(pl.Series("score", score)).filter(
            pl.col("score") >= self.threshold - 1e-12
        )
        if scored.height == 0:
            return empty
        return (
            scored.with_columns(
                both.get_column(entity).gather(scored.get_column("j")).alias(entity),
                ref.get_column("resolved_id")
                .gather(scored.get_column("i"))
                .alias("resolved_id"),
            )
            .sort(["score", "i"], descending=[True, False])
            .group_by(entity, maintain_order=True)
            .agg(pl.col("resolved_id").first())
        )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        assert self.mapping_ is not None
        ent = panel.entity_col
        mapping = self.mapping_.select(ent, "resolved_id")
        recs = self._records(panel)
        unseen = recs.join(mapping, on=ent, how="anti")
        if unseen.height:
            mapping = pl.concat(
                [mapping, self._match_unseen(unseen, ent)], how="vertical_relaxed"
            )
        target = self.output or ent
        resolved = "__panelary_resolved"
        lf = (
            panel.lazy()
            .with_row_index("__panelary_pos")
            .join(mapping.rename({"resolved_id": resolved}).lazy(), on=ent, how="left")
            .sort("__panelary_pos")
            .with_columns(pl.coalesce(resolved, pl.col(ent)).alias(target))
            .drop("__panelary_pos", resolved)
        )
        return PanelFrame(lf, entity=ent, time=panel.time_col, validate=False)
