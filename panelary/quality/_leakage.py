"""Leak-safety invariants: two questions no schema library asks.

1. **Does a fitted transform's stored state derive only from train rows?**
   :func:`check_fitted_state` answers it two independent ways. *Provenance*: a
   :class:`~panelary.core.protocol.PanelTransformer` records the panel it was
   fitted on (``_fit_panel``), so any fitted ``(entity, time)`` key outside the
   train fold is a direct witness. *Counterfactual refit*: a deep copy is
   refitted on the train rows alone and its learned state (sklearn-convention
   trailing-underscore attributes) must equal the original's. The second works
   for any object with a ``fit`` and needs a deterministic fit -- which the
   library's seeded-RNG contract already demands.

2. **Do any near-duplicate rows straddle the train/test boundary?**
   :func:`check_near_duplicate_straddle` clusters rows with
   :func:`panelary.clean.near_duplicate_clusters` (panel-global: every entity,
   every time) and fails if one cluster has rows on both sides of a split. A
   row that sits in train *and* test is the degenerate case and is caught too.

Both build on the library's existing primitives rather than inventing parallel
ones: splits are those of :mod:`panelary.validation` (an ``IndexSplit`` or a
``(train, test)`` pair, exactly as :func:`panelary.testing.assert_no_train_test_leak`
reads them), and a ``warn``-level finding is raised as
:class:`panelary.preprocessing.LeakageWarning`.
"""

from __future__ import annotations

import contextlib
import copy
import math
import numbers
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary.quality._common import (
    EVIDENCE_COUNTERFACTUAL,
    EVIDENCE_STATIC,
    CheckResult,
    Impact,
    check_impact,
    jsonable,
    produced_by,
    resolve_keys,
    to_dataframe,
)

__all__ = ["check_fitted_state", "check_near_duplicate_straddle"]


# --------------------------------------------------------------------------- #
# Split handling
# --------------------------------------------------------------------------- #
def _is_scalar(x: Any) -> bool:
    return not isinstance(
        x,
        (
            list,
            tuple,
            set,
            frozenset,
            np.ndarray,
            pl.Series,
            pl.DataFrame,
            pl.LazyFrame,
        ),
    ) and not hasattr(x, "collect")


def _is_index_split(x: Any) -> bool:
    return (
        hasattr(x, "train") and hasattr(x, "test") and not isinstance(x, (tuple, list))
    )


def _is_pair(x: Any) -> bool:
    return isinstance(x, (tuple, list)) and len(x) == 2 and not _is_scalar(x[0])


def normalise_splits(split: Any) -> list[Any] | str:
    """Return a list of single splits, or a fold-label column name.

    Accepted forms: a fold-label column name (``str``); one
    :class:`~panelary.validation.IndexSplit` (positions into the sorted unique
    time index); one ``(train, test)`` pair whose members are frames /
    PanelFrames (matched on ``(entity, time)``) or array-likes of time values;
    or a sequence of any of those (one per fold).
    """
    if isinstance(split, str):
        return split
    if _is_index_split(split):
        return [split]
    if (
        isinstance(split, (tuple, list))
        and split
        and all(_is_index_split(s) or _is_pair(s) for s in split)
    ):
        # A (train, test) pair of frames also satisfies `_is_pair` element-wise
        # only if each member is itself a 2-sequence of non-scalars; frames are
        # not sequences, so a pair of frames never lands here.
        return list(split)
    if _is_pair(split):
        return [split]
    raise ValueError(
        "`split` must be a fold-label column name, an IndexSplit, a (train, test) "
        "pair (frames, PanelFrames or array-likes of time values), or a sequence "
        f"of those; got {type(split).__name__!r}."
    )


def _membership(
    df: pl.DataFrame, member: Any, entity: str, time: str, times: pl.Series
) -> pl.Series:
    """Per-row flag: is the row in ``member`` (one side of a split)?"""
    from panelary.core.panel_frame import PanelFrame

    if isinstance(member, (PanelFrame, pl.DataFrame, pl.LazyFrame)):
        side = to_dataframe(member)
        on = [c for c in (entity, time) if c in side.columns]
        if time not in on:
            raise ValueError(
                f"a split member frame must carry the time column {time!r}; it "
                f"has {side.columns}."
            )
        flagged = side.select(on).unique().with_columns(pl.lit(True).alias("__in__"))
        joined = (
            df.select(on)
            .with_row_index("__row__")
            .join(flagged, on=on, how="left")
            .sort("__row__")
        )
        return joined.get_column("__in__").fill_null(False)
    if isinstance(member, pl.Series):
        values = member
    else:
        values = pl.Series(values=list(np.asarray(member).tolist()))
    with contextlib.suppress(Exception):  # else keep the caller's dtype
        values = values.cast(df.schema[time])
    return df.select(pl.col(time).is_in(values.implode()).fill_null(False)).to_series()


def split_masks(
    df: pl.DataFrame, split: Any, entity: str, time: str
) -> tuple[pl.Series, pl.Series]:
    """``(in_train, in_test)`` row flags for one split."""
    times = df.get_column(time).drop_nulls().unique().sort()
    if _is_index_split(split):
        train_pos = np.asarray(split.train, dtype=np.int64)
        test_pos = np.asarray(split.test, dtype=np.int64)
        if (
            train_pos.size
            and train_pos.max() >= times.len()
            or (test_pos.size and test_pos.max() >= times.len())
        ):
            raise ValueError(
                f"IndexSplit positions exceed the panel's {times.len()} unique "
                "time steps; was it built for a different panel?"
            )
        return (
            _membership(df, times.gather(train_pos), entity, time, times),
            _membership(df, times.gather(test_pos), entity, time, times),
        )
    train, test = split
    return (
        _membership(df, train, entity, time, times),
        _membership(df, test, entity, time, times),
    )


# --------------------------------------------------------------------------- #
# Near-duplicate straddle
# --------------------------------------------------------------------------- #
def _cluster_ids(
    df: pl.DataFrame,
    *,
    entity: str,
    time: str,
    clusters: str | pl.Series | None,
    exclude: Sequence[str],
    options: Mapping[str, Any],
) -> pl.Series:
    """One int64 cluster id per row, from the caller or from ``panelary.clean``."""
    if isinstance(clusters, str):
        return df.get_column(clusters).cast(pl.Int64)
    if isinstance(clusters, pl.Series):
        if clusters.len() != df.height:
            raise ValueError(
                f"`clusters` has {clusters.len()} values for {df.height} rows."
            )
        return clusters.cast(pl.Int64)
    from panelary.clean import near_duplicate_clusters

    kwargs = dict(options)
    if kwargs.get("columns") is None:
        kwargs["columns"] = [c for c in df.columns if c not in (entity, time, *exclude)]
    out = near_duplicate_clusters(df, entity=entity, time=time, **kwargs)
    series = out if isinstance(out, pl.Series) else out.get_column("cluster_id")
    if series.len() != df.height:
        raise RuntimeError(
            "panelary.clean.near_duplicate_clusters returned "
            f"{series.len()} cluster ids for {df.height} rows; expected one per row."
        )
    if isinstance(out, pl.DataFrame):
        for key in (entity, time):
            if key in out.columns and not out.get_column(key).equals(
                df.get_column(key), null_equal=True
            ):
                raise RuntimeError(
                    "panelary.clean.near_duplicate_clusters did not preserve the "
                    f"input row order (column {key!r} differs); cannot align ids."
                )
    return series.cast(pl.Int64)


def _keys_list(frame: pl.DataFrame, entity: str, time: str, limit: int) -> list[Any]:
    sub = frame.select(entity, time).sort(entity, time, nulls_last=True).head(limit)
    return [{entity: jsonable(r[0]), time: jsonable(r[1])} for r in sub.iter_rows()]


def _straddles_for(
    base: pl.DataFrame,
    in_a: pl.Series,
    in_b: pl.Series,
    *,
    entity: str,
    time: str,
    fold: int | None,
    limit: int,
) -> tuple[pl.Series, list[dict[str, Any]], int]:
    """Clusters with rows on both sides; returns (row mask, sample, n_clusters)."""
    frame = base.with_columns(in_a.alias("__a__"), in_b.alias("__b__"))
    per = frame.group_by("__cluster__").agg(
        pl.col("__a__").any().alias("a"),
        pl.col("__b__").any().alias("b"),
        pl.len().cast(pl.Int64).alias("size"),
    )
    bad = per.filter(pl.col("a") & pl.col("b")).get_column("__cluster__")
    mask = frame.get_column("__cluster__").is_in(bad.implode())
    sample: list[dict[str, Any]] = []
    if bad.len():
        hit = frame.filter(mask)
        # Order clusters by their smallest key so the sample does not depend on
        # how the clustering numbered them.
        order = (
            hit.group_by("__cluster__")
            .agg(pl.struct(entity, time).sort().first().alias("__first__"))
            .sort("__first__", nulls_last=True)
            .head(limit)
            .get_column("__cluster__")
            .to_list()
        )
        for cid in order:
            members = hit.filter(pl.col("__cluster__") == cid)
            record: dict[str, Any] = {
                "keys": _keys_list(members, entity, time, 1)[0],
                "size": members.height,
                "train_keys": _keys_list(
                    members.filter(pl.col("__a__")), entity, time, 3
                ),
                "test_keys": _keys_list(
                    members.filter(pl.col("__b__")), entity, time, 3
                ),
            }
            if fold is not None:
                record["fold"] = fold
            sample.append(record)
    return mask, sample, int(bad.len())


def straddle_check(
    df: pl.DataFrame,
    split: Any,
    *,
    entity: str,
    time: str,
    clusters: str | pl.Series | None = None,
    options: Mapping[str, Any] | None = None,
    impact: Impact = "fail",
    limit: int = 5,
    producer: str = "",
) -> CheckResult:
    """Core of :func:`check_near_duplicate_straddle` on resolved inputs."""
    splits = normalise_splits(split)
    exclude: list[str] = [splits] if isinstance(splits, str) else []
    if isinstance(clusters, str):
        exclude.append(clusters)
    cluster_ids = _cluster_ids(
        df,
        entity=entity,
        time=time,
        clusters=clusters,
        exclude=exclude,
        options=options or {},
    )
    base = df.select(entity, time).with_columns(cluster_ids.alias("__cluster__"))
    mask = pl.Series("m", [False] * df.height, dtype=pl.Boolean)
    sample: list[dict[str, Any]] = []
    n_clusters = 0
    folds_hit: list[int] = []
    if isinstance(splits, str):
        labels = df.get_column(splits)
        per = (
            base.with_columns(labels.alias("__label__"))
            .filter(pl.col("__label__").is_not_null())
            .group_by("__cluster__")
            .agg(pl.col("__label__").n_unique().alias("n"))
        )
        bad = per.filter(pl.col("n") > 1).get_column("__cluster__")
        mask = base.get_column("__cluster__").is_in(bad.implode())
        n_clusters = int(bad.len())
        if n_clusters:
            hit = base.with_columns(labels.alias("__label__")).filter(mask)
            order = (
                hit.group_by("__cluster__")
                .agg(pl.struct(entity, time).sort().first().alias("__first__"))
                .sort("__first__", nulls_last=True)
                .head(limit)
                .get_column("__cluster__")
                .to_list()
            )
            for cid in order:
                members = hit.filter(pl.col("__cluster__") == cid)
                sample.append(
                    {
                        "keys": _keys_list(members, entity, time, 1)[0],
                        "size": members.height,
                        "folds": jsonable(
                            sorted(
                                members.get_column("__label__")
                                .drop_nulls()
                                .unique()
                                .to_list(),
                                key=str,
                            )
                        ),
                        "member_keys": _keys_list(members, entity, time, 3),
                    }
                )
        side_desc = f"folds of {splits!r}"
    else:
        multi = len(splits) > 1
        seen: set[int] = set()
        for i, one in enumerate(splits):
            in_train, in_test = split_masks(df, one, entity, time)
            fmask, fsample, _ = _straddles_for(
                base,
                in_train,
                in_test,
                entity=entity,
                time=time,
                fold=i if multi else None,
                limit=limit,
            )
            if fmask.any():
                folds_hit.append(i)
                mask = mask | fmask
                seen.update(
                    base.filter(fmask).get_column("__cluster__").unique().to_list()
                )
                sample.extend(fsample)
        n_clusters = len(seen)
        sample = sample[:limit]
        side_desc = (
            "train and test" if not multi else f"train and test of {len(splits)} folds"
        )
    mask = mask.alias("m")
    n_rows = int(mask.sum())
    observed: dict[str, Any] = {"straddling_clusters": n_clusters, "rows": n_rows}
    if not isinstance(splits, str) and len(splits) > 1:
        observed["folds"] = folds_hit
    return CheckResult(
        name="near_duplicate_straddle",
        category="leakage",
        impact=impact,
        passed=n_clusters == 0,
        message=(
            f"no near-duplicate cluster spans {side_desc}."
            if n_clusters == 0
            else f"{n_clusters} near-duplicate cluster(s) ({n_rows} rows) span "
            f"{side_desc}: the model is scored on rows it has effectively seen. "
            "Collapse near-duplicates panel-wide (panelary.clean) before "
            "assigning folds."
        ),
        n_failing=n_clusters,
        unit="clusters",
        observed=observed,
        expected={"straddling_clusters": 0},
        offending=tuple(sample),
        evidence_kind=EVIDENCE_STATIC,
        produced_by=producer,
        row_mask=mask,
    )


def check_near_duplicate_straddle(
    data: Any,
    split: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    clusters: str | pl.Series | None = None,
    columns: Sequence[str] | None = None,
    method: str = "minhash",
    threshold: float = 0.8,
    num_perm: int = 128,
    seed: int = 0,
    impact: Impact = "fail",
    max_examples: int = 5,
) -> CheckResult:
    """Fail if any near-duplicate cluster has rows on both sides of a split.

    Near-duplicate rows on opposite sides of a train/test boundary leak the
    target just as surely as a shared row: the model is scored on data it has
    effectively trained on. Clusters are computed **panel-globally** (across
    every entity and every time) by
    :func:`panelary.clean.near_duplicate_clusters`, then checked against the
    split -- the invariant ``panelary.clean`` is built to establish.

    Parameters
    ----------
    data : polars.DataFrame | polars.LazyFrame | PanelFrame
        The full panel the split is taken from.
    split : str | IndexSplit | (train, test) | sequence of those
        A fold-label column name (a cluster straddles if its rows carry more
        than one non-null label); a :class:`~panelary.validation.IndexSplit`
        (positions into the sorted unique-time index, as the
        :mod:`panelary.validation` splitters emit); a ``(train, test)`` pair of
        frames / PanelFrames (matched on ``(entity, time)``) or of array-likes
        of time values (as :func:`panelary.testing.assert_no_train_test_leak`
        reads them); or a sequence of splits, one per fold. Rows in neither
        side (purged, embargoed) are ignored.
    entity, time : str, optional
        Panel keys for a bare frame (default: columns 0 and 1).
    clusters : str | polars.Series, optional
        Precomputed cluster ids (a column name or one value per row). When
        given, ``panelary.clean`` is not called and the clustering options
        below are ignored.
    columns, method, threshold, num_perm, seed
        Forwarded to :func:`panelary.clean.near_duplicate_clusters`.
        ``columns=None`` means every column except the keys and a fold-label
        column.
    impact : {"fail", "warn"}, default="fail"
        Recorded on the result.
    max_examples : int, default=5
        Size of the offending-cluster sample.

    Returns
    -------
    CheckResult
        ``name="near_duplicate_straddle"``, one offending record per straddling
        cluster: its smallest key, its size and up to three train / test keys
        (no cluster id: ids depend on row order, the locator must not).
    """
    check_impact(impact, name="near_duplicate_straddle")
    df, entity_col, time_col = resolve_keys(data, entity, time)
    return straddle_check(
        df,
        split,
        entity=entity_col,
        time=time_col,
        clusters=clusters,
        options={
            "columns": None if columns is None else list(columns),
            "method": method,
            "threshold": threshold,
            "num_perm": num_perm,
            "seed": seed,
        },
        impact=impact,
        limit=max_examples,
        producer=produced_by("check_near_duplicate_straddle"),
    )


# --------------------------------------------------------------------------- #
# Fitted-state provenance
# --------------------------------------------------------------------------- #
#: Bookkeeping attributes a refit legitimately changes; never compared.
_BOOKKEEPING = frozenset({"_fit_panel", "_fitted", "_warned"})


def learned_state(obj: Any, attributes: Sequence[str] | None = None) -> dict[str, Any]:
    """The learned state of a fitted object.

    ``attributes`` names it explicitly; by default it is every public instance
    attribute whose name ends in ``_`` (``mean_``, ``components_``) -- the
    sklearn convention :class:`~panelary.core.protocol.PanelTransformer`
    documents for ``_fit``.
    """
    if attributes is not None:
        return {name: getattr(obj, name) for name in attributes}
    try:
        items = vars(obj).items()
    except TypeError:
        return {}
    return {k: v for k, v in items if k.endswith("_") and not k.startswith("_")}


def _numeric_close(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _series_close(a: pl.Series, b: pl.Series, rtol: float, atol: float) -> bool:
    if a.len() != b.len() or a.dtype != b.dtype:
        return False
    if not a.is_null().equals(b.is_null()):
        return False
    if a.dtype.is_float():
        x = a.fill_null(0.0).to_numpy()
        y = b.fill_null(0.0).to_numpy()
        return bool(np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
    return bool(a.equals(b, null_equal=True))


def _canonical_rows(frame: pl.DataFrame) -> pl.DataFrame | None:
    """``frame`` sorted by its exact columns, then its float columns; None if
    some column cannot be sorted."""
    exact = [c for c in frame.columns if not frame.schema[c].is_float()]
    floats = [c for c in frame.columns if frame.schema[c].is_float()]
    try:
        return frame.sort([*exact, *floats], nulls_last=True, maintain_order=True)
    except Exception:
        return None


def _frame_difference(
    a: pl.DataFrame, b: pl.DataFrame, path: str, *, rtol: float, atol: float
) -> str | None:
    """Compare two state frames as **sets of rows**.

    Learned-state tables usually come out of a ``group_by``, whose row order
    polars does not guarantee, so two fits on identical rows can list the same
    statistics in a different order. Row order is therefore ignored: the frames
    are compared as laid out, and if that fails, again after a canonical sort.
    """

    def diff(x: pl.DataFrame, y: pl.DataFrame) -> str | None:
        for col in x.columns:
            if not _series_close(x.get_column(col), y.get_column(col), rtol, atol):
                return f"{path}[{col!r}]"
        return None

    hit = diff(a, b)
    if hit is None:
        return None
    ca, cb = _canonical_rows(a), _canonical_rows(b)
    if ca is None or cb is None:
        return hit
    return diff(ca, cb)


def first_difference(
    a: Any, b: Any, path: str, *, rtol: float, atol: float, depth: int = 0
) -> str | None:
    """Path of the first place two learned states differ, or None if equal."""
    from panelary.core.panel_frame import PanelFrame

    if a is b:
        return None
    if isinstance(a, (pl.LazyFrame, PanelFrame)):
        a = a.collect()
    if isinstance(b, (pl.LazyFrame, PanelFrame)):
        b = b.collect()
    if isinstance(a, pl.DataFrame) or isinstance(b, pl.DataFrame):
        if not (isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame)):
            return path
        if a.columns != b.columns or a.height != b.height:
            return path
        return _frame_difference(a, b, path, rtol=rtol, atol=atol)
    if isinstance(a, pl.Series) or isinstance(b, pl.Series):
        if not (isinstance(a, pl.Series) and isinstance(b, pl.Series)):
            return path
        return None if _series_close(a, b, rtol, atol) else path
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape:
            return path
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            same = np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = np.array_equal(x, y)
        return None if same else path
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return path
        for key in sorted(a, key=repr):
            hit = first_difference(
                a[key],
                b[key],
                f"{path}[{key!r}]",
                rtol=rtol,
                atol=atol,
                depth=depth + 1,
            )
            if hit is not None:
                return hit
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            hit = first_difference(
                x, y, f"{path}[{i}]", rtol=rtol, atol=atol, depth=depth + 1
            )
            if hit is not None:
                return hit
        return None
    if (
        isinstance(a, numbers.Number)
        and isinstance(b, numbers.Number)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
            return None if _numeric_close(float(a), float(b), rtol, atol) else path
        return None if a == b else path
    if (
        type(a) is type(b)
        and hasattr(a, "__dict__")
        and not isinstance(a, type)
        and depth < 6
    ):
        va = {k: v for k, v in vars(a).items() if k not in _BOOKKEEPING}
        vb = {k: v for k, v in vars(b).items() if k not in _BOOKKEEPING}
        return first_difference(va, vb, path, rtol=rtol, atol=atol, depth=depth + 1)
    try:
        return None if bool(a == b) else path
    except Exception:
        return None if repr(a) == repr(b) else path


def _default_refit(obj: Any, train: pl.DataFrame, entity: str, time: str) -> Any:
    from panelary.core.panel_frame import PanelFrame
    from panelary.core.protocol import PanelTransformer

    if isinstance(obj, PanelTransformer):
        return obj.fit(PanelFrame(train, entity=entity, time=time, validate=False))
    fit = getattr(obj, "fit", None)
    if callable(fit):
        out = fit(train)
        return obj if out is None else out
    raise TypeError(
        f"cannot refit a {type(obj).__name__!r}: it has no `fit` method. Pass "
        "`refit=lambda obj, train: ...` to say how it learns its state."
    )


def fitted_state_check(
    fitted: Any,
    train: pl.DataFrame,
    *,
    entity: str,
    time: str,
    name: str,
    attributes: Sequence[str] | None = None,
    refit: Callable[[Any, pl.DataFrame], Any] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    impact: Impact = "fail",
    limit: int = 5,
    producer: str = "",
) -> CheckResult:
    """Core of :func:`check_fitted_state` on resolved inputs."""
    observed: dict[str, Any] = {}
    offending: list[dict[str, Any]] = []
    problems: list[str] = []
    fit_entity, fit_time = entity, time

    # 1. Provenance: the panel the transform says it was fitted on.
    fit_panel = getattr(fitted, "_fit_panel", None)
    if fit_panel is not None and hasattr(fit_panel, "entity_col"):
        fit_entity, fit_time = fit_panel.entity_col, fit_panel.time_col
        missing = [k for k in (fit_entity, fit_time) if k not in train.columns]
        if missing:
            raise ValueError(
                f"{name}: the transform was fitted on keys ({fit_entity!r}, "
                f"{fit_time!r}) but the train rows lack {missing}."
            )
        fit_keys = fit_panel.lazy().select(fit_entity, fit_time).unique().collect()
        train_keys = train.select(fit_entity, fit_time).unique()
        outside = fit_keys.join(train_keys, on=[fit_entity, fit_time], how="anti").sort(
            fit_entity, fit_time, nulls_last=True
        )
        observed["fit_rows"] = fit_keys.height
        observed["fit_rows_outside_train"] = outside.height
        if outside.height:
            problems.append(
                f"it was fitted on {outside.height} (entity, time) key(s) outside "
                "the train rows"
            )
            offending.extend(
                {
                    "keys": {k: jsonable(v) for k, v in row.items()},
                    "source": "fit_panel",
                }
                for row in outside.head(limit).iter_rows(named=True)
            )

    # 2. Counterfactual: refit a copy on the train rows alone, compare state.
    state = learned_state(fitted, attributes)
    if not state and fit_panel is None:
        raise ValueError(
            f"{name}: found no learned state to compare (no public attributes "
            "ending in '_') and no recorded fit panel. Pass `attributes=[...]` "
            "naming where the transform stores what it learned."
        )
    if state:
        try:
            clone = copy.deepcopy(fitted)
        except Exception:  # pragma: no cover - uncopyable: fall back to shallow
            clone = copy.copy(fitted)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if refit is not None:
                refitted = refit(clone, train)
                clone = clone if refitted is None else refitted
            else:
                clone = _default_refit(clone, train, fit_entity, fit_time)
        clone_state = learned_state(clone, attributes)
        diff = first_difference(state, clone_state, "state", rtol=rtol, atol=atol)
        observed["compared_attributes"] = sorted(state)
        observed["state_matches_train_refit"] = diff is None
        if diff is not None:
            observed["first_difference"] = diff
            problems.append(
                f"its learned state differs from a refit on the train rows alone "
                f"(first difference at {diff})"
            )
            if not offending:
                offending.append({"keys": {"attribute": diff}, "source": "refit"})

    passed = not problems
    return CheckResult(
        name="fitted_state",
        category="leakage",
        impact=impact,
        passed=passed,
        message=(
            f"{name}: stored state derives only from the train rows."
            if passed
            else f"{name}: stored state does not derive only from the train rows -- "
            + "; ".join(problems)
            + ". Fit it on the train fold only (never the full panel)."
        ),
        n_failing=0 if passed else 1,
        unit="transforms",
        observed=observed,
        expected={"fit_rows_outside_train": 0, "state_matches_train_refit": True},
        offending=tuple(offending[:limit]),
        target=name,
        evidence_kind=EVIDENCE_COUNTERFACTUAL,
        produced_by=producer,
    )


def check_fitted_state(
    fitted: Any,
    train: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    name: str | None = None,
    attributes: Sequence[str] | None = None,
    refit: Callable[[Any, pl.DataFrame], Any] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    impact: Impact = "fail",
    max_examples: int = 5,
) -> CheckResult:
    """Does a fitted transform's stored state derive only from ``train``?

    Two independent tests; the check fails if either does:

    1. **Provenance** (``PanelTransformer`` only). ``fit`` records the panel it
       saw; any of its ``(entity, time)`` keys outside ``train`` is a direct
       witness that the state saw non-train rows. Those keys are the locator.
    2. **Counterfactual refit.** A deep copy is refitted on ``train`` alone
       and its learned state compared with the original's (within
       ``rtol`` / ``atol``; NaN equals NaN). A difference means the original
       learned from something other than the train rows. This works for any
       object with ``fit`` and assumes the fit is deterministic -- the
       library's seeded-RNG contract.

    Parameters
    ----------
    fitted : object
        The fitted transform (a ``PanelTransformer`` or anything with ``fit``).
    train : polars.DataFrame | polars.LazyFrame | PanelFrame
        Exactly the rows the transform was *supposed* to be fitted on.
    entity, time : str, optional
        Panel keys of a bare ``train`` frame (default: columns 0 and 1).
    name : str, optional
        Label for the transform (default: its class name); the check target.
    attributes : sequence of str, optional
        Where the learned state lives (default: public attributes ending in
        ``_``).
    refit : callable, optional
        ``refit(clone, train_df) -> fitted`` for objects whose fit is not
        ``fit(frame)``. Default: ``clone.fit(PanelFrame(train))`` for a
        ``PanelTransformer``, else ``clone.fit(train_df)``.
    rtol, atol : float
        Tolerances for comparing floating state.
    impact : {"fail", "warn"}, default="fail"
    max_examples : int, default=5

    Returns
    -------
    CheckResult
        ``name="fitted_state"``, evidence kind ``"counterfactual-run"``.

    Raises
    ------
    ValueError
        If the object exposes neither learned state nor a fit panel, so
        nothing could be verified.
    """
    check_impact(impact, name="fitted_state")
    df, entity_col, time_col = resolve_keys(train, entity, time)
    return fitted_state_check(
        fitted,
        df,
        entity=entity_col,
        time=time_col,
        name=name or type(fitted).__name__,
        attributes=attributes,
        refit=refit,
        rtol=rtol,
        atol=atol,
        impact=impact,
        limit=max_examples,
        producer=produced_by("check_fitted_state"),
    )
