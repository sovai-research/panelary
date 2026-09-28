"""The materialization boundary: long Polars panel <-> dense ``(entity, time, value)`` tensor.

Every method in the shape algebra assumes a dense rectangular
``X[entity, time, feature]``; panels are long-format, ragged, misaligned and
missing-valued. This module is the one place that crosses that boundary, with an
explicit, declared policy for raggedness, and the one place that crosses back.

:func:`build_tensor` was moved here unchanged from ``panelary.cluster._tensor``
(which now re-exports it). It replaces SovAI's ``pandas_to_array``
(``clustering.py`` L13-70), whose two leaks are deliberately *not* reproduced:

* it back-filled missing values (``bfill`` -- pulls future observations
  backward), and
* it fit a ``StandardScaler`` over the **entire** sample (mean/std computed
  across all dates, including the future).

The causal replacement here:

* pivots the long panel onto a dense ``(entity x time)`` grid (a cross join, so
  every entity has an equal-length series),
* **forward-fills within each entity only** (``forward_fill().over(entity)``),
  never backward, and
* optionally applies a per-entity **expanding (causal) z-normalisation**: the
  value at time ``t`` is standardised using only that entity's observations up to
  and including ``t``. Rows without enough history to estimate a variance are
  emitted as nulls (NaN in the tensor), never fabricated.

The result is a NaN-containing ``(n_entities, n_times, n_values)`` float tensor
plus the sorted entity / time indices and the matching dense key frame, so
callers can align computed features straight back onto ``(entity, time)`` rows.

Hard rules
----------
* The tensor's time axis is **always** ascending and **always** dense on the
  union grid. No transform downstream re-sorts it.
* NaN means "no observation". Every consumer states its NaN policy; none calls
  ``np.nan_to_num`` silently.
* :func:`build_tensor` and :func:`to_long` round-trip: for a complete panel,
  ``to_long(build_tensor(df, cols, z_normalize=False))`` reproduces ``df`` on the
  value columns, up to dtype.

Raggedness
----------
:func:`build_sequences` is the *positional* view -- each entity's own
observations in time order, not the union date grid -- and takes an explicit
:class:`Ragged` policy. ``REFUSE`` is the default because silently padding a
12-observation entity out to 3,000 and running an SVD on it produces a number,
and the number is garbage.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "PanelTensor",
    "Ragged",
    "RaggedSequences",
    "build_sequences",
    "build_tensor",
    "refuse_missing",
    "to_long",
]


class PanelTensor(NamedTuple):
    """A dense panel tensor plus the indices needed to realign it.

    Attributes
    ----------
    tensor : numpy.ndarray
        ``(n_entities, n_times, n_values)`` float array, NaN where a value is
        missing / has insufficient history. Entity-major, time-ordered.
    entities : polars.Series
        Sorted unique entity ids (row order of ``tensor``).
    times : polars.Series
        Sorted unique time values (column order of ``tensor``).
    keys : polars.DataFrame
        The dense ``(entity, time)`` grid, sorted entity-major then by time, so
        ``keys`` rows line up with ``tensor.reshape(n_entities * n_times, ...)``.
    value_cols : list of str
        The value column names, in the tensor's last-axis order.
    """

    tensor: np.ndarray
    entities: pl.Series
    times: pl.Series
    keys: pl.DataFrame
    value_cols: list[str]


def build_tensor(
    panel: PanelFrame,
    value_cols: Sequence[str],
    *,
    forward_fill: bool = True,
    z_normalize: bool = True,
    ddof: int = 1,
    min_history: int = 2,
) -> PanelTensor:
    """Build a causal, dense ``(entity, time, value)`` tensor from ``panel``.

    Parameters
    ----------
    panel : PanelFrame
        The source panel.
    value_cols : sequence of str
        Feature columns whose per-entity series form the tensor's last axis.
    forward_fill : bool, default=True
        Forward-fill missing values within each entity (never backward).
    z_normalize : bool, default=True
        Apply per-entity expanding (causal) z-normalisation.
    ddof : int, default=1
        Delta degrees of freedom for the causal standard deviation.
    min_history : int, default=2
        Minimum number of observations required before a z-normalised value is
        emitted; earlier rows become NaN.

    Returns
    -------
    PanelTensor
    """
    value_cols = list(value_cols)
    if not value_cols:
        raise ValueError("`value_cols` must name at least one feature column.")
    ent, tim = panel.entity_col, panel.time_col
    missing = [c for c in value_cols if c not in panel]
    if missing:
        raise ValueError(
            f"value column(s) {missing} not found in panel. "
            f"Available columns: {panel.columns}."
        )

    base = panel.lazy().select([ent, tim, *value_cols]).collect()
    entities = base.select(pl.col(ent).unique().sort()).to_series()
    times = base.select(pl.col(tim).unique().sort()).to_series()
    n_ent, n_times = entities.len(), times.len()

    # Dense (entity x time) grid via cross join, so every entity is equal-length.
    grid = entities.to_frame().join(times.to_frame(), how="cross")
    merged = (
        grid.join(base, on=[ent, tim], how="left")
        .sort([ent, tim])
        .with_columns([pl.col(v).cast(pl.Float64) for v in value_cols])
    )

    if forward_fill:
        merged = merged.with_columns(
            [pl.col(v).forward_fill().over(ent).alias(v) for v in value_cols]
        )

    if z_normalize:
        znorm_exprs = []
        for v in value_cols:
            x = pl.col(v)
            cnt = x.is_not_null().cum_sum().over(ent)
            s1 = x.fill_null(0.0).cum_sum().over(ent)
            s2 = (x.fill_null(0.0) ** 2).cum_sum().over(ent)
            mean = s1 / cnt
            var = (s2 - cnt * mean**2) / (cnt - ddof)
            std = var.sqrt()
            z = (x - mean) / std
            znorm_exprs.append(
                pl.when((cnt >= min_history) & (std > 0))
                .then(z)
                .otherwise(None)
                .alias(v)
            )
        merged = merged.with_columns(znorm_exprs)

    keys = merged.select([ent, tim])
    tensor = np.empty((n_ent, n_times, len(value_cols)), dtype=float)
    for vi, v in enumerate(value_cols):
        tensor[:, :, vi] = merged.get_column(v).to_numpy().reshape(n_ent, n_times)

    return PanelTensor(
        tensor=tensor,
        entities=entities,
        times=times,
        keys=keys,
        value_cols=value_cols,
    )


def to_long(
    t: PanelTensor,
    *,
    prefix: str | None = None,
    dtype: Any = pl.Float32,
) -> pl.DataFrame:
    """Flatten a :class:`PanelTensor` back to a long ``(entity, time, ...)`` frame.

    The exact inverse of :func:`build_tensor`'s indexing: row ``i`` of the
    result is ``t.keys[i]`` and carries ``t.tensor.reshape(E * T, K)[i]``. Rows
    whose key has a null time (the padding cells of a positional
    :func:`build_sequences` tensor) are dropped -- they are not observations.

    Parameters
    ----------
    t : PanelTensor
        The tensor. Its last axis may differ in width from ``t.value_cols`` (a
        transformed tensor), in which case ``prefix`` is required.
    prefix : str, optional
        Name the ``K`` value columns ``{prefix}_0 .. {prefix}_{K-1}``. ``None``
        reuses ``t.value_cols`` (requires ``K == len(t.value_cols)``).
    dtype : polars data type, default ``pl.Float32``
        Storage dtype of the value columns. Pass ``pl.Float64`` for anything a
        user will check reconstruction error on.

    Returns
    -------
    polars.DataFrame
        Keys plus ``K`` value columns, NaN cells preserved as NaN.
    """
    arr = np.asarray(t.tensor)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"to_long: expected a 3-D tensor, got shape {arr.shape}.")
    n_ent, n_times, k = arr.shape
    if t.keys.height != n_ent * n_times:
        raise ValueError(
            f"to_long: tensor has {n_ent} x {n_times} cells but `keys` has "
            f"{t.keys.height} rows; they must line up."
        )
    if prefix is None:
        if k != len(t.value_cols):
            raise ValueError(
                f"to_long: tensor has {k} value slices but value_cols has "
                f"{len(t.value_cols)}; pass `prefix=` to name the new columns."
            )
        names = list(t.value_cols)
    else:
        names = [f"{prefix}_{j}" for j in range(k)]
    flat = arr.reshape(n_ent * n_times, k)
    out = t.keys.with_columns(
        [
            pl.Series(name, flat[:, j], dtype=pl.Float64).cast(dtype)
            for j, name in enumerate(names)
        ]
    )
    time_col = t.keys.columns[1]
    if out[time_col].null_count():
        out = out.filter(pl.col(time_col).is_not_null())
    return out


class Ragged(str, enum.Enum):
    """What to do when entities have different numbers of observations.

    ``REFUSE`` (default)
        Raise, naming the offending entities.
    ``PAD``
        Right-pad every entity with NaN to the longest length -- NaN, never a
        value. Statistically questionable; an escape hatch, not a method.
    ``TRUNCATE``
        Keep each entity's most recent ``length`` observations; entities with
        fewer are refused by name.
    ``NATIVE``
        Hand the list of per-entity matrices through unpadded (for methods that
        genuinely handle varying row counts, e.g. PARAFAC2 in Wave 3).
    """

    REFUSE = "refuse"
    PAD = "pad"
    TRUNCATE = "truncate"
    NATIVE = "native"

    def __str__(self) -> str:
        return str(self.value)


class RaggedSequences(NamedTuple):
    """Unpadded per-entity sequences (the :attr:`Ragged.NATIVE` result).

    Attributes
    ----------
    arrays : list of numpy.ndarray
        One ``(n_obs_e, n_values)`` float64 array per entity, time-ordered.
    entities : polars.Series
        Sorted entity ids, aligned with ``arrays``.
    keys : list of polars.Series
        Each entity's time values, aligned with its array's rows.
    value_cols : list of str
    """

    arrays: list[NDArray[np.float64]]
    entities: pl.Series
    keys: list[pl.Series]
    value_cols: list[str]


def _name_entities(
    entities: Sequence[Any], lengths: Sequence[int], limit: int = 8
) -> str:
    shown = ", ".join(
        f"{e!r} ({n} obs)" for e, n in list(zip(entities, lengths, strict=True))[:limit]
    )
    more = len(entities) - limit
    return shown + (f", and {more} more" if more > 0 else "")


def build_sequences(
    panel: PanelFrame,
    value_cols: Sequence[str],
    *,
    ragged: Ragged | str = Ragged.REFUSE,
    length: int | None = None,
) -> PanelTensor | RaggedSequences:
    """Stack each entity's own observations, in time order, under a ragged policy.

    Unlike :func:`build_tensor` (the union date grid), the time axis here is
    **positional**: position ``p`` is an entity's ``p``-th kept observation.
    Nothing is filled, normalised or re-scaled; nulls become NaN.

    Parameters
    ----------
    panel : PanelFrame
        The source panel. ``(entity, time)`` keys must be unique.
    value_cols : sequence of str
        Value columns (the last axis).
    ragged : Ragged or str, default "refuse"
        The policy when entities have different observation counts (see
        :class:`Ragged`).
    length : int, optional
        Required for ``"truncate"``: the number of most-recent observations to
        keep per entity.

    Returns
    -------
    PanelTensor
        For ``refuse`` / ``pad`` / ``truncate``: ``(E, T, K)`` float64, with
        ``times`` the positions ``0..T-1`` and ``keys`` the real
        ``(entity, time)`` of every cell (null time on padding cells).
    RaggedSequences
        For ``native``.

    Raises
    ------
    ValueError
        Under ``refuse`` when lengths differ, under ``truncate`` for entities
        shorter than ``length``, or on duplicate keys -- always naming the
        offending entities.
    """
    policy = Ragged(ragged)
    value_cols = list(value_cols)
    if not value_cols:
        raise ValueError("`value_cols` must name at least one feature column.")
    ent, tim = panel.entity_col, panel.time_col
    base = (
        panel.lazy()
        .select([ent, tim, *[pl.col(c).cast(pl.Float64) for c in value_cols]])
        .sort([ent, tim])
        .collect()
    )
    if base.select(pl.struct(ent, tim).is_duplicated().any()).item():
        raise ValueError(
            "build_sequences: duplicate (entity, time) keys; each entity may be "
            "observed at most once per time."
        )
    runs = base.group_by(ent, maintain_order=True).len()
    entities = runs[ent]
    lengths = runs["len"].to_numpy().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    values = base.select(value_cols).to_numpy().astype(np.float64, copy=False)
    times_all = base[tim]

    if policy is Ragged.NATIVE:
        arrays = [
            values[s : s + n].copy() for s, n in zip(starts, lengths, strict=True)
        ]
        keys = [
            times_all.slice(int(s), int(n))
            for s, n in zip(starts, lengths, strict=True)
        ]
        return RaggedSequences(arrays, entities, keys, value_cols)

    if policy is Ragged.TRUNCATE:
        if length is None or length < 1:
            raise ValueError("build_sequences: ragged='truncate' needs `length=` >= 1.")
        short = lengths < length
        if short.any():
            raise ValueError(
                f"build_sequences: ragged='truncate' keeps the last {length} "
                "observations per entity, but these entities have fewer: "
                f"{_name_entities(entities.filter(pl.Series(short)).to_list(), lengths[short].tolist())}."
                " Lower `length=` or drop them."
            )
        width = int(length)
        first = starts + lengths - width
    elif policy is Ragged.REFUSE:
        if length is not None:
            raise ValueError(
                "build_sequences: `length=` is only used with ragged='truncate'."
            )
        if lengths.size and (lengths != lengths.max()).any():
            odd = lengths != lengths.max()
            raise ValueError(
                "build_sequences: entities have different numbers of observations "
                f"(the longest has {int(lengths.max())}); refusing to pad or cut "
                "silently. Offending: "
                f"{_name_entities(entities.filter(pl.Series(odd)).to_list(), lengths[odd].tolist())}."
                " Pass ragged='truncate' with `length=`, or ragged='pad' if you "
                "accept NaN padding."
            )
        width = int(lengths.max()) if lengths.size else 0
        first = starts
    else:  # PAD
        width = int(lengths.max()) if lengths.size else 0
        first = starts

    n_ent = len(lengths)
    k = len(value_cols)
    tensor = np.full((n_ent, width, k), np.nan, dtype=np.float64)
    key_times: list[Any] = [None] * (n_ent * width)
    times_list = times_all.to_list()
    for e in range(n_ent):
        n_keep = min(int(lengths[e]), width) if policy is not Ragged.TRUNCATE else width
        s = int(first[e])
        tensor[e, :n_keep] = values[s : s + n_keep]
        key_times[e * width : e * width + n_keep] = times_list[s : s + n_keep]
    idx = np.repeat(np.arange(n_ent), width)
    key_frame = pl.DataFrame(
        [
            entities.gather(idx).alias(ent),
            pl.Series(tim, key_times, dtype=times_all.dtype),
        ]
    )
    positions = pl.Series("position", np.arange(width, dtype=np.int64))
    return PanelTensor(
        tensor=tensor,
        entities=entities,
        times=positions,
        keys=key_frame.select([ent, tim]),
        value_cols=value_cols,
    )


def refuse_missing(
    pt: PanelTensor,
    *,
    owner: str,
    ragged: Ragged | str = Ragged.REFUSE,
    length: int | None = None,
) -> PanelTensor:
    """Apply a ragged policy to a union-grid tensor for a NaN-intolerant consumer.

    A dense-grid :class:`PanelTensor` is ragged when some entity has NaN cells
    (dates it was not observed on). Linear algebra cannot consume NaN, so:

    * ``refuse`` -- raise if any cell is NaN, naming the entities;
    * ``truncate`` -- keep the most recent ``length`` grid times, then refuse
      any NaN that remains;
    * ``pad`` / ``native`` -- refused here with the reason (padding leaves NaN
      the consumer cannot use; unpadded input needs PARAFAC2, Wave 3).

    Parameters
    ----------
    pt : PanelTensor
    owner : str
        The consumer's name, for the error message.
    ragged : Ragged or str, default "refuse"
    length : int, optional
        For ``truncate``.

    Returns
    -------
    PanelTensor
        ``pt`` (possibly truncated along time), guaranteed NaN-free.
    """
    policy = Ragged(ragged)
    if policy is Ragged.PAD:
        raise ValueError(
            f"{owner}: ragged='pad' leaves NaN cells, which this decomposition "
            "cannot consume. Use 'truncate' with `length=`, or impute upstream "
            "with a point-in-time imputer."
        )
    if policy is Ragged.NATIVE:
        raise ValueError(
            f"{owner}: ragged='native' (unpadded per-entity matrices) needs a "
            "method that handles varying row counts, i.e. PARAFAC2 -- not built "
            "yet (Wave 3). Use 'refuse' or 'truncate'."
        )
    if policy is Ragged.TRUNCATE:
        if length is None or length < 1:
            raise ValueError(f"{owner}: ragged='truncate' needs `length=` >= 1.")
        n_t = pt.tensor.shape[1]
        if length < n_t:
            n_ent = pt.tensor.shape[0]
            mask = np.tile(np.arange(n_t) >= n_t - length, n_ent)
            pt = PanelTensor(
                tensor=pt.tensor[:, n_t - length :, :],
                entities=pt.entities,
                times=pt.times.tail(length),
                keys=pt.keys.filter(pl.Series(mask)),
                value_cols=pt.value_cols,
            )
    bad = ~np.isfinite(pt.tensor).all(axis=(1, 2))
    if bad.any():
        n_missing = (~np.isfinite(pt.tensor)).any(axis=2).sum(axis=1)[bad]
        names = pt.entities.filter(pl.Series(bad)).to_list()
        shown = ", ".join(
            f"{e!r} ({int(m)} missing dates)"
            for e, m in list(zip(names, n_missing, strict=True))[:8]
        )
        more = len(names) - 8
        raise ValueError(
            f"{owner}: the panel is ragged on the {pt.tensor.shape[1]}-date grid; "
            f"these entities have missing cells: {shown}"
            + (f", and {more} more" if more > 0 else "")
            + ". Refusing to fabricate values. Pass ragged='truncate' with "
            "`length=` to keep only the most recent dates, or impute upstream."
        )
    return pt
