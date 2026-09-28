"""Panel-aware dependence: per-entity estimates, their aggregate, and pooling.

Pooling a panel into one sample lets the longest entity dominate and mixes
within-entity with cross-sectional variation -- Simpson's paradox with extra
steps. So :mod:`panelary.depend` computes **within entity by default**
(:func:`by_entity`), aggregates with an explicit rule and **always** reports a
heterogeneity statistic (:func:`aggregate`), and makes pooling an explicit,
labelled choice (:func:`pooled`, with the demeaning in the ``transform`` field).

The minimum-observation gate (:func:`min_obs_for`) is what keeps a
12-observation entity out of a Fisher average: below it, an entity's estimate
is NaN and a warning says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from panelary.depend._engine import (
    DEFAULT_RESAMPLES,
    _expand_methods,
    _shift_pair,
    aggregate_estimates,
    heterogeneity,
    pooled_test,
    series_test,
)
from panelary.depend._frame import extract, result_frame
from panelary.depend._kernels import get_kernel, min_obs_for

__all__ = ["AggregateResult", "aggregate", "by_entity", "min_obs_for", "pooled"]


@dataclass(frozen=True)
class AggregateResult:
    """A panel aggregate of per-entity dependence estimates.

    Attributes
    ----------
    estimate : float
        The aggregate.
    how : str
        ``"fisher"``, ``"precision"``, ``"mean"`` or ``"median"``.
    n_entities : int
        Entities with a finite estimate.
    n_obs : int
        Complete pairs across those entities.
    i2 : float
        Higgins' ``I^2`` in ``[0, 1]``: the share of between-entity variation
        not explained by sampling noise.
    q : float
        Cochran's ``Q``.
    """

    estimate: float
    how: str
    n_entities: int
    n_obs: int
    i2: float
    q: float


def by_entity(
    df: Any,
    x: str,
    y: str,
    *,
    method: str | Sequence[str] = "xi",
    entity: str | None = None,
    time: str | None = None,
    null: str = "auto",
    lag: int = 0,
    q: float = 0.05,
    min_obs: int | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Per-entity dependence estimates and p-values, one row per entity.

    The frame is extracted **once** as a time-sorted numpy array and entities
    are sliced by their run offsets -- never ``partition_by`` into thousands of
    small frames. Each entity's p-value uses the single-series null policy
    (closed form, or a block null when the entity fails the serial pre-check).
    Entities below ``min_obs`` get a NaN estimate and a warning.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
    x, y : str
        Columns (``x -> y`` for directed methods).
    method : str or sequence of str, default="xi"
        See :func:`~panelary.depend.dependence`.
    entity, time : str, optional
        Keys for a bare frame.
    null : str, default="auto"
        Per-entity null; the panel-level ``"common-time"`` / ``"entity"`` nulls
        are not per-entity tests and are refused here.
    lag : int, default=0
        Pair ``x[t - lag]`` with ``y[t]``.
    q : float, default=0.05
        Tail probability for tail statistics.
    min_obs : int, optional
        Override the method's gate (:func:`min_obs_for`).
    n_resamples, block_length, seed
        For resampling nulls.

    Returns
    -------
    polars.DataFrame
        The entity key column, ``x``, ``y`` and the fixed result schema.
    """
    if null in {"common-time", "entity"}:
        raise ValueError(
            f"null={null!r} is a panel-level null; use dependence(..., by='entity') "
            "for the aggregate test."
        )
    pa = extract(df, [x, y], entity=entity, time=time)
    X, Y = _shift_pair(pa.dense(x), pa.dense(y), lag)
    ent_name = pa.entity_col or "entity"
    rows: list[dict[str, Any]] = []
    for m in _expand_methods(method):
        kernel = get_kernel(m, q=q)
        for i, label in enumerate(pa.entities):
            row = series_test(
                X[i],
                Y[i],
                kernel,
                null=null,
                n_resamples=n_resamples,
                block_length=block_length,
                seed=seed + i,
                min_obs=min_obs,
            )
            row.update({ent_name: label, "x": x, "y": y, "lag": int(lag)})
            rows.append(row)
    ent_values = pl.Series(pa.entities) if pa.entities != [None] else pl.Series([None])
    keys = {
        ent_name: ent_values.dtype if ent_values.dtype != pl.Null else pl.Utf8,
        "x": pl.Utf8,
        "y": pl.Utf8,
    }
    return result_frame(rows, keys=keys)


def aggregate(
    per_entity: pl.DataFrame, *, how: str | None = None, method: str | None = None
) -> AggregateResult:
    """Aggregate a :func:`by_entity` table into one number plus heterogeneity.

    Parameters
    ----------
    per_entity : polars.DataFrame
        Output of :func:`by_entity` for **one** method (needs ``estimate``,
        ``n_obs`` and ``method``).
    how : {"fisher", "precision", "mean", "median"}, optional
        ``"fisher"``: ``arctanh``, inverse-variance weights, back-transform --
        the default for correlation-like statistics. ``"precision"``:
        DerSimonian-Laird random-effects weights. ``"mean"`` / ``"median"``:
        unweighted. Defaults to the method's own default.
    method : str, optional
        Override the method read from the table.

    Returns
    -------
    AggregateResult
        Always carries ``I^2`` and Cochran's ``Q``: a pooled dependence number
        without heterogeneity is the panel equivalent of a Sharpe ratio without
        a t-stat.

    Raises
    ------
    ValueError
        If the table mixes several methods.
    """
    methods = per_entity.get_column("method").drop_nulls().unique().to_list()
    if method is None:
        if len(methods) != 1:
            raise ValueError(
                f"aggregate() needs a single method, found {sorted(methods)}; filter first."
            )
        method = str(methods[0])
    kernel = get_kernel(method)
    how_ = how or kernel.default_how
    est = (
        per_entity.get_column("estimate")
        .cast(pl.Float64)
        .fill_null(float("nan"))
        .to_numpy()
    )
    cnt = per_entity.get_column("n_obs").cast(pl.Float64).fill_null(0.0).to_numpy()
    valid = np.isfinite(est) & (cnt > 0)
    value = (
        float(aggregate_estimates(est, cnt, kernel, how_))
        if valid.any()
        else float("nan")
    )
    i2, qstat, k = heterogeneity(est, cnt, kernel, how_)
    return AggregateResult(
        estimate=value,
        how=how_,
        n_entities=k,
        n_obs=int(cnt[valid].sum()),
        i2=i2,
        q=qstat,
    )


def pooled(
    df: Any,
    x: str,
    y: str,
    *,
    method: str | Sequence[str] = "xi",
    demean: str = "none",
    entity: str | None = None,
    time: str | None = None,
    null: str = "auto",
    lag: int = 0,
    q: float = 0.05,
    min_obs: int | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """One dependence statistic on every complete pair of the panel.

    ``demean`` in ``{"none", "entity", "time", "two-way"}`` is applied first
    (alternating projections, :func:`panelary.econ._hdfe.demean`) and is
    **always** reported in ``transform``: pooled-raw and pooled-two-way-
    demeaned are different statistics, and conflating them is how Simpson's
    paradox gets into a research log. The null resamples the demeaned panel
    (common-time by default for a panel with a time column).

    Returns
    -------
    polars.DataFrame
        ``x``, ``y`` and the fixed result schema, one row per method.
    """
    pa = extract(df, [x, y], entity=entity, time=time)
    X, Y = _shift_pair(pa.dense(x), pa.dense(y), lag)
    rows: list[dict[str, Any]] = []
    for m in _expand_methods(method):
        kernel = get_kernel(m, q=q)
        row = pooled_test(
            X,
            Y,
            kernel,
            demean=demean,
            null=null,
            has_time=pa.has_time,
            n_resamples=n_resamples,
            block_length=block_length,
            seed=seed,
            min_obs=min_obs,
        )
        row.update(x=x, y=y, lag=int(lag))
        rows.append(row)
    return result_frame(rows, keys={"x": pl.Utf8, "y": pl.Utf8})
