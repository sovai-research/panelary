"""Cross-sectional and quantile labels built on the audited forward return.

Both labels read the forward return from :func:`panelary.factor.forward_return`,
the library's single audited negative-shift site (with its gap guard): there is
no second ``shift(-h)`` here. The label end time ``t1`` comes from the same
function applied to the time column (``ret=<time>`` shifts it by ``-horizon``
exactly as it shifts the return), so the return and its end time can never
disagree.

* :func:`excess_over_median` -- the forward return minus the per-date median of
  the cross-section's forward returns (optionally its sign).
* :func:`quantile_label` -- the forward return's quantile bucket, either within
  the date's cross-section or against the entity's own trailing distribution
  of **already resolved** forward returns.

Leak-safety notes
-----------------
* A cross-sectional label reads *every* constituent's forward return, so its
  information end is the latest constituent ``t1`` on that date, not the row's
  own. On a regular grid (the default gap guard) the two coincide.
* The trailing mode's thresholds use ``fwd.shift(horizon)``: at ``t`` the
  forward return of ``t - horizon`` has just resolved; nothing later is read
  (trap T13). Using the full-sample distribution would be a look-ahead.
* Survivorship (trap T14): an entity whose data ends before ``t + horizon`` has
  no forward return, so the median/rank at ``t`` is over survivors only. This
  is not a look-ahead -- the censored row is flagged and unlabelled -- but it is
  a selection effect the caller should know about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from panelary.factor._align import forward_return
from panelary.namespaces.xs import XSExprNamespace
from panelary.registry import FeatureSpec, registry

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["excess_over_median", "quantile_label"]

_FWD = "fwd_ret"
_T1_OWN = "__t1_own__"


def _prepare(
    df: PolarsFrame,
    *,
    entity: str | None,
    time: str | None,
    price: str,
    horizon: int,
    allow_gaps: bool,
    name: str,
) -> tuple[pl.DataFrame, str, str]:
    """Sorted frame with ``fwd_ret`` and the row's own end time ``__t1_own__``."""
    frame = df.collect() if isinstance(df, pl.LazyFrame) else df
    cols = frame.columns
    entity_col = entity if entity is not None else cols[0]
    time_col = time if time is not None else cols[1]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"{name}: `horizon` must be an integer >= 1, got {horizon!r}.")
    out = forward_return(
        frame,
        entity=entity_col,
        time=time_col,
        price=price,
        horizon=horizon,
        out=_FWD,
        allow_gaps=allow_gaps,
    )
    # The time column shifted by the same audited call: t1 = time[t + horizon].
    # Not `forward_return(end_time=...)`: that nulls t1 wherever the return is
    # null (e.g. a missing price), which would mark such rows as censored; here
    # t1 is null only where the horizon runs past the data.
    out = forward_return(
        out,
        entity=entity_col,
        time=time_col,
        ret=time_col,
        horizon=horizon,
        out=_T1_OWN,
        allow_gaps=True,  # the guard already ran on the first call
    )
    assert isinstance(out, pl.DataFrame)
    return out, entity_col, time_col


def _clean(fwd: pl.Expr) -> pl.Expr:
    """NaN (e.g. a 0/0 price ratio) is not a resolved return: make it null."""
    return pl.when(fwd.is_nan()).then(None).otherwise(fwd)


def _finish(frame: pl.DataFrame, label: pl.Expr, t1: pl.Expr) -> pl.DataFrame:
    """Attach ``label``, ``t1`` and ``censored``; null the label where censored."""
    censored = pl.col(_T1_OWN).is_null()
    return frame.with_columns(
        pl.when(censored).then(None).otherwise(label).alias("label"),
        pl.when(censored).then(None).otherwise(t1).alias("t1"),
        censored.alias("censored"),
    ).drop(_T1_OWN)


def excess_over_median(
    df: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "close",
    horizon: int = 21,
    binary: bool = False,
    min_count: int = 20,
    allow_gaps: bool = False,
) -> pl.DataFrame:
    """Forward return in excess of the date's cross-sectional median.

    ``label = fwd - median(fwd over the date's cross-section)`` where ``fwd`` is
    the ``horizon``-row forward return ``price[t+h] / price[t] - 1``. With
    ``binary=True`` the label is its sign (``-1 / 0 / +1``).

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel with entity, time and ``price`` columns.
    entity, time : str, optional
        Key columns. Default to the first and second column.
    price : str, default "close"
        Price column.
    horizon : int, default 21
        Forward horizon in rows of each entity's (regular) grid.
    binary : bool, default False
        Emit the sign of the excess return instead of its value.
    min_count : int, default 20
        Minimum number of entities with a resolved forward return on the date;
        dates with fewer get a null label (a median of a handful of names is
        noise).
    allow_gaps : bool, default False
        Passed to :func:`~panelary.factor.forward_return`: by default an
        irregular per-entity grid raises rather than letting the shift jump a
        gap.

    Returns
    -------
    polars.DataFrame
        Input rows sorted by ``[entity, time]`` plus ``fwd_ret`` (Float64),
        ``label`` (Float64, or Int64 when ``binary``), ``t1`` (time dtype: the
        latest constituent end time on the row's date -- the label reads every
        constituent's forward return) and ``censored`` (Boolean: the row's own
        forward window runs past its entity's data; label and ``t1`` are null).

    Notes
    -----
    Survivorship: entities without a forward return on a date (delisted within
    the horizon) are excluded from that date's median. Across gaps
    (``allow_gaps=True``) a constituent that is missing in a data prefix may
    reappear later, so resolved-prefix invariance is only guaranteed on a
    gap-free grid.
    """
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError(
            f"excess_over_median: `min_count` must be an integer >= 1, got {min_count!r}."
        )
    frame, _entity_col, time_col = _prepare(
        df,
        entity=entity,
        time=time,
        price=price,
        horizon=horizon,
        allow_gaps=allow_gaps,
        name="excess_over_median",
    )
    fwd = _clean(pl.col(_FWD))
    n_ok = fwd.is_not_null().sum().over(time_col)
    median = fwd.median().over(time_col)
    excess = pl.when(n_ok >= min_count).then(fwd - median).otherwise(None)
    label = excess.sign().cast(pl.Int64) if binary else excess
    t1 = pl.col(_T1_OWN).max().over(time_col)
    return _finish(frame, label, t1)


def _symmetric(bucket: pl.Expr, q: int) -> pl.Expr:
    """Map buckets ``0 .. q-1`` onto a symmetric integer scale.

    Odd ``q``: ``b - (q-1)/2`` (``q=3 -> -1, 0, +1``). Even ``q``: no zero bucket,
    ``q=4 -> -2, -1, +1, +2``.
    """
    half = q // 2
    if q % 2:
        return (bucket - half).cast(pl.Int64)
    return (
        pl.when(bucket < half).then(bucket - half).otherwise(bucket - half + 1)
    ).cast(pl.Int64)


def _trailing_thresholds(
    fwd: pl.Expr, *, entity: str, horizon: int, q: int, lookback: int
) -> list[pl.Expr]:
    """The ``q - 1`` causal bucket edges at each row, per entity.

    Built from ``fwd.shift(horizon)``: at ``t`` that is the forward return of
    ``t - horizon``, which resolved at ``t``. So the thresholds at ``t`` read
    prices up to ``t`` only (trap T13).
    """
    past = fwd.shift(horizon)
    return [
        past.rolling_quantile(
            quantile=k / q,
            interpolation="linear",
            window_size=lookback,
            min_samples=lookback,
        ).over(entity)
        for k in range(1, q)
    ]


def quantile_label(
    df: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "close",
    horizon: int = 21,
    q: int = 3,
    mode: Literal["cross_section", "trailing"] = "cross_section",
    lookback: int = 252,
    allow_gaps: bool = False,
) -> pl.DataFrame:
    """Quantile-bucket label of the forward return, on a symmetric integer scale.

    ``mode="cross_section"``: the forward return's rank within the date's
    cross-section, cut into ``q`` equal-count buckets
    (:meth:`.xs.quantile_bin`); dates with fewer than ``q`` resolved names are
    null. ``mode="trailing"``: the forward return is compared with ``q - 1``
    thresholds, the ``k/q`` quantiles (linear interpolation) of the entity's
    previous ``lookback`` **resolved** forward returns; null until ``lookback``
    of them exist.

    Buckets ``0 .. q-1`` map to ``-(q-1)/2 .. +(q-1)/2`` for odd ``q``
    (``q=3 -> -1, 0, +1``) and to ``-q/2 .. -1, +1 .. +q/2`` for even ``q``.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel with entity, time and ``price`` columns.
    entity, time : str, optional
        Key columns. Default to the first and second column.
    price : str, default "close"
        Price column.
    horizon : int, default 21
        Forward horizon in rows.
    q : int, default 3
        Number of buckets, ``>= 2``.
    mode : {"cross_section", "trailing"}, default "cross_section"
        Where the bucket edges come from.
    lookback : int, default 252
        Trailing mode: number of resolved forward returns per threshold.
    allow_gaps : bool, default False
        Passed to :func:`~panelary.factor.forward_return`.

    Returns
    -------
    polars.DataFrame
        Input rows sorted by ``[entity, time]`` plus ``fwd_ret``, ``label``
        (Int64), ``t1`` (time dtype) and ``censored`` (Boolean). In
        cross-section mode ``t1`` is the latest constituent end time on the
        date; in trailing mode it is the row's own.
    """
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise ValueError(f"quantile_label: `q` must be an integer >= 2, got {q!r}.")
    if mode not in ("cross_section", "trailing"):
        raise ValueError(
            f"quantile_label: `mode` must be 'cross_section' or 'trailing', got {mode!r}."
        )
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError(
            f"quantile_label: `lookback` must be a fixed integer >= 2, got {lookback!r}."
        )
    frame, entity_col, time_col = _prepare(
        df,
        entity=entity,
        time=time,
        price=price,
        horizon=horizon,
        allow_gaps=allow_gaps,
        name="quantile_label",
    )
    fwd_clean = _clean(pl.col(_FWD))
    if mode == "cross_section":
        n_ok = fwd_clean.is_not_null().sum().over(time_col)
        bucket = XSExprNamespace(fwd_clean).quantile_bin(q).over(time_col)
        label = pl.when(n_ok >= q).then(_symmetric(bucket, q)).otherwise(None)
        t1 = pl.col(_T1_OWN).max().over(time_col)
    else:
        edges = _trailing_thresholds(
            fwd_clean, entity=entity_col, horizon=horizon, q=q, lookback=lookback
        )
        bucket = pl.sum_horizontal([(fwd_clean > e).cast(pl.Int64) for e in edges])
        known = pl.all_horizontal([e.is_not_null() for e in edges]) & (
            fwd_clean.is_not_null()
        )
        label = pl.when(known).then(_symmetric(bucket, q)).otherwise(None)
        t1 = pl.col(_T1_OWN)
    return _finish(frame, label, t1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _label_spec(name: str, params: dict[str, type]) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        namespace="label",
        input_shape="frame",
        output_shape="frame",
        params=params,
        tier="B",
        # The forward return is per entity; the median / rank mixes entities on
        # a date by design, like `.xs` operators -- see
        # `_INTENTIONALLY_NOT_PANEL_SAFE` in tests/test_registry_conformance.py.
        panel_safe=False,
        leakage_safe=True,
        # Forward labels: the `forward_return` precedent (it reads t + horizon
        # by definition, so it must never be read as a row-safe feature).
        safe_scope="window",
        source="Panelary",
        license="Apache-2.0",
    )


for _s in (
    _label_spec(
        "excess_over_median",
        {"price": str, "horizon": int, "binary": bool, "min_count": int},
    ),
    _label_spec(
        "quantile_label",
        {"price": str, "horizon": int, "q": int, "mode": str, "lookback": int},
    ),
):
    registry.register(_s)
