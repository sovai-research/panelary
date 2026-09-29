"""Sample weights from label spans: return attribution, time decay, class balance.

AFML ch. 4.10-4.11 and the scikit-learn ``"balanced"`` rule, computed on the
one span table. The kernels (``_*`` functions) take a :class:`SpanTable` and
are shared by the global frame functions here and by the fold-local
:class:`~panelary.weights.FoldWeights`; only *which labels* they see differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core._spans import (
    SpanTable,
    _concurrency,
    _span_sums,
    _spans_from_t1,
    _uniqueness,
)
from panelary.weights._concurrency import _label_column

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["attach", "class_weights", "return_attribution", "time_decay"]

_KINDS = ("uniqueness", "return", None)


# --------------------------------------------------------------------------- #
# Kernels on a span table
# --------------------------------------------------------------------------- #
def _log_returns(frame: pl.DataFrame, price: str, seg_first: np.ndarray) -> np.ndarray:
    """Per-row log return within entity; 0 at each entity's first row."""
    if price not in frame.columns:
        raise ValueError(
            f"price column {price!r} not found; columns are {frame.columns}."
        )
    p = frame.get_column(price).cast(pl.Float64).fill_null(np.nan).to_numpy()
    r = np.zeros(p.shape[0], dtype=np.float64)
    if p.shape[0] > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = np.log(p[1:] / p[:-1])
    r[seg_first] = 0.0
    return r


def _entity_first_rows(frame: pl.DataFrame, entity: str) -> np.ndarray:
    seg = frame.get_column(entity).rle_id().to_numpy()
    return np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]]) if seg.size else seg


def _return_attribution(
    spans: SpanTable,
    log_ret: np.ndarray,
    concurrency: NDArray[np.int64] | None = None,
    *,
    afml_compat: bool = False,
) -> NDArray[np.float64]:
    """``|sum_{r in (t0, t1]} ret_r / c_r|`` per span (AFML 4.10).

    The return at ``t0`` is earned *before* the event, so it is excluded by
    default; ``afml_compat=True`` sums the inclusive ``[t0, t1]`` as AFML's
    snippet does. A span containing a non-finite return gets ``NaN``.
    """
    c = (
        _concurrency(spans.start, spans.end, spans.n_rows)
        if concurrency is None
        else concurrency
    )
    covered = c > 0
    v = np.zeros(c.shape[0], dtype=np.float64)
    v[covered] = log_ret[covered] / c[covered]
    # A NaN inside a block would poison every prefix after it in that block,
    # so sum a cleaned copy and mark the spans that really contain one.
    bad = ~np.isfinite(v)
    offset = 0 if afml_compat else 1
    clean = np.where(bad, 0.0, v)
    out = np.abs(
        _span_sums(
            clean, spans.start, spans.end, offset=offset, segments=spans.segments
        )
    )
    if bool(bad.any()):
        hit = _span_sums(
            bad.astype(np.float64),
            spans.start,
            spans.end,
            offset=offset,
            segments=spans.segments,
        )
        out[hit > 0] = np.nan
    return out


def _time_decay(
    uniqueness: NDArray[np.float64],
    spans: SpanTable,
    c: float,
) -> NDArray[np.float64]:
    """Linear decay on cumulative uniqueness (AFML 4.11); newest label -> 1.

    Labels are ordered by ``(t0, entity)``; ``x`` is the running sum of their
    uniqueness and ``X`` its total -- a *fit* over whichever labels are passed
    (the training fold, inside CV). ``c >= 0``: the oldest label decays to
    about ``c``; ``c < 0``: the oldest ``|c|`` share of cumulative uniqueness
    gets weight 0.
    """
    _check_decay(c)
    n = uniqueness.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    # (t0, entity) is unique per label, so one composite key orders them
    # deterministically; polars' parallel arg_sort is ~10x numpy's lexsort here.
    n_ent = int(spans.entity_code.max()) + 1
    key = spans.start_tpos * n_ent + spans.entity_code
    order = pl.Series(key).arg_sort().to_numpy().astype(np.int64)
    x = np.cumsum(uniqueness[order])
    total = x[-1]
    slope = (1.0 - c) / total if c >= 0 else 1.0 / ((c + 1.0) * total)
    const = 1.0 - slope * total
    out[order] = np.maximum(0.0, const + slope * x)
    return out


def _check_decay(c: float) -> None:
    if not (-1.0 < float(c) <= 1.0):
        raise ValueError(f"time-decay `c` must lie in (-1, 1], got {c!r}.")


def _class_factor(
    labels: np.ndarray, uniqueness: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Per-label class weight ``n / (K * n_k)`` over the classes present."""
    out = np.ones(labels.shape[0], dtype=np.float64)
    if labels.shape[0] == 0:
        return out
    for cls, w in class_weights(labels, effective=uniqueness).items():
        out[np.asarray(labels == cls, dtype=bool)] = w
    return out


def _combine(
    spans: SpanTable,
    *,
    kind: str | None,
    log_ret: np.ndarray | None,
    decay: float | None,
    labels: np.ndarray | None,
    normalize: str | None,
    afml_compat: bool = False,
) -> NDArray[np.float64]:
    """base (uniqueness or return attribution) x decay x class, normalised."""
    n = len(spans)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    c = _concurrency(spans.start, spans.end, spans.n_rows)
    u = _uniqueness(spans, c) if (kind == "uniqueness" or decay is not None) else None
    if kind == "uniqueness":
        assert u is not None
        w = u.copy()
    elif kind == "return":
        assert log_ret is not None
        w = _return_attribution(spans, log_ret, c, afml_compat=afml_compat)
    else:
        w = np.ones(n, dtype=np.float64)
    if decay is not None:
        assert u is not None
        w = w * _time_decay(u, spans, decay)
    if labels is not None:
        w = w * _class_factor(labels)
    if normalize == "sum_n":
        total = float(np.sum(w))
        if total > 0 and np.isfinite(total):
            w = w * (n / total)
    return w


def _check_kind(kind: str | None, price: str | None) -> None:
    if kind not in _KINDS:
        raise ValueError(f"`kind` must be one of {_KINDS}, got {kind!r}.")
    if kind == "return" and price is None:
        raise ValueError('kind="return" needs the `price=` column.')


def _labels_of(frame: pl.DataFrame, spans: SpanTable, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(
            f"balance column {column!r} not found; columns are {frame.columns}."
        )
    values = frame.get_column(column).gather(spans.label_row)
    if values.null_count():
        raise ValueError(
            f"balance column {column!r} is null on {values.null_count()} resolved "
            "label row(s); class weights need a class for every label."
        )
    return values.to_numpy()


# --------------------------------------------------------------------------- #
# Public frame functions (global)
# --------------------------------------------------------------------------- #
def return_attribution(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    price: str = "close",
    afml_compat: bool = False,
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
    out: str = "w_ret",
) -> pl.DataFrame:
    """Attach return-attribution weights ``|sum_{(t0, t1]} r / c|`` (AFML 4.10).

    **Global** (every label counts in the concurrency ``c``); inside
    cross-validation use :class:`FoldWeights` with ``kind="return"``.
    A label earns the log returns over its span, each shared equally with the
    labels active on that row, so large moves weigh more and crowded periods
    are split. Already embeds concurrency: do not multiply it by uniqueness.

    Parameters
    ----------
    df, t1, entity, time, censored
        As in :func:`panelary.weights.spans`.
    price : str, default "close"
        Price column; returns are ``log(p_r / p_{r-1})`` within entity.
    afml_compat : bool, default False
        ``False`` sums returns over ``(t0, t1]`` -- the return at ``t0`` is
        earned before the event. ``True`` uses AFML's inclusive ``[t0, t1]``.
    out : str, default "w_ret"
        Output column: raw (unnormalised) weights on label rows, null elsewhere.

    Returns
    -------
    polars.DataFrame
        The frame sorted by ``(entity, time)`` with the ``out`` column.
    """
    frame, table = _spans_from_t1(
        df, t1=t1, entity=entity, time=time, censored=censored
    )
    ent = frame.columns[0] if entity is None else entity
    log_ret = _log_returns(frame, price, _entity_first_rows(frame, ent))
    w = _return_attribution(table, log_ret, afml_compat=afml_compat)
    return frame.with_columns(_label_column(table.n_rows, table.label_row, w, out))


def time_decay(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    c: float = 1.0,
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
    out: str = "w_decay",
) -> pl.DataFrame:
    """Attach linear time-decay factors on cumulative uniqueness (AFML 4.11).

    **Global**: the decay is anchored to the total uniqueness of every label in
    the frame -- a fit. Inside cross-validation use :class:`FoldWeights` with
    ``decay=`` so the anchor is the training fold's.

    Labels are ordered by ``(t0, entity)``. The newest gets ``1``. With
    ``c >= 0`` the factor falls linearly in cumulative uniqueness to about
    ``c`` for the oldest; with ``-1 < c < 0`` the oldest ``|c|`` share gets
    ``0``. ``c = 1`` means no decay.

    Parameters
    ----------
    df, t1, entity, time, censored
        As in :func:`panelary.weights.spans`.
    c : float, default 1.0
        Weight of the oldest label, in ``(-1, 1]``.
    out : str, default "w_decay"
        Output column on label rows, null elsewhere.

    Returns
    -------
    polars.DataFrame
        The frame sorted by ``(entity, time)`` with the ``out`` column.
    """
    _check_decay(c)
    frame, table = _spans_from_t1(
        df, t1=t1, entity=entity, time=time, censored=censored
    )
    d = _time_decay(_uniqueness(table), table, c)
    return frame.with_columns(_label_column(table.n_rows, table.label_row, d, out))


def class_weights(
    labels: ArrayLike | pl.Series, effective: ArrayLike | None = None
) -> dict[Any, float]:
    """Balanced class weights ``n / (K * n_k)`` over the classes present.

    scikit-learn's ``compute_class_weight("balanced")`` rule. Inside
    cross-validation it must be computed from the training fold's labels only
    (:class:`FoldWeights` does); from the full sample it leaks the test fold's
    class mix into training.

    Parameters
    ----------
    labels : array-like or polars.Series
        One class per label; nulls / NaN are ignored.
    effective : array-like, optional
        Per-label average uniqueness. When given, class mass is
        ``n_k = sum_{i in k} ubar_i`` and ``n = sum ubar``: overlapping labels of
        one class count once, not ``k`` times.

    Returns
    -------
    dict
        ``{class: weight}``, classes in sorted order.
    """
    y = labels.to_numpy() if isinstance(labels, pl.Series) else np.asarray(labels)
    m = None if effective is None else np.asarray(effective, dtype=np.float64)
    if m is not None and m.shape != y.shape:
        raise ValueError(f"`effective` has shape {m.shape} but `labels` has {y.shape}.")
    keep = ~_missing(y)
    y = y[keep]
    if y.size == 0:
        return {}
    classes, inverse = np.unique(y, return_inverse=True)
    if m is None:
        n_k = np.bincount(inverse).astype(np.float64)
        n = float(y.size)
    else:
        n_k = np.bincount(inverse, weights=m[keep])
        n = float(np.sum(m[keep]))
    k = classes.shape[0]
    w = n / (k * n_k)
    return {
        (cls.item() if hasattr(cls, "item") else cls): float(wk)
        for cls, wk in zip(classes, w, strict=True)
    }


def _missing(y: np.ndarray) -> NDArray[np.bool_]:
    if y.dtype.kind == "f":
        return np.isnan(y)
    if y.dtype.kind == "O":
        return np.array([v is None or v != v for v in y.tolist()], dtype=bool)
    return np.zeros(y.shape[0], dtype=bool)


def attach(
    df: PolarsFrame,
    *,
    t1: str = "t1",
    kind: str | None = "uniqueness",
    price: str | None = None,
    decay: float | None = None,
    balance: str | None = None,
    afml_compat: bool = False,
    entity: str | None = None,
    time: str | None = None,
    censored: str | None = "censored",
    out: str = "w",
) -> pl.DataFrame:
    """Attach **global** sample weights -- final refit only, never inside CV.

    Every label in the frame shapes every weight (concurrency, the decay anchor,
    class frequencies, the normalisation), so a model cross-validated with
    these weights has seen the test folds' label spans, prices and class mix
    (trap T1). Use :class:`FoldWeights` inside cross-validation, and this for
    exploration or for the last fit on all data.

    Weight = base x decay x class, normalised so the weights of the resolved
    labels sum to their number:

    * base: ``kind="uniqueness"`` (average uniqueness), ``"return"`` (return
      attribution, needs ``price=``; it already embeds concurrency, so the two
      are mutually exclusive) or ``None`` (1);
    * decay: ``decay=c`` multiplies by :func:`time_decay` factors;
    * class: ``balance=<label column>`` multiplies by :func:`class_weights`.

    Parameters
    ----------
    df, t1, entity, time, censored
        As in :func:`panelary.weights.spans`.
    kind, price, decay, balance, afml_compat
        See above.
    out : str, default "w"
        Output Float64 column. Rows that are not resolved labels (null ``t1``,
        censored, beyond the data) get ``0.0`` so they drop out of a weighted
        fit.

    Returns
    -------
    polars.DataFrame
        The frame sorted by ``(entity, time)`` with the ``out`` column.
    """
    _check_kind(kind, price)
    if decay is not None:
        _check_decay(decay)
    frame, table = _spans_from_t1(
        df, t1=t1, entity=entity, time=time, censored=censored
    )
    ent = frame.columns[0] if entity is None else entity
    log_ret = (
        _log_returns(frame, price, _entity_first_rows(frame, ent))
        if kind == "return" and price is not None
        else None
    )
    labels = _labels_of(frame, table, balance) if balance is not None else None
    w = _combine(
        table,
        kind=kind,
        log_ret=log_ret,
        decay=decay,
        labels=labels,
        normalize="sum_n",
        afml_compat=afml_compat,
    )
    col = np.zeros(table.n_rows, dtype=np.float64)
    col[table.label_row] = w
    return frame.with_columns(pl.Series(out, col, dtype=pl.Float64))
