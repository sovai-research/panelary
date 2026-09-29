"""Bet sizing from predicted probabilities (AFML ch. 10).

* :func:`bet_size` -- ``m = pred * side * (2 Phi(z) - 1)`` with
  ``z = (p - 1/K) / sqrt(p (1 - p))``: the size implied by the probability ``p``
  of the predicted class, under the null that the classifier is no better than
  uniform over ``K`` classes.
* :func:`average_active` -- at every row, the mean size of the entity's bets
  that are active there (``t0 <= t < exit``), via an event stream on the grid.
* :func:`discretize` -- round sizes to a step, half-to-even.

Leak-safety
-----------
* Probabilities must be **out-of-fold** (trap T16): a size computed from
  in-sample probabilities inherits the fit's overconfidence. Use the
  out-of-fold predictions of :func:`panelary.cross_validate`.
* :func:`average_active` takes an ``exit_time`` column, deliberately not
  ``t1``: a scanning label's ``t1`` is an information end, not a causal exit
  (trap T15). The average at ``t`` depends only on bets opened at or before
  ``t`` and on whether each has exited by ``t`` -- moving an exit anywhere
  later than ``t`` leaves it unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import polars as pl

from panelary._internal._special import norm_cdf

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = ["average_active", "bet_size", "discretize"]

#: Probabilities are clipped to ``[_EPS, 1 - _EPS]`` so ``z`` stays finite.
_EPS = 1e-12


def _frame(df: PolarsFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def bet_size(
    df: PolarsFrame,
    *,
    prob: str = "prob",
    side: str | None = "side",
    pred: str | None = None,
    n_classes: int = 2,
    out: str = "size",
) -> pl.DataFrame:
    """Bet size from the predicted class's probability (AFML snippet 10.1).

    ``z = (p - 1/K) / sqrt(p (1 - p))`` tests the probability ``p`` of the
    predicted class against the uniform ``1/K``; the size is
    ``m = (2 Phi(z) - 1) * pred * side`` in ``[-1, 1]``, with ``pred`` / ``side``
    taken as 1 when not given. ``p`` is clipped to ``[1e-12, 1 - 1e-12]``.

    Typical uses: meta-labelling (``side`` = the primary model's side, ``prob``
    = the meta-model's probability of "act") or a sided classifier (``pred`` =
    the predicted label in ``{-1, +1}``, ``side=None``).

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Frame with the probability (and optional side / prediction) columns.
    prob : str, default "prob"
        Probability of the predicted class, in ``[0, 1]``. **Out-of-fold.**
    side : str or None, default "side"
        Primary side (its sign is used); ``None`` for no meta-labelling.
    pred : str or None, default None
        Predicted label (its value multiplies the size, e.g. ``{-1, 0, +1}``).
    n_classes : int, default 2
        Number of classes ``K``.
    out : str, default "size"
        Output column.

    Returns
    -------
    polars.DataFrame
        The input with ``out`` (Float64, null where ``prob`` is null).

    Notes
    -----
    Phi is :func:`panelary._internal._special.norm_cdf` (``erfc``-exact, no scipy).
    """
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError(
            f"bet_size: `n_classes` must be an integer >= 2, got {n_classes!r}."
        )
    frame = _frame(df)
    p = frame.get_column(prob).cast(pl.Float64)
    valid = p.is_not_null() & ~p.is_nan()
    pv = p.to_numpy().astype(np.float64)
    finite = pv[valid.to_numpy()]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError(f"bet_size: `{prob}` must lie in [0, 1].")
    with np.errstate(invalid="ignore"):
        pc = np.clip(pv, _EPS, 1.0 - _EPS)
        z = (pc - 1.0 / n_classes) / np.sqrt(pc * (1.0 - pc))
        m = 2.0 * np.asarray(norm_cdf(np.where(np.isfinite(z), z, 0.0))) - 1.0
    size = pl.Series(out, m, dtype=pl.Float64)
    if pred is not None:
        size = size * frame.get_column(pred).cast(pl.Float64)
    if side is not None:
        size = size * frame.get_column(side).cast(pl.Float64).sign()
    size = pl.select(pl.when(valid).then(size).otherwise(None)).to_series().alias(out)
    return frame.with_columns(size)


def average_active(
    df: PolarsFrame,
    *,
    size: str = "size",
    exit_time: str = "exit",
    entity: str | None = None,
    time: str | None = None,
    out: str = "avg_size",
) -> pl.DataFrame:
    """Mean size of the entity's active bets at every row (AFML snippet 10.2).

    A row with a non-null ``size`` opens a bet at its time ``t0``; the bet is
    active on the entity's rows with ``t0 <= t < exit`` (a null ``exit`` keeps
    it open to the entity's last row). At each row the output is the mean size
    of the active bets, and exactly ``0.0`` where none is active.

    Computed as an event stream on the grid -- ``+m`` at the opening row, ``-m``
    at the first row at or after the exit -- with the running sum restarted at
    every idle row and entity start (a busy-period reset), so float drift never
    survives an idle period. O(rows + bets); AFML's reference re-scans every
    bet at every change point.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel on the evaluation grid; bet rows carry ``size`` and
        ``exit_time``, other rows have a null ``size``.
    size : str, default "size"
        Bet size (e.g. from :func:`bet_size`).
    exit_time : str, default "exit"
        When the bet is closed, in the time column's dtype. A **causal** exit,
        not a label's ``t1``.
    entity, time : str, optional
        Key columns; default to the first and second column.
    out : str, default "avg_size"
        Output column.

    Returns
    -------
    polars.DataFrame
        The input sorted by ``(entity, time)`` with ``out`` (Float64).
    """
    frame = _frame(df)
    cols = frame.columns
    entity_col = entity if entity is not None else cols[0]
    time_col = time if time is not None else cols[1]
    frame = frame.sort([entity_col, time_col])
    n = frame.height
    if n == 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(out))

    # Composite (entity, time-rank) key: strictly increasing along the grid.
    seg = frame.get_column(entity_col).rle_id().to_numpy().astype(np.int64)
    t_phys = frame.select(pl.col(time_col).to_physical()).to_series().to_numpy()
    times = np.unique(t_phys)
    n_t = times.shape[0] + 1
    grid_key = seg * n_t + np.searchsorted(times, t_phys, side="left")
    seg_last = np.r_[np.flatnonzero(seg[1:] != seg[:-1]), n - 1]
    seg_end = seg_last[seg]

    m = frame.get_column(size).cast(pl.Float64).to_numpy()
    is_bet = ~np.isnan(m)
    rows = np.flatnonzero(is_bet)
    ex = frame.get_column(exit_time)
    if ex.dtype != frame.schema[time_col]:
        raise TypeError(
            f"average_active: `{exit_time}` must have the time column's dtype "
            f"({frame.schema[time_col]}), got {ex.dtype}."
        )
    ex_null = ex.is_null().to_numpy()[rows]
    ex_phys = frame.select(pl.col(exit_time).to_physical()).to_series().to_numpy()[rows]
    if np.any(~ex_null & (ex_phys < t_phys[rows])):
        raise ValueError(
            f"average_active: `{exit_time}` is earlier than the bet's own time on "
            "some rows; an exit must not precede its entry."
        )
    safe_ex = np.where(ex_null, times[-1], ex_phys)
    exit_key = seg[rows] * n_t + np.searchsorted(times, safe_ex, side="left")
    exit_row = np.searchsorted(grid_key, exit_key, side="left")
    exit_row = np.where(
        ex_null, seg_end[rows] + 1, np.minimum(exit_row, seg_end[rows] + 1)
    )

    # Counts: exact integers; the removal at seg_end + 1 returns them to zero.
    d_cnt = np.bincount(rows, minlength=n + 1) - np.bincount(exit_row, minlength=n + 1)
    count = np.cumsum(d_cnt[:n])
    # Sizes: removals only inside the entity (the reset handles the boundary).
    inside = exit_row <= seg_end[rows]
    d_sum = np.bincount(rows, weights=m[rows], minlength=n + 1)[:n]
    d_sum = (
        d_sum
        - np.bincount(exit_row[inside], weights=m[rows][inside], minlength=n + 1)[:n]
    )
    prev_idle = np.r_[True, count[:-1] == 0]
    new_segment = prev_idle | np.r_[True, seg[1:] != seg[:-1]]
    busy_id = np.cumsum(new_segment)
    running = (
        pl.DataFrame({"d": d_sum, "g": busy_id})
        .select(pl.col("d").cum_sum().over("g"))
        .to_series()
        .to_numpy()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(count > 0, running / np.maximum(count, 1), 0.0)
    return frame.with_columns(pl.Series(out, avg, dtype=pl.Float64))


@overload
def discretize(x: float, step: float = ...) -> float: ...
@overload
def discretize(x: pl.Series, step: float = ...) -> pl.Series: ...
@overload
def discretize(x: np.ndarray, step: float = ...) -> np.ndarray: ...
def discretize(x: Any, step: float = 0.1) -> Any:
    """Round sizes to multiples of ``step`` (half-to-even) and clip to ``[-1, 1]``.

    ``np.round`` is used, not ``Expr.round``, whose rounding mode is not
    guaranteed across polars versions (trap T21). Returns the input's type
    (float, numpy array or polars Series).
    """
    if not (
        isinstance(step, (int, float)) and math.isfinite(step) and 0.0 < step <= 1.0
    ):
        raise ValueError(f"discretize: `step` must be in (0, 1], got {step!r}.")
    if isinstance(x, pl.Series):
        vals = x.cast(pl.Float64).to_numpy()
        return pl.Series(x.name, np.clip(np.round(vals / step) * step, -1.0, 1.0))
    arr = np.asarray(x, dtype=np.float64)
    res = np.clip(np.round(arr / step) * step, -1.0, 1.0)
    return float(res) if res.ndim == 0 else res
