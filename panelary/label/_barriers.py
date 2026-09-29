"""López de Prado labeling primitives (AFML Ch. 3), Polars-native and leak-safe.

All labelers look *forward* only over an event span ``[t, t1]`` and emit a
``t1`` column -- the timestamp at which the label is resolved (a barrier touch
or the vertical/horizon barrier). ``t1`` is a shared contract consumed by
purged cross-validation: it is always returned in the *same dtype* as the input
time column so that a training row whose span ``[t, t1]`` overlaps a test set
can be purged.

The volatility estimate used to scale the barriers is strictly *trailing*
(past-and-present only), so no information beyond ``[t, t1]`` enters a label.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from panelary.factor._align import _check_no_gaps, forward_return

if TYPE_CHECKING:
    from panelary._internal._type_aliases import PolarsFrame

__all__ = [
    "triple_barrier",
    "fixed_horizon",
    "meta_label",
]


def _as_dataframe(df: PolarsFrame) -> pl.DataFrame:
    """Materialize a LazyFrame to a DataFrame; pass a DataFrame through."""
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


def _resolve_cols(
    df: pl.DataFrame,
    entity: str | None,
    time: str | None,
) -> tuple[str, str]:
    """Resolve entity/time columns, defaulting to the first two columns."""
    cols = df.columns
    entity_col = entity if entity is not None else cols[0]
    time_col = time if time is not None else cols[1]
    return entity_col, time_col


def _trailing_std(returns: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling standard deviation of ``returns``.

    Element ``i`` uses only ``returns[max(0, i - window + 1) : i + 1]`` (i.e. a
    window ending at and including ``i``), so the estimate never peeks forward.
    NaNs (e.g. the undefined first return) are ignored; positions with fewer
    than two valid observations are ``NaN``.

    Parameters
    ----------
    returns : numpy.ndarray
        1-D array of period returns.
    window : int
        Trailing look-back length in observations.

    Returns
    -------
    numpy.ndarray
        Trailing sample standard deviation (``ddof=1``), same length as input.
    """
    n = returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        w = returns[lo : i + 1]
        w = w[~np.isnan(w)]
        if w.shape[0] >= 2:
            out[i] = w.std(ddof=1)
    return out


def _triple_barrier_entity(
    prices: np.ndarray,
    pt: float,
    sl: float,
    max_holding: int,
    vol_lookback: int,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute triple-barrier outcomes for one entity's time-ordered path.

    Returns the ``(label, ret, t1_index, censored)`` arrays, where
    ``t1_index[i]`` is the positional index (into this entity's rows) at which
    observation ``i`` is resolved and ``censored[i]`` is True when no
    horizontal barrier was touched *and* the vertical barrier
    ``i + max_holding`` lies beyond the entity's last row (the label was never
    resolved). ``sigma`` overrides the built-in trailing standard deviation.
    """
    n = prices.shape[0]
    if sigma is None:
        returns = np.empty(n, dtype=np.float64)
        returns[0] = np.nan
        if n > 1:
            returns[1:] = prices[1:] / prices[:-1] - 1.0
        sigma = _trailing_std(returns, vol_lookback)

    label = np.zeros(n, dtype=np.int64)
    ret_out = np.zeros(n, dtype=np.float64)
    t1_idx = np.arange(n, dtype=np.int64)

    for i in range(n):
        end = min(i + max_holding, n - 1)
        touch = end  # vertical barrier by default
        lab = 0
        s = sigma[i]

        if not np.isnan(s) and s > 0.0:
            upper = prices[i] * (1.0 + pt * s)
            lower = prices[i] * (1.0 - sl * s)
            for j in range(i + 1, end + 1):
                if pt > 0.0 and prices[j] >= upper:
                    touch, lab = j, 1
                    break
                if sl > 0.0 and prices[j] <= lower:
                    touch, lab = j, -1
                    break

        t1_idx[i] = touch
        ret_out[i] = prices[touch] / prices[i] - 1.0
        label[i] = lab

    rows = np.arange(n, dtype=np.int64)
    censored = (label == 0) & (rows + max_holding > n - 1)
    return label, ret_out, t1_idx, censored


def triple_barrier(
    df: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "close",
    pt: float = 2.0,
    sl: float = 1.0,
    max_holding: int,
    vol_lookback: int = 20,
    vol: str | None = None,
) -> pl.DataFrame:
    """Triple-barrier labels (AFML Ch. 3).

    For each observation the method looks forward over the window
    ``[t, t + max_holding]`` (in time order, within the observation's entity)
    and places a profit-take and a stop-loss barrier, both scaled by a
    *trailing* volatility estimate. The label is the sign of the first barrier
    touched; if neither horizontal barrier is touched before the vertical
    barrier (``t + max_holding`` steps), the label is ``0``.

    Leak-safety: the volatility estimate is trailing (past-and-present only)
    and the forward scan stops at the first touch, so nothing beyond ``[t, t1]``
    can influence a row's label.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Panel with at least an entity, a time, and a ``price`` column.
    entity : str, optional
        Entity (grouping) column. Defaults to the first column.
    time : str, optional
        Time column. Defaults to the second column.
    price : str, default "close"
        Price column used for barriers and returns.
    pt : float, default 2.0
        Profit-take barrier multiple of trailing volatility. ``0`` disables it.
    sl : float, default 1.0
        Stop-loss barrier multiple of trailing volatility. ``0`` disables it.
    max_holding : int
        Vertical-barrier horizon, in number of forward steps (rows) per entity.
    vol_lookback : int, default 20
        Trailing look-back (in rows) for the built-in volatility estimate.
        Ignored when ``vol`` is given.
    vol : str, optional
        Column holding a **causal** (trailing) volatility estimate, in return
        units, to scale the barriers instead of the built-in trailing standard
        deviation -- e.g. a range-based estimator. It must only use data up to
        and including each row; a centred or full-sample estimate leaks the
        future into the barrier widths. Null / NaN / non-positive values
        disable the horizontal barriers on that row, as a warm-up does.

    Returns
    -------
    polars.DataFrame
        The input rows (sorted by ``[entity, time]``) with four columns added:

        ``label`` : Int64
            ``+1`` (profit-take), ``-1`` (stop-loss), or ``0`` (vertical).
        ``ret`` : Float64
            Realized return from ``t`` to the touched barrier.
        ``t1`` : same dtype as ``time``
            Timestamp at which the label is resolved. Always ``t <= t1`` and
            within ``max_holding`` steps of ``t``.
        ``censored`` : Boolean
            ``True`` when no horizontal barrier was touched **and** the
            vertical barrier ``t + max_holding`` steps lies beyond the entity's
            last row. Such a row's ``label = 0`` / ``t1`` describe a *truncated*
            horizon (the last row gets ``label = 0, t1 = t``): the data cannot
            support the label yet, and it changes as data arrives. ``label``,
            ``ret`` and ``t1`` are unchanged for compatibility; drop censored
            rows before training (the :mod:`panelary.weights` span table and
            ``FoldWeights`` exclude them by default).
    """
    if max_holding < 1:
        raise ValueError("max_holding must be a positive integer")

    frame = _as_dataframe(df)
    entity_col, time_col = _resolve_cols(frame, entity, time)
    frame = frame.sort([entity_col, time_col])

    labels: list[np.ndarray] = []
    rets: list[np.ndarray] = []
    t1s: list[pl.Series] = []
    cens: list[np.ndarray] = []

    for (_key,), sub in frame.group_by([entity_col], maintain_order=True):
        prices = sub.get_column(price).to_numpy().astype(np.float64)
        times = sub.get_column(time_col)
        sigma = (
            None
            if vol is None
            else sub.get_column(vol).cast(pl.Float64).fill_null(np.nan).to_numpy()
        )
        label, ret_out, t1_idx, censored = _triple_barrier_entity(
            prices, pt, sl, max_holding, vol_lookback, sigma
        )
        labels.append(label)
        rets.append(ret_out)
        t1s.append(times.gather(t1_idx))
        cens.append(censored)

    out = frame.with_columns(
        pl.Series("label", np.concatenate(labels), dtype=pl.Int64),
        pl.Series("ret", np.concatenate(rets), dtype=pl.Float64),
        pl.concat(t1s, rechunk=True).alias("t1"),
        pl.Series("censored", np.concatenate(cens), dtype=pl.Boolean),
    )
    return out


def fixed_horizon(
    df: PolarsFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    price: str = "close",
    horizon: int,
    threshold: float | None = None,
    allow_gaps: bool | None = None,
) -> pl.DataFrame:
    """Fixed-horizon forward-return label.

    Labels each observation by its forward return over a fixed number of steps
    ``horizon`` (within the entity, in time order). With ``threshold=None`` the
    label is the continuous forward return; otherwise it is the sign of the
    return outside a symmetric dead-band: ``+1`` if ``ret > threshold``, ``-1``
    if ``ret < -threshold``, else ``0``.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Panel with at least an entity, a time, and a ``price`` column.
    entity : str, optional
        Entity column. Defaults to the first column.
    time : str, optional
        Time column. Defaults to the second column.
    price : str, default "close"
        Price column used for the forward return.
    horizon : int
        Forward horizon, in number of steps (rows) per entity.
    threshold : float, optional
        Symmetric return threshold for the ternary sign label. If ``None`` the
        continuous forward return is returned in ``label``.
    allow_gaps : bool or None, default None
        The forward return comes from :func:`panelary.factor.forward_return`,
        the library's single audited negative-shift site, including its gap
        guard. On an irregular per-entity time grid (more than one distinct
        time step, e.g. a missing day -- or weekends on a business-day
        ``Date`` panel) a ``horizon``-row shift spans a different amount of
        calendar time on different rows.

        - ``None`` (default): on an irregular grid, emit a ``UserWarning`` and
          use the row-count horizon (the pre-0.6 behaviour, unchanged output).
        - ``False``: raise ``ValueError`` on an irregular grid.
        - ``True``: accept a row-count horizon silently.

        ``t1`` always records the true end time of every label, which is what
        the purge uses.

    Returns
    -------
    polars.DataFrame
        The input rows (sorted by ``[entity, time]``) with three columns added:

        ``label`` : Float64 or Int64
            Continuous forward return (``threshold=None``) or ``{-1, 0, +1}``.
        ``ret`` : Float64
            The forward return over ``horizon`` steps.
        ``t1`` : same dtype as ``time``
            Timestamp ``horizon`` steps ahead (``null`` at the tail where the
            full horizon is unavailable -- the unresolved, censored labels).

    Raises
    ------
    ValueError
        If ``horizon < 1``, or the time grid is irregular and ``allow_gaps``
        is ``False``.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")

    frame = _as_dataframe(df)
    entity_col, time_col = _resolve_cols(frame, entity, time)
    frame = frame.sort([entity_col, time_col])

    if allow_gaps is None:
        try:
            _check_no_gaps(frame, entity=entity_col, time=time_col)
        except ValueError:
            warnings.warn(
                "fixed_horizon: irregular per-entity time grid (e.g. weekends on "
                "a business-day Date panel, or a missing period), so a "
                f"horizon={horizon}-row label spans different amounts of time on "
                "different rows. Using the row-count horizon; `t1` records each "
                "label's true end. Pass allow_gaps=True to accept this silently, "
                "or allow_gaps=False to raise.",
                UserWarning,
                stacklevel=2,
            )
        allow_gaps = True

    aligned = forward_return(
        frame,
        entity=entity_col,
        time=time_col,
        price=price,
        horizon=horizon,
        out="ret",
        allow_gaps=allow_gaps,
        end_time="t1",
    )
    assert isinstance(aligned, pl.DataFrame)  # a DataFrame in, a DataFrame out
    frame = aligned

    if threshold is None:
        label_expr = pl.col("ret").alias("label")
    else:
        label_expr = (
            pl.when(pl.col("ret") > threshold)
            .then(1)
            .when(pl.col("ret") < -threshold)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("label")
        )

    return frame.with_columns(label_expr)


def meta_label(
    df: PolarsFrame,
    primary_signal: str,
    label: str,
) -> pl.DataFrame:
    """Meta-labeling: turn a primary side signal + realized label into act/pass.

    Meta-labeling (AFML Ch. 3.6) trains a secondary model to decide *whether to
    act* on a primary model that has already picked a *side*. The binary
    meta-label is ``1`` (act) when the primary side agrees with the realized
    outcome and the outcome is non-zero, and ``0`` (pass) otherwise -- i.e. the
    primary bet would have been correct.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Frame containing the ``primary_signal`` and ``label`` columns.
    primary_signal : str
        Column holding the primary model's side (its sign is used).
    label : str
        Column holding the realized label / outcome (e.g. from
        :func:`triple_barrier`; its sign is used).

    Returns
    -------
    polars.DataFrame
        The input rows with a ``meta_label`` column of ``{0, 1}`` (Int64).
    """
    frame = _as_dataframe(df)
    side = pl.col(primary_signal).sign()
    outcome = pl.col(label).sign()
    meta = (
        pl.when((outcome != 0) & (side == outcome))
        .then(1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("meta_label")
    )
    return frame.with_columns(meta)
