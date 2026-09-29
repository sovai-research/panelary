"""Causal per-row sampling expressions: the AFML CUSUM event filter and the tick rule.

Both are Polars expression factories meant to run within an entity, in time
order: ``pn.sample.cusum_filter(pl.col("ret"), threshold="h").over("id")``.
Each row's output depends only on that row and earlier rows, so both are
prefix-invariant: appending data never changes an earlier value.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from panelary.sample._kernels import _cusum_filter

__all__ = ["cusum_filter", "tick_rule"]


def _as_expr(x: pl.Expr | str) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def cusum_filter(
    expr: pl.Expr | str,
    threshold: float | str | pl.Expr,
) -> pl.Expr:
    """AFML's symmetric CUSUM filter: flag the rows where an event is sampled.

    With ``y_t`` the input (typically a return) and ``h_t`` the threshold::

        S+ = max(0, S+ + y_t),   S- = min(0, S- + y_t)
        if S- < -h_t: event, S- = 0
        elif S+ > h_t: event, S+ = 0

    Only the side that fired resets (AFML snippet 2.4). This is an **event
    sampler**, not a change-point monitor: it has no calibration and no
    warm-up standardisation -- for the SPC CUSUM with those, see
    ``pl.col(...).ts.cusum``; for a monitoring statistic see
    :func:`panelary.detect.page_cusum_expr`. Page's closed form does not apply
    here because the resets make each event depend on the previous one.

    Parameters
    ----------
    expr : polars.Expr or str
        The series to accumulate (e.g. a log return).
    threshold : float, str or polars.Expr
        A constant, or a column / expression of per-row thresholds. A
        per-row threshold must be **causal** (e.g. a trailing volatility
        multiple); a full-sample estimate makes every event depend on the
        future.

    Returns
    -------
    polars.Expr
        Boolean, ``True`` on event rows. Apply ``.over(entity)`` on a panel
        sorted by ``(entity, time)``.

    Notes
    -----
    A null / NaN ``y_t`` is skipped (state unchanged, no event). A null / NaN
    threshold fires nothing, but the sums keep accumulating. With the ``fast``
    extra the recursion runs in numba; otherwise a pure-Python loop gives
    identical events.

    Examples
    --------
    >>> events = df.with_columns(  # doctest: +SKIP
    ...     pn.sample.cusum_filter(pl.col("ret"), threshold="h").over("id").alias("event")
    ... )
    """
    y = _as_expr(expr).cast(pl.Float64)
    if not isinstance(threshold, (str, pl.Expr)):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(
                "`threshold` must be a number, a column name or an expression, "
                f"got {type(threshold).__name__}."
            )
        h_const = float(threshold)

        def _batch_const(s: pl.Series) -> pl.Series:
            vals = s.fill_null(np.nan).to_numpy()
            h = np.full(vals.shape[0], h_const, dtype=np.float64)
            return pl.Series(_cusum_filter(vals, h), dtype=pl.Boolean)

        return y.map_batches(_batch_const, return_dtype=pl.Boolean)

    h_expr = _as_expr(threshold).cast(pl.Float64)

    def _batch(s: pl.Series) -> pl.Series:
        vals = s.struct.field("y").fill_null(np.nan).to_numpy()
        h = s.struct.field("h").fill_null(np.nan).to_numpy()
        return pl.Series(_cusum_filter(vals, h), dtype=pl.Boolean)

    return pl.struct(y.alias("y"), h_expr.alias("h")).map_batches(
        _batch, return_dtype=pl.Boolean
    )


def tick_rule(expr: pl.Expr | str) -> pl.Expr:
    """Trade side by the tick rule: ``+1`` on an up-tick, ``-1`` on a down-tick.

    A zero tick carries the previous side forward (Lee and Ready, 1991). The
    head of each series is **null** until the first non-zero move: no side is
    fabricated (implementations that seed it with ``+1`` bias the first
    imbalance).

    Parameters
    ----------
    expr : polars.Expr or str
        Trade prices.

    Returns
    -------
    polars.Expr
        Int8 in ``{-1, +1}`` (null before the first move). Apply
        ``.over(entity)`` on a frame sorted by ``(entity, time)``.
    """
    d = _as_expr(expr).diff()
    return (
        pl.when(d > 0)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(d < 0)
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.lit(None, dtype=pl.Int8))
        .forward_fill()
    )
