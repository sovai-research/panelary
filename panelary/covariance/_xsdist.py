"""Cross-date features of the cross-sectional return distribution.

The single-date summaries (dispersion, Hill tail index, up-share, entropy) are
Polars expressions in the ``.xs`` namespace. The features here need more than
one date, so they are frame operations:

* :func:`kelly_jiang_tail` -- Kelly & Jiang (2014) tail risk: the Hill
  estimator on the pooled returns of the trailing ``window`` dates.
* :func:`kelly_jiang_beta` -- each name's trailing beta on the *lagged* tail
  risk, through the existing :func:`panelary.econ.features._common.rolling_beta`.
* :func:`xs_wasserstein` -- the 1-Wasserstein distance between consecutive
  cross-sections (Vallender 1973), exact in O(N log N) per date.
* :func:`avg_skewness` -- Jondeau, Zhang & Zhu (2019) average skewness: the
  cross-sectional mean of each name's trailing return skewness.

Each returns a per-date frame keyed by the panel's time column
(:func:`kelly_jiang_beta` returns panel rows); ``broadcast=True`` joins it onto
the panel rows. All values are "as of the close of t".

Leak safety
-----------
Every window trails ``t`` on the panel's time axis and every threshold is
estimated inside it: the Kelly-Jiang threshold is the pooled quantile of the
trailing ``window`` dates, **not** of the calendar month containing ``t`` (plan
trap T10: the paper's monthly design, reused daily, reads the rest of the
month). Per-date reductions run over each date's compacted members (see
:mod:`panelary.covariance._xsdense`), so values are bit-identical when later
dates or later-listing names are appended.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.covariance._xsdense import (
    broadcast_rows,
    coerce_panel,
    coverage_count,
    dense_matrix,
    segment_residuals,
    segment_sums,
    validate_coverage,
    validate_int,
)
from panelary.econ.features._common import rolling_beta
from panelary.registry import FeatureSpec, registry

__all__ = [
    "avg_skewness",
    "kelly_jiang_beta",
    "kelly_jiang_tail",
    "xs_wasserstein",
]

_LICENSE = "Apache-2.0"
_EPS = float(np.finfo(np.float64).eps)

#: Elements per chunk of the vectorised Wasserstein merge. Each element costs
#: about 150 bytes of int64/float64 scratch across the pass, so the peak is
#: bounded near 80 MB whatever the panel size.
_W1_CHUNK = 1 << 19


def _validate_q(q: Any) -> float:
    value = float(q)
    if not 0.0 < value < 1.0:
        raise ValueError(f"`q` must lie in (0, 1), got {q!r}.")
    return value


def _nullify(values: NDArray[np.float64]) -> pl.Series:
    return pl.Series(values, dtype=pl.Float64).fill_nan(None)


def _date_segments(
    pf: Any, value: str, *, observed_only: bool = True
) -> tuple[pl.Series, NDArray[np.float64], NDArray[np.int64], pl.DataFrame]:
    """The panel's time axis and its finite values sorted within each date.

    Returns ``(times, values, counts, frame)`` where ``values`` holds each
    date's finite values in ascending order, dates in time order, ``counts[t]``
    of them for date ``t`` of ``times`` (the panel's sorted unique times,
    including dates with no finite value).
    """
    tim = pf.time_col
    if value not in pf.columns:
        raise ValueError(
            f"column {value!r} not found in the panel; available: {pf.columns}."
        )
    frame = pf.lazy().select([tim, pl.col(value).cast(pl.Float64)]).collect()
    times = frame.get_column(tim).unique().sort()
    finite = frame.filter(pl.col(value).is_finite())
    t_idx = times.search_sorted(finite.get_column(tim)).cast(pl.Int64)
    ordered = pl.DataFrame({"t": t_idx, "v": finite.get_column(value)}).sort(["t", "v"])
    counts = np.bincount(ordered.get_column("t").to_numpy(), minlength=times.len())
    values = np.ascontiguousarray(ordered.get_column("v").to_numpy(), dtype=np.float64)
    return times, values, counts.astype(np.int64), frame


# --------------------------------------------------------------------------- #
# Kelly-Jiang tail risk
# --------------------------------------------------------------------------- #
def kelly_jiang_tail(
    panel: Any,
    *,
    returns: str,
    window: int = 21,
    q: float = 0.05,
    min_exceedances: int = 10,
    residual: bool = False,
    n_factors: int = 1,
    refit_every: int = 21,
    fit_window: int = 252,
    min_coverage: float = 0.95,
    entity: str | None = None,
    time: str | None = None,
    broadcast: bool = False,
) -> pl.DataFrame:
    """Kelly & Jiang (2014) cross-sectional tail risk, per date.

    Pool the finite returns of every name over the trailing ``window`` dates
    ``(t - window, t]``: ``n`` values. With ``k = floor(q * n)``, the threshold
    ``u_t`` is the ``(k + 1)``-th smallest pooled return and::

        lambda_t = mean( log(R / u_t) )   over the k smallest pooled returns

    Kelly and Jiang's ``lambda`` is the Hill (1975) estimator in its ``xi``
    convention (larger means a heavier tail), the reciprocal of
    :func:`panelary.econ.features._evt.hill_index`'s ``alpha``: here
    ``lambda_t == 1 / hill_index(pooled, k=k, tail="lower")`` exactly. The
    threshold is an order statistic of the trailing window -- never of the
    calendar month containing ``t``, which is the paper's monthly design and a
    look-ahead when reused on daily data. One ``np.partition`` of ``n`` values
    per date, O(window * N).

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel of (daily) returns.
    returns : str
        Return column; non-finite values are unobserved.
    window : int, default 21
        Trailing dates pooled (Kelly and Jiang pool one month).
    q : float, default 0.05
        Tail fraction; Kelly and Jiang use the 5th percentile.
    min_exceedances : int, default 10
        Fewer exceedances than this gives null.
    residual : bool, default False
        Pool statistical-factor residuals instead of raw returns (Kelly and Jiang
        use Fama-French residuals): ``n_factors`` PCA factors are removed by
        :func:`panelary.detect._panel.residualise` with loadings fit on the
        trailing ``fit_window`` dates and refit every ``refit_every`` dates, as
        in :func:`~panelary.covariance.common_idio_vol`. Residuals start at date
        index ``fit_window``.
    n_factors, refit_every, fit_window, min_coverage
        The residual model; ignored unless ``residual=True``.
    entity, time : str, optional
        Key columns, used only when ``panel`` is a bare frame.
    broadcast : bool, default False
        Left-join the result onto the panel rows.

    Returns
    -------
    polars.DataFrame
        ``time, kj_tail, kj_threshold, kj_n_exceed, kj_n_obs``. ``kj_tail`` and
        ``kj_threshold`` are null for the first ``window - 1`` dates, where
        ``k < min_exceedances``, where the threshold is not negative (the
        logarithm needs losses) or where every exceedance ties it.
    """
    pf = coerce_panel(panel, entity, time)
    window = validate_int("window", window, minimum=1)
    q = _validate_q(q)
    min_exceedances = validate_int("min_exceedances", min_exceedances, minimum=1)
    dense = dense_matrix(pf, returns)
    values = dense.values
    if residual:
        coverage = validate_coverage(min_coverage)
        values = segment_residuals(
            values,
            n_factors=validate_int("n_factors", n_factors, minimum=0),
            refit_every=validate_int("refit_every", refit_every, minimum=1),
            fit_window=validate_int("fit_window", fit_window, minimum=2),
            min_obs=coverage_count(fit_window, coverage),
        )
    n_time = values.shape[0]
    tail = np.full(n_time, np.nan, dtype=np.float64)
    threshold = np.full(n_time, np.nan, dtype=np.float64)
    n_exceed = np.zeros(n_time, dtype=np.int64)
    n_obs = np.zeros(n_time, dtype=np.int64)
    for t in range(window - 1, n_time):
        block = values[t - window + 1 : t + 1]
        pooled = block[np.isfinite(block)]
        n = pooled.size
        n_obs[t] = n
        k = math.floor(q * n)
        n_exceed[t] = k
        if k < min_exceedances:
            continue
        part = np.partition(pooled, k)
        u = float(part[k])
        if not u < 0.0:
            continue
        xi = float(np.mean(np.log(-part[:k]) - math.log(-u)))
        if xi > 0.0:
            tail[t] = xi
            threshold[t] = u
    state = pl.DataFrame(
        {
            dense.time_col: dense.times,
            "kj_tail": _nullify(tail),
            "kj_threshold": _nullify(threshold),
            "kj_n_exceed": pl.Series(n_exceed, dtype=pl.Int64),
            "kj_n_obs": pl.Series(n_obs, dtype=pl.Int64),
        }
    )
    return broadcast_rows(pf, state, [dense.time_col]) if broadcast else state


def kelly_jiang_beta(
    panel: Any,
    *,
    returns: str,
    window: int = 252,
    min_periods: int | None = None,
    tail_window: int = 21,
    q: float = 0.05,
    min_exceedances: int = 10,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Each name's trailing beta on lagged Kelly-Jiang tail risk.

    ``kj_beta`` is the slope of ``r_{i,t}`` on ``lambda_{t-1}`` (the tail risk
    of :func:`kelly_jiang_tail`, lagged one date on the panel's time axis) over
    the name's trailing ``window`` rows -- Kelly and Jiang's tail-risk beta in
    daily form. It is computed by the existing
    :func:`panelary.econ.features._common.rolling_beta`, reused unchanged.

    ``rolling_beta`` uses the one-pass moment formula
    ``cov = E[xy] - E[x] E[y]``, which cancels badly when ``x`` sits far from
    zero relative to its spread -- and ``lambda`` does (a level near 0.4 moving
    by a few hundredths). The regressor is therefore shifted by a constant
    first, the first defined ``lambda_{t-1}`` on the time axis (a causal
    reference: it never changes once set). A slope is shift-invariant, so this
    changes only the rounding. ``rolling_beta`` also takes its three rolling
    means over each column's own non-null rows; both inputs are nulled wherever
    either is missing, so every moment uses the same rows.

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel of returns.
    returns : str
        Return column.
    window : int, default 252
        Trailing rows of each name in the regression.
    min_periods : int, optional
        Fewest complete rows for a beta; default ``window``.
    tail_window, q, min_exceedances
        Passed to :func:`kelly_jiang_tail` as ``window``, ``q`` and
        ``min_exceedances``.
    entity, time : str, optional
        Key columns, used only when ``panel`` is a bare frame.

    Returns
    -------
    polars.DataFrame
        The panel rows, sorted by ``(entity, time)``, with ``kj_tail_lag``
        (``lambda_{t-1}``, unshifted) and ``kj_beta``.
    """
    pf = coerce_panel(panel, entity, time)
    window = validate_int("window", window, minimum=2)
    if min_periods is not None:
        min_periods = validate_int("min_periods", min_periods, minimum=2)
    ent, tim = pf.entity_col, pf.time_col
    clash = sorted({"kj_tail_lag", "kj_beta"} & set(pf.columns))
    if clash:
        raise ValueError(f"kelly_jiang_beta would overwrite panel column(s) {clash}.")
    tail = kelly_jiang_tail(
        pf,
        returns=returns,
        window=tail_window,
        q=q,
        min_exceedances=min_exceedances,
    )
    lagged = tail.select(tim, pl.col("kj_tail").shift(1).alias("kj_tail_lag"))
    defined = lagged.get_column("kj_tail_lag").drop_nulls()
    reference = float(defined[0]) if defined.len() else 0.0
    y_col, x_col = "__panelary_kj_y__", "__panelary_kj_x__"
    y = pl.col(returns).cast(pl.Float64).fill_nan(None)
    x = pl.col("kj_tail_lag") - reference
    frame = (
        pf.collect()
        .join(lagged, on=tim, how="left", maintain_order="left")
        .with_columns(
            pl.when(x.is_not_null()).then(y).alias(y_col),
            pl.when(y.is_not_null()).then(x).alias(x_col),
        )
    )
    out = rolling_beta(
        frame,
        entity=ent,
        time=tim,
        y=y_col,
        x=x_col,
        window=window,
        min_periods=min_periods,
        alias="kj_beta",
    )
    return out.drop([y_col, x_col])


# --------------------------------------------------------------------------- #
# Wasserstein distance between consecutive cross-sections
# --------------------------------------------------------------------------- #
def _w1_pairs(
    values: NDArray[np.float64],
    offsets: NDArray[np.int64],
    counts: NDArray[np.int64],
    dates: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Exact W1 between date ``t`` and ``t - 1`` for each ``t`` in ``dates``.

    With ``n`` names on ``t`` (sorted ``a``) and ``m`` on ``t - 1`` (sorted
    ``b``), ``W1 = int_0^1 |Q_a(u) - Q_b(u)| du`` for the two empirical quantile
    functions. In units of ``1 / (n m)`` the breakpoints of ``Q_a`` are the
    multiples of ``m`` and those of ``Q_b`` the multiples of ``n``; their merged
    order is known in closed form (the ``i``-th multiple of ``m`` is preceded by
    ``i + ceil(i m / n)`` points, the ``j``-th multiple of ``n`` by
    ``j + floor(j n / m) + 1``), so the merge is a scatter, not a sort. On the
    interval starting at merged point ``x``, ``Q_a = a[x // m]`` and
    ``Q_b = b[x // n]``. Integer arithmetic throughout, one ``reduceat`` per
    pair: every pair's value depends only on its own two samples.
    """
    n = counts[dates]
    m = counts[dates - 1]
    size = n + m
    n_pairs = dates.size
    seg = np.cumsum(size) - size
    total = int(size.sum())
    pair = np.arange(n_pairs)

    rep_a = np.repeat(pair, n)
    i = np.arange(rep_a.size) - np.repeat(np.cumsum(n) - n, n)
    n_a, m_a = n[rep_a], m[rep_a]
    pos_a = seg[rep_a] + i + (i * m_a + n_a - 1) // n_a
    rep_b = np.repeat(pair, m)
    j = np.arange(rep_b.size) - np.repeat(np.cumsum(m) - m, m)
    n_b, m_b = n[rep_b], m[rep_b]
    pos_b = seg[rep_b] + j + (j * n_b) // m_b + 1

    point = np.empty(total, dtype=np.int64)
    point[pos_a] = i * m_a
    point[pos_b] = j * n_b
    span = n * m
    nxt = np.empty(total, dtype=np.int64)
    nxt[:-1] = point[1:]
    nxt[seg + size - 1] = span
    owner = np.repeat(pair, size)
    a = values[offsets[dates][owner] + point // m[owner]]
    b = values[offsets[dates - 1][owner] + point // n[owner]]
    contrib = (nxt - point).astype(np.float64) * np.abs(a - b)
    result: NDArray[np.float64] = np.add.reduceat(contrib, seg) / span.astype(
        np.float64
    )
    return result


def _w1_exact(
    values: NDArray[np.float64],
    counts: NDArray[np.int64],
    valid: NDArray[np.bool_],
) -> NDArray[np.float64]:
    n_time = counts.size
    out = np.full(n_time, np.nan, dtype=np.float64)
    if n_time < 2:
        return out
    offsets = np.cumsum(counts) - counts
    dates = np.flatnonzero(valid[1:] & valid[:-1]) + 1
    if dates.size == 0:
        return out
    cumulative = np.cumsum(counts[dates] + counts[dates - 1])
    start = 0
    while start < dates.size:
        base = cumulative[start - 1] if start else 0
        stop = max(
            int(np.searchsorted(cumulative, base + _W1_CHUNK, "right")), start + 1
        )
        chunk = dates[start:stop]
        out[chunk] = _w1_pairs(values, offsets, counts, chunk)
        start = stop
    return out


def _w1_grid(
    values: NDArray[np.float64],
    counts: NDArray[np.int64],
    valid: NDArray[np.bool_],
    grid: int,
) -> NDArray[np.float64]:
    """Midpoint-rule W1 on ``grid`` quantile levels ``(k + 1/2) / grid``.

    ``Q(u) = x_(ceil(u n))`` is the same empirical quantile function as the
    exact path, so this converges to it as ``grid`` grows.
    """
    n_time = counts.size
    out = np.full(n_time, np.nan, dtype=np.float64)
    if n_time < 2:
        return out
    offsets = np.cumsum(counts) - counts
    quant = np.full((n_time, grid), np.nan, dtype=np.float64)
    have = np.flatnonzero(valid)
    if have.size:
        level = 2 * np.arange(grid, dtype=np.int64) + 1
        n = counts[have][:, None]
        idx = (level[None, :] * n + 2 * grid - 1) // (2 * grid) - 1
        quant[have] = values[offsets[have][:, None] + idx]
    pairs = valid[1:] & valid[:-1]
    diff = np.abs(quant[1:] - quant[:-1]).mean(axis=1)
    out[1:] = np.where(pairs, diff, np.nan)
    return out


def xs_wasserstein(
    panel: Any,
    *,
    value: str,
    standardize: bool = False,
    grid: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    broadcast: bool = False,
) -> pl.DataFrame:
    """1-Wasserstein distance between each date's cross-section and the previous one.

    ``W1_t = int |F_t(x) - F_{t-1}(x)| dx`` for the empirical distributions of
    ``value`` on date ``t`` and on the previous date of the panel's time axis
    (Vallender 1973): how far the cross-section's distribution moved overnight.
    The default is exact, O(N log N) per date (a sort, then a closed-form merge
    of the two quantile grids -- no second sort), and agrees with
    ``scipy.stats.wasserstein_distance`` to rounding.

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel.
    value : str
        Column whose cross-sectional distribution is compared; non-finite
        values are unobserved.
    standardize : bool, default False
        Compare same-date z-scores (population sd), isolating a change of shape
        from a change of location and scale. A date whose values are all equal
        has no z-scores, so both of its pairs are null.
    grid : int, optional
        Approximate with the midpoint rule on ``grid`` quantile levels instead:
        ``mean_k |Q_t(u_k) - Q_{t-1}(u_k)|``, ``u_k = (k + 1/2) / grid``.
    entity, time : str, optional
        Key columns, used only when ``panel`` is a bare frame.
    broadcast : bool, default False
        Left-join the result onto the panel rows.

    Returns
    -------
    polars.DataFrame
        ``time, xs_w1``; null on the first date and wherever either date has no
        finite value.
    """
    pf = coerce_panel(panel, entity, time)
    if grid is not None:
        grid = validate_int("grid", grid, minimum=1)
    times, values, counts, _frame = _date_segments(pf, value)
    valid = counts > 0
    if standardize:
        safe = np.maximum(counts, 1).astype(np.float64)
        mean = segment_sums(values, counts) / safe
        dev = values - np.repeat(mean, counts)
        sd = np.sqrt(segment_sums(dev * dev, counts) / safe)
        # A constant date centres to a few ulps, not zeros: it has no z-scores.
        valid &= sd > safe * _EPS * np.abs(mean)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = dev / np.repeat(sd, counts)
    if grid is None:
        w1 = _w1_exact(values, counts, valid)
    else:
        w1 = _w1_grid(values, counts, valid, grid)
    state = pl.DataFrame({pf.time_col: times, "xs_w1": _nullify(w1)})
    return broadcast_rows(pf, state, [pf.time_col]) if broadcast else state


# --------------------------------------------------------------------------- #
# Average skewness
# --------------------------------------------------------------------------- #
def avg_skewness(
    panel: Any,
    *,
    returns: str,
    window: int = 21,
    min_periods: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    broadcast: bool = False,
) -> pl.DataFrame:
    """Average skewness (Jondeau, Zhang & Zhu 2019), per date.

    ``pl.col(returns).rolling_skew(window).over(entity)`` -- each name's
    skewness over its own trailing ``window`` rows, Polars' native (biased,
    ``g1``) estimator -- then its equal-weight mean over the names observed at
    ``t`` with a defined value. The window counts the name's own rows, so a
    gap in its history widens it in calendar terms.

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel of returns.
    returns : str
        Return column; NaN is treated as missing.
    window : int, default 21
        Trailing rows per name (JZZ use one month of daily returns).
    min_periods : int, optional
        Fewest observed rows for a skewness; default ``window``.
    entity, time : str, optional
        Key columns, used only when ``panel`` is a bare frame.
    broadcast : bool, default False
        Left-join the result onto the panel rows.

    Returns
    -------
    polars.DataFrame
        ``time, avg_skew, n_entities``.
    """
    pf = coerce_panel(panel, entity, time)
    window = validate_int("window", window, minimum=3)
    mp = (
        window
        if min_periods is None
        else validate_int("min_periods", min_periods, minimum=3)
    )
    if mp > window:
        raise ValueError(f"`min_periods` ({mp}) must not exceed `window` ({window}).")
    ent, tim = pf.entity_col, pf.time_col
    if returns not in pf.columns:
        raise ValueError(
            f"column {returns!r} not found in the panel; available: {pf.columns}."
        )
    r = pl.col(returns).cast(pl.Float64).fill_nan(None)
    frame = (
        pf.lazy()
        .select([ent, tim, r.alias("__r")])
        .sort([ent, tim])
        .with_columns(
            pl.col("__r").rolling_skew(window, min_samples=mp).over(ent).alias("__skew")
        )
        .collect()
    )
    times = frame.get_column(tim).unique().sort()
    rows = frame.filter(
        pl.col("__r").is_not_null() & pl.col("__skew").is_finite()
    ).sort([tim, ent])
    t_idx = times.search_sorted(rows.get_column(tim)).cast(pl.Int64).to_numpy()
    counts = np.bincount(t_idx, minlength=times.len()).astype(np.int64)
    sums = segment_sums(rows.get_column("__skew").to_numpy(), counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    state = pl.DataFrame(
        {
            tim: times,
            "avg_skew": _nullify(avg),
            "n_entities": pl.Series(counts, dtype=pl.Int64),
        }
    )
    return broadcast_rows(pf, state, [tim]) if broadcast else state


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Frame -> frame, mixing entities by design (panel_safe=False). Every window
# trails t and every threshold is estimated inside it, so the per-row claim is
# "rowwise", verified by tests/test_registry_conformance.py.
_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="kelly_jiang_tail",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "q": float,
            "min_exceedances": int,
            "residual": bool,
            "broadcast": bool,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Kelly & Jiang (2014), Review of Financial Studies 27(10)",
        license=_LICENSE,
        backend_fn=kelly_jiang_tail,
        cost_hint="O(T W N) (one partition per date)",
    ),
    FeatureSpec(
        name="kelly_jiang_beta",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "min_periods": object,
            "tail_window": int,
            "q": float,
            "min_exceedances": int,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Kelly & Jiang (2014), Review of Financial Studies 27(10)",
        license=_LICENSE,
        backend_fn=kelly_jiang_beta,
        cost_hint="kelly_jiang_tail + O(T N) rolling moments",
    ),
    FeatureSpec(
        name="xs_wasserstein",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "value": str,
            "standardize": bool,
            "grid": object,
            "broadcast": bool,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Vallender (1973), Theory of Probability and Its Applications 18(4)",
        license=_LICENSE,
        backend_fn=xs_wasserstein,
        cost_hint="O(T N log N)",
    ),
    FeatureSpec(
        name="avg_skewness",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "min_periods": object,
            "broadcast": bool,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Jondeau, Zhang & Zhu (2019), Journal of Financial Economics 134(1)",
        license=_LICENSE,
        backend_fn=avg_skewness,
        cost_hint="O(T N) rolling moments",
    ),
)

for _spec in _SPECS:
    registry.register(_spec)
