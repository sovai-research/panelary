"""Average correlation and common idiosyncratic volatility, as of each date.

Two per-date measures of how much the cross-section moves together, both
computed from a trailing window that ends at (and includes) date ``t``:

* :func:`avg_correlation` -- the average pairwise correlation of the names in
  the date-``t`` universe over the trailing ``window``. ``kind="equal"`` is the
  plain mean of the off-diagonal correlations; ``kind="pollet_wilson"`` is the
  ``sigma_i sigma_j``-weighted mean of Pollet & Wilson (2010), which has an
  O(N)-per-date shortcut through the variance of the equal-weight portfolio.
* :func:`common_idio_vol` -- Herskovic, Kelly, Lustig & Van Nieuwerburgh
  (2016): the cross-sectional mean of each name's trailing idiosyncratic
  volatility, with statistical (PCA) factors in place of their Fama-French
  factors and loadings frozen on a trailing, index-anchored refit schedule.

Both return a per-date frame keyed by the panel's time column (and the group
column for a group variant); ``broadcast=True`` joins it onto the panel rows.
Values are "as of the close of t": the window includes ``r_t``, so a decision
taken at the open of ``t`` must lag them one date (plan trap T16).

Leak safety
-----------
Every quantity is window-local -- the universe, its size, the means, the
scales, the refit dates -- so nothing depends on the panel's length, its total
entity count or its last date. The date-``t`` universe is compacted before any
reduction (see :mod:`panelary.covariance._xsdense`), so the value at ``t`` is
bit-identical whether or not later dates, or later-listing names, are present:
``tests/test_covariance_comovement.py`` checks that with ``tol=0.0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from panelary.covariance._xsdense import (
    Dense,
    broadcast_rows,
    coerce_panel,
    coverage_count,
    dense_matrix,
    long_from_dense,
    masked_row_sums,
    segment_residuals,
    segment_sums,
    validate_coverage,
    validate_int,
    window_counts,
)
from panelary.registry import FeatureSpec, registry

__all__ = ["avg_correlation", "common_idio_vol"]

_KINDS = ("equal", "pollet_wilson")
_LICENSE = "Apache-2.0"


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
_EPS = float(np.finfo(np.float64).eps)


def _has_variance(
    sumsq: NDArray[np.float64],
    counts: NDArray[Any] | int,
    mean: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Which names have a sample variance distinguishable from rounding.

    A constant window does not centre to exact zeros: the mean of ``n`` equal
    values is off by a few ulps, so its "variance" is a positive number of order
    ``(n eps |x|)^2`` and its z-scores are noise of order one. A name counts as
    varying only if ``sum d^2 > n (n eps |mean|)^2`` -- far below any real
    return series, far above that rounding floor.
    """
    floor = counts * (counts * _EPS * np.abs(mean)) ** 2
    keep: NDArray[np.bool_] = sumsq > floor
    return keep


def _window_kernel(
    block: NDArray[np.float64],
    kind: str,
    codes: NDArray[np.int64] | None = None,
    *,
    overwrite: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Average correlation of one compacted window, pooled or per group.

    ``block`` is one date's window (rows = dates, columns = that date's
    universe). Two-pass centring per name (no raw-sum cancellation), zero after
    demeaning for the few missing cells a coverage threshold below 1 admits,
    and the scale ``sigma_i = sqrt(sum_obs d^2 / (n_i - 1))``. With ``D`` the
    centred window, ``n_eff = rows - 1`` and, per group ``g`` of ``N_g`` names:

    * ``equal``: ``Z = D / sigma`` and ``rho_g = (||Z 1_g||^2 - sum_{i in g}
      ||z_i||^2) / (n_eff N_g (N_g - 1))`` -- the mean off-diagonal entry of
      ``Z'Z / n_eff`` within the group, from one pass over the window instead
      of an ``N x N`` matrix. On a complete window ``||z_i||^2 = rows - 1``
      and this is ``rho = (||Z 1||^2 / n_eff - N) / (N (N - 1))``.
    * ``pollet_wilson``: ``sum_{i != j} cov_ij / sum_{i != j} sigma_i sigma_j``
      with ``sum_{i != j} cov_ij = (||D 1_g||^2 - sum_i ||d_i||^2) / n_eff``.

    Every group's ``Z 1_g`` comes from one product ``D @ M``, ``M[i, g]`` the
    weight of name ``i`` in its group ``g``: a date's moments are computed once
    however many groups it has. All shapes are those of the date's own
    universe, so the arithmetic does not change when later data arrives.

    Names with no variance in the window have no correlation and are dropped
    (see :func:`_has_variance`). ``codes`` gives each column's group (``None``:
    one group). Returns ``(rho, n, labels)`` per group present in ``codes``:
    ``rho`` is NaN where fewer than two names with a variance remain, and
    ``n`` counts those names. ``overwrite=True`` lets ``block`` (the caller's
    own compacted copy) be centred in place.
    """
    rows, width = block.shape
    if codes is None:
        labels = np.zeros(1, dtype=np.int64)
        gid_all = np.zeros(width, dtype=np.int64)
    else:
        labels, gid_all = np.unique(codes, return_inverse=True)
    n_groups = labels.size
    finite = np.isfinite(block)
    if finite.all():
        counts = np.full(width, rows, dtype=np.int64)
        mean = block.mean(axis=0)
        if overwrite:
            dev = block
            dev -= mean
        else:
            dev = block - mean
    else:
        counts = finite.sum(axis=0)
        missing = ~finite
        dev = block if overwrite else block.copy()
        np.copyto(dev, 0.0, where=missing)
        mean = dev.sum(axis=0) / np.maximum(counts, 1)
        dev -= mean
        np.copyto(dev, 0.0, where=missing)  # zero after demeaning
    sumsq = np.einsum("ij,ij->j", dev, dev)
    keep = _has_variance(sumsq, counts, mean) & (counts >= 2)
    if not keep.all():
        dev, sumsq, counts = dev[:, keep], sumsq[keep], counts[keep]
    gid = gid_all[keep]
    n_g = np.bincount(gid, minlength=n_groups).astype(np.int64)
    rho = np.full(n_groups, np.nan, dtype=np.float64)
    if rows < 2 or not (n_g >= 2).any():
        return rho, n_g, labels
    var = sumsq / (counts - 1)
    sd = np.sqrt(var)
    weights = 1.0 / sd if kind == "equal" else np.ones_like(sd)
    if n_groups == 1:
        row_sums = (dev @ weights)[:, None]
    else:
        member_weight = np.zeros((gid.size, n_groups), dtype=np.float64)
        member_weight[np.arange(gid.size), gid] = weights
        row_sums = dev @ member_weight
    sq = np.einsum("ij,ij->j", row_sums, row_sums)
    n_eff = float(rows - 1)
    ng = n_g.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "equal":
            diag = np.bincount(gid, weights=counts - 1.0, minlength=n_groups)
            value = (sq - diag) / (n_eff * ng * (ng - 1.0))
        else:
            own = np.bincount(gid, weights=sumsq, minlength=n_groups)
            sd_sum = np.bincount(gid, weights=sd, minlength=n_groups)
            var_sum = np.bincount(gid, weights=var, minlength=n_groups)
            value = ((sq - own) / n_eff) / (sd_sum * sd_sum - var_sum)
    ok = n_g >= 2
    rho[ok] = value[ok]
    return rho, n_g, labels


def _rolling_columns(
    matrix: NDArray[np.float64],
    window: int,
    min_samples: int,
    *,
    stats: tuple[str, ...],
) -> dict[str, NDArray[np.float64]]:
    """Trailing per-column moments over the time axis, ``ddof=1``.

    Polars' native rolling ``mean`` / ``var`` / ``std``, one column per name,
    NaN treated as missing (``min_samples`` counts the observed rows). Each
    column's value at ``t`` is a function of that column's rows ``<= t`` only.
    """
    frame = pl.from_numpy(matrix, orient="row")
    exprs = []
    for stat in stats:
        for name in frame.columns:
            col = pl.col(name).fill_nan(None)
            if stat == "mean":
                expr = col.rolling_mean(window, min_samples=min_samples)
            elif stat == "var":
                expr = col.rolling_var(window, min_samples=min_samples)
            else:
                expr = col.rolling_std(window, min_samples=min_samples)
            exprs.append(expr.alias(f"{stat}_{name}"))
    out = frame.select(exprs).to_numpy()
    width = matrix.shape[1]
    return {
        stat: np.ascontiguousarray(
            out[:, k * width : (k + 1) * width], dtype=np.float64
        )
        for k, stat in enumerate(stats)
    }


def _pollet_wilson_fast(
    values: NDArray[np.float64], window: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int64]]:
    """The O(N)-per-date Pollet-Wilson path, and where it is exact.

    ``rho_PW = (N^2 var(p) - sum var_i) / ((sum sd_i)^2 - sum var_i)`` with
    ``p`` the equal-weight portfolio of the names observed on each date and all
    moments trailing ``window``-row sample moments (``ddof=1``). That equals the
    kernel's ``sum_{i != j} cov_ij / sum_{i != j} sd_i sd_j`` exactly when the
    set of names observed is the **same on every date of the window** (and every
    one of them has a positive variance): then ``p`` is the portfolio of exactly
    that universe. A date is flagged ``stable`` only when that holds, which is
    checked from integer counts: the names observed throughout the window number
    ``n_full(t)``, and no date of the window has more observed names than that.

    Returns ``(rho, stable, n_full)``; ``rho`` is meaningful only where
    ``stable``.
    """
    n_time, _n_ent = values.shape
    finite = np.isfinite(values)
    n_obs = finite.sum(axis=1).astype(np.int64)
    full = window_counts(finite, window) == window
    n_full = full.sum(axis=1).astype(np.int64)
    rho = np.full(n_time, np.nan, dtype=np.float64)
    stable = np.zeros(n_time, dtype=bool)
    if n_time < window:
        return rho, stable, n_full
    most = sliding_window_view(n_obs, window).max(axis=1)
    stable[window - 1 :] = (most == n_full[window - 1 :]) & (n_full[window - 1 :] >= 2)
    if not stable.any():
        return rho, stable, n_full

    moments = _rolling_columns(values, window, window, stats=("var", "mean"))
    var = moments["var"]
    member = full & stable[:, None]
    # A name without variance has no correlation; the kernel drops it, so such
    # a date is not on the fast path.
    with np.errstate(invalid="ignore"):
        varies = _has_variance(var * (window - 1), window, moments["mean"])
    stable &= ~(member & ~varies).any(axis=1)
    member = full & stable[:, None]
    with np.errstate(invalid="ignore"):  # non-members may round below zero
        sd = np.sqrt(var)
    sd_sum, n_mem = masked_row_sums(sd, member)
    var_sum, _ = masked_row_sums(var, member)

    port_sum = segment_sums(values[finite], n_obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.where(n_obs > 0, port_sum / np.maximum(n_obs, 1), np.nan)
    var_p = (
        pl.Series(port)
        .fill_nan(None)
        .rolling_var(window, min_samples=window)
        .to_numpy()
        .astype(np.float64)
    )
    n_f = n_mem.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        fast = (n_f * n_f * var_p - var_sum) / (sd_sum * sd_sum - var_sum)
    rho[stable] = fast[stable]
    return rho, stable, n_full


def _nullify(values: NDArray[np.float64]) -> pl.Series:
    """A Float64 series with NaN turned into null."""
    return pl.Series(values, dtype=pl.Float64).fill_nan(None)


def _group_default(min_entities: int | None, group: str | None, pooled: int) -> int:
    if min_entities is None:
        return 10 if group is not None else pooled
    return validate_int("min_entities", min_entities, minimum=2)


# --------------------------------------------------------------------------- #
# Average correlation
# --------------------------------------------------------------------------- #
def avg_correlation(
    panel: Any,
    *,
    returns: str,
    window: int = 63,
    kind: str = "equal",
    min_coverage: float = 0.95,
    group: str | None = None,
    min_entities: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    broadcast: bool = False,
) -> pl.DataFrame:
    """Average pairwise correlation of the as-of universe, per date.

    For each date ``t`` the universe is the names observed at ``t`` with at
    least ``ceil(min_coverage * window)`` observations in the trailing window
    ``(t - window, t]`` (and a non-zero variance in it). Its window is centred
    per name (two passes, never raw sums), missing cells are zero after
    demeaning, and the average correlation comes from one pass over the
    compacted ``window x N_t`` block, O(W N) per date -- never an ``N x N``
    matrix.

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel of returns, one row per ``(entity, time)``.
    returns : str
        Return column. Non-finite values are treated as unobserved.
    window : int, default 63
        Trailing window length, in dates of the panel's time axis.
    kind : {"equal", "pollet_wilson"}, default "equal"
        ``"equal"``: the plain mean of the off-diagonal correlations.
        ``"pollet_wilson"``: the ``sigma_i sigma_j``-weighted mean of Pollet &
        Wilson (2010), ``(sigma_p^2 - sum w_i^2 sigma_i^2) / ((sum w_i
        sigma_i)^2 - sum w_i^2 sigma_i^2)`` with equal weights. It is evaluated
        in O(N) per date from the equal-weight portfolio's variance wherever the
        observed universe is constant across the window (``universe_stable``),
        and recomputed on the O(W N) kernel wherever it is not -- the shortcut is
        exact only for a constant universe. It weights high-volatility names
        more, so it approximates, but does not equal, ``"equal"``.
    min_coverage : float, default 0.95
        Minimum share of the window a name must be observed in to enter the
        universe. Zero-filling the rest attenuates a correlation by roughly
        ``sqrt(f_i f_j)``, so 0.95 bounds that at 5%.
    group : str, optional
        Group column (e.g. sector). Date ``t``'s universe is partitioned by each
        name's label **on date t**; its label history inside the window is
        ignored. One row per ``(time, group)``, computed on the kernel path for
        both kinds.
    min_entities : int, optional
        Fewest names a (group's) universe needs for a value; default 2, or 10
        with ``group``.
    entity, time : str, optional
        Key columns, used only when ``panel`` is a bare frame.
    broadcast : bool, default False
        Left-join the result onto the panel rows (on time, and group).

    Returns
    -------
    polars.DataFrame
        ``time [, group], avg_corr, n_entities`` -- plus ``universe_stable`` for
        the pooled ``"pollet_wilson"`` kind -- or the panel with those columns
        when ``broadcast=True``. ``n_entities`` counts the (group's) universe
        names with a variance in the window, i.e. the names averaged over;
        ``avg_corr`` is null where it is below ``min_entities``. A group
        appears on a date when at least one of its names is in that date's
        universe.

    Notes
    -----
    The value at ``t`` uses rows ``(t - window, t]`` of names that qualify at
    ``t``; a name listed after ``t`` cannot enter it, and a delisted one leaves
    after its last observation. Everything is recomputed per date, so no state
    is carried from one date to the next.
    """
    pf = coerce_panel(panel, entity, time)
    window = validate_int("window", window, minimum=2)
    if kind not in _KINDS:
        raise ValueError(f"`kind` must be one of {list(_KINDS)}, got {kind!r}.")
    coverage = validate_coverage(min_coverage)
    min_n = _group_default(min_entities, group, pooled=2)

    dense = dense_matrix(pf, returns, group=group)
    values = dense.values
    finite = np.isfinite(values)
    universe = finite & (
        window_counts(finite, window) >= coverage_count(window, coverage)
    )
    if group is not None:
        state = _avg_corr_groups(dense, universe, window, kind, min_n, group)
        on = [dense.time_col, group]
    else:
        state = _avg_corr_pooled(dense, universe, window, kind, min_n)
        on = [dense.time_col]
    return broadcast_rows(pf, state, on) if broadcast else state


def _avg_corr_pooled(
    dense: Dense, universe: NDArray[np.bool_], window: int, kind: str, min_n: int
) -> pl.DataFrame:
    values = dense.values
    n_time = values.shape[0]
    rho = np.full(n_time, np.nan, dtype=np.float64)
    used = np.zeros(n_time, dtype=np.int64)
    todo = np.flatnonzero(universe.any(axis=1))
    stable: NDArray[np.bool_] | None = None
    if kind == "pollet_wilson":
        fast, stable, n_full = _pollet_wilson_fast(values, window)
        stable &= n_full >= min_n
        rho[stable] = fast[stable]
        used[stable] = n_full[stable]
        todo = todo[~stable[todo]]
    for t in todo:
        lo = max(0, int(t) - window + 1)
        members = np.flatnonzero(universe[t])
        value, n, _labels = _window_kernel(
            values[lo : t + 1, members], kind, overwrite=True
        )
        used[t] = n[0]
        if n[0] >= min_n:
            rho[t] = value[0]
    columns: dict[str, Any] = {
        dense.time_col: dense.times,
        "avg_corr": _nullify(rho),
        "n_entities": pl.Series(used, dtype=pl.Int64),
    }
    if stable is not None:
        columns["universe_stable"] = pl.Series(stable, dtype=pl.Boolean)
    return pl.DataFrame(columns)


def _avg_corr_groups(
    dense: Dense,
    universe: NDArray[np.bool_],
    window: int,
    kind: str,
    min_n: int,
    group: str,
) -> pl.DataFrame:
    """One kernel call per date; each group's value from the shared moments."""
    values = dense.values
    codes = dense.group_codes
    labels = dense.group_labels
    if codes is None or labels is None:  # pragma: no cover - set by dense_matrix
        raise RuntimeError("group codes missing from a grouped pivot")
    out_t: list[NDArray[np.int64]] = []
    out_g: list[NDArray[np.int64]] = []
    out_rho: list[NDArray[np.float64]] = []
    out_n: list[NDArray[np.int64]] = []
    for t in np.flatnonzero((universe & (codes >= 0)).any(axis=1)):
        members = np.flatnonzero(universe[t] & (codes[t] >= 0))
        lo = max(0, int(t) - window + 1)
        rho, n_g, present = _window_kernel(
            values[lo : t + 1, members], kind, codes[t, members], overwrite=True
        )
        rho[n_g < min_n] = np.nan
        out_t.append(np.full(present.size, t, dtype=np.int64))
        out_g.append(present)
        out_rho.append(rho)
        out_n.append(n_g)
    empty_i = np.zeros(0, dtype=np.int64)
    t_idx = np.concatenate(out_t) if out_t else empty_i
    g_idx = np.concatenate(out_g) if out_g else empty_i
    state = pl.DataFrame(
        {
            dense.time_col: dense.times.gather(t_idx),
            group: labels.gather(g_idx),
            "avg_corr": _nullify(np.concatenate(out_rho) if out_rho else np.zeros(0)),
            "n_entities": pl.Series(
                np.concatenate(out_n) if out_n else empty_i, dtype=pl.Int64
            ),
        }
    )
    return state.sort([dense.time_col, group], maintain_order=True)


# --------------------------------------------------------------------------- #
# Common idiosyncratic volatility
# --------------------------------------------------------------------------- #
def common_idio_vol(
    panel: Any,
    *,
    returns: str,
    window: int = 63,
    n_factors: int = 1,
    refit_every: int = 21,
    fit_window: int = 252,
    min_coverage: float = 0.95,
    group: str | None = None,
    min_entities: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    broadcast: bool = False,
) -> pl.DataFrame:
    """Common idiosyncratic volatility (Herskovic et al. 2016), per date.

    1. **Residuals.** ``n_factors`` statistical factors are stripped out by
       :func:`panelary.detect._panel.residualise` (reused, not forked): loadings
       are fit by PCA on the ``fit_window`` dates before each refit date
       ``fit_window + k * refit_every`` -- a grid anchored at the panel's first
       date, so appending data never moves it -- frozen, and applied to each
       later date's own cross-section. Only names with at least
       ``ceil(min_coverage * fit_window)`` observations in the training block
       take part; the rest get no residual that segment.
    2. **Idiosyncratic volatility.** ``idio_vol`` is the trailing sample sd
       (``ddof=1``) of a name's residuals over ``window`` dates, needing
       ``ceil(min_coverage * window)`` of them. Residuals have mean near zero,
       so the native rolling moments are accurate.
    3. **CIV.** The equal-weight cross-sectional mean of ``idio_vol`` over the
       names observed at ``t`` that have one.

    Herskovic et al. use Fama-French residuals within each calendar month; this
    is the causal, statistical-factor version.

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long panel of returns.
    returns : str
        Return column; non-finite values are unobserved.
    window : int, default 63
        Trailing window of the idiosyncratic-volatility sd, in dates.
    n_factors : int, default 1
        Statistical factors removed (``0`` removes only each name's training
        mean).
    refit_every : int, default 21
        Dates between loading refits.
    fit_window : int, default 252
        Training block for the loadings, in dates. The first residual is at
        date index ``fit_window``.
    min_coverage : float, default 0.95
        Coverage a name needs in the training block and in the sd window.
    group : str, optional
        Group column: CIV is then the mean over each group's members, grouped by
        their label **on date t** (the residuals are the pooled ones).
    min_entities : int, optional
        Fewest names for a value; default 2, or 10 with ``group``.
    entity, time : str, optional
        Key columns, used only when ``panel`` is a bare frame.
    broadcast : bool, default False
        Return the panel rows with ``common_idio_vol`` (joined on time, and
        group) and each row's own ``idio_vol``.

    Returns
    -------
    polars.DataFrame
        ``time [, group], common_idio_vol, n_entities``, or the broadcast panel.
    """
    pf = coerce_panel(panel, entity, time)
    window = validate_int("window", window, minimum=2)
    n_factors = validate_int("n_factors", n_factors, minimum=0)
    refit_every = validate_int("refit_every", refit_every, minimum=1)
    fit_window = validate_int("fit_window", fit_window, minimum=2)
    coverage = validate_coverage(min_coverage)
    min_n = _group_default(min_entities, group, pooled=2)

    dense = dense_matrix(pf, returns, group=group)
    values = dense.values
    resid = segment_residuals(
        values,
        n_factors=n_factors,
        refit_every=refit_every,
        fit_window=fit_window,
        min_obs=coverage_count(fit_window, coverage),
    )
    idio = _rolling_columns(
        resid, window, coverage_count(window, coverage), stats=("std",)
    )["std"]
    member = np.isfinite(values) & np.isfinite(idio)
    tim = dense.time_col
    if group is None:
        sums, counts = masked_row_sums(idio, member)
        with np.errstate(invalid="ignore", divide="ignore"):
            civ = np.where(counts >= min_n, sums / np.maximum(counts, 1), np.nan)
        state = pl.DataFrame(
            {
                tim: dense.times,
                "common_idio_vol": _nullify(civ),
                "n_entities": pl.Series(counts, dtype=pl.Int64),
            }
        )
        on = [tim]
    else:
        state = _civ_groups(dense, idio, member, min_n, group)
        on = [tim, group]
    if not broadcast:
        return state
    if "idio_vol" in pf.columns:
        raise ValueError(
            "broadcast would overwrite panel column(s) ['idio_vol']; rename it "
            "first or call with broadcast=False."
        )
    rows = broadcast_rows(pf, state, on)
    per_name = long_from_dense(dense, {"idio_vol": idio})
    return rows.join(
        per_name, on=[dense.entity_col, tim], how="left", maintain_order="left"
    )


def _civ_groups(
    dense: Dense,
    idio: NDArray[np.float64],
    member: NDArray[np.bool_],
    min_n: int,
    group: str,
) -> pl.DataFrame:
    """Per ``(date, date-t group)`` means of ``idio`` over the members, vectorised.

    Members are taken row-major (date, then entity), labelled with their date-t
    code, and stably sorted by ``(date, code)``, so each segment keeps entity
    order and is reduced on its own.
    """
    codes = dense.group_codes
    labels = dense.group_labels
    if codes is None or labels is None:  # pragma: no cover - set by dense_matrix
        raise RuntimeError("group codes missing from a grouped pivot")
    t_idx, e_idx = np.nonzero(member)
    g_idx = codes[t_idx, e_idx]
    labelled = g_idx >= 0
    t_idx, e_idx, g_idx = t_idx[labelled], e_idx[labelled], g_idx[labelled]
    order = np.lexsort((g_idx, t_idx))
    t_idx, e_idx, g_idx = t_idx[order], e_idx[order], g_idx[order]
    vals = idio[t_idx, e_idx]
    if t_idx.size:
        new = np.empty(t_idx.size, dtype=bool)
        new[0] = True
        new[1:] = (t_idx[1:] != t_idx[:-1]) | (g_idx[1:] != g_idx[:-1])
        starts = np.flatnonzero(new)
    else:
        starts = np.zeros(0, dtype=np.int64)
    counts = np.diff(np.append(starts, t_idx.size))
    sums = segment_sums(vals, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        civ = np.where(counts >= min_n, sums / np.maximum(counts, 1), np.nan)
    tim = dense.time_col
    return pl.DataFrame(
        {
            tim: dense.times.gather(t_idx[starts]),
            group: labels.gather(g_idx[starts]),
            "common_idio_vol": _nullify(civ),
            "n_entities": pl.Series(counts, dtype=pl.Int64),
        }
    ).sort([tim, group], maintain_order=True)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Both are frame -> frame and mix entities by design (panel_safe=False). The
# value at t is a function of rows <= t only -- trailing windows, date-t
# universes and labels, an index-anchored refit grid -- so the per-row claim is
# "rowwise", verified by tests/test_registry_conformance.py through the
# broadcast form.
_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="avg_correlation",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "kind": str,
            "min_coverage": float,
            "group": object,
            "min_entities": object,
            "broadcast": bool,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Pollet & Wilson (2010), Journal of Financial Economics 96(3)",
        license=_LICENSE,
        backend_fn=avg_correlation,
        cost_hint="O(T N W) exact; O(T N) Pollet-Wilson on a stable universe",
    ),
    FeatureSpec(
        name="common_idio_vol",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "n_factors": int,
            "refit_every": int,
            "fit_window": int,
            "min_coverage": float,
            "group": object,
            "min_entities": object,
            "broadcast": bool,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source=(
            "Herskovic, Kelly, Lustig & Van Nieuwerburgh (2016), Journal of "
            "Financial Economics 119(2)"
        ),
        license=_LICENSE,
        backend_fn=common_idio_vol,
        cost_hint="residualise per refit + O(T N)",
    ),
)

for _spec in _SPECS:
    registry.register(_spec)
