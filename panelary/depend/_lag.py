"""Lag scans with family-wise error control across lags.

Lag scans are where naive libraries manufacture false discoveries: 49 lags x
200 features is 9800 tests. Here ``p_value_adj`` is **mandatory**, and the
default is :func:`~panelary.validation._selection_stats.romano_wolf` over one
shared ``(B x n_lags)`` null matrix: the resamples are drawn **once** and every
lag is evaluated on the same draw, so the joint null keeps the strong
correlation between adjacent lags that Romano-Wolf exploits (and a fresh
resample per lag would destroy). Statistics are studentised against their own
null draws before the max-stat stepdown so that lags with different sample
sizes are comparable.

Conventions
-----------
* ``lag = k > 0`` pairs ``x[t - k]`` with ``y[t]`` (``x`` leads); ``k < 0``
  pairs ``x[t + |k|]`` with ``y[t]`` (``y`` leads). Lags are in units of the
  shared date axis (row positions when there is no time column).
* :func:`nonlinear_acf` tests **serial independence** of one series, so its
  natural null is a plain permutation (or ``"iaaft"`` / ``"phase"`` to test for
  nonlinear serial structure *beyond* the linear autocorrelation). Block and
  shift nulls preserve exactly the dependence an ACF looks for and are refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary.depend._engine import (
    _CLOSED,
    DEFAULT_RESAMPLES,
    _base_row,
    _resolve,
    _sample_rows,
    aggregate_estimates,
    heterogeneity,
    panel_block_length,
    panel_serial,
    ragged_rows,
    resample_series,
    resolve_block_length,
)
from panelary.depend._frame import extract, result_frame
from panelary.depend._kernels import Kernel, get_kernel
from panelary.depend._null import (
    common_time_indices,
    entity_permutation_indices,
    pvalue,
    serial_dependence,
)
from panelary.depend._ranks import pairwise_complete
from panelary.validation._selection_stats import (
    benjamini_hochberg,
    benjamini_yekutieli,
    holm_bonferroni,
    romano_wolf,
)

__all__ = [
    "adjust_pvalues",
    "lag_dependence",
    "nonlinear_acf",
    "nonlinear_ccf",
    "optimal_lag",
]

_CORRECTIONS = frozenset(
    {"romano_wolf", "holm", "benjamini_hochberg", "benjamini_yekutieli", "none"}
)
_ACF_REFUSED = frozenset({"block", "shift", "stationary"})


def adjust_pvalues(
    pvals: np.ndarray,
    *,
    correction: str,
    stats: np.ndarray | None = None,
    draws: np.ndarray | None = None,
    two_sided: bool = False,
) -> tuple[np.ndarray, str]:
    """Multiplicity-adjust ``pvals`` (non-finite entries stay NaN).

    ``"romano_wolf"`` needs the observed ``stats`` and the shared ``(B, S)``
    ``draws``; both are studentised against the draws' per-hypothesis mean and
    standard deviation first. Without draws it falls back to Holm (reported in
    the returned label).

    Returns
    -------
    (adjusted, label)
    """
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"unknown `correction` {correction!r}; expected one of {sorted(_CORRECTIONS)}."
        )
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if correction == "none" or not ok.any():
        out[ok] = p[ok]
        return out, "none"
    if correction == "romano_wolf" and stats is not None and draws is not None:
        s = np.asarray(stats, dtype=np.float64)
        d = np.asarray(draws, dtype=np.float64)
        mu = np.nanmean(d, axis=0)
        sd = np.nanstd(d, axis=0)
        ok &= np.isfinite(s) & np.isfinite(sd) & (sd > 0) & np.isfinite(d).all(axis=0)
        if ok.sum() >= 1:
            ts = (s[ok] - mu[ok]) / sd[ok]
            td = (d[:, ok] - mu[ok]) / sd[ok]
            res = romano_wolf(ts, td, two_sided=two_sided)
            out[ok] = res.adjusted_pvalues
            return out, "romano_wolf"
        return out, "romano_wolf"
    label = correction
    if correction == "romano_wolf":
        label = "holm"
    fn = {
        "holm": holm_bonferroni,
        "benjamini_hochberg": benjamini_hochberg,
        "benjamini_yekutieli": benjamini_yekutieli,
    }[label]
    out[ok] = fn(np.clip(p[ok], 0.0, 1.0)).adjusted_pvalues
    return out, label


def _shift(X: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return X
    out = np.full(X.shape, np.nan)
    if k > 0:
        out[:, k:] = X[:, :-k]
    else:
        out[:, :k] = X[:, -k:]
    return out


def _resample_rows(
    scheme: str,
    V: np.ndarray,
    *,
    n_resamples: int,
    block_length: int | None,
    rng: np.random.Generator,
    partner: np.ndarray | None,
) -> np.ndarray:
    """``(B, N, T)`` per-row resamples of ``V`` on the date grid.

    Each row is resampled over its finite span (first to last finite date);
    NaN gaps inside the span travel with their blocks. IAAFT / phase
    surrogates need a gap-free span.
    """
    n_ent, t_len = V.shape
    b = int(n_resamples)
    out = np.full((b, n_ent, t_len), np.nan)
    for i in range(n_ent):
        fin = np.flatnonzero(np.isfinite(V[i]))
        if fin.size < 4:
            continue
        lo, hi = int(fin[0]), int(fin[-1]) + 1
        span = V[i, lo:hi]
        gaps = not np.isfinite(span).all()
        if gaps and scheme in {"iaaft", "phase"}:
            raise ValueError(
                f"null={scheme!r} needs gap-free series; entity row {i} has missing "
                "values inside its span. Use 'block' or 'shift'."
            )
        blen = block_length
        if blen is None and scheme in {"block", "stationary", "shift"}:
            other = partner[i, lo:hi] if partner is not None else span
            xa, ya, _ = pairwise_complete(other, span)
            blen = resolve_block_length(scheme, xa, ya, None) if xa.size >= 8 else 1
        if gaps:
            # Resample positions (a permutation of the span) so NaN travel along.
            pos = np.arange(span.size, dtype=np.float64)
            idx = resample_series(
                scheme, pos, n_resamples=b, block_length=blen, rng=rng
            ).astype(np.int64)
            draws = span[idx]
        else:
            draws = resample_series(
                scheme, span, n_resamples=b, block_length=blen, rng=rng
            )
        out[: draws.shape[0], i, lo:hi] = draws
    return out


def _lag_stats(
    kernel: Kernel,
    X: np.ndarray,
    Y: np.ndarray,
    lags: Sequence[int],
    gate: int,
    how: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Observed per-lag aggregate, total n_obs, n_entities and per-entity stats."""
    n_l = len(lags)
    agg = np.full(n_l, np.nan)
    nobs = np.zeros(n_l, dtype=np.int64)
    nent = np.zeros(n_l, dtype=np.int64)
    per: list[tuple[np.ndarray, np.ndarray]] = []
    for j, k in enumerate(lags):
        est, cnt = ragged_rows(kernel.rows, _shift(X, k), Y, gate)
        v = np.isfinite(est)
        nobs[j] = int(cnt[v].sum())
        nent[j] = int(v.sum())
        if v.any():
            agg[j] = (
                est[0]
                if how is None
                else float(aggregate_estimates(est, cnt, kernel, how))
            )
        per.append((est, cnt))
    return agg, nobs, nent, per


def _draw_matrix(
    kernel: Kernel,
    X: np.ndarray,
    Yp: np.ndarray,
    lags: Sequence[int],
    gate: int,
    how: str | None,
    acf: bool,
) -> np.ndarray:
    """``(nb, n_lags)`` null statistics for a ``(nb, N, T)`` block of draws."""
    nb, n_ent, t_len = Yp.shape
    out = np.full((nb, len(lags)), np.nan)
    Yr = Yp.reshape(nb * n_ent, t_len)
    for j, k in enumerate(lags):
        if acf:
            Xs = _shift(Yr, k)
        else:
            Xs = np.broadcast_to(_shift(X, k), Yp.shape).reshape(nb * n_ent, t_len)
        est, cnt = ragged_rows(kernel.rows, Xs, Yr, gate)
        est = est.reshape(nb, n_ent)
        cnt = cnt.reshape(nb, n_ent)
        out[:, j] = (
            est[:, 0] if how is None else aggregate_estimates(est, cnt, kernel, how)
        )
    return out


def lag_dependence(
    df: Any,
    x: str,
    y: str,
    *,
    max_lag: int = 24,
    lags: Sequence[int] | None = None,
    method: str = "xi",
    null: str = "auto",
    by: str = "entity",
    entity: str | None = None,
    time: str | None = None,
    how: str | None = None,
    correction: str = "romano_wolf",
    q: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
    _acf: bool = False,
) -> pl.DataFrame:
    """Dependence of ``y[t]`` on ``x[t - k]`` for every lag ``k``, FWER-adjusted.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
    x, y : str
        Columns.
    max_lag : int, default=24
        Scan ``k`` in ``[-max_lag, max_lag]`` unless ``lags`` is given. A fixed
        integer, never derived from the series length.
    lags : sequence of int, optional
        Explicit lags.
    method : str, default="xi"
        Any :func:`~panelary.depend.dependence` method.
    null : str, default="auto"
        As in :func:`~panelary.depend.dependence`. With ``null="auto"`` and the
        default ``correction="romano_wolf"`` a closed-form resolution is
        replaced by an i.i.d. permutation null so that one joint resample
        matrix exists; an explicit closed-form null falls back to Holm.
    by : {"entity"}, default="entity"
        Per-entity statistics aggregated per lag.
    entity, time : str, optional
        Keys for a bare frame.
    how : str, optional
        Aggregation across entities (method default).
    correction : {"romano_wolf", "holm", "benjamini_hochberg", "benjamini_yekutieli", "none"}
        Multiplicity correction across lags for ``p_value_adj``.
    q, n_resamples, block_length, seed, min_obs
        As in :func:`~panelary.depend.dependence`.

    Returns
    -------
    polars.DataFrame
        One row per lag: ``x``, ``y`` and the fixed schema (``lag``,
        ``estimate``, ``p_value``, ``p_value_adj``, ``null_method``,
        ``n_obs``, ...).
    """
    if by != "entity":
        raise ValueError(
            "lag scans aggregate per entity; by='entity' is the only option."
        )
    lag_list = (
        [int(k) for k in lags]
        if lags is not None
        else list(range(1 if _acf else -int(max_lag), int(max_lag) + 1))
    )
    if not lag_list:
        raise ValueError("no lags to scan.")
    kernel = get_kernel(method, q=q)
    cols = [x] if _acf else [x, y]
    pa = extract(df, cols, entity=entity, time=time)
    Y = pa.dense(y)
    X = Y if _acf else pa.dense(x)
    n_ent = X.shape[0]
    single = n_ent == 1
    how_ = None if single else (how or kernel.default_how)
    gate = int(min_obs if min_obs is not None else kernel.min_obs)
    obs, nobs, nent, per = _lag_stats(kernel, X, Y, lag_list, gate, how_)
    notes: list[str] = []
    # ---- resolve the null -------------------------------------------------
    rows_s = _sample_rows(n_ent)
    if single:
        xa, ya, _ = pairwise_complete(X[0], Y[0])
        serial = serial_dependence(ya) if _acf else serial_dependence(xa, ya)
    else:
        serial = panel_serial(X, Y, rows_s)
    panel_ok = n_ent >= 2 and pa.has_time
    if _acf:
        if null in _ACF_REFUSED:
            raise ValueError(
                f"null={null!r} preserves the serial dependence an ACF tests for; use "
                "'permutation' (serial independence) or 'iaaft' / 'phase' (nonlinear "
                "structure beyond the linear ACF)."
            )
        if null == "auto":
            scheme = "common-time" if panel_ok else "permutation"
        elif null in {"iid", "asymptotic"}:
            scheme = "asymptotic"
        else:
            scheme = null
    else:
        scheme, w = _resolve(kernel, null, serial=serial, panel=panel_ok)
        notes += w
        if scheme == "common-time" and single:
            scheme = "block"
    if scheme == "common-time" and not pa.has_time:
        raise ValueError("null='common-time' needs a `time` column.")
    if scheme in _CLOSED and correction == "romano_wolf" and null == "auto":
        notes.append(
            "romano_wolf needs a joint resample matrix: the closed-form null was "
            "replaced by an i.i.d. permutation null (valid: the serial pre-check passed)."
        )
        scheme = "permutation"
    # ---- closed form per lag ---------------------------------------------
    B = int(n_resamples)
    blen: int | None = None
    draws: np.ndarray | None = None
    pvals = np.full(len(lag_list), np.nan)
    if scheme in _CLOSED:
        if kernel.closed_form is None:
            raise ValueError(f"{kernel.name} has no closed-form null.")
        for j, k in enumerate(lag_list):
            Xs = _shift(X, k)
            if single:
                xa, ya, n = pairwise_complete(Xs[0], Y[0])
                if n >= gate and np.isfinite(obs[j]):
                    pvals[j] = kernel.closed_form(float(obs[j]), xa, ya)[0]
            else:
                from panelary.depend._engine import stouffer

                est, cnt = per[j]
                pv = np.full(n_ent, np.nan)
                for i in np.flatnonzero(np.isfinite(est)):
                    xa, ya, _ = pairwise_complete(Xs[i], Y[i])
                    pv[i] = kernel.closed_form(float(est[i]), xa, ya)[0]
                v = np.isfinite(est)
                pvals[j] = stouffer(
                    pv[v], est[v], cnt[v], alternative=kernel.alternative
                )
        null_label = scheme if single else f"{scheme}+stouffer"
    else:
        null_label = scheme
        if scheme in {"common-time", "entity"}:
            t_len = X.shape[1]
            if scheme == "common-time":
                blen = (
                    1
                    if _acf
                    else (
                        panel_block_length(X, Y, rows_s, block_length)
                        if serial > 0.2 or block_length is not None
                        else 1
                    )
                )
                idx = common_time_indices(
                    np.arange(t_len), n_resamples=B, seed=seed, block=blen
                )
            else:
                idx = entity_permutation_indices(n_ent, n_resamples=B, seed=seed)
            draws = np.full((B, len(lag_list)), np.nan)
            step = max(1, 2_000_000 // max(n_ent * t_len, 1))
            for s in range(0, B, step):
                sl = idx[s : s + step]
                Yp = (
                    np.transpose(Y[:, sl], (1, 0, 2))
                    if scheme == "common-time"
                    else Y[sl]
                )
                if _acf and scheme == "entity":
                    raise ValueError("null='entity' is not defined for an ACF.")
                draws[s : s + step] = _draw_matrix(
                    kernel, X, Yp, lag_list, gate, how_, _acf
                )
        else:
            rng = np.random.default_rng(seed)
            if scheme in {"block", "stationary", "shift"} and block_length is not None:
                blen = int(block_length)
            Yp = _resample_rows(
                scheme,
                Y,
                n_resamples=B,
                block_length=block_length,
                rng=rng,
                partner=None if _acf else X,
            )
            draws = _draw_matrix(kernel, X, Yp, lag_list, gate, how_, _acf)
            if single and blen is None and scheme in {"block", "stationary", "shift"}:
                xa, ya, _ = pairwise_complete(X[0], Y[0])
                blen = resolve_block_length(scheme, xa, ya, None)
        for j in range(len(lag_list)):
            pvals[j] = pvalue(
                float(obs[j]), draws[:, j], alternative=kernel.alternative
            )
    adj, adj_label = adjust_pvalues(
        pvals,
        correction=correction,
        stats=obs,
        draws=draws,
        two_sided=kernel.alternative == "two-sided",
    )
    if adj_label != correction:
        notes.append(f"p_value_adj: {correction} unavailable here; used {adj_label}.")
    notes.append(f"p_value_adj: {adj_label} across {len(lag_list)} lags.")
    rows: list[dict[str, Any]] = []
    for j, k in enumerate(lag_list):
        row = _base_row(kernel)
        if how_ is not None:
            row["estimator"] = f"{row['estimator']}; aggregate={how_}"
            est, cnt = per[j]
            row["heterogeneity"] = heterogeneity(est, cnt, kernel, how_)[0]
        row.update(
            x=x,
            y=x if _acf else y,
            lag=k,
            estimate=float(obs[j]),
            p_value=float(pvals[j]),
            p_value_adj=float(adj[j]),
            null_method=null_label,
            n_resamples=None if draws is None else int(draws.shape[0]),
            block_length=blen,
            n_obs=int(nobs[j]),
            n_entities=int(nent[j]),
            coverage=(nent[j] / n_ent) if n_ent else float("nan"),
            seed=None if draws is None else int(seed),
            warnings=list(notes),
        )
        if _acf:
            row["direction"] = "x[t-k]->x[t]" if kernel.directed else "symmetric"
        rows.append(row)
    return result_frame(rows, keys={"x": pl.Utf8, "y": pl.Utf8})


def nonlinear_acf(
    df: Any,
    x: str,
    *,
    max_lag: int = 24,
    lags: Sequence[int] | None = None,
    method: str = "xi",
    null: str = "auto",
    entity: str | None = None,
    time: str | None = None,
    correction: str = "romano_wolf",
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Nonlinear autocorrelation function: dependence of ``x[t]`` on ``x[t - k]``.

    Lags ``1..max_lag``. ``null="auto"`` is an i.i.d. permutation (a
    common-time permutation of whole dates for a panel), which tests serial
    **independence**; ``null="iaaft"`` or ``"phase"`` keeps the linear
    autocorrelation and tests for nonlinear serial structure beyond it -- the
    diagnostic to run on model residuals.

    Returns
    -------
    polars.DataFrame
        One row per lag, the fixed schema with ``p_value_adj`` across lags.
    """
    return lag_dependence(
        df,
        x,
        x,
        max_lag=max_lag,
        lags=lags,
        method=method,
        null=null,
        entity=entity,
        time=time,
        correction=correction,
        n_resamples=n_resamples,
        seed=seed,
        min_obs=min_obs,
        _acf=True,
    )


def nonlinear_ccf(
    df: Any,
    x: str,
    y: str,
    *,
    max_lag: int = 24,
    lags: Sequence[int] | None = None,
    method: str = "dcor",
    null: str = "auto",
    entity: str | None = None,
    time: str | None = None,
    correction: str = "romano_wolf",
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Nonlinear cross-correlation function over ``[-max_lag, max_lag]``.

    ``lag = k > 0``: ``x`` leads ``y`` by ``k``. See :func:`lag_dependence`.
    """
    return lag_dependence(
        df,
        x,
        y,
        max_lag=max_lag,
        lags=lags,
        method=method,
        null=null,
        entity=entity,
        time=time,
        correction=correction,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        min_obs=min_obs,
    )


def optimal_lag(profile: pl.DataFrame) -> int:
    """The most significant lag of a lag profile.

    Smallest ``p_value_adj`` (then ``p_value``), ties broken by the larger
    ``|estimate|`` and then the smaller ``|lag|``.

    Raises
    ------
    ValueError
        If no lag has a finite estimate.
    """
    df = profile.filter(
        pl.col("estimate").is_not_null() & pl.col("estimate").is_not_nan()
    )
    if df.height == 0:
        raise ValueError("no lag has a finite estimate.")
    df = df.with_columns(
        pl.col("p_value_adj").fill_nan(None).fill_null(2.0).alias("_pa"),
        pl.col("p_value").fill_nan(None).fill_null(2.0).alias("_p"),
        (-pl.col("estimate").abs()).alias("_e"),
        pl.col("lag").abs().alias("_l"),
    ).sort(["_pa", "_p", "_e", "_l"])
    return int(df.get_column("lag")[0])
