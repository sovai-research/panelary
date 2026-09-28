"""The resampling engine behind every test in :mod:`panelary.depend`.

One statistic, three data shapes, one null policy:

* **a single series** (:func:`series_test`) -- complete pairs, the null
  resolved by :func:`~panelary.depend._null.resolve_null` from the lag-1 rank
  autocorrelation pre-check, then either a closed form or ``B`` resampled
  copies of ``y`` evaluated in **one** row-batched kernel call per chunk;
* **a panel, by entity** (:func:`panel_test`) -- per-entity estimates on the
  shared date axis, aggregated (Fisher by default) into one number whose null
  is the **same aggregate** recomputed on resampled panels: whole date columns
  permuted jointly for ``"common-time"``, entities re-paired for ``"entity"``,
  or each entity resampled on its own for the per-series schemes;
* **a panel, pooled** (:func:`pooled_test`) -- every complete pair stacked into
  one sample after an explicit, reported demeaning.

The engine never knows what statistic it runs: a
:class:`~panelary.depend._kernels.Kernel` supplies the row-batched
implementation, its alternative, its closed form and its sample-size gate.

Every result is a dict on the fixed :data:`~panelary.depend._frame.RESULT_SCHEMA`
-- no bare p-value ever leaves this module without ``null_method``,
``n_resamples``, ``block_length`` and ``n_obs`` beside it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary.depend._frame import PanelArrays, extract, result_frame
from panelary.depend._kernels import Kernel, get_kernel
from panelary.depend._null import (
    SERIAL_THRESHOLD,
    block_permutation_indices,
    circular_shift,
    common_time_indices,
    entity_permutation_indices,
    iaaft,
    pair_block_length,
    phase_randomise,
    pvalue,
    resolve_null,
    serial_dependence,
    stationary_indices,
)
from panelary.depend._ranks import pairwise_complete
from panelary.depend._special import norm_sf
from panelary.econ._common import norm_ppf

__all__ = [
    "DEFAULT_RESAMPLES",
    "aggregate_estimates",
    "dependence",
    "devolatilise",
    "heterogeneity",
    "independence_test",
    "ragged_rows",
    "resample_series",
    "panel_test",
    "pooled_test",
    "series_test",
]

RowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: Default number of resamples. The smallest achievable p-value is
#: ``1 / (B + 1) = 0.005``; raise it when correcting over many hypotheses.
DEFAULT_RESAMPLES = 199

#: Rough ceiling on the elements handed to one kernel call (~16 MB float64).
_MAX_ELEMS = 2_000_000
#: Clip before ``arctanh`` so a perfect estimate cannot produce ``inf``.
_CLIP = 1.0 - 1e-12

_CLOSED = frozenset({"asymptotic", "gamma", "t", "exact"})
_PER_SERIES = frozenset(
    {"permutation", "block", "stationary", "shift", "iaaft", "phase"}
)
_NEEDS_BLOCK = frozenset({"block", "stationary", "shift", "common-time"})
_HOW = frozenset({"fisher", "precision", "mean", "median"})
_DEMEAN = frozenset({"none", "entity", "time", "two-way"})

#: A ``(B, m)`` evaluation is split so no chunk exceeds this many elements.
_GCMI_NOTE = (
    "gcmi on a single pair of columns is a strictly monotone function of the "
    "van der Waerden rank correlation (Spearman in a hat): it cannot see "
    "non-monotone dependence. Use method='xi', 'dcor' or 'hsic' for that; "
    "gcmi earns its place for multivariate, conditional and matrix MI."
)


# --------------------------------------------------------------------------- #
# Batched evaluation
# --------------------------------------------------------------------------- #
def rows_chunked(fn: RowFn, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Evaluate ``fn`` on complete ``(B, m)`` arrays in memory-bounded chunks."""
    bsz, m = Y.shape
    step = max(1, _MAX_ELEMS // max(m, 1))
    if step >= bsz:
        return np.asarray(fn(np.ascontiguousarray(X), np.ascontiguousarray(Y)), float)
    out = np.empty(bsz)
    for s in range(0, bsz, step):
        out[s : s + step] = fn(
            np.ascontiguousarray(X[s : s + step]), np.ascontiguousarray(Y[s : s + step])
        )
    return out


def ragged_rows(
    fn: RowFn, X: np.ndarray, Y: np.ndarray, min_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row statistics of ``(R, T)`` arrays that may contain NaN.

    Each row is restricted to its complete pairs (time order kept). Rows are
    bucketed by their complete count so every bucket is one batched kernel
    call -- a balanced panel is a single call.

    Returns
    -------
    (stats, counts)
        ``(R,)`` float64 statistics (NaN below ``min_obs``) and int64 complete
        counts.
    """
    mask = np.isfinite(X) & np.isfinite(Y)
    cnt = mask.sum(axis=1).astype(np.int64)
    out = np.full(X.shape[0], np.nan)
    ok = cnt >= max(int(min_obs), 2)
    if not ok.any():
        return out, cnt
    t_len = X.shape[1]
    for m in np.unique(cnt[ok]):
        rows = np.flatnonzero(cnt == m)
        if m == t_len:
            xs, ys = X[rows], Y[rows]
        else:
            order = np.argsort(~mask[rows], axis=1, kind="stable")[:, :m]
            xs = np.take_along_axis(X[rows], order, axis=1)
            ys = np.take_along_axis(Y[rows], order, axis=1)
        out[rows] = rows_chunked(fn, xs, ys)
    return out, cnt


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def _child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**63 - 1))


def resolve_block_length(
    scheme: str, x: np.ndarray, y: np.ndarray, block_length: int | None
) -> int | None:
    """Block length for ``scheme``: the user's, else :func:`pair_block_length`.

    ``None`` for schemes without blocks, and for ``"shift"`` (a rotation test is
    exact only over the full rotation group, so it takes no block length).
    """
    if scheme not in _NEEDS_BLOCK or scheme == "shift":
        return None
    if block_length is not None:
        if int(block_length) < 1:
            raise ValueError(f"`block_length` must be >= 1, got {block_length}.")
        return int(block_length)
    return pair_block_length(x, y)


def resample_series(
    scheme: str,
    y: np.ndarray,
    *,
    n_resamples: int,
    block_length: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(B', n)`` null copies of the complete series ``y`` under ``scheme``.

    ``B' = n_resamples`` except for ``"shift"``, which enumerates every
    admissible rotation (and returns fewer) when asked for more than exist.
    """
    ya = np.asarray(y, dtype=np.float64)
    n = ya.size
    b = int(n_resamples)
    if scheme == "permutation":
        return rng.permuted(np.broadcast_to(ya, (b, n)), axis=1)
    if scheme == "block":
        idx = block_permutation_indices(
            n, block_length=int(block_length or 1), n_resamples=b, seed=rng
        )
        return ya[idx]
    if scheme == "stationary":
        idx = stationary_indices(
            n, block_length=int(block_length or 1), n_resamples=b, seed=_child_seed(rng)
        )
        return ya[idx]
    if scheme == "shift":
        # Every non-trivial rotation is admissible: excluding the short shifts
        # makes the draw set a non-group and the test anti-conservative
        # (measured: min_shift=150 -> 9.5% at nominal 5% on AR(0.95) pairs).
        idx = circular_shift(n, n_resamples=b, seed=_child_seed(rng), min_shift=1)
        return ya[idx]
    if scheme == "iaaft":
        return iaaft(ya, n_resamples=b, seed=_child_seed(rng)).values
    if scheme == "phase":
        return phase_randomise(ya, n_resamples=b, seed=_child_seed(rng)).values
    raise ValueError(f"scheme {scheme!r} does not resample a single series.")


# --------------------------------------------------------------------------- #
# Aggregation across entities
# --------------------------------------------------------------------------- #
def _to_scale(est: np.ndarray, how: str) -> np.ndarray:
    if how in {"fisher", "precision"}:
        return np.arctanh(np.clip(est, -_CLIP, _CLIP))
    return est


def aggregate_estimates(
    est: np.ndarray, cnt: np.ndarray, kernel: Kernel, how: str
) -> np.ndarray:
    """Aggregate per-entity estimates along the last axis (NaN entries skipped).

    ``"fisher"``: inverse-variance weighted mean of ``arctanh(est)`` with
    weights ``1 / kernel.z_var(n_i)``, back-transformed. ``"precision"``: the
    DerSimonian-Laird random-effects version of the same. ``"mean"`` /
    ``"median"``: unweighted, on the raw scale. Works on ``(N,)`` or ``(B, N)``.
    """
    if how not in _HOW:
        raise ValueError(f"unknown `how` {how!r}; expected one of {sorted(_HOW)}.")
    e = np.asarray(est, dtype=np.float64)
    c = np.asarray(cnt, dtype=np.float64)
    valid = np.isfinite(e) & (c > 0)
    k = valid.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "mean":
            s = np.where(valid, e, 0.0).sum(axis=-1)
            return np.where(k > 0, s / np.maximum(k, 1), np.nan)
        if how == "median":
            out = (
                np.nanmedian(np.where(valid, e, np.nan), axis=-1) if e.size else np.nan
            )
            return np.where(k > 0, out, np.nan)
        z = np.where(valid, _to_scale(e, how), 0.0)
        w = np.where(valid, 1.0 / kernel.z_var(np.where(valid, c, 1.0)), 0.0)
        sw = w.sum(axis=-1)
        zbar = (w * z).sum(axis=-1) / np.where(sw > 0, sw, 1.0)
        if how == "precision":
            q = (w * (z - np.expand_dims(zbar, -1)) ** 2).sum(axis=-1)
            c_dl = sw - (w * w).sum(axis=-1) / np.where(sw > 0, sw, 1.0)
            tau2 = np.maximum(0.0, (q - (k - 1)) / np.where(c_dl > 0, c_dl, np.inf))
            v = np.where(valid, 1.0 / np.where(w > 0, w, 1.0), np.inf)
            ws = np.where(valid, 1.0 / (v + np.expand_dims(tau2, -1)), 0.0)
            sws = ws.sum(axis=-1)
            zbar = (ws * z).sum(axis=-1) / np.where(sws > 0, sws, 1.0)
        return np.where(k > 0, np.tanh(zbar), np.nan)


def heterogeneity(
    est: np.ndarray, cnt: np.ndarray, kernel: Kernel, how: str
) -> tuple[float, float, int]:
    """Cochran's ``Q`` and Higgins' ``I^2`` of per-entity estimates.

    Computed on the aggregation scale (``arctanh`` for Fisher / precision
    weighting, raw otherwise) with the kernel's working variances. Returns
    ``(I2, Q, k)``; ``I2`` is ``nan`` for fewer than two entities.
    """
    e = np.asarray(est, dtype=np.float64)
    c = np.asarray(cnt, dtype=np.float64)
    valid = np.isfinite(e) & (c > 0)
    k = int(valid.sum())
    if k < 2:
        return float("nan"), float("nan"), k
    scale = "fisher" if how in {"fisher", "precision"} else "raw"
    z = _to_scale(e[valid], scale)
    w = 1.0 / kernel.z_var(c[valid])
    zbar = float((w * z).sum() / w.sum())
    q = float((w * (z - zbar) ** 2).sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return i2, q, k


def stouffer(
    pvals: np.ndarray, est: np.ndarray, n: np.ndarray, *, alternative: str
) -> float:
    """Weighted (``sqrt(n_i)``) Stouffer combination of per-entity p-values.

    Two-sided p-values are signed by their estimate first, so opposite-signed
    entities cancel instead of adding up. Valid only for cross-sectionally
    independent entities.
    """
    p = np.clip(np.asarray(pvals, dtype=np.float64), 1e-300, 1.0)
    e = np.asarray(est, dtype=np.float64)
    w = np.sqrt(np.asarray(n, dtype=np.float64))
    ok = np.isfinite(p) & np.isfinite(e)
    if not ok.any():
        return float("nan")
    p, e, w = p[ok], e[ok], w[ok]
    # Lower-tail quantiles (-Phi^-1(p)) stay accurate for p -> 0, where
    # Phi^-1(1 - p) would round 1 - p to 1 and return inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        if alternative == "two-sided":
            z = -np.sign(e) * np.asarray(norm_ppf(np.minimum(p / 2.0, 0.5)), float)
        else:
            z = -np.asarray(norm_ppf(np.clip(p, 1e-300, 1.0 - 1e-16)), float)
    zz = float((w * z).sum() / math.sqrt(float((w * w).sum())))
    if alternative == "two-sided":
        return float(min(1.0, 2.0 * float(norm_sf(abs(zz)))))
    return float(norm_sf(zz))


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def devolatilise(v: np.ndarray, window: int) -> np.ndarray:
    """Divide by a **causal** rolling volatility, row by row.

    ``v[:, t] / sd(v[:, t - window : t])`` (``ddof=1``) -- the window ends at
    ``t - 1``, so the scale never contains the value it scales. The first
    ``window`` dates, and any window with a missing value, are NaN. A fixed
    integer window keeps it prefix-invariant.

    Parameters
    ----------
    v : numpy.ndarray
        ``(N, T)`` or ``(T,)`` values on a date axis.
    window : int
        Trailing window length ``>= 2``.

    Returns
    -------
    numpy.ndarray
        Same shape, float64.
    """
    w = int(window)
    if w < 2:
        raise ValueError(f"`devol` window must be an integer >= 2, got {window!r}.")
    arr = np.asarray(v, dtype=np.float64)
    one = arr.ndim == 1
    a = arr[None, :] if one else arr
    out = np.full(a.shape, np.nan)
    t_len = a.shape[1]
    if t_len > w:
        step = max(1, _MAX_ELEMS // (t_len * w))
        for s in range(0, a.shape[0], step):
            blk = a[s : s + step]
            view = np.lib.stride_tricks.sliding_window_view(blk[:, :-1], w, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sd = view.std(axis=-1, ddof=1)
                out[s : s + step, w:] = np.where(sd > 0, blk[:, w:] / sd, np.nan)
    return out[0] if one else out


def _shift_pair(
    X: np.ndarray, Y: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pair ``x[t - lag]`` with ``y[t]`` along the date axis (NaN-padded)."""
    k = int(lag)
    if k == 0:
        return X, Y
    Xs = np.full(X.shape, np.nan)
    if k > 0:
        Xs[:, k:] = X[:, :-k]
    else:
        Xs[:, :k] = X[:, -k:]
    return Xs, Y


def demean_panel(V: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """Remove entity, time or two-way means from the complete cells of ``V``.

    Uses :func:`panelary.econ._hdfe.demean` (alternating projections) on the
    cells in ``mask``; other cells become NaN.
    """
    if how not in _DEMEAN:
        raise ValueError(
            f"unknown `demean` {how!r}; expected one of {sorted(_DEMEAN)}."
        )
    out = np.where(mask, V, np.nan)
    if how == "none":
        return out
    from panelary.econ._hdfe import demean

    e_idx, t_idx = np.nonzero(mask)
    codes: list[np.ndarray] = []
    levels: list[int] = []
    if how in {"entity", "two-way"}:
        codes.append(e_idx.astype(np.int64))
        levels.append(V.shape[0])
    if how in {"time", "two-way"}:
        codes.append(t_idx.astype(np.int64))
        levels.append(V.shape[1])
    resid = demean(V[mask], codes, levels)[0]
    out[mask] = resid
    return out


# --------------------------------------------------------------------------- #
# Result rows
# --------------------------------------------------------------------------- #
def _base_row(kernel: Kernel) -> dict[str, Any]:
    return {
        "method": kernel.name,
        "estimator": kernel.estimator or kernel.name,
        "direction": "x->y" if kernel.directed else "symmetric",
        "lag": 0,
        "transform": "none",
        "approximate": False,
        "warnings": [],
    }


def _resolve(
    kernel: Kernel, null: str, *, serial: float, panel: bool
) -> tuple[str, list[str]]:
    scheme, notes = resolve_null(
        kernel.policy_key,
        null,
        serial=serial,
        panel=panel,
        closed_form=kernel.closed_form is not None,
    )
    if scheme == "hac" and kernel.hac is None:
        raise ValueError(
            f"null='hac' is only defined for method='gcm', not {kernel.name!r}."
        )
    return scheme, notes


def _fallback_scheme(serial: float) -> str:
    return "block" if serial > SERIAL_THRESHOLD else "permutation"


# --------------------------------------------------------------------------- #
# Single series
# --------------------------------------------------------------------------- #
def series_test(
    x: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    *,
    null: str = "auto",
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> dict[str, Any]:
    """Estimate and test ``kernel`` on one pair of series (NaN rows dropped).

    Returns a dict on the fixed result schema.
    """
    xa, ya, n = pairwise_complete(np.ravel(x), np.ravel(y))
    total = int(np.ravel(x).size)
    row = _base_row(kernel)
    gate = int(min_obs if min_obs is not None else kernel.min_obs)
    row.update(n_obs=n, n_entities=1, coverage=(n / total) if total else float("nan"))
    notes: list[str] = []
    if n < gate:
        notes.append(
            f"n_obs={n} is below min_obs={gate} for {kernel.name}; estimate withheld "
            "(NaN) rather than reporting a noisy number."
        )
        row.update(estimate=float("nan"), p_value=float("nan"), null_method="none")
        row["warnings"] = notes
        return row
    est = float(kernel.rows(xa[None, :], ya[None, :])[0])
    serial = serial_dependence(xa, ya)
    if null == "entity":
        raise ValueError("null='entity' re-pairs entities and needs a panel of >= 2.")
    scheme, w = _resolve(kernel, null, serial=serial, panel=False)
    notes += w
    if scheme == "common-time":
        scheme = "block"
        notes.append(
            "single series: the common-time null reduces to a block permutation "
            "of the date axis."
        )
    row["estimate"] = est
    p = float("nan")
    if scheme in _CLOSED or scheme == "hac":
        if scheme == "hac":
            assert kernel.hac is not None
            p, label = kernel.hac(est, xa, ya)
        else:
            assert kernel.closed_form is not None
            p, label = kernel.closed_form(est, xa, ya)
        if np.isfinite(p) or not np.isfinite(est):
            row.update(
                p_value=p, null_method=label, n_resamples=None, block_length=None
            )
            row["warnings"] = notes
            return row
        scheme = _fallback_scheme(serial)
        notes.append(
            f"the closed-form null for {kernel.name} refused this sample (heavy "
            f"ties); fell back to a {scheme} null."
        )
    blen = resolve_block_length(scheme, xa, ya, block_length)
    rng = np.random.default_rng(seed)
    yb = resample_series(
        scheme, ya, n_resamples=n_resamples, block_length=blen, rng=rng
    )
    draws = rows_chunked(kernel.rows, np.broadcast_to(xa, yb.shape), yb)
    p = pvalue(est, draws, alternative=kernel.alternative)
    row.update(
        p_value=p,
        null_method=scheme,
        n_resamples=int(yb.shape[0]),
        block_length=blen,
        seed=int(seed),
        _draws=draws,
    )
    row["warnings"] = notes
    return row


# --------------------------------------------------------------------------- #
# Panel, by entity
# --------------------------------------------------------------------------- #
def _sample_rows(n_rows: int, cap: int = 100) -> np.ndarray:
    if n_rows <= cap:
        return np.arange(n_rows)
    return np.unique(np.linspace(0, n_rows - 1, cap).round().astype(np.int64))


def panel_serial(X: np.ndarray, Y: np.ndarray, rows: np.ndarray) -> float:
    """Median over (up to 100 evenly spaced) entities of the serial pre-check."""
    vals = []
    for i in rows:
        xa, ya, n = pairwise_complete(X[i], Y[i])
        if n >= 4:
            vals.append(serial_dependence(xa, ya))
    return float(np.median(vals)) if vals else 0.0


def panel_block_length(
    X: np.ndarray, Y: np.ndarray, rows: np.ndarray, block_length: int | None
) -> int:
    """Common-time block length: the user's, or the median per-entity
    :func:`~panelary.depend._null.pair_block_length` over up to 100 entities."""
    if block_length is not None:
        return int(block_length)
    vals = []
    for i in rows:
        xa, ya, n = pairwise_complete(X[i], Y[i])
        if n >= 8:
            vals.append(pair_block_length(xa, ya))
    return int(np.median(vals)) if vals else 1


def _stack_draws(
    kernel: Kernel, X: np.ndarray, Yp: np.ndarray, gate: int
) -> tuple[np.ndarray, np.ndarray]:
    """``Yp`` is ``(nb, N, T)``; evaluate each draw's per-entity statistics."""
    nb, n_ent, t_len = Yp.shape
    Xr = np.broadcast_to(X, Yp.shape).reshape(nb * n_ent, t_len)
    est, cnt = ragged_rows(kernel.rows, Xr, Yp.reshape(nb * n_ent, t_len), gate)
    return est.reshape(nb, n_ent), cnt.reshape(nb, n_ent)


def panel_null_draws(
    kernel: Kernel,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    scheme: str,
    n_resamples: int,
    block_length: int | None,
    seed: int,
    gate: int,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(B, N)`` per-entity null statistics and counts under ``scheme``.

    ``"common-time"`` permutes whole date columns of ``Y`` (in blocks),
    identically for every entity; ``"entity"`` re-pairs entities; every other
    scheme resamples each entity's complete series independently.
    """
    n_ent, t_len = X.shape
    b = int(n_resamples)
    if scheme in {"common-time", "entity"}:
        if scheme == "common-time":
            idx = common_time_indices(
                np.arange(t_len), n_resamples=b, seed=seed, block=int(block_length or 1)
            )
        else:
            idx = entity_permutation_indices(n_ent, n_resamples=b, seed=seed)
        est = np.full((b, n_ent), np.nan)
        cnt = np.zeros((b, n_ent), dtype=np.int64)
        step = max(1, _MAX_ELEMS // max(n_ent * t_len, 1))
        for s in range(0, b, step):
            sl = idx[s : s + step]
            Yp = np.transpose(Y[:, sl], (1, 0, 2)) if scheme == "common-time" else Y[sl]
            est[s : s + step], cnt[s : s + step] = _stack_draws(kernel, X, Yp, gate)
        return est, cnt, b
    rng = np.random.default_rng(seed)
    est = np.full((b, n_ent), np.nan)
    cnt = np.zeros((b, n_ent), dtype=np.int64)
    b_eff = b
    for i in np.flatnonzero(valid):
        xa, ya, m = pairwise_complete(X[i], Y[i])
        blen = resolve_block_length(scheme, xa, ya, block_length)
        yb = resample_series(scheme, ya, n_resamples=b, block_length=blen, rng=rng)
        b_eff = min(b_eff, yb.shape[0])
        est[: yb.shape[0], i] = rows_chunked(
            kernel.rows, np.broadcast_to(xa, yb.shape), yb
        )
        cnt[:, i] = m
    return est[:b_eff], cnt[:b_eff], b_eff


def panel_test(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Kernel,
    *,
    null: str = "auto",
    how: str | None = None,
    has_time: bool = True,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Per-entity estimates on ``(N, T)`` arrays, aggregated and tested jointly.

    Returns
    -------
    (row, per_entity_estimates, per_entity_counts)
    """
    how = how or kernel.default_how
    gate = int(min_obs if min_obs is not None else kernel.min_obs)
    est, cnt = ragged_rows(kernel.rows, X, Y, gate)
    n_ent = X.shape[0]
    valid = np.isfinite(est)
    k = int(valid.sum())
    row = _base_row(kernel)
    row["estimator"] = f"{row['estimator']}; aggregate={how}"
    notes: list[str] = []
    n_obs = int(cnt[valid].sum())
    dropped = int(((cnt > 0) & ~valid).sum())
    if dropped:
        notes.append(
            f"{dropped} entit{'y' if dropped == 1 else 'ies'} below min_obs={gate} "
            "excluded from the aggregate."
        )
    agg = float(aggregate_estimates(est, cnt, kernel, how)) if k else float("nan")
    i2, _, _ = heterogeneity(est, cnt, kernel, how)
    row.update(
        estimate=agg,
        n_obs=n_obs,
        n_entities=k,
        coverage=k / n_ent if n_ent else float("nan"),
        heterogeneity=i2,
    )
    if k == 0:
        row.update(p_value=float("nan"), null_method="none")
        row["warnings"] = notes
        return row, est, cnt
    rows = _sample_rows(n_ent)
    serial = panel_serial(X, Y, rows[valid[rows]] if valid[rows].any() else rows)
    panel_ok = n_ent >= 2 and has_time
    if null == "common-time" and not has_time:
        raise ValueError(
            "null='common-time' permutes the shared date axis and needs a `time` "
            "column; pass time=... or a PanelFrame."
        )
    if null == "entity" and n_ent < 2:
        raise ValueError("null='entity' needs at least 2 entities.")
    scheme, w = _resolve(kernel, null, serial=serial, panel=panel_ok)
    notes += w
    if scheme in _CLOSED or scheme == "hac":
        pv = np.full(n_ent, np.nan)
        for i in np.flatnonzero(valid):
            xa, ya, _ = pairwise_complete(X[i], Y[i])
            if scheme == "hac":
                assert kernel.hac is not None
                pv[i] = kernel.hac(est[i], xa, ya)[0]
            else:
                assert kernel.closed_form is not None
                pv[i] = kernel.closed_form(est[i], xa, ya)[0]
        if np.isfinite(pv[valid]).all():
            p = stouffer(
                pv[valid], est[valid], cnt[valid], alternative=kernel.alternative
            )
            label = "hac" if scheme == "hac" else scheme
            notes.append(
                f"panel p-value: sqrt(n)-weighted Stouffer combination of per-entity "
                f"{label} p-values -- valid only if entities are cross-sectionally "
                "independent (no common shocks). null='common-time' drops that "
                "assumption."
            )
            row.update(p_value=p, null_method=f"{label}+stouffer")
            row["warnings"] = notes
            return row, est, cnt
        scheme = _fallback_scheme(serial)
        notes.append(
            f"a per-entity closed-form null for {kernel.name} refused (heavy ties); "
            f"fell back to a per-entity {scheme} null."
        )
    blen: int | None = None
    if scheme == "common-time":
        blen = (
            panel_block_length(X, Y, rows, block_length)
            if serial > SERIAL_THRESHOLD or block_length is not None
            else 1
        )
    elif scheme in _NEEDS_BLOCK and block_length is not None:
        blen = int(block_length)
    draws_e, draws_c, b_eff = panel_null_draws(
        kernel,
        X,
        Y,
        scheme=scheme,
        n_resamples=n_resamples,
        block_length=blen,
        seed=seed,
        gate=gate,
        valid=valid,
    )
    draws = aggregate_estimates(draws_e, draws_c, kernel, how)
    p = pvalue(agg, draws, alternative=kernel.alternative)
    if scheme in _PER_SERIES and panel_ok:
        notes.append(
            f"null={scheme!r} resamples each entity independently: invalid if "
            "entities share common shocks. null='common-time' is robust to them."
        )
    if scheme in {"block", "stationary", "shift"} and blen is None:
        notes.append("block_length: per-entity Politis-White (varies by entity).")
    row.update(
        p_value=p,
        null_method=scheme,
        n_resamples=b_eff,
        block_length=blen,
        seed=int(seed),
        _draws=draws,
    )
    row["warnings"] = notes
    return row, est, cnt


# --------------------------------------------------------------------------- #
# Panel, pooled
# --------------------------------------------------------------------------- #
def pooled_test(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Kernel,
    *,
    demean: str = "none",
    null: str = "auto",
    has_time: bool = True,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> dict[str, Any]:
    """One statistic on every complete pair of the panel, after ``demean``.

    The demeaning is applied to complete cells only and reported in
    ``transform``: pooled-raw and pooled-two-way-demeaned are different
    statistics. The null resamples the (demeaned) panel exactly as
    :func:`panel_test` does and recomputes the pooled statistic.
    """
    mask = np.isfinite(X) & np.isfinite(Y)
    Xd = demean_panel(X, mask, demean)
    Yd = demean_panel(Y, mask, demean)
    gate = int(min_obs if min_obs is not None else kernel.min_obs)
    xa, ya = Xd[mask], Yd[mask]
    n = int(mask.sum())
    row = _base_row(kernel)
    row["transform"] = f"pooled(demean={demean})"
    n_ent = X.shape[0]
    ent_obs = mask.any(axis=1)
    row.update(
        n_obs=n,
        n_entities=int(ent_obs.sum()),
        coverage=float(ent_obs.mean()) if n_ent else float("nan"),
    )
    notes: list[str] = []
    if n < gate:
        notes.append(f"n_obs={n} is below min_obs={gate}; estimate withheld (NaN).")
        row.update(estimate=float("nan"), p_value=float("nan"), null_method="none")
        row["warnings"] = notes
        return row
    est = float(kernel.rows(xa[None, :], ya[None, :])[0])
    row["estimate"] = est
    rows = _sample_rows(n_ent)
    serial = panel_serial(Xd, Yd, rows)
    panel_ok = n_ent >= 2 and has_time
    if null == "common-time" and not has_time:
        raise ValueError("null='common-time' needs a `time` column.")
    scheme, w = _resolve(kernel, null, serial=serial, panel=panel_ok)
    notes += w
    if n_ent >= 2 and demean == "none":
        notes.append(
            "pooled without demeaning mixes within-entity and cross-sectional "
            "variation; compare demean='two-way' or by='entity'."
        )
    if scheme in _CLOSED:
        assert kernel.closed_form is not None
        p, label = kernel.closed_form(est, xa, ya)
        if np.isfinite(p):
            if panel_ok:
                notes.append(
                    "closed-form null treats all pooled pairs as independent; "
                    "invalid under common shocks or serial dependence."
                )
            row.update(p_value=p, null_method=label)
            row["warnings"] = notes
            return row
        scheme = _fallback_scheme(serial)
    b = int(n_resamples)
    blen: int | None = None
    if scheme in {"common-time", "entity"}:
        if scheme == "common-time":
            blen = (
                panel_block_length(Xd, Yd, rows, block_length)
                if serial > SERIAL_THRESHOLD or block_length is not None
                else 1
            )
            idx = common_time_indices(
                np.arange(X.shape[1]), n_resamples=b, seed=seed, block=blen
            )
        else:
            idx = entity_permutation_indices(n_ent, n_resamples=b, seed=seed)
        draws = np.full(b, np.nan)
        for s in range(b):
            Yp = Yd[:, idx[s]] if scheme == "common-time" else Yd[idx[s]]
            m2 = np.isfinite(Xd) & np.isfinite(Yp)
            if m2.sum() >= gate:
                draws[s] = kernel.rows(Xd[m2][None, :], Yp[m2][None, :])[0]
        b_eff = b
    else:
        rng = np.random.default_rng(seed)
        parts_x, parts_y = [], []
        b_eff = b
        for i in np.flatnonzero(ent_obs):
            xi_, yi_, m = pairwise_complete(Xd[i], Yd[i])
            if m < 4:
                continue
            bl = resolve_block_length(scheme, xi_, yi_, block_length)
            yb = resample_series(scheme, yi_, n_resamples=b, block_length=bl, rng=rng)
            b_eff = min(b_eff, yb.shape[0])
            parts_x.append(xi_)
            parts_y.append(yb)
        xcat = np.concatenate(parts_x)
        ycat = np.concatenate([p_[:b_eff] for p_ in parts_y], axis=1)
        draws = rows_chunked(kernel.rows, np.broadcast_to(xcat, ycat.shape), ycat)
        if block_length is not None and scheme in _NEEDS_BLOCK:
            blen = int(block_length)
    p = pvalue(est, draws, alternative=kernel.alternative)
    row.update(
        p_value=p,
        null_method=scheme,
        n_resamples=b_eff,
        block_length=blen,
        seed=int(seed),
        _draws=draws,
    )
    row["warnings"] = notes
    return row


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _expand_methods(method: str | Sequence[str]) -> list[str]:
    ms = [method] if isinstance(method, str) else list(method)
    out: list[str] = []
    for m in ms:
        out += ["tail_lower", "tail_upper"] if m == "tail_dependence" else [m]
    return out


def _check_by(by: str | None) -> str:
    if by in (None, "none", "pooled"):
        return "pooled"
    if by == "entity":
        return "entity"
    raise ValueError(f"unknown `by` {by!r}; expected 'entity' or 'pooled'.")


def run_pair(
    pa: PanelArrays,
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Kernel,
    *,
    by: str,
    null: str,
    how: str | None,
    demean: str,
    n_resamples: int,
    block_length: int | None,
    seed: int,
    min_obs: int | None,
) -> dict[str, Any]:
    """Dispatch one prepared ``(N, T)`` pair to the series / panel / pooled test."""
    if X.shape[0] == 1:
        row = series_test(
            X[0],
            Y[0],
            kernel,
            null=null,
            n_resamples=n_resamples,
            block_length=block_length,
            seed=seed,
            min_obs=min_obs,
        )
    elif by == "entity":
        row, _, _ = panel_test(
            X,
            Y,
            kernel,
            null=null,
            how=how,
            has_time=pa.has_time,
            n_resamples=n_resamples,
            block_length=block_length,
            seed=seed,
            min_obs=min_obs,
        )
    else:
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
    if kernel.name == "gcmi":
        row["warnings"] = [*row["warnings"], _GCMI_NOTE]
    return row


def dependence(
    df: Any,
    x: str,
    y: str,
    *,
    method: str | Sequence[str] = "xi",
    by: str | None = "entity",
    null: str = "auto",
    entity: str | None = None,
    time: str | None = None,
    how: str | None = None,
    lag: int = 0,
    devol: int | None = None,
    demean: str = "none",
    q: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Measure and test the dependence of ``y`` on ``x`` -- with an honest null.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
        The data. Without an entity column it is one series; without a time
        column its row order is the time order.
    x, y : str
        Column names. For directed methods (``xi``) the result is ``x -> y``:
        "is ``y`` a function of ``x``".
    method : str or sequence of str, default="xi"
        ``"xi"``, ``"spearman"``, ``"pearson"``, ``"kendall"``,
        ``"hoeffding"``, ``"dcor"``, ``"gcmi"``, ``"hsic"``, ``"tail_lower"``,
        ``"tail_upper"`` or ``"tail_dependence"`` (both tails). A sequence
        returns one row per method.
    by : {"entity", "pooled"}, default="entity"
        ``"entity"``: per-entity estimates, aggregated (``how``) into one number
        with a heterogeneity statistic. ``"pooled"``: one statistic on every
        complete pair after ``demean``.
    null : str, default="auto"
        ``"auto"`` applies the null policy table: common-time for a panel with a
        time column; for one series, the closed form when the lag-1 rank
        autocorrelation pre-check passes (``|rho| <= 0.2``) and a block
        permutation when it fails. Explicit choices: ``"asymptotic"`` (alias
        ``"iid"``), ``"permutation"``, ``"block"``, ``"stationary"``,
        ``"shift"``, ``"iaaft"``, ``"phase"``, ``"common-time"``, ``"entity"``.
        An i.i.d. null on serially dependent data warns
        (:class:`~panelary.depend.SerialDependenceWarning`).
    entity, time : str, optional
        Panel keys for a bare frame (a PanelFrame supplies its own).
    how : {"fisher", "precision", "mean", "median"}, optional
        Per-entity aggregation; defaults to the method's (Fisher for
        correlation-like statistics, mean for GCMI and tail statistics).
    lag : int, default=0
        Pair ``x[t - lag]`` with ``y[t]`` on the shared date axis.
    devol : int, optional
        If given, **also** report the statistic after dividing both series by a
        causal rolling volatility over the previous ``devol`` dates -- two rows,
        ``transform="none"`` and ``transform="devol(window=...)"`` -- so
        volatility clustering cannot pass for dependence unnoticed.
    demean : {"none", "entity", "time", "two-way"}, default="none"
        Demeaning for ``by="pooled"``; reported in ``transform``.
    q : float, default=0.05
        Tail probability for the tail statistics.
    n_resamples : int, default=199
        Resamples for resampling nulls (smallest p-value ``1 / (B + 1)``).
    block_length : int, optional
        Block length for block / shift / common-time nulls; Politis-White
        automatic selection otherwise (reported in ``block_length``).
    seed : int, default=0
        Seed for every random draw. Same seed, same bits.
    min_obs : int, optional
        Override the method's minimum complete pairs per estimate.

    Returns
    -------
    polars.DataFrame
        Columns ``x``, ``y`` and the fixed result schema (``estimate``,
        ``p_value``, ``p_value_adj``, ``method``, ``estimator``,
        ``null_method``, ``n_resamples``, ``block_length``, ``direction``,
        ``lag``, ``n_obs``, ``n_entities``, ``coverage``, ``heterogeneity``,
        ``transform``, ``approximate``, ``seed``, ``warnings``).

    Examples
    --------
    >>> import numpy as np, polars as pl
    >>> rng = np.random.default_rng(0)
    >>> a = rng.standard_normal(300)
    >>> df = pl.DataFrame({"a": a, "b": a**2 + 0.1 * rng.standard_normal(300)})
    >>> out = dependence(df, "a", "b", method="xi")
    >>> bool(out["p_value"][0] < 0.001)
    True
    """
    by_ = _check_by(by)
    methods = _expand_methods(method)
    pa = extract(df, [x, y], entity=entity, time=time)
    X, Y = pa.dense(x), pa.dense(y)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = [("none", X, Y)]
    if devol is not None:
        variants.append(
            (
                f"devol(window={int(devol)})",
                devolatilise(X, devol),
                devolatilise(Y, devol),
            )
        )
    rows: list[dict[str, Any]] = []
    for tname, Xv, Yv in variants:
        Xl, Yl = _shift_pair(Xv, Yv, lag)
        for m in methods:
            kernel = get_kernel(m, q=q)
            row = run_pair(
                pa,
                Xl,
                Yl,
                kernel,
                by=by_,
                null=null,
                how=how,
                demean=demean,
                n_resamples=n_resamples,
                block_length=block_length,
                seed=seed,
                min_obs=min_obs,
            )
            row.update(x=x, y=y, lag=int(lag))
            if tname != "none":
                base = row.get("transform") or "none"
                row["transform"] = tname if base == "none" else f"{base}; {tname}"
            rows.append(row)
    return result_frame(rows, keys={"x": pl.Utf8, "y": pl.Utf8})


def independence_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "xi",
    null: str = "auto",
    q: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Array-level test of one pair of series; a one-row result frame.

    The same engine as :func:`dependence` for numpy inputs (``x`` and ``y`` are
    one series each, in time order; non-finite rows dropped).

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length.
    method, null, q, n_resamples, block_length, seed, min_obs
        As in :func:`dependence`.

    Returns
    -------
    polars.DataFrame
        One row on the fixed result schema.
    """
    kernel = get_kernel(method, q=q)
    row = series_test(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        kernel,
        null=null,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        min_obs=min_obs,
    )
    if kernel.name == "gcmi":
        row["warnings"] = [*row["warnings"], _GCMI_NOTE]
    return result_frame([row])
