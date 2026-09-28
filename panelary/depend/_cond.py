"""Conditional dependence: GCM, CODEC and FOCI.

:func:`gcm` -- the generalised covariance measure (Shah & Peters, 2020) --
is the cheapest defensible conditional-independence test: regress ``x`` and
``y`` on ``z``, and test whether the residual product has mean zero,
``T = sqrt(n) mean(rx ry) / sd(rx ry) -> N(0, 1)``. Two OLS fits and a closed
form. For serially dependent data the standard deviation becomes a Newey-West
long-run standard deviation, which is what makes it panel-valid (and why it is
preferred here to kernel CI tests with more power on paper).

The regression is **linear** (``regressor="ols"``): GCM is valid when the
conditional means ``E[x | z]``, ``E[y | z]`` are linear in the supplied ``z``
columns; supply nonlinear basis columns in ``z`` otherwise. That is the price
of a numpy-only, closed-form test.

:func:`codec` is Azadkia & Chatterjee's (2021) conditional dependence
coefficient ``T(Y, Z | X)`` -- xi's conditional sibling, built on nearest
neighbours (class D in the build contract: exact blocked ``O(n^2)`` search,
capped at ``max_n`` with a seeded subsample). :func:`foci` is their greedy
forward selection on it, stopping when no candidate adds information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from panelary.depend._engine import (
    DEFAULT_RESAMPLES,
    _check_by,
    run_pair,
)
from panelary.depend._frame import _resolve, extract, result_frame
from panelary.depend._kernels import Kernel
from panelary.depend._ranks import ranks
from panelary.depend._special import norm_sf
from panelary.econ._common import auto_bandwidth, newey_west_scalar, ols

__all__ = ["GCMResult", "codec", "conditional_dependence", "foci", "gcm"]

_REGRESSORS = frozenset({"ols"})


# --------------------------------------------------------------------------- #
# GCM
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GCMResult:
    """Generalised covariance measure test.

    Attributes
    ----------
    statistic : float
        ``T``, asymptotically N(0, 1) under conditional independence.
    p_value : float
        Two-sided.
    residual_corr : float
        Correlation of the two residual series (an effect size).
    n_obs : int
    hac_lags : int or None
        Newey-West lags used (``None``: i.i.d. standard deviation).
    """

    statistic: float
    p_value: float
    residual_corr: float
    n_obs: int
    hac_lags: int | None


def _design(z: np.ndarray | None, n: int) -> np.ndarray:
    if z is None:
        return np.ones((n, 1))
    za = np.asarray(z, dtype=np.float64)
    za = za[:, None] if za.ndim == 1 else za
    return np.column_stack([np.ones(n), za])


def residualise(v: np.ndarray, z: np.ndarray | None) -> np.ndarray:
    """OLS residuals of ``v`` on ``[1, z]`` (``econ._common.ols``)."""
    va = np.asarray(v, dtype=np.float64)
    return ols(_design(z, va.shape[0]), va)[1]


def _gcm_stat(rx: np.ndarray, ry: np.ndarray, hac_lags: int | None) -> float:
    r = rx * ry
    n = r.size
    if hac_lags is None:
        sd = float(r.std())
        return math.sqrt(n) * float(r.mean()) / sd if sd > 0 else float("nan")
    var = newey_west_scalar(r, int(hac_lags))
    return float(r.mean()) / math.sqrt(var) if var > 0 else float("nan")


def _two_sided(t: float) -> float:
    return (
        float(min(1.0, 2.0 * float(norm_sf(abs(t)))))
        if np.isfinite(t)
        else float("nan")
    )


def gcm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    *,
    regressor: str = "ols",
    hac_lags: int | str | None = None,
) -> GCMResult:
    """Generalised covariance measure test of ``x _||_ y | z``.

    Parameters
    ----------
    x, y : array_like
        1-D series.
    z : array_like, optional
        ``(n,)`` or ``(n, d)`` conditioning set (an intercept is always added).
        ``None`` tests plain (linear-covariance) independence.
    regressor : {"ols"}, default="ols"
    hac_lags : int or "auto", optional
        Newey-West lags for serially dependent data; ``"auto"`` uses
        ``floor(4 (n/100)^(2/9))``. ``None``: i.i.d. standard deviation.

    Returns
    -------
    GCMResult

    References
    ----------
    Shah, R. D. & Peters, J. (2020). The hardness of conditional independence
    testing and the generalised covariance measure. *Ann. Statist.* 48(3).
    """
    if regressor not in _REGRESSORS:
        raise ValueError(f"unknown `regressor` {regressor!r}; only 'ols' is available.")
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    za = None if z is None else np.asarray(z, dtype=np.float64)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if za is not None:
        za = za[:, None] if za.ndim == 1 else za
        keep &= np.isfinite(za).all(axis=1)
        za = za[keep]
    xa, ya = xa[keep], ya[keep]
    n = xa.size
    lags: int | None
    if hac_lags == "auto":
        lags = auto_bandwidth(n)
    elif hac_lags is None:
        lags = None
    else:
        lags = int(hac_lags)
    if n < (za.shape[1] if za is not None else 0) + 4:
        nan = float("nan")
        return GCMResult(nan, nan, nan, n, lags)
    rx = residualise(xa, za)
    ry = residualise(ya, za)
    t = _gcm_stat(rx, ry, lags)
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    rc = float(rx @ ry) / den if den > 0 else float("nan")
    return GCMResult(t, _two_sided(t), rc, n, lags)


def _gcm_rows(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    r = np.atleast_2d(rx) * np.atleast_2d(ry)
    m = r.shape[-1]
    sd = r.std(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            sd > 0, math.sqrt(m) * r.mean(axis=-1) / np.where(sd > 0, sd, 1.0), np.nan
        )


def _gcm_cf(est: float, rx: np.ndarray, ry: np.ndarray) -> tuple[float, str]:
    return _two_sided(est), "asymptotic"


def _gcm_hac(est: float, rx: np.ndarray, ry: np.ndarray) -> tuple[float, str]:
    t = _gcm_stat(rx, ry, auto_bandwidth(rx.size))
    return _two_sided(t), "hac"


#: Engine kernel on residual pairs: the GCM statistic, N(0,1) / HAC closed forms.
GCM_KERNEL = Kernel(
    "gcm",
    _gcm_rows,
    "two-sided",
    False,
    30,
    _gcm_cf,
    "mean",
    lambda n: np.ones_like(np.asarray(n, dtype=np.float64)),
    estimator="gcm T (ols residuals)",
    policy="gcm",
    hac=_gcm_hac,
)


# --------------------------------------------------------------------------- #
# CODEC / FOCI
# --------------------------------------------------------------------------- #
def _nearest(P: np.ndarray, *, block: int = 1024) -> np.ndarray:
    """Index of each row's nearest *other* row (Euclidean); ties -> lowest index."""
    n = P.shape[0]
    sq = (P * P).sum(axis=1)
    out = np.empty(n, dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * P[s:e] @ P.T
        d2[np.arange(e - s), np.arange(s, e)] = np.inf
        out[s:e] = np.argmin(d2, axis=1)
    return out


def _codec_from(
    r: np.ndarray, ell: np.ndarray, x: np.ndarray | None, xz: np.ndarray
) -> float:
    n = r.size
    m_idx = _nearest(xz)
    if x is None:
        num = float((n * np.minimum(r, r[m_idx]) - ell * ell).sum())
        den = float((ell * (n - ell)).sum())
    else:
        n_idx = _nearest(x)
        num = float((np.minimum(r, r[m_idx]) - np.minimum(r, r[n_idx])).sum())
        den = float((r - np.minimum(r, r[n_idx])).sum())
    return num / den if den > 0 else float("nan")


def _as_mat(v: np.ndarray | None) -> np.ndarray | None:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def codec(
    y: np.ndarray,
    z: np.ndarray,
    x: np.ndarray | None = None,
    *,
    max_n: int = 5000,
    seed: int = 0,
) -> float:
    """Azadkia-Chatterjee conditional dependence coefficient ``T(Y, Z | X)``.

    ``0`` iff ``Y`` is conditionally independent of ``Z`` given ``X``; ``1`` iff
    ``Y`` is a measurable function of ``(X, Z)``. With ``x=None`` it is the
    unconditional ``T(Y, Z)``. Scale matters for nearest neighbours: pass
    comparably scaled columns (:func:`foci` standardises).

    Nearest neighbours are exact (blocked ``O(n^2)``); ties are broken by the
    lowest index (deterministic, unlike the paper's random tie-breaking).
    Above ``max_n`` rows a seeded subsample is used.

    References
    ----------
    Azadkia, M. & Chatterjee, S. (2021). A simple measure of conditional
    dependence. *Ann. Statist.* 49(6).
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    za = _as_mat(z)
    xa = _as_mat(x)
    assert za is not None
    keep = np.isfinite(ya) & np.isfinite(za).all(axis=1)
    if xa is not None:
        keep &= np.isfinite(xa).all(axis=1)
    idx = np.flatnonzero(keep)
    if idx.size > max_n:
        idx = np.sort(
            np.random.default_rng(seed).choice(idx, size=int(max_n), replace=False)
        )
    if idx.size < 3:
        return float("nan")
    ys = ya[idx]
    r = ranks(ys, method="max").astype(np.float64)
    ell = (idx.size + 1 - ranks(ys, method="min")).astype(np.float64)
    zs = za[idx]
    xs = None if xa is None else xa[idx]
    xz = zs if xs is None else np.column_stack([xs, zs])
    return _codec_from(r, ell, xs, xz)


def _standardise(A: np.ndarray, how: str) -> np.ndarray:
    if how == "none":
        return A
    if how == "rank":
        return ranks(A, axis=0) / A.shape[0]
    if how == "scale":
        sd = A.std(axis=0)
        return (A - A.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    raise ValueError(
        f"unknown `standardize` {how!r}; expected 'scale', 'rank' or 'none'."
    )


def foci(
    df: Any,
    target: str,
    features: Sequence[str] | None = None,
    *,
    k: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    standardize: str = "scale",
    max_n: int = 2000,
    seed: int = 0,
) -> list[str]:
    """Feature ordering by conditional dependence (Azadkia & Chatterjee, 2021).

    Greedy forward selection: add the feature ``j`` maximising
    ``T(target, X_j | X_selected)`` and stop when no candidate has a positive
    value (or after ``k``). Computed on the pooled complete rows, capped at
    ``max_n`` by a seeded subsample -- each step is ``p`` exact nearest
    neighbour searches, ``O(p n^2)``.

    **Leakage**: like :func:`~panelary.depend.feature_screen` this is a
    selection -- run it inside each training fold, never on the full panel.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
    target : str
    features : sequence of str, optional
        Candidates (default: numeric non-key, non-target columns).
    k : int, optional
        Maximum number of features.
    entity, time : str, optional
        Keys for a bare frame (excluded from the default candidates).
    standardize : {"scale", "rank", "none"}, default="scale"
        Column scaling before nearest-neighbour search.
    max_n : int, default=2000
    seed : int, default=0

    Returns
    -------
    list of str
        Selected features in selection order.
    """
    frame, ent, tme = _resolve(df, entity, time)
    keys = {c for c in (ent, tme) if c}
    if features is None:
        feats = [
            c
            for c, dt in frame.schema.items()
            if c not in keys | {target} and dt.is_numeric() and dt != pl.Boolean
        ]
    else:
        feats = [f for f in dict.fromkeys(features) if f != target]
    pa = extract(frame, [*feats, target], entity=ent, time=tme)
    A = np.column_stack([pa.values[f] for f in feats])
    y = pa.values[target]
    keep = np.isfinite(y) & np.isfinite(A).all(axis=1)
    idx = np.flatnonzero(keep)
    if idx.size > max_n:
        idx = np.sort(
            np.random.default_rng(seed).choice(idx, size=int(max_n), replace=False)
        )
    A = _standardise(A[idx], standardize)
    ys = y[idx]
    n = ys.size
    if n < 3:
        return []
    r = ranks(ys, method="max").astype(np.float64)
    ell = (n + 1 - ranks(ys, method="min")).astype(np.float64)
    selected: list[int] = []
    remaining = list(range(len(feats)))
    limit = len(feats) if k is None else int(k)
    while remaining and len(selected) < limit:
        base = A[:, selected] if selected else None
        scores = []
        for j in remaining:
            xz = A[:, [*selected, j]]
            scores.append(_codec_from(r, ell, base, xz))
        best = int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1
        if best < 0 or not scores[best] > 0:
            break
        selected.append(remaining.pop(best))
    return [feats[j] for j in selected]


# --------------------------------------------------------------------------- #
# Frame-level conditional dependence
# --------------------------------------------------------------------------- #
def conditional_dependence(
    df: Any,
    x: str,
    y: str,
    *,
    given: Sequence[str],
    method: str = "gcm",
    by: str | None = "entity",
    null: str = "auto",
    entity: str | None = None,
    time: str | None = None,
    how: str | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Test ``x _||_ y | given`` on a panel.

    ``method="gcm"`` (default): ``x`` and ``y`` are residualised on ``given``
    by OLS **within each entity** (``by="entity"``) or on the pooled panel
    (``by="pooled"``), and the residual pairs go through the same engine and
    null policy as :func:`~panelary.depend.dependence`: N(0, 1) for serially
    independent data, a Newey-West (HAC) studentisation when the serial
    pre-check fails, and ``"common-time"`` for a panel. ``method="gcmi"``:
    conditional Gaussian-copula MI with its chi-square closed form (single
    series / pooled only).

    Returns
    -------
    polars.DataFrame
        ``x``, ``y`` and the fixed schema; ``transform`` records the
        conditioning set.
    """
    given = list(dict.fromkeys(given))
    if not given:
        raise ValueError("`given` must name at least one conditioning column.")
    by_ = _check_by(by)
    pa = extract(df, [x, y, *given], entity=entity, time=time)
    label = f"residualised on {given}"
    if method == "gcm":
        X = pa.dense(x)
        Y = pa.dense(y)
        Z = np.stack([pa.dense(g) for g in given], axis=-1)  # (N, T, d)
        RX = np.full(X.shape, np.nan)
        RY = np.full(Y.shape, np.nan)
        if by_ == "entity" or X.shape[0] == 1:
            for i in range(X.shape[0]):
                ok = (
                    np.isfinite(X[i])
                    & np.isfinite(Y[i])
                    & np.isfinite(Z[i]).all(axis=1)
                )
                if ok.sum() >= len(given) + 4:
                    RX[i, ok] = residualise(X[i, ok], Z[i, ok])
                    RY[i, ok] = residualise(Y[i, ok], Z[i, ok])
        else:
            ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z).all(axis=-1)
            RX[ok] = residualise(X[ok], Z[ok])
            RY[ok] = residualise(Y[ok], Z[ok])
            label = f"pooled residualised on {given}"
        row = run_pair(
            pa,
            RX,
            RY,
            GCM_KERNEL,
            by="entity" if by_ == "entity" else "pooled",
            null=null,
            how=how,
            demean="none",
            n_resamples=n_resamples,
            block_length=block_length,
            seed=seed,
            min_obs=min_obs,
        )
        row["transform"] = (
            label
            if row.get("transform") in (None, "none")
            else f"{row['transform']}; {label}"
        )
    elif method == "gcmi":
        from panelary.depend._info import gcmi_conditional, gcmi_pvalue

        cols = np.column_stack([pa.values[c] for c in (x, y, *given)])
        cols = cols[np.isfinite(cols).all(axis=1)]
        xa, ya, za = cols[:, 0], cols[:, 1], cols[:, 2:]
        n = cols.shape[0]
        est = gcmi_conditional(xa, ya, za) if n >= len(given) + 5 else float("nan")
        row = {
            "estimate": est,
            "p_value": gcmi_pvalue(xa, ya, za) if np.isfinite(est) else float("nan"),
            "method": "gcmi",
            "estimator": "conditional gaussian-copula MI (nats, bias-corrected)",
            "null_method": "asymptotic",
            "direction": "symmetric",
            "n_obs": n,
            "n_entities": pa.n_entities,
            "transform": f"pooled; conditioned on {given}",
            "approximate": False,
            "warnings": (
                [
                    "pooled chi-square null: assumes independent rows (no serial or common-shock dependence)."
                ]
                if pa.n_entities > 1 or n > 0
                else []
            ),
        }
    else:
        raise ValueError(
            f"unknown conditional `method` {method!r}; expected 'gcm' or 'gcmi'."
        )
    row.update(x=x, y=y, lag=0)
    return result_frame([row], keys={"x": pl.Utf8, "y": pl.Utf8})
