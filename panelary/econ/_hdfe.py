"""High-dimensional fixed effects (HDFE) by alternating projections.

This is the "fast" headline of the ``econ`` package: absorb an arbitrary number
of high-cardinality fixed-effect dimensions (firm, date, industry x year, ...)
**without ever materialising a dummy matrix**, then run OLS on the partialled-out
data. The equivalence is Frisch-Waugh-Lovell: regressing the residualised ``y``
on the residualised ``X`` reproduces the coefficients (and, block-wise, the
standard errors) of the full dummy-variable regression exactly.

Method
------
Absorbing a set of fixed effects means projecting onto the orthogonal complement
of the span of the dummy matrices ``D_1, ..., D_G``. Each individual projection
``M_g`` is just *demeaning within the groups of dimension g* -- a Polars
``group_by(...).mean()``. The joint projection ``M = M_1 M_2 ... M_G`` is
obtained by **alternating projections** (von Neumann / Halperin; Gaure 2013;
the engine behind ``reghdfe`` and ``fixest``): sweep the dimensions in turn,
demeaning the current residual each time, until the largest group mean removed
in a full sweep falls below a tolerance. The sweep is linear in the number of
rows, so this scales to millions of levels.

Because each sweep removes a *group mean*, accumulating the means removed for
each level gives a per-level **offset** ``alpha_g[level]``. Those offsets are
the learned parameters: a fitted :class:`HDFETransformer` freezes them and
applies them at ``transform`` time, so the partialled-out residual columns are
leak-controlled features usable inside cross-validation.

Standard errors
---------------
* ``"classical"`` -- homoskedastic ``sigma^2 (X'X)^-1``.
* ``"robust"`` -- Huber-White HC1.
* ``"cluster"`` -- one-way clustered (Liang-Zeger) with the usual finite-sample
  factor ``G/(G-1) * (n-1)/(n-k)``.
* two-way clustered -- Cameron-Gelbach-Miller: ``V1 + V2 - V12``, optionally
  eigenvalue-clipped to stay positive semi-definite.
* ``"driscoll-kraay"`` -- Driscoll-Kraay: cross-sectionally aggregate the scores
  by time, then apply a Bartlett/Newey-West HAC over the resulting time series.
  Robust to arbitrary cross-sectional dependence.

References
----------
Frisch, R. & Waugh, F. (1933); Lovell, M. (1963).
Gaure, S. (2013). *OLS with multiple high dimensional category variables*.
Correia, S. (2016). *reghdfe: Linear models with high-dimensional fixed effects*.
Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). *Robust inference with
multiway clustering*, JBES 29(2).
Driscoll, J. & Kraay, A. (1998). *Consistent covariance matrix estimation with
spatially dependent panel data*, ReStat 80(4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.econ._common import (
    auto_bandwidth,
    chi2_sf,
    factorize,
    group_mean,
    newey_west_lrv,
    pinv_sym,
    t_sf,
)

__all__ = [
    "demean",
    "hdfe",
    "HDFEResult",
    "HDFETransformer",
]

_VCOV_KINDS = ("classical", "robust", "cluster", "driscoll-kraay")


# --------------------------------------------------------------------------- #
# Alternating projections
# --------------------------------------------------------------------------- #
def demean(
    values: np.ndarray,
    codes: Sequence[np.ndarray],
    n_levels: Sequence[int],
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, list[np.ndarray], int, float]:
    """Absorb ``len(codes)`` fixed-effect dimensions by alternating projections.

    Parameters
    ----------
    values : ndarray, shape (n,) or (n, k)
        Columns to residualise.
    codes : sequence of ndarray
        One integer code array per absorbed dimension, each of length ``n``.
    n_levels : sequence of int
        Number of levels in each dimension (``codes[g].max() + 1`` or more).
    tol : float, default=1e-12
        Convergence tolerance, applied **relative** to the scale of ``values``:
        a sweep converges when the largest absolute group mean it removed is
        below ``tol * (1 + max|values|)``.
    max_iter : int, default=10000
        Maximum number of sweeps.

    Returns
    -------
    resid : ndarray, same shape as ``values``
        The partialled-out data.
    offsets : list of ndarray
        ``offsets[g]`` has shape ``(n_levels[g], k)``: the accumulated mean
        removed for each level of dimension ``g``. Applying
        ``values - sum_g offsets[g][codes[g]]`` reproduces ``resid`` exactly.
        A level with **no observation** (possible because ``n_levels`` may
        exceed the levels present) has no identified offset and is ``NaN``,
        so anything built from it -- e.g. an imputation estimator's
        ``Y(0)`` for a period in which every unit is treated -- fails closed
        instead of silently using an offset of zero.
    n_iter : int
        Number of sweeps performed.
    max_dev : float
        The largest group mean removed in the final sweep.

    Notes
    -----
    With a single dimension the projection is exact after one sweep. With two or
    more the convergence is linear in the "angle" between the dummy spaces; the
    tolerance above is what makes the resulting coefficients match a dense
    dummy-variable OLS to machine precision.

    With two or more dimensions, even observed levels are identified only up
    to a constant within each connected component of the levels'
    co-occurrence graph: ``offsets[0][a] + offsets[1][b]`` is identified only
    when ``a`` and ``b`` fall in the same component. That is not checked here.
    """
    arr = np.asarray(values, dtype=float)
    single = arr.ndim == 1
    resid = (arr[:, None] if single else arr).copy()
    n_dims = len(codes)
    if n_dims != len(n_levels):
        raise ValueError("`codes` and `n_levels` must have the same length.")
    offsets = [np.zeros((int(g), resid.shape[1])) for g in n_levels]
    if n_dims == 0:
        return (resid[:, 0] if single else resid), offsets, 0, 0.0

    scale = float(np.abs(resid).max()) if resid.size else 0.0
    threshold = tol * (1.0 + scale)
    max_dev = np.inf
    n_iter = 0
    for sweep in range(1, max_iter + 1):
        n_iter = sweep
        max_dev = 0.0
        for g in range(n_dims):
            means = group_mean(resid, codes[g], int(n_levels[g]))
            resid -= means[codes[g]]
            offsets[g] += means
            dev = float(np.abs(means).max()) if means.size else 0.0
            if dev > max_dev:
                max_dev = dev
        # A single dimension is an exact projection: one sweep is enough, so
        # skip the confirmation pass over the data.
        if max_dev <= threshold or n_dims == 1:
            if n_dims == 1:
                max_dev = 0.0
            break
    # `group_mean` reports an empty group's mean as 0, so a level nobody
    # observed would otherwise come back with an offset of exactly 0 -- a
    # number, where the data identify none. No row carries such a level, so
    # `resid` and every observed level's offset are untouched by this.
    for g in range(n_dims):
        seen = np.bincount(codes[g], minlength=int(n_levels[g])) > 0
        offsets[g][~seen] = np.nan
    return (resid[:, 0] if single else resid), offsets, n_iter, max_dev


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class HDFEResult:
    """Fitted high-dimensional fixed-effects regression.

    Attributes
    ----------
    names : list of str
        Regressor names, in the order of :attr:`params`.
    params : ndarray, shape (k,)
        Coefficients on the (partialled-out) regressors.
    std_errors : ndarray, shape (k,)
    tstats, pvalues : ndarray, shape (k,)
        ``t = b / se``; p-values use the t distribution with :attr:`df_resid`
        degrees of freedom (normal for HAC-type covariances would be
        asymptotically equivalent; the t is the conservative choice).
    vcov : ndarray, shape (k, k)
    resid : ndarray, shape (n,)
        Regression residuals (already free of the fixed effects).
    resid_y : ndarray, shape (n,)
        Partialled-out dependent variable.
    resid_x : ndarray, shape (n, k)
        Partialled-out regressors -- the leak-controlled *features*.
    nobs : int
    df_absorbed : int
        Parameters consumed by the fixed effects,
        ``sum_g n_levels[g] - (n_dims - 1)`` (the shared intercept is counted
        once).
    df_resid : int
        ``nobs - k - df_absorbed``.
    rss, tss_within : float
        Residual and within (post-absorption) total sum of squares.
    r2_within : float
        ``1 - rss / tss_within``.
    vcov_type : str
    converged : bool
    n_iter : int
    """

    names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    vcov: np.ndarray
    resid: np.ndarray
    resid_y: np.ndarray
    resid_x: np.ndarray
    nobs: int
    df_absorbed: int
    df_resid: int
    rss: float
    tss_within: float
    r2_within: float
    vcov_type: str
    converged: bool
    n_iter: int
    fe_offsets: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    fe_levels: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def summary(self) -> pl.DataFrame:
        """Coefficient table as a Polars DataFrame."""
        return pl.DataFrame(
            {
                "term": self.names,
                "estimate": np.asarray(self.params, dtype=float),
                "std_error": np.asarray(self.std_errors, dtype=float),
                "t_stat": np.asarray(self.tstats, dtype=float),
                "p_value": np.asarray(self.pvalues, dtype=float),
            }
        )

    def conf_int(self, alpha: float = 0.05) -> pl.DataFrame:
        """Two-sided Wald confidence intervals at level ``1 - alpha``."""
        from panelary.econ._common import norm_ppf

        z = float(norm_ppf(1.0 - alpha / 2.0))
        lo = self.params - z * self.std_errors
        hi = self.params + z * self.std_errors
        return pl.DataFrame({"term": self.names, "lower": lo, "upper": hi})

    def wald_test(self, restrictions: Sequence[str] | None = None) -> dict[str, float]:
        """Joint Wald test that the named coefficients are all zero.

        Parameters
        ----------
        restrictions : sequence of str, optional
            Terms to test. Defaults to every regressor.

        Returns
        -------
        dict
            ``{"statistic": ..., "df": ..., "p_value": ...}``; the statistic is
            asymptotically chi-squared.
        """
        terms = list(restrictions) if restrictions is not None else list(self.names)
        idx = [self.names.index(t) for t in terms]
        b = self.params[idx]
        v = self.vcov[np.ix_(idx, idx)]
        stat = float(b @ pinv_sym(v) @ b)
        return {
            "statistic": stat,
            "df": float(len(idx)),
            "p_value": float(chi2_sf(stat, len(idx))),
        }


# --------------------------------------------------------------------------- #
# Covariance estimators
# --------------------------------------------------------------------------- #
def _cluster_meat(scores: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """``sum_g (sum_{i in g} s_i)(sum_{i in g} s_i)'`` for score rows ``s_i``."""
    agg = np.zeros((n_groups, scores.shape[1]))
    for j in range(scores.shape[1]):
        agg[:, j] = np.bincount(codes, weights=scores[:, j], minlength=n_groups)
    return agg.T @ agg


def _psd_clip(v: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone (CGM's eigenvalue fix)."""
    v = 0.5 * (v + v.T)
    w, q = np.linalg.eigh(v)
    if (w >= -1e-12 * max(1.0, float(np.abs(w).max()))).all():
        return v
    w = np.clip(w, 0.0, None)
    return (q * w) @ q.T


def _hdfe_vcov(
    resid_x: np.ndarray,
    resid: np.ndarray,
    xtx_inv: np.ndarray,
    *,
    vcov: str,
    cluster_codes: list[np.ndarray],
    cluster_levels: list[int],
    time_codes: np.ndarray | None,
    n_times: int,
    lags: int,
    df_resid: int,
    n_params: int,
    dof_correction: bool,
    psd_clip: bool,
) -> tuple[np.ndarray, str]:
    """Dispatch to the requested covariance estimator. Returns ``(V, label)``."""
    n = resid_x.shape[0]
    scores = resid_x * resid[:, None]

    if vcov == "classical":
        sigma2 = float(resid @ resid) / max(df_resid, 1)
        return sigma2 * xtx_inv, "classical"

    if vcov == "robust":
        meat = scores.T @ scores
        scale = n / max(df_resid, 1) if dof_correction else 1.0
        return scale * (xtx_inv @ meat @ xtx_inv), "robust (HC1)"

    if vcov == "driscoll-kraay":
        if time_codes is None:
            raise ValueError("Driscoll-Kraay standard errors require a time index.")
        agg = np.zeros((n_times, scores.shape[1]))
        for j in range(scores.shape[1]):
            agg[:, j] = np.bincount(time_codes, weights=scores[:, j], minlength=n_times)
        band = lags if lags >= 0 else auto_bandwidth(n_times)
        meat = newey_west_lrv(agg, band)
        scale = (
            n_times / max(n_times - n_params, 1)
            if dof_correction and n_times > n_params
            else 1.0
        )
        v = scale * (xtx_inv @ meat @ xtx_inv)
        return (_psd_clip(v) if psd_clip else v), f"Driscoll-Kraay (lags={band})"

    # ---- clustered -------------------------------------------------------- #
    if not cluster_codes:
        raise ValueError("`vcov='cluster'` requires at least one `cluster` column.")

    def _one(codes: np.ndarray, n_groups: int) -> np.ndarray:
        meat = _cluster_meat(scores, codes, n_groups)
        c = 1.0
        if dof_correction and n_groups > 1:
            c = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / max(n - n_params, 1))
        return c * (xtx_inv @ meat @ xtx_inv)

    if len(cluster_codes) == 1:
        v = _one(cluster_codes[0], cluster_levels[0])
        return v, f"cluster ({cluster_levels[0]} groups)"

    if len(cluster_codes) != 2:
        raise ValueError(
            "only one-way and two-way clustering are supported "
            f"(got {len(cluster_codes)} cluster dimensions)."
        )
    c1, c2 = cluster_codes
    g1, g2 = cluster_levels
    # Intersection clusters for the CGM correction term.
    pair = c1.astype(np.int64) * (g2 + 1) + c2.astype(np.int64)
    c12, _ = factorize(pair)
    g12 = int(c12.max()) + 1
    v = _one(c1, g1) + _one(c2, g2) - _one(c12, g12)
    if psd_clip:
        v = _psd_clip(v)
    return v, f"two-way cluster (CGM, {g1} x {g2} groups)"


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def hdfe(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    absorb: Sequence[str],
    cluster: Sequence[str] | str | None = None,
    vcov: str = "classical",
    entity: str | None = None,
    time: str | None = None,
    lags: int = -1,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    dof_correction: bool = True,
    psd_clip: bool = True,
) -> HDFEResult:
    """Estimate ``y ~ x | absorb`` by absorbing N-way fixed effects.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format data. Any frame works: ``absorb`` names the fixed-effect
        columns explicitly, so the panel keys are only needed for
        Driscoll-Kraay (which uses the time column).
    y : str
        Dependent variable.
    x : sequence of str
        Regressors. No intercept is added (it is absorbed by the fixed effects);
        pass an explicit constant column only if ``absorb`` is empty.
    absorb : sequence of str
        Fixed-effect dimensions to absorb. May be empty (plain OLS).
    cluster : str or sequence of str, optional
        One or two columns to cluster on. Implies ``vcov="cluster"``.
    vcov : {"classical", "robust", "cluster", "driscoll-kraay"}, default="classical"
    entity, time : str, optional
        Panel keys, used when ``data`` is a bare frame.
    lags : int, default=-1
        Bartlett bandwidth for Driscoll-Kraay. ``-1`` selects Newey-West's
        ``floor(4 (T/100)^(2/9))`` rule of thumb.
    tol, max_iter : float, int
        Alternating-projection convergence controls (see :func:`demean`).
    dof_correction : bool, default=True
        Apply the usual finite-sample factors (HC1 / Liang-Zeger).
    psd_clip : bool, default=True
        Clip negative eigenvalues of two-way-clustered and Driscoll-Kraay
        covariances to zero (they are not PSD by construction).

    Returns
    -------
    HDFEResult

    Raises
    ------
    ValueError
        On unknown columns, an unknown ``vcov``, or a rank-deficient design.

    Examples
    --------
    >>> res = hdfe(df, y="ret", x=["size", "bm"], absorb=["firm", "date"],
    ...            cluster=["firm", "date"])  # doctest: +SKIP
    >>> res.summary()  # doctest: +SKIP
    """
    from panelary.core.panel_frame import as_panel

    panel = (
        data
        if isinstance(data, PanelFrame)
        else as_panel(data, entity=entity, time=time)
    )

    x = list(x)
    absorb = list(absorb)
    if isinstance(cluster, str):
        cluster = [cluster]
    cluster = list(cluster or [])
    if cluster and vcov == "classical":
        vcov = "cluster"
    if vcov not in _VCOV_KINDS:
        raise ValueError(f"unknown `vcov={vcov!r}`; expected one of {_VCOV_KINDS}.")
    if not x:
        raise ValueError("`x` must name at least one regressor.")

    needed = [y, *x]
    key_cols = list(
        dict.fromkeys([panel.entity_col, panel.time_col, *absorb, *cluster])
    )
    available = set(panel.columns)
    for col in [*needed, *key_cols]:
        if col not in available:
            raise ValueError(
                f"column {col!r} not found in the data. Available: {panel.columns}."
            )

    frame = (
        panel.lazy()
        .select(
            [
                *[pl.col(c) for c in key_cols],
                *[pl.col(c).cast(pl.Float64).alias(c) for c in needed],
            ]
        )
        .collect()
    )
    mask = pl.all_horizontal(
        [pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in needed]
    )
    frame = frame.filter(mask)
    n = frame.height
    if n <= len(x):
        raise ValueError(
            f"not enough complete observations ({n}) for {len(x)} regressor(s)."
        )

    fe_codes: list[np.ndarray] = []
    fe_levels: list[int] = []
    fe_level_values: dict[str, np.ndarray] = {}
    for col in absorb:
        codes, levels = factorize(frame[col].to_numpy())
        fe_codes.append(codes)
        fe_levels.append(int(levels.size))
        fe_level_values[col] = levels

    mat = np.column_stack([frame[c].to_numpy() for c in needed]).astype(float)
    resid_all, offsets, n_iter, max_dev = demean(
        mat, fe_codes, fe_levels, tol=tol, max_iter=max_iter
    )
    converged = max_dev <= tol * (1.0 + float(np.abs(mat).max())) if fe_codes else True

    ry = resid_all[:, 0]
    rx = resid_all[:, 1:]

    xtx = rx.T @ rx
    if np.linalg.matrix_rank(xtx) < xtx.shape[0]:
        raise ValueError(
            "the partialled-out regressors are rank deficient: one or more of "
            f"{x} is collinear with the absorbed fixed effects {absorb}."
        )
    xtx_inv = pinv_sym(xtx)
    params = xtx_inv @ (rx.T @ ry)
    resid = ry - rx @ params

    df_absorbed = (sum(fe_levels) - (len(fe_levels) - 1)) if fe_levels else 0
    n_params = len(x) + df_absorbed
    df_resid = max(n - n_params, 1)

    cl_codes: list[np.ndarray] = []
    cl_levels: list[int] = []
    for col in cluster:
        codes, levels = factorize(frame[col].to_numpy())
        cl_codes.append(codes)
        cl_levels.append(int(levels.size))

    time_codes = n_times = None
    if vcov == "driscoll-kraay":
        time_codes, tlevels = factorize(frame[panel.time_col].to_numpy())
        n_times = int(tlevels.size)

    v, label = _hdfe_vcov(
        rx,
        resid,
        xtx_inv,
        vcov=vcov,
        cluster_codes=cl_codes,
        cluster_levels=cl_levels,
        time_codes=time_codes,
        n_times=n_times or 0,
        lags=lags,
        df_resid=df_resid,
        n_params=n_params,
        dof_correction=dof_correction,
        psd_clip=psd_clip,
    )

    se = np.sqrt(np.clip(np.diag(v), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, params / se, np.nan)
    pvalues = 2.0 * np.asarray(t_sf(np.abs(tstats), df_resid))

    rss = float(resid @ resid)
    tss_within = float(ry @ ry)
    r2_within = 1.0 - rss / tss_within if tss_within > 0 else float("nan")

    return HDFEResult(
        names=list(x),
        params=params,
        std_errors=se,
        tstats=tstats,
        pvalues=pvalues,
        vcov=v,
        resid=resid,
        resid_y=ry,
        resid_x=rx,
        nobs=n,
        df_absorbed=df_absorbed,
        df_resid=df_resid,
        rss=rss,
        tss_within=tss_within,
        r2_within=r2_within,
        vcov_type=label,
        converged=bool(converged),
        n_iter=n_iter,
        fe_offsets={col: offsets[i] for i, col in enumerate(absorb)},
        fe_levels=fe_level_values,
    )


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #
class HDFETransformer(PanelTransformer):
    """Emit fixed-effect-partialled-out columns as leak-controlled features.

    ``fit`` runs the alternating projections on the **training rows only** and
    freezes the per-level offsets ``alpha_g[level]`` for every absorbed
    dimension. ``transform`` applies those frozen offsets:

    .. math::

        \\tilde{z}_i = z_i - \\sum_g \\alpha_g[\\text{level}_g(i)]

    On the training rows this reproduces the alternating-projection residual
    exactly; on held-out rows it is a genuine out-of-sample projection using
    train-estimated fixed effects. Levels unseen in training get an offset of
    zero (or ``null``, with ``unseen="null"``).

    Parameters
    ----------
    columns : sequence of str
        Columns to residualise.
    absorb : sequence of str
        Fixed-effect dimensions to absorb.
    suffix : str, default="_hdfe"
        Suffix for the emitted columns.
    drop_original : bool, default=False
        Replace the source columns instead of appending.
    unseen : {"zero", "null"}, default="zero"
        What to do with a level not seen during ``fit``.
    tol, max_iter :
        Alternating-projection controls (see :func:`demean`).

    Attributes
    ----------
    offsets_ : dict of str -> dict
        ``{absorb_col: {"levels": ndarray, "offsets": ndarray (n_levels, k)}}``.
    n_iter_ : int
    converged_ : bool

    Examples
    --------
    >>> tr = HDFETransformer(columns=["ret"], absorb=["firm", "date"])
    >>> tr.fit(train).transform(test)  # doctest: +SKIP
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        columns: Sequence[str],
        absorb: Sequence[str],
        *,
        suffix: str = "_hdfe",
        drop_original: bool = False,
        unseen: str = "zero",
        tol: float = 1e-12,
        max_iter: int = 10_000,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.columns = list(columns)
        self.absorb = list(absorb)
        if not self.columns:
            raise ValueError("`columns` must name at least one column to residualise.")
        if not self.absorb:
            raise ValueError("`absorb` must name at least one fixed-effect dimension.")
        if unseen not in ("zero", "null"):
            raise ValueError(f"`unseen` must be 'zero' or 'null', got {unseen!r}.")
        self.suffix = suffix
        self.drop_original = bool(drop_original)
        self.unseen = unseen
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.offsets_: dict[str, dict[str, np.ndarray]] = {}
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _fit(self, panel: PanelFrame) -> None:
        available = set(panel.columns)
        missing = [c for c in [*self.columns, *self.absorb] if c not in available]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel. Available: {panel.columns}."
            )
        frame = (
            panel.lazy()
            .select(
                [
                    *[pl.col(c) for c in self.absorb],
                    *[pl.col(c).cast(pl.Float64) for c in self.columns],
                ]
            )
            .collect()
        )
        mask = pl.all_horizontal(
            [pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in self.columns]
        )
        frame = frame.filter(mask)
        if frame.height == 0:
            raise ValueError("no complete rows to fit the fixed effects on.")
        codes, levels_list = [], []
        for col in self.absorb:
            c, lv = factorize(frame[col].to_numpy())
            codes.append(c)
            levels_list.append(lv)
        mat = np.column_stack([frame[c].to_numpy() for c in self.columns]).astype(float)
        _, offsets, n_iter, max_dev = demean(
            mat,
            codes,
            [int(lv.size) for lv in levels_list],
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.offsets_ = {
            col: {"levels": levels_list[i], "offsets": offsets[i]}
            for i, col in enumerate(self.absorb)
        }
        self.n_iter_ = n_iter
        self.converged_ = max_dev <= self.tol * (1.0 + float(np.abs(mat).max()))

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        lf = panel.lazy()
        # Build one lookup frame per absorbed dimension, then join.
        exprs: list[pl.Expr] = []
        contribs: dict[str, list[pl.Expr]] = {c: [] for c in self.columns}
        for dim_idx, col in enumerate(self.absorb):
            state = self.offsets_[col]
            levels = state["levels"]
            offs = state["offsets"]
            lut = pl.DataFrame(
                {
                    col: pl.Series(levels),
                    **{
                        f"__off_{dim_idx}_{j}": offs[:, j]
                        for j in range(len(self.columns))
                    },
                }
            ).lazy()
            lf = lf.join(lut, on=col, how="left")
            for j, src in enumerate(self.columns):
                name = f"__off_{dim_idx}_{j}"
                expr = pl.col(name)
                if self.unseen == "zero":
                    expr = expr.fill_null(0.0)
                contribs[src].append(expr)

        for src in self.columns:
            total = contribs[src][0]
            for extra in contribs[src][1:]:
                total = total + extra
            exprs.append(
                (pl.col(src).cast(pl.Float64) - total).alias(f"{src}{self.suffix}")
            )
        lf = lf.with_columns(exprs)
        drop_cols = [
            f"__off_{d}_{j}"
            for d in range(len(self.absorb))
            for j in range(len(self.columns))
        ]
        if self.drop_original:
            drop_cols += list(self.columns)
        lf = lf.drop(drop_cols)
        return PanelFrame(
            lf, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    @property
    def output_names(self) -> list[str]:
        """Names of the emitted residualised columns."""
        return [f"{c}{self.suffix}" for c in self.columns]
