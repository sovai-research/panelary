"""Distance covariance and correlation (Szekely, Rizzo & Bakirov, 2007).

Two computational paths, chosen automatically by :func:`dcov2`:

* **univariate x and y -> exact O(n^1.5) with no n x n matrix.** The
  cross-term ``sum_{i,j} |x_i - x_j| |y_i - y_j|`` is rewritten, after sorting
  by ``x``, as four *dominance sums* ``sum_{j<i, y_j < y_i} w_j`` with weights
  ``{1, y, x, xy}`` -- the idea of Huo & Szekely (2016), implemented clean-room
  on top of the blocked dominance primitive in
  :mod:`panelary.depend._ranks`. Row sums ``a_i. = sum_j |x_i - x_j|`` come from
  sorted prefix sums. Ties need no special case: a tied pair contributes
  ``|x_i - x_j| = 0`` whatever order it is visited in.
* **multivariate -> blocked O(n^2).** Row sums and the cross sum are accumulated
  over ``2000 x 2000`` tiles, so the full distance matrix is never
  materialised (measured: a dense ``n = 10 000`` matrix is 800 MB, ``n = 20 000``
  is 3.2 GB). Above ``max_n`` the sample is **subsampled with the supplied
  seed** and the result says so (``approximate=True``, ``n_subsample``) --
  never a silent OOM, never a silent truncation.

``bias_corrected=True`` (the default) is the U-centred, unbiased estimator of
Szekely & Rizzo (2014)::

    Omega = S_ab / (n (n-3)) - 2 sum_i a_i. b_i. / (n (n-2) (n-3))
            + a.. b.. / (n (n-1) (n-2) (n-3))

The null distribution -- read this before trusting a p-value
------------------------------------------------------------
The Szekely-Rizzo t-test ``T = sqrt(v - 1) R* / sqrt(1 - R*^2)``,
``v = n (n - 3) / 2``, is derived for the **high-dimensional** limit. For
univariate ``x`` and ``y`` -- the panel case -- ``n Omega`` converges to a
centred weighted chi-square that is strongly right-skewed, and the t-test
over-rejects. :func:`dcor_pvalue` with ``method="auto"`` therefore uses the
t-test only when both dimensions are ``>= 10`` and otherwise a **gamma
approximation matched to the exact permutation moments**: under permutation
the U-statistic has mean exactly 0 and variance exactly
``2 Omega_xx Omega_yy / (n (n - 3))`` (derived in the source), and
``n Omega + mean_a mean_b`` is matched to a gamma with that mean and variance.
Serially dependent input must go through :mod:`panelary.depend._null` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from panelary.depend._ranks import _dominance_sums, ranks
from panelary.econ._common import chi2_sf, t_sf

__all__ = [
    "DcorResult",
    "dcor",
    "dcor_pvalue",
    "dcov2",
    "partial_dcor",
]

#: Tile edge for the blocked multivariate path (a 2000 x 2000 float64 tile is 32 MB).
_TILE = 2000


@dataclass(frozen=True)
class DcorResult:
    """Distance covariance / correlation of one pair of samples.

    Attributes
    ----------
    dcov2 : float
        Squared distance covariance (U-statistic if ``bias_corrected``).
    dvar_x, dvar_y : float
        Squared distance variances, same estimator.
    dcor : float
        ``dcov2 / sqrt(dvar_x dvar_y)`` (the bias-corrected ``R*`` when
        ``bias_corrected``; it can be slightly negative under independence).
    n_obs : int
        Complete rows in the input.
    n_used : int
        Rows actually used (``n_subsample`` if subsampled, else ``n_obs``).
    dim_x, dim_y : int
        Dimensions of ``x`` and ``y``.
    bias_corrected : bool
    approximate : bool
        True if the statistic was computed on a seeded subsample.
    n_subsample : int or None
    path : str
        ``"univariate"`` (no matrix) or ``"blocked"`` (tiled ``O(n^2)``).
    mean_a, mean_b : float
        Mean off-diagonal distances ``a.. / (n (n - 1))`` -- the gamma null's
        location.
    """

    dcov2: float
    dvar_x: float
    dvar_y: float
    dcor: float
    n_obs: int
    n_used: int
    dim_x: int
    dim_y: int
    bias_corrected: bool
    approximate: bool
    n_subsample: int | None
    path: str
    mean_a: float
    mean_b: float


def _combine(
    s_ab: np.ndarray,
    ab_dot: np.ndarray,
    a_dd: np.ndarray,
    b_dd: np.ndarray,
    n: int,
    bias_corrected: bool,
) -> np.ndarray:
    """Assemble dCov^2 from ``sum a_ij b_ij``, ``sum a_i. b_i.``, ``a..``, ``b..``."""
    if bias_corrected:
        if n < 4:
            return np.full(np.shape(s_ab), np.nan)
        return (
            s_ab / (n * (n - 3.0))
            - 2.0 * ab_dot / (n * (n - 2.0) * (n - 3.0))
            + a_dd * b_dd / (n * (n - 1.0) * (n - 2.0) * (n - 3.0))
        )
    return s_ab / n**2 - 2.0 * ab_dot / n**3 + a_dd * b_dd / n**4


def _row_dist_sums(v: np.ndarray) -> np.ndarray:
    """``a_i. = sum_j |v_i - v_j|`` for every row of a ``(B, m)`` array."""
    m = v.shape[-1]
    order = np.argsort(v, axis=-1, kind="stable")
    vs = np.take_along_axis(v, order, axis=-1)
    prefix = np.cumsum(vs, axis=-1) - vs
    total = vs.sum(axis=-1, keepdims=True)
    i = np.arange(m, dtype=np.float64)
    sums_sorted = vs * (2.0 * i - m) + total - 2.0 * prefix
    out = np.empty_like(v)
    np.put_along_axis(out, order, sums_sorted, axis=-1)
    return out


def _univariate_terms(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-batched sufficient sums for univariate dCov (``(B, m)`` inputs).

    Returns ``(S_ab, sum_i a_i. b_i., a.., b.., sum a_ij^2, sum b_ij^2)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    bsz, m = x.shape
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    a_dot = _row_dist_sums(xc)
    b_dot = _row_dist_sums(yc)
    order = np.argsort(xc, axis=-1, kind="stable")
    xs = np.take_along_axis(xc, order, axis=-1)
    ys = np.take_along_axis(yc, order, axis=-1)
    oy = ranks(ys, method="ordinal") - 1
    w = np.stack([np.ones_like(xs), ys, xs, xs * ys], axis=-1)
    dom = _dominance_sums(oy, w)
    prefix = np.cumsum(w, axis=1) - w
    sgn = 2.0 * dom - prefix
    term = xs * ys * sgn[..., 0] - xs * sgn[..., 1] - ys * sgn[..., 2] + sgn[..., 3]
    s_ab = 2.0 * term.sum(axis=-1)
    ab_dot = np.einsum("bm,bm->b", a_dot, b_dot)
    a_dd = a_dot.sum(axis=-1)
    b_dd = b_dot.sum(axis=-1)
    s_aa = 2.0 * m * np.einsum("bm,bm->b", xc, xc)
    s_bb = 2.0 * m * np.einsum("bm,bm->b", yc, yc)
    aa_dot = np.einsum("bm,bm->b", a_dot, a_dot)
    bb_dot = np.einsum("bm,bm->b", b_dot, b_dot)
    return s_ab, ab_dot, a_dd, b_dd, np.stack([s_aa, aa_dot]), np.stack([s_bb, bb_dot])


def _dcor_rows(
    x: np.ndarray, y: np.ndarray, *, bias_corrected: bool = True
) -> np.ndarray:
    """Row-wise (bias-corrected) distance correlation of univariate ``(B, m)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    m = x.shape[-1]
    s_ab, ab_dot, a_dd, b_dd, xx, yy = _univariate_terms(x, y)
    num = _combine(s_ab, ab_dot, a_dd, b_dd, m, bias_corrected)
    vx = _combine(xx[0], xx[1], a_dd, a_dd, m, bias_corrected)
    vy = _combine(yy[0], yy[1], b_dd, b_dd, m, bias_corrected)
    den = np.sqrt(np.clip(vx, 0.0, None) * np.clip(vy, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _pairwise_dist_tile(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of two small matrices."""
    sq = (
        np.einsum("ij,ij->i", a, a)[:, None]
        + np.einsum("ij,ij->i", b, b)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.sqrt(np.clip(sq, 0.0, None))


def _blocked_terms(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Tiled ``O(n^2)`` sufficient sums for multivariate dCov."""
    n = x.shape[0]
    a_dot = np.zeros(n)
    b_dot = np.zeros(n)
    s_ab = s_aa = s_bb = 0.0
    for i0 in range(0, n, _TILE):
        i1 = min(i0 + _TILE, n)
        for j0 in range(0, n, _TILE):
            j1 = min(j0 + _TILE, n)
            da = _pairwise_dist_tile(x[i0:i1], x[j0:j1])
            db = _pairwise_dist_tile(y[i0:i1], y[j0:j1])
            if i0 == j0:
                np.fill_diagonal(da, 0.0)
                np.fill_diagonal(db, 0.0)
            a_dot[i0:i1] += da.sum(axis=1)
            b_dot[i0:i1] += db.sum(axis=1)
            s_ab += float(np.einsum("ij,ij->", da, db))
            s_aa += float(np.einsum("ij,ij->", da, da))
            s_bb += float(np.einsum("ij,ij->", db, db))
    return {
        "s_ab": s_ab,
        "ab_dot": float(a_dot @ b_dot),
        "a_dd": float(a_dot.sum()),
        "b_dd": float(b_dot.sum()),
        "s_aa": s_aa,
        "aa_dot": float(a_dot @ a_dot),
        "s_bb": s_bb,
        "bb_dot": float(b_dot @ b_dot),
    }


def _as_2d(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr[:, None] if arr.ndim == 1 else arr


def dcov2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bias_corrected: bool = True,
    max_n: int = 10_000,
    seed: int = 0,
) -> DcorResult:
    """Squared distance covariance, variances and correlation of ``x`` and ``y``.

    Parameters
    ----------
    x, y : array_like
        ``(n,)`` or ``(n, d)`` samples with the same ``n``. Rows with any
        non-finite entry are dropped.
    bias_corrected : bool, default=True
        U-centred unbiased estimator (Szekely & Rizzo, 2014). ``False`` gives
        the original V-statistic.
    max_n : int, default=10_000
        Cap for the multivariate ``O(n^2)`` path; above it a seeded subsample of
        ``max_n`` rows is used and ``approximate=True`` is recorded. The
        univariate path is exact at any ``n``.
    seed : int, default=0
        Seed for the subsample.

    Returns
    -------
    DcorResult

    Raises
    ------
    ValueError
        If the row counts differ.
    """
    xa = _as_2d(x)
    ya = _as_2d(y)
    if xa.shape[0] != ya.shape[0]:
        raise ValueError(
            f"`x` and `y` must have the same number of rows, got {xa.shape[0]} "
            f"and {ya.shape[0]}."
        )
    keep = np.isfinite(xa).all(axis=1) & np.isfinite(ya).all(axis=1)
    xa, ya = xa[keep], ya[keep]
    n_obs = int(keep.sum())
    dx, dy = xa.shape[1], ya.shape[1]
    nan = float("nan")
    if n_obs < (4 if bias_corrected else 2):
        return DcorResult(
            nan,
            nan,
            nan,
            nan,
            n_obs,
            n_obs,
            dx,
            dy,
            bias_corrected,
            False,
            None,
            "univariate",
            nan,
            nan,
        )
    if dx == 1 and dy == 1:
        s_ab, ab_dot, a_dd, b_dd, xx, yy = _univariate_terms(xa[:, 0], ya[:, 0])
        terms = {
            "s_ab": float(s_ab[0]),
            "ab_dot": float(ab_dot[0]),
            "a_dd": float(a_dd[0]),
            "b_dd": float(b_dd[0]),
            "s_aa": float(xx[0, 0]),
            "aa_dot": float(xx[1, 0]),
            "s_bb": float(yy[0, 0]),
            "bb_dot": float(yy[1, 0]),
        }
        n_used, approximate, n_sub, path = n_obs, False, None, "univariate"
    else:
        if n_obs > max_n:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n_obs, size=int(max_n), replace=False))
            xa, ya = xa[idx], ya[idx]
            n_used, approximate, n_sub = int(max_n), True, int(max_n)
        else:
            n_used, approximate, n_sub = n_obs, False, None
        xa = xa - xa.mean(axis=0)
        ya = ya - ya.mean(axis=0)
        terms = _blocked_terms(xa, ya)
        path = "blocked"
    n = n_used
    cov = float(
        _combine(
            np.asarray(terms["s_ab"]),
            np.asarray(terms["ab_dot"]),
            np.asarray(terms["a_dd"]),
            np.asarray(terms["b_dd"]),
            n,
            bias_corrected,
        )
    )
    vx = float(
        _combine(
            np.asarray(terms["s_aa"]),
            np.asarray(terms["aa_dot"]),
            np.asarray(terms["a_dd"]),
            np.asarray(terms["a_dd"]),
            n,
            bias_corrected,
        )
    )
    vy = float(
        _combine(
            np.asarray(terms["s_bb"]),
            np.asarray(terms["bb_dot"]),
            np.asarray(terms["b_dd"]),
            np.asarray(terms["b_dd"]),
            n,
            bias_corrected,
        )
    )
    den = math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0
    r = cov / den if den > 0 else nan
    return DcorResult(
        dcov2=cov,
        dvar_x=vx,
        dvar_y=vy,
        dcor=r,
        n_obs=n_obs,
        n_used=n,
        dim_x=dx,
        dim_y=dy,
        bias_corrected=bias_corrected,
        approximate=approximate,
        n_subsample=n_sub,
        path=path,
        mean_a=terms["a_dd"] / (n * (n - 1.0)),
        mean_b=terms["b_dd"] / (n * (n - 1.0)),
    )


def dcor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bias_corrected: bool = True,
    max_n: int = 10_000,
    seed: int = 0,
) -> float:
    """Distance correlation (bias-corrected ``R*`` by default). See :func:`dcov2`."""
    return dcov2(x, y, bias_corrected=bias_corrected, max_n=max_n, seed=seed).dcor


def dcor_pvalue(result: DcorResult, *, method: str = "auto") -> float:
    """i.i.d. p-value for a bias-corrected distance covariance.

    Parameters
    ----------
    result : DcorResult
        From :func:`dcov2` with ``bias_corrected=True``.
    method : {"auto", "t", "gamma"}, default="auto"
        ``"t"``: the Szekely-Rizzo (2013) t-test, valid in the high-dimensional
        limit only. ``"gamma"``: gamma approximation to ``n Omega + mean_a
        mean_b`` matched to the *exact permutation* mean and variance.
        ``"auto"`` picks ``"t"`` when ``dim_x >= 10`` and ``dim_y >= 10``, else
        ``"gamma"``. The t-test is **not** valid for univariate data --
        ``tests/test_depend_calibration.py`` measures how far off it is.

    Returns
    -------
    float
        One-sided (upper-tail) p-value; ``nan`` if undefined.

    Raises
    ------
    ValueError
        For an unknown ``method`` or a V-statistic input.
    """
    if not result.bias_corrected:
        raise ValueError("dcor_pvalue needs a bias-corrected (U-statistic) DcorResult.")
    if method == "auto":
        method = "t" if (result.dim_x >= 10 and result.dim_y >= 10) else "gamma"
    n = result.n_used
    if not np.isfinite(result.dcor) or n < 5:
        return float("nan")
    if method == "t":
        v = n * (n - 3.0) / 2.0
        r = result.dcor
        if r >= 1.0:
            return 0.0
        t = math.sqrt(v - 1.0) * r / math.sqrt(1.0 - r * r)
        return float(t_sf(t, v - 1.0))
    if method == "gamma":
        return _gamma_dcov_pvalue(
            result.dcov2, result.dvar_x, result.dvar_y, result.mean_a, result.mean_b, n
        )
    raise ValueError(f"unknown `method` {method!r}; expected 'auto', 't' or 'gamma'.")


def _gamma_dcov_pvalue(
    omega: float, vx: float, vy: float, mean_a: float, mean_b: float, n: int
) -> float:
    """Gamma tail for ``n Omega + mean_a mean_b`` (exact permutation moments)."""
    loc = mean_a * mean_b
    var = 2.0 * n * vx * vy / (n - 3.0)
    if not (loc > 0 and var > 0 and np.isfinite(omega)):
        return float("nan")
    obs = n * omega + loc
    if obs <= 0:
        return 1.0
    shape = loc * loc / var
    scale = var / loc
    return float(chi2_sf(2.0 * obs / scale, 2.0 * shape))


def partial_dcor(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial distance correlation ``pdCor(x, y; z)`` (Szekely & Rizzo, 2014).

    Computed in the U-centred inner-product space as
    ``(R_xy - R_xz R_yz) / sqrt((1 - R_xz^2)(1 - R_yz^2))`` from three
    bias-corrected distance correlations. ``0`` when ``z`` explains ``x`` or
    ``y`` completely.
    """
    xa, ya, za = _as_2d(x), _as_2d(y), _as_2d(z)
    keep = (
        np.isfinite(xa).all(axis=1)
        & np.isfinite(ya).all(axis=1)
        & np.isfinite(za).all(axis=1)
    )
    xa, ya, za = xa[keep], ya[keep], za[keep]
    rxy = dcov2(xa, ya).dcor
    rxz = dcov2(xa, za).dcor
    ryz = dcov2(ya, za).dcor
    den = (1.0 - rxz * rxz) * (1.0 - ryz * ryz)
    if not np.isfinite(den):
        return float("nan")
    if den <= 0:
        return 0.0
    return float((rxy - rxz * ryz) / math.sqrt(den))
