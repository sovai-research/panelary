"""Information-theoretic dependence: Gaussian-copula MI, conditional MI,
variation of information, and the digamma function they need.

Read this first: the honesty trap in GCMI
-----------------------------------------
Bivariate Gaussian-copula mutual information is ``-1/2 log(1 - rho_z^2)``
where ``rho_z`` is the Pearson correlation of the **normal scores** -- a
strictly monotone function of the van der Waerden rank correlation. For a
single pair of columns it is therefore *Spearman in a hat*: it cannot see a
U-shape, a circle, or anything else non-monotone, whatever the words "mutual
information" suggest. Use :func:`~panelary.depend.xi`,
:func:`~panelary.depend.dcor` or :func:`~panelary.depend.hsic` for genuine
nonlinearity. GCMI earns its place for three things a rank correlation cannot
do: (a) **multivariate** ``I(X; Y_1 ... Y_k)``, (b) **conditional**
``I(X; Y | Z)`` from four log-determinants, and (c) a full ``p x p`` MI matrix
from **one** covariance of normal scores.

Clean-room
----------
The GPL ``gcmi`` package is a reference only. Everything here is derived from
Ince et al. (2017) and from the Wishart moment identity
``E[log det S] = log det Sigma + d log(2 / (n - 1)) + sum_{i=1}^d psi((n - i) / 2)``
for the ``(n - 1)``-divisor sample covariance ``S``; the tests validate against
the closed-form MI of a Gaussian, not against GPL code.

All values are in **nats**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from panelary._internal._special import psi
from panelary.depend._ranks import normal_scores, ranks
from panelary.econ._common import chi2_sf, ols

__all__ = [
    "TEResult",
    "cmi_ksg",
    "gcmi",
    "gcmi_conditional",
    "gcmi_matrix",
    "gcmi_pvalue",
    "mi_ksg",
    "mi_to_r",
    "psi",
    "transfer_entropy_array",
    "variation_of_information",
]

_EULER_GAMMA = 0.5772156649015329


def _logdet_bias(d: int, n: int) -> float:
    """``E[log det S] - log det Sigma`` for the ``(n-1)``-divisor covariance."""
    i = np.arange(1, d + 1, dtype=np.float64)
    return float(d * math.log(2.0 / (n - 1.0)) + np.sum(psi((n - i) / 2.0)))


def _as_2d(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr[:, None] if arr.ndim == 1 else arr


def _complete(*arrays: np.ndarray) -> tuple[list[np.ndarray], int]:
    mats = [_as_2d(a) for a in arrays]
    n = mats[0].shape[0]
    if any(m.shape[0] != n for m in mats):
        raise ValueError("all inputs must have the same number of rows.")
    keep = np.ones(n, dtype=bool)
    for m in mats:
        keep &= np.isfinite(m).all(axis=1)
    return [m[keep] for m in mats], int(keep.sum())


def _copula(m: np.ndarray) -> np.ndarray:
    return normal_scores(m, axis=0)


def _logdet(c: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(c)
    return float(ld) if sign > 0 else float("-inf")


def _mi_from_cov(c: np.ndarray, dx: int, n: int, biascorrect: bool) -> float:
    d = c.shape[0]
    dy = d - dx
    ldx = _logdet(c[:dx, :dx])
    ldy = _logdet(c[dx:, dx:])
    ldxy = _logdet(c)
    if biascorrect:
        ldx -= _logdet_bias(dx, n)
        ldy -= _logdet_bias(dy, n)
        ldxy -= _logdet_bias(d, n)
    return 0.5 * (ldx + ldy - ldxy)


def gcmi(x: np.ndarray, y: np.ndarray, *, biascorrect: bool = True) -> float:
    """Gaussian-copula mutual information ``I(X; Y)`` in nats.

    **For a single pair of columns this is a monotone function of a rank
    correlation** (see the module notes, "the honesty trap"): it detects
    monotone dependence only. Its value is for multivariate ``x`` / ``y``, for
    conditional MI (:func:`gcmi_conditional`) and for MI matrices
    (:func:`gcmi_matrix`). For nonlinear bivariate dependence use ``xi``,
    ``dcor`` or ``hsic``.

    Each column is mapped to normal scores (the copula transform), then
    ``I = 1/2 [log det S_xx + log det S_yy - log det S]`` with the Wishart
    finite-sample bias correction of every log-determinant (Ince et al., 2017).
    The copula MI is a **lower bound** on the true MI, exact when the copula is
    Gaussian.

    Parameters
    ----------
    x, y : array_like
        ``(n,)`` or ``(n, d)``. Rows with any non-finite value are dropped.
    biascorrect : bool, default=True
        Apply the finite-sample entropy bias correction.

    Returns
    -------
    float
        MI in nats (may be slightly negative under independence when
        bias-corrected); ``nan`` if too few rows for the covariance.

    References
    ----------
    Ince, R. A. A. et al. (2017). A statistical framework for neuroimaging data
    analysis based on mutual information estimated via a Gaussian copula.
    *Human Brain Mapping* 38(3).
    """
    (xa, ya), n = _complete(x, y)
    dx, dy = xa.shape[1], ya.shape[1]
    if n < dx + dy + 3:
        return float("nan")
    z = np.hstack([_copula(xa), _copula(ya)])
    c = np.cov(z, rowvar=False)
    return _mi_from_cov(np.atleast_2d(c), dx, n, biascorrect)


def gcmi_pvalue(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> float:
    """i.i.d. p-value for ``I(X; Y)`` (or ``I(X; Y | Z)``) from the Gaussian
    likelihood-ratio test: ``(n - 1 - d_z - (d_x + d_y + 1)/2) * 2 I`` is
    asymptotically ``chi^2(d_x d_y)`` (Bartlett-corrected), computed on the
    **uncorrected** copula MI.
    """
    arrays = (x, y) if z is None else (x, y, z)
    mats, n = _complete(*arrays)
    dx, dy = mats[0].shape[1], mats[1].shape[1]
    dz = 0 if z is None else mats[2].shape[1]
    if n < dx + dy + dz + 4:
        return float("nan")
    if z is None:
        mi = gcmi(mats[0], mats[1], biascorrect=False)
    else:
        mi = gcmi_conditional(mats[0], mats[1], mats[2], biascorrect=False)
    factor = n - 1.0 - dz - (dx + dy + 1.0) / 2.0
    if not np.isfinite(mi) or factor <= 0:
        return float("nan")
    return float(chi2_sf(max(0.0, factor * 2.0 * mi), dx * dy))


def gcmi_conditional(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, biascorrect: bool = True
) -> float:
    """Conditional Gaussian-copula MI ``I(X; Y | Z)`` in nats.

    ``I = 1/2 [log det S_xz + log det S_yz - log det S_z - log det S_xyz]``,
    each log-determinant bias-corrected. This is the case where GCMI is
    genuinely more than a rank correlation.

    Parameters
    ----------
    x, y, z : array_like
        ``(n,)`` or ``(n, d)``; rows with any non-finite value are dropped.
    biascorrect : bool, default=True

    Returns
    -------
    float
    """
    (xa, ya, za), n = _complete(x, y, z)
    dx, dy, dz = xa.shape[1], ya.shape[1], za.shape[1]
    d = dx + dy + dz
    if n < d + 3:
        return float("nan")
    g = np.hstack([_copula(xa), _copula(ya), _copula(za)])
    c = np.atleast_2d(np.cov(g, rowvar=False))
    ix = list(range(dx))
    iy = list(range(dx, dx + dy))
    iz = list(range(dx + dy, d))

    def ld(idx: list[int]) -> float:
        val = _logdet(c[np.ix_(idx, idx)])
        return val - _logdet_bias(len(idx), n) if biascorrect else val

    return 0.5 * (ld(ix + iz) + ld(iy + iz) - ld(iz) - ld(ix + iy + iz))


def _gcmi_rows(x: np.ndarray, y: np.ndarray, *, biascorrect: bool = True) -> np.ndarray:
    """Row-wise bivariate GCMI of two ``(B, m)`` arrays."""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    m = x.shape[-1]
    zx = normal_scores(x, axis=-1)
    zy = normal_scores(y, axis=-1)
    zx = zx - zx.mean(axis=-1, keepdims=True)
    zy = zy - zy.mean(axis=-1, keepdims=True)
    num = np.einsum("bm,bm->b", zx, zy)
    den = np.sqrt(np.einsum("bm,bm->b", zx, zx) * np.einsum("bm,bm->b", zy, zy))
    r = np.clip(num / np.where(den > 0, den, 1.0), -1.0 + 1e-15, 1.0 - 1e-15)
    mi = -0.5 * np.log1p(-r * r)
    if biascorrect and m > 3:
        mi = mi + 0.5 * (float(psi((m - 2.0) / 2.0)) - float(psi((m - 1.0) / 2.0)))
    return np.where(den > 0, mi, np.nan)


def gcmi_matrix(X: np.ndarray, *, biascorrect: bool = True) -> np.ndarray:
    """``(p, p)`` bivariate GCMI matrix from **one** covariance of normal scores.

    Listwise-complete rows. ``M[i, j] = -1/2 log(1 - r_ij^2)`` with ``r`` the
    normal-score correlation, plus the bivariate bias correction
    ``1/2 [psi((n-2)/2) - psi((n-1)/2)]``. The diagonal is ``inf``.
    """
    arr = np.asarray(X, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    n, p = arr.shape
    if n < 4:
        return np.full((p, p), np.nan)
    z = _copula(arr)
    r = np.corrcoef(z, rowvar=False)
    with np.errstate(divide="ignore"):
        mi = -0.5 * np.log1p(-np.clip(r * r, 0.0, 1.0))
    if biascorrect:
        mi = mi + 0.5 * (float(psi((n - 2.0) / 2.0)) - float(psi((n - 1.0) / 2.0)))
    np.fill_diagonal(mi, np.inf)
    return mi


def mi_to_r(mi: float | np.ndarray) -> float | np.ndarray:
    """``sqrt(1 - exp(-2 mi))`` -- **the Gaussian-equivalent correlation**.

    The absolute correlation a bivariate Gaussian would need to carry ``mi``
    nats. A reading aid for MI values, never a substitute for MI itself.
    Negative (bias-corrected) MI maps to 0.
    """
    arr = np.clip(np.asarray(mi, dtype=np.float64), 0.0, None)
    out = np.sqrt(-np.expm1(-2.0 * arr))
    return float(out) if out.ndim == 0 else out


def _optimal_bins(n: int, corr: float) -> int:
    """Hacine-Gharbi & Ravier (2018) bin count for joint entropy estimation."""
    c2 = min(corr * corr, 0.999999)
    b = (1.0 / math.sqrt(2.0)) * math.sqrt(1.0 + math.sqrt(1.0 + 24.0 * n / (1.0 - c2)))
    return max(2, int(round(b)))


def variation_of_information(
    x: np.ndarray, y: np.ndarray, *, bins: int | None = None, normalize: bool = True
) -> float:
    """Variation of information ``VI = H(X, Y) - I(X; Y)`` on equal-frequency bins.

    A true **metric** on the induced partitions (Meila, 2007), so it can feed
    :mod:`panelary.cluster` directly. Both variables are discretised into
    ``bins`` equal-frequency (rank-based) bins; ``bins=None`` uses the
    Hacine-Gharbi & Ravier (2018) joint-entropy rule, which depends on ``n`` and
    the sample correlation -- an analysis quantity, not a feature.

    Parameters
    ----------
    x, y : array_like
        1-D arrays. Non-finite rows are dropped.
    bins : int, optional
        Number of bins per variable.
    normalize : bool, default=True
        Return ``VI / H(X, Y)`` in ``[0, 1]``.

    Returns
    -------
    float
    """
    (xa, ya), n = _complete(x, y)
    if n < 4:
        return float("nan")
    xa, ya = xa[:, 0], ya[:, 0]
    if bins is None:
        rc = np.corrcoef(xa, ya)[0, 1] if xa.std() > 0 and ya.std() > 0 else 0.0
        bins = _optimal_bins(n, float(rc))
    b = int(bins)
    bx = np.minimum(((ranks(xa) - 0.5) * b / n).astype(np.int64), b - 1)
    by = np.minimum(((ranks(ya) - 0.5) * b / n).astype(np.int64), b - 1)
    joint = np.bincount(bx * b + by, minlength=b * b).astype(np.float64) / n

    def ent(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hx = ent(np.bincount(bx, minlength=b) / n)
    hy = ent(np.bincount(by, minlength=b) / n)
    hxy = ent(joint)
    vi = 2.0 * hxy - hx - hy
    if normalize:
        return vi / hxy if hxy > 0 else 0.0
    return vi


# --------------------------------------------------------------------------- #
# KSG (k-nearest-neighbour) mutual information -- class D, capped
# --------------------------------------------------------------------------- #
def _cheb(a_blk: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``(b, n)`` Chebyshev (max-norm) distances between block rows and all rows."""
    d = np.abs(a_blk[:, None, 0] - a[None, :, 0])
    for j in range(1, a.shape[1]):
        np.maximum(d, np.abs(a_blk[:, None, j] - a[None, :, j]), out=d)
    return d


def _ksg_prepare(
    arrays: tuple[np.ndarray, ...], max_n: int, seed: int
) -> tuple[list[np.ndarray], int, bool]:
    mats, n = _complete(*arrays)
    approximate = False
    if n > max_n:
        idx = np.sort(
            np.random.default_rng(seed).choice(n, size=int(max_n), replace=False)
        )
        mats = [m[idx] for m in mats]
        n = int(max_n)
        approximate = True
    return mats, n, approximate


def _block_rows(n: int) -> int:
    return max(1, min(n, 2_000_000 // max(n, 1)))


def mi_ksg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
    max_n: int = 20_000,
    seed: int = 0,
) -> float:
    """Kraskov-Stoegbauer-Grassberger mutual information (algorithm 1), nats.

    For every point, ``eps_i`` is the max-norm distance to its ``k``-th nearest
    neighbour in the joint space and ``n_x(i)``, ``n_y(i)`` count the points
    **strictly** closer than ``eps_i`` in each marginal;
    ``I = psi(k) + psi(n) - <psi(n_x + 1) + psi(n_y + 1)>``.

    Genuinely nonlinear (unlike bivariate GCMI) but a nearest-neighbour method:
    exact blocked ``O(n^2)`` search, capped at ``max_n`` rows by a seeded
    subsample. Assumes continuous data -- heavy ties make ``eps_i = 0`` and
    bias the estimate. Returns raw nats, never silently normalised (see
    :func:`mi_to_r` for a reading aid).

    Note: the neighbour counts need a **per-point** radius, so
    ``panelary._internal._numpy_stats.chebyshev_neighbour_counts`` (one scalar
    radius) does not apply; the search is a direct blocked max-norm scan.

    References
    ----------
    Kraskov, A., Stoegbauer, H. & Grassberger, P. (2004). Estimating mutual
    information. *Phys. Rev. E* 69, 066138.
    """
    (xa, ya), n, _ = _ksg_prepare((x, y), max_n, seed)
    k = int(k)
    if n <= k + 1:
        return float("nan")
    total = 0.0
    step = _block_rows(n)
    for s in range(0, n, step):
        e = min(s + step, n)
        dx = _cheb(xa[s:e], xa)
        dy = _cheb(ya[s:e], ya)
        dj = np.maximum(dx, dy)
        dj[np.arange(e - s), np.arange(s, e)] = np.inf
        eps = np.partition(dj, k - 1, axis=1)[:, k - 1][:, None]
        nx = np.maximum((dx < eps).sum(axis=1) - 1, 0)
        ny = np.maximum((dy < eps).sum(axis=1) - 1, 0)
        total += float(np.sum(psi(nx + 1.0) + psi(ny + 1.0)))
    return float(psi(float(k)) + psi(float(n)) - total / n)


def cmi_ksg(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    k: int = 5,
    max_n: int = 20_000,
    seed: int = 0,
) -> float:
    """Conditional mutual information ``I(X; Y | Z)`` (Frenzel & Pompe, 2007).

    ``I = psi(k) - <psi(n_xz + 1) + psi(n_yz + 1) - psi(n_z + 1)>`` with the
    radius from the joint ``(x, y, z)`` max-norm ``k``-NN. Same caveats and cap
    as :func:`mi_ksg`.
    """
    (xa, ya, za), n, _ = _ksg_prepare((x, y, z), max_n, seed)
    k = int(k)
    if n <= k + 1:
        return float("nan")
    total = 0.0
    step = _block_rows(n)
    for s in range(0, n, step):
        e = min(s + step, n)
        dz = _cheb(za[s:e], za)
        dxz = np.maximum(_cheb(xa[s:e], xa), dz)
        dyz = np.maximum(_cheb(ya[s:e], ya), dz)
        dj = np.maximum(dxz, dyz)
        dj[np.arange(e - s), np.arange(s, e)] = np.inf
        eps = np.partition(dj, k - 1, axis=1)[:, k - 1][:, None]
        nxz = np.maximum((dxz < eps).sum(axis=1) - 1, 0)
        nyz = np.maximum((dyz < eps).sum(axis=1) - 1, 0)
        nz = np.maximum((dz < eps).sum(axis=1) - 1, 0)
        total += float(np.sum(psi(nxz + 1.0) + psi(nyz + 1.0) - psi(nz + 1.0)))
    return float(psi(float(k)) - total / n)


# --------------------------------------------------------------------------- #
# Transfer entropy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TEResult:
    """Transfer entropy ``source -> target``.

    Attributes
    ----------
    te : float
        Nats.
    p_value : float
        chi-square(``lag``) for the Gaussian / copula estimators; circular-shift
        permutation for ``"ksg"``.
    lag : int
        Source lags ``x[t-1] .. x[t-lag]``.
    history : int
        Target lags ``y[t-1] .. y[t-history]`` conditioned on.
    n_obs : int
    estimator : str
    """

    te: float
    p_value: float
    lag: int
    history: int
    n_obs: int
    estimator: str


def _lagged(v: np.ndarray, lags: int) -> np.ndarray:
    n = v.size
    out = np.full((n, lags), np.nan)
    for j in range(1, lags + 1):
        out[j:, j - 1] = v[:-j]
    return out


def transfer_entropy_array(
    source: np.ndarray,
    target: np.ndarray,
    *,
    lag: int = 1,
    history: int = 1,
    estimator: str = "gaussian",
    k: int = 5,
    max_n: int = 5000,
    n_resamples: int = 99,
    seed: int = 0,
) -> TEResult:
    """Transfer entropy from ``source`` to ``target`` (arrays in time order).

    ``TE = I(y_t ; x_{t-1..t-lag} | y_{t-1..t-history})``.

    * ``estimator="gaussian"`` (default, cheap): Gaussian TE equals half the
      log ratio of the restricted and full OLS residual variances -- the
      Granger-causality log-likelihood ratio (Barnett, Barrett & Seth, 2009) --
      so ``2 n TE ~ chi^2(lag)`` under the null. Two OLS fits.
    * ``estimator="copula"``: the same on normal scores (rank-invariant).
    * ``estimator="ksg"``: Frenzel-Pompe KSG conditional MI, capped at
      ``max_n``; p-value from circular shifts of the source.

    **Transfer entropy is not evidence of causation.** It measures predictive
    information flow given the target's own past. Under a common latent driver
    that reaches the two series at different delays, *both* directions inflate;
    omitted confounders and too-short ``history`` do the same.

    Returns
    -------
    TEResult
    """
    if estimator not in {"gaussian", "copula", "ksg"}:
        raise ValueError(
            f"unknown `estimator` {estimator!r}; expected 'gaussian', 'copula' or 'ksg'."
        )
    lag, history = int(lag), int(history)
    if lag < 1 or history < 0:
        raise ValueError("`lag` must be >= 1 and `history` >= 0.")
    xs = np.asarray(source, dtype=np.float64).ravel()
    ys = np.asarray(target, dtype=np.float64).ravel()
    if xs.size != ys.size:
        raise ValueError("`source` and `target` must have the same length.")
    if estimator == "copula":
        fin = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs.copy(), ys.copy()
        xs[fin] = normal_scores(xs[fin])
        ys[fin] = normal_scores(ys[fin])
    xl = _lagged(xs, lag)
    yl = _lagged(ys, history) if history else np.empty((ys.size, 0))
    ok = np.isfinite(ys) & np.isfinite(xl).all(axis=1) & np.isfinite(yl).all(axis=1)
    y0, X, Yp = ys[ok], xl[ok], yl[ok]
    n = y0.size
    nan = float("nan")
    if n < lag + history + 5:
        return TEResult(nan, nan, lag, history, n, estimator)
    if estimator in {"gaussian", "copula"}:
        ones = np.ones((n, 1))
        r_res = ols(np.hstack([ones, Yp]), y0)[1]
        r_full = ols(np.hstack([ones, Yp, X]), y0)[1]
        rss_r, rss_f = float(r_res @ r_res), float(r_full @ r_full)
        te = 0.5 * math.log(rss_r / rss_f) if rss_f > 0 and rss_r > 0 else nan
        p = float(chi2_sf(2.0 * n * te, lag)) if np.isfinite(te) else nan
        return TEResult(te, p, lag, history, n, estimator)
    zc = Yp if history else np.zeros((n, 1))
    te = cmi_ksg(y0, X, zc, k=k, max_n=max_n, seed=seed)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(max(1, n // 10), max(2, n - n // 10), size=int(n_resamples))
    draws = np.array(
        [
            cmi_ksg(y0, np.roll(X, int(sh), axis=0), zc, k=k, max_n=max_n, seed=seed)
            for sh in shifts
        ]
    )
    p = (1.0 + float(np.sum(draws >= te))) / (1.0 + draws.size)
    return TEResult(te, p, lag, history, n, estimator)
