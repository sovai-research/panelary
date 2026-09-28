"""Kernel dependence: random-Fourier-feature HSIC.

``HSIC ~= || Zx^T Zy / n ||_F^2`` with ``Zx``, ``Zy`` column-centred random
Fourier features of a Gaussian kernel -- two GEMMs, ``O(n D^2)``, no ``n x n``
Gram matrix. ``D`` is the accuracy/cost knob (``D = 256`` for a pair,
``32-64`` for a full matrix).

A bandwidth is a fit
--------------------
The median-heuristic bandwidth **and** the drawn frequencies ``(w, b)`` are
fitted parameters (build-contract invariant 7): :class:`RFFMap` freezes both
at :meth:`RFFMap.fit` and reuses them in :meth:`RFFMap.transform`, and every
entry of :func:`hsic_matrix` uses one shared map so the entries are comparable.
By default the data are first mapped to **normal scores** (the copula
transform), whose marginal is exactly standard normal whatever the input; the
bandwidth is then the constant ``1`` -- no fitted quantity depends on the
sample at all, which is what makes the rolling / resampled uses leak-free.

One engine, two names
---------------------
With distance-induced kernels HSIC and distance covariance are the **same**
statistic (Sejdinovic, Sriperumbudur, Gretton & Fukumizu, 2013). This module
ships the Gaussian-kernel version because its random-feature approximation is
matmul-shaped; :func:`~panelary.depend.dcor` is the exact distance version.
They are not two independent pieces of evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from panelary.depend._kernels import Kernel, _fisher_var
from panelary.depend._null import gamma_pvalue
from panelary.depend._ranks import normal_scores

RowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
ClosedForm = Callable[[float, np.ndarray, np.ndarray], "tuple[float, str]"]

__all__ = [
    "HSICResult",
    "RFFMap",
    "hsic",
    "hsic_kernel",
    "hsic_matrix",
    "median_heuristic",
    "rff",
]

#: Bandwidth used on normal scores (the median heuristic of N(0, 1) is 0.954).
COPULA_BANDWIDTH = 1.0
_MAX_ELEMS = 4_000_000


def median_heuristic(x: np.ndarray, *, max_points: int = 1000) -> float:
    """Median pairwise Euclidean distance on up to ``max_points`` evenly spaced
    rows (deterministic, no RNG)."""
    a = np.asarray(x, dtype=np.float64)
    a = a[:, None] if a.ndim == 1 else a
    a = a[np.isfinite(a).all(axis=1)]
    if a.shape[0] < 2:
        return 1.0
    if a.shape[0] > max_points:
        a = a[np.linspace(0, a.shape[0] - 1, max_points).round().astype(np.int64)]
    sq = (a * a).sum(1)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * a @ a.T, 0.0, None)
    iu = np.triu_indices(a.shape[0], 1)
    med = float(np.sqrt(np.median(d2[iu])))
    return med if med > 0 else 1.0


@dataclass(frozen=True)
class RFFMap:
    """A frozen random-Fourier-feature map ``z(x) = sqrt(2/D) cos(x W / s + b)``.

    Attributes
    ----------
    bandwidth : float
        Gaussian-kernel bandwidth ``s`` (``k(x, x') = exp(-|x - x'|^2 / (2 s^2))``).
    W : numpy.ndarray
        ``(d, D)`` standard-normal frequencies.
    b : numpy.ndarray
        ``(D,)`` phases, uniform on ``[0, 2 pi)``.
    copula : bool
        Whether inputs are mapped to normal scores first.
    """

    bandwidth: float
    W: np.ndarray
    b: np.ndarray
    copula: bool

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        *,
        n_features: int = 256,
        bandwidth: float | None = None,
        copula: bool = True,
        seed: int = 0,
    ) -> RFFMap:
        """Fit the map on training data: bandwidth (if not given) and frequencies.

        With ``copula=True`` and no ``bandwidth``, the bandwidth is the constant
        :data:`COPULA_BANDWIDTH` -- nothing is estimated from ``x`` but its
        dimension.
        """
        a = np.asarray(x, dtype=np.float64)
        d = 1 if a.ndim == 1 else a.shape[1]
        if bandwidth is None:
            bw = COPULA_BANDWIDTH if copula else median_heuristic(a)
        else:
            bw = float(bandwidth)
        if not bw > 0:
            raise ValueError(f"`bandwidth` must be positive, got {bw}.")
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((d, int(n_features)))
        b = rng.uniform(0.0, 2.0 * math.pi, int(n_features))
        return cls(bw, W, b, copula)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """``(n, D)`` features (normal scores taken within ``x`` if ``copula``)."""
        a = np.asarray(x, dtype=np.float64)
        a = a[:, None] if a.ndim == 1 else a
        if self.copula:
            a = normal_scores(a, axis=0)
        return math.sqrt(2.0 / self.b.size) * np.cos(
            a @ self.W / self.bandwidth + self.b
        )


def rff(
    x: np.ndarray,
    *,
    n_features: int = 256,
    bandwidth: float | None = None,
    copula: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Random Fourier features of ``x`` in one call (fit + transform on ``x``).

    For anything reused across folds or pairs, fit an :class:`RFFMap` once.
    """
    return RFFMap.fit(
        x, n_features=n_features, bandwidth=bandwidth, copula=copula, seed=seed
    ).transform(x)


@dataclass(frozen=True)
class HSICResult:
    """RFF-HSIC of one pair.

    Attributes
    ----------
    hsic : float
        ``|| Cxy ||_F^2`` with ``Cxy = Zx_c^T Zy_c / n`` (biased HSIC).
    normalised : float
        ``hsic / sqrt(|| Cxx ||_F^2 || Cyy ||_F^2)`` in ``[0, 1]``.
    p_value : float
        Gamma approximation with the permutation mean and the Gaussian-limit
        variance of ``hsic``.
    n_obs : int
    n_features : int
    bandwidth_x, bandwidth_y : float
    """

    hsic: float
    normalised: float
    p_value: float
    n_obs: int
    n_features: int
    bandwidth_x: float
    bandwidth_y: float


def _moments(
    zx: np.ndarray, zy: np.ndarray
) -> tuple[float, float, float, float, float]:
    n = zx.shape[0]
    cxy = zx.T @ zy / n
    sx = zx.T @ zx / n
    sy = zy.T @ zy / n
    stat = float((cxy * cxy).sum())
    fx = float((sx * sx).sum())
    fy = float((sy * sy).sum())
    mean = float(np.trace(sx) * np.trace(sy)) / (n - 1.0)
    var = 2.0 * fx * fy / (n * n)
    return stat, fx, fy, mean, var


def hsic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int = 256,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    copula: bool = True,
    seed: int = 0,
) -> HSICResult:
    """Hilbert-Schmidt independence criterion via random Fourier features.

    Parameters
    ----------
    x, y : array_like
        ``(n,)`` or ``(n, d)``; rows with any non-finite value are dropped.
    n_features : int, default=256
        ``D``. Cost ``O(n D^2)``; measured 312 ms at ``n = 100 000``, ``D = 256``
        in the build contract.
    bandwidth_x, bandwidth_y : float, optional
        Gaussian bandwidths. Default: 1 on normal scores (``copula=True``), the
        median heuristic otherwise.
    copula : bool, default=True
        Map each margin to normal scores first (rank-invariant, robust to fat
        tails, and no fitted bandwidth).
    seed : int, default=0
        Seed for the frequencies (``x`` uses ``seed``, ``y`` uses ``seed + 1``).

    Returns
    -------
    HSICResult

    Notes
    -----
    With a distance-induced kernel HSIC *is* distance covariance (Sejdinovic et
    al., 2013): use one or the other, not both as separate evidence.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xa = xa[:, None] if xa.ndim == 1 else xa
    ya = ya[:, None] if ya.ndim == 1 else ya
    keep = np.isfinite(xa).all(axis=1) & np.isfinite(ya).all(axis=1)
    xa, ya = xa[keep], ya[keep]
    n = xa.shape[0]
    if n < 4:
        nan = float("nan")
        return HSICResult(nan, nan, nan, n, int(n_features), nan, nan)
    mx = RFFMap.fit(
        xa, n_features=n_features, bandwidth=bandwidth_x, copula=copula, seed=seed
    )
    my = RFFMap.fit(
        ya, n_features=n_features, bandwidth=bandwidth_y, copula=copula, seed=seed + 1
    )
    zx = mx.transform(xa)
    zy = my.transform(ya)
    zx -= zx.mean(axis=0)
    zy -= zy.mean(axis=0)
    stat, fx, fy, mean, var = _moments(zx, zy)
    norm = stat / math.sqrt(fx * fy) if fx > 0 and fy > 0 else float("nan")
    return HSICResult(
        stat,
        norm,
        gamma_pvalue(stat, mean, var),
        n,
        int(n_features),
        mx.bandwidth,
        my.bandwidth,
    )


def _hsic_rows_factory(n_features: int, seed: int) -> tuple[RowFn, ClosedForm]:
    rng = np.random.default_rng(seed)
    wx = rng.standard_normal(n_features) / COPULA_BANDWIDTH
    bx = rng.uniform(0.0, 2.0 * math.pi, n_features)
    wy = rng.standard_normal(n_features) / COPULA_BANDWIDTH
    by = rng.uniform(0.0, 2.0 * math.pi, n_features)
    scale = math.sqrt(2.0 / n_features)

    def rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        bsz, m = X.shape
        out = np.empty(bsz)
        step = max(1, _MAX_ELEMS // max(m * n_features, 1))
        for s in range(0, bsz, step):
            zx = scale * np.cos(
                normal_scores(X[s : s + step], axis=-1)[..., None] * wx + bx
            )
            zy = scale * np.cos(
                normal_scores(Y[s : s + step], axis=-1)[..., None] * wy + by
            )
            zx -= zx.mean(axis=1, keepdims=True)
            zy -= zy.mean(axis=1, keepdims=True)
            cxy = np.einsum("bmd,bme->bde", zx, zy) / m
            sx = np.einsum("bmd,bme->bde", zx, zx) / m
            sy = np.einsum("bmd,bme->bde", zy, zy) / m
            num = (cxy * cxy).sum(axis=(1, 2))
            den = np.sqrt((sx * sx).sum(axis=(1, 2)) * (sy * sy).sum(axis=(1, 2)))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[s : s + step] = np.where(
                    den > 0, num / np.where(den > 0, den, 1.0), np.nan
                )
        return out

    def closed_form(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
        zx = scale * np.cos(normal_scores(x)[:, None] * wx + bx)
        zy = scale * np.cos(normal_scores(y)[:, None] * wy + by)
        zx -= zx.mean(axis=0)
        zy -= zy.mean(axis=0)
        stat, _, _, mean, var = _moments(zx, zy)
        return gamma_pvalue(stat, mean, var), "gamma"

    return rows, closed_form


def hsic_kernel(*, n_features: int = 64, seed: int = 0) -> Kernel:
    """The engine kernel for ``method="hsic"``: normalised RFF-HSIC on normal
    scores with frequencies frozen once (shared by the estimate and every null
    draw), gamma closed form."""
    rows, closed_form = _hsic_rows_factory(int(n_features), int(seed))
    return Kernel(
        "hsic",
        rows,
        "greater",
        False,
        30,
        closed_form,
        "mean",
        _fisher_var,
        estimator=f"normalised rff-hsic (copula, D={int(n_features)}, seed={int(seed)})",
    )


def hsic_matrix(A: np.ndarray, *, n_features: int = 64, seed: int = 0) -> np.ndarray:
    """``(p, p)`` normalised RFF-HSIC matrix with **one** frozen frequency draw.

    Every column is mapped to normal scores and through the same ``(w, b)``, so
    entries are comparable; the ``p D x p D`` feature Gram is one GEMM.
    """
    arr = np.asarray(A, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    n, p = arr.shape
    if n < 4:
        return np.full((p, p), np.nan)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_features) / COPULA_BANDWIDTH
    b = rng.uniform(0.0, 2.0 * math.pi, n_features)
    Z = math.sqrt(2.0 / n_features) * np.cos(
        normal_scores(arr, axis=0)[..., None] * w + b
    )
    Z -= Z.mean(axis=0)
    G = Z.reshape(n, p * n_features).T @ Z.reshape(n, p * n_features) / n
    G = G.reshape(p, n_features, p, n_features)
    H = np.einsum("idje,idje->ij", G, G)
    d = np.sqrt(np.clip(np.diag(H), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return H / np.outer(d, d)
