"""Multivariate proper scores for ensemble forecasts: energy and variogram scores.

* :func:`energy_score` (Gneiting & Raftery, 2007; Gneiting et al., 2008) is the
  multivariate generalisation of the CRPS::

      ES = (1/m) sum_k ||X_k - y|| - 1/(2 m^2) sum_{k,l} ||X_k - X_l||

  It is strictly proper, but known to be weak at detecting a misspecified
  *dependence* structure (Pinson & Tastu, 2013).
* :func:`variogram_score` (Scheuerer & Hamill, 2015) targets exactly that
  weakness. It compares the pairwise variogram of the observation with the
  ensemble's expected variogram::

      VS_p = sum_{i<j} w_ij (|y_i - y_j|^p - (1/m) sum_k |X_ki - X_kj|^p)^2

  It is proper (not strictly), and it is sensitive to correlation errors that
  the energy score barely sees. Report both.

Both return one score per observation (lower is better), ready for
Diebold–Mariano or the model confidence set. Work is chunked over
observations so no temporary exceeds ``chunk_bytes``. numpy only; the
variogram score has an optional numba kernel (the ``fast`` extra) that fuses
the ensemble mean and never builds the ``(m, d(d-1)/2)`` temporary.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np

from panelary._internal._jit import lazy_njit
from panelary._internal._special import lgamma
from panelary.validation._forecast_tests import crps_ensemble

__all__ = ["energy_score", "variogram_score"]

_EPS = float(np.finfo(np.float64).eps)
#: Largest ensemble for which ``method="direct"`` is accepted.
_DIRECT_MAX_M = 256
#: Power codes shared by the variogram kernel and its numpy twin.
_POW_SQRT, _POW_ABS, _POW_SQUARE, _POW_GENERAL = 0, 1, 2, 3


def _check_inputs(
    y: np.ndarray | Sequence[Sequence[float]],
    samples: np.ndarray,
    chunk_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``y (T, d)``, ``samples (T, m, d)`` and the finite-row mask.

    Rows with any non-finite value are zero-filled here and set to NaN by the
    caller.
    """
    yy = np.asarray(y, dtype=np.float64)
    xs = np.asarray(samples, dtype=np.float64)
    if yy.ndim == 1:
        yy = yy[:, None]
        if xs.ndim == 2:
            xs = xs[:, :, None]
    if yy.ndim != 2 or xs.ndim != 3:
        raise ValueError(
            f"`y` must be (T, d) and `samples` (T, m, d); got {yy.shape} and "
            f"{xs.shape}."
        )
    if xs.shape[0] != yy.shape[0] or xs.shape[2] != yy.shape[1]:
        raise ValueError(
            f"`samples` {xs.shape} does not match `y` {yy.shape} (expected "
            f"({yy.shape[0]}, m, {yy.shape[1]}))."
        )
    if xs.shape[1] < 1:
        raise ValueError("`samples` needs at least one ensemble member.")
    if int(chunk_bytes) < 1:
        raise ValueError(f"`chunk_bytes` must be positive, got {chunk_bytes}.")
    ok = np.asarray(
        np.isfinite(yy).all(axis=1) & np.isfinite(xs).all(axis=(1, 2)), dtype=bool
    )
    if not ok.all():
        yy = np.where(ok[:, None], yy, 0.0)
        xs = np.where(ok[:, None, None], xs, 0.0)
    return yy, xs, ok


def _chunks(n: int, per_row_bytes: int, chunk_bytes: int) -> range:
    step = max(1, int(chunk_bytes) // max(per_row_bytes, 1))
    return range(0, n, step)


# --------------------------------------------------------------------------- #
# Energy score
# --------------------------------------------------------------------------- #
def _energy_exact(
    y: np.ndarray, x: np.ndarray, fair: bool, chunk_bytes: int
) -> np.ndarray:
    t, m, _ = x.shape
    out = np.empty(t)
    denom = 2.0 * m * (m - 1) if fair else 2.0 * m * m
    step = _chunks(t, 3 * 8 * m * m, chunk_bytes).step
    for s0 in range(0, t, step):
        s1 = min(t, s0 + step)
        xb = x[s0:s1]
        # The score is translation invariant: centring on the ensemble mean keeps
        # the Gram entries small and the cancellation in s_k + s_l - 2 G_kl mild.
        mu = xb.mean(axis=1, keepdims=True)
        xc = xb - mu
        yc = y[s0:s1, None, :] - mu
        term1 = np.sqrt(((xc - yc) ** 2).sum(axis=-1)).mean(axis=1)
        d2 = xc @ np.swapaxes(xc, 1, 2)  # the Gram matrix, turned into D^2 in place
        sq = np.einsum("bkk->bk", d2).copy()
        d2 *= -2.0
        d2 += sq[:, :, None]
        d2 += sq[:, None, :]
        # Shift by the rounding floor and clamp: rounding noise (exact duplicate
        # members) becomes exactly 0 instead of sqrt(eps) * ||x||, while every
        # other distance moves by a relative tol / (2 D^2), i.e. ~eps.
        tol = 16.0 * _EPS * sq.max(axis=1)
        d2 -= tol[:, None, None]
        np.maximum(d2, 0.0, out=d2)
        np.sqrt(d2, out=d2)
        out[s0:s1] = term1 - d2.sum(axis=(1, 2)) / denom
    return out


def _energy_direct(
    y: np.ndarray, x: np.ndarray, fair: bool, chunk_bytes: int
) -> np.ndarray:
    t, m, d = x.shape
    out = np.empty(t)
    denom = 2.0 * m * (m - 1) if fair else 2.0 * m * m
    step = _chunks(t, 8 * m * m * d, chunk_bytes).step
    for s0 in range(0, t, step):
        s1 = min(t, s0 + step)
        xb = x[s0:s1]
        term1 = np.sqrt(((xb - y[s0:s1, None, :]) ** 2).sum(axis=-1)).mean(axis=1)
        diff = xb[:, :, None, :] - xb[:, None, :, :]
        pair = np.sqrt((diff * diff).sum(axis=-1)).sum(axis=(1, 2))
        out[s0:s1] = term1 - pair / denom
    return out


def _crps_fair_aware(y: np.ndarray, x: np.ndarray, fair: bool) -> np.ndarray:
    """CRPS of ``(n,)`` outcomes against ``(n, m)`` ensembles via ``crps_ensemble``.

    The fair version rescales the ensemble-spread term by ``m / (m - 1)``.
    """
    crps = crps_ensemble(y, x)
    if not fair:
        return crps
    m = x.shape[1]
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    return term1 - (term1 - crps) * (m / (m - 1.0))


def _sphere_directions(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``(d, k)`` unit directions, orthonormal within blocks of ``d`` (Haar)."""
    cols: list[np.ndarray] = []
    left = k
    while left > 0:
        size = min(d, left)
        q, r = np.linalg.qr(rng.standard_normal((d, size)))
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        cols.append(q * signs)
        left -= size
    return np.concatenate(cols, axis=1)


def _energy_sliced(
    y: np.ndarray,
    x: np.ndarray,
    fair: bool,
    n_projections: int,
    seed: int,
    chunk_bytes: int,
) -> np.ndarray:
    t, m, d = x.shape
    c_d = math.sqrt(math.pi) * math.exp(
        float(lgamma((d + 1) / 2.0)) - float(lgamma(d / 2.0))
    )
    theta = _sphere_directions(d, n_projections, np.random.default_rng(seed))
    out = np.empty(t)
    step = _chunks(t, 3 * 8 * m * n_projections, chunk_bytes).step
    for s0 in range(0, t, step):
        s1 = min(t, s0 + step)
        proj = x[s0:s1] @ theta  # (c, m, K)
        py = y[s0:s1] @ theta  # (c, K)
        c = s1 - s0
        flat = np.swapaxes(proj, 1, 2).reshape(c * n_projections, m)
        crps = _crps_fair_aware(py.ravel(), flat, fair).reshape(c, n_projections)
        out[s0:s1] = c_d * crps.mean(axis=1)
    return out


def energy_score(
    y: np.ndarray | Sequence[Sequence[float]],
    samples: np.ndarray,
    *,
    method: str = "exact",
    fair: bool = False,
    n_projections: int = 128,
    seed: int = 0,
    chunk_bytes: int = 256 * 2**20,
) -> np.ndarray:
    """Energy score of multivariate ensemble forecasts, one value per observation.

    ``ES = (1/m) sum_k ||X_k - y|| - 1/(2 m^2) sum_{k,l} ||X_k - X_l||`` (Gneiting
    & Raftery, 2007). ``fair=True`` uses ``1/(2 m (m - 1))`` in the second term,
    Ferro's (2014) fair version. It is an unbiased estimate of the score of the
    distribution the ensemble was drawn from, so ensembles of different sizes
    compare fairly.

    Parameters
    ----------
    y : array-like of shape (T, d)
        Observations. A ``(T,)`` input is read as ``d = 1``.
    samples : array-like of shape (T, m, d)
        ``m`` ensemble members per observation (``(T, m)`` when ``d = 1``).
    method : {"exact", "direct", "sliced"}, default "exact"
        ``"exact"`` computes all pairwise distances through the Gram trick
        ``||a - b||^2 = |a|^2 + |b|^2 - 2 a.b``, after centring each ensemble on its
        mean. The cost is O(m^2 d) per observation, and no ``(m, m, d)``
        temporary is built. Squared distances are shifted down by the rounding
        floor ``16 eps max_k |x_k|^2`` and clamped at 0, so exact duplicate
        members contribute exactly 0. Near-duplicates keep a relative error of
        about ``eps ||x||^2 / ||x_k - x_l||^2``. ``"direct"`` differences every pair
        explicitly (``m <= 256``; a reference path for tests). ``"sliced"`` uses
        ``||v|| = c_d E_theta |theta' v|`` to average the exact 1-D CRPS over
        ``n_projections`` random directions: O(K m (d + log m)) per observation.
        It is an unbiased Monte Carlo approximation, and a ``UserWarning`` says so.
        For ``d = 1`` every method is the exact sorted-sample CRPS
        (:func:`~panelary.validation.crps_ensemble`).
    fair : bool, default False
        Use the fair ensemble estimator.
    n_projections : int, default 128
        Directions for ``method="sliced"``. They are drawn once per call,
        orthonormal within blocks of ``d``.
    seed : int, default 0
        Seed for the sliced directions.
    chunk_bytes : int, default 256 MiB
        Memory cap for the per-chunk temporaries.

    Returns
    -------
    numpy.ndarray of shape (T,)
        Per-observation energy score (lower is better). NaN where ``y`` or any
        member has a non-finite value.

    References
    ----------
    Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. *JASA* 102(477), 359–378.

    Gneiting, T., Stanberry, L., Grimit, E., Held, L. & Johnson, N. (2008).
    Assessing probabilistic forecasts of multivariate quantities. *TEST* 17(2),
    211–235.

    Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *QJRMS* 140,
    1917–1923.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal((4, 3))
    >>> ens = rng.standard_normal((4, 50, 3))
    >>> energy_score(y, ens).shape
    (4,)
    """
    if method not in ("exact", "direct", "sliced"):
        raise ValueError(
            f"`method` must be 'exact', 'direct' or 'sliced', got {method!r}."
        )
    yy, xs, ok = _check_inputs(y, samples, chunk_bytes)
    t, m, d = xs.shape
    if fair and m < 2:
        raise ValueError("fair=True needs at least 2 ensemble members.")
    if d == 1:
        out = _crps_fair_aware(yy[:, 0], xs[:, :, 0], fair)
    elif method == "exact":
        out = _energy_exact(yy, xs, fair, chunk_bytes)
    elif method == "direct":
        if m > _DIRECT_MAX_M:
            raise ValueError(
                f"method='direct' builds (m, m, d) differences; it is limited to "
                f"m <= {_DIRECT_MAX_M} (got m={m}). Use method='exact'."
            )
        out = _energy_direct(yy, xs, fair, chunk_bytes)
    else:
        k = int(n_projections)
        if k < 1:
            raise ValueError(f"`n_projections` must be >= 1, got {n_projections}.")
        warnings.warn(
            f"energy_score(method='sliced') is an unbiased Monte Carlo "
            f"approximation from {k} random directions, not the exact score",
            UserWarning,
            stacklevel=2,
        )
        out = _energy_sliced(yy, xs, fair, k, seed, chunk_bytes)
    return np.where(ok, out, np.nan)


# --------------------------------------------------------------------------- #
# Variogram score
# --------------------------------------------------------------------------- #
@lazy_njit
def _variogram_kernel(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    p: float,
    code: int,
    out: np.ndarray,
) -> None:  # pragma: no cover - exercised only when numba is installed
    t_n, m, d = x.shape
    n_pairs = d * (d - 1) // 2
    acc = np.empty(n_pairs, dtype=np.float64)
    for t in range(t_n):
        for q in range(n_pairs):
            acc[q] = 0.0
        for k in range(m):
            pos = 0
            for i in range(d - 1):
                xi = x[t, k, i]
                for j in range(i + 1, d):
                    diff = abs(xi - x[t, k, j])
                    if code == 0:
                        val = math.sqrt(diff)
                    elif code == 1:
                        val = diff
                    elif code == 2:
                        val = diff * diff
                    else:
                        val = diff**p
                    acc[pos] = acc[pos] + val
                    pos += 1
        total = 0.0
        pos = 0
        for i in range(d - 1):
            for j in range(i + 1, d):
                diff = abs(y[t, i] - y[t, j])
                if code == 0:
                    yv = math.sqrt(diff)
                elif code == 1:
                    yv = diff
                elif code == 2:
                    yv = diff * diff
                else:
                    yv = diff**p
                r = yv - acc[pos] / m
                total = total + w[i, j] * (r * r)
                pos += 1
        out[t] = total


def _pow(a: np.ndarray, p: float, code: int) -> np.ndarray:
    if code == _POW_SQRT:
        return np.sqrt(a)
    if code == _POW_ABS:
        return a
    if code == _POW_SQUARE:
        return a * a
    return np.power(a, p)


def _variogram_numpy(
    y: np.ndarray, x: np.ndarray, w: np.ndarray, p: float, code: int, chunk_bytes: int
) -> np.ndarray:
    """Numpy twin of :func:`_variogram_kernel` (same summation order, bitwise).

    Members are accumulated one at a time (member 0, 1, 2, ...) into a
    ``(chunk, d(d-1)/2)`` buffer, so no ``(m, d(d-1)/2)`` temporary is built,
    and the pair contributions are summed sequentially (``cumsum``), exactly as
    the kernel does.
    """
    t, m, d = x.shape
    iu, ju = np.triu_indices(d, 1)
    n_pairs = iu.shape[0]
    out = np.zeros(t)
    if n_pairs == 0:
        return out
    wv = w[iu, ju]
    step = _chunks(t, 8 * (3 * n_pairs + m * d), chunk_bytes).step
    for s0 in range(0, t, step):
        s1 = min(t, s0 + step)
        members = np.ascontiguousarray(np.swapaxes(x[s0:s1], 0, 1))  # (m, c, d)
        acc = np.zeros((s1 - s0, n_pairs))
        for k in range(m):
            xk = members[k]
            acc += _pow(np.abs(xk[:, iu] - xk[:, ju]), p, code)
        yb = y[s0:s1]
        r = _pow(np.abs(yb[:, iu] - yb[:, ju]), p, code) - acc / m
        contrib = wv * (r * r)
        out[s0:s1] = np.cumsum(contrib, axis=1)[:, -1]
    return out


def variogram_score(
    y: np.ndarray | Sequence[Sequence[float]],
    samples: np.ndarray,
    *,
    p: float = 0.5,
    weights: np.ndarray | None = None,
    chunk_bytes: int = 256 * 2**20,
) -> np.ndarray:
    """Variogram score of order ``p`` (Scheuerer & Hamill, 2015).

    ``VS_p = sum_{i<j} w_ij (|y_i - y_j|^p - (1/m) sum_k |X_ki - X_kj|^p)^2``.

    It compares the observed pairwise "variogram" with the ensemble's expected
    one, so it is sensitive to misspecified correlations, where the energy score
    is known to be weak (Pinson & Tastu, 2013). It is proper but not strictly
    proper: it cannot see a common shift of every component. Use it alongside
    :func:`energy_score`, not instead of it. ``p = 0.5`` is the authors'
    recommendation for robustness to outliers.

    Parameters
    ----------
    y : array-like of shape (T, d)
        Observations.
    samples : array-like of shape (T, m, d)
        ``m`` ensemble members per observation.
    p : float, default 0.5
        Order of the variogram (``> 0``). ``p`` in ``{0.5, 1, 2}`` takes exact
        fast paths (``sqrt``, ``abs``, square).
    weights : array-like of shape (d, d), optional
        Non-negative pair weights ``w_ij``. Only the upper triangle is used.
        The default is all ones.
    chunk_bytes : int, default 256 MiB
        Memory cap for the numpy path's per-chunk temporaries.

    Returns
    -------
    numpy.ndarray of shape (T,)
        Per-observation score (lower is better). NaN where ``y`` or any member
        has a non-finite value.

    Notes
    -----
    With numba installed (the ``fast`` extra) a fused kernel accumulates the
    ensemble mean in place. Its output matches the numpy path bitwise for
    ``p`` in ``{0.5, 1, 2}``. For other ``p`` the two may differ in the last
    bits (numba's and numpy's ``pow`` are different implementations).

    References
    ----------
    Scheuerer, M. & Hamill, T. M. (2015). Variogram-based proper scoring rules
    for probabilistic forecasts of multivariate quantities. *MWR* 143(4),
    1321–1334.
    """
    pv = float(p)
    if not (pv > 0.0 and math.isfinite(pv)):
        raise ValueError(f"`p` must be a positive finite number, got {p}.")
    yy, xs, ok = _check_inputs(y, samples, chunk_bytes)
    d = yy.shape[1]
    if weights is None:
        w = np.ones((d, d))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (d, d):
            raise ValueError(f"`weights` must have shape ({d}, {d}), got {w.shape}.")
        upper = w[np.triu_indices(d, 1)]
        if not np.all(np.isfinite(upper)) or np.any(upper < 0):
            raise ValueError("`weights` must be finite and non-negative.")
    code = {0.5: _POW_SQRT, 1.0: _POW_ABS, 2.0: _POW_SQUARE}.get(pv, _POW_GENERAL)
    kernel = _variogram_kernel.compiled()
    if kernel is not None:
        out = np.empty(yy.shape[0])
        kernel(
            np.ascontiguousarray(yy),
            np.ascontiguousarray(xs),
            np.ascontiguousarray(w),
            pv,
            code,
            out,
        )
    else:
        out = _variogram_numpy(yy, xs, w, pv, code, chunk_bytes)
    return np.where(ok, out, np.nan)
