"""Column long-run variances (HAC) for score series. Private engine.

Every serial-correlation-robust test in the forecast-evaluation family reduces
to the long-run variance (LRV) of one or more *score* series (a loss
differential, a Sharpe-ratio influence function, a regression score). This
module computes those LRVs **column-wise**: one call handles ``M`` series, so
``M`` models cost one vectorised pass instead of ``M`` Python loops.

What is here
------------
* :func:`autocovariances` -- lag-``0..L`` autocovariances by a lag loop of column
  dot products (``O(LTM)``), and :func:`autocovariances_fft` -- all ``T`` lags
  by one batched, zero-padded rFFT (``O(TM log T)``), chunked over columns.
* :func:`bartlett_lrv` -- Newey-West with a fixed truncation lag; parity with the
  scalar :func:`~panelary.validation.newey_west_variance`.
* :func:`qs_kernel`, :func:`parzen_kernel` and :func:`kernel_lrv` -- the
  quadratic-spectral (infinite support, all lags via FFT) and Parzen kernels.
* :func:`andrews_ar1` / :func:`andrews_bandwidth` -- the Andrews (1991) AR(1)
  plug-in bandwidth, per column.
* :func:`prewhitened_lrv` -- Andrews & Monahan (1992) AR(1) prewhitening and
  recolouring, per column, with ``|rho|`` capped at 0.97.
* :func:`vector_lrv_prewhitened` -- the ``k x k`` prewhitened kernel LRV of a
  vector series (VAR(1) prewhitening, singular values capped at 0.97,
  multivariate AR(1) bandwidth, optional ``T/(T-k)`` factor). This is Ledoit &
  Wolf's (2008, eq. 5) recipe for the Sharpe-difference HAC standard error.
* :func:`expanding_bartlett_lrv` -- a prefix-invariant *path* of Bartlett LRVs
  with a fixed lag, from running sums (for monitoring paths).
* :func:`column_lrv` -- the dispatcher every test calls, returning the LRVs and
  a label such as ``"qs-pw(S=3.17)"`` for the result's ``hac`` field.

This is *not* an asset-covariance estimator (see the covariance package for
shrinkage); it is the spectral density at frequency zero of score series.

Conventions: autocovariances divide by ``T`` (not ``T - j``), series are
demeaned by default, input is float64 ``(T,)`` or ``(T, M)`` with no missing
values (callers group columns by availability first; see :func:`mask_groups`).

References
----------
Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
covariance matrix estimation. *Econometrica* 59, 817-858.
Andrews, D. W. K. & Monahan, J. C. (1992). An improved heteroskedasticity and
autocorrelation consistent covariance matrix estimator. *Econometrica* 60,
953-966.
Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
*Econometrica* 55, 703-708.
Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with the
Sharpe ratio. *Journal of Empirical Finance* 15, 850-859.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LRV_METHODS",
    "LRVResult",
    "andrews_ar1",
    "andrews_bandwidth",
    "autocovariances",
    "autocovariances_fft",
    "bartlett_lrv",
    "colmean",
    "coldot",
    "column_lrv",
    "expanding_bartlett_lrv",
    "kernel_lrv",
    "mask_groups",
    "newey_west_lags",
    "parzen_kernel",
    "prewhitened_lrv",
    "qs_kernel",
    "vector_lrv_prewhitened",
]

#: The LRV estimators :func:`column_lrv` accepts.
LRV_METHODS = ("bartlett", "qs", "qs_pw", "parzen", "parzen_pw")

#: Andrews & Monahan's cap on the prewhitening coefficient (and on the singular
#: values of the VAR(1) matrix): recolouring divides by ``(1 - rho)^2``, which
#: blows up as ``rho -> 1``.
_PW_CAP = 0.97

#: Andrews (1991) optimal-bandwidth constants ``c_k`` (``S = c_k (alpha T)^(1/(2q+1))``).
_ANDREWS_CONST = {"bartlett": 1.1447, "parzen": 2.6614, "qs": 1.3221}

#: Column chunk for the FFT path: 256 columns at T = 5000 is ~33 MB of complex
#: workspace.
_FFT_CHUNK = 256


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(T, M)`` float64 view, and whether the input was 1-D.

    Layout is left alone; every column reduction goes through :func:`coldot` /
    :func:`colmean`, which make columns contiguous when needed.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim != 2:
        raise ValueError(f"expected a (T,) or (T, M) array, got shape {arr.shape}.")
    return arr, False


def coldot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column dot products ``sum_t a[t, m] b[t, m]``, independent of the column count.

    numpy sums a lone contiguous column with an unrolled/pairwise kernel but
    accumulates the columns of a C-ordered matrix row by row, so a model's
    result would move in the last bit when other models join the call. Here
    every input is C-ordered and a single column is padded to two, so each
    column is always reduced the same way: its value is bitwise identical
    whether it is computed alone or with others.
    """
    if a.ndim == 1:
        return coldot(a[:, None], b[:, None])[0]
    m = a.shape[1]
    return np.einsum("tm,tm->m", _row_major(a), _row_major(b))[:m]


def _row_major(a: np.ndarray) -> np.ndarray:
    """C-ordered ``a``, with a single column duplicated (see :func:`coldot`)."""
    out = np.ascontiguousarray(a)
    if out.shape[1] == 1:
        out = np.concatenate([out, out], axis=1)
    return out


def _demeaned(arr: np.ndarray, demean: bool) -> np.ndarray:
    """C-ordered demeaned copy (one allocation), or ``arr`` untouched."""
    if not demean:
        return arr
    out = np.array(arr, dtype=np.float64, order="C", copy=True)
    out -= colmean(out)
    return out


def colmean(a: np.ndarray) -> np.ndarray:
    """Column means, bitwise independent of the column count (see :func:`coldot`)."""
    if a.ndim == 1:
        return colmean(a[:, None])[0]
    m = a.shape[1]
    return _row_major(a).mean(axis=0)[:m]


def _squeeze(out: np.ndarray, was_1d: bool) -> np.ndarray:
    return out[..., 0] if was_1d else out


def newey_west_lags(n_obs: int) -> int:
    """Newey-West's rule-of-thumb truncation lag ``floor(4 (T/100)^(2/9))``."""
    if n_obs <= 1:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


# --------------------------------------------------------------------------- #
# Autocovariances
# --------------------------------------------------------------------------- #
def autocovariances(x: np.ndarray, max_lag: int, *, demean: bool = True) -> np.ndarray:
    """Column autocovariances ``gamma_j = (1/T) sum_t x_t x_{t-j}``, ``j = 0..L``.

    A lag loop of column dot products, ``O(L T M)``; the right choice for a small
    fixed ``L`` (Bartlett, Parzen). Returns ``(L + 1, M)`` (or ``(L + 1,)``).
    """
    arr, was_1d = _as_2d(x)
    n = arr.shape[0]
    e = _demeaned(arr, demean)
    lags = int(min(max(max_lag, 0), max(n - 1, 0)))
    out = np.empty((lags + 1, arr.shape[1]), dtype=np.float64)
    out[0] = coldot(e, e) / n
    for j in range(1, lags + 1):
        out[j] = coldot(e[j:], e[:-j]) / n
    return _squeeze(out, was_1d)


def _fast_len(n: int) -> int:
    """Smallest ``2^a 3^b 5^c >= n`` (pocketfft is fastest on 5-smooth sizes)."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


def autocovariances_fft(x: np.ndarray, *, demean: bool = True) -> np.ndarray:
    """All ``T`` column autocovariances at once, by a batched zero-padded rFFT.

    ``gamma_j = (1/T) sum_t x_t x_{t-j}`` for ``j = 0..T-1``, from
    ``irfft(|rfft(x, n)|^2)`` zero-padded to a 5-smooth ``n >= 2T - 1`` (so the
    circular correlation equals the linear one). ``O(T M log T)``, chunked over 256
    columns. Agrees with :func:`autocovariances` to rounding (~1e-17 absolute on
    unit-variance data).
    """
    arr, was_1d = _as_2d(x)
    n, m = arr.shape
    e = _demeaned(arr, demean)
    nfft = _fast_len(max(2 * n - 1, 1))
    out = np.empty((n, m), dtype=np.float64)
    for s in range(0, m, _FFT_CHUNK):
        block = e[:, s : s + _FFT_CHUNK]
        spec = np.fft.rfft(block, n=nfft, axis=0)
        power = spec.real * spec.real + spec.imag * spec.imag
        out[:, s : s + _FFT_CHUNK] = np.fft.irfft(power, n=nfft, axis=0)[:n] / n
    return _squeeze(out, was_1d)


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def qs_kernel(x: np.ndarray | float) -> np.ndarray:
    """Quadratic-spectral kernel ``k(x) = 3/z^2 (sin z / z - cos z)``, ``z = 6 pi x / 5``.

    Equal to Andrews' ``25/(12 pi^2 x^2) [sin(6 pi x/5)/(6 pi x/5) - cos(6 pi x/5)]``.
    Near ``0`` a Taylor series avoids the cancellation in the bracket.
    """
    z = 1.2 * math.pi * np.abs(np.asarray(x, dtype=np.float64))
    out = np.empty_like(z)
    small = z < 0.2
    zs = z[small]
    z2 = zs * zs
    # 3 * sum_{n>=1} (-1)^{n+1} 2n z^{2n-2} / (2n+1)!
    out[small] = 1.0 - z2 * (
        1.0 / 10.0 - z2 * (1.0 / 280.0 - z2 * (1.0 / 15120.0 - z2 / 1330560.0))
    )
    zb = z[~small]
    with np.errstate(over="ignore", invalid="ignore"):
        big = 3.0 / (zb * zb) * (np.sin(zb) / zb - np.cos(zb))
    out[~small] = np.where(np.isfinite(zb), big, 0.0)
    return out


def parzen_kernel(x: np.ndarray | float) -> np.ndarray:
    """Parzen kernel (support ``|x| <= 1``)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(
        a <= 0.5,
        1.0 - 6.0 * a * a + 6.0 * a**3,
        np.where(a <= 1.0, 2.0 * (1.0 - a) ** 3, 0.0),
    )


def _bartlett_weights(lags: int) -> np.ndarray:
    j = np.arange(1, lags + 1, dtype=np.float64)
    return 1.0 - j / (lags + 1.0)


# --------------------------------------------------------------------------- #
# Bandwidth
# --------------------------------------------------------------------------- #
def andrews_ar1(x: np.ndarray, *, demean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-column AR(1) fit ``x_t = rho x_{t-1} + e_t`` by least squares.

    Returns ``(rho, sigma2)`` with ``sigma2`` the residual variance. Columns with
    no variation give ``rho = 0``.
    """
    arr, was_1d = _as_2d(x)
    e = _demeaned(arr, demean)
    lead, lag = e[1:], e[:-1]
    den = coldot(lag, lag)
    num = coldot(lead, lag)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, 0.0)
    resid = lead - lag * rho
    sigma2 = coldot(resid, resid) / max(lead.shape[0], 1)
    return _squeeze(rho, was_1d), _squeeze(sigma2, was_1d)


def andrews_bandwidth(
    x: np.ndarray, *, kernel: str = "qs", demean: bool = True
) -> np.ndarray:
    """Andrews (1991) AR(1) plug-in bandwidth, per column.

    ``S_T = 1.3221 (alpha(2) T)^{1/5}`` (QS), ``2.6614 (alpha(2) T)^{1/5}``
    (Parzen) or ``1.1447 (alpha(1) T)^{1/3}`` (Bartlett), with the scalar
    ``alpha(2) = 4 rho^2 / (1 - rho)^4`` and ``alpha(1) = 4 rho^2 / ((1 - rho)^2
    (1 + rho)^2)`` (the innovation variance cancels for one series).
    """
    if kernel not in _ANDREWS_CONST:
        raise ValueError(
            f"unknown kernel {kernel!r}; expected one of {sorted(_ANDREWS_CONST)}."
        )
    arr, was_1d = _as_2d(x)
    n = arr.shape[0]
    rho, _ = andrews_ar1(arr, demean=demean)
    rho = np.clip(np.atleast_1d(rho), -_PW_CAP, _PW_CAP)
    if kernel == "bartlett":
        alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
        s = _ANDREWS_CONST[kernel] * (alpha * n) ** (1.0 / 3.0)
    else:
        alpha = 4.0 * rho**2 / (1.0 - rho) ** 4
        s = _ANDREWS_CONST[kernel] * (alpha * n) ** 0.2
    return _squeeze(s, was_1d)


# --------------------------------------------------------------------------- #
# LRV estimators
# --------------------------------------------------------------------------- #
def bartlett_lrv(x: np.ndarray, lags: int, *, demean: bool = True) -> np.ndarray:
    """Newey-West (Bartlett) LRV per column: ``gamma_0 + 2 sum_j (1 - j/(L+1)) gamma_j``.

    Floored at 0 like :func:`~panelary.validation.newey_west_variance`, with
    which it agrees to ~1e-16 relative for every column.
    """
    if lags < 0:
        raise ValueError(f"`lags` must be >= 0, got {lags}.")
    arr, was_1d = _as_2d(x)
    lags = int(min(lags, max(arr.shape[0] - 1, 0)))
    gam = autocovariances(arr, lags, demean=demean)
    lrv = gam[0].copy()
    for j, w in enumerate(_bartlett_weights(lags), start=1):
        lrv += 2.0 * w * gam[j]
    return _squeeze(np.maximum(lrv, 0.0), was_1d)


def kernel_lrv(
    x: np.ndarray,
    bandwidth: np.ndarray | float,
    *,
    kernel: str = "qs",
    demean: bool = True,
) -> np.ndarray:
    """Kernel LRV ``sum_{|j|<T} k(j / S) gamma_j`` with a per-column bandwidth ``S``.

    ``kernel="qs"`` uses every lag (FFT autocovariances); ``"parzen"`` and
    ``"bartlett"`` (here with real-valued ``S``: weight ``1 - |j|/S``) truncate
    at ``S``. ``S = 0`` gives ``gamma_0``.
    """
    arr, was_1d = _as_2d(x)
    n, m = arr.shape
    s = np.broadcast_to(np.asarray(bandwidth, dtype=np.float64), (m,)).copy()
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError("`bandwidth` must be finite and >= 0.")
    if kernel == "qs":
        gam = autocovariances_fft(arr, demean=demean)
        j = np.arange(1, n, dtype=np.float64)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(s > 0, j / np.where(s > 0, s, 1.0), np.inf)
        w = qs_kernel(ratio)
        lrv = gam[0] + 2.0 * coldot(w, gam[1:])
    elif kernel in ("parzen", "bartlett"):
        max_lag = int(min(n - 1, math.ceil(float(s.max())) if m else 0))
        gam = autocovariances(arr, max_lag, demean=demean)
        if gam.ndim == 1:
            gam = gam[:, None]
        j = np.arange(1, max_lag + 1, dtype=np.float64)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(s > 0, j / np.where(s > 0, s, 1.0), np.inf)
        if kernel == "parzen":
            w = parzen_kernel(ratio)
        else:
            w = np.clip(1.0 - ratio, 0.0, None)
        lrv = gam[0] + 2.0 * coldot(w, gam[1:])
    else:
        raise ValueError(
            f"unknown kernel {kernel!r}; expected 'qs', 'parzen' or 'bartlett'."
        )
    return _squeeze(lrv, was_1d)


def prewhitened_lrv(
    x: np.ndarray,
    *,
    kernel: str = "qs",
    bandwidth: np.ndarray | float | None = None,
    demean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Andrews-Monahan (1992) AR(1)-prewhitened kernel LRV, per column.

    1. ``e_t = x_t - rho x_{t-1}`` with ``rho`` the column's AR(1) coefficient,
       capped at ``|rho| <= 0.97``;
    2. a kernel LRV of ``e`` (Andrews bandwidth fitted on ``e`` unless given);
    3. recolour: ``LRV_x = LRV_e / (1 - rho)^2``.

    Returns ``(lrv, bandwidth, rho)``, each ``(M,)`` (or scalar-shaped).
    """
    arr, was_1d = _as_2d(x)
    e0 = _demeaned(arr, demean)
    rho, _ = andrews_ar1(e0, demean=False)
    rho = np.clip(np.atleast_1d(rho), -_PW_CAP, _PW_CAP)
    resid = e0[1:] - e0[:-1] * rho
    if bandwidth is None:
        s = np.atleast_1d(andrews_bandwidth(resid, kernel=kernel))
    else:
        s = np.broadcast_to(
            np.asarray(bandwidth, dtype=np.float64), (arr.shape[1],)
        ).copy()
    lrv_e = np.atleast_1d(kernel_lrv(resid, s, kernel=kernel, demean=True))
    lrv = lrv_e / (1.0 - rho) ** 2
    return _squeeze(lrv, was_1d), _squeeze(s, was_1d), _squeeze(rho, was_1d)


def vector_lrv_prewhitened(
    y: np.ndarray,
    *,
    kernel: str = "qs",
    bandwidth: float | None = None,
    prewhiten: bool = True,
    small_sample: bool = True,
    demean: bool = True,
) -> tuple[np.ndarray, float]:
    """``k x k`` prewhitened kernel LRV of a vector series (Ledoit-Wolf's recipe).

    1. VAR(1) ``y_t = A y_{t-1} + v_t`` by least squares (no intercept on the
       centred series); singular values of ``A`` capped at 0.97;
    2. Andrews' multivariate AR(1) bandwidth on ``v`` (unit weights):
       ``alpha(2) = sum_a 4 rho_a^2 s_a^4 / (1 - rho_a)^8 / sum_a s_a^4 / (1 - rho_a)^4``;
    3. ``Psi_v = sum_j k(j/S) Gamma_v(j)``; recolour
       ``Psi = (I - A)^{-1} Psi_v (I - A)^{-1}'``;
    4. ``small_sample=True`` multiplies by ``T / (T - k)`` (LW 2008, eq. 5 uses
       ``T/(T-4)`` for the 4-vector).

    Returns ``(Psi, bandwidth)``.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`y` must be (T, k), got shape {arr.shape}.")
    n, k = arr.shape
    e0 = arr - arr.mean(axis=0) if demean else arr
    if prewhiten:
        lag, lead = e0[:-1], e0[1:]
        coef, *_ = np.linalg.lstsq(lag, lead, rcond=None)
        a_mat = coef.T  # lead_t = A @ lag_t
        u, sv, vt = np.linalg.svd(a_mat)
        a_mat = (u * np.minimum(sv, _PW_CAP)) @ vt
        v = lead - lag @ a_mat.T
    else:
        a_mat = np.zeros((k, k))
        v = e0
    if bandwidth is None:
        rho, sig2 = andrews_ar1(v)
        rho = np.clip(np.atleast_1d(rho), -_PW_CAP, _PW_CAP)
        sig4 = np.atleast_1d(sig2) ** 2
        num = float(np.sum(4.0 * rho**2 * sig4 / (1.0 - rho) ** 8))
        den = float(np.sum(sig4 / (1.0 - rho) ** 4))
        alpha = num / den if den > 0 else 0.0
        const = _ANDREWS_CONST[kernel]
        power = 1.0 / 3.0 if kernel == "bartlett" else 0.2
        s = float(const * (alpha * v.shape[0]) ** power)
    else:
        s = float(bandwidth)
    vc = v - v.mean(axis=0)
    nv = vc.shape[0]
    psi = vc.T @ vc / nv
    if s > 0:
        max_lag = nv - 1 if kernel == "qs" else int(min(nv - 1, math.ceil(s)))
        lags = np.arange(1, max_lag + 1, dtype=np.float64)
        if kernel == "qs":
            w = qs_kernel(lags / s)
        elif kernel == "parzen":
            w = parzen_kernel(lags / s)
        else:
            w = np.clip(1.0 - lags / s, 0.0, None)
        for j, wj in enumerate(w, start=1):
            if wj == 0.0:
                continue
            gamma = vc[j:].T @ vc[:-j] / nv
            psi = psi + wj * (gamma + gamma.T)
    inv = np.linalg.inv(np.eye(k) - a_mat)
    psi = inv @ psi @ inv.T
    if small_sample:
        psi = psi * (n / (n - k))
    return psi, s


def expanding_bartlett_lrv(x: np.ndarray, lags: int) -> np.ndarray:
    """Prefix-invariant path of Bartlett LRVs: row ``t`` uses ``x[:t + 1]`` only.

    ``gamma_j(t) = (1/n_t) sum_{s=j}^{t} (x_s - xbar_t)(x_{s-j} - xbar_t)`` is
    expanded into running sums (cumulative ``x_s x_{s-j}``, cumulative ``x``), so
    the whole ``(T, M)`` path costs ``O(T L M)`` and is a *scan*: appending rows
    never changes an earlier row. ``lags`` must be a fixed integer -- a data- or
    ``T``-dependent lag would break that property. Rows with fewer than
    ``lags + 2`` observations are ``nan``.
    """
    if not isinstance(lags, (int, np.integer)) or lags < 0:
        raise ValueError("`lags` must be a fixed non-negative integer.")
    arr, was_1d = _as_2d(x)
    n, m = arr.shape
    if n:
        # The LRV is shift-invariant; centring on the first row (a fixed value,
        # so still prefix-invariant) tames the cancellation in the expansion.
        arr = arr - arr[0]
    t = np.arange(1, n + 1, dtype=np.float64)[:, None]
    cx = np.cumsum(arr, axis=0)
    xbar = cx / t
    out = np.cumsum(arr * arr, axis=0) / t - xbar * xbar
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        prod = np.zeros((n, m))
        prod[j:] = np.cumsum(arr[j:] * arr[:-j], axis=0)
        # sum_{s=j}^{t} x_s = cx_t - cx_{j-1};  sum_{s=0}^{t-j} x_s = cx_{t-j}
        head = np.zeros((n, m))
        head[j:] = cx[j:] - cx[j - 1]
        tail = np.zeros((n, m))
        tail[j:] = cx[:-j]
        cnt = np.maximum(t - j, 0.0)
        gamma = (prod - xbar * (head + tail) + cnt * xbar * xbar) / t
        out = out + 2.0 * w * gamma
    out[: min(n, lags + 1)] = np.nan
    return _squeeze(np.maximum(out, 0.0), was_1d)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LRVResult:
    """Output of :func:`column_lrv`."""

    lrv: np.ndarray
    label: str
    bandwidth: np.ndarray | None = None
    rho: np.ndarray | None = None


def _fmt(values: np.ndarray) -> str:
    v = np.atleast_1d(values)
    if v.size == 1 or np.allclose(v, v[0]):
        return f"{float(v[0]):.3g}"
    return f"{float(np.median(v)):.3g} median"


def column_lrv(
    x: np.ndarray,
    method: str = "qs_pw",
    *,
    lags: int | None = None,
    bandwidth: np.ndarray | float | None = None,
    demean: bool = True,
) -> LRVResult:
    """Column LRVs by name, with a label for the result's ``hac`` field.

    Parameters
    ----------
    x : ndarray of shape (T,) or (T, M)
        Complete (no NaN) score series.
    method : {"bartlett", "qs", "qs_pw", "parzen", "parzen_pw"}
        ``"bartlett"`` uses the fixed lag ``lags`` (default: Newey-West's
        ``floor(4 (T/100)^(2/9))``); the others use Andrews' AR(1) bandwidth
        unless ``bandwidth`` is given; ``*_pw`` adds Andrews-Monahan
        prewhitening.
    """
    arr, was_1d = _as_2d(x)
    n = arr.shape[0]
    if method == "bartlett":
        lag = newey_west_lags(n) if lags is None else int(lags)
        lrv = bartlett_lrv(arr, lag, demean=demean)
        return LRVResult(_squeeze(lrv, was_1d), f"bartlett(L={lag})")
    if method in ("qs", "parzen"):
        s = (
            np.atleast_1d(andrews_bandwidth(arr, kernel=method, demean=demean))
            if bandwidth is None
            else np.broadcast_to(np.asarray(bandwidth, float), (arr.shape[1],))
        )
        lrv = kernel_lrv(arr, s, kernel=method, demean=demean)
        return LRVResult(
            _squeeze(lrv, was_1d),
            f"{method}(S={_fmt(s)})",
            _squeeze(np.asarray(s), was_1d),
        )
    if method in ("qs_pw", "parzen_pw"):
        kern = method[:-3]
        lrv, s, rho = prewhitened_lrv(
            arr, kernel=kern, bandwidth=bandwidth, demean=demean
        )
        return LRVResult(
            _squeeze(lrv, was_1d),
            f"{kern}-pw(S={_fmt(s)})",
            _squeeze(s, was_1d),
            _squeeze(rho, was_1d),
        )
    raise ValueError(f"unknown HAC method {method!r}; expected one of {LRV_METHODS}.")


# --------------------------------------------------------------------------- #
# Availability grouping
# --------------------------------------------------------------------------- #
def mask_groups(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group the columns of a ``(T, M)`` boolean mask by identical pattern.

    Returns ``[(rows, cols), ...]`` -- the row positions where the pattern is
    true and the columns sharing it -- largest group (most columns, then most
    rows) first; ties keep the first-seen column order. Columns with no true
    row form a group with empty ``rows``.
    """
    mk = np.asarray(mask, dtype=bool)
    if mk.ndim != 2:
        raise ValueError(f"`mask` must be 2-D, got shape {mk.shape}.")
    if mk.shape[1] == 0:
        return []
    packed = np.packbits(mk, axis=0)
    _, first, inverse = np.unique(
        packed.T, axis=0, return_index=True, return_inverse=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    groups = []
    for g in np.argsort(first, kind="stable"):
        cols = np.flatnonzero(inverse == g)
        rows = np.flatnonzero(mk[:, cols[0]])
        groups.append((rows, cols))
    groups.sort(key=lambda rc: (-rc[1].size, -rc[0].size, int(rc[1][0])))
    return groups
