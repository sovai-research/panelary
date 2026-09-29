"""Sharpe-ratio inference that holds its size under fat tails and serial dependence.

The textbook tests are wrong in known directions. Jobson-Korkie with Memmel's
correction (JKM) assumes i.i.d. bivariate normal returns; Ledoit & Wolf (2008,
Table 1) measure it rejecting a true null of equal Sharpe ratios **10.7 %** of
the time at nominal 5 % under t6 returns and **9.5 %** under AR(1) returns with
``phi = 0.2`` (a typical hedge-fund autocorrelation). HAC inference is
consistent but "often liberal in finite samples". The studentized circular-block
bootstrap (LW's "Boot-TS") holds its size in both settings.

One engine: the influence function
-----------------------------------
For ``SR = mu / sigma`` the influence series is
``psi_t = z_t - (SR / 2)(z_t^2 - 1)`` with ``z_t = (r_t - mu) / sigma``, and
``Var(psi) = 1 - g3 SR + (g4 - 1)/4 SR^2`` is the Mertens / Opdyke formula.
Every standard error here is the variance of an influence series, estimated one
way or another:

============== ============================================== ===================
method         estimator of ``Var(sqrt(T) SR_hat)``            reference
============== ============================================== ===================
``iid_normal`` ``1 + SR^2 / 2`` (Lo 2002 "IID"; JKM for Δ)      ``N(0,1)``
``iid``        ``1 - g3 SR + (g4 - 1)/4 SR^2`` (Mertens/Opdyke) ``N(0,1)``
``hac``        ``LRV(psi)``: prewhitened QS by default          ``N(0,1)``
``bootstrap``  studentized circular-block bootstrap (LW 2008)  resampling
``exact_norm`` ``sqrt(T) SR_hat ~ nct(T-1, sqrt(T) SR)``        exact if i.i.d. N
============== ============================================== ===================

For a *difference* the influence series is ``psi_i - psi_b``; for the
log-variance statistic (Ledoit & Wolf 2011) it is ``z^2 - 1``.

Bootstrap: the Goetze-Kuensch studentized circular-block bootstrap of LW (2008,
§3.2.2), computed without materialising a single resample: one shared matrix of
block-start counts ``N (B, T)`` and circular block sums of the centred returns
and squared returns turn every replicate moment into a BLAS product (see
:mod:`panelary.validation._resample`). The p-value is LW eq. (9),
``(#{|Δ* - Δ̂| / s(Δ*) >= |Δ̂| / s(Δ̂)} + 1) / (B + 1)``.

Conventions
-----------
* Sharpe ratios are **per period** and use the sample standard deviation with
  ``ddof = 1`` (as :func:`panelary.core.model_selection._sharpe`); influence
  series and gradients use the plug-in (``ddof = 0``) moments, as LW do.
* Degenerate input (zero variance, too few observations) gives ``nan`` plus a
  ``warnings`` entry, never 0.
* Columns with different missing-value patterns are grouped by availability;
  each group is analysed on its own common dates. A multiplicity adjustment by
  Romano-Wolf runs within the largest group only.

References
----------
Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with the
Sharpe ratio. *Journal of Empirical Finance* 15, 850-859.
Ledoit, O. & Wolf, M. (2011). Robust performances hypothesis testing with the
variance. *Wilmott* 55, 86-89.
Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*
58(4), 36-52.
Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio.
*Finance Letters* 1, 21-23.
Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002).
Opdyke, J. D. (2007). Comparing Sharpe ratios: so where are the p-values?
*Journal of Asset Management* 8(5), 308-336.
Goetze, F. & Kuensch, H. R. (1996). Second-order correctness of the blockwise
bootstrap for stationary observations. *Annals of Statistics* 24, 1914-1933.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from panelary._internal import _special
from panelary.validation import _hac
from panelary.validation._arrays import (
    as_matrix,
    as_vector,
    check_alternative,
    normal_pvalue,
    resample_pvalue,
    resolve_names,
)
from panelary.validation._resample import (
    DEFAULT_CHUNK_BYTES,
    CircularBlockDraw,
    circular_block_draw,
    circular_block_sums,
    matmul_cols,
    row_chunks,
    start_counts,
    stationary_indices,
)
from panelary.validation._results import EvaluationResult
from panelary.validation._selection_stats import (
    benjamini_hochberg,
    benjamini_yekutieli,
    holm_bonferroni,
    romano_wolf,
)

__all__ = [
    "BlockLengthCalibration",
    "annualize_sharpe",
    "sharpe_block_length",
    "sharpe_ratio_inference",
    "sharpe_ratio_test",
]

_SINGLE_METHODS = ("iid_normal", "iid", "hac", "exact_normal", "bootstrap")
_DIFF_METHODS = ("bootstrap", "hac", "iid", "jkm")
_HAC_CHOICES = ("qs_pw", "qs", "parzen_pw", "parzen", "bartlett")
_STATISTICS = ("sharpe", "log_variance")
_MULTIPLE = ("romano_wolf", "holm", "bh", "by", "none")

#: Ledoit & Wolf's (2008) block-length grid at T = 120.
_LW_GRID = (1, 2, 4, 6, 8, 10)

#: ``exact_normal`` evaluates a scalar noncentral-t CDF per model and bound.
_MAX_EXACT_MODELS = 50

#: Fewest complete observations a group needs.
_MIN_OBS = 8

_JKM_WARNING = (
    "JKM (Jobson-Korkie with Memmel's correction) assumes i.i.d. bivariate normal "
    "returns and over-rejects otherwise: Ledoit & Wolf (2008, Table 1) measure "
    "10.7% rejections at nominal 5% under t6 returns and 9.5% under AR(1) "
    "phi=0.2 (T=120). Use method='bootstrap' for evidence."
)


# --------------------------------------------------------------------------- #
# Moments and influence functions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Moments:
    n: int
    mu: np.ndarray
    var0: np.ndarray
    sr: np.ndarray  # mu / sd(ddof=1): the reported estimate
    sr0: np.ndarray  # mu / sd(ddof=0): the plug-in used by influence functions
    skew: np.ndarray
    kurt: np.ndarray
    z: np.ndarray


def _moments(x: np.ndarray) -> _Moments:
    n = x.shape[0]
    mu = _hac.colmean(x)
    c = x - mu
    var0 = _hac.coldot(c, c) / n
    # Relative test: a constant series leaves a rounding-level variance, not 0.
    ok = var0 > 1e-24 * (var0 + mu * mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        var0 = np.where(ok, var0, 0.0)
        sd0 = np.where(ok, np.sqrt(var0), np.nan)
        z = c / sd0
        sr0 = mu / sd0
        sr = mu / np.sqrt(var0 * n / (n - 1)) if n > 1 else np.full_like(mu, np.nan)
        sr = np.where(ok, sr, np.nan)
        z2 = z * z
        skew = _hac.coldot(z2, z) / n
        kurt = _hac.coldot(z2, z2) / n
    return _Moments(n, mu, var0, sr, sr0, skew, kurt, z)


def _influence(m: _Moments, statistic: str) -> np.ndarray:
    if statistic == "sharpe":
        return m.z - 0.5 * m.sr0 * (m.z * m.z - 1.0)
    return m.z * m.z - 1.0


def _estimate(m: _Moments, statistic: str) -> np.ndarray:
    if statistic == "sharpe":
        return m.sr
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where(m.var0 > 0, m.var0, np.nan))


def _hac_variance(
    psi: np.ndarray, hac: str, hac_lags: int | None, horizon: int
) -> tuple[np.ndarray, str]:
    """Long-run variance of influence columns, with the small-sample factor.

    Andrews-type kernels get ``T / (T - 2)`` (two estimated moments; the
    scalar analogue of LW's ``T / (T - 4)``); fixed-lag Bartlett gets none, so it
    matches Lo (2002) and :func:`~panelary.validation.newey_west_variance`.
    """
    n = psi.shape[0]
    if hac == "bartlett":
        lags = (
            max(_hac.newey_west_lags(n), horizon - 1)
            if hac_lags is None
            else int(hac_lags)
        )
        res = _hac.column_lrv(psi, "bartlett", lags=lags)
        return np.asarray(res.lrv, dtype=np.float64), res.label
    res = _hac.column_lrv(psi, hac)
    return np.asarray(res.lrv, dtype=np.float64) * (n / (n - 2.0)), res.label


def _vector_hac_variance(x: np.ndarray, y: np.ndarray, statistic: str) -> np.ndarray:
    """LW (2008) eq. (5): ``grad' Psi grad`` with the prewhitened-QS 4-vector ``Psi``."""
    out = np.empty(x.shape[1])
    ry = y
    for j in range(x.shape[1]):
        rx = x[:, j]
        a, b = rx.mean(), ry.mean()
        c, d = np.mean(rx * rx), np.mean(ry * ry)
        vec = np.column_stack([rx - a, ry - b, rx * rx - c, ry * ry - d])
        psi, _ = _hac.vector_lrv_prewhitened(vec, kernel="qs", small_sample=True)
        va, vb = c - a * a, d - b * b
        if statistic == "sharpe":
            grad = np.array(
                [c / va**1.5, -d / vb**1.5, -0.5 * a / va**1.5, 0.5 * b / vb**1.5]
            )
        else:
            grad = np.array([-2.0 * a / va, 2.0 * b / vb, 1.0 / va, -1.0 / vb])
        out[j] = float(grad @ psi @ grad)
    return out


# --------------------------------------------------------------------------- #
# The studentized circular-block bootstrap engine (A2)
# --------------------------------------------------------------------------- #
def _components(
    x: np.ndarray, block: int, segments: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centred coordinates ``c = r - mu``, ``u = c^2 - mean(c^2)`` and their block sums."""
    mu = _hac.colmean(x)
    c = x - mu
    v0 = _hac.colmean(c * c)
    u = c * c - v0
    return (
        mu,
        v0,
        circular_block_sums(c, block, segments),
        circular_block_sums(u, block, segments),
    )


def _stat_and_gradient(
    statistic: str, mu_star: np.ndarray, m1: np.ndarray, v: np.ndarray, n_star: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate estimate and its gradient in ``(c, u)`` coordinates.

    Sharpe: ``h_c = (v + mu* m1) / v^1.5``, ``h_u = -mu* / (2 v^1.5)`` -- LW's
    ``(mu, gamma)`` gradient mapped through ``r = c + mu_hat``. Log-variance:
    ``h_c = -2 m1 / v``, ``h_u = 1 / v``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        vv = np.where(v > 0, v, np.nan)
        if statistic == "sharpe":
            est = mu_star / np.sqrt(vv * n_star / (n_star - 1.0))
            v15 = vv * np.sqrt(vv)
            return est, (vv + mu_star * m1) / v15, -0.5 * mu_star / v15
        return np.log(vv), -2.0 * m1 / vv, 1.0 / vv


def _cbb_engine(
    x: np.ndarray,
    y: np.ndarray | None,
    draw: CircularBlockDraw,
    statistic: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap estimates ``Δ*`` and Goetze-Kuensch standard errors ``s(Δ*)``.

    ``x`` is ``(T, M)`` strategies; ``y`` is ``None`` (single Sharpe), a shared
    ``(T,)`` benchmark, or ``(T, M)`` per-column benchmarks (pairs). Returns two
    ``(B, M)`` arrays. With ``z = (c, u)`` the centred components,
    ``Psi* = S2 / n* - b zbar* zbar*'`` where ``S2 = sum_j Q_j Q_j'`` is one
    product ``N @ (Q_a * Q_c)`` per entry, and ``s(Δ*)^2 = w' Psi* w / n*``.
    """
    n, m = x.shape
    b = draw.block_length
    ns = draw.n_sample
    segs = draw.segments
    n_boot = draw.n_boot
    est = np.empty((n_boot, m))
    se = np.empty((n_boot, m))
    paired = y is not None and y.ndim == 2
    if y is not None and not paired:
        muy, v0y, qcy, quy = _components(y, b, segs)
    n_arrays = 10 if y is not None else 4
    for cs in row_chunks(m, n * 8 * n_arrays, max(chunk_bytes // 2, 1)):
        mux, v0x, qcx, qux = _components(x[:, cs], b, segs)
        if paired:
            assert y is not None
            muy, v0y, qcy, quy = _components(y[:, cs], b, segs)
        mc = qcx.shape[1]
        for rs in row_chunks(n_boot, n * 8 + mc * 8 * 24, max(chunk_bytes // 2, 1)):
            nmat = start_counts(draw, rs)
            m1x = matmul_cols(nmat, qcx) / ns
            m2x = matmul_cols(nmat, qux) / ns
            pcc = matmul_cols(nmat, qcx * qcx) / ns - b * m1x * m1x
            pcu = matmul_cols(nmat, qcx * qux) / ns - b * m1x * m2x
            puu = matmul_cols(nmat, qux * qux) / ns - b * m2x * m2x
            vx = v0x + m2x - m1x * m1x
            ex, hcx, hux = _stat_and_gradient(statistic, mux + m1x, m1x, vx, ns)
            var = hcx * hcx * pcc + 2.0 * hcx * hux * pcu + hux * hux * puu
            if y is not None:
                if paired:
                    m1y = matmul_cols(nmat, qcy) / ns
                    m2y = matmul_cols(nmat, quy) / ns
                    ycc = matmul_cols(nmat, qcy * qcy) / ns
                    ycu = matmul_cols(nmat, qcy * quy) / ns
                    yuu = matmul_cols(nmat, quy * quy) / ns
                    qcy2, quy2 = qcy, quy
                else:
                    m1y = (nmat @ qcy / ns)[:, None]
                    m2y = (nmat @ quy / ns)[:, None]
                    ycc = (nmat @ (qcy * qcy) / ns)[:, None]
                    ycu = (nmat @ (qcy * quy) / ns)[:, None]
                    yuu = (nmat @ (quy * quy) / ns)[:, None]
                    qcy2, quy2 = qcy[:, None], quy[:, None]
                ycc = ycc - b * m1y * m1y
                ycu = ycu - b * m1y * m2y
                yuu = yuu - b * m2y * m2y
                xcc = matmul_cols(nmat, qcx * qcy2) / ns - b * m1x * m1y
                xcu = matmul_cols(nmat, qcx * quy2) / ns - b * m1x * m2y
                xuc = matmul_cols(nmat, qux * qcy2) / ns - b * m2x * m1y
                xuu = matmul_cols(nmat, qux * quy2) / ns - b * m2x * m2y
                vy = v0y + m2y - m1y * m1y
                ey, hcy, huy = _stat_and_gradient(statistic, muy + m1y, m1y, vy, ns)
                var = (
                    var
                    + hcy * hcy * ycc
                    + 2.0 * hcy * huy * ycu
                    + huy * huy * yuu
                    - 2.0
                    * (
                        hcx * hcy * xcc
                        + hcx * huy * xcu
                        + hux * hcy * xuc
                        + hux * huy * xuu
                    )
                )
                ex = ex - ey
            est[rs, cs] = ex
            with np.errstate(invalid="ignore"):
                se[rs, cs] = np.sqrt(np.maximum(var, 0.0) / ns)
    return est, se


# --------------------------------------------------------------------------- #
# Block length: Politis-White and LW Algorithm 3.1
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockLengthCalibration:
    """Outcome of Ledoit & Wolf's (2008) Algorithm 3.1 block-length calibration.

    Attributes
    ----------
    grid : numpy.ndarray
        ``(G,)`` candidate block lengths ``b``.
    coverage : numpy.ndarray
        ``(G,)`` estimated coverage ``g_hat(b)`` of the nominal ``level``
        bootstrap interval on pseudo-data with known parameter.
    block_length : int
        The ``b`` whose ``g_hat(b)`` is closest to ``level`` (smallest on ties).
    level : float
        Nominal coverage ``1 - alpha``.
    estimate : float
        The pseudo-true parameter (the statistic on the observed data).
    statistic : str
        ``"sharpe"`` or ``"log_variance"``.
    n_sims, n_boot : int
        Pseudo-sequences ``K`` and bootstrap replicates per interval.
    seed : int or None
        Seed.
    warnings : tuple of str
        Caveats (e.g. a near-unit-root VAR was shrunk).
    """

    grid: np.ndarray
    coverage: np.ndarray
    block_length: int
    level: float
    estimate: float
    statistic: str
    n_sims: int
    n_boot: int
    seed: int | None
    warnings: tuple[str, ...] = ()

    def to_frame(self) -> pl.DataFrame:
        """One row per candidate ``b``: ``block_length, coverage, selected``."""
        return pl.DataFrame(
            {
                "block_length": self.grid.astype(np.int64),
                "coverage": self.coverage.astype(np.float64),
                "selected": self.grid == self.block_length,
            }
        )


def _default_grid(n: int) -> tuple[int, ...]:
    """LW's grid at T = 120, scaled by ``(T / 120)^(1/3)``, unique, ``<= T / 2``."""
    scale = (n / 120.0) ** (1.0 / 3.0)
    cap = max(1, n // 2)
    return tuple(sorted({min(cap, max(1, int(round(v * scale)))) for v in _LW_GRID}))


def _pw_block_length(psi: np.ndarray) -> int:
    """Politis-White (PPW-corrected) block length, maximised over columns."""
    from panelary.depend._null import auto_block_length  # noqa: PLC0415  (cycle)

    return max(
        int(auto_block_length(psi[:, j], kind="circular")) for j in range(psi.shape[1])
    )


def _var1_fit(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """VAR(1) with intercept by least squares; residuals centred."""
    n, k = z.shape
    design = np.column_stack([np.ones(n - 1), z[:-1]])
    coef, *_ = np.linalg.lstsq(design, z[1:], rcond=None)
    nu = coef[0]
    a_mat = coef[1:].T
    resid = z[1:] - design @ coef
    resid = resid - resid.mean(axis=0)
    notes: list[str] = []
    radius = float(np.max(np.abs(np.linalg.eigvals(a_mat)))) if k else 0.0
    if radius >= 0.99:
        a_mat = a_mat * (0.97 / radius)
        notes.append(
            f"the fitted VAR(1) had spectral radius {radius:.3f}; shrunk to 0.97 "
            "to generate stationary pseudo-sequences."
        )
    return nu, a_mat, resid, notes


def _pseudo_sequences(
    z: np.ndarray,
    n_sims: int,
    mean_block: float,
    rng: np.random.Generator,
    burn: int = 100,
) -> tuple[np.ndarray, list[str]]:
    """``(T, K, k)`` VAR(1) pseudo-sequences with stationary-bootstrapped residuals."""
    n, k = z.shape
    nu, a_mat, resid, notes = _var1_fit(z)
    idx = stationary_indices(
        resid.shape[0], mean_block, n_sims, length=n + burn, seed=rng
    )
    eps = resid[idx]  # (K, n + burn, k)
    state = z[rng.integers(0, n, size=n_sims)]
    out = np.empty((n, n_sims, k))
    at = a_mat.T
    for s in range(n + burn):
        state = nu + state @ at + eps[:, s, :]
        if s >= burn:
            out[s - burn] = state
    return out, notes


def _calibrate(
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    statistic: str,
    grid: Sequence[int],
    n_sims: int,
    level: float,
    mean_block: float,
    n_boot: int,
    hac: str,
    hac_lags: int | None,
    rng: np.random.Generator,
    chunk_bytes: int,
) -> tuple[int, np.ndarray, np.ndarray, float, list[str]]:
    """LW (2008) Algorithm 3.1: pick ``b`` whose bootstrap CI coverage is closest to ``level``.

    The ``K`` pseudo-sequences enter the engine as ``K`` extra columns, and one
    ``N`` per ``b`` is shared across them (unbiased for ``g(b)``; it only adds
    some Monte Carlo variance).
    """
    n = x.shape[0]
    data = x[:, None] if y is None else np.column_stack([x, y])
    mom = _moments(data)
    est_all = _estimate(mom, statistic)
    theta = float(est_all[0] - est_all[1]) if y is not None else float(est_all[0])
    pseudo, notes = _pseudo_sequences(data, n_sims, mean_block, rng)
    px = pseudo[:, :, 0]
    py = pseudo[:, :, 1] if y is not None else None
    mx = _moments(px)
    est_k = _estimate(mx, statistic)
    psi = _influence(mx, statistic)
    if py is not None:
        my = _moments(py)
        est_k = est_k - _estimate(my, statistic)
        psi = psi - _influence(my, statistic)
    var_k, _ = _hac_variance(psi, hac, hac_lags, 1)
    s_k = np.sqrt(var_k / n)
    grid_arr = np.asarray(
        sorted({int(g) for g in grid if 1 <= int(g) <= n}), dtype=np.int64
    )
    if grid_arr.size == 0:
        raise ValueError(f"no admissible block length in `grid` for T={n}.")
    coverage = np.empty(grid_arr.size)
    valid = np.isfinite(est_k) & np.isfinite(s_k)
    for gi, b in enumerate(grid_arr):
        draw = circular_block_draw(n, int(b), n_boot, seed=rng)
        e_star, s_star = _cbb_engine(px, py, draw, statistic, chunk_bytes)
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = np.abs(e_star - est_k) / s_star
        dev[~np.isfinite(dev)] = np.inf
        q = np.quantile(dev, level, axis=0)
        covered = np.abs(est_k - theta) <= q * s_k
        coverage[gi] = float(np.mean(covered[valid])) if valid.any() else np.nan
    best = int(np.nanargmin(np.abs(coverage - level)))
    return int(grid_arr[best]), grid_arr, coverage, theta, notes


def sharpe_block_length(
    returns: Any,
    benchmark: Any = None,
    *,
    statistic: str = "sharpe",
    grid: Sequence[int] | None = None,
    n_sims: int = 1000,
    level: float = 0.95,
    var_bootstrap_block: float = 5.0,
    n_boot: int = 499,
    hac: str = "qs_pw",
    seed: int | None = 0,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> BlockLengthCalibration:
    """Calibrate the circular-block length by Ledoit & Wolf's (2008) Algorithm 3.1.

    1. Fit a VAR(1) (with intercept) to the returns (``(r_i, r_b)`` for a
       difference, ``r`` alone for a single Sharpe).
    2. Generate ``n_sims`` pseudo-sequences of length ``T`` from it, resampling
       its residuals by a stationary bootstrap with mean block
       ``var_bootstrap_block`` (LW use 5), after a 100-step burn-in.
    3. For each ``b`` in ``grid``, build the studentized bootstrap interval at
       ``level`` on every pseudo-sequence and record whether it covers the
       pseudo-true value (the statistic on the observed data).
    4. Return the ``b`` whose coverage ``g_hat(b)`` is closest to ``level``.

    Parameters
    ----------
    returns : array-like of shape (T,)
        Strategy returns (complete; no NaN).
    benchmark : array-like of shape (T,), optional
        Benchmark returns for a difference; ``None`` calibrates a single Sharpe.
    statistic : {"sharpe", "log_variance"}, default "sharpe"
        ``"log_variance"`` requires ``benchmark``.
    grid : sequence of int, optional
        Candidate block lengths. Default: LW's ``(1, 2, 4, 6, 8, 10)`` at
        ``T = 120``, scaled by ``(T/120)^(1/3)`` and rounded to unique integers
        (so exactly LW's grid at ``T = 120``).
    n_sims : int, default 1000
        Pseudo-sequences ``K`` (LW: 1000 is the lower limit, 5000 suffices).
    level : float, default 0.95
        Nominal interval coverage.
    var_bootstrap_block : float, default 5.0
        Mean block of the residual stationary bootstrap.
    n_boot : int, default 499
        Bootstrap replicates per interval (LW's simulations use 499).
    hac : str, default "qs_pw"
        LRV estimator for the interval's standard error.
    seed : int, optional
        Seed.
    chunk_bytes : int
        Working-set budget per chunk.

    Returns
    -------
    BlockLengthCalibration
    """
    if statistic not in _STATISTICS:
        raise ValueError(f"unknown `statistic` {statistic!r}; expected {_STATISTICS}.")
    x = as_vector(returns, "returns")
    y = None if benchmark is None else as_vector(benchmark, "benchmark", x.shape[0])
    if statistic == "log_variance" and y is None:
        raise ValueError("statistic='log_variance' needs a `benchmark`.")
    ok = np.isfinite(x) if y is None else np.isfinite(x) & np.isfinite(y)
    if not ok.all():
        raise ValueError(
            "`returns`/`benchmark` must be complete (no NaN) for calibration."
        )
    if not (0.0 < level < 1.0):
        raise ValueError(f"`level` must be in (0, 1), got {level}.")
    if x.shape[0] < _MIN_OBS:
        raise ValueError(f"need at least {_MIN_OBS} observations, got {x.shape[0]}.")
    rng = np.random.default_rng(seed)
    b, grid_arr, cov, theta, notes = _calibrate(
        x,
        y,
        statistic=statistic,
        grid=_default_grid(x.shape[0]) if grid is None else grid,
        n_sims=int(n_sims),
        level=float(level),
        mean_block=float(var_bootstrap_block),
        n_boot=int(n_boot),
        hac=hac,
        hac_lags=None,
        rng=rng,
        chunk_bytes=chunk_bytes,
    )
    return BlockLengthCalibration(
        grid_arr,
        cov,
        b,
        float(level),
        theta,
        statistic,
        int(n_sims),
        int(n_boot),
        seed,
        tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Noncentral t (exact normal inference)
# --------------------------------------------------------------------------- #
def _nct_cdf_nonneg(t: float, df: float, delta: float) -> float:
    """``P(T <= t)`` for ``t >= 0``: Poisson mixture of incomplete betas.

    Benton & Krishnamoorthy (2003): start at the Poisson mode and recurse the
    incomplete betas both ways (forward subtracting, backward adding), which
    stays accurate for large noncentrality where AS 243 (starting at 0)
    underflows.
    """
    base = float(_special.norm_cdf(-delta))
    if t == 0.0:
        return base
    x = t * t / (t * t + df)
    lam = 0.5 * delta * delta
    half_df = 0.5 * df
    log_x, log_1mx = math.log(x), math.log1p(-x)
    lg = math.lgamma
    k = int(lam)
    if lam > 0:
        log_lam = math.log(lam)
        p_k = math.exp(-lam + k * log_lam - lg(k + 1.0))
        q_k = delta * math.exp(-lam + k * log_lam - lg(k + 1.5)) / math.sqrt(2.0)
    else:
        p_k, q_k = 1.0, 0.0
    # Incomplete betas at a = k + 1/2 and a = k + 1, and the recursion terms
    # g(a) = Gamma(a + b) / (Gamma(a + 1) Gamma(b)) x^a (1 - x)^b.
    ia = float(_special.betainc(k + 0.5, half_df, x))
    ib = float(_special.betainc(k + 1.0, half_df, x))

    def g_term(a: float) -> float:
        return math.exp(
            lg(a + half_df) - lg(a + 1.0) - lg(half_df) + a * log_x + half_df * log_1mx
        )

    total = p_k * ia + q_k * ib
    # forward
    p, q, ja, jb = p_k, q_k, ia, ib
    ga, gb = g_term(k + 0.5), g_term(k + 1.0)
    i = k
    for _ in range(100000):
        ja -= ga
        jb -= gb
        ga *= x * (k + 0.5 + (i - k) + half_df) / (k + 1.5 + (i - k))
        gb *= x * (k + 1.0 + (i - k) + half_df) / (k + 2.0 + (i - k))
        i += 1
        p *= lam / i
        q *= lam / (i + 0.5)
        term = p * ja + q * jb
        total += term
        if i > lam and abs(p) + abs(q) < 1e-17:
            break
    # backward
    p, q, ja, jb = p_k, q_k, ia, ib
    i = k
    while i > 0:
        # I_x(a - 1, b) = I_x(a, b) + g(a - 1)
        ja += g_term(i - 0.5)
        jb += g_term(float(i))
        p *= i / lam
        q *= (i + 0.5) / lam
        i -= 1
        total += p * ja + q * jb
        if abs(p) + abs(q) < 1e-17:
            break
    return min(1.0, max(0.0, base + 0.5 * total))


def _nct_cdf(t: float, df: float, delta: float) -> float:
    if not (math.isfinite(t) and math.isfinite(delta)):
        return float("nan")
    if t >= 0.0:
        return _nct_cdf_nonneg(t, df, delta)
    return 1.0 - _nct_cdf_nonneg(-t, df, -delta)


def _nct_delta_for(t_obs: float, df: float, target: float) -> float:
    """Solve ``F(t_obs; df, delta) = target`` for ``delta`` (F decreases in delta)."""

    def f(d: float) -> float:
        return _nct_cdf(t_obs, df, d) - target

    lo, hi = t_obs - 1.0, t_obs + 1.0
    step = 1.0
    while f(lo) < 0.0 and step < 1e4:
        lo -= step
        step *= 2.0
    step = 1.0
    while f(hi) > 0.0 and step < 1e4:
        hi += step
        step *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Helpers shared by the two public tests
# --------------------------------------------------------------------------- #
def _check_common(
    alternative: str, n_boot: int, confidence: float, statistic: str | None = None
) -> None:
    check_alternative(alternative)
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1, got {n_boot}.")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"`confidence` must be in (0, 1), got {confidence}.")
    if statistic is not None and statistic not in _STATISTICS:
        raise ValueError(f"unknown `statistic` {statistic!r}; expected {_STATISTICS}.")


def _local_boundaries(rows: np.ndarray, boundaries: Any) -> list[int] | None:
    if boundaries is None:
        return None
    cuts = np.asarray(boundaries, dtype=np.int64).ravel()
    local = np.searchsorted(rows, cuts, side="left")
    return sorted({int(v) for v in local if 0 < v < rows.size})


def _resolve_block(
    block_length: int | str,
    psi: np.ndarray,
    n: int,
    local_bounds: list[int] | None,
    notes: list[str],
) -> int:
    if isinstance(block_length, str):
        if block_length != "auto":
            raise ValueError(f"unknown `block_length` {block_length!r}.")
        b = _pw_block_length(psi)
    else:
        b = int(block_length)
        if b < 1:
            raise ValueError(f"`block_length` must be >= 1, got {block_length}.")
    edges = [0, *(local_bounds or []), n]
    shortest = min(e2 - e1 for e1, e2 in zip(edges[:-1], edges[1:], strict=True))
    if b > shortest:
        notes.append(
            f"block length {b} exceeds the shortest fold segment ({shortest}); clipped."
        )
        b = shortest
    return b


def _abs_quantile(z: np.ndarray, level: float) -> np.ndarray:
    a = np.abs(z)
    a[~np.isfinite(a)] = np.inf
    return np.quantile(a, level, axis=0)


# --------------------------------------------------------------------------- #
# Public: single Sharpe ratio
# --------------------------------------------------------------------------- #
def sharpe_ratio_inference(
    returns: Any,
    *,
    method: str = "hac",
    benchmark_sharpe: float = 0.0,
    alternative: str = "greater",
    confidence: float = 0.95,
    horizon: int = 1,
    hac: str = "qs_pw",
    hac_lags: int | None = None,
    small_sample: str = "asymptotic",
    block_length: int | str = "auto",
    n_boot: int = 4999,
    boundaries: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    seed: int | None = 0,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> EvaluationResult:
    """Standard error, confidence interval and test for per-period Sharpe ratios.

    Tests ``H0: SR = benchmark_sharpe`` for each of ``M`` return series.

    Parameters
    ----------
    returns : array-like of shape (T,) or (T, M)
        Per-period (excess) returns; ``nan`` marks missing values. A
        :class:`polars.DataFrame` supplies model names from its columns.
    method : {"hac", "iid", "iid_normal", "exact_normal", "bootstrap"}, default "hac"
        See the module docstring. ``"exact_normal"`` inverts the noncentral t
        (exact under i.i.d. normal returns only; at most 50 models).
    benchmark_sharpe : float, default 0.0
        Per-period Sharpe under the null.
    alternative : {"greater", "less", "two-sided"}, default "greater"
    confidence : float, default 0.95
        Level of the two-sided confidence interval in ``ci``.
    horizon : int, default 1
        Overlap of the return periods; the fixed-lag Bartlett HAC uses at
        least ``horizon - 1`` lags.
    hac : {"qs_pw", "qs", "parzen_pw", "parzen", "bartlett"}, default "qs_pw"
        LRV estimator of the influence series (``method="hac"`` and the
        bootstrap's studentization). ``"bartlett"`` with ``hac_lags`` is Lo's
        (2002) non-i.i.d. GMM standard error.
    hac_lags : int, optional
        Bartlett truncation lag (default Newey-West's rule).
    small_sample : {"asymptotic", "t-1"}, default "asymptotic"
        Divide the variance by ``T`` or ``T - 1`` (``"t-1"`` reproduces the
        PSR/DSR standard error, ``_sharpe_estimator_std``).
    block_length : int or {"auto", "calibrate"}, default "auto"
        Circular block length for ``method="bootstrap"``: Politis-White
        (``"auto"``) or LW Algorithm 3.1 on an AR(1) (``"calibrate"``).
    n_boot : int, default 4999
        Bootstrap replicates.
    boundaries : sequence of int, optional
        Fold boundaries the bootstrap blocks must not straddle.
    names : sequence of str, optional
        Model labels.
    seed : int, optional
        Seed (one generator per call; the same ``N`` for every model in a group).
    chunk_bytes : int
        Working-set budget for the bootstrap.

    Returns
    -------
    EvaluationResult
        ``test="sharpe_ratio"``; ``estimate`` is the per-period Sharpe (ddof=1),
        ``statistic`` is ``(SR - SR0) / se`` (``sqrt(T) SR`` for
        ``exact_normal``); ``details`` carries ``skewness``, ``kurtosis`` and
        per-model ``block_length``.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.random.default_rng(0).standard_normal(500) * 0.01 + 0.001
    >>> res = sharpe_ratio_inference(r, method="iid")
    >>> res.to_frame()["reference"][0]
    'N(0,1)'
    """
    if method not in _SINGLE_METHODS:
        raise ValueError(
            f"unknown `method` {method!r}; expected one of {_SINGLE_METHODS}."
        )
    if hac not in _HAC_CHOICES:
        raise ValueError(f"unknown `hac` {hac!r}; expected one of {_HAC_CHOICES}.")
    if small_sample not in ("asymptotic", "t-1"):
        raise ValueError("`small_sample` must be 'asymptotic' or 't-1'.")
    if horizon < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    _check_common(alternative, n_boot, confidence)
    mat, colnames = as_matrix(returns, "returns")
    n_total, m_total = mat.shape
    labels = resolve_names(names, m_total, colnames)
    if method == "exact_normal" and m_total > _MAX_EXACT_MODELS:
        raise ValueError(
            f"method='exact_normal' is limited to {_MAX_EXACT_MODELS} models "
            f"(scalar noncentral-t inversions); got {m_total}."
        )
    sr0 = float(benchmark_sharpe)
    est = np.full(m_total, np.nan)
    stat = np.full(m_total, np.nan)
    pval = np.full(m_total, np.nan)
    se_out = np.full(m_total, np.nan)
    ci = np.full((m_total, 2), np.nan)
    nobs = np.zeros(m_total, dtype=np.int64)
    skew = np.full(m_total, np.nan)
    kurt = np.full(m_total, np.nan)
    blocks = np.full(m_total, -1, dtype=np.int64)
    notes: list[str] = []
    hac_label: str | None = None
    main_block: int | None = None
    zq = float(_special.norm_ppf(0.5 + 0.5 * confidence))
    reference = {
        "iid_normal": "N(0,1)",
        "iid": "N(0,1)",
        "hac": "N(0,1)",
        "exact_normal": "nct(T-1) [assumes i.i.d. normal returns]",
    }.get(method, "")
    for gi, (rows, cols) in enumerate(_hac.mask_groups(np.isfinite(mat))):
        n = rows.size
        nobs[cols] = n
        if n < _MIN_OBS:
            notes.append(
                f"{len(cols)} model(s) have fewer than {_MIN_OBS} observations; nan."
            )
            continue
        x = mat[rows][:, cols]
        mom = _moments(x)
        if not np.all(mom.var0 > 0):
            notes.append("zero-variance return series give nan.")
        est[cols] = mom.sr
        skew[cols] = mom.skew
        kurt[cols] = mom.kurt
        denom = n - 1.0 if small_sample == "t-1" else float(n)
        if method == "exact_normal":
            df = n - 1.0
            for j, col in enumerate(cols):
                if not np.isfinite(mom.sr[j]):
                    continue
                t_obs = math.sqrt(n) * float(mom.sr[j])
                cdf = _nct_cdf(t_obs, df, math.sqrt(n) * sr0)
                stat[col] = t_obs
                if alternative == "greater":
                    pval[col] = 1.0 - cdf
                elif alternative == "less":
                    pval[col] = cdf
                else:
                    pval[col] = min(1.0, 2.0 * min(cdf, 1.0 - cdf))
                tail = 0.5 * (1.0 - confidence)
                ci[col, 0] = _nct_delta_for(t_obs, df, 1.0 - tail) / math.sqrt(n)
                ci[col, 1] = _nct_delta_for(t_obs, df, tail) / math.sqrt(n)
                se_out[col] = math.sqrt((1.0 + 0.5 * float(mom.sr[j]) ** 2) / denom)
            continue
        psi = _influence(mom, "sharpe")
        if method == "iid_normal":
            var = 1.0 + 0.5 * mom.sr**2
        elif method == "iid":
            var = 1.0 - mom.skew * mom.sr + 0.25 * (mom.kurt - 1.0) * mom.sr**2
        else:
            var, label = _hac_variance(psi, hac, hac_lags, horizon)
            if gi == 0:
                hac_label = label
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.where(var > 0, var, np.nan) / denom)
        z = (mom.sr - sr0) / se
        se_out[cols] = se
        stat[cols] = z
        if method != "bootstrap":
            pval[cols] = normal_pvalue(z, alternative)
            ci[cols, 0] = mom.sr - zq * se
            ci[cols, 1] = mom.sr + zq * se
            continue
        local = _local_boundaries(rows, boundaries)
        if block_length == "calibrate":
            pick = int(np.argsort(var)[len(cols) // 2]) if len(cols) > 1 else 0
            b, _, _, _, cal_notes = _calibrate(
                x[:, pick],
                None,
                statistic="sharpe",
                grid=_default_grid(n),
                n_sims=1000,
                level=confidence,
                mean_block=5.0,
                n_boot=min(n_boot, 499),
                hac=hac,
                hac_lags=hac_lags,
                rng=np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,))),
                chunk_bytes=chunk_bytes,
            )
            notes.extend(cal_notes)
            b = _resolve_block(b, psi, n, local, notes)
        else:
            b = _resolve_block(block_length, psi, n, local, notes)
        blocks[cols] = b
        if gi == 0:
            main_block = b
        draw = circular_block_draw(
            n, b, n_boot, boundaries=local, seed=np.random.default_rng(seed)
        )
        e_star, s_star = _cbb_engine(x, None, draw, "sharpe", chunk_bytes)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_star = (e_star - mom.sr) / s_star
        pval[cols] = resample_pvalue(z_star, z, alternative)
        q = _abs_quantile(z_star, confidence)
        ci[cols, 0] = mom.sr - q * se
        ci[cols, 1] = mom.sr + q * se
        reference = f"cbb-studentized(b={main_block},B={n_boot})"
    if method == "bootstrap" and main_block is None:
        reference = f"cbb-studentized(B={n_boot})"
    return EvaluationResult(
        test="sharpe_ratio",
        names=labels,
        estimate=est,
        statistic=stat,
        pvalue=pval,
        reference=reference,
        alternative=alternative,
        n_obs=nobs,
        std_error=se_out,
        ci=ci,
        horizon=int(horizon),
        hac=hac_label,
        block_length=main_block,
        n_resamples=int(n_boot) if method == "bootstrap" else None,
        seed=seed if method == "bootstrap" else None,
        details={
            "benchmark_sharpe": np.full(m_total, sr0),
            "skewness": skew,
            "kurtosis": kurt,
            "block_length": blocks,
        },
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Public: Sharpe (or log-variance) difference against a benchmark
# --------------------------------------------------------------------------- #
def sharpe_ratio_test(
    returns: Any,
    benchmark: Any,
    *,
    statistic: str = "sharpe",
    method: str = "bootstrap",
    alternative: str = "two-sided",
    hac: str = "qs_pw",
    hac_lags: int | None = None,
    block_length: int | str = "auto",
    n_boot: int = 4999,
    calibration_sims: int = 1000,
    multiple: str | None = "romano_wolf",
    alpha: float = 0.05,
    boundaries: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    seed: int | None = 0,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> EvaluationResult:
    """Test ``H0: SR_i = SR_b`` (or ``sigma_i = sigma_b``) for ``M`` strategies vs one benchmark.

    Parameters
    ----------
    returns : array-like of shape (T,) or (T, M)
        Strategy returns (``nan`` = missing).
    benchmark : array-like of shape (T,)
        Benchmark returns on the same dates.
    statistic : {"sharpe", "log_variance"}, default "sharpe"
        ``"log_variance"`` tests equal volatility through ``log(s_i^2 / s_b^2)``
        (Ledoit & Wolf 2011); not the Lo-MacKinlay variance-ratio test.
    method : {"bootstrap", "hac", "iid", "jkm"}, default "bootstrap"
        ``"bootstrap"`` is LW's studentized circular-block bootstrap
        ("Boot-TS"); ``"hac"`` the HAC delta method; ``"iid"`` the i.i.d.
        non-normal (Opdyke) delta method; ``"jkm"`` Jobson-Korkie/Memmel,
        kept only as the labelled naive baseline (it always carries a warning).
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        ``"greater"`` is "the strategy beats the benchmark".
    hac : {"qs_pw", "qs", "parzen_pw", "parzen", "bartlett", "qs_pw_vector"}
        LRV estimator for ``s(Δ̂)``. The default is the scalar influence path;
        ``"qs_pw_vector"`` reproduces LW's 4-vector prewhitened-QS ``Psi`` with
        ``T/(T-4)`` for every column (an ``O(M)`` loop of 4x4 work).
    hac_lags : int, optional
        Bartlett truncation lag.
    block_length : int or {"auto", "calibrate"}, default "auto"
        ``"auto"``: Politis-White on each column's difference-influence series,
        maximised over columns (shared indices need one ``b``).
        ``"calibrate"``: LW Algorithm 3.1 (on the median-LRV column when
        ``M > 1``) with ``calibration_sims`` pseudo-sequences.
    n_boot : int, default 4999
        Bootstrap replicates ``B``.
    calibration_sims : int, default 1000
        Pseudo-sequences for ``block_length="calibrate"``.
    multiple : {"romano_wolf", "holm", "bh", "by", "none"} or None
        Multiplicity adjustment when ``M > 1``. Romano-Wolf uses the shared
        bootstrap null (studentized StepM) and so needs ``method="bootstrap"``;
        other methods fall back to Holm with a warning.
    alpha : float, default 0.05
        FWER/FDR level; the confidence interval is at ``1 - alpha``.
    boundaries : sequence of int, optional
        Fold boundaries the blocks must not straddle.
    names : sequence of str, optional
        Model labels.
    seed : int, optional
        Seed. All models in a group share one ``N``, drawn before any chunking,
        so adding or removing a model never changes another's marginal result
        (for a fixed integer ``block_length``).
    chunk_bytes : int
        Working-set budget.

    Returns
    -------
    EvaluationResult
        ``test="sharpe_difference"`` (or ``"log_variance_difference"``);
        ``estimate`` is ``SR_i - SR_b`` (or ``log(s_i^2 / s_b^2)``),
        ``statistic`` is ``estimate / s(estimate)``; ``details`` carries the
        per-model ``sharpe`` and ``benchmark_sharpe``, availability ``group``,
        and (when calibrated) ``calibration_grid`` / ``calibration_coverage``.

    References
    ----------
    Ledoit & Wolf (2008), eq. (9) and Algorithm 3.1; Romano & Wolf (2005).
    """
    if method not in _DIFF_METHODS:
        raise ValueError(
            f"unknown `method` {method!r}; expected one of {_DIFF_METHODS}."
        )
    if hac not in (*_HAC_CHOICES, "qs_pw_vector"):
        raise ValueError(f"unknown `hac` {hac!r}.")
    if multiple is None:
        multiple = "none"
    if multiple not in _MULTIPLE:
        raise ValueError(
            f"unknown `multiple` {multiple!r}; expected one of {_MULTIPLE}."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    _check_common(alternative, n_boot, 1.0 - alpha, statistic)
    mat, colnames = as_matrix(returns, "returns")
    n_total, m_total = mat.shape
    bench = as_vector(benchmark, "benchmark", n_total)
    labels = resolve_names(names, m_total, colnames)
    test_name = (
        "sharpe_difference" if statistic == "sharpe" else "log_variance_difference"
    )

    est = np.full(m_total, np.nan)
    stat = np.full(m_total, np.nan)
    pval = np.full(m_total, np.nan)
    se_out = np.full(m_total, np.nan)
    ci = np.full((m_total, 2), np.nan)
    nobs = np.zeros(m_total, dtype=np.int64)
    own = np.full(m_total, np.nan)
    ref_sr = np.full(m_total, np.nan)
    group_id = np.full(m_total, -1, dtype=np.int64)
    notes: list[str] = []
    if method == "jkm":
        notes.append(_JKM_WARNING)
    hac_label: str | None = None
    main_block: int | None = None
    details: dict[str, np.ndarray] = {}
    z_alpha = float(_special.norm_ppf(1.0 - 0.5 * alpha))
    rw_null: np.ndarray | None = None
    rw_cols: np.ndarray | None = None

    groups = _hac.mask_groups(np.isfinite(mat) & np.isfinite(bench)[:, None])
    if len(groups) > 1:
        notes.append(
            f"models fall into {len(groups)} availability groups; each is tested on "
            "its own common dates, and Romano-Wolf runs within the largest group only."
        )
    for gi, (rows, cols) in enumerate(groups):
        n = rows.size
        nobs[cols] = n
        group_id[cols] = gi
        if n < _MIN_OBS:
            notes.append(
                f"{len(cols)} model(s) have fewer than {_MIN_OBS} common observations; nan."
            )
            continue
        x = mat[rows][:, cols]
        yy = bench[rows]
        mx = _moments(x)
        my = _moments(yy[:, None])
        if not (my.var0[0] > 0) or not np.all(mx.var0 > 0):
            notes.append("zero-variance return series give nan.")
        own[cols] = mx.sr
        ref_sr[cols] = my.sr[0]
        delta = _estimate(mx, statistic) - _estimate(my, statistic)[0]
        est[cols] = delta
        psi = _influence(mx, statistic) - _influence(my, statistic)
        if method == "jkm":
            rho = _hac.coldot(mx.z, np.broadcast_to(my.z, mx.z.shape)) / n
            if statistic == "sharpe":
                s_i, s_b = mx.sr, my.sr[0]
                var = (
                    2.0 - 2.0 * rho + 0.5 * (s_i**2 + s_b**2 - 2.0 * s_i * s_b * rho**2)
                )
            else:
                var = 4.0 - 4.0 * rho**2
        elif method == "iid":
            var = _hac.coldot(psi, psi) / n - _hac.colmean(psi) ** 2
        elif hac == "qs_pw_vector":
            var = _vector_hac_variance(x, yy, statistic)
            if gi == 0:
                hac_label = "qs-pw-vector(T/(T-4))"
        else:
            var, label = _hac_variance(psi, hac, hac_lags, 1)
            if gi == 0:
                hac_label = label
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(np.where(var > 0, var, np.nan) / n)
            z = delta / se
        se_out[cols] = se
        stat[cols] = z
        if method != "bootstrap":
            pval[cols] = normal_pvalue(z, alternative)
            ci[cols, 0] = delta - z_alpha * se
            ci[cols, 1] = delta + z_alpha * se
            continue
        local = _local_boundaries(rows, boundaries)
        if block_length == "calibrate":
            pick = int(np.argsort(var)[len(cols) // 2]) if len(cols) > 1 else 0
            b_cal, grid_arr, cov, _, cal_notes = _calibrate(
                x[:, pick],
                yy,
                statistic=statistic,
                grid=_default_grid(n),
                n_sims=int(calibration_sims),
                level=1.0 - alpha,
                mean_block=5.0,
                n_boot=min(int(n_boot), 499),
                hac="qs_pw" if hac == "qs_pw_vector" else hac,
                hac_lags=hac_lags,
                rng=np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,))),
                chunk_bytes=chunk_bytes,
            )
            notes.extend(cal_notes)
            if gi == 0:
                details["calibration_grid"] = grid_arr[None, :].astype(np.float64)
                details["calibration_coverage"] = cov[None, :]
            b = _resolve_block(b_cal, psi, n, local, notes)
        else:
            b = _resolve_block(block_length, psi, n, local, notes)
        if gi == 0:
            main_block = b
        draw = circular_block_draw(
            n, b, n_boot, boundaries=local, seed=np.random.default_rng(seed)
        )
        e_star, s_star = _cbb_engine(x, yy, draw, statistic, chunk_bytes)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_star = (e_star - delta) / s_star
        pval[cols] = resample_pvalue(z_star, z, alternative)
        q = _abs_quantile(z_star, 1.0 - alpha)
        ci[cols, 0] = delta - q * se
        ci[cols, 1] = delta + q * se
        if gi == 0:
            rw_null, rw_cols = z_star, cols

    pvalue_adj: np.ndarray | None = None
    adjustment: str | None = None
    if m_total > 1 and multiple != "none":
        if multiple == "romano_wolf" and method != "bootstrap":
            notes.append(
                "Romano-Wolf needs the bootstrap null (method='bootstrap'); used Holm instead."
            )
            multiple = "holm"
        pvalue_adj = np.full(m_total, np.nan)
        if multiple == "romano_wolf":
            adjustment = "romano-wolf"
            if rw_null is not None and rw_cols is not None:
                sgn = -1.0 if alternative == "less" else 1.0
                keep = np.isfinite(stat[rw_cols])
                if keep.any():
                    null = sgn * rw_null[:, keep]
                    null[~np.isfinite(null)] = np.inf
                    res = romano_wolf(
                        sgn * stat[rw_cols][keep],
                        null,
                        alpha=alpha,
                        two_sided=alternative == "two-sided",
                    )
                    pvalue_adj[rw_cols[keep]] = res.adjusted_pvalues
        else:
            fin = np.isfinite(pval)
            if fin.any():
                fn = {
                    "holm": holm_bonferroni,
                    "bh": benjamini_hochberg,
                    "by": benjamini_yekutieli,
                }[multiple]
                pvalue_adj[fin] = fn(pval[fin], alpha=alpha).adjusted_pvalues
            adjustment = multiple

    if method == "bootstrap":
        reference = (
            f"cbb-studentized(b={main_block},B={n_boot})"
            if main_block is not None
            else f"cbb-studentized(B={n_boot})"
        )
    else:
        reference = "N(0,1)"
    details.update(
        {
            "sharpe": own,
            "benchmark_sharpe": ref_sr,
            "group": group_id,
        }
    )
    return EvaluationResult(
        test=test_name,
        names=labels,
        estimate=est,
        statistic=stat,
        pvalue=pval,
        reference=reference,
        alternative=alternative,
        n_obs=nobs,
        std_error=se_out,
        ci=ci,
        pvalue_adj=pvalue_adj,
        adjustment=adjustment,
        hac=hac_label if method in ("hac", "bootstrap") else None,
        block_length=main_block,
        n_resamples=int(n_boot) if method == "bootstrap" else None,
        seed=seed if method == "bootstrap" else None,
        details=details,
        warnings=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------- #
# Public: Lo (2002) annualisation
# --------------------------------------------------------------------------- #
def annualize_sharpe(
    sharpe: float | np.ndarray,
    periods: int,
    *,
    autocorrelations: Sequence[float] | np.ndarray | None = None,
) -> float | np.ndarray:
    """Annualise a per-period Sharpe ratio, correcting for autocorrelation (Lo 2002).

    ``SR_q = eta(q) SR`` with
    ``eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k)``.
    With ``autocorrelations=None`` this is the familiar ``sqrt(q)``, which is
    only right for serially uncorrelated returns, so a :class:`UserWarning` is
    issued. The autocorrelations are an *input*: no estimation hides here.

    Parameters
    ----------
    sharpe : float or ndarray of shape (M,)
        Per-period Sharpe ratio(s).
    periods : int
        Periods per year ``q`` (12 for monthly data).
    autocorrelations : array-like of shape (K,) or (K, M), optional
        ``rho_1, ..., rho_K``; lags beyond ``K`` (up to ``q - 1``) count as 0,
        and lags beyond ``q - 1`` are ignored. A 2-D array gives one column per
        Sharpe ratio.

    Returns
    -------
    float or ndarray
        ``nan`` where the variance term ``q + 2 sum (q - k) rho_k`` is not
        positive.

    Examples
    --------
    >>> round(annualize_sharpe(0.1, 12, autocorrelations=[0.0]), 6)
    0.34641
    """
    q = int(periods)
    if q < 1:
        raise ValueError(f"`periods` must be >= 1, got {periods}.")
    sr = np.asarray(sharpe, dtype=np.float64)
    if autocorrelations is None:
        warnings.warn(
            "annualize_sharpe: sqrt(q) scaling assumes serially uncorrelated returns; "
            "pass `autocorrelations` for Lo's (2002) correction.",
            UserWarning,
            stacklevel=2,
        )
        eta: float | np.ndarray = math.sqrt(q)
    else:
        rho = np.asarray(autocorrelations, dtype=np.float64)
        if rho.ndim == 0:
            rho = rho[None]
        k_max = q - 1
        rho = rho[:k_max]
        k = np.arange(1, rho.shape[0] + 1, dtype=np.float64)
        weights = (q - k) if rho.ndim == 1 else (q - k)[:, None]
        denom = q + 2.0 * np.sum(weights * rho, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            eta = np.where(
                denom > 0, q / np.sqrt(np.where(denom > 0, denom, 1.0)), np.nan
            )
    out = sr * eta
    if np.ndim(out) == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)
