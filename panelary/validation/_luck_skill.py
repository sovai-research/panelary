"""Luck versus skill across many strategies, funds or agents.

When hundreds of strategies are tested, some look skilled by luck alone. Three
complementary answers:

* :func:`storey_pi0` -- the share of true nulls ``pi0`` among the tested
  hypotheses (Storey 2002), with the bootstrap choice of the tuning ``lambda``.
  It feeds :func:`~panelary.validation.benjamini_hochberg` (``pi0=``, adaptive
  BH / Storey q-values).
* :func:`luck_versus_skill` -- the Barras, Scaillet & Wermers (2010) false
  discovery decomposition: at each significance level ``gamma``, how many of the
  "significant" strategies are expected to be lucky, split by sign.
* :func:`alpha_bootstrap` -- the cross-sectional bootstrap of factor-model
  alpha t-statistics. ``scheme="joint_dates"`` is Fama & French (2010): impose
  zero alpha, then resample **dates jointly** for every fund, which keeps the
  cross-fund correlation that makes extreme t-statistics easy to reach by chance.
  ``scheme="residual"`` is Kosowski, Timmermann, Wermers & White (2006):
  independent residual resampling per fund with factors held fixed; under a
  common omitted factor its null is too narrow for the extreme quantiles (a
  pinned test shows it over-rejects while the joint-date scheme stays sized).

The joint-date scheme is computed with the count-matrix engine: with ``C (B, T)``
counting how often each date appears in each replicate and ``A (T, N)`` the
availability mask, every per-fund regression ingredient (Gram entries, ``X'y``,
``y'y``, ``n``) is one BLAS product, and the ``B * N`` small OLS systems are
solved in one batched call. Unbalanced panels are supported.

References
----------
Storey, J. D. (2002). A direct approach to false discovery rates. *JRSS-B*
64(3), 479-498. Storey, J. D., Taylor, J. E. & Siegmund, D. (2004). *JRSS-B*
66(1), 187-205.
Barras, L., Scaillet, O. & Wermers, R. (2010). False discoveries in mutual fund
performance. *J. Finance* 65(1), 179-216.
Fama, E. F. & French, K. R. (2010). Luck versus skill in the cross-section of
mutual fund returns. *J. Finance* 65(5), 1915-1947.
Kosowski, R., Timmermann, A., Wermers, R. & White, H. (2006). Can mutual fund
"stars" really pick stocks? *J. Finance* 61(6), 2551-2595.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from panelary._internal import _special
from panelary.validation._arrays import as_matrix, resolve_names
from panelary.validation._bootstrap import block_bootstrap_indices
from panelary.validation._resample import (
    count_matrix,
    matmul_cols,
    row_chunks,
    stationary_indices,
)
from panelary.validation._results import EvaluationResult

__all__ = [
    "AlphaBootstrapResult",
    "Pi0Estimate",
    "alpha_bootstrap",
    "luck_versus_skill",
    "storey_pi0",
]

_DEFAULT_LAMBDAS = tuple(round(0.05 * k, 2) for k in range(1, 20))
_CHUNK_BYTES = 64 * 2**20
#: Hadamard ratio det(G) / prod(diag G) below which a replicate's Gram is
#: treated as ill-conditioned and solved by gather + lstsq instead.
_HADAMARD_FLOOR = 1e-10


# --------------------------------------------------------------------------- #
# Storey pi0
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pi0Estimate:
    """Storey's estimate of the proportion of true null hypotheses.

    Attributes
    ----------
    pi0 : float
        ``min(1, pi0_hat(lambda*))``.
    lambda_ : float
        The tuning value chosen by bootstrap MSE.
    lambdas, pi0_lambda, mse : numpy.ndarray
        The grid, ``#{p > lambda} / (M (1 - lambda))`` on it, and the bootstrap
        mean squared error against ``min_lambda pi0_hat(lambda)``.
    n_tests, n_boot : int
    seed : int or None
    """

    pi0: float
    lambda_: float
    lambdas: np.ndarray
    pi0_lambda: np.ndarray
    mse: np.ndarray
    n_tests: int
    n_boot: int
    seed: int | None

    def to_frame(self) -> pl.DataFrame:
        """One row per ``lambda``: ``lambda, pi0_lambda, mse, selected``."""
        return pl.DataFrame(
            {
                "lambda": self.lambdas,
                "pi0_lambda": self.pi0_lambda,
                "mse": self.mse,
                "selected": self.lambdas == self.lambda_,
            }
        )

    def __float__(self) -> float:
        return float(self.pi0)


def _check_p(pvalues: Any) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("`pvalues` must not be empty.")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("`pvalues` must all be finite and in [0, 1].")
    return p


def storey_pi0(
    pvalues: Any,
    *,
    lambdas: Sequence[float] | None = None,
    n_boot: int = 1000,
    seed: int | None = 0,
) -> Pi0Estimate:
    """Storey (2002) estimate of the share of true nulls, bootstrap-tuned.

    ``pi0_hat(lambda) = #{p > lambda} / (M (1 - lambda))`` on a grid of
    ``lambda`` (default 0.05, 0.10, ..., 0.95). ``lambda*`` minimises the
    bootstrap mean squared error of ``pi0_hat*(lambda)`` against
    ``min_lambda pi0_hat(lambda)`` (Storey 2002; the choice of Barras et al.
    2010). All ``B`` bootstrap p-vectors are bucketed with one ``bincount``, so
    the cost is ``O(B M)`` with no Python loop over replicates. The
    Storey-Tibshirani spline smoother is not implemented (it needs scipy).

    Parameters
    ----------
    pvalues : array-like
        Finite p-values in ``[0, 1]``.
    lambdas : sequence of float, optional
        Grid in ``(0, 1)``.
    n_boot : int, default 1000
    seed : int, optional

    Returns
    -------
    Pi0Estimate
    """
    p = _check_p(pvalues)
    m = p.size
    lam = np.asarray(_DEFAULT_LAMBDAS if lambdas is None else lambdas, dtype=np.float64)
    lam = np.unique(lam)
    if lam.size == 0 or np.any((lam <= 0) | (lam >= 1)):
        raise ValueError("`lambdas` must lie in (0, 1).")
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1, got {n_boot}.")
    n_lam = lam.size
    bucket = np.searchsorted(lam, p, side="left")  # number of lambdas strictly below p

    def greater_counts(counts: np.ndarray) -> np.ndarray:
        # #{p > lambda_j} = sum of buckets j+1 .. L
        rev = np.cumsum(counts[..., ::-1], axis=-1)[..., ::-1]
        return rev[..., 1:]

    obs = greater_counts(np.bincount(bucket, minlength=n_lam + 1).astype(np.float64))
    pi0_lam = obs / (m * (1.0 - lam))
    target = float(pi0_lam.min())
    rng = np.random.default_rng(seed)
    sq = np.zeros(n_lam)
    for rows in row_chunks(int(n_boot), m * 8 * 2, _CHUNK_BYTES):
        nb = rows.stop - rows.start
        ids = bucket[rng.integers(0, m, size=(nb, m))]
        flat = (np.arange(nb)[:, None] * (n_lam + 1) + ids).ravel()
        counts = np.bincount(flat, minlength=nb * (n_lam + 1)).reshape(nb, n_lam + 1)
        boot = greater_counts(counts.astype(np.float64)) / (m * (1.0 - lam))
        sq += np.sum((boot - target) ** 2, axis=0)
    mse = sq / n_boot
    best = int(np.argmin(mse))
    return Pi0Estimate(
        pi0=float(min(1.0, pi0_lam[best])),
        lambda_=float(lam[best]),
        lambdas=lam,
        pi0_lambda=pi0_lam,
        mse=mse,
        n_tests=m,
        n_boot=int(n_boot),
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Barras-Scaillet-Wermers decomposition
# --------------------------------------------------------------------------- #
def luck_versus_skill(
    tstats: Any,
    *,
    gammas: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    pi0: str | float | Pi0Estimate = "storey",
    dof: float | np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int | None = 0,
) -> pl.DataFrame:
    """Barras, Scaillet & Wermers (2010) false-discovery decomposition.

    Two-sided p-values from the t-statistics (Student-t with ``dof``, or
    ``N(0, 1)``). For each significance level ``gamma``:

    * ``s_plus`` / ``s_minus``: share significant with a positive / negative t;
    * ``f_plus = f_minus = pi0 gamma / 2``: expected share of lucky (resp.
      unlucky) nulls;
    * ``t_plus = s_plus - f_plus`` and ``t_minus = s_minus - f_minus``: shares of
      truly skilled / unskilled among the significant;
    * ``fdr_plus = f_plus / s_plus`` (``nan`` when nothing is significant).

    ``pi_a_plus`` and ``pi_a_minus`` (the population shares of skilled and
    unskilled) are read at the largest ``gamma`` and floored at 0.

    Parameters
    ----------
    tstats : array-like of shape (M,)
        Alpha (or performance) t-statistics; ``nan`` entries are dropped.
    gammas : sequence of float
        Significance levels in ``(0, 1)``.
    pi0 : "storey", float or Pi0Estimate, default "storey"
        Share of true nulls; ``"storey"`` estimates it with :func:`storey_pi0`.
    dof : float or array-like, optional
        Degrees of freedom for Student-t p-values (per test allowed).
    n_boot, seed
        Passed to :func:`storey_pi0`.

    Returns
    -------
    polars.DataFrame
        One row per ``gamma``.
    """
    t = np.asarray(tstats, dtype=np.float64).ravel()
    keep = np.isfinite(t)
    if dof is not None and np.ndim(dof) > 0:
        dof_arr = np.asarray(dof, dtype=np.float64).ravel()
        if dof_arr.size != t.size:
            raise ValueError("`dof` must be a scalar or match `tstats`.")
        dof_arr = dof_arr[keep]
    else:
        dof_arr = None
    t = t[keep]
    if t.size == 0:
        raise ValueError("`tstats` has no finite values.")
    g = np.asarray(gammas, dtype=np.float64)
    if g.size == 0 or np.any((g <= 0) | (g >= 1)):
        raise ValueError("`gammas` must lie in (0, 1).")
    if dof is None:
        p = np.minimum(_two_sided_normal(t), 1.0)
    else:
        df = dof_arr if dof_arr is not None else float(dof)
        p = np.minimum(
            2.0 * np.asarray(_special.t_sf(np.abs(t), df), dtype=np.float64), 1.0
        )
    if isinstance(pi0, str):
        if pi0 != "storey":
            raise ValueError("`pi0` must be 'storey', a float or a Pi0Estimate.")
        pi0_val = storey_pi0(p, n_boot=n_boot, seed=seed).pi0
    else:
        pi0_val = float(pi0)
    if not 0.0 < pi0_val <= 1.0:
        raise ValueError(f"`pi0` must be in (0, 1], got {pi0_val}.")
    m = t.size
    sig = p[None, :] <= g[:, None]
    s_plus = np.sum(sig & (t > 0), axis=1) / m
    s_minus = np.sum(sig & (t < 0), axis=1) / m
    f = pi0_val * g / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr_plus = np.where(s_plus > 0, f / s_plus, np.nan)
        fdr_minus = np.where(s_minus > 0, f / s_minus, np.nan)
    top = int(np.argmax(g))
    return pl.DataFrame(
        {
            "gamma": g,
            "s_plus": s_plus,
            "s_minus": s_minus,
            "f_plus": f,
            "f_minus": f,
            "t_plus": s_plus - f,
            "t_minus": s_minus - f,
            "fdr_plus": fdr_plus,
            "fdr_minus": fdr_minus,
            "pi0": np.full(g.size, pi0_val),
            "pi_a_plus": np.full(g.size, max(0.0, float(s_plus[top] - f[top]))),
            "pi_a_minus": np.full(g.size, max(0.0, float(s_minus[top] - f[top]))),
            "n_tests": np.full(g.size, m, dtype=np.int64),
        }
    )


def _two_sided_normal(t: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(_special.norm_sf(np.abs(t)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Cross-sectional alpha bootstrap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class AlphaBootstrapResult:
    """Outcome of :func:`alpha_bootstrap`.

    Attributes
    ----------
    quantiles : polars.DataFrame
        The Fama-French (2010) table, one row per cross-sectional quantile:
        ``quantile, observed, null_median, null_p05, null_p95, pvalue,
        pct_null_below`` (``pvalue`` is the upper tail for quantiles >= 0.5 and
        the lower tail below; ``pct_null_below`` is FF's "%<Act").
    names : tuple of str
    alpha, t_alpha : numpy.ndarray
        ``(N,)`` observed alphas (per period) and OLS t-statistics.
    n_obs : numpy.ndarray
        ``(N,)`` observations per fund.
    pvalue : numpy.ndarray
        ``(N,)`` per-fund one-sided bootstrap p-value ``P(t* >= t)``.
    null_size : numpy.ndarray
        ``(N,)`` valid replicates per fund (those with ``>= min_obs`` dates).
    null_quantiles : numpy.ndarray
        ``(B, Q)`` bootstrap cross-sectional quantiles.
    scheme : str
    block_length, n_boot : int
    seed : int or None
    factors : int
        Number of factors ``K``.
    warnings : tuple of str
    """

    quantiles: pl.DataFrame
    names: tuple[str, ...]
    alpha: np.ndarray
    t_alpha: np.ndarray
    n_obs: np.ndarray
    pvalue: np.ndarray
    null_size: np.ndarray
    null_quantiles: np.ndarray
    scheme: str
    block_length: int
    n_boot: int
    seed: int | None
    factors: int
    warnings: tuple[str, ...] = field(default=())

    def to_frame(self) -> pl.DataFrame:
        """Per-fund frame: ``fund, alpha, t_alpha, n_obs, pvalue, null_size``."""
        return pl.DataFrame(
            {
                "fund": list(self.names),
                "alpha": self.alpha,
                "t_alpha": self.t_alpha,
                "n_obs": self.n_obs,
                "pvalue": self.pvalue,
                "null_size": self.null_size,
            }
        )

    def to_evaluation(self) -> EvaluationResult:
        """Per-fund bootstrap tests as an :class:`EvaluationResult`."""
        ref = "ff2010-joint-dates" if self.scheme == "joint_dates" else "ktww-residual"
        return EvaluationResult(
            test="alpha_bootstrap",
            names=self.names,
            estimate=self.alpha,
            statistic=self.t_alpha,
            pvalue=self.pvalue,
            reference=f"{ref}(b={self.block_length},B={self.n_boot})",
            alternative="greater",
            n_obs=self.n_obs,
            block_length=self.block_length,
            n_resamples=self.n_boot,
            seed=self.seed,
            details={"null_size": self.null_size.astype(np.float64)},
            warnings=self.warnings,
        )


def _pair_index(k: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(k) for b in range(a, k)]


def _spd_solve(gram: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched Cholesky solve of SPD systems ``gram x = rhs`` for small ``k``.

    Vectorised over the batch (``k`` is tiny, so the loops are over ``k`` only),
    which is several times faster than ``numpy.linalg.solve`` on millions of
    4x4 systems. Returns ``(x, hadamard)`` with ``hadamard = det / prod(diag)``
    in ``(0, 1]`` from the pivots (``<= 0`` or ``nan`` flags a non-SPD system).
    """
    k = gram.shape[-1]
    chol = np.zeros_like(gram)
    ratio = np.ones(gram.shape[:-2])
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(k):
            piv = gram[..., j, j] - np.sum(chol[..., j, :j] ** 2, axis=-1)
            ratio = ratio * piv / gram[..., j, j]
            ljj = np.sqrt(np.where(piv > 0, piv, np.nan))
            chol[..., j, j] = ljj
            for i in range(j + 1, k):
                chol[..., i, j] = (
                    gram[..., i, j]
                    - np.sum(chol[..., i, :j] * chol[..., j, :j], axis=-1)
                ) / ljj
        z = np.empty_like(rhs)
        for i in range(k):
            z[..., i, :] = (
                rhs[..., i, :]
                - np.einsum("...m,...mr->...r", chol[..., i, :i], z[..., :i, :])
            ) / chol[..., i, i][..., None]
        x = np.empty_like(rhs)
        for i in range(k - 1, -1, -1):
            x[..., i, :] = (
                z[..., i, :]
                - np.einsum(
                    "...m,...mr->...r", chol[..., i + 1 :, i], x[..., i + 1 :, :]
                )
            ) / chol[..., i, i][..., None]
    return x, ratio


def _ols_from_moments(
    gram: np.ndarray, xy: np.ndarray, yy: np.ndarray, n: np.ndarray, min_obs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched OLS alpha and its t from sufficient statistics.

    ``gram (..., k, k)``, ``xy (..., k)``, ``yy (...)``, ``n (...)``. Returns
    ``(alpha, t, ill)`` where ``ill`` flags systems with enough observations
    but an ill-conditioned Gram (Hadamard ratio below :data:`_HADAMARD_FLOOR`),
    which the caller re-solves by lstsq.
    """
    k = gram.shape[-1]
    enough = n >= max(min_obs, k + 1)
    rhs = np.zeros((*xy.shape, 2))
    rhs[..., 0] = xy
    rhs[..., 0, 1] = 1.0
    sol, ratio = _spd_solve(gram, rhs)
    ill = enough & ~(ratio > _HADAMARD_FLOOR)
    ok = enough & ~ill
    beta = sol[..., 0]
    inv00 = sol[..., 0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rss = np.maximum(yy - np.einsum("...k,...k->...", beta, xy), 0.0)
        s2 = rss / (n - k)
        alpha = np.where(ok, beta[..., 0], np.nan)
        tval = np.where(ok, beta[..., 0] / np.sqrt(s2 * inv00), np.nan)
    return alpha, tval, ill


def _joint_dates_stats(
    counts: np.ndarray,
    avail: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-replicate, per-fund OLS alpha and t via count-matrix products.

    ``counts (B, T)``, ``avail (T, N)`` in {0, 1}, ``x (T, k)`` regressors,
    ``y (T, N)`` returns (zero where unavailable). Returns ``alpha, t, n, ill``,
    each ``(B, N)``.
    """
    k = x.shape[1]
    nb = counts.shape[0]
    nf = avail.shape[1]
    n = matmul_cols(counts, avail)
    gram = np.empty((nb, nf, k, k))
    for a, b in _pair_index(k):
        val = matmul_cols(counts, avail * (x[:, a] * x[:, b])[:, None])
        gram[:, :, a, b] = val
        gram[:, :, b, a] = val
    xy = np.empty((nb, nf, k))
    for a in range(k):
        xy[:, :, a] = matmul_cols(counts, y * x[:, [a]])
    yy = matmul_cols(counts, y * y)
    alpha, tval, ill = _ols_from_moments(gram, xy, yy, n, min_obs)
    return alpha, tval, n, ill


def _gather_ols(
    x: np.ndarray, y: np.ndarray, rows: np.ndarray, min_obs: int
) -> tuple[float, float]:
    """Alpha and t of one resampled regression, by lstsq (the ill-conditioned fallback)."""
    xx, yy = x[rows], y[rows]
    k = xx.shape[1]
    if rows.size < max(min_obs, k + 1):
        return float("nan"), float("nan")
    beta, _, rank, _ = np.linalg.lstsq(xx, yy, rcond=None)
    if rank < k:
        return float("nan"), float("nan")
    resid = yy - xx @ beta
    s2 = float(resid @ resid) / (rows.size - k)
    inv = np.linalg.pinv(xx.T @ xx)
    se = math.sqrt(s2 * inv[0, 0])
    return float(beta[0]), float(beta[0] / se) if se > 0 else float("nan")


def _date_indices(
    n_t: int, block_length: int, n_boot: int, seed: int | None
) -> np.ndarray:
    """Stationary-bootstrap date indices, the stream of ``block_bootstrap_indices``.

    For ``block_length == 1`` the stationary scheme draws one uniform position
    per element, so a single vectorised ``integers`` call reproduces its stream
    exactly (pinned by a test) without the per-replicate Python loop.
    """
    if block_length == 1:
        return np.random.default_rng(seed).integers(0, n_t, size=(n_boot, n_t))
    return block_bootstrap_indices(
        n_t, block_length=block_length, n_boot=n_boot, scheme="stationary", seed=seed
    )


def _cross_quantiles(t: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Cross-sectional quantiles along the last axis, ignoring nan (``(..., Q)``)."""
    if np.isnan(t).any():
        with np.errstate(invalid="ignore"):
            out = np.nanquantile(t, qs, axis=-1)
    else:
        out = np.quantile(t, qs, axis=-1)
    return np.moveaxis(np.asarray(out), 0, -1)


def alpha_bootstrap(
    returns: Any,
    factors: Any = None,
    *,
    scheme: str = "joint_dates",
    block_length: int = 1,
    n_boot: int = 1000,
    quantiles: Sequence[float] = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99),
    min_obs: int = 24,
    seed: int | None = 0,
    names: Sequence[str] | None = None,
    chunk_bytes: int = _CHUNK_BYTES,
) -> AlphaBootstrapResult:
    """Cross-sectional bootstrap of alpha t-statistics under the null of no skill.

    Each fund's excess returns are regressed on the factors (with an
    intercept); the null of zero alpha is imposed by subtracting ``alpha_hat``
    and the distribution of cross-sectional t-statistic quantiles is simulated.

    Parameters
    ----------
    returns : array-like of shape (T, N)
        Excess returns, ``nan`` where a fund does not exist (unbalanced panels).
    factors : array-like of shape (T, K), optional
        Factor returns (complete). ``None`` tests raw mean returns (``K = 0``).
    scheme : {"joint_dates", "residual"}, default "joint_dates"
        ``"joint_dates"`` (Fama-French 2010): one index vector per replicate,
        applied to every fund's returns *and* the factors, so cross-fund
        correlation is preserved; a stationary bootstrap over dates
        (``block_length = 1`` is FF's i.i.d. date resampling).
        ``"residual"`` (Kosowski et al. 2006): per-fund residual resampling
        with the factors held fixed; each fund's generator is
        ``default_rng([seed, crc32(name)])``, so adding a fund never changes
        another's draws.
    block_length : int, default 1
    n_boot : int, default 1000
    quantiles : sequence of float
        Cross-sectional quantiles of ``t(alpha)`` to test.
    min_obs : int, default 24
        Funds (and replicates) with fewer observations are dropped.
    seed : int, optional
    names : sequence of str, optional
    chunk_bytes : int
        Working-set budget for the count-matrix products.

    Returns
    -------
    AlphaBootstrapResult

    Notes
    -----
    ``factors=None`` with ``block_length`` and ``seed`` equal to
    :func:`panelary.evolve.cross_sectional_bootstrap`'s reproduces its
    statistics (to ~1e-15) and p-values: both draw the same stationary-bootstrap
    indices.
    """
    if scheme not in ("joint_dates", "residual"):
        raise ValueError("`scheme` must be 'joint_dates' or 'residual'.")
    if n_boot < 1 or block_length < 1:
        raise ValueError("`n_boot` and `block_length` must be >= 1.")
    mat, colnames = as_matrix(returns, "returns")
    n_t, n_f = mat.shape
    labels = resolve_names(names, n_f, colnames)
    if factors is None:
        fac = np.zeros((n_t, 0))
    else:
        fac, _ = as_matrix(factors, "factors")
        if fac.shape[0] != n_t:
            raise ValueError(f"`factors` has {fac.shape[0]} rows; expected {n_t}.")
        if not np.all(np.isfinite(fac)):
            raise ValueError("`factors` must be complete (no NaN).")
    x = np.column_stack([np.ones(n_t), fac])
    k = x.shape[1]
    qs = np.asarray(quantiles, dtype=np.float64)
    if qs.size == 0 or np.any((qs < 0) | (qs > 1)):
        raise ValueError("`quantiles` must lie in [0, 1].")
    notes: list[str] = []

    avail = np.isfinite(mat).astype(np.float64)
    y = np.where(avail > 0, mat, 0.0)
    ones = np.ones((1, n_t))
    a_obs, t_obs, n_obs, ill_obs = _joint_dates_stats(ones, avail, x, y, min_obs)
    alpha_hat, t_hat, nobs = a_obs[0], t_obs[0], n_obs[0].astype(np.int64)
    for i in np.flatnonzero(ill_obs[0]):
        alpha_hat[i], t_hat[i] = _gather_ols(
            x, mat[:, i], np.flatnonzero(avail[:, i] > 0), min_obs
        )
    dropped = int(np.sum(nobs < max(min_obs, k + 1)))
    if dropped:
        notes.append(
            f"{dropped} fund(s) have fewer than {min_obs} observations and are excluded."
        )
    active = np.isfinite(t_hat)
    if not active.any():
        raise ValueError(f"no fund has at least {min_obs} observations.")
    y0 = np.where(avail > 0, mat - np.where(active, alpha_hat, 0.0), 0.0)  # impose H0

    t_star = np.full((int(n_boot), n_f), np.nan)
    if scheme == "joint_dates":
        idx = _date_indices(n_t, int(block_length), int(n_boot), seed)
        row_bytes = 8 * (n_t + n_f * (k * k + 2 * k + 4))
        n_fallback = 0
        for rows in row_chunks(int(n_boot), row_bytes, chunk_bytes):
            counts = count_matrix(idx[rows], n_t)
            _, tb, _, ill = _joint_dates_stats(counts, avail, x, y0, min_obs)
            for b, i in zip(*np.nonzero(ill), strict=True):
                pos = idx[rows.start + b]
                pos = pos[avail[pos, i] > 0]
                tb[b, i] = _gather_ols(x, y0[:, i], pos, min_obs)[1]
                n_fallback += 1
            t_star[rows] = tb
        if n_fallback:
            notes.append(
                f"{n_fallback} ill-conditioned replicate regression(s) solved by lstsq."
            )
    else:
        for i in np.flatnonzero(active):
            rows_i = np.flatnonzero(avail[:, i] > 0)
            xi, yi = x[rows_i], mat[rows_i, i]
            beta, *_ = np.linalg.lstsq(xi, yi, rcond=None)
            resid = yi - xi @ beta
            n_i = rows_i.size
            rng = np.random.default_rng(
                [int(seed if seed is not None else 0), zlib.crc32(labels[i].encode())]
            )
            if block_length == 1:
                pos = rng.integers(0, n_i, size=(int(n_boot), n_i))
            else:
                pos = stationary_indices(
                    n_i, float(block_length), int(n_boot), seed=rng
                )
            e_star = resid[pos]  # (B, n_i): X held fixed, residuals resampled
            xte = e_star @ xi  # (B, k)
            xtx_inv = np.linalg.inv(xi.T @ xi)
            coef = xte @ xtx_inv.T
            rss = np.einsum("bt,bt->b", e_star, e_star) - np.einsum(
                "bk,bk->b", coef, xte
            )
            s2 = np.maximum(rss, 0.0) / (n_i - k)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_star[:, i] = coef[:, 0] / np.sqrt(s2 * xtx_inv[0, 0])

    obs_q = _cross_quantiles(t_hat[active], qs)
    null_q = _cross_quantiles(t_star[:, active], qs)  # (B, Q)
    upper = qs >= 0.5
    with np.errstate(invalid="ignore"):
        ge = np.sum(null_q >= obs_q, axis=0)
        le = np.sum(null_q <= obs_q, axis=0)
        below = np.sum(null_q < obs_q, axis=0)
    pq = (1.0 + np.where(upper, ge, le)) / (1.0 + n_boot)
    table = pl.DataFrame(
        {
            "quantile": qs,
            "observed": obs_q,
            "null_median": np.median(null_q, axis=0),
            "null_p05": np.quantile(null_q, 0.05, axis=0),
            "null_p95": np.quantile(null_q, 0.95, axis=0),
            "pvalue": pq,
            "pct_null_below": below / n_boot,
        }
    )
    valid = np.isfinite(t_star)
    null_size = valid.sum(axis=0).astype(np.int64)
    with np.errstate(invalid="ignore"):
        exceed = np.sum(valid & (t_star >= t_hat), axis=0)
    fund_p = np.where(
        active & (null_size > 0), (1.0 + exceed) / (1.0 + null_size), np.nan
    )
    return AlphaBootstrapResult(
        quantiles=table,
        names=labels,
        alpha=alpha_hat,
        t_alpha=t_hat,
        n_obs=nobs,
        pvalue=fund_p,
        null_size=null_size,
        null_quantiles=null_q,
        scheme=scheme,
        block_length=int(block_length),
        n_boot=int(n_boot),
        seed=seed,
        factors=k - 1,
        warnings=tuple(notes),
    )
