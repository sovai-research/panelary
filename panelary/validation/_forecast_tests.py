"""Forecast-comparison tests and proper scoring rules.

Two families:

**Is model A really better than model B?**

* :func:`diebold_mariano` — the pairwise Diebold-Mariano (1995) test on a loss
  differential, with a Newey-West/Bartlett HAC long-run variance and the
  Harvey-Leybourne-Newbold (1997) small-sample correction, referred to a
  Student-t distribution. Implemented natively (no SciPy, no ``arch``).
* :func:`superior_predictive_ability` — Hansen's (2005) SPA test: is *any* of
  ``M`` models better than a benchmark, correcting for the search over ``M``?
* :func:`model_confidence_set` — Hansen, Lunde & Nason (2011): the set of models
  that contains the best one with a given confidence.

SPA and the MCS are built on the stationary bootstrap in
:mod:`panelary.validation._bootstrap` rather than wrapping ``arch``: the
resampling engine already had to exist for the Romano-Wolf stepdown, ``arch`` is
not a dependency Panelary can take, and re-using Panelary's own bootstrap is
what lets the fold-boundary guarantee ("blocks never straddle a fold") extend to
these tests too.

**How good is a probabilistic forecast?** — proper scoring rules:
:func:`crps_ensemble`, :func:`crps_gaussian`, :func:`crps_from_quantiles`,
:func:`pinball_loss`, :func:`interval_score`, :func:`pit_values` and
:func:`pit_histogram`, plus the Polars-native aggregator
:func:`score_quantile_forecasts` for scoring a whole panel in one pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from panelary._internal import _special
from panelary.core.model_selection import _norm_cdf
from panelary.validation._bootstrap import block_bootstrap_indices

__all__ = [
    "DieboldMarianoResult",
    "MCSResult",
    "SPAResult",
    "crps_ensemble",
    "crps_from_quantiles",
    "crps_gaussian",
    "diebold_mariano",
    "interval_score",
    "model_confidence_set",
    "newey_west_variance",
    "pinball_loss",
    "pinball_loss_expr",
    "pit_histogram",
    "pit_values",
    "score_quantile_forecasts",
    "superior_predictive_ability",
]

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Small scipy-free distribution helpers (private; `_numpy_stats` is not ours)
# --------------------------------------------------------------------------- #
def _norm_pdf(x: float | np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


# The incomplete beta and Student-t tail live in `panelary._internal._special`;
# the private names below are kept for existing importers.
_betacf = _special._betacf
_betainc = _special._betainc_scalar
_t_sf = _special._t_sf_scalar


# --------------------------------------------------------------------------- #
# HAC long-run variance
# --------------------------------------------------------------------------- #
def newey_west_variance(
    x: np.ndarray, *, lags: int | None = None, demean: bool = True
) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of a series.

    ``LRV = gamma_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) * gamma_j`` where
    ``gamma_j`` is the sample autocovariance at lag ``j``. This is the HAC
    variance every serial-correlation-robust test in this module uses.

    Parameters
    ----------
    x : ndarray of shape (T,)
        The series.
    lags : int, optional
        Bartlett truncation lag ``L``. Defaults to the Newey-West automatic rule
        ``floor(4 * (T / 100) ** (2/9))``.
    demean : bool, default=True
        Subtract the sample mean before computing autocovariances.

    Returns
    -------
    float
        The long-run variance (non-negative by construction of the Bartlett
        kernel).

    Raises
    ------
    ValueError
        If ``x`` has fewer than 2 finite observations or ``lags`` is negative.
    """
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        raise ValueError(f"need at least 2 finite observations, got {n}.")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    if lags < 0:
        raise ValueError(f"`lags` must be >= 0, got {lags}.")
    lags = min(lags, n - 1)
    e = arr - arr.mean() if demean else arr
    lrv = float(np.dot(e, e) / n)
    for j in range(1, lags + 1):
        gamma = float(np.dot(e[j:], e[:-j]) / n)
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * gamma
    return float(max(lrv, 0.0))


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DieboldMarianoResult:
    """Result of a Diebold-Mariano test.

    Attributes
    ----------
    statistic : float
        The (HLN-corrected, if requested) DM statistic. Positive means model A
        has the *higher* loss, i.e. model B forecasts better.
    pvalue : float
        Two-sided or one-sided p-value from the Student-t reference
        distribution with ``T - 1`` degrees of freedom.
    mean_loss_differential : float
        ``mean(loss_a - loss_b)``.
    n_obs : int
        Number of paired observations used.
    horizon : int
        Forecast horizon ``h`` (drives the HAC lag and the HLN correction).
    alternative : str
        The alternative hypothesis tested.
    harvey_correction : bool
        Whether the small-sample correction was applied.
    """

    statistic: float
    pvalue: float
    mean_loss_differential: float
    n_obs: int
    horizon: int
    alternative: str
    harvey_correction: bool


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    alternative: str = "two-sided",
    lags: int | None = None,
    harvey_correction: bool = True,
) -> DieboldMarianoResult:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests ``H0: E[loss_a - loss_b] = 0`` using the HAC-standardised mean loss
    differential. With ``horizon = h`` the default HAC truncation is ``h - 1``
    lags (the MA order an optimal ``h``-step forecast error can have). The
    Harvey-Leybourne-Newbold (1997) correction rescales the statistic by
    ``sqrt((T + 1 - 2h + h(h-1)/T) / T)`` and refers it to ``t_{T-1}`` rather
    than the standard normal, which materially improves size in small samples.

    Parameters
    ----------
    loss_a, loss_b : ndarray of shape (T,)
        Per-period **losses** (not forecasts) of the two models: squared errors,
        absolute errors, pinball losses, negative log-likelihoods — any loss.
    horizon : int, default=1
        Forecast horizon ``h >= 1``.
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        ``"greater"`` tests ``E[loss_a - loss_b] > 0`` (model B is better).
    lags : int, optional
        HAC truncation lag. Defaults to ``horizon - 1``.
    harvey_correction : bool, default=True
        Apply the HLN small-sample correction.

    Returns
    -------
    DieboldMarianoResult

    Raises
    ------
    ValueError
        If the loss series differ in length, are too short, or the loss
        differential is degenerate (identical models).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.standard_normal(200) ** 2
    >>> b = a * 0.5
    >>> res = diebold_mariano(a, b, alternative="greater")
    >>> res.pvalue < 0.01
    True

    References
    ----------
    Diebold, F. X., & Mariano, R. S. (1995). "Comparing Predictive Accuracy."
    *Journal of Business & Economic Statistics*, 13(3).
    Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the Equality of
    Prediction Mean Squared Errors." *International Journal of Forecasting*,
    13(2).
    """
    la = np.asarray(loss_a, dtype=float).ravel()
    lb = np.asarray(loss_b, dtype=float).ravel()
    if la.shape != lb.shape:
        raise ValueError(
            f"`loss_a` and `loss_b` must have the same shape, got {la.shape} "
            f"and {lb.shape}."
        )
    if horizon < 1:
        raise ValueError(f"`horizon` must be >= 1, got {horizon}.")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(
            f"unknown `alternative` {alternative!r}; expected 'two-sided', "
            "'less' or 'greater'."
        )
    d = la - lb
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise ValueError(f"need at least 3 finite paired observations, got {n}.")

    lrv = newey_west_variance(d, lags=horizon - 1 if lags is None else lags)
    if lrv <= 0:
        raise ValueError(
            "the loss differential has zero long-run variance (the two models "
            "produce identical losses); the DM test is undefined."
        )
    dbar = float(d.mean())
    stat = dbar / math.sqrt(lrv / n)

    if harvey_correction:
        h = horizon
        factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
        if factor <= 0:
            raise ValueError(
                f"horizon={h} is too large relative to T={n} for the "
                "Harvey-Leybourne-Newbold correction."
            )
        stat *= math.sqrt(factor)

    df = float(n - 1)
    if alternative == "two-sided":
        pvalue = 2.0 * _t_sf(abs(stat), df)
    elif alternative == "greater":
        pvalue = _t_sf(stat, df)
    else:
        pvalue = 1.0 - _t_sf(stat, df)
    return DieboldMarianoResult(
        statistic=float(stat),
        pvalue=float(min(max(pvalue, 0.0), 1.0)),
        mean_loss_differential=dbar,
        n_obs=n,
        horizon=horizon,
        alternative=alternative,
        harvey_correction=harvey_correction,
    )


# --------------------------------------------------------------------------- #
# Hansen's SPA
# --------------------------------------------------------------------------- #
def _stationary_variance(d: np.ndarray, block_length: float) -> np.ndarray:
    """Hansen's ``omega_k^2``: the stationary-bootstrap-consistent LRV.

    Uses the kernel weights implied by the stationary bootstrap,
    ``kappa_j = (1 - j/n)(1-q)^j + (j/n)(1-q)^{n-j}`` with ``q = 1/block_length``
    (Hansen 2005, eq. 9; Politis & Romano 1994).
    """
    n, k = d.shape
    q = 1.0 / float(block_length)
    e = d - d.mean(axis=0, keepdims=True)
    omega2 = np.einsum("tk,tk->k", e, e) / n
    for j in range(1, n):
        w = (1.0 - j / n) * (1.0 - q) ** j + (j / n) * (1.0 - q) ** (n - j)
        if w < 1e-12:
            break
        gamma = np.einsum("tk,tk->k", e[j:], e[:-j]) / n
        omega2 += 2.0 * w * gamma
    return np.maximum(omega2, 1e-18)


@dataclass(frozen=True)
class SPAResult:
    """Result of Hansen's Superior Predictive Ability test.

    Attributes
    ----------
    statistic : float
        ``T_SPA = max_k max(0, sqrt(n) * dbar_k / omega_k)``.
    pvalue_consistent : float
        The recommended p-value (Hansen's ``mu^c`` recentring).
    pvalue_lower : float
        Liberal bound (poor models assumed genuinely poor).
    pvalue_upper : float
        Conservative bound (least-favourable configuration: every model on the
        null boundary).
    n_obs : int
        Number of periods.
    n_models : int
        Number of competing models.
    block_length : float
        Mean block length of the stationary bootstrap.
    """

    statistic: float
    pvalue_consistent: float
    pvalue_lower: float
    pvalue_upper: float
    n_obs: int
    n_models: int
    block_length: float

    @property
    def pvalue(self) -> float:
        """Alias for :attr:`pvalue_consistent`."""
        return self.pvalue_consistent


def superior_predictive_ability(
    benchmark_loss: np.ndarray,
    model_losses: np.ndarray,
    *,
    n_boot: int = 1000,
    block_length: float | None = None,
    boundaries: Sequence[int] | None = None,
    seed: int | np.random.Generator | None = None,
) -> SPAResult:
    """Hansen's (2005) test for Superior Predictive Ability.

    ``H0``: the benchmark is not outperformed by **any** of the ``M`` competing
    models, i.e. ``max_k E[L_benchmark - L_k] <= 0``. Rejecting means at least
    one model is genuinely better after correcting for the search across all
    ``M`` — the data-snooping-robust upgrade of a pile of pairwise DM tests.

    Three p-values are returned (Hansen 2005, §3.2). They differ only in how the
    bootstrap is recentred to impose the null, and satisfy
    ``p_lower <= p_consistent <= p_upper``; report the *consistent* one and use
    the other two as a sensitivity band.

    Parameters
    ----------
    benchmark_loss : ndarray of shape (T,)
        Per-period loss of the benchmark model.
    model_losses : ndarray of shape (T, M)
        Per-period losses of the competitors.
    n_boot : int, default=1000
        Stationary-bootstrap replications.
    block_length : float, optional
        Mean block length. Defaults to ``max(2, round(T ** (1/3)))``.
    boundaries : sequence of int, optional
        Fold boundaries the bootstrap blocks must not straddle.
    seed : int | numpy.random.Generator, optional
        Seed.

    Returns
    -------
    SPAResult

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``T < 3``.

    References
    ----------
    Hansen, P. R. (2005). "A Test for Superior Predictive Ability."
    *Journal of Business & Economic Statistics*, 23(4), 365-380.
    """
    bench = np.asarray(benchmark_loss, dtype=float).ravel()
    models = np.asarray(model_losses, dtype=float)
    if models.ndim == 1:
        models = models[:, None]
    if models.ndim != 2 or models.shape[0] != bench.shape[0]:
        raise ValueError(
            f"`model_losses` must be (T, M) aligned with `benchmark_loss` "
            f"(T={bench.shape[0]}), got {models.shape}."
        )
    n, m = models.shape
    if n < 3:
        raise ValueError(f"need at least 3 periods, got {n}.")
    if block_length is None:
        block_length = max(2.0, float(round(n ** (1.0 / 3.0))))

    # d_k = L_benchmark - L_k  (positive => model k beats the benchmark)
    d = bench[:, None] - models
    dbar = d.mean(axis=0)
    omega = np.sqrt(_stationary_variance(d, block_length))
    scaled = math.sqrt(n) * dbar / omega
    statistic = float(max(0.0, float(np.max(scaled))))

    idx = block_bootstrap_indices(
        n,
        block_length=int(max(1, round(block_length))),
        n_boot=n_boot,
        scheme="stationary",
        boundaries=boundaries,
        seed=seed,
    )
    boot_means = np.stack([d[idx[b]].mean(axis=0) for b in range(n_boot)])

    # Recentring g_k imposes the null. A model whose observed mean differential
    # is far enough below zero is treated as genuinely inferior (g_k = 0), which
    # drops it out of the maximum; the remaining models are placed on the null
    # boundary (g_k = dbar_k). Widening the "boundary" set raises the p-value,
    # hence p_lower <= p_consistent <= p_upper (Hansen 2005, sec. 3.2).
    threshold = 0.25 * n ** (-0.25) * omega
    mu_lower = np.where(dbar >= 0.0, dbar, 0.0)
    mu_consistent = np.where(dbar >= -threshold, dbar, 0.0)
    mu_upper = dbar

    def _pvalue(mu: np.ndarray) -> float:
        z = math.sqrt(n) * (boot_means - mu) / omega
        boot_stat = np.maximum(z.max(axis=1), 0.0)
        return float((1.0 + np.count_nonzero(boot_stat >= statistic)) / (n_boot + 1.0))

    return SPAResult(
        statistic=statistic,
        pvalue_consistent=_pvalue(mu_consistent),
        pvalue_lower=_pvalue(mu_lower),
        pvalue_upper=_pvalue(mu_upper),
        n_obs=n,
        n_models=m,
        block_length=float(block_length),
    )


# --------------------------------------------------------------------------- #
# Model Confidence Set
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MCSResult:
    """Result of the Model Confidence Set procedure.

    Attributes
    ----------
    included : list of int
        Column indices of the models in the ``(1 - alpha)`` confidence set.
    excluded : list of int
        Eliminated models, in elimination order (worst first).
    pvalues : numpy.ndarray
        MCS p-value per model (monotone along the elimination sequence). A model
        is in the set iff its MCS p-value exceeds ``alpha``.
    alpha : float
        The confidence level used.
    statistic : str
        Which elimination statistic was used (``"tmax"`` or ``"trange"``).
    names : list, optional
        Model labels, if supplied.
    """

    included: list[int]
    excluded: list[int]
    pvalues: np.ndarray
    alpha: float
    statistic: str
    names: list[object] | None = field(default=None)


def model_confidence_set(
    losses: np.ndarray,
    *,
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_length: float | None = None,
    statistic: str = "tmax",
    boundaries: Sequence[int] | None = None,
    names: Sequence[object] | None = None,
    seed: int | np.random.Generator | None = None,
) -> MCSResult:
    """Model Confidence Set (Hansen, Lunde & Nason, 2011).

    Repeatedly tests ``H0``: all surviving models have equal expected loss. When
    rejected, the worst model is eliminated and the test repeats. What remains
    is the smallest set that contains the truly best model with probability at
    least ``1 - alpha`` — the honest answer to "which of my models can I *not*
    tell apart?".

    Parameters
    ----------
    losses : ndarray of shape (T, M)
        Per-period losses, one column per model.
    alpha : float, default=0.10
        Significance level; the set has confidence ``1 - alpha``.
    n_boot : int, default=1000
        Stationary-bootstrap replications.
    block_length : float, optional
        Mean block length. Defaults to ``max(2, round(T ** (1/3)))``.
    statistic : {"tmax", "trange"}, default="tmax"
        ``"tmax"`` uses ``max_i t_{i.}`` (deviation from the surviving average);
        ``"trange"`` uses ``max_{i,j} |t_{ij}|``.
    boundaries : sequence of int, optional
        Fold boundaries the bootstrap blocks must not straddle.
    names : sequence, optional
        Labels for the models, echoed on the result.
    seed : int | numpy.random.Generator, optional
        Seed.

    Returns
    -------
    MCSResult

    Raises
    ------
    ValueError
        If fewer than 2 models or fewer than 3 periods are supplied, or
        ``statistic`` is unknown.

    References
    ----------
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011). "The Model Confidence Set."
    *Econometrica*, 79(2), 453-497.
    """
    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 2:
        raise ValueError(f"`losses` must be 2-D (T, M), got shape {loss.shape}.")
    n, m = loss.shape
    if m < 2:
        raise ValueError(f"need at least 2 models, got {m}.")
    if n < 3:
        raise ValueError(f"need at least 3 periods, got {n}.")
    if statistic not in {"tmax", "trange"}:
        raise ValueError(
            f"unknown `statistic` {statistic!r}; expected 'tmax' or 'trange'."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    if block_length is None:
        block_length = max(2.0, float(round(n ** (1.0 / 3.0))))

    idx = block_bootstrap_indices(
        n,
        block_length=int(max(1, round(block_length))),
        n_boot=n_boot,
        scheme="stationary",
        boundaries=boundaries,
        seed=seed,
    )
    # Bootstrapped column means, computed once and reused at every elimination
    # step (HLN's recommended implementation).
    boot_means = np.stack([loss[idx[b]].mean(axis=0) for b in range(n_boot)])
    obs_means = loss.mean(axis=0)

    alive = list(range(m))
    excluded: list[int] = []
    pvalues = np.ones(m, dtype=float)
    running = 0.0

    while len(alive) > 1:
        cols = np.asarray(alive, dtype=np.int64)
        mu = obs_means[cols]
        bmu = boot_means[:, cols]
        centred = bmu - mu  # bootstrap deviation under the null

        # Deviation of each model from the surviving average (HLN's d_{i.}),
        # studentised by the bootstrap variance of that deviation. This is also
        # the elimination rule (e_max) for both statistics.
        d_i = mu - mu.mean()
        boot_d = centred - centred.mean(axis=1, keepdims=True)
        var_i = np.maximum((boot_d**2).mean(axis=0), 1e-18)
        t_i = d_i / np.sqrt(var_i)
        worst_local = int(np.argmax(t_i))

        if statistic == "tmax":
            stat_obs = float(np.max(t_i))
            stat_boot = (boot_d / np.sqrt(var_i)).max(axis=1)
        else:
            d_ij = mu[:, None] - mu[None, :]
            boot_dij = centred[:, :, None] - centred[:, None, :]
            var_ij = np.maximum((boot_dij**2).mean(axis=0), 1e-18)
            stat_obs = float(np.max(np.abs(d_ij) / np.sqrt(var_ij)))
            stat_boot = (np.abs(boot_dij) / np.sqrt(var_ij)).max(axis=(1, 2))

        p = float((1.0 + np.count_nonzero(stat_boot >= stat_obs)) / (n_boot + 1.0))
        running = max(running, p)
        loser = alive[worst_local]
        pvalues[loser] = running
        if running > alpha:
            break
        excluded.append(loser)
        alive.remove(loser)

    # Survivors share the p-value at which the set stopped shrinking; a lone
    # survivor (every rival eliminated) has an MCS p-value of 1 by convention.
    for c in alive:
        pvalues[c] = 1.0 if len(alive) == 1 else running
    return MCSResult(
        included=sorted(alive),
        excluded=excluded,
        pvalues=pvalues,
        alpha=float(alpha),
        statistic=statistic,
        names=list(names) if names is not None else None,
    )


# --------------------------------------------------------------------------- #
# Proper scoring rules
# --------------------------------------------------------------------------- #
def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score from an ensemble of draws.

    Uses the energy form ``CRPS = E|X - y| - 0.5 E|X - X'|``, evaluated exactly
    on the empirical ensemble via the sorted-sample identity
    ``sum_{i,j}|x_i - x_j| = 2 * sum_i (2i - m - 1) x_(i)`` — ``O(m log m)``
    rather than ``O(m^2)``.

    Parameters
    ----------
    y_true : ndarray of shape (T,)
        Realised values.
    samples : ndarray of shape (T, M)
        ``M`` predictive draws per observation.

    Returns
    -------
    ndarray of shape (T,)
        Per-observation CRPS (lower is better; 0 is a perfect point forecast).

    Raises
    ------
    ValueError
        If the shapes do not align.

    Examples
    --------
    >>> import numpy as np
    >>> float(crps_ensemble(np.array([0.0]), np.array([[0.0, 0.0]]))[0])
    0.0
    """
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"`samples` must have one row per observation ({y.shape[0]}), got "
            f"{s.shape[0]}."
        )
    m = s.shape[1]
    if m < 1:
        raise ValueError("`samples` must have at least one column.")
    term1 = np.abs(s - y[:, None]).mean(axis=1)
    ordered = np.sort(s, axis=1)
    weights = 2.0 * np.arange(1, m + 1) - m - 1.0
    # sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - m - 1) x_(i), so
    # 0.5 * E|X - X'| = sum_i (2i - m - 1) x_(i) / m^2.
    term2 = (ordered * weights).sum(axis=1) / (m * m)
    return term1 - term2


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution.

    ``CRPS = sigma * [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`` with
    ``z = (y - mu) / sigma`` (Gneiting & Raftery, 2007).

    Parameters
    ----------
    y_true : ndarray
        Realised values.
    mu, sigma : ndarray
        Predictive mean and standard deviation (``sigma > 0``), broadcastable to
        ``y_true``.

    Returns
    -------
    ndarray
        Per-observation CRPS.

    Raises
    ------
    ValueError
        If any ``sigma`` is non-positive.
    """
    y = np.asarray(y_true, dtype=float)
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if np.any(s <= 0):
        raise ValueError("`sigma` must be strictly positive.")
    z = (y - m) / s
    cdf = np.vectorize(_norm_cdf, otypes=[float])(z)
    return s * (z * (2.0 * cdf - 1.0) + 2.0 * _norm_pdf(z) - _INV_SQRT_PI)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    """Pinball (quantile) loss at level ``quantile``.

    ``L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))`` — the proper
    scoring rule for a single quantile.

    Parameters
    ----------
    y_true : ndarray
        Realised values.
    y_pred : ndarray
        Predicted ``quantile``-quantile, broadcastable to ``y_true``.
    quantile : float
        Level ``tau`` in ``(0, 1)``.

    Returns
    -------
    ndarray
        Per-observation loss.

    Raises
    ------
    ValueError
        If ``quantile`` is not in ``(0, 1)``.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"`quantile` must be in (0, 1), got {quantile}.")
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)


def pinball_loss_expr(
    y_true: str | pl.Expr, y_pred: str | pl.Expr, quantile: float
) -> pl.Expr:
    """Polars expression form of :func:`pinball_loss`.

    Lets the loss be computed inside a lazy plan, e.g. grouped per entity, with
    no materialisation.

    Parameters
    ----------
    y_true, y_pred : str | polars.Expr
        Column names or expressions.
    quantile : float
        Level ``tau`` in ``(0, 1)``.

    Returns
    -------
    polars.Expr

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"y": [1.0, 2.0], "q": [0.5, 2.5]})
    >>> out = df.select(pinball_loss_expr("y", "q", 0.9).alias("loss"))
    >>> round(out.item(0, "loss"), 3), round(out.item(1, "loss"), 3)
    (0.45, 0.05)
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"`quantile` must be in (0, 1), got {quantile}.")
    yt = pl.col(y_true) if isinstance(y_true, str) else y_true
    yp = pl.col(y_pred) if isinstance(y_pred, str) else y_pred
    diff = yt - yp
    return pl.max_horizontal(quantile * diff, (quantile - 1.0) * diff)


def crps_from_quantiles(
    y_true: np.ndarray, quantile_preds: np.ndarray, levels: Sequence[float]
) -> np.ndarray:
    """Approximate CRPS from a set of predicted quantiles.

    The CRPS is the integral of the pinball loss over ``tau in (0, 1)``, so a
    quantile grid gives ``CRPS ~ 2 * mean_tau L_tau`` (exact in the limit of a
    dense uniform grid; the standard approximation used by the quantile-forecast
    literature).

    Parameters
    ----------
    y_true : ndarray of shape (T,)
        Realised values.
    quantile_preds : ndarray of shape (T, Q)
        Predicted quantiles.
    levels : sequence of float
        The ``Q`` quantile levels, each in ``(0, 1)``.

    Returns
    -------
    ndarray of shape (T,)
        Per-observation approximate CRPS.

    Raises
    ------
    ValueError
        If the number of levels does not match the prediction columns.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    q = np.asarray(quantile_preds, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    taus = np.asarray(levels, dtype=float).ravel()
    if q.shape[1] != taus.size:
        raise ValueError(
            f"`quantile_preds` has {q.shape[1]} columns but {taus.size} levels "
            "were given."
        )
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("every level must lie in (0, 1).")
    diff = y[:, None] - q
    losses = np.maximum(taus[None, :] * diff, (taus[None, :] - 1.0) * diff)
    return 2.0 * losses.mean(axis=1)


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    """Winkler / interval score for a central ``1 - alpha`` prediction interval.

    ``IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}``:
    a proper score that rewards narrow intervals but penalises misses in
    proportion to how badly the nominal coverage was missed.

    Parameters
    ----------
    y_true : ndarray
        Realised values.
    lower, upper : ndarray
        Interval bounds, broadcastable to ``y_true``.
    alpha : float
        Miscoverage level in ``(0, 1)`` (a 90% interval has ``alpha = 0.1``).

    Returns
    -------
    ndarray
        Per-observation score (lower is better).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    below = np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0)
    above = np.where(y > hi, (2.0 / alpha) * (y - hi), 0.0)
    return width + below + above


def pit_values(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Probability Integral Transform values from an ensemble forecast.

    ``u_t = F_t(y_t)`` estimated as the fraction of predictive draws at or below
    the realisation. If the forecasts are calibrated the ``u_t`` are uniform on
    ``[0, 1]``; systematic deviations diagnose bias (shifted histogram) or
    over/under-dispersion (U-shaped / hump-shaped histogram).

    Parameters
    ----------
    y_true : ndarray of shape (T,)
        Realised values.
    samples : ndarray of shape (T, M)
        Predictive draws.

    Returns
    -------
    ndarray of shape (T,)
        PIT values in ``[0, 1]``.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"`samples` must have one row per observation ({y.shape[0]}), got "
            f"{s.shape[0]}."
        )
    return (s <= y[:, None]).mean(axis=1)


def pit_histogram(pit: np.ndarray, *, n_bins: int = 10) -> pl.DataFrame:
    """Reliability table for PIT values.

    Parameters
    ----------
    pit : ndarray
        PIT values in ``[0, 1]`` (see :func:`pit_values`).
    n_bins : int, default=10
        Number of equal-width bins.

    Returns
    -------
    polars.DataFrame
        Columns ``bin_left``, ``bin_right``, ``count``, ``frequency`` and
        ``expected`` (``1 / n_bins``). A calibrated forecast has
        ``frequency ~ expected`` in every bin.
    """
    u = np.asarray(pit, dtype=float).ravel()
    u = u[np.isfinite(u)]
    if u.size == 0:
        raise ValueError("`pit` contains no finite values.")
    if n_bins < 1:
        raise ValueError(f"`n_bins` must be >= 1, got {n_bins}.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(np.clip(u, 0.0, 1.0), bins=edges)
    return pl.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts.astype(np.int64),
            "frequency": counts / u.size,
            "expected": np.full(n_bins, 1.0 / n_bins),
        }
    )


def score_quantile_forecasts(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    y_true: str,
    quantile_cols: Sequence[str],
    levels: Sequence[float],
    by: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    """Score a panel of quantile forecasts entirely in Polars.

    One pass over the frame produces, per group: the mean pinball loss at each
    level, the quantile-approximated CRPS, and the empirical coverage of every
    quantile (the fraction of realisations at or below it, which should equal
    the nominal level).

    Parameters
    ----------
    frame : polars.DataFrame | polars.LazyFrame
        Long-format frame holding the realisations and the predicted quantiles.
    y_true : str
        Column with the realised values.
    quantile_cols : sequence of str
        One column per predicted quantile, aligned with ``levels``.
    levels : sequence of float
        The nominal quantile levels, each in ``(0, 1)``.
    by : str | sequence of str, optional
        Grouping keys (e.g. the entity column). ``None`` scores the whole frame.

    Returns
    -------
    polars.DataFrame
        Columns: the grouping keys, ``pinball_<level>`` and ``coverage_<level>``
        per level, plus ``crps``.

    Raises
    ------
    ValueError
        If ``quantile_cols`` and ``levels`` differ in length.
    """
    cols = list(quantile_cols)
    taus = [float(t) for t in levels]
    if len(cols) != len(taus):
        raise ValueError(
            f"`quantile_cols` has {len(cols)} entries but {len(taus)} levels "
            "were given."
        )
    lf = frame.lazy()
    aggs: list[pl.Expr] = []
    pin_names: list[str] = []
    for col, tau in zip(cols, taus, strict=True):
        name = f"pinball_{tau:g}"
        pin_names.append(name)
        aggs.append(pinball_loss_expr(y_true, col, tau).mean().alias(name))
        aggs.append((pl.col(y_true) <= pl.col(col)).mean().alias(f"coverage_{tau:g}"))
    keys = [by] if isinstance(by, str) else (list(by) if by is not None else [])
    out = lf.group_by(keys).agg(aggs) if keys else lf.select(aggs)
    crps = 2.0 * pl.mean_horizontal([pl.col(n) for n in pin_names])
    result = out.with_columns(crps.alias("crps")).collect()
    return result.sort(keys) if keys else result
