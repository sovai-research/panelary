"""Honest selection statistics: DSR/PSR, PBO, Romano-Wolf, BH/BY.

When many strategies, feature sets or hyper-parameter settings are compared on
the same data, the winner's performance is contaminated by the selection itself.
This module is the correction layer:

* **Probabilistic / Deflated Sharpe** (Bailey & Lopez de Prado, 2014) —
  :func:`probabilistic_sharpe_ratio`, :func:`expected_maximum_sharpe`,
  :func:`deflated_sharpe_ratio`: how likely is the observed Sharpe to be real
  once the number of trials, the sample length and the return distribution's
  skew/kurtosis are accounted for?
* **PBO / CSCV** (Bailey, Borwein, Lopez de Prado & Zhu, 2017) —
  :func:`probability_of_backtest_overfitting`: how often does the in-sample
  winner land in the bottom half out-of-sample?
* **Family-wise error control** — :func:`romano_wolf`, the studentized stepdown
  of Romano & Wolf (2005), which is *strictly* more powerful than Holm because
  it learns the dependence between candidates from a bootstrap instead of
  assuming the worst case. :func:`holm_bonferroni` is kept as the assumption-free
  reference.
* **False-discovery control** — :func:`benjamini_hochberg` (valid under
  independence or positive regression dependence) and
  :func:`benjamini_yekutieli` (valid under *arbitrary* dependence, which is the
  realistic assumption for a cross-correlated panel of features).

``deflated_sharpe_ratio`` and ``probability_of_backtest_overfitting`` live in
:mod:`panelary.core.model_selection` (where the CV runner consumes them)
and are re-exported here so the whole selection surface has one import site.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from panelary.core.model_selection import (
    _norm_cdf,
    _norm_ppf,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from panelary.validation._bootstrap import block_bootstrap_indices

__all__ = [
    "MultipleTestResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "holm_bonferroni",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "romano_wolf",
    "romano_wolf_mean_test",
]

#: Euler-Mascheroni constant, the ``gamma`` of the expected-maximum formula.
_EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Sharpe-ratio inference
# --------------------------------------------------------------------------- #
def _sharpe_estimator_std(
    sharpe: float, n_observations: int, skewness: float, kurtosis: float
) -> float:
    """Standard error of a Sharpe estimator under non-normal returns.

    ``sigma(SR_hat) = sqrt((1 - g3 SR + (g4 - 1)/4 SR^2) / (T - 1))`` where
    ``g3`` is skewness and ``g4`` is (non-excess) kurtosis, so a Gaussian series
    gives the familiar ``sqrt((1 + SR^2 / 2) / (T - 1))``.
    """
    if n_observations < 2:
        raise ValueError("`n_observations` must be >= 2.")
    var = (1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe) / (
        n_observations - 1
    )
    if var <= 0:
        raise ValueError(
            "computed Sharpe-estimator variance is non-positive; check the "
            "`skewness`/`kurtosis`/`sharpe` inputs."
        )
    return math.sqrt(var)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR), Bailey & Lopez de Prado (2012).

    The probability that the *true* Sharpe exceeds ``benchmark_sharpe``, given a
    finite sample of possibly skewed, fat-tailed returns:

    .. math::

        \\widehat{PSR}(SR^*) = \\Phi\\!\\left(
        \\frac{(\\widehat{SR} - SR^*)\\sqrt{T-1}}
             {\\sqrt{1 - \\gamma_3\\widehat{SR} + \\frac{\\gamma_4-1}{4}
              \\widehat{SR}^2}}\\right)

    The Deflated Sharpe Ratio is exactly this quantity evaluated at the
    *selection-adjusted* threshold :func:`expected_maximum_sharpe`.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe **per observation** (do not annualise).
    n_observations : int
        Sample length ``T``.
    benchmark_sharpe : float, default=0.0
        The threshold ``SR*`` to beat.
    skewness : float, default=0.0
        Skewness of the returns.
    kurtosis : float, default=3.0
        Non-excess kurtosis (3.0 = normal).

    Returns
    -------
    float
        Probability in ``[0, 1]``.

    Examples
    --------
    >>> round(probabilistic_sharpe_ratio(0.1, n_observations=1001), 6)
    0.999196
    """
    std = _sharpe_estimator_std(
        float(observed_sharpe), int(n_observations), float(skewness), float(kurtosis)
    )
    return float(_norm_cdf((float(observed_sharpe) - float(benchmark_sharpe)) / std))


def expected_maximum_sharpe(
    n_trials: int, sharpe_variance_across_trials: float
) -> float:
    """Expected maximum Sharpe across ``n_trials`` zero-skill trials.

    Bailey & Lopez de Prado's Gumbel approximation to the expected maximum of
    ``N`` independent draws with variance ``V``:

    .. math::

        SR_0 = \\sqrt{V}\\left[(1-\\gamma)\\,\\Phi^{-1}\\!\\left(1-\\tfrac1N\\right)
        + \\gamma\\,\\Phi^{-1}\\!\\left(1-\\tfrac{1}{N e}\\right)\\right]

    with ``gamma`` the Euler-Mascheroni constant. This is the threshold the
    Deflated Sharpe Ratio deflates against: **N and V must be paired honestly**
    — ``N`` is the number of configurations actually searched and ``V`` is the
    variance of the Sharpes *across those configurations*, not the sampling
    variance of a single strategy's Sharpe estimator.

    Parameters
    ----------
    n_trials : int
        Number ``N`` of trials (``>= 1``). ``N = 1`` gives ``0.0``.
    sharpe_variance_across_trials : float
        Variance ``V`` of the trial Sharpes (``>= 0``), per observation.

    Returns
    -------
    float
        The expected maximum Sharpe under the null of no skill.

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``V < 0``.

    Examples
    --------
    >>> round(expected_maximum_sharpe(100, 0.01), 8)
    0.25306029
    """
    n = int(n_trials)
    v = float(sharpe_variance_across_trials)
    if n < 1:
        raise ValueError(f"`n_trials` must be >= 1, got {n}.")
    if v < 0:
        raise ValueError(f"`sharpe_variance_across_trials` must be >= 0, got {v}.")
    if n == 1 or v == 0.0:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(v) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def minimum_track_record_length(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Minimum Track Record Length (MinTRL), Bailey & Lopez de Prado (2012).

    The number of observations needed before the PSR would reach ``confidence``:

    .. math::

        MinTRL = 1 + \\left(1 - \\gamma_3 \\widehat{SR}
        + \\tfrac{\\gamma_4-1}{4}\\widehat{SR}^2\\right)
        \\left(\\frac{\\Phi^{-1}(c)}{\\widehat{SR} - SR^*}\\right)^2

    Parameters
    ----------
    observed_sharpe : float
        Observed per-observation Sharpe. Must exceed ``benchmark_sharpe``.
    benchmark_sharpe : float, default=0.0
        Threshold Sharpe.
    confidence : float, default=0.95
        Target confidence level in ``(0, 1)``.
    skewness, kurtosis : float
        Return-distribution moments (kurtosis non-excess).

    Returns
    -------
    float
        The required number of observations.

    Raises
    ------
    ValueError
        If ``observed_sharpe <= benchmark_sharpe`` or ``confidence`` is invalid.
    """
    sr = float(observed_sharpe)
    b = float(benchmark_sharpe)
    if sr <= b:
        raise ValueError(
            "`observed_sharpe` must exceed `benchmark_sharpe` for a finite "
            f"track-record length (got {sr} <= {b})."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"`confidence` must be in (0, 1), got {confidence}.")
    num = 1.0 - float(skewness) * sr + (float(kurtosis) - 1.0) / 4.0 * sr * sr
    return float(1.0 + num * (_norm_ppf(confidence) / (sr - b)) ** 2)


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MultipleTestResult:
    """Outcome of a multiple-testing correction.

    Attributes
    ----------
    rejected : numpy.ndarray of bool
        Which hypotheses are rejected at the requested level.
    adjusted_pvalues : numpy.ndarray of float
        Multiplicity-adjusted p-values, monotone in the original ordering of the
        statistics, and directly comparable to ``alpha``.
    alpha : float
        The level the rejections were taken at.
    method : str
        Name of the correction.
    """

    rejected: np.ndarray
    adjusted_pvalues: np.ndarray
    alpha: float
    method: str

    @property
    def n_rejected(self) -> int:
        """Number of rejected hypotheses."""
        return int(np.count_nonzero(self.rejected))


def _check_pvalues(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("`pvalues` must not be empty.")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("`pvalues` must all be finite and in [0, 1].")
    return p


def holm_bonferroni(
    pvalues: Sequence[float] | np.ndarray, *, alpha: float = 0.05
) -> MultipleTestResult:
    """Holm (1979) stepdown control of the family-wise error rate.

    Assumption-free (valid under any dependence) and therefore conservative;
    :func:`romano_wolf` dominates it when a bootstrap null is available.

    Parameters
    ----------
    pvalues : array-like of float
        Raw p-values, one per hypothesis.
    alpha : float, default=0.05
        Target FWER.

    Returns
    -------
    MultipleTestResult
    """
    p = _check_pvalues(pvalues)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return MultipleTestResult(adj <= alpha, adj, float(alpha), "holm-bonferroni")


def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray, *, alpha: float = 0.05
) -> MultipleTestResult:
    """Benjamini-Hochberg (1995) false-discovery-rate control.

    Controls the FDR at ``alpha`` under independence or positive regression
    dependence (PRDS). For a cross-correlated panel where the sign of the
    dependence is unknown, prefer :func:`benjamini_yekutieli`.

    Parameters
    ----------
    pvalues : array-like of float
        Raw p-values.
    alpha : float, default=0.05
        Target FDR.

    Returns
    -------
    MultipleTestResult

    Examples
    --------
    >>> res = benjamini_hochberg([0.001, 0.02, 0.04, 0.6], alpha=0.05)
    >>> res.n_rejected
    2
    """
    return _bh_family(pvalues, alpha=alpha, c_m=1.0, method="benjamini-hochberg")


def benjamini_yekutieli(
    pvalues: Sequence[float] | np.ndarray, *, alpha: float = 0.05
) -> MultipleTestResult:
    """Benjamini-Yekutieli (2001) FDR control under arbitrary dependence.

    Identical to :func:`benjamini_hochberg` but with the harmonic penalty
    ``c(m) = sum_{i=1..m} 1/i``, which makes the procedure valid for **any**
    dependence structure — the right default for panels of features that are
    correlated in unknown ways across entities and time.

    Parameters
    ----------
    pvalues : array-like of float
        Raw p-values.
    alpha : float, default=0.05
        Target FDR.

    Returns
    -------
    MultipleTestResult
    """
    m = _check_pvalues(pvalues).size
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    return _bh_family(pvalues, alpha=alpha, c_m=c_m, method="benjamini-yekutieli")


def _bh_family(
    pvalues: Sequence[float] | np.ndarray, *, alpha: float, c_m: float, method: str
) -> MultipleTestResult:
    p = _check_pvalues(pvalues)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=float)
    scaled = p[order] * m * c_m / ranks
    # Step-up: enforce monotonicity from the largest p-value downwards.
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return MultipleTestResult(adj <= alpha, adj, float(alpha), method)


def romano_wolf(
    statistics: Sequence[float] | np.ndarray,
    null_distribution: Sequence[Sequence[float]] | np.ndarray,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> MultipleTestResult:
    """Romano-Wolf (2005) stepdown control of the family-wise error rate.

    The max-T stepdown: at each step the critical value is the bootstrap
    distribution of the **maximum** statistic over the hypotheses *not yet
    rejected*. Because the maximum is taken over the resampled joint
    distribution, the procedure automatically exploits the correlation between
    candidates — on a panel of highly correlated features it rejects far more
    than Holm/Bonferroni while still controlling the FWER.

    Parameters
    ----------
    statistics : array-like of shape (S,)
        Observed test statistics, one per hypothesis. Should be *studentized*
        (t-like); the procedure is only asymptotically balanced if they are.
    null_distribution : array-like of shape (B, S)
        ``B`` resampled statistics per hypothesis, **centred at the null** (e.g.
        ``(mean_b - mean_obs) / se_b`` from a stationary bootstrap). See
        :func:`romano_wolf_mean_test` for the standard construction.
    alpha : float, default=0.05
        Target FWER.
    two_sided : bool, default=True
        Compare absolute values (two-sided) or raw values (one-sided, "greater").

    Returns
    -------
    MultipleTestResult
        ``adjusted_pvalues`` are the stepdown-adjusted p-values
        ``(1 + #{max_b >= t}) / (B + 1)``, made monotone along the stepdown
        order so they can be thresholded at any level.

    Raises
    ------
    ValueError
        If the shapes disagree or ``alpha`` is not in ``(0, 1)``.

    References
    ----------
    Romano, J. P., & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized
    Data Snooping." *Econometrica*, 73(4), 1237-1282.
    """
    t = np.asarray(statistics, dtype=float).ravel()
    boot = np.asarray(null_distribution, dtype=float)
    if boot.ndim != 2:
        raise ValueError(
            f"`null_distribution` must be 2-D (B x S), got shape {boot.shape}."
        )
    if boot.shape[1] != t.size:
        raise ValueError(
            f"`null_distribution` has {boot.shape[1]} columns but there are "
            f"{t.size} statistics."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")

    if two_sided:
        t_use = np.abs(t)
        boot_use = np.abs(boot)
    else:
        t_use = t
        boot_use = boot

    n_boot, n_hyp = boot_use.shape
    order = np.argsort(-t_use)  # most significant first
    # Step j's critical distribution is the row-wise max over the hypotheses not
    # yet rejected, order[j:], i.e. a *suffix* max of the columns in stepdown
    # order: one reversed maximum.accumulate gives every step at once, O(B S)
    # instead of the O(B S^2) loop (bitwise identical: max and counts are exact).
    sorted_boot = boot_use[:, order]
    suffix_max = np.maximum.accumulate(sorted_boot[:, ::-1], axis=1)[:, ::-1]
    exceed = np.count_nonzero(suffix_max >= t_use[order][None, :], axis=0)
    p_raw = (1.0 + exceed.astype(float)) / (n_boot + 1.0)
    adj = np.empty(n_hyp, dtype=float)
    adj[order] = np.maximum.accumulate(p_raw)
    return MultipleTestResult(adj <= alpha, adj, float(alpha), "romano-wolf")


def romano_wolf_mean_test(
    x: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 999,
    block_length: int | None = None,
    boundaries: Sequence[int] | None = None,
    two_sided: bool = True,
    seed: int | np.random.Generator | None = None,
) -> MultipleTestResult:
    """Romano-Wolf stepdown for ``H0_j: E[x_j] = 0`` on a panel of series.

    The workhorse form: given a ``(T, S)`` matrix of per-period values (strategy
    returns, per-feature information coefficients, per-model loss differentials),
    test all ``S`` hypotheses jointly at family-wise level ``alpha``. The
    null distribution is a **stationary bootstrap** of the studentized means,
    recentred at the observed means, so serial dependence *and* cross-sectional
    correlation are respected.

    Parameters
    ----------
    x : ndarray of shape (T, S)
        Per-period values, one column per hypothesis.
    alpha : float, default=0.05
        Target FWER.
    n_boot : int, default=999
        Bootstrap replications ``B``.
    block_length : int, optional
        Mean block length of the stationary bootstrap. Defaults to
        ``max(2, round(T ** (1/3)))``.
    boundaries : sequence of int, optional
        Fold boundaries the resampled blocks must not straddle.
    two_sided : bool, default=True
        Two-sided or "greater-than" alternative.
    seed : int | numpy.random.Generator, optional
        Seed.

    Returns
    -------
    MultipleTestResult

    Raises
    ------
    ValueError
        If ``x`` is not a 2-D matrix with at least 2 rows.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"`x` must be a (T, S) matrix with T >= 2, got {arr.shape}.")
    n_obs, _ = arr.shape
    if block_length is None:
        block_length = max(2, int(round(n_obs ** (1.0 / 3.0))))

    def _studentized(sample: np.ndarray) -> np.ndarray:
        mean = sample.mean(axis=0)
        se = sample.std(axis=0, ddof=1) / math.sqrt(sample.shape[0])
        se = np.where(se <= 0, np.inf, se)
        return mean / se

    t_obs = _studentized(arr)
    idx = block_bootstrap_indices(
        n_obs,
        block_length=block_length,
        n_boot=n_boot,
        scheme="stationary",
        boundaries=boundaries,
        seed=seed,
    )
    obs_mean = arr.mean(axis=0)
    boot = np.empty((n_boot, arr.shape[1]), dtype=float)
    for b in range(n_boot):
        sample = arr[idx[b]]
        mean = sample.mean(axis=0)
        se = sample.std(axis=0, ddof=1) / math.sqrt(n_obs)
        se = np.where(se <= 0, np.inf, se)
        boot[b] = (mean - obs_mean) / se  # recentred at the null
    return romano_wolf(t_obs, boot, alpha=alpha, two_sided=two_sided)
