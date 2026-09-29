"""Honest reporting for mined alphas: trial accounting, deflation, causality.

This module exists because of a single, reproducible failure mode:
**search-count amnesia**. A genetic / symbolic search evaluates :math:`M`
candidates, keeps the best one, and reports its in-sample score as if it had
been the only thing ever tried. Every surveyed alpha-mining repository does
this. AlphaGen is the clearest case: it maintains ``pool.eval_cnt``, increments
it on every evaluation, and the *only* consumer of that counter is a
TensorBoard logging call. The trial count is measured and then thrown away. Not
one paper in the surveyed literature reports :math:`N`. Without :math:`N` you
cannot deflate anything.

The stakes, measured with Panelary's own
:func:`~panelary.validation.expected_maximum_sharpe`: a 100,000-trial
search on **pure noise** is expected to produce a best in-sample Sharpe of
**4.39**; at :math:`N = 10` it is already **1.57**. A raw best-of-search score is
therefore not a weak statistic, it is a fabrication.

What is here
------------
* :class:`TrialLedger` — records every evaluated candidate and computes
  :math:`M`, :math:`V[\\{SR_n\\}]`, :math:`\\bar\\rho` and the *implied number of
  independent trials* :math:`\\hat N`, then feeds those into Panelary's existing
  :func:`~panelary.core.model_selection.deflated_sharpe_ratio` and
  :func:`~panelary.core.model_selection.probability_of_backtest_overfitting`.
* :func:`minimum_backtest_length` — the pre-flight budget constraint: how much
  history you need before :math:`N` trials stop guaranteeing a false positive.
* :func:`assert_causal` — prefix invariance made executable; the look-ahead
  detector.
* :func:`haircut_sharpe_ratio` — Harvey & Liu's multiple-testing haircut.
* :func:`cross_sectional_bootstrap` — the Yan-Zheng joint-date bootstrap, the
  one test a mechanically-mined signal can legitimately *pass*.
* :func:`search_diagnostics` — the in-sample vs held-out scatter over **all**
  candidates, which tells you whether the search found anything at all.

Honest about the limits of this honesty tooling
-----------------------------------------------
The Deflated Sharpe Ratio is reported here as **a conservative screen with
unquantified type-I error control, not a p-value**. Steven Pav (author of R's
``SharpeR``) makes the case in *"Pseudo-Statistics and Quantitative
Charlatanism"* (https://sharperat.io/do-not-deflate-I.html, 2026-07-11): the DSR
paper states no null hypothesis and offers no proof that the statistic is
uniform under one, and the maximum of :math:`N` i.i.d. Gaussians is Gumbel, not
Gaussian, so treating :math:`\\Phi` of a centred maximum as a probability is an
approximation of unknown quality. His :math:`10^6`-replication simulations find
DSR is **conservative rather than anti-conservative** — a nominal 0.10 test
rejects at 0.042 unconditionally, and roughly 1 in 3,361 conditionally.

Practically: a low DSR is strong evidence *against* a candidate; a high DSR is
weak evidence *for* one, and is certainly not "the probability the strategy is
real". The PBO gate is the sharper instrument, and Bailey, Borwein, Lopez de
Prado & Zhu state the rule plainly — *"reject models for which PBO is estimated
to be greater than 0.05"* — which is what :meth:`TrialLedger.summary` enforces.

References
----------
Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
*Journal of Portfolio Management*, 40(5), 94-107. SSRN 2460551 — Appendix 3
gives the effective-independent-trials reduction implemented in
:meth:`TrialLedger.implied_independent_trials`.

Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014).
"Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
Overfitting on Out-of-Sample Performance." *Notices of the AMS*, 61(5), 458-471
— Theorem 3.1, the minimum backtest length.

Harvey, C. R., & Liu, Y. (2015). "Backtesting." *Journal of Portfolio
Management*, 42(1), 13-28.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). "... and the Cross-Section of Expected
Returns." *Review of Financial Studies*, 29(1), 5-68.

Yan, X., & Zheng, L. (2017). "Fundamental Analysis and the Cross-Section of
Stock Returns: A Data-Mining Approach." *Review of Financial Studies*, 30(4),
1382-1423.

Lopez de Prado, M. (2019). "A Data Science Solution to the Multi-Testing
Problem." *Journal of Financial Data Science*, 1(1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from panelary.core.model_selection import (
    _norm_cdf,
    _norm_ppf,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from panelary.validation._bootstrap import block_bootstrap_indices
from panelary.validation._selection_stats import (
    benjamini_yekutieli,
    expected_maximum_sharpe,
    holm_bonferroni,
)

__all__ = [
    "TrialLedger",
    "assert_causal",
    "cross_sectional_bootstrap",
    "haircut_sharpe_ratio",
    "minimum_backtest_length",
    "search_diagnostics",
]

#: Euler-Mascheroni constant; the ``gamma`` of the expected-maximum formula.
_EULER_GAMMA = 0.5772156649015329

#: PBO above this is a rejection. Bailey, Borwein, Lopez de Prado & Zhu (2017),
#: verbatim: "reject models for which PBO is estimated to be greater than 0.05".
PBO_REJECT_ABOVE: float = 0.05

#: DSR below this fails the screen. Note the module docstring: this is a
#: conservative screen, not a calibrated p-value.
DSR_ACCEPT_ABOVE: float = 0.95


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _sharpe_of(series: np.ndarray, ddof: int = 1) -> float:
    """Per-observation Sharpe of a return series (mean / std)."""
    x = _finite(np.asarray(series, dtype=np.float64).ravel())
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=ddof))
    if sd <= 0.0:
        return float("nan")
    return float(np.mean(x) / sd)


def _moments(series: np.ndarray) -> tuple[float, float]:
    """(skewness, **raw** kurtosis) of a series; (0.0, 3.0) if degenerate.

    Raw, *not* excess, kurtosis — that is what
    :func:`~panelary.core.model_selection.deflated_sharpe_ratio` wants
    (its default is 3.0 = Gaussian). Passing excess kurtosis here is a
    documented bug in ``vectorbt`` and ``quantstats``; Panelary does not repeat
    it.
    """
    x = _finite(np.asarray(series, dtype=np.float64).ravel())
    if x.size < 4:
        return 0.0, 3.0
    xc = x - x.mean()
    m2 = float(np.mean(xc**2))
    if m2 <= 0.0:
        return 0.0, 3.0
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    return m3 / m2**1.5, m4 / m2**2


def _null_second_moment(values: np.ndarray | None) -> tuple[float | None, int]:
    """Null variance from zero-skill decoy statistics: ``(E[x^2], k)``.

    The second moment about **zero**, not the sample variance: a zero-skill
    statistic has mean zero by construction, so this spends no degree of
    freedom on estimating it, and any bias the scoring path adds to the null
    is charged to the variance rather than silently absorbed. ``None`` when
    fewer than two finite values were given.
    """
    if values is None:
        return None, 0
    x = _finite(np.asarray(values, dtype=np.float64).ravel())
    if x.size < 2:
        return None, int(x.size)
    return float(np.mean(x * x)), int(x.size)


# --------------------------------------------------------------------------- #
# TrialLedger
# --------------------------------------------------------------------------- #
class TrialLedger:
    """Record every candidate a search evaluates, so it can be deflated.

    The ledger is the antidote to search-count amnesia. Give it one
    :meth:`record` call per *evaluated* candidate — not per survivor, not per
    generation, per **evaluation** — and it maintains everything the deflation
    machinery needs:

    ``M``
        the exact number of trials.
    ``V[{SR_n}]``
        the variance of the scores *across trials*. This is the term everyone
        omits and the one the Deflated Sharpe Ratio actually needs: the DSR
        deflates against ``sqrt(V) * f(N)``, so with ``V`` missing the whole
        correction is missing.
    ``rho_bar``
        the average pairwise correlation between candidate return streams.
    ``N_hat``
        the *implied number of independent trials*, see
        :meth:`implied_independent_trials`.

    Memory discipline
    -----------------
    Storing full per-period series for :math:`10^5` candidates is not
    affordable. The ledger therefore keeps:

    * **exact** running moments (Welford) and exact scalars — score, held-out
      score, complexity, generation — for *every* trial, ~32 bytes each;
    * a **seeded reservoir sample** (Algorithm R, default 2000 candidates) of
      the full per-period series, used for the correlation matrix, PBO and the
      PCA cross-check;
    * the best candidate's series unconditionally, so the headline statistic is
      never an approximation.

    So ``M``, ``V`` and the best score are exact; ``rho_bar``, ``N_hat`` and
    ``PBO`` are estimated from a uniform random sample of the search. With the
    default reservoir the standard error of ``rho_bar`` is negligible relative
    to the order-of-magnitude precision ``N_hat`` is used at.

    Parameters
    ----------
    reservoir_size : int, default=2000
        Number of per-period series retained for correlation/PBO work.
    seed : int, default=0
        Seed for the reservoir sampler. Determinism is a hard invariant.
    series_kind : {"returns", "cumulative"}, default="returns"
        Whether :meth:`record` is given per-period values (returns, per-period
        ICs) or a cumulative PnL curve. Correlations are **always** computed on
        first differences: correlating cumulative curves manufactures
        correlation out of the common trend, which would collapse ``N_hat``
        toward 1 and make the deflation vanish. This is WorldQuant's stated
        convention and it is not optional.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ledger = TrialLedger(seed=0)
    >>> for i in range(50):
    ...     s = rng.standard_normal(240) * 0.01
    ...     ledger.record(i, _sharpe_of(s), s)
    >>> ledger.n_trials
    50
    """

    __slots__ = (
        "_best_generation",
        "_best_key",
        "_best_score",
        "_best_series",
        "_complexity",
        "_generation",
        "_held_out",
        "_key",
        "_m2",
        "_mean",
        "_n",
        "_n_series_seen",
        "_reservoir",
        "_reservoir_keys",
        "_reservoir_scores",
        "_rng",
        "_scores",
        "_series_len",
        "reservoir_size",
        "seed",
        "series_kind",
    )

    def __init__(
        self,
        *,
        reservoir_size: int = 2000,
        seed: int = 0,
        series_kind: Literal["returns", "cumulative"] = "returns",
    ) -> None:
        if reservoir_size < 2:
            raise ValueError(f"`reservoir_size` must be >= 2, got {reservoir_size}.")
        if series_kind not in {"returns", "cumulative"}:
            raise ValueError(
                f"`series_kind` must be 'returns' or 'cumulative', got {series_kind!r}."
            )
        self.reservoir_size = int(reservoir_size)
        self.seed = int(seed)
        self.series_kind = series_kind
        self._rng = np.random.default_rng(seed)

        # exact running moments (Welford)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

        # exact per-trial scalars (cheap: ~32 B/trial)
        self._key: list[int] = []
        self._scores: list[float] = []
        self._held_out: list[float] = []
        self._complexity: list[int] = []
        self._generation: list[int] = []

        # best-so-far, kept exactly
        self._best_score = -math.inf
        self._best_key: int | None = None
        self._best_series: np.ndarray | None = None
        self._best_generation = -1

        # reservoir of per-period series
        self._reservoir: np.ndarray | None = None
        self._reservoir_keys: list[int] = []
        self._reservoir_scores: list[float] = []
        self._n_series_seen = 0
        self._series_len: int | None = None

    # -- recording --------------------------------------------------------- #
    def record(
        self,
        key: int,
        score: float,
        series: np.ndarray | None = None,
        *,
        held_out: float | None = None,
        complexity: int = 0,
        generation: int = 0,
    ) -> None:
        """Record one evaluated candidate.

        Parameters
        ----------
        key : int
            Genome / semantic key identifying the candidate. Not deduplicated:
            re-evaluating the same expression still consumes a trial, because
            it still had a chance to win.
        score : float
            The selection score. **Record it on the same scale as the statistic
            you intend to deflate** — a per-observation Sharpe if you will call
            :meth:`deflated_sharpe`, since ``V`` is the variance of *these*
            numbers and the DSR threshold is ``sqrt(V) * f(N_hat)``.
        series : ndarray, optional
            Per-period returns or ICs for this candidate (or a cumulative PnL
            curve if ``series_kind="cumulative"``). All series must share one
            length. Without series the ledger can still report ``M`` and ``V``
            but not ``rho_bar``, ``N_hat`` or PBO.
        held_out : float, optional
            The same candidate's score on data not used for selection. Enables
            :meth:`diagnostics`.
        complexity : int, default=0
            Active node count, for reporting.
        generation : int, default=0
            Generation index, for reporting.

        Raises
        ------
        ValueError
            If ``score`` is not finite, or ``series`` has a length different
            from the first series recorded.
        """
        s = float(score)
        if not math.isfinite(s):
            raise ValueError(
                f"`score` must be finite, got {score!r}. Map failed evaluations "
                "to a finite sentinel (e.g. 0.0) — silently dropping them "
                "re-introduces search-count amnesia."
            )
        self._n += 1
        delta = s - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (s - self._mean)

        self._key.append(int(key))
        self._scores.append(s)
        self._held_out.append(float("nan") if held_out is None else float(held_out))
        self._complexity.append(int(complexity))
        self._generation.append(int(generation))

        arr: np.ndarray | None = None
        if series is not None:
            arr = np.asarray(series, dtype=np.float64).ravel()
            if self._series_len is None:
                self._series_len = int(arr.size)
                self._reservoir = np.empty(
                    (self.reservoir_size, self._series_len), dtype=np.float64
                )
            elif arr.size != self._series_len:
                raise ValueError(
                    f"all recorded series must have the same length; expected "
                    f"{self._series_len}, got {arr.size}."
                )
            self._reservoir_add(int(key), s, arr)

        if s > self._best_score:
            self._best_score = s
            self._best_key = int(key)
            self._best_generation = int(generation)
            if arr is not None:
                self._best_series = arr.copy()

    def _reservoir_add(self, key: int, score: float, arr: np.ndarray) -> None:
        """Vitter's Algorithm R: a uniform sample of the series seen so far."""
        assert self._reservoir is not None  # set by the caller
        self._n_series_seen += 1
        filled = len(self._reservoir_keys)
        if filled < self.reservoir_size:
            self._reservoir[filled] = arr
            self._reservoir_keys.append(key)
            self._reservoir_scores.append(score)
            return
        j = int(self._rng.integers(0, self._n_series_seen))
        if j < self.reservoir_size:
            self._reservoir[j] = arr
            self._reservoir_keys[j] = key
            self._reservoir_scores[j] = score

    # -- exact scalars ------------------------------------------------------ #
    def __len__(self) -> int:
        return self._n

    @property
    def n_trials(self) -> int:
        """``M``: the exact number of evaluations recorded."""
        return self._n

    @property
    def score_variance(self) -> float:
        """``V[{SR_n}]``: exact sample variance of the scores across trials."""
        if self._n < 2:
            return 0.0
        return float(self._m2 / (self._n - 1))

    @property
    def best_score(self) -> float:
        """Best raw in-sample score. On its own, not a reportable number."""
        return float(self._best_score) if self._n else float("nan")

    @property
    def best_key(self) -> int | None:
        """Key of the best-scoring candidate."""
        return self._best_key

    @property
    def best_series(self) -> np.ndarray | None:
        """Per-period series of the best candidate (kept unconditionally)."""
        return self._best_series

    def scores(self) -> np.ndarray:
        """All recorded scores, in evaluation order."""
        return np.asarray(self._scores, dtype=np.float64)

    def held_out(self) -> np.ndarray:
        """All recorded held-out scores (NaN where none was supplied)."""
        return np.asarray(self._held_out, dtype=np.float64)

    # -- correlation structure --------------------------------------------- #
    def _reservoir_matrix(self) -> np.ndarray | None:
        """Reservoir series as ``(n_kept, T')``, differenced if cumulative."""
        if self._reservoir is None or len(self._reservoir_keys) < 2:
            return None
        mat = self._reservoir[: len(self._reservoir_keys)]
        if self.series_kind == "cumulative":
            mat = np.diff(mat, axis=1)
        if mat.shape[1] < 3:
            return None
        return np.ascontiguousarray(mat, dtype=np.float64)

    def correlation_matrix(self) -> np.ndarray | None:
        """Pairwise correlation of the sampled candidates' **period returns**.

        Returns ``None`` if fewer than two series were recorded. Rows/columns
        for degenerate (zero-variance) candidates are ``NaN`` and are excluded
        from every downstream average.
        """
        mat = self._reservoir_matrix()
        if mat is None:
            return None
        sd = mat.std(axis=1)
        good = sd > 0.0
        corr = np.full((mat.shape[0], mat.shape[0]), np.nan, dtype=np.float64)
        if int(np.count_nonzero(good)) < 2:
            return corr
        sub = np.corrcoef(mat[good])
        ix = np.flatnonzero(good)
        corr[np.ix_(ix, ix)] = sub
        return corr

    def mean_pairwise_correlation(self) -> float:
        """``rho_bar``, Bailey & Lopez de Prado (2014) Appendix 3, Eq. 8.

        .. math::

            \\bar\\rho = \\frac{\\sum_i \\sum_{j \\neq i} \\rho_{i,j}}{M(M-1)}

        Estimated from the reservoir sample and computed on **first differences
        of PnL**, never on cumulative curves. Returns ``NaN`` when no series
        were recorded.
        """
        corr = self.correlation_matrix()
        if corr is None:
            return float("nan")
        m = corr.shape[0]
        if m < 2:
            return float("nan")
        off = ~np.eye(m, dtype=bool)
        vals = corr[off]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        return float(vals.mean())

    def implied_independent_trials(self) -> float:
        """``N_hat``: how many *independent* trials the search was worth.

        This is the single most important number in the module. A genetic
        search evaluates :math:`M` candidates, but they are not :math:`M`
        independent bets — a population is a cloud of near-duplicates, and the
        expected maximum of :math:`M` *correlated* draws is far below that of
        :math:`M` independent ones. Bailey & Lopez de Prado (SSRN 2460551,
        Appendix 3) give the reduction:

        .. math::

            \\bar\\rho = \\frac{\\sum_i\\sum_{j\\neq i}\\rho_{i,j}}{M(M-1)}
            \\qquad\\text{(Eq. 8)}

        .. math::

            \\hat N = \\bar\\rho + (1 - \\bar\\rho)\\,M \\qquad\\text{(Eq. 9)}

        Why it matters: with 50,000 candidates at :math:`\\bar\\rho \\approx
        0.9`, passing ``n_trials = M`` makes the DSR absurd (it deflates against
        the maximum of 50,000 independent draws that were never taken) and
        passing ``n_trials = 1`` is simply a lie. Eq. 9 gives ~5,001, which is
        the defensible number. No Python package implements this.

        Returns
        -------
        float
            ``N_hat``, clipped to ``[1, M]``. Falls back to the conservative
            ``float(M)`` when no per-period series were recorded, since without
            series there is no evidence that the trials were redundant.

        See Also
        --------
        effective_trials_pca : spectral cross-check on the same reservoir.
        """
        m = float(self._n)
        if m <= 1.0:
            return max(m, 1.0)
        rho = self.mean_pairwise_correlation()
        if not math.isfinite(rho):
            return m
        n_hat = rho + (1.0 - rho) * m
        return float(min(max(n_hat, 1.0), m))

    def effective_trials_pca(self) -> dict[str, float]:
        """Spectral cross-check on ``N_hat`` (Lopez de Prado, 2019).

        Eq. 9 is a single-number summary of the whole correlation matrix and it
        assumes the dependence is well described by an average. When the
        population has structure — a handful of tight clusters plus a long tail
        of one-offs — the eigenvalue spectrum says more. Two standard effective
        dimensions of the reservoir correlation matrix, each rescaled by
        ``M / n_sampled`` so they are directly comparable to ``N_hat``:

        ``n_eff_entropy``
            :math:`\\exp(H)` with :math:`H = -\\sum_i p_i \\ln p_i`,
            :math:`p_i = \\lambda_i / \\sum \\lambda`. The perplexity of the
            spectrum.
        ``n_eff_participation``
            :math:`(\\sum \\lambda)^2 / \\sum \\lambda^2`, the participation
            ratio.

        A large disagreement with ``implied_independent_trials`` is a signal
        that the population is clustered rather than uniformly correlated;
        report the **smaller** of the two if you want to stay conservative
        about how much credit the search gets.

        Returns
        -------
        dict of str to float
            ``n_sampled``, ``n_eff_entropy``, ``n_eff_participation`` and
            ``n_hat_eq9`` for side-by-side comparison. Values are ``NaN`` when
            no series were recorded.
        """
        out = {
            "n_sampled": float("nan"),
            "n_eff_entropy": float("nan"),
            "n_eff_participation": float("nan"),
            "n_hat_eq9": self.implied_independent_trials(),
        }
        corr = self.correlation_matrix()
        if corr is None:
            return out
        keep = np.isfinite(corr).all(axis=1)
        if int(np.count_nonzero(keep)) < 2:
            return out
        sub = corr[np.ix_(keep, keep)]
        n_sampled = float(sub.shape[0])
        out["n_sampled"] = n_sampled
        eig = np.linalg.eigvalsh(sub)
        eig = np.clip(eig, 0.0, None)
        total = float(eig.sum())
        if total <= 0.0:
            return out
        p = eig / total
        p_pos = p[p > 0.0]
        entropy = float(-(p_pos * np.log(p_pos)).sum())
        scale = float(self._n) / n_sampled
        out["n_eff_entropy"] = min(math.exp(entropy) * scale, float(self._n))
        out["n_eff_participation"] = min(
            float(total**2 / float(np.sum(eig**2))) * scale, float(self._n)
        )
        return out

    # -- deflation wrappers ------------------------------------------------- #
    def deflated_sharpe(
        self,
        *,
        observed_sharpe: float | None = None,
        n_observations: int | None = None,
        n_trials: float | None = None,
        sharpe_variance: float | None = None,
        skewness: float | None = None,
        kurtosis: float | None = None,
    ) -> float:
        """DSR of the best candidate, using the ledger's own ``N_hat`` and ``V``.

        A thin wrapper over
        :func:`~panelary.core.model_selection.deflated_sharpe_ratio` —
        the arithmetic already lives there and is correct (in particular it
        takes **raw**, not excess, kurtosis, where ``vectorbt`` and
        ``quantstats`` both have a confirmed bug). All this method does is
        supply the two inputs the literature keeps omitting: :math:`\\hat N`
        from :meth:`implied_independent_trials` and :math:`V` from the recorded
        scores.

        Parameters
        ----------
        observed_sharpe : float, optional
            Defaults to the per-observation Sharpe of the best candidate's
            series.
        n_observations : int, optional
            Defaults to the recorded series length.
        n_trials : float, optional
            Defaults to :meth:`implied_independent_trials`.
        sharpe_variance : float, optional
            Defaults to :attr:`score_variance`. Only meaningful if scores were
            recorded on the Sharpe scale — see :meth:`record`.
        skewness, kurtosis : float, optional
            Default to the sample moments of the best candidate's series
            (kurtosis raw, 3.0 = Gaussian).

        Returns
        -------
        float
            DSR in ``[0, 1]``, or ``NaN`` if there is nothing to deflate.

        Notes
        -----
        Read this as a **conservative screen, not a p-value** — see the module
        docstring and Pav (2026).
        """
        if observed_sharpe is None:
            if self._best_series is None:
                return float("nan")
            observed_sharpe = _sharpe_of(self._best_series)
        if not math.isfinite(float(observed_sharpe)):
            return float("nan")
        if n_observations is None:
            n_observations = self._series_len
        if n_observations is None or n_observations < 2:
            return float("nan")
        if skewness is None or kurtosis is None:
            sk, ku = (
                _moments(self._best_series)
                if self._best_series is not None
                else (0.0, 3.0)
            )
            skewness = sk if skewness is None else skewness
            kurtosis = ku if kurtosis is None else kurtosis
        if n_trials is None:
            n_trials = self.implied_independent_trials()
        if sharpe_variance is None:
            sharpe_variance = self.score_variance
        return deflated_sharpe_ratio(
            float(observed_sharpe),
            n_trials=max(1, int(round(float(n_trials)))),
            n_observations=int(n_observations),
            sharpe_variance_across_trials=float(sharpe_variance),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
        )

    def pbo(
        self,
        *,
        n_partitions: int = 16,
        statistic: str = "sharpe",
    ) -> float:
        """PBO of the recorded population via CSCV.

        A thin wrapper over
        :func:`~panelary.core.model_selection.probability_of_backtest_overfitting`
        fed with the reservoir sample as the ``(T, S)`` performance matrix.
        PBO answers a different question from the DSR: not "is the winner's
        score too big?" but "does picking the in-sample winner tell you
        anything at all about out-of-sample rank?". At PBO = 0.5 the selection
        procedure is worthless regardless of how good the winner looks.

        Parameters
        ----------
        n_partitions : int, default=16
            CSCV partition count; silently reduced to the largest even value
            the series length supports.
        statistic : str, default="sharpe"
            Ranking statistic within each IS/OS slice.

        Returns
        -------
        float
            PBO in ``[0, 1]``, or ``NaN`` if fewer than two series were kept.
        """
        mat = self._reservoir_matrix()
        if mat is None or mat.shape[0] < 2:
            return float("nan")
        perf = np.ascontiguousarray(mat.T)  # (T, S)
        t_len = perf.shape[0]
        parts = min(int(n_partitions), t_len)
        if parts % 2:
            parts -= 1
        if parts < 2:
            return float("nan")
        return float(
            probability_of_backtest_overfitting(
                perf, n_partitions=parts, statistic=statistic
            )
        )

    # -- reporting ---------------------------------------------------------- #
    def summary(
        self,
        *,
        null_scores: np.ndarray | None = None,
        null_series: np.ndarray | None = None,
    ) -> dict[str, float | str]:
        """The full honest report, with a PASS/FAIL verdict.

        Parameters
        ----------
        null_scores : ndarray of shape (k,), optional
            Scores of ``k`` zero-skill decoys pushed through the *identical*
            scoring path, on the scale of the recorded scores. When given (at
            least two finite), the null variance behind
            ``expected_max_under_null`` is their second moment about zero,
            instead of the variance of the recorded scores across trials.
        null_series : ndarray of shape (k, T), optional
            The same decoys' per-period series. When given, the DSR deflates
            the best candidate's per-observation Sharpe against the decoys'
            Sharpe variance -- the same scale -- instead of against the
            variance of the recorded scores.

        Returns
        -------
        dict
            ``n_trials`` (M), ``n_independent_trials`` (N_hat),
            ``mean_pairwise_correlation`` (rho_bar), ``score_variance`` (V),
            ``best_score``, ``best_sharpe``, ``expected_max_under_null``,
            ``deflated_sharpe``, ``pbo``, ``null_source``, ``verdict`` and
            ``reason``.

        Notes
        -----
        The verdict fails on any of three grounds, and it fails **closed**:

        1. ``pbo > 0.05`` — Bailey, Borwein, Lopez de Prado & Zhu's own rule,
           verbatim: *"reject models for which PBO is estimated to be greater
           than 0.05."*
        2. ``best_score <= expected_max_under_null`` — the winner did not beat
           what a search of this size gets from pure noise.
        3. ``deflated_sharpe < 0.95``.

        A missing ingredient (no series recorded, ``M < 2``) yields
        ``INCONCLUSIVE``, never ``PASS``.

        **Pass decoys whenever the population may contain real signal.** The
        cross-trial variance ``V`` is the null variance only if *every* trial
        is zero-skill. A search that finds a signal breeds a population of its
        variants, so ``V`` then measures the gap between signal-carrying and
        signal-free formulas and the "null" bar rises with the signal itself:
        a planted signal scoring IC 0.53 was failed against a bar of 0.68.
        Decoys estimate the null variance under the null, which is the only
        place a null distribution can be estimated from.
        """
        m = self._n
        v = self.score_variance
        rho = self.mean_pairwise_correlation()
        n_hat = self.implied_independent_trials()
        v_null, n_null = _null_second_moment(null_scores)
        sharpe_null, _ = _null_second_moment(
            None
            if null_series is None
            else np.asarray(
                [_sharpe_of(row) for row in np.atleast_2d(null_series)],
                dtype=np.float64,
            )
        )
        e_max = (
            expected_maximum_sharpe(
                max(1, int(round(n_hat))), v if v_null is None else v_null
            )
            if m >= 2
            else float("nan")
        )
        dsr = self.deflated_sharpe(sharpe_variance=sharpe_null)
        pbo = self.pbo()
        best = self.best_score
        best_sharpe = (
            _sharpe_of(self._best_series)
            if self._best_series is not None
            else float("nan")
        )

        reasons: list[str] = []
        if m < 2:
            reasons.append("fewer than 2 trials recorded")
        if self._reservoir is None:
            reasons.append(
                "no per-period series recorded: rho_bar, N_hat, DSR and PBO "
                "are unavailable"
            )
        if reasons:
            verdict = "INCONCLUSIVE"
            reason = "; ".join(reasons)
        else:
            fails: list[str] = []
            if math.isfinite(pbo) and pbo > PBO_REJECT_ABOVE:
                fails.append(f"PBO {pbo:.3f} > {PBO_REJECT_ABOVE:.2f}")
            if math.isfinite(e_max) and best <= e_max:
                fails.append(
                    f"best score {best:.4f} <= expected max under the null "
                    f"{e_max:.4f} for N_hat={n_hat:.0f}"
                )
            if math.isfinite(dsr) and dsr < DSR_ACCEPT_ABOVE:
                fails.append(f"DSR {dsr:.3f} < {DSR_ACCEPT_ABOVE:.2f}")
            if not math.isfinite(dsr) or not math.isfinite(pbo):
                verdict = "INCONCLUSIVE"
                reason = "DSR or PBO could not be computed"
            elif fails:
                verdict = "FAIL"
                reason = "; ".join(fails)
            else:
                verdict = "PASS"
                reason = (
                    f"survived {m} trials (N_hat={n_hat:.0f}, rho_bar={rho:.3f}): "
                    f"PBO {pbo:.3f}, DSR {dsr:.3f}"
                )

        return {
            "n_trials": float(m),
            "n_independent_trials": float(n_hat),
            "mean_pairwise_correlation": float(rho),
            "score_variance": float(v),
            "best_score": float(best),
            "best_sharpe": float(best_sharpe),
            "expected_max_under_null": float(e_max),
            "deflated_sharpe": float(dsr),
            "pbo": float(pbo),
            "null_source": (
                "cross-trial variance" if v_null is None else f"{n_null} decoys"
            ),
            "verdict": verdict,
            "reason": reason,
        }

    def diagnostics(self) -> dict[str, Any]:
        """:func:`search_diagnostics` over every candidate that has both scores."""
        return search_diagnostics(self.scores(), self.held_out())


# --------------------------------------------------------------------------- #
# Minimum backtest length
# --------------------------------------------------------------------------- #
def minimum_backtest_length(n_trials: int, target_sharpe: float) -> float:
    """Minimum backtest length (MinBTL), Bailey, Borwein, LdP & Zhu, Thm 3.1.

    How much history you must have before ``n_trials`` configurations stop
    being *guaranteed* to produce a spurious winner at ``target_sharpe``:

    .. math::

        MinBTL \\approx \\left[\\frac{(1-\\gamma)Z^{-1}[1-\\tfrac1N]
        + \\gamma Z^{-1}[1-\\tfrac{e^{-1}}{N}]}
        {E[\\max_N]}\\right]^2 < \\frac{2\\ln N}{E[\\max_N]^2}

    with :math:`\\gamma` the Euler-Mascheroni constant. The units of the answer
    are the units ``target_sharpe`` is expressed in: pass an *annualised*
    target and you get **years**; pass a per-observation target and you get
    observations.

    The authors' own verdict, which is the reason this function exists:

        *"if only 5 years of data are available, no more than 45 independent
        model configurations should be tried, or we are almost guaranteed to
        produce strategies with an annualized Sharpe ratio IS of 1, but an
        expected Sharpe ratio OOS of zero."*

    Use it as a **pre-flight budget constraint**: decide the search size before
    the search, from the history you actually have, rather than discovering
    afterwards that no amount of deflation can rescue the result.

    Parameters
    ----------
    n_trials : int
        Number ``N`` of *independent* configurations to be tried. For a
        correlated population use
        :meth:`TrialLedger.implied_independent_trials`, not the raw count.
    target_sharpe : float
        The Sharpe ``E[max_N]`` you would consider a "finding" — typically 1.0
        annualised.

    Returns
    -------
    float
        Required sample length. ``0.0`` for ``n_trials == 1`` (with a single
        trial there is no selection bias to outrun).

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``target_sharpe <= 0``.

    Examples
    --------
    The paper's headline pair, recovered exactly:

    >>> round(minimum_backtest_length(45, 1.0), 2)
    5.0
    >>> round(minimum_backtest_length(100, 1.0), 2)
    6.4
    """
    n = int(n_trials)
    sr = float(target_sharpe)
    if n < 1:
        raise ValueError(f"`n_trials` must be >= 1, got {n}.")
    if sr <= 0.0:
        raise ValueError(f"`target_sharpe` must be > 0, got {sr}.")
    if n == 1:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - math.exp(-1.0) / n)
    num = (1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2
    return float((num / sr) ** 2)


# --------------------------------------------------------------------------- #
# Causality (prefix invariance)
# --------------------------------------------------------------------------- #
def _series_match(a: pl.Series, b: pl.Series, rtol: float, atol: float) -> bool:
    """Null-aware, tolerance-aware equality of two Polars Series."""
    if a.len() != b.len():
        return False
    na = a.is_null().to_numpy()
    nb = b.is_null().to_numpy()
    if not np.array_equal(na, nb):
        return False
    if a.dtype.is_numeric() and b.dtype.is_numeric():
        va = a.fill_null(0.0).cast(pl.Float64).to_numpy()
        vb = b.fill_null(0.0).cast(pl.Float64).to_numpy()
        return bool(np.allclose(va, vb, rtol=rtol, atol=atol, equal_nan=True))
    return bool(a.equals(b))


def assert_causal(
    expr: pl.Expr,
    df: pl.DataFrame,
    *,
    entity: str,
    time: str,
    cut_fractions: Sequence[float] = (0.4, 0.6, 0.8),
    rtol: float = 1e-9,
    atol: float = 1e-12,
    raise_on_fail: bool = False,
) -> bool:
    """Look-ahead detector: is ``expr`` prefix-invariant on this panel?

    Panelary's first hard invariant, made executable. An expression is causally
    valid iff truncating the future cannot change the past:

    .. math::

        f(x[:T])[t] = f(x[:T+k])[t] \\quad \\forall\\, t \\le T

    Everything that leaks violates this — an explicit ``shift(-1)``, but also a
    global mean, a full-sample z-score, a threshold estimated on all the data,
    a window whose size depends on ``len(x)``. The test does not care *how* the
    leak happened, only that the value at time ``t`` moved when data after
    ``t`` appeared.

    The frame is sorted by ``(time, entity)`` so that every time-prefix is a
    literal row-prefix; the expression is evaluated on the whole frame and on
    each truncated frame, and the overlapping rows are compared null-aware.
    Nulls are compared as nulls, which is what catches ``shift(-1)``: at the
    prefix boundary the truncated frame has a null where the full frame has a
    value.

    Parameters
    ----------
    expr : polars.Expr
        The expression to check, **including** its ``.over(...)``. The
        ``evolve`` compiler owns grouping, so this is the object it emits.
    df : polars.DataFrame
        A panel frame containing ``entity`` and ``time``.
    entity, time : str
        Panel key columns.
    cut_fractions : sequence of float, default=(0.4, 0.6, 0.8)
        Fractions of the distinct timestamps at which to truncate.
    rtol, atol : float
        Numeric tolerances for the comparison.
    raise_on_fail : bool, default=False
        Raise :class:`AssertionError` instead of returning ``False``.

    Returns
    -------
    bool
        ``True`` if the expression is prefix-invariant at every cut point.

    Raises
    ------
    ValueError
        If a key column is missing or there are fewer than 3 distinct times.
    AssertionError
        If ``raise_on_fail`` and the expression looks ahead.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "sym": ["a"] * 6 + ["b"] * 6,
    ...         "date": list(range(6)) * 2,
    ...         "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
    ...     }
    ... )
    >>> assert_causal(
    ...     pl.col("x").rolling_mean(2).over("sym"), df, entity="sym", time="date"
    ... )
    True
    >>> assert_causal(pl.col("x").shift(-1).over("sym"), df, entity="sym", time="date")
    False
    """
    for col in (entity, time):
        if col not in df.columns:
            raise ValueError(f"column {col!r} is not in the frame.")
    frame = df.sort([time, entity])
    times = frame.get_column(time).unique().sort()
    n_times = times.len()
    if n_times < 3:
        raise ValueError(
            f"need at least 3 distinct timestamps to test prefix invariance, "
            f"got {n_times}."
        )

    out = "__pk_causal__"
    full = frame.select(expr.alias(out)).get_column(out)

    cuts = sorted(
        {
            k
            for k in (int(round(f * n_times)) for f in cut_fractions)
            if 2 <= k < n_times
        }
    )
    if not cuts:
        cuts = [max(2, n_times - 1)]

    for k in cuts:
        cutoff = times[k - 1]
        prefix = frame.filter(pl.col(time) <= cutoff)
        got = prefix.select(expr.alias(out)).get_column(out)
        expected = full.head(prefix.height)
        if not _series_match(got, expected, rtol, atol):
            if raise_on_fail:
                raise AssertionError(
                    f"expression is not prefix-invariant: truncating the panel "
                    f"after timestamp {cutoff!r} ({prefix.height} of "
                    f"{frame.height} rows) changed values at earlier "
                    f"timestamps. The expression looks ahead."
                )
            return False
    return True


# --------------------------------------------------------------------------- #
# Harvey-Liu multiple-testing haircut
# --------------------------------------------------------------------------- #
#: Harvey, Liu & Zhu (2016) structural-model estimates, as published: for each
#: assumed average cross-correlation ``rho`` between strategy returns, the
#: implied total number of tests ``M``, the fraction ``p0`` of tests that are
#: truly null, and the mean ``lambda`` of the exponential distribution of the
#: *monthly* mean returns of the non-null strategies.
_HLZ_TABLE: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 1295.0, 0.39660, 0.0054995),
    (0.2, 1377.0, 0.44589, 0.0055508),
    (0.4, 1476.0, 0.48604, 0.0055413),
    (0.6, 1773.0, 0.59902, 0.0055512),
    (0.8, 3109.0, 0.83901, 0.0055956),
)

#: The HLZ calibration sample: 240 monthly observations at 15% annual vol.
_HLZ_N_OBS = 240
_HLZ_MONTHLY_VOL = 0.15 / math.sqrt(12.0)


def _hlz_params(rho: float) -> tuple[float, float, float]:
    """Linearly interpolate ``(M, p0, lambda)`` from the published table."""
    r = min(max(float(rho), _HLZ_TABLE[0][0]), _HLZ_TABLE[-1][0])
    grid = np.array([row[0] for row in _HLZ_TABLE], dtype=np.float64)
    out = []
    for col in (1, 2, 3):
        vals = np.array([row[col] for row in _HLZ_TABLE], dtype=np.float64)
        out.append(float(np.interp(r, grid, vals)))
    return out[0], out[1], out[2]


#: ``math.erfc`` lifted to arrays. Two-sided p-values are computed as
#: ``erfc(|t| / sqrt(2))`` rather than ``2 * (1 - Phi(|t|))``: the latter
#: cancels catastrophically and pins to 0.0 at around ``|t| = 8.3``, which would
#: silently flatten the haircut for every Sharpe above ~1.9 annualised.
_ERFC = np.vectorize(math.erfc, otypes=[np.float64])


def _two_sided_p(t: np.ndarray) -> np.ndarray:
    """Two-sided normal p-values, precise into the far tail."""
    z = np.abs(np.asarray(t, dtype=np.float64)) / math.sqrt(2.0)
    return np.clip(_ERFC(z), 0.0, 1.0)


def _p_to_t(p: float) -> float:
    """Invert a two-sided p-value to a t-statistic, stable in the far tail.

    Uses ``Phi^{-1}(1 - p/2) = -Phi^{-1}(p/2)``; evaluating the left-hand form
    directly loses every significant digit once ``p`` drops below ~1e-16.
    """
    p_clipped = min(max(float(p), 1e-300), 1.0)
    return float(-_norm_ppf(p_clipped / 2.0))


def haircut_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    *,
    n_observations: int = _HLZ_N_OBS,
    rho: float = 0.2,
    n_sim: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Multiple-testing haircut of a Sharpe ratio (Harvey & Liu, 2015).

    Harvey & Liu, *"Backtesting"*, JPM 42(1):13-28. Their warning, which is the
    reason this cannot be replaced by a rule of thumb:

        *"the multiple testing haircut is nonlinear... it is a serious mistake
        to use the rule of thumb 50% haircut."*

    The haircut is small for genuinely exceptional Sharpes and brutal for
    marginal ones; a flat percentage gets both ends wrong.

    Method (implemented from the published equations; the authors' reference
    code is MATLAB-only and the sole faithful port is GPL-3, so this is a
    clean-room implementation)

    1. Convert the observed Sharpe to a t-statistic ``t = SR * sqrt(T)`` and a
       two-sided p-value.
    2. Draw the t-statistics of the *other* ``n_trials`` tests from the
       Harvey-Liu-Zhu structural model, calibrated to the published table: a
       fraction ``p0`` of tests are truly null, the rest have true mean returns
       drawn from an exponential with mean ``lambda``, and all tests share an
       equicorrelation ``rho`` through a single common factor,
       ``t_j = nu_j + sqrt(rho) w + sqrt(1-rho) u_j``. At ``rho = 0.2`` the
       table gives ``M = 1377``, ``p0 = 0.44589``, ``lambda = 0.0055508``
       monthly, i.e. a non-centrality of ``lambda*sqrt(240)/(0.15/sqrt(12))``
       ~ 1.99.
    3. Insert the observed p-value into that family and apply Bonferroni, Holm
       and Benjamini-Yekutieli (BHY) — reusing Panelary's
       :func:`~panelary.validation.holm_bonferroni` and
       :func:`~panelary.validation.benjamini_yekutieli`.
    4. Take the **median** adjusted p-value over ``n_sim`` draws, invert it to a
       t-statistic and back to a Sharpe.

    Parameters
    ----------
    sharpe : float
        Observed **per-observation** Sharpe, matching ``n_observations``. Must
        be positive (a negative Sharpe needs no haircut).
    n_trials : int
        Size of the multiple-testing family — your own search count, for which
        :meth:`TrialLedger.implied_independent_trials` is the defensible input.
        Passing the ``hlz_total_tests`` value returned in the result reproduces
        the paper's published, literature-wide haircuts.
    n_observations : int, default=240
        Sample length ``T`` used for ``t = SR * sqrt(T)``. The default is HLZ's
        own calibration sample, 20 years of monthly returns.
    rho : float, default=0.2
        Assumed average correlation between test statistics. Clipped to the
        published range ``[0, 0.8]`` and linearly interpolated within it.
    n_sim : int, default=2000
        Number of simulated families. Cost is ``O(n_sim * n_trials)``.
    seed : int, default=0
        RNG seed.

    Returns
    -------
    dict of str to float
        ``sharpe``, ``t_stat``, ``p_value``, ``n_trials``, ``n_observations``,
        ``rho``, ``p0``, ``lambda``, ``hlz_total_tests``, and for each of
        ``bonferroni`` / ``holm`` / ``bhy`` / ``average`` the median adjusted
        p-value (``*_pvalue``), the haircut Sharpe (``*_sharpe``) and the
        proportional haircut (``*_haircut``, in ``[0, 1]``).

    Raises
    ------
    ValueError
        If ``sharpe <= 0``, ``n_trials < 1``, ``n_observations < 2`` or
        ``n_sim < 1``.

    References
    ----------
    Harvey, C. R., & Liu, Y. (2015). "Backtesting." *JPM*, 42(1), 13-28.
    Harvey, C. R., Liu, Y., & Zhu, H. (2016). *RFS*, 29(1), 5-68.
    """
    sr = float(sharpe)
    if sr <= 0.0:
        raise ValueError(f"`sharpe` must be > 0 to be haircut, got {sr}.")
    if n_trials < 1:
        raise ValueError(f"`n_trials` must be >= 1, got {n_trials}.")
    if n_observations < 2:
        raise ValueError(f"`n_observations` must be >= 2, got {n_observations}.")
    if n_sim < 1:
        raise ValueError(f"`n_sim` must be >= 1, got {n_sim}.")

    m_hlz, p0, lam = _hlz_params(rho)
    r = min(max(float(rho), 0.0), 0.8)
    t_obs = sr * math.sqrt(n_observations)
    p_obs = float(math.erfc(abs(t_obs) / math.sqrt(2.0)))
    p_obs = min(max(p_obs, 1e-300), 1.0)

    m = int(n_trials)
    # Non-centrality of a non-null test under the HLZ calibration: the mean
    # return is Exponential(mean=lambda) and the standard error of the mean is
    # sigma/sqrt(N), so nu ~ Exponential(mean = lambda*sqrt(N)/sigma).
    nu_scale = lam * math.sqrt(_HLZ_N_OBS) / _HLZ_MONTHLY_VOL
    rng = np.random.default_rng(seed)
    sqrt_rho = math.sqrt(r)
    sqrt_one_minus = math.sqrt(1.0 - r)

    bonf = np.empty(n_sim, dtype=np.float64)
    holm = np.empty(n_sim, dtype=np.float64)
    bhy = np.empty(n_sim, dtype=np.float64)
    p_all = np.empty(m + 1, dtype=np.float64)
    p_all[0] = p_obs
    for b in range(n_sim):
        is_alt = rng.random(m) >= p0
        nu = np.where(is_alt, rng.exponential(nu_scale, size=m), 0.0)
        common = float(rng.standard_normal())
        noise = sqrt_rho * common + sqrt_one_minus * rng.standard_normal(m)
        p_all[1:] = _two_sided_p(nu + noise)
        bonf[b] = min(1.0, p_obs * (m + 1.0))
        holm[b] = float(holm_bonferroni(p_all).adjusted_pvalues[0])
        bhy[b] = float(benjamini_yekutieli(p_all).adjusted_pvalues[0])

    out: dict[str, float] = {
        "sharpe": sr,
        "t_stat": float(t_obs),
        "p_value": p_obs,
        "n_trials": float(m),
        "n_observations": float(n_observations),
        "rho": r,
        "p0": p0,
        "lambda": lam,
        "hlz_total_tests": m_hlz,
    }
    adjusted = {
        "bonferroni": float(np.median(bonf)),
        "holm": float(np.median(holm)),
        "bhy": float(np.median(bhy)),
    }
    adjusted["average"] = float(np.mean(list(adjusted.values())))
    for name, p_adj in adjusted.items():
        sr_adj = max(_p_to_t(p_adj), 0.0) / math.sqrt(n_observations)
        out[f"{name}_pvalue"] = float(p_adj)
        out[f"{name}_sharpe"] = float(sr_adj)
        out[f"{name}_haircut"] = float(min(max((sr - sr_adj) / sr, 0.0), 1.0))
    return out


# --------------------------------------------------------------------------- #
# Yan-Zheng cross-sectional bootstrap
# --------------------------------------------------------------------------- #
def cross_sectional_bootstrap(
    returns: np.ndarray | Sequence[Sequence[float]],
    *,
    n_boot: int = 1000,
    quantiles: Sequence[float] = (0.90, 0.95, 0.99, 1.00),
    block_length: int = 1,
    seed: int = 0,
) -> dict[str, float]:
    """Yan-Zheng (2017) joint-date bootstrap under the null of zero alpha.

    Yan & Zheng, *RFS* 30(4):1382-1423, mined 18,000 fundamental signals and
    then did the thing the rest of the literature does not: they built the null
    distribution *of the mining process itself*. The construction is what makes
    it work:

    * every candidate's series is **demeaned**, imposing ``alpha = 0`` exactly;
    * dates are resampled **jointly across all candidates** — one index vector
      per replication, applied to every column — so the cross-sectional
      correlation between candidates, which is what makes a large maximum easy
      to obtain by chance, is preserved;
    * the statistic compared is an **extreme percentile** of the candidate
      t-statistics, not an average.

    This is the one test under which a mechanically-mined signal can
    legitimately *pass*: Yan & Zheng found a 99th-percentile t-statistic of 4.82
    that 10,000 null simulations never reached. Treat it as the evidence path,
    not merely as a gate.

    Parameters
    ----------
    returns : ndarray of shape (T, S)
        Per-period returns (or ICs) for ``S`` candidates over ``T`` common
        dates. Rows must be aligned in time across columns — the joint
        resampling is meaningless otherwise.
    n_boot : int, default=1000
        Number of null replications.
    quantiles : sequence of float, default=(0.90, 0.95, 0.99, 1.00)
        Cross-sectional quantiles of the t-statistics to test. ``1.00`` is the
        maximum.
    block_length : int, default=1
        Mean block length of a stationary bootstrap over dates. ``1`` is the
        i.i.d. date resampling of the original paper; raise it if the series
        are serially dependent.
    seed : int, default=0
        RNG seed.

    Returns
    -------
    dict of str to float
        ``n_periods``, ``n_candidates``, ``n_boot``, and per requested quantile
        ``q``: ``t_q{q}_observed``, ``t_q{q}_null_median``, ``t_q{q}_null_p95``
        and ``t_q{q}_pvalue`` — the empirical probability that the null process
        reaches the observed extreme. Both tails are reported for the maximum
        via ``t_absmax_*``.

    Raises
    ------
    ValueError
        If the input is not a ``(T, S)`` matrix with ``T >= 3`` and ``S >= 2``,
        or if a quantile is outside ``[0, 1]``.
    """
    mat = np.asarray(returns, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"`returns` must be 2-D (T x S), got shape {mat.shape}.")
    t_len, n_cand = mat.shape
    if t_len < 3 or n_cand < 2:
        raise ValueError(
            f"need at least 3 periods and 2 candidates, got T={t_len}, S={n_cand}."
        )
    qs = [float(q) for q in quantiles]
    if any(not (0.0 <= q <= 1.0) for q in qs):
        raise ValueError(f"`quantiles` must lie in [0, 1], got {quantiles!r}.")
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1, got {n_boot}.")

    def _tstats(sample: np.ndarray) -> np.ndarray:
        mean = sample.mean(axis=0)
        se = sample.std(axis=0, ddof=1) / math.sqrt(sample.shape[0])
        se = np.where(se <= 0.0, np.inf, se)
        return mean / se

    t_obs = _tstats(mat)
    obs_q = {q: float(np.nanquantile(t_obs, q)) for q in qs}
    obs_absmax = float(np.nanmax(np.abs(t_obs)))

    # Impose the null: zero alpha for every candidate, dependence untouched.
    null_mat = mat - mat.mean(axis=0, keepdims=True)
    idx = block_bootstrap_indices(
        t_len,
        block_length=max(1, int(block_length)),
        n_boot=int(n_boot),
        scheme="stationary",
        seed=seed,
    )
    sim_q = {q: np.empty(n_boot, dtype=np.float64) for q in qs}
    sim_absmax = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        t_sim = _tstats(null_mat[idx[b]])
        for q in qs:
            sim_q[q][b] = float(np.nanquantile(t_sim, q))
        sim_absmax[b] = float(np.nanmax(np.abs(t_sim)))

    out: dict[str, float] = {
        "n_periods": float(t_len),
        "n_candidates": float(n_cand),
        "n_boot": float(n_boot),
        "t_absmax_observed": obs_absmax,
        "t_absmax_null_median": float(np.median(sim_absmax)),
        "t_absmax_null_p95": float(np.quantile(sim_absmax, 0.95)),
        "t_absmax_pvalue": float(
            (1.0 + np.count_nonzero(sim_absmax >= obs_absmax)) / (n_boot + 1.0)
        ),
    }
    for q in qs:
        tag = f"q{int(round(q * 100)):d}"
        draws = sim_q[q]
        out[f"t_{tag}_observed"] = obs_q[q]
        out[f"t_{tag}_null_median"] = float(np.median(draws))
        out[f"t_{tag}_null_p95"] = float(np.quantile(draws, 0.95))
        out[f"t_{tag}_pvalue"] = float(
            (1.0 + np.count_nonzero(draws >= obs_q[q])) / (n_boot + 1.0)
        )
    return out


# --------------------------------------------------------------------------- #
# Search diagnostics
# --------------------------------------------------------------------------- #
def search_diagnostics(
    in_sample: np.ndarray,
    held_out: np.ndarray,
    *,
    top_fraction: float = 0.1,
) -> dict[str, Any]:
    """The (in-sample, held-out) scatter over **all** candidates.

    Adapted from Numerai's Signal Miner writeup: the single most informative
    plot a search can emit is not the winner's equity curve, it is the cloud of
    every candidate's in-sample score against its held-out score. If the cloud
    is round — correlation near zero — then in-sample rank carries no
    information about held-out rank, the search found nothing, and the "winner"
    is the luckiest point on a noise cloud. You should know that *before* you
    ship, and it costs one scatter plot.

    Reporting only the winners hides exactly this: a survivorship-filtered
    sample of a round cloud still looks like a line.

    Parameters
    ----------
    in_sample : ndarray of shape (M,)
        Selection scores for every evaluated candidate.
    held_out : ndarray of shape (M,)
        The same candidates' scores on data not used for selection. Pairs where
        either value is non-finite are dropped.
    top_fraction : float, default=0.1
        Fraction of the in-sample ranking treated as "the winners" for the lift
        statistic.

    Returns
    -------
    dict
        ``scatter`` — an ``(n, 2)`` float64 array of the surviving pairs, ready
        to plot; ``n_pairs``; ``pearson`` and ``spearman`` correlations;
        ``t_stat`` and ``p_value`` of the Pearson correlation; ``slope`` and
        ``intercept`` of the held-out-on-in-sample regression;
        ``top_decile_held_out`` and ``mean_held_out`` and their difference
        ``top_decile_lift``; and a ``verdict`` string.

    Raises
    ------
    ValueError
        If the two arrays differ in length or ``top_fraction`` is not in
        ``(0, 1]``.
    """
    a = np.asarray(in_sample, dtype=np.float64).ravel()
    b = np.asarray(held_out, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(
            f"`in_sample` and `held_out` must be the same length, got "
            f"{a.size} and {b.size}."
        )
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(f"`top_fraction` must be in (0, 1], got {top_fraction}.")
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    n = int(a.size)
    scatter = np.column_stack([a, b]) if n else np.empty((0, 2), dtype=np.float64)

    out: dict[str, Any] = {
        "scatter": scatter,
        "n_pairs": n,
        "pearson": float("nan"),
        "spearman": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "slope": float("nan"),
        "intercept": float("nan"),
        "top_decile_held_out": float("nan"),
        "mean_held_out": float("nan"),
        "top_decile_lift": float("nan"),
        "verdict": "",
    }
    if n < 3 or a.std() <= 0.0 or b.std() <= 0.0:
        out["verdict"] = (
            "INCONCLUSIVE: need at least 3 candidates with both an in-sample "
            "and a held-out score, and non-degenerate spread in each. Record "
            "held-out scores for every candidate, not only the winners."
        )
        return out

    pearson = float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    denom = max(1.0 - pearson * pearson, 1e-15)
    t_stat = pearson * math.sqrt(max(n - 2, 1) / denom)
    p_value = float(2.0 * (1.0 - _norm_cdf(abs(t_stat))))
    slope = float(np.cov(a, b, ddof=1)[0, 1] / max(float(np.var(a, ddof=1)), 1e-300))
    intercept = float(b.mean() - slope * a.mean())

    k = max(1, int(math.ceil(top_fraction * n)))
    top = np.argsort(a)[::-1][:k]
    top_mean = float(b[top].mean())
    all_mean = float(b.mean())

    if p_value > 0.05 or spearman <= 0.0:
        verdict = (
            f"NO SIGNAL: in-sample vs held-out correlation is {pearson:+.3f} "
            f"(Spearman {spearman:+.3f}, p={p_value:.3g}) over {n} candidates. "
            "The cloud is round: in-sample rank does not predict held-out rank, "
            "so the search selected noise and the winner is the luckiest point. "
            "Do not ship this."
        )
    elif spearman < 0.2:
        verdict = (
            f"WEAK: in-sample vs held-out correlation is {pearson:+.3f} "
            f"(Spearman {spearman:+.3f}, p={p_value:.3g}) over {n} candidates. "
            "Selection carries some information but most of the in-sample "
            "spread is noise; deflate hard and expect a large haircut."
        )
    else:
        verdict = (
            f"SIGNAL: in-sample vs held-out correlation is {pearson:+.3f} "
            f"(Spearman {spearman:+.3f}, p={p_value:.3g}) over {n} candidates; "
            f"the top {top_fraction:.0%} of candidates average "
            f"{top_mean:+.4f} held-out against {all_mean:+.4f} overall. "
            "Selection is doing real work — still report the deflated numbers."
        )

    out.update(
        {
            "pearson": pearson,
            "spearman": spearman,
            "t_stat": float(t_stat),
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
            "top_decile_held_out": top_mean,
            "mean_held_out": all_mean,
            "top_decile_lift": top_mean - all_mean,
            "verdict": verdict,
        }
    )
    return out
