"""The 22 canonical catch22 feature functions (clean-room re-implementation).

Each function takes a 1-D array-like, z-scores it (mean 0, ``ddof=1``
standard deviation) and returns a single ``float``.  Written directly from
the algorithmic descriptions in Lubba et al. (2019); no code is vendored
from the GPL-licensed ``pycatch22`` / ``hctsa`` sources.

Every function here is a thin wrapper over the batched kernel of the same
name in :mod:`panelary.catch22._batch`, called on a one-row batch, so the
scalar and batched entry points cannot drift apart. NaNs are dropped from the
input first, as before.
"""

from __future__ import annotations

import numpy as np

from panelary._internal import _numpy_stats

from ._batch import (
    _fluctuation_rows,
    _histogram_mode_rows,
    _outlier_include_1d,
    feature_rows,
)
from ._helpers import _as_1d, _zscore


def _one(name: str, x) -> float:
    """Evaluate one feature on one series through the batched kernel."""
    return float(feature_rows(name, _as_1d(x)[None, :])[0])


# ---------------------------------------------------------------------------
# Distribution features
# ---------------------------------------------------------------------------
def DN_HistogramMode_5(x) -> float:
    """Mode of the z-scored distribution estimated from a 5-bin histogram.

    Returns the centre of the most-populated bin; ties are resolved by
    averaging the centres of all bins sharing the maximum count.
    """
    return _one("DN_HistogramMode_5", x)


def DN_HistogramMode_10(x) -> float:
    """Mode of the z-scored distribution estimated from a 10-bin histogram.

    See :func:`DN_HistogramMode_5`.
    """
    return _one("DN_HistogramMode_10", x)


def _histogram_mode(y: np.ndarray, n_bins: int) -> float:
    """Histogram mode of an already z-scored series (NaN if constant)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(_histogram_mode_rows(y[None, :], n_bins)[0])


def CO_trev_1_num(x) -> float:
    """Numerator of the time-reversibility statistic *trev* at lag 1.

    ``trev_num = mean((y[t+1] - y[t]) ** 3)`` on the z-scored series -- a
    measure of temporal (a)symmetry.
    """
    return _one("CO_trev_1_num", x)


def MD_hrv_classic_pnn40(x) -> float:
    """pNN40: fraction of successive differences with ``|diff| > 0.04``.

    Adapted from the classic heart-rate-variability pNNx statistic.  catch22
    scales successive differences by 1000 and thresholds at 40, i.e. an
    absolute threshold of ``0.04`` on the z-scored series.
    """
    return _one("MD_hrv_classic_pnn40", x)


# ---------------------------------------------------------------------------
# Autocorrelation-based features
# ---------------------------------------------------------------------------
def CO_f1ecac(x) -> float:
    """First ``1/e`` crossing of the autocorrelation function.

    The (linearly interpolated) lag at which the ACF first drops below
    ``1/e``.  Falls back to the series length if the ACF never crosses.
    """
    return _one("CO_f1ecac", x)


def CO_FirstMin_ac(x) -> float:
    """Lag of the first local minimum of the autocorrelation function."""
    return _one("CO_FirstMin_ac", x)


def CO_HistogramAMI_even_2_5(x) -> float:
    """Automutual information at lag 2 using 5 equal-width bins.

    The AMI ``sum_ij p_ij * log(p_ij / (p_i p_j))`` (natural log) between the
    series and its lag-2 copy, with the joint distribution estimated from a
    5x5 equal-width histogram spanning ``[min-0.1, max+0.1]``.
    """
    return _one("CO_HistogramAMI_even_2_5", x)


def IN_AutoMutualInfoStats_40_gaussian_fmmi(x) -> float:
    """First minimum of the Gaussian automutual information over lags 1..40.

    Under a Gaussian assumption the AMI at lag ``k`` is
    ``-0.5 * log(1 - r_k**2)`` where ``r_k`` is the Pearson correlation between
    the series and its lag-``k`` copy.  Returns the (0-based) index into the
    lag array of the first local minimum, or the number of lags if none.
    """
    return _one("IN_AutoMutualInfoStats_40_gaussian_fmmi", x)


def CO_Embed2_Dist_tau_d_expfit_meandiff(x) -> float:
    """Goodness of an exponential fit to successive distances in a 2-D embedding.

    The series is embedded in 2-D with delay ``tau`` (the first ACF zero-
    crossing, capped at ``n/10``).  The Euclidean distances between successive
    embedded points are histogrammed (Scott's rule) and compared, bin by bin,
    to an exponential density with rate ``1/mean(distance)``; the feature is
    the mean absolute density difference.
    """
    return _one("CO_Embed2_Dist_tau_d_expfit_meandiff", x)


# ---------------------------------------------------------------------------
# Binary / symbolic features
# ---------------------------------------------------------------------------
def SB_BinaryStats_mean_longstretch1(x) -> float:
    """Longest run of consecutive values above the mean (binarise by mean)."""
    return _one("SB_BinaryStats_mean_longstretch1", x)


def SB_BinaryStats_diff_longstretch0(x) -> float:
    """Longest run of consecutive non-increases (binarise the diff by 0).

    The successive-difference series is binarised (``1`` if ``> 0`` else ``0``)
    and the length of the longest run of ``0`` s (consecutive decreases /
    flats) is returned.
    """
    return _one("SB_BinaryStats_diff_longstretch0", x)


def SB_MotifThree_quantile_hh(x) -> float:
    """Entropy of the length-2 motif distribution over a 3-letter alphabet.

    The series is symbolised into 3 equiprobable (quantile) letters; the 3x3
    matrix of consecutive letter pairs is normalised into a probability
    distribution whose (natural-log) Shannon entropy ``hh`` is returned.
    """
    return _one("SB_MotifThree_quantile_hh", x)


def SB_TransitionMatrix_3ac_sumdiagcov(x) -> float:
    """Trace of the covariance of a 3-state transition matrix (ac downsampling).

    The series is downsampled by ``tau`` (the first ACF zero-crossing),
    symbolised into 3 equiprobable states, and its 3x3 transition-probability
    matrix formed.  The feature is the sum of the diagonal (trace) of the
    covariance matrix of that transition matrix's columns.
    """
    return _one("SB_TransitionMatrix_3ac_sumdiagcov", x)


# ---------------------------------------------------------------------------
# Periodicity / forecasting features
# ---------------------------------------------------------------------------
def PD_PeriodicityWang_th0_01(x) -> float:
    """Periodicity measure of Wang et al. with threshold ``0.01``.

    The series is spline-detrended, its ACF computed, and the lag of the first
    ACF peak whose height exceeds the preceding trough by more than ``0.01``
    (and is positive) is returned.

    NEEDS REVIEW: the reference uses a bespoke piecewise-cubic ``splinefit``;
    here a least-squares cubic spline with 3 interior knots is used, so exact
    numerical parity with ``pycatch22`` is not guaranteed.

    The spline fit is the projection onto the cubic B-spline design's column
    space (pure NumPy, the design from :func:`_bspline_design`), so this
    feature returns the same value with and without SciPy installed.
    """
    return _one("PD_PeriodicityWang_th0_01", x)


def FC_LocalSimple_mean1_tauresrat(x) -> float:
    """Ratio of first ACF-zero of the residuals to that of the series.

    "Local simple" mean forecasting with a training window of 1 point; the
    feature is ``firstzero_ac(residuals) / firstzero_ac(series)``.
    """
    return _one("FC_LocalSimple_mean1_tauresrat", x)


def FC_LocalSimple_mean3_stderr(x) -> float:
    """Standard deviation of "local simple" mean-forecast residuals (window 3)."""
    return _one("FC_LocalSimple_mean3_stderr", x)


# ---------------------------------------------------------------------------
# Outlier-timing features
# ---------------------------------------------------------------------------
def _outlier_include(y: np.ndarray, sign: int) -> float:
    """Shared core of the ``DN_OutlierInclude`` features (z-scored input).

    NEEDS REVIEW: the trimming rule (which thresholds contribute to the final
    median) is a faithful best-effort re-derivation; exact parity with the
    reference C implementation is not guaranteed.
    """
    return _outlier_include_1d(np.asarray(y, dtype=np.float64).ravel(), sign)


def DN_OutlierInclude_p_001_mdrmd(x) -> float:
    """Median outlier timing as positive outliers are progressively included.

    Thresholds increase in steps of ``0.01``; at each level the centred median
    time-index of points above the threshold is recorded, and the median of
    those values (over the well-populated threshold range) is returned.
    """
    return _one("DN_OutlierInclude_p_001_mdrmd", x)


def DN_OutlierInclude_n_001_mdrmd(x) -> float:
    """Negative-outlier counterpart of :func:`DN_OutlierInclude_p_001_mdrmd`."""
    return _one("DN_OutlierInclude_n_001_mdrmd", x)


# ---------------------------------------------------------------------------
# Power-spectrum features
# ---------------------------------------------------------------------------
def _welch_spectrum(y: np.ndarray):
    """Welch power spectrum with a rectangular window over the whole series.

    Returns ``(w, S)`` where ``w`` is the angular frequency in ``[0, pi]`` and
    ``S`` the power spectral density.
    """
    n = y.size
    f, pxx = _numpy_stats.welch(
        y,
        window="boxcar",
        nperseg=n,
        noverlap=0,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    return 2.0 * np.pi * f, pxx


def _welch_cumulative(y: np.ndarray):
    if y.size < 4:
        return None
    w, s = _welch_spectrum(y)
    if w.size < 2:
        return None
    dw = w[1] - w[0]
    area = np.sum(s) * dw
    if area <= 0:
        return None
    s_norm = s / area
    cs = np.cumsum(s_norm) * dw
    return w, cs


def SP_Summaries_welch_rect_area_5_1(x) -> float:
    """Normalised spectral power in the first fifth of the frequency range.

    The rectangular-window Welch spectrum is normalised to unit area; the
    feature is the cumulative power up to one fifth of the frequency axis.
    """
    return _one("SP_Summaries_welch_rect_area_5_1", x)


def SP_Summaries_welch_rect_centroid(x) -> float:
    """Spectral centroid: angular frequency where cumulative power reaches 0.5."""
    return _one("SP_Summaries_welch_rect_centroid", x)


# ---------------------------------------------------------------------------
# Fluctuation-scaling (DFA family) features
# ---------------------------------------------------------------------------
def _fluctuation_analysis(x, how: str) -> float:
    """Shared core of the ``SC_FluctAnal`` features.

    Computes the fluctuation ``F(tau)`` of the integrated (profile) series over
    log-spaced window sizes ``tau`` -- root-mean-square of linearly-detrended
    residuals for ``dfa`` (order-2), or of the detrended range for
    ``rsrangefit`` -- then fits two straight lines to ``log F`` vs ``log tau``
    and returns the fraction of scales in the first (small-scale) regime at the
    best breakpoint.

    NEEDS REVIEW: the exact scale grid and breakpoint convention differ subtly
    between implementations; this is a documented best-effort re-derivation and
    may not match ``pycatch22`` to the last digit.  It returns a value in
    ``[0, 1]``.
    """
    y = _zscore(x)
    dfa, rs = _fluctuation_rows(y[None, :])
    return float(dfa[0] if how == "dfa" else rs[0])


def SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x) -> float:
    """DFA fluctuation-scaling: proportion of scales in the first regime.

    Detrended fluctuation analysis (order-2 RMS) variant.  See
    :func:`_fluctuation_analysis`.
    """
    return _one("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", x)


def SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(x) -> float:
    """Rescaled-range fluctuation-scaling: proportion of scales in first regime.

    Rescaled-range (range-of-detrended-profile) variant.  See
    :func:`_fluctuation_analysis`.
    """
    return _one("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", x)
