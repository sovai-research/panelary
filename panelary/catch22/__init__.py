"""Clean-room, Polars-native implementation of the *catch22* feature set.

catch22 (Lubba et al., 2019, "catch22: CAnonical Time-series CHaracteristics",
*Data Mining and Knowledge Discovery* 33, 1821-1852) is a curated collection of
22 time-series features selected from the ~7700 features of the *hctsa* library
for high classification performance and low mutual redundancy.

This module is a **clean-room re-implementation** written directly from the
published algorithmic descriptions -- it does **not** vendor or copy any code
from the GPL-licensed ``pycatch22`` / ``hctsa`` sources.  Each of the 22
canonical features is implemented as a small, documented Python function that
operates on a 1-D :class:`numpy.ndarray`.

Following the standard "catch22" variant, every feature *z-scores* the input
series (mean 0, standard deviation 1, using the sample standard deviation with
``ddof=1``) before computing.  Z-scoring is idempotent, so the feature functions
are safe to call directly on raw data.

Three convenience entry points are provided:

* :func:`catch22_all` -- compute all 22 features (or 24 with ``catch24=True``,
  which appends the raw mean and standard deviation) for a single 1-D array,
  returning a ``dict[str, float]``.
* :func:`catch22_batch` -- the same features for every row of an
  ``(n_windows, L)`` array in one vectorised pass (each row's result depends
  only on that row). Every scalar feature function is a thin wrapper over the
  batched kernel, so the two cannot disagree.
* :func:`catch22_features` -- a Polars-friendly entry point that computes the
  features **per entity** on a (long-format) panel :class:`polars.DataFrame` /
  :class:`polars.LazyFrame`, returning one row per entity.  The per-entity
  ``group_by`` makes the computation leak-safe: every feature uses only the
  values of its own series.

Correctness notes
-----------------
A handful of the canonical features rely on fairly intricate sub-routines
(spline detrending, adaptive histogram binning, piecewise-linear fitting of
fluctuation scaling).  Those are implemented as faithful, documented
best-effort re-derivations and are flagged in their docstrings as
``NEEDS REVIEW`` for exact numerical parity with ``pycatch22``.  They are
algorithmically sound and return finite, correctly-signed values, but their
last-digit outputs may differ from the reference C implementation.
"""

from __future__ import annotations

# `_namespace` is imported for its side effect: it registers the
# `pl.col(...).catch22` expression namespace.
from panelary.catch22 import _namespace
from panelary.catch22._catalogue import (
    CATCH22_FUNCS,
    CATCH22_NAMES,
    CATCH24_EXTRA_NAMES,
    _compute,
    _resolve_names,
    catch22_all,
    catch22_all_expr,
    catch22_batch,
    catch22_features,
)
from panelary.catch22._features import (
    CO_Embed2_Dist_tau_d_expfit_meandiff,
    CO_f1ecac,
    CO_FirstMin_ac,
    CO_HistogramAMI_even_2_5,
    CO_trev_1_num,
    DN_HistogramMode_5,
    DN_HistogramMode_10,
    DN_OutlierInclude_n_001_mdrmd,
    DN_OutlierInclude_p_001_mdrmd,
    FC_LocalSimple_mean1_tauresrat,
    FC_LocalSimple_mean3_stderr,
    IN_AutoMutualInfoStats_40_gaussian_fmmi,
    MD_hrv_classic_pnn40,
    PD_PeriodicityWang_th0_01,
    SB_BinaryStats_diff_longstretch0,
    SB_BinaryStats_mean_longstretch1,
    SB_MotifThree_quantile_hh,
    SB_TransitionMatrix_3ac_sumdiagcov,
    SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    SP_Summaries_welch_rect_area_5_1,
    SP_Summaries_welch_rect_centroid,
    _fluctuation_analysis,
    _histogram_mode,
    _outlier_include,
    _welch_cumulative,
    _welch_spectrum,
)
from panelary.catch22._helpers import (
    _acf,
    _as_1d,
    _bspline_design,
    _coarsegrain_quantile,
    _first_zero_ac,
    _histcounts,
    _line_sse,
    _local_simple_residuals,
    _longest_run,
    _lsq_spline_fit,
    _num_bins_auto,
    _pair_counts,
    _zscore,
)

__all__ = [
    "CATCH22_NAMES",
    "CATCH24_EXTRA_NAMES",
    "catch22_all",
    "catch22_batch",
    "catch22_features",
    "catch22_all_expr",
    # individual features
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
]
