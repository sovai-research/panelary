"""Nonlinear dependence measurement, screening and inference for panel data.

Every dependence library ships a coefficient and calls the job done. The
coefficient is the easy half. :mod:`panelary.depend` is built around the other
half -- **the p-value being real**:

1. **The i.i.d. null is wrong on persistent data, by an order of magnitude.**
   Two *independent* AR(1) series with ``phi = 0.95`` (n = 500), tested with
   Chatterjee's xi under its i.i.d. asymptotic null, reject at ~56% at a
   nominal 5% (``tests/test_depend_calibration.py`` asserts it). ``null="auto"``
   runs a lag-1 rank-autocorrelation pre-check and switches to a block
   permutation whose block length is sized for *dependence tests*
   (:func:`pair_block_length`), and an explicit i.i.d. null on such data warns
   (:class:`SerialDependenceWarning`).
2. **Dependence is not volatility clustering.** ``devol=`` reports every
   statistic twice -- raw and divided by a causal rolling volatility -- so a
   GARCH effect cannot pass for a feature.
3. **Panels are not one long series.** Estimates are within entity by default,
   aggregated with a heterogeneity statistic, and the default panel null is
   ``"common-time"``: whole date columns permuted jointly, so every common
   shock survives in the null. Per-entity shuffles are not valid under common
   factors (asserted in the calibration tests).

Statistics (array level)
------------------------
:func:`xi` (Chatterjee, **directional**), :func:`spearman`, :func:`kendall`,
:func:`pearson`, :func:`hoeffding_d`, :func:`dcor` / :func:`dcov2` (exact
``O(n^1.5)`` univariate path, tiled multivariate path), :func:`gcmi` (read its
"honesty trap" note), :func:`gcmi_conditional`, :func:`tail_dependence`,
:func:`exceedance_corr`, :func:`variation_of_information`, :func:`hsic`
(random-Fourier-feature HSIC with a frozen :class:`RFFMap`), :func:`mi_ksg`,
:func:`cmi_ksg`, :func:`transfer_entropy_array`, :func:`gcm`, :func:`codec`,
and the matrix forms :func:`xi_matrix`, :func:`gcmi_matrix`,
:func:`tail_dependence_matrix`, :func:`hsic_matrix`.

Analysis (frame level, fixed result schema)
-------------------------------------------
:func:`dependence`, :func:`independence_test`, :func:`by_entity`,
:func:`aggregate`, :func:`pooled`, :func:`lag_dependence`,
:func:`nonlinear_acf`, :func:`nonlinear_ccf`, :func:`dependence_matrix`,
:func:`feature_screen` (+ the leak-safe :class:`ScreenSelector`),
:func:`mutual_information`, :func:`transfer_entropy`,
:func:`conditional_dependence`, :func:`foci`.

Features (one value per row, trailing window, prefix-invariant)
---------------------------------------------------------------
``pl.col(y).ts.rolling_xi(x, window=60)``, ``.ts.rolling_dcor``,
``.ts.rolling_tail_dep`` and ``.ts.rolling_gcmi`` -- added to the ``.ts``
namespace (and the operator registry) when this package is imported. Use them
with ``.over(entity)`` on a time-sorted panel.

Every analysis function returns a ``pl.DataFrame`` on one schema
(:data:`RESULT_SCHEMA`) -- ``estimate, p_value, p_value_adj, method,
estimator, null_method, n_resamples, block_length, direction, lag, n_obs,
n_entities, coverage, heterogeneity, transform, approximate, seed, warnings``
-- so ``pl.concat`` across methods, nulls and transforms just works.

Pure numpy + polars. No SciPy in the import path; oracles are test-only.
"""

from __future__ import annotations

# Side effect: add the rolling dependence operators to the ``.ts`` expression
# namespace and register their FeatureSpecs (additive; see panelary.namespaces.ts).
import panelary.namespaces.ts  # noqa: F401
from panelary.depend._coef import (
    exceedance_corr,
    hoeffding_d,
    hoeffding_pvalue,
    kendall,
    kendall_pvalue,
    pearson,
    pearson_pvalue,
    spearman,
    tail_dependence,
    tail_dependence_matrix,
    tail_pvalue,
    xi,
    xi_matrix,
    xi_null_sd,
    xi_pvalue,
)
from panelary.depend._cond import GCMResult, codec, conditional_dependence, foci, gcm
from panelary.depend._energy import DcorResult, dcor, dcor_pvalue, dcov2, partial_dcor
from panelary.depend._engine import dependence, devolatilise, independence_test
from panelary.depend._frame import RESULT_SCHEMA
from panelary.depend._info import (
    TEResult,
    cmi_ksg,
    gcmi,
    gcmi_conditional,
    gcmi_matrix,
    gcmi_pvalue,
    mi_ksg,
    mi_to_r,
    psi,
    transfer_entropy_array,
    variation_of_information,
)
from panelary.depend._information import mutual_information, transfer_entropy
from panelary.depend._kernel import HSICResult, RFFMap, hsic, hsic_matrix, rff
from panelary.depend._kernels import METHODS, min_obs_for
from panelary.depend._lag import (
    lag_dependence,
    nonlinear_acf,
    nonlinear_ccf,
    optimal_lag,
)
from panelary.depend._matrix import (
    ScreenSelector,
    dependence_matrix,
    feature_screen,
    matrix_values,
    psd_repair,
    to_distance,
)
from panelary.depend._null import (
    NULL_POLICY,
    SerialDependenceWarning,
    SurrogateResult,
    auto_block_length,
    block_permutation_indices,
    circular_shift,
    common_time_indices,
    entity_permutation_indices,
    gamma_pvalue,
    iaaft,
    pair_block_length,
    phase_randomise,
    pvalue,
    serial_dependence,
)
from panelary.depend._panel import AggregateResult, aggregate, by_entity, pooled
from panelary.depend._ranks import (
    dominance_counts,
    normal_scores,
    pairwise_complete,
    ranks,
    sliding_ranks,
)
from panelary.depend._rolling import (
    rolling_dcor,
    rolling_gcmi,
    rolling_tail_dep,
    rolling_xi,
)

__all__ = [
    "METHODS",
    "NULL_POLICY",
    "RESULT_SCHEMA",
    "AggregateResult",
    "DcorResult",
    "GCMResult",
    "HSICResult",
    "RFFMap",
    "ScreenSelector",
    "SerialDependenceWarning",
    "SurrogateResult",
    "TEResult",
    "aggregate",
    "auto_block_length",
    "block_permutation_indices",
    "by_entity",
    "circular_shift",
    "cmi_ksg",
    "codec",
    "common_time_indices",
    "conditional_dependence",
    "dcor",
    "dcor_pvalue",
    "dcov2",
    "dependence",
    "dependence_matrix",
    "devolatilise",
    "dominance_counts",
    "entity_permutation_indices",
    "exceedance_corr",
    "feature_screen",
    "foci",
    "gamma_pvalue",
    "gcm",
    "gcmi",
    "gcmi_conditional",
    "gcmi_matrix",
    "gcmi_pvalue",
    "hoeffding_d",
    "hoeffding_pvalue",
    "hsic",
    "hsic_matrix",
    "iaaft",
    "independence_test",
    "kendall",
    "kendall_pvalue",
    "lag_dependence",
    "matrix_values",
    "mi_ksg",
    "mi_to_r",
    "min_obs_for",
    "mutual_information",
    "nonlinear_acf",
    "nonlinear_ccf",
    "normal_scores",
    "optimal_lag",
    "pair_block_length",
    "pairwise_complete",
    "partial_dcor",
    "pearson",
    "pearson_pvalue",
    "phase_randomise",
    "pooled",
    "psd_repair",
    "psi",
    "pvalue",
    "ranks",
    "rff",
    "rolling_dcor",
    "rolling_gcmi",
    "rolling_tail_dep",
    "rolling_xi",
    "serial_dependence",
    "sliding_ranks",
    "spearman",
    "tail_dependence",
    "tail_dependence_matrix",
    "tail_pvalue",
    "to_distance",
    "transfer_entropy",
    "transfer_entropy_array",
    "variation_of_information",
    "xi",
    "xi_matrix",
    "xi_null_sd",
    "xi_pvalue",
]
