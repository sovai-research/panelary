"""Cheap, causal econometric **feature generators** for panels.

A self-contained subpackage of backward-looking feature builders: each one is
either a pure trailing-window statistic or an estimator that learns its
parameters on the training rows and freezes them, so every generator here can be
dropped into a leak-safe pipeline without further care.

Contents
--------
Unit roots and persistence (:mod:`._unitroot`)
    :func:`adf`, :func:`kpss`, :func:`phillips_perron`, :func:`dfgls`,
    :func:`ng_perron`, :func:`zivot_andrews` -- test statistics, approximate
    p-values from embedded published critical-value tables, and the
    Zivot-Andrews break date. :func:`unit_root_table` runs the battery per
    entity, :func:`rolling_unit_root_features` turns it into causal per-row
    features, and :class:`StationarityDifferencer` picks a per-entity
    differencing order on the training rows only.

Long memory (:mod:`._longmemory`)
    :func:`gph` and :func:`local_whittle` estimate the fractional-integration
    order ``d``; :func:`estimate_fractional_order` handles the non-stationary
    region; :func:`estimate_ffd_order` clips that estimate into a legal
    fixed-width kernel order; :class:`AutoFracDiff` makes
    :mod:`panelary._internal._ffd` **data driven per entity**, fitted train-only.

Realized volatility (:mod:`._harrv`)
    :func:`realized_measures` (RV / bipower variation / jumps),
    :func:`har_terms` and :func:`har_features` (Corsi's daily-weekly-monthly
    cascade), :func:`daily_realized_measures` for intraday aggregation, and
    :class:`HARModel` (one train-fitted OLS per entity).

Yield curves (:mod:`._nelson_siegel`)
    :func:`nelson_siegel_fit` / :func:`nelson_siegel_factors` and
    :class:`NelsonSiegel` -- level / slope / curvature from each date's maturity
    cross-section, with a train-fitted decay.

Liquidity (:mod:`._liquidity`)
    :func:`amihud_illiquidity`, :func:`roll_spread`, :func:`amivest_liquidity`
    and the combined :func:`liquidity_features`.

OHLC volatility (:mod:`._range`)
    :func:`range_volatility` -- Parkinson, Garman-Klass, Rogers-Satchell,
    Garman-Klass with overnight, Yang-Zhang and close-to-close volatility from
    open/high/low/close bars, with an optional discrete-monitoring correction;
    :func:`ohlc_variance_terms` exposes the per-bar terms for custom (e.g.
    EWMA) aggregation.

OHLC spreads (:mod:`._spread`)
    :func:`ohlc_spread` -- the EDGE estimator of Ardia, Guidotti & Kroencke
    (2024), Corwin-Schultz and Abdi-Ranaldo bid-ask spreads from OHLC bars,
    with the signed, averaging-safe moment alongside every spread.

Tails (:mod:`._evt`)
    :func:`hill_index`, :func:`gpd_fit`, :func:`pot_var_es` and
    :func:`evt_features` -- trailing-window EVT VaR / expected shortfall.

Seasonality (:mod:`._decompose`)
    :func:`causal_seasonal_decompose`, :func:`seasonal_strength`,
    :func:`decompose_features` and :class:`CausalSeasonalDecomposer` -- a
    trailing-window seasonal-trend decomposition. A classical two-sided STL
    leaks and is deliberately not offered.

Everything is pure NumPy + Polars: no ``scipy``, no ``statsmodels``, no Rust.

Examples
--------
>>> import polars as pl
>>> from panelary.econ.features import har_features, evt_features
>>> feats = har_features(  # doctest: +SKIP
...     panel, entity="ticker", time="date", returns="ret", window=22
... )
"""

from __future__ import annotations

from panelary.econ.features._common import (
    OLSResult,
    interp_pvalue,
    norm_cdf,
    norm_ppf,
    norm_sf,
    ols,
    per_entity_apply,
    per_entity_reduce,
    rolling_apply,
    rolling_beta,
)
from panelary.econ.features._decompose import (
    CausalSeasonalDecomposer,
    causal_seasonal_decompose,
    decompose_features,
    seasonal_strength,
)
from panelary.econ.features._evt import (
    GPDFit,
    evt_features,
    gpd_fit,
    hill_index,
    pot_var_es,
)
from panelary.econ.features._harrv import (
    HARModel,
    bipower_variation,
    daily_realized_measures,
    har_features,
    har_terms,
    jump_component,
    realized_measures,
    realized_variance,
)
from panelary.econ.features._liquidity import (
    amihud_illiquidity,
    amivest_liquidity,
    liquidity_features,
    roll_spread,
)
from panelary.econ.features._longmemory import (
    AutoFracDiff,
    LongMemoryResult,
    estimate_ffd_order,
    estimate_fractional_order,
    gph,
    local_whittle,
    long_memory_table,
    rolling_long_memory_features,
)
from panelary.econ.features._nelson_siegel import (
    NelsonSiegel,
    NelsonSiegelFit,
    nelson_siegel_factors,
    nelson_siegel_fit,
    nelson_siegel_loadings,
)
from panelary.econ.features._unitroot import (
    NgPerronResult,
    StationarityDifferencer,
    UnitRootResult,
    ZivotAndrewsResult,
    adf,
    dfgls,
    kpss,
    ng_perron,
    phillips_perron,
    rolling_unit_root_features,
    unit_root_table,
    zivot_andrews,
)

__all__ = [
    # unit roots
    "UnitRootResult",
    "ZivotAndrewsResult",
    "NgPerronResult",
    "adf",
    "kpss",
    "phillips_perron",
    "dfgls",
    "ng_perron",
    "zivot_andrews",
    "unit_root_table",
    "rolling_unit_root_features",
    "StationarityDifferencer",
    # long memory
    "LongMemoryResult",
    "gph",
    "local_whittle",
    "estimate_fractional_order",
    "estimate_ffd_order",
    "long_memory_table",
    "rolling_long_memory_features",
    "AutoFracDiff",
    # realized volatility / HAR
    "realized_variance",
    "bipower_variation",
    "jump_component",
    "realized_measures",
    "har_terms",
    "har_features",
    "daily_realized_measures",
    "HARModel",
    # yield curve
    "NelsonSiegelFit",
    "nelson_siegel_loadings",
    "nelson_siegel_fit",
    "nelson_siegel_factors",
    "NelsonSiegel",
    # liquidity
    "amihud_illiquidity",
    "roll_spread",
    "amivest_liquidity",
    "liquidity_features",
    # tails
    "GPDFit",
    "hill_index",
    "gpd_fit",
    "pot_var_es",
    "evt_features",
    # seasonality
    "causal_seasonal_decompose",
    "seasonal_strength",
    "decompose_features",
    "CausalSeasonalDecomposer",
    # shared numerics
    "OLSResult",
    "ols",
    "norm_cdf",
    "norm_sf",
    "norm_ppf",
    "interp_pvalue",
    "rolling_apply",
    "rolling_beta",
    "per_entity_apply",
    "per_entity_reduce",
]

# --- OHLC volatility, spreads and low-frequency liquidity (plan 5, M1-M3) ----
# One self-contained block (imports, __all__, registry registration) so it merges
# independently of the realized-measure block.
from panelary.econ.features._range import _SPECS as _RANGE_SPECS
from panelary.econ.features._range import ohlc_variance_terms, range_volatility
from panelary.econ.features._spread import _SPECS as _SPREAD_SPECS
from panelary.econ.features._spread import ohlc_spread
from panelary.registry import registry as _registry

__all__ += [
    "range_volatility",
    "ohlc_variance_terms",
    "ohlc_spread",
]
for _spec in (*_RANGE_SPECS, *_SPREAD_SPECS):
    _registry.register(_spec)
del _spec
