"""As-of covariance estimation, random-matrix cleaning and market state.

``panelary.covariance`` estimates the entity x entity second-moment matrix of
a return panel **as of each date**, from a trailing window, for panels of
50 to 5,000 assets -- and the market-state features that are functions of it
(absorption ratio, eigenvalue shares, effective rank, Marchenko--Pastur signal
count, market-mode localisation, turbulence).

Pure numpy + polars. Nothing here materialises a ``T x N x N`` tensor: each
estimate is a diagonal-plus-low-rank :class:`CovEstimate` built from the
``min(N, W)``-side Gram matrix of one window.

Layers
------
* **Functional core** -- :func:`estimate`: a ``(W, N)`` window in, a
  :class:`CovEstimate` out (``solve``, ``inv_quad``, ``logdet``, ``cond``,
  ``gmv_weights``, ``risk``, ``corr``, ``subset``, ``to_dense``). No panel, no
  time. Default ``method="qis"`` in ``space="correlation"``.
* **As-of engine** -- :func:`rolling` returns a lazy
  :class:`CovarianceSeries`; ``series.at(date)`` is the estimate from the last
  scheduled date ``<= date``. :class:`Schedule` / :class:`Refit` (shared with
  the rest of the package via :mod:`panelary.core._schedule`) say which dates
  are evaluated: every ``k``-th index from the panel's first date, or the
  first date of each calendar period -- never an end-anchored grid.
* **Market state** -- :func:`market_state`, :func:`turbulence`,
  :func:`market_loading`, and the :class:`MarketState` / :class:`Turbulence`
  transformers.

Leak safety
-----------
Every per-date value uses only rows dated ``<=`` that date, and every
parameter (``N_t``, ``n_eff``, the shrinkage intensity, the MP noise level,
the factor count) is computed on that window alone. The universe at ``t`` is
the entities observed at ``t`` with ``min_coverage`` of the window; it is
compacted before any matrix product, so appending data -- or entities that
list later -- leaves every earlier value bit-identical.

Import cost
-----------
This package is **lazy** (``panelary._LAZY_SUBMODULES``): ``import panelary``
does not load it. The first public name read from it imports the modules and
registers their :class:`~panelary.registry.FeatureSpec` entries.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

#: Public name -> the private module that defines it.
_EXPORTS: dict[str, str] = {}

#: Modules that register FeatureSpecs when imported.
_CATALOGUE_MODULES: list[str] = []

# --------------------------------------------------------------------------- #
# cov-core block (plan 3, M1-M3): estimators, as-of engine, market state.
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:  # pragma: no cover - static names for type checkers and IDEs
    from panelary.core._schedule import Refit, Schedule
    from panelary.covariance._estimate import METHODS, estimate
    from panelary.covariance._estimator import MarketState, Turbulence
    from panelary.covariance._state import (
        FEATURES,
        market_loading,
        market_state,
    )
    from panelary.covariance._turbulence import turbulence
    from panelary.covariance._types import (
        CovEstimate,
        SingularCovarianceError,
        Spectrum,
        WindowStats,
    )
    from panelary.covariance._window import CovarianceSeries, rolling

_EXPORTS.update(
    {
        "CovEstimate": "_types",
        "SingularCovarianceError": "_types",
        "Spectrum": "_types",
        "WindowStats": "_types",
        "METHODS": "_estimate",
        "estimate": "_estimate",
        # the shared schedule lives in panelary.core._schedule (note D1);
        # these are thin re-exports, not a second implementation
        "Schedule": "_schedule_reexport",
        "Refit": "_schedule_reexport",
        "CovarianceSeries": "_window",
        "rolling": "_window",
        "FEATURES": "_state",
        "market_state": "_state",
        "market_loading": "_state",
        "turbulence": "_turbulence",
        "MarketState": "_estimator",
        "Turbulence": "_estimator",
    }
)
_CATALOGUE_MODULES += ["_estimator"]  # registers market_state / turbulence
# --------------------------------------------------------------------------- #
# end cov-core block
# --------------------------------------------------------------------------- #

__all__ = sorted(_EXPORTS)


def _load_catalogue() -> None:
    """Import every spec-registering module, once."""
    for mod in _CATALOGUE_MODULES:
        importlib.import_module(f"{__name__}.{mod}")


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_catalogue()
    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
