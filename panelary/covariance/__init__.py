"""As-of covariance, market state and cross-sectional distribution features.

``panelary.covariance`` estimates, per date and on a trailing window, how the
cross-section of returns moves together and how it is distributed. Every
feature is a function of data at or before its own date: windows trail,
universes and group labels are taken as of the date, refit grids are anchored
at the panel's first date, and nothing is fit on the full sample. Pure numpy +
polars.

Import cost and the operator catalogue
--------------------------------------
This package initialiser is **lazy** (PEP 562), and ``panelary.covariance`` is
itself a lazy submodule of ``panelary``: ``import panelary`` loads neither. The
first public name read from here imports every catalogue module, which
registers one :class:`~panelary.registry.FeatureSpec` per frame operation under
``namespace="covariance"``. The single-date cross-sectional summaries
(``.xs.dispersion``, ``.xs.tail_index``, ``.xs.up_share``, ``.xs.entropy``) are
Polars expressions in :mod:`panelary.namespaces.xs` and are registered with it.

The export table below is assembled from one block per contributing milestone,
so each can be added or removed without touching the others.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static names for type checkers and IDEs
    # --- cov-xs block (plan 3: M3 co-movement subset, M4) ---------------- #
    from panelary.covariance._comovement import avg_correlation, common_idio_vol
    from panelary.covariance._xsdist import (
        avg_skewness,
        kelly_jiang_beta,
        kelly_jiang_tail,
        xs_wasserstein,
    )
    # --- end cov-xs block ------------------------------------------------- #

#: Public name -> the private module that defines it.
_EXPORTS: dict[str, str] = {}

#: Modules that register FeatureSpecs at import time.
_CATALOGUE_MODULES: tuple[str, ...] = ()

# --- cov-xs block (plan 3: M3 co-movement subset, M4) --------------------- #
# Average correlation and common idiosyncratic volatility (M3), and the
# cross-date distribution features (M4). Their dense plumbing is in
# `_xsdense`, which registers nothing.
_EXPORTS.update(
    {
        "avg_correlation": "_comovement",
        "common_idio_vol": "_comovement",
        "avg_skewness": "_xsdist",
        "kelly_jiang_beta": "_xsdist",
        "kelly_jiang_tail": "_xsdist",
        "xs_wasserstein": "_xsdist",
    }
)
_CATALOGUE_MODULES += ("_comovement", "_xsdist")
# --- end cov-xs block ----------------------------------------------------- #

__all__ = sorted(_EXPORTS)


def _load_catalogue() -> None:
    """Import every catalogue module, registering the covariance specs once."""
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
