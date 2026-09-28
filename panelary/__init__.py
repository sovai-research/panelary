"""Panelary: leak-safe, fast feature engineering and ML for panel data, built on Polars.

Panelary turns a long-format ``(entity, time, *features)`` panel into a first-class,
lazy, leak-safe object. It provides the :class:`PanelFrame` view, a leak-safe
sklearn-shaped :class:`Pipeline`, the ``.panel`` / ``.xs`` Polars expression and
frame namespaces, finance-grade cross-validation (:class:`PurgedKFold`,
:class:`CombinatorialPurgedCV`) and backtest-overfitting metrics
(:func:`deflated_sharpe_ratio`, :func:`probability_of_backtest_overfitting`).

Panelary is built on the foundations of `functime <https://github.com/functime-org/functime>`_
(Apache-2.0, actively maintained) and interoperates with functime and Nixtla rather
than replacing them. The project is Panelary, the distribution is ``panelary``,
and the import convention is ``import panelary as pn``.

Public symbols are imported lazily and guarded: an optional component that fails to
import emits a warning but never breaks the rest of the package. Whatever is actually
available is listed in :data:`__all__`.
"""

from __future__ import annotations

import importlib as _importlib
import warnings

#: Kept in step with ``[project] version`` in ``pyproject.toml``; the two are
#: asserted equal by ``tests/test_verbs.py``.
__version__ = "0.5.0"

__all__ = ["__version__"]

# from panelary.feature_extractors import FeatureExtractor  # noqa: F401


def _warn_unavailable(component: str, exc: Exception) -> None:
    """Warn that an optional Panelary component could not be imported."""
    warnings.warn(
        f"Panelary: optional component {component!r} is unavailable "
        f"({exc.__class__.__name__}: {exc}). The rest of the package is unaffected.",
        stacklevel=3,
    )


# --- Panelary core (leak-safe panel ML kernel) ------------------------------
try:
    from panelary.core import (
        BusinessDays,
        CombinatorialPurgedCV,
        PanelEstimator,
        PanelFrame,
        PanelTransformer,
        Pipeline,
        PurgedKFold,
        as_panel,
        asof_join,
        deflated_sharpe_ratio,
        expanding_window_split,
        probability_of_backtest_overfitting,
        sliding_window_split,
    )
except ImportError as exc:
    _warn_unavailable("panelary.core", exc)
else:
    __all__ += [
        "BusinessDays",
        "CombinatorialPurgedCV",
        "PanelEstimator",
        "PanelFrame",
        "PanelTransformer",
        "Pipeline",
        "PurgedKFold",
        "as_panel",
        "asof_join",
        "deflated_sharpe_ratio",
        "expanding_window_split",
        "probability_of_backtest_overfitting",
        "sliding_window_split",
    ]

# --- CPCV runner (cross_validate / validate / CVReport) ---------------------
try:
    from panelary.core.model_selection import (
        CVReport,
        cross_validate,
        validate,
    )
except ImportError as exc:
    _warn_unavailable("panelary.core.model_selection runner", exc)
else:
    __all__ += ["CVReport", "cross_validate", "validate"]

# --- Feature registry + expression namespaces -------------------------------
# Importing `namespaces` registers the `.panel` / `.xs` Polars expression and
# frame namespaces as a side effect (idempotent).
try:
    from panelary import namespaces as namespaces
    from panelary.registry import (
        FeatureRegistry,
        FeatureSpec,
        register_feature,
        registry,
    )
except ImportError as exc:
    _warn_unavailable("panelary.namespaces/registry", exc)
else:
    __all__ += [
        "namespaces",
        "FeatureRegistry",
        "FeatureSpec",
        "register_feature",
        "registry",
    ]

# --- Panel-native transforms ------------------------------------------------
try:
    from panelary import transform as transform
except ImportError as exc:
    _warn_unavailable("panelary.transform", exc)
else:
    __all__ += ["transform"]

# --- Feature extraction -----------------------------------------------------
# Importing `feature_extractors` registers the `ts` Polars namespace as a side
# effect and exposes the bulk `extract_features` entry point.
try:
    from panelary import feature_extractors as feature_extractors
except ImportError as exc:
    _warn_unavailable("panelary.feature_extractors", exc)
else:
    __all__ += ["feature_extractors"]
    try:
        from panelary.feature_extractors import extract_features
    except ImportError as exc:
        _warn_unavailable("panelary.feature_extractors.extract_features", exc)
    else:
        __all__ += ["extract_features"]

# --- CAFE imputation --------------------------------------------------------
try:
    from panelary.preprocessing import cafe_impute
except ImportError as exc:
    _warn_unavailable("panelary.preprocessing.cafe_impute", exc)
else:
    __all__ += ["cafe_impute"]

try:
    from panelary.imputation import CafeImputer
except ImportError as exc:
    _warn_unavailable("panelary.imputation.CafeImputer", exc)
else:
    __all__ += ["CafeImputer"]

# --- Leakage verifier -------------------------------------------------------
try:
    from panelary.testing import assert_no_lookahead
except ImportError as exc:
    _warn_unavailable("panelary.testing.assert_no_lookahead", exc)
else:
    __all__ += ["assert_no_lookahead"]

# --- catch22 feature set (clean-room) ---------------------------------------
try:
    from panelary import catch22 as catch22
except ImportError as exc:
    _warn_unavailable("panelary.catch22", exc)
else:
    __all__ += ["catch22"]

# --- Labeling (triple-barrier) ----------------------------------------------
try:
    from panelary import label as label
    from panelary.label import triple_barrier
except ImportError as exc:
    _warn_unavailable("panelary.label", exc)
else:
    __all__ += ["label", "triple_barrier"]

# --- Models & feature selection subpackages ---------------------------------
try:
    from panelary import models as models
except ImportError as exc:
    _warn_unavailable("panelary.models", exc)
else:
    __all__ += ["models"]

# The subpackage is imported for its side effects, but deliberately **not**
# bound to the name `select` here: the `pn.select(...)` verb wired at the
# bottom owns it. Binding the module too would leave type checkers seeing
# `pn.select` as a module and reporting "Module not callable" at every call
# site. `from panelary.select import mrmr` and `pn.select.mrmr` both work.
try:
    _importlib.import_module("panelary.select")
except ImportError as exc:
    _warn_unavailable("panelary.select", exc)
else:
    __all__ += ["select"]

# --- Cross-sectional / factor evaluation ------------------------------------
try:
    from panelary import factor as factor
except ImportError as exc:
    _warn_unavailable("panelary.factor", exc)
else:
    __all__ += ["factor"]

# --- Dimensionality reduction -----------------------------------------------
# Imported for its side effects only; the `pn.reduce(...)` verb owns the name
# (see the note on `select` above).
try:
    _importlib.import_module("panelary.reduce")
except ImportError as exc:
    _warn_unavailable("panelary.reduce", exc)
else:
    __all__ += ["reduce"]

# --- Clustering -------------------------------------------------------------
# Imported for its side effects only; the `pn.cluster(...)` verb owns the name
# (see the note on `select` above).
try:
    _importlib.import_module("panelary.cluster")
except ImportError as exc:
    _warn_unavailable("panelary.cluster", exc)
else:
    __all__ += ["cluster"]

# --- Feature attribution (leak-safe, panel-aware SHAP) ----------------------
try:
    from panelary import explain as explain
except ImportError as exc:
    _warn_unavailable("panelary.explain", exc)
else:
    __all__ += ["explain"]

# --- Honest validation & selection ------------------------------------------
try:
    from panelary import validation as validation
except ImportError as exc:
    _warn_unavailable("panelary.validation", exc)
else:
    __all__ += ["validation"]

# --- Panel & time-series econometrics ---------------------------------------
try:
    from panelary import econ as econ
except ImportError as exc:
    _warn_unavailable("panelary.econ", exc)
else:
    __all__ += ["econ"]

# --- Causal bubble, regime & change-point detection --------------------------
try:
    from panelary import detect as detect
except ImportError as exc:
    _warn_unavailable("panelary.detect", exc)
else:
    __all__ += ["detect"]

# --- Point-in-time compiler & borrowed-accuracy metric -----------------------
# numpy + polars only, so it is eager like `detect` above. The package is
# `leakage`, not `causal`: `causal` is already one of the eight verbs.
try:
    from panelary import leakage as leakage
except ImportError as exc:
    _warn_unavailable("panelary.leakage", exc)
else:
    __all__ += ["leakage"]

# --- Data validation: panel invariants, contracts, leak-safety --------------
# numpy + polars only, so it is eager like `leakage` above. `quality`, not
# `validation`: that name is the statistical layer (CPCV, DSR, bootstrap).
try:
    from panelary import quality as quality
except ImportError as exc:
    _warn_unavailable("panelary.quality", exc)
else:
    __all__ += ["quality"]

# --- Data cleaning: dedup, canonicalisation, outliers, entity resolution ----
# numpy + polars only (rapidfuzz is imported lazily), so eager like `quality`.
try:
    from panelary import clean as clean
except ImportError as exc:
    _warn_unavailable("panelary.clean", exc)
else:
    __all__ += ["clean"]

# --- Synthetic panels with planted ground truth -----------------------------
# numpy + polars only, so it is eager like `leakage` above.
try:
    from panelary import synth as synth
except ImportError as exc:
    _warn_unavailable("panelary.synth", exc)
else:
    __all__ += ["synth"]

# --- Shape algebra: compress / lift / factorize / sketch a named axis --------
# The subpackage init is lazy: no transform module loads until a public name is
# first read, so this costs only the init (and `_tensor`, which `cluster` already
# needs).
try:
    from panelary import shape as shape
except ImportError as exc:
    _warn_unavailable("panelary.shape", exc)
else:
    __all__ += ["shape"]

# --- Remaining light-core modules -------------------------------------------
# These cost ~0-10 ms on top of the base import and pull no optional
# dependency (verified: sklearn/scipy/pandas/plotly stay out of sys.modules),
# so they are imported eagerly like everything above. `detect` is separate only
# because it is the largest of them.
for _name in (
    "core",
    "base",
    "testing",
    "offsets",
    "imputation",
    "preprocessing",
    "seasonality",
    "metrics",
    "conformal",
    "cross_validation",
    "evaluation",
):
    try:
        _mod = _importlib.import_module(f"panelary.{_name}")
    except ImportError as exc:  # pragma: no cover - depends on the install
        _warn_unavailable(f"panelary.{_name}", exc)
    else:
        globals()[_name] = _mod
        __all__ += [_name]
del _name

# --- The golden path: eight top-level verbs ---------------------------------
# One verb per task, uniform signature, thin facade over the subpackages above
# (see `panelary/_internal/_verbs.py`). This block must stay **last**, because `select`,
# `cluster` and `reduce` are also subpackage names and the verb is what
# `pn.select` / `pn.cluster` / `pn.reduce` must resolve to.
#
# Nothing is lost to that collision: those three verbs are `_SubpackageVerb`
# instances, which forward any attribute they lack to the module of the same
# name. So `pn.select(df, ...)`, `pn.select.mrmr`, `from panelary.select import
# mrmr` and `import panelary.select as s` all work. The other five verbs shadow
# nothing and are plain functions.
#
# `_verbs` imports polars, numpy and every subpackage *inside* the function
# bodies, so wiring it here costs ~0 ms of import time.
try:
    from panelary._internal._verbs import (
        bubbles,
        causal,
        cluster,
        features,
        impute,
        reduce,
        regression,
        select,
    )
except ImportError as exc:  # pragma: no cover - defensive
    _warn_unavailable("panelary._internal._verbs", exc)
else:
    # `cluster` / `reduce` / `select` are already in `__all__` as subpackages;
    # the name now resolves to the verb, so don't list it twice.
    __all__ += [
        _verb
        for _verb in (
            "bubbles",
            "causal",
            "cluster",
            "features",
            "impute",
            "reduce",
            "regression",
            "select",
        )
        if _verb not in __all__
    ]

#: Submodules that are reachable as ``pn.<name>`` but are **not** imported
#: eagerly, because doing so would violate the light-core guarantee. Measured
#: cost of importing them at ``import panelary`` time:
#:
#:   forecasting  828 ms, pulls sklearn + scipy + pandas + lightgbm
#:   backtesting  imports forecasting._reduction, so it drags in all of the
#:                above (measured: it took the cold import from 80 ms to 779 ms)
#:   plotting     143 ms, pulls plotly
#:   llm          raises outright unless the `llm` extra is installed
#:   embed        ~5 ms, but it builds on `shape`'s windowing and projection
#:                modules, and `import panelary` promises to load no shape
#:                transform (tests/test_shape_leak_safety.py)
#:
#: They resolve on first attribute access via the PEP 562 hook below, so
#: ``pn.forecasting`` works without making every ``import panelary``
#: pay for it. They are deliberately absent from ``__all__`` so that
#: ``from panelary import *`` cannot trigger a heavy or failing import.
_LAZY_SUBMODULES = ("forecasting", "llm", "plotting", "backtesting", "embed")


def __getattr__(name: str):
    """Resolve heavy submodules on first access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        _mod = _importlib.import_module(f"panelary.{name}")
        globals()[name] = _mod  # cache: __getattr__ is not called again
        return _mod
    raise AttributeError(f"module 'panelary' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazy submodules in tab-completion and ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
