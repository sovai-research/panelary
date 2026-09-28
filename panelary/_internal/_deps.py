"""Optional-dependency helpers for a light, CAFE-style core.

Panelary's mandatory footprint is intentionally small (numpy + polars). Anything
heavier — scikit-learn, scipy, flaml, holidays, plotting, LLM back-ends, boosters
— is an *optional extra*. Feature code that needs one of those imports it lazily
(inside the function/method that uses it) and routes the import through
:func:`require`, so that a missing dependency produces a single, actionable
message (``pip install 'panelary[ml]'``) instead of a bare ``ImportError``
or, worse, a heavy import at ``import panelary`` time.

This module has **no third-party imports** and must stay that way — importing it
must never pull anything beyond the standard library.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Maps an importable top-level module name to the Panelary extra that provides
#: it. Used to turn a missing import into ``pip install 'panelary[<extra>]'``.
_MODULE_TO_EXTRA: dict[str, str] = {
    "sklearn": "ml",
    "scipy": "scipy",
    "flaml": "forecasting",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "holidays": "seasonality",
    "umap": "dimreduce",
    "plotly": "viz",
    "cafe": "cafe",
    "tqdm": "progress",
    "openai": "llm",
    "anthropic": "llm",
    "tiktoken": "llm",
    "tenacity": "llm",
    "numba": "fast",
    "shapiq": "explain",
    "shap": "explain",
    "dataframely": "schema",
    "rapidfuzz": "fuzzy",
}

#: PyPI distribution name (for the pip hint).
_DIST_NAME = "panelary"


def require(
    module: str, *, extra: str | None = None, feature: str | None = None
) -> ModuleType:
    """Import and return an optional dependency, or raise a helpful error.

    Parameters
    ----------
    module : str
        The importable module name, e.g. ``"sklearn"`` or ``"sklearn.decomposition"``.
        The extra is looked up from the top-level package (``sklearn``).
    extra : str, optional
        Override the extra name to suggest. Defaults to the mapping in
        :data:`_MODULE_TO_EXTRA`, falling back to the top-level module name.
    feature : str, optional
        Human-readable name of the Panelary feature needing the dependency, used
        to make the error message concrete.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed, with a ``pip install`` hint.
    """
    top = module.split(".", 1)[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via feature code
        chosen = extra or _MODULE_TO_EXTRA.get(top, top)
        what = f"{feature} requires" if feature else "This feature requires"
        raise ImportError(
            f"{what} the optional '{top}' dependency, which is not installed. "
            f"Install it with:  pip install '{_DIST_NAME}[{chosen}]'"
        ) from exc


def have(module: str) -> bool:
    """Return True if an optional module can be imported, without raising."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
