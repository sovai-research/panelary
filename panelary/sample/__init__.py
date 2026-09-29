"""Event sampling, sequential bootstrap and bagging for labelled panels (AFML ch. 2, 4, 6).

numpy + polars only at import; the sequential recursions compile lazily
through :mod:`panelary._internal._jit` when the ``fast`` extra (numba) is
installed, and fall back to bitwise-identical numpy / pure-Python twins.

Public API
----------
cusum_filter
    AFML's symmetric CUSUM event filter (a Boolean expression; ``.over(entity)``).
tick_rule
    Trade side ``{-1, +1}`` by the tick rule, null until the first move.
sequential_bootstrap
    Label row indices drawn in proportion to uniqueness given earlier draws.
SequentialBagging
    A panel estimator bagging any sklearn-shaped model over such samples.
"""

from __future__ import annotations

__all__: list[str] = []

# --- [label-spans] event sampling, sequential bootstrap, bagging -------------
from panelary.sample._bagging import SequentialBagging
from panelary.sample._bootstrap import sequential_bootstrap
from panelary.sample._cusum import cusum_filter, tick_rule

__all__ += [
    "SequentialBagging",
    "cusum_filter",
    "sequential_bootstrap",
    "tick_rule",
]
# --- [/label-spans] ------------------------------------------------------------
