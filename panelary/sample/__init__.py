"""Event sampling, bar construction, sequential bootstrap and bagging (AFML ch. 2, 4, 6).

Which rows a label is defined on, and how to resample them honestly.

numpy + polars only at import; the sequential recursions compile lazily
through :mod:`panelary._internal._jit` when the ``fast`` extra (numba) is
installed, and fall back to bitwise-identical numpy / pure-Python twins.

Public API
----------
bars
    Tick, volume and dollar bars (fixed or adaptive threshold) as a long panel,
    one row per completed bar stamped at its last tick.
imbalance_bars
    Tick / volume / dollar imbalance and run bars (AFML 2.3.2) with clamped,
    causal expected-bar-size EWMAs.
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

from panelary.sample._bagging import SequentialBagging
from panelary.sample._bars import bars
from panelary.sample._bootstrap import sequential_bootstrap
from panelary.sample._cusum import cusum_filter, tick_rule
from panelary.sample._imbalance import imbalance_bars

__all__ = [
    "SequentialBagging",
    "bars",
    "cusum_filter",
    "imbalance_bars",
    "sequential_bootstrap",
    "tick_rule",
]
