"""Bet sizing: from a side and a probability to a position (AFML ch. 10).

Public API
----------
bet_size
    ``m = pred * side * (2 Phi(z) - 1)`` from the predicted class's probability.
average_active
    Mean size of the active bets at every row (event stream, busy-period reset).
discretize
    Round sizes to a step (half-to-even), clipped to ``[-1, 1]``.
sigmoid_w, sigmoid_size, target_position, inverse_price, limit_price
    Dynamic (sigmoid) sizing from a forecast-price divergence, and the
    breakeven limit price of the order that gets there.

Frame utilities, not catalogue operators: nothing here is registered. Sizes
must come from **out-of-fold** probabilities (``panelary.cross_validate``).
numpy + polars only.
"""

from __future__ import annotations

from panelary.sizing._bet import average_active, bet_size, discretize
from panelary.sizing._dynamic import (
    inverse_price,
    limit_price,
    sigmoid_size,
    sigmoid_w,
    target_position,
)

__all__ = [
    "average_active",
    "bet_size",
    "discretize",
    "inverse_price",
    "limit_price",
    "sigmoid_size",
    "sigmoid_w",
    "target_position",
]
