"""Leak-safe, Polars-native labeling (López de Prado, AFML Ch. 3).

Public API
----------
triple_barrier
    Triple-barrier method with a trailing-volatility-scaled profit-take /
    stop-loss / vertical barrier.
fixed_horizon
    Fixed-horizon forward-return label (sign or continuous).
meta_label
    Meta-labeling: primary side signal + realized label -> act/pass.
trend_scanning
    Trend-scanning label (best-|t| OLS horizon), with the information-end ``t1``.
excess_over_median
    Forward return in excess of the date's cross-sectional median.
quantile_label
    Forward-return quantile bucket (cross-sectional or trailing thresholds).

Every labeler emits a ``t1`` column (the event-end timestamp) in the same
dtype as the input time column, forming the shared span contract consumed by
purged cross-validation.
"""

from __future__ import annotations

from panelary.label._barriers import (
    fixed_horizon,
    meta_label,
    triple_barrier,
)

__all__ = [
    "triple_barrier",
    "fixed_horizon",
    "meta_label",
]

# --- trend scanning + cross-sectional labels (plan 4, M3) ------------------
from panelary.label._trend import trend_scanning  # noqa: E402
from panelary.label._xsection import excess_over_median, quantile_label  # noqa: E402

__all__ += [
    "trend_scanning",
    "excess_over_median",
    "quantile_label",
]
# --- end plan 4 M3 -----------------------------------------------------------
