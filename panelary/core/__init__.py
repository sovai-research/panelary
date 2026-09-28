"""Panelary core: the correctness-by-construction panel ML kernel.

This subpackage is the *moat* of Panelary. It provides a small, stable, fully
typed, lazy-first (pure Polars) foundation that every higher-level component
(concrete transformers, the feature registry, models) builds on:

* :class:`~panelary.core.panel_frame.PanelFrame` — a typed, lazy view over
  a long-format ``(entity, time, *features)`` panel that validates the keys once
  and stays lazy.
* :class:`~panelary.core.protocol.PanelTransformer` /
  :class:`~panelary.core.protocol.PanelEstimator` — sklearn-shaped base
  classes carrying an explicit, machine-checkable leakage contract
  (``panel_safe`` / ``leakage_safe``).
* :class:`~panelary.core.pipeline.Pipeline` — a leak-safe chain of steps
  with sklearn ergonomics.
* :mod:`~panelary.core.model_selection` — finance-grade, leak-safe
  cross-validation (:class:`PurgedKFold`, :class:`CombinatorialPurgedCV`,
  walk-forward splitters) and de Prado overfitting metrics
  (:func:`deflated_sharpe_ratio`, :func:`probability_of_backtest_overfitting`).
  Embargo and purge horizons also accept calendar durations and
  :class:`BusinessDays`.
* :func:`~panelary.core.asof.asof_join` — the bitemporal point-in-time join: each
  row sees a value only once it was known.

All public names are re-exported here and (additively) from the top-level
``panelary`` package.
"""

from __future__ import annotations

from panelary.core.asof import asof_join
from panelary.core.model_selection import (
    BusinessDays,
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe_ratio,
    expanding_window_split,
    probability_of_backtest_overfitting,
    sliding_window_split,
)
from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.core.pipeline import Pipeline
from panelary.core.protocol import PanelEstimator, PanelTransformer

__all__ = [
    # Data view
    "PanelFrame",
    "as_panel",
    # Protocols
    "PanelTransformer",
    "PanelEstimator",
    # Pipeline
    "Pipeline",
    # Point-in-time join
    "asof_join",
    # Cross-validation
    "BusinessDays",
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "expanding_window_split",
    "sliding_window_split",
    # Metrics
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
]
