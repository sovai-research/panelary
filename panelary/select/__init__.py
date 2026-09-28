"""Leak-safe feature selection for panels.

Public API:

* :func:`mrmr` -- minimum-Redundancy-Maximum-Relevance selection, computed only
  on the rows passed in (in-fold).
* :func:`mda` -- Mean-Decrease-Accuracy (permutation) importance evaluated
  through a purged CV splitter, so importances are leak-free.
* :func:`mdi` -- Mean-Decrease-Impurity importance from a fitted tree ensemble.
* :func:`refit_shapley` -- the Shapley value of each feature group (or data
  source) to a model **refit** on it through a purged CV splitter; redundant
  groups split their credit instead of both scoring zero.
  :class:`RefitShapleyReport` is what it returns.
* :class:`MRMRSelector` -- a :class:`~panelary.core.protocol.PanelTransformer`
  wrapping :func:`mrmr` for use as a ``"select"`` step in a
  :class:`~panelary.core.pipeline.Pipeline`.

Unsupervised (no target ``y``; still leak-safe, selection frozen at fit time):

* :func:`pfa` -- Principal Feature Analysis: cluster PCA feature-loading
  vectors, keep one representative per cluster.
* :func:`variance` -- keep the ``k`` highest-variance features.
* :func:`correlation` -- prune features above an absolute-correlation threshold.
* :func:`projection_importance` -- feature energy in a projected space.
* :func:`select_top` -- pick from a ranking table by top-``k`` / cumulative share.
* :class:`PFASelector`, :class:`VarianceSelector`, :class:`CorrelationSelector` --
  the corresponding ``"select"`` pipeline steps.
"""

from __future__ import annotations

from panelary.select._methods import MRMRSelector, mda, mdi, mrmr
from panelary.select._refit_shapley import RefitShapleyReport, refit_shapley
from panelary.select._unsupervised import (
    CorrelationSelector,
    PFASelector,
    VarianceSelector,
    correlation,
    pfa,
    projection_importance,
    select_top,
    variance,
)

__all__ = [
    "mrmr",
    "mda",
    "mdi",
    "refit_shapley",
    "RefitShapleyReport",
    "MRMRSelector",
    "pfa",
    "variance",
    "correlation",
    "projection_importance",
    "select_top",
    "PFASelector",
    "VarianceSelector",
    "CorrelationSelector",
]
