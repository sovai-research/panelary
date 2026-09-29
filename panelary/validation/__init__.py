"""Honest validation & selection — the layer that makes "leak-safe" measurable.

Panelary's other pillars produce features (``feature_extractors``), factors
(``reduce/``) and attributions (``explain/``). This one produces *evidence*:
after a search over many candidates, what performance is real and what is an
artefact of the search? It has four parts.

**Splitting** (:mod:`~panelary.validation._cv`) — Combinatorial Purged CV
(Lopez de Prado, Ch. 12) turns a backtest into a *distribution* of paths instead
of a single trajectory, plus purged/embargoed walk-forward splits and the
purged train/calibration split conformal prediction needs.

**Selection statistics** (:mod:`~panelary.validation._selection_stats`) —
the Probabilistic and Deflated Sharpe ratios, the Probability of Backtest
Overfitting, the Romano-Wolf stepdown for family-wise error and
Benjamini-Hochberg / Benjamini-Yekutieli for false-discovery control (BY is the
one valid under the arbitrary cross-sectional dependence a panel has).

**Forecast comparison** (:mod:`~panelary.validation._forecast_tests`) —
Diebold-Mariano with HAC + the Harvey-Leybourne-Newbold correction, Hansen's
SPA test, the Model Confidence Set, and proper scoring rules (CRPS, pinball,
interval score, PIT/reliability) vectorised in Polars.

**Resampling** (:mod:`~panelary.validation._bootstrap`) — moving-block,
circular-block, stationary, wild and sieve bootstraps, all seeded and all
honouring fold boundaries so a resampled block can never splice one fold's data
into another's.

Everything here is pure NumPy + Polars: no SciPy, no ``arch``, no
``statsmodels``.

Examples
--------
>>> import numpy as np
>>> from panelary.validation import cpcv_backtest_paths, deflated_sharpe_ratio
>>> returns = np.random.default_rng(0).standard_normal(120) * 0.01
>>> paths = cpcv_backtest_paths(
...     120, lambda train, test: returns[test], n_groups=6, n_test_groups=2
... )
>>> paths.shape
(120, 5)
"""

from __future__ import annotations

from panelary.validation._bootstrap import (
    block_bootstrap_indices,
    circular_block_bootstrap,
    moving_block_bootstrap,
    resolve_segments,
    sieve_bootstrap,
    stationary_bootstrap,
    wild_bootstrap,
)
from panelary.validation._cv import (
    CombinatorialPurgedCV,
    CVReport,
    IndexSplit,
    PurgedKFold,
    cpcv_backtest_paths,
    cpcv_splits,
    cross_validate,
    expanding_window_split,
    fold_boundaries,
    purged_calibration_split,
    sliding_window_split,
    walk_forward_backtest_path,
    walk_forward_splits,
)
from panelary.validation._forecast_tests import (
    DieboldMarianoResult,
    MCSResult,
    SPAResult,
    crps_ensemble,
    crps_from_quantiles,
    crps_gaussian,
    diebold_mariano,
    interval_score,
    model_confidence_set,
    newey_west_variance,
    pinball_loss,
    pinball_loss_expr,
    pit_histogram,
    pit_values,
    score_quantile_forecasts,
    superior_predictive_ability,
)

# --- forecast evaluation & Sharpe inference (plan 1: fe-sharpe) --------------
from panelary.validation._results import (
    EVALUATION_SCHEMA,
    EvaluationResult,
    evaluation_table,
)
from panelary.validation._selection_stats import (
    MultipleTestResult,
    benjamini_hochberg,
    benjamini_yekutieli,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    holm_bonferroni,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    romano_wolf,
    romano_wolf_mean_test,
)
from panelary.validation._sharpe import (
    BlockLengthCalibration,
    annualize_sharpe,
    sharpe_block_length,
    sharpe_ratio_inference,
    sharpe_ratio_test,
)

_FE_SHARPE_ALL = [
    "BlockLengthCalibration",
    "EVALUATION_SCHEMA",
    "EvaluationResult",
    "annualize_sharpe",
    "evaluation_table",
    "sharpe_block_length",
    "sharpe_ratio_inference",
    "sharpe_ratio_test",
]
# --- end fe-sharpe -------------------------------------------------------------

__all__ = [
    "CVReport",
    "CombinatorialPurgedCV",
    "DieboldMarianoResult",
    "IndexSplit",
    "MCSResult",
    "MultipleTestResult",
    "PurgedKFold",
    "SPAResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "block_bootstrap_indices",
    "circular_block_bootstrap",
    "cpcv_backtest_paths",
    "cpcv_splits",
    "cross_validate",
    "crps_ensemble",
    "crps_from_quantiles",
    "crps_gaussian",
    "deflated_sharpe_ratio",
    "diebold_mariano",
    "expanding_window_split",
    "expected_maximum_sharpe",
    "fold_boundaries",
    "holm_bonferroni",
    "interval_score",
    "minimum_track_record_length",
    "model_confidence_set",
    "moving_block_bootstrap",
    "newey_west_variance",
    "pinball_loss",
    "pinball_loss_expr",
    "pit_histogram",
    "pit_values",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_calibration_split",
    "resolve_segments",
    "romano_wolf",
    "romano_wolf_mean_test",
    "score_quantile_forecasts",
    "sieve_bootstrap",
    "sliding_window_split",
    "stationary_bootstrap",
    "superior_predictive_ability",
    "walk_forward_backtest_path",
    "walk_forward_splits",
    "wild_bootstrap",
    *_FE_SHARPE_ALL,
]
