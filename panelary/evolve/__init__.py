"""Evolutionary alpha-factor mining for panel data.

``panelary.evolve`` searches a space of *formulaic* features -- Polars
expressions built from a typed, leak-safe operator grammar -- and returns a
decorrelated pool with an honest statistic attached to it.

What makes this different from the rest of the field
----------------------------------------------------
A survey of ~40 symbolic-regression / genetic-programming libraries and 9
open-source alpha miners found that **none** of them are panel-aware and
**none** implement purging, embargo, or any multiple-testing control. Two
consequences follow, and this package is built around both:

* **Leak-safety is structural, not advisory.** The grammar admits an operator
  only if its :class:`~panelary.registry.FeatureSpec` is
  ``leakage_safe``; ``.over(entity)`` / ``.over(time)`` are applied by the
  compiler, never hand-written into an operator body. A look-ahead feature is
  not penalised, it is *unconstructible*. :func:`assert_causal` provides an
  independent empirical check via prefix invariance.

* **Every candidate is counted.** Searching 100,000 expressions against a
  return target is the purest multiple-testing machine ever built: under the
  null, the expected best in-sample Sharpe is about 4.4. :class:`TrialLedger`
  records every evaluation, estimates the *implied independent* trial count
  (Bailey & Lopez de Prado, Deflated Sharpe Ratio, Appendix 3 -- most
  candidates are highly correlated, so the raw count is the wrong N), and
  deflates accordingly.

A caution about expectations
----------------------------
The independent AlphaEval benchmark (KDD 2026) re-evaluated the published
alpha miners under one protocol and found the entire field scoring 0.017-0.041
predictive power against a **random-formula baseline of 0.009**, with every
method *less diverse* than random formulas. Treat a high in-sample score as a
hypothesis, read :meth:`EvolveResult.summary` before believing it, and prefer
:func:`minimum_backtest_length` to decide the budget *before* spending it.

Examples
--------
>>> import panelary as pn  # doctest: +SKIP
>>> res = pn.evolve.evolve_features(  # doctest: +SKIP
...     panel, target="fwd_ret_5d", entity="ticker", time="date"
... )
>>> print(res.summary())  # doctest: +SKIP
>>> res.to_frame()  # doctest: +SKIP
"""

from __future__ import annotations

from ._compile import compile_population, dag_stats, semantic_key
from ._fitness import (
    AlphaPool,
    PanelEvaluator,
    ic_ir,
    numerai_corr,
    rank_ic,
    turnover,
)
from ._genome import (
    canonical_form,
    complexity,
    mutate,
    ramped_population,
    random_genome,
    subgraph_crossover,
    to_infix,
)
from ._honest import (
    TrialLedger,
    assert_causal,
    cross_sectional_bootstrap,
    haircut_sharpe_ratio,
    minimum_backtest_length,
    search_diagnostics,
    significance_hurdle,
)
from ._ops import OPS, default_grammar, ops_by_kind
from ._search import EvolveConfig, EvolveResult, evolve_features
from ._select import Archive, crowding_distance, eps_lexicase, fast_non_dominated_sort
from ._types import Descriptors, EvalContext, FitnessResult, Gene, Genome, Op

__all__ = [
    "OPS",
    "AlphaPool",
    "Archive",
    "Descriptors",
    "EvalContext",
    "EvolveConfig",
    "EvolveResult",
    "FitnessResult",
    "Gene",
    "Genome",
    "Op",
    "PanelEvaluator",
    "TrialLedger",
    "assert_causal",
    "canonical_form",
    "compile_population",
    "complexity",
    "cross_sectional_bootstrap",
    "crowding_distance",
    "dag_stats",
    "default_grammar",
    "eps_lexicase",
    "evolve_features",
    "fast_non_dominated_sort",
    "haircut_sharpe_ratio",
    "ic_ir",
    "minimum_backtest_length",
    "mutate",
    "numerai_corr",
    "ops_by_kind",
    "ramped_population",
    "random_genome",
    "rank_ic",
    "search_diagnostics",
    "semantic_key",
    "significance_hurdle",
    "subgraph_crossover",
    "to_infix",
    "turnover",
]
