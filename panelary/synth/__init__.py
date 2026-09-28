"""Seeded, dial-parameterised synthetic panels with planted ground truth.

The *prior* half of the PanelPFN plan (``plans/done/panel-pfn.md``): a
generator of long-format panels whose every messy property is a dial, and whose
every draw comes back with the truth it was drawn from. It is useful on its own
as the fixture that leakage suites otherwise hand-roll, and as the task
distribution a panel foundation model would be pretrained on (that model lives
in a separate repository; nothing here trains anything).

What's here
-----------
:class:`SynthConfig`
    The dials: latent factors with Markov regime switching and stochastic
    volatility, Student-t innovations, cross-sectional clusters, entity entry
    and exit (varying ``N`` over time), asynchronous observation schedules,
    informative missingness, reporting lags, revisions, structural breaks, and
    the panel size. :meth:`SynthConfig.plain` switches every complication off.
:func:`sample_config`
    A prior over the dials, for sampling panels across the dial space.
:func:`generate_panel`
    One draw -> :class:`SyntheticPanel`: the revised ``panel``
    (``entity`` / ``time``), the as-first-observed panel, and the
    **bitemporal** ``vintages`` (``entity``, ``event_time``,
    ``knowledge_time``, ``value``, ``revision``).
:class:`GroundTruth`
    The dials, loadings, clusters, regimes, break dates, the full latent grid,
    and the **planted predictive signal** with its known lag and coefficient.
:func:`check_planted_lag`
    The leak detector the planted signal makes possible: a feature that
    carries the signal at a lag shorter than any causal feature could has read
    the future.

Guarantees
----------
* **Deterministic.** The same seed and dials give byte-identical frames on the
  same platform, across calls and across processes. All randomness flows from
  ``numpy.random.SeedSequence(seed, spawn_key=...)``; there is no global RNG
  and no dependence on hash randomisation.
* **Prefix-consistent.** Generating ``T + k`` periods with the same seed
  reproduces every row with event time ``< T``; generating ``N + m`` entities
  reproduces the first ``N`` entities. Every random draw lives in a stream keyed
  by its time step or its entity, and every hazard is per step, never a
  fraction of ``T``.
* **float64 throughout**, numpy + polars only.

Examples
--------
>>> import polars as pl
>>> from panelary.synth import SynthConfig, generate_panel, check_planted_lag
>>> data = generate_panel(SynthConfig(n_entities=30, n_periods=120), seed=7)
>>> sorted(data.vintages.columns)
['entity', 'event_time', 'knowledge_time', 'revision', 'value']
>>> causal = data.panel.with_columns(
...     f=pl.col("value").shift(1).over("entity", order_by="time"))
>>> check_planted_lag(causal, "f", data.truth).leaks
False
"""

from __future__ import annotations

from panelary.synth._config import SynthConfig, sample_config
from panelary.synth._generate import SyntheticPanel, generate_panel
from panelary.synth._truth import GroundTruth, PlantedLagReport, check_planted_lag

__all__ = [
    "GroundTruth",
    "PlantedLagReport",
    "SynthConfig",
    "SyntheticPanel",
    "check_planted_lag",
    "generate_panel",
    "sample_config",
]
