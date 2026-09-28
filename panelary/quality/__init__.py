"""Data validation for panels: invariants, contracts and leak-safety.

``panelary.quality`` is the *data* integrity gate -- distinct from
:mod:`panelary.validation`, which is the *statistical* honesty layer (purged
CV, Deflated Sharpe, bootstraps). It answers "is this panel what every
operation downstream assumes it is?" before anything is fitted on it:

* **Panel invariants** -- usable keys, no null keys, unique
  ``(entity, time)``, time non-decreasing within every entity (measured with
  :meth:`PanelFrame.is_sorted_per_entity`), no internal gaps against the
  panel's own calendar or an explicit ``frequency``, minimum history and
  calendar coverage per entity.
* **Column contracts** -- dtype, nullability, NaN, uniqueness, allowed values,
  ranges and named row rules (:class:`ColumnContract`), in pure Polars, or
  delegated to a ``dataframely.Schema`` (the optional ``schema`` extra).
* **Leak-safety invariants** -- does a fitted transform's stored state derive
  only from train rows (:func:`check_fitted_state`)? Do near-duplicate rows
  straddle the train/test boundary (:func:`check_near_duplicate_straddle`)?

:class:`PanelValidator` runs them all, splits rows dataframely-style into
``valid`` / ``invalid``, and treats each failure by its impact: ``"fail"``
rejects (and raises :class:`PanelValidationError` unless told not to),
``"warn"`` records and warns. :func:`quality_report` profiles a frame -- null
%, duplicate %, constant columns, dtype drift -- without judging it.

Every result serialises deterministically (``to_dict()`` / ``to_json()``) and
each finding carries a ``locator`` / ``observed`` / ``expected`` triple, so it
can be filed as evidence (``to_evidence()``).

Examples
--------
>>> import polars as pl
>>> from panelary.quality import ColumnContract, validate_panel
>>> df = pl.DataFrame(
...     {"id": ["a", "a", "b", "b"], "t": [1, 2, 1, 2], "px": [10.0, None, 9.0, 9.5]}
... )
>>> report = validate_panel(
...     df,
...     entity="id",
...     time="t",
...     schema={"px": ColumnContract(dtype=pl.Float64, nullable=False)},
...     raise_on_fail=False,
... )
>>> report.status, [c.id for c in report.failures]
('fail', ['nullability:px'])
>>> report.invalid.height
1
"""

from __future__ import annotations

from panelary.quality._common import (
    CheckResult,
    Impact,
    PanelValidationError,
    QualityWarning,
    ValidationReport,
)
from panelary.quality._leakage import check_fitted_state, check_near_duplicate_straddle
from panelary.quality._report import QualityReport, quality_report
from panelary.quality._schema import ColumnContract
from panelary.quality._validator import DEFAULT_IMPACT, PanelValidator, validate_panel

__all__ = [
    "DEFAULT_IMPACT",
    "CheckResult",
    "ColumnContract",
    "Impact",
    "PanelValidationError",
    "PanelValidator",
    "QualityReport",
    "QualityWarning",
    "ValidationReport",
    "check_fitted_state",
    "check_near_duplicate_straddle",
    "quality_report",
    "validate_panel",
]
