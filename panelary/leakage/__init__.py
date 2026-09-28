"""Point-in-time compilation, and a number for what leakage is worth.

Panelary's other defences are contracts you have to honour: ``panel_safe`` /
``leakage_safe`` on a transformer, ``.over(entity_col)`` in time order, a fit
that happens per fold. This subpackage is the pair of tools that stop that
being a matter of discipline — one that *prevents* leakage and one that
*prices* it.

**The compiler** (:mod:`~panelary.leakage._compile`) — :func:`audit` walks the
serialised Polars expression tree
(``Expr.meta.serialize(format="json")``) and reports what each node does with
information from the future; :func:`causalize` rewrites what has an exact
point-in-time equivalent (a backward fill becomes forward, a centred rolling
window becomes trailing, an ``.over(entity)`` with no ``order_by`` gains one)
and **refuses** what does not. It fails closed: an unrecognised node kind is
:data:`~panelary.leakage.Classification.REFUSE`, never ``SAFE``. The rule
table lives in :mod:`~panelary.leakage._rules`, keyed by qualified node kind.

**The metric** (:mod:`~panelary.leakage._borrowed`) —
:func:`borrowed_accuracy` runs a pipeline twice, permissively and constrained
to information available at prediction time, and reports the gap: the accuracy
*borrowed* from data the method will not have at prediction time. Because
components interact, the attribution to each :class:`Component` is the exact
Shapley value over all ``2^k`` subsets — no sampling — so the parts sum to the
whole by construction.

The two are one package on purpose: the compiler prevents, the metric measures,
and the metric is the compiler's test suite.

Pure NumPy + Polars, like the rest of the light core: no SciPy, no scikit-learn,
no statsmodels.

Notes
-----
The serialised expression format is **not** a stable Polars API. The compiler
is checked against the versions in
:data:`~panelary.leakage._types.POLARS_TREE_FORMAT_TESTED` and fails closed on
anything it does not recognise, so a format change costs you refusals, never a
silent leak.

Examples
--------
>>> import polars as pl
>>> from panelary.leakage import audit, causalize
>>> leaky = pl.col("x").fill_null(strategy="backward").rolling_mean(5, center=True)
>>> audit(leaky, time="date", entity="ticker").verdict  # doctest: +SKIP
<Verdict.REWRITTEN: 'rewritten'>
>>> safe = causalize(leaky, time="date", entity="ticker")  # doctest: +SKIP
"""

from __future__ import annotations

from panelary.leakage._borrowed import (
    BorrowedAccuracyReport,
    Component,
    borrowed_accuracy,
)
from panelary.leakage._compile import (
    audit,
    audit_features,
    causalize,
)
from panelary.leakage._types import (
    Classification,
    CompileResult,
    Context,
    FeatureSetAudit,
    Finding,
    LeakageRefused,
    Verdict,
)

__all__ = [
    "BorrowedAccuracyReport",
    "Classification",
    "CompileResult",
    "Component",
    "Context",
    "FeatureSetAudit",
    "Finding",
    "LeakageRefused",
    "Verdict",
    "audit",
    "audit_features",
    "borrowed_accuracy",
    "causalize",
]
