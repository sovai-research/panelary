"""Label concurrency, uniqueness and sample weights (AFML ch. 4), leak-safe.

Overlapping labels share information: a 20-day label on every day has
average uniqueness near 1/21, so the label count overstates the sample size
and an unweighted fit over-counts crowded periods. This package measures the
overlap and turns it into sample weights, all from **one span table** built
from the label's ``t1`` column -- the same spans the purged cross-validators
purge on, so labels, weights and purge can never disagree.

Public API
----------
spans
    The canonical :class:`~panelary.core._spans.SpanTable` of a labelled panel.
concurrency
    Number of labels active on each row (exact, int64).
average_uniqueness
    Each label's average uniqueness ``mean 1/c`` over its span.
effective_n
    ``sum`` of average uniqueness: the honest number of independent labels.

The functions above are **global** (they count every label in the frame): use
them to describe a labelled panel or for the final refit. Inside
cross-validation, weights must be recomputed from the training fold's labels
only.
"""

from __future__ import annotations

from panelary.core._spans import SpanTable
from panelary.weights._concurrency import (
    average_uniqueness,
    concurrency,
    effective_n,
    spans,
)

__all__ = [
    "SpanTable",
    "average_uniqueness",
    "concurrency",
    "effective_n",
    "spans",
]
