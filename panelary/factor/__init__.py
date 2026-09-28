"""Tier-2 cross-sectional / factor evaluation toolkit.

The ``factor`` package holds the *evaluation and estimation* surface for
cross-sectional factor research: leak-safe forward-return alignment, the
information coefficient (IC/ICIR), portfolio sorts, and per-date feature-set
orthogonalization. It is **Tier 2** — it may import numpy and freely imports the
Tier-1 ``namespaces`` layer, but is never imported by it, preserving the
documented import boundary.

Leak-safety is the moat and it is centralized, not hoped for:

* forward returns come from one audited backward-shift site
  (:func:`forward_return`) with a gap guard;
* every cross-sectional statistic is computed **per date**, never pooled/global
  (IC ``.over(time)``; sorts bucket per date; orthogonalization decomposes each
  date's cross-section on its own).

Public API
----------
:func:`forward_return`
    Leak-safe forward-return alignment (backward per-entity shift + gap guard).
:func:`ic`, :func:`ic_summary`
    Per-date information coefficient and its ICIR / t-stat / hit-rate summary.
:func:`portfolio_sort`
    Per-date quantile sort, long-short spread, and monotonicity.
:func:`orthogonalize`
    Per-date feature-set de-correlation (Gram-Schmidt / QR).

A note on ``safe_scope``
------------------------
Three of the four specs below classify cleanly under
:data:`~panelary.registry.VALID_SAFE_SCOPES`. :func:`forward_return` does not:
it is the supervised *target* builder, and a target is causal in neither of the
senses the vocabulary offers. It is not ``"rowwise"`` -- the value on row ``t``
is drawn from ``t + horizon`` -- and it is not a window summary either. It is
recorded as ``"window"`` because that is the scope whose operative warning
(do not broadcast this per row) is true of it, and because the alternative
would advertise the label as a row-safe feature. If a ``"target"`` scope is
ever added to the registry, this is the spec that wants it.
"""

from __future__ import annotations

from panelary.registry import FeatureSpec, registry

from ._align import forward_return
from .ic import ic, ic_summary
from .neutralize import orthogonalize
from .portfolio import SortResult, portfolio_sort

__all__ = [
    "forward_return",
    "ic",
    "ic_summary",
    "portfolio_sort",
    "SortResult",
    "orthogonalize",
]

_LICENSE = "Apache-2.0"
_SOURCE = "Panelary"

_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="forward_return",
        namespace="factor",
        input_shape="frame",
        output_shape="series",
        params={"horizon": int, "price": str, "ret": str},
        tier="B",
        panel_safe=True,  # backward per-entity shift; respects entity boundaries
        leakage_safe=True,  # the single audited negative-shift site + gap guard
        # NEITHER scope describes this operator, and "rowwise" would be a lie.
        # It is the *label* builder: the value it writes on row ``t`` is the
        # return realised over ``(t, t+horizon]``, so it reads ``t+horizon`` by
        # definition -- the one place in the library where looking forward is
        # the point. Both instruments say so (it fails ``assert_no_lookahead``,
        # and truncating the panel turns a value into a null), and that is the
        # operator working correctly, not a defect. "rowwise" asserts the value
        # at ``t`` uses only data at ``<= t``; declaring it here would tell a
        # consumer filtering the catalogue for row-safe features that the
        # target is one of them. "window" is the conservative reading -- its
        # first clause (a summary of a delimited window) fits badly, but its
        # operative clause, that broadcasting this per row is a look-ahead by
        # construction, is exactly right. See the module note below: this spec
        # wants a third scope (a "target"/"label" value) that does not exist.
        safe_scope="window",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="ic",
        namespace="factor",
        input_shape="frame",
        output_shape="frame",
        params={"method": str},
        tier="B",
        panel_safe=False,  # cross-sectional: correlates across entities per date
        leakage_safe=True,
        # One output row per date, correlating that date's cross-section and
        # nothing else (``group_by(time)``), so each row uses only data at its
        # own date. The whole-sample reduction over dates lives in the separate,
        # deliberately unregistered ``ic_summary``. Verified: truncating the
        # later dates leaves every earlier date's IC bit-identical.
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="portfolio_sort",
        namespace="factor",
        input_shape="frame",
        output_shape="frame",
        params={"q": int},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        # Mixed, so it takes the conservative scope. The per-date half is
        # rowwise -- buckets are assigned ``.over(time)`` and the long-short
        # spread is computed within each date (verified stable under truncating
        # later dates). But unlike ``ic``, this call also returns whole-sample
        # scalars on its ``SortResult``: ``mean_spread``, ``t_stat``,
        # ``monotonicity`` and ``bucket_means`` average across every date in the
        # frame, so they move when the sample grows (measured: mean_spread
        # 0.00807 -> 0.00356 on adding later dates). Those are summaries of a
        # completed backtest window and must never be carried back per row.
        safe_scope="window",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="orthogonalize",
        namespace="factor",
        input_shape="frame",
        output_shape="frame",
        params={"method": str, "by": object},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        # One value per row, decomposed against a basis built from that row's
        # own date (the group key is ``[time, *by]`` and each group gathers only
        # its own row positions), so no other date is ever read. Verified: the
        # orthogonalized columns are unchanged when later dates are truncated.
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
)

for _spec in _SPECS:
    registry.register(_spec)
