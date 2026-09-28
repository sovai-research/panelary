"""Panel-aware data cleaning: deduplication, canonicalisation, outliers, ER.

The input-integrity stage that precedes features, factors, attribution and
honest validation. The ordering doctrine: deterministic identity work first
(canonicalise, then deduplicate -- on the whole panel, before any split),
anything that *learns* a parameter last and fit on training rows only.

Deduplication (M1)
    :class:`Deduplicator` / :func:`dedup` -- exact and near-duplicate removal
    (MinHash, C-MinHash, SimHash + LSH, verified), panel-global and
    point-in-time by default. :func:`near_duplicate_clusters` /
    :func:`near_duplicate_pairs` are the diagnostic views.
Split-awareness
    :func:`straddling_pairs`, :func:`assert_no_straddle`,
    :func:`straddling_clusters`, :func:`purge_near_duplicates` and
    :class:`SplitAwareCV` guarantee no near-duplicate pair crosses a
    train/test boundary.
Canonicalisation and survivorship (M3)
    :class:`Canonicalizer`, :func:`normalize_text`, :func:`fingerprint`,
    :func:`ngram_fingerprint`, :func:`fingerprint_clusters`,
    :class:`Survivorship`, :func:`golden_records`.
Outliers (M4)
    :class:`OutlierCleaner` -- MAD / IQR / z-score / quantile thresholds fit
    on train (per entity, pooled, or per date), causal Hampel / rolling
    filters.
Entity resolution (M5)
    :class:`EntityResolver`, :func:`resolve_entities`, :class:`LSHBlocking`,
    :func:`string_similarity` (``rapidfuzz`` optional, numpy fallback).

Every transformer rides the :class:`~panelary.core.protocol.PanelTransformer`
contract and states ``panel_safe`` / ``leakage_safe`` honestly, per instance
where an option changes the answer.
"""

from __future__ import annotations

from panelary.clean._canonicalize import (
    Canonicalizer,
    fingerprint,
    fingerprint_clusters,
    ngram_fingerprint,
    normalize_text,
)
from panelary.clean._cluster import connected_components
from panelary.clean._dedup import (
    Deduplicator,
    dedup,
    near_duplicate_clusters,
    near_duplicate_pairs,
)
from panelary.clean._exact import exact_unique
from panelary.clean._outliers import OutlierCleaner
from panelary.clean._resolve import EntityResolver, LSHBlocking, resolve_entities
from panelary.clean._split import (
    SplitAwareCV,
    assert_no_straddle,
    purge_near_duplicates,
    straddling_clusters,
    straddling_pairs,
)
from panelary.clean._strsim import string_similarity
from panelary.clean._survivorship import Survivorship, golden_records

__all__ = [
    "Canonicalizer",
    "Deduplicator",
    "EntityResolver",
    "LSHBlocking",
    "OutlierCleaner",
    "SplitAwareCV",
    "Survivorship",
    "assert_no_straddle",
    "connected_components",
    "dedup",
    "exact_unique",
    "fingerprint",
    "fingerprint_clusters",
    "golden_records",
    "near_duplicate_clusters",
    "near_duplicate_pairs",
    "ngram_fingerprint",
    "normalize_text",
    "purge_near_duplicates",
    "resolve_entities",
    "straddling_clusters",
    "straddling_pairs",
    "string_similarity",
]
