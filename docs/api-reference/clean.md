# clean

Panel-aware data cleaning: canonicalisation, deduplication, survivorship,
outlier treatment and entity resolution. It is pure Polars and NumPy, with
`rapidfuzz` as an optional speed-up. See the
[Data Cleaning guide](../user-guide/cleaning.md) for the walkthrough and the
order the stages run in.

## What's here

**Canonicalisation.**

- `normalize_text`: one Polars expression for Unicode form, case, accents,
  punctuation, whitespace and ordered regex replacements.
- `fingerprint` and `ngram_fingerprint`: key-collision keys.
- `fingerprint_clusters`: the distinct values of a column, grouped by key.
- `Canonicalizer`: the transformer form. With `cluster=`, it learns the
  canonical spelling of each key from training rows only.

**Deduplication.**

- `Deduplicator` and `dedup`: exact, MinHash, C-MinHash and SimHash with LSH
  and exact verification. Panel-global and point-in-time by default.
- `near_duplicate_clusters`: the connected-component view (`row_index`, keys,
  `cluster_id`, `cluster_size`). `panelary.quality` builds on it.
- `near_duplicate_pairs`: the verified pairs, with their similarity.
- `exact_unique`: null-equal exact de-duplication of a frame.
- `connected_components`: union-find over an edge list, in NumPy.

**Split-awareness.**

- `straddling_pairs` and `assert_no_straddle`: measure, or assert, that no
  near-duplicate pair crosses a train/test split.
- `straddling_clusters`: the same check from a precomputed cluster table.
- `purge_near_duplicates`: drops training rows that near-duplicate a test row.
- `SplitAwareCV`: wraps any splitter and purges every fold it yields.

**Survivorship.**

- `Survivorship` and `golden_records`: per-column golden-record rules
  (`first`, `last`, `first_non_null`, `last_non_null`, `most_frequent`,
  `longest`, `max`, `min`, `most_complete`, `source_priority`, or a callable).

**Outliers.**

- `OutlierCleaner`:
    - `mad`, `iqr`, `zscore` and `quantile` thresholds, fit on train, per
      entity, pooled, or per date;
    - causal `hampel` and `rolling` filters;
    - `clip`, `null`, `flag` and `drop` actions.

**Entity resolution.**

- `resolve_entities`: blocking, field comparison and union-find, returning
  `id -> resolved_id`.
- `EntityResolver`: the transformer form. Ids unseen at fit are matched only
  against the stored training records, and never merged with each other.
- `LSHBlocking`: MinHash-LSH blocking on character shingles.
- `string_similarity`: `levenshtein`, `jaro`, `jaro_winkler`, `token_jaccard`
  and `exact`. It uses `rapidfuzz` when installed (`pip install
  'panelary[fuzzy]'`) and a NumPy / pure-Python fallback otherwise.

## Contracts

Every transformer subclasses `PanelTransformer` and states `panel_safe` and
`leakage_safe`. When an option changes the answer, the instance's own values
are updated to match. For example, `Deduplicator(linkage="component")` sets
`leakage_safe = False`, and `OutlierCleaner(pooling="cross_section")` sets
`panel_safe = False`. The guide has the
[full table](../user-guide/cleaning.md#contracts-at-a-glance).

`preprocessing.scale` and `preprocessing.trim` compute no robust statistic,
so they share nothing with `OutlierCleaner` and are unchanged.
`preprocessing.reindex(drop_duplicates=True)` now runs through
`exact_unique`, with the same behaviour as before.

## API

::: panelary.clean
