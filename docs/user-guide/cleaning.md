# Data Cleaning

A fitted statistic is only as honest as the rows it was fit on. Two copies of
one filing count twice in every training mean, quantile and covariance. A
near-copy of a test row sitting in the training fold hands the model the
answer. Three vendor ids for the same company split that company's history
across three entities that `.over(entity)` treats as strangers.

`panelary.clean` handles this stage, before features are built. It covers
canonicalisation, deduplication, survivorship, outlier treatment and entity
resolution. Everything is pure Polars and NumPy. `rapidfuzz` is an optional
speed-up for string similarity; nothing needs it.

```python
from panelary.clean import (
    Canonicalizer, Deduplicator, OutlierCleaner, EntityResolver,
    Survivorship, golden_records, normalize_text, fingerprint,
    near_duplicate_clusters, SplitAwareCV, assert_no_straddle,
    resolve_entities, string_similarity,
)
```

## The ordering rule

Anything deterministic runs first, on every row. Anything that learns a
parameter runs last, fit on training rows only.

1. Integrity gate ([`quality`](data-quality.md)): dtypes, unique
   `(entity, time)` keys, time order. Runs on all rows.
2. **Canonicalise** (`Canonicalizer`). Runs on all rows. It must come before
   deduplication, so that spelling variants collapse onto one key.
3. **Deduplicate** (`Deduplicator`, `EntityResolver`). Runs across the whole
   panel, **before any split**.
4. Constraint checks (`quality`).
5. **Outliers** (`OutlierCleaner`): thresholds fit on train only.
6. Imputation ([`imputation`](imputation.md)): fit on train only, and last.

Deduplicating before the fitted stages matters for two reasons. Duplicate rows
inflate an entity's effective sample, and they skew every stored parameter
fitted after them.

## Canonicalise

`normalize_text` builds one expression. Its steps run in a fixed order:
Unicode normal form, accent stripping, case folding, your regex
`replacements`, punctuation removal, then whitespace collapsing.
`fingerprint` goes further and produces a key-collision key. It normalises
the string, then sorts and de-duplicates its tokens, so word order and
repetition stop mattering. This is the OpenRefine idea, reimplemented from its
published description. `ngram_fingerprint` builds the key from character
n-grams, which absorbs spacing variants.

```python
names = pl.DataFrame({"name": ["  ACME, Corp. ", "corp acme", "Société Générale"]})
names.with_columns(
    normalize_text("name").alias("norm"),
    normalize_text("name", strip_accents=True, strip_punctuation=True).alias("loose"),
    fingerprint("name").alias("key"),
)
```

```text
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ name             ┆ norm             ┆ loose            ┆ key              │
╞══════════════════╪══════════════════╪══════════════════╪══════════════════╡
│   ACME, Corp.    ┆ acme, corp.      ┆ acme corp        ┆ acme corp        │
│ corp acme        ┆ corp acme        ┆ corp acme        ┆ acme corp        │
│ Société Générale ┆ société générale ┆ societe generale ┆ generale societe │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

`Canonicalizer` is the transformer form. It works in two layers:

- **Stateless** (always on): `normalize_text`, plus an optional per-column
  `mapping=` crosswalk such as `{"country": {"united states": "us"}}`. This
  layer learns nothing and is row-local.
- **Learned** (`cluster="fingerprint"` or `"ngram"`): at `fit`, the most
  frequent spelling of each key *in the training rows* becomes its canonical
  form. The learned table is `vocab_`. `transform` rewrites values whose key
  was seen in training. Keys first seen at transform time pass through
  normalised but otherwise unchanged. Nothing is learned from transform-time
  rows.

```python
canon = Canonicalizer(columns="issuer", cluster="fingerprint",
                      strip_punctuation=True, entity="ticker", time="date")
canon.fit(train).transform(panel)
```

If you canonicalise the entity column in place (with no `suffix`), two entity
ids can merge into one. In that case the instance sets `panel_safe = False` at
fit.

`fingerprint_clusters(df, column)` gives the diagnostic view: every distinct
value with its key, count and canonical spelling.

## Deduplicate

`Deduplicator` removes exact and near-duplicate rows, or flags them with
`action="flag"`.

- `method="exact"` catches identical content.
- `"minhash"` and `"cminhash"` catch token-set Jaccard similarity. Rows are
  shingled into cells or into character or word n-grams.
- `"simhash"` catches cosine similarity of standardised numeric rows. Its
  standardisation statistics are fit on the training panel.

MinHash signatures, LSH banding and union-find are all implemented in Polars
and NumPy. By default each LSH candidate is then verified exactly
(`verify="exact"`), so the sketch only proposes pairs and never decides them.

By default, deduplication is **panel-global**: every row is compared with every
other row, across entities and dates. That is what catches a vendor re-sending
yesterday's record, or one filing mapped to two tickers. The content columns
default to every non-key column. This has a consequence: an entity whose
values are genuinely unchanged from one date to the next *is* a duplicate
under the default. Choose the comparison with `scope=`:

- `"entity"`: only rows of the same entity are compared.
- `"time"`: only rows on the same date are compared.
- `"key"`: only rows in the same `(entity, time)` cell are compared. Combine
  it with `method="exact"` to drop only repeated rows.

```python
px = pl.DataFrame({
    "ticker": ["A", "A", "B", "B", "A"],
    "date":   [1, 2, 1, 2, 3],
    "close":  [10.0, 10.5, 20.0, 20.0, 10.5],
    "volume": [100, 110, 300, 300, 110],
})
Deduplicator(method="exact", entity="ticker", time="date", action="flag").fit_transform(px)
```

```text
┌────────┬──────┬───────┬────────┬──────────────┐
│ ticker ┆ date ┆ close ┆ volume ┆ is_duplicate │
╞════════╪══════╪═══════╪════════╪══════════════╡
│ A      ┆ 1    ┆ 10.0  ┆ 100    ┆ false        │
│ A      ┆ 2    ┆ 10.5  ┆ 110    ┆ false        │
│ B      ┆ 1    ┆ 20.0  ┆ 300    ┆ false        │
│ B      ┆ 2    ┆ 20.0  ┆ 300    ┆ true         │
│ A      ┆ 3    ┆ 10.5  ┆ 110    ┆ true         │
└────────┴──────┴───────┴────────┴──────────────┘
```

### Why the default drops only *later* copies

A connected-component dedup reads the future. A row that arrives later can
bridge two earlier clusters. That changes which rows survive in the past, so
the past output depends on data that did not exist yet. The default rule
(`keep="first"`, `linkage="pairwise"`) avoids this. It drops a row only if the
row near-duplicates an **earlier** row, where "earlier" means by time, then by
input order. Each decision depends only on the row and its past, so the
transform is prefix-invariant and `leakage_safe`.

`linkage="component"`, `keep="last"` and an attached `survivorship=` merge all
read the future. Any of them sets the instance's `leakage_safe` to `False`.
Use them on reference data, not inside a backtest.

`near_duplicate_clusters` and `near_duplicate_pairs` are the diagnostic views.
The first gives each row's connected-component `cluster_id`; the second gives
the verified pairs with their similarity.

## No near-duplicate may straddle a split

A purged, embargoed splitter makes sure no *label window* crosses the
train/test boundary. It says nothing about *content*. There are three ways to
close that gap:

- **Deduplicate before you split.** After the point-in-time `Deduplicator`
  runs, no near-duplicate pair survives anywhere in the panel. So none can
  straddle any split you build afterwards.
- **Purge each fold.** `SplitAwareCV(cv, threshold=0.9)` wraps any splitter,
  for example `PurgedKFold` or `CombinatorialPurgedCV`. It uses only the
  splitter's public `split(panel)`, computes the similarity graph once across
  the whole panel, and drops from each fold's training set the rows that
  near-duplicate a test row. `purge_near_duplicates(train, test)` does the
  same for a single split.
- **Assert.** `assert_no_straddle(train, test)` raises on a violation, and
  `straddling_pairs(train, test)` lists the offending pairs.
  `quality.check_near_duplicate_straddle` is the same check reported as a
  `LeakageWarning` (see [Data Quality](data-quality.md)).

## Survivorship: golden records

Once you know rows describe the same thing, a rule has to pick which *value*
of each column survives. `golden_records(df, by=..., order_by=..., rules=...)`
applies the standard master-data rules per column:

| Rule | Keeps |
| --- | --- |
| `first` / `last` | the value on the first / last row, nulls included |
| `first_non_null` / `last_non_null` | the first / last non-null value (the default is `first_non_null`) |
| `most_frequent` | the modal non-null value; ties go to the value seen first |
| `longest` | the longest string, for example the fullest company name |
| `max` / `min` | the column maximum / minimum |
| `most_complete` | the value from the row with the most non-null fields |
| `source_priority` | the non-null value from the most trusted `source_col`; unknown sources rank last |

A callable `col -> pl.Expr` also works as a rule.

```python
s = pl.DataFrame({"cluster": [1, 1, 1], "date": [1, 2, 3],
                  "name": ["ACME", "Acme Corporation", None], "px": [None, 10.0, 11.0]})
golden_records(s, by="cluster", order_by="date",
               rules={"name": "longest", "px": "last_non_null"})
# cluster=1, date=1, name="Acme Corporation", px=11.0
```

A golden record reads every row of its group, including later ones, so it is
**not** point-in-time. Use it for entity or security masters, or attach it to
a `Deduplicator(linkage="component", survivorship=Survivorship(...))`, which
then declares `leakage_safe = False`.

## Outliers

`OutlierCleaner` learns thresholds at `fit`, from the training rows only.
Choose the method with `method=`:

- `"mad"`: median ± k·1.4826·MAD, where k defaults to 3.5 (the
  Iglewicz–Hoaglin cut-off).
- `"iqr"`: Tukey fences, where k defaults to 1.5.
- `"zscore"`: mean ± k standard deviations.
- `"quantile"`: winsorisation limits.

Choose where the thresholds come from with `pooling=`:

- `"entity"`: per entity. Entities with fewer than `min_obs` training rows, and
  entities unseen at fit, fall back to the pooled bounds.
- `"global"`: pooled across the whole panel.
- `"cross_section"`: recomputed from each date's own cross-section. Nothing is
  learned, but entities mix within a date, so this setting is not
  `panel_safe`.

`transform` only reads the stored bounds.

Two rolling methods learn nothing and judge each row against the previous
`window` observations of its own entity:

- `"hampel"`: a trailing Hampel filter (median ± k robust sigmas, with the MAD
  computed exactly).
- `"rolling"`: a trailing z-score.

The current row never enters its own window, and window sizes are constants,
so the output is prefix-invariant.

Choose the action with `action=`:

- `"clip"` winsorises the value.
- `"null"` blanks it.
- `"flag"` adds `{col}_outlier`.
- `"drop"` removes the row.

```python
cleaner = OutlierCleaner(method="iqr", entity="id", time="t").fit(train)
cleaner.transform(test)       # train-fold bounds, applied frozen
```

`preprocessing.scale` and `preprocessing.trim` do **not** share robust
statistics with `OutlierCleaner`. `scale` standardises with per-entity
mean and standard deviation, and `trim` aligns the entities' *time ranges*.
Neither computes a robust statistic, so their behaviour is unchanged.

## Entity resolution

"AAPL US Equity", "Apple Inc" and "APPLE INC." may arrive under three vendor
ids. `resolve_entities` maps every id to a canonical id, in four steps:

1. **Normalise** string fields.
2. **Block**, so that only records sharing a key are compared. Blocking keys
   can be:
    - exact keys: a column, or any Polars expression, such as a name prefix or
      an exchange code;
    - `neighbourhood=(key, window)`: a sorted-neighbourhood window;
    - `LSHBlocking(field, threshold=...)`: MinHash-LSH on character shingles,
      which catches typos and reordered words.

   Blocks are OR-ed. With no blocking at all, every pair is compared, subject
   to the `max_pairs` guard.
3. **Compare** fields with `string_similarity`. The metrics are
   `levenshtein`, `jaro`, `jaro_winkler`, `token_jaccard` and `exact`, plus
   `numeric`, a relative tolerance. Scores are combined as a weighted mean
   over the fields both records actually have.
4. **Cluster** matched pairs with union-find, then choose each cluster's id by
   `canonical="first" | "most_frequent" | "min"`.

```python
recs = pl.DataFrame({
    "vendor_id": ["AAPL US", "APPLE", "aapl.o", "MSFT", "Microsoft Corp"],
    "name": ["Apple Inc", "APPLE INC.", "Apple Inc", "Microsoft", "Microsoft Corp"],
})
resolve_entities(recs, id_col="vendor_id", fields={"name": "jaro_winkler"}, threshold=0.9)
```

```text
┌────────────────┬─────────────┬──────────────┐
│ vendor_id      ┆ resolved_id ┆ cluster_size │
╞════════════════╪═════════════╪══════════════╡
│ AAPL US        ┆ AAPL US     ┆ 3            │
│ APPLE          ┆ AAPL US     ┆ 3            │
│ aapl.o         ┆ AAPL US     ┆ 3            │
│ MSFT           ┆ MSFT        ┆ 2            │
│ Microsoft Corp ┆ MSFT        ┆ 2            │
└────────────────┴─────────────┴──────────────┘
```

`EntityResolver` re-keys a panel's entity column, or writes the result to
`output=`.

- At `fit` it resolves the training panel's distinct `(entity, fields...)`
  records and stores the map (`mapping_`) and the training records
  (`records_`).
- At `transform`, ids known from training go through the map. An id unseen at
  fit is compared **only against the stored training records**, using the
  same blocking and threshold. It adopts the best match's resolved id, or
  keeps its own.
- Unseen ids are never merged with each other, because that would be
  learning at transform time.

Re-keying can produce repeated `(entity, time)` rows. Follow it with
`Deduplicator(scope="key")` or a survivorship merge.

`string_similarity(a, b, metric=...)` scores aligned pairs in `[0, 1]`. A
missing value scores `NaN`, so the field is left out of a weighted comparison
rather than counted as "different". The `backend` option controls the engine:

- `"auto"` (the default) uses `rapidfuzz` when it is installed
  (`pip install 'panelary[fuzzy]'`).
- Otherwise it uses the built-in fallback: a vectorised NumPy edit distance
  and a pure-Python Jaro–Winkler.
- `backend="rapidfuzz"` requests the extra explicitly, and raises an
  actionable `ImportError` if it is missing.

Probabilistic Fellegi–Sunter linkage is out of scope here. It is deferred to an
optional `splink` backend.

## Contracts at a glance

| Transformer | `panel_safe` | `leakage_safe` | Learned state |
| --- | --- | --- | --- |
| `Canonicalizer` | `True` (`False` if it rewrites the entity key) | `True` | `vocab_` (with `cluster=`) |
| `Deduplicator` | `True` for `scope="entity"`/`"key"`, else `False` | `True` only for `keep="first"`, `linkage="pairwise"`, no `survivorship` | SimHash `center_`/`scale_` |
| `OutlierCleaner` | `False` only for `pooling="cross_section"` | `True` | `bounds_`, `global_bounds_` |
| `EntityResolver` | `False` (merging ids combines entities) | `True` | `mapping_`, `records_` |

Every learned state is fit on the rows passed to `fit`. The test suites back
each claim with a deliberately leaky variant that must fail. One variant fits
on all rows; the other re-learns at transform time.

`preprocessing.reindex(drop_duplicates=True)` now takes the unique entity and
time values through `panelary.clean.exact_unique`. Its behaviour is unchanged.
For row-level deduplication, use `Deduplicator`.
