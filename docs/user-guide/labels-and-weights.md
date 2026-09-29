# Label spans and sample weights

Financial labels overlap. A 20-day forward-return label on every trading day
shares 19 of its 20 days with its neighbour, so a "5,000-day backtest" does not
hold 5,000 independent observations -- it holds about 240. An unweighted model
over-counts crowded periods, and a sample-size claim that ignores the overlap
overstates the evidence by an order of magnitude.

`panelary.weights` measures that overlap and turns it into sample weights. It
reads one thing: the label end time `t1` that every labeler in
[`panelary.label`](labeling.md) emits. The purged cross-validators purge on the
same `t1`, through the same span table, so labels, weights and purge can never
disagree about which rows a label covers.

```python
import panelary as pn

labels = pn.label.triple_barrier(prices, entity="id", time="date", max_holding=20)

pn.weights.effective_n(labels, t1="t1")          # the honest sample size, sum(ubar)
pn.weights.concurrency(labels, t1="t1")          # + "concurrency": labels active per row
pn.weights.average_uniqueness(labels, t1="t1")   # + "uniqueness": mean 1/c over each span
table = pn.weights.spans(labels, t1="t1")        # the SpanTable itself
```

## The span table

A label at row `(entity, t0)` with end time `t1` covers the closed interval
`[t0, t1]`: the rows of **its own entity** whose time lies in it. A row with a
null `t1`, a row flagged `censored`, or a row whose `t1` lies beyond its
entity's last observation is *not* a resolved label and is dropped (counted in
`SpanTable.n_dropped`). Unresolved labels are never stretched to the end of the
data -- AFML's `t1.fillna(last_date)` makes every weight depend on how long the
sample is.

The table is built with no per-entity loop, on the flat row axis of the
`(entity, time)`-sorted frame, in O(R + N log R).

## Censored labels

`triple_barrier` looks up to `max_holding` rows ahead. For the last
`max_holding` rows of every entity that horizon is truncated: if no barrier is
touched in the rows that exist, the label is reported as a vertical-barrier
`0` although the data cannot support it (the last row is `label=0, t1=t`).
These rows now carry `censored=True`. They are exactly the rows whose label can
still change when more data arrives, and the span table and weights exclude
them by default (`censored="censored"`). Drop them before training:

```python
train_rows = labels.filter(~pl.col("censored"))
```

`fixed_horizon` marks its unresolved tail with a null `t1` instead.

## Concurrency and uniqueness

- **Concurrency** `c_r` -- the number of labels active on row `r` -- is an
  event stream (`+1` at each start, `-1` after each end, one cumulative sum),
  exact in int64.
- **Average uniqueness** `ubar_i = mean_{r in span i} 1 / c_r` is `1` for a
  label nobody overlaps and `1/k` for `k` identical labels.
- **`effective_n = sum_i ubar_i`** is the number of independent labels.

These are **global**: they count every label in the frame. That is right for
describing a labelled panel and for a final refit, and wrong inside
cross-validation, where a weight must be computed from the training fold's
labels only.

## Sample weights

```python
pn.weights.return_attribution(labels, t1="t1", price="close")   # + "w_ret"
pn.weights.time_decay(labels, t1="t1", c=0.5)                    # + "w_decay"
pn.weights.class_weights(labels["label"])                        # {class: weight}
pn.weights.attach(labels, t1="t1", kind="return", price="close",
                  decay=0.5, balance="label", out="w")           # global: final refit only
```

- **Return attribution** (AFML 4.10): `|sum_{r in (t0, t1]} ret_r / c_r|`. A
  label earns the log returns over its span, each shared with the labels active
  on that row. The return *at* `t0` is earned before the event and is excluded;
  `afml_compat=True` sums AFML's inclusive `[t0, t1]`. It already embeds
  concurrency, so `kind="return"` and `kind="uniqueness"` are alternatives,
  never multiplied.
- **Time decay** (AFML 4.11): linear in cumulative uniqueness, newest label `1`,
  oldest about `c`; with `c < 0` the oldest `|c|` share gets `0`. The total
  uniqueness it is anchored to is a *fit*.
- **Class weights**: scikit-learn's `"balanced"`, `n / (K n_k)`, over the
  classes present; `effective=uniqueness` counts class mass in independent
  labels instead of rows.
- **`attach`**: base x decay x class, normalised so the resolved labels'
  weights sum to their number; unresolved rows get `0`.

## Weights inside cross-validation: `FoldWeights`

AFML computes concurrency on the whole sample and then cross-validates. Near a
fold boundary that concurrency counts purged and test labels whose `t1` -- a
barrier-touch time -- depends on test-period prices; the normalisation, the
decay anchor and the class frequencies include the test fold too. The test fold
shapes the training weights.

`FoldWeights` is the same recipe as `attach`, evaluated **per fold on that
fold's labels only**:

```python
fw = pn.weights.FoldWeights(t1="t1", kind="return", price="close", decay=0.5,
                            balance="label")
report = pn.cross_validate(model, labels, y="label",
                           cv=pn.PurgedKFold(5, t1="t1", embargo=5),
                           sample_weight=fw,   # training labels of each fold
                           score_weight=fw)    # test labels of each fold
w_train = fw.compute(labels, train_positions, test_positions=test_positions)
```

- Concurrency is counted on the full row axis but only over the fold's labels.
- The weights reach an sklearn-shaped model as `fit(X, y, sample_weight=w)`,
  and a Panelary model wrapper through a `__w__` column.
- `score_weight` calls `metric(y_true, y_pred, sample_weight=w)`; a metric
  without that parameter raises rather than silently scoring unweighted. The
  default metric becomes the weighted negative MSE.
- A column name is also accepted, as precomputed weights, with a warning: a
  column computed on the full sample is exactly the leak above.
- **Guard**: if a training label's span covers a test time, `compute` raises.
  That is an under-purge -- a splitter without `t1=`, or a null `t1` at a test
  time -- and it would put test prices into both the label and its weight.
- With training blocks on both sides of the test block (CPCV, middle folds of
  `PurgedKFold`) the decay runs across the gap towards the most recent training
  label; a warning says so once.

The tests check each trap with power: perturbing test-fold prices (which moves
triple-barrier `t1`s) leaves every fold-local training weight bitwise unchanged
while the global `attach` / `time_decay` weights move; flipping test labels
leaves fold-local class weights unchanged; appending data leaves the weights of
an earlier fold unchanged.

The positional splitters take the same spans: `cpcv_splits(..., t1=...)` and
`walk_forward_splits(..., t1=...)` accept per-position end times.

## Sequential bootstrap and bagging

An i.i.d. bootstrap of overlapping labels draws near-duplicates. The sequential
bootstrap (AFML 4.5) draws one label at a time with probability proportional to
its average uniqueness **given the labels already drawn**:

```python
rows = pn.sample.sequential_bootstrap(train_labels, t1="t1", seed=0)
rows = pn.sample.sequential_bootstrap(train_labels, t1="t1", stratify="entity")
bag = pn.sample.SequentialBagging(DecisionTreeClassifier(), n_estimators=100,
                                  t1="t1", target="label", features=[...])
```

- `rows` are int64 row indices into the `(entity, time)`-sorted frame, with
  multiplicity, in draw order.
- A draw changes only the labels that overlap it; those are recomputed from
  scratch and sampled through a two-level inverse CDF, instead of rebuilding
  AFML's dense `rows x labels` matrix every draw. Draws match the dense
  definition given the same uniforms (tested on 20 seeds). The numba kernel and
  its numpy twin draw bitwise-identically.
- `stratify="entity"` seeds each entity from `(seed, blake2b(entity))`, so its
  draws survive adding or reordering other entities and are identical across
  processes.
- `method="uniqueness_iid"` draws i.i.d. with `p` proportional to average
  uniqueness: cheaper, but it ignores redundancy among the drawn labels.
- `SequentialBagging` draws from the **fit panel's** spans, so inside
  `cross_validate` the bags come from the training fold only. It offers no
  out-of-bag score: with overlapping labels an out-of-bag label is rarely
  independent of the bag, and OOB accuracy is inflated (AFML ch. 6).

## Event sampling: the CUSUM filter and the tick rule

```python
df.with_columns(
    pn.sample.cusum_filter(pl.col("ret"), threshold="h").over("id").alias("event"),
    pn.sample.tick_rule("price").over("id").alias("side"),
)
```

`cusum_filter` is AFML's symmetric filter: `S+ = max(0, S+ + y)`,
`S- = min(0, S- + y)`, an event when either crosses the threshold, and only the
side that fired resets. It is an event sampler, not a calibrated monitor (for
those see `.ts.cusum` and `detect.page_cusum_expr`). A per-row threshold must
be causal, such as a trailing volatility multiple; the tests show a full-sample
threshold failing `assert_prefix_invariant`. NaN inputs are skipped.
`tick_rule` gives `+1` / `-1` by the sign of the price change, carries zero
ticks forward, and is null until the first move rather than guessing a side.

## The purge

`PurgedKFold(t1=...)` and `CombinatorialPurgedCV(t1=...)` purge a training time
whose label span `[t_j, t1_j]` overlaps any test span `[t_i, t1_i]`. The purge
now runs on the span table's position encoding -- each end time becomes the last
grid position `<= t1`, so the pairwise test collapses to a running maximum and
one `searchsorted` -- and produces **byte-identical** folds to the previous
O(times x test size) loop.

A null `t1` at a *test* time means that test label's span is unknown, so it
cannot purge the training labels that overlap it. Folds are unchanged, but the
splitters now emit a `UserWarning` when a surviving training time is at risk.

## Performance

`benchmarks/bench_labelweights.py` (`--big` adds the 25M-row sizes) on seeded
synthetic data: an Apple M5 Pro (15 cores), CPython 3.13, NumPy 2.5, polars
1.44, numba 0.67, measured 2026-09-29 on a **shared, loaded machine** (load
average 15-25), so treat these as upper bounds. numba compile time is excluded
(first call: under 1 s, then cached on disk).

| Operation | Size | numpy / polars | numba |
|---|---|---|---|
| purge per fold, old loop | 5k / 50k times | 9-17 ms / 358-852 ms | -- |
| purge per fold, span table | 5k / 50k / 500k times | 0.2-0.4 ms / 2.5-3.6 ms / 25-69 ms | -- |
| `_spans_from_t1` | 25M rows | 2.8 s | -- |
| concurrency + uniqueness (kernels) | 25M spans | 0.95 s | -- |
| `weights.effective_n` | 25M rows | 2.4 s | -- |
| `FoldWeights.compute` (return + decay), one fold | 25M rows | 2.0 s | -- |
| sequential bootstrap | N = 5,000 | 91-133 ms | 1.7-2.7 ms |
| sequential bootstrap | N = 100,000 | 1.9-2.3 s | 85-127 ms |
| `sequential_bootstrap(stratify="entity")` | N = 1M, E = 5,000 | 17 s | 0.37-0.41 s |
| CUSUM filter kernel | 1M rows | 115 ms (pure Python) | 2.5 ms |
| `cusum_filter(...).over(id)` | 1M rows, 5,000 entities | 213 ms | (same; per-group dispatch dominates) |

Ranges are over two or three runs. The span-table purge is 46-60x faster than the old
loop at 5,000 times and 127-239x at 50,000; the old loop is O(times x test
size), so at 500,000 times it would take roughly 100x its 50k time (about a
minute per fold) against 25-69 ms. Folds are byte-identical.

Accuracy: uniqueness and return attribution agree with brute-force `math.fsum`
references to `rtol` 1e-12. The block-local span sums keep every rounded partial
sum to at most 64 terms, so at 2M rows their worst relative error was 3e-14,
where a single global prefix sum reached 5.5e-10.
