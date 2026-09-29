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
