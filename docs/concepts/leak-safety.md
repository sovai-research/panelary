# Leak-safety: the conceptual foundation

> **Leak-safety is Panelary's moat.** Everything else is a convenience; this is the
> reason the library exists.

A model that scores brilliantly in research and dies in production has almost
always been *fed the future*. Panelary is designed so that the future cannot get
in — not by convention or code review, but by the shape of the API and by
mechanical verifiers you can run on any transform. This page explains what leakage
is in panel data, the two independent axes it travels along, exactly how Panelary
closes each one, and why the safety of an operator is a claim about *how it is
used* rather than about the operator alone.

## What is a panel, and what is leakage?

A **panel** is many entities observed over time: stocks by day, customers by
month, sensors by minute. Every row is one observation of one **entity** at one
**time**, and the features live in the remaining columns. Panelary models this
directly as a [`PanelFrame`][panelframe] — a lazy view that remembers which
column is the entity key and which is the time key.

**Leakage** is any path by which information the model would not have had at
prediction time flows into a feature, a label, or a fitting decision. The output
at time `t` is supposed to depend only on data available *at or before* `t`. When
it depends on anything later — even subtly — the backtest is measuring
information the strategy will never actually have.

## The two leak axes

Panel data leaks along **two independent axes**, and a correct library must close
both.

### Axis 1 — Temporal (per-entity look-ahead)

Information from time `t+k` influences a value at time `t`, *within the same
entity*. Classic culprits:

- A misaligned shift that pulls the future backwards (`shift(-1)` instead of
  `shift(1)`).
- A rolling or aggregate window that includes the current or a future row.
- A label whose horizon overlaps the features without being purged.

### Axis 2 — Cross-sectional (cross-entity)

Information from *other entities*, or from the whole sample, contaminates a value:

- A cross-sectional rank or z-score computed **over the entire sample** instead of
  *per date* — so an entity's score at one date depends on entities' values at
  other dates.
- A per-entity window silently computed **across the whole frame**, so one
  entity's history bleeds into the next.
- A global scaler, winsorizer, or feature selector fit on data that includes the
  test period.

The two axes are orthogonal. A `shift(1)` is temporally safe but says nothing
about cross-entity bleed; a per-date rank is cross-sectionally scoped but says
nothing about the future. Panelary tracks both with two explicit flags on every
operator.

## The `panel_safe` / `leakage_safe` contract

Every operator Panelary ships is registered as a
[`FeatureSpec`][featurespec] in a process-wide registry, and each spec carries two
booleans that map exactly onto the two axes:

| Flag | Meaning | Axis |
| --- | --- | --- |
| `leakage_safe` | The operator is **causal**: its value at time `t` never uses data after `t`. | Temporal |
| `panel_safe` | The operator does not bleed information **across entities** when applied per entity. | Cross-sectional |

These are not aspirational labels. They are backed by construction (see below),
audited (`registry.audit()` flags any operator missing either flag or carrying a
non-permissive license), and — for the temporal axis — independently checkable
with [`assert_no_lookahead`](#how-assert_no_lookahead-works) and
[`assert_prefix_invariant`](#the-second-instrument-assert_prefix_invariant).

They are also, on their own, not quite enough: `leakage_safe=True` does not say
*evaluated how*, and for a whole namespace of Panelary's operators the answer
matters. The next section is that gap, and `safe_scope` is the field that closes
it.

```python
import panelary  # registers the .panel / .xs namespaces + specs
from panelary import registry

for name in ["frac_diff", "zscore", "rank", "demean"]:
    sp = registry.get(name)
    print(f"{sp.qualified_name}: panel_safe={sp.panel_safe} leakage_safe={sp.leakage_safe}")
```

```text
panel.frac_diff: panel_safe=True leakage_safe=True
panel.zscore: panel_safe=True leakage_safe=True
xs.rank: panel_safe=False leakage_safe=True
xs.demean: panel_safe=False leakage_safe=True
```

Read `xs.rank` carefully: it is `leakage_safe=True` (it only ever looks at one
date) but `panel_safe=False` — *by design*. A cross-sectional rank **must** mix
entities within a date; that is the whole point. The flag is honest about it
rather than pretending the op is something it isn't.

## Leak-safety is a property of an operator *under a usage*

Two booleans are still not enough, and the reason is worth dwelling on, because it
is the most common way a correctness claim goes quietly wrong.

**An operator is not leak-safe or leaky. A *use* of an operator is.** The same
lines of code can be both, and which one you get depends on how the result is
consumed — something the operator itself cannot see and a boolean attached to it
cannot express.

The `.ts` namespace is the whole case in one place. All 42 `.ts` operators are
`series -> scalar`: they consume a series and return one number. Fed a completed
window — which is exactly what
[`extract_features`](../api-reference/feature-extractors.md) does — such an
aggregate is perfectly causal. It is a summary of data that has already happened.
Written the way the expression namespace invites, it is not:

```python
pl.col("x").ts.count_above().over("ticker")   # one number, broadcast to every row
```

`.over("ticker")` does not window anything. It partitions, computes the aggregate
over the **entire** partition, and broadcasts that single number back to every row
in it — including the rows at the beginning, which now carry a statistic computed
from observations that had not happened yet.

### The measurement

`ts.count_above()` returns the percentage of a series at or above a threshold
(default `0.0`). Take one entity, and ask what the row at `t=0` reports as the
series grows from 12 observations to 24:

```python
import numpy as np
import polars as pl
import panelary  # noqa: F401  — registers the .ts namespace

x = np.random.default_rng(0).standard_normal(24)
expr = pl.col("x").ts.count_above().over("e").alias("pct_above")

for n in (12, 24):
    df = pl.DataFrame({"e": ["A"] * n, "t": range(n), "x": x[:n]})
    print(n, round(df.with_columns(expr)["pct_above"][0], 2))
```

```text
12 58.33
24 45.83
```

Nothing about the first 12 rows changed. Twelve more observations arrived *after*
them, and the value the model would have read at `t=0` moved by 12.5 points. Every
row of that column is a number from the future.

`count_above` is not special. Run all 42 `.ts` operators through
`assert_no_lookahead` written as `.over(entity)` and, on one two-entity panel of
30 normal draws each, **27 of the 42 fail outright**. Six more of the fifteen
survivors are caught by [`assert_prefix_invariant`](#the-second-instrument-assert_prefix_invariant).
The last nine pass both — *vacuously*: on continuous random draws `has_duplicate`
is `False` and `ratio_n_unique_to_length` is `1.0` whatever you do to the data.

The exact count moves with the panel you test on — across four seeds it ranged
from 24 to 32 — and that variability is the real lesson. A look-ahead that is
forced by the operator's *shape* should not be something a test can miss by luck.
That is why the verdict is recorded as a declaration rather than discovered by a
verifier.

### `safe_scope` names which one you are claiming

So a `FeatureSpec` records not just *whether* it is `leakage_safe` but the usage
under which that is claimed, in
[`safe_scope`](../api-reference/registry.md#safe_scope-the-usage-a-safety-claim-is-made-under):

- **`"rowwise"`** — safe evaluated at every row: the value at `t` uses only data at
  `<= t`. Expanding and trailing-window operators qualify, and broadcasting with
  `.over(...)` is fine. The `.panel` and `.xs` namespaces are `rowwise`.
- **`"window"`** — safe only as a summary of an already-delimited window.
  Broadcasting the result per-row is a look-ahead **by construction**. All 42 `.ts`
  operators are `window`.

Read the `window` verdict as a statement about shape, not about quality. A
`series -> scalar` aggregate is not a broken operator; `extract_features` uses
these 42 correctly every day. What is broken is the sentence "`ts.count_above` is
leakage-safe" with the usage left out.

The compiler in [point-in-time compilation](point-in-time.md) is the other half of
this: it treats a registered `FeatureSpec` as a **trusted leaf** and reads the
declaration rather than the callable, which is only defensible if the declaration
says what it is claiming.

## How Panelary prevents each leak by construction

### Mandatory `over` keys close the cross-sectional axis

The `.panel` and `.xs` namespaces are the two Polars-native operator families,
and they enforce the axis discipline at the frame level by **requiring an `over`
key**.

- `.panel.*` operators are per-entity time-series transforms. On a bare frame they
  demand `over=<entity>`. Omit it and you get a hard error — because a per-entity
  op computed across the whole frame would bleed one entity's history into the
  next (an Axis-2 leak).
- `.xs.*` operators are cross-sectional. They demand `over=<time>` — a
  cross-sectional operation is undefined without a cross-section.

```python
import polars as pl
import panelary  # noqa: F401

df = pl.DataFrame({
    "ticker": ["A", "A", "A", "B", "B", "B"],
    "date":   [1, 2, 3, 1, 2, 3],
    "close":  [10.0, 11.0, 12.0, 20.0, 19.0, 21.0],
})

# SAFE: trailing z-score within each ticker, in date order.
safe = df.panel.zscore("close", window=2, over="ticker", suffix="_z")

# LEAKY (and refused): no entity key -> would compute across every ticker at once.
try:
    df.panel.zscore("close", window=2)
except ValueError as e:
    print(str(e)[:70])
```

```text
panel.zscore requires an `over` entity key (e.g. over='ticker'). Witho
```

The [expression form][panel-ns] (`pl.col("close").panel.zscore(2).over("ticker")`)
leaves grouping to you, so you can compose it however you need; the frame-level
form is the guard-railed default that cannot be called wrong.

### Causal windows close the temporal axis

Every `.panel` operator is a **causal kernel**: the value at row `t` is a function
of `x[t], x[t-1], …` only. `zscore` uses a trailing rolling mean/std; `frac_diff`
is a fixed-width convolution of lagged terms with leading rows set to `null` until
the window fills; `rs_vol` is a trailing standard deviation. None of them can see
a future row, so `leakage_safe=True` holds regardless of how you group.

Cross-sectional `.xs` operators are causal for free: they only ever touch one
timestamp's slice, so no future information is even in scope. That is why they are
uniformly `leakage_safe=True`.

### `t1` label spans + purge/embargo close the label axis

A label almost always looks *forward* — a triple-barrier or fixed-horizon label
resolves somewhere in `[t, t1]`. That forward span is itself a leakage vector: if
a training row's `[t, t1]` overlaps a test row's, the two share outcome
information across the fold boundary. Panelary's labelers emit an explicit **`t1`
column** (the event-end timestamp, in the same dtype as the time axis), and the
cross-validators consume it to **purge** overlapping training rows and **embargo**
a buffer after each test block. The mechanics live in the practical guide,
[Leakage & correctness-by-construction](../leakage.md).

## How `assert_no_lookahead` works

Flags and construction arguments are only as trustworthy as your ability to
*check* them. Panelary's keystone check is a **future-perturbation experiment** —
a model-agnostic test that any temporal leak must fail:

1. Run the operation on a panel and record its output.
2. Corrupt **every value strictly in the future** (past a cut time), leaving the
   past untouched.
3. Run the operation again.
4. If the operation is leak-free, every output cell **in the past** must be
   bit-identical (within `tol`) across the two runs. Any difference means
   information from a perturbed future row flowed backwards — a look-ahead.

The perturbation is deliberately violent (a ~1e6 random offset), so any leak
shows up far above tolerance. On failure the assertion names the first offending
`(column, entity, time)`.

```python
import polars as pl
from panelary import PanelFrame, assert_no_lookahead

df = pl.DataFrame({
    "ticker": ["A", "A", "A", "B", "B", "B"],
    "date":   [1, 2, 3, 1, 2, 3],
    "close":  [10.0, 11.0, 12.0, 20.0, 19.0, 21.0],
})
panel = PanelFrame(df, entity="ticker", time="date")

# A trailing lag is causal -> passes silently.
assert_no_lookahead(pl.col("close").shift(1).over("ticker").alias("lag"), panel)

# A forward lag reads t+1 -> caught.
try:
    assert_no_lookahead(pl.col("close").shift(-1).over("ticker").alias("lead"), panel)
except AssertionError as e:
    print(str(e)[:52])
```

```text
LOOK-AHEAD LEAK DETECTED: perturbing the future (dat
```

`op` may be a `polars.Expr` (applied via `with_columns`) or a callable
`frame -> frame` (a `PanelFrame`, `DataFrame`, or `LazyFrame` in and out — the
calling convention is auto-detected), so you can wrap an entire feature function,
not just a single expression.

The cross-validation-boundary form,
[`assert_no_train_test_leak`][testing], applies the same mechanism to a
`(train, test)` split: it perturbs the *test* fold and asserts the *train*-fold
outputs are unchanged. Use it with walk-forward splits (train entirely before
test) or against a purged train set — for an interior test block even a correct
backward-looking feature on a later train row legitimately depends on test-period
values, which is precisely why purging exists.

## The second instrument: `assert_prefix_invariant`

Perturbation has a blind spot, and it is exactly the one the `.ts` case walks
into. The experiment corrupts future **values**; it leaves the future **row
count** alone. An operator whose output at `t` moves because more rows exist after
`t` — a window, threshold, lag order or normalisation constant derived from
`len(x)`, or a whole-series aggregate broadcast back over the entity — is
bit-identical under any value perturbation and passes silently.

That is the second hard invariant, **prefix invariance**: `f(x[:T])[t] ==
f(x[:T+k])[t]` for all `t <= T`. Its assertion truncates the panel instead of
corrupting it.

```python
import numpy as np
import polars as pl
from panelary import PanelFrame
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

x = np.random.default_rng(0).standard_normal(24)
panel = PanelFrame(
    pl.DataFrame({"e": ["A"] * 24, "t": range(24), "x": x}),
    entity="e", time="t",
)
expr = pl.col("x").ts.count_above().over("e").alias("pct")

assert_no_lookahead(expr, panel)        # passes
assert_prefix_invariant(expr, panel)    # raises
```

```text
AssertionError: PREFIX-INVARIANCE VIOLATION: truncating the panel to t <= 5
changed output column 'pct' at e='A', t=0. ...
```

The perturbation test is not wrong. No future *value* reaches row 0 — the
percentage is a count, and the offending observations were above the threshold
before and after being shifted by a million. It is the future *row count* that
reaches back, and only truncation can see it.

**Neither assertion subsumes the other**, so run both on any new operator.
Perturbation catches value-dependence; prefix invariance catches
length-dependence. A declaration of `safe_scope="rowwise"` is a claim against
both.

## The mental model

| Operation kind | Safe scope | Panelary expression | Flags |
| --- | --- | --- | --- |
| Within-entity time-series transform | one entity, time-ordered | `df.panel.*(..., over=entity)` | `panel_safe`, `leakage_safe`, `safe_scope="rowwise"` |
| Cross-sectional comparison | one timestamp, across entities | `df.xs.*(..., over=time)` | `leakage_safe`, `safe_scope="rowwise"` (mixes entities by design) |
| Series characteristic | one **completed** window, summarised | `extract_features(...)` — not `.ts.*().over(entity)` | `leakage_safe`, `safe_scope="window"` |
| Forward-looking label | explicit `[t, t1]` span | `label.triple_barrier(...)` → `t1` column | n/a — the span is the contract |
| Validation | purged + embargoed folds | `PurgedKFold` / `CombinatorialPurgedCV` | n/a — consumes `t1` |

If an operation can't be expressed within one of these scopes, that's the tooling
telling you it would leak.

## See also

- [Point-in-time compilation](point-in-time.md) — the other half: what to do about
  code that was never built this way and declared nothing.
- [Leak verifiers](../api-reference/testing.md) — `assert_no_lookahead`,
  `assert_prefix_invariant`, `assert_no_train_test_leak`.
- [Feature registry](../api-reference/registry.md) — `safe_scope` and the
  operator catalogue.
- [Leakage & correctness-by-construction](../leakage.md) — the practical guide:
  purge, embargo, CPCV, and the backtest-overfitting statistics.
- [Quickstart](../quickstart.md)
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

[panelframe]: ../api-reference/panel-frame.md
[featurespec]: ../api-reference/registry.md
[panel-ns]: two-tier-api.md#tier-1-bare-frame-namespaces-panel-xs-ts
[testing]: ../api-reference/testing.md
