# Leak verifiers

Mechanical leak verifiers — the keystone of Panelary's leak-safety guarantee. A
transform is *leak-free* if its output at time `t` depends only on data at times
`<= t` (within each entity). There are two independent ways to break that, so
there are two independent checks.

**Future perturbation** catches *value*-dependence: run the operation, corrupt
every value strictly in the future (per entity), run it again, and assert every
output cell in the past is bit-identical. Any difference is a look-ahead.

**Prefix invariance** catches *length*-dependence: truncate the panel to
`time <= cut`, re-run, and assert every output cell in the truncated region
equals the full-panel run. This is hard invariant 1 — `f(x[:T])[t] ==
f(x[:T+k])[t]` — stated operationally. Perturbation cannot see this class of
defect at all: corrupting future *values* leaves the *row count* untouched, so
a quantity derived from `len(x)` is bit-identical under any perturbation and
sails through.

## What's here

- `assert_no_lookahead` — split the shared time axis at a cut `t` and assert that
  perturbing `time > t` never changes any output at `time <= t`.
- `assert_no_train_test_leak` — perturb a *test* fold and assert the *train*-fold
  outputs are unchanged (the CV-boundary version of the same idea).
- `assert_prefix_invariant` — truncate the panel at a cut and assert the output
  over the surviving rows is unchanged.

`assert_no_lookahead` and `assert_prefix_invariant` test **eight cuts by
default**: a consecutive pair at 20%, 40%, 60% and 80% of the time axis. One
cut is not a test. A leak confined to a calendar period, such as a period mean
broadcast back to its rows, is invisible at a cut on a period's last step: on 250
dates with 5-step periods it passed the old single median cut and failed at 200
of the other 249. Of two consecutive cuts, at most one can end a period longer
than one step. Pass `cut=` for one cut or `cuts=[...]` for your own list; each
cut costs one more run of the operation.

All three accept `op` as either a `polars.Expr` or a callable `frame -> frame`,
and failures name the first offending `(column, entity, time)`.

## Neither assertion subsumes the other

A `series -> scalar` aggregate broadcast back over an entity is the case that
makes this concrete. `ts.count_above()` returns the percentage of a series at or
above a threshold; written per-row with `.over(entity)`, every row gets the whole
entity's answer — including rows that had not happened yet.

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

assert_no_lookahead(expr, panel)        # passes — the future's *values* are irrelevant
assert_prefix_invariant(expr, panel)    # raises  — the future's *count* is not
```

```text
AssertionError: PREFIX-INVARIANCE VIOLATION: truncating the panel to t <= 5
changed output column 'pct' at ...
```

The perturbation test is right that no future *value* reaches row 0; it is the
future *row count* that does. Run both on any new operator.

## See also

- [Leakage & Correctness](../leakage.md) — the project-wide correctness contract.
- [Leak-safety](../concepts/leak-safety.md) — the `panel_safe` / `leakage_safe`
  declarations, and why `safe_scope` has to name the usage they are claimed under.
- [Feature registry](registry.md) — `safe_scope`, the declaration these
  assertions check.

## API

::: panelary.testing
