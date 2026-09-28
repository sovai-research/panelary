# PanelFrame

`panelary.core.panel_frame` defines **`PanelFrame`**, the type every other part of Panelary
speaks. A *panel* is a long-format table indexed by an **entity** (a security, a customer, a
country) and a **time** axis; each row is one observation of one entity at one time, and the
remaining columns are features.

`PanelFrame` is a **view**, not a container. It wraps a `polars.LazyFrame` and remembers which
column is the entity and which is the time. It never copies data and stays lazy until you call
`collect()`. Construction validates only the *schema* — that both key columns exist, are
distinct, and that the time column has an orderable dtype — so wrapping a frame is cheap
enough to do freely.

Read the leakage contract precisely: **a `PanelFrame` guarantees the keys are valid; it does
not by itself prevent look-ahead.** What it provides is the machinery that makes leak-safety
expressible and checkable — `over_entity()` for within-entity work, `sort_panel()` for
deterministic ordering, and a remembered `(entity_col, time_col)` pair so every downstream
transform, splitter and verifier agrees on what "the past" means without being told again.

## What's here

| Your problem | Entry point |
| --- | --- |
| Wrap a frame, declaring its keys | `PanelFrame(data, entity=..., time=...)` |
| Accept "a frame or a panel" in your own API | `as_panel` |
| Which columns are keys, which are features? | `entity_col`, `time_col`, `feature_cols`, `columns`, `schema` |
| Per-entity computation in time order | `over_entity`, `group_by_entity` |
| Deterministic `(entity, time)` ordering | `sort_panel`, `is_sorted_per_entity` |
| Inspect the panel's axes | `entities`, `n_entities`, `time_index` |
| Assert one row per `(entity, time)` | `assert_unique_keys` |
| Lazy frame operations, keys preserved | `select`, `with_columns`, `filter`, `pipe` |
| Join a vintage table **as it was known** at each row's time | `asof_join` |
| Drop back to Polars | `collect`, `lazy`, `to_frame`, `to_native` |
| Single-key accessors | `entity`, `time` |

## Why a view and not a subclass

Subclassing `polars.LazyFrame` would make every Polars method silently return the wrong type
and would couple Panelary to Polars internals. Wrapping instead keeps one rule: operations that
*preserve* the panel contract (`select`, `with_columns`, `filter`, `pipe`, `sort_panel`) return
a `PanelFrame` with the keys carried through, and operations that leave the panel world
(`collect`, `to_native`) hand you plain Polars. `validate=False` exists for the internal case of
re-wrapping a frame whose keys were already checked.

## Point-in-time joins: `asof_join`

Fundamentals, estimates and macro series arrive as **vintages**: a value is
*about* an `event_time` (the fiscal period) and became *known* at a
`knowledge_time` (the filing), and may later be restated. Joining on the period,
or taking the latest vintage per period, uses figures nobody had on the date
being simulated. `panelary.core.asof.asof_join(panel, vintages, ...)` — also
`PanelFrame.asof_join(vintages, ...)` — gives each panel row `(entity, t)` the
vintage with the latest `event_time`, and for that period the latest
`knowledge_time`, among those with `knowledge_time + lag <= t` **and**
`event_time <= t`:

```python
from panelary.core.asof import asof_join

pit = asof_join(
    prices,                      # PanelFrame / DataFrame / LazyFrame
    fundamentals,                # (ticker, event_time, knowledge_time, eps, ...)
    entity="ticker", time="date",
    lag="1bd",                   # a filing is usable from the next business day
    provenance=True,             # also attach which vintage each value came from
)
```

The contract: **strictly causal** (the value at `t` depends only on vintages
eligible by `t`), **prefix-invariant**, **order-independent** (neither frame needs
sorting; rows come back in the panel's order), and **fail-closed** — null
vintage keys, conflicting duplicate vintages, and `Date`-vs-`Datetime` clock
mismatches are errors, not guesses. A restatement is invisible before it was
published and returned from then on; a restatement of an *older* period never
displaces a newer period. `lag` takes the same duration vocabulary as the
calendar embargo (`int` steps on an integer axis; `"1d"`, `timedelta`, `"2bd"`,
`BusinessDays` on a temporal one). The column roles default to `event_time` /
`knowledge_time` — the shape of the `vintages` table `panelary.synth`'s
`generate_panel` emits.

## Stability

`PanelFrame` is **not** marked experimental in the source; the only "experimental" marker in
the repository is a historical line in `CHANGELOG.md` under `[0.2.0]` (`Added — panel core
(shipped in-repo, experimental)`), describing when the panel core first landed in a release
that was never published to PyPI. That said, Panelary is still `0.x`: the constructor contract
above (entity/time validation, laziness, `feature_cols`) is settled and widely depended on
internally, but no formal stability guarantee applies before `1.0.0`.

## See also

- [Panel data](../concepts/panel-data.md) — what a panel is and why the shape matters.
- [The two-tier API](../concepts/two-tier-api.md) — where `PanelFrame` sits in the stack.
- [Leak-safety](../concepts/leak-safety.md) — the contract `PanelFrame` makes expressible.
- [`registry`](registry.md) — the catalogue of operators that consume panels.

## API

::: panelary.core.panel_frame

::: panelary.core.asof
