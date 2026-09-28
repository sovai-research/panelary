# Feature registry (`FeatureSpec`)

`panelary.registry` is the single source of truth for **operators**: the metadata layer that
turns a loose collection of Polars expressions into an auditable, machine-introspectable
catalogue. Two objects do the work. `FeatureSpec` is a frozen, fully typed description of one
operator, and `FeatureRegistry` is the process-wide singleton (exported as `registry`) that
collects them.

A `FeatureSpec` carries more than a call signature. It records the **provenance and safety
contract**: which expression namespace the operator lives in, its input and output shape, its
`panel_safe` / `leakage_safe` / `safe_scope` declarations, where the implementation came from
and under what license. That last pair is how Panelary demonstrates it is an independent,
permissively licensed implementation rather than a copy of a copyleft upstream — `audit()` is
a mechanical check, not a claim in a README.

The registry is deliberately **decoupled from the implementations**. Namespaces register their
specs at import time via `register_feature`, but `panelary/registry.py` imports nothing from
the rest of Panelary, which keeps it cheap and side-effect-free to import from anywhere.

## What's here

| Your problem | Entry point |
| --- | --- |
| Describe one operator, with provenance and safety | `FeatureSpec` |
| Register an operator as an import side effect | `register_feature` |
| The process-wide catalogue | `registry` (a `FeatureRegistry`) |
| Look one operator up by name | `registry.get(name)` |
| Every spec, or every spec in one namespace | `registry.all()`, `registry.by_namespace(ns)` |
| Which namespaces exist? | `registry.namespaces()` |
| Serialise the catalogue | `registry.to_records()`, `registry.to_llms_txt()` |
| Check for missing safety metadata or copyleft | `registry.audit()` |
| Reset (tests only) | `registry.clear()` |
| The permissive-license allowlist and the tier vocabulary | `PERMISSIVE_LICENSES`, `VALID_TIERS` |
| The vocabulary of safety scopes | `VALID_SAFE_SCOPES` |

## What is registered today

`registry.namespaces()` returns **four** namespaces holding **56** operators in total:

| Namespace | Count | Scope | Examples |
| --- | ---: | --- | --- |
| `.ts` | 42 | Per-series time-series characteristics | `absolute_energy`, `max_drawdown`, `longest_winning_streak`, `cid_ce` |
| `.xs` | 7 | Cross-sectional, per date | `cs_zscore`, `demean`, `neutralize`, `quantile_bin`, `rank`, `standardize`, `winsorize` |
| `.factor` | 4 | Signal evaluation across the cross-section | `forward_return`, `ic`, `orthogonalize`, `portfolio_sort` |
| `.panel` | 3 | Per-entity, causal | `frac_diff`, `rs_vol`, `zscore` |

The `.factor` namespace is easy to miss — it is registered alongside the other three and is
the expression-level counterpart to [`panelary.factor`](factor.md).

## `safe_scope`: the usage a safety claim is made under

`leakage_safe` is a boolean answer to a question that has no boolean answer: *safe
evaluated how?* A `series -> scalar` aggregate such as `ts.absolute_energy` is perfectly
causal as a summary of a completed window — which is what
[`extract_features`](feature-extractors.md) does with it — and a look-ahead the moment it is
broadcast back over an entity with `.over(entity_col)`, because the value at every row is then
a function of the entity's whole series, its future included. Same code, two verdicts. A
`FeatureSpec` therefore names the usage its `leakage_safe` claim is made under, in
`safe_scope`, one of `VALID_SAFE_SCOPES`:

| `safe_scope` | Means | Read it as |
| --- | --- | --- |
| `"rowwise"` | The value at `t` uses only data at `<= t` when evaluated at **every row**. Expanding and trailing-window operators qualify. | Safe to broadcast with `.over(...)`. |
| `"window"` | The claim holds only for a **summary of an already-delimited window**. Broadcasting the result per-row is a look-ahead *by construction* — provable from the shape, so no test is needed and none should be trusted to find it. | Safe inside `extract_features`; not safe per-row. |
| `"unspecified"` | Not yet classified. Permitted so third-party specs keep working. | Panelary's own specs are held to a stricter standard. |

The split across the shipped catalogue:

| Namespace | `rowwise` | `window` |
| --- | ---: | ---: |
| `.ts` | 0 | 42 |
| `.xs` | 7 | 0 |
| `.panel` | 3 | 0 |
| `.factor` | 2 | 2 |

Every `.ts` operator is `series -> scalar`, so the whole namespace is `window` — that is what
the namespace *is*, not a defect in it. The two `window` entries under `.factor` are
`forward_return` and `portfolio_sort`.

**A new operator must declare `safe_scope`.** `tests/test_registry_conformance.py`
parametrises over `registry.all()` and enforces it, along with three checks that make the
declaration mean something: a `"scalar"` `output_shape` may not be `"rowwise"` (proved from
metadata, no data involved); every `"rowwise"` spec must survive *both*
[leak verifiers](testing.md); and every `"window"` spec must be demonstrably **not**
prefix-invariant when broadcast per row, so the label cannot become a rubber stamp. Note that
a passing `assert_no_lookahead` is never treated as evidence of safety for a scalar
aggregate — on any given probe panel some statistics are degenerate and pass vacuously.
Failures are evidence; passes are not.

One caveat on the counts: the 56 above are the catalogue registered by `import panelary`.
Importing `panelary.evolve` registers a further 52 specs in an `evolve` namespace, which the
conformance suite also covers.

## Reading `audit()`

`audit()` returns a dict of four lists. On the current tree three of them are empty
(`missing_leakage_safe`, `missing_provenance`, `non_permissive_license` — all 56 specs are
Apache-2.0), and one is not:

```python
>>> from panelary.registry import registry
>>> registry.audit()["missing_panel_safe"]
['ic', 'orthogonalize', 'portfolio_sort', 'demean', 'rank', 'standardize']
```

**These six are not a defect.** `panel_safe` means "respects entity boundaries", and all six
are deliberately cross-sectional: three `.factor` operators and three `.xs` operators whose
entire purpose is to compare entities *against each other* within a date. They declare
`panel_safe=False` honestly, and they remain `leakage_safe=True` because they only ever read a
single date's rows. `audit()` surfaces them so the exception stays visible rather than
becoming folklore — the flag to look for in the six is that each is scoped by `.over(time_col)`,
never `.over(entity_col)`.

## See also

- [`PanelFrame`](panel-frame.md) — the type these operators consume.
- [The two-tier API](../concepts/two-tier-api.md) — namespaces versus the fitted-transformer tier.
- [Leak-safety](../concepts/leak-safety.md) — what `panel_safe`, `leakage_safe` and
  `safe_scope` mean, and why leak-safety is a property of an operator *under a usage*.
- [Leak verifiers](testing.md) — `assert_no_lookahead` and `assert_prefix_invariant`,
  the two assertions that check a `rowwise` claim.
- [`factor`](factor.md) — the function-level surface behind the `.factor` namespace.

## API

::: panelary.registry
