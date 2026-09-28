# `frac_diff` warm-up: a hard invariant vs. a tested design decision

**Stage:** todo — needs an owner's decision, not a refactor · **Found:** 2026-09-12

Two things this project wants are in direct conflict in one line of
`panelary/_internal/_ffd.py`. Both were deliberate. Only one can hold.

## The measurement

```python
e = frac_diff_expr(pl.col("x"), d=0.4)          # kernel width 90
x = <200-row random walk>

prefix T= 51: row 50 = 3.8109289036087635
prefix T=200: row 50 = None
```

The same row of the same series is a number when computed on 51 rows and
`null` when computed on 200. Nothing about row 50's own history changed.

## The cause

`_ffd.py:225` — `n_null = min(width - 1, n - 1)`.

The `n - 1` term is a quantity that depends on `len(x)`. When the kernel is
longer than the series it caps the warm-up so the last row gets a value
computed against a *truncated* kernel; when the series later grows past the
kernel width, that row reverts to `null`.

## Why it is not obviously wrong

It was a considered choice, documented in the function's "Warm-up / null
policy" section and pinned by a test whose name states the intent:
`tests/test_fracdiff_perf.py::test_kernel_longer_than_series_returns_partial_not_all_null`
— *"the warm-up is capped so the final (fullest-window) row is a valid partial
output rather than all-null."* The value it produces is causal: it uses only
past observations. The previous implementation returned all-null here, which is
arguably less useful.

## Why it is wrong anyway

`AGENTS.md` hard invariant 1: *"No quantity may depend on `len(x)`: not window
sizes, thresholds, lag orders, nor normalisation constants."* This is that,
exactly. The practical consequence is the failure mode the library exists to
prevent: in a rolling backtest whose window grows, a feature value changes
because of how much data happens to be loaded, not because of anything in its
own past. `assert_no_lookahead` does **not** catch it — the partial output is
causal — which is why it survived. Only `assert_prefix_invariant` sees it.

## The options

1. **Null everything when `width > n`** (`n_null = min(width - 1, n)`). Restores
   prefix invariance; reverts the test's intent; turns numbers into nulls for
   anyone in that regime today.
2. **Keep the partial output, make it opt-in** (`partial=False` default). Honest
   and non-breaking, at the cost of a parameter on a hot path.
3. **Keep it and document the exception**, narrowing invariant 1 to "in the
   regime `n >= width`". Cheapest; weakens a hard invariant to fit one call site.

Recommendation: **(1)**, with (2) as the compromise if the partial output has a
real consumer. A hard invariant that has a silent exception is not one, and the
regime where they differ (`kernel longer than the series`) is one where the
output is of marginal use either way.

**Do not change this without deciding which of the two you are giving up.**
