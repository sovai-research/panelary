# Detection

Causal detection of explosive regimes, bubbles and changepoints, written
**clean-room from the published papers** and depending on nothing beyond `numpy`
and `polars`.

Every statistic here is a function of data up to `t` and nothing after it.
Recomputing a feature at date `t` with more data appended returns the **bitwise
identical** value — asserted mechanically in `tests/test_detect_leak_safety.py`
across fifteen expanding cut points and every public entry point, not merely
intended.

## What's here

| Problem | Entry point |
| --- | --- |
| Is this series in a mildly explosive regime *right now*? | `bsadf_sequence` |
| The same, across a panel, vectorised over entities | `bsadf_panel` |
| A threshold that does not change when tomorrow's data arrives | `mc_table`, `align_cv` |
| A threshold that simulates nothing at all | `kurozumi_boundary`, `training_max_cv` |
| Detect a shift without knowing its size | `focus` |
| Cheapest correct bubble monitor (analytic boundary, no bootstrap) | `hb_cusum` |
| A continuous, smooth alarm statistic for use as a regressor | `shiryaev_roberts` |
| Is a bubble ending *now*? | `end_of_sample_S` |
| How much of the cross-section is involved? | `breadth` |
| Strip the market factor before asking any of the above | `residualise` |
| Everything, as a tidy panel frame | `panel_features` |

## The recursive right-tailed unit root

For a window `[t1, t2]` the ADF regression is

```
Δy_t = α + β·y_{t−1} + Σ_{i=1..p} ψ_i·Δy_{t−i} + ε_t
```

and the test is right-tailed on `β`: `β = 0` is the unit root, `β > 0` is
explosive. The backward sup-ADF pins the window's **right** edge at `t` and
sweeps the start date:

```
BSADF_t = sup over t1 ≤ t − w0  of  ADF(y[t1 : t])
```

Every regression in that supremum terminates at `t`, which is what makes it a
legitimate per-row feature. The full-sample `GSADF` — a supremum over *both*
endpoints — is not; its point-in-time analogue is `cummax(bsadf)`.

### Why it is fast

Stack the regressors into `z_t` and let `d_t = Δy_t`. Every window needs only
`A = Σ z z'`, `b = Σ z d` and `s = Σ d²` — each a window sum, hence a difference
of two cumulative sums. Build them once in `O(T k²)` and any of the `~T²/2`
nested windows costs `O(1)`. The step that removes the last `O(n)` term, and
which no published implementation exploits:

```
SSR = s − 2β̂'b + β̂'Aβ̂ = s − β̂'b        (exact, because Aβ̂ = b)
```

The fastest published kernel still forms residuals in full inside its inner
loop and is therefore `O(T³)`, not `O(T²)`. Measured here: `T = 1000`
exhaustive — all 452,676 windows — in **~0.012 s of pure NumPy**, against 0.78 s
for that reference implementation in C++.

## Calibration without leakage

Three rules, each with a measured consequence.

**No quantity may depend on the length of the data you happen to hold.** The
conventional minimum-window rule `⌊T(0.01 + 1.8/√T)⌋` revises **41.2%** of
already-published cells by more than 0.05 when `T` grows from 800 to 1600 (mean
`|Δ|` 0.123, max 2.515). With `min_window` frozen the revision is exactly `0.0`.
It is therefore a required absolute-integer argument; `psy_min_window` exists
but demands `acknowledge_leak=True`.

**Simulated critical values are safe; bootstrapped ones are not.** The Monte
Carlo null is `cumsum(randn(n))` — it contains no data and depends only on
`(n, min_window, lag)`. A wild bootstrap fits its null model on the whole sample
and is a genuine look-ahead, so none is shipped.

**Read the table at the right index.** On a driftless null the BSADF sequence is
itself point-in-time, so its value at index `n` *is* the length-`n` statistic
and paths nest exactly — one simulation yields `cv[t]` for every `t`. Reading
the *last* column instead of column `t` is the classic error: the honest 95%
value at length 100 is `0.4331`, while the last column of a `T = 400` table is
`0.6606`, and it moves again to `0.6906` at `T = 800`. Use `align_cv`.

## Sequential monitors

`page_cusum` ships in closed form. Lindley's recursion `g_t = max(0, g_{t−1} +
z_t)` solves to a reflection, `g_t = S_t − min_{j≤t} S_j`, so it is two
cumulative aggregations and a subtraction — `page_cusum_expr` gives the Polars
form, which composes with `.over(entity)` directly.

`focus` is the one to reach for when the shift size is unknown, which is every
financial application. It is provably equivalent to running Page's CUSUM at
*every* magnitude and *every* window length simultaneously, at `O(log n)`
amortised, with no tuning parameter. Pruning is permanent because for
candidates `i < j` the difference of their quadratics is independent of `t`.

## The panel

Use `breadth` — the fraction of the cross-section exceeding its threshold — not
the conventional cross-sectional mean. The statistic diverges exponentially
under the alternative, so the mean is hostage to a few extreme names: holding
the bubbling fraction fixed and raising `δ` from 1.05 to 1.08 sends the mean to
`+15.06` then `+51.99` while breadth stays at `0.052` and `0.202`. The mean
measures how violent the worst few names are; breadth measures how many are
involved.

`residualise` is not optional. With equicorrelated shocks and **no bubble
anywhere** (`N = 200`, `T = 300`, family-wise entity threshold), the 95th
percentile of sup-over-`t` breadth runs `0.010` at `ρ = 0`, `0.096` at `ρ = 0.6`
and `0.251` at `ρ = 0.9` — the effective sample size is
roughly `1/ρ`, not `N`. Projecting out backward-looking factors with betas
frozen at `t` restores it to the `ρ = 0` level at every `ρ` tested.

## What this can and cannot do

Detection delay, conditional on detection, is short: median 3–16 observations,
5–13% of a bubble's own duration. **Unconditional power is the binding
constraint** — a 10-period bubble at `δ = 1.03` is detected with probability
about 0.2. Short mild episodes are missed entirely, and no amount of
implementation quality changes that.

Two things are deliberately refused: a full-sample `GSADF` as a per-row feature,
and episode peak / end / duration, which are knowable only once an episode has
ended. The leak-safe substitute for the latter is the backward-looking
consecutive-exceedance run length emitted by `panel_features`.

## See also

- [`econ`](econ.md) — panel unit-root tests and cross-sectional dependence.
- [`econ.features`](econ-features.md) — trailing-window unit-root and long-memory features.

## API

::: panelary.detect
