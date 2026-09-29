# Cross-sectional distribution and co-movement

How is the cross-section of returns distributed today, and how much does it
move together? These features answer that one date at a time: dispersion, tail
heaviness, breadth and concentration of a single cross-section; tail risk,
distribution shift and skewness across a few dates; the average correlation and
the common idiosyncratic volatility of the as-of universe.

Every value is a function of data at or before its own date, checked bit for
bit. Pure `numpy` + `polars`.

## The 60-second version

```python
import polars as pl
import panelary as pn

cov = pn.covariance  # lazy: nothing loads until first use

# One cross-section at a time: Polars expressions, grouped by date.
panel.with_columns(
    pl.col("ret").xs.dispersion(kind="mad").over("date").alias("disp"),
    pl.col("ret").xs.tail_index(q=0.05).over("date").alias("alpha"),
    pl.col("ret").xs.up_share().over("date").alias("breadth"),
    pl.col("ret").xs.entropy(kind="share").over("date").alias("concentration"),
    pl.col("ret").xs.dispersion(kind="iqr").over(["date", "sector"]).alias("sector_iqr"),
)

# Across dates: frame operations, one row per date (broadcast=True joins them back).
cov.avg_correlation(panel, returns="ret", window=63)                     # equal-weight
cov.avg_correlation(panel, returns="ret", window=63, kind="pollet_wilson")
cov.avg_correlation(panel, returns="ret", window=63, group="sector")     # per sector
cov.common_idio_vol(panel, returns="ret", window=63, n_factors=1, refit_every=21)
cov.kelly_jiang_tail(panel, returns="ret", window=21, q=0.05)
cov.kelly_jiang_beta(panel, returns="ret", window=252)                   # per name
cov.xs_wasserstein(panel, value="ret", standardize=False)
cov.avg_skewness(panel, returns="ret", window=21)
```

`panel` is a `PanelFrame`, or a Polars frame with `entity=` and `time=`.

## Single-date summaries: the `.xs` operators

Each reduces its cross-section to one number and broadcasts it to every row of
that cross-section. **The `.over(date)` is load-bearing**: evaluated bare, the
"cross-section" is the whole frame and every value depends on the future
(`tests/test_xs_distribution.py` shows the look-ahead verifier catching it).
`.over(["date", "sector"])` gives the within-group value.

| Method | Registry name | Value |
| --- | --- | --- |
| `.xs.dispersion(kind=)` | `xs_dispersion` | `"sd"` (ddof 1), `"mad"` (`1.4826 · median|x − median|`), `"iqr"` (q75 − q25), `"idr"` (q90 − q10) |
| `.xs.tail_index(q=, tail=, min_exceedances=)` | `xs_tail_index` | Hill (1975) `α` of the `k = ⌊q·n⌋` largest losses |
| `.xs.up_share()` | `xs_up_share` | `#(x > 0) / #observed` |
| `.xs.entropy(kind=)` | `xs_entropy` | `"share"`: `−Σ p log p / log n`, `p = |x| / Σ|x|`; `"hist"`: Freedman–Diaconis histogram entropy `/ log(n_bins)` |

Conventions, the same for all four:

- NaN is missing, exactly like null. An undefined summary (too few names, a zero
  denominator) is **null**, never NaN.
- Quantiles use linear interpolation (numpy's default), set explicitly because
  the Polars default is `"nearest"`.
- The Hill threshold is the `(k + 1)`-th largest loss, an order statistic, so
  `.xs.tail_index` equals `panelary.econ.features._evt.hill_index` with `k`
  matched (tested to 1e-12). It is null when the threshold is not a loss;
  `hill_index` shifts the sample instead.
- Registry names carry an `xs_` prefix because the registry keys on bare names
  and per-entity siblings (a time-series `entropy`) want the short ones.

## Across dates: frame operations

| Function | What it measures | Per-date cost |
| --- | --- | --- |
| `avg_correlation(kind="equal")` | mean off-diagonal correlation of the as-of universe over the trailing window | O(W·N), one pass over the window; never an N×N matrix |
| `avg_correlation(kind="pollet_wilson")` | Pollet & Wilson (2010) `σᵢσⱼ`-weighted mean correlation | O(N) where the universe is stable, O(W·N) elsewhere |
| `common_idio_vol` | Herskovic et al. (2016) mean trailing idiosyncratic volatility; `broadcast=True` adds each row's `idio_vol` | `residualise` per refit + O(N) |
| `kelly_jiang_tail` | Kelly & Jiang (2014) tail risk `λ` of the pooled trailing window | one `np.partition` of W·N values |
| `kelly_jiang_beta` | each name's trailing slope on `λ_{t−1}` | O(1) per row (rolling moments) |
| `xs_wasserstein` | W₁ between the cross-section at `t` and at the previous date | O(N log N), exact |
| `avg_skewness` | Jondeau, Zhang & Zhu (2019) mean trailing skewness | O(1) per row |

Output columns: `avg_corr, n_entities` (plus `universe_stable` for the pooled
Pollet–Wilson kind); `common_idio_vol, n_entities`; `kj_tail, kj_threshold,
kj_n_exceed, kj_n_obs`; `kj_tail_lag, kj_beta`; `xs_w1`; `avg_skew,
n_entities`. All are "as of the close of t": the window includes `r_t`.

**The as-of universe.** For `avg_correlation` a name is in date `t`'s universe
if it is observed at `t` and in at least `ceil(min_coverage · window)` dates of
`(t − window, t]` (default 0.95). A name listed after `t` cannot enter; a
delisted one leaves after its last observation; the pivot never forward-fills.
The few missing cells a coverage below 1 admits are zero after demeaning, which
attenuates a correlation by about `√(fᵢfⱼ)`: at most 5% at the default.

**Pollet–Wilson.** `(σₚ² − Σwᵢ²σᵢ²) / ((Σwᵢσᵢ)² − Σwᵢ²σᵢ²)` is exactly the
`σᵢσⱼ`-weighted mean of the pairwise correlations, but only when the equal-weight
portfolio `p` is built from the same names on every date of the window. Dates
where the observed universe changed are flagged `universe_stable=False` and
recomputed on the kernel. It weights volatile names more, so it approximates,
but does not equal, `kind="equal"`.

**Common idiosyncratic volatility.** Residuals come from
`panelary.detect._panel.residualise`, reused unchanged: PCA loadings fit on the
`fit_window` dates before each refit date `fit_window + k · refit_every`,
frozen, and applied to each later date's own cross-section. The wrapper calls it
one refit segment at a time on the names with enough history in the training
block. That is what makes the result bit-identical when a name lists later:
`residualise` over the whole entity axis would treat it as an all-missing row
and change its eigensolver's rounding (measured 5.6e-15).

**Kelly–Jiang.** `λ_t = mean(log(R / u_t))` over the `k = ⌊q·n⌋` smallest of the
`n` returns pooled over the trailing `window` dates, with `u_t` the `(k + 1)`-th
smallest. It is Hill's estimator in the `ξ` convention (larger means heavier),
the reciprocal of `hill_index`'s `α`. `residual=True` pools
statistical-factor residuals instead, as the paper pools Fama–French residuals.
`kelly_jiang_beta` reuses `econ.features._common.rolling_beta`. That function
uses the one-pass `E[xy] − E[x]E[y]` formula, which cancels badly on `λ`, a
level near 0.4 moving by hundredths. So the regressor is first shifted by a
causal constant, its first defined value, which leaves the slope unchanged up
to rounding.

**Wasserstein.** The exact path sorts each date once and merges the two
quantile grids in closed form. In units of `1/(n·m)` the breakpoints are the
multiples of `m` and of `n`, and their merged order is arithmetic, not a second
sort. It matches `scipy.stats.wasserstein_distance` to 1e-12 (the oracle is
test-only). `grid=K` is the midpoint rule on `K` quantile levels.
`standardize=True` compares same-date z-scores, isolating a change of shape
from a change of location and scale.

**Group variants.** `group=` partitions date `t`'s universe by each name's label
**on date t**. Its label history inside the window is ignored, so a later
reclassification cannot rewrite the past (plan trap T12). Joining a
"latest classification" snapshot onto the history is exactly the look-ahead
the tests catch. Groups smaller than `min_entities` (default 10) give null.
Each date's window moments are computed once, and every group's value comes from
a single product with a names × groups weight matrix, so the group variant
costs about the same as the pooled one.

## Leak safety

| Trap | Where it appears | Guard | Test |
| --- | --- | --- | --- |
| T10: Kelly–Jiang threshold over the calendar month containing `t` | the paper's monthly design, reused daily | trailing window only | the calendar-month variant is caught by `assert_no_lookahead`; ours passes bitwise |
| T11: `.xs` op evaluated without `.over(date)` | user error | documented composition; registry scope claimed for it | the bare call is caught as a look-ahead |
| T12: group labels from the latest classification | GICS snapshot joins | date-`t` labels | future relabels leave the past bit-identical; the snapshot join is caught |
| T5: delisted returns forward-filled | `build_tensor`'s default | `forward_fill=False` | a name's values stop at its exit |

Beyond the traps, every operator runs through `assert_prefix_invariant` and
`assert_no_lookahead` with `tol=0.0` on a panel whose names enter and leave,
under `.over(date)` and `.over([date, group])`. The per-date reductions run
over each date's compacted members, never over the full entity axis with a
mask. A name listing later would otherwise change the summation order and move
earlier values in the last bit.

## Measured performance

T = 2,500 dates × N = 3,000 names (7.5M rows, 1% of cells missing at random),
Apple M5 Pro, NumPy 2.5.3, Polars 1.44.2, best of two runs. The machine was
shared with other jobs (load average 10–11 on 15 cores), so treat these as
upper bounds. Every frame operation includes the long → dense pivot
(`build_tensor`, 0.45 s here).

| Operation | Time | Plan budget (T = 5,000) |
| --- | --- | --- |
| the four `.xs` ops together, `.over(date)` | 0.35 s | ≤ 3 s at 15M rows |
| `.xs.dispersion` sd / mad | 0.02 / 0.10 s | |
| `.xs.tail_index` | 0.16 s | |
| `.xs.up_share` | 0.03 s | |
| `.xs.entropy` share / hist | 0.18 / 0.28 s | |
| `avg_correlation`, W = 63 | 1.34 s (1.07 s on a complete panel) | ≤ 3 s |
| `avg_correlation(kind="pollet_wilson")`, complete panel (O(N) path) | 0.74 s | ≤ 2 s at 15M rows |
| `avg_correlation(kind="pollet_wilson")`, 1% missing (kernel fallback on every date) | 1.45 s | |
| `avg_correlation(group=...)`, 11 sectors | 1.64 s | |
| `common_idio_vol`, W = 63, `fit_window` = 252, `refit_every` = 21 | 4.51 s | |
| `kelly_jiang_tail`, 21 dates | 1.35 s | ≤ 3 s |
| `kelly_jiang_tail(residual=True)` | 4.82 s | |
| `kelly_jiang_beta`, W = 252 | 1.82 s | |
| `xs_wasserstein`, exact / `grid=100` | 0.39 / 0.21 s | ≤ 2 s |
| `avg_skewness`, 21 dates | 0.35 s | |

What dominates:

- **`common_idio_vol` and the residual Kelly–Jiang tail.** About 4 s of each is
  `residualise`'s own loading fit, about 107 refits at roughly 40 ms each. With
  more names than training dates it takes an SVD of the 252 × 3,000 block
  (42 ms). An eigendecomposition of the 252 × 252 dual Gram gives the same
  loadings in 6 ms. That change belongs to `detect`'s owner and is not made
  here.
- **The exact average correlation** is about 0.35 ms per date: copying the
  date's compacted window, then two passes over it. The copy is what keeps
  values bit-identical when later-listing names arrive.

## References

- Freedman, D., Diaconis, P. (1981). On the histogram as a density estimator: L2 theory. *Z. Wahrscheinlichkeitstheorie* 57, 453–476.
- Herskovic, B., Kelly, B., Lustig, H., Van Nieuwerburgh, S. (2016). The common factor in idiosyncratic volatility. *J. Financial Economics* 119(2), 249–283.
- Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics* 3(5), 1163–1174.
- Jondeau, E., Zhang, Q., Zhu, X. (2019). Average skewness matters. *J. Financial Economics* 134(1), 29–47.
- Kelly, B., Jiang, H. (2014). Tail risk and asset prices. *Review of Financial Studies* 27(10), 2841–2871.
- Pollet, J. M., Wilson, M. (2010). Average correlation and stock market returns. *J. Financial Economics* 96(3), 364–380.
- Vallender, S. S. (1973). Calculation of the Wasserstein distance between probability distributions on the line. *Theory of Probability & Its Applications* 18(4), 784–786.
