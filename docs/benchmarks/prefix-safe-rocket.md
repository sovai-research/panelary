# Prefix-safe MiniRocket — which leak channel carries the accuracy

MiniRocket turns a series into features in three steps. It convolves the series with 84
fixed length-9 kernels, subtracts a bias, and pools each response into the *proportion of
positive values* (PPV). Applied the way the paper applies it, to a whole series at once, the
feature used at row `t` can see the future through two separate channels:

- **Pooling.** PPV over an entity's whole series averages convolution outputs from every
  date, including the ones after `t`.
- **Bias fitting.** Each bias is a quantile of convolution outputs. If the biases are fitted
  once on the whole sample, the future sets the thresholds that define every feature. That
  leak survives even when the pooling is made trailing.

`CausalMiniRocket` (`panelary/embed/_rocket.py`) closes both. It convolves left-only,
computes trailing (or expanding) PPV from a per-entity cumulative sum, and fits the biases in
`fit` from the rows it is given and nothing else. This page prices each channel on a real
panel. The instrument is the same exact Shapley decomposition as the
[borrowed-accuracy table](leakage-table.md): all `2² = 4` coalitions are executed as real
backtests, so the attribution is exact and sums to the total.

**The result: the pooling channel carries all of the borrowed accuracy, and bias fitting
carries essentially none.** That is the reverse of the plan's guess that bias fitting was
"likely the larger" channel.

Reproduce: `python benchmarks/prefix_safe_rocket/bench_prefix_safe_rocket.py --suite --save benchmarks/prefix_safe_rocket/results.json`

Environment: polars 1.44.2, numpy 2.5.3, Python 3.13.12, arm64 (Apple Silicon).
Every number below is read from `benchmarks/prefix_safe_rocket/results.json`, except the
cost spot check, which says so.

## Setup

- **Data.** `data/sp500.parquet`: daily prices for 503 S&P 500 names × 251 trading days,
  2022-06-01 to 2023-05-31, balanced. The input is `log(price)`. GEHC (spun off in January
  2023) carries `price = 0.0` placeholders for its 137 pre-listing days. Those rows are
  treated as missing.
- **Features.** `CausalMiniRocket("logp", n_features=504, window=21, max_dilation=4)`:
  84 kernels × 6 biases at dilations 1–4, padding `"none"`, PPV over a trailing 21-day
  window with a full window required. The kernels sum to zero, so a convolution of log price
  is a weighted sum of log returns, a local trend contrast, and one bias means the same
  thing for every name.
- **Target.** The 5-day forward log return minus that date's cross-sectional mean over all
  503 names (a market-relative return). This is a per-date label transform, and no feature
  sees it.
- **Validation.** `PurgedKFold(5, horizon=5, embargo=5)` on the shared date axis. The model
  is a ridge on standardised features with penalty `λ·n`, λ ∈ {0.01, 0.1, 1, 10, 100, 1000}.
  λ is chosen per fold on a purged inner holdout (the last 20% of the training dates) and is
  never zero.
- **Scores.** Pooled out-of-sample R², and the mean daily cross-sectional rank IC (Spearman)
  of the same predictions. Each is decomposed separately.
- **Replicates.** One replicate is 300 names drawn at random plus a feature seed. The main
  configuration runs 9 replicates (seeds 0–8), and each sensitivity configuration runs 5
  (seeds 0–4). A replicate scores 300 names × 194 dates after warm-up = 58,200 rows, or
  58,063 when the draw includes GEHC (7 of the 9 main replicates).

The two arms of each channel:

| Channel | permissive | point-in-time |
|---|---|---|
| `pooling` | `LeakyMiniRocketReference`: same kernels and biases, PPV over each name's whole 251-day series, broadcast back to every row | trailing 21-day PPV |
| `bias_fit` | biases fitted once on every row | biases fitted per fold on a copy of the panel whose input is nulled outside the training dates, so no test-fold value enters the fit and no convolution window straddles the gap |

## The decomposition

Main configuration, 9 replicates. Each cell is the median over replicates, with the range
in brackets.

| Features | pooled OOS R² | rank IC |
|---|---:|---:|
| point-in-time (both channels closed) | −0.0003 [−0.0024, −0.0002] | −0.0278 [−0.0582, −0.0155] |
| bias fitting permissive only | −0.0003 [−0.0019, −0.0002] | −0.0278 [−0.0578, −0.0154] |
| pooling permissive only | +0.0091 [+0.0058, +0.0096] | +0.0918 [+0.0820, +0.0952] |
| both permissive | +0.0091 [+0.0059, +0.0099] | +0.0926 [+0.0830, +0.0956] |
| _[control] whole-series PPV from training rows only_ | _−0.0237 [−0.0292, −0.0146]_ | _−0.0136 [−0.0386, −0.0092]_ |

| Channel | R² φ median | min | max | mean | IC φ median | min | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pooling | **+0.0094** | +0.0061 | +0.0116 | +0.0089 | **+0.1202** | +0.0979 | +0.1423 | +0.1196 |
| Bias fitting | +0.0001 | +0.0000 | +0.0003 | +0.0001 | +0.0006 | −0.0001 | +0.0022 | +0.0006 |
| **Total borrowed** | **+0.0094** | +0.0061 | +0.0119 | +0.0090 | **+0.1207** | +0.0985 | +0.1429 | +0.1201 |

The total is paired: it is the median of the per-replicate gap between "both permissive"
and "point-in-time", so it need not equal the difference of the two medians. Efficiency
holds on every replicate. The largest `|sum(φ) − total|` is `0.0` for R² and `2.8e-17` for
IC.

## What the numbers say

**Pooling is the whole leak.** Closing only the bias channel changes nothing measurable,
and closing only pooling removes everything. The honest features have no skill on a
market-relative 5-day return (R² −0.0003, IC −0.028), which is what a year of price-only
trend features should be expected to show. Whole-series pooling turns the same features
into IC +0.093. This matches the target-encoding row of the leakage table: the leak does not
make a good feature better; it makes a useless feature look good.

**Bias fitting is a real information flow worth almost nothing.** Both verifiers flag it
(see below). Even so, it borrows at most 0.0003 R² in any main replicate, and its median IC
contribution is +0.0006. A plausible mechanism, not separately measured, is that each bias
is one quantile of a pooled sample of up to 65,536 convolution outputs across 300 names.
Adding the test fold's rows barely moves that quantile, and a threshold that barely moves
flips the indicator for very few cells.

**The gain comes from test-fold data, not from the functional form.** The control row
builds the same static, per-name, whole-series PPV from training rows only, so the
functional form matches the leaky arm but the feature holds no test-fold information. It
scores IC −0.014 and R² −0.024, worse than the honest trailing features. A static per-name
summary is not what helps. What helps is that the summary includes the dates being
predicted.

**The plan guessed wrong about which channel matters.** It called bias fitting "likely the
larger and less-known channel". On this panel it is the smaller one by two orders of
magnitude in IC (+0.0006 against +0.120).

## Sensitivity

The same decomposition was run in four more configurations, with 5 replicates each. Each
Shapley cell is the median, with the range in brackets.

**Rank IC**

| Configuration | honest | both permissive | pooling φ | bias-fitting φ |
|---|---:|---:|---:|---:|
| main: h=5, trailing 21d PPV (9 reps) | −0.0278 | +0.0926 | **+0.1202** [+0.0979, +0.1423] | +0.0006 [−0.0001, +0.0022] |
| h=21 | −0.1025 | +0.2154 | **+0.3270** [+0.3104, +0.3420] | +0.0004 [−0.0062, +0.0011] |
| expanding PPV (min 21 obs) | −0.0425 | +0.0876 | **+0.1307** [+0.1088, +0.1397] | +0.0004 [+0.0003, +0.0071] |
| PPV + MPV | −0.0246 | +0.0844 | **+0.1099** [+0.1087, +0.1322] | +0.0005 [+0.0002, +0.0014] |
| raw (not demeaned) target | +0.0252 | +0.0876 | **+0.0638** [+0.0591, +0.0669] | +0.0006 [+0.0004, +0.0009] |

**Pooled OOS R²**

| Configuration | honest | both permissive | pooling φ | bias-fitting φ |
|---|---:|---:|---:|---:|
| main: h=5, trailing 21d PPV (9 reps) | −0.0003 | +0.0091 | **+0.0094** [+0.0061, +0.0116] | +0.0001 [+0.0000, +0.0003] |
| h=21 | −0.0081 | +0.0307 | **+0.0378** [+0.0256, +0.0534] | +0.0005 [+0.0004, +0.0062] |
| expanding PPV (min 21 obs) | −0.0218 | +0.0086 | **+0.0309** [+0.0070, +0.0715] | +0.0001 [−0.0000, +0.0041] |
| PPV + MPV | −0.0006 | +0.0069 | **+0.0083** [+0.0069, +0.0129] | +0.0001 [+0.0000, +0.0005] |
| raw (not demeaned) target | −0.0174 | −0.0098 | **+0.0067** [+0.0061, +0.0090] | −0.0000 [−0.0001, −0.0000] |

- **The ordering never changes.** Pooling carries the gap in every configuration. The
  bias channel's *median* never exceeds +0.0006 IC or +0.0005 R². Single replicates reach
  +0.0071 IC (expanding PPV) and +0.0062 R² (h=21), so the channel is not identically zero,
  but it is never close to the pooling channel.
- **A longer horizon borrows more.** At 21 days the pooling channel is worth +0.327 IC, and
  the honest arm falls to −0.10. This is consistent with the whole-series summary containing
  the target window, since a 21-day return is a larger slice of the year than a 5-day one.
- **Expanding pooling has an unstable honest arm.** Its honest R² ranges from −0.066 to
  −0.0003, so the R² gap (+0.031) mostly reflects a bad honest arm on some replicates. The
  IC gap (+0.131) is steadier and close to the main configuration's.
- **MPV adds nothing to the leak.** PPV + MPV borrows +0.110 IC against +0.120 for PPV alone.
- **The honest arm's sign is noise.** On the raw target, honest IC is +0.025, against −0.028
  on the demeaned target. Rank IC is computed per date, so demeaning does not change the
  ranks of `y`. What changes is the ridge fit. The gap stays large and positive either way
  (+0.064).
- Efficiency holds in every configuration. The largest per-replicate `|sum(φ) − total|` is
  `5.6e-17`.

## Where the two instruments agree

In each configuration, `assert_no_lookahead` and `assert_prefix_invariant` are run on a
12-name subsample (seed 0), with the cut at the median date, against three constructions:

| Construction | `assert_no_lookahead` | `assert_prefix_invariant` |
|---|:--:|:--:|
| point-in-time: trailing PPV, biases fitted before the cut | clean | clean |
| pooling permissive: whole-series PPV | FLAG | FLAG |
| bias fitting permissive: biases fitted on the frame being checked | FLAG | FLAG |

The result is identical in all five configurations. The test suite
(`tests/test_embed_rocket.py`, 52 tests) makes the same pairing: every leak test runs against
a deliberately leaky construction that must fail it, so a pass means the instrument can see
the leak.

The instruments agree on direction and disagree on magnitude, which is the lesson of the
[leakage table](leakage-table.md#where-the-two-instruments-disagree) again. The verifiers
flag bias fitting exactly as firmly as pooling, and borrowed accuracy prices it at about
zero. The verifier answers whether information flows backwards, and the gap answers whether
that flow is worth anything. On this panel, the answers differ for the channel the plan
worried about most.

## Cost

This is a spot check, not part of `results.json`. It was measured on 2026-09-28 on a shared
machine, with BLAS and polars at one thread. One `transform` of the full panel (126,253
rows, 504 features, `window=21, max_dilation=4`) takes 0.46 s, about 3.6 µs per row. Fitting
the biases takes about 0.5 s. The implementation is pure numpy, with no numba.

## Honesty notes

Read these before quoting a number.

- **This does not show that published ROCKET backtests are inflated.** It shows that the
  whole-series form borrows accuracy on this panel. A published study is inflated only if it
  pooled over whole series *and* used the result as a feature at earlier dates, and this
  benchmark does not check any study. The plan's line "if it is non-zero, every published
  ROCKET backtest in finance is inflated" needs that second step, and it has not been taken.
- **One year of data.** The sample is 251 days of one regime sequence, mid-2022 to mid-2023.
  Every replicate uses all the same dates and 300 of the same 503 names, so the ranges
  measure sensitivity to the name draw and the feature seed, not to the period. They
  understate the real uncertainty.
- **Not survivorship-clean.** The ticker list is a mid-2023 snapshot (it contains GEHC), and
  the file documents neither its price adjustment nor its constituent date. Treat it as a
  convenient real panel, not a research sample.
- **k-fold, not walk-forward.** `PurgedKFold` also trains on dates after the test block. That
  is the right harness for pricing a leak, because both arms see identical folds, but the
  honest arm is not a tradable backtest.
- **The ridge wants more shrinkage than its grid offers.** In the main configuration λ = 1000,
  the top of the grid, was chosen in 74 of 180 fits (9 replicates × 5 folds × 4 coalitions).
  Each sensitivity configuration chose it in 36–40 of 100 fits. A wider grid was not tried.
- **Do not read the honest arm as a verdict on MiniRocket for returns.** It is one year of
  price-only features on one target, and its sign flips with the label definition.
- **Left out, and so untested:**
  - the optional numba path from the build contract;
  - the separate scale features the contract pairs with `normalise=False`;
  - the paper's bias design, which takes each bias from one randomly chosen training
    series. `CausalMiniRocket` instead pools a seeded subsample of training *rows* across
    names, a deliberate choice documented in the module. The bias channel's near-zero price
    may depend on that choice. A one-series design estimates each quantile from far fewer
    values, and it has not been measured.
- **The control row covers the main configuration only.** The sensitivity configurations
  ran before a fix to the control arm's missing-value fill (an entity with too few training
  responses, GEHC, has no train-only PPV). Their control values were invalid and have been
  removed from `results.json`. No other arm depends on the control, and the main
  configuration's four coalitions reproduced bit for bit across the two runs.
- **Correctness is checked against a brute-force definition**, not against aeon or wildboar
  as a black-box oracle. The build contract's clean-room procedure asks for that step, and it
  has not been done.

## Reproducing

```bash
# all five configurations; writes results.json (~5 min of replicate time)
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 POLARS_MAX_THREADS=2 \
    python benchmarks/prefix_safe_rocket/bench_prefix_safe_rocket.py --suite \
    --save benchmarks/prefix_safe_rocket/results.json

# one configuration, replacing only its entry in an existing results file
python benchmarks/prefix_safe_rocket/bench_prefix_safe_rocket.py --only main \
    --save benchmarks/prefix_safe_rocket/results.json

# a custom run: 21-day horizon, expanding pooling, 3 replicates
python benchmarks/prefix_safe_rocket/bench_prefix_safe_rocket.py \
    --horizon 21 --window 0 --min-periods 21 --replicates 3
```

Recorded replicate time per configuration: 83 s for the main configuration, and 41, 45, 98
and 46 s for the other four, 313 s in total. That excludes data loading and the verifier
cross-check. The run is deterministic (seeded RNG throughout, float64 arithmetic) and uses
numpy and polars only, with no scikit-learn.

## References

Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). *MiniRocket: A very fast (almost)
deterministic transform for time series classification.* KDD '21.

Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). *MultiRocket: multiple pooling
operators and transformations for fast and effective time series classification.* Data
Mining and Knowledge Discovery. (The MPV pooling.)

Jorge, M., & Ruben, C. (2024). *Time series clustering with random convolutional kernels.*
Data Mining and Knowledge Discovery. (R-Clustering: ~500 features and permuted biases.)
