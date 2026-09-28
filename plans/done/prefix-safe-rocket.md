# Prefix-safe temporal ROCKET

> **Status (2026-09-28): implemented.** Shipped: `panelary/embed/_rocket.py`, which
> contains `CausalMiniRocket` and `LeakyMiniRocketReference`.
> `CausalMiniRocket` has 84 fixed length-9 kernels and left-only convolution. It computes
> trailing or expanding PPV/MPV from a per-entity cumulative sum, and its biases are fitted
> state learned in `fit`, so `fit_is_empty = False`. It uses about 500 permuted biases,
> following R-Clustering. `LeakyMiniRocketReference` pools over the whole series. It is
> for tests and the benchmark only, is marked `leakage_safe = False`, and must never be
> exported. `CausalMiniRocket` is exported from `panelary.embed`; the leaky reference is
> not (a test in the embed suite checks that).
>
> Tests: `tests/test_embed_rocket.py`, 52 tests. Every leak test is paired with a leaky
> construction that must fail it.
>
> Benchmark: `benchmarks/prefix_safe_rocket/` (the script and `results.json`), written up
> in `docs/benchmarks/prefix-safe-rocket.md`.
>
> **Headline.** The setup is `data/sp500.parquet`, 300-name replicates, a 5-day
> market-relative return and `PurgedKFold(5, 5, 5)`. The exact two-channel Shapley split
> puts all the borrowed accuracy in **pooling**: rank IC +0.120 [+0.098, +0.142] and R²
> +0.0094 over 9 replicates. It puts almost none in **bias fitting**: IC +0.0006 and R²
> +0.0001. Honest features score IC −0.028. This is the reverse of the "likely the larger"
> guess below. The ordering holds for a 21-day horizon, expanding pooling, PPV+MPV and a raw
> target. It does **not** show that published ROCKET backtests are inflated.
>
> Left out, and so untested:
> - the numba path;
> - separate scale features;
> - the paper's one-random-training-series-per-bias design;
> - black-box numerical verification against aeon or wildboar, as the contract's
>   clean-room procedure asks. Correctness is checked against a brute-force definition
>   instead.

**Stage:** todo — needs a build contract before implementation · **Priority:** 2 (small, fast, uses borrowed-accuracy as its instrument) · **Home:** Panelary

## Pitch

ROCKET's proportion-of-positive-values (PPV) pooling is computed over the
entire series. In a rolling backtest the feature at time t depends on values
after t. No causal variant of MiniRocket exists in any library, and the same
applies to max, min, slope and local volatility summaries.

ROCKET is the best-performing family in both time-series classification and
extrinsic regression — window to scalar, the shape of every return-prediction
problem — and its canonical implementation leaks in exactly the setting
finance uses it. Fixing it produces stage one of later items, and the
measurement alone is publishable.

First experiment: expanding-window PPV against batch PPV on one panel; report
the borrowed-accuracy delta. If it is non-zero, every published ROCKET
backtest in finance is inflated.

## Assessment

**Sharpen the leak claim before building — there are two channels, not one.**

1. *Pooling.* Only leaks if ROCKET runs over the full series and the result is
   used as a feature at earlier `t`. Applied to a trailing window ending at
   `t`, PPV over that window is already causal. So the claim holds for
   *how finance uses it*, not for ROCKET itself — the paper has to show
   people use it the leaky way.
2. *Bias fitting — likely the larger and less-known channel.* MiniRocket's
   `fit` draws each kernel's bias from quantiles of convolution outputs on
   training examples. Fit on the whole sample, and future data sets the
   thresholds that define every feature. That leaks even with trailing
   windows.

Measure both separately; that is a two-component borrowed-accuracy
decomposition and a clean first use of that metric.

**Build is cheap and fits Panelary's invariants exactly.** Expanding PPV is a
cumulative count of positives over `t`, which is prefix-invariant by
construction (AGENTS.md invariant 1). Expanding max/min are native
`cum_max`/`cum_min`. No `rolling_map` needed.

**Honest caveat:** "every published ROCKET backtest is inflated" follows only
if the delta is non-zero *and* the papers used the leaky form. A null result
is still reportable, just as a smaller paper.
