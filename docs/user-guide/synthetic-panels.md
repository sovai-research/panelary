# Synthetic panels

Every leakage test needs a panel, and until now every leakage suite wrote its
own. `panelary.synth` is the shared version: a seeded generator in which each
messy property of real panel data is a dial, and every draw comes back with
the **ground truth** it was drawn from. That includes one planted predictive
relationship with a known lag and coefficient, which turns "is this pipeline
leaking?" into a measurement.

```python
from panelary.synth import (
    SynthConfig, generate_panel, sample_config, check_planted_lag,
)
```

Pure NumPy + Polars, no optional dependency.

---

## 1. One draw

```python
data = generate_panel(seed=0)                 # every mechanism on, moderately
data = generate_panel(SynthConfig.plain(), seed=0)   # all complications off
data = generate_panel(seed=0, n_entities=200, tail_df=3.5, missing_rate=0.2)
```

`seed` is required, and all randomness flows from it: the same seed and dials
give byte-identical frames. A draw is a `SyntheticPanel` holding three views of
the same data plus the truth:

| Attribute | Shape | What it is |
| --- | --- | --- |
| `panel` | `entity`, `time`, `value` [, `signal`] | The **fully revised** long panel: one row per scheduled observation of a live entity, `value` null where missing. What a research database shows with hindsight. |
| `first_observed` | `entity`, `time`, `value`, `knowledge_time` [, `signal`] | Row-aligned with `panel`, but each value is its **first release** and `knowledge_time` says when that was. The real-time panel. |
| `vintages` | `entity`, `event_time`, `knowledge_time`, `value`, `revision` | The **bitemporal** release history: one row per release, `revision` 0 first. The input shape for an as-of join. |
| `truth` | `GroundTruth` | Dials, loadings, clusters, regimes, break dates, the latent grid, and the planted signal. |

`data.panel_frame()` wraps `panel` as a `PanelFrame` keyed on `entity` / `time`.
Times are `Int64` steps unless you pass `start=` and `every=` (for example
`start=date(2020, 1, 31), every="1mo"`), which relabel steps as calendar times.

## 2. The model

On a grid of steps `t = 0 .. n_periods - 1`, entity `i` in cluster `c` has

```text
value[i, t] = level[i, t]                              intercept + break shifts
            + loadings[i] @ factors[t]                 regimes + stochastic vol
            + cluster_strength * cluster_factor[c, t]
            + signal_coef * signal[i, t - signal_lag]  the planted signal
            + idio[i, t]                               AR(1), Student-t shocks
```

and then an observation layer decides which of those values anyone sees, when,
and how accurately.

| Dial family | Dials | What it does |
| --- | --- | --- |
| Size | `n_entities`, `n_periods` | The entity pool and the series length. |
| Factors | `n_factors`, `factor_persistence` | AR(1) common factors; loadings scatter around cluster centres. |
| Regimes | `n_regimes`, `regime_persistence`, `regime_vol_ratio`, `regime_mean_shift` | A Markov chain that switches factor volatility and means. |
| Stochastic volatility | `sv_persistence`, `sv_vol` | AR(1) log-variance on each factor. |
| Tails | `tail_df` | Student-t shocks scaled to unit variance; `None` = Gaussian. |
| Clusters | `n_clusters`, `cluster_strength`, `loading_dispersion` | Groups of entities that share loadings and a cluster factor. |
| Idiosyncratic | `idio_persistence`, `idio_vol`, `intercept_dispersion` | Entity-level noise and levels. |
| Planted signal | `signal_lag`, `signal_coef`, `signal_observed` | See section 4. |
| Breaks | `break_rate` or `break_times`, `break_size` | Dates at which every entity's level jumps. |
| Entry and exit | `initial_fraction`, `entry_rate`, `exit_rate` | Varying `N` over time; exit is permanent. |
| Asynchronous observation | `observation_periods`, `observation_prob` | Each entity observes on its own period and phase; scheduled observations can be skipped. |
| Missingness | `missing_rate`, `missing_informativeness` | Null values whose probability rises (or falls) with the value itself. |
| Reporting and revisions | `reporting_lag`, `reporting_jitter`, `revision_prob`, `max_revisions`, `revision_noise`, `revision_gap` | When each value is first known, how wrong the first release is, and how it is revised. The last revision is exactly the true value. |

Every dial has a neutral setting that switches its mechanism off, and
`SynthConfig.plain(**overrides)` starts from all of them. Each dial is tested
to do what it says on the returned truth: Student-t shocks have excess
kurtosis, informative missingness correlates with the value, releases never
precede `event_time + reporting_lag`, and so on.

## 3. Guarantees

- **Deterministic.** Byte-identical output for the same seed and dials, across
  calls and processes, on the same platform. There is no global RNG and no
  dependence on hash randomisation. Across platforms, `exp` / `tanh` may differ
  in the last bit, and that can flip a knife-edge missingness draw.
- **Prefix-consistent in time.** Generating `T + k` periods with the same seed
  reproduces every row with event time `< T`, in every frame. The latent
  process is causal, every hazard is per step rather than a fraction of `T`,
  and nothing is normalised by a sample statistic.
- **Prefix-consistent in entities.** Generating `N + m` entities reproduces the
  first `N`. Every random draw comes from a stream keyed by
  `SeedSequence(seed, spawn_key=(component, step_or_entity))`, and no stream
  makes more than one entity-sized draw.

`vintages` include releases whose `knowledge_time` falls after the last period.
To reconstruct the database as it stood at the end of the sample, filter
`knowledge_time < n_periods`.

## 4. The planted signal, and the leak check

`signal` is i.i.d. standard normal, independent of every other quantity in the
simulation, and it enters `value` exactly `signal_lag = L` steps after it is
drawn, with coefficient `signal_coef`. By default it is **latent**: the only
way to learn `signal[s]` is to read `value[s + L]`.

That makes the lag a leak detector. A feature computed at time `t` from data
available at `t` can be correlated with `signal[t - l]` only for `l >= L`. If a
feature is correlated with the signal at a shorter lag, it has read `value`
from after `t`.

```python
import polars as pl

data = generate_panel(seed=1, n_entities=60, n_periods=250)
frame = data.panel.with_columns(
    causal=pl.col("value").shift(1).over("entity", order_by="time"),
    leaky=pl.col("value").shift(-1).over("entity", order_by="time"),
)
check_planted_lag(frame, "causal", data.truth).leaks   # False: recovered at L + 1
check_planted_lag(frame, "leaky", data.truth).leaks    # True:  recovered at L - 1
```

`check_planted_lag` computes the pooled Spearman correlation between the
feature at `(entity, t)` and the true signal at `(entity, t - l)` for a range of
lags. It flags a leak when `|z| > threshold` (default 5) at any lag below the
**knowable lag**. The `PlantedLagReport` carries the whole profile
(`to_frame()`), the lag at which the signal was recovered, and the offending
lags.

The knowable lag depends on what the feature was allowed to see:

| Situation | `truth.knowable_lag(...)` |
| --- | --- |
| Features on `panel`, signal latent (default) | `L` |
| Features at *decision* time on published data (`clock="knowledge"`) | `L + min_reporting_lag` |
| `signal_observed=True` (the signal is a column) | `0` |

`clock="knowledge"` catches the point-in-time mistake: a feature that uses the
value for event `t` at decision time `t`, although that value was only
published `reporting_lag` steps later. An as-of join on `first_observed` or
`vintages` passes, and the naive event-time feature fails.

The check has limits:

- **It detects concentrated look-ahead**: shifts, centred windows, backward
  fills, differences with a lead. Look-ahead spread thinly over the whole sample
  is invisible to it. A full-sample mean gives each future step a weight of
  `1/T`, so use `panelary.testing.assert_no_lookahead` for that; it is the exact,
  perturbation-based check.
- **Asynchronous panels stretch a shift.** On a period-5 schedule, a one-row
  shift reaches five steps ahead. Widen `max_lead` if schedules are sparse.
- **Exposing the signal blunts the check for `value`-only features.** With
  `signal_observed=True` the default floor is `0`. Pass
  `min_lag=truth.signal_lag` if the pipeline under test never reads `signal`.
- **Power scales with rows and with `|signal_coef|`** relative to the other
  variance in `value`. At the defaults, a 60 x 250 panel recovers the signal
  at the planted lag with `|z|` between 17 and 23 (seeds 0 to 2).

## 5. Sampling across the dial space

`sample_config(seed)` draws the dials themselves from a broad prior, with
log-uniform `n_entities` and `n_periods`. Use it when the task distribution is
the object of study, as in pretraining a panel foundation model on the prior:

```python
for k in range(10_000):
    cfg = sample_config(k, n_entities=(10, 200), n_periods=(50, 500))
    data = generate_panel(cfg, seed=k)
    ...
```

Pinned dials (`sample_config(k, tail_df=None)`) leave the other draws
unchanged. The model half of that plan, the pretraining and GPU code, lives in
its own repository. Panelary supplies only the prior.

## A note on CAFE

The PanelPFN plan assumed CAFE "is already a generator you can sample from".
It is not: `cafe` (`cafe-impute` 0.1.0) is a causal imputation estimator. Its
`forecast` is a deterministic point forecast, and its only sampler is a private
ten-line benchmark helper (`cafe.benchmark._synthetic`: four Gaussian AR(1)
factors plus a sinusoid, as a wide matrix with no panel structure and no
dials). So this generator is written from scratch and does not depend on
`cafe`.
