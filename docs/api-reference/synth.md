# Synthetic panels (`panelary.synth`)

A seeded, dial-parameterised generator of long-format panels with **planted
ground truth**. Pure `numpy` + `polars`, no optional dependency. For the model,
the guarantees and the worked leak check, read
[Synthetic panels](../user-guide/synthetic-panels.md) first.

## What's here

| Your problem | Entry point |
| --- | --- |
| Draw one panel with known truth | `generate_panel(config, seed=...)` |
| Describe the dials (all mechanisms on by default) | `SynthConfig` |
| A clean, balanced baseline panel | `SynthConfig.plain(**overrides)` |
| Draw the dials themselves from a prior | `sample_config(seed, ...)` |
| The data: revised panel, first releases, bitemporal vintages | `SyntheticPanel` |
| Loadings, clusters, regimes, break dates, latent grid, planted lag and coefficient | `GroundTruth` |
| Does this feature know the signal earlier than it could? | `check_planted_lag` → `PlantedLagReport` |

## Column contract

| Frame | Columns |
| --- | --- |
| `panel` | `entity` (String), `time`, `value` (Float64, nulls = missing), `signal` if `signal_observed` |
| `first_observed` | as `panel`, plus `knowledge_time`; `value` is the first release |
| `vintages` | `entity`, `event_time`, `knowledge_time`, `value`, `revision` (0 = first release) |
| `truth.latent` | `entity`, `time`, `step`, `alive`, `observed`, `missing`, `value`, `signal`, `shock`, `level`, `common`, `regime`, over the full entity x step grid |
| `truth.factors` | `step`, `time`, `regime`, `factor_k`, `log_vol_k`, `cluster_factor_c` |
| `truth.entities` | `entity`, `cluster`, `intercept`, `idio_vol`, `entry_step`, `exit_step`, `observation_period`, `observation_phase`, `reporting_lag`, `loading_k` |

Time columns are `Int64` steps, or `Date` / `Datetime` when `start` is set.
Every frame is sorted by entity, then time.

## Invariants (tested)

- The same seed and dials give byte-identical frames, across processes.
- `T + k` periods reproduce every row with event time `< T`, and `N + m`
  entities reproduce the first `N`.
- `knowledge_time >= event_time + reporting_lag` for every release. Releases of
  one value are strictly increasing in `knowledge_time`, and the last one equals
  the `panel` value exactly.
- `signal` is independent of everything except `value[t + signal_lag]`, so a
  causal feature has zero population correlation with `signal[t - l]` for every
  `l < truth.knowable_lag()`.

## API

::: panelary.synth
