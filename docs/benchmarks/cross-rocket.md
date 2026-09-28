# CrossROCKET — random cross-sectional operators against handcrafted features

ROCKET applies fixed random operators *along time*. `CrossRocket`
(`panelary/embed/_cross_rocket.py`) applies a fixed, seeded bank of random operators
*across entities* at each date. This page asks two questions. The plan's question is
whether random cross-sectional operators match handcrafted cross-sectional features under
a ridge head on a return-prediction task. The reviewer's question is whether they beat
random features of each asset's *own* characteristics, the Kelly–Malamud construction.

**The results.**

- **Real data.** Nothing works. Every fitted design has a negative rank IC, and none is
  significant, so the real data cannot rank the designs.
- **Synthetic data with a planted peer-relative signal.**
  - CrossRocket's peer operators carry the signal.
  - CrossRocket matches handcrafted features rather than beating them.
  - It beats random features of each asset's own characteristics.
  - No fitted design reaches the unfitted reversal signal, even though every design is
    given that signal as an input.

Reproduce: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 POLARS_MAX_THREADS=2 python benchmarks/cross_rocket/bench_cross_rocket.py`

Environment: polars 1.44.2, numpy 2.5.3, arm64 (Apple Silicon). The recorded wall time is
357 s for all three panels (122 s real, 118 s and 117 s synthetic). The documented command
caps BLAS and polars at two threads each; `results.json` does not record the thread count.
Every number below is read from `benchmarks/cross_rocket/results.json`.

## What CrossRocket is

Each operator draws, from a seeded RNG, 1–3 input columns and a unit-norm Gaussian weight
vector over them. At each date, for entity `i`, the four families compute:

| Family | Pre-activation at date `t` |
|---|---|
| `median_dev` | robust-z (median/MAD, clipped at ±5) each input, combine, robust-z again, minus `b ~ U(−1, 1)` |
| `rank_threshold` | combine the centred cross-sectional ranks, re-rank to `(0, 1)`, minus `q ~ U(0.05, 0.95)` |
| `subset_agg` | robust-z combination minus its mean over a *value-defined* subset (entities whose rank on a random member column falls in a random band), minus `b` |
| `peer_dev` | robust-z combination minus its mean over `i`'s `m` most-correlated peers, with correlations on `peer_col` over the `W` dates **strictly before** `t`, minus `b` |

The output is `max(0, ·)` by default (`"indicator"` and `"identity"` are also available).
Every operator is permutation-equivariant over entities: relabelling the entities relabels
the output rows and changes nothing else. `bank_` lists every drawn parameter in one table.

The class declares `panel_safe = False` (it mixes entities within a date on purpose),
`leakage_safe = True`, `fit_is_empty = True` (the bank depends only on the seed and the
*number* of input columns) and `is_cross_sectional = True`. Because it is stateless and
causal, transform the whole panel once and split afterwards. Transforming a test fold in
isolation discards the trailing history the peer family needs.

Four design decisions depart from the plan or the embed contract:

- **Per-entity outputs by default, not pooled ones.** The plan proposed pooling across
  entities the way MiniRocket pools along time. A pooled per-date statistic takes one value
  per date, though, so under a linear head it shifts every entity's prediction on that date
  by the same amount and cannot change a single cross-sectional rank. Pooled PPV and MPV are
  available through `output="pooled"` or `"both"` as market-state descriptors. The
  `rank_threshold` operators are excluded from pooling, because their PPV is about `1 − q`
  on every date by construction.
- **Float64 columns by default.** Float32 and a single `pl.Array` column (`as_array=True`)
  are optional. This departs from the embed contract's default of Float32 fixed-size arrays.
- **Median/MAD in numpy.** `.xs.demean` subtracts the mean, so the robust standardisation is
  a numpy kernel rather than a composition of `.xs` primitives. The tests check it against
  the equivalent polars expression, and check the rank family against `.xs.rank`.
- **Strictly trailing peers.** Peer baskets read `peer_col` only on the `W` dates before `t`.
  `_peer_history` is the one place the transform reads another date.

## Setup

- **Data.** `data/sp500.parquet`, with GEHC's `price = 0.0` placeholders set to missing.
  There are eight price-only characteristics per name, all trailing: `r1`, `r5`, `r21`,
  `mom` (t−60 to t−5), `vol21`, `maxr21`, `hi60` (distance from the 60-day high) and
  `beta60`. The characteristic builder passes `assert_no_lookahead` on a 40-name subsample
  before anything else runs.
- **Sample.** Rows where every characteristic, every handcrafted feature and the label are
  defined: 93,421 stock-days, 503 names, 186 dates, 2022-08-26 to 2023-05-23.
- **Target.** The 5-day forward log return, winsorised at 1%/99% per date and demeaned per
  date. Rank IC is measured against the unwinsorised forward return.
- **Validation.** Designs are scored under two schemes on the same test blocks.
  - `PurgedKFold(5, horizon=5, embargo=5)`, with test blocks of 38, 37, 37, 37 and 37 dates.
  - Walk-forward on blocks 2–5, training only on earlier dates, purged by the horizon.
    Block 1 has no past and is skipped.

  The model is a ridge on fold-standardised features, with λ ∈ {1e-2, …, 1e6}. λ is chosen
  by an inner purged 3-fold split of the training dates (minimum MSE), then the model is
  refit on the whole training fold. It is never ridgeless. A fixed-λ path (1, 1e2, 1e4, 1e6)
  is reported alongside, so a shrink-everything selection cannot hide a design's signal.
- **Score.** Mean daily rank IC over test dates, with a Newey–West t (5 lags) on the daily
  IC series. The random designs run on seeds 0–4. The deterministic designs run once.

All designs are scored on exactly the same rows and folds:

| Design | p | What it is |
|---|---:|---|
| H | 16 | handcrafted: each characteristic's per-date centred rank and 1%-winsorised z-score |
| RFF | 256 | random Fourier features of each row's *own* characteristic ranks, bandwidths {0.5, 1, 2}, in the style of Kelly & Malamud |
| CR | 256 | CrossRocket over the raw characteristics, 64 operators per family, peers from trailing `r1` correlations with windows {20, 60} and basket sizes {5, 10, 20} |
| H+RFF, H+CR | 272 | concatenations |
| CR no-peer | 256 | ablation: the first three families only |
| CR identity | 256 | ablation: identity activation instead of hinge |
| reversal (−`r5`), momentum (`mom`) | 1 | unfitted: the characteristic itself, with no model |
| oracle (synthetic only) | 1 | −(`r5` minus its hidden-cluster mean): the planted signal, scored with labels no design sees |
| _[null]_ H, H+CR | 16, 272 | the same designs against a target shuffled across names within each date (seed 0) |

## Real data: S&P 500, 2022-08 to 2023-05

Rank IC. Random designs show the median over five seeds, with the range in brackets.

| Design | k-fold rank IC | NW t | walk-forward rank IC | NW t |
|---|---:|---:|---:|---:|
| reversal (−`r5`), unfitted | +0.0173 | +0.75 | +0.0112 | +0.41 |
| momentum (`mom`), unfitted | −0.0111 | −0.44 | −0.0295 | −1.01 |
| H | −0.0623 | −1.70 | −0.0345 | −1.02 |
| RFF | −0.0427 [−0.0450, −0.0312] | −1.64 | −0.0252 [−0.0319, −0.0149] | −1.02 |
| H+RFF | −0.0508 [−0.0522, −0.0437] | −1.70 | −0.0292 [−0.0336, −0.0205] | −1.03 |
| CR | −0.0623 [−0.0658, −0.0610] | −1.81 | −0.0307 [−0.0381, −0.0261] | −0.94 |
| H+CR | −0.0626 [−0.0656, −0.0615] | −1.80 | −0.0312 [−0.0379, −0.0270] | −0.95 |
| CR no-peer | −0.0631 [−0.0659, −0.0617] | −1.74 | −0.0262 [−0.0373, −0.0234] | −0.78 |
| CR identity | −0.0610 [−0.0629, −0.0597] | −1.76 | −0.0289 [−0.0316, −0.0259] | −0.89 |
| _[null] H, shuffled target_ | _+0.0006_ | _+0.18_ | _+0.0022_ | _+0.59_ |
| _[null] H+CR, shuffled target_ | _−0.0013_ | _−0.44_ | _+0.0008_ | _+0.25_ |

Paired differences use the same seed and the same folds. Each cell gives the median, the
range, and the number of seeds on which the difference is positive.

| Paired difference | k-fold | walk-forward |
|---|---:|---:|
| H+CR − H | −0.0003 [−0.0033, +0.0009], 1/5 | +0.0033 [−0.0034, +0.0075], 3/5 |
| CR − RFF | −0.0192 [−0.0311, −0.0178], 0/5 | −0.0059 [−0.0232, +0.0038], 1/5 |
| CR − CR no-peer | +0.0012 [−0.0012, +0.0018], 4/5 | −0.0013 [−0.0047, −0.0001], 0/5 |
| CR identity − CR | +0.0013 [+0.0010, +0.0029], 5/5 | +0.0048 [−0.0027, +0.0085], 4/5 |

**Every fitted design loses, and none of them significantly.** All seven fitted designs are
negative under both schemes, and the largest |NW t| is 1.81. The within-date shuffled null
gives |IC| ≤ 0.0022, so the pipeline is not manufacturing signal. The relations learned in
training simply do not hold out of sample in this year. The fixed-λ path is negative at
every λ for every fitted design, so this is not an artefact of how λ was selected.

**One block dominates the k-fold number.** Test block 3 (2022-12-13 to 2023-02-06) scores
−0.184 for H and −0.187 for CR (median over seeds). The other four blocks score between
−0.006 and −0.071. Under walk-forward the same block scores −0.024 for H and −0.013 for CR,
and the worst block is 2023-02-07 to 2023-03-30 (−0.067 for H, −0.068 for CR). The k-fold's
worst block comes from a model that was partly trained on later dates.

**The ridge sits at the top of its grid.** Inner validation chose λ = 1e6, the largest
value, in every k-fold fit of every real-data fitted design. The shuffled-target H+CR null
chose 100 in 3 of its 5 folds. Under walk-forward, 1e6 is the most common choice but not
the only one: it was chosen in 12–15 of 20 fits for the RFF and CrossRocket designs and in
3 of 4 for H. At that penalty on standardised features the ridge is at its limit. Each
coefficient is proportional to that feature's marginal covariance with the target, so the
fit is a covariance-weighted composite rather than a multivariate fit. The path confirms
this: IC at λ = 1e4 and at λ = 1e6 agree to within 0.0001. R² is `0.000%` because the
predictions shrink to nearly a constant. Rank IC does not depend on scale, so it survives.
The real-data comparison is therefore a comparison between marginal composites.

**Adding CrossRocket to handcrafted features changes nothing measurable.** The change is
−0.0003 under k-fold (positive on 1 of 5 seeds) and +0.0033 under walk-forward (3 of 5).
Both ranges span zero.

**Own-characteristic RFF is the least-bad fitted design on real data** (CR − RFF = −0.019
under k-fold, 0 of 5 seeds positive). This is not a win for RFF, since it is the least
negative of seven negative numbers. It is recorded because the synthetic control below
orders the two the other way.

## Synthetic positive control

A null on real data cannot show that the peer machinery works when there is something to
find, so the same pipeline also runs on a synthetic panel of identical shape: the same 503
tickers and the same dates, 93,558 rows. The daily return is:

- a market factor (1% vol),
- plus one of 12 hidden-cluster factors (1.5% or 3% vol),
- plus an idiosyncratic shock (1.5% vol),
- minus 0.04 × the name's own idiosyncratic shocks over the previous 5 days.

The planted signal is that reversal of the idiosyncratic 5-day move, which is cleanly
visible only relative to peers. The oracle scores it using the hidden cluster labels.

**The 3% setting was added after the 1.5% setting had been run and seen.** Treat it as a
second look chosen with knowledge of the first, not as a pre-registered test.

**Cluster-factor vol 1.5%** (equal to the idiosyncratic vol)

| Design | k-fold rank IC | NW t | walk-forward rank IC | NW t |
|---|---:|---:|---:|---:|
| _oracle (hidden clusters)_ | _+0.0945_ | _+30.04_ | _+0.0951_ | _+26.81_ |
| reversal (−`r5`), unfitted | +0.0820 | +4.73 | +0.0793 | +3.84 |
| H | +0.0692 | +2.87 | +0.0629 | +2.35 |
| RFF | +0.0597 [+0.0557, +0.0701] | +2.72 | +0.0544 [+0.0501, +0.0650] | +2.28 |
| H+RFF | +0.0673 [+0.0644, +0.0709] | +2.90 | +0.0596 [+0.0584, +0.0676] | +2.42 |
| CR | +0.0736 [+0.0713, +0.0739] | +3.59 | +0.0655 [+0.0645, +0.0665] | +2.74 |
| H+CR | +0.0735 [+0.0727, +0.0751] | +3.57 | +0.0649 [+0.0640, +0.0658] | +2.69 |
| CR no-peer | +0.0674 [+0.0660, +0.0716] | +2.91 | +0.0601 [+0.0576, +0.0615] | +2.34 |
| CR identity | +0.0756 [+0.0726, +0.0758] | +3.66 | +0.0615 [+0.0598, +0.0688] | +2.26 |
| _[null] H / H+CR_ | _+0.0017 / +0.0034_ | | _+0.0038 / +0.0044_ | |

**Cluster-factor vol 3%** (cluster moves dominate)

| Design | k-fold rank IC | NW t | walk-forward rank IC | NW t |
|---|---:|---:|---:|---:|
| _oracle (hidden clusters)_ | _+0.0635_ | _+27.14_ | _+0.0638_ | _+23.71_ |
| reversal (−`r5`), unfitted | +0.0517 | +1.98 | +0.0470 | +1.52 |
| H | +0.0291 | +0.76 | +0.0113 | +0.27 |
| RFF | +0.0004 [−0.0047, +0.0295] | +0.01 | −0.0011 [−0.0069, +0.0184] | −0.04 |
| H+RFF | +0.0113 [+0.0076, +0.0305] | +0.33 | +0.0073 [+0.0009, +0.0156] | +0.20 |
| CR | +0.0275 [+0.0230, +0.0330] | +0.81 | +0.0108 [+0.0053, +0.0145] | +0.29 |
| H+CR | +0.0287 [+0.0238, +0.0324] | +0.82 | +0.0108 [+0.0055, +0.0143] | +0.28 |
| CR no-peer | +0.0238 [+0.0193, +0.0292] | +0.65 | +0.0025 [+0.0008, +0.0080] | +0.06 |
| CR identity | +0.0294 [+0.0265, +0.0353] | +0.82 | +0.0119 [+0.0040, +0.0169] | +0.29 |
| _[null] H / H+CR_ | _+0.0014 / +0.0022_ | | _+0.0036 / +0.0028_ | |

Paired differences (median [range], seeds positive out of 5):

| Paired difference | 1.5% k-fold | 1.5% walk-forward | 3% k-fold | 3% walk-forward |
|---|---:|---:|---:|---:|
| CR − CR no-peer | +0.0053 [+0.0014, +0.0077], 5/5 | +0.0062 [+0.0030, +0.0075], 5/5 | +0.0032 [−0.0001, +0.0096], 4/5 | +0.0079 [+0.0028, +0.0093], 5/5 |
| CR − RFF | +0.0133 [+0.0038, +0.0179], 5/5 | +0.0100 [+0.0004, +0.0150], 5/5 | +0.0278 [−0.0006, +0.0326], 3/5 | +0.0120 [−0.0039, +0.0131], 3/5 |
| H+CR − H | +0.0044 [+0.0036, +0.0059], 5/5 | +0.0020 [+0.0011, +0.0029], 5/5 | −0.0004 [−0.0053, +0.0033], 1/5 | −0.0005 [−0.0058, +0.0030], 1/5 |
| CR identity − CR | +0.0018 [+0.0013, +0.0020], 5/5 | −0.0034 [−0.0053, +0.0032], 2/5 | +0.0035 [−0.0036, +0.0064], 4/5 | +0.0006 [−0.0056, +0.0038], 3/5 |

**The peer family carries signal when there is signal to carry.** CR beats its no-peer
ablation on 5 of 5 seeds in three of the four settings, and on 4 of 5 in the fourth. The
margins are small, +0.003 to +0.008 IC. They cannot be large, because on this DGP the
oracle's peer-relative version beats the raw reversal by only about 0.012 IC (+0.0945
against +0.0820 at 1.5%, and +0.0635 against +0.0517 at 3%). That is the most any
peer-relative feature could add here.

**CrossRocket matches handcrafted features; it does not beat them.** At 1.5%, CR alone is
+0.004 above H under k-fold. Adding it to H gains +0.002 to +0.004, positive on every seed
but small. At 3%, H+CR − H is about zero (positive on 1 of 5 seeds). H is a single
deterministic run, so it has no seed range. By the plan's own criterion, "if random
operators match handcrafted ones", they do match, on this synthetic control. On real data
nothing can be tested either way.

**CrossRocket beats own-characteristic random features on the synthetic panel.** It wins on
5 of 5 seeds at 1.5% and on 3 of 5 at 3%, which is the opposite of the real-data ordering.
Only the synthetic panel has signal, so only it can rank the two. Do not quote the ranking
without saying which setting it comes from.

**The fitted head is the bottleneck, not the representation.** The unfitted reversal beats
every fitted design in every setting. At 1.5% under k-fold it scores +0.082 against the best
fitted +0.076. At 3% it scores +0.052 against +0.029. `r5` is itself one of the
characteristics, and H contains its rank and z-score, so a linear head could represent the
reversal exactly. It fails to find it from at most about 140 training dates under an MSE loss. The
fixed-λ path does not rescue it: the best path value at 1.5% under k-fold is +0.0790
(CR identity at λ = 1e6).

The synthetic nulls stay within |IC| ≤ 0.0044.

## The contract checks

`tests/test_embed_cross_rocket.py` has 48 tests, all passing:

- `assert_no_lookahead` and `assert_prefix_invariant` for every output mode, with and
  without nulls. A peer basket built on *full-sample* correlations must fail both.
- Peers read only strictly earlier dates, and the non-peer families depend only on their
  own date.
- `fit_is_empty`: fitting on disjoint data gives byte-identical state.
- Per-entity outputs are permutation-equivariant and pooled outputs are
  permutation-invariant. A label-dependent variant must fail the same check.
- Determinism under a fixed seed, and one family's draws do not depend on which other
  families are enabled.
- The single-channel operators match `.xs.rank` and a polars median/MAD expression.

## Honesty notes

Read these before quoting a number.

- **The real-data result is a null, not a ranking.** The sample is one year: 186 dates in
  five test blocks, one of which dominates. Newey–West |t| is below 2 everywhere. These data
  cannot separate designs whose ICs differ by 0.02.
- **On real data, the ridge sits at the upper edge of its grid.** λ = 1e6 was chosen in every
  k-fold fit. A larger grid should give the same IC, because the path is flat from 1e4 upward
  and rank IC does not depend on scale in that limit. Even so, the real-data "models" are
  marginal-covariance composites, not multivariate fits.
- **Not survivorship-clean.** The ticker list is a mid-2023 snapshot with undocumented price
  adjustment, as on the [prefix-safe MiniRocket page](prefix-safe-rocket.md).
- **The synthetic DGP's choices are the result.** Those choices are 12 clusters, κ = 0.04, a
  planted 5-day horizon that matches the label horizon, the stated vols, and one data seed
  (2026) per setting. The 3% setting was added after the 1.5% one had been seen.
- **The seed ranges cover the random bank, not the data.** RFF and CR ranges are over five
  seeds of the feature bank on the same rows. H and the unfitted signals run once.
- **Price-only characteristics.** There are eight trailing price features, with no
  fundamentals and no volume. The RFF baseline is "in the style of" Kelly & Malamud, not a
  replication, since their work uses firm characteristics.
- **Pooled outputs are not evaluated.** The benchmark uses the per-entity default, so the
  pooled per-date statistics (`output="pooled"`) have no measured value yet.
- **Prior art, not a novelty claim.** The nearest lines of work are:
  - Kelly & Malamud's random-feature ridge, where the random features act on each asset's
    own characteristics;
  - multivariate MiniRocket's random channel combinations, where channels have fixed
    identity and entities do not;
  - untrained permutation-equivariant set and graph layers, of which `subset_agg` and
    `peer_dev` are special cases.

  None of these was checked against the papers for this benchmark. The property this page
  emphasises is equivariance plus strict causality at each date, not the idea of random
  features.

## Reproducing

```bash
# real data + both synthetic settings, 5 seeds, 256 features (~6 min)
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 POLARS_MAX_THREADS=2 \
    python benchmarks/cross_rocket/bench_cross_rocket.py

# real data only (~2 min)
python benchmarks/cross_rocket/bench_cross_rocket.py --skip-synthetic

# fewer seeds, a smaller bank
python benchmarks/cross_rocket/bench_cross_rocket.py --seeds 3 --n-features 128
```

**Every run overwrites `benchmarks/cross_rocket/results.json`**, including partial runs such
as `--skip-synthetic`. Copy the file first if you want to keep the published numbers. The
run is deterministic (seeded RNG throughout) and uses numpy and polars only.

## References

Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). *MiniRocket: A very fast (almost)
deterministic transform for time series classification.* KDD '21.

Kelly, B., Malamud, S., & Zhou, K. (2024). *The virtue of complexity in return prediction.*
Journal of Finance, 79(1).
