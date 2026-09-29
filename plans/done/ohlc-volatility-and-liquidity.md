# OHLC volatility, spreads, liquidity and realized-measure extensions — build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](../todo/00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): implemented (M1–M6).** The user directed integration, which
> stands in for the M0 caller gate. Shipped in `panelary/econ/features/`: `range_volatility`
> and `.panel.rolling_vol` with the `rs_vol` deprecation (M1, bug B6), `ohlc_spread` (M2),
> `price_impact` / `pastor_stambaugh_gamma` / `zero_return_share` / `fht_spread` (M3),
> `intraday_realized_measures` and the HAR variants (M4), realized kernel / TSRV /
> pre-averaging (M5, `_internal/_realized_kernel.py`) and `rough_hurst` / `RFSVForecaster`
> (M6, `_internal/_variogram.py`). Yang–Zhang stays the default. Not built: stored R
> `TTR` / `highfrequency` parity (no R available); the multivariate kernel moved to the
> covariance plan (note D7).

Leak-safe, vectorised estimators of **volatility from OHLC bars**, **bid–ask
spreads from OHLC bars**, **low-frequency price-impact / liquidity proxies**,
**intraday → daily realized measures** (semivariance, quarticity, jump-robust and
noise-robust variance, HAR extensions), and a **trailing rough-volatility
(Hurst) estimate**, for long panels of many entities.

Pure **numpy + polars**. **Zero new dependencies — not even the `fast` (numba)
extra**: nothing in this scope needs a sequential kernel (§8.6). Oracles
(`bidask` MIT, R `highfrequency` / `TTR` GPL, `scipy`) are **test-only**, behind
`pytest.importorskip` or stored reference values with provenance.

Reuse, do not reinvent:
`panelary.econ.features._common` (`sorted_panel`, `entity_arrays`, `ols`,
`per_entity_apply`), `panelary.econ.features._harrv` (`realized_variance`,
`bipower_variation`, `jump_component`, `har_terms`, `har_features`,
`daily_realized_measures`, `HARModel`, `_MU1_SQ_INV`), `panelary.econ.features._liquidity`
(`amihud_illiquidity`, `roll_spread`, `amivest_liquidity`, `liquidity_features`
incl. its inline `turnover_{w}`), `panelary._internal._numpy_stats` (`norm_ppf`,
`norm_cdf`), `panelary._internal._ffd` (the fixed-width causal FIR pattern, for
the RFSV forecaster), `panelary.core.protocol.PanelTransformer`,
`panelary.testing` (`assert_no_lookahead`, `assert_prefix_invariant`),
`panelary.registry` (`FeatureSpec`, `registry`).

---

## 1. Why this exists

**1a. The library ships a volatility operator whose name is false.**
`.panel.rs_vol` (`panelary/namespaces/panel.py:_expr_rs_vol`, line 122) is
`expr.rolling_std(window)`. Its own docstring admits it is "a causal,
single-series proxy", yet `docs/user-guide/features.md:324` advertises it as
"`rs_vol` (Rogers-Satchell-style volatility)", and it is one of the three
registered `.panel` specs. A user who reads the docs gets a close-to-close
standard deviation believing it is a range estimator that is 6× more efficient.
There is no Parkinson, Garman–Klass, Rogers–Satchell or Yang–Zhang estimator
anywhere in the package (verified by grep over `panelary/`, 2026-09-29).

**1b. The choice of range estimator is not cosmetic — measured.** Monte Carlo,
Brownian log-price, intraday variance 1 per bar, efficiency = Var(close-to-close
estimator) / Var(estimator), per-bar (`M` = monitoring steps per bar):

| Estimator | M=20 000 bias / eff | M=390 bias | M=78 bias | M=26 bias | bias, drift μ=0.5σ | bias, drift μ=σ |
|---|---|---|---|---|---|---|
| close-to-close c² (no demeaning) | −0.5% / 1.01 | +0.1% | −0.2% | +0.1% | **+25.2%** | **+100.8%** |
| Parkinson (1980) | −0.8% / **4.95** | −6.7% | −14.2% | −22.8% | **+7.7%** | **+35.8%** |
| Garman–Klass (1980) | −0.9% / **7.47** | −9.4% | −19.6% | −31.6% | +1.0% | **+10.7%** |
| Rogers–Satchell (1991) | −0.8% / **6.03** | −9.5% | −19.9% | −32.3% | −2.5% | −2.9% |

(drift columns at M=5000, whose discreteness bias alone is −2.5%; RS's drift
column is therefore *pure discreteness*: RS is drift-free, P and GK are not.)
Multi-bar estimators over n=21 bars (3000 windows), relative to the n=21
sample variance of close-to-close returns:

| Estimator (n=21) | f=0 (no overnight) eff | f=0.25 eff | f=0.25, M=390: bias / MSE-eff |
|---|---|---|---|
| Yang–Zhang (2000), α=1.34 | **7.26** | **8.10** | −6.2% / 6.14 |
| GK + overnight² ("GKYZ") | 7.90 | 8.64 | −7.1% / 5.98 |
| RS + overnight² | 6.25 | 7.35 | −7.3% / 5.19 |
| Parkinson + overnight² | 5.24 | 6.29 | −5.0% / 5.47 |
| RS only (ignores overnight) | — | bias **−21.9%**, MSE-eff 1.73 | bias −27.3%, MSE-eff 1.16 |

Four consequences drive this contract. (i) Every range estimator is worth
5–8× the data of close-to-close — a 21-day range volatility is as precise as
a ~150-day close-to-close one. (ii) Discrete monitoring biases them *down*
by ≈1.7–1.9/√M (RS, GK) and ≈1.2–1.3/√M (Parkinson): −9.5% on 1-minute-sampled highs
and lows, −20% on 5-minute bars. (iii) An intraday-only estimator silently
drops the overnight variance (−22% when overnight is a quarter of the
total). (iv) The often-quoted "Yang–Zhang is 14× close-to-close" is an
upper bound from the paper's setting; at n=21 we measure 7.3–8.1×, in line with
the "≈8×" also reported for it. The literature's efficiency numbers are
reproduced: P 5.2, GK 7.4, RS ≈6 (theory) vs 4.95 / 7.47 / 6.03 measured.

**1c. OHLC spread estimators differ by 2–3× in error — measured.** Simulated
efficient log-price (390 trade opportunities a day, 20% overnight variance),
trades at mid ± S/2 with random side, OHLC from observed trades, 21-bar windows,
1500 windows; EDGE computed with the MIT reference `bidask.edge` 2.1.0:

| Regime | EDGE bias / RMSE÷S | Corwin–Schultz (0-clip) | Abdi–Ranaldo (2-day corr.) | Roll |
|---|---|---|---|---|
| mid: S=0.5%, σ=2%/day | +34% / **0.67** | +82% / 0.92 | +69% / 0.84 | +101% / 2.23 |
| thin: same, 10% of opportunities trade | +38% / 0.71 | +44% / **0.57** | +69% / 0.84 | +99% / 2.18 |
| illiquid: S=2%, σ=3%, 2% trade | −4% / 0.32 | −41% / 0.44 | −13% / **0.25** | +8% / 0.82 |
| liquid: S=0.1%, σ=1.5% | RMSE÷S 4.2 | 4.6 | 5.3 | 9.5 |

And, decisively, as the window grows: EDGE's RMSE÷S falls 0.64 → 0.44 → 0.34
(n = 21 → 63 → 252, mid regime) and 0.33 → 0.20 → 0.10 (illiquid), while
Corwin–Schultz is stuck at 0.92 → 0.85 → 0.83 and 0.43 → 0.42 → 0.41 — the
zero-clipping of two-day estimates is a **bias floor** that more data cannot
remove. EDGE's signed squared spread averaged over windows is unbiased:
√mean(ŝ²) = 0.00479 / 0.00481 / 0.00485 for S = 0.005, and 0.0200–0.0204 for
S = 0.02. The liquid row is the honest caveat every doc must carry: **no OHLC
estimator resolves a spread that is small relative to daily volatility** on a
monthly window.

**1d. EDGE vectorises exactly, and our derivation is verified.** The reference
`edge_rolling` (bidask, pandas) expands EDGE into 34 rolling means. A
polars-native staged re-derivation (§7.2) matched the reference's published
test values to **3.9e-15** (`ohlc.csv`, full sample, 0.0101849034905478) and
**5.1e-16** (`ohlc-miss.csv`, 0.01013284969780197), and the per-window
`bidask.edge` at **every** index for windows 3, 4, 5, 21, 252 on complete data
(max abs error 1.0e-10 at w=3, ≤3.2e-16 at w≥21, zero NaN-pattern mismatches).
Two defects in the naive expansion were found and fixed in the prototype: (a)
with missing data, the rolling moments must be **null-harmonised per moment
family** or they average over different row sets than `edge()` does
(`bidask.edge_rolling` does not do this; it is only tested on complete data);
(b) when a family has ≤1 valid row in the window, the reference's own branch
choice is decided by floating-point residue (observed: v1 = 2.07e-25 vs v2 = 0,
so the reference puts all weight on e2 where exact arithmetic gives
(e1+e2)/2). With the guard of §7.2, all remaining mismatches on the
missing-data file (22/24/17/3 of ~10 000 rows at w = 3/4/5/8, zero at w ≥ 21)
are exactly those degenerate windows.

**1e. Rough-volatility estimates from range proxies are an artefact unless
corrected — measured.** fBM log-variance (ν=0.3), 1000-day windows, 200 reps,
H from the OLS slope of the log-variogram over lags 1..10:

| true H | oracle (true log σ²) | log RV (78 bars) | log RV, known-noise corrected | log Parkinson | log Parkinson, corrected |
|---|---|---|---|---|---|
| 0.1 | 0.101 (sd 0.014) | 0.090 | **0.101** (0.015) | **0.040** | **0.103** (0.028) |
| 0.3 | 0.298 (0.020) | 0.277 | **0.297** (0.021) | **0.151** | **0.300** (0.034) |
| 0.5 | 0.498 (0.025) | 0.474 | **0.498** (0.025) | **0.307** | **0.498** (0.038) |

A *smooth* (H = 0.5) volatility read through daily Parkinson ranges comes out
"rough" (H ≈ 0.31). Subtracting the proxy's known measurement-noise variance
from the variogram before the log-log fit removes the bias. Two alternatives were
tried and rejected: a noise-cancelling ratio estimator
H = ½ log₂[(m(4)−m(2))/(m(2)−m(1))] is unbiased but has sd 0.51 at H = 0.1;
a profile fit of m(Δ) = a + bΔ^{2H} is biased (0.163) and noisy (sd 0.098) at
H = 0.1.

**1f. The obvious polars formulation is 37–89× too slow — measured.** See §2.

We are not competing on having the most estimators. We are competing on each
number being the most accurate one available, computed at panel scale, with no
lookahead.

---

## 2. Measured facts that dictate the design

| Fact | Measurement | Consequence |
|---|---|---|
| EDGE as ONE nested expression, per-node `.over(id)`, 5000×5000 | **554 s, 17.5 GB RSS** | Banned. |
| Same, one outer `.over(id)`, 1000×5000 | **46.1 s** (streaming engine 84.9 s) vs staged **1.23 s** | polars does not dedupe the shared `shift`/`log` subtrees under ~40 rolling means. **Stage the computation** (§8.1). |
| EDGE staged (materialised per-bar columns → one `with_columns` of rolling means → row-local closed form), 25M rows | 13.7 s, 12.3 GB RSS; products inline 18.0 s, 9.6 GB | Staging is mandatory. |
| EDGE staged, **entity batches of 250 / 500 / 1000**, 25M rows | **6.23 s / 5.98 s / 6.71 s**, RSS 3.1 GB (250), 5.6 GB (1000); **bitwise identical to unbatched** | Batch by contiguous entity blocks, default 256. |
| Parkinson + GK + RS (rolling mean 21, `.over`) on 25M rows | **1.21 s** (20.7M rows/s) | Budget §11. |
| Yang–Zhang (21) on 25M rows | **1.50 s**; close-to-close `rolling_std` baseline 0.53 s | Budget §11. |
| Intraday, 19.7M 1-min rows (100 entities × 504 days × 390) | RV+BV+RS±+RQ+TPQ in one `group_by(entity, day)` **0.62 s**; + MedRV/MinRV 0.86 s; subsample-averaged 5-min RV 0.22 s; realized-kernel autocovariances h ≤ 20: 1.23 s; pre-averaging (k=20): 0.50 s | 500 entities × 2 years ≈ 98M rows: whole battery ≈ 17 s extrapolated. |
| polars `rolling_sum` accuracy (1e6 rows, spikes) vs `math.fsum` | max rel. err **3.4e-14**; recovers exactly after a 1e20 value or a NaN/inf leaves the window; **bitwise** prefix-invariant, chunked == contiguous, `.over` == per-series; identical on polars 1.35.2 and 1.44.2 | Native rolling sums are the accumulation primitive. |
| cumsum-difference (as in `_harrv._rolling_sum`) at T=1e6 | max rel. err **4.8e-11** (grows ∝ T/w) | New code must not use cumsum-difference. Existing function untouched (§3). |
| `pl.rolling_cov` and E[xy]−E[x]E[y] at mean/sd = 1e6 | both **5.6e-4** rel. err (same formula), both versions | Naive moments cancel. |
| Same after anchoring each variable at the entity's **first finite value** | **6.2e-16** | Anchoring is mandatory for regression proxies (§7.7). |
| `rolling_var` at mean/sd = 1e6 | 1.6e-10 (polars 1.44.2) vs **9.3e-9 (1.35.2)** | Stable but **version-dependent**: golden values across polars versions use `rtol=1e-8`, never bitwise. |
| Pre-averaging triangular filter vs half-block price-sum identity | max abs diff **2.0e-18** | O(n) pre-averaging independent of k (§7.5). |
| Mean of K-offset subsampled RVs vs (1/K)·Σ(K-bar rolling return)² | rel. diff **1.4e-16** (n divisible by K) | Subsampled RV is one rolling sum (§7.4). |
| EDGE parity with stored reference values / per-window `bidask.edge` | see §1d | Oracle design §10.3. |

---

## 3. What already exists — and must NOT be duplicated (verified 2026-09-29)

| File : symbol | What it is | This plan's relation |
|---|---|---|
| `econ/features/_harrv.py: realized_variance, bipower_variation, jump_component, realized_measures` | trailing **daily-return** RV/BV/J (cumsum-difference `_rolling_sum`, partial-window sums not rescaled) | Reused as-is for close-to-close RV of daily returns. Not modified (numerics noted in §13). |
| `econ/features/_harrv.py: daily_realized_measures` | intraday → per-day RV, BV, jump, rel_jump, n_obs | **Refactored** onto the shared per-day expression builders of `_realized.py` with a bitwise golden test; public signature and outputs unchanged. The new `intraday_realized_measures` is its superset. |
| `econ/features/_harrv.py: har_terms, har_features, HARModel` | Corsi HAR cascade + per-entity train-fitted OLS | **Extended additively** (HARQ / SHAR / HAR-CJ specs); `spec="har"` default is bitwise unchanged. |
| `econ/features/_liquidity.py: amihud_illiquidity, roll_spread, amivest_liquidity, liquidity_features` (+ inline `turnover_{w}`, `ret_vol_{w}`) | Amihud, Roll (on price *levels* by default), Amivest, turnover | Not duplicated. New proxies are added to the same file. Roll is the fourth column in §1c and is **not** reimplemented. |
| `econ/features/_common.py: rolling_beta` | naive E[xy]−E[x]E[y] rolling beta | Not reused for new regressions (cancellation, §2); flagged in §13, not changed. |
| `feature_extractors/_finance.py: realized_volatility` | whole-series scalar √Σ(Δlog p)² (`window` scope) | Different estimand; untouched. |
| `feature_extractors/_finance.py: signed_mci`, `.ts.marginal_cost_of_immediacy` | quoted half-spread from bid/ask | OHLC spreads are for when quotes are **absent**; no overlap. |
| `namespaces/panel.py: _expr_rs_vol` / `.panel.rs_vol` (expr + frame) | rolling std mislabelled | **Renamed** to `.panel.rolling_vol` with a warning alias (§6.4); the true RS lives in `range_volatility`. |
| `detect/_monitors.py: spot_variance, volatility_rescale` | one-sided kernel variance of level increments | Not duplicated; kernel-weighted range volatility is a non-goal (§4). |
| `_internal/_numpy_stats.py: norm_ppf` | Acklam + Halley (Halley step uses `np.vectorize(math.erf)`) | Reused for the FHT lookup table (≤ (w+1)² evaluations), never per row (§7.3). |

---

## 4. Scope, non-goals, sibling boundaries

**In scope.** (A) range volatility: Parkinson, Garman–Klass, Rogers–Satchell,
Yang–Zhang, GK-with-overnight, close-to-close, with annualisation, a
discreteness correction and a zero-range policy; (B) OHLC spreads: EDGE,
Corwin–Schultz (overnight-adjusted), Abdi–Ranaldo; (C) low-frequency proxies:
Kyle-λ price impact (signed √dollar-volume), Pástor–Stambaugh γ, zero-return
share (Zeros, Zeros2), FHT; (D) intraday → daily: RS⁺/RS⁻, signed jump
variation, RQ, TPQ, MedRV, MinRV, subsample-averaged RV, BNS/Huang–Tauchen jump
z and C/J split, and HARQ / SHAR / HAR-CJ in `HARModel`; (E) noise-robust
univariate: Parzen realized kernel with a day-t bandwidth, TSRV, pre-averaging;
(F) trailing rough-volatility H with proxy-noise correction; RFSV forecast (M6).

**Non-goals (decisions, not omissions).**
* **Meilijson (2011)**: efficiency 7.73 vs GK 7.4 (+4.5%), not drift-robust;
  the MLE bound is 8.47. Revisit only if §10.4's benchmark shows the gap matters.
* **Kunitomo drift-corrected range, Brownian-bridge estimators**: need the
  path, not OHLC.
* **LOT (Lesmond–Ogden–Trzcinka 1999) MLE and Hasbrouck (2009) Gibbs Roll**:
  a per-window numerical optimisation / MCMC — no O(1) rolling form. FHT is the
  closed-form LOT descendant and is shipped.
* **EWMA / kernel-weighted range volatility**: `pl.col(term).ewm_mean(...)` on
  the per-bar terms (`ohlc_variance_terms`, §6.1) is a one-liner and causal.
* **Bar construction** (time/volume/dollar/imbalance bars) — sibling 4.
* **GARCH / realized-GARCH / Hawkes** — sibling 9.
* **Volatility forecast evaluation (QLIKE, MZ regressions)** — sibling 1.

**Sibling boundaries — stated so nobody builds the same thing twice.**

| Sibling | Theirs | Ours | Seam |
|---|---|---|---|
| 3 covariance-and-market-state | realized covariance, **refresh-time sampling, Hayashi–Yoshida**, multivariate realized kernel, **realized semicovariance matrices** (BPQ 2020), any numba overlap sweep | only **univariate diagonal** measures (RS±, univariate RK, TSRV, PAV) | Their multivariate RK imports our Parzen weights + bandwidth rule from `_internal/_realized_kernel.py` (leaf) rather than re-deriving them. They own `refresh_time`. |
| 4 label-weights-and-event-sampling | bars | consumers of any bar series; docs state RV on business-time bars is a different estimand | none |
| 8 multiscale-complexity-features | generic Hurst / DFA / generalized-Hurst of returns or prices | H of a **log-volatility proxy** with proxy-noise correction | Whoever lands first owns `_internal/_variogram.py` (trailing multi-lag variogram as rolling means); the other imports it. |
| 9 tail-risk-and-self-excitation | GARCH family, EVT beyond existing `_evt.py`, Hawkes jumps | jump tests only as HAR-CJ inputs | none |
| 7 causal-state-space-and-regimes | Kalman / state-space stochastic volatility | — | none |
| 1 forecast-evaluation | losses for HAR/RFSV forecasts | forecasts as features | none |
| 2, 6, 10 | — | — | no overlap |

---

## 5. Hard invariants (every function)

1. **Prefix invariance, bitwise.** `f(x[:T])[t] == f(x[:T+k])[t]` for all
   `t ≤ T`, within one polars version. No quantity depends on `len(x)`: not
   windows, not Yang–Zhang's k, not bandwidths, not annualisation factors, not
   noise variances, not discreteness factors.
2. **Intraday → daily uses only day-t intraday data**, and the daily value is
   stamped at the **session close**. Broadcasting it onto that day's intraday
   rows is look-ahead by construction (`safe_scope="window"`, §9).
3. **float64 everywhere**; upcast Float32 before any arithmetic. **Log-price
   formulations only** (ratios, never level differences) in new code.
4. **Pairwise (two-bar) terms are indexed by the later bar** (t−1, t). The
   papers write CS and AR on (t, t+1); a `shift(-1)` implementation leaks one
   day and is a named trap (§9).
5. **Staged evaluation** (§8.1). Never a nested expression tree of many
   rolling means over shared shifted sub-expressions.
6. **No `rolling_map`, `map_elements`, per-window Python, cumsum-difference, or
   naive un-anchored second moments** in new code.
7. **Every `fit` is per fold on training rows** (HARQ demeaning, insanity-filter
   bounds, RFSV's H). Nothing is fitted on the full frame.
8. **Output nulls, not NaN**, for undefined values (Polars nulls, matching
   `har_features` / `per_entity_apply`), with a companion `n_valid` column
   where a count is informative.

---

## 6. Module placement and public API

### 6.1 Files

| File | Contents | Milestone |
|---|---|---|
| `panelary/_internal/_ohlc.py` (**new leaf**, numpy + polars only) | per-bar log decomposition (o, u, d, c), per-bar variance terms, CS/AR two-bar terms, the EDGE term table and closed-form combiner, invalid-bar policy, entity batching helper. Shared by `econ.features` and `namespaces.panel` — the `_ffd` pattern. | M1, M2 |
| `panelary/_internal/_realized_kernel.py` (**new leaf**) | Parzen weights, BNHLS bandwidth rule, pre-averaging constants ψ₁ᵏ, ψ₂ᵏ | M5 |
| `panelary/_internal/_variogram.py` (**new leaf**, or sibling 8's) | trailing multi-lag variogram + fixed OLS weights | M6 |
| `panelary/econ/features/_range.py` (**new**) | `range_volatility`, `ohlc_variance_terms` | M1 |
| `panelary/econ/features/_spread.py` (**new**) | `ohlc_spread` | M2 |
| `panelary/econ/features/_liquidity.py` (**extend**) | `price_impact`, `pastor_stambaugh_gamma`, `zero_return_share`, `fht_spread` | M3 |
| `panelary/econ/features/_realized.py` (**new**) | `intraday_realized_measures` + shared per-day expression builders | M4, M5 |
| `panelary/econ/features/_harrv.py` (**extend**) | `HARModel(spec=...)`; `daily_realized_measures` refactor | M4 |
| `panelary/econ/features/_rough.py` (**new**) | `rough_hurst`, `RFSVForecaster` | M6 |
| `panelary/namespaces/panel.py`, `panel.pyi` | `rolling_vol` (expr + frame), `rs_vol` alias | M1 |
| `panelary/econ/features/__init__.py` | re-exports + registry registration | each M |

### 6.2 Functions (frame-level, the primary surface)

All follow the existing `econ.features` convention: accept `DataFrame |
LazyFrame`, return a `DataFrame` sorted by `(entity, time)` with columns
appended; `window` is a row count; `min_periods=None` means `window`.

```python
def range_volatility(
    df, *, entity: str, time: str,
    open: str = "open", high: str = "high", low: str = "low", close: str = "close",
    method: Literal["yang_zhang", "gk_overnight", "rogers_satchell",
                    "garman_klass", "parkinson", "close_to_close"] = "yang_zhang",
    window: int = 21, min_periods: int | None = None, alpha: float = 1.34,
    output: Literal["vol", "var"] = "vol", periods_per_year: float | None = None,
    discrete_bars: int | None = None,
    invalid: Literal["null", "clip", "raise", "keep"] = "null",
    alias: str | None = None,
) -> pl.DataFrame            # adds f"vol_{method}_{window}" (+ "n_valid_{window}")

def ohlc_variance_terms(df, *, entity, time, open, high, low, close,
                        invalid="null") -> pl.DataFrame
    # per-bar o, u, d, c and the P / GK / RS / GK+o² terms, for users who want
    # their own (e.g. EWMA) aggregation. Row-local except o (uses C_{t-1}).

def ohlc_spread(
    df, *, entity: str, time: str, open=..., high=..., low=..., close=...,
    method: Literal["edge", "corwin_schultz", "abdi_ranaldo"] = "edge",
    window: int = 21, min_periods: int | None = None,
    negative: Literal["literature", "signed", "abs", "zero", "null"] = "literature",
    overnight_adjust: bool = True,          # Corwin-Schultz only
    batch_entities: int | None = 256,
    invalid="null", alias: str | None = None,
) -> pl.DataFrame
    # adds f"spread_{method}_{window}" AND f"spread_{method}_moment_{window}"
    # (the signed, averaging-safe moment: EDGE s^2, AR mean s^2, CS mean
    # unclipped S). Aggregate the moment, then take the root -- never average
    # clipped roots.

def price_impact(df, *, entity, time, returns, dollar_volume,
                 signed_volume: str | None = None, window: int = 63,
                 min_periods=None, intercept: bool = True,
                 volume_scale: float = 1e-6, alias=None) -> pl.DataFrame
def pastor_stambaugh_gamma(df, *, entity, time, returns, market_returns,
                           dollar_volume, window: int = 21, min_periods=None,
                           volume_scale: float = 1e-6, alias=None) -> pl.DataFrame
def zero_return_share(df, *, entity, time, returns, volume: str | None = None,
                      window: int = 21, min_periods=None, tol: float = 0.0,
                      alias=None) -> pl.DataFrame     # zeros_w; zeros2_w if volume
def fht_spread(df, *, entity, time, returns, window: int = 21,
               min_periods=None, tol: float = 0.0, alias=None) -> pl.DataFrame

def intraday_realized_measures(
    df, *, entity: str, session: str, time: str,
    price: str | None = None, returns: str | None = None,
    measures: Sequence[str] = ("rv", "rv_ss", "bv", "medrv", "minrv", "rs_pos",
                               "rs_neg", "sjv", "rq", "tpq", "jump_z", "n_obs"),
    subsample: int = 5, jump_alpha: float = 0.999,
    kernel_bandwidth: Literal["bnhls"] | int = "bnhls", kernel_max_lags: int = 30,
    preaverage_theta: float = 1.0, tsrv_k: Literal["auto"] | int = "auto",
    min_obs: int = 10, label: Literal["right", "left"] = "right",
) -> pl.DataFrame   # one row per (entity, session); opt-in: "rk", "tsrv", "pav", "medrq"

class HARModel(PanelTransformer):   # existing; additive keyword-only params
    def __init__(self, *, ..., spec: Literal["har", "harq", "shar", "har_cj"] = "har",
                 rq: str | None = None, rs_pos: str | None = None,
                 rs_neg: str | None = None, jump: str | None = None,
                 insanity_filter: bool = False) -> None: ...

def rough_hurst(df, *, entity, time, log_variance: str, window: int = 500,
                lags: Sequence[int] = tuple(range(1, 11)),
                noise_var: str | float | None = None, min_periods=None,
                alias=None) -> pl.DataFrame
class RFSVForecaster(PanelTransformer): ...     # M6: H fitted per fold
```

### 6.3 Namespace ops

* `.panel.rolling_vol(window)` (expression) and `df.panel.rolling_vol(columns,
  *, window, over, alias, suffix)` (frame) — **the honest name** for today's
  `rs_vol`; identical expression, bitwise-identical output.
* `.panel.rs_vol` — kept, emits `FutureWarning` once per process:
  *"rs_vol is a rolling standard deviation, not Rogers–Satchell. Use
  .panel.rolling_vol for identical output, or
  pn.econ.features.range_volatility(method='rogers_satchell') for the OHLC
  estimator."*
* **No `.panel` expression for the OHLC estimators.** A four-column estimator
  does not fit `pl.col(x).panel.op()`, and §2 shows the nested-expression form
  is 37–89× slower. A frame-level `df.panel.range_vol(...)` wrapper is
  deferred until a caller asks for it (AGENTS.md rule).

### 6.4 `rs_vol` deprecation path (never silently change outputs)

1. M1: add `rolling_vol` (expr + frame + `.pyi` stubs + `FeatureSpec`); `rs_vol`
   becomes a thin alias with `FutureWarning`; its `FeatureSpec` stays (tier B →
   **D**, `source` notes "deprecated alias of rolling_vol").
2. Fix `docs/user-guide/features.md:324`, `docs/quickstart.md:51`,
   `docs/concepts/{panel-data,two-tier-api,leak-safety}.md`,
   `docs/api-reference/registry.md:45` to say "rolling standard deviation"; add a
   `MIGRATING.md` entry and a `CHANGELOG.md` line.
3. Golden test: `rs_vol` output on a fixed frame is bitwise equal to the
   pre-change value **and** to `rolling_vol`.
4. Removal no earlier than two minor releases later. **The name `rs_vol` is
   never repurposed to mean Rogers–Satchell** — that would change existing
   users' numbers without an error.

---

## 7. Algorithms

Notation per bar t (natural logs): O, H, L, C; overnight o = O_t − C_{t−1};
u = H − O; d = L − O; c = C − O. Intraday session t has M returns r_i.

### 7.1 Range volatility (daily or any bar frequency)

| Method | Per-bar term / window formula | Eff. (theory; measured §1b) | Drift | Opening jump | Notes |
|---|---|---|---|---|---|
| `close_to_close` | sample var of r = C_t − C_{t−1}, ddof 1 | 1 | demeaned: robust | included | baseline; zero-mean variant = existing `realized_variance` |
| `parkinson` | (u − d)² / (4 ln 2) | 5.2; 4.95 | biased up (+36% at μ=σ) | missed | |
| `garman_klass` | ½(u − d)² − (2 ln 2 − 1)c² | 7.4; 7.47 | biased up (+11% at μ=σ) | missed | GK's "best" 0.511/0.019/0.383 form measured identical (7.47); not shipped |
| `rogers_satchell` | u(u − c) + d(d − c) | ≈6; 6.03 | **unbiased** | missed | |
| `gk_overnight` | mean over window of o² + GK term | n=21: 7.90 / 8.64 | biased (GK part, and o not demeaned) | included | practitioner "GKYZ" † |
| `yang_zhang` | V_o + k·V_c + (1 − k)·V_RS; V_o, V_c sample variances (ddof 1) of o and c over the window; V_RS mean RS term; **k = (α−1)/(α + (n+1)/(n−1))**, α = 1.34 | n=21: 7.26 / 8.10 | **unbiased** | **consistent** | n = the row's own rolling count of valid bars (causal), never `len(x)` |

**Default `yang_zhang`**: the only estimator here that is drift-free *and*
consistent under opening jumps, at ≈8× efficiency. `gk_overnight` measured
~7% more efficient at zero drift; the M1 benchmark (§11) re-measures both under a
calibrated equity drift (|μ/σ| ≤ 0.1 per day) *before release*. The default flips
to `gk_overnight` only if its MSE is lower by > 5% in every regime of §10.4 —
decided by the benchmark, recorded in the docs, and never changed after release.

**Discreteness correction** (`discrete_bars=M`): divide the P / GK / RS
components by the stored factor b(M) = E[estimate]/σ² for a Gaussian random walk
monitored at M points. Measured today: M=390 → P 0.933, GK 0.906, RS 0.905;
M=78 → 0.858, 0.804, 0.802; M=26 → 0.772, 0.684, 0.677. M1 regenerates the table
at ≥ 10⁶ bars per M on a log grid M ∈ [5, 23 400] with standard errors
(`benchmarks/ohlc_vol/discreteness_table.py`, seeded), stores it as constants, and
interpolates in log M. It is a constant per M, so it is leak-free. **Off by
default**: on real data M (the trade count) is unknown and microstructure noise
widens the range, partially offsetting the bias. In Yang–Zhang only the RS
component is corrected. Rogers, Satchell & Yoon (1994) † give an analytic RS
correction; the stored table supersedes it (it also covers P and GK).

**Invalid bars** (`invalid=`): a bar is valid iff all of O, H, L, C are finite
and > 0 and L ≤ min(O, C) ≤ max(O, C) ≤ H. `"null"` (default) nulls the bar and
counts it; `"clip"` sets H = max(H, O, C) and L = min(L, O, C) (row-local, so
causal); `"keep"` is for parity tests; `"raise"`. CRSP-style negative prices
(bid/ask midpoint flag) must be `abs()`-ed by the caller; this is documented, and
the function never flips signs itself.

**Zero-range bars** (H = L) are legitimate observations of a small variance and
are **kept** (term = 0). They are never back- or forward-filled.

**Annualisation**: `periods_per_year` is a user constant (252, 365, 252·390 …),
applied after the rolling mean. It is **never** inferred from the data.

**Complexity**: O(1) per row per entity: 1–3 native rolling moments.

### 7.2 EDGE (Ardia, Guidotti & Kroencke 2024) — rolling closed form

Per bar t (pairs (t−1, t), indexed by t): m = (H + L)/2; r₁ = m_t − O_t,
r₂ = O_t − m_{t−1}, r₃ = m_t − C_{t−1}, r₄ = C_{t−1} − m_{t−1}, r₅ = O_t − C_{t−1};
τ = 1{H_t ≠ L_t or L_t ≠ C_{t−1}} (null if H_t, L_t or C_{t−1} is null);
π_{o1} = τ·1{O_t ≠ H_t}, π_{o2} = τ·1{O_t ≠ L_t}, π_{c1} = τ·1{C_{t−1} ≠ H_{t−1}},
π_{c2} = τ·1{C_{t−1} ≠ L_{t−1}}.
Validity masks: **V₁ = valid(r₁, r₂, r₃, r₄, τ)**, **V₂ = valid(r₁, r₄, r₅, τ)**.

Window scalars (rolling means over w−1 pairs, **each over its own non-null
rows**): p_τ = E[τ], p_o = E[π_{o1}] + E[π_{o2}], p_c = E[π_{c1}] + E[π_{c2}],
a₁ = E[r₁]/p_τ, a₃ = E[r₃]/p_τ, a₅ = E[r₅]/p_τ, A = −4/p_o, B = −4/p_c.

The reference sets x₁ = A(r₁r₂ − a₁τr₂) + B(r₃r₄ − a₃τr₄) and
x₂ = A(r₁r₅ − a₁τr₅) + B(r₄r₅ − a₅τr₄). Because τ² = τ, every moment of x₁, x₂
is linear in rolling means of per-bar products. **Family 1 (masked to V₁)**:
r₁r₂, τr₂, r₃r₄, τr₄, (r₁r₂)², τr₁r₂², τr₂², (r₃r₄)², τr₃r₄², τr₄², r₁r₂r₃r₄,
τr₁r₂r₄, τr₂r₃r₄, τr₂r₄. **Family 2 (masked to V₂)**: r₁r₅, τr₅, r₄r₅, τr₄,
(r₁r₅)², τr₁r₅², τr₅², (r₄r₅)², τr₄²r₅, τr₄², r₁r₄r₅², τr₁r₄r₅, τr₄r₅², τr₄r₅.
Plus E[τ], E[π·], E[r₁], E[r₃], E[r₅] (unmasked) and counts n₁ = Σ1{V₁},
n₂ = Σ1{V₂}, n_τ = Στ. That is 36 rolling means plus 3 rolling sums.

```
e1 = A(E r1r2 − a1 E τr2) + B(E r3r4 − a3 E τr4)
E[x1²] = A²(E(r1r2)² − 2a1 E τr1r2² + a1² E τr2²)
       + B²(E(r3r4)² − 2a3 E τr3r4² + a3² E τr4²)
       + 2AB(E r1r2r3r4 − a3 E τr1r2r4 − a1 E τr2r3r4 + a1a3 E τr2r4)
e2, E[x2²]: the same with (r1r5, τr5, r4r5, τr4; a1, a5)
V_k = 0 if n_k ≤ 1 else max(E[x_k²] − e_k², 0)            # degenerate-family guard
s² = (V2·e1 + V1·e2)/(V1+V2) if V1+V2 > 0 else (e1+e2)/2
null unless n_τ ≥ 2 and p_o ≠ 0 and p_c ≠ 0 and n_1, n_2 ≥ min_periods
```

This equals the per-window `bidask.edge` exactly in exact arithmetic
(**verified**, §1d), including missing data, which `bidask.edge_rolling` does
not handle. **Window semantics match bidask**: `window` counts price bars; there
are w − 1 pairs. `min_periods=None` requires all w − 1 pairs valid in both
families, like `edge_rolling`'s pandas default; a smaller `min_periods`
reproduces `edge()`'s nan-mean semantics on gappy windows. `window=None` means
expanding, with an explicit `min_periods` — **never** `window=len(df)` (bidask's
`edge_expanding` does exactly that, which is length-dependent under pandas'
default `min_periods`).

`negative="literature"` for EDGE = √|s²| (the reference default); `"signed"` =
sign(s²)·√|s²| (monotone in the unbiased moment, **recommended for ML
features**); `"zero"`, `"null"`. The moment column always carries s².

**Numerics**: V_k = E[x²] − e² does not cancel for return products (e² ~ 1e-12
vs E[x²] ~ 1e-7 at a 0.5% spread). √ near 0 amplifies: |Δs| ≈ |Δs²|/(2s), so
parity is asserted on **s²**, not s (the missing-data cases with s = 0 vs 1e-9
are s² differences of 1e-18).
**Cost**: O(1) per row; 39 rolling reductions; 6.2 s per 25M rows batched (§2).

### 7.3 Corwin–Schultz (2012), Abdi–Ranaldo (2017), low-frequency proxies

**Corwin–Schultz**, pair (t−1, t). Overnight adjustment (CS 2012 §III.A): if
L_t > C_{t−1}, subtract (L_t − C_{t−1}) from H_t and L_t; if H_t < C_{t−1}, add
(C_{t−1} − H_t). Then β = (H_{t−1} − L_{t−1})² + (H_t − L_t)²,
γ = (max(H_{t−1}, H_t) − min(L_{t−1}, L_t))², α = (√(2β) − √β)/(3 − 2√2) −
√(γ/(3 − 2√2)), and **S = 2(e^α − 1)/(1 + e^α) = 2·tanh(α/2)** — computed as
`tanh`, which is accurate near α = 0, where e^α − 1 loses digits.
`"literature"` = clip S at 0, then average over the window (CS 2012 §III.C). The
moment column is the mean of unclipped S.

**Abdi–Ranaldo**: η = (H + L)/2; s²_t = 4(C_{t−1} − η_{t−1})(C_{t−1} − η_t).
`"literature"` = the two-day-corrected mean of √max(s²_t, 0). The moment column is
mean(s²_t), whose `"signed"` root is AR's "monthly" estimator.

**Price impact (Kyle-λ proxy, Hasbrouck 2009 style)**: x_t = signed_volume_t, or
sign(r_t)·√(DV_t·volume_scale); λ = Cov_w(r, x)/Var_w(x) (`intercept=True`) or
Σrx/Σx². **Honesty note in the docstring**: with sign(r) as the order-flow sign,
Cov(r, x) = E[|r|√DV] > 0 mechanically, so λ is a price-impact *magnitude*
measure akin to Amihud. GHT (2009) found Amihud the best low-frequency
price-impact proxy, and the docstring must say so.

**Pástor–Stambaugh γ**: y_s = r^e_s (= r_s − r^m_s), X_s = [1, r_{s−1},
sign(r^e_{s−1})·v_{s−1}] over the window ending at t (all inputs ≤ t). By
Frisch–Waugh on window-centred moments S_ab (anchored, §7.7):
γ = (S₁₁S₂y − S₁₂S₁y)/(S₁₁S₂₂ − S₁₂²). That is nine rolling means, and the result
is null when the determinant is ≤ 1e-12·S₁₁S₂₂. Tier C: PS (2003) themselves
note individual-stock γ is very noisy.

**Zero-return share**: zeros = mean 1{|r| ≤ tol}; zeros2 = mean 1{|r| ≤ tol and
volume > 0} (GHT 2009). Stale vendor fills (carried-forward close, volume 0) are
the reason zeros2 exists.

**FHT (Fong, Holden & Trzcinka 2017)**: FHT = 2σ_w·Φ⁻¹((1 + z)/2), σ_w the
rolling std (ddof 1), z the zero share; null if z ≥ 1. **Φ⁻¹ is a lookup**: z =
k/n with integer counts, so evaluate `_numpy_stats.norm_ppf` once on the
(window + 1)² grid of (k, n) and gather. That costs O(w²) at construction and
avoids the per-row `np.vectorize(math.erf)` Halley step.

**Turnover** exists (`liquidity_features`); not duplicated.

### 7.4 Intraday → daily realized measures (one `group_by(entity, session)`)

Returns are within-session log differences; **the first return of a session is
from the session's first price, never from the previous close** (overnight is
separate). Rows are sorted by `(entity, session, time)`; `label="right"` means
bars are stamped at their close. With `"left"` (open-stamped bars) the function
shifts stamps to the close before any as-of use (§9 trap 10).

| Measure | Formula (per session, M returns) | Robust to | Asy. var / IQ (†) |
|---|---|---|---|
| `rv` | Σr² | — | 2 |
| `rv_ss` (K = `subsample`) | (1/K)·Σ_{i≥K}(p_i − p_{i−K})² = mean of the K offset RVs (verified identity) | noise (mildly) | — |
| `bv` | (π/2)·(M/(M−1))·Σ\|r_i\|\|r_{i−1}\| (existing constant `_MU1_SQ_INV`) | jumps | 2.61 |
| `medrv` | (π/(6 − 4√3 + π))·(M/(M−2))·Σ med(\|r_{i−1}\|,\|r_i\|,\|r_{i+1}\|)², with med = a+b+c−max−min | jumps, zero returns | 2.96 |
| `minrv` | (π/(π−2))·(M/(M−1))·Σ min(\|r_i\|,\|r_{i+1}\|)² | jumps | 3.81 |
| `rs_pos`, `rs_neg` | Σr²1{r>0}, Σr²1{r<0}; rs_pos + rs_neg = rv exactly | — | — |
| `sjv` | rs_pos − rs_neg (Patton–Sheppard 2015) | — | — |
| `rq` | (M/3)·Σr⁴ | — | — |
| `tpq` | M·μ_{4/3}⁻³·(M/(M−2))·Σ\|r_i r_{i−1} r_{i−2}\|^{4/3}, μ_{4/3} = 2^{2/3}Γ(7/6)/Γ(1/2) | jumps | — |
| `medrq` (opt-in) | ADS (2012) median quarticity † | jumps | — |
| `jump_z` | [(RV − BV)/RV] / √(θ·(1/M)·max(1, TPQ/BV²)), θ = π²/4 + π − 5 | — | — |
| C/J | J = 1{z > Φ⁻¹(α)}·max(RV − BV, 0), C = RV − J (α = 0.999 constant) | — | — |

Shifts inside `agg` are within-session, and MedRV's r_{i+1} is day-t data, so it
is allowed. Sessions with M < `min_obs` give nulls. Default headline variance
is `rv_ss` with K=5 on 1-minute bars: Liu, Patton & Sheppard (2015) find 5-minute
RV hard to beat, and averaging the 5 offsets is free here (0.22 s per 19.7M rows).

**HAR extensions** (in `HARModel`, fit per fold):
* HARQ (BPQ 2016): RV_{t+h} = β₀ + (β₁ + β₁Q·(RQ_t^{½} − m̄))·RV_t + β₂RV_w + β₃RV_m;
  m̄ is the **training-fold** mean of RQ^{½}, frozen at fit.
* SHAR (Patton–Sheppard 2015): RV_d → (RS⁺_d, RS⁻_d).
* HAR-CJ (ABD 2007): C_d, C_w, C_m, J_d, J_w, J_m.
* `insanity_filter` (BPQ 2016): clip forecasts to the [min, max] of the training
  target, falling back to the training mean; both are frozen at fit.
* `spec="har"` is bitwise unchanged (golden test).

### 7.5 Noise-robust univariate measures (M5)

* **Realized kernel (BNHLS 2008), Parzen**, which is non-negative by
  construction: K = γ₀ + Σ_{h=1}^{H} k((h−1)/H)·2γ_h, γ_h = Σ_j r_j r_{j−h},
  k(x) = 1 − 6x² + 6x³ on [0, ½], 2(1 − x)³ on [½, 1]. End points are jittered by
  averaging m = 2 prices. **Bandwidth, per day from day-t data only** (BNHLS
  2009): H*_t = c*·ξ_t^{4/5}·n_t^{3/5}, c* = 3.5134 †, ξ²_t = ω̂²_t/IV̂_t,
  ω̂²_t = mean over q offsets of RV_dense/(2n), and IV̂_t = 20-minute subsampled RV.
  **Vectorised**: aggregate γ_0..γ_{H_max} as fixed `agg` expressions
  (`kernel_max_lags`, default 30), then compute K = Σ_h w_h(H*_t)·γ_h row-wise
  with the Parzen weights as expressions of the H*_t column. If H*_t > H_max the
  value is capped and the row is flagged. Measured 1.23 s per 19.7M rows at H_max
  = 20. On 1-minute bars ω̂² is dominated by IV/(2n) (H* ≈ 9 with zero noise);
  document this, and offer `omega2="debiased"` = max(RV_dense − RV_sparse, 0)/(2n).
* **TSRV (ZMA 2005)**: (1 − n̄/n)⁻¹·(RV_ss(K) − (n̄/n)·RV_all), n̄ = (n − K + 1)/K.
  With `tsrv_k="auto"`, K_t = ⌈c*_t·n^{2/3}⌉ with c*_t from day-t ω̂² and RQ †.
  Negative values are kept, with a flag.
* **Pre-averaging (JLMPV 2009)**, g(x) = min(x, 1 − x), k = 2⌈θ√n/2⌉. Pre-averaged
  returns via the **half-block identity** Ȳ_i = (1/k)(Σ_{j=k/2}^{k−1}p_{i+j} −
  Σ_{j=0}^{k/2−1}p_{i+j}), which is two within-session `rolling_sum`s of log
  price (verified to 2e-18), so the cost is O(n) independent of k.
  PAV = (n/(n − k + 2))·(1/(kψ₂ᵏ))·ΣȲ_i² − (ψ₁ᵏ/(2k²ψ₂ᵏ))·Σr_i², with
  ψ₁ᵏ = kΣ_{j=0}^{k−1}(g((j+1)/k) − g(j/k))² and ψ₂ᵏ = (1/k)Σ_{j=1}^{k−1}g(j/k)².
  **This normalisation was derived for this plan from E[Ȳ²] ≈ kψ₂ᵏ·IV/n +
  (ψ₁ᵏ/k)·ω²**; the implementer confirms it against JLMPV (2009) † and the
  stored `highfrequency::rMRCov` values before M5 ships.

### 7.6 Rough volatility (M6)

x_t = a log-variance proxy (log `rv`/`rv_ss`, or log of a range-variance term).
For each lag Δ ∈ `lags`: m̂_Δ(t) = rolling_mean over (W − Δ) rows of
(x_s − x_{s−Δ})², a native rolling mean per lag (10 of them). Noise correction:
m̃_Δ = m̂_Δ − rolling_mean(ν_s + ν_{s−Δ}), where ν_s is the proxy's log-noise
variance on day s:
* RV proxy, `noise_var="rq"` (**default for RV**): ν_s = (2/3)·Σr⁴/RV², by the
  delta method from day-s data only. It equals ≈2/M under constant intraday
  vol (the exact trigamma(M/2) = 0.02597 at M = 78 was used in §1e) and picks
  up intraday seasonality automatically. `"tpq"` swaps in TPQ for jump
  robustness.
* Range proxy: ν = the stored constant for the estimator and `discrete_bars`
  (Parkinson at M = 390: 0.360, measured). With `noise_var=None` on a range
  proxy the function **raises** (the §1e bias is too large to allow silently).

Ĥ_t = ½·Σ_Δ w_Δ·log m̃_Δ(t), with fixed OLS weights w_Δ = (ℓ_Δ − ℓ̄)/Σ(ℓ − ℓ̄)²
and ℓ = log Δ. It is a pure expression; any m̃_Δ ≤ 0 gives null. Default W = 500.
**Docstring caveat, mandatory**: this is a *descriptive* roughness feature of
the proxy path. Cont & Das (2024) † show that roughness can be an estimation
artefact even for smooth volatility, and Fukasawa, Takabatake & Westphal (2022) †
find H < 0.5 after correcting. Our correction addresses only i.i.d. proxy
error (validated in §1e).
**RFSV forecast** (GJR 2018 †): E[log σ²_{t+Δ}] ∝ Δ^{H+½}∫ log σ²_s /
((t − s + Δ)(t − s)^{H+½}) ds, discretised as a normalised, truncated (L lags)
causal FIR over x. H is fitted per fold (pooled or per entity) and frozen, and
the FIR reuses the `_ffd.frac_diff_expr` shape (weighted sum of shifts) with
these weights.

### 7.7 Numerical-accuracy rules (all estimators)

1. **Accumulate with polars native `rolling_sum` / `rolling_mean`** (error
   bounds per §2). For a dense (E, T) numpy path, if one is ever needed, use
   **blocked re-anchored prefix sums** (block B = 4096): error ≤ ~B·ε·max|x|,
   independent of T. A plain cumsum-difference is banned.
2. **Anchor second moments**: before any rolling covariance or regression,
   subtract each variable's **first finite value within the entity**
   (`x.drop_nulls().first().over(entity)`), which is causal, constant per entity
   and therefore prefix-invariant. Measured 5.6e-4 → 6.2e-16.
3. **Log first**: all estimators work on log-price differences. The within-bar
   terms (u, d, c, RS, GK, P, EDGE's r₁) are invariant to any multiplicative
   factor common to a bar's O, H, L, C.
4. **Variance clamps**: sample variances and EDGE's V_k are clamped at 0; RS and
   GK per-bar terms may be negative on inconsistent bars (hence `invalid=`).
5. **Transcendental hygiene**: `tanh` for CS; `log1p`/`expm1` wherever x ≈ 0.
6. **Cross-version tolerance**: `rtol=1e-8` golden values across polars 1.35 and
   1.42+ (the `rolling_var` difference, §2); bitwise only within a version.

---

## 8. Vectorisation and parallelism rules

1. **Stage** every multi-moment estimator: (i) `with_columns` of logs and the
   single `shift(1).over(entity)`; (ii) per-bar terms, row-local; (iii) **one**
   `with_columns` holding every `rolling_mean(...).over(entity)`; (iv) a
   row-local closed form. Products may be written inline in stage (iii); that
   saves memory at a small time cost (§2).
2. **Batch entities** in contiguous blocks (`batch_entities`, default 256).
   Measured 2–3× faster with 4× less memory, bitwise identical. Implemented in
   `_internal/_ohlc.py` as a loop over `lf.filter(entity in block)` on the
   `(entity, time)`-sorted frame, then `pl.concat`.
3. **`.over(entity)` is the parallelism**: polars multithreads windowed
   expressions across groups; no Python threads, no multiprocessing.
4. **Banned shortcut**: rolling over the whole sorted column and nulling each
   entity's first w − 1 rows. The rolling kernel's floating state crosses entity
   boundaries, so appending rows to entity A could perturb entity B at the ulp
   level. This is unmeasured and must stay banned unless a test proves it bitwise
   (§10.1).
5. **Intraday**: a single `group_by([entity, session]).agg([...])` carrying every
   requested measure; within-session shifts and `rolling_sum` inside `agg`.
6. **numba**: none needed here. Refresh-time / Hayashi–Yoshida sweeps (the
   genuinely sequential kernels) belong to sibling 3.
7. **Lazy in, eager out**: accept `LazyFrame`; do not rely on CSE across stages.

---

## 9. Leak-safety design and named traps

| # | Trap | Where it bites | Safe default here | Test |
|---|---|---|---|---|
| 1 | Two-day estimators written on (t, t+1) implemented with `shift(-1)` | CS, AR | pairs indexed by the later bar | perturbation at t+1 leaves row t unchanged; a deliberately leaky variant is **detected** |
| 2 | Yang–Zhang k with n = `len(series)` or a k "optimised" on the full sample | YZ | k from α and the row's rolling count | prefix test; leaky-k variant detected |
| 3 | Realized-kernel bandwidth / ω² / IV pooled over all days | RK, TSRV | day-t only | perturb day d+1 intraday rows; day ≤ d outputs unchanged |
| 4 | Hansen–Lunde whole-day scaling with a full-sample ratio | RV + overnight | not offered; RV_total = RV + o² explicitly, or a trailing ratio | — (documented) |
| 5 | Broadcasting a day's RV onto that day's intraday rows | all intraday | `safe_scope="window"`; output keyed by session | conformance §4 evidence: per-row broadcast fails prefix check |
| 6 | Back-adjusted price **levels** (factor from future splits/dividends) | Roll on levels (existing default), Amihud's DV from adjusted price × raw volume, λ in $ | log-ratio estimators; cross-bar ratios (o, EDGE r₂..r₅, CS γ) need raw prices plus a PIT adjustment factor as-of t (see `plans/done/pit-asof-join-and-calendar-embargo.md`), or total-return-adjusted series whose *ratios* are PIT-correct | synthetic split at day s: level-based measures before s change after re-adjustment; ours do not |
| 7 | Zero-range / missing bars filled from the next bar or interpolated | all OHLC | kept as 0 / nulled; never filled | perturb t+1, row t unchanged |
| 8 | `periods_per_year` inferred from the sample | annualisation | user constant | signature has no inference path |
| 9 | Intraday cleaning thresholds (e.g. 10×MAD) from the full sample | intraday | only day-t or trailing thresholds, if any | — |
| 10 | Open-stamped bars as-of joined at their stamp | intraday → decision rows | `label=` handling, stamp at close | unit test |
| 11 | HARQ demeaning / insanity bounds from the full sample | `HARModel` | training-fold constants | `assert_no_train_test_leak` |
| 12 | Rough H noise variance pooled over the sample; RFSV's H fitted on all data | `rough_hurst`, `RFSVForecaster` | day-s ν; H per fold | prefix + train/test tests |
| 13 | `edge_expanding`-style `window=len(df)` | EDGE | `window=None` + explicit `min_periods` | prefix test |
| 14 | Discreteness factor estimated from the panel | range vol | stored constants per M | signature |

All rowwise ops pass **both** `assert_no_lookahead` and `assert_prefix_invariant`
(neither subsumes the other; see `panelary/testing.py`).

---

## 10. Tests

`--strict-markers` is on; the existing `slow` / `benchmark` markers suffice.

### 10.1 Leak-safety and conformance
* **Registry conformance**: extend `tests/test_registry_conformance.py` with
  `_FRAME_OPS` adapters that build a **positive OHLC probe** from the probe
  panel, causally (close = 100·exp(x/20); open = previous close·exp(z/100);
  high = max(O, C)·exp(|z|/50); low = min(O, C)·exp(−|x|/60)). The probe keeps
  the interior null and adds one zero-range bar and one H = L = C_{t−1} bar
  (τ = 0). Fix `_call`'s rendering (it hard-codes `panelary.factor`).
  `intraday_realized_measures` goes into the `"window"` bucket with a
  broadcast adapter.
* `tests/test_ohlc_leak_safety.py`: every trap in §9 marked "detected", with
  the leaky variant written in the test to prove the check has teeth;
  `batch_entities` ∈ {None, 1, 3, 256} bitwise equal; the banned shortcut in
  §8.4 shown non-bitwise (or, if it turns out bitwise, the ban is lifted with
  the evidence recorded).

### 10.2 Hand-computed and identity tests
* 2–3-bar hand cases for every range and spread estimator; YZ's k formula;
  annualisation; invalid-bar policies; Float32 upcast; scale invariance
  (O, H, L, C × 7.3 → |Δ| ≤ 1e-14).
* RS drift-independence identity; CS `tanh` form vs the exponential form.
* Intraday identities: rs_pos + rs_neg == rv (bitwise when summed in the same
  order; 1e-15 otherwise); subsample identity; pre-averaging half-block identity
  vs direct triangular weights (≤ 1e-15); `daily_realized_measures` golden
  bitwise after the refactor; `HARModel(spec="har")` golden bitwise.
* Regression proxies vs `np.linalg.lstsq` per window (≤ 1e-10), including the
  mean/sd = 1e6 adversarial input (anchored: ≤ 1e-12).

### 10.3 Parity with reference implementations (`tests/test_ohlc_oracle.py`)

| Ours | Oracle | Licence | Form |
|---|---|---|---|
| EDGE | `bidask` 2.1.0 `edge` / `edge_rolling` | MIT | **vendored** first 2 000 rows of `pseudocode/ohlc.csv` and `ohlc-miss.csv` (gzipped, ~50 KB, with MIT notice in `tests/data/NOTICE`) + stored values from bidask 2.1.0 (e.g. first 1000 rows of `ohlc.csv`: 0.008861913688845176); with `importorskip("bidask")`, per-window parity at **every** index for w ∈ {3, 4, 5, 21, 63, 252}, excluding degenerate windows (n₁ ≤ 1 or n₂ ≤ 1), asserted on s² at ≤ 1e-11 abs (measured worst: 1.0e-10 on the spread at w=3, i.e. ~2e-12 on s²) |
| CS, AR | published formulas, hand cases; R package values if available † | — | stored |
| P, GK, RS, YZ, GKYZ | R `TTR::volatility(calc=...)` | GPL-2 — **reference values only**, generated once on our synthetic data, provenance JSON | stored, rtol 1e-12 |
| RV, BV, MedRV, MinRV, RQ, TPQ, RS±, RK (fixed H), TSRV, PAV (fixed θ), HARQ | R `highfrequency` (`rBPCov`, `rMedRVar`, `rMinRVar`, `rRQ`/`rTPQuar`, `rSemiCov`, `rKernelCov`, `rTSCov`, `rMRCov`, `HARmodel(type="HARQ")`) † | GPL — **reference values only** | stored |
| FHT | hand-computed with `scipy.stats.norm.ppf` | BSD | importorskip |

No GPL code is copied, and no GPL package is imported at runtime. Every
`FeatureSpec` records `source` / `license`, and `registry.audit()` enforces it.

### 10.4 Monte Carlo accuracy (`tests/test_ohlc_accuracy.py`, `slow`, seeded)
Each assertion reproduces a number in §1, with tolerances set to ≥ 3 standard
errors at the chosen replication count.
* Range: efficiency ordering GK > RS > P > 1 at M → large (M = 20 000); RS |drift
  bias| minus its discreteness bias < 1% at μ = σ, P > +30%; discreteness bias
  within 1% of the stored table and **corrected estimates unbiased within 1%**;
  RS-only bias < −18% at f = 0.25 while YZ is within 3%; YZ and GKYZ eff ≥ 7 at
  n = 21. The regimes here also run the YZ-vs-GKYZ default gate of §7.1.
* Spreads: EDGE pooled √mean(s²) within 5% of S (mid, illiquid); EDGE RMSE
  strictly decreasing in n ∈ {21, 63, 252}; EDGE RMSE < CS RMSE at n ≥ 63 in all
  regimes; the liquid regime asserts RMSE÷S > 1 for every estimator (**the test
  exists to prove the caveat is real; never "fix" it by loosening**).
* Intraday: GBM plus i.i.d. noise with known IV: RV biased up with noise; RK,
  TSRV and PAV unbiased within stated tolerance; with injected jumps,
  BV/MedRV/MinRV within tolerance of IV while RV is not; jump_z size ≈ 1 − α
  under no jumps.
* Heston-type stochastic variance (Euler, seeded), both range and intraday:
  efficiency *ordering* preserved (not the constants).
* Rough: the §1e table; uncorrected Parkinson at true H = 0.5 must read
  < 0.35 (**the trap is real**), corrected estimates within ±0.02.

### 10.5 Packaging guardrails
`tests/test_import_hygiene.py` (no scipy / numba at import);
`tests/test_dependency_drift.py` (**zero** new mandatory or optional deps);
`tests/test_wheel_guardrails.py`; `tests/test_namespace_stubs.py` (the
`rolling_vol` stubs); `mypy panelary` ratchet; ruff.

---

## 11. Benchmarks and performance budgets

Scripts in `benchmarks/ohlc_vol/` (ported from the 2026-09-29 prototypes):
`bench_daily.py` (range + spread + proxies, 5000 × 5000 synthetic OHLC),
`bench_intraday.py` (500 × 504 × 390), `range_mc.py`, `spread_mc.py`,
`rough_mc.py`, `discreteness_table.py`. Reference machine: the one in the
status line.

| Workload | Measured | Budget (fail CI at 2×) |
|---|---|---|
| P + GK + RS, w=21, 25M rows | 1.21 s | ≤ 2.5 s |
| Yang–Zhang, w=21, 25M rows | 1.50 s | ≤ 3 s |
| EDGE, w=21, 25M rows, batched 256 | 6.23 s, 3.1 GB RSS | ≤ 12 s, ≤ 6 GB |
| CS / AR, 25M rows | not measured | target ≤ 3 s |
| price_impact / PS γ / FHT, 25M rows | not measured | target ≤ 3 s / 5 s / 3 s |
| intraday core battery, 98M rows | ≈ 3.1 s (extrapolated from 0.62 s / 19.7M) | ≤ 8 s |
| intraday full battery incl. RK (H_max 30), PAV, TSRV, 98M rows | ≈ 17 s (extrapolated) | ≤ 40 s |
| `rough_hurst`, 10 lags, 25M daily rows | not measured | target ≤ 4 s |

`tests/test_ohlc_perf.py` (marked `benchmark`) runs a 500 × 2000 scale-down of
each row, with budgets scaled linearly.

---

## 12. Dependencies

None added. `numpy` + `polars` only at runtime. Test-only, all behind
`importorskip`: `bidask` (MIT), `scipy` (BSD). R `TTR` / `highfrequency` are
used offline once to generate stored values and are never installed in CI. The
`fast` extra stays unused by this module.

---

## 13. Milestones

| M | Contents | Ships |
|---|---|---|
| **M0** (gate) | Named caller plus an engine-side test exist (§14), or a written owner waiver. | Permission to start. |
| **M1** | `_internal/_ohlc.py` (terms, invalid policy, batching); `range_volatility` (6 methods, annualisation, discreteness table), `ohlc_variance_terms`; `.panel.rolling_vol` + `rs_vol` `FutureWarning` alias + docs / MIGRATING / CHANGELOG; FeatureSpecs; conformance adapters; hand, oracle (TTR stored) and MC range tests; YZ-vs-GKYZ default gate; `bench_daily` range rows. | Correct, fast range volatility; the misnomer fixed without changing any existing number. |
| **M2** | `ohlc_spread` (EDGE staged + batched, CS, AR, negative policies, moment columns); vendored bidask fixtures; EDGE parity at every index; spread MC. | The most accurate OHLC spread estimator at panel scale. |
| **M3** | `price_impact`, `pastor_stambaugh_gamma`, `zero_return_share`, `fht_spread` (anchored moments, FHT lookup). | Low-frequency liquidity block complete. |
| **M4** | `_realized.py` (RS±, SJV, RQ, TPQ, MedRV, MinRV, `rv_ss`, jump z, C/J); `daily_realized_measures` refactor (bitwise golden); `HARModel` HARQ / SHAR / HAR-CJ / insanity filter. | Intraday → daily battery and HAR family. |
| **M5** | Realized kernel (Parzen, day-t bandwidth), TSRV, PAV; `_internal/_realized_kernel.py`; noise MC; highfrequency stored parity. | Noise-robust univariate IV. |
| **M6** | `rough_hurst` (noise-corrected variogram), `RFSVForecaster`; rough MC. | Roughness feature + RFSV forecast. |

M1 alone is shippable. Do not start M5 before the M4 MC suite is green, or M6
before `rv_ss` / `rq` exist (M4). Each milestone after M2 needs its own named
caller (AGENTS.md).

---

## 14. Caller

> **AGENTS.md: "No new public surface without a named caller in the engine,
> and a test in the engine that exercises it."**

**Today there is none.** `truepoint/src/truepoint/quant/` (the caller directory
AGENTS.md names) does not exist, and nothing in `truepoint/src` computes
volatility or spreads. The **proposed** caller:

* `truepoint/src/truepoint/generate/family_computation.py`: a
  computation-fidelity template kind whose gold answer is a trailing
  range-volatility (M1) or OHLC-spread (M2) estimate over recorded OHLC, with
  the estimator, window, annualisation and invalid-bar policy stated. "Which
  volatility?" is exactly the ambiguity that makes a financial agent silently
  wrong. It is graded by the existing
  `truepoint/src/truepoint/score/m02_computation.py` numeric tolerance, unchanged.
* Engine test: a `truepoint/tests/` case asserting the template's gold value
  equals `panelary.econ.features.range_volatility(...)` on a synthetic fixture.
  Synthetic data only; no sealed items or answer keys may appear in panelary.
* **Firewall**: `truepoint/src/truepoint/_firewall.py` forbids `panelary` in
  `diagnose/`. These estimators may enter only through `generate/` (answer keys)
  or a future `quant/`, never `diagnose/`.

Until that test exists, M1 does not start (M0). The `rs_vol` docs correction
(§6.4 step 2) is a documentation fix with no new surface and may land
independently.

---

## 15. Risks and open questions (resolve by benchmark, not opinion)

1. **YZ vs GKYZ default** under realistic drift (§7.1 gate).
2. **Discreteness table on real data**: M is unknown and noise inflates ranges.
   Is `discrete_bars` ever net-helpful on real TAQ-derived OHLC? Measure against
   RV_ss from 1-minute bars on the same days before documenting it as more than
   an option.
3. **Banned shortcut §8.4**: is polars' kernel actually bitwise across entity
   boundaries? If so, the whole-column path may beat `.over` — measure.
4. **EDGE degenerate windows**: we return the exact-arithmetic value; the
   reference returns a rounding-determined one. Confirm with the bidask authors
   whether nulling n_k ≤ 1 windows is preferable.
5. **PAV normalisation constants** (§7.5, derived here) must match JLMPV (2009)
   and `rMRCov` before M5 ships.
6. **c* = 3.5134** and the BNHLS recipe details (q offsets, 20-minute IV) are †,
   to be verified against BNHLS (2009).
7. **MedRV / MinRV asymptotic-variance constants** (2.96, 3.81) are †; used only
   in docs.
8. `_harrv._rolling_sum` (cumsum-difference, partial windows not rescaled) and
   `_common.rolling_beta` (naive covariance) have the numerics §2 measures. Out
   of scope: changing them would change outputs. File as separate issues.
9. Existing `roll_spread` runs on price **levels** by default (back-adjustment
   and scale dependence, §9 trap 6). Document `price_change=` with log-returns as
   the recommended input; do not change the default.
10. Registry growth: a new `econ` namespace (~9 specs). README's "56 registered"
    line predates `depend` / `evolve` / `shape` (122 specs register once those
    import) — update it in M1.

---

## 16. Registry FeatureSpecs

All are `panel_safe=True`, `leakage_safe=True`, `source="Panelary"` (clean-room;
papers cited in docstrings), `license="Apache-2.0"`, `axis="time"`,
`flavour="trailing"`, and `input_shape="frame"` / `output_shape="frame"` unless
noted.

| name | namespace | safe_scope | tier | cost_hint |
|---|---|---|---|---|
| `rolling_vol` | panel | rowwise | A | O(N) |
| `rs_vol` (deprecated alias) | panel | rowwise | D | O(N) |
| `range_volatility` | econ | rowwise | B | O(N) |
| `ohlc_spread` | econ | rowwise | B | O(N·39) |
| `price_impact` | econ | rowwise | C | O(N) |
| `pastor_stambaugh_gamma` | econ | rowwise | C | O(N) |
| `zero_return_share` | econ | rowwise | B | O(N) |
| `fht_spread` | econ | rowwise | B | O(N + w²) |
| `intraday_realized_measures` | econ | **window** (one row per session; broadcasting per intraday row is look-ahead) | B | O(N·H_max) |
| `rough_hurst` | econ | rowwise | C | O(N·L) |

`econ` is not an expression namespace; every spec above is driven through
`_FRAME_OPS` adapters (§10.1).

---

## 17. References

Range volatility: Parkinson (1980) *J. Business* 53(1); Garman & Klass (1980)
*J. Business* 53(1); Rogers & Satchell (1991) *Ann. Appl. Probab.* 1(4); Rogers,
Satchell & Yoon (1994) *Appl. Financial Econ.* 4(3) †; Yang & Zhang (2000)
*J. Business* 73(3) (verified: "≈8×, up to 14×" efficiency, α ≈ 1.34); Meilijson
(2011) *REVSTAT* 9(3):199–212 (verified: 7.7322, CR bound 8.471); Broadie,
Glasserman & Kou (1997) *Math. Finance* 7(4) (discrete-monitoring correction);
Christensen & Podolskij (2007) *J. Econometrics* 141(2) †; Martens & van Dijk
(2007) *J. Econometrics* 138 †.

Spreads and liquidity: Roll (1984) *JF*; Corwin & Schultz (2012) *JF* 67(2); Abdi
& Ranaldo (2017) *RFS* 30(12); Ardia, Guidotti & Kroencke (2024) *JFE* 161,
103916, doi:10.1016/j.jfineco.2024.103916, reference code github.com/eguidotti/bidask
(MIT, v2.1.0, read 2026-09-29); Kyle (1985) *Econometrica*; Hasbrouck (2009) *JF*
64(3); Pástor & Stambaugh (2003) *JPE* 111(3); Lesmond, Ogden & Trzcinka (1999)
*RFS* 12(5); Goyenko, Holden & Trzcinka (2009) *JFE* 92(2); Fong, Holden &
Trzcinka (2017) *Rev. Finance* 21(4); Amihud (2002) *JFM*.

Realized measures: Andersen & Bollerslev (1998); Barndorff-Nielsen & Shephard
(2004, 2006) (bipower, jump test); Huang & Tauchen (2005) *J. Fin. Econometrics*
3(4); Andersen, Bollerslev & Diebold (2007) *REStat* 89(4) (HAR-CJ);
Barndorff-Nielsen, Kinnebrock & Shephard (2010), "Measuring downside risk —
realised semivariance", in *Volatility and Time Series Econometrics* (OUP);
Patton & Sheppard (2015) *REStat* 97(3); Andersen, Dobrev & Schaumburg (2012)
*J. Econometrics* 169(1) (MedRV / MinRV; variance constants †); Bollerslev,
Patton & Quaedvlieg (2016) *J. Econometrics* 192(1) (HARQ); Corsi (2009); Liu,
Patton & Sheppard (2015) *J. Econometrics* 187(1) †; Hansen & Lunde (2005) *J.
Fin. Econometrics* 3(4) †.

Noise-robust: Zhang, Mykland & Aït-Sahalia (2005) *JASA* 100; Barndorff-Nielsen,
Hansen, Lunde & Shephard (2008) *Econometrica* 76(6); (2009) *Econometrics J.*
12(3) (bandwidth recipe, c* †); (2011) *J. Econometrics* 162(2) (multivariate;
sibling 3); Jacod, Li, Mykland, Podolskij & Vetter (2009) *Stoch. Proc. Appl.*
119(7) (constants †); Christensen, Kinnebrock & Podolskij (2010) *J.
Econometrics* 159(1) †; Hayashi & Yoshida (2005) *Bernoulli* 11(2) (sibling 3);
Bollerslev, Li, Patton & Quaedvlieg (2020) *Econometrica* 88(4) (realized
semicovariances; sibling 3).

Rough volatility: Gatheral, Jaisson & Rosenbaum (2018) *Quant. Finance* 18(6);
Bennedsen, Lunde & Pakkanen (2022) *J. Fin. Econometrics* 20(5) †; Fukasawa,
Takabatake & Westphal (2022) *Math. Finance* 32(4) †; Cont & Das (2024)
*Sankhya B* 86 †.

† = not re-verified against the primary source in this session; the
implementer confirms each before the milestone that uses it ships.
