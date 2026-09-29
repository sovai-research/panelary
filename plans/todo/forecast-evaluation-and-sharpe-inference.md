# `panelary/validation/` — forecast evaluation & Sharpe inference: build contract

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.**

Extends the honest-validation layer (`panelary/validation/`) with the statistics that
decide whether a performance claim survives contact with serial dependence, fat tails,
nested models, instability, and a search over many candidates. Pure **numpy + polars**
on every path. `numba` (the existing `fast` extra) may only accelerate two kernels (PAV
bands and the variogram score). Oracles such as scipy, sklearn, statsmodels, `arch`, and
published R values are **test-only**, loaded behind `pytest.importorskip` or frozen as
fixtures.

Sibling plans written in parallel, for boundaries: drift-monitoring-and-sequential-inference (2),
covariance-and-market-state (3), label-weights-and-event-sampling (4),
ohlc-volatility-and-liquidity (5), network-and-spatial-panel (6),
causal-state-space-and-regimes (7), multiscale-complexity-features (8),
tail-risk-and-self-excitation (9), panel-causal-inference (10).

---

## 1. Why this exists

### 1.1 The naive tests are wrong, and in known directions

These were measured on this machine this session with seeded Monte Carlo
(`default_rng(20260929)`, 20,000 reps unless stated). They are measurements, not claims
from the literature:

| Test (what practitioners run today) | Setting | Size at nominal | Consequence |
|---|---|---|---|
| Jobson–Korkie with Memmel correction (JKM), Sharpe difference | T=120, ρ=0.5, equal Sharpe, i.i.d. normal | **0.053** at 5 % | fine only here |
| same | i.i.d. bivariate t₅ | **0.085** at 5 % | 1.7× over-rejection: fat tails create false "skill" |
| same | AR(1) φ=0.2 (typical monthly hedge-fund autocorrelation) | **0.089** at 5 % | 1.8× over-rejection |
| Kupiec POF, asymptotic χ²₁ | T=250, VaR 1 % | **0.097** at 5 % | 2× over-rejection: good VaR models get flagged |
| Kupiec POF, **exact binomial** | same | **0.042** | valid (conservative, because the statistic is discrete) |
| Christoffersen LR_ind, asymptotic χ²₁ | T=250 / 500 / 1000, VaR 1 % | **0.013 / 0.013 / 0.018** | badly undersized, so almost no power against clustered exceptions |
| Diebold–Mariano (HLN), nested models under the null | rolling OLS R=120, P=240, one-sided 10 % | **0.003** | nested DM has almost no power, so real predictability gets reported as "none" |
| Clark–West, same design | same | **0.055** | usable (slightly undersized, as the paper reports) |

Ledoit & Wolf (2008), Table 1 and §4.3 (read this session), reach the same verdict. JKM
"is not robust against fat tails or time series effects, where it becomes liberal". HAC
inference "is often liberal in finite samples". Only the studentized circular-block
bootstrap ("Boot-TS") "works well both for i.i.d. and time series data".

### 1.2 The existing resampling idiom is two orders of magnitude slower than it needs to be

Every bootstrap in `validation/` and `evolve/` computes replicate statistics with a
per-replicate Python loop over fancy-indexed copies (`np.stack([d[idx[b]].mean(0) for b
in ...])`). Measured this session (Apple Silicon, Python 3.13.12, NumPy 2.5.3 +
Accelerate, polars 1.44.2):

| Operation | Current idiom | Proposed | Speed-up | Agreement |
|---|---|---|---|---|
| B=1000 bootstrap means of a (T=5000, M=1000) matrix | gather loop **2.98 s** | count matrix `C @ X` (9 ms build + **24 ms** gemm) | **~90×** | max \|Δ\| 4.8e-16 |
| Romano–Wolf stepdown, S=1000, B=1000 | O(B S²) loop **0.50 s** | suffix-max, O(B S) **6 ms** | **80×** | **bitwise identical** |
| Studentized circular-block bootstrap SE, the LW (2008) Götze–Künsch form | literal gather: 10 ms at B=200, T=600, M=3 | start-count gemms: **0.7 ms** | 14× (grows with M) | 9.1e-16 relative (s*), 2.3e-13 (Δ*) |
| 5 gemms of (B=1000 × T=5000) @ (T × M=1000) | — | **0.12 s** | — | — |
| `superior_predictive_ability`, T=5000, M=200, B=1000 | **1.17 s** | (optional retrofit, M7) | — | — |

The count-matrix and start-count formulations are what make "M = 1,000 strategies ×
T = 5,000 dates with a shared studentized bootstrap in about a second" true on a laptop.

### 1.3 The capabilities are missing, and one caller already needs them

- `docs/roadmap.md` §Validation lists "stationary-bootstrap Sharpe CIs / difference
  tests" as unbuilt.
- Nothing in the package backtests a VaR or ES forecast. `econ/features/_evt.py`
  produces them (`pot_var_es`), and sibling 9 will produce more.
- There is no test for nested models, conditional or unstable predictive ability,
  forecast rationality, or directional accuracy.
- There is no multivariate proper score, and no reliability diagram beyond
  `pit_histogram`.
- There is no FDR procedure that estimates the null proportion (π₀), which is the
  luck-versus-skill question across many agents or strategies.
- **truepoint's M07 calibration metric (`truepoint/src/truepoint/inference/calibration.py::ece`)
  is binned ECE.** Dimitriadis, Gneiting & Jordan (2021) show that binned reliability is
  unstable: bin count and bin edges change the verdict. CORP (PAV-based) is the stable
  replacement, and it comes with an exact MCB/DSC/UNC score decomposition.

---

## 2. What already exists — build on it, never duplicate it (verified by reading the code)

| Location | What is there | How this plan uses it |
|---|---|---|
| `validation/_forecast_tests.py::newey_west_variance` | scalar Bartlett LRV (lag loop) | kept. The new column-vectorised LRV in `_hac.py` is parity-tested against it at 1e-12 |
| `…::diebold_mariano` | HLN-corrected DM, t_{T−1}, scalar | kept. HLN encompassing reuses its math via a vectorised private `_dm_columns`, parity-tested at 1e-12. `diebold_mariano` is re-pointed to `_dm_columns` only if `tests/test_validation_forecast_tests.py` passes unchanged |
| `…::superior_predictive_ability`, `model_confidence_set` | Hansen SPA and HLN MCS on the stationary bootstrap (gather loops) | untouched except for an optional retrofit onto the count-matrix engine (M7) |
| `…::crps_ensemble` | sorted-sample identity, O(m log m) | the d=1 case of `energy_score`, and the per-projection kernel of the sliced energy score |
| `…::crps_gaussian`, `pinball_loss(_expr)`, `interval_score`, `pit_values`, `pit_histogram`, `score_quantile_forecasts` | univariate proper scores | not duplicated. The Murphy diagram's quantile elementary scores integrate to `pinball_loss`, and that identity is a test |
| `…::_betainc`, `_t_sf` (private) | copies of `econ/_common` functions | **do not add a third copy.** New code imports `econ._common.t_sf` / `chi2_sf` |
| `validation/_selection_stats.py::_sharpe_estimator_std` | the Mertens/Opdyke i.i.d. non-normal Sharpe SE, √((1 − γ₃SR + (γ₄−1)/4·SR²)/(T−1)) | is exactly `sharpe_ratio_inference(method="iid", small_sample="t-1")`. Parity at 1e-14 |
| `…::probabilistic_sharpe_ratio`, `expected_maximum_sharpe`, `minimum_track_record_length` | PSR, SR₀, MinTRL (i.i.d. moments) | unchanged. HAC/bootstrap SEs feed DSR through its existing `sharpe_std=` argument |
| `…::holm_bonferroni`, `benjamini_hochberg`, `benjamini_yekutieli` (`_bh_family` with `c_m`) | FWER/FDR | `benjamini_hochberg` gains one keyword, `pi0: float \| None = None` (Storey q-values, adaptive BH). Default output is bitwise unchanged |
| `…::romano_wolf`, `romano_wolf_mean_test` | max-T stepdown (O(B S²) loop), stationary-bootstrap mean test | many-vs-benchmark Sharpe tests pass their (B, M) studentized null to `romano_wolf`. Stepdown made O(B S) by suffix-max (bitwise identical, measured) |
| `validation/_bootstrap.py::block_bootstrap_indices`, `resolve_segments` | seeded (B, T) indices with fold `boundaries` (per-replicate Python loop: 0.66 s at B=1000, T=5000, L=10) | the source of indices for the stationary scheme. The circular scheme gets a vectorised start-draw in `_resample.py` with the same segment semantics |
| `core/model_selection.py::_sharpe`, `deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `_norm_cdf`, `_norm_ppf` | Sharpe (ddof=1, NaN on degenerate input), DSR, PBO | `_sharpe`'s NaN convention is adopted. DSR and PBO are untouched |
| `evolve/_honest.py::haircut_sharpe_ratio`, `_hlz_params`, `_two_sided_p`, `_p_to_t` | Harvey–Liu (2015) haircut on the HLZ structural family (Bonferroni/Holm/BHY) | **extended, not duplicated**: new `significance_hurdle` sits beside it and reuses `_hlz_params` and the same family draw |
| `evolve/_honest.py::cross_sectional_bootstrap` | Yan–Zheng joint-date bootstrap of demeaned t-stats. This is the Fama–French (2010) scheme on raw means, looped per replicate with `nanquantile` | **the FF2010 path of `alpha_bootstrap` generalises it** (factor alphas, unbalanced panels, count-matrix engine). It becomes a thin delegate only after a golden test shows identical p-values and ≤1e-12 statistics |
| `evolve/_honest.py::TrialLedger`, `minimum_backtest_length`, `search_diagnostics` | trial accounting | consumers: `TrialLedger.deflated_sharpe` can take the HAC SE. No change here |
| `econ/_common.py::t_sf`, `chi2_sf`, `f_sf`, `norm_cdf`, `norm_ppf` (vectorised, Halley-refined), `ols`, `pinv_sym`, `newey_west_lrv` (Bartlett k×k sum), `auto_bandwidth` | distributions, HAC | reused as-is. Import is safe because `econ` imports nothing from `validation` (verified) |
| `econ/features/_evt.py::pot_var_es`, `gpd_fit`, `hill_index` | VaR/ES **producers** | this plan **consumes** VaR/ES forecasts. It never estimates them |
| `depend/_null.py::auto_block_length` (Politis–White with the PPW correction), `pvalue` | block length, the (1+#)/(1+B) rule | block length is reused through a **function-local import**, because `depend` imports `validation` and a module-level import would create a cycle. The p-value rule is the same (1+#)/(1+B) |
| `depend/_special.py::norm_sf` | erfc-based normal tail (frompyfunc; measured 44 ns/element) | proposed promotion into `_internal/_numpy_stats.py`, with `depend` re-exporting it. Fallback: function-local import |
| `embed/_diagnostics.py::baseline_report` | `r2_oos` against a **zero** forecast (Gu–Kelly–Xiu convention) | `oos_r2(benchmark="zero")` reproduces it. The default is Campbell–Thompson's expanding historical mean |
| `detect/_critvals.py` (`default_table`, `verify_default_table`, `_encode_default_blob`) | shipped Monte Carlo critical-value tables plus a verification harness | **pattern reused** (not code) for the Giacomini–Rossi tables |
| `feature_extractors/_kernels.py::_get_cusum_numba` | lazy `have("numba")` → `@njit(cache=True, error_model="numpy")` plus a numba-equals-python test | pattern for the two numba kernels |
| `panelary/testing.py::assert_prefix_invariant`, `assert_no_lookahead` | leak-safety assertions | run on every time-indexed output (§6) |

---

## 3. Scope and non-goals

**In scope** (one milestone each, §10):

1. **Sharpe inference**:
   - single-Sharpe SE and CI: i.i.d.-normal (Lo), i.i.d. non-normal (Mertens/Opdyke),
     HAC (Lo non-i.i.d., and Ledoit–Wolf's prewhitened QS), exact-normal (noncentral t),
     and the studentized circular-block bootstrap;
   - Sharpe-**difference** tests (pairwise and M-vs-benchmark with shared indices and
     Romano–Wolf), plus the log-variance ratio (Ledoit–Wolf 2011);
   - JKM kept only as the labelled naive baseline;
   - Lo's autocorrelation-correct annualisation.
2. **Forecast comparison beyond DM**: Clark–West; Campbell–Thompson R²_OS (with the CW
   p-value); Mincer–Zarnowitz; Pesaran–Timmermann; HLN encompassing; Giacomini–White;
   the Giacomini–Rossi fluctuation and one-time-reversal tests; a panel `loss_panel`
   adaptor.
3. **VaR/ES backtests**: Kupiec (exact binomial), Christoffersen (asymptotic plus exact
   Monte Carlo), Engle–Manganelli DQ, Acerbi–Székely Z1/Z2 (Z3 as a stretch goal), the
   FZ0 joint loss, and QLIKE. All are framed as statistical evidence.
4. **Luck vs skill**: Storey π₀; the Barras–Scaillet–Wermers FDR decomposition; the
   Fama–French (2010) / Kosowski et al. (2006) cross-sectional alpha bootstrap;
   Harvey–Liu–Zhu hurdles.
5. **Calibration and multivariate scores**: CORP reliability (PAV) with MCB/DSC/UNC and
   consistency bands; Murphy diagrams (exact, O(n log n)); energy score (exact Gram,
   plus a sliced approximation); variogram score.

**Non-goals** (each is a decision, not an omission):

- **Anytime-valid or sequential monitoring** of Sharpe or loss (confidence sequences,
  e-values, CUSUM on loss differentials) belongs to **sibling 2**. The Giacomini–Rossi
  tests here are fixed-sample and retrospective. Their `mode="monitor"` path is only a
  prefix-invariant diagnostic, and it makes no anytime-validity claim.
- **Producing** VaR, ES, or volatility forecasts belongs to **sibling 9**
  (GARCH/Hawkes/POT) and `econ/features/_evt.py`. Realized-variance proxies for QLIKE
  belong to **sibling 5**. Drawdown statistics belong to sibling 9.
- **Asset covariance** estimation, including Ledoit–Wolf *shrinkage*, belongs to
  **sibling 3**. This is a name collision: here "Ledoit–Wolf" always means the 2008
  Sharpe test. `_hac.py` is a long-run variance of *score* series, not an asset
  covariance.
- **Recalibrating forecasts for use** (isotonic or conformal recalibration fitted on past
  data) is out of scope. It belongs to `conformal/` or a future calibration transformer.
  CORP here is an in-sample *diagnostic* (see trap T4).
- **Excluded tests**:
  - a Basel "traffic light" output (regulatory framing; exact tail probabilities are
    reported instead);
  - McCracken MSE-F / ENC-NEW (non-standard tables that depend on the scheme and k₂);
  - the Bayer–Dimitriadis ESR regression backtest (needs non-smooth joint quantile/ES
    M-estimation; candidate follow-up †);
  - Du–Escanciano cumulative violations and the Christoffersen–Pelletier duration tests
    (Weibull MLE);
  - Rossi–Inoue window-robust tests;
  - West (1996) parameter-uncertainty corrections (they need model internals);
  - rolling-Sharpe `.ts` features (not asked for; they would need a `FeatureSpec`).
- **No change to** DSR, PBO, PSR, CPCV, or the existing bootstrap RNG streams.

---

## 4. Module placement and public API

### 4.1 Files

| File | Contents | Milestone |
|---|---|---|
| `validation/_results.py` (new) | `EvaluationResult`, `EVALUATION_SCHEMA` | M1 |
| `validation/_hac.py` (new, private) | column LRV (Bartlett, QS, prewhitening, Andrews bandwidth, FFT autocovariances); 4-vector prewhitened-QS for LW parity | M1 |
| `validation/_resample.py` (new, private) | count matrix, circular start-count draws with segments, block sums, batched moment helpers | M1 |
| `validation/_sharpe.py` (new) | `sharpe_ratio_inference`, `sharpe_ratio_test`, `sharpe_block_length`, `annualize_sharpe` | M1 |
| `validation/_calibration.py` (new) | `corp_reliability`, `CORPResult`, `murphy_diagram`, private `_pav` (plus numba kernel) | M2 |
| `validation/_forecast_compare.py` (new) | `clark_west`, `oos_r2`, `mincer_zarnowitz`, `pesaran_timmermann`, `encompassing_test`, `giacomini_white`, `fluctuation_test`, `one_time_reversal_test`, `loss_panel` | M3 |
| `validation/_gr_tables.py` (new) | shipped Giacomini–Rossi critical values plus `verify_gr_tables()` | M3 |
| `validation/_risk_backtest.py` (new) | `exceedances`, `kupiec_test`, `christoffersen_test`, `dynamic_quantile_test`, `acerbi_szekely_test`, `PredictiveSpec`, `fz0_loss`, `qlike_loss`, `var_backtest` | M4 |
| `validation/_luck_skill.py` (new) | `storey_pi0`, `luck_versus_skill`, `alpha_bootstrap` | M5 |
| `validation/_selection_stats.py` (edit) | `benjamini_hochberg(pi0=)`; suffix-max `romano_wolf` | M1 / M5 |
| `evolve/_honest.py` (edit) | `significance_hurdle` | M5 |
| `validation/_multivariate_scores.py` (new) | `energy_score`, `variogram_score` | M6 |
| `validation/__init__.py`, docs, CHANGELOG | exports (orchestrator only) | each |

`validation` is imported eagerly by `import panelary` (`panelary/__init__.py:233`). New
modules therefore do nothing heavy at import time: no table decoding, and no numba or
scipy imports at module level. `tests/test_import_hygiene.py` (300 ms budget,
`FORBIDDEN_MODULES`) is the guard.

### 4.2 One result type, one fixed schema

Every test returns the same frozen dataclass, vectorised over M models. Results from
different tests then `pl.concat` into one evidence table, which is what a report
builder needs.

```python
@dataclass(frozen=True)
class EvaluationResult:
    test: str                          # "sharpe_difference", "clark_west", "kupiec", ...
    names: tuple[str, ...]             # (M,) model / strategy labels
    estimate: np.ndarray               # (M,) ΔSR, R²_OS, hit rate, Z2, ...
    statistic: np.ndarray              # (M,)
    pvalue: np.ndarray                 # (M,)
    reference: str                     # "t(T-1)", "N(0,1) one-sided", "chi2(2)", "binomial-exact",
                                       # "mc-exact(B=9999)", "cbb-studentized(b=6,B=4999)", "gr-table(mu=0.3)"
    alternative: str
    n_obs: np.ndarray                  # (M,) int64, pairwise-complete count actually used
    std_error: np.ndarray | None = None
    ci: np.ndarray | None = None       # (M, 2)
    pvalue_adj: np.ndarray | None = None
    adjustment: str | None = None      # "romano-wolf", "holm", "bh(pi0=0.81)", ...
    horizon: int = 1
    hac: str | None = None             # "bartlett(L=4)", "qs-pw(S=3.17)"
    block_length: int | None = None
    n_resamples: int | None = None
    seed: int | None = None
    details: Mapping[str, np.ndarray] = field(default_factory=dict)   # test-specific extras
    warnings: tuple[str, ...] = ()
    def to_frame(self) -> pl.DataFrame: ...        # EVALUATION_SCHEMA, one row per model
    def to_dict(self) -> dict[str, Any]: ...       # JSON-safe; the truepoint Evidence payload
    def __getitem__(self, name: str) -> "EvaluationResult": ...
```

`EVALUATION_SCHEMA` = `test, model, estimate, statistic, pvalue, pvalue_adj, adjustment,
reference, alternative, std_error, ci_low, ci_high, n_obs, horizon, hac, block_length,
n_resamples, seed, warnings`. Irrelevant fields are null. No function returns a bare
p-value: `reference` always names the null distribution actually used.

### 4.3 Signatures

Array arguments accept `(T,)` or `(T, M)` float arrays (upcast to float64). Every
function takes `names=`. Panels enter through `loss_panel` (§5.3).

```python
# --- M1: Sharpe inference (_sharpe.py) ------------------------------------------
def sharpe_ratio_inference(returns, *, method="hac", benchmark_sharpe=0.0,
        alternative="greater", confidence=0.95, horizon=1,
        hac="qs_pw", hac_lags=None, small_sample="asymptotic",
        block_length="auto", n_boot=4999, boundaries=None, names=None, seed=0
) -> EvaluationResult
#   method ∈ {"iid_normal" (Lo/JK), "iid" (Mertens/Opdyke), "hac" (Lo non-iid / LW),
#             "exact_normal" (noncentral t; M ≤ 50), "bootstrap" (studentized CBB)}

def sharpe_ratio_test(returns, benchmark, *, statistic="sharpe", method="bootstrap",
        alternative="two-sided", hac="qs_pw", block_length="auto", n_boot=4999,
        calibration_sims=1000, multiple="romano_wolf", alpha=0.05, boundaries=None,
        names=None, seed=0
) -> EvaluationResult
#   returns (T,) or (T, M) vs benchmark (T,); statistic ∈ {"sharpe", "log_variance"}
#   method ∈ {"bootstrap" (LW Boot-TS), "hac" (LW HAC), "iid" (Opdyke), "jkm" (baseline;
#   always attaches a warning naming the measured over-rejection)}
#   block_length ∈ {int, "auto" (Politis–White, max over strategies), "calibrate" (LW Alg. 3.1)}

def sharpe_block_length(returns, benchmark, *, grid=(1, 2, 4, 6, 8, 10), n_sims=1000,
        level=0.95, var_bootstrap_block=5, seed=0) -> BlockLengthCalibration  # g_hat(b) per b

def annualize_sharpe(sharpe, periods, *, autocorrelations=None) -> float | np.ndarray
#   Lo (2002): eta(q) = q / sqrt(q + 2 sum_{k<q} (q-k) rho_k); sqrt(q) when rho is None

# --- M2: calibration (_calibration.py) -----------------------------------------
def corp_reliability(forecast, outcome, *, functional="probability", score="brier",
        band="auto", n_boot=500, level=0.90, seed=0) -> CORPResult
#   functional ∈ {"probability" (binary y), "mean" (Gneiting–Resin, squared error)}
@dataclass(frozen=True)
class CORPResult:
    x: np.ndarray; recalibrated: np.ndarray; weight: np.ndarray        # (G,) per unique forecast
    band_low: np.ndarray | None; band_high: np.ndarray | None
    mean_score: float; mcb: float; dsc: float; unc: float; n_obs: int
    score: str; functional: str; warnings: tuple[str, ...]
    def to_frame(self) -> pl.DataFrame: ...; def to_dict(self) -> dict: ...

def murphy_diagram(forecasts, y, *, functional="mean", level=0.5, thetas=None,
        names=None) -> pl.DataFrame         # columns: theta, model, mean_elementary_score
#   functional ∈ {"probability", "mean", "expectile", "quantile"}

# --- M3: forecast comparison (_forecast_compare.py) -----------------------------
def clark_west(y, benchmark_forecast, forecasts, *, horizon=1, hac_lags=None, names=None)
def oos_r2(y, forecasts, *, benchmark="historical_mean", min_periods=60, horizon=1,
        truncate_at=None, names=None)       # CW p-value when the benchmark is nested
def mincer_zarnowitz(y, forecasts, *, horizon=1, cov="hac", hac_lags=None, names=None)
def pesaran_timmermann(y, forecasts, *, horizon=1, exact=False, threshold=0.0, names=None)
def encompassing_test(y, forecast_a, forecasts_b, *, horizon=1, names=None)  # both directions in details
def giacomini_white(loss_a, loss_b, *, instruments="lagged", horizon=1,
        estimation_scheme="rolling", names=None)
def fluctuation_test(loss_a, loss_b, *, window=0.3, mode="test", horizon=1,
        hac_lags=None, alpha=0.05, names=None) -> FluctuationResult
#   FluctuationResult: path (P, M) (NaN before window), critical_value, max_stat (M,),
#   argmax_index (M,), reject (M,), mode, window, plus .to_evaluation() -> EvaluationResult
def one_time_reversal_test(loss_a, loss_b, *, trim=0.15, horizon=1, hac_lags=None, names=None)
def loss_panel(frame, *, entity, time, target, forecasts, loss="squared", quantile=None,
        aggregate="date_mean", min_entities=1) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]
#   -> (dates (T,), losses (T, M), names); every array-level test above consumes it

# --- M4: risk backtests (_risk_backtest.py) ---------------------------------------
def exceedances(y, *, lower=None, upper=None) -> np.ndarray       # (T, M) in {0, 1, NaN}
def kupiec_test(hits, *, level, pvalue="exact", alternative="two-sided", names=None)
def christoffersen_test(hits, *, level, kind="conditional_coverage", null="mc",
        n_sims=9999, seed=0, names=None)
def dynamic_quantile_test(hits, var, *, level, n_lags=4, null="asymptotic", n_sims=999,
        seed=0, names=None)
@dataclass(frozen=True)
class PredictiveSpec:                        # what the ES backtest needs to simulate H0
    family: Literal["normal", "student_t", "fhs"]
    loc: np.ndarray; scale: np.ndarray       # (T, M), known at t - h
    df: float | None = None; residuals: np.ndarray | None = None   # fhs: standardized pool
def acerbi_szekely_test(returns, var, es, *, level, kind="z2", predictive=None,
        n_sims=2000, seed=0, names=None)
def fz0_loss(returns, var, es, *, level) -> np.ndarray             # PZC (2019) eq. (6)
def qlike_loss(realized_variance, forecast_variance) -> np.ndarray # Patton (2011)
def var_backtest(returns, var, es=None, *, level, var_convention,
        tests=("kupiec", "christoffersen", "dq", "z2"), predictive=None, seed=0,
        names=None) -> pl.DataFrame                                 # concatenated EVALUATION_SCHEMA rows
#   var_convention ∈ {"loss", "quantile"} is REQUIRED (no default): sign bugs are the
#   #1 VaR-backtest error. "loss": VaR > 0, a hit is y < -VaR. "quantile": VaR < 0, a hit is y < VaR.

# --- M5: luck vs skill (_luck_skill.py, _selection_stats.py, evolve/_honest.py) ----
def storey_pi0(pvalues, *, lambdas=None, n_boot=1000, seed=0) -> Pi0Estimate
def benjamini_hochberg(pvalues, *, alpha=0.05, pi0=None) -> MultipleTestResult  # additive kwarg
def luck_versus_skill(tstats, *, gammas=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        pi0="storey", dof=None, n_boot=1000, seed=0) -> pl.DataFrame
def alpha_bootstrap(returns, factors=None, *, scheme="joint_dates", block_length=1,
        n_boot=1000, quantiles=(0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99), min_obs=24,
        seed=0, names=None) -> AlphaBootstrapResult
def significance_hurdle(n_trials, *, alpha=0.05, method="bhy", rho=0.2, n_sim=2000,
        seed=0) -> float                                            # t-stat hurdle (HLZ)

# --- M6: multivariate scores (_multivariate_scores.py) ---------------------------
def energy_score(y, samples, *, method="exact", fair=False, n_projections=128, seed=0,
        chunk_bytes=256 * 2**20) -> np.ndarray                      # y (T,d), samples (T,m,d)
def variogram_score(y, samples, *, p=0.5, weights=None, chunk_bytes=256 * 2**20) -> np.ndarray
```

**No `FeatureSpec` registrations.** Nothing here is an ML feature. The time-indexed
outputs (the fluctuation monitor path, the expanding R²_OS path, the cumulative loss
difference) are evaluation diagnostics, returned as arrays or frames. If a `.ts` op is
ever added, for example a trailing exception rate, it must register with
`safe_scope="rowwise"`, `flavour="trailing"`, `panel_safe=True`,
`leakage_safe=True`, the paper in `source`, and a permissive `license`.

---

## 5. Algorithms

Global numerics:

- float64 throughout.
- Batched Grams use `@` (never bare `einsum`).
- Batched solves are `np.linalg.solve(A, b[..., None])[..., 0]` (the NumPy 2 p == batch
  trap).
- p-values from resampling are `(1 + #{null ≥ obs}) / (1 + B)`.
- Normal tails use erfc, never `1 − Φ` (which pins to 0 at |z| ≈ 8.3, per `_honest.py`).
- Degenerate inputs (σ = 0, singular Ω, no exceptions) yield NaN plus a `warnings`
  entry, never 0.

### 5.1 Shared engine (M1)

**A1 — count-matrix moments.** For an index matrix `idx (B, T)`:

- build `C = bincount(b·T + idx).reshape(B, T)` as float64. The counts are small
  integers, so they are exact in float64;
- then every replicate mean of every column is `C @ X / T`.

This is one BLAS gemm, O(BTM) flops with no Python loop and no (B, T, M) temporary.
Measured: 24 ms against 2.98 s, max |Δ| 4.8e-16. Chunk over B whenever `B·T·8 > 256 MB`.
Chunking must not change results: all indices are drawn first, in one call.

Rejected: the gather loop (≈90× slower); `np.add.at` (≈10× slower than `bincount`).

**A2 — start-count circular-block bootstrap (CBB).** This is LW (2008) §3.2.2 exactly,
reformulated.

1. Draw `starts (B, l)`, with `l = ⌊T/b⌋`, from one `default_rng(seed).integers` call.
   The bootstrap sample is l·b observations, as in LW.
2. Build `N = bincount(b·T + starts) (B, T)`.
3. Compute circular block sums of any series x as
   `Q_x = sliding_window_view(concat(x, x[:b-1]), b).sum(-1)`, shape (T, ·). This is
   pairwise summation inside each window, O(Tb).
4. Every bootstrap first and second **block** moment is then a gemm, `N @ (Q_a ∘ Q_c)`.

Götze–Künsch studentization:

- Ψ* = (1/l) Σⱼ ζⱼζⱼ′, with ζⱼ = b^{−1/2} Σ_{t∈block j} y*_t.
- It enters only through w′Ψ*w = (1/(lb)) [w′ S₂ w − l b² (w′v̄*)²], where
  S₂ = Σⱼ Q_{sⱼ}Q_{sⱼ}′.
- This is algebraically exact, so no bootstrap sample is ever materialised.

Numerics, to kill the cancellation in that difference:

- Work in **centred coordinates**: c = r − μ̂ and c², with μ̂ the full-sample column
  mean.
- The Sharpe gradient in (μ, γ) coordinates maps to (c, c²) coordinates as
  h = (g_μ + 2μ̂g_γ, g_γ), up to a constant that cancels.
- Compute σ*² = m₂* − m₁*² in centred moments.

Measured prototype (this session): 9.1e-16 relative on s(Δ*) and 2.3e-13 on Δ* against
the literal LW formulas.

Cost for M strategies against one benchmark: 3 strategy-only product gemms plus 4 cross
gemms plus 2 first-moment gemms, each (B × T) @ (T × M), plus O(T) benchmark gemvs. That
is ≈ 9 gemms ≈ **0.22 s** at B=1000, T=5000, M=1000 (5 measured at 0.12 s). Memory:

- N: 40 MB;
- one (T, M) product temporary: 40 MB;
- ≈ 12 (B, M) outputs: 96 MB.

Peak is under 300 MB.

`boundaries` (fold segments):

- each segment s gets `l_s = ⌊T_s/b⌋` blocks, started and wrapped **inside** the segment;
- block sums use segment-local circular padding;
- this mirrors the guarantee of `block_bootstrap_indices` that blocks never straddle a
  fold.

Rejected alternatives:

- the stationary bootstrap for the *studentized* test: Götze–Künsch studentization needs
  fixed blocks. The stationary scheme stays the SPA/MCS/RW-mean scheme;
- the moving-block bootstrap, which has edge effects (LW fn. 5);
- per-strategy independent indices, which destroy the joint null that Romano–Wolf needs.

**A3 — column long-run variances (`_hac.py`).**

1. **Bartlett, fixed L.** A lag loop with column dot products, O(LTM). Measured 9 ms at
   L=9, T=5000, M=1000.
2. **QS kernel.** k(x) = 25/(12π²x²)·[sin(6πx/5)/(6πx/5) − cos(6πx/5)]. QS has infinite
   support, so all T lags are needed. They come from one **batched rFFT autocovariance**
   (zero-pad to 2^⌈log₂(2T−1)⌉): O(TM log T), measured 0.14 s at T=5000, M=1000, 3e-17
   against direct sums. Chunk M by 256 columns, which keeps it at 33 MB of complex
   workspace.
3. **Andrews (1991) AR(1) plug-in bandwidth, per column.**
   S_T = 1.3221·(α̂(2)·T)^{1/5}, with α̂(2) = 4ρ̂²/(1−ρ̂)⁴ for a scalar.
4. **Andrews–Monahan (1992) prewhitening.**
   - Scalar: e_t = x_t − ρ̂x_{t−1}, then LRV_x = LRV_e/(1−ρ̂)², with |ρ̂| capped at 0.97 †
     against the recolouring blow-up.
   - 4-vector (`lrv_qs_prewhitened`, for pairwise LW parity only): VAR(1) Â, then
     Ψ = (I−Â)^{−1}Ψ_e(I−Â)^{−1′}.
5. LW's small-sample factor T/(T−4) is applied to the 4-vector path. The scalar path uses
   T/(T−2), which is the analogue for 2 estimated moments.

This does not duplicate `econ._common.newey_west_lrv`, which returns one k×k Bartlett sum
matrix. The new code needs M separate scalar LRVs and the QS kernel.

### 5.2 Sharpe inference (M1)

**One engine: the influence function.**

For a Sharpe ratio SR = μ/σ:

- ψ_t = z_t − (SR/2)(z_t² − 1), with z_t = (r_t − μ)/σ;
- Var(ψ) = 1 − γ₃SR + (γ₄−1)/4·SR², which is exactly the Mertens/Opdyke formula and the
  numerator in the existing `_sharpe_estimator_std`.

Every SE in this module is the variance of an influence series, estimated one way or
another:

| method | estimator of Var(√T·SR̂) | notes |
|---|---|---|
| `iid_normal` | 1 + SR²/2 | Lo (2002) "IID"; JK/Memmel for differences |
| `iid` | sample Var(ψ) | Mertens (2002)/Opdyke (2007). Opdyke's *time-series* formulas "are incorrect, since they are equivalent to the formulae for the i.i.d. case" (LW 2008, Remark 3.1), so they are not implemented |
| `hac` | LRV(ψ), prewhitened QS by default, Bartlett(L) for Lo parity | Lo (2002) "non-IID" GMM equals the HAC of ψ (delta method is bilinear) |
| `bootstrap` | studentized CBB (A2) | LW (2008) Boot-TS |
| `exact_normal` | SR̂·√T ~ noncentral t(T−1, √T·SR) | exact under i.i.d. normal. CI by bisection on δ. CDF by Lenth (1989, AS 243) through `econ._common` incomplete beta. Scalar-cost, so restricted to M ≤ 50 |

For the **difference**, the influence series is ψ_i − ψ_b. For the **log-variance**
statistic (Ledoit & Wolf 2011 †) it is ψ = (r−μ)²/σ² − 1, and the same engine applies.

**Pairwise, `hac`.** LW's exact recipe for parity with their published procedure:

- the 4-vector y_t = (r_i−μ_i, r_n−μ_n, r_i²−γ_i, r_n²−γ_n);
- the parametrisation f(a,b,c,d) = a/√(c−a²) − b/√(d−b²);
- the gradient ∇′f = (c/(c−a²)^{1.5}, −d/(d−b²)^{1.5}, −½a/(c−a²)^{1.5}, ½b/(d−b²)^{1.5});
- the prewhitened-QS Ψ with T/(T−4);
- s(Δ̂) = √(∇′Ψ̂∇/T).

For M > 1 the default is the **scalar** influence path (column LRV of ψ_i − ψ_b). It is
identical to the vector path when prewhitening is off and the bandwidth is common. With
prewhitening it is a different consistent estimator. Both are tested (§7), and
`hac="qs_pw_vector"` forces the LW form for every column (an O(M) loop of 4×4 work).

**Bootstrap p-value.** LW eq. (9) for the two-sided test:

PV = (#{|Δ*−Δ̂|/s(Δ*) ≥ |Δ̂|/s(Δ̂)} + 1)/(B + 1),

with s(Δ̂) from HAC and s(Δ*) from A2. A one-sided version uses signed comparisons.

**Many against a benchmark.** One N is shared by all M strategies. Build the (B, M) matrix
z* = (Δ* − Δ̂)/s(Δ*), then:

- the marginal p-values are its column-wise counts;
- `multiple="romano_wolf"` sends (Δ̂/s(Δ̂), z*) to `romano_wolf`. With
  `alternative="greater"` this is the studentized StepM ("which strategies beat the
  benchmark");
- adding or removing a column never changes another column's marginal statistic or
  p-value (N does not depend on M). This is tested.

Unbalanced strategies are handled this way:

- columns are grouped by identical availability mask (`np.unique` on mask columns);
- each group runs the engine on its own common-date block;
- Romano–Wolf runs only within the largest group;
- the result carries a warning naming the others.

**Block length.**

- `"auto"`: `depend._null.auto_block_length` (PPW-corrected Politis–White) on each
  column's difference-influence series. The **maximum** over columns is used, because
  shared indices need one b. `depend` measured Politis–White over-rejecting, so the
  calibration test (§7) decides whether "auto" can stay the default for M > 1.
- `"calibrate"` (the pairwise default when T ≤ 2000): LW Algorithm 3.1.
  1. Fit a VAR(1) to (r_i, r_n).
  2. Generate K pseudo-sequences from the VAR with **stationary-bootstrapped residuals**
     (mean block 5, as in LW §4.1).
  3. For each b in the grid, run the studentized CBB and record whether the CI covers
     Δ̂. Pick the b with ĝ(b) closest to 1 − α.
  4. **Vectorised**: the K pseudo-pairs are K extra columns in the same A2 gemms. One N
     per b is shared across the K pseudo-sequences; this keeps ĝ unbiased and costs only
     some variance.
  5. The grid is LW's (1, 2, 4, 6, 8, 10) at T = 120, scaled by (T/120)^{1/3} and
     rounded to unique integers †. K defaults to 1000 (LW's stated lower limit;
     5000 "will certainly suffice").
- b is recorded in the result either way.

**Exact normal.** Retained for tiny T. It must say, in the result `reference`, that it
assumes i.i.d. normal returns.

**Lo annualisation.** η(q) = q/√(q + 2Σ_{k=1}^{q−1}(q−k)ρ_k), taking the autocorrelations
as input so that no estimation hides inside. √q is used only when `autocorrelations=None`,
with a warning.

### 5.3 Forecast comparison (M3)

All tests take `(T, M)` model matrices against a `(T,)` benchmark and run a single
vectorised pass.

- **Default HAC lags = horizon − 1** (the MA order of an optimal h-step error). This is
  never a function of T; an explicit `hac_lags` overrides it.
- **`loss_panel`** is the panel adaptor.
  1. It sorts by (entity, time) and computes per-row losses in polars.
  2. It then takes the **cross-sectional mean loss per date**
     (`group_by(time).agg(mean, count)`, same-date data only, multithreaded), dropping
     dates with fewer than a fixed `min_entities`.
  3. Every test then runs on this (T, M) date series. This is the Gu–Kelly–Xiu † panel
     DM convention. It absorbs contemporaneous cross-entity dependence, which per-entity
     tests pooled naively ignore.
  4. `aggregate="none"` returns (T·N, M) for per-entity use with an FDR correction.

| Test | Algorithm (chosen) | Cost | Notes and rejected alternatives |
|---|---|---|---|
| **Clark–West (2007)** | f_t = e₀² − [e_i² − (ŷ₀−ŷ_i)²]; t = √P·f̄/√LRV(f); one-sided N(0,1) | O(TM) | The model must nest the benchmark (documented; not checkable). DM on nested models is undersized (0.003 measured) |
| **R²_OS (Campbell–Thompson 2008)** | 1 − Σe_i²/Σe₀². Default benchmark: **expanding historical mean as of t−h**, computed with polars `cum_sum` / non-null `cum_count` `.over(entity)` then `shift(h)`, with `min_periods` fixed. p-value from CW | O(n) | `truncate_at=0.0` is CT's non-negativity restriction. `benchmark="zero"` reproduces `embed.baseline_report`. Also returns the **cumulative SSE-difference path** (Goyal–Welch plot) |
| **Mincer–Zarnowitz** | y = a + β(ŷ − ȳ̂) + u (centred design, so X′X is diagonal and trivially conditioned). H₀: (α, β) = (0, 1), with α = a − βȳ̂. Wald with a column HAC sandwich (batched 2×2 closed-form inverses). χ²₂ sf = exp(−W/2) | O(LTM) | `cov="ols"` gives the exact F under Gaussian i.i.d. errors |
| **Pesaran–Timmermann (1992)** | P̂, P̂* = P_yP_x + (1−P_y)(1−P_x); S = (P̂−P̂*)/√(V̂(P̂)−V̂(P̂*)) → N(0,1) | O(TM) | `exact=True`: Fisher's exact test on the 2×2 table (hypergeometric via lgamma), valid for i.i.d. For h > 1, HAC-t on the regression of 1{y>0} on 1{ŷ>0} (PT 2009 †), because PT92 assumes independence |
| **HLN (1998) encompassing** | d_t = (e_a − e_b)e_a. HLN-corrected DM, t_{T−1}, one-sided. Both directions go in `details` | O(TM) | via `_dm_columns` |
| **Giacomini–White (2006)** | Z_t = h_t·ΔL_{t+τ}; default h_t = (1, ΔL_t); W = n·Z̄′Ω̂⁻¹Z̄ ~ χ²_q. Ω̂ is the outer-product mean (τ = 1) or Bartlett(τ−1). Batched (M, q, q) Grams by matmul and a batched solve. Also reports the decision-rule share (sign of δ̂′h_t) | O(q²TM) | Valid for **rolling** (fixed-window) estimation only. `estimation_scheme="expanding"` attaches a warning (trap T7) |
| **GR fluctuation (2010)** | F_t = m^{−1/2}Σ_{j=t−m+1}^{t}ΔL_j / σ̂, from `cumsum` differences, O(1) per t. **Labelled at the window end.** GR label at the centre; it is the same set of windows, so the sup is identical. Reject if max\|F\| > k_α(μ). `mode="test"`: σ̂ from the full-sample HAC (GR). `mode="monitor"`: σ̂_t from an **expanding** Bartlett HAC with fixed L (cumsum expansions of Σx_sx_{s−j}, O(PL)), so the path is prefix-invariant | O(PM) | k_α(μ) comes from a shipped table: sup_{r∈[μ,1]} \|B(r)−B(r−μ)\|/√μ simulated offline (2·10⁵ reps, 2·10⁴ steps, μ ∈ {0.1…0.9}, α ∈ {0.01, 0.05, 0.10}). It must match GR Table 1 † within MC error. `window` must be one of the tabulated μ in `mode="test"` |
| **GR one-time reversal** | Φ = sup_{τ∈[τ₀P,(1−τ₀)P]}[LM₁ + LM₂(τ)], with LM₁ = Pσ̂⁻²ΔL̄² and LM₂(τ) = σ̂⁻²P⁻¹(τ/P)⁻¹(1−τ/P)⁻¹[S_τ − (τ/P)S_P]², from partial sums | O(PM) | The limit is W(1)² + sup_r BB(r)²/(r(1−r)), and the two parts are independent. It is simulated into the same shipped table for trim ∈ {0.15, 0.20}. Must match GR's published values † |

The shipped tables follow the `detect/_critvals.py` pattern: an encoded constant blob, a
fingerprint, and `verify_gr_tables(quick=True)` regenerating a coarse table, marked
`slow`. The offline generator lives in `benchmarks/gr_critical_values.py`. Decoding
happens lazily on first use.

### 5.4 VaR / ES backtests (M4)

The framing is written into every docstring: *"statistical evidence about the
calibration of a risk forecast; not a regulatory determination."* No pass/fail zone is
emitted.

- **Hits.** `exceedances` returns {0, 1, NaN} (T, M). The same function serves conformal
  intervals (hit = y outside [lower, upper]), because Christoffersen (1998) is an
  interval-forecast test.
- **Kupiec.**
  - Exact binomial two-sided p-value by the "minlike" rule: Σ pmf(k) over k with
    pmf(k) ≤ pmf(x)·(1+1e-7), which is scipy's `binomtest` rule. The pmf comes from
    `math.lgamma` over k = 0…T in log space, O(T), computed once per distinct
    (T_eff, α) pair and shared across models.
  - One-sided "too many exceptions" = I_α(x, T−x+1) through `econ._common`'s incomplete
    beta.
  - The asymptotic LR is reported in `details`, because the measured χ² size at T=250 is
    0.097.
- **Christoffersen.**
  - Transition counts come from shifted boolean products over NaN-complete pairs.
  - The likelihood ratios use `xlogy` (0·log 0 = 0).
  - `null="mc"` is the default, the Dufour (2006) exact Monte Carlo test. H₀ (i.i.d.
    Bernoulli(α)) is fully specified, so B = 9999 paths of length T_eff are simulated
    **once per distinct (T_eff, α) and shared by all M models**. Per-model p-values then
    come from `searchsorted` on the sorted null. Measured 43 ms for B = 9999, T = 1000.
  - Ties in this discrete statistic count as "≥", which is conservative and needs no
    randomised tie-breaking (that would require a seed-dependent verdict).
  - The asymptotic χ² is kept in `details`; its size was measured at 0.013–0.018.
- **DQ (Engle–Manganelli 2004).**
  - Hit_t − α regressed on X_t = [1, Hit_{t−1…t−4}, VaR_t].
  - DQ = β̂′X′Xβ̂/(α(1−α)) ~ χ²_k.
  - Batched (M, k, k) Grams and a batched solve.
  - `null="mc"` simulates hits holding VaR_t fixed: per-model and chunked,
    (B·M) solves of k×k, where 3·10⁶ 4×4 solves were measured at 0.70 s. It is opt-in.
- **Acerbi–Székely (2014).**
  - Z1 = Σ X_tI_t/(N_T·ES_t) + 1 (conditional on exceptions) and
    Z2 = Σ X_tI_t/(TαES_t) + 1, with the sign convention stated.
  - The null is simulated from `PredictiveSpec`: normal, standardized Student-t (via
    `rng.standard_t`, so no t-ppf is needed), or filtered historical simulation (resampled
    standardized residual pool × scale + loc). That is (B, T) draws per model, chunked,
    and each model's generator is `default_rng([seed, crc32(name)])`.
  - Without `predictive`, the result carries only the statistic plus the published
    "stable" Z2 5 % thresholds (−0.70 to −0.82 across t₃–t₇ †), and no p-value.
  - Z3 (needs predictive quantile functions and beta-integral expectations) is a stretch
    goal for normal and t only.
- **FZ0** (Patton, Ziegel & Chen 2019, eq. (6), verified):
  L = −1/(αe)·1{Y≤v}(v−Y) + v/e + log(−e) − 1, for v, e < 0.
  - This is the unique FZ loss whose differences are homogeneous of degree 0, so model
    comparisons are unit-free.
  - Inputs with v ≥ 0, e ≥ 0, or e > v give NaN plus a warning.
  - Comparative backtesting (Nolde–Ziegel 2017) = FZ0 losses into the existing DM/MCS.
- **QLIKE** (Patton 2011): L = RV/h − log(RV/h) − 1. It is robust to noise in the
  volatility proxy, which is why it is used rather than MSE-on-log.

**Panel (T, N) mode.** Every test is column-wise, so N entities = N columns. Per-entity
p-values go to `benjamini_yekutieli` (arbitrary dependence).

### 5.5 Luck vs skill (M5)

- **Storey π₀.**
  1. π̂₀(λ) = #{p > λ}/(M(1−λ)) on λ ∈ {0.05, …, 0.95}, all at once from one sort and
     `searchsorted`.
  2. λ* minimises bootstrap MSE against min_λ π̂₀ (Storey 2002; the choice BSW use).
     Bootstrap p-vectors come as (B, M) indices, bucketed with one `bincount` per λ
     bucket over flattened (b, bucket) ids, O(BM).
  3. π̂₀ is capped at 1.
  - The spline smoother (Storey–Tibshirani 2003) is rejected, because it needs scipy.
- **Adaptive BH / q-values.** `benjamini_hochberg(pi0=π̂₀)` scales by π̂₀ inside
  `_bh_family` (c_m = π̂₀). With `pi0=None` it is bitwise identical to today.
- **BSW (2010).**
  - Two-sided p from t-stats, via t_{dof} or N(0,1) when `dof=None`.
  - For each γ: Ŝ_γ^± (the share significant, split by sign), F̂_γ^± = π̂₀γ/2,
    T̂_γ^± = Ŝ_γ^± − F̂_γ^±, and FDR̂_γ^± = F̂_γ^±/Ŝ_γ^±.
  - π̂_A^± is read at the largest γ. Output is a γ-indexed frame.
- **`alpha_bootstrap`.**
  - `scheme="joint_dates"` (Fama–French 2010, the default):
    1. Impose H₀ by r̃_i = r_i − α̂_i.
    2. Resample **dates jointly** for all funds (stationary scheme via
       `block_bootstrap_indices`, `block_length=1` i.i.d. as in FF2010). This keeps the
       cross-fund correlation that makes extreme t-quantiles easy to reach under the
       null.
    3. Unbalanced panels use an availability mask A (T, N). With count matrix C
       (A1), every regression ingredient is a gemm:
       - Gram entries G_{b,i,kl} = C @ (A∘x_k x_l), for (K+1)(K+2)/2 gemms;
       - X′y = C @ (A∘x_k∘r̃), for K+1 gemms;
       - y′y and n_i, one gemm each.
    4. Batched (B·N) (K+1)² solves: 3·10⁶ 4×4 solves measured at 0.70 s.
    5. RSS = y′y − β′X′y, then OLS t(α).
    6. Replicates with n_i < `min_obs` are dropped, and the per-fund null sizes are
       reported.
    7. The cancellation in RSS is bounded because factors are O(1)-scale and
       near-zero-mean. A replicate whose Gram has condition number above 1e10 falls back
       to gather plus `lstsq` for that (b, i).
  - `scheme="residual"` (Kosowski et al. 2006): per-fund independent residual
    resampling with factors held fixed. The hat vector a_i = e₁′(X_i′X_i)⁻¹X_i′ is fixed
    per fund, so α*_{b,i} = C_i @ (a_i∘ê_i). This needs one count matrix per fund:
    ~1.5 s at N=3000, T=360, and O(BTN) flops.
  - Outputs:
    - per-quantile observed / null-median / null-p95 / p-value (the FF2010 table);
    - per-fund bootstrap p-values (KTWW);
    - null sizes;
    - scheme, block length and seed.
  - A studentized t(α) is used on both sides. The docs explain the scheme choice: KTWW
    under a common factor gives too narrow a null for the extreme quantiles. That is a
    pinned test (§7).
- **`significance_hurdle`** (HLZ 2016; in `evolve/_honest.py`).
  - Bonferroni has a closed form: t* = −Φ⁻¹(α/(2M)).
  - Holm and BHY reuse the haircut's HLZ family draw with common random numbers.
    1. Per simulated family, sort the other M p-values once and precompute:
       - Holm: prefix-max of (M+2−j)·q_(j);
       - BHY: suffix-min of c(M+1)·q_(j)/(j+1).
    2. The adjusted p of an inserted test with p₀ is then O(1) after the rank lookup:
       - Holm: max(prefmax[r−1], (M+2−r)·p₀), with r from a flat row-offset
         `searchsorted`;
       - BHY: min(c(M+1)·p₀/r, sufmin[r]).
    3. The median over families is monotone in p₀, so bisection on t is exact to 1e-10
       at ≈50 × O(n_sim·log M).
  - `haircut_sharpe_ratio` is untouched (its RNG stream is preserved).

### 5.6 Calibration: CORP and Murphy diagrams (M2)

- **PAV** (Ayer et al. 1955; O(n) stack algorithm, Best–Chakravarti 1990).
  1. Sort x once (stable argsort).
  2. **Pool ties in x first**: `np.unique(return_inverse)`, then `bincount` sums and
     weights, because CORP requires equal forecasts to get equal recalibrated values.
  3. Run the stack PAV over G unique values.
  - Pure Python measured 125 ms at n = 10⁶ (argsort 76 ms). The `fast` numba kernel is
    5.7 ms. scipy's `isotonic_regression` (oracle, 4 ms) agrees to 4.4e-16.
  - The same kernel serves `functional="mean"` (squared error, Gneiting–Resin †).
- **Decomposition** (exact identity):
  - S̄(x) = MCB − DSC + UNC;
  - MCB = S̄(x) − S̄(ĉ) ≥ 0;
  - DSC = S̄(r̄) − S̄(ĉ) ≥ 0;
  - UNC = S̄(r̄), with ĉ the PAV fit and r̄ = mean(y).
  - Non-negativity holds because PAV is simultaneously optimal for every Bregman loss
    (the Brier and log scores for binary y, squared error for the mean).
  - Log score: x ∈ {0, 1} with the opposite outcome gives an infinite score, reported as
    such with a warning, never clipped silently.
- **Consistency band** (DGJ 2021).
  - Resample under the hypothesis of calibration: k*_g ~ Binomial(n_g, x_g) per unique
    forecast, drawn vectorised as (B, G). Run PAV per replicate on G blocks and take
    pointwise quantiles.
  - `band="auto"` computes the band when G·B ≤ 5·10⁷ (pure) or ≤ 5·10⁹ (numba).
    Otherwise `band=None`, with a warning in the result. The band is never silently
    subsampled.
  - `band="confidence"` resamples (x, y) pairs instead.
- **Murphy diagram** (Ehm, Gneiting, Jordan & Krüger 2016). The mean elementary score is
  computed **exactly at every breakpoint**, the union of forecast and outcome values:
  - quantile: S_θ = (1{y<x}−α)(1{θ<x}−1{θ<y}) is piecewise constant, so weighted
    survival counts on sorted x and sorted y, then `searchsorted`;
  - expectile or mean: S_θ = |1{y<x}−τ|[(y−θ)₊ − (x−θ)₊ − (y−x)1{θ<x}] is piecewise
    linear, so suffix sums of weights and of weight×value;
  - probability: θ·1{y=0, x>θ} + (1−θ)·1{y=1, x≤θ} (up to the paper's constant).
  - Cost O((n + G)·log n) per model. A user θ grid is exact too, because the function is
    piecewise linear or constant.

### 5.7 Multivariate proper scores (M6)

- **Energy score** (Gneiting & Raftery 2007; Gneiting et al. 2008):
  ES = (1/m)Σ‖X_k − y‖ − (1/(2m²))Σ_{k,l}‖X_k − X_l‖. `fair=True` uses 1/(2m(m−1)) (Ferro
  2014 †).
  - `exact` uses the Gram trick: D² = s_k + s_l − 2X X′, batched matmul over T-chunks,
    clamped at ≥ 0, then sqrt. Measured 2.08 ms/obs at m = 1000, d = 50; 5.1e-12 relative
    to direct pairwise differences.
  - The cancellation for near-duplicate members is bounded by ε‖x‖²/‖x_k − x_l‖². A
    `method="direct"` path (m ≤ 256) exists for tests.
  - d = 1 dispatches to `crps_ensemble` (the sorted identity, exact).
  - `sliced`: ‖v‖ = c_d·E_θ|θ′v| with c_d = √π·Γ((d+1)/2)/Γ(d/2), so
    ES = c_d·E_θ[CRPS(θ′X ensemble, θ′y)]. The estimate averages `crps_ensemble` over K
    **orthogonalised** random directions (QR of a d×K Gaussian, blocks of d), at
    O(K·m·(d + log m)) per observation.
  - The sliced result is marked approximate in `warnings`. Its error at K = 128/1024 is a
    measured test, not a claim.
  - No exact sub-quadratic energy distance exists for d > 1, so `exact` stays the
    default for m ≤ 2000.
- **Variogram score** (Scheuerer & Hamill 2015):
  VS_p = Σ_{i<j} w_ij(|y_i−y_j|^p − (1/m)Σ_k|X_ki−X_kj|^p)², with p = 0.5 by default.
  - Computed over the upper triangle, chunked over pairs so the (m, d(d−1)/2) temporary
    stays under `chunk_bytes`.
  - Measured 1.81 ms/obs at m = 1000, d = 50.
  - The numba fused kernel (`fast`) removes the temporary entirely.
  - Documented as the complement of ES, which is known to be weak at detecting
    misspecified dependence (Pinson & Tastu 2013 †).

---

## 6. Leak-safety design and traps

**Scope of the prefix-invariance contract.**

- Whole-sample tests are statistics of an already-out-of-sample record. Their n-dependent
  constants (HAC bandwidth, block length, Storey λ, the MC null) are legitimate there,
  because they are **never** used inside a time-indexed output.
- Every time-indexed output **is** prefix-invariant, bitwise:
  - `FluctuationResult.path` in `mode="monitor"`;
  - the expanding R²_OS path;
  - the cumulative SSE-difference path;
  - the expanding historical-mean benchmark.
- Each uses only sequential `cumsum` (a scan, so prefix values do not change when rows
  are appended), fixed windows, fixed HAC lags, and fixed `min_periods`. There is no
  stride evaluation because all of them are O(1) per date.

The named traps, their safe defaults, and the tests that pin them:

| # | Trap | Safe default in the API | Test |
|---|---|---|---|
| T1 | **VaR/ES misalignment.** The forecast "for t" was computed with r_t in the window (a contemporaneous VaR) | `var_convention` has no default. Docs and the `var_backtest` example construct forecasts with an explicit `shift(h).over(entity)`. `horizon` is recorded | leaky contemporaneous VaR yields an implausibly low hit rate; documented as a symptom |
| T2 | **Full-sample mean as the R²_OS benchmark** (uses future returns) | expanding mean as of t−h, `min_periods` fixed | `assert_no_lookahead` and `assert_prefix_invariant` on the benchmark op. A pinned test shows the full-sample mean changes R²_OS |
| T3 | **GR path uses future data**: centred windows, and a full-sample σ̂ | windows labelled at their end. `mode="monitor"` uses expanding σ̂. `mode="test"` is documented as non-prefix-invariant (`details["prefix_invariant"] = False`) | prefix-invariance on the monitor path; the test-mode sup equals GR's centred-label sup |
| T4 | **CORP recalibration reused as a forecast** (in-sample isotonic fit) | `CORPResult` exposes the recalibrated values only as a diagnostic, with the docstring warning. No `transform`/`predict` method exists | API test: no fit/transform surface |
| T5 | **n-dependent constants leaking into time-indexed outputs** (bandwidth, b, λ) | forbidden by construction: monitor paths take fixed `hac_lags`; "auto"/"andrews" are rejected in `mode="monitor"` | `ValueError` test |
| T6 | **Bootstrap blocks straddling CV folds** | `boundaries=` on every resampling API (A2 segment semantics) | no index crosses a boundary |
| T7 | **Invalid comparison designs**: DM on nested models; GW with expanding windows; testing the best of M with a pairwise test | CW for nested models. GW warns unless `estimation_scheme="rolling"`. M > 1 applies `multiple="romano_wolf"` by default | pinned size tests (§1.1 numbers) |
| T8 | **Overlapping h-step returns or forecasts** | HAC lags ≥ h − 1 from `horizon`, never from T | size test with h = 5 overlapping sums |
| T9 | **Luck-vs-skill null without the null imposed, or with independent per-fund resampling** | α̂ subtracted, joint dates by default | calibration test under a common factor |
| T10 | **√q annualisation under autocorrelation** | `annualize_sharpe` warns when autocorrelations are omitted | hand-value test |
| T11 | **Seeded but order-dependent RNG** | one generator per call for shared indices, drawn before chunking. Per-model generators are `default_rng([seed, crc32(name)])`, never a running counter | adding a column or reordering models leaves other models' marginal results bitwise unchanged |

Panel semantics:

- within-entity benchmarks use `.over(entity)` in time order;
- cross-sectional aggregation (`loss_panel`) uses same-date rows only;
- nothing here has a `fit`. The only fitted quantities are the whole-sample test
  constants above.

---

## 7. Tests

Registered markers only (`slow`, `benchmark`); `--strict-markers` is on. Oracles are
behind `pytest.importorskip`, or are frozen published values. GPL R packages supply
**values only**, never code.

| File | Contents (tolerances stated) |
|---|---|
| `tests/test_validation_engine.py` | count-matrix means vs gather (≤1e-14 rel); start-count CBB Ψ* vs literal LW ζ-form (≤1e-12 rel); segment semantics; chunking does not change output (bitwise); `_hac` Bartlett vs `newey_west_variance` (≤1e-12); QS-FFT vs direct O(T²) sum (≤1e-12); Andrews bandwidth hand value; suffix-max `romano_wolf` **bitwise** equal to the current output on a fixture set |
| `tests/test_validation_sharpe.py` | `iid` equals `_sharpe_estimator_std` (≤1e-14); `iid_normal` difference equals the Memmel closed form (hand-computed); the influence-HAC scalar path equals the 4-vector ∇′Ψ∇ with prewhitening off (≤1e-12); `exact_normal` CI vs `scipy.stats.nct` inversion (≤1e-8); Lo η(q) hand values; LW (2008) mutual-fund / hedge-fund p-values from Table 3 † as frozen fixtures (tolerance ±0.01 for the bootstrap, ±1e-6 for HAC once the data are sourced — **open**, §11); log-variance test equals the delta-method hand value; a strategy column added or removed leaves the others bitwise unchanged |
| `tests/test_validation_sharpe_calibration.py` (`slow`) | LW (2008) §4.2 DGPs at T = 120 (Normal-IID, t₆-IID, Normal/t₆-GARCH with LW's C/A/B matrices, Normal/t₆-VAR φ = 0.2), 2000 reps, B = 499. **JKM must over-reject** on t and VAR (> 0.07; measured 0.085/0.089, a "the bug is real" test that must never be loosened). Boot-TS with `"calibrate"` in [0.03, 0.07]. HAC in [0.03, 0.10] (known liberal). Many-vs-benchmark: M = 50 null strategies, FWER ≤ 0.07 under Romano–Wolf. Power sanity at ΔSR = 0.3/period |
| `tests/test_validation_forecast_compare.py` | CW, MZ, PT, ENC, and GW each vs a hand-rolled per-model loop (≤1e-12); MZ vs `statsmodels` OLS HAC (importorskip, ≤1e-8); PT exact vs `scipy.stats.fisher_exact` (≤1e-12); GW equals n·R² of 1 on Z (≤1e-10); nested-null size pinned (DM ≤ 0.02 and CW in [0.03, 0.12] at 10 %, 1000 reps, `slow` for the full version); `verify_gr_tables(quick=True)` (`slow`); prefix-invariance via `assert_prefix_invariant` **and** `assert_no_lookahead` on the monitor path, the expanding benchmark, the R²_OS path, and the cumulative SSE path; T2/T3/T5 trap tests; `loss_panel` date-mean equals a manual polars `group_by` on polars 1.35 semantics |
| `tests/test_validation_risk_backtest.py` | Kupiec exact vs `scipy.stats.binomtest` (≤1e-12); asymptotic size pinned (> 0.08 at T = 250, 1 %); exact size ≤ 0.05 + 3·MC-SE; Christoffersen MC null size ≤ 0.05 + MC-SE and the asymptotic LR_ind undersize pinned (< 0.025); the shared-null equivalence (per-model simulation vs shared: identical p-values); DQ vs a hand OLS; Z2 mean ≈ 0 under a correct PredictiveSpec (\|mean\| < 3 SE); Z2 rejects when VaR/ES are scaled by 0.8; FZ0 equals eq. (6) at the paper's Fig. 1 point (Y = −1, v = −1.64, e = −2.06); FZ0 expected loss minimised at the true (v, e) for N(0,1) on a grid; homogeneity of degree 0 of loss differences; QLIKE ranking robust to proxy noise (Patton 2011 property); `var_convention` required |
| `tests/test_validation_luck_skill.py` | `storey_pi0` on a 0.8/0.2 mixture (M = 5000): \|π̂₀ − 0.8\| < 0.05; `benjamini_hochberg(pi0=None)` bitwise equal to the current output; q ≤ BH; BSW identities (T̂ + F̂ = Ŝ); `alpha_bootstrap` count-matrix vs a gather+`lstsq` reference (≤1e-9 rel, balanced and unbalanced); `factors=None` reproduces `cross_sectional_bootstrap` (≤1e-12, identical p-values); **KTWW under a common factor gives an over-sized extreme-quantile test while FF2010 stays calibrated** (`slow`); `significance_hurdle`: Bonferroni closed form exact, Holm ≤ Bonferroni, monotone in M, ≈3 at HLZ's M † (loose) |
| `tests/test_validation_calibration.py` | PAV vs `scipy.optimize.isotonic_regression` and sklearn `IsotonicRegression` (≤1e-12); numba equals Python (the `test_numba_matches_python` pattern, ≤1e-15); tie pooling; decomposition identity (≤1e-12); MCB = 0 for already-recalibrated forecasts; DSC = 0 for climatology; band covers the diagonal ≥ 88 % for a calibrated simulation at level 0.90 (`slow`); the Murphy exact path vs a brute-force dense θ grid (≤1e-12); ∫S_θ dθ equals the pinball / squared / Brier score (Ehm et al. mixture identity, ≤1e-10); a worked example contrasting binned ECE across bin counts with CORP (docs) |
| `tests/test_validation_multivariate_scores.py` | ES exact vs direct (≤1e-10); d = 1 equals `crps_ensemble` (≤1e-12); sliced relative error at K = 1024 below a measured bound (seeded); propriety (the true distribution has the lowest mean score over 2000 draws, for ES and VS); VS vs a direct triple loop (≤1e-12); `scoringrules` † oracle (importorskip) |
| `tests/test_validation_eval_perf.py` (`benchmark`) | the §8 budgets with 2× headroom, skipped on slow CI like `test_depend_perf.py` |

Standing guards: `test_import_hygiene.py` (no scipy or numba at import; within the
300 ms budget), `test_dependency_drift.py` (zero new mandatory dependencies),
`test_wheel_guardrails.py`, mypy (zero new errors against the ratchet), and ruff.

---

## 8. Benchmarks and performance budgets

Reference machine: this laptop (Apple Silicon, Python 3.13.12, NumPy 2.5.3 + Accelerate,
polars 1.44.2), single process. "Measured" rows were measured this session. Every other
row is a target the implementation must meet or explain.

| Operation | Size | Budget | Basis |
|---|---|---|---|
| bootstrap means, count matrix | B=1000, T=5000, M=1000 | ≤ 50 ms | **measured 33 ms** (gather: 2.98 s) |
| studentized CBB, M vs benchmark | B=1000, T=5000, M=1000 | ≤ 1.0 s | 5 gemms **measured 0.12 s**; ≈ 9 needed |
| pairwise LW Boot-TS + calibration | T=120, K=1000, grid 6, B=499 | ≤ 1 s | K pseudo-sequences as columns |
| same | T=5000 | ≤ 10 s | 14 gemms per b at 499×5000×1000 |
| QS-PW column LRV | T=5000, M=1000 | ≤ 0.3 s | FFT **measured 0.14 s** |
| Bartlett column LRV, L=9 | T=5000, M=1000 | ≤ 20 ms | **measured 9 ms** |
| Romano–Wolf stepdown | S=1000, B=1000 | ≤ 20 ms | **measured 6 ms** (current: 0.50 s) |
| CW / MZ / PT / ENC / GW | T=5000, M=1000 | ≤ 0.3 s each | O(LTM) |
| fluctuation (both modes) | P=5000, M=1000 | ≤ 0.3 s | cumsum |
| Kupiec exact | M=1000, T=5000 | ≤ 50 ms | pmf shared per (T, α) |
| Christoffersen exact MC null | B=9999, T=1000 | ≤ 0.1 s, once | **measured 43 ms** |
| DQ asymptotic | M=1000, T=5000 | ≤ 0.2 s | batched 6×6 solves |
| AS Z2 MC | M=100, T=1000, B=2000 | ≤ 3 s | (B, T) draws per model |
| Storey π₀ bootstrap | M=10⁴, B=1000 | ≤ 0.2 s | O(BM) bincount |
| `alpha_bootstrap` FF2010, unbalanced | T=360, N=3000, K=3, B=1000 | ≤ 3 s | 3·10⁶ 4×4 solves **measured 0.70 s** |
| CORP, no band | n=10⁶ binary | ≤ 0.3 s | argsort 76 ms + PAV 125 ms **measured** |
| CORP consistency band | G=10⁴, B=1000 | ≤ 2 s pure / ≤ 0.1 s numba | PAV 1 ms at 10⁴ **measured** |
| Murphy exact | n=10⁶, M=10 | ≤ 2 s | O(n log n) |
| energy score exact | m=1000, d=50 | ≤ 2.5 ms/obs | **measured 2.08** |
| variogram score | m=1000, d=50 | ≤ 2.5 ms/obs | **measured 1.81** |

Memory ceilings:

- the resampling engine stays under 300 MB at the sizes above;
- every function takes or derives a `chunk_bytes` (default 256 MB) and never allocates
  (B, T, M) or (T, m, m) whole.

The benchmark script `benchmarks/bench_forecast_eval.py` reproduces every row.
`tests/test_validation_eval_perf.py` asserts the budgets with 2× headroom.

---

## 9. Dependencies

- **Core: zero new dependencies.** Everything runs on numpy + polars plus
  `econ._common` / `_internal` helpers.
- **Optional (`fast` = numba, already declared).** Only two kernels use it: the PAV
  kernel (bands and large G) and the fused variogram kernel. Both load lazily through
  `have("numba")` with `@njit(cache=True, error_model="numpy")`, and each has a
  numba-equals-python test.
- **Test-only oracles** (behind `importorskip`):
  - scipy: `nct`, `binomtest`, `fisher_exact`, `optimize.isotonic_regression`, `chi2`;
  - scikit-learn: `IsotonicRegression`;
  - statsmodels: HAC OLS (not installed in the local venv; CI skips gracefully);
  - `arch`: SPA/MCS cross-check for the M7 retrofit;
  - `scoringrules` †.
- **Frozen fixture values** (never code) from the R packages PeerPerformance,
  murphydiagram, reliabilitydiag, and esback, whose licences may be GPL.

---

## 10. Milestones

Each milestone ships independently, with its own tests, docs section, and exports. The
AGENTS.md gate (§12) decides what becomes public.

| M | Contents | Ships | Gate |
|---|---|---|---|
| **M1** | `_results`, `_hac`, `_resample`, `_sharpe` (all methods, pairwise and M-vs-benchmark, log-variance, calibration of b, Lo annualisation); suffix-max `romano_wolf` | Sharpe CIs and difference tests that hold their size under fat tails and serial correlation; the roadmap item closes | `test_validation_sharpe_calibration.py` green (slow suite) |
| **M2** | `_calibration`: PAV (+ numba), `corp_reliability`, `murphy_diagram` | stable reliability diagrams and the score decomposition; truepoint M07 has a drop-in successor to binned ECE | PAV parity against both oracles |
| **M3** | `_forecast_compare`, `_gr_tables`, `loss_panel`, `_dm_columns` | nested, conditional, and unstable forecast comparison on panels | GR tables verified; prefix-invariance suite green |
| **M4** | `_risk_backtest` (Kupiec exact, Christoffersen MC, DQ, AS Z1/Z2, FZ0, QLIKE, `var_backtest`) | honest VaR/ES backtests with exact small-sample nulls | size tests at T = 250 green |
| **M5** | `storey_pi0`, `benjamini_hochberg(pi0=)`, `luck_versus_skill`, `alpha_bootstrap`, `significance_hurdle` | luck vs skill across many strategies or agents | KTWW vs FF2010 calibration test green |
| **M6** | `energy_score`, `variogram_score` | multivariate proper scores | propriety tests |
| **M7** (optional) | retrofit SPA, MCS, `romano_wolf_mean_test`, and `cross_sectional_bootstrap` onto the count-matrix engine (same indices, so identical p-values) | 10–100× faster existing tests | golden-value regression on the current fixtures; `arch` cross-check |

Order rationale:

- M1 is the core of the request and unlocks the shared engine.
- M2 has the **only caller that exists today** (truepoint M07), and it can move to first
  if the orchestrator enforces the caller rule strictly.
- M4 can run in parallel with M3 once M1's `_results` / `_hac` exist.
- Do not start M7 before M1's engine tests are green.

---

## 11. Risks and open questions

1. **LW parity details.** The exact Andrews–Monahan multivariate bandwidth weights and
   prewhitening caps in Wolf's published code are unverified †. Resolve by freezing
   p-values from his code (or PeerPerformance) on public or synthetic data, then aligning
   the choices. If they cannot be matched, document the deviation, as the depend
   contract did.
2. **Block length for M > 1.** Politis–White was measured over-rejecting in `depend`.
   Using the maximum over columns is conservative but unproven. The calibration test
   decides; the fallback is `"calibrate"` on the median-influence strategy.
3. **GR critical values.** The shipped tables must reproduce GR Table 1 and the
   one-time-reversal values †. `mode="monitor"` uses the same asymptotic k_α. Its
   finite-sample size is unknown and must be measured. If it is off by more than 2
   percentage points at P = 500, the monitor path is labelled a diagnostic with no
   p-value.
4. **Discreteness.** The Kupiec exact and Christoffersen MC tests are conservative
   (measured exact size 0.042 at a nominal 0.05). Mid-p or randomised tie-breaking would
   restore size, but it trades away determinism or validity. Keep the conservative rule;
   report mid-p in `details` only if a caller asks.
5. **ES backtests need a predictive distribution** for p-values. Without `PredictiveSpec`
   only thresholds exist. The ESR test (Bayer–Dimitriadis) avoids this and is the
   natural follow-up if a caller needs it.
6. **CORP bands at large G without numba.** The `band="auto"` policy drops the band with
   a warning. The asymptotic bands of DGJ 2021 † are a later addition.
7. **Count-matrix RSS cancellation** in `alpha_bootstrap`. There is a
   condition-number fallback. The measured worst case on real fund data is an open item.
8. **Name collisions.**
   - "Ledoit–Wolf": the Sharpe test here versus shrinkage in sibling 3.
   - "Variance ratio": the LW log-variance test here versus the Lo–MacKinlay random-walk
     VR test. The latter is not in scope, and the docs must say so.
   - "Truepoint" versus the RIA, noted in the universe CLAUDE.md. This plan never ships
     that name.
9. **Shared primitives with siblings.**
   - `_hac.py` (column LRV) and `_resample.py` (count matrix, CBB) are proposed as the
     single homes. Siblings 2 (sequential tests on loss differentials) and 9 (GARCH
     bootstrap) should import them rather than write their own.
   - If a sibling lands a vectorised normal-tail primitive first, reuse it and drop the
     `norm_sf` promotion.
   - PAV is proposed as private here. If sibling 2 needs isotonic fits for calibration
     drift, promote it to `_internal`.
10. **Retrofitting existing tests (M7) changes floating output at the 1e-16 level.**
    Accept only with golden p-value equality. Otherwise the old paths stay.

---

## 12. Caller

**AGENTS.md rule:** *"No new public surface without a named caller in the engine, and a
test in the engine that exercises it."* The user asked for this plan explicitly, but the
rule still governs exports. Every function in §4.3 ships **private** (`_`-prefixed,
unexported) until its caller and engine test exist. Only then is it added to
`validation.__all__`.

| Capability | Most plausible caller | Status |
|---|---|---|
| `corp_reliability`, `murphy_diagram` | `truepoint/src/truepoint/inference/calibration.py` (M07: binned `ece` today; `aurc` is a Wave-2 stub) and `truepoint/src/truepoint/validity/calibration_check.py` | **the caller exists.** Engine test: CORP MCB/DSC/UNC on a synthetic confidence/correctness fixture (never sealed data) |
| Sharpe inference, VaR/ES backtests, forecast-comparison tests | `truepoint/src/truepoint/quant/` (AGENTS.md's named caller, not yet created): re-testing an assessed financial AI's quantitative claims ("beats the benchmark", "VaR is calibrated"). `EvaluationResult.to_dict()` becomes `Evidence` | caller planned, not built |
| `luck_versus_skill`, `storey_pi0`, `benjamini_hochberg(pi0=)` | assessments over many agents or models: what share of apparent outperformance survives the FDR | plausible, via truepoint `inference/` |
| in-repo consumers | `evolve.TrialLedger` (HAC SE into DSR via `sharpe_std=`); `core.model_selection.cross_validate` (path-Sharpe CIs); `embed.baseline_report` (`oos_r2` with a CW p-value) | internal; does not satisfy the rule by itself |

No sealed ground-truth questions, seeds, or answer keys appear in any test, fixture, or
doc produced under this plan.

---

## 13. References

Marked † where not verified this session. Ledoit–Wolf (2008) and Patton–Ziegel–Chen
(2019) were read in full or in the relevant sections. Giacomini–Rossi (2010) and
Acerbi–Székely (2014) were confirmed by search, but their tables were not read.

- Acerbi, C. & Székely, B. (2014). Back-testing expected shortfall. *Risk* 27(11), 76–81. (Z2 thresholds −0.70…−0.82 via MathWorks docs †)
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica* 59, 817–858.
- Andrews, D. W. K. & Monahan, J. C. (1992). An improved HAC covariance matrix estimator. *Econometrica* 60, 953–966.
- Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. & Silverman, E. (1955). *Ann. Math. Stat.* 26(4), 641–647. †
- Bailey, D. H. & López de Prado, M. (2012, 2014). PSR; Deflated Sharpe Ratio. *J. Risk* 15(2) †; *JPM* 40(5).
- Barras, L., Scaillet, O. & Wermers, R. (2010). False discoveries in mutual fund performance. *J. Finance* 65(1), 179–216. †
- Bayer, S. & Dimitriadis, T. (2022). Regression-based expected shortfall backtesting. *J. Financial Econometrics* 20(3), 437–471. †
- Best, M. J. & Chakravarti, N. (1990). Active set algorithms for isotonic regression. *Math. Programming* 47, 425–439. †
- Campbell, J. Y. & Thompson, S. B. (2008). Predicting excess stock returns out of sample. *RFS* 21(4), 1509–1531. †
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *IER* 39(4), 841–862. †
- Clark, T. E. & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *J. Econometrics* 138(1), 291–311. †
- Dimitriadis, T., Gneiting, T. & Jordan, A. I. (2021). Stable reliability diagrams for probabilistic classifiers. *PNAS* 118(8), e2016191118. †
- Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters. *J. Econometrics* 133(2), 443–477. †
- Ehm, W., Gneiting, T., Jordan, A. & Krüger, F. (2016). Of quantiles and expectiles: consistent scoring functions, Choquet representations and forecast rankings. *JRSS-B* 78(3), 505–562. †
- Engle, R. F. & Manganelli, S. (2004). CAViaR. *JBES* 22(4), 367–381. †
- Fama, E. F. & French, K. R. (2010). Luck versus skill in the cross-section of mutual fund returns. *J. Finance* 65(5), 1915–1947. †
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *QJRMS* 140, 1917–1923. †
- Fissler, T. & Ziegel, J. F. (2016). Higher order elicitability and Osband's principle. *Ann. Statist.* 44(4), 1680–1707. †
- Giacomini, R. & Rossi, B. (2010). Forecast comparisons in unstable environments. *J. Applied Econometrics* 25(4), 595–620. (fluctuation and one-time-reversal tests confirmed; Table 1 values not read †)
- Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6), 1545–1578. †
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477), 359–378. †
- Gneiting, T., Stanberry, L., Grimit, E., Held, L. & Johnson, N. (2008). Assessing probabilistic forecasts of multivariate quantities. *TEST* 17(2), 211–235. †
- Gneiting, T. & Resin, J. (2023). Regression diagnostics meets forecast evaluation. *Electron. J. Statist.* 17, 3226–3286. †
- Götze, F. & Künsch, H. R. (1996). Second-order correctness of the blockwise bootstrap for stationary observations. *Ann. Statist.* 24, 1914–1933.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *RFS* 33(5), 2223–2273. †
- Harvey, C. R. & Liu, Y. (2015). Backtesting. *JPM* 42(1), 13–28.
- Harvey, C. R., Liu, Y. & Zhu, H. (2016). … and the cross-section of expected returns. *RFS* 29(1), 5–68.
- Harvey, D., Leybourne, S. & Newbold, P. (1997, 1998). *IJF* 13(2); Tests for forecast encompassing, *JBES* 16(2), 254–259. †
- Jobson, J. D. & Korkie, B. (1981). Performance hypothesis testing with the Sharpe and Treynor measures. *J. Finance* 36(4), 889–908.
- Kosowski, R., Timmermann, A., Wermers, R. & White, H. (2006). Can mutual fund "stars" really pick stocks? *J. Finance* 61(6), 2551–2595. †
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *J. Derivatives* 3(2), 73–84. †
- Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *J. Empirical Finance* 15(5), 850–859. **Read this session** (§3, Alg. 3.1, eq. (9), §4–5).
- Ledoit, O. & Wolf, M. (2011). Robust performances hypothesis testing with the variance. *Wilmott* 2011(55), 86–89. (existence confirmed by search; content †)
- Lenth, R. V. (1989). Algorithm AS 243: cumulative distribution function of the non-central t distribution. *Applied Statistics* 38(1), 185–189. †
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36–52.
- Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters* 1, 21–23.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper, Univ. Basel. †
- Mincer, J. & Zarnowitz, V. (1969). The evaluation of economic forecasts. NBER. †
- Nolde, N. & Ziegel, J. F. (2017). Elicitability and backtesting: perspectives for banking regulation. *Ann. Appl. Stat.* 11(4), 1833–1874. †
- Opdyke, J. D. (2007). Comparing Sharpe ratios: so where are the p-values? *J. Asset Management* 8(5), 308–336. (critique of its time-series formulas: LW 2008 Remark 3.1, read)
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *J. Econometrics* 160(1), 246–256. †
- Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and Value-at-Risk). *J. Econometrics* 211(2), 388–413. **Eq. (6) read this session.**
- Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *JBES* 10(4), 461–465. †
- Pesaran, M. H. & Timmermann, A. (2009). Testing dependence among serially correlated multicategory variables. *JASA* 104(485), 325–337. †
- Pinson, P. & Tastu, J. (2013). Discrimination ability of the energy score. DTU technical report. †
- Politis, D. N. & Romano, J. P. (1992, 1994). Circular block resampling; The stationary bootstrap. †
- Politis, D. N. & White, H. (2004); Patton, A., Politis, D. N. & White, H. (2009). Automatic block-length selection. *Econometric Reviews* 23(1); 28(4). †
- Romano, J. P. & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4), 1237–1282.
- Scheuerer, M. & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *MWR* 143(4), 1321–1334. †
- Storey, J. D. (2002). A direct approach to false discovery rates. *JRSS-B* 64(3), 479–498. †
- Storey, J. D., Taylor, J. E. & Siegmund, D. (2004). *JRSS-B* 66(1), 187–205. †
- West, K. D. (1996). Asymptotic inference about predictive ability. *Econometrica* 64(5), 1067–1084. †
- Yan, X. & Zheng, L. (2017). Fundamental analysis and the cross-section of stock returns: a data-mining approach. *RFS* 30(4), 1382–1423.
