"""VaR / ES backtests (``validation._risk_backtest``).

The size tests at T = 250 pin the measured facts that motivate the exact
nulls: the asymptotic Kupiec LR over-rejects about 2x, and the asymptotic
Christoffersen ``LR_ind`` is badly undersized. The exact versions hold their
size. All of it is vectorised over columns, so 20,000 replications run in a
fraction of a second.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

from panelary._internal._special import norm_pdf, norm_ppf
from panelary.testing import assert_no_lookahead, assert_prefix_invariant
from panelary.validation import _risk_backtest as rb
from panelary.validation._results import EVALUATION_SCHEMA
from panelary.validation._risk_backtest import (
    PredictiveSpec,
    acerbi_szekely_test,
    christoffersen_test,
    dynamic_quantile_test,
    exceedances,
    fz0_loss,
    kupiec_test,
    qlike_loss,
    var_backtest,
)

ALPHA_ES = 0.025
Q_ES = float(norm_ppf(ALPHA_ES))
ES_STD = -float(norm_pdf(Q_ES)) / ALPHA_ES


def _mc_se(p: float, reps: int) -> float:
    return math.sqrt(p * (1 - p) / reps)


# --------------------------------------------------------------------------- #
# exceedances
# --------------------------------------------------------------------------- #
def test_exceedances_lower_upper_and_nan() -> None:
    y = np.array([-0.03, -0.01, 0.02, np.nan, 0.05])
    lo = np.array([-0.02, -0.02, -0.02, -0.02, np.nan])
    out = exceedances(y, lower=lo)
    assert out.shape == (5, 1)
    np.testing.assert_array_equal(out[:, 0], [1.0, 0.0, 0.0, np.nan, np.nan])
    band = exceedances(y, lower=np.full(5, -0.02), upper=np.full(5, 0.03))
    np.testing.assert_array_equal(band[:, 0], [1.0, 0.0, 0.0, np.nan, 1.0])
    # strict inequality: touching the bound is not a hit
    assert exceedances(np.array([-0.02]), lower=np.array([-0.02]))[0, 0] == 0.0


def test_exceedances_broadcasts_models() -> None:
    y = np.array([-0.03, 0.0, 0.01])
    lo = np.column_stack([np.full(3, -0.02), np.full(3, -0.005)])
    out = exceedances(y, lower=lo)
    np.testing.assert_array_equal(out, [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="lower"):
        exceedances(y)
    with pytest.raises(ValueError, match="row counts"):
        exceedances(y, lower=np.zeros(4))


# --------------------------------------------------------------------------- #
# Kupiec
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [250, 1000])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_kupiec_exact_matches_scipy_binomtest(n: int, alternative: str) -> None:
    st = pytest.importorskip("scipy.stats")
    alpha = 0.01
    xs = np.arange(0, 30)
    hits = np.zeros((n, xs.size))
    for j, x in enumerate(xs):
        hits[:x, j] = 1.0
    res = kupiec_test(hits, level=alpha, alternative=alternative)
    ref = np.array(
        [st.binomtest(int(x), n, alpha, alternative=alternative).pvalue for x in xs]
    )
    np.testing.assert_allclose(res.pvalue, ref, rtol=1e-12, atol=0)
    assert res.reference == "binomial-exact"
    np.testing.assert_array_equal(res.statistic, xs)
    np.testing.assert_allclose(res.estimate, xs / n)


def test_kupiec_size_at_t250_exact_holds_asymptotic_over_rejects() -> None:
    reps, t, alpha = 20_000, 250, 0.01
    hits = (np.random.default_rng(20260929).random((t, reps)) < alpha).astype(float)
    res = kupiec_test(hits, level=alpha)
    exact = float(np.mean(res.pvalue <= 0.05))
    asym = float(np.mean(res.details["pvalue_asymptotic"] <= 0.05))
    assert exact <= 0.05 + 3 * _mc_se(0.05, reps)
    # "the bug is real": chi-square(1) over-rejects about 2x here. Never loosen.
    assert asym > 0.08


def test_kupiec_handles_missing_rows_per_model() -> None:
    rng = np.random.default_rng(1)
    hits = (rng.random((300, 3)) < 0.02).astype(float)
    hits[:50, 1] = np.nan
    hits[:, 2] = np.nan
    res = kupiec_test(hits, level=0.02, names=["a", "b", "c"])
    np.testing.assert_array_equal(res.n_obs, [300, 250, 0])
    solo = kupiec_test(hits[50:, 1], level=0.02)
    assert res.pvalue[1] == solo.pvalue[0]
    assert np.isnan(res.pvalue[2]) and any("no finite hits" in w for w in res.warnings)
    asym = kupiec_test(hits[:, :2], level=0.02, pvalue="asymptotic")
    assert asym.reference == "chi2(1)"
    np.testing.assert_allclose(asym.statistic, res.details["lr_pof"][:2])
    with pytest.raises(ValueError, match="two-sided"):
        kupiec_test(hits, level=0.02, pvalue="asymptotic", alternative="greater")
    with pytest.raises(ValueError, match="0, 1 or NaN"):
        kupiec_test(np.array([0.0, 2.0]), level=0.02)


# --------------------------------------------------------------------------- #
# Christoffersen
# --------------------------------------------------------------------------- #
def test_christoffersen_counts_and_lr_by_hand() -> None:
    h = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=float)
    res = christoffersen_test(h, level=0.1, kind="independence", null="asymptotic")
    d = res.details
    assert (d["n00"][0], d["n01"][0], d["n10"][0], d["n11"][0]) == (4, 2, 2, 1)
    pi01, pi11, pi = 2 / 6, 1 / 3, 3 / 9
    l1 = 4 * math.log(1 - pi01) + 2 * math.log(pi01) + 2 * math.log(1 - pi11)
    l1 += math.log(pi11)
    l0 = 6 * math.log(1 - pi) + 3 * math.log(pi)
    assert res.statistic[0] == pytest.approx(2 * (l1 - l0), abs=1e-12)
    lr_uc = -2 * (
        7 * math.log(0.9) + 3 * math.log(0.1) - 7 * math.log(0.7) - 3 * math.log(0.3)
    )
    cc = christoffersen_test(h, level=0.1, null="asymptotic")
    assert cc.statistic[0] == pytest.approx(lr_uc + 2 * (l1 - l0), abs=1e-12)
    assert cc.reference == "chi2(2)"
    assert cc.pvalue[0] == pytest.approx(math.exp(-cc.statistic[0] / 2), rel=1e-14)


def test_christoffersen_nan_breaks_pairs() -> None:
    h = np.array([1, 1, np.nan, 1, 0, 1, 1], dtype=float)
    res = christoffersen_test(h, level=0.1, null="asymptotic")
    d = res.details
    # pairs: (1,1), (1,0), (0,1), (1,1); the NaN removes (1,nan) and (nan,1)
    assert (d["n00"][0], d["n01"][0], d["n10"][0], d["n11"][0]) == (0, 1, 1, 2)
    assert res.n_obs[0] == 6


def test_christoffersen_sizes_at_t250() -> None:
    reps, t, alpha = 20_000, 250, 0.01
    hits = (np.random.default_rng(7).random((t, reps)) < alpha).astype(float)
    ind = christoffersen_test(hits, level=alpha, kind="independence", n_sims=9999)
    testable = np.isfinite(ind.pvalue)
    mc = float(np.mean(ind.pvalue[testable] <= 0.05))
    asym = float(np.mean(ind.details["pvalue_asymptotic"][testable] <= 0.05))
    assert mc <= 0.05 + _mc_se(0.05, int(testable.sum()))
    # the asymptotic LR_ind is badly undersized (measured 0.013); never loosen
    assert asym < 0.025
    cc = christoffersen_test(hits, level=alpha, n_sims=9999)
    assert float(np.mean(cc.pvalue <= 0.05)) <= 0.05 + _mc_se(0.05, reps)


def test_christoffersen_shared_null_equals_per_model_null() -> None:
    rng = np.random.default_rng(8)
    hits = (rng.random((400, 6)) < 0.03).astype(float)
    hits[:37, 4] = np.nan  # a second missing-value pattern
    names = [f"m{j}" for j in range(6)]
    together = christoffersen_test(hits, level=0.03, n_sims=999, seed=5, names=names)
    for j in range(6):
        alone = christoffersen_test(hits[:, j], level=0.03, n_sims=999, seed=5)
        assert alone.pvalue[0] == together.pvalue[j]
    # reordering or dropping models never changes another model's p-value
    order = [5, 2, 0]
    sub = christoffersen_test(
        hits[:, order], level=0.03, n_sims=999, seed=5, names=[names[i] for i in order]
    )
    np.testing.assert_array_equal(sub.pvalue, together.pvalue[order])


def test_christoffersen_detects_clustered_exceptions() -> None:
    t = 500
    h = np.zeros(t)
    h[100:105] = 1.0  # five consecutive exceptions at the expected rate 1%
    res = christoffersen_test(h, level=0.01, kind="independence", n_sims=9999)
    assert res.pvalue[0] < 0.01
    # pi11 = 4/5 (after an exception), pi01 = 1/494 (after a quiet day)
    assert res.estimate[0] == pytest.approx(0.8 - 1 / 494, abs=1e-12)


def test_christoffersen_no_exceptions_is_untestable() -> None:
    h = np.zeros((300, 2))
    h[10, 1] = 1.0
    res = christoffersen_test(h, level=0.01, kind="independence")
    assert np.isnan(res.pvalue[0]) and np.isnan(res.statistic[0])
    assert any("untestable" in w for w in res.warnings)
    cc = christoffersen_test(h, level=0.01)
    assert np.isfinite(cc.pvalue).all()  # coverage is still informative


def test_mc_null_chunking_does_not_change_results(monkeypatch) -> None:
    rng = np.random.default_rng(9)
    hits = (rng.random((300, 3)) < 0.02).astype(float)
    base = christoffersen_test(hits, level=0.02, n_sims=499)
    monkeypatch.setattr(rb, "_SIM_CHUNK_CELLS", 1000)
    chunked = christoffersen_test(hits, level=0.02, n_sims=499)
    np.testing.assert_array_equal(base.pvalue, chunked.pvalue)


# --------------------------------------------------------------------------- #
# DQ
# --------------------------------------------------------------------------- #
def _dq_by_hand(h: np.ndarray, v: np.ndarray, alpha: float, lags: int) -> float:
    t = h.shape[0]
    cols = [np.ones(t - lags)] + [h[lags - j : t - j] for j in range(1, lags + 1)]
    x = np.column_stack([*cols, v[lags:]])
    y = h[lags:] - alpha
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(beta @ x.T @ x @ beta / (alpha * (1 - alpha)))


def test_dq_matches_hand_ols() -> None:
    rng = np.random.default_rng(10)
    t, m = 600, 4
    hits = (rng.random((t, m)) < 0.05).astype(float)
    var = -1.64 + 0.2 * rng.standard_normal((t, m))
    res = dynamic_quantile_test(hits, var, level=0.05)
    for j in range(m):
        assert res.statistic[j] == pytest.approx(
            _dq_by_hand(hits[:, j], var[:, j], 0.05, 4), rel=1e-10
        )
    assert res.reference == "chi2(6)"
    np.testing.assert_array_equal(res.details["df"], 6.0)
    assert np.all(res.n_obs == t - 4)


def test_dq_rank_deficient_design_reduces_df() -> None:
    rng = np.random.default_rng(11)
    t = 500
    h = (rng.random(t) < 0.05).astype(float)
    const = np.full(t, -1.64)  # collinear with the intercept
    res = dynamic_quantile_test(h, const, level=0.05, n_lags=2)
    assert res.details["df"][0] == 3.0
    reduced = np.column_stack([np.ones(t - 2), h[1:-1], h[:-2]])
    y = h[2:] - 0.05
    beta = np.linalg.lstsq(reduced, y, rcond=None)[0]
    hand = beta @ reduced.T @ reduced @ beta / (0.05 * 0.95)
    assert res.statistic[0] == pytest.approx(hand, rel=1e-10)
    assert any("rank-deficient" in w for w in res.warnings)
    # no exceptions at all is informative (coverage), not a crash
    none = dynamic_quantile_test(np.zeros(t), rng.standard_normal(t), level=0.05)
    # 496 rows of (0 - 0.05) projected on [1, VaR]: DQ = 496 * 0.05 / 0.95, df 2
    assert none.statistic[0] == pytest.approx(496 * 0.05 / 0.95, rel=1e-10)
    assert none.pvalue[0] == pytest.approx(math.exp(-none.statistic[0] / 2), rel=1e-12)


def test_dq_mc_null_is_seeded_by_name_and_detects_dependence() -> None:
    rng = np.random.default_rng(12)
    t = 400
    iid = (rng.random(t) < 0.05).astype(float)
    clustered = np.zeros(t)
    for s in rng.choice(t - 4, 5, replace=False):
        clustered[s : s + 4] = 1.0
    var = -1.64 + 0.1 * rng.standard_normal(t)
    hits = np.column_stack([iid, clustered])
    res = dynamic_quantile_test(
        hits, var, level=0.05, null="mc", n_sims=199, names=["iid", "clus"]
    )
    assert res.reference == "mc-exact(B=199)"
    assert res.pvalue[1] <= 0.01 and res.pvalue[0] > 0.01
    again = dynamic_quantile_test(
        clustered, var, level=0.05, null="mc", n_sims=199, names=["clus"]
    )
    assert again.pvalue[0] == res.pvalue[1]


# --------------------------------------------------------------------------- #
# Acerbi–Székely
# --------------------------------------------------------------------------- #
def _normal_case(t: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sig = 0.01 * np.exp(0.2 * rng.standard_normal(t))
    return sig, sig[:, None] * rng.standard_normal((t, m))


def test_z2_mean_zero_under_correct_predictive() -> None:
    sig, r = _normal_case(500, 300, 13)
    res = acerbi_szekely_test(
        r, sig * Q_ES, sig * ES_STD, level=ALPHA_ES, var_convention="quantile"
    )
    z = res.statistic
    assert abs(z.mean()) < 3 * z.std(ddof=1) / math.sqrt(z.size)
    assert np.isnan(res.pvalue).all() and res.reference == "none (statistic only)"
    assert float(res.details["z2_threshold_5pct_high"]) == -0.70
    assert any("no PredictiveSpec" in w for w in res.warnings)


def test_z2_rejects_when_var_es_scaled_down() -> None:
    sig, r = _normal_case(1000, 20, 14)
    spec = PredictiveSpec("normal", loc=np.zeros(1000), scale=0.8 * sig)
    res = acerbi_szekely_test(
        r,
        0.8 * sig * Q_ES,
        0.8 * sig * ES_STD,
        level=ALPHA_ES,
        var_convention="quantile",
        predictive=spec,
        n_sims=500,
    )
    assert float(np.mean(res.pvalue <= 0.05)) >= 0.9
    assert res.alternative == "less" and res.reference.startswith("mc-exact(B=500")
    good = acerbi_szekely_test(
        r[:, :5],
        sig * Q_ES,
        sig * ES_STD,
        level=ALPHA_ES,
        var_convention="quantile",
        predictive=PredictiveSpec("normal", loc=np.zeros(1000), scale=sig),
        n_sims=500,
    )
    assert np.all(good.pvalue > 0.01)


def test_as_loss_and_quantile_conventions_agree_and_sign_is_checked() -> None:
    sig, r = _normal_case(300, 3, 15)
    spec = PredictiveSpec("normal", loc=np.zeros(300), scale=sig)
    q = acerbi_szekely_test(
        r,
        sig * Q_ES,
        sig * ES_STD,
        level=ALPHA_ES,
        var_convention="quantile",
        predictive=spec,
        n_sims=200,
    )
    loss = acerbi_szekely_test(
        r,
        -sig * Q_ES,
        -sig * ES_STD,
        level=ALPHA_ES,
        var_convention="loss",
        predictive=spec,
        n_sims=200,
    )
    np.testing.assert_array_equal(q.statistic, loss.statistic)
    np.testing.assert_array_equal(q.pvalue, loss.pvalue)
    with pytest.raises(ValueError, match="did you mean var_convention='loss'"):
        acerbi_szekely_test(
            r, -sig * Q_ES, -sig * ES_STD, level=ALPHA_ES, var_convention="quantile"
        )
    with pytest.raises(ValueError, match="confidence level"):
        acerbi_szekely_test(r, sig, sig, level=0.975, var_convention="loss")


def test_z1_and_other_predictive_families() -> None:
    sig, r = _normal_case(400, 4, 16)
    for spec in (
        PredictiveSpec("normal", loc=np.zeros(400), scale=sig),
        PredictiveSpec("student_t", loc=np.zeros(400), scale=sig, df=5.0),
        PredictiveSpec(
            "fhs",
            loc=np.zeros(400),
            scale=sig,
            residuals=np.random.default_rng(0).standard_normal(600),
        ),
    ):
        res = acerbi_szekely_test(
            r,
            sig * Q_ES,
            sig * ES_STD,
            level=ALPHA_ES,
            var_convention="quantile",
            kind="z1",
            predictive=spec,
            n_sims=300,
        )
        assert res.test == "acerbi_szekely_z1"
        assert np.all((res.pvalue > 0) & (res.pvalue <= 1))
        assert np.all(res.details["n_null"] > 250)
    none = acerbi_szekely_test(
        np.full(50, 1.0),
        np.full(50, -1.0),
        np.full(50, -1.5),
        level=0.05,
        var_convention="quantile",
        kind="z1",
    )
    assert np.isnan(none.statistic[0])


def test_as_chunking_and_model_order_do_not_change_results(monkeypatch) -> None:
    sig, r = _normal_case(300, 3, 17)
    spec = PredictiveSpec("normal", loc=np.zeros(300), scale=sig)
    kwargs = {"level": ALPHA_ES, "var_convention": "quantile", "predictive": spec}
    base = acerbi_szekely_test(
        r, sig * Q_ES, sig * ES_STD, n_sims=300, names=["a", "b", "c"], **kwargs
    )
    monkeypatch.setattr(rb, "_SIM_CHUNK_CELLS", 700)
    chunked = acerbi_szekely_test(
        r, sig * Q_ES, sig * ES_STD, n_sims=300, names=["a", "b", "c"], **kwargs
    )
    np.testing.assert_array_equal(base.pvalue, chunked.pvalue)
    rev = acerbi_szekely_test(
        r[:, ::-1],
        sig * Q_ES,
        sig * ES_STD,
        n_sims=300,
        names=["c", "b", "a"],
        **kwargs,
    )
    np.testing.assert_array_equal(rev.pvalue[::-1], base.pvalue)


def test_predictive_spec_validation() -> None:
    with pytest.raises(ValueError, match="family"):
        PredictiveSpec("laplace", loc=np.zeros(3), scale=np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        PredictiveSpec("normal", loc=np.zeros(3), scale=np.zeros(3))
    with pytest.raises(ValueError, match="df"):
        PredictiveSpec("student_t", loc=np.zeros(3), scale=np.ones(3), df=2.0)
    with pytest.raises(ValueError, match="residuals"):
        PredictiveSpec("fhs", loc=np.zeros(3), scale=np.ones(3))


# --------------------------------------------------------------------------- #
# FZ0 and QLIKE
# --------------------------------------------------------------------------- #
def test_fz0_matches_equation_6_at_figure_1_point() -> None:
    y, v, e, a = -1.0, -1.64, -2.06, 0.05
    expected = v / e + math.log(-e) - 1.0  # y > v, so the hit term vanishes
    assert float(fz0_loss(y, v, e, level=a)) == pytest.approx(expected, abs=1e-15)
    y2 = -3.0  # a hit
    expected2 = -1.0 / (a * e) * (v - y2) + v / e + math.log(-e) - 1.0
    assert float(fz0_loss(y2, v, e, level=a)) == pytest.approx(expected2, abs=1e-15)


def test_fz0_expected_loss_is_minimised_at_the_true_pair() -> None:
    a = 0.05
    v0 = float(norm_ppf(a))
    e0 = -float(norm_pdf(v0)) / a
    y = np.random.default_rng(18).standard_normal(400_000)
    steps = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])  # every (v, e) pair keeps e < v
    means = np.array(
        [
            [np.mean(fz0_loss(y, v0 + dv, e0 + de, level=a)) for de in steps]
            for dv in steps
        ]
    )
    assert np.isfinite(means).all()
    assert np.unravel_index(np.argmin(means), means.shape) == (2, 2)


def test_fz0_differences_are_homogeneous_of_degree_zero() -> None:
    rng = np.random.default_rng(19)
    y = rng.standard_normal(50)
    va, ea, vb, eb = -1.5, -2.0, -1.8, -2.4
    diff = fz0_loss(y, va, ea, level=0.05) - fz0_loss(y, vb, eb, level=0.05)
    for c in (0.01, 3.0, 250.0):
        scaled = fz0_loss(c * y, c * va, c * ea, level=0.05) - fz0_loss(
            c * y, c * vb, c * eb, level=0.05
        )
        np.testing.assert_allclose(scaled, diff, rtol=1e-12, atol=1e-12)


def test_fz0_invalid_inputs_are_nan_with_warning() -> None:
    with pytest.warns(UserWarning, match="e <= v < 0"):
        out = fz0_loss(
            np.zeros(3),
            np.array([-1.0, 1.0, -1.0]),
            np.array([-2.0, -2.0, -0.5]),
            level=0.05,
        )
    assert np.isfinite(out[0]) and np.isnan(out[1:]).all()


def test_qlike_is_robust_to_proxy_noise_where_mse_log_is_not() -> None:
    rng = np.random.default_rng(20)
    h_true = np.exp(rng.standard_normal(200_000) * 0.3) * 1e-4
    proxy = h_true * rng.standard_normal(h_true.size) ** 2  # unbiased, very noisy
    scales = np.array([0.6, 0.8, 1.0, 1.25, 1.5])
    qlike = [np.mean(qlike_loss(proxy, s * h_true)) for s in scales]
    assert int(np.argmin(qlike)) == 2  # the true variance wins
    ok = proxy > 0
    mse_log = [
        np.mean((np.log(proxy[ok]) - np.log(s * h_true[ok])) ** 2) for s in scales
    ]
    assert int(np.argmin(mse_log)) == 0  # biased towards under-prediction
    assert float(qlike_loss(2.0, 2.0)) == 0.0
    with pytest.warns(UserWarning, match="non-positive"):
        bad = qlike_loss(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert np.isnan(bad[0]) and bad[1] == 0.0


# --------------------------------------------------------------------------- #
# var_backtest
# --------------------------------------------------------------------------- #
def test_var_convention_is_required_and_checked() -> None:
    r = np.random.default_rng(21).standard_normal(300) * 0.01
    with pytest.raises(TypeError, match="var_convention"):
        var_backtest(r, np.full(300, 0.0233), level=0.01)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="did you mean var_convention='quantile'"):
        var_backtest(r, np.full(300, -0.0233), level=0.01, var_convention="loss")
    with pytest.raises(ValueError, match="need `es`"):
        var_backtest(
            r, np.full(300, 0.0233), level=0.01, var_convention="loss", tests=("z2",)
        )
    with pytest.raises(ValueError, match="unknown tests"):
        var_backtest(
            r, np.full(300, 0.0233), level=0.01, var_convention="loss", tests=("basel",)
        )


def test_var_backtest_table() -> None:
    sig, r = _normal_case(500, 2, 22)
    spec = PredictiveSpec("normal", loc=np.zeros(500), scale=sig)
    table = var_backtest(
        r,
        -sig * Q_ES,
        -sig * ES_STD,
        level=ALPHA_ES,
        var_convention="loss",
        predictive=spec,
        names=["a", "b"],
    )
    assert table.schema == pl.Schema(EVALUATION_SCHEMA)
    assert table["test"].to_list() == [
        "kupiec",
        "kupiec",
        "christoffersen_cc",
        "christoffersen_cc",
        "dynamic_quantile",
        "dynamic_quantile",
        "acerbi_szekely_z2",
        "acerbi_szekely_z2",
    ]
    assert table["pvalue"].is_not_null().all()
    quantile = var_backtest(
        r,
        sig * Q_ES,
        sig * ES_STD,
        level=ALPHA_ES,
        var_convention="quantile",
        predictive=spec,
        names=["a", "b"],
    )
    assert table.equals(quantile)
    no_es = var_backtest(r, -sig * Q_ES, level=ALPHA_ES, var_convention="loss")
    assert "acerbi_szekely_z2" not in no_es["test"].to_list()


def test_contemporaneous_var_symptom_is_a_low_hit_rate() -> None:
    """Trap T1: a VaR that saw y_t hides its own exceptions."""
    rng = np.random.default_rng(23)
    r = rng.standard_normal(1000) * 0.01
    honest = np.full(1000, 0.0233)
    leaky = np.maximum(honest, -r + 1e-9)  # "knows" today's loss
    res = kupiec_test(
        exceedances(r, lower=-np.column_stack([honest, leaky])), level=0.01
    )
    assert res.estimate[1] == 0.0 < res.estimate[0]
    assert res.pvalue[1] < 0.01


# --------------------------------------------------------------------------- #
# Leak safety of the row-wise outputs
# --------------------------------------------------------------------------- #
def _panel() -> pl.DataFrame:
    rng = np.random.default_rng(24)
    n, t = 3, 40
    return pl.DataFrame(
        {
            "entity": np.repeat(["a", "b", "c"], t),
            "time": np.tile(np.arange(t), n),
            "ret": rng.standard_normal(n * t) * 0.01,
            "var": np.full(n * t, -0.02),
            "es": np.full(n * t, -0.025),
            "rv": np.exp(rng.standard_normal(n * t)) * 1e-4,
            "h": np.full(n * t, 1e-4),
        }
    )


def _rowwise_op(frame: pl.DataFrame) -> pl.DataFrame:
    hit = exceedances(frame["ret"].to_numpy(), lower=frame["var"].to_numpy())[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fz = fz0_loss(
            frame["ret"].to_numpy(),
            frame["var"].to_numpy(),
            frame["es"].to_numpy(),
            level=0.05,
        )
        ql = qlike_loss(frame["rv"].to_numpy(), frame["h"].to_numpy())
    return frame.select("entity", "time").with_columns(
        pl.Series("hit", hit), pl.Series("fz0", fz), pl.Series("qlike", ql)
    )


def test_rowwise_outputs_are_leak_safe() -> None:
    panel = _panel()
    assert_prefix_invariant(_rowwise_op, panel, entity="entity", time="time")
    assert_no_lookahead(_rowwise_op, panel, entity="entity", time="time")


def test_dq_asymptotic_result_carries_the_oversize_caveat() -> None:
    rng = np.random.default_rng(25)
    hits = (rng.random((300, 2)) < 0.05).astype(float)
    res = dynamic_quantile_test(hits, rng.standard_normal((300, 2)), level=0.05)
    assert any("oversized" in w for w in res.warnings)


@pytest.mark.slow
def test_dq_sizes_at_t250() -> None:
    """Asymptotic DQ over-rejects with rare exceptions; the MC null does not."""
    reps, t, alpha = 1000, 250, 0.01
    rng = np.random.default_rng(26)
    hits = (rng.random((t, reps)) < alpha).astype(float)
    var = rng.standard_normal((t, reps))
    res = dynamic_quantile_test(hits, var, level=alpha, null="mc", n_sims=199)
    assert float(np.mean(res.pvalue <= 0.05)) <= 0.05 + 3 * _mc_se(0.05, reps)
    assert float(np.mean(res.details["pvalue_asymptotic"] <= 0.05)) > 0.07
