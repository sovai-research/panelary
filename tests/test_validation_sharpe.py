"""Sharpe-ratio inference: closed forms, parity, bootstrap mechanics, API contract."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import polars as pl
import pytest

from panelary.validation import (
    EvaluationResult,
    _hac,
    annualize_sharpe,
    evaluation_table,
    sharpe_block_length,
    sharpe_ratio_inference,
    sharpe_ratio_test,
)
from panelary.validation._selection_stats import _sharpe_estimator_std
from panelary.validation._sharpe import (
    _influence,
    _moments,
    _nct_cdf,
    _vector_hac_variance,
)


def _pair(
    seed: int = 0, n: int = 240, rho: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = 0.01 + 0.04 * z[:, 0]
    y = 0.008 + 0.04 * (rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1])
    return x, y


def _skew_kurt(r: np.ndarray) -> tuple[float, float]:
    c = r - r.mean()
    s = math.sqrt(np.mean(c * c))
    z = c / s
    return float(np.mean(z**3)), float(np.mean(z**4))


# --------------------------------------------------------------------------- #
# Single Sharpe
# --------------------------------------------------------------------------- #
def test_iid_equals_sharpe_estimator_std() -> None:
    r = np.random.default_rng(1).standard_t(5, 500) * 0.02 + 0.003
    res = sharpe_ratio_inference(r, method="iid", small_sample="t-1")
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = _skew_kurt(r)
    assert res.estimate[0] == pytest.approx(sr, rel=1e-14)
    assert res.std_error[0] == pytest.approx(
        _sharpe_estimator_std(sr, 500, g3, g4), rel=1e-14
    )
    assert res.details["skewness"][0] == pytest.approx(g3, rel=1e-12)


def test_iid_normal_is_lo_and_pvalue_is_normal_tail() -> None:
    r = np.random.default_rng(2).standard_normal(250) * 0.01 + 0.001
    res = sharpe_ratio_inference(
        r, method="iid_normal", benchmark_sharpe=0.02, alternative="greater"
    )
    sr = r.mean() / r.std(ddof=1)
    se = math.sqrt((1 + sr * sr / 2) / 250)
    z = (sr - 0.02) / se
    assert res.std_error[0] == pytest.approx(se, rel=1e-14)
    assert res.statistic[0] == pytest.approx(z, rel=1e-13)
    assert res.pvalue[0] == pytest.approx(0.5 * math.erfc(z / math.sqrt(2)), rel=1e-12)
    zq = 1.959963984540054
    np.testing.assert_allclose(res.ci[0], [sr - zq * se, sr + zq * se], rtol=1e-12)
    assert res.reference == "N(0,1)"


def test_hac_bartlett_is_lo_non_iid_standard_error() -> None:
    x = _pair(3)[0]
    res = sharpe_ratio_inference(x, method="hac", hac="bartlett", hac_lags=4)
    mom = _moments(x[:, None])
    psi = _influence(mom, "sharpe")[:, 0]
    from panelary.validation import newey_west_variance

    assert res.std_error[0] == pytest.approx(
        math.sqrt(newey_west_variance(psi, lags=4) / x.size), rel=1e-12
    )
    assert res.hac == "bartlett(L=4)"


def test_influence_variance_is_mertens_formula() -> None:
    r = np.random.default_rng(4).standard_t(6, 400) + 0.2
    mom = _moments(r[:, None])
    psi = _influence(mom, "sharpe")[:, 0]
    g3, g4 = _skew_kurt(r)
    sr0 = float(mom.sr0[0])
    assert np.mean(psi * psi) == pytest.approx(
        1 - g3 * sr0 + (g4 - 1) / 4 * sr0**2, rel=1e-12
    )


def test_exact_normal_matches_scipy_nct_inversion() -> None:
    stats = pytest.importorskip("scipy.stats")
    optimize = pytest.importorskip("scipy.optimize")
    r = np.random.default_rng(5).standard_normal(60) * 0.03 + 0.008
    res = sharpe_ratio_inference(
        r, method="exact_normal", benchmark_sharpe=0.05, confidence=0.9
    )
    n = r.size
    t_obs = math.sqrt(n) * r.mean() / r.std(ddof=1)
    assert res.statistic[0] == pytest.approx(t_obs, rel=1e-14)
    assert res.pvalue[0] == pytest.approx(
        stats.nct.sf(t_obs, n - 1, math.sqrt(n) * 0.05), abs=1e-10
    )
    lo = optimize.brentq(
        lambda d: stats.nct.cdf(t_obs, n - 1, d) - 0.95, -20, 20, xtol=1e-14
    )
    hi = optimize.brentq(
        lambda d: stats.nct.cdf(t_obs, n - 1, d) - 0.05, -20, 20, xtol=1e-14
    )
    np.testing.assert_allclose(
        res.ci[0], [lo / math.sqrt(n), hi / math.sqrt(n)], rtol=0, atol=1e-8
    )
    assert "i.i.d. normal" in res.reference


@pytest.mark.parametrize(
    ("t", "df", "delta"),
    [(2.3, 7, 1.1), (-1.4, 30, 0.4), (45.0, 4999, 40.0), (0.0, 12, -2.0)],
)
def test_nct_cdf_matches_scipy(t: float, df: float, delta: float) -> None:
    stats = pytest.importorskip("scipy.stats")
    assert _nct_cdf(t, df, delta) == pytest.approx(
        stats.nct.cdf(t, df, delta), abs=1e-12
    )


def test_exact_normal_limits_models() -> None:
    with pytest.raises(ValueError, match="limited to 50"):
        sharpe_ratio_inference(np.ones((20, 51)), method="exact_normal")


def test_single_bootstrap_contract() -> None:
    x = _pair(6, n=300)[0]
    res = sharpe_ratio_inference(
        x, method="bootstrap", n_boot=499, block_length=4, seed=3
    )
    assert res.reference == "cbb-studentized(b=4,B=499)"
    assert res.block_length == 4 and res.n_resamples == 499 and res.seed == 3
    assert 1 / 500 <= res.pvalue[0] <= 1
    assert res.ci[0, 0] < res.estimate[0] < res.ci[0, 1]
    again = sharpe_ratio_inference(
        x, method="bootstrap", n_boot=499, block_length=4, seed=3
    )
    np.testing.assert_array_equal(res.pvalue, again.pvalue)


def test_single_calibrated_bootstrap_runs() -> None:
    x = _pair(7, n=120)[0]
    res = sharpe_ratio_inference(
        x, method="bootstrap", block_length="calibrate", n_boot=199
    )
    assert res.block_length in (1, 2, 4, 6, 8, 10)


def test_degenerate_series_give_nan_and_warning() -> None:
    r = np.column_stack(
        [np.full(50, 0.01), np.random.default_rng(0).standard_normal(50)]
    )
    res = sharpe_ratio_inference(r, method="hac")
    assert np.isnan(res.estimate[0]) and np.isnan(res.pvalue[0])
    assert np.isfinite(res.pvalue[1])
    assert any("zero-variance" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# Differences
# --------------------------------------------------------------------------- #
def test_jkm_equals_memmel_closed_form_and_warns() -> None:
    x, y = _pair(8)
    res = sharpe_ratio_test(x, y, method="jkm")
    n = x.size
    s1, s2 = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    rho = float(np.corrcoef(x, y)[0, 1])
    theta = (2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / n
    z = (s1 - s2) / math.sqrt(theta)
    assert res.statistic[0] == pytest.approx(z, rel=1e-12)
    assert res.pvalue[0] == pytest.approx(math.erfc(abs(z) / math.sqrt(2)), rel=1e-12)
    assert any("10.7%" in w for w in res.warnings)
    assert res.hac is None


def test_scalar_influence_path_equals_lw_vector_without_prewhitening() -> None:
    x, y = _pair(9, n=200)
    y = y + 0.3 * np.roll(y, 1)
    mx, my = _moments(x[:, None]), _moments(y[:, None])
    psi = (_influence(mx, "sharpe") - _influence(my, "sharpe"))[:, 0]
    a, b = x.mean(), y.mean()
    c, d = np.mean(x * x), np.mean(y * y)
    vec = np.column_stack([x - a, y - b, x * x - c, y * y - d])
    s_band = 2.7
    psi4, _ = _hac.vector_lrv_prewhitened(
        vec, bandwidth=s_band, prewhiten=False, small_sample=False
    )
    va, vb = c - a * a, d - b * b
    grad = np.array([c / va**1.5, -d / vb**1.5, -0.5 * a / va**1.5, 0.5 * b / vb**1.5])
    scalar = float(_hac.kernel_lrv(psi, s_band, kernel="qs"))
    assert float(grad @ psi4 @ grad) == pytest.approx(scalar, rel=1e-12)


def test_hac_vector_option_is_lw_recipe() -> None:
    x, y = _pair(10)
    res = sharpe_ratio_test(x, y, method="hac", hac="qs_pw_vector")
    var = _vector_hac_variance(x[:, None], y, "sharpe")[0]
    assert res.std_error[0] == pytest.approx(math.sqrt(var / x.size), rel=1e-14)
    assert res.hac == "qs-pw-vector(T/(T-4))"
    scalar = sharpe_ratio_test(x, y, method="hac")
    # two consistent estimators of the same quantity
    assert scalar.std_error[0] == pytest.approx(res.std_error[0], rel=0.25)


def test_log_variance_is_delta_method_hand_value() -> None:
    x, y = _pair(11)
    res = sharpe_ratio_test(x, y, statistic="log_variance", method="iid")
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    d = (zx**2 - 1) - (zy**2 - 1)
    se = math.sqrt(np.var(d) / x.size)
    est = math.log(np.var(x, ddof=1) / np.var(y, ddof=1))
    assert res.estimate[0] == pytest.approx(est, rel=1e-12)
    assert res.std_error[0] == pytest.approx(se, rel=1e-12)
    assert res.test == "log_variance_difference"


def test_bootstrap_difference_reference_and_ci() -> None:
    x, y = _pair(12, n=300)
    res = sharpe_ratio_test(x, y, n_boot=999, block_length=5, seed=1)
    assert res.reference == "cbb-studentized(b=5,B=999)"
    assert res.ci[0, 0] < res.estimate[0] < res.ci[0, 1]
    assert res.estimate[0] == pytest.approx(
        x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1), rel=1e-12
    )
    # the CI excludes 0 exactly when the two-sided p-value is below alpha (up to
    # the discreteness of the bootstrap quantile)
    excl = res.ci[0, 0] > 0 or res.ci[0, 1] < 0
    assert excl == (res.pvalue[0] < 0.05) or abs(res.pvalue[0] - 0.05) < 0.01


def test_one_sided_alternatives_are_consistent() -> None:
    x, y = _pair(13, n=400)
    x = x + 0.004
    g = sharpe_ratio_test(x, y, alternative="greater", n_boot=999, block_length=3)
    lo = sharpe_ratio_test(x, y, alternative="less", n_boot=999, block_length=3)
    assert g.pvalue[0] < 0.5 < lo.pvalue[0]


def test_adding_a_strategy_leaves_others_bitwise_unchanged() -> None:
    rng = np.random.default_rng(14)
    x = rng.standard_normal((300, 5)) * 0.02 + 0.002
    y = rng.standard_normal(300) * 0.02 + 0.001
    full = sharpe_ratio_test(x, y, block_length=5, n_boot=499)
    part = sharpe_ratio_test(x[:, [1, 3]], y, block_length=5, n_boot=499)
    single = sharpe_ratio_test(x[:, [3]], y, block_length=5, n_boot=499)
    for field in ("estimate", "statistic", "pvalue", "std_error"):
        assert (
            getattr(full, field)[3]
            == getattr(part, field)[1]
            == getattr(single, field)[0]
        )
    # The interval comes from a bootstrap quantile of a gemm product, which some
    # BLAS builds (CI's py3.12 OpenBLAS) round differently by shape: 1 ulp.
    np.testing.assert_allclose(full.ci[3], single.ci[0], rtol=1e-12, atol=1e-15)


def test_romano_wolf_on_shared_null_controls_and_detects() -> None:
    rng = np.random.default_rng(15)
    n = 500
    y = rng.standard_normal(n) * 0.02 + 0.001
    x = 0.5 * y[:, None] + rng.standard_normal((n, 6)) * 0.02 + 0.0005
    x[:, 0] += 0.006  # one genuinely better strategy
    res = sharpe_ratio_test(x, y, alternative="greater", n_boot=999, block_length=3)
    assert res.adjustment == "romano-wolf"
    assert res.pvalue_adj is not None
    assert np.all(res.pvalue_adj >= res.pvalue - 1e-15)
    assert res.pvalue_adj[0] < 0.05
    assert np.all(res.pvalue_adj[1:] > 0.05)


def test_multiple_fallback_and_alternatives() -> None:
    rng = np.random.default_rng(16)
    x = rng.standard_normal((200, 3)) * 0.02
    y = rng.standard_normal(200) * 0.02
    res = sharpe_ratio_test(x, y, method="hac")
    assert res.adjustment == "holm"
    assert any("Holm" in w for w in res.warnings)
    for multiple, label in (("bh", "bh"), ("by", "by"), ("none", None), (None, None)):
        r = sharpe_ratio_test(x, y, method="iid", multiple=multiple)
        assert r.adjustment == label


def test_unbalanced_columns_are_grouped() -> None:
    rng = np.random.default_rng(17)
    x = rng.standard_normal((300, 4)) * 0.02
    y = rng.standard_normal(300) * 0.02
    x[:50, 2] = np.nan
    res = sharpe_ratio_test(x, y, block_length=4, n_boot=199)
    assert res.n_obs.tolist() == [300, 300, 250, 300]
    assert res.details["group"].tolist() == [0, 0, 1, 0]
    assert any("availability groups" in w for w in res.warnings)
    alone = sharpe_ratio_test(x[50:, 2], y[50:], block_length=4, n_boot=199)
    assert res.estimate[2] == alone.estimate[0]
    assert res.pvalue[2] == alone.pvalue[0]
    assert res.pvalue_adj is not None and np.isnan(res.pvalue_adj[2])


def test_boundaries_are_mapped_and_respected() -> None:
    x, y = _pair(18, n=240)
    res = sharpe_ratio_test(x, y, block_length=8, n_boot=199, boundaries=[80, 160])
    assert res.block_length == 8
    clipped = sharpe_ratio_test(x, y, block_length=100, n_boot=99, boundaries=[80, 160])
    assert clipped.block_length == 80
    assert any("clipped" in w for w in clipped.warnings)


def test_calibrated_block_length_is_recorded() -> None:
    x, y = _pair(19, n=120)
    res = sharpe_ratio_test(
        x, y, block_length="calibrate", calibration_sims=200, n_boot=299
    )
    grid = res.details["calibration_grid"][0]
    cov = res.details["calibration_coverage"][0]
    assert grid.tolist() == [1, 2, 4, 6, 8, 10]
    assert res.block_length == int(grid[np.argmin(np.abs(cov - 0.95))])


def test_sharpe_block_length_api() -> None:
    x, y = _pair(20, n=150)
    cal = sharpe_block_length(x, y, n_sims=100, n_boot=199, grid=(1, 3, 6))
    assert cal.grid.tolist() == [1, 3, 6]
    assert cal.block_length in (1, 3, 6)
    frame = cal.to_frame()
    assert frame["selected"].sum() == 1
    assert np.all((cal.coverage >= 0) & (cal.coverage <= 1))
    single = sharpe_block_length(x, n_sims=50, n_boot=99)
    assert single.statistic == "sharpe"
    with pytest.raises(ValueError, match="needs a `benchmark`"):
        sharpe_block_length(x, statistic="log_variance")


def test_polars_input_names_and_evidence_table() -> None:
    x, y = _pair(21)
    frame = pl.DataFrame({"alpha": x, "beta": x * 0.5 + y})
    res = sharpe_ratio_test(frame, pl.Series("bench", y), method="hac")
    assert res.names == ("alpha", "beta")
    single = sharpe_ratio_inference(frame, method="iid")
    table = evaluation_table([res, single])
    assert table.height == 4
    json.dumps(res.to_dict())
    assert isinstance(res["beta"], EvaluationResult)


def test_input_validation() -> None:
    x, y = _pair(22)
    with pytest.raises(ValueError, match="unknown `method`"):
        sharpe_ratio_test(x, y, method="nope")
    with pytest.raises(ValueError, match="unknown `statistic`"):
        sharpe_ratio_test(x, y, statistic="nope")
    with pytest.raises(ValueError, match="rows"):
        sharpe_ratio_test(x, y[:-1])
    with pytest.raises(ValueError, match="alternative"):
        sharpe_ratio_inference(x, alternative="bigger")


# --------------------------------------------------------------------------- #
# Lo annualisation
# --------------------------------------------------------------------------- #
def test_annualize_sharpe_hand_values() -> None:
    # Lo (2002): eta(q) = q / sqrt(q + 2 sum_{k<q} (q - k) rho_k)
    assert annualize_sharpe(0.1, 12, autocorrelations=[0.0]) == pytest.approx(
        0.1 * math.sqrt(12)
    )
    rho = 0.2
    q = 12
    denom = q + 2 * sum((q - k) * rho**k for k in range(1, q))
    got = annualize_sharpe(0.1, q, autocorrelations=[rho**k for k in range(1, q)])
    assert got == pytest.approx(0.1 * q / math.sqrt(denom), rel=1e-14)
    # positive autocorrelation shrinks the annualised Sharpe below sqrt(q)
    assert got < 0.1 * math.sqrt(q)
    # lags beyond q-1 are ignored; missing lags are zero
    assert annualize_sharpe(0.1, 2, autocorrelations=[0.1, 0.9]) == pytest.approx(
        0.1 * 2 / math.sqrt(2 + 2 * 0.1)
    )
    arr = annualize_sharpe(
        np.array([0.1, 0.2]), 4, autocorrelations=np.array([[0.1, -0.1]])
    )
    assert arr.shape == (2,)
    with pytest.warns(UserWarning, match="serially uncorrelated"):
        annualize_sharpe(0.1, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        annualize_sharpe(0.1, 12, autocorrelations=[0.05])
    assert math.isnan(annualize_sharpe(0.1, 3, autocorrelations=[-0.99, -0.99]))
