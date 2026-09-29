"""Luck versus skill: Storey pi0, adaptive BH, BSW, the alpha bootstrap, HLZ hurdles."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary.evolve import (
    cross_sectional_bootstrap,
    haircut_sharpe_ratio,
    significance_hurdle,
)
from panelary.evolve._honest import _hlz_family_pvalues
from panelary.validation import (
    AlphaBootstrapResult,
    EvaluationResult,
    Pi0Estimate,
    alpha_bootstrap,
    benjamini_hochberg,
    benjamini_yekutieli,
    block_bootstrap_indices,
    holm_bonferroni,
    luck_versus_skill,
    storey_pi0,
)
from panelary.validation._luck_skill import (
    _date_indices,
    _gather_ols,
    _joint_dates_stats,
    _spd_solve,
)
from panelary.validation._resample import count_matrix


def _mixture(seed: int, m: int = 5000, pi0: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n0 = int(m * pi0)
    return np.concatenate([rng.random(n0), rng.beta(0.1, 8.0, m - n0)])


# --------------------------------------------------------------------------- #
# Storey pi0 and adaptive BH
# --------------------------------------------------------------------------- #
def test_storey_pi0_recovers_null_share() -> None:
    est = storey_pi0(_mixture(0))
    assert isinstance(est, Pi0Estimate)
    assert abs(est.pi0 - 0.8) < 0.05
    assert est.lambda_ in est.lambdas
    assert est.to_frame()["selected"].sum() == 1
    assert float(est) == est.pi0
    assert storey_pi0(np.random.default_rng(1).random(2000)).pi0 == pytest.approx(
        1.0, abs=0.06
    )


def test_storey_bootstrap_matches_a_replicate_loop() -> None:
    p = _mixture(2, m=300)
    lam = np.round(np.arange(0.05, 0.96, 0.05), 2)
    est = storey_pi0(p, n_boot=50, seed=3)
    pi0_lam = np.array([(p > lv).sum() / (p.size * (1 - lv)) for lv in lam])
    np.testing.assert_allclose(est.pi0_lambda, pi0_lam, rtol=1e-14)
    idx = np.random.default_rng(3).integers(0, p.size, size=(50, p.size))
    boot = np.array(
        [[(p[i] > lv).sum() / (p.size * (1 - lv)) for lv in lam] for i in idx]
    )
    mse = np.mean((boot - pi0_lam.min()) ** 2, axis=0)
    np.testing.assert_allclose(est.mse, mse, rtol=1e-12)
    assert est.pi0 == pytest.approx(min(1.0, pi0_lam[np.argmin(mse)]))


def _bh_reference(p: np.ndarray, alpha: float) -> np.ndarray:
    """The pre-``pi0`` implementation, verbatim."""
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=float)
    scaled = p[order] * m * 1.0 / ranks
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj


def test_benjamini_hochberg_default_is_bitwise_unchanged() -> None:
    p = _mixture(4, m=400)
    res = benjamini_hochberg(p, alpha=0.1)
    np.testing.assert_array_equal(res.adjusted_pvalues, _bh_reference(p, 0.1))
    assert res.method == "benjamini-hochberg"


def test_adaptive_bh_q_values() -> None:
    p = _mixture(5, m=1000)
    bh = benjamini_hochberg(p)
    est = storey_pi0(p)
    q = benjamini_hochberg(p, pi0=est)
    assert np.all(q.adjusted_pvalues <= bh.adjusted_pvalues)
    assert q.n_rejected >= bh.n_rejected
    assert q.method.startswith("benjamini-hochberg(pi0=")
    np.testing.assert_array_equal(
        benjamini_hochberg(p, pi0=1.0).adjusted_pvalues, bh.adjusted_pvalues
    )
    with pytest.raises(ValueError, match="pi0"):
        benjamini_hochberg(p, pi0=0.0)


# --------------------------------------------------------------------------- #
# Barras-Scaillet-Wermers
# --------------------------------------------------------------------------- #
def test_luck_versus_skill_identities_and_recovery() -> None:
    rng = np.random.default_rng(6)
    t = np.concatenate(
        [rng.standard_normal(1400), rng.normal(3.5, 1, 400), rng.normal(-3.5, 1, 200)]
    )
    frame = luck_versus_skill(t, pi0=0.7)
    col = {c: frame[c].to_numpy() for c in frame.columns}
    np.testing.assert_allclose(col["t_plus"] + col["f_plus"], col["s_plus"], rtol=1e-14)
    np.testing.assert_allclose(
        col["t_minus"] + col["f_minus"], col["s_minus"], rtol=1e-14
    )
    np.testing.assert_allclose(col["f_plus"], 0.7 * col["gamma"] / 2)
    assert frame["pi_a_plus"][0] == pytest.approx(0.2, abs=0.03)
    assert frame["pi_a_minus"][0] == pytest.approx(0.1, abs=0.03)
    est = luck_versus_skill(t)  # Storey pi0
    assert est["pi0"][0] == pytest.approx(0.7, abs=0.06)
    dof = luck_versus_skill(t, pi0=0.7, dof=np.full(t.size, 30.0))
    # fatter tails, fewer rejections
    assert np.all(dof["s_plus"].to_numpy() <= col["s_plus"])


# --------------------------------------------------------------------------- #
# Alpha bootstrap
# --------------------------------------------------------------------------- #
def test_date_indices_fast_path_is_the_stationary_stream() -> None:
    for seed in (0, 11):
        np.testing.assert_array_equal(
            _date_indices(97, 1, 40, seed),
            block_bootstrap_indices(
                97, block_length=1, n_boot=40, scheme="stationary", seed=seed
            ),
        )


def test_spd_solve_matches_numpy() -> None:
    rng = np.random.default_rng(7)
    a = rng.standard_normal((50, 3, 4, 4))
    g = a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(4)
    r = rng.standard_normal((50, 3, 4, 2))
    x, ratio = _spd_solve(g, r)
    np.testing.assert_allclose(x, np.linalg.solve(g, r), rtol=1e-10, atol=1e-12)
    det = np.linalg.det(g) / np.prod(np.diagonal(g, axis1=-2, axis2=-1), axis=-1)
    np.testing.assert_allclose(ratio, det, rtol=1e-10)


@pytest.mark.parametrize("unbalanced", [False, True])
def test_count_matrix_regressions_match_gather_lstsq(unbalanced: bool) -> None:
    rng = np.random.default_rng(8)
    t_len, n_f = 80, 6
    fac = rng.standard_normal((t_len, 2))
    x = np.column_stack([np.ones(t_len), fac])
    r = (
        fac @ rng.standard_normal((2, n_f))
        + 0.01
        + rng.standard_normal((t_len, n_f)) * 0.5
    )
    if unbalanced:
        r[:20, 1] = np.nan
        r[50:, 4] = np.nan
    avail = np.isfinite(r).astype(float)
    y = np.where(avail > 0, r, 0.0)
    idx = rng.integers(0, t_len, size=(30, t_len))
    _, tb, n, ill = _joint_dates_stats(count_matrix(idx, t_len), avail, x, y, 10)
    assert not ill.any()
    for b in range(30):
        for i in range(n_f):
            rows = idx[b][avail[idx[b], i] > 0]
            a_ref, t_ref = _gather_ols(x, r[:, i], rows, 10)
            assert tb[b, i] == pytest.approx(t_ref, rel=1e-9)
            assert n[b, i] == rows.size


def test_raw_means_reproduce_cross_sectional_bootstrap() -> None:
    rng = np.random.default_rng(9)
    r = rng.standard_normal((120, 40)) * 0.02 + 0.001
    r[:, :3] += 0.01
    ref = cross_sectional_bootstrap(
        r, n_boot=300, quantiles=(0.90, 0.95, 0.99, 1.00), seed=4
    )
    res = alpha_bootstrap(
        r, n_boot=300, quantiles=(0.90, 0.95, 0.99, 1.00), min_obs=2, seed=4
    )
    tab = res.quantiles
    for k, tag in enumerate(("q90", "q95", "q99", "q100")):
        assert tab["observed"][k] == pytest.approx(ref[f"t_{tag}_observed"], rel=1e-12)
        assert tab["null_median"][k] == pytest.approx(
            ref[f"t_{tag}_null_median"], rel=1e-12
        )
        assert tab["null_p95"][k] == pytest.approx(ref[f"t_{tag}_null_p95"], rel=1e-12)
        assert tab["pvalue"][k] == ref[f"t_{tag}_pvalue"]


def test_alpha_bootstrap_contract_and_power() -> None:
    rng = np.random.default_rng(10)
    t_len, n_f = 240, 60
    fac = rng.standard_normal((t_len, 1)) * 0.04
    r = fac @ rng.uniform(0.5, 1.5, (1, n_f)) + rng.standard_normal((t_len, n_f)) * 0.02
    r[:, :5] += 0.006  # five skilled funds (t ~ 3.7)
    r[:100, 7] = np.nan
    res = alpha_bootstrap(r, fac, n_boot=499, seed=1)
    assert isinstance(res, AlphaBootstrapResult)
    assert res.factors == 1 and res.n_obs[7] == 140
    assert np.all(res.pvalue[:5] < 0.05)
    assert res.quantiles.filter(res.quantiles["quantile"] == 0.99)["pvalue"][0] < 0.05
    ev = res.to_evaluation()
    assert isinstance(ev, EvaluationResult) and ev.reference.startswith(
        "ff2010-joint-dates"
    )
    assert res.to_frame().height == n_f


def test_alpha_bootstrap_rejects_too_few_observations() -> None:
    r = np.random.default_rng(11).standard_normal((30, 4))
    with pytest.raises(ValueError, match="no fund"):
        alpha_bootstrap(r, n_boot=19, min_obs=40)


def test_residual_scheme_is_per_fund_and_order_free() -> None:
    rng = np.random.default_rng(12)
    fac = rng.standard_normal((150, 1))
    r = fac @ rng.uniform(0.5, 1.5, (1, 5)) + rng.standard_normal((150, 5))
    names = [f"f{i}" for i in range(5)]
    full = alpha_bootstrap(r, fac, scheme="residual", n_boot=199, names=names)
    part = alpha_bootstrap(
        r[:, [3, 1]], fac, scheme="residual", n_boot=199, names=["f3", "f1"]
    )
    assert full.pvalue[3] == part.pvalue[0] and full.pvalue[1] == part.pvalue[1]
    assert full.to_evaluation().reference.startswith("ktww-residual")


@pytest.mark.slow
def test_ktww_oversized_under_common_factor_while_ff_is_sized() -> None:
    """A common shock missing from the factor model: the cross-section of t(alpha)
    is correlated. Per-fund residual resampling (KTWW) ignores that and its null
    for an extreme quantile is too narrow; joint-date resampling (FF2010) keeps it."""
    rng = np.random.default_rng(13)
    reps, t_len, n_f = 150, 120, 150
    rej = {"joint_dates": 0, "residual": 0}
    for _ in range(reps):
        f = rng.standard_normal((t_len, 1))
        g = rng.standard_normal((t_len, 1))  # omitted common factor
        r = f @ rng.uniform(0.5, 1.5, (1, n_f)) + g @ rng.uniform(0.6, 1.0, (1, n_f))
        r = r + rng.standard_normal((t_len, n_f))
        for scheme in rej:
            res = alpha_bootstrap(
                r,
                f,
                scheme=scheme,
                n_boot=199,
                quantiles=(0.95,),
                seed=int(rng.integers(1 << 30)),
            )
            rej[scheme] += bool(res.quantiles["pvalue"][0] < 0.05)
    ff, ktww = rej["joint_dates"] / reps, rej["residual"] / reps
    assert ff <= 0.10, ff
    # measured: FF 4.0 %, KTWW 16.7 % at nominal 5 % (150 replications)
    assert ktww > 0.10 and ktww > 2 * ff, (ktww, ff)


# --------------------------------------------------------------------------- #
# HLZ significance hurdle
# --------------------------------------------------------------------------- #
def test_hurdle_bonferroni_closed_form() -> None:
    for m in (1, 10, 316, 10_000):
        want = -_ppf(0.05 / (2 * m))
        assert significance_hurdle(m, method="bonferroni") == pytest.approx(
            want, rel=1e-12
        )
    # HLZ (2016): 316 factors -> Bonferroni hurdle 3.78
    assert significance_hurdle(316, method="bonferroni") == pytest.approx(
        3.78, abs=0.005
    )


def _ppf(p: float) -> float:
    from panelary._internal._special import norm_ppf

    return float(norm_ppf(p))


def test_hurdle_families_are_the_haircut_families() -> None:
    """Same draw as haircut_sharpe_ratio, so the O(1) adjusted-p formulas can be
    checked against the full Holm/BHY procedures on identical families."""
    h = haircut_sharpe_ratio(0.25, 60, n_sim=200, seed=4)
    fam = _hlz_family_pvalues(60, 0.2, 200, 4)
    p_obs = h["p_value"]
    holm = np.median(
        [holm_bonferroni(np.r_[p_obs, f]).adjusted_pvalues[0] for f in fam]
    )
    bhy = np.median(
        [benjamini_yekutieli(np.r_[p_obs, f]).adjusted_pvalues[0] for f in fam]
    )
    assert holm == h["holm_pvalue"] and bhy == h["bhy_pvalue"]


@pytest.mark.parametrize("method", ["holm", "bhy"])
def test_hurdle_is_where_the_median_adjusted_p_crosses_alpha(method: str) -> None:
    m, n_sim = 50, 300
    t_star = significance_hurdle(m, method=method, n_sim=n_sim, seed=2)
    fam = _hlz_family_pvalues(m - 1, 0.2, n_sim, 2)
    fn = holm_bonferroni if method == "holm" else benjamini_yekutieli

    def med(t: float) -> float:
        p0 = math.erfc(t / math.sqrt(2))
        return float(np.median([fn(np.r_[p0, f]).adjusted_pvalues[0] for f in fam]))

    assert med(t_star + 1e-6) <= 0.05 < med(t_star - 1e-3)


def test_hurdle_ordering_and_monotonicity() -> None:
    ms = (10, 100, 1000)
    bonf = [significance_hurdle(m, method="bonferroni") for m in ms]
    holm = [significance_hurdle(m, method="holm", n_sim=500) for m in ms]
    bhy = [significance_hurdle(m, method="bhy", n_sim=500) for m in ms]
    assert all(h <= b + 1e-12 for h, b in zip(holm, bonf, strict=True))
    for seq in (bonf, holm, bhy):
        assert seq[0] < seq[1] < seq[2]
    # HLZ's structural M at rho = 0.2 (1,377 tests): the BHY hurdle is near 3
    assert 2.5 < significance_hurdle(1377, method="bhy", n_sim=500) < 3.8
