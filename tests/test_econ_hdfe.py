"""Tests for high-dimensional fixed effects (``panelary.econ._hdfe``).

The acceptance criterion for M4 is exactness: absorbing fixed effects by
alternating projections must reproduce a **dense dummy-variable OLS** to ~1e-8,
coefficients *and* standard errors. The dense reference is hand-rolled here in
numpy so the test carries no dependency on ``pyfixest`` / ``fixest``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.econ import (
    HDFEResult,
    HDFETransformer,
    chi2_sf,
    demean,
    hdfe,
    newey_west_lrv,
    t_sf,
)
from panelary.econ._common import f_sf, t_cdf

TOL = 1e-8


# --------------------------------------------------------------------------- #
# Fixtures and the dense reference implementation
# --------------------------------------------------------------------------- #
def _make_panel(
    n_entities: int = 8,
    n_times: int = 12,
    *,
    seed: int = 0,
    drop_frac: float = 0.0,
) -> pl.DataFrame:
    """A small two-way panel with known slopes and additive fixed effects."""
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_entities), n_times)
    date = np.tile(np.arange(n_times), n_entities)
    if drop_frac > 0:
        keep = rng.random(firm.size) > drop_frac
        firm, date = firm[keep], date[keep]
    n = firm.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (
        1.5 * x1
        - 0.7 * x2
        + rng.normal(size=n_entities)[firm]
        + rng.normal(size=n_times)[date]
        + 0.3 * rng.normal(size=n)
    )
    return pl.DataFrame({"firm": firm, "date": date, "x1": x1, "x2": x2, "y": y})


def _dense_design(df: pl.DataFrame, absorb: list[str], x: list[str]) -> np.ndarray:
    """``[X, 1, drop-first dummies for each absorbed dimension]``."""
    n = df.height
    parts = [np.column_stack([df[c].to_numpy() for c in x]), np.ones((n, 1))]
    for col in absorb:
        codes = df[col].to_numpy()
        levels = np.unique(codes)
        dummies = (codes[:, None] == levels[None, :]).astype(float)
        parts.append(dummies[:, 1:])  # drop first level
    return np.hstack(parts)


def _dense_ols(df: pl.DataFrame, absorb: list[str], x: list[str]):
    """Full dummy-variable OLS. Returns ``(beta, resid, xtx_inv, n_params)``."""
    w = _dense_design(df, absorb, x)
    y = df["y"].to_numpy()
    xtx_inv = np.linalg.pinv(w.T @ w)
    beta = xtx_inv @ (w.T @ y)
    return beta, y - w @ beta, xtx_inv, w.shape[1], w


# --------------------------------------------------------------------------- #
# demean
# --------------------------------------------------------------------------- #
def test_demean_one_way_is_exact_group_demeaning():
    values = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    codes = [np.array([0, 0, 1, 1, 1])]
    resid, offsets, n_iter, _ = demean(values, codes, [2])
    np.testing.assert_allclose(resid, [-1.0, 1.0, -10.0, 0.0, 10.0], atol=1e-12)
    np.testing.assert_allclose(offsets[0].ravel(), [2.0, 20.0], atol=1e-12)
    assert n_iter == 1  # a single dimension converges in one sweep


def test_demean_offsets_reproduce_the_residual():
    df = _make_panel(seed=3, drop_frac=0.2)
    mat = np.column_stack([df["y"].to_numpy(), df["x1"].to_numpy()])
    f_codes = df["firm"].to_numpy()
    d_codes = df["date"].to_numpy()
    resid, offsets, _, _ = demean(
        mat, [f_codes, d_codes], [f_codes.max() + 1, d_codes.max() + 1]
    )
    rebuilt = mat - offsets[0][f_codes] - offsets[1][d_codes]
    np.testing.assert_allclose(rebuilt, resid, atol=1e-12)


def test_demean_no_dimensions_is_identity():
    values = np.arange(5.0)
    resid, offsets, n_iter, _ = demean(values, [], [])
    np.testing.assert_allclose(resid, values)
    assert offsets == [] and n_iter == 0


# --------------------------------------------------------------------------- #
# B4: a level with no observation is unidentified, not "offset 0"
# --------------------------------------------------------------------------- #
def _staggered(n_units: int = 300, n_times: int = 40, *, seed: int = 0):
    """Three adoption cohorts (t = 10 / 20 / 30), no never-treated unit.

    Every unit is treated from t = 30, so the time effect of t >= 30 has no
    untreated observation: an imputation estimator fitted on the untreated
    cells cannot identify it.
    """
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_times)
    time = np.tile(np.arange(n_times), n_units)
    adopt = np.array([10, 20, 30])[unit % 3]
    treated = time >= adopt
    alpha = rng.normal(size=n_units)[unit]
    lam = np.linspace(0.0, 2.0, n_times)[time]
    effect = 0.2 * (time - adopt + 1)
    y = alpha + lam + np.where(treated, effect, 0.0) + rng.normal(size=unit.size)
    return unit, time, treated, y


def test_demean_leaves_levels_without_observations_unidentified():
    unit, time, treated, y = _staggered()
    fit = ~treated
    resid, offsets, _, _ = demean(y[fit], [unit[fit], time[fit]], [300, 40])
    empty_times = np.setdiff1d(np.arange(40), np.unique(time[fit]))
    np.testing.assert_array_equal(empty_times, np.arange(30, 40))
    assert np.isnan(offsets[1][empty_times]).all()
    assert np.isfinite(offsets[1][:30]).all() and np.isfinite(offsets[0]).all()
    # The naive imputation now fails closed on exactly the unidentified cells.
    y0_hat = offsets[0][unit[treated], 0] + offsets[1][time[treated], 0]
    np.testing.assert_array_equal(np.isnan(y0_hat), time[treated] >= 30)
    assert np.isfinite(resid).all()


def test_demean_identified_outputs_are_unchanged_by_empty_levels():
    """Bit-identical to the same problem with the empty levels coded away."""
    unit, time, treated, y = _staggered(seed=1)
    fit = ~treated
    u, t, v = unit[fit], time[fit], np.column_stack([y[fit], -2.0 * y[fit]])
    resid, offsets, n_iter, max_dev = demean(v, [u, t], [300, 40])
    present_t = np.unique(t)
    t_compact = np.searchsorted(present_t, t)
    ref_resid, ref_off, ref_iter, ref_dev = demean(
        v, [u, t_compact], [300, present_t.size]
    )
    np.testing.assert_array_equal(resid, ref_resid)
    np.testing.assert_array_equal(offsets[0], ref_off[0])
    np.testing.assert_array_equal(offsets[1][present_t], ref_off[1])
    assert (n_iter, max_dev) == (ref_iter, ref_dev)


# --------------------------------------------------------------------------- #
# Acceptance: exact match against dense dummy OLS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("drop_frac", [0.0, 0.25])
@pytest.mark.parametrize("absorb", [["firm"], ["firm", "date"]])
def test_hdfe_matches_dense_dummy_ols(absorb, drop_frac):
    df = _make_panel(seed=11, drop_frac=drop_frac)
    res = hdfe(df, y="y", x=["x1", "x2"], absorb=absorb, entity="firm", time="date")
    beta, resid, xtx_inv, n_params, _ = _dense_ols(df, absorb, ["x1", "x2"])

    np.testing.assert_allclose(res.params, beta[:2], atol=TOL, rtol=TOL)

    df_resid = df.height - n_params
    assert res.df_resid == df_resid
    sigma2 = float(resid @ resid) / df_resid
    se_ref = np.sqrt(np.diag(sigma2 * xtx_inv)[:2])
    np.testing.assert_allclose(res.std_errors, se_ref, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(res.rss, float(resid @ resid), atol=1e-8)


def test_hdfe_three_way_absorption_matches_dense():
    df = _make_panel(n_entities=9, n_times=12, seed=5)
    df = df.with_columns((pl.col("firm") % 3).alias("sector"))
    res = hdfe(
        df,
        y="y",
        x=["x1", "x2"],
        absorb=["firm", "date", "sector"],
        entity="firm",
        time="date",
    )
    # `sector` is nested in `firm`, so the dense design is rank deficient; the
    # pseudo-inverse still identifies the slopes uniquely.
    beta, _, _, _, _ = _dense_ols(df, ["firm", "date", "sector"], ["x1", "x2"])
    np.testing.assert_allclose(res.params, beta[:2], atol=1e-7, rtol=1e-7)
    assert res.converged


def test_hdfe_clustered_se_matches_dense_sandwich():
    df = _make_panel(seed=17, drop_frac=0.15)
    res = hdfe(
        df,
        y="y",
        x=["x1", "x2"],
        absorb=["firm", "date"],
        cluster="firm",
        entity="firm",
        time="date",
    )
    _, resid, xtx_inv, n_params, w = _dense_ols(df, ["firm", "date"], ["x1", "x2"])
    scores = w * resid[:, None]
    firms = df["firm"].to_numpy()
    meat = np.zeros((w.shape[1], w.shape[1]))
    for g in np.unique(firms):
        s = scores[firms == g].sum(axis=0)
        meat += np.outer(s, s)
    n, n_groups = df.height, np.unique(firms).size
    corr = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / (n - n_params))
    v = corr * (xtx_inv @ meat @ xtx_inv)
    np.testing.assert_allclose(
        res.std_errors, np.sqrt(np.diag(v)[:2]), atol=TOL, rtol=TOL
    )
    assert "cluster" in res.vcov_type


def test_hdfe_two_way_cluster_and_driscoll_kraay_are_finite_and_psd():
    df = _make_panel(n_entities=15, n_times=25, seed=23)
    two_way = hdfe(
        df,
        y="y",
        x=["x1", "x2"],
        absorb=["firm"],
        cluster=["firm", "date"],
        entity="firm",
        time="date",
    )
    assert np.all(np.isfinite(two_way.std_errors))
    assert np.all(np.linalg.eigvalsh(two_way.vcov) >= -1e-12)
    assert "two-way" in two_way.vcov_type

    dk = hdfe(
        df,
        y="y",
        x=["x1", "x2"],
        absorb=["firm"],
        vcov="driscoll-kraay",
        entity="firm",
        time="date",
    )
    assert np.all(np.isfinite(dk.std_errors))
    assert np.all(dk.std_errors > 0)
    assert "Driscoll-Kraay" in dk.vcov_type


def test_hdfe_recovers_known_slopes():
    df = _make_panel(n_entities=40, n_times=40, seed=1)
    res = hdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "date"], entity="firm", time="date"
    )
    assert abs(res.params[0] - 1.5) < 0.05
    assert abs(res.params[1] + 0.7) < 0.05
    assert 0.0 <= res.r2_within <= 1.0


def test_hdfe_summary_conf_int_and_wald():
    df = _make_panel(n_entities=20, n_times=20, seed=2)
    res = hdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "date"], entity="firm", time="date"
    )
    assert isinstance(res, HDFEResult)
    summary = res.summary()
    assert summary.columns == ["term", "estimate", "std_error", "t_stat", "p_value"]
    assert summary.height == 2
    ci = res.conf_int()
    assert (ci["lower"].to_numpy() < res.params).all()
    assert (ci["upper"].to_numpy() > res.params).all()
    wald = res.wald_test()
    assert wald["df"] == 2.0
    assert 0.0 <= wald["p_value"] <= 1.0
    assert wald["p_value"] < 0.01  # the slopes are real


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_hdfe_rejects_collinear_regressor():
    df = _make_panel(seed=4).with_columns(pl.col("firm").cast(pl.Float64).alias("dup"))
    with pytest.raises(ValueError, match="rank deficient"):
        hdfe(df, y="y", x=["dup"], absorb=["firm"], entity="firm", time="date")


def test_hdfe_rejects_unknown_column_and_vcov():
    df = _make_panel(seed=4)
    with pytest.raises(ValueError, match="not found"):
        hdfe(df, y="nope", x=["x1"], absorb=["firm"], entity="firm", time="date")
    with pytest.raises(ValueError, match="unknown `vcov"):
        hdfe(
            df,
            y="y",
            x=["x1"],
            absorb=["firm"],
            vcov="bootstrap",
            entity="firm",
            time="date",
        )


# --------------------------------------------------------------------------- #
# Transformer: leak-safety contract
# --------------------------------------------------------------------------- #
def test_transformer_reproduces_the_fit_residuals():
    df = _make_panel(n_entities=10, n_times=15, seed=8)
    res = hdfe(df, y="y", x=["x1"], absorb=["firm", "date"], entity="firm", time="date")
    tr = HDFETransformer(columns=["y", "x1"], absorb=["firm", "date"]).fit(
        df, entity="firm", time="date"
    )
    out = tr.transform(df).collect()
    np.testing.assert_allclose(out["y_hdfe"].to_numpy(), res.resid_y, atol=1e-9)
    np.testing.assert_allclose(out["x1_hdfe"].to_numpy(), res.resid_x[:, 0], atol=1e-9)
    assert tr.output_names == ["y_hdfe", "x1_hdfe"]


def test_transformer_declares_the_contract_and_needs_fitting():
    assert HDFETransformer.panel_safe is True
    assert HDFETransformer.leakage_safe is True
    tr = HDFETransformer(columns=["y"], absorb=["firm"])
    df = _make_panel(seed=9)
    with pytest.raises(RuntimeError, match="not fitted"):
        tr.transform(df, entity="firm", time="date")


def test_transformer_is_invariant_to_future_rows():
    """The train-fitted offsets must not change when future rows are appended."""
    df = _make_panel(n_entities=8, n_times=30, seed=13)
    train = df.filter(pl.col("date") < 15)
    tr = HDFETransformer(columns=["y"], absorb=["firm"]).fit(
        train, entity="firm", time="date"
    )
    on_train = tr.transform(train).collect()["y_hdfe"].to_numpy()
    on_full = (
        tr.transform(df).collect().filter(pl.col("date") < 15)["y_hdfe"].to_numpy()
    )
    np.testing.assert_allclose(on_train, on_full, atol=1e-12)


def test_transformer_handles_unseen_levels():
    df = _make_panel(n_entities=6, n_times=10, seed=21)
    train = df.filter(pl.col("firm") < 4)
    tr = HDFETransformer(columns=["y"], absorb=["firm"], unseen="zero").fit(
        train, entity="firm", time="date"
    )
    out = tr.transform(df).collect().filter(pl.col("firm") >= 4)
    # Unseen firms get a zero offset, so the residual is the raw value.
    np.testing.assert_allclose(
        out["y_hdfe"].to_numpy(), out["y"].to_numpy(), atol=1e-12
    )

    tr_null = HDFETransformer(columns=["y"], absorb=["firm"], unseen="null").fit(
        train, entity="firm", time="date"
    )
    out_null = tr_null.transform(df).collect().filter(pl.col("firm") >= 4)
    assert out_null["y_hdfe"].null_count() == out_null.height


def test_transformer_accepts_panelframe_and_can_drop_originals():
    df = _make_panel(seed=6)
    panel = PanelFrame(df, entity="firm", time="date")
    tr = HDFETransformer(
        columns=["y"], absorb=["firm"], drop_original=True, suffix="_w"
    ).fit(panel)
    out = tr.transform(panel).collect()
    assert "y" not in out.columns
    assert "y_w" in out.columns


def test_transformer_validates_its_arguments():
    with pytest.raises(ValueError, match="at least one column"):
        HDFETransformer(columns=[], absorb=["firm"])
    with pytest.raises(ValueError, match="at least one fixed-effect"):
        HDFETransformer(columns=["y"], absorb=[])
    with pytest.raises(ValueError, match="`unseen`"):
        HDFETransformer(columns=["y"], absorb=["firm"], unseen="mean")


# --------------------------------------------------------------------------- #
# Shared numerics (``econ._common``) -- the scipy-free distribution tails every
# estimator in the package uses for its p-values.
# --------------------------------------------------------------------------- #
def test_chi2_survival_matches_published_critical_values():
    # 95th percentiles of the chi-squared distribution.
    for df, crit in [(1, 3.8414588), (2, 5.9914645), (3, 7.8147279), (5, 11.070498)]:
        assert chi2_sf(crit, df) == pytest.approx(0.05, abs=1e-7)
    assert chi2_sf(-1.0, 1) == 1.0
    assert chi2_sf(0.0, 3) == 1.0
    with pytest.raises(ValueError, match="`df` must be positive"):
        chi2_sf(1.0, 0)


def test_chi2_survival_is_vectorised():
    out = chi2_sf(np.array([0.0, 3.8414588, 100.0]), 1)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [1.0, 0.05, 0.0], atol=1e-7)


def test_student_t_tails_match_published_critical_values():
    # 97.5th percentiles of Student's t.
    for df, crit in [(10, 2.2281389), (20, 2.0859634), (30, 2.0422725)]:
        assert t_sf(crit, df) == pytest.approx(0.025, abs=1e-7)
    assert t_sf(0.0, 7) == pytest.approx(0.5, abs=1e-12)
    assert t_cdf(0.0, 7) == pytest.approx(0.5, abs=1e-12)
    # Symmetry and the cdf/sf identity.
    assert t_sf(-1.3, 12) == pytest.approx(1.0 - t_sf(1.3, 12), abs=1e-12)
    assert t_cdf(1.3, 12) == pytest.approx(1.0 - t_sf(1.3, 12), abs=1e-12)


def test_f_tail_reduces_to_the_squared_t():
    """``F(1, m)`` is the square of ``t(m)``, so the tails must agree."""
    for x, m in [(0.5, 10), (2.0, 25), (9.0, 40)]:
        assert f_sf(x, 1, m) == pytest.approx(2.0 * t_sf(np.sqrt(x), m), rel=1e-10)
    assert f_sf(-1.0, 2, 3) == 1.0


def test_newey_west_long_run_variance():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(50, 2))
    # Zero lags is just the outer-product sum.
    np.testing.assert_allclose(newey_west_lrv(s, 0), s.T @ s, atol=1e-12)
    # With lags it stays symmetric and adds the weighted autocovariances.
    lrv = newey_west_lrv(s, 4)
    np.testing.assert_allclose(lrv, lrv.T, atol=1e-12)
    assert not np.allclose(lrv, s.T @ s)
