"""Forecast comparison beyond DM: parity with per-model loops, oracles, traps, prefix invariance."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary.econ._common import newey_west_lrv
from panelary.testing import assert_no_lookahead, assert_prefix_invariant
from panelary.validation import (
    FluctuationResult,
    _gr_tables,
    clark_west,
    diebold_mariano,
    encompassing_test,
    fluctuation_test,
    giacomini_white,
    loss_panel,
    mincer_zarnowitz,
    newey_west_variance,
    one_time_reversal_test,
    oos_r2,
    pesaran_timmermann,
)
from panelary.validation._forecast_compare import (
    _dm_columns,
    historical_mean_benchmark,
    historical_mean_expr,
)


def _data(
    seed: int = 0, n: int = 300, m: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, m))
    y = 0.4 * x[:, 0] + rng.standard_normal(n)
    y[1:] += 0.3 * y[:-1]
    fc = 0.3 * x + 0.1 * rng.standard_normal((n, m))
    bench = np.full(n, 0.02)
    return y, bench, fc


# --------------------------------------------------------------------------- #
# DM columns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_dm_columns_match_diebold_mariano(horizon: int, alternative: str) -> None:
    rng = np.random.default_rng(1)
    la = rng.standard_normal((200, 5)) ** 2
    lb = rng.standard_normal(200) ** 2 * 1.1
    la[:7, 2] = np.nan
    out = _dm_columns(la - lb[:, None], horizon=horizon, alternative=alternative)
    for j in range(5):
        ref = diebold_mariano(la[:, j], lb, horizon=horizon, alternative=alternative)
        assert out["statistic"][j] == pytest.approx(ref.statistic, rel=1e-12)
        assert out["pvalue"][j] == pytest.approx(ref.pvalue, rel=1e-10, abs=1e-14)
        assert out["n"][j] == ref.n_obs


# --------------------------------------------------------------------------- #
# Parity with per-model loops
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon", [1, 2])
def test_clark_west_matches_loop(horizon: int) -> None:
    y, bench, fc = _data(2)
    res = clark_west(y, bench, fc, horizon=horizon)
    for j in range(fc.shape[1]):
        f = (y - bench) ** 2 - ((y - fc[:, j]) ** 2 - (bench - fc[:, j]) ** 2)
        t = f.mean() / math.sqrt(newey_west_variance(f, lags=horizon - 1) / f.size)
        assert res.statistic[j] == pytest.approx(t, rel=1e-12)
        assert res.pvalue[j] == pytest.approx(
            0.5 * math.erfc(t / math.sqrt(2)), rel=1e-10
        )
        assert res.estimate[j] == pytest.approx(f.mean(), rel=1e-12)
    assert (
        res.reference == "N(0,1) one-sided" and res.hac == f"bartlett(L={horizon - 1})"
    )


@pytest.mark.parametrize("lags", [0, 3])
def test_mincer_zarnowitz_hac_matches_uncentred_sandwich(lags: int) -> None:
    y, _, fc = _data(3)
    res = mincer_zarnowitz(y, fc, hac_lags=lags)
    for j in range(fc.shape[1]):
        x = np.column_stack([np.ones(y.size), fc[:, j]])
        theta, *_ = np.linalg.lstsq(x, y, rcond=None)
        u = y - x @ theta
        s = newey_west_lrv(x * u[:, None], lags)
        xtx_inv = np.linalg.inv(x.T @ x)
        v = xtx_inv @ s @ xtx_inv
        delta = theta - np.array([0.0, 1.0])
        wald = float(delta @ np.linalg.solve(v, delta))
        assert res.statistic[j] == pytest.approx(wald, rel=1e-10)
        assert res.estimate[j] == pytest.approx(theta[1], rel=1e-12)
        assert res.details["intercept"][j] == pytest.approx(
            theta[0], rel=1e-10, abs=1e-14
        )
        assert res.details["intercept_se"][j] == pytest.approx(
            math.sqrt(v[0, 0]), rel=1e-10
        )
        assert res.pvalue[j] == pytest.approx(math.exp(-wald / 2), rel=1e-10)


def test_mincer_zarnowitz_ols_is_exact_f() -> None:
    stats = pytest.importorskip("scipy.stats")
    y, _, fc = _data(4, n=80)
    res = mincer_zarnowitz(y, fc, cov="ols")
    n = y.size
    for j in range(fc.shape[1]):
        x = np.column_stack([np.ones(n), fc[:, j]])
        theta, *_ = np.linalg.lstsq(x, y, rcond=None)
        rss_u = float(np.sum((y - x @ theta) ** 2))
        rss_r = float(np.sum((y - fc[:, j]) ** 2))
        f = ((rss_r - rss_u) / 2) / (rss_u / (n - 2))
        assert res.statistic[j] / 2 == pytest.approx(f, rel=1e-10)
        assert res.pvalue[j] == pytest.approx(stats.f.sf(f, 2, n - 2), rel=1e-9)
    assert res.reference == "F(2,T-2)"


def test_mincer_zarnowitz_statsmodels_oracle() -> None:
    sm = pytest.importorskip("statsmodels.api")
    y, _, fc = _data(5)
    res = mincer_zarnowitz(y, fc[:, 0], hac_lags=2)
    fit = sm.OLS(y, sm.add_constant(fc[:, 0])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2, "use_correction": False}
    )
    wald = fit.wald_test("const = 0, x1 = 1", use_f=False, scalar=True)
    assert res.statistic[0] == pytest.approx(float(wald.statistic), rel=1e-8)


def test_pesaran_timmermann_matches_formula() -> None:
    y, _, fc = _data(6)
    res = pesaran_timmermann(y, fc)
    n = y.size
    for j in range(fc.shape[1]):
        yd, xd = y > 0, fc[:, j] > 0
        p = np.mean(yd == xd)
        py, px = yd.mean(), xd.mean()
        ps = py * px + (1 - py) * (1 - px)
        vp = ps * (1 - ps) / n
        vps = (
            (2 * py - 1) ** 2 * px * (1 - px) / n
            + (2 * px - 1) ** 2 * py * (1 - py) / n
            + 4 * py * px * (1 - py) * (1 - px) / n**2
        )
        s = (p - ps) / math.sqrt(vp - vps)
        assert res.statistic[j] == pytest.approx(s, rel=1e-12)
        assert res.estimate[j] == pytest.approx(p, rel=1e-14)


def test_pesaran_timmermann_exact_is_fisher() -> None:
    stats = pytest.importorskip("scipy.stats")
    y, _, fc = _data(7, n=60)
    res = pesaran_timmermann(y, fc, exact=True)
    for j in range(fc.shape[1]):
        yd, xd = y > 0, fc[:, j] > 0
        table = [
            [np.sum(yd & xd), np.sum(~yd & xd)],
            [np.sum(yd & ~xd), np.sum(~yd & ~xd)],
        ]
        _, p = stats.fisher_exact(table, alternative="greater")
        assert res.pvalue[j] == pytest.approx(p, rel=1e-12)
    assert res.reference == "hypergeometric-exact one-sided"


def test_pesaran_timmermann_multi_step_uses_hac_regression() -> None:
    y, _, fc = _data(8)
    res = pesaran_timmermann(y, fc, horizon=4)
    assert res.hac == "bartlett(L=3)"
    assert "HAC regression" in res.reference
    assert np.all(np.isfinite(res.pvalue))


def test_encompassing_matches_dm_loop() -> None:
    y, _, fc = _data(9)
    fa = 0.2 * fc[:, 0] + 0.1
    res = encompassing_test(y, fa, fc[:, 1:], horizon=2)
    for j in range(3):
        ea, eb = y - fa, y - fc[:, 1 + j]
        ref = diebold_mariano(
            (ea - eb) * ea, np.zeros(y.size), horizon=2, alternative="greater"
        )
        rev = diebold_mariano(
            (eb - ea) * eb, np.zeros(y.size), horizon=2, alternative="greater"
        )
        assert res.statistic[j] == pytest.approx(ref.statistic, rel=1e-12)
        assert res.pvalue[j] == pytest.approx(ref.pvalue, rel=1e-10)
        assert res.details["reverse_pvalue"][j] == pytest.approx(rev.pvalue, rel=1e-10)
        lam = np.sum((ea - eb) * ea) / np.sum((ea - eb) ** 2)
        assert res.estimate[j] == pytest.approx(lam, rel=1e-12)


def test_giacomini_white_is_n_r2_of_one_on_z() -> None:
    rng = np.random.default_rng(10)
    la = rng.standard_normal((250, 3)) ** 2
    lb = rng.standard_normal(250) ** 2
    lb[1:] += 0.4 * lb[:-1]
    res = giacomini_white(la, lb)
    for j in range(3):
        d = la[:, j] - lb
        z = np.column_stack([d[1:], d[:-1] * d[1:]])
        n = z.shape[0]
        beta, *_ = np.linalg.lstsq(z, np.ones(n), rcond=None)
        ssr = float(np.sum((np.ones(n) - z @ beta) ** 2))
        assert res.statistic[j] == pytest.approx(n - ssr, rel=1e-10)
        assert res.pvalue[j] == pytest.approx(
            math.exp(-res.statistic[j] / 2), rel=1e-10
        )
    assert res.reference == "chi2(2)"


def test_giacomini_white_multi_step_and_instruments() -> None:
    rng = np.random.default_rng(11)
    d = rng.standard_normal((300, 2))
    zeros = np.zeros((300, 2))
    res = giacomini_white(d, zeros, horizon=3)
    for j in range(2):
        x = d[:, j]
        lead, h1 = x[3:], x[:-3]
        z = np.column_stack([lead, h1 * lead])
        n = z.shape[0]
        omega = newey_west_lrv(z, 2) / n
        zbar = z.mean(0)
        assert res.statistic[j] == pytest.approx(
            n * zbar @ np.linalg.solve(omega, zbar), rel=1e-10
        )
    const = giacomini_white(d, zeros, instruments="constant")
    assert const.reference == "chi2(1)"
    user = giacomini_white(d, zeros, instruments=rng.standard_normal((300, 3)))
    assert user.reference == "chi2(3)"
    warned = giacomini_white(d, zeros, estimation_scheme="expanding")
    assert any("rolling" in w for w in warned.warnings)


# --------------------------------------------------------------------------- #
# Nested-model null: DM has no power, CW is sized (plan §1.1)
# --------------------------------------------------------------------------- #
def _nested_null(reps: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS, R = 120, P = 240: y_{t+1} = mu + e, x irrelevant."""
    rng = np.random.default_rng(seed)
    r_win, p_oos = 120, 240
    n = r_win + p_oos + 1
    x = rng.standard_normal((n, reps))
    y = 0.5 + rng.standard_normal((n, reps))
    # pairs (x_{s-1}, y_s), s = 1..n-1
    xs, ys = x[:-1], y[1:]
    c = lambda a: np.vstack([np.zeros((1, reps)), np.cumsum(a, axis=0)])  # noqa: E731
    sx, sy, sxy, sxx = c(xs), c(ys), c(xs * ys), c(xs * xs)
    t0 = np.arange(r_win, r_win + p_oos)  # forecast y[t0 + 1] from pairs t0-R..t0-1
    win = lambda s: s[t0] - s[t0 - r_win]  # noqa: E731
    mx, my = win(sx) / r_win, win(sy) / r_win
    b = (win(sxy) / r_win - mx * my) / (win(sxx) / r_win - mx * mx)
    a = my - b * mx
    target = y[t0 + 1]
    f1 = a + b * x[t0]
    return target, my, f1


def test_nested_null_dm_is_undersized_and_cw_is_not() -> None:
    target, f0, f1 = _nested_null(1000, 12)
    l0, l1 = (target - f0) ** 2, (target - f1) ** 2
    dm = _dm_columns(l0 - l1, alternative="greater")["pvalue"]
    cw_rej, dm_rej = 0, float(np.mean(dm < 0.10))
    for j in range(target.shape[1]):
        res = clark_west(target[:, j], f0[:, j], f1[:, j])
        cw_rej += bool(res.pvalue[0] < 0.10)
    assert dm_rej <= 0.02, dm_rej
    assert 0.03 <= cw_rej / target.shape[1] <= 0.12, cw_rej


# --------------------------------------------------------------------------- #
# R^2_OS and the historical-mean benchmark (trap T2)
# --------------------------------------------------------------------------- #
def test_historical_mean_benchmark_values() -> None:
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    got = historical_mean_benchmark(y, horizon=1, min_periods=2)
    np.testing.assert_allclose(got, [np.nan, np.nan, 1.5, 1.5, 7 / 3, 3.0])
    got2 = historical_mean_benchmark(y, horizon=2, min_periods=1)
    np.testing.assert_allclose(got2, [np.nan, np.nan, 1.0, 1.5, 1.5, 7 / 3])


def test_historical_mean_expr_matches_numpy_and_is_leak_free() -> None:
    rng = np.random.default_rng(13)
    frame = pl.DataFrame(
        {
            "e": ["a"] * 40 + ["b"] * 40,
            "t": list(range(40)) * 2,
            "y": rng.standard_normal(80),
        }
    )
    expr = historical_mean_expr("y", entity="e", horizon=2, min_periods=5)
    out = frame.with_columns(expr)["y_hist_mean"].to_numpy()
    for k, ent in enumerate(("a", "b")):
        ref = historical_mean_benchmark(
            frame["y"].to_numpy()[k * 40 : (k + 1) * 40], horizon=2, min_periods=5
        )
        np.testing.assert_allclose(
            out[k * 40 : (k + 1) * 40], ref, rtol=1e-14, equal_nan=True
        )
    assert_no_lookahead(expr, frame, entity="e", time="t")
    assert_prefix_invariant(expr, frame, entity="e", time="t")


def test_oos_r2_matches_definition_and_zero_benchmark() -> None:
    y, _, fc = _data(14)
    res = oos_r2(y, fc, min_periods=30)
    bench = historical_mean_benchmark(y, min_periods=30)
    ok = np.isfinite(bench)
    for j in range(fc.shape[1]):
        r2 = 1 - np.sum((y[ok] - fc[ok, j]) ** 2) / np.sum((y[ok] - bench[ok]) ** 2)
        assert res.estimate[j] == pytest.approx(r2, rel=1e-12)
    assert res.n_obs.tolist() == [int(ok.sum())] * fc.shape[1]
    zero = oos_r2(y, fc, benchmark="zero")
    # the Gu-Kelly-Xiu convention of embed.baseline_report
    assert zero.estimate[0] == pytest.approx(
        1 - np.sum((y - fc[:, 0]) ** 2) / np.sum(y**2), rel=1e-12
    )
    trunc = oos_r2(y, fc, truncate_at=0.0, min_periods=30)
    assert np.all(trunc.estimate != res.estimate)


def test_full_sample_mean_benchmark_changes_r2_and_leaks() -> None:
    """Trap T2: the full-sample mean uses future returns."""
    y, _, fc = _data(15)
    honest = oos_r2(y, fc, min_periods=30)
    full = np.full(y.size, y.mean())
    full[:30] = np.nan
    leaky = oos_r2(y, fc, benchmark=full)
    assert np.all(np.abs(honest.estimate - leaky.estimate) > 1e-3)
    frame = pl.DataFrame({"e": ["a"] * y.size, "t": np.arange(y.size), "y": y})
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(
            pl.col("y").mean().over("e").alias("full_mean"), frame, entity="e", time="t"
        )


def test_r2_paths_are_prefix_invariant() -> None:
    y, _, fc = _data(16, n=200, m=2)
    frame = pl.DataFrame(
        {"e": ["a"] * 200, "t": np.arange(200), "y": y, "f0": fc[:, 0], "f1": fc[:, 1]}
    )

    def op(df: pl.DataFrame) -> pl.DataFrame:
        res = oos_r2(
            df["y"].to_numpy(), df.select("f0", "f1").to_numpy(), min_periods=20
        )
        return df.with_columns(
            pl.Series("r2_path", res.details["r2_path"][0]),
            pl.Series("gw_path", res.details["cumulative_sse_difference"][1]),
            pl.Series("bench", res.details["benchmark_forecast"][0]),
        ).select("e", "t", "r2_path", "gw_path", "bench")

    assert_prefix_invariant(op, frame, entity="e", time="t")
    assert_no_lookahead(op, frame, entity="e", time="t")
    res = oos_r2(y, fc, min_periods=20)
    assert res.details["r2_path"][0, -1] == pytest.approx(res.estimate[0], rel=1e-12)


# --------------------------------------------------------------------------- #
# Giacomini-Rossi fluctuation and one-time reversal
# --------------------------------------------------------------------------- #
def test_fluctuation_path_matches_loop_and_centred_sup() -> None:
    rng = np.random.default_rng(17)
    d = rng.standard_normal((200, 2))
    d[120:, 0] += 0.8
    res = fluctuation_test(d, np.zeros((200, 2)), window=0.3, hac_lags=2)
    assert isinstance(res, FluctuationResult)
    m = 60
    for j in range(2):
        sig = math.sqrt(newey_west_variance(d[:, j], lags=2))
        want = np.array(
            [
                d[t - m + 1 : t + 1, j].sum() / (math.sqrt(m) * sig)
                for t in range(m - 1, 200)
            ]
        )
        np.testing.assert_allclose(res.path[m - 1 :, j], want, rtol=1e-12)
        assert np.isnan(res.path[: m - 1, j]).all()
        # GR label at the centre: same windows, same supremum
        centred = max(
            abs(d[c - m // 2 : c + m // 2, j].sum()) / (math.sqrt(m) * sig)
            for c in range(m // 2, 200 - m // 2 + 1)
        )
        assert res.max_stat[j] == pytest.approx(centred, rel=1e-12)
    assert res.critical_value == pytest.approx(
        _gr_tables.fluctuation_critical_value(0.3, 0.05)
    )
    assert res.reject[0] and not res.reject[1]
    ev = res.to_evaluation()
    assert ev.test == "fluctuation" and ev.reference == "gr-table(mu=0.3)"
    assert ev.details["prefix_invariant"].tolist() == [0.0, 0.0]
    assert res.to_frame().height == 400


def test_fluctuation_monitor_is_prefix_invariant_and_test_mode_is_not() -> None:
    """Trap T3."""
    rng = np.random.default_rng(18)
    la = rng.standard_normal(160) ** 2
    lb = rng.standard_normal(160) ** 2
    frame = pl.DataFrame({"e": ["a"] * 160, "t": np.arange(160), "la": la, "lb": lb})

    def monitor(df: pl.DataFrame) -> pl.DataFrame:
        r = fluctuation_test(
            df["la"].to_numpy(),
            df["lb"].to_numpy(),
            window=30,
            mode="monitor",
            hac_lags=1,
        )
        return df.with_columns(pl.Series("f", r.path[:, 0])).select("e", "t", "f")

    def test_mode(df: pl.DataFrame) -> pl.DataFrame:
        r = fluctuation_test(
            df["la"].to_numpy(), df["lb"].to_numpy(), window=0.2, hac_lags=1
        )
        return df.with_columns(pl.Series("f", r.path[:, 0])).select("e", "t", "f")

    assert_prefix_invariant(monitor, frame, entity="e", time="t")
    assert_no_lookahead(monitor, frame, entity="e", time="t")
    # test mode standardises by the full-sample LRV, so the past moves with the future
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(test_mode, frame, entity="e", time="t")


def test_fluctuation_rejects_n_dependent_settings_in_monitor_mode() -> None:
    """Trap T5: no T-dependent window or bandwidth in a time-indexed output."""
    d = np.random.default_rng(19).standard_normal(100)
    with pytest.raises(ValueError, match="fixed integer"):
        fluctuation_test(d, np.zeros(100), window=0.3, mode="monitor")
    with pytest.raises(ValueError, match="non-negative integer"):
        fluctuation_test(
            d, np.zeros(100), window=20, mode="monitor", hac_lags="andrews"
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tabulated"):
        fluctuation_test(d, np.zeros(100), window=0.33)


def test_one_time_reversal_detects_planted_break() -> None:
    rng = np.random.default_rng(20)
    d = rng.standard_normal((400, 2))
    d[:200, 0] += 0.5
    d[200:, 0] -= 0.5
    res = one_time_reversal_test(d, np.zeros((400, 2)))
    assert res.pvalue[0] < 0.01 < res.pvalue[1]
    assert abs(res.details["break_index"][0] - 200) <= 15
    assert res.details["mean_before"][0] > 0 > res.details["mean_after"][0]
    n = 400
    x = d[:, 0]
    sig2 = newey_west_variance(x, lags=0)
    s = np.cumsum(x)
    lm2 = [
        (s[t - 1] - t / n * s[-1]) ** 2 / (sig2 * n * (t / n) * (1 - t / n))
        for t in range(60, 341)
    ]
    assert res.statistic[0] == pytest.approx(
        n * x.mean() ** 2 / sig2 + max(lm2), rel=1e-12
    )
    with pytest.raises(ValueError, match="trim"):
        one_time_reversal_test(d, np.zeros((400, 2)), trim=0.1)


def test_gr_table_lookup() -> None:
    k05 = _gr_tables.fluctuation_critical_value(0.3, 0.05)
    k10 = _gr_tables.fluctuation_critical_value(0.3, 0.10)
    assert k05 > k10 > 0
    p, clipped = _gr_tables.fluctuation_pvalue(np.array([k05, k10]), 0.3)
    np.testing.assert_allclose(p, [0.05, 0.10], rtol=1e-9)
    assert not clipped
    # monotone in mu: longer windows average more, so the sup is smaller
    ks = [_gr_tables.fluctuation_critical_value(mu, 0.05) for mu in _gr_tables.MU_GRID]
    assert all(a > b for a, b in zip(ks[:-1], ks[1:], strict=True))
    assert _gr_tables.reversal_critical_value(
        0.15
    ) > _gr_tables.reversal_critical_value(0.20)


# --------------------------------------------------------------------------- #
# loss_panel
# --------------------------------------------------------------------------- #
def test_loss_panel_date_mean_equals_group_by() -> None:
    rng = np.random.default_rng(21)
    n_e, n_t = 6, 30
    frame = pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_e), n_t),
            "t": np.tile(np.arange(n_t), n_e),
            "y": rng.standard_normal(n_e * n_t),
            "fa": rng.standard_normal(n_e * n_t),
            "fb": rng.standard_normal(n_e * n_t),
        }
    ).with_columns(
        pl.when(pl.col("e") == 0).then(None).otherwise(pl.col("fa")).alias("fa")
    )
    dates, losses, names = loss_panel(
        frame, entity="e", time="t", target="y", forecasts=["fa", "fb"]
    )
    manual = (
        frame.drop_nulls()
        .group_by("t")
        .agg(
            ((pl.col("y") - pl.col("fa")) ** 2).mean().alias("la"),
            ((pl.col("y") - pl.col("fb")) ** 2).mean().alias("lb"),
        )
        .sort("t")
    )
    assert names == ("fa", "fb")
    np.testing.assert_array_equal(dates, manual["t"].to_numpy())
    np.testing.assert_allclose(losses, manual.select("la", "lb").to_numpy(), rtol=1e-14)
    _, few, _ = loss_panel(
        frame, entity="e", time="t", target="y", forecasts="fb", min_entities=7
    )
    assert few.shape == (0, 1)
    d_rows, row_losses, _ = loss_panel(
        frame,
        entity="e",
        time="t",
        target="y",
        forecasts=["fa"],
        aggregate="none",
        loss="quantile",
        quantile=0.1,
    )
    assert row_losses.shape == ((n_e - 1) * n_t, 1)
    assert np.all(row_losses >= 0)
    with pytest.raises(ValueError, match="quantile"):
        loss_panel(
            frame, entity="e", time="t", target="y", forecasts="fa", loss="quantile"
        )


def test_panel_to_dm_pipeline() -> None:
    rng = np.random.default_rng(22)
    frame = pl.DataFrame(
        {
            "e": np.repeat(np.arange(20), 50),
            "t": np.tile(np.arange(50), 20),
            "y": rng.standard_normal(1000),
        }
    ).with_columns((pl.col("y") * 0.5).alias("good"), pl.lit(0.0).alias("zero"))
    _, losses, names = loss_panel(
        frame, entity="e", time="t", target="y", forecasts=["zero", "good"]
    )
    res = giacomini_white(losses[:, 0], losses[:, 1:], names=names[1:])
    assert res.names == ("good",) and res.pvalue[0] < 0.05


# --------------------------------------------------------------------------- #
# Slow: table verification and finite-sample size
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_verify_gr_tables_quick() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _gr_tables.verify_gr_tables(quick=True)
    assert out["fluctuation_max_abs_dev"] < out["fluctuation_tolerance"]
    assert out["reversal_max_abs_dev"] < out["reversal_tolerance"]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["test", "monitor"])
def test_fluctuation_size_at_p500(mode: str) -> None:
    """Plan §11.3: asymptotic k_alpha at P = 500 must be within 2 pp of nominal."""
    d = np.random.default_rng(23).standard_normal((500, 4000))
    window: float | int = 0.3 if mode == "test" else 150
    res = fluctuation_test(d, np.zeros_like(d), window=window, mode=mode)
    size = float(np.mean(res.reject))
    assert abs(size - 0.05) <= 0.02, size


@pytest.mark.slow
def test_one_time_reversal_size() -> None:
    d = np.random.default_rng(24).standard_normal((400, 4000))
    res = one_time_reversal_test(d, np.zeros_like(d))
    size = float(np.mean(res.pvalue < 0.05))
    assert abs(size - 0.05) <= 0.02, size
