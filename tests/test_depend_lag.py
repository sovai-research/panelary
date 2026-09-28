"""Lag scans with FWER control across lags."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp
from panelary.depend._lag import adjust_pvalues


def _lead_lag(
    n: int = 400, lag: int = 3, n_ent: int = 1, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_ent):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n) * 0.4
        y[lag:] += np.abs(x[:-lag])
        frames.append(pl.DataFrame({"e": [i] * n, "t": np.arange(n), "x": x, "y": y}))
    return pl.concat(frames)


def test_ccf_finds_the_planted_lag_single_series() -> None:
    df = _lead_lag()
    prof = dp.nonlinear_ccf(
        df, "x", "y", time="t", max_lag=6, method="xi", n_resamples=99
    )
    assert prof["lag"].to_list() == list(range(-6, 7))
    assert dp.optimal_lag(prof) == 3
    row = prof.filter(pl.col("lag") == 3)
    assert row["p_value_adj"][0] <= 0.02
    others = prof.filter(pl.col("lag") != 3)
    assert (others["p_value_adj"] > 0.05).mean() > 0.8
    # Romano-Wolf adjusted p-values are never below the raw ones.
    assert (prof["p_value_adj"] >= prof["p_value"] - 1e-12).all()
    assert prof["null_method"][0] == "permutation"  # serial pre-check passed
    assert any("romano_wolf" in w for w in prof["warnings"][0])


def test_ccf_panel_common_time_and_n_obs() -> None:
    df = _lead_lag(n=150, n_ent=6, seed=1)
    prof = dp.lag_dependence(
        df,
        "x",
        "y",
        entity="e",
        time="t",
        lags=[0, 3, 5],
        method="spearman",
        n_resamples=49,
    )
    assert prof["null_method"].unique().to_list() == ["common-time"]
    assert prof.filter(pl.col("lag") == 3)["n_obs"][0] == 6 * 147
    assert prof.filter(pl.col("lag") == 3)["n_entities"][0] == 6


def test_closed_form_lag_scan_uses_holm() -> None:
    df = _lead_lag(seed=2)
    prof = dp.lag_dependence(
        df, "x", "y", time="t", lags=[1, 2, 3], method="xi", null="asymptotic"
    )
    assert prof["null_method"][0] == "asymptotic"
    assert any("holm" in w for w in prof["warnings"][0])
    p = prof["p_value"].to_numpy()
    np.testing.assert_allclose(
        prof["p_value_adj"].to_numpy(), adjust_pvalues(p, correction="holm")[0]
    )


def test_acf_permutation_and_surrogate_nulls() -> None:
    rng = np.random.default_rng(3)
    n = 600
    e = rng.standard_normal(n)
    lin = np.empty(n)
    lin[0] = e[0]
    for t in range(1, n):
        lin[t] = 0.6 * lin[t - 1] + e[t]
    nl = np.empty(n)
    nl[0] = e[0]
    for t in range(1, n):
        nl[t] = 0.9 * np.cos(2.0 * nl[t - 1]) + 0.3 * e[t]
    df = pl.DataFrame({"t": np.arange(n), "lin": lin, "nl": nl})
    # Serial independence is rejected for both.
    a_lin = dp.nonlinear_acf(df, "lin", time="t", max_lag=3, n_resamples=99)
    assert a_lin["null_method"][0] == "permutation"
    assert a_lin.filter(pl.col("lag") == 1)["p_value_adj"][0] < 0.05
    # IAAFT keeps the linear ACF: a linear AR(1) shows nothing beyond it ...
    s_lin = dp.nonlinear_acf(
        df, "lin", time="t", lags=[1], null="iaaft", n_resamples=49
    )
    assert s_lin["p_value"][0] > 0.05
    # ... while the nonlinear map does.
    s_nl = dp.nonlinear_acf(df, "nl", time="t", lags=[1], null="iaaft", n_resamples=49)
    assert s_nl["p_value"][0] <= 0.04
    with pytest.raises(ValueError, match="preserves the serial dependence"):
        dp.nonlinear_acf(df, "lin", time="t", null="block")


def test_optimal_lag_tie_breaks() -> None:
    prof = pl.DataFrame(
        {
            "lag": [-2, 1, 2],
            "estimate": [0.3, 0.3, 0.1],
            "p_value": [0.01, 0.01, 0.01],
            "p_value_adj": [0.02, 0.02, 0.02],
        }
    )
    assert dp.optimal_lag(prof) == 1
    with pytest.raises(ValueError):
        dp.optimal_lag(prof.with_columns(pl.lit(float("nan")).alias("estimate")))


def test_adjust_pvalues_variants() -> None:
    p = np.array([0.001, 0.02, 0.04, np.nan])
    for c in ("holm", "benjamini_hochberg", "benjamini_yekutieli"):
        adj, label = adjust_pvalues(p, correction=c)
        assert label == c and np.isnan(adj[3]) and (adj[:3] >= p[:3]).all()
    adj, label = adjust_pvalues(p, correction="romano_wolf")
    assert label == "holm"  # no draws: documented fallback
    with pytest.raises(ValueError):
        adjust_pvalues(p, correction="bonferroni-ish")
