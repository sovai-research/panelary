"""Turbulence (traps T1, T2), the observed-marginal Mahalanobis distance, and
the MarketState / Turbulence transformers."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core.panel_frame import PanelFrame
from panelary.covariance._estimate import estimate
from panelary.covariance._estimator import MarketState, Turbulence
from panelary.covariance._turbulence import inv_quad_observed, turbulence
from panelary.covariance._window import rolling
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

W = 20
COV = 0.9


def _frame() -> pl.DataFrame:
    return pn.synth.generate_panel(
        seed=11,
        n_entities=24,
        n_periods=140,
        start=dt.date(2022, 1, 3),
        every="1d",
        missing_rate=0.02,
        entry_rate=0.02,
        exit_rate=0.01,
        initial_fraction=0.7,
    ).panel


DF = _frame()
PANEL = PanelFrame(DF, entity="entity", time="time")
TIMES = PANEL.time_index().to_list()


# --------------------------------------------------------------------------- #
# observed-marginal Mahalanobis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_missing", [0, 1, 3, 20])
@pytest.mark.parametrize("method", ["qis", "lw_diagonal", "factor"])
def test_inv_quad_observed_is_the_exact_marginal(n_missing: int, method: str) -> None:
    rng = np.random.default_rng(n_missing)
    X = rng.standard_normal((40, 60)) + rng.standard_normal((40, 1))
    est = estimate(X, method=method)
    x = rng.standard_normal(60)
    miss = rng.choice(60, size=n_missing, replace=False)
    x[miss] = np.nan
    obs = np.isfinite(x)
    got, n_obs = inv_quad_observed(est, x)
    D = est.to_dense()[np.ix_(obs, obs)]
    ref = float(x[obs] @ np.linalg.solve(D, x[obs][:, None])[:, 0])
    assert n_obs == obs.sum()
    assert got == pytest.approx(ref, rel=1e-9)


# --------------------------------------------------------------------------- #
# T1: full-sample mean and covariance (Kritzman & Li as published)
# --------------------------------------------------------------------------- #
def test_T1_full_sample_turbulence_is_caught() -> None:
    def leaky(frame):
        df = frame.collect()
        wide = df.pivot(index="time", on="entity", values="value").sort("time")
        R = wide.drop("time").to_numpy()
        keep = np.isfinite(R).mean(axis=0) > 0.9
        R = R[:, keep]
        full = estimate(np.where(np.isfinite(R), R, np.nan), method="qis")
        d = [inv_quad_observed(full, r - full.location)[0] for r in R]
        st = wide.select("time").with_columns(pl.Series("turbulence", d))
        return df.join(st, on="time", how="left")

    def ours(frame):
        return turbulence(
            frame,
            returns="value",
            window=W,
            min_coverage=COV,
            refit="1w",
            broadcast=True,
        )

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky, PANEL, tol=0.0)
    assert ours(PANEL).get_column("turbulence").drop_nulls().len() > 1000
    assert_no_lookahead(ours, PANEL, tol=0.0)
    assert_prefix_invariant(ours, PANEL, tol=0.0)


# --------------------------------------------------------------------------- #
# T2: r_t inside its own covariance
# --------------------------------------------------------------------------- #
def test_T2_lag_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="T2"):
        turbulence(PANEL, returns="value", window=W, lag=0)
    with pytest.raises(ValueError, match="T2"):
        Turbulence("value", W, lag=0)


@pytest.mark.parametrize("refit", ["every", "1w"])
def test_T2_turbulence_equals_mahalanobis_under_the_estimate_at_t_minus_1(
    refit: str,
) -> None:
    tb = turbulence(PANEL, returns="value", window=W, min_coverage=COV, refit=refit)
    series = rolling(PANEL, returns="value", window=W, min_coverage=COV, schedule=refit)
    wide = DF.pivot(index="time", on="entity", values="value").sort("time")
    checked = 0
    for i in range(W + 2, len(TIMES), 7):
        row = tb.row(i, named=True)
        if row["turbulence"] is None:
            continue
        est = series.at(TIMES[i - 1])  # the estimate in force at t - 1
        assert row["asof_date"] == est.asof
        r = wide.row(i, named=True)
        x = np.array([r[e] for e in est.entities], dtype=float) - est.location
        q, n = inv_quad_observed(est, x)
        assert row["turbulence"] == q / n
        assert row["n_scored"] == n
        checked += 1
    assert checked > 10


def test_turbulence_rises_in_a_crisis_window() -> None:
    rng = np.random.default_rng(0)
    T, N = 200, 30
    R = rng.standard_normal((T, N)) * 0.01 + rng.standard_normal((T, 1)) * 0.005
    R[150:160] *= 6.0  # a burst of unusual moves
    df = pl.DataFrame(
        {
            "entity": np.repeat(np.arange(N), T),
            "time": np.tile(np.arange(T), N),
            "ret": R.T.reshape(-1),
        }
    )
    tb = turbulence(
        df, returns="ret", window=60, refit=10, entity="entity", time="time"
    )
    d = tb.get_column("turbulence").to_numpy()
    assert np.nanmean(d[150:160]) > 5 * np.nanmean(d[100:150])
    assert tb.get_column("turbulence_pct")[155] > 0.9


# --------------------------------------------------------------------------- #
# transformers
# --------------------------------------------------------------------------- #
def test_market_state_transformer_matches_the_function() -> None:
    ms = MarketState(
        "value", W, min_coverage=COV, features=("absorption_ratio", "avg_corr")
    )
    out = ms.fit_transform(PANEL).collect()
    ref = pn.covariance.market_state(
        PANEL, returns="value", window=W, min_coverage=COV,
        features=("absorption_ratio", "avg_corr"), broadcast=True,
    )  # fmt: skip
    assert out.equals(ref)
    assert MarketState.panel_safe is False and MarketState.leakage_safe is True


def test_transformers_fit_nothing_and_are_causal() -> None:
    train = PanelFrame(
        DF.filter(pl.col("time") <= TIMES[70]), entity="entity", time="time"
    )
    makers = (
        lambda: MarketState(
            "value", W, min_coverage=COV, features=("absorption_ratio",)
        ),
        lambda: Turbulence("value", W, min_coverage=COV, refit="1w"),
    )
    for make in makers:
        # fit learns nothing: fitting on a training fold or on everything is
        # the same transform
        a = make().fit(train).transform(PANEL).collect()
        b = make().fit(PANEL).transform(PANEL).collect()
        assert a.equals(b)
        assert a.height == DF.height

        def op(frame, _make=make):
            return _make().fit_transform(frame)

        assert_prefix_invariant(op, PANEL, tol=0.0)
        assert_no_lookahead(op, PANEL, tol=0.0)


def test_transformer_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="not found"):
        MarketState("nope", W).fit(PANEL)
    with pytest.raises(ValueError, match="not found"):
        Turbulence("nope", W).fit(PANEL)


def test_public_surface_is_lazy_and_complete() -> None:
    import subprocess
    import sys

    code = (
        "import sys, panelary as pn; "
        "assert 'panelary.covariance' not in sys.modules; "
        "cov = pn.covariance; "
        "names = ['estimate', 'rolling', 'market_state', 'market_loading', "
        "'turbulence', 'MarketState', 'Turbulence', 'Schedule', 'Refit', "
        "'CovEstimate']; "
        "assert all(hasattr(cov, n) for n in names); "
        "from panelary.core._schedule import Schedule; "
        "assert cov.Schedule is Schedule; "
        "from panelary.registry import registry; "
        "assert registry.get('market_state').namespace == 'covariance'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- #
# group variants
# --------------------------------------------------------------------------- #
def _sectored() -> pl.DataFrame:
    return DF.with_columns(
        (pl.col("entity").str.slice(-1).cast(pl.Int64) % 2)
        .cast(pl.String)
        .alias("sector")
    )


def test_group_turbulence_matches_each_group_alone() -> None:
    df = _sectored()
    kw = {"returns": "value", "window": W, "min_coverage": COV, "refit": "1w"}
    grouped = turbulence(
        df, group="sector", min_entities=4, entity="entity", time="time", **kw
    )
    assert set(grouped.get_column("sector").unique()) == {"0", "1"}
    for sec in ("0", "1"):
        alone = turbulence(
            df.filter(pl.col("sector") == sec),
            min_entities=4,
            entity="entity",
            time="time",
            **kw,
        ).drop_nulls("turbulence")
        got = grouped.filter(pl.col("sector") == sec).drop("sector")
        got = got.filter(pl.col("time").is_in(alone["time"].implode()))
        assert got.select("time", "turbulence", "n_scored").equals(
            alone.select("time", "turbulence", "n_scored")
        )


def test_group_market_loading_matches_each_group_alone() -> None:
    from panelary.covariance._state import market_loading

    df = _sectored()
    grouped = market_loading(
        df, returns="value", window=W, min_coverage=COV, group="sector",
        min_entities=4, entity="entity", time="time",
    )  # fmt: skip
    for sec in ("0", "1"):
        sub = df.filter(pl.col("sector") == sec)
        alone = market_loading(
            sub, returns="value", window=W, min_coverage=COV, min_entities=4,
            entity="entity", time="time",
        )  # fmt: skip
        got = sub.select("entity", "time").join(
            grouped, on=["entity", "time"], how="left"
        )
        assert got.get_column("market_loading").equals(
            alone.get_column("market_loading")
        )
