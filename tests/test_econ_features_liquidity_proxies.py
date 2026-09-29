"""Low-frequency liquidity proxies (plan 5, M3).

``price_impact`` and ``pastor_stambaugh_gamma`` are checked window by window
against ``np.linalg.lstsq`` -- including an adversarial input whose mean is a
million standard deviations, where the naive ``E[xy] - E[x]E[y]`` loses about
four digits and the anchored moments must not. ``zero_return_share`` and
``fht_spread`` are checked against hand counts and, when scipy is installed,
against ``scipy.stats.norm.ppf``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary._internal._special import norm_ppf
from panelary.econ.features import (
    fht_spread,
    pastor_stambaugh_gamma,
    price_impact,
    zero_return_share,
)
from panelary.econ.features._liquidity import _fht_quantiles
from panelary.registry import registry


def _panel(n=120, entities=("A", "B"), seed=11, offset=0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in entities:
        for t in range(n):
            mkt = rng.normal(0, 0.01)
            r = 0.8 * mkt + rng.normal(0, 0.015)
            dv = math.exp(rng.normal(14, 1))
            flow = offset + rng.normal(0, 1.0)
            rows.append((e, t, r, mkt, dv, flow))
    return pl.DataFrame(
        rows, schema=["e", "t", "r", "m", "dv", "flow"], orient="row"
    ).with_columns(pl.col("r").round(3))  # rounding manufactures zero returns


def _lstsq_slope(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# --------------------------------------------------------------------------- #
# price_impact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("intercept", [True, False])
def test_price_impact_matches_lstsq_per_window(intercept) -> None:
    df = _panel()
    w = 15
    out = price_impact(
        df,
        entity="e",
        time="t",
        returns="r",
        dollar_volume="dv",
        window=w,
        intercept=intercept,
    )
    for e in ("A", "B"):
        sub = out.filter(pl.col("e") == e)
        r = sub["r"].to_numpy()
        x = np.sign(r) * np.sqrt(sub["dv"].to_numpy() * 1e-6)
        got = sub[f"price_impact_{w}"].to_numpy().astype(float)
        assert np.isnan(got[: w - 1]).all()
        for t in range(w - 1, len(r)):
            s = slice(t - w + 1, t + 1)
            X = np.column_stack([np.ones(w), x[s]]) if intercept else x[s][:, None]
            beta = _lstsq_slope(r[s], X)[-1]
            assert got[t] == pytest.approx(beta, rel=1e-10, abs=1e-13)


def test_price_impact_anchoring_survives_a_huge_mean() -> None:
    """Order flow with mean/sd = 1e6: the anchored moments keep ~1e-9 accuracy.

    (Without anchoring, E[xy] - E[x]E[y] cancels ~12 digits here.)
    """
    df = _panel(offset=1e6)
    w = 20
    out = price_impact(
        df, entity="e", time="t", returns="r", signed_volume="flow", window=w
    )
    sub = out.filter(pl.col("e") == "A")
    r, x = sub["r"].to_numpy(), sub["flow"].to_numpy()
    got = sub[f"price_impact_{w}"].to_numpy()
    for t in range(w - 1, len(r)):
        s = slice(t - w + 1, t + 1)
        xc = x[s] - x[s].mean()  # lstsq on centred data: the accurate reference
        beta = np.dot(xc, r[s] - r[s].mean()) / np.dot(xc, xc)
        assert got[t] == pytest.approx(beta, rel=1e-8, abs=1e-12)


def test_price_impact_sign_flow_slope_is_mechanically_positive() -> None:
    """The docstring's honesty note, as a test: sign(r) makes Cov(r, x) > 0."""
    out = price_impact(
        _panel(), entity="e", time="t", returns="r", dollar_volume="dv", window=30
    )
    vals = out["price_impact_30"].drop_nulls()
    assert (vals > 0).all()


def test_price_impact_is_null_without_order_flow_variance() -> None:
    df = _panel(n=10).with_columns(pl.lit(3.0).alias("flow"))
    out = price_impact(
        df, entity="e", time="t", returns="r", signed_volume="flow", window=5
    )
    assert out["price_impact_5"].null_count() == out.height


def test_price_impact_needs_exactly_one_flow_column() -> None:
    df = _panel(n=10)
    with pytest.raises(ValueError, match="exactly one"):
        price_impact(df, entity="e", time="t", returns="r")
    with pytest.raises(ValueError, match="exactly one"):
        price_impact(
            df,
            entity="e",
            time="t",
            returns="r",
            dollar_volume="dv",
            signed_volume="flow",
        )


# --------------------------------------------------------------------------- #
# pastor_stambaugh_gamma
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_ps_gamma_matches_lstsq_per_window(offset) -> None:
    df = _panel()
    if offset:  # adversarial: the dollar-volume regressor sits far from zero
        df = df.with_columns(pl.col("dv") + offset * 1e6)
    w = 21
    out = pastor_stambaugh_gamma(
        df,
        entity="e",
        time="t",
        returns="r",
        market_returns="m",
        dollar_volume="dv",
        window=w,
    )
    for e in ("A", "B"):
        sub = out.filter(pl.col("e") == e)
        r, m, dv = (sub[c].to_numpy() for c in ("r", "m", "dv"))
        ex = r - m
        y = ex[1:]
        x1 = r[:-1]
        x2 = np.sign(ex[:-1]) * dv[:-1] * 1e-6
        got = sub[f"ps_gamma_{w}"].to_numpy().astype(float)
        assert np.isnan(got[:w]).all()  # the first row has no lagged regressors
        for t in range(w, len(r)):
            s = slice(t - w, t)  # regression rows t-w+1 .. t (shifted by one)
            X = np.column_stack([np.ones(w), x1[s], x2[s]])
            gamma = _lstsq_slope(y[s], X)[2]
            assert got[t] == pytest.approx(gamma, rel=1e-9, abs=1e-15)


def test_ps_gamma_is_null_for_a_collinear_window() -> None:
    df = _panel(n=30).with_columns(pl.lit(0.001).alias("r"), pl.lit(0.0).alias("m"))
    out = pastor_stambaugh_gamma(
        df,
        entity="e",
        time="t",
        returns="r",
        market_returns="m",
        dollar_volume="dv",
        window=5,
    )
    assert out["ps_gamma_5"].null_count() == out.height


# --------------------------------------------------------------------------- #
# zero_return_share
# --------------------------------------------------------------------------- #
def test_zero_return_share_hand_counts() -> None:
    df = pl.DataFrame(
        {
            "e": ["a"] * 6,
            "t": range(6),
            "r": [0.0, 0.01, 0.0, None, 0.0, -0.02],
            "v": [100.0, 50.0, 0.0, 10.0, 5.0, None],
        }
    )
    out = zero_return_share(
        df, entity="e", time="t", returns="r", volume="v", window=3, min_periods=2
    )
    # rows with a return in each 3-row window, and the zeros among them
    assert out["zeros_3"].to_list() == [None, 0.5, 2 / 3, 0.5, 1.0, 0.5]
    # Zeros2: zero return AND positive volume, over rows with both (the last
    # window has only one such row, below min_periods)
    assert out["zeros2_3"].to_list() == [None, 0.5, 1 / 3, 0.0, 0.5, None]


def test_zero_return_share_tolerance_and_alias() -> None:
    df = pl.DataFrame({"e": ["a"] * 4, "t": range(4), "r": [1e-5, -2e-5, 0.1, 3e-6]})
    out = zero_return_share(
        df, entity="e", time="t", returns="r", window=4, tol=2e-5, alias="z"
    )
    assert out["z"][-1] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# fht_spread
# --------------------------------------------------------------------------- #
def test_fht_quantile_table_matches_norm_ppf() -> None:
    w = 12
    table = _fht_quantiles(w)
    for n in range(1, w + 1):
        for k in range(n + 1):
            v = table[n * (w + 1) + k]
            if k == n:
                assert np.isnan(v)
            else:
                assert v == norm_ppf((1 + k / n) / 2)


def test_fht_quantiles_match_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    w = 30
    table = _fht_quantiles(w)
    for n in range(1, w + 1):
        k = np.arange(n)
        np.testing.assert_allclose(
            table[n * (w + 1) + k],
            stats.norm.ppf((1 + k / n) / 2),
            rtol=1e-12,
            atol=1e-15,
        )


def test_fht_hand_value_and_all_zero_window() -> None:
    r = [0.0, 0.02, -0.01, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0]
    df = pl.DataFrame({"e": ["a"] * len(r), "t": range(len(r)), "r": r})
    out = fht_spread(df, entity="e", time="t", returns="r", window=5)["fht_5"]
    window = r[:5]
    sd = float(np.std(window, ddof=1))
    z = 2 / 5
    expected = 2 * sd * norm_ppf((1 + z) / 2)
    assert out[4] == pytest.approx(expected, rel=1e-13)
    assert out[:4].null_count() == 4
    assert out[9] is None  # the last window is all zeros: FHT undefined


def test_fht_is_zero_without_zero_returns() -> None:
    df = pl.DataFrame(
        {"e": ["a"] * 6, "t": range(6), "r": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]}
    )
    out = fht_spread(df, entity="e", time="t", returns="r", window=4)["fht_4"]
    assert out[3:].to_list() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Validation and registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("fn", "kwargs", "match"),
    [
        (price_impact, {"dollar_volume": "dv", "window": 1}, "window"),
        (price_impact, {"dollar_volume": "dv", "volume_scale": 0.0}, "volume_scale"),
        (price_impact, {"dollar_volume": "nope"}, "not found"),
        (
            pastor_stambaugh_gamma,
            {"market_returns": "m", "dollar_volume": "dv", "window": 2},
            "window",
        ),
        (
            pastor_stambaugh_gamma,
            {
                "market_returns": "m",
                "dollar_volume": "dv",
                "window": 5,
                "min_periods": 6,
            },
            "min_periods",
        ),
        (zero_return_share, {"tol": -1.0}, "tol"),
        (fht_spread, {"window": 1}, "window"),
        (fht_spread, {"tol": float("nan")}, "tol"),
    ],
)
def test_bad_arguments_raise(fn, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        fn(_panel(n=10), entity="e", time="t", returns="r", **kwargs)


def test_registry_specs() -> None:
    tiers = {
        "price_impact": "C",
        "pastor_stambaugh_gamma": "C",
        "zero_return_share": "B",
        "fht_spread": "B",
    }
    for name, tier in tiers.items():
        spec = registry.get(name)
        assert (spec.namespace, spec.safe_scope, spec.tier) == ("econ", "rowwise", tier)
    assert registry.audit()["quadratic_cost_in_core_tier"] == []
