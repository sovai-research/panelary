"""The frame-level API of :mod:`panelary.depend`: dependence, by_entity,
aggregate, pooled, and the fixed result schema."""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp
from panelary.core.panel_frame import PanelFrame

SCHEMA = dict(dp.RESULT_SCHEMA)


def _panel(
    n_ent: int = 8, t_len: int = 200, seed: int = 0, sign: np.ndarray | None = None
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_ent, t_len))
    s = np.ones(n_ent) if sign is None else sign
    y = s[:, None] * x + 1.0 * rng.standard_normal((n_ent, t_len))
    return pl.DataFrame(
        {
            "ticker": np.repeat([f"e{i:02d}" for i in range(n_ent)], t_len),
            "date": np.tile(np.arange(t_len), n_ent),
            "x": x.ravel(),
            "y": y.ravel(),
            "noise": rng.standard_normal(n_ent * t_len),
        }
    )


def test_fixed_schema_and_concat() -> None:
    df = _panel()
    a = dp.dependence(df, "x", "y", entity="ticker", time="date", method="xi")
    b = dp.dependence(
        df,
        "x",
        "y",
        entity="ticker",
        time="date",
        method=["dcor", "gcmi"],
        null="asymptotic",
    )
    for out in (a, b):
        assert out.columns == ["x", "y", *SCHEMA]
        for name, dtype in SCHEMA.items():
            assert out.schema[name] == dtype, name
    both = pl.concat([a, b])
    assert both.height == 3
    assert a["null_method"][0] == "common-time"
    assert a["n_entities"][0] == 8 and a["coverage"][0] == 1.0
    assert a["direction"][0] == "x->y" and b["direction"][0] == "symmetric"


def test_inputs_panelframe_lazy_and_float32() -> None:
    df = _panel().with_columns(pl.col("x").cast(pl.Float32))
    pf = PanelFrame(df, entity="ticker", time="date")
    r1 = dp.dependence(pf, "x", "y", method="spearman", null="asymptotic")
    r2 = dp.dependence(
        df.lazy(),
        "x",
        "y",
        entity="ticker",
        time="date",
        method="spearman",
        null="asymptotic",
    )
    assert r1["estimate"][0] == r2["estimate"][0]
    with pytest.raises(ValueError, match="conflicts"):
        dp.dependence(pf, "x", "y", entity="other")


def test_determinism_and_seed() -> None:
    df = _panel()
    kw = {
        "entity": "ticker",
        "time": "date",
        "method": "xi",
        "null": "common-time",
        "n_resamples": 49,
    }
    a = dp.dependence(df, "x", "noise", seed=1, **kw)
    b = dp.dependence(df, "x", "noise", seed=1, **kw)
    assert a.equals(b)
    assert a["seed"][0] == 1 and a["n_resamples"][0] == 49


def test_single_series_paths() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    df = pl.DataFrame({"x": x, "y": np.cos(2 * x) + 0.2 * rng.standard_normal(300)})
    out = dp.dependence(df, "x", "y", method=["xi", "spearman"])
    xi_row = out.filter(pl.col("method") == "xi")
    assert xi_row["null_method"][0] == "asymptotic" and xi_row["p_value"][0] < 1e-10
    assert xi_row["n_entities"][0] == 1
    with pytest.raises(ValueError, match="entity"):
        dp.dependence(df, "x", "y", null="entity")
    ct = dp.dependence(df, "x", "y", null="common-time", n_resamples=19)
    assert ct["null_method"][0] == "block"
    assert any("common-time" in w for w in ct["warnings"][0])


def test_min_obs_gate() -> None:
    df = pl.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) ** 2})
    out = dp.dependence(df, "x", "y", method="xi")
    assert math.isnan(out["estimate"][0])
    assert "min_obs" in out["warnings"][0][0]
    ok = dp.dependence(df, "x", "y", method="xi", min_obs=10)
    assert ok["estimate"][0] > 0.5


def test_unbalanced_panel_gate_excludes_short_entities() -> None:
    df = _panel(n_ent=5, t_len=120)
    df = df.filter(~((pl.col("ticker") == "e00") & (pl.col("date") > 20)))
    out = dp.dependence(
        df, "x", "y", entity="ticker", time="date", method="xi", null="asymptotic"
    )
    assert out["n_entities"][0] == 4 and out["coverage"][0] == pytest.approx(0.8)
    assert any("excluded" in w for w in out["warnings"][0])


def test_gcmi_honesty_note() -> None:
    df = _panel(n_ent=1)
    out = dp.dependence(df, "x", "y", method="gcmi")
    assert any("Spearman in a hat" in w for w in out["warnings"][0])


def test_tail_dependence_expands_to_two_rows() -> None:
    df = _panel(n_ent=1, t_len=800)
    out = dp.dependence(df, "x", "y", method="tail_dependence", q=0.1)
    assert out["method"].to_list() == ["tail_lower", "tail_upper"]
    with pytest.raises(ValueError, match="unknown dependence"):
        dp.dependence(df, "x", "y", method="mic")


def test_lag_pairs_x_leading() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    y = np.r_[rng.standard_normal(2), np.abs(x[:-2])] + 0.1 * rng.standard_normal(500)
    df = pl.DataFrame({"t": np.arange(500), "x": x, "y": y})
    at2 = dp.dependence(df, "x", "y", time="t", lag=2, method="xi")
    at0 = dp.dependence(df, "x", "y", time="t", lag=0, method="xi")
    assert at2["estimate"][0] > 0.5 > at0["estimate"][0]
    assert at2["lag"][0] == 2 and at2["n_obs"][0] == 498


def _garch(seed: int, n: int, a: float = 0.2, b: float = 0.75) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = 0.05 / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 0.05 + a * r[t] ** 2 + b * s2
    return r


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_devol_reports_both_numbers_and_removes_vol_clustering(seed: int) -> None:
    """GARCH(1,1) returns: lag-1 dependence of r_t on r_{t-1} is volatility
    clustering. devol= reports raw and devolatilised side by side.

    Measured (a=0.2, b=0.75, n=5000, dcor): a 20-date causal window removes
    75-85% of the estimate; a 60-date window only ~30% -- the rolling SD lags
    the true volatility. Neither removes all of it.
    """
    df = pl.DataFrame({"t": np.arange(5000), "r": _garch(seed, 5000)})
    out = dp.dependence(
        df, "r", "r", time="t", lag=1, method="dcor", devol=20, null="asymptotic"
    )
    assert out["transform"].to_list() == ["none", "devol(window=20)"]
    raw, dev = out["estimate"].to_list()
    assert out["p_value"][0] < 1e-3  # raw: "significant nonlinear dependence"
    assert dev < 0.35 * raw  # most of it was volatility clustering


def test_devolatilise_is_causal() -> None:
    v = np.arange(1.0, 21.0) ** 1.5
    d = dp.devolatilise(v, 5)
    assert np.isnan(d[:5]).all()
    assert d[5] == pytest.approx(v[5] / np.std(v[0:5], ddof=1))
    d_short = dp.devolatilise(v[:12], 5)
    np.testing.assert_array_equal(d_short, d[:12])  # prefix-invariant


def test_heterogeneity_and_aggregate() -> None:
    same = _panel(n_ent=10, seed=4)
    mixed = _panel(n_ent=10, seed=4, sign=np.array([1, -1] * 5, dtype=float))
    kw = {"entity": "ticker", "time": "date", "method": "pearson", "null": "asymptotic"}
    h_same = dp.dependence(same, "x", "y", **kw)["heterogeneity"][0]
    h_mixed = dp.dependence(mixed, "x", "y", **kw)["heterogeneity"][0]
    assert h_same < 0.3 < 0.9 < h_mixed
    per = dp.by_entity(same, "x", "y", entity="ticker", time="date", method="pearson")
    assert per.height == 10 and per.columns[:3] == ["ticker", "x", "y"]
    agg = dp.aggregate(per)
    assert agg.estimate == pytest.approx(
        dp.dependence(same, "x", "y", **kw)["estimate"][0], abs=1e-12
    )
    assert agg.n_entities == 10 and 0.0 <= agg.i2 <= 1.0 and agg.how == "fisher"
    for how in ("precision", "mean", "median"):
        assert 0.5 < dp.aggregate(per, how=how).estimate < 0.9
    with pytest.raises(ValueError):
        dp.aggregate(pl.concat([per, per.with_columns(pl.lit("xi").alias("method"))]))
    with pytest.raises(ValueError, match="panel-level"):
        dp.by_entity(same, "x", "y", entity="ticker", time="date", null="common-time")


def test_pooled_transform_is_reported() -> None:
    df = _panel(n_ent=6, seed=5)
    out = pl.concat(
        [
            dp.pooled(
                df,
                "x",
                "y",
                entity="ticker",
                time="date",
                method="spearman",
                demean=d,
                null="asymptotic",
            )
            for d in ("none", "entity", "time", "two-way")
        ]
    )
    assert out["transform"].to_list() == [
        "pooled(demean=none)",
        "pooled(demean=entity)",
        "pooled(demean=time)",
        "pooled(demean=two-way)",
    ]
    assert out["n_obs"].to_list() == [1200] * 4
    via = dp.dependence(
        df,
        "x",
        "y",
        entity="ticker",
        time="date",
        by="pooled",
        method="spearman",
        null="asymptotic",
    )
    assert via["estimate"][0] == out["estimate"][0]
    with pytest.raises(ValueError):
        dp.pooled(df, "x", "y", entity="ticker", time="date", demean="three-way")


def test_simpson_pooled_vs_within() -> None:
    """Between-entity levels drive pooled-raw; within-entity is the opposite sign."""
    rng = np.random.default_rng(6)
    rows = []
    for i in range(6):
        x = i + rng.standard_normal(150)
        y = 3 * i - 0.8 * (x - i) + 0.3 * rng.standard_normal(150)
        rows.append(pl.DataFrame({"e": [i] * 150, "t": np.arange(150), "x": x, "y": y}))
    df = pl.concat(rows)
    kw = {"entity": "e", "time": "t", "method": "spearman", "null": "asymptotic"}
    raw = dp.pooled(df, "x", "y", **kw)["estimate"][0]
    within = dp.dependence(df, "x", "y", **kw)["estimate"][0]
    dem = dp.pooled(df, "x", "y", demean="entity", **kw)["estimate"][0]
    assert raw > 0.5 and within < -0.5 and dem < -0.5


def test_independence_test_array_api() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(400)
    out = dp.independence_test(x, np.sin(3 * x), method="dcor")
    assert out.height == 1 and out["p_value"][0] < 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        null = dp.independence_test(x, rng.standard_normal(400), method="hoeffding")
    assert 0.0 < null["p_value"][0] <= 1.0
