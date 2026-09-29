"""``rough_hurst`` and ``RFSVForecaster``: formulas, arguments and leak safety.

The accuracy claims (the plan's section 1e table, forecast skill) are Monte
Carlo and live in ``tests/test_rough_accuracy.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from panelary._internal._variogram import log_slope_weights, trailing_variogram
from panelary.core import PanelFrame
from panelary.econ.features import RFSVForecaster, rough_hurst
from panelary.econ.features._rough import _rfsv_apply, rfsv_weights
from panelary.testing import (
    assert_no_lookahead,
    assert_no_train_test_leak,
    assert_prefix_invariant,
)


def _panel(
    n: int = 400, entities: tuple[str, ...] = ("A", "B", "C"), seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, ent in enumerate(entities):
        m = n - 37 * i  # ragged lengths
        x = np.cumsum(rng.standard_normal(m)) * 0.2 + rng.standard_normal(m) * 0.1
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * m,
                    "time": np.arange(m, dtype=np.int64),
                    "logv": x,
                    "nv": np.full(m, 0.01) * (1 + rng.random(m)),
                }
            )
        )
    return pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=seed)


# --------------------------------------------------------------------------- #
# rough_hurst
# --------------------------------------------------------------------------- #
def test_rough_hurst_matches_the_numpy_formula() -> None:
    df = _panel()
    lags = (1, 2, 3, 5, 8)
    out = rough_hurst(
        df,
        entity="entity",
        time="time",
        log_variance="logv",
        noise_var="nv",
        window=60,
        lags=lags,
    )
    assert out.columns[-1] == "rough_hurst_60"
    w = log_slope_weights(lags)
    for ent in ("A", "C"):
        sub = out.filter(pl.col("entity") == ent)
        m = trailing_variogram(
            sub["logv"].to_numpy(), lags, 60, noise=sub["nv"].to_numpy()
        )
        with np.errstate(invalid="ignore"):
            want = 0.5 * np.log(m) @ w
        got = sub["rough_hurst_60"].to_numpy()
        assert np.isnan(got[:59]).all() and np.isfinite(got[59:]).all()
        np.testing.assert_allclose(got[59:], want[59:], rtol=1e-10)


def test_a_constant_noise_equals_a_constant_column() -> None:
    df = _panel().with_columns(c=pl.lit(0.02))
    kw: dict[str, Any] = {
        "entity": "entity",
        "time": "time",
        "log_variance": "logv",
        "window": 50,
    }
    a = rough_hurst(df, noise_var=0.02, **kw)["rough_hurst_50"].to_numpy()
    b = rough_hurst(df, noise_var="c", **kw)["rough_hurst_50"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12, equal_nan=True)


def test_noise_that_swamps_the_signal_gives_null() -> None:
    df = _panel()
    out = rough_hurst(
        df, entity="entity", time="time", log_variance="logv", noise_var=10.0, window=50
    )
    assert out["rough_hurst_50"].null_count() == out.height


def test_non_finite_proxy_values_are_missing() -> None:
    df = _panel(n=200, entities=("A",)).sort("time")
    holes = df.with_columns(
        pl.when(pl.col("time").is_in([100, 101]))
        .then(float("-inf"))
        .otherwise(pl.col("logv"))
        .alias("logv")
    )
    kw: dict[str, Any] = {
        "entity": "entity",
        "time": "time",
        "log_variance": "logv",
        "noise_var": 0.0,
        "window": 40,
        "min_periods": 30,
    }
    got = rough_hurst(holes, **kw)["rough_hurst_40"]
    nulled = rough_hurst(
        holes.with_columns(
            pl.when(pl.col("logv").is_infinite())
            .then(None)
            .otherwise(pl.col("logv"))
            .alias("logv")
        ),
        **kw,
    )["rough_hurst_40"]
    assert got.is_nan().sum() == 0
    assert got.equals(nulled)
    assert got.drop_nulls().len() > 150


def test_rough_hurst_is_leak_free() -> None:
    panel = PanelFrame(_panel(), entity="entity", time="time")

    def op(frame: Any) -> pl.DataFrame:
        return rough_hurst(
            frame,
            entity="entity",
            time="time",
            log_variance="logv",
            noise_var="nv",
            window=60,
            min_periods=40,
        )

    assert_no_lookahead(op, panel, tol=0.0)
    assert_prefix_invariant(op, panel, tol=0.0)


def test_rough_hurst_needs_an_explicit_noise_choice() -> None:
    df = _panel()
    kw: dict[str, Any] = {"entity": "entity", "time": "time", "log_variance": "logv"}
    with pytest.raises(TypeError):
        rough_hurst(df, **kw)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="not found"):
        rough_hurst(df, noise_var="missing", **kw)
    with pytest.raises(ValueError, match=">= 0"):
        rough_hurst(df, noise_var=-0.1, **kw)
    with pytest.raises(TypeError):
        rough_hurst(df, noise_var=True, **kw)
    with pytest.raises(ValueError, match="window"):
        rough_hurst(df, noise_var=0.0, window=10, lags=(1, 9), **kw)
    with pytest.raises(ValueError, match="min_periods"):
        rough_hurst(df, noise_var=0.0, window=50, min_periods=60, **kw)


# --------------------------------------------------------------------------- #
# RFSV weights and forecaster
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hurst", [0.05, 0.1, 0.3, 0.45])
@pytest.mark.parametrize("horizon", [1, 5])
def test_rfsv_weights_match_a_fine_quadrature(hurst: float, horizon: int) -> None:
    w = rfsv_weights(hurst, horizon, 30)
    a = hurst + 0.5
    raw = np.empty(30)
    for k in range(30):
        # midpoint rule on a geometric grid resolves the u^{-a} spike at 0
        lo = 1e-12 if k == 0 else k
        edges = (
            np.geomspace(lo, k + 1, 200_001)
            if k == 0
            else np.linspace(k, k + 1, 20_001)
        )
        mid = 0.5 * (edges[1:] + edges[:-1])
        body = np.sum(np.diff(edges) / ((mid + horizon) * mid**a))
        raw[k] = body + (lo ** (1 - a) / (1 - a) / horizon if k == 0 else 0.0)
    np.testing.assert_allclose(w, raw / raw.sum(), rtol=2e-5)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all(np.diff(w) < 0)


def test_smoother_volatility_leans_on_the_latest_value() -> None:
    """As H approaches 1/2 the forecast approaches the random walk's last value;
    rough volatility averages further back."""
    first = [rfsv_weights(h, 1, 100)[0] for h in (0.05, 0.1, 0.3, 0.45)]
    assert first == sorted(first)


def test_rfsv_filter_is_causal_and_renormalises_over_gaps() -> None:
    w = np.array([0.5, 0.3, 0.2])
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    got = _rfsv_apply(x, w)
    assert np.isnan(got[:2]).all()
    assert got[2] == pytest.approx(0.5 * 3 + 0.3 * 2 + 0.2 * 1)
    assert got[3] == pytest.approx((0.3 * 3 + 0.2 * 2) / 0.5)
    assert got[4] == pytest.approx((0.5 * 5 + 0.2 * 3) / 0.7)


def test_rfsv_forecaster_fits_pooled_and_per_entity() -> None:
    panel = PanelFrame(_panel(), entity="entity", time="time")
    pooled = RFSVForecaster(
        log_variance="logv", noise_var="nv", n_lags=40, min_train_rows=100
    ).fit(panel)
    assert pooled.pooled_hurst_ is not None and 0.01 <= pooled.pooled_hurst_ <= 0.49
    assert pooled.hurst_ == {}
    own = RFSVForecaster(
        log_variance="logv", noise_var="nv", n_lags=40, pooled=False, min_train_rows=350
    ).fit(panel)
    assert set(own.hurst_) == {"A", "B"}  # C has 326 rows: pooled fallback
    out = own.transform(panel).collect()
    assert out.columns[-1] == "rfsv_forecast"
    for ent in ("A", "B", "C"):
        sub = out.filter(pl.col("entity") == ent)
        f = sub["rfsv_forecast"]
        assert f.head(39).is_null().all() and f.slice(39).is_not_null().all()
        h = own.hurst_.get(ent, own.pooled_hurst_)
        want = _rfsv_apply(sub["logv"].to_numpy(), rfsv_weights(h, 1, 40))
        np.testing.assert_allclose(f.to_numpy()[39:], want[39:], rtol=1e-14)


def test_rfsv_h_is_fitted_on_the_training_fold_only() -> None:
    """Trap 12: H is frozen at fit; the transform reads each row's past only."""
    df = _panel()
    panel = PanelFrame(df, entity="entity", time="time")
    times = df["time"].unique().sort()
    split = (times.filter(times < 200), times.filter(times >= 200))

    def op(frame: PanelFrame) -> PanelFrame:
        train = frame.filter(pl.col("time") < 200)
        model = RFSVForecaster(
            log_variance="logv",
            noise_var="nv",
            n_lags=30,
            pooled=False,
            min_train_rows=150,
        )
        return model.fit(train).transform(frame)

    assert_no_train_test_leak(op, panel, split)
    fitted = RFSVForecaster(log_variance="logv", noise_var="nv", n_lags=30).fit(panel)
    assert_no_lookahead(fitted.transform, panel, tol=0.0)
    assert_prefix_invariant(fitted.transform, panel, tol=0.0)


def test_rfsv_argument_validation() -> None:
    with pytest.raises(ValueError, match="horizon"):
        RFSVForecaster(log_variance="x", noise_var=0.0, horizon=0)
    with pytest.raises(TypeError):
        RFSVForecaster(log_variance="x", noise_var=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hurst"):
        rfsv_weights(0.5, 1, 10)
    panel = PanelFrame(_panel(), entity="entity", time="time")
    with pytest.raises(ValueError, match="not positive"):
        RFSVForecaster(log_variance="logv", noise_var=50.0).fit(panel)
