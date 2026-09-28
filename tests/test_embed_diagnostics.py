"""Guardrail harness (`panelary.embed._diagnostics`).

Contracts touched: the section-5 guardrails. The reversal check must pass an
honest leak-safe embedding pipeline and fail a mechanical-momentum one; the
baselines must only read targets already realised (``leakage_safe``); the
null panels must keep the real panel's shape and missingness.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import panelary.embed._diagnostics as diag
from panelary.embed import (
    PreValidatedRidge,
    QuantEmbedder,
    baseline_report,
    mechanical_baseline,
    naive_baselines,
    null_distribution,
    null_panel,
    reversal_check,
)
from panelary.testing import assert_no_lookahead

E, T = "entity", "time"


def _returns(n_ent: int = 15, n_time: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.repeat(rng.uniform(0.5, 2.0, n_ent), n_time)
    return pl.DataFrame(
        {
            E: np.repeat(np.arange(n_ent), n_time),
            T: np.tile(np.arange(n_time, dtype=np.int64), n_ent),
            "r": rng.standard_normal(n_ent * n_time) * vol,
        }
    )


def _embedding_pipeline(
    train: pl.DataFrame, test: pl.DataFrame, target: str
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = QuantEmbedder(window=8, columns="r", warmup="null", entity=E, time=T).fit(
            train
        )
        full = pl.concat([train.drop(target), test], how="diagonal_relaxed")
        emb = q.transform(full).collect()
        tr = emb.join(train.select(E, T, target), on=[E, T])
        model = PreValidatedRidge().fit_frame(tr, features=["r_quant"], target=target)
        te = test.select(E, T).join(emb, on=[E, T], how="left")
        return model.predict_frame(te, features=["r_quant"])


def test_reversal_check_passes_an_honest_pipeline():
    rep = reversal_check(
        _embedding_pipeline, _returns(), return_col="r", entity=E, time=T
    )
    assert rep.passed
    assert rep.learned_coefs[0] < 0 and rep.signal_corr > 0.9
    assert rep.true_coefs[0] < 0 and rep.n_test > 0


def test_reversal_check_fails_mechanical_momentum():
    def momentum(train: pl.DataFrame, test: pl.DataFrame, target: str) -> np.ndarray:
        return test["r"].to_numpy() / test["r"].std()

    rep = reversal_check(momentum, _returns(), return_col="r", entity=E, time=T)
    assert not rep.passed and rep.learned_coefs[0] > 0


def test_reversal_check_validates_inputs():
    with pytest.raises(ValueError, match="reversal"):
        reversal_check(
            lambda a, b, c: np.zeros(b.height),
            _returns(),
            return_col="r",
            coefs=(-0.5, 0.1),
            entity=E,
            time=T,
        )
    with pytest.raises(ValueError, match="predictions"):
        reversal_check(
            lambda a, b, c: np.zeros(3), _returns(), return_col="r", entity=E, time=T
        )


def test_mechanical_baseline_is_the_vol_scaled_ewma_of_realised_targets():
    df = _returns(n_ent=2, n_time=60).with_columns(
        pl.col("r").shift(-1).over(E).alias("y")
    )
    out = mechanical_baseline(
        df, "y", halflife=5, vol_window=10, horizon=1, entity=E, time=T
    )
    one = df.filter(pl.col(E) == 0).sort(T)
    y = one["y"].to_numpy()
    sd = one["y"].rolling_std(10, min_samples=5).to_numpy()
    z = np.r_[np.nan, (y / sd)[:-1]]
    alpha = 1 - np.exp(np.log(0.5) / 5)
    num = den = 0.0
    expect = np.full(60, np.nan)
    for t in range(60):
        num *= 1 - alpha
        den *= 1 - alpha
        if np.isfinite(z[t]):
            num += z[t]
            den += 1.0
        if den > 0:
            expect[t] = num / den
    got = out.filter(pl.col(E) == 0).sort(T)["mechanical_baseline"].to_numpy()
    ok = np.isfinite(expect)
    np.testing.assert_allclose(got[ok], expect[ok], rtol=1e-10)


@pytest.mark.parametrize("which", ["mechanical", "naive"])
def test_baselines_only_read_realised_targets(which):
    df = _returns(n_ent=3, n_time=40).with_columns(pl.col("r").alias("y"))

    def op(frame: pl.DataFrame) -> pl.DataFrame:
        if which == "mechanical":
            return mechanical_baseline(
                frame, "y", halflife=5, vol_window=6, horizon=2, entity=E, time=T
            )
        return naive_baselines(frame, "y", horizon=2, season=5, entity=E, time=T)

    # The target at t is realised at t + horizon: perturbing y after `cut`
    # may move a baseline only after cut + horizon.
    for cut in (10, 25):
        shifted = lambda f, c=cut: op(f).filter(pl.col(T) <= c + 2)  # noqa: E731
        assert_no_lookahead(shifted, df, entity=E, time=T, cut=cut)


def test_naive_baselines_and_validation():
    df = _returns(n_ent=2, n_time=10)
    out = (
        naive_baselines(df, "r", horizon=1, season=3, entity=E, time=T)
        .filter(pl.col(E) == 1)
        .sort(T)
    )
    r = df.filter(pl.col(E) == 1).sort(T)["r"].to_numpy()
    np.testing.assert_array_equal(out["naive_last"].to_numpy()[1:], r[:-1])
    np.testing.assert_array_equal(out["naive_seasonal"].to_numpy()[3:], r[:-3])
    with pytest.raises(ValueError, match="season"):
        naive_baselines(df, "r", horizon=3, season=2, entity=E, time=T)


def test_baseline_report_always_scores_both():
    df = _returns().with_columns(pl.col("r").shift(-1).over(E).alias("y"))
    df = df.with_columns((pl.col("y") * 0.5 + pl.col("r") * 0.1).alias("oracle"))
    rep = baseline_report(
        df, target="y", predictions={"leaky_oracle": "oracle"}, entity=E, time=T
    )
    assert rep["kind"].to_list() == ["model", "baseline", "baseline"]
    assert set(rep["forecast"]) == {"leaky_oracle", "naive_last", "mechanical_baseline"}
    oracle = rep.row(0, named=True)
    assert oracle["ic_mean"] > 0.5 and oracle["r2_oos"] > 0.3


@pytest.mark.parametrize("method", ["sign", "phase"])
def test_null_panels_keep_shape_order_and_missingness(method):
    df = _returns(n_ent=4, n_time=50).with_columns(
        pl.when(pl.col(T) % 7 == 0).then(None).otherwise(pl.col("r")).alias("r")
    )
    df = df.with_columns((pl.col("r") * 2).alias("s"))  # same missingness as r
    df = df.sample(fraction=1.0, shuffle=True, seed=0)
    null = null_panel(df, ["r", "s"], method=method, seed=1, entity=E, time=T)
    assert null.columns == df.columns and null.height == df.height
    assert null.select(E, T).equals(df.select(E, T))
    np.testing.assert_array_equal(
        null["r"].is_null().to_numpy(), df["r"].is_null().to_numpy()
    )
    assert not null["r"].equals(df["r"])
    # joint=True keeps the contemporaneous relation s = 2 r exactly.
    ok = null["r"].is_not_null()
    np.testing.assert_allclose(
        null.filter(ok)["s"].to_numpy(),
        2 * null.filter(ok)["r"].to_numpy(),
        rtol=1e-9,
        atol=1e-12,
    )
    if method == "sign":
        # |x - mean| about the original mean is preserved (vol clustering survives).
        for e in range(4):
            a = df.filter((pl.col(E) == e) & ok)["r"].to_numpy()
            b = null.filter((pl.col(E) == e) & null["r"].is_not_null())["r"].to_numpy()
            np.testing.assert_allclose(
                np.sort(np.abs(a - a.mean())), np.sort(np.abs(b - a.mean())), atol=1e-9
            )
    else:
        a = df.filter(pl.col(E) == 0).sort(T)["r"].drop_nulls().to_numpy()
        b = null.filter(pl.col(E) == 0).sort(T)["r"].drop_nulls().to_numpy()
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(a - a.mean())),
            np.abs(np.fft.rfft(b - b.mean())),
            atol=1e-8,
        )


def test_null_distribution_and_docstring_disclaimer():
    df = _returns(n_ent=5, n_time=40)
    rep = null_distribution(
        lambda f: float(f["r"].abs().mean()), df, ["r"], n_draws=4, entity=E, time=T
    )
    assert rep.null.shape == (4,) and 0 < rep.p_value <= 1
    # The sign null preserves |x - mean|, so this score is (nearly) unchanged.
    assert np.allclose(rep.null, rep.observed, rtol=0.05)
    assert "Sharpe 35" in (diag.__doc__ or "") and "search" in (diag.__doc__ or "")
    with pytest.raises(ValueError):
        null_panel(df, ["r"], method="bootstrap", entity=E, time=T)
