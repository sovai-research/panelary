"""Random features with direct links and a past-only bandwidth (`panelary.embed.RandomFourierFeatures`).

Contracts touched: ``leakage_safe`` -- the scaler and the median-heuristic
bandwidth are *expanding* statistics over the fitted rows, so even a fit on
the whole sample gives in-sample features free of later information (the
perturbation suites in ``test_embed_leakage.py`` pin that down); the direct
links are always present (build contract section 3.5).
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from panelary.embed import QuantEmbedder, RandomFourierFeatures
from panelary.embed._rff import _RunningMedian, orthogonal_gaussian

E, T = "entity", "time"


def _panel(n_ent: int = 6, n_time: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            E: np.repeat([f"e{i}" for i in range(n_ent)], n_time),
            T: np.tile(np.arange(n_time, dtype=np.int64), n_ent),
            "a": rng.standard_normal(n_ent * n_time) * 3 + 1,
            "b": rng.random(n_ent * n_time),
        }
    )


def _mat(frame: pl.DataFrame, col: str = "rff") -> np.ndarray:
    return np.asarray(frame[col].to_list(), dtype=np.float64)


def test_direct_links_are_the_expanding_standardised_inputs():
    df = _panel()
    rff = RandomFourierFeatures(
        n_components=10,
        columns=["a", "b"],
        min_periods=2,
        dtype="float64",
        entity=E,
        time=T,
    )
    out = rff.fit(df).transform(df).collect().sort(T, E)
    Z = _mat(out)[:, :2]
    src = df.sort(T, E)
    for t in (0, 7, 39):
        past = src.filter(pl.col(T) <= t).select("a", "b").to_numpy()
        now = src.filter(pl.col(T) == t).select("a", "b").to_numpy()
        expect = (now - past.mean(axis=0)) / past.std(axis=0)
        np.testing.assert_allclose(Z[out[T].to_numpy() == t], expect, rtol=1e-9)
    assert "direct" not in inspect.signature(RandomFourierFeatures).parameters


def test_schedule_is_expanding_so_a_longer_fit_leaves_early_entries_unchanged():
    df = _panel()
    short = RandomFourierFeatures(columns=["a", "b"], entity=E, time=T).fit(
        df.filter(pl.col(T) <= 20)
    )
    long = RandomFourierFeatures(columns=["a", "b"], entity=E, time=T).fit(df)
    k = short.schedule_time_.shape[0]
    for name in (
        "schedule_mean_",
        "schedule_std_",
        "schedule_count_",
        "schedule_sigma_",
    ):
        np.testing.assert_array_equal(getattr(short, name), getattr(long, name)[:k])
    np.testing.assert_array_equal(short.weights_, long.weights_)


def test_fit_is_independent_of_row_order():
    df = _panel()
    a = RandomFourierFeatures(columns=["a", "b"], entity=E, time=T).fit(df)
    b = RandomFourierFeatures(columns=["a", "b"], entity=E, time=T).fit(
        df.sample(fraction=1.0, shuffle=True, seed=1)
    )
    assert a.get_state().fingerprint() == b.get_state().fingerprint()


def test_gaussian_features_approximate_the_kernel():
    df = _panel(n_ent=2, n_time=40)
    rff = RandomFourierFeatures(
        n_components=20000,
        scales=(1.0,),
        columns=["a", "b"],
        min_periods=2,
        dtype="float64",
        entity=E,
        time=T,
    ).fit(df)
    out = rff.transform(df.filter(pl.col(T) == 39)).collect()
    M = _mat(out)
    Z, F = M[:, :2], M[:, 2:]
    sigma = rff.schedule_sigma_[-1]
    approx = float(F[0] @ F[1])
    exact = float(np.exp(-np.sum((Z[0] - Z[1]) ** 2) / (2 * sigma**2)))
    assert abs(approx - exact) < 0.03


def test_arccos_features_approximate_the_arc_cosine_kernel():
    df = _panel(n_ent=2, n_time=40)
    rff = RandomFourierFeatures(
        kernel="arccos",
        n_components=40000,
        columns=["a", "b"],
        min_periods=2,
        dtype="float64",
        entity=E,
        time=T,
    ).fit(df)
    M = _mat(rff.transform(df.filter(pl.col(T) == 39)).collect())
    Z, F = M[:, :2], M[:, 2:]
    nx, ny = np.linalg.norm(Z[0]), np.linalg.norm(Z[1])
    theta = np.arccos(np.clip(Z[0] @ Z[1] / (nx * ny), -1, 1))
    exact = nx * ny * (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / np.pi
    assert abs(float(F[0] @ F[1]) - exact) / max(exact, 1e-3) < 0.05


def test_orthogonal_blocks_are_orthogonal():
    W = orthogonal_gaussian(4, 10, np.random.default_rng(0))
    assert W.shape == (4, 10)
    G = W[:, :4].T @ W[:, :4]
    np.testing.assert_allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-10)


def test_running_median():
    rm = _RunningMedian()
    vals = np.random.default_rng(0).random(101)
    for i, v in enumerate(vals):
        rm.push(float(v))
        assert rm.median() == pytest.approx(float(np.median(vals[: i + 1])))


def test_rows_before_min_periods_or_before_the_fit_are_null():
    df = _panel(n_ent=3)
    rff = RandomFourierFeatures(
        n_components=10, columns=["a", "b"], min_periods=9, entity=E, time=T
    )
    rff.fit(df.filter(pl.col(T) >= 5))
    out = rff.transform(df).collect().sort(T, E)
    nulls = out.filter(pl.col("rff").is_null())[T].unique().sort().to_list()
    # Dates 0-4 precede the fit; at date 5 only 3 rows exist, at 6 six, at 7 nine.
    assert nulls == [0, 1, 2, 3, 4, 5, 6]


def test_array_input_and_width_checks():
    df = _panel(n_ent=3, n_time=40)
    emb = QuantEmbedder(window=8, columns="a", warmup="drop", entity=E, time=T)
    with pytest.warns(UserWarning):
        frame = emb.fit_transform(df).collect()
    rff = RandomFourierFeatures(
        n_components=10, columns=["a_quant"], min_periods=2, entity=E, time=T
    ).fit(frame)
    out = rff.transform(frame).collect()
    assert out.schema["rff"] == pl.Array(pl.Float32, emb.n_outputs + 10)
    with pytest.raises(TypeError, match="time dtype"):
        rff.transform(frame.with_columns(pl.col(T).cast(pl.Float64)))


def test_state_roundtrip_and_invalid_parameters():
    df = _panel()
    rff = RandomFourierFeatures(
        n_components=10, columns=["a", "b"], entity=E, time=T
    ).fit(df)
    again = RandomFourierFeatures.from_state(rff.get_state())
    assert again.transform(df).collect().equals(rff.transform(df).collect())
    with pytest.raises(ValueError):
        RandomFourierFeatures(kernel="laplace")
    with pytest.raises(ValueError):
        RandomFourierFeatures(scales=(1.0, -2.0))
    with pytest.raises(ValueError):
        RandomFourierFeatures(n_components=3)  # fewer than the 5 default scales
