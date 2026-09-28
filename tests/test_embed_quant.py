"""QUANT over trailing windows (`panelary.embed.QuantEmbedder`).

Contracts touched: ``panel_safe`` (a window never spans two entities; input
order does not matter), ``leakage_safe`` (strictly trailing windows; the
leak suites live in ``test_embed_leakage.py`` / ``test_embed_prefix_invariance.py``)
and the ``fit_is_empty = True`` claim (see ``test_embed_stateless.py``).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import polars as pl
import pytest

from panelary.embed import (
    EmbeddingWarmupWarning,
    QuantEmbedder,
    dyadic_intervals,
    quant_features,
)

E, T = "entity", "time"


def _panel(n_entities: int = 3, n_time: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for e in range(n_entities):
        n = n_time - 5 * e
        parts.append(
            pl.DataFrame(
                {
                    E: np.full(n, f"e{e}"),
                    T: np.arange(n, dtype=np.int64) + 2 * e,
                    "x": rng.standard_normal(n).cumsum() * (1 + e),
                    "y": rng.standard_normal(n),
                }
            )
        )
    return pl.concat(parts)


def _reference(window: np.ndarray, depth: int = 6) -> np.ndarray:
    """Brute force, one window at a time, with np.quantile and np.convolve."""
    x = np.asarray(window, dtype=np.float64)
    d = np.diff(x)
    padded = np.concatenate([[d[0], d[0]], d, [d[-1], d[-1]]])
    reps = {
        "x": x,
        "dx": np.convolve(padded, np.ones(5) / 5.0, mode="valid"),
        "d2x": np.diff(x, n=2),
        "fft": np.abs(np.fft.rfft(x)),
    }
    feats: list[float] = []
    for r in ("x", "dx", "d2x", "fft"):
        v = reps[r]
        for iv in dyadic_intervals(v.shape[0], depth):
            seg = v[iv.start : iv.stop]
            k = max(1, seg.shape[0] // 4)
            q = np.quantile(seg, (2 * np.arange(k) + 1) / (2 * k))
            q[1::2] -= seg.mean()
            feats.extend(q.tolist())
    return np.asarray(feats)


@pytest.mark.parametrize("window", [8, 16, 23, 64])
def test_matches_bruteforce_reference(window):
    rng = np.random.default_rng(window)
    W = rng.standard_normal((5, window)).cumsum(axis=1)
    F = quant_features(W)
    for i in range(W.shape[0]):
        np.testing.assert_allclose(F[i], _reference(W[i]), rtol=1e-11, atol=1e-11)


def test_rows_are_independent_of_the_batch():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((50, 32))
    full = quant_features(W)
    np.testing.assert_array_equal(quant_features(W[:3]), full[:3])
    np.testing.assert_array_equal(quant_features(W[::-1])[::-1], full)


def test_transform_matches_windows_of_the_series():
    df = _panel()
    q = QuantEmbedder(
        window=16, columns="x", warmup="null", dtype="float64", entity=E, time=T
    )
    out = q.fit_transform(df).collect().sort(E, T)
    for ent in ("e0", "e2"):
        x = df.filter(pl.col(E) == ent).sort(T)["x"].to_numpy()
        got = out.filter(pl.col(E) == ent)["x_quant"]
        assert got[:15].null_count() == 15
        for t in (15, 30, x.shape[0] - 1):
            np.testing.assert_allclose(
                np.asarray(got[t]),
                _reference(x[t - 15 : t + 1]),
                rtol=1e-11,
                atol=1e-11,
            )


def test_amplitude_is_retained():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((4, 32))
    np.testing.assert_allclose(
        quant_features(3.0 * W), 3.0 * quant_features(W), rtol=1e-12
    )


def test_class_attributes_and_no_center_argument():
    assert QuantEmbedder.fit_is_empty is True
    assert QuantEmbedder.is_cross_sectional is False
    assert QuantEmbedder.panel_safe is True and QuantEmbedder.leakage_safe is True
    assert "center" not in inspect.signature(QuantEmbedder).parameters


def test_widths():
    assert (
        QuantEmbedder(window=64).n_outputs == quant_features(np.zeros((1, 64))).shape[1]
    )
    assert (
        QuantEmbedder(window=32, representations=("x",)).n_outputs
        < QuantEmbedder(window=32).n_outputs
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 3},
        {"representations": ("x", "bogus")},
        {"representations": ()},
        {"warmup": "fill"},
        {"output": "wide"},
        {"dtype": "int8"},
        {"interval_depth": 0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        QuantEmbedder(**kwargs)


def test_warmup_drop_warns_and_names_short_entities():
    df = pl.concat(
        [
            _panel(),
            pl.DataFrame(
                {
                    E: ["tiny"] * 3,
                    T: [0, 1, 2],
                    "x": [1.0, 2.0, 3.0],
                    "y": [0.0, 0.0, 0.0],
                }
            ),
        ]
    )
    q = QuantEmbedder(window=16, columns="x", entity=E, time=T)
    with pytest.warns(EmbeddingWarmupWarning, match="'tiny'"):
        out = q.fit_transform(df).collect()
    assert "tiny" not in out[E].to_list()
    assert out["x_quant"].null_count() == 0
    counts = dict(
        zip(
            *df.group_by(E).len().sort(E).to_dict(as_series=False).values(), strict=True
        )
    )
    got = dict(
        zip(
            *out.group_by(E).len().sort(E).to_dict(as_series=False).values(),
            strict=True,
        )
    )
    for ent, n in got.items():
        assert n == counts[ent] - 15


def test_missing_values_null_the_row_and_are_never_imputed():
    df = _panel().with_columns(
        pl.when((pl.col(E) == "e0") & (pl.col(T) == 30))
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    q = QuantEmbedder(window=8, columns="x", warmup="null", entity=E, time=T)
    out = q.fit_transform(df).collect().filter(pl.col(E) == "e0").sort(T)
    col = out["x_quant"]
    # Rows 30..37 see the gap; row 29 and 38 do not.
    assert col[30:38].null_count() == 8
    assert col[29] is not None and col[38] is not None


def test_input_order_and_entity_isolation():
    df = _panel()
    q = QuantEmbedder(window=8, columns=["x", "y"], warmup="null", entity=E, time=T)
    base = q.fit_transform(df).collect().sort(E, T)
    shuffled = (
        q.transform(df.sample(fraction=1.0, shuffle=True, seed=5)).collect().sort(E, T)
    )
    assert base.equals(shuffled)
    # Changing one entity never moves another's rows.
    bumped = df.with_columns(
        pl.when(pl.col(E) == "e1")
        .then(pl.col("x") * 100)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    other = q.transform(bumped).collect().sort(E, T)
    keep = pl.col(E) != "e1"
    assert base.filter(keep).equals(other.filter(keep))


def test_output_layouts_and_dtypes():
    df = _panel()
    q = QuantEmbedder(window=8, columns="x", warmup="null", entity=E, time=T)
    arr = q.fit_transform(df).collect()
    assert arr.schema["x_quant"] == pl.Array(pl.Float32, q.n_outputs)
    cols = (
        QuantEmbedder(
            window=8, columns="x", warmup="null", output="columns", entity=E, time=T
        )
        .fit_transform(df)
        .collect()
    )
    assert cols.width == 2 + q.n_outputs
    assert f"x_quant_{q.feature_names_[0]}" in cols.columns
    half = (
        QuantEmbedder(
            window=8, columns="x", warmup="null", dtype="float16", entity=E, time=T
        )
        .fit_transform(df)
        .collect()
    )
    assert half.schema["x_quant"] == pl.Array(pl.Float16, q.n_outputs)
    a = np.asarray(arr["x_quant"].drop_nulls().to_list(), dtype=np.float64)
    h = np.asarray(half["x_quant"].drop_nulls().to_list(), dtype=np.float64)
    big = np.abs(a) > 1e-2
    assert np.median(np.abs(h[big] - a[big]) / np.abs(a[big])) < 1e-3
    kept = (
        QuantEmbedder(
            window=8, columns="x", warmup="null", keep_features=True, entity=E, time=T
        )
        .fit_transform(df)
        .collect()
    )
    assert {"x", "y", "x_quant"} <= set(kept.columns)


def test_default_columns_are_every_numeric_feature():
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmbeddingWarmupWarning)
        out = QuantEmbedder(window=8, entity=E, time=T).fit_transform(df).collect()
    assert {"x_quant", "y_quant"} <= set(out.columns)


def test_budget_is_checked_before_allocating():
    from panelary.shape import ShapeBudgetError

    q = QuantEmbedder(
        window=8, columns="x", warmup="null", max_bytes=10, entity=E, time=T
    )
    with pytest.raises(ShapeBudgetError, match="QuantEmbedder"):
        q.fit_transform(_panel())
    plan = (
        QuantEmbedder(window=8, columns="x")
        .fit(_panel(), entity=E, time=T)
        .plan(_panel(), entity=E, time=T)
    )
    assert plan.width == QuantEmbedder(window=8).n_outputs


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        QuantEmbedder(window=8, entity=E, time=T).transform(_panel())


def test_duplicate_keys_raise():
    df = _panel()
    with pytest.raises(ValueError, match="duplicate"):
        QuantEmbedder(window=8, columns="x", entity=E, time=T).fit(df).transform(
            pl.concat([df, df.head(2)])
        )


def test_a_gap_in_one_column_never_nulls_another():
    df = _panel().with_columns(
        pl.when((pl.col(E) == "e0") & (pl.col(T) == 30))
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    q = QuantEmbedder(window=8, columns=["x", "y"], warmup="null", entity=E, time=T)
    out = q.fit_transform(df).collect().filter(pl.col(E) == "e0").sort(T)
    assert out["x_quant"][30:38].null_count() == 8
    assert out["y_quant"][30:38].null_count() == 0
