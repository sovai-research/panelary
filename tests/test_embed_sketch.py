"""TensorSketch (`panelary.embed.TensorSketch`) -- CountSketch composed with the FFT.

Contracts touched: ``fit_is_empty = True`` (stateless given the seed; see
``test_embed_stateless.py``), a row-wise map (``panel_safe``), and the
composition boundary with `panelary.shape`: the CountSketch is shape's.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.embed import TensorSketch
from panelary.shape._sketch import CountSketch

E, T = "entity", "time"


def _frame(X: np.ndarray) -> pl.DataFrame:
    n = X.shape[0]
    return pl.DataFrame({E: np.arange(n), T: np.zeros(n, dtype=np.int64)}).hstack(
        pl.from_numpy(X, schema=[f"c{i}" for i in range(X.shape[1])], orient="row")
    )


def _sketch(df: pl.DataFrame, **kw: object) -> np.ndarray:
    ts = TensorSketch(dtype="float64", entity=E, time=T, **kw).fit(df)  # type: ignore[arg-type]
    return np.asarray(ts.transform(df).collect()["tensor_sketch"].to_list())


def test_degree_one_homogeneous_is_the_count_sketch():
    X = np.random.default_rng(0).standard_normal((6, 5))
    df = _frame(X)
    ts = TensorSketch(
        n_components=8, degree=1, coef0=0.0, seed=3, dtype="float64", entity=E, time=T
    ).fit(df)
    got = np.asarray(ts.transform(df).collect()["tensor_sketch"].to_list())
    cs = ts.sketches_[0]
    assert isinstance(cs, CountSketch)
    names = [f"__pn_ts{i}" for i in range(5)]
    ref = cs.transform(
        _frame(X).rename(dict(zip([f"c{i}" for i in range(5)], names, strict=True))),
        entity=E,
        time=T,
    )
    np.testing.assert_allclose(
        got, ref.collect().select(pl.exclude(E, T)).to_numpy(), atol=1e-12
    )


def test_equals_the_explicit_circular_convolution_of_sketches():
    X = np.random.default_rng(1).standard_normal((4, 3))
    df = _frame(X)
    ts = TensorSketch(
        n_components=16, degree=2, coef0=0.5, seed=2, dtype="float64", entity=E, time=T
    ).fit(df)
    got = np.asarray(ts.transform(df).collect()["tensor_sketch"].to_list())
    Xa = np.hstack([X, np.full((4, 1), np.sqrt(0.5))])
    names = [f"__pn_ts{i}" for i in range(4)]
    S = []
    for cs in ts.sketches_:
        res = cs.transform(
            _frame(Xa).rename(
                dict(zip([f"c{i}" for i in range(4)], names, strict=True))
            ),
            entity=E,
            time=T,
        )
        S.append(res.collect().select(pl.exclude(E, T)).to_numpy())
    conv = np.zeros((4, 16))
    for a in range(16):
        for b in range(16):
            conv[:, (a + b) % 16] += S[0][:, a] * S[1][:, b]
    np.testing.assert_allclose(got, conv, atol=1e-12)


def test_inner_products_are_unbiased_for_the_polynomial_kernel():
    X = np.random.default_rng(2).standard_normal((2, 5)) / 2
    df = _frame(X)
    est = []
    for s in range(150):
        Z = _sketch(df, n_components=256, degree=2, coef0=1.0, seed=s)
        est.append(Z[0] @ Z[1])
    exact = (X[0] @ X[1] + 1.0) ** 2
    assert abs(np.mean(est) - exact) < 4 * np.std(est) / np.sqrt(len(est)) + 1e-3


def test_nulls_state_and_validation():
    X = np.random.default_rng(3).standard_normal((5, 3))
    df = _frame(X).with_columns(
        pl.when(pl.col(E) == 2).then(None).otherwise(pl.col("c1")).alias("c1")
    )
    ts = TensorSketch(n_components=8, seed=1, entity=E, time=T).fit(df)
    out = ts.transform(df).collect()
    assert out.filter(pl.col(E) == 2)["tensor_sketch"].null_count() == 1
    again = TensorSketch.from_state(ts.get_state())
    assert again.transform(df).collect().equals(out)
    with pytest.raises(ValueError, match="width"):
        widened = df.with_columns(
            pl.concat_list("c0", "c1").list.to_array(2).alias("c2")
        )
        ts.transform(widened)
    with pytest.raises(ValueError):
        TensorSketch(coef0=-1.0)
    with pytest.raises(ValueError):
        TensorSketch(degree=0)
