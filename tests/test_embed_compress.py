"""Compaction layer (`panelary.embed.EmbeddingCompressor`) -- a dispatch to `panelary.shape`.

Contracts touched: ``leakage_safe`` for the fitted methods (``pca`` / ``svd``
learn from the rows passed to ``fit`` only; see ``test_embed_leakage.py``) and
statelessness of ``srp`` (see ``test_embed_stateless.py``). The key property
here is that nothing is reimplemented: outputs equal the shape primitives'.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.embed import EmbeddingCompressor
from panelary.embed._compress import __doc__ as compress_doc
from panelary.shape._project import SparseRandomProjection
from panelary.shape._rsvd import RandomizedPCA

E, T = "entity", "time"


def _frame(n: int = 120, d: int = 12, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, d)) @ rng.standard_normal((d, d))
    df = pl.DataFrame(
        {
            E: np.repeat(["a", "b", "c"], n // 3),
            T: np.tile(np.arange(n // 3, dtype=np.int64), 3),
            "emb": M,
        }
    ).with_columns(pl.col("emb").cast(pl.Array(pl.Float64, d)))
    return df, M


def _wide(M: np.ndarray) -> pl.DataFrame:
    n = M.shape[0]
    return pl.DataFrame(
        {E: np.zeros(n, dtype=np.int64), T: np.arange(n, dtype=np.int64)}
    ).hstack(
        pl.from_numpy(M, schema=[f"c{i}" for i in range(M.shape[1])], orient="row")
    )


def _mat(frame: pl.DataFrame, col: str = "embedding") -> np.ndarray:
    return np.asarray(frame[col].to_list(), dtype=np.float64)


def test_srp_is_the_shape_primitive():
    df, M = _frame()
    out = (
        EmbeddingCompressor("srp", dim=5, seed=3, dtype="float64", entity=E, time=T)
        .fit(df)
        .transform(df)
        .collect()
    )
    ref = (
        SparseRandomProjection(n_components=5, seed=3)
        .fit(_wide(M), entity=E, time=T)
        .transform(_wide(M))
        .collect()
    )
    np.testing.assert_allclose(
        _mat(out), ref.select(pl.exclude(E, T)).to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize(("method", "standardize"), [("pca", True), ("svd", False)])
def test_pca_and_svd_are_randomized_pca(method, standardize):
    df, M = _frame()
    train = df.filter(pl.col(T) < 25)
    comp = EmbeddingCompressor(
        method, dim=4, seed=1, dtype="float64", entity=E, time=T
    ).fit(train)
    out = _mat(comp.transform(df).collect())
    Mt = np.asarray(train["emb"].to_list())
    ref = RandomizedPCA(n_components=4, standardize=standardize, seed=1).fit(
        _wide(Mt), entity=E, time=T
    )
    expect = ref.transform(_wide(M)).collect().select(pl.exclude(E, T)).to_numpy()
    np.testing.assert_allclose(out, expect, rtol=1e-10, atol=1e-10)


def test_none_is_identity_and_concatenates_inputs():
    df, M = _frame()
    df = df.with_columns(pl.lit(2.0).alias("s"))
    out = (
        EmbeddingCompressor(
            "none", columns=["emb", "s"], dtype="float64", entity=E, time=T
        )
        .fit(df)
        .transform(df)
        .collect()
    )
    np.testing.assert_array_equal(
        _mat(out), np.hstack([M, np.full((M.shape[0], 1), 2.0)])
    )


def test_default_columns_prefer_array_columns_and_nulls_propagate():
    df, _ = _frame()
    df = df.with_columns(pl.lit(1.0).alias("other"))
    df = df.with_columns(
        pl.when(pl.col(T) == 3).then(None).otherwise(pl.col("emb")).alias("emb")
    )
    comp = EmbeddingCompressor("srp", dim=3, entity=E, time=T).fit(df)
    assert comp.feature_names_in_ == ["emb"]
    out = comp.transform(df).collect()
    assert out.filter(pl.col(T) == 3)["embedding"].null_count() == 3
    assert out.filter(pl.col(T) != 3)["embedding"].null_count() == 0


def test_state_roundtrip_every_method():
    df, _ = _frame()
    for method in ("srp", "pca", "svd", "none"):
        comp = EmbeddingCompressor(method, dim=4, entity=E, time=T).fit(df)
        again = EmbeddingCompressor.from_state(comp.get_state())
        assert again.transform(df).collect().equals(comp.transform(df).collect()), (
            method
        )


def test_width_mismatch_and_bad_method_raise():
    df, _ = _frame()
    comp = EmbeddingCompressor("srp", dim=3, entity=E, time=T).fit(df)
    narrower = df.with_columns(pl.col("emb").arr.slice(0, 5).list.to_array(5))
    with pytest.raises(ValueError, match="width"):
        comp.transform(narrower)
    with pytest.raises(ValueError, match="method"):
        EmbeddingCompressor("umap")


def test_docstring_carries_the_jl_honesty_note():
    assert "does not license" in (compress_doc or "")
