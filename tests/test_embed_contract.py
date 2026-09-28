"""The embedding contract (`panelary.embed._contract`): state, protocol, helpers.

Contract touched: deterministic inference -- identical state gives identical
output, and the state survives a save/load round trip byte for byte.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary.embed as pe
from panelary.embed import EMBED_STATE_VERSION, Embedder, EmbeddingState, QuantEmbedder
from panelary.embed._contract import matrix_from_columns
from panelary.shape import ShapeTransform

E, T = "entity", "time"


def _panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            E: np.repeat(["a", "b"], 30),
            T: np.tile(np.arange(30, dtype=np.int64), 2),
            "x": rng.standard_normal(60),
        }
    )


def test_state_save_load_roundtrip(tmp_path):
    q = QuantEmbedder(window=8, columns="x", warmup="null", entity=E, time=T).fit(
        _panel()
    )
    st = q.get_state()
    assert st.version == EMBED_STATE_VERSION and st.method == "QuantEmbedder"
    assert st.columns == ("x",) and st.params["window"] == 8
    path = st.save(tmp_path / "q")
    back = EmbeddingState.load(path)
    assert back.fingerprint() == st.fingerprint()
    q2 = QuantEmbedder.from_state(back)
    assert q2.transform(_panel()).collect().equals(q.transform(_panel()).collect())


def test_fingerprint_sees_parameters_and_arrays():
    a = EmbeddingState("M", 1, {"seed": 0}, ("x",), {"w": np.zeros(3)})
    assert (
        a.fingerprint()
        == EmbeddingState("M", 1, {"seed": 0}, ("x",), {"w": np.zeros(3)}).fingerprint()
    )
    assert (
        a.fingerprint()
        != EmbeddingState("M", 1, {"seed": 1}, ("x",), {"w": np.zeros(3)}).fingerprint()
    )
    assert (
        a.fingerprint()
        != EmbeddingState("M", 1, {"seed": 0}, ("y",), {"w": np.zeros(3)}).fingerprint()
    )
    assert (
        a.fingerprint()
        != EmbeddingState("M", 1, {"seed": 0}, ("x",), {"w": np.ones(3)}).fingerprint()
    )


def test_newer_state_version_is_refused(tmp_path):
    st = EmbeddingState("QuantEmbedder", EMBED_STATE_VERSION + 1, {}, ("x",), {})
    path = st.save(tmp_path / "future")
    with pytest.raises(ValueError, match="newer"):
        EmbeddingState.load(path)


def test_from_state_refuses_another_class():
    st = (
        QuantEmbedder(window=8, columns="x", entity=E, time=T).fit(_panel()).get_state()
    )
    with pytest.raises(ValueError, match="QuantEmbedder"):
        pe.RandIntC22.from_state(st)


def test_get_state_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        QuantEmbedder(window=8).get_state()


def test_unserialisable_parameter_is_refused():
    st = EmbeddingState("M", 1, {"bad": object()}, (), {})
    with pytest.raises(TypeError, match="serialisable"):
        st.fingerprint()


def test_every_public_transform_satisfies_the_protocol():
    classes = [
        getattr(pe, name) for name in pe.__all__ if isinstance(getattr(pe, name), type)
    ]
    transforms = [c for c in classes if hasattr(c, "fit_is_empty")]
    assert len(transforms) >= 6
    for cls in transforms:
        for attr in (
            "panel_safe",
            "leakage_safe",
            "fit_is_empty",
            "is_cross_sectional",
        ):
            assert isinstance(getattr(cls, attr), bool), (cls.__name__, attr)
        assert callable(cls.fit) and callable(cls.transform)
    built_here = [c for c in transforms if issubclass(c, ShapeTransform)]
    assert {c.__name__ for c in built_here} >= {
        "QuantEmbedder",
        "RandIntC22",
        "RandomFourierFeatures",
        "EmbeddingCompressor",
    }
    for cls in built_here:
        assert hasattr(cls, "get_state") and hasattr(cls, "from_state")
    assert isinstance(QuantEmbedder(window=8), Embedder)


def test_leaky_reference_is_not_exported():
    assert "LeakyMiniRocketReference" not in pe.__all__
    assert not hasattr(pe, "LeakyMiniRocketReference")


def test_matrix_from_columns_handles_arrays_scalars_and_nulls():
    df = pl.DataFrame(
        {"a": [[1.0, 2.0], None, [5.0, 6.0]], "b": [1, None, 3]}
    ).with_columns(pl.col("a").cast(pl.Array(pl.Float64, 2)))
    M = matrix_from_columns(df, ["a", "b"])
    np.testing.assert_array_equal(M[0], [1.0, 2.0, 1.0])
    assert np.isnan(M[1]).all()
    np.testing.assert_array_equal(M[2], [5.0, 6.0, 3.0])
