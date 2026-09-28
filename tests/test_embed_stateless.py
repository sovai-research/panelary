"""``fit_is_empty = True`` is a testable claim (build contract section 2).

Every class that declares it must produce **byte-identical state** when fitted
on two disjoint datasets with the same schema. The check has teeth: fitted
classes (``fit_is_empty = False``) are shown to give *different* state on the
same two datasets, and a class that lies about ``fit_is_empty`` cannot read a
data row even if it tries (``ShapeTransform.fit`` hands it zero rows).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import panelary.embed as pe
from panelary.core.panel_frame import PanelFrame
from panelary.embed import (
    CrossSectionalEmbedder,
    EmbeddingCompressor,
    HydraEmbedder,
    QuantEmbedder,
    RandIntC22,
    RandomFourierFeatures,
    TensorSketch,
)

E, T = "entity", "time"


def _data(seed: int, offset: int) -> pl.DataFrame:
    """Same schema, disjoint entities / times / values for each seed."""
    rng = np.random.default_rng(seed)
    n_e, n_t = 4, 40
    return pl.DataFrame(
        {
            E: np.repeat([f"s{seed}_{i}" for i in range(n_e)], n_t),
            T: np.tile(np.arange(n_t, dtype=np.int64) + offset, n_e),
            "x": rng.standard_normal(n_e * n_t) * (1 + seed),
            "y": rng.random(n_e * n_t) + seed,
        }
    )


def _with_array(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.concat_list("x", "y").list.to_array(2).alias("emb"))


STATELESS: dict[str, Callable[[], Any]] = {
    "QuantEmbedder": lambda: QuantEmbedder(
        window=16, columns=["x", "y"], entity=E, time=T
    ),
    "RandIntC22": lambda: RandIntC22(
        window=24, n_intervals=4, seed=5, entity=E, time=T
    ),
    "EmbeddingCompressor[srp]": lambda: EmbeddingCompressor(
        "srp", dim=8, columns=["emb"], entity=E, time=T
    ),
    "CrossSectionalEmbedder": lambda: CrossSectionalEmbedder(
        align="procrustes", n_components=2, columns=["x", "y"], entity=E, time=T
    ),
    "HydraEmbedder": lambda: HydraEmbedder(
        window=16, n_groups=2, n_kernels=3, seed=4, entity=E, time=T
    ),
    "TensorSketch": lambda: TensorSketch(
        n_components=16, degree=2, seed=9, columns=["x", "emb"], entity=E, time=T
    ),
}

FITTED: dict[str, Callable[[], Any]] = {
    "RandomFourierFeatures": lambda: RandomFourierFeatures(
        n_components=10, columns=["x", "y"], entity=E, time=T
    ),
    "EmbeddingCompressor[pca]": lambda: EmbeddingCompressor(
        "pca", dim=2, columns=["emb"], entity=E, time=T
    ),
}


def _state(factory: Callable[[], Any], df: pl.DataFrame) -> Any:
    return factory().fit(_with_array(df)).get_state()


@pytest.mark.parametrize("name", sorted(STATELESS))
def test_disjoint_fits_give_byte_identical_state(name):
    a = _state(STATELESS[name], _data(1, 0))
    b = _state(STATELESS[name], _data(2, 1000))
    assert a.fingerprint() == b.fingerprint(), name
    assert a.arrays.keys() == b.arrays.keys()
    for k in a.arrays:
        assert a.arrays[k].tobytes() == b.arrays[k].tobytes(), (name, k)


@pytest.mark.parametrize("name", sorted(FITTED))
def test_fitted_classes_do_learn_from_data(name):
    a = _state(FITTED[name], _data(1, 0))
    b = _state(FITTED[name], _data(2, 1000))
    assert a.fingerprint() != b.fingerprint(), name


def test_every_stateless_public_class_is_covered():
    declared = {
        n
        for n in pe.__all__
        if isinstance(getattr(pe, n), type)
        and getattr(getattr(pe, n), "fit_is_empty", None) is True
        and hasattr(getattr(pe, n), "get_state")
    }
    covered = {n.split("[")[0] for n in STATELESS}
    assert declared <= covered, declared - covered


def test_fit_is_empty_is_enforced_not_documentary():
    seen: list[int] = []

    class Peeking(QuantEmbedder):
        def _fit(self, panel: PanelFrame) -> None:
            seen.append(panel.collect().height)
            super()._fit(panel)

    Peeking(window=8, columns="x", entity=E, time=T).fit(_data(0, 0))
    assert seen == [0]


def test_state_depends_on_schema():
    a = _state(STATELESS["QuantEmbedder"], _data(1, 0))
    renamed = _data(1, 0).rename({"x": "z"})
    q = QuantEmbedder(window=16, columns=["z", "y"], entity=E, time=T).fit(renamed)
    assert q.get_state().fingerprint() != a.fingerprint()
