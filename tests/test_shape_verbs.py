"""Golden-path reachability of the shape algebra (plan section 7.3).

``pn.reduce`` gains the compress-intent methods (``rsvd``, ``sparse_rp``,
``srht``, ``id``, ``cur``, ``paa``, ``spectral``) and ``pn.features`` the two
trailing lifts (``delay``, ``spectral``) -- with the verbs' existing signatures
and defaults untouched. Each method must be exactly the class it names, fitted
on what was passed.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

import panelary as pn
import panelary.shape as shape


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n_t = 40
    return pl.DataFrame(
        {
            "id": np.repeat(["A", "B", "C"], n_t),
            "t": np.tile(np.arange(n_t), 3),
            **{c: rng.standard_normal(3 * n_t) for c in "abcdef"},
        }
    )


REDUCE = {
    "rsvd": (lambda: shape.RandomizedPCA(n_components=2), {}),
    "sparse_rp": (lambda: shape.SparseRandomProjection(n_components=2), {}),
    "srht": (lambda: shape.SRHT(n_components=2), {}),
    "id": (lambda: shape.ColumnSubset(k=2), {}),
    "cur": (lambda: shape.CUR(k=2), {}),
    "paa": (lambda: shape.PAA(window=8, segments=2), {"window": 8}),
    "spectral": (lambda: shape.Spectral(window=8, k=2), {"window": 8}),
}


@pytest.mark.parametrize("method", sorted(REDUCE))
def test_reduce_shape_method_is_the_class_it_names(
    df: pl.DataFrame, method: str
) -> None:
    factory, kwargs = REDUCE[method]
    via_verb = pn.reduce(
        df, method=method, n_components=2, entity="id", time="t", **kwargs
    )
    assert isinstance(via_verb, pn.PanelFrame)
    direct = factory().fit_transform(df, entity="id", time="t")
    assert via_verb.collect().equals(direct.collect())


def test_reduce_time_axis_methods_need_a_window(df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="window"):
        pn.reduce(df, method="paa", n_components=2, entity="id", time="t")


def test_reduce_whole_series_is_an_explicit_opt_in(df: pl.DataFrame) -> None:
    out = pn.reduce(
        df,
        method="paa",
        columns="a",
        n_components=4,
        flavour="whole_series",
        entity="id",
        time="t",
    ).collect()
    assert out.height == 3 and out["t"].unique().to_list() == ["whole_series"]


def test_reduce_signature_and_default_are_unchanged() -> None:
    params = inspect.signature(pn.reduce).parameters
    assert params["method"].default == "pca"
    assert params["n_components"].default is None
    assert list(params)[:3] == ["data", "method", "columns"]


@pytest.mark.parametrize(
    ("method", "kwargs", "factory"),
    [
        ("delay", {"lags": 3}, lambda: shape.Delay(lags=3, columns=["a"])),
        (
            "spectral",
            {"window": 16, "k": 3},
            lambda: shape.Spectral(window=16, k=3, mode="band", columns=["a"]),
        ),
    ],
    ids=["delay", "spectral"],
)
def test_features_trailing_lifts(
    df: pl.DataFrame, method: str, kwargs: dict, factory
) -> None:  # type: ignore[no-untyped-def]
    via_verb = pn.features(
        df, method=method, columns="a", entity="id", time="t", **kwargs
    )
    assert isinstance(via_verb, pn.PanelFrame)
    out = via_verb.collect()
    assert out.height == df.height  # one row per input row, not per entity
    assert out.equals(factory().fit_transform(df, entity="id", time="t").collect())


def test_features_default_path_is_untouched(df: pl.DataFrame) -> None:
    out = pn.features(
        df, method=["absolute_energy"], columns="a", entity="id", time="t"
    )
    frame = out.collect() if isinstance(out, pl.LazyFrame) else out
    assert frame.height == 3  # one row per entity
