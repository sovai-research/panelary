"""Statelessness of the ``fit_is_empty`` shape transforms (plan section 9, item 3).

``SparseRandomProjection``, ``SRHT``, ``CountSketch``, ``PAA``, ``Spectral`` and
``Delay`` claim ``fit_is_empty = True``: their fitted state is a function of the
column names, the parameters and the seed -- never of a data row. The claim is
enforced structurally (``ShapeTransform.fit`` hands ``_fit`` a zero-row frame);
this file checks the consequence the ``embed/`` contract names: fitting on two
**disjoint** datasets with the same seed gives byte-identical state.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import panelary.shape as shape
from panelary.core.panel_frame import PanelFrame
from panelary.shape import (
    PAA,
    SRHT,
    CountSketch,
    CrossSectionalRandomizedPCA,
    Delay,
    Flavour,
    Intent,
    ShapeSpec,
    ShapeTransform,
    SparseRandomProjection,
    Spectral,
)

COLS = ["a", "b", "c", "d", "e"]

STATELESS: dict[str, Callable[[], ShapeTransform]] = {
    "SparseRandomProjection": lambda: SparseRandomProjection(n_components=4, seed=7),
    "SRHT": lambda: SRHT(n_components=4, seed=7),
    "CountSketch": lambda: CountSketch(n_components=3, seed=7),
    "PAA": lambda: PAA(window=4, segments=2),
    "Spectral": lambda: Spectral(window=8, k=3),
    "Delay": lambda: Delay(lags=3, dilation=2),
    "CrossSectionalRandomizedPCA": lambda: CrossSectionalRandomizedPCA(
        n_components=2, seed=7, align="procrustes"
    ),
}


def _frame(entities: list[str], t0: int, seed: int, scale: float) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_t = 20
    n = len(entities) * n_t
    data: dict[str, Any] = {
        "id": np.repeat(entities, n_t),
        "t": np.tile(np.arange(t0, t0 + n_t), len(entities)),
    }
    for j, c in enumerate(COLS):
        data[c] = rng.standard_normal(n) * scale * (j + 1) + j
    return pl.DataFrame(data)


# Disjoint in every respect: entities, dates, and value distribution.
_A = _frame(["a0", "a1", "a2"], 0, seed=1, scale=1.0)
_B = _frame(["b0", "b1", "b2", "b3"], 100, seed=2, scale=50.0)
_C = _frame(["c0", "c1"], 500, seed=3, scale=5.0)


def _canon(value: Any) -> Any:
    """A byte-exact, comparable rendering of one attribute value."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (pl.DataFrame, pl.Series)):
        return (
            "polars",
            repr(value.schema if isinstance(value, pl.DataFrame) else value.dtype),
            value.to_numpy().tobytes(),
        )
    if isinstance(value, dict):
        return tuple((k, _canon(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, float):
        return ("float", np.float64(value).tobytes())
    return value


def _state(t: ShapeTransform) -> dict[str, Any]:
    # `_fit_panel` is the protocol's record of *which* panel was fitted (used by
    # Pipeline's train/test leakage gate), not learned state.
    return {k: _canon(v) for k, v in vars(t).items() if k != "_fit_panel"}


def _fit(factory: Callable[[], ShapeTransform], df: pl.DataFrame) -> ShapeTransform:
    return factory().fit(df, entity="id", time="t")


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_every_fit_is_empty_transform_is_covered() -> None:
    declared = set()
    for name in shape.__all__:
        obj = getattr(shape, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, ShapeTransform)
            and not inspect.isabstract(obj)
            and obj is not ShapeTransform
            and obj.fit_is_empty
        ):
            declared.add(obj.__name__)
    assert declared == set(STATELESS), (
        f"fit_is_empty classes {sorted(declared)} vs covered {sorted(STATELESS)}"
    )


# --------------------------------------------------------------------------- #
# The consequence: byte-identical state from disjoint data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(STATELESS))
def test_disjoint_fits_give_byte_identical_state(name: str) -> None:
    a = _state(_fit(STATELESS[name], _A))
    b = _state(_fit(STATELESS[name], _B))
    assert a == b, (
        f"{name} declares fit_is_empty=True but fitting on two disjoint datasets "
        f"produced different state in {sorted(k for k in a if a[k] != b.get(k))}."
    )


@pytest.mark.parametrize("name", sorted(STATELESS))
def test_output_does_not_depend_on_the_fit_data(name: str) -> None:
    oa = _fit(STATELESS[name], _A).transform(_C, entity="id", time="t").collect()
    ob = _fit(STATELESS[name], _B).transform(_C, entity="id", time="t").collect()
    assert oa.equals(ob)


@pytest.mark.parametrize(
    "factory",
    [
        lambda s: SparseRandomProjection(n_components=4, seed=s),
        lambda s: SRHT(n_components=4, seed=s),
        lambda s: CountSketch(n_components=3, seed=s),
    ],
    ids=["srp", "srht", "count_sketch"],
)
def test_state_is_a_function_of_the_seed(
    factory: Callable[[int], ShapeTransform],
) -> None:
    s1 = _state(factory(1).fit(_A, entity="id", time="t"))
    s1b = _state(factory(1).fit(_A, entity="id", time="t"))
    s2 = _state(factory(2).fit(_A, entity="id", time="t"))
    assert s1 == s1b
    assert s1 != s2


@pytest.mark.parametrize(
    "cls", [SparseRandomProjection, SRHT, CountSketch], ids=lambda c: c.__name__
)
def test_seed_must_be_explicit_int(cls: type[ShapeTransform]) -> None:
    with pytest.raises(TypeError, match="explicit int"):
        cls(seed=None)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# The enforcement: _fit is handed zero rows
# --------------------------------------------------------------------------- #
class _Spy(ShapeTransform):
    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(intent=Intent.COMPRESS, axis="feature")

    def _fit(self, panel: PanelFrame) -> None:
        self.seen_rows_ = panel.collect().height
        self.seen_columns_ = list(panel.columns)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel


class _SpyFitted(_Spy):
    fit_is_empty = False


def test_fit_is_empty_is_enforced_structurally() -> None:
    spy = _Spy().fit(_A, entity="id", time="t")
    assert spy.seen_rows_ == 0, "a fit_is_empty transform could read data rows"
    assert spy.seen_columns_ == _A.columns, "the schema must still be visible"
    assert _SpyFitted().fit(_A, entity="id", time="t").seen_rows_ == _A.height


def test_a_trailing_transform_without_fit_is_empty_is_refused() -> None:
    with pytest.raises(TypeError, match="fit_is_empty"):

        class _NoDecl(ShapeTransform):  # noqa: F841 - defined for the side effect
            panel_safe = True
            leakage_safe = True
            is_cross_sectional = False
            spec = ShapeSpec(
                intent=Intent.COMPRESS, axis="time", flavour=Flavour.TRAILING
            )

            def _fit(self, panel: PanelFrame) -> None: ...

            def _transform(self, panel: PanelFrame) -> PanelFrame:
                return panel
