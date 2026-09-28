"""Additive ``.ts`` operators from :mod:`panelary.depend`: rolling dependence.

The ``.ts`` namespace itself is registered by
:mod:`panelary.feature_extractors._namespace`. This module only **adds**
methods to it -- it never replaces an existing one -- and registers one
:class:`~panelary.registry.FeatureSpec` per operator. It is imported by
:mod:`panelary.depend`, so ``import panelary.depend`` is enough to enable::

    df.with_columns(pl.col("ret").ts.rolling_xi("mkt", window=60).over("ticker"))

Every operator is a trailing-window statistic evaluated at every row
(``safe_scope="rowwise"``): the value at ``t`` uses rows ``t - window + 1 ..
t`` of its own entity only, with ranks, tail thresholds and copula scores
re-derived inside each window. All parameters have defaults (``other=None``
pairs the column with its own lag) so the registry conformance suite can build
and verify each operator without an adapter.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from panelary.depend._rolling import (
    rolling_dcor,
    rolling_gcmi,
    rolling_tail_dep,
    rolling_xi,
)
from panelary.feature_extractors._namespace import FeatureExtractor
from panelary.registry import FeatureSpec, registry

__all__ = ["ROLLING_SPECS"]


def _ts_rolling_xi(
    self: Any, other: str | pl.Expr | None = None, *, window: int = 20
) -> pl.Expr:
    """Trailing-window Chatterjee xi of this column on ``other`` (see
    :func:`panelary.depend._rolling.rolling_xi`)."""
    return rolling_xi(self._expr, other, window=window)


def _ts_rolling_dcor(
    self: Any, other: str | pl.Expr | None = None, *, window: int = 20
) -> pl.Expr:
    """Trailing-window distance correlation with ``other`` (see
    :func:`panelary.depend._rolling.rolling_dcor`)."""
    return rolling_dcor(self._expr, other, window=window)


def _ts_rolling_tail_dep(
    self: Any,
    other: str | pl.Expr | None = None,
    *,
    window: int = 60,
    q: float = 0.1,
    side: str = "lower",
) -> pl.Expr:
    """Trailing-window tail dependence with ``other`` (see
    :func:`panelary.depend._rolling.rolling_tail_dep`)."""
    return rolling_tail_dep(self._expr, other, window=window, q=q, side=side)


def _ts_rolling_gcmi(
    self: Any, other: str | pl.Expr | None = None, *, window: int = 20
) -> pl.Expr:
    """Trailing-window Gaussian-copula MI with ``other`` (see
    :func:`panelary.depend._rolling.rolling_gcmi`)."""
    return rolling_gcmi(self._expr, other, window=window)


_METHODS = {
    "rolling_xi": _ts_rolling_xi,
    "rolling_dcor": _ts_rolling_dcor,
    "rolling_tail_dep": _ts_rolling_tail_dep,
    "rolling_gcmi": _ts_rolling_gcmi,
}

for _name, _fn in _METHODS.items():
    _fn.__name__ = _name
    _fn.__qualname__ = f"{FeatureExtractor.__qualname__}.{_name}"
    if not hasattr(FeatureExtractor, _name):  # additive only: never replace
        setattr(FeatureExtractor, _name, _fn)


def _spec(name: str, params: dict[str, Any], cost: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        namespace="ts",
        input_shape="series",
        output_shape="series",
        params=params,
        tier="B",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        # "Panelary" = clean-room implementation, the convention every `.ts`
        # spec follows; the papers are cited in the operator docstrings.
        source="Panelary",
        license="Apache-2.0",
        axis="time",
        flavour="trailing",
        cost_hint=cost,
    )


#: The registered specs, in registration order.
ROLLING_SPECS: tuple[FeatureSpec, ...] = (
    _spec("rolling_xi", {"other": "str | Expr | None", "window": int}, "O(T w log w)"),
    _spec("rolling_dcor", {"other": "str | Expr | None", "window": int}, "O(T w^1.5)"),
    _spec(
        "rolling_tail_dep",
        {"other": "str | Expr | None", "window": int, "q": float, "side": str},
        "O(T w log w)",
    ),
    _spec(
        "rolling_gcmi", {"other": "str | Expr | None", "window": int}, "O(T w log w)"
    ),
)

for _s in ROLLING_SPECS:
    registry.register(_s)
