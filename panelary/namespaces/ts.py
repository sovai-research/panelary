"""Additive ``.ts`` operators: rolling dependence and the trailing trend scan.

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

``.ts.trend_scan`` (the look-back half of :mod:`panelary.label._trend`) is added
here too: the best-|t| OLS trend over a trailing window grid, also
``safe_scope="rowwise"``::

    df.with_columns(pl.col("close").log().ts.trend_scan(max_window=60).over("id"))
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
from panelary.label._trend import trend_scan_expr
from panelary.registry import FeatureSpec, registry

__all__ = ["ROLLING_SPECS", "TREND_SPECS"]


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


# --- trend scanning (label/_trend.py) -------------------------------------
def _ts_trend_scan(
    self: Any,
    *,
    min_window: int = 5,
    max_window: int = 20,
    step: int = 1,
    output: str = "t",
) -> pl.Expr:
    """Trailing trend scan: best-|t| OLS trend over the window grid
    ``min_window, min_window+step, ..., <= max_window`` ending at each row;
    ``output`` is ``"t"``, ``"horizon"`` or ``"slope"`` (see
    :func:`panelary.label._trend.trend_scan_expr`)."""
    return trend_scan_expr(
        self._expr,
        min_window=min_window,
        max_window=max_window,
        step=step,
        output=output,
    )


_METHODS = {
    "rolling_xi": _ts_rolling_xi,
    "rolling_dcor": _ts_rolling_dcor,
    "rolling_tail_dep": _ts_rolling_tail_dep,
    "rolling_gcmi": _ts_rolling_gcmi,
    "trend_scan": _ts_trend_scan,
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

#: The trend-scan spec (the causal look-back feature; the forward label
#: ``label.trend_scanning`` is registered in :mod:`panelary.label._trend`).
TREND_SPECS: tuple[FeatureSpec, ...] = (
    _spec(
        "trend_scan",
        {"min_window": int, "max_window": int, "step": int, "output": str},
        "O(T L_max)",
    ),
)

for _s in TREND_SPECS:
    registry.register(_s)
