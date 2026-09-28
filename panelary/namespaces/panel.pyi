"""Type stubs for the ``.panel`` Polars expression namespace.

These stubs give IDEs and static type checkers (mypy, Pyright) full knowledge of
the methods registered onto :class:`polars.Expr` via
``pl.api.register_expr_namespace("panel")``. Because Polars adds the namespace
dynamically at runtime, without these stubs ``pl.col(...).panel.frac_diff(...)``
would not autocomplete or type-check.

The package ships ``py.typed`` (PEP 561), so a checker that resolves
``panelary.namespaces.panel`` will read this ``.pyi`` in preference to the
runtime module. Importing :class:`PanelExprNamespace` from the package is enough
for the stubs to be discovered.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from panelary.quality import QualityReport, ValidationReport

__all__ = [
    "PanelExprNamespace",
    "PanelLazyFrameNamespace",
    "PanelDataFrameNamespace",
    "register",
]

class PanelExprNamespace:
    def __init__(self, expr: pl.Expr) -> None: ...
    def frac_diff(self, d: float, *, threshold: float = ...) -> pl.Expr: ...
    def zscore(self, window: int) -> pl.Expr: ...
    def rs_vol(self, window: int) -> pl.Expr: ...

class PanelLazyFrameNamespace:
    def __init__(self, lf: pl.LazyFrame) -> None: ...
    def frac_diff(
        self,
        column: str,
        *,
        d: float,
        over: str | None = ...,
        alias: str | None = ...,
        threshold: float = ...,
    ) -> pl.LazyFrame: ...
    def zscore(
        self,
        column: str,
        *,
        window: int,
        over: str | None = ...,
        alias: str | None = ...,
    ) -> pl.LazyFrame: ...
    def rs_vol(
        self,
        column: str,
        *,
        window: int,
        over: str | None = ...,
        alias: str | None = ...,
    ) -> pl.LazyFrame: ...
    def validate(
        self,
        *,
        entity: str | None = ...,
        time: str | None = ...,
        **kwargs: Any,
    ) -> ValidationReport: ...
    def quality_report(
        self,
        *,
        entity: str | None = ...,
        time: str | None = ...,
        **kwargs: Any,
    ) -> QualityReport: ...
    def dedup(
        self,
        *,
        entity: str | None = ...,
        time: str | None = ...,
        **kwargs: Any,
    ) -> pl.LazyFrame: ...

class PanelDataFrameNamespace:
    def __init__(self, df: pl.DataFrame) -> None: ...
    def frac_diff(
        self,
        column: str,
        *,
        d: float,
        over: str | None = ...,
        alias: str | None = ...,
        threshold: float = ...,
    ) -> pl.DataFrame: ...
    def zscore(
        self,
        column: str,
        *,
        window: int,
        over: str | None = ...,
        alias: str | None = ...,
    ) -> pl.DataFrame: ...
    def rs_vol(
        self,
        column: str,
        *,
        window: int,
        over: str | None = ...,
        alias: str | None = ...,
    ) -> pl.DataFrame: ...
    def validate(
        self,
        *,
        entity: str | None = ...,
        time: str | None = ...,
        **kwargs: Any,
    ) -> ValidationReport: ...
    def quality_report(
        self,
        *,
        entity: str | None = ...,
        time: str | None = ...,
        **kwargs: Any,
    ) -> QualityReport: ...
    def dedup(
        self,
        *,
        entity: str | None = ...,
        time: str | None = ...,
        **kwargs: Any,
    ) -> pl.DataFrame: ...

def register() -> None: ...
