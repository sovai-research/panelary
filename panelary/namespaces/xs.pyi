"""Type stubs for the ``.xs`` cross-sectional Polars namespaces.

These stubs give IDEs and static type checkers (mypy, Pyright) full knowledge of
the methods registered onto :class:`polars.Expr`, :class:`polars.LazyFrame` and
:class:`polars.DataFrame` via ``pl.api.register_*_namespace("xs")``. Because
Polars adds the namespaces dynamically at runtime, without these stubs
``pl.col(...).xs.zscore()`` would neither autocomplete nor type-check.

The package ships ``py.typed`` (PEP 561), so a checker that resolves
``panelary.namespaces.xs`` reads this ``.pyi`` in preference to the runtime
module. That makes the stub user-visible API: **every method registered at
runtime must appear here, with the runtime signature.** ``tests/test_namespaces.py``
asserts the two stay in step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl

__all__ = [
    "XSExprNamespace",
    "XSLazyFrameNamespace",
    "XSDataFrameNamespace",
    "register",
]

_RankMethod = Literal["average", "min", "max", "dense", "ordinal", "random"]

#: ``True`` is an alias for ``"uniform_plus"``; ``False`` returns the raw rank.
_RankNormalize = bool | Literal["uniform_plus", "unit", "centered", "uniform"]

_Columns = str | Sequence[str]
_Limits = float | Sequence[float]

_DispersionKind = Literal["sd", "mad", "iqr", "idr"]
_Tail = Literal["lower", "upper"]
_EntropyKind = Literal["share", "hist"]

class XSExprNamespace:
    def __init__(self, expr: pl.Expr) -> None: ...
    def demean(self) -> pl.Expr: ...
    def dispersion(self, *, kind: _DispersionKind = ...) -> pl.Expr: ...
    def entropy(self, *, kind: _EntropyKind = ...) -> pl.Expr: ...
    def neutralize(self, by: _Columns, *, add_intercept: bool = ...) -> pl.Expr: ...
    def quantile_bin(self, q: int) -> pl.Expr: ...
    def rank(
        self, *, method: _RankMethod = ..., normalize: _RankNormalize = ...
    ) -> pl.Expr: ...
    def standardize(self, *, winsor: _Limits | None = ...) -> pl.Expr: ...
    def tail_index(
        self, *, q: float = ..., tail: _Tail = ..., min_exceedances: int = ...
    ) -> pl.Expr: ...
    def up_share(self) -> pl.Expr: ...
    def winsorize(self, limits: _Limits) -> pl.Expr: ...
    def zscore(self) -> pl.Expr: ...

class XSLazyFrameNamespace:
    def __init__(self, lf: pl.LazyFrame) -> None: ...
    def demean(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...
    def neutralize(
        self,
        columns: _Columns,
        *,
        by: _Columns,
        over: str | None = ...,
        add_intercept: bool = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...
    def quantile_bin(
        self,
        columns: _Columns,
        *,
        q: int,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...
    def rank(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        method: _RankMethod = ...,
        normalize: _RankNormalize = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...
    def standardize(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        winsor: _Limits | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...
    def winsorize(
        self,
        columns: _Columns,
        *,
        limits: _Limits,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...
    def zscore(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.LazyFrame: ...

class XSDataFrameNamespace:
    def __init__(self, df: pl.DataFrame) -> None: ...
    def demean(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...
    def neutralize(
        self,
        columns: _Columns,
        *,
        by: _Columns,
        over: str | None = ...,
        add_intercept: bool = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...
    def quantile_bin(
        self,
        columns: _Columns,
        *,
        q: int,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...
    def rank(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        method: _RankMethod = ...,
        normalize: _RankNormalize = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...
    def standardize(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        winsor: _Limits | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...
    def winsorize(
        self,
        columns: _Columns,
        *,
        limits: _Limits,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...
    def zscore(
        self,
        columns: _Columns,
        *,
        over: str | None = ...,
        alias: str | None = ...,
        suffix: str | None = ...,
    ) -> pl.DataFrame: ...

def register() -> None: ...
