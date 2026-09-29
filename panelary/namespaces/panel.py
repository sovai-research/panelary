"""Per-entity panel operators exposed under the ``.panel`` expression namespace.

The operators here are *time-series* transforms intended to be applied **within a
single entity** of a long-format ``(entity, time, *features)`` panel. They return
plain :class:`polars.Expr` objects and carry **no** grouping of their own — the
user composes them with ``.over(entity)`` so the same expression works on one
series or a million.

In addition to the expression namespace, this module registers **frame-level**
``.panel`` namespaces on :class:`polars.LazyFrame` and :class:`polars.DataFrame`
so users can operate on a bare frame without hand-composing
``pl.col(...).panel.<op>().over(entity)``::

    lf.panel.frac_diff("ret", d=0.4, over="ticker", alias="ret_fd")
    df.panel.zscore("ret", window=21, over="ticker")

All operators in this namespace are **causal** (leakage-safe): the value at time
``t`` depends only on observations at or before ``t``. Combined with
``.over(entity)`` they are also panel-safe (no cross-entity bleed).

Architecture / import boundary
------------------------------
This module is part of the ``namespaces`` package — **Tier 1, the Polars-native
extension layer**. It imports ONLY from :mod:`polars`, the Rust plugin,
:mod:`panelary.registry`, and the dependency-free leaf module
:mod:`panelary._internal._ffd` (numpy/polars only). It must NOT import from
``panelary.core`` or ``panelary.transform`` (the estimator layer),
so the expression layer stays independently splittable into a standalone
``polars-panel`` plugin later. The frac-diff kernel lives in ``_ffd`` precisely
so both this layer and the estimator layer share one weight recursion without
either importing the other.

The expression-building logic is factored into module-level ``_expr_*`` helpers
so the expression namespace and both frame-level namespaces share **one**
implementation with no duplication.

Usage
-----
>>> import polars as pl
>>> import panelary.namespaces  # registers the namespace (side-effect)
>>> df = pl.DataFrame(
...     {"entity": ["a", "a", "a"], "ret": [0.1, -0.2, 0.05]}
... )
>>> out = df.with_columns(
...     fd=pl.col("ret").panel.frac_diff(0.4).over("entity")
... )  # doctest: +SKIP

Notes
-----
Importing this module is a side effect: it registers the ``"panel"`` namespace
with Polars (idempotently) and registers each operator as a
:class:`~panelary.registry.FeatureSpec` in the global registry.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

from panelary._internal._ffd import DEFAULT_THRESHOLD, frac_diff_expr
from panelary.registry import FeatureSpec, registry

if TYPE_CHECKING:  # annotations only: the quality layer is imported lazily
    from panelary.quality import QualityReport, ValidationReport

__all__ = [
    "PanelExprNamespace",
    "PanelLazyFrameNamespace",
    "PanelDataFrameNamespace",
    "register",
]

#: A LazyFrame or DataFrame — both expose ``with_columns`` with identical
#: semantics, so the frame-level operators are generic over the two.
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

_LICENSE = "Apache-2.0"
_SOURCE = "Panelary"


# ---------------------------------------------------------------------------
# Shared expression builders.
#
# These module-level functions are the SINGLE source of truth for each
# operator's expression tree. The ``.panel`` expression namespace and both
# frame-level namespaces (LazyFrame/DataFrame) call them, so there is exactly
# one implementation per operator and no duplication.
#
# ``frac_diff`` is the exception: its kernel + causal builder live in the
# dependency-free leaf module :mod:`panelary._internal._ffd`
# (:func:`~panelary._internal._ffd.frac_diff_expr`) so the estimator layer
# (:class:`panelary.transform.frac_diff.FracDiff`) and this namespace
# layer share ONE weight recursion without either importing the other. The
# thin ``_expr_frac_diff`` wrapper below adapts that shared builder to this
# module's ``(expr, d, *, threshold)`` calling convention.
# ---------------------------------------------------------------------------


def _expr_frac_diff(
    expr: pl.Expr, d: float, *, threshold: float = DEFAULT_THRESHOLD
) -> pl.Expr:
    """Build the fixed-width fractional-differencing expression (causal).

    Thin adapter over the shared :func:`panelary._internal._ffd.frac_diff_expr`
    builder (the single source of truth). See :meth:`PanelExprNamespace.frac_diff`
    for the full description.
    """
    return frac_diff_expr(expr, d=d, threshold=threshold)


def _expr_zscore(expr: pl.Expr, window: int) -> pl.Expr:
    """Build the causal rolling z-score expression over a trailing ``window``."""
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"zscore window must be a positive integer, got {window!r}.")
    mean = expr.rolling_mean(window_size=window)
    std = expr.rolling_std(window_size=window)
    return (expr - mean) / std


def _expr_rolling_vol(
    expr: pl.Expr, window: int, *, op: str = "rolling_vol"
) -> pl.Expr:
    """Build the trailing rolling-standard-deviation expression over ``window``."""
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"{op} window must be a positive integer, got {window!r}.")
    return expr.rolling_std(window_size=window)


#: ``rs_vol`` warns once per process, not once per call site.
_RS_VOL_WARNED = False

_RS_VOL_MESSAGE = (
    "rs_vol is a rolling standard deviation, not Rogers-Satchell. Use "
    ".panel.rolling_vol for identical output, or "
    "pn.econ.features.range_volatility(method='rogers_satchell') for the OHLC "
    "estimator. rs_vol will be removed no earlier than two minor releases after "
    "0.5, and the name will never be reused for a different estimator."
)


def _warn_rs_vol() -> None:
    """Emit the ``rs_vol`` deprecation ``FutureWarning`` (first call only)."""
    global _RS_VOL_WARNED
    if not _RS_VOL_WARNED:
        _RS_VOL_WARNED = True
        warnings.warn(_RS_VOL_MESSAGE, FutureWarning, stacklevel=3)


def _expr_rs_vol(expr: pl.Expr, window: int) -> pl.Expr:
    """Deprecated alias: exactly :func:`_expr_rolling_vol` (bitwise-identical)."""
    return _expr_rolling_vol(expr, window, op="rs_vol")


def _normalize_columns(columns: str | Sequence[str], *, op: str) -> list[str]:
    """Normalise a ``str | Sequence[str]`` column argument to a list of names."""
    if isinstance(columns, str):
        return [columns]
    cols = list(columns)
    if not cols:
        raise ValueError(
            f"panel.{op}: `columns` was empty; pass a column name or a "
            "non-empty list of column names."
        )
    return cols


def _resolve_output_names(
    cols: list[str], *, alias: str | None, suffix: str | None, op: str
) -> dict[str, str]:
    """Map each input column to its output column name.

    ``alias`` names the (single) output explicitly; ``suffix`` appends
    ``<col><suffix>`` for every column; with neither, each column is replaced in
    place. ``alias`` is single-column only — pass ``suffix`` to engineer many.
    """
    if alias is not None:
        if suffix is not None:
            raise ValueError(f"panel.{op}: pass either `alias` or `suffix`, not both.")
        if len(cols) != 1:
            raise ValueError(
                f"panel.{op}: `alias` is only valid for a single column; use "
                f"`suffix=` to feature-engineer multiple columns (got {cols})."
            )
        return {cols[0]: alias}
    if suffix is not None:
        return {c: f"{c}{suffix}" for c in cols}
    return {c: c for c in cols}


def _apply_over(
    frame: FrameT,
    build: Callable[[str], pl.Expr],
    columns: str | Sequence[str],
    *,
    over: str | None,
    alias: str | None,
    suffix: str | None,
    op: str,
) -> FrameT:
    """Apply a per-entity panel op to one or more ``columns`` within ``over``.

    Shared by the LazyFrame and DataFrame ``.panel`` namespaces. ``over`` (the
    entity key, e.g. ``ticker``) is **required**: without it the operator would
    be computed across the WHOLE frame, bleeding one entity's history into the
    next — a silent cross-entity leak in a leak-safe library. ``build`` maps a
    column name to its (ungrouped) expression; the result is grouped by ``over``
    and written per :func:`_resolve_output_names`.
    """
    cols = _normalize_columns(columns, op=op)
    if over is None:
        raise ValueError(
            f"panel.{op} requires an `over` entity key (e.g. over='ticker'). "
            "Without it the operator is computed across the WHOLE frame, "
            "bleeding one entity's history into the next (a cross-entity leak). "
            "Pass the entity column via `over=`, or drop to the expression form "
            f"pl.col(...).panel.{op}(...).over(entity) to control grouping "
            "yourself."
        )
    names = _resolve_output_names(cols, alias=alias, suffix=suffix, op=op)
    exprs = [build(c).over(over).alias(names[c]) for c in cols]
    return frame.with_columns(exprs)


class PanelExprNamespace:
    """Expression methods registered under the ``.panel`` namespace.

    Instances are created by Polars when you access ``expr.panel``; you do not
    construct this class directly. Every method returns a :class:`polars.Expr`.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def frac_diff(self, d: float, *, threshold: float = DEFAULT_THRESHOLD) -> pl.Expr:
        """Fixed-width fractional differencing (causal).

        Applies a fractional-difference filter of order ``d`` using a truncated,
        fixed-width weight kernel. Fractional differencing removes the
        unit-root / trend while preserving more memory than an integer
        difference — useful for making financial series stationary without
        destroying predictive signal.

        The implementation is a causal convolution: the value at row ``t`` is a
        weighted sum of ``x[t], x[t-1], ...`` with the weights from
        :func:`panelary._internal._ffd.ffd_weights`. No future rows are used. The first
        ``len(weights) - 1`` rows are ``null`` (insufficient history), matching
        the semantics of a rolling window.

        Parameters
        ----------
        d : float
            Order of fractional differencing (typically ``0 < d < 1``).
        threshold : float, keyword-only, default \
            :data:`~panelary._internal._ffd.DEFAULT_THRESHOLD` (``5e-4``)
            Weight-magnitude cutoff controlling the kernel width. Smaller values
            yield a longer kernel (more memory, more leading nulls).

        Returns
        -------
        pl.Expr
            The fractionally differenced series. Combine with ``.over(entity)``
            to apply per entity in a panel.

        Notes
        -----
        Because the kernel is fixed-width and applied via lagged shifts, the
        expression is fully causal and therefore leakage-safe.
        """
        return _expr_frac_diff(self._expr, d, threshold=threshold)

    def zscore(self, window: int) -> pl.Expr:
        """Causal rolling z-score over a trailing ``window``.

        Computes ``(x[t] - rolling_mean) / rolling_std`` using only the trailing
        ``window`` observations (inclusive of ``t``). Leading rows with fewer
        than ``window`` observations are ``null``.

        Parameters
        ----------
        window : int
            Trailing window length in rows; must be a positive integer.

        Returns
        -------
        pl.Expr
            The rolling z-score. Combine with ``.over(entity)`` for panels.
        """
        return _expr_zscore(self._expr, window)

    def rolling_vol(self, window: int) -> pl.Expr:
        """Trailing rolling standard deviation over ``window`` rows (causal).

        The sample standard deviation (``ddof=1``) of the input over the
        trailing ``window`` rows, inclusive of ``t``. Applied to a return
        series this is the classical close-to-close volatility; it is **not** a
        range (OHLC) estimator. For Parkinson, Garman-Klass, Rogers-Satchell or
        Yang-Zhang volatility from open/high/low/close bars, use
        :func:`panelary.econ.features.range_volatility`.

        Parameters
        ----------
        window : int
            Trailing window length in rows; must be a positive integer.

        Returns
        -------
        pl.Expr
            Rolling standard deviation. Combine with ``.over(entity)`` for
            panels.
        """
        return _expr_rolling_vol(self._expr, window)

    def rs_vol(self, window: int) -> pl.Expr:
        """Deprecated alias of :meth:`rolling_vol` (identical output).

        Despite the name this was never Rogers-Satchell: it is a trailing
        rolling standard deviation, and so is every value it has ever returned.
        It emits a :class:`FutureWarning` (once per process) and returns exactly
        :meth:`rolling_vol`'s expression. For the Rogers-Satchell OHLC
        estimator use ``panelary.econ.features.range_volatility(method=
        "rogers_satchell")``.

        Parameters
        ----------
        window : int
            Trailing window length in rows; must be a positive integer.

        Returns
        -------
        pl.Expr
            Rolling standard deviation, bitwise equal to :meth:`rolling_vol`.
        """
        _warn_rs_vol()
        return _expr_rs_vol(self._expr, window)


class _PanelFrameNamespace:
    """Shared frame-level ``.panel`` operators for LazyFrame and DataFrame.

    Each method names the target ``columns`` (a single name or a list) as its
    first argument and takes a **required** ``over`` entity key — omitting it
    raises, because a per-entity op computed across the whole frame would bleed
    one entity's history into the next (a cross-entity leak). Output naming is
    controlled by ``alias`` (single column) or ``suffix`` (``f"{col}{suffix}"``
    per column); with neither, columns are replaced in place. The expression
    itself is built by the shared ``_expr_*`` helpers, so there is no logic
    duplicated with the expression namespace.

    The panel-safety guarantee is enforced here: only the ``.over(entity)`` path
    is reachable from the frame API, so :data:`_SPECS` can honestly declare
    ``panel_safe=True``. The expression namespace stays unconstrained — the user
    composes ``.over(...)`` themselves there.

    LazyFrame and DataFrame both expose ``with_columns`` with identical
    semantics, so a single implementation serves both via :func:`_apply_over`.
    """

    _frame: pl.LazyFrame | pl.DataFrame

    def frac_diff(
        self,
        columns: str | Sequence[str],
        *,
        d: float,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Fixed-width fractional differencing of ``columns`` (causal).

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to fractionally difference.
        d : float, keyword-only
            Order of fractional differencing (typically ``0 < d < 1``).
        over : str, keyword-only
            Entity key to group by (``.over(over)``). **Required** — raises
            :class:`ValueError` if omitted, to prevent a cross-entity leak.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to replacing the
            column in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"`` (the way to
            feature-engineer multiple columns in one call).
        threshold : float, keyword-only, default \
            :data:`~panelary._internal._ffd.DEFAULT_THRESHOLD` (``5e-4``)
            Weight-magnitude cutoff controlling the kernel width.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s) (same type as the input).
        """
        return _apply_over(
            self._frame,
            lambda c: _expr_frac_diff(pl.col(c), d, threshold=threshold),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="frac_diff",
        )

    def zscore(
        self,
        columns: str | Sequence[str],
        *,
        window: int,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Causal rolling z-score of ``columns`` over a trailing ``window``.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to z-score.
        window : int, keyword-only
            Trailing window length in rows; must be a positive integer.
        over : str, keyword-only
            Entity key to group by. **Required** — raises :class:`ValueError`
            if omitted, to prevent a cross-entity leak.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        return _apply_over(
            self._frame,
            lambda c: _expr_zscore(pl.col(c), window),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="zscore",
        )

    def rolling_vol(
        self,
        columns: str | Sequence[str],
        *,
        window: int,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Trailing rolling standard deviation of ``columns`` over ``window``.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, whose trailing standard
            deviation is computed.
        window : int, keyword-only
            Trailing window length in rows; must be a positive integer.
        over : str, keyword-only
            Entity key to group by. **Required** — raises :class:`ValueError`
            if omitted, to prevent a cross-entity leak.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        return _apply_over(
            self._frame,
            lambda c: _expr_rolling_vol(pl.col(c), window),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="rolling_vol",
        )

    def rs_vol(
        self,
        columns: str | Sequence[str],
        *,
        window: int,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Deprecated alias of :meth:`rolling_vol` (identical output).

        A rolling standard deviation, never Rogers-Satchell; emits a
        :class:`FutureWarning` once per process.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, whose trailing standard
            deviation is computed.
        window : int, keyword-only
            Trailing window length in rows; must be a positive integer.
        over : str, keyword-only
            Entity key to group by. **Required** — raises :class:`ValueError`
            if omitted, to prevent a cross-entity leak.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        _warn_rs_vol()
        return _apply_over(
            self._frame,
            lambda c: _expr_rs_vol(pl.col(c), window),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="rs_vol",
        )

    # ------------------------------------------------------------------ #
    # Data validation (panelary.quality). Imported lazily inside each method
    # so this Tier-1 module keeps its import boundary: nothing from the
    # estimator layer loads until one of these is actually called.
    # ------------------------------------------------------------------ #
    def validate(
        self,
        *,
        entity: str | None = None,
        time: str | None = None,
        **kwargs: Any,
    ) -> ValidationReport:
        """Validate this frame as a panel: invariants, contracts, leak-safety.

        Thin front for :func:`panelary.quality.validate_panel` -- unique
        ``(entity, time)`` keys, per-entity time order, gaps, history and
        coverage, optional column contracts, and (given ``split=``) the
        leak-safety checks. A LazyFrame is collected once.

        Parameters
        ----------
        entity, time : str, optional
            Panel keys. Default: column 0 is the entity, column 1 the time.
        **kwargs
            Forwarded to :func:`panelary.quality.validate_panel` (``schema``,
            ``min_obs``, ``frequency``, ``impact``, ``split``, ``fitted``,
            ``raise_on_fail``, ...).

        Returns
        -------
        panelary.quality.ValidationReport

        Raises
        ------
        panelary.quality.PanelValidationError
            If a check with ``impact="fail"`` failed and ``raise_on_fail`` is
            True (the default).
        """
        from panelary.quality import validate_panel

        return validate_panel(self._frame, entity=entity, time=time, **kwargs)

    def quality_report(
        self,
        *,
        entity: str | None = None,
        time: str | None = None,
        **kwargs: Any,
    ) -> QualityReport:
        """Profile this frame: null %, duplicate %, constant columns, dtype drift.

        Thin front for :func:`panelary.quality.quality_report`. With
        ``entity`` and ``time`` the profile is panel-aware (duplicate keys,
        entities per time, constancy within each entity); without them it is
        frame-level only.

        Parameters
        ----------
        entity, time : str, optional
            Panel keys (both or neither).
        **kwargs
            Forwarded to :func:`panelary.quality.quality_report`
            (``reference``, ``columns``, ``max_null_pct``, ``max_examples``).

        Returns
        -------
        panelary.quality.QualityReport
        """
        from panelary.quality import quality_report

        return quality_report(self._frame, entity=entity, time=time, **kwargs)

    # ------------------------------------------------------------------ #
    # Data cleaning (panelary.clean). Imported lazily, like the quality
    # methods above.
    # ------------------------------------------------------------------ #
    def dedup(
        self,
        *,
        entity: str | None = None,
        time: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Remove (or flag) exact and near-duplicate rows.

        Thin front for :func:`panelary.clean.dedup`: panel-global and
        point-in-time by default -- a row is dropped only if it copies an
        *earlier* row, so no near-duplicate pair can straddle a split built
        afterwards. The result has the same container type as the frame.

        Parameters
        ----------
        entity, time : str, optional
            Panel keys. Default: column 0 is the entity, column 1 the time.
        **kwargs
            Any :class:`panelary.clean.Deduplicator` parameter (``method``,
            ``threshold``, ``columns``, ``action``, ...).
        """
        from panelary.clean import dedup as _dedup

        return _dedup(self._frame, entity=entity, time=time, **kwargs)


class PanelLazyFrameNamespace(_PanelFrameNamespace):
    """Frame-level ``.panel`` operators on :class:`polars.LazyFrame`.

    Accessed as ``lf.panel.<op>(...)``. Returns a new :class:`polars.LazyFrame`.
    """

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._frame = lf


class PanelDataFrameNamespace(_PanelFrameNamespace):
    """Frame-level ``.panel`` operators on :class:`polars.DataFrame`.

    Accessed as ``df.panel.<op>(...)``. Returns a new :class:`polars.DataFrame`.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._frame = df


def register() -> None:
    """Register the ``.panel`` namespace and its feature specs (idempotent).

    Registration is guarded so that re-importing this module (a common
    occurrence under reloaders, test runners, or repeated imports) does not
    raise the Polars "namespace already registered" error.
    """
    # Polars exposes custom expression namespaces as attributes on ``pl.Expr``.
    # Re-registering an existing name triggers a noisy "overriding existing
    # custom namespace" UserWarning, so only register when absent — this makes
    # re-import a true no-op. The try/except is a belt-and-braces guard for the
    # documented "already registered" condition across Polars versions.
    if not _is_registered(pl.Expr, "panel"):
        try:
            pl.api.register_expr_namespace("panel")(PanelExprNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    # Frame-level ergonomics: ``lf.panel.<op>`` / ``df.panel.<op>``. Guarded the
    # same way to keep re-import a no-op and avoid the override UserWarning.
    if not _is_registered(pl.LazyFrame, "panel"):
        try:
            pl.api.register_lazyframe_namespace("panel")(PanelLazyFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise
    if not _is_registered(pl.DataFrame, "panel"):
        try:
            pl.api.register_dataframe_namespace("panel")(PanelDataFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    for spec in _SPECS:
        registry.register(spec)


def _is_registered(cls: type, name: str) -> bool:
    """Return ``True`` if a custom namespace ``name`` already exists on ``cls``."""
    return hasattr(cls, name)


# ``panel_safe=True`` is now an enforced guarantee, not a hope: the frame-level
# ``.panel`` API requires an ``over`` entity key (see ``_apply_over``), so a
# frame op can no longer be silently computed across entity boundaries. The
# expression form is panel-safe once the user composes ``.over(entity)``.
#
# All four are ``safe_scope="rowwise"``: each is a ``series -> series`` filter
# that emits one value per row from a bounded *trailing* neighbourhood of that
# row — a fixed-width causal FIR convolution (``frac_diff``) or a trailing
# rolling window (``zscore``, ``rolling_vol`` and its deprecated alias
# ``rs_vol``). The value at ``t`` is a function of
# ``x[t], x[t-1], ...`` only, so evaluating them at every row is the intended
# and safe usage; there is no window-summary caveat to attach.
_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="frac_diff",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"d": float, "threshold": float},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="zscore",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"window": int},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="rolling_vol",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"window": int},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
        axis="time",
        flavour="trailing",
        cost_hint="O(N)",
    ),
    FeatureSpec(
        name="rs_vol",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        params={"window": int},
        tier="D",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=f"{_SOURCE} (deprecated alias of rolling_vol)",
        license=_LICENSE,
    ),
)


register()
