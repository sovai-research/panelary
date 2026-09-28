"""Intent, axis and flavour: the vocabulary and the base class of the shape algebra.

A shape transform changes the *shape* of a panel -- the width of one named axis,
the order of the tensor, or the numerical rank of a factorisation -- while
leaving the meaning of ``entity`` / ``time`` / ``feature`` intact. This module
holds the four pieces every transform in :mod:`panelary.shape` is built from:

* :class:`Intent`, :class:`Axis`, :class:`Flavour` -- the closed vocabularies.
  The axis label is not decoration: it *is* the leak contract (see the table on
  :class:`Axis`), and :class:`ShapeTransform` refuses, at class-definition time,
  a declaration that contradicts it.
* :class:`ShapeSpec` -- the per-class (and per-instance, resolved) shape
  contract, mirroring the optional shape fields on
  :class:`panelary.registry.FeatureSpec`.
* :class:`Plan` / :class:`ShapeBudgetError` -- predicted output shape and peak
  scratch bytes, computed from the input shape and the parameters alone, so a
  chain of transforms can refuse an accidental O(T^2) expansion *before* it
  allocates anything.
* :class:`ShapeTransform` -- the :class:`~panelary.core.protocol.PanelTransformer`
  subclass that every shape transform (and every ``embed/`` transform) derives
  from. It declares the two enforced class attributes the ``embed/`` contract
  introduced, ``fit_is_empty`` and ``is_cross_sectional``, here -- once.

Enforcement, not convention
---------------------------
``fit_is_empty = True`` is enforced **structurally**: :meth:`ShapeTransform.fit`
hands ``_fit`` a zero-row frame with the input's schema, so a transform that
claims an empty fit cannot read a single data row even by accident. Whatever it
stores can depend on the column names, the parameters and the seed -- nothing
else. ``tests/test_shape_stateless.py`` then checks the consequence (fitting on
two disjoint datasets gives byte-identical state).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

if TYPE_CHECKING:
    import sys
    from collections.abc import Sequence

    if sys.version_info >= (3, 11):
        from typing import Self
    else:  # pragma: no cover - typing_extensions ships with every type checker
        from typing_extensions import Self

    from numpy.typing import NDArray

__all__ = [
    "Axis",
    "Flavour",
    "Intent",
    "Plan",
    "ShapeBudgetError",
    "ShapeSpec",
    "ShapeTransform",
    "default_max_bytes",
    "plan_chain",
]


class _StrEnum(str, enum.Enum):
    """``enum.StrEnum`` for Python 3.10 (``StrEnum`` itself is 3.11+)."""

    def __str__(self) -> str:
        return str(self.value)


class Intent(_StrEnum):
    """What a transform does to the shape of its input. A transform declares one.

    ``COMPRESS``
        Narrows one named axis; same axes out, one of them narrower.
    ``LIFT``
        Adds width or a new axis of declared width (a basis expansion).
    ``FACTORIZE``
        Returns factors (plus an optional reconstruction), not just a frame.
    ``SKETCH``
        Maintains bounded, mergeable state over a stream.
    """

    COMPRESS = "compress"
    LIFT = "lift"
    FACTORIZE = "factorize"
    SKETCH = "sketch"


class Axis(_StrEnum):
    """The semantic axis a transform acts along -- and therefore its leak contract.

    ============  ===============================  ===========  =====================
    ``axis=``     means                            panel_safe   leakage_safe
    ============  ===============================  ===========  =====================
    ``feature``   across columns within a row      True         iff fit on train rows
    ``time``      along one entity's history       True         iff flavour trailing
    ``entity``    across entities at one date      **False**    True (date-local)
    ``lag``       the axis a ``LIFT`` creates      inherited    inherited
    ============  ===============================  ===========  =====================

    ``axis="entity"`` is the *safe* direction in time and ``axis="time"`` the
    dangerous one -- the opposite of the usual intuition. A cross-sectional
    reduction refit each date sees only that date; a whole-series reduction over
    time sees the future.
    """

    FEATURE = "feature"
    TIME = "time"
    ENTITY = "entity"
    LAG = "lag"


class Flavour(_StrEnum):
    """The two flavours of every time-axis transform. There is no third.

    ``TRAILING`` (the default)
        One output row per input row, from that row's own trailing ``window``
        observations within its entity. ``leakage_safe = True``.
    ``WHOLE_SERIES`` (explicit opt-in)
        One output row per **entity**, from all of that entity's data.
        ``leakage_safe = False``. The result is keyed by ``entity`` alone, so
        using it as a row feature is a join error rather than a silent leak.
    """

    TRAILING = "trailing"
    WHOLE_SERIES = "whole_series"


#: How a transform's output width is determined.
WIDTH_RULES: frozenset[str] = frozenset({"exact", "rank_dependent", "data_dependent"})
#: Whether (and how) a transform can be inverted.
INVERTIBLE: frozenset[str] = frozenset({"exact", "approximate", "from_factors", "none"})
#: How a transform consumes data.
STREAMING: frozenset[str] = frozenset({"batch", "partial_fit", "mergeable"})


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    """The shape contract of a transform: class default, or resolved per instance.

    The fields mirror the optional shape fields of
    :class:`panelary.registry.FeatureSpec` (``intent``, ``axis``, ``flavour``,
    ``width_rule``, ``invertible``, ``streaming``, ``cost_hint``) plus the two
    widths, which only an instance can know.

    Parameters
    ----------
    intent : Intent or str
        One of :class:`Intent`.
    axis : Axis or str
        One of :class:`Axis`.
    flavour : Flavour or str, optional
        Required for ``axis="time"``; ``None`` otherwise.
    in_width, out_width : int, optional
        Width of the transformed axis in and out. ``None`` until known.
    width_rule : str, default "exact"
        One of :data:`WIDTH_RULES`.
    invertible : str, default "none"
        One of :data:`INVERTIBLE`.
    streaming : str, default "batch"
        One of :data:`STREAMING`.
    cost_hint : str, default ""
        Big-O cost, e.g. ``"O(n d k)"``. A ``T^2`` here keeps a spec out of
        tiers A/B (see :meth:`panelary.registry.FeatureRegistry.audit`).
    """

    intent: Intent
    axis: Axis
    flavour: Flavour | None = None
    in_width: int | None = None
    out_width: int | None = None
    width_rule: str = "exact"
    invertible: str = "none"
    streaming: str = "batch"
    cost_hint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", Intent(self.intent))
        object.__setattr__(self, "axis", Axis(self.axis))
        if self.flavour is not None:
            object.__setattr__(self, "flavour", Flavour(self.flavour))
        for name, value, allowed in (
            ("width_rule", self.width_rule, WIDTH_RULES),
            ("invertible", self.invertible, INVERTIBLE),
            ("streaming", self.streaming, STREAMING),
        ):
            if value not in allowed:
                raise ValueError(
                    f"ShapeSpec.{name} must be one of {sorted(allowed)}, got {value!r}."
                )
        if self.axis is Axis.TIME and self.flavour is None:
            raise ValueError(
                "ShapeSpec(axis='time') must declare a flavour ('trailing' or "
                "'whole_series'): along time, the flavour *is* the leak contract."
            )

    def resolved(self, **changes: Any) -> ShapeSpec:
        """Return a copy with ``changes`` applied (e.g. widths, flavour)."""
        return replace(self, **changes)

    def registry_fields(self) -> dict[str, Any]:
        """The :class:`~panelary.registry.FeatureSpec` shape fields, as strings."""
        return {
            "intent": str(self.intent),
            "axis": str(self.axis),
            "flavour": None if self.flavour is None else str(self.flavour),
            "width_rule": self.width_rule,
            "invertible": self.invertible,
            "streaming": self.streaming,
            "cost_hint": self.cost_hint or None,
        }


# --------------------------------------------------------------------------- #
# Plan and budget
# --------------------------------------------------------------------------- #
class ShapeBudgetError(MemoryError):
    """A planned step would exceed the scratch-memory budget.

    Raised by :meth:`Plan.check` -- and therefore by every
    :class:`ShapeTransform` before it allocates -- naming the offending step and
    the parameter to lower. The estimate is a guardrail against
    order-of-magnitude mistakes, not an accounting system: LAPACK workspace is
    not modelled.
    """


_FALLBACK_MAX_BYTES = 8 * 1024**3


def default_max_bytes() -> int:
    """Default scratch budget: 25% of free memory if ``psutil`` is importable, else 8 GiB.

    ``psutil`` is never required -- it is consulted only if it happens to be
    installed, and the fixed fallback is used otherwise.
    """
    try:
        import psutil  # optional; deliberately not routed through require()
    except ImportError:
        return _FALLBACK_MAX_BYTES
    try:
        return int(psutil.virtual_memory().available * 0.25)
    except Exception:  # noqa: BLE001 - any psutil failure falls back
        return _FALLBACK_MAX_BYTES


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"  # pragma: no cover - unreachable


@dataclass(frozen=True, slots=True)
class Plan:
    """What a step will produce and roughly what it will cost, before it runs.

    Attributes
    ----------
    step : str
        The transform's class name (or a user label in :func:`plan_chain`).
    in_rows, in_width : int
        The input shape the plan was computed for.
    rows, width : int
        Predicted output shape: rows of the returned frame and number of emitted
        value columns.
    output : str
        What a row of the output is keyed by: ``"rows"`` (entity, time),
        ``"entities"`` (entity only -- the ``whole_series`` flavour),
        ``"factors"`` or ``"state"``.
    peak_bytes : int
        Predicted peak scratch bytes (float64 working arrays plus the output).
    cost_hint : str
        The transform's big-O hint.
    knob : str
        The parameter to lower if the step is over budget.
    """

    step: str
    in_rows: int
    in_width: int
    rows: int
    width: int
    output: str
    peak_bytes: int
    cost_hint: str
    knob: str

    def check(self, max_bytes: int | None = None) -> Plan:
        """Raise :class:`ShapeBudgetError` if ``peak_bytes`` exceeds ``max_bytes``.

        Parameters
        ----------
        max_bytes : int, optional
            The budget. ``None`` uses :func:`default_max_bytes`.

        Returns
        -------
        Plan
            ``self``, for chaining.

        Raises
        ------
        ShapeBudgetError
            If the predicted peak exceeds the budget.
        """
        budget = default_max_bytes() if max_bytes is None else int(max_bytes)
        if self.peak_bytes > budget:
            raise ShapeBudgetError(
                f"{self.step}: this step would materialize about "
                f"{_fmt_bytes(self.peak_bytes)} of scratch for an input of "
                f"{self.in_rows:,} rows x {self.in_width:,} columns "
                f"(cost {self.cost_hint or 'unstated'}), over the "
                f"{_fmt_bytes(budget)} budget. Lower `{self.knob}`, reduce the "
                "input, or raise `max_bytes=` if you really mean it."
            )
        return self


@dataclass(frozen=True, slots=True)
class InputShape:
    """A frame's shape, as far as planning needs it (no data)."""

    rows: int
    width: int
    entities: int
    columns: tuple[str, ...] = ()


def plan_chain(
    steps: Sequence[ShapeTransform | tuple[str, ShapeTransform]],
    X: PanelFrame | pl.DataFrame | pl.LazyFrame | InputShape,
    *,
    max_bytes: int | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> list[Plan]:
    """Plan every step of a chain before executing any of them.

    Each step's predicted output shape is fed to the next step's planner, so a
    ``lift -> compress`` chain reports "this will materialize 41 GB" before the
    first byte is allocated. Every plan is checked against ``max_bytes``.

    Parameters
    ----------
    steps : sequence of ShapeTransform or (name, ShapeTransform)
        The chain, in order (a :class:`~panelary.core.pipeline.Pipeline`'s
        ``steps`` list works as-is).
    X : PanelFrame | polars.DataFrame | polars.LazyFrame | InputShape
        The chain's input.
    max_bytes : int, optional
        Budget per step; ``None`` uses :func:`default_max_bytes`.
    entity, time : str, optional
        Keys for a bare polars frame.

    Returns
    -------
    list of Plan

    Raises
    ------
    ShapeBudgetError
        Naming the first step that would exceed the budget.
    """
    plans: list[Plan] = []
    shape: Any = X
    for item in steps:
        name, step = item if isinstance(item, tuple) else (None, item)
        if not isinstance(step, ShapeTransform):
            raise TypeError(
                f"plan_chain: step {name or step!r} is not a ShapeTransform, so it "
                "has no planner; plan the shape-changing steps only."
            )
        p = step.plan(shape, entity=entity, time=time)
        if name is not None:
            p = replace(p, step=f"{name} ({p.step})")
        p.check(max_bytes)
        plans.append(p)
        shape = InputShape(
            rows=p.rows,
            width=p.width,
            entities=getattr(shape, "entities", 0)
            if isinstance(shape, InputShape)
            else 0,
        )
    return plans


# --------------------------------------------------------------------------- #
# Contract checks shared by class definition and instance resolution
# --------------------------------------------------------------------------- #
def _check_contract(
    owner: str,
    spec: ShapeSpec,
    *,
    panel_safe: bool,
    leakage_safe: bool,
    is_cross_sectional: bool,
) -> None:
    """Refuse a declaration that contradicts the axis/flavour table."""
    if spec.flavour is Flavour.WHOLE_SERIES and leakage_safe:
        raise TypeError(
            f"{owner}: flavour='whole_series' with leakage_safe=True. A "
            "whole-series transform summarises each entity's entire history, "
            "future included; it can never be leakage-safe as a row feature."
        )
    if spec.axis is Axis.ENTITY and panel_safe:
        raise TypeError(
            f"{owner}: axis='entity' with panel_safe=True. A transform across "
            "entities at one date mixes entities by design; declare "
            "panel_safe=False."
        )
    if is_cross_sectional and (panel_safe or spec.axis is not Axis.ENTITY):
        raise TypeError(
            f"{owner}: is_cross_sectional=True requires axis='entity' and "
            "panel_safe=False -- it fits per date, across entities."
        )


# --------------------------------------------------------------------------- #
# The base class
# --------------------------------------------------------------------------- #
class ShapeTransform(PanelTransformer):
    """Base class of every shape transform (and of every ``embed/`` transform).

    Concrete subclasses must declare, at class level:

    * ``panel_safe`` / ``leakage_safe`` -- the :class:`PanelTransformer` contract;
    * ``fit_is_empty`` -- ``True`` if ``transform`` is a pure function of each
      row's own data (and the seed): nothing is learned. **Enforced**: ``_fit``
      is handed a zero-row frame, so it cannot read a data row;
    * ``is_cross_sectional`` -- ``True`` if the transform fits per date across
      entities (``axis="entity"``), never across dates;
    * ``spec`` -- a :class:`ShapeSpec` (intent, axis, flavour, ...).

    Declarations that contradict the axis table (a ``whole_series`` flavour
    claiming ``leakage_safe``, an ``entity`` axis claiming ``panel_safe``) are
    refused when the class is defined. An *instance* may narrow its contract
    (a ``whole_series`` PAA sets ``leakage_safe = False`` on itself; a
    ``PartialTucker`` compressing the entity mode sets ``panel_safe = False``) --
    never widen it.

    Parameters
    ----------
    columns : sequence of str, optional
        Value columns to transform. ``None`` = every numeric feature column.
    max_bytes : int, optional
        Scratch budget checked (via :meth:`plan`) before any allocation;
        ``None`` uses :func:`default_max_bytes`.
    entity, time : str, optional
        Default panel keys for bare polars frames.
    """

    spec: ClassVar[ShapeSpec] = None  # type: ignore[assignment]
    fit_is_empty: ClassVar[bool] = None  # type: ignore[assignment]
    is_cross_sectional: ClassVar[bool] = None  # type: ignore[assignment]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        is_abstract = any(
            getattr(getattr(cls, name, None), "__isabstractmethod__", False)
            for name in dir(cls)
        )
        if is_abstract:
            return
        for attr in ("fit_is_empty", "is_cross_sectional"):
            value = getattr(cls, attr, None)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{cls.__name__} must declare a class-level boolean `{attr}` "
                    f"(got {value!r}). `fit_is_empty = True` means transform() is "
                    "a pure function of each row's own window (and the seed); "
                    "`is_cross_sectional = True` means it fits per date, never "
                    "across dates. Both are enforced, not documentary."
                )
        if not isinstance(getattr(cls, "spec", None), ShapeSpec):
            raise TypeError(
                f"{cls.__name__} must declare a class-level `spec = ShapeSpec(...)` "
                "naming its intent and axis (and flavour, for axis='time')."
            )
        _check_contract(
            cls.__name__,
            cls.spec,
            panel_safe=cls.panel_safe,
            leakage_safe=cls.leakage_safe,
            is_cross_sectional=cls.is_cross_sectional,
        )

    def __init__(
        self,
        *,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if isinstance(columns, str):
            columns = [columns]
        self.columns: list[str] | None = list(columns) if columns is not None else None
        self.max_bytes = max_bytes
        self.feature_names_in_: list[str] = []

    # ------------------------------------------------------------------ #
    # Instance-level contract
    # ------------------------------------------------------------------ #
    def _narrow_contract(
        self,
        *,
        panel_safe: bool | None = None,
        leakage_safe: bool | None = None,
    ) -> None:
        """Narrow (never widen) this instance's ``panel_safe`` / ``leakage_safe``."""
        if panel_safe is not None:
            if panel_safe and not type(self).panel_safe:
                raise TypeError(f"{type(self).__name__}: cannot widen panel_safe.")
            self.panel_safe = panel_safe
        if leakage_safe is not None:
            if leakage_safe and not type(self).leakage_safe:
                raise TypeError(f"{type(self).__name__}: cannot widen leakage_safe.")
            self.leakage_safe = leakage_safe
        _check_contract(
            type(self).__name__,
            self.shape_spec,
            panel_safe=self.panel_safe,
            leakage_safe=self.leakage_safe,
            is_cross_sectional=self.is_cross_sectional,
        )

    @property
    def shape_spec(self) -> ShapeSpec:
        """This instance's resolved :class:`ShapeSpec` (flavour, widths)."""
        return self._resolve_spec(type(self).spec)

    def _resolve_spec(self, spec: ShapeSpec) -> ShapeSpec:
        """Hook: fill in instance-specific fields. Default adds the widths."""
        in_w = len(self.feature_names_in_) if self.feature_names_in_ else None
        out_w = self._out_width(in_w) if in_w is not None else None
        return spec.resolved(in_width=in_w, out_width=out_w)

    def _out_width(self, in_width: int) -> int | None:
        """Hook: output width for an input of ``in_width`` columns."""
        return None

    # ------------------------------------------------------------------ #
    # Columns
    # ------------------------------------------------------------------ #
    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        """Resolve and validate the numeric value columns to transform."""
        schema = panel.schema
        if self.columns is not None:
            missing = [c for c in self.columns if c not in panel]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: column(s) {missing} not found in "
                    f"panel. Available columns: {panel.columns}."
                )
            cols = list(self.columns)
        else:
            cols = [c for c in panel.feature_cols if schema[c].is_numeric()]
        if not cols:
            raise ValueError(
                f"{type(self).__name__}: no numeric feature columns to transform "
                f"(columns={panel.columns}). Pass `columns=` explicitly."
            )
        bad = [c for c in cols if not schema[c].is_numeric()]
        if bad:
            raise ValueError(
                f"{type(self).__name__}: column(s) {bad} are not numeric. Encode "
                "them first or pass a numeric `columns=` list."
            )
        return cols

    # ------------------------------------------------------------------ #
    # fit: the fit_is_empty enforcement
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> Self:
        """Learn parameters from ``X`` (training rows only).

        When the class declares ``fit_is_empty = True``, ``_fit`` receives a
        **zero-row** frame with ``X``'s schema: the fitted state can depend on
        column names, parameters and the seed, and provably on nothing else.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Training data.
        entity, time : str, optional
            Panel keys for a bare polars frame.

        Returns
        -------
        Self
        """
        panel = self._as_panel(X, method="fit", entity=entity, time=time)
        if self.fit_is_empty:
            schema_only = PanelFrame(
                panel.lazy().clear(),
                entity=panel.entity_col,
                time=panel.time_col,
                validate=False,
            )
            self._fit(schema_only)
        else:
            self._fit(panel)
        self._fit_panel = panel
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # plan
    # ------------------------------------------------------------------ #
    def _input_shape(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame | InputShape,
        *,
        entity: str | None,
        time: str | None,
    ) -> InputShape:
        if isinstance(X, InputShape):
            return X
        panel = self._as_panel(X, method="plan", entity=entity, time=time)
        cols = self.feature_names_in_ or self._resolve_columns(panel)
        counts = (
            panel.lazy()
            .select(
                pl.len().alias("rows"),
                pl.col(panel.entity_col).n_unique().alias("entities"),
            )
            .collect()
        )
        return InputShape(
            rows=int(counts["rows"][0]),
            width=len(cols),
            entities=int(counts["entities"][0]),
            columns=tuple(cols),
        )

    def plan(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame | InputShape,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> Plan:
        """Predict output shape and peak scratch bytes, **without executing**.

        Computed from the input shape and the parameters alone (one cheap
        ``count`` over a lazy input; no transform work).

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame | InputShape
            The input, or just its shape.
        entity, time : str, optional
            Keys for a bare polars frame.

        Returns
        -------
        Plan
        """
        shape = self._input_shape(X, entity=entity, time=time)
        return self._plan(shape)

    def _plan(self, shape: InputShape) -> Plan:
        """Hook: subclasses override with their own shape/byte model."""
        out_w = self._out_width(shape.width) or shape.width
        return self._make_plan(
            shape,
            rows=shape.rows,
            width=out_w,
            scratch=8 * shape.rows * (shape.width + out_w),
            knob="columns",
        )

    def _make_plan(
        self,
        shape: InputShape,
        *,
        rows: int,
        width: int,
        scratch: float,
        knob: str,
        output: str = "rows",
    ) -> Plan:
        return Plan(
            step=type(self).__name__,
            in_rows=shape.rows,
            in_width=shape.width,
            rows=int(rows),
            width=int(width),
            output=output,
            peak_bytes=int(min(scratch, 2**62)),
            cost_hint=self.shape_spec.cost_hint,
            knob=knob,
        )

    def _enforce_budget(self, shape: InputShape) -> Plan:
        """Plan for ``shape`` and raise :class:`ShapeBudgetError` if over budget."""
        return self._plan(shape).check(self.max_bytes)

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self) -> pl.DataFrame:
        """Tidy description of what each output component is made of.

        Returns
        -------
        polars.DataFrame
            Columns ``component``, ``source_feature``, ``loading``,
            ``abs_loading``, ``rank``, ``explained_variance_ratio``,
            ``reconstruction_error`` (see :mod:`panelary.shape._explain`).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        NotImplementedError
            If the transform has no loadings to report.
        """
        self._check_fitted("explain")
        return self._explain()

    def _explain(self) -> pl.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement explain()."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.shape_spec
        flav = f", flavour={s.flavour}" if s.flavour is not None else ""
        state = "fitted" if getattr(self, "_fitted", False) else "not fitted"
        return (
            f"{type(self).__name__}(intent={s.intent}, axis={s.axis}{flav}, "
            f"panel_safe={self.panel_safe}, leakage_safe={self.leakage_safe}, "
            f"{state})"
        )


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def check_seed(seed: object, owner: str) -> int:
    """Validate an explicit integer seed (``AGENTS.md`` invariant 2)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"{owner}: `seed` must be an explicit int (never None / a global RNG), "
            f"got {seed!r}."
        )
    return int(seed)


def check_positive_int(value: object, name: str, owner: str) -> int:
    """Validate a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{owner}: `{name}` must be a positive int, got {value!r}.")
    return int(value)


def relative_frobenius_error(X: NDArray[Any], X_hat: NDArray[Any]) -> float:
    """``||X - X_hat||_F / ||X||_F`` over the finite cells of ``X`` (0 if ``X`` is 0)."""
    X = np.asarray(X, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)
    ok = np.isfinite(X) & np.isfinite(X_hat)
    num = float(np.sqrt(np.sum((X[ok] - X_hat[ok]) ** 2)))
    den = float(np.sqrt(np.sum(X[ok] ** 2)))
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def register_shape_spec(
    cls: type[ShapeTransform],
    *,
    name: str,
    params: dict[str, Any],
    tier: str,
    safe_scope: str,
    source: str,
    input_shape: str = "frame",
    output_shape: str = "frame",
    license: str = "Apache-2.0",  # noqa: A002 - mirrors FeatureSpec.license
) -> None:
    """Register ``cls`` in :data:`panelary.registry.registry` under namespace ``shape``.

    The contract fields (``panel_safe``, ``leakage_safe``) and the shape fields
    (``intent``, ``axis``, ``flavour``, ...) are read off the class, so the
    catalogue cannot disagree with the code. Re-registration of an identical
    spec (a module re-import) is a no-op.
    """
    from panelary.registry import FeatureSpec, registry

    registry.register(
        FeatureSpec(
            name=name,
            namespace="shape",
            input_shape=input_shape,
            output_shape=output_shape,
            params=params,
            tier=tier,
            panel_safe=bool(cls.panel_safe),
            leakage_safe=bool(cls.leakage_safe),
            safe_scope=safe_scope,
            source=source,
            license=license,
            backend_fn=cls,
            **cls.spec.registry_fields(),
        )
    )
