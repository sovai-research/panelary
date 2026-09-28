"""``PanelValidator`` and ``validate_panel``: one call, every invariant.

The validator runs, in a fixed order, the panel invariants of
:mod:`~panelary.quality._panel`, the column contracts of
:mod:`~panelary.quality._schema` and -- when given a split -- the leak-safety
invariants of :mod:`~panelary.quality._leakage`, then:

* **splits rows** dataframely-style into ``report.valid`` and
  ``report.invalid`` (the latter with a ``__failed_checks__`` column), using
  only row-level checks whose impact is ``"fail"``;
* **warns** once per failed ``"warn"`` check -- as
  :class:`~panelary.core.panel_frame.PanelOrderWarning` for row order,
  :class:`~panelary.preprocessing.LeakageWarning` for leak-safety, and
  :class:`~panelary.quality.QualityWarning` otherwise;
* **raises** :class:`~panelary.quality.PanelValidationError` (carrying the
  report) if any ``"fail"`` check failed, unless ``raise_on_fail=False``.

Validation learns nothing, so it has no ``fit``: per the cleaning doctrine it is
deterministic identity work that runs on all rows before any split. The one
learned quantity in sight -- a transform's fitted state -- is exactly what the
``fitted_state`` check audits.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import polars as pl

from panelary.quality._common import (
    CheckResult,
    Impact,
    ValidationReport,
    check_impact,
    panel_counts,
    produced_by,
    resolve_keys,
    split_rows,
)
from panelary.quality._leakage import fitted_state_check, split_masks, straddle_check
from panelary.quality._panel import (
    Frequency,
    _normalise_frequency,
    check_coverage,
    check_gaps,
    check_key_columns,
    check_min_obs,
    check_monotone_time,
    check_null_keys,
    check_unique_keys,
)
from panelary.quality._schema import (
    ColumnContract,
    contracts_to_dataframely,
    is_dataframely_schema,
    normalise_schema,
    run_dataframely,
    run_schema,
)

__all__ = ["DEFAULT_IMPACT", "PanelValidator", "validate_panel"]

#: The impact of each check when the caller does not override it. Structural
#: breakage fails; row order, gaps and coverage warn (the panel is usable, but a
#: window may mean something other than it says).
DEFAULT_IMPACT: Mapping[str, Impact] = MappingProxyType(
    {
        "null_keys": "fail",
        "unique_keys": "fail",
        "monotone_time": "warn",
        "gaps": "warn",
        "min_obs": "fail",
        "coverage": "warn",
        "columns_present": "fail",
        "extra_columns": "fail",
        "dtype": "fail",
        "nullability": "fail",
        "nan": "fail",
        "unique": "fail",
        "allowed_values": "fail",
        "range": "fail",
        "rule": "fail",
        "dataframely": "fail",
        "fitted_state": "fail",
        "near_duplicate_straddle": "fail",
    }
)

Backend = Literal["polars", "dataframely"]


def _warning_category(check: CheckResult) -> type[Warning]:
    if check.name == "monotone_time":
        from panelary.core.panel_frame import PanelOrderWarning

        return PanelOrderWarning
    if check.category == "leakage":
        from panelary.preprocessing._base import LeakageWarning

        return LeakageWarning
    from panelary.quality._common import QualityWarning

    return QualityWarning


class PanelValidator:
    """Validate panel invariants, column contracts and leak-safety in one call.

    Parameters
    ----------
    entity, time : str, optional
        Panel keys. A :class:`~panelary.core.panel_frame.PanelFrame` input
        supplies its own; a bare frame without them follows the codebase
        convention (column 0 is the entity, column 1 the time).
    schema : mapping or dataframely.Schema subclass, optional
        Column contracts: ``{column: ColumnContract | polars dtype | family}``
        (see :class:`ColumnContract`), or a ``dataframely.Schema`` subclass to
        delegate to (requires the ``schema`` extra).
    strict : bool, default=False
        With a mapping ``schema``, fail ``extra_columns`` for any column that is
        neither a key nor in the contract.
    rules : mapping of str -> polars.Expr, optional
        Named row-level boolean rules; ``True`` (or null) is valid.
    frequency : str | timedelta | int | float, optional
        Expected time step for ``gaps`` / ``coverage``: a polars duration
        (``"1d"``, ``"1mo"``) or timedelta for a Date/Datetime axis, a number
        for a numeric one. Default: the panel's own calendar (every time any
        entity is observed), which needs no business-day configuration.
    min_obs : int, optional
        Minimum distinct observed times per entity (``min_obs`` check).
    min_coverage : float, optional
        Minimum fraction of the calendar each entity must cover
        (``coverage`` check), in ``(0, 1]``.
    impact : mapping of check name -> {"fail", "warn", "off"}, optional
        Overrides of :data:`DEFAULT_IMPACT`, e.g.
        ``{"monotone_time": "fail", "gaps": "off"}``.
    backend : {"polars", "dataframely"}, default="polars"
        Who evaluates a mapping ``schema``'s dtype / nullability contracts.
        ``"dataframely"`` translates them into a ``dataframely.Schema``;
        uniqueness, allowed values, ranges and rules stay pure Polars.
    near_duplicates : bool or mapping, default=True
        When a ``split`` is passed to :meth:`validate`, check that no
        near-duplicate cluster straddles it. A mapping is forwarded to
        :func:`panelary.clean.near_duplicate_clusters` (``columns``,
        ``method``, ``threshold``, ``num_perm``, ``seed``) or may carry
        ``{"clusters": <column name>}`` to use precomputed ids. ``False``
        skips the check.
    max_examples : int, default=5
        Size of every check's offending-key sample.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.quality import PanelValidator
    >>> df = pl.DataFrame(
    ...     {"id": ["a", "a", "b", "b"], "t": [1, 2, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]}
    ... )
    >>> report = PanelValidator(entity="id", time="t").validate(df, raise_on_fail=False)
    >>> report.status, report.check("unique_keys").n_failing
    ('fail', 2)
    >>> report.valid.height, report.invalid.height
    (2, 2)
    """

    def __init__(
        self,
        *,
        entity: str | None = None,
        time: str | None = None,
        schema: Mapping[str, Any] | Any = None,
        strict: bool = False,
        rules: Mapping[str, pl.Expr] | None = None,
        frequency: Frequency | None = None,
        min_obs: int | None = None,
        min_coverage: float | None = None,
        impact: Mapping[str, Impact] | None = None,
        backend: Backend = "polars",
        near_duplicates: bool | Mapping[str, Any] = True,
        max_examples: int = 5,
    ) -> None:
        if backend not in ("polars", "dataframely"):
            raise ValueError(
                f"`backend` must be 'polars' or 'dataframely', got {backend!r}."
            )
        impacts: dict[str, Impact] = dict(DEFAULT_IMPACT)
        for name, level in (impact or {}).items():
            if name not in DEFAULT_IMPACT:
                raise ValueError(
                    f"unknown check {name!r} in `impact`; known checks: "
                    f"{sorted(DEFAULT_IMPACT)}."
                )
            impacts[name] = check_impact(level, name=name)
        if min_obs is not None and (
            isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 1
        ):
            raise ValueError(f"`min_obs` must be a positive integer, got {min_obs!r}.")
        if min_coverage is not None and not 0.0 < float(min_coverage) <= 1.0:
            raise ValueError(
                f"`min_coverage` must lie in (0, 1], got {min_coverage!r}."
            )
        if not isinstance(max_examples, int) or max_examples < 0:
            raise ValueError(
                f"`max_examples` must be a non-negative integer, got {max_examples!r}."
            )
        self._dy_schema: Any = None
        self._contracts: dict[str, ColumnContract] = {}
        if schema is not None:
            if is_dataframely_schema(schema):
                self._dy_schema = schema
            elif isinstance(schema, Mapping):
                self._contracts = normalise_schema(schema)
            else:
                raise TypeError(
                    "`schema` must be a mapping of column -> ColumnContract / dtype, "
                    f"or a dataframely.Schema subclass; got {type(schema).__name__!r}."
                )
        if backend == "dataframely" and self._contracts and self._dy_schema is None:
            self._dy_schema = contracts_to_dataframely(self._contracts)
            self._contracts = {
                col: ColumnContract(
                    dtype=None,
                    nullable=True,
                    allow_nan=spec.allow_nan,
                    unique=spec.unique,
                    allowed=spec.allowed,
                    ge=spec.ge,
                    le=spec.le,
                    required=spec.required,
                    impact=spec.impact,
                )
                for col, spec in self._contracts.items()
            }
        if rules is not None and not isinstance(rules, Mapping):
            raise TypeError("`rules` must be a mapping of rule name -> polars.Expr.")
        self.entity = entity
        self.time = time
        self.strict = bool(strict)
        self.rules: dict[str, pl.Expr] = dict(rules or {})
        self.frequency = frequency
        self.min_obs = min_obs
        self.min_coverage = None if min_coverage is None else float(min_coverage)
        self.impact: dict[str, Impact] = impacts
        self.backend: Backend = backend
        self.near_duplicates = near_duplicates
        self.max_examples = max_examples

    def __repr__(self) -> str:
        parts = [f"entity={self.entity!r}", f"time={self.time!r}"]
        if self._contracts:
            parts.append(f"schema={sorted(self._contracts)!r}")
        if self._dy_schema is not None:
            parts.append(f"dataframely={getattr(self._dy_schema, '__name__', '?')}")
        for name in ("frequency", "min_obs", "min_coverage"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value!r}")
        return f"PanelValidator({', '.join(parts)})"

    # ------------------------------------------------------------------ #
    def _panel_checks(
        self, df: pl.DataFrame, entity: str, time: str
    ) -> list[CheckResult]:
        imp, lim = self.impact, self.max_examples
        keys = check_key_columns(df, entity, time)
        out = [keys]
        if not keys.passed:
            return out
        if self.frequency is not None:
            _normalise_frequency(self.frequency, df.schema[time])
        if imp["null_keys"] != "off":
            out.append(
                check_null_keys(df, entity, time, impact=imp["null_keys"], limit=lim)
            )
        if imp["unique_keys"] != "off":
            out.append(
                check_unique_keys(
                    df, entity, time, impact=imp["unique_keys"], limit=lim
                )
            )
        if imp["monotone_time"] != "off":
            out.append(
                check_monotone_time(
                    df, entity, time, impact=imp["monotone_time"], limit=lim
                )
            )
        if imp["gaps"] != "off":
            out.append(
                check_gaps(
                    df,
                    entity,
                    time,
                    frequency=self.frequency,
                    impact=imp["gaps"],
                    limit=lim,
                )
            )
        if self.min_obs is not None and imp["min_obs"] != "off":
            out.append(
                check_min_obs(
                    df,
                    entity,
                    time,
                    min_obs=self.min_obs,
                    impact=imp["min_obs"],
                    limit=lim,
                )
            )
        if self.min_coverage is not None and imp["coverage"] != "off":
            out.append(
                check_coverage(
                    df,
                    entity,
                    time,
                    min_coverage=self.min_coverage,
                    frequency=self.frequency,
                    impact=imp["coverage"],
                    limit=lim,
                )
            )
        return out

    def _leakage_checks(
        self,
        df: pl.DataFrame,
        entity: str,
        time: str,
        *,
        split: Any,
        fitted: Any,
    ) -> list[CheckResult]:
        out: list[CheckResult] = []
        if split is None:
            if fitted is not None:
                raise ValueError(
                    "`fitted` needs a `split`: the fitted-state check compares a "
                    "transform with a refit on the train rows, and the split is "
                    "what says which rows those are."
                )
            return out
        if fitted is not None and self.impact["fitted_state"] != "off":
            from panelary.quality._leakage import normalise_splits

            splits = normalise_splits(split)
            if isinstance(splits, str) or len(splits) != 1:
                raise ValueError(
                    "`fitted` needs a single (train, test) split or IndexSplit, not "
                    "a fold-label column or several folds: one fitted transform "
                    "has one train fold."
                )
            in_train, _ = split_masks(df, splits[0], entity, time)
            train = df.filter(in_train)
            items: list[tuple[str, Any]]
            if isinstance(fitted, Mapping):
                items = [(str(k), v) for k, v in fitted.items()]
            elif isinstance(fitted, Sequence) and not isinstance(fitted, (str, bytes)):
                items = [(type(v).__name__, v) for v in fitted]
            else:
                items = [(type(fitted).__name__, fitted)]
            for name, obj in items:
                out.append(
                    fitted_state_check(
                        obj,
                        train,
                        entity=entity,
                        time=time,
                        name=name,
                        impact=self.impact["fitted_state"],
                        limit=self.max_examples,
                    )
                )
        nd = self.near_duplicates
        if nd is not False and self.impact["near_duplicate_straddle"] != "off":
            options = dict(nd) if isinstance(nd, Mapping) else {}
            clusters = options.pop("clusters", None)
            out.append(
                straddle_check(
                    df,
                    split,
                    entity=entity,
                    time=time,
                    clusters=clusters,
                    options=options,
                    impact=self.impact["near_duplicate_straddle"],
                    limit=self.max_examples,
                )
            )
        return out

    def _run(
        self,
        data: Any,
        *,
        split: Any,
        fitted: Any,
        producer: str,
        emit_warnings: bool,
    ) -> ValidationReport:
        df, entity, time = resolve_keys(data, self.entity, self.time)
        checks = self._panel_checks(df, entity, time)
        keys_ok = checks[0].passed
        key_cols = [entity, time] if keys_ok else []
        if self._contracts or self.rules or (self.strict and self._contracts):
            checks.extend(
                run_schema(
                    df,
                    self._contracts,
                    keys=key_cols,
                    strict=self.strict,
                    rules=self.rules,
                    impacts=self.impact,
                    limit=self.max_examples,
                )
            )
        if self._dy_schema is not None and self.impact["dataframely"] != "off":
            checks.append(
                run_dataframely(
                    df,
                    self._dy_schema,
                    keys=key_cols,
                    impact=self.impact["dataframely"],
                    limit=self.max_examples,
                )
            )
        if keys_ok:
            checks.extend(
                self._leakage_checks(df, entity, time, split=split, fitted=fitted)
            )
        elif split is not None or fitted is not None:
            raise ValueError(
                "leak-safety checks need usable panel keys; fix the key_columns "
                f"failure first: {checks[0].message}"
            )
        stamped = tuple(c.with_produced_by(producer) for c in checks)
        valid, invalid = split_rows(df, stamped)
        n_entities: int | None
        n_times: int | None
        if keys_ok:
            n_entities, n_times = panel_counts(df, entity, time)
        else:
            n_entities = n_times = None
        report = ValidationReport(
            checks=stamped,
            entity=entity,
            time=time,
            n_rows=df.height,
            n_entities=n_entities,
            n_times=n_times,
            produced_by=producer,
            valid=valid,
            invalid=invalid,
        )
        if emit_warnings:
            for c in report.warnings:
                warnings.warn(
                    f"[{c.id}] {c.message}", _warning_category(c), stacklevel=3
                )
        return report

    # ------------------------------------------------------------------ #
    def validate(
        self,
        data: Any,
        *,
        split: Any = None,
        fitted: Any = None,
        raise_on_fail: bool = True,
        emit_warnings: bool = True,
    ) -> ValidationReport:
        """Run every configured check on ``data``.

        Parameters
        ----------
        data : polars.DataFrame | polars.LazyFrame | PanelFrame
            The panel. A LazyFrame is collected once.
        split : optional
            Enables the leak-safety checks. A ``(train, test)`` pair, an
            :class:`~panelary.validation.IndexSplit`, a fold-label column name,
            or a sequence of splits (see
            :func:`~panelary.quality.check_near_duplicate_straddle`).
        fitted : transform, sequence or mapping of name -> transform, optional
            Fitted transforms whose state must derive only from the train side
            of ``split`` (see :func:`~panelary.quality.check_fitted_state`).
            Needs a single split.
        raise_on_fail : bool, default=True
            Raise :class:`~panelary.quality.PanelValidationError` if any
            ``"fail"`` check failed. The report is on the exception.
        emit_warnings : bool, default=True
            Emit one warning per failed ``"warn"`` check.

        Returns
        -------
        ValidationReport

        Raises
        ------
        PanelValidationError
            If ``raise_on_fail`` and a ``"fail"`` check failed.
        """
        report = self._run(
            data,
            split=split,
            fitted=fitted,
            producer=produced_by("PanelValidator"),
            emit_warnings=emit_warnings,
        )
        if raise_on_fail:
            report.raise_for_status()
        return report

    def filter(
        self,
        data: Any,
        *,
        split: Any = None,
        fitted: Any = None,
        emit_warnings: bool = True,
    ) -> tuple[pl.DataFrame, ValidationReport]:
        """Split ``data`` into valid rows and a report (``dataframely.filter``-style).

        Row-level ``"fail"`` findings do not raise here -- dropping those rows
        is the point -- but a *frame-level* ``"fail"`` finding (unusable keys,
        a missing column, a wrong dtype, a leaky fitted transform) does,
        because no subset of rows can repair it.

        Returns
        -------
        (valid, report) : (polars.DataFrame, ValidationReport)
            ``valid`` is ``report.valid``; ``report.invalid`` holds the
            rejected rows with their ``__failed_checks__``.

        Raises
        ------
        PanelValidationError
            If a frame-level check with ``impact="fail"`` failed.
        """
        report = self._run(
            data,
            split=split,
            fitted=fitted,
            producer=produced_by("PanelValidator"),
            emit_warnings=emit_warnings,
        )
        frame_level = [c for c in report.failures if c.row_mask is None]
        if frame_level:
            from panelary.quality._common import PanelValidationError

            lines = "\n".join(f"  - [{c.id}] {c.message}" for c in frame_level)
            raise PanelValidationError(
                "cannot split rows: frame-level check(s) failed with "
                f'impact="fail":\n{lines}',
                report,
            )
        return report.valid, report


def validate_panel(
    data: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    schema: Mapping[str, Any] | Any = None,
    strict: bool = False,
    rules: Mapping[str, pl.Expr] | None = None,
    frequency: Frequency | None = None,
    min_obs: int | None = None,
    min_coverage: float | None = None,
    impact: Mapping[str, Impact] | None = None,
    backend: Backend = "polars",
    split: Any = None,
    fitted: Any = None,
    near_duplicates: bool | Mapping[str, Any] = True,
    max_examples: int = 5,
    raise_on_fail: bool = True,
    emit_warnings: bool = True,
) -> ValidationReport:
    """Validate a panel in one call: ``PanelValidator(...).validate(data, ...)``.

    Every parameter is that of :class:`PanelValidator` or of
    :meth:`PanelValidator.validate`; the report's ``produced_by`` names this
    function.

    Returns
    -------
    ValidationReport

    Raises
    ------
    PanelValidationError
        If ``raise_on_fail`` (the default) and a ``"fail"`` check failed.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.quality import validate_panel
    >>> df = pl.DataFrame({"id": ["a", "a", "b"], "t": [1, 2, 1], "x": [1.0, 2.0, 3.0]})
    >>> validate_panel(df, entity="id", time="t").status
    'pass'
    """
    validator = PanelValidator(
        entity=entity,
        time=time,
        schema=schema,
        strict=strict,
        rules=rules,
        frequency=frequency,
        min_obs=min_obs,
        min_coverage=min_coverage,
        impact=impact,
        backend=backend,
        near_duplicates=near_duplicates,
        max_examples=max_examples,
    )
    report = validator._run(
        data,
        split=split,
        fitted=fitted,
        producer=produced_by("validate_panel"),
        emit_warnings=emit_warnings,
    )
    if raise_on_fail:
        report.raise_for_status()
    return report
