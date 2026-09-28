"""Shared result types and plumbing for :mod:`panelary.quality`.

Everything a check produces is a :class:`CheckResult`; everything a validator
produces is a :class:`ValidationReport`. Both follow the package-wide
serialisation convention so a finding can travel, unchanged, into an assessment
report:

* ``to_dict()`` returns a JSON-safe plain ``dict`` (dates as ISO strings,
  non-finite floats as ``"nan"`` / ``"inf"`` / ``"-inf"``, numpy scalars as
  Python scalars);
* ``to_json(indent=None)`` is ``json.dumps(to_dict(), sort_keys=True,
  separators=(",", ":"))`` -- **byte-deterministic**: no timestamps, no set
  ordering, offending-key samples sorted before they are truncated;
* the top level names its ``"schema"`` (``"panelary.<TypeName>/1"``) and what
  ``"produced_by"`` it (``"panelary.quality.<fn>@<version>"``);
* each failed check carries a ``locator`` (check name, target, offending keys)
  plus ``observed`` / ``expected`` -- the three fields an evidence record needs.

Nothing here imports beyond ``numpy`` / ``polars`` and the standard library.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from panelary.core.panel_frame import PanelFrame

__all__ = [
    "CheckResult",
    "Impact",
    "PanelValidationError",
    "QualityWarning",
    "ValidationReport",
]

#: How a failed check is treated. ``"fail"`` makes the report fail (and, for a
#: row-level check, sends the offending rows to ``invalid``); ``"warn"`` records
#: the finding and emits a warning but rejects nothing; ``"off"`` does not run
#: the check at all.
Impact = Literal["fail", "warn", "off"]

_IMPACTS: frozenset[str] = frozenset({"fail", "warn", "off"})

#: Column name carrying the failed-check ids on :attr:`ValidationReport.invalid`.
FAILED_CHECKS_COL = "__failed_checks__"

#: Evidence kinds, spelled exactly as the assessment engine's ``EvidenceKind``.
EVIDENCE_SCHEMA = "schema-validation"
EVIDENCE_STATIC = "static-analysis"
EVIDENCE_COUNTERFACTUAL = "counterfactual-run"


class QualityWarning(UserWarning):
    """A data-quality check with ``impact="warn"`` found a problem.

    Row-order findings are raised as
    :class:`~panelary.core.panel_frame.PanelOrderWarning` and leak-safety
    findings as :class:`~panelary.preprocessing.LeakageWarning` instead, so the
    existing filters for those two keep working; everything else is this.
    """


class PanelValidationError(ValueError):
    """A validation run found at least one failure with ``impact="fail"``.

    Attributes
    ----------
    report : ValidationReport
        The full report, so a caller that catches the error still has every
        finding, the ``valid`` / ``invalid`` split and ``to_json()``.
    """

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------- #
# JSON-safe conversion
# --------------------------------------------------------------------------- #
def jsonable(value: Any) -> Any:
    """Convert ``value`` into a JSON-safe structure, deterministically.

    ``json.dumps(jsonable(x), allow_nan=False)`` never raises for the values a
    check produces: temporal values become ISO-8601 strings, non-finite floats
    the strings ``"nan"`` / ``"inf"`` / ``"-inf"``, numpy / polars containers
    lists, mapping keys strings, and anything unrecognised its ``str``.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (pl.DataType, type)) and _is_polars_dtype(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, pl.Series):
        return [jsonable(v) for v in value.to_list()]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(v) for v in value), key=_sort_token)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def _is_polars_dtype(value: Any) -> bool:
    if isinstance(value, pl.DataType):
        return True
    return isinstance(value, type) and issubclass(value, pl.DataType)


def _sort_token(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """The package's byte-deterministic JSON encoding of ``value``."""
    if indent is None:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, sort_keys=True, indent=indent)


def produced_by(fn: str) -> str:
    """``"panelary.quality.<fn>@<version>"`` -- what to re-run to reproduce."""
    from panelary import __version__

    return f"panelary.quality.{fn}@{__version__}"


def ratio(num: float, den: float) -> float:
    """``num / den`` rounded to 12 places (0.0 when ``den`` is 0).

    Rounding keeps the JSON stable against last-ulp differences between
    platforms; twelve places is far below any threshold a check compares to.
    """
    if not den:
        return 0.0
    return round(float(num) / float(den), 12)


# --------------------------------------------------------------------------- #
# Frame helpers
# --------------------------------------------------------------------------- #
def to_dataframe(data: Any) -> pl.DataFrame:
    """Materialise a DataFrame / LazyFrame / PanelFrame into a DataFrame."""
    from panelary.core.panel_frame import PanelFrame

    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, PanelFrame):
        return data.collect()
    raise TypeError(
        "expected a polars DataFrame, LazyFrame or PanelFrame, got "
        f"{type(data).__name__!r}."
    )


def resolve_keys(
    data: Any, entity: str | None, time: str | None
) -> tuple[pl.DataFrame, str, str]:
    """Return ``(df, entity, time)``, honouring a PanelFrame's own keys.

    Bare frames without explicit keys follow the codebase convention of
    :func:`~panelary.core.panel_frame.as_panel`: column 0 is the entity,
    column 1 the time.
    """
    from panelary.core.panel_frame import PanelFrame

    if isinstance(data, PanelFrame):
        for given, own, what in (
            (entity, data.entity_col, "entity"),
            (time, data.time_col, "time"),
        ):
            if given is not None and given != own:
                raise ValueError(
                    f"`{what}={given!r}` conflicts with the PanelFrame's own "
                    f"{what} column {own!r}. Omit `{what}=` when passing a "
                    "PanelFrame (its keys win), or pass a bare frame instead."
                )
        pf: PanelFrame = data
        return pf.collect(), pf.entity_col, pf.time_col
    df = to_dataframe(data)
    if entity is None or time is None:
        names = df.columns
        if len(names) < 2:
            raise ValueError(
                "cannot infer panel keys: a panel needs at least an entity and "
                f"a time column, but the frame has columns {names}. Pass "
                "`entity=` and `time=` explicitly."
            )
        entity = names[0] if entity is None else entity
        time = names[1] if time is None else time
    return df, entity, time


def key_records(
    frame: pl.DataFrame,
    keys: Sequence[str],
    *,
    extra: Sequence[str] = (),
    limit: int,
) -> list[dict[str, Any]]:
    """Deterministic, truncated sample of offending rows as locator records.

    Rows are sorted by every key column (nulls last) before truncation, so the
    sample -- and therefore the JSON -- does not depend on the input row order.
    Each record is ``{"keys": {<key>: value, ...}, <extra>: value, ...}``.
    """
    if frame.height == 0 or limit <= 0:
        return []
    cols = [*keys, *extra]
    sub = frame.select(cols)
    if keys:
        # An unsortable key dtype keeps frame order (still deterministic).
        with contextlib.suppress(Exception):
            sub = sub.sort(list(keys), nulls_last=True, maintain_order=True)
    out: list[dict[str, Any]] = []
    for row in sub.head(limit).iter_rows(named=True):
        record: dict[str, Any] = {"keys": {k: jsonable(row[k]) for k in keys}}
        for name in extra:
            record[name] = jsonable(row[name])
        out.append(record)
    return out


def check_impact(value: Any, *, name: str) -> Impact:
    """Validate an impact level, with a message naming the check."""
    if value not in _IMPACTS:
        raise ValueError(
            f"impact for {name!r} must be one of 'fail', 'warn' or 'off', got "
            f"{value!r}."
        )
    out: Impact = value
    return out


# --------------------------------------------------------------------------- #
# CheckResult
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class CheckResult:
    """The outcome of one check, in a shape that maps onto an evidence record.

    Attributes
    ----------
    name : str
        Stable check identifier, e.g. ``"unique_keys"``, ``"nullability"``,
        ``"near_duplicate_straddle"``.
    category : {"panel", "schema", "leakage", "quality"}
        Which family the check belongs to.
    impact : {"fail", "warn"}
        How a failure of this check is treated.
    passed : bool
        True if no violation was found.
    message : str
        Human-readable one-paragraph verdict.
    n_failing : int
        Number of offending ``unit`` s (0 when ``passed``).
    unit : str
        What ``n_failing`` counts: ``"rows"``, ``"entities"``, ``"columns"``,
        ``"clusters"`` or ``"transforms"``.
    observed, expected : JSON-safe
        What was measured, and what a passing panel would show.
    offending : tuple of dict
        A deterministic, truncated sample of offending keys (see
        :func:`key_records`), each ``{"keys": {...}, ...}``.
    target : str or None
        The column, rule or transform the check is about, if any.
    evidence_kind : str
        The engine ``EvidenceKind`` this finding maps to
        (``"schema-validation"``, ``"static-analysis"`` or
        ``"counterfactual-run"``).
    produced_by : str
        ``"panelary.quality.<fn>@<version>"``.
    row_mask : polars.Series or None
        Per-row boolean flag of offending rows, for row-level checks. Not
        serialised; it drives the ``valid`` / ``invalid`` split.
    """

    name: str
    category: str
    impact: Impact
    passed: bool
    message: str
    n_failing: int = 0
    unit: str = "rows"
    observed: Any = None
    expected: Any = None
    offending: tuple[dict[str, Any], ...] = ()
    target: str | None = None
    evidence_kind: str = EVIDENCE_SCHEMA
    produced_by: str = ""
    row_mask: pl.Series | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    @property
    def id(self) -> str:
        """``name`` or ``"name:target"`` -- unique within one report."""
        return self.name if self.target is None else f"{self.name}:{self.target}"

    @property
    def status(self) -> Literal["pass", "warn", "fail"]:
        """``"pass"``, or the impact level of the failure."""
        if self.passed:
            return "pass"
        return "warn" if self.impact == "warn" else "fail"

    @property
    def locator(self) -> str:
        """Where to look: ``panelary.quality/<name>[/<target>][#<keys>]``.

        The offending-key sample is appended as canonical JSON, so the locator
        is stable across runs and pinpoints the rows a reader should inspect.
        """
        loc = f"panelary.quality/{self.name}"
        if self.target is not None:
            loc += f"/{self.target}"
        if self.offending:
            loc += "#" + canonical_json(jsonable(list(self.offending)))
        return loc

    def with_produced_by(self, value: str) -> CheckResult:
        """Return a copy stamped with ``produced_by=value``."""
        return CheckResult(
            name=self.name,
            category=self.category,
            impact=self.impact,
            passed=self.passed,
            message=self.message,
            n_failing=self.n_failing,
            unit=self.unit,
            observed=self.observed,
            expected=self.expected,
            offending=self.offending,
            target=self.target,
            evidence_kind=self.evidence_kind,
            produced_by=value,
            row_mask=self.row_mask,
        )

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe plain dict (the row mask is not serialised)."""
        return {
            "schema": "panelary.CheckResult/1",
            "produced_by": self.produced_by,
            "check": self.name,
            "id": self.id,
            "target": self.target,
            "category": self.category,
            "impact": self.impact,
            "status": self.status,
            "passed": self.passed,
            "message": self.message,
            "n_failing": int(self.n_failing),
            "unit": self.unit,
            "observed": jsonable(self.observed),
            "expected": jsonable(self.expected),
            "offending": jsonable(list(self.offending)),
            "locator": self.locator,
            "evidence_kind": self.evidence_kind,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Byte-deterministic JSON of :meth:`to_dict`."""
        return canonical_json(self.to_dict(), indent=indent)

    def to_evidence(self) -> dict[str, Any]:
        """The finding as an evidence record: ``kind``, ``locator``, ``observed``,
        ``expected``, ``produced_by``.

        The keys are exactly the assessment engine's ``Evidence`` fields, so
        ``Evidence(**check.to_evidence())`` builds one; this package does not
        import the engine.
        """
        return {
            "kind": self.evidence_kind,
            "locator": self.locator,
            "observed": jsonable(self.observed),
            "expected": jsonable(self.expected),
            "produced_by": self.produced_by,
        }


# --------------------------------------------------------------------------- #
# ValidationReport
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ValidationReport:
    """Everything one validation run found, plus the valid / invalid row split.

    Attributes
    ----------
    checks : tuple of CheckResult
        Every check that ran, in a fixed order (panel, schema, leakage).
    entity, time : str
        The panel keys the checks were run against.
    n_rows : int
        Rows in the validated frame.
    n_entities, n_times : int or None
        Distinct entities / times (``None`` if the key columns are unusable).
    produced_by : str
        ``"panelary.quality.<fn>@<version>"``.
    valid : polars.DataFrame
        Rows that pass every row-level check with ``impact="fail"``.
    invalid : polars.DataFrame
        The other rows, with a ``__failed_checks__`` list column naming the
        check ids each row failed (``dataframely``-style).
    """

    checks: tuple[CheckResult, ...]
    entity: str
    time: str
    n_rows: int
    n_entities: int | None
    n_times: int | None
    produced_by: str
    valid: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)
    invalid: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)

    # ------------------------------------------------------------------ #
    @property
    def ok(self) -> bool:
        """True if no check with ``impact="fail"`` failed (warnings allowed)."""
        return not self.failures

    @property
    def status(self) -> Literal["pass", "warn", "fail"]:
        """``"fail"`` if any fail-impact check failed, else ``"warn"`` if any
        warn-impact check did, else ``"pass"``."""
        if self.failures:
            return "fail"
        return "warn" if self.warnings else "pass"

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """Failed checks with ``impact="fail"``."""
        return tuple(c for c in self.checks if c.status == "fail")

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        """Failed checks with ``impact="warn"``."""
        return tuple(c for c in self.checks if c.status == "warn")

    def check(self, name: str, target: str | None = None) -> CheckResult:
        """Return the check called ``name`` (and ``target``, if given).

        Raises
        ------
        KeyError
            If no such check ran.
        """
        for c in self.checks:
            if c.name == name and (target is None or c.target == target):
                return c
        raise KeyError(
            f"no check {name!r}"
            + (f" with target {target!r}" if target is not None else "")
            + f" in this report; checks that ran: {[c.id for c in self.checks]}."
        )

    def raise_for_status(self) -> ValidationReport:
        """Raise :class:`PanelValidationError` if :attr:`ok` is False.

        Returns
        -------
        ValidationReport
            ``self``, for chaining, when nothing failed.
        """
        if self.ok:
            return self
        lines = [f"  - [{c.id}] {c.message}" for c in self.failures]
        raise PanelValidationError(
            f"panel validation failed {len(self.failures)} check(s) with "
            'impact="fail":\n'
            + "\n".join(lines)
            + "\nThe full report is on the exception's `.report` attribute.",
            self,
        )

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe plain dict (the row frames are summarised, not embedded)."""
        return {
            "schema": "panelary.ValidationReport/1",
            "produced_by": self.produced_by,
            "status": self.status,
            "ok": self.ok,
            "panel": {
                "entity": self.entity,
                "time": self.time,
                "n_rows": int(self.n_rows),
                "n_entities": self.n_entities,
                "n_times": self.n_times,
                "n_valid_rows": int(self.valid.height),
                "n_invalid_rows": int(self.invalid.height),
            },
            "summary": {
                "n_checks": len(self.checks),
                "n_passed": sum(1 for c in self.checks if c.passed),
                "n_warned": len(self.warnings),
                "n_failed": len(self.failures),
            },
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Byte-deterministic JSON of :meth:`to_dict`."""
        return canonical_json(self.to_dict(), indent=indent)

    def to_evidence(self) -> list[dict[str, Any]]:
        """One evidence record per failed or warned check (see
        :meth:`CheckResult.to_evidence`)."""
        return [c.to_evidence() for c in self.checks if not c.passed]

    def to_frame(self) -> pl.DataFrame:
        """The checks as a tidy polars DataFrame (one row per check)."""
        return pl.DataFrame(
            {
                "check": [c.name for c in self.checks],
                "target": [c.target for c in self.checks],
                "category": [c.category for c in self.checks],
                "impact": [c.impact for c in self.checks],
                "status": [c.status for c in self.checks],
                "n_failing": [int(c.n_failing) for c in self.checks],
                "unit": [c.unit for c in self.checks],
                "message": [c.message for c in self.checks],
            },
            schema={
                "check": pl.String,
                "target": pl.String,
                "category": pl.String,
                "impact": pl.String,
                "status": pl.String,
                "n_failing": pl.Int64,
                "unit": pl.String,
                "message": pl.String,
            },
        )

    def summary(self) -> str:
        """A short multi-line, human-readable summary."""
        head = (
            f"ValidationReport[{self.status}] {self.n_rows} rows, "
            f"{self.n_entities} entities, {self.n_times} times "
            f"({self.entity!r}, {self.time!r}): "
            f"{len(self.checks)} checks, {len(self.failures)} failed, "
            f"{len(self.warnings)} warned"
        )
        lines = [head]
        for c in self.checks:
            if not c.passed:
                lines.append(f"  {c.status.upper():4} {c.id}: {c.message}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def split_rows(
    df: pl.DataFrame, checks: Iterable[CheckResult]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition ``df`` into ``(valid, invalid)`` by row-level fail checks.

    A row is invalid if any failed check with ``impact="fail"`` flags it in its
    ``row_mask``. ``invalid`` carries :data:`FAILED_CHECKS_COL`, the sorted ids
    of every such check the row failed.
    """
    flags: list[pl.Series] = []
    ids: list[str] = []
    for c in checks:
        if c.passed or c.impact != "fail" or c.row_mask is None:
            continue
        flags.append(c.row_mask.fill_null(False).alias(f"__f{len(flags)}__"))
        ids.append(c.id)
    if not flags:
        empty = df.clear().with_columns(
            pl.lit(None, dtype=pl.List(pl.String)).alias(FAILED_CHECKS_COL)
        )
        return df, empty
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    failed = pl.concat_list(
        [
            pl.when(pl.col(flags[i].name)).then(pl.lit(ids[i])).otherwise(None)
            for i in order
        ]
    ).list.drop_nulls()
    tagged = df.with_columns([flags[i] for i in order]).with_columns(
        failed.alias(FAILED_CHECKS_COL)
    )
    is_bad = pl.col(FAILED_CHECKS_COL).list.len() > 0
    valid = tagged.filter(~is_bad).select(df.columns)
    invalid = tagged.filter(is_bad).select([*df.columns, FAILED_CHECKS_COL])
    return valid, invalid


def panel_counts(df: pl.DataFrame, entity: str, time: str) -> tuple[int, int]:
    """``(n_entities, n_times)`` over non-null key values."""
    row = df.select(
        pl.col(entity).drop_nulls().n_unique().alias("e"),
        pl.col(time).drop_nulls().n_unique().alias("t"),
    ).row(0)
    return int(row[0]), int(row[1])


def as_panel_frame(df: pl.DataFrame, entity: str, time: str) -> PanelFrame:
    """Wrap ``df`` in a fresh PanelFrame of *unknown* order (no promise kept)."""
    from panelary.core.panel_frame import PanelFrame

    return PanelFrame(df, entity=entity, time=time, validate=False)
