"""Column contracts: dtype, nullability, uniqueness, domain and custom rules.

The pure-Polars default is :class:`ColumnContract` plus :func:`run_schema`.
The optional backend delegates to `dataframely
<https://github.com/Quantco/dataframely>`_ (BSD-3, the ``schema`` extra):
pass a ``dataframely.Schema`` subclass as the validator's ``schema``, or keep a
mapping of contracts and set ``backend="dataframely"`` to have the dtype and
nullability part translated into one. Either way panel and leak-safety checks
stay pure Polars -- they are what no schema library offers.

A rule is a boolean :class:`polars.Expr` evaluated per row; ``True`` is valid.
A null result counts as valid (the SQL ``CHECK`` convention), so a rule never
double-reports what a nullability contract already covers.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from panelary.quality._common import (
    EVIDENCE_SCHEMA,
    CheckResult,
    Impact,
    check_impact,
    jsonable,
    key_records,
)

__all__ = ["ColumnContract", "run_dataframely", "run_schema"]

#: Loose dtype families accepted as ``ColumnContract(dtype="numeric")`` etc.
_FAMILIES: dict[str, str] = {
    "numeric": "is_numeric",
    "integer": "is_integer",
    "float": "is_float",
    "temporal": "is_temporal",
    "decimal": "is_decimal",
}


@dataclass(frozen=True)
class ColumnContract:
    """What one column must satisfy. Every field is optional.

    Parameters
    ----------
    dtype : polars dtype, dtype class, or str, optional
        Required dtype. A class (``pl.Datetime``) matches any parametrisation;
        an instance (``pl.Datetime("us", "UTC")``) must match exactly. The
        strings ``"numeric"``, ``"integer"``, ``"float"``, ``"temporal"``,
        ``"decimal"``, ``"string"`` and ``"boolean"`` name loose families.
    nullable : bool, default=True
        If False, a null in this column fails the ``nullability`` check.
    allow_nan : bool, default=True
        If False, a floating NaN fails the ``nan`` check (float columns only).
    unique : bool, default=False
        If True, a repeated non-null value fails the ``unique`` check.
    allowed : sequence, optional
        The permitted non-null values (``allowed_values`` check).
    ge, le : optional
        Inclusive lower / upper bounds on non-null values (``range`` check).
    required : bool, default=True
        If False, a missing column is not an error (its checks are skipped).
    impact : {"fail", "warn", "off"}, optional
        Overrides the validator's impact for this column's checks.
    """

    dtype: Any = None
    nullable: bool = True
    allow_nan: bool = True
    unique: bool = False
    allowed: Sequence[Any] | None = None
    ge: Any = None
    le: Any = None
    required: bool = True
    impact: Impact | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of the contract (used as ``expected`` evidence)."""
        return {
            "dtype": _dtype_label(self.dtype),
            "nullable": self.nullable,
            "allow_nan": self.allow_nan,
            "unique": self.unique,
            "allowed": None if self.allowed is None else jsonable(list(self.allowed)),
            "ge": jsonable(self.ge),
            "le": jsonable(self.le),
            "required": self.required,
        }


def _dtype_label(dtype: Any) -> str | None:
    if dtype is None:
        return None
    return str(dtype)


def normalise_schema(schema: Mapping[str, Any]) -> dict[str, ColumnContract]:
    """Coerce ``{column: ColumnContract | dtype | family}`` to contracts."""
    out: dict[str, ColumnContract] = {}
    for col, spec in schema.items():
        if not isinstance(col, str):
            raise TypeError(f"schema keys must be column names, got {col!r}.")
        if isinstance(spec, ColumnContract):
            contract = spec
        else:
            contract = ColumnContract(dtype=spec)
        if contract.impact is not None:
            check_impact(contract.impact, name=f"schema[{col!r}]")
        if contract.dtype is not None:
            _dtype_matches(pl.Int64(), contract.dtype)  # validates the spec
        out[col] = contract
    return out


def _dtype_matches(observed: pl.DataType, expected: Any) -> bool:
    """Does ``observed`` satisfy the ``expected`` dtype spec?"""
    if isinstance(expected, str):
        key = expected.lower()
        if key in _FAMILIES:
            return bool(getattr(observed, _FAMILIES[key])())
        if key in ("string", "str", "utf8"):
            return observed == pl.String
        if key in ("boolean", "bool"):
            return observed == pl.Boolean
        raise ValueError(
            f"unknown dtype family {expected!r}; use a polars dtype or one of "
            f"{sorted([*_FAMILIES, 'string', 'boolean'])}."
        )
    if isinstance(expected, pl.DataType) or (
        isinstance(expected, type) and issubclass(expected, pl.DataType)
    ):
        return bool(observed == expected)
    raise TypeError(
        f"a dtype contract must be a polars dtype or a family name, got {expected!r}."
    )


def _impact(contract: ColumnContract, default: Impact) -> Impact:
    return contract.impact if contract.impact is not None else default


def _row_check(
    df: pl.DataFrame,
    *,
    name: str,
    target: str,
    bad: pl.Expr,
    impact: Impact,
    keys: Sequence[str],
    limit: int,
    ok_message: str,
    bad_message: str,
    observed_key: str,
    expected: Any,
    show_value: bool = True,
) -> CheckResult:
    """Shared body of every row-level column check."""
    mask = df.select(bad.fill_null(False).alias("m")).to_series()
    n = int(mask.sum())
    offenders = df.with_row_index("row").filter(mask)
    extra: tuple[str, ...] = ("row",)
    if show_value and target in df.columns and target not in keys:
        extra = ("row", target)
    return CheckResult(
        name=name,
        category="schema",
        impact=impact,
        passed=n == 0,
        message=ok_message if n == 0 else bad_message.format(n=n),
        n_failing=n,
        observed={observed_key: n},
        expected=expected,
        offending=tuple(key_records(offenders, list(keys), extra=extra, limit=limit)),
        target=target,
        evidence_kind=EVIDENCE_SCHEMA,
        row_mask=mask,
    )


def run_schema(
    df: pl.DataFrame,
    contracts: Mapping[str, ColumnContract],
    *,
    keys: Sequence[str],
    strict: bool,
    rules: Mapping[str, pl.Expr] | None,
    impacts: Mapping[str, Impact],
    limit: int,
) -> list[CheckResult]:
    """Evaluate column contracts and rules on ``df`` (pure Polars).

    Parameters
    ----------
    df : polars.DataFrame
    contracts : mapping of column -> ColumnContract
    keys : sequence of str
        Columns identifying a row in the offending-key samples (the panel keys
        when present, else the row number alone).
    strict : bool
        If True, columns not named in ``contracts`` (or ``keys``) fail
        ``extra_columns``.
    rules : mapping of name -> polars.Expr, optional
        Row-level boolean rules (``True`` = valid, null = valid).
    impacts : mapping of check name -> impact
        Resolved impact per check name.
    limit : int
        Offending-key sample size.

    Returns
    -------
    list of CheckResult
    """
    present = set(df.columns)
    usable_keys = [k for k in keys if k in present]
    out: list[CheckResult] = []

    if impacts["columns_present"] != "off":
        missing = [
            c for c, spec in contracts.items() if spec.required and c not in present
        ]
        out.append(
            CheckResult(
                name="columns_present",
                category="schema",
                impact=impacts["columns_present"],
                passed=not missing,
                message=(
                    "every required column is present."
                    if not missing
                    else f"{len(missing)} required column(s) missing: {missing}."
                ),
                n_failing=len(missing),
                unit="columns",
                observed={"missing": missing},
                expected={"missing": []},
                offending=tuple({"keys": {"column": c}} for c in missing),
            )
        )
    if strict and impacts["extra_columns"] != "off":
        allowed = set(contracts) | set(keys)
        extra_cols = [c for c in df.columns if c not in allowed]
        out.append(
            CheckResult(
                name="extra_columns",
                category="schema",
                impact=impacts["extra_columns"],
                passed=not extra_cols,
                message=(
                    "no columns outside the contract."
                    if not extra_cols
                    else f"{len(extra_cols)} column(s) not in the strict contract: "
                    f"{extra_cols}."
                ),
                n_failing=len(extra_cols),
                unit="columns",
                observed={"extra": extra_cols},
                expected={"extra": []},
                offending=tuple({"keys": {"column": c}} for c in extra_cols),
            )
        )

    schema = df.schema
    for col, spec in contracts.items():
        if col not in present:
            continue
        observed_dtype = schema[col]
        if spec.dtype is not None:
            impact = _impact(spec, impacts["dtype"])
            if impact != "off":
                ok = _dtype_matches(observed_dtype, spec.dtype)
                out.append(
                    CheckResult(
                        name="dtype",
                        category="schema",
                        impact=impact,
                        passed=ok,
                        message=(
                            f"{col!r} is {observed_dtype}."
                            if ok
                            else f"{col!r} is {observed_dtype}, expected "
                            f"{_dtype_label(spec.dtype)}."
                        ),
                        n_failing=0 if ok else 1,
                        unit="columns",
                        observed={"dtype": str(observed_dtype)},
                        expected={"dtype": _dtype_label(spec.dtype)},
                        offending=() if ok else ({"keys": {"column": col}},),
                        target=col,
                    )
                )
        if not spec.nullable:
            impact = _impact(spec, impacts["nullability"])
            if impact != "off":
                out.append(
                    _row_check(
                        df,
                        name="nullability",
                        target=col,
                        bad=pl.col(col).is_null(),
                        impact=impact,
                        keys=usable_keys,
                        limit=limit,
                        ok_message=f"{col!r} has no nulls.",
                        bad_message=f"{{n}} null value(s) in non-nullable {col!r}.",
                        observed_key="nulls",
                        expected={"nulls": 0},
                        show_value=False,
                    )
                )
        if not spec.allow_nan and observed_dtype.is_float():
            impact = _impact(spec, impacts["nan"])
            if impact != "off":
                out.append(
                    _row_check(
                        df,
                        name="nan",
                        target=col,
                        bad=pl.col(col).is_nan(),
                        impact=impact,
                        keys=usable_keys,
                        limit=limit,
                        ok_message=f"{col!r} has no NaN.",
                        bad_message=f"{{n}} NaN value(s) in {col!r}, which forbids NaN.",
                        observed_key="nans",
                        expected={"nans": 0},
                        show_value=False,
                    )
                )
        if spec.unique:
            impact = _impact(spec, impacts["unique"])
            if impact != "off":
                out.append(
                    _row_check(
                        df,
                        name="unique",
                        target=col,
                        bad=pl.col(col).is_duplicated() & pl.col(col).is_not_null(),
                        impact=impact,
                        keys=usable_keys,
                        limit=limit,
                        ok_message=f"{col!r} values are unique.",
                        bad_message=f"{{n}} row(s) share a repeated {col!r} value.",
                        observed_key="rows_with_repeats",
                        expected={"rows_with_repeats": 0},
                    )
                )
        if spec.allowed is not None:
            impact = _impact(spec, impacts["allowed_values"])
            if impact != "off":
                allowed_s = pl.Series(values=list(spec.allowed))
                with contextlib.suppress(Exception):  # else keep the given dtype
                    allowed_s = allowed_s.cast(observed_dtype)
                out.append(
                    _row_check(
                        df,
                        name="allowed_values",
                        target=col,
                        bad=pl.col(col).is_not_null()
                        & ~pl.col(col).is_in(allowed_s.implode()),
                        impact=impact,
                        keys=usable_keys,
                        limit=limit,
                        ok_message=f"every {col!r} value is allowed.",
                        bad_message=f"{{n}} {col!r} value(s) outside the allowed set.",
                        observed_key="disallowed",
                        expected={"allowed": jsonable(list(spec.allowed))},
                    )
                )
        if spec.ge is not None or spec.le is not None:
            impact = _impact(spec, impacts["range"])
            if impact != "off":
                bad = pl.lit(False)
                if spec.ge is not None:
                    bad = bad | (pl.col(col) < spec.ge)
                if spec.le is not None:
                    bad = bad | (pl.col(col) > spec.le)
                if observed_dtype.is_float():
                    bad = bad & ~pl.col(col).is_nan()
                bounds = f"[{spec.ge!r}, {spec.le!r}]"
                out.append(
                    _row_check(
                        df,
                        name="range",
                        target=col,
                        bad=bad,
                        impact=impact,
                        keys=usable_keys,
                        limit=limit,
                        ok_message=f"every {col!r} value lies in {bounds}.",
                        bad_message=f"{{n}} {col!r} value(s) outside {bounds}.",
                        observed_key="out_of_range",
                        expected={"ge": jsonable(spec.ge), "le": jsonable(spec.le)},
                    )
                )

    for rule_name, expr in (rules or {}).items():
        impact = impacts["rule"]
        if impact == "off":
            continue
        if not isinstance(expr, pl.Expr):
            raise TypeError(
                f"rule {rule_name!r} must be a boolean polars expression, got "
                f"{type(expr).__name__!r}."
            )
        out.append(
            _row_check(
                df,
                name="rule",
                target=rule_name,
                bad=~expr,
                impact=impact,
                keys=usable_keys,
                limit=limit,
                ok_message=f"rule {rule_name!r} holds on every row.",
                bad_message=f"rule {rule_name!r} is violated on {{n}} row(s).",
                observed_key="violations",
                expected={"violations": 0},
                show_value=False,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Optional dataframely backend
# --------------------------------------------------------------------------- #
#: Column carrying the row number through a dataframely ``filter`` call.
_ROW_ID = "panelary_row_id"


def _require_dataframely() -> Any:
    from panelary._internal._deps import require

    return require(
        "dataframely", extra="schema", feature="the dataframely schema backend"
    )


def is_dataframely_schema(obj: Any) -> bool:
    """True if ``obj`` is a ``dataframely.Schema`` subclass (without importing it
    unless dataframely is already loaded)."""
    if not isinstance(obj, type):
        return False
    return any(
        base.__module__.split(".", 1)[0] == "dataframely" and base.__name__ == "Schema"
        for base in obj.__mro__
    )


def contracts_to_dataframely(contracts: Mapping[str, ColumnContract]) -> Any:
    """Translate the dtype + nullability part of ``contracts`` into a
    ``dataframely.Schema`` subclass.

    Only exact dtypes translate (``pl.Int64`` -> ``dy.Int64``, ``pl.Boolean``
    -> ``dy.Bool``, ``pl.Datetime("us", "UTC")`` ->
    ``dy.Datetime(time_unit="us", time_zone="UTC")``); a column whose contract
    is a family or absent becomes ``dy.Any``. The remaining constraints
    (``unique``, ``allowed``, ``ge`` / ``le``, NaN) are always evaluated by the
    pure-Polars path, whichever backend is chosen.
    """
    dy = _require_dataframely()
    attrs: dict[str, Any] = {}
    for col, spec in contracts.items():
        attrs[col] = _dy_column(dy, spec)
    return type("PanelaryContract", (dy.Schema,), attrs)


def _dy_column(dy: Any, spec: ColumnContract) -> Any:
    dtype = spec.dtype
    nullable = spec.nullable
    if dtype is None or isinstance(dtype, str):
        return dy.Any()
    inst = dtype() if isinstance(dtype, type) else dtype
    name = type(inst).__name__
    if name == "Boolean":
        name = "Bool"
    if name == "Datetime" and not isinstance(dtype, type):
        return dy.Datetime(
            nullable=nullable, time_unit=inst.time_unit, time_zone=inst.time_zone
        )
    ctor = getattr(dy, name, None)
    if ctor is None:
        return dy.Any()
    return ctor(nullable=nullable)


def run_dataframely(
    df: pl.DataFrame,
    schema: Any,
    *,
    keys: Sequence[str],
    impact: Impact,
    limit: int,
) -> CheckResult:
    """Run a ``dataframely.Schema`` subclass over ``df`` and report one check.

    The schema's ``filter`` does the work. Its per-rule failure counts land in
    ``observed["rule_counts"]``; the rows it rejects are flagged in the row
    mask (via a row-id column threaded through the call), so they join the
    report's ``invalid`` split. A schema-level error (a missing column, a dtype
    that cannot be validated) becomes a frame-level failure with the library's
    own message.
    """
    _require_dataframely()
    usable_keys = [k for k in keys if k in df.columns]
    name = getattr(schema, "__name__", "Schema")
    tracked: Any = type(f"{name}WithRowId", (schema,), {_ROW_ID: _dy_row_id()})
    frame = df.with_row_index(_ROW_ID).with_columns(pl.col(_ROW_ID).cast(pl.Int64))
    try:
        good, failure = tracked.filter(frame, cast=False)
    except Exception as exc:
        return CheckResult(
            name="dataframely",
            category="schema",
            impact=impact,
            passed=False,
            message=f"dataframely schema {name!r} could not validate the frame: {exc}",
            n_failing=1,
            unit="columns",
            observed={"error": type(exc).__name__},
            expected={"schema": name},
            target=name,
        )
    counts = {str(k): int(v) for k, v in dict(failure.counts()).items() if int(v)}
    good_ids = good.get_column(_ROW_ID)
    mask = ~frame.get_column(_ROW_ID).is_in(good_ids.implode())
    n = int(mask.sum())
    offenders = frame.filter(mask).rename({_ROW_ID: "row"})
    return CheckResult(
        name="dataframely",
        category="schema",
        impact=impact,
        passed=n == 0,
        message=(
            f"every row satisfies dataframely schema {name!r}."
            if n == 0
            else f"{n} row(s) fail dataframely schema {name!r} (rule counts: "
            f"{dict(sorted(counts.items()))})."
        ),
        n_failing=n,
        observed={"rule_counts": dict(sorted(counts.items())), "rows": n},
        expected={"rule_counts": {}},
        offending=tuple(
            key_records(offenders, usable_keys, extra=("row",), limit=limit)
        ),
        target=name,
        row_mask=mask.alias("m"),
    )


def _dy_row_id() -> Any:
    dy = _require_dataframely()
    return dy.Int64(nullable=False)
