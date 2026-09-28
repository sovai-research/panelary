"""A lightweight, panel-aware data-quality report.

:func:`quality_report` profiles a frame in a handful of polars passes and
returns a :class:`QualityReport`: per-column null / NaN rates, distinct counts,
constant columns (globally and within every entity), exact duplicate rows and
duplicate ``(entity, time)`` keys, and dtype drift against a reference frame
or schema. It *describes*; it does not reject. Anything worth a second look is
also recorded as a warn-level :class:`~panelary.quality._common.CheckResult`
finding, so it serialises and becomes evidence the same way validation does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from panelary.quality._common import (
    EVIDENCE_SCHEMA,
    CheckResult,
    canonical_json,
    jsonable,
    key_records,
    produced_by,
    ratio,
    to_dataframe,
)

__all__ = ["QualityReport", "quality_report"]


@dataclass(frozen=True, eq=False)
class QualityReport:
    """What :func:`quality_report` measured.

    Attributes
    ----------
    n_rows, n_columns : int
    entity, time : str or None
        Panel keys, when given.
    n_entities, n_times : int or None
    columns : tuple of dict
        One JSON-safe profile per column: ``column``, ``dtype``, ``n_null``,
        ``null_pct``, ``n_nan``, ``nan_pct``, ``n_unique``, ``constant``,
        ``constant_within_entity``, ``max_entity_null_pct``, ``min``, ``max``.
    duplicates : dict
        ``n_duplicate_rows`` / ``duplicate_row_pct`` (rows repeating an earlier
        row exactly) and, for a panel, ``n_duplicate_keys`` /
        ``duplicate_key_pct``.
    entities_per_time : dict or None
        ``min`` / ``max`` / ``mean`` entities observed per time (panel only).
    dtype_drift : dict or None
        ``changed`` / ``added`` / ``removed`` columns against ``reference``.
    findings : tuple of CheckResult
        Warn-level findings (duplicates, constant columns, null rates over
        ``max_null_pct``, dtype drift).
    produced_by : str
    """

    n_rows: int
    n_columns: int
    entity: str | None
    time: str | None
    n_entities: int | None
    n_times: int | None
    columns: tuple[dict[str, Any], ...]
    duplicates: dict[str, Any]
    entities_per_time: dict[str, Any] | None
    dtype_drift: dict[str, Any] | None
    findings: tuple[CheckResult, ...]
    produced_by: str

    @property
    def constant_columns(self) -> list[str]:
        """Columns with at most one distinct non-null value (all-null included)."""
        return [c["column"] for c in self.columns if c["constant"]]

    def column(self, name: str) -> dict[str, Any]:
        """The profile of column ``name``."""
        for c in self.columns:
            if c["column"] == name:
                return c
        raise KeyError(f"no column {name!r} in this report.")

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe plain dict."""
        return {
            "schema": "panelary.QualityReport/1",
            "produced_by": self.produced_by,
            "frame": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns,
                "entity": self.entity,
                "time": self.time,
                "n_entities": self.n_entities,
                "n_times": self.n_times,
            },
            "duplicates": jsonable(self.duplicates),
            "entities_per_time": jsonable(self.entities_per_time),
            "columns": [jsonable(c) for c in self.columns],
            "constant_columns": self.constant_columns,
            "dtype_drift": jsonable(self.dtype_drift),
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Byte-deterministic JSON of :meth:`to_dict`."""
        return canonical_json(self.to_dict(), indent=indent)

    def to_evidence(self) -> list[dict[str, Any]]:
        """One evidence record per finding (see ``CheckResult.to_evidence``)."""
        return [f.to_evidence() for f in self.findings]

    def to_frame(self) -> pl.DataFrame:
        """The per-column profile as a polars DataFrame."""
        fields = (
            ("column", pl.String),
            ("dtype", pl.String),
            ("n_null", pl.Int64),
            ("null_pct", pl.Float64),
            ("n_nan", pl.Int64),
            ("nan_pct", pl.Float64),
            ("n_unique", pl.Int64),
            ("constant", pl.Boolean),
            ("constant_within_entity", pl.Boolean),
            ("max_entity_null_pct", pl.Float64),
        )
        return pl.DataFrame(
            {name: [c[name] for c in self.columns] for name, _ in fields},
            schema=dict(fields),
        )

    def summary(self) -> str:
        """A short human-readable summary."""
        dup = self.duplicates
        lines = [
            f"QualityReport: {self.n_rows} rows x {self.n_columns} columns"
            + (
                f", {self.n_entities} entities x {self.n_times} times"
                if self.entity is not None
                else ""
            ),
            f"  duplicate rows: {dup['n_duplicate_rows']} "
            f"({dup['duplicate_row_pct']:.2%})",
        ]
        if "n_duplicate_keys" in dup:
            lines.append(
                f"  duplicate keys: {dup['n_duplicate_keys']} "
                f"({dup['duplicate_key_pct']:.2%})"
            )
        lines.append(f"  constant columns: {self.constant_columns}")
        worst = sorted(self.columns, key=lambda c: -c["null_pct"])[:5]
        lines.append(
            "  highest null %: "
            + ", ".join(f"{c['column']}={c['null_pct']:.1%}" for c in worst)
        )
        if self.dtype_drift is not None:
            lines.append(f"  dtype drift: {self.dtype_drift}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:  # pragma: no cover - notebook cosmetics
        html = self.to_frame()._repr_html_()
        return f"<pre>{self.summary()}</pre>{html}"


def _reference_schema(reference: Any) -> dict[str, pl.DataType]:
    from panelary.core.panel_frame import PanelFrame

    if isinstance(reference, pl.LazyFrame):
        return dict(reference.collect_schema())
    if isinstance(reference, pl.DataFrame):
        return dict(reference.schema)
    if isinstance(reference, PanelFrame):
        return dict(reference.schema)
    if isinstance(reference, Mapping):
        out: dict[str, pl.DataType] = {}
        for k, v in reference.items():
            out[str(k)] = v() if isinstance(v, type) else v
        return out
    raise TypeError(
        "`reference` must be a polars frame, a PanelFrame or a mapping of "
        f"column -> dtype, got {type(reference).__name__!r}."
    )


def _minmax_ok(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() or dtype.is_temporal() or dtype == pl.String


def quality_report(
    data: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    reference: Any = None,
    columns: Sequence[str] | None = None,
    max_null_pct: float | None = None,
    max_examples: int = 5,
) -> QualityReport:
    """Profile ``data``: null %, duplicate %, constant columns, dtype drift.

    Parameters
    ----------
    data : polars.DataFrame | polars.LazyFrame | PanelFrame
        The frame to profile. A LazyFrame is collected once.
    entity, time : str, optional
        Panel keys. With both, the report adds duplicate ``(entity, time)``
        keys, entities per time, per-column constancy *within* every entity
        (a static attribute, e.g. a sector code) and the worst per-entity null
        rate. A PanelFrame supplies its own keys. Without keys the report is
        frame-level only (nothing is inferred).
    reference : frame, PanelFrame or mapping of column -> dtype, optional
        What the schema *should* be -- typically the training frame or the
        previous delivery. Columns whose dtype changed, appeared or vanished
        are reported as ``dtype_drift``.
    columns : sequence of str, optional
        Restrict the per-column profile to these columns (default: all).
    max_null_pct : float, optional
        Record a ``null_rate`` finding for every column whose null fraction
        exceeds this (``0.2`` means 20%).
    max_examples : int, default=5
        Size of each finding's offending-key sample.

    Returns
    -------
    QualityReport
    """
    from panelary.core.panel_frame import PanelFrame

    if isinstance(data, PanelFrame):
        entity = data.entity_col if entity is None else entity
        time = data.time_col if time is None else time
    df = to_dataframe(data)
    if (entity is None) != (time is None):
        raise ValueError("pass both `entity` and `time`, or neither.")
    for key in (entity, time):
        if key is not None and key not in df.columns:
            raise ValueError(f"key column {key!r} not in the frame: {df.columns}.")
    if max_null_pct is not None and not 0.0 <= max_null_pct <= 1.0:
        raise ValueError(f"`max_null_pct` must lie in [0, 1], got {max_null_pct!r}.")
    cols = list(df.columns) if columns is None else list(columns)
    unknown = [c for c in cols if c not in df.columns]
    if unknown:
        raise ValueError(f"`columns` names columns not in the frame: {unknown}.")

    producer = produced_by("quality_report")
    n = df.height
    schema = df.schema
    keys: list[str] = []
    if entity is not None and time is not None:
        keys = [entity, time]
    panel = bool(keys)
    ent, tim = (keys[0], keys[1]) if panel else ("", "")

    # ---- per-column profile, one pass --------------------------------- #
    aggs: list[pl.Expr] = []
    for c in cols:
        dt = schema[c]
        aggs.append(pl.col(c).null_count().cast(pl.Int64).alias(f"{c}\x00null"))
        if dt.is_float():
            aggs.append(pl.col(c).is_nan().sum().cast(pl.Int64).alias(f"{c}\x00nan"))
        if dt != pl.Object:
            aggs.append(pl.col(c).n_unique().cast(pl.Int64).alias(f"{c}\x00uniq"))
            aggs.append(
                pl.col(c).drop_nulls().n_unique().cast(pl.Int64).alias(f"{c}\x00nnuniq")
            )
        if _minmax_ok(dt):
            aggs.append(pl.col(c).min().alias(f"{c}\x00min"))
            aggs.append(pl.col(c).max().alias(f"{c}\x00max"))
    stats = df.select(aggs).row(0, named=True) if aggs and n else {}

    per_entity: dict[str, Any] = {}
    if panel and n:
        value_cols = [c for c in cols if c not in keys and schema[c] != pl.Object]
        if value_cols:
            grouped = df.group_by(ent).agg(
                *[
                    pl.col(c).drop_nulls().n_unique().cast(pl.Int64).alias(f"{c}\x00u")
                    for c in value_cols
                ],
                *[
                    (pl.col(c).null_count() / pl.len()).alias(f"{c}\x00np")
                    for c in value_cols
                ],
            )
            per_entity = grouped.select(
                *[pl.col(f"{c}\x00u").max().alias(f"{c}\x00u") for c in value_cols],
                *[pl.col(f"{c}\x00np").max().alias(f"{c}\x00np") for c in value_cols],
            ).row(0, named=True)

    profiles: list[dict[str, Any]] = []
    for c in cols:
        dt = schema[c]
        n_null = int(stats.get(f"{c}\x00null", 0) or 0)
        n_nan = int(stats.get(f"{c}\x00nan", 0) or 0)
        nn_unique = stats.get(f"{c}\x00nnuniq")
        n_unique = stats.get(f"{c}\x00uniq")
        within = per_entity.get(f"{c}\x00u")
        profiles.append(
            {
                "column": c,
                "dtype": str(dt),
                "n_null": n_null,
                "null_pct": ratio(n_null, n),
                "n_nan": n_nan,
                "nan_pct": ratio(n_nan, n),
                "n_unique": None if n_unique is None else int(n_unique),
                "constant": bool(nn_unique is not None and nn_unique <= 1) or n == 0,
                "constant_within_entity": (
                    None
                    if not panel or c in keys or within is None
                    else bool(within <= 1)
                ),
                "max_entity_null_pct": (
                    None
                    if f"{c}\x00np" not in per_entity
                    else ratio(per_entity[f"{c}\x00np"] or 0.0, 1.0)
                ),
                "min": jsonable(stats.get(f"{c}\x00min")),
                "max": jsonable(stats.get(f"{c}\x00max")),
            }
        )

    # ---- duplicates ---------------------------------------------------- #
    findings: list[CheckResult] = []
    try:
        dup_mask = df.is_duplicated()
        repeat_mask = ~df.select(pl.struct(pl.all()).is_first_distinct()).to_series()
        n_dup_rows = int(repeat_mask.sum())
    except Exception:  # pragma: no cover - unhashable (Object) columns
        dup_mask = None
        repeat_mask = None
        n_dup_rows = 0
    duplicates: dict[str, Any] = {
        "n_duplicate_rows": n_dup_rows,
        "duplicate_row_pct": ratio(n_dup_rows, n),
    }
    if repeat_mask is not None and n_dup_rows:
        rows = df.with_row_index("row").filter(repeat_mask)
        findings.append(
            CheckResult(
                name="duplicate_rows",
                category="quality",
                impact="warn",
                passed=False,
                message=(
                    f"{n_dup_rows} row(s) ({ratio(n_dup_rows, n):.2%}) exactly "
                    "repeat an earlier row."
                ),
                n_failing=n_dup_rows,
                observed={"n_duplicate_rows": n_dup_rows},
                expected={"n_duplicate_rows": 0},
                offending=tuple(
                    key_records(rows, keys, extra=("row",), limit=max_examples)
                ),
                evidence_kind=EVIDENCE_SCHEMA,
                produced_by=producer,
                row_mask=dup_mask,
            )
        )

    entities_per_time: dict[str, Any] | None = None
    n_entities = n_times = None
    if panel:
        key_repeat = ~df.select(pl.struct(ent, tim).is_first_distinct()).to_series()
        n_dup_keys = int(key_repeat.sum())
        duplicates["n_duplicate_keys"] = n_dup_keys
        duplicates["duplicate_key_pct"] = ratio(n_dup_keys, n)
        n_entities = int(df.get_column(ent).drop_nulls().n_unique())
        n_times = int(df.get_column(tim).drop_nulls().n_unique())
        if n:
            per_time = (
                df.select(ent, tim)
                .drop_nulls()
                .group_by(tim)
                .agg(pl.col(ent).n_unique().alias("k"))
                .get_column("k")
            )
            if per_time.len():
                lo: Any = per_time.min()
                hi: Any = per_time.max()
                entities_per_time = {
                    "min": int(lo),
                    "max": int(hi),
                    "mean": ratio(float(per_time.sum()), float(per_time.len())),
                }
        if n_dup_keys:
            dup_keys = (
                df.filter(df.select(pl.struct(ent, tim).is_duplicated()).to_series())
                .group_by(ent, tim)
                .agg(pl.len().cast(pl.Int64).alias("count"))
            )
            findings.append(
                CheckResult(
                    name="duplicate_keys",
                    category="quality",
                    impact="warn",
                    passed=False,
                    message=(
                        f"{n_dup_keys} row(s) repeat an earlier ({ent}, {tim}) key."
                    ),
                    n_failing=n_dup_keys,
                    observed={"n_duplicate_keys": n_dup_keys},
                    expected={"n_duplicate_keys": 0},
                    offending=tuple(
                        key_records(
                            dup_keys, keys, extra=("count",), limit=max_examples
                        )
                    ),
                    produced_by=producer,
                )
            )

    for p in profiles:
        if p["column"] in keys:
            continue
        if p["constant"] and n:
            findings.append(
                CheckResult(
                    name="constant_column",
                    category="quality",
                    impact="warn",
                    passed=False,
                    message=(
                        f"{p['column']!r} has at most one distinct non-null value; "
                        "it carries no information."
                    ),
                    n_failing=1,
                    unit="columns",
                    observed={
                        "n_unique_non_null": int(
                            stats.get(f"{p['column']}\x00nnuniq") or 0
                        )
                    },
                    expected={"min_n_unique_non_null": 2},
                    target=p["column"],
                    produced_by=producer,
                )
            )
        if max_null_pct is not None and p["null_pct"] > max_null_pct:
            findings.append(
                CheckResult(
                    name="null_rate",
                    category="quality",
                    impact="warn",
                    passed=False,
                    message=(
                        f"{p['column']!r} is {p['null_pct']:.1%} null, above the "
                        f"{max_null_pct:.1%} threshold."
                    ),
                    n_failing=p["n_null"],
                    observed={"null_pct": p["null_pct"]},
                    expected={"max_null_pct": float(max_null_pct)},
                    target=p["column"],
                    produced_by=producer,
                )
            )

    # ---- dtype drift --------------------------------------------------- #
    dtype_drift: dict[str, Any] | None = None
    if reference is not None:
        ref = _reference_schema(reference)
        changed = [
            {"column": c, "reference": str(ref[c]), "current": str(schema[c])}
            for c in df.columns
            if c in ref and ref[c] != schema[c]
        ]
        added = [
            {"column": c, "current": str(schema[c])} for c in df.columns if c not in ref
        ]
        removed = [
            {"column": c, "reference": str(ref[c])} for c in ref if c not in schema
        ]
        dtype_drift = {"changed": changed, "added": added, "removed": removed}
        for item in changed:
            findings.append(
                CheckResult(
                    name="dtype_drift",
                    category="quality",
                    impact="warn",
                    passed=False,
                    message=(
                        f"{item['column']!r} is {item['current']}, was "
                        f"{item['reference']} in the reference."
                    ),
                    n_failing=1,
                    unit="columns",
                    observed={"dtype": item["current"]},
                    expected={"dtype": item["reference"]},
                    target=item["column"],
                    produced_by=producer,
                )
            )
        for item in added:
            findings.append(
                CheckResult(
                    name="dtype_drift",
                    category="quality",
                    impact="warn",
                    passed=False,
                    message=f"{item['column']!r} is new (not in the reference).",
                    n_failing=1,
                    unit="columns",
                    observed={"dtype": item["current"]},
                    expected={"dtype": None},
                    target=item["column"],
                    produced_by=producer,
                )
            )
        for item in removed:
            findings.append(
                CheckResult(
                    name="dtype_drift",
                    category="quality",
                    impact="warn",
                    passed=False,
                    message=f"{item['column']!r} is missing (present in the reference).",
                    n_failing=1,
                    unit="columns",
                    observed={"dtype": None},
                    expected={"dtype": item["reference"]},
                    target=item["column"],
                    produced_by=producer,
                )
            )

    return QualityReport(
        n_rows=n,
        n_columns=df.width,
        entity=entity,
        time=time,
        n_entities=n_entities,
        n_times=n_times,
        columns=tuple(profiles),
        duplicates=duplicates,
        entities_per_time=entities_per_time,
        dtype_drift=dtype_drift,
        findings=tuple(findings),
        produced_by=producer,
    )
