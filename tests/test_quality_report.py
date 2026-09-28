"""quality_report(), the shared JSON / evidence convention, and the Tier-1
``.panel.validate()`` / ``.panel.quality_report()`` namespace methods."""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl
import pytest

import panelary
import panelary.namespaces  # noqa: F401  (registers the .panel namespace)
from panelary.core.panel_frame import PanelFrame
from panelary.quality import (
    ColumnContract,
    PanelValidationError,
    QualityReport,
    ValidationReport,
    check_near_duplicate_straddle,
    quality_report,
    validate_panel,
)

#: The engine's EvidenceKind values these results may carry.
_EVIDENCE_KINDS = {"schema-validation", "static-analysis", "counterfactual-run"}
_PRODUCED_BY = re.compile(
    r"^panelary\.quality\.[A-Za-z_.]+@" + re.escape(panelary.__version__) + "$"
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b"],
            "t": [1, 2, 1, 1, 2],
            "x": [1.0, None, float("nan"), float("nan"), 2.0],
            "sector": ["s1", "s1", "s2", "s2", "s2"],
            "k": [7, 7, 7, 7, 7],
        }
    )


# --------------------------------------------------------------------------- #
# quality_report
# --------------------------------------------------------------------------- #
def test_column_profile():
    report = quality_report(_frame(), entity="id", time="t")
    x = report.column("x")
    assert (x["n_null"], x["null_pct"], x["n_nan"], x["nan_pct"]) == (1, 0.2, 2, 0.4)
    assert x["constant"] is False and x["constant_within_entity"] is False
    assert x["max_entity_null_pct"] == 0.5
    sector = report.column("sector")
    assert sector["constant"] is False and sector["constant_within_entity"] is True
    assert report.column("k")["constant"] is True
    assert report.constant_columns == ["k"]
    assert report.column("id")["constant_within_entity"] is None  # keys excluded
    assert (report.column("t")["min"], report.column("t")["max"]) == (1, 2)


def test_duplicates_and_entities_per_time():
    report = quality_report(_frame(), entity="id", time="t")
    assert report.duplicates == {
        "n_duplicate_rows": 1,
        "duplicate_row_pct": 0.2,
        "n_duplicate_keys": 1,
        "duplicate_key_pct": 0.2,
    }
    assert report.entities_per_time == {"min": 2, "max": 2, "mean": 2.0}
    assert (report.n_entities, report.n_times) == (2, 2)
    names = [f.id for f in report.findings]
    assert names[:2] == ["duplicate_rows", "duplicate_keys"]
    assert "constant_column:k" in names


def test_frame_level_only_without_keys():
    report = quality_report(_frame().lazy())
    assert report.entity is None and report.n_entities is None
    assert "n_duplicate_keys" not in report.duplicates
    assert report.column("x")["constant_within_entity"] is None


def test_panelframe_supplies_keys():
    report = quality_report(PanelFrame(_frame(), entity="id", time="t"))
    assert (report.entity, report.time) == ("id", "t")


def test_dtype_drift_against_frame_and_mapping():
    current = _frame()
    reference = (
        current.with_columns(pl.col("t").cast(pl.Int32))
        .drop("k")
        .with_columns(pl.lit(1).alias("gone"))
    )
    report = quality_report(current, reference=reference)
    assert report.dtype_drift == {
        "changed": [{"column": "t", "reference": "Int32", "current": "Int64"}],
        "added": [{"column": "k", "current": "Int64"}],
        "removed": [{"column": "gone", "reference": "Int32"}],
    }
    by_mapping = quality_report(current, reference=dict(current.schema))
    assert by_mapping.dtype_drift == {"changed": [], "added": [], "removed": []}
    drift = [f for f in report.findings if f.name == "dtype_drift"]
    assert [f.target for f in drift] == ["t", "k", "gone"]
    assert drift[0].observed == {"dtype": "Int64"} and drift[0].expected == {
        "dtype": "Int32"
    }


def test_null_rate_threshold_and_argument_validation():
    report = quality_report(_frame(), max_null_pct=0.1)
    assert [f.target for f in report.findings if f.name == "null_rate"] == ["x"]
    with pytest.raises(ValueError, match="max_null_pct"):
        quality_report(_frame(), max_null_pct=5)
    with pytest.raises(ValueError, match="both"):
        quality_report(_frame(), entity="id")
    with pytest.raises(ValueError, match="not in the frame"):
        quality_report(_frame(), columns=["nope"])
    with pytest.raises(TypeError, match="reference"):
        quality_report(_frame(), reference=42)


def test_columns_subset_and_to_frame():
    report = quality_report(_frame(), columns=["x", "k"])
    assert [c["column"] for c in report.columns] == ["x", "k"]
    frame = report.to_frame()
    assert frame.columns[:2] == ["column", "dtype"] and frame.height == 2
    assert "QualityReport" in repr(report)


# --------------------------------------------------------------------------- #
# The shared serialisation convention
# --------------------------------------------------------------------------- #
def _failing_validation() -> ValidationReport:
    df = pl.concat([_frame(), _frame().head(1)])
    return validate_panel(
        df,
        entity="id",
        time="t",
        schema={"x": ColumnContract(nullable=False)},
        min_obs=3,
        raise_on_fail=False,
        emit_warnings=False,
    )


def _results():
    df = _frame()
    straddle = check_near_duplicate_straddle(
        df, ([1], [2]), entity="id", time="t", clusters=pl.Series([0, 0, 1, 1, 2])
    )
    return [
        ("ValidationReport", _failing_validation()),
        (
            "QualityReport",
            quality_report(df, entity="id", time="t", reference={"x": pl.Int64}),
        ),
        ("CheckResult", straddle),
    ]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_to_json_is_the_canonical_dump_of_to_dict(index):
    type_name, result = _results()[index]
    payload = result.to_dict()
    assert payload["schema"] == f"panelary.{type_name}/1"
    assert _PRODUCED_BY.match(payload["produced_by"]), payload["produced_by"]
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert result.to_json() == canonical
    assert json.loads(result.to_json(indent=2)) == payload


def test_to_json_is_byte_deterministic_across_runs():
    first = [r.to_json() for _, r in _results()]
    second = [r.to_json() for _, r in _results()]
    assert first == second


def test_json_is_safe_for_nan_and_dates():
    df = pl.DataFrame(
        {
            "id": ["a", "a"],
            "d": pl.Series([0, 1], dtype=pl.Date),
            "x": [float("nan"), float("inf")],
        }
    )
    report = quality_report(df, entity="id", time="d")
    text = report.to_json()
    json.loads(text)
    assert '"1970-01-02"' in text
    assert report.column("x")["max"] in ("inf", "nan")


def test_every_failed_check_maps_onto_an_evidence_record():
    report = _failing_validation()
    assert report.status == "fail"
    records = report.to_evidence()
    assert len(records) == len(report.failures) + len(report.warnings) > 0
    for check, record in zip(
        [c for c in report.checks if not c.passed], records, strict=True
    ):
        assert set(record) == {"kind", "locator", "observed", "expected", "produced_by"}
        assert record["kind"] in _EVIDENCE_KINDS
        assert record["locator"].startswith(f"panelary.quality/{check.name}")
        assert record["produced_by"] == report.produced_by
        json.dumps(record, allow_nan=False)
    unique = report.check("unique_keys")
    assert unique.locator == (
        'panelary.quality/unique_keys#[{"count":2,"keys":{"id":"a","t":1}},'
        '{"count":2,"keys":{"id":"b","t":1}}]'
    )


def test_checks_in_a_report_carry_the_report_producer():
    report = _failing_validation()
    assert {c.produced_by for c in report.checks} == {report.produced_by}
    assert report.produced_by.startswith("panelary.quality.validate_panel@")


def test_summary_counts_match():
    payload = _failing_validation().to_dict()
    summary = payload["summary"]
    assert summary["n_checks"] == len(payload["checks"])
    assert (
        summary["n_checks"]
        == summary["n_passed"] + summary["n_warned"] + summary["n_failed"]
    )
    assert payload["panel"]["n_invalid_rows"] + payload["panel"]["n_valid_rows"] == 6


# --------------------------------------------------------------------------- #
# Tier-1 namespace surface
# --------------------------------------------------------------------------- #
def test_dataframe_and_lazyframe_namespace_methods():
    df = pl.DataFrame(
        {
            "id": np.repeat(["a", "b"], 4),
            "t": np.tile(np.arange(4), 2),
            "x": np.arange(8, dtype=np.float64),
        }
    )
    assert isinstance(df.panel.validate(entity="id", time="t"), ValidationReport)
    assert df.lazy().panel.validate().ok
    report = df.panel.quality_report(entity="id", time="t")
    assert isinstance(report, QualityReport) and report.n_entities == 2
    assert isinstance(df.lazy().panel.quality_report(), QualityReport)
    with pytest.raises(PanelValidationError):
        pl.concat([df, df.head(1)]).panel.validate(
            entity="id", time="t", emit_warnings=False
        )


def test_namespace_module_does_not_import_the_quality_layer_at_module_scope():
    # `import panelary` loads `quality` eagerly (it is ~2 ms, like `leakage`), so
    # a sys.modules probe after `import panelary.namespaces` cannot see the
    # boundary. What the Tier-1 namespace owes is that *its own* module never
    # imports the quality layer at module scope -- only inside the methods.
    import ast
    import inspect

    import panelary.namespaces.panel as ns

    tree = ast.parse(inspect.getsource(ns))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("panelary.quality")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("panelary.quality") for a in node.names)
