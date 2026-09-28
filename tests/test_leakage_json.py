"""JSON serialisation of leakage findings, and the one-call feature-set sweep.

The assessment engine consumes a leakage audit as report *evidence*: it stores
``kind`` / ``locator`` / ``observed`` / ``expected`` / ``produced_by`` and
content-addresses the payload. That only works if the serialised form is

* **JSON-safe** -- ``json.loads(to_json()) == to_dict()``, nothing exotic;
* **byte-deterministic** -- the same audit is the same bytes on every run, in
  every process (no memory addresses, no timestamps, no ids);
* **self-describing** -- a ``schema`` tag and a ``produced_by`` tag naming what
  to re-run.

These tests pin all three for :class:`~panelary.leakage.CompileResult`,
:class:`~panelary.leakage.Finding`, :class:`~panelary.leakage.FeatureSetAudit`
and :class:`~panelary.core.pipeline.PipelineAudit`, and exercise
:func:`~panelary.leakage._compile.audit_features`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import polars as pl
import pytest

import panelary
from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline, PipelineAudit
from panelary.core.protocol import PanelTransformer
from panelary.leakage._compile import audit, audit_features
from panelary.leakage._types import (
    POLARS_TREE_FORMAT_TESTED,
    Classification,
    CompileResult,
    FeatureSetAudit,
    Finding,
    LeakageRefused,
    Verdict,
    canonical_json,
    render_expr,
)

_POLARS_MINOR = ".".join(pl.__version__.split(".")[:2])

pytestmark = pytest.mark.skipif(
    _POLARS_MINOR not in POLARS_TREE_FORMAT_TESTED,
    reason=(
        f"polars {pl.__version__}: the serialised expression format is not a "
        "stable public API and the compiler reads it directly; it has only been "
        f"checked against {', '.join(POLARS_TREE_FORMAT_TESTED)}."
    ),
)

#: Polars pickles the callable inside ``map_batches`` / ``map_elements`` with
#: ``cloudpickle`` when it serialises the expression tree. The bare-core install
#: (numpy + polars only) does not have it, so these cases cannot run there.
requires_cloudpickle = pytest.mark.skipif(
    importlib.util.find_spec("cloudpickle") is None,
    reason="serialising a Python UDF needs cloudpickle (absent in the bare-core install)",
)
from panelary.testing import assert_no_lookahead

VERSION = panelary.__version__

LAG = pl.col("x").shift(1).over("e", order_by="t")
LEAD = pl.col("x").shift(-1).over("e", order_by="t")
BFILL = pl.col("x").fill_null(strategy="backward").over("e")


def _feature_set() -> dict[str, pl.Expr]:
    return {"lag": LAG, "lead": LEAD, "bfill": BFILL}


def _panel(n_entities: int = 3, n_time: int = 12, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_entities * n_time)
    x[::5] = np.nan
    return pl.DataFrame(
        {
            "e": np.repeat([f"E{i}" for i in range(n_entities)], n_time),
            "t": np.tile(np.arange(n_time, dtype=np.int64), n_entities),
            "x": x,
        }
    ).with_columns(pl.col("x").fill_nan(None))


# --------------------------------------------------------------------------- #
# The shared convention
# --------------------------------------------------------------------------- #
def test_canonical_json_is_the_documented_dumps() -> None:
    payload = {"b": [1, "two", None], "a": {"z": True, "y": 1.5}}
    assert canonical_json(payload) == json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    assert canonical_json(payload, indent=2) == json.dumps(
        payload, sort_keys=True, indent=2
    )


def test_canonical_json_refuses_nan() -> None:
    """``NaN`` is not JSON; a report must say ``null`` instead of emitting it."""
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_render_expr_masks_memory_addresses() -> None:
    class _Weird:
        def __str__(self) -> str:
            return "thing at 0x10BB45C50"

    assert render_expr(_Weird()) == "thing at 0x?"
    assert render_expr(None) is None
    assert "0x" not in render_expr(pl.col("x").map_batches(lambda s: s))


# --------------------------------------------------------------------------- #
# Finding / CompileResult
# --------------------------------------------------------------------------- #
def test_finding_to_dict_shape_and_locator() -> None:
    f = Finding(
        kind="Function.Shift",
        classification=Classification.REFUSE,
        reason="negative shift",
        path=("input", 0, "function"),
    )
    d = f.to_dict()
    assert d == {
        "kind": "Function.Shift",
        "classification": "refuse",
        "reason": "negative shift",
        "path": ["input", 0, "function"],
        "locator": "input.0.function",
        "rewrote_to": None,
    }
    assert json.loads(f.to_json()) == d
    assert Finding("Over", Classification.REWRITE, "r").locator == "<root>"


def test_compile_result_refused_payload() -> None:
    res = audit(LEAD, time="t", entity="e")
    d = res.to_dict()
    assert d["schema"] == "panelary.CompileResult/1"
    assert d["produced_by"] == f"panelary.leakage.audit@{VERSION}"
    assert d["verdict"] == "refused"
    assert d["expr"] is None  # nothing compiled
    assert d["source"] == render_expr(LEAD)  # but we know what was audited
    assert d["context"] == {
        "time": "t",
        "entity": "e",
        "allow_approximate": False,
        "trust": [],
    }
    assert d["counts"]["refuse"] >= 1
    kinds = [f["kind"] for f in d["findings"]]
    assert "Function.Shift" in kinds
    assert json.loads(res.to_json()) == d


def test_compile_result_rewritten_payload_carries_both_forms() -> None:
    res = audit(BFILL, time="t", entity="e")
    d = res.to_dict()
    assert d["verdict"] == "rewritten"
    assert d["source"] != d["expr"]  # observed vs. the compiled replacement
    rewrites = [f for f in d["findings"] if f["classification"] == "rewrite"]
    assert rewrites and all(f["rewrote_to"] for f in rewrites)


def test_safe_result_has_no_findings_and_echoes_source() -> None:
    d = audit(LAG, time="t", entity="e").to_dict()
    assert d["verdict"] == "safe"
    assert d["findings"] == []
    assert d["counts"] == {"safe": 0, "rewrite": 0, "refuse": 0}
    assert d["source"] == d["expr"]


def test_hand_built_result_serialises_without_source_or_context() -> None:
    res = CompileResult(verdict=Verdict.SAFE)
    d = res.to_dict()
    assert d["source"] is None and d["context"] is None and d["expr"] is None
    json.loads(res.to_json())


def test_new_fields_do_not_change_equality() -> None:
    """``source`` / ``context`` are metadata: two results for the same decision
    still compare equal when their expressions are absent."""
    a = CompileResult(verdict=Verdict.REFUSED, source="a")
    b = CompileResult(verdict=Verdict.REFUSED, source="b")
    assert a == b


@requires_cloudpickle
def test_trust_is_recorded_in_the_context() -> None:
    """A trust-based acceptance must be visible in the evidence, not silent."""
    expr = pl.col("x").panel.frac_diff(0.4).over("e", order_by="t")
    res = audit(expr, time="t", entity="e", trust="panel.frac_diff")
    d = res.to_dict()
    assert d["context"]["trust"], "declared trust missing from serialised context"
    assert any("declared trust" in f["reason"] for f in d["findings"])


def test_to_json_is_byte_identical_within_a_process() -> None:
    a = audit_features(_feature_set(), time="t", entity="e").to_json()
    b = audit_features(_feature_set(), time="t", entity="e").to_json()
    assert a == b
    # compact, key-sorted form: re-dumping the parsed payload reproduces it
    assert a == json.dumps(json.loads(a), sort_keys=True, separators=(",", ":"))


_SUBPROCESS = textwrap.dedent(
    """
    import polars as pl
    from panelary.leakage._compile import audit_features
    fs = {
        "lag": pl.col("x").shift(1).over("e", order_by="t"),
        "lead": pl.col("x").shift(-1).over("e", order_by="t"),
        "bfill": pl.col("x").fill_null(strategy="backward").over("e"),
        "udf": pl.col("x").map_batches(lambda s: s),
    }
    print(audit_features(fs, time="t", entity="e").to_json(), end="")
    """
)


@requires_cloudpickle
def test_to_json_is_byte_identical_across_processes() -> None:
    """Different interpreters, different hash seeds, different heap addresses:
    the evidence payload must not move by a byte."""
    outs = []
    for seed in ("1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        )
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
    assert "0x" not in outs[0]
    json.loads(outs[0])


# --------------------------------------------------------------------------- #
# audit_features -- a whole feature set in one call
# --------------------------------------------------------------------------- #
def test_audit_features_aggregates_in_input_order() -> None:
    report = audit_features(_feature_set(), time="t", entity="e")
    assert isinstance(report, FeatureSetAudit)
    assert report.names == ("lag", "lead", "bfill")
    assert report.verdict is Verdict.REFUSED
    assert not report.ok
    assert report.refused == ("lead",)
    assert report.rewritten == ("bfill",)
    assert report["lag"].verdict is Verdict.SAFE
    assert len(report) == 3
    with pytest.raises(KeyError):
        report["missing"]


def test_audit_features_matches_per_expression_audit() -> None:
    """The sweep is exactly ``audit`` per feature -- no second rule set."""
    report = audit_features(_feature_set(), time="t", entity="e")
    for name, expr in _feature_set().items():
        single = audit(expr, time="t", entity="e")
        assert report[name].verdict is single.verdict
        assert report[name].findings == single.findings
        assert report[name].to_json() == single.to_json()


def test_audit_features_payload() -> None:
    d = audit_features(_feature_set(), time="t", entity="e").to_dict()
    assert d["schema"] == "panelary.FeatureSetAudit/1"
    assert d["produced_by"] == f"panelary.leakage.audit_features@{VERSION}"
    assert d["verdict"] == "refused" and d["ok"] is False
    assert d["counts"] == {"safe": 1, "rewritten": 1, "refused": 1}
    assert d["refused"] == ["lead"] and d["rewritten"] == ["bfill"]
    assert [f["name"] for f in d["features"]] == ["lag", "lead", "bfill"]
    for f in d["features"]:
        assert f["result"]["schema"] == "panelary.CompileResult/1"


def test_audit_features_accepts_named_expressions() -> None:
    report = audit_features([LAG.alias("lag"), BFILL.alias("bf")], time="t", entity="e")
    assert report.names == ("lag", "bf")
    assert report.verdict is Verdict.REWRITTEN and report.ok


def test_audit_features_empty_set_is_safe() -> None:
    report = audit_features({}, time="t", entity="e")
    assert report.verdict is Verdict.SAFE and report.ok
    assert report.to_dict()["features"] == []


@pytest.mark.parametrize(
    ("features", "err", "match"),
    [
        (LAG, TypeError, "feature \\*set\\*"),
        ([LAG, LAG], ValueError, "appears twice"),
        ([pl.all().shift(1)], ValueError, "no output name"),
        ({"a": "not an expr"}, TypeError, "not a polars.Expr"),
        ({1: LAG}, TypeError, "names must be str"),
        (["nope"], TypeError, "not a polars.Expr"),
    ],
)
def test_audit_features_rejects_malformed_sets(
    features: Any, err: type[Exception], match: str
) -> None:
    with pytest.raises(err, match=match):
        audit_features(features, time="t", entity="e")


def test_audit_features_validates_trust_once() -> None:
    with pytest.raises(ValueError):
        audit_features({"lag": LAG}, time="t", trust="no.such_operator")


def test_exprs_are_applicable_and_point_in_time() -> None:
    """The compiled set is usable as-is, and the rewrite really is causal."""
    report = audit_features({"lag": LAG, "bfill": BFILL}, time="t", entity="e")
    exprs = report.exprs()
    assert list(exprs) == ["lag", "bfill"]
    df = _panel()
    out = df.with_columns(*exprs.values())
    assert {"lag", "bfill"} <= set(out.columns)
    for expr in exprs.values():
        assert_no_lookahead(expr, df, entity="e", time="t")


def test_exprs_raise_with_every_refusal_named() -> None:
    report = audit_features(_feature_set(), time="t", entity="e")
    with pytest.raises(LeakageRefused, match="lead") as info:
        report.exprs()
    paths = [f.path for f in info.value.result.findings]
    assert paths and all(p[0] == "lead" for p in paths)


# --------------------------------------------------------------------------- #
# PipelineAudit
# --------------------------------------------------------------------------- #
class _Passthrough(PanelTransformer):
    panel_safe = True
    leakage_safe = True

    def _fit(self, panel: PanelFrame) -> None:
        return None

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel


class _Leaky(_Passthrough):
    leakage_safe = False


class _ExprStep(PanelTransformer):
    panel_safe = True
    leakage_safe = False

    def __init__(self, exprs: dict[str, pl.Expr]) -> None:
        super().__init__()
        self._exprs = dict(exprs)

    def leakage_exprs(self) -> dict[str, pl.Expr]:
        return dict(self._exprs)

    def with_leakage_exprs(self, exprs: dict[str, pl.Expr]) -> _ExprStep:
        return type(self)(exprs)

    def _fit(self, panel: PanelFrame) -> None:
        return None

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel.with_columns(*[e.alias(k) for k, e in self._exprs.items()])


def _pipeline() -> Pipeline:
    inner = Pipeline([("pass", _Passthrough()), ("leaky", _Leaky())])
    return Pipeline(
        [
            ("feats", _ExprStep({"lag": LAG, "bfill": BFILL})),
            ("inner", inner),
            ("pass", _Passthrough()),
        ],
        entity="e",
        time="t",
    )


def test_pipeline_audit_to_dict() -> None:
    report = _pipeline().audit()
    assert isinstance(report, PipelineAudit)
    d = report.to_dict()
    assert d["schema"] == "panelary.PipelineAudit/1"
    assert d["produced_by"] == f"panelary.core.pipeline.Pipeline.audit@{VERSION}"
    assert d["verdict"] == "refused" and d["ok"] is False
    assert d["refused"] == ["inner"]
    assert d["rewritten"] == ["feats"]
    assert d["counts"] == {"safe": 1, "rewritten": 1, "refused": 1}

    feats, inner, passthrough = d["steps"]
    assert feats["audited_by"] == "expressions"
    assert [r["name"] for r in feats["results"]] == ["lag", "bfill"]
    assert feats["results"][1]["result"]["verdict"] == "rewritten"
    assert inner["audited_by"] == "pipeline"
    assert inner["nested"]["schema"] == "panelary.PipelineAudit/1"
    assert inner["nested"]["refused"] == ["leaky"]
    assert passthrough["audited_by"] == "declaration"
    assert passthrough["results"] == [] and passthrough["nested"] is None

    assert json.loads(report.to_json()) == d


def test_pipeline_audit_to_json_is_deterministic() -> None:
    a = _pipeline().audit().to_json()
    b = _pipeline().audit().to_json()
    assert a == b
    assert "0x" not in a


def test_step_audit_records_expression_names() -> None:
    step = _pipeline().audit().steps[0]
    assert step.expr_names == ("lag", "bfill")
    assert len(step.results) == 2


def test_pipeline_audit_behaviour_unchanged() -> None:
    """Serialisation is additive: the verdicts are what they always were."""
    report = _pipeline().audit()
    assert report.verdict == "refused"
    assert [s.name for s in report.refused] == ["inner"]
    assert [s.name for s in report.rewritten] == ["feats"]


# --------------------------------------------------------------------------- #
# The documented mapping onto report evidence
# --------------------------------------------------------------------------- #
def _to_evidence(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The recipe in docs/api-reference/leakage.md, verbatim in spirit.

    One evidence record per non-safe feature; ``kind`` is left to the consumer
    (``STATIC_ANALYSIS`` in the engine).
    """
    records = []
    for feature in report["features"]:
        result = feature["result"]
        if result["verdict"] == "safe":
            continue
        for f in result["findings"]:
            if f["classification"] == "safe":
                continue
            records.append(
                {
                    "locator": f"feature:{feature['name']}#{f['locator']}",
                    "observed": {
                        "expr": result["source"],
                        "node": f["kind"],
                        "classification": f["classification"],
                        "reason": f["reason"],
                    },
                    "expected": {
                        "expr": result["expr"],
                        "rewrote_to": f["rewrote_to"],
                    },
                    "produced_by": result["produced_by"],
                }
            )
    return records


def test_documented_evidence_mapping_is_json_and_complete() -> None:
    report = audit_features(_feature_set(), time="t", entity="e").to_dict()
    records = _to_evidence(report)
    features = {r["locator"].split("#")[0] for r in records}
    assert features == {"feature:lead", "feature:bfill"}
    for r in records:
        assert r["produced_by"].startswith("panelary.leakage.audit@")
        # the engine hashes observed/expected with exactly this dumps
        json.dumps(r["observed"], sort_keys=True, separators=(",", ":"))
        json.dumps(r["expected"], sort_keys=True, separators=(",", ":"))
    lead = [r for r in records if r["locator"].startswith("feature:lead")]
    assert all(r["expected"]["expr"] is None for r in lead)  # refused: no fix
