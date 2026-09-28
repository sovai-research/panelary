"""Tests for the point-in-time compiler's tree walker (``leakage/_compile.py``).

The walker is deliberately decoupled from the rule table: it looks kinds up in
a ``Mapping[str, Rule]`` and knows nothing about which kinds exist. So most of
these tests drive it with the small fixture table below, which exercises every
branch that matters -- safe leaf, rewrite, refusal, context-dependent repair,
sequence payloads -- without depending on ``_rules.py``. The tests marked
``requires_rules`` run the same proofs against the real table.

The central proof is :func:`test_leaky_expression_becomes_causal`: a backward
fill inside a centred rolling mean *fails* ``assert_no_lookahead``, and the
compiled expression *passes* it.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.leakage import _compile
from panelary.leakage._compile import audit, causalize

#: Polars pickles the callable inside ``map_batches`` / ``map_elements`` with
#: ``cloudpickle`` when it serialises the expression tree. The bare-core install
#: (numpy + polars only) does not have it, so these cases cannot run there.
requires_cloudpickle = pytest.mark.skipif(
    importlib.util.find_spec("cloudpickle") is None,
    reason="serialising a Python UDF needs cloudpickle (absent in the bare-core install)",
)
from panelary.leakage._types import (
    Classification,
    Context,
    Finding,
    LeakageRefused,
    Rule,
    Verdict,
)
from panelary.testing import assert_no_lookahead

_HAVE_RULES = importlib.util.find_spec("panelary.leakage._rules") is not None
requires_rules = pytest.mark.skipif(
    not _HAVE_RULES, reason="panelary.leakage._rules is not written yet"
)


# --------------------------------------------------------------------------
# A fixture rule table: the smallest thing that is shaped like the real one.
# --------------------------------------------------------------------------

SORT_OPTIONS: dict[str, Any] = {
    "descending": False,
    "nulls_last": False,
    "multithreaded": True,
    "maintain_order": False,
    "limit": None,
}


def _safe(_payload: Any, _ctx: Context) -> Classification:
    return Classification.SAFE


def _classify_fill_null(payload: Any, _ctx: Context) -> Classification:
    strategy = payload["function"]["FillNullWithStrategy"]
    if isinstance(strategy, dict) and "Backward" in strategy:
        return Classification.REWRITE
    return Classification.SAFE


def _rewrite_fill_null(payload: Any, _ctx: Context) -> Any:
    out = json.loads(json.dumps(payload))
    out["function"]["FillNullWithStrategy"] = {"Forward": None}
    return {"Function": out}


def _classify_rolling(payload: Any, _ctx: Context) -> Classification:
    options = payload["function"]["RollingExpr"]["options"]
    return Classification.REWRITE if options["center"] else Classification.SAFE


def _rewrite_rolling(payload: Any, _ctx: Context) -> Any:
    out = json.loads(json.dumps(payload))
    out["function"]["RollingExpr"]["options"]["center"] = False
    return {"Function": out}


def _classify_over(payload: Any, ctx: Context) -> Classification:
    if payload["order_by"] is not None:
        return Classification.SAFE
    return Classification.REWRITE if ctx.time else Classification.REFUSE


def _rewrite_over(payload: Any, ctx: Context) -> Any:
    out = dict(payload)
    out["order_by"] = [{"Column": ctx.time}, dict(SORT_OPTIONS)]
    return {"Over": out}


def _refuse(_payload: Any, _ctx: Context) -> Classification:
    return Classification.REFUSE


FIXTURE_RULES: dict[str, Rule] = {
    "Column": Rule(kind="Column", classify=_safe, reason="a column reference"),
    "Literal": Rule(kind="Literal", classify=_safe, reason="a constant"),
    "Alias": Rule(
        kind="Alias", classify=_safe, reason="renaming is safe", children=("0",)
    ),
    "Function.Shift": Rule(
        kind="Function.Shift",
        classify=_safe,
        reason="a positive shift looks backwards",
        children=("input",),
    ),
    "Function.FillNullWithStrategy": Rule(
        kind="Function.FillNullWithStrategy",
        classify=_classify_fill_null,
        reason="a backward fill carries future values into the past",
        rewrite=_rewrite_fill_null,
        rewrote_to="fill_null(strategy='forward')",
        children=("input",),
    ),
    "Function.RollingExpr": Rule(
        kind="Function.RollingExpr",
        classify=_classify_rolling,
        reason="a centred window straddles the observation",
        rewrite=_rewrite_rolling,
        rewrote_to="a trailing window",
        children=("input",),
    ),
    "Over": Rule(
        kind="Over",
        classify=_classify_over,
        reason="over(entity) with no order_by is only correct on a sorted frame",
        rewrite=_rewrite_over,
        rewrote_to="over(entity, order_by=time)",
        children=("function", "partition_by"),
    ),
    "BinaryExpr": Rule(
        kind="BinaryExpr",
        classify=_safe,
        reason="arithmetic is elementwise",
        children=("left", "right"),
    ),
    "Agg.Mean": Rule(
        kind="Agg.Mean",
        classify=_refuse,
        reason="a whole-column mean sees the entire sample",
        children=("Mean",),
    ),
}


@pytest.fixture
def fixture_rules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Rule]:
    """Point the walker at :data:`FIXTURE_RULES` instead of the real table."""
    monkeypatch.setattr(_compile, "_rule_table", lambda: FIXTURE_RULES)
    return FIXTURE_RULES


def make_panel() -> PanelFrame:
    """A two-entity panel with gaps, so fills and rolling windows do work."""
    n = 8
    return PanelFrame(
        pl.DataFrame(
            {
                "e": ["a"] * n + ["b"] * n,
                "t": list(range(n)) * 2,
                "x": [1.0, None, 3.0, None, 5.0, 6.0, None, 8.0]
                + [2.0, 4.0, None, 8.0, 1.0, None, 3.0, 5.0],
            }
        ),
        entity="e",
        time="t",
    )


LEAKY = (
    pl.col("x")
    .fill_null(strategy="backward")
    .rolling_mean(3, center=True)
    .over("e")
    .alias("f")
)
SAFE_EXPR = pl.col("x").shift(1).over("e", order_by="t").alias("f")


def _paths(findings: tuple[Finding, ...]) -> set[tuple[Any, ...]]:
    return {(f.kind, f.path) for f in findings}


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_safe_expression_round_trips(fixture_rules: dict[str, Rule]) -> None:
    """A wholly safe expression compiles to one that evaluates identically."""
    result = audit(SAFE_EXPR, time="t", entity="e")

    assert result.verdict is Verdict.SAFE
    assert result.findings == ()
    assert isinstance(result.expr, pl.Expr)

    frame = make_panel().collect()
    before = frame.with_columns(SAFE_EXPR)
    after = frame.with_columns(result.expr)
    assert before.equals(after)


def test_safe_expression_is_a_real_round_trip(fixture_rules: dict[str, Rule]) -> None:
    """The returned expression really went through serialise/deserialise."""
    result = audit(SAFE_EXPR, time="t", entity="e")
    assert result.expr is not SAFE_EXPR
    assert result.expr.meta.serialize(format="json") == SAFE_EXPR.meta.serialize(
        format="json"
    )


# --------------------------------------------------------------------------
# The end-to-end proof
# --------------------------------------------------------------------------


def test_original_leaky_expression_fails_the_verifier() -> None:
    """Invariant 3: the "leaky" form must actually leak, or it is not a leak."""
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(LEAKY, make_panel())


def test_leaky_expression_becomes_causal(fixture_rules: dict[str, Rule]) -> None:
    """Backward fill + centred rolling compiles to something that cannot leak."""
    result = audit(LEAKY, time="t", entity="e")

    assert result.verdict is Verdict.REWRITTEN
    assert result.refused == ()
    assert {f.kind for f in result.rewritten} == {
        "Function.FillNullWithStrategy",
        "Function.RollingExpr",
        "Over",
    }
    # The proof: the compiled expression passes the empirical verifier that the
    # original fails.
    assert_no_lookahead(result.expr, make_panel())


def test_causalize_returns_the_compiled_expression(
    fixture_rules: dict[str, Rule],
) -> None:
    compiled = causalize(LEAKY, time="t", entity="e")
    assert isinstance(compiled, pl.Expr)
    assert_no_lookahead(compiled, make_panel())


def test_rewrite_changes_the_values(fixture_rules: dict[str, Rule]) -> None:
    """A rewrite is not a no-op: the leaky and causal columns differ."""
    frame = make_panel().collect()
    compiled = causalize(LEAKY, time="t", entity="e")
    leaky = frame.with_columns(LEAKY)["f"]
    causal = frame.with_columns(compiled)["f"]
    assert not leaky.equals(causal)


@requires_rules
def test_leaky_expression_becomes_causal_with_real_rules() -> None:
    """The same proof, against the table that ships."""
    result = audit(LEAKY, time="t", entity="e")
    assert result.verdict is Verdict.REWRITTEN, [str(f) for f in result.findings]
    assert_no_lookahead(result.expr, make_panel())


# --------------------------------------------------------------------------
# Fail closed (contract invariant 1)
# --------------------------------------------------------------------------


def test_unknown_kind_is_refused_not_safe(fixture_rules: dict[str, Rule]) -> None:
    """A real expression whose kind has no rule is REFUSED, and names the kind."""
    result = audit(pl.col("x").rank(), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert result.expr is None
    kinds = {f.kind for f in result.refused}
    assert "Function.Rank" in kinds
    assert all(f.classification is not Classification.SAFE for f in result.findings)
    assert "fails closed" in next(
        f.reason for f in result.refused if f.kind == "Function.Rank"
    )


def test_synthetic_unknown_node_is_refused() -> None:
    """A hand-made node of an invented kind refuses, without weakening anything."""
    findings: list[Finding] = []
    node = {"Frobnicate": {"input": [{"Column": "x"}]}}
    out = _compile._walk(node, FIXTURE_RULES, Context(), (), findings)

    assert out == node  # unknown nodes are left untouched
    assert len(findings) == 1
    assert findings[0].kind == "Frobnicate"
    assert findings[0].classification is Classification.REFUSE
    assert findings[0].path == ()


def test_unknown_child_node_is_refused_with_its_path() -> None:
    """An unknown kind nested in a declared child slot refuses, at its path."""
    findings: list[Finding] = []
    node = {
        "Function": {
            "input": [{"Column": "x"}, {"Frobnicate": None}],
            "function": "Shift",
        }
    }
    _compile._walk(node, FIXTURE_RULES, Context(), (), findings)

    assert _paths(tuple(findings)) == {("Frobnicate", ("input", 1))}


def test_non_node_in_a_declared_child_slot_is_refused() -> None:
    """A declared child holding something that is not a node refuses."""
    findings: list[Finding] = []
    node = {"Function": {"input": "not a node", "function": "Shift"}}
    _compile._walk(node, FIXTURE_RULES, Context(), (), findings)

    assert [f.kind for f in findings] == [_compile.NOT_A_NODE]
    assert findings[0].classification is Classification.REFUSE
    assert findings[0].path == ("input",)


def test_options_dict_is_never_mistaken_for_a_node() -> None:
    """Undeclared payload keys are not walked, whatever they contain."""
    findings: list[Finding] = []
    node = {
        "Function": {
            "input": [{"Column": "x"}],
            "function": "Shift",
            "options": {"Frobnicate": None},
        }
    }
    out = _compile._walk(node, FIXTURE_RULES, Context(), (), findings)

    assert findings == []
    assert out["Function"]["options"] == {"Frobnicate": None}


def test_a_rule_that_raises_is_refused() -> None:
    """Fail closed even when the rule table itself is broken."""

    def boom(_payload: Any, _ctx: Context) -> Classification:
        raise ValueError("rule is broken")

    table = {"Column": Rule(kind="Column", classify=boom, reason="unused")}
    findings: list[Finding] = []
    _compile._walk({"Column": "x"}, table, Context(), (), findings)

    assert findings[0].classification is Classification.REFUSE
    assert "rule is broken" in findings[0].reason


def test_rewrite_without_a_rewriter_is_refused() -> None:
    """A rule that says REWRITE but supplies none cannot silently pass."""
    table = {
        "Column": Rule(
            kind="Column",
            classify=lambda _p, _c: Classification.REWRITE,
            reason="needs a rewrite",
        )
    }
    findings: list[Finding] = []
    _compile._walk({"Column": "x"}, table, Context(), (), findings)

    assert findings[0].classification is Classification.REFUSE
    assert "supplies no rewrite" in findings[0].reason


def test_non_classification_return_is_refused() -> None:
    table = {
        "Column": Rule(
            kind="Column", classify=lambda _p, _c: True, reason="not a verdict"
        )
    }
    findings: list[Finding] = []
    _compile._walk({"Column": "x"}, table, Context(), (), findings)

    assert findings[0].classification is Classification.REFUSE
    assert "Classification" in findings[0].reason


@requires_cloudpickle
def test_map_batches_is_refused(fixture_rules: dict[str, Rule]) -> None:
    result = audit(pl.col("x").map_batches(lambda s: s), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert result.expr is None
    assert result.refused


@requires_rules
@requires_cloudpickle
def test_map_batches_is_refused_by_a_named_rule() -> None:
    """The real table refuses the opaque node explicitly, not by omission."""
    result = audit(pl.col("x").map_batches(lambda s: s), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert any(f.kind.startswith("AnonymousFunction") for f in result.refused)


# --------------------------------------------------------------------------
# Findings, paths, and the strict entry point
# --------------------------------------------------------------------------


def test_findings_carry_accurate_paths(fixture_rules: dict[str, Rule]) -> None:
    """Each finding points at the node it is about, in serialised-tree terms."""
    result = audit(LEAKY, time="t", entity="e")

    assert _paths(result.findings) == {
        ("Function.FillNullWithStrategy", (0, "function", "input", 0)),
        ("Function.RollingExpr", (0, "function")),
        ("Over", (0,)),
    }

    # The paths address the real tree: follow the deepest one by hand.
    tree = json.loads(LEAKY.meta.serialize(format="json"))
    node = tree["Alias"][0]["Over"]["function"]
    assert "RollingExpr" in node["Function"]["function"]
    assert (
        "FillNullWithStrategy" in node["Function"]["input"][0]["Function"]["function"]
    )


def test_audit_reports_every_problem_not_just_the_first(
    fixture_rules: dict[str, Rule],
) -> None:
    """Two independent leaks in one expression produce two findings."""
    expr = (pl.col("x").rank() - pl.col("x").mean()).alias("f")
    result = audit(expr, time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert _paths(result.findings) == {
        ("Function.Rank", (0, "left")),
        ("Agg.Mean", (0, "right")),
    }


def test_findings_are_bottom_up(fixture_rules: dict[str, Rule]) -> None:
    """Children are decided before their parents, so a parent sees the rewrite."""
    result = audit(LEAKY, time="t", entity="e")
    order = [f.kind for f in result.findings]
    assert order.index("Function.FillNullWithStrategy") < order.index(
        "Function.RollingExpr"
    )
    assert order.index("Function.RollingExpr") < order.index("Over")


def test_a_parent_rule_sees_its_rewritten_child() -> None:
    """The parent classifies the payload its children were rewritten into."""
    seen: list[Any] = []

    child = Rule(
        kind="Column",
        classify=lambda _p, _c: Classification.REWRITE,
        rewrite=lambda _p, _c: {"Column": "y"},
        reason="rename the column",
        rewrote_to="col('y')",
    )

    def parent_classify(payload: Any, _ctx: Context) -> Classification:
        seen.append(payload["input"][0])
        return Classification.SAFE

    parent = Rule(
        kind="Function.Shift",
        classify=parent_classify,
        reason="safe",
        children=("input",),
    )
    findings: list[Finding] = []
    node = {"Function": {"input": [{"Column": "x"}], "function": "Shift"}}
    out = _compile._walk(
        node, {"Column": child, "Function.Shift": parent}, Context(), (), findings
    )

    assert seen == [{"Column": "y"}]
    assert out["Function"]["input"][0] == {"Column": "y"}


def test_causalize_raises_leakage_refused_carrying_the_result(
    fixture_rules: dict[str, Rule],
) -> None:
    expr = (pl.col("x") - pl.col("x").mean()).alias("f")

    with pytest.raises(LeakageRefused) as excinfo:
        causalize(expr, time="t", entity="e")

    result = excinfo.value.result
    assert result.verdict is Verdict.REFUSED
    assert result.expr is None
    assert {f.kind for f in result.refused} >= {"Agg.Mean"}
    assert "Agg.Mean" in str(excinfo.value)


@requires_cloudpickle
def test_audit_never_raises_on_a_leak(fixture_rules: dict[str, Rule]) -> None:
    """Contrast with causalize: audit reports, it does not raise."""
    result = audit(pl.col("x").map_batches(lambda s: s))
    assert result.verdict is Verdict.REFUSED


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


def test_over_without_time_is_refused(fixture_rules: dict[str, Rule]) -> None:
    """Invariant 5: no time column means no way to repair, so refuse."""
    result = audit(LEAKY, entity="e")

    assert result.verdict is Verdict.REFUSED
    assert any(f.kind == "Over" for f in result.refused)


def test_over_with_time_gets_order_by_injected(fixture_rules: dict[str, Rule]) -> None:
    compiled = causalize(LEAKY, time="t", entity="e")
    tree = json.loads(compiled.meta.serialize(format="json"))
    assert tree["Alias"][0]["Over"]["order_by"][0] == {"Column": "t"}


def test_context_is_threaded_through_verbatim(fixture_rules: dict[str, Rule]) -> None:
    seen: list[Context] = []

    def record(_payload: Any, ctx: Context) -> Classification:
        seen.append(ctx)
        return Classification.SAFE

    table = {"Column": Rule(kind="Column", classify=record, reason="leaf")}
    findings: list[Finding] = []
    ctx = Context(time="t", entity="e", allow_approximate=True)
    _compile._walk({"Column": "x"}, table, ctx, (), findings)

    assert seen == [ctx]


def test_allow_approximate_reaches_the_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool] = []

    def record(_payload: Any, ctx: Context) -> Classification:
        seen.append(ctx.allow_approximate)
        return Classification.SAFE

    monkeypatch.setattr(
        _compile,
        "_rule_table",
        lambda: {"Column": Rule(kind="Column", classify=record, reason="leaf")},
    )
    audit(pl.col("x"), allow_approximate=True)
    assert seen == [True]


# --------------------------------------------------------------------------
# Misuse
# --------------------------------------------------------------------------


def test_audit_rejects_a_non_expression() -> None:
    with pytest.raises(TypeError, match="polars.Expr"):
        audit("x")  # type: ignore[arg-type]


def test_deserialisation_failure_is_loud(
    monkeypatch: pytest.MonkeyPatch, fixture_rules: dict[str, Rule]
) -> None:
    """Format drift must raise, never return a silently-wrong expression."""

    def broken(*_args: Any, **_kwargs: Any) -> pl.Expr:
        raise ValueError("unknown variant")

    monkeypatch.setattr(pl.Expr, "deserialize", staticmethod(broken))
    with pytest.raises(RuntimeError, match="POLARS_TREE_FORMAT_TESTED"):
        audit(SAFE_EXPR, time="t", entity="e")
