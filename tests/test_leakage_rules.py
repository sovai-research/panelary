"""Tests for the leakage rule table (:mod:`panelary.leakage._rules`).

Three things are pinned here, in increasing order of importance.

1. **Golden trees.** The serialised Polars expression format is not a stable
   public API, so every rule carries a representative expression whose exact
   serialisation is asserted. If Polars changes the shape of a node, these fail
   loudly rather than the compiler silently mis-classifying it. The whole module
   skips on a Polars minor version that is not in
   ``POLARS_TREE_FORMAT_TESTED``.
2. **Reachability.** ``node_kind`` must return the key each rule is registered
   under, or the rule is dead code.
3. **Invariant 3 of the build contract** -- every rewrite is proved
   *empirically*: the leaky expression fails
   :func:`panelary.testing.assert_no_lookahead` and the rewritten one passes it.
   The single exemption is ``Agg.Count``; see
   ``test_count_rewrite_restores_prefix_invariance`` for why, and for the
   stronger proof used in its place.
"""

from __future__ import annotations

import importlib.util
import io
import json
from typing import Any

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.leakage._rules import (
    CROSS_SECTIONAL_SAFE_KINDS,
    RULES,
    child_nodes,
    classify_node,
    payload_of,
    rewrite_node,
    rule_for,
)
from panelary.leakage._types import (
    POLARS_TREE_FORMAT_TESTED,
    Classification,
    Context,
    node_kind,
)
from panelary.testing import assert_no_lookahead

_POLARS_MINOR = ".".join(pl.__version__.split(".")[:2])

pytestmark = pytest.mark.skipif(
    _POLARS_MINOR not in POLARS_TREE_FORMAT_TESTED,
    reason=(
        f"polars {pl.__version__}: the serialised expression format is not a "
        "stable public API and this rule table has only been checked against "
        f"{', '.join(POLARS_TREE_FORMAT_TESTED)}. Re-run the shapes against the "
        "new version, update panelary.leakage._rules and the golden trees "
        "below, then add the minor version to POLARS_TREE_FORMAT_TESTED."
    ),
)


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def tree(expr: pl.Expr) -> Any:
    """The serialised expression tree of ``expr``.

    Polars pickles the callable inside ``map_batches`` / ``map_elements`` with
    ``cloudpickle``; in the bare-core install (numpy + polars only) that case
    cannot be serialised, so it is skipped rather than failed.
    """
    try:
        return json.loads(expr.meta.serialize(format="json"))
    except Exception:
        if importlib.util.find_spec("cloudpickle") is None:
            pytest.skip(
                "serialising a Python UDF needs cloudpickle (absent in the bare-core install)"
            )
        raise


def unparse(node: Any) -> pl.Expr:
    """Read a serialised tree back into a :class:`polars.Expr`."""
    return pl.Expr.deserialize(io.BytesIO(json.dumps(node).encode()), format="json")


def repaired(expr: pl.Expr, ctx: Context | None = None) -> pl.Expr:
    """Apply the rule table's rewrite to the root node of ``expr``."""
    return unparse(rewrite_node(tree(expr), ctx))


def normalise(node: Any) -> Any:
    """Replace opaque pickled payloads with a marker so goldens stay readable."""
    if isinstance(node, list):
        if len(node) > 16 and all(isinstance(v, int) for v in node):
            return "<opaque bytes>"
        return [normalise(v) for v in node]
    if isinstance(node, dict):
        return {k: normalise(v) for k, v in node.items()}
    return node


# --------------------------------------------------------------------------- #
# One representative expression per rule
# --------------------------------------------------------------------------- #
def _map_batches() -> pl.Expr:
    return pl.col("x").map_batches(lambda s: s)


REPRESENTATIVE: dict[str, pl.Expr] = {
    "Column": pl.col("x"),
    "Literal": pl.lit(1.0),
    "BinaryExpr": pl.col("x") + pl.lit(1.0),
    "Cast": pl.col("x").cast(pl.Float64),
    "Ternary": pl.when(pl.col("x") > pl.lit(0.0))
    .then(pl.col("x"))
    .otherwise(pl.lit(0.0)),
    "Alias": pl.col("x").alias("y"),
    "Function.Abs": pl.col("x").abs(),
    "Function.Pow": pl.col("x").pow(2.0),
    "Function.FillNull": pl.col("x").fill_null(0.0),
    "Function.EwmMean": pl.col("x").ewm_mean(alpha=0.5),
    "Agg.First": pl.col("x").first(),
    "Function.Boolean": pl.col("x").is_null(),
    "Function.CumSum": pl.col("x").cum_sum(),
    "Function.CumProd": pl.col("x").cum_prod(),
    "Function.CumMin": pl.col("x").cum_min(),
    "Function.CumMax": pl.col("x").cum_max(),
    "Function.CumCount": pl.col("x").cum_count(),
    "Function.Shift": pl.col("x").shift(1),
    "Function.Diff": pl.col("x").diff(1),
    "Function.PctChange": pl.col("x").pct_change(1),
    "Function.FillNullWithStrategy": pl.col("x").fill_null(strategy="backward"),
    "Function.RollingExpr": pl.col("x").rolling_mean(3, center=True),
    "Agg.Sum": pl.col("x").sum(),
    "Agg.Min": pl.col("x").min(),
    "Agg.Max": pl.col("x").max(),
    "Agg.Count": pl.col("x").count(),
    "Agg.Mean": pl.col("x").mean(),
    "Agg.Std": pl.col("x").std(),
    "Agg.Var": pl.col("x").var(),
    "Over": pl.col("x").cum_sum().over("e"),
    "AnonymousFunction": _map_batches(),
    "Function.Interpolate": pl.col("x").interpolate(),
    "Function.Rank": pl.col("x").rank(),
    "Function.Reverse": pl.col("x").reverse(),
    "Function.Quantile": pl.col("x").quantile(0.5),
    "Agg.Median": pl.col("x").median(),
    "Agg.Last": pl.col("x").last(),
    "Agg.NUnique": pl.col("x").n_unique(),
    "Sort": pl.col("x").sort(),
}


# The tripwire: the exact serialisation each rule keys on, on Polars 1.44.
GOLDEN_TREES: dict[str, str] = {
    "Column": '{"Column": "x"}',
    "Literal": '{"Literal": {"Dyn": {"Float": 1.0}}}',
    "BinaryExpr": '{"BinaryExpr": {"left": {"Column": "x"}, "op": "Plus", "right": {"Literal": {"Dyn": {"Float": 1.0}}}}}',
    "Cast": '{"Cast": {"expr": {"Column": "x"}, "dtype": {"Literal": "Float64"}, "options": "Strict"}}',
    "Ternary": '{"Ternary": {"predicate": {"BinaryExpr": {"left": {"Column": "x"}, "op": "Gt", "right": {"Literal": {"Dyn": {"Float": 0.0}}}}}, "truthy": {"Column": "x"}, "falsy": {"Literal": {"Dyn": {"Float": 0.0}}}}}',
    "Alias": '{"Alias": [{"Column": "x"}, "y"]}',
    "Function.Abs": '{"Function": {"input": [{"Column": "x"}], "function": "Abs"}}',
    "Function.Pow": '{"Function": {"input": [{"Column": "x"}, {"Literal": {"Dyn": {"Float": 2.0}}}], "function": {"Pow": "Generic"}}}',
    "Function.FillNull": '{"Function": {"input": [{"Column": "x"}, {"Literal": {"Dyn": {"Float": 0.0}}}], "function": "FillNull"}}',
    "Function.EwmMean": '{"Function": {"input": [{"Column": "x"}], "function": {"EwmMean": {"options": {"alpha": 0.5, "adjust": true, "bias": false, "min_periods": 1, "ignore_nulls": false}}}}}',
    "Agg.First": '{"Agg": {"First": {"Column": "x"}}}',
    "Function.Boolean": '{"Function": {"input": [{"Column": "x"}], "function": {"Boolean": "IsNull"}}}',
    "Function.CumSum": '{"Function": {"input": [{"Column": "x"}], "function": {"CumSum": {"reverse": false}}}}',
    "Function.CumProd": '{"Function": {"input": [{"Column": "x"}], "function": {"CumProd": {"reverse": false}}}}',
    "Function.CumMin": '{"Function": {"input": [{"Column": "x"}], "function": {"CumMin": {"reverse": false}}}}',
    "Function.CumMax": '{"Function": {"input": [{"Column": "x"}], "function": {"CumMax": {"reverse": false}}}}',
    "Function.CumCount": '{"Function": {"input": [{"Column": "x"}], "function": {"CumCount": {"reverse": false}}}}',
    "Function.Shift": '{"Function": {"input": [{"Column": "x"}, {"Literal": {"Dyn": {"Int": 1}}}], "function": "Shift"}}',
    "Function.Diff": '{"Function": {"input": [{"Column": "x"}, {"Literal": {"Dyn": {"Int": 1}}}], "function": {"Diff": "Ignore"}}}',
    "Function.PctChange": '{"Function": {"input": [{"Column": "x"}, {"Literal": {"Dyn": {"Int": 1}}}], "function": "PctChange"}}',
    "Function.FillNullWithStrategy": '{"Function": {"input": [{"Column": "x"}], "function": {"FillNullWithStrategy": {"Backward": null}}}}',
    "Function.RollingExpr": '{"Function": {"input": [{"Column": "x"}], "function": {"RollingExpr": {"function": "Mean", "options": {"window_size": 3, "min_periods": 3, "weights": null, "center": true, "fn_params": null}}}}}',
    "Agg.Sum": '{"Agg": {"Sum": {"Column": "x"}}}',
    "Agg.Min": '{"Agg": {"Min": {"input": {"Column": "x"}, "propagate_nans": false}}}',
    "Agg.Max": '{"Agg": {"Max": {"input": {"Column": "x"}, "propagate_nans": false}}}',
    "Agg.Count": '{"Agg": {"Count": {"input": {"Column": "x"}, "include_nulls": false}}}',
    "Agg.Mean": '{"Agg": {"Mean": {"Column": "x"}}}',
    "Agg.Std": '{"Agg": {"Std": [{"Column": "x"}, 1]}}',
    "Agg.Var": '{"Agg": {"Var": [{"Column": "x"}, 1]}}',
    "Over": '{"Over": {"function": {"Function": {"input": [{"Column": "x"}], "function": {"CumSum": {"reverse": false}}}}, "partition_by": [{"Column": "e"}], "order_by": null, "mapping": "GroupsToRows"}}',
    "AnonymousFunction": '{"AnonymousFunction": {"input": [{"Column": "x"}], "function": "<opaque bytes>", "options": {"check_lengths": true, "flags": "OPTIONAL_RE_ENTRANT"}}}',
    "Function.Interpolate": '{"Function": {"input": [{"Column": "x"}], "function": {"Interpolate": "Linear"}}}',
    "Function.Rank": '{"Function": {"input": [{"Column": "x"}], "function": {"Rank": {"options": {"method": "Average", "descending": false}, "seed": null}}}}',
    "Function.Reverse": '{"Function": {"input": [{"Column": "x"}], "function": "Reverse"}}',
    "Function.Quantile": '{"Function": {"input": [{"Column": "x"}, {"Literal": {"Dyn": {"Float": 0.5}}}], "function": {"Quantile": {"method": "Nearest"}}}}',
    "Agg.Median": '{"Agg": {"Median": {"Column": "x"}}}',
    "Agg.Last": '{"Agg": {"Last": {"Column": "x"}}}',
    "Agg.NUnique": '{"Agg": {"NUnique": {"Column": "x"}}}',
    "Sort": '{"Sort": {"expr": {"Column": "x"}, "options": {"descending": false, "nulls_last": false, "multithreaded": true, "maintain_order": false, "limit": null}}}',
}


# --------------------------------------------------------------------------- #
# Table hygiene
# --------------------------------------------------------------------------- #
def test_every_rule_has_a_representative_and_a_golden_tree() -> None:
    assert set(REPRESENTATIVE) == set(RULES)
    assert set(GOLDEN_TREES) == set(RULES)


@pytest.mark.parametrize("kind", sorted(RULES))
def test_node_kind_matches_the_key_the_rule_is_registered_under(kind: str) -> None:
    """No unreachable rules: ``node_kind`` must produce each registered key."""
    assert node_kind(tree(REPRESENTATIVE[kind])) == kind
    assert RULES[kind].kind == kind


@pytest.mark.parametrize("kind", sorted(RULES))
def test_golden_tree(kind: str) -> None:
    """Tripwire for Polars changing its serialised expression format."""
    assert normalise(tree(REPRESENTATIVE[kind])) == json.loads(GOLDEN_TREES[kind])


@pytest.mark.parametrize("kind", sorted(RULES))
def test_every_rule_has_an_actionable_reason(kind: str) -> None:
    reason = RULES[kind].reason
    assert len(reason) > 30
    assert reason == reason.strip()
    assert reason.lower() != "unsafe"


def test_a_rule_that_can_rewrite_says_what_it_rewrites_to() -> None:
    for rule in RULES.values():
        assert (rule.rewrite is None) == (rule.rewrote_to is None), rule.kind


def test_cross_sectional_hint_only_names_registered_kinds() -> None:
    assert set(RULES) >= CROSS_SECTIONAL_SAFE_KINDS


# --------------------------------------------------------------------------- #
# Children: where the walker must recurse
# --------------------------------------------------------------------------- #
EXPECTED_CHILDREN: dict[str, tuple[str, ...]] = {
    "Alias": ("0",),
    "BinaryExpr": ("left", "right"),
    "Cast": ("expr",),
    "Ternary": ("predicate", "truthy", "falsy"),
    "Over": ("function", "partition_by"),
    "AnonymousFunction": ("input",),
    "Column": (),
    "Literal": (),
    "Sort": ("expr",),
}


@pytest.mark.parametrize(("kind", "expected"), sorted(EXPECTED_CHILDREN.items()))
def test_children_keys(kind: str, expected: tuple[str, ...]) -> None:
    assert RULES[kind].children == expected


@pytest.mark.parametrize("kind", sorted(k for k in RULES if k.startswith("Function.")))
def test_function_children_are_the_input_list(kind: str) -> None:
    assert RULES[kind].children == ("input",)


#: Polars buries these aggregates' operand two levels down
#: (``{"Min": {"input": <node>, ...}}``), which no single ``children`` key can
#: name; they declare no children and vouch for the operand in ``classify``.
NESTED_OPERAND_AGGS = ("Agg.Min", "Agg.Max", "Agg.Count")


@pytest.mark.parametrize(
    "kind",
    sorted(k for k in RULES if k.startswith("Agg.") and k not in NESTED_OPERAND_AGGS),
)
def test_agg_children_name_the_variant(kind: str) -> None:
    assert RULES[kind].children == (kind.split(".", 1)[1],)


@pytest.mark.parametrize("kind", NESTED_OPERAND_AGGS)
def test_nested_operand_aggs_declare_no_children(kind: str) -> None:
    assert RULES[kind].children == ()


@pytest.mark.parametrize("kind", sorted(RULES))
def test_child_nodes_reaches_the_operand_column(kind: str) -> None:
    """Every representative wraps ``pl.col("x")``; the walker must find it."""
    node = tree(REPRESENTATIVE[kind])
    if kind in ("Column", "Literal"):
        pytest.skip("leaf node: nothing to recurse into")
    if kind in NESTED_OPERAND_AGGS:
        pytest.skip("operand is audited by the rule, not walked (see classify)")
    found = {node_kind(child) for _path, child in _descendants(node)}
    assert "Column" in found


def _descendants(node: Any) -> list[tuple[Any, Any]]:
    """Every node reachable from ``node`` through the rule table's children."""
    out: list[tuple[Any, Any]] = []
    rule = rule_for(node)
    if rule is None:
        return out
    for path, child in child_nodes(payload_of(node), rule.children):
        out.append((path, child))
        out.extend(_descendants(child))
    return out


def test_alias_is_walked_through() -> None:
    """The canonical leak sits under an Alias; the "0" slot must expose it."""
    node = tree(pl.col("x").shift(-1).alias("lead"))
    kinds = [node_kind(child) for _p, child in _descendants(node)]
    assert "Function.Shift" in kinds


@pytest.mark.parametrize(
    "leaky",
    [
        pl.col("x").shift(-1).min(),
        pl.col("x").shift(-1).max(),
        pl.col("x").shift(-1).count(),
    ],
    ids=NESTED_OPERAND_AGGS,
)
def test_nested_operand_aggs_refuse_an_operand_they_cannot_vouch_for(
    leaky: pl.Expr,
) -> None:
    """Nothing walks their operand, so an unaudited one is refused, not rewritten."""
    assert classify_node(tree(leaky), Context()) is REFUSE


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").shift(1).min(),
        (pl.col("x") * 2).max(),
        pl.col("x").cum_sum().count(),
    ],
    ids=["shift_lag_min", "scaled_max", "cum_sum_count"],
)
def test_nested_operand_aggs_still_rewrite_a_causal_operand(expr: pl.Expr) -> None:
    assert classify_node(tree(expr), Context()) is REWRITE


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
SAFE = Classification.SAFE
REWRITE = Classification.REWRITE
REFUSE = Classification.REFUSE

_TIME = Context(time="t", entity="e")

CLASSIFICATIONS: list[tuple[str, pl.Expr, Context, Classification]] = [
    # Leaves and plumbing.
    ("column", pl.col("x"), Context(), SAFE),
    ("literal", pl.lit(1.0), Context(), SAFE),
    ("binary", pl.col("x") + pl.lit(1.0), Context(), SAFE),
    ("cast", pl.col("x").cast(pl.Float64), Context(), SAFE),
    ("ternary", pl.when(pl.col("x") > 0).then(1.0).otherwise(0.0), Context(), SAFE),
    ("alias", pl.col("x").alias("y"), Context(), SAFE),
    ("first", pl.col("x").first(), Context(), SAFE),
    ("fill_null_value", pl.col("x").fill_null(0.0), Context(), SAFE),
    ("ewm_mean", pl.col("x").ewm_mean(alpha=0.5), Context(), SAFE),
    # Row-wise predicates are safe; whole-column ones are not.
    ("is_null", pl.col("x").is_null(), Context(), SAFE),
    ("is_finite", pl.col("x").is_finite(), Context(), SAFE),
    ("is_unique", pl.col("x").is_unique(), Context(), REFUSE),
    ("is_duplicated", pl.col("x").is_duplicated(), Context(), REFUSE),
    ("any", pl.col("b").any(), Context(), REFUSE),
    # Cumulative: forwards causal, reversed fatal.
    ("cum_sum", pl.col("x").cum_sum(), Context(), SAFE),
    ("cum_sum_reverse", pl.col("x").cum_sum(reverse=True), Context(), REFUSE),
    ("cum_max_reverse", pl.col("x").cum_max(reverse=True), Context(), REFUSE),
    ("cum_count_reverse", pl.col("x").cum_count(reverse=True), Context(), REFUSE),
    # Offsets: the sign of the literal decides.
    ("shift_lag", pl.col("x").shift(1), Context(), SAFE),
    ("shift_zero", pl.col("x").shift(0), Context(), SAFE),
    ("shift_lead", pl.col("x").shift(-1), Context(), REFUSE),
    (
        "shift_lead_typed",
        pl.col("x").shift(pl.lit(-2, dtype=pl.Int32)),
        Context(),
        REFUSE,
    ),
    ("shift_dynamic", pl.col("x").shift(pl.col("k")), Context(), REFUSE),
    ("diff_lag", pl.col("x").diff(1), Context(), SAFE),
    ("diff_lead", pl.col("x").diff(-1), Context(), REFUSE),
    ("pct_change_lag", pl.col("x").pct_change(1), Context(), SAFE),
    ("pct_change_lead", pl.col("x").pct_change(-1), Context(), REFUSE),
    # fill_null strategies.
    ("fill_forward", pl.col("x").fill_null(strategy="forward"), Context(), SAFE),
    ("fill_zero", pl.col("x").fill_null(strategy="zero"), Context(), SAFE),
    ("fill_backward", pl.col("x").fill_null(strategy="backward"), Context(), REWRITE),
    ("fill_mean", pl.col("x").fill_null(strategy="mean"), Context(), REFUSE),
    ("fill_max", pl.col("x").fill_null(strategy="max"), Context(), REFUSE),
    # Rolling windows.
    ("rolling_trailing", pl.col("x").rolling_mean(3), Context(), SAFE),
    ("rolling_centred", pl.col("x").rolling_mean(3, center=True), Context(), REWRITE),
    (
        "rolling_quantile_centred",
        pl.col("x").rolling_quantile(0.5, window_size=3, center=True),
        Context(),
        REWRITE,
    ),
    # Aggregates: a leak over data, safe over a constant.
    ("sum", pl.col("x").sum(), Context(), REWRITE),
    ("sum_of_constant", pl.lit(3.0).sum(), Context(), SAFE),
    ("min", pl.col("x").min(), Context(), REWRITE),
    ("max", pl.col("x").max(), Context(), REWRITE),
    ("count", pl.col("x").count(), Context(), REWRITE),
    ("len", pl.col("x").len(), Context(), REWRITE),
    ("mean", pl.col("x").mean(), Context(), REWRITE),
    ("std", pl.col("x").std(), Context(), REWRITE),
    ("var", pl.col("x").var(), Context(), REWRITE),
    ("median", pl.col("x").median(), Context(), REFUSE),
    ("median_approx_ok", pl.col("x").median(), Context(allow_approximate=True), REFUSE),
    ("quantile", pl.col("x").quantile(0.5), Context(), REFUSE),
    (
        "quantile_approx_ok",
        pl.col("x").quantile(0.5),
        Context(allow_approximate=True),
        REFUSE,
    ),
    ("last", pl.col("x").last(), Context(), REFUSE),
    ("n_unique", pl.col("x").n_unique(), Context(), REFUSE),
    # Windows.
    ("over_unordered", pl.col("x").cum_sum().over("e"), _TIME, REWRITE),
    ("over_unordered_no_time", pl.col("x").cum_sum().over("e"), Context(), REFUSE),
    (
        "over_ordered",
        pl.col("x").cum_sum().over("e", order_by="t"),
        Context(),
        SAFE,
    ),
    # Fatal.
    ("map_batches", _map_batches(), Context(), REFUSE),
    (
        "map_elements",
        pl.col("x").map_elements(lambda v: v, return_dtype=pl.Float64),
        Context(),
        REFUSE,
    ),
    ("interpolate", pl.col("x").interpolate(), Context(), REFUSE),
    ("rank", pl.col("x").rank(), Context(), REFUSE),
    ("reverse", pl.col("x").reverse(), Context(), REFUSE),
    ("sort", pl.col("x").sort(), Context(), REFUSE),
]


@pytest.mark.parametrize(
    ("expr", "ctx", "expected"),
    [case[1:] for case in CLASSIFICATIONS],
    ids=[case[0] for case in CLASSIFICATIONS],
)
def test_classification(expr: pl.Expr, ctx: Context, expected: Classification) -> None:
    assert classify_node(tree(expr), ctx) is expected


def test_unrecognised_nodes_fail_closed() -> None:
    assert classify_node({"SomeFutureNode": {"input": []}}) is REFUSE
    assert rule_for({"SomeFutureNode": {}}) is None


def test_bare_string_nodes_fail_closed() -> None:
    """``pl.len()`` serialises to the bare string ``"Len"``, not a node dict.

    ``node_kind`` cannot qualify it, so it has no rule and is refused. Pinned
    here so the gap is a known, deliberate one rather than a surprise.
    """
    assert node_kind(tree(pl.len())) is None
    assert classify_node(tree(pl.len())) is REFUSE


def test_rewrite_is_refused_for_nodes_that_do_not_have_one() -> None:
    with pytest.raises(ValueError, match="no rewrite"):
        rewrite_node(tree(pl.col("x")))
    with pytest.raises(ValueError, match="no rewrite"):
        rewrite_node(tree(pl.col("x").cum_sum().over("e")), Context())


# --------------------------------------------------------------------------- #
# Invariant 3: every rewrite is proved empirically
# --------------------------------------------------------------------------- #
_CLEAN = [1.0, 2.0, 3.5, 4.0, 5.0, 6.5, 7.0, 8.0]
_GAPPY = [1.0, 2.0, None, None, 5.0, 6.0, 7.0, 8.0]


def _panel(values: list[float | None], *, shuffle: bool = False) -> PanelFrame:
    df = pl.DataFrame(
        {"e": ["a"] * len(values), "t": list(range(len(values))), "x": values}
    )
    if shuffle:
        # Interleave the two halves: still one entity, still every timestamp,
        # but no longer in time order -- which is exactly the situation an
        # `.over()` without `order_by` silently trusts.
        df = df[[0, 4, 1, 5, 2, 6, 3, 7]]
    return PanelFrame(df, entity="e", time="t")


def _leaks(expr: pl.Expr, panel: PanelFrame) -> bool:
    try:
        assert_no_lookahead(expr.alias("__f__"), panel)
    except AssertionError:
        return True
    return False


# (kind, leaky expression, panel, context) -- one per rewriting rule.
LOOKAHEAD_PROOFS: list[tuple[str, pl.Expr, PanelFrame, Context]] = [
    (
        "Function.FillNullWithStrategy",
        pl.col("x").fill_null(strategy="backward"),
        _panel(_GAPPY),
        Context(),
    ),
    (
        "Function.RollingExpr",
        pl.col("x").rolling_mean(3, center=True),
        _panel(_CLEAN),
        Context(),
    ),
    (
        "Over",
        pl.col("x").cum_sum().over("e"),
        _panel(_CLEAN, shuffle=True),
        _TIME,
    ),
    ("Agg.Sum", pl.col("x").sum(), _panel(_CLEAN), Context()),
    ("Agg.Min", pl.col("x").min(), _panel(_CLEAN), Context()),
    ("Agg.Max", pl.col("x").max(), _panel(_CLEAN), Context()),
    ("Agg.Mean", pl.col("x").mean(), _panel(_CLEAN), Context()),
    ("Agg.Std", pl.col("x").std(), _panel(_CLEAN), Context()),
    ("Agg.Var", pl.col("x").var(), _panel(_CLEAN), Context()),
]


@pytest.mark.parametrize(
    ("leaky", "panel", "ctx"),
    [case[1:] for case in LOOKAHEAD_PROOFS],
    ids=[case[0] for case in LOOKAHEAD_PROOFS],
)
def test_the_leaky_form_really_does_leak(
    leaky: pl.Expr, panel: PanelFrame, ctx: Context
) -> None:
    """Invariant 3, first half: a rule whose 'leak' does not reproduce is not a rule."""
    assert _leaks(leaky, panel)


@pytest.mark.parametrize(
    ("leaky", "panel", "ctx"),
    [case[1:] for case in LOOKAHEAD_PROOFS],
    ids=[case[0] for case in LOOKAHEAD_PROOFS],
)
def test_the_rewrite_removes_the_leak(
    leaky: pl.Expr, panel: PanelFrame, ctx: Context
) -> None:
    """Invariant 3, second half: the rewritten expression passes the verifier."""
    assert_no_lookahead(repaired(leaky, ctx).alias("__f__"), panel)


def test_every_rewriting_rule_is_proved() -> None:
    """No rewrite may ship without an empirical proof somewhere in this file."""
    rewriting = {kind for kind, rule in RULES.items() if rule.rewrite is not None}
    proved = {case[0] for case in LOOKAHEAD_PROOFS} | {"Agg.Count"}
    assert rewriting == proved


def test_count_rewrite_restores_prefix_invariance() -> None:
    """``Agg.Count``: the exemption from the perturbation proof, and its stand-in.

    ``assert_no_lookahead`` perturbs *values*, never the null pattern, so a
    whole-column count is bit-identical before and after and the verifier
    cannot see the leak. The leak is real all the same, and it is the
    length-dependence kind: ``count()`` at row ``t`` changes when rows after
    ``t`` are appended. Prefix invariance is the sharper instrument here.
    """
    leaky = pl.col("x").count()
    assert not _leaks(leaky, _panel(_GAPPY))  # documented blind spot

    def values(expr: pl.Expr, upto: int) -> list[Any]:
        frame = pl.DataFrame({"x": _GAPPY[:upto]})
        return frame.select(expr.alias("f"))["f"].to_list()

    assert values(leaky, 4) != values(leaky, 8)[:4]
    fixed = repaired(leaky)
    assert values(fixed, 4) == values(fixed, 8)[:4]


@pytest.mark.parametrize(
    ("leaky", "panel", "ctx"),
    [case[1:] for case in LOOKAHEAD_PROOFS if not case[0].startswith("Over")],
    ids=[case[0] for case in LOOKAHEAD_PROOFS if not case[0].startswith("Over")],
)
def test_rewrites_are_prefix_invariant(
    leaky: pl.Expr, panel: PanelFrame, ctx: Context
) -> None:
    """``f(x[:T])[t] == f(x[:T+k])[t]``: the rewrite may not depend on len(x)."""
    frame = panel.collect()
    fixed = repaired(leaky, ctx)
    short = frame.head(5).select(fixed.alias("f"))["f"].to_list()
    full = frame.select(fixed.alias("f"))["f"].to_list()[:5]
    assert short == pytest.approx(full, nan_ok=True)


# --------------------------------------------------------------------------- #
# Invariant 2: a rewrite is exact
# --------------------------------------------------------------------------- #
_EXACTNESS = [1.0, 2.0, None, 4.0, 5.5, -3.0, 7.0, 2.5]


def _point_in_time(reduce: Any) -> list[float | None]:
    values = np.array([np.nan if v is None else v for v in _EXACTNESS])
    out: list[float | None] = []
    for i in range(len(values)):
        seen = values[: i + 1]
        out.append(reduce(seen[~np.isnan(seen)]))
    return out


EXACTNESS_CASES = [
    ("sum", pl.col("x").sum(), lambda a: float(a.sum())),
    ("min", pl.col("x").min(), lambda a: float(a.min()) if len(a) else None),
    ("max", pl.col("x").max(), lambda a: float(a.max()) if len(a) else None),
    ("count", pl.col("x").count(), lambda a: len(a)),
    ("len", pl.col("x").len(), None),
    ("mean", pl.col("x").mean(), lambda a: float(a.mean()) if len(a) else None),
    (
        "std",
        pl.col("x").std(),
        lambda a: float(a.std(ddof=1)) if len(a) > 1 else None,
    ),
    (
        "var",
        pl.col("x").var(),
        lambda a: float(a.var(ddof=1)) if len(a) > 1 else None,
    ),
    (
        "std_ddof0",
        pl.col("x").std(ddof=0),
        lambda a: float(a.std(ddof=0)) if len(a) > 0 else None,
    ),
]


@pytest.mark.parametrize(
    ("leaky", "reduce"),
    [case[1:] for case in EXACTNESS_CASES],
    ids=[case[0] for case in EXACTNESS_CASES],
)
def test_expanding_rewrites_equal_the_point_in_time_aggregate(
    leaky: pl.Expr, reduce: Any
) -> None:
    frame = pl.DataFrame({"x": _EXACTNESS})
    got = frame.select(repaired(leaky).alias("f"))["f"].to_list()
    expected = (
        list(range(1, len(_EXACTNESS) + 1))
        if reduce is None
        else _point_in_time(reduce)
    )
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if e is None:
            assert g is None
        else:
            assert g is not None
            assert g == pytest.approx(e, abs=1e-9)


def test_backward_fill_rewrite_is_the_forward_fill() -> None:
    frame = pl.DataFrame({"x": _GAPPY})
    leaky = pl.col("x").fill_null(strategy="backward")
    got = frame.select(repaired(leaky).alias("f"))["f"].to_list()
    assert (
        got
        == frame.select(pl.col("x").fill_null(strategy="forward").alias("f"))[
            "f"
        ].to_list()
    )


def test_backward_fill_rewrite_keeps_the_limit() -> None:
    node = rewrite_node(tree(pl.col("x").fill_null(strategy="backward", limit=2)))
    assert node["Function"]["function"] == {"FillNullWithStrategy": {"Forward": 2}}


def test_rolling_rewrite_only_flips_center() -> None:
    leaky = pl.col("x").rolling_mean(5, center=True)
    assert tree(unparse(rewrite_node(tree(leaky)))) == tree(
        pl.col("x").rolling_mean(5, center=False)
    )


def test_over_rewrite_injects_the_context_time_column() -> None:
    leaky = pl.col("x").cum_sum().over("e")
    assert tree(unparse(rewrite_node(tree(leaky), _TIME))) == tree(
        pl.col("x").cum_sum().over("e", order_by="t")
    )


# --------------------------------------------------------------------------- #
# A rewrite must not smuggle in something the table would refuse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("leaky", "ctx"),
    [(case[1], case[3]) for case in LOOKAHEAD_PROOFS]
    + [(pl.col("x").count(), Context()), (pl.col("x").len(), Context())],
    ids=[case[0] for case in LOOKAHEAD_PROOFS] + ["Agg.Count", "Agg.Count.len"],
)
def test_the_replacement_subtree_is_itself_classified_safe(
    leaky: pl.Expr, ctx: Context
) -> None:
    replacement = rewrite_node(tree(leaky), ctx)
    nodes = [replacement] + [child for _p, child in _descendants(replacement)]
    for node in nodes:
        assert classify_node(node, ctx) is not REFUSE, node_kind(node)
