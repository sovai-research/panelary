"""Trusted leaves and cross-sectional scope: the two things a node cannot say.

Build-contract hard invariant 4 makes the registry the type system for
Panelary's own operators -- *"a registered op is a trusted leaf, which is what
stops the compiler refusing Panelary's own ``map_batches``-based operators"*.
This file is that invariant's test suite, and it is written around the fact
that made the obvious implementation impossible:

**A serialised ``AnonymousFunction`` carries no identity.** Its ``function``
slot is a cloudpickle blob; there is no name, no ``fmt_str``, and the only text
inside the pickle is the ``__qualname__`` and source path of whatever callable
was handed to ``map_batches`` -- forgeable (any lambda may live in a scope
called ``frac_diff``) and unstable (the bytes move with the closure's captured
values, the Python version and the cloudpickle version).
``test_nothing_in_the_node_identifies_the_operator`` pins that finding, because
the whole trust design follows from it: the compiler never *infers* trust, the
caller *declares* it, and the registry then decides whether that declaration is
allowed to stand.

So three locks, all of which must turn before an opaque node is cleared:

1. the caller names an operator in ``audit(..., trust=...)``;
2. the registry holds that name with ``leakage_safe=True``;
3. and its ``safe_scope`` is ``"rowwise"`` -- a ``"window"`` operator is a
   look-ahead the moment it is broadcast per row, which is how it appears
   inside an expression, so ``leakage_safe`` alone is never enough.

The second half of the file covers the other thing a node cannot see: its
parent. A bare ``rank()`` is a look-ahead and ``rank().over(<time>)`` is the
canonical cross-sectional feature, and telling them apart needs the enclosing
``Over``. Both halves keep the same obligation as the rest of the compiler:
nothing here may turn a refusal into an acceptance by accident, so every
addition is paired with a test that it did not.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import polars as pl
import pytest

import panelary  # noqa: F401  -- registers the namespaces' FeatureSpecs
from panelary.core.panel_frame import PanelFrame
from panelary.leakage import Classification, LeakageRefused, Verdict, audit, causalize
from panelary.leakage._compile import _walk
from panelary.leakage._rules import (
    CROSS_SECTIONAL_SAFE_KINDS,
    OPAQUE_KINDS,
    RULES,
    ScopedContext,
    build_context,
    child_context,
    classify_node,
    in_cross_section,
    resolve_trust,
    trusted_names,
)
from panelary.leakage._types import POLARS_TREE_FORMAT_TESTED, Context, Finding
from panelary.registry import FeatureRegistry, FeatureSpec, registry
from panelary.testing import assert_no_lookahead

_POLARS_MINOR = ".".join(pl.__version__.split(".")[:2])

pytestmark = pytest.mark.skipif(
    _POLARS_MINOR not in POLARS_TREE_FORMAT_TESTED,
    reason=(
        f"polars {pl.__version__}: the serialised expression format is not a "
        "stable public API and this suite reads it directly; it has only been "
        f"checked against {', '.join(POLARS_TREE_FORMAT_TESTED)}."
    ),
)


def tree(expr: pl.Expr) -> Any:
    return json.loads(expr.meta.serialize(format="json"))


def kinds(result: Any) -> set[str]:
    return {f.kind for f in result.findings}


def refused_kinds(result: Any) -> set[str]:
    return {f.kind for f in result.refused}


# --------------------------------------------------------------------------- #
# Specs to decide against. The first group is synthetic, registered in a
# throwaway registry so the process-wide singleton is never mutated; the second
# is whatever Panelary itself has registered today.
# --------------------------------------------------------------------------- #
def _spec(name: str, *, leakage_safe: bool, safe_scope: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        namespace="fake",
        input_shape="series",
        output_shape="series",
        tier="A",
        panel_safe=True,
        leakage_safe=leakage_safe,
        safe_scope=safe_scope,
        source="Panelary test",
        license="Apache-2.0",
    )


@pytest.fixture
def fake_registry() -> FeatureRegistry:
    """A registry holding one spec of every interesting shape."""
    reg = FeatureRegistry()
    reg.register(_spec("rowwise_op", leakage_safe=True, safe_scope="rowwise"))
    reg.register(_spec("window_op", leakage_safe=True, safe_scope="window"))
    reg.register(_spec("unclassified_op", leakage_safe=True, safe_scope="unspecified"))
    reg.register(_spec("leaky_op", leakage_safe=False, safe_scope="rowwise"))
    return reg


def _registered(*, safe_scope: str, leakage_safe: bool = True) -> str | None:
    """A real qualified name with these properties, or ``None`` if there is none."""
    preferred = ("panel.frac_diff", "factor.forward_return")
    matching = [
        spec.qualified_name
        for spec in registry.all()
        if spec.leakage_safe is leakage_safe
        and getattr(spec, "safe_scope", "unspecified") == safe_scope
    ]
    for name in preferred:
        if name in matching:
            return name
    return matching[0] if matching else None


@pytest.fixture
def rowwise_name() -> str:
    """A really-registered operator that trust is allowed to name."""
    name = _registered(safe_scope="rowwise")
    if name is None:
        pytest.skip(
            "no registered FeatureSpec has leakage_safe=True and "
            "safe_scope='rowwise' yet, so nothing is trustable and the "
            "compiler fails closed -- which is the correct behaviour, but "
            "leaves the accept path untestable end to end."
        )
    return name


@pytest.fixture
def window_name() -> str:
    """A really-registered operator that trust must refuse to name."""
    name = _registered(safe_scope="window")
    if name is None:
        pytest.skip("no registered FeatureSpec is safe_scope='window'")
    return name


def opaque(fn: Any = None) -> pl.Expr:
    """An expression whose root is an ``AnonymousFunction``."""
    return pl.col("x").map_batches(fn if fn is not None else (lambda s: s))


# --------------------------------------------------------------------------- #
# 1. The finding the whole design rests on
# --------------------------------------------------------------------------- #
def test_nothing_in_the_node_identifies_the_operator() -> None:
    """Two different UDFs are indistinguishable in every field a rule can read.

    The payload has exactly three keys. ``input`` is the operand subtree,
    ``options`` is two booleans-worth of call protocol, and ``function`` is a
    pickle. Nothing names the operator, so a rule that tried to recognise a
    registered op from the tree would be matching on bytes -- which is why
    trust is declared by the caller instead.
    """

    def looks_causal(s: pl.Series) -> pl.Series:
        return s

    def is_a_flagrant_leak(s: pl.Series) -> pl.Series:
        return s.shift(-1)

    honest = tree(opaque(looks_causal))["AnonymousFunction"]
    leaky = tree(opaque(is_a_flagrant_leak))["AnonymousFunction"]

    assert set(honest) == {"input", "function", "options"}
    assert honest["options"] == leaky["options"], (
        "options would be the only structured place a name could live"
    )
    assert honest["input"] == leaky["input"]
    # The single differing field is an opaque byte string, not an identifier.
    assert isinstance(honest["function"], list)
    assert all(isinstance(b, int) for b in honest["function"])
    assert honest["function"] != leaky["function"]


def test_the_opaque_kind_is_the_only_one_trust_can_clear() -> None:
    """Trust exists for the node that cannot be read, and for nothing else."""
    assert {"AnonymousFunction"} == OPAQUE_KINDS
    assert set(RULES) >= OPAQUE_KINDS


# --------------------------------------------------------------------------- #
# 2. resolve_trust: registered AND leakage_safe AND rowwise
# --------------------------------------------------------------------------- #
def test_a_rowwise_registered_op_can_be_trusted(fake_registry: FeatureRegistry) -> None:
    assert resolve_trust(["rowwise_op"], target=fake_registry) == {"fake.rowwise_op"}
    assert resolve_trust(["fake.rowwise_op"], target=fake_registry) == {
        "fake.rowwise_op"
    }
    assert resolve_trust("rowwise_op", target=fake_registry) == {"fake.rowwise_op"}


def test_a_window_scoped_op_cannot_be_trusted(fake_registry: FeatureRegistry) -> None:
    """``leakage_safe=True`` is not sufficient. This is the subtle half.

    A ``"window"`` operator is causal as a summary of an already-delimited
    window and a look-ahead the moment it is broadcast per row -- which is
    precisely how it appears inside an expression.
    """
    with pytest.raises(ValueError, match="safe_scope"):
        resolve_trust(["window_op"], target=fake_registry)


def test_an_unclassified_op_cannot_be_trusted(fake_registry: FeatureRegistry) -> None:
    """``"unspecified"`` is the absence of a claim, and absence fails closed."""
    with pytest.raises(ValueError, match="unspecified"):
        resolve_trust(["unclassified_op"], target=fake_registry)


def test_an_op_registered_as_leaky_cannot_be_trusted(
    fake_registry: FeatureRegistry,
) -> None:
    with pytest.raises(ValueError, match="leakage_safe=False"):
        resolve_trust(["leaky_op"], target=fake_registry)


def test_an_unregistered_name_cannot_be_trusted(
    fake_registry: FeatureRegistry,
) -> None:
    """Trust may only name what the registry vouches for, never a free string."""
    with pytest.raises(ValueError, match="no such operator is registered"):
        resolve_trust(["rowwise_op_v2"], target=fake_registry)


def test_one_bad_name_poisons_the_whole_declaration(
    fake_registry: FeatureRegistry,
) -> None:
    """No partial credit: a declaration is accepted whole or not at all."""
    with pytest.raises(ValueError):
        resolve_trust(["rowwise_op", "window_op"], target=fake_registry)


def test_a_non_string_name_is_a_type_error(fake_registry: FeatureRegistry) -> None:
    with pytest.raises(TypeError, match="strings"):
        resolve_trust([object()], target=fake_registry)  # type: ignore[list-item]


@pytest.mark.parametrize("empty", [(), [], None])
def test_no_declaration_means_no_trust(
    empty: Any, fake_registry: FeatureRegistry
) -> None:
    assert resolve_trust(empty, target=fake_registry) == frozenset()


def test_an_empty_name_is_not_an_empty_declaration(
    fake_registry: FeatureRegistry,
) -> None:
    """``trust=""`` names one operator, spelled badly -- it is not "no trust"."""
    with pytest.raises(ValueError, match="no such operator"):
        resolve_trust("", target=fake_registry)


def test_resolution_reads_the_live_registry(rowwise_name: str) -> None:
    """The default target is the process-wide singleton, not a copy."""
    assert resolve_trust([rowwise_name]) == {rowwise_name}
    bare = rowwise_name.split(".", 1)[1]
    assert resolve_trust([bare]) == {rowwise_name}


def test_a_real_window_scoped_operator_is_refused(window_name: str) -> None:
    """Not just the synthetic case: Panelary registers such operators today.

    ``factor.forward_return`` is the one to think about -- a forward return is
    registered ``leakage_safe=True`` because it is a label built over a
    delimited horizon, and it is the purest possible look-ahead per row.
    """
    with pytest.raises(ValueError, match="safe_scope"):
        audit(opaque(), trust=(window_name,))


# --------------------------------------------------------------------------- #
# 3. The escape hatch, end to end
# --------------------------------------------------------------------------- #
def test_map_batches_is_still_refused_by_default() -> None:
    """The default has not moved: no declaration, no acceptance."""
    result = audit(opaque(), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert result.expr is None
    assert refused_kinds(result) == {"AnonymousFunction"}


def test_a_trusted_declaration_clears_the_opaque_node(rowwise_name: str) -> None:
    result = audit(opaque(), time="t", entity="e", trust=(rowwise_name,))

    assert result.verdict is Verdict.SAFE
    assert isinstance(result.expr, pl.Expr)
    assert not result.refused


def test_the_acceptance_is_recorded_not_silent(rowwise_name: str) -> None:
    """An audit that waves a node through must say that it did.

    There is no fourth ``Classification``, so the finding is filed ``SAFE``:
    it leaves the verdict alone and stays out of ``refused`` / ``rewritten``,
    but a report built from ``findings`` shows the one node the compiler did
    not verify, and says whose word it took.
    """
    result = audit(opaque(), time="t", entity="e", trust=(rowwise_name,))

    trusted = [f for f in result.findings if f.kind == "AnonymousFunction"]
    assert len(trusted) == 1
    assert trusted[0].classification is Classification.SAFE
    assert rowwise_name in trusted[0].reason
    assert "did NOT verify" in trusted[0].reason


def test_trust_clears_every_opaque_node_because_it_cannot_tell_them_apart(
    rowwise_name: str,
) -> None:
    """The honest consequence of an unidentifiable node, asserted out loud.

    Trust is a property of the *call*, not of a node: the compiler has no way
    to check that this particular pickle is the operator that was named. Two
    opaque nodes therefore both clear, and both are recorded.
    """
    expr = opaque(lambda s: s) + opaque(lambda s: s * 2)
    result = audit(expr, time="t", entity="e", trust=(rowwise_name,))

    assert result.verdict is Verdict.SAFE
    assert [f.kind for f in result.findings] == ["AnonymousFunction"] * 2


def test_causalize_returns_a_usable_expression_for_a_trusted_op(
    rowwise_name: str,
) -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    compiled = causalize(opaque(lambda s: s * 2), time="t", trust=(rowwise_name,))

    assert frame.select(compiled.alias("f"))["f"].to_list() == [2.0, 4.0, 6.0, 8.0]


def test_causalize_still_refuses_without_the_declaration() -> None:
    with pytest.raises(LeakageRefused, match="AnonymousFunction"):
        causalize(opaque(), time="t", entity="e")


def test_a_registered_panelary_operator_compiles(rowwise_name: str) -> None:
    """The regression the gap was about: Panelary refusing its own features.

    ``pl.col(...).panel.frac_diff(...)`` is a registered, ``leakage_safe``,
    ``rowwise`` operator implemented with ``map_batches`` -- the AGENTS.md
    invariant-5 escape hatch -- and before this it was an unconditional
    refusal.
    """
    if rowwise_name != "panel.frac_diff":
        pytest.skip("panel.frac_diff is not registered as rowwise in this build")
    expr = pl.col("x").panel.frac_diff(0.5)

    assert audit(expr, time="t", entity="e").verdict is Verdict.REFUSED
    assert (
        audit(expr, time="t", entity="e", trust=("panel.frac_diff",)).verdict
        is Verdict.SAFE
    )


def test_the_registered_operator_really_is_causal(rowwise_name: str) -> None:
    """Invariant 3: the claim the registry makes is checked, not assumed."""
    if rowwise_name != "panel.frac_diff":
        pytest.skip("panel.frac_diff is not registered as rowwise in this build")
    panel = _wide_panel()
    assert_no_lookahead(
        pl.col("x").panel.frac_diff(0.5).over("e").alias("__f__"), panel
    )


# --------------------------------------------------------------------------- #
# 4. Trust weakens nothing else
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "expr"),
    [
        ("shift_lead", pl.col("x").shift(-1)),
        ("reverse_cum_sum", pl.col("x").cum_sum(reverse=True)),
        ("backward_interpolate", pl.col("x").interpolate()),
        ("whole_column_rank", pl.col("x").rank()),
        ("reverse", pl.col("x").reverse()),
        ("whole_column_median", pl.col("x").median()),
        ("unordered_over_no_time", pl.col("x").cum_sum().over("e")),
    ],
)
def test_trust_does_not_clear_anything_but_the_opaque_node(
    label: str, expr: pl.Expr, rowwise_name: str
) -> None:
    """Declaring trust must not become a blanket ``--force``."""
    assert audit(expr, trust=(rowwise_name,)).verdict is Verdict.REFUSED


def test_a_leak_beside_a_trusted_node_is_still_caught(rowwise_name: str) -> None:
    """The trusted leaf clears itself, and only itself."""
    expr = (opaque() + pl.col("y").shift(-1)).alias("f")
    result = audit(expr, time="t", entity="e", trust=(rowwise_name,))

    assert result.verdict is Verdict.REFUSED
    assert refused_kinds(result) == {"Function.Shift"}


def test_an_unknown_node_kind_is_refused_even_under_trust(rowwise_name: str) -> None:
    """Invariant 1 survives trust: unknown still means unproven."""
    findings: list[Finding] = []
    ctx = build_context(
        time="t", entity="e", allow_approximate=False, trust=(rowwise_name,)
    )
    _walk({"SomeFutureNode": {"input": []}}, RULES, ctx, (), findings)

    assert [f.classification for f in findings] == [Classification.REFUSE]


def test_a_plain_context_carries_no_trust() -> None:
    """The fail-closed default, at the one place it is read."""
    assert trusted_names(Context(time="t")) == frozenset()
    assert classify_node(tree(opaque()), Context(time="t")) is Classification.REFUSE
    assert (
        classify_node(tree(opaque()), ScopedContext(time="t")) is Classification.REFUSE
    )


def test_build_context_stays_a_plain_context_without_trust() -> None:
    """No declaration, no change: the default walk is what it always was."""
    plain = build_context(time="t", entity="e", allow_approximate=True, trust=())
    assert type(plain) is Context
    assert plain == Context(time="t", entity="e", allow_approximate=True)

    scoped = build_context(
        time="t", entity="e", allow_approximate=True, trust=("frac_diff",)
    )
    assert isinstance(scoped, ScopedContext)
    assert scoped.time == "t" and scoped.allow_approximate is True


def test_the_rules_module_does_not_import_the_registry_at_module_level() -> None:
    """Invariant: no import cycle, and no cost added to ``import panelary``.

    ``panelary.registry`` deliberately imports nothing from the rest of the
    package, so the direction is safe -- but the import still happens inside
    :func:`resolve_trust`, i.e. only when a caller actually declares trust.
    """
    from panelary.leakage import _rules

    text = pathlib.Path(_rules.__file__).read_text(encoding="utf-8")
    offenders = [
        line
        for line in text.splitlines()
        if line.startswith(("import panelary.registry", "from panelary.registry"))
    ]
    assert not offenders, offenders


# --------------------------------------------------------------------------- #
# 5. Cross-sectional scope: the other thing a node cannot see
# --------------------------------------------------------------------------- #
def _wide_panel() -> PanelFrame:
    """Six entities over eight dates -- wide enough for a cross-section."""
    entities = ["a", "b", "c", "d", "e", "f"]
    times = list(range(8))
    rows = {
        "e": [ent for ent in entities for _ in times],
        "t": [t for _ in entities for t in times],
    }
    rows["x"] = [float((i * 7 + 3) % 11) + i * 0.25 for i in range(len(rows["e"]))]
    rows["y"] = [float((i * 5 + 1) % 13) for i in range(len(rows["e"]))]
    return PanelFrame(pl.DataFrame(rows), entity="e", time="t")


def test_a_bare_rank_is_still_refused() -> None:
    """Unchanged, and the reason scope had to be threaded rather than dropped."""
    result = audit(pl.col("x").rank(), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert refused_kinds(result) == {"Function.Rank"}


def test_a_rank_over_the_time_column_is_accepted() -> None:
    """The false positive the hint set was written for, now fixed."""
    result = audit(pl.col("x").rank().over("t"), time="t", entity="e")

    assert result.verdict is Verdict.SAFE, [str(f) for f in result.findings]
    assert result.findings == ()


def test_a_rank_over_the_entity_column_is_still_refused() -> None:
    """Over one entity the window spans every date, so the rank reads the future."""
    result = audit(pl.col("x").rank().over("e"), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert "Function.Rank" in refused_kinds(result)


def test_a_rank_over_time_needs_the_time_column_to_be_known() -> None:
    """With no ``time``, ``.over("t")`` is just some column; fail closed."""
    result = audit(pl.col("x").rank().over("t"))

    assert result.verdict is Verdict.REFUSED
    assert "Function.Rank" in refused_kinds(result)


def test_scope_does_not_leak_out_through_a_nested_window() -> None:
    """An inner window resets it: the inner partition is the one that applies."""
    result = audit(pl.col("x").rank().over("e").over("t"), time="t", entity="e")

    assert result.verdict is Verdict.REFUSED
    assert "Function.Rank" in refused_kinds(result)


def test_extra_partition_keys_only_subdivide_the_date() -> None:
    """``.over(["sector", "date"])`` is as cross-sectional as ``.over("date")``."""
    result = audit(pl.col("x").rank().over(["y", "t"]), time="t", entity="e")

    assert result.verdict is Verdict.SAFE


def test_a_derived_partition_key_is_not_a_cross_section() -> None:
    """``.over(pl.col(t).cast(...))`` groups coarsely; groups then span dates."""
    result = audit(pl.col("x").rank().over(pl.col("t").cast(pl.Int8)), time="t")

    assert result.verdict is Verdict.REFUSED
    assert "Function.Rank" in refused_kinds(result)


def test_a_cross_sectional_mean_is_not_silently_rewritten() -> None:
    """The bug beneath the false positive: a rewrite that changed the meaning.

    ``x - x.mean().over(date)`` is the canonical cross-sectional demean.
    Without the enclosing ``Over``, ``Agg.Mean`` classified ``REWRITE`` and the
    compiler turned it into an *expanding* mean within the date -- a different
    feature, handed back as if it were a repair.
    """
    expr = pl.col("x") - pl.col("x").mean().over("t")
    result = audit(expr, time="t", entity="e")

    assert result.verdict is Verdict.SAFE
    assert result.findings == ()


def test_an_entity_window_still_rewrites_its_aggregate() -> None:
    """The contrast: over an entity a whole-column mean is a leak, as before."""
    expr = pl.col("x") - pl.col("x").mean().over("e")
    result = audit(expr, time="t", entity="e")

    assert result.verdict is Verdict.REWRITTEN
    assert "Agg.Mean" in kinds(result)


def test_over_the_time_column_needs_no_order_by() -> None:
    """Invariant 5 is about order, and inside one timestamp order is vacuous."""
    result = audit(pl.col("x").rank().over("t"), time="t")
    assert result.verdict is Verdict.SAFE

    unordered_over_entity = audit(pl.col("x").cum_sum().over("e"), time="t")
    assert unordered_over_entity.verdict is Verdict.REWRITTEN


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("rank", lambda order: pl.col("x").rank().over("t", order_by=order)),
        ("mean", lambda order: pl.col("x").mean().over("t", order_by=order)),
    ],
)
def test_dropping_the_order_by_injection_changes_no_value(
    label: str, build: Any
) -> None:
    """Why ``.over(<time>)`` may classify SAFE rather than be 'repaired'.

    Not classifying it ``REWRITE`` means the compiler stops injecting
    ``order_by=<time>`` there, so that injection has to be shown to be a no-op
    rather than argued to be one. It is: the sort key is constant inside the
    partition. Proved on a deliberately shuffled frame, which is the situation
    an unordered ``.over()`` silently trusts everywhere else.
    """
    shuffled = pl.DataFrame(
        {
            "e": list("abcabcabc"),
            "t": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [3.0, 1.0, 2.0, 9.0, 7.0, 8.0, 5.0, 6.0, 4.0],
        }
    )[[4, 0, 8, 2, 6, 1, 7, 3, 5]]

    unordered = shuffled.select(build(None).alias("f"))["f"].to_list()
    ordered = shuffled.select(build("t").alias("f"))["f"].to_list()
    assert unordered == ordered


@pytest.mark.parametrize(
    ("label", "expr"),
    [
        ("lead", pl.col("x").shift(-1).over("t")),
        ("reverse_cum_sum", pl.col("x").cum_sum(reverse=True).over("t")),
        ("interpolate", pl.col("x").interpolate().over("t")),
        ("reverse", pl.col("x").reverse().over("t")),
    ],
)
def test_a_cross_section_clears_only_the_listed_kinds(
    label: str, expr: pl.Expr
) -> None:
    """Narrow on purpose: the cross-section is not an amnesty for the subtree."""
    assert audit(expr, time="t", entity="e").verdict is Verdict.REFUSED


def test_an_aggregate_that_vouches_for_its_own_operand_keeps_vouching() -> None:
    """``Agg.Min`` / ``Agg.Max`` declare no children, so nothing else audits theirs.

    Short-circuiting them to SAFE inside a cross-section would wave an
    unaudited subtree through -- the one thing the nested-operand aggregates
    exist to prevent.
    """
    leaky = audit(pl.col("x").shift(-1).max().over("t"), time="t", entity="e")
    assert leaky.verdict is Verdict.REFUSED
    assert "Agg.Max" in refused_kinds(leaky)

    causal = audit(pl.col("x").shift(1).max().over("t"), time="t", entity="e")
    assert causal.verdict is Verdict.SAFE


def test_the_hint_set_is_what_a_cross_section_clears() -> None:
    """The documented set is the implemented policy, not a parallel list."""
    ctx = ScopedContext(time="t", cross_sectional=True)
    for kind in CROSS_SECTIONAL_SAFE_KINDS:
        rule = RULES[kind]
        if not rule.children:
            continue  # vouches for its own operand; covered by the test above
        assert rule.classify({}, ctx) is Classification.SAFE, kind


def test_child_context_only_reshapes_the_windowed_expression() -> None:
    """Grouping keys are keys, not values the feature is built from."""
    payload = tree(pl.col("x").rank().over("t"))["Over"]
    ctx = Context(time="t", entity="e")

    assert in_cross_section(child_context("Over", "function", payload, ctx))
    assert not in_cross_section(child_context("Over", "partition_by", payload, ctx))
    assert child_context("Function.Rank", "input", payload, ctx) is ctx


def test_child_context_returns_the_same_object_when_nothing_changes() -> None:
    """A plain context is threaded verbatim, as the walker's contract promises."""
    payload = tree(pl.col("x").rank().over("e"))["Over"]
    ctx = Context(time="t", entity="e")

    assert child_context("Over", "function", payload, ctx) is ctx


# --------------------------------------------------------------------------- #
# 6. Invariant 3: the cross-sectional claim is proved, not asserted
# --------------------------------------------------------------------------- #
def _leaks(expr: pl.Expr) -> bool:
    try:
        assert_no_lookahead(expr.alias("__f__"), _wide_panel())
    except AssertionError:
        return True
    return False


def test_a_whole_column_rank_really_does_leak() -> None:
    """First half: a 'leak' that does not reproduce is not a leak."""
    assert _leaks(pl.col("x").rank())


def test_a_cross_sectional_rank_really_does_not() -> None:
    """Second half: corrupting the future leaves every past rank untouched."""
    assert not _leaks(pl.col("x").rank().over("t"))


def test_a_cross_sectional_demean_really_does_not_leak() -> None:
    assert _leaks(pl.col("x") - pl.col("x").mean())
    assert not _leaks(pl.col("x") - pl.col("x").mean().over("t"))


def test_the_compiled_expression_evaluates_to_the_cross_sectional_answer() -> None:
    """A rewrite that changed the meaning would show up here as a wrong number."""
    frame = _wide_panel().collect()
    compiled = causalize(pl.col("x").rank().over("t"), time="t", entity="e")

    got = frame.select(compiled.alias("f"))["f"].to_list()
    want = frame.select(pl.col("x").rank().over("t").alias("f"))["f"].to_list()
    assert got == want
