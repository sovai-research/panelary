"""The rule table: what the point-in-time compiler may do with each node kind.

:data:`RULES` maps a *qualified node kind* -- exactly what
:func:`panelary.leakage._types.node_kind` returns for a serialised Polars
expression node -- to a :class:`~panelary.leakage._types.Rule` saying whether
that node is causal, repairable, or fatal.

Reading the table
-----------------

Classification is by **what the node does**, never by its name. A
``Function.CumSum`` is safe forwards and fatal in reverse; an ``Agg.Sum`` is
safe over a constant and a leak over a column, because a whole-column sum is
broadcast back onto every row and so hands rows before ``t`` a number computed
from rows after ``t``. Three families therefore classify dynamically:

* the cumulative family, on ``reverse``;
* ``Function.Shift`` / ``Diff`` / ``PctChange``, on the sign of the literal
  offset in ``input[1]``;
* the aggregates, on whether their input is a constant.

Everything not in the table is refused -- invariant 1, fail closed.

The one assumption every rule shares
------------------------------------

"Before ``t``" means "earlier in the frame's row order". Every SAFE
classification and every rewrite emitted here is causal *provided rows are in
time order within the window they are evaluated over*. That is precisely why
``Over`` with ``order_by: null`` is not safe: it is the one node that can
silently redefine the order. Repair it (inject ``Context.time``) and the
assumption holds again.

Two things a node cannot see about itself
-----------------------------------------

A rule is handed one node and decides on its merits. Two decisions are not
properties of the node at all, so the walker puts them on the context first
(see :class:`ScopedContext`):

**Scope.** A whole-column ``rank()`` is a look-ahead; the same ``rank()``
inside ``.over(<time>)`` is the canonical cross-sectional feature, because
every row of the partition shares one timestamp and so no row in it is in any
other's future. :func:`child_context` marks the subtree under such an ``Over``
and :data:`CROSS_SECTIONAL_SAFE_KINDS` says which kinds that clears.

**Trust.** ``map_batches`` / ``map_elements`` serialise to an
``AnonymousFunction`` whose ``function`` slot is a *cloudpickle blob*: there is
no stable name, no ``fmt_str``, nothing in the tree that says which operator it
is. Panelary's own per-group operators are built on it (AGENTS.md invariant 5),
so an unconditional refusal refuses the library's own features -- but nothing
can be recovered from the node, and matching on text found inside a pickle
would be an unauthenticated, forgeable claim. So the identification is the
*caller's*: ``audit(..., trust=("panel.frac_diff",))`` declares it, and
:func:`resolve_trust` will only accept a name the registry holds with
``leakage_safe=True`` **and** ``safe_scope="rowwise"``. Both locks must turn;
neither is inferred from the tree. See :func:`resolve_trust` for what that does
and does not prove.

Calling convention
------------------

``Rule.classify`` and ``Rule.rewrite`` receive the node's **payload** -- the
value under the single top-level discriminator key, i.e. ``node["Function"]``
for a ``Function.*`` kind and ``node["Agg"]`` for an ``Agg.*`` kind -- plus the
:class:`~panelary.leakage._types.Context`.

``Rule.rewrite`` returns a **complete replacement node** (a single-key dict),
not a payload. It has to: a rewrite may change the node's kind outright --
``Agg.Mean`` becomes a ``Ternary`` wrapping a ``BinaryExpr``, because the
expanding mean is ``cum_sum() / cum_count()`` and there is no ``Agg`` variant
that means "expanding mean". Callers should go through :func:`rewrite_node`,
which takes and returns whole nodes and never exposes the distinction.

``Rule.children`` names the payload slots under which child nodes live, in the
same vocabulary the walker in ``_compile.py`` understands: a mapping key
(``"input"``, ``"left"``), or a positional key for a sequence payload --
``Alias`` is ``[node, name]``, so its child is ``"0"``. A slot may hold a
single node or a list of them (``Over.partition_by``); :func:`child_nodes`
resolves both and skips the non-node entries that share those lists (a ddof
integer, an alias string).

``Agg.Min``, ``Agg.Max`` and ``Agg.Count`` are the one place this vocabulary
runs out. Polars nests their operand two levels down -- ``{"Min": {"input":
<node>, "propagate_nans": false}}`` -- and a single key cannot name that, so
they declare no children and take responsibility themselves: their ``classify``
audits the operand with :func:`_subtree_is_safe` and refuses unless every node
in it is already causal. That keeps invariant 1 (nothing unaudited is ever
rewritten) at the cost of refusing an aggregate over an operand that would
itself have needed repairing. Widen it the moment ``children`` can express a
two-level path.

Rewrites are built by serialising a real Polars expression written against a
placeholder column and substituting the original child subtree back in, so the
replacement JSON is whatever the installed Polars would itself emit -- never
hand-written.
"""

from __future__ import annotations

import copy
import functools
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import polars as pl

from panelary.leakage._types import Classification, Context, Rule, node_kind

__all__ = [
    "CROSS_SECTIONAL_SAFE_KINDS",
    "OPAQUE_KINDS",
    "RULES",
    "ScopedContext",
    "build_context",
    "child_context",
    "child_nodes",
    "classify_node",
    "in_cross_section",
    "payload_of",
    "resolve_trust",
    "rewrite_node",
    "rule_for",
    "trusted_names",
]

#: Kinds that leak over a whole column but are perfectly causal when evaluated
#: inside ``.over(Context.time)`` -- a cross-sectional statistic uses only rows
#: sharing the same timestamp. A node cannot see its parent, so these classify
#: on their own (leaky) merits by default; :func:`child_context` marks the
#: subtree of a time-partitioned ``Over`` and :func:`_cross_sectionally_safe`
#: clears exactly these kinds inside it, so a legitimate cross-sectional feature
#: such as ``pl.col("x").rank().over("date")`` is no longer a false positive.
#:
#: The set is deliberately narrow: every member is an *order-invariant* summary
#: of the partition, so its value cannot depend on where a row sits inside the
#: date. Kinds that are order-sensitive (``Agg.Last``, the cumulative family)
#: stay out, even though a single-timestamp partition also makes them causal --
#: their value would depend on the frame's row order, which is not something to
#: bless silently.
CROSS_SECTIONAL_SAFE_KINDS: frozenset[str] = frozenset(
    {
        "Agg.Max",
        "Agg.Mean",
        "Agg.Median",
        "Agg.Min",
        "Agg.NUnique",
        "Agg.Std",
        "Agg.Sum",
        "Agg.Var",
        "Function.Quantile",
        "Function.Rank",
    }
)

#: Kinds the serialised tree cannot identify. ``map_batches`` / ``map_elements``
#: carry their callable as a cloudpickle blob, so two of these nodes are
#: indistinguishable to the compiler no matter what they compute. They are
#: refused unless the caller declares trust; see :func:`resolve_trust`.
OPAQUE_KINDS: frozenset[str] = frozenset({"AnonymousFunction"})


# --------------------------------------------------------------------------- #
# Scope and trust: what the walker knows and a node does not
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScopedContext(Context):
    """A :class:`Context` carrying what only the walker can establish.

    Both extra fields are **positional facts about the walk**, not properties
    of any node, which is why they cannot live in a rule's payload:

    Parameters
    ----------
    trusted : frozenset[str]
        Qualified names of registered operators the *caller* has declared it
        trusts, already checked against the registry by :func:`resolve_trust`.
        Non-empty means an opaque node (see :data:`OPAQUE_KINDS`) is accepted.
    cross_sectional : bool
        ``True`` inside the subtree of an ``.over(...)`` whose partition keys
        include :attr:`Context.time`, i.e. where every row of the window shares
        one timestamp.

    Notes
    -----
    The rules read both through :func:`trusted_names` and
    :func:`in_cross_section`, which fall back to "no trust, no cross-section"
    for a plain :class:`Context`. That fallback is the fail-closed default: a
    context that does not say a subtree is trusted or cross-sectional is
    treated as saying it is not.
    """

    trusted: frozenset[str] = frozenset()
    cross_sectional: bool = False


def trusted_names(ctx: Context) -> frozenset[str]:
    """Operator names the caller declared trusted; empty for a plain context."""
    names = getattr(ctx, "trusted", None)
    return names if isinstance(names, frozenset) else frozenset()


def in_cross_section(ctx: Context) -> bool:
    """True inside a window partitioned by the time column."""
    return getattr(ctx, "cross_sectional", False) is True


def _trust_index(reg: Any) -> dict[str, Any]:
    """Map both ``name`` and ``namespace.name`` to each registered spec."""
    index: dict[str, Any] = {}
    for spec in reg:
        index[spec.name] = spec
        index[spec.qualified_name] = spec
    return index


def resolve_trust(
    names: str | Iterable[str] | None, *, target: Any = None
) -> frozenset[str]:
    """Check declared trust against the registry, returning qualified names.

    Hard invariant 4 of the build contract makes the registry the type system
    for Panelary's own operators: a registered :class:`~panelary.registry.FeatureSpec`
    carries the safety contract, so a registered op can be a *trusted leaf* and
    the compiler need not refuse the library's own ``map_batches``-based
    features. This is the function that enforces it.

    Two conditions, both required:

    * the name is registered, and its ``leakage_safe`` is ``True``;
    * its ``safe_scope`` is ``"rowwise"``.

    The second is not redundant. ``leakage_safe`` alone answers a question that
    has no single answer -- *safe evaluated how?* A ``"window"`` operator is
    causal as a summary of an already-delimited window and a look-ahead the
    moment it is broadcast per row, which is exactly how it appears inside an
    expression. ``factor.forward_return`` is registered ``leakage_safe=True``
    for that reason and must never be a trusted leaf. ``"unspecified"`` means
    nobody has classified it, which is not a claim, so it is refused too.

    What this proves, and what it does not
    --------------------------------------
    It proves the *named operator* is one Panelary vouches for row by row. It
    does **not** prove the opaque node in the tree is that operator: nothing
    can, because the node carries only a cloudpickle blob. Trust is therefore a
    property of the audit *call*, asserted by the caller, and it clears every
    opaque node in that one expression. The compiler records the fact in a
    :class:`~panelary.leakage._types.Finding` rather than passing silently.

    Parameters
    ----------
    names : str or iterable of str, optional
        Operator names, bare (``"frac_diff"``) or qualified
        (``"panel.frac_diff"``). ``None`` or empty means no trust.
    target : FeatureRegistry, optional
        Registry to resolve against; defaults to the process-wide singleton.

    Returns
    -------
    frozenset[str]
        The qualified names, once every one has passed both checks.

    Raises
    ------
    TypeError
        If ``names`` is not a string or an iterable of strings.
    ValueError
        If a name is unregistered, is not ``leakage_safe``, or is not
        ``safe_scope="rowwise"``. Refusing loudly at the boundary beats
        refusing quietly at the node: the caller asked for trust and is told
        precisely why it was not granted.
    """
    if names is None:
        return frozenset()
    wanted = (names,) if isinstance(names, str) else tuple(names)
    if not wanted:
        return frozenset()

    if target is None:
        # Lazy and first-party: `panelary.registry` imports nothing from the
        # rest of the package, so this cannot cycle, and deferring it to call
        # time keeps `import panelary.leakage` exactly as cheap as it was.
        from panelary.registry import registry

        target = registry

    index = _trust_index(target)
    resolved: set[str] = set()
    for raw in wanted:
        if not isinstance(raw, str):
            raise TypeError(
                f"trust must name registered operators as strings, got "
                f"{type(raw).__name__}: {raw!r}"
            )
        name = raw.strip()
        spec = index.get(name)
        if spec is None:
            raise ValueError(
                f"cannot trust {name!r}: no such operator is registered. Trust "
                "may only name an operator in `panelary.registry` -- that "
                "registration is what carries the safety contract. Use the "
                "bare name ('frac_diff') or the qualified one "
                "('panel.frac_diff')."
            )
        if not spec.leakage_safe:
            raise ValueError(
                f"cannot trust {spec.qualified_name!r}: it is registered with "
                "leakage_safe=False, i.e. Panelary itself does not claim it is "
                "causal. Fix the FeatureSpec if the claim is wrong; do not "
                "route around it here."
            )
        scope = getattr(spec, "safe_scope", "unspecified")
        if scope != "rowwise":
            raise ValueError(
                f"cannot trust {spec.qualified_name!r}: its safe_scope is "
                f"{scope!r}, not 'rowwise'. leakage_safe=True alone is not "
                "enough to trust a leaf. A 'window' operator is causal only as "
                "a summary of an already-delimited window -- broadcast per row, "
                "which is how it appears inside an expression, it is a "
                "look-ahead by construction. 'unspecified' means nobody has "
                "classified it, so it is refused for the same reason an "
                "unknown node kind is."
            )
        resolved.add(spec.qualified_name)
    return frozenset(resolved)


def build_context(
    *,
    time: str | None = None,
    entity: str | None = None,
    allow_approximate: bool = False,
    trust: str | Iterable[str] | None = (),
) -> Context:
    """The context an audit starts from, with any declared trust resolved.

    Returns a plain :class:`Context` when nothing extra is carried, so the
    default walk is unchanged in every observable way, and a
    :class:`ScopedContext` only once there is something to put on it.
    """
    resolved = resolve_trust(trust)
    if not resolved:
        return Context(time=time, entity=entity, allow_approximate=allow_approximate)
    return ScopedContext(
        time=time,
        entity=entity,
        allow_approximate=allow_approximate,
        trusted=resolved,
    )


def _with_cross_section(ctx: Context, flag: bool) -> Context:
    """``ctx`` with :attr:`ScopedContext.cross_sectional` set to ``flag``."""
    if isinstance(ctx, ScopedContext):
        return (
            ctx if ctx.cross_sectional is flag else replace(ctx, cross_sectional=flag)
        )
    if not flag:
        return ctx  # a plain Context already means "not in a cross-section"
    return ScopedContext(
        time=ctx.time,
        entity=ctx.entity,
        allow_approximate=ctx.allow_approximate,
        cross_sectional=True,
    )


def _partitions_by_time(payload: Any, ctx: Context) -> bool:
    """True when an ``Over`` payload partitions by :attr:`Context.time`.

    Every row of such a window carries the same timestamp, so no row in it is
    in any other's future. Extra partition keys only subdivide that set, which
    is why *containing* the time column is enough: ``.over(["sector", "date"])``
    is as cross-sectional as ``.over("date")``.

    The key must be a bare column reference equal to ``ctx.time``. A derived
    key such as ``.over(pl.col("date").dt.year())`` is *coarser* than the date
    -- its groups span many timestamps -- so it is deliberately not recognised.
    """
    if ctx.time is None:
        return False
    keys = _get(payload, "partition_by")
    if not isinstance(keys, list):
        return False
    return any(key == {"Column": ctx.time} for key in keys)


def child_context(kind: str, slot: Any, payload: Any, ctx: Context) -> Context:
    """The context in which ``payload``'s ``slot`` child subtree is evaluated.

    The walker consults this before descending, which is what gives a node the
    one thing it cannot read off itself: what it is nested *inside*. Only
    ``Over`` changes anything, and only for the expression it windows --
    ``partition_by`` holds grouping keys, not values the feature is built from.

    The flag is always **set**, never merely propagated, so scope cannot leak
    outward through a nested window: in ``rank().over(entity).over(time)`` the
    inner ``.over(entity)`` resets it and the rank is refused, which is right,
    because that window spans every date of one entity.
    """
    if kind != "Over" or slot != "function":
        return ctx
    return _with_cross_section(ctx, _partitions_by_time(payload, ctx))


# --------------------------------------------------------------------------- #
# Node plumbing
# --------------------------------------------------------------------------- #
def payload_of(node: Any) -> Any:
    """Value under a node's single discriminator key."""
    return next(iter(node.values()))


def _is_node(value: Any) -> bool:
    return node_kind(value) is not None


def child_nodes(payload: Any, children: tuple[str, ...]) -> Iterator[tuple[Any, Any]]:
    """Yield ``(path, child_node)`` for every child reachable from ``payload``.

    ``path`` is a tuple of keys/indices relative to the payload, suitable for
    extending a :attr:`panelary.leakage._types.Finding.path`. Non-node entries
    (a ddof integer, an alias string, an options dict) are skipped.
    """
    for key in children:
        value = _get(payload, key)
        slot: Any = int(key) if key.lstrip("-").isdigit() else key
        if value is None:
            continue
        if _is_node(value):
            yield (slot,), value
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if _is_node(item):
                    yield (slot, i), item


def _get(payload: Any, key: str) -> Any:
    """The value at ``key``, for mapping *and* sequence payloads.

    Sequence payloads (``Alias`` is ``[expr, name]``) address their children by
    positional key -- ``"0"`` -- which is the same convention the walker in
    ``_compile.py`` uses. The two must agree: when they did not, every aliased
    expression was refused instead of audited, which would have hidden a leak
    behind a rename.
    """
    if isinstance(payload, dict):
        return payload.get(key)
    if isinstance(payload, (list, tuple)) and key.lstrip("-").isdigit():
        index = int(key)
        return payload[index] if -len(payload) <= index < len(payload) else None
    return None


def rule_for(node: Any) -> Rule | None:
    """The rule governing ``node``, or ``None`` when it is unrecognised."""
    kind = node_kind(node)
    return RULES.get(kind) if kind is not None else None


def classify_node(node: Any, ctx: Context | None = None) -> Classification:
    """Classify a whole node. Unrecognised kinds are :data:`REFUSE`."""
    rule = rule_for(node)
    if rule is None:
        return Classification.REFUSE
    return rule.classify(payload_of(node), ctx if ctx is not None else Context())


def rewrite_node(node: Any, ctx: Context | None = None) -> Any:
    """Return the point-in-time replacement for ``node`` (a whole node).

    Raises ``ValueError`` if the node is not classified
    :data:`~panelary.leakage._types.Classification.REWRITE` under ``ctx``.
    """
    ctx = ctx if ctx is not None else Context()
    rule = rule_for(node)
    payload = payload_of(node)
    if (
        rule is None
        or rule.rewrite is None
        or rule.classify(payload, ctx) is not Classification.REWRITE
    ):
        raise ValueError(f"{node_kind(node)!r} has no rewrite under this context")
    return rule.rewrite(payload, ctx)


# --------------------------------------------------------------------------- #
# Building replacement subtrees
# --------------------------------------------------------------------------- #
_PLACEHOLDER = "__panelary_leakage_child__"


def _serialize(expr: pl.Expr) -> Any:
    return json.loads(expr.meta.serialize(format="json"))


@functools.cache
def _template(name: str, arg: int = 0) -> Any:
    """Serialised expression tree for ``name``, written against a placeholder."""
    p = pl.col(_PLACEHOLDER)
    builder = _TEMPLATES[name]
    return _serialize(builder(p, arg))


def _graft(tree: Any, child: Any) -> Any:
    """Copy ``tree``, replacing every placeholder column with ``child``."""
    if isinstance(tree, dict):
        if tree == {"Column": _PLACEHOLDER}:
            return copy.deepcopy(child)
        return {k: _graft(v, child) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_graft(v, child) for v in tree]
    return tree


def _expanding(name: str, child: Any, arg: int = 0) -> Any:
    return _graft(_template(name, arg), child)


def _null_f64() -> pl.Expr:
    return pl.lit(None, dtype=pl.Float64)


def _count(p: pl.Expr) -> pl.Expr:
    return p.cum_count().cast(pl.Int64)


def _expanding_mean_expr(p: pl.Expr) -> pl.Expr:
    n = _count(p)
    total = p.fill_null(0).cum_sum()
    return pl.when(n > 0).then(total / n).otherwise(_null_f64())


def _expanding_var_expr(p: pl.Expr, ddof: int) -> pl.Expr:
    n = _count(p)
    x = p.fill_null(0).cast(pl.Float64)
    s1 = x.cum_sum()
    s2 = (x * x).cum_sum()
    var = (s2 - s1 * s1 / n) / (n - ddof)
    return pl.when(n > ddof).then(var).otherwise(_null_f64())


def _expanding_std_expr(p: pl.Expr, ddof: int) -> pl.Expr:
    n = _count(p)
    x = p.fill_null(0).cast(pl.Float64)
    s1 = x.cum_sum()
    s2 = (x * x).cum_sum()
    var = (s2 - s1 * s1 / n) / (n - ddof)
    # Cancellation in the sum-of-squares form can push an exactly-zero variance
    # a hair below zero; clamp before the square root rather than emit NaN.
    nonneg = pl.when(var > 0).then(var).otherwise(pl.lit(0.0))
    return pl.when(n > ddof).then(nonneg.sqrt()).otherwise(_null_f64())


# Null handling is where exactness is won or lost. `sum()` treats nulls as
# absent (an all-null column sums to 0) while `cum_sum()` emits a null *at* a
# null row; `min()`/`max()` skip nulls while `cum_min()`/`cum_max()` emit one.
# Filling first restores the aggregate's own null semantics without changing a
# single value: a forward fill can only repeat a value already seen, so it
# cannot move the running minimum or maximum.
_TEMPLATES: dict[str, Callable[[pl.Expr, int], pl.Expr]] = {
    "cum_sum": lambda p, _: p.fill_null(0).cum_sum(),
    "cum_min": lambda p, _: p.fill_null(strategy="forward").cum_min(),
    "cum_max": lambda p, _: p.fill_null(strategy="forward").cum_max(),
    "cum_count": lambda p, _: p.cum_count(),
    "cum_count_all": lambda p, _: p.is_null().cum_count(),
    "mean": lambda p, _: _expanding_mean_expr(p),
    "var": _expanding_var_expr,
    "std": _expanding_std_expr,
}


@functools.cache
def _order_by_payload(time: str) -> Any:
    """The ``Over.order_by`` payload Polars itself emits for ``order_by=time``."""
    probe = pl.col("__x__").cum_sum().over("__g__", order_by=time)
    return _serialize(probe)["Over"]["order_by"]


# --------------------------------------------------------------------------- #
# Node predicates
# --------------------------------------------------------------------------- #
def _fn_variant(payload: Any) -> tuple[str, Any]:
    """``(variant name, variant body)`` of a ``Function`` payload."""
    fn = _get(payload, "function")
    if isinstance(fn, str):
        return fn, None
    if isinstance(fn, dict) and len(fn) == 1:
        key = next(iter(fn))
        return key, fn[key]
    return "", None


def _inner_variant(body: Any) -> tuple[str, Any]:
    """``(name, body)`` of a nested single-key enum such as a fill strategy."""
    if isinstance(body, str):
        return body, None
    if isinstance(body, dict) and len(body) == 1:
        key = next(iter(body))
        return key, body[key]
    return "", None


def _literal_number(node: Any) -> float | None:
    """The numeric value of a ``Literal`` node, or ``None``."""
    if node_kind(node) != "Literal":
        return None
    value: Any = node["Literal"]
    while isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _is_constant(node: Any) -> bool:
    """True when ``node`` cannot depend on any row of data."""
    kind = node_kind(node)
    if kind == "Literal":
        return True
    if kind == "Cast":
        return _is_constant(node["Cast"]["expr"])
    if kind == "Alias":
        return _is_constant(node["Alias"][0])
    if kind == "BinaryExpr":
        body = node["BinaryExpr"]
        return _is_constant(body["left"]) and _is_constant(body["right"])
    return False


def _agg_input(payload: Any, variant: str) -> Any:
    """The aggregated child node of an ``Agg.<variant>`` payload."""
    body = _get(payload, variant)
    if _is_node(body):
        return body
    if isinstance(body, list) and body and _is_node(body[0]):
        return body[0]
    if isinstance(body, dict) and _is_node(body.get("input")):
        return body["input"]
    return None


def _agg_ddof(payload: Any, variant: str, default: int = 1) -> int:
    body = _get(payload, variant)
    if isinstance(body, list) and len(body) > 1 and isinstance(body[1], int):
        return int(body[1])
    return default


def _offset_sign_ok(payload: Any) -> bool | None:
    """True/False for a non-negative/negative literal offset, ``None`` if opaque."""
    inputs = _get(payload, "input")
    if not isinstance(inputs, list) or len(inputs) < 2:
        return None
    value = _literal_number(inputs[1])
    if value is None:
        return None
    return value >= 0


# --------------------------------------------------------------------------- #
# Rule constructors
# --------------------------------------------------------------------------- #
def _const(result: Classification) -> Callable[[Any, Context], Classification]:
    def classify(_payload: Any, _ctx: Context) -> Classification:
        return result

    return classify


def _safe(kind: str, reason: str, children: tuple[str, ...] = ()) -> Rule:
    return Rule(
        kind=kind,
        classify=_const(Classification.SAFE),
        reason=reason,
        children=children,
    )


def _refuse(kind: str, reason: str, children: tuple[str, ...] = ()) -> Rule:
    return Rule(
        kind=kind,
        classify=_const(Classification.REFUSE),
        reason=reason,
        children=children,
    )


_FN: tuple[str, ...] = ("input",)


# --------------------------------------------------------------------------- #
# Leaves and plumbing
# --------------------------------------------------------------------------- #
_LEAVES: list[Rule] = [
    _safe("Column", "a column reference reads the current row only."),
    _safe("Literal", "a literal is the same at every row."),
    _safe(
        "BinaryExpr",
        "an arithmetic or comparison operator is row-wise; it can only leak if "
        "one of its operands does.",
        ("left", "right"),
    ),
    _safe(
        "Cast",
        "a dtype cast is row-wise; it cannot move information between rows.",
        ("expr",),
    ),
    _safe(
        "Ternary",
        "when/then/otherwise selects row by row; it can only leak if one of its "
        "branches does.",
        ("predicate", "truthy", "falsy"),
    ),
    # The payload is the sequence [expr, name], so the child lives at the
    # positional key "0". Get this wrong and the walker refuses every aliased
    # expression instead of auditing what is under the alias -- which would
    # hide a leak behind a rename, since `shift(-1).alias("lead")` puts the
    # Alias at the root.
    _safe("Alias", "renaming an expression cannot move information in time.", ("0",)),
    _safe(
        "Function.Abs",
        "absolute value is row-wise; it cannot move information between rows.",
        _FN,
    ),
    _safe(
        "Function.Pow",
        "exponentiation is row-wise; it cannot move information between rows.",
        _FN,
    ),
    _safe(
        "Function.FillNull",
        "filling nulls with an explicit value or expression is row-wise; the "
        "fill value is audited on its own merits.",
        _FN,
    ),
    _safe(
        "Function.EwmMean",
        "an exponentially weighted mean is a trailing, causal average.",
        _FN,
    ),
    _safe(
        "Agg.First",
        "the first value of a column is known from the first row onwards, so "
        "broadcasting it backwards moves no information into the past.",
        ("First",),
    ),
]


# --------------------------------------------------------------------------- #
# Boolean predicates: elementwise ones are safe, whole-column ones are not
# --------------------------------------------------------------------------- #
_ROWWISE_BOOLEAN = frozenset(
    {
        "IsNull",
        "IsNotNull",
        "IsNan",
        "IsNotNan",
        "IsFinite",
        "IsInfinite",
        "Not",
        "IsIn",
        "IsBetween",
    }
)


def _classify_boolean(payload: Any, _ctx: Context) -> Classification:
    _, body = _fn_variant(payload)
    name, _ = _inner_variant(body)
    return Classification.SAFE if name in _ROWWISE_BOOLEAN else Classification.REFUSE


_BOOLEAN_RULE = Rule(
    kind="Function.Boolean",
    classify=_classify_boolean,
    reason=(
        "row-wise predicates (is_null, is_nan, is_finite, not, is_in) are "
        "causal, but the whole-column ones (is_unique, is_duplicated, any, all) "
        "read every row, including rows after t; compute them per date with "
        "`.over(<time>)` or over a trailing window."
    ),
    children=_FN,
)


# --------------------------------------------------------------------------- #
# The cumulative family: causal forwards, pure future in reverse
# --------------------------------------------------------------------------- #
def _classify_cumulative(payload: Any, _ctx: Context) -> Classification:
    _, body = _fn_variant(payload)
    reverse = body.get("reverse") if isinstance(body, dict) else None
    if reverse is None:
        return Classification.REFUSE
    return Classification.REFUSE if reverse else Classification.SAFE


_CUMULATIVE_REASON = (
    "a reversed cumulative accumulates rows t..n-1, i.e. the value at t is a "
    "function of the future and nothing else; there is no causal equivalent, "
    "because the quantity itself is defined by what has not happened yet. Drop "
    "`reverse=True` to accumulate forwards."
)

_CUMULATIVE: list[Rule] = [
    Rule(
        kind=f"Function.{name}",
        classify=_classify_cumulative,
        reason=_CUMULATIVE_REASON,
        children=_FN,
    )
    for name in ("CumSum", "CumProd", "CumMin", "CumMax", "CumCount")
]


# --------------------------------------------------------------------------- #
# Offsets: shift / diff / pct_change
# --------------------------------------------------------------------------- #
def _classify_offset(payload: Any, _ctx: Context) -> Classification:
    ok = _offset_sign_ok(payload)
    if ok is None:
        return Classification.REFUSE
    return Classification.SAFE if ok else Classification.REFUSE


_OFFSET_RULES: list[Rule] = [
    Rule(
        kind="Function.Shift",
        classify=_classify_offset,
        reason=(
            "a negative shift copies a future observation backwards onto row t; "
            "use `shift(k)` with k >= 0 for a lag. A non-literal offset is "
            "refused because its sign cannot be checked."
        ),
        children=_FN,
    ),
    Rule(
        kind="Function.Diff",
        classify=_classify_offset,
        reason=(
            "`diff(-k)` differences against a future row; use `diff(k)` with "
            "k >= 0. A non-literal offset is refused because its sign cannot be "
            "checked."
        ),
        children=_FN,
    ),
    Rule(
        kind="Function.PctChange",
        classify=_classify_offset,
        reason=(
            "`pct_change(-k)` compares row t against a future row; use a "
            "non-negative period. A non-literal offset is refused because its "
            "sign cannot be checked."
        ),
        children=_FN,
    ),
]


# --------------------------------------------------------------------------- #
# fill_null(strategy=...)
# --------------------------------------------------------------------------- #
def _fill_strategy(payload: Any) -> tuple[str, Any]:
    _, body = _fn_variant(payload)
    return _inner_variant(body)


def _classify_fill_strategy(payload: Any, _ctx: Context) -> Classification:
    name, _ = _fill_strategy(payload)
    if name == "Backward":
        return Classification.REWRITE
    if name in ("Forward", "Zero", "One"):
        return Classification.SAFE
    return Classification.REFUSE


def _rewrite_fill_strategy(payload: Any, _ctx: Context) -> Any:
    _, limit = _fill_strategy(payload)
    replacement = copy.deepcopy(payload)
    replacement["function"] = {"FillNullWithStrategy": {"Forward": limit}}
    return {"Function": replacement}


_FILL_RULE = Rule(
    kind="Function.FillNullWithStrategy",
    classify=_classify_fill_strategy,
    reason=(
        "a backward fill copies a future observation into the past; use a "
        "forward fill, which carries the last value you actually had. The "
        "`min`/`max`/`mean` strategies fill from a whole-column statistic that "
        "includes rows after t and are refused: fill with an explicitly "
        "expanding statistic instead."
    ),
    rewrite=_rewrite_fill_strategy,
    rewrote_to="fill_null(strategy='forward')",
    children=_FN,
)


# --------------------------------------------------------------------------- #
# Rolling windows
# --------------------------------------------------------------------------- #
def _rolling_options(payload: Any) -> Any:
    _, body = _fn_variant(payload)
    return body.get("options") if isinstance(body, dict) else None


def _classify_rolling(payload: Any, _ctx: Context) -> Classification:
    options = _rolling_options(payload)
    if not isinstance(options, dict) or "center" not in options:
        return Classification.REFUSE
    return Classification.REWRITE if options["center"] else Classification.SAFE


def _rewrite_rolling(payload: Any, _ctx: Context) -> Any:
    replacement = copy.deepcopy(payload)
    variant, body = _fn_variant(replacement)
    body["options"]["center"] = False
    replacement["function"] = {variant: body}
    return {"Function": replacement}


_ROLLING_RULE = Rule(
    kind="Function.RollingExpr",
    classify=_classify_rolling,
    reason=(
        "a centred rolling window straddles row t, so half of every window is "
        "the future; the causal form is the trailing window of the same width, "
        "`center=False`. Note the values change: the window now ends at t "
        "instead of being centred on it."
    ),
    rewrite=_rewrite_rolling,
    rewrote_to="the same rolling window with center=False (trailing)",
    children=_FN,
)


# --------------------------------------------------------------------------- #
# Aggregates broadcast back over rows
# --------------------------------------------------------------------------- #
def _subtree_is_safe(node: Any, ctx: Context) -> bool:
    """True when ``node`` and everything under it already classify SAFE.

    Used by the aggregates whose operand the walker cannot reach (see the
    module docstring): they vouch for their own operand rather than rewrite one
    that nothing has audited.
    """
    rule = rule_for(node)
    if rule is None:
        return False
    payload = payload_of(node)
    if rule.classify(payload, ctx) is not Classification.SAFE:
        return False
    return all(
        _subtree_is_safe(child, ctx)
        for _path, child in child_nodes(payload, rule.children)
    )


def _classify_agg(
    variant: str, *, vouch_for_operand: bool = False
) -> Callable[[Any, Context], Classification]:
    def classify(payload: Any, ctx: Context) -> Classification:
        child = _agg_input(payload, variant)
        if child is None:
            return Classification.REFUSE
        if _is_constant(child):
            return Classification.SAFE
        if vouch_for_operand and not _subtree_is_safe(child, ctx):
            return Classification.REFUSE
        return Classification.REWRITE

    return classify


def _rewrite_agg(variant: str, template: str) -> Callable[[Any, Context], Any]:
    def rewrite(payload: Any, _ctx: Context) -> Any:
        child = _agg_input(payload, variant)
        return _expanding(template, child)

    return rewrite


def _rewrite_agg_moment(variant: str, template: str) -> Callable[[Any, Context], Any]:
    def rewrite(payload: Any, _ctx: Context) -> Any:
        child = _agg_input(payload, variant)
        return _expanding(template, child, _agg_ddof(payload, variant))

    return rewrite


def _rewrite_agg_count(payload: Any, _ctx: Context) -> Any:
    child = _agg_input(payload, "Count")
    body = _get(payload, "Count")
    include_nulls = bool(body.get("include_nulls")) if isinstance(body, dict) else False
    return _expanding("cum_count_all" if include_nulls else "cum_count", child)


def _agg_rule(
    variant: str,
    reason: str,
    rewrite: Callable[[Any, Context], Any],
    rewrote_to: str,
    *,
    nested_operand: bool = False,
) -> Rule:
    """One aggregate rule.

    ``nested_operand`` marks the variants whose payload buries the operand one
    level deeper than ``children`` can name (``Min``, ``Max``, ``Count``): they
    declare no children and vouch for the operand in ``classify`` instead.
    """
    return Rule(
        kind=f"Agg.{variant}",
        classify=_classify_agg(variant, vouch_for_operand=nested_operand),
        reason=reason,
        rewrite=rewrite,
        rewrote_to=rewrote_to,
        children=() if nested_operand else (variant,),
    )


_NESTED_OPERAND = (
    " Polars buries this aggregate's operand one level deeper than a `children` "
    "key can name, so the operand is audited by the rule itself and anything "
    "that is not already causal is refused rather than rewritten unchecked."
)

_BROADCAST = (
    "a whole-column {name} is reduced to one number and broadcast back onto "
    "every row, so rows before t are handed a value computed from rows after "
    "t. The exact point-in-time equivalent is the expanding {name}: {how}."
)

_AGGREGATES: list[Rule] = [
    _agg_rule(
        "Sum",
        _BROADCAST.format(name="sum", how="`cum_sum()`"),
        _rewrite_agg("Sum", "cum_sum"),
        "cum_sum() (expanding sum)",
    ),
    _agg_rule(
        "Min",
        _BROADCAST.format(name="minimum", how="`cum_min()`") + _NESTED_OPERAND,
        _rewrite_agg("Min", "cum_min"),
        "cum_min() (expanding minimum)",
        nested_operand=True,
    ),
    _agg_rule(
        "Max",
        _BROADCAST.format(name="maximum", how="`cum_max()`") + _NESTED_OPERAND,
        _rewrite_agg("Max", "cum_max"),
        "cum_max() (expanding maximum)",
        nested_operand=True,
    ),
    _agg_rule(
        "Count",
        "a whole-column count depends on how many rows exist in total, so the "
        "value at t changes when rows after t are added or removed -- the "
        "prefix-invariance failure that a length-dependent quantity always is. "
        "The point-in-time equivalent is the expanding count, `cum_count()`."
        + _NESTED_OPERAND,
        _rewrite_agg_count,
        "cum_count() (expanding count)",
        nested_operand=True,
    ),
    _agg_rule(
        "Mean",
        _BROADCAST.format(name="mean", how="`cum_sum() / cum_count()`"),
        _rewrite_agg("Mean", "mean"),
        "cum_sum() / cum_count() (expanding mean)",
    ),
    _agg_rule(
        "Std",
        _BROADCAST.format(
            name="standard deviation",
            how="the expanding standard deviation built from `cum_sum()` of x "
            "and of x**2",
        ),
        _rewrite_agg_moment("Std", "std"),
        "expanding standard deviation",
    ),
    _agg_rule(
        "Var",
        _BROADCAST.format(
            name="variance",
            how="the expanding variance built from `cum_sum()` of x and of x**2",
        ),
        _rewrite_agg_moment("Var", "var"),
        "expanding variance",
    ),
]


# --------------------------------------------------------------------------- #
# .over(...)
# --------------------------------------------------------------------------- #
def _classify_over(payload: Any, ctx: Context) -> Classification:
    if _get(payload, "order_by") is not None:
        return Classification.SAFE
    if _partitions_by_time(payload, ctx):
        # Invariant 5 is about order: an unordered window is only correct if
        # the frame happens to be sorted. Partitioned by time, that worry is
        # vacuous -- every row of the window carries the same timestamp, so no
        # row order within it can expose a later one. Injecting
        # `order_by=<time>` here would sort by a column that is constant inside
        # the partition, i.e. report a repair that changes nothing.
        return Classification.SAFE
    if ctx.time is None:
        return Classification.REFUSE
    return Classification.REWRITE


def _rewrite_over(payload: Any, ctx: Context) -> Any:
    assert ctx.time is not None  # guaranteed by _classify_over
    replacement = copy.deepcopy(payload)
    replacement["order_by"] = copy.deepcopy(_order_by_payload(ctx.time))
    return {"Over": replacement}


_OVER_RULE = Rule(
    kind="Over",
    classify=_classify_over,
    reason=(
        "`.over(...)` without `order_by` evaluates the window in whatever order "
        "the rows happen to sit in, so a window function inside it can read "
        "rows that are later in time; it is only correct if the frame is "
        "already sorted, and sortedness is not something the compiler may "
        "assume. Pass `order_by=<time column>` -- injected automatically when "
        "`Context.time` is set, and refused when it is not."
    ),
    rewrite=_rewrite_over,
    rewrote_to="the same window with order_by=<Context.time>",
    # `order_by` is deliberately not walked: it is a sort key, not a value the
    # feature is built from, and the repair path writes it rather than reads it.
    children=("function", "partition_by"),
)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def _classify_anonymous(_payload: Any, ctx: Context) -> Classification:
    """SAFE only when the caller has declared trust; REFUSE otherwise.

    There is nothing in the payload to decide on. ``function`` is a cloudpickle
    blob -- no name, no ``fmt_str``, and the only text inside it is the pickled
    ``__qualname__`` and source path of whatever Python object was handed to
    ``map_batches``. Matching on that would be fail-open twice over: it is
    forgeable (any lambda can live in a scope called ``frac_diff``) and it is
    unstable (the bytes change with the closure's captured values, the Python
    version and the cloudpickle version). So this rule reads the *context*,
    where an explicit, registry-checked declaration from the caller is the only
    thing that can put trust.
    """
    return Classification.SAFE if trusted_names(ctx) else Classification.REFUSE


_ANONYMOUS_RULE = Rule(
    kind="AnonymousFunction",
    classify=_classify_anonymous,
    reason=(
        "`map_batches` / `map_elements` run arbitrary Python over the whole "
        "column, and the serialised tree keeps the callable as an opaque "
        "pickle -- no name, nothing that says which operator it is -- so the "
        "compiler cannot prove the output at t ignores rows after t, and "
        "cannot recognise the operator either. Express it with native, "
        "trailing Polars operators; or, if it is a Panelary operator "
        "registered with leakage_safe=True and safe_scope='rowwise', name it "
        "in `audit(..., trust=(...))` to vouch for it explicitly. Trust is "
        "never inferred from the node."
    ),
    children=_FN,
)


_REFUSALS: list[Rule] = [
    _ANONYMOUS_RULE,
    _refuse(
        "Function.Interpolate",
        "interpolation fills a gap by drawing a line between the observation "
        "before it and the observation after it, so every filled value carries "
        "the future one. Use `fill_null(strategy='forward')`.",
        _FN,
    ),
    _refuse(
        "Function.Rank",
        "ranking a whole column places row t relative to every other row, "
        "including rows after t. Rank cross-sectionally within a date "
        "(`.rank().over(<time>)`), or over a trailing window.",
        _FN,
    ),
    _refuse(
        "Function.Reverse",
        "reversing a column puts the last observation first; every subsequent "
        "operation then reads the future as if it were the past. There is no "
        "causal equivalent.",
        _FN,
    ),
    _refuse(
        "Function.Quantile",
        "a whole-column quantile is computed from every row, including rows "
        "after t, and is then broadcast back over all of them. Polars has no "
        "expanding-quantile expression, so this cannot be repaired even with "
        "`Context.allow_approximate`: use `rolling_quantile(q, window_size=k)` "
        "with an explicit trailing window, or take the quantile per date with "
        "`.over(<time>)`.",
        _FN,
    ),
    _refuse(
        "Agg.Median",
        "a whole-column median is computed from every row, including rows after "
        "t. Polars has no expanding-median expression, so this cannot be "
        "repaired even with `Context.allow_approximate`: use "
        "`rolling_median(window_size=k)` with an explicit trailing window, or "
        "take the median per date with `.over(<time>)`.",
        ("Median",),
    ),
    _refuse(
        "Agg.Last",
        "`last()` broadcasts the final observation of the column onto every "
        "row, which is the future by construction. Its point-in-time reading "
        "-- the latest value as of t -- is just the column itself, so the "
        "compiler refuses rather than silently turning an aggregate into the "
        "identity: write `pl.col(x)` (or "
        "`pl.col(x).fill_null(strategy='forward')`) if that is what you meant.",
        ("Last",),
    ),
    _refuse(
        "Agg.NUnique",
        "the number of distinct values in the whole column counts values that "
        "first appear after t. Count distinct values per date with "
        "`.over(<time>)`, or over a trailing window.",
        ("NUnique",),
    ),
    _refuse(
        "Sort",
        "sorting a column detaches values from their rows, so a value observed "
        "in the future can land on a row in the past. Sorting is a frame "
        "operation; do it on the frame, not inside a feature expression.",
        ("expr",),
    ),
]


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _cross_sectionally_safe(rule: Rule) -> Rule:
    """Clear ``rule`` inside a time-partitioned window, if it is clearable.

    Applied to exactly :data:`CROSS_SECTIONAL_SAFE_KINDS`, and only when
    :func:`child_context` has marked the subtree, so a rule's own verdict is
    unchanged everywhere else -- a bare ``rank()`` is still refused, and
    ``rank().over(entity)`` still is too.

    The operand still has to be audited. Most of these kinds declare
    ``children``, so the walker has already audited theirs by the time this
    runs. ``Agg.Min`` / ``Agg.Max`` declare none and vouch for their own
    operand in ``classify`` (see the module docstring), so for those the
    original classification is still consulted: a REWRITE from it means "the
    operand is causal", which inside a cross-section needs no rewrite, and a
    REFUSE means the operand could not be vouched for and stays refused.
    """
    if rule.kind not in CROSS_SECTIONAL_SAFE_KINDS:
        return rule
    inner = rule.classify
    audits_its_own_operand = not rule.children

    def classify(payload: Any, ctx: Context) -> Classification:
        if not in_cross_section(ctx):
            return inner(payload, ctx)
        if not audits_its_own_operand:
            return Classification.SAFE
        verdict = inner(payload, ctx)
        return Classification.SAFE if verdict is Classification.REWRITE else verdict

    return replace(rule, classify=classify)


def _table(*groups: list[Rule] | Rule) -> dict[str, Rule]:
    table: dict[str, Rule] = {}
    for group in groups:
        rules = group if isinstance(group, list) else [group]
        for rule in rules:
            if rule.kind in table:  # pragma: no cover - guards authoring slips
                raise ValueError(f"duplicate rule for {rule.kind!r}")
            table[rule.kind] = _cross_sectionally_safe(rule)
    return table


RULES: dict[str, Rule] = _table(
    _LEAVES,
    _BOOLEAN_RULE,
    _CUMULATIVE,
    _OFFSET_RULES,
    _FILL_RULE,
    _ROLLING_RULE,
    _AGGREGATES,
    _OVER_RULE,
    _REFUSALS,
)
