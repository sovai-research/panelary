"""The point-in-time compiler: walk a serialised Polars expression and fix it.

Two entry points, both operating on one :class:`polars.Expr`:

``audit``
    Never raises because of a leak. It returns a :class:`CompileResult` holding
    one :class:`Finding` per non-trivial decision, so a caller can report every
    problem in an expression rather than only the first.
``causalize``
    ``audit`` plus :meth:`CompileResult.raise_if_refused`, i.e. the strict form
    that hands back a point-in-time expression or explains why it cannot.

How it works
------------
``Expr.meta.serialize(format="json")`` renders the whole tree, and
``pl.Expr.deserialize`` reads a mutated one back. The walker therefore works on
plain JSON: it computes each node's kind with :func:`~panelary.leakage._types.node_kind`,
looks the kind up in ``_rules.RULES``, recurses **only** into the payload keys
the matching :class:`~panelary.leakage._types.Rule` names in ``children`` (so an
options dict can never be mistaken for a node), and applies rewrites bottom-up
so that a parent rule classifies a payload whose children are already causal.

Descending is where the walker adds what a rule cannot see from one node: it
asks ``_rules.child_context`` for the context each child subtree is evaluated
in, which is how ``rank()`` can be refused on its own and accepted inside
``.over(<time>)``, where every row of the window shares one timestamp.

Trust
-----
``map_batches`` serialises to an ``AnonymousFunction`` holding a cloudpickle
blob: there is nothing in the tree that identifies which operator it is, so the
compiler never infers trust from a node. A caller may *declare* it --
``audit(..., trust=("panel.frac_diff",))`` -- and the name is then checked
against the registry (``leakage_safe`` **and** ``safe_scope="rowwise"``) before
any opaque node is cleared. Every node cleared that way is recorded as a
:class:`~panelary.leakage._types.Finding`, so a trust-based acceptance appears
in the audit rather than passing in silence.

Fail closed
-----------
The serialised format is not a stable Polars API, so every branch on which the
compiler cannot *prove* safety produces :data:`Classification.REFUSE`:

* a kind with no entry in the rule table;
* a value in a declared child slot that is not an expression node;
* a rule whose ``classify``/``rewrite`` raises, or returns a non-classification;
* a rule that classifies ``REWRITE`` but supplies no ``rewrite`` callable.

There is no path on which an unrecognised node is reported as ``SAFE``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import polars as pl

from ._types import (
    POLARS_TREE_FORMAT_TESTED,
    Classification,
    CompileResult,
    Context,
    FeatureSetAudit,
    Finding,
    Rule,
    Verdict,
    node_kind,
    render_expr,
)

__all__ = ["audit", "audit_features", "causalize"]

#: Reported as the ``kind`` of a finding when a slot a rule declared to hold a
#: child expression holds something that is not an expression node at all.
NOT_A_NODE = "<not-an-expression-node>"


def _rule_table() -> Mapping[str, Rule]:
    """The live rule table, imported lazily.

    Indirection on purpose: it keeps the import out of module scope (so the
    walker can be exercised against a fixture table) and gives tests a single
    seam to monkeypatch.
    """
    from ._rules import RULES

    return RULES


def _child_context(kind: str, slot: Any, payload: Any, ctx: Context) -> Context:
    """The context the subtree in ``payload[slot]`` is evaluated in.

    Scoping is a walker concern -- it is about where a node sits, not what it
    holds -- but *which* nesting means what is rule-table policy, so the answer
    comes from ``_rules``. Lazily imported for the same reason as the table:
    the walker must stay usable against a fixture table, and the import is a
    ``sys.modules`` lookup after the first call.
    """
    from ._rules import child_context

    return child_context(kind, slot, payload, ctx)


def _context(
    *,
    time: str | None,
    entity: str | None,
    allow_approximate: bool,
    trust: str | Iterable[str] | None,
) -> Context:
    """Build the starting context, resolving any declared trust up front.

    Validation belongs here rather than in the rule: a caller who names an
    operator that is not registered, or whose ``safe_scope`` is not
    ``"rowwise"``, gets a ``ValueError`` naming the problem instead of a
    refusal buried in a finding that looks like the default refusal.
    """
    from ._rules import build_context

    return build_context(
        time=time, entity=entity, allow_approximate=allow_approximate, trust=trust
    )


def _format_error(action: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"could not {action} the Polars expression tree with polars "
        f"{pl.__version__}: {type(exc).__name__}: {exc}. The serialised "
        "expression format is not a stable Polars API; this compiler is "
        f"tested against POLARS_TREE_FORMAT_TESTED="
        f"{POLARS_TREE_FORMAT_TESTED}. Pin a tested Polars version or extend "
        "panelary.leakage._rules; the expression has NOT been made "
        "point-in-time."
    )


def _to_tree(expr: pl.Expr) -> Any:
    try:
        return json.loads(expr.meta.serialize(format="json"))
    except Exception as exc:  # pragma: no cover - depends on Polars internals
        raise _format_error("serialise", exc) from exc


def _from_tree(tree: Any) -> pl.Expr:
    try:
        return pl.Expr.deserialize(io.StringIO(json.dumps(tree)), format="json")
    except Exception as exc:
        raise _format_error("deserialise", exc) from exc


def _brief(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _refuse(
    kind: str, reason: str, path: tuple[str | int, ...], findings: list[Finding]
) -> None:
    findings.append(
        Finding(
            kind=kind,
            classification=Classification.REFUSE,
            reason=reason,
            path=path,
        )
    )


def _walk_slot(
    value: Any,
    rules: Mapping[str, Rule],
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> Any:
    """Walk one declared child slot, which holds a node or a list of nodes."""
    if value is None:
        return None
    if isinstance(value, list):
        out = list(value)
        for i, item in enumerate(out):
            # Heterogeneous lists are common (``Alias`` is ``[node, name]``,
            # ``Agg.Std`` is ``[node, ddof]``). Only real nodes are walked;
            # anything single-key that merely looks like one still reaches
            # ``_walk`` and is refused there, so this cannot open a hole.
            if node_kind(item) is not None:
                out[i] = _walk(item, rules, ctx, (*path, i), findings)
        return out
    return _walk(value, rules, ctx, path, findings)


def _walk_children(
    payload: Any,
    rule: Rule,
    rules: Mapping[str, Rule],
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> Any:
    """Return ``payload`` with every declared child slot walked (bottom-up)."""
    if not rule.children:
        return payload
    if isinstance(payload, Mapping):
        new_map = dict(payload)
        for key in rule.children:
            if key not in new_map:
                continue
            slot_ctx = _child_context(rule.kind, key, payload, ctx)
            new_map[key] = _walk_slot(
                new_map[key], rules, slot_ctx, (*path, key), findings
            )
        return new_map
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        new_seq = list(payload)
        for key in rule.children:
            if not (isinstance(key, str) and key.isdigit()):
                _refuse(
                    rule.kind,
                    f"rule declares child {key!r} but the payload is a sequence; "
                    "sequence payloads take positional child keys such as '0'",
                    path,
                    findings,
                )
                continue
            index = int(key)
            if index >= len(new_seq):
                _refuse(
                    rule.kind,
                    f"rule declares child at position {index} but the payload "
                    f"has only {len(new_seq)} element(s)",
                    path,
                    findings,
                )
                continue
            slot_ctx = _child_context(rule.kind, key, payload, ctx)
            new_seq[index] = _walk_slot(
                new_seq[index], rules, slot_ctx, (*path, index), findings
            )
        return new_seq
    _refuse(
        rule.kind,
        f"rule declares children {rule.children!r} but the payload is "
        f"{_brief(payload)}, which holds none",
        path,
        findings,
    )
    return payload


def _record_trusted_leaf(
    kind: str,
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> None:
    """Record that an opaque node was cleared by the caller's declared trust.

    Silence would make an audit a rubber stamp exactly where it matters most:
    the one node the compiler did not, and could not, verify. There is no
    fourth :class:`Classification`, so the finding is filed ``SAFE`` -- it
    leaves ``verdict`` alone and stays out of ``refused`` / ``rewritten``, but
    it is there in ``findings`` for anyone reading the report.
    """
    names = getattr(ctx, "trusted", None)
    if not names:
        return  # the ordinary path: no trust declared, so nothing to record
    from ._rules import OPAQUE_KINDS

    if kind not in OPAQUE_KINDS:
        return
    findings.append(
        Finding(
            kind=kind,
            classification=Classification.SAFE,
            reason=(
                "accepted on the caller's declared trust in "
                f"{', '.join(sorted(names))} -- each registered with "
                "leakage_safe=True and safe_scope='rowwise'. The compiler did "
                "NOT verify that this node is one of them: a serialised "
                "map_batches carries only an opaque pickle, so trust applies "
                "to the whole audited expression, not to this node in "
                "particular."
            ),
            path=path,
        )
    )


def _walk(
    node: Any,
    rules: Mapping[str, Rule],
    ctx: Context,
    path: tuple[str | int, ...],
    findings: list[Finding],
) -> Any:
    """Classify and possibly rewrite one node, children first."""
    kind = node_kind(node)
    if kind is None:
        _refuse(
            NOT_A_NODE,
            f"expected an expression node here, found {_brief(node)}",
            path,
            findings,
        )
        return node

    rule = rules.get(kind)
    if rule is None:
        # Invariant 1. Unknown means unproven, and unproven means refused.
        _refuse(
            kind,
            f"no rule for node kind {kind!r}, so it cannot be proven "
            "point-in-time (the compiler fails closed rather than guess)",
            path,
            findings,
        )
        return node

    key = next(iter(node))
    payload = _walk_children(node[key], rule, rules, ctx, path, findings)

    try:
        classification = rule.classify(payload, ctx)
    except Exception as exc:
        _refuse(
            kind,
            f"rule for {kind!r} raised while classifying: {type(exc).__name__}: {exc}",
            path,
            findings,
        )
        return {key: payload}

    if not isinstance(classification, Classification):
        _refuse(
            kind,
            f"rule for {kind!r} returned {_brief(classification)} instead of a "
            "Classification",
            path,
            findings,
        )
        return {key: payload}

    if classification is Classification.SAFE:
        _record_trusted_leaf(kind, ctx, path, findings)
        return {key: payload}

    if classification is Classification.REFUSE:
        _refuse(kind, rule.reason, path, findings)
        return {key: payload}

    if rule.rewrite is None:
        _refuse(
            kind,
            f"rule for {kind!r} classified REWRITE but supplies no rewrite "
            f"({rule.reason})",
            path,
            findings,
        )
        return {key: payload}

    try:
        rewritten = rule.rewrite(payload, ctx)
    except Exception as exc:
        _refuse(
            kind,
            f"rule for {kind!r} raised while rewriting: {type(exc).__name__}: {exc}",
            path,
            findings,
        )
        return {key: payload}

    findings.append(
        Finding(
            kind=kind,
            classification=Classification.REWRITE,
            reason=rule.reason,
            path=path,
            rewrote_to=rule.rewrote_to,
        )
    )
    # `rewrite` returns a whole NODE, not a payload: a rewrite is allowed to
    # change the node's kind, which is what makes the valuable ones possible --
    # a whole-column `Agg.Mean` becomes an expanding `BinaryExpr`, and no
    # payload swap under the original discriminator could express that. Guard
    # the contract here rather than letting a malformed tree reach Polars,
    # where it surfaces as an opaque deserialisation error.
    if node_kind(rewritten) is None:
        _refuse(
            kind,
            f"rule for {kind!r} returned {rewritten!r}, which is not a node; "
            f"`Rule.rewrite` must return a complete node such as "
            f'{{"Column": "x"}}, not a bare payload',
            path,
            findings,
        )
        return {key: payload}
    return rewritten


def audit(
    expr: pl.Expr,
    *,
    time: str | None = None,
    entity: str | None = None,
    allow_approximate: bool = False,
    trust: str | Iterable[str] | None = (),
) -> CompileResult:
    """Audit one expression for look-ahead, rewriting what can be rewritten.

    This never raises because an expression leaks: it reports. Every node the
    compiler could not prove safe produces a :class:`Finding` carrying the
    qualified node kind, the reason, and the ``path`` at which it sits, so a
    single call reports every problem rather than the first.

    Parameters
    ----------
    expr : polars.Expr
        The expression to audit.
    time, entity : str, optional
        Panel key columns. ``time`` is what lets the compiler repair an
        ``.over(entity)`` that carries no ``order_by``; without it such a node
        is refused rather than assumed sorted. ``time`` is also what makes a
        cross-sectional feature recognisable: inside ``.over(<time>)`` every
        row of the window shares a timestamp, so ``rank()`` and the other
        order-invariant statistics there are causal rather than refused.
    allow_approximate : bool, default=False
        Permit rewrites that are causal but not numerically identical to the
        original.
    trust : str or iterable of str, optional
        Registered operator names -- bare (``"frac_diff"``) or qualified
        (``"panel.frac_diff"``) -- that this expression is built from. A
        ``map_batches`` / ``map_elements`` node serialises to an opaque pickle,
        so the compiler refuses it by default and **cannot tell one from
        another**; naming an operator here is the caller declaring which. Each
        name must be registered with ``leakage_safe=True`` *and*
        ``safe_scope="rowwise"``, or the call raises. Because the node cannot
        be identified, trust clears *every* opaque node in this expression, and
        each is recorded as a ``SAFE`` finding saying so. Declare it only for
        expressions you built from operators you named.

    Returns
    -------
    CompileResult
        ``verdict`` is ``REFUSED`` if any finding refuses, else ``REWRITTEN``
        if any rewrote, else ``SAFE``. ``expr`` holds the compiled expression,
        and is ``None`` when refused. Only ``REWRITE`` and ``REFUSE``
        decisions affect the verdict; a node the table proves safe is silent,
        the one exception being an opaque node cleared by ``trust``.

    Raises
    ------
    TypeError
        If ``expr`` is not a :class:`polars.Expr`, or ``trust`` does not hold
        strings.
    ValueError
        If a name in ``trust`` is not registered, is registered
        ``leakage_safe=False``, or does not have ``safe_scope="rowwise"``.
    RuntimeError
        If the serialised-tree round trip itself fails, which means this
        Polars version's format is not one the compiler was tested against.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.leakage import audit
    >>> audit(pl.col("x").map_batches(lambda s: s)).verdict.value
    'refused'
    """
    if not isinstance(expr, pl.Expr):
        raise TypeError(f"audit() expects a polars.Expr, got {type(expr).__name__}")

    ctx = _context(
        time=time, entity=entity, allow_approximate=allow_approximate, trust=trust
    )
    findings: list[Finding] = []
    tree = _walk(_to_tree(expr), _rule_table(), ctx, (), findings)
    frozen = tuple(findings)

    # ``source`` / ``context`` ride along so the result can be serialised as
    # report evidence (``CompileResult.to_json``) without the caller having to
    # remember what was audited, or against which keys.
    source = render_expr(expr)
    if any(f.classification is Classification.REFUSE for f in frozen):
        return CompileResult(
            verdict=Verdict.REFUSED,
            findings=frozen,
            expr=None,
            source=source,
            context=ctx,
        )

    verdict = (
        Verdict.REWRITTEN
        if any(f.classification is Classification.REWRITE for f in frozen)
        else Verdict.SAFE
    )
    # Deserialise even when nothing changed: it costs microseconds and keeps
    # the round trip on the tested path for every call, so format drift shows
    # up as a loud RuntimeError instead of a silently unrewritten expression.
    return CompileResult(
        verdict=verdict,
        findings=frozen,
        expr=_from_tree(tree),
        source=source,
        context=ctx,
    )


def _named_features(
    features: Mapping[str, pl.Expr] | Iterable[pl.Expr],
) -> list[tuple[str, pl.Expr]]:
    """Normalise a feature set to ``[(name, expr), ...]``, in the given order."""
    if isinstance(features, pl.Expr):
        raise TypeError(
            "audit_features() expects a feature *set* -- a mapping of name to "
            "polars.Expr, or an iterable of named expressions -- not a single "
            "expression; use audit() for one expression."
        )
    pairs: list[tuple[str, pl.Expr]] = []
    if isinstance(features, Mapping):
        for name, expr in features.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"feature names must be str, got {type(name).__name__!r} "
                    f"({name!r})."
                )
            pairs.append((name, expr))
    else:
        for i, expr in enumerate(features):
            if not isinstance(expr, pl.Expr):
                raise TypeError(
                    f"audit_features() item {i} is {type(expr).__name__!r}, "
                    "not a polars.Expr."
                )
            try:
                name = expr.meta.output_name()
            except Exception as exc:
                raise ValueError(
                    f"audit_features() item {i} ({render_expr(expr)}) has no "
                    "output name to key the report by; pass a mapping "
                    "{name: expr}, or give it one with .alias(...)."
                ) from exc
            pairs.append((name, expr))
    seen: set[str] = set()
    for name, expr in pairs:
        if not isinstance(expr, pl.Expr):
            raise TypeError(
                f"feature {name!r} is {type(expr).__name__!r}, not a polars.Expr."
            )
        if name in seen:
            raise ValueError(
                f"feature name {name!r} appears twice; a report is keyed by "
                "feature name, so names must be unique."
            )
        seen.add(name)
    return pairs


def audit_features(
    features: Mapping[str, pl.Expr] | Iterable[pl.Expr],
    *,
    time: str | None = None,
    entity: str | None = None,
    allow_approximate: bool = False,
    trust: str | Iterable[str] | None = (),
) -> FeatureSetAudit:
    """Audit a whole named feature set in one call.

    :func:`audit` answers for one expression; a feature set is dozens. This
    runs :func:`audit` over every feature with the **same** panel keys, options
    and declared trust, and aggregates the answers into one
    :class:`~panelary.leakage.FeatureSetAudit` whose verdict is the worst of
    its parts and whose :meth:`~panelary.leakage.FeatureSetAudit.to_json` is a
    single deterministic evidence payload. Like :func:`audit`, it never raises
    because a feature leaks: it reports every feature, not the first refusal.

    Parameters
    ----------
    features : mapping of str to polars.Expr, or iterable of polars.Expr
        The feature set. A mapping is keyed by feature name. An iterable of
        expressions is keyed by each expression's output name
        (``expr.meta.output_name()``), so alias them. Order is preserved in the
        report.
    time, entity : str, optional
        Panel keys, applied to every feature; see :func:`audit`.
    allow_approximate : bool, default=False
        Permit causal-but-not-identical rewrites; see :func:`audit`.
    trust : str or iterable of str, optional
        Registered operators the feature set is built from; see :func:`audit`.
        Validated once, and applied to every feature -- and recorded in every
        result's ``context``, so a trust-based acceptance is visible in the
        report.

    Returns
    -------
    FeatureSetAudit
        One :class:`CompileResult` per feature, in input order.

    Raises
    ------
    TypeError
        If ``features`` is a single expression, or holds a non-expression or a
        non-string name.
    ValueError
        If two features share a name, an unnamed expression is given in an
        iterable, or ``trust`` names an operator that cannot be trusted (see
        :func:`audit`).
    RuntimeError
        If the serialised-tree round trip fails (untested Polars format).

    See Also
    --------
    audit : The single-expression compiler this sweeps.
    panelary.core.pipeline.Pipeline.audit : The same sweep over a pipeline's steps.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.leakage import audit_features
    >>> report = audit_features(
    ...     {
    ...         "lag": pl.col("x").shift(1).over("e", order_by="t"),
    ...         "lead": pl.col("x").shift(-1).over("e", order_by="t"),
    ...     },
    ...     time="t",
    ...     entity="e",
    ... )
    >>> report.verdict.value, report.refused
    ('refused', ('lead',))
    >>> report.to_json()[:40]
    '{"counts":{"refused":1,"rewritten":0,"sa'
    """
    pairs = _named_features(features)
    # Resolve trust once, up front: a bad name is a ValueError before any work,
    # not a repeated error per feature.
    _context(time=time, entity=entity, allow_approximate=allow_approximate, trust=trust)
    return FeatureSetAudit(
        results=tuple(
            (
                name,
                audit(
                    expr,
                    time=time,
                    entity=entity,
                    allow_approximate=allow_approximate,
                    trust=trust,
                ),
            )
            for name, expr in pairs
        )
    )


def causalize(
    expr: pl.Expr,
    *,
    time: str | None = None,
    entity: str | None = None,
    allow_approximate: bool = False,
    trust: str | Iterable[str] | None = (),
) -> pl.Expr:
    """Compile one expression to its point-in-time equivalent, or refuse.

    The strict form of :func:`audit`.

    Parameters
    ----------
    expr : polars.Expr
        The expression to compile.
    time, entity : str, optional
        Panel key columns; see :func:`audit`.
    allow_approximate : bool, default=False
        Permit rewrites that are causal but not numerically identical.
    trust : str or iterable of str, optional
        Registered operators this expression is built from; see :func:`audit`
        for what declaring trust does and does not prove.

    Returns
    -------
    polars.Expr
        An expression whose value at ``t`` cannot depend on data after ``t``.

    Raises
    ------
    LeakageRefused
        If any node leaks with no causal equivalent. The exception carries the
        full :class:`CompileResult` on ``.result``.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.leakage import causalize
    >>> expr = causalize(pl.col("x").shift(1))
    """
    result = audit(
        expr,
        time=time,
        entity=entity,
        allow_approximate=allow_approximate,
        trust=trust,
    )
    compiled: pl.Expr = result.raise_if_refused()
    return compiled
