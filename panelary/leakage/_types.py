"""Shared vocabulary for the point-in-time compiler and the leakage metric.

This module is the contract every other module in :mod:`panelary.leakage`
codes against. It holds no logic beyond :func:`node_kind`: the rule table
lives in ``_rules.py``, the walker in ``_compile.py``, the metric in
``_borrowed.py``.

The compiler works on the **serialised Polars expression tree**
(``Expr.meta.serialize(format="json")``), rewrites it as JSON, and reads it
back with ``pl.Expr.deserialize``. That round trip is what makes a rewrite
possible at all, and it is the one external dependency of this subpackage
that Polars does not promise to keep stable -- see
:data:`POLARS_TREE_FORMAT_TESTED`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Classification",
    "CompileResult",
    "Context",
    "FeatureSetAudit",
    "Finding",
    "LeakageRefused",
    "POLARS_TREE_FORMAT_TESTED",
    "Rule",
    "Verdict",
    "canonical_json",
    "node_kind",
    "produced_by",
    "render_expr",
]

#: Polars minor versions whose serialised expression format this rule table has
#: been checked against. The format is NOT a stable public API, so the compiler
#: fails closed on anything it does not recognise and
#: ``tests/test_leakage_rules.py`` pins the shapes with golden trees.
POLARS_TREE_FORMAT_TESTED: tuple[str, ...] = ("1.44",)


# --------------------------------------------------------------------------- #
# Report serialisation -- the shared convention for machine-readable evidence
# --------------------------------------------------------------------------- #
#: Matches a CPython memory address (``0x10bb45c50``) inside a ``repr``. An
#: address differs on every run, so it must never reach serialised evidence.
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def produced_by(fn: str) -> str:
    """The ``produced_by`` tag for serialised evidence: ``"<fn>@<version>"``.

    ``fn`` is the fully qualified producer, e.g. ``"panelary.leakage.audit"``;
    the version is ``panelary.__version__``, read at call time (the package is
    fully imported by then, and importing it here at module scope would be a
    cycle). A reader of a report uses this tag to know what to re-run.

    >>> produced_by("panelary.leakage.audit").startswith("panelary.leakage.audit@")
    True
    """
    import panelary

    return f"{fn}@{panelary.__version__}"


def canonical_json(obj: Any, *, indent: int | None = None) -> str:
    """Serialise a ``to_dict()`` payload the way every panelary report does.

    With ``indent=None`` (the default) this is
    ``json.dumps(obj, sort_keys=True, separators=(",", ":"))`` -- compact,
    key-sorted and ASCII-escaped, so the same report is the same bytes on every
    run and every machine, which is what lets a consumer content-address it.
    ``indent`` pretty-prints for humans (keys still sorted). ``allow_nan`` is
    off: a report that needs ``NaN`` must say so as ``null``, because bare
    ``NaN`` is not JSON.
    """
    if indent is None:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.dumps(obj, sort_keys=True, indent=indent, allow_nan=False)


def render_expr(expr: Any) -> str | None:
    """A stable, human-readable string for a Polars expression (``None`` -> ``None``).

    ``str(expr)`` is Polars' own display form (``col("x").shift([dyn int:
    1])``). It is stable for a given Polars version and carries no memory
    address; any address that does appear is masked, so the result is
    byte-identical across runs. It is a *display* string, not a round-trippable
    serialisation: an opaque ``map_batches`` renders as ``python_udf()`` and a
    rolling window omits its length. The serialised tree is deliberately not
    used -- for a ``map_batches`` it embeds a pickle, which is neither stable
    nor safe to ship in a report.
    """
    if expr is None:
        return None
    return _ADDRESS.sub("0x?", str(expr))


class Classification(str, Enum):
    """What the rule table says about one node."""

    SAFE = "safe"
    """Output at ``t`` cannot depend on data after ``t``. Leave it alone."""

    REWRITE = "rewrite"
    """Leaks as written, but has an exact point-in-time equivalent."""

    REFUSE = "refuse"
    """Leaks with no causal equivalent, or is opaque. Fail closed."""


class Verdict(str, Enum):
    """The outcome for a whole expression."""

    SAFE = "safe"
    REWRITTEN = "rewritten"
    REFUSED = "refused"


class LeakageRefused(Exception):
    """Raised when an expression cannot be compiled to a point-in-time form.

    Carries the :class:`CompileResult` so callers can inspect every finding
    rather than just the message.
    """

    def __init__(self, message: str, result: CompileResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class Context:
    """Panel keys and options the rules need in order to decide.

    Parameters
    ----------
    time, entity : str, optional
        Panel key columns. ``time`` is required to repair an ``.over(entity)``
        that carries no ``order_by``: without it the compiler cannot know which
        order "within entity" means, and must refuse instead of guess.
    allow_approximate : bool, default=False
        Permit rewrites that are causal but not numerically identical to the
        leaky original (an expanding quantile, say). Off by default so that a
        rewrite never silently changes results.
    """

    time: str | None = None
    entity: str | None = None
    allow_approximate: bool = False


@dataclass(frozen=True)
class Finding:
    """One decision the compiler made, at one position in the tree."""

    kind: str
    """Qualified node kind, as returned by :func:`node_kind`."""

    classification: Classification
    reason: str
    """Why, in a sentence a user can act on."""

    path: tuple[str | int, ...] = ()
    """Position in the serialised tree, for pointing at the offending node."""

    rewrote_to: str | None = None
    """Short description of the replacement, when ``REWRITE``."""

    @property
    def locator(self) -> str:
        """``path`` as a dotted string (``"<root>"`` for the root node).

        The natural ``locator`` for report evidence: it names the node inside
        the serialised expression tree, so a reader can find it again.
        """
        return ".".join(str(p) for p in self.path) or "<root>"

    def to_dict(self) -> dict[str, Any]:
        """This finding as a JSON-safe plain dict.

        Keys: ``kind``, ``classification`` (the enum *value*), ``reason``,
        ``path`` (a list of str/int), ``locator`` (see :attr:`locator`) and
        ``rewrote_to`` (``None`` unless ``REWRITE``). A finding carries no
        ``schema`` / ``produced_by`` of its own: it is always nested in the
        :class:`CompileResult` that produced it.
        """
        return {
            "kind": self.kind,
            "classification": self.classification.value,
            "reason": self.reason,
            "path": [p if isinstance(p, int) else str(p) for p in self.path],
            "locator": self.locator,
            "rewrote_to": self.rewrote_to,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict`, serialised by :func:`canonical_json`."""
        return canonical_json(self.to_dict(), indent=indent)

    def __str__(self) -> str:  # pragma: no cover - display only
        tail = f" -> {self.rewrote_to}" if self.rewrote_to else ""
        return (
            f"{self.locator}: {self.kind} [{self.classification.value}] "
            f"{self.reason}{tail}"
        )


@dataclass(frozen=True)
class CompileResult:
    """The result of auditing or compiling one expression."""

    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    expr: Any | None = None
    """The rewritten :class:`polars.Expr`; ``None`` when refused."""

    source: str | None = field(default=None, compare=False)
    """The audited expression as given, rendered by :func:`render_expr`.

    Set by :func:`~panelary.leakage.audit`; ``None`` on a result built by hand.
    Kept as a string, not the expression, so the result stays serialisable.
    """

    context: Context | None = field(default=None, compare=False, repr=False)
    """The :class:`Context` the rules decided against (keys, options, trust)."""

    #: Serialisation schema tag; bump the suffix on any breaking change to
    #: the shape :meth:`to_dict` returns.
    SCHEMA = "panelary.CompileResult/1"

    @property
    def refused(self) -> tuple[Finding, ...]:
        return tuple(
            f for f in self.findings if f.classification is Classification.REFUSE
        )

    @property
    def rewritten(self) -> tuple[Finding, ...]:
        return tuple(
            f for f in self.findings if f.classification is Classification.REWRITE
        )

    def raise_if_refused(self) -> Any:
        """Return the compiled expression, or raise :class:`LeakageRefused`."""
        if self.verdict is Verdict.REFUSED:
            detail = "\n  ".join(str(f) for f in self.refused)
            raise LeakageRefused(
                f"expression cannot be made point-in-time:\n  {detail}", self
            )
        return self.expr

    def to_dict(self) -> dict[str, Any]:
        """This result as a JSON-safe plain dict -- report evidence.

        Returns
        -------
        dict
            ``schema`` (``"panelary.CompileResult/1"``), ``produced_by``
            (``"panelary.leakage.audit@<version>"``), ``verdict`` (enum
            value), ``source`` (the audited expression, rendered),
            ``expr`` (the compiled expression, rendered; ``None`` when
            refused), ``context`` (``time``, ``entity``,
            ``allow_approximate``, sorted ``trust``; ``None`` if unknown),
            ``counts`` (findings per classification) and ``findings`` (each
            :meth:`Finding.to_dict`, in walk order).

        Notes
        -----
        Nothing in the payload varies between runs: expressions are display
        strings with no memory address, there are no timestamps or ids, and
        findings keep the walker's deterministic order. See
        ``docs/api-reference/leakage.md`` for how the fields map onto an
        assessment report's evidence (``observed`` / ``expected`` /
        ``locator`` / ``produced_by``).
        """
        counts = {c.value: 0 for c in Classification}
        for f in self.findings:
            counts[f.classification.value] += 1
        return {
            "schema": self.SCHEMA,
            "produced_by": produced_by("panelary.leakage.audit"),
            "verdict": self.verdict.value,
            "source": self.source,
            "expr": render_expr(self.expr),
            "context": _context_to_dict(self.context),
            "counts": counts,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict`, serialised by :func:`canonical_json`.

        Byte-identical across runs for the same expression and options when
        ``indent`` is ``None``.
        """
        return canonical_json(self.to_dict(), indent=indent)


def _context_to_dict(ctx: Context | None) -> dict[str, Any] | None:
    """The decision context as JSON: keys, options and any declared trust."""
    if ctx is None:
        return None
    trusted = getattr(ctx, "trusted", None) or ()
    return {
        "time": ctx.time,
        "entity": ctx.entity,
        "allow_approximate": bool(ctx.allow_approximate),
        "trust": sorted(str(t) for t in trusted),
    }


def _worst_verdict(verdicts: Iterator[Verdict]) -> Verdict:
    """``REFUSED`` > ``REWRITTEN`` > ``SAFE``; ``SAFE`` when empty."""
    rank = {Verdict.SAFE: 0, Verdict.REWRITTEN: 1, Verdict.REFUSED: 2}
    return max(verdicts, key=rank.__getitem__, default=Verdict.SAFE)


@dataclass(frozen=True)
class FeatureSetAudit:
    """The audit of a whole named feature set, from one call.

    Returned by :func:`~panelary.leakage.audit_features`. It is the
    feature-set analogue of a :class:`CompileResult`: one result per named
    expression, an aggregate :attr:`verdict`, and a single serialisable payload
    for a report.

    Attributes
    ----------
    results : tuple of (str, CompileResult)
        ``(name, result)`` per feature, in the order the features were given.
    """

    results: tuple[tuple[str, CompileResult], ...] = ()

    SCHEMA = "panelary.FeatureSetAudit/1"

    @property
    def verdict(self) -> Verdict:
        """The most severe feature verdict (``REFUSED`` > ``REWRITTEN`` > ``SAFE``)."""
        return _worst_verdict(r.verdict for _, r in self.results)

    @property
    def ok(self) -> bool:
        """Whether every feature can be made point-in-time (nothing refused)."""
        return self.verdict is not Verdict.REFUSED

    @property
    def names(self) -> tuple[str, ...]:
        """Feature names, in input order."""
        return tuple(name for name, _ in self.results)

    @property
    def refused(self) -> tuple[str, ...]:
        """Names of the features that cannot be made point-in-time."""
        return tuple(n for n, r in self.results if r.verdict is Verdict.REFUSED)

    @property
    def rewritten(self) -> tuple[str, ...]:
        """Names of the features the compiler rewrote."""
        return tuple(n for n, r in self.results if r.verdict is Verdict.REWRITTEN)

    def __getitem__(self, name: str) -> CompileResult:
        for key, result in self.results:
            if key == name:
                return result
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.results)

    def exprs(self) -> dict[str, Any]:
        """The compiled expressions, aliased to their feature names.

        Ready for ``frame.with_columns(*audit.exprs().values())``.

        Raises
        ------
        LeakageRefused
            If any feature was refused; the exception carries a
            :class:`CompileResult` holding every refused finding, each with the
            feature name prepended to its ``path``.
        """
        if not self.ok:
            findings = tuple(
                Finding(
                    kind=f.kind,
                    classification=f.classification,
                    reason=f.reason,
                    path=(name, *f.path),
                    rewrote_to=f.rewrote_to,
                )
                for name, r in self.results
                for f in r.refused
            )
            raise LeakageRefused(
                "feature set cannot be made point-in-time; refused: "
                f"{', '.join(self.refused)}",
                CompileResult(verdict=Verdict.REFUSED, findings=findings),
            )
        out: dict[str, Any] = {}
        for name, r in self.results:
            if r.expr is None:
                raise ValueError(
                    f"feature {name!r} has no compiled expression (a hand-built "
                    "result?); only results returned by audit() carry one."
                )
            out[name] = r.expr.alias(name)
        return out

    def to_dict(self) -> dict[str, Any]:
        """The whole sweep as a JSON-safe plain dict -- report evidence.

        Returns
        -------
        dict
            ``schema`` (``"panelary.FeatureSetAudit/1"``), ``produced_by``
            (``"panelary.leakage.audit_features@<version>"``), ``verdict``,
            ``ok``, ``counts`` (features per verdict), ``refused`` /
            ``rewritten`` (feature names, input order), and ``features``: a
            list of ``{"name": ..., "result": CompileResult.to_dict()}`` in
            input order.
        """
        counts = {v.value: 0 for v in Verdict}
        for _, r in self.results:
            counts[r.verdict.value] += 1
        return {
            "schema": self.SCHEMA,
            "produced_by": produced_by("panelary.leakage.audit_features"),
            "verdict": self.verdict.value,
            "ok": self.ok,
            "counts": counts,
            "refused": list(self.refused),
            "rewritten": list(self.rewritten),
            "features": [
                {"name": name, "result": r.to_dict()} for name, r in self.results
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict`, serialised by :func:`canonical_json`."""
        return canonical_json(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class Rule:
    """How to treat one kind of node.

    ``classify`` and ``rewrite`` both receive the node's *payload* (the value
    under the discriminator key) and the :class:`Context`. ``rewrite`` is only
    called when ``classify`` returned :data:`Classification.REWRITE`, and it
    returns a **complete replacement node**, not a payload -- that is, a
    single-key dict such as ``{"Function": {...}}``.

    Returning a node rather than a payload is deliberate: a rewrite is allowed
    to change the node's *kind*, which is what makes the valuable rewrites
    expressible at all. A whole-column ``Agg.Mean`` becomes an expanding
    ``BinaryExpr`` over ``cum_sum`` and a running count; no payload swap under
    the original discriminator could say that. The walker validates the return
    with :func:`node_kind` and refuses if it is not a node, so a rule that
    returns a bare payload fails loudly instead of producing a double-wrapped
    tree that Polars rejects with an opaque deserialisation error.
    """

    kind: str
    classify: Callable[[Any, Context], Classification]
    reason: str
    rewrite: Callable[[Any, Context], Any] | None = None
    rewrote_to: str | None = None
    children: tuple[str, ...] = field(default_factory=tuple)
    """Payload keys holding child nodes (a value or a list of values)."""


def node_kind(node: Any) -> str | None:
    """Qualified kind of a serialised expression node, or ``None``.

    Polars encodes a node as a single-key dict. Most kinds are the key itself
    (``Column``, ``Over``, ``BinaryExpr``), but ``Function`` and ``Agg`` carry
    the real identity one level down, so those are qualified:

    >>> node_kind({"Column": "x"})
    'Column'
    >>> node_kind({"Agg": {"Mean": {"Column": "x"}}})
    'Agg.Mean'
    >>> node_kind({"Function": {"input": [], "function": "Shift"}})
    'Function.Shift'
    >>> node_kind({"Function": {"input": [], "function": {"RollingExpr": {}}}})
    'Function.RollingExpr'
    >>> node_kind("not a node") is None
    True
    """
    if not isinstance(node, dict) or len(node) != 1:
        return None
    key = next(iter(node))
    payload = node[key]
    if key in ("Function", "AnonymousFunction"):
        fn = payload.get("function") if isinstance(payload, dict) else None
        if isinstance(fn, str):
            return f"{key}.{fn}"
        if isinstance(fn, dict) and len(fn) == 1:
            return f"{key}.{next(iter(fn))}"
        return key
    if key == "Agg" and isinstance(payload, dict) and len(payload) == 1:
        return f"Agg.{next(iter(payload))}"
    return key
