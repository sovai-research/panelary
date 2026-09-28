"""Panelary feature registry — the single source of truth for operators.

This module provides the two pieces of metadata infrastructure that turn a loose
collection of Polars expressions into an auditable, machine-introspectable
*operator catalogue*:

* :class:`FeatureSpec` — a frozen, fully typed description of one feature
  (operator). It carries not just the call signature but the **provenance and
  safety contract**: which expression namespace it lives in, its input/output
  shape, its leakage/panel-safety guarantees, where it came from, and under what
  license. This is how Panelary proves it is an independent, permissively
  licensed implementation rather than a copy of a GPL/AGPL upstream.
* :class:`FeatureRegistry` — a process-wide singleton that collects every
  :class:`FeatureSpec`. It supports lookup, serialization
  (:meth:`FeatureRegistry.to_records`), rendering a machine-readable catalogue
  for the agent layer / ``llms.txt`` (:meth:`FeatureRegistry.to_llms_txt`), and
  a provenance/license :meth:`FeatureRegistry.audit`.

The registry is deliberately decoupled from the implementations: namespaces
register their specs at import time via :func:`register_feature` or
:meth:`FeatureRegistry.register`, but the registry itself imports nothing from
the rest of Panelary. This keeps it cheap, side-effect-free, and safe to import
from anywhere (including the pure-Python, Rust-free build).

Examples
--------
>>> from panelary.registry import registry, register_feature
>>> @register_feature(
...     name="my_op",
...     namespace="panel",
...     input_shape="series",
...     output_shape="series",
...     params={"window": int},
...     tier="A",
...     panel_safe=True,
...     leakage_safe=True,
...     source="Panelary",
...     license="Apache-2.0",
... )
... def _my_op_backend():  # doctest: +SKIP
...     ...
>>> registry.get("my_op").namespace  # doctest: +SKIP
'panel'
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "FeatureSpec",
    "FeatureRegistry",
    "register_feature",
    "registry",
    "PERMISSIVE_LICENSES",
    "VALID_SAFE_SCOPES",
    "VALID_TIERS",
]

#: Licenses considered permissive enough to redistribute without copyleft
#: obligations. Used by :meth:`FeatureRegistry.audit`. Compared case-insensitively
#: against a normalized form of the spec's ``license`` field.
PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "apache 2.0",
        "mit",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "unlicense",
        "public-domain",
        "cc0-1.0",
        "psf",
        "python-2.0",
    }
)

#: The set of tier labels recognised by the registry. Tiers express a rough
#: stability / maturity ranking (A = battle-tested core, D = experimental).
VALID_TIERS: frozenset[str] = frozenset({"A", "B", "C", "D"})

#: Scopes under which :attr:`FeatureSpec.leakage_safe` is being claimed.
#:
#: ``leakage_safe`` on its own is a boolean answer to a question that has no
#: boolean answer: *safe evaluated how?* A ``series -> scalar`` aggregate such
#: as ``ts.absolute_energy`` is perfectly causal as a summary of a completed
#: window, and a look-ahead the moment it is broadcast back over an entity with
#: ``.over(entity)`` -- the value at every row then depends on the entity's
#: whole series, including its future. The same code, two verdicts, so the
#: claim has to name the usage it is claimed under.
#:
#: - ``"rowwise"``  -- safe evaluated at every row: the value at ``t`` uses only
#:   data at ``<= t``. Expanding and trailing-window operators qualify.
#: - ``"window"``   -- safe only as a summary of an already-delimited window
#:   (what :func:`panelary.extract_features` does). Broadcasting the result
#:   per-row is a look-ahead **by construction**, provable from the shape; no
#:   test is required and none should be trusted to find it.
#: - ``"unspecified"`` -- not yet classified. Permitted so third-party specs
#:   keep working, but Panelary's own registry is held to a stricter standard
#:   by ``tests/test_registry_conformance.py``.
VALID_SAFE_SCOPES: frozenset[str] = frozenset({"rowwise", "window", "unspecified"})


def _normalize_license(license_: str) -> str:
    """Normalize a license string for permissive-set membership testing."""
    return license_.strip().lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """An immutable description of a single Panelary feature (operator).

    A :class:`FeatureSpec` is metadata only: it describes *what* an operator is,
    *how* it is called, and *whether it is safe* to use in a leak-free panel ML
    pipeline. The actual computation lives in ``backend_fn`` (or, more commonly,
    in a Polars expression-namespace method that registers this spec).

    Parameters
    ----------
    name : str
        Globally unique operator name within the registry, e.g. ``"frac_diff"``.
    namespace : str
        The Polars expression namespace the operator is exposed under, e.g.
        ``"panel"`` (per-entity time-series ops) or ``"xs"`` (cross-sectional).
    input_shape : str
        Shape of the input the operator consumes. Conventionally ``"series"``
        (a single column / ``pl.Expr``) or ``"frame"`` (multiple columns).
    output_shape : str
        Shape of the operator's output, using the same vocabulary as
        ``input_shape``.
    params : dict[str, type] | dict[str, Any]
        Mapping of parameter name to either its type (for documentation /
        schema generation) or a richer descriptor. Empty for nullary operators.
    tier : str
        Maturity tier, one of :data:`VALID_TIERS` (``"A"``/``"B"``/``"C"``/``"D"``).
    panel_safe : bool
        ``True`` if the operator is safe to apply per entity in a long-format
        panel (i.e. it does not leak information across entities). Operators
        meant to be combined with ``.over(entity)`` are panel-safe.
    leakage_safe : bool
        ``True`` if the operator is *causal* — it never uses future information
        to compute a value at time ``t``. This is the core correctness contract.
    safe_scope : str
        The usage under which ``leakage_safe`` is claimed; one of
        :data:`VALID_SAFE_SCOPES`. ``"rowwise"`` means the value at ``t`` uses
        only data at ``<= t`` when evaluated at every row. ``"window"`` means
        the claim holds only for a summary of an already-delimited window --
        broadcasting such an operator per-row is a look-ahead by construction.
        Defaults to ``"unspecified"`` so existing third-party specs keep
        working; Panelary's own operators must declare it.
    source : str
        Human-readable provenance, e.g. ``"Panelary"`` for original
        implementations or a citation for re-implemented published methods.
    license : str
        SPDX-style license identifier of the implementation, e.g.
        ``"Apache-2.0"``. Audited against :data:`PERMISSIVE_LICENSES`.
    backend_fn : Callable | None, optional
        The callable backing the operator, if one is registered directly.
        Often ``None`` because the implementation is a namespace method.

    Notes
    -----
    The dataclass is frozen and uses ``slots=True``, so instances are hashable,
    immutable, and memory-cheap — safe to treat as value objects.
    """

    name: str
    namespace: str
    input_shape: str
    output_shape: str
    params: dict[str, type] | dict[str, Any] = field(default_factory=dict)
    tier: str = "C"
    panel_safe: bool = False
    leakage_safe: bool = False
    safe_scope: str = "unspecified"
    source: str = ""
    license: str = ""
    backend_fn: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(
                f"FeatureSpec.name must be a non-empty string, got {self.name!r}."
            )
        if not self.namespace or not isinstance(self.namespace, str):
            raise ValueError(
                f"FeatureSpec(name={self.name!r}).namespace must be a non-empty "
                f"string, got {self.namespace!r}."
            )
        if self.safe_scope not in VALID_SAFE_SCOPES:
            raise ValueError(
                f"FeatureSpec(name={self.name!r}).safe_scope must be one of "
                f"{sorted(VALID_SAFE_SCOPES)}, got {self.safe_scope!r}."
            )
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"FeatureSpec(name={self.name!r}).tier must be one of "
                f"{sorted(VALID_TIERS)}, got {self.tier!r}."
            )

    @property
    def qualified_name(self) -> str:
        """Return the dotted ``namespace.name`` identifier."""
        return f"{self.namespace}.{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly ``dict`` view of this spec.

        ``backend_fn`` is rendered as its qualified name (a string) or ``None``
        so the result is JSON-serializable.
        """
        data = asdict(self)
        fn = self.backend_fn
        data["backend_fn"] = (
            getattr(fn, "__qualname__", repr(fn)) if fn is not None else None
        )
        # ``params`` may contain ``type`` objects which are not JSON-serializable;
        # render them as their qualified names while leaving other values intact.
        data["params"] = {
            key: (getattr(val, "__qualname__", val) if isinstance(val, type) else val)
            for key, val in self.params.items()
        }
        return data


class FeatureRegistry:
    """Process-wide collection of :class:`FeatureSpec` operator descriptors.

    Construct via the module-level singleton :data:`registry` rather than
    instantiating directly in application code; a fresh instance is useful in
    tests for isolation.

    The registry enforces name uniqueness and is *idempotent on re-registration
    of an identical spec*: re-importing a namespace module (a common occurrence)
    will not raise. Registering a *different* spec under an existing name raises,
    surfacing accidental name collisions early.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    # -- registration --------------------------------------------------------

    def register(self, spec: FeatureSpec, *, overwrite: bool = False) -> FeatureSpec:
        """Register ``spec``, returning it for convenient chaining.

        Parameters
        ----------
        spec : FeatureSpec
            The specification to add.
        overwrite : bool, default False
            If ``True``, replace any existing spec with the same name. If
            ``False`` (the default), re-registering an *identical* spec is a
            no-op, while registering a *different* spec under an existing name
            raises :class:`ValueError`.

        Returns
        -------
        FeatureSpec
            The registered spec.

        Raises
        ------
        TypeError
            If ``spec`` is not a :class:`FeatureSpec`.
        ValueError
            If a *different* spec is already registered under ``spec.name`` and
            ``overwrite`` is ``False``.
        """
        if not isinstance(spec, FeatureSpec):
            raise TypeError(
                f"FeatureRegistry.register expected a FeatureSpec, got "
                f"{type(spec).__name__!r}."
            )
        existing = self._specs.get(spec.name)
        if existing is not None and not overwrite:
            if existing == spec:
                return existing  # idempotent re-registration
            raise ValueError(
                f"A different feature named {spec.name!r} is already registered "
                f"(existing namespace={existing.namespace!r}, "
                f"new namespace={spec.namespace!r}). Pass overwrite=True to "
                f"replace it, or choose a unique name."
            )
        self._specs[spec.name] = spec
        return spec

    # -- lookup --------------------------------------------------------------

    def get(self, name: str) -> FeatureSpec:
        """Return the spec named ``name``.

        Raises
        ------
        KeyError
            If no feature is registered under ``name``. The message lists the
            closest available names to aid discovery.
        """
        try:
            return self._specs[name]
        except KeyError:
            available = ", ".join(sorted(self._specs)) or "<none registered>"
            raise KeyError(
                f"No feature named {name!r} is registered. "
                f"Available features: {available}."
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterable[FeatureSpec]:  # type: ignore[override]
        return iter(self._specs.values())

    def all(self) -> list[FeatureSpec]:
        """Return all registered specs, sorted by qualified name."""
        return sorted(self._specs.values(), key=lambda s: s.qualified_name)

    def by_namespace(self, ns: str) -> list[FeatureSpec]:
        """Return all specs registered under namespace ``ns``, sorted by name."""
        return sorted(
            (s for s in self._specs.values() if s.namespace == ns),
            key=lambda s: s.name,
        )

    def namespaces(self) -> list[str]:
        """Return the sorted list of distinct namespaces currently registered."""
        return sorted({s.namespace for s in self._specs.values()})

    # -- serialization -------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Return a list of serialization-friendly dicts, one per spec.

        Suitable for JSON dumping, building a Polars/pandas DataFrame, or
        feeding a documentation generator.
        """
        return [spec.to_dict() for spec in self.all()]

    def to_llms_txt(self) -> str:
        """Render a machine-readable operator catalogue for the agent layer.

        The output follows the spirit of the ``llms.txt`` convention: a compact,
        deterministic, line-oriented description of every operator that an LLM
        agent can parse to discover what Panelary can do and how to call it. The
        rendering is stable (operators sorted by qualified name) so it diffs
        cleanly and can be checked into ``llms.txt``.

        Returns
        -------
        str
            The catalogue text.
        """
        lines: list[str] = [
            "# Panelary operator catalogue",
            "",
            "Machine-readable list of registered Polars expression operators.",
            "Each operator is exposed as `pl.col(...).<namespace>.<name>(...)`",
            "and is typically combined with `.over(entity)` (panel) or",
            "`.over(date)` (cross-sectional, xs).",
            "",
            f"operators: {len(self._specs)}",
            "",
        ]
        for ns in self.namespaces():
            lines.append(f"## namespace: {ns}")
            lines.append("")
            for spec in self.by_namespace(ns):
                params = ", ".join(
                    f"{key}: {getattr(val, '__name__', val)}"
                    for key, val in spec.params.items()
                )
                signature = f"pl.col(...).{spec.namespace}.{spec.name}({params})"
                lines.append(f"- {signature}")
                lines.append(
                    f"    tier={spec.tier} "
                    f"input={spec.input_shape} output={spec.output_shape} "
                    f"panel_safe={spec.panel_safe} leakage_safe={spec.leakage_safe}"
                )
                lines.append(f"    source={spec.source!r} license={spec.license!r}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # -- auditing ------------------------------------------------------------

    def audit(self) -> dict[str, list[str]]:
        """Audit specs for missing safety metadata or non-permissive licenses.

        This is the provenance/license guard: it returns, by category, the names
        of operators that are not fully cleared for inclusion. An empty value
        for every category means the catalogue is clean.

        Returns
        -------
        dict[str, list[str]]
            A mapping with the keys:

            ``"missing_panel_safe"``
                Operators whose ``panel_safe`` is ``False`` (i.e. unverified or
                explicitly unsafe for per-entity application).
            ``"missing_leakage_safe"``
                Operators whose ``leakage_safe`` is ``False`` (unverified or
                non-causal — a leakage risk).
            ``"missing_provenance"``
                Operators with an empty ``source`` field.
            ``"non_permissive_license"``
                Operators whose ``license`` is empty or not in
                :data:`PERMISSIVE_LICENSES` — the GPL/copyleft tripwire.

        Each list is sorted by operator name.
        """
        missing_panel_safe: list[str] = []
        missing_leakage_safe: list[str] = []
        missing_provenance: list[str] = []
        non_permissive_license: list[str] = []

        for spec in self.all():
            if not spec.panel_safe:
                missing_panel_safe.append(spec.name)
            if not spec.leakage_safe:
                missing_leakage_safe.append(spec.name)
            if not spec.source.strip():
                missing_provenance.append(spec.name)
            if (
                not spec.license.strip()
                or _normalize_license(spec.license) not in PERMISSIVE_LICENSES
            ):
                non_permissive_license.append(spec.name)

        return {
            "missing_panel_safe": missing_panel_safe,
            "missing_leakage_safe": missing_leakage_safe,
            "missing_provenance": missing_provenance,
            "non_permissive_license": non_permissive_license,
        }

    def clear(self) -> None:
        """Remove all registered specs. Primarily useful in tests."""
        self._specs.clear()


def register_feature(
    *,
    name: str,
    namespace: str,
    input_shape: str,
    output_shape: str,
    params: dict[str, type] | dict[str, Any] | None = None,
    tier: str = "C",
    panel_safe: bool = False,
    leakage_safe: bool = False,
    safe_scope: str = "unspecified",
    source: str = "",
    license: str = "",
    target: FeatureRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that builds a :class:`FeatureSpec` and registers it.

    The decorated callable is stored as the spec's ``backend_fn`` and returned
    unchanged, so the decorator is transparent — it can wrap a real backend
    function or a tiny marker function on a namespace method.

    Parameters
    ----------
    name, namespace, input_shape, output_shape, tier, panel_safe, leakage_safe, safe_scope, source, license
        Forwarded to :class:`FeatureSpec`. ``safe_scope`` names the usage under
        which ``leakage_safe`` is claimed and should be set explicitly — a
        decorator-registered operator that leaves it at ``"unspecified"`` is
        rejected by ``tests/test_registry_conformance.py``.
    params : dict[str, type] | dict[str, Any] | None, optional
        Forwarded to :class:`FeatureSpec`; defaults to an empty mapping.
    target : FeatureRegistry | None, optional
        Registry to register into; defaults to the module-level
        :data:`registry` singleton.

    Returns
    -------
    Callable
        A decorator that registers the spec and returns the wrapped function.
    """
    reg = target if target is not None else registry

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = FeatureSpec(
            name=name,
            namespace=namespace,
            input_shape=input_shape,
            output_shape=output_shape,
            params=dict(params) if params is not None else {},
            tier=tier,
            panel_safe=panel_safe,
            leakage_safe=leakage_safe,
            safe_scope=safe_scope,
            source=source,
            license=license,
            backend_fn=fn,
        )
        reg.register(spec)
        return fn

    return _decorator


#: The module-level :class:`FeatureRegistry` singleton. Import this everywhere.
registry = FeatureRegistry()
