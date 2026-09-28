"""Leak-safe pipeline of panel transformers and estimators.

A :class:`Pipeline` chains named steps, each a
:class:`~panelary.core.protocol.PanelTransformer` (and optionally a final
:class:`~panelary.core.protocol.PanelEstimator`). It mirrors sklearn's
``Pipeline`` ergonomics — named steps, ``named_steps``, integer/string
``__getitem__``, slicing — while guaranteeing panel and leakage safety:

* :meth:`Pipeline.fit` fits each step **only on training-fold data**, threading
  the transformed :class:`~panelary.core.panel_frame.PanelFrame` forward.
  Step *k* is fitted on the output of steps *0..k-1* applied to the train fold,
  so no step ever sees a future row or a test-fold statistic.
* :meth:`Pipeline.transform` / :meth:`Pipeline.predict` apply the **already
  fitted** steps in order; nothing is re-fitted, so applying the pipeline to a
  test fold cannot leak.

All steps except the last must be transformers. The last may be a transformer
(pure feature pipeline) or an estimator (then :meth:`predict` is available).

:meth:`Pipeline.audit` and :meth:`Pipeline.causalize` are the pipeline-level
face of :mod:`panelary.leakage`. Note what a :class:`Pipeline` actually holds:
``(name, PanelTransformer)`` tuples, and nothing else — no Polars expressions.
So the audit is, for an ordinary step, a reading of that step's mandatory
``panel_safe`` / ``leakage_safe`` declaration, not a decompilation of a Python
object. The point-in-time compiler only has something to walk when a step
*offers* its expressions through the small duck-typed hook documented on
:meth:`Pipeline.audit`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelEstimator, PanelTransformer

if TYPE_CHECKING:
    from typing import Self

__all__ = ["Pipeline", "PipelineAudit", "StepAudit"]

Step = tuple[str, PanelTransformer]

# Verdict strings. These are deliberately the *values* of
# :class:`panelary.leakage.Verdict` (a ``str`` Enum), so
# ``step.verdict == Verdict.REFUSED`` compares True either way, without
# ``core.pipeline`` having to import the leakage package to name a constant.
_SAFE = "safe"
_REWRITTEN = "rewritten"
_REFUSED = "refused"
_VERDICT_RANK = {_SAFE: 0, _REWRITTEN: 1, _REFUSED: 2}


def _worst(verdicts: Iterator[str] | tuple[str, ...] | list[str]) -> str:
    """The most severe verdict in ``verdicts`` (``safe`` if empty)."""
    return max(verdicts, key=lambda v: _VERDICT_RANK[v], default=_SAFE)


def _import_leakage() -> tuple[Any, str | None]:
    """Import :mod:`panelary.leakage`, returning ``(module, None)`` or ``(None, why)``.

    The import is deliberately deferred to call time rather than done at module
    scope. ``panelary.core.pipeline`` sits near the bottom of the import graph
    (``core.protocol`` -> ``core.panel_frame``) and ``panelary.leakage`` sits
    above it, so a top-level import would risk a cycle. This mirrors what
    ``panelary/base/forecaster.py`` does — and yes, the layout contract flags
    that file's deferred-import pattern as a smell; here it is the cycle, not
    an optional dependency, that forces it. :mod:`panelary.leakage` is pure
    numpy + polars and is imported eagerly by ``panelary/__init__.py``, so this
    costs nothing at runtime.
    """
    try:
        from panelary import leakage
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"
    return leakage, None


def _step_exprs(step: PanelTransformer) -> dict[str, pl.Expr] | None:
    """The Polars expressions a step offers the compiler, or ``None``.

    A step opts in to expression-level auditing by implementing
    ``leakage_exprs() -> dict[str, polars.Expr]``. Returning ``None`` (or not
    implementing the method at all) means the step is audited from its declared
    ``panel_safe`` / ``leakage_safe`` attributes instead.
    """
    getter = getattr(step, "leakage_exprs", None)
    if not callable(getter):
        return None
    exprs = getter()
    if exprs is None:
        return None
    if not isinstance(exprs, dict):
        raise TypeError(
            f"{type(step).__name__}.leakage_exprs() must return a "
            f"dict[str, polars.Expr] or None, got {type(exprs).__name__!r}."
        )
    for key, value in exprs.items():
        if not isinstance(key, str) or not isinstance(value, pl.Expr):
            raise TypeError(
                f"{type(step).__name__}.leakage_exprs() must return a "
                f"dict[str, polars.Expr]; got key {key!r} -> "
                f"{type(value).__name__!r}."
            )
    return dict(exprs)


@dataclass(frozen=True)
class StepAudit:
    """The leakage verdict for one pipeline step.

    Attributes
    ----------
    name : str
        The step's name within the pipeline.
    step : str
        Class name of the step object.
    verdict : str
        One of ``"safe"``, ``"rewritten"`` or ``"refused"``. The values match
        :class:`panelary.leakage.Verdict`, which is a ``str`` enum, so they
        compare equal to it.
    panel_safe, leakage_safe : bool
        The step's declared class attributes, reported verbatim.
    reasons : tuple of str
        Human-readable justification for ``verdict``, one entry per reason.
    results : tuple
        One :class:`panelary.leakage.CompileResult` per offered expression, in
        the order the step offered them. Empty for an attribute-audited step.
    nested : PipelineAudit or None
        The sub-report when the step is itself a :class:`Pipeline`.
    expr_names : tuple of str
        The names under which the step offered its expressions, aligned with
        ``results`` (``leakage_exprs()`` keys). Empty for an attribute-audited
        step.
    """

    name: str
    step: str
    verdict: str
    panel_safe: bool
    leakage_safe: bool
    reasons: tuple[str, ...] = ()
    results: tuple[Any, ...] = ()
    nested: PipelineAudit | None = None
    expr_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """This step's verdict as a JSON-safe plain dict.

        Keys: ``name``, ``step`` (class name), ``verdict``, ``panel_safe``,
        ``leakage_safe``, ``reasons``, ``audited_by`` (``"expressions"`` when
        the compiler walked offered expressions, ``"pipeline"`` for a nested
        pipeline, else ``"declaration"`` -- the step was judged on its
        ``panel_safe`` / ``leakage_safe`` attributes alone), ``results`` (one
        ``{"name", "result"}`` per offered expression, each result a
        :meth:`panelary.leakage.CompileResult.to_dict`) and ``nested`` (a
        :meth:`PipelineAudit.to_dict`, or ``None``).
        """
        names = list(self.expr_names) + [
            f"#{i}" for i in range(len(self.expr_names), len(self.results))
        ]
        if self.nested is not None:
            audited_by = "pipeline"
        elif self.results or self.expr_names:
            audited_by = "expressions"
        else:
            audited_by = "declaration"
        return {
            "name": self.name,
            "step": self.step,
            "verdict": str(getattr(self.verdict, "value", self.verdict)),
            "panel_safe": bool(self.panel_safe),
            "leakage_safe": bool(self.leakage_safe),
            "reasons": [str(r) for r in self.reasons],
            "audited_by": audited_by,
            "results": [
                {"name": n, "result": r.to_dict()}
                for n, r in zip(names, self.results, strict=False)
            ],
            "nested": None if self.nested is None else self.nested.to_dict(),
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        why = "; ".join(self.reasons)
        return f"{self.name} ({self.step}): {self.verdict}" + (
            f" — {why}" if why else ""
        )


@dataclass(frozen=True)
class PipelineAudit:
    """The per-step leakage report for a whole :class:`Pipeline`.

    Attributes
    ----------
    steps : tuple of StepAudit
        One entry per pipeline step, in pipeline order.
    """

    steps: tuple[StepAudit, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        """The most severe step verdict (``refused`` > ``rewritten`` > ``safe``)."""
        return _worst([s.verdict for s in self.steps])

    @property
    def ok(self) -> bool:
        """Whether :meth:`Pipeline.causalize` would succeed (nothing refused)."""
        return self.verdict != _REFUSED

    @property
    def refused(self) -> tuple[StepAudit, ...]:
        """The steps that cannot be made point-in-time."""
        return tuple(s for s in self.steps if s.verdict == _REFUSED)

    @property
    def rewritten(self) -> tuple[StepAudit, ...]:
        """The steps :meth:`Pipeline.causalize` would rewrite."""
        return tuple(s for s in self.steps if s.verdict == _REWRITTEN)

    #: Serialisation schema tag; bump the suffix on any breaking change.
    SCHEMA = "panelary.PipelineAudit/1"

    def to_dict(self) -> dict[str, Any]:
        """The whole pipeline audit as a JSON-safe plain dict -- report evidence.

        Returns
        -------
        dict
            ``schema`` (``"panelary.PipelineAudit/1"``), ``produced_by``
            (``"panelary.core.pipeline.Pipeline.audit@<version>"``),
            ``verdict``, ``ok``, ``counts`` (steps per verdict), ``refused`` /
            ``rewritten`` (step names, pipeline order) and ``steps`` (each
            :meth:`StepAudit.to_dict`, in pipeline order).

        Notes
        -----
        Same convention as :meth:`panelary.leakage.CompileResult.to_dict`:
        no timestamps, ids or memory addresses, so :meth:`to_json` is
        byte-identical across runs.
        """
        from panelary.leakage._types import produced_by

        counts = {_SAFE: 0, _REWRITTEN: 0, _REFUSED: 0}
        for step in self.steps:
            counts[str(getattr(step.verdict, "value", step.verdict))] += 1
        return {
            "schema": self.SCHEMA,
            "produced_by": produced_by("panelary.core.pipeline.Pipeline.audit"),
            "verdict": self.verdict,
            "ok": self.ok,
            "counts": counts,
            "refused": [s.name for s in self.refused],
            "rewritten": [s.name for s in self.rewritten],
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict` as canonical JSON (sorted keys, compact by default).

        With ``indent=None`` this is ``json.dumps(self.to_dict(),
        sort_keys=True, separators=(",", ":"))``: byte-identical across runs.
        """
        from panelary.leakage._types import canonical_json

        return canonical_json(self.to_dict(), indent=indent)

    def __str__(self) -> str:  # pragma: no cover - display only
        head = f"PipelineAudit: {self.verdict} ({len(self.steps)} steps)"
        return "\n  ".join([head, *(str(s) for s in self.steps)])


class Pipeline(PanelTransformer):
    """A sequential chain of panel transformers with a leak-safe ``fit``.

    Parameters
    ----------
    steps : list of (str, PanelTransformer)
        Named steps, applied in order. Names must be unique and must not contain
        ``"__"`` (reserved, sklearn-style, for nested parameter access). Every
        step except the last must be a
        :class:`~panelary.core.protocol.PanelTransformer`; the last may be
        a :class:`~panelary.core.protocol.PanelEstimator`.

    Attributes
    ----------
    steps : list of (str, PanelTransformer)
        The configured steps.
    named_steps : dict[str, PanelTransformer]
        Mapping from step name to the step object (sklearn-compatible).
    panel_safe : bool
        True iff every step is panel-safe.
    leakage_safe : bool
        True iff every step is leakage-safe.

    Raises
    ------
    TypeError
        If ``steps`` is malformed, a non-final step is not a transformer, or a
        step object is not a PanelTransformer.
    ValueError
        If step names are non-unique or contain ``"__"``.

    Examples
    --------
    >>> pipe = Pipeline([("scale", scaler), ("model", forecaster)])  # doctest: +SKIP
    >>> pipe.fit(train_panel)                                        # doctest: +SKIP
    >>> preds = pipe.predict(test_panel)                            # doctest: +SKIP

    Notes
    -----
    **Leakage guarantee.** Because every step is fitted strictly on the
    train-fold output of its predecessors and never re-fitted at transform time,
    the pipeline preserves the per-step leakage contract end to end. A pipeline
    is leakage-safe iff all of its steps are.
    """

    # Computed in __init__ from the constituent steps; declared here so the
    # PanelTransformer subclass-check passes (these are overwritten per-instance).
    panel_safe: bool = True
    leakage_safe: bool = True

    def __init__(
        self,
        steps: list[Step],
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self._validate_steps(steps)
        self.steps: list[Step] = list(steps)
        # Instance-level safety flags derived from the steps.
        self.panel_safe = all(t.panel_safe for _, t in self.steps)
        self.leakage_safe = all(t.leakage_safe for _, t in self.steps)
        # Set at fit-time so transform/predict can tell whether they are being
        # applied across a train/test boundary (see `_is_cross_boundary`).
        self._train_panel: PanelFrame | None = None
        self._train_times: set[object] | None = None

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_steps(steps: object) -> None:
        if not isinstance(steps, (list, tuple)) or len(steps) == 0:
            raise TypeError(
                "Pipeline `steps` must be a non-empty list of (name, transformer) "
                f"tuples, got {steps!r}."
            )
        names: list[str] = []
        for i, step in enumerate(steps):
            if not (isinstance(step, tuple) and len(step) == 2):
                raise TypeError(
                    f"step {i} must be a (name, transformer) tuple, got {step!r}."
                )
            name, obj = step
            if not isinstance(name, str):
                raise TypeError(
                    f"step {i} name must be a string, got {type(name).__name__!r}."
                )
            if "__" in name:
                raise ValueError(
                    f"step name {name!r} must not contain '__' (reserved for "
                    "nested parameter access)."
                )
            if not isinstance(obj, PanelTransformer):
                raise TypeError(
                    f"step {name!r} must be a PanelTransformer, got "
                    f"{type(obj).__name__!r}."
                )
            names.append(name)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Pipeline step names must be unique; duplicates: {dupes}."
            )
        # All but last must be transformers (not necessarily estimators). Every
        # PanelTransformer can transform, so the only thing we forbid is a
        # *non-final* step that cannot transform — which is impossible given the
        # type check above. We additionally warn-by-raising if a non-final step
        # is an estimator used purely as a transformer is fine, so nothing else
        # to enforce here.

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        """Fit each step on the train-fold output of its predecessors."""
        # Remember the training fold so transform/predict can detect a later
        # train/test boundary and enforce each step's leakage contract.
        self._train_panel = panel
        self._train_times = set(panel.collect()[panel.time_col].to_list())
        out = panel
        # Fit-transform every step except the last; fit the last appropriately.
        for name, transformer in self.steps[:-1]:
            out = transformer.fit_transform(out)
            if not isinstance(out, PanelFrame):
                raise TypeError(
                    f"step {name!r} ({type(transformer).__name__}) returned "
                    f"{type(out).__name__!r} from transform; every step must "
                    "return a PanelFrame so the next step can consume it."
                )
        # Final step: fit on the threaded output. Do not transform here; the
        # threaded value is only needed for downstream fitting which is done.
        last_name, last = self.steps[-1]
        last.fit(out)

    def _is_cross_boundary(self, panel: PanelFrame) -> bool:
        """Whether ``panel`` contains times not seen at fit time (a test fold).

        Returns False when the pipeline was not fitted with recorded times, or
        when every time in ``panel`` was already present in the training fold
        (so applying the fitted steps cannot cross a train/test boundary).
        """
        if self._train_times is None:
            return False
        times = set(panel.collect()[panel.time_col].to_list())
        return not times.issubset(self._train_times)

    def _enforce_leakage(self, panel: PanelFrame) -> None:
        """Refuse a not-leakage-safe pipeline applied across a boundary.

        Delegates to each step's :meth:`~panelary.core.protocol.
        PanelTransformer._check_leakage`, which raises for any step declaring
        ``leakage_safe = False``. A no-op when ``panel`` is the training fold.
        """
        if not self._is_cross_boundary(panel):
            return
        train = self._train_panel
        for _name, transformer in self.steps:
            transformer._check_leakage(train, panel)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        """Apply every fitted step's ``transform`` in order."""
        self._enforce_leakage(panel)
        out = panel
        for name, transformer in self.steps:
            out = transformer.transform(out)
            if not isinstance(out, PanelFrame):
                raise TypeError(
                    f"step {name!r} ({type(transformer).__name__}) returned "
                    f"{type(out).__name__!r} from transform; expected PanelFrame."
                )
        return out

    def predict(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Apply all transformers, then ``predict`` with the final estimator.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
            Data to predict on (train or test). Accepts a :class:`PanelFrame` or
            a bare polars frame wrapped on the fly using ``entity`` / ``time``
            (or the keys configured on the constructor).
        entity, time : str, optional
            Panel keys for a bare polars frame.

        Returns
        -------
        PanelFrame
            Predictions from the final estimator, keyed by ``(entity, time)``.

        Raises
        ------
        RuntimeError
            If the pipeline is not fitted.
        TypeError
            If the final step is not a
            :class:`~panelary.core.protocol.PanelEstimator`.
        """
        panel = self._as_panel(X, method="predict", entity=entity, time=time)
        self._check_fitted("predict")
        last_name, last = self.steps[-1]
        if not isinstance(last, PanelEstimator):
            raise TypeError(
                f"final step {last_name!r} is a {type(last).__name__}, which is "
                "not a PanelEstimator; `predict` requires the last step to be an "
                "estimator. Use `transform` for a feature-only pipeline."
            )
        self._enforce_leakage(panel)
        out = panel
        for _name, transformer in self.steps[:-1]:
            out = transformer.transform(out)
        return last.predict(out)

    # ------------------------------------------------------------------ #
    # Leakage: audit / causalize
    # ------------------------------------------------------------------ #
    def _audit_step(
        self,
        name: str,
        step: PanelTransformer,
        *,
        leakage: Any,
        why_no_leakage: str | None,
        time: str | None,
        entity: str | None,
        allow_approximate: bool,
    ) -> StepAudit:
        """Audit one step. Fails closed: anything unclear is ``refused``."""
        kind = type(step).__name__
        declared_panel_safe = bool(step.panel_safe)
        declared_leakage_safe = bool(step.leakage_safe)

        def _verdict(
            verdict: str,
            reasons: tuple[str, ...] = (),
            results: tuple[Any, ...] = (),
            nested: PipelineAudit | None = None,
        ) -> StepAudit:
            return StepAudit(
                name=name,
                step=kind,
                verdict=verdict,
                panel_safe=declared_panel_safe,
                leakage_safe=declared_leakage_safe,
                reasons=reasons,
                results=results,
                nested=nested,
                # Names of the offered expressions actually audited, aligned
                # with ``results`` (a fail-closed early return may cut short).
                expr_names=tuple(expr_names[: len(results)]),
            )

        expr_names: list[str] = []

        # A nested Pipeline is audited as a pipeline, recursively.
        if isinstance(step, Pipeline):
            nested = step.audit(
                time=time, entity=entity, allow_approximate=allow_approximate
            )
            return _verdict(
                nested.verdict,
                reasons=tuple(
                    f"sub-step {s.name!r}: {'; '.join(s.reasons)}"
                    for s in nested.steps
                    if s.verdict != _SAFE
                ),
                nested=nested,
            )

        exprs = _step_exprs(step)
        if exprs is not None:
            expr_names.extend(exprs)

        if exprs is None:
            # Nothing for the compiler to walk. The honest reading of an
            # ordinary step is its own mandatory declaration (AGENTS.md: every
            # concrete PanelTransformer must set both attributes).
            declared: list[str] = []
            if not declared_panel_safe:
                declared.append(
                    "declares `panel_safe = False`: it may mix rows across "
                    "entities. A Python object cannot be rewritten, so this "
                    "cannot be repaired here."
                )
            if not declared_leakage_safe:
                declared.append(
                    "declares `leakage_safe = False`: it may use information "
                    "from the future, or fit on transform-time data. A Python "
                    "object cannot be rewritten, so this cannot be repaired "
                    "here — replace the step, or fit it per fold and accept "
                    "the risk explicitly."
                )
            return _verdict(_REFUSED if declared else _SAFE, reasons=tuple(declared))

        # The step offered its expressions: by doing so it declares that those
        # expressions are its entire leakage surface, and the compiler governs.
        if leakage is None:
            return _verdict(
                _REFUSED,
                reasons=(
                    f"offers {len(exprs)} expression(s) but the point-in-time "
                    f"compiler could not be imported ({why_no_leakage}); "
                    "failing closed rather than assuming they are safe.",
                ),
            )

        results: list[Any] = []
        reasons: list[str] = []
        for key, expr in exprs.items():
            try:
                res = leakage.audit(
                    expr,
                    time=time,
                    entity=entity,
                    allow_approximate=allow_approximate,
                )
            except Exception as exc:
                return _verdict(
                    _REFUSED,
                    reasons=(
                        f"expression {key!r} could not be audited "
                        f"({type(exc).__name__}: {exc}); failing closed.",
                    ),
                    results=tuple(results),
                )
            results.append(res)
            res_verdict = str(getattr(res.verdict, "value", res.verdict))
            if res_verdict != _SAFE:
                detail = "; ".join(str(f) for f in res.findings) or res_verdict
                reasons.append(f"expression {key!r}: {detail}")

        verdict = _worst([str(getattr(r.verdict, "value", r.verdict)) for r in results])
        if verdict == _REWRITTEN and not callable(
            getattr(step, "with_leakage_exprs", None)
        ):
            verdict = _REFUSED
            reasons.append(
                f"{kind} offers expressions that need rewriting but does not "
                "implement `with_leakage_exprs(mapping)`, so a corrected step "
                "cannot be built."
            )
        return _verdict(verdict, reasons=tuple(reasons), results=tuple(results))

    def audit(
        self,
        *,
        time: str | None = None,
        entity: str | None = None,
        allow_approximate: bool = False,
    ) -> PipelineAudit:
        """Report, per step, what this pipeline does with information from the future.

        A :class:`Pipeline` holds ``(name, PanelTransformer)`` tuples and
        nothing else, so most steps are audited from the ``panel_safe`` /
        ``leakage_safe`` class attributes that every concrete
        :class:`~panelary.core.protocol.PanelTransformer` is required to
        declare. That declaration is the contract; a fitted Python object is
        not something the point-in-time compiler can walk.

        A step may opt in to expression-level auditing by implementing a pair
        of optional methods:

        ``leakage_exprs() -> dict[str, polars.Expr] | None``
            The expressions the step will apply. Offering them declares that
            they are the step's entire leakage surface, and
            :func:`panelary.leakage.audit` then governs the verdict.
        ``with_leakage_exprs(mapping) -> PanelTransformer``
            Return a **new** step using the rewritten expressions. Required
            only if the step's expressions might need rewriting.

        No transformer shipped with Panelary implements this hook today; it is
        the seam through which an expression-shaped step becomes compilable.

        Parameters
        ----------
        time, entity : str, optional
            Panel keys handed to the compiler. They default to the keys
            configured on the constructor. ``time`` in particular is what lets
            the compiler repair an ``.over(entity)`` that carries no
            ``order_by``; without it that node is refused rather than guessed.
        allow_approximate : bool, default=False
            Permit rewrites that are causal but not numerically identical to
            the original (an expanding quantile, say).

        Returns
        -------
        PipelineAudit
            One :class:`StepAudit` per step, in pipeline order, with an
            overall :attr:`PipelineAudit.verdict`.

        See Also
        --------
        Pipeline.causalize : Build the corrected pipeline, or refuse.
        panelary.leakage.audit : The expression-level compiler this wraps.

        Examples
        --------
        >>> report = pipe.audit(time="date", entity="ticker")   # doctest: +SKIP
        >>> report.verdict                                       # doctest: +SKIP
        'safe'
        """
        leakage, why = _import_leakage()
        return PipelineAudit(
            steps=tuple(
                self._audit_step(
                    name,
                    step,
                    leakage=leakage,
                    why_no_leakage=why,
                    time=time if time is not None else self._time,
                    entity=entity if entity is not None else self._entity,
                    allow_approximate=allow_approximate,
                )
                for name, step in self.steps
            )
        )

    def causalize(
        self,
        *,
        time: str | None = None,
        entity: str | None = None,
        allow_approximate: bool = False,
    ) -> Pipeline:
        """Return a new pipeline with every rewritable step made point-in-time.

        A thin facade over :func:`panelary.leakage.causalize`. It never mutates
        this pipeline: the returned :class:`Pipeline` is a new object with a new
        ``steps`` list. Steps that need no rewrite are carried across by
        reference (they are unchanged by construction); a step that needed one
        is replaced by whatever its ``with_leakage_exprs`` returned.

        Because it fails closed, a pipeline that cannot be fully repaired raises
        rather than returning a partially-corrected object.

        Parameters
        ----------
        time, entity : str, optional
            Panel keys handed to the compiler; see :meth:`audit`.
        allow_approximate : bool, default=False
            Permit causal-but-not-identical rewrites; see :meth:`audit`.

        Returns
        -------
        Pipeline
            A new pipeline whose audit is free of refusals.

        Raises
        ------
        panelary.leakage.LeakageRefused
            If any step cannot be made point-in-time. The exception carries a
            :class:`panelary.leakage.CompileResult` whose findings name every
            offending step.
        RuntimeError
            If a step must be refused but :mod:`panelary.leakage` could not be
            imported, so the typed exception is unavailable.

        Notes
        -----
        Call this **before** fitting. Rewritten steps are fresh, unfitted
        objects, and unchanged steps are shared with the original pipeline, so
        fitting the result after fitting the original would refit those shared
        steps.

        Examples
        --------
        >>> safe_pipe = pipe.causalize(time="date", entity="ticker")  # doctest: +SKIP
        """
        leakage, why = _import_leakage()
        time = time if time is not None else self._time
        entity = entity if entity is not None else self._entity
        report = self.audit(
            time=time, entity=entity, allow_approximate=allow_approximate
        )
        if not report.ok:
            self._raise_refused(report, leakage=leakage, why_no_leakage=why)

        new_steps: list[Step] = []
        for (name, step), step_report in zip(self.steps, report.steps, strict=True):
            if isinstance(step, Pipeline):
                new_steps.append(
                    (
                        name,
                        step.causalize(
                            time=time,
                            entity=entity,
                            allow_approximate=allow_approximate,
                        ),
                    )
                )
                continue
            if step_report.verdict != _REWRITTEN:
                new_steps.append((name, step))
                continue
            new_steps.append(
                (
                    name,
                    self._rewrite_step(
                        name,
                        step,
                        leakage=leakage,
                        time=time,
                        entity=entity,
                        allow_approximate=allow_approximate,
                    ),
                )
            )
        return type(self)(new_steps, entity=self._entity, time=self._time)

    def _rewrite_step(
        self,
        name: str,
        step: PanelTransformer,
        *,
        leakage: Any,
        time: str | None,
        entity: str | None,
        allow_approximate: bool,
    ) -> PanelTransformer:
        """Rebuild one expression-carrying step from causalized expressions."""
        exprs = _step_exprs(step) or {}
        rewritten = {
            key: leakage.causalize(
                expr, time=time, entity=entity, allow_approximate=allow_approximate
            )
            for key, expr in exprs.items()
        }
        new_step = step.with_leakage_exprs(rewritten)  # type: ignore[attr-defined]
        if not isinstance(new_step, PanelTransformer):
            raise TypeError(
                f"step {name!r} ({type(step).__name__}).with_leakage_exprs() must "
                f"return a PanelTransformer, got {type(new_step).__name__!r}."
            )
        if new_step is step:
            raise ValueError(
                f"step {name!r} ({type(step).__name__}).with_leakage_exprs() "
                "returned the same object; `causalize` must not mutate the "
                "original pipeline, so it must return a new step."
            )
        # Verify rather than trust: the rebuilt step's expressions must now
        # compile clean. A step that dropped the rewrite on the floor is a
        # silent leak, which is exactly what this method exists to prevent.
        check = self._audit_step(
            name,
            new_step,
            leakage=leakage,
            why_no_leakage=None,
            time=time,
            entity=entity,
            allow_approximate=allow_approximate,
        )
        if check.verdict != _SAFE:
            raise ValueError(
                f"step {name!r} ({type(step).__name__}) was rebuilt from "
                f"causalized expressions but still audits as {check.verdict!r}: "
                f"{'; '.join(check.reasons)}. `with_leakage_exprs` must apply the "
                "expressions it is given."
            )
        return new_step

    @staticmethod
    def _raise_refused(
        report: PipelineAudit, *, leakage: Any, why_no_leakage: str | None
    ) -> None:
        """Raise ``LeakageRefused`` describing every refused step."""
        summary = "\n  ".join(
            f"step {s.name!r} ({s.step}): {'; '.join(s.reasons) or s.verdict}"
            for s in report.refused
        )
        message = f"pipeline cannot be made point-in-time:\n  {summary}"
        if leakage is None:  # pragma: no cover - defensive
            raise RuntimeError(
                f"{message}\n(panelary.leakage is unavailable — "
                f"{why_no_leakage} — so LeakageRefused cannot be raised.)"
            )
        findings: list[Any] = []
        for step_report in report.refused:
            nested = [f for r in step_report.results for f in r.refused]
            if nested:
                findings.extend(nested)
            else:
                findings.append(
                    leakage.Finding(
                        kind=f"Step.{step_report.step}",
                        classification=leakage.Classification.REFUSE,
                        reason="; ".join(step_report.reasons) or step_report.verdict,
                        path=(step_report.name,),
                    )
                )
        raise leakage.LeakageRefused(
            message,
            leakage.CompileResult(
                verdict=leakage.Verdict.REFUSED, findings=tuple(findings)
            ),
        )

    # ------------------------------------------------------------------ #
    # sklearn-shaped ergonomics
    # ------------------------------------------------------------------ #
    @property
    def named_steps(self) -> dict[str, PanelTransformer]:
        """Mapping from step name to step object."""
        return dict(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, key: int | str | slice) -> PanelTransformer | Self:
        """Index a step by position or name, or slice into a sub-pipeline.

        Parameters
        ----------
        key : int | str | slice
            * ``int`` -> the transformer at that position.
            * ``str`` -> the transformer with that name.
            * ``slice`` -> a new :class:`Pipeline` of the selected steps.

        Returns
        -------
        PanelTransformer or Pipeline
        """
        if isinstance(key, slice):
            return type(self)(self.steps[key])
        if isinstance(key, str):
            try:
                return self.named_steps[key]
            except KeyError:
                raise KeyError(
                    f"no step named {key!r}; available steps: "
                    f"{[n for n, _ in self.steps]}."
                ) from None
        if isinstance(key, int):
            return self.steps[key][1]
        raise TypeError(
            f"Pipeline indices must be int, str, or slice, not {type(key).__name__!r}."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        body = ", ".join(f"({n!r}, {type(o).__name__})" for n, o in self.steps)
        state = "fitted" if getattr(self, "_fitted", False) else "not fitted"
        return f"Pipeline(steps=[{body}], {state})"
