"""Borrowed accuracy: the leakage gap, decomposed exactly.

Every pipeline can be run twice: **permissively**, with each component fit once
on everything, and **point-in-time**, with each component refit per fold on the
training rows only. The gap between the two scores is the accuracy *borrowed*
from data the method will not have at prediction time::

    B = v(all permissive) - v(all point-in-time)

``B`` on its own is a single number and says nothing about *which* component
borrowed it. Ablating one component at a time does not answer that either: two
components can be individually harmless and leak badly together, and
one-at-a-time ablations attribute zero to both (see the interaction test in
``tests/test_leakage_borrowed.py``). The honest attribution is the **exact
Shapley value** of the cooperative game whose characteristic function is
``v(S)`` -- the score when exactly the components in ``S`` run permissively::

    phi_i = sum_{S subset of N\\{i}} |S|! (k-|S|-1)! / k! * ( v(S + i) - v(S) )

Shapley's efficiency axiom gives ``sum_i phi_i == B`` by construction, which is
what makes this a *decomposition* rather than a pile of ablations.

**This is a causal attribution, not an associational one.** In the terms of
Jung et al. (ICML 2022), it is an *interventional* (do-)Shapley value:
``v(S) = E[score | do(mode_S = permissive, mode_rest = point-in-time)]``. Their
Sections 4-5 exist because an outcome produced by nature cannot be rerun, so
every ``E[Y | do(v_S)]`` must be identified from a causal graph and estimated.
A pipeline *is* its own causal model: an intervention is a rerun, every
coalition here is **executed** rather than estimated, and identification is
trivial. The paper's axioms carry over one for one -- efficiency is
``sum(phi) == B``, causal irrelevance is the null-player property. Strictly it
is the *baseline* form: ``v({})`` is the all-point-in-time run, itself an
intervention, rather than a natural, un-intervened regime.

This module is deliberately numeric and side-effect free: it never fits a
model, never touches a frame, and never decides what "permissive" means. The
caller supplies ``evaluate(selection) -> score``; everything here is the
combinatorics, and the combinatorics live in the neutral
:mod:`panelary._internal._shapley` core (shared with
:func:`panelary.select.refit_shapley`). That is what makes it testable against
the Shapley axioms, and reusable for any notion of component, mode and score.

**Replicates.** ``evaluate`` may return one score or a 1-D vector of
*independent replicate* scores -- one per seed, the same seeds in the same
order for every subset. Shapley is linear in ``v``, so each replicate is solved
exactly as its own game and efficiency holds within each replicate; the report
gives the per-replicate values and their median and range per component.

Notes
-----
Cost is ``2 ** k`` evaluations of ``evaluate`` (``method="exact"``), memoised
so each subset is scored exactly once. This is exact -- there is no sampling --
and it is why :func:`borrowed_accuracy` refuses more than ``max_components``
components rather than quietly taking exponential time. Beyond that,
``method="permutation"`` is an explicit, opt-in Monte-Carlo estimate that keeps
efficiency exact and reports a standard error per component.

References
----------
Jung, Y., Kasiviswanathan, S., Tian, J., Janzing, D., Bloebaum, P., &
Bareinboim, E. (2022). On measuring causal contributions via do-interventions.
*Proceedings of the 39th International Conference on Machine Learning* (ICML
2022), PMLR 162:10476-10501.

Heskes, T., Sijben, E., Bucur, I. G., & Claassen, T. (2020). Causal Shapley
values: exploiting causal knowledge to explain individual predictions of
complex models. *NeurIPS 2020*.

See Also
--------
panelary.testing.assert_no_lookahead : the yes/no form of the same experiment.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from panelary._internal import _shapley

__all__ = [
    "DEFAULT_MAX_COMPONENTS",
    "BorrowedAccuracyReport",
    "Component",
    "borrowed_accuracy",
    "resolve_modes",
]

#: Refusal threshold for :func:`borrowed_accuracy`. ``2 ** 12 == 4096``
#: evaluations is already a long afternoon of backtests; beyond it the caller
#: should group components, not wait.
DEFAULT_MAX_COMPONENTS: int = 12

#: ``schema`` tag of :meth:`BorrowedAccuracyReport.to_dict`.
_SCHEMA = "panelary.BorrowedAccuracyReport/1"


@dataclass(frozen=True)
class Component:
    """One pipeline stage, runnable in two modes.

    The two callables are opaque to this module -- it never calls them. They
    exist so that a caller can keep the pair together and let
    :func:`resolve_modes` pick one per subset while building ``evaluate``.

    Parameters
    ----------
    name : str
        Identifier, unique within a run. It is the key in
        :attr:`BorrowedAccuracyReport.attribution`.
    permissive : callable
        The leaky mode: fit once, on everything.
    point_in_time : callable
        The honest mode: refit per fold, on training rows only.
    description : str, optional
        Free text for the report.

    Examples
    --------
    >>> scaler = Component("scaler", permissive=lambda: "all", point_in_time=lambda: "train")
    >>> scaler.mode(permissive=True)()
    'all'
    """

    name: str
    permissive: Callable[..., Any]
    point_in_time: Callable[..., Any]
    description: str = ""

    def mode(self, permissive: bool) -> Callable[..., Any]:
        """Return :attr:`permissive` if ``permissive`` else :attr:`point_in_time`.

        Parameters
        ----------
        permissive : bool
            Which mode to select.

        Returns
        -------
        callable
        """
        return self.permissive if permissive else self.point_in_time


def resolve_modes(
    components: Iterable[Component], selection: Iterable[str]
) -> dict[str, Callable[..., Any]]:
    """Map each component name to the callable its mode selects.

    Parameters
    ----------
    components : iterable of Component
        The pipeline stages.
    selection : iterable of str
        Names to run permissively; every other component runs point-in-time.

    Returns
    -------
    dict of str to callable
        One entry per component, in the order given.

    Raises
    ------
    ValueError
        If ``selection`` names something that is not a component.

    Examples
    --------
    >>> c = Component("a", permissive=lambda: 1, point_in_time=lambda: 0)
    >>> resolve_modes([c], {"a"})["a"]()
    1
    >>> resolve_modes([c], set())["a"]()
    0
    """
    chosen = frozenset(selection)
    comps = list(components)
    known = {c.name for c in comps}
    unknown = chosen - known
    if unknown:
        raise ValueError(
            f"selection names unknown component(s) {sorted(unknown)}; "
            f"known components are {sorted(known)}."
        )
    return {c.name: c.mode(c.name in chosen) for c in comps}


@dataclass(frozen=True)
class BorrowedAccuracyReport:
    """Borrowed accuracy and its Shapley decomposition.

    With a scalar ``evaluate`` and ``method="exact"`` -- the original form --
    only the first eight attributes carry information and the rest keep their
    defaults.

    Attributes
    ----------
    total : float
        ``v(all permissive) - v(all point-in-time)``, sign-oriented so that a
        **positive** value always means accuracy was borrowed, whichever way
        the score points. With replicates, the mean over replicates.
    attribution : dict of str to float
        Shapley value per component. Sums to :attr:`total` (efficiency). With
        replicates, the mean over replicates of the per-replicate values --
        itself the exact Shapley value of the replicate-mean game, which is
        why it, and not the median, is the headline: medians need not sum to
        anything.
    n_evaluations : int
        Distinct subsets passed to ``evaluate``: ``2 ** k`` for ``exact``, at
        most ``n_permutations * (k - 1) + 2`` for ``permutation``.
    components : tuple of str
        Component names, in the order supplied.
    permissive_score, point_in_time_score : float
        The raw ``v(N)`` and ``v({})``, in the caller's own sign convention
        (means over replicates, when there are replicates).
    higher_is_better : bool
        The convention the score was reported in.
    coalition_values : dict of frozenset to float
        Every scored ``v(S)``, raw (the replicate mean, with replicates), for
        inspection. Keyed by the set of permissively-run component names.
        Complete (``2 ** k`` entries) for ``exact``; only the visited subsets
        for ``permutation``.
    method : str
        ``"exact"`` or ``"permutation"``.
    n_permutations : int or None
        Orderings sampled, for ``permutation``.
    seed : int or None
        Sampler seed, for ``permutation``.
    standard_error : dict of str to float, or None
        Monte-Carlo standard error of each :attr:`attribution` entry, for
        ``permutation``; ``None`` for ``exact``, which has none.
    n_replicates : int or None
        Replicate scores per subset; ``None`` when ``evaluate`` returned a
        scalar.
    replicate_totals : tuple of float
        :attr:`total` per replicate (empty without replicates).
    replicate_attribution : dict of str to tuple of float
        Shapley value per component per replicate. Each replicate's values sum
        to that replicate's total (empty without replicates).
    replicate_coalition_values : dict of frozenset to tuple of float
        Every scored ``v(S)``, per replicate (empty without replicates).
    """

    total: float
    attribution: dict[str, float]
    n_evaluations: int
    components: tuple[str, ...]
    permissive_score: float
    point_in_time_score: float
    higher_is_better: bool = True
    coalition_values: dict[frozenset[str], float] = field(
        default_factory=dict, repr=False
    )
    method: str = "exact"
    n_permutations: int | None = None
    seed: int | None = None
    standard_error: dict[str, float] | None = None
    n_replicates: int | None = None
    replicate_totals: tuple[float, ...] = ()
    replicate_attribution: dict[str, tuple[float, ...]] = field(
        default_factory=dict, repr=False
    )
    replicate_coalition_values: dict[frozenset[str], tuple[float, ...]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_components(self) -> int:
        """Number of components, ``k``."""
        return len(self.components)

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """``(name, phi)`` pairs, largest borrower first."""
        return sorted(self.attribution.items(), key=lambda kv: -kv[1])

    def share(self, name: str) -> float:
        """Fraction of :attr:`total` attributed to ``name``.

        Parameters
        ----------
        name : str
            A component name.

        Returns
        -------
        float
            ``phi_name / total``, or ``nan`` when ``total`` is zero.

        Raises
        ------
        KeyError
            If ``name`` is not a component.
        """
        phi = self.attribution[name]
        return phi / self.total if self.total != 0.0 else math.nan

    # ----------------------------------------------------------------- #
    # Replicate summaries
    # ----------------------------------------------------------------- #
    def _per_replicate(self, name: str) -> tuple[float, ...]:
        if self.n_replicates is None:
            return (self.attribution[name],)
        return self.replicate_attribution[name]

    @property
    def attribution_median(self) -> dict[str, float]:
        """Median over replicates of each component's Shapley value.

        Without replicates this is :attr:`attribution`. Medians are robust to
        one bad seed but, unlike :attr:`attribution`, need not sum to
        :attr:`total`.
        """
        return {
            name: float(np.median(self._per_replicate(name)))
            for name in self.components
        }

    @property
    def attribution_range(self) -> dict[str, tuple[float, float]]:
        """``(min, max)`` over replicates of each component's Shapley value."""
        out: dict[str, tuple[float, float]] = {}
        for name in self.components:
            vals = self._per_replicate(name)
            out[name] = (float(min(vals)), float(max(vals)))
        return out

    @property
    def total_median(self) -> float:
        """Median over replicates of :attr:`total` (``total`` itself without)."""
        if self.n_replicates is None:
            return self.total
        return float(np.median(self.replicate_totals))

    @property
    def total_range(self) -> tuple[float, float]:
        """``(min, max)`` over replicates of :attr:`total`."""
        if self.n_replicates is None:
            return (self.total, self.total)
        return (float(min(self.replicate_totals)), float(max(self.replicate_totals)))

    # ----------------------------------------------------------------- #
    # Rendering
    # ----------------------------------------------------------------- #
    def __str__(self) -> str:
        direction = "higher is better" if self.higher_is_better else "lower is better"
        width = max((len(n) for n in self.components), default=1)
        width = max(width, len("point-in-time"))
        if self.method == "permutation":
            how = (
                f"{self.n_evaluations} evaluations, {self.n_permutations} "
                f"sampled orderings, {direction}"
            )
        else:
            how = f"{self.n_evaluations} evaluations, {direction}"
        reps = (
            ""
            if self.n_replicates is None
            else f", {self.n_replicates} replicates (mean shown)"
        )
        lines = [
            f"Borrowed accuracy: {self.total:+.6g}"
            f"  ({self.n_components} components, {how}{reps})",
            f"  {'permissive':<{width}}  {self.permissive_score:+12.6g}",
            f"  {'point-in-time':<{width}}  {self.point_in_time_score:+12.6g}",
        ]
        if self.n_replicates is not None:
            lo, hi = self.total_range
            lines.append(
                f"  {'total median':<{width}}  {self.total_median:+12.6g}"
                f"  [{lo:+.4g}, {hi:+.4g}]"
            )
        if self.components:
            if self.method == "permutation":
                lines.append(
                    "  attribution (permutation Shapley, +/- Monte-Carlo SE; "
                    "sums to total):"
                )
            else:
                lines.append("  attribution (exact Shapley):")
            medians = self.attribution_median
            ranges = self.attribution_range
            for name, phi in self.ranked:
                share = self.share(name)
                pct = "     -" if math.isnan(share) else f"{share:6.1%}"
                line = f"    {name:<{width}}  {phi:+12.6g}  {pct}"
                if self.standard_error is not None:
                    line += f"  +/- {self.standard_error[name]:.2g}"
                if self.n_replicates is not None:
                    lo, hi = ranges[name]
                    line += f"  median {medians[name]:+.4g} [{lo:+.4g}, {hi:+.4g}]"
                lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe plain dict of the report.

        Top-level ``schema`` is ``"panelary.BorrowedAccuracyReport/1"`` and
        ``produced_by`` names the function and Panelary version. Non-finite
        floats (a standard error from a single pair, say) become ``None``.
        Coalitions are listed in bitmask order over :attr:`components`, each
        with its permissive members in component order. There is no
        timestamp, so equal reports serialise to equal bytes.

        Returns
        -------
        dict
        """
        from panelary import __version__

        f = _shapley.json_float
        names = self.components
        coalitions: list[dict[str, Any]] = []
        for mask in range(1 << len(names)):
            key = _shapley.members(mask, names)
            if key not in self.coalition_values:
                continue
            entry: dict[str, Any] = {
                "permissive": [n for n in names if n in key],
                "value": f(self.coalition_values[key]),
            }
            if self.n_replicates is not None:
                entry["replicates"] = [
                    f(x) for x in self.replicate_coalition_values[key]
                ]
            coalitions.append(entry)

        out: dict[str, Any] = {
            "schema": _SCHEMA,
            "produced_by": f"panelary.leakage.borrowed_accuracy@{__version__}",
            "components": list(names),
            "higher_is_better": self.higher_is_better,
            "method": self.method,
            "n_evaluations": self.n_evaluations,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
            "n_replicates": self.n_replicates,
            "total": f(self.total),
            "permissive_score": f(self.permissive_score),
            "point_in_time_score": f(self.point_in_time_score),
            "attribution": {n: f(self.attribution[n]) for n in names},
            "standard_error": (
                None
                if self.standard_error is None
                else {n: f(self.standard_error[n]) for n in names}
            ),
            "coalitions": coalitions,
        }
        if self.n_replicates is not None:
            medians = self.attribution_median
            ranges = self.attribution_range
            out["replicate_totals"] = [f(x) for x in self.replicate_totals]
            out["replicate_attribution"] = {
                n: [f(x) for x in self.replicate_attribution[n]] for n in names
            }
            out["attribution_summary"] = {
                n: {
                    "median": f(medians[n]),
                    "min": f(ranges[n][0]),
                    "max": f(ranges[n][1]),
                }
                for n in names
            }
            lo, hi = self.total_range
            out["total_summary"] = {
                "median": f(self.total_median),
                "min": f(lo),
                "max": f(hi),
            }
        return out

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict` as JSON; byte-deterministic when ``indent is None``.

        Parameters
        ----------
        indent : int, optional
            Pretty-print indent. ``None`` (default) gives the compact,
            key-sorted canonical form.

        Returns
        -------
        str
        """
        return _shapley.canonical_json(self.to_dict(), indent=indent)


def _validated_names(components: Iterable[Component | str]) -> tuple[str, ...]:
    """Extract unique component names, preserving order."""
    names: list[str] = []
    for item in components:
        name = item.name if isinstance(item, Component) else item
        if not isinstance(name, str):
            raise TypeError(
                "each component must be a Component or a str name, got "
                f"{type(item).__name__!r}."
            )
        names.append(name)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"component names must be unique; repeated: {duplicates}. "
            "Shapley attribution is keyed by name, so a duplicate would be "
            "two players sharing one slot."
        )
    return tuple(names)


def _as_scores(value: Any, selection: frozenset[str]) -> np.ndarray:
    """Coerce one ``evaluate`` return to finite float64: 0-d, or 1-D replicates."""
    shown = set(selection) or "{}"
    if isinstance(value, (str, bytes)):
        # float("0.5") would succeed, and quietly turn a bug into a score.
        raise TypeError(
            f"evaluate({shown}) returned {value!r}, which is not a real number."
        )
    if np.ndim(value) == 0:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"evaluate({shown}) returned {value!r}, which is not a real number."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"evaluate({shown}) returned {score!r}; "
                "borrowed accuracy needs a finite score for every subset."
            )
        return np.asarray(score, dtype=np.float64)

    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise TypeError(
            f"evaluate({shown}) returned {value!r}, which is neither a real "
            "number nor a vector of real replicate scores."
        )
    if raw.ndim != 1:
        raise ValueError(
            f"evaluate({shown}) returned an array of shape {raw.shape}; return a "
            "scalar, or a 1-D vector of independent replicate scores (one per "
            "seed)."
        )
    if raw.size == 0:
        raise ValueError(
            f"evaluate({shown}) returned an empty vector; a replicate vector "
            "needs at least one score."
        )
    scores = raw.astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(
            f"evaluate({shown}) returned {scores.tolist()!r}; "
            "borrowed accuracy needs a finite score for every subset and replicate."
        )
    return scores


def borrowed_accuracy(
    components: Iterable[Component | str],
    evaluate: Callable[[frozenset[str]], Any],
    *,
    higher_is_better: bool = True,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    method: str = "exact",
    n_permutations: int | None = None,
    seed: int = 0,
) -> BorrowedAccuracyReport:
    r"""Measure borrowed accuracy and attribute it by Shapley value.

    ``evaluate(S)`` must return the score obtained when exactly the components
    named in ``S`` run **permissively** (fit once, on everything) and every
    other component runs **point-in-time** (refit per fold, on training rows
    only). Borrowed accuracy is then ``v(N) - v({})``, oriented by
    ``higher_is_better`` so a positive total always reads "borrowed".

    Attribution is the Shapley value

    .. math::

        \phi_i = \sum_{S \subseteq N \setminus \{i\}}
                 \frac{|S|!\,(k-|S|-1)!}{k!} \bigl( v(S \cup \{i\}) - v(S) \bigr)

    of an interventional (do-)Shapley game in the sense of Jung et al. (ICML
    2022): each ``v(S)`` is the pipeline *run* under the intervention "these
    stages permissive, the rest point-in-time", so nothing has to be identified
    or estimated. With ``method="exact"`` (the default) it is computed from
    every one of the ``2 ** k`` subsets. Nothing is sampled, so
    ``sum(attribution.values()) == total`` holds to floating-point tolerance --
    the efficiency axiom, and the property that makes this a decomposition
    rather than a set of ablations.

    Parameters
    ----------
    components : iterable of Component or str
        The pipeline stages, ``k`` of them. Only the names are used here; pass
        :class:`Component` objects when you want :func:`resolve_modes` to pick
        the modes for you while building ``evaluate``.
    evaluate : callable
        ``evaluate(selection: frozenset[str]) -> float | vector``. Memoised:
        each subset is passed at most once, the empty subset first. It must be
        deterministic -- the same subset must score the same, or the
        decomposition is meaningless.

        It may return a **1-D vector of independent replicate scores** instead
        of a float: one per seed, the same seeds in the same order for every
        subset. Replicates must be independent reruns (a fresh panel draw, a
        fresh model seed). Per-fold values of a *pooled* metric do not
        qualify: pooling is nonlinear, so they are not replicates of the
        pooled score.
    higher_is_better : bool, default=True
        Whether a larger score is a better one. When ``False`` (an error
        metric, say) every difference is negated, so ``total > 0`` still means
        the permissive run was flattered by leakage.
    max_components : int, default=12
        With ``method="exact"``, refuse more components than this rather than
        spend ``2 ** k`` evaluations. ``k = 12`` is 4096 backtests. Does not
        apply to ``method="permutation"``, whose cost ``n_permutations`` sets.
    method : {"exact", "permutation"}, default="exact"
        ``"permutation"`` is an opt-in Monte-Carlo estimate over sampled
        orderings of the components, drawn in antithetic (reversed) pairs.
        Each ordering's marginal contributions telescope to the total, so
        efficiency still holds to floating-point tolerance; each component
        gets a Monte-Carlo standard error. It is never selected for you.
    n_permutations : int, optional
        Orderings to sample; required for, and only for,
        ``method="permutation"``. Even, and at least 4. Cost is at most
        ``n_permutations * (k - 1) + 2`` distinct evaluations.
    seed : int, default=0
        Seed of the ordering sampler (``method="permutation"`` only).

    Returns
    -------
    BorrowedAccuracyReport
        ``total``, the per-component ``attribution``, ``n_evaluations``, the
        scored characteristic function and -- with replicates -- the
        per-replicate values and their medians and ranges.

    Raises
    ------
    ValueError
        If ``components`` is empty, contains duplicate names, is longer than
        ``max_components`` under ``method="exact"``; if ``method`` or
        ``n_permutations`` is invalid; or if ``evaluate`` returns a non-finite
        score, or replicate vectors of differing length.
    TypeError
        If a component is neither a :class:`Component` nor a ``str``, or if
        ``evaluate`` returns something that is not a real number or a vector
        of them.

    Notes
    -----
    **Cost:** ``2 ** k`` calls to ``evaluate`` (``k = len(components)``) for
    ``exact``: 8 for 3 components, 64 for 6, 1024 for 10. Each call is
    typically a full walk-forward backtest, so the evaluation dominates; the
    combinatorics here are free. ``permutation`` at ``k = 20`` and
    ``n_permutations = 50`` costs at most 952 evaluations, against 1,048,576.

    References
    ----------
    Jung, Y., Kasiviswanathan, S., Tian, J., Janzing, D., Bloebaum, P., &
    Bareinboim, E. (2022). On measuring causal contributions via
    do-interventions. *ICML 2022*, PMLR 162:10476-10501.

    Examples
    --------
    A game where one component leaks 0.1 and the other leaks nothing:

    >>> def evaluate(selection):
    ...     return 0.1 if "scaler" in selection else 0.0
    >>> report = borrowed_accuracy(["scaler", "imputer"], evaluate)
    >>> round(report.total, 12)
    0.1
    >>> {k: round(v, 12) for k, v in report.attribution.items()}
    {'scaler': 0.1, 'imputer': 0.0}
    >>> report.n_evaluations
    4

    A game where neither leaks alone but both leak together -- Shapley splits
    it evenly, where one-at-a-time ablation would attribute zero to each:

    >>> def joint(selection):
    ...     return 0.1 if {"a", "b"} <= selection else 0.0
    >>> {k: round(v, 12) for k, v in borrowed_accuracy(["a", "b"], joint).attribution.items()}
    {'a': 0.05, 'b': 0.05}

    Replicates -- one score per seed -- give a range per component:

    >>> def per_seed(selection):
    ...     return [0.10, 0.12, 0.08] if "scaler" in selection else [0.0, 0.0, 0.0]
    >>> r = borrowed_accuracy(["scaler", "imputer"], per_seed)
    >>> r.n_replicates, round(r.attribution_median["scaler"], 12)
    (3, 0.1)
    >>> tuple(round(x, 12) for x in r.attribution_range["scaler"])
    (0.08, 0.12)
    """
    names = _validated_names(components)
    k = len(names)
    if k == 0:
        raise ValueError(
            "borrowed_accuracy needs at least one component; with none there "
            "is no pipeline to run two ways."
        )
    _shapley.validate_method(method, n_permutations, caller="borrowed_accuracy")
    if method == "exact" and k > max_components:
        example_m = 50
        raise ValueError(
            f"{k} components would need 2**{k} = {2**k} evaluations, above "
            f"max_components={max_components} (2**{max_components} = "
            f"{2**max_components}). Group components into coarser stages; or "
            "opt in to sampling with method='permutation', "
            f"n_permutations=M (at most M*(k-1)+2 evaluations -- "
            f"{_shapley.permutation_cost_bound(k, example_m)} for M={example_m} -- "
            "efficiency still exact, a standard error per component); or raise "
            "max_components deliberately and budget the time."
        )

    orient = 1.0 if higher_is_better else -1.0

    # --- the value function over bitmasks ---------------------------------- #
    # Memoised by the core, so a subset is never scored twice and
    # `n_evaluations` counts real work. The first call is always the empty
    # subset; its shape (scalar, or R replicates) is the one every later
    # subset must match.
    reference: list[tuple[int, ...]] = []

    def value(mask: int) -> np.ndarray:
        selection = _shapley.members(mask, names)
        scores = _as_scores(evaluate(selection), selection)
        if not reference:
            reference.append(scores.shape)
        elif scores.shape != reference[0]:
            first = reference[0]
            raise ValueError(
                f"evaluate({set(selection) or '{}'}) returned "
                f"{_describe(scores.shape)} but evaluate({{}}) returned "
                f"{_describe(first)}; every subset must be scored on the same "
                "replicates (one score per seed, the same seeds in the same order)."
            )
        return scores

    sol = _shapley.solve(
        value, k, method=method, n_permutations=n_permutations, seed=seed
    )

    full = (1 << k) - 1
    phi = orient * sol.phi  # (k,) or (k, R)
    se = sol.standard_error  # orientation does not change a standard error
    v_empty = sol.values[0]
    v_full = sol.values[full]
    totals = orient * (v_full - v_empty)

    replicated = v_empty.ndim == 1
    if replicated:
        n_rep = int(v_empty.shape[0])
        attribution = {n: float(phi[i].mean()) for i, n in enumerate(names)}
        standard_error = (
            None
            if se is None
            else {
                # SE of the replicate-mean estimate: each unit's samples are
                # averaged over replicates before their spread is taken.
                n: _mean_se(sol, i)
                for i, n in enumerate(names)
            }
        )
        return BorrowedAccuracyReport(
            total=float(totals.mean()),
            attribution=attribution,
            n_evaluations=sol.n_evaluations,
            components=names,
            permissive_score=float(v_full.mean()),
            point_in_time_score=float(v_empty.mean()),
            higher_is_better=higher_is_better,
            coalition_values={
                _shapley.members(m, names): float(v.mean())
                for m, v in sol.values.items()
            },
            method=sol.method,
            n_permutations=sol.n_permutations,
            seed=sol.seed,
            standard_error=standard_error,
            n_replicates=n_rep,
            replicate_totals=tuple(float(t) for t in totals),
            replicate_attribution={
                n: tuple(float(x) for x in phi[i]) for i, n in enumerate(names)
            },
            replicate_coalition_values={
                _shapley.members(m, names): tuple(float(x) for x in v)
                for m, v in sol.values.items()
            },
        )

    return BorrowedAccuracyReport(
        total=float(totals),
        attribution={n: float(phi[i]) for i, n in enumerate(names)},
        n_evaluations=sol.n_evaluations,
        components=names,
        permissive_score=float(v_full),
        point_in_time_score=float(v_empty),
        higher_is_better=higher_is_better,
        coalition_values={
            _shapley.members(m, names): float(v) for m, v in sol.values.items()
        },
        method=sol.method,
        n_permutations=sol.n_permutations,
        seed=sol.seed,
        standard_error=(
            None if se is None else {n: float(se[i]) for i, n in enumerate(names)}
        ),
    )


def _describe(shape: tuple[int, ...]) -> str:
    return "a scalar" if shape == () else f"{shape[0]} replicate scores"


def _mean_se(sol: _shapley.ShapleySolution, i: int) -> float:
    """Monte-Carlo SE of the replicate-mean Shapley value of player ``i``."""
    assert sol.samples is not None
    return float(_shapley.mc_standard_error(sol.samples[:, i].mean(axis=-1)))
