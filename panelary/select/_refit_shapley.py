"""Refit Shapley: what each feature group is worth to the learner.

"What is this feature group -- or this dataset -- worth to my pipeline?" has
two partial answers elsewhere in Panelary, and neither can split credit between
redundant inputs:

* :func:`~panelary.select.mda` permutes one feature against a *fixed* fitted
  model. Substitutes mask each other: permute one of two near-duplicates and
  the other still carries the signal, so both can look worthless.
* :func:`~panelary.explain.joint_group_shap` intervenes on the model's inputs
  while holding the model fixed. It says what *this model* uses, not what the
  data offers the learner.

:func:`refit_shapley` plays a cooperative game whose players are feature groups
and whose value ``v(S)`` is the purged-CV score of the model **refit** on the
columns of ``S``. It is an interventional (do-)Shapley value on the training
procedure (Jung et al., ICML 2022) -- every coalition is executed, nothing is
estimated -- so the Shapley axioms hold for the thing you actually care about:
efficiency (the parts sum to ``v(all) - v(none)``), null player (a group the
learner cannot use gets zero) and symmetry (two interchangeable groups split
their joint worth instead of both scoring zero).

The combinatorics are the neutral core in :mod:`panelary._internal._shapley`,
shared with :func:`panelary.leakage.borrowed_accuracy`.

Leakage contract
----------------
``leakage_safe``: every fit sees one fold's training rows only, through the
caller's purged splitter; the baseline mean is the training fold's; nothing is
fitted on the whole panel. The features themselves are taken as given -- they
must already be point-in-time (see :func:`panelary.leakage.audit` and
:func:`panelary.testing.assert_no_lookahead`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from panelary._internal import _shapley
from panelary.core.panel_frame import PanelFrame, as_panel
from panelary.select._methods import _clone_estimator, _is_classifier

__all__ = [
    "DEFAULT_MAX_PLAYERS",
    "PeriodAttribution",
    "RefitShapleyReport",
    "refit_shapley",
]

#: Exact refit Shapley refuses more players than this: ``2 ** 12 = 4096``
#: coalitions, each a full purged-CV refit.
DEFAULT_MAX_PLAYERS: int = 12

_SCHEMA = "panelary.RefitShapleyReport/1"

Scorer = Callable[[np.ndarray, np.ndarray], float]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``1 - SSE/SST`` about the test-fold mean (sklearn's ``r2_score``)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    sst = float(((yt - yt.mean()) ** 2).sum())
    if sst <= 0.0:
        return math.nan
    return 1.0 - float(((yt - yp) ** 2).sum()) / sst


def _neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return -float(((yt - yp) ** 2).mean())


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


_SCORERS: dict[str, Scorer] = {"r2": _r2, "neg_mse": _neg_mse, "accuracy": _accuracy}


def _resolve_scorer(scoring: str | Scorer | None, is_clf: bool) -> tuple[str, Scorer]:
    if scoring is None:
        scoring = "accuracy" if is_clf else "r2"
    if isinstance(scoring, str):
        if scoring not in _SCORERS:
            raise ValueError(
                f"refit_shapley: scoring must be one of {sorted(_SCORERS)} or a "
                f"callable (y_true, y_pred) -> float, got {scoring!r}."
            )
        return scoring, _SCORERS[scoring]
    if callable(scoring):
        return getattr(scoring, "__name__", "callable"), scoring
    raise TypeError(
        "refit_shapley: scoring must be a str or a callable, got "
        f"{type(scoring).__name__}."
    )


# --------------------------------------------------------------------------- #
# Estimators and seeds
# --------------------------------------------------------------------------- #
def _is_factory(estimator: Any) -> bool:
    return not hasattr(estimator, "fit") and callable(estimator)


def _seed_into(est: Any, seed: int) -> Any:
    """Set ``random_state=seed`` on a fresh estimator, or refuse."""
    params = est.get_params() if hasattr(est, "get_params") else {}
    if "random_state" in params:
        est.set_params(random_state=seed)
    elif hasattr(est, "random_state"):
        est.random_state = seed
    else:
        raise ValueError(
            f"refit_shapley: seeds= were given but {type(est).__name__} has no "
            "`random_state` to set. Pass an estimator that takes one, or a "
            "factory `make(seed) -> estimator` as `estimator`."
        )
    return est


def _make(estimator: Any, seed: int | None) -> Any:
    """A fresh, unfitted estimator for one fit, seeded for its replicate."""
    if _is_factory(estimator):
        return estimator(seed)
    est = _clone_estimator(estimator)
    return est if seed is None else _seed_into(est, seed)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _resolve_groups(
    groups: Mapping[str, Sequence[str]] | Sequence[str],
) -> dict[str, tuple[str, ...]]:
    if isinstance(groups, str):
        raise TypeError(
            "refit_shapley: groups must be a mapping {name: [columns]} or a list "
            f"of column names, got the string {groups!r}."
        )
    if isinstance(groups, Mapping):
        out: dict[str, tuple[str, ...]] = {}
        for name, cols in groups.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"refit_shapley: group names must be str, got {name!r}."
                )
            if isinstance(cols, str):
                cols = [cols]
            cols = tuple(cols)
            if not cols:
                raise ValueError(
                    f"refit_shapley: group {name!r} is empty; a player with no "
                    "columns is a null player by construction -- drop it."
                )
            out[name] = cols
    else:
        cols_list = list(groups)
        dupes = sorted({c for c in cols_list if cols_list.count(c) > 1})
        if dupes:
            raise ValueError(f"refit_shapley: repeated feature(s) {dupes}.")
        out = {c: (c,) for c in cols_list}
    if not out:
        raise ValueError("refit_shapley: needs at least one group (player).")
    return out


def _check_columns(
    panel: PanelFrame,
    target: str,
    groups: dict[str, tuple[str, ...]],
    base: tuple[str, ...],
) -> None:
    seen: dict[str, str] = {}
    for name, cols in groups.items():
        for c in cols:
            if c in seen:
                raise ValueError(
                    f"refit_shapley: column {c!r} is in both group {seen[c]!r} and "
                    f"group {name!r}; groups must be disjoint, or one column "
                    "would be two players."
                )
            seen[c] = name
    overlap = sorted(set(base) & set(seen))
    if overlap:
        raise ValueError(
            f"refit_shapley: base_features {overlap} are also in a group; a base "
            "feature is in every coalition, so it cannot be a player."
        )
    keys = {panel.entity_col, panel.time_col, target}
    schema = panel.schema
    for c in (*seen, *base):
        if c not in schema:
            raise ValueError(
                f"refit_shapley: column {c!r} not found in the panel. "
                f"Available columns: {panel.columns}."
            )
        if c in keys:
            raise ValueError(
                f"refit_shapley: {c!r} is the target or a panel key and cannot "
                "be a feature."
            )
        if not schema[c].is_numeric():
            raise ValueError(
                f"refit_shapley: feature {c!r} has dtype {schema[c]}; encode it "
                "as numeric first."
            )
    if target not in schema:
        raise ValueError(
            f"refit_shapley: target column {target!r} not found in the panel. "
            f"Available columns: {panel.columns}."
        )


@dataclass(frozen=True)
class _Fold:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    start: Any
    end: Any
    baseline: np.ndarray  # v({}) predictions when there are no base features


def _usable_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x).all(axis=1)
    if y.dtype.kind in "fc":
        ok &= np.isfinite(y)
    elif y.dtype.kind == "O":
        ok &= np.array([v is not None for v in y], dtype=bool)
    return ok


def _baseline_prediction(y_train: np.ndarray, n: int, is_clf: bool) -> np.ndarray:
    """``v({})``'s prediction: the training-fold mean (or majority class)."""
    if is_clf:
        labels, counts = np.unique(y_train, return_counts=True)
        return np.full(n, labels[int(np.argmax(counts))], dtype=labels.dtype)
    return np.full(n, float(np.mean(y_train.astype(np.float64))), dtype=np.float64)


def _materialise_folds(
    panel: PanelFrame,
    cv: Any,
    target: str,
    columns: list[str],
    is_clf: bool,
) -> tuple[list[_Fold], int]:
    """Collect every fold to numpy once. Returns the folds and how many were dropped."""
    time_col = panel.time_col
    folds: list[_Fold] = []
    dropped = 0
    for train, test in cv.split(panel):
        tr = train.lazy().select([*columns, target]).collect()
        te = test.lazy().select([*columns, target, time_col]).collect()
        x_tr = tr.select(columns).to_numpy().astype(np.float64)
        x_te = te.select(columns).to_numpy().astype(np.float64)
        y_tr = tr.get_column(target).to_numpy()
        y_te = te.get_column(target).to_numpy()
        if not is_clf:
            y_tr = y_tr.astype(np.float64)
            y_te = y_te.astype(np.float64)
        ok_tr = _usable_rows(x_tr, y_tr)
        ok_te = _usable_rows(x_te, y_te)
        if ok_tr.sum() < 2 or ok_te.sum() == 0:
            dropped += 1
            continue
        y_tr, y_te = y_tr[ok_tr], y_te[ok_te]
        if is_clf and np.unique(y_tr).size < 2:
            dropped += 1
            continue
        times = te.get_column(time_col).filter(pl.Series(ok_te))
        folds.append(
            _Fold(
                x_train=x_tr[ok_tr],
                y_train=y_tr,
                x_test=x_te[ok_te],
                y_test=y_te,
                start=times.min(),
                end=times.max(),
                baseline=_baseline_prediction(y_tr, int(ok_te.sum()), is_clf),
            )
        )
    return folds, dropped


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodAttribution:
    """One fold's own decomposition (``per_period=True``).

    Attributes
    ----------
    fold : int
        Index among the scored folds, in the splitter's order.
    start, end : object
        First and last time value of the fold's scored test rows.
    attribution : dict of str to float
        Shapley value per group on this fold's score (mean over replicates).
        Sums to :attr:`total`.
    total : float
        ``v_fold(all) - v_fold(none)`` (mean over replicates).
    full_score, baseline_score : float
        This fold's ``v(N)`` and ``v({})`` (means over replicates).
    """

    fold: int
    start: Any
    end: Any
    attribution: dict[str, float]
    total: float
    full_score: float
    baseline_score: float


def _json_time(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return _shapley.json_float(x)
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)


@dataclass(frozen=True)
class RefitShapleyReport:
    """What each feature group is worth to the refit learner, by Shapley value.

    Attributes
    ----------
    attribution : dict of str to float
        Shapley value per group, in score units. Sums to :attr:`total`
        (efficiency). With ``seeds``, the mean over replicates.
    total : float
        ``v(all groups) - v(no groups)``: what the groups are jointly worth
        over the baseline.
    full_score, baseline_score : float
        ``v(N)`` and ``v({})``: the fold-mean score with every group, and of
        the baseline (training-fold mean, or the ``base_features`` model).
    groups : tuple of str
        Player names, in the order supplied.
    group_features : dict of str to tuple of str
        The columns of each group.
    base_features : tuple of str
        Columns in every coalition, including the baseline.
    baseline : str
        What ``v({})`` is: ``"train_mean"``, ``"train_majority"`` or
        ``"base_features"``.
    scoring : str
        The score's name (higher is better).
    n_evaluations : int
        Distinct coalitions scored.
    n_fits : int
        Estimator fits performed (the real cost).
    n_folds : int
        Folds scored (after dropping unusable ones).
    n_folds_dropped : int
        Folds the splitter yielded that had too few usable rows.
    method : str
        ``"exact"`` or ``"permutation"``.
    n_permutations, seed : int or None
        Sampler settings, for ``permutation``.
    standard_error : dict of str to float, or None
        Monte-Carlo standard error of each :attr:`attribution` entry, for
        ``permutation``.
    seeds : tuple of int, or None
        Replicate seeds, when given.
    replicate_attribution : dict of str to tuple of float
        Per-replicate Shapley values (empty without ``seeds``).
    replicate_totals : tuple of float
        Per-replicate totals (empty without ``seeds``).
    per_period : tuple of PeriodAttribution, or None
        Per-fold decomposition, with ``per_period=True``.
    coalition_values : dict of frozenset to float
        Every scored ``v(S)`` (fold mean, replicate mean), keyed by the set of
        groups in the coalition.
    """

    attribution: dict[str, float]
    total: float
    full_score: float
    baseline_score: float
    groups: tuple[str, ...]
    group_features: dict[str, tuple[str, ...]]
    base_features: tuple[str, ...]
    baseline: str
    scoring: str
    n_evaluations: int
    n_fits: int
    n_folds: int
    n_folds_dropped: int = 0
    method: str = "exact"
    n_permutations: int | None = None
    seed: int | None = None
    standard_error: dict[str, float] | None = None
    seeds: tuple[int, ...] | None = None
    replicate_attribution: dict[str, tuple[float, ...]] = field(
        default_factory=dict, repr=False
    )
    replicate_totals: tuple[float, ...] = ()
    per_period: tuple[PeriodAttribution, ...] | None = field(default=None, repr=False)
    coalition_values: dict[frozenset[str], float] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_players(self) -> int:
        """Number of groups, ``k``."""
        return len(self.groups)

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """``(group, phi)`` pairs, most valuable first."""
        return sorted(self.attribution.items(), key=lambda kv: -kv[1])

    def share(self, name: str) -> float:
        """Fraction of :attr:`total` attributed to ``name`` (``nan`` if zero)."""
        phi = self.attribution[name]
        return phi / self.total if self.total != 0.0 else math.nan

    def _per_replicate(self, name: str) -> tuple[float, ...]:
        if self.seeds is None:
            return (self.attribution[name],)
        return self.replicate_attribution[name]

    @property
    def attribution_median(self) -> dict[str, float]:
        """Median over replicates of each group's value (``attribution`` without)."""
        return {g: float(np.median(self._per_replicate(g))) for g in self.groups}

    @property
    def attribution_range(self) -> dict[str, tuple[float, float]]:
        """``(min, max)`` over replicates of each group's value."""
        out: dict[str, tuple[float, float]] = {}
        for g in self.groups:
            vals = self._per_replicate(g)
            out[g] = (float(min(vals)), float(max(vals)))
        return out

    def to_frame(self) -> pl.DataFrame:
        """One row per group, most valuable first.

        Columns ``group``, ``n_features``, ``phi``, ``share``, and -- when they
        exist -- ``se`` (permutation) and ``median`` / ``min`` / ``max``
        (replicates).

        Returns
        -------
        polars.DataFrame
        """
        order = [g for g, _ in self.ranked]
        data: dict[str, Any] = {
            "group": order,
            "n_features": [len(self.group_features[g]) for g in order],
            "phi": [self.attribution[g] for g in order],
            "share": [self.share(g) for g in order],
        }
        if self.standard_error is not None:
            data["se"] = [self.standard_error[g] for g in order]
        if self.seeds is not None:
            med, rng = self.attribution_median, self.attribution_range
            data["median"] = [med[g] for g in order]
            data["min"] = [rng[g][0] for g in order]
            data["max"] = [rng[g][1] for g in order]
        return pl.DataFrame(data)

    def __str__(self) -> str:
        width = max([len(g) for g in self.groups] + [len("baseline")])
        how = (
            f"{self.n_permutations} sampled orderings"
            if self.method == "permutation"
            else "exact"
        )
        reps = "" if self.seeds is None else f", {len(self.seeds)} seeds (mean shown)"
        lines = [
            f"Refit Shapley ({self.scoring}): total {self.total:+.6g} over the "
            f"baseline  ({self.n_players} groups, {how}, {self.n_evaluations} "
            f"coalitions, {self.n_fits} fits, {self.n_folds} folds{reps})",
            f"  {'all groups':<{width}}  {self.full_score:+12.6g}",
            f"  {'baseline':<{width}}  {self.baseline_score:+12.6g}  ({self.baseline})",
        ]
        med, rng = self.attribution_median, self.attribution_range
        for g, phi in self.ranked:
            s = self.share(g)
            pct = "     -" if math.isnan(s) else f"{s:6.1%}"
            line = f"    {g:<{width}}  {phi:+12.6g}  {pct}"
            if self.standard_error is not None:
                line += f"  +/- {self.standard_error[g]:.2g}"
            if self.seeds is not None:
                lo, hi = rng[g]
                line += f"  median {med[g]:+.4g} [{lo:+.4g}, {hi:+.4g}]"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe plain dict of the report.

        ``schema`` is ``"panelary.RefitShapleyReport/1"``; ``produced_by``
        names the function and Panelary version. Non-finite floats become
        ``None``; time values become ISO strings; coalitions are listed in
        bitmask order over :attr:`groups`. No timestamps.

        Returns
        -------
        dict
        """
        from panelary import __version__

        f = _shapley.json_float
        names = self.groups
        coalitions = []
        for mask in range(1 << len(names)):
            key = _shapley.members(mask, names)
            if key in self.coalition_values:
                coalitions.append(
                    {
                        "groups": [g for g in names if g in key],
                        "value": f(self.coalition_values[key]),
                    }
                )
        out: dict[str, Any] = {
            "schema": _SCHEMA,
            "produced_by": f"panelary.select.refit_shapley@{__version__}",
            "groups": list(names),
            "group_features": {g: list(self.group_features[g]) for g in names},
            "base_features": list(self.base_features),
            "baseline": self.baseline,
            "scoring": self.scoring,
            "method": self.method,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
            "seeds": None if self.seeds is None else list(self.seeds),
            "n_evaluations": self.n_evaluations,
            "n_fits": self.n_fits,
            "n_folds": self.n_folds,
            "n_folds_dropped": self.n_folds_dropped,
            "total": f(self.total),
            "full_score": f(self.full_score),
            "baseline_score": f(self.baseline_score),
            "attribution": {g: f(self.attribution[g]) for g in names},
            "standard_error": (
                None
                if self.standard_error is None
                else {g: f(self.standard_error[g]) for g in names}
            ),
            "coalitions": coalitions,
        }
        if self.seeds is not None:
            med, rng = self.attribution_median, self.attribution_range
            out["replicate_totals"] = [f(x) for x in self.replicate_totals]
            out["replicate_attribution"] = {
                g: [f(x) for x in self.replicate_attribution[g]] for g in names
            }
            out["attribution_summary"] = {
                g: {"median": f(med[g]), "min": f(rng[g][0]), "max": f(rng[g][1])}
                for g in names
            }
        if self.per_period is not None:
            out["per_period"] = [
                {
                    "fold": p.fold,
                    "start": _json_time(p.start),
                    "end": _json_time(p.end),
                    "total": f(p.total),
                    "full_score": f(p.full_score),
                    "baseline_score": f(p.baseline_score),
                    "attribution": {g: f(p.attribution[g]) for g in names},
                }
                for p in self.per_period
            ]
        return out

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict` as JSON; byte-deterministic when ``indent is None``."""
        return _shapley.canonical_json(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- #
# The function
# --------------------------------------------------------------------------- #
def refit_shapley(
    estimator: Any,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    y: str,
    cv: Any,
    groups: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    base_features: Sequence[str] = (),
    scoring: str | Scorer | None = None,
    seeds: Sequence[int] | None = None,
    per_period: bool = False,
    method: str = "exact",
    n_permutations: int | None = None,
    seed: int = 0,
    max_players: int = DEFAULT_MAX_PLAYERS,
    entity: str | None = None,
    time: str | None = None,
) -> RefitShapleyReport:
    """Shapley value of each feature group to a model refit through purged CV.

    Players are feature groups (or raw data sources). The value of a coalition
    ``S`` is the fold-mean out-of-sample score of a fresh ``estimator`` fitted
    on each training block using the columns of ``S`` (plus
    ``base_features``), scored on the matching test block of ``cv`` -- the
    same splitter protocol :func:`~panelary.select.mda` scores through. The
    empty coalition predicts the **training-fold** mean of ``y`` (majority
    class for a classifier), or refits on ``base_features`` alone when there
    are any. Every group's value is measured against that baseline, and the
    values sum to ``v(all) - v(baseline)`` (efficiency).

    Unlike permutation importance, the model is refit for every coalition, so
    two redundant groups split their joint worth (symmetry) instead of both
    scoring zero, and interactions between groups are shared rather than
    missed.

    Parameters
    ----------
    estimator : object or callable
        An sklearn-style estimator (``fit`` / ``predict``), cloned for every
        fit (sklearn's ``clone`` when available); or a factory
        ``make(seed: int | None) -> estimator`` returning a fresh, unfitted one.
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel holding every feature and the target. Features are
        taken as computed: nothing here recomputes them.
    y : str
        Target column.
    cv : object
        Splitter with ``split(panel) -> iterable of (train, test)`` PanelFrames,
        e.g. :class:`~panelary.core.model_selection.PurgedKFold`. Materialised
        once and shared by every coalition.
    groups : mapping of str to sequence of str, or sequence of str
        ``{player: [columns]}``; pairwise disjoint, non-empty, numeric. A plain
        list of columns means one player per column.
    base_features : sequence of str, default=()
        Columns present in every coalition, including the baseline -- "what is
        each group worth *on top of* these?".
    scoring : {"r2", "neg_mse", "accuracy"} or callable, optional
        Higher-is-better score; ``callable(y_true, y_pred) -> float``. Defaults
        to ``"r2"`` (regressors) or ``"accuracy"`` (classifiers). ``"r2"`` is
        per fold, about the test fold's own mean.
    seeds : sequence of int, optional
        Independent replicate seeds. Replicate ``r`` sets the estimator's
        ``random_state`` to ``seeds[r]`` (or calls the factory with it) for
        **every** coalition and fold -- common random numbers, so the noise
        cancels in the differences Shapley is built from. The report then
        carries the per-replicate values and their median and range.
    per_period : bool, default=False
        Also decompose each fold's own score. Shapley is linear, so each fold's
        values sum to that fold's total, and the headline is exactly the mean
        of the per-fold values.
    method : {"exact", "permutation"}, default="exact"
        ``"exact"`` scores all ``2 ** k`` coalitions. ``"permutation"`` is the
        opt-in Monte-Carlo estimate over sampled orderings (antithetic pairs,
        efficiency kept, a standard error per group); never chosen for you.
    n_permutations : int, optional
        Orderings to sample; required for, and only for, ``"permutation"``.
        Even, at least 4.
    seed : int, default=0
        Ordering sampler seed (``"permutation"`` only).
    max_players : int, default=12
        With ``method="exact"``, refuse more groups than this.
    entity, time : str, optional
        Panel keys, when ``X`` is a bare polars frame.

    Returns
    -------
    RefitShapleyReport

    Raises
    ------
    ValueError
        On overlapping, empty or unknown groups; a group containing the target
        or a panel key; too many groups for ``exact``; an invalid ``method`` /
        ``n_permutations``; ``seeds`` with an estimator that has no
        ``random_state``; no usable fold; or a non-finite score.
    TypeError
        On a ``groups`` or ``scoring`` of the wrong type.

    Notes
    -----
    **Cost:** (coalitions) x (folds) x (replicates) estimator fits, less the
    fit-free baseline. Exact: ``2 ** k`` coalitions -- 64 for 6 groups, 4096
    for 12. Permutation: at most ``n_permutations * (k - 1) + 2``.

    **What it is, causally:** an interventional (do-)Shapley value on the
    training procedure (Jung et al., ICML 2022). Each coalition is *run*, so
    nothing has to be identified. It measures what the data offers this
    learner under this CV, not a causal effect of a feature on the target.

    References
    ----------
    Jung, Y., Kasiviswanathan, S., Tian, J., Janzing, D., Bloebaum, P., &
    Bareinboim, E. (2022). On measuring causal contributions via
    do-interventions. *ICML 2022*, PMLR 162:10476-10501.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression  # doctest: +SKIP
    >>> from panelary.core.model_selection import PurgedKFold  # doctest: +SKIP
    >>> report = refit_shapley(  # doctest: +SKIP
    ...     LinearRegression(), panel, "ret_fwd", PurgedKFold(5, embargo=1),
    ...     {"momentum": ["mom", "mom_lag"], "vol": ["vol"], "noise": ["noise"]},
    ... )
    >>> report.to_frame()  # doctest: +SKIP
    """
    player_features = _resolve_groups(groups)
    names = tuple(player_features)
    k = len(names)
    base = tuple(base_features)
    _shapley.validate_method(method, n_permutations, caller="refit_shapley")
    if method == "exact" and k > max_players:
        m = 50
        raise ValueError(
            f"refit_shapley: {k} groups would need 2**{k} = {2**k} coalitions, "
            f"each a full purged-CV refit, above max_players={max_players}. "
            "Group your features into fewer, coarser groups (a data source, a "
            "factor family); or opt in to sampling with method='permutation', "
            f"n_permutations=M (at most M*(k-1)+2 coalitions -- "
            f"{_shapley.permutation_cost_bound(k, m)} for M={m}); or raise "
            "max_players deliberately and budget the time."
        )

    panel = as_panel(X, entity, time)
    _check_columns(panel, y, player_features, base)

    seed_tuple: tuple[int, ...] | None = None
    if seeds is not None:
        seed_tuple = tuple(int(s) for s in seeds)
        if not seed_tuple:
            raise ValueError("refit_shapley: seeds= is empty; omit it or pass seeds.")
        if len(set(seed_tuple)) != len(seed_tuple):
            raise ValueError(
                f"refit_shapley: repeated seeds {list(seed_tuple)}; replicates must "
                "be independent reruns."
            )
    seed_list: list[int | None] = [None] if seed_tuple is None else list(seed_tuple)

    probe = _make(estimator, seed_list[0])  # fails early on an unseedable estimator
    if not (hasattr(probe, "fit") and hasattr(probe, "predict")):
        raise TypeError(
            "refit_shapley: estimator must provide fit/predict (or be a factory "
            f"returning one); got {type(probe).__name__}."
        )
    is_clf = _is_classifier(probe)
    if not is_clf and not panel.schema[y].is_numeric():
        raise ValueError(
            f"refit_shapley: target {y!r} has dtype {panel.schema[y]}, but the "
            "estimator is not a classifier; a regression target must be numeric."
        )
    score_name, scorer = _resolve_scorer(scoring, is_clf)

    columns = [*base, *(c for g in names for c in player_features[g])]
    col_index = {c: i for i, c in enumerate(columns)}
    base_idx = [col_index[c] for c in base]
    group_idx = [[col_index[c] for c in player_features[g]] for g in names]

    folds, n_dropped = _materialise_folds(panel, cv, y, columns, is_clf)
    if not folds:
        raise ValueError(
            "refit_shapley: no fold had usable rows (at least 2 training rows and "
            "1 test row with a finite target and finite features"
            + (", and two classes" if is_clf else "")
            + "). Check the splitter, the panel size and missing values."
        )

    n_fits = 0

    def value(mask: int) -> np.ndarray:
        nonlocal n_fits
        cols = list(base_idx)
        for i in range(k):
            if mask >> i & 1:
                cols.extend(group_idx[i])
        out = np.empty((len(seed_list), len(folds)), dtype=np.float64)
        for r, s in enumerate(seed_list):
            for fi, fold in enumerate(folds):
                if cols:
                    est = _make(estimator, s)
                    est.fit(fold.x_train[:, cols], fold.y_train)
                    pred = np.asarray(est.predict(fold.x_test[:, cols])).ravel()
                    n_fits += 1
                else:
                    pred = fold.baseline
                score = float(scorer(fold.y_test, pred))
                if not math.isfinite(score):
                    members = sorted(_shapley.members(mask, names)) or "{}"
                    raise ValueError(
                        f"refit_shapley: score {score!r} on fold {fi} for "
                        f"coalition {members} (seed {s}); every coalition needs a "
                        "finite score (an R^2 is undefined on a constant test fold)."
                    )
                out[r, fi] = score
        return out

    sol = _shapley.solve(
        value, k, method=method, n_permutations=n_permutations, seed=seed
    )

    full = (1 << k) - 1
    # phi: (k, R, F). Headline = fold mean, then replicate mean.
    phi_rep = sol.phi.mean(axis=-1)  # (k, R)
    v_empty = sol.values[0]  # (R, F)
    v_full = sol.values[full]
    totals_rep = (v_full - v_empty).mean(axis=-1)  # (R,)

    standard_error = None
    if sol.samples is not None:
        se = _shapley.mc_standard_error(sol.samples.mean(axis=(-2, -1)))
        standard_error = {g: float(se[i]) for i, g in enumerate(names)}

    periods = None
    if per_period:
        phi_fold = sol.phi.mean(axis=1)  # (k, F), mean over replicates
        periods = tuple(
            PeriodAttribution(
                fold=fi,
                start=fold.start,
                end=fold.end,
                attribution={g: float(phi_fold[i, fi]) for i, g in enumerate(names)},
                total=float((v_full[:, fi] - v_empty[:, fi]).mean()),
                full_score=float(v_full[:, fi].mean()),
                baseline_score=float(v_empty[:, fi].mean()),
            )
            for fi, fold in enumerate(folds)
        )

    baseline = "train_majority" if is_clf else "train_mean"
    if base:
        baseline = "base_features"

    return RefitShapleyReport(
        attribution={g: float(phi_rep[i].mean()) for i, g in enumerate(names)},
        total=float(totals_rep.mean()),
        full_score=float(v_full.mean()),
        baseline_score=float(v_empty.mean()),
        groups=names,
        group_features=dict(player_features),
        base_features=base,
        baseline=baseline,
        scoring=score_name,
        n_evaluations=sol.n_evaluations,
        n_fits=n_fits,
        n_folds=len(folds),
        n_folds_dropped=n_dropped,
        method=sol.method,
        n_permutations=sol.n_permutations,
        seed=sol.seed,
        standard_error=standard_error,
        seeds=seed_tuple,
        replicate_attribution=(
            {}
            if seed_tuple is None
            else {g: tuple(float(x) for x in phi_rep[i]) for i, g in enumerate(names)}
        ),
        replicate_totals=(
            () if seed_tuple is None else tuple(float(t) for t in totals_rep)
        ),
        per_period=periods,
        coalition_values={
            _shapley.members(m, names): float(v.mean()) for m, v in sol.values.items()
        },
    )
