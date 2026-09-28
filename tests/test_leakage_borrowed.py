"""Borrowed accuracy and its exact Shapley decomposition.

The Shapley axioms *are* the proof of correctness for
:mod:`panelary.leakage._borrowed`, so most of this file is axioms rather than
examples: efficiency (the attribution sums to the measured gap), null player,
symmetry, additivity on a separable game, and the interaction case that
justifies Shapley over one-at-a-time ablation.

The last test is the real thing: a panel, Panelary's own
:func:`~panelary.core.model_selection.expanding_window_split`, and a pipeline
whose scaler/imputer/selector are each fit either on everything (leaky) or per
fold on training rows only (honest). The leaky pipeline shows positive
out-of-sample R^2 where the honest one shows none; borrowed accuracy is
exactly that difference, and the decomposition says which stage stole it.
"""

from __future__ import annotations

import doctest
import itertools
import json
import math

import numpy as np
import polars as pl
import pytest

from panelary.core.model_selection import expanding_window_split
from panelary.core.panel_frame import PanelFrame
from panelary.leakage import _borrowed
from panelary.leakage._borrowed import (
    DEFAULT_MAX_COMPONENTS,
    BorrowedAccuracyReport,
    Component,
    borrowed_accuracy,
    resolve_modes,
)

# Shapley weights are exact rationals over factorials; the only error is
# floating-point accumulation over at most 2**k terms.
TOL = 1e-12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def counting(fn):
    """Wrap ``fn``, recording every selection it is called with."""
    calls: list[frozenset[str]] = []

    def wrapped(selection: frozenset[str]) -> float:
        calls.append(selection)
        return fn(selection)

    wrapped.calls = calls  # type: ignore[attr-defined]
    return wrapped


def additive_game(weights: dict[str, float]):
    """``v(S) = sum of weights[i] for i in S`` -- a game with no interaction."""

    def evaluate(selection: frozenset[str]) -> float:
        return float(sum(weights[name] for name in selection))

    return evaluate


def random_game(names, seed: int):
    """A dense, arbitrary characteristic function over ``names``."""
    rng = np.random.default_rng(seed)
    table = {
        frozenset(subset): float(rng.standard_normal())
        for r in range(len(names) + 1)
        for subset in itertools.combinations(names, r)
    }

    def evaluate(selection: frozenset[str]) -> float:
        return table[selection]

    return evaluate


def ablation(names, evaluate):
    """One-at-a-time attribution: ``v({i}) - v({})``, the naive alternative."""
    base = evaluate(frozenset())
    return {name: evaluate(frozenset({name})) - base for name in names}


# --------------------------------------------------------------------------- #
# Efficiency: the axiom that makes this a decomposition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7])
def test_efficiency_on_random_games(k: int) -> None:
    """sum(phi) == total, for arbitrary v. Non-negotiable."""
    names = [f"c{i}" for i in range(k)]
    report = borrowed_accuracy(names, random_game(names, seed=100 + k))
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_efficiency_holds_under_both_orientations(higher_is_better: bool) -> None:
    names = ["a", "b", "c", "d"]
    report = borrowed_accuracy(
        names, random_game(names, seed=7), higher_is_better=higher_is_better
    )
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


def test_total_is_the_gap_between_the_two_full_runs() -> None:
    names = ["a", "b", "c"]
    evaluate = random_game(names, seed=11)
    report = borrowed_accuracy(names, evaluate)
    expected = evaluate(frozenset(names)) - evaluate(frozenset())
    assert report.total == pytest.approx(expected, abs=TOL)
    assert report.permissive_score == evaluate(frozenset(names))
    assert report.point_in_time_score == evaluate(frozenset())


# --------------------------------------------------------------------------- #
# Null player, symmetry, additivity
# --------------------------------------------------------------------------- #
def test_null_player_gets_exactly_zero() -> None:
    """A component whose two modes score identically borrows nothing."""
    names = ["leaky", "inert", "other"]
    inner = random_game(["leaky", "other"], seed=3)

    def evaluate(selection: frozenset[str]) -> float:
        # "inert" never changes the score: v(S) depends only on S \ {inert}.
        return inner(frozenset(selection) - {"inert"})

    report = borrowed_accuracy(names, evaluate)
    assert report.attribution["inert"] == pytest.approx(0.0, abs=TOL)
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


def test_symmetric_components_get_equal_phi() -> None:
    """Interchangeable components share the credit exactly."""

    def evaluate(selection: frozenset[str]) -> float:
        # Depends on how many of {a, b} are permissive, never on which.
        n_ab = len({"a", "b"} & selection)
        return 0.3 * n_ab + 0.4 * n_ab**2 + (0.1 if "c" in selection else 0.0)

    report = borrowed_accuracy(["a", "b", "c"], evaluate)
    assert report.attribution["a"] == pytest.approx(report.attribution["b"], abs=TOL)
    assert report.attribution["a"] != pytest.approx(0.0, abs=1e-6)
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


def test_additive_game_recovers_each_weight() -> None:
    """With no interaction, Shapley degenerates to the obvious answer."""
    weights = {"scaler": 0.11, "imputer": -0.02, "encoder": 0.0, "selector": 0.4}
    report = borrowed_accuracy(list(weights), additive_game(weights))
    for name, w in weights.items():
        assert report.attribution[name] == pytest.approx(w, abs=TOL)
    assert report.total == pytest.approx(sum(weights.values()), abs=TOL)


# --------------------------------------------------------------------------- #
# Concentration and interaction
# --------------------------------------------------------------------------- #
def test_concentration_one_component_leaks_a_known_amount() -> None:
    """Exactly one component leaks 0.137; phi recovers it, the rest are zero."""
    leak = 0.137
    names = ["scaler", "imputer", "encoder", "selector"]

    def evaluate(selection: frozenset[str]) -> float:
        return leak if "scaler" in selection else 0.0

    report = borrowed_accuracy(names, evaluate)
    assert report.total == pytest.approx(leak, abs=TOL)
    assert report.attribution["scaler"] == pytest.approx(leak, abs=TOL)
    for name in names[1:]:
        assert report.attribution[name] == pytest.approx(0.0, abs=TOL)
    assert report.share("scaler") == pytest.approx(1.0, abs=TOL)


def test_interaction_is_split_evenly_where_ablation_says_zero() -> None:
    """Two components that only leak *together*.

    This is the test that justifies Shapley over ablation: one-at-a-time
    ablation attributes zero to both, missing the entire gap, while Shapley
    splits the joint effect evenly and still sums to the measured total.
    """
    leak = 0.2
    names = ["a", "b", "c"]

    def evaluate(selection: frozenset[str]) -> float:
        return leak if {"a", "b"} <= selection else 0.0

    report = borrowed_accuracy(names, evaluate)

    assert report.total == pytest.approx(leak, abs=TOL)
    assert report.attribution["a"] == pytest.approx(leak / 2, abs=TOL)
    assert report.attribution["b"] == pytest.approx(leak / 2, abs=TOL)
    assert report.attribution["c"] == pytest.approx(0.0, abs=TOL)
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)

    naive = ablation(names, evaluate)
    assert naive == {"a": 0.0, "b": 0.0, "c": 0.0}
    assert sum(naive.values()) != pytest.approx(report.total, abs=1e-6)


def test_three_way_interaction_splits_three_ways() -> None:
    leak = 0.3
    names = ["a", "b", "c"]

    def evaluate(selection: frozenset[str]) -> float:
        return leak if len(selection) == 3 else 0.0

    report = borrowed_accuracy(names, evaluate)
    for name in names:
        assert report.attribution[name] == pytest.approx(leak / 3, abs=TOL)


# --------------------------------------------------------------------------- #
# Accounting, determinism, orientation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_n_evaluations_is_two_to_the_k_and_no_subset_is_scored_twice(k: int) -> None:
    names = [f"c{i}" for i in range(k)]
    evaluate = counting(random_game(names, seed=k))
    report = borrowed_accuracy(names, evaluate)

    assert report.n_evaluations == 2**k
    assert len(evaluate.calls) == 2**k
    assert len(set(evaluate.calls)) == 2**k  # memoised: every subset exactly once
    assert set(evaluate.calls) == set(report.coalition_values)


def test_deterministic_across_runs() -> None:
    names = ["a", "b", "c", "d"]
    evaluate = random_game(names, seed=42)
    first = borrowed_accuracy(names, evaluate)
    second = borrowed_accuracy(names, evaluate)
    assert first.attribution == second.attribution
    assert first.total == second.total
    assert first.n_evaluations == second.n_evaluations


def test_lower_is_better_flips_the_sign_so_positive_still_means_borrowed() -> None:
    """An error metric: permissive MSE 0.4, point-in-time 0.9 -> borrowed 0.5."""

    def mse(selection: frozenset[str]) -> float:
        return 0.9 - 0.5 * len(selection) / 2

    up = borrowed_accuracy(["a", "b"], mse, higher_is_better=True)
    down = borrowed_accuracy(["a", "b"], mse, higher_is_better=False)

    assert up.total == pytest.approx(-0.5, abs=TOL)
    assert down.total == pytest.approx(0.5, abs=TOL)
    assert down.attribution == {
        name: pytest.approx(-phi, abs=TOL) for name, phi in up.attribution.items()
    }
    # Raw scores are reported in the caller's own convention, unflipped.
    assert down.permissive_score == pytest.approx(0.4, abs=TOL)
    assert down.point_in_time_score == pytest.approx(0.9, abs=TOL)


def test_single_component_is_just_the_gap() -> None:
    report = borrowed_accuracy(["only"], lambda s: 1.0 if s else 0.25)
    assert report.total == pytest.approx(0.75, abs=TOL)
    assert report.attribution == {"only": pytest.approx(0.75, abs=TOL)}
    assert report.n_evaluations == 2


# --------------------------------------------------------------------------- #
# Errors and the small surface around the metric
# --------------------------------------------------------------------------- #
def test_refuses_empty_component_list() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        borrowed_accuracy([], lambda s: 0.0)


def test_refuses_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        borrowed_accuracy(["a", "b", "a"], lambda s: 0.0)


def test_refuses_too_many_components_rather_than_taking_exponential_time() -> None:
    names = [f"c{i}" for i in range(DEFAULT_MAX_COMPONENTS + 1)]
    evaluate = counting(lambda s: 0.0)
    with pytest.raises(ValueError, match=r"2\*\*13 = 8192"):
        borrowed_accuracy(names, evaluate)
    assert evaluate.calls == []  # refused before spending anything

    with pytest.raises(ValueError, match="max_components=2"):
        borrowed_accuracy(["a", "b", "c"], evaluate, max_components=2)


def test_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="finite score"):
        borrowed_accuracy(["a"], lambda s: math.inf if s else 0.0)
    with pytest.raises(ValueError, match="finite score"):
        borrowed_accuracy(["a"], lambda s: math.nan)


def test_rejects_non_numeric_scores() -> None:
    with pytest.raises(TypeError, match="not a real number"):
        borrowed_accuracy(["a"], lambda s: "0.5")  # type: ignore[arg-type]


def test_rejects_non_component_entries() -> None:
    with pytest.raises(TypeError, match="Component or a str"):
        borrowed_accuracy([object()], lambda s: 0.0)  # type: ignore[list-item]


def test_component_mode_and_resolve_modes() -> None:
    a = Component("a", permissive=lambda: "leaky-a", point_in_time=lambda: "pit-a")
    b = Component("b", permissive=lambda: "leaky-b", point_in_time=lambda: "pit-b")

    assert a.mode(permissive=True)() == "leaky-a"
    assert a.mode(permissive=False)() == "pit-a"

    modes = resolve_modes([a, b], {"b"})
    assert list(modes) == ["a", "b"]
    assert modes["a"]() == "pit-a"
    assert modes["b"]() == "leaky-b"

    with pytest.raises(ValueError, match="unknown component"):
        resolve_modes([a, b], {"c"})


def test_components_accepts_component_objects() -> None:
    comps = [
        Component(name, permissive=lambda: None, point_in_time=lambda: None)
        for name in ("scaler", "imputer")
    ]
    report = borrowed_accuracy(comps, additive_game({"scaler": 0.2, "imputer": 0.05}))
    assert report.components == ("scaler", "imputer")
    assert report.attribution["scaler"] == pytest.approx(0.2, abs=TOL)


def test_report_is_readable() -> None:
    report = borrowed_accuracy(
        ["scaler", "imputer"], additive_game({"scaler": 0.2, "imputer": 0.05})
    )
    text = str(report)
    assert "Borrowed accuracy" in text
    assert "scaler" in text and "imputer" in text
    assert "4 evaluations" in text
    assert "exact Shapley" in text
    # Ranked largest borrower first; shares sum to one.
    assert [name for name, _ in report.ranked] == ["scaler", "imputer"]
    assert report.share("scaler") + report.share("imputer") == pytest.approx(1.0)
    assert report.n_components == 2
    assert isinstance(report, BorrowedAccuracyReport)


def test_share_is_nan_when_nothing_was_borrowed() -> None:
    report = borrowed_accuracy(["a", "b"], lambda s: 1.0 if s == {"a"} else 1.0)
    assert report.total == 0.0
    assert math.isnan(report.share("a"))
    assert "-" in str(report)


def test_module_doctests_pass() -> None:
    results = doctest.testmod(_borrowed, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0


# --------------------------------------------------------------------------- #
# End to end: a real panel, real splits, a deliberately leaky pipeline
# --------------------------------------------------------------------------- #
N_FEATURES = 20
SIGNAL = (0, 1)
N_KEEP = 3
RIDGE_LAMBDA = 1e-3


def make_leaky_panel(
    seed: int = 0, n_entities: int = 12, n_times: int = 40
) -> PanelFrame:
    """A drifting panel with two signal features, 18 noise ones, and gaps.

    Three properties make the leakage measurable:

    * the target **drifts** upward in time, so a mean computed over all rows is
      closer to the test block than a mean computed over training rows -- the
      scaler's leak;
    * most features are pure noise, so picking the top few by correlation with
      the target is a **target-aware** choice, and computing that correlation
      over all rows is the selector's leak;
    * a signal feature has missing values, so the fill value is a statistic
      that can be computed honestly or not -- the imputer's leak.
    """
    rng = np.random.default_rng(seed)
    n = n_entities * n_times
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)

    x = rng.standard_normal((n, N_FEATURES))
    beta = np.zeros(N_FEATURES, dtype=np.float64)
    beta[SIGNAL[0]], beta[SIGNAL[1]] = 0.8, 0.5
    y = x @ beta + 0.05 * time + 0.5 * rng.standard_normal(n)
    x[rng.random(n) < 0.15, SIGNAL[0]] = np.nan

    data: dict[str, np.ndarray] = {"entity": entity, "time": time}
    data.update({f"x{j}": x[:, j] for j in range(N_FEATURES)})
    data["y"] = y
    return PanelFrame(pl.DataFrame(data), entity="entity", time="time")


def _arrays(panel: PanelFrame) -> tuple[np.ndarray, np.ndarray]:
    df = panel.collect()
    x = df.select([f"x{j}" for j in range(N_FEATURES)]).to_numpy().astype(np.float64)
    return x, df["y"].to_numpy().astype(np.float64)


def _moments(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centering/scaling constants; zero-variance columns are left alone."""
    scale = x.std(axis=0)
    y_scale = float(y.std())
    return (
        x.mean(axis=0),
        np.where(scale > 0, scale, 1.0),
        float(y.mean()),
        y_scale if y_scale > 0 else 1.0,
    )


def _corr_rank(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Feature indices ordered by |corr(x_j, y)|, strongest first."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    corr = np.divide(xc.T @ yc, denom, out=np.zeros(x.shape[1]), where=denom > 0)
    return np.argsort(-np.abs(corr), kind="stable")


def _ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gram = x.T @ x + RIDGE_LAMBDA * np.eye(x.shape[1])
    return np.linalg.solve(gram, (x.T @ y)[..., None])[..., 0]


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / total


#: Each stage fit two ways. ``permissive`` sees ``full`` (train + test rows);
#: ``point_in_time`` sees only ``train``. The callables are what
#: :func:`resolve_modes` hands back, and nothing else about them differs.
PIPELINE = (
    Component(
        "imputer",
        permissive=lambda train, full: np.nanmean(full[0], axis=0),
        point_in_time=lambda train, full: np.nanmean(train[0], axis=0),
        description="fill value for missing features",
    ),
    Component(
        "scaler",
        permissive=lambda train, full: _moments(*full),
        point_in_time=lambda train, full: _moments(*train),
        description="centre/scale X and y (the model has no intercept)",
    ),
    Component(
        "selector",
        permissive=lambda train, full: _corr_rank(*full),
        point_in_time=lambda train, full: _corr_rank(*train),
        description="top-k features by |corr| with the target",
    ),
)


def score_pipeline(folds, selection: frozenset[str]) -> float:
    """Mean out-of-sample R^2 over the folds, with ``selection`` run leakily."""
    modes = resolve_modes(PIPELINE, selection)
    scores = []
    for (x_train, y_train), (x_test, y_test) in folds:
        x_full = np.vstack([x_train, x_test])
        y_full = np.concatenate([y_train, y_test])

        fill = modes["imputer"]((x_train, y_train), (x_full, y_full))

        def impute(m: np.ndarray, fill: np.ndarray = fill) -> np.ndarray:
            out = m.copy()
            rows, cols = np.where(np.isnan(out))
            out[rows, cols] = fill[cols]
            return out

        x_train_i, x_test_i = impute(x_train), impute(x_test)
        x_full_i = np.vstack([x_train_i, x_test_i])

        mx, sx, my, sy = modes["scaler"]((x_train_i, y_train), (x_full_i, y_full))
        z_train = (x_train_i - mx) / sx
        z_test = (x_test_i - mx) / sx
        t_train = (y_train - my) / sy

        order = modes["selector"](
            (z_train, t_train), ((x_full_i - mx) / sx, (y_full - my) / sy)
        )
        keep = order[:N_KEEP]

        coef = _ridge(z_train[:, keep], t_train)
        pred = z_test[:, keep] @ coef * sy + my
        scores.append(_r2(y_test, pred))
    return float(np.mean(scores))


@pytest.fixture(scope="module")
def leaky_pipeline_report() -> BorrowedAccuracyReport:
    panel = make_leaky_panel()
    splitter = expanding_window_split(test_size=4, n_splits=5, step_size=4)
    folds = [(_arrays(train), _arrays(test)) for train, test in splitter(panel)]
    assert len(folds) == 5
    return borrowed_accuracy(
        PIPELINE, lambda selection: score_pipeline(folds, selection)
    )


def test_end_to_end_leaky_pipeline_borrows_accuracy(
    leaky_pipeline_report: BorrowedAccuracyReport,
) -> None:
    report = leaky_pipeline_report

    # The headline: fitting on everything manufactures skill that is not there.
    assert report.total > 0.1
    assert report.permissive_score > 0.0 > report.point_in_time_score
    assert report.total == pytest.approx(
        report.permissive_score - report.point_in_time_score, abs=TOL
    )
    assert report.n_evaluations == 2 ** len(PIPELINE) == 8


def test_end_to_end_attribution_is_a_decomposition(
    leaky_pipeline_report: BorrowedAccuracyReport,
) -> None:
    report = leaky_pipeline_report
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=1e-10)
    # Every stage here is genuinely leaky, and the target drift makes the
    # scaler's global mean the biggest single theft.
    assert set(report.attribution) == {"imputer", "scaler", "selector"}
    for phi in report.attribution.values():
        assert phi > 0.0
    assert report.ranked[0][0] == "scaler"
    assert report.share("scaler") > 0.5


def test_end_to_end_point_in_time_run_is_reproducible(
    leaky_pipeline_report: BorrowedAccuracyReport,
) -> None:
    """Re-running the same panel and splits gives bit-identical values."""
    panel = make_leaky_panel()
    splitter = expanding_window_split(test_size=4, n_splits=5, step_size=4)
    folds = [(_arrays(train), _arrays(test)) for train, test in splitter(panel)]
    again = borrowed_accuracy(PIPELINE, lambda s: score_pipeline(folds, s))
    assert again.attribution == leaky_pipeline_report.attribution
    assert again.total == leaky_pipeline_report.total


# --------------------------------------------------------------------------- #
# C1: replicate-aware evaluate (one score per seed)
# --------------------------------------------------------------------------- #
def replicate_game(names, n_rep: int, seed: int):
    """``n_rep`` independent dense games, returned as one vector per subset."""
    games = [random_game(names, seed=seed + r) for r in range(n_rep)]

    def evaluate(selection: frozenset[str]) -> list[float]:
        return [g(selection) for g in games]

    return evaluate, games


def test_replicates_are_each_solved_exactly() -> None:
    """Per-replicate phi equal the scalar decomposition of that replicate."""
    names = ["a", "b", "c", "d"]
    evaluate, games = replicate_game(names, n_rep=5, seed=200)
    report = borrowed_accuracy(names, evaluate)

    assert report.n_replicates == 5
    assert report.n_evaluations == 2 ** len(names)  # one call per subset, not per seed
    for r, game in enumerate(games):
        single = borrowed_accuracy(names, game)
        for name in names:
            assert report.replicate_attribution[name][r] == pytest.approx(
                single.attribution[name], abs=TOL
            )
        assert report.replicate_totals[r] == pytest.approx(single.total, abs=TOL)


def test_efficiency_holds_within_every_replicate_and_for_the_mean() -> None:
    names = ["a", "b", "c", "d", "e"]
    evaluate, _ = replicate_game(names, n_rep=7, seed=300)
    for higher in (True, False):
        report = borrowed_accuracy(names, evaluate, higher_is_better=higher)
        for r in range(7):
            phis = [report.replicate_attribution[n][r] for n in names]
            assert sum(phis) == pytest.approx(report.replicate_totals[r], abs=TOL)
        # The headline is the replicate mean, so it is a decomposition too.
        assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)
        assert report.total == pytest.approx(
            float(np.mean(report.replicate_totals)), abs=TOL
        )


def test_replicate_median_and_range_per_component() -> None:
    def evaluate(selection: frozenset[str]) -> list[float]:
        leak = [0.10, 0.30, 0.20, 0.00, 0.25] if "scaler" in selection else [0.0] * 5
        return leak

    report = borrowed_accuracy(["scaler", "imputer"], evaluate)
    assert report.attribution_median["scaler"] == pytest.approx(0.20, abs=TOL)
    assert report.attribution_range["scaler"] == pytest.approx((0.0, 0.30), abs=TOL)
    assert report.attribution["scaler"] == pytest.approx(0.17, abs=TOL)  # the mean
    assert report.attribution_range["imputer"] == (0.0, 0.0)
    assert report.total_median == pytest.approx(0.20, abs=TOL)
    assert report.total_range == pytest.approx((0.0, 0.30), abs=TOL)
    text = str(report)
    assert "5 replicates" in text and "median" in text


def test_scalar_evaluate_keeps_the_original_report_shape() -> None:
    report = borrowed_accuracy(["a", "b"], additive_game({"a": 0.2, "b": 0.1}))
    assert report.n_replicates is None
    assert report.replicate_totals == ()
    assert report.replicate_attribution == {}
    assert report.method == "exact" and report.standard_error is None
    assert report.attribution_median == report.attribution
    assert report.attribution_range["a"] == (report.attribution["a"],) * 2
    assert report.total_median == report.total
    assert isinstance(report.total, float)
    assert all(isinstance(v, float) for v in report.coalition_values.values())


def test_a_single_replicate_vector_is_still_replicate_mode() -> None:
    report = borrowed_accuracy(["a"], lambda s: [1.0] if s else [0.25])
    assert report.n_replicates == 1
    assert report.total == pytest.approx(0.75, abs=TOL)


def test_numpy_vectors_are_accepted() -> None:
    report = borrowed_accuracy(
        ["a"], lambda s: np.array([1.0, 2.0]) if s else np.zeros(2)
    )
    assert report.replicate_totals == (1.0, 2.0)


def test_replicate_vectors_must_have_one_length() -> None:
    with pytest.raises(ValueError, match="same replicates"):
        borrowed_accuracy(["a"], lambda s: [1.0, 2.0, 3.0] if s else [0.0, 0.0])
    with pytest.raises(ValueError, match="a scalar"):
        borrowed_accuracy(["a"], lambda s: [1.0, 2.0] if s else 0.0)


def test_replicate_vectors_are_validated() -> None:
    with pytest.raises(ValueError, match="1-D vector"):
        borrowed_accuracy(["a"], lambda s: [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="empty vector"):
        borrowed_accuracy(["a"], lambda s: [])
    with pytest.raises(ValueError, match="finite score"):
        borrowed_accuracy(["a"], lambda s: [0.1, math.nan])
    with pytest.raises(TypeError, match="replicate scores"):
        borrowed_accuracy(["a"], lambda s: ["0.1", "0.2"])  # type: ignore[arg-type,return-value]


def test_end_to_end_replicates_over_independent_panels() -> None:
    """Three independent panel draws, one evaluate call per subset."""
    splitter = expanding_window_split(test_size=4, n_splits=5, step_size=4)
    fold_sets = []
    for seed in (0, 1, 2):
        panel = make_leaky_panel(seed=seed)
        fold_sets.append(
            [(_arrays(train), _arrays(test)) for train, test in splitter(panel)]
        )

    report = borrowed_accuracy(
        PIPELINE, lambda s: [score_pipeline(folds, s) for folds in fold_sets]
    )
    assert report.n_replicates == 3
    assert report.n_evaluations == 8
    # Seed 0 is the single-seed fixture: its replicate must reproduce it exactly.
    single = borrowed_accuracy(PIPELINE, lambda s: score_pipeline(fold_sets[0], s))
    for name in single.attribution:
        assert report.replicate_attribution[name][0] == pytest.approx(
            single.attribution[name], abs=TOL
        )
    for r in range(3):
        assert report.replicate_totals[r] > 0.0  # every draw borrows
        assert sum(report.replicate_attribution[n][r] for n in single.attribution) == (
            pytest.approx(report.replicate_totals[r], abs=1e-10)
        )


# --------------------------------------------------------------------------- #
# C2: opt-in permutation sampling beyond max_components
# --------------------------------------------------------------------------- #
def test_permutation_lifts_the_limit_at_a_bounded_cost() -> None:
    k, m = 20, 50
    names = [f"c{i}" for i in range(k)]
    weights = {n: 0.01 * (i + 1) for i, n in enumerate(names)}

    def evaluate(selection: frozenset[str]) -> float:
        joint = 0.3 if {"c1", "c2"} <= selection else 0.0
        return sum(weights[n] for n in selection) + joint

    counted = counting(evaluate)
    report = borrowed_accuracy(
        names, counted, method="permutation", n_permutations=m, seed=0
    )
    assert report.method == "permutation"
    assert report.n_permutations == m and report.seed == 0
    assert report.n_evaluations == len(set(counted.calls)) == len(counted.calls)
    assert report.n_evaluations <= m * (k - 1) + 2 == 952
    # Efficiency to floating-point tolerance, not in expectation.
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=1e-10)
    # Additive plus one pairwise interaction: antithetic pairs make it exact.
    assert report.attribution["c1"] == pytest.approx(weights["c1"] + 0.15, abs=1e-10)
    assert report.attribution["c2"] == pytest.approx(weights["c2"] + 0.15, abs=1e-10)
    assert report.attribution["c7"] == pytest.approx(weights["c7"], abs=1e-10)
    assert report.standard_error is not None
    assert set(report.standard_error) == set(names)
    assert "permutation Shapley" in str(report)


def test_permutation_agrees_with_exact_within_its_standard_error() -> None:
    names = [f"c{i}" for i in range(6)]
    evaluate = random_game(names, seed=5)
    exact = borrowed_accuracy(names, evaluate)
    sampled = borrowed_accuracy(
        names, evaluate, method="permutation", n_permutations=300, seed=1
    )
    assert sampled.standard_error is not None
    for name in names:
        se = sampled.standard_error[name]
        assert se > 0
        assert abs(sampled.attribution[name] - exact.attribution[name]) < 5 * se
    assert sampled.total == pytest.approx(exact.total, abs=TOL)


def test_permutation_is_seeded() -> None:
    names = [f"c{i}" for i in range(14)]
    evaluate = random_game(names, seed=8)
    a = borrowed_accuracy(
        names, evaluate, method="permutation", n_permutations=8, seed=3
    )
    b = borrowed_accuracy(
        names, evaluate, method="permutation", n_permutations=8, seed=3
    )
    c = borrowed_accuracy(
        names, evaluate, method="permutation", n_permutations=8, seed=4
    )
    assert a.attribution == b.attribution
    assert a.to_json() == b.to_json()
    assert a.attribution != c.attribution


def test_permutation_with_replicates() -> None:
    names = [f"c{i}" for i in range(13)]
    evaluate, _ = replicate_game(names, n_rep=3, seed=40)
    report = borrowed_accuracy(
        names, evaluate, method="permutation", n_permutations=6, seed=0
    )
    assert report.n_replicates == 3
    assert report.standard_error is not None
    for r in range(3):
        phis = [report.replicate_attribution[n][r] for n in names]
        assert sum(phis) == pytest.approx(report.replicate_totals[r], abs=1e-10)


def test_refusal_suggests_grouping_first_and_names_the_opt_in() -> None:
    names = [f"c{i}" for i in range(13)]
    with pytest.raises(ValueError) as info:
        borrowed_accuracy(names, lambda s: 0.0)
    message = str(info.value)
    assert message.index("Group components") < message.index("method='permutation'")
    assert "2**13 = 8192" in message


def test_permutation_is_never_a_silent_fallback() -> None:
    with pytest.raises(ValueError, match="needs n_permutations"):
        borrowed_accuracy(["a", "b"], lambda s: 0.0, method="permutation")
    with pytest.raises(ValueError, match="only applies to method='permutation'"):
        borrowed_accuracy(["a", "b"], lambda s: 0.0, n_permutations=10)
    with pytest.raises(ValueError, match="even"):
        borrowed_accuracy(
            ["a", "b"], lambda s: 0.0, method="permutation", n_permutations=5
        )
    with pytest.raises(ValueError, match="method must be one of"):
        borrowed_accuracy(["a", "b"], lambda s: 0.0, method="sampled")


# --------------------------------------------------------------------------- #
# to_dict / to_json
# --------------------------------------------------------------------------- #
def test_to_dict_is_json_safe_and_carries_schema_and_provenance() -> None:
    import panelary

    report = borrowed_accuracy(
        ["scaler", "imputer"], additive_game({"scaler": 0.2, "imputer": 0.05})
    )
    d = report.to_dict()
    assert d["schema"] == "panelary.BorrowedAccuracyReport/1"
    assert d["produced_by"] == (
        f"panelary.leakage.borrowed_accuracy@{panelary.__version__}"
    )
    assert d["components"] == ["scaler", "imputer"]
    assert d["method"] == "exact" and d["n_replicates"] is None
    assert len(d["coalitions"]) == 4
    assert d["coalitions"][0] == {"permissive": [], "value": 0.0}
    assert d["coalitions"][-1]["permissive"] == ["scaler", "imputer"]
    assert json.loads(report.to_json()) == d
    assert json.loads(report.to_json(indent=2)) == d


def test_to_json_is_byte_deterministic() -> None:
    names = ["a", "b", "c"]
    evaluate, _ = replicate_game(names, n_rep=3, seed=1)
    first = borrowed_accuracy(names, evaluate).to_json()
    second = borrowed_accuracy(names, evaluate).to_json()
    assert first == second
    assert first == json.dumps(json.loads(first), sort_keys=True, separators=(",", ":"))
    for forbidden in ("0x", "timestamp", "created"):
        assert forbidden not in first


def test_to_dict_with_replicates_and_permutation() -> None:
    names = [f"c{i}" for i in range(5)]
    evaluate, _ = replicate_game(names, n_rep=2, seed=9)
    d = borrowed_accuracy(
        names, evaluate, method="permutation", n_permutations=4, seed=0
    ).to_dict()
    assert d["method"] == "permutation" and d["n_permutations"] == 4
    assert d["n_replicates"] == 2
    assert set(d["standard_error"]) == set(names)
    assert len(d["replicate_totals"]) == 2
    assert set(d["attribution_summary"]["c0"]) == {"median", "min", "max"}
    assert all(len(c["replicates"]) == 2 for c in d["coalitions"])
    json.dumps(d, allow_nan=False)  # strictly valid JSON
