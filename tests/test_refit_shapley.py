"""Refit Shapley: feature groups valued by refitting through purged CV.

The Shapley axioms are the correctness proof again -- efficiency, null player,
symmetry -- but on a game whose every coalition is a real purged-CV refit. On
top of those, the leakage contract (``leakage_safe``): fits see training rows
only, the baseline is the *training* fold's mean, and every coalition shares
the same folds, rows and seeds (common random numbers).

Most tests use a tiny numpy OLS so the file runs on the bare-core install and
in well under a second; the sklearn comparisons are guarded.
"""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

from panelary.core.model_selection import PurgedKFold
from panelary.select._refit_shapley import (
    DEFAULT_MAX_PLAYERS,
    RefitShapleyReport,
    refit_shapley,
)

TOL = 1e-10
ENTITY, TIME = "entity", "time"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
class OLS:
    """Least squares with an intercept; min-norm, so duplicate columns are fine.

    ``subsample`` < 1 fits on a random subset of rows drawn from
    ``random_state`` -- a stochastic learner whose replicates genuinely differ.
    """

    def __init__(self, random_state: int | None = None, subsample: float = 1.0):
        self.random_state = random_state
        self.subsample = subsample

    def get_params(self, deep: bool = True) -> dict:
        return {"random_state": self.random_state, "subsample": self.subsample}

    def set_params(self, **params) -> OLS:
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> OLS:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.subsample < 1.0:
            rng = np.random.default_rng(self.random_state)
            keep = rng.choice(len(x), int(self.subsample * len(x)), replace=False)
            x, y = x[keep], y[keep]
        self.mx_, self.my_ = x.mean(axis=0), float(y.mean())
        self.coef_ = np.linalg.lstsq(x - self.mx_, y - self.my_, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mx_) @ self.coef_ + self.my_


def make_panel(seed: int = 0, n_entities: int = 8, n_times: int = 36) -> pl.DataFrame:
    """Two signal features, an exact duplicate, a near-duplicate, noise, drift."""
    rng = np.random.default_rng(seed)
    n = n_entities * n_times
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = 1.0 * a - 0.7 * b + 0.04 * time + 0.5 * rng.standard_normal(n)
    return pl.DataFrame(
        {
            ENTITY: entity,
            TIME: time,
            "a": a,
            "a_copy": a.copy(),
            "a_near": a + 0.05 * rng.standard_normal(n),
            "b": b,
            "noise": rng.standard_normal(n),
            "const": np.full(n, 3.0),
            "y": y,
        }
    )


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return make_panel()


def cv() -> PurgedKFold:
    return PurgedKFold(n_splits=4, horizon=1, embargo=1)


def run(panel: pl.DataFrame, groups, **kw) -> RefitShapleyReport:
    return refit_shapley(
        kw.pop("estimator", OLS()),
        panel,
        "y",
        kw.pop("cv", cv()),
        groups,
        entity=ENTITY,
        time=TIME,
        **kw,
    )


def manual_train_mean_r2(panel: pl.DataFrame) -> float:
    """``v({})`` computed by hand: predict each training fold's mean."""
    from panelary.core.panel_frame import PanelFrame

    scores = []
    for train, test in cv().split(PanelFrame(panel, entity=ENTITY, time=TIME)):
        y_tr = train.collect()["y"].to_numpy()
        y_te = test.collect()["y"].to_numpy()
        pred = np.full(len(y_te), y_tr.mean())
        scores.append(
            1.0 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
        )
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Axioms
# --------------------------------------------------------------------------- #
def test_efficiency(panel: pl.DataFrame) -> None:
    report = run(panel, {"a": ["a"], "b": ["b"], "noise": ["noise"]})
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)
    assert report.total == pytest.approx(
        report.full_score - report.baseline_score, abs=TOL
    )
    assert report.n_evaluations == 8
    # Seven non-empty coalitions x four folds; the baseline fits nothing.
    assert report.n_fits == 7 * 4 and report.n_folds == 4


def test_null_player_a_useless_group_gets_zero(panel: pl.DataFrame) -> None:
    """A constant column cannot change a refit OLS's predictions: phi == 0."""
    report = run(panel, {"a": ["a"], "b": ["b"], "const": ["const"]})
    assert report.attribution["const"] == pytest.approx(0.0, abs=TOL)
    assert report.attribution["a"] > 0.1 and report.attribution["b"] > 0.05


def test_symmetry_a_redundant_pair_splits_the_credit(panel: pl.DataFrame) -> None:
    """Two copies of one signal share its worth instead of both scoring zero.

    With players a, a' (an exact copy) and b, and v measured from the baseline:
    v(a) = v(a') = v(a, a') =: A and v(a, b) = v(a', b) = v(a, a', b) =: AB.
    Shapley then gives each copy A/3 + (AB - v(b))/6 -- equal (symmetry), and
    not zero, which is what permuting one copy against a fixed model reports.
    Note what it is *not*: half of what ``a`` gets when it has no copy. Adding
    a duplicate player shifts credit towards the others, which is why players
    should be groups chosen on purpose, not whatever columns happen to exist.
    """
    report = run(panel, {"a": ["a"], "a_copy": ["a_copy"], "b": ["b"]})
    v = {
        frozenset(s): x - report.baseline_score
        for s, x in report.coalition_values.items()
    }
    big_a = v[frozenset({"a"})]
    ab = v[frozenset({"a", "b"})]
    assert v[frozenset({"a_copy"})] == pytest.approx(big_a, abs=TOL)
    assert v[frozenset({"a", "a_copy"})] == pytest.approx(big_a, abs=TOL)
    expected = big_a / 3 + (ab - v[frozenset({"b"})]) / 6
    assert report.attribution["a"] == pytest.approx(expected, abs=TOL)
    assert report.attribution["a_copy"] == pytest.approx(
        report.attribution["a"], abs=TOL
    )
    assert report.attribution["a"] > 0.1
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


def test_group_players_own_their_columns_jointly(panel: pl.DataFrame) -> None:
    grouped = run(panel, {"momentum": ["a", "a_near"], "b": ["b"]})
    assert grouped.group_features == {"momentum": ("a", "a_near"), "b": ("b",)}
    assert grouped.attribution["momentum"] > grouped.attribution["b"] > 0.0


def test_a_plain_list_means_one_player_per_column(panel: pl.DataFrame) -> None:
    report = run(panel, ["a", "b"])
    assert report.groups == ("a", "b")
    assert report.group_features == {"a": ("a",), "b": ("b",)}


# --------------------------------------------------------------------------- #
# The baseline, explicitly
# --------------------------------------------------------------------------- #
def test_baseline_is_the_training_fold_mean(panel: pl.DataFrame) -> None:
    """v({}) predicts each TRAINING fold's mean.

    The target drifts, so a training mean misses every test fold and scores
    below zero; a leaky baseline built from the test fold's own mean would
    score exactly 0.0 on every fold. That difference is what this pins.
    """
    report = run(panel, {"a": ["a"]})
    assert report.baseline == "train_mean"
    assert report.baseline_score == pytest.approx(manual_train_mean_r2(panel), abs=TOL)
    assert report.baseline_score < -0.01


def test_base_features_are_in_every_coalition(panel: pl.DataFrame) -> None:
    report = run(panel, {"a": ["a"], "noise": ["noise"]}, base_features=["b"])
    only_b = run(panel, {"b": ["b"]})
    assert report.baseline == "base_features"
    assert report.base_features == ("b",)
    # v({}) with base features is the model refit on them alone.
    assert report.baseline_score == pytest.approx(only_b.full_score, abs=TOL)
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


# --------------------------------------------------------------------------- #
# Leakage contract and common random numbers
# --------------------------------------------------------------------------- #
class Spy(OLS):
    """Records the row ids and seed of every fit (row id is encoded in column 0)."""

    log: list[tuple[frozenset[int], int | None, int]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> Spy:
        Spy.log.append(
            (
                frozenset(np.floor(x[:, 0]).astype(int).tolist()),
                self.random_state,
                x.shape[1],
            )
        )
        return super().fit(x, y)


def test_every_fit_sees_training_rows_only_and_the_same_folds() -> None:
    df = make_panel(seed=1)
    n = df.height
    df = df.with_columns(
        rid=pl.Series(np.arange(n, dtype=np.float64)),
        rid2=pl.Series(np.arange(n, dtype=np.float64) + 0.25),
    )
    from panelary.core.panel_frame import PanelFrame

    expected = []
    for train, test in cv().split(PanelFrame(df, entity=ENTITY, time=TIME)):
        tr = frozenset(train.collect()["rid"].cast(pl.Int64).to_list())
        te = frozenset(test.collect()["rid"].cast(pl.Int64).to_list())
        assert not tr & te
        expected.append(tr)

    Spy.log = []
    report = refit_shapley(
        Spy(),
        df,
        "y",
        cv(),
        {"g1": ["rid"], "g2": ["rid2"]},
        seeds=[11, 22],
        entity=ENTITY,
        time=TIME,
    )
    # 3 non-empty coalitions x 2 seeds x 4 folds.
    assert len(Spy.log) == report.n_fits == 3 * 2 * 4
    fits = [rows for rows, _, _ in Spy.log]
    # Each fit's rows are exactly one fold's training block, in fold order --
    # the same blocks for every coalition and every seed (common random numbers).
    assert fits == expected * (3 * 2)
    # Seeds: every fit of replicate r used seeds[r].
    seeds = [s for _, s, _ in Spy.log]
    assert seeds == ([11] * 4 + [22] * 4) * 3


def test_the_splitter_is_materialised_once(panel: pl.DataFrame) -> None:
    calls = []

    class Counting(PurgedKFold):
        def split(self, p):
            calls.append(1)
            return super().split(p)

    run(panel, {"a": ["a"], "b": ["b"], "noise": ["noise"]}, cv=Counting(4, embargo=1))
    assert len(calls) == 1


def test_rows_with_missing_values_are_dropped_for_every_coalition() -> None:
    """A row missing in ANY group is dropped from every coalition's fits."""
    df = make_panel(seed=2)
    mask = np.zeros(df.height, dtype=bool)
    mask[::7] = True
    rid = np.arange(df.height, dtype=np.float64)
    df = df.with_columns(
        rid=pl.Series(rid),
        rid2=pl.when(pl.Series(mask)).then(None).otherwise(pl.Series(rid + 0.25)),
    )
    Spy.log = []
    report = refit_shapley(
        Spy(),
        df,
        "y",
        cv(),
        {"rid": ["rid"], "rid2": ["rid2"]},
        entity=ENTITY,
        time=TIME,
    )
    dropped = set(np.flatnonzero(mask).tolist())
    assert len(Spy.log) == 3 * 4
    for rows, _, _ in Spy.log:
        assert not rows & dropped  # even the coalition {rid}, which has no gaps
    assert report.n_folds == 4


# --------------------------------------------------------------------------- #
# Replicates (C1), periods, permutation (C2)
# --------------------------------------------------------------------------- #
def test_replicates_give_a_spread_and_keep_efficiency(panel: pl.DataFrame) -> None:
    report = run(
        panel,
        {"a": ["a"], "b": ["b"], "noise": ["noise"]},
        estimator=OLS(subsample=0.6),
        seeds=[0, 1, 2, 3],
    )
    assert report.seeds == (0, 1, 2, 3)
    for r in range(4):
        phis = [report.replicate_attribution[g][r] for g in report.groups]
        assert sum(phis) == pytest.approx(report.replicate_totals[r], abs=TOL)
    assert report.attribution["a"] == pytest.approx(
        float(np.mean(report.replicate_attribution["a"])), abs=TOL
    )
    lo, hi = report.attribution_range["a"]
    assert lo < hi  # the replicates genuinely differ
    assert lo <= report.attribution_median["a"] <= hi
    frame = report.to_frame()
    assert {"median", "min", "max"} <= set(frame.columns)


def test_seeds_need_a_seedable_estimator(panel: pl.DataFrame) -> None:
    class Plain:
        def fit(self, x, y):
            self.m = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.m)

    with pytest.raises(ValueError, match="random_state"):
        run(panel, {"a": ["a"]}, estimator=Plain(), seeds=[0, 1])
    with pytest.raises(ValueError, match="repeated seeds"):
        run(panel, {"a": ["a"]}, seeds=[1, 1])


def test_an_estimator_factory_receives_the_seed(panel: pl.DataFrame) -> None:
    seen: list[int | None] = []

    def make(seed: int | None) -> OLS:
        seen.append(seed)
        return OLS(random_state=seed, subsample=0.7)

    report = run(panel, {"a": ["a"], "b": ["b"]}, estimator=make, seeds=[5, 6])
    assert set(seen) == {5, 6}
    assert report.seeds == (5, 6)


def test_per_period_decomposition_is_exact_and_averages_to_the_headline(
    panel: pl.DataFrame,
) -> None:
    report = run(panel, {"a": ["a"], "b": ["b"], "noise": ["noise"]}, per_period=True)
    assert report.per_period is not None and len(report.per_period) == report.n_folds
    for period in report.per_period:
        assert sum(period.attribution.values()) == pytest.approx(period.total, abs=TOL)
        assert period.start <= period.end
    for g in report.groups:
        mean_over_folds = np.mean([p.attribution[g] for p in report.per_period])
        assert report.attribution[g] == pytest.approx(mean_over_folds, abs=TOL)
    starts = [p.start for p in report.per_period]
    assert starts == sorted(starts)


def test_permutation_beyond_max_players(panel: pl.DataFrame) -> None:
    df = panel.with_columns(
        [pl.col("noise").shift(i + 1).fill_null(0.0).alias(f"n{i}") for i in range(11)]
    )
    groups = ["a", "b"] + [f"n{i}" for i in range(11)]
    assert len(groups) == DEFAULT_MAX_PLAYERS + 1

    with pytest.raises(ValueError) as info:
        run(df, groups)
    message = str(info.value)
    assert message.index("Group your features") < message.index("method='permutation'")

    report = run(df, groups, method="permutation", n_permutations=4, seed=0)
    assert report.method == "permutation"
    assert report.n_evaluations <= 4 * 12 + 2
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)
    assert report.standard_error is not None and set(report.standard_error) == set(
        groups
    )
    assert report.ranked[0][0] == "a"


def test_permutation_is_opt_in_only(panel: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="needs n_permutations"):
        run(panel, ["a", "b"], method="permutation")
    with pytest.raises(ValueError, match="only applies"):
        run(panel, ["a", "b"], n_permutations=10)


# --------------------------------------------------------------------------- #
# Report, serialisation, determinism
# --------------------------------------------------------------------------- #
def test_deterministic_and_serialisable(panel: pl.DataFrame) -> None:
    import panelary

    kw = {"per_period": True, "seeds": [0, 1], "estimator": OLS(subsample=0.8)}
    first = run(panel, {"a": ["a"], "b": ["b"]}, **kw)
    second = run(panel, {"a": ["a"], "b": ["b"]}, **kw)
    assert first.to_json() == second.to_json()
    d = json.loads(first.to_json())
    assert d == first.to_dict()
    assert d["schema"] == "panelary.RefitShapleyReport/1"
    assert d["produced_by"] == f"panelary.select.refit_shapley@{panelary.__version__}"
    assert d["seeds"] == [0, 1] and len(d["per_period"]) == first.n_folds
    assert d["coalitions"][0] == {"groups": [], "value": first.baseline_score}
    json.dumps(d, allow_nan=False)


def test_readable_report(panel: pl.DataFrame) -> None:
    report = run(panel, {"a": ["a"], "b": ["b"]})
    text = str(report)
    assert "Refit Shapley (r2)" in text and "train_mean" in text
    assert report.ranked[0][0] == "a"
    assert report.share("a") + report.share("b") == pytest.approx(1.0)
    frame = report.to_frame()
    assert frame.columns == ["group", "n_features", "phi", "share"]
    assert frame["group"].to_list() == ["a", "b"]


def test_neg_mse_and_callable_scoring(panel: pl.DataFrame) -> None:
    mse = run(panel, ["a", "b"], scoring="neg_mse")
    assert mse.scoring == "neg_mse" and mse.total > 0

    def mae(y_true, y_pred):
        return -float(np.mean(np.abs(y_true - y_pred)))

    custom = run(panel, ["a", "b"], scoring=mae)
    assert custom.scoring == "mae" and custom.total > 0


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("groups", "kw", "match"),
    [
        ({"x": ["a"], "z": ["a", "b"]}, {}, "disjoint"),
        ({"x": []}, {}, "empty"),
        ({"x": ["missing"]}, {}, "not found"),
        ({"x": ["y"]}, {}, "target or a panel key"),
        ({"x": [TIME]}, {}, "target or a panel key"),
        ({"x": ["a"]}, {"base_features": ["a"]}, "also in a group"),
        ({}, {}, "at least one group"),
        (["a", "a"], {}, "repeated"),
        (["a"], {"scoring": "auc"}, "scoring must be one of"),
        (["a"], {"seeds": []}, "empty"),
    ],
)
def test_refusals(panel: pl.DataFrame, groups, kw, match) -> None:
    with pytest.raises(ValueError, match=match):
        run(panel, groups, **kw)


def test_type_refusals(panel: pl.DataFrame) -> None:
    with pytest.raises(TypeError, match="groups must be a mapping"):
        run(panel, "a")
    with pytest.raises(TypeError, match="scoring must be a str"):
        run(panel, ["a"], scoring=3)


def test_a_regression_target_must_be_numeric(panel: pl.DataFrame) -> None:
    df = panel.with_columns(y=pl.col("y").cast(pl.String))
    with pytest.raises(ValueError, match="must be numeric"):
        run(df, ["a"])


def test_a_non_finite_score_is_an_error() -> None:
    df = make_panel(seed=3).with_columns(y=pl.lit(1.0))  # R^2 undefined
    with pytest.raises(ValueError, match="finite score"):
        refit_shapley(OLS(), df, "y", cv(), ["a"], entity=ENTITY, time=TIME)


def test_no_usable_fold_is_an_error() -> None:
    df = make_panel(seed=4).with_columns(a=pl.lit(None, dtype=pl.Float64))
    with pytest.raises(ValueError, match="no fold had usable rows"):
        refit_shapley(OLS(), df, "y", cv(), ["a"], entity=ENTITY, time=TIME)


# --------------------------------------------------------------------------- #
# Against the incumbents (sklearn)
# --------------------------------------------------------------------------- #
def test_near_duplicates_get_near_equal_credit(panel: pl.DataFrame) -> None:
    """The motivating case, through sklearn's clone path.

    Near-duplicates are close to interchangeable once the model is refit, so
    refit Shapley gives them close to equal credit. Permutation importance
    against one fitted model has no such property: its split follows whatever
    the fit did with two collinear columns.
    """
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    report = run(panel, ["a", "a_near", "b"], estimator=LinearRegression())
    lo, hi = sorted([report.attribution["a"], report.attribution["a_near"]])
    assert lo > 0.8 * hi
    assert report.attribution["b"] > 0.0
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)


def test_classifier_baseline_is_the_training_majority(panel: pl.DataFrame) -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    df = panel.with_columns(up=(pl.col("y") > pl.col("y").median()).cast(pl.Int64))
    report = refit_shapley(
        LogisticRegression(),
        df,
        "up",
        cv(),
        ["a", "b", "noise"],
        entity=ENTITY,
        time=TIME,
    )
    assert report.baseline == "train_majority"
    assert report.scoring == "accuracy"
    assert 0.0 <= report.baseline_score <= 1.0
    assert sum(report.attribution.values()) == pytest.approx(report.total, abs=TOL)
    assert report.attribution["a"] > report.attribution["noise"]


def test_reachable_from_the_select_package() -> None:
    import panelary as pn
    from panelary.select import refit_shapley as exported

    assert exported is refit_shapley
    assert pn.select.refit_shapley is refit_shapley
    assert not math.isnan(DEFAULT_MAX_PLAYERS)
