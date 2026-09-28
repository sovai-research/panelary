"""Leak-safety invariants: fitted-state provenance and near-duplicate straddle.

Contract touched: ``leakage_safe`` ("fit on training rows only") and the
dedup-before-split doctrine. Each check is exercised against a deliberately
leaky setup it must catch -- a scaler fitted on the whole panel, a transformer
that re-fits at transform time, a near-duplicate planted across the boundary --
and against the honest counterpart it must pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.preprocessing import LeakageWarning
from panelary.quality import (
    PanelValidator,
    check_fitted_state,
    check_near_duplicate_straddle,
    validate_panel,
)
from panelary.transform.scaling import TimeSeriesScaler
from panelary.validation import IndexSplit


def _panel(n_times: int = 20, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "id": np.repeat(["a", "b", "c"], n_times),
            "t": np.tile(np.arange(n_times, dtype=np.int64), 3),
            "x": rng.normal(size=3 * n_times),
        }
    )


def _split(df: pl.DataFrame, cut: int = 12) -> tuple[pl.DataFrame, pl.DataFrame]:
    return df.filter(pl.col("t") < cut), df.filter(pl.col("t") >= cut)


# --------------------------------------------------------------------------- #
# Fitted-state provenance
# --------------------------------------------------------------------------- #
def test_train_only_fit_passes_repeatedly():
    # group_by output order is not deterministic in polars; the comparison of
    # state frames must not care, or this would flake.
    df = _panel()
    train, _ = _split(df)
    for _ in range(5):
        scaler = TimeSeriesScaler().fit(PanelFrame(train, entity="id", time="t"))
        result = check_fitted_state(scaler, train, entity="id", time="t")
        assert result.passed, result.message
        assert result.observed["fit_rows_outside_train"] == 0
        assert result.observed["state_matches_train_refit"] is True


def test_full_panel_fit_is_caught_by_provenance_and_refit():
    df = _panel()
    train, test = _split(df)
    leaky = TimeSeriesScaler().fit(PanelFrame(df, entity="id", time="t"))
    result = check_fitted_state(leaky, train, entity="id", time="t", name="scaler")
    assert result.status == "fail"
    assert result.observed["fit_rows_outside_train"] == test.height
    assert result.observed["state_matches_train_refit"] is False
    witness = result.offending[0]
    assert witness["source"] == "fit_panel" and witness["keys"]["t"] >= 12
    assert result.evidence_kind == "counterfactual-run"
    assert result.locator.startswith("panelary.quality/fitted_state/scaler#")


def test_refit_catches_the_leak_without_provenance():
    df = _panel()
    train, _ = _split(df)
    leaky = TimeSeriesScaler().fit(PanelFrame(df, entity="id", time="t"))
    leaky._fit_panel = None  # an object that does not record what it saw
    result = check_fitted_state(leaky, train, entity="id", time="t")
    assert result.status == "fail"
    assert "fit_rows_outside_train" not in result.observed
    assert result.observed["first_difference"].startswith("state['stats_']")


class RefitsAtTransform(PanelTransformer):
    """Deliberately leaky: re-learns its mean from whatever it transforms."""

    panel_safe = True
    leakage_safe = True  # a lie -- which is the point of the test

    def _fit(self, panel: PanelFrame) -> None:
        self.mean_ = float(panel.collect()["x"].mean())

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        self.mean_ = float(panel.collect()["x"].mean())  # forbidden re-fit
        return panel.with_columns(pl.col("x") - self.mean_)


class HonestDemean(RefitsAtTransform):
    def _transform(self, panel: PanelFrame) -> PanelFrame:
        return panel.with_columns(pl.col("x") - self.mean_)


def test_transform_time_refit_is_caught_even_with_clean_provenance():
    df = _panel()
    train, test = _split(df)
    pf_train = PanelFrame(train, entity="id", time="t")
    pf_test = PanelFrame(test, entity="id", time="t")

    leaky = RefitsAtTransform().fit(pf_train)
    leaky.transform(pf_test)
    result = check_fitted_state(leaky, train, entity="id", time="t")
    assert result.observed["fit_rows_outside_train"] == 0  # provenance is clean...
    assert result.status == "fail"  # ...but the state no longer is
    assert result.observed["first_difference"] == "state['mean_']"

    honest = HonestDemean().fit(pf_train)
    honest.transform(pf_test)
    assert check_fitted_state(honest, train, entity="id", time="t").passed


class SklearnStyleMean:
    """Not a PanelTransformer: plain ``fit(frame)`` storing ``mean_``."""

    def fit(self, frame: pl.DataFrame) -> SklearnStyleMean:
        self.mean_ = frame.select(pl.col("x").mean()).to_numpy().ravel()
        return self


def test_objects_with_plain_fit_are_refitted_generically():
    df = _panel()
    train, _ = _split(df)
    assert check_fitted_state(SklearnStyleMean().fit(train), train).passed
    leaky = check_fitted_state(SklearnStyleMean().fit(df), train)
    assert leaky.status == "fail" and leaky.offending[0]["source"] == "refit"


class ParamsHolder:
    def __init__(self) -> None:
        self.params: dict[str, float] = {}

    def learn(self, frame: pl.DataFrame) -> None:
        self.params = {"max": float(frame["x"].max())}


def test_attributes_and_refit_hooks():
    df = _panel()
    train, _ = _split(df)
    obj = ParamsHolder()
    obj.learn(df)

    def refit(clone: ParamsHolder, rows: pl.DataFrame) -> ParamsHolder:
        clone.learn(rows)
        return clone

    with pytest.raises(ValueError, match="no learned state"):
        check_fitted_state(obj, train)
    result = check_fitted_state(obj, train, attributes=["params"], refit=refit)
    honest = ParamsHolder()
    honest.learn(train)
    ok = check_fitted_state(honest, train, attributes=["params"], refit=refit)
    assert ok.passed
    # the full-panel max only differs if the max lies in the test period
    expected_leak = float(df["x"].max()) != float(train["x"].max())
    assert result.passed is not expected_leak


def test_validator_runs_fitted_state_against_the_split():
    df = _panel()
    train, test = _split(df)
    leaky = TimeSeriesScaler().fit(PanelFrame(df, entity="id", time="t"))
    good = TimeSeriesScaler().fit(PanelFrame(train, entity="id", time="t"))
    report = validate_panel(
        df,
        entity="id",
        time="t",
        split=(train, test),
        fitted={"good": good, "leaky": leaky},
        near_duplicates=False,
        raise_on_fail=False,
    )
    assert report.check("fitted_state", "good").passed
    assert report.check("fitted_state", "leaky").status == "fail"
    with pytest.raises(ValueError, match="needs a `split`"):
        validate_panel(df, entity="id", time="t", fitted=good)
    with pytest.raises(ValueError, match="single"):
        validate_panel(
            df.with_columns((pl.col("t") % 2).alias("fold")),
            entity="id",
            time="t",
            split="fold",
            fitted=good,
        )


def test_warn_level_leak_raises_leakage_warning():
    df = _panel()
    train, test = _split(df)
    leaky = TimeSeriesScaler().fit(PanelFrame(df, entity="id", time="t"))
    with pytest.warns(LeakageWarning, match="fitted_state"):
        report = validate_panel(
            df,
            entity="id",
            time="t",
            split=(train, test),
            fitted=leaky,
            near_duplicates=False,
            impact={"fitted_state": "warn"},
        )
    assert report.ok and report.status == "warn"


# --------------------------------------------------------------------------- #
# Near-duplicate straddle with precomputed cluster ids (no panelary.clean)
# --------------------------------------------------------------------------- #
def _clustered() -> pl.DataFrame:
    """Two entities x 10 times; (b, 2) and (a, 8) share a cluster."""
    df = pl.DataFrame(
        {
            "id": np.repeat(["a", "b"], 10),
            "t": np.tile(np.arange(10, dtype=np.int64), 2),
            "x": np.arange(20, dtype=np.float64),
        }
    )
    cid = pl.int_range(pl.len(), dtype=pl.Int64)
    planted = (pl.col("id") == "a") & (pl.col("t") == 8)
    return df.with_columns(
        pl.when(planted).then(pl.lit(12, dtype=pl.Int64)).otherwise(cid).alias("cl")
    )


def test_time_split_straddle_and_index_split_agree():
    df = _clustered()
    by_times = check_near_duplicate_straddle(
        df, (list(range(6)), list(range(6, 10))), entity="id", time="t", clusters="cl"
    )
    by_index = check_near_duplicate_straddle(
        df,
        IndexSplit(np.arange(6), np.arange(6, 10)),
        entity="id",
        time="t",
        clusters="cl",
    )
    assert by_times.status == "fail"
    assert by_times.observed == {"straddling_clusters": 1, "rows": 2}
    assert by_times.offending == (
        {
            "keys": {"id": "a", "t": 8},
            "size": 2,
            "train_keys": [{"id": "b", "t": 2}],
            "test_keys": [{"id": "a", "t": 8}],
        },
    )
    assert by_times.evidence_kind == "static-analysis"
    assert by_index.to_dict()["offending"] == by_times.to_dict()["offending"]
    assert by_index.observed == by_times.observed


def test_purged_rows_and_same_side_clusters_do_not_straddle():
    df = _clustered()
    # t=2 in train, t=8 purged (in neither side): no straddle
    purged = check_near_duplicate_straddle(
        df, (list(range(6)), [9]), entity="id", time="t", clusters="cl"
    )
    assert purged.passed
    same_side = check_near_duplicate_straddle(
        df, (list(range(9)), [9]), entity="id", time="t", clusters="cl"
    )
    assert same_side.passed


def test_entity_split_with_frame_members():
    df = _clustered()
    train = df.filter(pl.col("id") == "b")
    test = df.filter(pl.col("id") == "a")
    result = check_near_duplicate_straddle(
        df, (train, test), entity="id", time="t", clusters="cl"
    )
    assert result.status == "fail" and result.n_failing == 1


def test_fold_label_column_and_multiple_folds():
    df = _clustered().with_columns((pl.col("t") // 5).alias("fold"))
    by_label = check_near_duplicate_straddle(
        df, "fold", entity="id", time="t", clusters="cl"
    )
    assert by_label.status == "fail"
    assert by_label.offending[0]["folds"] == [0, 1]
    folds = [
        IndexSplit(np.arange(5), np.arange(5, 10)),
        IndexSplit(np.arange(5, 10), np.arange(5)),
        IndexSplit(np.arange(3), np.arange(3, 5)),
    ]
    multi = check_near_duplicate_straddle(
        df, folds, entity="id", time="t", clusters="cl"
    )
    assert multi.observed == {"straddling_clusters": 1, "rows": 2, "folds": [0, 1]}
    assert [r["fold"] for r in multi.offending] == [0, 1]


def test_row_in_both_sides_is_a_straddle_of_its_own():
    df = _clustered().with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("cl"))
    result = check_near_duplicate_straddle(
        df, ([0, 1, 2], [2, 3]), entity="id", time="t", clusters="cl"
    )
    assert result.observed == {"straddling_clusters": 2, "rows": 2}


def test_malformed_splits_and_clusters():
    df = _clustered()
    with pytest.raises(ValueError, match="`split` must be"):
        check_near_duplicate_straddle(df, 3, entity="id", time="t", clusters="cl")
    with pytest.raises(ValueError, match="exceed"):
        check_near_duplicate_straddle(
            df,
            IndexSplit(np.arange(3), np.arange(3, 40)),
            entity="id",
            time="t",
            clusters="cl",
        )
    with pytest.raises(ValueError, match="values for"):
        check_near_duplicate_straddle(
            df, ([0], [1]), entity="id", time="t", clusters=pl.Series([1, 2])
        )


def test_validator_uses_precomputed_clusters_and_warns():
    df = _clustered()
    validator = PanelValidator(
        entity="id",
        time="t",
        near_duplicates={"clusters": "cl"},
        impact={"near_duplicate_straddle": "warn"},
    )
    with pytest.warns(LeakageWarning, match="near_duplicate_straddle"):
        report = validator.validate(df, split=(list(range(6)), list(range(6, 10))))
    assert report.check("near_duplicate_straddle").status == "warn"


# --------------------------------------------------------------------------- #
# Near-duplicate straddle through panelary.clean
# --------------------------------------------------------------------------- #
def _texts(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    rows = []
    for e in ("a", "b", "c"):
        for t in range(10):
            text = (
                " ".join(rng.choice(words, size=6))
                + f" {e}{t}{rng.integers(1_000_000)}"
            )
            rows.append((e, t, text))
    df = pl.DataFrame(rows, schema=["id", "t", "text"], orient="row")
    source = df.filter((pl.col("id") == "b") & (pl.col("t") == 1))["text"][0]
    return df.with_columns(
        pl.when((pl.col("id") == "c") & (pl.col("t") == 8))
        .then(pl.lit(source + "."))  # a near-copy, not an exact one
        .otherwise(pl.col("text"))
        .alias("text")
    )


def test_planted_near_duplicate_across_boundary_is_detected():
    pytest.importorskip("panelary.clean")
    df = _texts()
    split = (list(range(6)), list(range(6, 10)))
    result = check_near_duplicate_straddle(
        df, split, entity="id", time="t", threshold=0.7
    )
    assert result.status == "fail"
    assert result.offending[0]["train_keys"] == [{"id": "b", "t": 1}]
    assert result.offending[0]["test_keys"] == [{"id": "c", "t": 8}]
    shuffled = check_near_duplicate_straddle(
        df.sample(fraction=1.0, shuffle=True, seed=5),
        split,
        entity="id",
        time="t",
        threshold=0.7,
    )
    assert (
        shuffled.to_json() == result.to_json()
    )  # row order does not leak into the locator


def test_agrees_with_clean_straddling_clusters_oracle():
    clean = pytest.importorskip("panelary.clean")
    df = _texts(seed=1)
    train_t, test_t = list(range(5)), list(range(5, 10))
    ours = check_near_duplicate_straddle(
        df, (train_t, test_t), entity="id", time="t", threshold=0.7
    )
    clusters = clean.near_duplicate_clusters(
        df, entity="id", time="t", columns=["text"], threshold=0.7
    )
    theirs = clean.straddling_clusters(clusters, train_t, test_t, time="t")
    flagged = clusters.filter(
        pl.col("cluster_id").is_in(theirs["cluster_id"].implode())
    )
    assert ours.observed["straddling_clusters"] == theirs.height
    assert ours.row_mask is not None
    assert ours.row_mask.sum() == flagged.height


def test_dedup_before_split_clears_the_straddle():
    clean = pytest.importorskip("panelary.clean")
    df = _texts(seed=2)
    split = (list(range(6)), list(range(6, 10)))
    before = check_near_duplicate_straddle(
        df, split, entity="id", time="t", threshold=0.7
    )
    assert not before.passed
    ids = clean.near_duplicate_clusters(
        df, entity="id", time="t", columns=["text"], threshold=0.7
    )["cluster_id"]
    deduped = df.filter(ids.is_first_distinct())
    after = check_near_duplicate_straddle(
        deduped, split, entity="id", time="t", threshold=0.7
    )
    assert after.passed


def test_fold_label_column_is_excluded_from_clustering():
    pytest.importorskip("panelary.clean")
    df = _texts().with_columns((pl.col("t") >= 6).cast(pl.Int8).alias("is_test"))
    result = check_near_duplicate_straddle(
        df, "is_test", entity="id", time="t", threshold=0.7
    )
    assert result.status == "fail"
    assert result.offending[0]["folds"] == [0, 1]
