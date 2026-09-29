"""Fold-local weights (T1-T5 with power) and the cross_validate weight plumbing.

Every leak test pairs the fold-local computation, which must be *bitwise*
unchanged by a perturbation outside the fold, with the global variant
(``weights.attach`` / ``weights.time_decay``), which must change -- a leak test
the naive implementation passes is a rubber stamp.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core.model_selection import CombinatorialPurgedCV, PurgedKFold
from panelary.validation import cpcv_splits, walk_forward_splits
from panelary.weights import FoldWeights
from tests import _spans_reference as ref

LOOKBACK = 5
TB = {
    "entity": "id",
    "time": "date",
    "max_holding": 10,
    "vol_lookback": LOOKBACK,
    "pt": 1.0,
    "sl": 1.0,
}


def _prices(seed: int = 0, n_ent: int = 3, n_t: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "id": [f"e{e}"] * n_t,
                    "date": [
                        dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n_t)
                    ],
                    "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_t))),
                    "x": rng.normal(size=n_t),
                }
            )
            for e in range(n_ent)
        ]
    )


def _middle_fold(labels: pl.DataFrame, embargo: int = LOOKBACK + 1):
    cv = PurgedKFold(n_splits=5, t1="t1", embargo=embargo, return_indices=True)
    return list(cv.split(labels))[2]


RECIPES = [
    FoldWeights(t1="t1"),
    FoldWeights(t1="t1", kind="return", price="close"),
    FoldWeights(t1="t1", decay=0.3),
    FoldWeights(t1="t1", kind="return", price="close", decay=-0.4, balance="label"),
    FoldWeights(t1="t1", kind=None, balance="label", normalize=None),
]


def _perturb_test_prices(
    prices: pl.DataFrame, test_dates, seed: int = 9
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    shock = pl.Series(np.exp(rng.normal(0, 0.05, prices.height)))
    return prices.with_columns(
        pl.when(pl.col("date").is_in(pl.Series(test_dates).implode()))
        .then(pl.col("close") * shock)
        .otherwise(pl.col("close"))
        .alias("close")
    )


def test_t1_t3_t4_fold_local_is_blind_to_test_prices() -> None:
    """T1/T3/T4: moving test-period prices (and so non-training t1s)."""
    prices = _prices(0)
    lab = pn.label.triple_barrier(prices, **TB)
    train, test = _middle_fold(lab)
    times = lab.get_column("date").unique().sort()
    pert = pn.label.triple_barrier(
        _perturb_test_prices(prices, times.gather(test).to_list()), **TB
    )

    # Precondition: every training row is identical; some other t1 moved.
    in_train = pl.col("date").is_in(times.gather(train).implode())
    cols = ["id", "date", "close", "label", "t1", "censored"]
    assert lab.filter(in_train).select(cols).equals(pert.filter(in_train).select(cols))
    assert not lab.get_column("t1").equals(pert.get_column("t1"))

    for fw in RECIPES:
        a = fw.compute(lab, train, test_positions=test, entity="id", time="date")
        b = fw.compute(pert, train, test_positions=test, entity="id", time="date")
        assert a.dtype == b.dtype == np.float64
        assert np.array_equal(a, b), fw  # bitwise

    # Power: the global variants of the same weights move.
    rows = lab.with_row_index().filter(in_train).get_column("index").to_numpy()
    for kw in (
        {"kind": "uniqueness"},
        {"kind": "return", "price": "close"},
        {"kind": "uniqueness", "decay": 0.3},
    ):
        g1 = pn.weights.attach(lab, t1="t1", **kw).get_column("w").to_numpy()[rows]
        g2 = pn.weights.attach(pert, t1="t1", **kw).get_column("w").to_numpy()[rows]
        assert not np.array_equal(g1, g2), kw
    d1 = (
        pn.weights.time_decay(lab, t1="t1", c=0.3)
        .get_column("w_decay")
        .to_numpy()[rows]
    )
    d2 = (
        pn.weights.time_decay(pert, t1="t1", c=0.3)
        .get_column("w_decay")
        .to_numpy()[rows]
    )
    assert not np.allclose(d1, d2, equal_nan=True)


def test_t5_class_weights_blind_to_test_labels() -> None:
    lab = pn.label.triple_barrier(_prices(1), **TB)
    train, test = _middle_fold(lab)
    times = lab.get_column("date").unique().sort()
    flipped = lab.with_columns(
        pl.when(pl.col("date").is_in(times.gather(test).implode()))
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    fw = FoldWeights(t1="t1", kind=None, balance="label")
    a = fw.compute(lab, train, test_positions=test)
    b = fw.compute(flipped, train, test_positions=test)
    assert np.array_equal(a, b)
    g1 = pn.weights.attach(lab, t1="t1", kind=None, balance="label").get_column("w")
    g2 = pn.weights.attach(flipped, t1="t1", kind=None, balance="label").get_column("w")
    in_train = lab.get_column("date").is_in(times.gather(train).implode())
    assert not g1.filter(in_train).equals(g2.filter(in_train))


def test_t2_appending_data_leaves_resolved_fold_weights_unchanged() -> None:
    more = _prices(2, n_t=190)
    prices = more.filter(pl.col("date") < dt.date(2020, 5, 30))  # the first 150 days
    lab = pn.label.triple_barrier(prices, **TB)
    lab_more = pn.label.triple_barrier(more, **TB)
    # walk-forward: train on the first 80 days, test the next 20
    train, test = np.arange(0, 80), np.arange(95, 115)
    for fw in RECIPES:
        a = fw.compute(lab, train, test_positions=test)
        b = fw.compute(lab_more, train, test_positions=test)
        assert np.array_equal(a, b), fw
    rows = np.flatnonzero(lab.get_column("date") < dt.date(2020, 3, 21))
    g1 = pn.weights.attach(lab, t1="t1").get_column("w").to_numpy()
    g2 = pn.weights.attach(lab_more, t1="t1").get_column("w").to_numpy()
    idx_more = (
        lab_more.with_row_index()
        .join(
            lab.with_row_index().select("index", "id", "date"),
            on=["id", "date"],
            suffix="_o",
        )
        .sort("index_o")
        .get_column("index")
        .to_numpy()
    )
    assert not np.array_equal(g1[rows], g2[idx_more[rows]])


def test_guard_raises_on_an_under_purge() -> None:
    lab = pn.label.triple_barrier(_prices(3), **TB)
    fw = FoldWeights(t1="t1")
    # A splitter that ignores the label spans (horizon=0, no t1) under-purges.
    train, test = list(PurgedKFold(n_splits=5, return_indices=True).split(lab))[2]
    with pytest.raises(ValueError, match="cover a test time"):
        fw.compute(lab, train, test_positions=test)
    # ...and the purged splitter passes the guard.
    train, test = _middle_fold(lab)
    fw.compute(lab, train, test_positions=test)


def test_compute_output_alignment_and_zero_rows() -> None:
    lab = pn.label.triple_barrier(_prices(4), **TB)
    train, test = _middle_fold(lab)
    w = FoldWeights(t1="t1").compute(lab, train, test_positions=test)
    times = lab.get_column("date").unique().sort()
    fold = lab.sort(["id", "date"]).filter(
        pl.col("date").is_in(times.gather(train).implode())
    )
    assert w.shape == (fold.height,)
    assert np.all(w[fold.get_column("censored").to_numpy()] == 0.0)
    assert w.sum() == pytest.approx(np.count_nonzero(w), rel=1e-12)
    with pytest.raises(ValueError, match="positions must lie"):
        FoldWeights(t1="t1").compute(lab, [10_000])


def test_decay_warns_once_when_training_straddles_the_test_block() -> None:
    lab = pn.label.triple_barrier(_prices(5), **TB)
    bound = FoldWeights(t1="t1", decay=0.5).bind(lab)
    train, test = _middle_fold(lab)
    with pytest.warns(UserWarning, match="decay runs across the gap"):
        bound.compute(train, test_positions=test)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bound.compute(train, test_positions=test)


# --------------------------------------------------------------------------- #
# cross_validate plumbing
# --------------------------------------------------------------------------- #
class _SpyRegressor:
    """sklearn-shaped estimator that records the weights it was fitted with."""

    calls: list[tuple[int, np.ndarray | None]] = []

    def fit(self, X, y, sample_weight=None):
        _SpyRegressor.calls.append((X.shape[0], sample_weight))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


def _data():
    lab = pn.label.triple_barrier(_prices(6), **TB)
    return lab.filter(~pl.col("censored")).select(
        "id", "date", "x", pl.col("ret").alias("y"), "t1", "close", "label"
    )


def test_cross_validate_passes_fold_local_train_weights() -> None:
    data = _data()
    cv = PurgedKFold(n_splits=4, t1="t1", embargo=LOOKBACK + 1)
    fw = FoldWeights(t1="t1")
    _SpyRegressor.calls.clear()
    pn.cross_validate(
        _SpyRegressor(),
        data.select("id", "date", "x", "y", "t1"),
        y="y",
        cv=cv,
        sample_weight=fw,
    )
    folds = list(
        PurgedKFold(
            n_splits=4, t1="t1", embargo=LOOKBACK + 1, return_indices=True
        ).split(data)
    )
    assert len(_SpyRegressor.calls) == len(folds)
    for (n, w), (train, test) in zip(_SpyRegressor.calls, folds, strict=True):
        assert w is not None and w.shape == (n,)
        want = fw.compute(data, train, test_positions=test)
        assert np.array_equal(w, want)


def test_cross_validate_score_weights_and_metric_contract() -> None:
    data = _data().select("id", "date", "x", "y", "t1")
    cv = PurgedKFold(n_splits=4, t1="t1", embargo=LOOKBACK + 1)
    seen = []

    def metric(y_true, y_pred, sample_weight=None):
        seen.append(sample_weight)
        return -float(np.average((y_true - y_pred) ** 2, weights=sample_weight))

    fw = FoldWeights(t1="t1")
    report = pn.cross_validate(
        _SpyRegressor(), data, y="y", cv=cv, metric=metric, score_weight=fw
    )
    assert len(seen) == 4 and all(w is not None for w in seen)
    folds = list(
        PurgedKFold(
            n_splits=4, t1="t1", embargo=LOOKBACK + 1, return_indices=True
        ).split(data)
    )
    for w, (_train, test) in zip(seen, folds, strict=True):
        assert np.array_equal(w, fw.compute(data, test))

    # default metric becomes the weighted negative MSE
    rep2 = pn.cross_validate(_SpyRegressor(), data, y="y", cv=cv, score_weight=fw)
    np.testing.assert_allclose(rep2.fold_scores, report.fold_scores, rtol=1e-12)

    def unweighted(y_true, y_pred):
        return 0.0

    with pytest.raises(TypeError, match="sample_weight"):
        pn.cross_validate(
            _SpyRegressor(), data, y="y", cv=cv, metric=unweighted, score_weight=fw
        )


def test_cross_validate_without_weights_is_unchanged() -> None:
    data = _data().select("id", "date", "x", "y")
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pn.cross_validate(_SpyRegressor(), data, y="y", cv=cv, n_trials=3)
        b = pn.cross_validate(
            _SpyRegressor(),
            data,
            y="y",
            cv=cv,
            n_trials=3,
            sample_weight=None,
            score_weight=None,
        )
    assert a.fold_scores == b.fold_scores and a.path_sharpes == b.path_sharpes
    assert np.array_equal(a.performance_matrix, b.performance_matrix)
    # and equal to a hand-written loop over the same splits
    folds = list(
        CombinatorialPurgedCV(
            n_groups=5, n_test_groups=2, embargo=2, return_indices=True
        ).split(data)
    )
    df = data.sort(["id", "date"])
    times = df.get_column("date").unique().sort()
    for score, (train, test) in zip(a.fold_scores, folds, strict=True):
        tr = df.filter(pl.col("date").is_in(times.gather(train).implode()))
        te = df.filter(pl.col("date").is_in(times.gather(test).implode()))
        mu = float(np.mean(tr.get_column("y").to_numpy()))
        assert score == -float(np.mean((te.get_column("y").to_numpy() - mu) ** 2))


def test_precomputed_weight_column_warns_and_is_not_a_feature() -> None:
    data = pn.weights.attach(
        _data().select("id", "date", "x", "y", "t1"), t1="t1", out="w"
    )
    cv = PurgedKFold(n_splits=3, t1="t1", embargo=LOOKBACK + 1)
    _SpyRegressor.calls.clear()
    with pytest.warns(UserWarning, match="trap T1"):
        pn.cross_validate(
            _SpyRegressor(),
            data.drop("t1"),
            y="y",
            cv=PurgedKFold(3),
            sample_weight="w",
        )
    assert all(w is not None for _n, w in _SpyRegressor.calls)
    with pytest.raises(ValueError, match="not a column"):
        pn.cross_validate(_SpyRegressor(), data, y="y", cv=cv, sample_weight="nope")
    with pytest.raises(TypeError, match="FoldWeights"):
        pn.cross_validate(_SpyRegressor(), data, y="y", cv=cv, sample_weight=np.ones(3))


def test_cross_validate_under_purge_raises() -> None:
    data = _data().select("id", "date", "x", "y", "t1")
    with pytest.raises(ValueError, match="cover a test time"):
        pn.cross_validate(
            _SpyRegressor(),
            data,
            y="y",
            cv=PurgedKFold(n_splits=4),
            sample_weight=FoldWeights(t1="t1"),
        )


_PANEL_CALLS: list[np.ndarray | None] = []


class _PanelSpy:
    def fit(self, X, y, sample_weight=None):
        _PANEL_CALLS.append(sample_weight)
        self.coef_ = 0.0
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


def test_panel_estimators_receive_weights_through_a_column() -> None:
    from panelary.models import PanelSklearnRegressor

    data = _data().select("id", "date", "x", "y", "t1")
    est = PanelSklearnRegressor(
        _PanelSpy(), target="y", features=["x"], entity="id", time="date"
    )
    cv = PurgedKFold(n_splits=3, t1="t1", embargo=LOOKBACK + 1)
    _PANEL_CALLS.clear()
    pn.cross_validate(est, data, y="y", cv=cv, sample_weight=FoldWeights(t1="t1"))
    assert len(_PANEL_CALLS) == 3 and all(
        w is not None and w.size for w in _PANEL_CALLS
    )
    assert est.sample_weight is None  # the per-fold copy was changed, not the original


def test_validate_pass_through() -> None:
    data = _data().select("id", "date", "x", "y", "t1")
    _SpyRegressor.calls.clear()
    pn.validate.purged_kfold(
        _SpyRegressor(),
        data,
        y="y",
        n_splits=3,
        t1="t1",
        embargo=LOOKBACK + 1,
        sample_weight=FoldWeights(t1="t1"),
    )
    assert all(w is not None for _n, w in _SpyRegressor.calls)


# --------------------------------------------------------------------------- #
# t1= on the positional splitters
# --------------------------------------------------------------------------- #
def test_cpcv_splits_t1_matches_the_panel_splitter() -> None:
    lab = pn.label.triple_barrier(_prices(7), **TB)
    pf = pn.as_panel(lab, "id", "date")
    times = pf.time_index()
    from panelary.core.model_selection import _resolve_t1

    t1 = _resolve_t1("t1", pf, times.to_numpy())
    got = cpcv_splits(
        times.len(), n_groups=6, n_test_groups=2, embargo=3, times=times, t1=t1
    )
    want = list(
        CombinatorialPurgedCV(6, 2, t1="t1", embargo=3, return_indices=True).split(lab)
    )
    assert len(got) == len(want)
    for split, (train, test) in zip(got, want, strict=True):
        assert np.array_equal(split.train, train) and np.array_equal(split.test, test)


def test_positional_t1_in_positions_and_walk_forward() -> None:
    n = 60
    rng = np.random.default_rng(0)
    end = np.minimum(np.arange(n) + rng.integers(0, 6, n), n - 1).astype(np.int64)
    splits = walk_forward_splits(n, n_splits=3, test_size=10, embargo=1, t1=end)
    for sp in splits:
        start = int(sp.test[0])
        allowed = ref.frozen_purge_embargo_positions_t1(
            n, sp.test, np.arange(n), end, 1
        )
        want = allowed[(allowed < start) & (allowed < start - 1)]
        assert np.array_equal(sp.train, want)
        assert np.all(end[sp.train] < start)  # every training label resolves first
    splits_cp = cpcv_splits(n, n_groups=4, n_test_groups=1, t1=end)
    for sp in splits_cp:
        assert np.array_equal(
            sp.train,
            ref.frozen_purge_embargo_positions_t1(n, sp.test, np.arange(n), end, 0),
        )
    with pytest.raises(ValueError, match="one label end time per position"):
        cpcv_splits(n, t1=end[:-1])
