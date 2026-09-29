"""Sequential bootstrap: oracle-identical draws, bitwise backends, stable seeds."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary._internal._jit import assert_backend_parity, force_numpy, numba_available
from panelary.core._spans import _spans_from_t1, _uniqueness
from panelary.sample._kernels import _seq_boot
from tests import _spans_reference as ref


def _random_spans(rng: np.random.Generator, n: int, n_rows: int, h: int):
    start = np.sort(rng.integers(0, n_rows, n)).astype(np.int64)
    end = np.minimum(start + rng.integers(0, h, n), n_rows - 1).astype(np.int64)
    return start, end


def _dense_reference(start, end, u):
    """AFML 4.5 on the dense indicator matrix, driven by the given uniforms.

    Returns the draws and the smallest distance of a uniform to a CDF
    boundary (a draw that close is decided by rounding, not by the method).
    """
    n = start.size
    n_rows = int(end.max()) + 1
    ind = np.zeros((n_rows, n))
    for k, (s, e) in enumerate(zip(start, end, strict=True)):
        ind[s : e + 1, k] = 1.0
    phi: list[int] = []
    margin = np.inf
    for uk in u:
        c = ind[:, phi].sum(axis=1) if phi else np.zeros(n_rows)
        avg = np.array(
            [
                np.mean(1.0 / (c[s : e + 1] + 1.0))
                for s, e in zip(start, end, strict=True)
            ]
        )
        cdf = np.cumsum(avg / avg.sum())
        cdf /= cdf[-1]
        j = int(np.searchsorted(cdf, uk, side="right"))
        margin = min(margin, float(np.min(np.abs(cdf - uk))))
        phi.append(j)
    return np.array(phi, dtype=np.int64), margin


@pytest.mark.parametrize("seed", range(20))
def test_draws_match_the_dense_afml_definition(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 150))
    start, end = _random_spans(rng, n, n_rows=int(rng.integers(n, 3 * n)), h=12)
    u = rng.random(n)
    want, margin = _dense_reference(start, end, u)
    if margin < 1e-9:
        pytest.skip("a uniform lies within rounding of a CDF boundary")
    with force_numpy():
        got_np = _seq_boot(start, end, u)
    assert np.array_equal(got_np, want)
    assert np.array_equal(_seq_boot(start, end, u), want)  # numba when installed


@pytest.mark.parametrize("n", [1, 2, 37, 10_000])
def test_numba_and_numpy_are_bitwise_identical(n: int) -> None:
    if not numba_available():
        pytest.skip("numba (the `fast` extra) is not installed")
    rng = np.random.default_rng(n)
    start, end = _random_spans(rng, n, n_rows=2 * n + 5, h=25)
    u = rng.random(n)
    u[: min(3, n)] = [0.0, 1.0 - 2**-53, 0.5][: min(3, n)]  # edges of the CDF
    fast = _seq_boot(start, end, u)
    with force_numpy():
        twin = _seq_boot(start, end, u)
    assert_backend_parity(fast, twin)


def _panel(n_ent: int = 4, n_t: int = 60, seed: int = 0, names=None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    names = names or [f"e{e}" for e in range(n_ent)]
    frames = []
    for name in names:
        # each entity's spans depend on its own name only, not on the others
        r = np.random.default_rng(ref_hash(name) % (2**32))
        t1 = np.minimum(np.arange(n_t) + r.integers(0, 10, n_t), n_t - 1)
        frames.append(
            pl.DataFrame(
                {
                    "id": [name] * n_t,
                    "date": [
                        dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n_t)
                    ],
                    "t1": [dt.date(2020, 1, 1) + dt.timedelta(days=int(i)) for i in t1],
                    "x": rng.normal(size=n_t),
                    "label": r.choice([-1, 1], n_t),
                }
            )
        )
    return pl.concat(frames)


def ref_hash(name: str) -> int:
    return sum(ord(ch) * 31**i for i, ch in enumerate(name))


def test_output_contract() -> None:
    df = _panel()
    rows = pn.sample.sequential_bootstrap(df, t1="t1", seed=3)
    frame, spans = _spans_from_t1(df, t1="t1")
    assert rows.dtype == np.int64 and rows.size == len(spans)
    assert set(rows.tolist()) <= set(spans.label_row.tolist())
    assert np.unique(rows).size < rows.size  # with replacement
    assert pn.sample.sequential_bootstrap(df, t1="t1", n_draws=0.25).size == round(
        0.25 * len(spans)
    )
    assert pn.sample.sequential_bootstrap(df, t1="t1", n_draws=7).size == 7
    same = pn.sample.sequential_bootstrap(
        df.sample(fraction=1.0, shuffle=True, seed=1), t1="t1", seed=3
    )
    assert np.array_equal(rows, same)  # rows index the sorted frame
    with pytest.raises(ValueError, match="method"):
        pn.sample.sequential_bootstrap(df, t1="t1", method="dense")
    with pytest.raises(ValueError, match="stratify"):
        pn.sample.sequential_bootstrap(df, t1="t1", stratify="time")


def test_stratified_draws_are_stable_under_entity_changes() -> None:
    """T18: per-entity streams come from (seed, stable hash of the entity)."""
    base = _panel(names=["b", "c"])
    more = _panel(names=["a", "b", "c", "z"])
    r1 = pn.sample.sequential_bootstrap(base, t1="t1", stratify="entity", seed=11)
    r2 = pn.sample.sequential_bootstrap(more, t1="t1", stratify="entity", seed=11)
    f1 = base.sort(["id", "date"])
    f2 = more.sort(["id", "date"])
    keys1 = f1.select("id", "date")[r1.tolist()]
    keys2 = f2.select("id", "date")[r2.tolist()]
    for name in ("b", "c"):
        assert keys1.filter(pl.col("id") == name).equals(
            keys2.filter(pl.col("id") == name)
        )


_SUBPROCESS = r"""
import json, sys, datetime as dt
import numpy as np, polars as pl
import panelary as pn
from panelary._internal._jit import force_numpy
n = 40
df = pl.DataFrame({
    "id": ["x"] * n + ["y"] * n,
    "date": [dt.date(2020, 1, 1) + dt.timedelta(days=i % n) for i in range(2 * n)],
    "t1": [dt.date(2020, 1, 1) + dt.timedelta(days=min(i % n + 3, n - 1)) for i in range(2 * n)],
})
with force_numpy():  # the seeds are the subject here, not the backend
    rows = pn.sample.sequential_bootstrap(df, t1="t1", stratify="entity", seed=5)
print(json.dumps(rows.tolist()))
"""


def test_stratified_draws_identical_across_processes() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = []
    for salt in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS],
            capture_output=True,
            text=True,
            env=env,
            cwd=root,
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]


def test_uniqueness_iid_follows_uniqueness() -> None:
    df = _panel(n_ent=2, n_t=80)
    frame, spans = _spans_from_t1(df, t1="t1")
    rows = pn.sample.sequential_bootstrap(
        df, t1="t1", method="uniqueness_iid", seed=2, n_draws=500
    )
    u = np.random.default_rng(np.random.SeedSequence(2)).random(500)
    cs = np.cumsum(_uniqueness(spans))
    want = spans.label_row[
        np.minimum(np.searchsorted(cs, u * cs[-1], side="right"), len(spans) - 1)
    ]
    assert np.array_equal(rows, want)


def test_sequential_samples_are_more_unique_than_iid() -> None:
    """The point of the method (AFML 4.9 setup): bags overlap less than i.i.d. ones."""
    seq_u, iid_u = [], []
    for rep in range(150):
        rng = np.random.default_rng(rep)
        start = np.sort(rng.integers(0, 100, 10)).astype(np.int64)
        end = np.minimum(start + rng.integers(0, 5, 10), 99).astype(np.int64)
        draws = _seq_boot(start, end, rng.random(10))
        iid = rng.integers(0, 10, 10)
        for picks, acc in ((draws, seq_u), (iid, iid_u)):
            c = ref.ref_concurrency([(int(start[i]), int(end[i])) for i in picks], 100)
            acc.append(
                np.mean([np.mean(1.0 / c[start[i] : end[i] + 1]) for i in picks])
            )
    assert np.mean(seq_u) > np.mean(iid_u) + 0.02


class _Spy:
    """Records the design rows it is fitted on (column 0 carries a row id)."""

    seen: list[np.ndarray] = []

    def fit(self, X, y):
        _Spy.seen.append(X[:, 0].copy())
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.m)


def test_bagging_inside_cv_uses_only_training_spans() -> None:
    """T17: every bootstrap draw is a training-fold label of that fold."""
    df = (
        _panel(n_ent=3, n_t=90)
        .with_row_index("rid")
        .with_columns(pl.col("rid").cast(pl.Float64))
        .select("id", "date", "rid", "x", "t1", "label")
    )
    bag = pn.sample.SequentialBagging(
        _Spy(), n_estimators=3, t1="t1", target="label", features=["rid", "x"], seed=1
    )
    cv = pn.PurgedKFold(n_splits=3, t1="t1", embargo=2, return_indices=True)
    times = df.get_column("date").unique().sort()
    folds = list(cv.split(df))
    _Spy.seen.clear()
    pn.cross_validate(
        bag, df, y="label", cv=pn.PurgedKFold(n_splits=3, t1="t1", embargo=2)
    )
    assert len(_Spy.seen) == 3 * len(folds)
    for f, (train, _test) in enumerate(folds):
        rows_train = df.filter(pl.col("date").is_in(times.gather(train).implode()))
        allowed = set(rows_train.get_column("rid").to_list())
        for member in _Spy.seen[3 * f : 3 * f + 3]:
            assert set(member.tolist()) <= allowed


def test_bagging_predictions_and_determinism() -> None:
    df = _panel(n_ent=2, n_t=50)
    bag = pn.sample.SequentialBagging(
        _Spy(), n_estimators=4, t1="t1", features=["x"], seed=7
    )
    pred = (
        bag.fit(df, entity="id", time="date")
        .predict(df, entity="id", time="date")
        .collect()
    )
    assert pred.columns == ["id", "date", "prediction"]
    again = pn.sample.SequentialBagging(
        _Spy(), n_estimators=4, t1="t1", features=["x"], seed=7
    )
    again.fit(df, entity="id", time="date")
    assert all(
        np.array_equal(a, b) for a, b in zip(bag.samples_, again.samples_, strict=True)
    )
    frame, spans = _spans_from_t1(df, t1="t1")
    labels = frame.get_column("label").to_numpy()
    means = [labels[s].mean() for s in bag.samples_]
    np.testing.assert_allclose(
        pred.get_column("prediction").to_numpy(), np.mean(means), rtol=1e-12
    )


def test_bagging_classifier_votes_by_mean_probability() -> None:
    sk = pytest.importorskip("sklearn.linear_model")
    df = _panel(n_ent=2, n_t=80)
    bag = pn.sample.SequentialBagging(
        sk.LogisticRegression(), n_estimators=5, t1="t1", features=["x"], seed=0
    ).fit(df, entity="id", time="date")
    proba = bag.predict_proba(df, entity="id", time="date").collect()
    assert proba.columns == ["id", "date", "prediction_proba_-1", "prediction_proba_1"]
    np.testing.assert_allclose(
        proba.select(pl.sum_horizontal(pl.exclude("id", "date"))).to_series(), 1.0
    )
    pred = bag.predict(df, entity="id", time="date").collect().get_column("prediction")
    assert set(pred.unique().to_list()) <= {-1, 1}
