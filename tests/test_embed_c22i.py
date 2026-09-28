"""RandIntC22: catch24 over random dilated intervals (`panelary.embed.RandIntC22`).

Contracts touched: ``leakage_safe`` via trailing windows and the
``fit_is_empty = True`` claim (the intervals depend on the window length and
the seed only). Numerics are checked against the scalar ``catch22_all``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.catch22 import CATCH22_NAMES, catch22_all
from panelary.embed import RandIntC22

E, T = "entity", "time"


def _panel(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 70
    return pl.DataFrame(
        {
            E: np.repeat(["a", "b"], n),
            T: np.tile(np.arange(n, dtype=np.int64), 2),
            "x": rng.standard_normal(2 * n).cumsum(),
        }
    )


def test_features_equal_scalar_catch24_on_each_interval():
    rc = RandIntC22(window=40, n_intervals=5, seed=2)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 40)).cumsum(axis=1)
    F = rc._embed_windows(W)
    assert F.shape == (6, 5 * 24)
    for i in (0, 5):
        for j in range(5):
            ref = catch22_all(W[i, rc.intervals_.indices(j)], catch24=True)
            np.testing.assert_allclose(
                F[i, j * 24 : (j + 1) * 24],
                list(ref.values()),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )


def test_rows_are_independent_of_the_batch():
    rc = RandIntC22(window=32, n_intervals=4, seed=0)
    W = np.random.default_rng(3).standard_normal((20, 32))
    np.testing.assert_array_equal(rc._embed_windows(W[:2]), rc._embed_windows(W)[:2])


def test_feature_names_and_catch22_only():
    rc = RandIntC22(window=32, n_intervals=3, catch24=False)
    assert rc.n_outputs == 3 * 22
    assert rc.feature_names_[:22] == [f"i0_{n}" for n in CATCH22_NAMES]
    assert RandIntC22(window=32, n_intervals=3).feature_names_[23] == "i0_DN_Spread_Std"


def test_intervals_depend_on_window_and_seed_only():
    a = RandIntC22(window=48, n_intervals=6, seed=7)
    b = RandIntC22(window=48, n_intervals=6, seed=7)
    np.testing.assert_array_equal(a.interval_start_, b.interval_start_)
    c = RandIntC22(window=48, n_intervals=6, seed=8)
    assert not np.array_equal(a.interval_start_, c.interval_start_)


def test_transform_and_state_roundtrip():
    df = _panel()
    rc = RandIntC22(
        window=24, n_intervals=3, warmup="null", dtype="float64", entity=E, time=T
    )
    out = rc.fit_transform(df).collect().sort(E, T)
    assert out.schema["x_c22i"] == pl.Array(pl.Float64, 72)
    x = df.filter(pl.col(E) == "b").sort(T)["x"].to_numpy()
    row = np.asarray(out.filter(pl.col(E) == "b")["x_c22i"][50])
    np.testing.assert_allclose(
        row, rc._embed_windows(x[None, 27:51])[0], equal_nan=True
    )
    again = RandIntC22.from_state(rc.get_state())
    assert again.transform(df).collect().sort(E, T).equals(out)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        RandIntC22(n_intervals=0)
    with pytest.raises(TypeError):
        RandIntC22(seed=1.5)  # type: ignore[arg-type]
