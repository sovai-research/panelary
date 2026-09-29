"""Market-state features: exact identities and the per-date driver (7.2).

The identities hold to 1e-12 on constructed windows whose sample
correlation matrix is known exactly.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from panelary.covariance._gram import window_stats
from panelary.covariance._state import (
    FEATURES,
    market_loading,
    market_state,
    spectrum_features,
    top_eigenvector,
)
from panelary.covariance._window import panel_matrix, window_at


def _exact_window(C: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """An ``(n, N)`` window whose sample covariance is exactly ``C``."""
    N = C.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, N))
    A -= A.mean(axis=0)
    Q, _ = np.linalg.qr(A)  # orthonormal columns, orthogonal to 1
    w, V = np.linalg.eigh(C)
    root = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    return math.sqrt(n - 1) * Q @ root


def _one_factor(N: int, rho: float) -> np.ndarray:
    return (1 - rho) * np.eye(N) + rho * np.ones((N, N))


def test_avg_corr_is_the_mean_off_diagonal_correlation() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 30)) + rng.standard_normal((80, 1))
    ws = window_stats(X)
    got = spectrum_features(ws, ["avg_corr"])["avg_corr"]
    R = np.corrcoef(X, rowvar=False)
    ref = (R.sum() - np.trace(R)) / (30 * 29)
    assert got == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize(("n", "p"), [(80, 30), (30, 80)])
def test_participation_ratio_from_the_gram_equals_dense(n: int, p: int) -> None:
    X = np.random.default_rng(2).standard_normal((n, p)) + np.linspace(0, 1, p)
    ws = window_stats(X)
    got = spectrum_features(ws, ["participation_ratio"])["participation_ratio"]
    C = ws.Z.T @ ws.Z / ws.n_eff
    assert got == pytest.approx(np.trace(C) ** 2 / np.sum(C * C), rel=1e-12)


def test_absorption_ratio_on_an_exact_one_factor_population() -> None:
    N, rho = 40, 0.3
    ws = window_stats(_exact_window(_one_factor(N, rho), 120))
    out = spectrum_features(ws, ["absorption_ratio", "lambda1_share"], ar_k=1)
    share = (1 + (N - 1) * rho) / N
    assert out["absorption_ratio"] == pytest.approx(share, abs=1e-12)
    assert out["lambda1_share"] == pytest.approx(share, abs=1e-12)
    assert out["ar_k"] == 1


def test_effective_rank_limits() -> None:
    N = 25
    ident = window_stats(_exact_window(np.eye(N), 90))
    assert spectrum_features(ident, ["effective_rank"])[
        "effective_rank"
    ] == pytest.approx(N, rel=1e-12)
    f = np.random.default_rng(3).standard_normal(90)
    rank1 = window_stats(
        np.outer(f - f.mean(), np.linspace(0.5, 2, N)), space="covariance"
    )
    out = spectrum_features(rank1, ["effective_rank", "eigen_entropy"])
    assert out["effective_rank"] == pytest.approx(1.0, abs=1e-12)
    assert out["eigen_entropy"] == 0.0


def test_ipr_of_a_uniform_market_mode_is_one_over_n() -> None:
    N = 30
    ws = window_stats(_exact_window(_one_factor(N, 0.4), 100))
    out = spectrum_features(ws, ["market_ipr"])
    assert out["market_ipr"] == pytest.approx(1 / N, abs=1e-12)
    v = top_eigenvector(ws)
    assert v.sum() > 0
    assert np.allclose(v, 1 / math.sqrt(N), atol=1e-12)


def test_ipr_is_null_without_a_spectral_gap() -> None:
    ws = window_stats(_exact_window(np.eye(20), 60))
    assert spectrum_features(ws, ["market_ipr"])["market_ipr"] is None


@pytest.mark.parametrize(("n", "p"), [(40, 200), (300, 60)])
def test_power_iteration_matches_eigh(n: int, p: int) -> None:
    rng = np.random.default_rng(4)
    X = rng.standard_normal((n, 1)) @ np.full((1, p), 1.2) + rng.standard_normal((n, p))
    ws = window_stats(X)
    v = top_eigenvector(ws)
    ref = window_stats(X).spectrum(vectors=True).vectors[:, 0]
    assert np.abs(v - ref).max() < 1e-10


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def _panel(n_ent: int = 24, n_t: int = 90, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = [dt.date(2023, 1, 2) + dt.timedelta(days=i) for i in range(n_t)]
    f = rng.standard_normal(n_t)
    rows = []
    for e in range(n_ent):
        r = 0.8 * f + rng.standard_normal(n_t)
        for t in range(n_t):
            rows.append((f"E{e:02d}", days[t], float(r[t]), f"S{e % 3}"))
    return pl.DataFrame(rows, schema=["entity", "time", "ret", "sector"], orient="row")


def test_market_state_rows_match_the_window_kernel() -> None:
    df = _panel()
    st = market_state(
        df, returns="ret", window=30, features=FEATURES, entity="entity", time="time"
    )
    pm = panel_matrix(df, "ret", entity="entity", time="time")
    for t in (29, 50, 89):
        ws, _ = window_at(pm, t, window=30)
        ref = spectrum_features(ws, [f for f in FEATURES if f != "ar_shift"])
        row = st.row(t, named=True)
        for k in (
            "absorption_ratio",
            "avg_corr",
            "participation_ratio",
            "effective_rank",
        ):
            assert row[k] == ref[k]
        assert row["n_entities"] == 24
        assert row["asof_date"] == row["time"]
    assert st.get_column("absorption_ratio")[:29].null_count() == 29


def test_group_variant_matches_a_manual_group_loop() -> None:
    df = _panel()
    st = market_state(
        df, returns="ret", window=30, group="sector", min_entities=5,
        features=("absorption_ratio", "avg_corr"), entity="entity", time="time",
    )  # fmt: skip
    assert set(st.get_column("sector").unique()) == {"S0", "S1", "S2"}
    for s in ("S0", "S2"):
        sub = df.filter(pl.col("sector") == s)
        ref = market_state(
            sub, returns="ret", window=30, features=("absorption_ratio", "avg_corr"),
            entity="entity", time="time",
        )  # fmt: skip
        got = st.filter(pl.col("sector") == s).drop("sector")
        assert got.select(ref.columns).equals(
            ref.filter(pl.col("time").is_in(got["time"].implode()))
        )


def test_broadcast_joins_on_time_and_group() -> None:
    df = _panel()
    out = market_state(
        df, returns="ret", window=30, group="sector", min_entities=5,
        features=("absorption_ratio",), broadcast=True, entity="entity", time="time",
    )  # fmt: skip
    assert out.height == df.height
    assert out.select("entity", "time").equals(df.select("entity", "time"))
    st = market_state(
        df, returns="ret", window=30, group="sector", min_entities=5,
        features=("absorption_ratio",), entity="entity", time="time",
    )  # fmt: skip
    one = out.filter((pl.col("entity") == "E04") & (pl.col("time") == df["time"][60]))
    ref = st.filter((pl.col("sector") == "S1") & (pl.col("time") == df["time"][60]))
    assert one.item(0, "absorption_ratio") == ref.item(0, "absorption_ratio")


def test_market_loading_is_the_oriented_top_eigenvector() -> None:
    df = _panel()
    ml = market_loading(df, returns="ret", window=30, entity="entity", time="time")
    pm = panel_matrix(df, "ret", entity="entity", time="time")
    ws, idx = window_at(pm, 70, window=30)
    v = top_eigenvector(ws)
    got = (
        ml.filter(pl.col("time") == pm.times[70])
        .sort("entity")["market_loading"]
        .to_numpy()
    )
    assert np.array_equal(got, v)
    assert (got > 0).all()


def test_bad_arguments() -> None:
    df = _panel(6, 20)
    with pytest.raises(ValueError, match="unknown feature"):
        market_state(
            df,
            returns="ret",
            window=5,
            features=("nope",),
            entity="entity",
            time="time",
        )
    with pytest.raises(ValueError, match="min_coverage"):
        market_state(
            df, returns="ret", window=5, min_coverage=0.0, entity="entity", time="time"
        )
    with pytest.raises(ValueError, match="not found"):
        market_state(df, returns="x", window=5, entity="entity", time="time")
    with pytest.raises(ValueError, match="overwrite"):
        market_state(
            df.with_columns(pl.lit(1.0).alias("avg_corr")), returns="ret", window=5,
            features=("avg_corr",), broadcast=True, entity="entity", time="time",
        )  # fmt: skip
