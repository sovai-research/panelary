"""Dependence matrices, distances, feature screening and ScreenSelector."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp
from panelary.core.panel_frame import PanelFrame


def _frame(n_ent: int = 5, t_len: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_ent * t_len
    a = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_ent), t_len),
            "t": np.tile(np.arange(t_len), n_ent),
            "a": a,
            "b": a**2 + 0.2 * rng.standard_normal(n),
            "c": rng.standard_normal(n),
            "d": -a + 0.5 * rng.standard_normal(n),
        }
    )


def test_pooled_matrix_matches_pairwise() -> None:
    df = _frame(n_ent=1)
    cols = ["a", "b", "c", "d"]
    for method, fn in [
        ("xi", dp.xi),
        ("spearman", dp.spearman),
        ("hoeffding", dp.hoeffding_d),
        ("dcor", dp.dcor),
    ]:
        M, names = dp.matrix_values(
            dp.dependence_matrix(df, cols, method=method, by="pooled")
        )
        assert names == cols
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                assert M[i, j] == pytest.approx(
                    fn(df[ci].to_numpy(), df[cj].to_numpy()), abs=1e-10
                ), (method, ci, cj)


def test_matrix_direction_symmetrise_and_wide() -> None:
    df = _frame()
    long = dp.dependence_matrix(df, ["a", "b", "c"], entity="e", time="t", method="xi")
    assert long.height == 9 and long["direction"][0] == "x->y"
    M, _ = dp.matrix_values(long)
    assert M[0, 1] > 0.7 > 0.4 > M[1, 0]  # b is a function of a, not vice versa
    sym = dp.dependence_matrix(
        df, ["a", "b", "c"], entity="e", time="t", method="xi", symmetrise="max"
    )
    S, _ = dp.matrix_values(sym)
    np.testing.assert_allclose(S, S.T)
    assert sym["transform"][0] == "symmetrised(max)"
    wide = dp.dependence_matrix(
        df, ["a", "b", "c"], entity="e", time="t", method="spearman", output="wide"
    )
    assert wide.columns == ["feature", "a", "b", "c"]
    W, _ = dp.matrix_values(wide)
    np.testing.assert_allclose(W, W.T, atol=1e-12)


def test_matrix_by_entity_caps_are_recorded() -> None:
    df = _frame(n_ent=6, t_len=200)
    out = dp.dependence_matrix(
        df,
        ["a", "d"],
        entity="e",
        time="t",
        method="spearman",
        max_entities=3,
        subsample=100,
        seed=4,
    )
    assert (
        out["approximate"].all()
        and out["n_entities"][0] == 3
        and out["n_obs"][0] == 300
    )
    assert out["seed"][0] == 4
    exact = dp.dependence_matrix(
        df, ["a", "d"], entity="e", time="t", method="spearman"
    )
    assert not exact["approximate"].any()
    with pytest.raises(ValueError, match="pick a tail"):
        dp.dependence_matrix(df, ["a", "d"], method="tail_dependence")


def test_to_distance_and_psd_repair() -> None:
    df = _frame(n_ent=1)
    M, _ = dp.matrix_values(
        dp.dependence_matrix(df, ["a", "c", "d"], method="spearman", by="pooled")
    )
    D = dp.to_distance(M)
    assert np.allclose(np.diag(D), 0) and np.allclose(D, D.T)
    assert (
        D[0, 2] > D[0, 1]
    )  # a and d are anti-correlated: far apart on the angular scale
    Da = dp.to_distance(M, kind="absolute")
    assert Da[0, 2] < Da[0, 1]
    with pytest.raises(ValueError, match="not symmetric"):
        dp.to_distance(
            dp.matrix_values(
                dp.dependence_matrix(df, ["a", "b"], method="xi", by="pooled")
            )[0]
        )
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    R, lam = dp.psd_repair(bad)
    assert (
        lam < 0 and np.linalg.eigvalsh(R).min() > -1e-10 and np.allclose(np.diag(R), 1)
    )
    with pytest.warns(RuntimeWarning, match="not PSD"):
        dp.to_distance(bad, repair=True)
    G = dp.gcmi_matrix(np.column_stack([df["a"], df["d"], df["c"]]))
    Di = dp.to_distance(G, kind="info")
    assert Di[0, 1] < Di[0, 2] <= 1.0


def test_feature_screen_ranks_and_corrects() -> None:
    df = _frame()
    sc = dp.feature_screen(
        df, "b", ["a", "c", "d"], entity="e", time="t", method="xi", n_resamples=99
    )
    assert sc.columns[:2] == ["feature", "target"]
    assert sc["feature"][0] == "a"
    assert sc["null_method"].unique().to_list() == ["common-time"]
    assert sc.filter(pl.col("feature") == "c")["p_value_adj"][0] > 0.05
    assert (sc["p_value_adj"] >= sc["p_value"] - 1e-12).all()
    # Default feature list: every numeric non-key, non-target column.
    auto = dp.feature_screen(
        df, "b", entity="e", time="t", method="spearman", null="asymptotic"
    )
    assert set(auto["feature"]) == {"a", "c", "d"}
    rw = dp.feature_screen(
        df,
        "b",
        ["a", "c", "d"],
        entity="e",
        time="t",
        method="xi",
        correction="romano_wolf",
        n_resamples=99,
    )
    assert any("romano_wolf" in w for w in rw["warnings"][0])


def test_feature_screen_warns_when_resamples_cannot_survive_correction() -> None:
    df = _frame(n_ent=3, t_len=100)
    sc = dp.feature_screen(
        df, "b", ["a", "c", "d"], entity="e", time="t", n_resamples=19, alpha=0.05
    )
    assert any("n_resamples >=" in w for w in sc["warnings"][0])


def test_screen_selector_fits_on_train_only() -> None:
    df = _frame(n_ent=4, t_len=250)
    pf = PanelFrame(df, entity="e", time="t")
    sel = dp.ScreenSelector("b", features=["a", "c", "d"], method="xi", n_resamples=99)
    out = sel.fit(pf).transform(pf).collect()
    assert "a" in sel.selected_ and "c" not in sel.selected_
    assert out.columns[:2] == ["e", "t"] and "b" in out.columns
    assert sel.screen_ is not None and sel.screen_.height == 3
    assert dp.ScreenSelector.panel_safe and dp.ScreenSelector.leakage_safe
    top1 = dp.ScreenSelector("b", features=["a", "c", "d"], k=1, n_resamples=99).fit(pf)
    assert top1.selected_ == ["a"]
    with pytest.raises(ValueError):
        dp.ScreenSelector("b", k=0)
    empty = dp.ScreenSelector("b", features=["c"], n_resamples=19)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        empty.fit(pf)
    assert empty.selected_ == [] and any("no feature" in str(w.message) for w in rec)
