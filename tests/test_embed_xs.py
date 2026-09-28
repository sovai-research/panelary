"""Cross-sectional mode (`panelary.embed.CrossSectionalEmbedder`, `procrustes_rotation`).

Contracts touched: ``is_cross_sectional`` (every per-date fit sees only that
date's rows), ``panel_safe = False`` by design, ``leakage_safe`` (alignment
reads dates ``<= t`` only; the perturbation / prefix suites cover it), and the
section-4 refusal: unaligned per-date components are never emitted.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from panelary.embed import CrossSectionalEmbedder, procrustes_rotation
from panelary.embed._xs import _standardise

E, T = "entity", "time"


def _panel(n_ent: int = 40, n_time: int = 12, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_time, n_ent))
    a = f + 0.3 * rng.standard_normal((n_time, n_ent))
    b = 0.8 * f + 0.3 * rng.standard_normal((n_time, n_ent))
    c = rng.standard_normal((n_time, n_ent))
    return pl.DataFrame(
        {
            E: np.tile([f"s{i:02d}" for i in range(n_ent)], n_time),
            T: np.repeat(np.arange(n_time, dtype=np.int64), n_ent),
            "a": a.ravel(),
            "b": b.ravel(),
            "c": c.ravel() * 2,
        }
    )


def _mat(frame: pl.DataFrame, col: str = "xs_embedding") -> np.ndarray:
    return np.asarray(frame[col].to_list(), dtype=np.float64)


def _xs(**kw: object) -> CrossSectionalEmbedder:
    params: dict[str, object] = {
        "align": "procrustes",
        "n_components": 2,
        "min_cross_section": 10,
        "dtype": "float64",
    }
    params.update(kw)
    return CrossSectionalEmbedder(entity=E, time=T, **params)  # type: ignore[arg-type]


def test_align_is_required_and_there_is_no_unaligned_option():
    assert (
        inspect.signature(CrossSectionalEmbedder).parameters["align"].default
        is inspect.Parameter.empty
    )
    with pytest.raises(TypeError):
        CrossSectionalEmbedder()  # type: ignore[call-arg]
    for bad in (None, "none", "raw"):
        with pytest.raises(ValueError, match="Gabaix"):
            CrossSectionalEmbedder(align=bad)  # type: ignore[arg-type]


def test_class_contract():
    assert CrossSectionalEmbedder.is_cross_sectional is True
    assert CrossSectionalEmbedder.panel_safe is False
    assert CrossSectionalEmbedder.leakage_safe is True
    assert CrossSectionalEmbedder.fit_is_empty is True


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(0)
    V = np.linalg.qr(rng.standard_normal((6, 3)))[0]
    Q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    R = procrustes_rotation(V @ Q, V)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(V @ Q @ R, V, atol=1e-12)
    with pytest.raises(ValueError, match="shapes"):
        procrustes_rotation(V, V[:, :2])


def test_scores_span_each_dates_own_principal_subspace():
    df = _panel()
    out = _xs().fit_transform(df).collect()
    for t in (0, 7):
        Z = _mat(out.filter(pl.col(T) == t).sort(E))
        M = df.filter(pl.col(T) == t).sort(E).select("a", "b", "c").to_numpy()
        Ms = _standardise(M, "z")
        U, s, _ = np.linalg.svd(Ms, full_matrices=False)
        ref = U[:, :2] * s[:2]
        P = lambda A: A @ np.linalg.pinv(A)  # noqa: E731
        np.testing.assert_allclose(P(Z), P(ref), atol=1e-9)


def test_alignment_makes_consecutive_loadings_continuous():
    df = _panel(n_time=20)
    xs = _xs().fit(df)
    L = xs.date_loadings(df).sort(T, "component", "input_index")
    V = L["loading"].to_numpy().reshape(20, 2, 3)
    raw = xs.date_loadings(df, aligned=False).sort(T, "component", "input_index")
    Vr = raw["loading"].to_numpy().reshape(20, 2, 3)
    aligned_dots = np.einsum("tki,tki->tk", V[1:], V[:-1])
    raw_dots = np.einsum("tki,tki->tk", Vr[1:], Vr[:-1])
    # Aligned: each component points the same way as yesterday's.
    assert (aligned_dots[:, 0] > 0.9).all()
    # The raw per-date axes are not identified: their signs flip.
    assert (raw_dots[:, 0] < 0).any() or (raw_dots[:, 1] < 0).any()


def test_link_penalty_shrinks_towards_yesterdays_score():
    df = _panel()
    plain = _mat(_xs().fit_transform(df).collect().sort(T, E)).reshape(12, 40, 2)
    lam = 2.0
    linked = _mat(
        _xs(align="link", link_penalty=lam).fit_transform(df).collect().sort(T, E)
    ).reshape(12, 40, 2)
    np.testing.assert_allclose(linked[0], plain[0])
    np.testing.assert_allclose(
        linked[1], (plain[1] + lam * linked[0]) / (1 + lam), atol=1e-12
    )


def test_thin_dates_are_null_and_the_chain_continues():
    df = _panel()
    df = df.filter(~((pl.col(T) == 5) & (pl.col(E) > "s05")))  # date 5 keeps 6 entities
    out = _xs().fit_transform(df).collect()
    assert out.filter(pl.col(T) == 5)["xs_embedding"].null_count() == 6
    assert out.filter(pl.col(T) != 5)["xs_embedding"].null_count() == 0


def test_permutation_equivariant_and_rank_invariant():
    df = _panel()
    base = _xs().fit_transform(df).collect().sort(E, T)
    shuffled = (
        _xs()
        .fit_transform(df.sample(fraction=1.0, shuffle=True, seed=3))
        .collect()
        .sort(E, T)
    )
    np.testing.assert_allclose(_mat(base), _mat(shuffled), atol=1e-12)
    rank = _xs(standardize="rank").fit_transform(df).collect().sort(E, T)
    monotone = df.with_columns(pl.col("a").exp(), pl.col("c") ** 3)
    rank2 = _xs(standardize="rank").fit_transform(monotone).collect().sort(E, T)
    np.testing.assert_allclose(_mat(rank), _mat(rank2), atol=1e-12)


def test_array_inputs_state_and_validation():
    df = _panel().with_columns(
        pl.concat_list("a", "b", "c").list.to_array(3).alias("emb")
    )
    xs = _xs(columns=["emb"]).fit(df)
    out = xs.transform(df).collect()
    ref = _xs(columns=["a", "b", "c"]).fit(df).transform(df).collect()
    np.testing.assert_allclose(_mat(out), _mat(ref))
    again = CrossSectionalEmbedder.from_state(xs.get_state())
    assert again.transform(df).collect().equals(out)
    with pytest.raises(ValueError, match="n_components"):
        _xs(n_components=4, columns=["a", "b", "c"]).fit(df)
    with pytest.raises(ValueError, match="exceed"):
        _xs(n_components=5, min_cross_section=5)
    with pytest.raises(ValueError, match="link_penalty"):
        _xs(align="link", link_penalty=0.0)
