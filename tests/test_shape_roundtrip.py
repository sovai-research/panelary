"""Round trips and reconstruction bounds of the shape algebra (plan section 9, item 4).

* ``to_long(build_tensor(df))`` reproduces ``df``; the ragged policies do what
  they say; ``as_embedding`` / ``explode_embedding`` invert each other.
* ``inverse_transform`` reconstruction error is below a stated bound for a
  synthetic exactly-rank-``k`` input: ``< 1e-10`` for ``RandomizedPCA``,
  ``ColumnSubset``, ``CUR`` and ``PartialTucker`` at ``k`` = the true rank, and
  exact step / low-pass reconstructions for ``PAA`` / ``Spectral``.
* ``FrequentDirections.merge`` is associative -- exactly (to rounding) when no
  shrink fires, and within the proven error bar when one does.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.shape import (
    CUR,
    PAA,
    ColumnSubset,
    CountSketch,
    Delay,
    FrequentDirections,
    PartialTucker,
    Ragged,
    RaggedSequences,
    RandomizedPCA,
    Spectral,
    as_embedding,
    build_sequences,
    build_tensor,
    explode_embedding,
    mode_dot,
    to_long,
)
from panelary.shape._tensor import refuse_missing

BOUND = 1e-10


def _long(tensor: np.ndarray, *, times: np.ndarray | None = None) -> pl.DataFrame:
    E, T, F = tensor.shape
    t = np.arange(T) if times is None else times
    return pl.DataFrame(
        {
            "id": np.repeat([f"e{i:02d}" for i in range(E)], T),
            "t": np.tile(t, E),
            **{f"f{j}": tensor[:, :, j].reshape(-1) for j in range(F)},
        }
    )


def _rank_k_frame(
    n: int, d: int, k: int, seed: int, *, offset: float = 3.0
) -> pl.DataFrame:
    """An exactly rank-``k`` matrix after centring: ``S W + mean`` with zero-mean ``S``."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n, k))
    S -= S.mean(axis=0)
    M = S @ rng.standard_normal((k, d)) + offset * np.arange(1, d + 1)
    E = 5
    return pl.DataFrame(
        {
            "id": np.repeat([f"e{i}" for i in range(E)], n // E),
            "t": np.tile(np.arange(n // E), E),
            **{f"f{j}": M[:, j] * (j + 1) for j in range(d)},
        }
    )


def _matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select([c for c in df.columns if c.startswith("f")]).to_numpy()


# --------------------------------------------------------------------------- #
# 1. The materialization boundary
# --------------------------------------------------------------------------- #
def test_to_long_inverts_build_tensor_on_a_complete_panel() -> None:
    rng = np.random.default_rng(0)
    df = _long(rng.standard_normal((4, 9, 3)))
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
    pt = build_tensor(
        PanelFrame(shuffled, entity="id", time="t"),
        ["f0", "f1", "f2"],
        z_normalize=False,
    )
    back = to_long(pt, dtype=pl.Float64)
    assert back.equals(df), "to_long(build_tensor(df)) must reproduce df exactly"


def test_to_long_names_a_transformed_tensor_and_checks_shapes() -> None:
    rng = np.random.default_rng(1)
    df = _long(rng.standard_normal((3, 5, 2)))
    pt = build_tensor(
        PanelFrame(df, entity="id", time="t"), ["f0", "f1"], z_normalize=False
    )
    wider = pt._replace(tensor=np.concatenate([pt.tensor, pt.tensor], axis=2))
    out = to_long(wider, prefix="z", dtype=pl.Float64)
    assert out.columns == ["id", "t", "z_0", "z_1", "z_2", "z_3"]
    with pytest.raises(ValueError, match="prefix"):
        to_long(wider)
    with pytest.raises(ValueError, match="line up"):
        to_long(pt._replace(tensor=pt.tensor[:, :-1]))


def test_build_tensor_is_the_same_object_from_the_old_cluster_path() -> None:
    import panelary.shape as shape
    from panelary.cluster._tensor import PanelTensor as OldPT
    from panelary.cluster._tensor import build_tensor as old

    assert old is shape.build_tensor
    assert OldPT is shape.PanelTensor


def test_build_tensor_never_backfills() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b"],
            "t": [0, 1, 2, 1, 2],
            "x": [1.0, None, 3.0, 5.0, 6.0],
        }
    )
    pt = build_tensor(PanelFrame(df, entity="id", time="t"), ["x"], z_normalize=False)
    assert pt.tensor[0, 1, 0] == 1.0  # forward-filled from t=0
    assert np.isnan(
        pt.tensor[1, 0, 0]
    )  # b has no observation at t=0: never back-filled


_RAGGED = pl.DataFrame(
    {
        "id": ["a"] * 5 + ["b"] * 3 + ["c"] * 5,
        "t": [0, 1, 2, 3, 4, 2, 3, 4, 0, 1, 2, 3, 4],
        "x": np.arange(13, dtype=float),
    }
)
_RP = PanelFrame(_RAGGED, entity="id", time="t")


def test_ragged_refuse_names_the_offending_entities() -> None:
    with pytest.raises(ValueError, match=r"'b' \(3 obs\)"):
        build_sequences(_RP, ["x"])
    assert Ragged("refuse") is Ragged.REFUSE


def test_ragged_pad_pads_with_nan_and_to_long_drops_the_padding() -> None:
    pt = build_sequences(_RP, ["x"], ragged="pad")
    assert not isinstance(pt, RaggedSequences)
    assert pt.tensor.shape == (3, 5, 1)
    assert np.isnan(pt.tensor[1, 3:, 0]).all()
    assert pt.tensor[1, :3, 0].tolist() == [5.0, 6.0, 7.0]
    back = to_long(pt, dtype=pl.Float64).sort(["id", "t"])
    assert back.equals(_RAGGED.sort(["id", "t"]))


def test_ragged_truncate_keeps_the_most_recent_observations() -> None:
    pt = build_sequences(_RP, ["x"], ragged="truncate", length=3)
    assert not isinstance(pt, RaggedSequences)
    assert pt.tensor[:, :, 0].tolist() == [
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [10.0, 11.0, 12.0],
    ]
    with pytest.raises(ValueError, match="fewer"):
        build_sequences(_RP, ["x"], ragged="truncate", length=4)


def test_ragged_native_hands_through_unpadded_arrays() -> None:
    rs = build_sequences(_RP, ["x"], ragged="native")
    assert isinstance(rs, RaggedSequences)
    assert [a.shape[0] for a in rs.arrays] == [5, 3, 5]
    assert rs.keys[1].to_list() == [2, 3, 4]


def test_refuse_missing_policies_on_the_date_grid() -> None:
    pt = build_tensor(_RP, ["x"], forward_fill=False, z_normalize=False)
    with pytest.raises(ValueError, match="'b' \\(2 missing dates\\)"):
        refuse_missing(pt, owner="T")
    kept = refuse_missing(pt, owner="T", ragged="truncate", length=3)
    assert kept.tensor.shape == (3, 3, 1)
    assert kept.times.to_list() == [2, 3, 4]
    for policy in ("pad", "native"):
        with pytest.raises(ValueError):
            refuse_missing(pt, owner="T", ragged=policy)


def test_embedding_round_trip() -> None:
    rng = np.random.default_rng(2)
    df = pl.DataFrame(
        {"id": ["a"] * 4, **{f"v_{j}": rng.standard_normal(4) for j in range(6)}}
    )
    cols = [f"v_{j}" for j in range(6)]
    packed = as_embedding(df, cols, name="emb", dtype=pl.Float64)
    assert packed.schema["emb"] == pl.Array(pl.Float64, 6)
    assert explode_embedding(packed, "emb", prefix="v").equals(df)
    f32 = as_embedding(df, cols, name="emb")
    assert f32.schema["emb"] == pl.Array(pl.Float32, 6)
    with pytest.raises(ValueError, match="fixed-width"):
        explode_embedding(df, "id")


# --------------------------------------------------------------------------- #
# 2. Reconstruction on exactly rank-k input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("standardize", [True, False])
def test_randomized_pca_reconstructs_exact_rank_k(standardize: bool) -> None:
    df = _rank_k_frame(200, 12, 4, seed=3)
    p = RandomizedPCA(
        n_components=4, standardize=standardize, entity="id", time="t"
    ).fit(df)
    Z = p.transform(df).collect()
    rec = p.inverse_transform(Z)
    X = _matrix(df)
    assert np.linalg.norm(rec - X) / np.linalg.norm(X) < BOUND
    assert p.reconstruction_error(df) < BOUND
    assert p.reconstruction_error_ is not None and p.reconstruction_error_ < 1e-6


def test_randomized_pca_error_is_a_number_below_rank() -> None:
    df = _rank_k_frame(200, 12, 4, seed=4)
    err = (
        RandomizedPCA(n_components=2, entity="id", time="t")
        .fit(df)
        .reconstruction_error(df)
    )
    assert 1e-3 < err < 1.0


@pytest.mark.parametrize("cls", [ColumnSubset, CUR], ids=lambda c: c.__name__)
def test_id_and_cur_reconstruct_exact_rank_k(cls: type[ColumnSubset]) -> None:
    df = _rank_k_frame(150, 10, 3, seed=5)
    t = cls(k=3, entity="id", time="t").fit(df)
    out = t.transform(df).collect()
    X = _matrix(df)
    kept = out.select(t.selected_).to_numpy()
    assert np.array_equal(kept, df.select(t.selected_).to_numpy()), (
        "kept columns must be real, unchanged"
    )
    rec = t.inverse_transform(out)
    assert np.linalg.norm(rec - X) / np.linalg.norm(X) < BOUND
    assert t.reconstruction_error_ is not None and t.reconstruction_error_ < BOUND
    assert np.allclose(rec[:, t.selected_idx_], kept, rtol=1e-12, atol=1e-12)


def test_cur_factors_name_real_rows() -> None:
    df = _rank_k_frame(150, 10, 3, seed=6)
    f = CUR(k=3, k_rows=4, entity="id", time="t").fit(df).factors()
    assert f.rows.columns == ["id", "t"] and f.rows.height == 3  # rank 3 caps the rows
    joined = f.rows.join(df, on=["id", "t"], how="inner")
    assert joined.height == f.rows.height


def test_partial_tucker_reconstructs_exact_multilinear_rank() -> None:
    rng = np.random.default_rng(7)
    G = rng.standard_normal((6, 3, 2))
    Ut = np.linalg.qr(rng.standard_normal((25, 3)))[0]
    Uf = np.linalg.qr(rng.standard_normal((5, 2)))[0]
    X = mode_dot(mode_dot(G, Ut, 1), Uf, 2)
    df = _long(X)
    pt = PartialTucker({"time": 3, "feature": 2}, entity="id", time="t")
    out = pt.fit_transform(df).collect()
    assert out.height == 6 and pt.reconstruction_error_ is not None
    assert pt.reconstruction_error_ < BOUND
    rec = pt.inverse_transform(out)
    assert np.linalg.norm(rec - X) / np.linalg.norm(X) < BOUND


def test_paa_step_reconstruction_is_exact_on_piecewise_constant_windows() -> None:
    # Every trailing window of a series that is constant on aligned blocks of
    # length L = W // S is itself piecewise constant on the PAA segments only when
    # the window is block-aligned, so test the inverse directly on segment means.
    p = PAA(window=6, segments=3).fit(
        pl.DataFrame({"id": ["a"], "t": [0], "x": [0.0]}), entity="id", time="t"
    )
    block = np.array([[[1.0, 2.0, 3.0]]])  # (n=1, k=1, S=3)
    rec = p.inverse_transform(block)
    assert rec.shape == (1, 6, 1)
    assert rec[0, :, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    x = np.repeat([4.0, -1.0, 2.5, 7.0], 2)
    df = pl.DataFrame({"id": ["a"] * 8, "t": np.arange(8), "x": x})
    p = PAA(window=4, segments=2).fit(df, entity="id", time="t")
    Z = p.transform(df).collect()
    full = Z.filter(pl.col("t").is_in([3, 5, 7]))  # block-aligned full windows
    rec = p.inverse_transform(full)[:, :, 0]
    assert rec.tolist() == [x[0:4].tolist(), x[2:6].tolist(), x[4:8].tolist()]
    with pytest.raises(ValueError, match="no reconstruction"):
        PAA(window=4, segments=2, pool="max").fit(
            df, entity="id", time="t"
        ).inverse_transform(Z)


def test_spectral_with_every_bin_and_phase_is_exactly_invertible() -> None:
    rng = np.random.default_rng(8)
    W = 8
    df = pl.DataFrame(
        {"id": ["a"] * 30, "t": np.arange(30), "x": rng.standard_normal(30)}
    )
    s = Spectral(window=W, k=W // 2 + 1, phase=True).fit(df, entity="id", time="t")
    Z = s.transform(df).collect()
    rows = Z.filter(pl.col("t") >= W - 1)
    rec = s.inverse_transform(rows)[:, :, 0]
    x = df["x"].to_numpy()
    windows = np.stack([x[t - W + 1 : t + 1] for t in range(W - 1, 30)])
    assert np.max(np.abs(rec - windows)) < BOUND
    with pytest.raises(ValueError, match="phase"):
        Spectral(window=W, k=3).fit(df, entity="id", time="t").inverse_transform(rows)


def test_delay_lag_zero_is_the_input() -> None:
    rng = np.random.default_rng(9)
    x = rng.standard_normal(12)
    df = pl.DataFrame({"id": ["a"] * 12, "t": np.arange(12), "x": x})
    out = (
        Delay(lags=3, dilation=2, dtype=pl.Float64)
        .fit_transform(df, entity="id", time="t")
        .collect()
    )
    assert np.array_equal(out["x__lag_0"].to_numpy(), x)
    assert np.array_equal(out["x__lag_4"].to_numpy()[4:], x[:-4])
    assert np.isnan(out["x__lag_4"].to_numpy()[:4]).all()


# --------------------------------------------------------------------------- #
# 3. Sketch merges
# --------------------------------------------------------------------------- #
def _fd(rows: np.ndarray, ell: int) -> FrequentDirections:
    df = pl.DataFrame(
        {
            "id": ["a"] * rows.shape[0],
            "t": np.arange(rows.shape[0]),
            **{f"f{j}": rows[:, j] for j in range(rows.shape[1])},
        }
    )
    return FrequentDirections(ell=ell, n_components=2, entity="id", time="t").fit(df)


def _gram(fd: FrequentDirections) -> np.ndarray:
    assert fd.sketch_ is not None
    return fd.sketch_.T @ fd.sketch_


def test_frequent_directions_merge_is_exactly_associative_without_shrinkage() -> None:
    rng = np.random.default_rng(10)
    low = rng.standard_normal((90, 3)) @ rng.standard_normal((3, 8))  # rank 3 < ell
    a, b, c = (_fd(low[i : i + 30], ell=6) for i in (0, 30, 60))
    left = a.merge(b).merge(c)
    right = a.merge(b.merge(c))
    full = low.T @ low
    scale = np.linalg.norm(full)
    assert np.linalg.norm(_gram(left) - _gram(right)) / scale < 1e-10
    assert np.linalg.norm(_gram(left) - full) / scale < 1e-10


def test_frequent_directions_merge_is_associative_up_to_the_bound() -> None:
    rng = np.random.default_rng(11)
    A = rng.standard_normal((120, 10)) * np.linspace(3.0, 0.2, 10)
    parts = [_fd(A[i : i + 40], ell=4) for i in (0, 40, 80)]
    before = [p.sketch_.copy() for p in parts]  # type: ignore[union-attr]
    left = parts[0].merge(parts[1]).merge(parts[2])
    right = parts[0].merge(parts[1].merge(parts[2]))
    for p, s in zip(parts, before, strict=True):
        assert np.array_equal(p.sketch_, s), "merge must not modify its inputs"
    bound = float(np.sum(A * A)) / 4
    for merged in (left, right):
        assert merged.frob2_ == pytest.approx(float(np.sum(A * A)))
        err = np.linalg.norm(A.T @ A - _gram(merged), 2)
        assert err <= bound * (1 + 1e-12)
        assert merged.covariance_error_bound() == pytest.approx(bound)
        assert merged.n_seen_ == 120


def test_frequent_directions_merge_refuses_mismatched_sketches() -> None:
    rng = np.random.default_rng(12)
    A = rng.standard_normal((20, 5))
    with pytest.raises(ValueError, match="ell"):
        _fd(A, ell=4).merge(_fd(A, ell=3))


def _cs(seed: int, df: pl.DataFrame) -> CountSketch:
    return CountSketch(n_components=4, seed=seed, entity="id", time="t").fit(df)


def test_count_sketch_is_linear_and_merges_only_with_its_own_hash() -> None:
    rng = np.random.default_rng(13)
    X1, X2 = rng.standard_normal((10, 9)), rng.standard_normal((10, 9))

    def frame(X: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id": ["a"] * 10,
                "t": np.arange(10),
                **{f"f{j}": X[:, j] for j in range(9)},
            }
        )

    cs = _cs(3, frame(X1))
    s = lambda X: cs.transform(frame(X)).collect().drop("id", "t").to_numpy()  # noqa: E731
    assert np.allclose(s(X1 + X2), s(X1) + s(X2), rtol=0, atol=1e-12)
    assert cs.merge(_cs(3, frame(X2))) is cs
    with pytest.raises(ValueError, match="share seed"):
        cs.merge(_cs(4, frame(X2)))
