"""Tests for ``panelary.clean`` deduplication (M1): exact + MinHash-LSH.

Covers the kernels (union-find, sketches, LSH), the oracle comparison against
brute-force Jaccard, the ``near_duplicate_clusters`` contract other packages
build on, the point-in-time guarantee of :class:`Deduplicator` (prefix
invariance, with a deliberately leaky linkage that must fail it), the
split-aware guarantee (no near-duplicate pair straddles a fold), and the
``preprocessing.reindex(drop_duplicates=True)`` reconciliation.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from panelary.clean import (
    Deduplicator,
    SplitAwareCV,
    assert_no_straddle,
    connected_components,
    dedup,
    exact_unique,
    near_duplicate_clusters,
    near_duplicate_pairs,
    purge_near_duplicates,
    straddling_clusters,
    straddling_pairs,
)
from panelary.clean._cluster import later_endpoints
from panelary.clean._sketch import (
    bbit,
    estimate_jaccard,
    lsh_candidate_pairs,
    lsh_params,
    minhash_signatures,
    simhash_bits,
)
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _cells_panel(n: int = 300, n_cols: int = 20, seed: int = 0) -> pl.DataFrame:
    """Random integer rows; cells almost never collide by chance."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10_000, size=(n, n_cols))
    data: dict[str, object] = {
        "id": [f"e{i % 7}" for i in range(n)],
        "t": np.arange(n) // 7,
    }
    for j in range(n_cols):
        data[f"f{j}"] = X[:, j]
    return pl.DataFrame(data)


def _perturb(df: pl.DataFrame, src: int, dst: int, n_changed: int) -> pl.DataFrame:
    """Make row ``dst`` a copy of row ``src`` with ``n_changed`` cells altered."""
    feats = [c for c in df.columns if c.startswith("f")]
    row = df.row(src, named=True)
    for j, c in enumerate(feats):
        val = row[c] + 10_000_000 + dst if j < n_changed else row[c]
        df = df.with_columns(
            pl.when(pl.int_range(pl.len()) == dst)
            .then(pl.lit(val))
            .otherwise(pl.col(c))
            .alias(c)
        )
    return df


def _cell_sets(df: pl.DataFrame) -> list[set[tuple[str, object]]]:
    feats = [c for c in df.columns if c not in ("id", "t")]
    return [{(c, r[c]) for c in feats} for r in df.iter_rows(named=True)]


def _brute_pairs(df: pl.DataFrame, threshold: float) -> dict[tuple[int, int], float]:
    sets = _cell_sets(df)
    out = {}
    for i, j in itertools.combinations(range(len(sets)), 2):
        a, b = sets[i], sets[j]
        jac = len(a & b) / len(a | b)
        if jac >= threshold:
            out[(i, j)] = jac
    return out


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def _bfs_components(n, ei, ej):
    adj = [[] for _ in range(n)]
    for a, b in zip(ei, ej, strict=True):
        adj[a].append(b)
        adj[b].append(a)
    label = [-1] * n
    for s in range(n):
        if label[s] >= 0:
            continue
        stack, comp = [s], []
        label[s] = s
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if label[v] < 0:
                    label[v] = s
                    stack.append(v)
        m = min(comp)
        for u in comp:
            label[u] = m
    return np.array(label)


@pytest.mark.parametrize("seed", range(5))
def test_connected_components_matches_bfs(seed):
    rng = np.random.default_rng(seed)
    n = 200
    m = int(rng.integers(0, 250))
    ei = rng.integers(0, n, m)
    ej = rng.integers(0, n, m)
    got = connected_components(n, ei, ej)
    assert got.dtype == np.int64
    assert np.array_equal(got, _bfs_components(n, ei, ej))


def test_connected_components_long_chain_and_errors():
    n = 1000
    ei = np.arange(n - 1)[::-1]
    ej = ei + 1
    assert (connected_components(n, ei, ej) == 0).all()
    assert connected_components(3, np.array([]), np.array([])).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        connected_components(3, np.array([0]), np.array([3]))
    with pytest.raises(ValueError):
        connected_components(3, np.array([0, 1]), np.array([1]))


def test_later_endpoints_is_pairwise():
    # 0 -- 1 -- 2 chain, ranks = ids: 1 and 2 each have an earlier neighbour.
    mask = later_endpoints(3, np.array([0, 1]), np.array([1, 2]), np.arange(3))
    assert mask.tolist() == [False, True, True]
    # Reverse the order: now 0 and 1 are the later endpoints.
    mask = later_endpoints(3, np.array([0, 1]), np.array([1, 2]), np.arange(3)[::-1])
    assert mask.tolist() == [True, True, False]


@pytest.mark.parametrize("variant", ["minhash", "cminhash"])
def test_minhash_estimates_jaccard(variant):
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(40):
        base = rng.choice(10**9, size=200, replace=False)
        k = int(rng.integers(0, 200))
        other = np.concatenate([base[:k], rng.choice(10**9, 200 - k) + 10**9])
        a, b = set(base.tolist()), set(other.tolist())
        true = len(a & b) / len(a | b)
        rows = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(np.int64)
        h = np.array(sorted(a) + sorted(b), dtype=np.uint64)
        sig = minhash_signatures(rows, h, 2, num_perm=256, seed=3, variant=variant)
        errs.append(abs(estimate_jaccard(sig[0], sig[1]) - true))
    assert np.mean(errs) < 0.04
    assert np.max(errs) < 0.15


def test_minhash_signature_is_deterministic_and_seeded():
    rows = np.array([0, 0, 1], dtype=np.int64)
    h = np.array([5, 9, 5], dtype=np.uint64)
    s1 = minhash_signatures(rows, h, 3, num_perm=16, seed=7)
    s2 = minhash_signatures(rows, h, 3, num_perm=16, seed=7)
    s3 = minhash_signatures(rows, h, 3, num_perm=16, seed=8)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    # Row 2 has no tokens: all slots are the empty sentinel.
    assert (s1[2] == np.iinfo(np.uint64).max).all()
    with pytest.raises(ValueError, match="sorted"):
        minhash_signatures(rows[::-1].copy(), h, 3, num_perm=4)


def test_bbit_correction_removes_the_collision_bias():
    rng = np.random.default_rng(2)
    naive_bias, corrected_bias = [], []
    for _ in range(30):
        a = set(rng.choice(10**8, 300, replace=False).tolist())
        b = set(list(a)[:200]) | set((rng.choice(10**8, 100) + 10**8).tolist())
        true = len(a & b) / len(a | b)
        rows = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(np.int64)
        h = np.array(sorted(a) + sorted(b), dtype=np.uint64)
        sig = minhash_signatures(rows, h, 2, num_perm=256, seed=int(rng.integers(99)))
        small = bbit(sig, 4)
        assert small.dtype == np.uint8
        naive_bias.append(float(np.mean(small[0] == small[1])) - true)
        corrected_bias.append(
            float(estimate_jaccard(small[0], small[1], b_bits=4)) - true
        )
    # 4-bit slots collide with probability 1/16: a positive bias of ~(1-J)/16.
    assert np.mean(naive_bias) > 0.015
    assert abs(np.mean(corrected_bias)) < 0.015
    with pytest.raises(ValueError):
        bbit(sig, 0)


def test_simhash_bits_track_angle():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20)
    y = x + 0.05 * rng.standard_normal(20)
    z = rng.standard_normal(20)
    bits = simhash_bits(np.vstack([x, y, z]), n_bits=256, seed=1)
    agree_xy = np.mean(bits[0] == bits[1])
    agree_xz = np.mean(bits[0] == bits[2])
    cos_xz = x @ z / np.linalg.norm(x) / np.linalg.norm(z)
    assert agree_xy > 0.95
    assert abs(agree_xz - (1 - np.arccos(cos_xz) / np.pi)) < 0.1


def test_lsh_params_and_candidates():
    b, r = lsh_params(0.8, 128)
    assert b * r <= 128
    # The S-curve knee sits near the threshold.
    assert 0.6 < (1 / b) ** (1 / r) < 0.95
    sig = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [7, 7, 3, 4], [0, 0, 0, 0]])
    i, j = lsh_candidate_pairs(sig, bands=2, rows=2)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 1), (0, 2)]
    # Blocking forbids pairs across blocks.
    i, j = lsh_candidate_pairs(sig, bands=2, rows=2, block=np.array([0, 1, 0, 0]))
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 2)]
    with pytest.raises(ValueError, match="max_pairs"):
        lsh_candidate_pairs(np.zeros((50, 4), dtype=int), bands=1, rows=4, max_pairs=10)
    with pytest.raises(ValueError):
        lsh_params(0.0, 64)


# --------------------------------------------------------------------------- #
# Oracle: brute-force Jaccard
# --------------------------------------------------------------------------- #
def _planted(n=240, seed=0):
    df = _cells_panel(n=n, seed=seed)
    # (src, dst, cells changed): Jaccard 0.905, 0.818, 1.0, 0.739 (below 0.8,
    # so NOT a near-duplicate at the default threshold), 0.905.
    plan = [(3, 150, 1), (10, 200, 2), (40, 41, 0), (60, 230, 3), (70, 120, 1)]
    for src, dst, k in plan:
        df = _perturb(df, src, dst, k)
    return df


@pytest.mark.parametrize("method", ["minhash", "cminhash"])
def test_verified_pairs_match_brute_force(method):
    df = _planted()
    truth = _brute_pairs(df, 0.8)
    got = near_duplicate_pairs(df, entity="id", time="t", method=method, threshold=0.8)
    pairs = {
        (i, j): s for i, j, s in got.select("row_i", "row_j", "similarity").iter_rows()
    }
    # Precision is exact: every reported pair truly clears the threshold, with
    # its exact Jaccard.
    for key, sim in pairs.items():
        assert key in truth
        assert sim == pytest.approx(truth[key])
    # Recall: every pair at or above the threshold (the four planted ones).
    assert set(truth) == set(pairs)
    assert set(truth) == {(3, 150), (10, 200), (40, 41), (70, 120)}


def test_datasketch_oracle():
    datasketch = pytest.importorskip("datasketch")
    df = _planted()
    sets = _cell_sets(df)
    lsh = datasketch.MinHashLSH(threshold=0.8, num_perm=128)
    for i, s in enumerate(sets):
        m = datasketch.MinHash(num_perm=128, seed=1)
        for tok in s:
            m.update(repr(tok).encode())
        lsh.insert(str(i), m)
    got = near_duplicate_pairs(df, entity="id", time="t", threshold=0.8)
    for i, j in got.select("row_i", "row_j").iter_rows():
        m = datasketch.MinHash(num_perm=128, seed=1)
        for tok in sets[i]:
            m.update(repr(tok).encode())
        assert str(j) in lsh.query(m) or str(i) in lsh.query(m)


def test_estimate_and_none_verification_modes():
    df = _planted()
    truth = _brute_pairs(df, 0.9)
    est = near_duplicate_pairs(
        df, entity="id", time="t", threshold=0.8, verify="estimate"
    )
    none = near_duplicate_pairs(df, entity="id", time="t", threshold=0.8, verify="none")
    # Estimated similarities clear the threshold; unverified candidates still
    # contain every pair comfortably above it.
    assert (est["similarity"] >= 0.8).all()
    found = set(none.select("row_i", "row_j").rows())
    assert set(truth) <= found


# --------------------------------------------------------------------------- #
# The near_duplicate_clusters contract
# --------------------------------------------------------------------------- #
def test_near_duplicate_clusters_contract():
    df = _planted().sample(fraction=1.0, shuffle=True, seed=3)
    out = near_duplicate_clusters(df, entity="id", time="t")
    assert out.height == df.height
    assert out.columns[:4] == ["row_index", "id", "t", "cluster_id"]
    assert out.schema["cluster_id"] == pl.Int64
    assert out.schema["row_index"] == pl.Int64
    assert out.get_column("row_index").to_list() == list(range(df.height))
    # Keys echo the input, in input order.
    assert out.get_column("id").to_list() == df.get_column("id").to_list()
    # Singletons carry their own row index.
    single = out.filter(pl.col("cluster_size") == 1)
    assert (single["cluster_id"] == single["row_index"]).all()
    # A cluster id is the row index of the cluster's earliest member.
    for cid, members in out.filter(pl.col("cluster_size") > 1).group_by("cluster_id"):
        first = members.sort(["t", "row_index"]).row(0, named=True)
        assert first["row_index"] == cid[0]
    assert out.filter(pl.col("cluster_size") > 1)["cluster_id"].n_unique() == 4


def test_near_duplicate_clusters_is_deterministic():
    df = _planted()
    a = near_duplicate_clusters(df, entity="id", time="t", seed=11)
    b = near_duplicate_clusters(df, entity="id", time="t", seed=11)
    assert a.equals(b)


def test_text_tokenizers():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "t": [1, 2, 3, 4],
            "text": [
                "the quick brown fox jumps over the lazy dog",
                "the quick brown fox jumped over the lazy dog",
                "completely unrelated sentence about markets",
                None,
            ],
        }
    )
    char = near_duplicate_clusters(df, entity="id", time="t", threshold=0.6)
    assert char["cluster_id"].to_list() == [0, 0, 2, 3]
    word = near_duplicate_clusters(
        df, entity="id", time="t", tokenizer="word", ngram=2, threshold=0.5
    )
    assert word["cluster_id"].to_list() == [0, 0, 2, 3]


def test_scope_restricts_comparisons():
    df = pl.DataFrame({"id": ["a", "b", "a"], "t": [1, 1, 2], "x": [1.0, 1.0, 1.0]})
    glob = near_duplicate_clusters(df, entity="id", time="t", method="exact")
    ent = near_duplicate_clusters(
        df, entity="id", time="t", method="exact", scope="entity"
    )
    tim = near_duplicate_clusters(
        df, entity="id", time="t", method="exact", scope="time"
    )
    key = near_duplicate_clusters(
        df, entity="id", time="t", method="exact", scope="key"
    )
    assert glob["cluster_id"].to_list() == [0, 0, 0]
    assert ent["cluster_id"].to_list() == [0, 1, 0]
    assert tim["cluster_id"].to_list() == [0, 0, 2]
    assert key["cluster_id"].to_list() == [0, 1, 2]


def test_simhash_finds_numeric_near_duplicates():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 40))
    X[150] = X[5] + 1e-3 * rng.standard_normal(40)
    df = pl.DataFrame({"id": np.arange(200) % 5, "t": np.arange(200)}).hstack(
        pl.DataFrame(X, schema=[f"f{j}" for j in range(40)])
    )
    out = near_duplicate_clusters(
        df, entity="id", time="t", method="simhash", threshold=0.99
    )
    dup = out.filter(pl.col("cluster_size") > 1)
    assert sorted(dup["row_index"].to_list()) == [5, 150]
    with pytest.raises(ValueError, match="numeric"):
        near_duplicate_clusters(
            df.with_columns(pl.col("f0").cast(pl.String)),
            entity="id",
            time="t",
            method="simhash",
        )


# --------------------------------------------------------------------------- #
# Deduplicator
# --------------------------------------------------------------------------- #
def test_deduplicator_declares_contract():
    assert issubclass(Deduplicator, PanelTransformer)
    d = Deduplicator()
    assert d.panel_safe is False and d.leakage_safe is True
    assert Deduplicator(scope="entity").panel_safe is True
    assert Deduplicator(linkage="component").leakage_safe is False
    assert Deduplicator(keep="last").leakage_safe is False


def test_exact_dedup_matches_polars_unique():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "a", "c"],
            "t": [1, 1, 1, 2, 1, 3],
            "x": [1.0, 1.0, 2.0, 2.0, None, None],
            "y": ["u", "u", "v", "v", "w", "w"],
        }
    )
    got = (
        Deduplicator(method="exact", scope="key", entity="id", time="t")
        .fit_transform(df)
        .collect()
    )
    want = df.unique(keep="first", maintain_order=True)
    assert got.sort(got.columns).equals(want.sort(want.columns))
    # Panel-global: (b, 2) repeats (b, 1)'s content one period later, and
    # (c, 3) repeats (a, 1)'s null/"w" content.
    glob = Deduplicator(method="exact", entity="id", time="t").fit_transform(df)
    assert glob.collect().height == 3


def test_deduplicator_drops_later_copies_and_flags():
    df = _planted()
    d = Deduplicator(entity="id", time="t")
    out = d.fit_transform(df).collect()
    assert d.n_duplicates_ == 4
    assert out.height == df.height - 4
    flagged = Deduplicator(entity="id", time="t", action="flag").fit_transform(df)
    flagged = flagged.collect()
    assert flagged.height == df.height
    assert flagged["is_duplicate"].sum() == 4
    # The planted copies are the later rows.
    idx = np.flatnonzero(flagged["is_duplicate"].to_numpy()).tolist()
    assert sorted(idx) == [41, 120, 150, 200]


def test_keep_last_and_component_linkage():
    df = _planted()
    last = Deduplicator(entity="id", time="t", keep="last", action="flag")
    idx = np.flatnonzero(last.fit_transform(df).collect()["is_duplicate"].to_numpy())
    assert sorted(idx.tolist()) == [3, 10, 40, 70]
    comp = Deduplicator(
        entity="id", time="t", linkage="component", action="flag"
    ).fit_transform(df)
    comp = comp.collect()
    assert "cluster_id" in comp.columns
    assert comp["is_duplicate"].sum() == 4


def test_dedup_function_preserves_container_type():
    df = _planted()
    assert isinstance(dedup(df, entity="id", time="t"), pl.DataFrame)
    assert isinstance(dedup(df.lazy(), entity="id", time="t"), pl.LazyFrame)
    pf = PanelFrame(df, entity="id", time="t")
    assert isinstance(dedup(pf), PanelFrame)


def test_deduplicator_validation_errors():
    with pytest.raises(ValueError):
        Deduplicator(method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Deduplicator(threshold=1.5)
    with pytest.raises(ValueError):
        Deduplicator(num_perm=0)
    with pytest.raises(ValueError, match="survivorship"):
        from panelary.clean import Survivorship

        Deduplicator(survivorship=Survivorship())
    df = pl.DataFrame({"id": ["a"], "t": [1], "__panelary_row": [0]})
    with pytest.raises(ValueError, match="reserved"):
        Deduplicator(entity="id", time="t", columns=["id"]).fit_transform(df)
    with pytest.raises(RuntimeError, match="not fitted"):
        Deduplicator().transform(df, entity="id", time="t")


# --------------------------------------------------------------------------- #
# Point-in-time: prefix invariance, and a leaky variant that must fail it
# --------------------------------------------------------------------------- #
def _chain_panel() -> pl.DataFrame:
    """A(t=0) ~ C(t=2) and B(t=1) ~ C, but A !~ B (20 cells, threshold 0.85).

    Jaccard(A, C) = Jaccard(B, C) = 19/21 = 0.905; Jaccard(A, B) = 18/22 = 0.818.
    """
    base = np.arange(20) * 100
    a = base.copy()
    a[0] = -1
    b = base.copy()
    b[1] = -2
    c = base.copy()
    rows = np.vstack([a, b, c])
    df = pl.DataFrame({"id": ["A", "B", "C"], "t": [0, 1, 2]})
    return df.hstack(pl.DataFrame(rows, schema=[f"f{j}" for j in range(20)]))


def _prefix_invariant(make, df: pl.DataFrame, cuts) -> bool:
    full = make().fit_transform(df, entity="id", time="t").collect()
    for cut in cuts:
        prefix = df.filter(pl.col("t") <= cut)
        got = make().fit_transform(prefix, entity="id", time="t").collect()
        want = full.filter(pl.col("t") <= cut)
        if not got.equals(want):
            return False
    return True


def test_pairwise_rule_is_prefix_invariant_component_is_not():
    df = _chain_panel()
    pairs = near_duplicate_pairs(df, entity="id", time="t", threshold=0.85)
    assert pairs.select("row_i", "row_j").rows() == [(0, 2), (1, 2)]

    def honest():
        return Deduplicator(threshold=0.85)

    def leaky():  # connected components read the future
        return Deduplicator(threshold=0.85, linkage="component")

    assert _prefix_invariant(honest, df, cuts=[0, 1])
    assert not _prefix_invariant(leaky, df, cuts=[0, 1])


def test_pairwise_rule_prefix_invariant_on_random_panel():
    df = _planted().sample(fraction=1.0, shuffle=True, seed=0)
    ts = sorted(df["t"].unique().to_list())
    cuts = [ts[len(ts) // 4], ts[len(ts) // 2], ts[3 * len(ts) // 4]]
    assert _prefix_invariant(lambda: Deduplicator(threshold=0.8), df, cuts)


def test_simhash_state_is_fit_on_train_only():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.standard_normal((n, 8)) + np.linspace(0, 5, n)[:, None]
    df = pl.DataFrame({"id": np.arange(n) % 4, "t": np.arange(n) // 4}).hstack(
        pl.DataFrame(X, schema=[f"f{j}" for j in range(8)])
    )
    train, test = df.filter(pl.col("t") < 20), df.filter(pl.col("t") >= 20)
    d = Deduplicator(method="simhash", threshold=0.99, entity="id", time="t").fit(train)
    feats = [f"f{j}" for j in range(8)]
    assert np.allclose(d.center_, train.select(feats).to_numpy().mean(axis=0))
    leaky = Deduplicator(method="simhash", threshold=0.99, entity="id", time="t").fit(
        df
    )
    assert not np.allclose(d.center_, leaky.center_)
    before = (d.center_.copy(), d.scale_.copy())
    d.transform(test)
    assert np.array_equal(d.center_, before[0]) and np.array_equal(d.scale_, before[1])


# --------------------------------------------------------------------------- #
# Split-aware: no near-duplicate pair straddles a fold
# --------------------------------------------------------------------------- #
def _straddle_panel() -> pl.DataFrame:
    df = _cells_panel(n=280, seed=4)
    # Row 20 (t=2) is copied, one cell changed, into row 270 (t=38).
    return _perturb(df, 20, 270, 1)


def test_assert_no_straddle_catches_a_planted_leak():
    df = _straddle_panel()
    train, test = df.filter(pl.col("t") < 30), df.filter(pl.col("t") >= 30)
    bad = straddling_pairs(train, test, entity="id", time="t")
    assert bad.height == 1
    assert bad.row(0)[:2] == (20, 270 - train.height)
    with pytest.raises(AssertionError, match="NEAR-DUPLICATE LEAK"):
        assert_no_straddle(train, test, entity="id", time="t")
    purged = purge_near_duplicates(train, test, entity="id", time="t")
    assert isinstance(purged, pl.DataFrame)
    assert purged.height == train.height - 1
    assert_no_straddle(purged, test, entity="id", time="t")


def test_straddling_clusters_from_the_cluster_table():
    df = _straddle_panel()
    clusters = near_duplicate_clusters(df, entity="id", time="t")
    tr = df.filter(pl.col("t") < 30)["t"].unique()
    te = df.filter(pl.col("t") >= 30)["t"].unique()
    bad = straddling_clusters(clusters, tr, te, time="t")
    assert bad.to_dicts() == [{"cluster_id": 20, "n_train": 1, "n_test": 1}]
    # Row 20 sits at t=2; a train fold that excludes it is clean.
    early = df.filter((pl.col("t") >= 3) & (pl.col("t") < 10))["t"].unique()
    assert straddling_clusters(clusters, early, te, time="t").height == 0


def test_dedup_before_split_leaves_no_straddle():
    from panelary.validation import PurgedKFold

    df = _straddle_panel()
    clean = dedup(df, entity="id", time="t")
    pf = PanelFrame(clean, entity="id", time="t")
    for train, test in PurgedKFold(n_splits=4).split(pf):
        assert_no_straddle(train, test)


def test_split_aware_cv_purges_every_fold():
    from panelary.validation import PurgedKFold

    df = _straddle_panel()
    pf = PanelFrame(df, entity="id", time="t")
    plain = list(PurgedKFold(n_splits=4).split(pf))
    # The unwrapped splitter leaks in at least one fold.
    leaks = sum(straddling_pairs(tr, te).height for tr, te in plain)
    assert leaks >= 1
    cv = SplitAwareCV(PurgedKFold(n_splits=4))
    wrapped = list(cv.split(pf))
    assert len(wrapped) == len(plain) == cv.get_n_splits()
    assert sum(cv.n_purged_) == leaks
    for (tr, te), (_ptr, pte) in zip(wrapped, plain, strict=True):
        assert_no_straddle(tr, te)
        assert te.collect().equals(pte.collect())
        assert "__panelary_row" not in tr.columns


def test_split_aware_cv_accepts_callable_splitters():
    from panelary.validation import expanding_window_split

    df = _straddle_panel()
    pf = PanelFrame(df, entity="id", time="t")
    cv = SplitAwareCV(expanding_window_split(test_size=5, n_splits=2))
    for tr, te in cv.split(pf):
        assert_no_straddle(tr, te)
    with pytest.raises(ValueError, match="return_indices"):
        from panelary.validation import PurgedKFold

        SplitAwareCV(PurgedKFold(return_indices=True))


# --------------------------------------------------------------------------- #
# Reconciliation: preprocessing.reindex(drop_duplicates=True)
# --------------------------------------------------------------------------- #
def test_reindex_drop_duplicates_behaviour_is_unchanged():
    from panelary.preprocessing import reindex

    df = pl.DataFrame(
        {"e": ["a", "a", "b", "b", "a"], "t": [1, 2, 1, 3, 1], "x": [1.0, 2, 3, 4, 5]}
    )
    X = df.lazy()
    ents = X.select(pl.col("e").unique())
    ts = X.select(pl.col("t").unique())
    legacy = ents.join(ts, how="cross").join(X, how="left", on=["e", "t"]).collect()
    got = reindex(drop_duplicates=True)(X).collect()
    key = ["e", "t", "x"]
    assert got.sort(key, nulls_last=True).equals(legacy.sort(key, nulls_last=True))
    assert exact_unique(pl.DataFrame({"a": [1, 1, 2]}), maintain_order=True)[
        "a"
    ].to_list() == [1, 2]
