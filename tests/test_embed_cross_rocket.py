"""Tests for CrossROCKET (``panelary.embed._cross_rocket.CrossRocket``).

Contract under test (AGENTS.md, ``plans/todo/embed-build-contract.md`` §2):

* ``leakage_safe`` -- causal at each date: ``assert_no_lookahead`` and
  ``assert_prefix_invariant`` pass, and a deliberately leaky peer basket built on
  *full-sample* correlations fails both.
* ``fit_is_empty`` -- fitting on disjoint data gives byte-identical state.
* Permutation equivariance of per-entity outputs and invariance of pooled ones,
  with a label-dependent variant that must fail the same check.
* Determinism (seeded RNG, invariant 2).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary  # noqa: F401  (registers the .xs namespace)
from panelary.core.panel_frame import PanelFrame
from panelary.embed._cross_rocket import FAMILIES, CrossRocket
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

E, T = "ticker", "date"
COLS = ["a", "b", "c", "ret"]


def make_panel(
    n_ent: int = 24, n_t: int = 30, seed: int = 0, nulls: bool = False
) -> pl.DataFrame:
    """A small balanced panel with a clustered return so peers are meaningful."""
    rng = np.random.default_rng(seed)
    groups = np.arange(n_ent) % 3
    common = rng.standard_normal((n_t, 3))
    ret = common[:, groups] + 0.7 * rng.standard_normal((n_t, n_ent))
    df = pl.DataFrame(
        {
            E: np.repeat([f"s{i:02d}" for i in range(n_ent)], n_t),
            T: np.tile(np.arange(n_t), n_ent),
            "a": rng.standard_normal(n_ent * n_t),
            "b": rng.standard_t(3, n_ent * n_t),
            "c": rng.random(n_ent * n_t).round(1),  # ties on purpose
            "ret": ret.T.reshape(-1),
        }
    )
    if nulls:
        mask = rng.random(df.height) < 0.05
        df = df.with_columns(
            pl.when(pl.Series(mask)).then(None).otherwise(pl.col("b")).alias("b"),
            pl.when(pl.Series(np.roll(mask, 7)))
            .then(None)
            .otherwise(pl.col("ret"))
            .alias("ret"),
        )
    return df


def rocket(**kw: object) -> CrossRocket:
    params: dict[str, object] = {
        "n_operators": 32,
        "peer_col": "ret",
        "peer_windows": (5, 8),
        "peer_sizes": (3, 6),
        "entity": E,
        "time": T,
        "seed": 3,
    }
    params.update(kw)
    return CrossRocket(**params)  # type: ignore[arg-type]


def run(xr: CrossRocket, df: pl.DataFrame) -> pl.DataFrame:
    return xr.fit_transform(df).collect()


# --------------------------------------------------------------------------- #
# Declarations and shape
# --------------------------------------------------------------------------- #
def test_class_attributes() -> None:
    assert CrossRocket.panel_safe is False
    assert CrossRocket.leakage_safe is True
    assert CrossRocket.fit_is_empty is True
    assert CrossRocket.is_cross_sectional is True


def test_output_shape_names_and_row_order() -> None:
    df = make_panel()
    xr = rocket()
    out = run(xr, df)
    assert out.columns == [E, T, *xr.output_names_]
    assert len(xr.output_names_) == 32
    assert out.select(E, T).equals(df.select(E, T))  # input row order kept
    assert xr.bank_.height == 32
    assert set(xr.bank_["family"]) == set(FAMILIES)
    # Operators are split evenly over the four families.
    assert xr.bank_.group_by("family").len()["len"].to_list() == [8, 8, 8, 8]


def test_default_families_depend_on_peer_col() -> None:
    assert CrossRocket().families == FAMILIES[:3]
    assert CrossRocket(peer_col="ret").families == FAMILIES


@pytest.mark.parametrize(
    ("output", "n_cols"),
    [("entity", 32), ("pooled", 24), ("both", 56)],
)
def test_output_modes(output: str, n_cols: int) -> None:
    df = make_panel()
    xr = rocket(output=output)
    out = run(xr, df)
    assert len(xr.output_names_) == n_cols  # rank ops are not pooled: 32 - 8
    assert out.width == 2 + n_cols


def test_pooled_values_are_per_date_constants_in_unit_interval() -> None:
    df = make_panel()
    xr = rocket(output="pooled", pool_stats=("ppv",))
    out = run(xr, df)
    per_date = out.group_by(T).agg(pl.col(xr.output_names_).n_unique())
    assert (per_date.select(xr.output_names_).to_numpy() == 1).all()
    vals = out.select(xr.output_names_).to_numpy()
    finite = vals[~np.isnan(vals)]
    assert finite.size and finite.min() >= 0.0 and finite.max() <= 1.0


def test_keep_features_array_and_float32() -> None:
    df = make_panel()
    out = run(rocket(keep_features=True, dtype="float32"), df)
    assert out.columns[: df.width] == df.columns
    assert out.schema["xr_med_000"] == pl.Float32
    arr = run(rocket(as_array=True), df)
    assert arr.columns == [E, T, "xr"]
    assert arr.schema["xr"] == pl.Array(pl.Float64, 32)


def test_accepts_panelframe() -> None:
    df = make_panel()
    pf = PanelFrame(df, entity=E, time=T)
    kw = {"n_operators": 8, "peer_col": "ret", "peer_windows": (5,), "seed": 0}
    a = CrossRocket(**kw).fit_transform(pf)  # type: ignore[arg-type]
    b = CrossRocket(**kw, entity=E, time=T).fit_transform(df)  # type: ignore[arg-type]
    assert a.collect().equals(b.collect())


# --------------------------------------------------------------------------- #
# Equivariance / invariance
# --------------------------------------------------------------------------- #
def _relabelled(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, dict[str, str]]:
    """Permute entity labels and shuffle row order; return the label mapping."""
    rng = np.random.default_rng(seed)
    labels = sorted(df[E].unique().to_list())
    new = [labels[i] for i in rng.permutation(len(labels))]
    mapping = dict(zip(labels, new))
    perm = df.with_columns(pl.col(E).replace_strict(mapping)).sample(
        fraction=1.0, shuffle=True, seed=seed
    )
    return perm, mapping


def assert_equivariant(xr_factory, df: pl.DataFrame, seed: int = 11) -> None:
    base = run(xr_factory(), df)
    perm_df, mapping = _relabelled(df, seed)
    perm = run(xr_factory(), perm_df)
    # Map the permuted run's labels back and align on (entity, time).
    back = {v: k for k, v in mapping.items()}
    perm = perm.with_columns(pl.col(E).replace_strict(back)).sort(E, T)
    base = base.sort(E, T)
    feats = [c for c in base.columns if c not in (E, T)]
    a = base.select(feats).to_numpy()
    b = perm.select(feats).to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), "missingness pattern moved"
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("activation", ["identity", "hinge", "indicator"])
def test_entity_outputs_are_permutation_equivariant(activation: str) -> None:
    df = make_panel(nulls=True)
    assert_equivariant(lambda: rocket(activation=activation), df)


def test_pooled_outputs_are_permutation_invariant() -> None:
    df = make_panel(nulls=True)
    xr = rocket(output="pooled", pool_stats=("ppv", "mpv"))
    base = run(xr, df).group_by(T).agg(pl.col(xr.output_names_).first()).sort(T)
    perm_df, _ = _relabelled(df, 5)
    perm = run(rocket(output="pooled", pool_stats=("ppv", "mpv")), perm_df)
    perm = perm.group_by(T).agg(pl.col(xr.output_names_).first()).sort(T)
    np.testing.assert_allclose(
        base.select(xr.output_names_).to_numpy(),
        perm.select(xr.output_names_).to_numpy(),
        rtol=0,
        atol=1e-12,
    )


class _LabelPeers(CrossRocket):
    """Non-equivariant control: peers are the next entities *by label order*."""

    def _peer_table(self, P, ents, t, window, m_max, candidate):  # type: ignore[override]
        n = ents.size
        m = min(m_max, n - 1)
        idx = (np.arange(n)[:, None] + 1 + np.arange(m)[None, :]) % n
        return idx.astype(np.int64), np.ones((n, m), dtype=bool)


def test_equivariance_check_has_teeth() -> None:
    df = make_panel()
    with pytest.raises(AssertionError):
        assert_equivariant(
            lambda: _LabelPeers(
                n_operators=8,
                families=("peer_dev",),
                peer_col="ret",
                peer_windows=(5,),
                entity=E,
                time=T,
            ),
            df,
        )


# --------------------------------------------------------------------------- #
# Leakage: causal, prefix-invariant, and a leaky variant that must fail
# --------------------------------------------------------------------------- #
def _op(cls: type[CrossRocket] = CrossRocket, **kw: object):
    def op(frame: pl.DataFrame) -> pl.DataFrame:
        params: dict[str, object] = {
            "n_operators": 32,
            "peer_col": "ret",
            "peer_windows": (5, 8),
            "peer_sizes": (3, 6),
            "entity": E,
            "time": T,
            "seed": 3,
        }
        params.update(kw)
        return cls(**params).fit_transform(frame).collect()  # type: ignore[arg-type]

    return op


@pytest.mark.parametrize("output", ["entity", "both"])
@pytest.mark.parametrize("nulls", [False, True])
def test_no_lookahead(output: str, nulls: bool) -> None:
    assert_no_lookahead(_op(output=output), make_panel(nulls=nulls), entity=E, time=T)


@pytest.mark.parametrize("output", ["entity", "both"])
@pytest.mark.parametrize("nulls", [False, True])
def test_prefix_invariant(output: str, nulls: bool) -> None:
    assert_prefix_invariant(
        _op(output=output), make_panel(nulls=nulls), entity=E, time=T
    )


class _FullSamplePeers(CrossRocket):
    """Deliberately leaky: peers from correlations over the *whole* sample."""

    def _peer_history(self, P, t, window):  # type: ignore[override]
        return P


def test_full_sample_peer_correlations_fail_lookahead() -> None:
    df = make_panel()
    op = _op(_FullSamplePeers, families=("peer_dev",))
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK"):
        assert_no_lookahead(op, df, entity=E, time=T)


def test_full_sample_peer_correlations_fail_prefix_invariance() -> None:
    df = make_panel()
    op = _op(_FullSamplePeers, families=("peer_dev",))
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(op, df, entity=E, time=T)


def test_peers_use_only_strictly_earlier_dates() -> None:
    """Changing ``peer_col`` *at* date t must not move peer outputs at t."""
    df = make_panel()
    xr_kw = {"columns": ["a", "b", "c"], "families": ("peer_dev",)}
    base = run(rocket(**xr_kw), df)
    t0 = 15
    bumped = df.with_columns(
        pl.when(pl.col(T) == t0)
        .then(pl.col("ret") * -50.0 + 3.0)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    out = run(rocket(**xr_kw), bumped)
    feats = [c for c in base.columns if c not in (E, T)]
    same = base.filter(pl.col(T) <= t0).select(feats).to_numpy()
    new = out.filter(pl.col(T) <= t0).select(feats).to_numpy()
    np.testing.assert_array_equal(same, new)
    later_a = base.filter(pl.col(T) == t0 + 1).select(feats).to_numpy()
    later_b = out.filter(pl.col(T) == t0 + 1).select(feats).to_numpy()
    assert not np.allclose(later_a, later_b, equal_nan=True)


def test_peer_outputs_null_before_min_periods() -> None:
    df = make_panel()
    xr = rocket(families=("peer_dev",), peer_windows=(8,), peer_min_periods=6)
    out = run(xr, df)
    early = out.filter(pl.col(T) < 6).select(xr.output_names_)
    assert early.null_count().sum_horizontal().item() == early.height * early.width
    late = out.filter(pl.col(T) >= 6).select(xr.output_names_)
    assert late.null_count().sum_horizontal().item() == 0


def test_non_peer_families_depend_only_on_their_own_date() -> None:
    df = make_panel()
    xr_kw = {"families": ("median_dev", "rank_threshold", "subset_agg")}
    one_date = df.filter(pl.col(T) == 12)
    alone = run(rocket(**xr_kw), one_date).sort(E)
    within = run(rocket(**xr_kw), df).filter(pl.col(T) == 12).sort(E)
    assert alone.equals(within)


# --------------------------------------------------------------------------- #
# Stateless fit and determinism
# --------------------------------------------------------------------------- #
def test_fit_is_empty_state_identical_on_disjoint_data() -> None:
    a = make_panel(seed=1)
    b = (
        make_panel(n_ent=9, n_t=13, seed=99)
        .with_columns((pl.col(E) + "_other"), pl.col(T) + 1000)
        .with_columns(pl.col(["a", "b", "c", "ret"]) * 1e3)
    )
    sa = rocket().fit(a).state()
    sb = rocket().fit(b).state()
    assert sa.keys() == sb.keys()
    for key in sa:
        va, vb = sa[key], sb[key]
        if isinstance(va, np.ndarray):
            assert va.dtype == vb.dtype and va.tobytes() == vb.tobytes(), key
        else:
            assert va == vb, key


def test_deterministic_and_seed_sensitive() -> None:
    df = make_panel(nulls=True)
    assert run(rocket(seed=7), df).equals(run(rocket(seed=7), df))
    assert not run(rocket(seed=7), df).equals(run(rocket(seed=8), df))


def test_family_draws_do_not_depend_on_other_families() -> None:
    both = rocket(n_operators=16, families=("median_dev", "peer_dev")).fit(make_panel())
    alone = rocket(n_operators=8, families=("median_dev",)).fit(make_panel())
    med = both.bank_.filter(pl.col("family") == "median_dev")
    assert med.equals(alone.bank_)


# --------------------------------------------------------------------------- #
# The operators are compositions of the .xs vocabulary
# --------------------------------------------------------------------------- #
def test_single_channel_rank_operator_matches_xs_rank() -> None:
    df = make_panel()
    xr = rocket(
        families=("rank_threshold",), max_channels=1, activation="identity"
    ).fit(df)
    out = xr.transform(df).collect()
    for row in xr.bank_.iter_rows(named=True):
        col, sign, q = row["channels"][0], np.sign(row["weights"][0]), row["threshold"]
        ref = df.select(
            E,
            T,
            ((pl.col(col) * sign).xs.rank(normalize=True).over(T) - q).alias("ref"),
        )
        got = out.join(ref, on=[E, T])
        np.testing.assert_allclose(got[row["name"]], got["ref"], atol=1e-12)


def test_single_channel_median_operator_matches_robust_z() -> None:
    df = make_panel()
    xr = rocket(
        families=("median_dev",), max_channels=1, activation="identity", clip=5.0
    ).fit(df)
    out = xr.transform(df).collect()
    for row in xr.bank_.iter_rows(named=True):
        col, sign, b = row["channels"][0], np.sign(row["weights"][0]), row["threshold"]
        x = pl.col(col)
        dev = x - x.median().over(T)
        z = (dev / (1.4826 * dev.abs().median().over(T))).clip(-5.0, 5.0)
        ref = df.select(E, T, (z * sign - b).alias("ref"))
        got = out.join(ref, on=[E, T])
        np.testing.assert_allclose(got[row["name"]], got["ref"], atol=1e-9)


# --------------------------------------------------------------------------- #
# Missing data and validation
# --------------------------------------------------------------------------- #
def test_missing_inputs_give_null_only_where_used() -> None:
    df = make_panel(nulls=True)
    xr = rocket(families=("median_dev",), n_operators=16)
    raw = df.select(E, T, *(pl.col(c).alias(f"raw_{c}") for c in COLS))
    out = run(xr, df).join(raw, on=[E, T])
    assert out[xr.output_names_].null_count().sum_horizontal().item() > 0
    for row in xr.bank_.iter_rows(named=True):
        expect = pl.any_horizontal(
            pl.col(f"raw_{c}").is_null() for c in row["channels"]
        )
        assert (out[row["name"]].is_null() == out.select(expect).to_series()).all()


def test_duplicate_keys_raise() -> None:
    df = make_panel()
    with pytest.raises(ValueError, match="duplicate"):
        run(rocket(), pl.concat([df, df.head(3)]))


@pytest.mark.parametrize(
    ("kw", "exc"),
    [
        ({"n_operators": 0}, ValueError),
        ({"families": ("bogus",)}, ValueError),
        ({"families": ("peer_dev",), "peer_col": None}, ValueError),
        ({"activation": "relu"}, ValueError),
        ({"output": "mean"}, ValueError),
        ({"pool_stats": ("max",)}, ValueError),
        ({"peer_windows": (1,)}, ValueError),
        ({"peer_windows": ()}, ValueError),
        ({"peer_min_periods": 50, "peer_windows": (5, 8)}, ValueError),
        ({"clip": 0.0}, ValueError),
        ({"seed": -1}, ValueError),
        ({"n_operators": 2.5}, TypeError),
    ],
)
def test_invalid_parameters(kw: dict[str, object], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        rocket(**kw)


def test_pooled_rank_only_raises() -> None:
    with pytest.raises(ValueError, match="rank_threshold"):
        rocket(families=("rank_threshold",), output="pooled").fit(make_panel())


def test_missing_peer_col_in_panel_raises() -> None:
    with pytest.raises(ValueError, match="peer_col"):
        rocket(peer_col="nope").fit(make_panel())
