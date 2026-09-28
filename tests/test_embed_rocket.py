"""Causal rolling MiniRocket-PPV (`panelary.embed._rocket`).

Contracts touched: ``panel_safe`` (entities never mix, unsorted input handled),
``leakage_safe`` (no look-ahead, prefix invariance, biases fitted on the train
rows only), plus determinism and the ``fit_is_empty = False`` claim.

Every leak test is paired with a deliberately leaky construction that must
**fail** it -- the whole-series-pooling reference, and fitting the biases on
the panel being checked -- so a pass means the instrument can see the leak.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from panelary.core.protocol import PanelTransformer
from panelary.embed._rocket import (
    N_KERNELS,
    CausalMiniRocket,
    LeakyMiniRocketReference,
    _dilation_schedule,
    kernel_weights,
)
from panelary.testing import (
    assert_no_lookahead,
    assert_no_train_test_leak,
    assert_prefix_invariant,
)

E, T = "entity", "time"


def _panel(n_entities: int = 4, n_time: int = 90, seed: int = 0) -> pl.DataFrame:
    """Random-walk levels plus returns; entities have different vol and length."""
    rng = np.random.default_rng(seed)
    parts = []
    for e in range(n_entities):
        n = n_time - 7 * e  # unbalanced on purpose
        r = rng.standard_normal(n) * (0.5 + 0.5 * e)
        parts.append(
            pl.DataFrame(
                {
                    E: np.full(n, f"e{e}"),
                    T: np.arange(n, dtype=np.int64) + 3 * e,  # staggered starts
                    "x": r.cumsum(),
                    "r": r,
                }
            )
        )
    return pl.concat(parts)


def _rocket(**kw: object) -> CausalMiniRocket:
    params: dict[str, object] = {
        "n_features": 168,
        "window": 12,
        "max_dilation": 3,
        "entity": E,
        "time": T,
    }
    params.update(kw)
    return CausalMiniRocket(params.pop("columns", "x"), **params)  # type: ignore[arg-type]


def _matrix(frame: pl.DataFrame, col: str = "x_rocket") -> np.ndarray:
    return np.asarray(frame[col].to_list(), dtype=np.float64)


# --------------------------------------------------------------------------- #
# The fixed structure
# --------------------------------------------------------------------------- #
def test_kernel_bank_is_the_84_two_valued_zero_sum_kernels():
    w = kernel_weights()
    assert w.shape == (N_KERNELS, 9) == (84, 9)
    assert set(np.unique(w)) == {-1.0, 2.0}
    assert (np.sum(w == 2.0, axis=1) == 3).all()
    assert np.all(w.sum(axis=1) == 0.0)
    assert len({tuple(row) for row in w}) == 84
    expected = {tuple(c) for c in itertools.combinations(range(9), 3)}
    assert {tuple(np.flatnonzero(row == 2.0)) for row in w} == expected


@pytest.mark.parametrize(
    ("n_per_kernel", "max_dilation", "expected"),
    [
        (1, 7, [(1, 1)]),
        (6, 1, [(1, 6)]),
        (6, 7, [(1, 2), (2, 1), (3, 1), (4, 1), (7, 1)]),
        (6, 4, [(1, 3), (2, 1), (3, 1), (4, 1)]),
    ],
)
def test_dilation_schedule(n_per_kernel, max_dilation, expected):
    sched = _dilation_schedule(n_per_kernel, max_dilation)
    assert sched == expected
    assert sum(c for _, c in sched) == n_per_kernel


def test_defaults_and_class_attributes():
    rocket = CausalMiniRocket("x")
    assert isinstance(rocket, PanelTransformer)
    assert rocket.n_features == 504 == 6 * N_KERNELS
    assert rocket.window == 63 and rocket.min_periods == 63
    assert rocket.max_dilation == 7  # (63 - 1) // 8: the kernel fits the window
    assert CausalMiniRocket.panel_safe is True
    assert CausalMiniRocket.leakage_safe is True
    assert CausalMiniRocket.fit_is_empty is False
    assert CausalMiniRocket.is_cross_sectional is False
    assert LeakyMiniRocketReference.leakage_safe is False
    # Quantile levels are a permutation of a golden-ratio sequence, all in (0, 1).
    q = rocket.quantile_levels_
    assert q.shape == (504,) and ((q > 0) & (q < 1)).all()
    assert len(rocket.feature_names_) == 504


def test_n_features_rounds_down_to_a_multiple_of_84():
    assert CausalMiniRocket("x", n_features=500).n_features == 420
    assert CausalMiniRocket("x", n_features=84).n_features == 84


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 83},
        {"window": 0},
        {"window": 10, "min_periods": 11},
        {"min_periods": 0},
        {"max_dilation": 0},
        {"padding": "centre"},
        {"pooling": ("ppv", "max")},
        {"pooling": ()},
        {"pooling": ("ppv", "ppv")},
        {"max_fit_rows": 0},
        {"output": "wide"},
        {"dtype": "float16"},
        {"columns": []},
        {"columns": ["x", "x"]},
    ],
)
def test_invalid_parameters_raise(kwargs):
    cols = kwargs.pop("columns", "x")
    with pytest.raises(ValueError):
        CausalMiniRocket(cols, **kwargs)


def test_no_center_argument_exists():
    with pytest.raises(TypeError):
        CausalMiniRocket("x", center=True)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Correctness against a brute-force definition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("padding", ["none", "zero"])
@pytest.mark.parametrize("window", [12, None])
def test_matches_bruteforce_convolution_and_pooling(padding, window):
    df = _panel(n_entities=2, n_time=70)
    rocket = _rocket(
        window=window, padding=padding, pooling=("ppv", "mpv"), dtype="float64"
    )
    out = rocket.fit(df).transform(df).collect()
    feats = _matrix(out)
    w = kernel_weights()
    mp = rocket.min_periods
    for ent in ("e0", "e1"):
        rows = np.flatnonzero(out[E].to_numpy() == ent)
        x = out["x"].to_numpy()[rows]
        n = x.size
        for f in (0, 37, 90, 167):
            k, d, b = rocket.kernel_index_[f], rocket.dilation_[f], rocket.biases_[0, f]
            c = np.full(n, np.nan)
            for t in range(n):
                taps = []
                for j in range(9):
                    s = t - (8 - j) * d
                    taps.append(
                        x[s] if s >= 0 else (0.0 if padding == "zero" else np.nan)
                    )
                c[t] = float(np.dot(w[k], taps))
            for t in range(n):
                lo = 0 if window is None else max(0, t - window + 1)
                win = c[lo : t + 1]
                ok = np.isfinite(win)
                if ok.sum() < mp:
                    assert np.isnan(feats[rows[t], f])
                    assert np.isnan(feats[rows[t], f + rocket.n_features])
                    continue
                pos = win[ok] > b
                assert feats[rows[t], f] == pytest.approx(pos.mean(), abs=1e-12)
                mpv = (win[ok][pos] - b).mean() if pos.any() else 0.0
                assert feats[rows[t], f + rocket.n_features] == pytest.approx(
                    mpv, abs=1e-9
                )


def test_padding_none_needs_a_full_receptive_field():
    df = _panel(n_entities=1, n_time=60)
    rocket = _rocket(window=5, max_dilation=2)
    feats = _matrix(rocket.fit(df).transform(df).collect())
    d2 = rocket.dilation_ == 2
    # Dilation 2 is defined from row 16 on; a full 5-row window from row 20.
    assert np.isnan(feats[:20, d2]).all() and np.isfinite(feats[20:, d2]).all()
    d1 = rocket.dilation_ == 1
    assert np.isnan(feats[:12, d1]).all() and np.isfinite(feats[12:, d1]).all()


def test_biases_are_quantiles_of_the_training_responses():
    df = _panel(n_entities=3, n_time=80)
    rocket = _rocket(max_fit_rows=None).fit(df)
    w = kernel_weights()
    srt = df.sort(E, T)
    for f in (0, 50, 120, 167):
        k, d = rocket.kernel_index_[f], rocket.dilation_[f]
        resp = []
        for _, grp in srt.group_by(E, maintain_order=True):
            x = grp["x"].to_numpy()
            for t in range(8 * d, x.size):
                resp.append(float(np.dot(w[k], x[t - 8 * d : t + 1 : d])))
        resp_arr = np.asarray(resp)
        above = float(np.mean(resp_arr > rocket.biases_[0, f]))
        # Linear-interpolated quantile: within one sample of 1 - level.
        assert (
            abs(above - (1.0 - rocket.quantile_levels_[f]))
            <= 1.0 / resp_arr.size + 1e-12
        )
        assert rocket.n_fit_rows_[int(d)] == resp_arr.size


# --------------------------------------------------------------------------- #
# Leak safety: prefix invariance and no look-ahead, with leaky controls
# --------------------------------------------------------------------------- #
_CONFIGS = [
    {},
    {"window": None},
    {"padding": "zero", "columns": "r"},
    {"pooling": ("ppv", "mpv"), "output": "columns"},
    {"columns": ["x", "r"], "dtype": "float64"},
]


def _fixed_state(df: pl.DataFrame, **kw: object) -> CausalMiniRocket:
    """Fit on a fixed early slice -- the state is then data, not a function of df."""
    return _rocket(**kw).fit(df.filter(pl.col(T) <= 40))


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_prefix_invariant(cfg):
    df = _panel()
    rocket = _fixed_state(df, **cfg)
    assert_prefix_invariant(rocket.transform, df, entity=E, time=T, tol=0.0)


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_no_lookahead(cfg):
    df = _panel()
    rocket = _fixed_state(df, **cfg)
    for cut in (30, 50, 70):
        assert_no_lookahead(rocket.transform, df, entity=E, time=T, tol=0.0, cut=cut)


def test_no_train_test_leak_walk_forward():
    df = _panel()
    train = df.filter(pl.col(T) <= 50)
    test = df.filter(pl.col(T) > 50)
    rocket = _rocket().fit(train)
    assert_no_train_test_leak(rocket.transform, df, (train, test), entity=E, time=T)


def test_leaky_pooling_fails_both_instruments():
    df = _panel()
    leaky = LeakyMiniRocketReference.from_fitted(_fixed_state(df))
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky.transform, df, entity=E, time=T)
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky.transform, df, entity=E, time=T)


def test_bias_fitted_on_the_checked_panel_fails_both_instruments():
    """The bias channel: trailing pooling, but biases fitted on everything."""
    df = _panel()

    def fit_on_everything(frame: pl.DataFrame) -> pl.DataFrame:
        return _rocket(max_fit_rows=None).fit(frame).transform(frame).collect()

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(fit_on_everything, df, entity=E, time=T)
    # One late cut: an early truncation is too short to fit at all.
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(fit_on_everything, df, entity=E, time=T, cut=60)


def test_bias_fitted_inside_the_protected_region_passes():
    """Same pipeline, biases fit on ``time <= cut`` only: clean."""
    df = _panel()
    cut = 45

    def fit_on_past(frame: pl.DataFrame) -> pl.DataFrame:
        past = frame.filter(pl.col(T) <= cut)
        return _rocket(max_fit_rows=None).fit(past).transform(frame).collect()

    assert_no_lookahead(fit_on_past, df, entity=E, time=T, cut=cut, tol=0.0)


# --------------------------------------------------------------------------- #
# Fitted state: honest `fit_is_empty = False`, determinism
# --------------------------------------------------------------------------- #
def test_disjoint_fits_give_different_state():
    df = _panel(n_time=120)
    first = _rocket().fit(df.filter(pl.col(T) < 60))
    second = _rocket().fit(df.filter(pl.col(T) >= 60))
    assert first.biases_.shape == second.biases_.shape == (1, 168)
    assert not np.array_equal(first.biases_, second.biases_)
    # ... and therefore different features on the same rows.
    a = _matrix(first.transform(df).collect())
    b = _matrix(second.transform(df).collect())
    assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_deterministic_given_seed():
    df = _panel()
    a = _rocket(seed=7, max_fit_rows=100).fit(df)
    b = _rocket(seed=7, max_fit_rows=100).fit(df)
    assert np.array_equal(a.biases_, b.biases_)
    assert np.array_equal(a.quantile_levels_, b.quantile_levels_)
    fa = _matrix(a.transform(df).collect())
    fb = _matrix(b.transform(df).collect())
    assert np.array_equal(fa, fb, equal_nan=True)
    c = _rocket(seed=8, max_fit_rows=100).fit(df)
    assert not np.array_equal(a.quantile_levels_, c.quantile_levels_)
    assert not np.array_equal(a.biases_, c.biases_)


def test_levels_are_permuted_not_in_sequence_order():
    q = CausalMiniRocket("x", n_features=504, seed=0).quantile_levels_
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    # In sequence order every consecutive gap would be the golden step (mod 1).
    assert not np.allclose(np.mod(np.diff(q), 1.0), golden)
    # ... but the set of levels is still a golden-ratio sequence: by the
    # three-gap theorem its sorted gaps take at most three distinct lengths.
    gaps = np.diff(np.sort(q))
    assert len(np.unique(np.round(gaps, 9))) <= 3
    assert gaps.max() < 3.0 / q.size


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        _rocket().transform(_panel())


def test_fit_on_too_short_series_raises():
    df = _panel(n_entities=1, n_time=10)
    with pytest.raises(ValueError, match="no valid convolution output"):
        _rocket(max_dilation=3).fit(df)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="not in panel"):
        _rocket(columns="nope").fit(_panel())


# --------------------------------------------------------------------------- #
# panel_safe: within-entity, order-independent
# --------------------------------------------------------------------------- #
def test_unsorted_input_gives_the_same_features_in_input_order():
    df = _panel()
    rocket = _rocket(pooling=("ppv", "mpv")).fit(df)
    ref = rocket.transform(df).collect()
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=3)
    out = rocket.transform(shuffled).collect()
    # Row order is the caller's, untouched.
    assert out.select(E, T).equals(shuffled.select(E, T))
    joined = out.join(ref, on=[E, T], suffix="_ref")
    assert np.array_equal(
        _matrix(joined), _matrix(joined, "x_rocket_ref"), equal_nan=True
    )
    # Fitting on shuffled rows gives the same biases, too.
    assert np.array_equal(
        _rocket(pooling=("ppv", "mpv")).fit(shuffled).biases_, rocket.biases_
    )


def test_entities_never_mix():
    df = _panel()
    rocket = _rocket().fit(df)
    full = rocket.transform(df).collect()
    # One entity alone gives exactly its rows of the joint run.
    alone = rocket.transform(df.filter(pl.col(E) == "e2")).collect()
    joint = full.filter(pl.col(E) == "e2")
    assert np.array_equal(_matrix(alone), _matrix(joint), equal_nan=True)
    # Corrupting another entity changes nothing for this one.
    noisy = df.with_columns(
        pl.when(pl.col(E) == "e1").then(pl.col("x") * 1e6).otherwise(pl.col("x"))
    )
    other = rocket.transform(noisy).collect().filter(pl.col(E) == "e2")
    assert np.array_equal(_matrix(other), _matrix(joint), equal_nan=True)


def test_nulls_and_float32_input():
    df = _panel(n_entities=2).with_columns(pl.col("x").cast(pl.Float32))
    df = df.with_columns(
        pl.when(pl.col(T) == 40).then(None).otherwise(pl.col("x")).alias("x")
    )
    rocket = _rocket(dtype="float64").fit(df)
    feats = _matrix(rocket.transform(df).collect())
    assert feats.dtype == np.float64 and np.isfinite(feats).any()
    # A null at t=40 invalidates the responses whose taps touch it, so with a
    # full-window requirement the features just after it are NaN, not garbage.
    rows = np.flatnonzero((df[E].to_numpy() == "e0") & (df[T].to_numpy() == 41))
    assert np.isnan(feats[rows[0]]).all()


def test_output_layouts():
    df = _panel(n_entities=2)
    arr = _rocket(columns=["x", "r"], pooling=("ppv", "mpv")).fit(df)
    out = arr.transform(df).collect()
    assert out["x_rocket"].dtype == pl.Array(pl.Float32, 336)
    assert out["r_rocket"].dtype == pl.Array(pl.Float32, 336)
    wide = _rocket(output="columns", suffix="_mr").fit(df).transform(df).collect()
    names = [c for c in wide.columns if c.startswith("x_mr_")]
    assert len(names) == 168 and names[0] == "x_mr_ppv_d1_k00_0"
    assert all(wide[c].dtype == pl.Float32 for c in names)
    empty = arr.transform(df.head(0)).collect()
    assert empty.height == 0 and empty["x_rocket"].dtype == pl.Array(pl.Float32, 336)
