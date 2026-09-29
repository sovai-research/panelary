"""Numerical correctness of the BSADF kernel and its critical values.

The leak tests in ``test_detect_leak_safety.py`` prove the kernel never reads
the future. These prove it computes the *right* number while doing so:

* **Invariant 6** -- the O(1) prefix-sum window ADF matches a per-window OLS
  fit written here from scratch with ``np.linalg.lstsq``, to < 1e-11, at lags
  0-3. Two independent derivations of the same t-statistic.
* **The NumPy 2.0 solve trap** -- ``np.linalg.solve(A, b)`` with
  ``b.shape == (K, p)`` silently solves a *different* problem when ``p == K``.
  The observable symptom is that a batch of windows gets a different answer
  depending on how many windows were in the batch.
* **Invariant 7** -- Monte Carlo GSADF critical values reproduce the PSY (2015)
  finite-sample table; the two monotonicities (cv rises as the minimum window
  fraction falls; GSADF >= SADF >= ADF) double as assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import test_detect_support as S

LAGS = (0, 1, 2, 3)

#: Short enough that the O(T^3) naive reference stays in the fast suite.
N_REF = 80
MIN_W_REF = 25


# --------------------------------------------------------------------------- #
# 6. Vectorised versus naive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lag", LAGS)
def test_bsadf_matches_per_window_ols_reference(lag: int) -> None:
    """The exhaustive sup equals a brute-force ``lstsq`` sup, to < 1e-11."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    y = S.bubble_series(N_REF, seed=21)

    got = np.asarray(fn(y, min_window=MIN_W_REF, lag=lag, grid=None), dtype=np.float64)
    ref_c = S.naive_bsadf(y, min_window=MIN_W_REF, lag=lag, trend="c")

    assert got.shape == y.shape, f"bsadf_sequence returned {got.shape}, want {y.shape}"

    # Compare well clear of any +/-1 disagreement about where the first window
    # closes, so the test measures the arithmetic and not an index convention.
    lo = MIN_W_REF + lag + 2
    mask = np.isfinite(got[lo:]) & np.isfinite(ref_c[lo:])
    assert mask.sum() >= 20, "too few comparable endpoints; check the NaN prefix"

    dev = float(np.max(np.abs(got[lo:][mask] - ref_c[lo:][mask])))
    if dev >= 1e-11:
        ref_ct = S.naive_bsadf(y, min_window=MIN_W_REF, lag=lag, trend="ct")
        m2 = np.isfinite(got[lo:]) & np.isfinite(ref_ct[lo:])
        dev_ct = (
            float(np.max(np.abs(got[lo:][m2] - ref_ct[lo:][m2])))
            if m2.any()
            else np.inf
        )
        pytest.fail(
            f"lag={lag}: BSADF differs from the intercept-only per-window OLS "
            f"reference by {dev:.3g} (want < 1e-11). Deviation against the "
            f"constant+trend specification is {dev_ct:.3g} -- if that one is "
            "small the kernel is fitting the wrong deterministic terms; PSY "
            "GSADF is intercept-only."
        )


def test_bsadf_nan_prefix_is_a_frozen_constant() -> None:
    """The NaN prefix is set by ``min_window``, never by ``len(y)``."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    firsts = []
    for n in (90, 140, 200):
        out = np.asarray(
            fn(S.random_walk(n, seed=2), min_window=MIN_W_REF, lag=0, grid=None),
            dtype=np.float64,
        )
        finite = np.flatnonzero(np.isfinite(out))
        assert finite.size, f"no finite values at n={n}"
        firsts.append(int(finite[0]))
    assert len(set(firsts)) == 1, (
        f"the first finite endpoint moved with the sample size: {firsts}. "
        "`min_window` (or the warm-up) is a function of len(y)."
    )
    assert firsts[0] <= MIN_W_REF + 2, (
        f"first finite endpoint is {firsts[0]} for min_window={MIN_W_REF}"
    )


@pytest.mark.parametrize("lag", LAGS)
def test_window_adf_matches_per_window_ols_reference(lag: int) -> None:
    """``window_adf`` over an explicit (start, end) batch, against ``lstsq``."""
    cm = S.require_attr(S.moments, "cumulative_moments", "_moments")
    wadf = S.require_attr(S.moments, "window_adf", "_moments")

    y = S.bubble_series(160, seed=22)
    m = cm(y, lag=lag)

    rng = np.random.default_rng(31)
    ends = rng.integers(70, 160, size=60)
    starts = np.array([rng.integers(0, e - 45) for e in ends])

    got = np.asarray(wadf(m, starts, ends), dtype=np.float64)
    assert got.shape == (starts.size,)

    # Fix the (start, end) convention from a single probe, then hold the whole
    # batch to it. Inclusive means the window is y[s : e+1].
    probe_inc = S.naive_adf(y[starts[0] : ends[0] + 1], lag)
    probe_exc = S.naive_adf(y[starts[0] : ends[0]], lag)
    err_inc = abs(got[0] - probe_inc)
    err_exc = abs(got[0] - probe_exc)
    if min(err_inc, err_exc) > 1e-6:
        pytest.fail(
            f"lag={lag}: window_adf({starts[0]}, {ends[0]}) = {got[0]:.8f} matches "
            f"neither y[s:e+1] ({probe_inc:.8f}, err {err_inc:.3g}) nor y[s:e] "
            f"({probe_exc:.8f}, err {err_exc:.3g})."
        )
    inclusive = err_inc <= err_exc
    ref = np.array(
        [
            S.naive_adf(y[s : e + 1] if inclusive else y[s:e], lag)
            for s, e in zip(starts, ends)
        ]
    )
    dev = float(np.max(np.abs(got - ref)))
    assert dev < 1e-11, (
        f"lag={lag}: window_adf differs from per-window lstsq by {dev:.3g} over "
        f"{starts.size} windows (end convention: "
        f"{'inclusive' if inclusive else 'exclusive'})."
    )


def test_cumulative_moments_block_reset_does_not_change_the_answer() -> None:
    """The accumulator's block-reset period is a conditioning knob, not a result."""
    cm = S.require_attr(S.moments, "cumulative_moments", "_moments")
    wadf = S.require_attr(S.moments, "window_adf", "_moments")
    y = S.bubble_series(300, seed=23) + 1e5

    starts = np.arange(0, 200, 7)
    ends = starts + 90
    a = np.asarray(wadf(cm(y, lag=1, block=7), starts, ends), dtype=np.float64)
    b = np.asarray(wadf(cm(y, lag=1, block=1_000_000), starts, ends), dtype=np.float64)
    dev = S.max_abs_dev(a, b)
    assert dev < 1e-10, (
        f"block=7 and block=1e6 disagree by {dev:.3g}; the block reset must be "
        "numerically transparent."
    )


# --------------------------------------------------------------------------- #
# The NumPy 2.0 `solve` trap: p == K
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lag", [0, 1, 2])
def test_window_adf_is_independent_of_batch_size(lag: int) -> None:
    """A window's answer must not depend on how many windows shared its call.

    ``np.linalg.solve(A, b)`` with ``b.shape == (K, p)`` is interpreted as a
    *stack* of problems when ``p != K`` and as a single matrix solve when
    ``p == K``, silently returning the wrong coefficients in exactly one case:
    batch size equal to the number of regressors (``2 + lag`` here). This test
    walks the batch size straight through that point.
    """
    cm = S.require_attr(S.moments, "cumulative_moments", "_moments")
    wadf = S.require_attr(S.moments, "window_adf", "_moments")

    y = S.bubble_series(200, seed=24)
    m = cm(y, lag=lag)
    n_reg = 2 + lag  # intercept, y[t-1], and `lag` differenced terms

    all_starts = np.arange(0, 8) * 5
    all_ends = all_starts + 110
    ref = S.naive_adf(y[all_starts[0] : all_ends[0] + 1], lag)
    ref_exc = S.naive_adf(y[all_starts[0] : all_ends[0]], lag)

    values = {}
    for k in range(1, 9):
        out = np.asarray(wadf(m, all_starts[:k], all_ends[:k]), dtype=np.float64)
        assert out.shape == (k,)
        values[k] = float(out[0])

    spread = max(values.values()) - min(values.values())
    assert spread < 1e-11, (
        f"lag={lag} (K={n_reg} regressors): the first window's ADF t-stat "
        f"changed by {spread:.3g} as the batch grew from 1 to 8 windows: "
        f"{values}. This is the NumPy 2.0 `solve` trap at batch size == K == "
        f"{n_reg}; use `np.linalg.solve(A, b[..., None])[..., 0]`."
    )
    err = min(abs(values[n_reg] - ref), abs(values[n_reg] - ref_exc))
    assert err < 1e-11, (
        f"lag={lag}: with a batch of exactly K={n_reg} windows the kernel "
        f"returned {values[n_reg]:.8f}; per-window lstsq gives {ref:.8f}."
    )


# --------------------------------------------------------------------------- #
# Grid versus exhaustive
# --------------------------------------------------------------------------- #
def test_geometric_grid_tracks_the_exhaustive_sup() -> None:
    """32 window lengths recover the sup to corr > 0.99 with no overshoot."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    y = S.bubble_series(400, seed=25)

    ex = np.asarray(fn(y, min_window=S.MIN_W, lag=0, grid=None), dtype=np.float64)
    gr = np.asarray(fn(y, min_window=S.MIN_W, lag=0, grid=32), dtype=np.float64)

    mask = np.isfinite(ex) & np.isfinite(gr)
    assert mask.sum() > 200

    shortfall = ex[mask] - gr[mask]
    assert shortfall.min() > -1e-10, (
        f"the grid sup exceeded the exhaustive sup by {-shortfall.min():.3g}; a "
        "sup over a subset of window lengths can never be larger."
    )
    corr = float(np.corrcoef(ex[mask], gr[mask])[0, 1])
    assert corr > 0.99, f"grid vs exhaustive correlation is only {corr:.4f}"

    again = np.asarray(fn(y, min_window=S.MIN_W, lag=0, grid=32), dtype=np.float64)
    assert S.bit_identical(gr, again), "the geometric ladder is not deterministic"


def test_refinement_closes_the_grid_gap_without_overshooting() -> None:
    """``refine_top_k`` may only move the grid statistic up, and never past the sup."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    y = S.bubble_series(300, seed=26)
    ex = np.asarray(fn(y, min_window=S.MIN_W, lag=0, grid=None), dtype=np.float64)
    gr = np.asarray(fn(y, min_window=S.MIN_W, lag=0, grid=32), dtype=np.float64)
    rf = np.asarray(
        fn(y, min_window=S.MIN_W, lag=0, grid=32, refine_top_k=4), dtype=np.float64
    )
    mask = np.isfinite(ex) & np.isfinite(gr) & np.isfinite(rf)
    assert (rf[mask] >= gr[mask] - 1e-10).all(), "refinement lowered the statistic"
    assert (rf[mask] <= ex[mask] + 1e-10).all(), "refinement exceeded the true sup"


def test_shrinking_the_minimum_window_can_only_raise_the_statistic() -> None:
    """Pathwise monotonicity: a sup over a superset of windows is never smaller."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    y = S.bubble_series(300, seed=27)
    wide = np.asarray(fn(y, min_window=60, lag=0, grid=None), dtype=np.float64)
    narrow = np.asarray(fn(y, min_window=20, lag=0, grid=None), dtype=np.float64)
    mask = np.isfinite(wide) & np.isfinite(narrow)
    assert mask.sum() > 100
    assert (narrow[mask] >= wide[mask] - 1e-10).all(), (
        "min_window=20 gave a smaller statistic than min_window=60 somewhere; "
        "the smaller minimum searches a strict superset of start points."
    )


# --------------------------------------------------------------------------- #
# `_critvals`: the Monte Carlo table
# --------------------------------------------------------------------------- #
LEVELS = (0.90, 0.95, 0.99)


def test_mc_table_shape_levels_and_determinism() -> None:
    mc = S.require_attr(S.critvals, "mc_table", "_critvals")
    t_max = 120
    cv = np.asarray(
        mc(min_window=25, lag=0, t_max=t_max, nrep=200, seed=0, levels=LEVELS, grid=32),
        dtype=np.float64,
    )
    assert cv.shape == (len(LEVELS), t_max), (
        f"mc_table returned {cv.shape}, contract says (len(levels), t_max)"
    )

    tail = cv[:, 40:]
    ok = np.isfinite(tail).all(axis=0)
    assert ok.sum() > 40
    assert (tail[0, ok] <= tail[1, ok] + 1e-12).all(), "90% cv exceeded 95% cv"
    assert (tail[1, ok] <= tail[2, ok] + 1e-12).all(), "95% cv exceeded 99% cv"

    again = np.asarray(
        mc(min_window=25, lag=0, t_max=t_max, nrep=200, seed=0, levels=LEVELS, grid=32),
        dtype=np.float64,
    )
    assert S.bit_identical(cv, again), "mc_table is not deterministic at a fixed seed"

    other = np.asarray(
        mc(min_window=25, lag=0, t_max=t_max, nrep=200, seed=1, levels=LEVELS, grid=32),
        dtype=np.float64,
    )
    assert not S.bit_identical(cv, other), "mc_table ignores its `seed` argument"


def test_mc_table_column_t_does_not_depend_on_t_max() -> None:
    """``cv[:, t]`` is the length-``t`` critical value -- ``t_max`` is not a parameter.

    This is the critical-value analogue of prefix invariance: if the caller's
    own sample size can move a critical value, every rejection decision is
    contaminated by the future.
    """
    mc = S.require_attr(S.critvals, "mc_table", "_critvals")
    kw = {
        "min_window": 25,
        "lag": 0,
        "nrep": 600,
        "seed": 0,
        "levels": LEVELS,
        "grid": 32,
    }
    short = np.asarray(mc(t_max=120, **kw), dtype=np.float64)
    long = np.asarray(mc(t_max=220, **kw), dtype=np.float64)[:, :120]

    cols = np.isfinite(short).all(axis=0) & np.isfinite(long).all(axis=0)
    assert cols.sum() > 50
    dev = float(np.max(np.abs(short[:, cols] - long[:, cols])))
    # Monte Carlo noise at nrep=600 is ~0.08 on a 95% quantile; a `t_max`-driven
    # window rule shifts the whole table by several tenths.
    assert dev < 0.30, (
        f"cv[:, t] moved by {dev:.3g} when t_max went from 120 to 220. The "
        "critical value is being indexed by the simulated sample size."
    )


def test_align_cv_indexes_by_date_position_never_by_sample_size() -> None:
    """The threshold at date ``t`` is the length-``t+1`` critical value.

    This is the second half of prefix invariance. A table is simulated once,
    then *sliced positionally*: the value applied at date ``t`` is the same
    whether the caller holds 200 rows or 2000. Broadcasting the last column --
    the length-``t_max`` critical value -- across the whole series is the
    classic leak, because every historical signal then moves when new data
    arrives.
    """
    align = S.require_attr(S.critvals, "align_cv", "_critvals")
    table = np.arange(3 * 50, dtype=np.float64).reshape(3, 50)

    short = np.asarray(align(table, 20), dtype=np.float64)
    long = np.asarray(align(table, 45), dtype=np.float64)
    assert short.shape == (3, 20) and long.shape == (3, 45)
    assert S.bit_identical(short, long[:, :20]), (
        "align_cv gave date t a different threshold depending on how long the "
        "series was."
    )
    assert S.bit_identical(short, table[:, :20])

    with pytest.raises(ValueError):
        align(table, 51)


def test_align_cv_composes_with_bsadf_into_a_prefix_invariant_signal() -> None:
    """Table + sequence + slice: the exceedance flag at ``t`` must never move."""
    mc = S.require_attr(S.critvals, "mc_table", "_critvals")
    align = S.require_attr(S.critvals, "align_cv", "_critvals")
    seq = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")

    table = mc(min_window=25, lag=0, t_max=200, nrep=200, seed=0, grid=32)
    y = S.bubble_series(200, seed=71)

    flags = {}
    for n in (120, 160, 200):
        stat = np.asarray(seq(y[:n], min_window=25, lag=0, grid=32), dtype=np.float64)
        cv = np.asarray(align(table, n), dtype=np.float64)[1]
        flags[n] = np.where(np.isfinite(stat) & np.isfinite(cv), stat > cv, False)
    assert np.array_equal(flags[120], flags[200][:120]), (
        "the published exceedance flags were revised when the sample grew from "
        "120 to 200 rows."
    )
    assert np.array_equal(flags[160], flags[200][:160])
    assert flags[200].any(), "the bubble fixture never triggered a single flag"


def test_kurozumi_boundary_matches_its_closed_form() -> None:
    fn = S.require_attr(S.critvals, "kurozumi_boundary", "_critvals")
    k = np.linspace(0.0, 3.0, 41)
    for q in (1.0, 1.96, 2.5):
        got = np.asarray(fn(k, q), dtype=np.float64)
        want = q * (0.73 + 0.93 * np.log(0.90 + k))
        assert got.shape == k.shape
        assert np.max(np.abs(got - want)) < 1e-12
    # The boundary must widen with the horizon and scale linearly in q.
    b = np.asarray(fn(k, 2.0), dtype=np.float64)
    assert (np.diff(b) > 0).all(), "the Kurozumi boundary must increase in k/m"
    assert np.max(np.abs(np.asarray(fn(k, 4.0)) - 2.0 * b)) < 1e-12


def test_training_max_cv_is_an_order_statistic_of_the_training_sample() -> None:
    fn = S.require_attr(S.critvals, "training_max_cv", "_critvals")
    train = np.asarray(S.random_walk(500, seed=41), dtype=np.float64)

    top = float(fn(train, alpha=None))
    assert top == pytest.approx(float(train.max()), abs=0.0), (
        "with alpha=None the Astill-style critical value is the training maximum"
    )

    cv = float(fn(train, alpha=0.10))
    assert np.isclose(train, cv).any(), (
        f"training_max_cv(alpha=0.10) returned {cv}, which is not one of the "
        "training statistics -- it must be an order statistic, not a fitted "
        "quantile of an assumed distribution."
    )
    assert cv <= top
    assert cv >= float(np.quantile(train, 0.80))

    # An order statistic is permutation invariant; a fitted quantile need not be.
    shuffled = train[np.random.default_rng(0).permutation(train.size)]
    assert float(fn(shuffled, alpha=0.10)) == cv


# --------------------------------------------------------------------------- #
# 7. Published-table reproduction
# --------------------------------------------------------------------------- #
def _gsadf_draws(
    *, t: int, r0: float, nrep: int, seed: int, lag: int = 0
) -> np.ndarray:
    """Monte Carlo GSADF: ``sup_t BSADF_t`` on ``nrep`` driftless random walks."""
    fn = S.bsadf.bsadf_sequence
    rng = np.random.default_rng(seed)
    min_window = int(np.floor(r0 * t))
    out = np.empty(nrep, dtype=np.float64)
    for i in range(nrep):
        y = np.cumsum(rng.standard_normal(t))
        seq = np.asarray(
            fn(y, min_window=min_window, lag=lag, grid=None), dtype=np.float64
        )
        finite = seq[np.isfinite(seq)]
        out[i] = finite.max() if finite.size else np.nan
    return out


def _gsadf_draws_mw(
    *, t: int, min_window: int, nrep: int, seed: int, lag: int = 0
) -> np.ndarray:
    fn = S.bsadf.bsadf_sequence
    rng = np.random.default_rng(seed)
    out = np.empty(nrep, dtype=np.float64)
    for i in range(nrep):
        y = np.cumsum(rng.standard_normal(t))
        seq = np.asarray(
            fn(y, min_window=min_window, lag=lag, grid=None), dtype=np.float64
        )
        finite = seq[np.isfinite(seq)]
        out[i] = finite.max() if finite.size else np.nan
    return out


def test_gsadf_dominates_sadf_dominates_adf_pathwise() -> None:
    """GSADF >= SADF >= ADF must hold on *every* path, not just in distribution."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    t, min_window = 160, 40
    rng = np.random.default_rng(51)
    for _ in range(12):
        y = np.cumsum(rng.standard_normal(t))
        seq = np.asarray(
            fn(y, min_window=min_window, lag=0, grid=None), dtype=np.float64
        )
        gsadf = float(np.nanmax(seq))
        fixed_start = np.array(
            [S.naive_adf(y[: e + 1], 0) for e in range(min_window - 1, t)]
        )
        sadf = float(np.nanmax(fixed_start))
        adf = S.naive_adf(y, 0)
        assert gsadf >= sadf - 1e-9, (
            f"GSADF {gsadf:.4f} < SADF {sadf:.4f}: the backward sup must include "
            "the fixed start point s=0."
        )
        assert sadf >= adf - 1e-9, f"SADF {sadf:.4f} < full-sample ADF {adf:.4f}"


@pytest.mark.slow
def test_psy_2015_finite_sample_gsadf_table() -> None:
    """The 95% GSADF critical value at T=400, r0=0.10 is 2.20-2.23 (PSY 2015)."""
    S.require("_bsadf")
    draws = _gsadf_draws(t=400, r0=0.10, nrep=4000, seed=1234)
    cv95 = float(np.quantile(draws, 0.95))
    # 4000 replications put the Monte Carlo standard error of this quantile at
    # roughly 0.02, so the band is the published 2.20-2.23 plus ~4 standard
    # errors either side. Measured here: 2.205.
    assert 2.13 <= cv95 <= 2.30, (
        f"simulated 95% GSADF cv at T=400, r0=0.10 is {cv95:.3f}; PSY (2015) "
        "report 2.20-2.23. A value near the single-window ADF critical value "
        "(-2.86) means the sup is not being taken; a much larger one means the "
        "window search is unrestricted."
    )
    cv90 = float(np.quantile(draws, 0.90))
    cv99 = float(np.quantile(draws, 0.99))
    assert cv90 < cv95 < cv99


@pytest.mark.slow
def test_gsadf_critical_values_rise_as_the_window_fraction_falls() -> None:
    """Smaller ``r0`` searches more windows, so the null sup is larger."""
    S.require("_bsadf")
    wide = _gsadf_draws(t=200, r0=0.20, nrep=600, seed=77)
    narrow = _gsadf_draws(t=200, r0=0.05, nrep=600, seed=77)
    for level in (0.90, 0.95, 0.99):
        a = float(np.quantile(wide, level))
        b = float(np.quantile(narrow, level))
        assert b > a, (
            f"at level {level:.2f} the r0=0.05 cv ({b:.3f}) did not exceed the "
            f"r0=0.20 cv ({a:.3f}); the minimum window fraction is being ignored."
        )


@pytest.mark.slow
def test_gsadf_critical_values_reproduce_the_psy_table_across_sample_sizes() -> None:
    """PSY (2015) Table 1, 95% GSADF, at their own ``r0 = 0.01 + 1.8/sqrt(T)`` rule.

    The window rule may be used *here* -- this is table generation, where the
    sample size is the thing being tabulated. Using it inside a feature, keyed
    to the caller's live sample size, is the leak that
    ``test_prefix_invariance`` exists to catch; ``psy_min_window`` is gated
    behind ``acknowledge_leak`` for exactly that reason.
    """
    S.require("_bsadf")
    published = {100: 1.99, 200: 2.08, 400: 2.23}
    got = {}
    for t, target in published.items():
        min_window = int(np.floor(t * (0.01 + 1.8 / np.sqrt(t))))
        draws = _gsadf_draws_mw(t=t, min_window=min_window, nrep=2000, seed=88)
        got[t] = float(np.quantile(draws, 0.95))
        assert abs(got[t] - target) < 0.15, (
            f"T={t}, min_window={min_window}: simulated 95% GSADF cv is "
            f"{got[t]:.3f}, PSY (2015) publish {target:.2f}."
        )
    assert got[100] < got[200] < got[400], (
        f"the critical value must rise with T under the PSY window rule: {got}"
    )


@pytest.mark.slow
def test_psy_min_window_is_gated_behind_an_explicit_leak_acknowledgement() -> None:
    """The ``floor(T*(0.01 + 1.8/sqrt(T)))`` rule must not be reachable by accident.

    It is the single most common source of look-ahead in this literature: with
    the window rule frozen, prefix invariance is exactly 0.0; with this rule the
    deviation is mean 0.123 / max 2.515 and 41.2% of dates get revised.
    """
    fn = S.require_attr(S.critvals, "psy_min_window", "_critvals")
    with pytest.raises(Exception):
        fn(500)
    assert int(fn(500, acknowledge_leak=True)) == int(
        np.floor(500 * (0.01 + 1.8 / np.sqrt(500)))
    )


def test_bsadf_sequence_has_no_default_min_window() -> None:
    """A default window would have to be a function of ``len(y)``, which leaks."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    with pytest.raises(TypeError):
        fn(S.random_walk(200, seed=0))


def test_full_sample_gsadf_is_a_scalar_not_a_per_row_feature() -> None:
    """``gsadf(y)`` must reduce to one number, so it cannot be joined onto rows.

    A GSADF computed on the whole sample and broadcast across every row hands
    the model the answer for dates that had not happened yet.
    """
    fn = S.require_attr(S.bsadf, "gsadf", "_bsadf")
    y = S.bubble_series(200, seed=61)
    out = fn(y, min_window=S.MIN_W, grid=None)
    assert np.isscalar(out) or np.ndim(out) == 0, (
        f"gsadf returned {type(out).__name__} of shape {np.shape(out)}; a "
        "full-sample GSADF must be a scalar so it cannot masquerade as a "
        "per-row feature."
    )
    seq = np.asarray(
        S.bsadf.bsadf_sequence(y, min_window=S.MIN_W, lag=0, grid=None),
        dtype=np.float64,
    )
    assert float(out) == pytest.approx(float(np.nanmax(seq)), rel=1e-12)
