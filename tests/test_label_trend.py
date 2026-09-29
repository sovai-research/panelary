"""Trend scanning: the recursive-residual sweep, the label and the ``.ts`` feature.

Oracles are clean-room from the definition of an OLS slope t-statistic: a
two-pass centred OLS, ``numpy.linalg.lstsq``, and an exact rational
(``fractions.Fraction``) evaluation for the near-perfect-line regime where every
double-precision method loses digits.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from fractions import Fraction

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary._internal._jit import assert_backend_parity, force_numpy
from panelary.label import _trend
from panelary.registry import registry
from panelary.testing import assert_no_lookahead, assert_prefix_invariant


# --------------------------------------------------------------------------- #
# Oracles and fixtures
# --------------------------------------------------------------------------- #
def _two_pass_t(w: np.ndarray) -> float:
    """Slope t-statistic of ``w ~ a + b x``, ``x = 0..L-1``, two-pass centred."""
    n = w.size
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), w.mean()
    sxx = ((x - xm) ** 2).sum()
    beta = ((x - xm) * (w - ym)).sum() / sxx
    sse = ((w - ym - beta * (x - xm)) ** 2).sum()
    return float(beta / np.sqrt(sse / (n - 2) / sxx))


def _exact_t(w: np.ndarray) -> float:
    """The same statistic with every step in exact rational arithmetic."""
    n = w.size
    ys = [Fraction(float(v)) for v in w]
    xs = [Fraction(i) for i in range(n)]
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    beta = sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=True)) / sxx
    sse = sum((y - ym - beta * (x - xm)) ** 2 for x, y in zip(xs, ys, strict=True))
    if sse == 0:
        return math.copysign(math.inf, float(beta))
    return math.copysign(
        math.sqrt(float(beta * beta * sxx * (n - 2) / sse)), float(beta)
    )


def _single_horizon_t(y: np.ndarray, length: int, backend: str = "numpy") -> np.ndarray:
    """``t(L)`` at every row (NaN where the forward window does not fit)."""
    n = y.size
    avail = n - np.arange(n)
    t, h, _b, _ok = _trend._sweep(
        y, avail, direction=1, min_w=length, l_max=length, step=1, _backend=backend
    )
    return np.where(h == length, t, np.nan)


def _random_walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.log(100.0) + np.cumsum(rng.standard_normal(n) * 0.01)


def _panel(
    lengths: dict[str, int], seed: int = 0, *, dates: bool = False
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for name, n in lengths.items():
        p = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        t = (
            [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
            if dates
            else list(range(n))
        )
        frames.append(pl.DataFrame({"id": [name] * n, "t": t, "close": p}))
    return pl.concat(frames)


def _brute_label(
    y: np.ndarray, i: int, min_w: int, l_max: int, step: int
) -> tuple[float, int]:
    """Best-|t| horizon at row ``i`` by explicit OLS on every grid window."""
    best, best_h = -1.0, 0
    for length in range(min_w, l_max + 1, step):
        w = y[i : i + length]
        if w.size < length or not np.isfinite(w).all():
            break
        t = _two_pass_t(w)
        if abs(t) > best:
            best, best_h = abs(t), length
    return best, best_h


# --------------------------------------------------------------------------- #
# Accuracy of the sweep
# --------------------------------------------------------------------------- #
def test_every_horizon_matches_two_pass_ols() -> None:
    """``t(L)`` for every row and every L in 5..50 to rtol 1e-10 (random walks).

    A t-statistic near zero (a flat window) carries an *absolute* error of a few
    1e-13 in either method, so the relative bound gets a matching 1e-10 floor.
    """
    y = _random_walk(600, seed=1)
    for length in range(5, 51):
        got = _single_horizon_t(y, length)
        want = np.array(
            [
                _two_pass_t(y[i : i + length]) if i + length <= y.size else np.nan
                for i in range(y.size)
            ]
        )
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_matches_lstsq_on_raw_prices() -> None:
    """Independent check with ``numpy.linalg.lstsq`` (raw price levels)."""
    rng = np.random.default_rng(2)
    y = 100.0 + np.cumsum(rng.standard_normal(200))
    for length in (5, 17, 40):
        got = _single_horizon_t(y, length)
        for i in range(0, y.size - length, 13):
            w = y[i : i + length]
            x = np.column_stack([np.ones(length), np.arange(length, dtype=float)])
            coef, res, *_ = np.linalg.lstsq(x, w, rcond=None)
            cov = res[0] / (length - 2) * np.linalg.inv(x.T @ x)
            np.testing.assert_allclose(got[i], coef[1] / np.sqrt(cov[1, 1]), rtol=1e-9)


def test_random_walk_accuracy_vs_exact_rational() -> None:
    y = _random_walk(80, seed=3)
    for length in (5, 12, 33, 50):
        got = _single_horizon_t(y, length)
        for i in range(0, y.size - length, 7):
            want = _exact_t(y[i : i + length])
            assert abs(got[i] - want) <= 1e-10 * abs(want)


def test_near_perfect_lines_stay_accurate() -> None:
    """R^2 -> 1, where prefix-sum shortcuts are wrong by up to 99 %.

    Relative noise ~2e-10: every double-precision method loses digits here, so
    the oracle is exact rational arithmetic. Measured max ~1.6e-7; asserted 1e-6.
    """
    rng = np.random.default_rng(4)
    for _ in range(20):
        length = int(rng.integers(5, 51))
        y = 4.6 + 1e-3 * np.arange(length) + 1e-9 * rng.standard_normal(length)
        got = _single_horizon_t(y, length)[0]
        want = _exact_t(y)
        assert abs(got - want) <= 1e-6 * abs(want)


def test_perfect_integer_line_is_infinite_and_ties_go_to_the_shortest() -> None:
    """SSE == 0 exactly -> t = +inf at every horizon; the tie picks min_window."""
    df = pl.DataFrame({"id": ["a"] * 30, "t": range(30), "close": np.arange(30.0)})
    out = pn.label.trend_scanning(df, min_window=4, max_window=10, log=False)
    first = out.row(0, named=True)
    assert first["t_value"] == math.inf
    assert first["horizon"] == 4
    assert first["label"] == 1
    assert first["slope"] == 1.0


def test_ties_resolve_to_the_smallest_horizon_in_the_kernel() -> None:
    y = np.arange(20.0) * 2.0  # exact line: every horizon has t = inf
    t, h, _b, _ok = _trend._sweep(
        y, 20 - np.arange(20), direction=1, min_w=5, l_max=12, step=1, _backend="numpy"
    )
    assert h[0] == 5 and t[0] == math.inf


def test_label_matches_brute_force_argmax() -> None:
    df = _panel({"a": 60, "b": 45}, seed=5)
    out = pn.label.trend_scanning(df, min_window=5, max_window=15, step=2)
    for name in ("a", "b"):
        sub = out.filter(pl.col("id") == name)
        y = np.log(sub["close"].to_numpy())
        for i in range(sub.height):
            row = sub.row(i, named=True)
            if row["censored"]:
                assert row["label"] is None
                continue
            best, best_h = _brute_label(y, i, 5, 15, 2)
            assert row["horizon"] == best_h
            np.testing.assert_allclose(abs(row["t_value"]), best, rtol=1e-10)
            assert row["label"] == int(np.sign(row["t_value"]))


# --------------------------------------------------------------------------- #
# Missing values, degenerate windows, thresholds, validation
# --------------------------------------------------------------------------- #
def test_missing_value_invalidates_that_and_every_longer_window() -> None:
    y = _random_walk(60, seed=6)
    y_nan = y.copy()
    y_nan[10] = np.nan  # rows 0..9: windows of length <= 10 - i stay valid
    n = y.size
    avail = n - np.arange(n)
    t_nan, h_nan, _, ok_nan = _trend._sweep(
        y_nan, avail, direction=1, min_w=3, l_max=20, step=1, _backend="numpy"
    )
    for i in range(0, 8):
        # The scan must equal a scan truncated just before the missing value.
        best, best_h = _brute_label(y, i, 3, min(20, 10 - i), 1)
        assert h_nan[i] == best_h
        np.testing.assert_allclose(abs(t_nan[i]), best, rtol=1e-12)
    # The shortest window contains the NaN -> no statistic at all.
    assert not ok_nan[8] and not ok_nan[10]
    assert h_nan[10] == 0 and np.isnan(t_nan[10])


def test_missing_price_gives_a_null_label_not_a_zero() -> None:
    df = _panel({"a": 40}, seed=7).with_columns(
        pl.when(pl.col("t") == 5).then(None).otherwise(pl.col("close")).alias("close")
    )
    out = pn.label.trend_scanning(df, min_window=3, max_window=6)
    assert out.filter(pl.col("t") == 5)["label"].item() is None
    assert (
        out.filter(pl.col("t") == 4)["label"].item() is None
    )  # [4, 5, 6] has the null
    assert out.filter(pl.col("t") == 2)["label"].item() is not None  # [2, 3, 4] is fine


def test_constant_window_is_label_zero_with_null_statistic() -> None:
    df = pl.DataFrame({"id": ["a"] * 12, "t": range(12), "close": [5.0] * 12})
    out = pn.label.trend_scanning(df, min_window=3, max_window=5)
    row = out.row(0, named=True)
    assert row["label"] == 0
    assert row["t_value"] is None and row["horizon"] is None
    assert row["t1"] == 4 and row["censored"] is False


def test_t_threshold_zeroes_weak_trends() -> None:
    df = _panel({"a": 80}, seed=8)
    loose = pn.label.trend_scanning(df, min_window=5, max_window=10)
    strict = pn.label.trend_scanning(df, min_window=5, max_window=10, t_threshold=3.0)
    weak = loose["t_value"].abs() < 3.0
    assert (strict.filter(weak)["label"].drop_nulls() == 0).all()
    kept = loose.filter(~weak & pl.col("label").is_not_null())
    assert kept.height > 0
    assert (
        strict.filter(~weak & pl.col("label").is_not_null())["label"] == kept["label"]
    ).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_window": 2}, "min_window"),
        ({"min_window": 6, "max_window": 5}, "max_window"),
        ({"step": 0}, "step"),
        ({"t_threshold": -1.0}, "t_threshold"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pn.label.trend_scanning(_panel({"a": 30}), **kwargs)


def test_window_arguments_must_be_integers() -> None:
    with pytest.raises(TypeError, match="fixed integer"):
        pn.label.trend_scanning(_panel({"a": 30}), max_window=10.0)  # type: ignore[arg-type]


def test_log_requires_positive_prices() -> None:
    df = pl.DataFrame({"id": ["a"] * 10, "t": range(10), "close": np.arange(10.0) - 3})
    with pytest.raises(ValueError, match="log=False"):
        pn.label.trend_scanning(df, min_window=3, max_window=4)
    out = pn.label.trend_scanning(df, min_window=3, max_window=4, log=False)
    assert out["label"][0] == 1


def test_grid_end_is_the_last_scanned_horizon() -> None:
    """max_window off the grid: L_max = 5 + 2*3 = 11, and t1 is set by it."""
    out = pn.label.trend_scanning(
        _panel({"a": 30}), min_window=5, max_window=12, step=3
    )
    assert out["t1"][0] == 10
    assert out.filter(~pl.col("censored"))["t"].max() == 30 - 11


# --------------------------------------------------------------------------- #
# Output contract: t1, t1_trend, censoring, ragged panels, dtypes
# --------------------------------------------------------------------------- #
def test_t1_is_the_information_end_and_t1_trend_the_chosen_horizon() -> None:
    df = _panel({"a": 50, "b": 33, "c": 12}, seed=9, dates=True)
    out = pn.label.trend_scanning(df, min_window=4, max_window=9)
    assert out.schema["t1"] == pl.Date and out.schema["t1_trend"] == pl.Date
    ok = out.filter(pl.col("label").is_not_null())
    assert ((ok["t1"] - ok["t"]).dt.total_days() == 8).all()
    assert ((ok["t1_trend"] - ok["t"]).dt.total_days() == ok["horizon"] - 1).all()
    assert (ok["t1_trend"] <= ok["t1"]).all()


def test_censoring_is_per_entity_on_ragged_panels() -> None:
    lengths = {"a": 40, "b": 25, "c": 7}
    out = pn.label.trend_scanning(_panel(lengths, seed=10), min_window=3, max_window=8)
    for name, n in lengths.items():
        sub = out.filter(pl.col("id") == name)
        expected = np.arange(n) + 8 > n
        assert sub["censored"].to_list() == expected.tolist()
        assert sub.filter(pl.col("censored"))["label"].null_count() == int(
            expected.sum()
        )
        assert sub.filter(pl.col("censored"))["t1"].null_count() == int(expected.sum())


def test_allow_partial_labels_the_tail_but_keeps_it_censored() -> None:
    df = _panel({"a": 30}, seed=11)
    out = pn.label.trend_scanning(df, min_window=3, max_window=8, allow_partial=True)
    tail = out.filter(pl.col("censored"))
    assert tail.height == 7
    labelled = tail.filter(pl.col("label").is_not_null())
    assert labelled.height == 5  # rows 22..26 have >= 3 forward rows
    assert (labelled["t1"] == 29).all()  # information end = entity's last row
    assert (labelled["horizon"] <= 30 - labelled["t"]).all()


def test_lazyframe_input_and_default_key_columns() -> None:
    df = _panel({"a": 20, "b": 20}, seed=12)
    eager = pn.label.trend_scanning(df, min_window=3, max_window=5)
    lazy = pn.label.trend_scanning(df.lazy(), min_window=3, max_window=5)
    assert eager.equals(lazy)
    assert eager.schema["label"] == pl.Int64 and eager.schema["censored"] == pl.Boolean


# --------------------------------------------------------------------------- #
# Leak-safety: traps T6 and T7
# --------------------------------------------------------------------------- #
def test_t6_label_depends_on_data_after_t1_trend_but_never_after_t1() -> None:
    """T6: ``t1_trend`` is not a valid purge end; ``t1`` is."""
    df = _panel({"a": 120}, seed=13)
    out = pn.label.trend_scanning(df, min_window=5, max_window=30)
    short = out.filter((pl.col("horizon") < 15) & ~pl.col("censored"))
    assert short.height > 0
    row = short.row(0, named=True)
    i, h = row["t"], row["horizon"]

    # (a) Rewrite (t1_trend, t1] as a violent opposite trend: the label moves.
    p = df["close"].to_numpy().copy()
    seg = np.arange(i + h, i + 30)
    p[seg] = p[i + h - 1] * np.exp(-row["label"] * 0.5 * (seg - (i + h - 1)))
    moved = pn.label.trend_scanning(
        df.with_columns(pl.Series("close", p)), min_window=5, max_window=30
    ).row(i, named=True)
    assert (moved["label"], moved["horizon"]) != (row["label"], row["horizon"])

    # (b) Perturb everything after t1 = i + 29: no row with t1 < cut moves.
    cut = i + 30
    p2 = df["close"].to_numpy().copy()
    p2[cut:] *= np.exp(np.random.default_rng(0).standard_normal(p2.size - cut))
    after = pn.label.trend_scanning(
        df.with_columns(pl.Series("close", p2)), min_window=5, max_window=30
    )
    resolved = out["t1"].fill_null(10**9) < cut
    cols = ["label", "t_value", "horizon", "slope", "t1", "t1_trend", "censored"]
    assert out.filter(resolved).select(cols).equals(after.filter(resolved).select(cols))


@pytest.mark.parametrize("allow_partial", [False, True])
def test_t7_resolved_prefix_invariance(allow_partial: bool) -> None:
    """Labels with ``t1 <= tau`` and not censored are bitwise stable under truncation."""
    df = _panel({"a": 90, "b": 70, "c": 40}, seed=14)
    kw = {"min_window": 4, "max_window": 12, "allow_partial": allow_partial}
    full = pn.label.trend_scanning(df, **kw)
    cols = ["label", "t_value", "horizon", "slope", "t1", "t1_trend"]
    for tau in (20, 45, 66):
        pref = pn.label.trend_scanning(df.filter(pl.col("t") <= tau), **kw)
        resolved = ~pl.col("censored") & (pl.col("t1") <= tau)
        mine = pref.filter(resolved)
        theirs = full.join(mine.select("id", "t"), on=["id", "t"]).sort("id", "t")
        assert mine.height > 0
        assert mine.sort("id", "t").select(cols).equals(theirs.select(cols))
        # And every resolved-in-full row that fits the prefix is resolved in it.
        fits = full.filter(resolved).height
        assert mine.height == fits


# --------------------------------------------------------------------------- #
# Backends: numba == numpy, bitwise, at any thread count
# --------------------------------------------------------------------------- #
def _ragged_inputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 400, size=60)
    y = np.concatenate([_random_walk(int(n), int(s)) for s, n in enumerate(lengths)])
    y[rng.choice(y.size, 25, replace=False)] = np.nan
    y[100:110] = 3.0  # a constant stretch
    ends = np.repeat(np.cumsum(lengths) - 1, lengths)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.arange(y.size)
    return y, ends - rows + 1, rows - starts + 1


@pytest.mark.parametrize("direction", [1, -1])
def test_numba_matches_numpy_bitwise(direction: int) -> None:
    pytest.importorskip("numba")
    y, fwd, bwd = _ragged_inputs(15)
    avail = fwd if direction == 1 else bwd
    kw = {"direction": direction, "min_w": 4, "l_max": 37, "step": 3}
    twin = _trend._sweep(y, avail, _backend="numpy", **kw)
    fast = _trend._sweep(y, avail, _backend="numba", **kw)
    assert_backend_parity(fast, twin)


def test_thread_count_does_not_change_a_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numba")
    monkeypatch.setattr(_trend, "_MIN_ROWS_PER_THREAD", 16)
    y, fwd, _ = _ragged_inputs(16)
    kw = {"direction": 1, "min_w": 5, "l_max": 30, "step": 1, "_backend": "numba"}
    one = _trend._sweep(y, fwd, threads=1, **kw)  # type: ignore[arg-type]
    many = _trend._sweep(y, fwd, threads=7, **kw)  # type: ignore[arg-type]
    assert_backend_parity(many, one)


def test_force_numpy_gives_identical_labels() -> None:
    df = _panel({"a": 300, "b": 150}, seed=17)
    auto = pn.label.trend_scanning(df, min_window=5, max_window=40)
    with force_numpy():
        twin = pn.label.trend_scanning(df, min_window=5, max_window=40)
    assert auto.equals(twin)


def test_numba_backend_request_without_numba_raises() -> None:
    with force_numpy(), pytest.raises(RuntimeError, match="unavailable"):
        _trend._sweep(
            np.ones(10),
            np.arange(10),
            direction=1,
            min_w=3,
            l_max=4,
            step=1,
            _backend="numba",
        )


# --------------------------------------------------------------------------- #
# The look-back feature: .ts.trend_scan
# --------------------------------------------------------------------------- #
def test_ts_trend_scan_matches_a_brute_force_trailing_scan() -> None:
    df = _panel({"a": 70, "b": 30}, seed=18).with_columns(pl.col("close").log())
    out = df.with_columns(
        pl.col("close")
        .ts.trend_scan(min_window=4, max_window=12)
        .over("id")
        .alias("t_"),
        pl.col("close")
        .ts.trend_scan(min_window=4, max_window=12, output="horizon")
        .over("id")
        .alias("h_"),
        pl.col("close")
        .ts.trend_scan(min_window=4, max_window=12, output="slope")
        .over("id")
        .alias("b_"),
    )
    assert out.schema["h_"] == pl.Int64
    for name in ("a", "b"):
        sub = out.filter(pl.col("id") == name)
        y = sub["close"].to_numpy()
        for i in range(sub.height):
            if i < 3:
                assert sub["t_"][i] is None and sub["h_"][i] is None
                continue
            best, best_h = -1.0, 0
            for length in range(4, min(12, i + 1) + 1):
                t = _two_pass_t(y[i - length + 1 : i + 1])
                if abs(t) > best:
                    best, best_h = abs(t), length
            assert sub["h_"][i] == best_h
            np.testing.assert_allclose(abs(sub["t_"][i]), best, rtol=1e-10)
            w = y[i - best_h + 1 : i + 1]
            np.testing.assert_allclose(
                sub["b_"][i], np.polyfit(np.arange(best_h), w, 1)[0]
            )


def test_ts_trend_scan_is_causal_and_prefix_invariant() -> None:
    df = _panel({"a": 40, "b": 33, "c": 18}, seed=19).with_columns(
        pl.when((pl.col("id") == "b") & (pl.col("t") == 9))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    panel = pn.PanelFrame(df, entity="id", time="t")
    for output in ("t", "horizon", "slope"):
        op = (
            pl.col("close")
            .ts.trend_scan(min_window=3, max_window=9, output=output)
            .over("id")
            .alias("f")
        )
        assert_no_lookahead(op, panel)
        assert_prefix_invariant(op, panel)


def test_ts_trend_scan_rejects_bad_output() -> None:
    with pytest.raises(ValueError, match="output"):
        pl.col("x").ts.trend_scan(output="r2")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_scopes() -> None:
    feat = registry.get("trend_scan")
    assert (feat.namespace, feat.safe_scope, feat.flavour) == (
        "ts",
        "rowwise",
        "trailing",
    )
    lab = registry.get("trend_scanning")
    assert (lab.namespace, lab.safe_scope) == ("label", "window")
