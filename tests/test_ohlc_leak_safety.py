"""Leak-safety of the OHLC volatility, spread and liquidity estimators (plan 5 §9).

Every estimator is run through BOTH instruments of :mod:`panelary.testing` --
value perturbation (``assert_no_lookahead``) and truncation
(``assert_prefix_invariant``) -- on a panel whose entities have different
lengths, gaps and invalid bars. Each named trap of the plan's §9 that can be
checked by data is written here twice: the shipped estimator passes, and a
deliberately leaky variant written in this file is *caught*, which is what
proves the check has teeth.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary.econ.features import ohlc_spread, range_volatility
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

ENTITY, TIME = "e", "t"
_LENGTHS = {"a": 60, "b": 48, "c": 33, "d": 20}


def ohlc_panel(seed: int = 7, overnight: float = 0.004) -> pl.DataFrame:
    """Random-walk OHLC bars; entity ``b`` has a missing and an invalid bar."""
    rng = np.random.default_rng(seed)
    rows = []
    for e, n in _LENGTHS.items():
        prev = 20.0 + 30.0 * rng.random()
        for t in range(n):
            o = prev * math.exp(rng.normal(0.0, overnight))
            path = o * np.exp(np.cumsum(rng.normal(0.0, 0.012 / 6.0, 36)))
            h, lo, c = max(o, path.max()), min(o, path.min()), float(path[-1])
            if e == "b" and t == 10:
                h = c * 0.99  # invalid: high below close
            if e == "b" and t == 20:
                lo = None  # missing
            rows.append((e, t, o, h, lo, c))
            prev = c
    return pl.DataFrame(
        rows,
        schema={
            ENTITY: pl.Utf8,
            TIME: pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
        orient="row",
    )


_METHODS = [
    "yang_zhang",
    "gk_overnight",
    "rogers_satchell",
    "garman_klass",
    "parkinson",
    "close_to_close",
]


def _both(op, panel) -> None:
    assert_no_lookahead(op, panel, entity=ENTITY, time=TIME)
    assert_prefix_invariant(op, panel, entity=ENTITY, time=TIME)


# --------------------------------------------------------------------------- #
# Range volatility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("min_periods", [None, 4])
def test_range_volatility_is_causal_and_prefix_invariant(method, min_periods) -> None:
    def op(df):
        return range_volatility(
            df,
            entity=ENTITY,
            time=TIME,
            method=method,
            window=8,
            min_periods=min_periods,
        )

    _both(op, ohlc_panel())


def test_range_volatility_with_correction_and_annualisation_is_causal() -> None:
    def op(df):
        return range_volatility(
            df,
            entity=ENTITY,
            time=TIME,
            window=6,
            discrete_bars=390,
            periods_per_year=252.0,
            invalid="clip",
        )

    _both(op, ohlc_panel())


def _leaky_yang_zhang(df: pl.DataFrame, window: int = 8) -> pl.DataFrame:
    """Trap 2: Yang-Zhang with k from n = len(series) -- a length leak."""
    lo_, hi_ = pl.col("low").log(), pl.col("high").log()
    o, c = pl.col("open").log(), pl.col("close").log()
    staged = df.sort(ENTITY, TIME).with_columns(
        (o - c.shift(1).over(ENTITY)).alias("ov"),
        (c - o).alias("oc"),
        ((hi_ - o) * (hi_ - c) + (lo_ - o) * (lo_ - c)).alias("rs"),
        pl.len().over(ENTITY).cast(pl.Float64).alias("n"),  # <- the leak
    )
    k = 0.34 / (1.34 + (pl.col("n") + 1) / (pl.col("n") - 1))
    return staged.with_columns(
        pl.col("ov").rolling_var(window).over(ENTITY).alias("v_o"),
        pl.col("oc").rolling_var(window).over(ENTITY).alias("v_c"),
        pl.col("rs").rolling_mean(window).over(ENTITY).alias("v_rs"),
    ).with_columns(
        (pl.col("v_o") + k * pl.col("v_c") + (1 - k) * pl.col("v_rs")).alias("yz")
    )


def test_trap2_series_length_k_is_detected() -> None:
    panel = ohlc_panel().filter(pl.col(ENTITY) != "b")
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(_leaky_yang_zhang, panel, entity=ENTITY, time=TIME)
    # the value-perturbation instrument alone cannot see a length leak
    assert_no_lookahead(_leaky_yang_zhang, panel, entity=ENTITY, time=TIME)


def test_trap6_back_adjustment_does_not_move_log_ratio_estimates() -> None:
    """A split at t = s, back-adjusted: range estimates before s do not move.

    Every term is a log *ratio* of prices at most one bar apart, so rescaling
    all prices before the split leaves every earlier row bit-for-bit close
    (the same ratios). A price-*level* measure does move.
    """
    raw = ohlc_panel().filter(pl.col(ENTITY) == "a")
    s = 30
    adjusted = raw.with_columns(
        pl.when(pl.col(TIME) < s).then(pl.col(c) * 0.5).otherwise(pl.col(c)).alias(c)
        for c in ("open", "high", "low", "close")
    )
    for method in _METHODS:
        a = range_volatility(raw, entity=ENTITY, time=TIME, method=method, window=8)
        b = range_volatility(
            adjusted, entity=ENTITY, time=TIME, method=method, window=8
        )
        col = f"vol_{method}_8"
        before = pl.col(TIME) < s
        diff = (a.filter(before)[col] - b.filter(before)[col]).abs().max()
        assert diff is not None and diff < 1e-15

    # A level-based measure: the trailing std of close-price *changes* (the
    # existing ``roll_spread`` on levels has the same exposure).
    def level_vol(df: pl.DataFrame) -> pl.Series:
        staged = df.sort(ENTITY, TIME).with_columns(
            pl.col("close").diff().over(ENTITY).alias("dp")
        )
        return staged.select(pl.col("dp").rolling_std(8).over(ENTITY))["dp"]

    before = (raw[TIME] < s).to_list()
    moved = (level_vol(raw) - level_vol(adjusted)).filter(pl.Series(before)).abs()
    assert moved.max() > 1e-6


def test_trap7_a_bad_future_bar_is_never_filled_backwards() -> None:
    """Perturb bar t+1 into an invalid bar: row t is unchanged under every policy."""
    base = ohlc_panel().filter(pl.col(ENTITY) == "a")
    broken = base.with_columns(
        pl.when(pl.col(TIME) == 31).then(-1.0).otherwise(pl.col("low")).alias("low")
    )
    for invalid in ("null", "clip"):
        for method in _METHODS:
            a = range_volatility(
                base, entity=ENTITY, time=TIME, method=method, window=5, invalid=invalid
            )
            b = range_volatility(
                broken,
                entity=ENTITY,
                time=TIME,
                method=method,
                window=5,
                invalid=invalid,
            )
            col = f"vol_{method}_5"
            upto = pl.col(TIME) <= 30
            assert a.filter(upto)[col].to_list() == b.filter(upto)[col].to_list()


# --------------------------------------------------------------------------- #
# OHLC spreads
# --------------------------------------------------------------------------- #
_SPREADS = ["edge", "corwin_schultz", "abdi_ranaldo"]


@pytest.mark.parametrize("method", _SPREADS)
@pytest.mark.parametrize(("window", "min_periods"), [(8, None), (8, 4), (None, 5)])
def test_ohlc_spread_is_causal_and_prefix_invariant(
    method, window, min_periods
) -> None:
    def op(df):
        return ohlc_spread(
            df,
            entity=ENTITY,
            time=TIME,
            method=method,
            window=window,
            min_periods=min_periods,
            negative="signed",
            batch_entities=2,
        )

    _both(op, ohlc_panel())


def _leaky_corwin_schultz(df: pl.DataFrame) -> pl.DataFrame:
    """Trap 1: the paper's (t, t+1) pair written with shift(-1), on row t."""
    h, lo = pl.col("high").log(), pl.col("low").log()
    staged = df.sort(ENTITY, TIME).with_columns(
        h.alias("h"),
        lo.alias("l"),
        h.shift(-1).over(ENTITY).alias("h1"),  # <- the leak
        lo.shift(-1).over(ENTITY).alias("l1"),
    )
    h, lo, h1, l1 = (pl.col(c) for c in ("h", "l", "h1", "l1"))
    beta = (h - lo) ** 2 + (h1 - l1) ** 2
    hi2 = pl.when(h >= h1).then(h).otherwise(h1)
    lo2 = pl.when(lo <= l1).then(lo).otherwise(l1)
    k = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = ((2 * beta).sqrt() - beta.sqrt()) / k - ((hi2 - lo2) ** 2 / k).sqrt()
    return staged.with_columns((2.0 * (alpha / 2).tanh()).alias("s2")).with_columns(
        pl.col("s2").rolling_mean(5).over(ENTITY).alias("cs")
    )


def test_trap1_forward_indexed_two_day_pairs_are_detected() -> None:
    panel = ohlc_panel().filter(pl.col(ENTITY) != "b")
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_lookahead(_leaky_corwin_schultz, panel, entity=ENTITY, time=TIME)

    # ... and the shipped estimator, indexed by the later bar, is clean
    def ours(df):
        return ohlc_spread(
            df, entity=ENTITY, time=TIME, method="corwin_schultz", window=6
        )

    assert_no_lookahead(ours, panel, entity=ENTITY, time=TIME)


def test_trap13_window_equal_to_the_data_length_is_detected() -> None:
    """``bidask.edge_expanding``-style: window = len(df), pandas-default periods."""
    panel = ohlc_panel().filter(pl.col(ENTITY) == "a")

    def leaky(df):
        # (aliased, so both runs emit the same column name to compare)
        return ohlc_spread(df, entity=ENTITY, time=TIME, window=df.height, alias="s")

    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(leaky, panel, entity=ENTITY, time=TIME)

    def ours(df):
        return ohlc_spread(df, entity=ENTITY, time=TIME, window=None, min_periods=10)

    assert_prefix_invariant(ours, panel, entity=ENTITY, time=TIME)


def test_trap7_spreads_never_fill_a_bad_future_bar_backwards() -> None:
    base = ohlc_panel().filter(pl.col(ENTITY) == "a")
    broken = base.with_columns(
        pl.when(pl.col(TIME) == 31).then(-1.0).otherwise(pl.col("low")).alias("low")
    )
    for method in _SPREADS:
        a = ohlc_spread(base, entity=ENTITY, time=TIME, method=method, window=5)
        b = ohlc_spread(broken, entity=ENTITY, time=TIME, method=method, window=5)
        col = f"spread_{method}_5"
        upto = pl.col(TIME) <= 30
        assert a.filter(upto)[col].to_list() == b.filter(upto)[col].to_list()


# --------------------------------------------------------------------------- #
# Plan §8.4: the banned whole-column shortcut
# --------------------------------------------------------------------------- #
def test_whole_column_rolling_is_not_bitwise_per_entity_rolling() -> None:
    """Evidence for keeping the §8.4 ban (rolling over the whole sorted column).

    Rolling over the concatenated column and nulling each entity's warm-up is
    tempting (no ``.over``), but the rolling kernel's floating state crosses the
    entity boundary: appending rows to entity ``a`` perturbs entity ``b``. If
    this ever starts passing bitwise on a new polars, the ban can be lifted --
    with this test flipped as the recorded evidence.
    """
    rng = np.random.default_rng(0)
    n, w = 500, 21
    df = pl.DataFrame(
        {
            "e": ["a"] * n + ["b"] * n,
            "x": np.r_[rng.normal(0, 1e8, n), rng.normal(0, 1e-3, n)],
        }
    ).with_columns(i=pl.int_range(pl.len()).over("e"))
    over = df.select(pl.col("x").rolling_mean(w).over("e"))["x"]
    whole = df.select(pl.when(pl.col("i") >= w - 1).then(pl.col("x").rolling_mean(w)))[
        "x"
    ]
    assert not (over == whole).fill_null(value=True).all()
