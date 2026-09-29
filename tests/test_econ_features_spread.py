"""Hand-computed and behavioural tests for OHLC spreads (plan 5, M2).

The per-window reference parity (EDGE vs ``bidask`` at every index, CS and AR
vs their published formulas) lives in ``tests/test_ohlc_oracle.py``; this file
pins the small cases worked out by hand, the negative-value policies, the
moment columns, window semantics, batching and argument validation.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary.econ.features import ohlc_spread
from panelary.registry import registry

K = 3.0 - 2.0 * math.sqrt(2.0)


def _frame(bars, entity: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "e": [entity] * len(bars),
            "t": list(range(len(bars))),
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
        }
    )


def _spread(df, **kw):
    return ohlc_spread(df, entity="e", time="t", **kw)


def _cs_hand(prev, bar, adjust=True):
    h0, l0, c0 = (math.log(v) for v in prev[1:])
    h1, l1 = math.log(bar[1]), math.log(bar[2])
    beta = (h0 - l0) ** 2 + (h1 - l1) ** 2
    if adjust:
        gap = max(0.0, c0 - h1) + min(0.0, c0 - l1)
        h1a, l1a = h1 + gap, l1 + gap
    else:
        h1a, l1a = h1, l1
    gamma = (max(h0, h1a) - min(l0, l1a)) ** 2
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / K - math.sqrt(gamma / K)
    return 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))


_DAY0 = (100.0, 102.0, 99.0, 101.0)
_DAY1 = (101.0, 103.0, 100.0, 102.0)
_GAP_DOWN = (98.0, 99.0, 97.0, 98.5)  # prior close 101 above today's high


# --------------------------------------------------------------------------- #
# Hand cases
# --------------------------------------------------------------------------- #
def test_corwin_schultz_two_bar_hand_value() -> None:
    out = _spread(_frame([_DAY0, _DAY1]), method="corwin_schultz", window=2)
    s = _cs_hand(_DAY0, _DAY1)
    assert out["spread_corwin_schultz_2"][0] is None
    assert out["spread_corwin_schultz_moment_2"][1] == pytest.approx(s, rel=1e-12)
    assert out["spread_corwin_schultz_2"][1] == pytest.approx(max(s, 0.0), abs=1e-15)


@pytest.mark.parametrize("adjust", [True, False])
def test_corwin_schultz_overnight_adjustment(adjust) -> None:
    out = _spread(
        _frame([_DAY0, _GAP_DOWN]),
        method="corwin_schultz",
        window=2,
        negative="signed",
        overnight_adjust=adjust,
    )
    expected = _cs_hand(_DAY0, _GAP_DOWN, adjust)
    assert out["spread_corwin_schultz_2"][1] == pytest.approx(expected, rel=1e-12)
    # the adjustment matters here: it moves the two-day range by the gap
    assert _cs_hand(_DAY0, _GAP_DOWN, True) != pytest.approx(
        _cs_hand(_DAY0, _GAP_DOWN, False), rel=1e-6
    )


def test_corwin_schultz_tanh_form_is_accurate_near_zero() -> None:
    # alpha ~ 1e-9: the exponential form 2(e^a - 1)/(1 + e^a) loses about half
    # its digits to cancellation, 2 tanh(a/2) does not.
    a = 1e-9
    exact = 2 * math.tanh(a / 2)
    assert 2 * math.expm1(a) / (1 + math.exp(a)) == pytest.approx(exact, rel=1e-15)
    assert abs(2 * (math.exp(a) - 1) / (1 + math.exp(a)) - exact) / exact > 1e-10


def test_abdi_ranaldo_two_bar_hand_value() -> None:
    out = _spread(
        _frame([_DAY0, _DAY1]), method="abdi_ranaldo", window=2, negative="signed"
    )
    h0, l0, c0 = (math.log(v) for v in _DAY0[1:])
    h1, l1 = math.log(_DAY1[1]), math.log(_DAY1[2])
    s2 = 4 * (c0 - (h0 + l0) / 2) * (c0 - (h1 + l1) / 2)
    assert out["spread_abdi_ranaldo_moment_2"][1] == pytest.approx(s2, rel=1e-12)
    assert out["spread_abdi_ranaldo_2"][1] == pytest.approx(
        math.copysign(math.sqrt(abs(s2)), s2), rel=1e-12
    )


def test_edge_is_null_when_fewer_than_two_pairs_have_tau_one() -> None:
    # bidask's own degenerate test case: every pair has H = L = C_{t-1}
    bars = [(18.21, 18.21, 17.61, 17.61), (17.61, 17.61, 17.61, 17.61)] + [
        (17.61, 17.61, 17.61, 17.61)
    ]
    out = _spread(_frame(bars), window=3)
    assert out["spread_edge_3"].to_list() == [None, None, None]
    assert out["spread_edge_moment_3"].to_list() == [None, None, None]


# --------------------------------------------------------------------------- #
# Negative-value policies and the moment column
# --------------------------------------------------------------------------- #
def _noisy_bars(n=80, seed=3, spread=0.01, entities=("A",)):
    rng = np.random.default_rng(seed)
    rows = []
    for e in entities:
        mid = 50.0
        for t in range(n):
            mid *= math.exp(rng.normal(0, 0.01))
            path = mid * np.exp(np.cumsum(rng.normal(0, 0.02 / 6, 36)))
            trades = path * (1 + rng.choice([-1, 1], 36) * spread / 2)
            rows.append((e, t, trades[0], trades.max(), trades.min(), trades[-1]))
            mid = path[-1]
    return pl.DataFrame(
        rows, schema=["e", "t", "open", "high", "low", "close"], orient="row"
    )


@pytest.mark.parametrize("method", ["edge", "corwin_schultz", "abdi_ranaldo"])
def test_negative_policies_are_transforms_of_one_moment(method) -> None:
    df = _noisy_bars()
    outs = {
        neg: _spread(df, method=method, window=6, negative=neg)
        for neg in ("literature", "signed", "abs", "zero", "null")
    }
    col, mcol = f"spread_{method}_6", f"spread_{method}_moment_6"
    moment = outs["signed"][mcol]
    for neg, out in outs.items():
        assert out[mcol].to_list() == moment.to_list(), neg  # always the same moment
    m = moment.to_numpy().astype(float)
    assert np.nanmin(m) < 0 < np.nanmax(m)  # both signs occur on this data
    root = (lambda x: x) if method == "corwin_schultz" else np.sqrt
    signed = outs["signed"][col].to_numpy().astype(float)
    np.testing.assert_allclose(signed, np.sign(m) * root(np.abs(m)), rtol=1e-15)
    np.testing.assert_allclose(
        outs["abs"][col].to_numpy().astype(float), root(np.abs(m)), rtol=1e-15
    )
    np.testing.assert_allclose(
        outs["zero"][col].to_numpy().astype(float),
        root(np.maximum(m, 0.0)),
        rtol=1e-15,
    )
    nul = outs["null"][col]
    assert nul.filter(pl.Series(m < 0)).null_count() == int(np.sum(m < 0))
    lit = outs["literature"][col].to_numpy().astype(float)
    assert np.all(lit[~np.isnan(lit)] >= 0)
    if method == "edge":  # the reference default: sqrt(|s^2|)
        np.testing.assert_allclose(lit, np.sqrt(np.abs(m)), rtol=1e-15)


def test_moment_averaging_is_the_unbiased_aggregate() -> None:
    """Averaging clipped roots is biased; averaging the moment is not.

    On one constant-spread series, the mean of the literature (zero-clipped) CS
    spreads exceeds the mean of the unclipped two-day spreads -- the clipping
    floor -- which is why ``*_moment`` carries the unclipped mean.
    """
    df = _noisy_bars(n=400, spread=0.002)
    out = _spread(df, method="corwin_schultz", window=6)
    clipped_mean = out["spread_corwin_schultz_6"].mean()
    moment_mean = out["spread_corwin_schultz_moment_6"].mean()
    assert clipped_mean > moment_mean


# --------------------------------------------------------------------------- #
# Window semantics, batching, missing data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["edge", "corwin_schultz", "abdi_ranaldo"])
def test_window_counts_price_bars(method) -> None:
    df = _noisy_bars(n=12)
    out = _spread(df, method=method, window=5)
    col = f"spread_{method}_5"
    assert out[col][:4].null_count() == 4  # 4 bars = 3 pairs < 4 pairs
    assert out[col][4] is not None


@pytest.mark.parametrize("method", ["edge", "corwin_schultz", "abdi_ranaldo"])
def test_batching_is_bitwise_identical(method) -> None:
    df = _noisy_bars(n=40, entities=tuple("ABCDEFG"))
    outs = [
        _spread(df, method=method, window=10, batch_entities=b)
        for b in (None, 1, 3, 256)
    ]
    for other in outs[1:]:
        assert other.equals(outs[0])


def test_expanding_window_needs_explicit_min_periods() -> None:
    df = _noisy_bars(n=20)
    with pytest.raises(ValueError, match="explicit `min_periods`"):
        _spread(df, window=None)
    out = _spread(df, window=None, min_periods=5)
    assert out.columns[-2:] == ["spread_edge_expanding", "spread_edge_moment_expanding"]
    assert out["spread_edge_expanding"][:4].null_count() == 4
    assert out["spread_edge_expanding"][4:].null_count() == 0


def test_expanding_equals_a_window_longer_than_the_series() -> None:
    df = _noisy_bars(n=30)
    a = _spread(df, window=None, min_periods=3)["spread_edge_moment_expanding"]
    b = _spread(df, window=40, min_periods=3)["spread_edge_moment_40"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)


def test_min_periods_admits_gappy_windows() -> None:
    df = _noisy_bars(n=30).with_columns(
        pl.when(pl.col("t") == 10).then(None).otherwise(pl.col("open")).alias("open")
    )
    full = _spread(df, window=8)["spread_edge_8"]
    gappy = _spread(df, window=8, min_periods=5)["spread_edge_8"]
    # a missing open invalidates the pair ending on its bar (r1, r2, r5 there),
    # so every full 7-pair window that contains row 10 is null
    assert full[10:17].null_count() == 7
    assert full[17] is not None
    assert gappy[10:17].null_count() == 0


def test_invalid_bars_are_dropped_by_default() -> None:
    df = _noisy_bars(n=30).with_columns(
        pl.when(pl.col("t") == 12)
        .then(pl.col("low") * 0.5)
        .otherwise(pl.col("high"))
        .alias("high")  # high below low
    )
    out = _spread(df, method="abdi_ranaldo", window=4, negative="signed")
    assert out["spread_abdi_ranaldo_4"][12:16].null_count() == 4
    with pytest.raises(ValueError, match="invalid OHLC bar"):
        _spread(df, method="abdi_ranaldo", window=4, invalid="raise")


def test_alias_names_both_columns_and_input_is_left_intact() -> None:
    df = _noisy_bars(n=12)
    out = _spread(df, window=4, alias="s")
    assert out.columns == [*df.columns, "s", "s_moment"]
    assert out.select(df.columns).equals(df.sort("e", "t"))


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "roll"}, "method"),
        ({"negative": "clip"}, "negative"),
        ({"invalid": "drop"}, "invalid"),
        ({"window": 2}, "window"),
        ({"window": 1, "method": "corwin_schultz"}, "window"),
        ({"window": 5, "min_periods": 6}, "min_periods"),
        ({"window": 5, "min_periods": 1}, "min_periods"),
        ({"batch_entities": 0}, "batch_entities"),
    ],
)
def test_bad_arguments_raise(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _spread(_noisy_bars(n=10), **kwargs)


def test_registry_spec() -> None:
    spec = registry.get("ohlc_spread")
    assert (spec.namespace, spec.safe_scope, spec.tier) == ("econ", "rowwise", "B")
    assert spec.panel_safe and spec.leakage_safe
