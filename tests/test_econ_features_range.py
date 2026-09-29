"""Hand-computed and identity tests for OHLC range volatility (plan 5, M1).

Every estimator is checked on a three-bar example worked out with ``math``
alone, then against the identities that define it: Yang-Zhang's ``k`` uses the
row's own valid count, annualisation is a constant factor, the within-bar terms
are invariant to a common price scale, Rogers-Satchell is exactly zero on a pure
trend bar, and the discrete-monitoring correction is a division by the stored
factor. The ``.panel.rolling_vol`` rename (bug B6) is pinned bitwise.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

import panelary.namespaces.panel as panel_ns
from panelary._internal import _ohlc
from panelary.econ.features import ohlc_variance_terms, range_volatility
from panelary.registry import registry

LN2 = math.log(2.0)
GKW = 2.0 * LN2 - 1.0

#: (O, H, L, C) of three bars of one entity.
BARS = [
    (100.0, 110.0, 95.0, 105.0),
    (106.0, 112.0, 100.0, 101.0),
    (100.0, 104.0, 96.0, 103.0),
]


def _frame(bars=BARS, entity: str = "A") -> pl.DataFrame:
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


def _logs(bar):
    return tuple(math.log(v) for v in bar)


def _p(bar):
    _, h, lo, _ = _logs(bar)
    return (h - lo) ** 2 / (4.0 * LN2)


def _gk(bar):
    o, h, lo, c = _logs(bar)
    return 0.5 * (h - lo) ** 2 - GKW * (c - o) ** 2


def _rs(bar):
    o, h, lo, c = _logs(bar)
    return (h - o) * (h - c) + (lo - o) * (lo - c)


def _overnight(prev, bar):
    return math.log(bar[0]) - math.log(prev[3])


def _var(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _rv(frame, **kw):
    kw.setdefault("output", "var")
    return range_volatility(frame, entity="e", time="t", **kw)


# --------------------------------------------------------------------------- #
# Hand cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("method", "term"),
    [("parkinson", _p), ("garman_klass", _gk), ("rogers_satchell", _rs)],
)
def test_within_bar_estimators_match_hand_values(method, term) -> None:
    out = _rv(_frame(), method=method, window=2)
    got = out[f"var_{method}_2"].to_list()
    assert got[0] is None  # one bar < window
    assert got[1] == pytest.approx((term(BARS[0]) + term(BARS[1])) / 2, rel=1e-14)
    assert got[2] == pytest.approx((term(BARS[1]) + term(BARS[2])) / 2, rel=1e-14)
    assert out[f"var_{method}_2_n_valid"].to_list() == [1, 2, 2]


def test_gk_overnight_matches_hand_value() -> None:
    out = _rv(_frame(), method="gk_overnight", window=2)
    got = out["var_gk_overnight_2"].to_list()
    terms = [_overnight(BARS[i - 1], BARS[i]) ** 2 + _gk(BARS[i]) for i in (1, 2)]
    # the first bar has no previous close, so it is not a valid bar here
    assert got[:2] == [None, None]
    assert got[2] == pytest.approx(sum(terms) / 2, rel=1e-14)
    assert out["var_gk_overnight_2_n_valid"].to_list() == [0, 1, 2]


def test_close_to_close_is_the_sample_variance_of_log_returns() -> None:
    bars = [*BARS, (103.0, 108.0, 101.0, 107.5)]
    out = _rv(_frame(bars), method="close_to_close", window=3)
    r = [math.log(bars[i][3] / bars[i - 1][3]) for i in (1, 2, 3)]
    got = out["var_close_to_close_3"].to_list()
    assert got[:3] == [None, None, None]
    assert got[3] == pytest.approx(_var(r), rel=1e-13)


def test_yang_zhang_k_uses_the_rows_own_valid_count() -> None:
    # window 3, min_periods 2: at t=2 only two bars (t=1, t=2) have an
    # overnight return, so n = 2 -- not the window, not the series length.
    out = _rv(_frame(), method="yang_zhang", window=3, min_periods=2)
    got = out["var_yang_zhang_3"].to_list()
    alpha, n = 1.34, 2
    k = (alpha - 1) / (alpha + (n + 1) / (n - 1))
    ov = [_overnight(BARS[i - 1], BARS[i]) for i in (1, 2)]
    oc = [math.log(BARS[i][3] / BARS[i][0]) for i in (1, 2)]
    rs = [_rs(BARS[i]) for i in (1, 2)]
    expected = _var(ov) + k * _var(oc) + (1 - k) * sum(rs) / 2
    assert got[:2] == [None, None]
    assert got[2] == pytest.approx(expected, rel=1e-13)
    assert out["var_yang_zhang_3_n_valid"].to_list() == [0, 1, 2]


def test_yang_zhang_alpha_one_drops_the_open_to_close_variance() -> None:
    out = _rv(_frame(), method="yang_zhang", window=3, min_periods=2, alpha=1.0)
    ov = [_overnight(BARS[i - 1], BARS[i]) for i in (1, 2)]
    rs = [_rs(BARS[i]) for i in (1, 2)]
    assert out["var_yang_zhang_3"][2] == pytest.approx(
        _var(ov) + sum(rs) / 2, rel=1e-13
    )


def test_vol_is_the_square_root_of_var_and_annualisation_is_a_constant() -> None:
    var = _rv(_frame(), method="rogers_satchell", window=2)["var_rogers_satchell_2"]
    vol = range_volatility(
        _frame(), entity="e", time="t", method="rogers_satchell", window=2
    )["vol_rogers_satchell_2"]
    ann = range_volatility(
        _frame(),
        entity="e",
        time="t",
        method="rogers_satchell",
        window=2,
        periods_per_year=252,
    )["vol_rogers_satchell_2"]
    for v, s, a in zip(var.to_list()[1:], vol.to_list()[1:], ann.to_list()[1:]):
        assert s == pytest.approx(math.sqrt(v), rel=1e-15)
        assert a == pytest.approx(math.sqrt(252 * v), rel=1e-14)


def test_alias_names_both_columns() -> None:
    out = range_volatility(
        _frame(), entity="e", time="t", method="parkinson", window=2, alias="pk"
    )
    assert out.columns[-2:] == ["pk", "pk_n_valid"]


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #
def _random_bars(n_entities=3, n=60, seed=0, overnight=0.004, drift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        prev = 50.0 * (e + 1)
        for t in range(n):
            o = prev * math.exp(rng.normal(0.0, overnight))
            steps = rng.normal(drift / 40, 0.01 / math.sqrt(40), 40)
            path = o * np.exp(np.cumsum(steps))
            rows.append(
                (f"e{e}", t, o, max(o, path.max()), min(o, path.min()), path[-1])
            )
            prev = path[-1]
    return pl.DataFrame(
        rows, schema=["e", "t", "open", "high", "low", "close"], orient="row"
    )


_METHODS = [
    "yang_zhang",
    "gk_overnight",
    "rogers_satchell",
    "garman_klass",
    "parkinson",
    "close_to_close",
]


@pytest.mark.parametrize("method", _METHODS)
def test_scale_invariance(method) -> None:
    df = _random_bars()
    scaled = df.with_columns(pl.col(c) * 7.3 for c in ("open", "high", "low", "close"))
    a = range_volatility(df, entity="e", time="t", method=method, window=10)
    b = range_volatility(scaled, entity="e", time="t", method=method, window=10)
    col = f"vol_{method}_10"
    diff = (a[col] - b[col]).abs().max()
    assert diff is not None and diff <= 1e-14


@pytest.mark.parametrize("method", _METHODS)
def test_float32_prices_are_upcast_before_any_arithmetic(method) -> None:
    df = _random_bars(n_entities=2, n=30).with_columns(
        pl.col(c).cast(pl.Float32) for c in ("open", "high", "low", "close")
    )
    upcast = df.with_columns(
        pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")
    )
    a = range_volatility(df, entity="e", time="t", method=method, window=5)
    b = range_volatility(upcast, entity="e", time="t", method=method, window=5)
    col = f"vol_{method}_5"
    assert a[col].dtype == pl.Float64
    assert a[col].to_list() == b[col].to_list()


def test_rogers_satchell_is_zero_on_a_pure_trend_bar() -> None:
    # A bar that moves monotonically from open to close has H = max(O, C) and
    # L = min(O, C): RS is exactly 0, Parkinson and GK are not -- the drift
    # bias the RS estimator was designed to remove.
    up = [(100.0, 103.0, 100.0, 103.0), (103.0, 103.0, 99.0, 99.0)]
    df = _frame(up)
    rs = _rv(df, method="rogers_satchell", window=1)["var_rogers_satchell_1"]
    pk = _rv(df, method="parkinson", window=1)["var_parkinson_1"]
    gk = _rv(df, method="garman_klass", window=1)["var_garman_klass_1"]
    assert rs.to_list() == [0.0, 0.0]
    assert all(v > 0 for v in pk.to_list())
    assert all(v > 0 for v in gk.to_list())


def test_zero_range_bars_are_kept_as_zero_variance() -> None:
    bars = [(100.0, 100.0, 100.0, 100.0), *BARS[1:]]
    out = _rv(_frame(bars), method="parkinson", window=2)
    assert out["var_parkinson_2_n_valid"].to_list() == [1, 2, 2]
    assert out["var_parkinson_2"][1] == pytest.approx(_p(bars[1]) / 2, rel=1e-14)


def test_variance_terms_match_the_windowed_estimators() -> None:
    df = _random_bars()
    terms = ohlc_variance_terms(df, entity="e", time="t")
    for method, col in [
        ("parkinson", "ohlc_parkinson"),
        ("garman_klass", "ohlc_garman_klass"),
        ("rogers_satchell", "ohlc_rogers_satchell"),
        ("gk_overnight", "ohlc_gk_overnight"),
    ]:
        mean = terms.select(pl.col(col).rolling_mean(7).over("e"))[col]
        got = _rv(df, method=method, window=7)[f"var_{method}_7"]
        assert (mean - got).abs().max() <= 1e-18
    o, h, lo, c = (terms[k].to_numpy() for k in ("open", "high", "low", "close"))
    np.testing.assert_allclose(terms["ohlc_u"].to_numpy(), np.log(h / o), atol=2e-15)
    np.testing.assert_allclose(terms["ohlc_d"].to_numpy(), np.log(lo / o), atol=2e-15)
    np.testing.assert_allclose(terms["ohlc_c"].to_numpy(), np.log(c / o), atol=2e-15)
    first = terms.group_by("e").agg(pl.col("ohlc_o").first())["ohlc_o"]
    assert first.null_count() == first.len()  # no previous close on row 0


def test_lazyframe_input_is_accepted_and_output_sorted() -> None:
    df = _random_bars(n=20)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=3)
    a = range_volatility(df, entity="e", time="t", window=5)
    b = range_volatility(shuffled.lazy(), entity="e", time="t", window=5)
    assert a.equals(b)


# --------------------------------------------------------------------------- #
# Discrete-monitoring correction
# --------------------------------------------------------------------------- #
def _exact_rs_factor(m: int) -> float:
    a = np.arange(1, m + 1, dtype=float) ** -0.5
    prefix = np.concatenate([[0.0], np.cumsum(a)])
    j = np.arange(1, m)
    return float(np.sum(a[j - 1] * prefix[m - j]) / (math.pi * m))


def test_stored_rogers_satchell_factors_are_the_exact_values() -> None:
    for m, b in zip(_ohlc._DISCRETE_M, _ohlc._DISCRETE_FACTORS["rogers_satchell"]):
        assert b == pytest.approx(_exact_rs_factor(m), abs=5e-7)


def test_discreteness_factor_interpolates_and_is_monotone() -> None:
    for term in ("parkinson", "garman_klass", "rogers_satchell"):
        table = _ohlc._DISCRETE_FACTORS[term]
        for m, b in zip(_ohlc._DISCRETE_M, table):
            assert _ohlc.discreteness_factor(term, m) == pytest.approx(b, abs=1e-15)
        ms = [2, 7, 50, 77, 390, 391, 5000, 23400, 10**6]
        vals = [_ohlc.discreteness_factor(term, m) for m in ms]
        assert all(a < b for a, b in zip(vals, vals[1:]))
        assert vals[-1] < 1.0
    # the plan's measured spot values (M=390: P 0.933, GK 0.906, RS 0.905)
    assert _ohlc.discreteness_factor("parkinson", 390) == pytest.approx(0.933, abs=3e-3)
    assert _ohlc.discreteness_factor("garman_klass", 390) == pytest.approx(
        0.906, abs=5e-3
    )
    assert _ohlc.discreteness_factor("rogers_satchell", 390) == pytest.approx(
        0.905, abs=5e-3
    )
    # interpolation in 1/sqrt(M) is accurate between grid points
    assert _ohlc.discreteness_factor("rogers_satchell", 300) == pytest.approx(
        _exact_rs_factor(300), abs=2e-4
    )


@pytest.mark.parametrize("bad", [1, 0, -5, 2.5, True, "78"])
def test_discreteness_factor_rejects_bad_bar_counts(bad) -> None:
    with pytest.raises(ValueError, match="discrete_bars"):
        _ohlc.discreteness_factor("parkinson", bad)


@pytest.mark.parametrize(
    ("method", "term"),
    [
        ("parkinson", "parkinson"),
        ("garman_klass", "garman_klass"),
        ("rogers_satchell", "rogers_satchell"),
    ],
)
def test_discrete_bars_divides_the_term_by_the_stored_factor(method, term) -> None:
    df = _random_bars()
    raw = _rv(df, method=method, window=10)[f"var_{method}_10"]
    cor = _rv(df, method=method, window=10, discrete_bars=78)[f"var_{method}_10"]
    b = _ohlc.discreteness_factor(term, 78)
    np.testing.assert_allclose(cor.to_numpy(), raw.to_numpy() / b, rtol=1e-13)


def test_discrete_bars_corrects_only_the_range_part_of_yz_and_gkyz() -> None:
    df = _random_bars()
    terms = ohlc_variance_terms(df, entity="e", time="t")
    b_gk = _ohlc.discreteness_factor("garman_klass", 26)
    gko = (
        terms.select(
            (pl.col("ohlc_o") ** 2 + pl.col("ohlc_garman_klass") / b_gk)
            .rolling_mean(10)
            .over("e")
        )
        .to_series()
        .to_numpy()
    )
    got = _rv(df, method="gk_overnight", window=10, discrete_bars=26)
    np.testing.assert_allclose(got["var_gk_overnight_10"].to_numpy(), gko, rtol=1e-13)

    b_rs = _ohlc.discreteness_factor("rogers_satchell", 26)
    k = 0.34 / (1.34 + 11 / 9)
    yz = terms.select(
        (
            pl.col("ohlc_o").rolling_var(10).over("e")
            + k * pl.col("ohlc_c").rolling_var(10).over("e")
            + (1 - k)
            * (pl.col("ohlc_rogers_satchell") / b_rs).rolling_mean(10).over("e")
        ).alias("yz")
    )["yz"]
    # the first bar of each entity has no overnight return, so the windows of
    # the reference above (which do not mask it) differ only in warm-up rows
    got_yz = _rv(df, method="yang_zhang", window=10, discrete_bars=26)
    ok = got_yz["var_yang_zhang_10"].is_not_null() & yz.is_not_null()
    np.testing.assert_allclose(
        got_yz["var_yang_zhang_10"].filter(ok).to_numpy(),
        yz.filter(ok).to_numpy(),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Invalid-bar policy
# --------------------------------------------------------------------------- #
_BROKEN = [BARS[0], (106.0, 104.0, 100.0, 105.0), BARS[2]]  # high < close


def test_invalid_null_drops_the_bar() -> None:
    out = _rv(_frame(_BROKEN), method="rogers_satchell", window=3, min_periods=1)
    assert out["var_rogers_satchell_3_n_valid"].to_list() == [1, 1, 2]
    assert out["var_rogers_satchell_3"][2] == pytest.approx(
        (_rs(BARS[0]) + _rs(BARS[2])) / 2, rel=1e-14
    )


def test_invalid_clip_repairs_the_ordering_row_locally() -> None:
    out = _rv(
        _frame(_BROKEN),
        method="rogers_satchell",
        window=3,
        min_periods=1,
        invalid="clip",
    )
    fixed = (106.0, 106.0, 100.0, 105.0)  # H = max(H, O, C)
    assert out["var_rogers_satchell_3"][2] == pytest.approx(
        (_rs(BARS[0]) + _rs(fixed) + _rs(BARS[2])) / 3, rel=1e-14
    )


def test_invalid_clip_still_drops_non_positive_prices() -> None:
    bars = [BARS[0], (106.0, 112.0, 0.0, 101.0), BARS[2]]
    out = _rv(_frame(bars), method="parkinson", window=3, min_periods=1, invalid="clip")
    assert out["var_parkinson_3_n_valid"].to_list() == [1, 1, 2]


def test_invalid_raise_names_the_first_bad_bar() -> None:
    with pytest.raises(ValueError, match=r"1 invalid OHLC bar.*e='A', t=1"):
        _rv(_frame(_BROKEN), method="rogers_satchell", window=2, invalid="raise")


def test_invalid_keep_uses_the_bar_as_is() -> None:
    out = _rv(
        _frame(_BROKEN),
        method="rogers_satchell",
        window=3,
        min_periods=1,
        invalid="keep",
    )
    assert out["var_rogers_satchell_3"][2] == pytest.approx(
        (_rs(BARS[0]) + _rs(_BROKEN[1]) + _rs(BARS[2])) / 3, rel=1e-14
    )


def test_missing_prices_are_not_filled_from_a_neighbour() -> None:
    df = _frame().with_columns(
        pl.when(pl.col("t") == 1).then(None).otherwise(pl.col("high")).alias("high")
    )
    out = _rv(df, method="parkinson", window=3, min_periods=1)
    assert out["var_parkinson_3_n_valid"].to_list() == [1, 1, 2]


def test_validity_is_judged_on_the_columns_the_method_reads() -> None:
    # A bad open does not invalidate a Parkinson bar (which never reads it).
    bars = [BARS[0], (-1.0, 112.0, 100.0, 101.0), BARS[2]]
    pk = _rv(_frame(bars), method="parkinson", window=3)
    rs = _rv(_frame(bars), method="rogers_satchell", window=3, min_periods=1)
    assert pk["var_parkinson_3_n_valid"].to_list() == [1, 2, 3]
    assert rs["var_rogers_satchell_3_n_valid"].to_list() == [1, 1, 2]
    parkinson_only = _frame(bars).drop("open", "close")
    assert _rv(parkinson_only, method="parkinson", window=3)["var_parkinson_3"][2] > 0


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "garman"}, "method"),
        ({"output": "std"}, "output"),
        ({"invalid": "drop"}, "invalid"),
        ({"window": 0}, "window"),
        ({"window": 1, "method": "yang_zhang"}, "window"),
        ({"window": 1, "method": "close_to_close"}, "window"),
        ({"window": 5, "min_periods": 6}, "min_periods"),
        ({"min_periods": 0}, "min_periods"),
        ({"alpha": 0.5}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"periods_per_year": 0}, "periods_per_year"),
        ({"periods_per_year": float("inf")}, "periods_per_year"),
        ({"method": "close_to_close", "discrete_bars": 78}, "discrete_bars"),
        ({"discrete_bars": 1}, "discrete_bars"),
    ],
)
def test_bad_arguments_raise(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        range_volatility(_frame(), entity="e", time="t", **kwargs)


def test_missing_price_column_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        range_volatility(_frame().drop("open"), entity="e", time="t")


def test_periods_per_year_has_no_inference_path() -> None:
    import inspect

    params = inspect.signature(range_volatility).parameters
    assert params["periods_per_year"].default is None
    assert not any("infer" in name for name in params)


# --------------------------------------------------------------------------- #
# .panel.rolling_vol and the rs_vol deprecation (bug B6)
# --------------------------------------------------------------------------- #
_X = [1.0, 2.0, 4.0, 7.0, 11.0, 3.0, None, 5.0]


def _series_frame() -> pl.DataFrame:
    return pl.DataFrame({"e": ["a"] * 8 + ["b"] * 8, "x": _X + [v and -v for v in _X]})


def test_rolling_vol_is_the_pre_change_expression_bitwise() -> None:
    df = _series_frame()
    pre_change = df.select(pl.col("x").rolling_std(window_size=3).over("e"))["x"]
    got = df.select(pl.col("x").panel.rolling_vol(3).over("e"))["x"]
    assert got.to_list() == pre_change.to_list()
    # golden values, hand computed: std of (1, 2, 4) and (2, 4, 7), ddof 1
    assert got[2] == pytest.approx(math.sqrt(7 / 3), rel=1e-12)
    assert got[3] == pytest.approx(math.sqrt(19 / 3), rel=1e-12)


def test_rs_vol_is_bitwise_rolling_vol_and_warns_once(monkeypatch) -> None:
    monkeypatch.setattr(panel_ns, "_RS_VOL_WARNED", False)
    df = _series_frame()
    with pytest.warns(FutureWarning, match="rolling standard deviation, not Rogers"):
        legacy = df.select(pl.col("x").panel.rs_vol(3).over("e"))["x"]
    honest = df.select(pl.col("x").panel.rolling_vol(3).over("e"))["x"]
    assert legacy.to_list() == honest.to_list()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second warning would raise here
        df.select(pl.col("x").panel.rs_vol(3).over("e"))


def test_frame_rs_vol_is_bitwise_rolling_vol_and_warns(monkeypatch) -> None:
    monkeypatch.setattr(panel_ns, "_RS_VOL_WARNED", False)
    df = _series_frame()
    with pytest.warns(FutureWarning, match="range_volatility"):
        legacy = df.panel.rs_vol("x", window=3, over="e", alias="v")
    honest = df.panel.rolling_vol("x", window=3, over="e", alias="v")
    lazy = df.lazy().panel.rolling_vol("x", window=3, over="e", alias="v").collect()
    assert legacy.equals(honest)
    assert lazy.equals(honest)


def test_rolling_vol_frame_op_requires_over() -> None:
    with pytest.raises(ValueError, match="requires an `over`"):
        _series_frame().panel.rolling_vol("x", window=3)


def test_rolling_vol_rejects_bad_windows() -> None:
    with pytest.raises(ValueError, match="rolling_vol window"):
        pl.col("x").panel.rolling_vol(0)


def test_registry_specs_for_the_rename() -> None:
    new, old = registry.get("rolling_vol"), registry.get("rs_vol")
    assert (new.tier, new.safe_scope, new.panel_safe) == ("A", "rowwise", True)
    assert old.tier == "D"
    assert "deprecated alias of rolling_vol" in old.source
    spec = registry.get("range_volatility")
    assert (spec.namespace, spec.safe_scope, spec.tier) == ("econ", "rowwise", "B")
