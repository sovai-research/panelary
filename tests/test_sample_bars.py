"""Bars: lattice/reset vs brute-force loops, adaptive thresholds, imbalance/run bars.

The references below are written from the definitions, one explicit Python loop
per entity, with no code shared with the implementation.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary._internal._jit import assert_backend_parity, force_numpy
from panelary.core.panel_frame import PanelFrame
from panelary.registry import registry
from panelary.sample import _bars, _imbalance
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

T0 = datetime(2024, 3, 1, 9, 30)


def _ticks(
    n: int,
    *,
    entities: tuple[str, ...] = ("A", "B"),
    seed: int = 0,
    grid: float = 0.25,
    step_s: float = 7.0,
    days: bool = False,
) -> pl.DataFrame:
    """Ticks on a price grid (exact in binary for grid=0.25) with integer sizes."""
    rng = np.random.default_rng(seed)
    frames = []
    for k, name in enumerate(entities):
        m = n + 13 * k  # ragged
        px = 100.0 + grid * np.cumsum(rng.integers(-2, 3, m))
        qty = rng.integers(1, 200, m)
        if days:
            secs = np.sort(rng.uniform(0, 30 * 86400, m))
        else:
            secs = np.cumsum(rng.exponential(step_s, m))
        ts = [T0 + timedelta(seconds=float(s)) for s in secs]
        frames.append(pl.DataFrame({"sym": [name] * m, "ts": ts, "px": px, "qty": qty}))
    return pl.concat(frames)


def _brute_carry(amounts: list, theta: float) -> list[int]:
    """Indices of the ticks that close a bar: first reach of each next multiple."""
    closes, c = [], 0
    for i, v in enumerate(amounts):
        prev, c = c, c + v
        if math.floor(c / theta) > math.floor(prev / theta):
            closes.append(i)
    return closes


def _brute_reset(amounts: list, theta: float) -> list[int]:
    closes, acc = [], 0.0
    for i, v in enumerate(amounts):
        acc += v
        if acc >= theta:
            closes.append(i)
            acc = 0.0
    return closes


def _bars_from_closes(sub: pl.DataFrame, closes: list[int]) -> list[dict]:
    """Aggregate one entity's ticks into completed bars given its close indices."""
    out, start = [], 0
    px = sub["px"].to_list()
    qty = sub["qty"].to_list()
    ts = sub["ts"].to_list()
    for k, end in enumerate(closes):
        seg = slice(start, end + 1)
        out.append(
            {
                "ts": ts[end],
                "t_open": ts[start],
                "open": px[start],
                "high": max(px[seg]),
                "low": min(px[seg]),
                "close": px[end],
                "volume": sum(qty[seg]),
                "n_ticks": end - start + 1,
                "bar_index": k,
            }
        )
        start = end + 1
    return out


def _compare(got: pl.DataFrame, want: list[dict]) -> None:
    cols = [
        "ts",
        "t_open",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "n_ticks",
        "bar_index",
    ]
    assert got.height == len(want)
    assert got.select(cols).to_dicts() == want


# --------------------------------------------------------------------------- #
# Fixed bars: the lattice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "theta"), [("tick", 37), ("volume", 2500), ("dollar", 1.5e5)]
)
def test_lattice_matches_brute_force(kind: str, theta: float) -> None:
    ticks = _ticks(3000, seed=1)
    got = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind=kind,
        threshold=theta,
    )
    for name in ("A", "B"):
        sub = ticks.filter(pl.col("sym") == name)
        amounts = {
            "tick": [1] * sub.height,
            "volume": sub["qty"].to_list(),
            "dollar": (sub["px"] * sub["qty"]).to_list(),
        }[kind]
        want = _bars_from_closes(sub, _brute_carry(amounts, theta))
        _compare(got.filter(pl.col("sym") == name), want)


def test_crossing_tick_belongs_to_the_bar_it_closes() -> None:
    """sizes 3,3,3,3 at theta=5: bars [0,1] and [2,3], not [0], [1,2], [3]."""
    ticks = pl.DataFrame(
        {"sym": ["A"] * 4, "ts": [1, 2, 3, 4], "px": [1.0] * 4, "qty": [3] * 4}
    )
    out = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="volume",
        threshold=5,
    )
    assert out["t_open"].to_list() == [1, 3]
    assert out["ts"].to_list() == [2, 4]
    assert out["volume"].to_list() == [6, 6]


def test_a_trade_crossing_several_multiples_closes_one_bar() -> None:
    ticks = pl.DataFrame(
        {"sym": ["A"] * 5, "ts": range(5), "px": [1.0] * 5, "qty": [2, 20, 1, 1, 1]}
    )
    out = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="volume",
        threshold=5,
    )
    assert out.height == 2  # [0, 1] (22 crosses 5..20) and [2, 3, 4] reaches 25
    assert out["n_ticks"].to_list() == [2, 3]
    assert out["bar_index"].to_list() == [0, 1]


def test_integer_amounts_stay_exact_and_trailing_bar_is_dropped() -> None:
    ticks = _ticks(500, seed=2)
    out = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="volume",
        threshold=1000,
    )
    assert out.schema["volume"] == pl.Int64 and out.schema["buy_volume"] == pl.Int64
    # The lattice carries the overshoot: bar k ends on the first tick whose
    # cumulative volume reaches (k + 1) * theta, so single bars may be short.
    for name in ("A", "B"):
        vol = out.filter(pl.col("sym") == name)["volume"].to_numpy()
        assert (np.cumsum(vol) >= 1000 * np.arange(1, vol.size + 1)).all()
    per = ticks.group_by("sym").agg(pl.col("qty").sum())
    for name, total in per.iter_rows():
        assert out.filter(pl.col("sym") == name)["volume"].sum() <= total


def test_float_sizes_and_vwap() -> None:
    ticks = _ticks(800, seed=3).with_columns(pl.col("qty").cast(pl.Float64) / 3.0)
    out = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="volume",
        threshold=400.0,
    )
    assert out.schema["volume"] == pl.Float64
    np.testing.assert_allclose(out["vwap"], out["dollar_volume"] / out["volume"])
    assert (
        (out["vwap"] >= out["low"] - 1e-9) & (out["vwap"] <= out["high"] + 1e-9)
    ).all()


def test_tick_bars_without_size_count_ticks() -> None:
    ticks = _ticks(300, seed=4).drop("qty")
    out = pn.sample.bars(
        ticks, entity="sym", time="ts", price="px", kind="tick", threshold=10
    )
    assert (out["n_ticks"] == 10).all() and (out["volume"] == 10).all()


def test_buy_volume_follows_the_tick_rule() -> None:
    ticks = pl.DataFrame(
        {
            "sym": ["A"] * 6,
            "ts": range(6),
            "px": [10.0, 10.0, 11.0, 11.0, 10.5, 10.5],
            "qty": [1, 2, 4, 8, 16, 32],
        }
    )
    out = pn.sample.bars(
        ticks, entity="sym", time="ts", price="px", size="qty", kind="tick", threshold=6
    )
    # sides: null, null, +1, +1 (carried), -1, -1 (carried) -> buys 4 + 8
    assert out["buy_volume"].to_list() == [12]


def test_entities_are_independent() -> None:
    ticks = _ticks(600, seed=5)
    both = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="dollar",
        threshold=5e4,
    )
    alone = pn.sample.bars(
        ticks.filter(pl.col("sym") == "A"),
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="dollar",
        threshold=5e4,
    )
    assert both.filter(pl.col("sym") == "A").equals(alone)


def test_missing_ticks_are_dropped_and_bad_inputs_raise() -> None:
    ticks = _ticks(100, seed=6).with_columns(
        pl.when(pl.int_range(pl.len()) % 10 == 3)
        .then(None)
        .otherwise(pl.col("px"))
        .alias("px")
    )
    clean = ticks.drop_nulls("px")
    kw = {
        "entity": "sym",
        "time": "ts",
        "price": "px",
        "size": "qty",
        "kind": "tick",
        "threshold": 7,
    }
    assert pn.sample.bars(ticks, **kw).equals(pn.sample.bars(clean, **kw))
    with pytest.raises(ValueError, match="non-negative"):
        pn.sample.bars(clean.with_columns(-pl.col("qty")), **kw)
    with pytest.raises(ValueError, match="exactly one"):
        pn.sample.bars(
            clean, entity="sym", time="ts", price="px", size="qty", kind="tick"
        )
    with pytest.raises(ValueError, match="size"):
        pn.sample.bars(
            clean.drop("qty"),
            entity="sym",
            time="ts",
            price="px",
            kind="volume",
            threshold=5,
        )
    with pytest.raises(ValueError, match="integer"):
        pn.sample.bars(clean, **{**kw, "threshold": 7.5})
    with pytest.raises(ValueError, match="kind"):
        pn.sample.bars(clean, **{**kw, "kind": "time"})


# --------------------------------------------------------------------------- #
# Timestamps: last tick (T11) and duplicate policies
# --------------------------------------------------------------------------- #
def test_t11_bars_are_stamped_at_their_last_tick_and_are_causal() -> None:
    ticks = _ticks(400, seed=7)
    panel = PanelFrame(ticks, entity="sym", time="ts")

    def feature(frame: pl.DataFrame) -> pl.DataFrame:
        # The verifier adds +-1e6 noise to every numeric column; sizes must stay
        # valid (non-negative), so the op reads |qty|.
        frame = frame.with_columns(pl.col("qty").abs())
        b = pn.sample.bars(
            frame,
            entity="sym",
            time="ts",
            price="px",
            size="qty",
            kind="volume",
            threshold=900,
        )
        return b.with_columns((pl.col("close") / pl.col("open") - 1).alias("bar_ret"))

    assert_no_lookahead(feature, panel)
    assert_prefix_invariant(feature, panel)

    def stamped_at_open(frame: pl.DataFrame) -> pl.DataFrame:
        return feature(frame).with_columns(pl.col("t_open").alias("ts"))

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(stamped_at_open, panel)


def _dup_ticks() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sym": ["A"] * 6,
            "ts": [10, 10, 10, 10, 11, 12],
            "px": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "qty": [1] * 6,
        }
    )


def test_duplicate_close_times_nudge_error_keep() -> None:
    kw = {
        "entity": "sym",
        "time": "ts",
        "price": "px",
        "size": "qty",
        "kind": "tick",
        "threshold": 1,
    }
    nudged = pn.sample.bars(_dup_ticks(), **kw)
    assert nudged["ts"].to_list() == [10, 11, 12, 13, 14, 15]  # never earlier, unique
    kept = pn.sample.bars(_dup_ticks(), on_duplicate_time="keep", **kw)
    assert kept["ts"].to_list() == [10, 10, 10, 10, 11, 12]
    with pytest.raises(ValueError, match="same timestamp"):
        pn.sample.bars(_dup_ticks(), on_duplicate_time="error", **kw)


def test_nudge_uses_the_time_unit_of_datetimes() -> None:
    ticks = _dup_ticks().with_columns(
        (pl.lit(T0) + pl.duration(seconds=pl.col("ts"))).alias("ts")
    )
    out = pn.sample.bars(
        ticks, entity="sym", time="ts", price="px", size="qty", kind="tick", threshold=1
    )
    diffs = out["ts"].diff().drop_nulls()
    assert diffs.min() == timedelta(microseconds=1)
    assert out["ts"].n_unique() == out.height


# --------------------------------------------------------------------------- #
# Reset variant
# --------------------------------------------------------------------------- #
def test_reset_matches_brute_force_and_backends_agree() -> None:
    ticks = _ticks(2500, seed=8).with_columns(pl.col("qty").cast(pl.Float64) * 0.37)
    got = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="dollar",
        threshold=2e4,
        overshoot="reset",
    )
    for name in ("A", "B"):
        sub = ticks.filter(pl.col("sym") == name)
        want = _bars_from_closes(
            sub, _brute_reset((sub["px"] * sub["qty"]).to_list(), 2e4)
        )
        g = got.filter(pl.col("sym") == name)
        assert g["ts"].to_list() == [w["ts"] for w in want]
        assert g["n_ticks"].to_list() == [w["n_ticks"] for w in want]
    rng = np.random.default_rng(0)
    amounts = rng.exponential(3.0, 20_000)
    starts = np.zeros(amounts.size, dtype=bool)
    starts[[0, 5000, 12_345]] = True
    twin = _bars._reset_closes(amounts, starts, 17.0, _backend="numpy")
    if pytest.importorskip("numba"):
        assert_backend_parity(
            _bars._reset_closes(amounts, starts, 17.0, _backend="numba"), twin
        )


# --------------------------------------------------------------------------- #
# Adaptive thresholds (T10)
# --------------------------------------------------------------------------- #
def test_adaptive_threshold_is_trailing_and_prefix_invariant() -> None:
    ticks = _ticks(6000, seed=9, days=True, entities=("A",))
    kw = {
        "entity": "sym",
        "time": "ts",
        "price": "px",
        "size": "qty",
        "kind": "dollar",
        "bars_per_day": 10.0,
        "lookback_days": 3,
    }
    full = pn.sample.bars(ticks, **kw)
    assert full.height > 0
    days = ticks.with_columns(pl.col("ts").dt.date().alias("d"))["d"].unique().sort()
    assert full["t_open"].dt.date().min() >= days[3]  # first 3 days: no threshold yet
    for cut_day in (6, 12, 20):
        cut = datetime.combine(days[cut_day], datetime.min.time()) + timedelta(hours=13)
        pref = pn.sample.bars(ticks.filter(pl.col("ts") <= cut), **kw)
        assert pref.height > 0
        assert pref.equals(full.filter(pl.col("ts") <= cut).head(pref.height))
        assert full.filter(pl.col("ts") <= cut).height == pref.height


def test_adaptive_threshold_matches_hand_computation() -> None:
    """Exact-in-binary data: every summation order gives the same thresholds."""
    ticks = _ticks(3000, seed=10, days=True, entities=("A",))
    out = pn.sample.bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="dollar",
        bars_per_day=4.0,
        lookback_days=2,
    )
    t = ticks.sort("ts").with_columns(
        pl.col("ts").dt.date().alias("d"), (pl.col("px") * pl.col("qty")).alias("v")
    )
    daily = t.group_by("d", maintain_order=True).agg(pl.col("v").sum())
    theta = {}
    vals = daily["v"].to_list()
    for i, d in enumerate(daily["d"].to_list()):
        if i >= 2:
            theta[d] = (vals[i - 1] + vals[i - 2]) / 2 / 4.0
    kept = t.filter(pl.col("d").is_in(list(theta)))
    u = [
        v / theta[d]
        for v, d in zip(kept["v"].to_list(), kept["d"].to_list(), strict=True)
    ]
    want = _bars_from_closes(kept.rename({}), _brute_carry(u, 1.0))
    _compare(out, want)


def test_full_sample_threshold_is_a_lookahead() -> None:
    """T10 with power: a theta from the full-sample mean daily amount moves when
    the sample grows; the trailing one does not."""
    ticks = _ticks(4000, seed=11, days=True, entities=("A",))
    cut = ticks["ts"].sort()[2000]

    def naive(frame: pl.DataFrame) -> pl.DataFrame:
        per_day = frame.group_by(pl.col("ts").dt.date()).agg(
            (pl.col("px") * pl.col("qty")).sum()
        )
        theta = float(per_day["px"].mean()) / 10.0
        return pn.sample.bars(
            frame,
            entity="sym",
            time="ts",
            price="px",
            size="qty",
            kind="dollar",
            threshold=theta,
        )

    full, pref = naive(ticks), naive(ticks.filter(pl.col("ts") <= cut))
    assert not pref.equals(full.filter(pl.col("ts") <= cut).head(pref.height))


def test_init_threshold_fills_the_first_days() -> None:
    ticks = _ticks(2000, seed=12, days=True, entities=("A",))
    kw = {
        "entity": "sym",
        "time": "ts",
        "price": "px",
        "size": "qty",
        "kind": "dollar",
        "bars_per_day": 5.0,
        "lookback_days": 5,
    }
    without = pn.sample.bars(ticks, **kw)
    with_init = pn.sample.bars(ticks, init_threshold=2e4, **kw)
    assert with_init["t_open"].min() < without["t_open"].min()
    with pytest.raises(ValueError, match="Date or Datetime"):
        pn.sample.bars(ticks.with_columns(pl.int_range(pl.len()).alias("ts")), **kw)


# --------------------------------------------------------------------------- #
# Imbalance and run bars
# --------------------------------------------------------------------------- #
def _ref_info_bars(side, v, *, run, init_t, lo, hi, alpha, warmup, init_imb):
    """Reference imbalance / run bar closes for ONE entity, from the definition."""
    closes, thr_at, clamp_at = [], [], []
    e_t, clamped_prev = float(init_t), False
    w = list(zip(side[:warmup], v[:warmup], strict=True))
    if init_imb is not None:
        e_bv = init_imb
    else:
        e_bv = sum(s * x for s, x in w) / warmup if warmup else 0.0
    buys = [x for s, x in w if s > 0]
    sells = [x for s, x in w if s < 0]
    mean_v = sum(x for _, x in w) / warmup if warmup else 0.0
    p_buy = len(buys) / (len(buys) + len(sells)) if buys or sells else 0.5
    e_vb = sum(buys) / len(buys) if buys else mean_v
    e_vs = sum(sells) / len(sells) if sells else mean_v
    bar: list[tuple[int, float]] = []
    for i, (s, x) in enumerate(zip(side, v, strict=True)):
        bar.append((s, x))
        if i < warmup:
            continue
        th = 0.0
        up = dn = 0.0
        for s_, x_ in bar:
            if s_ > 0:
                th, up = th + x_, up + x_
            elif s_ < 0:
                th, dn = th - x_, dn + x_
        if run:
            thr = e_t * max(p_buy * e_vb, (1.0 - p_buy) * e_vs)
            hit = max(up, dn) >= thr
        else:
            thr = e_t * abs(e_bv)
            hit = abs(th) >= thr
        if not hit:
            continue
        closes.append(i)
        thr_at.append(thr)
        clamp_at.append(clamped_prev)
        n_b = sum(1 for s_, _ in bar if s_ > 0)
        n_s = sum(1 for s_, _ in bar if s_ < 0)
        new_t = alpha * len(bar) + (1.0 - alpha) * e_t
        clamped_prev = not (lo <= new_t <= hi)
        e_t = min(max(new_t, lo), hi)
        if run:
            if n_b + n_s:
                p_buy = alpha * (n_b / (n_b + n_s)) + (1.0 - alpha) * p_buy
            if n_b:
                e_vb = alpha * (up / n_b) + (1.0 - alpha) * e_vb
            if n_s:
                e_vs = alpha * (dn / n_s) + (1.0 - alpha) * e_vs
        else:
            e_bv = alpha * (th / len(bar)) + (1.0 - alpha) * e_bv
        bar = []
    return closes, thr_at, clamp_at


@pytest.mark.parametrize("run", [False, True])
@pytest.mark.parametrize("kind", ["tick", "volume", "dollar"])
def test_information_bars_match_the_reference(run: bool, kind: str) -> None:
    ticks = _ticks(1500, seed=13)
    init_t = 40
    got = pn.sample.imbalance_bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind=kind,
        run=run,
        init_expected_ticks=init_t,
        span_bars=5,
        bounds=(0.5, 2.0),
    )
    assert got.height > 4
    for name in ("A", "B"):
        sub = ticks.filter(pl.col("sym") == name).sort("ts", maintain_order=True)
        px = sub["px"].to_list()
        side, last = [], 0
        for j in range(len(px)):
            move = 0 if j == 0 else (px[j] > px[j - 1]) - (px[j] < px[j - 1])
            last = move if move != 0 else last
            side.append(last)
        v = {
            "tick": [1.0] * sub.height,
            "volume": [float(q) for q in sub["qty"]],
            "dollar": (sub["px"] * sub["qty"].cast(pl.Float64)).to_list(),
        }[kind]
        closes, thr, clamp = _ref_info_bars(
            side,
            v,
            run=run,
            init_t=init_t,
            lo=0.5 * init_t,
            hi=2.0 * init_t,
            alpha=2 / 6,
            warmup=init_t,
            init_imb=None,
        )
        g = got.filter(pl.col("sym") == name)
        assert g["ts"].to_list() == [sub["ts"][c] for c in closes]
        np.testing.assert_allclose(g["threshold"].to_numpy(), thr, rtol=1e-12)
        assert g["clamped"].to_list() == clamp


def test_information_bar_backends_agree_bitwise() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(1)
    n = 50_000
    side = rng.choice([-1, 0, 1], n, p=[0.45, 0.05, 0.5]).astype(np.int8)
    amount = rng.lognormal(3.0, 1.0, n)
    starts = np.zeros(n, dtype=bool)
    starts[[0, 10_000, 31_000]] = True
    for run in (False, True):
        kw = {
            "run": run,
            "init_t": 50.0,
            "lo_t": 5.0,
            "hi_t": 500.0,
            "alpha": 0.1,
            "warmup": 50,
            "init_imb": None,
        }
        twin = _imbalance._info_bars_scan(side, amount, starts, _backend="numpy", **kw)
        fast = _imbalance._info_bars_scan(side, amount, starts, _backend="numba", **kw)
        assert_backend_parity(fast, twin)
        assert twin[0].sum() > 10


def test_force_numpy_gives_identical_bars() -> None:
    ticks = _ticks(3000, seed=14)
    kw = {
        "entity": "sym",
        "time": "ts",
        "price": "px",
        "size": "qty",
        "kind": "volume",
        "init_expected_ticks": 30,
    }
    auto = pn.sample.imbalance_bars(ticks, **kw)
    with force_numpy():
        twin = pn.sample.imbalance_bars(ticks, **kw)
    assert auto.equals(twin)


def test_t9_thresholds_use_completed_bars_only() -> None:
    """Perturb ticks after bar k closes: bars <= k (and thresholds) are unchanged.
    A threshold initialised from the full sample moves (the test has power)."""
    ticks = _ticks(2000, seed=15, entities=("A",))
    kw = {
        "entity": "sym",
        "time": "ts",
        "price": "px",
        "size": "qty",
        "kind": "volume",
        "init_expected_ticks": 25,
    }
    base = pn.sample.imbalance_bars(ticks, **kw)
    k = base.height // 2
    cut = base["ts"][k]
    rng = np.random.default_rng(3)
    late = pl.col("ts") > cut
    pert = ticks.with_columns(
        pl.when(late)
        .then(pl.col("qty") * pl.lit(rng.integers(1, 9, ticks.height)))
        .otherwise(pl.col("qty"))
        .alias("qty"),
        pl.when(late)
        .then(pl.col("px") + pl.lit(rng.integers(-4, 5, ticks.height)) * 0.25)
        .otherwise(pl.col("px"))
        .alias("px"),
    )
    after = pn.sample.imbalance_bars(pert, **kw)
    assert after.head(k + 1).equals(base.head(k + 1))
    assert not after.equals(base)

    def full_init(frame: pl.DataFrame) -> pl.DataFrame:
        s = frame.sort("ts").with_columns(
            _imbalance._tick_sign(pl.col("px")).fill_null(0).alias("b")
        )
        imb = float((s["b"] * s["qty"]).mean())
        return pn.sample.imbalance_bars(frame, init_imbalance=imb, **kw)

    assert not full_init(pert).head(k + 1).equals(full_init(ticks).head(k + 1))


def test_clamp_is_reported() -> None:
    ticks = _ticks(3000, seed=16, entities=("A",))
    out = pn.sample.imbalance_bars(
        ticks,
        entity="sym",
        time="ts",
        price="px",
        size="qty",
        kind="tick",
        init_expected_ticks=20,
        bounds=(1.0, 1.0),
        span_bars=3,
    )
    assert out.height > 3
    assert not out["clamped"][0]  # the first bar uses the initial E[T]
    # bounds (1, 1) pin E[T] at 20: the update after a bar of length != 20 is
    # clamped (alpha = 0.5 makes 0.5 * T + 10 == 20 exact iff T == 20).
    prev_len = out["n_ticks"].to_list()[:-1]
    assert out["clamped"].to_list()[1:] == [n != 20 for n in prev_len]
    assert any(n != 20 for n in prev_len)


def test_imbalance_bars_validation() -> None:
    ticks = _ticks(100, seed=17)
    kw = {"entity": "sym", "time": "ts", "price": "px", "size": "qty"}
    with pytest.raises(ValueError, match="bounds"):
        pn.sample.imbalance_bars(ticks, init_expected_ticks=10, bounds=(2.0, 3.0), **kw)
    with pytest.raises(ValueError, match="imbalance bars only"):
        pn.sample.imbalance_bars(
            ticks, init_expected_ticks=10, run=True, init_imbalance=1.0, **kw
        )
    with pytest.raises(ValueError, match="warmup_ticks=0"):
        pn.sample.imbalance_bars(ticks, init_expected_ticks=10, warmup_ticks=0, **kw)
    with pytest.raises(TypeError, match="init_expected_ticks"):
        pn.sample.imbalance_bars(ticks, init_expected_ticks=10.5, **kw)  # type: ignore[arg-type]
    out = pn.sample.imbalance_bars(
        ticks, init_expected_ticks=10, init_imbalance=0.3, **kw
    )
    assert out.height > 0 and out["t_open"][0] == ticks["ts"][0]


def test_tick_sign_has_a_null_head_and_carries_zeros() -> None:
    s = pl.DataFrame({"p": [10.0, 10.0, 11.0, 11.0, 10.5, 10.5, 12.0]}).select(
        _imbalance._tick_sign(pl.col("p"))
    )
    assert s.to_series().to_list() == [None, None, 1, 1, -1, -1, 1]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_specs() -> None:
    for name in ("bars", "imbalance_bars"):
        spec = registry.get(name)
        assert (spec.namespace, spec.safe_scope, spec.intent, spec.flavour) == (
            "sample",
            "rowwise",
            "compress",
            "trailing",
        )
    assert "panelary.sample.bars(" in registry.to_llms_txt()
