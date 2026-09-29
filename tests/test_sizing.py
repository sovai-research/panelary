"""Bet sizing: closed-form parity, the active-bet average vs brute force, T15, T21."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary._internal._special import norm_cdf


# --------------------------------------------------------------------------- #
# bet_size
# --------------------------------------------------------------------------- #
def _closed_form(p: np.ndarray, k: int) -> np.ndarray:
    z = (p - 1.0 / k) / np.sqrt(p * (1.0 - p))
    return 2.0 * np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in z]) - 1.0


def test_bet_size_matches_scipy_closed_form() -> None:
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 500)
    side = rng.choice([-1.0, 1.0], 500)
    df = pl.DataFrame({"prob": p, "side": side})
    out = pn.sizing.bet_size(df)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    np.testing.assert_allclose(
        out["size"], side * (2 * st.norm.cdf(z) - 1), rtol=1e-13, atol=1e-15
    )


def test_bet_size_multiclass_pred_and_no_side() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.34, 0.99, 200)
    pred = rng.choice([-1, 0, 1], 200)
    df = pl.DataFrame({"prob": p, "pred": pred})
    out = pn.sizing.bet_size(df, side=None, pred="pred", n_classes=3)
    np.testing.assert_allclose(
        out["size"], pred * _closed_form(p, 3), rtol=1e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(norm_cdf(0.3)), 0.5 * math.erfc(-0.3 / math.sqrt(2)), rtol=0
    )


def test_bet_size_edges_clipping_and_nulls() -> None:
    df = pl.DataFrame({"prob": [0.0, 0.5, 1.0, None], "side": [1.0, -1.0, -1.0, 1.0]})
    out = pn.sizing.bet_size(df)["size"].to_list()
    assert out[0] == -1.0 and out[1] == 0.0 and out[2] == -1.0 and out[3] is None
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pn.sizing.bet_size(pl.DataFrame({"prob": [1.2], "side": [1.0]}))
    with pytest.raises(ValueError, match="n_classes"):
        pn.sizing.bet_size(df, n_classes=1)


def test_bet_size_meta_labelling_sign() -> None:
    df = pl.DataFrame({"prob": [0.8, 0.8, 0.8], "side": [3.0, -2.0, 0.0]})
    out = pn.sizing.bet_size(df)["size"].to_list()
    assert out[0] == -out[1] > 0 and out[2] == 0.0


# --------------------------------------------------------------------------- #
# average_active
# --------------------------------------------------------------------------- #
def _panel_with_bets(
    seed: int, lengths: dict[str, int], density: float = 0.35, max_hold: int = 12
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for name, n in lengths.items():
        days = [date(2022, 1, 3) + timedelta(days=i) for i in range(n)]
        is_bet = rng.random(n) < density
        size = np.where(is_bet, rng.uniform(-1, 1, n), np.nan)
        hold = rng.integers(0, max_hold, n)
        exit_ = [days[0] + timedelta(days=int(i + h)) for i, h in enumerate(hold)]
        exit_null = rng.random(n) < 0.1
        frames.append(
            pl.DataFrame(
                {"id": [name] * n, "d": days, "size": size, "exit": exit_}
            ).with_columns(
                pl.when(pl.Series(exit_null))
                .then(None)
                .otherwise(pl.col("exit"))
                .alias("exit"),
                pl.col("size").fill_nan(None),
            )
        )
    return pl.concat(frames)


def _brute_active(df: pl.DataFrame) -> list[float]:
    """AFML 10.2 definition: mean size of bets with t0 <= t < exit, per entity."""
    out = []
    rows = df.sort("id", "d").to_dicts()
    for r in rows:
        active = [
            b["size"]
            for b in rows
            if b["id"] == r["id"]
            and b["size"] is not None
            and b["d"] <= r["d"]
            and (b["exit"] is None or r["d"] < b["exit"])
        ]
        out.append(float(np.mean(active)) if active else 0.0)
    return out


def test_average_active_matches_brute_force_on_every_row() -> None:
    df = _panel_with_bets(2, {"a": 80, "b": 55, "c": 9})
    out = pn.sizing.average_active(df, entity="id", time="d")
    np.testing.assert_allclose(
        out["avg_size"], _brute_active(df), rtol=1e-12, atol=1e-15
    )


def test_idle_rows_are_exactly_zero() -> None:
    df = _panel_with_bets(3, {"a": 300}, density=0.08, max_hold=6)
    out = pn.sizing.average_active(df, entity="id", time="d")
    brute = np.array(_brute_active(df))
    idle = brute == 0.0
    assert idle.any()
    assert (out["avg_size"].to_numpy()[idle] == 0.0).all()


def test_exit_off_grid_and_same_day_exit() -> None:
    df = pl.DataFrame(
        {
            "id": ["a"] * 5,
            "t": [0, 2, 4, 6, 8],
            "size": [0.5, None, -1.0, None, None],
            "exit": [3, None, 4, None, None],  # 3 is off-grid; 4 == its own time
        }
    )
    out = pn.sizing.average_active(df, entity="id", time="t", exit_time="exit")
    assert out["avg_size"].to_list() == [0.5, 0.5, 0.0, 0.0, 0.0]


def test_open_bets_run_to_the_entity_end_but_not_beyond() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "t": [0, 1, 0, 1],
            "size": [0.4, None, None, 0.2],
            "exit": [None, None, None, None],
        },
        schema_overrides={"exit": pl.Int64},
    )
    out = pn.sizing.average_active(df, entity="id", time="t", exit_time="exit")
    assert out["avg_size"].to_list() == [0.4, 0.4, 0.0, 0.2]


def test_t15_moving_a_later_exit_never_changes_earlier_sizes() -> None:
    df = _panel_with_bets(4, {"a": 60})
    base = pn.sizing.average_active(df, entity="id", time="d")
    t_star = date(2022, 1, 3) + timedelta(days=30)
    later = df.with_columns(
        pl.when(pl.col("exit") > t_star)
        .then(pl.col("exit") + timedelta(days=17))
        .otherwise(pl.col("exit"))
        .alias("exit")
    )
    moved = pn.sizing.average_active(later, entity="id", time="d")
    keep = base["d"] <= t_star
    assert base.filter(keep)["avg_size"].equals(moved.filter(keep)["avg_size"])


def test_average_active_validation() -> None:
    df = pl.DataFrame(
        {
            "id": ["a"] * 3,
            "t": [0, 1, 2],
            "size": [0.5, None, None],
            "exit": [-1, None, None],
        }
    )
    with pytest.raises(ValueError, match="precede"):
        pn.sizing.average_active(df, entity="id", time="t")
    with pytest.raises(TypeError, match="dtype"):
        pn.sizing.average_active(
            df.with_columns(pl.col("exit").cast(pl.Float64)), entity="id", time="t"
        )


# --------------------------------------------------------------------------- #
# discretize (T21)
# --------------------------------------------------------------------------- #
def test_discretize_is_half_to_even_and_clipped() -> None:
    x = np.array([0.25, 0.75, -0.25, 1.25, -3.0, 0.1])
    got = pn.sizing.discretize(x, step=0.5)
    np.testing.assert_array_equal(got, [0.0, 1.0, -0.0, 1.0, -1.0, 0.0])
    assert pn.sizing.discretize(0.26, step=0.1) == pytest.approx(0.3)
    s = pn.sizing.discretize(pl.Series("m", [0.25, 0.75]), step=0.5)
    assert isinstance(s, pl.Series) and s.to_list() == [0.0, 1.0]
    with pytest.raises(ValueError, match="step"):
        pn.sizing.discretize(x, step=0.0)


# --------------------------------------------------------------------------- #
# Dynamic sizing and limit prices
# --------------------------------------------------------------------------- #
def test_sigmoid_round_trips() -> None:
    w = pn.sizing.sigmoid_w(divergence=10.0, size=0.95)
    assert pn.sizing.sigmoid_size(w, 10.0) == pytest.approx(0.95, abs=1e-15)
    f = 100.0
    for m in (-0.9, -0.3, 0.0, 0.42, 0.99):
        p = pn.sizing.inverse_price(f, w, m)
        assert pn.sizing.sigmoid_size(w, f - p) == pytest.approx(m, abs=1e-12)
    xs = np.linspace(-50, 50, 11)
    np.testing.assert_allclose(pn.sizing.sigmoid_size(w, xs), xs / np.sqrt(w + xs**2))


def test_target_position_truncates() -> None:
    w = pn.sizing.sigmoid_w(10.0, 0.95)
    got = pn.sizing.target_position(
        w, np.array([110.0, 90.0, 100.0, 103.0]), 100.0, max_pos=100
    )
    m3 = 3.0 / math.sqrt(w + 9.0)
    np.testing.assert_array_equal(got, [95, -95, 0, math.trunc(m3 * 100)])


def _signed_loop(tgt: int, pos: int, f: float, w: float, q: int) -> float:
    sgn = 1 if tgt >= pos else -1
    path = range(pos + sgn, tgt + sgn, sgn)
    return float(
        np.mean([f - (j / q) * math.sqrt(w / (1 - (j / q) ** 2)) for j in path])
    )


def _afml_loop(tgt: int, pos: int, f: float, w: float, q: int) -> float:
    """AFML snippet 10.4 as printed (valid for 0 <= pos < tgt)."""
    sgn = 1 if tgt >= pos else -1
    lp = 0.0
    for j in range(abs(pos + sgn), abs(tgt + 1)):
        lp += f - (j / q) * math.sqrt(w / (1 - (j / q) ** 2))
    return lp / (tgt - pos)


def test_limit_price_matches_the_explicit_signed_loop() -> None:
    w = pn.sizing.sigmoid_w(10.0, 0.95)
    rng = np.random.default_rng(5)
    tgt = rng.integers(-99, 100, 400)
    pos = rng.integers(-99, 100, 400)
    f = rng.uniform(90, 110, 400)
    got = pn.sizing.limit_price(tgt, pos, f, w, max_pos=100)
    for i in range(400):
        if tgt[i] == pos[i]:
            assert math.isnan(got[i])
        else:
            want = _signed_loop(int(tgt[i]), int(pos[i]), float(f[i]), w, 100)
            assert got[i] == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_limit_price_parity_with_afml_on_its_valid_case() -> None:
    w = pn.sizing.sigmoid_w(10.0, 0.95)
    for tgt, pos in ((50, 0), (99, 10), (3, 2)):
        got = pn.sizing.limit_price(tgt, pos, 100.0, w, max_pos=100)
        assert got == pytest.approx(_afml_loop(tgt, pos, 100.0, w, 100), rel=1e-12)


def test_limit_price_rejects_full_size() -> None:
    w = pn.sizing.sigmoid_w(10.0, 0.95)
    with pytest.raises(ValueError, match="max_pos"):
        pn.sizing.limit_price(100, 0, 100.0, w, max_pos=100)
    with pytest.raises(ValueError, match="max_pos"):
        pn.sizing.limit_price(-100, 5, 100.0, w, max_pos=100)
    with pytest.raises(ValueError, match="size"):
        pn.sizing.sigmoid_w(10.0, 1.0)
    with pytest.raises(ValueError, match="same sign"):
        pn.sizing.sigmoid_w(-10.0, 0.5)
    with pytest.raises(ValueError, match=r"\|m\| < 1"):
        pn.sizing.inverse_price(100.0, w, 1.0)
