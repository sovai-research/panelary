"""Hand-computed, identity and reference tests for ``intraday_realized_measures``.

Every measure is checked three ways: a small hand-computed session, a
per-session numpy reference on random sessions (price and return inputs), and
the algebraic identities that tie measures together (``rs_pos + rs_neg = rv``,
``rv_ss`` = mean of the offset-subsampled RVs, pre-averaging as a double box
filter). The input-handling rules -- sessions start from their own first price,
invalid prices are dropped, stamps sit at the session close -- each get a test.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from panelary._internal._realized_kernel import (
    pre_averaged_returns,
    preaverage_k,
    preaverage_psi,
    realized_kernel,
)
from panelary.econ.features import intraday_realized_measures

_MU43 = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
_ALL = [
    "rv",
    "rv_ss",
    "bv",
    "medrv",
    "minrv",
    "rs_pos",
    "rs_neg",
    "sjv",
    "rq",
    "tpq",
    "jump_z",
    "n_obs",
    "jump",
    "rel_jump",
    "jump_sig",
    "cont",
    "medrq",
    "log_rv_var",
    "log_rv_var_tpq",
    "rk",
    "tsrv",
    "pav",
]


def _one_session(returns: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "e": ["A"] * len(returns),
            "s": [0] * len(returns),
            "t": list(range(len(returns))),
            "r": returns,
        }
    )


def _price_panel(seed: int = 1, *, sessions: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, int, int, float]] = []
    for ent in ("A", "B"):
        for sess in range(sessions):
            m = int(rng.integers(40, 400))
            logp = np.cumsum(
                np.r_[np.log(50.0 + 10 * sess), rng.standard_normal(m) * 1e-3]
            )
            rows.extend((ent, sess, i, float(np.exp(v))) for i, v in enumerate(logp))
    return pl.DataFrame(rows, schema=["e", "s", "t", "px"], orient="row")


def _reference(
    r: np.ndarray, *, subsample: int, tsrv_k: int, rk_h: int
) -> dict[str, float]:
    """Straight numpy formulas for one session."""
    m = r.size
    a = np.abs(r)
    p = np.r_[0.0, np.cumsum(r)]
    med = np.median(np.c_[a[:-2], a[1:-1], a[2:]], axis=1)
    out = {
        "rv": float(np.sum(r**2)),
        "rv_ss": float(np.sum((p[subsample:] - p[:-subsample]) ** 2) / subsample),
        "bv": np.pi / 2 * m / (m - 1) * float(np.sum(a[1:] * a[:-1])),
        "medrv": np.pi
        / (6 - 4 * np.sqrt(3) + np.pi)
        * m
        / (m - 2)
        * float(np.sum(med**2)),
        "minrv": np.pi
        / (np.pi - 2)
        * m
        / (m - 1)
        * float(np.sum(np.minimum(a[1:], a[:-1]) ** 2)),
        "rs_pos": float(np.sum(r[r > 0] ** 2)),
        "rs_neg": float(np.sum(r[r < 0] ** 2)),
        "rq": m / 3 * float(np.sum(r**4)),
        "tpq": m
        * _MU43**-3
        * m
        / (m - 2)
        * float(np.sum((a[2:] * a[1:-1] * a[:-2]) ** (4 / 3))),
        "medrq": 3
        * np.pi
        / (9 * np.pi + 72 - 52 * np.sqrt(3))
        * m
        * m
        / (m - 2)
        * float(np.sum(med**4)),
    }
    out["sjv"] = out["rs_pos"] - out["rs_neg"]
    theta = np.pi**2 / 4 + np.pi - 5
    out["jump_z"] = ((out["rv"] - out["bv"]) / out["rv"]) / math.sqrt(
        theta / m * max(1.0, out["tpq"] / out["bv"] ** 2)
    )
    out["log_rv_var"] = 2 / 3 * float(np.sum(r**4)) / out["rv"] ** 2
    out["log_rv_var_tpq"] = 2 * out["tpq"] / (m * out["bv"] ** 2)
    k = tsrv_k
    rv_k = float(np.sum((p[k:] - p[:-k]) ** 2)) / k
    ratio = ((m - k + 1) / k) / m
    out["tsrv"] = (rv_k - ratio * out["rv"]) / (1 - ratio)
    kp = preaverage_k(m, 1.0)
    psi1, psi2 = preaverage_psi(kp)
    ybar = pre_averaged_returns(r, kp)
    out["pav"] = (
        m / (m - kp + 2) / (kp * psi2) * float(np.sum(ybar**2))
        - psi1 / (2 * kp * kp * psi2) * out["rv"]
    )
    out["rk"] = realized_kernel(r, rk_h)
    return out


# --------------------------------------------------------------------------- #
# Hand-computed session
# --------------------------------------------------------------------------- #
def test_a_hand_computed_session() -> None:
    r = [0.01, -0.02, 0.03, -0.01, 0.02]
    out = intraday_realized_measures(
        _one_session(r),
        entity="e",
        session="s",
        time="t",
        returns="r",
        subsample=2,
        min_obs=3,
    ).row(0, named=True)
    assert out["n_obs"] == 5
    assert out["rv"] == pytest.approx(1.9e-3, rel=1e-14)
    assert out["bv"] == pytest.approx(np.pi / 2 * 1.25 * 1.3e-3, rel=1e-14)
    # |r| = 1,2,3,1,2 (x1e-2): the three medians are all 2.
    assert out["medrv"] == pytest.approx(
        np.pi / (6 - 4 * np.sqrt(3) + np.pi) * (5 / 3) * 1.2e-3, rel=1e-14
    )
    # adjacent minima 1,2,1,1 -> sum of squares 7e-4.
    assert out["minrv"] == pytest.approx(np.pi / (np.pi - 2) * 1.25 * 7e-4, rel=1e-14)
    assert out["rs_pos"] == pytest.approx(1.4e-3, rel=1e-14)
    assert out["rs_neg"] == pytest.approx(5e-4, rel=1e-14)
    assert out["sjv"] == pytest.approx(9e-4, rel=1e-13)
    assert out["rq"] == pytest.approx(5 / 3 * 115e-8, rel=1e-14)
    # three triples, each |product| = 6e-6.
    assert out["tpq"] == pytest.approx(
        5 * _MU43**-3 * (5 / 3) * 3 * 6e-6 ** (4 / 3), rel=1e-13
    )
    # 2-step moves of p = 0, .01, -.01, .02, .01, .03: -.01, .01, .02, .01.
    assert out["rv_ss"] == pytest.approx(7e-4 / 2, rel=1e-14)


@pytest.mark.parametrize("input_kind", ["price", "returns"])
def test_every_measure_matches_a_numpy_reference(input_kind: str) -> None:
    df = _price_panel()
    kwargs: dict[str, object]
    if input_kind == "price":
        frame, kwargs = df, {"price": "px"}
    else:
        frame = (
            df.sort("e", "s", "t")
            .with_columns(ret=pl.col("px").log().diff().over("e", "s"))
            .drop_nulls("ret")
        )
        kwargs = {"returns": "ret"}
    out = intraday_realized_measures(
        frame, entity="e", session="s", time="t", measures=_ALL, tsrv_k=3, **kwargs
    )
    assert out.height == 6
    for row in out.iter_rows(named=True):
        px = df.filter((pl.col("e") == row["e"]) & (pl.col("s") == row["s"])).sort("t")[
            "px"
        ]
        r = np.diff(np.log(px.to_numpy()))
        ref = _reference(r, subsample=5, tsrv_k=3, rk_h=row["rk_h"])
        assert row["n_obs"] == r.size
        for name, want in ref.items():
            assert row[name] == pytest.approx(want, rel=1e-12, abs=1e-300), name


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #
def test_semivariances_add_up_to_rv() -> None:
    out = intraday_realized_measures(
        _price_panel(5),
        entity="e",
        session="s",
        time="t",
        price="px",
        measures=["rv", "rs_pos", "rs_neg", "sjv"],
    )
    rv, pos, neg = (out[c].to_numpy() for c in ("rv", "rs_pos", "rs_neg"))
    np.testing.assert_allclose(pos + neg, rv, rtol=1e-15)
    np.testing.assert_allclose(out["sjv"].to_numpy(), pos - neg, rtol=1e-15)


@pytest.mark.parametrize("k", [1, 2, 5, 7])
def test_rv_ss_is_the_mean_of_the_offset_subsampled_rvs(k: int) -> None:
    rng = np.random.default_rng(k)
    r = rng.standard_normal(123) * 1e-3
    p = np.r_[0.0, np.cumsum(r)]
    offsets = [np.sum(np.diff(p[j::k]) ** 2) for j in range(k)]
    out = intraday_realized_measures(
        _one_session(list(r)),
        entity="e",
        session="s",
        time="t",
        returns="r",
        measures=["rv_ss"],
        subsample=k,
    )
    assert out["rv_ss"][0] == pytest.approx(np.mean(offsets), rel=1e-13)


def test_jump_split_adds_up_and_flags_a_planted_jump() -> None:
    rng = np.random.default_rng(3)
    calm = rng.standard_normal(390) * 5e-4
    jumpy = calm.copy()
    jumpy[200] += 0.02
    df = pl.concat(
        [
            _one_session(list(calm)),
            _one_session(list(jumpy)).with_columns(s=pl.lit(1, dtype=pl.Int64)),
        ]
    )
    out = intraday_realized_measures(
        df,
        entity="e",
        session="s",
        time="t",
        returns="r",
        measures=["rv", "bv", "jump_z", "jump_sig", "cont"],
    ).sort("s")
    np.testing.assert_allclose(out["cont"] + out["jump_sig"], out["rv"], rtol=1e-15)
    assert out["jump_sig"][0] == 0.0
    assert out["jump_z"][1] > 3.09 > out["jump_z"][0]
    assert out["jump_sig"][1] == pytest.approx(out["rv"][1] - out["bv"][1], rel=1e-14)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def test_a_session_starts_from_its_own_first_price() -> None:
    """Overnight moves never leak in: the previous close is not a return."""
    df = _price_panel(2)
    base = intraday_realized_measures(df, entity="e", session="s", time="t", price="px")
    gapped = df.with_columns(
        pl.when(pl.col("s") == 1)
        .then(pl.col("px") * 1.37)
        .otherwise(pl.col("px"))
        .alias("px")
    )
    moved = intraday_realized_measures(
        gapped, entity="e", session="s", time="t", price="px"
    )
    # Rescaling a session's prices moves each log return by at most an ulp; the
    # (rv - bv) / rv cancellation in jump_z amplifies that to ~1e-11.
    for col in ("rv", "bv", "medrv", "tpq", "rq", "jump_z"):
        np.testing.assert_allclose(
            moved[col].to_numpy(), base[col].to_numpy(), rtol=1e-9
        )


def test_row_order_does_not_matter() -> None:
    df = _price_panel(4)
    base = intraday_realized_measures(
        df, entity="e", session="s", time="t", price="px", measures=_ALL
    )
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=0)
    other = intraday_realized_measures(
        shuffled, entity="e", session="s", time="t", price="px", measures=_ALL
    )
    assert other.equals(base)


def test_invalid_prices_are_dropped_and_short_sessions_are_null() -> None:
    df = pl.DataFrame(
        {
            "e": ["A"] * 14,
            "s": [0] * 12 + [1] * 2,
            "t": list(range(12)) + [0, 1],
            "px": [
                10.0,
                10.1,
                None,
                10.05,
                -1.0,
                10.2,
                float("nan"),
                10.1,
                10.15,
                10.2,
                10.25,
                10.3,
            ]
            + [9.0, 9.1],
        }
    )
    out = intraday_realized_measures(
        df, entity="e", session="s", time="t", price="px", min_obs=3
    ).sort("s")
    valid = np.array([10.0, 10.1, 10.05, 10.2, 10.1, 10.15, 10.2, 10.25, 10.3])
    assert out["n_obs"].to_list() == [8, 1]
    assert out["rv"][0] == pytest.approx(
        float(np.sum(np.diff(np.log(valid)) ** 2)), rel=1e-14
    )
    # One return is below min_obs: the session is present, its measures null.
    assert out.filter(pl.col("s") == 1)["rv"].is_null().all()


def test_a_session_without_returns_is_reported_with_zero_observations() -> None:
    df = pl.DataFrame(
        {
            "e": ["A", "A", "A", "B"],
            "s": [0, 0, 0, 0],
            "t": [0, 1, 2, 0],
            "px": [1.0, 1.1, 1.2, 5.0],
        }
    )
    out = intraday_realized_measures(
        df, entity="e", session="s", time="t", price="px", min_obs=3
    )
    assert out.select("e", "n_obs").to_dicts() == [
        {"e": "A", "n_obs": 2},
        {"e": "B", "n_obs": 0},
    ]
    assert out["rv"].is_null().all()


def test_float32_input_is_upcast() -> None:
    df = _price_panel(6)
    base = intraday_realized_measures(df, entity="e", session="s", time="t", price="px")
    f32 = intraday_realized_measures(
        df.with_columns(pl.col("px").cast(pl.Float32)),
        entity="e",
        session="s",
        time="t",
        price="px",
    )
    assert f32.schema["rv"] == pl.Float64
    np.testing.assert_allclose(f32["rv"].to_numpy(), base["rv"].to_numpy(), rtol=1e-3)


def test_stamps_sit_at_the_session_close() -> None:
    """Trap 10: open-stamped bars are shifted to their close."""
    start = dt.datetime(2024, 1, 2, 9, 30)
    stamps = [start + dt.timedelta(minutes=i) for i in range(20)]
    df = pl.DataFrame(
        {"e": ["A"] * 20, "s": [0] * 20, "t": stamps, "r": np.linspace(-1e-3, 1e-3, 20)}
    )
    right = intraday_realized_measures(
        df, entity="e", session="s", time="t", returns="r"
    )
    left = intraday_realized_measures(
        df, entity="e", session="s", time="t", returns="r", label="left"
    )
    assert right["t"][0] == stamps[-1]
    assert left["t"][0] == stamps[-1] + dt.timedelta(minutes=1)
    ints = df.with_columns(t=pl.int_range(0, 20 * 60, 60))
    left_int = intraday_realized_measures(
        ints, entity="e", session="s", time="t", returns="r", label="left"
    )
    assert left_int.schema["t"] == pl.Int64
    assert left_int["t"][0] == 19 * 60 + 60


def test_measures_come_out_in_the_requested_order_with_diagnostics() -> None:
    out = intraday_realized_measures(
        _price_panel(),
        entity="e",
        session="s",
        time="t",
        price="px",
        measures=["pav", "rv", "rk", "tsrv"],
    )
    assert out.columns == [
        "e",
        "s",
        "t",
        "pav",
        "pav_k",
        "rv",
        "rk",
        "rk_h",
        "rk_capped",
        "tsrv",
        "tsrv_k",
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({}, ValueError, "exactly one"),
        ({"returns": "r", "price": "r"}, ValueError, "exactly one"),
        ({"returns": "r", "measures": ["rv", "nope"]}, ValueError, "unknown"),
        ({"returns": "r", "measures": "rv"}, TypeError, "sequence"),
        ({"returns": "r", "measures": ["rv", "rv"]}, ValueError, "duplicate"),
        ({"returns": "r", "min_obs": 2}, ValueError, "min_obs"),
        ({"returns": "r", "jump_alpha": 1.0}, ValueError, "jump_alpha"),
        ({"returns": "r", "kernel_bandwidth": 99}, ValueError, "kernel_bandwidth"),
        ({"returns": "r", "tsrv_k": 1}, ValueError, "tsrv_k"),
        ({"returns": "r", "label": "centre"}, ValueError, "label"),
        ({"returns": "missing"}, ValueError, "not found"),
    ],
)
def test_argument_validation(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        intraday_realized_measures(
            _one_session([0.01, 0.02, 0.03]),
            entity="e",
            session="s",
            time="t",
            **kwargs,  # type: ignore[arg-type]
        )


def test_null_keys_are_rejected() -> None:
    df = _one_session([0.01, 0.02, 0.03]).with_columns(s=pl.lit(None, dtype=pl.Int64))
    with pytest.raises(ValueError, match="nulls"):
        intraday_realized_measures(df, entity="e", session="s", time="t", returns="r")


# --------------------------------------------------------------------------- #
# Noise-robust measures: settings
# --------------------------------------------------------------------------- #
def test_kernel_bandwidth_is_capped_and_flagged() -> None:
    rng = np.random.default_rng(9)
    efficient = np.cumsum(rng.standard_normal(2000) * 1e-4)
    noisy = np.diff(efficient + rng.standard_normal(2000) * 5e-4)
    df = _one_session(list(noisy))
    out = intraday_realized_measures(
        df,
        entity="e",
        session="s",
        time="t",
        returns="r",
        measures=["rk"],
        kernel_max_lags=5,
    )
    assert out["rk_capped"][0] is True
    assert out["rk_h"][0] == 5
    assert out["rk"][0] == pytest.approx(realized_kernel(noisy, 5), rel=1e-12)
    fixed = intraday_realized_measures(
        df,
        entity="e",
        session="s",
        time="t",
        returns="r",
        measures=["rk"],
        kernel_bandwidth=3,
    )
    assert fixed["rk_h"][0] == 3 and fixed["rk_capped"][0] is False
    assert fixed["rk"][0] == pytest.approx(realized_kernel(noisy, 3), rel=1e-12)


def test_bnhls_bandwidth_on_clean_one_minute_bars() -> None:
    """BNHLS read IV / (2 n_dense) as noise on clean 1-minute bars: H ~ 12."""
    rng = np.random.default_rng(10)
    r = rng.standard_normal(390) * 5e-4
    out = intraday_realized_measures(
        _one_session(list(r)),
        entity="e",
        session="s",
        time="t",
        returns="r",
        measures=["rk"],
    )
    assert 8 <= out["rk_h"][0] <= 16
    assert out["rk_capped"][0] is False


def test_tsrv_auto_k_is_small_without_noise_and_grows_with_it() -> None:
    rng = np.random.default_rng(11)
    efficient = np.cumsum(rng.standard_normal(5000) * 1e-4)
    rows = []
    for sess, noise in enumerate((0.0, 3e-4)):
        r = np.diff(efficient + rng.standard_normal(5000) * noise)
        rows.append(_one_session(list(r)).with_columns(s=pl.lit(sess, dtype=pl.Int64)))
    out = intraday_realized_measures(
        pl.concat(rows),
        entity="e",
        session="s",
        time="t",
        returns="r",
        measures=["tsrv"],
    ).sort("s")
    k_clean, k_noisy = out["tsrv_k"].to_list()
    assert 2 <= k_clean < k_noisy
