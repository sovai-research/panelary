"""One test per named leak trap (plan section 6).

Each test builds the leaky variant inline and shows that the instrument
catches it, next to the shipped path passing the same instrument -- so the
instruments are proven sharp, not just green.

T7 / T8 (online co-moment recompute anchoring, EWMA initialisation) belong to
``OnlineCovariance`` and the EWMA estimators, which ship in milestone M5.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core.panel_frame import PanelFrame
from panelary.covariance._estimate import estimate
from panelary.covariance._gram import window_stats
from panelary.covariance._rmt import mp_fit
from panelary.covariance._state import market_loading, market_state
from panelary.covariance._window import rolling
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

W = 20
COV = 0.9


def _frame(seed: int = 5) -> pl.DataFrame:
    return pn.synth.generate_panel(
        seed=seed,
        n_entities=26,
        n_periods=140,
        start=dt.date(2022, 1, 3),
        every="1d",
        missing_rate=0.0,
        entry_rate=0.03,
        exit_rate=0.01,
        initial_fraction=0.6,
    ).panel


DF = _frame()
PANEL = PanelFrame(DF, entity="entity", time="time")
TIMES = PANEL.time_index().to_list()


def _broadcast(frame, state: pl.DataFrame, on: str = "time") -> pl.DataFrame:
    df = frame.collect() if isinstance(frame, PanelFrame) else frame
    return df.join(state, on=on, how="left")


# --------------------------------------------------------------------------- #
# T3: Delta-AR z-scored with the full-sample mean and sd
# --------------------------------------------------------------------------- #
def test_T3_full_sample_zscore_of_ar_is_caught() -> None:
    def leaky(frame):
        st = market_state(
            frame, returns="value", window=W, min_coverage=COV,
            features=("absorption_ratio",),
        )  # fmt: skip
        ar = pl.col("absorption_ratio")
        st = st.select("time", ((ar - ar.mean()) / ar.std()).alias("ar_z"))
        return _broadcast(frame, st)

    def ours(frame):
        return market_state(
            frame, returns="value", window=W, min_coverage=COV,
            features=("ar_shift",), ar_short=3, ar_long=15, broadcast=True,
        )  # fmt: skip

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky, PANEL, tol=0.0)
    assert_no_lookahead(ours, PANEL, tol=0.0)
    assert_prefix_invariant(ours, PANEL, tol=0.0)
    assert ours(PANEL).get_column("ar_shift").drop_nulls().len() > 100


# --------------------------------------------------------------------------- #
# T4: parameters sized by the panel (k = N/5 with the panel's N)
# --------------------------------------------------------------------------- #
def test_T4_panel_sized_parameters_are_caught() -> None:
    def leaky(frame):
        n_total = frame.collect().get_column("entity").n_unique()
        return market_state(
            frame, returns="value", window=W, min_coverage=COV,
            features=("absorption_ratio",), ar_k=math.ceil(0.2 * n_total),
            broadcast=True,
        )  # fmt: skip

    def ours(frame):
        return market_state(
            frame, returns="value", window=W, min_coverage=COV,
            features=("absorption_ratio",), broadcast=True,
        )  # fmt: skip

    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky, PANEL, tol=0.0)
    assert_prefix_invariant(ours, PANEL, tol=0.0)
    out = ours(PANEL).drop_nulls("absorption_ratio")
    # k follows the as-of universe, not the panel
    assert out.get_column("ar_k").n_unique() > 1


# --------------------------------------------------------------------------- #
# T5: survivorship and forward-filled delisted returns
# --------------------------------------------------------------------------- #
def _state(df: pl.DataFrame) -> pl.DataFrame:
    return market_state(
        df, returns="value", window=W, min_coverage=COV, entity="entity", time="time",
        features=("absorption_ratio", "avg_corr", "participation_ratio"),
    )  # fmt: skip


def _exiting_entity() -> tuple[str, dt.date]:
    last = DF.group_by("entity").agg(pl.col("time").max().alias("last"))
    mid = TIMES[len(TIMES) // 2]
    row = last.filter(pl.col("last") < TIMES[-W]).sort("last").row(-1)
    assert row[1] > TIMES[W], "the fixture needs an entity that exits mid-sample"
    del mid
    return row[0], row[1]


def test_T5_entities_after_exit_and_before_listing_do_not_move_values() -> None:
    base = _state(DF)
    # (a) entities that list after t0 do not affect anything at t <= t0
    first = DF.group_by("entity").agg(pl.col("time").min().alias("first"))
    listings = first.filter(pl.col("first") > TIMES[W]).get_column("first").sort()
    assert listings.len() > 0, "the fixture needs entities that list mid-sample"
    t0 = TIMES[TIMES.index(listings[0]) - 1]
    late = first.filter(pl.col("first") > t0).get_column("entity")
    no_late = _state(DF.filter(~pl.col("entity").is_in(late.implode())))
    a = base.filter(pl.col("time") <= t0)
    b = no_late.filter(pl.col("time") <= t0)
    assert a.equals(b)
    # (b) an entity leaves the universe right after its last observation
    ent, last = _exiting_entity()
    without = _state(DF.filter(pl.col("entity") != ent))
    assert base.filter(pl.col("time") > last).equals(
        without.filter(pl.col("time") > last)
    )


def test_T5_forward_filled_delisted_returns_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import panelary.shape._tensor as tensor

    real = tensor.build_tensor

    def leaky_build(panel, cols, **kw):  # noqa: ANN001, ANN202 - test double
        kw["forward_fill"] = True  # the build_tensor default: survivorship leak
        return real(panel, cols, **kw)

    ent, last = _exiting_entity()
    monkeypatch.setattr(tensor, "build_tensor", leaky_build)
    base = _state(DF)
    without = _state(DF.filter(pl.col("entity") != ent))
    after = pl.col("time") > last
    assert not base.filter(after).equals(without.filter(after))


# --------------------------------------------------------------------------- #
# T6: end-anchored / "last trading day" grids
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["last", "end"])
def test_T6_end_anchored_schedules_raise(bad: str) -> None:
    with pytest.raises(ValueError, match="T6"):
        market_state(PANEL, returns="value", window=W, schedule=bad)
    with pytest.raises(ValueError, match="T6"):
        rolling(PANEL, returns="value", window=W, schedule=bad)


def test_T6_a_last_of_month_grid_is_caught_by_the_prefix_instrument() -> None:
    def leaky(frame):
        df = frame.collect()
        month = pl.col("time").dt.truncate("1mo")
        last = (
            df.select(
                (month != month.shift(-1)).fill_null(value=True).alias("m"), "time"
            )
            .unique("time")
            .sort("time")
        )
        positions = np.flatnonzero(last.get_column("m").to_numpy())
        st = market_state(
            frame, returns="value", window=W, min_coverage=COV,
            features=("absorption_ratio",),
        )  # fmt: skip
        keep = st.with_row_index("i").filter(pl.col("i").is_in(positions.tolist()))
        filled = st.select("time").join(
            keep.select("time", "absorption_ratio"), on="time", how="left"
        )
        return _broadcast(
            frame, filled.select("time", pl.col("absorption_ratio").forward_fill())
        )

    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky, PANEL, tol=0.0)


# --------------------------------------------------------------------------- #
# T9: intensity / sigma^2 / k fitted on the full sample and reused per date
# --------------------------------------------------------------------------- #
def _hand_window(t_index: int) -> tuple[np.ndarray, list[str]]:
    wide = DF.pivot(index="time", on="entity", values="value").sort("time")
    ents = sorted(c for c in wide.columns if c != "time")
    R = wide.select(ents).to_numpy()
    win = R[t_index + 1 - W : t_index + 1]
    cnt = np.isfinite(win).sum(axis=0)
    ok = np.isfinite(R[t_index]) & (cnt >= math.ceil(COV * W))
    return np.ascontiguousarray(win[:, ok]), [
        e for e, k in zip(ents, ok, strict=True) if k
    ]


@pytest.mark.parametrize("method", ["lw_identity", "oas", "qis", "mp_clip"])
def test_T9_per_date_parameters_equal_the_window_estimate_bitwise(method: str) -> None:
    series = rolling(PANEL, returns="value", window=W, min_coverage=COV, method=method)
    for ti in (W + 5, len(TIMES) // 2, len(TIMES) - 1):
        X, ents = _hand_window(ti)
        ref = estimate(X, method=method, entities=tuple(ents))
        got = series.at(TIMES[ti])
        assert got.entities == ref.entities
        assert got.shrinkage == ref.shrinkage
        assert got.info.get("sigma2") == ref.info.get("sigma2")
        assert np.array_equal(got.to_dense(), ref.to_dense())


def test_T9_market_state_mp_noise_is_window_local() -> None:
    st = market_state(
        PANEL, returns="value", window=W, min_coverage=COV,
        features=("mp_signal_count", "mp_sigma2"),
    )  # fmt: skip
    ti = len(TIMES) - 3
    X, _ = _hand_window(ti)
    ws = window_stats(X)
    fit = mp_fit(ws.spectrum().values, ws.p, ws.n_eff)
    row = st.filter(pl.col("time") == TIMES[ti])
    assert row.item(0, "mp_sigma2") == fit["sigma2"]
    assert row.item(0, "mp_signal_count") == fit["n_signal"]


def test_T9_a_full_sample_intensity_is_caught() -> None:
    def leaky(frame):
        df = frame.collect()
        wide = df.pivot(index="time", on="entity", values="value").sort("time")
        R = wide.drop("time").to_numpy()
        R = R[:, np.isfinite(R).all(axis=0)] if np.isfinite(R).all(axis=0).any() else R
        full = window_stats(np.nan_to_num(R))  # the whole sample, once
        from panelary.covariance._linear import lw_identity_intensity

        rho, _ = lw_identity_intensity(full)
        st = wide.select("time").with_columns(pl.lit(rho).alias("rho"))
        return _broadcast(frame, st)

    def ours(frame):
        series = rolling(
            frame, returns="value", window=W, min_coverage=COV, method="lw_identity"
        )
        rows = []
        for d, est in series:
            rows.append((d, None if est is None else est.shrinkage))
        st = pl.DataFrame(
            rows, schema={"time": pl.Date, "rho": pl.Float64}, orient="row"
        )
        return _broadcast(frame, st)

    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky, PANEL, tol=0.0)
    assert_prefix_invariant(ours, PANEL, tol=0.0)


# --------------------------------------------------------------------------- #
# T13: eigenvector sign / order flips
# --------------------------------------------------------------------------- #
def test_T13_market_loading_is_deterministic_and_sign_continuous() -> None:
    # add a market factor (positive loadings) so the top mode is a market mode
    m = pl.DataFrame(
        {"time": TIMES, "mkt": np.random.default_rng(9).standard_normal(len(TIMES)) * 2}
    )
    df = DF.join(m, on="time").with_columns(pl.col("value") + pl.col("mkt")).drop("mkt")
    panel = PanelFrame(df, entity="entity", time="time")
    a = market_loading(panel, returns="value", window=W, min_coverage=COV)
    b = market_loading(panel, returns="value", window=W, min_coverage=COV)
    assert a.equals(b)
    wide = (
        a.filter(pl.col("asof_date") == pl.col("time"))
        .pivot(index="time", on="entity", values="market_loading")
        .sort("time")
    )
    L = wide.drop("time").to_numpy()
    flips = 0
    pairs = 0
    for t in range(1, L.shape[0]):
        both = np.isfinite(L[t]) & np.isfinite(L[t - 1])
        if both.sum() >= 5:
            pairs += 1
            flips += int(L[t, both] @ L[t - 1, both] < 0)
    assert pairs > 50
    assert flips == 0
    # "market up" is the positive direction on every date
    sums = np.nansum(L, axis=1)
    assert (sums[np.isfinite(L).any(axis=1)] > 0).all()


# --------------------------------------------------------------------------- #
# T12: group labels from the latest classification
# --------------------------------------------------------------------------- #
def _with_sector(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("entity").str.slice(-1).cast(pl.Int64) % 2)
        .cast(pl.String)
        .alias("sector")
    )


def _grouped(df: pl.DataFrame) -> pl.DataFrame:
    return market_state(
        df, returns="value", window=W, min_coverage=COV, group="sector",
        min_entities=3, features=("absorption_ratio", "avg_corr"),
        entity="entity", time="time",
    )  # fmt: skip


def test_T12_a_future_reclassification_leaves_the_past_unchanged() -> None:
    df = _with_sector(DF)
    t0 = TIMES[len(TIMES) // 2]
    movers = df.filter(pl.col("time") > t0).get_column("entity").unique().head(4)
    moved = df.with_columns(
        pl.when(pl.col("entity").is_in(movers.implode()) & (pl.col("time") > t0))
        .then(pl.lit("9"))
        .otherwise(pl.col("sector"))
        .alias("sector")
    )
    base, new = _grouped(df), _grouped(moved)
    past = pl.col("time") <= t0
    assert base.filter(past).equals(new.filter(past))
    assert not base.equals(new)


def test_T12_latest_classification_snapshot_is_caught() -> None:
    df = _with_sector(DF)
    t0 = TIMES[len(TIMES) // 2]
    movers = df.filter(pl.col("time") > t0).get_column("entity").unique().head(4)
    moved = df.with_columns(
        pl.when(pl.col("entity").is_in(movers.implode()) & (pl.col("time") > t0))
        .then(pl.lit("9"))
        .otherwise(pl.col("sector"))
        .alias("sector")
    )

    def snapshot(frame: pl.DataFrame) -> pl.DataFrame:
        # the leak: every row gets the entity's LAST known sector
        latest = frame.sort("time").group_by("entity").agg(pl.col("sector").last())
        return _grouped(frame.drop("sector").join(latest, on="entity"))

    past = pl.col("time") <= t0
    assert not snapshot(df).filter(past).equals(snapshot(moved).filter(past))
