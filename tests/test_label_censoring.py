"""Label censoring (B1 / F1), `vol=` (F7) and fixed_horizon via forward_return (B2 / F2).

The contract for forward-looking labels is *resolved-prefix invariance*: a row
whose label is resolved (not censored, ``t1 <= tau``) on data up to ``tau`` gets
the same label on the full data, bitwise. Censored rows are exactly the ones
that may still change.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from panelary.factor import forward_return
from panelary.label import fixed_horizon, triple_barrier
from panelary.label._barriers import _trailing_std
from tests import _spans_reference as ref


def _prices(seed: int, n_ent: int = 3, n_t: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_ent):
        n = n_t - int(rng.integers(0, 10))
        frames.append(
            pl.DataFrame(
                {
                    "id": [f"e{e}"] * n,
                    "date": [
                        dt.date(2021, 1, 1) + dt.timedelta(days=i) for i in range(n)
                    ],
                    "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
                }
            )
        )
    return pl.concat(frames)


KW = {
    "entity": "id",
    "time": "date",
    "pt": 1.0,
    "sl": 1.0,
    "max_holding": 7,
    "vol_lookback": 6,
}


def test_censored_flag_definition() -> None:
    df = _prices(0)
    out = triple_barrier(df, **KW)
    assert out.schema["censored"] == pl.Boolean
    assert out.columns[-4:] == ["label", "ret", "t1", "censored"]
    for (_e,), sub in out.group_by(["id"], maintain_order=True):
        n = sub.height
        pos = np.arange(n)
        label = sub.get_column("label").to_numpy()
        want = (label == 0) & (pos + KW["max_holding"] > n - 1)
        assert np.array_equal(sub.get_column("censored").to_numpy(), want)
        # the last row is always censored, and it is the "label = 0, t1 = t" row
        assert sub.get_column("censored")[-1]
        assert sub.get_column("t1")[-1] == sub.get_column("date")[-1]


@pytest.mark.parametrize("seed", range(4))
def test_triple_barrier_resolved_prefix_invariance(seed: int) -> None:
    df = _prices(seed)
    full = triple_barrier(df, **KW).sort(["id", "date"])
    cut_dates = sorted(df.get_column("date").unique().to_list())
    n_changed_censored = 0
    for tau in cut_dates[20::9]:
        part = triple_barrier(df.filter(pl.col("date") <= tau), **KW)
        joined = part.join(full, on=["id", "date"], suffix="_full")
        resolved = joined.filter(~pl.col("censored"))
        for col in ("label", "ret", "t1"):
            assert resolved.get_column(col).equals(resolved.get_column(f"{col}_full"))
        # Power (T8): the rows the flag marks are the ones that can change.
        cens = joined.filter(pl.col("censored"))
        n_changed_censored += cens.filter(
            (pl.col("t1") != pl.col("t1_full"))
            | (pl.col("label") != pl.col("label_full"))
        ).height
    assert n_changed_censored > 0


def test_existing_outputs_unchanged_by_the_flag() -> None:
    """label / ret / t1 are what they were before `censored` existed."""
    df = _prices(5, n_ent=1)
    out = triple_barrier(df, **KW)
    prices = df.get_column("close").to_numpy()
    n = prices.size
    rets = np.r_[np.nan, prices[1:] / prices[:-1] - 1.0]
    sigma = _trailing_std(rets, KW["vol_lookback"])
    for i in range(n):
        end = min(i + KW["max_holding"], n - 1)
        touch, lab = end, 0
        if not np.isnan(sigma[i]) and sigma[i] > 0:
            up, dn = prices[i] * (1 + sigma[i]), prices[i] * (1 - sigma[i])
            for j in range(i + 1, end + 1):
                if prices[j] >= up:
                    touch, lab = j, 1
                    break
                if prices[j] <= dn:
                    touch, lab = j, -1
                    break
        assert out["label"][i] == lab
        assert out["t1"][i] == out["date"][touch]


def test_vol_column_reproduces_default_and_changes_labels() -> None:
    df = _prices(2)
    default = triple_barrier(df, **KW)
    sig = []
    for (_e,), sub in df.sort(["id", "date"]).group_by(["id"], maintain_order=True):
        p = sub.get_column("close").to_numpy()
        sig.append(
            _trailing_std(np.r_[np.nan, p[1:] / p[:-1] - 1.0], KW["vol_lookback"])
        )
    with_vol = df.sort(["id", "date"]).with_columns(
        pl.Series("sigma", np.concatenate(sig)).fill_nan(None)
    )
    via_col = triple_barrier(with_vol, vol="sigma", **KW)
    assert via_col.drop("sigma").equals(default)
    # A much wider causal sigma removes horizontal touches.
    wide = triple_barrier(
        with_vol.with_columns(pl.col("sigma") * 50), vol="sigma", **KW
    )
    assert (wide.get_column("label") != 0).sum() < (
        default.get_column("label") != 0
    ).sum()


def test_vol_nonpositive_or_null_disables_horizontal_barriers() -> None:
    df = _prices(1, n_ent=1).with_columns(pl.lit(None, dtype=pl.Float64).alias("v"))
    out = triple_barrier(df, vol="v", **KW)
    assert (out.get_column("label") == 0).all()


# --------------------------------------------------------------------------- #
# fixed_horizon through forward_return (B2 / F2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("threshold", [None, 0.01])
@pytest.mark.parametrize("horizon", [1, 5])
def test_fixed_horizon_byte_identical_on_regular_grid(threshold, horizon) -> None:
    df = _prices(3).sample(fraction=1.0, shuffle=True, seed=0)
    got = fixed_horizon(
        df, entity="id", time="date", horizon=horizon, threshold=threshold
    )
    want = ref.frozen_fixed_horizon(
        df,
        entity="id",
        time="date",
        price="close",
        horizon=horizon,
        threshold=threshold,
    )
    assert got.equals(want)
    assert got.columns == want.columns


def test_fixed_horizon_gap_guard() -> None:
    df = _prices(4).filter(pl.col("date") != dt.date(2021, 1, 10))  # a missing day
    with pytest.raises(ValueError, match="irregular"):
        fixed_horizon(df, entity="id", time="date", horizon=3, allow_gaps=False)
    got = fixed_horizon(df, entity="id", time="date", horizon=3, allow_gaps=True)
    want = ref.frozen_fixed_horizon(
        df, entity="id", time="date", price="close", horizon=3, threshold=None
    )
    assert got.equals(want)
    # The default warns but keeps the pre-0.6 output, so business-day panels
    # (weekend gaps) do not start raising.
    with pytest.warns(UserWarning, match="irregular"):
        default = fixed_horizon(df, entity="id", time="date", horizon=3)
    assert default.equals(want)


def test_fixed_horizon_resolved_prefix_invariance() -> None:
    df = _prices(6)
    full = fixed_horizon(df, entity="id", time="date", horizon=4)
    for tau in (dt.date(2021, 1, 20), dt.date(2021, 2, 15)):
        part = fixed_horizon(
            df.filter(pl.col("date") <= tau), entity="id", time="date", horizon=4
        )
        joined = part.drop_nulls("t1").join(full, on=["id", "date"], suffix="_full")
        for col in ("label", "ret", "t1"):
            assert joined.get_column(col).equals(joined.get_column(f"{col}_full"))


def test_forward_return_end_time_is_additive() -> None:
    df = _prices(7)
    base = forward_return(df, entity="id", time="date", price="close", horizon=3)
    with_t1 = forward_return(
        df, entity="id", time="date", price="close", horizon=3, end_time="t1"
    )
    assert with_t1.drop("t1").equals(base)
    assert with_t1.schema["t1"] == with_t1.schema["date"]
    expected = with_t1.select(pl.col("date").shift(-3).over("id")).to_series()
    assert with_t1.get_column("t1").equals(expected.alias("t1"))
    assert with_t1.filter(pl.col("fwd_ret").is_null()).get_column(
        "t1"
    ).null_count() == (with_t1.get_column("fwd_ret").null_count())
