"""Cross-sectional and quantile labels: hand-computed panels, T13, T14, censoring."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.label._xsection import _trailing_thresholds
from panelary.registry import registry


def _panel(
    n_entities: int, n_times: int, seed: int = 0, *, dates: bool = False
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_entities):
        p = 50.0 * np.exp(np.cumsum(rng.standard_normal(n_times) * 0.03))
        t = (
            [date(2021, 1, 4) + timedelta(days=i) for i in range(n_times)]
            if dates
            else list(range(n_times))
        )
        frames.append(pl.DataFrame({"id": [f"e{k:02d}"] * n_times, "t": t, "close": p}))
    return pl.concat(frames)


def _fwd(df: pl.DataFrame, h: int) -> dict[tuple[str, int], float]:
    """Forward returns by explicit per-entity indexing (no polars shift)."""
    out: dict[tuple[str, int], float] = {}
    for (name,), sub in df.sort("id", "t").group_by(["id"], maintain_order=True):
        p = sub["close"].to_numpy()
        for i, t in enumerate(sub["t"].to_list()):
            if i + h < p.size:
                out[(name, t)] = p[i + h] / p[i] - 1.0
    return out


# --------------------------------------------------------------------------- #
# excess_over_median
# --------------------------------------------------------------------------- #
def test_excess_over_median_matches_hand_computation() -> None:
    df = _panel(7, 30, seed=1)
    h = 3
    out = pn.label.excess_over_median(df, horizon=h, min_count=5)
    fwd = _fwd(df, h)
    for t in range(30 - h):
        vals = [fwd[(f"e{k:02d}", t)] for k in range(7)]
        med = float(np.median(vals))
        got = out.filter(pl.col("t") == t).sort("id")["label"].to_numpy()
        np.testing.assert_allclose(got, np.array(vals) - med, rtol=1e-12, atol=1e-15)


def test_binary_is_the_sign_and_min_count_nulls_thin_dates() -> None:
    df = _panel(4, 20, seed=2)
    cont = pn.label.excess_over_median(df, horizon=2, min_count=4)
    binary = pn.label.excess_over_median(df, horizon=2, min_count=4, binary=True)
    assert binary.schema["label"] == pl.Int64
    ok = cont["label"].is_not_null()
    assert (binary.filter(ok)["label"] == cont.filter(ok)["label"].sign()).all()
    thin = pn.label.excess_over_median(df, horizon=2, min_count=5)
    assert thin["label"].null_count() == thin.height


def test_censored_tail_has_null_label_and_t1() -> None:
    df = _panel(5, 15, seed=3, dates=True)
    out = pn.label.excess_over_median(df, horizon=4, min_count=2)
    assert out.schema["t1"] == pl.Date
    tail = out.filter(pl.col("censored"))
    assert tail.height == 5 * 4
    assert (
        tail["label"].null_count() == tail.height
        and tail["t1"].null_count() == tail.height
    )
    body = out.filter(~pl.col("censored"))
    assert ((body["t1"] - body["t"]).dt.total_days() == 4).all()


def test_t14_survivorship_median_is_over_survivors_only() -> None:
    """An entity that delists inside the horizon is excluded from the median."""
    df = _panel(5, 20, seed=4).filter(~((pl.col("id") == "e00") & (pl.col("t") > 10)))
    out = pn.label.excess_over_median(df, horizon=3, min_count=2)
    fwd = _fwd(df, 3)
    t = 9  # e00 ends at 10: no forward return at 9, so four survivors remain
    assert out.filter((pl.col("id") == "e00") & (pl.col("t") == t))["censored"].item()
    vals = [fwd[(f"e{k:02d}", t)] for k in range(1, 5)]
    got = out.filter((pl.col("t") == t) & (pl.col("id") != "e00")).sort("id")["label"]
    np.testing.assert_allclose(got.to_numpy(), np.array(vals) - np.median(vals))


def test_gap_guard_is_on_by_default() -> None:
    df = _panel(3, 12, seed=5).filter(pl.col("t") != 6)
    with pytest.raises(ValueError, match="irregular"):
        pn.label.excess_over_median(df, horizon=2, min_count=2)
    out = pn.label.excess_over_median(df, horizon=2, min_count=2, allow_gaps=True)
    assert out.height == df.height


def test_t1_is_the_latest_constituent_end_on_a_gapped_grid() -> None:
    """With gaps, a date's constituents resolve at different times: take the max."""
    df = _panel(3, 12, seed=6).filter(~((pl.col("id") == "e01") & (pl.col("t") == 5)))
    out = pn.label.excess_over_median(df, horizon=2, min_count=2, allow_gaps=True)
    # At t=3, e01 resolves at row t+2 = time 6 (it skips 5); the others at 5.
    at3 = out.filter(pl.col("t") == 3)
    assert (at3["t1"] == 6).all()


@pytest.mark.parametrize("label_fn", ["excess_over_median", "quantile_label"])
def test_resolved_prefix_invariance(label_fn: str) -> None:
    df = _panel(6, 40, seed=7)
    fn = getattr(pn.label, label_fn)
    kw = (
        {"horizon": 5, "min_count": 3}
        if label_fn == "excess_over_median"
        else {"horizon": 5}
    )
    full = fn(df, **kw)
    for tau in (12, 25, 33):
        pref = fn(df.filter(pl.col("t") <= tau), **kw)
        resolved = pref.filter(~pl.col("censored") & (pl.col("t1") <= tau))
        theirs = full.join(resolved.select("id", "t"), on=["id", "t"]).sort("id", "t")
        assert resolved.height > 0
        cols = ["fwd_ret", "label", "t1"]
        assert resolved.sort("id", "t").select(cols).equals(theirs.select(cols))


# --------------------------------------------------------------------------- #
# quantile_label
# --------------------------------------------------------------------------- #
def test_cross_section_buckets_match_hand_computation() -> None:
    df = _panel(9, 25, seed=8)
    h = 2
    out = pn.label.quantile_label(df, horizon=h, q=3)
    fwd = _fwd(df, h)
    for t in range(25 - h):
        vals = np.array([fwd[(f"e{k:02d}", t)] for k in range(9)])
        order = vals.argsort().argsort()  # 0-based rank, no ties in continuous data
        want = np.floor(order / 9 * 3).astype(int) - 1
        got = out.filter(pl.col("t") == t).sort("id")["label"].to_numpy()
        np.testing.assert_array_equal(got, want)


def test_even_q_has_no_zero_bucket_and_thin_dates_are_null() -> None:
    df = _panel(8, 12, seed=9)
    out = pn.label.quantile_label(df, horizon=1, q=4)
    got = set(out["label"].drop_nulls().to_list())
    assert got == {-2, -1, 1, 2}
    thin = pn.label.quantile_label(_panel(2, 12, seed=9), horizon=1, q=3)
    assert thin["label"].null_count() == thin.height


def test_trailing_thresholds_match_brute_force() -> None:
    df = _panel(2, 60, seed=10)
    h, q, lb = 3, 3, 10
    out = pn.label.quantile_label(df, horizon=h, q=q, mode="trailing", lookback=lb)
    for (name,), sub in out.sort("id", "t").group_by(["id"], maintain_order=True):
        fwd = sub["fwd_ret"].to_numpy()
        labels = sub["label"].to_list()
        for i in range(sub.height):
            past = (
                fwd[max(0, i - h - lb + 1) : i - h + 1] if i - h >= 0 else np.array([])
            )
            if past.size < lb or np.isnan(fwd[i]) or sub["censored"][i]:
                assert labels[i] is None, (name, i)
                continue
            edges = np.quantile(past, [1 / 3, 2 / 3])  # linear interpolation
            want = int((fwd[i] > edges).sum()) - 1
            assert labels[i] == want, (name, i)


def test_t13_trailing_thresholds_read_only_resolved_returns() -> None:
    """Perturb prices after t: thresholds at t are unchanged; a naive full-sample
    threshold moves (the check has power)."""
    df = _panel(1, 80, seed=11)
    h, lb, cut = 4, 15, 50
    frame = pn.factor.forward_return(
        df, entity="id", time="t", price="close", horizon=h
    )
    edges = _trailing_thresholds(
        pl.col("fwd_ret"), entity="id", horizon=h, q=3, lookback=lb
    )
    base = frame.select(e.alias(f"e{k}") for k, e in enumerate(edges))

    rng = np.random.default_rng(0)
    p = df["close"].to_numpy().copy()
    p[cut + 1 :] *= np.exp(rng.standard_normal(p.size - cut - 1))
    pert_frame = pn.factor.forward_return(
        df.with_columns(pl.Series("close", p)),
        entity="id",
        time="t",
        price="close",
        horizon=h,
    )
    pert = pert_frame.select(e.alias(f"e{k}") for k, e in enumerate(edges))
    assert base.head(cut + 1).equals(pert.head(cut + 1))

    naive = pl.col("fwd_ret").quantile(1 / 3).over("id")  # full-sample distribution
    assert (
        not frame.select(naive)
        .head(cut + 1)
        .equals(pert_frame.select(naive).head(cut + 1))
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"q": 1}, "q"),
        ({"mode": "global"}, "mode"),
        ({"lookback": 1, "mode": "trailing"}, "lookback"),
        ({"horizon": 0}, "horizon"),
    ],
)
def test_quantile_label_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pn.label.quantile_label(_panel(3, 10), **kwargs)


def test_excess_over_median_validation() -> None:
    with pytest.raises(ValueError, match="min_count"):
        pn.label.excess_over_median(_panel(3, 10), min_count=0)


def test_registry_scopes() -> None:
    for name in ("excess_over_median", "quantile_label"):
        spec = registry.get(name)
        assert (spec.namespace, spec.safe_scope, spec.panel_safe) == (
            "label",
            "window",
            False,
        )
