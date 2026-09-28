"""Tests for ``panelary.clean.OutlierCleaner`` (M4) and its robust helpers.

The leak-safety contract, cashed out:

1. **Fit on train only.** Fitted bounds equal statistics of the training rows
   and differ from a full-sample fit on a drifting panel.
2. **Transform never refits.** Test output equals the frozen training bounds
   applied to the test rows; a deliberately leaky subclass that refits at
   transform time fails that check.
3. **Train outputs cannot see the test fold.** Fitting on train and
   transforming the whole panel passes ``assert_no_train_test_leak``; the leaky
   variant that fits on all rows fails it.
4. **Causal rolling filters.** Hampel / rolling pass ``assert_no_lookahead``
   and ``assert_prefix_invariant``; a variant reading future windows fails.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.clean import OutlierCleaner
from panelary.clean._outliers import _hi, _lo
from panelary.clean._robust import trailing_median_sigma
from panelary.core.protocol import PanelTransformer
from panelary.testing import (
    assert_no_lookahead,
    assert_no_train_test_leak,
    assert_prefix_invariant,
)


def _panel(n_entities: int = 3, n_periods: int = 80, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    drift = np.tile(np.linspace(0.0, 4.0, n_periods), n_entities)
    x = rng.standard_normal(n) + drift
    y = rng.standard_normal(n) * 2.0
    # Spikes every 17th row.
    spikes = np.arange(n) % 17 == 16
    x = np.where(spikes, x + 25.0, x)
    return pl.DataFrame(
        {
            "id": np.repeat([f"e{i}" for i in range(n_entities)], n_periods),
            "t": np.tile(np.arange(n_periods), n_entities),
            "x": x,
            "y": y,
        }
    )


def _split(df: pl.DataFrame, cut: int = 50):
    return df.filter(pl.col("t") < cut), df.filter(pl.col("t") >= cut)


def _mad_bounds(v: np.ndarray, k: float) -> tuple[float, float]:
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    s = 1.4826 * mad if mad > 0 else 1.2533 * np.mean(np.abs(v - v.mean()))
    return med - k * s, med + k * s


# --------------------------------------------------------------------------- #
# Contract declarations
# --------------------------------------------------------------------------- #
def test_declares_contract():
    assert issubclass(OutlierCleaner, PanelTransformer)
    oc = OutlierCleaner()
    assert oc.panel_safe is True and oc.leakage_safe is True
    assert OutlierCleaner(pooling="cross_section").panel_safe is False
    assert OutlierCleaner(method="hampel", pooling="cross_section").panel_safe is True


def test_validation_errors():
    with pytest.raises(ValueError):
        OutlierCleaner(method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OutlierCleaner(quantiles=(0.9, 0.1))
    with pytest.raises(ValueError):
        OutlierCleaner(k=-1)
    with pytest.raises(ValueError):
        OutlierCleaner(window=5, min_periods=6)
    df = pl.DataFrame({"id": ["a"], "t": [1], "s": ["x"]})
    with pytest.raises(ValueError, match="numeric"):
        OutlierCleaner(columns="s", entity="id", time="t").fit(df)


# --------------------------------------------------------------------------- #
# 1. Fit on train only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["mad", "iqr", "zscore", "quantile"])
def test_bounds_are_train_statistics(method):
    df = _panel()
    train, _test = _split(df)
    oc = OutlierCleaner(method=method, columns=["x"], entity="id", time="t").fit(train)
    full = OutlierCleaner(method=method, columns=["x"], entity="id", time="t").fit(df)
    for e in ["e0", "e1", "e2"]:
        v = train.filter(pl.col("id") == e)["x"].to_numpy()
        if method == "mad":
            lo, hi = _mad_bounds(v, 3.5)
        elif method == "iqr":
            q1, q3 = np.quantile(v, [0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        elif method == "zscore":
            lo, hi = v.mean() - 3 * v.std(ddof=1), v.mean() + 3 * v.std(ddof=1)
        else:
            lo, hi = np.quantile(v, [0.01, 0.99])
        row = oc.bounds_.filter(pl.col("id") == e).row(0, named=True)
        assert row[_lo("x")] == pytest.approx(lo)
        assert row[_hi("x")] == pytest.approx(hi)
    # A full-sample fit on the drifting panel learns different bounds.
    assert not np.allclose(
        oc.bounds_[_hi("x")].to_numpy(), full.bounds_[_hi("x")].to_numpy()
    )
    pooled = train["x"].to_numpy()
    if method == "mad":
        assert oc.global_bounds_["x"] == pytest.approx(_mad_bounds(pooled, 3.5))


def test_thin_and_unseen_entities_fall_back_to_pooled_bounds():
    df = _panel()
    train, test = _split(df)
    # e2 has only 5 training rows (< min_obs=10).
    train = train.filter((pl.col("id") != "e2") | (pl.col("t") < 5))
    test = pl.concat(
        [test, test.filter(pl.col("id") == "e0").with_columns(id=pl.lit("new"))]
    )
    oc = OutlierCleaner(columns=["x"], entity="id", time="t", action="flag").fit(train)
    row = oc.bounds_.filter(pl.col("id") == "e2").row(0, named=True)
    assert row[_lo("x")] is None and row[_hi("x")] is None
    out = oc._with_bounds(oc._as_panel(test, method="t")).collect()
    glo, ghi = oc.global_bounds_["x"]
    for e in ["e2", "new"]:
        sub = out.filter(pl.col("id") == e)
        assert (sub[_lo("x")] == glo).all() and (sub[_hi("x")] == ghi).all()


# --------------------------------------------------------------------------- #
# 2. Transform never refits (and a leaky subclass that does is caught)
# --------------------------------------------------------------------------- #
class _RefitsAtTransform(OutlierCleaner):
    """Deliberately leaky: re-learns its bounds from the transform-time rows."""

    def _transform(self, panel):
        self._fit(panel)
        return super()._transform(panel)


def _check_frozen(cls, train, test):
    """Test output must equal the *training* bounds applied to the test rows."""
    kw = {"columns": ["x", "y"], "entity": "id", "time": "t"}
    got = cls(**kw).fit(train).transform(test).collect()
    ref = OutlierCleaner(**kw).fit(train)
    frame = ref._with_bounds(ref._as_panel(test, method="t")).collect()
    for c in ["x", "y"]:
        want = np.clip(
            frame[c].to_numpy(), frame[_lo(c)].to_numpy(), frame[_hi(c)].to_numpy()
        )
        if not np.allclose(got[c].to_numpy(), want):
            raise AssertionError(f"transform re-learned the bounds of {c!r}")


def test_transform_applies_frozen_bounds():
    train, test = _split(_panel())
    oc = OutlierCleaner(columns=["x", "y"], entity="id", time="t").fit(train)
    before = oc.bounds_.clone()
    oc.transform(test)
    assert oc.bounds_.equals(before)
    _check_frozen(OutlierCleaner, train, test)
    with pytest.raises(AssertionError, match="re-learned"):
        _check_frozen(_RefitsAtTransform, train, test)


# --------------------------------------------------------------------------- #
# 3. Train outputs cannot see the test fold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["mad", "iqr", "zscore", "quantile"])
@pytest.mark.parametrize("pooling", ["entity", "global"])
def test_fit_on_train_passes_and_fit_on_all_rows_fails(method, pooling):
    df = _panel()
    cut = 50

    def honest(frame):
        oc = OutlierCleaner(method=method, pooling=pooling, entity="id", time="t")
        return oc.fit(frame.filter(pl.col("t") < cut)).transform(frame)

    def leaky(frame):  # fit on all rows, test fold included
        oc = OutlierCleaner(method=method, pooling=pooling, entity="id", time="t")
        return oc.fit(frame).transform(frame)

    split = (list(range(cut)), list(range(cut, 80)))
    assert_no_train_test_leak(honest, df, split, entity="id", time="t")
    with pytest.raises(AssertionError, match="LEAK"):
        assert_no_train_test_leak(leaky, df, split, entity="id", time="t")


def test_cross_section_bounds_are_per_date():
    n_e, n_t = 30, 4
    base = np.linspace(-1.0, 1.0, n_e)
    df = pl.DataFrame(
        {
            "id": np.tile(np.arange(n_e), n_t),
            "t": np.repeat(np.arange(n_t), n_e),
            "x": np.tile(base, n_t) + np.repeat(np.arange(n_t) * 10.0, n_e),
        }
    ).with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("t") == 2))
        .then(40.0)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    oc = OutlierCleaner(
        pooling="cross_section", action="flag", columns="x", entity="id", time="t"
    )
    out = oc.fit_transform(df).collect()
    assert out.filter(pl.col("x_outlier"))[["id", "t"]].rows() == [(0, 2)]
    assert oc.bounds_ is None and oc.global_bounds_ is None
    # Same-date only: never looks at another date.
    assert_no_lookahead(lambda f: oc.transform(f), df, entity="id", time="t")


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def test_actions():
    train, test = _split(_panel())
    kw = {"columns": ["x"], "entity": "id", "time": "t"}
    clip = OutlierCleaner(action="clip", **kw).fit(train)
    out = clip.transform(test).collect()
    n = clip.n_outliers_["x"]
    assert n > 0
    assert out.schema["x"] == pl.Float64
    nul = OutlierCleaner(action="null", **kw).fit(train).transform(test).collect()
    assert nul["x"].null_count() == n
    assert nul["y"].equals(test["y"])
    flag = OutlierCleaner(action="flag", **kw).fit(train).transform(test).collect()
    assert flag["x_outlier"].sum() == n
    assert flag.columns == [*test.columns, "x_outlier"]
    drop = OutlierCleaner(action="drop", **kw).fit(train).transform(test).collect()
    assert drop.height == test.height - n
    # Input order is preserved.
    shuffled = test.sample(fraction=1.0, shuffle=True, seed=1)
    got = OutlierCleaner(action="flag", **kw).fit(train).transform(shuffled).collect()
    assert got.select("id", "t").equals(shuffled.select("id", "t"))


def test_nan_and_null_are_never_outliers():
    df = pl.DataFrame(
        {
            "id": ["a"] * 12,
            "t": range(12),
            "x": [1.0, 2.0, 1.5] * 3 + [float("nan"), None, 99.0],
        }
    )
    out = (
        OutlierCleaner(method="iqr", action="flag", entity="id", time="t", min_obs=3)
        .fit(df.head(9))
        .transform(df)
        .collect()
    )
    assert out["x_outlier"].to_list()[-3:] == [False, False, True]


def test_mad_zero_falls_back_to_mean_absolute_deviation():
    v = np.array([5.0] * 9 + [6.0, 7.0])
    df = pl.DataFrame({"id": ["a"] * v.size, "t": range(v.size), "x": v})
    oc = OutlierCleaner(columns="x", entity="id", time="t", min_obs=1).fit(df)
    lo, hi = oc.global_bounds_["x"]
    s = 1.2533 * np.mean(np.abs(v - v.mean()))
    assert hi == pytest.approx(5.0 + 3.5 * s)
    assert hi > 5.0 > lo


# --------------------------------------------------------------------------- #
# 4. Causal rolling filters
# --------------------------------------------------------------------------- #
def test_trailing_median_sigma_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(60)
    x[[5, 17]] = np.nan
    g = np.repeat([0, 1, 2], 20)
    med, sig = trailing_median_sigma(x, g, window=6, min_periods=3)
    for i in range(60):
        start = max(g.tolist().index(g[i]), i - 6)
        w = x[start:i]
        w = w[~np.isnan(w)]
        if w.size < 3:
            assert np.isnan(med[i])
            continue
        m = np.median(w)
        mad = np.median(np.abs(w - m))
        s = 1.4826 * mad if mad > 0 else 1.2533 * np.mean(np.abs(w - w.mean()))
        assert med[i] == pytest.approx(m)
        assert sig[i] == pytest.approx(s)


def test_rolling_matches_brute_force():
    df = _panel(n_entities=2, n_periods=40)
    oc = OutlierCleaner(method="rolling", window=8, columns="x", entity="id", time="t")
    frame = oc.fit(df)._with_bounds(oc._as_panel(df, method="t")).collect()
    x = df["x"].to_numpy()
    for i in range(df.height):
        e0 = (i // 40) * 40
        w = x[max(e0, i - 8) : i]
        if w.size < 4:
            assert frame[_lo("x")][i] is None
            continue
        assert frame[_lo("x")][i] == pytest.approx(w.mean() - 3 * w.std(ddof=1))


@pytest.mark.parametrize("method", ["hampel", "rolling"])
def test_rolling_filters_are_causal(method):
    df = _panel(n_entities=3, n_periods=60)
    oc = OutlierCleaner(method=method, window=10, entity="id", time="t")

    def op(frame):
        return oc.fit(frame).transform(frame)

    assert_no_lookahead(op, df, entity="id", time="t")
    assert_prefix_invariant(op, df, entity="id", time="t")
    out = OutlierCleaner(
        method=method, window=10, action="flag", columns="x"
    ).fit_transform(df, entity="id", time="t")
    flagged = out.collect().filter(pl.col("x_outlier"))
    # The +25 spikes dominate what gets flagged.
    spikes = df.filter(pl.col("x") - pl.col("t") * 4.0 / 59 > 15)
    caught = flagged.join(spikes.select("id", "t"), on=["id", "t"], how="semi")
    assert caught.height >= 0.8 * spikes.height
    assert caught.height >= 0.6 * flagged.height


class _ReadsFutureWindow(OutlierCleaner):
    """Deliberately leaky: judges each row by the window *after* it."""

    def _with_bounds(self, panel):
        lf = super()._with_bounds(panel)
        cols = self.feature_names_in_
        return lf.with_columns(
            [
                pl.col(x).shift(-self.window).over(panel.entity_col)
                for c in cols
                for x in (_lo(c), _hi(c))
            ]
        )


def test_future_window_variant_is_caught():
    df = _panel(n_entities=3, n_periods=60)
    leaky = _ReadsFutureWindow(method="hampel", window=10, entity="id", time="t")
    with pytest.raises(AssertionError, match="LEAK"):
        assert_no_lookahead(
            lambda f: leaky.fit(f).transform(f), df, entity="id", time="t"
        )
