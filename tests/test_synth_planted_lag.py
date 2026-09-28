"""The planted signal, and the leak check it makes possible.

``value[t]`` loads on ``signal[t - L]`` with coefficient ``beta``; the signal is
i.i.d. and otherwise invisible. So a causal feature can carry the signal only
at lags ``>= L`` -- and a feature that carries it earlier read the future. Each
test here pairs a deliberately leaky feature with its causal counterpart, so
a check that stopped detecting anything would fail.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.synth import (
    PlantedLagReport,
    SynthConfig,
    check_planted_lag,
    generate_panel,
    sample_config,
)

OVER = {"partition_by": "entity", "order_by": "time"}


def _with(panel: pl.DataFrame, **exprs: pl.Expr) -> pl.DataFrame:
    return panel.with_columns(**{k: v.over(**OVER) for k, v in exprs.items()})


# --------------------------------------------------------------------------- #
# The planted relationship itself
# --------------------------------------------------------------------------- #
def test_planted_coefficient_is_recovered_at_the_planted_lag_only() -> None:
    L, beta = 3, 0.7
    data = generate_panel(
        SynthConfig.plain(n_entities=60, n_periods=250),
        seed=0,
        signal_lag=L,
        signal_coef=beta,
    )
    assert (data.truth.signal_lag, data.truth.signal_coef) == (L, beta)
    lat = data.truth.latent
    slopes = {}
    for lag in range(0, 7):
        pairs = lat.select("entity", "step", "value").join(
            lat.select("entity", (pl.col("step") + lag).alias("step"), "signal"),
            on=["entity", "step"],
        )
        x, y = pairs["signal"].to_numpy(), pairs["value"].to_numpy()
        slopes[lag] = float(np.cov(x, y)[0, 1] / x.var(ddof=1))
    # ~15,000 pairs, residual s.d. ~1.5: slope s.e. ~0.012
    assert abs(slopes[L] - beta) < 0.06
    for lag, b in slopes.items():
        if lag != L:
            assert abs(b) < 0.06, (lag, b)


def test_signal_is_independent_of_everything_but_its_own_effect() -> None:
    data = generate_panel(seed=2, n_periods=300)
    lat = data.truth.latent
    sig = lat["signal"].to_numpy()
    assert abs(sig.mean()) < 0.05 and abs(sig.std() - 1.0) < 0.05
    for col in ("shock", "level", "common"):
        assert abs(np.corrcoef(sig, lat[col].to_numpy())[0, 1]) < 0.04


# --------------------------------------------------------------------------- #
# The check: leaky features flagged, causal ones passed
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    return generate_panel(SynthConfig(n_entities=60, n_periods=250), seed=1)


def test_shift_minus_one_is_flagged_and_shift_one_passes(data) -> None:
    L = data.truth.signal_lag
    frame = _with(
        data.panel,
        leaky=pl.col("value").shift(-1),
        causal=pl.col("value").shift(1),
    )
    leaky = check_planted_lag(frame, "leaky", data.truth)
    causal = check_planted_lag(frame, "causal", data.truth)

    assert leaky.leaks
    assert leaky.recovered_lag == L - 1
    assert L - 1 in leaky.leak_lags
    assert "LEAKS" in str(leaky)

    assert not causal.leaks
    assert causal.recovered_lag == L + 1
    assert causal.leak_lags == ()


def test_value_itself_recovers_the_signal_at_exactly_the_planted_lag(data) -> None:
    rep = check_planted_lag(data.panel, "value", data.truth)
    assert not rep.leaks
    assert rep.recovered_lag == data.truth.signal_lag
    assert rep.min_lag == data.truth.signal_lag


@pytest.mark.parametrize(
    ("name", "expr", "leaks"),
    [
        ("trailing_mean", pl.col("value").rolling_mean(5, min_samples=1), False),
        ("ewm", pl.col("value").ewm_mean(alpha=0.3, ignore_nulls=True), False),
        ("expanding_max", pl.col("value").cum_max(), False),
        ("forward_fill", pl.col("value").forward_fill(), False),
        (
            "centred_mean",
            pl.col("value").rolling_mean(5, center=True, min_samples=1),
            True,
        ),
        ("lead_diff", pl.col("value").shift(-2) - pl.col("value"), True),
    ],
)
def test_common_operators(data, name: str, expr: pl.Expr, leaks: bool) -> None:
    frame = _with(data.panel, f=expr)
    rep = check_planted_lag(frame, "f", data.truth)
    assert rep.leaks is leaks, str(rep)


@pytest.mark.parametrize("seed", range(5))
def test_no_false_alarms_on_causal_features(seed: int) -> None:
    d = generate_panel(seed=100 + seed, n_entities=50, n_periods=200)
    frame = _with(
        d.panel,
        a=pl.col("value").shift(1),
        b=pl.col("value").rolling_mean(10, min_samples=1),
        c=pl.col("value").diff(),
    )
    for f in ("a", "b", "c", "value"):
        rep = check_planted_lag(frame, f, d.truth)
        assert not rep.leaks, str(rep)


def test_null_z_is_calibrated_across_the_prior() -> None:
    """Causal features on panels drawn across the dial space -- Student-t tails,
    informative missingness, sparse schedules, observed or latent signal -- stay
    well inside the threshold at every lag below the knowable one."""
    null_z: list[float] = []
    for k in range(8):
        cfg = sample_config(k, n_entities=(30, 80), n_periods=(100, 200))
        d = generate_panel(cfg, seed=k)
        frame = _with(
            d.panel,
            a=pl.col("value").shift(1),
            b=pl.col("value").rolling_mean(10, min_samples=1),
        )
        for f in ("value", "a", "b"):
            rep = check_planted_lag(frame, f, d.truth, max_lead=30)
            assert not rep.leaks, (k, str(rep))
            null_z += [
                abs(z)
                for lag, z in zip(rep.lags, rep.z, strict=True)
                if lag < rep.min_lag and np.isfinite(z)
            ]
    # ~700 null z-scores: the largest of that many |N(0,1)| is ~3.2 on average
    assert len(null_z) > 500
    assert max(null_z) < 4.5
    assert 0.6 < float(np.mean(np.square(null_z))) < 1.4


def test_one_row_lookahead_on_an_asynchronous_panel() -> None:
    """A one-*row* shift is five steps on a period-5 schedule; the check sees it."""
    d = generate_panel(
        SynthConfig.plain(n_entities=100, n_periods=300, observation_periods=(5,)),
        seed=3,
    )
    L = d.truth.signal_lag
    frame = _with(
        d.panel, leaky=pl.col("value").shift(-1), causal=pl.col("value").shift(1)
    )
    leaky = check_planted_lag(frame, "leaky", d.truth)
    assert leaky.leaks and leaky.recovered_lag == L - 5
    causal = check_planted_lag(frame, "causal", d.truth)
    assert not causal.leaks and causal.recovered_lag == L + 5


def test_an_observed_signal_lowers_the_knowable_lag() -> None:
    d = generate_panel(
        SynthConfig.plain(n_entities=60, n_periods=250, signal_observed=True), seed=4
    )
    L = d.truth.signal_lag
    assert d.truth.knowable_lag() == 0
    frame = _with(
        d.panel,
        sig_lead=pl.col("signal").shift(-1),
        sig_lag=pl.col("signal").shift(L),
        val_lead=pl.col("value").shift(-1),
    )
    assert check_planted_lag(frame, "sig_lead", d.truth).leaks
    ok = check_planted_lag(frame, "sig_lag", d.truth)
    assert not ok.leaks and ok.recovered_lag == L
    # A value-only lead carries the signal at L - 1 >= 0: invisible at the
    # default floor of 0, caught once the caller says the signal is unused.
    assert not check_planted_lag(frame, "val_lead", d.truth).leaks
    assert check_planted_lag(frame, "val_lead", d.truth, min_lag=L).leaks


def test_knowledge_clock_flags_features_that_ignore_the_reporting_lag() -> None:
    """Using an event-time value on its event date leaks the reporting lag.

    The point-in-time version -- the latest first release whose knowledge time
    is at or before the decision time, via an as-of join -- does not.
    """
    d = generate_panel(
        seed=5,
        n_entities=60,
        n_periods=250,
        reporting_lag=(2, 4),
        reporting_jitter=0.0,
        revision_prob=0.5,
    )
    L, lag_min = d.truth.signal_lag, d.truth.min_reporting_lag
    assert lag_min >= 2
    assert d.truth.knowable_lag("knowledge") == L + lag_min

    # naive: the value for event t, used at decision time t
    naive = check_planted_lag(d.panel, "value", d.truth, clock="knowledge")
    assert naive.leaks and naive.recovered_lag == L

    # point-in-time: as-of join on knowledge_time
    decisions = d.panel.select("entity", "time").sort("time")
    releases = (
        d.first_observed.drop_nulls("value")
        .select("entity", "knowledge_time", pl.col("value").alias("pit"))
        .sort("knowledge_time")
    )
    pit = decisions.join_asof(
        releases,
        left_on="time",
        right_on="knowledge_time",
        by="entity",
        strategy="backward",
        check_sortedness=False,  # both sides are sorted on the key; `by` hides it
    )
    rep = check_planted_lag(pit, "pit", d.truth, clock="knowledge")
    assert not rep.leaks, str(rep)
    assert rep.recovered_lag is not None and rep.recovered_lag >= L + lag_min


# --------------------------------------------------------------------------- #
# Report and argument handling
# --------------------------------------------------------------------------- #
def test_report_shape_and_inputs(data) -> None:
    frame = _with(data.panel, f=pl.col("value").shift(-1))
    rep = check_planted_lag(frame, "f", data.truth, max_lead=4, max_lag=2)
    assert isinstance(rep, PlantedLagReport)
    L = data.truth.signal_lag
    assert rep.lags == tuple(range(L - 4, L + 2 + 1))
    tab = rep.to_frame()
    assert tab.columns == ["lag", "correlation", "z", "n_obs", "significant", "leaks"]
    assert tab.filter("leaks")["lag"].to_list() == list(rep.leak_lags)
    # PanelFrame and LazyFrame inputs give the same answer
    from panelary.core.panel_frame import PanelFrame

    via_pf = check_planted_lag(
        PanelFrame(frame, entity="entity", time="time"), "f", data.truth
    )
    via_lazy = check_planted_lag(frame.lazy(), "f", data.truth)
    assert via_pf.z == via_lazy.z == check_planted_lag(frame, "f", data.truth).z


def test_report_argument_errors(data) -> None:
    with pytest.raises(KeyError):
        check_planted_lag(data.panel, "nope", data.truth)
    with pytest.raises(ValueError):
        check_planted_lag(data.panel, "value", data.truth, max_lead=-1)
    with pytest.raises(ValueError):
        check_planted_lag(data.panel, "value", data.truth, threshold=0.0)
    with pytest.raises(ValueError):
        check_planted_lag(data.panel, "value", data.truth, clock="wall")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        check_planted_lag([1, 2], "value", data.truth)
