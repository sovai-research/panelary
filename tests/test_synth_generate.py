"""The synthetic panel generator: determinism, prefix consistency, and dials.

Every dial test measures the mechanism on the ground truth the generator
returns, so a dial that silently stopped doing anything fails here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from panelary.synth import (
    GroundTruth,
    SynthConfig,
    SyntheticPanel,
    generate_panel,
    sample_config,
)

FRAMES = ("panel", "first_observed", "vintages")
TRUTH_FRAMES = ("latent", "factors", "entities")


def _sha(df: pl.DataFrame) -> str:
    buf = io.BytesIO()
    df.write_ipc(buf, compression="uncompressed")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _hashes(data: SyntheticPanel) -> dict[str, str]:
    out = {name: _sha(getattr(data, name)) for name in FRAMES}
    out.update({name: _sha(getattr(data.truth, name)) for name in TRUTH_FRAMES})
    return out


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x**4).mean() / (x**2).mean() ** 2 - 3.0)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_and_dials_give_byte_identical_frames() -> None:
    a = generate_panel(seed=11, n_entities=40, n_periods=150)
    b = generate_panel(seed=11, n_entities=40, n_periods=150)
    assert _hashes(a) == _hashes(b)
    assert a.truth.break_steps == b.truth.break_steps
    np.testing.assert_array_equal(a.truth.loadings, b.truth.loadings)


def test_different_seeds_differ() -> None:
    a = generate_panel(seed=1, n_entities=20, n_periods=80)
    b = generate_panel(seed=2, n_entities=20, n_periods=80)
    assert _sha(a.truth.latent) != _sha(b.truth.latent)


def test_config_and_keyword_dials_are_equivalent() -> None:
    cfg = SynthConfig(n_entities=15, n_periods=60)
    a = generate_panel(cfg.replace(tail_df=None, n_clusters=2), seed=4)
    b = generate_panel(cfg, seed=4, tail_df=None, n_clusters=2)
    assert _hashes(a) == _hashes(b)
    assert a.truth.config == b.truth.config


def test_byte_identical_across_processes() -> None:
    """A fresh interpreter, with different hash randomisation, gets the same bytes."""
    kwargs = {"n_entities": 25, "n_periods": 90, "signal_observed": True}
    here = _hashes(generate_panel(seed=9, **kwargs))
    code = (
        "import hashlib, io, json\n"
        "from panelary.synth import generate_panel\n"
        f"d = generate_panel(seed=9, **{kwargs!r})\n"
        "def sha(df):\n"
        "    b = io.BytesIO(); df.write_ipc(b, compression='uncompressed')\n"
        "    return hashlib.sha256(b.getvalue()).hexdigest()\n"
        "out = {n: sha(getattr(d, n)) for n in ('panel', 'first_observed', 'vintages')}\n"
        "out.update({n: sha(getattr(d.truth, n)) for n in ('latent', 'factors', 'entities')})\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    res = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=300,
    )
    there = json.loads(res.stdout.strip().splitlines()[-1])
    assert there == here


# --------------------------------------------------------------------------- #
# Prefix consistency
# --------------------------------------------------------------------------- #
def test_prefix_consistent_in_time() -> None:
    """T + k periods with the same seed reproduce every row with event time < T."""
    cfg = SynthConfig(n_entities=30, n_periods=120, break_rate=0.03)
    short = generate_panel(cfg, seed=3)
    long = generate_panel(cfg.replace(n_periods=200), seed=3)
    T = cfg.n_periods
    assert_frame_equal(short.panel, long.panel.filter(pl.col("time") < T))
    assert_frame_equal(
        short.first_observed, long.first_observed.filter(pl.col("time") < T)
    )
    assert_frame_equal(short.vintages, long.vintages.filter(pl.col("event_time") < T))
    assert_frame_equal(short.truth.latent, long.truth.latent.filter(pl.col("step") < T))
    assert_frame_equal(
        short.truth.factors, long.truth.factors.filter(pl.col("step") < T)
    )
    assert_frame_equal(short.truth.entities, long.truth.entities)
    assert short.truth.break_steps == tuple(b for b in long.truth.break_steps if b < T)
    # the test is only meaningful if something happened in both windows
    assert long.truth.break_steps and short.truth.break_steps


def test_prefix_consistent_in_entities() -> None:
    """N + m entities with the same seed reproduce the first N entities."""
    cfg = SynthConfig(n_entities=20, n_periods=100)
    small = generate_panel(cfg, seed=8)
    big = generate_panel(cfg.replace(n_entities=35), seed=8)
    keep = pl.col("entity").is_in(small.truth.entities["entity"].implode())
    for name in FRAMES:
        assert_frame_equal(getattr(small, name), getattr(big, name).filter(keep))
    assert_frame_equal(small.truth.latent, big.truth.latent.filter(keep))
    assert_frame_equal(small.truth.factors, big.truth.factors)
    np.testing.assert_array_equal(small.truth.loadings, big.truth.loadings[:20])


def test_prefix_consistent_with_calendar_time() -> None:
    cfg = SynthConfig(
        n_entities=10, n_periods=40, start=dt.date(2020, 1, 31), every="1mo"
    )
    short = generate_panel(cfg, seed=2)
    long = generate_panel(cfg.replace(n_periods=70), seed=2)
    cutoff = short.truth.latent["time"].max()
    assert_frame_equal(short.panel, long.panel.filter(pl.col("time") <= cutoff))
    assert_frame_equal(
        short.vintages, long.vintages.filter(pl.col("event_time") <= cutoff)
    )


# --------------------------------------------------------------------------- #
# Shape and schema
# --------------------------------------------------------------------------- #
def test_frames_are_long_sorted_and_keyed() -> None:
    data = generate_panel(seed=0)
    panel, fo, vint = data.panel, data.first_observed, data.vintages
    assert panel.columns == ["entity", "time", "value"]
    assert panel.schema["entity"] == pl.String
    assert panel.schema["time"] == pl.Int64
    assert panel.schema["value"] == pl.Float64
    assert panel.equals(panel.sort("entity", "time"))
    assert not panel.select(pl.struct("entity", "time").is_duplicated().any()).item()
    # first_observed is row-aligned with panel
    assert fo.columns == ["entity", "time", "value", "knowledge_time"]
    assert_frame_equal(fo.select("entity", "time"), panel.select("entity", "time"))
    assert fo["value"].is_null().equals(panel["value"].is_null())
    # vintages are bitemporal
    assert vint.columns == [
        "entity",
        "event_time",
        "knowledge_time",
        "value",
        "revision",
    ]
    assert vint.equals(vint.sort("entity", "event_time", "knowledge_time"))
    assert not vint.select(
        pl.struct("entity", "event_time", "revision").is_duplicated().any()
    ).item()
    assert isinstance(data.truth, GroundTruth)
    assert data.panel_frame().entity_col == "entity"


def test_signal_column_only_when_observed() -> None:
    hidden = generate_panel(seed=1, n_entities=10, n_periods=50)
    shown = generate_panel(seed=1, n_entities=10, n_periods=50, signal_observed=True)
    assert "signal" not in hidden.panel.columns
    assert shown.panel.columns == ["entity", "time", "value", "signal"]
    joined = shown.panel.join(
        shown.truth.latent.select("entity", "time", pl.col("signal").alias("true")),
        on=["entity", "time"],
    )
    assert (joined["signal"] == joined["true"]).all()
    # exposing the signal changes nothing else
    assert_frame_equal(hidden.panel, shown.panel.drop("signal"))


def test_calendar_time_labels() -> None:
    data = generate_panel(
        seed=0,
        n_entities=5,
        n_periods=30,
        start=dt.datetime(2024, 1, 2, 9, 30),
        every="5m",
        break_times=(10,),
    )
    assert data.panel.schema["time"] == pl.Datetime("us")
    assert data.vintages.schema["knowledge_time"] == pl.Datetime("us")
    assert data.truth.break_times == (dt.datetime(2024, 1, 2, 10, 20),)
    steps = data.truth.latent.filter(pl.col("entity") == "E00000")
    assert steps["time"][3] - steps["time"][2] == dt.timedelta(minutes=5)


# --------------------------------------------------------------------------- #
# Dials, each measured
# --------------------------------------------------------------------------- #
def test_student_t_innovations_are_heavy_tailed() -> None:
    gauss = generate_panel(seed=0, tail_df=None).truth.latent["shock"].to_numpy()
    heavy = generate_panel(seed=0, tail_df=5.0).truth.latent["shock"].to_numpy()
    # 10,000 shocks: the Gaussian excess kurtosis has s.e. ~0.05
    assert abs(_excess_kurtosis(gauss)) < 0.3
    assert _excess_kurtosis(heavy) > 1.5
    # both are scaled to unit variance
    assert abs(gauss.std() - 1.0) < 0.05
    assert abs(heavy.std() - 1.0) < 0.1


@pytest.mark.parametrize(("kappa", "sign"), [(1.5, 1), (-1.5, -1), (0.0, 0)])
def test_missingness_is_informative_only_when_asked(kappa: float, sign: int) -> None:
    data = generate_panel(
        seed=0, missing_rate=0.2, missing_informativeness=kappa, n_periods=300
    )
    obs = data.truth.latent.filter("observed")
    deviation = (obs["value"] - obs["level"]).to_numpy()
    missing = obs["missing"].to_numpy().astype(float)
    r = np.corrcoef(deviation, missing)[0, 1]
    if sign == 0:
        assert abs(r) < 0.05
    else:
        assert sign * r > 0.3
    # the observable panel carries exactly those nulls
    assert data.panel["value"].null_count() == int(missing.sum())


def test_no_missing_values_at_zero_rate() -> None:
    data = generate_panel(seed=0, missing_rate=0.0)
    assert data.panel["value"].null_count() == 0
    assert not data.truth.latent["missing"].any()


def test_entry_and_exit_vary_the_cross_section() -> None:
    T = 300
    data = generate_panel(
        seed=0,
        n_entities=200,
        n_periods=T,
        initial_fraction=0.6,
        entry_rate=0.02,
        exit_rate=0.01,
    )
    ent = data.truth.entities
    alive0 = data.truth.latent.filter(pl.col("step") == 0)["alive"].sum()
    assert alive0 == ent.filter(pl.col("entry_step") == 0).height
    assert 0.45 * 200 < alive0 < 0.75 * 200  # initial_fraction=0.6
    assert ent.filter(pl.col("entry_step") > 0).height > 30  # late entrants
    assert ent.filter(pl.col("exit_step") < T).height > 100  # exits happen
    # the latent alive flag is exactly [entry, exit)
    lat = data.truth.latent.join(
        ent.select("entity", "entry_step", "exit_step"), on="entity"
    )
    expected = (pl.col("step") >= pl.col("entry_step")) & (
        pl.col("exit_step").is_null() | (pl.col("step") < pl.col("exit_step"))
    )
    assert lat.select((pl.col("alive") == expected.fill_null(False)).all()).item()
    # N varies over time in the observable panel
    n_t = data.panel.group_by("time").len()["len"]
    assert n_t.max() - n_t.min() > 50


def test_plain_config_is_a_balanced_panel() -> None:
    cfg = SynthConfig.plain(n_entities=12, n_periods=40)
    data = generate_panel(cfg, seed=0)
    assert data.panel.height == 12 * 40
    assert data.panel["value"].null_count() == 0
    assert data.truth.break_steps == ()
    assert (data.first_observed["knowledge_time"] == data.first_observed["time"]).all()
    assert data.vintages.height == data.panel.height  # one release each


def test_asynchronous_observation_schedules() -> None:
    data = generate_panel(
        seed=0, n_entities=40, n_periods=100, observation_periods=(1, 5)
    )
    ent = data.truth.entities
    assert set(ent["observation_period"].unique().to_list()) == {1, 5}
    rows = data.panel.join(
        ent.select("entity", "observation_period", "observation_phase"), on="entity"
    )
    on_schedule = (
        (pl.col("time") - pl.col("observation_phase")) % pl.col("observation_period")
    ) == 0
    assert rows.select(on_schedule.all()).item()
    # slow observers have ~1/5 as many rows as fast ones
    per = rows.group_by("entity", "observation_period").len()
    mean_len = per.group_by("observation_period").agg(pl.col("len").mean())
    ratio = mean_len.sort("observation_period")["len"].to_list()
    assert ratio[0] / ratio[1] > 3.0
    # so entities are observed on different time grids
    grids = rows.group_by("entity").agg(pl.col("time").min()).n_unique("time")
    assert grids > 1


def test_observation_prob_thins_the_schedule() -> None:
    full = generate_panel(seed=0, observation_prob=1.0, exit_rate=0.0)
    thin = generate_panel(seed=0, observation_prob=0.5, exit_rate=0.0)
    assert 0.4 < thin.panel.height / full.panel.height < 0.6


def test_reporting_lag_and_bitemporal_vintages() -> None:
    data = generate_panel(
        seed=0, reporting_lag=(1, 4), reporting_jitter=0.3, revision_prob=0.5
    )
    vint = data.vintages.join(
        data.truth.entities.select("entity", "reporting_lag"), on="entity"
    )
    # nothing is known before its event, nor before its reporting lag
    assert (vint["knowledge_time"] >= vint["event_time"]).all()
    assert (vint["knowledge_time"] - vint["event_time"] >= vint["reporting_lag"]).all()
    first = vint.filter(pl.col("revision") == 0)
    delay = first["knowledge_time"] - first["event_time"] - first["reporting_lag"]
    assert set(delay.unique().to_list()) == {0, 1}  # jitter adds at most one step
    # releases of one value are strictly later than the previous one
    gaps = vint.select(
        pl.col("knowledge_time").diff().over("entity", "event_time").alias("g")
    )["g"].drop_nulls()
    assert (gaps >= 1).all()
    # revision numbers run 0..n without holes
    runs = vint.group_by("entity", "event_time").agg(
        pl.col("revision").max().alias("mx"), pl.len().alias("n")
    )
    assert (runs["mx"] + 1 == runs["n"]).all()


def test_revisions_converge_to_the_panel_value() -> None:
    data = generate_panel(seed=0, revision_prob=0.6, max_revisions=3)
    last = data.vintages.group_by("entity", "event_time").agg(
        pl.col("value").sort_by("revision").last()
    )
    panel = data.panel.drop_nulls("value").rename({"time": "event_time"})
    both = panel.join(last, on=["entity", "event_time"], suffix="_final")
    assert both.height == panel.height  # every non-missing value is released
    assert (both["value"] == both["value_final"]).all()  # exactly
    # first_observed carries revision 0
    first = data.vintages.filter(pl.col("revision") == 0).rename({"event_time": "time"})
    fo = data.first_observed.drop_nulls("value").join(
        first, on=["entity", "time"], suffix="_v"
    )
    assert (fo["value"] == fo["value_v"]).all()
    assert (fo["knowledge_time"] == fo["knowledge_time_v"]).all()
    # some values were revised, and their first release was wrong
    revised = (
        data.vintages.group_by("entity", "event_time").len().filter(pl.col("len") > 1)
    )
    assert revised.height > 0.4 * panel.height
    changed = data.first_observed["value"] != data.panel["value"]
    assert changed.sum() > 0.4 * panel.height


def test_no_revisions_means_first_release_is_final() -> None:
    data = generate_panel(seed=0, revision_prob=0.0)
    assert (data.vintages["revision"] == 0).all()
    assert_frame_equal(
        data.first_observed.select("entity", "time", "value"), data.panel
    )


def test_explicit_structural_break_shifts_the_level_there_only() -> None:
    data = generate_panel(seed=0, n_periods=120, break_times=(60,), break_size=3.0)
    assert data.truth.break_steps == (60,)
    jumps = data.truth.latent.with_columns(
        d=pl.col("level").diff().over("entity", order_by="step")
    ).filter(pl.col("d").abs() > 0)
    assert set(jumps["step"].unique().to_list()) == {60}
    # every entity's level moves, by roughly break_size on average
    assert jumps["entity"].n_unique() == 50
    assert 1.5 < jumps["d"].abs().mean() < 4.5


def test_break_hazard_sets_the_break_count() -> None:
    T, rate = 2000, 0.05
    data = generate_panel(seed=0, n_entities=3, n_periods=T, break_rate=rate)
    n = len(data.truth.break_steps)
    expected = rate * (T - 1)  # ~100, s.d. ~10
    assert 0.6 * expected < n < 1.4 * expected
    assert (
        generate_panel(
            seed=0, n_entities=3, n_periods=T, break_rate=0.0
        ).truth.break_steps
        == ()
    )


def test_regimes_switch_volatility_with_the_set_persistence() -> None:
    T = 3000
    data = generate_panel(
        seed=0,
        n_entities=3,
        n_periods=T,
        n_regimes=2,
        regime_persistence=0.97,
        regime_vol_ratio=3.0,
        regime_mean_shift=0.0,
        sv_vol=0.0,
        factor_persistence=0.0,
        tail_df=None,
    )
    fac = data.truth.factors
    sd = fac.group_by("regime").agg(pl.col("factor_0").std()).sort("regime")["factor_0"]
    assert 2.4 < sd[1] / sd[0] < 3.6
    switches = int((fac["regime"].diff().drop_nulls() != 0).sum())
    assert 0.5 * 0.03 * T < switches < 1.5 * 0.03 * T
    assert data.truth.regime_vol.tolist() == [1.0, 3.0]


def test_stochastic_volatility_dial() -> None:
    off = generate_panel(seed=0, sv_vol=0.0, n_periods=400).truth.factors
    on = generate_panel(
        seed=0, sv_vol=0.3, sv_persistence=0.9, n_periods=400
    ).truth.factors
    assert (off["log_vol_0"] == 0.0).all()
    # stationary variance sv_vol^2 / (1 - phi^2) = 0.474
    assert 0.2 < on["log_vol_0"].var() < 1.0


def test_clusters_share_comovement() -> None:
    data = generate_panel(
        seed=0,
        n_entities=45,
        n_periods=400,
        n_clusters=3,
        cluster_strength=1.0,
        loading_dispersion=0.1,
    )
    wide = (
        data.truth.latent.pivot(on="entity", index="step", values="common")
        .drop("step")
        .to_numpy()
    )
    corr = np.corrcoef(wide.T)
    cl = data.truth.clusters
    same = cl[:, None] == cl[None, :]
    off_diag = ~np.eye(len(cl), dtype=bool)
    within = corr[same & off_diag].mean()
    between = corr[~same].mean()
    assert within > between + 0.3


def test_sample_config_spans_the_dial_space() -> None:
    cfgs = [sample_config(k) for k in range(40)]
    assert sample_config(3) == sample_config(3)
    assert len({c.n_entities for c in cfgs}) > 20
    assert len({c.n_periods for c in cfgs}) > 20
    assert {c.tail_df is None for c in cfgs} == {True, False}
    assert {c.n_regimes for c in cfgs} == {1, 2, 3}
    assert all(10 <= c.n_entities <= 200 and 50 <= c.n_periods <= 500 for c in cfgs)
    pinned = sample_config(3, n_entities=(7, 7), signal_lag=4)
    assert pinned.n_entities == 7 and pinned.signal_lag == 4
    # pinning a dial does not reshuffle the others
    assert pinned.tail_df == sample_config(3).tail_df
    for k in range(3):
        data = generate_panel(sample_config(k, n_periods=(50, 80)), seed=k)
        assert data.panel.columns[:3] == ["entity", "time", "value"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dials",
    [
        {"tail_df": 2.0},
        {"n_entities": 0},
        {"factor_persistence": 1.0},
        {"missing_rate": 1.0},
        {"observation_periods": ()},
        {"reporting_lag": (3, 1)},
        {"revision_gap": (0, 2)},
        {"break_times": (0,)},
        {"start": dt.date(2020, 1, 1), "every": "1h"},
        {"start": dt.date(2020, 1, 1), "every": "daily"},
        {"signal_lag": -1},
    ],
)
def test_invalid_dials_are_refused(dials: dict) -> None:
    with pytest.raises(ValueError):
        SynthConfig(**dials)


def test_bad_seed_and_unknown_dial() -> None:
    with pytest.raises(ValueError):
        generate_panel(seed=-1)
    with pytest.raises(ValueError):
        generate_panel(seed=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unknown SynthConfig dial"):
        generate_panel(seed=0, n_entitys=5)
    with pytest.raises(ValueError):
        sample_config(-3)
