"""The point-in-time (bitemporal) as-of join, :func:`panelary.core.asof.asof_join`.

Contract under test: each panel row ``(entity, t)`` receives, among the
vintages of that entity with ``knowledge_time + lag <= t`` and
``event_time <= t``, the one with the latest ``event_time`` and -- for that
period -- the latest ``knowledge_time``.

* a hand-built **restatement** example, with every cell written out;
* a brute-force **oracle** on random bitemporal tables (int and Date axes);
* the library's own leak instruments, in both directions -- perturbing and
  truncating the *vintages* by knowledge time, and the *panel* by time -- plus
  a deliberately leaky join that those instruments must catch;
* **unsorted** input, **missing** entities, publication **lag**, provenance,
  dtype refusals and the fail-closed validation.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from panelary.core._calendar import BusinessDays
from panelary.core.asof import asof_join
from panelary.core.panel_frame import PanelFrame
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

D = dt.date


# --------------------------------------------------------------------------- #
# Hand-built restatement example
# --------------------------------------------------------------------------- #
#: Two tickers. "A" reports Q1 on 05-01, restates Q1 on 06-10 (before Q2 is
#: out), reports Q2 on 08-01, and restates Q1 *again* on 09-01 -- after Q2, so
#: it must not displace Q2 as the latest figure. It also pre-announces Q3
#: guidance on 08-15 (event 09-30 after knowledge -- a value about the future)
#: which must stay invisible until 09-30. "B" reports once. "Z" is in the
#: vintages but not the panel.
VINTAGES = pl.DataFrame(
    {
        "ticker": ["A", "A", "A", "A", "A", "B", "Z"],
        "event_time": [
            D(2024, 3, 31),
            D(2024, 3, 31),
            D(2024, 6, 30),
            D(2024, 3, 31),
            D(2024, 9, 30),
            D(2024, 3, 31),
            D(2024, 3, 31),
        ],
        "knowledge_time": [
            D(2024, 5, 1),
            D(2024, 6, 10),
            D(2024, 8, 1),
            D(2024, 9, 1),
            D(2024, 8, 15),
            D(2024, 7, 1),
            D(2024, 4, 1),
        ],
        "eps": [1.00, 0.95, 1.10, 0.90, 1.30, 2.00, 9.99],
    }
)

PANEL_DATES = [
    D(2024, 4, 30),  # nothing known yet
    D(2024, 5, 1),  # Q1 original, known that very day
    D(2024, 6, 9),  # still the original
    D(2024, 6, 10),  # first restatement
    D(2024, 7, 31),  # still restated Q1
    D(2024, 8, 1),  # Q2
    D(2024, 9, 2),  # Q2: Q1's second restatement does not displace it
    D(2024, 9, 30),  # Q3 guidance becomes about-the-past
]
EXPECTED_A = [None, 1.00, 1.00, 0.95, 0.95, 1.10, 1.10, 1.30]
EXPECTED_B = [None, None, None, None, 2.00, 2.00, 2.00, 2.00]


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A"] * 8 + ["B"] * 8 + ["C"] * 2,
            "date": PANEL_DATES * 2 + PANEL_DATES[:2],
            "px": np.arange(18, dtype=np.float64),
        }
    )


def test_restatement_example_cell_by_cell() -> None:
    out = asof_join(_panel(), VINTAGES, entity="ticker", time="date")
    assert out.columns == ["ticker", "date", "px", "eps"]
    assert out["eps"].to_list() == EXPECTED_A + EXPECTED_B + [None, None]


def test_restatement_example_provenance() -> None:
    out = asof_join(_panel(), VINTAGES, entity="ticker", time="date", provenance=True)
    a = out.filter(pl.col("ticker") == "A")
    assert a["event_time"].to_list() == [
        None,
        D(2024, 3, 31),
        D(2024, 3, 31),
        D(2024, 3, 31),
        D(2024, 3, 31),
        D(2024, 6, 30),
        D(2024, 6, 30),
        D(2024, 9, 30),
    ]
    assert a["knowledge_time"].to_list() == [
        None,
        D(2024, 5, 1),
        D(2024, 5, 1),
        D(2024, 6, 10),
        D(2024, 6, 10),
        D(2024, 8, 1),
        D(2024, 8, 1),
        D(2024, 8, 15),
    ]


def test_suffix_and_value_selection() -> None:
    out = asof_join(
        _panel(), VINTAGES, entity="ticker", time="date", values="eps", suffix="_pit"
    )
    assert out.columns[-1] == "eps_pit"
    with pytest.raises(ValueError, match="overwrite"):
        asof_join(
            _panel().with_columns(eps=pl.lit(0.0)),
            VINTAGES,
            entity="ticker",
            time="date",
        )


def test_panelframe_in_panelframe_out_and_method() -> None:
    pf = PanelFrame(_panel(), entity="ticker", time="date").sort_panel()
    out = pf.asof_join(VINTAGES)
    assert isinstance(out, PanelFrame)
    assert (out.entity_col, out.time_col) == ("ticker", "date")
    assert out.sortedness == "panel"  # rows were not reordered
    assert out.collect()["eps"].to_list()[:8] == EXPECTED_A


def test_lazy_in_lazy_out() -> None:
    out = asof_join(_panel().lazy(), VINTAGES.lazy(), entity="ticker", time="date")
    assert isinstance(out, pl.LazyFrame)
    assert out.collect()["eps"].to_list()[:8] == EXPECTED_A


# --------------------------------------------------------------------------- #
# Brute-force oracle on random bitemporal tables
# --------------------------------------------------------------------------- #
def _oracle(panel: pl.DataFrame, vint: pl.DataFrame, lag: int = 0) -> list[Any]:
    vrows = vint.rows(named=True)
    out: list[Any] = []
    for e, t in panel.select("entity", "t").iter_rows():
        if t is None:
            out.append(None)
            continue
        cands = [
            v
            for v in vrows
            if v["entity"] == e
            and v["knowledge_time"] + lag <= t
            and v["event_time"] <= t
        ]
        best = max(
            cands, key=lambda v: (v["event_time"], v["knowledge_time"]), default=None
        )
        out.append(None if best is None else best["value"])
    return out


def _random_case(
    seed: int, n_ent: int = 4, n_t: int = 30, n_v: int = 60
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ents = [f"E{i}" for i in range(n_ent)]
    panel = pl.DataFrame(
        {
            "entity": np.repeat(ents, n_t),
            "t": np.tile(np.arange(n_t, dtype=np.int64), n_ent),
            "x": rng.standard_normal(n_ent * n_t),
        }
    ).sample(fraction=1.0, shuffle=True, seed=seed)  # unsorted on purpose
    event = rng.integers(0, n_t, n_v)
    # knowledge mostly after the event (a publication delay), sometimes before
    # it (guidance about a future period) -- both regimes must be right.
    knowledge = event + rng.integers(-3, 8, n_v)
    vint = (
        pl.DataFrame(
            {
                "entity": rng.choice(ents + ["ghost"], n_v),
                "event_time": event.astype(np.int64),
                "knowledge_time": knowledge.astype(np.int64),
                "value": rng.standard_normal(n_v),
            }
        )
        .unique(
            subset=["entity", "event_time", "knowledge_time"],
            keep="first",
            maintain_order=True,
        )
        .sample(fraction=1.0, shuffle=True, seed=seed + 1)
    )
    return panel, vint


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("lag", [0, 2])
def test_matches_brute_force_oracle(seed: int, lag: int) -> None:
    panel, vint = _random_case(seed)
    out = asof_join(panel, vint, entity="entity", time="t", lag=lag or None)
    assert out.select("entity", "t", "x").equals(panel)  # rows untouched, order kept
    assert out["value"].to_list() == _oracle(panel, vint, lag)


def test_matches_oracle_on_a_date_axis() -> None:
    panel, vint = _random_case(11)
    base = D(2024, 1, 1)
    panel_d = panel.with_columns(
        (pl.lit(base) + pl.duration(days=pl.col("t"))).cast(pl.Date).alias("t")
    )
    vint_d = vint.with_columns(
        (pl.lit(base) + pl.duration(days=pl.col(c))).cast(pl.Date).alias(c)
        for c in ("event_time", "knowledge_time")
    )
    out = asof_join(panel_d, vint_d, entity="entity", time="t")
    assert out["value"].to_list() == _oracle(panel, vint, 0)


# --------------------------------------------------------------------------- #
# The leak instruments -- both directions, and a leaky join they must catch
# --------------------------------------------------------------------------- #
def _date_case() -> tuple[pl.DataFrame, pl.DataFrame]:
    panel, vint = _random_case(3, n_ent=3, n_t=40, n_v=90)
    base = D(2024, 1, 1)
    panel_d = panel.with_columns(
        (pl.lit(base) + pl.duration(days=pl.col("t"))).cast(pl.Date).alias("t")
    )
    vint_d = vint.with_columns(
        (pl.lit(base) + pl.duration(days=pl.col(c))).cast(pl.Date).alias(c)
        for c in ("event_time", "knowledge_time")
    )
    return panel_d, vint_d


def _join_onto(grid: pl.DataFrame, join: Any) -> Any:
    """``vintages -> panel`` op whose output time key is named like the
    vintage table's knowledge time, so the instruments can align rows."""

    def op(v: pl.DataFrame) -> pl.DataFrame:
        return join(grid, v).rename({"t": "knowledge_time"}).drop("x")

    return op


def _pit(grid: pl.DataFrame, v: pl.DataFrame) -> pl.DataFrame:
    return asof_join(grid, v, entity="entity", time="t", values="value")


def _leaky(grid: pl.DataFrame, v: pl.DataFrame) -> pl.DataFrame:
    """The classic mistake: latest *vintage* per period, joined on event time --
    a restatement from the future overwrites what was known at the time."""
    latest = (
        v.sort("knowledge_time")
        .group_by("entity", "event_time")
        .agg(pl.col("value").last())
    )
    return (
        grid.sort("entity", "t")
        .join_asof(
            latest.sort("event_time"),
            left_on="t",
            right_on="event_time",
            by="entity",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("event_time")
    )


def test_no_lookahead_in_knowledge_time() -> None:
    """Corrupting every vintage known after ``cut`` leaves every row at
    ``t <= cut`` untouched."""
    grid, vint = _date_case()
    op = _join_onto(grid, _pit)
    assert_no_lookahead(op, vint, entity="entity", time="knowledge_time")


def test_prefix_invariant_in_knowledge_time() -> None:
    """Deleting every vintage known after ``cut`` leaves every row at
    ``t <= cut`` untouched -- the point-in-time property itself."""
    grid, vint = _date_case()
    op = _join_onto(grid, _pit)
    assert_prefix_invariant(op, vint, entity="entity", time="knowledge_time")


def test_leaky_join_is_caught_by_both_instruments() -> None:
    """Guard the guard: the latest-vintage join fails both checks."""
    grid, vint = _date_case()
    op = _join_onto(grid, _leaky)
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(op, vint, entity="entity", time="knowledge_time")
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(op, vint, entity="entity", time="knowledge_time")


def test_panel_side_is_causal_and_prefix_invariant() -> None:
    grid, vint = _date_case()

    def op(p: pl.DataFrame) -> pl.DataFrame:
        return asof_join(p, vint, entity="entity", time="t")

    assert_no_lookahead(op, grid, entity="entity", time="t")
    assert_prefix_invariant(op, grid, entity="entity", time="t")


# --------------------------------------------------------------------------- #
# Order, missing entities, nulls
# --------------------------------------------------------------------------- #
def test_unsorted_input_gives_the_same_answer_in_the_callers_order() -> None:
    panel = _panel()
    ref = asof_join(panel, VINTAGES, entity="ticker", time="date")
    perm = np.random.default_rng(0).permutation(panel.height)
    shuffled = panel[perm]
    vshuf = VINTAGES.sample(fraction=1.0, shuffle=True, seed=5)
    out = asof_join(shuffled, vshuf, entity="ticker", time="date")
    assert out.equals(ref[perm])


def test_missing_entities_and_null_time() -> None:
    panel = pl.DataFrame(
        {
            "ticker": ["A", "C", None, "A"],
            "date": [D(2024, 9, 2), D(2024, 9, 2), D(2024, 9, 2), None],
        }
    )
    out = asof_join(panel, VINTAGES, entity="ticker", time="date")
    # C has no vintages; a null entity matches nothing; a null time is at no
    # point in time and must not inherit A's latest vintage.
    assert out["eps"].to_list() == [1.10, None, None, None]


def test_empty_vintage_table() -> None:
    out = asof_join(_panel(), VINTAGES.clear(), entity="ticker", time="date")
    assert out["eps"].null_count() == out.height


def test_known_null_value_is_returned_as_null() -> None:
    """A vintage that reports null is still the latest thing known."""
    v = VINTAGES.with_columns(
        pl.when(pl.col("knowledge_time") == D(2024, 8, 1))
        .then(None)
        .otherwise(pl.col("eps"))
        .alias("eps")
    )
    out = asof_join(_panel(), v, entity="ticker", time="date")
    assert out["eps"].to_list()[5] is None  # A on 08-01: Q2, reported as null


# --------------------------------------------------------------------------- #
# Publication lag
# --------------------------------------------------------------------------- #
def test_business_day_lag() -> None:
    """Filed Friday 05-03 with a 1-business-day lag -> usable Monday 05-06."""
    v = pl.DataFrame(
        {
            "ticker": ["A"],
            "event_time": [D(2024, 3, 31)],
            "knowledge_time": [D(2024, 5, 3)],
            "eps": [1.0],
        }
    )
    panel = pl.DataFrame(
        {"ticker": ["A"] * 4, "date": [D(2024, 5, d) for d in (3, 4, 5, 6)]}
    )
    for lag in ("1bd", BusinessDays(1)):
        out = asof_join(panel, v, entity="ticker", time="date", lag=lag)
        assert out["eps"].to_list() == [None, None, None, 1.0]
    out = asof_join(panel, v, entity="ticker", time="date", lag="1d")
    assert out["eps"].to_list() == [None, 1.0, 1.0, 1.0]


def test_datetime_lag_and_mixed_units() -> None:
    """A 16:30 filing with a 30-minute lag, on a ns panel against ms vintages."""
    v = pl.DataFrame(
        {
            "ticker": ["A"],
            "event_time": [dt.datetime(2024, 3, 31)],
            "knowledge_time": [dt.datetime(2024, 5, 1, 16, 30)],
            "eps": [1.0],
        }
    ).with_columns(pl.col("event_time", "knowledge_time").cast(pl.Datetime("ms")))
    panel = pl.DataFrame(
        {
            "ticker": ["A"] * 3,
            "date": [
                dt.datetime(2024, 5, 1, 16, 45),
                dt.datetime(2024, 5, 1, 17, 0),
                dt.datetime(2024, 5, 2, 9, 30),
            ],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    out = asof_join(
        panel, v, entity="ticker", time="date", lag=dt.timedelta(minutes=30)
    )
    assert out["eps"].to_list() == [None, 1.0, 1.0]
    assert out["date"].dtype == pl.Datetime("ns")  # the panel's column untouched


def test_integer_lag() -> None:
    panel, vint = _random_case(2)
    out = asof_join(panel, vint, entity="entity", time="t", lag=3)
    assert out["value"].to_list() == _oracle(panel, vint, 3)


# --------------------------------------------------------------------------- #
# Fail closed
# --------------------------------------------------------------------------- #
def test_date_panel_refuses_datetime_knowledge_time() -> None:
    v = VINTAGES.with_columns(pl.col("knowledge_time").cast(pl.Datetime("us")))
    with pytest.raises(TypeError, match="not silently reconciled"):
        asof_join(_panel(), v, entity="ticker", time="date")


def test_time_zone_mismatch_refused() -> None:
    v = VINTAGES.with_columns(
        pl.col("event_time", "knowledge_time")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
    )
    panel = _panel().with_columns(
        pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone("America/New_York")
    )
    with pytest.raises(TypeError, match="time zones"):
        asof_join(panel, v, entity="ticker", time="date")


def test_null_vintage_keys_refused() -> None:
    v = VINTAGES.with_columns(
        pl.when(pl.col("eps") == 2.0)
        .then(None)
        .otherwise(pl.col("knowledge_time"))
        .alias("knowledge_time")
    )
    with pytest.raises(ValueError, match="nulls"):
        asof_join(_panel(), v, entity="ticker", time="date")


def test_conflicting_duplicate_vintages_refused_exact_ones_collapsed() -> None:
    doubled = pl.concat([VINTAGES, VINTAGES])
    out = asof_join(_panel(), doubled, entity="ticker", time="date")
    assert out["eps"].to_list()[:8] == EXPECTED_A
    conflict = pl.concat([VINTAGES, VINTAGES.head(1).with_columns(eps=pl.lit(5.0))])
    with pytest.raises(ValueError, match="ambiguous"):
        asof_join(_panel(), conflict, entity="ticker", time="date")


@pytest.mark.parametrize(
    ("kwargs", "err", "match"),
    [
        ({"event_time": "nope"}, ValueError, "not in the vintage table"),
        ({"values": ["nope"]}, ValueError, "not in the vintage table"),
        ({"values": ["knowledge_time"]}, ValueError, "key columns"),
        ({"lag": "-1d"}, ValueError, "non-negative"),
        ({"lag": 2}, TypeError, "integer duration"),
        ({"lag": "36h"}, ValueError, "sub-day"),
    ],
)
def test_argument_validation(
    kwargs: dict[str, Any], err: type[Exception], match: str
) -> None:
    with pytest.raises(err, match=match):
        asof_join(_panel(), VINTAGES, entity="ticker", time="date", **kwargs)


def test_vintage_entity_column_can_differ() -> None:
    v = VINTAGES.rename({"ticker": "entity"})
    out = asof_join(_panel(), v, entity="ticker", time="date", vintage_entity="entity")
    assert out["eps"].to_list()[:8] == EXPECTED_A


def test_float_time_axis_refused() -> None:
    with pytest.raises(TypeError, match="integer, Date or Datetime"):
        asof_join(_panel(), VINTAGES, entity="ticker", time="px")


# --------------------------------------------------------------------------- #
# Integration: the synthetic generator's bitemporal vintages
# --------------------------------------------------------------------------- #
def test_on_synthetic_vintages() -> None:
    """``panelary.synth`` emits ``(entity, event_time, knowledge_time, value,
    revision)`` -- the column roles ``asof_join`` defaults to."""
    synth = pytest.importorskip("panelary.synth")
    try:
        data = synth.generate_panel(
            synth.SynthConfig(n_entities=4, n_periods=30), seed=3
        )
    except Exception as exc:  # pragma: no cover - a sibling module mid-edit
        pytest.skip(f"panelary.synth unavailable: {exc}")
    grid = data.panel.select("entity", "time").rename({"time": "t"})
    vint = data.vintages
    out = asof_join(grid, vint, entity="entity", time="t", values=["value", "revision"])
    assert out["value"].to_list() == _oracle(grid, vint)

    # The instruments perturb every numeric non-key column, so move the integer
    # clocks onto a Date axis (non-numeric) before handing the table over.
    base = D(2020, 1, 1)
    as_date = {
        c: (pl.lit(base) + pl.duration(days=pl.col(c))).cast(pl.Date)
        for c in ("t", "event_time", "knowledge_time")
    }
    grid_d = grid.with_columns(as_date["t"].alias("t"), x=pl.lit(0.0))
    vint_d = vint.drop("revision").with_columns(
        as_date["event_time"].alias("event_time"),
        as_date["knowledge_time"].alias("knowledge_time"),
    )
    op = _join_onto(grid_d, _pit)
    assert_no_lookahead(op, vint_d, entity="entity", time="knowledge_time")
    assert_prefix_invariant(op, vint_d, entity="entity", time="knowledge_time")
