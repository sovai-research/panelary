"""Property-based and golden tests for PanelFrame invariants.

These exercise the *correctness-by-construction* core in
``panelary.core.panel_frame`` only. They require no compiled Rust
extension (pure-Python Phase-1 core).

The last section is a bug report with assertions: ``panel_safe`` says
"``.over(entity_col)`` on a **time-sorted** panel", and until the ordering
affordance below existed, nothing anywhere checked the second half of that
sentence.
"""

from __future__ import annotations

import time as _time
import warnings

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from panelary.core.panel_frame import PanelFrame, PanelOrderWarning, as_panel

# --------------------------------------------------------------------------- #
# Hypothesis strategies for small synthetic panels
# --------------------------------------------------------------------------- #
ENTITIES = ["A", "B", "C"]


@st.composite
def panel_rows(draw):
    """Draw a small panel as a list of (entity, time, value) rows.

    Each entity may appear with a different (possibly unsorted, possibly
    sparse) set of integer times. Keys are kept unique per (entity, time).
    """
    n = draw(st.integers(min_value=1, max_value=24))
    rows = []
    seen = set()
    for _ in range(n):
        ent = draw(st.sampled_from(ENTITIES))
        t = draw(st.integers(min_value=0, max_value=15))
        if (ent, t) in seen:
            continue
        seen.add((ent, t))
        val = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
        rows.append((ent, t, val))
    # Guarantee at least one row.
    if not rows:
        rows.append(("A", 0, 0.0))
    return rows


def _df_from_rows(rows):
    return pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "value": [r[2] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# Construction / schema validation
# --------------------------------------------------------------------------- #
def test_construction_rejects_missing_entity():
    df = pl.DataFrame({"time": [1, 2], "value": [0.1, 0.2]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="entity", time="time")
    assert "entity" in str(exc.value)


def test_construction_rejects_missing_time():
    df = pl.DataFrame({"entity": ["A", "B"], "value": [0.1, 0.2]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="entity", time="time")
    assert "time" in str(exc.value)


def test_construction_rejects_same_entity_and_time():
    df = pl.DataFrame({"x": ["A"], "value": [0.1]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="x", time="x")
    assert "different columns" in str(exc.value)


def test_construction_rejects_non_orderable_time():
    df = pl.DataFrame({"entity": ["A"], "time": ["not-a-time"], "value": [0.1]})
    with pytest.raises(ValueError) as exc:
        PanelFrame(df, entity="entity", time="time")
    assert "orderable" in str(exc.value)


def test_feature_cols_excludes_keys():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "f1": [0.1], "f2": [0.2]})
    pf = PanelFrame(df, entity="entity", time="time")
    assert set(pf.feature_cols) == {"f1", "f2"}
    assert "entity" not in pf.feature_cols
    assert "time" not in pf.feature_cols


def test_as_panel_infers_keys_from_first_two_columns():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [0.1]})
    pf = as_panel(df)
    assert pf.entity_col == "entity"
    assert pf.time_col == "time"


def test_as_panel_returns_panelframe_unchanged():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [0.1]})
    pf = PanelFrame(df, entity="entity", time="time")
    assert as_panel(pf) is pf


# --------------------------------------------------------------------------- #
# sort_panel: per-entity time-monotonic ordering
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(rows=panel_rows())
def test_sort_panel_is_time_monotonic_per_entity(rows):
    df = _df_from_rows(rows)
    pf = PanelFrame(df, entity="entity", time="time")
    sorted_pf = pf.sort_panel()
    # is_sorted_per_entity inspects current row order; after sort it must hold.
    assert sorted_pf.is_sorted_per_entity()

    # Explicitly verify monotonicity per entity on the materialised frame.
    out = sorted_pf.collect()
    for ent in out["entity"].unique().to_list():
        times = out.filter(pl.col("entity") == ent)["time"].to_list()
        assert times == sorted(times)


@settings(max_examples=50, deadline=None)
@given(rows=panel_rows())
def test_sort_panel_descending_time_within_entity(rows):
    df = _df_from_rows(rows)
    pf = PanelFrame(df, entity="entity", time="time")
    out = pf.sort_panel(descending=True).collect()
    for ent in out["entity"].unique().to_list():
        times = out.filter(pl.col("entity") == ent)["time"].to_list()
        assert times == sorted(times, reverse=True)


def test_is_sorted_per_entity_detects_unsorted():
    # B is out of time order within its entity.
    df = pl.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "time": [1, 2, 5, 3],
            "value": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pf = PanelFrame(df, entity="entity", time="time")
    assert not pf.is_sorted_per_entity()
    assert pf.sort_panel().is_sorted_per_entity()


# --------------------------------------------------------------------------- #
# assert_unique_keys
# --------------------------------------------------------------------------- #
def test_assert_unique_keys_passes_for_unique():
    df = pl.DataFrame(
        {"entity": ["A", "A", "B"], "time": [1, 2, 1], "value": [0.0, 0.0, 0.0]}
    )
    pf = PanelFrame(df, entity="entity", time="time")
    assert pf.assert_unique_keys() is pf


def test_assert_unique_keys_raises_on_duplicates():
    df = pl.DataFrame({"entity": ["A", "A"], "time": [1, 1], "value": [0.0, 1.0]})
    pf = PanelFrame(df, entity="entity", time="time")
    with pytest.raises(ValueError) as exc:
        pf.assert_unique_keys()
    assert "duplicated" in str(exc.value)


@settings(max_examples=50, deadline=None)
@given(rows=panel_rows())
def test_assert_unique_keys_property(rows):
    # `rows` is constructed with unique (entity, time) keys -> must pass.
    df = _df_from_rows(rows)
    pf = PanelFrame(df, entity="entity", time="time")
    assert pf.assert_unique_keys() is pf

    # Duplicating any single row must trigger the raise.
    dup = pl.concat([df, df.head(1)])
    pf_dup = PanelFrame(dup, entity="entity", time="time")
    with pytest.raises(ValueError):
        pf_dup.assert_unique_keys()


# --------------------------------------------------------------------------- #
# with_columns: preserves keys + laziness
# --------------------------------------------------------------------------- #
def test_with_columns_preserves_keys_and_laziness():
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [2.0]})
    pf = PanelFrame(df, entity="entity", time="time")
    out = pf.with_columns((pl.col("value") * 2).alias("doubled"))
    assert isinstance(out, PanelFrame)
    assert out.entity_col == "entity"
    assert out.time_col == "time"
    assert "doubled" in out.columns
    # Stays lazy: the underlying handle is a LazyFrame, not materialised.
    assert isinstance(out.lazy(), pl.LazyFrame)


def test_with_columns_rejects_dropping_a_key():
    # Re-wrap a frame missing the 'time' key (validate=False trusts the caller),
    # then any with_columns must trip the cheap key-survival guard.
    df = pl.DataFrame({"entity": ["A"], "time": [1], "value": [2.0]})
    pf_broken = PanelFrame(
        df.lazy().drop("time"), entity="entity", time="time", validate=False
    )
    with pytest.raises(ValueError) as exc:
        pf_broken.with_columns(pl.col("value").alias("v2"))
    assert "must not drop the panel keys" in str(exc.value)


# --------------------------------------------------------------------------- #
# over_entity: grouping never mixes values across entities
# --------------------------------------------------------------------------- #
@st.composite
def dense_panel(draw):
    """Draw a panel where every entity has a contiguous dense time axis.

    This makes per-entity cumulative/shift comparisons well-defined regardless
    of row order.
    """
    ents = draw(
        st.lists(st.sampled_from(ENTITIES), min_size=1, max_size=3, unique=True)
    )
    lengths = {e: draw(st.integers(min_value=1, max_value=8)) for e in ents}
    rows = []
    for e in ents:
        for t in range(lengths[e]):
            val = draw(st.floats(min_value=-10, max_value=10, allow_nan=False))
            rows.append((e, t, val))
    return rows


@settings(max_examples=50, deadline=None)
@given(rows=dense_panel(), seed=st.integers(min_value=0, max_value=10_000))
def test_over_entity_cumsum_invariant_to_row_order(rows, seed):
    """A per-entity cumulative sum computed via .over(entity) must be invariant
    to the input row order (after sorting the panel), and must never mix values
    across entities."""
    df = _df_from_rows(rows)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    def cumsum_per_entity(frame):
        pf = PanelFrame(frame, entity="entity", time="time").sort_panel()
        out = pf.with_columns(
            pl.col("value").cum_sum().over(pf.over_entity()).alias("cs")
        )
        return out.collect().sort(["entity", "time"])

    ref = cumsum_per_entity(df)
    got = cumsum_per_entity(shuffled)
    assert ref.equals(got)

    # Cross-check: the cumulative sum for each entity equals an independent
    # per-entity computation (i.e. no leakage of values across entities).
    for ent in ref["entity"].unique().to_list():
        sub = ref.filter(pl.col("entity") == ent).sort("time")
        expected = sub["value"].cum_sum().to_list()
        assert sub["cs"].to_list() == pytest.approx(expected)


@settings(max_examples=50, deadline=None)
@given(rows=dense_panel(), seed=st.integers(min_value=0, max_value=10_000))
def test_over_entity_shift_does_not_mix_entities(rows, seed):
    """A per-entity lag (shift(1).over(entity)) must produce a null for each
    entity's first observation and never borrow a value from another entity."""
    df = _df_from_rows(rows)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    pf = PanelFrame(shuffled, entity="entity", time="time").sort_panel()
    out = pf.with_columns(
        pl.col("value").shift(1).over(pf.over_entity()).alias("lag")
    ).collect()

    for ent in out["entity"].unique().to_list():
        sub = out.filter(pl.col("entity") == ent).sort("time")
        lags = sub["lag"].to_list()
        vals = sub["value"].to_list()
        # First lag of each entity is null (no borrowing across entity bounds).
        assert lags[0] is None
        # Remaining lags equal the entity's own previous value.
        for i in range(1, len(vals)):
            assert lags[i] == pytest.approx(vals[i - 1])


# --------------------------------------------------------------------------- #
# The bug report: `.over(entity)` has no notion of time
# --------------------------------------------------------------------------- #
#
# `panel_safe` reads: "within-entity operations stay inside their entity and run
# in time order: `.over(entity_col)` on a **time-sorted** panel". Polars honours
# the first half and knows nothing of the second: `.over("entity")` serialises
# with `order_by: null` and walks each entity's rows in *frame* order. On a panel
# that arrived shuffled, a shift or a rolling window is therefore not an error --
# it is a plausible wrong number, and in the worst case it reaches into the
# future. These tests pin that down before testing the affordance that catches
# it.
_N_PERIODS = 40
_ENTITY_IDS = ("A", "B", "C")


def _walk_panel(seed: int = 20240917) -> pl.DataFrame:
    """A tidy, sorted 3-entity panel: `value` is a seeded random walk."""
    rng = np.random.default_rng(seed)
    n_ent = len(_ENTITY_IDS)
    return pl.DataFrame(
        {
            "entity": np.repeat(np.array(_ENTITY_IDS), _N_PERIODS),
            "time": np.tile(np.arange(_N_PERIODS), n_ent),
            "value": np.cumsum(rng.normal(size=n_ent * _N_PERIODS)),
        }
    )


def _shuffled(df: pl.DataFrame, seed: int = 7) -> pl.DataFrame:
    """The same rows, in an order nobody promised anything about."""
    rng = np.random.default_rng(seed)
    return df[rng.permutation(df.height)]


def _lag_expr() -> pl.Expr:
    return pl.col("value").shift(1).over("entity").alias("lag")


def _roll_expr() -> pl.Expr:
    return pl.col("value").rolling_mean(window_size=3).over("entity").alias("roll3")


def test_shuffled_panel_gives_a_different_and_wrong_lag():
    """The same `shift(1).over(entity)`, two row orders, two different answers.

    The sorted answer is the true lag. The shuffled answer is not an error, is
    not null, and is not the same number -- it is whichever row happened to sit
    above this one in the frame.
    """
    tidy = _walk_panel()
    shuffled = _shuffled(tidy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanelOrderWarning)
        right = (
            PanelFrame(tidy, entity="entity", time="time")
            .with_columns(_lag_expr())
            .collect()
            .sort(["entity", "time"])
        )
        wrong = (
            PanelFrame(shuffled, entity="entity", time="time")
            .with_columns(_lag_expr())
            .collect()
            .sort(["entity", "time"])
        )

    # Same rows, same keys, same expression -- and a different column of numbers.
    assert right.drop("lag").equals(wrong.drop("lag"))
    assert not right["lag"].equals(wrong["lag"])

    # The sorted lag is the real thing: each entity's own previous value.
    for ent in _ENTITY_IDS:
        sub = right.filter(pl.col("entity") == ent)
        assert sub["lag"].to_list()[1:] == pytest.approx(sub["value"].to_list()[:-1])

    # The shuffled one disagrees nearly everywhere, and quietly.
    disagreements = (right["lag"] != wrong["lag"]).sum()
    assert disagreements > 100, (
        f"expected the shuffled lag to be wrong on most rows, got "
        f"{disagreements} disagreements out of {right.height}"
    )


def test_shuffled_panel_lag_reaches_into_the_future():
    """Worse than wrong: on a shuffled panel the "lag" is sometimes a *future* row.

    Lag the time column with the same within-entity expression and look at where
    the value came from. Sorted, it always comes from the past. Shuffled, it
    comes from wherever -- which is look-ahead leakage produced by code that
    looks perfectly `panel_safe`.
    """
    lagged_time = pl.col("time").shift(1).over("entity").alias("lag_time")
    tidy = _walk_panel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanelOrderWarning)
        right = (
            PanelFrame(tidy, entity="entity", time="time")
            .with_columns(lagged_time)
            .collect()
        )
        wrong = (
            PanelFrame(_shuffled(tidy), entity="entity", time="time")
            .with_columns(lagged_time)
            .collect()
        )

    from_the_future = pl.col("lag_time") > pl.col("time")
    assert right.filter(from_the_future).height == 0
    leaked = wrong.filter(from_the_future).height
    assert leaked > 0, "expected the shuffled lag to read future rows"
    # Roughly half of a random permutation's predecessors are in the future.
    assert leaked > wrong.height // 4


def test_shuffled_panel_gives_a_different_and_wrong_rolling_mean():
    """A rolling mean averages three arbitrary observations of the entity."""
    tidy = _walk_panel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanelOrderWarning)
        right = (
            PanelFrame(tidy, entity="entity", time="time")
            .with_columns(_roll_expr())
            .collect()
            .sort(["entity", "time"])
        )
        wrong = (
            PanelFrame(_shuffled(tidy), entity="entity", time="time")
            .with_columns(_roll_expr())
            .collect()
            .sort(["entity", "time"])
        )

    assert not right["roll3"].equals(wrong["roll3"])
    for ent in _ENTITY_IDS:
        sub = right.filter(pl.col("entity") == ent)
        values = sub["value"].to_list()
        expected = [
            None if i < 2 else sum(values[i - 2 : i + 1]) / 3.0
            for i in range(len(values))
        ]
        got = sub["roll3"].to_list()
        assert got[:2] == [None, None]
        assert got[2:] == pytest.approx(expected[2:])


# --------------------------------------------------------------------------- #
# The affordance: sortedness is tracked, checkable, and warned about
# --------------------------------------------------------------------------- #
def test_default_construction_knows_nothing_and_collects_nothing(monkeypatch):
    """Today's behaviour is untouched: no flag, no collect, no complaint.

    The class is lazy on purpose -- `is_sorted_per_entity()` materialises, so it
    cannot run on construction. The flag starts at "unknown", which is exactly
    what every panel built before this existed is.
    """
    collects = _count_collects(monkeypatch)
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    assert pf.sortedness == "unknown"
    assert collects["n"] == 0


def test_check_sorted_catches_the_shuffled_panel():
    """The opt-in construction check refuses the panel the bug report used."""
    shuffled = _shuffled(_walk_panel())
    with pytest.raises(ValueError) as exc:
        PanelFrame(shuffled, entity="entity", time="time", check_sorted=True)
    message = str(exc.value)
    assert "not time-sorted" in message
    assert "sort_panel" in message
    # The error names offenders, like assert_unique_keys does.
    assert "First offenders" in message


def test_assert_sorted_catches_it_and_passes_after_sorting():
    shuffled = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with pytest.raises(ValueError):
        shuffled.assert_sorted()
    assert shuffled.sortedness == "unsorted"

    ordered = shuffled.sort_panel()
    assert ordered.assert_sorted() is ordered
    assert ordered.sortedness == "panel"


def test_within_entity_op_on_unknown_order_warns_once():
    """A shift over the entity key on a panel of unknown order warns, once."""
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with pytest.warns(PanelOrderWarning, match="order_by"):
        out = pf.with_columns(_lag_expr())

    # Warned, not raised: the (wrong) answer is still produced, because
    # refusing would break every caller that has been living with this.
    assert out.collect().height == pf.collect().height

    # Once per frame: the second call on the same frame is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.with_columns(_roll_expr())


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda pf: pf.with_columns(_lag_expr()), id="with_columns"),
        pytest.param(lambda pf: pf.with_columns([_lag_expr()]), id="with_columns-list"),
        pytest.param(lambda pf: pf.with_columns(lag=_lag_expr()), id="with_columns-kw"),
        pytest.param(lambda pf: pf.select(_lag_expr()), id="select"),
        pytest.param(lambda pf: pf.filter(pl.col("value") > _lag_expr()), id="filter"),
        pytest.param(lambda pf: pf.over_entity(), id="over_entity"),
        pytest.param(lambda pf: pf.group_by_entity(), id="group_by_entity"),
    ],
)
def test_every_within_entity_door_warns(operation):
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with pytest.warns(PanelOrderWarning):
        operation(pf)


def test_cross_sectional_and_plain_work_never_warns():
    """`.over(time)` and ordinary column maths are not within-entity work."""
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.with_columns((pl.col("value") * 2).alias("doubled"))
        pf.with_columns(pl.col("value").rank().over("time").alias("xs_rank"))
        pf.select(pl.col("value").mean().over("time").alias("xs_mean"))
        pf.filter(pl.col("value") > 0)


def test_order_free_within_entity_work_never_warns():
    """A per-entity mean is the same number in any row order -- so, no alarm.

    Entity demeaning is the fixed-effects idiom and is everywhere in panel work.
    Warning about it would be false, and a warning that cries wolf on the most
    common operation in the library is a warning everyone filters out.
    """
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.with_columns(
            (pl.col("value") - pl.col("value").mean().over("entity")).alias("demeaned")
        )
        pf.with_columns(pl.col("value").std().over("entity").alias("vol"))
        pf.with_columns(pl.col("value").count().over("entity").alias("n"))
        pf.with_columns(pl.col("value").max().over("entity").alias("hi"))


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(pl.col("value").shift(1), id="shift"),
        pytest.param(pl.col("value").diff(), id="diff"),
        pytest.param(pl.col("value").pct_change(), id="pct_change"),
        pytest.param(pl.col("value").cum_sum(), id="cum_sum"),
        pytest.param(pl.col("value").rolling_mean(3), id="rolling_mean"),
        pytest.param(pl.col("value").ewm_mean(alpha=0.5), id="ewm_mean"),
        pytest.param(pl.col("value").forward_fill(), id="forward_fill"),
        pytest.param(pl.col("value").last(), id="last"),
        pytest.param(pl.col("value").arg_max(), id="arg_max"),
    ],
)
def test_order_dependent_operations_are_recognised(expr):
    """The operations whose answer a shuffle changes all trip the guard."""
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with pytest.warns(PanelOrderWarning):
        pf.with_columns(expr.over("entity").alias("out"))


def test_within_entity_is_correct_on_a_shuffled_panel():
    """The fix, not just the alarm: `within_entity` pins `order_by` into the window.

    Same shuffled frame, same shift -- but the expression now carries the time
    axis, so polars orders each entity before evaluating and scatters the
    results back. No sort, no warning, no wrong number.
    """
    tidy = _walk_panel()
    shuffled = _shuffled(tidy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanelOrderWarning)
        right = (
            PanelFrame(tidy, entity="entity", time="time")
            .with_columns(_lag_expr())
            .collect()
            .sort(["entity", "time"])
        )

    pf = PanelFrame(shuffled, entity="entity", time="time")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the safe form must not warn
        fixed = (
            pf.with_columns(pf.within_entity(pl.col("value").shift(1)).alias("lag"))
            .collect()
            .sort(["entity", "time"])
        )

    assert fixed["lag"].to_list() == pytest.approx(right["lag"].to_list(), nan_ok=True)
    # And the caller's row order is untouched -- nothing was sorted behind them.
    assert pf.collect()["time"].to_list() == shuffled["time"].to_list()


def test_mark_sorted_is_an_o1_promise(monkeypatch):
    """`mark_sorted()` silences the warning without looking at the data."""
    collects = _count_collects(monkeypatch)
    pf = PanelFrame(_walk_panel(), entity="entity", time="time").mark_sorted()
    assert pf.sortedness == "panel"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.with_columns(_lag_expr())
    assert collects["n"] == 0
    assert pf.mark_sorted() is pf


def test_sortedness_distinguishes_interleaved_from_key_ordered():
    """A frame sorted by time alone is `panel_safe`, but is not `(entity, time)`.

    The distinction is what keeps `sort_panel()`'s no-op honest: this panel
    satisfies the correctness precondition, so it must not warn, but a real sort
    would still move its rows, so the sort must not be skipped.
    """
    interleaved = _walk_panel().sort("time")
    pf = PanelFrame(interleaved, entity="entity", time="time")
    assert pf.is_sorted_per_entity()
    assert pf.sortedness == "time"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.with_columns(_lag_expr())

    resorted = pf.sort_panel()
    assert resorted is not pf
    assert resorted.collect()["entity"].to_list() == sorted(
        interleaved["entity"].to_list()
    )


def test_is_sorted_per_entity_caches_its_answer(monkeypatch):
    pf = PanelFrame(_walk_panel(), entity="entity", time="time")
    collects = _count_collects(monkeypatch)
    assert pf.is_sorted_per_entity()
    after_first = collects["n"]
    assert after_first >= 1
    assert pf.is_sorted_per_entity()
    assert collects["n"] == after_first, "the second call must not materialise"


# --------------------------------------------------------------------------- #
# A sorted panel is unaffected, and pays nothing
# --------------------------------------------------------------------------- #
def _count_collects(monkeypatch) -> dict[str, int]:
    """Count real materialisations (`LazyFrame.collect`), not schema lookups."""
    calls = {"n": 0}
    original = pl.LazyFrame.collect

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counting)
    return calls


def _count_expr_renders(monkeypatch) -> dict[str, int]:
    """Count expression renderings -- the only per-call work the guard adds."""
    calls = {"n": 0}
    original = pl.Expr.__str__

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(pl.Expr, "__str__", counting)
    return calls


def test_known_sorted_panel_does_no_extra_work(monkeypatch):
    """On the happy path the guard is one set-membership test and nothing else.

    Deterministic rather than wall-clock: zero materialisations, and zero
    expression renderings, because the flag short-circuits before the sniffing
    that an unknown-order panel pays for.
    """
    pf = PanelFrame(_walk_panel(), entity="entity", time="time").sort_panel()
    collects = _count_collects(monkeypatch)
    renders = _count_expr_renders(monkeypatch)

    pf.sort_panel()
    pf.with_columns(_lag_expr())
    pf.select(_lag_expr())
    pf.filter(pl.col("value") > 0)
    pf.over_entity()
    pf.is_sorted_per_entity()

    assert collects["n"] == 0
    assert renders["n"] == 0


def test_unknown_order_panel_is_what_pays_for_the_sniffing(monkeypatch):
    """The counterpart: an unknown-order frame renders expressions, once."""
    pf = PanelFrame(_walk_panel(), entity="entity", time="time")
    renders = _count_expr_renders(monkeypatch)
    with pytest.warns(PanelOrderWarning):
        pf.with_columns(_lag_expr())
    assert renders["n"] >= 1


def test_sorted_panel_is_not_measurably_slower_than_raw_polars():
    """The wrapper's per-call overhead stays in the noise of what it already did.

    `with_columns` has always called `collect_schema()` (tens of microseconds);
    the guard adds a frozenset lookup and one `meta.output_name()`. Generous
    bound, `min` of three runs: this catches an accidental collect or a
    per-row scan, not a percent.
    """
    df = _walk_panel()
    pf = PanelFrame(df, entity="entity", time="time").sort_panel()
    lf = df.lazy()
    reps, iterations = 3, 300

    def _time_it(fn) -> float:
        best = float("inf")
        for _ in range(reps):
            start = _time.perf_counter()
            for _ in range(iterations):
                fn()
            best = min(best, _time.perf_counter() - start)
        return best

    # What the method has always done, minus the panel bookkeeping.
    baseline = _time_it(lambda: lf.with_columns(_lag_expr()).collect_schema().names())
    guarded = _time_it(lambda: pf.with_columns(_lag_expr()))
    assert guarded < baseline * 3.0, (
        f"PanelFrame.with_columns on a known-sorted panel took {guarded:.4f}s "
        f"vs {baseline:.4f}s for the underlying polars work"
    )


# --------------------------------------------------------------------------- #
# sort_panel: idempotent, and a genuine no-op when it already holds
# --------------------------------------------------------------------------- #
def test_sort_panel_on_a_sorted_panel_is_a_no_op():
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    once = pf.sort_panel()
    twice = once.sort_panel()

    # Not merely equal: the same object, with no second sort in the plan.
    assert twice is once
    assert twice.lazy() is once.lazy()


def test_sort_panel_is_idempotent_in_data():
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    once = pf.sort_panel().collect()
    twice = pf.sort_panel().sort_panel().collect()
    thrice = PanelFrame(once, entity="entity", time="time").sort_panel().collect()
    assert once.equals(twice)
    assert once.equals(thrice)


def test_sort_panel_no_op_still_returns_sorted_data():
    """The skipped sort must not change the answer -- only the work."""
    tidy = _walk_panel()
    promised = PanelFrame(tidy, entity="entity", time="time", assume_sorted=True)
    assert promised.sort_panel() is promised
    assert (
        promised.sort_panel()
        .collect()
        .equals(PanelFrame(tidy, entity="entity", time="time").sort_panel().collect())
    )


def test_descending_sort_is_not_recorded_as_sorted():
    """Reverse time order is not the `panel_safe` precondition, so it stays unknown."""
    pf = PanelFrame(_walk_panel(), entity="entity", time="time").sort_panel(
        descending=True
    )
    assert pf.sortedness == "unknown"
    with pytest.warns(PanelOrderWarning):
        pf.with_columns(_lag_expr())


# --------------------------------------------------------------------------- #
# The flag travels with order-preserving operations, and only those
# --------------------------------------------------------------------------- #
def test_order_survives_order_preserving_operations():
    pf = PanelFrame(_walk_panel(), entity="entity", time="time").sort_panel()
    assert pf.with_columns((pl.col("value") * 2).alias("d")).sortedness == "panel"
    assert pf.select(pl.col("value")).sortedness == "panel"
    assert pf.filter(pl.col("value") > 0).sortedness == "panel"
    assert pf.with_columns("value").sortedness == "panel"


def test_rewriting_a_key_column_forgets_the_order():
    """Rewrite `time` and the promise no longer means anything -- so drop it."""
    pf = PanelFrame(_walk_panel(), entity="entity", time="time").sort_panel()
    assert pf.with_columns(time=-pl.col("time")).sortedness == "unknown"
    assert pf.with_columns((-pl.col("time")).alias("time")).sortedness == "unknown"
    assert pf.with_columns(entity=pl.lit("Z")).sortedness == "unknown"
    # Undeterminable output names are treated as a rewrite, not waved through.
    assert pf.with_columns(pl.all().shuffle(seed=0)).sortedness == "unknown"
    # ... and the resulting frame warns again, as it should.
    with pytest.warns(PanelOrderWarning):
        pf.with_columns(time=-pl.col("time")).with_columns(_lag_expr())


def test_unsorted_state_survives_a_projection():
    pf = PanelFrame(_shuffled(_walk_panel()), entity="entity", time="time")
    with pytest.raises(ValueError):
        pf.assert_sorted()
    assert pf.filter(pl.col("value") > -1e9).sortedness == "unsorted"


# --------------------------------------------------------------------------- #
# Back-compatibility: every existing signature still means what it meant
# --------------------------------------------------------------------------- #
def test_existing_signatures_are_unchanged():
    df = _walk_panel()
    pf = PanelFrame(df, entity="entity", time="time")
    assert PanelFrame(df.lazy(), "entity", "time", validate=False).sortedness == (
        "unknown"
    )
    assert as_panel(df).sortedness == "unknown"
    assert as_panel(df, "entity", "time").sortedness == "unknown"
    assert as_panel(pf) is pf
    assert as_panel(df, "entity", "time", assume_sorted=True).sortedness == "panel"
    assert as_panel(pf, assume_sorted=True).sortedness == "panel"
    assert as_panel(pf.sort_panel(), check_sorted=True).sortedness == "panel"
    # is_sorted_per_entity keeps its old meaning: current row order, not sorted.
    assert not PanelFrame(
        _shuffled(df), entity="entity", time="time"
    ).is_sorted_per_entity()


def test_check_sorted_beats_assume_sorted():
    """A measurement supersedes a promise, including a false one."""
    shuffled = _shuffled(_walk_panel())
    with pytest.raises(ValueError):
        PanelFrame(
            shuffled,
            entity="entity",
            time="time",
            assume_sorted=True,
            check_sorted=True,
        )
