"""Tests for the future-perturbation leak verifier (``panelary.testing``).

The verifier is the keystone that makes Panelary's leak-safety claim checkable:
perturb the future, re-run the op, and assert the past is bit-identical. These
tests (property-based where useful) confirm it *catches* an injected look-ahead
(``shift(-1)``) and *passes* a genuinely walk-forward op (``shift(1).over``).

Two further groups follow the original ones.

* The **missing-value comparison semantics**, which were wrong: NaN-vs-NaN read
  as a difference, so any causal transform that emits NaN (an expanding min-max
  scaler divides 0/0 at each entity's first row) was reported as a leak.
* **Prefix invariance**, the second instrument. Perturbation corrupts future
  *values*, so it is structurally blind to an operator whose output at ``t``
  depends on how many rows follow ``t``. ``count()`` and
  ``ts.count_above().over(entity)`` are pinned here as the pair that passes
  ``assert_no_lookahead`` and fails ``assert_prefix_invariant``.
"""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline
from panelary.core.protocol import PanelTransformer
from panelary.testing import (
    _first_mismatch,
    _nan_mask,
    assert_no_lookahead,
    assert_no_train_test_leak,
    assert_prefix_invariant,
)

ENTITIES = ["A", "B", "C"]


def make_panel(n_times: int, entities=ENTITIES, seed: int = 0) -> PanelFrame:
    """A dense panel with a distinct per-(entity,time) value, integer time axis."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows_e, rows_t, rows_v = [], [], []
    for e in entities:
        for t in range(n_times):
            rows_e.append(e)
            rows_t.append(t)
            rows_v.append(float(rng.standard_normal()))
    df = pl.DataFrame({"entity": rows_e, "time": rows_t, "value": rows_v})
    return PanelFrame(df, entity="entity", time="time")


# --------------------------------------------------------------------------- #
# Expression ops
# --------------------------------------------------------------------------- #
def _safe_expr() -> pl.Expr:
    # Backward-looking lag within each entity -> no look-ahead.
    return pl.col("value").shift(1).over("entity").alias("lag")


def _leaky_expr() -> pl.Expr:
    # Forward-looking lead within each entity -> pure look-ahead.
    return pl.col("value").shift(-1).over("entity").alias("lead")


@settings(max_examples=30, deadline=None)
@given(n_times=st.integers(min_value=4, max_value=25))
def test_safe_expr_passes(n_times):
    panel = make_panel(n_times)
    # Must not raise: shift(1).over(entity) is walk-forward.
    assert_no_lookahead(_safe_expr(), panel)


@settings(max_examples=30, deadline=None)
@given(n_times=st.integers(min_value=4, max_value=25))
def test_leaky_expr_is_detected(n_times):
    panel = make_panel(n_times)
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_lookahead(_leaky_expr(), panel)


def test_trailing_rolling_mean_is_safe():
    panel = make_panel(15)
    op = pl.col("value").rolling_mean(window_size=3).over("entity").alias("roll")
    assert_no_lookahead(op, panel)


def test_centered_rolling_mean_leaks():
    panel = make_panel(15)
    # A centered window peeks forward -> leak.
    op = (
        pl.col("value")
        .rolling_mean(window_size=3, center=True)
        .over("entity")
        .alias("roll_c")
    )
    with pytest.raises(AssertionError):
        assert_no_lookahead(op, panel)


def test_failure_message_names_column_and_keys():
    panel = make_panel(8)
    with pytest.raises(AssertionError) as exc:
        assert_no_lookahead(_leaky_expr(), panel)
    msg = str(exc.value)
    assert "lead" in msg  # the offending column
    assert "entity=" in msg and "time=" in msg  # the offending row keys


# --------------------------------------------------------------------------- #
# Callable ops (frame -> frame)
# --------------------------------------------------------------------------- #
def test_callable_dataframe_op_safe():
    panel = make_panel(12)

    def op(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("value").shift(2).over("entity").alias("lag2"))

    assert_no_lookahead(op, panel)


def test_callable_dataframe_op_leaky():
    panel = make_panel(12)

    def op(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("value").shift(-2).over("entity").alias("lead2"))

    with pytest.raises(AssertionError):
        assert_no_lookahead(op, panel)


def test_callable_panelframe_op_safe():
    panel = make_panel(12)

    def op(pf: PanelFrame) -> PanelFrame:
        return pf.with_columns(pl.col("value").cum_sum().over("entity").alias("cum"))

    # cumulative sum is a trailing aggregation -> no look-ahead.
    assert_no_lookahead(op, panel)


# --------------------------------------------------------------------------- #
# Bare frames + explicit keys
# --------------------------------------------------------------------------- #
def test_accepts_bare_dataframe_with_keys():
    panel = make_panel(10)
    df = panel.collect()
    assert_no_lookahead(_safe_expr(), df, entity="entity", time="time")
    with pytest.raises(AssertionError):
        assert_no_lookahead(_leaky_expr(), df, entity="entity", time="time")


def test_explicit_cut_controls_split():
    panel = make_panel(10)
    # A cut well inside the axis still detects the lead leak at the boundary.
    with pytest.raises(AssertionError):
        assert_no_lookahead(_leaky_expr(), panel, cut=3)
    assert_no_lookahead(_safe_expr(), panel, cut=3)


def test_requires_two_times():
    panel = make_panel(1)
    with pytest.raises(ValueError, match="two distinct"):
        assert_no_lookahead(_safe_expr(), panel)


# --------------------------------------------------------------------------- #
# Train/test-boundary variant
# --------------------------------------------------------------------------- #
def _walk_forward_split(panel: PanelFrame, cut: int, test_len: int):
    train = panel.filter(pl.col("time") < cut)
    test = panel.filter((pl.col("time") >= cut) & (pl.col("time") < cut + test_len))
    return train, test


def test_train_test_leak_safe_op_passes():
    panel = make_panel(20)
    split = _walk_forward_split(panel, cut=10, test_len=4)
    # Backward lag on a train-before-test split cannot see the perturbed test.
    assert_no_train_test_leak(_safe_expr(), panel, split)


def test_train_test_leak_leaky_op_detected():
    panel = make_panel(20)
    split = _walk_forward_split(panel, cut=10, test_len=4)
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_train_test_leak(_leaky_expr(), panel, split)


def test_train_test_leak_rejects_bad_split():
    panel = make_panel(10)
    with pytest.raises(ValueError, match="train, test"):
        assert_no_train_test_leak(_safe_expr(), panel, (panel,))


# --------------------------------------------------------------------------- #
# Pipeline enforcement (_check_leakage wired into the flow)
# --------------------------------------------------------------------------- #
class _Safe(PanelTransformer):
    panel_safe = True
    leakage_safe = True

    def _fit(self, panel):
        pass

    def _transform(self, panel):
        return panel


class _Unsafe(PanelTransformer):
    panel_safe = True
    leakage_safe = False

    def _fit(self, panel):
        pass

    def _transform(self, panel):
        return panel


def test_pipeline_refuses_unsafe_step_across_boundary():
    panel = make_panel(12)
    train = panel.filter(pl.col("time") < 8)
    test = panel.filter(pl.col("time") >= 8)
    pipe = Pipeline([("u", _Unsafe())])
    pipe.fit(train)
    # Transforming the train fold (no boundary) is fine.
    pipe.transform(train)
    # Transforming a test fold crosses the boundary -> refused.
    with pytest.raises(RuntimeError, match="leakage_safe = False"):
        pipe.transform(test)


def test_pipeline_allows_safe_step_across_boundary():
    panel = make_panel(12)
    train = panel.filter(pl.col("time") < 8)
    test = panel.filter(pl.col("time") >= 8)
    pipe = Pipeline([("s", _Safe())])
    pipe.fit(train)
    # A leakage-safe pipeline transforms the test fold without complaint.
    out = pipe.transform(test)
    assert isinstance(out, PanelFrame)


# --------------------------------------------------------------------------- #
# Missing-value comparison semantics
#
# Regression: the numeric comparison used
# ``(bs - ps).abs().fill_null(0.0) > tol``. In Polars null and NaN are distinct
# values, ``fill_null`` does not touch NaN, and NaN orders *above* every float,
# so ``NaN > tol`` is True and a NaN-vs-NaN cell was reported as a leak. Any
# genuinely causal transform that emits NaN -- an expanding min-max scaler
# divides 0/0 at each entity's first observation -- was therefore flagged.
# --------------------------------------------------------------------------- #
NAN = float("nan")

_CMP_SCHEMA = {"entity": pl.String, "time": pl.Int64, "v": pl.Float64}


def _cmp(left: list[float | None], right: list[float | None], tol: float = 1e-9):
    """Run the verifier's cell comparison over two one-column frames."""
    n = len(left)
    base = pl.DataFrame(
        {"entity": ["A"] * n, "time": list(range(n)), "v": left}, schema=_CMP_SCHEMA
    )
    pert = pl.DataFrame(
        {"entity": ["A"] * n, "time": list(range(n)), "v": right}, schema=_CMP_SCHEMA
    )
    return _first_mismatch(base, pert, entity="entity", time="time", tol=tol)


@pytest.mark.parametrize(
    ("left", "right", "differs"),
    [
        ([NAN], [NAN], False),  # NaN vs NaN -> EQUAL (the bug)
        ([NAN], [1.0], True),  # NaN vs number -> mismatch
        ([1.0], [NAN], True),  # number vs NaN -> mismatch
        ([None], [NAN], True),  # null vs NaN -> mismatch (distinct in Polars)
        ([NAN], [None], True),  # NaN vs null -> mismatch
        ([None], [None], False),  # null vs null -> EQUAL
        ([None], [1.0], True),  # null vs number -> mismatch
        ([1.0], [1.0 + 1e-12], False),  # inside tol -> equal (unchanged)
        ([1.0], [1.0 + 1e-3], True),  # outside tol -> mismatch (unchanged)
        ([0.0], [-0.0], False),  # signed zero -> equal
    ],
    ids=[
        "nan-nan-equal",
        "nan-number-differs",
        "number-nan-differs",
        "null-nan-differs",
        "nan-null-differs",
        "null-null-equal",
        "null-number-differs",
        "within-tol-equal",
        "outside-tol-differs",
        "signed-zero-equal",
    ],
)
def test_missing_value_comparison_semantics(left, right, differs):
    assert (_cmp(left, right) is not None) is differs


def test_tol_boundary_semantics_are_unchanged():
    """``tol`` keeps its plain absolute-difference meaning for ordinary floats."""
    assert _cmp([1.0], [1.0 + 5e-10], tol=1e-9) is None
    assert _cmp([1.0], [1.0 + 2e-9], tol=1e-9) is not None
    # And the tolerance is honoured, not hard-coded.
    assert _cmp([1.0], [1.5], tol=1.0) is None
    assert _cmp([1.0], [1.5], tol=0.1) is not None


def test_nan_rows_do_not_suppress_a_real_mismatch_elsewhere():
    """A NaN cell is ignored, but the row after it is still compared."""
    hit = _cmp([NAN, 1.0, 2.0], [NAN, 1.0, 3.0])
    assert hit is not None
    assert hit[0] == "v" and hit[2] == 2  # column, then the offending time


def test_nan_mask_normalises_null_and_non_float_dtypes():
    """``Series.is_nan()`` returns null for nulls and raises on Decimal."""
    assert _nan_mask(pl.Series([NAN, 1.0, None])).to_list() == [True, False, False]
    assert _nan_mask(pl.Series([1, 2, 3])).to_list() == [False, False, False]
    assert _nan_mask(pl.Series([], dtype=pl.Float64)).to_list() == []


def _expanding_minmax() -> pl.Expr:
    """A textbook causal scaler: it emits 0/0 = NaN at each entity's first row."""
    lo = pl.col("value").cum_min().over("entity")
    hi = pl.col("value").cum_max().over("entity")
    return ((pl.col("value") - lo) / (hi - lo)).alias("mm")


def test_causal_expanding_minmax_scaler_is_not_flagged():
    """Regression for the false positive: NaN in both runs is not a leak."""
    panel = make_panel(15)
    first = (
        panel.collect().with_columns(_expanding_minmax()).filter(pl.col("time") == 0)
    )
    assert all(v != v for v in first["mm"].to_list())  # NaN at t=0, as designed
    assert_no_lookahead(_expanding_minmax(), panel)


def test_leaky_op_that_also_emits_nan_is_still_flagged():
    """The NaN allowance must not blunt the instrument on genuine leaks."""
    panel = make_panel(15)
    op = (_expanding_minmax() + pl.col("value").shift(-1).over("entity")).alias(
        "mm_lead"
    )
    with pytest.raises(AssertionError, match="LOOK-AHEAD LEAK DETECTED"):
        assert_no_lookahead(op, panel)


# --------------------------------------------------------------------------- #
# Prefix invariance: the length-dependence instrument
#
# `assert_no_lookahead` perturbs future *values*, so it is structurally blind to
# an operator whose output at `t` depends on how many rows follow `t`. These
# tests pin the complementarity: the two assertions catch different defects.
# --------------------------------------------------------------------------- #
def test_prefix_invariant_causal_expr_passes():
    panel = make_panel(16)
    assert_prefix_invariant(_safe_expr(), panel)
    assert_prefix_invariant(
        pl.col("value").rolling_mean(window_size=3).over("entity").alias("roll"), panel
    )


def test_prefix_invariant_cum_sum_passes():
    panel = make_panel(16)
    op = pl.col("value").cum_sum().over("entity").alias("cum")
    assert_prefix_invariant(op, panel)


def test_prefix_invariant_expanding_minmax_passes():
    """NaN at the first row must compare equal across the two runs here too."""
    assert_prefix_invariant(_expanding_minmax(), make_panel(16))


def test_whole_series_aggregate_broadcast_fails_prefix_invariance():
    panel = make_panel(16)
    op = pl.col("value").mean().over("entity").alias("entity_mean")
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(op, panel)


def test_count_passes_lookahead_but_fails_prefix_invariance():
    """``Agg.Count``: the documented blind spot of the perturbation test.

    Perturbation never changes the number of rows, so a whole-column count is
    bit-identical between the two runs and ``assert_no_lookahead`` sees nothing.
    The leak is real: the value at row 0 moves the moment the series grows.
    """
    panel = make_panel(16)
    op = pl.col("value").count().over("entity").alias("n")
    assert_no_lookahead(op, panel)  # blind
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(op, panel)


def test_count_above_passes_lookahead_but_fails_prefix_invariance():
    """The pair that motivates the new assertion, pinned on a real operator.

    ``ts.count_above`` reports the percentage of the series above its mean, so
    its value at ``t=0`` moves as the entity's history grows (58.33 -> 45.83 for
    12 vs 24 rows) while being bit-identical under value perturbation.
    """
    op = pl.col("value").ts.count_above().over("entity").alias("ca")

    def at_t0(n_times: int) -> list[float]:
        frame = make_panel(n_times).collect().with_columns(op)
        return frame.filter(pl.col("time") == 0)["ca"].to_list()

    assert at_t0(12) != at_t0(24)  # the value depends on the series length

    panel = make_panel(24)
    assert_no_lookahead(op, panel)  # blind: perturbation cannot see it
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(op, panel)


def test_prefix_failure_message_names_column_and_keys():
    panel = make_panel(12)
    op = pl.col("value").sum().over("entity").alias("entity_sum")
    with pytest.raises(AssertionError) as exc:
        assert_prefix_invariant(op, panel)
    msg = str(exc.value)
    assert "entity_sum" in msg  # the offending column
    assert "entity=" in msg and "time=" in msg  # the offending row keys
    assert "assert_no_lookahead" in msg  # points at the complementary check


def test_several_cuts_catch_what_a_single_cut_would_miss():
    """Why the default sweeps the axis instead of cutting once."""
    panel = make_panel(12)
    n = pl.len().over("entity").cast(pl.Int64)
    op = pl.when(n < 5).then(n).otherwise(pl.lit(0, dtype=pl.Int64)).alias("sneaky")
    # The median cut alone (a 6-row prefix) agrees with the full panel...
    assert_prefix_invariant(op, panel, cut=5)
    # ...but the default sweep also probes a short prefix, which does not.
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(op, panel)


def test_prefix_invariant_callable_and_bare_frame_conventions():
    panel = make_panel(12)

    def safe(pf: PanelFrame) -> PanelFrame:
        return pf.with_columns(pl.col("value").cum_sum().over("entity").alias("cum"))

    def leaky(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("value").max().over("entity").alias("hi"))

    assert_prefix_invariant(safe, panel)
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(leaky, panel)

    df = panel.collect()
    assert_prefix_invariant(_safe_expr(), df, entity="entity", time="time")
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE VIOLATION"):
        assert_prefix_invariant(
            pl.col("value").min().over("entity").alias("lo"),
            df,
            entity="entity",
            time="time",
        )


def test_prefix_invariant_rejects_degenerate_panels_and_cuts():
    with pytest.raises(ValueError, match="two distinct"):
        assert_prefix_invariant(_safe_expr(), make_panel(1))
    panel = make_panel(10)
    with pytest.raises(ValueError, match="truncates nothing"):
        assert_prefix_invariant(_safe_expr(), panel, cut=9)
    with pytest.raises(ValueError, match="no rows to compare"):
        assert_prefix_invariant(_safe_expr(), panel, cut=-1)


@settings(max_examples=20, deadline=None)
@given(n_times=st.integers(min_value=4, max_value=25))
def test_prefix_invariance_holds_for_walk_forward_ops(n_times):
    panel = make_panel(n_times)
    assert_prefix_invariant(_safe_expr(), panel)
    assert_prefix_invariant(
        pl.col("value").cum_sum().over("entity").alias("cum"), panel
    )
