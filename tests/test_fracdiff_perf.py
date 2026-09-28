"""Performance + correctness regression tests for fractional differencing.

These tests guard the convolution rewrite of the shared frac-diff kernel
(:func:`panelary._internal._ffd.frac_diff_expr`). The builder no longer expands a
``pl.sum_horizontal`` of ``width`` lagged ``.shift()`` terms (an ``O(n * width)``
expression tree, ~1458 nodes for ``d=0.4`` at the old ``1e-5`` default); it now
applies a causal FIR convolution (``numpy.convolve``) per entity inside a
``map_batches`` UDF grouped ``.over(entity)``.

The tests assert, in order:

* **(a) parity** -- the convolution matches a direct weight-dot-product
  reference to ``atol=1e-9`` on every row that has a full trailing window, for
  several ``(d, threshold)`` pairs;
* **(b) correctness** -- a default ``d=0.4`` call on an 800-row-per-entity panel
  returns a majority of non-null values (regression for the old all-null footgun
  where the ``1e-5`` kernel was longer than the series), and a row is null
  exactly when it lacks a full trailing window -- never a truncated-kernel
  partial value, which would break prefix invariance (hard invariant 1);
* **(c) perf** -- a 0.5M-row panel completes far under the old multi-second cost;
* **(d) causality** -- appending future rows never changes an earlier output.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

import panelary.namespaces  # noqa: F401  (registers the .panel namespace)
from panelary._internal._ffd import DEFAULT_THRESHOLD, ffd_weights, frac_diff_expr


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_panel(n_entities: int, n_time: int, *, seed: int = 20260904) -> pl.DataFrame:
    """Long ``(entity, time, px)`` panel of per-entity random-walk price paths."""
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n_entities), n_time)
    t = np.tile(np.arange(n_time, dtype=np.int64), n_entities)
    steps = rng.standard_normal(n_entities * n_time) * 0.5
    px = 100.0 + steps.reshape(n_entities, n_time).cumsum(axis=1).reshape(-1)
    return pl.DataFrame({"entity": ent, "time": t, "px": px}).sort("entity", "time")


def _reference_fracdiff(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Direct causal weight-dot-product reference (oldest-to-newest weights).

    ``out[t] = sum_j weights[j] * x[t - (width-1) + j]`` for every ``t`` with a
    full trailing window; ``NaN`` for the leading warm-up. This is the textbook
    definition the fast convolution must reproduce.
    """
    n = x.shape[0]
    width = weights.shape[0]
    out = np.full(n, np.nan)
    for t in range(width - 1, n):
        out[t] = float(np.dot(weights, x[t - width + 1 : t + 1]))
    return out


def _panel_out(df: pl.DataFrame, d: float, thr: float) -> np.ndarray:
    return (
        df.with_columns(
            pl.col("px").panel.frac_diff(d, threshold=thr).over("entity").alias("fd")
        )
        .get_column("fd")
        .to_numpy()
    )


# --------------------------------------------------------------------------- #
# (a) parity vs a direct weight-dot-product reference
# --------------------------------------------------------------------------- #
PARITY_CASES = [
    (0.3, 1e-3),
    (0.4, 5e-4),
    (0.4, 1e-4),
    (0.5, 1e-4),
    (0.9, 1e-4),
    (1.0, 1e-3),
]


@pytest.mark.parametrize(("d", "thr"), PARITY_CASES)
def test_convolution_matches_direct_dot_product(d: float, thr: float) -> None:
    df = _make_panel(n_entities=4, n_time=600)
    weights = ffd_weights(d, thr)
    got = _panel_out(df, d, thr)

    # Build the per-entity reference over the SAME sorted layout.
    ref = np.empty_like(got)
    for code in df.get_column("entity").unique().to_list():
        mask = (df.get_column("entity") == code).to_numpy()
        ref[mask] = _reference_fracdiff(df.get_column("px").to_numpy()[mask], weights)

    # Every row the reference computes (full trailing window) must match exactly;
    # null positions must line up.
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    valid = ~np.isnan(ref)
    assert valid.any(), f"reference produced no valid rows for d={d}, thr={thr}"
    np.testing.assert_allclose(got[valid], ref[valid], atol=1e-9, rtol=0.0)


# --------------------------------------------------------------------------- #
# (b) correctness: the old all-null footgun must be gone
# --------------------------------------------------------------------------- #
def test_default_d04_returns_mostly_nonnull_on_800_rows() -> None:
    """Regression: with the old ``1e-5`` default the ``d=0.4`` kernel (width 1458)
    exceeded an 800-row series and nulled *every* row. The new default returns a
    majority of non-null values."""
    df = _make_panel(n_entities=3, n_time=800)
    out = df.panel.frac_diff("px", d=0.4, over="entity", alias="fd")
    fd = out.get_column("fd")

    n_valid = int(fd.is_not_null().sum())
    assert n_valid > 0, "d=0.4 default call returned all-null (the regressed bug)"
    # Majority non-null on an 800-row/entity panel at the default threshold.
    assert n_valid > df.height // 2

    # The default kernel must fit comfortably inside an 800-row series.
    width = ffd_weights(0.4, DEFAULT_THRESHOLD).shape[0]
    assert width < 800


def _fd_last_rows(n: int, weights: np.ndarray) -> np.ndarray:
    """frac-diff of the series ``1..n`` for one entity, with explicit weights."""
    df = pl.DataFrame(
        {"entity": ["A"] * n, "time": np.arange(n), "px": np.arange(1.0, n + 1.0)}
    )
    return (
        df.with_columns(
            pl.col("px")
            .pipe(frac_diff_expr, weights=weights)
            .over("entity")
            .alias("fd")
        )
        .get_column("fd")
        .to_numpy()
    )


def test_kernel_longer_than_series_is_all_null() -> None:
    """When the kernel is longer than the series, every row is null.

    Intent reversed on 2026-09-28 (plans/done/fracdiff-warmup-decision.md,
    option 1). The warm-up used to be capped at ``n - 1`` so the final row was a
    truncated-kernel "partial" value; that value became ``null`` once the series
    grew past the kernel width, i.e. it depended on ``len(x)`` (hard invariant 1).
    A row is now null exactly when it has no full trailing window.
    """
    n = 20
    weights = ffd_weights(0.4, 1e-5)  # width 1458 >> 20
    assert weights.shape[0] > n
    out = _fd_last_rows(n, weights)
    assert out.shape == (n,)
    assert np.isnan(out).all(), "a row without a full trailing window must be null"


@pytest.mark.parametrize("n", [5, 6, 7, 10])
def test_warmup_is_width_minus_one_at_every_length(n: int) -> None:
    """Null count is ``min(width - 1, n)``: a function of the row, not of ``n``.

    Width 6 here; ``n = 5`` (kernel longer than the series), ``n = 6`` (exactly
    one full window) and longer series must all agree on the rows they share.
    """
    weights = ffd_weights(0.5, 1e-4, max_width=6)
    width = weights.shape[0]
    assert width == 6  # pin the geometry this test reasons about
    out = _fd_last_rows(n, weights)
    assert int(np.isnan(out).sum()) == min(width - 1, n)
    # Every computed row is the full weight-dot-product, never a partial one.
    x = np.arange(1.0, n + 1.0)
    for t in range(width - 1, n):
        expected = float(np.dot(weights, x[t - width + 1 : t + 1]))
        assert out[t] == pytest.approx(expected, abs=1e-12)
    # And the rows shared with the longest series are identical (prefix invariance).
    longest = _fd_last_rows(10, weights)
    np.testing.assert_array_equal(out, longest[:n])


@pytest.mark.parametrize(
    ("d", "thr"),
    [(0.4, DEFAULT_THRESHOLD), (0.4, 1e-3), (0.9, 1e-4), (0.6, 0.02)],
)
def test_frac_diff_is_prefix_invariant(d: float, thr: float) -> None:
    """Hard invariant 1, checked by the library's own instrument.

    Uses ``assert_prefix_invariant`` (truncate the panel, re-run, compare) on a
    panel whose entities are *shorter than the kernel* at the early cuts -- the
    exact regime the old ``n - 1`` cap got wrong (``d=0.4`` at the default
    threshold has width 90 against 120 rows, so the quarter cut has 30 rows).
    ``assert_no_lookahead`` could never see that defect; the partial value was
    causal.
    """
    from panelary.testing import assert_no_lookahead, assert_prefix_invariant

    df = _make_panel(n_entities=3, n_time=120)
    op = pl.col("px").panel.frac_diff(d, threshold=thr).over("entity").alias("fd")
    assert_prefix_invariant(op, df, entity="entity", time="time")
    assert_no_lookahead(op, df, entity="entity", time="time")


def test_prefix_check_would_catch_the_old_n_minus_one_cap() -> None:
    """The instrument above fails a deliberately leaky (``n - 1``) warm-up.

    Guards the guard: if ``assert_prefix_invariant`` stopped seeing a
    length-dependent warm-up, the test above would pass vacuously.
    """
    from panelary.testing import assert_prefix_invariant

    kernel = ffd_weights(0.4, DEFAULT_THRESHOLD)[::-1].copy()
    width = kernel.shape[0]

    def _old(s: pl.Series) -> pl.Series:
        n = s.len()
        out = np.convolve(s.cast(pl.Float64).to_numpy(), kernel)[:n]
        res = pl.Series(s.name, out, dtype=pl.Float64)
        n_null = min(width - 1, n - 1)  # the removed, len(x)-dependent cap
        return res.scatter(np.arange(n_null), None) if n_null > 0 else res

    df = _make_panel(n_entities=2, n_time=120)
    leaky = (
        pl.col("px")
        .map_batches(_old, return_dtype=pl.Float64)
        .over("entity")
        .alias("fd")
    )
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky, df, entity="entity", time="time")


# --------------------------------------------------------------------------- #
# (c) perf sanity: a 0.5M-row panel finishes far under the old multi-second cost
# --------------------------------------------------------------------------- #
def test_perf_half_million_rows_is_fast() -> None:
    df = _make_panel(n_entities=1000, n_time=500)  # ~0.5M rows
    assert df.height == 500_000

    # Warm any import/JIT costs, then time a single full evaluation.
    _ = _panel_out(df.head(2000), 0.4, DEFAULT_THRESHOLD)
    t0 = time.perf_counter()
    out = (
        df.lazy()
        .with_columns(pl.col("px").panel.frac_diff(0.4).over("entity").alias("fd"))
        .collect()
    )
    elapsed = time.perf_counter() - t0

    assert out.get_column("fd").is_not_null().sum() > 0
    # The old sum_horizontal path cost ~2.5s here (and returned all-null at the
    # old default). Generous bound to stay non-flaky on shared CI while still
    # catching an O(n*width) expression-tree regression.
    assert elapsed < 1.5, (
        f"frac_diff on 0.5M rows took {elapsed:.3f}s (expected < 1.5s)"
    )


# --------------------------------------------------------------------------- #
# (d) causality: future rows never change earlier outputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [0.4, 0.9, 1.0])
def test_future_rows_do_not_change_past_outputs(d: float) -> None:
    thr = 1e-3
    base = _make_panel(n_entities=1, n_time=200)
    px0 = base.get_column("px").to_numpy()

    rng = np.random.default_rng(11)
    extra_px = px0[-1] + np.cumsum(rng.standard_normal(80) * 0.5)
    extended = pl.concat(
        [
            base,
            pl.DataFrame(
                {
                    "entity": [0] * 80,
                    "time": np.arange(200, 280, dtype=np.int64),
                    "px": extra_px,
                }
            ),
        ]
    ).sort("entity", "time")

    out_base = _panel_out(base, d, thr)
    out_ext = _panel_out(extended, d, thr)
    np.testing.assert_allclose(
        out_base, out_ext[:200], atol=1e-9, rtol=0.0, equal_nan=True
    )
