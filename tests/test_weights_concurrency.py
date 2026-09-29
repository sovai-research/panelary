"""Concurrency, average uniqueness and effective N against brute force."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core._spans import _concurrency, _span_sums, _spans_from_t1, _uniqueness
from tests import _spans_reference as ref


def _panel(seed: int, n_ent: int = 3, n_t: int = 60, h_max: int = 8) -> pl.DataFrame:
    """A ragged panel with random spans (some null, some past the entity end)."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_ent):
        ts = np.sort(rng.choice(n_t, size=int(rng.integers(5, n_t)), replace=False))
        for t in ts:
            end = None if rng.random() < 0.2 else int(t + rng.integers(0, h_max))
            rows.append((f"e{e}", int(t), end))
    return pl.DataFrame(
        rows, schema={"id": pl.String, "t": pl.Int64, "t1": pl.Int64}, orient="row"
    )


@pytest.mark.parametrize("seed", range(15))
def test_concurrency_and_uniqueness_match_brute_force(seed: int) -> None:
    df = _panel(seed)
    spans = ref.ref_spans(df, entity="id", time="t", t1="t1")
    n_rows = df.height

    out = pn.weights.concurrency(df, t1="t1", entity="id", time="t")
    assert out.schema["concurrency"] == pl.Int64
    assert np.array_equal(
        out.get_column("concurrency").to_numpy(), ref.ref_concurrency(spans, n_rows)
    )

    uq = pn.weights.average_uniqueness(df, t1="t1", entity="id", time="t")
    got = uq.get_column("uniqueness").to_numpy()
    rows = [s for s, _e in spans]
    want = ref.ref_uniqueness(spans, n_rows)
    np.testing.assert_allclose(got[rows], want, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        got[rows], ref.dense_uniqueness(spans, n_rows), rtol=1e-12
    )
    others = np.setdiff1d(np.arange(n_rows), rows)
    assert uq.get_column("uniqueness").gather(others).null_count() == others.size

    assert pn.weights.effective_n(df, t1="t1", entity="id", time="t") == pytest.approx(
        math.fsum(want), rel=1e-12
    )


def test_uniqueness_properties() -> None:
    df = _panel(99, n_ent=4, n_t=200)
    _, table = _spans_from_t1(df, t1="t1", entity="id", time="t")
    c = _concurrency(table.start, table.end, table.n_rows)
    u = _uniqueness(table)
    assert np.all(u > 0) and np.all(u <= 1)
    assert int(c.sum()) == int(table.length.sum())
    assert pn.weights.effective_n(df, t1="t1", entity="id", time="t") <= len(table)


def test_no_overlap_is_fully_unique_and_identical_spans_share() -> None:
    df = pl.DataFrame({"id": ["a"] * 6, "t": list(range(6)), "t1": [0, 1, 2, 3, 4, 5]})
    u = pn.weights.average_uniqueness(df, t1="t1").get_column("uniqueness")
    assert u.to_list() == [1.0] * 6
    # k labels on k entities, each alone in its entity: all unique.
    wide = pl.DataFrame({"id": ["a", "b", "c"], "t": [0, 0, 0], "t1": [0, 0, 0]})
    assert pn.weights.effective_n(wide, t1="t1") == 3.0
    # k identical spans on one entity: each gets 1/k.
    _, table = _spans_from_t1(
        pl.DataFrame({"id": ["a"] * 3, "t": [0, 1, 2], "t1": [2, 2, 2]}), t1="t1"
    )
    stacked = replace(
        table, start=np.zeros(3, dtype=np.int64), end=np.full(3, 2, dtype=np.int64)
    )
    np.testing.assert_allclose(_uniqueness(stacked), [1 / 3] * 3, rtol=0, atol=1e-15)


@pytest.mark.parametrize("h", [1, 3, 20])
def test_fixed_horizon_closed_form(h: int) -> None:
    """h-row labels on every row of T rows: sum(ubar) has a closed form."""
    n = 120
    df = pl.DataFrame({"id": ["a"] * n, "t": list(range(n))}).with_columns(
        pl.when(pl.col("t") + h < n).then(pl.col("t") + h).alias("t1")
    )
    _, table = _spans_from_t1(df, t1="t1")
    c = ref.ref_concurrency(list(zip(table.start, table.end, strict=True)), n)
    # Every covered row r contributes 1/c_r spread over the labels covering it,
    # each weighted 1/(h+1): sum(ubar) = (#covered rows) / (h + 1).
    covered = int(np.count_nonzero(c))
    assert pn.weights.effective_n(df, t1="t1") == pytest.approx(
        covered / (h + 1), rel=1e-12
    )


@pytest.mark.parametrize("block", [1, 3, 7, 4096])
@pytest.mark.parametrize("offset", [0, 1])
def test_span_sums_block_paths(block: int, offset: int) -> None:
    rng = np.random.default_rng(block + offset)
    v = rng.normal(size=500)
    start = np.sort(rng.integers(0, 500, 300))
    end = np.minimum(start + rng.integers(0, 40, 300), 499)
    spans = list(zip(start.tolist(), end.tolist(), strict=True))
    got = _span_sums(v, start, end, offset=offset, block=block)
    np.testing.assert_allclose(
        got, ref.ref_span_sums(v, spans, offset), rtol=1e-12, atol=1e-13
    )
    # empty spans (start + offset > end) sum to exactly zero
    if offset:
        single = start == end
        assert np.all(got[single] == 0.0)


@pytest.mark.parametrize("block", [2, 5, 64])
def test_span_sums_segment_aligned(block: int) -> None:
    """Blocks restart at every segment: spans inside segments, many lengths."""
    rng = np.random.default_rng(block)
    lens = rng.integers(1, 400, 30)
    segments = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    n = int(lens.sum())
    v = rng.normal(size=n)
    seg_of = rng.integers(0, 30, 500)
    lo = segments[seg_of]
    hi = lo + lens[seg_of] - 1
    start = np.minimum(lo + rng.integers(0, 400, 500), hi)
    end = np.minimum(start + rng.integers(0, 300, 500), hi)
    order = np.argsort(start, kind="stable")
    start, end = start[order], end[order]
    spans = list(zip(start.tolist(), end.tolist(), strict=True))
    for offset in (0, 1):
        got = _span_sums(v, start, end, offset=offset, segments=segments, block=block)
        np.testing.assert_allclose(
            got, ref.ref_span_sums(v, spans, offset), rtol=1e-12, atol=1e-12
        )
    # A segment's sums do not depend on the rows of other segments.
    k = int(seg_of[order][0])
    mine = (start >= segments[k]) & (end < segments[k] + lens[k])
    v2 = v.copy()
    v2[: segments[k]] = rng.normal(size=int(segments[k]))
    a = _span_sums(v, start[mine], end[mine], segments=segments, block=block)
    b = _span_sums(v2, start[mine], end[mine], segments=segments, block=block)
    assert np.array_equal(a, b)


def test_span_sums_accuracy_does_not_grow_with_the_grid() -> None:
    """A single global prefix would lose ~R*eps; block-local prefixes do not."""
    rng = np.random.default_rng(0)
    n = 1_000_000
    v = 1.0 / rng.integers(1, 40, n)
    start = np.sort(rng.integers(0, n - 50, 2000))
    end = start + rng.integers(0, 50, 2000)
    got = _span_sums(v, start, end)
    want = ref.ref_span_sums(v, list(zip(start.tolist(), end.tolist(), strict=True)))
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)


def test_frames_are_sorted_and_lazy_accepted() -> None:
    df = _panel(3).sample(fraction=1.0, shuffle=True, seed=1)
    out = pn.weights.concurrency(df.lazy(), t1="t1", entity="id", time="t")
    assert out.select(["id", "t"]).equals(df.sort(["id", "t"]).select(["id", "t"]))
    table = pn.weights.spans(df, t1="t1", entity="id", time="t")
    assert isinstance(table, pn.weights.SpanTable)


def test_censored_rows_are_excluded_by_default() -> None:
    df = pl.DataFrame(
        {
            "id": ["a"] * 4,
            "t": [0, 1, 2, 3],
            "t1": [1, 2, 3, 3],
            "censored": [False, False, True, True],
        }
    )
    assert pn.weights.spans(df, t1="t1").start.tolist() == [0, 1]
    assert pn.weights.spans(df, t1="t1", censored=None).start.tolist() == [0, 1, 2, 3]
