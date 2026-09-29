"""Brute-force references for the span table, purge and label weights.

Clean-room from the definitions (AFML ch. 4 and 7), not transcribed book code,
and deliberately slow: O(N^2) loops over Python scalars, so they share no code
path with the vectorised implementations they check.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def frozen_purge_embargo_positions_t1(
    n_times: int,
    test_positions: np.ndarray,
    times: np.ndarray,
    t1: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """Verbatim copy of ``_purge_embargo_positions_t1`` before the span rewrite."""
    test_set = {int(p) for p in test_positions}
    blocked = set(test_set)

    test_arr = np.array(sorted(test_set), dtype=np.int64)
    ti = times[test_arr]  # test label start times
    ei = t1[test_arr]  # test label end times

    for j in range(n_times):
        if j in test_set:
            continue
        tj = times[j]
        ej = t1[j]
        # Overlap with any test interval [ti_k, ei_k]: tj <= ei_k and ti_k <= ej.
        if bool(np.any((tj <= ei) & (ti <= ej))):
            blocked.add(j)

    if embargo > 0:
        for _start, end in _contiguous_blocks(test_arr):
            lo = end + 1
            hi = min(n_times - 1, end + embargo)
            for j in range(lo, hi + 1):
                blocked.add(j)

    return np.array([p for p in range(n_times) if p not in blocked], dtype=np.int64)


def _contiguous_blocks(positions: np.ndarray) -> list[tuple[int, int]]:
    if positions.size == 0:
        return []
    blocks: list[tuple[int, int]] = []
    start = prev = int(positions[0])
    for p in positions[1:]:
        p = int(p)
        if p == prev + 1:
            prev = p
        else:
            blocks.append((start, prev))
            start = prev = p
    blocks.append((start, prev))
    return blocks


def frozen_fixed_horizon(
    frame: pl.DataFrame,
    *,
    entity: str,
    time: str,
    price: str,
    horizon: int,
    threshold: float | None,
) -> pl.DataFrame:
    """Verbatim body of ``fixed_horizon`` before it was routed via forward_return."""
    frame = frame.sort([entity, time])
    fwd_price = pl.col(price).shift(-horizon).over(entity)
    ret_expr = (fwd_price / pl.col(price) - 1.0).alias("ret")
    t1_expr = pl.col(time).shift(-horizon).over(entity).alias("t1")
    frame = frame.with_columns(ret_expr, t1_expr)
    if threshold is None:
        label_expr = pl.col("ret").alias("label")
    else:
        label_expr = (
            pl.when(pl.col("ret") > threshold)
            .then(1)
            .when(pl.col("ret") < -threshold)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("label")
        )
    return frame.with_columns(label_expr)


def pairwise_purge(
    n_times: int, test: np.ndarray, end_pos: np.ndarray, embargo: int
) -> np.ndarray:
    """Pairwise closed-interval purge on positions (``-1`` = null end)."""
    test_set = {int(p) for p in test}
    keep = []
    ends = sorted(test_set)
    block_ends = [b for _a, b in _contiguous_blocks(np.array(ends, dtype=np.int64))]
    for j in range(n_times):
        if j in test_set:
            continue
        ej = int(end_pos[j])
        hit = False
        for i in ends:
            ei = int(end_pos[i])
            if ej >= 0 and ei >= 0 and i <= ej and j <= ei:
                hit = True
                break
        if not hit and embargo > 0:
            hit = any(b < j <= b + embargo for b in block_ends)
        if not hit:
            keep.append(j)
    return np.array(keep, dtype=np.int64)


def ref_spans(
    frame: pl.DataFrame,
    *,
    entity: str,
    time: str,
    t1: str,
    censored: str | None = None,
) -> list[tuple[int, int]]:
    """``(start_row, end_row)`` of every kept label on the sorted grid, by loops."""
    df = frame.sort([entity, time])
    ents = df.get_column(entity).to_list()
    times = df.get_column(time).to_list()
    ends = df.get_column(t1).to_list()
    cens = (
        df.get_column(censored).to_list()
        if censored is not None and censored in df.columns
        else [False] * df.height
    )
    out = []
    for i in range(df.height):
        e = ends[i]
        if e is None or (isinstance(e, float) and math.isnan(e)) or cens[i]:
            continue
        rows = [r for r in range(df.height) if ents[r] == ents[i]]
        if e > times[rows[-1]]:
            continue  # beyond the entity's data: unresolved
        last = i
        for r in rows:
            if times[i] <= times[r] <= e:
                last = max(last, r)
        out.append((i, last))
    return out


def ref_concurrency(spans: list[tuple[int, int]], n_rows: int) -> np.ndarray:
    c = np.zeros(n_rows, dtype=np.int64)
    for s, e in spans:
        for r in range(s, e + 1):
            c[r] += 1
    return c


def ref_uniqueness(spans: list[tuple[int, int]], n_rows: int) -> np.ndarray:
    c = ref_concurrency(spans, n_rows)
    return np.array(
        [math.fsum(1.0 / c[r] for r in range(s, e + 1)) / (e - s + 1) for s, e in spans]
    )


def dense_uniqueness(spans: list[tuple[int, int]], n_rows: int) -> np.ndarray:
    """Uniqueness through the dense indicator matrix (the mlfinlab shape)."""
    ind = np.zeros((n_rows, len(spans)))
    for k, (s, e) in enumerate(spans):
        ind[s : e + 1, k] = 1.0
    c = ind.sum(axis=1)
    u = ind / np.where(c > 0, c, 1.0)[:, None]
    return (u.sum(axis=0) / ind.sum(axis=0)).astype(np.float64)


def ref_span_sums(v: np.ndarray, spans: list[tuple[int, int]], offset: int = 0):
    return np.array([math.fsum(v[s + offset : e + 1]) for s, e in spans])
