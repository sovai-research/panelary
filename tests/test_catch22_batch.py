"""The batched catch22 kernels (:func:`panelary.catch22.catch22_batch`).

Parity with the pre-batch implementation is pinned by the golden baseline in
``tests/test_perf_parity_vectorization.py`` (the scalar functions are now thin
wrappers over these kernels, so that file exercises them). This file pins the
properties the *embedding* layer relies on:

* a batch row equals the scalar call on that row, bit for bit;
* a row's features do not depend on which other rows share the batch -- the
  determinism promise of every catch22-based embedder;
* NaN rows follow the scalar "drop NaNs first" contract;
* the two re-derived vectorised algorithms (the linked-list sweep behind
  ``DN_OutlierInclude`` and the NumPy-mirroring histogram) are exact.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import panelary.catch22 as c22
from panelary.catch22._batch import (
    _outlier_include_1d,
    _outlier_include_sweep,
    _rowsum_compact,
    _uniform_hist,
    _zscore_rows,
    catch22_rows,
)

NAMES24 = [*c22.CATCH22_NAMES, *c22.CATCH24_EXTRA_NAMES]


def _bank(seed: int, L: int, n: int) -> np.ndarray:
    """Varied series: noise, walks, noisy sines, heavy tails, block-constant, ties."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        k = i % 7
        if k == 0:
            x = rng.standard_normal(L)
        elif k == 1:
            x = rng.standard_normal(L).cumsum()
        elif k == 2:
            period = rng.uniform(3, 40)
            x = np.sin(2 * np.pi * np.arange(L) / period) + 0.3 * rng.standard_normal(L)
        elif k == 3:
            x = rng.standard_t(2.5, L)
        elif k == 4:
            x = np.repeat(rng.standard_normal(L // 5 + 1), 5)[:L]
        elif k == 5:
            x = np.round(rng.standard_normal(L) * 2) / 2
        else:
            x = np.linspace(0, 3, L) + 0.2 * rng.standard_normal(L)
        rows.append(x)
    return np.array(rows)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


@pytest.mark.parametrize("L", [0, 1, 3, 7, 12, 20, 33, 64, 128])
def test_batch_row_equals_scalar_call(L):
    X = _bank(L, L, 21) if L else np.zeros((3, 0))
    batch = c22.catch22_batch(X, catch24=True)
    assert batch.shape == (X.shape[0], 24)
    for i, x in enumerate(X):
        scalar = c22.catch22_all(x, catch24=True)
        assert _same(batch[i], np.array([scalar[n] for n in NAMES24])), i


@pytest.mark.parametrize("L", [9, 20, 64, 128])
def test_row_does_not_depend_on_batch_composition(L):
    X = _bank(100 + L, L, 70)
    full = c22.catch22_batch(X, catch24=True)
    for chunk in (1, 5, 17):
        parts = np.vstack(
            [
                c22.catch22_batch(X[i : i + chunk], catch24=True)
                for i in range(0, 70, chunk)
            ]
        )
        assert _same(full, parts), chunk
    # and permuting the batch permutes the rows, nothing more
    perm = np.random.default_rng(0).permutation(70)
    assert _same(full[perm], c22.catch22_batch(X[perm], catch24=True))


def test_nan_rows_follow_the_scalar_contract():
    X = _bank(3, 50, 6)
    X[1, [3, 10, 11]] = np.nan
    X[4, :] = np.nan
    batch = c22.catch22_batch(X, catch24=True)
    for i in (1, 4):
        scalar = c22.catch22_all(X[i], catch24=True)
        assert _same(batch[i], np.array([scalar[n] for n in NAMES24]))
    # a NaN row does not disturb its clean neighbours
    clean = c22.catch22_batch(X[[0, 2, 3, 5]], catch24=True)
    assert _same(batch[[0, 2, 3, 5]], clean)


def test_subset_and_order_of_names():
    X = _bank(5, 64, 8)
    which = ["CO_trev_1_num", "DN_HistogramMode_5"]
    sub = c22.catch22_batch(X, which=which)
    full = c22.catch22_batch(X)
    cols = [c22.CATCH22_NAMES.index(n) for n in which]
    assert _same(sub, full[:, cols])


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        c22.catch22_batch(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError):
        c22.catch22_batch(np.zeros((2, 30)), which=["nope"])
    with pytest.raises(KeyError):
        catch22_rows(np.zeros((2, 30)), ["nope"])


def test_one_dimensional_input_is_one_row():
    x = _bank(9, 40, 1)[0]
    assert _same(c22.catch22_batch(x), c22.catch22_batch(x[None, :]))


@pytest.mark.parametrize("L", [5, 16, 40, 128, 300])
def test_outlier_sweep_equals_per_row_scan(L):
    Y = _zscore_rows(_bank(L, L, 40))
    for sign in (1, -1):
        swept = _outlier_include_sweep(Y, sign)
        scanned = np.array([_outlier_include_1d(y, sign) for y in Y])
        assert _same(swept, scanned), sign


def test_uniform_hist_matches_numpy_histogram():
    rng = np.random.default_rng(11)
    V = np.vstack([rng.standard_normal(97), rng.uniform(-1, 1, 97), np.arange(97.0)])
    V = np.vstack([V, np.round(V * 3) / 3])  # exact ties on bin edges
    for n_bins in (1, 5, 10, 23):
        counts, edges = _uniform_hist(V, V.min(axis=1), V.max(axis=1), n_bins)
        for i, v in enumerate(V):
            ref_counts, ref_edges = np.histogram(v, bins=n_bins)
            assert np.array_equal(counts[i], ref_counts)
            assert np.array_equal(edges[i], ref_edges)


def test_rowsum_compact_is_the_one_dimensional_sum():
    rng = np.random.default_rng(2)
    T = rng.standard_normal((200, 25)) * 10.0 ** rng.integers(-8, 8, (200, 25))
    M = rng.random((200, 25)) < rng.random((200, 1))
    got = _rowsum_compact(T, M)
    ref = np.array([np.sum(t[m]) for t, m in zip(T, M)])
    assert np.array_equal(got, ref)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true"
    and os.environ.get("PANELARY_STRICT_TIMING", "").lower()
    not in {"1", "true", "yes"},
    reason="wall-clock floor; hosted runners measured 822-920 windows/s against "
    "~11,500 locally (PANELARY_STRICT_TIMING=1 enforces it anywhere)",
)
def test_batch_throughput_floor():
    """Plan acceptance: >= 1,000 windows/s at L=128 (measured ~11,500 locally)."""
    X = np.random.default_rng(0).standard_normal((2048, 128)).cumsum(axis=1)
    c22.catch22_batch(X[:32])
    t0 = time.perf_counter()
    c22.catch22_batch(X, catch24=True)
    rate = X.shape[0] / (time.perf_counter() - t0)
    assert rate >= 1_000, f"{rate:.0f} windows/s"
