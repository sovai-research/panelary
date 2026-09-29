"""Leak-safe fitness evaluation for evolutionary alpha mining.

This is the module Panelary exists for. Across the ~40 symbolic-regression / GP
libraries and 9 alpha-mining repositories surveyed for
``plans/done/evolve-build-contract.md``, **zero** implement purging, embargo, or
any overfitting control, and none are panel-aware. Everything here is about the
two properties nobody else ships: *out-of-fold-by-construction* scoring, and an
acceptance threshold calibrated against noise rather than against zero.

Four design decisions, and the evidence for each
------------------------------------------------

**1. Rank IC, not R² / MSE.** The practitioner standard for a cross-sectional
signal is the per-date Spearman correlation with the forward return, averaged
over dates (:func:`rank_ic`). Pooling across dates would let a level difference
between 2008 and 2021 masquerade as predictive power. :func:`numerai_corr`
(CORR20V2, ported from numerai-tools, MIT) additionally gaussianises the
prediction and raises both sides to the 1.5 power, which accentuates the tails —
appropriate because only the extremes of a signal are ever traded.

**2. Marginal contribution, never standalone IC.** Two independent literatures
converged on this. AlphaGen (Yu et al., KDD 2023) rewards the *marginal* change
in the combined pool's IC; their own Table 2 reports PPO optimising standalone
IC scoring **-0.0166** on the CSI300 test set — worse than doing nothing at all.
OpenFE (Zhang et al., ICML 2023) scores a candidate by the *residual reduction*
it achieves against a warm-started baseline rather than by refitting from
scratch. :class:`AlphaPool` implements the former, ``mode="residual"`` on
:class:`PanelEvaluator` the latter.

*A nuance worth internalising*: AlphaGen's own results show that mutual-IC
**filtering** is a proxy for the wrong thing. They report two alphas with 0.9746
mutual IC that nevertheless *combined* to a higher test IC than either alone.
The geometry is elementary — as two unit vectors converge, their difference
becomes orthogonal to both, and it is the difference the least-squares
combination trades. So correlation to the library is kept here as an **archive
descriptor and a reporting axis** (``Descriptors.max_corr``,
``FitnessResult.max_corr_to_library``) and is deliberately **not** a hard
admission gate. The gate is marginal contribution.

**3. Leak-safety by construction.** Base scores are out-of-fold under
:class:`~panelary.core.model_selection.PurgedKFold` or
:class:`~panelary.core.model_selection.CombinatorialPurgedCV` — never
random, never stratified. The embargo defaults to ``max(1, horizon)`` and is on
by default; passing a smaller one warns. Every quantity with a *fit* — the sign
orientation, the pool weights, the ridge/LightGBM baseline, the behaviour
descriptors — is fit on the training folds of that fold only. Ambroise &
McLachlan (PNAS 99(10):6562-6566, 2002) showed that selection on the full sample
yields a near-zero cross-validated error even on **permuted labels**, while
fold-local selection correctly reports the ~0.40-0.45 error of pure noise. The
tests for this module reproduce exactly that contrast.

**4. The acceptance threshold is the best noise individual, not zero.** Ported
in spirit from autofeat (MIT): shuffled real columns and ``N(0, 1)`` columns are
pushed through the *identical* scoring path, and :func:`null_threshold` returns
the top of their score distribution. A 100,000-trial search on pure noise
produces a best in-sample Sharpe near 4.4; comparing to zero is a fabrication.

Performance notes
-----------------
Per-date ranking is the only Python-level loop, and it runs once per column
chunk over the date blocks. Everything downstream — standardisation, the
per-date IC matrix, mutual-IC matrices — is expressed as ``np.add.reduceat``
segment sums or a single matmul, exploiting the identity that for per-date
standardised columns ``Za`` and ``Zb``, ``Za.T @ Zb`` over *all* rows equals the
sum over dates of the per-date cross-sectional correlations.

Everything is numpy + polars. LightGBM is optional, routed through
:func:`panelary._internal._deps.require`, and the pure-numpy ridge baseline is the
**default**.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import polars as pl

from panelary.core.model_selection import (
    CombinatorialPurgedCV,
    PurgedKFold,
    _norm_ppf,
)
from panelary.core.panel_frame import PanelFrame
from panelary.evolve._types import (
    CaseMatrix,
    Descriptors,
    EvalContext,
    FitnessResult,
    Genome,
)

__all__ = [
    "rank_ic",
    "rank_ic_series",
    "ic_ir",
    "numerai_corr",
    "turnover",
    "decile_monotonicity",
    "AlphaPool",
    "PanelEvaluator",
    "descriptors",
    "null_threshold",
    "noise_matrix",
    "Metric",
    "PoolObjective",
]

#: Scoring transforms available to :class:`PanelEvaluator`.
Metric = Literal["rank_ic", "numerai", "pearson"]

#: Objective variants for :class:`AlphaPool`.
PoolObjective = Literal["ic", "icir", "lcb"]

#: Minimum cross-section size for a date to contribute an IC. Below this the
#: per-date correlation is pure noise and the date is dropped, not shrunk.
MIN_CROSS_SECTION: int = 5

#: Row-index column :class:`PanelEvaluator` adds to its frame. It is how every
#: compiled value is put back at its evaluator row, whatever order the compiler
#: returns rows in. Shares the compiler's ``__ev_`` prefix, never a digest.
_ROW: str = "__ev_row"

#: 1/e — the conventional decay threshold for the ACF-crossing horizon.
_INV_E: float = 1.0 / math.e


# --------------------------------------------------------------------------- #
# Grouped-numpy primitives
#
# Every array below is laid out with rows sorted by (time, entity), so each date
# is a *contiguous slice*. `starts` is the (T + 1,) array of slice boundaries.
# This layout is what makes the whole module reduceat-shaped instead of loopy.
# --------------------------------------------------------------------------- #
def _group_starts(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Slice boundaries of contiguous, non-decreasing group ``codes``.

    Parameters
    ----------
    codes : ndarray of int
        Non-decreasing group labels in ``[0, n_groups)``.
    n_groups : int
        Number of groups.

    Returns
    -------
    ndarray of int64, shape (n_groups + 1,)
        ``starts[i]:starts[i + 1]`` is the row slice of group ``i``.
    """
    return np.searchsorted(codes, np.arange(n_groups + 1, dtype=np.int64)).astype(
        np.int64
    )


def _rank_block(block: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) down ``axis=0``, NaN-aware.

    Ties receive the mean of the ranks they span — the same convention as
    ``scipy.stats.rankdata(method="average")`` and pandas ``rank("average")``,
    reimplemented here so the core stays numpy-only. Handling ties properly is
    not pedantry: a genome emitting a constant or a binary column would, under
    ordinal ranks, acquire a spurious ordering aligned with the row order and
    therefore a spurious IC.

    Non-finite entries map to NaN and are excluded from the ranking of the rest.
    """
    n, k = block.shape
    if n == 0:
        return np.empty((0, k), dtype=np.float64)
    order = np.argsort(block, axis=0, kind="stable")  # NaN sorts last
    srt = np.take_along_axis(block, order, axis=0)
    pos = np.arange(n, dtype=np.float64)[:, None]

    eq = np.zeros((n, k), dtype=bool)
    if n > 1:
        # NaN != NaN, so every NaN starts its own run; they are masked out below.
        eq[1:] = srt[1:] == srt[:-1]

    run_start = np.where(eq, -1.0, pos)
    np.maximum.accumulate(run_start, axis=0, out=run_start)

    nxt_new = np.ones((n, k), dtype=bool)
    if n > 1:
        nxt_new[:-1] = ~eq[1:]
    run_end = np.where(nxt_new, pos, float(n))
    run_end = np.minimum.accumulate(run_end[::-1], axis=0)[::-1]

    avg = (run_start + run_end) * 0.5 + 1.0
    avg[~np.isfinite(srt)] = np.nan

    out = np.empty((n, k), dtype=np.float64)
    np.put_along_axis(out, order, avg, axis=0)
    return out


@lru_cache(maxsize=512)
def _ppf_table(n: int) -> np.ndarray:
    """Gaussian quantiles for every attainable average rank in a size-``n`` group.

    Average ranks under ties are always multiples of ``0.5`` in ``[1, n]``, so
    the whole map ``rank -> norm.ppf((rank - 0.5) / n)`` is a table of ``2n - 1``
    entries indexed by ``2 * rank - 2``. Built from the Acklam ``_norm_ppf``
    already in :mod:`panelary.core.model_selection` — no SciPy, and no
    second copy of the coefficients.
    """
    ranks = 1.0 + 0.5 * np.arange(2 * n - 1, dtype=np.float64)
    return np.array([_norm_ppf((r - 0.5) / n) for r in ranks], dtype=np.float64)


def _gaussianize_ranks(ranks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Map average ranks to gaussian quantiles, per column valid-count.

    ``ranks`` is ``(n, k)`` with NaN for missing; ``counts`` is ``(k,)``, the
    number of valid observations per column in this block.
    """
    out = np.full(ranks.shape, np.nan, dtype=np.float64)
    for m in np.unique(counts):
        m_int = int(m)
        if m_int < 1:
            continue
        cols = np.flatnonzero(counts == m)
        table = _ppf_table(m_int)
        sub = ranks[:, cols]
        idx = np.rint(2.0 * np.nan_to_num(sub, nan=1.0) - 2.0).astype(np.int64)
        np.clip(idx, 0, table.shape[0] - 1, out=idx)
        vals = table[idx]
        vals[~np.isfinite(sub)] = np.nan
        out[:, cols] = vals
    return out


def _pow_1_5(x: np.ndarray) -> np.ndarray:
    """``sign(x) * |x| ** 1.5`` — CORR20V2's tail accentuation."""
    return np.sign(x) * np.abs(x) ** 1.5


def _transform_columns(
    values: np.ndarray, starts: np.ndarray, kind: Metric
) -> np.ndarray:
    """Apply the per-date score transform to every column of ``values``.

    ``"rank_ic"`` -> average ranks; ``"numerai"`` -> gaussianised ranks raised to
    the 1.5 power; ``"pearson"`` -> the raw values. This is the only per-date
    Python loop in the module.
    """
    if kind == "pearson":
        out = np.array(values, dtype=np.float64, copy=True)
        out[~np.isfinite(out)] = np.nan
        return out
    out = np.empty(values.shape, dtype=np.float64)
    for lo, hi in zip(starts[:-1], starts[1:], strict=True):
        block = np.asarray(values[lo:hi], dtype=np.float64)
        ranks = _rank_block(block)
        if kind == "numerai":
            counts = np.isfinite(ranks).sum(axis=0)
            ranks = _pow_1_5(_gaussianize_ranks(ranks, counts))
        out[lo:hi] = ranks
    return out


def _transform_target(y: np.ndarray, starts: np.ndarray, kind: Metric) -> np.ndarray:
    """Per-date target transform matching :func:`_transform_columns`.

    CORR20V2 does **not** gaussianise the target: it centres it within the
    cross-section and applies the same 1.5 power, which keeps the magnitude
    information in the label while still weighting the tails.
    """
    col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if kind == "numerai":
        out = np.empty_like(col)
        for lo, hi in zip(starts[:-1], starts[1:], strict=True):
            block = col[lo:hi]
            finite = np.isfinite(block)
            mean = block[finite].mean() if finite.any() else 0.0
            out[lo:hi] = _pow_1_5(block - mean)
        return out[:, 0]
    return _transform_columns(col, starts, kind)[:, 0]


def _standardize_by_date(
    x: np.ndarray, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-date centre-and-unit-norm, NaN-safe.

    Returns ``(z, counts)`` where ``z`` is ``(n, k)`` with missing entries set to
    exactly ``0.0`` (so they contribute nothing to any dot product) and
    ``counts`` is the ``(T, k)`` per-date valid-observation count.

    Because each column is standardised *within* a date, the global product
    ``za.T @ zb`` equals the sum over dates of the per-date cross-sectional
    correlations. That identity is what removes the second Python loop.

    A date whose column is constant (zero norm) or has fewer than
    :data:`MIN_CROSS_SECTION` valid rows is zeroed out and reported through
    ``counts`` so callers can mark it missing rather than silently scoring it.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    idx = starts[:-1]
    sizes = np.diff(starts)

    counts = np.add.reduceat(finite.astype(np.float64), idx, axis=0)
    safe_counts = np.where(counts > 0, counts, 1.0)
    means = np.add.reduceat(filled, idx, axis=0) / safe_counts

    centred = (filled - np.repeat(means, sizes, axis=0)) * finite
    ss = np.add.reduceat(centred * centred, idx, axis=0)
    norms = np.sqrt(ss)
    ok = (norms > 0.0) & (counts >= MIN_CROSS_SECTION)
    safe_norms = np.where(ok, norms, 1.0)
    z = centred / np.repeat(safe_norms, sizes, axis=0)
    z *= np.repeat(ok.astype(np.float64), sizes, axis=0)
    return z, np.where(ok, counts, 0.0)


def _per_date_ic(
    z_feat: np.ndarray, z_target: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """The ``(T, k)`` per-date IC matrix from standardised columns.

    Missing dates (constant column, or too small a cross-section) come back NaN
    rather than 0, so that "no opinion" never gets averaged in as "no skill".
    """
    prod = z_feat * z_target.reshape(-1, 1)
    out = np.add.reduceat(prod, starts[:-1], axis=0)
    out[counts <= 0] = np.nan
    np.clip(out, -1.0, 1.0, out=out)
    return out


def _nanmean(a: np.ndarray, axis: int | None = None) -> Any:
    """``np.nanmean`` without the all-NaN RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a: np.ndarray, axis: int | None = None, ddof: int = 1) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis, ddof=ddof)


def _codes(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense integer codes for an arbitrary label array, order-preserving."""
    uniq, inv = np.unique(values, return_inverse=True)
    return inv.astype(np.int64), int(uniq.shape[0])


def _prepare_1d(
    feature: Any, target: Any, by_time: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Sort a flat ``(feature, target, time)`` triple into date-contiguous order.

    Rows with a non-finite target are dropped, and the date groups are built
    from the rows that remain, so every group is non-empty. That matters:
    ``np.add.reduceat`` raises ``IndexError`` on a trailing empty group and
    returns a neighbouring row (not zero) for an interior one. A forward label
    is null on its last ``horizon`` dates, so trailing empty dates are the
    normal case for a held-out window, not an edge case.

    Returns
    -------
    (feature, target, starts, present, n_times)
        ``starts`` delimits the non-empty date groups; ``present[g]`` is the
        position of group ``g`` among all ``n_times`` distinct dates of
        ``by_time``.
    """
    f = np.asarray(
        feature.to_numpy() if isinstance(feature, pl.Series) else feature,
        dtype=np.float64,
    )
    y = np.asarray(
        target.to_numpy() if isinstance(target, pl.Series) else target, dtype=np.float64
    )
    t = by_time.to_numpy() if isinstance(by_time, pl.Series) else np.asarray(by_time)
    if not (f.shape[0] == y.shape[0] == t.shape[0]):
        raise ValueError(
            "feature, target and by_time must be the same length, got "
            f"{f.shape[0]}, {y.shape[0]}, {t.shape[0]}."
        )
    codes, n_times = _codes(t)
    order = np.argsort(codes, kind="stable")
    order = order[np.isfinite(y[order])]
    codes = codes[order]
    present = np.unique(codes)
    starts = _group_starts(np.searchsorted(present, codes), present.shape[0])
    return f[order], y[order], starts, present, n_times


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rank_ic_series(
    feature: Any, fwd_ret: Any, *, by_time: Any, metric: Metric = "rank_ic"
) -> np.ndarray:
    """Per-date information coefficient of ``feature`` against ``fwd_ret``.

    Parameters
    ----------
    feature, fwd_ret : array-like or polars.Series
        The signal and its (already leak-safe) forward return, row-aligned.
    by_time : array-like or polars.Series, keyword-only
        The date key. Rows are grouped by it; nothing crosses a date boundary.
    metric : {"rank_ic", "numerai", "pearson"}, keyword-only
        The per-date transform. ``"rank_ic"`` is Spearman.

    Returns
    -------
    ndarray of float64, shape (n_dates,)
        One entry per distinct value of ``by_time``, in sorted order. NaN for
        dates with no finite target, a constant signal, or fewer than
        :data:`MIN_CROSS_SECTION` usable observations.
    """
    f, y, starts, present, n_times = _prepare_1d(feature, fwd_ret, by_time)
    out = np.full(n_times, np.nan, dtype=np.float64)
    if present.shape[0] == 0:
        return out
    xf = _transform_columns(f.reshape(-1, 1), starts, metric)
    xy = _transform_target(y, starts, metric)
    zf, counts = _standardize_by_date(xf, starts)
    zy, ycounts = _standardize_by_date(xy, starts)
    ic = _per_date_ic(zf, zy[:, 0], starts, np.minimum(counts, ycounts))[:, 0]
    out[present] = ic
    return out


def rank_ic(
    feature: Any, fwd_ret: Any, *, by_time: Any, metric: Metric = "rank_ic"
) -> float:
    """Mean per-date rank IC — the practitioner-standard signal-quality metric.

    Spearman correlation is computed *within* each cross-section and only then
    averaged, so a level shift between regimes can never masquerade as skill.
    NaN dates are skipped, not treated as zero.

    Parameters
    ----------
    feature, fwd_ret : array-like or polars.Series
    by_time : array-like or polars.Series, keyword-only
    metric : {"rank_ic", "numerai", "pearson"}, keyword-only

    Returns
    -------
    float
        Mean IC, or NaN if no date yielded one.
    """
    series = rank_ic_series(feature, fwd_ret, by_time=by_time, metric=metric)
    return float(_nanmean(series))


def ic_ir(
    feature: Any,
    fwd_ret: Any,
    *,
    by_time: Any,
    metric: Metric = "rank_ic",
    ddof: int = 1,
) -> float:
    """Information ratio of the IC series: ``mean(IC) / std(IC)``.

    The stability of the IC matters more than its level for a tradable signal;
    this is the quantity :class:`AlphaPool`'s ``objective="icir"`` maximises.

    Returns
    -------
    float
        NaN when fewer than two dates produce an IC, or the IC is constant.
    """
    series = rank_ic_series(feature, fwd_ret, by_time=by_time, metric=metric)
    mu = _nanmean(series)
    sd = _nanstd(series, ddof=ddof)
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan")
    return float(mu / sd)


def numerai_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """CORR20V2 — Numerai's tail-weighted rank correlation.

    A clean-room port of the definition published in ``numerai-tools`` (MIT):

    1. average-rank the prediction and map to ``(r - 0.5) / n``;
    2. push through the inverse normal CDF (gaussianise);
    3. centre the target within the sample;
    4. raise **both** sides to the 1.5 power, preserving sign;
    5. take the Pearson correlation.

    The 1.5 power is the whole point. Only the extremes of a signal are ever
    traded, so a metric that weights the middle of the book equally with the
    tails is measuring something nobody acts on. Empirically this separates
    alphas materially better than raw Spearman for mining purposes.

    The inverse-CDF is the Acklam approximation already in
    :mod:`panelary.core.model_selection` (accurate to ~1e-9); SciPy is
    deliberately not a dependency.

    Parameters
    ----------
    pred, target : ndarray
        One cross-section, row-aligned. Non-finite pairs are dropped.

    Returns
    -------
    float
        Pearson correlation of the transformed pair, or NaN if degenerate.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    if p.shape != t.shape:
        raise ValueError(f"pred and target must align, got {p.shape} vs {t.shape}.")
    keep = np.isfinite(p) & np.isfinite(t)
    p, t = p[keep], t[keep]
    n = p.shape[0]
    if n < 2:
        return float("nan")
    ranks = _rank_block(p.reshape(-1, 1))
    gauss = _gaussianize_ranks(ranks, np.array([n]))[:, 0]
    pp = _pow_1_5(gauss)
    tt = _pow_1_5(t - t.mean())
    sp, st = pp.std(), tt.std()
    if sp <= 0.0 or st <= 0.0:
        return float("nan")
    return float(np.mean((pp - pp.mean()) * (tt - tt.mean())) / (sp * st))


def turnover(feature: Any, *, by_time: Any, by_entity: Any) -> float:
    """``1 - mean_t corr(rank_t, rank_{t-1})`` over the cross-section.

    Zero means the book never changes; one means today's ranking is unrelated to
    yesterday's; two is a perfect daily reversal. Entities are matched by key
    across consecutive dates, so an unbalanced panel is handled correctly.

    Returns
    -------
    float
        In ``[0, 2]``, or NaN if no consecutive date pair is usable.
    """
    f = np.asarray(
        feature.to_numpy() if isinstance(feature, pl.Series) else feature,
        dtype=np.float64,
    )
    t = by_time.to_numpy() if isinstance(by_time, pl.Series) else np.asarray(by_time)
    e = (
        by_entity.to_numpy()
        if isinstance(by_entity, pl.Series)
        else np.asarray(by_entity)
    )
    tc, n_times = _codes(t)
    ec, n_ent = _codes(e)
    order = np.lexsort((ec, tc))
    tc, ec, f = tc[order], ec[order], f[order]
    starts = _group_starts(tc, n_times)
    ranks = _transform_columns(f.reshape(-1, 1), starts, "rank_ic")
    dense = _dense_by_entity(ranks, tc, ec, n_times, n_ent)
    return float(1.0 - _mean_lag_autocorr(dense))


def decile_monotonicity(
    feature: Any, fwd_ret: Any, *, by_time: Any, n_bins: int = 10
) -> float:
    """Rank correlation between signal decile and mean forward return.

    Buckets each cross-section into ``n_bins`` equal-count groups by signal
    rank, averages the forward return within each bucket, averages those bucket
    means across dates, and returns the Spearman correlation between bucket
    index and bucket mean. ``+1`` is a perfectly monotone factor; values near
    zero mean the signal only works in its extremes (or not at all).

    Returns
    -------
    float
        In ``[-1, 1]``, or NaN if fewer than two buckets ever fill.
    """
    if n_bins < 2:
        raise ValueError(f"`n_bins` must be >= 2, got {n_bins}.")
    f, y, starts, present, _ = _prepare_1d(feature, fwd_ret, by_time)
    if present.shape[0] == 0:
        return float("nan")
    ranks = _transform_columns(f.reshape(-1, 1), starts, "rank_ic")[:, 0]
    sizes = np.diff(starts)
    counts_per_date = np.add.reduceat(
        np.isfinite(ranks).astype(np.float64), starts[:-1]
    )
    denom = np.repeat(np.where(counts_per_date > 0, counts_per_date, 1.0), sizes)
    bucket = np.floor((ranks - 1.0) * n_bins / denom)
    valid = np.isfinite(bucket)
    b = np.clip(np.nan_to_num(bucket, nan=0.0), 0, n_bins - 1).astype(np.int64)
    weight = valid.astype(np.float64)
    tot = np.bincount(b, weights=y * weight, minlength=n_bins)
    cnt = np.bincount(b, weights=weight, minlength=n_bins)
    filled = cnt > 0
    if filled.sum() < 2:
        return float("nan")
    means = tot[filled] / cnt[filled]
    idx = np.flatnonzero(filled).astype(np.float64)
    mr = _rank_block(means.reshape(-1, 1))[:, 0]
    ir = _rank_block(idx.reshape(-1, 1))[:, 0]
    sd_m, sd_i = mr.std(), ir.std()
    if sd_m <= 0.0 or sd_i <= 0.0:
        return float("nan")
    return float(np.mean((mr - mr.mean()) * (ir - ir.mean())) / (sd_m * sd_i))


# --------------------------------------------------------------------------- #
# Dense (date x entity) helpers, used by the behaviour descriptors
# --------------------------------------------------------------------------- #
def _dense_by_entity(
    values: np.ndarray,
    time_code: np.ndarray,
    entity_code: np.ndarray,
    n_times: int,
    n_entities: int,
) -> np.ndarray:
    """Scatter a long column block into a dense ``(T, E, k)`` cube of NaN."""
    if values.ndim == 1:
        values = values[:, None]
    k = values.shape[1]
    dense = np.full((n_times * n_entities, k), np.nan, dtype=np.float64)
    dense[time_code * n_entities + entity_code] = values
    return dense.reshape(n_times, n_entities, k)


def _masked_corr_axis0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation down ``axis=0`` with pairwise NaN deletion."""
    mask = np.isfinite(a) & np.isfinite(b)
    af = np.where(mask, a, 0.0)
    bf = np.where(mask, b, 0.0)
    n = mask.sum(axis=0).astype(np.float64)
    safe = np.where(n > 1, n, np.nan)
    sa, sb = af.sum(axis=0), bf.sum(axis=0)
    cov = (af * bf).sum(axis=0) - sa * sb / safe
    va = (af * af).sum(axis=0) - sa * sa / safe
    vb = (bf * bf).sum(axis=0) - sb * sb / safe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = cov / np.sqrt(va * vb)
    out[~np.isfinite(out)] = np.nan
    return out


def _mean_lag_autocorr(dense: np.ndarray) -> np.ndarray | float:
    """Mean over dates of the cross-sectional lag-1 rank autocorrelation."""
    if dense.shape[0] < 2:
        return float("nan")
    per_pair = np.stack(
        [_masked_corr_axis0(dense[t], dense[t + 1]) for t in range(dense.shape[0] - 1)]
    )
    out = _nanmean(per_pair, axis=0)
    return float(out[0]) if out.shape == (1,) else out


def _acf_horizon(dense: np.ndarray, *, max_lag: int = 32) -> np.ndarray:
    """First lag at which the mean per-entity ACF drops below ``1/e``.

    ``dense`` is ``(T, E, k)``. The autocorrelation is computed *within* each
    entity's own series (never pooled across entities, which would confound the
    cross-sectional spread with persistence), averaged over entities, and the
    crossing lag is reported per column. Lags are walked in order with an early
    exit as soon as every column has crossed, so a fast-decaying population
    costs one or two lags rather than ``max_lag``.

    A column that never crosses is assigned ``max_lag`` — a censored value, and
    the archive treats it as such.
    """
    n_times, _, k = dense.shape
    horizon = np.full(k, float(max_lag), dtype=np.float64)
    done = np.zeros(k, dtype=bool)
    for lag in range(1, min(max_lag, max(n_times - 2, 1)) + 1):
        acf = _nanmean(_masked_corr_axis0(dense[:-lag], dense[lag:]), axis=0)
        acf = np.asarray(acf, dtype=np.float64).reshape(k)
        crossed = ~done & (~np.isfinite(acf) | (acf < _INV_E))
        horizon[crossed] = float(lag)
        done |= crossed
        if done.all():
            break
    return horizon


# --------------------------------------------------------------------------- #
# AlphaPool — marginal contribution as the objective
# --------------------------------------------------------------------------- #
class AlphaPool:
    """A least-squares combination of alphas, scored by *marginal* contribution.

    The pool holds ``K`` alphas through two small matrices: the mutual-IC matrix
    ``M`` (``K x K``, the mean over dates of the cross-sectional correlation
    between members) and the single-IC vector ``c`` (``K``, each member's mean
    IC against the target). Treating each alpha as per-date standardised, the
    squared error of the combination ``sum_i w_i z_i`` against the standardised
    target reduces exactly to

    .. math:: L(w) = w' M w - 2 w' c + 1 \\;(+\\; \\lambda \\lVert w \\rVert_1)

    which is minimised in closed form by ``w = M^{-1} c`` when ``lambda = 0``
    (the maximum-IR combination), and by ISTA otherwise.

    **Why marginal, not standalone.** AlphaGen (KDD 2023) rewards the change in
    the *pool's* objective from admitting a candidate; their Table 2 reports
    that optimising standalone IC instead scores ``-0.0166`` on the CSI300 test
    set, i.e. worse than an empty pool. :meth:`marginal_contribution` is that
    reward.

    **Why correlation is not an admission gate.** The same paper reports two
    alphas at 0.9746 mutual IC combining to a *higher* test IC than either
    alone. As two unit vectors converge their difference becomes orthogonal to
    both, and the least-squares weights trade that difference. Mutual IC is
    therefore carried here only as a descriptor; the gate is
    :meth:`marginal_contribution`.

    **Incremental update.** :meth:`add` appends one row and one column to ``M``
    in ``O(K)``, using mutual ICs the caller already has in hand. Recomputing a
    50x50 mutual-IC matrix on a 500-entity x 3000-date panel was measured at
    5.58 s on this repo — a cost that must never be paid inside a search loop.

    Parameters
    ----------
    max_size : int, default 50
        Cap on ``K``. When full, :meth:`add` evicts the member with the smallest
        absolute weight.
    l1 : float, default 0.0
        L1 penalty. ``0`` uses the closed form; ``> 0`` runs ISTA.
    objective : {"ic", "icir", "lcb"}, default "ic"
        ``"ic"`` maximises the combined mean IC, ``"icir"`` its information
        ratio, ``"lcb"`` the lower confidence bound ``mean - beta * std``.
    lcb_beta : float, default 1.0
        The ``beta`` in the LCB objective.
    ridge : float, default 1e-8
        Tikhonov term added to ``M``'s diagonal before solving. Necessary, not
        cosmetic: near-duplicate alphas make ``M`` numerically singular, which
        is exactly the regime the 0.9746 example lives in.
    cond_max : float, default 1e6
        Conditioning guard. When the ridge solve returns a weight vector larger
        than ``cond_max * ||c||`` the pool falls back to the truncated
        pseudo-inverse, dropping the unidentifiable directions instead of
        amplifying them. Without this a rank-deficient ``M`` can report a
        combined "IC" in the hundreds.

    Notes
    -----
    For the ``"icir"``/``"lcb"`` variants the pool needs a per-date IC series per
    member. The combined signal's per-date IC is approximated as
    ``(w . c_t) / sqrt(w' M w)`` — exact when the per-date mutual-IC matrix
    equals its time average, and an ``O(K)`` approximation otherwise. Recomputing
    it exactly would require the full ``K x K x T`` cube and defeat the purpose.
    """

    def __init__(
        self,
        *,
        max_size: int = 50,
        l1: float = 0.0,
        objective: PoolObjective = "ic",
        lcb_beta: float = 1.0,
        ridge: float = 1e-8,
        cond_max: float = 1e6,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"`max_size` must be >= 1, got {max_size}.")
        if l1 < 0.0:
            raise ValueError(f"`l1` must be >= 0, got {l1}.")
        if objective not in ("ic", "icir", "lcb"):
            raise ValueError(
                f"`objective` must be one of 'ic', 'icir', 'lcb', got {objective!r}."
            )
        self.max_size = int(max_size)
        self.l1 = float(l1)
        self.objective: PoolObjective = objective
        self.lcb_beta = float(lcb_beta)
        self.ridge = float(ridge)
        self.cond_max = float(cond_max)
        self._m: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self._c: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._ct: np.ndarray | None = None  # (K, T) per-date ICs
        self.names: list[str] = []

    # -- state ------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._c.shape[0]

    @property
    def mutual_ic(self) -> np.ndarray:
        """The ``(K, K)`` mutual-IC matrix. Descriptor only, never a gate."""
        return self._m.copy()

    @property
    def single_ic(self) -> np.ndarray:
        """The ``(K,)`` vector of member mean ICs."""
        return self._c.copy()

    # -- solving ----------------------------------------------------------- #
    def _solve(self, m: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Minimise ``w'Mw - 2w'c`` (plus the L1 term) for ``w``."""
        k = c.shape[0]
        if k == 0:
            return np.zeros((0,), dtype=np.float64)
        a = m + self.ridge * np.eye(k, dtype=np.float64)
        if self.l1 <= 0.0:
            w: np.ndarray | None
            try:
                # NumPy 2.0 silently mis-solves ``solve(A, b)`` when the length
                # of ``b`` equals the batch dimension. The trailing-axis form is
                # the only safe spelling. See AGENTS.md invariant 4.
                w = np.linalg.solve(a, c[..., None])[..., 0]
            except np.linalg.LinAlgError:
                w = None
            # Near-duplicate alphas make `M` rank-deficient, and estimation
            # noise can leave `c` with mass in the numerically null space. The
            # ridge would then *amplify* that mass into an enormous `w` and an
            # "IC" of several hundred. Mass in an unidentifiable direction must
            # be dropped, not inflated, so fall back to the truncated
            # pseudo-inverse. Reuses `econ._common.pinv_sym`.
            norm_c = float(np.linalg.norm(c))
            if (
                w is None
                or not np.all(np.isfinite(w))
                or float(np.linalg.norm(w)) > self.cond_max * max(norm_c, 1e-12)
            ):
                from panelary.econ._common import pinv_sym  # noqa: PLC0415

                w = pinv_sym(m, rcond=1.0 / self.cond_max) @ c
            return np.asarray(w, dtype=np.float64)
        # ISTA on the smooth part 2(Mw - c) with a soft-threshold prox.
        lip = 2.0 * float(np.max(np.abs(np.linalg.eigvalsh(a)))) + 1e-12
        step = 1.0 / lip
        w = np.zeros(k, dtype=np.float64)
        for _ in range(200):
            grad = 2.0 * (a @ w - c)
            nxt = w - step * grad
            nxt = np.sign(nxt) * np.maximum(np.abs(nxt) - step * self.l1, 0.0)
            if np.max(np.abs(nxt - w)) < 1e-10:
                w = nxt
                break
            w = nxt
        return w

    def weights(self) -> np.ndarray:
        """Current least-squares (max-IR) combination weights."""
        return self._solve(self._m, self._c)

    def loss(self, w: np.ndarray | None = None) -> float:
        """``w'Mw - 2w'c + 1`` at ``w`` (default: the optimal weights)."""
        if len(self) == 0:
            return 1.0
        w = self.weights() if w is None else np.asarray(w, dtype=np.float64)
        val = float(w @ self._m @ w - 2.0 * w @ self._c + 1.0)
        if self.l1 > 0.0:
            val += self.l1 * float(np.abs(w).sum())
        return val

    # -- objective --------------------------------------------------------- #
    def _score(self, m: np.ndarray, c: np.ndarray, ct: np.ndarray | None) -> float:
        """Objective value of a pool described by ``(M, c, c_t)``. Higher better."""
        if c.shape[0] == 0:
            return 0.0
        w = self._solve(m, c)
        quad = float(w @ m @ w)
        if quad <= 0.0:
            return 0.0
        scale = math.sqrt(quad)
        # The combined signal's IC is a correlation and cannot leave [-1, 1].
        # Estimated `M` and `c` need not respect Cauchy-Schwarz, and a
        # near-singular `M` (which is precisely the near-duplicate-alpha regime
        # this pool is designed to live in) can otherwise emit an "IC" in the
        # hundreds. Clip rather than reject: the ordering stays usable.
        if self.objective == "ic" or ct is None:
            return float(np.clip(float(w @ c) / scale, -1.0, 1.0))
        series = np.clip((w @ ct) / scale, -1.0, 1.0)
        mu = _nanmean(series)
        if not np.isfinite(mu):
            return 0.0
        sd = _nanstd(series, ddof=1)
        if not np.isfinite(sd) or sd <= 0.0:
            return float(mu)
        if self.objective == "icir":
            return float(mu / sd)
        return float(mu - self.lcb_beta * sd)

    def score(self) -> float:
        """The pool's current objective value."""
        return self._score(self._m, self._c, self._ct)

    def marginal_contribution(
        self,
        ic_vec: np.ndarray,
        mutual_ic: np.ndarray | None = None,
    ) -> float:
        """Increase in the pool objective from admitting one candidate.

        This is the fitness signal. It is *not* the candidate's standalone IC:
        a candidate that is merely a rotation of what the pool already holds
        scores near zero however high its own IC, and a weak-but-orthogonal
        candidate can score highly.

        Parameters
        ----------
        ic_vec : ndarray
            The candidate's per-date IC series ``(T,)``; a scalar (or length-1
            array) is accepted and treated as a mean IC with no dispersion
            information (the ``"icir"``/``"lcb"`` objectives then fall back to
            the mean).
        mutual_ic : ndarray, optional
            The candidate's mean mutual IC against each of the ``K`` current
            members. Costs ``K`` correlations for the caller, which is the whole
            reason the update is ``O(K)``. Defaults to zeros — i.e. *assumed
            orthogonal*, which is optimistic; pass the real vector whenever the
            values are available.

        Returns
        -------
        float
            ``objective(pool + candidate) - objective(pool)``. Negative means
            the candidate makes the combination worse.
        """
        vec = np.atleast_1d(np.asarray(ic_vec, dtype=np.float64))
        mean_ic = float(_nanmean(vec))
        if not np.isfinite(mean_ic):
            return float("-inf")
        k = len(self)
        cross = (
            np.zeros(k, dtype=np.float64)
            if mutual_ic is None
            else np.asarray(mutual_ic, dtype=np.float64).ravel()
        )
        if cross.shape[0] != k:
            raise ValueError(
                f"`mutual_ic` must have one entry per pool member ({k}), got "
                f"{cross.shape[0]}."
            )
        m_new = np.empty((k + 1, k + 1), dtype=np.float64)
        m_new[:k, :k] = self._m
        m_new[:k, k] = cross
        m_new[k, :k] = cross
        m_new[k, k] = 1.0
        c_new = np.concatenate([self._c, [mean_ic]])
        ct_new: np.ndarray | None = None
        if self._ct is not None and vec.shape[0] == self._ct.shape[1]:
            ct_new = np.vstack([self._ct, vec[None, :]])
        elif self._ct is None and k == 0 and vec.shape[0] > 1:
            ct_new = vec[None, :]
        return float(self._score(m_new, c_new, ct_new) - self.score())

    # -- mutation ---------------------------------------------------------- #
    def add(
        self,
        ic_vec: np.ndarray,
        mutual_ic: np.ndarray | None = None,
        *,
        name: str | None = None,
    ) -> int:
        """Admit a candidate, evicting the least-weighted member if full.

        The update touches ``O(K)`` entries: one new row, one new column, one
        new entry of ``c``. The ``K x K`` block is never recomputed.

        Returns
        -------
        int
            The index of the admitted member, or ``-1`` if it was rejected
            because the pool was full and the candidate was itself the weakest.
        """
        vec = np.atleast_1d(np.asarray(ic_vec, dtype=np.float64))
        mean_ic = float(_nanmean(vec))
        if not np.isfinite(mean_ic):
            return -1
        k = len(self)
        cross = (
            np.zeros(k, dtype=np.float64)
            if mutual_ic is None
            else np.asarray(mutual_ic, dtype=np.float64).ravel()
        )
        if cross.shape[0] != k:
            raise ValueError(
                f"`mutual_ic` must have one entry per pool member ({k}), got "
                f"{cross.shape[0]}."
            )
        m_new = np.empty((k + 1, k + 1), dtype=np.float64)
        m_new[:k, :k] = self._m
        m_new[:k, k] = cross
        m_new[k, :k] = cross
        m_new[k, k] = 1.0
        c_new = np.concatenate([self._c, [mean_ic]])
        ct_new: np.ndarray | None
        if self._ct is not None and vec.shape[0] == self._ct.shape[1]:
            ct_new = np.vstack([self._ct, vec[None, :]])
        elif k == 0 and vec.shape[0] > 1:
            ct_new = vec[None, :]
        else:
            ct_new = self._ct
        names = [*self.names, name if name is not None else f"alpha_{k}"]

        if k + 1 > self.max_size:
            drop = int(np.argmin(np.abs(self._solve(m_new, c_new))))
            if drop == k:
                return -1
            keep = np.array([i for i in range(k + 1) if i != drop], dtype=np.int64)
            m_new = m_new[np.ix_(keep, keep)]
            c_new = c_new[keep]
            if ct_new is not None and ct_new.shape[0] == k + 1:
                ct_new = ct_new[keep]
            names = [names[i] for i in keep]
            new_index = int(np.flatnonzero(keep == k)[0])
        else:
            new_index = k

        self._m, self._c, self._ct, self.names = m_new, c_new, ct_new, names
        return new_index

    def reset(self) -> None:
        """Empty the pool. Call this between CV folds — the weights are a fit."""
        self._m = np.zeros((0, 0), dtype=np.float64)
        self._c = np.zeros((0,), dtype=np.float64)
        self._ct = None
        self.names = []


# --------------------------------------------------------------------------- #
# Null calibration
# --------------------------------------------------------------------------- #
def null_threshold(real: np.ndarray, noise: np.ndarray, *, q: float = 1.0) -> float:
    """The score a candidate must beat to be believed.

    autofeat's insight (MIT), and the only defensible acceptance rule for a
    search that examines thousands of candidates: the bar is not ``0``, it is
    **the best noise individual**. Noise individuals here are shuffled real
    columns and ``N(0, 1)`` columns pushed through the *identical* scoring path,
    so they absorb every artefact of the pipeline — the cross-section size, the
    date count, the transform, the CV geometry — that a theoretical null would
    miss. A 100,000-trial search on pure noise reaches an in-sample Sharpe near
    4.4; comparing to zero is a fabrication.

    Parameters
    ----------
    real : ndarray
        Scores of the real candidates. Used only for the degenerate case where
        no noise individual produced a finite score, in which case the threshold
        falls back to the maximum real score (accept nothing).
    noise : ndarray
        Scores of the noise individuals, from the same scoring path.
    q : float, keyword-only, default 1.0
        Quantile of the noise distribution to use. ``1.0`` is the maximum (the
        strictest, family-wise bar). ``0.95`` gives a slightly looser,
        FDR-flavoured threshold when many noise individuals are available.

    Returns
    -------
    float
        The acceptance threshold. Candidates with ``score > threshold`` are the
        ones worth reporting.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError(f"`q` must be in (0, 1], got {q}.")
    nz = np.asarray(noise, dtype=np.float64).ravel()
    nz = nz[np.isfinite(nz)]
    if nz.size == 0:
        rl = np.asarray(real, dtype=np.float64).ravel()
        rl = rl[np.isfinite(rl)]
        return float(rl.max()) if rl.size else float("inf")
    return float(np.max(nz)) if q >= 1.0 else float(np.quantile(nz, q))


def noise_matrix(
    base: np.ndarray,
    *,
    time_code: np.ndarray,
    n_shuffled: int,
    n_gaussian: int,
    seed: int,
) -> np.ndarray:
    """Build the noise individuals for :func:`null_threshold`.

    Two families, following autofeat:

    * **shuffled real columns** — a real base column permuted *within each
      date*, which preserves the column's cross-sectional distribution, its
      missingness pattern and its scale while destroying every relationship to
      the target. This is the strictly harder null;
    * **``N(0, 1)`` columns** — the naive null, kept because it calibrates the
      part of the score that comes from the CV geometry alone.

    Parameters
    ----------
    base : ndarray, shape (n_rows, n_base)
        The real base columns, in date-contiguous row order.
    time_code : ndarray, keyword-only
        Non-decreasing date codes aligned with ``base``.
    n_shuffled, n_gaussian : int, keyword-only
        How many of each family to generate.
    seed : int, keyword-only
        Explicit seed; the module never touches global RNG state.

    Returns
    -------
    ndarray, shape (n_rows, n_shuffled + n_gaussian)
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(base, dtype=np.float64)
    n_rows = base.shape[0]
    cols: list[np.ndarray] = []
    if n_shuffled > 0 and base.shape[1] > 0:
        n_times = int(time_code[-1]) + 1 if n_rows else 0
        starts = _group_starts(np.asarray(time_code, dtype=np.int64), n_times)
        for j in range(n_shuffled):
            src = base[:, j % base.shape[1]].copy()
            for lo, hi in zip(starts[:-1], starts[1:], strict=True):
                if hi > lo:
                    src[lo:hi] = rng.permutation(src[lo:hi])
            cols.append(src)
    for _ in range(n_gaussian):
        cols.append(np.asarray(rng.standard_normal(n_rows), dtype=np.float64))
    if not cols:
        return np.empty((n_rows, 0), dtype=np.float64)
    return np.column_stack(cols)


# --------------------------------------------------------------------------- #
# Behaviour descriptors
# --------------------------------------------------------------------------- #
def descriptors(
    values: np.ndarray,
    *,
    time_code: np.ndarray,
    entity_code: np.ndarray,
    n_times: int,
    n_entities: int,
    complexity: int | Sequence[int] = 0,
    library: np.ndarray | None = None,
    max_lag: int = 32,
    max_dense_cells: int = 50_000_000,
    seed: int = 0,
) -> Descriptors | list[Descriptors]:
    """Locate genomes in the quality-diversity archive.

    The five axes of :class:`~panelary.evolve._types.Descriptors`:

    ``turnover``
        ``1 - mean_t corr(rank_t, rank_{t-1})``. Separates slow value-like
        signals from fast reversal-like ones — an economic distinction, not a
        syntactic one.
    ``horizon``
        The lag at which the mean **per-entity** ACF first drops below ``1/e``.
        Computed within each entity's own series, never pooled.
    ``max_corr``
        Largest absolute mean cross-sectional correlation against ``library``.
        A *descriptor*, not a gate — see :class:`AlphaPool`.
    ``complexity``
        Active node count, supplied by the caller (``_genome.complexity``).
    ``coverage``
        Fraction of ``(entity, time)`` cells where the signal is finite.

    **All of this must be computed on training folds only.** Pass row-subset
    arrays, not the full panel: turnover and horizon are properties of the
    signal's dynamics, and measuring them on the test fold is a (mild, but
    real) leak into the archive's structure.

    Parameters
    ----------
    values : ndarray, shape (n_rows,) or (n_rows, k)
        Signal values in date-contiguous row order.
    time_code, entity_code : ndarray, keyword-only
        Dense integer codes; ``time_code`` must be non-decreasing.
    n_times, n_entities : int, keyword-only
    complexity : int or sequence of int, keyword-only, default 0
    library : ndarray, optional, keyword-only
        ``(n_rows, n_lib)`` reference signals for the correlation axis.
    max_lag : int, keyword-only, default 32
        Censoring point for the horizon search.
    max_dense_cells : int, keyword-only, default 50_000_000
        Guard for very wide panels: above this, entities are subsampled
        (deterministically, from ``seed``) before the dense cube is built.
    seed : int, keyword-only, default 0

    Returns
    -------
    Descriptors or list[Descriptors]
        A single object for 1-D ``values``, else one per column.
    """
    vals = np.asarray(values, dtype=np.float64)
    single = vals.ndim == 1
    if single:
        vals = vals[:, None]
    k = vals.shape[1]
    tc = np.asarray(time_code, dtype=np.int64)
    ec = np.asarray(entity_code, dtype=np.int64)
    starts = _group_starts(tc, n_times)

    coverage = np.isfinite(vals).sum(axis=0) / max(float(n_times * n_entities), 1.0)
    ranks = _transform_columns(vals, starts, "rank_ic")

    # Correlation to the library: per-date standardisation makes the global
    # matmul equal the sum over dates of the per-date cross-sectional
    # correlations, so this is one BLAS call rather than T * n_lib of them.
    z_self, counts = _standardize_by_date(ranks, starts)
    if library is not None and np.asarray(library).size:
        lib = np.asarray(library, dtype=np.float64)
        if lib.ndim == 1:
            lib = lib[:, None]
        z_lib, _ = _standardize_by_date(
            _transform_columns(lib, starts, "rank_ic"), starts
        )
        n_ok = np.maximum((counts > 0).sum(axis=0), 1)
        cross = np.abs((z_lib.T @ z_self) / n_ok[None, :])
        max_corr = np.clip(cross.max(axis=0), 0.0, 1.0)
    else:
        max_corr = np.zeros(k, dtype=np.float64)

    # Dense cube for the two dynamic axes; subsample entities if it would blow up.
    ec_use, n_ent_use, row_keep = ec, n_entities, None
    if n_times * n_entities > max_dense_cells:
        budget = max(2, max_dense_cells // max(n_times, 1))
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(n_entities, size=budget, replace=False))
        remap = np.full(n_entities, -1, dtype=np.int64)
        remap[chosen] = np.arange(chosen.shape[0], dtype=np.int64)
        row_keep = remap[ec] >= 0
        ec_use, n_ent_use = remap[ec][row_keep], int(chosen.shape[0])

    tc_use = tc if row_keep is None else tc[row_keep]
    rk_use = ranks if row_keep is None else ranks[row_keep]
    dense = _dense_by_entity(rk_use, tc_use, ec_use, n_times, n_ent_use)
    turn = np.atleast_1d(np.asarray(1.0 - np.asarray(_mean_lag_autocorr(dense))))
    hor = _acf_horizon(dense, max_lag=max_lag)

    comp = (
        [int(complexity)] * k
        if isinstance(complexity, (int, np.integer))
        else [int(c) for c in complexity]
    )
    if len(comp) != k:
        raise ValueError(f"`complexity` must have {k} entries, got {len(comp)}.")

    out = [
        Descriptors(
            turnover=float(np.nan_to_num(turn[j], nan=1.0)),
            horizon=float(hor[j]),
            max_corr=float(max_corr[j]),
            complexity=comp[j],
            coverage=float(coverage[j]),
        )
        for j in range(k)
    ]
    return out[0] if single else out


# --------------------------------------------------------------------------- #
# The leak-safe harness
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Fold:
    """One purged, embargoed split, expressed in *date positions*."""

    train: np.ndarray
    test: np.ndarray
    label: str


def _bucket_dates(dates: np.ndarray, n_buckets: int) -> list[np.ndarray]:
    """Split a fold's test dates into contiguous, time-ordered buckets."""
    if dates.size == 0:
        return []
    n = max(1, min(n_buckets, dates.size))
    return [b for b in np.array_split(np.sort(dates), n) if b.size]


class PanelEvaluator:
    """Out-of-fold, purged, embargoed fitness for a population of genomes.

    Implements :class:`panelary.evolve._types.Evaluator`.

    What makes it leak-safe
    -----------------------
    * Folds come from :class:`~panelary.core.model_selection.PurgedKFold`
      or :class:`~panelary.core.model_selection.CombinatorialPurgedCV`,
      operating on the sorted unique time index. Random or stratified splitting
      is not offered, because on a panel it is simply wrong.
    * ``embargo`` defaults to ``max(1, horizon)`` and is **on**. Passing a
      smaller value warns rather than silently obliging: with a horizon-``h``
      label, the ``h`` observations after a test block share information with it
      through the label window itself.
    * Every fitted quantity is fitted per fold on the training dates: the sign
      orientation, the :class:`AlphaPool` weights, the ridge/LightGBM baseline
      of ``mode="residual"``, and the behaviour descriptors. Ambroise &
      McLachlan (PNAS 2002) is the canonical demonstration of what happens
      otherwise — full-sample selection reports near-zero CV error on
      **permuted** labels, while fold-local selection correctly reports the
      0.40-0.45 of noise. ``tests`` for this module reproduce both numbers.
    * Per-date correlations are computed within a date and only then aggregated,
      so no quantity ever spans a date boundary.

    Successive halving
    ------------------
    Cheap rungs use only the earliest folds and survivors advance to later ones,
    adapted from OpenFE's evaluator with one deliberate change: **OpenFE
    shuffles its blocks and we must not.** Keeping blocks in time order makes
    the halving schedule double as a walk-forward check — a genome that only
    works late in the sample is eliminated on the early rungs. (OpenFE's own
    ablation found the mutual-information and correlation shortcuts strictly
    worse than evaluating the model, so no such shortcut is offered here.)

    It is off by default in ``mode="ic"`` and that is a measurement, not an
    oversight: with a fixed target the per-date IC matrix does not depend on the
    fold, so it is computed once per column chunk and the rungs would only
    re-rank the same rows (0.84 s with halving against 0.58 s without, on 64
    columns x 90k rows). In ``mode="residual"`` every fold refits a baseline, so
    eliminating candidates early removes real work and halving is on.

    Parameters
    ----------
    panel : polars.DataFrame | polars.LazyFrame | PanelFrame
        Long-format panel carrying the base columns and the target.
    ctx : EvalContext
        Supplies ``entity``, ``time`` and ``base_columns`` for the compiler.
    target : str
        Name of the **already leak-safe** forward-return column (build it with
        :func:`panelary.factor.forward_return`). Rows with a null target
        are dropped up front, which also makes the per-date rank arithmetic
        exact.
    horizon : int, keyword-only, default 1
        Label horizon in time-steps; drives purging.
    embargo : int | None, keyword-only, default None
        ``None`` -> ``max(1, horizon)``.
    n_splits : int, keyword-only, default 5
    cv : {"purged_kfold", "cpcv"}, keyword-only, default "purged_kfold"
    n_test_groups : int, keyword-only, default 2
        Only for ``cv="cpcv"``.
    metric : {"rank_ic", "numerai", "pearson"}, keyword-only, default "rank_ic"
    mode : {"ic", "residual"}, keyword-only, default "ic"
        ``"ic"`` scores the signal directly. ``"residual"`` scores it against
        the residual of a per-fold baseline fitted on the library features —
        OpenFE's "score by residual reduction, not refit".
    base_model : {"ridge", "lightgbm"}, keyword-only, default "ridge"
        Baseline for ``mode="residual"``. The pure-numpy ridge is the default so
        the core stays numpy+polars; LightGBM is routed through
        :func:`panelary._internal._deps.require`.
    pool : AlphaPool | None, keyword-only
        When given, the aggregate score is the candidate's **marginal
        contribution** to this pool rather than its standalone IC. Pool weights
        are refitted per fold on training dates only.
    library : ndarray | Sequence[str] | None, keyword-only
        Reference signals for ``max_corr_to_library`` and for the residual
        baseline. Column names are resolved against the panel.
    orient_sign : bool, keyword-only, default True
        Flip each candidate to its training-fold IC sign before scoring the test
        fold. This is a *fit*, and doing it per fold is what keeps the shuffled-
        label null at zero; doing it on the full sample is a textbook leak.
    n_time_buckets : int | None, keyword-only, default None
        Time buckets per fold for the :class:`CaseMatrix`. ``None`` targets
        roughly 120 cases in total (inside the 50-500 band the type documents).
    halving : bool | None, keyword-only, default None
        ``None`` picks the setting that actually helps: **on** for
        ``mode="residual"``, **off** for ``mode="ic"``. Measured on 64 columns x
        90k rows: halving cost 0.84 s against 0.58 s without it in ``"ic"``
        mode, because there the per-date IC matrix is fold-independent and
        already computed once, so the rungs re-rank rows for nothing. In
        ``"residual"`` mode each fold refits a baseline, so dropping candidates
        early removes real work. Pass an explicit bool to override.
    halving_rate : float, keyword-only, default 0.5
    min_survivors : int, keyword-only, default 8
    chunk_size : int, keyword-only, default 64
        Genomes scored per pass. Bounds peak memory at
        ``n_rows * chunk_size * 8`` bytes for the rank matrix.
    seed : int, keyword-only, default 0
    compile_fn : callable, optional, keyword-only
        Override for ``_compile.compile_population`` (imported lazily so this
        module is usable before the compiler lands, and testable without it).
        It is called as ``compile_fn(genomes, ctx, lf, keep=(row_col,))`` and
        must carry the ``keep`` columns through, as ``compile_population``
        does. Rows may come back in any order: values are matched to targets
        through that row-index column, never by position, so
        ``compile_population``'s ``sort=True`` and ``sort=False`` score alike.
    """

    def __init__(
        self,
        panel: pl.DataFrame | pl.LazyFrame | PanelFrame,
        *,
        ctx: EvalContext,
        target: str,
        horizon: int = 1,
        embargo: int | None = None,
        n_splits: int = 5,
        cv: Literal["purged_kfold", "cpcv"] = "purged_kfold",
        n_test_groups: int = 2,
        metric: Metric = "rank_ic",
        mode: Literal["ic", "residual"] = "ic",
        base_model: Literal["ridge", "lightgbm"] = "ridge",
        ridge_alpha: float = 1.0,
        pool: AlphaPool | None = None,
        library: np.ndarray | Sequence[str] | None = None,
        orient_sign: bool = True,
        n_time_buckets: int | None = None,
        halving: bool | None = None,
        halving_rate: float = 0.5,
        min_survivors: int = 8,
        chunk_size: int = 64,
        max_lag: int = 32,
        seed: int = 0,
        compile_fn: Callable[..., tuple[pl.LazyFrame, list[str]]] | None = None,
    ) -> None:
        if horizon < 0:
            raise ValueError(f"`horizon` must be >= 0, got {horizon}.")
        if metric not in ("rank_ic", "numerai", "pearson"):
            raise ValueError(f"unknown `metric` {metric!r}.")
        if mode not in ("ic", "residual"):
            raise ValueError(f"`mode` must be 'ic' or 'residual', got {mode!r}.")
        if base_model not in ("ridge", "lightgbm"):
            raise ValueError(f"unknown `base_model` {base_model!r}.")
        if not (0.0 < halving_rate < 1.0):
            raise ValueError(f"`halving_rate` must be in (0, 1), got {halving_rate}.")

        if embargo is None:
            embargo = max(1, horizon)
        elif embargo < horizon:
            warnings.warn(
                f"embargo={embargo} is shorter than the label horizon "
                f"({horizon}). With a horizon-{horizon} label the observations "
                "immediately after a test block share information with it "
                "through the label window; the default embargo=max(1, horizon) "
                "exists to remove exactly those. Proceeding as asked.",
                UserWarning,
                stacklevel=2,
            )

        self.ctx = ctx
        self.target = target
        self.horizon = int(horizon)
        self.embargo = int(embargo)
        self.n_splits = int(n_splits)
        self.cv = cv
        self.n_test_groups = int(n_test_groups)
        self.metric: Metric = metric
        self.mode = mode
        self.base_model = base_model
        self.ridge_alpha = float(ridge_alpha)
        self.pool = pool
        self.orient_sign = bool(orient_sign)
        self.halving = (mode == "residual") if halving is None else bool(halving)
        self.halving_rate = float(halving_rate)
        self.min_survivors = int(min_survivors)
        self.chunk_size = int(chunk_size)
        self.max_lag = int(max_lag)
        self.seed = int(seed)
        self._compile_fn = compile_fn

        self._lf = self._normalise_panel(panel)
        self._materialise_keys()
        self._library = self._resolve_library(library)
        self._folds = self._build_folds()
        self._n_time_buckets = (
            int(n_time_buckets)
            if n_time_buckets is not None
            else max(1, round(120 / max(len(self._folds), 1)))
        )
        self._cases = self._build_cases()

        #: Populated by :meth:`evaluate`; the search loop reads these.
        self.last_cases: CaseMatrix | None = None
        self.last_descriptors: list[Descriptors] = []
        self.last_null_threshold: float = float("nan")

    # -- setup ------------------------------------------------------------- #
    def _normalise_panel(
        self, panel: pl.DataFrame | pl.LazyFrame | PanelFrame
    ) -> pl.LazyFrame:
        lf = panel.lazy() if not isinstance(panel, PanelFrame) else panel.lazy()
        names = lf.collect_schema().names()
        for col in (self.ctx.entity, self.ctx.time, self.target):
            if col not in names:
                raise ValueError(f"column {col!r} not found in panel; have {names}.")
        missing = [c for c in self.ctx.base_columns if c not in names]
        if missing:
            raise ValueError(f"base columns {missing} not found in panel.")
        if _ROW in names:
            raise ValueError(
                f"column {_ROW!r} is reserved by the evaluator; rename it."
            )
        # Sorting by (time, entity) is doubly load-bearing: dates become
        # contiguous slices (the whole reduceat design), *and* every entity's
        # rows stay in time order, which is what `.over(entity)` rolling
        # operators require of the compiler.
        #
        # The row index records that order. The compiler is free to return
        # rows in another order -- `compile_population` sorts by
        # (entity, time) by default -- so `_materialise` puts every compiled
        # value back at its evaluator row through this index before a single
        # value is paired with `self.y`.
        return (
            lf.drop_nulls(subset=[self.target])
            .sort([self.ctx.time, self.ctx.entity])
            .with_row_index(_ROW)
        )

    def _materialise_keys(self) -> None:
        keys = self._lf.select(
            self.ctx.time, self.ctx.entity, pl.col(self.target).cast(pl.Float64)
        ).collect()
        self._time_values = keys[self.ctx.time]
        tc, n_times = _codes(keys[self.ctx.time].to_physical().to_numpy())
        ec, n_ent = _codes(keys[self.ctx.entity].to_physical().to_numpy())
        self.time_code = tc
        self.entity_code = ec
        self.n_times = n_times
        self.n_entities = n_ent
        self.starts = _group_starts(tc, n_times)
        self.y = np.asarray(keys[self.target].to_numpy(), dtype=np.float64)
        self.n_rows = self.y.shape[0]
        if n_times < 4:
            raise ValueError(
                f"need at least 4 distinct dates to cross-validate, got {n_times}."
            )

    def _resolve_library(
        self, library: np.ndarray | Sequence[str] | None
    ) -> np.ndarray:
        if library is None:
            return np.empty((self.n_rows, 0), dtype=np.float64)
        if isinstance(library, np.ndarray):
            arr = np.asarray(library, dtype=np.float64)
            return arr[:, None] if arr.ndim == 1 else arr
        cols = list(library)
        if not cols:
            return np.empty((self.n_rows, 0), dtype=np.float64)
        return (
            self._lf.select([pl.col(c).cast(pl.Float64) for c in cols])
            .collect()
            .to_numpy()
            .astype(np.float64, copy=False)
        )

    def _build_folds(self) -> list[_Fold]:
        """Purged, embargoed folds as *date-position* arrays."""
        key_df = pl.DataFrame(
            {
                self.ctx.entity: pl.Series(np.zeros(self.n_times, dtype=np.int64)),
                self.ctx.time: pl.Series(np.arange(self.n_times, dtype=np.int64)),
            }
        )
        pf = PanelFrame(key_df, entity=self.ctx.entity, time=self.ctx.time)
        if self.cv == "cpcv":
            splitter: Any = CombinatorialPurgedCV(
                n_groups=self.n_splits,
                n_test_groups=self.n_test_groups,
                horizon=self.horizon,
                embargo=self.embargo,
                return_indices=True,
            )
        else:
            splitter = PurgedKFold(
                n_splits=self.n_splits,
                horizon=self.horizon,
                embargo=self.embargo,
                return_indices=True,
            )
        folds = [
            _Fold(
                train=np.asarray(tr, dtype=np.int64),
                test=np.asarray(te, dtype=np.int64),
                label=f"fold{i}",
            )
            for i, (tr, te) in enumerate(splitter.split(pf))
        ]
        usable = [f for f in folds if f.train.size and f.test.size]
        if not usable:
            raise ValueError(
                "every fold came back empty after purging and embargo — reduce "
                f"`horizon` ({self.horizon}) / `embargo` ({self.embargo}), or "
                f"supply more than {self.n_times} dates."
            )
        return usable

    def _build_cases(self) -> list[tuple[int, np.ndarray, str]]:
        """(fold index, test-date positions, label) per lexicase case."""
        cases: list[tuple[int, np.ndarray, str]] = []
        for i, fold in enumerate(self._folds):
            for b, dates in enumerate(_bucket_dates(fold.test, self._n_time_buckets)):
                cases.append((i, dates, f"{fold.label}:b{b}"))
        return cases

    @property
    def case_labels(self) -> tuple[str, ...]:
        """Labels of the ``(fold x time-bucket)`` cases, in matrix order."""
        return tuple(lbl for _, _, lbl in self._cases)

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    @property
    def folds(self) -> list[_Fold]:
        """The purged, embargoed folds, as date positions. Exposed for tests."""
        return list(self._folds)

    # -- compilation ------------------------------------------------------- #
    def _materialise(self, genomes: Sequence[Genome]) -> np.ndarray:
        """Compile a population to an ``(n_rows, n_genomes)`` float64 matrix."""
        fn = self._compile_fn
        if fn is None:  # pragma: no cover - exercised once _compile.py lands
            from panelary.evolve._compile import compile_population

            fn = compile_population
        out_lf, cols = fn(list(genomes), self.ctx, self._lf, keep=(_ROW,))
        # `compile_population` returns one name per genome *in genome order*,
        # and semantically identical genomes deliberately collapse onto a single
        # shared DAG node — so `cols` may repeat a name. That is the
        # deduplication working, and 60-90% semantic duplicates is normal in a
        # GP population, so this is the common path, not an edge case.
        # Projecting `cols` raw would raise polars' DuplicateError; project the
        # distinct names once and re-expand, which preserves both the
        # (n_rows, n_genomes) shape and the genome order. Every duplicate keeps
        # its own FitnessResult: each consumed a trial, and the TrialLedger has
        # to count them or `N` is understated and the deflation too weak.
        uniq = list(dict.fromkeys(cols))
        frame = out_lf.select(
            [pl.col(_ROW), *[pl.col(c).cast(pl.Float64) for c in uniq]]
        ).collect()
        if frame.height != self.n_rows:
            raise ValueError(
                f"the compiler returned {frame.height} rows but the evaluator "
                f"holds {self.n_rows}; the compiled frame must preserve row "
                "cardinality."
            )
        # The compiler may reorder rows (`compile_population` returns them in
        # (entity, time) order unless `sort=False`), while `self.y` is in the
        # evaluator's (time, entity) order. Pairing the two by position scores
        # every formula against a shuffled target. Scatter each compiled row
        # back to its evaluator row instead, so the result does not depend on
        # the order the compiler chose.
        row = frame[_ROW].to_numpy().astype(np.int64, copy=False)
        if not np.array_equal(
            np.bincount(row, minlength=self.n_rows), np.ones(self.n_rows, np.int64)
        ):
            raise ValueError(
                "the compiled frame must carry every evaluator row exactly once "
                f"(via the {_ROW!r} column passed in `keep`)."
            )
        pos = {name: i for i, name in enumerate(uniq)}
        mat = np.empty((self.n_rows, len(uniq)), dtype=np.float64)
        mat[row] = frame.select(uniq).to_numpy()
        return mat[:, [pos[c] for c in cols]]

    # -- baselines --------------------------------------------------------- #
    def _ridge_residual(self, train_rows: np.ndarray) -> np.ndarray:
        """Target residualised on the library, ridge fitted on ``train_rows``.

        Pure numpy, and the **default** baseline — the core stays numpy+polars.
        The fit sees training rows only; the residual is then evaluated
        everywhere, which is the OpenFE "score the residual, don't refit" idea
        without OpenFE's dependency footprint.
        """
        if self._library.shape[1] == 0:
            return self.y - self.y[train_rows].mean()
        x = np.nan_to_num(self._library, nan=0.0)
        xt = np.column_stack([np.ones(x.shape[0]), x])
        a = xt[train_rows]
        b = self.y[train_rows]
        gram = a.T @ a + self.ridge_alpha * np.eye(xt.shape[1], dtype=np.float64)
        rhs = a.T @ b
        try:
            # Trailing-axis form: NumPy 2.0 mis-solves `solve(A, b)` when the
            # length of `b` matches the batch dimension (AGENTS.md invariant 4).
            beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate library
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        return self.y - xt @ beta

    def _lightgbm_residual(self, train_rows: np.ndarray) -> np.ndarray:
        """Optional LightGBM baseline with an ``init_score`` warm start.

        FeatureBoost/OpenFE geometry: 100 trees, 16 leaves, early stopping with
        patience 3, warm-started from the ridge prediction via ``init_score`` so
        the booster only has to learn what ridge could not. Behind
        :func:`panelary._internal._deps.require`; the ridge path above is the
        default and this is never on the import path.
        """
        from panelary._internal._deps import require

        lgb = require("lightgbm", extra="lightgbm", feature="evolve residual scoring")
        if self._library.shape[1] == 0:
            return self._ridge_residual(train_rows)

        x = np.nan_to_num(self._library, nan=0.0)
        base = self.y - self._ridge_residual(train_rows)  # the ridge prediction
        # A time-ordered tail of the training dates is the validation set: never
        # a random split, or early stopping itself becomes a leak.
        cut = max(1, int(train_rows.shape[0] * 0.8))
        fit_rows, val_rows = train_rows[:cut], train_rows[cut:]
        if val_rows.size < 8:
            fit_rows, val_rows = train_rows, train_rows
        dtrain = lgb.Dataset(
            x[fit_rows], label=self.y[fit_rows], init_score=base[fit_rows]
        )
        dvalid = lgb.Dataset(
            x[val_rows],
            label=self.y[val_rows],
            init_score=base[val_rows],
            reference=dtrain,
        )
        booster = lgb.train(
            {
                "objective": "regression",
                "num_leaves": 16,
                "learning_rate": 0.05,
                "verbosity": -1,
                "seed": self.seed,
                "deterministic": True,
                "force_row_wise": True,
            },
            dtrain,
            num_boost_round=100,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(3, verbose=False)],
        )
        return self.y - (base + booster.predict(x))

    # -- the scoring core -------------------------------------------------- #
    def _row_mask(self, dates: np.ndarray) -> np.ndarray:
        keep = np.zeros(self.n_times, dtype=bool)
        keep[dates] = True
        return keep[self.time_code]

    def _target_matrix(self, y: np.ndarray) -> np.ndarray:
        """Per-date standardised target, shaped ``(n_rows, 1)``."""
        z, _ = _standardize_by_date(
            _transform_target(y, self.starts, self.metric), self.starts
        )
        return z

    def _ic_matrix(
        self, values: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-date ICs of a column block.

        Returns ``(ic (T, k), z (n_rows, k), counts (T, k))``. The per-date IC is
        computed for the whole time axis at once — legitimate, because a per-date
        correlation cannot see across dates. Everything *fitted* (sign, pool
        weights, baseline) is restricted to training dates by the callers.
        """
        transformed = _transform_columns(values, self.starts, self.metric)
        z, counts = _standardize_by_date(transformed, self.starts)
        zy = self._target_matrix(y)
        ic = _per_date_ic(z, zy[:, 0], self.starts, counts)
        return ic, z, counts

    def _mutual_ic(self, z_a: np.ndarray, z_b: np.ndarray, n_dates: int) -> np.ndarray:
        """Mean per-date cross-sectional correlation between two column blocks.

        ``z_a.T @ z_b`` over the selected rows equals the *sum* over dates of the
        per-date correlations, because each column was standardised within its
        date. One matmul instead of ``T * n_a * n_b`` correlations.
        """
        if z_a.shape[1] == 0 or z_b.shape[1] == 0:
            return np.zeros((z_a.shape[1], z_b.shape[1]), dtype=np.float64)
        return (z_a.T @ z_b) / max(n_dates, 1)

    def _halving_rungs(self, n_pop: int) -> list[tuple[list[int], int]]:
        """``(fold indices for this rung, survivors to keep)``, time-ordered."""
        n_folds = len(self._folds)
        if not self.halving or n_folds < 2 or n_pop <= self.min_survivors:
            return [(list(range(n_folds)), n_pop)]
        rungs: list[tuple[list[int], int]] = []
        survivors = n_pop
        # Cumulative prefixes of the fold list. Folds are contiguous time blocks
        # in ascending order, so prefix r is "the first r blocks of history".
        cuts = sorted({max(1, (n_folds * (i + 1)) // 3) for i in range(3)} | {n_folds})
        for i, cut in enumerate(cuts):
            last = i == len(cuts) - 1
            survivors = (
                n_pop
                if last
                else max(
                    self.min_survivors, int(math.ceil(survivors * self.halving_rate))
                )
            )
            rungs.append((list(range(cut)), survivors))
        return rungs

    def _score_columns(
        self,
        values: np.ndarray,
        fold_ids: Sequence[int],
        cols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score ``values[:, cols]`` on ``fold_ids``.

        Returns ``(per_case (k, n_cases) with NaN outside ``fold_ids``,
        aggregate (k,))``. The aggregate is the mean out-of-fold IC, or the mean
        marginal contribution to :attr:`pool` when one is attached.
        """
        block = values[:, cols]
        k = block.shape[1]
        per_case = np.full((k, self.n_cases), np.nan, dtype=np.float64)
        fold_scores = np.full((len(fold_ids), k), np.nan, dtype=np.float64)

        # In `mode="ic"` the target never changes, so the per-date IC matrix is
        # fold-independent: compute it once instead of `n_folds` times. This is
        # not a shortcut around leak-safety — a per-date correlation cannot see
        # across dates, and everything *fitted* below is still restricted to the
        # fold's training dates. Measured: 4.4x on 200 columns x 200k rows.
        shared = None if self.mode == "residual" else self._ic_matrix(block, self.y)
        lib_z = None
        if self.pool is not None:
            lib_z, _ = _standardize_by_date(
                _transform_columns(self._library, self.starts, self.metric),
                self.starts,
            )

        for fi, fold_id in enumerate(fold_ids):
            fold = self._folds[fold_id]
            if shared is not None:
                ic, z, _counts = shared
            else:
                train_rows = np.flatnonzero(self._row_mask(fold.train))
                y = (
                    self._lightgbm_residual(train_rows)
                    if self.base_model == "lightgbm"
                    else self._ridge_residual(train_rows)
                )
                ic, z, _counts = self._ic_matrix(block, y)

            # --- the only fitted quantities, and they see training dates only.
            sign = np.ones(k, dtype=np.float64)
            if self.orient_sign:
                train_ic = _nanmean(ic[fold.train], axis=0)
                sign = np.where(np.asarray(train_ic) < 0.0, -1.0, 1.0)

            test_ic = ic[fold.test] * sign[None, :]

            if self.pool is not None and lib_z is not None:
                mask = self._row_mask(fold.train)
                mutual = self._mutual_ic(
                    lib_z[mask], z[mask] * sign[None, :], fold.train.size
                )
                agg = np.array(
                    [
                        self.pool.marginal_contribution(
                            test_ic[:, j],
                            mutual[:, j] if mutual.shape[0] == len(self.pool) else None,
                        )
                        for j in range(k)
                    ]
                )
            else:
                agg = np.asarray(_nanmean(test_ic, axis=0), dtype=np.float64)
            fold_scores[fi] = agg

            for ci, (case_fold, dates, _lbl) in enumerate(self._cases):
                if case_fold != fold_id:
                    continue
                per_case[:, ci] = _nanmean(ic[dates] * sign[None, :], axis=0)

        return per_case, np.asarray(_nanmean(fold_scores, axis=0), dtype=np.float64)

    def score_matrix(
        self, values: np.ndarray, *, complexity: Sequence[int] | None = None
    ) -> tuple[list[FitnessResult], CaseMatrix]:
        """Score a raw ``(n_rows, k)`` value matrix through the full harness.

        Exposed because it is the single scoring path: genomes, noise
        individuals and library columns all go through *this* function, which is
        what makes :func:`null_threshold` comparable to the real scores.

        Parameters
        ----------
        values : ndarray, shape (n_rows, k)
            Signals in the evaluator's row order.
        complexity : sequence of int, optional, keyword-only
            Active node counts, one per column.

        Returns
        -------
        (list[FitnessResult], CaseMatrix)
        """
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
        if vals.shape[0] != self.n_rows:
            raise ValueError(f"expected {self.n_rows} rows, got {vals.shape[0]}.")
        k = vals.shape[1]
        comp = [0] * k if complexity is None else [int(c) for c in complexity]
        if len(comp) != k:
            raise ValueError(f"`complexity` must have {k} entries, got {len(comp)}.")

        per_case = np.full((k, self.n_cases), np.nan, dtype=np.float64)
        agg = np.full(k, np.nan, dtype=np.float64)
        alive = np.arange(k, dtype=np.int64)

        for fold_ids, survivors in self._halving_rungs(k):
            for lo in range(0, alive.shape[0], self.chunk_size):
                cols = alive[lo : lo + self.chunk_size]
                pc, sc = self._score_columns(vals, fold_ids, cols)
                # A later rung strictly extends the fold set, so it supersedes.
                per_case[cols] = np.where(np.isnan(pc), per_case[cols], pc)
                agg[cols] = sc
            if survivors < alive.shape[0]:
                ranked = alive[np.argsort(-np.nan_to_num(agg[alive], nan=-np.inf))]
                alive = np.sort(ranked[:survivors])

        # Eliminated genomes never saw the late cases. Impute them with the
        # population's per-case minimum: pessimistic, deterministic, and it
        # keeps the matrix finite for epsilon-lexicase (which cannot rank NaN).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            floor = np.nanmin(per_case, axis=0)
        floor = np.where(np.isfinite(floor), floor, 0.0)
        per_case = np.where(np.isnan(per_case), floor[None, :], per_case)
        agg = np.where(np.isfinite(agg), agg, -np.inf)

        # Descriptors: training dates of the first fold only. The archive's
        # geometry is a fitted object too.
        train_rows = self._row_mask(self._folds[0].train)
        descs = descriptors(
            vals[train_rows],
            time_code=self.time_code[train_rows],
            entity_code=self.entity_code[train_rows],
            n_times=self.n_times,
            n_entities=self.n_entities,
            complexity=comp,
            library=self._library[train_rows] if self._library.shape[1] else None,
            max_lag=self.max_lag,
            seed=self.seed,
        )
        desc_list = descs if isinstance(descs, list) else [descs]
        self.last_descriptors = desc_list

        cases = CaseMatrix(
            scores=per_case.astype(np.float32), case_labels=self.case_labels
        )
        self.last_cases = cases
        results = [
            FitnessResult(
                score=float(agg[j]),
                per_case=per_case[j].astype(np.float64),
                complexity=comp[j],
                turnover=desc_list[j].turnover,
                max_corr_to_library=desc_list[j].max_corr,
                coverage=desc_list[j].coverage,
                n_evals=1,
            )
            for j in range(k)
        ]
        return results, cases

    # -- the Evaluator protocol -------------------------------------------- #
    def evaluate(self, genomes: Sequence[Genome], /) -> list[FitnessResult]:
        """Score a population out-of-fold. Implements ``_types.Evaluator``."""
        if not genomes:
            return []
        from panelary.evolve._compile import active_nodes  # noqa: PLC0415

        try:
            comp = [len(active_nodes(g)) for g in genomes]
        except Exception:  # pragma: no cover - compiler not yet importable
            comp = [len(g.genes) for g in genomes]
        results, _ = self.score_matrix(self._materialise(genomes), complexity=comp)
        return results

    def noise_scores(
        self, *, n_shuffled: int = 8, n_gaussian: int = 8, seed: int | None = None
    ) -> np.ndarray:
        """Score noise individuals through the identical path.

        Feed the result to :func:`null_threshold` together with the real scores.
        """
        results = self._noise_results(
            n_shuffled=n_shuffled, n_gaussian=n_gaussian, seed=seed
        )
        return np.array([r.score for r in results], dtype=np.float64)

    def _noise_results(
        self, *, n_shuffled: int, n_gaussian: int, seed: int | None = None
    ) -> list[FitnessResult]:
        """Noise individuals scored through the identical path, in full.

        :meth:`noise_scores` keeps only the aggregate; the search also needs
        each decoy's per-case series, to calibrate the null the trial ledger
        deflates against.
        """
        base = (
            self._lf.select([pl.col(c).cast(pl.Float64) for c in self.ctx.base_columns])
            .collect()
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        noise = noise_matrix(
            base,
            time_code=self.time_code,
            n_shuffled=n_shuffled,
            n_gaussian=n_gaussian,
            seed=self.seed if seed is None else seed,
        )
        if noise.shape[1] == 0:
            return []
        saved = self.halving
        self.halving = False  # every noise individual must see every fold
        try:
            results, _ = self.score_matrix(noise)
        finally:
            self.halving = saved
        return results

    def calibrate(
        self,
        results: Sequence[FitnessResult],
        *,
        n_shuffled: int = 8,
        n_gaussian: int = 8,
        q: float = 1.0,
    ) -> float:
        """Set :attr:`last_null_threshold` from a fresh batch of noise."""
        real = np.array([r.score for r in results], dtype=np.float64)
        noise = self.noise_scores(n_shuffled=n_shuffled, n_gaussian=n_gaussian)
        self.last_null_threshold = null_threshold(real, noise, q=q)
        return self.last_null_threshold
