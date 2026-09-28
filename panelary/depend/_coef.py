"""Rank and correlation coefficients: Pearson, Spearman, Kendall, Chatterjee's
xi, Hoeffding's D, and nonparametric tail dependence.

Every statistic has two faces:

* a public scalar function (``xi(x, y)``) that drops incomplete rows with
  :func:`~panelary.depend._ranks.pairwise_complete` and returns a float, and
* a private **row-batched** kernel (``_xi_rows(X, Y)`` on ``(B, m)`` arrays)
  that computes ``B`` statistics in one vectorised pass. The batched kernels are
  what make resampling nulls affordable: a block-permutation null is ``B``
  resampled rows, not ``B`` Python calls.

Closed-form i.i.d. p-values (``*_pvalue`` helpers) live next to their statistic.
They are **only** valid for serially independent data -- see
:mod:`panelary.depend._null` for the nulls that are not.

Chatterjee's xi
---------------
``xi(x, y)`` measures how much ``y`` is a (possibly noisy, non-monotone)
function of ``x`` (Chatterjee, 2021, JASA). It is **asymmetric by construction**
-- ``xi(x, y) != xi(y, x)`` and both are meaningful -- and is never silently
symmetrised here. Ties in ``y`` use Chatterjee's ties-corrected estimator. Ties
in ``x`` are broken by **expectation** (``tie_break="average"``): the value
returned is the exact mean of xi over every uniformly random ordering of the
tied ``x`` values, computed in closed form, so the answer is deterministic and
still unbiased for the randomised estimator SciPy and XICOR report.
``tie_break="random"`` reproduces the randomised estimator and requires a seed.

Tail dependence
---------------
``lambda_L = P(F_y(Y) <= q | F_x(X) <= q)`` estimated as
``#{both in the lower q-tail} / k`` with ``k = floor(q n)``, ranks taken within
the sample. Under independence the estimate sits near ``q`` (not 0): the
co-exceedance count is hypergeometric, which is the exact i.i.d. null used by
:func:`tail_pvalue`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from panelary.depend._ranks import (
    _dominance_sums,
    _group_bounds,
    pairwise_complete,
    ranks,
)
from panelary.depend._special import norm_sf
from panelary.econ._common import t_sf

__all__ = [
    "exceedance_corr",
    "hoeffding_d",
    "hoeffding_pvalue",
    "kendall",
    "kendall_pvalue",
    "pearson",
    "pearson_pvalue",
    "spearman",
    "tail_dependence",
    "tail_dependence_matrix",
    "tail_pvalue",
    "xi",
    "xi_matrix",
    "xi_null_sd",
    "xi_pvalue",
]

#: Tie fraction above which the closed-form xi null is refused (returns NaN).
XI_TIE_LIMIT = 0.05


def _as_rows(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr[None, :] if arr.ndim == 1 else arr


def _clean_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    xa, ya, n = pairwise_complete(np.ravel(x), np.ravel(y))
    return xa, ya, n


# --------------------------------------------------------------------------- #
# Pearson / Spearman
# --------------------------------------------------------------------------- #
def _pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two ``(B, m)`` arrays."""
    x = _as_rows(x)
    y = _as_rows(y)
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    num = np.einsum("bm,bm->b", xc, yc)
    den = np.sqrt(np.einsum("bm,bm->b", xc, xc) * np.einsum("bm,bm->b", yc, yc))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.clip(r, -1.0, 1.0)


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation (Pearson on average ranks)."""
    return _pearson_rows(ranks(_as_rows(x)), ranks(_as_rows(y)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson product-moment correlation on the pairwise-complete rows.

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length. Non-finite rows are dropped.

    Returns
    -------
    float
        The correlation in ``[-1, 1]``; ``nan`` below 3 complete rows or for a
        constant input.
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 3:
        return float("nan")
    return float(_pearson_rows(xa, ya)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties).

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length. Non-finite rows are dropped.

    Returns
    -------
    float
        Rank correlation in ``[-1, 1]``; ``nan`` below 3 complete rows.
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 3:
        return float("nan")
    return float(_spearman_rows(xa, ya)[0])


def pearson_pvalue(r: float, n: int) -> float:
    """Two-sided i.i.d. p-value of a (Pearson or Spearman) correlation.

    Student-t with ``n - 2`` degrees of freedom on
    ``t = r sqrt((n - 2) / (1 - r^2))`` -- exact for bivariate-normal Pearson,
    the standard large-sample approximation for Spearman.
    """
    if not np.isfinite(r) or n < 3:
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    t = abs(r) * math.sqrt((n - 2) / (1.0 - r * r))
    return float(min(1.0, 2.0 * float(t_sf(t, n - 2))))


# --------------------------------------------------------------------------- #
# Kendall's tau-b
# --------------------------------------------------------------------------- #
def _tie_pairs(v: np.ndarray) -> np.ndarray:
    """Per-row ``sum_g t_g (t_g - 1) / 2`` over the tie groups of ``v`` (B, m)."""
    srt = np.sort(v, axis=-1)
    _, start, end = _group_bounds(srt)
    size = (end - start + 1).astype(np.float64)
    # Each element of a group of size t contributes (t - 1) / 2.
    return ((size - 1.0) / 2.0).sum(axis=-1)


def _kendall_counts(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise (concordant - discordant, tau-b denominator) for ``(B, m)``."""
    x = _as_rows(x)
    y = _as_rows(y)
    bsz, m = x.shape
    order = np.lexsort((-y, x), axis=-1)
    ry = ranks(y, method="min") - 1
    ry_ord = np.take_along_axis(ry, order, axis=-1)
    conc = _dominance_sums(ry_ord, np.ones((bsz, m, 1)))[:, :, 0].sum(axis=-1)
    x_less = (ranks(x, method="min") - 1).astype(np.float64).sum(axis=-1)
    # #{j : x_j < x_i and y_j == y_i}: sort by (y, x); within a y-group the
    # start of the (y, x)-subgroup minus the start of the y-group.
    o2 = np.lexsort((x, y), axis=-1)
    ys = np.take_along_axis(y, o2, axis=-1)
    xs = np.take_along_axis(x, o2, axis=-1)
    pos = np.broadcast_to(np.arange(m), (bsz, m))
    new_y = np.ones((bsz, m), dtype=bool)
    new_xy = np.ones((bsz, m), dtype=bool)
    if m > 1:
        new_y[:, 1:] = ys[:, 1:] != ys[:, :-1]
        new_xy[:, 1:] = new_y[:, 1:] | (xs[:, 1:] != xs[:, :-1])
    gs = np.maximum.accumulate(np.where(new_y, pos, 0), axis=-1)
    ss = np.maximum.accumulate(np.where(new_xy, pos, 0), axis=-1)
    x_less_y_eq = (ss - gs).astype(np.float64).sum(axis=-1)
    disc = x_less - conc - x_less_y_eq
    n0 = m * (m - 1) / 2.0
    n1 = _tie_pairs(x)
    n2 = _tie_pairs(y)
    den = np.sqrt((n0 - n1) * (n0 - n2))
    return conc - disc, den


def _kendall_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Kendall tau-b via blocked dominance counts, ``O(B m^1.5)``."""
    s, den = _kendall_counts(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, s / np.where(den > 0, den, 1.0), np.nan)


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b (tie-adjusted), ``O(n^1.5)`` via dominance counts.

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length. Non-finite rows are dropped.

    Returns
    -------
    float
        tau-b in ``[-1, 1]``; ``nan`` below 3 complete rows.
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 3:
        return float("nan")
    return float(_kendall_rows(xa, ya)[0])


def _tie_sizes(v: np.ndarray) -> np.ndarray:
    _, counts = np.unique(v, return_counts=True)
    return counts[counts > 1].astype(np.float64)


def kendall_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Kendall tau-b and its two-sided i.i.d. p-value (tie-corrected variance).

    Uses the classical normal approximation to ``S = C - D`` with the variance
    corrected for ties in both margins (Kendall, 1970, eq. 4.4).

    Returns
    -------
    tau : float
    p_value : float
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 3:
        return float("nan"), float("nan")
    s, den = _kendall_counts(xa, ya)
    s0, d0 = float(s[0]), float(den[0])
    tau = s0 / d0 if d0 > 0 else float("nan")
    t = _tie_sizes(xa)
    u = _tie_sizes(ya)
    v0 = n * (n - 1.0) * (2.0 * n + 5.0)
    vt = float(np.sum(t * (t - 1.0) * (2.0 * t + 5.0)))
    vu = float(np.sum(u * (u - 1.0) * (2.0 * u + 5.0)))
    v1 = (
        float(np.sum(t * (t - 1.0)))
        * float(np.sum(u * (u - 1.0)))
        / (2.0 * n * (n - 1.0))
    )
    v2 = (
        float(np.sum(t * (t - 1.0) * (t - 2.0)))
        * float(np.sum(u * (u - 1.0) * (u - 2.0)))
        / (9.0 * n * (n - 1.0) * (n - 2.0))
    )
    var = (v0 - vt - vu) / 18.0 + v1 + v2
    if var <= 0 or not np.isfinite(tau):
        return tau, float("nan")
    z = abs(s0) / math.sqrt(var)
    return tau, float(min(1.0, 2.0 * float(norm_sf(z))))


# --------------------------------------------------------------------------- #
# Chatterjee's xi
# --------------------------------------------------------------------------- #
def _pairwise_abs_sum(values: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """``S_g = sum_{i<j in g} |v_i - v_j|`` for every group ``g`` in ``O(n log n)``."""
    if values.size == 0:
        return np.zeros(n_groups)
    order = np.lexsort((values, gid))
    vs = values[order]
    gs = gid[order]
    counts = np.bincount(gid, minlength=n_groups)
    starts = np.cumsum(counts) - counts
    k = np.arange(vs.size) - starts[gs] + 1
    msz = counts[gs]
    return np.bincount(gs, weights=vs * (2.0 * k - msz - 1.0), minlength=n_groups)


def _expected_abs_diff_sum(r: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Exact ``E[sum |r_{i+1} - r_i|]`` over random orderings of tied-``x`` runs.

    ``r`` is ``(B, m)`` -- the y-statistic in x-sorted order -- and ``new`` flags
    the first element of every run of tied ``x``. By linearity: a run of size
    ``m_g`` contributes ``(2 / m_g) sum_{i<j} |r_i - r_j|`` internally, and the
    boundary between runs ``g, g+1`` contributes the mean ``|a - b|`` over
    ``a in g, b in g+1``, obtained as ``S(g u g+1) - S(g) - S(g+1)``.
    """
    bsz, m = r.shape
    gid_row = np.cumsum(new, axis=-1) - 1
    n_per_row = gid_row[:, -1] + 1
    offs = np.concatenate([[0], np.cumsum(n_per_row)[:-1]])
    gid = (gid_row + offs[:, None]).ravel()
    n_groups = int(n_per_row.sum())
    vals = r.ravel().astype(np.float64)
    sizes = np.bincount(gid, minlength=n_groups).astype(np.float64)
    s_within = _pairwise_abs_sum(vals, gid, n_groups)
    internal = 2.0 * s_within / sizes
    last_of_row = np.zeros(n_groups, dtype=bool)
    last_of_row[offs + n_per_row - 1] = True
    first_mask = ~last_of_row[gid]
    second_mask = gid_row.ravel() > 0
    pv = np.concatenate([vals[first_mask], vals[second_mask]])
    pid = np.concatenate([gid[first_mask], gid[second_mask] - 1])
    s_union = _pairwise_abs_sum(pv, pid, n_groups)
    boundary = np.zeros(n_groups)
    g = np.flatnonzero(~last_of_row)
    boundary[g] = (s_union[g] - s_within[g] - s_within[g + 1]) / (
        sizes[g] * sizes[g + 1]
    )
    row_of_group = np.repeat(np.arange(bsz), n_per_row)
    return np.bincount(row_of_group, weights=internal + boundary, minlength=bsz)


def _xi_rows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tie_break: str = "average",
    seed: int | None = None,
) -> np.ndarray:
    """Row-wise Chatterjee xi(x -> y) on ``(B, m)`` complete arrays."""
    x = _as_rows(x)
    y = _as_rows(y)
    bsz, m = x.shape
    if m < 2:
        return np.full(bsz, np.nan)
    if tie_break == "random":
        if seed is None:
            raise ValueError("tie_break='random' requires an explicit integer `seed`.")
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(x.shape), x), axis=-1)
    elif tie_break == "average":
        order = np.argsort(x, axis=-1, kind="stable")
    else:
        raise ValueError(
            f"unknown `tie_break` {tie_break!r}; expected 'average' or 'random'."
        )
    r = ranks(y, method="max").astype(np.float64)
    l_ = (m + 1 - ranks(y, method="min")).astype(np.float64)
    den = 2.0 * np.sum(l_ * (m - l_), axis=-1)
    rs = np.take_along_axis(r, order, axis=-1)
    num = None
    if tie_break == "average":
        xs = np.take_along_axis(x, order, axis=-1)
        new = np.ones(x.shape, dtype=bool)
        new[:, 1:] = xs[:, 1:] != xs[:, :-1]
        if not new.all():
            num = _expected_abs_diff_sum(rs, new)
    if num is None:
        num = np.abs(np.diff(rs, axis=-1)).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1.0 - m * num / np.where(den > 0, den, 1.0), np.nan)


def xi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tie_break: str = "average",
    ties: str = "auto",
    seed: int | None = None,
) -> float:
    """Chatterjee's rank correlation ``xi(x -> y)``: is ``y`` a function of ``x``?

    Sort the pairs by ``x``; with ``r_i`` the rank of ``y`` in that order,

    * no ties in ``y``: ``xi = 1 - 3 sum |r_{i+1} - r_i| / (n^2 - 1)``,
    * ties in ``y``: ``xi = 1 - n sum |r_{i+1} - r_i| / (2 sum l_i (n - l_i))``
      with ``r_i = #{j : y_j <= y_(i)}`` and ``l_i = #{j : y_j >= y_(i)}``.

    ``xi -> 0`` under independence and ``xi -> 1`` iff ``y`` is a measurable
    function of ``x``. **Directional**: ``xi(x, y) != xi(y, x)``.

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length. Non-finite rows are dropped.
    tie_break : {"average", "random"}, default="average"
        How ties in ``x`` are ordered. ``"average"`` returns the exact
        expectation over random orderings (deterministic, no seed);
        ``"random"`` samples one ordering and needs ``seed``.
    ties : {"auto", "none"}, default="auto"
        ``"auto"`` uses the ties-corrected formula (identical to the continuous
        formula when ``y`` has no ties); ``"none"`` forces the continuous one.
    seed : int, optional
        Required for ``tie_break="random"``.

    Returns
    -------
    float
        The coefficient; ``nan`` below 2 complete rows or for constant ``y``.

    References
    ----------
    Chatterjee, S. (2021). A new coefficient of correlation. *JASA* 116(536).
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 2:
        return float("nan")
    if ties == "none":
        order = np.argsort(xa, kind="stable")
        r = ranks(ya, method="ordinal")[order].astype(np.float64)
        return float(1.0 - 3.0 * np.abs(np.diff(r)).sum() / (n * n - 1.0))
    if ties != "auto":
        raise ValueError(f"unknown `ties` {ties!r}; expected 'auto' or 'none'.")
    return float(_xi_rows(xa, ya, tie_break=tie_break, seed=seed)[0])


def _tie_fraction(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return 1.0 - np.unique(v).size / v.size


def xi_null_sd(n: int, *, ties: np.ndarray | None = None) -> float:
    """Standard deviation of ``xi`` under independence: ``sqrt(2 / (5 n))``.

    Verified in the build contract: ``sqrt(n) sd`` is 0.4151 at n=200, 0.3972
    at n=1000, 0.3943 at n=5000 against the theoretical 0.4, so the closed form
    is usable from n ~ 200 for serially independent data.

    Parameters
    ----------
    n : int
        Number of complete pairs.
    ties : array_like, optional
        The data whose ties to check (``y``, or ``x`` and ``y`` stacked). If
        more than 5% of the values are tied, returns ``nan`` so the caller must
        fall back to a permutation null instead of a p-value known to be wrong.

    Returns
    -------
    float
    """
    if n < 2:
        return float("nan")
    if ties is not None and _tie_fraction(ties) > XI_TIE_LIMIT:
        return float("nan")
    return math.sqrt(0.4 / n)


def xi_pvalue(estimate: float, n: int, *, ties: np.ndarray | None = None) -> float:
    """One-sided i.i.d. p-value ``P(Z >= xi / sd)`` with ``sd = xi_null_sd(n)``.

    ``nan`` when :func:`xi_null_sd` refuses (heavy ties) -- use a permutation
    null then.
    """
    sd = xi_null_sd(n, ties=ties)
    if not (np.isfinite(sd) and np.isfinite(estimate)):
        return float("nan")
    return float(norm_sf(estimate / sd))


def xi_matrix(X: np.ndarray, *, tie_break: str = "average") -> np.ndarray:
    """Directed ``(p, p)`` xi matrix, ``M[i, j] = xi(X[:, i] -> X[:, j])``.

    ``O(p)`` argsorts, not ``O(p^2)``: for each column ``i`` all other columns'
    ranks are reordered by ``i``'s argsort in one gather and differenced in one
    vectorised pass. Rows with any non-finite entry are dropped (listwise).

    Parameters
    ----------
    X : array_like
        ``(n, p)`` matrix.
    tie_break : {"average"}, default="average"
        Only the deterministic expectation is supported for matrices.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` float64; the diagonal is ``xi(x -> x)``, which is
        ``1 - 3 / (n + 1)`` for a tie-free column (xi reaches 1 only as
        ``n -> inf``).
    """
    if tie_break != "average":
        raise ValueError("xi_matrix supports tie_break='average' only.")
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`X` must be 2-D (n, p), got shape {arr.shape}.")
    arr = arr[np.isfinite(arr).all(axis=1)]
    n, p = arr.shape
    out = np.full((p, p), np.nan)
    if n < 2:
        return out
    r = ranks(arr, axis=0, method="max").astype(np.float64)  # (n, p)
    l_ = (n + 1 - ranks(arr, axis=0, method="min")).astype(np.float64)
    den = 2.0 * np.sum(l_ * (n - l_), axis=0)  # (p,)
    for i in range(p):
        order = np.argsort(arr[:, i], kind="stable")
        rs = r[order]  # (n, p): every column reordered by column i
        xs = arr[order, i]
        new = np.ones(n, dtype=bool)
        new[1:] = xs[1:] != xs[:-1]
        if new.all():
            num = np.abs(np.diff(rs, axis=0)).sum(axis=0)
        else:
            num = _expected_abs_diff_sum(rs.T, np.broadcast_to(new, (p, n)))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(
                den > 0, 1.0 - n * num / np.where(den > 0, den, 1.0), np.nan
            )
    return out


# --------------------------------------------------------------------------- #
# Hoeffding's D
# --------------------------------------------------------------------------- #
def _hoeffding_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Hoeffding's D (Hollander-Wolfe scaling, range ``[-0.5, 1]``)."""
    x = _as_rows(x)
    y = _as_rows(y)
    bsz, m = x.shape
    if m < 5:
        return np.full(bsz, np.nan)
    rx = ranks(x)
    ry = ranks(y)
    # A_i = #{x_j < x_i, y_j < y_i}
    order = np.lexsort((-y, x), axis=-1)
    rymin = ranks(y, method="min") - 1
    a_ord = _dominance_sums(
        np.take_along_axis(rymin, order, axis=-1), np.ones((bsz, m, 1))
    )
    a = np.empty((bsz, m))
    np.put_along_axis(a, order, a_ord[:, :, 0], axis=-1)
    pos = np.broadcast_to(np.arange(m), (bsz, m))

    def _within(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # For every element: #{j : first_j == first_i, second_j < second_i} and
        # the size of its exact (first, second) duplicate group, input order.
        o = np.lexsort((second, first), axis=-1)
        fs = np.take_along_axis(first, o, axis=-1)
        ss = np.take_along_axis(second, o, axis=-1)
        new_f = np.ones((bsz, m), dtype=bool)
        new_fs = np.ones((bsz, m), dtype=bool)
        new_f[:, 1:] = fs[:, 1:] != fs[:, :-1]
        new_fs[:, 1:] = new_f[:, 1:] | (ss[:, 1:] != ss[:, :-1])
        g0 = np.maximum.accumulate(np.where(new_f, pos, 0), axis=-1)
        s0 = np.maximum.accumulate(np.where(new_fs, pos, 0), axis=-1)
        last = np.ones((bsz, m), dtype=bool)
        last[:, :-1] = new_fs[:, 1:]
        s1 = np.minimum.accumulate(np.where(last, pos, m - 1)[:, ::-1], axis=-1)[
            :, ::-1
        ]
        cnt = np.empty((bsz, m))
        dup = np.empty((bsz, m))
        np.put_along_axis(cnt, o, (s0 - g0).astype(np.float64), axis=-1)
        np.put_along_axis(dup, o, (s1 - s0 + 1).astype(np.float64), axis=-1)
        return cnt, dup

    b, dup = _within(x, y)  # #{x_j == x_i, y_j < y_i}
    c, _ = _within(y, x)  # #{y_j == y_i, x_j < x_i}
    q = 1.0 + a + 0.5 * b + 0.5 * c + 0.25 * (dup - 1.0)
    d1 = np.sum((q - 1.0) * (q - 2.0), axis=-1)
    d2 = np.sum((rx - 1.0) * (rx - 2.0) * (ry - 1.0) * (ry - 2.0), axis=-1)
    d3 = np.sum((rx - 2.0) * (ry - 2.0) * (q - 1.0), axis=-1)
    n = float(m)
    return (
        30.0
        * ((n - 2.0) * (n - 3.0) * d1 + d2 - 2.0 * (n - 2.0) * d3)
        / (n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (n - 4.0))
    )


def hoeffding_d(x: np.ndarray, y: np.ndarray) -> float:
    """Hoeffding's D, the general rank test of independence.

    Consistent against *every* dependent alternative with a continuous joint
    density -- including the non-functional ones (circles, X shapes) that
    Chatterjee's xi is weak on. Scaled as in Hollander & Wolfe so that
    ``D in [-0.5, 1]``, ``D ~ 0`` under independence and ``D = 1`` under
    perfect monotone dependence. Built on blocked dominance counts,
    ``O(n^1.5)``; ties use the usual 1/2 and 1/4 weights.

    Parameters
    ----------
    x, y : array_like
        1-D arrays of equal length. Non-finite rows are dropped.

    Returns
    -------
    float
        ``nan`` below 5 complete rows.

    References
    ----------
    Hoeffding, W. (1948). A non-parametric test of independence. *Ann. Math.
    Statist.* 19(4). Hollander, M. & Wolfe, D. A. (1973). *Nonparametric
    Statistical Methods*.
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 5:
        return float("nan")
    return float(_hoeffding_rows(xa, ya)[0])


@lru_cache(maxsize=1)
def _hoeffding_imhof_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Precomputed Imhof integrand pieces for ``W = sum lambda_jk Z_jk^2``.

    ``lambda_jk = 1 / (pi^4 j^2 k^2)``, truncated at ``j, k <= 40``; the mean of
    the truncated remainder (tiny variance) is returned as a location shift.
    """
    kmax = 40
    j = np.arange(1, kmax + 1, dtype=np.float64)
    prod = np.outer(j, j).ravel()
    vals, mult = np.unique(prod, return_counts=True)
    lam = 1.0 / (math.pi**4 * vals**2)
    shift = 1.0 / 36.0 - float(np.sum(mult * lam))
    # u grid: dense near 0, geometric tail; trapezoid weights.
    u = np.concatenate(
        [np.linspace(1e-6, 50.0, 4001)[:-1], np.geomspace(50.0, 2e5, 6000)]
    )
    theta0 = np.zeros_like(u)
    logrho = np.zeros_like(u)
    # Every operand here is finite, but some BLAS backends (Apple Accelerate
    # under numpy 2.2) raise spurious floating-point flags from `@`. Silence the
    # flags for the matmul only, then check the result instead.
    with np.errstate(all="ignore"):
        for chunk in np.array_split(np.arange(vals.size), 8):
            lu = np.outer(u, lam[chunk])
            theta0 += 0.5 * (np.arctan(lu) @ mult[chunk])
            logrho += 0.25 * (np.log1p(lu * lu) @ mult[chunk])
    if not (np.isfinite(theta0).all() and np.isfinite(logrho).all()):
        raise FloatingPointError("non-finite Imhof grid for the Hoeffding law")
    weight = np.exp(-logrho) / u
    du = np.diff(u)
    trap = np.zeros_like(u)
    trap[:-1] += 0.5 * du
    trap[1:] += 0.5 * du
    return u, theta0, weight * trap, shift


@lru_cache(maxsize=1)
def _hoeffding_eigen() -> tuple[np.ndarray, np.ndarray, float]:
    kmax = 40
    j = np.arange(1, kmax + 1, dtype=np.float64)
    vals, mult = np.unique(np.outer(j, j).ravel(), return_counts=True)
    lam = 1.0 / (math.pi**4 * vals**2)
    shift = 1.0 / 36.0 - float(np.sum(mult * lam))
    return lam, mult.astype(np.float64), shift


def _hoeffding_chernoff(w: float) -> float:
    """Chernoff upper bound ``min_s exp(-s w) E[exp(s W)]`` on ``P(W >= w)``.

    ``log E[exp(s W)] = -1/2 sum m_k log(1 - 2 s lambda_k) + s * shift`` for
    ``s < 1 / (2 lambda_max)``. A rigorous (conservative) bound, used where the
    Imhof integral's absolute error (~1e-6) swamps the true tail.
    """
    lam, mult, shift = _hoeffding_eigen()
    s = np.linspace(0.0, 1.0, 401)[1:-1] / (2.0 * lam.max())
    with np.errstate(all="ignore"):  # spurious BLAS flags; see _hoeffding_imhof_grid
        logm = -0.5 * (np.log1p(-2.0 * np.outer(s, lam)) @ mult) + s * shift
    if not np.isfinite(logm).all():
        raise FloatingPointError("non-finite Chernoff exponent for the Hoeffding law")
    return float(min(1.0, np.exp(np.min(logm - s * w))))


def _hoeffding_sf(w: float) -> float:
    """``P(W >= w)`` for the Hoeffding/BKR limit law.

    Imhof inversion of the characteristic function (absolute error ~1e-6); in
    the far tail the smaller of that and a Chernoff upper bound, so a p-value
    never floors at the integration noise.
    """
    u, theta0, wt, shift = _hoeffding_imhof_grid()
    ww = w - shift
    if ww <= 0:
        return 1.0
    val = 0.5 + float(np.sum(np.sin(theta0 - 0.5 * ww * u) * wt)) / math.pi
    val = float(min(1.0, max(0.0, val)))
    if val < 1e-3:
        val = min(val, _hoeffding_chernoff(w))
    return val


def hoeffding_pvalue(d: float, n: int) -> float:
    """One-sided i.i.d. p-value of Hoeffding's D from its asymptotic law.

    Under independence ``n D / 30 + 1/36`` converges to
    ``W = sum_{j,k>=1} Z_jk^2 / (pi^4 j^2 k^2)`` (Hoeffding, 1948; Blum, Kiefer
    & Rosenblatt, 1961). The survival function is **tabulated by Imhof
    inversion** of ``W``'s characteristic function once per process -- no
    Monte Carlo, no SciPy.
    """
    if not np.isfinite(d) or n < 5:
        return float("nan")
    return _hoeffding_sf(n * d / 30.0 + 1.0 / 36.0)


# --------------------------------------------------------------------------- #
# Tail dependence
# --------------------------------------------------------------------------- #
def _tail_k(m: int, q: float) -> int:
    if not 0.0 < q < 0.5:
        raise ValueError(f"`q` must be in (0, 0.5), got {q}.")
    return int(math.floor(q * m))


def _tail_rows(x: np.ndarray, y: np.ndarray, *, q: float, side: str) -> np.ndarray:
    """Row-wise co-exceedance tail-dependence estimate (``side`` lower/upper)."""
    x = _as_rows(x)
    y = _as_rows(y)
    bsz, m = x.shape
    k = _tail_k(m, q)
    if k < 1:
        return np.full(bsz, np.nan)
    rx = ranks(x)
    ry = ranks(y)
    if side == "lower":
        both = (rx <= k) & (ry <= k)
    elif side == "upper":
        both = (rx > m - k) & (ry > m - k)
    else:
        raise ValueError(f"unknown tail `side` {side!r}; expected 'lower' or 'upper'.")
    return both.sum(axis=-1) / float(k)


def tail_dependence(
    x: np.ndarray, y: np.ndarray, *, q: float = 0.05
) -> tuple[float, float]:
    """Nonparametric lower and upper tail dependence ``(lambda_L, lambda_U)``.

    ``lambda_L = #{rank_x <= k and rank_y <= k} / k`` with ``k = floor(q n)``,
    ranks taken *within this sample*; ``lambda_U`` mirrors it in the upper
    tail. Under independence the estimate sits near ``q``; a Gaussian copula
    has ``lambda = 0`` in the limit, and this estimator does not manufacture
    one (it tends to ``q``-ish values that shrink with ``q``).

    ``q`` is a **fitted threshold** when used inside a transformer or rolling
    feature: it must be a constant or come from the training window, never from
    the full series.

    Parameters
    ----------
    x, y : array_like
        1-D arrays. Non-finite rows are dropped.
    q : float, default=0.05
        Tail probability in ``(0, 0.5)``.

    Returns
    -------
    (float, float)
        ``(lower, upper)``; ``nan`` when ``floor(q n) < 1``.
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 2:
        return float("nan"), float("nan")
    lo = float(_tail_rows(xa, ya, q=q, side="lower")[0])
    hi = float(_tail_rows(xa, ya, q=q, side="upper")[0])
    return lo, hi


def exceedance_corr(
    x: np.ndarray, y: np.ndarray, *, q: float = 0.05
) -> tuple[float, float]:
    """Exceedance correlations (Longin & Solnik, 2001): Pearson correlation of
    the pairs where **both** variables are in their lower (upper) ``q``-tail.

    Returns ``(lower, upper)``; each is ``nan`` with fewer than 3 joint
    exceedances -- at ``q = 0.05`` that needs a fairly large or fairly
    dependent sample, so a larger ``q`` (0.1-0.25) is usual.
    """
    xa, ya, n = _clean_pair(x, y)
    if n < 3:
        return float("nan"), float("nan")
    k = _tail_k(n, q)
    rx = ranks(xa)
    ry = ranks(ya)
    out = []
    for mask in ((rx <= k) & (ry <= k), (rx > n - k) & (ry > n - k)):
        if mask.sum() < 3:
            out.append(float("nan"))
        else:
            out.append(float(_pearson_rows(xa[mask], ya[mask])[0]))
    return out[0], out[1]


def tail_dependence_matrix(
    X: np.ndarray, *, q: float = 0.05, side: str = "lower"
) -> np.ndarray:
    """``(p, p)`` tail-dependence matrix in **one matmul**.

    ``I = (ranks(X) <= k)``, ``M = I.T @ I / k`` (listwise-complete rows).

    Parameters
    ----------
    X : array_like
        ``(n, p)`` matrix.
    q : float, default=0.05
        Tail probability.
    side : {"lower", "upper"}, default="lower"

    Returns
    -------
    numpy.ndarray
        Symmetric ``(p, p)``; the diagonal is 1.
    """
    arr = np.asarray(X, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    n, p = arr.shape
    k = _tail_k(n, q)
    if k < 1:
        return np.full((p, p), np.nan)
    r = ranks(arr, axis=0)
    if side == "lower":
        ind = (r <= k).astype(np.float64)
    elif side == "upper":
        ind = (r > n - k).astype(np.float64)
    else:
        raise ValueError(f"unknown tail `side` {side!r}; expected 'lower' or 'upper'.")
    return ind.T @ ind / float(k)


def tail_pvalue(
    estimate: float, x: np.ndarray, y: np.ndarray, *, q: float, side: str
) -> float:
    """Exact i.i.d. p-value of a co-exceedance count (hypergeometric upper tail).

    With ``K_x`` of the ``n`` ranks of ``x`` and ``K_y`` of ``y`` in the tail,
    the co-exceedance count under independence is hypergeometric
    ``HG(n, K_x, K_y)``; the p-value is ``P(C >= observed)``.
    """
    xa, ya, n = _clean_pair(x, y)
    k = _tail_k(n, q) if n else 0
    if k < 1 or not np.isfinite(estimate):
        return float("nan")
    rx = ranks(xa)
    ry = ranks(ya)
    if side == "lower":
        kx, ky = int((rx <= k).sum()), int((ry <= k).sum())
    else:
        kx, ky = int((rx > n - k).sum()), int((ry > n - k).sum())
    c = int(round(estimate * k))
    return _hypergeom_sf(c, n, kx, ky)


def _hypergeom_sf(c: int, n: int, kx: int, ky: int) -> float:
    """``P(C >= c)`` for ``C ~ HG(population n, successes kx, draws ky)``."""
    lo = max(0, kx + ky - n)
    hi = min(kx, ky)
    if c <= lo:
        return 1.0
    if c > hi:
        return 0.0
    i = np.arange(c, hi + 1, dtype=np.float64)
    lg = math.lgamma

    def lchoose(a: float, b: np.ndarray) -> np.ndarray:
        return lg(a + 1.0) - _lgamma_vec(b + 1.0) - _lgamma_vec(a - b + 1.0)

    logp = (
        lchoose(kx, i)
        + lchoose(n - kx, ky - i)
        - (lg(n + 1.0) - lg(ky + 1.0) - lg(n - ky + 1.0))
    )
    return float(min(1.0, np.exp(logp).sum()))


_LGAMMA = np.vectorize(math.lgamma, otypes=[np.float64])


def _lgamma_vec(v: np.ndarray) -> np.ndarray:
    return _LGAMMA(v)
