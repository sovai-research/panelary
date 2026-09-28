"""Null distributions that stay valid under serial and cross-sectional
dependence -- the differentiator of :mod:`panelary.depend`.

Why this module exists
----------------------
Two *independent* AR(1) series with ``phi = 0.95`` (n = 500), tested against
each other with Chatterjee's xi under the standard i.i.d. asymptotic null,
reject at **56%** at a nominal 5% (build-contract measurement, reproduced by
``tests/test_depend_calibration.py``). Persistent series are what every price
level, valuation ratio and macro series in a panel looks like. An i.i.d.
p-value on them is noise reported as discovery.

The resampling schemes here all break the ``x``-``y`` alignment while keeping
the serial structure of the resampled series:

``"block"``
    **Circular block permutation** (:func:`block_permutation_indices`): rotate
    by a random offset, cut into blocks of ``block_length``, shuffle the
    blocks. A permutation, so the marginal is preserved exactly and rank
    statistics see no artificial ties. Block length from
    :func:`auto_block_length` (Politis & White, 2004; Patton, Politis & White,
    2009) unless given.
``"stationary"``
    Politis-Romano stationary bootstrap indices, reused verbatim from
    :func:`panelary.validation._bootstrap.block_bootstrap_indices`
    (resampling *with* replacement).
``"shift"``
    :func:`circular_shift` -- cyclic rotations. Preserves the rotated series'
    autocorrelation exactly (up to one wrap point). Only ``n`` distinct
    rotations exist, so it enumerates them when asked for more.
``"iaaft"``
    :func:`iaaft` surrogates (Schreiber & Schmitz, 1996): same amplitude
    spectrum **and** same empirical marginal -- the right null for a rank
    statistic on a fat-tailed series. Returned as index permutations.
``"phase"``
    :func:`phase_randomise` Fourier surrogates (Gaussianised marginal).
``"common-time"``
    :func:`common_time_indices` -- **the panel null**. Permutes whole date
    columns in blocks, identically for every entity, so each date's full
    cross-section (hence every common macro shock) survives. Per-entity
    independent shuffles are invalid whenever entities share a factor.
``"entity"``
    :func:`entity_permutation_indices` -- re-pairs one entity's ``y`` with
    another entity's ``x``. A *different* hypothesis (cross-sectional matching),
    documented side by side so the choice is deliberate.
``"permutation"``
    Plain i.i.d. shuffles. Valid only for serially independent data.

The null policy
---------------
:data:`NULL_POLICY` encodes the build contract's table and ``null="auto"``
implements exactly it: a lag-1 rank-autocorrelation pre-check at
``|rho| > 0.2`` picks the serial branch, a multi-entity panel picks
``common-time``, and the choice is always reported in ``null_method``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from panelary.depend._ranks import lag1_rank_autocorr
from panelary.econ._common import chi2_sf
from panelary.preprocessing._base import LeakageWarning
from panelary.validation._bootstrap import block_bootstrap_indices, resolve_segments

__all__ = [
    "NULL_METHODS",
    "NULL_POLICY",
    "SERIAL_THRESHOLD",
    "SerialDependenceWarning",
    "SurrogateResult",
    "auto_block_length",
    "block_permutation_indices",
    "pair_block_length",
    "circular_shift",
    "common_time_indices",
    "entity_permutation_indices",
    "gamma_pvalue",
    "iaaft",
    "phase_randomise",
    "pvalue",
    "resolve_null",
    "serial_dependence",
]

#: ``|lag-1 rank autocorrelation|`` above which a series counts as serially
#: dependent for ``null="auto"`` and the ``null="iid"`` warning.
SERIAL_THRESHOLD = 0.2

#: Every null a caller may request by name.
NULL_METHODS = frozenset(
    {
        "auto",
        "iid",
        "asymptotic",
        "permutation",
        "block",
        "stationary",
        "shift",
        "iaaft",
        "phase",
        "common-time",
        "entity",
        "hac",
    }
)

#: The contract's null policy table. ``iid`` is the serially-independent
#: branch, ``serial`` the serially-dependent branch and ``panel`` the
#: common-shock panel branch. ``null="auto"`` implements exactly this.
NULL_POLICY: dict[str, dict[str, str]] = {
    "xi": {"iid": "asymptotic", "serial": "block", "panel": "common-time"},
    "spearman": {"iid": "asymptotic", "serial": "block", "panel": "common-time"},
    "pearson": {"iid": "asymptotic", "serial": "block", "panel": "common-time"},
    "kendall": {"iid": "asymptotic", "serial": "block", "panel": "common-time"},
    "hoeffding": {"iid": "asymptotic", "serial": "block", "panel": "common-time"},
    "dcor": {"iid": "gamma", "serial": "block", "panel": "common-time"},
    "dcor_t": {"iid": "t", "serial": "block", "panel": "common-time"},
    "gcmi": {"iid": "asymptotic", "serial": "block", "panel": "common-time"},
    "tail_lower": {"iid": "exact", "serial": "block", "panel": "common-time"},
    "tail_upper": {"iid": "exact", "serial": "block", "panel": "common-time"},
    "hsic": {"iid": "gamma", "serial": "shift", "panel": "common-time"},
    "mi_ksg": {"iid": "permutation", "serial": "block", "panel": "common-time"},
    "transfer_entropy": {
        "iid": "asymptotic",
        "serial": "block",
        "panel": "common-time",
    },
    "gcm": {"iid": "asymptotic", "serial": "hac", "panel": "common-time"},
}


class SerialDependenceWarning(LeakageWarning):
    """An i.i.d. null was requested on data that fails the serial pre-check.

    A :class:`~panelary.preprocessing._base.LeakageWarning` because the failure
    mode is the same class of error as unpurged cross-validation: the reported
    p-value is far too small (56% type-I error at nominal 5% for xi on two
    independent AR(1), phi = 0.95 series).
    """


@dataclass(frozen=True)
class SurrogateResult:
    """Surrogate series from :func:`iaaft` / :func:`phase_randomise`.

    Attributes
    ----------
    values : numpy.ndarray
        ``(B, n)`` surrogate series.
    indices : numpy.ndarray or None
        ``(B, n)`` permutation with ``values[b] == x[indices[b]]`` when the
        surrogate reuses the exact marginal (IAAFT); ``None`` otherwise.
    n_iter : int
        Iterations run (IAAFT).
    converged : numpy.ndarray
        Per-surrogate flag: rank order stopped changing within ``max_iter``.
    """

    values: np.ndarray
    indices: np.ndarray | None
    n_iter: int
    converged: np.ndarray


# --------------------------------------------------------------------------- #
# p-values
# --------------------------------------------------------------------------- #
def pvalue(
    observed: float, null_draws: np.ndarray, *, alternative: str = "greater"
) -> float:
    """Resampling p-value ``(1 + #{null >= observed}) / (1 + B)``.

    The ``+1`` is not optional: without it a p-value of exactly 0 is possible,
    and every downstream FDR procedure breaks on it.

    Parameters
    ----------
    observed : float
        The statistic on the real data.
    null_draws : array_like
        ``B`` statistics under the null; non-finite draws are ignored.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        ``"two-sided"`` is ``min(1, 2 min(p_greater, p_less))``, which does not
        assume a symmetric null.

    Returns
    -------
    float
        ``nan`` if ``observed`` is not finite or no draw is.
    """
    draws = np.asarray(null_draws, dtype=np.float64).ravel()
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(observed) or draws.size == 0:
        return float("nan")
    b = draws.size
    p_hi = (1.0 + float(np.count_nonzero(draws >= observed))) / (1.0 + b)
    if alternative == "greater":
        return p_hi
    p_lo = (1.0 + float(np.count_nonzero(draws <= observed))) / (1.0 + b)
    if alternative == "less":
        return p_lo
    if alternative == "two-sided":
        return float(min(1.0, 2.0 * min(p_hi, p_lo)))
    raise ValueError(
        f"unknown `alternative` {alternative!r}; expected 'greater', 'less' or 'two-sided'."
    )


def gamma_pvalue(observed: float, mean: float, var: float) -> float:
    """Upper tail of a gamma moment-matched to ``mean`` and ``var``.

    ``shape = mean^2 / var``, ``scale = var / mean``; evaluated through
    ``econ._common.chi2_sf`` (``2 G / scale ~ chi^2(2 shape)``), so no SciPy.
    Roughly three orders of magnitude cheaper than ``B`` resamples.
    """
    if not (np.isfinite(observed) and mean > 0 and var > 0):
        return float("nan")
    if observed <= 0:
        return 1.0
    shape = mean * mean / var
    scale = var / mean
    return float(chi2_sf(2.0 * observed / scale, 2.0 * shape))


# --------------------------------------------------------------------------- #
# Serial-dependence pre-check and block length
# --------------------------------------------------------------------------- #
def serial_dependence(*series: np.ndarray) -> float:
    """The pre-check statistic: ``min_k |lag-1 rank autocorrelation of series k|``.

    Serial dependence invalidates an i.i.d. independence null only when **both**
    sides are persistent -- if either is serially independent, the pairs are
    exchangeable and the i.i.d. null is exact -- so the check takes the minimum.
    A single series (an ACF) is checked on its own.
    """
    vals = [abs(lag1_rank_autocorr(s)) for s in series]
    vals = [v for v in vals if np.isfinite(v)]
    return float(min(vals)) if vals else 0.0


def _flat_top(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def auto_block_length(x: np.ndarray, *, kind: str = "circular") -> int:
    """Politis & White (2004) optimal block length, with the Patton, Politis &
    White (2009) correction.

    ``b = (2 G^2 / D)^{1/3} n^{1/3}`` with ``G = sum lambda(k/M) |k| R(k)`` and
    ``D = (4/3) g^2`` (circular) or ``2 g^2`` (stationary),
    ``g = sum lambda(k/M) R(k)``, ``lambda`` the flat-top lag window and ``M``
    twice the first lag after which ``K_n = max(5, sqrt(log10 n))`` consecutive
    autocorrelations are insignificant at ``2 sqrt(log10 n / n)``. Capped at
    ``min(3 sqrt(n), n / 3)``; falls back to ``ceil(n^{1/3})`` when the
    flat-top spectral estimate degenerates.

    **This depends on n by construction**, so it is a *fitted* parameter:
    freeze it at fit time, never recompute it per fold end.

    Parameters
    ----------
    x : array_like
        1-D series (non-finite values dropped).
    kind : {"circular", "stationary"}, default="circular"

    Returns
    -------
    int
        Block length ``>= 1``.
    """
    if kind not in {"circular", "stationary"}:
        raise ValueError(
            f"unknown `kind` {kind!r}; expected 'circular' or 'stationary'."
        )
    xa = np.asarray(x, dtype=np.float64).ravel()
    xa = xa[np.isfinite(xa)]
    n = xa.size
    fallback = max(1, math.ceil(n ** (1.0 / 3.0))) if n else 1
    if n < 8:
        return fallback
    xc = xa - xa.mean()
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = min(n - 1, math.ceil(math.sqrt(n)) + k_n)
    b_max = max(1, math.ceil(min(3.0 * math.sqrt(n), n / 3.0)))
    acov = np.array([xc[: n - k] @ xc[k:] / n for k in range(m_max + 1)])
    if acov[0] <= 0:
        return fallback
    rho = acov[1:] / acov[0]
    thresh = 2.0 * math.sqrt(math.log10(n) / n)
    small = np.abs(rho) < thresh
    m_hat = m_max
    for m in range(0, m_max - k_n + 1):
        if small[m : m + k_n].all():
            m_hat = m
            break
    big_m = min(2 * max(m_hat, 1), m_max)
    k = np.arange(-big_m, big_m + 1)
    lam = _flat_top(k / big_m)
    r = acov[np.abs(k)]
    g_hat = float(np.sum(lam * np.abs(k) * r))
    g0 = float(np.sum(lam * r))
    d = (4.0 / 3.0 if kind == "circular" else 2.0) * g0 * g0
    if not (d > 0 and np.isfinite(g_hat)) or g_hat == 0.0:
        return fallback
    b = (2.0 * g_hat * g_hat / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return int(min(b_max, max(1, math.ceil(b))))


def _rank_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelations ``rho(1..max_lag)`` of the (average) ranks of ``x``."""
    from panelary.depend._ranks import ranks

    r = ranks(x)
    r = r - r.mean()
    den = float(r @ r)
    n = r.size
    if den <= 0:
        return np.zeros(max_lag)
    return np.array([float(r[: n - k] @ r[k:]) / den for k in range(1, max_lag + 1)])


def _pw_truncation(rho: np.ndarray, n: int) -> int:
    """Politis-White ``m_hat``: first lag after which ``K_n`` consecutive
    autocorrelations are insignificant."""
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    thresh = 2.0 * math.sqrt(math.log10(n) / n)
    small = np.abs(rho) < thresh
    for m in range(0, rho.size - k_n + 1):
        if small[m : m + k_n].all():
            return m
    return rho.size


def pair_block_length(x: np.ndarray, y: np.ndarray, *, tol: float = 0.05) -> int:
    """Block length for a **permutation test of dependence** between two series.

    A block permutation of ``y`` keeps lag-``k`` pairs only inside blocks, so it
    loses a fraction ``~k / L`` of each cross term of the long-run variance of
    a dependence statistic, ``sum_k rho_x(k) rho_y(k)``. The null variance is
    then too small by ``G / (L D)`` with ``G = sum_k |k| rho_x(k) rho_y(k)``
    and ``D = sum_k rho_x(k) rho_y(k)`` (rank autocorrelations, flat-top
    weighted, ``k`` over ``[-M, M]``), and the test over-rejects. This returns
    the smallest ``L`` with that relative bias ``<= tol``, never shorter than
    :func:`auto_block_length` on either series and never longer than ``n / 4``
    (at least four blocks).

    Measured on two independent AR(1) series with ``phi = 0.95`` at ``n = 500``
    (xi, 1000 replications, nominal 5%): the Politis-White length (38) rejects
    at 8.2%; ``L = 60, 100, 125`` at 7.2%, 6.6%, 6.2%. Politis-White optimises
    the MSE of a *variance* estimate, not the size of a permutation test.

    Like :func:`auto_block_length` this depends on ``n`` by construction: it is
    a fitted parameter, frozen at fit time.

    Parameters
    ----------
    x, y : array_like
        Complete 1-D series of equal length (time order).
    tol : float, default=0.05
        Tolerated relative shortfall of the null variance. Sample
        autocorrelations of persistent series are biased towards zero, so the
        estimated ``G / D`` is itself an underestimate; ``0.05`` is what brings
        the measured size back inside ``[3%, 8%]`` (see the calibration tests).

    Returns
    -------
    int
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    base = max(auto_block_length(xa), auto_block_length(ya))
    if n < 16:
        return base
    cap = max(1, n // 4)
    m_max = min(n - 1, cap)
    rx = _rank_acf(xa, m_max)
    ry = _rank_acf(ya, m_max)
    big_m = min(2 * max(_pw_truncation(rx, n), _pw_truncation(ry, n), 1), m_max)
    k = np.arange(1, big_m + 1, dtype=np.float64)
    lam = _flat_top(k / big_m)
    prod = rx[:big_m] * ry[:big_m] * lam
    d = 1.0 + 2.0 * float(prod.sum())
    g = 2.0 * float((k * prod).sum())
    if not (d > 0 and g > 0):
        return min(base, cap)
    need = math.ceil(g / (tol * d))
    return int(min(cap, max(base, need)))


# --------------------------------------------------------------------------- #
# Index generators
# --------------------------------------------------------------------------- #
def _rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        raise ValueError("an explicit integer `seed` is required (no unseeded RNG).")
    return np.random.default_rng(int(seed))


def circular_shift(
    n: int, *, n_resamples: int, seed: int, min_shift: int = 1
) -> np.ndarray:
    """``(B, n)`` index matrix of cyclic rotations ``idx[b, t] = (t + s_b) mod n``.

    Preserves the rotated series' autocorrelation exactly (up to one wrap
    point), at ``O(n)`` per draw. There are only ``n - 2 min_shift + 1``
    admissible shifts in ``[min_shift, n - min_shift]``; when ``n_resamples``
    reaches that count **all of them are enumerated** (an exact null, smallest
    achievable p-value ``1 / (count + 1)``) instead of sampling with repeats.

    Parameters
    ----------
    n : int
        Series length.
    n_resamples : int
        Requested number of rotations ``B``.
    seed : int
        Seed for the sampled shifts.
    min_shift : int, default=1
        Smallest admissible |shift|. Shifts shorter than the dependence length
        are near-copies of the observed alignment and cost power; the
        resampling engine passes the block length here.

    Returns
    -------
    numpy.ndarray
        int64 ``(B', n)`` with ``B' = min(n_resamples, admissible)``.
    """
    if n < 2:
        raise ValueError(f"need n >= 2 for a circular shift, got {n}.")
    lo = max(1, int(min_shift))
    hi = n - lo
    if hi < lo:
        lo, hi = 1, n - 1
    shifts_all = np.arange(lo, hi + 1, dtype=np.int64)
    if n_resamples >= shifts_all.size:
        shifts = shifts_all
    else:
        shifts = np.sort(
            _rng(seed).choice(shifts_all, size=int(n_resamples), replace=False)
        )
    return (np.arange(n, dtype=np.int64)[None, :] + shifts[:, None]) % n


def block_permutation_indices(
    n: int,
    *,
    block_length: int,
    n_resamples: int,
    seed: int | np.random.Generator,
    boundaries: Sequence[int] | None = None,
) -> np.ndarray:
    """``(B, n)`` circular block **permutations** (sampling without replacement).

    Within each segment (see ``boundaries``): rotate by a uniform random offset,
    cut into consecutive blocks of ``block_length`` (the last may be shorter)
    and shuffle the block order. Every draw is a permutation, so the marginal
    is preserved exactly and ranks stay tie-free; serial structure survives
    inside each block. Blocks never straddle a boundary (the same guardrail as
    :mod:`panelary.validation._bootstrap`, whose segment logic is reused).
    """
    if block_length < 1:
        raise ValueError(f"`block_length` must be >= 1, got {block_length}.")
    rng = _rng(seed)
    segments = resolve_segments(n, boundaries)
    n_b = int(n_resamples)
    out = np.empty((n_b, n), dtype=np.int64)
    for start, stop in segments:
        seg = stop - start
        L = min(int(block_length), seg)
        n_blocks = math.ceil(seg / L)
        base = np.arange(seg, dtype=np.int64)
        block_id = base // L
        offsets = rng.integers(0, seg, size=n_b)
        rotated = (base[None, :] + offsets[:, None]) % seg
        # A uniformly random block order per draw; sorting positions by their
        # block's key moves whole blocks and keeps each block's internal order.
        keys = rng.random((n_b, n_blocks))[:, block_id]
        order = np.argsort(keys, axis=1, kind="stable")
        out[:, start:stop] = start + np.take_along_axis(rotated, order, axis=1)
    return out


def stationary_indices(
    n: int, *, block_length: int, n_resamples: int, seed: int
) -> np.ndarray:
    """Stationary-bootstrap indices, delegated to :mod:`panelary.validation`."""
    return block_bootstrap_indices(
        n,
        block_length=max(1, int(block_length)),
        n_boot=int(n_resamples),
        scheme="stationary",
        seed=seed,
    )


def common_time_indices(
    dates: np.ndarray, *, n_resamples: int, seed: int, block: int = 1
) -> np.ndarray:
    """**The panel null**: ``(B, T)`` block permutations of the date axis.

    Given the sorted unique date axis (length ``T``), return index matrices that
    permute whole **date columns** in blocks of ``block``, to be applied
    **identically to every entity**. This breaks the ``x``-``y`` temporal
    alignment while preserving (a) each date's full cross-section -- hence
    every common macro shock -- and (b) serial structure up to the block
    length. Per-entity independent shuffles do neither and are invalid whenever
    entities load on a common factor, which in an equity panel is always.
    """
    t = int(np.asarray(dates).shape[0])
    if t < 2:
        raise ValueError(f"need at least 2 dates, got {t}.")
    return block_permutation_indices(
        t, block_length=max(1, int(block)), n_resamples=n_resamples, seed=seed
    )


def entity_permutation_indices(
    n_entities: int, *, n_resamples: int, seed: int
) -> np.ndarray:
    """``(B, N)`` permutations re-pairing entity ``i``'s ``y`` with entity
    ``perm[i]``'s ``x`` on the same dates.

    Tests *cross-sectional matching* ("does this entity's x go with this
    entity's y?"), not time-series dependence. Common shocks are preserved on
    both sides, so it is valid under them -- but it is a different hypothesis
    from :func:`common_time_indices`; pick deliberately.
    """
    if n_entities < 2:
        raise ValueError(f"need at least 2 entities, got {n_entities}.")
    rng = _rng(seed)
    return np.stack([rng.permutation(n_entities) for _ in range(int(n_resamples))])


# --------------------------------------------------------------------------- #
# Surrogates
# --------------------------------------------------------------------------- #
def phase_randomise(x: np.ndarray, *, n_resamples: int, seed: int) -> SurrogateResult:
    """Fourier phase-randomised surrogates.

    ``rfft``, replace every phase by a uniform draw (the DC term, and the
    Nyquist term for even ``n``, stay real), ``irfft``. The power spectrum --
    hence the linear autocorrelation -- is preserved exactly; nonlinear
    structure is destroyed; the marginal is Gaussianised.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    n = xa.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    rng = _rng(seed)
    spec = np.fft.rfft(xa - xa.mean())
    amp = np.abs(spec)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=(int(n_resamples), amp.size))
    phases[:, 0] = 0.0
    if n % 2 == 0:
        phases[:, -1] = 0.0
    sur = np.fft.irfft(amp * np.exp(1j * phases), n=n, axis=-1) + xa.mean()
    return SurrogateResult(sur, None, 1, np.ones(int(n_resamples), dtype=bool))


def iaaft(
    x: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> SurrogateResult:
    """Iterative amplitude-adjusted Fourier-transform surrogates.

    Schreiber & Schmitz (1996): start from a random shuffle and alternate (1)
    impose the target amplitude spectrum in Fourier space, (2) impose the
    target rank order in the time domain. The surrogates keep **both** the
    spectrum (approximately) and the **exact empirical marginal**, which makes
    them the right null for a rank statistic on a fat-tailed series. Iteration
    stops when no surrogate's rank order changes (or the fraction changing is
    below ``tol``), reported as ``n_iter`` / ``converged``.

    Every surrogate is an exact rearrangement of ``x``, returned both as
    ``values`` and as the index permutation ``indices``.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    n = xa.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    rng = _rng(seed)
    b = int(n_resamples)
    order_x = np.argsort(xa, kind="stable")
    sorted_x = xa[order_x]
    amp = np.abs(np.fft.rfft(xa))
    idx = np.stack([rng.permutation(n) for _ in range(b)])
    sur = xa[idx]
    rank_x = np.empty(n, dtype=np.int64)
    rank_x[order_x] = np.arange(n)
    prev_rank = rank_x[idx]  # rank of each surrogate value (no double argsort)
    converged = np.zeros(b, dtype=bool)
    n_iter = 0
    for it in range(1, int(max_iter) + 1):
        n_iter = it
        spec = np.fft.rfft(sur, axis=-1)
        spec = amp * np.exp(1j * np.angle(spec))
        smooth = np.fft.irfft(spec, n=n, axis=-1)
        order = np.argsort(smooth, axis=-1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(b, axis=0), axis=-1)
        changed = (rank != prev_rank).mean(axis=-1)
        converged = changed <= tol
        prev_rank = rank
        sur = sorted_x[rank]
        if converged.all():
            break
    indices = order_x[prev_rank]
    return SurrogateResult(sur, indices, n_iter, converged)


# --------------------------------------------------------------------------- #
# Null resolution
# --------------------------------------------------------------------------- #
_ALIASES = {"iid": "asymptotic"}


def resolve_null(
    method: str,
    null: str,
    *,
    serial: float,
    panel: bool,
    closed_form: bool = True,
) -> tuple[str, list[str]]:
    """Map a requested ``null`` to a concrete scheme and collect warnings.

    Parameters
    ----------
    method : str
        Statistic name (a key of :data:`NULL_POLICY`).
    null : str
        Requested null (``"auto"`` applies the policy table).
    serial : float
        The :func:`serial_dependence` pre-check value.
    panel : bool
        True for a multi-entity panel on a shared date axis.
    closed_form : bool, default=True
        Whether the statistic has a closed-form i.i.d. null in this context;
        if not, ``"asymptotic"`` becomes ``"permutation"``.

    Returns
    -------
    (str, list of str)
        The concrete scheme and human-readable warnings (also emitted as a
        :class:`SerialDependenceWarning` for an explicit i.i.d. request on
        serially dependent data).
    """
    if null not in NULL_METHODS:
        raise ValueError(
            f"unknown `null` {null!r}; expected one of {sorted(NULL_METHODS)}."
        )
    policy = NULL_POLICY.get(method, NULL_POLICY["spearman"])
    notes: list[str] = []
    is_serial = serial > SERIAL_THRESHOLD
    if null == "auto":
        if panel:
            chosen = policy["panel"]
        elif is_serial:
            chosen = policy["serial"]
        else:
            chosen = policy["iid"]
    else:
        chosen = _ALIASES.get(null, null)
        if chosen in {"asymptotic", "permutation", "gamma", "t", "exact"} and is_serial:
            msg = (
                f"null={null!r} assumes serially independent data, but the lag-1 "
                f"rank autocorrelation pre-check is {serial:.2f} (> "
                f"{SERIAL_THRESHOLD}); the p-value is likely far too small. Use "
                "null='auto', 'block', 'shift' or 'iaaft'."
            )
            notes.append(msg)
            warnings.warn(msg, SerialDependenceWarning, stacklevel=3)
    if chosen in {"asymptotic", "gamma", "t", "exact"} and not closed_form:
        chosen = "permutation"
    return chosen, notes
