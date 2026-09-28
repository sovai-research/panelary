"""Interval and dilation samplers shared by the windowed embedders.

Every quantity here is a function of the **window length and the parameters
alone** -- never of the data, and never of how many rows follow a window. That
is what lets the interval-based embedders declare ``fit_is_empty = True``: the
intervals are drawn when the object is built, and fitting on any two datasets
gives byte-identical state.

Three samplers, one definition each:

* :func:`dilation_ladder` -- the exponential dilation schedule ``1, 2, 4, ...``
  up to the largest dilation at which a ``span``-tap pattern still fits inside
  the window. Used by :class:`~panelary.embed.HydraEmbedder` (``span = 9``) and
  by :func:`random_dilated_intervals` (``span = interval length``), so the
  dilation formula has exactly one definition.
* :func:`random_dilated_intervals` -- seeded random ``(start, length,
  dilation)`` triples for :class:`~panelary.embed.RandIntC22`.
* :func:`dyadic_intervals` -- QUANT's deterministic dyadic partition (plus the
  half-shifted copies) for :class:`~panelary.embed.QuantEmbedder`.

An interval with ``start = s``, ``length = m`` and ``dilation = d`` reads the
window positions ``s, s + d, ..., s + (m - 1) d`` (0-based from the oldest cell
of the window). Every sampler guarantees ``s + (m - 1) d <= window - 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IntervalSet",
    "dilation_ladder",
    "dyadic_intervals",
    "max_dilation_exponent",
    "random_dilated_intervals",
]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"`{name}` must be a positive int, got {value!r}.")
    return int(value)


def max_dilation_exponent(length: int, span: int) -> int:
    """Largest ``e >= 0`` with ``(span - 1) * 2**e <= length - 1``.

    The one dilation formula: a ``span``-tap pattern at dilation ``d`` covers
    ``(span - 1) * d + 1`` cells, which must fit in ``length``.

    Parameters
    ----------
    length : int
        Window (or series) length.
    span : int
        Number of taps / points in the dilated pattern (``>= 1``).

    Returns
    -------
    int
        ``0`` when even dilation 2 does not fit (or ``span == 1``).

    Raises
    ------
    ValueError
        If ``span > length``: the pattern does not fit even undilated.
    """
    length = _positive_int(length, "length")
    span = _positive_int(span, "span")
    if span > length:
        raise ValueError(
            f"a {span}-point pattern does not fit in a window of {length} cells."
        )
    if span == 1:
        return 0
    return int(math.floor(math.log2((length - 1) / (span - 1)) + 1e-12))


def dilation_ladder(length: int, span: int) -> NDArray[np.int64]:
    """Exponential dilations ``2**0 .. 2**max_dilation_exponent(length, span)``.

    Parameters
    ----------
    length : int
        Window length.
    span : int
        Taps in the dilated pattern (e.g. 9 for a length-9 kernel).

    Returns
    -------
    numpy.ndarray
        ``int64`` dilations, ascending.

    Examples
    --------
    >>> dilation_ladder(64, 9).tolist()
    [1, 2, 4]
    """
    e = max_dilation_exponent(length, span)
    return (2 ** np.arange(e + 1)).astype(np.int64)


@dataclass(frozen=True)
class IntervalSet:
    """A fixed set of dilated intervals over a window of known length.

    Attributes
    ----------
    window : int
        The window length the intervals were drawn for.
    start, length, dilation : numpy.ndarray
        ``int64`` arrays of equal size; interval ``i`` reads positions
        ``start[i] + dilation[i] * arange(length[i])``.
    """

    window: int
    start: NDArray[np.int64]
    length: NDArray[np.int64]
    dilation: NDArray[np.int64]

    def __post_init__(self) -> None:
        n = self.start.shape[0]
        if self.length.shape != (n,) or self.dilation.shape != (n,):
            raise ValueError("IntervalSet: start, length and dilation must align.")
        if n and (
            int(self.start.min()) < 0
            or int(self.length.min()) < 1
            or int(self.dilation.min()) < 1
            or int((self.start + (self.length - 1) * self.dilation).max())
            > self.window - 1
        ):
            raise ValueError("IntervalSet: an interval reaches outside the window.")

    def __len__(self) -> int:
        return int(self.start.shape[0])

    def indices(self, i: int) -> NDArray[np.int64]:
        """Window positions read by interval ``i``, oldest first."""
        return self.start[i] + self.dilation[i] * np.arange(
            self.length[i], dtype=np.int64
        )

    def as_dict(self) -> dict[str, Any]:
        """Plain arrays, for :class:`~panelary.embed.EmbeddingState`."""
        return {
            "interval_start": self.start.copy(),
            "interval_length": self.length.copy(),
            "interval_dilation": self.dilation.copy(),
        }


def random_dilated_intervals(
    window: int,
    n_intervals: int,
    *,
    seed: int,
    min_length: int = 20,
) -> IntervalSet:
    """Draw ``n_intervals`` random dilated intervals inside a window.

    For each interval, independently and from one seeded generator:

    1. a dilation ``d = floor(2**u)`` with ``u ~ U(0, e_max)``, where
       ``e_max = log2((window - 1) / (min_length - 1))`` -- the continuous form
       of :func:`max_dilation_exponent`, so small dilations are as likely per
       octave as large ones;
    2. a length ``m`` uniform on ``[min_length, (window - 1) // d + 1]`` (the
       longest interval that fits at that dilation);
    3. a start uniform on ``[0, window - ((m - 1) d + 1)]``.

    ``min_length`` is capped at ``window``; if the cap binds, every interval is
    the whole window at dilation 1.

    Parameters
    ----------
    window : int
        Window length.
    n_intervals : int
        Number of intervals.
    seed : int
        Explicit seed. The result is a deterministic function of
        ``(window, n_intervals, seed, min_length)``.
    min_length : int, default 20
        Shortest interval. Several catch22 features are undefined below ~20
        points (the fluctuation-analysis pair returns NaN), hence the default.

    Returns
    -------
    IntervalSet

    Raises
    ------
    ValueError
        On non-positive sizes.
    TypeError
        If ``seed`` is not an int.
    """
    window = _positive_int(window, "window")
    n_intervals = _positive_int(n_intervals, "n_intervals")
    min_length = _positive_int(min_length, "min_length")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"`seed` must be an explicit int, got {seed!r}.")
    m_min = min(min_length, window)
    rng = np.random.default_rng([int(seed), window, n_intervals, m_min])
    start = np.empty(n_intervals, dtype=np.int64)
    length = np.empty(n_intervals, dtype=np.int64)
    dilation = np.empty(n_intervals, dtype=np.int64)
    e_max = math.log2((window - 1) / (m_min - 1)) if m_min > 1 and window > 1 else 0.0
    for i in range(n_intervals):
        d = int(math.floor(2.0 ** rng.uniform(0.0, e_max))) if e_max > 0 else 1
        d = max(1, d)
        m_hi = (window - 1) // d + 1 if d > 0 else window
        m = int(rng.integers(m_min, max(m_min, m_hi) + 1))
        cover = (m - 1) * d + 1
        s = int(rng.integers(0, window - cover + 1))
        start[i], length[i], dilation[i] = s, m, d
    return IntervalSet(window=window, start=start, length=length, dilation=dilation)


@dataclass(frozen=True)
class DyadicInterval:
    """One QUANT interval: ``[start, stop)`` at ``depth`` (0 = whole series)."""

    start: int
    stop: int
    depth: int
    shifted: bool

    @property
    def length(self) -> int:
        return self.stop - self.start


def dyadic_intervals(length: int, depth: int) -> list[DyadicInterval]:
    """QUANT's dyadic interval set over a series of ``length`` cells.

    At depth ``j = 0 .. depth - 1`` the series is split into ``2**j``
    contiguous intervals of (near-)equal length; at every depth ``j >= 1`` the
    same partition shifted right by half an interval is added, keeping only the
    shifted intervals that fit. The effective depth is capped at
    ``floor(log2(length)) + 1`` so no interval is empty.

    Parameters
    ----------
    length : int
        Series (representation) length.
    depth : int
        Requested depth (``>= 1``).

    Returns
    -------
    list of DyadicInterval
        Depth-major order; within a depth, the unshifted intervals first.

    Examples
    --------
    >>> [(iv.start, iv.stop) for iv in dyadic_intervals(8, 2)]
    [(0, 8), (0, 4), (4, 8), (2, 6)]
    """
    length = _positive_int(length, "length")
    depth = _positive_int(depth, "depth")
    eff = min(depth, int(math.floor(math.log2(length))) + 1)
    out: list[DyadicInterval] = []
    for j in range(eff):
        n_parts = 2**j
        edges = np.floor(np.linspace(0, length, n_parts + 1)).astype(np.int64)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            if b > a:
                out.append(DyadicInterval(int(a), int(b), j, False))
        if j >= 1:
            shift = (length // n_parts) // 2
            if shift < 1:
                continue
            for a, b in zip(edges[:-1], edges[1:], strict=True):
                a2, b2 = int(a) + shift, int(b) + shift
                if b2 <= length and b2 > a2:
                    out.append(DyadicInterval(a2, b2, j, True))
    return out
