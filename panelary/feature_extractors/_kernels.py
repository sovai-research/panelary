"""Vendored numeric kernels and shared type aliases.

Base layer of :mod:`panelary.feature_extractors`: it imports nothing else from
inside the package, which keeps the dependency direction one-way
(``_kernels`` <- ``_stats``/``_finance`` <- ``_namespace``/``_catalogue``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import polars as pl

try:  # polars>=1.0 moved type aliases to the private `_typing` module
    from polars._typing import ClosedInterval  # noqa: F401  (re-exported below)
except ImportError:  # pragma: no cover - older polars
    from polars.type_aliases import ClosedInterval  # noqa: F401

from panelary._internal._jit import lazy_njit


def ricker(points: int, a: float) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet.

    Vendored from the implementation removed in SciPy 1.15 so the CWT-based
    feature extractors keep working across SciPy versions.
    """
    A = 2 / (np.sqrt(3 * a) * np.pi**0.25)
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss


def _lempel_ziv_complexity_count(bits: bytes) -> int:
    """Lempel-Ziv complexity of a binary sequence (count of distinct phrases).

    Pure-Python transcription of the former ``pl_lempel_ziv_complexity`` Rust
    kernel (a ``bytes`` + ``set`` implementation is competitive with / faster
    than the Rust ``HashSet<&[bool]>`` version per the D5 audit). ``bits`` holds
    one byte (0 or 1) per element.
    """
    n = len(bits)
    ind = 0
    inc = 1
    sub_strings: set[bytes] = set()
    while ind + inc <= n:
        subseq = bits[ind : ind + inc]
        if subseq in sub_strings:
            inc += 1
        else:
            sub_strings.add(subseq)
            ind += inc
            inc = 1
    return len(sub_strings)


def _lempel_ziv_complexity_batch(s: pl.Series) -> pl.Series:
    """map_batches kernel: boolean Series -> length-1 UInt32 complexity.

    Nulls are interpreted as ``False`` (0 in the bit sequence), matching the
    former Rust plugin.
    """
    arr = s.fill_null(False).to_numpy()
    bits = arr.astype(np.uint8).tobytes()
    return pl.Series([_lempel_ziv_complexity_count(bits)], dtype=pl.UInt32)


def _cusum_events_py(
    values: np.ndarray, threshold: float, warmup_period: int, drift: float
) -> np.ndarray:
    """Pure-Python/numpy CUSUM change-point filter.

    Exact transcription of the former Rust kernel
    (``src/changepoint_detection/cusum.rs``). ``values`` is a float64 array in
    which nulls are represented as ``NaN`` (treated as the Rust ``None``).

    Returns an ``int32`` array of the same length with ``1`` at each detected
    change point and ``0`` elsewhere.
    """
    n = values.shape[0]
    events = np.zeros(n, dtype=np.int32)

    s_pos = 0.0
    s_neg = 0.0
    t = 0
    mu = 0.0
    sigma = 0.0
    obs: list[float] = []

    # numpy IEEE division (inf/nan on divide-by-zero) matches the Rust f64
    # behaviour; Python's ``float`` division would instead raise. Values are kept
    # as ``np.float64`` scalars so the no-sigma-guard division mirrors Rust.
    for i in range(n):
        value = values[i]
        is_null = bool(np.isnan(value))
        warming_up = t < warmup_period
        warmup_end = t == warmup_period

        if warming_up:
            if not is_null:
                obs.append(value)
            events[i] = 0
            t += 1
            continue

        if warmup_end:
            # Two-pass population mean/std over collected observations, once —
            # summed left-to-right to match the Rust ``obs.iter().sum()`` order.
            count = len(obs)
            total = 0.0
            for x in obs:
                total += x
            # np.float64, not Python float: when every warmup observation is
            # NaN, `obs` is empty and count == 0. Python division would raise
            # ZeroDivisionError; Rust f64 (and the numba kernel under
            # error_model="numpy") yield NaN. Keep the IEEE behaviour.
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = np.float64(total) / np.float64(count)
                sq = 0.0
                for x in obs:
                    sq += (x - mu) ** 2
                sigma = np.sqrt(np.float64(sq) / np.float64(count))
            t += 1
            # fall through to process THIS value (no continue)

        if not is_null:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = (value - mu) / sigma  # no zero-sigma guard (match Rust exactly)
            s_pos = max(s_pos + v - drift, 0.0)
            s_neg = min(s_neg + v + drift, 0.0)
            if s_pos > threshold:
                events[i] = 1
                s_pos = 0.0
                s_neg = 0.0
                t = 0
                obs = []
            elif s_neg < -threshold:
                events[i] = 1
                s_neg = 0.0
                s_pos = 0.0
                t = 0
                obs = []
            else:
                events[i] = 0
        else:
            events[i] = 0

    return events


# error_model="numpy" (fixed by `_internal._jit`) is load-bearing, not a tuning
# knob: numba's default ("python") RAISES ZeroDivisionError on float division by
# zero, while the pure-Python path relies on numpy IEEE semantics (inf/nan) to
# reproduce the original Rust f64 behaviour exactly. Without it the `fast` extra
# silently changes results on a constant warmup window (sigma == 0), which is the
# divergence tests/test_cusum_pure.py::test_numba_matches_python detects.
@lazy_njit
def _cusum_events_kernel(
    values: np.ndarray, threshold: float, warmup_period: int, drift: float
) -> np.ndarray:  # pragma: no cover - exercised only when numba installed
    n = values.shape[0]
    events = np.zeros(n, dtype=np.int32)

    s_pos = 0.0
    s_neg = 0.0
    t = 0
    mu = 0.0
    sigma = 0.0
    obs = np.empty(n, dtype=np.float64)
    obs_count = 0

    for i in range(n):
        value = values[i]
        is_null = np.isnan(value)
        warming_up = t < warmup_period
        warmup_end = t == warmup_period

        if warming_up:
            if not is_null:
                obs[obs_count] = value
                obs_count += 1
            events[i] = 0
            t += 1
            continue

        if warmup_end:
            # Two-pass population mean/std, summed left-to-right to match Rust.
            total = 0.0
            for j in range(obs_count):
                total += obs[j]
            mu = total / obs_count
            sq = 0.0
            for j in range(obs_count):
                sq += (obs[j] - mu) ** 2
            sigma = math.sqrt(sq / obs_count)
            t += 1

        if not is_null:
            v = (value - mu) / sigma  # IEEE division (inf/nan on /0), matches Rust
            s_pos = max(s_pos + v - drift, 0.0)
            s_neg = min(s_neg + v + drift, 0.0)
            if s_pos > threshold:
                events[i] = 1
                s_pos = 0.0
                s_neg = 0.0
                t = 0
                obs_count = 0
            elif s_neg < -threshold:
                events[i] = 1
                s_neg = 0.0
                s_pos = 0.0
                t = 0
                obs_count = 0
            else:
                events[i] = 0
        else:
            events[i] = 0

    return events


def _get_cusum_numba() -> Callable[[np.ndarray, float, int, float], np.ndarray] | None:
    """Return a numba-compiled CUSUM kernel if ``numba`` is installed, else None.

    numba is the optional ``fast`` extra; it must never be imported eagerly or
    become a hard dependency. Compilation is deferred to first use and cached
    (see :mod:`panelary._internal._jit`).
    """
    return _cusum_events_kernel.compiled()


def _cusum_events(
    values: np.ndarray, threshold: float, warmup_period: int, drift: float
) -> np.ndarray:
    """CUSUM change-point filter, dispatching to the numba fast-path if available.

    Falls back to the pure-Python/numpy loop when numba (the ``fast`` extra) is
    not installed. Both paths produce identical ``int32`` event arrays.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    kernel = _get_cusum_numba()
    if kernel is not None:
        return np.asarray(
            kernel(values, float(threshold), int(warmup_period), float(drift))
        )
    return _cusum_events_py(values, float(threshold), int(warmup_period), float(drift))


TIME_SERIES_T = pl.Series | pl.Expr
FLOAT_EXPR = float | pl.Expr
FLOAT_INT_EXPR = int | float | pl.Expr
INT_EXPR = int | pl.Expr
LIST_EXPR = list | pl.Expr
BOOL_EXPR = bool | pl.Expr
MAP_EXPR = Mapping[str, float] | pl.Expr
MAP_LIST_EXPR = Mapping[str, list[float]] | pl.Expr


# from polars.type_aliases import IntoExpr
