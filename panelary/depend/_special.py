"""Accurate normal tail probabilities for the closed-form nulls.

``Phi(-z)`` computed as ``1 - Phi(z)`` (or through a CDF that clips) underflows
to exactly 0 around ``z ~ 8``; a p-value of 0 then outranks every honest small
p-value in a screen. ``math.erfc`` is accurate to its underflow near
``z ~ 38`` (``p ~ 1e-316``).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["norm_sf"]

_ERFC = np.frompyfunc(math.erfc, 1, 1)


def norm_sf(z: float | np.ndarray) -> float | np.ndarray:
    """Standard normal upper tail ``P(Z >= z)``, accurate far into the tail."""
    arr = np.asarray(z, dtype=np.float64)
    out = np.asarray(_ERFC(arr / math.sqrt(2.0)), dtype=np.float64) * 0.5
    out = np.where(np.isnan(arr), np.nan, out)
    return float(out) if out.ndim == 0 else out
