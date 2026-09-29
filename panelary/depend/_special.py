"""Accurate normal tail probabilities for the closed-form nulls.

``Phi(-z)`` computed as ``1 - Phi(z)`` (or through a CDF that clips) underflows
to exactly 0 around ``z ~ 8``; a p-value of 0 then outranks every honest small
p-value in a screen. ``math.erfc`` is accurate to its underflow near
``z ~ 38`` (``p ~ 1e-316``).
"""

from __future__ import annotations

from panelary._internal._special import norm_sf

__all__ = ["norm_sf"]
