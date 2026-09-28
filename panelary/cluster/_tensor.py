"""Re-export of the panel tensor builder, which now lives in :mod:`panelary.shape._tensor`.

``build_tensor`` and ``PanelTensor`` were written here as a k-Shape helper, but
nothing about them is k-Shape-specific: they are the materialization boundary
between a long Polars panel and a dense ``(entity, time, value)`` array, with the
causal policy (forward-fill only, expanding z-normalisation, NaN never
fabricated) that every shape transform needs. They moved to
:mod:`panelary.shape._tensor` unchanged; this module keeps the old import path
working so :mod:`panelary.cluster` is untouched.
"""

from __future__ import annotations

from panelary.shape._tensor import PanelTensor, build_tensor

__all__ = ["PanelTensor", "build_tensor"]
