"""Event sampling and bar construction: which rows a label is defined on.

Public API
----------
bars
    Tick, volume and dollar bars (fixed or adaptive threshold) as a long panel,
    one row per completed bar stamped at its last tick.
imbalance_bars
    Tick / volume / dollar imbalance and run bars (AFML 2.3.2) with clamped,
    causal expected-bar-size EWMAs.

numpy + polars only; numba (the ``fast`` extra) is an optional speed-up loaded
on first use, never at import.
"""

from __future__ import annotations

__all__: list[str] = []

# --- bars (plan 4, M5) -------------------------------------------------------
from panelary.sample._bars import bars  # noqa: E402
from panelary.sample._imbalance import imbalance_bars  # noqa: E402

__all__ += ["bars", "imbalance_bars"]
# --- end plan 4 M5 -------------------------------------------------------------
