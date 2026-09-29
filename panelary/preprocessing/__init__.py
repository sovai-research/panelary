"""Transformers for reshaping and conditioning a panel before modelling.

``panelary.preprocessing`` is the transformer layer: reindexing and resampling,
lagging and rolling, scaling and imputation, differencing and detrending. Each
entry point is a curried :func:`~panelary.base.transformer` -- call it with its
parameters to get a ``frame -> frame`` function, so transforms compose and drop
into a ``Pipeline`` unchanged.

Leak-safety splits them in two, and the distinction is the whole story:

* **Pure per-row transforms** -- :func:`lag`, :func:`diff`, :func:`roll`,
  :func:`trim`, :func:`resample`, :func:`one_hot_encode` -- are causal by
  construction when applied ``.over(entity_col)`` in time order. The one
  exception is :func:`resample` with its historical left-edge stamping
  (``label="left"``, today's default, which warns): pass ``label="right"``.
* **Fitted transforms** -- :func:`scale`, :func:`impute`, :func:`boxcox`,
  :func:`yeojohnson`, :func:`detrend`, :func:`deseasonalize_fourier`,
  :func:`fractional_diff` -- learn state and must be fit on training rows only,
  then applied frozen. Fitting one on a whole panel is a look-ahead, and this
  package raises :class:`LeakageWarning` where it can detect the mistake.

The implementation is split across private submodules (``_base``, ``_frame``,
``_features``, ``_scaling``, ``_impute``, ``_diff``, ``_power``, ``_detrend``)
purely for readability; the public surface is this module and is unchanged.
"""

from __future__ import annotations

from panelary.preprocessing._base import PL_NUMERIC_COLS, LeakageWarning
from panelary.preprocessing._detrend import deseasonalize_fourier, detrend
from panelary.preprocessing._diff import diff, fractional_diff
from panelary.preprocessing._features import lag, one_hot_encode, roll
from panelary.preprocessing._frame import (
    coerce_dtypes,
    reindex,
    resample,
    time_to_arange,
    trim,
)
from panelary.preprocessing._impute import (
    _cafe_impute_frame,
    _require_cafe,
    cafe_impute,
    impute,
)
from panelary.preprocessing._power import boxcox, yeojohnson
from panelary.preprocessing._scaling import scale

__all__ = [
    "PL_NUMERIC_COLS",
    "LeakageWarning",
    "boxcox",
    "cafe_impute",
    "coerce_dtypes",
    "deseasonalize_fourier",
    "detrend",
    "diff",
    "fractional_diff",
    "impute",
    "lag",
    "one_hot_encode",
    "reindex",
    "resample",
    "roll",
    "scale",
    "time_to_arange",
    "trim",
    "yeojohnson",
]
