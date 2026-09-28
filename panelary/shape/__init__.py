"""Shape algebra for panels: compress an axis, lift into a basis, factorise rank.

``panelary.shape`` holds transforms that change the *shape* of a panel -- the
width of a named axis, the order of the tensor, or the numerical rank of a
factorisation -- while leaving the meaning of ``entity`` / ``time`` /
``feature`` intact. It is pure numpy + polars: no scipy, sklearn or tensorly on
any default import path.

Intent, not "dimensionality reduction"
--------------------------------------
Every transform declares one :class:`Intent` (``COMPRESS``, ``LIFT``,
``FACTORIZE``, ``SKETCH``) and one :class:`Axis` (``feature``, ``time``,
``entity``, ``lag``). The axis *is* the leak contract:

* ``axis="feature"`` -- a row-wise map; leak-safe iff fit on training rows.
* ``axis="time"`` -- along one entity's history; leak-safe **iff** the flavour
  is ``"trailing"`` (the default): one output row per input row, from that
  row's own trailing ``window``. ``flavour="whole_series"`` is an explicit
  opt-in, returns one row per **entity**, and is ``leakage_safe = False``.
* ``axis="entity"`` -- across entities at one date; never panel-safe, always
  date-local, and refused as a panel column until components are aligned
  across dates (``align="procrustes"``).

The spine
---------
* :func:`build_tensor` / :func:`to_long` -- the materialization boundary
  (long frame <-> dense ``(entity, time, value)`` tensor), with
  :func:`build_sequences` and the :class:`Ragged` policy for unequal entities.
* :func:`trailing_windows` / :class:`PanelWindows` -- the causal windower.
* :func:`as_embedding` / :func:`explode_embedding` -- ``pl.Array`` output.
* :class:`ShapeTransform`, :class:`ShapeSpec`, :class:`Plan`,
  :class:`ShapeBudgetError`, :func:`plan_chain` -- the base class and the
  "refuse before allocating" budget.

The primitives
--------------
* Feature axis: :class:`RandomizedPCA` (:func:`randomized_svd`),
  :class:`SparseRandomProjection` (:func:`sparse_rp_matrix`), :class:`SRHT`
  (:func:`srht`, :func:`fwht`), :class:`CountSketch`,
  :class:`FrequentDirections`, :class:`ColumnSubset` (:func:`interpolative`,
  :func:`pivoted_qr`), :class:`CUR`.
* Time axis: :class:`PAA`, :class:`Spectral`, :class:`Delay`.
* Tensor modes: :class:`PartialTucker` (:func:`partial_tucker`, :func:`hosvd`,
  :func:`unfold`, :func:`fold`, :func:`mode_dot`).
* Entity axis: :class:`CrossSectionalRandomizedPCA`.
* :func:`stability` / :func:`procrustes` -- per-fold loading dispersion.

Import cost and the operator catalogue
--------------------------------------
This package initialiser is **lazy** (PEP 562). ``import panelary`` loads it
(``panelary.cluster`` re-exports :func:`build_tensor` from
:mod:`panelary.shape._tensor`) but no transform module: the kernels are not
imported and nothing is registered until a public name is first read from
``panelary.shape``. That first access imports every transform module, which
registers one :class:`~panelary.registry.FeatureSpec` per transform that emits
a panel column in its default configuration, under ``namespace="shape"``.

Two public transforms are deliberately **not** in the catalogue:
:class:`PartialTucker` (whole-series only, ``leakage_safe=False``) and
:class:`CrossSectionalRandomizedPCA` (its default refuses to emit a panel
column; it is ``panel_safe=False`` by design). The ``whole_series`` flavour of
``PAA`` / ``Spectral`` is a mode of the registered (trailing) spec, not a
separate entry.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static names for type checkers and IDEs
    from panelary.shape._array import as_embedding, explode_embedding
    from panelary.shape._axes import (
        Axis,
        Flavour,
        InputShape,
        Intent,
        Plan,
        ShapeBudgetError,
        ShapeSpec,
        ShapeTransform,
        default_max_bytes,
        plan_chain,
    )
    from panelary.shape._delay import Delay
    from panelary.shape._explain import procrustes, stability
    from panelary.shape._id import (
        CUR,
        ColumnSubset,
        CURFactors,
        interpolative,
        pivoted_qr,
    )
    from panelary.shape._paa import PAA
    from panelary.shape._project import (
        SRHT,
        SparseRandomProjection,
        fwht,
        sparse_rp_matrix,
        srht,
    )
    from panelary.shape._rsvd import (
        CrossSectionalRandomizedPCA,
        RandomizedPCA,
        randomized_svd,
    )
    from panelary.shape._sketch import CountSketch, FrequentDirections
    from panelary.shape._spectral import Spectral
    from panelary.shape._tensor import (
        PanelTensor,
        Ragged,
        RaggedSequences,
        build_sequences,
        build_tensor,
        to_long,
    )
    from panelary.shape._tucker import (
        PartialTucker,
        TuckerDecomposition,
        fold,
        hosvd,
        mode_dot,
        partial_tucker,
        tucker_to_tensor,
        unfold,
    )
    from panelary.shape._window import PanelWindows, trailing_windows

#: Public name -> the private module that defines it.
_EXPORTS: dict[str, str] = {
    # vocabulary + base
    "Axis": "_axes",
    "Flavour": "_axes",
    "InputShape": "_axes",
    "Intent": "_axes",
    "Plan": "_axes",
    "ShapeBudgetError": "_axes",
    "ShapeSpec": "_axes",
    "ShapeTransform": "_axes",
    "default_max_bytes": "_axes",
    "plan_chain": "_axes",
    # spine
    "PanelTensor": "_tensor",
    "Ragged": "_tensor",
    "RaggedSequences": "_tensor",
    "build_sequences": "_tensor",
    "build_tensor": "_tensor",
    "to_long": "_tensor",
    "PanelWindows": "_window",
    "trailing_windows": "_window",
    "as_embedding": "_array",
    "explode_embedding": "_array",
    # feature axis
    "RandomizedPCA": "_rsvd",
    "randomized_svd": "_rsvd",
    "SparseRandomProjection": "_project",
    "SRHT": "_project",
    "fwht": "_project",
    "sparse_rp_matrix": "_project",
    "srht": "_project",
    "CountSketch": "_sketch",
    "FrequentDirections": "_sketch",
    "ColumnSubset": "_id",
    "CUR": "_id",
    "CURFactors": "_id",
    "interpolative": "_id",
    "pivoted_qr": "_id",
    # time axis
    "Delay": "_delay",
    "PAA": "_paa",
    "Spectral": "_spectral",
    # tensor modes
    "PartialTucker": "_tucker",
    "TuckerDecomposition": "_tucker",
    "fold": "_tucker",
    "hosvd": "_tucker",
    "mode_dot": "_tucker",
    "partial_tucker": "_tucker",
    "tucker_to_tensor": "_tucker",
    "unfold": "_tucker",
    # entity axis
    "CrossSectionalRandomizedPCA": "_rsvd",
    # explainability
    "procrustes": "_explain",
    "stability": "_explain",
}

#: Every transform module (the ones that register FeatureSpecs at import time).
_CATALOGUE_MODULES: tuple[str, ...] = (
    "_paa",
    "_spectral",
    "_delay",
    "_rsvd",
    "_project",
    "_sketch",
    "_id",
    "_tucker",
)

__all__ = sorted(_EXPORTS)


def _load_catalogue() -> None:
    """Import every transform module, registering the ``shape`` catalogue once."""
    for mod in _CATALOGUE_MODULES:
        importlib.import_module(f"{__name__}.{mod}")


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_catalogue()
    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
