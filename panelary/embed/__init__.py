"""Leak-safe numerical embeddings for panel data.

Fast, CPU-only, pure numpy + polars: the numerical analogue of static text
embeddings. Four layers, kept separate on purpose:

.. code-block:: text

    [1 causal window] -> [2 nonlinear expansion] -> [3 compaction] -> [4 probe]
     trailing/per-date    QUANT, RandIntC22,         srp/pca/svd      PreValidated
     (never centred)      MiniRocket-PPV, Hydra,                      Ridge
                          RFF, TensorSketch

Layer 2 -- expansions (one output row per ``(entity, time)`` row)
    :class:`QuantEmbedder` (flagship: no fitted state at all),
    :class:`RandIntC22` (catch24 over random intervals), :class:`HydraEmbedder`
    (competing kernels), :class:`CausalMiniRocket` (fitted biases),
    :class:`RandomFourierFeatures` (with mandatory direct links and a past-only
    bandwidth), :class:`TensorSketch` (polynomial kernel), and the
    cross-sectional :class:`CrossRocket`.
Layer 3 -- :class:`EmbeddingCompressor` (a thin dispatch to
    :mod:`panelary.shape`).
Layer 4 -- :class:`PreValidatedRidge`, which refuses ``alpha = 0``.
Cross-sectional mode -- :class:`CrossSectionalEmbedder` (per-date fit, refuses
    to emit unaligned components; :func:`procrustes_rotation`).
Guardrails -- :func:`reversal_check`, :func:`mechanical_baseline`,
    :func:`naive_baselines`, :func:`null_panel`, :func:`null_distribution`,
    :func:`baseline_report`.

The contract
------------
Every public transform declares ``panel_safe`` / ``leakage_safe`` and the two
enforced attributes ``fit_is_empty`` (``transform`` is a pure function of each
row's own window and the seed; ``fit`` is handed a zero-row frame) and
``is_cross_sectional`` (fits per date, never across dates). Every one exposes
``get_state()`` -> :class:`EmbeddingState`. Outputs are fixed-size ``pl.Array``
columns (``Float32`` by default), never thousands of top-level columns unless
asked. See ``docs/user-guide/embeddings.md``.
"""

from __future__ import annotations

from panelary.embed._c22i import RandIntC22
from panelary.embed._compress import METHODS as COMPRESS_METHODS
from panelary.embed._compress import EmbeddingCompressor
from panelary.embed._contract import (
    EMBED_STATE_VERSION,
    Embedder,
    EmbeddingState,
    EmbeddingWarmupWarning,
)
from panelary.embed._cross_rocket import CrossRocket
from panelary.embed._diagnostics import (
    NullReport,
    ReversalReport,
    baseline_report,
    mechanical_baseline,
    naive_baselines,
    null_distribution,
    null_panel,
    reversal_check,
)
from panelary.embed._hydra import HydraEmbedder
from panelary.embed._intervals import (
    IntervalSet,
    dilation_ladder,
    dyadic_intervals,
    random_dilated_intervals,
)
from panelary.embed._probe import (
    DEFAULT_ALPHAS,
    OverparameterisedReadoutWarning,
    PreValidatedRidge,
)
from panelary.embed._quant import REPRESENTATIONS as QUANT_REPRESENTATIONS
from panelary.embed._quant import QuantEmbedder, quant_features
from panelary.embed._rff import RandomFourierFeatures
from panelary.embed._rocket import CausalMiniRocket
from panelary.embed._sketch import TensorSketch
from panelary.embed._xs import ALIGNMENTS, CrossSectionalEmbedder, procrustes_rotation

__all__ = [
    "ALIGNMENTS",
    "COMPRESS_METHODS",
    "DEFAULT_ALPHAS",
    "EMBED_STATE_VERSION",
    "QUANT_REPRESENTATIONS",
    "CausalMiniRocket",
    "CrossRocket",
    "CrossSectionalEmbedder",
    "Embedder",
    "EmbeddingCompressor",
    "EmbeddingState",
    "EmbeddingWarmupWarning",
    "HydraEmbedder",
    "IntervalSet",
    "NullReport",
    "OverparameterisedReadoutWarning",
    "PreValidatedRidge",
    "QuantEmbedder",
    "RandIntC22",
    "RandomFourierFeatures",
    "ReversalReport",
    "TensorSketch",
    "baseline_report",
    "dilation_ladder",
    "dyadic_intervals",
    "mechanical_baseline",
    "naive_baselines",
    "null_distribution",
    "null_panel",
    "procrustes_rotation",
    "quant_features",
    "random_dilated_intervals",
    "reversal_check",
]
