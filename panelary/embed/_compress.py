"""Layer 3 of the embed pipeline: compaction, as a thin dispatch to :mod:`panelary.shape`.

The best expansions are wide (QUANT ~10n features, MiniRocket ~10k); users
want 16 - 512 dimensions. The build contract keeps the two layers **separate**:
do not distort an expansion to force compactness -- compose it with a
compressor. This module owns no linear algebra. ``method`` picks a primitive
from :mod:`panelary.shape` and this class only moves rows in and out of it:

========  ===========================================================  ===========
method    primitive                                                    fitted?
========  ===========================================================  ===========
``srp``   :class:`panelary.shape._project.SparseRandomProjection`       no (seed)
``pca``   :class:`panelary.shape._rsvd.RandomizedPCA`, standardised      yes
``svd``   :class:`panelary.shape._rsvd.RandomizedPCA`, centred only      yes
``none``  identity (concatenate the inputs)                             no
========  ===========================================================  ===========

``srp`` is stateless given the seed (the projection is a function of the input
width, ``dim`` and ``seed``), so it adds no leak surface. ``pca`` / ``svd`` are
fitted on the rows passed to :meth:`fit` only -- fit them on the training fold.

**The Johnson-Lindenstrauss bound does not license ``dim=64``.** At
``n = 1e6`` and ``eps = 0.1`` it wants on the order of ``1e4`` dimensions.
Small outputs rest on empirical performance, not theory (the
:mod:`panelary.shape._project` docstring says the same). R-Clustering's evidence
puts 10 - 20 dimensions at the right size *for distance-based use*; the default
here is ``dim=512`` for feature use, and ``dim=16`` is the suggested setting for
clustering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.embed._contract import (
    _EmbedTransform,
    array_width,
    check_choice,
    check_pos_int,
    check_seed,
    emit_embedding,
    matrix_from_columns,
    resolve_matrix_columns,
)
from panelary.shape._axes import Axis, InputShape, Intent, Plan, ShapeSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["METHODS", "EmbeddingCompressor"]

#: Compression methods, in the order of the build contract.
METHODS: tuple[str, ...] = ("srp", "pca", "svd", "none")

_E, _T = "__pn_row_entity__", "__pn_row_time__"


def _as_shape_panel(M: NDArray[np.float64], names: Sequence[str]) -> PanelFrame:
    """Wrap a matrix as a panel with synthetic keys (the map is row-wise)."""
    n = M.shape[0]
    frame = pl.from_numpy(
        np.ascontiguousarray(M), schema=list(names), orient="row"
    ).with_columns(
        pl.zeros(n, dtype=pl.Int32, eager=True).alias(_E),
        pl.int_range(n, dtype=pl.Int64, eager=True).alias(_T),
    )
    return PanelFrame(frame, entity=_E, time=_T, validate=False)


class EmbeddingCompressor(_EmbedTransform):
    """Compress embedding columns to ``dim`` dimensions: ``srp | pca | svd | none``.

    Parameters
    ----------
    method : {"srp", "pca", "svd", "none"}, default "srp"
    dim : int, default 512
        Output width (``pca`` / ``svd`` cap it at ``min(n_fit_rows, width)``).
    columns : sequence of str, optional
        Input columns -- numeric and/or numeric ``pl.Array`` columns,
        concatenated in order. ``None`` = every ``pl.Array`` column if there is
        one, else every numeric feature column.
    seed : int, default 0
    density : float or "auto", default "auto"
        ``srp`` only; ``"auto"`` = ``1 / sqrt(width)`` (Li et al., 2006).
    max_fit_rows : int, default 200_000
        ``pca`` / ``svd`` fit on a seeded subsample of at most this many
        complete training rows.
    name : str, default "embedding"
        Output column (``output="array"``) or prefix (``output="columns"``).
    output : {"array", "columns"}, default "array"
    dtype : {"float32", "float16", "float64"}, default "float32"
    keep_features : bool, default False
        Keep the input columns (default: keys plus the compressed column).
    chunk_rows : int, default 16384
        Rows per call into the shape primitive at transform time.
    max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.

    Attributes
    ----------
    compressor_ : ShapeTransform or None
        The fitted shape primitive (``None`` for ``method="none"``).
    components_, mean_, scale_ : numpy.ndarray or None
        The compression state, copied off the primitive for serialisation.

    Notes
    -----
    NaN policy: a row with any missing input is emitted as null.
    ``fit_is_empty`` is declared ``False`` at class level because ``pca`` /
    ``svd`` fit; with ``method="srp"`` the fitted state is still a function of
    the schema and the seed only (``tests/test_embed_stateless.py`` checks it).
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = False
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="none",
        streaming="batch",
        cost_hint="O(n d k)",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        "method",
        "dim",
        "seed",
        "density",
        "max_fit_rows",
        "name",
        "output",
        "dtype",
        "keep_features",
        "chunk_rows",
        "max_bytes",
    )
    _state_arrays: ClassVar[tuple[str, ...]] = ("components_", "mean_", "scale_")

    def __init__(
        self,
        method: str = "srp",
        dim: int = 512,
        *,
        columns: Sequence[str] | str | None = None,
        seed: int = 0,
        density: float | str = "auto",
        max_fit_rows: int = 200_000,
        name: str = "embedding",
        output: str = "array",
        dtype: str = "float32",
        keep_features: bool = False,
        chunk_rows: int = 16384,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        owner = "EmbeddingCompressor"
        self.method = check_choice(method, "method", METHODS, owner)
        self.dim = check_pos_int(dim, "dim", owner)
        self.seed = check_seed(seed, owner)
        self.density = density
        self.max_fit_rows = check_pos_int(max_fit_rows, "max_fit_rows", owner)
        self.name = name
        self.output = check_choice(output, "output", ("array", "columns"), owner)
        self.dtype = check_choice(
            dtype, "dtype", ("float16", "float32", "float64"), owner
        )
        self.keep_features = bool(keep_features)
        self.chunk_rows = check_pos_int(chunk_rows, "chunk_rows", owner)
        self.compressor_: Any = None
        self.components_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.in_width_: int = 0
        self.out_width_: int = 0

    # ------------------------------------------------------------------ #
    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        if self.columns is None:
            arrays = [
                c for c in panel.feature_cols if isinstance(panel.schema[c], pl.Array)
            ]
            if arrays:
                return resolve_matrix_columns(panel, arrays, "EmbeddingCompressor")
        return resolve_matrix_columns(panel, self.columns, "EmbeddingCompressor")

    def _out_width(self, in_width: int) -> int:
        if self.method == "none":
            return in_width
        if self.out_width_:
            return self.out_width_
        return self.dim if self.method == "srp" else min(self.dim, in_width)

    def _plan(self, shape: InputShape) -> Plan:
        m = self._out_width(shape.width)
        chunk = min(shape.rows, self.chunk_rows)
        scratch = (
            8.0 * (3 * chunk * shape.width + shape.width * m) + 8.0 * shape.rows * m
        )
        return self._make_plan(
            shape, rows=shape.rows, width=m, scratch=scratch, knob="dim"
        )

    def _names(self, d: int) -> list[str]:
        return [f"__pn_c{i}" for i in range(d)]

    def _new_primitive(self, d: int) -> Any:
        from panelary.shape._project import SparseRandomProjection
        from panelary.shape._rsvd import RandomizedPCA

        cols = self._names(d)
        if self.method == "srp":
            return SparseRandomProjection(
                n_components=self.dim,
                density=self.density,
                seed=self.seed,
                prefix="__pn_o",
                columns=cols,
            )
        return RandomizedPCA(
            n_components=self.dim,
            standardize=self.method == "pca",
            seed=self.seed,
            prefix="__pn_o",
            columns=cols,
        )

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        self.feature_names_in_ = cols
        frame = panel.collect()
        d = array_width(frame.schema, cols)
        self.in_width_ = d
        if self.method == "none":
            self.out_width_ = d
            return
        prim = self._new_primitive(d)
        if self.method == "srp":
            prim.fit(_as_shape_panel(np.empty((0, d)), self._names(d)))
        else:
            M = matrix_from_columns(frame, cols)
            M = M[np.isfinite(M).all(axis=1)]
            if M.shape[0] > self.max_fit_rows:
                rng = np.random.default_rng([self.seed, 2])
                M = M[np.sort(rng.choice(M.shape[0], self.max_fit_rows, replace=False))]
            if M.shape[0] < 2:
                raise ValueError(
                    "EmbeddingCompressor.fit: fewer than 2 complete rows to fit on."
                )
            prim.fit(_as_shape_panel(M, self._names(d)))
        self.compressor_ = prim
        self.components_ = np.array(prim.components_, dtype=np.float64)
        self.mean_ = (
            None if getattr(prim, "mean_", None) is None else np.array(prim.mean_)
        )
        self.scale_ = (
            None if getattr(prim, "scale_", None) is None else np.array(prim.scale_)
        )
        self.out_width_ = int(self.components_.shape[0])

    def _restore_hook(self) -> None:
        """Rebuild the shape primitive from the serialised compression state."""
        if self.components_ is None:
            return
        d = int(np.asarray(self.components_).shape[1])
        self.in_width_ = d
        self.out_width_ = int(np.asarray(self.components_).shape[0])
        prim = self._new_primitive(d)
        if self.method == "srp":
            prim.fit(_as_shape_panel(np.empty((0, d)), self._names(d)))
        else:
            prim.feature_names_in_ = self._names(d)
            prim.components_ = np.asarray(self.components_, dtype=np.float64)
            prim.mean_ = np.asarray(self.mean_, dtype=np.float64)
            prim.scale_ = np.asarray(self.scale_, dtype=np.float64)
            prim.n_components_ = self.out_width_
            prim.component_names_ = [
                f"__pn_o_{i}" for i in range(1, self.out_width_ + 1)
            ]
            prim._fitted = True
        self.compressor_ = prim

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        cols = list(self.feature_names_in_)
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"EmbeddingCompressor.transform: column(s) {missing} not in panel."
            )
        frame = panel.collect()
        d = array_width(frame.schema, cols)
        if self.in_width_ and d != self.in_width_:
            raise ValueError(
                f"EmbeddingCompressor.transform: input width {d} differs from the fitted width {self.in_width_}."
            )
        self._enforce_budget(InputShape(rows=frame.height, width=d, entities=0))
        M = matrix_from_columns(frame, cols)
        valid = np.asarray(np.isfinite(M).all(axis=1), dtype=np.bool_)
        m = self._out_width(d)
        if self.method == "none":
            out = M
        else:
            out = np.full((frame.height, m), np.nan)
            names = self._names(d)
            for lo in range(0, frame.height, self.chunk_rows):
                hi = min(frame.height, lo + self.chunk_rows)
                res = self.compressor_.transform(
                    _as_shape_panel(M[lo:hi], names)
                ).collect()
                out_cols = [c for c in res.columns if c.startswith("__pn_o_")]
                out[lo:hi] = res.select(out_cols).to_numpy().astype(np.float64)
        base = (
            frame
            if self.keep_features
            else frame.select(panel.entity_col, panel.time_col)
        )
        res_frame = emit_embedding(
            base,
            self.name,
            out,
            valid,
            dtype=self.dtype,
            output=self.output,
            feature_names=[str(i) for i in range(m)],
        )
        return PanelFrame(
            res_frame, entity=panel.entity_col, time=panel.time_col, validate=False
        )
