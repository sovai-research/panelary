"""TensorSketch: a polynomial-kernel feature map from CountSketch and the FFT.

TensorSketch (Pham & Pagh, 2013, "Fast and scalable polynomial kernels via
explicit feature maps", KDD) approximates the degree-``p`` polynomial kernel
``(<x, y> + c)^p`` with a ``D``-dimensional map whose inner products are
unbiased for it:

    TS(x) = irfft( rfft(C_1 x) * rfft(C_2 x) * ... * rfft(C_p x) ),

where ``C_1 .. C_p`` are independent CountSketches ``R^d -> R^D`` (one hash
and one sign per input coordinate) and ``sqrt(c)`` is appended to ``x`` for
the inhomogeneous kernel. The product of FFTs is the circular convolution of
the sketches, which is the CountSketch of the ``p``-fold tensor product
``x (x) ... (x) x`` -- computed in ``O(p (d + D log D))`` without ever
forming it.

Division of labour (shape build contract section 7.1): the CountSketch is
:class:`panelary.shape._sketch.CountSketch`, used through its public
``fit`` / ``transform``; this module owns only the polynomial-kernel
composition (the ``sqrt(c)`` coordinate, the per-degree seeds, the FFT product).

Stateless given the seed: ``fit`` reads the schema only (``fit_is_empty =
True``). No scaling is fitted -- feed standardised inputs (e.g. the direct-link
block of :class:`~panelary.embed.RandomFourierFeatures`, or a cross-sectional
rank) if the columns are on different scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from panelary.core.panel_frame import PanelFrame
from panelary.embed._compress import _as_shape_panel
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

__all__ = ["TensorSketch"]


class TensorSketch(_EmbedTransform):
    """Degree-``p`` polynomial-kernel sketch of each row -- stateless given the seed.

    Parameters
    ----------
    n_components : int, default 256
        Sketch width ``D``.
    degree : int, default 2
        Polynomial degree ``p``.
    coef0 : float, default 1.0
        ``c`` in ``(<x, y> + c)^p``; ``0`` gives the homogeneous kernel.
    seed : int, default 0
        Seeds the ``p`` CountSketches (sketch ``j`` uses a seed derived from
        ``(seed, j)``).
    columns : sequence of str, optional
        Numeric and/or numeric ``pl.Array`` columns.
    name : str, default "tensor_sketch"
    output : {"array", "columns"}, default "array"
    dtype : {"float32", "float16", "float64"}, default "float32"
    keep_features : bool, default False
    chunk_rows : int, default 16384
    max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.

    Attributes
    ----------
    sketches_ : list of CountSketch
        The ``p`` fitted :class:`panelary.shape._sketch.CountSketch` objects.
    in_width_ : int
        Input width ``d`` (before the ``sqrt(c)`` coordinate).

    Notes
    -----
    ``E[<TS(x), TS(y)>] = (<x, y> + c)^p``; the variance falls as ``1 / D``.
    A row with any missing input is emitted as null.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.LIFT,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="none",
        streaming="mergeable",
        cost_hint="O(n p (d + D log D))",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        "n_components",
        "degree",
        "coef0",
        "seed",
        "name",
        "output",
        "dtype",
        "keep_features",
        "chunk_rows",
        "max_bytes",
    )
    _state_arrays: ClassVar[tuple[str, ...]] = ("in_width_",)

    def __init__(
        self,
        *,
        n_components: int = 256,
        degree: int = 2,
        coef0: float = 1.0,
        seed: int = 0,
        columns: Sequence[str] | str | None = None,
        name: str = "tensor_sketch",
        output: str = "array",
        dtype: str = "float32",
        keep_features: bool = False,
        chunk_rows: int = 16384,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        owner = "TensorSketch"
        self.n_components = check_pos_int(n_components, "n_components", owner)
        self.degree = check_pos_int(degree, "degree", owner)
        c0 = float(coef0)
        if not np.isfinite(c0) or c0 < 0:
            raise ValueError(f"{owner}: `coef0` must be >= 0, got {coef0!r}.")
        self.coef0 = c0
        self.seed = check_seed(seed, owner)
        self.name = name
        self.output = check_choice(output, "output", ("array", "columns"), owner)
        self.dtype = check_choice(
            dtype, "dtype", ("float16", "float32", "float64"), owner
        )
        self.keep_features = bool(keep_features)
        self.chunk_rows = check_pos_int(chunk_rows, "chunk_rows", owner)
        self.in_width_: Any = 0
        self.sketches_: list[Any] = []

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        return resolve_matrix_columns(panel, self.columns, "TensorSketch")

    def _out_width(self, in_width: int) -> int:
        return self.n_components

    def _plan(self, shape: InputShape) -> Plan:
        chunk = min(shape.rows, self.chunk_rows)
        D = self.n_components
        scratch = (
            8.0 * chunk * (shape.width + 1 + (self.degree + 2) * D)
            + 8.0 * shape.rows * D
        )
        return self._make_plan(
            shape, rows=shape.rows, width=D, scratch=scratch, knob="n_components"
        )

    def _sketch_names(self, d_aug: int) -> list[str]:
        return [f"__pn_ts{i}" for i in range(d_aug)]

    def _build_sketches(self) -> None:
        from panelary.shape._sketch import CountSketch

        d_aug = int(self.in_width_) + (1 if self.coef0 > 0 else 0)
        names = self._sketch_names(d_aug)
        empty = _as_shape_panel(np.empty((0, d_aug)), names)
        self.sketches_ = []
        for j in range(self.degree):
            sub_seed = int(np.random.default_rng([self.seed, j]).integers(0, 2**31 - 1))
            cs = CountSketch(
                n_components=self.n_components,
                seed=sub_seed,
                prefix=f"__pn_s{j}",
                columns=names,
            )
            self.sketches_.append(cs.fit(empty))

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        self.feature_names_in_ = cols
        self.in_width_ = array_width(panel.schema, cols)
        self._build_sketches()

    def _restore_hook(self) -> None:
        self.in_width_ = int(np.asarray(self.in_width_))
        self._build_sketches()

    def _map(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n, d)`` finite rows -> ``(n, D)`` sketches."""
        if self.coef0 > 0:
            M = np.hstack([M, np.full((M.shape[0], 1), np.sqrt(self.coef0))])
        panel = _as_shape_panel(M, self._sketch_names(M.shape[1]))
        D = self.n_components
        prod: NDArray[np.complex128] | None = None
        for j, cs in enumerate(self.sketches_):
            res = cs.transform(panel).collect()
            S = res.select(
                [c for c in res.columns if c.startswith(f"__pn_s{j}_")]
            ).to_numpy()
            F = np.fft.rfft(np.asarray(S, dtype=np.float64), axis=1)
            prod = F if prod is None else prod * F
        assert prod is not None
        return np.fft.irfft(prod, n=D, axis=1)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        cols = list(self.feature_names_in_)
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"TensorSketch.transform: column(s) {missing} not in panel."
            )
        frame = panel.collect()
        d = array_width(frame.schema, cols)
        if d != int(self.in_width_):
            raise ValueError(
                f"TensorSketch.transform: input width {d} differs from the fitted {self.in_width_}."
            )
        self._enforce_budget(InputShape(rows=frame.height, width=d, entities=0))
        M = matrix_from_columns(frame, cols)
        valid = np.isfinite(M).all(axis=1)
        out = np.full((frame.height, self.n_components), np.nan)
        rows = np.flatnonzero(valid)
        for lo in range(0, rows.size, self.chunk_rows):
            idx = rows[lo : lo + self.chunk_rows]
            out[idx] = self._map(M[idx])
        base = (
            frame
            if self.keep_features
            else frame.select(panel.entity_col, panel.time_col)
        )
        res = emit_embedding(
            base,
            self.name,
            out,
            valid,
            dtype=self.dtype,
            output=self.output,
            feature_names=[str(i) for i in range(self.n_components)],
        )
        return PanelFrame(
            res, entity=panel.entity_col, time=panel.time_col, validate=False
        )
