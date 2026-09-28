"""Shared machinery for time-axis transforms: the two flavours, once.

Every time-axis transform (:class:`~panelary.shape.PAA`,
:class:`~panelary.shape.Spectral`, :class:`~panelary.shape.Delay`) is a
:class:`_TimeAxisTransform`. The base owns everything flavour-related, so a
subclass only writes two numeric kernels:

* ``_trailing_kernel(pw, sp)`` -- one output row per input row, from the
  :class:`~panelary.shape._window.PanelWindows` buffer (the causal spine),
  returned **column-major** as ``(k, m, n)`` so each output column is one
  contiguous array (emitting ``n x k x m`` values from a row-major block was
  measured to cost more than the kernels themselves);
* ``_whole_kernel(tensor)`` -- one output row per entity, from an
  ``(E, T, k)`` positional tensor built under an explicit
  :class:`~panelary.shape._tensor.Ragged` policy, returned ``(E, k, m)``.

``flavour="trailing"`` is the default and ``leakage_safe``.
``flavour="whole_series"`` narrows the instance to ``leakage_safe = False`` --
so :meth:`~panelary.core.protocol.PanelTransformer._check_leakage` refuses it
across a train/test boundary -- and returns a frame keyed by ``entity`` alone,
so joining it back onto rows is a visible act, not an accident.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.shape._axes import (
    Flavour,
    InputShape,
    Plan,
    ShapeTransform,
    check_positive_int,
)
from panelary.shape._tensor import Ragged, build_sequences
from panelary.shape._window import PanelWindows, SortedPanel, sort_panel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["_TimeAxisTransform"]

#: Rows per chunk when a kernel must materialise whole windows (FFT).
DEFAULT_CHUNK_ROWS = 65_536


class _TimeAxisTransform(ShapeTransform):
    """Base for trailing / whole-series time-axis transforms (not public).

    Parameters
    ----------
    window : int, optional
        Trailing window length ``W`` (observations). Required for
        ``flavour="trailing"``; ignored for ``"whole_series"``.
    flavour : {"trailing", "whole_series"}, default "trailing"
    ragged : Ragged or str, default "refuse"
        ``whole_series`` only: policy for entities of different lengths.
    length : int, optional
        ``whole_series`` with ``ragged="truncate"``: observations kept per
        entity (the most recent).
    prefix : str
        Output columns are ``{column}__{prefix}_{j}``.
    keep_features : bool, default False
        Trailing only: append the new columns to the input frame instead of
        returning keys + new columns.
    as_array : bool, default False
        Emit one ``pl.Array`` column per input column (``{column}__{prefix}``)
        instead of loose columns.
    dtype : polars float dtype, optional
        Storage dtype of the output. ``None`` = the transform's default
        (float64 for COMPRESS, float32 for LIFT).
    columns, max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.
    """

    _default_prefix = "t"

    def __init__(
        self,
        *,
        window: int | None = None,
        flavour: Flavour | str = Flavour.TRAILING,
        ragged: Ragged | str = Ragged.REFUSE,
        length: int | None = None,
        prefix: str | None = None,
        keep_features: bool = False,
        as_array: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        self.flavour = Flavour(flavour)
        owner = type(self).__name__
        if self.flavour is Flavour.TRAILING:
            if window is None:
                raise ValueError(
                    f"{owner}: flavour='trailing' needs `window=` -- the number of "
                    "trailing observations each row may read. (Pass "
                    "flavour='whole_series' for one row per entity, which is not "
                    "leakage-safe.)"
                )
            self.window: int | None = check_positive_int(window, "window", owner)
        else:
            self.window = (
                None if window is None else check_positive_int(window, "window", owner)
            )
            if keep_features:
                raise ValueError(
                    f"{owner}: keep_features=True would broadcast a whole-series "
                    "summary onto every row -- a look-ahead by construction. The "
                    "whole_series flavour returns one row per entity on purpose."
                )
            self._narrow_contract(leakage_safe=False)
        self.ragged = Ragged(ragged)
        self.length = length
        self.prefix = prefix or self._default_prefix
        self.keep_features = bool(keep_features)
        self.as_array = bool(as_array)
        self.dtype = dtype

    # ------------------------------------------------------------------ #
    # Contract resolution
    # ------------------------------------------------------------------ #
    def _resolve_spec(self, spec: Any) -> Any:
        base = super()._resolve_spec(spec)
        return base.resolved(flavour=self.flavour)

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _width_per_column(self) -> int:
        """Output columns emitted per input column."""
        raise NotImplementedError

    def _reach(self) -> int:
        """Longest look-back (cells) the trailing kernel needs."""
        assert self.window is not None
        return self.window

    def _suffixes(self) -> list[str]:
        return [f"{self.prefix}_{j}" for j in range(self._width_per_column())]

    def _output_dtype(self) -> Any:
        return pl.Float64

    @abc.abstractmethod
    def _trailing_kernel(
        self, pw: PanelWindows, sp: SortedPanel
    ) -> NDArray[np.float64]:
        """``(k, m, n)`` column-major, in sorted row order."""
        raise NotImplementedError

    def _whole_kernel(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(E, T, k)`` -> ``(E, k, m)``."""
        raise NotImplementedError(f"{type(self).__name__} has no whole_series flavour.")

    def _whole_series_length(self, T: int) -> None:
        """Hook: validate the positional length for the whole-series kernel."""

    def _scratch_bytes(self, shape: InputShape) -> float:
        """Predicted peak scratch of the trailing kernel (float64 cells * 8)."""
        reach = self._reach()
        n_pad = shape.rows + max(shape.entities, 1) * (reach - 1)
        m = self._width_per_column()
        # buffer + two full-length working copies + the output block
        return 8.0 * (3 * n_pad * shape.width + shape.rows * shape.width * m)

    # ------------------------------------------------------------------ #
    # PanelTransformer hooks
    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        # Stateless: nothing but the resolved columns is recorded. (For
        # fit_is_empty classes, `panel` here has zero rows -- see ShapeTransform.fit.)
        self.feature_names_in_ = self._resolve_columns(panel)

    def _out_width(self, in_width: int) -> int:
        return in_width * self._width_per_column()

    def _plan(self, shape: InputShape) -> Plan:
        m = self._width_per_column()
        if self.flavour is Flavour.WHOLE_SERIES:
            T = (
                self.length
                if self.length is not None
                else max(1, shape.rows // max(shape.entities, 1))
            )
            scratch = (
                8.0 * shape.entities * T * shape.width * 3
                + 8.0 * shape.entities * shape.width * m
            )
            return self._make_plan(
                shape,
                rows=shape.entities,
                width=shape.width * m,
                scratch=scratch,
                knob="length",
                output="entities",
            )
        return self._make_plan(
            shape,
            rows=shape.rows,
            width=shape.width * m,
            scratch=self._scratch_bytes(shape),
            knob="window",
        )

    def _names(self) -> list[tuple[str, list[str]]]:
        return [
            (c, [f"{c}__{s}" for s in self._suffixes()]) for c in self.feature_names_in_
        ]

    def _emit_columns(self, block: NDArray[np.float64]) -> list[pl.Series]:
        """``block`` is column-major ``(k, m, rows)`` float64 -> output Series."""
        dtype = self.dtype if self.dtype is not None else self._output_dtype()
        series: list[pl.Series] = []
        m = block.shape[1]
        for ci, (col, names) in enumerate(self._names()):
            if self.as_array:
                arr = pl.Series(
                    f"{col}__{self.prefix}",
                    np.ascontiguousarray(block[ci].T),
                    dtype=pl.Array(pl.Float64, m),
                ).cast(pl.Array(dtype, m))
                series.append(arr)
            else:
                series.extend(
                    pl.Series(name, block[ci, j], dtype=pl.Float64).cast(dtype)
                    for j, name in enumerate(names)
                )
        return series

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.feature_names_in_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: column(s) {missing} not found. "
                f"Available columns: {panel.columns}."
            )
        if self.flavour is Flavour.WHOLE_SERIES:
            return self._transform_whole(panel)
        sp = sort_panel(panel, self.feature_names_in_, owner=type(self).__name__)
        self._enforce_budget(
            InputShape(
                rows=sp.n_rows,
                width=len(self.feature_names_in_),
                entities=int(sp.lengths.size),
            )
        )
        pw = PanelWindows(sp.values, sp.lengths, self._reach())
        block_sorted = self._trailing_kernel(pw, sp)  # (k, m, n), sorted rows
        cols = self._emit_columns(sp.unsort_last(block_sorted))
        del block_sorted
        keys = [panel.entity_col, panel.time_col]
        out = (
            sp.frame.with_columns(cols)
            if self.keep_features
            else sp.frame.select(keys).with_columns(cols)
        )
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    def _transform_whole(self, panel: PanelFrame) -> PanelFrame:
        pt = build_sequences(
            panel, self.feature_names_in_, ragged=self.ragged, length=self.length
        )
        if not hasattr(pt, "tensor"):
            raise ValueError(
                f"{type(self).__name__}: ragged='native' hands unpadded per-entity "
                "matrices through, which a fixed-width whole-series summary cannot "
                "consume. Use 'refuse', 'truncate' (with `length=`) or 'pad'."
            )
        tensor = pt.tensor  # type: ignore[union-attr]
        self._enforce_budget(
            InputShape(
                rows=int(tensor.shape[0] * tensor.shape[1]),
                width=int(tensor.shape[2]),
                entities=int(tensor.shape[0]),
            )
        )
        self._whole_series_length(int(tensor.shape[1]))
        block = self._whole_kernel(tensor)  # (E, k, m)
        cols = self._emit_columns(np.ascontiguousarray(np.moveaxis(block, 0, -1)))
        ent = panel.entity_col
        out = pt.entities.to_frame(ent).with_columns(cols)
        # Keyed by entity ALONE: this is deliberately not a PanelFrame-shaped
        # (entity, time) result. A PanelFrame needs a time column, so we carry a
        # constant "whole_series" marker to make any join onto rows fail loudly.
        out = out.with_columns(pl.lit("whole_series").alias(panel.time_col))
        return PanelFrame(
            out.select(
                [
                    ent,
                    panel.time_col,
                    *[c for c in out.columns if c not in (ent, panel.time_col)],
                ]
            ),
            entity=ent,
            time=panel.time_col,
            validate=False,
        )
