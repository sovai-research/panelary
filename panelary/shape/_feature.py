"""Shared machinery for feature-axis transforms (a row-wise map, no ``.over()``).

A feature-axis transform maps each row's ``d`` values to ``m`` new values. It
never mixes rows at transform time; whatever it learns (a rotation, a column
subset, a sketch) comes from the rows handed to ``fit`` -- the training fold --
and nothing else. The subclasses implement ``_fit_matrix`` and ``_map``.

NaN policy (all feature-axis transforms)
----------------------------------------
* ``fit`` uses only rows whose selected columns are all finite (the count is
  recorded as ``n_rows_fit_``); nothing is imputed.
* ``transform`` emits NaN in every output column of a row with any missing
  input. Nothing is filled.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.shape._axes import InputShape, Plan, ShapeTransform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["_FeatureAxisTransform", "fit_scaler"]


def fit_scaler(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Column mean and population std (``ddof=0``, as ``StandardScaler``); zero std -> 1."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, np.where(std == 0.0, 1.0, std)


class _FeatureAxisTransform(ShapeTransform):
    """Base for row-wise feature-axis transforms (not public).

    Parameters
    ----------
    prefix : str
        Output columns are ``{prefix}_1 .. {prefix}_m`` (``ColumnSubset`` /
        ``CUR`` emit the selected columns under their own names instead).
    keep_features : bool, default False
        Append the output columns to the input frame instead of returning keys
        plus outputs.
    dtype : polars float dtype, default ``pl.Float64``
        Storage dtype (COMPRESS results are float64 by default).
    columns, max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.
    """

    _default_prefix = "c"
    #: The constructor parameter a :class:`~panelary.shape.Plan` tells the user to lower.
    _knob: str = "n_components"

    def __init__(
        self,
        *,
        prefix: str | None = None,
        keep_features: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        self.prefix = prefix or self._default_prefix
        self.keep_features = bool(keep_features)
        self.dtype = dtype
        self.n_rows_fit_: int = 0

    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        """Learn from the finite ``(n, d)`` training matrix."""

    @abc.abstractmethod
    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(n, d)`` finite rows -> ``(n, m)``."""

    def _output_names(self) -> list[str]:
        m = self._out_width(len(self.feature_names_in_)) or 0
        return [f"{self.prefix}_{i}" for i in range(1, m + 1)]

    def _plan(self, shape: InputShape) -> Plan:
        m = self._out_width(shape.width) or shape.width
        return self._make_plan(
            shape,
            rows=shape.rows,
            width=m,
            scratch=self._scratch_bytes(shape, m),
            knob=self._knob,
        )

    def _scratch_bytes(self, shape: InputShape, m: int) -> float:
        return 8.0 * shape.rows * (shape.width + 2 * m)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _matrix(
        frame: pl.DataFrame | pl.LazyFrame, cols: list[str]
    ) -> NDArray[np.float64]:
        lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        X = lf.select([pl.col(c).cast(pl.Float64) for c in cols]).collect().to_numpy()
        return np.asarray(X, dtype=np.float64)

    def _fit(self, panel: PanelFrame) -> None:
        self.feature_names_in_ = self._resolve_columns(panel)
        X = self._matrix(panel.lazy(), self.feature_names_in_)
        if not self.fit_is_empty:
            X = X[np.isfinite(X).all(axis=1)]
            self._enforce_budget(
                InputShape(rows=X.shape[0], width=X.shape[1], entities=0)
            )
            if X.shape[0] == 0:
                raise ValueError(
                    f"{type(self).__name__}.fit: no row has all of "
                    f"{self.feature_names_in_} observed; nothing to fit on."
                )
        self.n_rows_fit_ = int(X.shape[0])
        self._fit_matrix(X)

    def _apply(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map rows, NaN-ing every output of a row with a missing input."""
        ok = np.isfinite(X).all(axis=1)
        m = len(self._output_names())
        out = np.full((X.shape[0], m), np.nan, dtype=np.float64)
        if ok.any():
            out[ok] = self._map(X[ok] if not ok.all() else X)
        return out

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.feature_names_in_ if c not in panel]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: column(s) {missing} not found. "
                f"Available columns: {panel.columns}."
            )
        full = panel.collect()
        self._enforce_budget(
            InputShape(rows=full.height, width=len(self.feature_names_in_), entities=0)
        )
        X = self._matrix(full, self.feature_names_in_)
        out = self._apply(X)
        dtype = self.dtype if self.dtype is not None else pl.Float64
        cols = [
            pl.Series(name, out[:, i], dtype=pl.Float64).cast(dtype)
            for i, name in enumerate(self._output_names())
        ]
        keys = [panel.entity_col, panel.time_col]
        if self.keep_features:
            res = full.with_columns(cols)
        else:
            res = full.select(keys).with_columns(cols)
        return PanelFrame(
            res, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    def _out_width(self, in_width: int) -> int | None:
        return None
