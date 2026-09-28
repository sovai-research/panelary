"""Delay (Takens / Hankel) embedding: time -> a new ``lag`` axis. The cheapest LIFT.

Lag ``j`` of row ``t`` is ``x[t - j * dilation]`` within the entity, for
``j = 0 .. lags - 1``. It is causal **by construction** -- there is no parameter
that could make it read ahead -- which makes it the natural input to
:class:`~panelary.shape.RandomizedPCA`: ``Delay`` followed by a rank-``r``
projection is singular spectrum analysis (SSA), fit on training rows only.

Reference: Takens (1981), "Detecting strange attractors in turbulence"; Broomhead
& King (1986), "Extracting qualitative dynamics from experimental data" (the
delay/trajectory matrix behind SSA). Clean-room, numpy only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.shape._axes import (
    Axis,
    Flavour,
    InputShape,
    Intent,
    ShapeSpec,
    check_positive_int,
    register_shape_spec,
)
from panelary.shape._temporal import _TimeAxisTransform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from panelary.shape._window import PanelWindows, SortedPanel

__all__ = ["Delay"]


class Delay(_TimeAxisTransform):
    """Delay embedding: ``lags`` columns per input column, ``x[t], x[t-d], ...``.

    ``Delay(lags=16, dilation=4)`` reaches back ``(16 - 1) * 4 = 60``
    observations for 16 columns. There is no stride: one output row per input
    row is the contract of every trailing transform.

    Parameters
    ----------
    lags : int, default 8
        Number of delay columns ``L`` (lag 0 is the row's own value).
    dilation : int, default 1
        Spacing ``d`` between consecutive lags.
    prefix : str, default "lag"
        Columns are ``{column}__lag_{j * dilation}`` -- named by how far back
        they reach.
    keep_features, as_array, dtype, columns, max_bytes, entity, time
        As for :class:`~panelary.shape.PAA`. Output is float32 by default (a
        LIFT); pass ``dtype=pl.Float64`` to keep the input's precision.

    NaN policy
    ----------
    A lag reaching before the entity's first observation is NaN; a missing
    input observation is NaN in every column that copies it. Nothing is filled.

    Notes
    -----
    ``fit_is_empty = True``, ``panel_safe = True``, ``leakage_safe = True``.
    Trailing only: a whole-series delay embedding is not a thing.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.LIFT,
        axis=Axis.TIME,
        flavour=Flavour.TRAILING,
        width_rule="exact",
        invertible="exact",
        streaming="batch",
        cost_hint="O(n L)",
    )
    _default_prefix = "lag"

    def __init__(
        self,
        *,
        lags: int = 8,
        dilation: int = 1,
        prefix: str | None = None,
        keep_features: bool = False,
        as_array: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        self.lags = check_positive_int(lags, "lags", "Delay")
        self.dilation = check_positive_int(dilation, "dilation", "Delay")
        super().__init__(
            window=(self.lags - 1) * self.dilation + 1,
            flavour=Flavour.TRAILING,
            prefix=prefix,
            keep_features=keep_features,
            as_array=as_array,
            dtype=dtype,
            columns=columns,
            max_bytes=max_bytes,
            entity=entity,
            time=time,
        )

    def _width_per_column(self) -> int:
        return self.lags

    def _suffixes(self) -> list[str]:
        return [f"{self.prefix}_{j * self.dilation}" for j in range(self.lags)]

    def _output_dtype(self) -> Any:
        return pl.Float32

    def _scratch_bytes(self, shape: InputShape) -> float:
        reach = self._reach()
        n_pad = shape.rows + max(shape.entities, 1) * (reach - 1)
        return 8.0 * (n_pad * shape.width + shape.rows * shape.width * self.lags)

    def _trailing_kernel(
        self, pw: PanelWindows, sp: SortedPanel
    ) -> NDArray[np.float64]:
        buf_t = pw.buf_t  # (k, N_pad)
        out = np.empty((pw.k, self.lags, pw.n), dtype=np.float64)
        for j in range(self.lags):
            idx = pw.end - j * self.dilation
            for ci in range(pw.k):  # 1-D gathers: ~15x faster than buf_t[:, idx]
                out[ci, j] = buf_t[ci][idx]
        return out

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_frame

        comps: list[str] = []
        sources: list[str] = []
        for col, names in self._names():
            for j, name in enumerate(names):
                comps.append(name)
                lag = j * self.dilation
                sources.append(f"{col}[t-{lag}]" if lag else f"{col}[t]")
        return explain_frame(comps, sources, np.ones(len(comps)))


register_shape_spec(
    Delay,
    name="delay",
    params={"lags": int, "dilation": int},
    tier="B",
    safe_scope="rowwise",
    source="Takens (1981); Broomhead & King (1986); clean-room, Panelary",
)
