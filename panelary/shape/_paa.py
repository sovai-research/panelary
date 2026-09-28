"""Piecewise aggregate approximation (PAA) -- the exemplar of the two-flavour rule.

Keogh, Chakrabarti, Pazzani & Mehrotra (2001), "Dimensionality reduction for
fast similarity search in large time series databases", *Knowledge and
Information Systems* 3(3). Clean-room implementation from the paper.

Read this file first when writing any other time-axis transform.

The PAA in the external report pools each entity's **entire** series into
``S`` segments, so segment 0 of that output is a function of the whole series,
future included -- textbook look-ahead when used as a row feature. Here the
default flavour is **trailing**: row ``t`` gets the ``S`` segment summaries of
*its own* trailing ``W`` observations, and nothing else.

Implementation note (batching). The plan's sketch is
``trailing_windows(x, W).reshape(n, S, W // S).mean(axis=2)``. This computes the
same numbers without materialising ``(n, W)``: segment ``j`` of row ``t``'s
window is the length-``L = W // S`` segment ending ``(S - 1 - j) * L`` cells
behind ``t``, so one length-``L`` rolling statistic over the padded panel buffer
plus ``S`` gathers gives every row's ``S`` segments -- ``O(n L + n S)`` work
instead of ``O(n W)``, no per-entity loop, and bit-identical under truncation
of the panel (each cell is a fixed sequence of elementwise operations on its own
``L`` inputs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.shape._axes import (
    Axis,
    Flavour,
    Intent,
    ShapeSpec,
    check_positive_int,
    register_shape_spec,
)
from panelary.shape._temporal import _TimeAxisTransform
from panelary.shape._window import PanelWindows, SortedPanel, sort_panel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from panelary.shape._tensor import Ragged

__all__ = ["PAA"]

_POOLS = ("mean", "max", "min", "std", "last")


class PAA(_TimeAxisTransform):
    """Piecewise aggregate approximation along time, trailing by default.

    ``flavour="trailing"`` (default): each row's trailing ``window`` observations
    are cut into ``segments`` equal pieces and each piece is pooled, giving
    ``segments`` columns per input column **per row**. ``flavour="whole_series"``:
    each entity's whole (positional) series is cut into ``segments`` pieces,
    giving ``segments`` columns per input column **per entity** --
    ``leakage_safe = False``, for clustering / describing a fixed sample.

    Parameters
    ----------
    window : int, optional
        Trailing window ``W``. Required for the trailing flavour. ``W %
        segments == 0`` is required.
    segments : int, default 8
        Number of segments ``S`` (output columns per input column).
    pool : {"mean", "max", "min", "std", "last"}, default "mean"
        Segment statistic. On a returns panel ``std`` is per-segment realised
        volatility (population, ``ddof=0``) and ``last`` a subsample.
    flavour : {"trailing", "whole_series"}, default "trailing"
    ragged, length
        ``whole_series`` only: see :func:`~panelary.shape.build_sequences`. The
        (kept) series length must be a multiple of ``segments``.
    prefix : str, default "paa"
        Columns are ``{column}__paa_0 .. {column}__paa_{S-1}``; segment 0 is
        the oldest.
    keep_features, as_array, dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._temporal._TimeAxisTransform`. Output is
        float64 by default (a COMPRESS result).

    NaN policy
    ----------
    A segment containing a missing observation (null/NaN, or a position before
    the entity's first observation) is NaN. Rows with fewer than ``W``
    observations are therefore NaN in their oldest segments -- never computed
    on a short window as if it were full.

    Notes
    -----
    ``panel_safe = True`` (windows never cross entities), ``leakage_safe = True``
    for the trailing flavour, and ``fit_is_empty = True``: ``fit`` records the
    column names and nothing else -- it is handed a zero-row frame.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.TIME,
        flavour=Flavour.TRAILING,
        width_rule="exact",
        invertible="approximate",
        streaming="batch",
        cost_hint="O(n W)",
    )
    _default_prefix = "paa"

    def __init__(
        self,
        *,
        window: int | None = None,
        segments: int = 8,
        pool: str = "mean",
        flavour: Flavour | str = Flavour.TRAILING,
        ragged: Ragged | str = "refuse",
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
        super().__init__(
            window=window,
            flavour=flavour,
            ragged=ragged,
            length=length,
            prefix=prefix,
            keep_features=keep_features,
            as_array=as_array,
            dtype=dtype,
            columns=columns,
            max_bytes=max_bytes,
            entity=entity,
            time=time,
        )
        self.segments = check_positive_int(segments, "segments", "PAA")
        if pool not in _POOLS:
            raise ValueError(
                f"PAA: `pool` must be one of {list(_POOLS)}, got {pool!r}."
            )
        self.pool = pool
        if self.window is not None and self.window % self.segments:
            raise ValueError(
                f"PAA: window={self.window} is not a multiple of segments="
                f"{self.segments}. Change `window` or `segments` so W % S == 0 "
                "-- segments are never ragged-split silently."
            )

    # ------------------------------------------------------------------ #
    def _width_per_column(self) -> int:
        return self.segments

    def _seg_len(self) -> int:
        assert self.window is not None
        return self.window // self.segments

    def _trailing_kernel(
        self, pw: PanelWindows, sp: SortedPanel
    ) -> NDArray[np.float64]:
        L, S = self._seg_len(), self.segments
        seg_t = np.ascontiguousarray(pw.segment_stat(L, self.pool).T)  # (k, N_pad)
        out = np.empty((pw.k, S, pw.n), dtype=np.float64)
        for j in range(S):
            idx = pw.end - (S - 1 - j) * L
            for ci in range(pw.k):  # 1-D gathers: ~15x faster than seg_t[:, idx]
                out[ci, j] = seg_t[ci][idx]
        return out

    def _whole_series_length(self, T: int) -> None:
        if T % self.segments:
            raise ValueError(
                f"PAA(flavour='whole_series'): each entity has {T} kept observations, "
                f"not a multiple of segments={self.segments}. Pass ragged='truncate' "
                f"with `length=` a multiple of {self.segments}."
            )

    def _whole_kernel(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        E, T, k = tensor.shape
        S = self.segments
        L = T // S
        blocks = tensor.reshape(E, S, L, k)
        if self.pool == "mean":
            res = blocks.mean(axis=2)
        elif self.pool == "max":
            res = blocks.max(axis=2)
        elif self.pool == "min":
            res = blocks.min(axis=2)
        elif self.pool == "std":
            res = blocks.std(axis=2)
        else:
            res = blocks[:, :, -1, :]
        return np.moveaxis(res, 1, 2)  # (E, k, S)

    # ------------------------------------------------------------------ #
    # Inverse & error
    # ------------------------------------------------------------------ #
    def inverse_transform(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        """Step-function reconstruction: each segment value repeated ``W // S`` times.

        Parameters
        ----------
        Z : PanelFrame | polars.DataFrame | numpy.ndarray
            This transform's output (loose columns), or an ``(n, k, S)`` array.

        Returns
        -------
        numpy.ndarray
            ``(n, W, k)`` reconstructed windows (``(n, T, k)`` for whole-series,
            with ``T = S * L`` inferred from ``length``), oldest observation
            first, in ``Z``'s row order.
        """
        self._check_fitted("inverse_transform")
        if self.pool not in ("mean", "last"):
            raise ValueError(
                f"PAA: pool={self.pool!r} has no reconstruction; only 'mean' (the "
                "PAA step function) and 'last' (a sample-and-hold) are invertible."
            )
        block = self._as_block(Z)  # (n, k, S)
        L = (
            self._seg_len()
            if self.window is not None
            else int(self.length or 1) // self.segments
        )
        rec = np.repeat(block, L, axis=2)  # (n, k, S*L)
        return np.moveaxis(rec, 1, 2)

    def _as_block(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        if isinstance(Z, np.ndarray):
            return np.asarray(Z, dtype=np.float64)
        frame = Z.collect() if isinstance(Z, PanelFrame) else Z
        k, S = len(self.feature_names_in_), self.segments
        block = np.empty((frame.height, k, S), dtype=np.float64)
        for ci, (_col, names) in enumerate(self._names()):
            block[:, ci, :] = frame.select(names).to_numpy().astype(np.float64)
        return block

    def reconstruction_error(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> float:
        """Relative Frobenius error of the step reconstruction of each full window.

        Trailing flavour only; rows without a full window are skipped.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
        entity, time : str, optional

        Returns
        -------
        float
            ``||windows - step(PAA(windows))||_F / ||windows||_F``.
        """
        self._check_fitted("reconstruction_error")
        if self.flavour is not Flavour.TRAILING or self.window is None:
            raise ValueError(
                "PAA.reconstruction_error is defined for the trailing flavour."
            )
        panel = self._as_panel(
            X, method="reconstruction_error", entity=entity, time=time
        )
        sp = sort_panel(panel, self.feature_names_in_, owner="PAA")
        pw = PanelWindows(sp.values, sp.lengths, self.window)
        full = np.flatnonzero(pw.position >= self.window - 1)
        kern = self._trailing_kernel(pw, sp)  # (k, S, n)
        # Accumulate over row chunks: materialising every (W, k) window at
        # once is n * W * k floats, which is the allocation to avoid.
        chunk = max(1, 8_000_000 // max(1, self.window * pw.k))
        num = den = 0.0
        for lo in range(0, full.size, chunk):
            idx = full[lo : lo + chunk]
            wins = pw.windows(self.window, idx)  # (c, W, k)
            rec = self.inverse_transform(np.moveaxis(kern[..., idx], -1, 0))
            ok = np.isfinite(wins) & np.isfinite(rec)
            num += float(np.sum((wins[ok] - rec[ok]) ** 2))
            den += float(np.sum(wins[ok] ** 2))
        if den == 0.0:
            return 0.0 if num == 0.0 else float("inf")
        return float(np.sqrt(num) / np.sqrt(den))

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_frame

        assert self.window is not None or self.flavour is Flavour.WHOLE_SERIES
        W = self.window if self.window is not None else self.segments
        L = max(1, W // self.segments)
        comps, sources, loads = [], [], []
        for col, names in self._names():
            for j, name in enumerate(names):
                for lag in range(W):
                    # lag 0 = the row itself; segment j covers the oldest-first slice.
                    pos = W - 1 - lag
                    if pos // L == j:
                        comps.append(name)
                        sources.append(f"{col}[t-{lag}]" if lag else f"{col}[t]")
                        loads.append(1.0 / L if self.pool == "mean" else 1.0)
        return explain_frame(comps, sources, np.asarray(loads, dtype=np.float64))


register_shape_spec(
    PAA,
    name="paa",
    params={"window": int, "segments": int, "pool": str, "flavour": str},
    tier="B",
    safe_scope="rowwise",
    source="Keogh, Chakrabarti, Pazzani & Mehrotra (2001), KAIS 3(3); clean-room, Panelary",
)
