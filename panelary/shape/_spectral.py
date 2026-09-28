"""Spectral transform along time: the DFT of each trailing window.

Frequency-domain summaries of each row's own trailing window, via
``numpy.fft.rfft`` (compiled pocketfft; no scipy). Three modes:

``"truncate"`` (COMPRESS)
    Magnitudes of the lowest ``k`` DFT bins (bin 0 = the window mean),
    normalised by ``W`` so they are amplitudes per observation. With
    ``phase=True`` the phases are emitted too (``2k`` columns) and the transform
    is approximately invertible via ``irfft``.
``"band"`` (LIFT)
    Share of the window's non-DC energy falling in each of ``k`` log-spaced
    frequency bands. Real-valued, scale-invariant (a share), and on a returns
    panel the one most likely to be useful.
``"power"`` (LIFT)
    Total non-DC power -- by Parseval, the window's population variance.

Magnitude-only by default: phase on a returns window is close to noise, and
complex columns do not survive a Polars round-trip cleanly.

Reference: Agrawal, Faloutsos & Swami (1993), "Efficient similarity search in
sequence databases" (DFT truncation as a time-series reduction); the band shares
are the standard log-spaced spectral-energy features. Clean-room, numpy only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.shape._axes import (
    Axis,
    Flavour,
    InputShape,
    Intent,
    ShapeSpec,
    check_positive_int,
    register_shape_spec,
)
from panelary.shape._temporal import DEFAULT_CHUNK_ROWS, _TimeAxisTransform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from panelary.shape._tensor import Ragged
    from panelary.shape._window import PanelWindows, SortedPanel

__all__ = ["Spectral", "log_bands"]

_MODES = ("truncate", "band", "power")


def log_bands(window: int, k: int) -> NDArray[np.int64]:
    """``k + 1`` integer bin edges splitting bins ``1 .. W // 2`` into log-spaced bands.

    Band ``b`` covers rfft bins ``edges[b] <= f < edges[b + 1]``. Raises if the
    window is too short to give ``k`` non-empty bands.
    """
    top = window // 2 + 1  # exclusive upper edge: bins 1..W//2
    if top - 1 < k:
        raise ValueError(
            f"Spectral(mode='band'): window={window} has only {top - 1} non-DC "
            f"frequency bins, fewer than k={k} bands. Lower `k` or lengthen `window`."
        )
    edges = np.unique(np.round(np.geomspace(1, top, k + 1)).astype(np.int64))
    # Guarantee k non-empty bands: widen from the bottom when rounding collapsed edges.
    if edges.size < k + 1:
        edges = np.arange(1, k + 2, dtype=np.int64)
        edges[-1] = top
        edges = np.maximum.accumulate(edges)
    return edges


class Spectral(_TimeAxisTransform):
    """DFT features of each row's trailing window (or of each entity's whole series).

    Parameters
    ----------
    window : int, optional
        Trailing window ``W``. Required for the trailing flavour.
    k : int, default 8
        ``truncate``: number of lowest bins kept (``<= W // 2 + 1``). ``band``:
        number of log-spaced bands. Ignored by ``power``.
    mode : {"truncate", "band", "power"}, default "truncate"
    phase : bool, default False
        ``truncate`` only: also emit the ``k`` phases (radians).
    flavour : {"trailing", "whole_series"}, default "trailing"
    ragged, length
        ``whole_series`` only (see :func:`~panelary.shape.build_sequences`);
        the DFT length is then each entity's kept series length.
    chunk_rows : int, default 65536
        Rows per batched FFT; bounds the ``(chunk_rows, W)`` scratch.
    prefix : str, default "spec"
    keep_features, as_array, dtype, columns, max_bytes, entity, time
        As for :class:`~panelary.shape.PAA`. Output is float64 for ``truncate``
        (COMPRESS) and float32 for ``band`` / ``power`` (LIFT) unless ``dtype``
        says otherwise.

    NaN policy
    ----------
    A window containing any missing observation (or a position before the
    entity's first) yields NaN for every column of that row. A window with zero
    non-DC energy yields NaN band shares (a share of nothing is undefined).

    Notes
    -----
    ``fit_is_empty = True``; ``panel_safe = True``; ``leakage_safe = True`` for
    the trailing flavour, ``False`` for ``whole_series``.
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
        cost_hint="O(n W log W)",
    )
    _default_prefix = "spec"

    def __init__(
        self,
        *,
        window: int | None = None,
        k: int = 8,
        mode: str = "truncate",
        phase: bool = False,
        flavour: Flavour | str = Flavour.TRAILING,
        ragged: Ragged | str = "refuse",
        length: int | None = None,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        prefix: str | None = None,
        keep_features: bool = False,
        as_array: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(
                f"Spectral: `mode` must be one of {list(_MODES)}, got {mode!r}."
            )
        self.mode = mode
        self.k = check_positive_int(k, "k", "Spectral")
        self.phase = bool(phase)
        if self.phase and mode != "truncate":
            raise ValueError(
                "Spectral: phase=True is only meaningful for mode='truncate'."
            )
        self.chunk_rows = check_positive_int(chunk_rows, "chunk_rows", "Spectral")
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
        if self.window is not None:
            self._validate_length(self.window)

    def _validate_length(self, W: int) -> None:
        if self.mode == "truncate" and self.k > W // 2 + 1:
            raise ValueError(
                f"Spectral: k={self.k} exceeds the {W // 2 + 1} rfft bins of a "
                f"length-{W} window."
            )
        if self.mode == "band":
            log_bands(W, self.k)

    def _whole_series_length(self, T: int) -> None:
        self._validate_length(T)

    def _resolve_spec(self, spec: Any) -> Any:
        base = super()._resolve_spec(spec)
        if self.mode == "truncate":
            return base.resolved(invertible="approximate" if self.phase else "none")
        return base.resolved(intent=Intent.LIFT, invertible="none")

    def _width_per_column(self) -> int:
        if self.mode == "power":
            return 1
        if self.mode == "band":
            return self.k
        return 2 * self.k if self.phase else self.k

    def _suffixes(self) -> list[str]:
        p = self.prefix
        if self.mode == "power":
            return [f"{p}_power"]
        if self.mode == "band":
            return [f"{p}_band_{j}" for j in range(self.k)]
        mags = [f"{p}_{j}" for j in range(self.k)]
        if self.phase:
            return mags + [f"{p}_phase_{j}" for j in range(self.k)]
        return mags

    def _output_dtype(self) -> Any:
        return pl.Float64 if self.mode == "truncate" else pl.Float32

    def _scratch_bytes(self, shape: InputShape) -> float:
        assert self.window is not None
        W = self.window
        n_pad = shape.rows + max(shape.entities, 1) * (W - 1)
        c = min(self.chunk_rows, max(shape.rows, 1))
        # buffer + one chunk of windows + its complex spectrum + the output
        chunk = 8.0 * c * shape.width * (W + 2 * (W // 2 + 1))
        return (
            8.0 * n_pad * shape.width
            + chunk
            + 8.0 * shape.rows * shape.width * self._width_per_column()
        )

    # ------------------------------------------------------------------ #
    # The kernel: windows (c, k, W) -> features (c, k, m)
    # ------------------------------------------------------------------ #
    def _features(self, wins: NDArray[np.float64]) -> NDArray[np.float64]:
        W = wins.shape[-1]
        spec = np.fft.rfft(wins, axis=-1)  # (c, k, F), complex, contiguous FFTs
        if self.mode == "truncate":
            head = spec[..., : self.k] / W
            mag = np.abs(head)
            if not self.phase:
                return mag
            return np.concatenate([mag, np.angle(head)], axis=-1)  # (c, k, 2K)
        energy = spec.real**2 + spec.imag**2  # (c, k, F)
        weight = np.full(energy.shape[-1], 2.0)
        weight[0] = 1.0
        if W % 2 == 0:
            weight[-1] = 1.0
        energy = energy * weight
        non_dc = energy[..., 1:].sum(axis=-1)  # (c, k)
        if self.mode == "power":
            return (non_dc / (W * W))[..., None]
        edges = log_bands(W, self.k)
        bands = np.stack(
            [
                energy[..., lo:hi].sum(axis=-1)
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            ],
            axis=-1,
        )  # (c, k, bands)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = bands / non_dc[..., None]
        share[~(non_dc > 0.0)] = np.nan
        return share

    def _trailing_kernel(
        self, pw: PanelWindows, sp: SortedPanel
    ) -> NDArray[np.float64]:
        assert self.window is not None
        out = np.empty((pw.k, self._width_per_column(), pw.n), dtype=np.float64)
        step = self.chunk_rows
        for lo in range(0, pw.n, step):
            sl = slice(lo, min(pw.n, lo + step))
            feats = self._features(pw.windows_last(self.window, sl))  # (c, k, m)
            out[:, :, sl] = np.moveaxis(feats, 0, -1)
        return out

    def _whole_kernel(self, tensor: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._features(np.ascontiguousarray(np.moveaxis(tensor, 1, 2)))

    # ------------------------------------------------------------------ #
    def inverse_transform(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        """Approximate window reconstruction via ``irfft`` of the kept bins.

        Only for ``mode="truncate", phase=True`` (magnitudes alone do not
        determine a signal).

        Returns
        -------
        numpy.ndarray
            ``(n, W, k)`` low-pass reconstructed windows, in ``Z``'s row order.
        """
        self._check_fitted("inverse_transform")
        if self.mode != "truncate" or not self.phase or self.window is None:
            raise ValueError(
                "Spectral.inverse_transform needs mode='truncate', phase=True and "
                "the trailing flavour: magnitudes alone do not determine a signal."
            )
        if isinstance(Z, np.ndarray):
            block = np.asarray(Z, dtype=np.float64)
        else:
            frame = Z.collect() if isinstance(Z, PanelFrame) else Z
            block = np.stack(
                [
                    frame.select(names).to_numpy().astype(np.float64)
                    for _c, names in self._names()
                ],
                axis=1,
            )  # (n, k, 2K)
        W, K = self.window, self.k
        mag, ph = block[:, :, :K], block[:, :, K:]
        spec = np.zeros(
            (block.shape[0], block.shape[1], W // 2 + 1), dtype=np.complex128
        )
        spec[:, :, :K] = mag * W * np.exp(1j * ph)
        rec = np.fft.irfft(spec, n=W, axis=2)  # (n, k, W)
        return np.moveaxis(rec, 1, 2)

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_frame

        W = self.window if self.window is not None else (self.length or 0)
        comps: list[str] = []
        sources: list[str] = []
        for col, names in self._names():
            for j, name in enumerate(names):
                comps.append(name)
                span = (
                    f"{col}[t-{W - 1}..t]"
                    if self.window is not None
                    else f"{col}[whole series]"
                )
                if self.mode == "power":
                    sources.append(f"{span}: total non-DC power")
                elif self.mode == "band" and W:
                    e = log_bands(W, self.k)
                    sources.append(f"{span}: energy share, bins {e[j]}..{e[j + 1] - 1}")
                else:
                    b = j % self.k
                    kind = "phase" if j >= self.k else "|amplitude|"
                    period = f"period {W / b:.3g} obs" if b and W else "mean"
                    sources.append(f"{span}: {kind} of bin {b} ({period})")
        return explain_frame(comps, sources, np.ones(len(comps)))


register_shape_spec(
    Spectral,
    name="spectral",
    params={"window": int, "k": int, "mode": str, "phase": bool, "flavour": str},
    tier="B",
    safe_scope="rowwise",
    source="Agrawal, Faloutsos & Swami (1993), FODO; DFT via numpy.fft; clean-room, Panelary",
)
