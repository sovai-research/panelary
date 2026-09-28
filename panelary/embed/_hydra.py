"""Hydra over trailing windows: competing random convolutional kernels, stateless.

Hydra (Dempster, Schmidt & Webb, 2023, "HYDRA: competing convolutional kernels
for fast and accurate time series classification", *Data Mining and Knowledge
Discovery*) arranges random length-9 kernels into ``g`` groups of ``k`` and
lets the kernels of a group **compete** at every time point: the kernel with
the largest response in its group collects that response ("soft" max
counting), and the kernel with the smallest response collects a count of one
("hard" min counting). A window's features are those per-kernel tallies,
at every dilation of an exponential ladder, over the window and its first
difference. It sits between dictionary methods (counting "words") and ROCKET
(pooling random-kernel responses).

This implementation, per trailing window of ``window`` observations:

* kernels: ``N(0, 1)`` weights, each kernel centred to zero mean and scaled to
  unit L1 norm; one independent bank of ``g * k`` kernels per representation
  and dilation, drawn from ``seed`` at construction -- ``fit_is_empty = True``;
* dilations: :func:`~panelary.embed._intervals.dilation_ladder` (the shared
  formula), valid positions only -- no padding, so nothing outside the window
  is read and edge effects are absent;
* representations: the window ``x`` and its first difference ``dx``;
* tallies: the scatter into ``(row, group, kernel)`` bins is one
  :func:`numpy.bincount` over flattened indices (never ``np.add.at``), divided
  by the number of valid positions so windows of every dilation are comparable.

Amplitude retention (build contract section 5, item 7)
------------------------------------------------------
``normalise=False`` is the default: soft tallies are in the input's units, so
a window's scale survives. Explicit scale features -- the window's standard
deviation, median absolute deviation and realised volatility
``sqrt(sum(dx^2))`` -- are appended by default (``scale_features=True``).
``normalise=True`` reproduces per-window z-normalisation exactly: the kernels
sum to zero and are applied linearly, so z-normalising a window divides every
response by the window's standard deviation and leaves every argmax / argmin
unchanged -- only the soft tallies change, by that factor.

Clean-room note
---------------
Implemented from the paper's description and the build contract (section 6)
only. The GPL-3.0 reference implementation and the aeon / sktime / wildboar
ports were not read. Kernel normalisation, per-dilation banks and the
valid-only convolution are this module's choices.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from panelary.embed._contract import _TemporalEmbedder, check_pos_int, check_seed
from panelary.embed._intervals import dilation_ladder
from panelary.shape._axes import Axis, Flavour, Intent, ShapeSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["KERNEL_TAPS", "HydraEmbedder", "hydra_kernels"]

#: Every Hydra kernel has nine taps.
KERNEL_TAPS: int = 9
_REPS: tuple[str, ...] = ("x", "dx")
_SCALE_NAMES: tuple[str, ...] = ("scale_std", "scale_mad", "scale_rv")


def hydra_kernels(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """``(n, 9)`` random kernels: ``N(0, 1)``, zero mean, unit L1 norm."""
    w = rng.standard_normal((n, KERNEL_TAPS))
    w -= w.mean(axis=1, keepdims=True)
    return w / np.abs(w).sum(axis=1, keepdims=True)


def _responses(
    R: NDArray[np.float64], kernels: NDArray[np.float64], d: int
) -> NDArray[np.float64]:
    """``(c, n_kernels, m)`` valid dilated responses; an explicit 9-tap sum (no BLAS)."""
    L = R.shape[1]
    m = L - (KERNEL_TAPS - 1) * d
    out = kernels[None, :, 0, None] * R[:, None, 0:m]
    for j in range(1, KERNEL_TAPS):
        out = out + kernels[None, :, j, None] * R[:, None, j * d : j * d + m]
    return out


def _tally(
    Rsp: NDArray[np.float64], g: int, k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Soft-max and hard-min tallies ``(c, g * k)`` each, normalised by position count."""
    c, _, m = Rsp.shape
    V = Rsp.reshape(c, g, k, m)
    amax = V.argmax(axis=2)  # (c, g, m)
    amin = V.argmin(axis=2)
    vmax = np.take_along_axis(V, amax[:, :, None, :], axis=2)[:, :, 0, :]
    base = (np.arange(c * g, dtype=np.int64) * k).reshape(c, g, 1)
    size = c * g * k
    soft = np.bincount((base + amax).ravel(), weights=vmax.ravel(), minlength=size)
    hard = np.bincount((base + amin).ravel(), minlength=size).astype(np.float64)
    return soft.reshape(c, g * k) / m, hard.reshape(c, g * k) / m


class HydraEmbedder(_TemporalEmbedder):
    """Hydra competing-kernel embedding of each row's strictly trailing window.

    Parameters
    ----------
    window : int, default 64
        Trailing window length; must be ``>= 10`` (one undilated kernel on the
        first difference).
    n_groups : int, default 8
        Groups ``g`` per representation and dilation.
    n_kernels : int, default 8
        Kernels ``k`` per group.
    seed : int, default 0
        Seeds every kernel bank.
    normalise : bool, default False
        Divide soft tallies by the window's standard deviation (exactly
        per-window z-normalisation; see the module docstring). Off by default
        because in finance amplitude *is* volatility.
    scale_features : bool, default True
        Append ``scale_std``, ``scale_mad``, ``scale_rv`` of the window.
    columns, warmup, output, dtype, keep_features, chunk_rows, max_bytes, entity, time
        As for :class:`~panelary.embed.QuantEmbedder`.
    suffix : str, default "_hydra"

    Attributes
    ----------
    kernels_ : numpy.ndarray
        ``(2, n_dilations, g * k, 9)`` kernel banks (``dx`` banks for a
        dilation the ``dx`` window cannot hold are unused).
    dilations_ : numpy.ndarray
        Dilations of the ``x`` representation (``dx`` uses those that fit).
    feature_names_ : list of str
        ``f"{rep}_d{d}_g{group}_k{kernel}_{max|min}"`` then the scale features.

    Notes
    -----
    Width: ``2 * g * k * (n_dil_x + n_dil_dx)`` (+3 scale features).
    """

    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.LIFT,
        axis=Axis.TIME,
        flavour=Flavour.TRAILING,
        width_rule="exact",
        invertible="none",
        streaming="batch",
        cost_hint="O(n W g k log W)",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        *_TemporalEmbedder._param_names,
        "n_groups",
        "n_kernels",
        "seed",
        "normalise",
        "scale_features",
    )
    _state_arrays: ClassVar[tuple[str, ...]] = ("kernels_", "dilations_")

    def __init__(
        self,
        *,
        window: int = 64,
        n_groups: int = 8,
        n_kernels: int = 8,
        seed: int = 0,
        normalise: bool = False,
        scale_features: bool = True,
        columns: Sequence[str] | str | None = None,
        warmup: str = "drop",
        output: str = "array",
        dtype: str = "float32",
        suffix: str = "_hydra",
        keep_features: bool = False,
        chunk_rows: int = 1024,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(
            window=window,
            columns=columns,
            warmup=warmup,
            output=output,
            dtype=dtype,
            suffix=suffix,
            keep_features=keep_features,
            chunk_rows=chunk_rows,
            max_bytes=max_bytes,
            entity=entity,
            time=time,
        )
        owner = "HydraEmbedder"
        if self.window < KERNEL_TAPS + 1:
            raise ValueError(
                f"{owner}: `window` must be >= {KERNEL_TAPS + 1}, got {window!r}."
            )
        self.n_groups = check_pos_int(n_groups, "n_groups", owner)
        self.n_kernels = check_pos_int(n_kernels, "n_kernels", owner, minimum=2)
        self.seed = check_seed(seed, owner)
        self.normalise = bool(normalise)
        self.scale_features = bool(scale_features)
        dil = dilation_ladder(self.window, KERNEL_TAPS)
        rng = np.random.default_rng([self.seed, self.window])
        gk = self.n_groups * self.n_kernels
        self.dilations_ = dil
        self.kernels_ = np.stack(
            [np.stack([hydra_kernels(gk, rng) for _ in dil]) for _ in _REPS]
        )
        self._restore_hook()

    def _restore_hook(self) -> None:
        dil = np.asarray(self.dilations_, dtype=np.int64)
        self.dilations_ = dil
        self.kernels_ = np.asarray(self.kernels_, dtype=np.float64)
        self._plan_reps: list[tuple[int, str, list[tuple[int, int]]]] = []
        names: list[str] = []
        for ri, rep in enumerate(_REPS):
            L = self.window - ri
            fits = [
                (di, int(d))
                for di, d in enumerate(dil.tolist())
                if (KERNEL_TAPS - 1) * d + 1 <= L
            ]
            self._plan_reps.append((ri, rep, fits))
            for _, d in fits:
                for kind in ("max", "min"):
                    names.extend(
                        f"{rep}_d{d}_g{g}_k{k}_{kind}"
                        for g in range(self.n_groups)
                        for k in range(self.n_kernels)
                    )
        if self.scale_features:
            names.extend(_SCALE_NAMES)
        self._names = names

    @property
    def feature_names_(self) -> list[str]:
        return list(self._names)

    def _embed_windows(self, W: NDArray[np.float64]) -> NDArray[np.float64]:
        c = W.shape[0]
        g, k = self.n_groups, self.n_kernels
        dx = W[:, 1:] - W[:, :-1]
        reps = (W, dx)
        sd = {0: W.std(axis=1), 1: dx.std(axis=1)}
        blocks: list[NDArray[np.float64]] = []
        for ri, _, fits in self._plan_reps:
            R = np.ascontiguousarray(reps[ri])
            for di, d in fits:
                soft, hard = _tally(_responses(R, self.kernels_[ri, di], d), g, k)
                if self.normalise:
                    # z-normalising the window divides x *and* dx by sigma(x).
                    s = sd[0]
                    soft = soft / np.where(s > 0, s, 1.0)[:, None]
                blocks.extend((soft, hard))
        if self.scale_features:
            med = np.median(W, axis=1)
            mad = np.median(np.abs(W - med[:, None]), axis=1)
            rv = np.sqrt((dx * dx).sum(axis=1))
            blocks.append(np.stack([sd[0], mad, rv], axis=1))
        return np.hstack(blocks) if blocks else np.empty((c, 0))
