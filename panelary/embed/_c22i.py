"""RandIntC22: catch24 over random dilated intervals of each trailing window.

Whole-series catch22 is a weak time-series representation; the same features
computed over **random intervals** are a strong one -- two independent studies
(arXiv:2201.12048, arXiv:2308.01071) put interval-catch22 about three rank
positions above whole-series catch22, level with tsfresh's ~780 features. This
is the highest-return item in the embed build on code the library already owns.

catch24 = the 22 catch22 features plus the raw mean and standard deviation,
the two that catch22 deliberately excludes. In a cross-section the level and
scale of a window *are* the signal, so they are on by default.

The intervals are drawn once, from a seeded generator, as a function of the
window length alone (:func:`~panelary.embed._intervals.random_dilated_intervals`),
so there is no fitted state: ``fit_is_empty = True``. The batched kernels of
:func:`panelary.catch22.catch22_batch` make it affordable: every interval of a
chunk of windows is one catch22 call over ``(n_windows, m)`` rows (intervals of
equal length share a call).

Clean-room note: interval sampling is this module's own design (documented on
:func:`~panelary.embed._intervals.random_dilated_intervals`); catch22 is the
library's clean-room implementation (Lubba et al., 2019).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from panelary.catch22 import CATCH22_NAMES, CATCH24_EXTRA_NAMES, catch22_batch
from panelary.embed._contract import _TemporalEmbedder, check_pos_int, check_seed
from panelary.embed._intervals import IntervalSet, random_dilated_intervals
from panelary.shape._axes import Axis, Flavour, Intent, ShapeSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["RandIntC22"]


class RandIntC22(_TemporalEmbedder):
    """catch24 over ``n_intervals`` seeded random dilated intervals of each trailing window.

    Parameters
    ----------
    window : int, default 64
        Trailing window length (observations).
    n_intervals : int, default 16
        Number of intervals ``k``; the output width is ``k * 24`` (``k * 22``
        with ``catch24=False``). The interval-catch22 literature uses
        ``k`` in the tens (e.g. 45); cost is linear in ``k``.
    min_interval_length : int, default 20
        Shortest interval (capped at ``window``). Below ~20 points the
        fluctuation-analysis features are undefined and come out NaN.
    seed : int, default 0
        Seeds the interval draw. The intervals are a deterministic function of
        ``(window, n_intervals, min_interval_length, seed)``.
    catch24 : bool, default True
        Append the raw mean and standard deviation to the 22 features.
    columns, warmup, output, dtype, keep_features, chunk_rows, max_bytes, entity, time
        As for :class:`~panelary.embed.QuantEmbedder`.
    suffix : str, default "_c22i"

    Attributes
    ----------
    intervals_ : IntervalSet
        The drawn intervals (``start``, ``length``, ``dilation``).
    feature_names_ : list of str
        ``f"i{j}_{feature}"``.

    Notes
    -----
    ``fit_is_empty = True``: two fits on disjoint data give byte-identical
    :meth:`get_state`. Features that catch22 cannot compute on an interval
    (e.g. a constant stretch) are NaN, never imputed.
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
        cost_hint="O(n k W log W)",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        *_TemporalEmbedder._param_names,
        "n_intervals",
        "min_interval_length",
        "seed",
        "catch24",
    )
    _state_arrays: ClassVar[tuple[str, ...]] = (
        "interval_start_",
        "interval_length_",
        "interval_dilation_",
    )

    def __init__(
        self,
        *,
        window: int = 64,
        n_intervals: int = 16,
        min_interval_length: int = 20,
        seed: int = 0,
        catch24: bool = True,
        columns: Sequence[str] | str | None = None,
        warmup: str = "drop",
        output: str = "array",
        dtype: str = "float32",
        suffix: str = "_c22i",
        keep_features: bool = False,
        chunk_rows: int = 2048,
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
        self.n_intervals = check_pos_int(n_intervals, "n_intervals", "RandIntC22")
        self.min_interval_length = check_pos_int(
            min_interval_length, "min_interval_length", "RandIntC22", minimum=2
        )
        self.seed = check_seed(seed, "RandIntC22")
        self.catch24 = bool(catch24)
        iv = random_dilated_intervals(
            self.window,
            self.n_intervals,
            seed=self.seed,
            min_length=self.min_interval_length,
        )
        self.interval_start_ = iv.start
        self.interval_length_ = iv.length
        self.interval_dilation_ = iv.dilation
        self._restore_hook()

    def _restore_hook(self) -> None:
        self.intervals_ = IntervalSet(
            window=self.window,
            start=np.asarray(self.interval_start_, dtype=np.int64),
            length=np.asarray(self.interval_length_, dtype=np.int64),
            dilation=np.asarray(self.interval_dilation_, dtype=np.int64),
        )
        self._base_names = list(CATCH22_NAMES) + (
            list(CATCH24_EXTRA_NAMES) if self.catch24 else []
        )
        by_len: dict[int, list[int]] = {}
        for i, m in enumerate(self.intervals_.length.tolist()):
            by_len.setdefault(int(m), []).append(i)
        self._by_len = {m: np.asarray(ids, dtype=np.int64) for m, ids in by_len.items()}
        self._idx = [self.intervals_.indices(i) for i in range(len(self.intervals_))]

    @property
    def feature_names_(self) -> list[str]:
        return [f"i{j}_{f}" for j in range(self.n_intervals) for f in self._base_names]

    def _embed_windows(self, W: NDArray[np.float64]) -> NDArray[np.float64]:
        c = W.shape[0]
        nf = len(self._base_names)
        out = np.empty((c, self.n_intervals, nf), dtype=np.float64)
        for m, ids in self._by_len.items():
            idx = np.stack([self._idx[i] for i in ids.tolist()])  # (g, m)
            block = W[:, idx].transpose(1, 0, 2).reshape(ids.size * c, m)
            feats = catch22_batch(block, catch24=self.catch24)  # (g * c, nf)
            out[:, ids, :] = feats.reshape(ids.size, c, nf).transpose(1, 0, 2)
        return out.reshape(c, self.n_intervals * nf)
