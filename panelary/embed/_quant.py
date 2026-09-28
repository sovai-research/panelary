"""QUANT over trailing windows: dyadic-interval quantiles, stateless by construction.

QUANT (Dempster, Schmidt & Webb, 2023, arXiv:2308.00928) summarises a series
by the **quantiles of fixed dyadic intervals** over a few representations of
it. Nothing is learned: the intervals are a deterministic function of the
series length alone, so fitting on wildly different data gives byte-identical
state. That is leak-safety *by construction rather than by discipline*, and it
is why this is the flagship of :mod:`panelary.embed` (build contract section 0).

What is computed, per trailing window of ``window`` observations
-----------------------------------------------------------------
Four representations of the window ``x`` (all within the window, so nothing
past ``t`` is ever read):

``x``
    the raw window;
``dx``
    the first difference, smoothed by a 5-tap moving average with replicate
    padding at the window's own edges;
``d2x``
    the second difference;
``fft``
    the magnitude of the real FFT, ``|rfft(x)|``.

For each representation of length ``n`` the dyadic interval set of
:func:`~panelary.embed._intervals.dyadic_intervals` is used, at depth
``min(interval_depth, floor(log2(n)) + 1)``: at depth ``j`` the representation
is split into ``2**j`` equal intervals, and at every ``j >= 1`` the same
partition shifted by half an interval is added. For an interval of length
``m``, ``k = max(1, m // 4)`` quantiles are taken at the mid-point levels
``(2i + 1) / (2k)`` (linear interpolation between order statistics), and **the
interval mean is subtracted from every second quantile** (``i = 1, 3, ...``).

Amplitude is retained: there is no per-window normalisation, so the features
are in the input's units (in finance, amplitude *is* volatility).

Honest limits
-------------
The paper validates these features with an ExtraTrees classifier on
univariate UCR data. Performance under a *linear* head, and on financial
panels, is unmeasured; treat the method as a strong, cheap, leak-free prior,
not as a promise.

Clean-room note
---------------
Implemented from the paper's description and ``plans/todo/embed-build-contract.md``
section 3.2. The GPL-3.0 reference implementation and the aeon / sktime /
wildboar ports were not read. Where the details (quantile levels, interval
edges, the smoothing of ``dx``) are not pinned down by the text, the choices
above are this module's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from panelary.embed._contract import _TemporalEmbedder, check_pos_int
from panelary.embed._intervals import dyadic_intervals
from panelary.shape._axes import Axis, Flavour, Intent, ShapeSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["REPRESENTATIONS", "QuantEmbedder", "quant_features"]

#: The four QUANT representations, in output order.
REPRESENTATIONS: tuple[str, ...] = ("x", "dx", "d2x", "fft")


def _rep_length(rep: str, window: int) -> int:
    return {"x": window, "dx": window - 1, "d2x": window - 2, "fft": window // 2 + 1}[
        rep
    ]


def _representation(rep: str, W: NDArray[np.float64]) -> NDArray[np.float64]:
    """One representation of every row of ``W`` (row-independent)."""
    if rep == "x":
        return W
    if rep == "dx":
        d = W[:, 1:] - W[:, :-1]
        p = np.concatenate([d[:, :1], d[:, :1], d, d[:, -1:], d[:, -1:]], axis=1)
        n = d.shape[1]
        acc = p[:, 0:n] + p[:, 1 : n + 1]
        acc = acc + p[:, 2 : n + 2]
        acc = acc + p[:, 3 : n + 3]
        acc = acc + p[:, 4 : n + 4]
        return acc / 5.0
    if rep == "d2x":
        return W[:, 2:] - 2.0 * W[:, 1:-1] + W[:, :-2]
    if rep == "fft":
        return np.abs(np.fft.rfft(W, axis=1))
    raise ValueError(f"unknown representation {rep!r}")


class _Group:
    """Intervals of one representation that share a length (sorted together)."""

    __slots__ = ("rep", "m", "starts", "levels", "odd", "cols")

    def __init__(
        self, rep: str, m: int, starts: list[int], offsets: list[int], k: int
    ) -> None:
        self.rep = rep
        self.m = m
        self.starts = np.asarray(starts, dtype=np.int64)
        self.levels = (2.0 * np.arange(k) + 1.0) / (2.0 * k)
        self.odd = (np.arange(k) % 2) == 1
        self.cols = (
            np.asarray(offsets, dtype=np.int64)[:, None] + np.arange(k)
        ).ravel()


def _build(
    window: int, depth: int, reps: Sequence[str], divisor: int
) -> tuple[list[_Group], list[str]]:
    groups: dict[tuple[str, int], tuple[list[int], list[int], int]] = {}
    names: list[str] = []
    for rep in reps:
        n = _rep_length(rep, window)
        for iv in dyadic_intervals(n, depth):
            k = max(1, iv.length // divisor)
            key = (rep, iv.length)
            starts, offsets, _ = groups.setdefault(key, ([], [], k))
            starts.append(iv.start)
            offsets.append(len(names))
            names.extend(
                f"{rep}_d{iv.depth}_{iv.start}_{iv.stop}_q{i}" for i in range(k)
            )
    built = [_Group(rep, m, s, o, k) for (rep, m), (s, o, k) in groups.items()]
    return built, names


def _interval_quantiles(R: NDArray[np.float64], g: _Group) -> NDArray[np.float64]:
    """``(c, n_intervals * k)`` quantile features of group ``g`` over ``R``."""
    idx = g.starts[:, None] + np.arange(g.m, dtype=np.int64)
    V = np.ascontiguousarray(R[:, idx])  # (c, g, m)
    mean = V.mean(axis=-1)
    S = np.sort(V, axis=-1)
    h = g.levels * (g.m - 1)
    lo = np.floor(h).astype(np.int64)
    hi = np.minimum(lo + 1, g.m - 1)
    frac = h - lo
    q = S[..., lo] + frac * (S[..., hi] - S[..., lo])  # (c, g, k)
    q[..., g.odd] -= mean[..., None]
    return q.reshape(q.shape[0], -1)


def quant_features(
    W: NDArray[Any],
    *,
    interval_depth: int = 6,
    representations: Sequence[str] = REPRESENTATIONS,
    quantile_divisor: int = 4,
) -> NDArray[np.float64]:
    """QUANT features for every row of a ``(n_windows, L)`` array.

    The functional core of :class:`QuantEmbedder`. Row ``i`` depends on
    ``W[i]`` alone.

    Parameters
    ----------
    W : array-like
        ``(n_windows, L)`` finite values, ``L >= 4``.
    interval_depth : int, default 6
    representations : sequence of str, default all four
        Subset of :data:`REPRESENTATIONS`.
    quantile_divisor : int, default 4
        ``k = max(1, m // quantile_divisor)`` quantiles per interval of length ``m``.

    Returns
    -------
    numpy.ndarray
        ``(n_windows, n_features)`` float64.
    """
    X = np.ascontiguousarray(np.asarray(W, dtype=np.float64))
    if X.ndim != 2 or X.shape[1] < 4:
        raise ValueError(f"quant_features: expected (n, L>=4), got shape {X.shape}.")
    groups, names = _build(
        X.shape[1], interval_depth, tuple(representations), quantile_divisor
    )
    return _apply(X, groups, len(names), tuple(representations))


def _apply(
    X: NDArray[np.float64], groups: list[_Group], width: int, reps: tuple[str, ...]
) -> NDArray[np.float64]:
    out = np.empty((X.shape[0], width), dtype=np.float64)
    cache = {rep: np.ascontiguousarray(_representation(rep, X)) for rep in reps}
    for g in groups:
        out[:, g.cols] = _interval_quantiles(cache[g.rep], g)
    return out


class QuantEmbedder(_TemporalEmbedder):
    """QUANT embedding of each row's strictly trailing window -- no fitted state.

    ``fit`` reads the schema only (``fit_is_empty = True`` is enforced: it is
    handed a zero-row frame), so the state is a function of the parameters
    and the input-column names. See the module docstring for the features.

    Parameters
    ----------
    window : int, default 64
        Trailing window length ``W`` (observations). Powers of two give the
        cleanest dyadic intervals. Must be ``>= 4``.
    interval_depth : int, default 6
        Maximum dyadic depth; capped per representation at
        ``floor(log2(n)) + 1``.
    representations : sequence of str, default ("x", "dx", "d2x", "fft")
    quantile_divisor : int, default 4
        ``m // quantile_divisor`` quantiles per interval of length ``m``.
    columns : str or sequence of str, optional
        Numeric columns to embed, each independently. ``None`` = every numeric
        feature column.
    warmup : {"drop", "null"}, default "drop"
        Rows without a full window: dropped with an
        :class:`~panelary.embed.EmbeddingWarmupWarning`, or kept as nulls.
    output : {"array", "columns"}, default "array"
        One fixed-size ``pl.Array`` column per input column
        (``f"{col}{suffix}"``), or one top-level column per feature.
    dtype : {"float32", "float16", "float64"}, default "float32"
        Storage dtype; arithmetic is float64.
    suffix : str, default "_quant"
    keep_features : bool, default False
        Keep the input columns alongside the keys.
    chunk_rows : int, default 2048
        Windows per kernel call (bounds scratch memory; invisible in output).
    max_bytes : int, optional
        Scratch budget checked before allocation.
    entity, time : str, optional

    Attributes
    ----------
    feature_names_ : list of str
        ``f"{rep}_d{depth}_{start}_{stop}_q{i}"`` per feature.
    n_outputs : int

    Examples
    --------
    >>> import numpy as np, polars as pl
    >>> rng = np.random.default_rng(0)
    >>> df = pl.DataFrame({"id": np.repeat(["a", "b"], 100),
    ...                    "t": np.tile(np.arange(100), 2),
    ...                    "r": rng.standard_normal(200)})
    >>> q = QuantEmbedder(window=32, columns="r", warmup="null", entity="id", time="t")
    >>> out = q.fit_transform(df).collect()
    >>> out["r_quant"].dtype
    Array(Float32, shape=(...,))
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
        cost_hint="O(n W log W)",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        *_TemporalEmbedder._param_names,
        "interval_depth",
        "representations",
        "quantile_divisor",
    )

    def __init__(
        self,
        *,
        window: int = 64,
        interval_depth: int = 6,
        representations: Sequence[str] = REPRESENTATIONS,
        quantile_divisor: int = 4,
        columns: Sequence[str] | str | None = None,
        warmup: str = "drop",
        output: str = "array",
        dtype: str = "float32",
        suffix: str = "_quant",
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
        if self.window < 4:
            raise ValueError(f"QuantEmbedder: `window` must be >= 4, got {window!r}.")
        self.interval_depth = check_pos_int(
            interval_depth, "interval_depth", "QuantEmbedder"
        )
        reps = (
            (representations,)
            if isinstance(representations, str)
            else tuple(representations)
        )
        bad = [r for r in reps if r not in REPRESENTATIONS]
        if not reps or bad or len(set(reps)) != len(reps):
            raise ValueError(
                f"QuantEmbedder: `representations` must be a non-empty, duplicate-free "
                f"subset of {REPRESENTATIONS}, got {representations!r}."
            )
        self.representations = tuple(r for r in REPRESENTATIONS if r in reps)
        self.quantile_divisor = check_pos_int(
            quantile_divisor, "quantile_divisor", "QuantEmbedder"
        )
        self._groups, self._names = _build(
            self.window,
            self.interval_depth,
            self.representations,
            self.quantile_divisor,
        )

    @property
    def feature_names_(self) -> list[str]:
        return list(self._names)

    def _embed_windows(self, W: NDArray[np.float64]) -> NDArray[np.float64]:
        return _apply(W, self._groups, len(self._names), self.representations)
