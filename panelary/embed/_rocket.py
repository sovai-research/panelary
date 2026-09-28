"""Causal rolling MiniRocket-PPV: a prefix-safe convolutional embedding.

MiniRocket (Dempster, Schmidt & Webb, KDD 2021) convolves a series with a fixed
bank of 84 length-9 kernels at several dilations, subtracts a bias, and pools
each response into the *proportion of positive values* (PPV). Applied the way
the paper applies it -- to a whole series at once -- it has two routes by which
information from after ``t`` reaches a feature used at ``t``:

1. **Pooling.** PPV over the whole series averages convolution outputs from
   every time step, so the value used at an early row is built partly from
   rows after it.
2. **Bias fitting.** Each bias is a quantile of convolution outputs on the
   training data. Fit once on the whole sample, and the future sets the
   thresholds that define every feature -- a leak that survives even when the
   pooling is made trailing.

:class:`CausalMiniRocket` closes both. The convolution is aligned so that
``C[t]`` is a function of ``x[t - 8d], ..., x[t]`` only (left-only padding),
pooling is a trailing (or expanding) mean of the indicator ``C > b`` computed
with a per-entity cumulative sum -- O(1) per step and exactly prefix-invariant
-- and the biases are fitted state, learned in :meth:`fit` from the rows it is
given and nothing else.

:class:`LeakyMiniRocketReference` is the *measuring stick*: the same kernels,
convolution and biases, but pooled over each entity's whole series. It exists
so the tests and ``benchmarks/prefix_safe_rocket`` can price the pooling
channel. It declares ``leakage_safe = False`` and must never be used as a
feature generator.

Clean-room note
---------------
Implemented from the paper's description and from
``plans/todo/embed-build-contract.md`` section 6 only. The reference
implementation (GPL-3.0) and the aeon/sktime/wildboar ports were not read.
Where this module differs from the paper, it differs on purpose:

* causal (left-only) alignment and trailing pooling instead of whole-series
  pooling with centred padding;
* biases are quantiles of the convolution output pooled over a seeded
  subsample of training *rows* across entities, rather than of one randomly
  chosen training series -- in a panel the natural unit of training data is a
  row, and a threshold drawn from one entity's scale would be meaningless for
  another's;
* the quantile levels (a low-discrepancy golden-ratio sequence with a seeded
  offset) are **permuted** across feature slots, following R-Clustering
  (Jorge & Ruben, DMKD 2024), which reports that MiniRocket's in-order bias
  assignment injects artificial structure into the feature vector;
* ~500 features by default rather than ~10,000, also following R-Clustering;
* no per-window normalisation, so amplitude is retained: biases are absolute
  thresholds in the input's units, and MPV is measured in those units too.

References
----------
Dempster, A., Schmidt, D. F. & Webb, G. I. (2021). MiniRocket: A very fast
(almost) deterministic transform for time series classification. *KDD '21*.
Tan, C. W., Dempster, A., Bergmeir, C. & Webb, G. I. (2022). MultiRocket:
multiple pooling operators and transformations for fast and effective time
series classification. *Data Mining and Knowledge Discovery* (the MPV pooling).
Jorge, M. & Ruben, C. (2024). Time series clustering with random convolutional
kernels. *Data Mining and Knowledge Discovery* (R-Clustering).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

__all__ = [
    "KERNEL_LENGTH",
    "N_KERNELS",
    "CausalMiniRocket",
    "LeakyMiniRocketReference",
    "kernel_weights",
]

#: Every kernel has nine taps.
KERNEL_LENGTH: int = 9
#: ``C(9, 3) = 84``: every way of placing the three weight-2 taps.
N_KERNELS: int = 84

#: Fractional part of the golden ratio; the step of the low-discrepancy
#: sequence the quantile levels are drawn from.
_GOLDEN_FRAC: float = (math.sqrt(5.0) - 1.0) / 2.0
#: Upper bound on ``entities x longest series`` cells processed at once. Keeps
#: the per-dilation working set to tens of MB whatever the panel size.
_CHUNK_CELLS: int = 1 << 15
_POOLINGS: tuple[str, ...] = ("ppv", "mpv")

Padding = Literal["none", "zero"]
Output = Literal["array", "columns"]
Scope = Literal["trailing", "series"]

# The 84 kernels, as the index triples of their weight-2 taps (lexicographic).
_TRIPLES: np.ndarray = np.array(
    list(itertools.combinations(range(KERNEL_LENGTH), 3)), dtype=np.int64
)


def kernel_weights() -> np.ndarray:
    """The fixed MiniRocket kernel bank as an ``(84, 9)`` weight matrix.

    Each row has three taps of weight ``2`` and six of weight ``-1``, so every
    kernel sums to zero and its output is invariant to an additive shift of the
    input. Tap ``j`` multiplies ``x[t - (8 - j) * d]``: tap 8 is the current
    observation, tap 0 the oldest.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(84, 9)``.

    Examples
    --------
    >>> w = kernel_weights()
    >>> w.shape, float(w.sum(axis=1).max())
    ((84, 9), 0.0)
    """
    w = np.full((N_KERNELS, KERNEL_LENGTH), -1.0, dtype=np.float64)
    np.put_along_axis(w, _TRIPLES, 2.0, axis=1)
    return w


# --------------------------------------------------------------------------- #
# Numerical core (pure numpy; nothing here sees more than one entity per row)
# --------------------------------------------------------------------------- #
def _dilation_schedule(n_per_kernel: int, max_dilation: int) -> list[tuple[int, int]]:
    """``[(dilation, n_biases), ...]`` with ``sum(n_biases) == n_per_kernel``.

    ``n_per_kernel`` exponents are spaced evenly on ``[0, log2(max_dilation)]``
    and floored to integers; a dilation that several exponents floor to gets
    that many biases per kernel. Small dilations therefore get more thresholds,
    and every quantity here is a function of the parameters alone -- never of
    series length (hard invariant 1).
    """
    if max_dilation <= 1 or n_per_kernel == 1:
        return [(1, n_per_kernel)]
    exps = np.linspace(0.0, math.log2(max_dilation), n_per_kernel)
    # The epsilon stops 2 ** log2(7) = 6.9999... from flooring to 6.
    dil = np.floor(np.exp2(exps) + 1e-9).astype(np.int64)
    values, counts = np.unique(dil, return_counts=True)
    return [(int(d), int(c)) for d, c in zip(values, counts, strict=True)]


def _lags(d: int) -> list[int]:
    """Row lag of each tap at dilation ``d`` (tap 8 is lag 0)."""
    return [(KERNEL_LENGTH - 1 - j) * d for j in range(KERNEL_LENGTH)]


def _conv(xl: np.ndarray) -> np.ndarray:
    """All 84 kernel responses from the nine lagged copies ``xl`` (``(9, ...)``).

    With weights ``2`` on the taps in ``T`` and ``-1`` elsewhere,
    ``C = 2 * sum_T x - (A - sum_T x) = 3 * sum_T x - A`` where ``A`` is the sum
    of all nine taps. The sums are written as explicit elementwise additions in
    a fixed order, so a cell's value never depends on the array's shape (a
    BLAS product could change its accumulation order with the panel size and
    break bit-exact prefix invariance).
    """
    a = xl[0] + xl[1]
    for j in range(2, KERNEL_LENGTH):
        a = a + xl[j]
    s = xl[_TRIPLES[:, 0]] + xl[_TRIPLES[:, 1]]
    s += xl[_TRIPLES[:, 2]]
    s *= 3.0
    s -= a[None, ...]
    return s


def _lagged_block(x: np.ndarray, d: int, padding: str) -> np.ndarray:
    """``(9, E, L)`` lagged copies of the entity-by-time block ``x``.

    Only the *left* edge is padded (NaN for ``"none"``, zero for ``"zero"``):
    tap ``j`` at cell ``t`` reads ``x[t - lag_j]`` from the same entity row, or
    the pad value when that index is before the entity's first observation.
    """
    n_e, n_t = x.shape
    pad = np.nan if padding == "none" else 0.0
    out = np.empty((KERNEL_LENGTH, n_e, n_t), dtype=np.float64)
    for j, lag in enumerate(_lags(d)):
        if lag == 0:
            out[j] = x
        elif lag >= n_t:
            out[j] = pad
        else:
            out[j, :, :lag] = pad
            out[j, :, lag:] = x[:, : n_t - lag]
    return out


def _lagged_flat(x: np.ndarray, pos: np.ndarray, d: int, padding: str) -> np.ndarray:
    """``(9, N)`` lagged copies of a flat, entity-contiguous, time-sorted ``x``.

    ``pos`` is each row's 0-based position within its entity, so ``pos >= lag``
    is exactly "the lagged row exists and belongs to the same entity".
    """
    n = x.shape[0]
    pad = np.nan if padding == "none" else 0.0
    out = np.full((KERNEL_LENGTH, n), pad, dtype=np.float64)
    for j, lag in enumerate(_lags(d)):
        if lag == 0:
            out[j] = x
            continue
        ok = pos >= lag
        src = np.flatnonzero(ok) - lag
        out[j, ok] = x[src]
    return out


def _window_diff(cum: np.ndarray, window: int | None) -> np.ndarray:
    """Trailing-window sums along the last axis from a per-row cumulative sum."""
    if window is None or window >= cum.shape[-1]:
        return cum
    out = cum.copy()
    out[..., window:] -= cum[..., :-window]
    return out


def _pool_block(
    c: np.ndarray,
    valid: np.ndarray,
    bias: np.ndarray,
    *,
    window: int | None,
    min_periods: int,
    pooling: Sequence[str],
    scope: str,
) -> list[np.ndarray]:
    """Pool responses ``c`` (``(F, E, L)``) into one ``(F, E, L)`` array per stat.

    ``scope="trailing"`` is the causal path: a cumulative sum along time of the
    indicator ``c > bias`` differenced at lag ``window`` (``None`` means
    expanding). Cumulative sums run along the time axis of one entity row, so a
    cell reads only its own entity's cells at or before it. ``scope="series"``
    is the leaky reference: the same statistics over the entity's whole
    series, broadcast back to every one of its rows.
    """
    ind = c > bias[:, None, None]  # NaN compares False
    if scope == "series":
        n_valid = valid.sum(axis=-1, dtype=np.int64)[None, :, None]
        n_pos = ind.sum(axis=-1, dtype=np.int64)[:, :, None]
    else:
        n_valid = _window_diff(np.cumsum(valid, axis=-1, dtype=np.int64), window)[
            None, ...
        ]
        n_pos = _window_diff(np.cumsum(ind, axis=-1, dtype=np.int64), window)
    enough = n_valid >= min_periods
    out: list[np.ndarray] = []
    shape = c.shape
    for stat in pooling:
        if stat == "ppv":
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(enough, n_pos / np.maximum(n_valid, 1), np.nan)
        else:  # "mpv": mean of (c - bias) over the cells where it is positive
            excess = np.where(ind, c - bias[:, None, None], 0.0)
            if scope == "series":
                total = excess.sum(axis=-1)[:, :, None]
            else:
                total = _window_diff(np.cumsum(excess, axis=-1), window)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_pos = np.where(n_pos > 0, total / np.maximum(n_pos, 1), 0.0)
            val = np.where(enough, mean_pos, np.nan)
        out.append(np.broadcast_to(val, shape))
    return out


def _chunks(lengths: np.ndarray) -> list[np.ndarray]:
    """Group entity indices into blocks of at most ``_CHUNK_CELLS`` padded cells.

    Entities are taken longest first so that each block's right-padding (up to
    its longest member) stays small on an unbalanced panel.
    """
    order = np.argsort(-lengths, kind="stable")
    blocks: list[np.ndarray] = []
    start = 0
    n = order.shape[0]
    while start < n:
        width = int(lengths[order[start]])
        per_block = max(1, _CHUNK_CELLS // max(width, 1))
        blocks.append(order[start : start + per_block])
        start += per_block
    return blocks


class _SortedPanel:
    """A panel sorted by ``(entity, time)``, with the map back to input order."""

    __slots__ = ("frame", "row", "lengths", "starts", "pos", "_sorted")

    def __init__(self, panel: PanelFrame) -> None:
        df = panel.collect()
        row_col = "__pn_rocket_row__"
        while row_col in df.columns:
            row_col = f"_{row_col}_"
        srt = df.with_row_index(row_col).sort(
            [panel.entity_col, panel.time_col], maintain_order=True
        )
        self.frame = df
        self.row = srt[row_col].to_numpy().astype(np.int64)
        ent = srt[panel.entity_col].rle_id().to_numpy().astype(np.int64)
        self.lengths = np.bincount(ent).astype(np.int64) if ent.size else ent
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(
            np.int64
        )
        self.pos = np.arange(ent.size, dtype=np.int64) - self.starts[ent]
        self._sorted = srt

    def values(self, column: str) -> np.ndarray:
        """``column`` as float64 in sorted order; nulls become NaN."""
        return (
            self._sorted[column]
            .cast(pl.Float64)
            .fill_null(np.nan)
            .to_numpy()
            .astype(np.float64, copy=False)
        )


# --------------------------------------------------------------------------- #
# The transformer
# --------------------------------------------------------------------------- #
class CausalMiniRocket(PanelTransformer):
    """Causal rolling MiniRocket-PPV embedding of one or more panel columns.

    For every row ``(entity, t)`` and every input column, emits ``n_features``
    pooled convolution statistics (per pooling operator) computed from that
    entity's observations at or before ``t`` only. The kernel bank is fixed
    (84 kernels of length 9, weights in ``{-1, 2}``); the dilations are fixed
    by the parameters; the only fitted state is one bias per feature, learned
    in :meth:`fit` from the rows passed to it.

    Parameters
    ----------
    columns : str or sequence of str
        Numeric column(s) to embed. Each is embedded independently, with its
        own biases. The kernels sum to zero, so the convolution ignores the
        series' level; feed something whose *scale* is comparable across
        entities (log prices or returns, not raw prices), because one bias is
        shared by every entity.
    n_features : int, default=504
        Features per column per pooling operator, rounded down to a multiple of
        84 (one bias per kernel per slot). The default follows R-Clustering's
        finding that ~500 features beat 10,000 for unsupervised use.
    window : int or None, default=63
        Trailing pooling window, in rows. ``None`` pools over the expanding
        history. Either way the value at ``t`` never depends on how many rows
        follow it.
    min_periods : int, optional
        Minimum number of valid convolution outputs in the pooling window for a
        feature to be emitted; otherwise it is NaN. Defaults to ``window`` (a
        full window), or ``1`` when ``window is None``.
    max_dilation : int, optional
        Largest dilation. Defaults to ``max(1, (window - 1) // 8)`` so that one
        kernel's receptive field fits the pooling window, or ``8`` when
        ``window is None``. Dilations are spaced exponentially from 1.
    padding : {"none", "zero"}, default="none"
        What the oldest taps read before an entity's first observation.
        ``"none"``: nothing -- the output is undefined until ``8 * d`` rows of
        history exist. ``"zero"``: zeros, which is only meaningful for a
        returns-like input whose natural "no information" value is 0. Padding
        is left-only either way.
    pooling : sequence of {"ppv", "mpv"}, default=("ppv",)
        ``"ppv"``: proportion of positive values of ``C - b``. ``"mpv"``: mean
        of ``C - b`` over the positive cells (0 when there are none), in the
        input's units -- the amplitude-retaining companion.
    seed : int, default=0
        Seeds the quantile-level offset, the permutation of levels across
        feature slots, and the row subsample used to fit the biases.
    max_fit_rows : int or None, default=65_536
        Biases are quantiles of the convolution output over at most this many
        valid training rows per dilation, drawn with the seeded RNG. ``None``
        uses every valid row.
    output : {"array", "columns"}, default="array"
        ``"array"`` adds one fixed-size ``pl.Array`` column per input column,
        ``f"{column}{suffix}"``. ``"columns"`` adds one top-level column per
        feature, ``f"{column}{suffix}_{name}"`` for each name in
        :attr:`feature_names_`.
    suffix : str, default="_rocket"
        Output-name suffix.
    dtype : {"float32", "float64"}, default="float32"
        Storage type of the output. All arithmetic is float64.
    entity, time : str, optional
        Default panel keys, used when a bare polars frame is passed.

    Attributes
    ----------
    biases_ : numpy.ndarray
        ``(n_columns, n_features)`` fitted thresholds.
    quantile_levels_ : numpy.ndarray
        ``(n_features,)`` quantile level behind each bias, after permutation.
    kernel_index_, dilation_ : numpy.ndarray
        ``(n_features,)`` kernel (row of :func:`kernel_weights`) and dilation of
        each feature slot.
    feature_names_ : list of str
        ``f"{pool}_d{dilation}_k{kernel:02d}_{i}"``, in output order.
    n_fit_rows_ : dict of int to int
        Rows the biases at each dilation were fitted on (after subsampling).

    Notes
    -----
    **Contracts.** ``panel_safe``: every operation acts on one entity's row of
    an entity-by-time block, so entities never mix. ``leakage_safe``: with the
    state fixed, ``C[t]`` reads ``x[t - 8d .. t]`` and the pooled value at
    ``t`` reads ``C[t - window + 1 .. t]`` of the same entity -- nothing later,
    and nothing that depends on the series length. The biases are the only
    fitted state; fit them inside the fold (``fit(train)``), never on the whole
    sample, or the second leak channel described in the module docstring is
    open. ``fit_is_empty = False`` records exactly that.

    **Guardrails.** This is layer 2 of the embed pipeline (nonlinear
    expansion). A linear readout on top needs a cross-validated, non-zero ridge
    penalty; with ``P > T`` and no penalty, random-feature regressions degenerate
    mechanically (Nagel, 2025).

    Examples
    --------
    >>> import numpy as np, polars as pl
    >>> rng = np.random.default_rng(0)
    >>> df = pl.DataFrame({
    ...     "ticker": np.repeat(["a", "b"], 80),
    ...     "t": np.tile(np.arange(80), 2),
    ...     "x": rng.standard_normal(160).cumsum(),
    ... })
    >>> rocket = CausalMiniRocket("x", n_features=84, window=20, entity="ticker", time="t")
    >>> out = rocket.fit(df.filter(pl.col("t") < 50)).transform(df).collect()
    >>> out["x_rocket"].dtype
    Array(Float32, shape=(84,))
    """

    panel_safe = True
    leakage_safe = True
    #: Biases are learned from data, so ``fit`` is not a no-op: two disjoint
    #: training sets give different state. Fit inside the fold.
    fit_is_empty: bool = False
    #: Pools along time within an entity; never fits across a single date.
    is_cross_sectional: bool = False

    #: Pooling scope. ``"trailing"`` here; the leaky reference overrides it.
    _scope: Scope = "trailing"

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_features: int = 504,
        window: int | None = 63,
        min_periods: int | None = None,
        max_dilation: int | None = None,
        padding: Padding = "none",
        pooling: Sequence[str] = ("ppv",),
        seed: int = 0,
        max_fit_rows: int | None = 65_536,
        output: Output = "array",
        suffix: str = "_rocket",
        dtype: Literal["float32", "float64"] = "float32",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        cols = [columns] if isinstance(columns, str) else list(columns)
        if not cols or not all(isinstance(c, str) for c in cols):
            raise ValueError(
                "`columns` must be a column name or a non-empty list of them."
            )
        if len(set(cols)) != len(cols):
            raise ValueError(f"`columns` contains duplicates: {cols}.")
        if isinstance(n_features, bool) or int(n_features) < N_KERNELS:
            raise ValueError(
                f"`n_features` must be >= {N_KERNELS} (one bias per kernel), "
                f"got {n_features!r}."
            )
        if window is not None and (isinstance(window, bool) or int(window) < 1):
            raise ValueError(
                f"`window` must be a positive int or None, got {window!r}."
            )
        default_mp = int(window) if window is not None else 1
        mp = default_mp if min_periods is None else int(min_periods)
        if mp < 1 or (window is not None and mp > int(window)):
            raise ValueError(
                f"`min_periods` must be in [1, window], got {min_periods!r} "
                f"(window={window!r})."
            )
        if max_dilation is None:
            max_d = max(1, (int(window) - 1) // 8) if window is not None else 8
        else:
            max_d = int(max_dilation)
        if max_d < 1:
            raise ValueError(f"`max_dilation` must be >= 1, got {max_dilation!r}.")
        if padding not in ("none", "zero"):
            raise ValueError(f"`padding` must be 'none' or 'zero', got {padding!r}.")
        pools = [pooling] if isinstance(pooling, str) else list(pooling)
        bad = [p for p in pools if p not in _POOLINGS]
        if not pools or bad or len(set(pools)) != len(pools):
            raise ValueError(
                f"`pooling` must be a non-empty, duplicate-free subset of "
                f"{_POOLINGS}, got {pooling!r}."
            )
        if max_fit_rows is not None and int(max_fit_rows) < 1:
            raise ValueError(
                f"`max_fit_rows` must be >= 1 or None, got {max_fit_rows!r}."
            )
        if output not in ("array", "columns"):
            raise ValueError(f"`output` must be 'array' or 'columns', got {output!r}.")
        if dtype not in ("float32", "float64"):
            raise ValueError(f"`dtype` must be 'float32' or 'float64', got {dtype!r}.")

        self.columns = cols
        self.n_features = (int(n_features) // N_KERNELS) * N_KERNELS
        self.window = None if window is None else int(window)
        self.min_periods = mp
        self.max_dilation = max_d
        self.padding: Padding = padding
        self.pooling = tuple(pools)
        self.seed = int(seed)
        self.max_fit_rows = None if max_fit_rows is None else int(max_fit_rows)
        self.output: Output = output
        self.suffix = suffix
        self.dtype = dtype

        # Parameter-only structure: a function of the arguments, never of data.
        self._schedule = _dilation_schedule(self.n_features // N_KERNELS, max_d)
        kern, dil, rep = [], [], []
        for d, count in self._schedule:
            for k in range(N_KERNELS):
                for i in range(count):
                    kern.append(k)
                    dil.append(d)
                    rep.append(i)
        self.kernel_index_ = np.asarray(kern, dtype=np.int64)
        self.dilation_ = np.asarray(dil, dtype=np.int64)
        self._slot_rep = np.asarray(rep, dtype=np.int64)
        rng = np.random.default_rng([self.seed, 0])
        offset = float(rng.random())
        levels = np.mod(offset + _GOLDEN_FRAC * np.arange(1, self.n_features + 1), 1.0)
        self.quantile_levels_ = levels[rng.permutation(self.n_features)]
        self.feature_names_ = [
            f"{p}_d{d}_k{k:02d}_{i}"
            for p in self.pooling
            for k, d, i in zip(
                self.kernel_index_, self.dilation_, self._slot_rep, strict=True
            )
        ]
        self.biases_: np.ndarray = np.empty((0, self.n_features), dtype=np.float64)
        self.n_fit_rows_: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    @property
    def dilations(self) -> list[int]:
        """The distinct dilations, ascending."""
        return [d for d, _ in self._schedule]

    @property
    def n_outputs(self) -> int:
        """Features emitted per input column: ``n_features * len(pooling)``."""
        return self.n_features * len(self.pooling)

    def _fit(self, panel: PanelFrame) -> None:
        missing = [c for c in self.columns if c not in panel.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.fit: column(s) {missing} not in panel."
            )
        sp = _SortedPanel(panel)
        biases = np.empty((len(self.columns), self.n_features), dtype=np.float64)
        n_rows: dict[int, int] = {}
        for ci, col in enumerate(self.columns):
            x = sp.values(col)
            for d in self.dilations:
                xl = _lagged_flat(x, sp.pos, d, self.padding)
                total = xl[0] + xl[1]
                for j in range(2, KERNEL_LENGTH):
                    total = total + xl[j]
                cand = np.flatnonzero(np.isfinite(total))
                if cand.size == 0:
                    raise ValueError(
                        f"{type(self).__name__}.fit: column {col!r} has no valid "
                        f"convolution output at dilation {d} (each needs "
                        f"{8 * d + 1} consecutive finite rows within an entity "
                        "with padding='none'). Fit on longer series, lower "
                        "`max_dilation`, or use padding='zero' for returns."
                    )
                if self.max_fit_rows is not None and cand.size > self.max_fit_rows:
                    rng = np.random.default_rng([self.seed, 1, ci, d])
                    cand = np.sort(rng.choice(cand, self.max_fit_rows, replace=False))
                resp = np.sort(_conv(xl[:, cand]), axis=1)  # (84, m)
                slots = np.flatnonzero(self.dilation_ == d)
                q = self.quantile_levels_[slots]
                h = q * (resp.shape[1] - 1)
                lo = np.floor(h).astype(np.int64)
                hi = np.minimum(lo + 1, resp.shape[1] - 1)
                k = self.kernel_index_[slots]
                lo_v, hi_v = resp[k, lo], resp[k, hi]
                biases[ci, slots] = lo_v + (h - lo) * (hi_v - lo_v)
                n_rows[d] = int(cand.size)
        self.biases_ = biases
        self.n_fit_rows_ = n_rows

    def _embed(self, sp: _SortedPanel, ci: int, col: str) -> np.ndarray:
        """``(N, n_outputs)`` features for column ``col``, in *sorted* row order."""
        x = sp.values(col)
        n = x.shape[0]
        out_dtype = np.float32 if self.dtype == "float32" else np.float64
        out = np.full((n, self.n_outputs), np.nan, dtype=out_dtype)
        if n == 0:
            return out
        bias = self.biases_[ci]
        for ents in _chunks(sp.lengths):
            lens = sp.lengths[ents]
            width = int(lens.max())
            block = np.full((ents.size, width), np.nan, dtype=np.float64)
            e_loc = np.repeat(np.arange(ents.size), lens)
            t_loc = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
            flat = np.repeat(sp.starts[ents], lens) + t_loc
            block[e_loc, t_loc] = x[flat]
            for d in self.dilations:
                slots = np.flatnonzero(self.dilation_ == d)
                resp = _conv(_lagged_block(block, d, self.padding))  # (84, E, L)
                valid = np.isfinite(resp[0])
                stats = _pool_block(
                    resp[self.kernel_index_[slots]],
                    valid,
                    bias[slots],
                    window=self.window,
                    min_periods=self.min_periods,
                    pooling=self.pooling,
                    scope=self._scope,
                )
                for pi, val in enumerate(stats):
                    cols = slots + pi * self.n_features
                    out[np.ix_(flat, cols)] = val[:, e_loc, t_loc].T
        return out

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.columns if c not in panel.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: column(s) {missing} not in panel."
            )
        if self.biases_.shape[0] != len(self.columns):
            raise RuntimeError(f"{type(self).__name__} has no fitted biases; call fit.")
        sp = _SortedPanel(panel)
        n = sp.frame.height
        new: list[pl.Series] = []
        for ci, col in enumerate(self.columns):
            feats = self._embed(sp, ci, col)
            restored = np.empty_like(feats)
            restored[sp.row] = feats
            name = f"{col}{self.suffix}"
            if self.output == "array":
                if n == 0:
                    dt = pl.Float32 if self.dtype == "float32" else pl.Float64
                    new.append(pl.Series(name, [], dtype=pl.Array(dt, self.n_outputs)))
                else:
                    new.append(pl.Series(name, restored))
            else:
                new.extend(
                    pl.Series(f"{name}_{fname}", restored[:, i])
                    for i, fname in enumerate(self.feature_names_)
                )
        out = sp.frame.with_columns(new)
        return PanelFrame(
            out.lazy(), entity=panel.entity_col, time=panel.time_col, validate=False
        )


class LeakyMiniRocketReference(CausalMiniRocket):
    """**Leaky by design.** Whole-series pooling, for measurement only.

    Identical kernels, dilations, convolution and biases to
    :class:`CausalMiniRocket`, but each statistic is pooled over the entity's
    *entire* series in the frame passed to :meth:`transform` and broadcast back
    to every row -- the way MiniRocket is applied to a whole series. The value
    at row ``t`` therefore depends on rows after ``t``, and on how many there
    are. ``window`` is ignored; ``min_periods`` applies to the whole series.

    This is the pooling channel of the two-channel leak. The bias channel is
    not a mode of any class: it is calling ``fit`` on the whole sample instead
    of the training fold, and ``fit_transform(full_panel)`` on this class opens
    both channels at once.

    It exists so ``tests/test_embed_rocket*.py`` can show the leak tests
    **fail** on it and so ``benchmarks/prefix_safe_rocket`` can price the
    pooling channel. It declares ``leakage_safe = False``; do not use it as a
    feature generator and do not export it.
    """

    panel_safe = True
    leakage_safe = False
    fit_is_empty: bool = False
    is_cross_sectional: bool = False
    _scope: Scope = "series"

    @classmethod
    def from_fitted(cls, fitted: CausalMiniRocket) -> LeakyMiniRocketReference:
        """A leaky reference sharing ``fitted``'s parameters and biases exactly.

        Parameters
        ----------
        fitted : CausalMiniRocket
            A fitted causal transformer.

        Returns
        -------
        LeakyMiniRocketReference
            Pools over the whole series, with byte-identical biases.

        Raises
        ------
        RuntimeError
            If ``fitted`` is not fitted.
        """
        fitted._check_fitted("from_fitted")
        ref = cls(
            fitted.columns,
            n_features=fitted.n_features,
            window=fitted.window,
            min_periods=fitted.min_periods,
            max_dilation=fitted.max_dilation,
            padding=fitted.padding,
            pooling=fitted.pooling,
            seed=fitted.seed,
            max_fit_rows=fitted.max_fit_rows,
            output=fitted.output,
            suffix=fitted.suffix,
            dtype=fitted.dtype,
            entity=fitted._entity,
            time=fitted._time,
        )
        ref.biases_ = fitted.biases_.copy()
        ref.n_fit_rows_ = dict(fitted.n_fit_rows_)
        ref._fit_panel = fitted._fit_panel
        ref._fitted = True
        return ref
