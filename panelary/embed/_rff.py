"""Random features with mandatory direct links and a past-only bandwidth.

Random Fourier features (Rahimi & Recht, 2007) map a row ``x`` to
``sqrt(2 / D) * cos(x W / sigma + b)`` so that inner products approximate a
Gaussian kernel of bandwidth ``sigma``. This module ships them with the three
finance guardrails of build-contract section 3.5 built in, not bolted on:

1. **Direct links are always on.** The standardised inputs are concatenated
   in front of the random features (the random-vector functional-link design
   of Pao et al., 1994). It is the older and better-performing design, and it
   partially breaks the rotation invariance that Grinsztajn et al. (NeurIPS
   2022) show is harmful on tabular data. There is no switch to turn it off.
2. **The bandwidth is a past-only statistic.** The median heuristic computed
   on the full sample is a leak. Here it is an **expanding** statistic over
   the fitted rows: the scaler (mean / std) and the bandwidth used for a row
   at time ``t`` are computed from fitted rows at times ``<= t`` only, and a
   row after the last fitted date uses the last entry. Nothing about the
   bandwidth at ``t`` changes when later rows are added or altered.
3. **A multi-scale bandwidth bank** (``0.25, 0.5, 1, 2, 4`` times the median
   heuristic by default) instead of one ``sigma``.

``kernel="arccos"`` gives arc-cosine (ReLU) features ``sqrt(2 / D) * max(0, x W)``
(Cho & Saul, 2009) -- no bandwidth. ``orthogonal=True`` draws ``W`` in
orthogonal blocks (QR of Gaussian blocks, rows rescaled to chi-distributed
norms; Yu et al., 2016), a lower-variance estimator of the Gaussian kernel.

The Nagel (2025) warning
------------------------
Random-feature regressions with more features than observations and no ridge
penalty degenerate *mechanically* into volatility-timed momentum and mis-learn
on mean-reverting data (Nagel, NBER w34104; the argument extends to random
features of firm characteristics). This transform is only layer 2: read it out
with :class:`~panelary.embed.PreValidatedRidge`, which refuses ``alpha = 0``
and warns when features outnumber dates, and run
:func:`~panelary.embed.reversal_check` before believing a result.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
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

__all__ = ["RandomFourierFeatures", "orthogonal_gaussian", "time_key"]


def time_key(s: pl.Series) -> NDArray[Any]:
    """A time column as an orderable numpy array (temporal types -> physical ints)."""
    if s.dtype.is_temporal():
        return s.to_physical().to_numpy()
    if s.dtype.is_numeric():
        return s.to_numpy()
    raise TypeError(f"time column of dtype {s.dtype} is not orderable numerically.")


def orthogonal_gaussian(
    d: int, D: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """``(d, D)`` orthogonal random features matrix (Yu et al., 2016).

    Blocks of ``d`` columns: ``Q`` from the QR of a ``d x d`` Gaussian (signs
    fixed by ``diag(R)``), each column rescaled by the norm of an independent
    ``d``-dimensional Gaussian, so columns are marginally ``N(0, I_d)`` but
    orthogonal within a block.
    """
    blocks = []
    for _ in range(-(-D // d)):
        G = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(G)
        Q = Q * np.where(np.diag(R) < 0, -1.0, 1.0)[None, :]
        norms = np.sqrt((rng.standard_normal((d, d)) ** 2).sum(axis=0))
        blocks.append(Q * norms[None, :])
    return np.ascontiguousarray(np.hstack(blocks)[:, :D])


class _RunningMedian:
    """Exact running median of a stream (two heaps)."""

    def __init__(self) -> None:
        self.lo: list[float] = []  # max-heap (negated)
        self.hi: list[float] = []

    def push(self, v: float) -> None:
        if not self.lo or v <= -self.lo[0]:
            heapq.heappush(self.lo, -v)
        else:
            heapq.heappush(self.hi, v)
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self) -> float:
        if not self.lo:
            return float("nan")
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return 0.5 * (-self.lo[0] + self.hi[0])


class RandomFourierFeatures(_EmbedTransform):
    """Random Fourier / arc-cosine features with direct links and a past-only bandwidth.

    Row-wise (``axis="feature"``): each output row is a function of that row's
    inputs and of the scaler / bandwidth schedule entry for its date.

    Parameters
    ----------
    n_components : int, default 256
        Total random features ``D``; split evenly over ``scales`` for the
        Gaussian kernel. The output width is ``d + D`` (direct links first).
    kernel : {"gaussian", "arccos"}, default "gaussian"
    orthogonal : bool, default False
        Orthogonal random features (Gaussian kernel only has a bandwidth, but
        the orthogonal draw applies to both).
    scales : sequence of float, default (0.25, 0.5, 1.0, 2.0, 4.0)
        Multiples of the median-heuristic bandwidth (Gaussian kernel).
    min_periods : int, default 30
        A row is emitted only once at least this many fitted rows exist at or
        before its date; earlier rows are null. Fixed, never length-dependent.
    pairs_per_date : int, default 16
        Random pairs added to the running median-heuristic sample at each fitted
        date (one member at that date, the other any fitted row at or before it).
    seed : int, default 0
        Seeds the random weights and the pair draws.
    columns : sequence of str, optional
        Numeric columns and/or numeric ``pl.Array`` columns (e.g. an upstream
        embedding). ``None`` = every numeric feature column.
    name : str, default "rff"
        Output column (``output="array"``) or prefix (``output="columns"``).
    output : {"array", "columns"}, default "array"
    dtype : {"float32", "float16", "float64"}, default "float32"
    keep_features : bool, default False
    max_bytes : int, optional
    entity, time : str, optional

    Attributes
    ----------
    schedule_time_ : numpy.ndarray
        Fitted dates (physical representation), ascending.
    schedule_mean_, schedule_std_ : numpy.ndarray
        ``(K, d)`` expanding scaler at each fitted date.
    schedule_count_ : numpy.ndarray
        Fitted rows at or before each date.
    schedule_sigma_ : numpy.ndarray
        ``(K,)`` expanding median-heuristic bandwidth (Gaussian kernel).
    weights_, offsets_ : numpy.ndarray
        Random ``(n_scales, d, D_s)`` weights and ``(n_scales, D_s)`` phases.

    Notes
    -----
    ``fit_is_empty = False``: the schedule is learned from the rows passed to
    :meth:`fit` -- fit it on the training fold. Because every schedule entry is
    an expanding statistic, even the *in-sample* features of a training row
    carry no information from later training rows.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = False
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.LIFT,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="none",
        streaming="batch",
        cost_hint="O(n d D)",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        "n_components",
        "kernel",
        "orthogonal",
        "scales",
        "min_periods",
        "pairs_per_date",
        "seed",
        "name",
        "output",
        "dtype",
        "keep_features",
        "max_bytes",
    )
    _state_arrays: ClassVar[tuple[str, ...]] = (
        "schedule_time_",
        "schedule_mean_",
        "schedule_std_",
        "schedule_count_",
        "schedule_sigma_",
        "weights_",
        "offsets_",
        "time_dtype_",
    )

    def __init__(
        self,
        *,
        n_components: int = 256,
        kernel: str = "gaussian",
        orthogonal: bool = False,
        scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        min_periods: int = 30,
        pairs_per_date: int = 16,
        seed: int = 0,
        columns: Sequence[str] | str | None = None,
        name: str = "rff",
        output: str = "array",
        dtype: str = "float32",
        keep_features: bool = False,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        owner = "RandomFourierFeatures"
        self.n_components = check_pos_int(n_components, "n_components", owner)
        self.kernel = check_choice(kernel, "kernel", ("gaussian", "arccos"), owner)
        self.orthogonal = bool(orthogonal)
        sc = tuple(float(s) for s in scales)
        if not sc or any(not np.isfinite(s) or s <= 0 for s in sc):
            raise ValueError(
                f"{owner}: `scales` must be positive floats, got {scales!r}."
            )
        self.scales = sc
        if self.kernel == "gaussian" and self.n_components < len(sc):
            raise ValueError(
                f"{owner}: `n_components` must be >= len(scales) = {len(sc)}."
            )
        self.min_periods = check_pos_int(min_periods, "min_periods", owner, minimum=2)
        self.pairs_per_date = check_pos_int(pairs_per_date, "pairs_per_date", owner)
        self.seed = check_seed(seed, owner)
        self.name = name
        self.output = check_choice(output, "output", ("array", "columns"), owner)
        self.dtype = check_choice(
            dtype, "dtype", ("float16", "float32", "float64"), owner
        )
        self.keep_features = bool(keep_features)
        self.schedule_time_: NDArray[Any] | None = None
        self.schedule_mean_: NDArray[np.float64] | None = None
        self.schedule_std_: NDArray[np.float64] | None = None
        self.schedule_count_: NDArray[np.int64] | None = None
        self.schedule_sigma_: NDArray[np.float64] | None = None
        self.weights_: NDArray[np.float64] | None = None
        self.offsets_: NDArray[np.float64] | None = None
        self.time_dtype_: str = ""

    # ------------------------------------------------------------------ #
    @property
    def _n_scales(self) -> int:
        return len(self.scales) if self.kernel == "gaussian" else 1

    @property
    def _per_scale(self) -> int:
        return self.n_components // self._n_scales

    def _in_width(self) -> int:
        assert self.schedule_mean_ is not None
        return int(self.schedule_mean_.shape[1])

    def _out_width(self, in_width: int) -> int:
        return in_width + self._n_scales * self._per_scale

    @property
    def feature_names_(self) -> list[str]:
        """Direct-link names (``z{i}``) then random-feature names."""
        d = self._in_width()
        names = [f"z{i}" for i in range(d)]
        if self.kernel == "gaussian":
            names += [
                f"s{c:g}_{j}" for c in self.scales for j in range(self._per_scale)
            ]
        else:
            names += [f"relu_{j}" for j in range(self._per_scale)]
        return names

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        return resolve_matrix_columns(panel, self.columns, "RandomFourierFeatures")

    def _plan(self, shape: InputShape) -> Plan:
        m = self._out_width(shape.width)
        return self._make_plan(
            shape,
            rows=shape.rows,
            width=m,
            scratch=8.0 * shape.rows * (2 * shape.width + 2 * m),
            knob="n_components",
        )

    def _draw_weights(self, d: int) -> None:
        rng = np.random.default_rng([self.seed, 0, d])
        S, D = self._n_scales, self._per_scale
        W = np.empty((S, d, D))
        for s in range(S):
            W[s] = (
                orthogonal_gaussian(d, D, rng)
                if self.orthogonal
                else rng.standard_normal((d, D))
            )
        self.weights_ = W
        self.offsets_ = rng.uniform(0.0, 2.0 * np.pi, size=(S, D))

    # ------------------------------------------------------------------ #
    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        self.feature_names_in_ = cols
        frame = panel.collect().sort(
            [panel.time_col, panel.entity_col], maintain_order=True
        )
        X = matrix_from_columns(frame, cols)
        tser = frame[panel.time_col]
        self.time_dtype_ = str(tser.dtype)
        tk = time_key(tser)
        ok = np.isfinite(X).all(axis=1)
        X, tk = X[ok], tk[ok]
        d = X.shape[1]
        self._draw_weights(d)
        if X.shape[0] == 0:
            raise ValueError(
                "RandomFourierFeatures.fit: no row has every input observed."
            )
        u, first, counts = np.unique(tk, return_index=True, return_counts=True)
        # Expanding scaler per fitted date (rows are time-sorted).
        csum = np.add.reduceat(X, first, axis=0).cumsum(axis=0)
        csq = np.add.reduceat(X * X, first, axis=0).cumsum(axis=0)
        n_cum = counts.cumsum()
        mean = csum / n_cum[:, None]
        var = np.maximum(csq / n_cum[:, None] - mean * mean, 0.0)
        std = np.sqrt(var)
        std = np.where(std > 1e-12 * np.maximum(1.0, np.abs(mean)), std, 1.0)
        sigma = np.full(u.shape[0], np.nan)
        if self.kernel == "gaussian":
            rng = np.random.default_rng([self.seed, 1])
            med = _RunningMedian()
            for k in range(u.shape[0]):
                a = first[k] + rng.integers(0, counts[k], size=self.pairs_per_date)
                b = rng.integers(0, n_cum[k], size=self.pairs_per_date)
                keep = a != b
                if keep.any():
                    Za = (X[a[keep]] - mean[k]) / std[k]
                    Zb = (X[b[keep]] - mean[k]) / std[k]
                    for v in np.sqrt(((Za - Zb) ** 2).sum(axis=1)).tolist():
                        if v > 0.0:
                            med.push(v)
                sigma[k] = med.median()
        self.schedule_time_ = u
        self.schedule_mean_ = mean
        self.schedule_std_ = std
        self.schedule_count_ = n_cum.astype(np.int64)
        self.schedule_sigma_ = sigma

    def _restore_hook(self) -> None:
        self.time_dtype_ = str(self.time_dtype_)

    def _map(self, X: NDArray[np.float64], k: NDArray[np.int64]) -> NDArray[np.float64]:
        """Rows ``X`` with schedule indices ``k`` (all valid) -> ``(n, d + D)``."""
        assert self.schedule_mean_ is not None and self.schedule_std_ is not None
        assert self.weights_ is not None and self.offsets_ is not None
        Z = (X - self.schedule_mean_[k]) / self.schedule_std_[k]
        parts = [Z]
        D = self._per_scale
        if self.kernel == "gaussian":
            assert self.schedule_sigma_ is not None
            sig = self.schedule_sigma_[k][:, None]
            for s, c in enumerate(self.scales):
                A = Z @ self.weights_[s]
                parts.append(
                    np.sqrt(2.0 / D) * np.cos(A / (c * sig) + self.offsets_[s])
                )
        else:
            parts.append(np.sqrt(2.0 / D) * np.maximum(Z @ self.weights_[0], 0.0))
        return np.hstack(parts)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        cols = list(self.feature_names_in_)
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"RandomFourierFeatures.transform: column(s) {missing} not in panel."
            )
        frame = panel.collect()
        width = array_width(frame.schema, cols)
        if width != self._in_width():
            raise ValueError(
                f"RandomFourierFeatures.transform: input width {width} differs from the "
                f"fitted width {self._in_width()}."
            )
        tser = frame[panel.time_col]
        if str(tser.dtype) != self.time_dtype_:
            raise TypeError(
                f"RandomFourierFeatures.transform: time dtype {tser.dtype} differs from "
                f"the fitted {self.time_dtype_}."
            )
        self._enforce_budget(InputShape(rows=frame.height, width=width, entities=0))
        X = matrix_from_columns(frame, cols)
        assert self.schedule_time_ is not None and self.schedule_count_ is not None
        k = np.searchsorted(self.schedule_time_, time_key(tser), side="right") - 1
        valid = (k >= 0) & np.isfinite(X).all(axis=1)
        kk = np.where(k >= 0, k, 0)
        valid &= self.schedule_count_[kk] >= self.min_periods
        if self.kernel == "gaussian":
            assert self.schedule_sigma_ is not None
            s = self.schedule_sigma_[kk]
            valid &= np.isfinite(s) & (s > 0)
        out = np.full((frame.height, self._out_width(width)), np.nan)
        if valid.any():
            out[valid] = self._map(X[valid], kk[valid])
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
            feature_names=self.feature_names_,
        )
        return PanelFrame(
            res, entity=panel.entity_col, time=panel.time_col, validate=False
        )
