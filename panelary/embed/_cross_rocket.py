"""CrossROCKET: a bank of fixed, seeded random operators *across* the cross-section.

ROCKET applies fixed random convolution kernels **along time**. CrossROCKET is the
cross-sectional counterpart: at every date it applies a fixed bank of random
operators **across entities**. Each operator is a random composition of the
per-date cross-sectional vocabulary Panelary already exposes under ``.xs`` --
robust (median/MAD) standardisation, quantile ranks, winsorisation/clipping,
quantile thresholds -- plus peer baskets built from strictly trailing
correlations. Every operator is **permutation-equivariant** over entities:
relabelling the entities relabels the output rows and changes nothing else.

Operator families
-----------------
Each operator draws, from a seeded RNG, a random subset of 1..``max_channels``
input columns and a unit-norm Gaussian weight vector over them. Per date:

``median_dev``
    Deviation-from-cross-sectional-median indicator. Robust-z each input column
    (``(x - median) / (1.4826 * MAD)``, clipped at ``+-clip``), combine with the
    random weights, robust-z the combination again, subtract a random threshold
    ``b ~ U(-1, 1)``.
``rank_threshold``
    Random quantile-rank threshold. Combine the columns' centred cross-sectional
    ranks, re-rank the combination to ``(0, 1)``, subtract a random quantile
    ``q ~ U(0.05, 0.95)``.
``subset_agg``
    Deviation from a random *value-defined* subset aggregate. The subset is the
    entities whose rank on a randomly chosen member column falls in a random band
    ``[lo, hi]`` (e.g. "the 30-55th percentile of volatility"); the operator is
    an entity's robust-z combination minus that subset's mean, minus ``b``.
    Membership is decided by values, never by labels, so it stays equivariant.
``peer_dev``
    Deviation from a random peer basket. Entity ``i``'s peers at date ``t`` are
    the ``m`` entities most correlated with it over the ``W`` dates **strictly
    before** ``t`` on ``peer_col`` (``m`` and ``W`` drawn from ``peer_sizes`` and
    ``peer_windows``); the operator is ``i``'s robust-z combination minus the
    peers' mean at ``t``, minus ``b``.

The pre-activation of each operator is passed through ``activation``: ``"hinge"``
(``max(0, .)``, the default -- continuous features suit a ridge head on a
regression target), ``"indicator"`` (``1[. > 0]``, the per-entity analogue of
ROCKET's proportion of positive values) or ``"identity"``.

Pooling -- the design decision
------------------------------
MiniRocket pools each kernel's response along time. Across entities there is no
order to pool along, so a pooled statistic must be permutation-*invariant*;
CrossROCKET offers PPV (proportion of entities with a positive pre-activation)
and MPV (mean activation over entities). **The default is nevertheless
``output="entity"`` -- per-entity, equivariant outputs, no pooling.** The reason
is the task the plan names: a pooled statistic takes one value per date, so under
a linear head it shifts every entity's prediction at that date by the same amount
and cannot change a single cross-sectional rank. Pooled outputs are
market-state (regime) descriptors -- useful for timing, or as interactions with
per-entity features -- not cross-sectional predictors. ``output="pooled"`` and
``output="both"`` expose them, broadcast onto each date's rows so the output
keeps the ``(entity, time)`` keys. ``rank_threshold`` operators are excluded from
pooling: the PPV of ``rank > q`` is ``~1 - q`` on every date by construction, so
it carries no information.

The leakage contract
--------------------
``fit_is_empty = True``. Every random quantity (channels, weights, thresholds,
bands, peer windows and sizes) is drawn from ``seed`` and the *number* of input
columns; :meth:`CrossRocket.fit` reads the schema only, so fitting on any two
datasets with the same columns yields byte-identical state. Every family except
``peer_dev`` is a pure function of date ``t``'s own cross-section, which is
observable at ``t``. ``peer_dev`` additionally reads ``peer_col`` on the ``W``
dates strictly before ``t`` -- and only there, via
:meth:`CrossRocket._peer_history`, the one place history is read. So the
transform is causal and prefix-invariant by construction; ``leakage_safe`` is
``True``. ``panel_safe`` is ``False`` because entities are mixed within a date on
purpose, exactly as for :class:`~panelary.reduce.xs.CrossSectionalPCA`.

Because the transform is stateless and causal, **transform the whole panel once
and split afterwards**. Transforming a test fold in isolation discards the
trailing history the peer family needs, so its first ``W`` dates come out null.

Prior art -- not a novelty claim
--------------------------------
The closest lines of work known to the author, **not verified against the papers
here**: Kelly and Malamud's random-feature ridge ("the virtue of complexity"),
whose random features act on each asset's *own* characteristics rather than
across entities; multivariate MiniRocket's random channel combinations, where
channels have fixed identity whereas entities are exchangeable; and random,
untrained permutation-equivariant set/graph layers (DeepSets-style equivariant
layers; untrained or reservoir graph networks), of which ``subset_agg`` and
``peer_dev`` are special cases. The distinguishing property emphasised here is
equivariance plus strict causality at each date, not the idea of random features.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer

__all__ = ["CrossRocket", "FAMILIES"]

#: The operator families, in the order their columns are emitted.
FAMILIES: tuple[str, ...] = ("median_dev", "rank_threshold", "subset_agg", "peer_dev")
_TAG = {
    "median_dev": "med",
    "rank_threshold": "rank",
    "subset_agg": "sub",
    "peer_dev": "peer",
}
# A stable per-family id feeds the RNG, so one family's draws do not depend on
# which other families are enabled (ablations compare like with like).
_FAMILY_ID = {name: i for i, name in enumerate(FAMILIES)}
_ACTIVATIONS = ("hinge", "indicator", "identity")
_OUTPUTS = ("entity", "pooled", "both")
_POOL_STATS = ("ppv", "mpv")
_DTYPES = {"float64": pl.Float64, "float32": pl.Float32}
#: Scales a median absolute deviation to a standard deviation under normality.
_MAD_TO_SD = 1.4826
_EPS = 1e-12

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --------------------------------------------------------------------------- #
# Per-date numpy kernels. Every one acts column-wise on an (n_entities, k)
# block of ONE date, treats NaN as missing, and is permutation-equivariant in
# the row axis.
# --------------------------------------------------------------------------- #
def _robust_z(X: FloatArray, clip: float, min_count: int) -> FloatArray:
    """Column-wise ``(x - median) / (1.4826 * MAD)``, clipped to ``+-clip``.

    Falls back to the standard deviation where the MAD is zero (more than half
    the cross-section tied) and to ``0`` where the column is constant. Columns
    with fewer than ``min_count`` finite values are all-NaN.
    """
    finite = ~np.isnan(X)
    n_ok = finite.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0) * _MAD_TO_SD
        sd = np.nanstd(X, axis=0)
    scale = np.where(mad > _EPS, mad, sd)
    good = scale > _EPS
    Z = (X - med) / np.where(good, scale, 1.0)
    Z = np.where(good, Z, 0.0)
    Z = np.where(finite, Z, np.nan)
    Z[:, n_ok < min_count] = np.nan
    return np.clip(Z, -clip, clip)


def _uniform_rank(X: FloatArray, min_count: int) -> FloatArray:
    """Column-wise average rank divided by ``n + 1``, so values lie in ``(0, 1)``.

    Ties get the average rank (``.xs.rank(method="average", normalize=True)``),
    which is what keeps tied entities exchangeable. NaN stays NaN and does not
    count towards ``n``; columns with fewer than ``min_count`` finite values are
    all-NaN.
    """
    n, k = X.shape
    nan = np.isnan(X)
    out = np.full((n, k), np.nan)
    if n == 0 or k == 0:
        return out
    order = np.argsort(X, axis=0, kind="stable")  # NaN sorts last
    Xs = np.take_along_axis(X, order, axis=0)
    pos = np.arange(n)[:, None]
    first = np.ones((n, k), dtype=bool)
    first[1:] = Xs[1:] != Xs[:-1]  # NaN != NaN, so every NaN is its own group
    last = np.ones((n, k), dtype=bool)
    last[:-1] = first[1:]
    start = np.maximum.accumulate(np.where(first, pos, 0), axis=0)
    end = np.minimum.accumulate(np.where(last, pos, n - 1)[::-1], axis=0)[::-1]
    avg = (start + end) / 2.0 + 1.0
    np.put_along_axis(out, order, avg, axis=0)
    n_ok = (~nan).sum(axis=0)
    out = out / (n_ok + 1.0)
    out[nan] = np.nan
    out[:, n_ok < min_count] = np.nan
    return out


def _combine(U: FloatArray, W: FloatArray) -> FloatArray:
    """``U @ W`` where a missing input in any *used* channel makes the output NaN."""
    missing = np.isnan(U)
    S = np.where(missing, 0.0, U) @ W
    bad = (missing.astype(np.float64) @ (W != 0.0).astype(np.float64)) > 0.0
    S[bad] = np.nan
    return S


def _activate(pre: FloatArray, activation: str) -> FloatArray:
    """Apply the output non-linearity, preserving NaN."""
    if activation == "hinge":
        return np.maximum(pre, 0.0)  # NaN propagates through np.maximum
    if activation == "indicator":
        return np.where(np.isnan(pre), np.nan, (pre > 0.0).astype(np.float64))
    return pre


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"`{name}` must be an int, got {type(value).__name__!r}.")
    if int(value) < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value!r}.")
    return int(value)


def _int_tuple(name: str, values: Sequence[int], *, minimum: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"`{name}` must be a sequence of ints, got {values!r}.")
    out = tuple(_positive_int(name, v, minimum=minimum) for v in values)
    if not out:
        raise ValueError(f"`{name}` must not be empty.")
    return out


def _choice(name: str, value: object, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise ValueError(f"`{name}` must be one of {list(allowed)}, got {value!r}.")
    return str(value)


class CrossRocket(PanelTransformer):
    """CrossROCKET -- fixed, seeded, permutation-equivariant random operators.

    A bank of ``n_operators`` random operators applied **across entities at each
    date** (see the module docstring for the four families). It is the
    cross-sectional counterpart of ROCKET's random kernels along time: cheap,
    fixed, auditable (:attr:`bank_` lists every operator), causal by
    construction at each date, and stateless (``fit_is_empty = True``).

    Parameters
    ----------
    columns : sequence of str, optional
        Input characteristic columns. ``None`` uses every numeric feature column
        (every column except the panel keys), resolved from the schema in
        :meth:`fit`.
    n_operators : int, default 256
        Number of random operators. They are split as evenly as possible over
        the enabled families, in :data:`FAMILIES` order.
    families : sequence of str, optional
        Subset of ``("median_dev", "rank_threshold", "subset_agg", "peer_dev")``.
        ``None`` enables all four when ``peer_col`` is given and the first three
        otherwise.
    peer_col : str, optional
        Column whose **trailing** history defines peer similarity (typically a
        one-period return). Required by ``peer_dev``; it need not be in
        ``columns``.
    peer_windows : sequence of int, default (20, 60)
        Candidate trailing windows ``W`` (in time steps of the panel's own time
        index). Peers at date ``t`` use the ``W`` dates strictly before ``t``.
    peer_sizes : sequence of int, default (5, 10, 20)
        Candidate peer-basket sizes ``m``.
    peer_min_periods : int, optional
        Minimum finite ``peer_col`` observations inside the window for an entity
        to have peers or be one. ``None`` means ``max(2, ceil(0.75 * W))`` per
        window -- a function of ``W`` only, never of the series length.
    max_channels : int, default 3
        Largest number of input columns one operator combines.
    activation : {"hinge", "indicator", "identity"}, default "hinge"
        Output non-linearity applied to each pre-activation.
    output : {"entity", "pooled", "both"}, default "entity"
        Per-entity equivariant outputs, per-date invariant pooled statistics
        (broadcast onto the date's rows), or both. See *Pooling* in the module
        docstring for why ``"entity"`` is the default.
    pool_stats : sequence of {"ppv", "mpv"}, default ("ppv",)
        Pooled statistics per operator: proportion of positive pre-activations,
        and/or the mean activation.
    clip : float, default 5.0
        Winsorisation bound, in robust-z units, applied after every robust
        standardisation.
    min_cross_section : int, default 5
        Minimum finite entities a column needs at a date; below it the date's
        outputs for that column are null.
    seed : int, default 0
        Seed of the operator bank. The same seed and the same number of input
        columns always give the same bank.
    prefix : str, default "xr"
        Output column prefix.
    dtype : {"float64", "float32"}, default "float64"
        Output dtype. Computation is always float64.
    as_array : bool, default False
        If True, emit one ``pl.Array`` column named ``prefix`` instead of one
        column per operator (missing cells are NaN inside the array).
    keep_features : bool, default False
        If True, append the outputs to the input panel; otherwise return the
        ``(entity, time)`` keys plus the outputs.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        ``False`` -- entities are mixed within each date on purpose.
    leakage_safe : bool
        ``True`` -- each date reads its own cross-section and, for ``peer_dev``,
        ``peer_col`` strictly before that date.
    fit_is_empty : bool
        ``True`` -- the bank is a function of ``seed`` and the schema only.
    is_cross_sectional : bool
        ``True`` -- operators act across entities within a date.
    feature_names_in_ : list of str
        The resolved input columns.
    families_ : tuple of str
        The enabled families.
    bank_ : polars.DataFrame
        One row per operator: name, family, channels, weights, threshold, subset
        band, peer window and size. Everything the transform does, in a table.
    output_names_ : list of str
        The emitted column names (per-entity first, then pooled).

    Raises
    ------
    ValueError
        On invalid parameters, or ``peer_dev`` without ``peer_col``.

    Examples
    --------
    >>> import numpy as np, polars as pl
    >>> rng = np.random.default_rng(0)
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": np.repeat([f"s{i}" for i in range(30)], 40),
    ...         "date": np.tile(np.arange(40), 30),
    ...         "ret": rng.standard_normal(1200),
    ...         "vol": rng.random(1200),
    ...     }
    ... )
    >>> xr = CrossRocket(n_operators=16, peer_col="ret", peer_windows=(10,), seed=1)
    >>> out = xr.fit_transform(df, entity="ticker", time="date").collect()
    >>> out.shape
    (1200, 18)
    """

    panel_safe = False
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = True

    def __init__(
        self,
        *,
        columns: Sequence[str] | None = None,
        n_operators: int = 256,
        families: Sequence[str] | None = None,
        peer_col: str | None = None,
        peer_windows: Sequence[int] = (20, 60),
        peer_sizes: Sequence[int] = (5, 10, 20),
        peer_min_periods: int | None = None,
        max_channels: int = 3,
        activation: str = "hinge",
        output: str = "entity",
        pool_stats: Sequence[str] = ("ppv",),
        clip: float = 5.0,
        min_cross_section: int = 5,
        seed: int = 0,
        prefix: str = "xr",
        dtype: str = "float64",
        as_array: bool = False,
        keep_features: bool = False,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.columns = list(columns) if columns is not None else None
        self.n_operators = _positive_int("n_operators", n_operators)
        if peer_col is not None and not isinstance(peer_col, str):
            raise TypeError(f"`peer_col` must be a column name, got {peer_col!r}.")
        self.peer_col = peer_col
        if families is None:
            fams = FAMILIES if peer_col is not None else FAMILIES[:3]
        else:
            if isinstance(families, str):
                families = (families,)
            unknown = [f for f in families if f not in FAMILIES]
            if unknown:
                raise ValueError(
                    f"unknown families {unknown}; choose from {list(FAMILIES)}."
                )
            if not families:
                raise ValueError("`families` must not be empty.")
            fams = tuple(f for f in FAMILIES if f in set(families))
        if "peer_dev" in fams and peer_col is None:
            raise ValueError(
                "the 'peer_dev' family needs `peer_col` (the column whose trailing "
                "history defines peer similarity, e.g. a one-period return)."
            )
        self.families = fams
        self.peer_windows = _int_tuple("peer_windows", peer_windows, minimum=2)
        self.peer_sizes = _int_tuple("peer_sizes", peer_sizes, minimum=1)
        if peer_min_periods is not None:
            peer_min_periods = _positive_int(
                "peer_min_periods", peer_min_periods, minimum=2
            )
            if peer_min_periods > min(self.peer_windows):
                raise ValueError(
                    f"`peer_min_periods={peer_min_periods}` exceeds the smallest "
                    f"peer window {min(self.peer_windows)}; no entity could "
                    "ever have peers in that window."
                )
        self.peer_min_periods = peer_min_periods
        self.max_channels = _positive_int("max_channels", max_channels)
        self.activation = _choice("activation", activation, _ACTIVATIONS)
        self.output = _choice("output", output, _OUTPUTS)
        if isinstance(pool_stats, str):
            pool_stats = (pool_stats,)
        stats = tuple(pool_stats)
        if not stats or any(s not in _POOL_STATS for s in stats):
            raise ValueError(
                f"`pool_stats` must be a non-empty subset of {list(_POOL_STATS)}, "
                f"got {pool_stats!r}."
            )
        self.pool_stats = tuple(s for s in _POOL_STATS if s in stats)
        if not isinstance(clip, (int, float)) or isinstance(clip, bool) or clip <= 0:
            raise ValueError(f"`clip` must be a positive number, got {clip!r}.")
        self.clip = float(clip)
        self.min_cross_section = _positive_int(
            "min_cross_section", min_cross_section, minimum=2
        )
        self.seed = _positive_int("seed", seed, minimum=0)
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(f"`prefix` must be a non-empty string, got {prefix!r}.")
        self.prefix = prefix
        self.dtype = _choice("dtype", dtype, tuple(_DTYPES))
        self.as_array = bool(as_array)
        self.keep_features = bool(keep_features)

        # Fitted state (schema-derived only; see `fit_is_empty`).
        self.feature_names_in_: list[str] = []
        self.families_: tuple[str, ...] = ()
        self.family_: NDArray[np.str_] = np.array([], dtype=str)
        self.channels_: IntArray = np.zeros((0, 0), dtype=np.int64)
        self.weights_: FloatArray = np.zeros((0, 0))
        self.threshold_: FloatArray = np.zeros(0)
        self.member_: IntArray = np.zeros(0, dtype=np.int64)
        self.band_: FloatArray = np.zeros((0, 2))
        self.peer_window_: IntArray = np.zeros(0, dtype=np.int64)
        self.peer_size_: IntArray = np.zeros(0, dtype=np.int64)
        self.operator_names_: list[str] = []
        self.pooled_names_: list[str] = []
        self.output_names_: list[str] = []
        self.bank_: pl.DataFrame = pl.DataFrame()
        self._pooled_ops: IntArray = np.zeros(0, dtype=np.int64)
        self._fam_cache: dict[str, tuple[IntArray, FloatArray]] = {}

    # ------------------------------------------------------------------ #
    # fit: schema only
    # ------------------------------------------------------------------ #
    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        schema = panel.schema
        if self.columns is not None:
            cols = list(self.columns)
            missing = [c for c in cols if c not in schema]
            if missing:
                raise ValueError(
                    f"{type(self).__name__}: column(s) {missing} not found in the "
                    f"panel. Available columns: {panel.columns}."
                )
            non_numeric = [c for c in cols if not schema[c].is_numeric()]
            if non_numeric:
                raise ValueError(
                    f"{type(self).__name__}: column(s) {non_numeric} are not numeric."
                )
        else:
            cols = [c for c in panel.feature_cols if schema[c].is_numeric()]
        if not cols:
            raise ValueError(
                f"{type(self).__name__}: no numeric input columns "
                f"(columns={panel.columns}). Pass `columns=` explicitly."
            )
        if self.peer_col is not None and "peer_dev" in self.families:
            if self.peer_col not in schema:
                raise ValueError(
                    f"{type(self).__name__}: `peer_col={self.peer_col!r}` not found "
                    f"in the panel. Available columns: {panel.columns}."
                )
            if not schema[self.peer_col].is_numeric():
                raise ValueError(
                    f"{type(self).__name__}: `peer_col={self.peer_col!r}` is not "
                    "numeric."
                )
        return cols

    def _fit(self, panel: PanelFrame) -> None:
        """Resolve the input columns from the schema and draw the operator bank.

        Reads no data values: the bank depends on ``seed``, the parameters and
        the number of input columns only, so ``fit_is_empty`` holds.
        """
        cols = self._resolve_columns(panel)
        self.feature_names_in_ = cols
        self._build_bank(len(cols))

    def _build_bank(self, n_channels: int) -> None:
        fams = self.families
        base, extra = divmod(self.n_operators, len(fams))
        counts = [base + (1 if i < extra else 0) for i in range(len(fams))]
        width = min(self.max_channels, n_channels)

        family: list[str] = []
        channels: list[IntArray] = []
        weights: list[FloatArray] = []
        threshold: list[float] = []
        member: list[int] = []
        band: list[tuple[float, float]] = []
        pwin: list[int] = []
        psize: list[int] = []
        names: list[str] = []
        for fam, count in zip(fams, counts, strict=True):
            rng = np.random.default_rng([self.seed, _FAMILY_ID[fam]])
            for k in range(count):
                d = int(rng.integers(1, width + 1))
                ch = np.sort(rng.choice(n_channels, size=d, replace=False))
                w = rng.standard_normal(d)
                w /= np.linalg.norm(w)
                if fam == "rank_threshold":
                    thr = float(rng.uniform(0.05, 0.95))
                else:
                    thr = float(rng.uniform(-1.0, 1.0))
                mem, lo, hi, win, size = -1, math.nan, math.nan, 0, 0
                if fam == "subset_agg":
                    mem = int(rng.integers(n_channels))
                    lo = float(rng.uniform(0.0, 0.8))
                    hi = min(1.0, lo + float(rng.uniform(0.1, 0.5)))
                elif fam == "peer_dev":
                    win = int(rng.choice(self.peer_windows))
                    size = int(rng.choice(self.peer_sizes))
                family.append(fam)
                channels.append(ch.astype(np.int64))
                weights.append(w)
                threshold.append(thr)
                member.append(mem)
                band.append((lo, hi))
                pwin.append(win)
                psize.append(size)
                names.append(f"{self.prefix}_{_TAG[fam]}_{k:03d}")

        n_ops = len(family)
        chan_mat = np.full((n_ops, width), -1, dtype=np.int64)
        w_mat = np.zeros((n_ops, width))
        for i, (ch, w) in enumerate(zip(channels, weights, strict=True)):
            chan_mat[i, : ch.size] = ch
            w_mat[i, : w.size] = w
        self.families_ = tuple(f for f, c in zip(fams, counts, strict=True) if c > 0)
        self.family_ = np.array(family, dtype=str)
        self.channels_ = chan_mat
        self.weights_ = w_mat
        self.threshold_ = np.array(threshold, dtype=np.float64)
        self.member_ = np.array(member, dtype=np.int64)
        self.band_ = np.array(band, dtype=np.float64).reshape(n_ops, 2)
        self.peer_window_ = np.array(pwin, dtype=np.int64)
        self.peer_size_ = np.array(psize, dtype=np.int64)
        self.operator_names_ = names
        self._fam_cache = {
            fam: self._family_weights(fam, n_channels) for fam in FAMILIES
        }

        pooled_ops = [i for i in range(n_ops) if family[i] != "rank_threshold"]
        self._pooled_ops = np.array(pooled_ops, dtype=np.int64)
        self.pooled_names_ = [
            f"{self.prefix}_{stat}_{names[i][len(self.prefix) + 1 :]}"
            for stat in self.pool_stats
            for i in pooled_ops
        ]
        if self.output in ("pooled", "both") and not pooled_ops:
            raise ValueError(
                "`output` asks for pooled statistics, but only 'rank_threshold' "
                "operators are enabled and their PPV is ~1 - q on every date by "
                "construction. Enable another family or use output='entity'."
            )
        emitted: list[str] = []
        if self.output in ("entity", "both"):
            emitted += names
        if self.output in ("pooled", "both"):
            emitted += self.pooled_names_
        self.output_names_ = emitted

        cols = self.feature_names_in_
        self.bank_ = pl.DataFrame(
            {
                "name": names,
                "family": family,
                "channels": [[cols[j] for j in ch] for ch in channels],
                "weights": [w.tolist() for w in weights],
                "threshold": threshold,
                "subset_member": [cols[m] if m >= 0 else None for m in member],
                "subset_lo": [b[0] for b in band],
                "subset_hi": [b[1] for b in band],
                "peer_window": pwin,
                "peer_size": psize,
            },
            schema_overrides={
                "channels": pl.List(pl.String),
                "weights": pl.List(pl.Float64),
                "subset_member": pl.String,
            },
        ).with_columns(pl.col("subset_lo", "subset_hi").fill_nan(None))

    def _family_weights(self, fam: str, n_channels: int) -> tuple[IntArray, FloatArray]:
        """Return ``(operator indices, dense (n_channels, k) weight matrix)``."""
        idx = np.flatnonzero(self.family_ == fam).astype(np.int64)
        W = np.zeros((n_channels, idx.size))
        for j, i in enumerate(idx):
            used = self.channels_[i] >= 0
            W[self.channels_[i][used], j] = self.weights_[i][used]
        return idx, W

    # ------------------------------------------------------------------ #
    # transform
    # ------------------------------------------------------------------ #
    def _peer_min(self, window: int) -> int:
        if self.peer_min_periods is not None:
            return self.peer_min_periods
        return max(2, math.ceil(0.75 * window))

    def _peer_history(self, P: FloatArray, t: int, window: int) -> FloatArray:
        """Rows of the ``(T, N)`` ``peer_col`` history that date ``t`` may read.

        The dates strictly before ``t``, at most ``window`` of them. This is the
        **only** place the transform reads another date's data; the leakage
        tests replace it with a full-sample read and require the checks to fail.
        """
        return P[max(0, t - window) : t]

    def _peer_table(
        self,
        P: FloatArray,
        ents: IntArray,
        t: int,
        window: int,
        m_max: int,
        candidate: NDArray[np.bool_],
    ) -> tuple[IntArray, NDArray[np.bool_]] | None:
        """Top-``m_max`` most-correlated peers of every entity present at ``t``.

        Returns ``(idx, ok)``: ``idx[i, r]`` is the row (within the date) of
        entity ``i``'s ``r``-th peer by trailing correlation, and ``ok`` marks
        the usable slots. Missing observations inside the window are zero-filled
        after per-entity standardisation, which shrinks the correlations of
        entities with gaps rather than dropping them.
        """
        hist = self._peer_history(P, t, window)[:, ents]
        if hist.shape[0] == 0:
            return None
        valid = ~np.isnan(hist)
        n_obs = valid.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(hist, axis=0)
            sd = np.nanstd(hist, axis=0)
        good_sd = np.nan_to_num(sd) > _EPS
        Z = np.where(valid, (hist - mu) / np.where(good_sd, sd, 1.0), 0.0)
        Z[:, ~good_sd] = 0.0
        eligible = (n_obs >= self._peer_min(window)) & good_sd
        C = Z.T @ Z
        C = np.where((eligible & candidate)[None, :], C, -np.inf)
        np.fill_diagonal(C, -np.inf)
        m = min(m_max, C.shape[1])
        idx = np.argsort(-C, axis=1, kind="stable")[:, :m].astype(np.int64)
        ok = np.take_along_axis(C, idx, axis=1) > -np.inf
        ok[~eligible] = False
        return idx, ok

    def _pre_activations(
        self, X: FloatArray, ents: IntArray, t: int, P: FloatArray | None
    ) -> FloatArray:
        """All operators' pre-activations for one date's ``(n, C)`` block."""
        n = X.shape[0]
        mcs, clip = self.min_cross_section, self.clip
        pre = np.full((n, self.family_.size), np.nan)
        if n == 0:
            return pre
        U = _robust_z(X, clip, mcs)
        R = _uniform_rank(X, mcs)

        idx, W = self._fam_cache["median_dev"]
        if idx.size:
            Z = _robust_z(_combine(U, W), clip, mcs)
            pre[:, idx] = Z - self.threshold_[idx]

        idx, W = self._fam_cache["rank_threshold"]
        if idx.size:
            rho = _uniform_rank(_combine(R - 0.5, W), mcs)
            pre[:, idx] = rho - self.threshold_[idx]

        idx, W = self._fam_cache["subset_agg"]
        if idx.size:
            Z = _robust_z(_combine(U, W), clip, mcs)
            rank_m = R[:, self.member_[idx]]
            lo, hi = self.band_[idx, 0], self.band_[idx, 1]
            inside = (lo <= rank_m) & (rank_m <= hi) & ~np.isnan(Z)
            cnt = inside.sum(axis=0)
            total = np.where(inside, Z, 0.0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                agg = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
            pre[:, idx] = Z - agg - self.threshold_[idx]

        idx, W = self._fam_cache["peer_dev"]
        if idx.size and P is not None:
            Z = _robust_z(_combine(U, W), clip, mcs)
            candidate = ~np.isnan(X).any(axis=1)
            wins = self.peer_window_[idx]
            sizes = self.peer_size_[idx]
            for win in np.unique(wins):
                in_win = wins == win
                table = self._peer_table(
                    P, ents, t, int(win), int(sizes[in_win].max()), candidate
                )
                if table is None:
                    continue
                top, ok = table
                for size in np.unique(sizes[in_win]):
                    sel = np.flatnonzero(in_win & (sizes == size))
                    m = min(int(size), top.shape[1])
                    G = Z[:, sel][top[:, :m]]  # (n, m, g)
                    use = ok[:, :m, None] & ~np.isnan(G)
                    den = use.sum(axis=1)
                    num = np.where(use, G, 0.0).sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        peer_mean = np.where(den > 0, num / np.maximum(den, 1), np.nan)
                    pre[:, idx[sel]] = Z[:, sel] - peer_mean - self.threshold_[idx[sel]]
        return pre

    def _pool(self, pre: FloatArray) -> FloatArray:
        """Permutation-invariant per-date statistics, one row."""
        sub = pre[:, self._pooled_ops]
        finite = ~np.isnan(sub)
        cnt = finite.sum(axis=0)
        enough = cnt >= self.min_cross_section
        denom = np.maximum(cnt, 1)
        parts = []
        for stat in self.pool_stats:
            if stat == "ppv":
                val = (sub > 0.0).sum(axis=0) / denom
            else:
                act = _activate(sub, self.activation)
                val = np.where(finite, act, 0.0).sum(axis=0) / denom
            parts.append(np.where(enough, val, np.nan))
        return np.concatenate(parts)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        cols = self.feature_names_in_
        missing = [c for c in cols if c not in panel]
        needs_peer = self.peer_col is not None and "peer_dev" in self.families_
        if needs_peer and self.peer_col not in panel:
            missing.append(str(self.peer_col))
        if missing:
            raise ValueError(
                f"{type(self).__name__}.transform: column(s) {missing} not found in "
                f"the panel. Available columns: {panel.columns}."
            )
        ecol, tcol = panel.entity_col, panel.time_col
        full = panel.collect()
        if full.select(ecol, tcol).is_duplicated().any():
            raise ValueError(
                f"{type(self).__name__}.transform: duplicate ({ecol!r}, {tcol!r}) "
                "keys; a cross-section needs one row per entity per date."
            )
        n_rows = full.height
        n_ent_out = len(self.operator_names_)
        n_pool_out = len(self.pooled_names_)
        out = np.full((n_rows, len(self.output_names_)), np.nan)

        if n_rows:
            t_idx = (full[tcol].rank("dense").cast(pl.Int64) - 1).to_numpy()
            e_idx = (full[ecol].rank("dense").cast(pl.Int64) - 1).to_numpy()
            X = full.select(pl.col(c).cast(pl.Float64) for c in cols).to_numpy()
            X = np.ascontiguousarray(X, dtype=np.float64)
            T, N = int(t_idx.max()) + 1, int(e_idx.max()) + 1

            P: FloatArray | None = None
            if "peer_dev" in self.families_ and self.peer_col is not None:
                P = np.full((T, N), np.nan)
                P[t_idx, e_idx] = full[self.peer_col].cast(pl.Float64).to_numpy()

            order = np.lexsort((e_idx, t_idx))
            bounds = np.concatenate([[0], np.cumsum(np.bincount(t_idx, minlength=T))])
            want_ent = self.output in ("entity", "both")
            want_pool = self.output in ("pooled", "both")
            for t in range(T):
                rows = order[bounds[t] : bounds[t + 1]]
                if rows.size == 0:
                    continue
                pre = self._pre_activations(X[rows], e_idx[rows], t, P)
                col0 = 0
                if want_ent:
                    out[rows, :n_ent_out] = _activate(pre, self.activation)
                    col0 = n_ent_out
                if want_pool:
                    out[rows, col0 : col0 + n_pool_out] = self._pool(pre)[None, :]

        dtype = _DTYPES[self.dtype]
        new_cols: list[pl.Series]
        if self.as_array:
            arr = out.astype(np.float32 if self.dtype == "float32" else np.float64)
            new_cols = [pl.Series(self.prefix, arr).cast(pl.Array(dtype, arr.shape[1]))]
        else:
            frame = pl.from_numpy(out, schema=self.output_names_).fill_nan(None)
            new_cols = [frame[c].cast(dtype) for c in self.output_names_]
        base = full if self.keep_features else full.select(ecol, tcol)
        result = base.with_columns(new_cols)
        return PanelFrame(result, entity=ecol, time=tcol, validate=False)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def state(self) -> dict[str, Any]:
        """The complete fitted state, as plain arrays and lists.

        Two fits with the same parameters and the same input schema return equal
        states whatever the data -- the testable form of ``fit_is_empty``.

        Returns
        -------
        dict
            Column names, families and every drawn operator parameter.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted("state")
        return {
            "feature_names_in": list(self.feature_names_in_),
            "families": tuple(self.families_),
            "family": self.family_.copy(),
            "channels": self.channels_.copy(),
            "weights": self.weights_.copy(),
            "threshold": self.threshold_.copy(),
            "member": self.member_.copy(),
            "band": self.band_.copy(),
            "peer_window": self.peer_window_.copy(),
            "peer_size": self.peer_size_.copy(),
            "output_names": list(self.output_names_),
        }
