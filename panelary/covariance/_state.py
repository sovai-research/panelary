"""Market-state features from the per-date window spectrum (plan section 5.14).

Every feature is a function of one window's sample correlation (or
covariance) structure, computed from the shared :class:`WindowStats` -- one
Gram matrix and at most one ``eigvalsh`` per date and group:

==========================  ==================================================
``absorption_ratio``        ``sum_{i<=k} lambda_i / sum lambda_i`` (Kritzman et
                            al. 2011); ``k = ceil(0.2 * min(N_t, n_eff))`` or a
                            fixed ``ar_k``; emits ``ar_k``
``ar_shift``                ``(mean_15(AR) - mean_252(AR)) / sd_252(AR)``,
                            trailing windows on the AR series
``lambda1_share``           ``lambda_1 / tr``
``effective_rank``          ``exp(H)``, ``H = -sum p_i log p_i``,
                            ``p = lambda / sum lambda`` (Roy & Vetterli 2007)
``eigen_entropy``           ``H / log m`` over the ``m`` non-null eigenvalues
``participation_ratio``     ``tr^2 / ||S||_F^2`` -- from the Gram, no eigensolve
``mp_signal_count``         eigenvalues above the Tracy--Widom edge, with
                            ``mp_sigma2`` and ``mp_edge`` (``_rmt.mp_fit``)
``market_ipr``              ``sum_i v_1i^4`` in ``[1/N, 1]`` (Plerou et al.
                            2002); ``null`` when ``lambda_1 / lambda_2 <
                            min_gap``
``avg_corr``                mean off-diagonal entry of ``S``, by the exact
                            identity ``(||Z 1||^2 / n_eff - tr) / (N (N - 1))``
==========================  ==================================================

Diagnostics on every row: ``n_entities``, ``coverage`` (mean window coverage
of the universe), ``q = N_t / n_eff``, ``n_eff``, ``lambda_gap`` and
``asof_date`` (the evaluation date whose window produced the row -- earlier
than ``time`` on a strided schedule, so staleness is visible).

Group variants partition the universe of date ``t`` by the **date-t** value
of the group column; a member's historical labels inside the window are
ignored (trap T12).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.core._schedule import Schedule, as_schedule
from panelary.covariance._gram import SPACES
from panelary.covariance._rmt import mp_fit
from panelary.covariance._types import WindowStats
from panelary.covariance._window import (
    PanelMatrix,
    _as_pf,
    _need,
    panel_matrix,
    universe,
    window_at,
)

__all__ = [
    "DEFAULT_FEATURES",
    "FEATURES",
    "market_loading",
    "market_state",
    "spectrum_features",
    "top_eigenvector",
]

FEATURES = (
    "absorption_ratio",
    "ar_shift",
    "lambda1_share",
    "effective_rank",
    "eigen_entropy",
    "participation_ratio",
    "mp_signal_count",
    "mp_sigma2",
    "mp_edge",
    "market_ipr",
    "avg_corr",
)

DEFAULT_FEATURES = (
    "absorption_ratio",
    "ar_shift",
    "lambda1_share",
    "effective_rank",
    "participation_ratio",
    "mp_signal_count",
    "market_ipr",
    "avg_corr",
)

_DIAGNOSTICS = ("n_entities", "coverage", "q", "n_eff", "lambda_gap")
_POWER_MAX_ITER = 1000
_POWER_TOL = 1e-13
#: Power iteration only pays off on a large Gram with a well-separated top
#: eigenvalue (it needs ~30 / log10(gap) steps of ~20 us each); otherwise one
#: ``eigh`` of the min-side Gram is used.
_POWER_MIN_DIM = 64
_POWER_MIN_GAP = 2.0


# --------------------------------------------------------------------------- #
# per-window kernels
# --------------------------------------------------------------------------- #
def top_eigenvector(ws: WindowStats) -> NDArray[np.float64]:
    """The top primal eigenvector ``v_1`` (unit norm, entries summing >= 0).

    From the cached vector spectrum when there is one; otherwise by power
    iteration on the min-side Gram, seeded deterministically with the
    equal-weight portfolio (``Z 1`` on the dual, ``1`` on the primal). The
    caller decides whether ``lambda_1 / lambda_2`` is large enough for the
    vector to be meaningful.
    """
    spec = ws._spectrum
    G = ws.gram()
    lam = ws.spectrum().values
    gap = float(lam[0] / lam[1]) if lam.size > 1 and lam[1] > 0.0 else math.inf
    if spec is not None and spec.vectors is not None and spec.vectors.shape[1]:
        v = spec.vectors[:, 0].copy()
    elif G.shape[0] <= _POWER_MIN_DIM or gap < _POWER_MIN_GAP:
        # Small or poorly separated: one eigh is cheaper than the iterations.
        _, U = np.linalg.eigh(G)
        x = U[:, -1]
        v = ws.Z.T @ x if ws.dual else x
        v = v / float(np.linalg.norm(v))
    else:
        x = ws.Z.sum(axis=1) if ws.dual else np.ones(ws.p)
        nx = float(np.linalg.norm(x))
        if nx == 0.0:
            x = np.ones(G.shape[0])
            nx = float(np.linalg.norm(x))
        x = x / nx
        converged = False
        for _ in range(_POWER_MAX_ITER):
            y = G @ x
            ny = float(np.linalg.norm(y))
            if ny == 0.0:
                break
            y /= ny
            if float(np.abs(y - x).max()) < _POWER_TOL:
                x = y
                converged = True
                break
            x = y
        if not converged:  # pragma: no cover - only for near-degenerate gaps
            w, U = np.linalg.eigh(G)
            x = U[:, -1]
        v = ws.Z.T @ x if ws.dual else x
        v = v / float(np.linalg.norm(v))
    total = float(v.sum())
    if total < 0.0 or (total == 0.0 and v[int(np.argmax(np.abs(v)))] < 0.0):
        v = -v
    return v


def _trailing_mean_sd(
    a: NDArray[np.float64], L: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and sd (ddof=1) of each full trailing window of ``L`` rows.

    Two-pass per window over a ``sliding_window_view``: each output depends on
    its own ``L`` values only, never on how long the series is, so it is
    bitwise prefix-invariant (Polars' rolling kernels switch algorithm on the
    whole column's null count, which is not). NaN when the window is short
    or holds a NaN.
    """
    n = a.size
    mean = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    if n < L:
        return mean, sd
    win = np.lib.stride_tricks.sliding_window_view(a, L)
    m = win.mean(axis=1)
    mean[L - 1 :] = m
    if L > 1:
        sd[L - 1 :] = np.sqrt(((win - m[:, None]) ** 2).sum(axis=1) / (L - 1))
    return mean, sd


def _ar_shift(ar: NDArray[np.float64], short: int, long: int) -> NDArray[np.float64]:
    """``(mean_short(AR) - mean_long(AR)) / sd_long(AR)``, trailing, per row."""
    s_mean, _ = _trailing_mean_sd(ar, short)
    l_mean, l_sd = _trailing_mean_sd(ar, long)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s_mean - l_mean) / l_sd
    out[~np.isfinite(out)] = np.nan
    return out


def _ar_k(p: int, n_eff: float, rank: int, ar_k: int | None) -> int:
    if ar_k is not None:
        return max(1, min(int(ar_k), rank))
    return max(1, min(math.ceil(0.2 * min(p, n_eff)), rank))


def spectrum_features(
    ws: WindowStats,
    features: Sequence[str],
    *,
    ar_k: int | None = None,
    edge: str = "tw",
    alpha: float = 0.95,
    min_gap: float = 1.05,
) -> dict[str, float | None]:
    """All requested features and diagnostics of one window.

    ``ar_shift`` is a function of the AR *series* and is filled in by
    :func:`market_state`, not here.
    """
    feats = set(features)
    out: dict[str, float | None] = {}
    p, n_eff = ws.p, ws.n_eff
    Z = ws.Z
    tr = float(np.einsum("ij,ij->", Z, Z)) / n_eff
    out["n_entities"] = float(p)
    out["coverage"] = float(ws.coverage.mean())
    out["q"] = p / n_eff
    out["n_eff"] = n_eff
    lam = ws.spectrum().values  # lambda_gap is always reported
    rank = int(np.count_nonzero(lam > 0.0))
    gap = float(lam[0] / lam[1]) if lam.size > 1 and lam[1] > 0.0 else math.inf
    out["lambda_gap"] = gap
    if "absorption_ratio" in feats or "ar_shift" in feats:
        k = _ar_k(p, n_eff, rank, ar_k)
        out["absorption_ratio"] = float(lam[:k].sum()) / tr
        out["ar_k"] = float(k)
    if "lambda1_share" in feats:
        out["lambda1_share"] = float(lam[0]) / tr
    if "effective_rank" in feats or "eigen_entropy" in feats:
        pos = lam[:rank]
        pr = pos / pos.sum()
        H = float(-(pr * np.log(pr)).sum())
        if "effective_rank" in feats:
            out["effective_rank"] = math.exp(H)
        if "eigen_entropy" in feats:
            out["eigen_entropy"] = H / math.log(rank) if rank > 1 else 0.0
    if "participation_ratio" in feats:
        G = ws.gram()
        f2 = float(np.einsum("ij,ij->", G, G))
        out["participation_ratio"] = tr * tr / f2
    if feats & {"mp_signal_count", "mp_sigma2", "mp_edge"}:
        fit = mp_fit(lam, p, n_eff, edge=edge, alpha=alpha)
        out["mp_signal_count"] = fit["n_signal"]
        out["mp_sigma2"] = fit["sigma2"]
        out["mp_edge"] = fit["edge"]
    if "market_ipr" in feats:
        if gap >= min_gap:
            v = top_eigenvector(ws)
            out["market_ipr"] = float(np.sum(v**4))
        else:
            out["market_ipr"] = None
    if "avg_corr" in feats:
        if p < 2:
            out["avg_corr"] = None
        else:
            u = Z.sum(axis=1)
            out["avg_corr"] = (float(u @ u) / n_eff - tr) / (p * (p - 1))
    return out


# --------------------------------------------------------------------------- #
# the per-date driver
# --------------------------------------------------------------------------- #
def _resolve_schedule(
    stride: int | None, schedule: Schedule | int | str | None
) -> Schedule:
    if schedule is not None:
        return as_schedule(schedule)
    return as_schedule(1 if stride is None else int(stride))


def _units(
    pm: PanelMatrix, s: int, idx: NDArray[np.intp]
) -> list[tuple[int | None, NDArray[np.intp]]]:
    if pm.groups is None:
        return [(None, idx)]
    codes = pm.groups[s, idx]
    return [(int(c), idx[codes == c]) for c in np.unique(codes[codes >= 0])]


def _validate_common(window: int, min_coverage: float, space: str) -> None:
    if int(window) < 2:
        raise ValueError(f"`window` must be >= 2, got {window}.")
    _need(int(window), min_coverage)
    if space not in SPACES:
        raise ValueError(f"`space` must be one of {SPACES}, got {space!r}.")


def _broadcast(
    panel: Any,
    state: pl.DataFrame,
    *,
    entity: str | None,
    time: str | None,
    on: list[str],
) -> pl.DataFrame:
    pf = _as_pf(panel, entity, time)
    df = pf.collect()
    clash = [c for c in state.columns if c in df.columns and c not in on]
    if clash:
        raise ValueError(
            f"broadcast would overwrite panel column(s) {clash}; rename them first."
        )
    idx = "__cov_row__"
    return df.with_row_index(idx).join(state, on=on, how="left").sort(idx).drop(idx)


def market_state(
    panel: Any,
    *,
    returns: str,
    window: int = 252,
    stride: int | None = 1,
    schedule: Schedule | int | str | None = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    group: str | None = None,
    broadcast: bool = False,
    min_coverage: float = 0.95,
    min_entities: int | None = None,
    space: str = "correlation",
    ar_k: int | None = None,
    ar_short: int = 15,
    ar_long: int = 252,
    edge: str = "tw",
    alpha: float = 0.95,
    min_gap: float = 1.05,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Per-date market-state features from trailing-window spectra.

    Parameters
    ----------
    panel : PanelFrame or polars frame
        Long panel of returns.
    returns : str
        Return column.
    window : int, default 252
        Trailing window ``W``: the value as of ``t`` uses rows ``(t-W, t]``.
    stride : int, default 1
        Evaluate every ``stride``-th date (anchored at the first date) and
        carry each value forward to the following dates (as-of forward fill).
    schedule : Schedule, int or str, optional
        An explicit evaluation schedule (overrides ``stride``), e.g. ``"1w"``
        or ``"1mo"`` for the first date of each week / month.
    features : sequence of str
        Any of :data:`FEATURES`.
    group : str, optional
        Group column (e.g. sector): one row per ``(time, group)``, each group
        estimated on its own members as of ``t``.
    broadcast : bool, default False
        Join the per-date values onto the panel's rows (on ``time``, and the
        row's own date-``t`` group value for a group variant).
    min_coverage : float, default 0.95
        Universe admission (see :func:`rolling`).
    min_entities : int, optional
        Smallest universe (or group) evaluated: default 2, or 10 with
        ``group``.
    space : {"correlation", "covariance"}, default "correlation"
    ar_k : int, optional
        Fixed number of eigenvalues in the absorption ratio (recommended when
        ``N_t`` varies); default ``ceil(0.2 * min(N_t, n_eff))``.
    ar_short, ar_long : int, default 15, 252
        ``ar_shift`` windows, in dates (rows of the output series).
    edge, alpha : MP edge options (``"tw"`` at 95% by default).
    min_gap : float, default 1.05
        ``market_ipr`` is null when ``lambda_1 / lambda_2`` is below this:
        the market mode is not identified.

    Returns
    -------
    polars.DataFrame
        ``(time[, group], features..., diagnostics..., asof_date)`` sorted by
        ``time`` -- or, with ``broadcast=True``, the panel's rows with those
        columns joined on.

    Notes
    -----
    Nothing here depends on the panel's length, its last date or its total
    entity count: ``N_t``, ``n_eff``, ``k`` and the MP noise level are all
    window-local, and the evaluation grid is anchored at the first date or at
    period starts. The output is bit-identical under appending data
    (``tests/test_covariance_prefix.py``).
    """
    feats = tuple(dict.fromkeys(features))
    bad = [f for f in feats if f not in FEATURES]
    if bad:
        raise ValueError(f"unknown feature(s) {bad}; expected any of {FEATURES}.")
    if not feats:
        raise ValueError("`features` must name at least one feature.")
    _validate_common(window, min_coverage, space)
    window = int(window)
    if min_entities is None:
        min_entities = 10 if group is not None else 2
    sched = _resolve_schedule(stride, schedule)
    pm = panel_matrix(panel, returns, entity=entity, time=time, group=group)
    tcol = pm.time_col
    grid = sched.positions(pm.times)

    kernel_feats = [f for f in feats if f != "ar_shift"]
    if "ar_shift" in feats and "absorption_ratio" not in kernel_feats:
        kernel_feats.append("absorption_ratio")
    out_cols = [f for f in FEATURES if f in kernel_feats and f != "ar_shift"]
    if "absorption_ratio" in kernel_feats:
        out_cols.append("ar_k")
    for extra in ("mp_sigma2", "mp_edge"):
        if extra in out_cols and extra not in feats:
            out_cols.remove(extra)
    cols: dict[str, list[Any]] = {c: [] for c in [*out_cols, *_DIAGNOSTICS]}
    pos_col: list[int] = []
    code_col: list[int] = []
    for s in grid:
        s = int(s)
        idx_all = universe(pm, s, window, min_coverage)
        for code, idx in _units(pm, s, idx_all):
            ws, idx2 = window_at(
                pm,
                s,
                window=window,
                min_coverage=min_coverage,
                space=space,
                idx=idx,
                min_entities=min_entities,
            )
            if ws is None:
                if pm.groups is not None:
                    continue  # a group too small at s has no row
                vals: dict[str, float | None] = {"n_entities": float(idx2.size)}
            else:
                vals = spectrum_features(
                    ws, kernel_feats, ar_k=ar_k, edge=edge, alpha=alpha, min_gap=min_gap
                )
            for c in cols:
                cols[c].append(vals.get(c))
            pos_col.append(s)
            code_col.append(-1 if code is None else code)

    schema: dict[str, Any] = dict.fromkeys(cols, pl.Float64)
    grid_frame = pl.DataFrame(
        {"__pos__": pl.Series(pos_col, dtype=pl.Int64), **cols}, schema_overrides=schema
    )
    if pm.groups is not None:
        labels = pl.Series(pm.group_values)
        grid_frame = grid_frame.with_columns(
            pl.Series(group, [pm.group_values[c] for c in code_col], dtype=labels.dtype)
        )
    grid_frame = grid_frame.with_columns(
        pl.col("n_entities").cast(pl.Int64),
        *([pl.col("ar_k").cast(pl.Int64)] if "ar_k" in cols else []),
        *(
            [pl.col("mp_signal_count").cast(pl.Int64)]
            if "mp_signal_count" in cols
            else []
        ),
    )
    asof = sched.asof_positions(pm.times)
    daily = pl.DataFrame({tcol: pm.times, "__pos__": pl.Series(asof, dtype=pl.Int64)})
    if pm.groups is not None:
        state = daily.join(grid_frame, on="__pos__", how="inner")
    else:
        state = daily.join(grid_frame, on="__pos__", how="left")
    pos_np = state.get_column("__pos__").to_numpy()
    state = (
        state.with_columns(pm.times.gather(np.clip(pos_np, 0, None)).alias("asof_date"))
        .with_columns(
            pl.when(pl.col("__pos__") >= 0)
            .then(pl.col("asof_date"))
            .otherwise(None)
            .alias("asof_date")
        )
        .drop("__pos__")
    )
    keys = [tcol] if group is None else [tcol, group]
    state = state.sort(keys)
    if "ar_shift" in feats:
        ar = state.get_column("absorption_ratio").fill_null(np.nan).to_numpy()
        shift = np.full(ar.size, np.nan)
        if group is None:
            shift[:] = _ar_shift(ar, ar_short, ar_long)
        else:
            gcol = state.get_column(group).to_numpy()
            for g in dict.fromkeys(gcol.tolist()):
                rows = np.flatnonzero(gcol == g)
                shift[rows] = _ar_shift(ar[rows], ar_short, ar_long)
        state = state.with_columns(
            pl.Series("ar_shift", shift, dtype=pl.Float64).fill_nan(None)
        )
        if "absorption_ratio" not in feats:
            state = state.drop(["absorption_ratio", "ar_k"])
    ordered = [
        *keys,
        *[c for c in FEATURES if c in state.columns],
        *[c for c in ("ar_k", *_DIAGNOSTICS, "asof_date") if c in state.columns],
    ]
    state = state.select(ordered)
    if broadcast:
        return _broadcast(panel, state, entity=entity, time=time, on=keys)
    return state


def market_loading(
    panel: Any,
    *,
    returns: str,
    window: int = 252,
    stride: int | None = 1,
    schedule: Schedule | int | str | None = None,
    min_coverage: float = 0.95,
    min_entities: int | None = None,
    group: str | None = None,
    space: str = "correlation",
    min_gap: float = 1.05,
    name: str = "market_loading",
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Per-entity loading on the market mode, as of each date.

    ``v_1i`` of the window's top eigenvector, oriented so that the loadings
    sum to a positive number ("market up" is positive; trap T13's sign
    guard). Null when the entity is outside the as-of universe or when
    ``lambda_1 / lambda_2 < min_gap``. With ``group``, each entity's loading
    is on the market mode of its date-``s`` group (groups with at least
    ``min_entities`` members; default 2, or 10 with ``group``).

    Returns
    -------
    polars.DataFrame
        The panel's ``(entity, time)`` keys with ``name`` and ``asof_date``.
    """
    _validate_common(window, min_coverage, space)
    window = int(window)
    sched = _resolve_schedule(stride, schedule)
    if min_entities is None:
        min_entities = 10 if group is not None else 2
    pm = panel_matrix(panel, returns, entity=entity, time=time, group=group)
    T, N = pm.R.shape
    L = np.full((T, N), np.nan)
    grid = sched.positions(pm.times)
    load_at: dict[int, NDArray[np.float64]] = {}
    for s in grid:
        s = int(s)
        row = np.full(N, np.nan)
        idx_all = universe(pm, s, window, min_coverage)
        for _code, members in _units(pm, s, idx_all):
            ws, idx = window_at(
                pm, s, window=window, min_coverage=min_coverage, space=space,
                idx=members, min_entities=min_entities,
            )  # fmt: skip
            if ws is None:
                continue
            lam = ws.spectrum().values
            gap = float(lam[0] / lam[1]) if lam.size > 1 and lam[1] > 0 else math.inf
            if gap >= min_gap:
                row[idx] = top_eigenvector(ws)
        load_at[s] = row
    asof = sched.asof_positions(pm.times)
    for t in range(T):
        s = int(asof[t])
        if s >= 0:
            L[t] = load_at[s]
    ent, tcol = pm.entity_col, pm.time_col
    grid_long = pl.DataFrame(
        {
            ent: pm.entities.gather(np.repeat(np.arange(N), T)),
            tcol: pm.times.gather(np.tile(np.arange(T), N)),
            name: L.T.reshape(-1),
            "__pos__": np.tile(asof, N),
        }
    )
    pos_np = grid_long.get_column("__pos__").to_numpy()
    grid_long = (
        grid_long.with_columns(
            pm.times.gather(np.clip(pos_np, 0, None)).alias("asof_date"),
            pl.col(name).fill_nan(None),
        )
        .with_columns(
            pl.when(pl.col("__pos__") >= 0)
            .then(pl.col("asof_date"))
            .otherwise(None)
            .alias("asof_date")
        )
        .drop("__pos__")
    )
    pf = _as_pf(panel, entity, time)
    keys = pf.collect().select(ent, tcol).with_row_index("__cov_row__")
    return (
        keys.join(grid_long, on=[ent, tcol], how="left")
        .sort("__cov_row__")
        .drop("__cov_row__")
    )
