"""Dependence matrices, distances, and leak-safe feature screening.

:func:`dependence_matrix` dispatches to the **batched** matrix kernels -- ``p``
argsorts for xi (:func:`~panelary.depend.xi_matrix`), one covariance of normal
scores for GCMI, one matmul for tail dependence, one row-batched call over all
pairs otherwise -- never a ``p^2`` Python loop. A matrix does not need
``n = 100 000`` rows to rank features, so it subsamples (seeded, recorded as
``approximate=True``).

:func:`feature_screen` is the headline function: a ranked table of features
against a target, each with an honest p-value and a multiplicity-adjusted one.
**A dependence estimate is fit-free; a dependence-based selection is not.**
Running :func:`feature_screen` on the full panel and then using its survivors
inside cross-validation is a leak -- the single most common misuse of this
module. :class:`ScreenSelector` is the leak-safe form: it re-runs the screen
inside ``fit`` on the training fold only.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

# `psd_repair` moved to `_internal/_linalg.py` (coordination note D6); it is
# re-exported here unchanged so `panelary.depend.psd_repair` keeps working.
from panelary._internal._linalg import psd_repair
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.depend._coef import tail_dependence_matrix, xi_matrix
from panelary.depend._engine import (
    DEFAULT_RESAMPLES,
    _check_by,
    _sample_rows,
    _shift_pair,
    aggregate_estimates,
    devolatilise,
    panel_block_length,
    panel_serial,
    rows_chunked,
    run_pair,
)
from panelary.depend._frame import _resolve, extract, result_frame
from panelary.depend._info import gcmi_matrix
from panelary.depend._kernels import get_kernel
from panelary.depend._lag import adjust_pvalues
from panelary.depend._null import SERIAL_THRESHOLD
from panelary.depend._ranks import ranks

__all__ = [
    "ScreenSelector",
    "dependence_matrix",
    "feature_screen",
    "matrix_values",
    "psd_repair",
    "to_distance",
]

_SYMMETRISE = {None, "max", "mean"}
_KINDS = frozenset({"angular", "absolute", "info"})


# --------------------------------------------------------------------------- #
# Matrix kernels
# --------------------------------------------------------------------------- #
def _matrix_block(
    method: str, A: np.ndarray, *, q: float, seed: int, n_features: int
) -> np.ndarray:
    """``(p, p)`` statistic matrix of the complete rows of ``A`` (``(n, p)``)."""
    n, p = A.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "xi":
            return xi_matrix(A)
        if method == "spearman":
            return np.atleast_2d(np.corrcoef(ranks(A, axis=0), rowvar=False))
        if method == "pearson":
            return np.atleast_2d(np.corrcoef(A, rowvar=False))
        if method == "gcmi":
            return gcmi_matrix(A)
        if method in {"tail_lower", "tail_upper"}:
            return tail_dependence_matrix(A, q=q, side=method.split("_")[1])
        if method == "hsic":
            from panelary.depend._kernel import hsic_matrix

            return hsic_matrix(A, n_features=n_features, seed=seed)
    kernel = get_kernel(method, q=q)
    out = np.full((p, p), np.nan)
    iu = np.triu_indices(p, 1)
    if iu[0].size:
        vals = rows_chunked(kernel.rows, A[:, iu[0]].T, A[:, iu[1]].T)
        out[iu] = vals
        out[iu[1], iu[0]] = vals
    out[np.diag_indices(p)] = rows_chunked(kernel.rows, A.T, A.T)
    return out


def _rows_subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray | None:
    if cap <= 0 or n <= cap:
        return None
    return np.sort(rng.choice(n, size=int(cap), replace=False))


def dependence_matrix(
    df: Any,
    cols: Sequence[str],
    *,
    method: str = "xi",
    by: str | None = "entity",
    entity: str | None = None,
    time: str | None = None,
    subsample: int = 5000,
    max_entities: int = 50,
    seed: int = 0,
    symmetrise: str | None = None,
    how: str | None = None,
    q: float = 0.05,
    n_features: int = 64,
    output: str = "long",
) -> pl.DataFrame:
    """``p x p`` dependence matrix of ``cols``.

    Directed for xi (``M[i, j] = xi(cols[i] -> cols[j])``), symmetric
    otherwise. ``symmetrise`` in ``{None, "max", "mean"}``; the default
    ``None`` **preserves direction**.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
    cols : sequence of str
        Columns (at least 2).
    method : str, default="xi"
        Any :func:`~panelary.depend.dependence` method except
        ``"tail_dependence"`` (pick a side).
    by : {"entity", "pooled"}, default="entity"
        ``"entity"``: one matrix per entity (listwise-complete rows), aggregated
        elementwise with the method's rule (Fisher-weighted by ``n_i`` for
        correlation-like statistics). ``"pooled"``: one matrix on all rows.
    subsample : int, default=5000
        Row cap per matrix (per entity for ``by="entity"``); a seeded subsample
        above it, recorded as ``approximate=True``. ``0`` disables.
    max_entities : int, default=50
        Entity cap for ``by="entity"`` (seeded choice; ``approximate=True``).
    seed : int, default=0
        Seed for subsampling (and the random features of ``"hsic"``).
    symmetrise : {None, "max", "mean"}, default=None
    how : str, optional
        Entity aggregation (method default).
    q : float, default=0.05
        Tail probability for tail statistics.
    n_features : int, default=64
        Random Fourier features for ``"hsic"`` (``D = 32-64`` for matrices).
    output : {"long", "wide"}, default="long"
        ``"long"``: ``p^2`` rows (``x``, ``y`` + the fixed schema).
        ``"wide"``: a ``feature`` column plus one column per ``col``.

    Returns
    -------
    polars.DataFrame
    """
    cols = list(dict.fromkeys(cols))
    if len(cols) < 2:
        raise ValueError("dependence_matrix needs at least 2 columns.")
    if symmetrise not in _SYMMETRISE:
        raise ValueError(
            f"unknown `symmetrise` {symmetrise!r}; expected None, 'max' or 'mean'."
        )
    if output not in {"long", "wide"}:
        raise ValueError(f"unknown `output` {output!r}; expected 'long' or 'wide'.")
    if method == "tail_dependence":
        raise ValueError("pick a tail: method='tail_lower' or 'tail_upper'.")
    by_ = _check_by(by)
    kernel = get_kernel(method, q=q) if method != "hsic" else None
    directed = method == "xi"
    pa = extract(df, cols, entity=entity, time=time)
    data = np.column_stack([pa.values[c] for c in cols])
    rng = np.random.default_rng(seed)
    approximate = False
    notes: list[str] = []
    if by_ == "pooled" or pa.n_entities == 1:
        A = data[np.isfinite(data).all(axis=1)]
        sub = _rows_subsample(A.shape[0], subsample, rng)
        if sub is not None:
            A = A[sub]
            approximate = True
        M = _matrix_block(method, A, q=q, seed=seed, n_features=n_features)
        n_obs, n_ent = int(A.shape[0]), int(pa.n_entities)
        how_used = None
    else:
        ent_ids: NDArray[np.signedinteger[Any]] = np.arange(pa.n_entities)
        if pa.n_entities > max_entities > 0:
            ent_ids = np.sort(
                rng.choice(pa.n_entities, size=int(max_entities), replace=False)
            )
            approximate = True
            notes.append(
                f"{len(ent_ids)} of {pa.n_entities} entities sampled (seed={seed})."
            )
        mats, counts = [], []
        min_n = get_kernel(method, q=q).min_obs if method != "hsic" else 30
        for i in ent_ids:
            A = data[pa.slice(int(i))]
            A = A[np.isfinite(A).all(axis=1)]
            if A.shape[0] < min_n:
                continue
            sub = _rows_subsample(A.shape[0], subsample, rng)
            if sub is not None:
                A = A[sub]
                approximate = True
            mats.append(_matrix_block(method, A, q=q, seed=seed, n_features=n_features))
            counts.append(A.shape[0])
        if not mats:
            M = np.full((len(cols), len(cols)), np.nan)
            n_obs = n_ent = 0
            notes.append("no entity reached the minimum number of complete rows.")
            how_used = None
        else:
            stack = np.moveaxis(np.stack(mats), 0, -1)  # (p, p, N)
            cnt = np.broadcast_to(np.asarray(counts, dtype=np.float64), stack.shape)
            kern = kernel if kernel is not None else get_kernel("pearson")
            how_used = how or (kernel.default_how if kernel is not None else "mean")
            M = aggregate_estimates(stack, cnt, kern, how_used)
            if method == "gcmi":
                np.fill_diagonal(M, np.inf)
            n_obs, n_ent = int(sum(counts)), len(counts)
    transform = "none"
    if symmetrise == "max":
        M = np.fmax(M, M.T)
        transform = "symmetrised(max)"
    elif symmetrise == "mean":
        M = 0.5 * (M + M.T)
        transform = "symmetrised(mean)"
    if output == "wide":
        return pl.DataFrame(
            {"feature": cols, **{c: M[:, j] for j, c in enumerate(cols)}},
            schema={"feature": pl.Utf8, **dict.fromkeys(cols, pl.Float64)},
        )
    label = kernel.estimator if kernel is not None else "rff-hsic (normalised)"
    if how_used is not None:
        label = f"{label}; aggregate={how_used}"
    direction = "x->y" if directed and symmetrise is None else "symmetric"
    rows = [
        {
            "x": cols[i],
            "y": cols[j],
            "estimate": float(M[i, j]),
            "method": method,
            "estimator": label,
            "direction": direction,
            "n_obs": n_obs,
            "n_entities": n_ent,
            "transform": transform,
            "approximate": approximate,
            "seed": int(seed) if approximate or method == "hsic" else None,
            "warnings": notes,
        }
        for i in range(len(cols))
        for j in range(len(cols))
    ]
    return result_frame(rows, keys={"x": pl.Utf8, "y": pl.Utf8})


def matrix_values(result: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """``(M, names)`` from a :func:`dependence_matrix` result (long or wide)."""
    if "feature" in result.columns:
        names = result.get_column("feature").to_list()
        return result.select(names).to_numpy().astype(np.float64), names
    names = list(dict.fromkeys(result.get_column("x").to_list()))
    pos = {c: i for i, c in enumerate(names)}
    M = np.full((len(names), len(names)), np.nan)
    for xv, yv, e in result.select("x", "y", "estimate").iter_rows():
        M[pos[xv], pos[yv]] = np.nan if e is None else e
    return M, names


def to_distance(
    M: np.ndarray, *, kind: str = "angular", repair: bool = False
) -> np.ndarray:
    """Turn a symmetric dependence matrix into a distance matrix.

    Feeds :mod:`panelary.cluster` (hierarchical / ONC codependence clustering)
    and ``select.pfa`` directly.

    Parameters
    ----------
    M : array_like
        Symmetric ``(p, p)`` matrix. A directed (xi) matrix must be symmetrised
        first (``dependence_matrix(..., symmetrise="max")``) -- the choice is
        the caller's, not a silent default.
    kind : {"angular", "absolute", "info"}, default="angular"
        ``"angular"``: ``sqrt(0.5 (1 - M))`` for signed correlations.
        ``"absolute"``: ``sqrt(1 - |M|)`` for unsigned measures (xi, dcor,
        Hoeffding, normalised HSIC). ``"info"``: ``exp(-M)`` =
        ``sqrt(1 - r_eq^2)`` for an MI matrix in nats, ``r_eq`` the
        Gaussian-equivalent correlation (``mi_to_r``); the exact variation of
        information needs the raw data (``variation_of_information``).
    repair : bool, default=False
        For ``"angular"`` / ``"absolute"``: project to the nearest PSD
        correlation matrix first (:func:`psd_repair`), warning with the
        smallest eigenvalue found.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` float64 with a zero diagonal.

    Raises
    ------
    ValueError
        For an asymmetric matrix or an unknown ``kind``.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown `kind` {kind!r}; expected one of {sorted(_KINDS)}.")
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"`M` must be square, got shape {A.shape}.")
    off = ~np.eye(A.shape[0], dtype=bool)
    fin = np.isfinite(A) & np.isfinite(A.T) & off
    if not np.allclose(A[fin], A.T[fin], atol=1e-12, equal_nan=True):
        raise ValueError(
            "`M` is not symmetric (a directed xi matrix?). Symmetrise it first, "
            "e.g. dependence_matrix(..., symmetrise='max')."
        )
    if kind == "info":
        D = np.exp(-np.clip(A, 0.0, None))
    else:
        if repair:
            R, lam = psd_repair(A)
            if lam < 0:
                warnings.warn(
                    f"dependence matrix was not PSD (smallest eigenvalue {lam:.3g}); "
                    "repaired by eigenvalue clipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            A = R
        if kind == "angular":
            D = np.sqrt(np.clip(0.5 * (1.0 - A), 0.0, 1.0))
        else:
            D = np.sqrt(np.clip(1.0 - np.abs(A), 0.0, 1.0))
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------------- #
# Feature screening
# --------------------------------------------------------------------------- #
def _numeric_features(frame: pl.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c
        for c, dt in frame.schema.items()
        if c not in exclude and dt.is_numeric() and dt != pl.Boolean
    ]


def feature_screen(
    df: Any,
    target: str,
    features: Sequence[str] | None = None,
    *,
    method: str = "xi",
    by: str | None = "entity",
    null: str = "auto",
    correction: str = "benjamini_hochberg",
    entity: str | None = None,
    time: str | None = None,
    lag: int = 0,
    how: str | None = None,
    devol: int | None = None,
    demean: str = "none",
    q: float = 0.05,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Rank ``features`` by their dependence with ``target``, with honest,
    multiplicity-adjusted p-values.

    Every feature is tested with the same null (the null policy of
    :func:`~panelary.depend.dependence`: common-time for a panel) and, for
    resampling nulls, the **same** resample draws and block length, so the
    ``(B x F)`` null matrix is joint and ``correction="romano_wolf"`` is valid.

    **Leakage.** The estimate is fit-free; the *selection* is not. Do not
    screen the full panel and then cross-validate on the survivors -- use
    :class:`ScreenSelector`, which re-screens inside every training fold.

    Parameters
    ----------
    df : PanelFrame | polars.DataFrame | polars.LazyFrame
    target : str
        The target column (``feature -> target`` for directed methods: "is the
        target a function of the feature").
    features : sequence of str, optional
        Candidates; default every numeric column except the keys and target.
    method, by, null, entity, time, lag, how, devol, demean, q, n_resamples, block_length, seed, min_obs
        As in :func:`~panelary.depend.dependence`. With ``devol`` every feature
        gets two rows (raw and devolatilised).
    correction : {"benjamini_hochberg", "benjamini_yekutieli", "holm", "romano_wolf", "none"}
        Multiplicity correction across features (per ``transform``).
    alpha : float, default=0.05
        Used only to warn when ``n_resamples`` is too small for any
        discovery to survive the correction.

    Returns
    -------
    polars.DataFrame
        ``feature``, ``target`` and the fixed schema, sorted by ``p_value_adj``,
        then ``p_value``, then strength.
    """
    frame, ent, tme = _resolve(df, entity, time)
    keys = {k for k in (ent, tme) if k}
    feats = (
        list(features)
        if features is not None
        else _numeric_features(frame, keys | {target})
    )
    feats = [f for f in dict.fromkeys(feats) if f != target]
    if not feats:
        raise ValueError("no candidate features to screen.")
    by_ = _check_by(by)
    kernel = get_kernel(method, q=q)
    if method == "tail_dependence":
        raise ValueError("pick a tail: method='tail_lower' or 'tail_upper'.")
    pa = extract(frame, [*feats, target], entity=ent, time=tme)
    Yraw = pa.dense(target)
    variants: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = [
        ("none", {f: pa.dense(f) for f in feats}, Yraw)
    ]
    if devol is not None:
        variants.append(
            (
                f"devol(window={int(devol)})",
                {f: devolatilise(pa.dense(f), devol) for f in feats},
                devolatilise(Yraw, devol),
            )
        )
    out_rows: list[dict[str, Any]] = []
    for tname, Xd, Y in variants:
        shifted = {f: _shift_pair(Xd[f], Y, lag) for f in feats}
        n_ent = Y.shape[0]
        # One block length for the whole screen, so every feature sees the
        # same resample indices (a joint null for Romano-Wolf).
        blen = block_length
        if blen is None and n_ent >= 2 and null in {"auto", "common-time"}:
            rows_s = _sample_rows(n_ent)
            blen_vals = [
                panel_block_length(Xs, Ys, rows_s, None)
                if panel_serial(Xs, Ys, rows_s) > SERIAL_THRESHOLD
                else 1
                for Xs, Ys in shifted.values()
            ]
            blen = int(max(blen_vals)) if blen_vals else None
        rows: list[dict[str, Any]] = []
        for f in feats:
            Xs, Ys = shifted[f]
            row = run_pair(
                pa,
                Xs,
                Ys,
                kernel,
                by=by_,
                null=null,
                how=how,
                demean=demean,
                n_resamples=n_resamples,
                block_length=blen,
                seed=seed,
                min_obs=min_obs,
            )
            row.update(feature=f, target=target, lag=int(lag))
            if tname != "none":
                base = row.get("transform") or "none"
                row["transform"] = tname if base == "none" else f"{base}; {tname}"
            rows.append(row)
        pvals = np.array([r.get("p_value", np.nan) for r in rows], dtype=np.float64)
        stats = np.array([r.get("estimate", np.nan) for r in rows], dtype=np.float64)
        draws = None
        schemes = {r.get("null_method") for r in rows}
        if correction == "romano_wolf":
            dl = [np.asarray(r["_draws"]) for r in rows if r.get("_draws") is not None]
            joint = (
                len(dl) == len(rows)
                and len({d.shape for d in dl}) == 1
                and len(schemes) == 1
                and (schemes <= {"common-time", "entity"} or n_ent == 1)
            )
            if joint:
                draws = np.column_stack([np.asarray(d, dtype=np.float64) for d in dl])
        adj, label = adjust_pvalues(
            pvals,
            correction=correction,
            stats=stats,
            draws=draws,
            two_sided=kernel.alternative == "two-sided",
        )
        screen_notes = [f"p_value_adj: {label} across {len(feats)} features."]
        if label != correction:
            screen_notes.append(
                f"{correction} needs one joint resample matrix across features, not "
                f"available for this null; used {label}."
            )
        b_used = [int(r["n_resamples"]) for r in rows if r.get("n_resamples")]
        if b_used and label in {"benjamini_hochberg", "benjamini_yekutieli", "holm"}:
            floor = len(feats) / (min(b_used) + 1.0)
            if label == "benjamini_yekutieli":
                floor *= sum(1.0 / i for i in range(1, len(feats) + 1))
            if floor > alpha:
                need = math.ceil(len(feats) / alpha)
                screen_notes.append(
                    f"with n_resamples={min(b_used)} the smallest achievable adjusted "
                    f"p-value is {min(floor, 1.0):.3g} > alpha={alpha}: no feature can be "
                    f"selected. Use n_resamples >= {need}."
                )
        for r, a in zip(rows, adj, strict=True):
            r["p_value_adj"] = float(a)
            r["warnings"] = [*r.get("warnings", []), *screen_notes]
        out_rows += rows
    res = result_frame(out_rows, keys={"feature": pl.Utf8, "target": pl.Utf8})
    strength = (
        pl.col("estimate").abs()
        if kernel.alternative == "two-sided"
        else pl.col("estimate")
    )
    return (
        res.with_columns(
            pl.col("p_value_adj").fill_nan(None).alias("_pa"),
            pl.col("p_value").fill_nan(None).alias("_p"),
            (-strength).fill_nan(None).alias("_s"),
        )
        .sort(["transform", "_pa", "_p", "_s"], nulls_last=True)
        .drop(["_pa", "_p", "_s"])
    )


class ScreenSelector(PanelTransformer):
    """Keep the features that survive :func:`feature_screen` -- re-screened per fold.

    ``fit`` runs the screen **on the training panel only** and freezes the
    surviving names; ``transform`` projects any panel onto them (+ the target
    when present). That, and only that, is why ``leakage_safe = True``: a
    screen run once on the full panel and reused inside cross-validation leaks
    the test folds into the selection (``tests/test_depend_leakage.py``
    measures by how much).

    Parameters
    ----------
    target : str
        Target column.
    features : sequence of str, optional
        Candidate pool (default: every numeric non-key, non-target column).
    method : str, default="xi"
    k : int, optional
        Keep at most the ``k`` best-ranked features.
    alpha : float, optional
        Keep features with ``p_value_adj <= alpha``. When neither ``k`` nor
        ``alpha`` is given, ``alpha=0.05``.
    null, correction, by, lag, q, n_resamples, seed, min_obs
        Passed to :func:`feature_screen`.
    keep_target : bool, default=True
        Retain the target column in the transformed panel.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    selected_ : list of str
        Surviving features, best first.
    screen_ : polars.DataFrame
        The full screen from the last ``fit``.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        target: str,
        *,
        features: Sequence[str] | None = None,
        method: str = "xi",
        k: int | None = None,
        alpha: float | None = None,
        null: str = "auto",
        correction: str = "benjamini_hochberg",
        by: str | None = "entity",
        lag: int = 0,
        q: float = 0.05,
        n_resamples: int = DEFAULT_RESAMPLES,
        seed: int = 0,
        min_obs: int | None = None,
        keep_target: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError(f"`k` must be a positive integer, got {k!r}.")
        self.target = target
        self.features = list(features) if features is not None else None
        self.method = method
        self.k = k
        self.alpha = alpha
        self.null = null
        self.correction = correction
        self.by = by
        self.lag = lag
        self.q = q
        self.n_resamples = n_resamples
        self.seed = seed
        self.min_obs = min_obs
        self.keep_target = bool(keep_target)
        self.selected_: list[str] = []
        self.screen_: pl.DataFrame | None = None

    def _fit(self, panel: PanelFrame) -> None:
        screen = feature_screen(
            panel,
            self.target,
            self.features,
            method=self.method,
            by=self.by,
            null=self.null,
            correction=self.correction,
            lag=self.lag,
            q=self.q,
            n_resamples=self.n_resamples,
            seed=self.seed,
            min_obs=self.min_obs,
        )
        self.screen_ = screen
        ranked = screen.filter(pl.col("estimate").is_not_nan())
        alpha = self.alpha if (self.alpha is not None or self.k is not None) else 0.05
        if alpha is not None:
            ranked = ranked.filter(pl.col("p_value_adj").fill_nan(None) <= alpha)
        names = ranked.get_column("feature").to_list()
        self.selected_ = names[: self.k] if self.k is not None else names
        if not self.selected_:
            warnings.warn(
                "ScreenSelector selected no feature on this training fold.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        missing = [c for c in self.selected_ if c not in panel]
        if missing:
            raise ValueError(
                f"ScreenSelector.transform: selected column(s) {missing} not found; "
                f"available: {panel.columns}."
            )
        keep = list(self.selected_)
        if self.keep_target and self.target in panel:
            keep.append(self.target)
        return panel.select(*(pl.col(c) for c in keep))
