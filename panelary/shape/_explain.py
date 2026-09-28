"""Explainability: one method, one return type, and a stability check on top.

Every fitted :class:`~panelary.shape.ShapeTransform` implements
``explain() -> polars.DataFrame`` with the columns

``component, source_feature, loading, abs_loading, rank,
explained_variance_ratio, reconstruction_error``

-- a tidy frame, so ``.filter(pl.col("component") == "rpc_1").sort("abs_loading",
descending=True).head(10)`` is the whole story of a component in one line.
``explained_variance_ratio`` and ``reconstruction_error`` are frame-level
quantities repeated on every row of their component (null where undefined).

Three rules keep the answers honest:

1. **Loadings are in the user's units.** If a transform standardised
   internally, :func:`explain_loadings` divides by the training scale, so a
   loading is "per unit of the original column" and features on different
   scales are comparable.
2. **Signs are fixed deterministically** (``_sign_of_max_abs`` from
   :mod:`panelary.reduce._base`, the same rule ``reduce/`` uses), so a
   component does not flip between folds.
3. **For ``ColumnSubset`` / ``CUR`` the answer is exact**: ``source_feature`` is
   a real column and ``loading`` is 1.0.

:func:`stability` refits a transform on each training fold of a
:class:`~panelary.core.model_selection.PurgedKFold`, aligns each fold's loadings
to the first fold's by orthogonal Procrustes, and reports per-feature loading
dispersion. A component whose loadings reshuffle every fold is an artifact, not
a factor.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from panelary.core.panel_frame import PanelFrame
    from panelary.shape._axes import ShapeTransform

__all__ = [
    "EXPLAIN_SCHEMA",
    "explain_frame",
    "explain_loadings",
    "procrustes",
    "stability",
]

#: The fixed schema of every ``explain()`` result.
EXPLAIN_SCHEMA: dict[str, Any] = {
    "component": pl.Utf8,
    "source_feature": pl.Utf8,
    "loading": pl.Float64,
    "abs_loading": pl.Float64,
    "rank": pl.Int64,
    "explained_variance_ratio": pl.Float64,
    "reconstruction_error": pl.Float64,
}


def explain_frame(
    components: Sequence[str],
    sources: Sequence[str],
    loadings: NDArray[Any] | Sequence[float],
    *,
    explained_variance_ratio: dict[str, float] | None = None,
    reconstruction_error: float | None = None,
) -> pl.DataFrame:
    """Build an ``explain()`` frame from parallel long-format lists.

    ``rank`` is the 1-based rank of ``abs_loading`` within each component
    (1 = the dominant source), ties broken by input order.
    """
    loads = np.asarray(loadings, dtype=np.float64)
    evr = explained_variance_ratio or {}
    df = pl.DataFrame(
        {
            "component": list(components),
            "source_feature": list(sources),
            "loading": loads,
            "abs_loading": np.abs(loads),
            "explained_variance_ratio": [evr.get(c) for c in components],
            "reconstruction_error": [reconstruction_error] * len(loads),
        },
        schema_overrides={
            "explained_variance_ratio": pl.Float64,
            "reconstruction_error": pl.Float64,
        },
    )
    df = df.with_columns(
        pl.col("abs_loading")
        .rank(method="ordinal", descending=True)
        .over("component")
        .cast(pl.Int64)
        .alias("rank")
    )
    return df.select(list(EXPLAIN_SCHEMA)).cast(EXPLAIN_SCHEMA)  # type: ignore[arg-type]


def explain_loadings(
    component_names: Sequence[str],
    feature_names: Sequence[str],
    loadings: NDArray[Any],
    *,
    scale: NDArray[Any] | None = None,
    explained_variance_ratio: NDArray[Any] | Sequence[float] | None = None,
    reconstruction_error: float | None = None,
) -> pl.DataFrame:
    """``explain()`` for a ``(k, d)`` loading matrix.

    Parameters
    ----------
    component_names : sequence of str
        ``k`` component names.
    feature_names : sequence of str
        ``d`` source feature names.
    loadings : numpy.ndarray
        ``(k, d)``: component ``i``'s weight on feature ``j``, in the
        transform's internal (possibly standardised) units.
    scale : numpy.ndarray, optional
        ``(d,)`` training standard deviations used to standardise internally.
        Loadings are divided by it, so they are per unit of the original
        column (rule 1).
    explained_variance_ratio : array-like, optional
        ``(k,)`` per component.
    reconstruction_error : float, optional
        Frame-level relative reconstruction error.
    """
    L = np.asarray(loadings, dtype=np.float64)
    if scale is not None:
        s = np.asarray(scale, dtype=np.float64)
        L = L / np.where(s == 0.0, 1.0, s)[None, :]
    k, d = L.shape
    comps = [c for c in component_names for _ in range(d)]
    srcs = list(feature_names) * k
    evr = (
        None
        if explained_variance_ratio is None
        else {
            c: float(v)
            for c, v in zip(
                component_names, np.asarray(explained_variance_ratio), strict=True
            )
        }
    )
    return explain_frame(
        comps,
        srcs,
        L.reshape(-1),
        explained_variance_ratio=evr,
        reconstruction_error=reconstruction_error,
    )


def procrustes(A: NDArray[Any], B: NDArray[Any]) -> NDArray[np.float64]:
    """Orthogonal ``R`` minimising ``||A @ R - B||_F`` (Schonemann, 1966).

    ``A`` and ``B`` are ``(d, k)`` (features x components). Returns ``(k, k)``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def _loadings_of(t: ShapeTransform) -> tuple[list[str], list[str], NDArray[np.float64]]:
    comps = list(getattr(t, "component_names_", []))
    feats = list(getattr(t, "feature_names_in_", []))
    L = getattr(t, "loadings_", None)
    if L is None or not comps:
        raise TypeError(
            f"stability(): {type(t).__name__} exposes no `loadings_` / "
            "`component_names_`; only loading-based transforms (RandomizedPCA, "
            "FrequentDirections, ColumnSubset, CUR) can be stability-checked."
        )
    return comps, feats, np.asarray(L, dtype=np.float64)


def stability(
    transform: ShapeTransform,
    X: PanelFrame | pl.DataFrame | pl.LazyFrame,
    cv: Any = None,
    *,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Refit ``transform`` per training fold and report loading dispersion.

    Each fold's ``(k, d)`` loadings are aligned to the first fold's by
    orthogonal Procrustes (component order and sign/rotation are not
    identified across refits), then per ``(component, source_feature)`` the
    mean, standard deviation and range of the aligned loading are reported.

    Parameters
    ----------
    transform : ShapeTransform
        An *unfitted template*; it is deep-copied per fold, never mutated.
    X : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel to split.
    cv : optional
        A splitter with ``split(panel) -> (train, test)`` pairs. Default
        ``PurgedKFold(n_splits=5)``. Only the training side is used.
    entity, time : str, optional
        Keys for a bare polars frame.

    Returns
    -------
    polars.DataFrame
        ``component, source_feature, mean_loading, std_loading, min_loading,
        max_loading, n_folds``, plus ``component_congruence`` -- the mean
        absolute Tucker congruence of the aligned component with the reference
        fold (1.0 = identical direction every fold).
    """
    from panelary.core.model_selection import PurgedKFold

    panel = transform._as_panel(X, method="stability", entity=entity, time=time)
    splitter = cv if cv is not None else PurgedKFold(n_splits=5)
    ref: NDArray[np.float64] | None = None
    comps: list[str] = []
    feats: list[str] = []
    aligned: list[NDArray[np.float64]] = []
    for train, _test in splitter.split(panel):
        t = copy.deepcopy(transform)
        t.fit(train)  # type: ignore[arg-type]  # splitters yield panel frames
        c, f, L = _loadings_of(t)
        if ref is None:
            ref, comps, feats = L, c, f
            aligned.append(L)
            continue
        if f != feats or L.shape != ref.shape:
            raise ValueError(
                "stability(): folds produced different feature sets or widths; "
                "pin `columns=` and the component count."
            )
        R = procrustes(L.T, ref.T)
        aligned.append((L.T @ R).T)
    if ref is None:
        raise ValueError("stability(): the splitter yielded no folds.")
    stack = np.stack(aligned)  # (folds, k, d)
    norms = np.linalg.norm(stack, axis=2)  # (folds, k)
    ref_n = np.linalg.norm(ref, axis=1)
    cong = np.abs(np.einsum("fkd,kd->fk", stack, ref)) / np.where(
        norms * ref_n[None, :] == 0.0, 1.0, norms * ref_n[None, :]
    )
    k, d = ref.shape
    return pl.DataFrame(
        {
            "component": [c for c in comps for _ in range(d)],
            "source_feature": feats * k,
            "mean_loading": stack.mean(axis=0).reshape(-1),
            "std_loading": stack.std(axis=0).reshape(-1),
            "min_loading": stack.min(axis=0).reshape(-1),
            "max_loading": stack.max(axis=0).reshape(-1),
            "n_folds": [stack.shape[0]] * (k * d),
            "component_congruence": np.repeat(cong.mean(axis=0), d),
        }
    )
