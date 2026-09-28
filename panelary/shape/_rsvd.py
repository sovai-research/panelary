"""Randomized SVD and randomized PCA -- pure numpy (Halko, Martinsson & Tropp 2011).

``randomized_svd`` is two ``@`` products, a few ``qr`` calls and one small
``svd``: all of its time is inside BLAS/LAPACK, which is why it needs neither
sklearn nor a compiled extension.

Algorithm (HMT 2011, Alg. 4.4 with QR re-orthonormalisation)
--------------------------------------------------------------
Draw ``Omega (d, k + p)`` from a seeded Gaussian, form ``Y = A @ Omega``, run
``q`` power iterations **re-orthonormalising between each** (without it,
``q > 1`` silently loses the small singular values to rounding), ``Q = qr(Y)``,
``B = Q.T @ A``, then the small ``svd(B)``. The re-orthonormalisations use
CholeskyQR2 with a checked Householder fallback (see ``_orthonormalize``). ``n_iter="auto"`` is 7 when
``k < 0.1 * min(A.shape)`` and 4 otherwise, the established heuristic.

Reference: N. Halko, P.-G. Martinsson, J. A. Tropp (2011), "Finding structure
with randomness", *SIAM Review* 53(2). Clean-room implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.reduce._base import _sign_of_max_abs
from panelary.shape._axes import (
    Axis,
    InputShape,
    Intent,
    ShapeSpec,
    ShapeTransform,
    check_positive_int,
    check_seed,
    register_shape_spec,
    relative_frobenius_error,
)
from panelary.shape._feature import _FeatureAxisTransform, fit_scaler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["CrossSectionalRandomizedPCA", "RandomizedPCA", "randomized_svd"]


def _orthonormalize(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormal basis of ``range(Y)`` for a tall ``(m, l)`` sketch.

    CholeskyQR2 -- twice ``Y <- Y L^{-T}`` with ``L = chol(Y^T Y)`` -- is a
    few GEMMs and an ``l x l`` Cholesky: measured ~3x cheaper than Householder
    QR on a ``640000 x 26`` sketch, and orthonormal to machine precision when
    ``Y`` is well conditioned. It is used only if the result passes an
    explicit orthogonality check; a (numerically) rank-deficient or badly
    conditioned ``Y`` -- an exactly rank-``k`` input with oversampling, say --
    falls back to Householder QR, so accuracy never depends on the fast path.
    """
    m, ell = Y.shape
    if m < 4 * ell:
        return np.linalg.qr(Y)[0]
    Q = Y
    try:
        for _ in range(2):
            L = np.linalg.cholesky(Q.T @ Q)
            Q = Q @ np.linalg.inv(L.T)
    except np.linalg.LinAlgError:
        return np.linalg.qr(Y)[0]
    if not np.isfinite(Q).all() or np.abs(Q.T @ Q - np.eye(ell)).max() > 1e-12:
        return np.linalg.qr(Y)[0]
    return Q


def randomized_svd(
    A: NDArray[Any],
    k: int,
    *,
    n_oversamples: int = 10,
    n_iter: int | str = "auto",
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Rank-``k`` truncated SVD by random range finding.

    Parameters
    ----------
    A : numpy.ndarray
        ``(m, n)`` matrix; finite values only (NaN is refused, not imputed).
    k : int
        Target rank, ``1 <= k <= min(m, n)``.
    n_oversamples : int, default 10
        Extra random directions ``p``; the sketch has ``k + p`` columns.
    n_iter : int or "auto", default "auto"
        Power iterations ``q``; ``"auto"`` = 7 if ``k < 0.1 * min(m, n)`` else 4.
    seed : int, default 0
        Explicit seed for the Gaussian test matrix (``np.random.default_rng``).

    Returns
    -------
    U : numpy.ndarray
        ``(m, k)`` left singular vectors.
    s : numpy.ndarray
        ``(k,)`` singular values, descending.
    Vt : numpy.ndarray
        ``(k, n)`` right singular vectors.

    Raises
    ------
    ValueError
        On a non-finite ``A`` or an out-of-range ``k``.

    Notes
    -----
    Signs are LAPACK's; callers that need stable signs across refits apply
    ``_sign_of_max_abs`` (as :class:`RandomizedPCA` does).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"randomized_svd: A must be 2-D, got shape {A.shape}.")
    m, n = A.shape
    r = min(m, n)
    k = check_positive_int(k, "k", "randomized_svd")
    if k > r:
        raise ValueError(f"randomized_svd: k={k} exceeds min(A.shape)={r}.")
    if not np.isfinite(A).all():
        raise ValueError("randomized_svd: A contains NaN/inf; refusing to impute.")
    seed = check_seed(seed, "randomized_svd")
    if n_iter == "auto":
        q = 7 if k < 0.1 * r else 4
    else:
        q = int(n_iter)
        if q < 0:
            raise ValueError(f"randomized_svd: n_iter must be >= 0, got {n_iter!r}.")
    ell = min(k + int(n_oversamples), r)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, ell))
    Q = _orthonormalize(A @ omega)
    for _ in range(q):
        Z = _orthonormalize(A.T @ Q)
        Q = _orthonormalize(A @ Z)
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], s[:k], Vt[:k]


class RandomizedPCA(_FeatureAxisTransform):
    """Leak-safe PCA via :func:`randomized_svd` -- no sklearn, fit on training rows only.

    Centres (and by default standardises) with training-row statistics, takes
    the rank-``n_components`` randomized SVD of the training matrix, fixes each
    component's sign deterministically (largest-magnitude loading positive --
    the same rule as :mod:`panelary.reduce`), and projects any panel onto
    ``rpc_1 .. rpc_k``.

    Parameters
    ----------
    n_components : int, default 8
        Components to keep; capped at ``min(n_rows_fit, n_features)``.
    standardize : bool, default True
        Scale columns to unit training variance before the SVD.
    n_oversamples : int, default 10
    n_iter : int or "auto", default "auto"
    seed : int, default 0
        Explicit seed for the random range finder.
    prefix : str, default "rpc"
    keep_features, dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._feature._FeatureAxisTransform`.

    Attributes
    ----------
    mean_, scale_ : numpy.ndarray
        Training column mean and scale (``scale_`` is all ones when
        ``standardize=False``).
    components_ : numpy.ndarray
        ``(k, d)`` sign-fixed principal axes, in standardised units.
    loadings_ : numpy.ndarray
        Alias of ``components_`` (what :func:`~panelary.shape.stability` reads).
    singular_values_, explained_variance_, explained_variance_ratio_ : numpy.ndarray
        ``explained_variance_ratio_`` is exact: the total variance is the
        squared Frobenius norm of the centred training matrix.
    reconstruction_error_ : float
        Relative Frobenius error of the rank-``k`` reconstruction of the
        (standardised) training matrix.
    n_components_ : int
    component_names_ : list of str

    Notes
    -----
    ``panel_safe = True`` (a row-wise map), ``leakage_safe = True`` *as an
    estimator*: everything is learned in ``fit`` from the training rows. Calling
    ``fit_transform`` on a full sample that is later cross-validated is still a
    leak -- that is the registry's ``safe_scope="window"``.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = False
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.FEATURE,
        width_rule="rank_dependent",
        invertible="approximate",
        streaming="batch",
        cost_hint="O(n d k)",
    )
    _default_prefix = "rpc"

    def __init__(
        self,
        *,
        n_components: int = 8,
        standardize: bool = True,
        n_oversamples: int = 10,
        n_iter: int | str = "auto",
        seed: int = 0,
        prefix: str | None = None,
        keep_features: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            keep_features=keep_features,
            dtype=dtype,
            columns=columns,
            max_bytes=max_bytes,
            entity=entity,
            time=time,
        )
        self.n_components = check_positive_int(
            n_components, "n_components", "RandomizedPCA"
        )
        self.standardize = bool(standardize)
        self.n_oversamples = int(n_oversamples)
        self.n_iter = n_iter
        self.seed = check_seed(seed, "RandomizedPCA")
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.reconstruction_error_: float | None = None
        self.n_components_: int = 0
        self.component_names_: list[str] = []

    @property
    def loadings_(self) -> NDArray[np.float64] | None:
        return self.components_

    def _out_width(self, in_width: int) -> int:
        k = self.n_components_ or self.n_components
        return min(k, in_width)

    def _scratch_bytes(self, shape: InputShape, m: int) -> float:
        ell = min(m + self.n_oversamples, shape.width)
        return 8.0 * (
            2 * shape.rows * shape.width + 3 * shape.rows * ell + ell * shape.width
        )

    def _output_names(self) -> list[str]:
        return list(self.component_names_)

    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        n, d = X.shape
        if self.standardize:
            mean, scale = fit_scaler(X)
        else:
            mean, scale = X.mean(axis=0), np.ones(d)
        Xs = (X - mean) / scale
        k = min(self.n_components, n, d)
        _, s, Vt = randomized_svd(
            Xs, k, n_oversamples=self.n_oversamples, n_iter=self.n_iter, seed=self.seed
        )
        sign = np.array([_sign_of_max_abs(Vt[i]) for i in range(k)], dtype=np.float64)
        total = float(np.sum(Xs * Xs))
        denom = max(n - 1, 1)
        self.mean_, self.scale_ = mean, scale
        self.components_ = Vt * sign[:, None]
        self.singular_values_ = s
        self.explained_variance_ = s**2 / denom
        self.explained_variance_ratio_ = (s**2 / total) if total > 0 else np.zeros(k)
        # ||Xs - Xs V V^T||_F^2 = ||Xs||_F^2 - sum(s^2) for orthonormal V.
        resid = max(total - float(np.sum(s * s)), 0.0)
        self.reconstruction_error_ = float(np.sqrt(resid / total)) if total > 0 else 0.0
        self.n_components_ = k
        self.component_names_ = [f"{self.prefix}_{i}" for i in range(1, k + 1)]

    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.components_ is not None
        assert self.mean_ is not None and self.scale_ is not None
        return ((X - self.mean_) / self.scale_) @ self.components_.T

    # ------------------------------------------------------------------ #
    def inverse_transform(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        """Map component scores back to the original feature space.

        Parameters
        ----------
        Z : PanelFrame | polars.DataFrame | numpy.ndarray
            This transform's output (component columns), or an ``(n, k)`` array.

        Returns
        -------
        numpy.ndarray
            ``(n, d)`` reconstruction in original units, columns in
            ``feature_names_in_`` order.
        """
        self._check_fitted("inverse_transform")
        assert self.components_ is not None
        if isinstance(Z, np.ndarray):
            S = np.asarray(Z, dtype=np.float64)
        else:
            frame = Z.collect() if isinstance(Z, PanelFrame) else Z
            S = frame.select(self.component_names_).to_numpy().astype(np.float64)
        assert self.mean_ is not None and self.scale_ is not None
        return (S @ self.components_) * self.scale_ + self.mean_

    def reconstruction_error(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> float:
        """Relative Frobenius error ``||X - X_hat||_F / ||X||_F`` over complete rows.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
        entity, time : str, optional

        Returns
        -------
        float
        """
        self._check_fitted("reconstruction_error")
        panel = self._as_panel(
            X, method="reconstruction_error", entity=entity, time=time
        )
        M = self._matrix(panel.lazy(), self.feature_names_in_)
        M = M[np.isfinite(M).all(axis=1)]
        return relative_frobenius_error(M, self.inverse_transform(self._map(M)))

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_loadings

        assert self.components_ is not None
        return explain_loadings(
            self.component_names_,
            self.feature_names_in_,
            self.components_,
            scale=self.scale_ if self.standardize else None,
            explained_variance_ratio=self.explained_variance_ratio_,
            reconstruction_error=self.reconstruction_error_,
        )


class CrossSectionalRandomizedPCA(ShapeTransform):
    """Per-date randomized PCA across entities (``axis="entity"``), aligned across dates.

    Each date's cross-section is standardised and reduced on its own, so nothing
    crosses dates in the fit: leak-safe in time by construction, and
    ``panel_safe = False`` by design (it mixes entities within a date). But
    per-date components are identified only up to rotation (and sign, and
    order) **across dates** (Gabaix et al., App. C), so a raw per-date score is
    not a comparable time series. This transform therefore **refuses** to emit
    per-date components as a panel column unless ``align="procrustes"``:

    * ``align=None`` (default) -- :meth:`transform` raises, naming the reason.
      :meth:`date_components` still gives the date-local loadings, keyed by
      ``(time, component, source_feature)`` -- a description of each
      cross-section, not a feature.
    * ``align="procrustes"`` -- date ``t``'s ``(d, k)`` loadings are rotated
      by the orthogonal Procrustes solution onto date ``t-1``'s *aligned*
      loadings (the first date is sign-fixed like :mod:`panelary.reduce`), and
      each row is scored on its own date's aligned loadings. Alignment reads
      only the previous date, so the scores are causal and prefix-invariant.
      The rotation acts within the date's ``k``-dimensional subspace: aligned
      components are a smoothly-continued basis of it, not the per-date
      principal axes.

    The Procrustes helper is :func:`panelary.shape.procrustes` (Schonemann
    1966), implemented in :mod:`panelary.shape._explain`; the ``embed/``
    contract's ``_xs.py`` specifies the same alignment-to-``t-1`` rule.

    Parameters
    ----------
    n_components : int, default 3
    standardize : bool, default True
        Standardise each date's cross-section with that date's statistics.
    align : {None, "procrustes"}, default None
    seed : int, default 0
    prefix : str, default "xspc"
        Output columns ``{prefix}_1 .. {prefix}_k``.
    columns, max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.

    NaN policy
    ----------
    Per date, only entities with every selected column observed enter the
    SVD and receive scores; the others are NaN. A date with fewer than
    ``n_components + 1`` complete entities is NaN throughout and does not
    advance the alignment chain.
    """

    panel_safe = False
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = True
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.ENTITY,
        width_rule="rank_dependent",
        invertible="none",
        streaming="batch",
        cost_hint="O(T E d k)",
    )

    def __init__(
        self,
        *,
        n_components: int = 3,
        standardize: bool = True,
        align: str | None = None,
        seed: int = 0,
        prefix: str = "xspc",
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        if align not in (None, "procrustes"):
            raise ValueError(
                f"CrossSectionalRandomizedPCA: align must be None or 'procrustes', got {align!r}."
            )
        self.n_components = check_positive_int(
            n_components, "n_components", "CrossSectionalRandomizedPCA"
        )
        self.standardize = bool(standardize)
        self.align = align
        self.seed = check_seed(seed, "CrossSectionalRandomizedPCA")
        self.prefix = prefix

    def _out_width(self, in_width: int) -> int:
        return min(self.n_components, in_width)

    def _fit(self, panel: PanelFrame) -> None:
        self.feature_names_in_ = self._resolve_columns(panel)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if self.align is None:
            raise ValueError(
                "CrossSectionalRandomizedPCA: per-date components are identified only "
                "up to rotation across dates, so emitting them as a panel column would "
                "make a time series out of arbitrarily rotated axes. Pass "
                "align='procrustes' to align each date to the previous one, or use "
                "`date_components(X)` for the date-local loadings."
            )
        from panelary.shape._explain import procrustes

        ent, tim = panel.entity_col, panel.time_col
        cols = self.feature_names_in_
        missing = [c for c in cols if c not in panel]
        if missing:
            raise ValueError(
                f"CrossSectionalRandomizedPCA.transform: column(s) {missing} not found. "
                f"Available columns: {panel.columns}."
            )
        frame = panel.collect()
        work = frame.select(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("__row__"),
            pl.col(ent),
            pl.col(tim),
            *[pl.col(c).cast(pl.Float64) for c in cols],
        ).sort([tim, ent], maintain_order=True)
        k = min(self.n_components, len(cols))
        self._enforce_budget(InputShape(rows=frame.height, width=len(cols), entities=0))
        out = np.full((frame.height, k), np.nan, dtype=np.float64)
        prev: NDArray[np.float64] | None = None
        for _key, g in work.group_by(tim, maintain_order=True):
            M = g.select(cols).to_numpy().astype(np.float64, copy=False)
            ok = np.isfinite(M).all(axis=1)
            if int(ok.sum()) < k + 1:
                continue
            M = M[ok]
            if self.standardize:
                mean, scale = fit_scaler(M)
            else:
                mean, scale = M.mean(axis=0), np.ones(M.shape[1])
            Ms = (M - mean) / scale
            _, _, Vt = randomized_svd(Ms, k, seed=self.seed)
            V = Vt.T  # (d, k)
            if prev is None:
                sign = np.array([_sign_of_max_abs(V[:, i]) for i in range(k)])
                V = V * sign[None, :]
            else:
                V = V @ procrustes(V, prev)
            prev = V
            rows = g["__row__"].to_numpy()[ok]
            out[rows] = Ms @ V
        names = [f"{self.prefix}_{i}" for i in range(1, k + 1)]
        res = frame.select(ent, tim).with_columns(
            [pl.Series(n, out[:, i], dtype=pl.Float64) for i, n in enumerate(names)]
        )
        return PanelFrame(res, entity=ent, time=tim, validate=False)

    def date_components(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> pl.DataFrame:
        """Date-local loadings: one randomized PCA per date's cross-section.

        Returns
        -------
        polars.DataFrame
            ``time, component, source_feature, loading, explained_variance_ratio``.
            Dates with fewer than two complete entities are skipped.
        """
        panel = self._as_panel(X, method="date_components", entity=entity, time=time)
        cols = self.feature_names_in_ or self._resolve_columns(panel)
        tim = panel.time_col
        frame = panel.collect().select(
            [tim, *[pl.col(c).cast(pl.Float64) for c in cols]]
        )
        rows: list[pl.DataFrame] = []
        for (t,), g in frame.sort(tim).group_by(tim, maintain_order=True):
            M = g.select(cols).to_numpy()
            M = M[np.isfinite(M).all(axis=1)]
            if M.shape[0] < 2:
                continue
            mean, scale = (
                fit_scaler(M)
                if self.standardize
                else (M.mean(axis=0), np.ones(M.shape[1]))
            )
            Ms = (M - mean) / scale
            k = min(self.n_components, *Ms.shape)
            _, s, Vt = randomized_svd(Ms, k, seed=self.seed)
            sign = np.array([_sign_of_max_abs(Vt[i]) for i in range(k)])
            Vt = Vt * sign[:, None]
            tot = float(np.sum(Ms * Ms))
            evr = s**2 / tot if tot > 0 else np.zeros(k)
            names = [f"xspc_{i}" for i in range(1, k + 1)]
            rows.append(
                pl.DataFrame(
                    {
                        "component": [c for c in names for _ in cols],
                        "source_feature": cols * k,
                        "loading": Vt.reshape(-1),
                        "explained_variance_ratio": np.repeat(evr, len(cols)),
                    }
                ).with_columns(pl.lit(t).alias(tim))
            )
        if not rows:
            return pl.DataFrame(
                schema={
                    tim: frame.schema[tim],
                    "component": pl.Utf8,
                    "source_feature": pl.Utf8,
                    "loading": pl.Float64,
                    "explained_variance_ratio": pl.Float64,
                }
            )
        out = pl.concat(rows)
        return out.select(
            [tim, "component", "source_feature", "loading", "explained_variance_ratio"]
        )


register_shape_spec(
    RandomizedPCA,
    name="rsvd",
    params={"n_components": int, "standardize": bool, "n_iter": int, "seed": int},
    tier="B",
    safe_scope="window",
    source="Halko, Martinsson & Tropp (2011), SIAM Review 53(2); clean-room, Panelary",
)
