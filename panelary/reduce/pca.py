"""Concrete leak-safe panel reducers (PCA family + friends).

Every class here is a thin subclass of
:class:`panelary.reduce._base._PanelReducer`: it only names the sklearn
estimator to build. The base supplies the leak-safe fit-on-train contract
(scaler + rotation + auto-``n_components`` learned on training rows only),
deterministic component sign-fixing, ``pc_1 .. pc_k`` naming and provenance
attributes.

The set mirrors SovAI's ``reducer_methods`` (``pca``, ``truncated_svd``,
``factor_analysis``, ``gaussian_random_projection``) plus ``kernel_pca`` and
``nmf``, and adds an optional, dependency-guarded :class:`PanelUMAP`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from panelary.reduce._base import _PanelReducer, _sign_of_max_abs

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from numpy.typing import NDArray

    from panelary.core.panel_frame import PanelFrame

__all__ = [
    "PanelPCA",
    "PanelSVD",
    "PanelFactorAnalysis",
    "PanelRandomProjection",
    "PanelKernelPCA",
    "PanelNMF",
    "PanelUMAP",
    "reduce_features",
]


# --------------------------------------------------------------------------- #
# numpy backends (panelary.shape), sklearn-shaped so _PanelReducer can drive them
# --------------------------------------------------------------------------- #
#: Accepted values of ``backend=`` on :class:`PanelPCA` / :class:`PanelSVD`.
_BACKENDS = ("sklearn", "numpy", "auto")
#: ``backend="auto"`` routes to numpy when ``n_components <= _AUTO_RATIO * n_features``.
_AUTO_RATIO = 0.1


class _NumpyScaler:
    """``StandardScaler``-compatible column standardiser (``ddof=0``; zero std -> 1)."""

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        from panelary.shape._feature import fit_scaler

        X = np.asarray(X, dtype=np.float64)
        self.mean_, self.scale_ = fit_scaler(X)
        return (X - self.mean_) / self.scale_

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class _NumpySVDReducer:
    """``sklearn.decomposition.PCA`` / ``TruncatedSVD``-shaped randomized SVD.

    ``center=True`` mirrors ``PCA`` (``explained_variance_ratio_ = s^2 / ||Xc||_F^2``);
    ``center=False`` mirrors ``TruncatedSVD`` (``explained_variance_ratio_`` is
    the variance of each score column over the total column variance).
    """

    def __init__(self, n_components: int, *, center: bool, seed: int) -> None:
        self.n_components = int(n_components)
        self.center = center
        self.seed = seed

    def fit(self, X: NDArray[Any]) -> _NumpySVDReducer:
        from panelary.shape._rsvd import randomized_svd

        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0) if self.center else np.zeros(X.shape[1])
        Xc = X - self.mean_
        _, s, Vt = randomized_svd(Xc, self.n_components, seed=self.seed)
        self.components_ = Vt
        self.singular_values_ = s
        if self.center:
            total = float(np.sum(Xc * Xc))
            self.explained_variance_ = s**2 / max(X.shape[0] - 1, 1)
            self.explained_variance_ratio_ = (
                s**2 / total if total > 0 else np.zeros_like(s)
            )
        else:
            var = float(np.var(X, axis=0).sum())
            scores_var = np.var(Xc @ Vt.T, axis=0)
            self.explained_variance_ = scores_var
            self.explained_variance_ratio_ = (
                scores_var / var if var > 0 else np.zeros_like(s)
            )
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        return self.fit(X).transform(X)


class _NumpyProjection:
    """Seeded sparse (Li et al. 2006) or SRHT (Ailon & Chazelle 2009) projection.

    Data-independent: ``fit`` reads only the input width. Backed by
    :mod:`panelary.shape._project`.
    """

    def __init__(
        self, n_components: int, *, method: str, density: float | str, seed: int
    ) -> None:
        self.n_components = int(n_components)
        self.method = method
        self.density = density
        self.seed = seed

    def fit(self, X: NDArray[Any]) -> _NumpyProjection:
        from panelary.shape._project import _srht_draw, hadamard_rows, sparse_rp_matrix

        d = int(np.asarray(X).shape[1])
        if self.method == "sparse":
            self.components_ = sparse_rp_matrix(
                d, self.n_components, density=self.density, seed=self.seed
            ).T.copy()
            self._signs = None
        else:
            signs, sample, D = _srht_draw(d, self.n_components, self.seed)
            self._signs, self._sample, self._padded = signs, sample, D
            self.components_ = (
                hadamard_rows(sample, d) * signs[None, :] / np.sqrt(self.n_components)
            )
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        X = np.asarray(X, dtype=np.float64)
        if self._signs is None:
            return X @ self.components_.T
        from panelary.shape._project import fwht

        padded = np.zeros((X.shape[0], self._padded), dtype=np.float64)
        padded[:, : X.shape[1]] = X * self._signs
        return fwht(padded)[:, self._sample] / np.sqrt(self.n_components)

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        return self.fit(X).transform(X)


def _explicit_seed(owner: str, random_state: Any) -> int:
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError(
            f"{owner}: the numpy backend needs an explicit int `random_state` (never "
            f"None / a global RNG), got {random_state!r}."
        )
    return int(random_state)


class _NumpyBackendMixin(_PanelReducer):
    """Fit through a numpy reducer (no sklearn at all) when :meth:`_numpy_path` says so.

    Same contract as :meth:`_PanelReducer._fit` -- scaler, rotation and
    component count learned on the training rows only, deterministic sign fix
    -- with the ``StandardScaler`` replaced by :class:`_NumpyScaler`.
    """

    _numpy_active: bool = False

    def _numpy_path(self, n_features: int) -> bool:  # pragma: no cover - overridden
        return False

    def _make_numpy_reducer(self, n_components: int) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _make_reducer(self, n_components: int) -> Any:
        if self._numpy_active:
            return self._make_numpy_reducer(n_components)
        return self._make_sklearn_reducer(n_components)

    def _make_sklearn_reducer(self, n_components: int) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        self._numpy_active = self._numpy_path(len(cols))
        if not self._numpy_active:
            super()._fit(panel)
            return
        self.feature_names_in_ = cols
        X = panel.lazy().select(cols).collect().to_numpy().astype(np.float64)
        self._validate_matrix(X)
        if not np.isfinite(X).all():
            raise ValueError(
                f"{type(self).__name__}: the numpy backend refuses NaN/inf input; "
                "impute upstream with a point-in-time imputer."
            )
        if self.standardize:
            scaler = _NumpyScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler_ = scaler
        else:
            X_scaled = X
            self.scaler_ = None
        k = self._resolve_n_components(X_scaled)
        reducer = self._make_reducer(k)
        reducer.fit(X_scaled)
        # Every numpy reducer exposes loadings, so the sign rule is always the
        # loading-based one (the base's score-based fallback is never needed).
        comps = np.asarray(reducer.components_)
        if self.sign_fix:
            sign = np.array(
                [_sign_of_max_abs(comps[i]) for i in range(k)], dtype=np.float64
            )
        else:
            sign = np.ones(k, dtype=np.float64)
        self.reducer_ = reducer
        self.n_components_ = k
        self.component_names_ = [f"{self.prefix}_{i}" for i in range(1, k + 1)]
        self.sign_flip_ = sign
        self.components_ = comps * sign[:, None]
        evr = getattr(reducer, "explained_variance_ratio_", None)
        self.explained_variance_ratio_ = None if evr is None else np.asarray(evr)


class _SVDBackend(_NumpyBackendMixin):
    """``backend=`` switch shared by :class:`PanelPCA` and :class:`PanelSVD`."""

    _center: bool = True

    def __init__(self, *, backend: str = "sklearn", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if backend not in _BACKENDS:
            raise ValueError(
                f"{type(self).__name__}: backend must be one of {list(_BACKENDS)}, "
                f"got {backend!r}."
            )
        self.backend = backend
        if backend != "sklearn":
            _explicit_seed(type(self).__name__, self.random_state)

    def _numpy_path(self, n_features: int) -> bool:
        if self.backend == "numpy":
            return True
        if self.backend == "auto":
            return (
                self.n_components is not None
                and self.n_components <= _AUTO_RATIO * n_features
            )
        return False

    def _make_numpy_reducer(self, n_components: int) -> Any:
        return _NumpySVDReducer(
            n_components,
            center=self._center,
            seed=_explicit_seed(type(self).__name__, self.random_state),
        )


class PanelPCA(_SVDBackend):
    """Leak-safe panel PCA (:class:`sklearn.decomposition.PCA`, or numpy).

    Learns the standardiser, the orthogonal rotation and -- when
    ``n_components`` is left as ``None`` -- the component count (from
    ``explained_variance``) on the **training** rows only, then projects any
    panel onto ``pc_1 .. pc_k``. This is the leak-safe replacement for SovAI's
    ``dimensionality_reduction(method="pca")``, which fits all three on the full
    sample.

    Parameters
    ----------
    backend : {"sklearn", "numpy", "auto"}, default "sklearn"
        ``"numpy"`` runs the scaler and a randomized SVD
        (:func:`panelary.shape.randomized_svd`, Halko et al. 2011) in pure
        numpy -- no scikit-learn -- and agrees with the sklearn backend to
        ``1e-6`` after the shared deterministic sign fix. ``"auto"`` uses numpy
        when ``n_components`` is given and ``<= 0.1 * n_features``, sklearn
        otherwise. The numpy path refuses NaN input and needs an int
        ``random_state``.
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer` for the full
        parameter list.
    """

    _supports_explained_variance = True
    _center = True

    def _make_sklearn_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=self.random_state)


class PanelSVD(_SVDBackend):
    """Leak-safe panel truncated SVD (:class:`sklearn.decomposition.TruncatedSVD`, or numpy).

    LSA-style SVD (no centering; sparse-friendly). ``TruncatedSVD`` requires
    ``n_components < n_features``, so the admissible count is capped one below the
    feature count.

    Parameters
    ----------
    backend : {"sklearn", "numpy", "auto"}, default "sklearn"
        As :class:`PanelPCA`; the numpy path is an uncentred randomized SVD
        with ``TruncatedSVD``'s ``explained_variance_ratio_`` definition.
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer`.
    """

    _supports_explained_variance = True
    _center = False

    def _cap_components(self, n_samples: int, n_features: int) -> int:
        return max(1, min(n_samples, n_features - 1))

    def _make_sklearn_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import TruncatedSVD

        return TruncatedSVD(n_components=n_components, random_state=self.random_state)


class PanelFactorAnalysis(_PanelReducer):
    """Leak-safe panel factor analysis (:class:`sklearn.decomposition.FactorAnalysis`)."""

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import FactorAnalysis

        return FactorAnalysis(n_components=n_components, random_state=self.random_state)


class PanelRandomProjection(_NumpyBackendMixin):
    """Leak-safe panel random projection: Gaussian (sklearn), very sparse, or SRHT.

    ``method="gaussian"`` (default) wraps
    :class:`sklearn.random_projection.GaussianRandomProjection`.
    ``method="sparse"`` (Li, Hastie & Church 2006) and ``method="srht"``
    (Ailon & Chazelle 2009) are pure numpy, routed to
    :mod:`panelary.shape._project`, and need no scikit-learn at all (the
    upstream standardiser is numpy too). In every case the projection matrix is
    drawn from ``random_state`` and is **data-independent**, so only the
    upstream standardiser could leak -- and that is fit on train rows only. For
    a fixed ``random_state`` the projection is identical regardless of the data.

    The Johnson-Lindenstrauss bound does not license a small ``n_components``
    (at ``n = 1e6``, ``eps = 0.1`` it wants ~1e4 dimensions); small outputs
    rest on empirical performance, not theory.

    Parameters
    ----------
    method : {"gaussian", "sparse", "srht"}, default "gaussian"
    density : float or "auto", default "auto"
        ``method="sparse"`` only: fraction of non-zero entries
        (``"auto"`` = ``1 / sqrt(n_features)``).
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer`. The numpy methods
        need an int ``random_state`` and refuse NaN input.
    """

    _METHODS = ("gaussian", "sparse", "srht")

    def __init__(
        self, *, method: str = "gaussian", density: float | str = "auto", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if method not in self._METHODS:
            raise ValueError(
                f"PanelRandomProjection: method must be one of {list(self._METHODS)}, "
                f"got {method!r}."
            )
        self.method = method
        self.density = density
        if method != "gaussian":
            _explicit_seed("PanelRandomProjection", self.random_state)

    def _numpy_path(self, n_features: int) -> bool:
        return self.method != "gaussian"

    def _make_numpy_reducer(self, n_components: int) -> Any:
        return _NumpyProjection(
            n_components,
            method=self.method,
            density=self.density,
            seed=_explicit_seed("PanelRandomProjection", self.random_state),
        )

    def _make_sklearn_reducer(self, n_components: int) -> Any:
        from sklearn.random_projection import GaussianRandomProjection

        return GaussianRandomProjection(
            n_components=n_components, random_state=self.random_state
        )


class PanelKernelPCA(_PanelReducer):
    """Leak-safe panel kernel PCA (:class:`sklearn.decomposition.KernelPCA`).

    Nonlinear PCA in an (RBF by default) kernel space. The landmark rows that
    define the kernel basis are fit on the training rows only. ``KernelPCA``
    exposes no ``components_``, so sign-fixing falls back to forcing each
    component's largest-magnitude *score* positive.

    Parameters
    ----------
    kernel : str, default "rbf"
        Kernel passed to :class:`sklearn.decomposition.KernelPCA`.
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer`.
    """

    def __init__(self, *, kernel: str = "rbf", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import KernelPCA

        return KernelPCA(
            n_components=n_components,
            kernel=self.kernel,
            random_state=self.random_state,
        )


class PanelNMF(_PanelReducer):
    """Leak-safe panel non-negative matrix factorisation.

    Wraps :class:`sklearn.decomposition.NMF`. NMF requires a non-negative input
    matrix; :meth:`_validate_matrix` rejects negative values with a clear error.
    Because standardisation would introduce negatives, ``standardize`` defaults
    to ``False`` here.
    """

    def __init__(self, *, standardize: bool = False, **kwargs: Any) -> None:
        super().__init__(standardize=standardize, **kwargs)

    def _validate_matrix(self, X: NDArray[Any]) -> None:
        import numpy as np

        if np.nanmin(X) < 0.0:
            raise ValueError(
                f"{type(self).__name__}: NMF requires a non-negative input "
                "matrix, but negative values were found. Shift/clip the features "
                "to be non-negative first, or use PanelPCA/PanelSVD instead."
            )

    def _make_reducer(self, n_components: int) -> Any:
        from sklearn.decomposition import NMF

        return NMF(
            n_components=n_components,
            random_state=self.random_state,
            init="nndsvda",
            max_iter=500,
        )


def _require_umap() -> Any:
    """Lazily import the optional ``umap-learn`` dependency with an actionable error."""
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "PanelUMAP requires the optional `umap-learn` dependency, which is "
            "not installed. Install it with `pip install panelary[umap]` "
            "(or `pip install umap-learn`)."
        ) from exc
    return umap


class PanelUMAP(_PanelReducer):
    """Leak-safe panel UMAP manifold embedding (optional dependency).

    Wraps :class:`umap.UMAP`, imported lazily so importing this module never
    requires ``umap-learn``. The embedding is fit on the training rows only and
    applied to new rows via ``UMAP.transform``. UMAP exposes no ``components_``,
    so sign-fixing falls back to the score-based rule.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimensionality (UMAP has no explained-variance target).
    n_neighbors : int, default 15
        UMAP local-neighbourhood size.
    min_dist : float, default 0.1
        UMAP minimum embedding distance.
    **kwargs
        See :class:`~panelary.reduce._base._PanelReducer`.

    Raises
    ------
    ImportError
        At :meth:`fit` time if ``umap-learn`` is not installed.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_components=n_components, **kwargs)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def _make_reducer(self, n_components: int) -> Any:
        umap = _require_umap()

        return umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )


_METHODS: dict[str, type[_PanelReducer]] = {
    "pca": PanelPCA,
    "truncated_svd": PanelSVD,
    "svd": PanelSVD,
    "factor_analysis": PanelFactorAnalysis,
    "gaussian_random_projection": PanelRandomProjection,
    "random_projection": PanelRandomProjection,
    "kernel_pca": PanelKernelPCA,
    "nmf": PanelNMF,
    "umap": PanelUMAP,
}


def reduce_features(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "pca",
    n_components: int | None = None,
    columns: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> PanelFrame:
    """Fit-and-transform convenience over the panel reducers.

    Builds the reducer named by ``method`` and returns
    ``reducer.fit_transform(data)``. Because this both fits and transforms on the
    same rows it is only appropriate on training data (or exploratory single
    frames); inside cross-validation, construct the reducer and use the
    ``fit`` / ``transform`` split explicitly.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        The panel to reduce.
    method : str, default "pca"
        One of ``pca``, ``truncated_svd``/``svd``, ``factor_analysis``,
        ``gaussian_random_projection``/``random_projection``, ``kernel_pca``,
        ``nmf``, ``umap``.
    n_components, columns, entity, time, **kwargs
        Forwarded to the reducer constructor (see
        :class:`~panelary.reduce._base._PanelReducer`).

    Returns
    -------
    PanelFrame
        Keys plus the component columns (or the panel with components appended
        when ``keep_features=True``).
    """
    key = method.lower()
    if key not in _METHODS:
        raise ValueError(
            f"reduce_features: unknown method {method!r}. "
            f"Choose one of {sorted(_METHODS)}."
        )
    reducer = _METHODS[key](
        n_components=n_components,
        columns=columns,
        entity=entity,
        time=time,
        **kwargs,
    )
    return reducer.fit_transform(data, entity=entity, time=time)
