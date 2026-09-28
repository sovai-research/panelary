"""Sketches: CountSketch (seeded feature hashing) and Frequent Directions (deterministic).

``CountSketch`` (Charikar, Chen & Farach-Colton 2002) maps ``d`` features to
``k`` buckets with one hash and one sign per feature -- ``O(nnz)`` work, a
linear map, stateless given the seed. It is the first half of TensorSketch
(Pham & Pagh 2013), which ``embed/_sketch.py`` composes with ``np.fft.rfft``;
this module owns the sketch, ``embed/`` owns the polynomial-kernel composition.

``FrequentDirections`` (Liberty 2013; Ghashami, Liberty, Phillips & Woodruff
2016) keeps an ``ell x d`` matrix ``B`` whose Gram matrix approximates the
data's, with a proven error bar:

    ||A^T A - B^T B||_2 <= ||A||_F^2 / ell        (and the merged sketch too)

It is **deterministic** -- no seed, so nothing to leak through -- and
**mergeable**: sketch each entity (or each shard) independently, then
:meth:`FrequentDirections.merge`. Merging is associative *up to the bound*:
different groupings give different ``B`` (each shrink subtracts a
data-dependent amount), but every grouping satisfies the same bound against the
full data, and when no shrink is triggered (total rank < ``ell``) every
grouping reproduces ``A^T A`` exactly up to rounding.

Clean-room implementations from the papers; numpy only.
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
    check_positive_int,
    check_seed,
    register_shape_spec,
)
from panelary.shape._feature import _FeatureAxisTransform

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from numpy.typing import NDArray

__all__ = [
    "CountSketch",
    "FrequentDirections",
    "count_sketch",
    "count_sketch_hashes",
    "fd_shrink",
]


def count_sketch_hashes(
    d: int, k: int, *, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Seeded bucket ``h: [d] -> [k]`` and sign ``s: [d] -> {-1, +1}``."""
    d = check_positive_int(d, "d", "count_sketch_hashes")
    k = check_positive_int(k, "k", "count_sketch_hashes")
    rng = np.random.default_rng(check_seed(seed, "count_sketch_hashes"))
    h = rng.integers(0, k, size=d, dtype=np.int64)
    s = np.where(rng.random(d) < 0.5, -1.0, 1.0)
    return h, s


def count_sketch(
    X: NDArray[Any], h: NDArray[np.int64], s: NDArray[np.float64], k: int
) -> NDArray[np.float64]:
    """Apply a CountSketch: ``Y[:, b] = sum_{j : h(j) = b} s(j) X[:, j]``.

    One pass over the columns (sorted by bucket, then ``np.add.reduceat``): the
    ``(d, k)`` matrix is never formed.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, d)``.
    h, s : numpy.ndarray
        From :func:`count_sketch_hashes`.
    k : int
        Number of buckets.

    Returns
    -------
    numpy.ndarray
        ``(n, k)``; empty buckets are zero.
    """
    X = np.asarray(X, dtype=np.float64)
    order = np.argsort(h, kind="stable")
    hs = h[order]
    Xs = X[:, order] * s[order]
    out = np.zeros((X.shape[0], k), dtype=np.float64)
    if Xs.shape[1] == 0:
        return out
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    out[:, hs[starts]] = np.add.reduceat(Xs, starts, axis=1)
    return out


class CountSketch(_FeatureAxisTransform):
    """Seeded CountSketch (feature hashing with signs) -- stateless given the seed.

    ``fit`` draws the hash and sign arrays from ``(d, n_components, seed)``
    alone (``fit_is_empty = True``). The map is linear, so the sketch of a sum
    is the sum of sketches: two sketches with the same seed and width are the
    same sketch, which is what :meth:`merge` checks.

    Parameters
    ----------
    n_components : int, default 64
        Number of buckets ``k``.
    seed : int, default 0
    prefix : str, default "cs"
    keep_features, dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._feature._FeatureAxisTransform`.

    Attributes
    ----------
    hash_ : numpy.ndarray
        ``(d,)`` bucket of each input column.
    sign_ : numpy.ndarray
        ``(d,)`` sign of each input column.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.SKETCH,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="none",
        streaming="mergeable",
        cost_hint="O(n d)",
    )
    _default_prefix = "cs"

    def __init__(
        self,
        *,
        n_components: int = 64,
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
            n_components, "n_components", "CountSketch"
        )
        self.seed = check_seed(seed, "CountSketch")
        self.hash_: NDArray[np.int64] | None = None
        self.sign_: NDArray[np.float64] | None = None

    def _out_width(self, in_width: int) -> int:
        return self.n_components

    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        self.hash_, self.sign_ = count_sketch_hashes(
            len(self.feature_names_in_), self.n_components, seed=self.seed
        )

    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.hash_ is not None and self.sign_ is not None
        return count_sketch(X, self.hash_, self.sign_, self.n_components)

    def merge(self, other: CountSketch) -> CountSketch:
        """Merge two fitted sketches: identical hash functions are required.

        Raises
        ------
        ValueError
            If the seeds, widths or columns differ (their outputs live in
            different bucket spaces and cannot be added).
        """
        self._check_fitted("merge")
        other._check_fitted("merge")
        if (
            self.seed != other.seed
            or self.n_components != other.n_components
            or self.feature_names_in_ != other.feature_names_in_
        ):
            raise ValueError(
                "CountSketch.merge: sketches must share seed, n_components and "
                "columns; different hashes put features in different buckets."
            )
        return self

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_frame

        assert self.hash_ is not None and self.sign_ is not None
        names = self._output_names()
        return explain_frame(
            [names[b] for b in self.hash_],
            list(self.feature_names_in_),
            self.sign_,
        )


def fd_shrink(M: NDArray[np.float64], ell: int) -> tuple[NDArray[np.float64], float]:
    """One Frequent Directions shrink: SVD, subtract ``s_ell^2``, keep ``<= ell`` rows.

    Returns ``(B, delta)`` with ``B`` of shape ``(ell, d)`` and ``delta`` the
    squared singular value subtracted (0 when ``M`` has rank < ``ell``).
    """
    d = M.shape[1]
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    delta = float(s[ell - 1] ** 2) if s.size >= ell else 0.0
    shrunk = np.sqrt(np.maximum(s**2 - delta, 0.0))
    B = np.zeros((ell, d), dtype=np.float64)
    r = min(ell, s.size)
    B[:r] = shrunk[:r, None] * Vt[:r]
    return B, delta


class FrequentDirections(_FeatureAxisTransform):
    """Deterministic, mergeable sketch of the row space; projects onto its top directions.

    ``partial_fit`` streams rows into an ``ell x d`` sketch ``B`` (with the
    fast ``2 * ell`` buffer); ``merge`` combines two sketches; ``transform``
    projects rows onto the top ``n_components`` right singular vectors of
    ``B``, sign-fixed like :mod:`panelary.reduce`.

    FD approximates the **uncentred** second-moment matrix ``A^T A``; centre
    the inputs upstream (with training-fold statistics) if you want PCA.

    Parameters
    ----------
    ell : int, default 32
        Sketch size. Error bound ``||A^T A - B^T B||_2 <= ||A||_F^2 / ell``.
    n_components : int, default 8
        Directions used by :meth:`transform` (``<= ell``).
    prefix : str, default "fd"
    keep_features, dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._feature._FeatureAxisTransform`.

    Attributes
    ----------
    sketch_ : numpy.ndarray
        ``(ell, d)`` the sketch ``B``.
    frob2_ : float
        ``||A||_F^2`` of all rows absorbed so far (the bound's numerator).
    shrinkage_ : float
        Total squared mass subtracted by shrinks; ``<= frob2_ / ell``.
    n_seen_ : int
    components_ : numpy.ndarray
        ``(n_components, d)`` sign-fixed top directions of ``B``.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = False
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.SKETCH,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="approximate",
        streaming="mergeable",
        cost_hint="O(n d ell)",
    )
    _default_prefix = "fd"

    def __init__(
        self,
        *,
        ell: int = 32,
        n_components: int = 8,
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
        self.ell = check_positive_int(ell, "ell", "FrequentDirections")
        self.n_components = check_positive_int(
            n_components, "n_components", "FrequentDirections"
        )
        if self.n_components > self.ell:
            raise ValueError(
                f"FrequentDirections: n_components={n_components} exceeds ell={ell}."
            )
        self._reset()

    def _reset(self) -> None:
        self.sketch_: NDArray[np.float64] | None = None
        self._buffer: NDArray[np.float64] | None = None
        self._fill: int = 0
        self.frob2_: float = 0.0
        self.shrinkage_: float = 0.0
        self.n_seen_: int = 0
        self.components_: NDArray[np.float64] | None = None
        self.component_names_: list[str] = []

    @property
    def loadings_(self) -> NDArray[np.float64] | None:
        return self.components_

    def _out_width(self, in_width: int) -> int:
        return min(self.n_components, in_width)

    def _output_names(self) -> list[str]:
        return list(self.component_names_)

    def _scratch_bytes(self, shape: InputShape, m: int) -> float:
        return 8.0 * (shape.rows * (shape.width + m) + 6 * self.ell * shape.width)

    # ------------------------------------------------------------------ #
    def _absorb(self, X: NDArray[np.float64]) -> None:
        d = X.shape[1]
        ell = self.ell
        if self._buffer is None:
            self._buffer = np.zeros((2 * ell, d), dtype=np.float64)
            self._fill = 0
        if self._buffer.shape[1] != d:
            raise ValueError(
                "FrequentDirections: width changed between partial_fit calls."
            )
        self.frob2_ += float(np.sum(X * X))
        self.n_seen_ += int(X.shape[0])
        i = 0
        while i < X.shape[0]:
            room = 2 * ell - self._fill
            take = min(room, X.shape[0] - i)
            self._buffer[self._fill : self._fill + take] = X[i : i + take]
            self._fill += take
            i += take
            if self._fill == 2 * ell:
                B, delta = fd_shrink(self._buffer, ell)
                self.shrinkage_ += delta
                self._buffer[:ell] = B
                self._buffer[ell:] = 0.0
                self._fill = int(np.count_nonzero(np.any(B != 0.0, axis=1)))
        self._finalise()

    def _finalise(self) -> None:
        assert self._buffer is not None
        B, delta = fd_shrink(self._buffer[: max(self._fill, 1)], self.ell)
        self.shrinkage_ += delta
        # Canonicalise the stored buffer to the shrunk sketch so the state is
        # the sketch, whatever the arrival pattern of the rows.
        self._buffer[: self.ell] = B
        self._buffer[self.ell :] = 0.0
        self._fill = int(np.count_nonzero(np.any(B != 0.0, axis=1)))
        self.sketch_ = B
        d = B.shape[1]
        k = min(self.n_components, d)
        _, _, Vt = np.linalg.svd(B, full_matrices=False)
        Vt = Vt[:k]
        sign = np.array([_sign_of_max_abs(Vt[i]) for i in range(k)], dtype=np.float64)
        self.components_ = Vt * sign[:, None]
        self.component_names_ = [f"{self.prefix}_{i}" for i in range(1, k + 1)]

    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        self._reset()
        self._absorb(X)

    def partial_fit(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> Self:
        """Absorb more (training) rows into the sketch.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
        entity, time : str, optional

        Returns
        -------
        Self
        """
        panel = self._as_panel(X, method="partial_fit", entity=entity, time=time)
        if not self._fitted:
            return self.fit(panel)
        M = self._matrix(panel.lazy(), self.feature_names_in_)
        M = M[np.isfinite(M).all(axis=1)]
        self.n_rows_fit_ += int(M.shape[0])
        self._absorb(M)
        return self

    def merge(self, other: FrequentDirections) -> FrequentDirections:
        """Combine two sketches into a new one (neither input is modified).

        The merged sketch satisfies the FD bound against the union of both
        inputs' rows: ``||A^T A - B^T B||_2 <= (||A_1||_F^2 + ||A_2||_F^2) / ell``.

        Raises
        ------
        ValueError
            If ``ell`` or the columns differ.
        """
        self._check_fitted("merge")
        other._check_fitted("merge")
        if self.ell != other.ell or self.feature_names_in_ != other.feature_names_in_:
            raise ValueError("FrequentDirections.merge: ell and columns must match.")
        assert self.sketch_ is not None and other.sketch_ is not None
        out = FrequentDirections(
            ell=self.ell,
            n_components=self.n_components,
            prefix=self.prefix,
            keep_features=self.keep_features,
            dtype=self.dtype,
            columns=self.columns,
            max_bytes=self.max_bytes,
            entity=self._entity,
            time=self._time,
        )
        out.feature_names_in_ = list(self.feature_names_in_)
        out._buffer = np.vstack([self.sketch_, other.sketch_])
        out._fill = 2 * self.ell
        out.frob2_ = self.frob2_ + other.frob2_
        out.shrinkage_ = self.shrinkage_ + other.shrinkage_
        out.n_seen_ = self.n_seen_ + other.n_seen_
        out.n_rows_fit_ = self.n_rows_fit_ + other.n_rows_fit_
        out._finalise()
        out._fitted = True
        out._fit_panel = self._fit_panel
        return out

    def covariance_error_bound(self) -> float:
        """The FD guarantee ``||A||_F^2 / ell`` for the rows absorbed so far."""
        return self.frob2_ / self.ell

    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.components_ is not None
        return X @ self.components_.T

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_loadings

        assert self.components_ is not None
        return explain_loadings(
            self.component_names_, self.feature_names_in_, self.components_
        )


register_shape_spec(
    CountSketch,
    name="count_sketch",
    params={"n_components": int, "seed": int},
    tier="B",
    safe_scope="rowwise",
    source="Charikar, Chen & Farach-Colton (2002), ICALP; clean-room, Panelary",
)
register_shape_spec(
    FrequentDirections,
    name="frequent_directions",
    params={"ell": int, "n_components": int},
    tier="C",
    safe_scope="window",
    source="Liberty (2013), KDD; Ghashami, Liberty, Phillips & Woodruff (2016), SIAM J. Comput. 45(5); clean-room, Panelary",
)
