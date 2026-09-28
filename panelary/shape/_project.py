"""Seeded random projections: very sparse JL and the subsampled randomized Hadamard transform.

Both are **stateless given the seed**: ``fit`` records the input columns (hence
the width ``d``), the seed and ``n_components`` -- the projection is a
deterministic function of those three and of nothing in the data. Fitting on two
disjoint datasets therefore gives byte-identical state, the same
``fit_is_empty = True`` claim (and test) as the ``embed/`` contract; and because
there is nothing to leak through, these outrank a fitted PCA as the
leak-conscious default compressor.

**The Johnson-Lindenstrauss bound does not license ``k = 64``.** Preserving all
pairwise distances among ``n = 1e6`` points to ``eps = 0.1`` needs on the order
of ``4 ln(n) / (eps^2/2 - eps^3/3) ~ 1e4`` dimensions. Small outputs rest on
empirical performance, not on the theorem. (The ``embed/`` contract's
``_compress.py`` says the same; the two must not contradict each other.)

References: P. Li, T. Hastie, K. Church (2006), "Very sparse random
projections", KDD; N. Ailon, B. Chazelle (2009), "The fast Johnson-Lindenstrauss
transform and approximate nearest neighbors", SIAM J. Comput. 39(1).
Clean-room implementations; ``fwht`` is a plain radix-2 butterfly (no scipy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

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

    from numpy.typing import NDArray

__all__ = [
    "SRHT",
    "SparseRandomProjection",
    "fwht",
    "hadamard_rows",
    "sparse_rp_matrix",
    "srht",
]


def _resolve_density(d: int, density: float | str) -> float:
    if density == "auto":
        return 1.0 / np.sqrt(d)
    if isinstance(density, str):
        raise ValueError(
            f"density must be 'auto' or a float in (0, 1], got {density!r}."
        )
    dens = float(density)
    if not 0.0 < dens <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density!r}.")
    return dens


def sparse_rp_matrix(
    d: int,
    k: int,
    *,
    density: float | str = "auto",
    seed: int,
) -> NDArray[np.float64]:
    """Li et al. (2006) very sparse random projection matrix, ``(d, k)``.

    Entries are ``+sqrt(s / k)`` and ``-sqrt(s / k)`` with probability
    ``1 / (2s)`` each and ``0`` otherwise, where ``s = 1 / density``; ``density
    ="auto"`` is ``1 / sqrt(d)``. ``X @ R`` preserves squared norms in
    expectation.

    Parameters
    ----------
    d, k : int
        Input and output width.
    density : float or "auto", default "auto"
    seed : int
        Explicit seed; the matrix is a deterministic function of
        ``(d, k, density, seed)``.

    Returns
    -------
    numpy.ndarray
        Dense ``(d, k)`` float64 (sparse storage would need scipy).
    """
    d = check_positive_int(d, "d", "sparse_rp_matrix")
    k = check_positive_int(k, "k", "sparse_rp_matrix")
    seed = check_seed(seed, "sparse_rp_matrix")
    dens = _resolve_density(d, density)
    rng = np.random.default_rng(seed)
    u = rng.random((d, k))
    sign = np.where(rng.random((d, k)) < 0.5, -1.0, 1.0)
    return np.where(u < dens, sign, 0.0) * np.sqrt(1.0 / (dens * k))


def fwht(x: NDArray[Any]) -> NDArray[np.float64]:
    """Unnormalised fast Walsh-Hadamard transform along the last axis.

    Iterative radix-2 butterfly (no bit reversal: the Walsh-Hadamard matrix is
    in natural / Sylvester order), ``log2(D)`` vectorised stages, done in place
    on a float64 copy. ``fwht(fwht(x)) == D * x``.

    Parameters
    ----------
    x : numpy.ndarray
        ``(..., D)`` with ``D`` a power of two. Pad with zeros first otherwise.

    Returns
    -------
    numpy.ndarray
        ``H_D @ x`` along the last axis, ``H`` with ``+-1`` entries.
    """
    a = np.array(x, dtype=np.float64, copy=True)
    D = a.shape[-1]
    if D < 1 or D & (D - 1):
        raise ValueError(
            f"fwht: the last axis has length {D}, not a power of two; zero-pad it "
            "to the next power of two."
        )
    lead = a.shape[:-1]
    h = 1
    while h < D:
        v = a.reshape(*lead, D // (2 * h), 2, h)
        top = v[..., 0, :].copy()
        v[..., 0, :] += v[..., 1, :]
        v[..., 1, :] = top - v[..., 1, :]
        h *= 2
    return a


def hadamard_rows(rows: NDArray[Any], d: int) -> NDArray[np.float64]:
    """Rows ``rows`` of the Sylvester Hadamard matrix, first ``d`` columns: ``(-1)^popcount(i & j)``."""
    i = np.asarray(rows, dtype=np.int64)[:, None]
    j = np.arange(d, dtype=np.int64)[None, :]
    v = i & j
    parity = np.zeros(v.shape, dtype=np.int64)
    while np.any(v):
        parity ^= v & 1
        v = v >> 1
    return np.asarray(1.0 - 2.0 * parity, dtype=np.float64)


def _srht_draw(
    d: int, k: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    D = 1 << max(0, int(np.ceil(np.log2(d)))) if d > 1 else 1
    if k > D:
        raise ValueError(f"srht: k={k} exceeds the padded width {D}.")
    rng = np.random.default_rng(seed)
    signs = np.where(rng.random(d) < 0.5, -1.0, 1.0)
    sample = np.sort(rng.choice(D, size=k, replace=False)).astype(np.int64)
    return signs, sample, D


def srht(X: NDArray[Any], k: int, *, seed: int) -> NDArray[np.float64]:
    """Subsampled randomized Hadamard transform: ``sample(H @ (D x)) / sqrt(k)``.

    ``D`` is a seeded random sign diagonal, ``H`` the Walsh-Hadamard transform
    on the input zero-padded to the next power of two, and ``k`` coordinates
    are sampled without replacement. Squared norms are preserved in
    expectation.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, d)`` finite.
    k : int
        Output width, ``<= next_pow2(d)``.
    seed : int
        Explicit seed for the signs and the sample.

    Returns
    -------
    numpy.ndarray
        ``(n, k)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"srht: X must be 2-D, got shape {X.shape}.")
    n, d = X.shape
    k = check_positive_int(k, "k", "srht")
    seed = check_seed(seed, "srht")
    signs, sample, D = _srht_draw(d, k, seed)
    padded = np.zeros((n, D), dtype=np.float64)
    padded[:, :d] = X * signs
    return fwht(padded)[:, sample] / np.sqrt(k)


class SparseRandomProjection(_FeatureAxisTransform):
    """Seeded very sparse random projection (Li et al. 2006) -- stateless given the seed.

    ``fit`` records the columns and draws the ``(d, k)`` matrix from
    ``(d, n_components, density, seed)`` alone; it is handed a zero-row frame
    (``fit_is_empty = True``), so it provably learns nothing from the data.
    No standardisation is applied (that would be a fitted statistic); scale
    columns upstream if they are on very different scales.

    **JL honesty note:** the Johnson-Lindenstrauss bound does not license small
    ``n_components``; at ``n = 1e6`` and ``eps = 0.1`` it wants ~1e4
    dimensions. Small outputs rest on empirical performance, not theory.

    Parameters
    ----------
    n_components : int, default 64
    density : float or "auto", default "auto"
        ``"auto"`` = ``1 / sqrt(d)``.
    seed : int, default 0
    prefix : str, default "srp"
    keep_features, dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._feature._FeatureAxisTransform`.

    Attributes
    ----------
    components_ : numpy.ndarray
        ``(k, d)`` projection (``output = X @ components_.T``).
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="none",
        streaming="mergeable",
        cost_hint="O(n d k density)",
    )
    _default_prefix = "srp"

    def __init__(
        self,
        *,
        n_components: int = 64,
        density: float | str = "auto",
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
            n_components, "n_components", type(self).__name__
        )
        self.density = density
        self.seed = check_seed(seed, type(self).__name__)
        self.components_: NDArray[np.float64] | None = None

    def _out_width(self, in_width: int) -> int:
        return self.n_components

    def _scratch_bytes(self, shape: InputShape, m: int) -> float:
        return 8.0 * (shape.rows * (shape.width + m) + shape.width * m)

    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        d = len(self.feature_names_in_)
        self.components_ = sparse_rp_matrix(
            d, self.n_components, density=self.density, seed=self.seed
        ).T.copy()

    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.components_ is not None
        return X @ self.components_.T

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_loadings

        assert self.components_ is not None
        return explain_loadings(
            self._output_names(), self.feature_names_in_, self.components_
        )


class SRHT(_FeatureAxisTransform):
    """Subsampled randomized Hadamard transform -- stateless given the seed.

    ``O(n D log D)`` with ``D = next_pow2(d)`` (a butterfly, not a matmul), and
    denser mixing than a sparse projection: every output mixes every input.
    Same statelessness (``fit_is_empty = True``) and the same JL honesty note as
    :class:`SparseRandomProjection`.

    Parameters
    ----------
    n_components : int, default 64
        ``<= next_pow2(d)``.
    seed : int, default 0
    prefix : str, default "srht"
    keep_features, dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._feature._FeatureAxisTransform`.

    Attributes
    ----------
    signs_ : numpy.ndarray
        ``(d,)`` random signs.
    sample_ : numpy.ndarray
        ``(k,)`` sampled Hadamard rows.
    """

    panel_safe = True
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.FEATURE,
        width_rule="exact",
        invertible="none",
        streaming="mergeable",
        cost_hint="O(n d log d)",
    )
    _default_prefix = "srht"

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
        self.n_components = check_positive_int(n_components, "n_components", "SRHT")
        self.seed = check_seed(seed, "SRHT")
        self.signs_: NDArray[np.float64] | None = None
        self.sample_: NDArray[np.int64] | None = None
        self.padded_width_: int = 0

    def _out_width(self, in_width: int) -> int:
        return self.n_components

    def _scratch_bytes(self, shape: InputShape, m: int) -> float:
        D = 1 << max(0, int(np.ceil(np.log2(max(shape.width, 1)))))
        return 8.0 * shape.rows * (shape.width + 2 * D + m)

    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        d = len(self.feature_names_in_)
        self.signs_, self.sample_, self.padded_width_ = _srht_draw(
            d, self.n_components, self.seed
        )

    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.signs_ is not None and self.sample_ is not None
        padded = np.zeros((X.shape[0], self.padded_width_), dtype=np.float64)
        padded[:, : X.shape[1]] = X * self.signs_
        return fwht(padded)[:, self.sample_] / np.sqrt(self.n_components)

    @property
    def components_(self) -> NDArray[np.float64] | None:
        """The explicit ``(k, d)`` matrix equivalent to the transform."""
        if self.signs_ is None or self.sample_ is None:
            return None
        H = hadamard_rows(self.sample_, len(self.feature_names_in_))
        return H * self.signs_[None, :] / np.sqrt(self.n_components)

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_loadings

        C = self.components_
        assert C is not None
        return explain_loadings(self._output_names(), self.feature_names_in_, C)


register_shape_spec(
    SparseRandomProjection,
    name="sparse_rp",
    params={"n_components": int, "density": float, "seed": int},
    tier="B",
    safe_scope="rowwise",
    source="Li, Hastie & Church (2006), KDD; clean-room, Panelary",
)
register_shape_spec(
    SRHT,
    name="srht",
    params={"n_components": int, "seed": int},
    tier="B",
    safe_scope="rowwise",
    source="Ailon & Chazelle (2009), SIAM J. Comput. 39(1); clean-room, Panelary",
)
