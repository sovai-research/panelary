"""Pivoted QR, interpolative decomposition and CUR: compression onto *real* columns.

``numpy.linalg.qr`` has no column pivoting and ``scipy.linalg.qr(pivoting=True)``
is a non-default dependency, so :func:`pivoted_qr` is hand-rolled: greedy
column-pivoted modified Gram-Schmidt that, at each step, takes the column of
largest residual norm and **downdates** the remaining norms instead of
recomputing them (recomputing only where cancellation has eaten the downdated
value). For ``k << d`` it is ``O(n d k)``, and every step is a BLAS
matrix-vector product.

:func:`interpolative` turns the pivots into an interpolative decomposition
(ID): ``A ~= A[:, cols] @ Z`` with ``Z[:, cols] = I``. That is the
explainability primitive of the shape algebra -- :class:`ColumnSubset` returns
``k`` of the user's actual columns, under their actual names, plus ``Z`` saying
how every other column is reconstructed from them. :class:`CUR` adds the row
side: ``A ~= C @ U @ R`` with ``C`` real columns and ``R`` real (training)
observations.

References: P. Businger, G. H. Golub (1965), "Linear least squares solutions by
Householder transformations" (column pivoting); H. Cheng, Z. Gimbutas, P.-G.
Martinsson, V. Rokhlin (2005), "On the compression of low rank matrices", SIAM
J. Sci. Comput. 26(4) (the ID); M. W. Mahoney, P. Drineas (2009), "CUR matrix
decompositions for improved data analysis", PNAS 106(3). Clean-room
implementations; numpy only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.shape._axes import (
    Axis,
    InputShape,
    Intent,
    ShapeSpec,
    check_positive_int,
    register_shape_spec,
    relative_frobenius_error,
)
from panelary.shape._feature import _FeatureAxisTransform, fit_scaler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "CUR",
    "ColumnSubset",
    "CURFactors",
    "interpolative",
    "pivoted_qr",
]

#: Relative residual norm below which a column is treated as already spanned.
_RANK_TOL = 1e-12


def pivoted_qr(
    A: NDArray[Any], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Greedy column-pivoted QR by modified Gram-Schmidt: ``A[:, piv] ~= Q @ R``.

    The first ``r`` pivoted columns are reproduced exactly
    (``A[:, piv[:r]] == Q @ R[:, :r]`` to rounding); every other column is
    represented by its projection onto ``span(Q)``, the residual being what a
    rank-``r`` truncation discards.

    Parameters
    ----------
    A : numpy.ndarray
        ``(m, n)`` finite matrix (NaN is refused, not imputed).
    k : int
        Number of pivots, ``1 <= k <= min(m, n)``.

    Returns
    -------
    Q : numpy.ndarray
        ``(m, r)`` orthonormal columns.
    R : numpy.ndarray
        ``(r, n)``, upper trapezoidal in pivoted column order: ``R[:, :r]`` is
        upper triangular with a non-increasing, positive diagonal.
    piv : numpy.ndarray
        ``(n,)`` int64 column permutation; ``piv[:r]`` are the chosen columns,
        in the order chosen.

    ``r == k`` unless ``A`` has numerical rank below ``k`` (every remaining
    residual under ``1e-12`` of the largest column norm), in which case the
    factorisation stops early at ``r = rank``: an exhausted column space has
    no further column worth choosing, and dividing by a zero residual would
    manufacture one.

    Raises
    ------
    ValueError
        On a non-finite ``A`` or an out-of-range ``k``.

    Notes
    -----
    Ties in the residual norm go to the lowest column index (``np.argmax``), so
    the pivots are deterministic.
    """
    W = np.array(A, dtype=np.float64, copy=True)
    if W.ndim != 2:
        raise ValueError(f"pivoted_qr: A must be 2-D, got shape {W.shape}.")
    m, n = W.shape
    k = check_positive_int(k, "k", "pivoted_qr")
    if k > min(m, n):
        raise ValueError(f"pivoted_qr: k={k} exceeds min(A.shape)={min(m, n)}.")
    if not np.isfinite(W).all():
        raise ValueError("pivoted_qr: A contains NaN/inf; refusing to impute.")
    piv = np.arange(n, dtype=np.int64)
    norms = np.einsum("ij,ij->j", W, W)
    ref = norms.copy()
    scale = float(norms.max()) if n else 0.0
    Q = np.zeros((m, k), dtype=np.float64)
    R = np.zeros((k, n), dtype=np.float64)
    r = 0
    for j in range(k):
        p = j + int(np.argmax(norms[j:]))
        if norms[p] <= _RANK_TOL**2 * scale or scale == 0.0:
            break
        if p != j:
            for arr in (W, R):
                arr[:, [j, p]] = arr[:, [p, j]]
            for vec in (piv, norms, ref):
                vec[[j, p]] = vec[[p, j]]
        w = W[:, j]
        # One re-orthogonalisation pass against the chosen directions ("twice
        # is enough"); the correction is folded into R so A[:, piv] = Q R holds.
        if j:
            c = Q[:, :j].T @ w
            w = w - Q[:, :j] @ c
            R[:j, j] += c
        rjj = float(np.sqrt(w @ w))
        if rjj <= _RANK_TOL * np.sqrt(scale):
            break
        q = w / rjj
        Q[:, j] = q
        R[j, j] = rjj
        if j + 1 < n:
            rrow = q @ W[:, j + 1 :]
            R[j, j + 1 :] = rrow
            W[:, j + 1 :] -= np.outer(q, rrow)
            norms[j + 1 :] -= rrow * rrow
            # Downdating subtracts nearly equal numbers once a column is mostly
            # spanned; recompute those norms exactly (LAPACK xGEQP3's safeguard).
            stale = np.flatnonzero(norms[j + 1 :] <= 1e-8 * ref[j + 1 :]) + j + 1
            if stale.size:
                norms[stale] = np.einsum("ij,ij->j", W[:, stale], W[:, stale])
                ref[stale] = norms[stale]
            np.maximum(norms, 0.0, out=norms)
        r = j + 1
    return Q[:, :r], R[:r], piv


def interpolative(A: NDArray[Any], k: int) -> tuple[list[int], NDArray[np.float64]]:
    """Rank-``k`` interpolative decomposition: ``A ~= A[:, cols] @ Z``.

    ``cols`` are the first ``k`` pivots of :func:`pivoted_qr` and
    ``Z = [I, R11^{-1} R12]`` un-permuted, so ``Z[:, cols]`` is exactly the
    identity -- the chosen columns reproduce themselves bit for bit.

    Parameters
    ----------
    A : numpy.ndarray
        ``(m, n)`` finite matrix.
    k : int
        Number of columns to keep.

    Returns
    -------
    cols : list of int
        Indices of the kept columns, in pivot order (most informative first).
        Shorter than ``k`` if ``A`` has numerical rank below ``k``.
    Z : numpy.ndarray
        ``(len(cols), n)`` interpolation matrix.
    """
    A = np.asarray(A, dtype=np.float64)
    _, R, piv = pivoted_qr(A, k)
    r = R.shape[0]
    cols = [int(c) for c in piv[:r]]
    n = A.shape[1]
    Z = np.zeros((r, n), dtype=np.float64)
    if r:
        T = np.linalg.solve(R[:, :r], R[:, r:]) if r < n else np.zeros((r, 0))
        Z[:, piv[r:]] = T
        Z[:, piv[:r]] = np.eye(r)
    return cols, Z


class CURFactors(NamedTuple):
    """The three factors of a CUR decomposition, ``A ~= C @ U @ R``.

    Attributes
    ----------
    columns : list of str
        Names of the kept columns (``C = A[:, columns]``).
    rows : polars.DataFrame
        ``(entity, time)`` keys of the kept training observations
        (``R = A[rows, :]``).
    U : numpy.ndarray
        ``(k_c, k_r)`` linking matrix, in the transform's internal
        (standardised, if ``standardize=True``) units.
    C : numpy.ndarray
        ``(n_fit, k_c)`` kept columns of the training matrix (internal units).
    R : numpy.ndarray
        ``(k_r, d)`` kept rows of the training matrix (internal units).
    """

    columns: list[str]
    rows: pl.DataFrame
    U: NDArray[np.float64]
    C: NDArray[np.float64]
    R: NDArray[np.float64]


class ColumnSubset(_FeatureAxisTransform):
    """Compress the feature axis onto ``k`` of the user's **real** columns (via ID).

    ``fit`` runs a pivoted QR on the training matrix (standardised by default,
    so a column is not chosen merely for being measured in larger units) and
    keeps the ``k`` columns it pivots on first; ``transform`` returns those
    columns, under their own names, unchanged. ``Z_`` records how every input
    column is reconstructed from the kept ones.

    Parameters
    ----------
    k : int, default 8
        Columns to keep (fewer if the training matrix has lower rank).
    standardize : bool, default True
        Choose (and interpolate) on training-standardised columns. The output
        is always the raw column values.
    dtype, columns, max_bytes, entity, time
        See :class:`~panelary.shape._feature._FeatureAxisTransform`.
        ``keep_features`` is refused: the output columns are input columns.

    Attributes
    ----------
    selected_ : list of str
        The kept columns, most informative first.
    Z_ : numpy.ndarray
        ``(k, d)`` interpolation matrix (internal units): ``X_s ~= X_s[:, sel] @ Z_``.
    mean_, scale_ : numpy.ndarray
        Training column mean and scale (ones when ``standardize=False``).
    reconstruction_error_ : float
        Relative Frobenius error of the ID on the training matrix (internal
        units).

    NaN policy
    ----------
    ``fit`` uses complete training rows only. ``transform`` copies the kept
    columns as they are (null -> NaN); a missing *unkept* column does not blank
    the row, since the output never reads it.

    Notes
    -----
    ``panel_safe = True``; ``leakage_safe = True`` *as an estimator* (the
    choice of columns is learned in ``fit`` from training rows only) --
    ``safe_scope="window"`` in the registry, exactly like PCA.
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
    _knob = "k"

    def __init__(
        self,
        *,
        k: int = 8,
        standardize: bool = True,
        keep_features: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        owner = type(self).__name__
        if keep_features:
            raise ValueError(
                f"{owner}: keep_features=True would duplicate the kept columns, "
                "which are input columns already; select them from the input "
                "instead (`frame.select(t.selected_)`)."
            )
        super().__init__(
            prefix=None,
            keep_features=False,
            dtype=dtype,
            columns=columns,
            max_bytes=max_bytes,
            entity=entity,
            time=time,
        )
        self.k = check_positive_int(k, "k", owner)
        self.standardize = bool(standardize)
        self.selected_: list[str] = []
        self.selected_idx_: NDArray[np.int64] | None = None
        self.Z_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.reconstruction_error_: float | None = None

    # ------------------------------------------------------------------ #
    def _out_width(self, in_width: int) -> int:
        return len(self.selected_) or min(self.k, in_width)

    def _output_names(self) -> list[str]:
        return list(self.selected_)

    def _scratch_bytes(self, shape: InputShape, m: int) -> float:
        return 8.0 * (3 * shape.rows * shape.width + m * shape.width)

    def _standardise(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.standardize:
            self.mean_, self.scale_ = fit_scaler(X)
        else:
            self.mean_, self.scale_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
        return (X - self.mean_) / self.scale_

    def _fit_matrix(self, X: NDArray[np.float64]) -> None:
        Xs = self._standardise(X)
        k = min(self.k, *Xs.shape)
        cols, Z = interpolative(Xs, k)
        self.selected_idx_ = np.asarray(cols, dtype=np.int64)
        self.selected_ = [self.feature_names_in_[c] for c in cols]
        self.Z_ = Z
        self.reconstruction_error_ = relative_frobenius_error(Xs, Xs[:, cols] @ Z)

    def _map(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.selected_idx_ is not None
        return X[:, self.selected_idx_]

    def _apply(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Real columns: copy them as they are; unkept columns are never read.
        return np.array(self._map(X), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def inverse_transform(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        """Reconstruct all ``d`` input columns from the ``k`` kept ones.

        Parameters
        ----------
        Z : PanelFrame | polars.DataFrame | numpy.ndarray
            This transform's output (the kept columns) or an ``(n, k)`` array.

        Returns
        -------
        numpy.ndarray
            ``(n, d)`` in original units, columns in ``feature_names_in_``
            order. The kept columns are reproduced exactly.
        """
        self._check_fitted("inverse_transform")
        assert self.Z_ is not None and self.selected_idx_ is not None
        if isinstance(Z, np.ndarray):
            C = np.asarray(Z, dtype=np.float64)
        else:
            frame = Z.collect() if isinstance(Z, PanelFrame) else Z
            C = frame.select(self.selected_).to_numpy().astype(np.float64)
        idx = self.selected_idx_
        assert self.mean_ is not None and self.scale_ is not None
        Cs = (C - self.mean_[idx]) / self.scale_[idx]
        return (Cs @ self.Z_) * self.scale_ + self.mean_

    def reconstruction_error(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> float:
        """Relative Frobenius error of the reconstruction over complete rows of ``X``."""
        self._check_fitted("reconstruction_error")
        panel = self._as_panel(
            X, method="reconstruction_error", entity=entity, time=time
        )
        M = self._matrix(panel.lazy(), self.feature_names_in_)
        M = M[np.isfinite(M).all(axis=1)]
        return relative_frobenius_error(M, self.inverse_transform(self._map(M)))

    def interpolation(self) -> pl.DataFrame:
        """``Z_`` as a tidy frame: how each input column is rebuilt from the kept ones.

        Returns
        -------
        polars.DataFrame
            ``kept_feature, reconstructed_feature, coefficient`` -- one row per
            (kept, input) pair, coefficients in internal (standardised) units.
        """
        self._check_fitted("interpolation")
        assert self.Z_ is not None
        k, d = self.Z_.shape
        return pl.DataFrame(
            {
                "kept_feature": [c for c in self.selected_ for _ in range(d)],
                "reconstructed_feature": list(self.feature_names_in_) * k,
                "coefficient": self.Z_.reshape(-1),
            }
        )

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_frame

        n = len(self.selected_)
        return explain_frame(
            list(self.selected_),
            list(self.selected_),
            np.ones(n),
            reconstruction_error=self.reconstruction_error_,
        )


class CUR(ColumnSubset):
    """CUR factorisation of the training matrix: real columns, real rows, a link.

    ``A ~= C @ U @ R`` where ``C`` holds ``k`` real columns (chosen by pivoted
    QR, as :class:`ColumnSubset`), ``R`` holds ``k_rows`` real training
    **observations** (pivoted QR on ``A.T``) and ``U = C^+ A R^+`` is the
    Frobenius-optimal link. :meth:`factors` returns all three, with the kept
    rows named by their ``(entity, time)`` keys -- "these dates of these
    entities are the most representative of the sample".

    ``transform`` returns the kept columns (the compressed view, as
    :class:`ColumnSubset`); :meth:`inverse_transform` rebuilds all columns
    through ``U @ R``, i.e. restricted to the row space of the kept
    observations.

    Parameters
    ----------
    k : int, default 8
        Columns to keep.
    k_rows : int, optional
        Rows to keep; default ``k``.
    standardize, dtype, columns, max_bytes, entity, time
        As :class:`ColumnSubset`.

    Attributes
    ----------
    selected_ : list of str
    rows_ : polars.DataFrame
        ``(entity, time)`` of the kept training observations.
    U_ : numpy.ndarray
        ``(k, k_rows)`` link.
    R_ : numpy.ndarray
        ``(k_rows, d)`` kept observations (internal units).
    """

    spec = ShapeSpec(
        intent=Intent.FACTORIZE,
        axis=Axis.FEATURE,
        width_rule="rank_dependent",
        invertible="from_factors",
        streaming="batch",
        cost_hint="O(n d k)",
    )

    def __init__(
        self,
        *,
        k: int = 8,
        k_rows: int | None = None,
        standardize: bool = True,
        keep_features: bool = False,
        dtype: Any = None,
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(
            k=k,
            standardize=standardize,
            keep_features=keep_features,
            dtype=dtype,
            columns=columns,
            max_bytes=max_bytes,
            entity=entity,
            time=time,
        )
        self.k_rows = (
            self.k if k_rows is None else check_positive_int(k_rows, "k_rows", "CUR")
        )
        self.rows_: pl.DataFrame | None = None
        self.U_: NDArray[np.float64] | None = None
        self.R_: NDArray[np.float64] | None = None
        self._C: NDArray[np.float64] | None = None

    def _fit(self, panel: PanelFrame) -> None:
        # Needs the training keys (to name the kept rows), so it reads the
        # frame itself rather than the bare matrix.
        self.feature_names_in_ = self._resolve_columns(panel)
        ent, tim = panel.entity_col, panel.time_col
        frame = panel.collect()
        X = self._matrix(frame, self.feature_names_in_)
        ok = np.asarray(np.isfinite(X).all(axis=1), dtype=np.bool_)
        X = X[ok]
        self._enforce_budget(InputShape(rows=X.shape[0], width=X.shape[1], entities=0))
        if X.shape[0] == 0:
            raise ValueError(
                f"CUR.fit: no row has all of {self.feature_names_in_} observed; "
                "nothing to fit on."
            )
        self.n_rows_fit_ = int(X.shape[0])
        self._fit_matrix(X)
        assert self.mean_ is not None and self.scale_ is not None
        Xs = (X - self.mean_) / self.scale_
        kr = min(self.k_rows, *Xs.shape)
        rows, _ = interpolative(Xs.T, kr)
        assert self.selected_idx_ is not None
        C = Xs[:, self.selected_idx_]
        R = Xs[rows]
        self.U_ = np.linalg.pinv(C) @ Xs @ np.linalg.pinv(R)
        self.R_ = R
        self._C = C
        self.rows_ = frame.select(ent, tim).filter(pl.Series(ok))[rows]
        self.reconstruction_error_ = relative_frobenius_error(Xs, C @ self.U_ @ R)

    def factors(self) -> CURFactors:
        """The fitted ``(C, U, R)`` with the kept columns and rows named.

        Returns
        -------
        CURFactors
        """
        self._check_fitted("factors")
        assert self.rows_ is not None and self.U_ is not None
        assert self._C is not None and self.R_ is not None
        return CURFactors(list(self.selected_), self.rows_, self.U_, self._C, self.R_)

    def inverse_transform(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        """Reconstruct all ``d`` columns as ``C @ U @ R`` (original units).

        Parameters
        ----------
        Z : PanelFrame | polars.DataFrame | numpy.ndarray
            This transform's output (the kept columns) or an ``(n, k)`` array.

        Returns
        -------
        numpy.ndarray
            ``(n, d)``, columns in ``feature_names_in_`` order.
        """
        self._check_fitted("inverse_transform")
        assert self.U_ is not None and self.R_ is not None
        assert self.selected_idx_ is not None
        if isinstance(Z, np.ndarray):
            C = np.asarray(Z, dtype=np.float64)
        else:
            frame = Z.collect() if isinstance(Z, PanelFrame) else Z
            C = frame.select(self.selected_).to_numpy().astype(np.float64)
        idx = self.selected_idx_
        assert self.mean_ is not None and self.scale_ is not None
        Cs = (C - self.mean_[idx]) / self.scale_[idx]
        return (Cs @ self.U_ @ self.R_) * self.scale_ + self.mean_


register_shape_spec(
    ColumnSubset,
    name="column_subset",
    params={"k": int, "standardize": bool},
    tier="B",
    safe_scope="window",
    source=(
        "Businger & Golub (1965); Cheng, Gimbutas, Martinsson & Rokhlin (2005), "
        "SIAM J. Sci. Comput. 26(4); clean-room, Panelary"
    ),
)
register_shape_spec(
    CUR,
    name="cur",
    params={"k": int, "k_rows": int, "standardize": bool},
    tier="C",
    safe_scope="window",
    source="Mahoney & Drineas (2009), PNAS 106(3); clean-room, Panelary",
)
