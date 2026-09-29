"""The estimate object and the per-window statistics every estimator shares.

:class:`CovEstimate` stores a covariance matrix in **diagonal-plus-low-rank**
form and never as ``N x N`` unless asked::

    Sigma = diag(scale) @ (B diag(g) B^T + E) @ diag(scale)

``E`` is either ``e0 * I`` (isotropic; with ``B`` orthonormal this is the
*spectral* fast path, where inverse, log-determinant and condition number are
exact in ``O(N r)``) or ``diag(e)`` (the *Woodbury* path, ``O(N r^2 + r^3)``).
Every estimator in this package emits one of the two forms.

:class:`WindowStats` is one centred (and in correlation space, standardised)
window. It computes the min-side Gram matrix and its eigendecomposition once,
lazily, and caches them, so every estimator and every market-state feature on
the same date share a single eigendecomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from panelary._internal._linalg import solve_vec

__all__ = [
    "CovEstimate",
    "SingularCovarianceError",
    "Spectrum",
    "WindowStats",
]

#: Above this many entities ``cond(exact=True)`` switches from a dense
#: ``eigvalsh`` to power iterations on ``Sigma`` and ``Sigma^{-1}``.
_DENSE_COND_MAX = 2_000
_POWER_ITERS = 50


class SingularCovarianceError(np.linalg.LinAlgError):
    """Raised by :meth:`CovEstimate.solve` on a singular estimate.

    A singular sample covariance is the failure mode that ``pinv`` hides: its
    minimum-variance weights carried 2,095x the oracle variance at ``N = 500``,
    ``W = 252`` (plan section 1.1). We raise instead of pseudo-inverting.
    """


def _as_f64(x: Any) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def _slogdet(A: NDArray[np.float64]) -> tuple[float, float]:
    """``np.linalg.slogdet`` without spurious FP flags.

    NumPy's Accelerate build raises "divide by zero encountered in slogdet"
    on well-conditioned matrices (measured on a 60 x 60 SPD matrix); the
    returned values are correct, so the flags are silenced here.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sign, ld = np.linalg.slogdet(A)
    return float(sign), float(ld)


@dataclass(frozen=True, slots=True, eq=False)
class Spectrum:
    """Eigen-decomposition of one window's sample second-moment matrix.

    Attributes
    ----------
    values : ndarray, shape (k,)
        Eigenvalues of ``S = Z^T Z / n_eff``, **descending**, ``k = min(n, p)``.
        Values below ``max(n, p) * eps * lambda_max`` are set to exactly 0.
        The remaining ``p - k`` eigenvalues of ``S`` are 0 and not stored.
    vectors : ndarray, shape (p, rank), or None
        Orthonormal primal eigenvectors for the non-null eigenvalues (``None``
        when only values were computed). Column signs follow
        :func:`panelary.reduce._common.sign_of_max_abs`, except the first,
        which is oriented so that its entries sum to a positive number (the
        "market up" direction).
    p : int
        Matrix dimension (entities).
    n_rows : int
        Rows in the window.
    n_eff : float
        Effective sample size (the divisor of ``S``).
    """

    values: NDArray[np.float64]
    vectors: NDArray[np.float64] | None
    p: int
    n_rows: int
    n_eff: float

    @property
    def rank(self) -> int:
        """Number of non-null eigenvalues."""
        return int(np.count_nonzero(self.values > 0.0))

    @property
    def trace(self) -> float:
        """``tr S`` (sum of all eigenvalues)."""
        return float(self.values.sum())

    @property
    def q(self) -> float:
        """Concentration ratio ``p / n_eff``."""
        return self.p / self.n_eff


@dataclass(eq=False)
class WindowStats:
    """One compacted, centred window and its cached Gram / spectrum.

    Build it with :func:`panelary.covariance._gram.window_stats`; do not fill
    the fields by hand.

    Attributes
    ----------
    Z : ndarray, shape (n, p)
        Centred (``space="covariance"``) or centred-and-standardised
        (``space="correlation"``) window, C-contiguous, missing cells 0.
    scale : ndarray, shape (p,)
        Per-entity scale: the window standard deviations in correlation space,
        ones in covariance space.
    mean : ndarray, shape (p,)
        Per-entity window means that were removed (zeros if
        ``assume_centered``).
    n_eff : float
        Effective sample size: ``n - 1``, or ``n`` when ``assume_centered``.
    coverage : ndarray, shape (p,)
        Fraction of the ``n`` rows at which each entity was observed.
    space : {"correlation", "covariance"}
    entities : tuple
        Entity labels, aligned with the columns of ``Z``.
    asof : Any
        The date whose window this is (``None`` for a bare matrix).
    """

    Z: NDArray[np.float64]
    scale: NDArray[np.float64]
    mean: NDArray[np.float64]
    n_eff: float
    coverage: NDArray[np.float64]
    space: str
    entities: tuple[Any, ...]
    asof: Any = None
    _gram: NDArray[np.float64] | None = field(default=None, repr=False)
    _spectrum: Spectrum | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        """Rows in the window."""
        return int(self.Z.shape[0])

    @property
    def p(self) -> int:
        """Entities in the window."""
        return int(self.Z.shape[1])

    @property
    def dual(self) -> bool:
        """Whether the min-side Gram is the ``n x n`` dual (``p > n``)."""
        return self.p > self.n

    def gram(self) -> NDArray[np.float64]:
        """The min-side Gram matrix, divided by ``n_eff`` and symmetrised.

        ``Z Z^T / n_eff`` (``n x n``) when ``p > n``, else ``Z^T Z / n_eff``
        (``p x p``). The two share their nonzero spectrum exactly.
        """
        if self._gram is None:
            from panelary.covariance._gram import min_side_gram

            self._gram = min_side_gram(self.Z, self.n_eff)
        return self._gram

    def spectrum(self, *, vectors: bool = False) -> Spectrum:
        """Eigenvalues (and optionally primal eigenvectors), cached.

        A request for vectors upgrades a cached values-only spectrum; a
        values-only request is served from a cached vector spectrum. Callers
        that need bitwise-stable values across feature sets must request the
        same ``vectors`` flag (``eigvalsh`` and ``eigh`` can differ in the last
        bit).
        """
        cached = self._spectrum
        if cached is not None and (not vectors or cached.vectors is not None):
            return cached
        from panelary.covariance._gram import eigen

        self._spectrum = eigen(self, vectors=vectors)
        return self._spectrum


@dataclass(frozen=True, slots=True, eq=False)
class CovEstimate:
    """``Sigma = diag(scale) (B diag(g) B^T + E) diag(scale)``, as of ``asof``.

    ``E`` is ``e * I`` when ``e`` is a float (isotropic) and ``diag(e)`` when
    it is an array. With ``orthonormal=True`` and an isotropic ``E`` every
    operation is exact in ``O(N r)`` (the spectral path); otherwise the
    Woodbury identity is used (``O(N r^2 + r^3)``). The ``N x N`` matrix is
    never formed unless :meth:`to_dense` is called.

    Attributes
    ----------
    asof : Any
        The date whose window produced the estimate (``None`` for a bare
        matrix).
    entities : tuple
        The as-of universe, aligned with every length-``N`` array here.
    scale, B, g, e, orthonormal
        The representation above.
    method, space : str
        Estimator name (``"qis"``, ``"lw_identity"``, ...) and
        ``"correlation"`` / ``"covariance"``.
    n_eff, q : float
        Effective sample size and ``N / n_eff``.
    shrinkage : float or None
        The linear shrinkage intensity, when the estimator has one.
    invertible : bool
        ``False`` for estimates that are singular by construction (detoned
        correlation matrices); :meth:`solve` then raises with a pointer to
        their intended use.
    info : dict
        Estimator diagnostics (MP edge, signal count, factor count, ...).
    """

    asof: Any
    entities: tuple[Any, ...]
    scale: NDArray[np.float64]
    B: NDArray[np.float64]
    g: NDArray[np.float64]
    e: float | NDArray[np.float64]
    orthonormal: bool
    method: str
    space: str
    n_eff: float
    q: float
    shrinkage: float | None = None
    invertible: bool = True
    dense: NDArray[np.float64] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # shape helpers
    # ------------------------------------------------------------------ #
    @property
    def n_entities(self) -> int:
        """``N``, the matrix dimension."""
        return int(self.scale.shape[0])

    @property
    def rank(self) -> int:
        """Columns of ``B``."""
        return int(self.B.shape[1])

    @property
    def isotropic(self) -> bool:
        """Whether ``E`` is a multiple of the identity."""
        return not isinstance(self.e, np.ndarray)

    @property
    def spectral(self) -> bool:
        """Whether the exact ``O(N r)`` spectral path applies."""
        return self.orthonormal and self.isotropic and self.dense is None

    def _e_vec(self) -> NDArray[np.float64]:
        if isinstance(self.e, np.ndarray):
            return self.e
        return np.full(self.n_entities, float(self.e))

    def inner_diag(self) -> NDArray[np.float64]:
        """``diag(B diag(g) B^T + E)`` in ``O(N r)``."""
        if self.dense is not None:
            return np.diag(self.dense).copy()
        return (self.B * self.B) @ self.g + self._e_vec()

    def diag(self) -> NDArray[np.float64]:
        """The variances, ``diag(Sigma)``."""
        return self.inner_diag() * self.scale * self.scale

    # ------------------------------------------------------------------ #
    # inner-matrix algebra (M = B G B^T + E)
    # ------------------------------------------------------------------ #
    def _refuse_singular(self) -> None:
        if not self.invertible:
            raise SingularCovarianceError(
                f"this {self.method!r} estimate is singular by construction "
                "(detoning removes the market directions). It is meant for "
                "clustering and graph construction (corr() / to_dense()), not "
                "for allocation; use method='mp_clip' or 'qis' to invert."
            )

    def _inner_solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        self._refuse_singular()
        if self.dense is not None:
            w = np.linalg.eigvalsh(self.dense)
            if w[0] <= self.n_entities * np.finfo(float).eps * max(w[-1], 0.0):
                raise SingularCovarianceError(
                    f"the {self.method!r} estimate is singular (cond = inf)."
                )
            return solve_vec(self.dense, b)
        B, g = self.B, self.g
        if self.spectral:
            e0 = float(self.e)
            lam = g + e0
            p, r = self.n_entities, self.rank
            full = r == p
            if np.any(lam <= 0.0) or (e0 <= 0.0 and not full):
                n_zero = int(np.sum(lam <= 0.0)) + (
                    p - r if (e0 <= 0.0 and not full) else 0
                )
                raise SingularCovarianceError(
                    f"the {self.method!r} estimate is singular (cond = inf): "
                    f"{n_zero} zero eigenvalue(s) out of {p}. A pseudo-inverse "
                    "would return silently wrong weights; use a shrinkage "
                    "estimator (method='qis' or 'lw_identity')."
                )
            Bt_b = B.T @ b
            coef = Bt_b / (lam[:, None] if b.ndim == 2 else lam)
            out = B @ coef
            if not full:
                out = out + (b - B @ Bt_b) / e0
            return out
        e = self._e_vec()
        if np.any(e <= 0.0):
            raise SingularCovarianceError(
                f"the {self.method!r} estimate has a zero residual variance "
                "on the Woodbury path, so it is singular (cond = inf)."
            )
        Einv_b = b / (e[:, None] if b.ndim == 2 else e)
        if self.rank == 0:
            return Einv_b
        BE = B / e[:, None]
        cap = np.eye(self.rank) + g[:, None] * (B.T @ BE)
        rhs = g[:, None] * (B.T @ Einv_b) if b.ndim == 2 else g * (B.T @ Einv_b)
        try:
            u = solve_vec(cap, rhs)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise SingularCovarianceError(
                f"the {self.method!r} estimate is singular (capacitance "
                "matrix not invertible)."
            ) from exc
        return Einv_b - BE @ u

    def _inner_logdet(self) -> float:
        if self.dense is not None:
            sign, ld = _slogdet(self.dense)
            return ld if sign > 0 else -math.inf
        if not self.invertible:
            return -math.inf
        if self.spectral:
            e0 = float(self.e)
            lam = self.g + e0
            p, r = self.n_entities, self.rank
            if np.any(lam <= 0.0) or (e0 <= 0.0 and r < p):
                return -math.inf
            out = float(np.sum(np.log(lam)))
            if r < p:
                out += (p - r) * math.log(e0)
            return out
        e = self._e_vec()
        if np.any(e <= 0.0):
            return -math.inf
        out = float(np.sum(np.log(e)))
        if self.rank:
            cap = np.eye(self.rank) + self.g[:, None] * (
                self.B.T @ (self.B / e[:, None])
            )
            sign, ld = _slogdet(cap)
            if sign <= 0:
                return -math.inf
            out += ld
        return out

    def _inner_matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.dense is not None:
            return self.dense @ x
        e = self._e_vec()
        return self.B @ (self.g * (self.B.T @ x)) + e * x

    # ------------------------------------------------------------------ #
    # public operations
    # ------------------------------------------------------------------ #
    def solve(self, b: Any) -> NDArray[np.float64]:
        """``Sigma^{-1} b`` for a ``(N,)`` or ``(N, k)`` right-hand side.

        Raises
        ------
        SingularCovarianceError
            If the estimate is singular (a sample covariance with ``N > W``,
            a detoned matrix). Never silently pseudo-inverts.
        """
        arr = _as_f64(b)
        if arr.shape[0] != self.n_entities:
            raise ValueError(
                f"solve: right-hand side has {arr.shape[0]} rows, the estimate "
                f"has {self.n_entities} entities."
            )
        s = self.scale[:, None] if arr.ndim == 2 else self.scale
        return self._inner_solve(arr / s) / s

    def inv_quad(self, x: Any) -> float:
        """``x^T Sigma^{-1} x`` (the squared Mahalanobis norm)."""
        arr = _as_f64(x)
        y = arr / self.scale
        return float(y @ self._inner_solve(y))

    def logdet(self) -> float:
        """``log det Sigma`` by the matrix determinant lemma (``-inf`` if singular)."""
        inner = self._inner_logdet()
        if not math.isfinite(inner):
            return inner
        return 2.0 * float(np.sum(np.log(self.scale))) + inner

    def cond(self, *, exact: bool = False) -> float:
        """Condition number of the **inner** matrix ``B diag(g) B^T + E``.

        In correlation space this is the correlation-matrix condition number;
        ``cond(Sigma) <= cond() * (max(scale) / min(scale))**2``.

        Parameters
        ----------
        exact : bool, default False
            ``False``: exact on the spectral path; on the Woodbury path the
            bound ``(max e + lambda_max(G+)) / (min e - lambda_max(G-))``.
            ``True``: a dense ``eigvalsh`` for ``N <= 2000``, otherwise 50
            deterministic power iterations on the matrix and its inverse.

        Returns
        -------
        float
            ``inf`` for a singular estimate.
        """
        p = self.n_entities
        if self.dense is not None or (exact and p <= _DENSE_COND_MAX):
            w = np.linalg.eigvalsh(self._inner_dense())
            tol = p * np.finfo(float).eps * max(float(w[-1]), 0.0)
            return math.inf if w[0] <= tol else float(w[-1] / w[0])
        if exact:
            return self._cond_power()
        if self.spectral:
            e0 = float(self.e)
            lam = self.g + e0
            if self.rank < p:
                lam = np.append(lam, e0)
            lo, hi = float(lam.min()), float(lam.max())
            tol = p * np.finfo(float).eps * max(hi, 0.0)
            return math.inf if lo <= tol else hi / lo
        e = self._e_vec()
        pos = np.clip(self.g, 0.0, None)
        neg = np.clip(-self.g, 0.0, None)
        K = self.B.T @ self.B
        top = float(np.linalg.eigvalsh(np.sqrt(pos)[:, None] * K * np.sqrt(pos)).max())
        bot = float(np.linalg.eigvalsh(np.sqrt(neg)[:, None] * K * np.sqrt(neg)).max())
        hi = float(e.max()) + max(top, 0.0)
        lo = float(e.min()) - max(bot, 0.0)
        return math.inf if lo <= 0.0 else hi / lo

    def _cond_power(self) -> float:
        p = self.n_entities
        x = np.ones(p) / math.sqrt(p)
        lam_max = 0.0
        for _ in range(_POWER_ITERS):
            y = self._inner_matvec(x)
            lam_max = float(np.linalg.norm(y))
            if lam_max == 0.0:
                return math.inf
            x = y / lam_max
        try:
            x = np.ones(p) / math.sqrt(p)
            mu = 0.0
            for _ in range(_POWER_ITERS):
                y = self._inner_solve(x)
                mu = float(np.linalg.norm(y))
                x = y / mu
        except SingularCovarianceError:
            return math.inf
        return lam_max * mu

    def gmv_weights(self) -> NDArray[np.float64]:
        """Global-minimum-variance weights ``Sigma^{-1} 1 / 1^T Sigma^{-1} 1``."""
        w = self.solve(np.ones(self.n_entities))
        return w / w.sum()

    def risk(self, w: Any) -> float:
        """Portfolio volatility ``sqrt(w^T Sigma w)`` in ``O(N r)``."""
        x = _as_f64(w) * self.scale
        return math.sqrt(max(float(x @ self._inner_matvec(x)), 0.0))

    def corr(self) -> CovEstimate:
        """The implied correlation matrix, as an estimate with unit diagonal.

        Same ``B``, ``g``, ``e``; ``scale = 1 / sqrt(diag(inner))``.
        """
        d = self.inner_diag()
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(d > 0.0, 1.0 / np.sqrt(np.where(d > 0.0, d, 1.0)), 0.0)
        return replace(self, scale=s)

    def subset(self, idx: Any) -> CovEstimate:
        """The marginal estimate on a subset of entities (positions or mask).

        Used to score a date on which some entities are unobserved. The
        subset of an orthonormal basis is not orthonormal, so the result is
        on the Woodbury path.
        """
        ix = np.asarray(idx)
        if ix.dtype == bool:
            ix = np.flatnonzero(ix)
        ix = ix.astype(np.intp, copy=False)
        e = self.e[ix] if isinstance(self.e, np.ndarray) else self.e
        ents = tuple(self.entities[int(i)] for i in ix)
        if self.dense is not None:
            return replace(
                self,
                entities=ents,
                scale=self.scale[ix],
                dense=self.dense[np.ix_(ix, ix)],
            )
        full = ix.size == self.n_entities and np.array_equal(
            ix, np.arange(self.n_entities)
        )
        return replace(
            self,
            entities=ents,
            scale=self.scale[ix],
            B=self.B[ix],
            e=e,
            orthonormal=self.orthonormal and full,
        )

    def _inner_dense(self) -> NDArray[np.float64]:
        if self.dense is not None:
            return self.dense
        M = (self.B * self.g) @ self.B.T
        M[np.diag_indices_from(M)] += self._e_vec()
        return 0.5 * (M + M.T)

    def to_dense(self, *, max_bytes: int = 512 * 2**20) -> NDArray[np.float64]:
        """The ``N x N`` covariance matrix; refuses above ``max_bytes``."""
        p = self.n_entities
        need = 8 * p * p
        if need > max_bytes:
            raise MemoryError(
                f"to_dense: an {p} x {p} float64 matrix needs {need / 2**20:.0f} "
                f"MiB, above max_bytes={max_bytes / 2**20:.0f} MiB. Use solve(), "
                "inv_quad(), risk() or subset() instead, or raise max_bytes."
            )
        M = self._inner_dense()
        return M * np.outer(self.scale, self.scale)
