"""Agreement of the numpy primitives with reference implementations (plan section 9, item 5).

* ``randomized_svd`` agrees with ``np.linalg.svd`` to ``1e-6`` on the leading
  ``k`` singular values and subspaces.
* ``RandomizedPCA`` / ``reduce.PanelPCA(backend="numpy")`` / ``PanelSVD`` agree
  with scikit-learn to ``1e-6`` after the shared deterministic sign fix
  (skipped where scikit-learn is absent).
* ``pivoted_qr`` picks the same pivots as ``scipy.linalg.qr(pivoting=True)``
  (skipped without scipy); ``fwht`` / ``srht`` equal the explicit Hadamard
  construction.
* ``hosvd`` equals the exact per-mode SVD; ``partial_tucker`` never does worse
  than its HOSVD start and agrees with ``tensorly`` where installed (test-only
  dependency, skipped otherwise).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from panelary.reduce._base import _sign_of_max_abs
from panelary.shape import (
    RandomizedPCA,
    fwht,
    hosvd,
    interpolative,
    mode_dot,
    partial_tucker,
    pivoted_qr,
    randomized_svd,
    sparse_rp_matrix,
    srht,
    tucker_to_tensor,
    unfold,
)
from panelary.shape._project import _srht_draw, hadamard_rows

TOL = 1e-6


def _decaying(m: int, n: int, seed: int, decay: float = 0.7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = min(m, n)
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]
    V = np.linalg.qr(rng.standard_normal((n, r)))[0]
    return (U * decay ** np.arange(r)) @ V.T


def _frame(X: np.ndarray, n_ent: int = 8) -> pl.DataFrame:
    n = X.shape[0]
    return pl.DataFrame(
        {
            "id": np.repeat([f"e{i}" for i in range(n_ent)], n // n_ent),
            "t": np.tile(np.arange(n // n_ent), n_ent),
            **{f"f{j}": X[:, j] for j in range(X.shape[1])},
        }
    )


def _subspace_gap(A: np.ndarray, B: np.ndarray) -> float:
    """``|| |A^T B| - I ||_max`` for orthonormal column blocks (sign-blind)."""
    return float(np.max(np.abs(np.abs(A.T @ B) - np.eye(A.shape[1]))))


# --------------------------------------------------------------------------- #
# randomized_svd vs np.linalg.svd
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("m", "n", "k"), [(300, 40, 5), (40, 300, 5), (120, 120, 10), (50, 20, 20)]
)
@pytest.mark.parametrize("n_iter", ["auto", 2])
def test_randomized_svd_matches_exact_svd(
    m: int, n: int, k: int, n_iter: int | str
) -> None:
    A = _decaying(m, n, seed=m + n)
    U, s, Vt = randomized_svd(A, k, n_iter=n_iter, seed=3)
    Ue, se, Vte = np.linalg.svd(A, full_matrices=False)
    assert np.max(np.abs(s - se[:k]) / se[0]) < TOL
    assert _subspace_gap(U, Ue[:, :k]) < TOL
    assert _subspace_gap(Vt.T, Vte[:k].T) < TOL
    assert U.shape == (m, k) and Vt.shape == (k, n)


def test_randomized_svd_is_deterministic_and_refuses_bad_input() -> None:
    A = _decaying(60, 30, seed=1)
    a = randomized_svd(A, 4, seed=9)
    b = randomized_svd(A, 4, seed=9)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError, match="NaN"):
        bad = A.copy()
        bad[3, 4] = np.nan
        randomized_svd(bad, 3)
    with pytest.raises(ValueError, match="exceeds"):
        randomized_svd(A, 31)
    with pytest.raises(TypeError, match="explicit int"):
        randomized_svd(A, 3, seed=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# PCA vs scikit-learn
# --------------------------------------------------------------------------- #
def _fixed_panel(
    n: int = 400, d: int = 12, seed: int = 0
) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3)) @ rng.standard_normal((3, d)) * np.arange(1, d + 1)
    X = X + 0.3 * rng.standard_normal((n, d)) + 5.0
    return _frame(X), X


def test_randomized_pca_matches_sklearn_pca() -> None:
    skd = pytest.importorskip("sklearn.decomposition")
    skp = pytest.importorskip("sklearn.preprocessing")
    df, X = _fixed_panel()
    ours = RandomizedPCA(n_components=4, entity="id", time="t").fit(df)
    scaler = skp.StandardScaler().fit(X)
    ref = skd.PCA(n_components=4, svd_solver="full").fit(scaler.transform(X))
    sign = np.array([_sign_of_max_abs(ref.components_[i]) for i in range(4)])
    assert ours.components_ is not None
    assert np.max(np.abs(ours.components_ - ref.components_ * sign[:, None])) < TOL
    assert (
        np.max(np.abs(ours.explained_variance_ratio_ - ref.explained_variance_ratio_))
        < TOL
    )
    Z = ours.transform(df).collect().select(ours.component_names_).to_numpy()
    Zr = ref.transform(scaler.transform(X)) * sign
    assert np.max(np.abs(Z - Zr)) < TOL * max(1.0, float(np.abs(Zr).max()))


@pytest.mark.parametrize("cls_name", ["PanelPCA", "PanelSVD"])
@pytest.mark.parametrize("n_components", [2, None])
def test_reduce_numpy_backend_matches_sklearn_backend(
    cls_name: str, n_components: int | None
) -> None:
    pytest.importorskip("sklearn")
    import panelary.reduce as reduce

    cls = getattr(reduce, cls_name)
    df, _ = _fixed_panel(d=10)
    a = cls(n_components=n_components).fit(df, entity="id", time="t")
    b = cls(n_components=n_components, backend="numpy").fit(df, entity="id", time="t")
    assert a.n_components_ == b.n_components_
    assert np.max(np.abs(a.components_ - b.components_)) < TOL
    assert (
        np.max(np.abs(a.explained_variance_ratio_ - b.explained_variance_ratio_)) < TOL
    )
    za = a.transform(df, entity="id", time="t").collect().drop("id", "t").to_numpy()
    zb = b.transform(df, entity="id", time="t").collect().drop("id", "t").to_numpy()
    assert np.max(np.abs(za - zb)) < TOL * max(1.0, float(np.abs(za).max()))
    assert type(b.scaler_).__name__ == "_NumpyScaler"


def test_reduce_backend_auto_routing_and_validation() -> None:
    from panelary.reduce import PanelPCA, PanelRandomProjection

    assert PanelPCA(n_components=3, backend="auto")._numpy_path(30)
    assert not PanelPCA(n_components=4, backend="auto")._numpy_path(30)
    assert not PanelPCA(backend="auto")._numpy_path(30)  # count unknown until fit
    assert not PanelPCA(n_components=1)._numpy_path(1000)  # default stays sklearn
    with pytest.raises(ValueError, match="backend"):
        PanelPCA(backend="torch")
    with pytest.raises(TypeError, match="explicit int"):
        PanelPCA(backend="numpy", random_state=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        PanelRandomProjection(method="achlioptas")


@pytest.mark.parametrize("method", ["sparse", "srht"])
def test_panel_random_projection_numpy_methods(method: str) -> None:
    from panelary.reduce import PanelRandomProjection

    df, X = _fixed_panel(d=10)
    r = PanelRandomProjection(n_components=4, method=method, random_state=5).fit(
        df, entity="id", time="t"
    )
    Z = r.transform(df, entity="id", time="t").collect().drop("id", "t").to_numpy()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    assert np.allclose(Z, Xs @ r.components_.T, rtol=1e-10, atol=1e-10)
    # Data-independent: the projection depends on the width and the seed only.
    df2, _ = _fixed_panel(d=10, seed=99)
    r2 = PanelRandomProjection(n_components=4, method=method, random_state=5).fit(
        df2, entity="id", time="t"
    )
    assert np.array_equal(np.abs(r.components_), np.abs(r2.components_))


# --------------------------------------------------------------------------- #
# Pivoted QR / ID vs scipy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("m", "n", "k"), [(80, 30, 10), (30, 80, 12), (50, 50, 50)])
def test_pivoted_qr_matches_scipy_pivots(m: int, n: int, k: int) -> None:
    sl = pytest.importorskip("scipy.linalg")
    A = np.random.default_rng(m * n).standard_normal((m, n))
    Q, R, piv = pivoted_qr(A, k)
    _, Rs, ps = sl.qr(A, pivoting=True, mode="economic")
    assert np.array_equal(piv[:k], ps[:k])
    assert np.allclose(np.abs(np.diag(R)), np.abs(np.diag(Rs))[:k], rtol=1e-10)
    Ap = A[:, piv]
    # The chosen columns are reproduced exactly; the rest are projected onto
    # span(Q) (A[:, piv] = Q R + residual, residual orthogonal to Q).
    assert np.allclose(Q @ R[:, :k], Ap[:, :k], atol=1e-10)
    assert np.allclose(Q @ R[:, k:], Q @ (Q.T @ Ap[:, k:]), atol=1e-10)
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-12)


def test_interpolative_decomposition_bound_is_near_the_svd_bound() -> None:
    A = _decaying(100, 60, seed=4, decay=0.5)
    k = 8
    cols, Z = interpolative(A, k)
    err = np.linalg.norm(A - A[:, cols] @ Z, 2)
    s = np.linalg.svd(A, compute_uv=False)
    # ID with greedy pivoting: ||A - A[:, J] Z||_2 <= sqrt(1 + k (n - k)) 2^k s_{k+1}
    # in the worst case; on a generic matrix it is within a small factor.
    assert err <= 50 * s[k]
    assert np.array_equal(Z[:, cols], np.eye(k))


# --------------------------------------------------------------------------- #
# Hadamard constructions
# --------------------------------------------------------------------------- #
def test_fwht_and_srht_equal_the_explicit_hadamard_construction() -> None:
    rng = np.random.default_rng(5)
    H = hadamard_rows(np.arange(16), 16)
    assert np.array_equal(H @ H.T, 16 * np.eye(16))
    x = rng.standard_normal((4, 16))
    assert np.allclose(fwht(x), x @ H.T, atol=1e-12)
    X = rng.standard_normal((7, 11))
    signs, sample, _D = _srht_draw(11, 5, 3)
    explicit = X @ (hadamard_rows(sample, 11) * signs).T / np.sqrt(5)
    assert np.allclose(srht(X, 5, seed=3), explicit, atol=1e-12)
    with pytest.raises(ValueError, match="power of two"):
        fwht(np.ones(12))


def test_sparse_rp_preserves_squared_norms_in_expectation() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal(400)
    ratios = [
        float(np.sum((x @ sparse_rp_matrix(400, 64, seed=s)) ** 2) / np.sum(x * x))
        for s in range(60)
    ]
    assert abs(np.mean(ratios) - 1.0) < 0.1
    R = sparse_rp_matrix(400, 64, seed=0)
    assert 0.02 < np.count_nonzero(R) / R.size < 0.08  # density 1/sqrt(400) = 0.05


# --------------------------------------------------------------------------- #
# Tucker
# --------------------------------------------------------------------------- #
def _tensor(seed: int, noise: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((6, 4, 3))
    X = mode_dot(
        mode_dot(G, np.linalg.qr(rng.standard_normal((20, 4)))[0], 1),
        np.linalg.qr(rng.standard_normal((7, 3)))[0],
        2,
    )
    return X + noise * rng.standard_normal(X.shape)


def test_hosvd_equals_the_exact_per_mode_svd() -> None:
    X = _tensor(7)
    h = hosvd(X, {"time": 4, "feature": 3})
    for m, r in ((1, 4), (2, 3)):
        Ue = np.linalg.svd(unfold(X, m), full_matrices=False)[0][:, :r]
        assert _subspace_gap(h.factors[m], Ue) < TOL
    assert h.n_iter == 0
    direct = np.linalg.norm(X - tucker_to_tensor(h.core, h.factors)) / np.linalg.norm(X)
    assert abs(h.reconstruction_error - direct) < 1e-12


def test_partial_tucker_improves_on_hosvd_and_keeps_entities() -> None:
    X = _tensor(8, noise=0.3)
    h = hosvd(X, {"time": 3, "feature": 2})
    p = partial_tucker(X, {"time": 3, "feature": 2}, n_iter=20)
    assert p.reconstruction_error <= h.reconstruction_error + 1e-12
    assert p.core.shape == (6, 3, 2)
    assert set(p.factors) == {1, 2}
    for U in p.factors.values():
        assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)


def test_partial_tucker_agrees_with_tensorly() -> None:
    tl_dec = pytest.importorskip("tensorly.decomposition")
    X = _tensor(9, noise=0.2)
    ours = partial_tucker(X, {"time": 4, "feature": 3}, n_iter=50, tol=1e-12)
    res = tl_dec.partial_tucker(
        X, rank=[4, 3], modes=[1, 2], n_iter_max=200, tol=1e-12, init="svd"
    )
    (core, factors), _ = res if isinstance(res[0], tuple) else (res, None)
    theirs = mode_dot(
        mode_dot(np.asarray(core), np.asarray(factors[0]), 1), np.asarray(factors[1]), 2
    )
    rec = tucker_to_tensor(ours.core, ours.factors)
    assert np.linalg.norm(rec - theirs) / np.linalg.norm(X) < 1e-4
