"""Kernel-level tests for ``panelary.covariance`` (plan section 7.1).

Primal = dual, two-pass centring, PSD-ness, the diagonal-plus-low-rank
algebra of :class:`CovEstimate` against dense linear algebra, and the
never-pseudo-invert rule (trap T14).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from panelary.covariance._estimate import METHODS, estimate
from panelary.covariance._gram import eigen, min_side_gram, window_stats
from panelary.covariance._nonlinear import lw2020_map, qis_map
from panelary.covariance._types import CovEstimate, SingularCovarianceError

ALL_METHODS = sorted(METHODS)
INVERTIBLE = [m for m in ALL_METHODS if m not in ("sample", "detone")]


def _factor_data(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 3))
    L = rng.standard_normal((3, p)) * np.array([1.5, 0.8, 0.4])[:, None]
    E = rng.standard_normal((n, p)) * rng.uniform(0.5, 2.0, p)
    return F @ L + E + rng.uniform(-1.0, 1.0, p)


def _primal_spectrum(X: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray]:
    ws = window_stats(X, space=space)
    S = ws.Z.T @ ws.Z / ws.n_eff
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return w[::-1], V[:, ::-1]


# --------------------------------------------------------------------------- #
# primal = dual
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("space", ["correlation", "covariance"])
def test_dual_spectrum_equals_primal(space: str) -> None:
    X = _factor_data(40, 120, seed=1)
    ws = window_stats(X, space=space)
    assert ws.dual
    spec = ws.spectrum(vectors=True)
    w_p, V_p = _primal_spectrum(X, space)
    k = spec.values.size
    assert k == 40
    np.testing.assert_allclose(spec.values[:39], w_p[:39], rtol=1e-12, atol=0)
    assert spec.values[39] == 0.0  # demeaning removes one direction
    # vectors: compare the projector onto the top-3 (well-separated) subspace
    V = spec.vectors
    assert V is not None
    P_dual = V[:, :3] @ V[:, :3].T
    P_prim = V_p[:, :3] @ V_p[:, :3].T
    assert np.abs(P_dual - P_prim).max() < 1e-10
    assert np.abs(V.T @ V - np.eye(V.shape[1])).max() < 1e-10


def test_min_side_gram_is_min_side_and_symmetric() -> None:
    Z = np.random.default_rng(2).standard_normal((10, 30))
    G = min_side_gram(Z, 9.0)
    assert G.shape == (10, 10)
    assert np.array_equal(G, G.T)
    G2 = min_side_gram(Z.T.copy(), 29.0)
    assert G2.shape == (10, 10)


@pytest.mark.parametrize(("fn", "rtol"), [(qis_map, 1e-12), (lw2020_map, 1e-8)])
@pytest.mark.parametrize(("n", "p"), [(60, 150), (200, 80)])
def test_nonlinear_maps_primal_equals_dual(fn, rtol: float, n: int, p: int) -> None:
    X = _factor_data(n, p, seed=3)
    ws = window_stats(X, space="correlation")
    lam_d = ws.spectrum().values
    lam_p, _ = _primal_spectrum(X, "correlation")
    lam_p = np.where(lam_p > max(n, p) * np.finfo(float).eps * lam_p[0], lam_p, 0.0)
    d0_d, d_d = fn(lam_d, ws.n_eff, p)
    d0_p, d_p = fn(lam_p[: min(n, p)], ws.n_eff, p)
    np.testing.assert_allclose(d_d, d_p, rtol=rtol)
    if not math.isnan(d0_d):
        assert abs(d0_d - d0_p) <= rtol * abs(d0_p)


@pytest.mark.parametrize("method", ["qis", "lw2020", "lw_identity", "oas", "mp_clip"])
def test_dual_estimate_equals_primal_dense(method: str) -> None:
    """The dual-path estimate equals the same estimator built from primal eigh."""
    n, p = 50, 140
    X = _factor_data(n, p, seed=4)
    est = estimate(X, method=method)
    D = est.to_dense()
    ws = window_stats(X)
    w_p, V_p = _primal_spectrum(X, "correlation")
    # rebuild through the primal eigenvectors: D_inner has the same action on them
    inner = D / np.outer(est.scale, est.scale)
    r = np.count_nonzero(ws.spectrum().values > 0)
    rq = np.einsum("ij,ik,kj->j", V_p[:, :r], inner, V_p[:, :r])
    assert est.spectral
    ref = np.zeros(r)
    ref[: est.g.size] = est.g[:r]
    ref += float(est.e)  # type: ignore[arg-type]
    np.testing.assert_allclose(rq, ref, rtol=1e-8 if method == "lw2020" else 1e-10)


# --------------------------------------------------------------------------- #
# two-pass centring
# --------------------------------------------------------------------------- #
def _exact_cov(X: np.ndarray) -> np.ndarray:
    """Covariance of the float64 input computed in exact rational arithmetic."""
    n, p = X.shape
    cols = [[Fraction(float(v)) for v in X[:, j]] for j in range(p)]
    means = [sum(c) / n for c in cols]
    out = np.empty((p, p))
    for a in range(p):
        for b in range(a, p):
            s = sum(
                (x - means[a]) * (y - means[b])
                for x, y in zip(cols[a], cols[b], strict=True)
            )
            out[a, b] = out[b, a] = float(s / (n - 1))
    return out


@pytest.mark.parametrize("offset", [0.0, 1e4, 1e8])
def test_two_pass_centring_is_exact_at_large_offsets(offset: float) -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, 4)) * 0.01 + offset
    est = estimate(X, method="sample", space="covariance")
    got = est.to_dense()
    ref = _exact_cov(X)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


# --------------------------------------------------------------------------- #
# PSD, cond, Woodbury algebra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("shape", [(60, 25), (25, 60)])
@pytest.mark.parametrize("space", ["correlation", "covariance"])
def test_every_estimator_is_psd(
    method: str, shape: tuple[int, int], space: str
) -> None:
    X = _factor_data(*shape, seed=6)
    est = estimate(X, method=method, space=space)
    D = est.to_dense()
    w = np.linalg.eigvalsh(D)
    assert w[0] >= -D.shape[0] * np.finfo(float).eps * w[-1] * 10


@pytest.mark.parametrize("method", ["qis", "lw_identity", "oas", "mp_clip", "lw2020"])
def test_cond_is_exact_on_the_spectral_path(method: str) -> None:
    X = _factor_data(30, 70, seed=7)
    est = estimate(X, method=method)
    assert est.spectral
    w = np.linalg.eigvalsh(est._inner_dense())
    assert est.cond() == pytest.approx(w[-1] / w[0], rel=1e-9)
    assert est.cond(exact=True) == pytest.approx(w[-1] / w[0], rel=1e-9)


@pytest.mark.parametrize("method", INVERTIBLE)
@pytest.mark.parametrize("shape", [(60, 25), (25, 60)])
def test_solve_inv_quad_logdet_match_dense(method: str, shape: tuple[int, int]) -> None:
    X = _factor_data(*shape, seed=8)
    est = estimate(X, method=method)
    D = est.to_dense()
    p = D.shape[0]
    rng = np.random.default_rng(9)
    b = rng.standard_normal(p)
    x = est.solve(b)
    np.testing.assert_allclose(
        D @ x, b, rtol=0, atol=1e-10 * np.abs(b).max() * est.cond()
    )
    np.testing.assert_allclose(x, np.linalg.solve(D, b[:, None])[:, 0], rtol=1e-8)
    assert est.inv_quad(b) == pytest.approx(float(b @ x), rel=1e-10)
    with np.errstate(all="ignore"):  # Accelerate raises spurious FP flags here
        sign, ld = np.linalg.slogdet(D)
    assert sign > 0
    assert est.logdet() == pytest.approx(ld, rel=1e-10, abs=1e-9)
    # the p == batch shape case: a (p, p) right-hand side is p columns
    Bm = rng.standard_normal((p, p))
    np.testing.assert_allclose(D @ est.solve(Bm), Bm, atol=1e-8 * est.cond())
    # cond(exact=False) is an upper bound off the spectral path
    w = np.linalg.eigvalsh(est._inner_dense())
    assert est.cond() >= (w[-1] / w[0]) * (1 - 1e-9)


def test_gmv_risk_corr_and_diag() -> None:
    X = _factor_data(80, 40, seed=10)
    est = estimate(X, method="qis")
    D = est.to_dense()
    w = est.gmv_weights()
    ref = np.linalg.solve(D, np.ones(40)[:, None])[:, 0]
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-9)
    assert est.risk(w) == pytest.approx(math.sqrt(w @ D @ w), rel=1e-12)
    np.testing.assert_allclose(est.diag(), np.diag(D), rtol=1e-12)
    C = est.corr().to_dense()
    np.testing.assert_allclose(np.diag(C), 1.0, rtol=1e-12)
    Dn = D / np.sqrt(np.outer(np.diag(D), np.diag(D)))
    np.testing.assert_allclose(C, Dn, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["qis", "lw_diagonal", "factor"])
def test_subset_is_the_dense_marginal(method: str) -> None:
    X = _factor_data(30, 50, seed=11)
    est = estimate(X, method=method, entities=tuple(f"e{i}" for i in range(50)))
    idx = np.array([0, 3, 7, 8, 20, 49])
    sub = est.subset(idx)
    np.testing.assert_allclose(
        sub.to_dense(), est.to_dense()[np.ix_(idx, idx)], rtol=1e-12, atol=1e-14
    )
    assert sub.entities == ("e0", "e3", "e7", "e8", "e20", "e49")
    mask = np.zeros(50, dtype=bool)
    mask[idx] = True
    np.testing.assert_array_equal(est.subset(mask).to_dense(), sub.to_dense())
    b = np.random.default_rng(12).standard_normal(idx.size)
    np.testing.assert_allclose(
        sub.solve(b), np.linalg.solve(sub.to_dense(), b[:, None])[:, 0], rtol=1e-8
    )


def test_float32_input_is_upcast_bitwise() -> None:
    X32 = _factor_data(40, 60, seed=13).astype(np.float32)
    a = estimate(X32, method="qis").to_dense()
    b = estimate(X32.astype(np.float64), method="qis").to_dense()
    assert a.dtype == np.float64
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# T14: a singular sample estimate never pseudo-inverts
# --------------------------------------------------------------------------- #
def test_T14_singular_sample_raises_and_reports_inf() -> None:
    X = _factor_data(30, 80, seed=14)
    est = estimate(X, method="sample")
    assert est.cond() == math.inf
    assert est.logdet() == -math.inf
    with pytest.raises(SingularCovarianceError, match="pseudo-inverse"):
        est.solve(np.ones(80))
    with pytest.raises(SingularCovarianceError):
        est.gmv_weights()
    # the leaky alternative the guard exists to refuse: pinv returns weights
    # without complaint -- the instrument must be able to tell them apart
    S = est.to_dense()
    w_pinv = np.linalg.pinv(S) @ np.ones(80)
    assert np.isfinite(w_pinv).all()


def test_detone_is_singular_by_construction() -> None:
    X = _factor_data(60, 30, seed=15)
    est = estimate(X, method="detone")
    assert not est.invertible
    assert est.cond() == math.inf
    with pytest.raises(SingularCovarianceError, match="clustering"):
        est.solve(np.ones(30))
    C = est.corr().to_dense()
    np.testing.assert_allclose(np.diag(C), 1.0, rtol=1e-10)


def test_to_dense_refuses_above_budget() -> None:
    est = estimate(_factor_data(20, 40, seed=16))
    with pytest.raises(MemoryError, match="max_bytes"):
        est.to_dense(max_bytes=1000)


# --------------------------------------------------------------------------- #
# window statistics and missing data
# --------------------------------------------------------------------------- #
def test_zero_after_demean_semantics() -> None:
    X = _factor_data(30, 5, seed=17)
    X[3, 1] = np.nan
    X[10:13, 4] = np.nan
    ws = window_stats(X, space="covariance")
    mean1 = np.nanmean(X[:, 1])
    assert ws.mean[1] == pytest.approx(mean1, rel=1e-14)
    assert ws.Z[3, 1] == 0.0
    assert (ws.Z[10:13, 4] == 0.0).all()
    np.testing.assert_allclose(ws.coverage, [1, 29 / 30, 1, 1, 27 / 30])
    wc = window_stats(X, space="correlation")
    # sd over the observed cells, ddof = 1
    assert wc.scale[4] == pytest.approx(np.nanstd(X[:, 4], ddof=1), rel=1e-13)


def test_correlation_space_refuses_constant_columns() -> None:
    X = _factor_data(20, 4, seed=18)
    X[:, 2] = 3.0
    with pytest.raises(ValueError, match="zero variance"):
        estimate(X)
    estimate(X, space="covariance", method="lw_identity")


def test_bad_arguments() -> None:
    X = _factor_data(20, 4, seed=19)
    with pytest.raises(ValueError, match="unknown covariance method"):
        estimate(X, method="nope")
    with pytest.raises(ValueError, match="space"):
        estimate(X, space="nope")
    with pytest.raises(ValueError, match="missing"):
        estimate(X, missing="pairwise")
    assert estimate(X, method="lw").method == "lw_identity"


def test_eigen_signs_market_up_and_max_abs_positive() -> None:
    X = _factor_data(80, 30, seed=20)
    ws = window_stats(X)
    spec = eigen(ws, vectors=True)
    V = spec.vectors
    assert V is not None
    assert V[:, 0].sum() > 0
    for j in range(1, V.shape[1]):
        col = V[:, j]
        assert col[np.argmax(np.abs(col))] > 0


def test_estimate_is_deterministic_bitwise() -> None:
    X = _factor_data(50, 120, seed=21)
    for m in ("qis", "lw_identity", "mp_clip", "factor"):
        a, b = estimate(X, method=m), estimate(X, method=m)
        assert np.array_equal(a.to_dense(), b.to_dense())


def test_cov_estimate_is_a_frozen_value() -> None:
    est = estimate(_factor_data(20, 10, seed=22))
    assert isinstance(est, CovEstimate)
    with pytest.raises(AttributeError):
        est.method = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# the shared-helper moves
# --------------------------------------------------------------------------- #
def test_psd_repair_moved_with_a_reexport() -> None:
    import panelary.depend as dp
    from panelary._internal._linalg import psd_repair

    assert dp.psd_repair is psd_repair
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    R, lam = psd_repair(bad)
    assert lam < 0 and np.linalg.eigvalsh(R).min() > -1e-10


def test_bai_ng_from_spectrum_matches_bai_ng() -> None:
    from panelary.reduce._common import prepare_matrix
    from panelary.reduce._n_factors import (
        _bai_ng_from_spectrum,
        _resolve_max_r,
        bai_ng,
    )

    for seed in range(5):
        X = _factor_data(200, 40, seed=100 + seed)
        Z = prepare_matrix(X, standardize=True)
        s = np.linalg.svd(Z, compute_uv=False)
        kmax = _resolve_max_r(*Z.shape, None)
        for crit in ("IC_p1", "IC_p2"):
            assert _bai_ng_from_spectrum(s**2, *Z.shape, kmax, crit) == bai_ng(
                X, criterion=crit
            )
