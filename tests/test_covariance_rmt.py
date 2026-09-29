"""Random-matrix properties on known populations (plan section 7.2).

White Wishart against the Marchenko--Pastur law and the Tracy--Widom edge,
the spiked model (BBP 2005 / Paul 2007 formulas), and the Ledoit--Peche
oracle ordering QIS < LW-identity < sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from panelary.covariance._estimate import estimate
from panelary.covariance._gram import window_stats
from panelary.covariance._rmt import mp_edges, mp_fit, tw_edge


def _mp_cdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko--Pastur CDF (q <= 1), by numerically integrating the density."""
    lo, hi = mp_edges(sigma2, q)
    grid = np.linspace(lo, hi, 200_001)
    dens = np.sqrt(np.clip((hi - grid) * (grid - lo), 0.0, None)) / (
        2.0 * np.pi * sigma2 * q * grid
    )
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))]
    )
    cdf /= cdf[-1]
    return np.interp(x, grid, cdf, left=0.0, right=1.0)


def test_white_wishart_spectrum_follows_marchenko_pastur() -> None:
    X = np.random.default_rng(0).standard_normal((1000, 400))
    ws = window_stats(X, space="covariance")
    lam = np.sort(ws.spectrum().values)
    q = 400 / ws.n_eff
    emp_hi = np.arange(1, lam.size + 1) / lam.size
    emp_lo = np.arange(0, lam.size) / lam.size
    F = _mp_cdf(lam, q)
    ks = max(np.abs(emp_hi - F).max(), np.abs(emp_lo - F).max())
    assert ks < 0.03


def test_tracy_widom_edge_has_its_nominal_size() -> None:
    """Over 200 white-noise windows, lambda_max exceeds the TW-95% edge ~5%."""
    reps, hits = 200, 0
    for r in range(reps):
        X = np.random.default_rng(10_000 + r).standard_normal((1000, 400))
        ws = window_stats(X, space="covariance")
        hits += int(ws.spectrum().values[0] > tw_edge(1.0, ws.n_eff, 400))
    assert abs(hits / reps - 0.05) <= 0.03 + 1e-12


def _spiked(n: int, p: int, spikes: tuple[float, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sd = np.ones(p)
    sd[: len(spikes)] = np.sqrt(spikes)
    return rng.standard_normal((n, p)) * sd


def test_spiked_model_bbp_formulas_and_signal_count() -> None:
    n, p, spikes = 800, 200, (10.0, 3.0, 1.4)  # 1.4 is below the BBP threshold 1.5
    reps = 20
    tops = np.zeros((reps, 2))
    overlaps = np.zeros((reps, 2))
    counts = []
    for r in range(reps):
        ws = window_stats(_spiked(n, p, spikes, seed=r), space="covariance")
        spec = ws.spectrum(vectors=True)
        V = spec.vectors
        assert V is not None
        tops[r] = spec.values[:2]
        overlaps[r] = [V[0, 0] ** 2, V[1, 1] ** 2]
        counts.append(int(mp_fit(spec.values, p, ws.n_eff)["n_signal"]))
    q = p / (n - 1)
    for j, ell in enumerate(spikes[:2]):
        expected = ell * (1.0 + q / (ell - 1.0))
        assert tops[:, j].mean() == pytest.approx(expected, rel=0.03)
        overlap = (1.0 - q / (ell - 1.0) ** 2) / (1.0 + q / (ell - 1.0))
        assert abs(overlaps[:, j].mean() - overlap) < 0.05
    assert np.mean(np.array(counts) == 2) >= 0.9


def test_mp_fit_converges_and_finds_no_signal_in_noise() -> None:
    X = np.random.default_rng(3).standard_normal((500, 250))
    ws = window_stats(X, space="correlation")
    fit = mp_fit(ws.spectrum().values, 250, ws.n_eff)
    assert fit["n_signal"] == 0
    assert fit["n_iter"] <= 10
    assert fit["sigma2"] == pytest.approx(1.0, abs=0.02)
    bare = mp_fit(ws.spectrum().values, 250, ws.n_eff, edge="bare")
    assert bare["edge"] < fit["edge"]


@pytest.mark.parametrize(("n", "p"), [(400, 200), (200, 400)])
def test_oracle_tracking_qis_beats_lw_beats_sample(n: int, p: int) -> None:
    """Mean relative error of d_i against the oracle u_i^T Sigma u_i.

    Population: Ledoit & Wolf's clustered spectrum (20% of eigenvalues at 1,
    40% at 3, 40% at 10), where the shape of the spectrum -- not just its
    level -- has to be recovered. (On a smooth log-uniform [1, 10] spectrum
    at q = 2 the linear estimator was measured slightly ahead: 0.085 vs
    0.111; no estimator dominates everywhere, which is what the walk-forward
    harness of plan section 8.3 is for.)
    """
    errs: dict[str, list[float]] = {"qis": [], "lw_identity": [], "sample": []}
    for r in range(10):
        rng = np.random.default_rng(100 + r)
        pop = np.repeat([1.0, 3.0, 10.0], [p // 5, 2 * p // 5, p - 3 * p // 5])
        X = rng.standard_normal((n, p)) * np.sqrt(pop)
        ws = window_stats(X, space="covariance")
        S = ws.Z.T @ ws.Z / ws.n_eff
        _, U = np.linalg.eigh(0.5 * (S + S.T))  # all p directions, nulls too
        oracle = np.einsum("ij,i,ij->j", U, pop, U)
        for m in errs:
            D = estimate(X, method=m, space="covariance").to_dense()
            d = np.einsum("ij,ik,kj->j", U, D, U)
            errs[m].append(float(np.mean(np.abs(d - oracle) / oracle)))
    mean = {m: float(np.mean(v)) for m, v in errs.items()}
    assert mean["qis"] < mean["lw_identity"] < mean["sample"]


def test_mp_clip_preserves_trace_and_unit_diagonal() -> None:
    # rotate the spikes so they are correlation structure, not single-variable
    # variance (a standardised single-coordinate spike is no spike at all)
    Q, _ = np.linalg.qr(np.random.default_rng(55).standard_normal((120, 120)))
    X = (_spiked(300, 120, (8.0, 3.0), seed=5) @ Q) * np.linspace(0.5, 2.0, 120)
    cov = estimate(X, method="mp_clip", space="covariance")
    S = np.cov(X, rowvar=False)
    assert np.trace(cov.to_dense()) == pytest.approx(np.trace(S), rel=1e-12)
    corr = estimate(X, method="mp_clip", space="correlation")
    assert corr.info["n_signal"] == 2
    C = corr.corr().to_dense()
    np.testing.assert_allclose(np.diag(C), 1.0, rtol=1e-12)
    # the estimate's own variances are the window variances
    np.testing.assert_allclose(corr.diag(), np.diag(S), rtol=1e-12)


def test_targeted_shrinkage_endpoints() -> None:
    X = _spiked(200, 60, (6.0, 2.5), seed=6)
    full = estimate(X, method="mp_targeted", alpha_shrink=1.0).to_dense()
    samp = estimate(X, method="sample").to_dense()
    np.testing.assert_allclose(full, samp, rtol=1e-10, atol=1e-12)
    zero = estimate(X, method="mp_targeted", alpha_shrink=0.0)
    k = int(zero.info["n_signal"])
    fac = estimate(X, method="factor", n_factors=k)
    np.testing.assert_allclose(zero.to_dense(), fac.to_dense(), rtol=1e-10)


def test_detone_removes_the_market_mode() -> None:
    rng = np.random.default_rng(7)
    f = rng.standard_normal((400, 1))
    X = f @ np.full((1, 50), 0.8) + rng.standard_normal((400, 50))
    est = estimate(X, method="detone")
    C = est._inner_dense()
    w = np.linalg.eigvalsh(C)
    assert abs(w[0]) < 1e-10 * w[-1]  # singular in the market direction
    v1 = estimate(X, method="sample").B[:, 0]
    assert abs(float(v1 @ (C @ v1))) < 1e-10 * w[-1]
    assert np.linalg.eigvalsh(est.to_dense())[0] > -1e-12


@pytest.mark.parametrize("rule", [1, 3, "eigenvalue_ratio", "bai_ng", "mp"])
def test_factor_count_rules_are_window_local(rule: int | str) -> None:
    # Equal communalities: the MP count assumes one noise level, which random
    # loadings would break in correlation space (it then over-counts).
    rng = np.random.default_rng(8)
    F = rng.standard_normal((300, 3)) * np.array([3.0, 2.0, 1.5])
    L = rng.choice([-1.0, 1.0], size=(3, 80))
    X = F @ L + rng.standard_normal((300, 80)) * 2.0
    est = estimate(X, method="factor", n_factors=rule)
    k = int(est.info["n_factors"])
    assert k == (rule if isinstance(rule, int) else 3)
    assert est.rank == k
