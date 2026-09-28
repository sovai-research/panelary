"""Coefficients of :mod:`panelary.depend._coef` against oracles and definitions."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from panelary.depend import (
    exceedance_corr,
    hoeffding_d,
    hoeffding_pvalue,
    kendall,
    kendall_pvalue,
    pearson,
    spearman,
    tail_dependence,
    tail_dependence_matrix,
    tail_pvalue,
    xi,
    xi_matrix,
    xi_null_sd,
    xi_pvalue,
)
from panelary.depend._coef import _xi_rows


def _xi_ref_no_x_ties(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee's formula verbatim (ties in y allowed, none in x)."""
    n = x.size
    order = np.argsort(x, kind="stable")
    ys = y[order]
    r = np.array([(ys <= v).sum() for v in ys], dtype=float)
    ell = np.array([(ys >= v).sum() for v in ys], dtype=float)
    return 1.0 - n * np.abs(np.diff(r)).sum() / (2.0 * (ell * (n - ell)).sum())


def test_xi_formula_and_asymmetry() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = np.round(x**2 + 0.2 * rng.standard_normal(300), 1)  # ties in y
    assert xi(x, y) == pytest.approx(_xi_ref_no_x_ties(x, y), abs=1e-12)
    # Directional: y is a function of x, not the other way round.
    assert xi(x, x**2) > 0.9
    assert xi(x**2, x) < 0.4  # two branches: x is not a function of x**2


def test_xi_x_ties_are_the_exact_expectation() -> None:
    """tie_break='average' equals the mean of xi over every ordering of tied x."""
    x = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 5], dtype=float)
    y = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=float)
    groups = [np.flatnonzero(x == v) for v in np.unique(x)]
    vals = []
    for perms in itertools.product(*[itertools.permutations(g) for g in groups]):
        order = np.concatenate([np.array(p) for p in perms])
        n = x.size
        ys = y[order]
        r = np.array([(y <= v).sum() for v in ys], dtype=float)
        ell = np.array([(y >= v).sum() for v in ys], dtype=float)
        vals.append(
            1.0 - n * np.abs(np.diff(r)).sum() / (2.0 * (ell * (n - ell)).sum())
        )
    assert xi(x, y) == pytest.approx(float(np.mean(vals)), abs=1e-12)
    # Deterministic: same call, same bits.
    assert xi(x, y) == xi(x, y)
    # tie_break="random" needs a seed and is one of the enumerated values.
    with pytest.raises(ValueError, match="seed"):
        xi(x, y, tie_break="random")
    assert any(abs(xi(x, y, tie_break="random", seed=5) - v) < 1e-12 for v in vals)


def test_xi_null_sd_and_pvalue() -> None:
    assert xi_null_sd(1000) == pytest.approx(math.sqrt(0.4 / 1000))
    heavy = np.repeat(np.arange(10.0), 20)
    assert math.isnan(xi_null_sd(200, ties=heavy))
    assert math.isnan(xi_pvalue(0.1, 200, ties=heavy))
    assert xi_pvalue(0.0, 500) == pytest.approx(0.5)


def test_xi_matrix_matches_pairwise_and_rows() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 5))
    X[:, 1] = X[:, 0] ** 2 + 0.1 * rng.standard_normal(400)
    X[:, 4] = np.round(X[:, 4], 1)
    M = xi_matrix(X)
    for i in range(5):
        for j in range(5):
            assert M[i, j] == pytest.approx(xi(X[:, i], X[:, j]), abs=1e-12)
    assert M[0, 1] > 0.8 > M[1, 0]
    np.testing.assert_allclose(
        _xi_rows(X.T[:3], X.T[1:4]), [xi(X[:, i], X[:, i + 1]) for i in range(3)]
    )


def test_pearson_spearman_kendall_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    x = rng.integers(0, 20, 300).astype(float)
    y = x + rng.integers(0, 15, 300)
    assert pearson(x, y) == pytest.approx(stats.pearsonr(x, y)[0], abs=1e-12)
    assert spearman(x, y) == pytest.approx(stats.spearmanr(x, y)[0], abs=1e-12)
    kt = stats.kendalltau(x, y, method="asymptotic")
    assert kendall(x, y) == pytest.approx(kt.statistic, abs=1e-12)
    tau, p = kendall_pvalue(x, y)
    assert tau == pytest.approx(kt.statistic, abs=1e-12)
    assert p == pytest.approx(kt.pvalue, rel=1e-6)


def _hoeffding_brute(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    R = np.array([(x < v).sum() + 0.5 * ((x == v).sum() + 1) for v in x])
    S = np.array([(y < v).sum() + 0.5 * ((y == v).sum() + 1) for v in y])
    Q = np.empty(n)
    for i in range(n):
        lt_x, eq_x = x < x[i], x == x[i]
        lt_y, eq_y = y < y[i], y == y[i]
        Q[i] = (
            1.0
            + (lt_x & lt_y).sum()
            + 0.25 * ((eq_x & eq_y).sum() - 1)
            + 0.5 * (eq_x & lt_y).sum()
            + 0.5 * (lt_x & eq_y).sum()
        )
    d1 = ((Q - 1) * (Q - 2)).sum()
    d2 = ((R - 1) * (R - 2) * (S - 1) * (S - 2)).sum()
    d3 = ((R - 2) * (S - 2) * (Q - 1)).sum()
    return (
        30.0
        * ((n - 2) * (n - 3) * d1 + d2 - 2 * (n - 2) * d3)
        / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    )


def test_hoeffding_bruteforce_with_ties() -> None:
    rng = np.random.default_rng(4)
    for n in (5, 30, 211):
        x = rng.integers(0, 8, n).astype(float)
        y = (x + rng.integers(0, 4, n)).astype(float)
        assert hoeffding_d(x, y) == pytest.approx(_hoeffding_brute(x, y), abs=1e-12)
    x = rng.standard_normal(200)
    assert hoeffding_d(x, x) == pytest.approx(1.0, abs=1e-12)
    assert math.isnan(hoeffding_d(x[:4], x[:4]))


def test_hoeffding_catches_what_xi_misses_and_pvalue() -> None:
    rng = np.random.default_rng(5)
    t = rng.uniform(0, 2 * np.pi, 600)
    x, y = np.cos(t), np.sin(t)  # circle: not a function either way
    assert hoeffding_pvalue(hoeffding_d(x, y), 600) < 1e-6
    assert xi(x, y) < 0.35  # xi is weak on non-functional dependence
    z = rng.standard_normal(600)
    p = hoeffding_pvalue(hoeffding_d(z, rng.standard_normal(600)), 600)
    assert 0.0 <= p <= 1.0


def test_hoeffding_limit_law_quantiles() -> None:
    """The tabulated survival function against a Monte Carlo of
    W = sum Z_jk^2 / (pi^4 j^2 k^2) (truncated at 60 x 60, remainder mean added)."""
    from panelary.depend._coef import _hoeffding_chernoff, _hoeffding_sf

    rng = np.random.default_rng(7)
    j = np.arange(1, 61, dtype=float)
    lam = (1.0 / (np.pi**4 * np.outer(j, j) ** 2)).ravel()
    rem = 1.0 / 36.0 - lam.sum()
    w = (
        np.concatenate(
            [(rng.standard_normal((20_000, lam.size)) ** 2) @ lam for _ in range(5)]
        )
        + rem
    )
    for level in (0.10, 0.05, 0.01, 0.001):
        qv = float(np.quantile(w, 1.0 - level))
        assert _hoeffding_sf(qv) == pytest.approx(level, rel=0.15, abs=2e-4)
    # Far tail: no floor at the integration noise, and never below the truth's bound.
    assert _hoeffding_sf(1.0) < 1e-12
    assert _hoeffding_sf(1.0) <= _hoeffding_chernoff(1.0)


def test_tail_dependence_and_exceedance() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal(4000)
    lo, hi = tail_dependence(x, x)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    lo_i, hi_i = tail_dependence(x, rng.standard_normal(4000), q=0.05)
    assert lo_i < 0.15 and hi_i < 0.15  # ~q under independence
    M = tail_dependence_matrix(np.column_stack([x, x, -x]), q=0.05)
    assert M[0, 1] == pytest.approx(1.0) and M[0, 2] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        tail_dependence(x, x, q=0.6)
    y = x + 0.5 * rng.standard_normal(4000)
    elo, ehi = exceedance_corr(x, y, q=0.2)
    assert np.isfinite(elo) and np.isfinite(ehi)
    p = tail_pvalue(tail_dependence(x, y)[0], x, y, q=0.05, side="lower")
    assert p < 1e-10
