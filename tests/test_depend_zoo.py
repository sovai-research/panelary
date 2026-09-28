"""A relation zoo: every statistic detects what it should -- and, just as
importantly, fails where it is documented to fail."""

from __future__ import annotations

import numpy as np
import pytest

import panelary.depend as dp

N = 1500


def _zoo(seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N)
    e = rng.standard_normal(N)
    t = rng.uniform(0, 2 * np.pi, N)
    a = rng.integers(0, 2, N)
    b = rng.integers(0, 2, N)
    return {
        "linear": (x, x + 0.5 * e),
        "monotone_exp": (x, np.exp(3 * x) + 0.5 * e),
        "quadratic": (x, x**2 + 0.05 * e),
        "sinusoid": (x, np.cos(4 * np.pi * x) + 0.2 * e),  # even: no monotone part
        "circle": (np.cos(t) + 0.02 * e, np.sin(t) + 0.02 * rng.standard_normal(N)),
        "xor": (
            np.column_stack([a, b]) + 0.3 * rng.uniform(size=(N, 2)),
            (a ^ b) + 0.3 * rng.uniform(size=N),
        ),
        "heteroskedastic": (x, np.abs(x) * e),
        "heavy_tailed": (rng.standard_t(3, N), None),  # filled below
        "discrete_ties": (rng.integers(0, 5, N).astype(float), None),
        "independent": (x, e),
    }


def _pvalues(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out = {}
    for m in ("xi", "spearman", "hoeffding", "dcor", "gcmi", "hsic"):
        res = dp.independence_test(
            x, y, method=m, null="asymptotic" if m != "hsic" else "auto"
        )
        out[m] = float(res["p_value"][0])
    return out


@pytest.fixture(scope="module")
def zoo() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    z = _zoo()
    rng = np.random.default_rng(1)
    tx = z["heavy_tailed"][0]
    z["heavy_tailed"] = (tx, tx + rng.standard_t(3, N))
    dx = z["discrete_ties"][0]
    z["discrete_ties"] = (dx, (dx % 2) + rng.integers(0, 2, N).astype(float))
    return z


@pytest.mark.parametrize(
    "relation",
    ["linear", "monotone_exp", "heavy_tailed", "discrete_ties"],
)
def test_monotone_relations_are_detected_by_everything(zoo, relation: str) -> None:  # type: ignore[no-untyped-def]
    p = _pvalues(*zoo[relation])
    if relation == "discrete_ties":
        # y = (x mod 2) + noise is not monotone in x. xi's closed form refuses
        # heavy ties (NaN) and falls back to permutation: p hits its 1/(B+1) floor.
        assert p["xi"] <= 1 / 200 and p["dcor"] < 1e-6 and p["hoeffding"] < 1e-3
        return
    for m, v in p.items():
        assert v < 1e-6, (relation, m, v)


@pytest.mark.parametrize("relation", ["quadratic", "sinusoid", "heteroskedastic"])
def test_nonmonotone_relations(zoo, relation: str) -> None:  # type: ignore[no-untyped-def]
    p = _pvalues(*zoo[relation])
    for m in ("xi", "hoeffding", "dcor", "hsic"):
        assert p[m] < 1e-4, (relation, m, p[m])
    # The documented weakness: a monotone (copula) statistic cannot see these.
    assert p["gcmi"] > 1e-3 and p["spearman"] > 1e-3, (relation, p)


def test_circle_xi_is_weak_the_others_are_not(zoo) -> None:  # type: ignore[no-untyped-def]
    x, y = zoo["circle"]
    # xi asks "is y a function of x": on a circle it is not (two branches).
    assert dp.xi(x, y) < 0.5 * dp.xi(np.sin(3 * x), np.sin(3 * x) ** 2 + 0.01 * y)
    p = _pvalues(x, y)
    assert p["hoeffding"] < 1e-6 and p["dcor"] < 1e-6 and p["hsic"] < 1e-6
    assert p["gcmi"] > 1e-3


def test_xor_defeats_every_pairwise_statistic(zoo) -> None:  # type: ignore[no-untyped-def]
    """y = x1 XOR x2: each input alone is independent of y, so every *pairwise*
    statistic must fail -- only a multivariate one sees it."""
    X, y = zoo["xor"]
    for col in (0, 1):
        p = _pvalues(X[:, col], y)
        assert min(p.values()) > 1e-3, (col, p)
    res = dp.dcov2(X, y)
    assert dp.dcor_pvalue(res) < 1e-10  # multivariate dcor: consistent against XOR
    assert abs(dp.gcmi(X, y)) < 0.01  # a Gaussian copula has no third-order term


def test_independent_pair_is_not_flagged(zoo) -> None:  # type: ignore[no-untyped-def]
    p = _pvalues(*zoo["independent"])
    assert min(p.values()) > 1e-3, p


def test_tail_dependence_gaussian_copula_is_not_manufactured() -> None:
    """A Gaussian copula has lambda_L = 0 in the limit: the estimator must shrink
    towards (roughly) q, not report strong tail dependence; a t-copula must not."""
    rng = np.random.default_rng(3)
    rho, n = 0.5, 200_000
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    lam = [dp.tail_dependence(z[:, 0], z[:, 1], q=q)[0] for q in (0.05, 0.01, 0.002)]
    assert lam[0] > lam[1] > lam[2]  # decays as the tail deepens
    assert lam[2] < 0.25
    w = np.sqrt(3.0 / rng.chisquare(3, n))[:, None]
    tcop = z * w  # t(3) copula: lambda_L > 0 in the limit
    lam_t = dp.tail_dependence(tcop[:, 0], tcop[:, 1], q=0.002)[0]
    assert lam_t > lam[2] + 0.1


def test_heteroskedastic_only_has_zero_linear_signal(zoo) -> None:  # type: ignore[no-untyped-def]
    x, y = zoo["heteroskedastic"]
    assert abs(dp.pearson(x, y)) < 0.12  # population value exactly 0
    assert dp.dcor(x, np.abs(y)) > 0.05
    assert dp.independence_test(x, np.abs(y), method="dcor")["p_value"][0] < 1e-6
