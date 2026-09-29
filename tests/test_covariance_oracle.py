"""Oracle parity for the covariance estimators (plan section 7.3).

* LW identity and OAS against ``sklearn.covariance`` (behind importorskip).
* Diagonal, constant-correlation, single-index and QIS against fixtures
  produced once by the authors' reference code (pald22/covShrinkage, BSD-2),
  committed in ``tests/data/covariance_covshrinkage_fixtures.json`` with
  provenance. The reference code is never imported.
* LW 2020 against a dense, primal port of the published formulas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from panelary.covariance._estimate import estimate
from panelary.covariance._gram import window_stats
from panelary.covariance._linear import oas_intensity

_FIXTURES = Path(__file__).parent / "data" / "covariance_covshrinkage_fixtures.json"


def _data(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 2))
    L = rng.standard_normal((2, p))
    return F @ L + rng.standard_normal((n, p)) * rng.uniform(0.5, 2.0, p) + 0.3


# --------------------------------------------------------------------------- #
# scikit-learn
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("n", "p"), [(80, 30), (30, 80), (252, 400)])
def test_lw_identity_matches_sklearn(n: int, p: int) -> None:
    skc = pytest.importorskip("sklearn.covariance")
    X = _data(n, p, seed=n + p)
    ref = skc.LedoitWolf(assume_centered=False).fit(X)
    Xc = X - X.mean(axis=0)
    est = estimate(Xc, method="lw_identity", space="covariance", assume_centered=True)
    assert est.shrinkage == pytest.approx(ref.shrinkage_, rel=1e-12)
    np.testing.assert_allclose(est.to_dense(), ref.covariance_, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n", "p"), [(80, 30), (30, 80), (252, 400)])
def test_oas_matches_sklearn_formula_and_paper_bound(n: int, p: int) -> None:
    skc = pytest.importorskip("sklearn.covariance")
    X = _data(n, p, seed=2 * n + p)
    ref = skc.OAS(assume_centered=False).fit(X)
    Xc = X - X.mean(axis=0)
    ws = window_stats(Xc, space="covariance", assume_centered=True)
    # sklearn's variant (no 2/p terms), recomputed from our dual traces
    G = ws.gram()
    tr = float(np.einsum("ij,ij->", ws.Z, ws.Z)) / ws.n_eff
    f2 = float(np.einsum("ij,ij->", G, G))
    mu = tr / p
    alpha = f2 / p**2
    num = alpha + mu**2
    den = (ws.n_eff + 1.0) * (alpha - mu**2 / p)
    sk_rho = 1.0 if den == 0 else min(num / den, 1.0)
    assert sk_rho == pytest.approx(ref.shrinkage_, rel=1e-12)
    # the paper's eq. 23 (what we ship) differs by at most 4 rho / p
    rho, _ = oas_intensity(ws)
    assert abs(rho - ref.shrinkage_) <= 4.0 * max(rho, ref.shrinkage_) / p


# --------------------------------------------------------------------------- #
# covShrinkage fixtures
# --------------------------------------------------------------------------- #
def _fixture_input(n: int, p: int, seed: int) -> np.ndarray:
    """The generator the fixtures were produced from (keep in sync with them)."""
    rng = np.random.default_rng(seed)
    k = 3
    F = rng.standard_normal((n, k))
    L = rng.standard_normal((k, p)) * np.array([2.0, 1.0, 0.5])[:, None]
    E = rng.standard_normal((n, p)) * rng.uniform(0.5, 1.5, p)
    return F @ L + E + 0.1


def _cases() -> list[tuple[str, str]]:
    fx = json.loads(_FIXTURES.read_text())
    return [
        (key, method)
        for key in fx["cases"]
        for method in (
            "qis",
            "lw_diagonal",
            "lw_constant_correlation",
            "lw_single_index",
        )
    ]


def test_fixture_file_records_its_provenance() -> None:
    fx = json.loads(_FIXTURES.read_text())
    prov = fx["provenance"]
    assert "pald22/covShrinkage" in prov["source"]
    assert "BSD-2" in prov["files"]
    for case in fx["cases"].values():
        assert case["min_rel_gap"] > 1e-3  # distinct-spectrum inputs only


@pytest.mark.parametrize(("key", "method"), _cases())
def test_matches_covshrinkage_reference(key: str, method: str) -> None:
    case = json.loads(_FIXTURES.read_text())["cases"][key]
    n, p, seed = case["n"], case["p"], case["seed"]
    Y = _fixture_input(n, p, seed)
    D = estimate(Y, method=method, space="covariance").to_dense()
    ref = case[method]
    if "rowspace" in ref:
        Yc = Y - Y.mean(axis=0)
        got = D @ Yc.T[:, :5]
        want = np.array(ref["rowspace"])
        assert np.abs(got - want).max() <= 1e-10 * np.abs(want).max()
        return
    probes = np.random.default_rng(1000 + seed).standard_normal((p, 3))
    want_d = np.array(ref["diag"])
    want_p = np.array(ref["probe"])
    assert np.abs(np.diag(D) - want_d).max() <= 1e-10 * np.abs(want_d).max()
    assert np.abs(D @ probes - want_p).max() <= 1e-10 * np.abs(want_p).max()
    assert np.linalg.norm(D) == pytest.approx(ref["fro"], rel=1e-10)


def test_qis_trace_is_preserved_when_p_exceeds_n() -> None:
    """Where the reference's null block is unusable (see the fixture note)."""
    Y = _fixture_input(20, 60, 12)
    est = estimate(Y, method="qis", space="covariance")
    S = np.cov(Y, rowvar=False)
    assert np.trace(est.to_dense()) == pytest.approx(np.trace(S), rel=1e-12)


# --------------------------------------------------------------------------- #
# LW 2020: a dense primal port of the published formulas
# --------------------------------------------------------------------------- #
def _lw2020_dense(Y: np.ndarray) -> np.ndarray:
    """Ledoit & Wolf (2020) analytical shrinkage, primal and literal.

    Demeans, ``n = N - 1``, eigh of the p x p sample covariance, and the
    kernel formulas (4.3), (4.7)-(4.9), (C.4), (C.5), (C.8) evaluated directly
    on the full ``m x m`` grid (no chunking, no series).
    """
    N, p = Y.shape
    n = N - 1
    Yc = Y - Y.mean(axis=0)
    S = Yc.T @ Yc / n
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    m = min(p, n)
    lam = lam[p - m :]
    U = U[:, p - m :]
    L = np.repeat(lam[:, None], m, axis=1)
    h = n ** (-1 / 3)
    H = h * L.T
    x = (L - L.T) / H
    ft = (3 / 4 / math.sqrt(5)) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = (-3 / 10 / math.pi) * x + (3 / 4 / math.sqrt(5) / math.pi) * (
            1 - x**2 / 5
        ) * np.log(np.abs((math.sqrt(5) - x) / (math.sqrt(5) + x)))
    hf[np.abs(x) == math.sqrt(5)] = (-3 / 10 / math.pi) * x[np.abs(x) == math.sqrt(5)]
    Hft = np.mean(hf / H, axis=1)
    c = p / n
    if p <= n:
        d = lam / (
            (math.pi * c * lam * ft) ** 2 + (1 - c - math.pi * c * lam * Hft) ** 2
        )
        return (U * d) @ U.T
    hf0 = (
        (1 / math.pi)
        * (
            3 / 10 / h**2
            + 3 / 4 / math.sqrt(5) / h * (1 - 1 / 5 / h**2)
            * math.log((1 + math.sqrt(5) * h) / (1 - math.sqrt(5) * h))
        )
        * np.mean(1 / lam)
    )  # fmt: skip
    d0 = 1 / (math.pi * (p - n) / n * hf0)
    d1 = lam / (math.pi**2 * lam**2 * (ft**2 + Hft**2))
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V[:, : p - m] * d0) @ V[:, : p - m].T + (U * d1) @ U.T


@pytest.mark.parametrize(("n", "p"), [(120, 40), (40, 120)])
def test_lw2020_matches_dense_port(n: int, p: int) -> None:
    Y = _data(n, p, seed=7 * n + p)
    got = estimate(Y, method="lw2020", space="covariance").to_dense()
    want = _lw2020_dense(Y)
    assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()
