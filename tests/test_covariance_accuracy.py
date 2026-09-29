"""Plan section 1.1 as assertions: the out-of-sample GMV experiment (slow).

Population: one market factor, eight sector factors and heterogeneous
idiosyncratic variance, Student-t(5) innovations, ``W = 252``; the metric is
the true variance of the global-minimum-variance weights built from each
estimator divided by the oracle minimum (1.0 is perfect), median of 30 draws.

Measured on this generator (Apple M5 Pro, Accelerate, 4 threads):

============================  =============  ==============
estimator                     N=100 (q=0.4)  N=500 (q=1.98)
============================  =============  ==============
QIS, correlation space        1.234          1.369
LW identity, covariance       1.455          2.462
naive sample (np.linalg.solve) --            ~970
sample, pinv                  1.63           5.6
============================  =============  ==============

The plan quotes 2,095x for "sample covariance, pinv when singular"; with a
true ``pinv`` the singular sample gives 5.6x on this generator, while the
path that raises nothing -- ``np.linalg.solve`` on the numerically singular
``np.cov`` -- gives ~970x. The >= 100x assertion is made on that silent path.
"""

from __future__ import annotations

import numpy as np
import pytest

from panelary.covariance._estimate import estimate

pytestmark = pytest.mark.slow

_DRAWS = 30
_W = 252


def _population(N: int, rng: np.random.Generator):
    beta = rng.uniform(0.5, 1.5, N)
    sector = np.arange(N) % 8
    sload = rng.uniform(0.3, 0.8, N)
    idio = rng.uniform(0.5, 2.0, N)
    L = np.zeros((N, 9))
    L[:, 0] = beta
    L[np.arange(N), 1 + sector] = sload
    fvol = np.array([1.0] + [0.5] * 8)
    return L, fvol, idio


def _draw(L, fvol, idio, rng: np.random.Generator, df: int = 5) -> np.ndarray:
    s = np.sqrt(df / (df - 2))
    F = rng.standard_t(df, (_W, L.shape[1])) / s * fvol
    E = rng.standard_t(df, (_W, L.shape[0])) / s * idio
    return F @ L.T + E


def _gmv_ratios(N: int) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for d in range(_DRAWS):
        rng = np.random.default_rng(1000 + d)
        L, fvol, idio = _population(N, rng)
        Sigma = (L * fvol**2) @ L.T + np.diag(idio**2)
        one = np.ones(N)
        opt = 1.0 / float(one @ np.linalg.solve(Sigma, one[:, None])[:, 0])
        X = _draw(L, fvol, idio, rng)

        def ratio(w: np.ndarray) -> float:
            return float(w @ Sigma @ w) / opt  # noqa: B023 - evaluated in-loop

        for name, method, space in (
            ("qis_corr", "qis", "correlation"),
            ("lw_identity", "lw_identity", "covariance"),
        ):
            est = estimate(X, method=method, space=space)
            out.setdefault(name, []).append(ratio(est.gmv_weights()))
        S = np.cov(X, rowvar=False)
        with np.errstate(all="ignore"):
            x = np.linalg.solve(S, one[:, None])[:, 0]
        out.setdefault("sample_solve", []).append(ratio(x / x.sum()))
        x = np.linalg.pinv(S) @ one
        out.setdefault("sample_pinv", []).append(ratio(x / x.sum()))
    return {k: float(np.median(v)) for k, v in out.items()}


def test_section_1_1_at_q_0_4() -> None:
    r = _gmv_ratios(100)
    assert r["qis_corr"] <= 1.25
    assert r["qis_corr"] < r["lw_identity"]


def test_section_1_1_at_q_2() -> None:
    r = _gmv_ratios(500)
    assert r["qis_corr"] <= 1.40
    assert r["qis_corr"] < r["lw_identity"]
    # The silently-wrong baseline must stay wrong. Never loosen this.
    assert r["sample_solve"] >= 100.0
    assert r["sample_pinv"] >= 3.0
