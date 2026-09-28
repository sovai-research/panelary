"""Calibration of :mod:`panelary.depend` nulls -- the most important tests of
the module.

These reproduce the build contract's motivating table as assertions:

* the i.i.d. null is **broken** on two independent persistent series
  (type-I error > 30% at phi = 0.95, nominal 5%). This test exists to prove
  the bug is real and must never be "fixed" by loosening it;
* the serial-dependence nulls (``block``, ``shift``, ``iaaft``) are
  **calibrated** (type-I error in ``[3%, 8%]``);
* on a panel whose entities load on common factors, per-entity shuffles fail
  and ``null="common-time"`` passes;
* the Szekely-Rizzo dCor t-test is miscalibrated at p = 1 and the gamma null
  is not.

The full Monte Carlo runs are ``@pytest.mark.slow``; the default run keeps a
cheap version of each claim.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from panelary.depend import (
    SerialDependenceWarning,
    dcor_pvalue,
    dcov2,
    independence_test,
    xi_pvalue,
)
from panelary.depend._coef import _xi_rows
from panelary.depend._engine import panel_test, series_test
from panelary.depend._kernels import get_kernel

LO, HI = 0.03, 0.08


def ar1_batch(
    rng: np.random.Generator, phi: float, reps: int, n: int, burn: int = 200
) -> np.ndarray:
    """``(reps, n)`` independent stationary AR(1) paths (unit innovation sd)."""
    e = rng.standard_normal((reps, n + burn))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(max(1e-12, 1.0 - phi * phi))
    for t in range(1, n + burn):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x[:, burn:]


def _rate(
    null: str,
    phi: float,
    reps: int,
    *,
    method: str = "xi",
    n: int = 500,
    n_resamples: int = 199,
    seed: int = 12345,
) -> float:
    rng = np.random.default_rng(seed)
    X = ar1_batch(rng, phi, reps, n)
    Y = ar1_batch(rng, phi, reps, n)
    k = get_kernel(method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SerialDependenceWarning)
        rej = [
            series_test(X[r], Y[r], k, null=null, n_resamples=n_resamples, seed=r)[
                "p_value"
            ]
            <= 0.05
            for r in range(reps)
        ]
    return float(np.mean(rej))


# --------------------------------------------------------------------------- #
# The i.i.d. null is broken on persistent data -- asserted, never loosened
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("phi", "lo", "hi"), [(0.0, LO, HI), (0.7, 0.05, 0.12), (0.95, 0.30, 1.0)]
)
def test_iid_null_type1_error(phi: float, lo: float, hi: float) -> None:
    """2000 reps, n=500, xi under N(0, 2/5): 5% / ~7% / >30% (contract: 5.2 / 6.9 / 56.3)."""
    n, reps = 500, 2000
    rng = np.random.default_rng(20260909)
    X = ar1_batch(rng, phi, reps, n)
    Y = ar1_batch(rng, phi, reps, n)
    est = _xi_rows(X, Y)
    p = np.array([xi_pvalue(e, n) for e in est])
    rate = float(np.mean(p <= 0.05))
    assert lo <= rate <= hi, rate
    if phi == 0.95:
        # The bug is an order of magnitude, not a rounding error.
        assert rate > 0.30


def test_iid_null_public_path_matches_and_warns() -> None:
    rng = np.random.default_rng(1)
    x = ar1_batch(rng, 0.95, 1, 500)[0]
    y = ar1_batch(rng, 0.95, 1, 500)[0]
    with pytest.warns(SerialDependenceWarning):
        out = independence_test(x, y, method="xi", null="iid")
    assert out["null_method"][0] == "asymptotic"
    assert out["p_value"][0] == pytest.approx(xi_pvalue(float(out["estimate"][0]), 500))
    assert any("serially independent" in w for w in out["warnings"][0])
    # null="auto" routes the same data to a serial-dependence-valid null, silently
    # for the user but visibly in the result.
    with warnings.catch_warnings():
        warnings.simplefilter("error", SerialDependenceWarning)
        auto = independence_test(x, y, method="xi")
    assert auto["null_method"][0] == "block"
    assert auto["block_length"][0] > 1


# --------------------------------------------------------------------------- #
# Serial-dependence nulls are calibrated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("null", ["block", "shift"])
def test_serial_nulls_calibrated_smoke(null: str) -> None:
    """Default-run version: 300 reps, B=99 at phi=0.95 (measured 4.3% / 5.0%)."""
    rate = _rate(null, 0.95, 300, n_resamples=99, seed=2024)
    assert 0.02 <= rate <= 0.09, rate


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.7, 0.95])
@pytest.mark.parametrize("null", ["block", "shift"])
@pytest.mark.parametrize("method", ["xi", "spearman"])
def test_serial_nulls_calibrated(method: str, null: str, phi: float) -> None:
    """2000 reps, n=500, B=199: type-I error in [3%, 8%]."""
    rate = _rate(null, phi, 2000, method=method)
    assert LO <= rate <= HI, rate


@pytest.mark.slow
def test_iaaft_null_calibrated() -> None:
    """IAAFT costs ~0.15 s per test at B=99, so 400 reps."""
    rate = _rate("iaaft", 0.95, 400, n_resamples=99)
    assert LO <= rate <= HI, rate


# --------------------------------------------------------------------------- #
# Panel with common factors: per-entity shuffles fail, common-time passes
# --------------------------------------------------------------------------- #
def _panel_rate(
    null: str,
    phi: float,
    reps: int,
    *,
    method: str = "spearman",
    n_resamples: int = 199,
    seed: int = 777,
) -> float:
    rng = np.random.default_rng(seed)
    n_ent, t_len = 20, 250
    k = get_kernel(method)
    rej = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SerialDependenceWarning)
        for r in range(reps):
            f = ar1_batch(rng, phi, 1, t_len, burn=100)[0] * np.sqrt(1 - phi**2)
            g = ar1_batch(rng, phi, 1, t_len, burn=100)[0] * np.sqrt(1 - phi**2)
            X = f[None, :] + 0.7 * rng.standard_normal((n_ent, t_len))
            Y = g[None, :] + 0.7 * rng.standard_normal((n_ent, t_len))
            row, _, _ = panel_test(X, Y, k, null=null, n_resamples=n_resamples, seed=r)
            rej.append(row["p_value"] <= 0.05)
    return float(np.mean(rej))


def test_panel_common_factor_smoke() -> None:
    """Default-run version (60 reps): the per-entity null is wildly off,
    common-time is not."""
    bad = _panel_rate("permutation", 0.0, 60, n_resamples=99)
    good = _panel_rate("common-time", 0.0, 60, n_resamples=99)
    assert bad > 0.30, bad
    assert good <= 0.12, good


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.9])
def test_panel_common_factor(phi: float) -> None:
    """Spearman (a global cross-product: common factors bite), 400 reps."""
    assert _panel_rate("permutation", phi, 400) > 0.30
    assert _panel_rate("block", phi, 400) > 0.30
    assert LO <= _panel_rate("common-time", phi, 400) <= HI


# --------------------------------------------------------------------------- #
# dCor: the t-test is for the high-dimensional limit, not p = 1
# --------------------------------------------------------------------------- #
def test_dcor_t_test_miscalibrated_at_p1() -> None:
    rng = np.random.default_rng(3)
    reps, n = 3000, 50
    rt = rg = 0
    for _ in range(reps):
        res = dcov2(rng.standard_normal(n), rng.standard_normal(n))
        rt += dcor_pvalue(res, method="t") <= 0.05
        rg += dcor_pvalue(res, method="gamma") <= 0.05
    t_rate, g_rate = rt / reps, rg / reps
    assert LO <= g_rate <= 0.07, g_rate
    # Measured 6.8% vs 5.45%: the t-test over-rejects at p=1.
    assert t_rate > g_rate + 0.007, (t_rate, g_rate)
    # method="auto" must therefore pick the gamma null for univariate data.
    res = dcov2(rng.standard_normal(n), rng.standard_normal(n))
    assert dcor_pvalue(res) == dcor_pvalue(res, method="gamma")
    assert dcor_pvalue(res) != dcor_pvalue(res, method="t")
