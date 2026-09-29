"""Size and power of the Sharpe-difference tests on Ledoit & Wolf's (2008, §4.2) DGPs.

All DGPs have equal Sharpe ratios (the null is true), T = 120 as in LW:

* Normal-IID / t6-IID: bivariate, mean 1, variance 1, correlation 0.5 (t6 via a
  common chi-square mixing variable, standardised to unit variance);
* Normal-GARCH / t6-GARCH: LW's diagonal-vech GARCH(1,1) with
  ``C = [[.15, .13], [.13, .15]]``, ``A = [[.075, .05], [.05, .075]]``,
  ``B = [[.90, .89], [.89, .90]]`` and mean ``16.5/52``;
* Normal-VAR / t6-VAR: the i.i.d. DGPs with AR(1) ``phi = 0.2`` per series.

JKM's over-rejection under fat tails and serial correlation is a "the bug is
real" assertion: it must never be loosened. Replications are vectorised: the
test helpers run the same private engine the public functions run, sharing one
block-start matrix across a batch of independent datasets (which leaves each
dataset's null distribution unchanged), and a check pins the helper to the
public function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary.validation import _sharpe as S
from panelary.validation import sharpe_ratio_test
from panelary.validation._arrays import normal_pvalue, resample_pvalue
from panelary.validation._resample import circular_block_draw

pytestmark = pytest.mark.slow

T = 120
DGPS = ("Normal-IID", "t6-IID", "Normal-GARCH", "t6-GARCH", "Normal-VAR", "t6-VAR")


# --------------------------------------------------------------------------- #
# DGPs (vectorised over R replications; returns (T, R) arrays)
# --------------------------------------------------------------------------- #
def _bivariate(
    rng: np.random.Generator, t: int, r: int, fat: bool, rho: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.standard_normal((t, r))
    z2 = rng.standard_normal((t, r))
    a = z1
    b = rho * z1 + math.sqrt(1.0 - rho * rho) * z2
    if fat:  # bivariate t6, standardised to unit variance (var of t6 = 6/4)
        scale = np.sqrt(rng.chisquare(6, size=(t, r)) / 6.0) * math.sqrt(6.0 / 4.0)
        a, b = a / scale, b / scale
    return a, b


def lw_dgp(
    name: str, t: int, r: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    fat = name.startswith("t6")
    if name.endswith("IID"):
        a, b = _bivariate(rng, t, r, fat)
        return 1.0 + a, 1.0 + b
    if name.endswith("VAR"):
        burn = 50
        a, b = _bivariate(rng, t + burn, r, fat)
        x, y = np.empty_like(a), np.empty_like(b)
        x[0], y[0] = a[0] / math.sqrt(0.96), b[0] / math.sqrt(0.96)
        for s in range(1, t + burn):
            x[s] = 0.2 * x[s - 1] + a[s]
            y[s] = 0.2 * y[s - 1] + b[s]
        return 1.0 + x[burn:], 1.0 + y[burn:]
    c11, c12, a11, a12, b11, b12 = 0.15, 0.13, 0.075, 0.05, 0.90, 0.89
    burn = 200
    e1, e2 = _bivariate(rng, t + burn, r, fat, rho=0.0)
    h11 = np.full(r, c11 / (1 - a11 - b11))
    h22 = h11.copy()
    h12 = np.full(r, c12 / (1 - a12 - b12))
    x, y = np.empty((t + burn, r)), np.empty((t + burn, r))
    px, py = np.zeros(r), np.zeros(r)
    for s in range(t + burn):
        if s:
            h11 = c11 + a11 * px * px + b11 * h11
            h22 = c11 + a11 * py * py + b11 * h22
            h12 = c12 + a12 * px * py + b12 * h12
        l11 = np.sqrt(h11)
        l21 = h12 / l11
        l22 = np.sqrt(np.maximum(h22 - l21 * l21, 1e-12))
        px, py = l11 * e1[s], l21 * e1[s] + l22 * e2[s]
        x[s], y[s] = px, py
    mu = 16.5 / 52
    return mu + x[burn:], mu + y[burn:]


# --------------------------------------------------------------------------- #
# Batched tests (R independent datasets at once)
# --------------------------------------------------------------------------- #
def _observed(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, my = S._moments(x), S._moments(y)
    est = mx.sr - my.sr
    psi = S._influence(mx, "sharpe") - S._influence(my, "sharpe")
    var, _ = S._hac_variance(psi, "qs_pw", None, 1)
    return est, np.sqrt(var / x.shape[0]), psi


def closed_form_pvalues(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    mx, my = S._moments(x), S._moments(y)
    est = mx.sr - my.sr
    n = x.shape[0]
    if method == "jkm":
        rho = np.mean(mx.z * my.z, axis=0)
        var = 2 - 2 * rho + 0.5 * (mx.sr**2 + my.sr**2 - 2 * mx.sr * my.sr * rho**2)
    else:
        psi = S._influence(mx, "sharpe") - S._influence(my, "sharpe")
        var, _ = S._hac_variance(psi, method, None, 1)
    return normal_pvalue(est / np.sqrt(var / n), "two-sided")


def boot_ts_pvalues(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block: str,
    main_rng: np.random.Generator,
    cal_rng: np.random.Generator | None = None,
    n_boot: int = 499,
    n_sims: int = 100,
    cal_boot: int = 199,
) -> tuple[np.ndarray, np.ndarray]:
    """LW Boot-TS for R datasets; ``block`` is "auto" (Politis-White) or "calibrate"."""
    t, r = x.shape
    est, se, psi = _observed(x, y)
    if block == "auto":
        blocks = np.array([S._pw_block_length(psi[:, [j]]) for j in range(r)])
    else:
        assert cal_rng is not None
        grid = np.array(S._default_grid(t))
        px, py = np.empty((t, r * n_sims)), np.empty((t, r * n_sims))
        for j in range(r):
            ps, _ = S._pseudo_sequences(
                np.column_stack([x[:, j], y[:, j]]), n_sims, 5.0, cal_rng
            )
            px[:, j * n_sims : (j + 1) * n_sims] = ps[:, :, 0]
            py[:, j * n_sims : (j + 1) * n_sims] = ps[:, :, 1]
        est_k, s_k, _ = _observed(px, py)
        theta = np.repeat(est, n_sims)
        cover = np.empty((grid.size, r))
        for gi, b in enumerate(grid):
            draw = circular_block_draw(t, int(b), cal_boot, seed=cal_rng)
            e_star, s_star = S._cbb_engine(px, py, draw, "sharpe")
            q = S._abs_quantile((e_star - est_k) / s_star, 0.95)
            cover[gi] = (
                (np.abs(est_k - theta) <= q * s_k).reshape(r, n_sims).mean(axis=1)
            )
        blocks = grid[np.argmin(np.abs(cover - 0.95), axis=0)]
    pvals = np.empty(r)
    for b in np.unique(blocks):
        cols = np.flatnonzero(blocks == b)
        draw = circular_block_draw(t, int(b), n_boot, seed=main_rng)
        e_star, s_star = S._cbb_engine(x[:, cols], y[:, cols], draw, "sharpe")
        pvals[cols] = resample_pvalue(
            (e_star - est[cols]) / s_star, est[cols] / se[cols], "two-sided"
        )
    return pvals, blocks


def _size(pvalues: np.ndarray) -> float:
    return float(np.mean(pvalues < 0.05))


# --------------------------------------------------------------------------- #
# The helper is the public procedure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", ["auto", "calibrate"])
def test_batched_helper_reproduces_public_function(block: str) -> None:
    x, y = lw_dgp("t6-VAR", T, 1, np.random.default_rng(1))
    seed = 5
    public = sharpe_ratio_test(
        x[:, 0], y[:, 0], block_length=block, n_boot=499, calibration_sims=60, seed=seed
    )
    p, b = boot_ts_pvalues(
        x,
        y,
        block=block,
        main_rng=np.random.default_rng(seed),
        cal_rng=np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,))),
        n_sims=60,
        cal_boot=499,
    )
    assert int(b[0]) == public.block_length
    assert p[0] == public.pvalue[0]


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def null_draws() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260929)
    return {name: lw_dgp(name, T, 2000, rng) for name in DGPS}


@pytest.mark.parametrize("name", DGPS)
def test_jkm_and_hac_sizes(name: str, null_draws: dict) -> None:
    x, y = null_draws[name]
    jkm = _size(closed_form_pvalues(x, y, "jkm"))
    if name in ("t6-IID", "Normal-VAR", "t6-VAR"):
        # The bug is real: JKM over-rejects under fat tails / serial correlation.
        # Never loosen this bound.
        assert jkm > 0.07, (name, jkm)
    if name == "Normal-IID":
        assert 0.03 <= jkm <= 0.07, jkm
    for method in ("qs", "qs_pw"):
        size = _size(closed_form_pvalues(x, y, method))
        assert 0.03 <= size <= 0.10, (name, method, size)


@pytest.mark.parametrize("name", DGPS)
def test_boot_ts_auto_size(name: str, null_draws: dict) -> None:
    x, y = null_draws[name]
    p, _ = boot_ts_pvalues(x, y, block="auto", main_rng=np.random.default_rng(7))
    assert 0.03 <= _size(p) <= 0.07, (name, _size(p))


@pytest.mark.parametrize("name", ["t6-IID", "Normal-VAR"])
def test_boot_ts_calibrated_size(name: str, null_draws: dict) -> None:
    x, y = null_draws[name]
    x, y = x[:, :1000], y[:, :1000]
    p = np.concatenate(
        [
            boot_ts_pvalues(
                x[:, s : s + 100],
                y[:, s : s + 100],
                block="calibrate",
                main_rng=np.random.default_rng(100 + s),
                cal_rng=np.random.default_rng(200 + s),
            )[0]
            for s in range(0, 1000, 100)
        ]
    )
    assert 0.03 <= _size(p) <= 0.07, (name, _size(p))


def test_many_vs_benchmark_fwer_under_romano_wolf() -> None:
    """M = 50 null strategies (equal Sharpe, correlated, AR(1)) vs one benchmark."""
    rng = np.random.default_rng(31)
    m, reps = 50, 200
    hits = 0
    for _ in range(reps):
        f = rng.standard_normal((T + 50, 1))
        e = rng.standard_normal((T + 50, m + 1))
        u = math.sqrt(0.5) * f + math.sqrt(0.5) * e
        z = np.empty_like(u)
        z[0] = u[0] / math.sqrt(0.96)
        for s in range(1, T + 50):
            z[s] = 0.2 * z[s - 1] + u[s]
        r = 1.0 + z[50:]
        res = sharpe_ratio_test(
            r[:, 1:], r[:, 0], n_boot=499, seed=int(rng.integers(1 << 31))
        )
        assert res.adjustment == "romano-wolf"
        hits += bool(np.any(res.pvalue_adj < 0.05))
    assert hits / reps <= 0.07, hits / reps


def test_power_at_delta_sharpe_0_3() -> None:
    rng = np.random.default_rng(41)
    x, y = lw_dgp("Normal-IID", T, 1000, rng)
    x = x + 0.3  # SR_i = 1.3 vs SR_b = 1.0 per period
    p, _ = boot_ts_pvalues(x, y, block="auto", main_rng=np.random.default_rng(9))
    power = _size(p)
    assert power > 0.4, power
    jkm_null = _size(closed_form_pvalues(*lw_dgp("Normal-IID", T, 1000, rng), "jkm"))
    assert power > 5 * jkm_null
