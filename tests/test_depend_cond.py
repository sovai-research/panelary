"""Conditional dependence (GCM, CODEC, FOCI), KSG mutual information and
transfer entropy."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp


def _fork(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return z + 0.6 * rng.standard_normal(n), z + 0.6 * rng.standard_normal(n), z


def test_gcm_calibrated_and_powerful() -> None:
    pv = np.array([dp.gcm(*_fork(300, s)).p_value for s in range(600)])
    rate = float(np.mean(pv <= 0.05))
    assert 0.03 <= rate <= 0.08, rate  # x _||_ y | z holds
    x, y, z = _fork(300, 1)
    assert dp.gcm(x, y).p_value < 1e-10  # marginally dependent
    rng = np.random.default_rng(2)
    x2 = z + 0.3 * y + 0.5 * rng.standard_normal(300)
    assert dp.gcm(x2, y, z).p_value < 1e-3
    with pytest.raises(ValueError):
        dp.gcm(x, y, z, regressor="forest")


def test_gcm_hac_under_serial_dependence() -> None:
    """Persistent residuals: the iid GCM over-rejects, the HAC version does not."""
    rng = np.random.default_rng(3)

    def ar(n: int) -> np.ndarray:
        e = rng.standard_normal(n + 100)
        v = np.empty_like(e)
        v[0] = e[0]
        for t in range(1, e.size):
            v[t] = 0.9 * v[t - 1] + e[t]
        return v[100:]

    iid, hac = [], []
    for _ in range(300):
        z = rng.standard_normal(400)
        x, y = z + ar(400), z + ar(400)
        iid.append(dp.gcm(x, y, z).p_value <= 0.05)
        hac.append(dp.gcm(x, y, z, hac_lags="auto").p_value <= 0.05)
    assert np.mean(iid) > 0.25
    assert np.mean(hac) < np.mean(iid) / 2


def test_conditional_dependence_frame() -> None:
    rng = np.random.default_rng(4)
    n_ent, t_len = 6, 200
    z = rng.standard_normal((n_ent, t_len))
    x = z + 0.5 * rng.standard_normal((n_ent, t_len))
    y = z + 0.5 * rng.standard_normal((n_ent, t_len))
    df = pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_ent), t_len),
            "t": np.tile(np.arange(t_len), n_ent),
            "x": x.ravel(),
            "y": y.ravel(),
            "z": z.ravel(),
            "w": rng.standard_normal(n_ent * t_len),
        }
    )
    ci = dp.conditional_dependence(
        df, "x", "y", given=["z"], entity="e", time="t", n_resamples=99
    )
    assert ci["method"][0] == "gcm" and ci["null_method"][0] == "common-time"
    assert ci["p_value"][0] > 0.05 and "residualised" in ci["transform"][0]
    dep = dp.conditional_dependence(
        df, "x", "y", given=["w"], entity="e", time="t", n_resamples=99
    )
    assert dep["p_value"][0] == pytest.approx(2 / 100)  # two-sided floor at B=99
    g = dp.conditional_dependence(
        df, "x", "y", given=["z"], entity="e", time="t", method="gcmi"
    )
    assert abs(g["estimate"][0]) < 0.02
    with pytest.raises(ValueError):
        dp.conditional_dependence(df, "x", "y", given=[], entity="e", time="t")


def _codec_brute(y: np.ndarray, z: np.ndarray, x: np.ndarray | None) -> float:
    n = y.size
    r = np.array([(y <= v).sum() for v in y], dtype=float)
    ell = np.array([(y >= v).sum() for v in y], dtype=float)

    def nn(P: np.ndarray) -> np.ndarray:
        d = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        return d.argmin(1)

    zz = z.reshape(n, -1)
    if x is None:
        m = nn(zz)
        return float((n * np.minimum(r, r[m]) - ell**2).sum() / (ell * (n - ell)).sum())
    xx = x.reshape(n, -1)
    m, nx = nn(np.column_stack([xx, zz])), nn(xx)
    return float(
        (np.minimum(r, r[m]) - np.minimum(r, r[nx])).sum()
        / (r - np.minimum(r, r[nx])).sum()
    )


def test_codec_matches_definition_and_foci_order() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300)
    z = rng.standard_normal(300)
    y = np.sin(2 * x) + 0.2 * z + 0.1 * rng.standard_normal(300)
    assert dp.codec(y, z, x) == pytest.approx(_codec_brute(y, z, x), abs=1e-12)
    assert dp.codec(y, x) == pytest.approx(_codec_brute(y, x, None), abs=1e-12)
    assert dp.codec(y, x) > 0.5
    df = pl.DataFrame({"x": x, "z": z, "w": rng.standard_normal(300), "y": y})
    order = dp.foci(df, "y", ["w", "z", "x"])
    assert order[:2] == ["x", "z"]
    assert dp.foci(df, "y", ["w", "z", "x"], k=1) == ["x"]


def test_mi_ksg_gaussian_and_nonlinear() -> None:
    rng = np.random.default_rng(6)
    rho = 0.6
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=4000)
    truth = -0.5 * math.log(1 - rho * rho)
    assert dp.mi_ksg(z[:, 0], z[:, 1], k=5) == pytest.approx(truth, abs=0.03)
    x = rng.standard_normal(3000)
    y = x**2 + 0.2 * rng.standard_normal(3000)
    assert dp.mi_ksg(x, y) > 0.8  # genuinely nonlinear ...
    assert abs(dp.gcmi(x, y)) < 0.02  # ... where GCMI is blind
    w = rng.standard_normal(3000)
    assert dp.cmi_ksg(x, y, w) > 0.7
    assert abs(dp.cmi_ksg(w, rng.standard_normal(3000), x)) < 0.05
    df = pl.DataFrame({"x": x[:400], "y": y[:400]})
    out = dp.mutual_information(df, "x", "y", estimator="ksg", n_resamples=49)
    assert out["null_method"][0] == "permutation" and out["p_value"][
        0
    ] == pytest.approx(1 / 50)


def test_transfer_entropy_direction_and_null() -> None:
    rng = np.random.default_rng(8)
    n = 1500
    s = rng.standard_normal(n)
    g = np.zeros(n)
    for t in range(1, n):
        g[t] = 0.4 * g[t - 1] + 0.35 * s[t - 1] + rng.standard_normal()
    fwd = dp.transfer_entropy_array(s, g)
    bwd = dp.transfer_entropy_array(g, s)
    assert fwd.p_value < 1e-10 and bwd.p_value > 0.01 and fwd.te > 10 * abs(bwd.te)
    # Gaussian TE is the Granger log-likelihood ratio: 2 n TE ~ chi2(lag).
    pv = [
        dp.transfer_entropy_array(
            rng.standard_normal(300), rng.standard_normal(300)
        ).p_value
        for _ in range(500)
    ]
    assert 0.03 <= float(np.mean(np.array(pv) <= 0.05)) <= 0.08
    df = pl.DataFrame({"t": np.arange(n), "s": s, "g": g})
    out = dp.transfer_entropy(df, "s", "g", time="t")
    assert out["null_method"][0] == "asymptotic" and out["p_value"][0] < 1e-10
    assert "not causation" in out["warnings"][0][0]
    blk = dp.transfer_entropy(df, "s", "g", time="t", null="block", n_resamples=49)
    assert blk["p_value"][0] == pytest.approx(1 / 50)
    with pytest.raises(ValueError):
        dp.transfer_entropy_array(s, g, estimator="neural")
