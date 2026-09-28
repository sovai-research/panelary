"""Distance covariance / correlation against brute force (and ``dcor`` if present)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary.depend import DcorResult, dcor, dcor_pvalue, dcov2, partial_dcor


def _brute(
    x: np.ndarray, y: np.ndarray, bias_corrected: bool = True
) -> tuple[float, float, float]:
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    n = x.shape[0]
    a = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    b = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(-1))

    def center(d: np.ndarray) -> np.ndarray:
        if not bias_corrected:
            return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
        c = (
            d
            - d.sum(0) / (n - 2)
            - d.sum(1)[:, None] / (n - 2)
            + d.sum() / ((n - 1) * (n - 2))
        )
        np.fill_diagonal(c, 0.0)
        return c

    A, B = center(a), center(b)
    f = 1.0 / (n * (n - 3)) if bias_corrected else 1.0 / n**2
    return f * (A * B).sum(), f * (A * A).sum(), f * (B * B).sum()


@pytest.mark.parametrize("bias_corrected", [True, False])
def test_univariate_matches_bruteforce(bias_corrected: bool) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(301)
    y = np.round(x**2 + 0.5 * rng.standard_normal(301), 1)  # ties
    res = dcov2(x, y, bias_corrected=bias_corrected)
    cov, vx, vy = _brute(x, y, bias_corrected)
    assert res.path == "univariate"
    assert res.dcov2 == pytest.approx(cov, rel=1e-9, abs=1e-12)
    assert res.dvar_x == pytest.approx(vx, rel=1e-9)
    assert res.dvar_y == pytest.approx(vy, rel=1e-9)
    assert res.dcor == pytest.approx(cov / math.sqrt(vx * vy), rel=1e-9)


def test_multivariate_blocked_matches_bruteforce() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((250, 3))
    y = np.column_stack(
        [np.sin(x[:, 0]), x[:, 1] * x[:, 2]]
    ) + 0.1 * rng.standard_normal((250, 2))
    res = dcov2(x, y)
    cov, vx, vy = _brute(x, y)
    assert res.path == "blocked" and not res.approximate
    assert res.dcov2 == pytest.approx(cov, rel=1e-9)
    assert res.dcor == pytest.approx(cov / math.sqrt(vx * vy), rel=1e-9)


def test_subsample_above_max_n_is_seeded_and_recorded() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((600, 2))
    y = x[:, :1] ** 2 + 0.1 * rng.standard_normal((600, 1))
    a = dcov2(x, y, max_n=200, seed=3)
    b = dcov2(x, y, max_n=200, seed=3)
    c = dcov2(x, y, max_n=200, seed=4)
    assert a.approximate and a.n_subsample == 200 and a.n_used == 200 and a.n_obs == 600
    assert a.dcor == b.dcor
    assert a.dcor != c.dcor
    # The univariate path is exact at any n: never subsampled.
    u = dcov2(x[:, 0], y[:, 0], max_n=50)
    assert not u.approximate and u.n_used == 600


def test_dcor_detects_nonmonotone_and_pvalues() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(400)
    res = dcov2(x, x**2 + 0.3 * rng.standard_normal(400))
    assert res.dcor > 0.2
    assert dcor_pvalue(res) < 1e-8
    null = dcov2(x, rng.standard_normal(400))
    assert 0.0 <= dcor_pvalue(null) <= 1.0
    with pytest.raises(ValueError):
        dcor_pvalue(dcov2(x, x, bias_corrected=False))
    with pytest.raises(ValueError):
        dcor_pvalue(res, method="nope")
    assert isinstance(res, DcorResult)
    assert dcor(x, x) == pytest.approx(1.0)


def test_partial_dcor() -> None:
    rng = np.random.default_rng(4)
    z = rng.standard_normal(300)
    x = z + 0.1 * rng.standard_normal(300)
    y = z + 0.1 * rng.standard_normal(300)
    assert dcor(x, y) > 0.8
    assert abs(partial_dcor(x, y, z)) < 0.15
