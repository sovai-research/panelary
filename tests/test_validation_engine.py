"""The shared evaluation engine: column HAC (`_hac`) and count-matrix resampling (`_resample`)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelary.validation import _hac, newey_west_variance, romano_wolf
from panelary.validation._resample import (
    bootstrap_means,
    circular_block_draw,
    circular_block_sums,
    count_matrix,
    matmul_cols,
    row_chunks,
    start_counts,
    stationary_indices,
)
from panelary.validation._sharpe import _cbb_engine


def _ar1(rng: np.random.Generator, n: int, m: int, phi: float) -> np.ndarray:
    e = rng.standard_normal((n, m))
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# --------------------------------------------------------------------------- #
# _hac
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lags", [0, 1, 4, 9])
def test_bartlett_matches_newey_west_variance(lags: int) -> None:
    x = _ar1(np.random.default_rng(0), 400, 6, 0.4)
    got = _hac.bartlett_lrv(x, lags)
    want = np.array([newey_west_variance(x[:, j], lags=lags) for j in range(6)])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert float(_hac.bartlett_lrv(x[:, 2], lags)) == pytest.approx(want[2], rel=1e-12)


def test_fft_autocovariances_match_direct_sums() -> None:
    x = _ar1(np.random.default_rng(1), 257, 5, 0.6)
    direct = _hac.autocovariances(x, 256)
    np.testing.assert_allclose(_hac.autocovariances_fft(x), direct, rtol=0, atol=1e-12)


def test_qs_lrv_matches_direct_quadratic_sum() -> None:
    x = _ar1(np.random.default_rng(2), 300, 4, 0.3)
    s = _hac.andrews_bandwidth(x)
    got = _hac.kernel_lrv(x, s, kernel="qs")
    n = x.shape[0]
    xc = x - x.mean(0)
    want = np.empty(4)
    for m in range(4):
        total = 0.0
        for j in range(-(n - 1), n):
            g = float(xc[abs(j) :, m] @ xc[: n - abs(j), m]) / n
            total += float(_hac.qs_kernel(j / s[m])) * g
        want[m] = total
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_qs_kernel_values() -> None:
    assert float(_hac.qs_kernel(0.0)) == 1.0
    # Series branch and closed form agree across the switch point.
    edge = 0.2 / (1.2 * math.pi)
    a, b = _hac.qs_kernel(np.array([edge * (1 - 1e-13), edge * (1 + 1e-13)]))
    assert abs(a - b) < 1e-14
    x = 0.7
    z = 6 * math.pi * x / 5
    want = 25 / (12 * math.pi**2 * x**2) * (math.sin(z) / z - math.cos(z))
    assert float(_hac.qs_kernel(x)) == pytest.approx(want, rel=1e-14)


def test_andrews_bandwidth_hand_value() -> None:
    x = _ar1(np.random.default_rng(3), 500, 1, 0.5)[:, 0]
    xc = x - x.mean()
    rho = float(xc[1:] @ xc[:-1]) / float(xc[:-1] @ xc[:-1])
    alpha2 = 4 * rho**2 / (1 - rho) ** 4
    assert float(_hac.andrews_bandwidth(x)) == pytest.approx(
        1.3221 * (alpha2 * 500) ** 0.2, rel=1e-13
    )
    alpha1 = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    assert float(_hac.andrews_bandwidth(x, kernel="bartlett")) == pytest.approx(
        1.1447 * (alpha1 * 500) ** (1 / 3), rel=1e-13
    )


def test_prewhitening_recolours_and_caps_rho() -> None:
    rng = np.random.default_rng(4)
    x = _ar1(rng, 2000, 1, 0.7)[:, 0]
    lrv, s, rho = _hac.prewhitened_lrv(x)
    assert float(rho) == pytest.approx(0.7, abs=0.05)
    # AR(1) with unit innovations: LRV = 1 / (1 - phi)^2 = 11.1
    assert float(lrv) == pytest.approx(1 / 0.3**2, rel=0.2)
    near_unit = np.cumsum(rng.standard_normal(500))
    _, _, rho_cap = _hac.prewhitened_lrv(near_unit)
    assert abs(float(rho_cap)) <= 0.97


def test_vector_lrv_without_prewhitening_is_quadratic_form_of_scalar_lrv() -> None:
    y = _ar1(np.random.default_rng(5), 400, 3, 0.4)
    w = np.array([0.3, -1.2, 0.8])
    psi, s = _hac.vector_lrv_prewhitened(y, prewhiten=False, small_sample=False)
    scalar = _hac.kernel_lrv(y @ w, s, kernel="qs")
    assert float(w @ psi @ w) == pytest.approx(float(scalar), rel=1e-12)


def test_expanding_bartlett_path_is_prefix_recomputation() -> None:
    x = _ar1(np.random.default_rng(6), 200, 3, 0.2) + 5.0
    path = _hac.expanding_bartlett_lrv(x, 3)
    assert np.isnan(path[:4]).all()
    for t in (4, 17, 120, 199):
        np.testing.assert_allclose(
            path[t], _hac.bartlett_lrv(x[: t + 1], 3), rtol=1e-10
        )
    # a scan: appending rows never changes earlier ones
    np.testing.assert_array_equal(_hac.expanding_bartlett_lrv(x[:150], 3), path[:150])
    with pytest.raises(ValueError, match="fixed non-negative integer"):
        _hac.expanding_bartlett_lrv(x, 2.5)  # type: ignore[arg-type]


def test_column_lrv_labels_and_methods() -> None:
    x = _ar1(np.random.default_rng(7), 300, 2, 0.3)
    assert _hac.column_lrv(x, "bartlett", lags=4).label == "bartlett(L=4)"
    for method in _hac.LRV_METHODS:
        res = _hac.column_lrv(x, method)
        assert res.lrv.shape == (2,)
        assert np.all(res.lrv > 0)
    assert np.ndim(_hac.column_lrv(x[:, 0], "qs_pw").lrv) == 0
    with pytest.raises(ValueError, match="unknown HAC"):
        _hac.column_lrv(x, "nope")


def test_mask_groups() -> None:
    mask = np.array([[1, 1, 0, 1], [1, 1, 1, 1], [0, 0, 1, 0]], bool)
    groups = _hac.mask_groups(mask)
    assert [c.tolist() for _, c in groups] == [[0, 1, 3], [2]]
    assert [r.tolist() for r, _ in groups] == [[0, 1], [1, 2]]


# --------------------------------------------------------------------------- #
# _resample
# --------------------------------------------------------------------------- #
def test_count_matrix_means_match_gather() -> None:
    rng = np.random.default_rng(10)
    x = rng.standard_normal((300, 7))
    idx = rng.integers(0, 300, size=(64, 300))
    want = np.stack([x[i].mean(0) for i in idx])
    np.testing.assert_allclose(bootstrap_means(x, idx), want, rtol=1e-14, atol=1e-15)
    c = count_matrix(idx, 300)
    assert c.sum(axis=1).tolist() == [300.0] * 64
    with pytest.raises(ValueError, match="outside"):
        count_matrix(idx + 300, 300)


def test_bootstrap_means_chunking_is_invariant_to_rounding() -> None:
    # Chunking changes the gemm shape, and BLAS (OpenBLAS; Accelerate on some
    # CPUs) may block a different shape differently: the draws are identical,
    # the means agree to rounding (about 1 ulp), not bitwise.
    rng = np.random.default_rng(11)
    x = rng.standard_normal((500, 9))
    idx = rng.integers(0, 500, size=(400, 500))
    whole = bootstrap_means(x, idx)
    chunked = bootstrap_means(x, idx, chunk_bytes=500 * 8 * 100)
    np.testing.assert_allclose(whole, chunked, rtol=1e-13, atol=1e-16)
    one = bootstrap_means(x[:, 3], idx)
    np.testing.assert_allclose(one, whole[:, 3], rtol=1e-13, atol=1e-16)


def test_row_chunks_are_balanced() -> None:
    parts = row_chunks(1001, 8, 8 * 1000)
    sizes = [p.stop - p.start for p in parts]
    assert sum(sizes) == 1001 and min(sizes) >= 500
    assert row_chunks(0, 8, 8) == []


def test_matmul_cols_pads_single_column() -> None:
    rng = np.random.default_rng(12)
    a = rng.integers(0, 3, (50, 80)).astype(float)
    b = rng.standard_normal((80, 5))
    np.testing.assert_array_equal(
        matmul_cols(a, b[:, 2:3])[:, 0], matmul_cols(a, b)[:, 2]
    )


def test_circular_draw_respects_segments() -> None:
    draw = circular_block_draw(103, 7, 200, boundaries=[40, 71], seed=3)
    assert draw.blocks_per_segment == (5, 4, 4)
    assert draw.n_sample == 13 * 7
    idx = draw.indices()
    col = 0
    for (a, stop), nb in zip(draw.segments, draw.blocks_per_segment, strict=True):
        block = idx[:, col : col + nb * 7]
        assert block.min() >= a and block.max() < stop
        col += nb * 7
    with pytest.raises(ValueError, match="shortest segment"):
        circular_block_draw(100, 30, 10, boundaries=[20], seed=0)


def test_start_counts_and_block_sums_reproduce_resample_sums() -> None:
    rng = np.random.default_rng(13)
    x = rng.standard_normal((90, 3))
    draw = circular_block_draw(90, 4, 30, boundaries=[50], seed=5)
    q = circular_block_sums(x, 4, draw.segments)
    got = start_counts(draw) @ q
    want = np.stack([x[i].sum(0) for i in draw.indices()])
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)
    # block sums wrap inside their own segment
    assert q[49, 0] == pytest.approx(x[49, 0] + x[0, 0] + x[1, 0] + x[2, 0])


def test_stationary_indices_structure() -> None:
    idx = stationary_indices(50, 5.0, 200, length=80, seed=1)
    assert idx.shape == (200, 80)
    assert idx.min() >= 0 and idx.max() < 50
    steps = (np.diff(idx, axis=1) % 50) == 1
    mean_run = 1.0 / (1.0 - steps.mean())
    assert 4.0 < mean_run < 6.0
    np.testing.assert_array_equal(
        idx, stationary_indices(50, 5.0, 200, length=80, seed=1)
    )


@pytest.mark.parametrize("statistic", ["sharpe", "log_variance"])
def test_start_count_cbb_matches_literal_ledoit_wolf(statistic: str) -> None:
    """Delta* and s(Delta*) equal LW's literal zeta-form on materialised resamples."""
    rng = np.random.default_rng(14)
    n, m, b = 125, 3, 5
    x = rng.standard_normal((n, m)) * 0.05 + 0.01
    y = rng.standard_normal(n) * 0.04 + 0.008
    draw = circular_block_draw(n, b, 40, boundaries=[60], seed=2)
    est, se = _cbb_engine(x, y, draw, statistic)
    idx = draw.indices()
    ns, blocks = draw.n_sample, draw.n_sample // b
    for rep in range(40):
        for j in range(m):
            r1, r2 = x[idx[rep], j], y[idx[rep]]
            mu1, mu2 = r1.mean(), r2.mean()
            g1, g2 = np.mean(r1 * r1), np.mean(r2 * r2)
            v1, v2 = g1 - mu1**2, g2 - mu2**2
            if statistic == "sharpe":
                lit = mu1 / r1.std(ddof=1) - mu2 / r2.std(ddof=1)
                grad = np.array(
                    [
                        g1 / v1**1.5,
                        -g2 / v2**1.5,
                        -0.5 * mu1 / v1**1.5,
                        0.5 * mu2 / v2**1.5,
                    ]
                )
            else:
                lit = math.log(r1.var()) - math.log(r2.var())
                grad = np.array([-2 * mu1 / v1, 2 * mu2 / v2, 1 / v1, -1 / v2])
            ystar = np.column_stack([r1 - mu1, r2 - mu2, r1 * r1 - g1, r2 * r2 - g2])
            psi = np.zeros((4, 4))
            for k in range(blocks):
                zeta = math.sqrt(b) * ystar[k * b : (k + 1) * b].mean(0)
                psi += np.outer(zeta, zeta)
            psi /= blocks
            assert est[rep, j] == pytest.approx(lit, rel=1e-12, abs=1e-13)
            assert se[rep, j] == pytest.approx(
                math.sqrt(grad @ psi @ grad / ns), rel=1e-12
            )


def test_cbb_engine_chunking_and_column_independence_agree_to_rounding() -> None:
    # Same draws; BLAS may round a differently shaped product differently (seen
    # on CI's OpenBLAS and Apple M1 runners), so compare to rounding.
    rng = np.random.default_rng(15)
    x = rng.standard_normal((400, 6)) * 0.02 + 0.001
    y = rng.standard_normal(400) * 0.02
    draw = circular_block_draw(400, 6, 300, seed=9)
    est, se = _cbb_engine(x, y, draw, "sharpe")
    est_c, se_c = _cbb_engine(x, y, draw, "sharpe", chunk_bytes=400 * 8 * 160 * 2)
    np.testing.assert_allclose(est, est_c, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(se, se_c, rtol=1e-12, atol=1e-15)
    est_1, se_1 = _cbb_engine(x[:, [4]], y, draw, "sharpe")
    np.testing.assert_allclose(est_1[:, 0], est[:, 4], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(se_1[:, 0], se[:, 4], rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------- #
# Romano-Wolf: the suffix-max stepdown is bitwise the old loop
# --------------------------------------------------------------------------- #
def _romano_wolf_reference(
    t: np.ndarray, boot: np.ndarray, two_sided: bool
) -> np.ndarray:
    """The pre-2026-09 O(B S^2) loop, kept verbatim as the golden reference."""
    t_use = np.abs(t) if two_sided else t
    boot_use = np.abs(boot) if two_sided else boot
    n_boot, n_hyp = boot_use.shape
    order = np.argsort(-t_use)
    adj = np.empty(n_hyp, dtype=float)
    remaining = list(order)
    running = 0.0
    for j, h in enumerate(order):
        cols = np.asarray(remaining, dtype=np.int64)
        max_null = boot_use[:, cols].max(axis=1)
        p_raw = (1.0 + float(np.count_nonzero(max_null >= t_use[h]))) / (n_boot + 1.0)
        running = max(running, p_raw)
        adj[h] = running
        remaining = list(order[j + 1 :])
        if not remaining:
            break
    return adj


@pytest.mark.parametrize(
    ("n_boot", "n_hyp"), [(99, 1), (199, 2), (499, 7), (999, 50), (300, 200)]
)
@pytest.mark.parametrize("two_sided", [True, False])
def test_romano_wolf_suffix_max_is_bitwise_the_loop(
    n_boot: int, n_hyp: int, two_sided: bool
) -> None:
    rng = np.random.default_rng(n_boot + n_hyp)
    t = rng.standard_normal(n_hyp) * 2
    boot = rng.standard_normal((n_boot, n_hyp))
    if n_hyp >= 7:
        t[3] = t[4]  # a tie
        boot[5, 2] = np.nan
        t[6] = np.nan
    got = romano_wolf(t, boot, alpha=0.05, two_sided=two_sided)
    want = _romano_wolf_reference(t, boot, two_sided)
    np.testing.assert_array_equal(got.adjusted_pvalues, want)
    np.testing.assert_array_equal(got.rejected, want <= 0.05)
