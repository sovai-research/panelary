"""Hydra competing kernels over trailing windows (`panelary.embed.HydraEmbedder`).

Contracts touched: ``leakage_safe`` via trailing windows (leak suites in
``test_embed_leakage.py`` / ``test_embed_prefix_invariance.py``),
``fit_is_empty = True`` (kernels drawn from the seed), and the amplitude
guardrail -- ``normalise=False`` by default plus explicit scale features.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from panelary.embed import HydraEmbedder
from panelary.embed._hydra import KERNEL_TAPS, hydra_kernels

E, T = "entity", "time"


def _bruteforce(
    w: np.ndarray, kernels: np.ndarray, d: int, g: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    L = w.shape[0]
    positions = range((KERNEL_TAPS - 1) * d, L)
    soft = np.zeros((g, k))
    hard = np.zeros((g, k))
    for p in positions:
        resp = np.array(
            [
                sum(
                    kernels[i, j] * w[p - (KERNEL_TAPS - 1 - j) * d]
                    for j in range(KERNEL_TAPS)
                )
                for i in range(g * k)
            ]
        )
        R = resp.reshape(g, k)
        for gi in range(g):
            a = int(R[gi].argmax())
            soft[gi, a] += R[gi, a]
            hard[gi, int(R[gi].argmin())] += 1
    m = len(positions)
    return soft.ravel() / m, hard.ravel() / m


def test_matches_a_bruteforce_implementation():
    h = HydraEmbedder(window=24, n_groups=3, n_kernels=4, seed=1, scale_features=False)
    W = np.random.default_rng(0).standard_normal((3, 24)).cumsum(axis=1)
    F = h._embed_windows(W)
    gk = 12
    col = 0
    for ri, _rep, fits in h._plan_reps:
        for di, d in fits:
            for row in range(3):
                series = W[row] if ri == 0 else np.diff(W[row])
                soft, hard = _bruteforce(series, h.kernels_[ri, di], d, 3, 4)
                np.testing.assert_allclose(
                    F[row, col : col + gk], soft, rtol=1e-12, atol=1e-14
                )
                np.testing.assert_allclose(F[row, col + gk : col + 2 * gk], hard)
            col += 2 * gk
    assert col == F.shape[1]


def test_kernels_are_centred_and_l1_normalised():
    w = hydra_kernels(50, np.random.default_rng(0))
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(w).sum(axis=1), 1.0)


def test_hard_tallies_sum_to_one_per_group_and_rows_are_independent():
    h = HydraEmbedder(window=32, n_groups=2, n_kernels=4, scale_features=False)
    W = np.random.default_rng(1).standard_normal((20, 32))
    F = h._embed_windows(W)
    hard = F[:, 8:16].reshape(20, 2, 4)  # x, first dilation, hard block
    np.testing.assert_allclose(hard.sum(axis=2), 1.0)
    np.testing.assert_array_equal(h._embed_windows(W[:2]), F[:2])


def test_normalise_equals_per_window_z_normalisation():
    raw = HydraEmbedder(window=30, scale_features=False)
    norm = HydraEmbedder(window=30, normalise=True, scale_features=False)
    W = np.random.default_rng(2).standard_normal((10, 30)).cumsum(axis=1) * 5 + 3
    Wz = (W - W.mean(axis=1, keepdims=True)) / W.std(axis=1, keepdims=True)
    np.testing.assert_allclose(
        norm._embed_windows(W), raw._embed_windows(Wz), rtol=1e-10, atol=1e-12
    )


def test_amplitude_is_retained_by_default_and_scale_features_are_emitted():
    h = HydraEmbedder(window=30)
    W = np.random.default_rng(3).standard_normal((5, 30))
    F1, F3 = h._embed_windows(W), h._embed_windows(3.0 * W)
    assert h.feature_names_[-3:] == ["scale_std", "scale_mad", "scale_rv"]
    np.testing.assert_allclose(F3[:, -3:], 3.0 * F1[:, -3:])
    first_soft = slice(0, h.n_groups * h.n_kernels)
    np.testing.assert_allclose(F3[:, first_soft], 3.0 * F1[:, first_soft], rtol=1e-12)
    med = np.median(W[0])
    np.testing.assert_allclose(
        F1[0, -3:],
        [
            W[0].std(),
            np.median(np.abs(W[0] - med)),
            np.sqrt((np.diff(W[0]) ** 2).sum()),
        ],
    )


def test_transform_state_and_parameters():
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            E: np.repeat(["a", "b"], 50),
            T: np.tile(np.arange(50, dtype=np.int64), 2),
            "x": rng.standard_normal(100).cumsum(),
        }
    )
    h = HydraEmbedder(
        window=20,
        n_groups=2,
        n_kernels=3,
        warmup="null",
        dtype="float64",
        entity=E,
        time=T,
    )
    out = h.fit_transform(df).collect().sort(E, T)
    assert out.schema["x_hydra"] == pl.Array(pl.Float64, h.n_outputs)
    x = df.filter(pl.col(E) == "a").sort(T)["x"].to_numpy()
    np.testing.assert_allclose(
        np.asarray(out["x_hydra"][40]), h._embed_windows(x[None, 21:41])[0]
    )
    again = HydraEmbedder.from_state(h.get_state())
    assert again.transform(df).collect().sort(E, T).equals(out)
    assert (
        HydraEmbedder.fit_is_empty is True
        and "center" not in inspect.signature(HydraEmbedder).parameters
    )
    with pytest.raises(ValueError):
        HydraEmbedder(window=9)
    with pytest.raises(ValueError):
        HydraEmbedder(n_kernels=1)


def test_scatter_uses_bincount_not_add_at():
    import panelary.embed._hydra as mod

    src = inspect.getsource(mod)
    assert "np.bincount" in src and "add.at(" not in src
