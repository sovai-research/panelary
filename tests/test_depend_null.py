"""Null distributions and the null policy of :mod:`panelary.depend`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from panelary.depend import (
    NULL_POLICY,
    SerialDependenceWarning,
    auto_block_length,
    block_permutation_indices,
    circular_shift,
    common_time_indices,
    entity_permutation_indices,
    gamma_pvalue,
    iaaft,
    pair_block_length,
    phase_randomise,
    pvalue,
    serial_dependence,
)
from panelary.depend._null import resolve_null
from panelary.preprocessing._base import LeakageWarning


def _ar1(rng: np.random.Generator, phi: float, n: int) -> np.ndarray:
    e = rng.standard_normal(n + 200)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, e.size):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def test_null_policy_table_is_the_contract() -> None:
    """The build contract's table, encoded as a dict."""
    expected = {
        "xi": ("asymptotic", "block", "common-time"),
        "spearman": ("asymptotic", "block", "common-time"),
        "pearson": ("asymptotic", "block", "common-time"),
        "hoeffding": ("asymptotic", "block", "common-time"),
        "dcor_t": ("t", "block", "common-time"),
        "dcor": ("gamma", "block", "common-time"),
        "gcmi": ("asymptotic", "block", "common-time"),
        "hsic": ("gamma", "shift", "common-time"),
        "mi_ksg": ("permutation", "block", "common-time"),
        "transfer_entropy": ("asymptotic", "block", "common-time"),
        "gcm": ("asymptotic", "hac", "common-time"),
    }
    for stat, (iid, serial, panel) in expected.items():
        assert NULL_POLICY[stat] == {"iid": iid, "serial": serial, "panel": panel}, stat


def test_resolve_null_auto_branches_and_warning() -> None:
    assert resolve_null("xi", "auto", serial=0.05, panel=False)[0] == "asymptotic"
    assert resolve_null("xi", "auto", serial=0.5, panel=False)[0] == "block"
    assert resolve_null("xi", "auto", serial=0.05, panel=True)[0] == "common-time"
    assert resolve_null("hsic", "auto", serial=0.5, panel=False)[0] == "shift"
    assert resolve_null("xi", "iid", serial=0.05, panel=False)[0] == "asymptotic"
    with pytest.warns(SerialDependenceWarning) as rec:
        scheme, notes = resolve_null("xi", "iid", serial=0.9, panel=False)
    assert scheme == "asymptotic" and notes
    assert issubclass(rec[0].category, LeakageWarning)
    with pytest.raises(ValueError):
        resolve_null("xi", "bogus", serial=0.0, panel=False)
    assert (
        resolve_null("xi", "asymptotic", serial=0.0, panel=False, closed_form=False)[0]
        == "permutation"
    )


def test_pvalue_plus_one() -> None:
    draws = np.zeros(99)
    assert pvalue(1.0, draws) == pytest.approx(1 / 100)  # never exactly 0
    assert pvalue(-1.0, draws) == 1.0
    assert pvalue(0.0, np.array([-1.0, 1.0]), alternative="two-sided") == 1.0
    assert np.isnan(pvalue(np.nan, draws))
    with pytest.raises(ValueError):
        pvalue(1.0, draws, alternative="sideways")


def test_gamma_pvalue_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    mean, var = 3.0, 2.0
    for obs in (0.5, 3.0, 9.0):
        ref = stats.gamma.sf(obs, a=mean**2 / var, scale=var / mean)
        assert gamma_pvalue(obs, mean, var) == pytest.approx(ref, rel=1e-8)
    assert gamma_pvalue(-1.0, mean, var) == 1.0


def test_circular_shift_enumerates_when_asked_for_more() -> None:
    idx = circular_shift(10, n_resamples=1000, seed=0)
    assert idx.shape == (9, 10)  # every non-trivial rotation, once
    assert sorted(idx[:, 0].tolist()) == list(range(1, 10))
    sub = circular_shift(100, n_resamples=20, seed=1)
    assert sub.shape == (20, 100)
    assert len(set(sub[:, 0].tolist())) == 20
    np.testing.assert_array_equal(circular_shift(100, n_resamples=20, seed=1), sub)


def test_block_permutation_is_a_permutation_within_segments() -> None:
    idx = block_permutation_indices(
        103, block_length=10, n_resamples=50, seed=0, boundaries=[0, 60]
    )
    for row in idx:
        np.testing.assert_array_equal(np.sort(row[:60]), np.arange(60))
        np.testing.assert_array_equal(np.sort(row[60:]), np.arange(60, 103))
    # Consecutive positions stay consecutive (circularly) inside blocks.
    steps = np.diff(idx[:, :60], axis=1)
    assert np.mean((steps == 1) | (steps == -59)) > 0.8


def test_common_time_and_entity_indices() -> None:
    ct = common_time_indices(np.arange(40), n_resamples=7, seed=2, block=5)
    assert ct.shape == (7, 40)
    for row in ct:
        np.testing.assert_array_equal(np.sort(row), np.arange(40))
    ep = entity_permutation_indices(6, n_resamples=4, seed=3)
    assert ep.shape == (4, 6)
    with pytest.raises(ValueError):
        entity_permutation_indices(1, n_resamples=4, seed=3)


def test_iaaft_keeps_the_marginal_and_the_spectrum() -> None:
    rng = np.random.default_rng(4)
    x = np.exp(_ar1(rng, 0.8, 256))  # skewed, autocorrelated
    res = iaaft(x, n_resamples=5, seed=0)
    for b in range(5):
        np.testing.assert_array_equal(np.sort(res.values[b]), np.sort(x))
        np.testing.assert_array_equal(res.values[b], x[res.indices[b]])

    def acf(v: np.ndarray, k: int) -> float:
        v = v - v.mean()
        return float(v[:-k] @ v[k:] / (v @ v))

    for k in range(1, 6):
        target = acf(x, k)
        got = np.mean([acf(res.values[b], k) for b in range(5)])
        assert got == pytest.approx(target, abs=0.1), k
    assert res.converged.all()
    # Deterministic in the seed.
    np.testing.assert_array_equal(iaaft(x, n_resamples=5, seed=0).values, res.values)


def test_phase_randomise_keeps_the_amplitude_spectrum() -> None:
    rng = np.random.default_rng(5)
    x = _ar1(rng, 0.5, 200)
    res = phase_randomise(x, n_resamples=3, seed=1)
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(res.values - x.mean(), axis=1)),
        np.broadcast_to(np.abs(np.fft.rfft(x - x.mean())), (3, 101)),
        atol=1e-9,
    )


def test_block_lengths_grow_with_persistence() -> None:
    rng = np.random.default_rng(6)
    iid = auto_block_length(rng.standard_normal(500))
    per = auto_block_length(_ar1(rng, 0.95, 500))
    assert iid <= 3 < per
    pb = [
        pair_block_length(_ar1(rng, phi, 500), _ar1(rng, phi, 500))
        for phi in (0.0, 0.7, 0.95)
    ]
    assert pb[0] < pb[1] < pb[2] <= 125  # capped at n / 4
    assert pair_block_length(_ar1(rng, 0.95, 500), _ar1(rng, 0.95, 500)) >= 60


def test_serial_dependence_takes_the_min() -> None:
    rng = np.random.default_rng(7)
    a = _ar1(rng, 0.95, 400)
    b = rng.standard_normal(400)
    # One serially independent side makes the i.i.d. null exact.
    assert serial_dependence(a, b) < 0.2 < serial_dependence(a, a)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert serial_dependence(a) > 0.8
