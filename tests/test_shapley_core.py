"""The neutral Shapley core: exact and permutation solvers, memo, JSON helpers.

:mod:`panelary._internal._shapley` is what :func:`panelary.leakage.borrowed_accuracy`
and :func:`panelary.select.refit_shapley` both stand on, so it is checked here
against the definition itself -- the average of marginal contributions over
all ``k!`` orderings -- rather than against either caller.
"""

from __future__ import annotations

import doctest
import itertools
import json
import math

import numpy as np
import pytest

from panelary._internal import _shapley
from panelary._internal._shapley import (
    CoalitionCache,
    canonical_json,
    exact_shapley,
    json_float,
    mask_of,
    mc_standard_error,
    members,
    permutation_cost_bound,
    permutation_shapley,
    popcounts,
    shapley_weights,
    solve,
    validate_method,
)

TOL = 1e-12


def random_table(k: int, seed: int, trailing: tuple[int, ...] = ()) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((1 << k, *trailing))


def by_orderings(table: np.ndarray, k: int) -> np.ndarray:
    """Shapley by definition: mean marginal contribution over all k! orderings."""
    total = np.zeros((k, *table.shape[1:]))
    count = 0
    for order in itertools.permutations(range(k)):
        mask = 0
        for player in order:
            total[player] += table[mask | 1 << player] - table[mask]
            mask |= 1 << player
        count += 1
    return total / count


# --------------------------------------------------------------------------- #
# Exact solver
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_exact_matches_the_definition_over_all_orderings(k: int) -> None:
    table = random_table(k, seed=10 + k)
    np.testing.assert_allclose(exact_shapley(table), by_orderings(table, k), atol=TOL)


@pytest.mark.parametrize("k", [1, 3, 6, 9])
def test_exact_efficiency(k: int) -> None:
    table = random_table(k, seed=k)
    phi = exact_shapley(table)
    assert phi.sum() == pytest.approx(table[-1] - table[0], abs=1e-11)


def test_trailing_dimensions_are_solved_independently() -> None:
    """Shapley is linear in v: an array game is the array of its entries' games."""
    k = 4
    table = random_table(k, seed=3, trailing=(3, 2))
    phi = exact_shapley(table)
    assert phi.shape == (k, 3, 2)
    for r in range(3):
        for f in range(2):
            np.testing.assert_allclose(
                phi[:, r, f], exact_shapley(table[:, r, f]), atol=TOL
            )
    # Efficiency per entry.
    np.testing.assert_allclose(phi.sum(axis=0), table[-1] - table[0], atol=1e-11)


def test_exact_rejects_a_table_that_is_not_2_to_the_k() -> None:
    with pytest.raises(ValueError, match="2\\*\\*k coalition values"):
        exact_shapley(np.zeros(6))
    with pytest.raises(ValueError, match="2\\*\\*k coalition values"):
        exact_shapley(np.zeros(1))


def test_weights_are_a_probability_distribution_over_predecessor_sets() -> None:
    for k in range(1, 12):
        w = shapley_weights(k)
        # Each size s has C(k-1, s) predecessor sets; the weights must sum to 1.
        assert sum(math.comb(k - 1, s) * w[s] for s in range(k)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        shapley_weights(0)


def test_popcounts_and_masks() -> None:
    assert popcounts(3).tolist() == [0, 1, 1, 2, 1, 2, 2, 3]
    names = ["a", "b", "c", "d"]
    for mask in range(16):
        assert mask_of(members(mask, names), names) == mask


# --------------------------------------------------------------------------- #
# Permutation solver
# --------------------------------------------------------------------------- #
def cached(table: np.ndarray) -> CoalitionCache:
    return CoalitionCache(lambda m: table[m])


@pytest.mark.parametrize("k", [3, 7, 15])
def test_permutation_keeps_efficiency_at_any_sample_size(k: int) -> None:
    table = random_table(k, seed=k)
    est = permutation_shapley(cached(table), k, n_permutations=4, seed=0)
    assert est.phi.sum() == pytest.approx(table[-1] - table[0], abs=1e-10)
    # ... and within every single unit, not just on average.
    np.testing.assert_allclose(
        est.samples.sum(axis=1), table[-1] - table[0], atol=1e-10
    )


def test_permutation_cost_is_bounded_by_m_k_minus_1_plus_2() -> None:
    k, m = 20, 50

    def value(mask: int) -> float:
        return float(bin(mask).count("1")) ** 1.5

    cache = CoalitionCache(value)
    permutation_shapley(cache, k, n_permutations=m, seed=1)
    assert cache.n_evaluations <= m * (k - 1) + 2 == permutation_cost_bound(k, m)
    assert permutation_cost_bound(20, 50) == 952
    assert permutation_cost_bound(3, 50) == 8  # never more than 2**k


def test_permutation_is_exact_on_an_additive_game() -> None:
    """No interaction: every ordering gives the same marginals, so SE is zero."""
    k = 6
    w = np.linspace(-1.0, 2.0, k)
    table = np.array([sum(w[i] for i in range(k) if m >> i & 1) for m in range(1 << k)])
    est = permutation_shapley(cached(table), k, n_permutations=6, seed=5)
    np.testing.assert_allclose(est.phi, w, atol=TOL)
    np.testing.assert_allclose(est.standard_error, 0.0, atol=TOL)


def test_antithetic_pairs_cancel_a_pairwise_interaction_exactly() -> None:
    """In (pi, reversed pi) exactly one ordering puts b before a: zero variance."""
    k = 5
    table = np.array(
        [1.0 if (m & 0b11) == 0b11 else 0.0 for m in range(1 << k)], dtype=np.float64
    )
    est = permutation_shapley(cached(table), k, n_permutations=4, seed=2)
    np.testing.assert_allclose(est.phi, [0.5, 0.5, 0.0, 0.0, 0.0], atol=TOL)
    np.testing.assert_allclose(est.standard_error, 0.0, atol=TOL)


def test_permutation_converges_to_exact_within_its_standard_error() -> None:
    k = 6
    table = random_table(k, seed=99)
    exact = exact_shapley(table)
    est = permutation_shapley(cached(table), k, n_permutations=400, seed=0)
    z = np.abs(est.phi - exact) / est.standard_error
    assert np.all(z < 5.0), z
    assert np.all(est.standard_error > 0)


def test_permutation_is_deterministic_in_the_seed() -> None:
    k = 8
    table = random_table(k, seed=4)
    a = permutation_shapley(cached(table), k, n_permutations=10, seed=7)
    b = permutation_shapley(cached(table), k, n_permutations=10, seed=7)
    c = permutation_shapley(cached(table), k, n_permutations=10, seed=8)
    np.testing.assert_array_equal(a.phi, b.phi)
    np.testing.assert_array_equal(a.standard_error, b.standard_error)
    assert not np.array_equal(a.phi, c.phi)


def test_permutation_supports_trailing_dimensions() -> None:
    k = 5
    table = random_table(k, seed=6, trailing=(4,))
    est = permutation_shapley(cached(table), k, n_permutations=8, seed=0)
    assert est.phi.shape == est.standard_error.shape == (k, 4)
    np.testing.assert_allclose(est.phi.sum(axis=0), table[-1] - table[0], atol=1e-10)
    # SE of a linear summary comes from the same summary of the samples.
    se_mean = mc_standard_error(est.samples.mean(axis=-1))
    assert se_mean.shape == (k,)


def test_non_antithetic_mode() -> None:
    k = 4
    table = random_table(k, seed=1)
    est = permutation_shapley(
        cached(table), k, n_permutations=3, seed=0, antithetic=False
    )
    assert est.n_units == 3 and est.n_permutations == 3
    with pytest.raises(ValueError, match="at least 2"):
        permutation_shapley(
            cached(table), k, n_permutations=1, seed=0, antithetic=False
        )


@pytest.mark.parametrize("bad", [0, 2, 3, 5, 51])
def test_antithetic_needs_an_even_count_of_at_least_four(bad: int) -> None:
    with pytest.raises(ValueError, match="even"):
        permutation_shapley(cached(np.zeros(8)), 3, n_permutations=bad, seed=0)


def test_n_permutations_must_be_an_int() -> None:
    with pytest.raises(TypeError, match="int"):
        permutation_shapley(cached(np.zeros(8)), 3, n_permutations=4.0, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        permutation_shapley(cached(np.zeros(8)), 3, n_permutations=True, seed=0)  # type: ignore[arg-type]


def test_mc_standard_error_of_one_unit_is_nan() -> None:
    assert np.isnan(mc_standard_error(np.ones((1, 3)))).all()


# --------------------------------------------------------------------------- #
# The memo and the entry point
# --------------------------------------------------------------------------- #
def test_cache_scores_each_coalition_once() -> None:
    calls: list[int] = []

    def value(mask: int) -> float:
        calls.append(mask)
        return float(mask)

    cache = CoalitionCache(value)
    for mask in [0, 3, 3, 0, 5]:
        cache(mask)
    assert calls == [0, 3, 5]
    assert cache.n_evaluations == 3


def test_solve_exact_evaluates_the_empty_coalition_first_and_every_mask_once() -> None:
    order: list[int] = []
    table = random_table(4, seed=0)

    def value(mask: int) -> float:
        order.append(mask)
        return float(table[mask])

    sol = solve(value, 4)
    assert order == list(range(16))
    assert sol.n_evaluations == 16 and sol.standard_error is None
    np.testing.assert_allclose(sol.phi, exact_shapley(table), atol=TOL)


def test_solve_permutation_evaluates_the_empty_coalition_first() -> None:
    order: list[int] = []

    def value(mask: int) -> float:
        order.append(mask)
        return float(mask % 7)

    sol = solve(value, 9, method="permutation", n_permutations=4, seed=0)
    assert order[0] == 0
    assert sol.method == "permutation" and sol.n_permutations == 4
    assert sol.standard_error is not None and sol.samples is not None


def test_validate_method_refuses_silent_knobs() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        validate_method("sampled", None, caller="x")
    with pytest.raises(ValueError, match="needs n_permutations"):
        validate_method("permutation", None, caller="x")
    with pytest.raises(ValueError, match="only applies to method='permutation'"):
        validate_method("exact", 10, caller="x")
    validate_method("exact", None, caller="x")
    validate_method("permutation", 10, caller="x")


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def test_canonical_json_is_byte_deterministic_and_strict() -> None:
    a = canonical_json({"b": 1.5, "a": [1, 2], "schema": "x/1"})
    b = canonical_json({"schema": "x/1", "a": [1, 2], "b": 1.5})
    assert a == b == '{"a":[1,2],"b":1.5,"schema":"x/1"}'
    assert json.loads(canonical_json({"a": 1}, indent=2)) == {"a": 1}
    with pytest.raises(ValueError):
        canonical_json({"a": math.nan})


def test_json_float() -> None:
    assert json_float(0.25) == 0.25
    assert json_float(math.nan) is None
    assert json_float(math.inf) is None


def test_module_doctests_pass() -> None:
    results = doctest.testmod(_shapley, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
