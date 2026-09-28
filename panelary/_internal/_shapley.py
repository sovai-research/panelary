"""Shapley combinatorics for games whose coalitions are *executed*.

This is the neutral core behind every "which part is worth how much?" question
Panelary answers by rerunning something:

* :func:`panelary.leakage.borrowed_accuracy` -- players are pipeline stages, a
  coalition runs its stages permissively, ``v(S)`` is a backtest score;
* :func:`panelary.select.refit_shapley` -- players are feature groups, a
  coalition is the set of groups the model is refit on, ``v(S)`` is a purged-CV
  score.

Both are interventional (do-)Shapley values in the sense of Jung et al. (ICML
2022): every ``v(S)`` is the outcome of *running* the intervention, not an
estimate identified from observational data, so none of that paper's
estimators are needed -- only the combinatorics, which live here.

Everything is numeric. A game is ``k`` players numbered ``0..k-1`` and a value
function over **bitmasks** (bit ``i`` set <=> player ``i`` in the coalition).
Values may be scalars or arrays of any fixed trailing shape -- one entry per
replicate seed, per fold, or both. Shapley is linear in ``v``, so the value of
an array game is the array of the per-entry values: efficiency holds within
every entry separately, exactly.

Two solvers:

* :func:`exact_shapley` -- all ``2**k`` coalitions, weights
  ``|S|! (k-|S|-1)! / k!``; nothing sampled.
* :func:`permutation_shapley` -- Monte-Carlo over player orderings (Castro,
  Gomez & Tejada, 2009; Jung et al., 2022, Algorithm 1), drawn in antithetic
  (reversed) pairs. Each ordering's marginal contributions telescope to
  ``v(N) - v({})``, so the estimate keeps efficiency to floating-point
  tolerance however few orderings are drawn; the per-player Monte-Carlo
  standard error is reported alongside.

:class:`CoalitionCache` memoises the value function so a coalition is never
paid for twice, and :func:`solve` puts the pieces together.

References
----------
Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory
of Games* II, 307-317.

Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of the
Shapley value based on sampling. *Computers & Operations Research*, 36(5),
1726-1730.

Jung, Y., Kasiviswanathan, S., Tian, J., Janzing, D., Bloebaum, P., &
Bareinboim, E. (2022). On measuring causal contributions via do-interventions.
*ICML 2022*, PMLR 162:10476-10501.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "METHODS",
    "CoalitionCache",
    "PermutationEstimate",
    "ShapleySolution",
    "canonical_json",
    "exact_shapley",
    "json_float",
    "mask_of",
    "mc_standard_error",
    "members",
    "permutation_cost_bound",
    "permutation_shapley",
    "popcounts",
    "shapley_weights",
    "solve",
    "validate_method",
]

#: Solvers :func:`solve` accepts.
METHODS: tuple[str, ...] = ("exact", "permutation")


# --------------------------------------------------------------------------- #
# Bitmask helpers
# --------------------------------------------------------------------------- #
def members(mask: int, names: Sequence[str]) -> frozenset[str]:
    """The names whose bits are set in ``mask``.

    Examples
    --------
    >>> sorted(members(0b101, ["a", "b", "c"]))
    ['a', 'c']
    """
    return frozenset(name for i, name in enumerate(names) if mask >> i & 1)


def mask_of(selection: Iterable[str], names: Sequence[str]) -> int:
    """The bitmask of ``selection`` over ``names`` (inverse of :func:`members`).

    Raises
    ------
    KeyError
        If ``selection`` contains a name that is not in ``names``.

    Examples
    --------
    >>> mask_of({"a", "c"}, ["a", "b", "c"])
    5
    """
    index = {name: i for i, name in enumerate(names)}
    mask = 0
    for name in selection:
        mask |= 1 << index[name]
    return mask


def popcounts(k: int) -> np.ndarray:
    """``popcounts(k)[m]`` is the number of set bits in ``m``, for ``m < 2**k``."""
    masks = np.arange(1 << k, dtype=np.int64)
    counts = np.zeros(1 << k, dtype=np.int64)
    for i in range(k):
        counts += (masks >> i) & 1
    return counts


def shapley_weights(k: int) -> np.ndarray:
    """``w[s] = s! (k-s-1)! / k!`` for ``s = 0..k-1``.

    ``w[s]`` is the probability that a uniformly random ordering puts exactly
    ``s`` given players before a given player -- the weight of a marginal
    contribution ``v(S + i) - v(S)`` with ``|S| = s``. Computed as
    ``1 / (k * C(k-1, s))`` so the only rounding is one division.

    Examples
    --------
    >>> [round(float(w), 6) for w in shapley_weights(3)]
    [0.333333, 0.166667, 0.333333]
    """
    if k < 1:
        raise ValueError(f"a game needs at least one player, got k={k}.")
    return np.array([1.0 / (k * math.comb(k - 1, s)) for s in range(k)])


# --------------------------------------------------------------------------- #
# Exact solver
# --------------------------------------------------------------------------- #
def exact_shapley(values: np.ndarray) -> np.ndarray:
    """Exact Shapley values from the whole characteristic function.

    Parameters
    ----------
    values : numpy.ndarray
        Shape ``(2**k, *trailing)``: ``values[m]`` is ``v`` of the coalition
        with bitmask ``m``. Any trailing shape is allowed; each trailing entry
        is solved as its own game (Shapley is linear in ``v``).

    Returns
    -------
    numpy.ndarray
        Shape ``(k, *trailing)``, float64. ``out.sum(axis=0)`` equals
        ``values[-1] - values[0]`` to floating-point tolerance (efficiency).

    Raises
    ------
    ValueError
        If the leading dimension is not a power of two ``>= 2``.

    Examples
    --------
    Two players who only score together split the joint value evenly:

    >>> exact_shapley(np.array([0.0, 0.0, 0.0, 1.0])).tolist()
    [0.5, 0.5]
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0] if v.ndim else 0
    k = n.bit_length() - 1
    if n < 2 or n != 1 << k:
        raise ValueError(
            "exact_shapley needs 2**k coalition values (k >= 1), one per "
            f"bitmask; got a leading dimension of {n}."
        )
    masks = np.arange(n, dtype=np.int64)
    counts = popcounts(k)
    weights = shapley_weights(k)
    trailing = (1,) * (v.ndim - 1)
    out = np.empty((k, *v.shape[1:]), dtype=np.float64)
    for i in range(k):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        gains = v[without | bit] - v[without]
        # numpy's pairwise summation, not BLAS: bit-reproducible run to run.
        out[i] = (weights[counts[without]].reshape(-1, *trailing) * gains).sum(axis=0)
    return out


# --------------------------------------------------------------------------- #
# Memoised value function
# --------------------------------------------------------------------------- #
class CoalitionCache:
    """Memoise a coalition value function so each coalition is scored once.

    Parameters
    ----------
    fn : callable
        ``fn(mask: int) -> array-like``. Called at most once per mask.

    Attributes
    ----------
    values : dict of int to numpy.ndarray
        Every coalition scored so far, keyed by bitmask, in evaluation order.
    """

    def __init__(self, fn: Callable[[int], Any]) -> None:
        self._fn = fn
        self.values: dict[int, np.ndarray] = {}

    def __call__(self, mask: int) -> np.ndarray:
        if mask not in self.values:
            self.values[mask] = np.asarray(self._fn(mask), dtype=np.float64)
        return self.values[mask]

    @property
    def n_evaluations(self) -> int:
        """Distinct coalitions scored: the real cost of the game."""
        return len(self.values)


# --------------------------------------------------------------------------- #
# Permutation (Monte-Carlo) solver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PermutationEstimate:
    """Result of :func:`permutation_shapley`.

    Attributes
    ----------
    phi : numpy.ndarray
        ``(k, *trailing)`` estimated Shapley values. Sums over players to
        ``v(N) - v({})`` to floating-point tolerance, whatever the sample size.
    standard_error : numpy.ndarray
        ``(k, *trailing)`` Monte-Carlo standard error of ``phi``: the sample
        standard deviation (``ddof=1``) of the independent units, over the
        square root of their number. ``nan`` when there is only one unit.
    n_permutations : int
        Orderings walked (both members of every antithetic pair count).
    n_units : int
        Independent samples the standard error rests on: antithetic pairs, or
        single orderings when ``antithetic=False``.
    samples : numpy.ndarray
        ``(n_units, k, *trailing)`` per-unit marginal contributions; ``phi`` is
        their mean. Kept so a caller can take the standard error of any linear
        summary (a mean over replicates, say) with :func:`mc_standard_error`.
    """

    phi: np.ndarray
    standard_error: np.ndarray
    n_permutations: int
    n_units: int
    samples: np.ndarray


def _walk(
    value: Callable[[int], np.ndarray],
    order: Sequence[int],
    v_empty: np.ndarray,
) -> np.ndarray:
    """Marginal contributions of each player along one ordering."""
    contrib = np.empty((len(order), *v_empty.shape), dtype=np.float64)
    mask = 0
    prev = v_empty
    for player in order:
        mask |= 1 << int(player)
        cur = value(mask)
        contrib[int(player)] = cur - prev
        prev = cur
    return contrib


def permutation_shapley(
    value: Callable[[int], np.ndarray],
    k: int,
    *,
    n_permutations: int,
    seed: int,
    antithetic: bool = True,
) -> PermutationEstimate:
    """Monte-Carlo Shapley values over random orderings of the players.

    Each ordering ``pi`` contributes, for every player ``i``, the marginal
    contribution ``v(Pre_i(pi) + i) - v(Pre_i(pi))``. Over a uniformly random
    ordering its expectation is exactly ``phi_i``. With ``antithetic=True``
    orderings come in pairs ``(pi, reversed(pi))`` and each pair's mean is one
    sample: a player late in ``pi`` is early in its reverse, which cancels much
    of the variance that comes from position.

    Parameters
    ----------
    value : callable
        ``value(mask) -> numpy.ndarray``. Wrap it in :class:`CoalitionCache`
        so that repeated coalitions -- ``v({})`` and ``v(N)`` on every walk --
        are paid for once; ``M`` orderings then cost at most ``M (k-1) + 2``
        distinct evaluations.
    k : int
        Number of players, ``>= 1``.
    n_permutations : int
        Orderings to walk. With ``antithetic=True`` it must be even and at
        least 4 (two pairs, so the standard error has a degree of freedom);
        otherwise at least 2.
    seed : int
        Seed for :func:`numpy.random.default_rng`. Same seed, same orderings.
    antithetic : bool, default=True
        Draw orderings in reversed pairs.

    Returns
    -------
    PermutationEstimate

    Raises
    ------
    ValueError
        On an invalid ``k`` or ``n_permutations``.
    """
    if k < 1:
        raise ValueError(f"a game needs at least one player, got k={k}.")
    _check_n_permutations(n_permutations, antithetic)
    rng = np.random.default_rng(seed)
    n_units = n_permutations // 2 if antithetic else n_permutations
    v_empty = value(0)
    samples = np.empty((n_units, k, *v_empty.shape), dtype=np.float64)
    for u in range(n_units):
        order = [int(p) for p in rng.permutation(k)]
        contrib = _walk(value, order, v_empty)
        if antithetic:
            contrib = 0.5 * (contrib + _walk(value, order[::-1], v_empty))
        samples[u] = contrib
    return PermutationEstimate(
        phi=samples.mean(axis=0),
        standard_error=mc_standard_error(samples),
        n_permutations=n_permutations,
        n_units=n_units,
        samples=samples,
    )


def mc_standard_error(samples: np.ndarray) -> np.ndarray:
    """Monte-Carlo standard error of the mean over the leading (unit) axis.

    ``std(ddof=1) / sqrt(n)``; ``nan`` everywhere when there is one unit.
    Because Shapley estimates are means of per-unit samples, the standard
    error of any linear summary -- ``phi`` averaged over replicates, say -- is
    this function applied to the same summary of ``samples``.
    """
    s = np.asarray(samples, dtype=np.float64)
    n = s.shape[0]
    if n < 2:
        return np.full(s.shape[1:], np.nan)
    return s.std(axis=0, ddof=1) / math.sqrt(n)


def _check_n_permutations(n_permutations: Any, antithetic: bool) -> None:
    if isinstance(n_permutations, bool) or not isinstance(
        n_permutations, (int, np.integer)
    ):
        raise TypeError(
            f"n_permutations must be an int, got {type(n_permutations).__name__}."
        )
    if antithetic:
        if n_permutations < 4 or n_permutations % 2:
            raise ValueError(
                f"n_permutations={n_permutations}: orderings are drawn in "
                "antithetic (reversed) pairs, so it must be even, and at least 4 "
                "so the Monte-Carlo standard error rests on two independent pairs."
            )
    elif n_permutations < 2:
        raise ValueError(
            f"n_permutations={n_permutations}: need at least 2 orderings for a "
            "Monte-Carlo standard error."
        )


def permutation_cost_bound(k: int, n_permutations: int) -> int:
    """Upper bound on distinct evaluations: ``min(M (k-1) + 2, 2**k)``.

    ``v({})`` and ``v(N)`` are shared by every ordering, and each ordering
    visits ``k - 1`` coalitions strictly between them.

    Examples
    --------
    >>> permutation_cost_bound(20, 50)
    952
    """
    return min(n_permutations * (k - 1) + 2, 1 << k)


# --------------------------------------------------------------------------- #
# One entry point
# --------------------------------------------------------------------------- #
def validate_method(method: str, n_permutations: int | None, *, caller: str) -> None:
    """Check a ``method`` / ``n_permutations`` pair before any work is done.

    ``n_permutations`` is required with ``method="permutation"`` and refused
    with ``method="exact"``: a knob that is silently ignored is a bug report
    waiting to happen.
    """
    if method not in METHODS:
        raise ValueError(
            f"{caller}: method must be one of {list(METHODS)}, got {method!r}."
        )
    if method == "permutation":
        if n_permutations is None:
            raise ValueError(
                f"{caller}: method='permutation' needs n_permutations=M, an even "
                "number of orderings to sample (e.g. 50); there is no default, "
                "because M sets both the cost and the Monte-Carlo error."
            )
        _check_n_permutations(n_permutations, antithetic=True)
    elif n_permutations is not None:
        raise ValueError(
            f"{caller}: n_permutations={n_permutations} only applies to "
            "method='permutation'; method='exact' samples nothing."
        )


@dataclass(frozen=True)
class ShapleySolution:
    """What :func:`solve` returns: values, error, and the whole game.

    Attributes
    ----------
    phi : numpy.ndarray
        ``(k, *trailing)`` Shapley values.
    standard_error : numpy.ndarray or None
        ``(k, *trailing)`` Monte-Carlo standard error; ``None`` for ``exact``.
    values : dict of int to numpy.ndarray
        Every coalition scored, keyed by bitmask, in evaluation order.
    method : str
        ``"exact"`` or ``"permutation"``.
    n_permutations : int or None
        Orderings walked, for ``permutation``.
    seed : int or None
        Sampler seed, for ``permutation``.
    samples : numpy.ndarray or None
        ``(n_units, k, *trailing)`` per-unit marginal contributions, for
        ``permutation``; see :func:`mc_standard_error`.
    """

    phi: np.ndarray
    standard_error: np.ndarray | None
    values: dict[int, np.ndarray]
    method: str
    n_permutations: int | None = None
    seed: int | None = None
    samples: np.ndarray | None = None

    @property
    def n_evaluations(self) -> int:
        """Distinct coalitions scored."""
        return len(self.values)


def solve(
    value: Callable[[int], Any],
    k: int,
    *,
    method: str = "exact",
    n_permutations: int | None = None,
    seed: int = 0,
) -> ShapleySolution:
    """Solve a ``k``-player game by ``method``, memoising every coalition.

    ``v({})`` (mask ``0``) is always evaluated first, so a caller validating
    the shape of what ``value`` returns can take the first call as the
    reference. ``exact`` then evaluates masks in ascending order.

    Parameters
    ----------
    value : callable
        ``value(mask) -> array-like`` of a fixed shape.
    k : int
        Number of players, ``>= 1``.
    method : {"exact", "permutation"}
        See the module docstring.
    n_permutations : int, optional
        Required for, and only for, ``method="permutation"``.
    seed : int, default=0
        Sampler seed for ``method="permutation"``; unused by ``exact``.

    Returns
    -------
    ShapleySolution
    """
    validate_method(method, n_permutations, caller="solve")
    if k < 1:
        raise ValueError(f"a game needs at least one player, got k={k}.")
    cache = CoalitionCache(value)
    v_empty = cache(0)
    if method == "exact":
        table = np.empty((1 << k, *v_empty.shape), dtype=np.float64)
        for mask in range(1 << k):
            table[mask] = cache(mask)
        return ShapleySolution(
            phi=exact_shapley(table),
            standard_error=None,
            values=cache.values,
            method="exact",
        )
    assert n_permutations is not None  # validate_method guarantees it
    est = permutation_shapley(cache, k, n_permutations=n_permutations, seed=seed)
    return ShapleySolution(
        phi=est.phi,
        standard_error=est.standard_error,
        values=cache.values,
        method="permutation",
        n_permutations=n_permutations,
        seed=seed,
        samples=est.samples,
    )


# --------------------------------------------------------------------------- #
# Serialisation helpers shared by the report types
# --------------------------------------------------------------------------- #
def json_float(x: float) -> float | None:
    """``x`` as a JSON-safe float: non-finite values become ``None``."""
    x = float(x)
    return x if math.isfinite(x) else None


def canonical_json(payload: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialise a report dict: byte-deterministic when ``indent is None``.

    ``indent=None`` gives ``json.dumps(payload, sort_keys=True,
    separators=(",", ":"))`` -- the shared convention for Panelary report
    types, so that two runs producing the same numbers produce the same bytes.
    ``allow_nan=False``: a non-finite float that escaped :func:`json_float` is
    an error, never an invalid ``NaN`` token in the output.
    """
    if indent is None:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False)
