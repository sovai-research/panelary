"""The statistic table: one :class:`Kernel` per dependence method.

A kernel bundles what the resampling engine needs to test a statistic without
knowing anything about it: a **row-batched** implementation (``(B, m)`` in,
``(B,)`` out -- so ``B`` null draws are one vectorised call), the direction of
the alternative, a closed-form i.i.d. p-value where one exists, the minimum
sample size below which it refuses to report a number, and how per-entity
estimates aggregate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from panelary.depend._coef import (
    _hoeffding_rows,
    _kendall_rows,
    _pearson_rows,
    _spearman_rows,
    _tail_rows,
    _xi_rows,
    hoeffding_pvalue,
    kendall_pvalue,
    pearson_pvalue,
    tail_pvalue,
    xi_pvalue,
)
from panelary.depend._energy import _dcor_rows, dcor_pvalue, dcov2
from panelary.depend._info import _gcmi_rows, gcmi_pvalue

__all__ = ["KERNELS", "METHODS", "Kernel", "get_kernel", "min_obs_for"]

ClosedForm = Callable[[float, np.ndarray, np.ndarray], "tuple[float, str]"]
ZVar = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Kernel:
    """Everything the engine needs to know about one statistic.

    Attributes
    ----------
    name : str
        Method name as accepted by the public API.
    rows : callable
        ``rows(X, Y) -> (B,)`` on ``(B, m)`` complete arrays.
    alternative : {"greater", "two-sided"}
        ``"greater"`` for statistics that are ~0 under independence and grow
        with dependence (xi, dcor, Hoeffding, GCMI, tail co-exceedance);
        ``"two-sided"`` for signed correlations.
    directed : bool
        True when ``stat(x, y) != stat(y, x)`` (xi).
    min_obs : int
        Below this many complete pairs the estimate is NaN plus a warning.
    closed_form : callable or None
        ``closed_form(estimate, x, y) -> (p_value, label)``; the p-value is
        ``nan`` when the closed form refuses (e.g. xi with heavy ties).
    default_how : str
        Default per-entity aggregation.
    z_var : callable
        ``z_var(n)`` (vectorised over an array of sample sizes): approximate
        sampling variance of one entity's estimate on the aggregation scale
        (Fisher ``arctanh`` for ``how="fisher"``), used for inverse-variance
        weights and the ``I^2`` heterogeneity statistic. Exact-asymptotic for
        Pearson (``1/(n-3)``), Fieller's constants for Spearman and Kendall,
        the null variance ``2/(5n)`` for xi; for dcor / Hoeffding / GCMI / tail
        statistics it is a documented ``1/(n-3)`` working approximation.
    null_under_indep : float
        Expected value under independence (for tail statistics this is ``q``,
        not 0).
    estimator : str
        Human-readable estimator label reported in the result's
        ``estimator`` field.
    policy : str
        Key into :data:`panelary.depend._null.NULL_POLICY` (defaults to
        ``name``).
    hac : callable or None
        ``hac(estimate, x, y) -> (p_value, label)``: a serial-dependence-robust
        closed form (HAC-studentised), for ``null="hac"`` (``gcm`` only).
    """

    name: str
    rows: Callable[[np.ndarray, np.ndarray], np.ndarray]
    alternative: str
    directed: bool
    min_obs: int
    closed_form: ClosedForm | None
    default_how: str
    z_var: ZVar
    null_under_indep: float = 0.0
    estimator: str = ""
    policy: str = ""
    hac: ClosedForm | None = None

    @property
    def policy_key(self) -> str:
        return self.policy or self.name


def _fisher_var(n: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(np.asarray(n, dtype=np.float64) - 3.0, 1.0)


def _spearman_var(n: np.ndarray) -> np.ndarray:
    return 1.06 / np.maximum(np.asarray(n, dtype=np.float64) - 3.0, 1.0)


def _kendall_var(n: np.ndarray) -> np.ndarray:
    return 0.437 / np.maximum(np.asarray(n, dtype=np.float64) - 4.0, 1.0)


def _xi_var(n: np.ndarray) -> np.ndarray:
    return 0.4 / np.maximum(np.asarray(n, dtype=np.float64), 1.0)


def _cf_pearson(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    return pearson_pvalue(est, x.size), "asymptotic"


def _cf_kendall(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    return kendall_pvalue(x, y)[1], "asymptotic"


def _cf_xi(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    return xi_pvalue(est, x.size, ties=np.concatenate([x, y])), "asymptotic"


def _cf_hoeffding(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    return hoeffding_pvalue(est, x.size), "asymptotic"


def _cf_dcor(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    return dcor_pvalue(dcov2(x, y), method="gamma"), "gamma"


def _cf_gcmi(est: float, x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    return gcmi_pvalue(x, y), "asymptotic"


def _cf_tail(
    est: float, x: np.ndarray, y: np.ndarray, *, q: float, side: str
) -> tuple[float, str]:
    return tail_pvalue(est, x, y, q=q, side=side), "exact"


_BASE: dict[str, Kernel] = {
    "pearson": Kernel(
        "pearson",
        _pearson_rows,
        "two-sided",
        False,
        10,
        _cf_pearson,
        "fisher",
        _fisher_var,
        estimator="pearson",
    ),
    "spearman": Kernel(
        "spearman",
        _spearman_rows,
        "two-sided",
        False,
        10,
        _cf_pearson,
        "fisher",
        _spearman_var,
        estimator="spearman (average ranks)",
    ),
    "kendall": Kernel(
        "kendall",
        _kendall_rows,
        "two-sided",
        False,
        10,
        _cf_kendall,
        "fisher",
        _kendall_var,
        estimator="kendall tau-b",
    ),
    "xi": Kernel(
        "xi",
        _xi_rows,
        "greater",
        True,
        50,
        _cf_xi,
        "fisher",
        _xi_var,
        estimator="chatterjee xi (x-ties by expectation)",
    ),
    "hoeffding": Kernel(
        "hoeffding",
        _hoeffding_rows,
        "greater",
        False,
        50,
        _cf_hoeffding,
        "fisher",
        _fisher_var,
        estimator="hoeffding D (Hollander-Wolfe scale)",
    ),
    "dcor": Kernel(
        "dcor",
        _dcor_rows,
        "greater",
        False,
        30,
        _cf_dcor,
        "fisher",
        _fisher_var,
        estimator="distance correlation (bias-corrected U-statistic)",
    ),
    "gcmi": Kernel(
        "gcmi",
        _gcmi_rows,
        "greater",
        False,
        20,
        _cf_gcmi,
        "mean",
        _fisher_var,
        estimator="gaussian-copula MI (nats, bias-corrected)",
    ),
}

#: Aliases accepted for convenience.
_ALIASES = {
    "hoeffding_d": "hoeffding",
    "distance_correlation": "dcor",
    "tau": "kendall",
}

#: Public method names accepted by :func:`panelary.depend.dependence`.
METHODS = frozenset({*_BASE, "tail_lower", "tail_upper", "tail_dependence", "hsic"})
KERNELS = _BASE


def get_kernel(method: str, *, q: float = 0.05, **options: int) -> Kernel:
    """Return the kernel for ``method`` (``q`` parameterises the tail kernels).

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    method = _ALIASES.get(method, method)
    if method in _BASE:
        return _BASE[method]
    if method in {"tail_lower", "tail_upper"}:
        side = method.split("_")[1]
        return Kernel(
            method,
            partial(_tail_rows, q=q, side=side),
            "greater",
            False,
            max(20, math.ceil(20.0 / q)),
            partial(_cf_tail, q=q, side=side),
            "mean",
            _fisher_var,
            null_under_indep=q,
            estimator=f"co-exceedance / floor(q n), q={q:g}",
        )
    if method == "hsic":
        from panelary.depend._kernel import hsic_kernel

        return hsic_kernel(**options)
    raise ValueError(
        f"unknown dependence `method` {method!r}; expected one of {sorted(METHODS)}."
    )


def min_obs_for(method: str, *, dim: int = 1, q: float = 0.05, k: int = 5) -> int:
    """Minimum complete pairs below which a method reports NaN, not a number.

    ``xi``: 50, ``dcor``: 30, ``hoeffding``: 50, ``gcmi``: ``10 * dim`` (dim =
    total dimension, at least 20), ``mi_ksg``: ``max(100, 10 k + 1)`` (so that
    ``k < n / 10``), tail dependence: ``20 / q``, linear/rank correlations: 10.
    Unbalanced panels are the norm; this gate is what stops a 12-observation
    entity from entering a Fisher average.
    """
    if method == "gcmi":
        return max(20, 10 * int(dim))
    if method == "mi_ksg":
        return max(100, 10 * int(k) + 1)
    if method in {"tail_lower", "tail_upper", "tail_dependence"}:
        return max(20, math.ceil(20.0 / q))
    if method in {"hsic", "transfer_entropy", "gcm"}:
        return 30
    return get_kernel(method).min_obs
