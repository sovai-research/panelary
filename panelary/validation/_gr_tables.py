"""Giacomini-Rossi (2010) critical values: shipped table, lookup, verification.

The fluctuation test and the one-time-reversal test of Giacomini & Rossi (2010)
have non-standard limits:

* fluctuation (two-sided): ``sup_{r in [mu, 1]} |B(r) - B(r - mu)| / sqrt(mu)``,
  ``B`` a standard Brownian motion, ``mu`` the window as a share of the sample;
* one-time reversal: ``W(1)^2 + sup_{r in [trim, 1 - trim]} BB(r)^2 / (r (1 - r))``,
  ``BB(r) = W(r) - r W(1)`` a Brownian bridge (independent of ``W(1)``).

Provenance (read before trusting a digit)
-----------------------------------------
The published tables of GR (2010) were **not** available when this module was
written, so no value here is copied from the paper. Every number is simulated
by :func:`simulate_gr_quantiles` with the settings in :data:`TABLE_SPEC`: Brownian
paths on a grid of ``n_steps`` points, the same paths subsampled to
``n_steps / 4`` points, and a Richardson step that removes the leading
``O(1/sqrt(n))`` discretisation bias of a discretely monitored supremum
(``k_inf = k_fine + (k_fine - k_coarse)``, because ``sqrt(4) - 1 = 1``). The
offline generator is ``benchmarks/gr_critical_values.py``; :func:`verify_gr_tables`
re-simulates a coarse table and reports the deviation. The Monte Carlo
standard error at the 5 % level is in :data:`TABLE_SPEC` (0.0044). The
discretisation correction it removes is 0.006-0.019 at 5 %, in line with the
theoretical ``0.5826 sqrt(2 / (mu n))`` for a discretely monitored supremum.

Two cross-checks: the Brownian-bridge part of the reversal statistic
(``sup BB^2 / (r(1-r))`` with 15 % trimming) reproduces Andrews' (1993, 2003)
sup-LM critical values for one parameter (8.88 against 8.85 at 5 %); and the
5 % fluctuation value near ``mu = 0.4`` (2.955) sits above the 2.89 quoted in
Rossi's Stata implementation note, which is what a supremum over a grid of a
few hundred points gives (simulated here: 2.875 at ``P = 200``, 2.91 at
``P = 500``). These are **continuous-limit** values: for a finite ``P`` the
discrete supremum is smaller, so the test is slightly conservative (measured
size about 4 % at nominal 5 % with ``P = 500``; see the tests).

Stored: upper-tail quantiles ``k(p)`` with ``P(stat > k(p)) = p`` on the grid
:data:`PROBS`, for ``mu`` in :data:`MU_GRID` and ``trim`` in :data:`TRIM_GRID`.
p-values interpolate ``log p`` linearly in ``k``, and are clipped (with a
warning from the caller) outside ``[PROBS[0], PROBS[-1]]``.

Nothing heavy runs at import: the table is a literal, turned into arrays on
first use.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

__all__ = [
    "MU_GRID",
    "PROBS",
    "TABLE_SPEC",
    "TRIM_GRID",
    "fluctuation_critical_value",
    "fluctuation_pvalue",
    "reversal_critical_value",
    "reversal_pvalue",
    "simulate_gr_quantiles",
    "verify_gr_tables",
]

MU_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TRIM_GRID: tuple[float, ...] = (0.15, 0.20)
#: Upper-tail probabilities of the stored quantiles.
PROBS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
)  # fmt: skip


# --------------------------------------------------------------------------- #
# Simulation (offline generator and verification)
# --------------------------------------------------------------------------- #
def simulate_gr_quantiles(
    *,
    n_reps: int,
    n_steps: int,
    seed: int,
    probs: tuple[float, ...] = PROBS,
    chunk: int = 250,
) -> dict[str, np.ndarray]:
    """Simulate the GR null quantiles on a fine grid and a 4x coarser subgrid.

    Returns ``fluctuation`` ``(len(MU_GRID), len(probs))`` and ``reversal``
    ``(len(TRIM_GRID), len(probs))`` extrapolated quantiles, plus the raw
    ``*_fine`` and ``*_coarse`` tables.
    """
    if n_steps % 40 != 0:
        raise ValueError("`n_steps` must be a multiple of 40 (4 x the mu grid step).")
    rng = np.random.default_rng(seed)
    n_c = n_steps // 4
    fl = {g: np.empty((len(MU_GRID), n_reps)) for g in ("fine", "coarse")}
    rv = {g: np.empty((len(TRIM_GRID), n_reps)) for g in ("fine", "coarse")}
    done = 0
    while done < n_reps:
        c = min(chunk, n_reps - done)
        w = np.zeros((c, n_steps + 1))
        np.cumsum(rng.standard_normal((c, n_steps)), axis=1, out=w[:, 1:])
        w /= np.sqrt(n_steps)
        for grid, path, n in (("fine", w, n_steps), ("coarse", w[:, ::4], n_c)):
            for i, mu in enumerate(MU_GRID):
                m = round(mu * n)
                d = path[:, m:] - path[:, : n + 1 - m]
                fl[grid][i, done : done + c] = np.abs(d).max(axis=1) / np.sqrt(mu)
            w1 = path[:, -1]
            r = np.arange(n + 1) / n
            for i, trim in enumerate(TRIM_GRID):
                lo, hi = round(trim * n), round((1.0 - trim) * n)
                rr = r[lo : hi + 1]
                bb = path[:, lo : hi + 1] - rr * w1[:, None]
                rv[grid][i, done : done + c] = w1 * w1 + (
                    bb * bb / (rr * (1.0 - rr))
                ).max(axis=1)
        done += c
    q = 1.0 - np.asarray(probs)
    out: dict[str, np.ndarray] = {}
    for name, store in (("fluctuation", fl), ("reversal", rv)):
        fine = np.quantile(store["fine"], q, axis=1).T
        coarse = np.quantile(store["coarse"], q, axis=1).T
        out[f"{name}_fine"] = fine
        out[f"{name}_coarse"] = coarse
        out[name] = fine + (fine - coarse)
    return out


# --------------------------------------------------------------------------- #
# Shipped table
# --------------------------------------------------------------------------- #
#: Settings that produced :data:`_FLUCTUATION` / :data:`_REVERSAL`.
TABLE_SPEC: dict[str, Any] = {
    "n_reps": 200000,
    "n_steps": 20000,
    "seed": 20260929,
    "mc_se_5pct": 0.0044,
    "generator": "benchmarks/gr_critical_values.py",
}

# Rows follow MU_GRID / TRIM_GRID, columns follow PROBS.
# fmt: off
_FLUCTUATION: tuple[tuple[float, ...], ...] = (
    (4.1766, 4.0029, 3.7405, 3.5290, 3.3961, 3.2983, 3.1544, 3.0419, 2.9489, 2.8695, 2.7307, 2.6050, 2.4861, 2.3664, 2.2339, 2.0637, 1.9337,),
    (3.9747, 3.7807, 3.5034, 3.2755, 3.1359, 3.0286, 2.8674, 2.7441, 2.6436, 2.5537, 2.4006, 2.2629, 2.1315, 2.0007, 1.8547, 1.6713, 1.5322,),
    (3.8134, 3.6252, 3.3355, 3.1004, 2.9515, 2.8398, 2.6696, 2.5406, 2.4325, 2.3387, 2.1731, 2.0280, 1.8897, 1.7510, 1.6001, 1.4127, 1.2783,),
    (3.7053, 3.4992, 3.2040, 2.9549, 2.7988, 2.6841, 2.5067, 2.3707, 2.2580, 2.1584, 1.9876, 1.8375, 1.6958, 1.5552, 1.4031, 1.2205, 1.0925,),
    (3.6059, 3.3844, 3.0765, 2.8244, 2.6658, 2.5460, 2.3612, 2.2209, 2.1039, 2.0027, 1.8269, 1.6712, 1.5284, 1.3886, 1.2408, 1.0639, 0.9428,),
    (3.5073, 3.2776, 2.9611, 2.7077, 2.5408, 2.4156, 2.2261, 2.0814, 1.9593, 1.8528, 1.6722, 1.5160, 1.3734, 1.2344, 1.0898, 0.9221, 0.8090,),
    (3.3886, 3.1615, 2.8499, 2.5822, 2.4151, 2.2887, 2.0908, 1.9398, 1.8128, 1.7057, 1.5224, 1.3642, 1.2215, 1.0851, 0.9441, 0.7860, 0.6809,),
    (3.2876, 3.0506, 2.7273, 2.4575, 2.2831, 2.1486, 1.9521, 1.7977, 1.6698, 1.5589, 1.3728, 1.2111, 1.0673, 0.9312, 0.7949, 0.6453, 0.5506,),
    (3.1371, 2.9084, 2.5766, 2.3017, 2.1280, 1.9931, 1.7890, 1.6331, 1.5054, 1.3938, 1.2008, 1.0366, 0.8902, 0.7553, 0.6235, 0.4865, 0.4049,),
)
_REVERSAL: tuple[tuple[float, ...], ...] = (
    (15.9617, 14.3860, 12.1423, 10.4981, 9.4981, 8.7792, 7.7775, 7.0248, 6.4303, 5.9173, 5.0872, 4.4007, 3.7997, 3.2374, 2.6764, 2.0631, 1.6626,),
    (15.5260, 13.8838, 11.6994, 10.0798, 9.1053, 8.4083, 7.3966, 6.6615, 6.0581, 5.5606, 4.7529, 4.0806, 3.4911, 2.9461, 2.4136, 1.8281, 1.4626,),
)
# fmt: on

_CACHE: dict[str, np.ndarray] = {}


def _table(kind: str) -> np.ndarray:
    if kind not in _CACHE:
        src = _FLUCTUATION if kind == "fluctuation" else _REVERSAL
        _CACHE[kind] = np.asarray(src, dtype=np.float64)
    return _CACHE[kind]


def _row(kind: str, key: float) -> np.ndarray:
    grid = MU_GRID if kind == "fluctuation" else TRIM_GRID
    tab = _table(kind)
    for i, g in enumerate(grid):
        if abs(g - key) < 1e-9:
            return tab[i]
    if kind == "fluctuation" and MU_GRID[0] <= key <= MU_GRID[-1]:
        # linear interpolation in mu between tabulated rows (monitor mode)
        return np.array(
            [np.interp(key, MU_GRID, tab[:, j]) for j in range(tab.shape[1])]
        )
    raise ValueError(
        f"{'window share' if kind == 'fluctuation' else 'trim'} {key} is not tabulated; "
        f"choose one of {grid}."
    )


def _critical(kind: str, key: float, alpha: float) -> float:
    row = _row(kind, key)
    p = np.asarray(PROBS)
    if not (p[0] <= alpha <= p[-1]):
        raise ValueError(f"`alpha` must lie in [{p[0]}, {p[-1]}], got {alpha}.")
    return float(np.interp(np.log(alpha), np.log(p), row))


def _pvalue(kind: str, key: float, stat: np.ndarray) -> tuple[np.ndarray, bool]:
    """Interpolated upper-tail p-values; flag whether any was clipped to the grid."""
    row = _row(kind, key)  # decreasing in p, i.e. increasing along reversed grid
    logp = np.log(np.asarray(PROBS))
    s = np.asarray(stat, dtype=np.float64)
    out = np.exp(np.interp(s, row[::-1], logp[::-1]))
    clipped = bool(np.any((s > row[0]) | (s < row[-1])))
    return np.where(np.isfinite(s), out, np.nan), clipped


def fluctuation_critical_value(mu: float, alpha: float = 0.05) -> float:
    """Two-sided fluctuation-test critical value ``k_alpha(mu)``."""
    return _critical("fluctuation", mu, alpha)


def fluctuation_pvalue(stat: np.ndarray, mu: float) -> tuple[np.ndarray, bool]:
    """Table-interpolated p-values of ``max_t |F_t|``; second item flags clipping."""
    return _pvalue("fluctuation", mu, stat)


def reversal_critical_value(trim: float, alpha: float = 0.05) -> float:
    """One-time-reversal critical value at ``trim`` in :data:`TRIM_GRID`."""
    return _critical("reversal", trim, alpha)


def reversal_pvalue(stat: np.ndarray, trim: float) -> tuple[np.ndarray, bool]:
    """Table-interpolated p-values of the one-time-reversal statistic."""
    return _pvalue("reversal", trim, stat)


def verify_gr_tables(*, quick: bool = True, seed: int = 12345) -> dict[str, float]:
    """Re-simulate the tables and report the deviation from the shipped values.

    ``quick=True`` (about 1 s): 20,000 paths of 2,000 steps, comparing the 5 %
    and 10 % critical values against tolerances of roughly five Monte Carlo
    standard errors (0.08 for the fluctuation statistic, 0.4 for the
    chi-square-scale reversal statistic). ``quick=False`` repeats the shipped
    simulation (:data:`TABLE_SPEC`, about 1-2 minutes) and also compares 1 %,
    with tolerances 0.02 / 0.1. Warns when a tolerance is exceeded.
    """
    levels: tuple[float, ...]
    if quick:
        reps, steps, levels = 20000, 2000, (0.05, 0.10)
        tol_f, tol_r = 0.08, 0.4
    else:
        reps, steps = int(TABLE_SPEC["n_reps"]), int(TABLE_SPEC["n_steps"])
        levels, tol_f, tol_r = (0.01, 0.05, 0.10), 0.02, 0.1
    fresh = simulate_gr_quantiles(n_reps=reps, n_steps=steps, seed=seed)
    cols = [PROBS.index(a) for a in levels]
    dev_f = np.abs(fresh["fluctuation"][:, cols] - _table("fluctuation")[:, cols])
    dev_r = np.abs(fresh["reversal"][:, cols] - _table("reversal")[:, cols])
    out = {
        "fluctuation_max_abs_dev": float(dev_f.max()),
        "reversal_max_abs_dev": float(dev_r.max()),
        "fluctuation_tolerance": tol_f,
        "reversal_tolerance": tol_r,
    }
    if out["fluctuation_max_abs_dev"] > tol_f or out["reversal_max_abs_dev"] > tol_r:
        warnings.warn(
            f"GR critical-value table deviates from a fresh simulation beyond tolerance: "
            f"{out}. Regenerate with benchmarks/gr_critical_values.py.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out
