"""Offline generator for the Giacomini-Rossi (2010) critical-value table.

Simulates the null limits of the fluctuation and one-time-reversal tests with
:func:`panelary.validation._gr_tables.simulate_gr_quantiles` and prints the
Python literals (``TABLE_SPEC``, ``_FLUCTUATION``, ``_REVERSAL``) to paste into
``panelary/validation/_gr_tables.py``. The shipped table was produced with the
defaults below (200,000 paths of 20,000 steps, seed 20260929; about 2 minutes on
an Apple-Silicon laptop).

Run::

    python benchmarks/gr_critical_values.py
    python benchmarks/gr_critical_values.py --reps 20000 --steps 4000   # quick look
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from panelary.validation._gr_tables import PROBS, simulate_gr_quantiles


def _literal(name: str, table: np.ndarray) -> str:
    rows = ",\n".join(
        "    (" + ", ".join(f"{v:.4f}" for v in row) + ",)" for row in np.asarray(table)
    )
    return f"{name}: tuple[tuple[float, ...], ...] = (\n{rows},\n)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    t0 = time.perf_counter()
    out = simulate_gr_quantiles(n_reps=args.reps, n_steps=args.steps, seed=args.seed)
    elapsed = time.perf_counter() - t0
    # MC standard error of a p-quantile: sqrt(p(1-p)/R) / f(q); the density at the
    # 5 % quantile is estimated from the neighbouring tabulated quantiles.
    j = PROBS.index(0.05)
    dk = (
        out["fluctuation"][:, PROBS.index(0.025)]
        - out["fluctuation"][:, PROBS.index(0.075)]
    )
    dens = 0.05 / np.maximum(dk, 1e-9)
    se = float(np.max(math.sqrt(0.05 * 0.95 / args.reps) / dens))
    print(f"# simulated in {elapsed:.1f} s")
    print("TABLE_SPEC: dict[str, Any] = {")
    print(f'    "n_reps": {args.reps},')
    print(f'    "n_steps": {args.steps},')
    print(f'    "seed": {args.seed},')
    print(f'    "mc_se_5pct": {se:.4f},')
    print('    "generator": "benchmarks/gr_critical_values.py",')
    print("}")
    print(_literal("_FLUCTUATION", out["fluctuation"]))
    print(_literal("_REVERSAL", out["reversal"]))
    print(
        "# 5 % critical values (fluctuation, by mu):",
        np.round(out["fluctuation"][:, j], 3),
    )
    print(
        "# 5 % critical values (reversal, by trim):", np.round(out["reversal"][:, j], 3)
    )
    print(
        "# discretisation correction (extrapolated - fine), 5 %:",
        np.round(out["fluctuation"][:, j] - out["fluctuation_fine"][:, j], 4),
    )


if __name__ == "__main__":
    main()
