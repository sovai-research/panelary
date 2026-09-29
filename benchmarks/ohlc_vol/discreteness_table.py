"""Regenerate the discrete-monitoring bias table of the range estimators.

A bar's high and low are the extremes of the prices that were actually
*observed*, not of the continuous path, so they understate the true range and
every range estimator is biased down. For a driftless Gaussian random walk of
``M`` increments per bar (``M + 1`` observed prices, the open included) and
bar variance ``sigma^2``, the bias is a constant factor

    b(M) = E[estimate] / sigma^2,

one per estimator. ``panelary._internal._ohlc`` stores the table this script
prints and divides the Parkinson / Garman-Klass / Rogers-Satchell per-bar terms
by it when ``range_volatility(discrete_bars=M)`` is passed.

What is exact and what is simulated
-----------------------------------
Write ``S_k`` for the walk (``S_0 = 0`` is the open), ``H = max_k S_k``,
``L = min_k S_k`` and ``C = S_M``; take ``sigma^2 = 1`` so each increment has
variance ``1/M``.

* **Rogers-Satchell is exact.** Its expected per-bar term is
  ``2 E[H (H - C)]`` by the reflection symmetry ``S -> -S``. The Baxter-Spitzer
  identity factorises the generating function of ``(H, H - C)``, which gives

      E[H (H - C)] = sum_{j + k <= M} E[S_j^+] E[S_k^+] / (j k),
      b_RS(M)      = (1 / (pi M)) * sum_{j, k >= 1, j + k <= M} (j k)^(-1/2),

  an ``O(M)`` sum (-> 1 as ``M -> oo``: the Dirichlet integral over the simplex
  is ``pi``). The same identity gives ``E[H]`` and ``E[H^2]`` exactly.
* **Parkinson and Garman-Klass are simulated.** Both are functions of the one
  moment ``R2(M) = E[(H - L)^2]``, whose cross term ``E[H L]`` has no closed
  form for a discrete walk:
  ``b_P = R2 / (4 ln 2)`` and ``b_GK = R2 / 2 - (2 ln 2 - 1)``. ``R2`` is
  estimated by Monte Carlo with four control variates whose means are known
  exactly -- the Rogers-Satchell term, ``H^2 + L^2``, ``H - L`` and ``C^2`` --
  which cuts the standard error several-fold at no bias beyond ``O(1/n)``.

Run (about ten minutes on one core; it is deterministic)::

    python benchmarks/ohlc_vol/discreteness_table.py            # 1e6 bars per M
    python benchmarks/ohlc_vol/discreteness_table.py --bars 1e5 # quick look

and paste the printed tuples into ``panelary/_internal/_ohlc.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

#: Log-spaced monitoring grid: 2 (the smallest ``M`` at which the RS factor is
#: non-zero) to 23 400 (one-second sampling of a 6.5-hour session).
GRID: tuple[int, ...] = (
    2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 39, 50, 65, 78, 100, 130, 160,
    195, 260, 390, 520, 780, 1170, 1560, 2340, 3900, 7800, 11700, 23400,
)  # fmt: skip

SEED = 20260929
LN2 = math.log(2.0)


def rs_factor(m: int) -> float:
    """Exact ``b_RS(M)`` (see the module docstring)."""
    a = np.arange(1, m + 1, dtype=float) ** -0.5
    prefix = np.concatenate([[0.0], np.cumsum(a)])
    # sum_{j=1}^{M-1} a_j * sum_{k=1}^{M-j} a_k
    j = np.arange(1, m)
    return float(np.sum(a[j - 1] * prefix[m - j]) / (math.pi * m))


def exact_moments(m: int) -> dict[str, float]:
    """Exact ``E[RS]``, ``E[H^2 + L^2]``, ``E[H - L]`` and ``E[C^2]``."""
    k = np.arange(1, m + 1, dtype=float)
    e_h = float(np.sum(np.sqrt(k / m) / math.sqrt(2.0 * math.pi) / k))
    b_rs = rs_factor(m)
    # E[H^2] = sum_k E[(S_k^+)^2] / k + E[H (H - C)] = 1/2 + b_RS / 2
    e_h2 = 0.5 + 0.5 * b_rs
    return {"rs": b_rs, "h2l2": 2.0 * e_h2, "range": 2.0 * e_h, "c2": 1.0}


def simulate(m: int, n_bars: int, rng: np.random.Generator) -> tuple[float, float]:
    """Control-variate Monte Carlo estimate of ``R2(M)`` and its standard error."""
    mu = exact_moments(m)
    mu_x = np.array([mu["rs"], mu["h2l2"], mu["range"], mu["c2"]])
    # Accumulate the sufficient statistics of the regression Y ~ 1 + X.
    zz = np.zeros((6, 6))
    rows_per_chunk = max(1, 10_000_000 // m)
    done = 0
    scale = 1.0 / math.sqrt(m)
    while done < n_bars:
        rows = min(rows_per_chunk, n_bars - done)
        x = rng.standard_normal((rows, m))
        np.cumsum(x, axis=1, out=x)
        h = np.maximum(x.max(axis=1), 0.0) * scale
        lo = np.minimum(x.min(axis=1), 0.0) * scale
        c = x[:, -1] * scale
        y = (h - lo) ** 2
        feats = np.column_stack(
            [
                np.ones(rows),
                h * (h - c) + lo * (lo - c),
                h * h + lo * lo,
                h - lo,
                c * c,
                y,
            ]
        )
        zz += feats.T @ feats
        done += rows
    n = float(n_bars)
    xtx = zz[:5, :5]
    xty = zz[:5, 5]
    beta = np.linalg.solve(xtx, xty)
    means = zz[0, :5] / n  # [1, mean(X)]
    mean_y = zz[0, 5] / n
    est = mean_y - float(beta[1:] @ (means[1:] - mu_x))
    # residual variance of the regression, for the standard error
    ssr = zz[5, 5] - 2.0 * float(beta @ xty) + float(beta @ xtx @ beta)
    se = math.sqrt(max(ssr, 0.0) / (n - 5.0) / n)
    return est, se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bars", type=float, default=1e6, help="bars per M")
    parser.add_argument("--json", type=str, default=None, help="also write JSON here")
    args = parser.parse_args()
    n_bars = int(args.bars)

    children = np.random.SeedSequence(SEED).spawn(len(GRID))
    rows = []
    t0 = time.perf_counter()
    for m, child in zip(GRID, children, strict=True):
        r2, se = simulate(m, n_bars, np.random.default_rng(child))
        b_rs = rs_factor(m)
        row = {
            "M": m,
            "parkinson": r2 / (4.0 * LN2),
            "parkinson_se": se / (4.0 * LN2),
            "garman_klass": 0.5 * r2 - (2.0 * LN2 - 1.0),
            "garman_klass_se": 0.5 * se,
            "rogers_satchell": b_rs,
            "bars": n_bars,
        }
        rows.append(row)
        print(
            f"M={m:>6}  P={row['parkinson']:.6f} (se {row['parkinson_se']:.1e})  "
            f"GK={row['garman_klass']:.6f} (se {row['garman_klass_se']:.1e})  "
            f"RS={b_rs:.6f} (exact)  [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )
    print("\n_DISCRETE_M =", tuple(r["M"] for r in rows))
    for key in ("parkinson", "garman_klass", "rogers_satchell"):
        vals = ", ".join(f"{r[key]:.6f}" for r in rows)
        print(f"_DISCRETE_{key.upper()} = ({vals})")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"seed": SEED, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
