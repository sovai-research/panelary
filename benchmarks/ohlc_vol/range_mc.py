"""Monte Carlo accuracy of the range estimators, and the Yang-Zhang default gate.

Plan 5 §7.1 makes ``yang_zhang`` the default of
:func:`panelary.econ.features.range_volatility` and states the one condition
under which it flips to ``gk_overnight``: *its MSE is lower by more than 5% in
every regime*. This script measures that, through the public function, on
simulated bars:

* log price a random walk with ``M`` intrabar steps (the high and low are the
  extremes of the ``M + 1`` sampled prices, so discrete monitoring bias is
  included, uncorrected -- the default);
* an overnight return carrying a fraction ``f`` of each bar's total variance;
* a drift of ``mu`` bar-standard-deviations per bar (a calibrated equity drift
  is ``|mu| <= 0.1``);
* optionally a Heston-type stochastic variance (mean-reverting square-root
  process, Euler, one variance per bar), in which case the target is the
  window's average true variance.

Each regime draws ``windows`` independent entities of ``n + 1`` bars and scores
the last row's ``n``-bar estimate against the true total variance per bar.
Efficiency is ``MSE(close-to-close sample variance) / MSE(estimator)``.

Run::

    python benchmarks/ohlc_vol/range_mc.py            # the gate grid, ~1 min
    python benchmarks/ohlc_vol/range_mc.py --windows 1000
"""

from __future__ import annotations

import argparse
import itertools
import math
import time

import numpy as np
import polars as pl

from panelary.econ.features import range_volatility

ESTIMATORS = (
    "yang_zhang",
    "gk_overnight",
    "rogers_satchell",
    "garman_klass",
    "parkinson",
    "close_to_close",
)


def simulate(
    n_entities: int,
    n_bars: int,
    m: int,
    *,
    f: float,
    mu: float,
    heston: bool,
    rng: np.random.Generator,
) -> tuple[pl.DataFrame, np.ndarray]:
    """OHLC bars for ``n_entities`` entities; returns (frame, true variance)."""
    if heston:
        # v_t: square-root process around 1 (Euler, reflected at 0): half-life
        # ~10 bars, stationary sd of v about 0.4.
        kappa, theta, xi = 0.07, 1.0, 0.15
        v = np.empty((n_entities, n_bars))
        v[:, 0] = theta + 0.4 * rng.standard_normal(n_entities).clip(-2, 2)
        for t in range(1, n_bars):
            prev = v[:, t - 1]
            v[:, t] = np.abs(
                prev
                + kappa * (theta - prev)
                + xi * np.sqrt(prev) * rng.standard_normal(n_entities)
            )
    else:
        v = np.ones((n_entities, n_bars))
    sd_intra = np.sqrt((1.0 - f) * v)
    sd_over = np.sqrt(f * v)
    rows_o = np.empty((n_entities, n_bars))
    rows_h = np.empty_like(rows_o)
    rows_l = np.empty_like(rows_o)
    rows_c = np.empty_like(rows_o)
    prev_close = np.zeros(n_entities)
    for t in range(n_bars):
        o = prev_close + sd_over[:, t] * rng.standard_normal(n_entities)
        steps = rng.standard_normal((n_entities, m)) * (
            sd_intra[:, t, None] / math.sqrt(m)
        )
        steps += mu * np.sqrt(v[:, t, None]) / m
        path = np.cumsum(steps, axis=1)
        rows_o[:, t] = o
        rows_h[:, t] = o + np.maximum(path.max(axis=1), 0.0)
        rows_l[:, t] = o + np.minimum(path.min(axis=1), 0.0)
        rows_c[:, t] = o + path[:, -1]
        prev_close = rows_c[:, t]
    frame = pl.DataFrame(
        {
            "e": np.repeat(np.arange(n_entities), n_bars),
            "t": np.tile(np.arange(n_bars), n_entities),
            "open": np.exp(rows_o.ravel() + 4.0),
            "high": np.exp(rows_h.ravel() + 4.0),
            "low": np.exp(rows_l.ravel() + 4.0),
            "close": np.exp(rows_c.ravel() + 4.0),
        }
    )
    return frame, v


def run_regime(
    *, m: int, f: float, mu: float, heston: bool, n: int, windows: int, seed: int
) -> dict[str, tuple[float, float]]:
    """``{estimator: (bias, mse)}`` of the ``n``-bar variance estimates."""
    rng = np.random.default_rng(seed)
    parts: list[pl.DataFrame] = []
    truths: list[np.ndarray] = []
    chunk = max(1, 2_000_000 // (m * (n + 1)))
    done = 0
    while done < windows:
        k = min(chunk, windows - done)
        frame, v = simulate(k, n + 1, m, f=f, mu=mu, heston=heston, rng=rng)
        frame = frame.with_columns(pl.col("e") + done)
        for method in ESTIMATORS:
            frame = range_volatility(
                frame, entity="e", time="t", method=method, window=n, output="var"
            )
        parts.append(frame.filter(pl.col("t") == n))
        truths.append(v[:, 1:].mean(axis=1))
        done += k
    last = pl.concat(parts)
    truth = np.concatenate(truths)
    out = {}
    for method in ESTIMATORS:
        est = last[f"var_{method}_{n}"].to_numpy()
        err = est / truth - 1.0
        out[method] = (float(np.mean(err)), float(np.mean(err**2)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--windows", type=int, default=3000)
    parser.add_argument("--n", type=int, default=21)
    args = parser.parse_args()

    grid = [
        {"m": m, "f": f, "mu": mu, "heston": False}
        for m, f, mu in itertools.product((78, 390, 5000), (0.0, 0.25), (0.0, 0.1))
    ] + [{"m": 390, "f": 0.25, "mu": 0.05, "heston": True}]
    flips = []
    t0 = time.perf_counter()
    header = f"{'M':>5} {'f':>5} {'mu':>5} {'SV':>3} | " + " | ".join(
        f"{e[:12]:>12}" for e in ESTIMATORS
    )
    print("bias / MSE-efficiency vs the n-bar close-to-close sample variance")
    print(header)
    for i, regime in enumerate(grid):
        res = run_regime(n=args.n, windows=args.windows, seed=1000 + i, **regime)
        base = res["close_to_close"][1]
        cells = " | ".join(
            f"{res[e][0]:+6.1%}/{base / res[e][1]:5.2f}" for e in ESTIMATORS
        )
        print(
            f"{regime['m']:>5} {regime['f']:>5} {regime['mu']:>5} "
            f"{'y' if regime['heston'] else 'n':>3} | {cells}",
            flush=True,
        )
        ratio = res["gk_overnight"][1] / res["yang_zhang"][1]
        flips.append(ratio < 0.95)
    decision = "gk_overnight" if all(flips) else "yang_zhang"
    print(
        f"\nGKYZ MSE < 0.95 x YZ MSE in {sum(flips)}/{len(flips)} regimes -> "
        f"default stays {decision!r}  [{time.perf_counter() - t0:.0f}s]"
    )


if __name__ == "__main__":
    main()
