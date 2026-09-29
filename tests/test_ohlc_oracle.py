"""Parity of the OHLC estimators with independent reference implementations.

Range volatility
----------------
The plan asks for values stored from R's ``TTR::volatility`` (GPL-2, so
reference values only). R is not available where this suite was written, so no
TTR output is stored. Instead :func:`_ttr_volatility` below is a direct,
per-window transcription of the formulas ``TTR::volatility`` documents for its
``calc`` options -- written from those formulas and the original papers, not
from TTR's source, and structurally unlike the implementation under test (a
Python loop over explicit windows versus staged polars rolling moments). The
match is asserted at ``rtol = 1e-12``. Replacing it with stored TTR output is a
one-line change once R is available; the formulas are the ones TTR evaluates:

* ``close``: ``runSD(ROC(C), n) * sqrt(N)`` (log returns, sample sd);
* ``parkinson``: ``sqrt(N / (4 n log 2) * runSum(log(H/L)^2, n))``;
* ``garman.klass``: ``sqrt(N/n * runSum(.5 log(H/L)^2) - N/n * runSum((2 log 2 - 1) log(C/O)^2))``;
* ``rogers.satchell``: ``sqrt(N/n * runSum(log(H/C) log(H/O) + log(L/C) log(L/O), n))``;
* ``gk.yz``: ``sqrt(N/n * runSum(log(O/C1)^2 + .5 log(H/L)^2 - (2 log 2 - 1) log(C/O)^2, n))``;
* ``yang.zhang``: ``sqrt(N runVar(log(O/C1), n) + k N runVar(log(C/O), n) + (1 - k) RS^2)``
  with ``k = (alpha - 1) / (alpha + (n + 1) / (n - 1))``, ``alpha = 1.34``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from panelary.econ.features import range_volatility

LN2 = math.log(2.0)


def _ttr_volatility(o, h, lo, c, calc: str, n: int, N: float) -> np.ndarray:
    """Per-window transcription of ``TTR::volatility`` (complete data only)."""
    o, h, lo, c = (np.asarray(x, dtype=float) for x in (o, h, lo, c))
    out = np.full(len(c), np.nan)
    c1 = np.r_[np.nan, c[:-1]]
    for t in range(len(c)):
        s = slice(t - n + 1, t + 1)
        if t - n + 1 < 0:
            continue
        op, hi, low, cl, cl1 = o[s], h[s], lo[s], c[s], c1[s]
        if calc == "close":
            if t - n < 0:
                continue  # ROC's first value is NA
            val = np.std(np.log(cl / cl1), ddof=1) * math.sqrt(N)
        elif calc == "parkinson":
            val = math.sqrt(N / (4 * n * LN2) * np.sum(np.log(hi / low) ** 2))
        elif calc == "garman.klass":
            val = math.sqrt(
                N / n * np.sum(0.5 * np.log(hi / low) ** 2)
                - N / n * np.sum((2 * LN2 - 1) * np.log(cl / op) ** 2)
            )
        elif calc == "rogers.satchell":
            val = math.sqrt(
                N
                / n
                * np.sum(
                    np.log(hi / cl) * np.log(hi / op)
                    + np.log(low / cl) * np.log(low / op)
                )
            )
        elif calc == "gk.yz":
            if t - n < 0:
                continue
            val = math.sqrt(
                N
                / n
                * np.sum(
                    np.log(op / cl1) ** 2
                    + 0.5 * np.log(hi / low) ** 2
                    - (2 * LN2 - 1) * np.log(cl / op) ** 2
                )
            )
        elif calc == "yang.zhang":
            if t - n < 0:
                continue
            k = (1.34 - 1) / (1.34 + (n + 1) / (n - 1))
            s2o = N * np.var(np.log(op / cl1), ddof=1)
            s2c = N * np.var(np.log(cl / op), ddof=1)
            rs = (
                N
                / n
                * np.sum(
                    np.log(hi / cl) * np.log(hi / op)
                    + np.log(low / cl) * np.log(low / op)
                )
            )
            val = math.sqrt(s2o + k * s2c + (1 - k) * rs)
        else:  # pragma: no cover
            raise ValueError(calc)
        out[t] = val
    return out


_CALC = {
    "close_to_close": "close",
    "parkinson": "parkinson",
    "garman_klass": "garman.klass",
    "rogers_satchell": "rogers.satchell",
    "gk_overnight": "gk.yz",
    "yang_zhang": "yang.zhang",
}


def _synthetic_ohlc(seed: int = 20260929, n_entities: int = 3, n: int = 120):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        prev = 10.0 ** (e + 1)
        for t in range(n):
            o = prev * math.exp(rng.normal(0.0, 0.006))
            path = o * np.exp(np.cumsum(rng.normal(2e-4, 0.015 / 8.0, 64)))
            rows.append(
                (e, t, o, max(o, path.max()), min(o, path.min()), float(path[-1]))
            )
            prev = float(path[-1])
    return pl.DataFrame(
        rows, schema=["e", "t", "open", "high", "low", "close"], orient="row"
    )


@pytest.mark.parametrize("method", list(_CALC))
@pytest.mark.parametrize("window", [2, 10, 21])
def test_range_volatility_matches_the_ttr_formulas(method, window) -> None:
    df = _synthetic_ohlc()
    out = range_volatility(
        df, entity="e", time="t", method=method, window=window, periods_per_year=260
    )
    col = f"vol_{method}_{window}"
    for e in range(3):
        sub = out.filter(pl.col("e") == e)
        ref = _ttr_volatility(
            *(sub[c].to_numpy() for c in ("open", "high", "low", "close")),
            _CALC[method],
            window,
            260.0,
        )
        got = sub[col].to_numpy().astype(float)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        # atol: polars' streaming rolling variance loses a few more digits than a
        # two-pass variance when the window's returns nearly coincide (window 2)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-13, equal_nan=True)
