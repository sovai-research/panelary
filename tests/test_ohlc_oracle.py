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


# --------------------------------------------------------------------------- #
# EDGE (Ardia, Guidotti & Kroencke 2024) vs the reference `bidask` package
# --------------------------------------------------------------------------- #
# The reference is the MIT-licensed `bidask` package (v2.1.0). Its published
# test data -- the first 2000 rows of `pseudocode/ohlc.csv` (complete) and
# `ohlc-miss.csv` (with missing prices) -- are vendored in tests/data/bidask/
# (MIT notice there), with values `bidask.edge` computed from them. Three layers:
#
# 1. `_reference_edge` below transcribes the per-window algorithm of the
#    package's published pseudocode (MIT), and is pinned to the package by the
#    stored values;
# 2. our staged rolling implementation matches that transcription at EVERY
#    index, for windows 3 to 252, with and without missing data;
# 3. with `bidask` installed (`importorskip`), the same every-index parity is
#    run against the package itself.
#
# Parity is asserted on s^2, not s: sqrt amplifies near zero (|ds| ~ |ds^2|/2s).
# Windows where one moment family has <= 1 valid pair are excluded: there the
# reference's weighting is decided by floating-point residue (v = 2e-25 vs 0),
# and ours returns the exact-arithmetic value (plan 5 section 1d).
import gzip  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

from panelary.econ.features import ohlc_spread  # noqa: E402

_BIDASK = pathlib.Path(__file__).parent / "data" / "bidask"


def _load_bidask(name: str) -> pl.DataFrame:
    with gzip.open(_BIDASK / name, "rb") as fh:
        raw = pl.read_csv(fh.read())
    return raw.rename(str.lower).with_columns(
        e=pl.lit("x"), t=pl.int_range(pl.len(), dtype=pl.Int64)
    )


def _reference_edge(o, h, lo, c):
    """Per-window EDGE, transcribed from bidask's published pseudocode (MIT).

    Returns ``(s2, n1, n2)``: the signed squared spread and the valid-pair
    counts of the two moment families.
    """
    o, h, lo, c = (np.log(np.asarray(x, dtype=float)) for x in (o, h, lo, c))
    if len(o) < 3:
        return np.nan, 0, 0
    m = (h + lo) / 2.0
    h1, l1, c1, m1 = h[:-1], lo[:-1], c[:-1], m[:-1]
    o, h, lo, c, m = o[1:], h[1:], lo[1:], c[1:], m[1:]
    r1, r2, r3, r4, r5 = m - o, o - m1, m - c1, c1 - m1, o - c1
    nan = np.isnan
    tau = np.where(nan(h) | nan(lo) | nan(c1), np.nan, (h != lo) | (lo != c1))
    po1 = tau * np.where(nan(o) | nan(h), np.nan, o != h)
    po2 = tau * np.where(nan(o) | nan(lo), np.nan, o != lo)
    pc1 = tau * np.where(nan(c1) | nan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(nan(c1) | nan(l1), np.nan, c1 != l1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pt = np.nanmean(tau)
        po = np.nanmean(po1) + np.nanmean(po2)
        pc = np.nanmean(pc1) + np.nanmean(pc2)
        if np.nansum(tau) < 2 or po == 0 or pc == 0:
            return np.nan, 0, 0
        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau
        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4
        e1, e2 = np.nanmean(x1), np.nanmean(x2)
        v1 = np.nanmean(x1**2) - e1**2
        v2 = np.nanmean(x2**2) - e2**2
    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    return s2, int(np.sum(~nan(x1))), int(np.sum(~nan(x2)))


def _prices(df: pl.DataFrame):
    return [df[c].to_numpy().astype(float) for c in ("open", "high", "low", "close")]


def test_reference_transcription_reproduces_the_stored_bidask_values() -> None:
    stored = json.loads((_BIDASK / "edge_reference_values.json").read_text())
    assert stored["provenance"]["version"] == "2.1.0"
    frames = {
        name: _prices(_load_bidask(name))
        for name in ("ohlc_2000.csv.gz", "ohlc-miss_2000.csv.gz")
    }
    for row in stored["values"]:
        a, b = row["start"], row["stop"]
        s2, _, _ = _reference_edge(*(x[a:b] for x in frames[row["file"]]))
        s = row["signed_spread"]
        assert s2 == pytest.approx(math.copysign(s * s, s), rel=1e-12, abs=1e-18)


def test_edge_reproduces_the_stored_bidask_values() -> None:
    stored = json.loads((_BIDASK / "edge_reference_values.json").read_text())
    for row in stored["values"]:
        df = _load_bidask(row["file"])
        a, b = row["start"], row["stop"]
        w = b - a
        out = ohlc_spread(
            df.slice(a, w),
            entity="e",
            time="t",
            window=w,
            min_periods=2,
            negative="signed",
            invalid="keep",
        )
        got = out[f"spread_edge_{w}"][-1]
        assert got == pytest.approx(row["signed_spread"], rel=1e-9, abs=1e-12)


def _edge_parity(name: str, window: int, oracle) -> None:
    df = _load_bidask(name)
    gappy = "miss" in name
    out = ohlc_spread(
        df,
        entity="e",
        time="t",
        window=window,
        min_periods=2 if gappy else None,
        negative="signed",
        invalid="keep",
    )
    got = out[f"spread_edge_moment_{window}"].to_numpy().astype(float)
    prices = _prices(df)
    checked = 0
    for t in range(len(got)):
        a = max(0, t - window + 1) if gappy else t - window + 1
        if a < 0:
            assert np.isnan(got[t]), (t, got[t])
            continue
        s2, n1, n2 = oracle(*(x[a : t + 1] for x in prices))
        if n1 <= 1 or n2 <= 1:
            continue  # degenerate: the reference's weighting is rounding noise
        if np.isnan(s2):
            assert np.isnan(got[t]), (t, got[t])
        else:
            assert abs(got[t] - s2) <= 1e-11, (t, got[t], s2)
        checked += 1
    assert checked >= 300


@pytest.mark.parametrize("window", [3, 4, 5, 21, 63, 252])
@pytest.mark.parametrize("name", ["ohlc_2000.csv.gz", "ohlc-miss_2000.csv.gz"])
def test_edge_matches_the_reference_at_every_index(name, window) -> None:
    _edge_parity(name, window, _reference_edge)


@pytest.mark.parametrize("window", [3, 4, 5, 21, 63, 252])
@pytest.mark.parametrize("name", ["ohlc_2000.csv.gz", "ohlc-miss_2000.csv.gz"])
def test_edge_matches_the_bidask_package_at_every_index(name, window) -> None:
    bidask = pytest.importorskip("bidask")

    def oracle(o, h, lo, c):
        _, n1, n2 = _reference_edge(o, h, lo, c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = bidask.edge(o, h, lo, c, sign=True)
        return math.copysign(s * s, s) if np.isfinite(s) else np.nan, n1, n2

    _edge_parity(name, window, oracle)


def test_edge_expanding_window_matches_the_reference() -> None:
    df = _load_bidask("ohlc_2000.csv.gz").head(300)
    out = ohlc_spread(
        df, entity="e", time="t", window=None, min_periods=3, negative="signed"
    )
    got = out["spread_edge_moment_expanding"].to_numpy()
    prices = _prices(df)
    for t in (2, 3, 10, 57, 299):
        s2, _, _ = _reference_edge(*(x[: t + 1] for x in prices))
        assert got[t] == pytest.approx(s2, rel=1e-9, abs=1e-15)


# --------------------------------------------------------------------------- #
# Corwin-Schultz and Abdi-Ranaldo vs their published formulas
# --------------------------------------------------------------------------- #
# Transcribed from Corwin & Schultz (2012, eqs. 14 and 18, with the section
# III.A overnight adjustment) and Abdi & Ranaldo (2017, eq. 11) -- in the
# EXPONENTIAL form 2(e^a - 1)/(1 + e^a) the paper writes, which our tanh form
# must equal -- and averaged over the w - 1 two-day pairs of a w-bar window
# (the convention of the R `bidask::spread` CS / CS2 / AR / AR2 methods).
_K = 3.0 - 2.0 * math.sqrt(2.0)


def _cs_pairs(h, lo, c, adjust=True):
    h, lo, c = np.log(h), np.log(lo), np.log(c)
    out = np.full(len(h), np.nan)
    for t in range(1, len(h)):
        ht, lt = h[t], lo[t]
        if adjust:
            gap = max(0.0, c[t - 1] - ht) + min(0.0, c[t - 1] - lt)
            ht, lt = ht + gap, lt + gap
        beta = (h[t] - lo[t]) ** 2 + (h[t - 1] - lo[t - 1]) ** 2
        gamma = (max(ht, h[t - 1]) - min(lt, lo[t - 1])) ** 2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / _K - math.sqrt(gamma / _K)
        out[t] = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return out


def _ar_pairs(h, lo, c):
    h, lo, c = np.log(h), np.log(lo), np.log(c)
    eta = (h + lo) / 2.0
    out = np.full(len(h), np.nan)
    out[1:] = 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    return out


@pytest.mark.parametrize("window", [2, 3, 21, 63])
@pytest.mark.parametrize("adjust", [True, False])
def test_corwin_schultz_matches_the_published_formula(window, adjust) -> None:
    df = _load_bidask("ohlc_2000.csv.gz").head(400)
    _, h, lo, c = _prices(df)
    s = _cs_pairs(h, lo, c, adjust)
    for negative, reduce in (
        ("literature", lambda x: np.mean(np.maximum(x, 0.0))),
        ("signed", np.mean),
    ):
        out = ohlc_spread(
            df,
            entity="e",
            time="t",
            method="corwin_schultz",
            window=window,
            negative=negative,
            overnight_adjust=adjust,
        )
        got = out[f"spread_corwin_schultz_{window}"].to_numpy().astype(float)
        moment = out[f"spread_corwin_schultz_moment_{window}"].to_numpy().astype(float)
        for t in range(len(got)):
            if t < window - 1:
                assert np.isnan(got[t])
                continue
            pairs = s[t - window + 2 : t + 1]
            assert got[t] == pytest.approx(reduce(pairs), rel=1e-10, abs=1e-15)
            assert moment[t] == pytest.approx(np.mean(pairs), rel=1e-10, abs=1e-15)


@pytest.mark.parametrize("window", [2, 3, 21, 63])
def test_abdi_ranaldo_matches_the_published_formula(window) -> None:
    df = _load_bidask("ohlc_2000.csv.gz").head(400)
    _, h, lo, c = _prices(df)
    s2 = _ar_pairs(h, lo, c)
    lit = ohlc_spread(df, entity="e", time="t", method="abdi_ranaldo", window=window)[
        f"spread_abdi_ranaldo_{window}"
    ].to_numpy()
    signed = ohlc_spread(
        df,
        entity="e",
        time="t",
        method="abdi_ranaldo",
        window=window,
        negative="signed",
    )
    got = signed[f"spread_abdi_ranaldo_{window}"].to_numpy()
    moment = signed[f"spread_abdi_ranaldo_moment_{window}"].to_numpy()
    for t in range(window - 1, len(got)):
        pairs = s2[t - window + 2 : t + 1]
        m = float(np.mean(pairs))
        assert lit[t] == pytest.approx(
            np.mean(np.sqrt(np.maximum(pairs, 0.0))), rel=1e-10, abs=1e-15
        )
        # the moment carries the rolling sum's residue (~1e-19 where the exact
        # mean is 0); its signed root amplifies that, so compare the root only
        # away from zero
        assert moment[t] == pytest.approx(m, rel=1e-10, abs=1e-17)
        if abs(m) > 1e-12:
            assert got[t] == pytest.approx(
                math.copysign(math.sqrt(abs(m)), m), rel=1e-9
            )
