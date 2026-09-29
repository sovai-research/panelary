"""OHLC bar arithmetic shared by the range-volatility and spread estimators.

Private leaf module (numpy + polars only, no panelary imports) behind
:mod:`panelary.econ.features._range` and :mod:`panelary.econ.features._spread`.
It owns the parts of the OHLC contract that must not drift between estimators:

* the **invalid-bar policy** (:func:`prepare_log_prices`) -- which bars are
  trusted, and what happens to the rest;
* the **per-bar log decomposition** -- overnight ``o = O_t - C_{t-1}``, and
  ``u = H - O``, ``d = L - O``, ``c = C - O`` -- and the Parkinson,
  Garman-Klass and Rogers-Satchell per-bar variance terms;
* the **discrete-monitoring bias table** of those three terms
  (:func:`discreteness_factor`);
* the **two-bar spread terms**: the EDGE moment table and its closed-form
  combiner (Ardia, Guidotti & Kroencke 2024), the Corwin-Schultz two-day
  spread and the Abdi-Ranaldo two-day squared spread;
* **trailing aggregation** (:func:`trailing_mean`, :func:`trailing_sum`) and
  **entity batching** (:func:`map_entity_blocks`).

Conventions every caller relies on
----------------------------------
* Log prices only, float64 only. A multiplicative factor common to one bar's
  O, H, L, C cancels out of every within-bar term.
* Every two-bar term is indexed by its **later** bar: the pair ``(t-1, t)``
  lands on row ``t``, built with ``shift(1).over(entity)``. Nothing here ever
  shifts by a negative amount.
* Missing prices (nulls) propagate price by price; a bar that is *present but
  invalid* is handled by the policy. A null or invalid bar is never filled from
  a neighbour.
* Polars nulls, never NaN, mark undefined outputs (except under
  ``invalid="keep"``, which exists for parity tests and passes NaN through).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import polars as pl

__all__ = [
    "GK_CLOSE_WEIGHT",
    "INVALID_POLICIES",
    "LOG_COLUMNS",
    "PARKINSON_SCALE",
    "cs_two_day_spread",
    "ar_two_day_square",
    "discreteness_factor",
    "edge_moment_exprs",
    "edge_pair_exprs",
    "edge_square_spread",
    "entity_blocks",
    "map_entity_blocks",
    "prepare_log_prices",
    "trailing_mean",
    "trailing_sum",
]

#: ``1 / (4 ln 2)``: Parkinson's scaling of the squared log range.
PARKINSON_SCALE: float = 1.0 / (4.0 * math.log(2.0))

#: ``2 ln 2 - 1``: the weight on the squared open-to-close return in the
#: Garman-Klass estimator.
GK_CLOSE_WEIGHT: float = 2.0 * math.log(2.0) - 1.0

#: What to do with a bar that is present but not a valid OHLC bar.
INVALID_POLICIES: tuple[str, ...] = ("null", "clip", "raise", "keep")

#: Internal log-price column per role (natural logs, float64).
LOG_COLUMNS: dict[str, str] = {
    "open": "__ohlc_o",
    "high": "__ohlc_h",
    "low": "__ohlc_l",
    "close": "__ohlc_c",
}

_ROLES: tuple[str, ...] = ("open", "high", "low", "close")


# --------------------------------------------------------------------------- #
# Invalid-bar policy and log prices
# --------------------------------------------------------------------------- #
def check_invalid_policy(invalid: str) -> None:
    """Raise unless ``invalid`` is one of :data:`INVALID_POLICIES`."""
    if invalid not in INVALID_POLICIES:
        raise ValueError(
            f"`invalid` must be one of {list(INVALID_POLICIES)}, got {invalid!r}."
        )


def _bad_value(col: str) -> pl.Expr:
    """True where ``col`` holds a value that is present but not finite and > 0."""
    x = pl.col(col)
    return x.is_not_null() & ~(x.is_finite() & (x > 0))


def _order_violation(cols: Mapping[str, str]) -> pl.Expr:
    """True where the present prices break ``L <= min(O, C) <= max(O, C) <= H``.

    Each pairwise check involving a missing price is treated as satisfied: a
    missing value cannot contradict anything, so it is left to propagate.
    """
    pairs: list[tuple[str, str]] = []  # (smaller role, larger role)
    for small, large in (
        ("low", "open"),
        ("low", "close"),
        ("open", "high"),
        ("close", "high"),
        ("low", "high"),
    ):
        if small in cols and large in cols:
            pairs.append((small, large))
    if not pairs:
        return pl.lit(False)
    return pl.any_horizontal(
        [(pl.col(cols[s]) > pl.col(cols[g])).fill_null(value=False) for s, g in pairs]
    )


def prepare_log_prices(
    frame: pl.DataFrame,
    *,
    entity: str,
    time: str,
    prices: Mapping[str, str],
    invalid: str,
) -> pl.DataFrame:
    """Append float64 log prices for ``prices`` after the invalid-bar policy.

    Parameters
    ----------
    frame : polars.DataFrame
        Panel sorted by ``(entity, time)``.
    entity, time : str
        Panel keys (``time`` only names the offending row in an error).
    prices : Mapping[str, str]
        Role (``"open"``, ``"high"``, ``"low"``, ``"close"``) to column name,
        for the roles the calling estimator uses. Validity is judged on exactly
        these columns.
    invalid : {"null", "clip", "raise", "keep"}
        A bar is valid iff every used price that is present is finite and
        ``> 0`` and ``L <= min(O, C) <= max(O, C) <= H`` holds among the
        present prices. ``"null"`` nulls every used price of an invalid bar;
        ``"clip"`` first nulls bars with a non-finite or non-positive price,
        then repairs the ordering row-locally (``H = max(H, O, C)``,
        ``L = min(L, O, C)``); ``"raise"`` raises :class:`ValueError` naming the
        first invalid bar; ``"keep"`` applies no check (parity tests only --
        logs of non-positive prices become NaN / -inf).

    Returns
    -------
    polars.DataFrame
        ``frame`` plus one :data:`LOG_COLUMNS` column per role in ``prices``.
    """
    check_invalid_policy(invalid)
    unknown = sorted(set(prices) - set(_ROLES))
    if unknown:  # pragma: no cover - internal misuse
        raise ValueError(f"unknown OHLC role(s) {unknown}.")
    missing = [c for c in prices.values() if c not in frame.columns]
    if missing:
        raise ValueError(
            f"price column(s) {missing} not found in frame; available: {frame.columns}."
        )
    tmp = {role: f"__ohlc_p_{role}" for role in prices}
    work = frame.with_columns(
        pl.col(col).cast(pl.Float64).alias(tmp[role]) for role, col in prices.items()
    )

    bad_value = pl.any_horizontal([_bad_value(c) for c in tmp.values()])
    violation = bad_value | _order_violation(tmp)

    if invalid == "raise":
        flags = work.select(violation.alias("__bad"))["__bad"]
        n_bad = int(flags.sum())
        if n_bad:
            first = work.filter(flags).row(0, named=True)
            raise ValueError(
                f"{n_bad} invalid OHLC bar(s): a bar needs every price finite and "
                "> 0 with low <= min(open, close) <= max(open, close) <= high. "
                f"First: {entity}={first[entity]!r}, {time}={first[time]!r} "
                f"({', '.join(f'{r}={first[c]!r}' for r, c in prices.items())}). "
                "Pass invalid='null' to drop such bars or invalid='clip' to "
                "repair the ordering."
            )
    elif invalid == "null":
        work = work.with_columns(
            pl.when(violation).then(None).otherwise(pl.col(c)).alias(c)
            for c in tmp.values()
        )
    elif invalid == "clip":
        work = work.with_columns(
            pl.when(bad_value).then(None).otherwise(pl.col(c)).alias(c)
            for c in tmp.values()
        )
        repairs: list[pl.Expr] = []
        others = [tmp[r] for r in ("open", "close") if r in tmp]
        if "high" in tmp and others:
            repairs.append(
                pl.max_horizontal(pl.col(tmp["high"]), *others).alias(tmp["high"])
            )
        if "low" in tmp and others:
            repairs.append(
                pl.min_horizontal(pl.col(tmp["low"]), *others).alias(tmp["low"])
            )
        if repairs:
            work = work.with_columns(repairs)
        # With neither open nor close in play, only an inverted H/L can remain.
        leftover = _order_violation(tmp)
        work = work.with_columns(
            pl.when(leftover).then(None).otherwise(pl.col(c)).alias(c)
            for c in tmp.values()
        )

    return work.with_columns(
        pl.col(tmp[role]).log().alias(LOG_COLUMNS[role]) for role in prices
    ).drop(list(tmp.values()))


# --------------------------------------------------------------------------- #
# Discrete-monitoring bias of the range terms
# --------------------------------------------------------------------------- #
# b(M) = E[per-bar term] / sigma^2 for a driftless Gaussian random walk with M
# increments per bar (M + 1 observed prices, the open included). Rogers-Satchell
# is exact (a Baxter-Spitzer identity, see the generator); Parkinson and
# Garman-Klass are control-variate Monte Carlo at 10^6 bars per M, standard
# error <= 1e-4. Generated by `benchmarks/ohlc_vol/discreteness_table.py`
# (seed 20260929); regenerate there, never by hand.
_DISCRETE_M: tuple[int, ...] = (
    2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 39, 50, 65, 78, 100, 130, 160,
    195, 260, 390, 520, 780, 1170, 1560, 2340, 3900, 7800, 11700, 23400,
)  # fmt: skip
_DISCRETE_FACTORS: dict[str, tuple[float, ...]] = {
    "parkinson": (
        0.442701,
        0.498473,
        0.539536,
        0.571346,
        0.596973,
        0.636172,
        0.665289,
        0.697664,
        0.721802,
        0.746173,
        0.772713,
        0.792209,
        0.809369,
        0.829228,
        0.848299,
        0.860494,
        0.875506,
        0.889728,
        0.899886,
        0.908853,
        0.920530,
        0.934711,
        0.943067,
        0.953284,
        0.961669,
        0.966589,
        0.972649,
        0.978908,
        0.984931,
        0.987716,
        0.991246,
    ),
    "garman_klass": (
        0.227419,
        0.304736,
        0.361661,
        0.405759,
        0.441287,
        0.495627,
        0.535992,
        0.580873,
        0.614336,
        0.648121,
        0.684914,
        0.711940,
        0.735729,
        0.763259,
        0.789698,
        0.806604,
        0.827415,
        0.847131,
        0.861212,
        0.873643,
        0.889831,
        0.909491,
        0.921075,
        0.935238,
        0.946862,
        0.953682,
        0.962083,
        0.970761,
        0.979110,
        0.982971,
        0.987864,
    ),
    "rogers_satchell": (
        0.159155,
        0.256156,
        0.323794,
        0.374677,
        0.414879,
        0.475326,
        0.519413,
        0.567903,
        0.603647,
        0.639460,
        0.678180,
        0.706308,
        0.731123,
        0.759681,
        0.786874,
        0.804097,
        0.825518,
        0.845765,
        0.860206,
        0.872770,
        0.889141,
        0.908826,
        0.920700,
        0.934920,
        0.946640,
        0.953674,
        0.962063,
        0.970527,
        0.979097,
        0.982911,
        0.987895,
    ),
}


def discreteness_factor(term: str, bars: int) -> float:
    """Expected downward bias ``b(M)`` of one per-bar range term.

    Interpolated linearly in ``1 / sqrt(M)`` -- the variable in which the
    discretisation error of a random walk's extremes is asymptotically linear
    (Asmussen, Glynn & Pitman 1995; Broadie, Glasserman & Kou 1997) -- between
    the stored grid points, and towards the exact limit ``b = 1`` at
    ``M -> oo`` beyond the last one. A constant per ``M``, so it is leak-free.

    Parameters
    ----------
    term : {"parkinson", "garman_klass", "rogers_satchell"}
        Which per-bar term.
    bars : int
        ``M``, the number of intrabar returns the high and low were taken over
        (390 for one-minute sampling of a 6.5-hour session). ``>= 2``.
    """
    table = _DISCRETE_FACTORS.get(term)
    if table is None:  # pragma: no cover - internal misuse
        raise ValueError(f"no discreteness table for {term!r}.")
    if isinstance(bars, bool) or not isinstance(bars, (int, np.integer)) or bars < 2:
        raise ValueError(
            f"`discrete_bars` must be an integer >= 2 (the number of intrabar "
            f"returns behind each high and low), got {bars!r}."
        )
    u_grid = np.asarray(_DISCRETE_M, dtype=float) ** -0.5
    # np.interp needs increasing x: 1/sqrt(M) decreases in M, so flip, and
    # append the M -> oo anchor (u = 0, b = 1).
    xs = np.concatenate([[0.0], u_grid[::-1]])
    ys = np.concatenate([[1.0], np.asarray(table, dtype=float)[::-1]])
    return float(np.interp(float(bars) ** -0.5, xs, ys))


# --------------------------------------------------------------------------- #
# Trailing aggregation and entity batching
# --------------------------------------------------------------------------- #
def trailing_mean(expr: pl.Expr, window: int | None, entity: str) -> pl.Expr:
    """Per-entity trailing mean of the non-null values of ``expr``.

    ``window`` rows, or expanding when ``None``. Always ``min_samples=1``:
    callers impose their own validity counts. Native rolling sums are the
    accumulation primitive -- never a cumsum difference.
    """
    if window is None:
        count = expr.is_not_null().cast(pl.Float64).cum_sum()
        total = expr.fill_null(0.0).cum_sum()
        return pl.when(count > 0).then(total / count).otherwise(None).over(entity)
    return expr.rolling_mean(window_size=window, min_samples=1).over(entity)


def trailing_sum(expr: pl.Expr, window: int | None, entity: str) -> pl.Expr:
    """Per-entity trailing sum of a non-null indicator ``expr`` (a count)."""
    if window is None:
        return expr.cum_sum().over(entity)
    return expr.rolling_sum(window_size=window, min_samples=1).over(entity)


def entity_blocks(
    frame: pl.DataFrame, entity: str, batch: int | None
) -> list[tuple[int, int]]:
    """Contiguous ``(offset, length)`` row blocks of at most ``batch`` entities.

    ``frame`` must be sorted by entity, so each entity is one run of rows.
    ``None`` is one block.
    """
    if batch is None or frame.height == 0:
        return [(0, frame.height)]
    runs = (
        frame.select(pl.col(entity).rle().struct.field("len"))
        .to_series()
        .to_numpy()
        .astype(np.int64)
    )
    ends = np.cumsum(runs)
    starts = ends - runs
    blocks: list[tuple[int, int]] = []
    for i in range(0, len(runs), batch):
        j = min(i + batch, len(runs)) - 1
        blocks.append((int(starts[i]), int(ends[j] - starts[i])))
    return blocks


def map_entity_blocks(
    frame: pl.DataFrame,
    entity: str,
    batch: int | None,
    fn: Callable[[pl.DataFrame], pl.DataFrame],
) -> pl.DataFrame:
    """Apply ``fn`` to contiguous blocks of whole entities and stack the results.

    Bounds the memory of a many-column staged computation: each block is a
    zero-copy slice of the ``(entity, time)``-sorted frame. Every expression
    ``fn`` evaluates must be partitioned ``.over(entity)``, so the output does
    not depend on how the entities are blocked (asserted bitwise in the tests).
    """
    parts = [fn(frame.slice(off, n)) for off, n in entity_blocks(frame, entity, batch)]
    return parts[0] if len(parts) == 1 else pl.concat(parts, how="vertical")


# --------------------------------------------------------------------------- #
# EDGE (Ardia, Guidotti & Kroencke 2024), staged
# --------------------------------------------------------------------------- #
# Stage-2 columns. r1..r5 are the paper's log returns, tau / po1 / po2 / pc1 /
# pc2 its indicators; family 1 is masked to V1 = valid(r1, r2, r3, r4, tau),
# family 2 to V2 = valid(r1, r4, r5, tau), so each family's moments average
# over exactly the rows the reference's nanmean of x1 / x2 would.
_EDGE_UNMASKED: tuple[str, ...] = ("tau", "po1", "po2", "pc1", "pc2", "r1", "r3", "r5")
_EDGE_FAMILY1: dict[str, Callable[[dict[str, pl.Expr]], pl.Expr]] = {
    "f1_r1r2": lambda v: v["r1"] * v["r2"],
    "f1_tr2": lambda v: v["tau"] * v["r2"],
    "f1_r3r4": lambda v: v["r3"] * v["r4"],
    "f1_tr4": lambda v: v["tau"] * v["r4"],
    "f1_r1r2_sq": lambda v: (v["r1"] * v["r2"]) ** 2,
    "f1_tr1r2r2": lambda v: v["tau"] * v["r1"] * v["r2"] * v["r2"],
    "f1_tr2r2": lambda v: v["tau"] * v["r2"] * v["r2"],
    "f1_r3r4_sq": lambda v: (v["r3"] * v["r4"]) ** 2,
    "f1_tr3r4r4": lambda v: v["tau"] * v["r3"] * v["r4"] * v["r4"],
    "f1_tr4r4": lambda v: v["tau"] * v["r4"] * v["r4"],
    "f1_r1r2r3r4": lambda v: v["r1"] * v["r2"] * v["r3"] * v["r4"],
    "f1_tr1r2r4": lambda v: v["tau"] * v["r1"] * v["r2"] * v["r4"],
    "f1_tr2r3r4": lambda v: v["tau"] * v["r2"] * v["r3"] * v["r4"],
    "f1_tr2r4": lambda v: v["tau"] * v["r2"] * v["r4"],
}
_EDGE_FAMILY2: dict[str, Callable[[dict[str, pl.Expr]], pl.Expr]] = {
    "f2_r1r5": lambda v: v["r1"] * v["r5"],
    "f2_tr5": lambda v: v["tau"] * v["r5"],
    "f2_r4r5": lambda v: v["r4"] * v["r5"],
    "f2_tr4": lambda v: v["tau"] * v["r4"],
    "f2_r1r5_sq": lambda v: (v["r1"] * v["r5"]) ** 2,
    "f2_tr1r5r5": lambda v: v["tau"] * v["r1"] * v["r5"] * v["r5"],
    "f2_tr5r5": lambda v: v["tau"] * v["r5"] * v["r5"],
    "f2_r4r5_sq": lambda v: (v["r4"] * v["r5"]) ** 2,
    "f2_tr4r4r5": lambda v: v["tau"] * v["r4"] * v["r4"] * v["r5"],
    "f2_tr4r4": lambda v: v["tau"] * v["r4"] * v["r4"],
    "f2_r1r4r5r5": lambda v: v["r1"] * v["r4"] * v["r5"] * v["r5"],
    "f2_tr1r4r5": lambda v: v["tau"] * v["r1"] * v["r4"] * v["r5"],
    "f2_tr4r5r5": lambda v: v["tau"] * v["r4"] * v["r5"] * v["r5"],
    "f2_tr4r5": lambda v: v["tau"] * v["r4"] * v["r5"],
}
_EDGE_COUNTS: tuple[str, ...] = ("n1", "n2", "ntau")
_P = "__edge_"  # stage-column prefix


def edge_pair_exprs(entity: str) -> list[list[pl.Expr]]:
    """Stages 1-2 of EDGE, as three consecutive ``with_columns`` batches.

    ``[lags, base, products]``: the lagged log prices; then the paper's log
    returns ``r1..r5`` and indicators ``tau, po1, po2, pc1, pc2``; then the
    family-masked products and the three count indicators. Each batch refers
    only to columns materialised by the previous ones, so no sub-expression is
    evaluated twice. Needs all four :data:`LOG_COLUMNS`.
    """
    o, h, lo, c = (pl.col(LOG_COLUMNS[r]) for r in _ROLES)
    lags = [
        h.shift(1).over(entity).alias(f"{_P}h1"),
        lo.shift(1).over(entity).alias(f"{_P}l1"),
        c.shift(1).over(entity).alias(f"{_P}c1"),
    ]
    h1, l1, c1 = pl.col(f"{_P}h1"), pl.col(f"{_P}l1"), pl.col(f"{_P}c1")
    m = (h + lo) / 2.0
    m1 = (h1 + l1) / 2.0

    def known(*xs: pl.Expr) -> pl.Expr:
        return pl.all_horizontal([x.is_not_null() for x in xs])

    def indicator(cond: pl.Expr, *xs: pl.Expr) -> pl.Expr:
        return pl.when(known(*xs)).then(cond.cast(pl.Float64)).otherwise(None)

    tau = indicator((h != lo) | (lo != c1), h, lo, c1)
    base = [
        (m - o).alias(f"{_P}r1"),
        (o - m1).alias(f"{_P}r2"),
        (m - c1).alias(f"{_P}r3"),
        (c1 - m1).alias(f"{_P}r4"),
        (o - c1).alias(f"{_P}r5"),
        tau.alias(f"{_P}tau"),
        (tau * indicator(o != h, o, h)).alias(f"{_P}po1"),
        (tau * indicator(o != lo, o, lo)).alias(f"{_P}po2"),
        (tau * indicator(c1 != h1, c1, h1)).alias(f"{_P}pc1"),
        (tau * indicator(c1 != l1, c1, l1)).alias(f"{_P}pc2"),
    ]
    v = {n: pl.col(f"{_P}{n}") for n in ("r1", "r2", "r3", "r4", "r5", "tau")}
    v1 = known(v["r1"], v["r2"], v["r3"], v["r4"], v["tau"])
    v2 = known(v["r1"], v["r4"], v["r5"], v["tau"])
    products = [
        pl.when(v1).then(f(v)).otherwise(None).alias(f"{_P}{name}")
        for name, f in _EDGE_FAMILY1.items()
    ]
    products += [
        pl.when(v2).then(f(v)).otherwise(None).alias(f"{_P}{name}")
        for name, f in _EDGE_FAMILY2.items()
    ]
    products += [
        v1.cast(pl.Float64).alias(f"{_P}ind_n1"),
        v2.cast(pl.Float64).alias(f"{_P}ind_n2"),
        v["tau"].fill_null(0.0).alias(f"{_P}ind_ntau"),
    ]
    return [lags, base, products]


def edge_moment_exprs(entity: str, pairs: int | None) -> list[pl.Expr]:
    """Stage 3 of EDGE: the 36 trailing means and 3 counts, over ``pairs`` rows."""
    names = [*_EDGE_UNMASKED, *_EDGE_FAMILY1, *_EDGE_FAMILY2]
    out = [
        trailing_mean(pl.col(f"{_P}{n}"), pairs, entity).alias(f"{_P}m_{n}")
        for n in names
    ]
    out += [
        trailing_sum(pl.col(f"{_P}ind_{n}"), pairs, entity).alias(f"{_P}m_{n}")
        for n in _EDGE_COUNTS
    ]
    return out


def edge_square_spread(min_pairs: int) -> pl.Expr:
    """Stage 4 of EDGE: the row-local closed form of the squared spread ``s^2``.

    Equals the per-window reference ``bidask.edge`` in exact arithmetic. When a
    family has at most one valid pair its variance is exactly zero -- set here
    rather than left to floating-point residue, which is what decides the
    reference's weighting in that degenerate case. Null unless the window holds
    at least two ``tau = 1`` pairs, both ``p_o`` and ``p_c`` are non-zero, and
    each family has at least ``min_pairs`` valid pairs.
    """

    def m(name: str) -> pl.Expr:
        return pl.col(f"{_P}m_{name}")

    p_tau = m("tau")
    p_o = m("po1") + m("po2")
    p_c = m("pc1") + m("pc2")
    a = -4.0 / p_o
    b = -4.0 / p_c
    a1 = m("r1") / p_tau
    a3 = m("r3") / p_tau
    a5 = m("r5") / p_tau

    e1 = a * (m("f1_r1r2") - a1 * m("f1_tr2")) + b * (m("f1_r3r4") - a3 * m("f1_tr4"))
    ex1 = (
        a * a * (m("f1_r1r2_sq") - 2.0 * a1 * m("f1_tr1r2r2") + a1 * a1 * m("f1_tr2r2"))
        + b
        * b
        * (m("f1_r3r4_sq") - 2.0 * a3 * m("f1_tr3r4r4") + a3 * a3 * m("f1_tr4r4"))
        + 2.0
        * a
        * b
        * (
            m("f1_r1r2r3r4")
            - a3 * m("f1_tr1r2r4")
            - a1 * m("f1_tr2r3r4")
            + a1 * a3 * m("f1_tr2r4")
        )
    )
    e2 = a * (m("f2_r1r5") - a1 * m("f2_tr5")) + b * (m("f2_r4r5") - a5 * m("f2_tr4"))
    ex2 = (
        a * a * (m("f2_r1r5_sq") - 2.0 * a1 * m("f2_tr1r5r5") + a1 * a1 * m("f2_tr5r5"))
        + b
        * b
        * (m("f2_r4r5_sq") - 2.0 * a5 * m("f2_tr4r4r5") + a5 * a5 * m("f2_tr4r4"))
        + 2.0
        * a
        * b
        * (
            m("f2_r1r4r5r5")
            - a5 * m("f2_tr1r4r5")
            - a1 * m("f2_tr4r5r5")
            + a1 * a5 * m("f2_tr4r5")
        )
    )
    n1, n2 = m("n1"), m("n2")
    v1 = pl.when(n1 <= 1).then(0.0).otherwise((ex1 - e1 * e1).clip(lower_bound=0.0))
    v2 = pl.when(n2 <= 1).then(0.0).otherwise((ex2 - e2 * e2).clip(lower_bound=0.0))
    vt = v1 + v2
    s2 = pl.when(vt > 0).then((v2 * e1 + v1 * e2) / vt).otherwise((e1 + e2) / 2.0)
    ok = (
        (m("ntau") >= 2)
        & (p_o != 0)
        & (p_c != 0)
        & (n1 >= max(min_pairs, 1))
        & (n2 >= max(min_pairs, 1))
    )
    return pl.when(ok.fill_null(value=False)).then(s2).otherwise(None)


# --------------------------------------------------------------------------- #
# Corwin-Schultz (2012) and Abdi-Ranaldo (2017) two-day terms
# --------------------------------------------------------------------------- #
_SQRT2 = math.sqrt(2.0)
_CS_DEN = 3.0 - 2.0 * _SQRT2


def cs_two_day_spread(entity: str, *, overnight_adjust: bool) -> pl.Expr:
    """Corwin-Schultz two-day spread ``S`` of the pair ``(t-1, t)``, on row ``t``.

    ``beta = (H_{t-1} - L_{t-1})^2 + (H_t - L_t)^2``,
    ``gamma = (max(H_{t-1}, H_t) - min(L_{t-1}, L_t))^2``,
    ``alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))``
    and ``S = 2 (e^alpha - 1) / (1 + e^alpha) = 2 tanh(alpha / 2)``, evaluated
    as ``tanh`` (accurate near ``alpha = 0``). With ``overnight_adjust`` day
    ``t``'s high and low are first shifted by the overnight gap when the
    prior close lies outside them (CS 2012, section III.A). Unclipped: may be
    negative.
    """
    h, lo, c = (pl.col(LOG_COLUMNS[r]) for r in ("high", "low", "close"))
    h1 = h.shift(1).over(entity)
    l1 = lo.shift(1).over(entity)
    if overnight_adjust:
        c1 = c.shift(1).over(entity)
        # max(0, C_{t-1} - H_t) + min(0, C_{t-1} - L_t): at most one is non-zero.
        gap = (c1 - h).clip(lower_bound=0.0) + (c1 - lo).clip(upper_bound=0.0)
        ah, al = h + gap, lo + gap
    else:
        ah, al = h, lo
    beta = (h - lo) ** 2 + (h1 - l1) ** 2
    # Null-propagating max / min (max_horizontal would skip a missing value).
    hi2 = pl.when(ah >= h1).then(ah).when(ah < h1).then(h1)
    lo2 = pl.when(al <= l1).then(al).when(al > l1).then(l1)
    gamma = (hi2 - lo2) ** 2
    alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / _CS_DEN - (gamma / _CS_DEN).sqrt()
    return 2.0 * (alpha / 2.0).tanh()


def ar_two_day_square(entity: str) -> pl.Expr:
    """Abdi-Ranaldo two-day squared spread of the pair ``(t-1, t)``, on row ``t``.

    ``s^2 = 4 (C_{t-1} - eta_{t-1}) (C_{t-1} - eta_t)`` with ``eta`` the log
    mid-range ``(H + L) / 2``. May be negative.
    """
    h, lo, c = (pl.col(LOG_COLUMNS[r]) for r in ("high", "low", "close"))
    eta = (h + lo) / 2.0
    eta1 = eta.shift(1).over(entity)
    c1 = c.shift(1).over(entity)
    return 4.0 * (c1 - eta1) * (c1 - eta)
