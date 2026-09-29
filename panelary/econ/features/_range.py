"""Range-based volatility from OHLC bars: Parkinson to Yang-Zhang.

A bar's high and low carry far more information about its variance than its
close does. Per bar, the classical range estimators are worth about five to
eight close-to-close squared returns (Parkinson 5.2, Garman-Klass 7.4,
Rogers-Satchell about 6), so a 21-bar range volatility is roughly as precise as
a 150-bar close-to-close one. They differ in what they are robust to:

=================  ====================================  ============  ===============
method             per-bar term / window formula         drift         opening jump
=================  ====================================  ============  ===============
close_to_close     sample variance of ``C_t - C_{t-1}``  demeaned      included
parkinson          ``(H - L)^2 / (4 ln 2)``              biased up     missed
garman_klass       ``(H - L)^2 / 2 - (2 ln 2 - 1) c^2``  biased up     missed
rogers_satchell    ``u (u - c) + d (d - c)``             unbiased      missed
gk_overnight       mean of ``o^2`` + Garman-Klass term   biased        included
yang_zhang         ``V_o + k V_c + (1 - k) V_RS``        unbiased      consistent
=================  ====================================  ============  ===============

with natural-log prices, ``u = H - O``, ``d = L - O``, ``c = C - O`` and the
overnight return ``o = O_t - C_{t-1}``. Yang-Zhang is the default: the only one
that is drift-free *and* consistent when the market opens away from the prior
close. Its weight is ``k = (alpha - 1) / (alpha + (n + 1) / (n - 1))`` with
``n`` the row's own trailing count of valid bars -- never the series length.

Every value at row ``t`` is a trailing-window statistic of the entity's bars
``<= t`` (the overnight return reaches back exactly one bar), computed with
native rolling moments ``.over(entity)``.

References
----------
Parkinson (1980), *J. Business* 53(1); Garman & Klass (1980), *J. Business*
53(1); Rogers & Satchell (1991), *Ann. Appl. Probab.* 1(4); Yang & Zhang
(2000), *J. Business* 73(3); Broadie, Glasserman & Kou (1997), *Math. Finance*
7(4) for discrete monitoring.
"""

from __future__ import annotations

import math
from typing import Literal

import polars as pl

from panelary._internal._ohlc import (
    GK_CLOSE_WEIGHT,
    LOG_COLUMNS,
    PARKINSON_SCALE,
    check_invalid_policy,
    discreteness_factor,
    prepare_log_prices,
)
from panelary.econ.features._common import sorted_panel
from panelary.registry import FeatureSpec

__all__ = ["range_volatility", "ohlc_variance_terms"]

RangeMethod = Literal[
    "yang_zhang",
    "gk_overnight",
    "rogers_satchell",
    "garman_klass",
    "parkinson",
    "close_to_close",
]
InvalidPolicy = Literal["null", "clip", "raise", "keep"]

_METHODS: tuple[str, ...] = (
    "yang_zhang",
    "gk_overnight",
    "rogers_satchell",
    "garman_klass",
    "parkinson",
    "close_to_close",
)

#: The price columns each estimator reads; bar validity is judged on these only.
_ROLES: dict[str, tuple[str, ...]] = {
    "close_to_close": ("close",),
    "parkinson": ("high", "low"),
    "garman_klass": ("open", "high", "low", "close"),
    "rogers_satchell": ("open", "high", "low", "close"),
    "gk_overnight": ("open", "high", "low", "close"),
    "yang_zhang": ("open", "high", "low", "close"),
}

# Stage-column names (dropped before returning).
_C1 = "__rv_c1"
_TERM = "__rv_term"
_OV = "__rv_ov"
_OC = "__rv_oc"
_RS = "__rv_rs"
_MASK = "__rv_mask"
_VAR = "__rv_var"
_N = "__rv_n"


def _check_int(name: str, value: object, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"`{name}` must be an integer >= {low}, got {value!r}.")
    return value


def _log_terms() -> dict[str, pl.Expr]:
    """Row-local log decomposition of one bar (needs all four log columns)."""
    o, h, lo, c = (pl.col(LOG_COLUMNS[r]) for r in ("open", "high", "low", "close"))
    return {
        "u": h - o,
        "d": lo - o,
        "c": c - o,
        "range2": (h - lo) ** 2,
        "rs": (h - o) * (h - c) + (lo - o) * (lo - c),
    }


def range_volatility(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    method: RangeMethod = "yang_zhang",
    window: int = 21,
    min_periods: int | None = None,
    alpha: float = 1.34,
    output: Literal["vol", "var"] = "vol",
    periods_per_year: float | None = None,
    discrete_bars: int | None = None,
    invalid: InvalidPolicy = "null",
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing range-based volatility of OHLC bars, per entity.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel of bars, one row per ``(entity, time)``.
    entity, time : str
        Panel keys.
    open, high, low, close : str
        Price columns (levels, not logs; any consistent currency). Only the
        columns ``method`` reads must exist: ``close`` for ``close_to_close``,
        ``high`` and ``low`` for ``parkinson``, all four otherwise.
    method : str, default "yang_zhang"
        ``"yang_zhang"``, ``"gk_overnight"``, ``"rogers_satchell"``,
        ``"garman_klass"``, ``"parkinson"`` or ``"close_to_close"`` (see the
        module docstring). Only ``yang_zhang``, ``gk_overnight`` and
        ``close_to_close`` see the overnight move; the pure intraday estimators
        drop it (a 22% understatement when overnight is a quarter of the
        variance).
    window : int, default 21
        Trailing window in bars (rows). ``>= 2`` for ``close_to_close`` and
        ``yang_zhang``, ``>= 1`` otherwise.
    min_periods : int, optional
        Valid bars required in the window (default ``window``). A bar missing
        the previous close (the first bar of an entity, or one after an invalid
        bar) is not valid for the estimators that use the overnight return.
    alpha : float, default 1.34
        Yang-Zhang's ``alpha`` (``>= 1``); ignored by the other methods.
    output : {"vol", "var"}, default "vol"
        Standard deviation (``sqrt`` of the variance) or the variance itself.
    periods_per_year : float, optional
        Annualisation constant applied to the variance after averaging (252,
        365, ``252 * 390``, ...). A user constant: it is **never** inferred from
        the data. ``None`` leaves the estimate per bar.
    discrete_bars : int, optional
        Correct the Parkinson / Garman-Klass / Rogers-Satchell terms for
        discrete monitoring: the high and low of ``discrete_bars`` sampled
        intrabar returns understate the true range (about -9.5% for RS at 390
        one-minute returns, -20% at 78 five-minute ones). Each term is divided
        by its stored factor ``b(M)`` for a Gaussian random walk. In Yang-Zhang
        only the RS component is corrected, in ``gk_overnight`` only the GK
        part. Off by default: on real data ``M`` (the trade count) is unknown and
        microstructure noise widens the range, partly offsetting the bias.
        Not accepted for ``close_to_close``.
    invalid : {"null", "clip", "raise", "keep"}, default "null"
        A bar is valid iff its used prices are finite and ``> 0`` and
        ``L <= min(O, C) <= max(O, C) <= H``. ``"null"`` drops an invalid bar
        (it does not count towards ``min_periods``); ``"clip"`` sets
        ``H = max(H, O, C)``, ``L = min(L, O, C)`` (non-positive or non-finite
        prices are still dropped); ``"raise"`` raises; ``"keep"`` uses the bar
        as is (parity tests only). Zero-range bars (``H = L``) are valid and
        kept. CRSP-style negative prices must be ``abs()``-ed by the caller.
    alias : str, optional
        Output column name. Defaults to ``f"{output}_{method}_{window}"``.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with two columns appended: the
        estimate, and ``f"{name}_n_valid"``, the number of valid bars in the
        window. The estimate is null until ``min_periods`` valid bars exist
        (``max(min_periods, 2)`` for the sample-variance methods).

    Raises
    ------
    ValueError
        On an unknown ``method`` / ``output`` / ``invalid``, a bad ``window``,
        ``min_periods``, ``alpha``, ``periods_per_year`` or ``discrete_bars``,
        a missing price column, or an invalid bar under ``invalid="raise"``.

    Notes
    -----
    Rows are ordered by ``time`` within each entity, and the window counts
    rows, not calendar time. Every row's value uses only bars ``<= t`` of its
    own entity, so the feature is leak-free and prefix-invariant. For an
    exponentially weighted estimator, aggregate the per-bar terms of
    :func:`ohlc_variance_terms` with ``ewm_mean`` instead.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.econ.features import range_volatility
    >>> bars = pl.DataFrame(
    ...     {
    ...         "id": ["a"] * 4,
    ...         "t": [1, 2, 3, 4],
    ...         "open": [100.0, 101.0, 99.5, 100.2],
    ...         "high": [101.5, 102.0, 100.8, 101.0],
    ...         "low": [99.2, 100.1, 98.9, 99.6],
    ...         "close": [100.8, 99.9, 100.4, 100.9],
    ...     }
    ... )
    >>> out = range_volatility(
    ...     bars, entity="id", time="t", method="parkinson", window=3
    ... )
    >>> out.columns[-2:]
    ['vol_parkinson_3', 'vol_parkinson_3_n_valid']
    """
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {list(_METHODS)}, got {method!r}.")
    if output not in ("vol", "var"):
        raise ValueError(f"`output` must be 'vol' or 'var', got {output!r}.")
    check_invalid_policy(invalid)
    sample_var = method in ("close_to_close", "yang_zhang")
    window = _check_int("window", window, 2 if sample_var else 1)
    mp = window if min_periods is None else _check_int("min_periods", min_periods, 1)
    if mp > window:
        raise ValueError(f"`min_periods` ({mp}) cannot exceed `window` ({window}).")
    mp_eff = max(mp, 2) if sample_var else mp
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha >= 1):
        raise ValueError(f"`alpha` must be a finite number >= 1, got {alpha!r}.")
    if periods_per_year is not None and not (
        isinstance(periods_per_year, (int, float))
        and math.isfinite(periods_per_year)
        and periods_per_year > 0
    ):
        raise ValueError(
            "`periods_per_year` must be a positive finite number (a user "
            f"constant, never inferred from the data), got {periods_per_year!r}."
        )
    if discrete_bars is not None and method == "close_to_close":
        raise ValueError(
            "`discrete_bars` corrects the high/low range terms; close_to_close "
            "uses no range, so it does not apply."
        )

    def corr(term: str) -> float:
        return (
            1.0 if discrete_bars is None else discreteness_factor(term, discrete_bars)
        )

    frame = sorted_panel(df, entity, time)
    columns = {"open": open, "high": high, "low": low, "close": close}
    work = prepare_log_prices(
        frame,
        entity=entity,
        time=time,
        prices={role: columns[role] for role in _ROLES[method]},
        invalid=invalid,
    )

    # Stage 1: the one cross-bar input, the previous bar's log close.
    if method in ("close_to_close", "gk_overnight", "yang_zhang"):
        work = work.with_columns(
            pl.col(LOG_COLUMNS["close"]).shift(1).over(entity).alias(_C1)
        )

    # Stage 2: row-local per-bar terms.
    if method == "close_to_close":
        work = work.with_columns(
            (pl.col(LOG_COLUMNS["close"]) - pl.col(_C1)).alias(_TERM)
        )
    elif method == "parkinson":
        h, lo = pl.col(LOG_COLUMNS["high"]), pl.col(LOG_COLUMNS["low"])
        work = work.with_columns(
            ((h - lo) ** 2 * (PARKINSON_SCALE / corr("parkinson"))).alias(_TERM)
        )
    else:
        t = _log_terms()
        gk = (0.5 * t["range2"] - GK_CLOSE_WEIGHT * t["c"] ** 2) / corr("garman_klass")
        overnight = pl.col(LOG_COLUMNS["open"]) - pl.col(_C1)
        if method == "garman_klass":
            work = work.with_columns(gk.alias(_TERM))
        elif method == "rogers_satchell":
            work = work.with_columns((t["rs"] / corr("rogers_satchell")).alias(_TERM))
        elif method == "gk_overnight":
            work = work.with_columns((overnight**2 + gk).alias(_TERM))
        else:  # yang_zhang: three components over one joint set of valid bars
            rs = t["rs"] / corr("rogers_satchell")
            joint = overnight.is_not_null() & t["c"].is_not_null() & rs.is_not_null()
            work = work.with_columns(
                pl.when(joint).then(overnight).otherwise(None).alias(_OV),
                pl.when(joint).then(t["c"]).otherwise(None).alias(_OC),
                pl.when(joint).then(rs).otherwise(None).alias(_RS),
                joint.alias(_MASK),
            )

    # Stage 3: every trailing moment in one pass, partitioned by entity.
    if method == "yang_zhang":
        n_valid = pl.col(_MASK).cast(pl.Int64).rolling_sum(window, min_samples=1)
        work = work.with_columns(
            pl.col(_OV)
            .rolling_var(window, min_samples=mp_eff, ddof=1)
            .over(entity)
            .clip(lower_bound=0.0)
            .alias(_OV),
            pl.col(_OC)
            .rolling_var(window, min_samples=mp_eff, ddof=1)
            .over(entity)
            .clip(lower_bound=0.0)
            .alias(_OC),
            pl.col(_RS)
            .rolling_mean(window, min_samples=mp_eff)
            .over(entity)
            .alias(_RS),
            n_valid.over(entity).alias(_N),
        )
        # Stage 4: row-local closed form. k uses the row's own valid count.
        n = pl.col(_N).cast(pl.Float64)
        k = (alpha - 1.0) / (alpha + (n + 1.0) / (n - 1.0))
        variance = pl.col(_OV) + k * pl.col(_OC) + (1.0 - k) * pl.col(_RS)
        work = work.with_columns(
            pl.when(pl.col(_N) >= mp_eff).then(variance).otherwise(None).alias(_VAR)
        )
    else:
        term = pl.col(_TERM)
        moment = (
            term.rolling_var(window, min_samples=mp_eff, ddof=1).clip(lower_bound=0.0)
            if method == "close_to_close"
            else term.rolling_mean(window, min_samples=mp_eff)
        )
        n_valid = term.is_not_null().cast(pl.Int64).rolling_sum(window, min_samples=1)
        work = work.with_columns(
            moment.over(entity).alias(_VAR), n_valid.over(entity).alias(_N)
        )

    name = alias or f"{output}_{method}_{window}"
    variance = pl.col(_VAR)
    if periods_per_year is not None:
        variance = variance * float(periods_per_year)
    value = (
        variance if output == "var" else pl.when(variance >= 0).then(variance.sqrt())
    )
    return work.select(
        *frame.columns,
        value.alias(name),
        pl.col(_N).alias(f"{name}_n_valid"),
    )


def ohlc_variance_terms(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    invalid: InvalidPolicy = "null",
) -> pl.DataFrame:
    """Per-bar log decomposition and range-variance terms, for custom aggregation.

    Everything :func:`range_volatility` averages, one row per bar, so an
    exponentially weighted (or any other causal) estimator is a one-liner, e.g.
    ``pl.col("ohlc_rogers_satchell").ewm_mean(half_life=10).over(entity)``.

    Appended columns (natural-log prices):

    * ``ohlc_o`` -- overnight return ``O_t - C_{t-1}`` (null on an entity's
      first bar and after an invalid bar);
    * ``ohlc_u``, ``ohlc_d``, ``ohlc_c`` -- ``H - O``, ``L - O``, ``C - O``;
    * ``ohlc_parkinson`` -- ``(H - L)^2 / (4 ln 2)``;
    * ``ohlc_garman_klass`` -- ``(H - L)^2 / 2 - (2 ln 2 - 1) c^2``;
    * ``ohlc_rogers_satchell`` -- ``u (u - c) + d (d - c)``;
    * ``ohlc_gk_overnight`` -- ``o^2`` plus the Garman-Klass term.

    Every column is row-local except ``ohlc_o`` (and so ``ohlc_gk_overnight``),
    which reaches back exactly one bar. ``invalid`` is as in
    :func:`range_volatility`; all four price columns are required.

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with the eight columns appended.
    """
    frame = sorted_panel(df, entity, time)
    work = prepare_log_prices(
        frame,
        entity=entity,
        time=time,
        prices={"open": open, "high": high, "low": low, "close": close},
        invalid=invalid,
    ).with_columns(pl.col(LOG_COLUMNS["close"]).shift(1).over(entity).alias(_C1))
    t = _log_terms()
    overnight = pl.col(LOG_COLUMNS["open"]) - pl.col(_C1)
    gk = 0.5 * t["range2"] - GK_CLOSE_WEIGHT * t["c"] ** 2
    return work.select(
        *frame.columns,
        overnight.alias("ohlc_o"),
        t["u"].alias("ohlc_u"),
        t["d"].alias("ohlc_d"),
        t["c"].alias("ohlc_c"),
        (t["range2"] * PARKINSON_SCALE).alias("ohlc_parkinson"),
        gk.alias("ohlc_garman_klass"),
        t["rs"].alias("ohlc_rogers_satchell"),
        (overnight**2 + gk).alias("ohlc_gk_overnight"),
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="range_volatility",
        namespace="econ",
        input_shape="frame",
        output_shape="frame",
        params={
            "method": str,
            "window": int,
            "min_periods": int,
            "alpha": float,
            "output": str,
            "periods_per_year": float,
            "discrete_bars": int,
            "invalid": str,
        },
        tier="B",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=(
            "Panelary (clean-room; Parkinson 1980, Garman & Klass 1980, "
            "Rogers & Satchell 1991, Yang & Zhang 2000)"
        ),
        license="Apache-2.0",
        backend_fn=range_volatility,
        axis="time",
        flavour="trailing",
        cost_hint="O(N)",
    ),
)
