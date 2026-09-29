"""Bid-ask spreads from OHLC bars: EDGE, Corwin-Schultz and Abdi-Ranaldo.

When quotes are not available, the effective spread still leaves a footprint in
daily bars: trades at the ask and the bid widen the high-low range and pull the
close away from the range midpoint. Three estimators read it:

* **EDGE** (Ardia, Guidotti & Kroencke 2024) -- the efficient generalised
  estimator. It combines two moment conditions built from open, high, low and
  close with GMM-optimal weights. Its *signed* squared spread averaged over
  windows is unbiased, and its error keeps falling as the window grows.
* **Corwin-Schultz** (2012) -- from two consecutive days' high-low ranges.
  The literature clips each two-day estimate at zero before averaging, which
  leaves a bias floor that more data cannot remove.
* **Abdi-Ranaldo** (2017) -- from the close's position relative to two
  consecutive range midpoints.

Every two-day term is indexed by its **later** bar, so the value at row ``t``
uses bars ``<= t`` only. No OHLC estimator resolves a spread that is small
relative to daily volatility on a monthly window: with a 0.1% spread and 1.5%
daily volatility, every one of them has an error several times the spread.

References
----------
Ardia, Guidotti & Kroencke (2024), *J. Financial Economics* 161, 103916
(reference code: github.com/eguidotti/bidask, MIT); Corwin & Schultz (2012),
*J. Finance* 67(2); Abdi & Ranaldo (2017), *Rev. Financial Studies* 30(12).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import polars as pl

from panelary._internal._ohlc import (
    EDGE_TERM,
    LOG_COLUMNS,
    ar_two_day_square,
    check_invalid_policy,
    cs_two_day_spread,
    edge_moment_exprs,
    edge_pair_exprs,
    edge_square_spread,
    lag_exprs,
    map_entity_blocks,
    prepare_log_prices,
    trailing_mean,
    trailing_sum,
)
from panelary.econ.features._common import sorted_panel
from panelary.registry import FeatureSpec

__all__ = ["ohlc_spread"]

SpreadMethod = Literal["edge", "corwin_schultz", "abdi_ranaldo"]
NegativePolicy = Literal["literature", "signed", "abs", "zero", "null"]

_METHODS: tuple[str, ...] = ("edge", "corwin_schultz", "abdi_ranaldo")
_NEGATIVE: tuple[str, ...] = ("literature", "signed", "abs", "zero", "null")

_MOMENT = "__spread_moment"
_CLIPPED = "__spread_clipped"
_COUNT = "__spread_n"
_TERM = "__spread_term"


def _check_int(name: str, value: object, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"`{name}` must be an integer >= {low}, got {value!r}.")
    return value


def _edge_block(
    entity: str, pairs: int | None, min_pairs: int
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Stages 1-4 of EDGE on one block of whole entities; returns ``s^2``."""

    def run(block: pl.DataFrame) -> pl.DataFrame:
        for batch in edge_pair_exprs(entity):
            block = block.with_columns(batch)
        block = block.with_columns(edge_moment_exprs(entity, pairs))
        for batch in edge_square_spread(min_pairs):
            block = block.with_columns(batch)
        return block.select(pl.col(EDGE_TERM).alias(_MOMENT))

    return run


def _two_day_block(
    entity: str,
    pairs: int | None,
    min_pairs: int,
    roles: tuple[str, ...],
    term: list[list[pl.Expr]],
    literature: pl.Expr | None,
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Trailing mean of a two-day term (and of its literature transform).

    ``term`` is a list of ``with_columns`` batches whose last expression is the
    per-pair term.
    """

    def run(block: pl.DataFrame) -> pl.DataFrame:
        block = block.with_columns(lag_exprs(entity, roles))
        for batch in term[:-1]:
            block = block.with_columns(batch)
        block = block.with_columns(term[-1][-1].alias(_TERM))
        t = pl.col(_TERM)
        ok = pl.col(_COUNT) >= min_pairs
        moments = [
            trailing_mean(t, pairs, entity).alias(_MOMENT),
            trailing_sum(t.is_not_null().cast(pl.Float64), pairs, entity).alias(_COUNT),
        ]
        if literature is not None:
            moments.append(trailing_mean(literature, pairs, entity).alias(_CLIPPED))
        block = block.with_columns(moments)
        out = [pl.when(ok).then(pl.col(_MOMENT)).alias(_MOMENT)]
        if literature is not None:
            out.append(pl.when(ok).then(pl.col(_CLIPPED)).alias(_CLIPPED))
        return block.select(out)

    return run


def _signed_root(x: pl.Expr) -> pl.Expr:
    return x.sign() * x.abs().sqrt()


def ohlc_spread(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    time: str,
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    method: SpreadMethod = "edge",
    window: int | None = 21,
    min_periods: int | None = None,
    negative: NegativePolicy = "literature",
    overnight_adjust: bool = True,
    batch_entities: int | None = 256,
    invalid: Literal["null", "clip", "raise", "keep"] = "null",
    alias: str | None = None,
) -> pl.DataFrame:
    """Trailing bid-ask spread estimated from OHLC bars, per entity.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long panel of bars, one row per ``(entity, time)``.
    entity, time : str
        Panel keys.
    open, high, low, close : str
        Price columns (levels). EDGE reads all four; Corwin-Schultz reads high,
        low and (for the overnight adjustment) close; Abdi-Ranaldo reads high,
        low and close. Cross-bar ratios need prices that are point-in-time
        consistent: raw prices, or a total-return adjustment whose *ratios*
        are correct as of each date.
    method : {"edge", "corwin_schultz", "abdi_ranaldo"}, default "edge"
        The estimator (see the module docstring).
    window : int or None, default 21
        Trailing window in **price bars**, as in the reference ``bidask``
        package: a window of ``w`` bars holds ``w - 1`` two-bar pairs. ``>= 3``
        for EDGE, ``>= 2`` otherwise. ``None`` is an expanding window, and then
        ``min_periods`` must be given -- never an implicit ``len(df)``, which
        would make every value depend on how much data follows it.
    min_periods : int, optional
        Bars required, again counting like ``window``: at least
        ``min_periods - 1`` valid pairs (in each of EDGE's two moment
        families). Defaults to ``window`` -- every pair in the window valid,
        like ``bidask.edge_rolling``. A smaller value reproduces
        ``bidask.edge``'s nan-mean treatment of gappy windows. ``>= 2``.
    negative : {"literature", "signed", "abs", "zero", "null"}, default "literature"
        What to do with negative estimates. ``"literature"`` follows each paper:
        EDGE ``sqrt(|s^2|)`` (the reference default); Corwin-Schultz clips each
        two-day spread at zero, then averages; Abdi-Ranaldo averages
        ``sqrt(max(s_t^2, 0))`` (their two-day-corrected estimator).
        ``"signed"`` is ``sign * sqrt(|moment|)`` for EDGE and Abdi-Ranaldo and
        the mean unclipped spread for Corwin-Schultz -- monotone in the unbiased
        moment, and the recommended choice for ML features. ``"abs"`` takes the
        absolute value, ``"zero"`` floors at zero, ``"null"`` nulls negatives.
    overnight_adjust : bool, default True
        Corwin-Schultz only: shift day ``t``'s high and low by the overnight gap
        when the prior close lies outside them (CS 2012, section III.A).
    batch_entities : int or None, default 256
        Evaluate contiguous blocks of this many entities at a time, which bounds
        the memory of EDGE's 39 rolling moments (2-3x faster at 25M rows, 4x
        less memory). The output is bitwise identical for every value, and
        ``None`` evaluates the whole panel at once.
    invalid : {"null", "clip", "raise", "keep"}, default "null"
        Invalid-bar policy, as in
        :func:`~panelary.econ.features.range_volatility`, judged on the columns
        the method reads. Missing prices propagate price by price (EDGE uses
        whatever parts of a bar are present, as the reference does).
    alias : str, optional
        Name of the spread column; the moment column becomes
        ``f"{alias}_moment"``. Defaults to ``f"spread_{method}_{window}"`` and
        ``f"spread_{method}_moment_{window}"`` (``window`` reads
        ``"expanding"`` when ``None``).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(entity, time)`` with two columns appended: the
        spread (a proportion: 0.01 is 1%) and its signed, averaging-safe moment
        -- EDGE's ``s^2``, Abdi-Ranaldo's mean ``s_t^2``, Corwin-Schultz's mean
        unclipped two-day spread. To aggregate across windows, entities or
        time, average the **moment** and then take the root: averaging clipped
        roots is biased.

    Notes
    -----
    EDGE equals the per-window reference ``bidask.edge`` in exact arithmetic,
    missing data included (which ``bidask.edge_rolling`` does not handle), except
    in degenerate windows where one moment family has at most one valid pair:
    there the reference's weighting is decided by floating-point residue, and
    this implementation uses the exact-arithmetic value.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.econ.features import ohlc_spread
    >>> bars = pl.DataFrame(
    ...     {
    ...         "id": ["a"] * 5,
    ...         "t": [1, 2, 3, 4, 5],
    ...         "open": [100.0, 100.4, 99.8, 100.9, 100.1],
    ...         "high": [100.9, 101.0, 100.5, 101.2, 100.8],
    ...         "low": [99.6, 99.9, 99.3, 100.2, 99.7],
    ...         "close": [100.3, 99.9, 100.4, 100.3, 100.6],
    ...     }
    ... )
    >>> out = ohlc_spread(bars, entity="id", time="t", window=4)
    >>> out.columns[-2:]
    ['spread_edge_4', 'spread_edge_moment_4']
    """
    if method not in _METHODS:
        raise ValueError(f"`method` must be one of {list(_METHODS)}, got {method!r}.")
    if negative not in _NEGATIVE:
        raise ValueError(
            f"`negative` must be one of {list(_NEGATIVE)}, got {negative!r}."
        )
    check_invalid_policy(invalid)
    if window is not None:
        window = _check_int("window", window, 3 if method == "edge" else 2)
    if min_periods is not None:
        mp = _check_int("min_periods", min_periods, 2)
    elif window is not None:
        mp = window
    else:
        raise ValueError(
            "an expanding window (window=None) needs an explicit `min_periods`; "
            "a default tied to the data length would make every value depend "
            "on how much data follows it."
        )
    if window is not None and mp > window:
        raise ValueError(f"`min_periods` ({mp}) cannot exceed `window` ({window}).")
    if batch_entities is not None:
        batch_entities = _check_int("batch_entities", batch_entities, 1)
    pairs = None if window is None else window - 1
    min_pairs = mp - 1

    frame = sorted_panel(df, entity, time)
    columns = {"open": open, "high": high, "low": low, "close": close}
    if method == "edge":
        roles: tuple[str, ...] = ("open", "high", "low", "close")
    elif method == "corwin_schultz" and not overnight_adjust:
        roles = ("high", "low")
    else:
        roles = ("high", "low", "close")
    prices = {r: columns[r] for r in roles}
    missing = [c for c in prices.values() if c not in frame.columns]
    if missing:
        raise ValueError(
            f"price column(s) {missing} not found in frame; available: {frame.columns}."
        )
    narrow = frame.select(entity, time, *dict.fromkeys(prices.values()))
    policy = invalid
    if invalid == "raise":
        # One up-front pass so the error counts every invalid bar; the blocks
        # then know every bar is valid.
        prepare_log_prices(
            narrow, entity=entity, time=time, prices=prices, invalid="raise"
        )
        policy = "keep"

    if method == "edge":
        body = _edge_block(entity, pairs, min_pairs)
    elif method == "corwin_schultz":
        s = pl.col(_TERM)
        lit = s.clip(lower_bound=0.0) if negative == "literature" else None
        body = _two_day_block(
            entity,
            pairs,
            min_pairs,
            roles,
            cs_two_day_spread(overnight_adjust=overnight_adjust),
            lit,
        )
    else:
        s2 = pl.col(_TERM)
        lit = s2.clip(lower_bound=0.0).sqrt() if negative == "literature" else None
        body = _two_day_block(
            entity, pairs, min_pairs, roles, [[ar_two_day_square()]], lit
        )

    def fn(block: pl.DataFrame) -> pl.DataFrame:
        # Logs are taken per block, so the log columns of the whole panel never
        # exist at once.
        logs = prepare_log_prices(
            block, entity=entity, time=time, prices=prices, invalid=policy
        )
        return body(logs.select(entity, *(LOG_COLUMNS[r] for r in roles)))

    result = map_entity_blocks(narrow, entity, batch_entities, fn)

    moment = pl.col(_MOMENT)
    if method == "corwin_schultz":
        spread = {
            "literature": pl.col(_CLIPPED) if negative == "literature" else moment,
            "signed": moment,
            "abs": moment.abs(),
            "zero": moment.clip(lower_bound=0.0),
            "null": pl.when(moment >= 0).then(moment),
        }[negative]
    else:
        literature = (
            pl.col(_CLIPPED)
            if method == "abdi_ranaldo" and negative == "literature"
            else moment.abs().sqrt()
        )
        spread = {
            "literature": literature,
            "signed": _signed_root(moment),
            "abs": moment.abs().sqrt(),
            "zero": moment.clip(lower_bound=0.0).sqrt(),
            "null": pl.when(moment >= 0).then(moment.sqrt()),
        }[negative]

    suffix = "expanding" if window is None else str(window)
    name = alias or f"spread_{method}_{suffix}"
    moment_name = f"{alias}_moment" if alias else f"spread_{method}_moment_{suffix}"
    new = result.select(spread.alias(name), moment.alias(moment_name))
    return frame.with_columns(new.get_column(name), new.get_column(moment_name))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="ohlc_spread",
        namespace="econ",
        input_shape="frame",
        output_shape="frame",
        params={
            "method": str,
            "window": int,
            "min_periods": int,
            "negative": str,
            "overnight_adjust": bool,
            "batch_entities": int,
            "invalid": str,
        },
        tier="B",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=(
            "Panelary (clean-room; Ardia, Guidotti & Kroencke 2024, "
            "Corwin & Schultz 2012, Abdi & Ranaldo 2017)"
        ),
        license="Apache-2.0",
        backend_fn=ohlc_spread,
        axis="time",
        flavour="trailing",
        cost_hint="O(N*39)",
    ),
)
