"""Typed, leak-safe operator vocabulary for formulaic-alpha genetic programming.

This module is the *primitive set* of :mod:`panelary.evolve`. It defines
52 operators as native Polars expressions and registers each one as a
:class:`~panelary.registry.FeatureSpec`, closing the gap the build
contract documents: before this module the registry held 56 specs of which
**zero** were time-series *series -> series* operators (42 ``ts`` aggregators,
7 ``xs``, 4 ``factor``, 3 ``panel``).

Provenance / licence
--------------------
Every definition here is a **clean-room re-implementation from published
formulae**, written against permissively licensed references only:

* Kakushadze (2016), "101 Formulaic Alphas", arXiv:1601.00991 -- the published
  paper, for ``delay``, ``delta``, ``correlation``, ``decay_linear``,
  ``signedpower``, ``scale``, ``ts_rank``, ``ts_argmax``/``ts_argmin``,
  ``product`` and the cross-sectional ``rank``;
* Microsoft **Qlib** (MIT) -- ``Ref``/``Mean``/``Std``/``Sum``/``Min``/``Max``/
  ``Med``/``Skew``/``Kurt``/``Corr``/``Beta``/``Rsv``/``IdxMax``/``IdxMin``
  operator taxonomy;
* wukan1986/**polars_ta** (MIT) -- the demonstration that the whole Alpha101
  vocabulary is expressible in native Polars rolling expressions.

No code was copied from any of the "all rights reserved" alpha-mining
repositories or from any GPL/AGPL symbolic-regression package; see the
licence-discipline section of ``plans/todo/evolve-build-contract.md``.
:mod:`panelary.catch22` is the clean-room precedent this module follows.

The four rules every operator here obeys
----------------------------------------
1. **No grouping.** :attr:`Op.build` returns a *bare* expression and never calls
   ``.over(...)``. :attr:`Op.kind` (``"elem"`` / ``"ts"`` / ``"xs"``) tells the
   compiler which partition scope to wrap it in, and the compiler owns that
   decision because grouping is what it hoists and shares across a population.
   Nesting ``.over(time)`` around ``.over(entity)`` silently returns all nulls
   (measured on this repo), so scope switches must be separate stages.
2. **Explicit ``min_samples``.** Every rolling expression passes
   ``min_samples=window`` (see :func:`min_samples_for`). An unbounded or
   partial window makes the value at ``t`` depend on how much history happens
   to exist, which breaks prefix invariance.
3. **No ``rolling_map``.** Measured ~249x penalty (AGENTS.md). The two
   operators that naively need a Python callback -- ``ts_argmax`` and
   ``product`` -- have exact vectorised forms and use them.
4. **Protected arithmetic is protected *inside* the operator.** Division,
   logarithm, z-scores and correlations emit ``null`` (never ``inf``/``NaN``)
   where they are undefined, because null-handling cost dominates arithmetic
   cost in Polars -- a downstream ``fill_nan`` pass is both slower and wrong.

The unit system
---------------
Operators are strongly typed over :data:`~panelary.evolve._types.Unit`
(Montana 1995, *Strongly Typed Genetic Programming*). Typing is expressed
*entirely* through per-slot ``in_units``, so it is enforced by construction --
by the genome builder, by ``_compile.validate`` and by :func:`accepts` alike --
and never depends on a caller remembering a side condition:

``"any"`` in ``in_units`` accepts any unit
    and ``out_unit="any"`` means the operator *propagates* its first argument's
    unit, so ``ts_mean`` of a ``price`` is a ``price``. Read the result with
    :func:`resolve_out_unit`, never from a bare :attr:`Op.out_unit`.

A concrete slot drawn from :data:`~._types.DIMENSIONLESS`
    (``ret``/``ratio``/``score``) accepts *any* dimensionless unit, so
    ``mul(price, ret)`` is legal and yields a ``price``.

**Sums are unit-specialised; quotients are not.** ``add``/``sub``/``max``/
``min`` exist as one variant per unit class -- ``add_price``, ``add_volume``,
``add_score`` (the last covering all of ``ret``/``ratio``/``score``) -- which
is what makes ``price + volume`` *unconstructible* rather than merely
penalised: no operator has slots it fits. ``div`` stays polymorphic
(``("any", "any") -> "ratio"``) because a quotient of unlike units is a
well-defined rate, and ``ts_corr``/``ts_beta`` stay polymorphic because
correlation is unit-free -- ``ts_corr(close, volume, 10)`` is an Alpha101
staple and must remain legal.

Use :func:`accepts` to test a candidate argument tuple and
:func:`default_grammar` to obtain the closed primitive set reachable from a
given set of base-column units.

Cost model
----------
:attr:`Op.cost` is taken from the measured table in the build contract:
``rank().over(time)`` costs ~18.4 ms/expr against ~0.85 ms/expr for
``rolling_mean().over(entity)`` on 200k rows, hence ``elem=1``, ``ts=3`` and
``xs=``:data:`~panelary.evolve._types.XS_COST_RATIO` ``=20``.

Inputs are cast to ``Float64`` inside every builder (invariant 3 of AGENTS.md);
outputs are guaranteed free of ``inf``/``NaN``. Base columns are assumed
``NaN``-free -- :func:`sanitize` is provided for the compiler to enforce that
once, at the leaves, instead of once per node.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from panelary.registry import FeatureSpec, registry

from ._types import DIMENSIONLESS, XS_COST_RATIO, Op, OpKind, Unit

__all__ = [
    "OPS",
    "ARITH_UNIT_CLASSES",
    "DEFAULT_WINDOWS",
    "STAT_WINDOWS",
    "HALF_LIVES",
    "ELEM_COST",
    "TS_COST",
    "XS_COST",
    "EPS",
    "accepts",
    "resolve_out_unit",
    "min_samples_for",
    "sanitize",
    "ops_by_kind",
    "ops_producing",
    "default_grammar",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Default window grid for shift-like operators (``ts_delay``, ``ts_delta``,
#: ``ts_pctchange``), where a window of 1 is meaningful (a one-period lag).
DEFAULT_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 40, 60)

#: Window grid for rolling *statistics*. ``1`` is dropped because
#: ``rolling_mean(1)`` is the identity and ``rolling_std(1)`` is null
#: everywhere -- degenerate genes that waste evaluation budget.
STAT_WINDOWS: tuple[int, ...] = (5, 10, 20, 40, 60)

#: Half-lives (in periods) for the exponentially weighted operator, chosen to
#: span the same effective memory as :data:`STAT_WINDOWS`.
HALF_LIVES: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)

#: Exponents for ``signed_power``. 0.5 compresses tails, 2/3 expand them.
POWERS: tuple[float, ...] = (0.5, 2.0, 3.0)

#: Winsorisation widths, in cross-sectional standard deviations.
WINSOR_K: tuple[float, ...] = (2.0, 3.0, 4.0)

#: Relative evaluation costs; see the module docstring.
ELEM_COST: int = 1
TS_COST: int = 3
XS_COST: int = XS_COST_RATIO

#: Magnitude below which a denominator/argument is treated as degenerate.
EPS: float = 1e-12

#: 1 / Phi^-1(0.75); scales a median absolute deviation to a standard-deviation
#: equivalent for a Gaussian sample.
_MAD_SCALE: float = 1.4826

_LICENSE = "Apache-2.0"
_SRC_PK = "Panelary"
_SRC_A101 = "Kakushadze (2016), 101 Formulaic Alphas, arXiv:1601.00991"
_SRC_QLIB = "Qlib operator taxonomy (MIT), re-derived"
_SRC_PTA = "polars_ta (MIT) operator taxonomy, re-derived"

#: Unit classes over which the additive operators (``add``/``sub``/``max``/
#: ``min``) are specialised, as ``(name suffix, slot unit)``. ``"score"`` covers
#: the whole of :data:`~._types.DIMENSIONLESS`, so one variant serves ``ret``,
#: ``ratio`` and ``score``. Specialising -- rather than declaring a single
#: wildcard ``add`` -- is what makes ``price + volume`` unconstructible.
ARITH_UNIT_CLASSES: tuple[tuple[str, Unit], ...] = (
    ("price", "price"),
    ("volume", "volume"),
    ("score", "score"),
)


# --------------------------------------------------------------------------- #
# Expression helpers
# --------------------------------------------------------------------------- #
def min_samples_for(window: int) -> int:
    """Return the ``min_samples`` a rolling operator of size ``window`` uses.

    Panelary's evolve operators require a **full** window: a value is emitted
    only once ``window`` observations exist. Partial windows would make the
    early part of a series follow a different distribution from the rest, and
    any length-dependent relaxation would break prefix invariance.

    Parameters
    ----------
    window : int
        Rolling window size in periods.

    Returns
    -------
    int
        The ``min_samples`` value, equal to ``window``.
    """
    return int(window)


def sanitize(expr: pl.Expr) -> pl.Expr:
    """Cast to ``Float64`` and turn any ``NaN`` into a ``null``.

    Operators in this module never *emit* ``NaN``, but a base column may
    contain one, and ``NaN`` poisons every rolling window it touches. The
    compiler should apply this once per base column rather than once per node.

    Parameters
    ----------
    expr : polars.Expr
        A base-column expression.

    Returns
    -------
    polars.Expr
        The ``Float64``, ``NaN``-free expression.
    """
    return expr.cast(pl.Float64).fill_nan(None)


def _f(expr: pl.Expr) -> pl.Expr:
    """Upcast to ``Float64`` (AGENTS.md invariant 3: float64 everywhere)."""
    return expr.cast(pl.Float64)


def _finite(expr: pl.Expr) -> pl.Expr:
    """Map ``inf``/``-inf``/``NaN`` to ``null``, leaving finite values alone."""
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def _window(param: int | float | None, name: str) -> int:
    """Validate and coerce a window parameter."""
    if param is None:
        raise ValueError(f"operator {name!r} requires a window parameter, got None")
    window = int(param)
    if window < 1:
        raise ValueError(f"operator {name!r}: window must be >= 1, got {param!r}")
    return window


def _float_param(param: int | float | None, name: str) -> float:
    """Validate and coerce a real-valued parameter."""
    if param is None:
        raise ValueError(f"operator {name!r} requires a parameter, got None")
    return float(param)


def _full_window_mask(x: pl.Expr, window: int) -> pl.Expr:
    """Boolean expression: the trailing ``window`` values are all non-null."""
    present = x.is_not_null().cast(pl.Float64)
    return present.rolling_sum(window, min_samples=window) >= float(window)


# --------------------------------------------------------------------------- #
# Element-wise builders (kind="elem")
# --------------------------------------------------------------------------- #
def _b_add(a: pl.Expr, b: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _f(a) + _f(b)


def _b_sub(a: pl.Expr, b: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _f(a) - _f(b)


def _b_mul(a: pl.Expr, b: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _finite(_f(a) * _f(b))


def _b_div(a: pl.Expr, b: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Protected division: ``a / b``, null where the quotient is not finite.

    Handling the degenerate denominator *inside* the operator (rather than with
    a downstream ``fill_nan``) is deliberate: ``0/0 -> NaN`` and ``x/0 -> inf``
    would otherwise survive into a rolling window and poison every value that
    window touches.
    """
    return _finite(_f(a) / _f(b))


def _b_neg(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return -_f(a)


def _b_abs(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _f(a).abs()


def _b_sign(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _f(a).sign()


def _b_log(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Protected logarithm ``log(|x|)``, null for ``|x| <= EPS``.

    The ``log(|x|)`` form (rather than ``log(x)``) is the standard protected
    logarithm of the GP literature (gplearn, BSD-3); it keeps the operator
    total on sign-varying inputs instead of nulling out half a return series.
    """
    mag = _f(a).abs()
    return pl.when(mag > EPS).then(mag.log()).otherwise(None)


def _b_sqrt(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Protected square root ``sqrt(|x|)`` (total, never ``NaN``)."""
    return _f(a).abs().sqrt()


def _b_signed_power(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """``sign(x) * |x| ** p`` -- Alpha101's ``signedpower``.

    A bare ``x.pow(p)`` on a negative base yields ``NaN`` for fractional ``p``,
    which is why the magnitude and the sign are handled separately.
    """
    p = _float_param(param, "signed_power")
    x = _f(a)
    return _finite(x.sign() * x.abs().pow(p))


def _b_max2(a: pl.Expr, b: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """``max(a, b)``, written arithmetically so nulls propagate.

    ``pl.max_horizontal`` *ignores* nulls, which would silently impute a
    missing observation with the other argument.
    """
    x, y = _f(a), _f(b)
    return (x + y + (x - y).abs()) / 2.0


def _b_min2(a: pl.Expr, b: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """``min(a, b)``, written arithmetically so nulls propagate."""
    x, y = _f(a), _f(b)
    return (x + y - (x - y).abs()) / 2.0


def _b_sigmoid(a: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Logistic squash, clipped at +/-30 so ``exp`` cannot overflow."""
    x = _f(a).clip(-30.0, 30.0)
    return 1.0 / (1.0 + (-x).exp())


def _b_where_positive(
    c: pl.Expr, a: pl.Expr, b: pl.Expr, param: int | float | None = None
) -> pl.Expr:
    """``a`` where the condition is strictly positive, else ``b``."""
    return pl.when(_f(c) > 0.0).then(_f(a)).otherwise(_f(b))


# --------------------------------------------------------------------------- #
# Time-series builders (kind="ts"; the compiler wraps these in .over(entity))
# --------------------------------------------------------------------------- #
def _b_ts_delay(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _f(x).shift(_window(param, "ts_delay"))


def _b_ts_delta(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _f(x).diff(_window(param, "ts_delta"))


def _b_ts_pctchange(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    return _finite(_f(x).pct_change(_window(param, "ts_pctchange")))


def _b_ts_mean(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_mean")
    return _f(x).rolling_mean(d, min_samples=min_samples_for(d))


def _b_ts_std(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_std")
    return _f(x).rolling_std(d, min_samples=min_samples_for(d))


def _b_ts_sum(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_sum")
    return _f(x).rolling_sum(d, min_samples=min_samples_for(d))


def _b_ts_min(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_min")
    return _f(x).rolling_min(d, min_samples=min_samples_for(d))


def _b_ts_max(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_max")
    return _f(x).rolling_max(d, min_samples=min_samples_for(d))


def _b_ts_median(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_median")
    return _f(x).rolling_median(d, min_samples=min_samples_for(d))


def _b_ts_range(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_range")
    ms = min_samples_for(d)
    y = _f(x)
    return y.rolling_max(d, min_samples=ms) - y.rolling_min(d, min_samples=ms)


def _b_ts_skew(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_skew")
    return _finite(_f(x).rolling_skew(d, min_samples=min_samples_for(d)))


def _b_ts_kurt(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_kurt")
    return _finite(_f(x).rolling_kurtosis(d, min_samples=min_samples_for(d)))


def _b_ts_zscore(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    d = _window(param, "ts_zscore")
    ms = min_samples_for(d)
    y = _f(x)
    return _finite(
        (y - y.rolling_mean(d, min_samples=ms)) / y.rolling_std(d, min_samples=ms)
    )


def _b_ts_ir(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Trailing information ratio: rolling mean over rolling standard deviation."""
    d = _window(param, "ts_ir")
    ms = min_samples_for(d)
    y = _f(x)
    return _finite(y.rolling_mean(d, min_samples=ms) / y.rolling_std(d, min_samples=ms))


def _b_ts_rsv(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Stochastic position of the current value inside its trailing range.

    ``(x - min) / (max - min)`` in ``[0, 1]``; Qlib calls this ``Rsv``.
    """
    d = _window(param, "ts_rsv")
    ms = min_samples_for(d)
    y = _f(x)
    lo = y.rolling_min(d, min_samples=ms)
    hi = y.rolling_max(d, min_samples=ms)
    return _finite((y - lo) / (hi - lo))


def _b_ts_drawdown(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Current value relative to the trailing running maximum, minus one."""
    d = _window(param, "ts_drawdown")
    ms = min_samples_for(d)
    y = _f(x)
    return _finite(y / y.rolling_max(d, min_samples=ms) - 1.0)


def _b_ts_rank(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Rank of the current value within its trailing window, normalised to (0, 1].

    Vectorised as a horizontal sum of ``d`` shifted comparisons rather than a
    ``rolling_map`` (forbidden: measured 249x penalty). Nulls contribute 0 to
    ``sum_horizontal``, so the result is explicitly masked to rows whose whole
    trailing window is observed.
    """
    d = _window(param, "ts_rank")
    y = _f(x)
    le_count = pl.sum_horizontal([(y.shift(k) <= y).cast(pl.Float64) for k in range(d)])
    return pl.when(_full_window_mask(y, d)).then(le_count / float(d)).otherwise(None)


def _b_ts_argmax(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Periods since the trailing-window maximum (0 = the maximum is now).

    Vectorised with a ``coalesce`` over ``d`` shifted equality tests, so no
    ``rolling_map`` and no Python callback is involved. Ties resolve to the
    most recent occurrence, which is the causal choice.
    """
    d = _window(param, "ts_argmax")
    y = _f(x)
    hi = y.rolling_max(d, min_samples=min_samples_for(d))
    return pl.coalesce(
        [
            pl.when(y.shift(k) == hi).then(pl.lit(float(k), dtype=pl.Float64))
            for k in range(d)
        ]
    )


def _b_ts_argmin(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Periods since the trailing-window minimum (0 = the minimum is now)."""
    d = _window(param, "ts_argmin")
    y = _f(x)
    lo = y.rolling_min(d, min_samples=min_samples_for(d))
    return pl.coalesce(
        [
            pl.when(y.shift(k) == lo).then(pl.lit(float(k), dtype=pl.Float64))
            for k in range(d)
        ]
    )


def _b_ts_decay_linear(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Alpha101 ``decay_linear``: linearly weighted trailing mean.

    Exactly ``rolling_mean(d, weights=[1, 2, ..., d])`` -- Polars normalises by
    the weight sum, so this is the published definition with no rescaling. The
    ``fill_null(0)`` is a workaround for a Polars limitation (weighted rolling
    kernels panic on null-containing input, polars 1.44.1); it is
    value-preserving because the result is masked to fully observed windows
    anyway.
    """
    d = _window(param, "ts_decay_linear")
    y = _f(x)
    weights = [float(k) for k in range(1, d + 1)]
    weighted = y.fill_null(0.0).rolling_mean(
        d, weights=weights, min_samples=min_samples_for(d)
    )
    return pl.when(_full_window_mask(y, d)).then(weighted).otherwise(None)


def _b_ts_product(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Alpha101 ``product``: running product over the trailing window.

    Computed in log space -- ``exp(rolling_sum(log|x|))`` -- with the sign
    recovered from the parity of the negative count and an explicit zero
    branch, because ``rolling_map(np.prod)`` is forbidden.
    """
    d = _window(param, "ts_product")
    ms = min_samples_for(d)
    y = _f(x)
    magnitude = pl.when(y.abs() > EPS).then(y.abs().log()).otherwise(None)
    log_sum = magnitude.rolling_sum(d, min_samples=ms)
    n_negative = (y < 0.0).cast(pl.Float64).rolling_sum(d, min_samples=ms)
    n_zero = (y.abs() <= EPS).cast(pl.Float64).rolling_sum(d, min_samples=ms)
    sign = 1.0 - 2.0 * (n_negative % 2.0)
    return (
        pl.when(~_full_window_mask(y, d))
        .then(None)
        .when(n_zero > 0.0)
        .then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(_finite(sign * log_sum.exp()))
    )


def _b_ts_ewm_mean(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Exponentially weighted mean with an explicit half-life.

    Unbounded in *history* but strictly causal and prefix-invariant: the
    recursion ``m_t = a*x_t + (1-a)*m_{t-1}`` with ``adjust=False`` reads only
    ``x[:t]`` and never rescales by the series length.
    """
    half_life = _float_param(param, "ts_ewm_mean")
    if half_life <= 0.0:
        raise ValueError(f"ts_ewm_mean: half_life must be > 0, got {half_life!r}")
    return _f(x).ewm_mean(
        half_life=half_life, adjust=False, min_samples=1, ignore_nulls=False
    )


def _b_ts_corr(x: pl.Expr, y: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Alpha101 ``correlation``: trailing Pearson correlation of two series."""
    d = _window(param, "ts_corr")
    return _finite(
        pl.rolling_corr(_f(x), _f(y), window_size=d, min_samples=min_samples_for(d))
    )


def _b_ts_beta(x: pl.Expr, y: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Trailing univariate regression slope of ``x`` on ``y``: ``cov / var``."""
    d = _window(param, "ts_beta")
    ms = min_samples_for(d)
    a, b = _f(x), _f(y)
    cov = pl.rolling_cov(a, b, window_size=d, min_samples=ms)
    var = b.rolling_var(d, min_samples=ms)
    return _finite(cov / var)


# --------------------------------------------------------------------------- #
# Cross-sectional builders (kind="xs"; the compiler wraps these in .over(time))
# --------------------------------------------------------------------------- #
def _b_cs_rank(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Alpha101 ``rank``: per-date percentile rank in ``(0, 1]``.

    Normalised by ``count()`` (non-null observations), **not** ``len()``: a
    date with missing values would otherwise produce ranks whose maximum drifts
    with the amount of missingness, making the feature a proxy for coverage.
    """
    y = _f(x)
    return _finite(_f(y.rank(method="average")) / _f(y.count()))


def _b_cs_zscore(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Per-date standardisation to mean 0, standard deviation 1."""
    y = _f(x)
    return _finite((y - y.mean()) / y.std())


def _b_cs_demean(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Per-date mean removal; keeps the input's unit."""
    y = _f(x)
    return y - y.mean()


def _b_cs_scale(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Alpha101 ``scale``: rescale so the per-date absolute values sum to 1."""
    y = _f(x)
    return _finite(y / y.abs().sum())


def _b_cs_winsor(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Clip to ``mean +/- k * std`` within each date."""
    k = _float_param(param, "cs_winsor")
    y = _f(x)
    centre = y.mean()
    spread = y.std()
    return y.clip(centre - k * spread, centre + k * spread)


def _b_cs_mad_zscore(x: pl.Expr, param: int | float | None = None) -> pl.Expr:
    """Robust per-date standardisation using the median absolute deviation."""
    y = _f(x)
    centre = y.median()
    mad = (y - centre).abs().median()
    return _finite((y - centre) / (_MAD_SCALE * mad))


# --------------------------------------------------------------------------- #
# The operator table
# --------------------------------------------------------------------------- #
def _arithmetic_variants() -> tuple[Op, ...]:
    """Build the unit-specialised additive operators.

    One variant per entry of :data:`ARITH_UNIT_CLASSES`, all sharing the same
    builder. Specialising the *slots* (rather than declaring a single wildcard
    ``add`` and checking argument agreement afterwards) is deliberate: it makes
    ``price + volume`` unconstructible for every consumer of the grammar --
    the genome builder, the compiler's ``validate`` and :func:`accepts` -- with
    no shared side condition any of them could forget.
    """
    template = (
        ("add", _b_add, True),
        ("sub", _b_sub, False),
        ("max", _b_max2, True),
        ("min", _b_min2, True),
    )
    return tuple(
        Op(
            name=f"{base}_{suffix}",
            arity=2,
            kind="elem",
            in_units=(unit, unit),
            out_unit=unit,
            build=builder,
            cost=ELEM_COST,
            commutative=commutative,
            source=_SRC_PK,
        )
        for suffix, unit in ARITH_UNIT_CLASSES
        for base, builder, commutative in template
    )


_OP_LIST: tuple[Op, ...] = _arithmetic_variants() + (
    # -- element-wise ------------------------------------------------------- #
    Op(
        name="mul",
        arity=2,
        kind="elem",
        in_units=("any", "score"),
        out_unit="any",
        build=_b_mul,
        cost=ELEM_COST,
        source=_SRC_PK,
    ),
    Op(
        name="div",
        arity=2,
        kind="elem",
        in_units=("any", "any"),
        out_unit="ratio",
        build=_b_div,
        cost=ELEM_COST,
        source=_SRC_PK,
    ),
    Op(
        name="neg",
        arity=1,
        kind="elem",
        in_units=("any",),
        out_unit="any",
        build=_b_neg,
        cost=ELEM_COST,
        source=_SRC_PK,
    ),
    Op(
        name="abs",
        arity=1,
        kind="elem",
        in_units=("any",),
        out_unit="any",
        build=_b_abs,
        cost=ELEM_COST,
        source=_SRC_PK,
    ),
    Op(
        name="sign",
        arity=1,
        kind="elem",
        in_units=("any",),
        out_unit="score",
        build=_b_sign,
        cost=ELEM_COST,
        source=_SRC_A101,
    ),
    Op(
        name="log",
        arity=1,
        kind="elem",
        in_units=("any",),
        out_unit="score",
        build=_b_log,
        cost=ELEM_COST,
        source="gplearn (BSD-3) protected-operator convention, re-derived",
    ),
    Op(
        name="sqrt",
        arity=1,
        kind="elem",
        in_units=("any",),
        out_unit="score",
        build=_b_sqrt,
        cost=ELEM_COST,
        source="gplearn (BSD-3) protected-operator convention, re-derived",
    ),
    Op(
        name="signed_power",
        arity=1,
        kind="elem",
        in_units=("any",),
        out_unit="score",
        build=_b_signed_power,
        params=POWERS,
        cost=ELEM_COST,
        source=_SRC_A101,
    ),
    Op(
        name="sigmoid",
        arity=1,
        kind="elem",
        in_units=("score",),
        out_unit="score",
        build=_b_sigmoid,
        cost=ELEM_COST,
        source=_SRC_PK,
    ),
    Op(
        name="where_positive",
        arity=3,
        kind="elem",
        in_units=("score", "score", "score"),
        out_unit="score",
        build=_b_where_positive,
        cost=ELEM_COST,
        source=_SRC_PK,
    ),
    # -- time series -------------------------------------------------------- #
    Op(
        name="ts_delay",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_delay,
        params=DEFAULT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_delta",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_delta,
        params=DEFAULT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_pctchange",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="ret",
        build=_b_ts_pctchange,
        params=DEFAULT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_mean",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_mean,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_std",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_std,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_sum",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_sum,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_min",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_min,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_max",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_max,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_median",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_median,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_range",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_range,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_skew",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_skew,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_kurt",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_kurt,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_zscore",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_zscore,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_PTA,
    ),
    Op(
        name="ts_ir",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_ir,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_rsv",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_rsv,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_drawdown",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="ratio",
        build=_b_ts_drawdown,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_PK,
    ),
    Op(
        name="ts_rank",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_rank,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_argmax",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_argmax,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_argmin",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_argmin,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_decay_linear",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_decay_linear,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_product",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="score",
        build=_b_ts_product,
        params=(5, 10, 20),
        cost=TS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="ts_ewm_mean",
        arity=1,
        kind="ts",
        in_units=("any",),
        out_unit="any",
        build=_b_ts_ewm_mean,
        params=HALF_LIVES,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    Op(
        name="ts_corr",
        arity=2,
        kind="ts",
        in_units=("any", "any"),
        out_unit="score",
        build=_b_ts_corr,
        params=STAT_WINDOWS,
        cost=TS_COST,
        commutative=True,
        source=_SRC_A101,
    ),
    Op(
        name="ts_beta",
        arity=2,
        kind="ts",
        in_units=("any", "any"),
        out_unit="score",
        build=_b_ts_beta,
        params=STAT_WINDOWS,
        cost=TS_COST,
        source=_SRC_QLIB,
    ),
    # -- cross-sectional ---------------------------------------------------- #
    Op(
        name="cs_rank",
        arity=1,
        kind="xs",
        in_units=("any",),
        out_unit="score",
        build=_b_cs_rank,
        cost=XS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="cs_zscore",
        arity=1,
        kind="xs",
        in_units=("any",),
        out_unit="score",
        build=_b_cs_zscore,
        cost=XS_COST,
        source=_SRC_PK,
    ),
    Op(
        name="cs_demean",
        arity=1,
        kind="xs",
        in_units=("any",),
        out_unit="any",
        build=_b_cs_demean,
        cost=XS_COST,
        source=_SRC_PK,
    ),
    Op(
        name="cs_scale",
        arity=1,
        kind="xs",
        in_units=("any",),
        out_unit="score",
        build=_b_cs_scale,
        cost=XS_COST,
        source=_SRC_A101,
    ),
    Op(
        name="cs_winsor",
        arity=1,
        kind="xs",
        in_units=("any",),
        out_unit="any",
        build=_b_cs_winsor,
        params=WINSOR_K,
        cost=XS_COST,
        source=_SRC_PK,
    ),
    Op(
        name="cs_mad_zscore",
        arity=1,
        kind="xs",
        in_units=("any",),
        out_unit="score",
        build=_b_cs_mad_zscore,
        cost=XS_COST,
        source=_SRC_PK,
    ),
)

#: The primitive set, keyed by operator name.
OPS: dict[str, Op] = {op.name: op for op in _OP_LIST}

if len(OPS) != len(_OP_LIST):  # pragma: no cover - defensive
    raise RuntimeError("duplicate operator name in panelary.evolve._ops")


# --------------------------------------------------------------------------- #
# Type checking / grammar construction
# --------------------------------------------------------------------------- #
def _slot_ok(required: Unit, actual: Unit) -> bool:
    """Return ``True`` if a value of unit ``actual`` may fill a ``required`` slot."""
    if required == "any" or actual == "any":
        return True
    if required in DIMENSIONLESS:
        return actual in DIMENSIONLESS
    return actual == required


def _wildcard_positions(op: Op) -> tuple[int, ...]:
    return tuple(i for i, u in enumerate(op.in_units) if u == "any")


def accepts(op: Op, arg_units: Sequence[Unit]) -> bool:
    """Return ``True`` if ``op`` may be applied to arguments of ``arg_units``.

    Parameters
    ----------
    op : Op
        The candidate operator.
    arg_units : Sequence[Unit]
        The units of the already-typed child expressions, in argument order.

    Returns
    -------
    bool
        ``True`` if the arity matches and every slot is unit-compatible.

    Examples
    --------
    >>> from panelary.evolve._ops import OPS, accepts
    >>> accepts(OPS["add_price"], ("price", "volume"))
    False
    >>> accepts(OPS["add_price"], ("price", "price"))
    True
    >>> accepts(OPS["ts_corr"], ("price", "volume"))
    True
    """
    if len(arg_units) != op.arity:
        return False
    return all(
        _slot_ok(req, act) for req, act in zip(op.in_units, arg_units, strict=False)
    )


def resolve_out_unit(op: Op, arg_units: Sequence[Unit]) -> Unit:
    """Return the unit produced by ``op`` on arguments of ``arg_units``.

    Reading :attr:`Op.out_unit` directly is not enough: ``"any"`` means *the
    unit bound at the wildcard slots*, so ``ts_mean`` of a ``price`` is a
    ``price`` while ``ts_mean`` of a ``ret`` is a ``ret``.

    Parameters
    ----------
    op : Op
        The operator being applied.
    arg_units : Sequence[Unit]
        Units of the child expressions, in argument order.

    Returns
    -------
    Unit
        The concrete output unit.

    Raises
    ------
    ValueError
        If ``arg_units`` is not acceptable for ``op``.
    """
    if not accepts(op, arg_units):
        raise ValueError(
            f"operator {op.name!r} does not accept argument units {tuple(arg_units)!r}"
        )
    if op.out_unit != "any":
        return op.out_unit
    wildcards = _wildcard_positions(op)
    if not wildcards:  # pragma: no cover - no such operator today
        return "any"
    return arg_units[wildcards[0]]


def _possible_outputs(op: Op, available: frozenset[Unit]) -> frozenset[Unit]:
    """Units ``op`` can emit given base/derived units in ``available``."""
    if not available:
        return frozenset()
    # Every non-wildcard slot must be fillable by something available.
    for req in op.in_units:
        if req == "any":
            continue
        if not any(_slot_ok(req, act) for act in available):
            return frozenset()
    wildcards = _wildcard_positions(op)
    if not wildcards:
        return frozenset({op.out_unit}) if op.out_unit != "any" else frozenset()
    if op.out_unit != "any":
        return frozenset({op.out_unit})
    # out_unit == "any": one output unit per unit the wildcards can bind to.
    return frozenset(available)


def ops_by_kind(kind: OpKind) -> tuple[Op, ...]:
    """Return every operator of the given :data:`~._types.OpKind`, name-sorted.

    Parameters
    ----------
    kind : OpKind
        ``"elem"``, ``"ts"`` or ``"xs"``.

    Returns
    -------
    tuple[Op, ...]
        The matching operators, sorted by name for determinism.
    """
    return tuple(sorted((o for o in OPS.values() if o.kind == kind), key=_by_name))


def ops_producing(unit: Unit) -> tuple[Op, ...]:
    """Return every operator that can emit ``unit``, name-sorted.

    An operator with ``out_unit="any"`` is unit-polymorphic and is therefore
    reported as a producer of every unit.

    Parameters
    ----------
    unit : Unit
        The desired output unit. ``"any"`` returns the whole primitive set.

    Returns
    -------
    tuple[Op, ...]
        The matching operators, sorted by name.
    """
    if unit == "any":
        return tuple(sorted(OPS.values(), key=_by_name))
    matches = [
        o
        for o in OPS.values()
        if o.out_unit == "any"
        or o.out_unit == unit
        or (unit in DIMENSIONLESS and o.out_unit in DIMENSIONLESS)
    ]
    return tuple(sorted(matches, key=_by_name))


def default_grammar(units: Sequence[Unit]) -> tuple[Op, ...]:
    """Return the primitive set reachable from base columns of ``units``.

    The result is the least fixed point of "an operator is admissible once all
    of its argument slots can be filled by a base unit or by the output of an
    already-admissible operator". Operators that could never be instantiated on
    this panel (for example :func:`sigmoid` when no dimensionless quantity is
    reachable) are excluded, which keeps the search space free of genes that
    can only ever be dead code.

    Parameters
    ----------
    units : Sequence[Unit]
        Units of the base columns available in the panel.

    Returns
    -------
    tuple[Op, ...]
        Admissible operators, sorted by name.

    Raises
    ------
    ValueError
        If ``units`` is empty.

    Examples
    --------
    >>> from panelary.evolve._ops import default_grammar
    >>> names = {op.name for op in default_grammar(["price", "volume"])}
    >>> "ts_corr" in names and "cs_rank" in names
    True
    """
    if not units:
        raise ValueError("default_grammar requires at least one base unit")
    available: frozenset[Unit] = frozenset(units)
    admissible: dict[str, Op] = {}
    changed = True
    while changed:
        changed = False
        for op in _OP_LIST:
            outputs = _possible_outputs(op, available)
            if not outputs:
                continue
            if op.name not in admissible:
                admissible[op.name] = op
                changed = True
            if not outputs <= available:
                available = available | outputs
                changed = True
    return tuple(sorted(admissible.values(), key=_by_name))


def _by_name(op: Op) -> str:
    return op.name


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _param_schema(op: Op) -> dict[str, type]:
    """Describe an operator's parameter grid for the registry."""
    if not op.params:
        return {}
    if all(isinstance(p, int) for p in op.params):
        return {"window": int}
    return {"param": float}


def _register_ops(specs: Iterable[Op]) -> None:
    """Register one :class:`FeatureSpec` per operator.

    Registry keys are prefixed with ``evolve_`` so the evolve vocabulary cannot
    collide with the ``xs``/``panel``/``ts`` namespaces (both define, for
    instance, a ``cs_zscore``). Every operator is ``panel_safe`` *and*
    ``leakage_safe``: bodies contain no ``.over`` and no forward shift, so the
    compiler is free to apply the partition scope named by ``Op.kind``.

    That same structural property is what makes every one of them
    ``safe_scope="rowwise"``: each is ``series -> series``, emitting one value
    per row from data at or before that row. Contrast the ``ts`` vocabulary,
    which is ``series -> scalar`` and therefore ``"window"`` -- broadcasting a
    whole-series summary back over an entity is a look-ahead by construction.
    The cross-sectional ops (``evolve_cs_*``) are rowwise for the same reason
    the ``xs`` namespace is: a same-date cross-section is data at ``t``, not
    after it. ``tests/test_registry_conformance.py`` checks these claims rather
    than taking them on faith.
    """
    for op in specs:
        registry.register(
            FeatureSpec(
                name=f"evolve_{op.name}",
                namespace="evolve",
                input_shape="series",
                output_shape="series",
                params=_param_schema(op),
                tier="C",
                panel_safe=True,
                leakage_safe=True,
                safe_scope="rowwise",
                source=op.source,
                license=_LICENSE,
                backend_fn=op.build,
            ),
            overwrite=True,
        )


_register_ops(_OP_LIST)
