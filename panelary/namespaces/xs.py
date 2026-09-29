"""Cross-sectional operators exposed under the ``.xs`` expression namespace.

Cross-sectional (``xs``) operators compare entities **against each other at a
single point in time**. They return plain :class:`polars.Expr` objects with no
grouping of their own; the user composes them with ``.over(date)`` so each
calendar date forms one cross-section.

Cross-sectional operators are inherently **leakage-safe** in the time dimension
(they use only contemporaneous data, never the future) but are *not* panel-safe
in the per-entity sense — by design they mix information across entities within
the same date.

In addition to the expression namespace, this module registers **frame-level**
``.xs`` namespaces on :class:`polars.LazyFrame` and :class:`polars.DataFrame` so
users can operate on a bare frame::

    lf.xs.rank("ret", over="date", normalize=True)   # rank within each date
    df.xs.demean("ret", over="date")

For these frame-level cross-sectional ops the ``over`` cross-section key is
**required** — a cross-sectional operation is meaningless without it.

Architecture / import boundary
------------------------------
This module is part of the ``namespaces`` package — **Tier 1, the Polars-native
extension layer**. It imports ONLY from :mod:`polars`, the Rust plugin,
:mod:`panelary.registry` and the dependency-free leaf
:mod:`panelary.namespaces._neutralize_kernel` (numpy + polars only, which backs
the per-date OLS in :meth:`XSExprNamespace.neutralize`). It must NOT import from
``panelary.core`` or ``panelary.transform`` (the estimator layer),
so the expression layer stays independently splittable into a standalone
``polars-panel`` plugin later. The cross-sectional OLS residual therefore lives
in that leaf and is shared with — not duplicated by —
:class:`panelary.transform.neutralize.Neutralize`, which imports the same
kernel; the import boundary stays intact because the dependency points from the
estimator layer *down* to the leaf, never the other way.

The expression-building logic is factored into module-level ``_expr_*`` helpers
so the expression namespace and both frame-level namespaces share **one**
implementation with no duplication.

Usage
-----
>>> import polars as pl
>>> import panelary.namespaces  # registers the namespace (side-effect)
>>> df = pl.DataFrame(
...     {"date": [1, 1, 1], "entity": ["a", "b", "c"], "ret": [0.1, -0.2, 0.05]}
... )
>>> out = df.with_columns(
...     r=pl.col("ret").xs.rank(normalize=True).over("date")
... )  # doctest: +SKIP

Notes
-----
Importing this module is a side effect: it registers the ``"xs"`` namespace with
Polars (idempotently) and registers each operator as a
:class:`~panelary.registry.FeatureSpec` in the global registry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import polars as pl

from panelary.namespaces._neutralize_kernel import cross_section_residuals
from panelary.registry import FeatureSpec, registry

__all__ = [
    "XSExprNamespace",
    "XSLazyFrameNamespace",
    "XSDataFrameNamespace",
    "register",
]

_LICENSE = "Apache-2.0"
_SOURCE = "Panelary"

#: A LazyFrame or DataFrame — both expose ``with_columns`` identically, so the
#: frame-level cross-sectional operators are generic over the two.
FrameT = TypeVar("FrameT", pl.LazyFrame, pl.DataFrame)

_VALID_RANK_METHODS = ("average", "min", "max", "dense", "ordinal", "random")


# ---------------------------------------------------------------------------
# Shared expression builders — the SINGLE source of truth for each operator's
# expression tree, called by both the expression namespace and the frame-level
# namespaces so there is no duplication.
# ---------------------------------------------------------------------------


#: Recognised string presets for :func:`_expr_rank`'s ``normalize`` argument.
#: ``True`` is an alias for ``"uniform_plus"`` (the original behaviour); ``False``
#: returns the raw rank. See :meth:`XSExprNamespace.rank` for the maps.
_VALID_RANK_NORMALIZE = ("uniform_plus", "unit", "centered", "uniform")


def _expr_rank(
    expr: pl.Expr, *, method: str = "average", normalize: bool | str = False
) -> pl.Expr:
    """Build the cross-sectional rank expression.

    ``normalize`` accepts a ``bool`` (back-compatible) or one of the string
    presets in :data:`_VALID_RANK_NORMALIZE`. With ``r`` the rank in ``1..N`` and
    ``N`` the non-null count of the cross-section:

    =====================  =======================  =========  ================
    ``normalize`` value    map                      support    convention
    =====================  =======================  =========  ================
    ``False`` (default)    ``r``                    ``1..N``   raw rank
    ``True``/uniform_plus  ``r / (N + 1)``          ``(0, 1)`` current behaviour
    ``"unit"``             ``2*(r-1)/(N-1) - 1``    ``[-1,1]`` Gu-Kelly-Xiu 2020
    ``"centered"``         ``r/(N+1) - 0.5``        ``≈±0.5``  centered
    ``"uniform"``          ``r / N``                ``(0,1]``  Freyberger et al.
    =====================  =======================  =========  ================

    The ``"unit"`` preset guards ``N == 1`` (a degenerate one-entity
    cross-section) by returning null, matching :func:`_expr_zscore`.

    See :meth:`XSExprNamespace.rank` for the full description.
    """
    if method not in _VALID_RANK_METHODS:
        raise ValueError(
            f"xs.rank method must be one of {sorted(_VALID_RANK_METHODS)}, "
            f"got {method!r}."
        )
    ranked = expr.rank(method=method)  # type: ignore[arg-type]
    # ``normalize is False`` must be an identity test: ``0 == False`` in Python,
    # and a string preset must not be mistaken for the raw-rank path.
    if normalize is False:
        return ranked
    # Count non-null observations in the cross-section.
    count = expr.is_not_null().sum()
    if normalize is True or normalize == "uniform_plus":
        return ranked / (count + 1)
    if normalize == "unit":
        # Gu-Kelly-Xiu [-1, 1]. N == 1 -> null (division by N-1 == 0).
        denom = count - 1
        scaled = 2 * (ranked - 1) / denom - 1
        return pl.when(denom > 0).then(scaled).otherwise(None)
    if normalize == "centered":
        return ranked / (count + 1) - 0.5
    if normalize == "uniform":
        return ranked / count
    raise ValueError(
        "xs.rank `normalize` must be a bool or one of "
        f"{sorted(_VALID_RANK_NORMALIZE)}, got {normalize!r}."
    )


def _expr_standardize(
    expr: pl.Expr, *, winsor: float | Sequence[float] | None = None
) -> pl.Expr:
    """Build the cross-sectional standardize expression (winsorize -> z-score).

    Composes the existing builders: optionally clip to per-cross-section
    quantiles (:func:`_expr_winsorize`) and then z-score
    (:func:`_expr_zscore`). This is the universal "clean then scale" recipe as a
    single expression; combine with ``.over(date)`` so both the winsor band and
    the mean/std are computed from each date's own contemporaneous cross-section
    (leak-safe, per-date, never global).
    """
    inner = _expr_winsorize(expr, winsor) if winsor is not None else expr
    return _expr_zscore(inner)


def _expr_demean(expr: pl.Expr) -> pl.Expr:
    """Build the cross-sectional demeaning expression (subtract group mean)."""
    return expr - expr.mean()


def _expr_zscore(expr: pl.Expr) -> pl.Expr:
    """Build the cross-sectional z-score expression ``(x - mean) / std``.

    Uses the population standard deviation (``ddof=0``) so a degenerate
    cross-section (one entity) yields ``0/0 -> null`` rather than a spurious
    value, matching :class:`CrossSectionalScaler`. Combine with ``.over(date)``.
    """
    return (expr - expr.mean()) / expr.std(ddof=0)


def _parse_limits(limits: float | Sequence[float]) -> tuple[float, float]:
    """Parse a winsorization ``limits`` argument into ``(low_q, high_q)``.

    A scalar ``p`` is symmetric: ``(p, 1 - p)``. A 2-sequence is taken as the
    explicit ``(low_quantile, high_quantile)`` pair. Both quantiles must satisfy
    ``0 <= low < high <= 1``.
    """
    if isinstance(limits, (int, float)) and not isinstance(limits, bool):
        low = float(limits)
        high = 1.0 - low
    else:
        pair = tuple(limits)
        if len(pair) != 2:
            raise ValueError(
                "xs.winsorize `limits` must be a scalar or a (low, high) pair, "
                f"got {limits!r}."
            )
        low, high = float(pair[0]), float(pair[1])
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(
            "xs.winsorize `limits` must give quantiles with 0 <= low < high <= 1, "
            f"got low={low!r}, high={high!r}."
        )
    return low, high


def _expr_winsorize(expr: pl.Expr, limits: float | Sequence[float]) -> pl.Expr:
    """Build the cross-sectional winsorization expression (clip to quantiles).

    Clips each value to the per-cross-section ``[q_low, q_high]`` quantile band.
    Combine with ``.over(date)`` so each date's band is computed from its own
    contemporaneous cross-section (leak-safe).
    """
    low, high = _parse_limits(limits)
    lower = expr.quantile(low)
    upper = expr.quantile(high)
    return expr.clip(lower_bound=lower, upper_bound=upper)


def _expr_quantile_bin(expr: pl.Expr, q: int) -> pl.Expr:
    """Build the cross-sectional quantile-bucket expression (integer bins).

    Assigns each value to one of ``q`` equal-count buckets ``0 .. q-1`` based on
    its rank within the cross-section, a rank-based quantile binning that is
    robust under ``.over(date)`` (unlike raw break-based cuts on ties/dupes).
    Null inputs map to null.
    """
    if not isinstance(q, int) or isinstance(q, bool) or q < 2:
        raise ValueError(f"xs.quantile_bin `q` must be an integer >= 2, got {q!r}.")
    ranked = expr.rank(method="average")
    count = expr.is_not_null().sum()
    # rank in 1..count -> position in [0, 1); scale into q integer buckets.
    idx = ((ranked - 1) / count * q).floor().clip(0, q - 1)
    return pl.when(expr.is_null()).then(None).otherwise(idx).cast(pl.Int32)


# ---------------------------------------------------------------------------
# Cross-sectional distribution summaries (plan covariance-and-market-state,
# section 5.15). Each one reduces its cross-section to ONE number and
# broadcasts it back over the rows of that cross-section, so under
# ``.over(date)`` every row of a date carries that date's summary.
#
# Shared conventions:
# * the input is cast to Float64 and NaN is treated as missing, exactly like
#   null (``fill_nan(None)``), so a NaN never poisons a whole date;
# * an undefined summary (too few observations, a zero denominator) is null,
#   never NaN and never a fabricated number;
# * quantiles are always ``interpolation="linear"`` (numpy's default), fixed
#   explicitly because the Polars default is ``"nearest"``.
# ---------------------------------------------------------------------------

#: Consistency constant of the MAD under normality: ``1 / Phi^{-1}(3/4)``.
#: Written out rather than computed, because this module may import nothing
#: beyond polars and the registry (see the import boundary above).
_MAD_SCALE = 1.482602218505602

_VALID_DISPERSION = ("sd", "mad", "iqr", "idr")
_VALID_TAILS = ("lower", "upper")
_VALID_ENTROPY = ("share", "hist")


def _as_float(expr: pl.Expr) -> pl.Expr:
    """Cast to Float64 and treat NaN as missing (null)."""
    return expr.cast(pl.Float64).fill_nan(None)


def _linear_quantile(x: pl.Expr, q: float) -> pl.Expr:
    """``q``-quantile with numpy's default (``linear``) interpolation."""
    return x.quantile(q, interpolation="linear")


def _expr_dispersion(expr: pl.Expr, *, kind: str = "sd") -> pl.Expr:
    """Build the cross-sectional dispersion expression.

    ``kind``: ``"sd"`` (sample standard deviation, ``ddof=1``), ``"mad"``
    (median absolute deviation scaled by :data:`_MAD_SCALE`, a consistent
    estimator of the sd under normality), ``"iqr"`` (q75 - q25) or ``"idr"``
    (q90 - q10, the interdecile range). See :meth:`XSExprNamespace.dispersion`.
    """
    if kind not in _VALID_DISPERSION:
        raise ValueError(
            f"xs.dispersion kind must be one of {list(_VALID_DISPERSION)}, "
            f"got {kind!r}."
        )
    x = _as_float(expr)
    if kind == "sd":
        return x.std(ddof=1)
    if kind == "mad":
        return (x - x.median()).abs().median() * _MAD_SCALE
    lo, hi = (0.25, 0.75) if kind == "iqr" else (0.1, 0.9)
    return _linear_quantile(x, hi) - _linear_quantile(x, lo)


def _expr_tail_index(
    expr: pl.Expr,
    *,
    q: float = 0.05,
    tail: str = "lower",
    min_exceedances: int = 10,
) -> pl.Expr:
    """Build the cross-sectional Hill tail-index expression.

    Losses are ``L = -x`` (``tail="lower"``) or ``L = x`` (``"upper"``). With
    ``n`` the non-null count and ``k = floor(q * n)``, the threshold ``u`` is the
    ``(k + 1)``-th largest loss and ``xi = mean(log L_(i) - log u)`` over the
    ``k`` largest losses; the result is ``alpha = 1 / xi``. That is Hill (1975)
    with the order-statistic threshold, i.e. exactly
    :func:`panelary.econ.features._evt.hill_index` with a matched ``k``. Null
    when ``k < min_exceedances``, when ``u <= 0`` (the logarithms are undefined;
    ``hill_index`` shifts the sample instead, a documented fallback this
    operator does not copy) or when ``xi <= 0`` (every exceedance ties ``u``).
    """
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError(f"xs.tail_index `q` must lie in (0, 1), got {q!r}.")
    if tail not in _VALID_TAILS:
        raise ValueError(
            f"xs.tail_index tail must be one of {list(_VALID_TAILS)}, got {tail!r}."
        )
    if (
        isinstance(min_exceedances, bool)
        or not isinstance(min_exceedances, int)
        or min_exceedances < 1
    ):
        raise ValueError(
            "xs.tail_index `min_exceedances` must be an integer >= 1, got "
            f"{min_exceedances!r}."
        )
    x = _as_float(expr)
    loss = -x if tail == "lower" else x
    n = loss.count()
    k = (n.cast(pl.Float64) * q).floor().cast(pl.Int64)
    # Ordinal rank, largest loss first; nulls keep a null rank. Tied losses get
    # distinct ranks, but tied values are equal, so which of them lands on rank
    # k + 1 cannot change u or the sum below.
    order = loss.rank(method="ordinal", descending=True)
    u = pl.when(order == k + 1).then(loss).max()
    xi = pl.when(order <= k).then(loss.log() - u.log()).mean()
    ok = (k >= min_exceedances) & (u > 0.0) & (xi > 0.0)
    return pl.when(ok).then(1.0 / xi).otherwise(None)


def _expr_up_share(expr: pl.Expr) -> pl.Expr:
    """Build the cross-sectional up-share expression ``#(x > 0) / #non-null``."""
    x = _as_float(expr)
    n = x.count()
    return pl.when(n > 0).then((x > 0.0).sum() / n).otherwise(None)


def _expr_entropy(expr: pl.Expr, *, kind: str = "share") -> pl.Expr:
    """Build the cross-sectional normalised-entropy expression.

    ``"share"``: ``-sum(p log p) / log n`` with ``p_i = |x_i| / sum |x|``.
    ``"hist"``: the Shannon entropy of a same-date Freedman-Diaconis histogram,
    divided by ``log(n_bins)``. Both lie in ``[0, 1]``. See
    :meth:`XSExprNamespace.entropy`.
    """
    if kind not in _VALID_ENTROPY:
        raise ValueError(
            f"xs.entropy kind must be one of {list(_VALID_ENTROPY)}, got {kind!r}."
        )
    x = _as_float(expr)
    n = x.count()
    n_f = n.cast(pl.Float64)
    if kind == "share":
        a = x.abs()
        total = a.sum()
        p = a / total
        # 0 * log 0 = 0; a null p (a missing row) falls through to 0 as well.
        plogp = pl.when(p > 0.0).then(p * p.log()).otherwise(0.0)
        h = -plogp.sum()
        return pl.when((n >= 2) & (total > 0.0)).then(h / n_f.log()).otherwise(None)
    # Freedman-Diaconis: bin width 2 * IQR * n^(-1/3), anchored at the date min.
    lo = x.min()
    width = (
        2.0
        * (_linear_quantile(x, 0.75) - _linear_quantile(x, 0.25))
        / n_f.pow(1.0 / 3.0)
    )
    n_bins = ((x.max() - lo) / width).ceil().clip(lower_bound=1.0)
    bin_id = ((x - lo) / width).floor().clip(0.0, n_bins - 1.0)
    # sum_k c_k log c_k over the occupied bins; entropy = log n - that / n.
    counts = bin_id.drop_nulls().unique_counts().cast(pl.Float64)
    h = n_f.log() - (counts * counts.log()).sum() / n_f
    ok = (width > 0.0) & (n_bins >= 2.0)
    return pl.when(ok).then(h / n_bins.log()).otherwise(None)


def _as_list(x: str | Sequence[str]) -> list[str]:
    """Normalise a ``str | Sequence[str]`` to a list of names."""
    return [x] if isinstance(x, str) else list(x)


def _neutralize_residual(
    struct_s: pl.Series, factors: list[str], *, add_intercept: bool
) -> pl.Series:
    """Per-cross-section OLS residual of the target on ``factors`` (leak-safe).

    ``struct_s`` is a struct Series whose first field is the target (aliased
    internally) and whose remaining fields are the factor columns, for the rows
    of a single cross-section (one date). Numeric factors enter directly;
    string/categorical/boolean factors are one-hot encoded per cross-section.
    Rows with a null target or any null numeric factor receive a null residual.

    The arithmetic itself lives in
    :func:`panelary.namespaces._neutralize_kernel.cross_section_residuals`, the
    single implementation shared with
    :class:`panelary.transform.neutralize.Neutralize`.
    """
    df = struct_s.struct.unnest()
    residual = cross_section_residuals(
        df, df.columns[0], factors, add_intercept=add_intercept
    )
    return pl.Series(residual)


def _expr_neutralize(
    target: pl.Expr, by: str | Sequence[str], *, add_intercept: bool = True
) -> pl.Expr:
    """Build the cross-sectional factor-neutralization expression (OLS residual).

    Regresses ``target`` on the ``by`` factor columns across the cross-section
    and keeps the residual. Combine with ``.over(date)`` so each date is
    regressed independently on its own same-date cross-section (Numerai-style
    feature neutralization; leak-safe by construction).
    """
    factors = _as_list(by)
    if not factors:
        raise ValueError("xs.neutralize `by` must name at least one factor column.")
    struct = pl.struct(target.alias("__xs_target__"), *(pl.col(f) for f in factors))
    return struct.map_batches(
        lambda s: _neutralize_residual(s, factors, add_intercept=add_intercept),
        return_dtype=pl.Float64,
    )


def _normalize_columns(columns: str | Sequence[str], *, op: str) -> list[str]:
    """Normalise a ``str | Sequence[str]`` column argument to a list of names."""
    if isinstance(columns, str):
        return [columns]
    cols = list(columns)
    if not cols:
        raise ValueError(
            f"xs.{op}: `columns` was empty; pass a column name or a non-empty "
            "list of column names."
        )
    return cols


def _resolve_output_names(
    cols: list[str], *, alias: str | None, suffix: str | None, op: str
) -> dict[str, str]:
    """Map each input column to its output column name (see panel.py twin)."""
    if alias is not None:
        if suffix is not None:
            raise ValueError(f"xs.{op}: pass either `alias` or `suffix`, not both.")
        if len(cols) != 1:
            raise ValueError(
                f"xs.{op}: `alias` is only valid for a single column; use "
                f"`suffix=` to feature-engineer multiple columns (got {cols})."
            )
        return {cols[0]: alias}
    if suffix is not None:
        return {c: f"{c}{suffix}" for c in cols}
    return {c: c for c in cols}


def _apply_xs(
    frame: FrameT,
    build: Callable[[str], pl.Expr],
    columns: str | Sequence[str],
    *,
    over: str | None,
    alias: str | None,
    suffix: str | None,
    op: str,
) -> FrameT:
    """Apply a cross-sectional op to one or more ``columns`` within each ``over``.

    Shared by the LazyFrame and DataFrame ``.xs`` namespaces. ``over`` (the
    cross-section key, e.g. ``date``) is **required**: cross-sectional ops are
    meaningless without a cross-section, so a missing key raises a clear error.
    ``build`` maps a column name to its (ungrouped) expression; the result is
    grouped by ``over`` and written per :func:`_resolve_output_names`.
    """
    cols = _normalize_columns(columns, op=op)
    if over is None:
        raise ValueError(
            f"xs.{op} requires an `over` cross-section key (e.g. over='date'); "
            "cross-sectional operations are undefined without one."
        )
    names = _resolve_output_names(cols, alias=alias, suffix=suffix, op=op)
    exprs = [build(c).over(over).alias(names[c]) for c in cols]
    return frame.with_columns(exprs)


class XSExprNamespace:
    """Expression methods registered under the ``.xs`` namespace.

    Instances are created by Polars when you access ``expr.xs``; you do not
    construct this class directly. Every method returns a :class:`polars.Expr`
    and is intended to be combined with ``.over(date)``.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rank(
        self, *, method: str = "average", normalize: bool | str = False
    ) -> pl.Expr:
        """Cross-sectional rank of the value within its group.

        Ranks the input across the cross-section (e.g. all entities on one
        date). ``normalize`` rescales the raw ranks into a canonical support:

        * ``False`` (default) -- raw rank ``1..N``.
        * ``True`` / ``"uniform_plus"`` -- ``rank / (N + 1)`` in ``(0, 1)``.
        * ``"unit"`` -- ``2*(r-1)/(N-1) - 1`` in ``[-1, 1]`` (Gu-Kelly-Xiu).
        * ``"centered"`` -- ``rank/(N+1) - 0.5``, centred on ``0``.
        * ``"uniform"`` -- ``rank / N`` in ``(0, 1]`` (Freyberger et al.).

        Parameters
        ----------
        method : str, keyword-only, default "average"
            Tie-handling method forwarded to :meth:`polars.Expr.rank`. One of
            ``"average"``, ``"min"``, ``"max"``, ``"dense"``, ``"ordinal"``,
            ``"random"``.
        normalize : bool | str, keyword-only, default False
            ``bool`` (back-compatible) or a string preset (see above). ``"unit"``
            returns null for a one-entity cross-section. Nulls are ignored in the
            count ``N``.

        Returns
        -------
        pl.Expr
            The cross-sectional rank. Combine with ``.over(date)``.
        """
        return _expr_rank(self._expr, method=method, normalize=normalize)

    def standardize(self, *, winsor: float | Sequence[float] | None = None) -> pl.Expr:
        """Cross-sectional standardize (winsorize -> z-score) within the group.

        The universal "clean then scale" recipe: optionally winsorize to
        per-cross-section quantiles, then z-score. Combine with ``.over(date)``
        so the winsor band and the mean/std come from each date's own
        cross-section (leak-safe, per-date, never global).

        Parameters
        ----------
        winsor : float | Sequence[float] | None, keyword-only, default None
            If given, winsorize first: a scalar ``p`` clips to the ``(p, 1 - p)``
            band, a ``(low, high)`` pair gives the band explicitly. ``None``
            skips winsorization (plain z-score).

        Returns
        -------
        pl.Expr
            The standardized series. Combine with ``.over(date)``.
        """
        return _expr_standardize(self._expr, winsor=winsor)

    def demean(self) -> pl.Expr:
        """Subtract the cross-sectional mean from each value.

        Centers the cross-section so it has zero mean (a market-neutralisation
        style operation when applied to returns ``.over(date)``). Nulls are
        ignored when computing the mean.

        Returns
        -------
        pl.Expr
            The demeaned series. Combine with ``.over(date)``.
        """
        return _expr_demean(self._expr)

    def zscore(self) -> pl.Expr:
        """Cross-sectional z-score ``(x - mean) / std`` within the group.

        Standardizes the value against its contemporaneous cross-section (e.g.
        all entities on one date). Uses the population standard deviation.

        Returns
        -------
        pl.Expr
            The cross-sectional z-score. Combine with ``.over(date)``.
        """
        return _expr_zscore(self._expr)

    def winsorize(self, limits: float | Sequence[float]) -> pl.Expr:
        """Clip values to per-cross-section quantiles (winsorization).

        Parameters
        ----------
        limits : float | Sequence[float]
            A scalar ``p`` clips to the ``(p, 1 - p)`` quantile band; a
            ``(low, high)`` pair gives the band explicitly. Both must satisfy
            ``0 <= low < high <= 1``.

        Returns
        -------
        pl.Expr
            The winsorized series. Combine with ``.over(date)`` so each date's
            band is computed from its own cross-section.
        """
        return _expr_winsorize(self._expr, limits)

    def quantile_bin(self, q: int) -> pl.Expr:
        """Assign each value to one of ``q`` equal-count cross-sectional buckets.

        Rank-based quantile binning: values are bucketed ``0 .. q-1`` by their
        rank within the cross-section, robust to ties under ``.over(date)``.

        Parameters
        ----------
        q : int
            Number of buckets; must be an integer ``>= 2``.

        Returns
        -------
        pl.Expr
            Integer bucket index in ``[0, q)`` (null in, null out). Combine with
            ``.over(date)``.
        """
        return _expr_quantile_bin(self._expr, q)

    def neutralize(
        self, by: str | Sequence[str], *, add_intercept: bool = True
    ) -> pl.Expr:
        """Cross-sectional OLS factor neutralization (keep the residual).

        Regresses this expression (the target) on the ``by`` factor columns
        across the cross-section and returns the residual — the component of the
        target not linearly explained by the factors. This is the Numerai-style
        feature-neutral operator. Numeric factors enter directly;
        string/categorical/boolean factors are one-hot encoded per date.

        Parameters
        ----------
        by : str | Sequence[str]
            Factor column name(s) to neutralize against.
        add_intercept : bool, keyword-only, default True
            Include an intercept, so the residual is also de-meaned per date.

        Returns
        -------
        pl.Expr
            The residualized (factor-neutral) series. Combine with
            ``.over(date)`` so each date is regressed on its own cross-section
            (leak-safe). Rows with a null target or factor yield null.
        """
        return _expr_neutralize(self._expr, by, add_intercept=add_intercept)

    # -- cross-sectional distribution summaries (one value per cross-section) --

    def dispersion(self, *, kind: str = "sd") -> pl.Expr:
        """Cross-sectional dispersion of the value within its group.

        One number per cross-section, broadcast to every row of it. NaN is
        treated as missing; nulls are ignored.

        * ``"sd"`` (default) -- sample standard deviation (``ddof=1``); null
          with fewer than two observations.
        * ``"mad"`` -- ``1.4826 * median(|x - median(x)|)``, the median absolute
          deviation scaled to estimate the sd under normality; robust to a few
          extreme names.
        * ``"iqr"`` -- ``q75 - q25``.
        * ``"idr"`` -- ``q90 - q10``, the interdecile range.

        Quantiles use linear interpolation (numpy's default), set explicitly
        because the Polars default is ``"nearest"``.

        Parameters
        ----------
        kind : {"sd", "mad", "iqr", "idr"}, keyword-only, default "sd"
            The dispersion measure.

        Returns
        -------
        pl.Expr
            The dispersion, Float64. Combine with ``.over(date)`` (or
            ``.over([date, sector])`` for a within-group value). **Evaluated
            without ``.over`` it pools every date of the frame**, so each value
            depends on the whole future: a look-ahead.
        """
        return _expr_dispersion(self._expr, kind=kind)

    def tail_index(
        self, *, q: float = 0.05, tail: str = "lower", min_exceedances: int = 10
    ) -> pl.Expr:
        """Hill (1975) tail index ``alpha`` of one cross-section.

        With losses ``L = -x`` (``tail="lower"``) or ``L = x`` (``"upper"``),
        ``n`` the number of non-null values and ``k = floor(q * n)``, the
        threshold ``u`` is the ``(k + 1)``-th largest loss and::

            xi = mean(log L_(i) - log u)   over the k largest losses
            alpha = 1 / xi

        That is :func:`panelary.econ.features._evt.hill_index` with ``k``
        matched, so the two agree to rounding; ``alpha`` is the Pareto tail
        exponent (smaller means a heavier tail). The threshold is an order
        statistic, never an interpolated quantile, which is what makes it the
        Hill estimator exactly.

        Parameters
        ----------
        q : float, keyword-only, default 0.05
            Tail fraction in ``(0, 1)``: the share of the cross-section treated
            as exceedances.
        tail : {"lower", "upper"}, keyword-only, default "lower"
            Which tail; ``"lower"`` (losses) is the convention for returns.
        min_exceedances : int, keyword-only, default 10
            Fewer exceedances than this gives null. With ``q=0.05`` a date needs
            at least 200 names for a value.

        Returns
        -------
        pl.Expr
            ``alpha``, Float64; null when ``k < min_exceedances``, when the
            threshold loss ``u`` is not positive (the logarithms are undefined),
            or when every exceedance ties ``u``. Combine with ``.over(date)``.
        """
        return _expr_tail_index(
            self._expr, q=q, tail=tail, min_exceedances=min_exceedances
        )

    def up_share(self) -> pl.Expr:
        """Share of the cross-section that is strictly positive.

        ``#(x > 0) / #non-null``: market breadth when applied to returns. A zero
        counts as observed but not up. NaN is treated as missing.

        Returns
        -------
        pl.Expr
            The up-share in ``[0, 1]``, Float64; null for an empty
            cross-section. Combine with ``.over(date)``.
        """
        return _expr_up_share(self._expr)

    def entropy(self, *, kind: str = "share") -> pl.Expr:
        """Normalised Shannon entropy of one cross-section, in ``[0, 1]``.

        * ``"share"`` (default) -- ``-sum(p log p) / log n`` with
          ``p_i = |x_i| / sum |x|``: how evenly the absolute moves are spread
          across names. ``1`` means every name moved by the same amount; near
          ``0`` means a few names carried the whole move. Null with fewer than
          two observations or when every value is zero.
        * ``"hist"`` -- the entropy of a same-date Freedman-Diaconis histogram
          (bin width ``2 * IQR * n**(-1/3)``, anchored at the date minimum),
          divided by ``log(n_bins)``: how close the cross-section is to uniform
          over its own range. Heavy tails stretch the range over mostly empty
          bins and pull it down. Null when the IQR is zero or there is only one
          bin.

        NaN is treated as missing. The registry name is ``xs_entropy``; the bare
        ``entropy`` is left to per-entity (time-series) entropies.

        Parameters
        ----------
        kind : {"share", "hist"}, keyword-only, default "share"
            The distribution the entropy is taken over.

        Returns
        -------
        pl.Expr
            The normalised entropy, Float64. Combine with ``.over(date)``.
        """
        return _expr_entropy(self._expr, kind=kind)


class _XSFrameNamespace:
    """Shared frame-level ``.xs`` operators for LazyFrame and DataFrame.

    Each method names the target ``columns`` (a single name or a list) first,
    takes a **required** ``over`` cross-section key (e.g. ``date``), and controls
    output naming via ``alias`` (single column) or ``suffix``
    (``f"{col}{suffix}"`` per column; with neither, replace in place). The
    expression is built by the shared ``_expr_*`` helpers, so nothing is
    duplicated with the expression namespace.

    LazyFrame and DataFrame share one implementation via :func:`_apply_xs`.
    """

    _frame: pl.LazyFrame | pl.DataFrame

    def rank(
        self,
        columns: str | Sequence[str],
        *,
        over: str | None = None,
        method: str = "average",
        normalize: bool | str = False,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Cross-sectional rank of ``columns`` within each ``over`` group.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to rank.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required** — raises
            :class:`ValueError` if omitted.
        method : str, keyword-only, default "average"
            Tie-handling method (``"average"``, ``"min"``, ``"max"``,
            ``"dense"``, ``"ordinal"``, ``"random"``).
        normalize : bool | str, keyword-only, default False
            ``bool`` (back-compatible; ``True`` -> ``rank / (count + 1)``) or a
            string preset ``"unit"`` (``[-1, 1]``), ``"centered"`` (``±0.5``),
            ``"uniform"`` (``(0, 1]``), or ``"uniform_plus"`` (alias of ``True``).
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s) (same type as the input).
        """
        return _apply_xs(
            self._frame,
            lambda c: _expr_rank(pl.col(c), method=method, normalize=normalize),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="rank",
        )

    def demean(
        self,
        columns: str | Sequence[str],
        *,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Subtract the cross-sectional mean of ``columns`` within each ``over``.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to demean.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required** — raises
            :class:`ValueError` if omitted.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        return _apply_xs(
            self._frame,
            lambda c: _expr_demean(pl.col(c)),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="demean",
        )

    def zscore(
        self,
        columns: str | Sequence[str],
        *,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Cross-sectional z-score of ``columns`` within each ``over`` group.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to standardize.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required**.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"`` (e.g. ``"_z"``).

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        return _apply_xs(
            self._frame,
            lambda c: _expr_zscore(pl.col(c)),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="zscore",
        )

    def standardize(
        self,
        columns: str | Sequence[str],
        *,
        over: str | None = None,
        winsor: float | Sequence[float] | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Cross-sectional standardize of ``columns`` within each ``over`` group.

        Sugar for the "clean then scale" recipe: optionally winsorize to
        per-cross-section quantiles, then z-score. One call replaces a
        ``winsorize`` followed by a ``zscore``; both steps are computed per
        ``over`` group (leak-safe, never global).

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to standardize.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required**.
        winsor : float | Sequence[float] | None, keyword-only, default None
            If given, winsorize first: a scalar ``p`` -> ``(p, 1 - p)`` band, or
            an explicit ``(low, high)`` quantile pair. ``None`` = plain z-score.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"`` (e.g. ``"_z"``).

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        return _apply_xs(
            self._frame,
            lambda c: _expr_standardize(pl.col(c), winsor=winsor),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="standardize",
        )

    def winsorize(
        self,
        columns: str | Sequence[str],
        *,
        limits: float | Sequence[float],
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Clip ``columns`` to per-cross-section quantiles within each ``over``.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to winsorize.
        limits : float | Sequence[float], keyword-only
            Scalar ``p`` -> ``(p, 1 - p)`` band, or an explicit ``(low, high)``
            quantile pair.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required**.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s).
        """
        return _apply_xs(
            self._frame,
            lambda c: _expr_winsorize(pl.col(c), limits),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="winsorize",
        )

    def quantile_bin(
        self,
        columns: str | Sequence[str],
        *,
        q: int,
        over: str | None = None,
        alias: str | None = None,
        suffix: str | None = None,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Bucket ``columns`` into ``q`` per-cross-section quantile bins.

        Parameters
        ----------
        columns : str | Sequence[str]
            Column name, or list of column names, to bin.
        q : int, keyword-only
            Number of equal-count buckets; integer ``>= 2``.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required**.
        alias : str | None, keyword-only, default None
            Output column name (single-column only). Defaults to in place.
        suffix : str | None, keyword-only, default None
            If given, write each output to ``f"{col}{suffix}"``.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the new/replaced column(s) (integer bin indices).
        """
        return _apply_xs(
            self._frame,
            lambda c: _expr_quantile_bin(pl.col(c), q),
            columns,
            over=over,
            alias=alias,
            suffix=suffix,
            op="quantile_bin",
        )

    def neutralize(
        self,
        columns: str | Sequence[str],
        *,
        by: str | Sequence[str],
        over: str | None = None,
        add_intercept: bool = True,
        alias: str | None = None,
        suffix: str | None = "_neutral",
    ) -> pl.LazyFrame | pl.DataFrame:
        """Cross-sectional OLS factor-neutralize ``columns`` within each ``over``.

        Regresses each target column on the ``by`` factor columns across each
        cross-section and keeps the residual (Numerai-style feature neutral).

        Parameters
        ----------
        columns : str | Sequence[str]
            Target column name(s) to neutralize.
        by : str | Sequence[str], keyword-only
            Factor column(s) to neutralize against.
        over : str, keyword-only
            Cross-section key (e.g. ``"date"``). **Required**.
        add_intercept : bool, keyword-only, default True
            Include an intercept (also de-means the residual per date).
        alias : str | None, keyword-only, default None
            Output column name (single-column only).
        suffix : str | None, keyword-only, default "_neutral"
            Written to ``f"{col}{suffix}"``. The default keeps the original
            column and adds ``<col>_neutral``; pass ``suffix=""`` to overwrite.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame
            The frame with the residualized column(s).
        """
        # This op carries a non-None default ``suffix`` ("_neutral"); an explicit
        # ``alias`` overrides it (rather than tripping the exclusivity guard).
        effective_suffix = None if alias is not None else suffix
        return _apply_xs(
            self._frame,
            lambda c: _expr_neutralize(pl.col(c), by, add_intercept=add_intercept),
            columns,
            over=over,
            alias=alias,
            suffix=effective_suffix,
            op="neutralize",
        )


class XSLazyFrameNamespace(_XSFrameNamespace):
    """Frame-level ``.xs`` operators on :class:`polars.LazyFrame`.

    Accessed as ``lf.xs.<op>(...)``. Returns a new :class:`polars.LazyFrame`.
    """

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._frame = lf


class XSDataFrameNamespace(_XSFrameNamespace):
    """Frame-level ``.xs`` operators on :class:`polars.DataFrame`.

    Accessed as ``df.xs.<op>(...)``. Returns a new :class:`polars.DataFrame`.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._frame = df


def register() -> None:
    """Register the ``.xs`` namespace and its feature specs (idempotent).

    Registration is guarded so that re-importing this module does not raise the
    Polars "namespace already registered" error.
    """
    # Only register when absent to avoid the "overriding existing custom
    # namespace" UserWarning on re-import; the try/except guards the documented
    # "already registered" condition across Polars versions.
    if not _is_registered(pl.Expr, "xs"):
        try:
            pl.api.register_expr_namespace("xs")(XSExprNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    # Frame-level ergonomics: ``lf.xs.<op>`` / ``df.xs.<op>``. Guarded the same
    # way to keep re-import a no-op and avoid the override UserWarning.
    if not _is_registered(pl.LazyFrame, "xs"):
        try:
            pl.api.register_lazyframe_namespace("xs")(XSLazyFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise
    if not _is_registered(pl.DataFrame, "xs"):
        try:
            pl.api.register_dataframe_namespace("xs")(XSDataFrameNamespace)
        except Exception as exc:  # pragma: no cover - exact type varies by version
            if "already" not in str(exc).lower():
                raise

    for spec in _SPECS:
        registry.register(spec)


def _is_registered(cls: type, name: str) -> bool:
    """Return ``True`` if a custom namespace ``name`` already exists on ``cls``."""
    return hasattr(cls, name)


# Every operator in this namespace is ``safe_scope="rowwise"``. Each reads
# **only its own cross-section** — the rows sharing the row's own date, once
# composed with the ``.over(time)`` this namespace is defined by. The statistic
# each one needs is drawn from that same-date group and nothing else: ``rank``
# takes the rank and the non-null count within the group; ``demean`` /
# ``cs_zscore`` / ``standardize`` its mean and std; ``winsorize`` and
# ``quantile_bin`` its quantiles and count; ``neutralize`` regresses on the
# factor columns of that group alone (``map_batches`` is handed one date's rows
# at a time). None of them reaches to another date, so the value at ``t`` is a
# function of data at exactly ``t`` — trivially ``<= t``, and evaluating them at
# every row is the intended usage.
#
# The dependence that *is* there runs across entities, not across time: adding
# an entity to a date changes that date's ranks and moments. That is the
# defining behaviour of a cross-sectional operator, it is what ``panel_safe``
# exists to record, and it is not a look-ahead.
#
# The ``.over(time)`` is load-bearing, exactly as ``.over(entity)`` is for the
# ``panel`` namespace. Evaluated bare, these pool the whole panel: the "cross
# section" then spans every date at once and each value depends on the entire
# future (measured — all five nullary ops fail ``assert_no_lookahead`` and
# breach prefix invariance without it). The scope is claimed for the documented
# composition, not for a bare call.
_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="rank",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        # `normalize` is bool | str (preset), so its schema type is `object`.
        params={"method": str, "normalize": object},
        tier="A",
        panel_safe=False,  # cross-sectional: deliberately mixes entities
        leakage_safe=True,  # uses only contemporaneous data
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    # Cross-sectional standardize (winsorize -> z-score). Its bare name
    # "standardize" is unique in the registry (which keys by name only).
    FeatureSpec(
        name="standardize",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"winsor": object},
        tier="A",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="demean",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={},
        tier="A",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    # Cross-sectional z-score. NOTE: the frame-level feature registry keys specs
    # by bare ``name`` (namespace-agnostic), so this cannot reuse the name
    # ``"zscore"`` already taken by the per-entity ``panel.zscore`` — it is
    # registered as ``"cs_zscore"`` (the *method* is still ``.xs.zscore``).
    FeatureSpec(
        name="cs_zscore",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="winsorize",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"limits": float},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="quantile_bin",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"q": int},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    FeatureSpec(
        name="neutralize",
        namespace="xs",
        input_shape="frame",
        output_shape="series",
        params={"by": list, "add_intercept": bool},
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
    ),
    # Cross-sectional distribution summaries (plan covariance-and-market-state,
    # section 4.4). Registered under ``xs_``-prefixed names because the registry
    # keys by bare name and per-entity siblings (an ``entropy`` of one series,
    # the econ ``tail_index`` family) want the short names; the *methods* are
    # ``.xs.dispersion`` / ``.xs.tail_index`` / ``.xs.up_share`` /
    # ``.xs.entropy``. Same scope argument as the block comment above: under
    # ``.over(time)`` each value is a function of its own date's rows only.
    FeatureSpec(
        name="xs_dispersion",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"kind": str},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Panelary; MAD consistency constant 1/Phi^-1(3/4)",
        license=_LICENSE,
        cost_hint="O(N log N) per date",
    ),
    FeatureSpec(
        name="xs_tail_index",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"q": float, "tail": str, "min_exceedances": int},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Hill (1975), Annals of Statistics 3(5)",
        license=_LICENSE,
        cost_hint="O(N log N) per date",
    ),
    FeatureSpec(
        name="xs_up_share",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source=_SOURCE,
        license=_LICENSE,
        cost_hint="O(N) per date",
    ),
    FeatureSpec(
        name="xs_entropy",
        namespace="xs",
        input_shape="series",
        output_shape="series",
        params={"kind": str},
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Shannon (1948); Freedman & Diaconis (1981) bins",
        license=_LICENSE,
        cost_hint="O(N log N) per date",
    ),
)


register()
