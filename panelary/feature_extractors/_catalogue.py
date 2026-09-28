"""The ``ts`` scalar-aggregation catalogue and the bulk ``extract_features``.

Importing this module registers one :class:`~panelary.registry.FeatureSpec` per
scalar aggregation with the process-wide registry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import polars as pl

from panelary.registry import FeatureSpec, registry

# Imported for its side effect: the builders below resolve `expr.ts.<name>`,
# so the `ts` namespace must be registered before any of them is called.
from ._namespace import FeatureExtractor  # noqa: F401

# ---------------------------------------------------------------------------
# Catalogue of ``.ts`` scalar-aggregation extractors + bulk ``extract_features``
# ---------------------------------------------------------------------------
#
# Every entry below is a *pure Polars expression* that reduces a per-entity time
# series to a single scalar, so it composes cleanly under ``group_by(entity)``
# (and, equivalently, ``.over(entity)``) without leaking information across
# entities or across time. That makes each one both ``panel_safe`` and
# ``leakage_safe``. Extractors that emit a list/struct (e.g. ``fft_coefficients``,
# ``linear_trend``, ``energy_ratios``, ``streak_length_stats``) or that require a
# Rust plugin (e.g. ``frac_diff``, ``cusum``, ``lempel_ziv_complexity``) are
# intentionally excluded from the bulk path — they do not reduce to one column.
#
# ``params`` records the tunable arguments (all with sensible defaults, so the
# bulk :func:`extract_features` path can invoke them argument-free).

#: Ordered mapping of registered ``ts`` feature name -> its ``params`` schema.
#: The order is stable so :func:`extract_features` and the catalogue render
#: deterministically.
_TS_SCALAR_AGG_SPECS: dict[str, dict[str, Any]] = {
    "absolute_energy": {},
    "absolute_maximum": {},
    "absolute_sum_of_changes": {},
    "root_mean_square": {},
    "benford_correlation": {},
    "count_above_mean": {},
    "count_below_mean": {},
    "first_location_of_maximum": {},
    "first_location_of_minimum": {},
    "has_duplicate": {},
    "has_duplicate_max": {},
    "has_duplicate_min": {},
    "last_location_of_maximum": {},
    "last_location_of_minimum": {},
    "mean_abs_change": {},
    "max_abs_change": {},
    "mean_change": {},
    "mean_second_derivative_central": {},
    "percent_reoccurring_points": {},
    "percent_reoccurring_values": {},
    "sum_reoccurring_points": {},
    "sum_reoccurring_values": {},
    "variation_coefficient": {},
    "harmonic_mean": {},
    "range_over_mean": {},
    "longest_streak_above_mean": {},
    "longest_streak_below_mean": {},
    "longest_winning_streak": {},
    "longest_losing_streak": {},
    "ratio_n_unique_to_length": {},
    "max_drawdown": {},
    "num_direction_changes": {},
    "return_kurtosis": {},
    "return_skew": {},
    "count_above": {"threshold": float},
    "count_below": {"threshold": float},
    "ratio_beyond_r_sigma": {"ratio": float},
    "large_standard_deviation": {"ratio": float},
    "var_gt_std": {"ddof": int},
    "symmetry_looking": {"ratio": float},
    "range_change": {"percentage": bool},
    "cid_ce": {"normalize": bool},
}


def _ts_scalar_agg_builder(name: str) -> Callable[[pl.Expr], pl.Expr]:
    """Return a builder that maps ``pl.col(c)`` -> ``pl.col(c).ts.<name>()``.

    Parametrised extractors are invoked with their defaults, so every builder is
    a plain ``Expr -> Expr`` reduction suitable for ``group_by(...).agg(...)``.
    """

    def _build(expr: pl.Expr) -> pl.Expr:
        return getattr(expr.ts, name)()

    _build.__name__ = f"ts_{name}"
    _build.__qualname__ = f"ts.{name}"
    return _build


#: Ordered mapping of feature name -> ``Expr -> Expr`` scalar-aggregation builder.
_TS_SCALAR_AGGS: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    name: _ts_scalar_agg_builder(name) for name in _TS_SCALAR_AGG_SPECS
}


# Register every scalar-aggregation extractor with the process-wide registry so
# the ``.ts`` catalogue becomes machine-introspectable (previously it was
# invisible to :class:`~panelary.registry.FeatureRegistry`). These are all
# pure per-entity aggregations: panel-safe, and leakage-safe *at window scope*.
#
# ``safe_scope="window"`` is forced by the shape, not by a test. Every entry
# here is ``series -> scalar``: it reduces an entity's whole series to one
# number. As the summary of an already-delimited window — which is exactly what
# :func:`extract_features` computes, one row per entity — that is perfectly
# causal. Broadcast back per row with ``pl.col(c).ts.<name>().over(entity)``,
# the value at *every* ``t`` is a function of the entity's whole series
# including its future, so it is a look-ahead **by construction**. No test is
# needed to establish that and none should be trusted to refute it: a
# perturbation verifier returns a vacuous pass whenever the statistic is
# degenerate on the sample (measured: all 10 of the 42 that "pass"
# ``assert_no_lookahead`` are constant across every row of the probe panel), so
# a green result here means the probe data was uninformative, not that the
# operator is row-safe.
for _name, _params in _TS_SCALAR_AGG_SPECS.items():
    registry.register(
        FeatureSpec(
            name=_name,
            namespace="ts",
            input_shape="series",
            output_shape="scalar",
            params=dict(_params),
            tier="B",
            panel_safe=True,
            leakage_safe=True,
            safe_scope="window",
            source="Panelary",
            license="Apache-2.0",
            backend_fn=_TS_SCALAR_AGGS[_name],
        ),
        overwrite=True,
    )
del _name, _params


def extract_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    features: str | Sequence[str] = "all",
    column: str | Sequence[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Compute registered ``ts`` scalar features per entity in a single lazy pass.

    This is a tsfresh-style bulk extractor, but lazy and panel-safe: it runs one
    ``group_by(entity).agg(...)`` over the requested value column(s), producing
    exactly one row per entity with one column per (value column x feature).
    Because every feature is a pure per-entity aggregation, no information leaks
    across entities or from the future.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Long-format panel. A ``DataFrame`` is processed lazily and collected on
        return; a ``LazyFrame`` stays lazy end-to-end.
    entity : str, optional
        Entity (group) column. Defaults to the first column, matching the repo
        convention.
    time : str, optional
        Time column. Defaults to the second column. It is not aggregated; it is
        only used to exclude it from the auto-selected value columns.
    features : str | Sequence[str], default "all"
        Either ``"all"`` (every registered ``ts`` scalar aggregation) or an
        explicit list of registered feature names to subset.
    column : str | Sequence[str], optional
        Value column(s) to featurise. Defaults to every numeric column that is
        neither ``entity`` nor ``time``. With a single value column the output
        columns are named ``<feature>``; with several they are namespaced as
        ``<column>__<feature>``.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        One row per entity. Lazy in, lazy out.

    Raises
    ------
    ValueError
        If a requested feature is not a registered ``ts`` scalar aggregation, or
        if no value columns can be resolved.
    """
    lazy_in = isinstance(df, pl.LazyFrame)
    lf = df if lazy_in else df.lazy()

    schema = lf.collect_schema()
    names = schema.names()
    if not names:
        raise ValueError("extract_features received a frame with no columns.")

    if entity is None:
        entity = names[0]
    if time is None:
        time = names[1] if len(names) > 1 else None

    if column is None:
        reserved = {entity} | ({time} if time is not None else set())
        value_cols = [c for c in names if c not in reserved and schema[c].is_numeric()]
        if not value_cols:
            raise ValueError(
                "extract_features could not find any numeric value column to "
                f"featurise (entity={entity!r}, time={time!r}). Pass `column=`."
            )
    elif isinstance(column, str):
        value_cols = [column]
    else:
        value_cols = list(column)

    if isinstance(features, str):
        feat_names = list(_TS_SCALAR_AGGS) if features == "all" else [features]
    else:
        feat_names = list(features)

    unknown = [f for f in feat_names if f not in _TS_SCALAR_AGGS]
    if unknown:
        available = ", ".join(_TS_SCALAR_AGGS)
        raise ValueError(
            f"Unknown ts feature(s) {unknown!r}. "
            f"Registered scalar aggregations: {available}."
        )

    multi = len(value_cols) > 1
    exprs: list[pl.Expr] = []
    for col in value_cols:
        for fname in feat_names:
            alias = f"{col}__{fname}" if multi else fname
            exprs.append(_TS_SCALAR_AGGS[fname](pl.col(col)).alias(alias))

    out = lf.group_by(entity, maintain_order=True).agg(*exprs)
    return out if lazy_in else out.collect()
