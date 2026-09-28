"""Frame hygiene: regularising the shape of the time axis.

``reindex``, ``coerce_dtypes``, ``time_to_arange``, ``resample`` and ``trim``.
None of these fit state; they reshape the panel index and are causal by
construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import polars as pl

from panelary.base.transformer import transformer
from panelary.preprocessing._impute import impute


@transformer
def reindex(drop_duplicates: bool = False):
    """Reindexes the entity and time columns to have every possible combination of (entity, time).

    Parameters
    ---------
    drop_duplicates : bool
        Defaults to False. If True, duplicate entity and time values are dropped
        before the cross join, via the exact path of :mod:`panelary.clean`
        (:func:`panelary.clean.exact_unique`). For row-level (and near-)
        deduplication use :class:`panelary.clean.Deduplicator`.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        if drop_duplicates:
            # Imported lazily: panelary.clean builds on preprocessing, not the
            # other way round, so there is no import cycle at load time.
            from panelary.clean._exact import exact_unique

            entities = exact_unique(X.select(entity_col))
            timestamps = exact_unique(X.select(time_col))
        else:
            entities = X.select(entity_col)
            timestamps = X.select(time_col)
        idx = entities.join(timestamps, how="cross")
        X_new = idx.join(X, how="left", on=[entity_col, time_col])
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def coerce_dtypes(schema: Mapping[str, pl.DataType]):
    """Coerces the column datatypes of a DataFrame using the provided schema.

    Parameters
    ----------
    schema : Mapping[str, pl.DataType]
        A dictionary-like object mapping column names to the desired data types.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        X_new = X.with_columns(
            [pl.col(col).cast(dtype) for col, dtype in schema.items()]
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def time_to_arange(eager: bool = False):
    """Coerces time column into arange per entity.

    Assumes even-spaced time-series and homogeneous start dates.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        time_range_expr = (
            pl.int_ranges(0, pl.count(time_col), dtype=pl.UInt32)
            .over(entity_col)
            .alias(time_col)
        )
        other_cols = pl.all().exclude([entity_col, time_col])
        X_new = X.select([entity_col, time_range_expr, other_cols])
        if eager:
            X_new = X_new.collect(streaming=True)
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def resample(freq: str, agg_method: str, impute_method: str | int | float):
    """
    Resamples and transforms a DataFrame using the specified frequency, aggregation method, and imputation method.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    agg_method : str
        The aggregation method to use for resampling. Supported values are 'sum', 'mean', and 'median'.
    impute_method : Union[str, int, float]
        The method used for imputing missing values. If a string, supported values are 'ffill' (forward fill)
        and 'bfill' (backward fill). If an int or float, missing values will be filled with the provided value.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col, target_col = X.columns
        agg_exprs = {
            "sum": pl.sum(target_col),
            "mean": pl.mean(target_col),
            "median": pl.median(target_col),
        }
        X_new = (
            # Defensive resampling
            X.lazy()
            .group_by_dynamic(time_col, every=freq, by=entity_col)
            .agg(agg_exprs[agg_method])
            # Must defensive sort columns otherwise time_col and target_col
            # positions are incorrectly swapped in lazy
            .select([entity_col, time_col, target_col])
            # Impute gaps after reindex
            .pipe(impute(impute_method))
            # Defensive fill null with 0 for impute method `ffill`
            .fill_null(0)
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def trim(direction: Literal["both", "left", "right"] = "both"):
    """Trims time-series in panel to have the same start or end dates as the shortest time-series.

    Parameters
    ----------
    direction : Literal["both", "left", "right"]
        Defaults to "both". If "left" trims from start date of the shortest time series);
        if "right" trims up to the end date of the shortest time-series; or otherwise
        "both" trims between start and end dates of the shortest time-series
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]

        start = pl.col(time_col).min().over(entity_col).max()
        end = pl.col(time_col).max().over(entity_col).min()

        if direction == "both":
            expr = (pl.col(time_col) >= start) & (pl.col(time_col) <= end)
        elif direction == "left":
            expr = pl.col(time_col) >= start
        else:
            expr = pl.col(time_col) <= start
        X_new = X.filter(expr)
        artifacts = {"X_new": X_new}
        return artifacts

    return transform
