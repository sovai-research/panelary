"""The catch22 catalogue and the aggregate entry points.

Holds the name -> function mapping, the canonical name orderings and the
three user-facing entry points (:func:`catch22_all`,
:func:`catch22_features`, :func:`catch22_all_expr`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ._batch import catch22_rows
from ._features import (
    CO_Embed2_Dist_tau_d_expfit_meandiff,
    CO_f1ecac,
    CO_FirstMin_ac,
    CO_HistogramAMI_even_2_5,
    CO_trev_1_num,
    DN_HistogramMode_5,
    DN_HistogramMode_10,
    DN_OutlierInclude_n_001_mdrmd,
    DN_OutlierInclude_p_001_mdrmd,
    FC_LocalSimple_mean1_tauresrat,
    FC_LocalSimple_mean3_stderr,
    IN_AutoMutualInfoStats_40_gaussian_fmmi,
    MD_hrv_classic_pnn40,
    PD_PeriodicityWang_th0_01,
    SB_BinaryStats_diff_longstretch0,
    SB_BinaryStats_mean_longstretch1,
    SB_MotifThree_quantile_hh,
    SB_TransitionMatrix_3ac_sumdiagcov,
    SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    SP_Summaries_welch_rect_area_5_1,
    SP_Summaries_welch_rect_centroid,
)
from ._helpers import _as_1d

# ---------------------------------------------------------------------------
# Registry & aggregate entry points
# ---------------------------------------------------------------------------
CATCH22_FUNCS = {
    "DN_HistogramMode_5": DN_HistogramMode_5,
    "DN_HistogramMode_10": DN_HistogramMode_10,
    "CO_f1ecac": CO_f1ecac,
    "CO_FirstMin_ac": CO_FirstMin_ac,
    "CO_HistogramAMI_even_2_5": CO_HistogramAMI_even_2_5,
    "CO_trev_1_num": CO_trev_1_num,
    "CO_Embed2_Dist_tau_d_expfit_meandiff": CO_Embed2_Dist_tau_d_expfit_meandiff,
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": IN_AutoMutualInfoStats_40_gaussian_fmmi,
    "MD_hrv_classic_pnn40": MD_hrv_classic_pnn40,
    "SB_BinaryStats_mean_longstretch1": SB_BinaryStats_mean_longstretch1,
    "SB_BinaryStats_diff_longstretch0": SB_BinaryStats_diff_longstretch0,
    "SB_MotifThree_quantile_hh": SB_MotifThree_quantile_hh,
    "SB_TransitionMatrix_3ac_sumdiagcov": SB_TransitionMatrix_3ac_sumdiagcov,
    "PD_PeriodicityWang_th0_01": PD_PeriodicityWang_th0_01,
    "FC_LocalSimple_mean1_tauresrat": FC_LocalSimple_mean1_tauresrat,
    "FC_LocalSimple_mean3_stderr": FC_LocalSimple_mean3_stderr,
    "DN_OutlierInclude_p_001_mdrmd": DN_OutlierInclude_p_001_mdrmd,
    "DN_OutlierInclude_n_001_mdrmd": DN_OutlierInclude_n_001_mdrmd,
    "SP_Summaries_welch_rect_area_5_1": SP_Summaries_welch_rect_area_5_1,
    "SP_Summaries_welch_rect_centroid": SP_Summaries_welch_rect_centroid,
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
}

#: Canonical ordering of the 22 catch22 feature names.
CATCH22_NAMES: tuple[str, ...] = tuple(CATCH22_FUNCS)

#: Extra features added by the "catch24" variant (raw mean and spread).
CATCH24_EXTRA_NAMES: tuple[str, ...] = ("DN_Mean", "DN_Spread_Std")


def _resolve_names(which, catch24: bool) -> list[str]:
    if which == "all":
        names = list(CATCH22_NAMES)
    elif isinstance(which, (list, tuple)):
        names = []
        for w in which:
            if w not in CATCH22_FUNCS:
                raise ValueError(
                    f"Unknown catch22 feature {w!r}. Valid names: {CATCH22_NAMES}"
                )
            names.append(w)
    else:
        raise TypeError("`which` must be 'all' or a list/tuple of feature names.")
    if catch24 and which == "all":
        names += list(CATCH24_EXTRA_NAMES)
    return names


def _compute(x: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    x = _as_1d(x)
    values = catch22_rows(x[None, :], names)[0]
    return {name: float(v) for name, v in zip(names, values, strict=True)}


def catch22_batch(X, *, which="all", catch24: bool = False) -> np.ndarray:
    """Compute catch22 (or catch24) for every row of a 2-D array in one pass.

    The batched counterpart of :func:`catch22_all`: the same features, the same
    per-feature NaN-on-failure contract, but computed with one set of NumPy
    calls for the whole batch instead of one per window. This is what makes
    catch22 usable as an embedding over rolling windows.

    Parameters
    ----------
    X : array-like
        ``(n_windows, L)`` array; each row is one series. 1-D input is treated
        as a single row. A row containing NaN is computed on its NaN-dropped
        values, exactly like :func:`catch22_all`.
    which : {"all"} or list of str, default "all"
        ``"all"`` for the canonical 22 features, or an explicit list of names.
    catch24 : bool, default False
        When ``which="all"``, also emit ``DN_Mean`` and ``DN_Spread_Std``.

    Returns
    -------
    numpy.ndarray
        ``(n_windows, n_features)`` float64, columns in canonical order (see
        :data:`CATCH22_NAMES`). Row ``i`` depends only on ``X[i]`` -- never on
        which other rows share the batch -- and equals ``catch22_all(X[i])``.

    Raises
    ------
    ValueError
        If ``X`` is not 1-D or 2-D, or ``which`` names an unknown feature.
    """
    names = _resolve_names(which, catch24)
    return catch22_rows(X, names)


def catch22_all(x, *, catch24: bool = False) -> dict[str, float]:
    """Compute the full catch22 (or catch24) feature set for one 1-D series.

    Parameters
    ----------
    x : array-like
        A single 1-D time series (list, :class:`numpy.ndarray` or
        :class:`polars.Series`).  The series is z-scored internally.
    catch24 : bool, default False
        If ``True``, additionally return the raw mean (``DN_Mean``) and sample
        standard deviation (``DN_Spread_Std``), yielding 24 features.

    Returns
    -------
    dict of str -> float
        Mapping from feature name to value, in canonical order.
    """
    names = _resolve_names("all", catch24)
    return _compute(x, names)


def catch22_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    column: str,
    which="all",
    catch24: bool = False,
) -> pl.DataFrame:
    """Compute catch22 features per entity for a long-format panel.

    The features are computed independently for each entity's sorted series,
    which makes the result leak-safe (no cross-entity or future information
    enters any feature).

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long-format panel.  A ``LazyFrame`` is collected internally.
    entity : str, optional
        Column identifying the entity/series.  If ``None``, the whole frame is
        treated as a single series and the result has one row.
    time : str, optional
        Column used to sort each entity's series in ascending order before
        feature extraction.  If ``None``, the existing row order is used.
    column : str
        Name of the value column to extract features from.
    which : {"all"} or list of str, default "all"
        Either ``"all"`` (the canonical 22 features) or an explicit list of
        feature names to compute.
    catch24 : bool, default False
        When ``which="all"``, also emit ``DN_Mean`` and ``DN_Spread_Std``.

    Returns
    -------
    polars.DataFrame
        One row per entity (or a single row when ``entity is None``), with the
        entity column (if any) followed by one column per requested feature.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    names = _resolve_names(which, catch24)

    if entity is None:
        sub = df.sort(time) if time is not None else df
        x = sub.get_column(column).to_numpy()
        return pl.DataFrame([_compute(x, names)])

    # Route through the lazy struct-expr path so Polars parallelises the
    # per-entity feature computation across threads (the former Python `for`
    # loop over groups was single-threaded). Sorting by [entity, time] yields
    # each group in time order (equivalent to the old per-group `sub.sort`);
    # the original first-appearance entity ordering is restored afterwards so
    # the output is row-for-row identical.
    sort_keys = [entity, time] if time is not None else [entity]
    result = (
        df.lazy()
        .sort(sort_keys, maintain_order=True)
        .group_by(entity, maintain_order=True)
        .agg(catch22_all_expr(column, which=which, catch24=catch24, alias="__c22"))
        .unnest("__c22")
        .collect()
    )

    order = (
        df.select(pl.col(entity)).unique(maintain_order=True).with_row_index("__ord")
    )
    return result.join(order, on=entity, how="left").sort("__ord").drop("__ord")


def catch22_all_expr(
    column: str,
    *,
    which="all",
    catch24: bool = False,
    alias: str = "catch22",
) -> pl.Expr:
    """Return a Polars expression producing a per-group catch22 struct.

    Designed for use inside ``group_by(entity).agg(...)`` so that features are
    computed per entity (leak-safe).  Unnest the resulting struct column to get
    one column per feature::

        (
            df.group_by("entity")
              .agg(catch22_all_expr("value"))
              .unnest("catch22")
        )

    Note: the caller is responsible for sorting each group by time beforehand
    (e.g. ``df.sort("entity", "time")``) if temporal ordering matters.
    """
    names = _resolve_names(which, catch24)
    struct_dtype = pl.Struct([pl.Field(n, pl.Float64) for n in names])

    def _fn(s: pl.Series) -> pl.Series:
        d = _compute(s.to_numpy(), names)
        return pl.Series([d], dtype=struct_dtype)

    return (
        pl.col(column)
        .map_batches(_fn, return_dtype=struct_dtype, returns_scalar=True)
        .alias(alias)
    )
