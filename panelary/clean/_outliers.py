"""Robust, panel-aware, leak-safe outlier treatment.

Two families, one transformer (:class:`OutlierCleaner`):

**Fitted thresholds** (``"mad"``, ``"iqr"``, ``"zscore"``, ``"quantile"``).
Bounds are learned at :meth:`~OutlierCleaner.fit` from the training rows only
-- per entity (``pooling="entity"``, falling back to pooled bounds for
entities with too little training history or unseen at fit), pooled across
the panel (``"global"``), or recomputed per date from that date's
cross-section (``"cross_section"``; nothing is learned, entities are mixed
within a date, so ``panel_safe`` is ``False``). Transform only *reads* the
stored bounds.

**Causal rolling filters** (``"hampel"``, ``"rolling"``). Each row is judged
against the *previous* ``window`` observations of its own entity -- a trailing
Hampel filter (rolling median +/- k robust sigmas, exact MAD via numpy
sliding windows) or a trailing rolling z-score (native Polars rolling mean /
std). The current row never enters its own window, and window sizes are
fixed constants, so the output is prefix-invariant. Nothing is learned.

Actions: ``"clip"`` (winsorise to the bounds; the column becomes Float64),
``"null"`` (blank the value), ``"flag"`` (add ``{col}{flag_suffix}``
booleans) or ``"drop"`` (remove rows with any outlier).

Relationship to :mod:`panelary.preprocessing`: ``scale`` standardises with
per-entity mean/std and ``trim`` aligns entities' *time ranges* -- neither
computes a robust statistic, so there is nothing to share beyond the numeric
column selector (:data:`~panelary.preprocessing.PL_NUMERIC_COLS`, reused
here). Their behaviour is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from panelary.clean._common import check_choice
from panelary.clean._robust import robust_bounds_exprs, trailing_median_sigma
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.preprocessing._base import PL_NUMERIC_COLS

__all__ = ["OutlierCleaner"]

Method = Literal["mad", "iqr", "zscore", "quantile", "hampel", "rolling"]
_FITTED: tuple[str, ...] = ("mad", "iqr", "zscore", "quantile")
_ROLLING: tuple[str, ...] = ("hampel", "rolling")
_DEFAULT_K: dict[str, float] = {
    "mad": 3.5,
    "iqr": 1.5,
    "zscore": 3.0,
    "quantile": 0.0,
    "hampel": 3.0,
    "rolling": 3.0,
}
_POS = "__panelary_pos"


def _lo(c: str) -> str:
    return f"__panelary_lo_{c}"


def _hi(c: str) -> str:
    return f"__panelary_hi_{c}"


def _n(c: str) -> str:
    return f"__panelary_n_{c}"


class OutlierCleaner(PanelTransformer):
    """Detect and treat outliers with robust, leak-safe thresholds.

    Parameters
    ----------
    columns : str | sequence of str, optional
        Numeric columns to clean. Default: every numeric non-key column.
    method : {"mad", "iqr", "zscore", "quantile", "hampel", "rolling"}, default="mad"
        See the module docs.
    k : float, optional
        Width multiplier. Defaults: 3.5 (``"mad"``, the Iglewicz-Hoaglin
        modified-z cut-off), 1.5 (``"iqr"``, Tukey), 3.0 otherwise.
    quantiles : (float, float), default=(0.01, 0.99)
        Winsorisation limits for ``method="quantile"``.
    pooling : {"entity", "global", "cross_section"}, default="entity"
        Where fitted thresholds come from (ignored by rolling methods, which
        are always per entity).
    min_obs : int, default=10
        Minimum training observations for an entity's own bounds (else the
        pooled bounds are used), or per date for ``"cross_section"``.
    window : int, default=20
        Trailing window length (previous observations) for rolling methods.
    min_periods : int, optional
        Minimum observations in a trailing window. Default
        ``max(3, window // 2)`` -- a constant, never a function of the
        series length.
    action : {"clip", "null", "flag", "drop"}, default="clip"
        What to do with an outlier.
    flag_suffix : str, default="_outlier"
        Suffix of the flag columns for ``action="flag"``.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Attributes
    ----------
    panel_safe : bool
        ``False`` only for ``pooling="cross_section"``.
    leakage_safe : bool
        ``True``: fitted bounds come from the fit panel only; rolling windows
        are trailing.
    feature_names_in_ : list of str
        Columns cleaned.
    bounds_ : polars.DataFrame or None
        Per-entity bounds (``pooling="entity"``): the entity key plus
        ``__panelary_lo_<col>`` / ``__panelary_hi_<col>`` / ``__panelary_n_<col>``.
    global_bounds_ : dict of str -> (float, float) or None
        Pooled training bounds per column (the fallback for ``"entity"``).
    n_outliers_ : dict of str -> int or None
        Outliers found per column by the most recent :meth:`transform`.

    Examples
    --------
    >>> import polars as pl
    >>> train = pl.DataFrame(
    ...     {"id": ["a"] * 12, "t": range(12), "x": [1.0, 2.0, 1.5] * 4}
    ... )
    >>> test = pl.DataFrame({"id": ["a", "a"], "t": [12, 13], "x": [1.2, 50.0]})
    >>> cleaner = OutlierCleaner(method="iqr", entity="id", time="t").fit(train)
    >>> cleaner.transform(test).collect()["x"].to_list()
    [1.2, 3.5]
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        columns: str | Sequence[str] | None = None,
        method: Method = "mad",
        k: float | None = None,
        quantiles: tuple[float, float] = (0.01, 0.99),
        pooling: Literal["entity", "global", "cross_section"] = "entity",
        min_obs: int = 10,
        window: int = 20,
        min_periods: int | None = None,
        action: Literal["clip", "null", "flag", "drop"] = "clip",
        flag_suffix: str = "_outlier",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        check_choice("method", method, (*_FITTED, *_ROLLING))
        check_choice("pooling", pooling, ("entity", "global", "cross_section"))
        check_choice("action", action, ("clip", "null", "flag", "drop"))
        if not (0.0 <= quantiles[0] < quantiles[1] <= 1.0):
            raise ValueError(
                f"`quantiles` must satisfy 0 <= lo < hi <= 1, got {quantiles}."
            )
        if k is not None and k <= 0:
            raise ValueError(f"`k` must be > 0, got {k}.")
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"`window` must be a positive integer, got {window!r}.")
        if min_obs < 1:
            raise ValueError(f"`min_obs` must be >= 1, got {min_obs}.")
        mp = max(3, window // 2) if min_periods is None else min_periods
        if not 1 <= mp <= window:
            raise ValueError(f"`min_periods` must be in [1, window], got {mp}.")
        self.columns = columns
        self.method = method
        self.k = float(_DEFAULT_K[method] if k is None else k)
        self.quantiles = (float(quantiles[0]), float(quantiles[1]))
        self.pooling = pooling
        self.min_obs = int(min_obs)
        self.window = window
        self.min_periods = int(mp)
        self.action = action
        self.flag_suffix = flag_suffix
        # Per-instance override: a per-date cross-section mixes entities.
        self.panel_safe = not (method in _FITTED and pooling == "cross_section")
        self.feature_names_in_: list[str] = []
        self.bounds_: pl.DataFrame | None = None
        self.global_bounds_: dict[str, tuple[float | None, float | None]] | None = None
        self.n_outliers_: dict[str, int] | None = None

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def _resolve(self, panel: PanelFrame) -> list[str]:
        if self.columns is None:
            cols = (
                panel.lazy()
                .select(PL_NUMERIC_COLS(panel.entity_col, panel.time_col))
                .collect_schema()
                .names()
            )
        else:
            cols = (
                [self.columns] if isinstance(self.columns, str) else list(self.columns)
            )
            schema = panel.schema
            bad = [c for c in cols if c not in schema or not schema[c].is_numeric()]
            if bad:
                raise ValueError(
                    f"OutlierCleaner: column(s) {bad} missing or not numeric."
                )
        if not cols:
            raise ValueError("OutlierCleaner: no numeric columns to clean.")
        return cols

    def _aggs(self, cols: list[str]) -> list[pl.Expr]:
        out: list[pl.Expr] = []
        for c in cols:
            out.extend(
                robust_bounds_exprs(
                    c,
                    method=self.method,  # type: ignore[arg-type]
                    k=self.k,
                    quantiles=self.quantiles,
                    lo_name=_lo(c),
                    hi_name=_hi(c),
                    n_name=_n(c),
                )
            )
        return out

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve(panel)
        self.feature_names_in_ = cols
        self.bounds_ = None
        self.global_bounds_ = None
        if self.method not in _FITTED or self.pooling == "cross_section":
            return
        lf = panel.lazy()
        pooled = lf.select(self._aggs(cols)).collect()
        self.global_bounds_ = {
            c: (pooled.item(0, _lo(c)), pooled.item(0, _hi(c))) for c in cols
        }
        if self.pooling == "entity":
            ent = lf.group_by(panel.entity_col).agg(self._aggs(cols)).collect()
            # Too little training history: fall back to the pooled bounds.
            thin: list[pl.Expr] = []
            for c in cols:
                ok = pl.col(_n(c)) >= self.min_obs
                thin.append(pl.when(ok).then(pl.col(_lo(c))).alias(_lo(c)))
                thin.append(pl.when(ok).then(pl.col(_hi(c))).alias(_hi(c)))
            self.bounds_ = ent.with_columns(thin).sort(panel.entity_col)

    # ------------------------------------------------------------------ #
    # transform
    # ------------------------------------------------------------------ #
    def _with_bounds(self, panel: PanelFrame) -> pl.LazyFrame:
        """The panel (input order, ``_POS``) plus a lo/hi column per feature."""
        cols = self.feature_names_in_
        ent, tme = panel.entity_col, panel.time_col
        lf = panel.lazy().with_row_index(_POS)
        if self.method in _FITTED and self.pooling == "cross_section":
            exprs: list[pl.Expr] = []
            for c in cols:
                lo, hi, n = robust_bounds_exprs(
                    c,
                    method=self.method,  # type: ignore[arg-type]
                    k=self.k,
                    quantiles=self.quantiles,
                    lo_name=_lo(c),
                    hi_name=_hi(c),
                    n_name=_n(c),
                )
                enough = n.over(tme) >= self.min_obs
                exprs.append(pl.when(enough).then(lo.over(tme)).alias(_lo(c)))
                exprs.append(pl.when(enough).then(hi.over(tme)).alias(_hi(c)))
            return lf.with_columns(exprs)
        if self.method in _FITTED:
            assert self.global_bounds_ is not None
            if self.pooling == "entity":
                assert self.bounds_ is not None
                table = self.bounds_.lazy().select(
                    [ent, *(x for c in cols for x in (_lo(c), _hi(c)))]
                )
                lf = lf.join(table, on=ent, how="left").sort(_POS)
            fill = []
            for c in cols:
                glo, ghi = self.global_bounds_[c]
                lo_lit = pl.lit(glo, dtype=pl.Float64)
                hi_lit = pl.lit(ghi, dtype=pl.Float64)
                if self.pooling == "entity":
                    fill.append(pl.coalesce(pl.col(_lo(c)), lo_lit).alias(_lo(c)))
                    fill.append(pl.coalesce(pl.col(_hi(c)), hi_lit).alias(_hi(c)))
                else:
                    fill.append(lo_lit.alias(_lo(c)))
                    fill.append(hi_lit.alias(_hi(c)))
            return lf.with_columns(fill)
        # Causal rolling methods: per entity, in time order, previous rows only.
        df = lf.sort([ent, tme, _POS]).collect()
        if self.method == "rolling":
            exprs = []
            for c in cols:
                x = pl.col(c).cast(pl.Float64).fill_nan(None).shift(1)
                m = x.rolling_mean(self.window, min_samples=self.min_periods).over(ent)
                s = x.rolling_std(self.window, min_samples=self.min_periods).over(ent)
                exprs.append((m - self.k * s).alias(_lo(c)))
                exprs.append((m + self.k * s).alias(_hi(c)))
            df = df.with_columns(exprs)
        else:  # hampel
            codes = (
                df.select(
                    (pl.col(ent) != pl.col(ent).shift(1)).fill_null(True).cum_sum()
                )
                .to_series()
                .to_numpy()
            )
            new = []
            for c in cols:
                vals = (
                    df.get_column(c).cast(pl.Float64).fill_nan(None).to_numpy()
                ).astype(np.float64)
                med, sigma = trailing_median_sigma(
                    vals, codes, self.window, self.min_periods
                )
                new.append(pl.Series(_lo(c), med - self.k * sigma).fill_nan(None))
                new.append(pl.Series(_hi(c), med + self.k * sigma).fill_nan(None))
            df = df.with_columns(new)
        return df.sort(_POS).lazy()

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        cols = self.feature_names_in_
        missing = [c for c in cols if c not in panel]
        if missing:
            raise ValueError(
                f"OutlierCleaner.transform: fitted column(s) {missing} are missing."
            )
        frame = self._with_bounds(panel).collect()
        # NaN is "missing", never an outlier (Polars orders NaN above every
        # number, so it must be masked before comparing with the bounds).
        val = {c: pl.col(c).cast(pl.Float64).fill_nan(None) for c in cols}
        is_out = {
            c: ((val[c] < pl.col(_lo(c))) | (val[c] > pl.col(_hi(c)))).fill_null(False)
            for c in cols
        }
        counts = frame.select([e.sum().alias(c) for c, e in is_out.items()]).row(0)
        self.n_outliers_ = {c: int(v) for c, v in zip(cols, counts, strict=True)}
        if self.action == "clip":
            treated = [
                pl.when(val[c] < pl.col(_lo(c)))
                .then(pl.col(_lo(c)))
                .when(val[c] > pl.col(_hi(c)))
                .then(pl.col(_hi(c)))
                .otherwise(pl.col(c).cast(pl.Float64))
                .alias(c)
                for c in cols
            ]
            out = frame.with_columns(treated)
        elif self.action == "null":
            out = frame.with_columns(
                [
                    pl.when(e).then(None).otherwise(pl.col(c)).alias(c)
                    for c, e in is_out.items()
                ]
            )
        elif self.action == "flag":
            out = frame.with_columns(
                [e.alias(f"{c}{self.flag_suffix}") for c, e in is_out.items()]
            )
        else:  # drop
            out = frame.filter(~pl.any_horizontal(list(is_out.values())))
        helpers = [_POS, *(x for c in cols for x in (_lo(c), _hi(c)))]
        out = out.drop([h for h in helpers if h in out.columns])
        return PanelFrame(
            out.lazy(), entity=panel.entity_col, time=panel.time_col, validate=False
        )
