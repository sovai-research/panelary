"""Pipeline transformers for the market-state features, and their specs.

:class:`MarketState` and :class:`Turbulence` wrap :func:`market_state` and
:func:`turbulence` with ``broadcast=True`` so they can sit in a
:class:`~panelary.core.pipeline.Pipeline`. **``fit`` learns nothing** -- it
only checks the columns: every value is computed as of its own date from the
frame being transformed, so a transform fitted on one fold and applied to
another uses no statistic from the fitting fold. They mix entities by design
(``panel_safe = False``) and never look ahead (``leakage_safe = True``).

Note the consequence for cross-validation: transforming a test fold on its
own starts every trailing window afresh at the fold's first date, so its
first ``window`` dates are null. Transform the full panel once (it is
prefix-invariant) and split afterwards if that warm-up matters.

This module also registers the ``market_state`` and ``turbulence``
:class:`~panelary.registry.FeatureSpec` entries (namespace ``covariance``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from panelary.core._schedule import Schedule
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelTransformer
from panelary.covariance._state import DEFAULT_FEATURES, market_state
from panelary.covariance._turbulence import turbulence
from panelary.registry import FeatureSpec, registry

__all__ = ["MarketState", "Turbulence"]


class MarketState(PanelTransformer):
    """Append per-date market-state features to every row (as of its date).

    Parameters
    ----------
    returns : str
        Return column.
    window : int, default 252
    features : sequence of str, default :data:`DEFAULT_FEATURES`
    stride, schedule, group, min_coverage, min_entities, space, ar_k,
    ar_short, ar_long, edge, alpha, min_gap
        As for :func:`panelary.covariance.market_state`.
    entity, time : str, optional
        Keys for bare polars frames.

    Examples
    --------
    >>> import panelary as pn
    >>> data = pn.synth.generate_panel(seed=0, n_entities=12, n_periods=80,
    ...                                missing_rate=0.0, entry_rate=0.0,
    ...                                exit_rate=0.0, initial_fraction=1.0)
    >>> ms = pn.covariance.MarketState(returns="value", window=20,
    ...                                features=("absorption_ratio",),
    ...                                entity="entity", time="time")
    >>> out = ms.fit_transform(data.panel).collect()
    >>> "absorption_ratio" in out.columns
    True
    """

    panel_safe = False
    leakage_safe = True

    def __init__(
        self,
        returns: str,
        window: int = 252,
        *,
        features: Sequence[str] = DEFAULT_FEATURES,
        stride: int | None = 1,
        schedule: Schedule | int | str | None = None,
        group: str | None = None,
        min_coverage: float = 0.95,
        min_entities: int | None = None,
        space: str = "correlation",
        ar_k: int | None = None,
        ar_short: int = 15,
        ar_long: int = 252,
        edge: str = "tw",
        alpha: float = 0.95,
        min_gap: float = 1.05,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        self.returns = returns
        self.window = int(window)
        self.features = tuple(features)
        self.stride = stride
        self.schedule = schedule
        self.group = group
        self.min_coverage = min_coverage
        self.min_entities = min_entities
        self.space = space
        self.ar_k = ar_k
        self.ar_short = ar_short
        self.ar_long = ar_long
        self.edge = edge
        self.alpha = alpha
        self.min_gap = min_gap

    def _fit(self, panel: PanelFrame) -> None:
        need = [self.returns] + ([self.group] if self.group else [])
        missing = [c for c in need if c not in panel.columns]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel. Available: {panel.columns}."
            )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        out = market_state(
            panel,
            returns=self.returns,
            window=self.window,
            stride=self.stride,
            schedule=self.schedule,
            features=self.features,
            group=self.group,
            broadcast=True,
            min_coverage=self.min_coverage,
            min_entities=self.min_entities,
            space=self.space,
            ar_k=self.ar_k,
            ar_short=self.ar_short,
            ar_long=self.ar_long,
            edge=self.edge,
            alpha=self.alpha,
            min_gap=self.min_gap,
        )
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )


class Turbulence(PanelTransformer):
    """Append causal turbulence (and its trailing percentile) to every row.

    Parameters
    ----------
    returns : str
    window : int, default 252
    method : str, default "qis"
    refit, lag, space, min_coverage, min_entities, group, pct_window
        As for :func:`panelary.covariance.turbulence`.
    **options
        Estimator options.
    """

    panel_safe = False
    leakage_safe = True

    def __init__(
        self,
        returns: str,
        window: int = 252,
        *,
        method: str = "qis",
        refit: Schedule | int | str | None = None,
        lag: int = 1,
        space: str = "correlation",
        min_coverage: float = 0.95,
        min_entities: int | None = None,
        group: str | None = None,
        pct_window: int = 252,
        entity: str | None = None,
        time: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if int(lag) < 1:
            raise ValueError(f"`lag` must be >= 1 (trap T2), got {lag}.")
        self.returns = returns
        self.window = int(window)
        self.method = method
        self.refit = refit
        self.lag = int(lag)
        self.space = space
        self.min_coverage = min_coverage
        self.min_entities = min_entities
        self.group = group
        self.pct_window = pct_window
        self.options = dict(options)

    def _fit(self, panel: PanelFrame) -> None:
        need = [self.returns] + ([self.group] if self.group else [])
        missing = [c for c in need if c not in panel.columns]
        if missing:
            raise ValueError(
                f"column(s) {missing} not found in panel. Available: {panel.columns}."
            )

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        out = turbulence(
            panel,
            returns=self.returns,
            window=self.window,
            method=self.method,
            refit=self.refit,
            lag=self.lag,
            space=self.space,
            min_coverage=self.min_coverage,
            min_entities=self.min_entities,
            group=self.group,
            pct_window=self.pct_window,
            broadcast=True,
            **self.options,
        )
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )


# --------------------------------------------------------------------------- #
# FeatureSpecs (names are globally unique; the registry keys by bare name)
# --------------------------------------------------------------------------- #
registry.register(
    FeatureSpec(
        name="market_state",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "stride": int,
            "features": tuple,
            "group": str,
            "min_coverage": float,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source=(
            "Kritzman, Li, Page & Rigobon (2011); Roy & Vetterli (2007); "
            "Plerou et al. (2002); Marchenko & Pastur (1967); Johnstone (2001)"
        ),
        license="Apache-2.0",
        backend_fn=market_state,
        cost_hint=("O(D (min(N,W)^2 max(N,W) + min(N,W)^3)) for D evaluated dates"),
    )
)
registry.register(
    FeatureSpec(
        name="turbulence",
        namespace="covariance",
        input_shape="frame",
        output_shape="frame",
        params={
            "returns": str,
            "window": int,
            "method": str,
            "refit": str,
            "lag": int,
        },
        tier="B",
        panel_safe=False,
        leakage_safe=True,
        safe_scope="rowwise",
        source=(
            "Chow, Jacquier, Kritzman & Lowry (1999); Kritzman & Li (2010), "
            "made causal (estimate lag >= 1)"
        ),
        license="Apache-2.0",
        backend_fn=turbulence,
        cost_hint="O(R estimate + T N r) for R refits and T scored dates",
    )
)
