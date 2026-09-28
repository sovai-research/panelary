"""Planted ground truth, and the check that turns it into a leak detector.

The generator plants one predictive relationship with a known lag ``L`` and
coefficient ``beta``::

    value[i, t] = ... + beta * signal[i, t - L] + ...

where ``signal`` is i.i.d. standard normal and independent of every other
random quantity in the simulation. That independence is what makes the lag
*informative*: ``signal[s]`` enters the observable data for the first time at
step ``s + L`` (through ``value``), or at step ``s`` if ``signal_observed``.
So a feature computed at step ``t`` from information available at ``t`` can be
correlated with ``signal[t - l]`` only for ``l >= knowable_lag``. A feature that
*is* correlated with ``signal[t - l]`` for some ``l < knowable_lag`` read data
from after ``t``. :func:`check_planted_lag` measures exactly that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from panelary.synth._config import SynthConfig

__all__ = ["GroundTruth", "PlantedLagReport", "check_planted_lag"]


@dataclass(frozen=True, eq=False)
class GroundTruth:
    """Everything the generator knows and the data does not show.

    Attributes
    ----------
    config : SynthConfig
        The dials the panel was drawn with.
    seed : int
        The seed it was drawn with.
    signal_lag : int
        The planted lag ``L``: ``value[t]`` loads on ``signal[t - L]``.
    signal_coef : float
        The planted coefficient ``beta`` on that lagged signal.
    signal_observed : bool
        Whether ``signal`` is an observable column of the panel.
    break_steps : tuple of int
        Steps at which a structural break occurred.
    break_times : tuple
        The same breaks in the panel's time labels (equal to ``break_steps``
        unless ``config.start`` is set).
    min_reporting_lag : int
        The smallest per-entity reporting lag actually drawn.
    loadings : numpy.ndarray
        ``(n_entities, n_factors)`` factor loadings.
    clusters : numpy.ndarray
        ``(n_entities,)`` cluster of each entity.
    regime_means, regime_vol : numpy.ndarray
        Per-regime factor means ``(n_regimes, n_factors)`` and factor-volatility
        multipliers ``(n_regimes,)``.
    entities : polars.DataFrame
        One row per entity in the pool: ``cluster``, ``intercept``,
        ``idio_vol``, ``entry_step`` / ``exit_step`` (null = never),
        ``observation_period`` / ``observation_phase``, ``reporting_lag`` and
        ``loading_k``.
    factors : polars.DataFrame
        One row per step: ``regime``, ``factor_k``, ``log_vol_k`` and
        ``cluster_factor_c``.
    latent : polars.DataFrame
        The full ``entity x step`` grid, including steps where an entity is not
        alive: ``alive``, ``observed``, ``missing``, the true ``value``, the
        ``signal``, the standardised idiosyncratic ``shock``, the
        break-shifted ``level``, the ``common`` (factor + cluster) component and
        the ``regime``.
    """

    config: SynthConfig
    seed: int
    signal_lag: int
    signal_coef: float
    signal_observed: bool
    break_steps: tuple[int, ...]
    break_times: tuple[Any, ...]
    min_reporting_lag: int
    loadings: np.ndarray
    clusters: np.ndarray
    regime_means: np.ndarray
    regime_vol: np.ndarray
    entities: pl.DataFrame
    factors: pl.DataFrame
    latent: pl.DataFrame

    def knowable_lag(self, clock: Literal["event", "knowledge"] = "event") -> int:
        """The smallest lag at which a causal feature can carry the signal.

        Parameters
        ----------
        clock : {"event", "knowledge"}
            What the feature frame's time column means. ``"event"``: the
            feature at time ``t`` may use every value with event time ``<= t``
            (ordinary feature engineering on ``panel``). ``"knowledge"``: the
            feature at time ``t`` may use only what was *published* by ``t``, so
            a value with event time ``s`` is usable from ``s + reporting lag``
            (point-in-time work on ``vintages`` / ``first_observed``).

        Returns
        -------
        int
            ``0`` if the signal is observable (it is known at its event time);
            otherwise ``signal_lag`` on the event clock and ``signal_lag +
            min_reporting_lag`` on the knowledge clock.

        Raises
        ------
        ValueError
            If ``clock`` is not one of the two values above.
        """
        if clock not in ("event", "knowledge"):
            raise ValueError(f"`clock` must be 'event' or 'knowledge'; got {clock!r}.")
        if self.signal_observed:
            return 0
        if clock == "knowledge":
            return self.signal_lag + self.min_reporting_lag
        return self.signal_lag


@dataclass(frozen=True)
class PlantedLagReport:
    """The signal-dependence profile of one feature, and the verdict.

    Attributes
    ----------
    feature : str
        The feature column checked.
    planted_lag, planted_coef : int, float
        The planted lag and coefficient (from the ground truth).
    min_lag : int
        The smallest lag a causal feature may carry the signal at.
    lags : tuple of int
        Lags ``l`` tested; the profile is ``corr(feature[t], signal[t - l])``.
    correlation : tuple of float
        Pooled Spearman correlation at each lag (``nan`` if undefined).
    z : tuple of float
        ``correlation * sqrt(n_obs - 1)``, approximately standard normal under
        the null of no dependence.
    n_obs : tuple of int
        Pairs used at each lag.
    threshold : float
        ``|z|`` above which a lag counts as carrying the signal.
    recovered_lag : int or None
        The smallest tested lag that carries the signal, or ``None``.
    leak_lags : tuple of int
        Lags below ``min_lag`` that carry the signal.
    leaks : bool
        ``True`` iff ``leak_lags`` is non-empty: the feature knows the signal
        earlier than any causal feature could.
    """

    feature: str
    planted_lag: int
    planted_coef: float
    min_lag: int
    lags: tuple[int, ...]
    correlation: tuple[float, ...]
    z: tuple[float, ...]
    n_obs: tuple[int, ...]
    threshold: float
    recovered_lag: int | None
    leak_lags: tuple[int, ...]
    leaks: bool

    def to_frame(self) -> pl.DataFrame:
        """The profile as a frame: ``lag``, ``correlation``, ``z``, ``n_obs``,
        ``significant``, ``leaks``."""
        return (
            pl.DataFrame(
                {
                    "lag": list(self.lags),
                    "correlation": list(self.correlation),
                    "z": list(self.z),
                    "n_obs": list(self.n_obs),
                },
                schema={
                    "lag": pl.Int64,
                    "correlation": pl.Float64,
                    "z": pl.Float64,
                    "n_obs": pl.Int64,
                },
            )
            .with_columns(
                significant=pl.col("z").abs().fill_nan(0.0) > self.threshold,
            )
            .with_columns(
                leaks=pl.col("significant") & (pl.col("lag") < self.min_lag),
            )
        )

    def __str__(self) -> str:
        verdict = (
            f"LEAKS: carries the signal at lag(s) {list(self.leak_lags)}, "
            f"earlier than the knowable lag {self.min_lag}"
            if self.leaks
            else f"ok: no signal before the knowable lag {self.min_lag}"
        )
        rec = "none" if self.recovered_lag is None else str(self.recovered_lag)
        return (
            f"PlantedLagReport({self.feature!r}; planted lag {self.planted_lag}, "
            f"coef {self.planted_coef:g}; recovered at lag {rec}) -- {verdict}"
        )


def _as_frame(frame: Any, entity: str, time: str) -> tuple[pl.DataFrame, str, str]:
    from panelary.core.panel_frame import PanelFrame

    if isinstance(frame, PanelFrame):
        return frame.collect(), frame.entity_col, frame.time_col
    if isinstance(frame, pl.LazyFrame):
        return frame.collect(), entity, time
    if isinstance(frame, pl.DataFrame):
        return frame, entity, time
    raise TypeError(
        "check_planted_lag: `frame` must be a polars DataFrame / LazyFrame or a "
        f"PanelFrame; got {type(frame).__name__!r}."
    )


def check_planted_lag(
    frame: Any,
    feature: str,
    truth: GroundTruth,
    *,
    entity: str = "entity",
    time: str = "time",
    clock: Literal["event", "knowledge"] = "event",
    min_lag: int | None = None,
    max_lead: int = 25,
    max_lag: int = 5,
    threshold: float = 5.0,
) -> PlantedLagReport:
    """Does ``feature`` know the planted signal earlier than it could?

    For each lag ``l`` in ``min_lag - max_lead .. max(min_lag, L) + max_lag``
    this computes the pooled Spearman correlation between ``feature`` at
    ``(entity, t)`` and the planted ``signal`` at ``(entity, t - l)``, taken
    from ``truth.latent``. Because the signal is i.i.d. and reaches the data
    for the first time ``knowable_lag`` steps after it is drawn, a feature
    computed only from information available at ``t`` has population
    correlation exactly zero at every ``l < knowable_lag``; the report flags a
    leak when ``|z| > threshold`` at any such lag. A causal feature built on
    ``value`` typically *recovers* the signal at ``l >= L`` (e.g. ``value``
    itself at ``L``, ``value.shift(1)`` at ``L + 1``); a ``value.shift(-1)``
    recovers it at ``L - 1`` and is flagged.

    Parameters
    ----------
    frame : polars.DataFrame, polars.LazyFrame or PanelFrame
        Holds the feature, keyed by entity and time labels from the generated
        panel. A PanelFrame's own keys override ``entity`` / ``time``.
    feature : str
        Column to check. Nulls and non-finite values are ignored.
    truth : GroundTruth
        From the :class:`~panelary.synth.SyntheticPanel` the feature was built
        on.
    entity, time : str
        Key columns of ``frame``.
    clock : {"event", "knowledge"}
        Meaning of ``time``; see :meth:`GroundTruth.knowable_lag`.
    min_lag : int, optional
        Override the knowable lag. Pass ``truth.signal_lag`` when the panel
        exposes ``signal`` but the pipeline under test never reads it (the
        default would then be ``0``, which is correct but blunt).
    max_lead : int
        How far below ``min_lag`` to look. A look-ahead of ``k`` steps shows up
        at ``min_lag - k``, so on an asynchronously observed panel a
        one-*row* look-ahead can be several steps; widen this accordingly.
    max_lag : int
        How far above the planted lag to report (for the recovery profile).
    threshold : float
        ``|z|`` needed to call a lag significant. The default ``5`` makes a
        false alarm over ~30 lags a ~1e-5 event.

    Returns
    -------
    PlantedLagReport

    Raises
    ------
    KeyError
        If ``feature``, ``entity`` or ``time`` is not a column of ``frame``.
    ValueError
        If ``max_lead`` / ``max_lag`` are negative or ``threshold`` is not
        positive.

    Notes
    -----
    This is a statistical detector for *concentrated* look-ahead -- a shift,
    a centred window, a backward fill -- where some future step carries a
    non-negligible share of the feature. It cannot see look-ahead that is
    spread thinly over the whole sample (a full-sample mean gives each future
    step a weight of ``1/T``); :func:`panelary.testing.assert_no_lookahead`
    is the exact, perturbation-based check for that. Its power grows with the
    number of rows and with ``|signal_coef|`` relative to the other variance in
    ``value``.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.synth import SynthConfig, generate_panel, check_planted_lag
    >>> data = generate_panel(SynthConfig.plain(n_entities=40, n_periods=250), seed=1)
    >>> leaky = data.panel.with_columns(
    ...     f=pl.col("value").shift(-1).over("entity", order_by="time"))
    >>> check_planted_lag(leaky, "f", data.truth).leaks
    True
    """
    if max_lead < 0 or max_lag < 0:
        raise ValueError("`max_lead` and `max_lag` must be >= 0.")
    if not (math.isfinite(threshold) and threshold > 0):
        raise ValueError(f"`threshold` must be positive; got {threshold!r}.")
    df, entity, time = _as_frame(frame, entity, time)
    for col in (entity, time, feature):
        if col not in df.columns:
            raise KeyError(f"check_planted_lag: column {col!r} not in frame.")
    floor = truth.knowable_lag(clock) if min_lag is None else int(min_lag)

    grid = truth.latent.select(
        pl.col("entity").alias(entity), pl.col("time").alias(time), "step"
    )
    feats = (
        df.select(entity, time, pl.col(feature).cast(pl.Float64).alias("__f"))
        .filter(pl.col("__f").is_not_null() & pl.col("__f").is_finite())
        .join(grid, on=[entity, time], how="inner")
    )
    sig = truth.latent.select(pl.col("entity").alias(entity), "step", "signal")

    top = max(floor, truth.signal_lag) + max_lag
    lags = tuple(range(floor - max_lead, top + 1))
    corr: list[float] = []
    zs: list[float] = []
    ns: list[int] = []
    for lag in lags:
        pairs = feats.with_columns(pl.col("step") - lag).join(
            sig, on=[entity, "step"], how="inner"
        )
        n = pairs.height
        r = float("nan")
        if n >= 3:
            out = pairs.select(pl.corr("__f", "signal", method="spearman")).item()
            r = float(out) if out is not None else float("nan")
        corr.append(r)
        zs.append(r * math.sqrt(n - 1) if n >= 2 and math.isfinite(r) else float("nan"))
        ns.append(n)

    significant = [
        lag
        for lag, z in zip(lags, zs, strict=True)
        if math.isfinite(z) and abs(z) > threshold
    ]
    leak_lags = tuple(lag for lag in significant if lag < floor)
    return PlantedLagReport(
        feature=feature,
        planted_lag=truth.signal_lag,
        planted_coef=truth.signal_coef,
        min_lag=floor,
        lags=lags,
        correlation=tuple(corr),
        z=tuple(zs),
        n_obs=tuple(ns),
        threshold=float(threshold),
        recovered_lag=min(significant) if significant else None,
        leak_lags=leak_lags,
        leaks=bool(leak_lags),
    )
