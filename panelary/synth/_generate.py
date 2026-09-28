"""The simulation: latent processes on a step grid, then what gets observed.

Randomness is organised so that the output is *prefix-consistent* in both
directions. Every draw comes from its own counter-keyed stream::

    SeedSequence(seed, spawn_key=(component, index))

where ``index`` is the time step for anything that evolves over time and the
entity index for anything static to an entity. A step's draws therefore never
depend on how many steps come after it, and an entity's draws never depend on
how many entities come after it: generating ``T + k`` steps (or ``N + m``
entities) with the same seed and dials reproduces the first ``T`` steps (``N``
entities) exactly. Within a stream, NumPy fills arrays element by element, so
the first ``N`` of ``N + m`` draws are the same numbers -- which is also why no
stream makes more than one entity-sized draw (a second one would start at an
``N``-dependent position).

Nothing is normalised by a sample statistic, and every hazard (breaks, entry,
exit) is a per-step probability rather than a fraction of ``T``, for the same
reason: a quantity that depends on the length of the sample would break
prefix consistency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from panelary.synth._config import SynthConfig
from panelary.synth._truth import GroundTruth

if TYPE_CHECKING:
    from panelary.core.panel_frame import PanelFrame

__all__ = ["SyntheticPanel", "generate_panel"]

# Stream components. The integers are part of the reproducibility contract:
# changing one changes every panel ever generated with that seed.
_PARAM_CLUSTER_CENTRES = 10
_PARAM_REGIME_MEANS = 11
_ENTITY_STRUCTURE = 20
_ENTITY_LOADINGS = 21
_ENTITY_LIFECYCLE = 22
_ENTITY_OBSERVATION = 23
_ENTITY_REPORTING = 24
_STEP_REGIME = 30
_STEP_SV = 31
_STEP_FACTOR = 32
_STEP_CLUSTER = 33
_STEP_BREAK = 34
_STEP_IDIO = 35
_STEP_SIGNAL = 36
_STEP_OBSERVE = 37
_STEP_MISSING = 38
_STEP_RELEASE = 39
_STEP_REVISION = 40

#: Log-normal dispersion of the per-entity idiosyncratic volatility.
_IDIO_VOL_DISPERSION = 0.25
_ENTITY = "entity"
_TIME = "time"
_EVERY_RE = re.compile(r"^(\d+)([a-z]+)$")


def _stream(seed: int, component: int, index: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(component, index)))
    )


def _shocks(rng: np.random.Generator, size: int, df: float | None) -> np.ndarray:
    """Unit-variance shocks: Gaussian, or Student-t scaled by sqrt((df-2)/df)."""
    if df is None:
        return rng.standard_normal(size)
    return rng.standard_t(df, size) * math.sqrt((df - 2.0) / df)


def _geometric(u: float, p: float) -> float:
    """Inverse-CDF geometric on {1, 2, ...}; ``inf`` when ``p == 0``.

    One uniform in, always, so the draw count never depends on a dial.
    """
    if p <= 0.0:
        return math.inf
    if p >= 1.0:
        return 1.0
    return float(max(1.0, math.ceil(math.log1p(-u) / math.log1p(-p))))


def _pick(u: float, n: int) -> int:
    """Map a uniform on [0, 1) to an index in ``0 .. n-1``."""
    return min(n - 1, int(u * n))


# ---------------------------------------------------------------------- #
# Result
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class SyntheticPanel:
    """One draw from the synthetic panel prior.

    Every frame is sorted by entity then time, and every frame is
    prefix-consistent in event time: regenerating with more periods and the
    same seed reproduces all rows with an earlier event time exactly.

    Attributes
    ----------
    panel : polars.DataFrame
        The **fully revised** panel in long format: ``entity``, ``time``,
        ``value`` (null where missing) and, when ``signal_observed``,
        ``signal``. One row per scheduled observation of a live entity, so
        entry / exit and asynchronous schedules show up as absent rows and
        missingness as nulls. This is the panel as a research database shows it
        with hindsight, including values not yet released by the end of the
        sample; use it for ordinary feature engineering.
    first_observed : polars.DataFrame
        Row-aligned with ``panel``, but ``value`` is each value's **first
        release** and ``knowledge_time`` is when that release happened (both
        null where the value is missing). This is the real-time, as-first-
        reported panel.
    vintages : polars.DataFrame
        The **bitemporal** release history: ``entity``, ``event_time``,
        ``knowledge_time``, ``value``, ``revision`` (0 = first release). One row
        per release of each non-missing value; ``knowledge_time >= event_time +
        reporting lag`` always, and the last revision of every value equals the
        ``panel`` value. Releases after the end of the sample are included
        (filter ``knowledge_time`` to reconstruct the database as of a date).
    truth : GroundTruth
        The dials, latent paths, loadings, clusters, break dates and the
        planted signal's lag and coefficient.
    """

    panel: pl.DataFrame
    first_observed: pl.DataFrame
    vintages: pl.DataFrame
    truth: GroundTruth

    def panel_frame(self) -> PanelFrame:
        """``panel`` wrapped as a :class:`~panelary.core.panel_frame.PanelFrame`.

        Returns
        -------
        PanelFrame
            Keyed on ``entity`` / ``time``.
        """
        from panelary.core.panel_frame import PanelFrame

        return PanelFrame(self.panel, entity=_ENTITY, time=_TIME)


# ---------------------------------------------------------------------- #
# Generation
# ---------------------------------------------------------------------- #
def generate_panel(
    config: SynthConfig | None = None, *, seed: int, **dials: Any
) -> SyntheticPanel:
    """Draw one synthetic panel, with its ground truth.

    Parameters
    ----------
    config : SynthConfig, optional
        The dials. Defaults to :class:`SynthConfig` ``()`` (every mechanism on,
        moderately).
    seed : int
        Non-negative seed. The same seed and dials give byte-identical frames
        on the same platform (same NumPy / Polars build and CPU); across
        platforms, transcendental functions may differ in the last bit.
    **dials : Any
        Dials to override on ``config`` (see :class:`SynthConfig`).

    Returns
    -------
    SyntheticPanel
        ``panel``, ``first_observed``, ``vintages`` and ``truth``.

    Raises
    ------
    ValueError
        If ``seed`` is negative or a dial is out of range.
    TypeError
        If a keyword is not a dial.

    Examples
    --------
    >>> from panelary.synth import generate_panel
    >>> data = generate_panel(seed=0, n_entities=20, n_periods=100)
    >>> data.panel.columns
    ['entity', 'time', 'value']
    >>> data.truth.signal_lag, data.truth.signal_coef
    (2, 0.5)
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"`seed` must be a non-negative int; got {seed!r}.")
    seed = int(seed)
    cfg = config if config is not None else SynthConfig()
    if not isinstance(cfg, SynthConfig):
        raise TypeError(f"`config` must be a SynthConfig; got {type(cfg).__name__}.")
    if dials:
        cfg = cfg.replace(**dials)

    N, T, K = cfg.n_entities, cfg.n_periods, cfg.n_factors
    G, R, M = cfg.n_clusters, cfg.n_regimes, cfg.max_revisions
    df = cfg.tail_df

    # ---- static parameters ------------------------------------------- #
    centres = _stream(seed, _PARAM_CLUSTER_CENTRES, 0).standard_normal((G, K))
    centres /= math.sqrt(max(K, 1))
    regime_means: NDArray[np.float64] = np.zeros((R, K))
    if R > 1:
        regime_means = (
            _stream(seed, _PARAM_REGIME_MEANS, 0).standard_normal((R, K))
            * cfg.regime_mean_shift
        )
    regime_vol = (
        np.array([cfg.regime_vol_ratio ** (r / (R - 1)) for r in range(R)])
        if R > 1
        else np.ones(1)
    )

    # ---- static per-entity draws (one stream per entity and purpose) -- #
    cluster = np.empty(N, dtype=np.int64)
    intercept = np.empty(N)
    idio_sd = np.empty(N)
    loadings = np.empty((N, K))
    entry = np.empty(N)
    exit_ = np.empty(N)
    period = np.empty(N, dtype=np.int64)
    phase = np.empty(N, dtype=np.int64)
    rep_lag = np.empty(N, dtype=np.int64)
    lag_lo, lag_hi = cfg.reporting_lag
    for i in range(N):
        s = _stream(seed, _ENTITY_STRUCTURE, i)
        u_cluster, z_intercept, z_vol = (
            s.random(),
            s.standard_normal(),
            s.standard_normal(),
        )
        cluster[i] = _pick(u_cluster, G)
        intercept[i] = cfg.intercept_dispersion * z_intercept
        idio_sd[i] = cfg.idio_vol * math.exp(_IDIO_VOL_DISPERSION * z_vol)
        loadings[i] = centres[cluster[i]] + cfg.loading_dispersion * _stream(
            seed, _ENTITY_LOADINGS, i
        ).standard_normal(K) / math.sqrt(max(K, 1))
        u_init, u_entry, u_exit = _stream(seed, _ENTITY_LIFECYCLE, i).random(3)
        entry[i] = (
            0.0
            if u_init < cfg.initial_fraction
            else _geometric(float(u_entry), cfg.entry_rate)
        )
        exit_[i] = entry[i] + _geometric(float(u_exit), cfg.exit_rate)
        u_period, u_phase = _stream(seed, _ENTITY_OBSERVATION, i).random(2)
        period[i] = cfg.observation_periods[
            _pick(float(u_period), len(cfg.observation_periods))
        ]
        phase[i] = _pick(float(u_phase), int(period[i]))
        rep_lag[i] = lag_lo + _pick(
            float(_stream(seed, _ENTITY_REPORTING, i).random()), lag_hi - lag_lo + 1
        )

    # ---- dynamic draws and the recursive latent processes ------------- #
    regime = np.empty(T, dtype=np.int64)
    log_vol = np.zeros((T, K))
    factors = np.zeros((T, K))
    cluster_f = np.zeros((T, G))
    idio = np.zeros((T, N))
    shock = np.zeros((T, N))
    signal = np.zeros((T, N))
    break_shift = np.zeros((T, N))
    obs_u = np.empty((T, N))
    miss_u = np.empty((T, N))
    rel_u = np.empty((T, N, 3 + M))
    rel_z = np.empty((T, N))

    rho_f = cfg.factor_persistence
    a_f = math.sqrt(1.0 - rho_f**2)
    rho_e = cfg.idio_persistence
    a_e = math.sqrt(1.0 - rho_e**2)
    phi_h, sig_h = cfg.sv_persistence, cfg.sv_vol
    sd_h0 = sig_h / math.sqrt(1.0 - phi_h**2)
    explicit_breaks = (
        frozenset(cfg.break_times) if cfg.break_times is not None else None
    )
    breaks: list[int] = []

    s_t = 0
    h: NDArray[np.float64] = np.zeros(K)
    f: NDArray[np.float64] = np.zeros(K)
    g: NDArray[np.float64] = np.zeros(G)
    e = np.zeros(N)
    for t in range(T):
        u_stay, u_move = _stream(seed, _STEP_REGIME, t).random(2)
        if t == 0:
            s_t = _pick(float(u_move), R)
        elif R > 1 and u_stay >= cfg.regime_persistence:
            s_t = (s_t + 1 + _pick(float(u_move), R - 1)) % R
        regime[t] = s_t

        eta = _stream(seed, _STEP_SV, t).standard_normal(K)
        h = sd_h0 * eta if t == 0 else phi_h * h + sig_h * eta
        log_vol[t] = h
        scale = regime_vol[s_t] * np.exp(h / 2.0)
        eps = _shocks(_stream(seed, _STEP_FACTOR, t), K, df)
        mu = regime_means[s_t]
        f = (
            mu + scale * eps
            if t == 0
            else rho_f * f + (1.0 - rho_f) * mu + a_f * scale * eps
        )
        factors[t] = f

        eps_c = _shocks(_stream(seed, _STEP_CLUSTER, t), G, df)
        g = eps_c if t == 0 else rho_f * g + a_f * eps_c
        cluster_f[t] = g

        zeta = _shocks(_stream(seed, _STEP_IDIO, t), N, df)
        shock[t] = zeta
        e = idio_sd * zeta if t == 0 else rho_e * e + a_e * idio_sd * zeta
        idio[t] = e

        signal[t] = _stream(seed, _STEP_SIGNAL, t).standard_normal(N)

        if t >= 1:
            bs = _stream(seed, _STEP_BREAK, t)
            hit = (
                t in explicit_breaks
                if explicit_breaks is not None
                else bool(bs.random() < cfg.break_rate)
            )
            if hit:
                breaks.append(t)
                break_shift[t] = cfg.break_size * bs.standard_normal(N)

        obs_u[t] = _stream(seed, _STEP_OBSERVE, t).random(N)
        miss_u[t] = _stream(seed, _STEP_MISSING, t).random(N)
        # Two streams, not one: a second N-sized draw from the same stream
        # would start at an N-dependent position and break entity-prefix
        # consistency.
        rel_u[t] = _stream(seed, _STEP_RELEASE, t).random((N, 3 + M))
        rel_z[t] = _stream(seed, _STEP_REVISION, t).standard_normal(N)

    # ---- the value --------------------------------------------------- #
    level = intercept[None, :] + np.cumsum(break_shift, axis=0)  # (T, N)
    common = factors @ loadings.T if K else np.zeros((T, N))
    clustered = cfg.cluster_strength * cluster_f[:, cluster]
    L = cfg.signal_lag
    planted = np.zeros((T, N))
    if L < T:
        planted[L:] = cfg.signal_coef * signal[: T - L]
    value = level + common + clustered + planted + idio

    # ---- what is observed -------------------------------------------- #
    steps = np.arange(T)
    alive = (steps[:, None] >= entry[None, :]) & (steps[:, None] < exit_[None, :])
    scheduled = ((steps[:, None] - phase[None, :]) % period[None, :]) == 0
    observed = alive & scheduled & (obs_u < cfg.observation_prob)
    if cfg.missing_rate > 0.0:
        logit = math.log(cfg.missing_rate / (1.0 - cfg.missing_rate))
        z = logit + cfg.missing_informativeness * (value - level)
        p_missing = 0.5 * (1.0 + np.tanh(0.5 * z))
        missing = observed & (miss_u < p_missing)
    else:
        missing = np.zeros((T, N), dtype=bool)

    # ---- releases (bitemporal) --------------------------------------- #
    # Cells in entity-major order, so every frame comes out sorted.
    obs_i, obs_t = np.nonzero(observed.T)
    obs_missing = missing[obs_t, obs_i]
    rel = ~obs_missing
    ci, ct = obs_i[rel], obs_t[rel]
    cu = rel_u[ct, ci]  # (C, 3 + M)
    delay = rep_lag[ci] + (cu[:, 0] < cfg.reporting_jitter).astype(np.int64)
    revised = (cu[:, 1] < cfg.revision_prob) if M > 0 else np.zeros(len(ci), bool)
    n_rev = np.where(revised, 1 + np.minimum(M - 1, (cu[:, 2] * M).astype(np.int64)), 0)
    gap_lo, gap_hi = cfg.revision_gap
    gaps = gap_lo + np.minimum(
        gap_hi - gap_lo, (cu[:, 3:] * (gap_hi - gap_lo + 1)).astype(np.int64)
    )
    cum_gap = np.concatenate(
        [np.zeros((len(ci), 1), np.int64), np.cumsum(gaps, axis=1)], axis=1
    )
    first_error = cfg.revision_noise * rel_z[ct, ci]
    y_rel = value[ct, ci]

    n_rel = n_rev + 1
    owner = np.repeat(np.arange(len(ci)), n_rel)
    starts = np.cumsum(n_rel) - n_rel
    r = np.arange(int(n_rel.sum()), dtype=np.int64) - np.repeat(starts, n_rel)
    k_time = ct[owner] + delay[owner] + cum_gap[owner, r]
    denom = np.maximum(n_rev[owner], 1)
    shrink = np.where(n_rev[owner] > 0, 1.0 - r / denom, 0.0)
    v_values = y_rel[owner] + first_error[owner] * shrink

    # ---- frames ------------------------------------------------------- #
    width = max(5, len(str(N - 1)))
    ids = pl.Series(_ENTITY, [f"E{i:0{width}d}" for i in range(N)], dtype=pl.String)
    max_step = int(max(T - 1, int(k_time.max()) if len(k_time) else 0))
    clock = _time_axis(cfg, max_step)

    def ent(idx: np.ndarray) -> pl.Series:
        return ids.gather(pl.Series(idx, dtype=pl.UInt32))

    def tm(name: str, idx: np.ndarray) -> pl.Series:
        return clock.gather(pl.Series(idx, dtype=pl.UInt32)).alias(name)

    panel_value = value[obs_t, obs_i]
    panel_cols = [
        ent(obs_i),
        tm(_TIME, obs_t),
        pl.Series("value", panel_value, dtype=pl.Float64).scatter(
            np.nonzero(obs_missing)[0], None
        ),
    ]
    if cfg.signal_observed:
        panel_cols.append(pl.Series("signal", signal[obs_t, obs_i], dtype=pl.Float64))
    panel = pl.DataFrame(panel_cols)

    # first release of each released cell, scattered back onto the panel rows
    first_val = np.full(len(obs_i), np.nan)
    first_k = np.zeros(len(obs_i), dtype=np.int64)
    rel_rows = np.nonzero(rel)[0]
    first_val[rel_rows] = y_rel + first_error * (n_rev > 0)
    first_k[rel_rows] = ct + delay
    miss_rows = np.nonzero(obs_missing)[0]
    fo_cols = [
        ent(obs_i),
        tm(_TIME, obs_t),
        pl.Series("value", first_val, dtype=pl.Float64).scatter(miss_rows, None),
        tm("knowledge_time", first_k).scatter(miss_rows, None),
    ]
    if cfg.signal_observed:
        fo_cols.append(pl.Series("signal", signal[obs_t, obs_i], dtype=pl.Float64))
    first_observed = pl.DataFrame(fo_cols)

    vintages = pl.DataFrame(
        [
            ent(ci[owner]),
            tm("event_time", ct[owner]),
            tm("knowledge_time", k_time),
            pl.Series("value", v_values, dtype=pl.Float64),
            pl.Series("revision", r, dtype=pl.Int64),
        ]
    )

    # ---- ground truth ------------------------------------------------- #
    grid_i = np.repeat(np.arange(N), T)
    grid_t = np.tile(steps, N)
    latent = pl.DataFrame(
        [
            ent(grid_i),
            tm(_TIME, grid_t),
            pl.Series("step", grid_t, dtype=pl.Int64),
            pl.Series("alive", alive.T.ravel(), dtype=pl.Boolean),
            pl.Series("observed", observed.T.ravel(), dtype=pl.Boolean),
            pl.Series("missing", missing.T.ravel(), dtype=pl.Boolean),
            pl.Series("value", value.T.ravel(), dtype=pl.Float64),
            pl.Series("signal", signal.T.ravel(), dtype=pl.Float64),
            pl.Series("shock", shock.T.ravel(), dtype=pl.Float64),
            pl.Series("level", level.T.ravel(), dtype=pl.Float64),
            pl.Series("common", (common + clustered).T.ravel(), dtype=pl.Float64),
            pl.Series("regime", regime[grid_t], dtype=pl.Int64),
        ]
    )
    factor_frame = pl.DataFrame(
        [
            pl.Series("step", steps, dtype=pl.Int64),
            tm(_TIME, steps),
            pl.Series("regime", regime, dtype=pl.Int64),
            *[
                pl.Series(f"factor_{k}", factors[:, k], dtype=pl.Float64)
                for k in range(K)
            ],
            *[
                pl.Series(f"log_vol_{k}", log_vol[:, k], dtype=pl.Float64)
                for k in range(K)
            ],
            *[
                pl.Series(f"cluster_factor_{c}", cluster_f[:, c], dtype=pl.Float64)
                for c in range(G)
            ],
        ]
    )
    never = np.isinf(exit_)
    entities = pl.DataFrame(
        [
            ids,
            pl.Series("cluster", cluster, dtype=pl.Int64),
            pl.Series("intercept", intercept, dtype=pl.Float64),
            pl.Series("idio_vol", idio_sd, dtype=pl.Float64),
            pl.Series(
                "entry_step",
                np.where(np.isinf(entry), -1, entry).astype(np.int64),
                dtype=pl.Int64,
            ).scatter(np.nonzero(np.isinf(entry))[0], None),
            pl.Series(
                "exit_step", np.where(never, -1, exit_).astype(np.int64), dtype=pl.Int64
            ).scatter(np.nonzero(never)[0], None),
            pl.Series("observation_period", period, dtype=pl.Int64),
            pl.Series("observation_phase", phase, dtype=pl.Int64),
            pl.Series("reporting_lag", rep_lag, dtype=pl.Int64),
            *[
                pl.Series(f"loading_{k}", loadings[:, k], dtype=pl.Float64)
                for k in range(K)
            ],
        ]
    )
    truth = GroundTruth(
        config=cfg,
        seed=seed,
        signal_lag=L,
        signal_coef=float(cfg.signal_coef),
        signal_observed=cfg.signal_observed,
        break_steps=tuple(breaks),
        break_times=tuple(clock.gather(pl.Series(breaks, dtype=pl.UInt32)).to_list()),
        min_reporting_lag=int(rep_lag.min()),
        loadings=loadings,
        clusters=cluster,
        regime_means=regime_means,
        regime_vol=regime_vol,
        entities=entities,
        factors=factor_frame,
        latent=latent,
    )
    return SyntheticPanel(
        panel=panel, first_observed=first_observed, vintages=vintages, truth=truth
    )


def _time_axis(cfg: SynthConfig, max_step: int) -> pl.Series:
    """The output label of every step ``0 .. max_step``.

    ``Int64`` steps by default; with ``start``, ``start + step * every``
    computed as one offset from ``start`` per step (not by iterating), so the
    label of a step never depends on how many steps there are.
    """
    steps = pl.Series("step", np.arange(max_step + 1), dtype=pl.Int64)
    if cfg.start is None:
        return steps.alias(_TIME)
    m = _EVERY_RE.match(cfg.every)
    assert m is not None  # validated by SynthConfig
    n, unit = int(m.group(1)), m.group(2)
    return (
        pl.DataFrame([steps])
        .select(
            pl.lit(cfg.start).dt.offset_by(pl.format("{}" + unit, pl.col("step") * n))
        )
        .to_series()
        .alias(_TIME)
    )
