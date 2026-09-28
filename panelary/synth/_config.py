"""The dials of the synthetic panel prior, and a prior over the dials.

:class:`SynthConfig` is a frozen, validated bag of dials; it holds no random
state. The seed is passed separately to :func:`~panelary.synth.generate_panel`,
so one config plus many seeds is many draws from the same prior, and
:func:`sample_config` draws the dials themselves for when the dial space is the
thing being sampled (the PanelPFN use case).
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np

__all__ = ["SynthConfig", "sample_config"]

#: ``every`` must look like a Polars duration with a single unit: ``"1d"``,
#: ``"5m"``, ``"1mo"``.
_EVERY_RE = re.compile(r"^(\d+)(ns|us|ms|s|m|h|d|w|mo|q|y)$")
_SUB_DAY_UNITS = frozenset({"ns", "us", "ms", "s", "m", "h"})


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


@dataclass(frozen=True)
class SynthConfig:
    """Dials of the synthetic panel generator.

    Every dial has a neutral value that switches its mechanism off, so the same
    class describes both a deliberately messy panel (the defaults: every
    mechanism on at a moderate setting) and a clean one
    (:meth:`SynthConfig.plain`). All times are **integer steps** on the latent
    grid ``0 .. n_periods - 1``; ``start`` / ``every`` only relabel them on
    output.

    The value of entity ``i`` at step ``t`` is ::

        value[i, t] = intercept[i, t]                    # breaks shift it
                    + loadings[i] @ factors[t]           # regimes + SV
                    + cluster_strength * cluster[c_i, t]
                    + signal_coef * signal[i, t - signal_lag]
                    + idio[i, t]                         # AR(1)

    Parameters
    ----------
    n_entities : int
        Size of the entity pool, i.e. the maximum cross-section. How many are
        alive at a given step is governed by the entry / exit dials.
    n_periods : int
        Length ``T`` of the latent time grid.
    n_factors : int
        Number of common latent factors (``0`` disables them). Loadings are
        scaled by ``1 / sqrt(n_factors)``, so the common variance does not grow
        with the number of factors.
    factor_persistence : float
        AR(1) coefficient of the factors and of the cluster factors, in
        ``[0, 1)``.
    n_regimes : int
        Number of Markov regimes (``1`` disables regime switching).
    regime_persistence : float
        Probability of staying in the current regime at each step.
    regime_vol_ratio : float
        Factor volatility of the most volatile regime relative to the calmest;
        regime ``r`` scales factor shocks by ``ratio ** (r / (n_regimes - 1))``.
    regime_mean_shift : float
        Standard deviation of the per-regime factor means.
    sv_persistence : float
        AR(1) coefficient of each factor's log-variance, in ``[0, 1)``.
    sv_vol : float
        Volatility of the log-variance (``0`` disables stochastic volatility).
    tail_df : float or None
        Degrees of freedom of the Student-t innovations (factor, cluster and
        idiosyncratic shocks), scaled to unit variance, so it must exceed 2.
        ``None`` gives Gaussian innovations.
    n_clusters : int
        Number of cross-sectional clusters (``1`` means a single cluster).
        Entities in a cluster share loading centres and a cluster factor.
    cluster_strength : float
        Loading of every entity on its cluster's factor.
    loading_dispersion : float
        Standard deviation of an entity's loadings around its cluster centre
        (before the ``1 / sqrt(n_factors)`` scaling).
    idio_persistence : float
        AR(1) coefficient of the idiosyncratic component, in ``[0, 1)``.
    idio_vol : float
        Typical idiosyncratic standard deviation (entities scatter around it
        log-normally with dispersion 0.25).
    intercept_dispersion : float
        Standard deviation of the entity intercepts.
    signal_lag : int
        The **planted lag** ``L >= 0``: ``value[t]`` loads on ``signal[t - L]``.
    signal_coef : float
        The **planted coefficient** on that lagged signal. The signal itself is
        i.i.d. standard normal, independent of everything else.
    signal_observed : bool
        Emit the signal as an observable ``signal`` column, known at its own
        event time. When ``False`` (the default) it is latent: it reaches the
        observable data only through ``value``, ``signal_lag`` steps later,
        which is what makes the planted-lag check sharp.
    break_rate : float
        Per-step hazard of a structural break (ignored if ``break_times`` is
        given). At a break every entity's intercept shifts by an independent
        ``N(0, break_size**2)`` draw.
    break_size : float
        Standard deviation of the intercept shift at a break.
    break_times : tuple of int or None
        Explicit break steps (each ``>= 1``), overriding ``break_rate``.
    initial_fraction : float
        Fraction of entities alive at step 0; the rest enter later.
    entry_rate : float
        Per-step hazard of entry for an entity not alive at step 0.
    exit_rate : float
        Per-step hazard of permanent exit once alive (``0`` = never exit).
    observation_periods : tuple of int
        Each entity observes on a regular schedule with a period drawn
        uniformly from this tuple and a random phase, so entities with
        different periods are observed at different times (asynchronous).
    observation_prob : float
        Probability that a scheduled observation actually happens.
    missing_rate : float
        Baseline probability that an observed value is missing (null).
    missing_informativeness : float
        Log-odds of a value being missing rise by this much per unit of its
        deviation from the entity's intercept (``0`` = missing completely at
        random; negative = low values go missing).
    reporting_lag : tuple of (int, int)
        Inclusive range of the per-entity reporting lag: a value for event step
        ``t`` is first known at ``t + lag`` (plus jitter).
    reporting_jitter : float
        Probability that a release is one step later than the entity's lag.
    revision_prob : float
        Probability that a value is first released with an error and then
        revised.
    max_revisions : int
        Maximum number of revisions of a revised value; the count is uniform on
        ``1 .. max_revisions``. The last revision is exactly the true value.
    revision_noise : float
        Standard deviation of the first-release error of a revised value.
        Intermediate revisions shrink it linearly to zero.
    revision_gap : tuple of (int, int)
        Inclusive range of the steps between successive releases.
    start : datetime.date or datetime.datetime or None
        If given, steps are relabelled as ``start + step * every`` on output;
        otherwise times are ``Int64`` steps.
    every : str
        Step length as a single-unit Polars duration (``"1d"``, ``"1mo"``,
        ``"5m"``). Only used with ``start``.

    Raises
    ------
    ValueError
        If a dial is outside its documented range.
    """

    # size
    n_entities: int = 50
    n_periods: int = 200
    # latent factors, regimes and stochastic volatility
    n_factors: int = 3
    factor_persistence: float = 0.9
    n_regimes: int = 2
    regime_persistence: float = 0.97
    regime_vol_ratio: float = 2.0
    regime_mean_shift: float = 0.25
    sv_persistence: float = 0.95
    sv_vol: float = 0.2
    # innovations
    tail_df: float | None = 5.0
    # clusters and entity heterogeneity
    n_clusters: int = 3
    cluster_strength: float = 0.5
    loading_dispersion: float = 0.3
    idio_persistence: float = 0.2
    idio_vol: float = 1.0
    intercept_dispersion: float = 0.5
    # planted predictive signal
    signal_lag: int = 2
    signal_coef: float = 0.5
    signal_observed: bool = False
    # structural breaks
    break_rate: float = 0.01
    break_size: float = 1.0
    break_times: tuple[int, ...] | None = None
    # entry and exit
    initial_fraction: float = 0.8
    entry_rate: float = 0.02
    exit_rate: float = 0.003
    # asynchronous observation
    observation_periods: tuple[int, ...] = (1,)
    observation_prob: float = 1.0
    # missingness
    missing_rate: float = 0.05
    missing_informativeness: float = 1.0
    # reporting lags and revisions
    reporting_lag: tuple[int, int] = (1, 3)
    reporting_jitter: float = 0.2
    revision_prob: float = 0.3
    max_revisions: int = 2
    revision_noise: float = 0.5
    revision_gap: tuple[int, int] = (1, 5)
    # output time axis
    start: date | datetime | None = field(default=None)
    every: str = "1d"

    def __post_init__(self) -> None:
        # Coerce sequence dials to tuples so the config stays hashable and a
        # list and a tuple describe the same config.
        for name in ("observation_periods", "reporting_lag", "revision_gap"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.break_times is not None:
            object.__setattr__(self, "break_times", tuple(self.break_times))
        self._validate()

    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        def need(ok: bool, msg: str) -> None:
            if not ok:
                raise ValueError(f"SynthConfig: {msg}")

        def unit(name: str, *, closed_right: bool = True) -> None:
            v = getattr(self, name)
            ok = 0.0 <= v <= 1.0 if closed_right else 0.0 <= v < 1.0
            rng = "[0, 1]" if closed_right else "[0, 1)"
            need(math.isfinite(v) and ok, f"`{name}` must be in {rng}; got {v!r}.")

        def nonneg(name: str) -> None:
            v = getattr(self, name)
            need(math.isfinite(v) and v >= 0, f"`{name}` must be >= 0; got {v!r}.")

        for name, lo in (
            ("n_entities", 1),
            ("n_periods", 1),
            ("n_factors", 0),
            ("n_regimes", 1),
            ("n_clusters", 1),
            ("signal_lag", 0),
            ("max_revisions", 0),
        ):
            v = getattr(self, name)
            need(_is_int(v) and v >= lo, f"`{name}` must be an int >= {lo}; got {v!r}.")
        for name in ("factor_persistence", "sv_persistence", "idio_persistence"):
            unit(name, closed_right=False)
        for name in (
            "regime_persistence",
            "break_rate",
            "initial_fraction",
            "entry_rate",
            "exit_rate",
            "reporting_jitter",
            "revision_prob",
        ):
            unit(name)
        for name in (
            "regime_mean_shift",
            "sv_vol",
            "cluster_strength",
            "loading_dispersion",
            "idio_vol",
            "intercept_dispersion",
            "break_size",
            "revision_noise",
        ):
            nonneg(name)
        need(
            math.isfinite(self.regime_vol_ratio) and self.regime_vol_ratio > 0,
            f"`regime_vol_ratio` must be > 0; got {self.regime_vol_ratio!r}.",
        )
        need(
            self.tail_df is None or (math.isfinite(self.tail_df) and self.tail_df > 2),
            "`tail_df` must be None (Gaussian) or > 2 so the Student-t shocks "
            f"can be scaled to unit variance; got {self.tail_df!r}.",
        )
        need(
            math.isfinite(self.signal_coef),
            f"`signal_coef` must be finite; got {self.signal_coef!r}.",
        )
        need(
            isinstance(self.signal_observed, bool),
            f"`signal_observed` must be a bool; got {self.signal_observed!r}.",
        )
        if self.break_times is not None:
            need(
                all(_is_int(b) and b >= 1 for b in self.break_times)
                and len(set(self.break_times)) == len(self.break_times),
                "`break_times` must be distinct int steps >= 1; got "
                f"{self.break_times!r}.",
            )
        need(
            len(self.observation_periods) >= 1
            and all(_is_int(p) and p >= 1 for p in self.observation_periods),
            "`observation_periods` must be a non-empty tuple of ints >= 1; got "
            f"{self.observation_periods!r}.",
        )
        need(
            math.isfinite(self.observation_prob) and 0 < self.observation_prob <= 1,
            f"`observation_prob` must be in (0, 1]; got {self.observation_prob!r}.",
        )
        need(
            math.isfinite(self.missing_rate) and 0 <= self.missing_rate < 1,
            f"`missing_rate` must be in [0, 1); got {self.missing_rate!r}.",
        )
        need(
            math.isfinite(self.missing_informativeness),
            "`missing_informativeness` must be finite; got "
            f"{self.missing_informativeness!r}.",
        )
        for name, lo in (("reporting_lag", 0), ("revision_gap", 1)):
            pair = getattr(self, name)
            need(
                len(pair) == 2
                and all(_is_int(v) for v in pair)
                and lo <= pair[0] <= pair[1],
                f"`{name}` must be an inclusive (lo, hi) int range with "
                f"{lo} <= lo <= hi; got {pair!r}.",
            )
        if self.start is not None:
            need(
                isinstance(self.start, (date, datetime)),
                f"`start` must be a date or datetime; got {self.start!r}.",
            )
            m = _EVERY_RE.match(self.every)
            need(
                m is not None and int(m.group(1)) >= 1,
                f"`every` must be a single-unit duration like '1d' or '1mo'; "
                f"got {self.every!r}.",
            )
            assert m is not None
            need(
                isinstance(self.start, datetime) or m.group(2) not in _SUB_DAY_UNITS,
                f"`every={self.every!r}` is shorter than a day, so `start` must be a "
                "datetime, not a date.",
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def plain(cls, **overrides: Any) -> SynthConfig:
        """A clean panel: every complicating mechanism switched off.

        One regime, Gaussian shocks, no stochastic volatility, one cluster, no
        breaks, a balanced panel (no entry or exit, every entity observed at
        every step), no missing values, and each value known at its own event
        time and never revised. The factors, the idiosyncratic AR(1) and the
        planted signal remain. Keyword arguments switch individual dials back
        on.

        Parameters
        ----------
        **overrides : Any
            Dials to set on top of the plain baseline.

        Returns
        -------
        SynthConfig
        """
        base: dict[str, Any] = {
            "n_regimes": 1,
            "sv_vol": 0.0,
            "tail_df": None,
            "n_clusters": 1,
            "cluster_strength": 0.0,
            "break_rate": 0.0,
            "initial_fraction": 1.0,
            "entry_rate": 0.0,
            "exit_rate": 0.0,
            "observation_periods": (1,),
            "observation_prob": 1.0,
            "missing_rate": 0.0,
            "missing_informativeness": 0.0,
            "reporting_lag": (0, 0),
            "reporting_jitter": 0.0,
            "revision_prob": 0.0,
        }
        base.update(overrides)
        return cls(**base)

    def replace(self, **changes: Any) -> SynthConfig:
        """Return a copy with some dials changed (validated).

        Parameters
        ----------
        **changes : Any
            Dial names and their new values.

        Returns
        -------
        SynthConfig

        Raises
        ------
        TypeError
            If a name is not a dial.
        """
        unknown = sorted(set(changes) - {f.name for f in dataclasses.fields(self)})
        if unknown:
            raise TypeError(
                f"Unknown SynthConfig dial(s) {unknown}; valid dials are "
                f"{sorted(f.name for f in dataclasses.fields(self))}."
            )
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """The dials as a plain ``dict`` (for logging or a manifest)."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------- #
# A prior over the dials
# ---------------------------------------------------------------------- #
_OBSERVATION_SCHEDULES: tuple[tuple[int, ...], ...] = (
    (1,),
    (1,),
    (1, 5),
    (1, 5, 21),
    (5,),
)


def sample_config(
    seed: int,
    *,
    n_entities: tuple[int, int] = (10, 200),
    n_periods: tuple[int, int] = (50, 500),
    **fixed: Any,
) -> SynthConfig:
    """Draw a :class:`SynthConfig` from a broad prior over the dial space.

    ``n_entities`` and ``n_periods`` are log-uniform on the given inclusive
    ranges; every other dial is drawn from a fixed, documented-in-code range
    that spans "clean" to "messy". Use it with a second seed for
    :func:`~panelary.synth.generate_panel` to sample many panels across the
    dial space::

        for k in range(1000):
            cfg = sample_config(seed=k)
            data = generate_panel(cfg, seed=k)

    Parameters
    ----------
    seed : int
        Seed of the dial draw (non-negative). The same seed always gives the
        same config.
    n_entities, n_periods : tuple of (int, int)
        Inclusive ranges of the entity pool size and the series length.
    **fixed : Any
        Dials to pin instead of drawing.

    Returns
    -------
    SynthConfig

    Raises
    ------
    ValueError
        If ``seed`` is negative or a range is empty.
    """
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"`seed` must be a non-negative int; got {seed!r}.")
    for name, (lo, hi) in (("n_entities", n_entities), ("n_periods", n_periods)):
        if not (_is_int(lo) and _is_int(hi) and 1 <= lo <= hi):
            raise ValueError(f"`{name}` must be an int range 1 <= lo <= hi.")
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(7919,)))
    )

    def log_int(lo: int, hi: int) -> int:
        return int(
            min(hi, math.floor(math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))))
        )

    def uni(lo: float, hi: float) -> float:
        return float(rng.uniform(lo, hi))

    def integer(lo: int, hi: int) -> int:
        return int(lo + min(hi - lo, math.floor(rng.random() * (hi - lo + 1))))

    lag_lo = integer(0, 3)
    gap_lo = integer(1, 3)
    drawn: dict[str, Any] = {
        "n_entities": log_int(*n_entities),
        "n_periods": log_int(*n_periods),
        "n_factors": integer(0, 5),
        "factor_persistence": uni(0.5, 0.98),
        "n_regimes": integer(1, 3),
        "regime_persistence": uni(0.9, 0.995),
        "regime_vol_ratio": uni(1.0, 3.0),
        "regime_mean_shift": uni(0.0, 0.5),
        "sv_persistence": uni(0.8, 0.98),
        "sv_vol": uni(0.0, 0.4),
        "tail_df": None
        if rng.random() < 0.3
        else float(math.exp(uni(math.log(3.0), math.log(30.0)))),
        "n_clusters": integer(1, 8),
        "cluster_strength": uni(0.0, 1.0),
        "loading_dispersion": uni(0.1, 0.6),
        "idio_persistence": uni(0.0, 0.8),
        "idio_vol": uni(0.5, 2.0),
        "intercept_dispersion": uni(0.0, 1.0),
        "signal_lag": integer(1, 5),
        "signal_coef": uni(-1.0, 1.0),
        "signal_observed": bool(rng.random() < 0.5),
        "break_rate": uni(0.0, 0.02),
        "break_size": uni(0.0, 2.0),
        "initial_fraction": uni(0.3, 1.0),
        "entry_rate": uni(0.0, 0.05),
        "exit_rate": uni(0.0, 0.02),
        "observation_periods": _OBSERVATION_SCHEDULES[
            integer(0, len(_OBSERVATION_SCHEDULES) - 1)
        ],
        "observation_prob": uni(0.7, 1.0),
        "missing_rate": uni(0.0, 0.3),
        "missing_informativeness": 0.0 if rng.random() < 0.4 else uni(-2.0, 2.0),
        "reporting_lag": (lag_lo, lag_lo + integer(0, 5)),
        "reporting_jitter": uni(0.0, 0.3),
        "revision_prob": uni(0.0, 0.6),
        "max_revisions": integer(1, 3),
        "revision_noise": uni(0.0, 1.0),
        "revision_gap": (gap_lo, gap_lo + integer(0, 7)),
    }
    drawn.update(fixed)
    return SynthConfig(**drawn)
