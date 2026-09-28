"""Guardrails for embedding-based forecasts: the checks none of the surveyed libraries ship.

Build contract section 5, items 3 - 6 and 8. Every function here is cheap, and
each answers one specific way an embedding result can be wrong while looking
right:

:func:`reversal_check`
    Is the pipeline learning, or mechanically timing momentum? Inject a strongly
    mean-reverting MA(2) target, re-run the user's pipeline, and read the sign
    of its response to recent returns. This is the test that exposed the
    flagship random-features-in-finance result (Nagel, 2025, NBER w34104).
:func:`mechanical_baseline`
    The recency-weighted, inverse-volatility-scaled average of the target
    over the past: what a ridgeless random-feature regression degenerates
    into. If an embedding does not beat it, it learned nothing.
:func:`naive_baselines`
    ``y_t = y_{t-h}`` and the seasonal naive -- "did you beat the random walk?"
:func:`null_panel` / :func:`null_distribution`
    A falsification harness: phase-randomised or martingale-difference panels
    with the real panel's shape and missingness, and the score distribution
    under them.
:func:`baseline_report`
    Model and baselines, side by side, always.

What these are not
------------------
**Deflated Sharpe ratios and PBO are not leakage detectors.** They correct for
*search intensity* (how many configurations were tried). A leaky oracle at
Sharpe 35 passes both (arXiv:2608.27734). Leakage has to be excluded
structurally -- by the ``leakage_safe`` / ``fit_is_empty`` contract and the
perturbation and prefix-invariance tests -- not detected after the fact. The
null harness here is a falsification test of *signal*, not a leak gate either:
a leaky pipeline can beat every null.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "NullReport",
    "ReversalReport",
    "baseline_report",
    "mechanical_baseline",
    "naive_baselines",
    "null_distribution",
    "null_panel",
    "reversal_check",
]


def _frame(
    panel: Any, entity: str | None, time: str | None
) -> tuple[pl.DataFrame, str, str]:
    pf = (
        panel
        if isinstance(panel, PanelFrame)
        else as_panel(panel, entity=entity, time=time)
    )
    return pf.collect(), pf.entity_col, pf.time_col


def _ols(y: NDArray[np.float64], X: NDArray[np.float64]) -> NDArray[np.float64]:
    A = np.column_stack([np.ones(X.shape[0]), X])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.asarray(coef[1:], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Reversal-synthetic diagnostic
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReversalReport:
    """Outcome of :func:`reversal_check`.

    Attributes
    ----------
    true_coefs : tuple of float
        The injected response of the target to ``(r_t, r_{t-1})`` (negative:
        mean reversion).
    learned_coefs : tuple of float
        OLS response of the pipeline's out-of-sample predictions to the same
        two returns.
    signal_corr : float
        Correlation of predictions with the noiseless injected signal.
    passed : bool
        ``learned_coefs[0] < 0``: the pipeline reversed, as the data do. A
        positive response on recent returns under a reversal target is the
        Nagel failure: mechanical (volatility-timed) momentum.
    n_train, n_test : int
    """

    true_coefs: tuple[float, float]
    learned_coefs: tuple[float, float]
    signal_corr: float
    passed: bool
    n_train: int
    n_test: int


def reversal_check(
    fit_predict: Callable[[pl.DataFrame, pl.DataFrame, str], Any],
    panel: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    return_col: str,
    coefs: tuple[float, float] = (0.6, 0.3),
    noise: float = 0.5,
    train_fraction: float = 0.6,
    target_name: str = "__reversal_target__",
    seed: int = 0,
    entity: str | None = None,
    time: str | None = None,
) -> ReversalReport:
    """Run a pipeline on a strongly mean-reverting synthetic target.

    The target at row ``t`` is ``-(a1 r_t + a2 r_{t-1}) / sd(r) + noise * e_t``
    (per entity, ``e`` standard normal) -- an MA(2) reversal in the *next*
    period. The panel is split by date at ``train_fraction``; the pipeline is
    trained on the early dates and predicts the late ones.

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(train, test, target_name) -> predictions`` for the rows of
        ``test`` in order. It must do everything the real pipeline does
        (embedding, compression, readout) -- fitting only on ``train``.
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
    return_col : str
        The one-period return column the pipeline sees.
    coefs : (float, float), default (0.6, 0.3)
        ``(a1, a2)``, both ``> 0`` (the target reverses).
    noise : float, default 0.5
    train_fraction : float, default 0.6
    target_name : str
    seed : int, default 0
    entity, time : str, optional

    Returns
    -------
    ReversalReport
    """
    df, ent, tim = _frame(panel, entity, time)
    a1, a2 = (float(c) for c in coefs)
    if a1 <= 0 or a2 < 0:
        raise ValueError(
            "reversal_check: `coefs` must be (a1 > 0, a2 >= 0) -- a reversal."
        )
    df = df.sort(ent, tim)
    r = df[return_col].cast(pl.Float64)
    r_np = r.drop_nans().drop_nulls().to_numpy()
    sd = float(r_np.std(ddof=1)) if r_np.size > 1 else 1.0
    sd = sd if sd > 0 else 1.0
    rng = np.random.default_rng([int(seed), 7])
    df = df.with_columns(
        pl.col(return_col).cast(pl.Float64).alias("__r0__"),
        pl.col(return_col).cast(pl.Float64).shift(1).over(ent).alias("__r1__"),
    ).with_columns(
        (
            -(a1 * pl.col("__r0__") + a2 * pl.col("__r1__")) / sd
            + pl.Series(rng.standard_normal(df.height) * noise)
        ).alias(target_name),
        (-(a1 * pl.col("__r0__") + a2 * pl.col("__r1__")) / sd).alias("__signal__"),
    )
    dates = df[tim].unique().sort()
    cut = dates[
        max(0, min(dates.len() - 2, int(np.floor(train_fraction * dates.len())) - 1))
    ]
    keep = [c for c in df.columns if c not in ("__r0__", "__r1__", "__signal__")]
    train = df.filter(pl.col(tim) <= cut)
    test = df.filter(pl.col(tim) > cut)
    pred = np.asarray(
        fit_predict(
            train.select(keep), test.select(keep).drop(target_name), target_name
        ),
        dtype=np.float64,
    )
    if pred.shape[0] != test.height:
        raise ValueError(
            f"reversal_check: fit_predict returned {pred.shape[0]} predictions for {test.height} test rows."
        )
    X = test.select("__r0__", "__r1__").to_numpy()
    sig = test["__signal__"].to_numpy()
    ok = np.isfinite(pred) & np.isfinite(X).all(axis=1) & np.isfinite(sig)
    if ok.sum() < 3:
        raise ValueError("reversal_check: fewer than 3 finite test predictions.")
    learned = _ols(pred[ok], X[ok])
    corr = (
        float(np.corrcoef(pred[ok], sig[ok])[0, 1])
        if np.std(pred[ok]) > 0
        else float("nan")
    )
    return ReversalReport(
        true_coefs=(-a1 / sd, -a2 / sd),
        learned_coefs=(float(learned[0]), float(learned[1])),
        signal_corr=corr,
        passed=bool(learned[0] < 0),
        n_train=train.height,
        n_test=int(ok.sum()),
    )


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def mechanical_baseline(
    panel: PanelFrame | pl.DataFrame | pl.LazyFrame,
    target: str,
    *,
    halflife: float = 20.0,
    vol_window: int = 20,
    horizon: int = 1,
    name: str = "mechanical_baseline",
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Recency-weighted, inverse-volatility-scaled average of the past target.

    Per entity, ``z_s = y_s / sd(y_{s-vol_window+1 .. s})`` and the prediction
    at ``t`` is the exponentially weighted mean (half-life ``halflife``) of
    ``z_s`` over ``s <= t - horizon`` -- only targets already realised at ``t``
    (a target at ``s`` is known at ``s + horizon``). This is the volatility-timed
    momentum a ridgeless random-feature regression degenerates into.

    Returns
    -------
    polars.DataFrame
        ``entity, time, name`` in the input's row order.
    """
    df, ent, tim = _frame(panel, entity, time)
    if horizon < 1:
        raise ValueError("mechanical_baseline: `horizon` must be >= 1.")
    y = pl.col(target).cast(pl.Float64)
    z = (y / y.rolling_std(vol_window, min_samples=max(2, vol_window // 2))).shift(
        horizon
    )
    out = (
        df.with_row_index("__pn_i__")
        .sort(ent, tim)
        .with_columns(z.over(ent).alias("__z__"))
        .with_columns(
            pl.col("__z__")
            .ewm_mean(half_life=halflife, ignore_nulls=True)
            .over(ent)
            .alias(name)
        )
        .sort("__pn_i__")
    )
    return out.select(ent, tim, name)


def naive_baselines(
    panel: PanelFrame | pl.DataFrame | pl.LazyFrame,
    target: str,
    *,
    horizon: int = 1,
    season: int | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """The random walk and the seasonal naive, inside the evaluation surface.

    ``naive_last`` is ``y_{t - horizon}`` (the last target already realised at
    ``t``); ``naive_seasonal`` (when ``season`` is given, ``season >=
    horizon``) is ``y_{t - season}``.

    Returns
    -------
    polars.DataFrame
        ``entity, time, naive_last[, naive_seasonal]`` in the input's row order.
    """
    df, ent, tim = _frame(panel, entity, time)
    if horizon < 1:
        raise ValueError("naive_baselines: `horizon` must be >= 1.")
    exprs = [
        pl.col(target).cast(pl.Float64).shift(horizon).over(ent).alias("naive_last")
    ]
    if season is not None:
        if season < horizon:
            raise ValueError(
                "naive_baselines: `season` must be >= `horizon` (else it reads the future)."
            )
        exprs.append(
            pl.col(target)
            .cast(pl.Float64)
            .shift(season)
            .over(ent)
            .alias("naive_seasonal")
        )
    out = (
        df.with_row_index("__pn_i__")
        .sort(ent, tim)
        .with_columns(exprs)
        .sort("__pn_i__")
    )
    return out.select(ent, tim, *[e.meta.output_name() for e in exprs])


def _date_ic(frame: pl.DataFrame, tim: str, y: str, p: str) -> tuple[float, float]:
    ic = (
        frame.filter(
            pl.col(y).is_not_null()
            & pl.col(p).is_not_null()
            & pl.col(y).is_not_nan()
            & pl.col(p).is_not_nan()
        )
        .group_by(tim)
        .agg(
            pl.corr(pl.col(y).rank(), pl.col(p).rank()).alias("ic"), pl.len().alias("n")
        )
        .filter((pl.col("n") >= 3) & pl.col("ic").is_not_nan())["ic"]
        .to_numpy()
    )
    if ic.size == 0:
        return float("nan"), float("nan")
    m = float(ic.mean())
    s = float(ic.std(ddof=1)) if ic.size > 1 else float("nan")
    return m, (
        m / s * np.sqrt(ic.size) if s and np.isfinite(s) and s > 0 else float("nan")
    )


def baseline_report(
    frame: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    target: str,
    predictions: Sequence[str] | Mapping[str, str],
    horizon: int = 1,
    halflife: float = 20.0,
    vol_window: int = 20,
    season: int | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """Score model predictions and the baselines side by side -- always both.

    Metrics per forecast, over rows where it and the target are present:
    mean per-date rank IC and its t-statistic, and the pooled out-of-sample
    ``R^2`` against a zero forecast, ``1 - sum (y - yhat)^2 / sum y^2``.

    Parameters
    ----------
    frame : PanelFrame | polars.DataFrame | polars.LazyFrame
        Must hold ``target`` and the prediction columns (evaluate on test rows).
    target : str
    predictions : sequence of str or mapping of label to column
    horizon, halflife, vol_window, season
        Passed to :func:`mechanical_baseline` / :func:`naive_baselines`.

    Returns
    -------
    polars.DataFrame
        ``forecast, kind ("model" | "baseline"), ic_mean, ic_tstat, r2_oos, n``.
    """
    df, ent, tim = _frame(frame, entity, time)
    preds = (
        dict(predictions)
        if isinstance(predictions, Mapping)
        else {p: p for p in predictions}
    )
    base = naive_baselines(
        df, target, horizon=horizon, season=season, entity=ent, time=tim
    )
    mech = mechanical_baseline(
        df,
        target,
        halflife=halflife,
        vol_window=vol_window,
        horizon=horizon,
        entity=ent,
        time=tim,
    )
    work = df.with_columns(base.drop(ent, tim).get_columns()).with_columns(
        mech.drop(ent, tim).get_columns()
    )
    rows: list[dict[str, Any]] = []
    entries = [(label, col, "model") for label, col in preds.items()]
    entries += [
        (c, c, "baseline")
        for c in [*base.drop(ent, tim).columns, "mechanical_baseline"]
    ]
    for label, col, kind in entries:
        y = work[target].cast(pl.Float64).to_numpy()
        p = work[col].cast(pl.Float64).to_numpy()
        ok = np.isfinite(y) & np.isfinite(p)
        den = float(np.sum(y[ok] ** 2))
        r2 = (
            1.0 - float(np.sum((y[ok] - p[ok]) ** 2)) / den if den > 0 else float("nan")
        )
        ic, t = _date_ic(
            work.select(
                tim,
                pl.col(target).cast(pl.Float64).alias("__y__"),
                pl.col(col).cast(pl.Float64).alias("__p__"),
            ),
            tim,
            "__y__",
            "__p__",
        )
        rows.append(
            {
                "forecast": label,
                "kind": kind,
                "ic_mean": ic,
                "ic_tstat": t,
                "r2_oos": r2,
                "n": int(ok.sum()),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "forecast": pl.Utf8,
            "kind": pl.Utf8,
            "ic_mean": pl.Float64,
            "ic_tstat": pl.Float64,
            "r2_oos": pl.Float64,
            "n": pl.Int64,
        },
    )


# --------------------------------------------------------------------------- #
# Null-panel falsification harness
# --------------------------------------------------------------------------- #
def _phase_surrogate(
    x: NDArray[np.float64], rng: np.random.Generator, phases: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    n = x.shape[0]
    if n < 3:
        return x.copy()
    mu = x.mean()
    F = np.fft.rfft(x - mu)
    ph = np.asarray(
        rng.uniform(0.0, 2.0 * np.pi, F.shape[0])
        if phases is None
        else phases[: F.shape[0]],
        dtype=np.float64,
    )
    rot = np.exp(1j * ph)
    rot[0] = 1.0
    if n % 2 == 0:
        # The Nyquist bin of a real series is real: keep its modulus, pick a sign.
        rot[-1] = 1.0 if np.cos(ph[-1]) >= 0 else -1.0
    return np.fft.irfft(F * rot, n=n) + mu


def null_panel(
    panel: PanelFrame | pl.DataFrame | pl.LazyFrame,
    columns: Sequence[str],
    *,
    method: str = "sign",
    joint: bool = True,
    seed: int = 0,
    entity: str | None = None,
    time: str | None = None,
) -> pl.DataFrame:
    """A null panel with the real panel's keys, shape and missingness.

    Parameters
    ----------
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
    columns : sequence of str
        Numeric columns to replace; every other column is kept as is.
    method : {"sign", "phase"}, default "sign"
        ``"sign"``: a martingale-difference panel -- each entity's observed
        values are demeaned and multiplied by iid random signs (volatility
        clustering in ``|x|`` survives; predictability of the sign does not).
        ``"phase"``: a phase-randomised surrogate per entity -- the power
        spectrum (hence the autocorrelation) and mean survive; everything
        nonlinear and every cross-entity alignment does not.
    joint : bool, default True
        Use the same signs / phases for all ``columns`` of an entity, keeping
        their contemporaneous cross-correlation.
    seed : int, default 0

    Returns
    -------
    polars.DataFrame
        Same rows, same order, nulls/NaNs exactly where the input had them.
    """
    if method not in ("sign", "phase"):
        raise ValueError(
            f"null_panel: `method` must be 'sign' or 'phase', got {method!r}."
        )
    df, ent, tim = _frame(panel, entity, time)
    work = df.with_row_index("__pn_i__").sort(ent, tim)
    vals = {
        c: work[c].cast(pl.Float64).fill_null(np.nan).to_numpy().copy() for c in columns
    }
    lengths = work.group_by(ent, maintain_order=True).len()["len"].to_numpy()
    rng = np.random.default_rng([int(seed), 11])
    start = 0
    for n_e in lengths.tolist():
        sl = slice(start, start + n_e)
        shared_sign = rng.choice([-1.0, 1.0], size=n_e)
        shared_phase = rng.uniform(0.0, 2.0 * np.pi, n_e // 2 + 1)
        for c in columns:
            x = vals[c][sl]
            ok = np.isfinite(x)
            if ok.sum() < 2:
                continue
            obs = x[ok]
            if method == "sign":
                s = (
                    shared_sign[ok]
                    if joint
                    else rng.choice([-1.0, 1.0], size=obs.shape[0])
                )
                x[ok] = obs.mean() + s * (obs - obs.mean())
            else:
                x[ok] = _phase_surrogate(obs, rng, shared_phase if joint else None)
            vals[c][sl] = x
        start += n_e
    out = work.with_columns(
        pl.Series(c, vals[c]).fill_nan(None).cast(df.schema[c])
        if df[c].null_count()
        else pl.Series(c, vals[c]).cast(df.schema[c])
        for c in columns
    )
    return out.sort("__pn_i__").drop("__pn_i__")


@dataclass(frozen=True)
class NullReport:
    """Outcome of :func:`null_distribution`.

    Attributes
    ----------
    observed : float
        The score on the real panel.
    null : numpy.ndarray
        Scores on ``n_draws`` null panels.
    p_value : float
        ``(1 + #{null >= observed}) / (1 + n_draws)``.
    """

    observed: float
    null: NDArray[np.float64]
    p_value: float


def null_distribution(
    score_fn: Callable[[pl.DataFrame], float],
    panel: PanelFrame | pl.DataFrame | pl.LazyFrame,
    columns: Sequence[str],
    *,
    n_draws: int = 20,
    method: str = "sign",
    joint: bool = True,
    seed: int = 0,
    entity: str | None = None,
    time: str | None = None,
) -> NullReport:
    """Score the real panel and ``n_draws`` null panels; report the null distribution.

    ``score_fn`` must run the *whole* pipeline (fit on its own training split,
    score out of sample) and return one number where larger is better. This is
    a falsification test of signal, **not** a leakage gate: a leaky pipeline
    can beat every null (see the module docstring on DSR / PBO).

    Returns
    -------
    NullReport
    """
    if n_draws < 1:
        raise ValueError("null_distribution: `n_draws` must be >= 1.")
    df, ent, tim = _frame(panel, entity, time)
    observed = float(score_fn(df))
    null = np.array(
        [
            float(
                score_fn(
                    null_panel(
                        df,
                        columns,
                        method=method,
                        joint=joint,
                        seed=seed + i,
                        entity=ent,
                        time=tim,
                    )
                )
            )
            for i in range(n_draws)
        ]
    )
    p = (1.0 + float(np.sum(null >= observed))) / (1.0 + n_draws)
    return NullReport(observed=observed, null=null, p_value=p)
