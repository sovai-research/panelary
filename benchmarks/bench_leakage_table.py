"""Borrowed accuracy: a scored table of common preprocessing steps.

Every preprocessing step can be written two ways -- *permissively*, the way a
practitioner reaches for it, and *point-in-time*, constrained to information
available at prediction time.  The out-of-sample gap between the two is the
accuracy **borrowed** from data the method will not have when it runs for real.
This script measures that gap, step by step, on a synthetic panel whose signal
is known by construction, and cross-checks every permissive form against
:func:`panelary.testing.assert_no_lookahead`.

The cross-check is the point.  A step with a large gap that the verifier does
*not* flag means one of the two instruments is wrong; a step the verifier flags
whose gap is ~0 means the leak is real but carries no accuracy on this
data-generating process.  Both cells are reported, neither is hidden.

**What is measured.**  For each step, the design matrix handed to the model is
that step's output and nothing else (plus an intercept), so the number isolates
the step rather than a pipeline.  Both modes are scored on exactly the same
rows -- the intersection of rows where both constructions are defined -- and
with the same purged, embargoed folds from
:mod:`panelary.core.model_selection`.  The learner is a ridge implemented here
in numpy; no scikit-learn, so this runs on the bare-core install.

**What this is not.**  The gap for any step is a property of *this* DGP with
*these* seeds, not a universal constant.  A step that borrows nothing here can
be ruinous on data with different persistence, drift or cardinality.  That is
why the table reports min/median/max across seeds rather than one number.

Run::

    python benchmarks/bench_leakage_table.py                  # ~20k rows, 5 seeds
    python benchmarks/bench_leakage_table.py --rows 200000
    python benchmarks/bench_leakage_table.py --long           # bigger, more seeds
    python benchmarks/bench_leakage_table.py --only shift_negative
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

import panelary  # noqa: F401  (registers the .panel/.xs/.ts namespaces)
from panelary.core.model_selection import PurgedKFold
from panelary.core.panel_frame import PanelFrame
from panelary.testing import assert_no_lookahead

ENTITY = "entity"
TIME = "time"

#: Trailing/centred window used by the rolling and robust-scaling rows.
WINDOW = 9
#: Resampling period (time steps) for the aggregation row.
PERIOD = 10
#: Window used to spell an *expanding* quantile as a rolling one.  It is a
#: fixed constant, not a function of series length, so the feature stays
#: prefix-invariant (AGENTS.md invariant 1); the window simply never fills.
EXPANDING = 1_000_000
#: Target rows per categorical level; fixes the target-encoding leak's strength
#: as ``--rows`` changes, rather than letting cardinality drift with panel size.
ROWS_PER_LEVEL = 40

FitFn = Callable[[pl.DataFrame, pl.DataFrame], np.ndarray]


# --------------------------------------------------------------------------- #
# Data-generating process
# --------------------------------------------------------------------------- #
def _ar1(
    rng: np.random.Generator, shape: tuple[int, ...], rho: float, sd: float = 1.0
) -> np.ndarray:
    """Stationary AR(1) along the last axis with *process* standard deviation ``sd``.

    Parameterising by the process sd rather than the innovation sd matters:
    an AR(1) at ``rho=0.95`` amplifies its innovations by ``1/sqrt(1-rho^2)``
    ~ 3.2x, so "scale the innovations by 0.9" silently produces a process with
    sd 2.9 -- which, once exponentiated into a volatility, gives one seed in ten
    an entity whose series runs to five figures and swamps every pooled score.
    """
    innovation = sd * math.sqrt(1.0 - rho * rho)
    eps = rng.standard_normal(shape) * innovation
    out = np.empty(shape, dtype=np.float64)
    out[..., 0] = rng.standard_normal(shape[:-1]) * sd
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + eps[..., t]
    return out


def make_panel(n_entities: int, n_time: int, seed: int) -> pl.DataFrame:
    """A panel with a known signal, persistence, drift, vol and a categorical.

    Row ``(e, t)`` carries features computed from data at or before ``t`` and a
    label realised at ``t + 1`` -- the standard forecasting alignment, and the
    reason a negative shift is fatal rather than merely untidy.  The label is a
    function of the DGP's shocks only; nothing in it depends on anything the
    *features* are not allowed to see.

    Four ingredients, each of which makes a different family of leak bite:

    - ``u``  a persistent idiosyncratic AR(1) -- so a *future* observation of
      ``x`` is informative about the present label (centred windows, backward
      fill, negative shifts).
    - ``level``  a per-entity random walk -- so whole-sample statistics encode
      where the series is going (scalers, imputers, global ranks).
    - ``sigma``  a persistent stochastic volatility -- so rolling *std* is a
      real feature rather than noise.
    - ``cat``  a moderate-cardinality categorical with a real additive effect --
      so target encoding has something honest to find, and something dishonest
      to absorb.
    """
    rng = np.random.default_rng(seed)
    t_gen = n_time + 1  # one extra step: the label at row t is realised at t+1

    # Log-volatility with a process sd of 0.35: sigma spans roughly 0.5x-2x,
    # persistent enough for a trailing rolling std to estimate, tame enough
    # that no single entity's excursion decides the seed's score.
    sigma = np.exp(_ar1(rng, (n_entities, t_gen), 0.95, sd=0.35))

    eps = rng.standard_normal((n_entities, t_gen)) * sigma
    u = np.empty((n_entities, t_gen), dtype=np.float64)
    u[:, 0] = eps[:, 0] / math.sqrt(1.0 - 0.81)
    for t in range(1, t_gen):
        u[:, t] = 0.90 * u[:, t - 1] + eps[:, t]

    f = _ar1(rng, (t_gen,), 0.85, sd=1.0)

    level = rng.standard_normal((n_entities, t_gen)).cumsum(axis=1) * 0.10
    level += rng.standard_normal((n_entities, 1)) * 1.50

    x_true = u + 0.5 * f[None, :] + level
    x = x_true + 0.6 * rng.standard_normal((n_entities, t_gen))

    n_rows = n_entities * n_time
    n_levels = max(20, n_rows // ROWS_PER_LEVEL)
    cat = rng.integers(0, n_levels, size=(n_entities, n_time))
    g = rng.standard_normal(n_levels)

    y = (
        1.0 * u[:, 1:]
        + 0.6 * f[None, 1:]
        + 0.9 * g[cat]
        + 3.0 * (sigma[:, 1:] - 1.0)
        + 4.0 * rng.standard_normal((n_entities, n_time))
    )

    # A gappy copy of x: runs of 1-4 consecutive missing steps, ~15% of rows.
    xg = x[:, :n_time].copy()
    n_runs = max(1, int(0.05 * n_rows))
    run_e = rng.integers(0, n_entities, size=n_runs)
    run_t = rng.integers(0, max(1, n_time - 5), size=n_runs)
    run_len = rng.integers(1, 5, size=n_runs)
    for e, t0, ln in zip(run_e, run_t, run_len, strict=True):
        xg[e, t0 : t0 + ln] = np.nan

    return pl.DataFrame(
        {
            ENTITY: np.repeat(np.arange(n_entities, dtype=np.int64), n_time),
            TIME: np.tile(np.arange(n_time, dtype=np.int64), n_entities),
            "x": x[:, :n_time].reshape(-1),
            "xg": xg.reshape(-1),
            # String on purpose: `assert_no_lookahead` perturbs every *numeric*
            # column, and scrambling the group labels is not the experiment.
            "cat": pl.Series(cat.reshape(-1)).cast(pl.Utf8),
            "y": y.reshape(-1),
        }
    ).with_columns(pl.col("xg").fill_nan(None))


# --------------------------------------------------------------------------- #
# Point-in-time expression helpers
# --------------------------------------------------------------------------- #
def _exp_moments(col: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Expanding (<= t) within-entity ``(n, sum, sum of squares)``, null-skipping.

    Polars' ``cum_sum`` emits *null* at a null input row rather than carrying
    the running total across it, so the naive ``cum_sum() / cum_count()`` is
    null exactly where an imputer needs a value.  Zero-filling the summands and
    counting non-nulls separately gives the running moments over the observed
    prefix, which is what a point-in-time statistic means on a gappy series.
    """
    c = pl.col(col)
    observed = c.is_not_null()
    n = observed.cast(pl.Int64).cum_sum().over(ENTITY, order_by=TIME)
    s1 = c.fill_null(0.0).cum_sum().over(ENTITY, order_by=TIME)
    s2 = (c.fill_null(0.0) ** 2).cum_sum().over(ENTITY, order_by=TIME)
    return n, s1, s2


def _exp_mean(col: str) -> pl.Expr:
    """Expanding (<= t) within-entity mean -- the exact PIT mean."""
    n, s1, _ = _exp_moments(col)
    return pl.when(n > 0).then(s1 / n).otherwise(None)


def _exp_std(col: str) -> pl.Expr:
    """Expanding (<= t) within-entity sample std -- the exact PIT std."""
    n, s1, s2 = _exp_moments(col)
    var = (s2 - s1 * s1 / n) / (n - 1)
    return pl.when(n > 1).then(var.clip(lower_bound=0.0).sqrt()).otherwise(None)


def _expanding_q(q: float) -> pl.Expr:
    """Expanding (<= t) within-entity quantile of ``x``.

    Polars has no cumulative quantile, so this is a rolling one whose window is
    a fixed constant far larger than any series here: the window never fills, so
    every row sees exactly its own prefix.
    """
    return _over(
        pl.col("x").rolling_quantile(q, window_size=EXPANDING, min_samples=WINDOW)
    )


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """``num / den``, but null rather than NaN where the scale is degenerate.

    A ``0 / 0`` at an entity's first observation is not a leak, but Polars
    orders NaN above every float, so the future-perturbation verifier's
    ``|a - b| > tol`` test fires on a NaN-to-NaN comparison and reports a
    perfectly causal expression as leaky.  Guarding the denominator keeps the
    verifier measuring causality rather than arithmetic.
    """
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(None)


def _over(expr: pl.Expr) -> pl.Expr:
    """Within-entity, in time order -- never bare ``.over(entity)``."""
    return expr.over(ENTITY, order_by=TIME)


# --------------------------------------------------------------------------- #
# Fitted steps (the "fit on everything vs fit on the train fold" family)
# --------------------------------------------------------------------------- #
def _fit_zscore(fit: pl.DataFrame, apply: pl.DataFrame) -> np.ndarray:
    mu = fit["x"].mean()
    sd = fit["x"].std()
    sd = 1.0 if sd is None or sd < 1e-12 else sd
    return ((apply["x"].to_numpy() - float(mu)) / float(sd)).reshape(-1, 1)


def _fit_winsorise(fit: pl.DataFrame, apply: pl.DataFrame) -> np.ndarray:
    lo = float(fit["x"].quantile(0.01))
    hi = float(fit["x"].quantile(0.99))
    return np.clip(apply["x"].to_numpy(), lo, hi).reshape(-1, 1)


def _fit_target_encode(fit: pl.DataFrame, apply: pl.DataFrame) -> np.ndarray:
    prior = float(fit["y"].mean())
    table = fit.group_by("cat").agg(pl.col("y").mean().alias("__te"))
    enc = (
        apply.select("cat")
        .join(table, on="cat", how="left")
        .select(pl.col("__te").fill_null(prior))
        .to_numpy()
    )
    return enc.astype(np.float64).reshape(-1, 1)


#: The four stages of the pipeline row, in the order they run.  Each can be
#: fitted on the whole panel or on the training fold independently, which is
#: what makes the Shapley decomposition below possible.
PIPELINE_COMPONENTS = ("impute", "winsorise", "scale", "encode")


def _pipeline(apply: pl.DataFrame, source: dict[str, pl.DataFrame]) -> np.ndarray:
    """Impute -> winsorise -> scale -> target-encode, each fitted independently.

    ``source[name]`` is the frame that stage's statistics are fitted on: the
    whole panel for a permissive stage, the training fold for a point-in-time
    one.  The "split last" antipattern is every stage pointed at the whole
    panel; the correct pipeline is every stage pointed at the training fold.
    """
    mean_xg = float(source["impute"]["xg"].mean())
    winsor_fit = source["winsorise"]["xg"].fill_null(mean_xg)
    lo = float(winsor_fit.quantile(0.01))
    hi = float(winsor_fit.quantile(0.99))

    scale_fit = source["scale"]["xg"].fill_null(mean_xg).clip(lo, hi)
    mu = float(scale_fit.mean())
    sd = float(scale_fit.std() or 1.0)
    sd = 1.0 if sd < 1e-12 else sd

    filled = apply["xg"].fill_null(mean_xg).to_numpy()
    scaled = (np.clip(filled, lo, hi) - mu) / sd
    encoded = _fit_target_encode(source["encode"], apply)[:, 0]
    return np.column_stack([scaled, encoded])


def _fit_pipeline(fit: pl.DataFrame, apply: pl.DataFrame) -> np.ndarray:
    """The whole pipeline with every stage fitted on ``fit``."""
    return _pipeline(apply, dict.fromkeys(PIPELINE_COMPONENTS, fit))


# --------------------------------------------------------------------------- #
# The step table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One preprocessing step, written permissively and point-in-time.

    ``kind="expr"`` steps are row-local: both constructions are whole-panel
    expressions, and the point-in-time one is prefix-invariant, so evaluating it
    on the full panel is identical to evaluating it fold by fold.
    ``kind="fit"`` steps have a fitted statistic: the permissive mode fits on
    the whole panel, the point-in-time mode refits on each training fold.
    """

    key: str
    label: str
    kind: str  # "expr" | "fit"
    note: str
    perm: Sequence[pl.Expr] | FitFn
    pit: Sequence[pl.Expr] | FitFn | None = None
    pit_checkable: bool = True
    _: dict = field(default_factory=dict, repr=False)


def _steps() -> list[Step]:
    x, xg = pl.col("x"), pl.col("xg")
    period = (pl.col(TIME) // PERIOD).alias("__period")
    period_mean = x.mean().over([pl.col(ENTITY), period])

    return [
        Step(
            "control_identity",
            "[control] same op both sides",
            "expr",
            "a null player: the gap must be exactly zero, or the harness is wrong",
            [_over(x.shift(1))],
            [_over(x.shift(1))],
        ),
        Step(
            "zscore_entity",
            "z-score, per entity",
            "expr",
            "whole-history mean/std vs expanding mean/std",
            [_safe_div(x - _over(x.mean()), _over(x.std()))],
            [_safe_div(x - _exp_mean("x"), _exp_std("x"))],
        ),
        Step(
            "minmax_entity",
            "min-max scale, per entity",
            "expr",
            "whole-history min/max vs cumulative min/max",
            [_safe_div(x - _over(x.min()), _over(x.max()) - _over(x.min()))],
            [
                _safe_div(
                    x - _over(x.cum_min()), _over(x.cum_max()) - _over(x.cum_min())
                )
            ],
        ),
        Step(
            "robust_entity",
            "robust scale (median/IQR)",
            "expr",
            "whole-history median/IQR vs expanding median/IQR",
            [
                _safe_div(
                    x - _over(x.median()),
                    _over(x.quantile(0.75)) - _over(x.quantile(0.25)),
                )
            ],
            [
                _safe_div(
                    x - _expanding_q(0.50),
                    _expanding_q(0.75) - _expanding_q(0.25),
                )
            ],
        ),
        Step(
            "impute_mean",
            "mean imputation",
            "expr",
            "whole-history entity mean vs expanding mean",
            [_over(xg.fill_null(xg.mean()))],
            [xg.fill_null(_exp_mean("xg"))],
        ),
        Step(
            "fill_backward",
            "backward vs forward fill",
            "expr",
            "bfill pulls the next observation back over the gap",
            [_over(xg.fill_null(strategy="backward"))],
            [_over(xg.fill_null(strategy="forward"))],
        ),
        Step(
            "interpolate",
            "linear interpolation",
            "expr",
            "bidirectional interpolate vs last-observation-carried-forward",
            [_over(xg.interpolate())],
            [_over(xg.fill_null(strategy="forward"))],
        ),
        Step(
            "rolling_mean",
            "rolling mean, centred vs trailing",
            "expr",
            f"window {WINDOW}",
            [_over(x.rolling_mean(WINDOW, center=True, min_samples=WINDOW))],
            [_over(x.rolling_mean(WINDOW, min_samples=WINDOW))],
        ),
        Step(
            "rolling_std",
            "rolling std, centred vs trailing",
            "expr",
            f"window {WINDOW}",
            [_over(x.rolling_std(WINDOW, center=True, min_samples=WINDOW))],
            [_over(x.rolling_std(WINDOW, min_samples=WINDOW))],
        ),
        Step(
            "xs_rank",
            "ranking, global vs cross-sectional",
            "expr",
            "rank over the whole column vs rank within each date",
            [x.rank()],
            [x.rank().over(TIME)],
        ),
        Step(
            "differencing",
            "differencing, forward vs backward",
            "expr",
            "x[t]-x[t+1] vs x[t]-x[t-1]",
            [_over(x.diff(-1))],
            [_over(x.diff(1))],
        ),
        Step(
            "shift_negative",
            "lag with a negative shift",
            "expr",
            "shift(-1) vs shift(+1)",
            [_over(x.shift(-1))],
            [_over(x.shift(1))],
        ),
        Step(
            "resample_agg",
            "period aggregation, broadcast back",
            "expr",
            f"current {PERIOD}-step period mean vs the previous completed one",
            [period_mean],
            [_over(period_mean.shift(PERIOD))],
        ),
        Step(
            "scale_before_split",
            "scaling before the train/test split",
            "fit",
            "one global affine transform; the learner is affine-invariant",
            _fit_zscore,
            _fit_zscore,
            pit_checkable=False,
        ),
        Step(
            "winsorise",
            "winsorise at global quantiles",
            "fit",
            "clip at 1%/99% of everything vs of the training fold",
            _fit_winsorise,
            _fit_winsorise,
            pit_checkable=False,
        ),
        Step(
            "target_encode",
            "target encoding of a categorical",
            "fit",
            f"mean of y per level; ~{ROWS_PER_LEVEL} rows per level",
            _fit_target_encode,
            _fit_target_encode,
            pit_checkable=False,
        ),
        Step(
            "pipeline_split_last",
            "whole pipeline fitted before the split",
            "fit",
            "impute + winsorise + scale + target-encode, together",
            _fit_pipeline,
            _fit_pipeline,
            pit_checkable=False,
        ),
    ]


# --------------------------------------------------------------------------- #
# Model + scoring
# --------------------------------------------------------------------------- #
def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Ridge fit on the training fold, predictions for the test fold.

    The feature standardisation here is itself fitted on the training fold, so
    the model contributes no leak of its own.  It also makes the learner exactly
    invariant to any global affine rescaling of a feature -- which is not a
    dodge but the finding: global feature scaling cannot borrow accuracy from a
    learner that does not care about feature scale.
    """
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z_train = (x_train - mu) / sd
    z_test = (x_test - mu) / sd
    y_bar = float(y_train.mean())

    k = z_train.shape[1]
    gram = z_train.T @ z_train + alpha * np.eye(k, dtype=np.float64)
    rhs = z_train.T @ (y_train - y_bar)
    # NumPy 2.0 mis-solves solve(A, b) when p == batch size (AGENTS.md #4).
    w = np.linalg.solve(gram, rhs[..., None])[..., 0]
    return z_test @ w + y_bar


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = float(((y_true - y_pred) ** 2).sum())
    total = float(((y_true - y_true.mean()) ** 2).sum())
    return float("nan") if total <= 0.0 else 1.0 - resid / total


def _finite_rows(mat: np.ndarray) -> np.ndarray:
    return np.isfinite(mat).all(axis=1)


def _matrix(df: pl.DataFrame, exprs: Sequence[pl.Expr]) -> np.ndarray:
    named = [e.alias(f"__f{i}") for i, e in enumerate(exprs)]
    return df.select(named).to_numpy().astype(np.float64)


def _fold_positions(df: pl.DataFrame, n_splits: int) -> list[tuple[np.ndarray, ...]]:
    """Purged, embargoed folds as row-position arrays into ``df``."""
    pf = PanelFrame(df, entity=ENTITY, time=TIME)
    times = np.sort(df[TIME].unique().to_numpy())
    row_time = df[TIME].to_numpy()
    cv = PurgedKFold(n_splits=n_splits, horizon=1, embargo=2, return_indices=True)
    folds = []
    for train_pos, test_pos in cv.split(pf):
        train_rows = np.flatnonzero(np.isin(row_time, times[train_pos]))
        test_rows = np.flatnonzero(np.isin(row_time, times[test_pos]))
        if train_rows.size and test_rows.size:
            folds.append((train_rows, test_rows))
    return folds


def score_step(
    df: pl.DataFrame,
    folds: list[tuple[np.ndarray, ...]],
    step: Step,
    alpha: float,
) -> tuple[float, float]:
    """Pooled out-of-sample R^2 for ``(permissive, point_in_time)``.

    Both modes are scored on exactly the same rows -- the intersection of rows
    where both constructions are defined -- so no part of the gap comes from one
    of them quietly evaluating on an easier subset.
    """
    y = df["y"].to_numpy().astype(np.float64)
    pooled: dict[str, list[np.ndarray]] = {"perm": [], "pit": [], "true": []}

    if step.kind == "expr":
        assert not callable(step.perm) and step.pit is not None
        mat_perm = _matrix(df, step.perm)  # type: ignore[arg-type]
        mat_pit = _matrix(df, step.pit)  # type: ignore[arg-type]
        ok = _finite_rows(mat_perm) & _finite_rows(mat_pit) & np.isfinite(y)
        for train_rows, test_rows in folds:
            tr = train_rows[ok[train_rows]]
            te = test_rows[ok[test_rows]]
            if tr.size <= mat_perm.shape[1] + 1 or te.size == 0:
                continue
            pooled["perm"].append(
                _ridge_predict(mat_perm[tr], y[tr], mat_perm[te], alpha)
            )
            pooled["pit"].append(_ridge_predict(mat_pit[tr], y[tr], mat_pit[te], alpha))
            pooled["true"].append(y[te])
    else:
        fit_fn = step.perm
        assert callable(fit_fn)
        for train_rows, test_rows in folds:
            train_df = df[train_rows]
            test_df = df[test_rows]
            # Permissive: every statistic fitted on the whole panel.
            p_tr = fit_fn(df, train_df)
            p_te = fit_fn(df, test_df)
            # Point-in-time: refitted on this fold's training rows only.
            q_tr = fit_fn(train_df, train_df)
            q_te = fit_fn(train_df, test_df)
            m_tr = _finite_rows(p_tr) & _finite_rows(q_tr) & np.isfinite(y[train_rows])
            m_te = _finite_rows(p_te) & _finite_rows(q_te) & np.isfinite(y[test_rows])
            if m_tr.sum() <= p_tr.shape[1] + 1 or m_te.sum() == 0:
                continue
            y_tr = y[train_rows][m_tr]
            pooled["perm"].append(_ridge_predict(p_tr[m_tr], y_tr, p_te[m_te], alpha))
            pooled["pit"].append(_ridge_predict(q_tr[m_tr], y_tr, q_te[m_te], alpha))
            pooled["true"].append(y[test_rows][m_te])

    if not pooled["true"]:
        return float("nan"), float("nan")
    truth = np.concatenate(pooled["true"])
    return (
        _r2(truth, np.concatenate(pooled["perm"])),
        _r2(truth, np.concatenate(pooled["pit"])),
    )


# --------------------------------------------------------------------------- #
# Shapley decomposition of the pipeline row
# --------------------------------------------------------------------------- #
def shapley_section(
    replicates: Sequence[tuple[int, pl.DataFrame, list[tuple[np.ndarray, ...]]]],
    alpha: float,
) -> None:
    """Attribute the pipeline row's gap to its four stages, per seed, exactly.

    The table's ``pipeline_split_last`` row is the whole preprocessing stack run
    permissively against the whole stack run point-in-time -- a single number
    for four interacting stages.  :func:`panelary.leakage.borrowed_accuracy`
    turns it into a decomposition: every one of the ``2 ** 4`` subsets is
    scored, and the exact Shapley value of each stage is reported.  It is an
    interventional (do-)Shapley value (Jung et al., ICML 2022) whose coalitions
    are *executed*: each subset is a real backtest, nothing is estimated.

    ``evaluate`` returns one pooled R^2 **per seed** -- each seed a fresh panel
    draw with its own folds, so the replicates are independent reruns.  (Per-fold
    R^2 values of one pooled score would not be: pooling is nonlinear.)  Shapley
    is linear in ``v``, so each seed's decomposition is exact and sums to that
    seed's gap; the section reports the median and range over seeds, like the
    table above.
    """
    try:
        from panelary.leakage import borrowed_accuracy
    except Exception as exc:  # noqa: BLE001 - the module is another agent's
        print(f"\n(skipping Shapley decomposition: {type(exc).__name__}: {exc})")
        return

    # Slice every seed's folds once, not once per subset: `evaluate` is called
    # 2 ** k times, and each call scores every seed.
    prepared = []
    for _, df, folds in replicates:
        y = df["y"].to_numpy().astype(np.float64)
        sliced = [
            (df[train_rows], df[test_rows], y[train_rows], y[test_rows])
            for train_rows, test_rows in folds
        ]
        prepared.append((df, sliced))

    def pooled_r2(
        df: pl.DataFrame,
        sliced: list[tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]],
        selection: frozenset[str],
    ) -> float:
        preds, truth = [], []
        for train_df, test_df, y_train, y_test in sliced:
            source = {
                name: (df if name in selection else train_df)
                for name in PIPELINE_COMPONENTS
            }
            x_train = _pipeline(train_df, source)
            x_test = _pipeline(test_df, source)
            preds.append(_ridge_predict(x_train, y_train, x_test, alpha))
            truth.append(y_test)
        return _r2(np.concatenate(truth), np.concatenate(preds))

    def evaluate(selection: frozenset[str]) -> list[float]:
        return [pooled_r2(df, sliced, selection) for df, sliced in prepared]

    try:
        report = borrowed_accuracy(PIPELINE_COMPONENTS, evaluate)
    except Exception as exc:  # noqa: BLE001 - API is not ours to pin
        print(f"\n(skipping Shapley decomposition: {type(exc).__name__}: {exc})")
        return

    seeds = [seed for seed, _, _ in replicates]
    n_rep = len(seeds)
    print("\n### Shapley decomposition of 'whole pipeline fitted before the split'")
    print(
        f"(exact, {2 ** len(PIPELINE_COMPONENTS)} subsets x {n_rep} seeds "
        f"{seeds[0]}..{seeds[-1]} = {2 ** len(PIPELINE_COMPONENTS) * n_rep} "
        "backtests, nothing sampled)"
    )
    head = f"  {'stage':<28}{'median':>9}{'min':>9}{'max':>9}{'mean':>9}"
    print(head)
    medians = report.attribution_median
    ranges = report.attribution_range
    for name, phi in sorted(report.attribution.items(), key=lambda kv: -kv[1]):
        lo, hi = ranges[name]
        print(f"  {name:<28}{medians[name]:+9.4f}{lo:+9.4f}{hi:+9.4f}{phi:+9.4f}")
    print(f"  {'-' * (len(head) - 2)}")
    t_lo, t_hi = report.total_range
    print(
        f"  {'total borrowed accuracy':<28}{report.total_median:+9.4f}"
        f"{t_lo:+9.4f}{t_hi:+9.4f}{report.total:+9.4f}"
    )
    worst = max(
        abs(
            sum(report.replicate_attribution[n][r] for n in PIPELINE_COMPONENTS)
            - report.replicate_totals[r]
        )
        for r in range(n_rep)
    )
    mean_gap = sum(report.attribution.values()) - report.total
    print(
        f"  efficiency: max over seeds |sum(phi) - total| = {worst:.1e}; "
        f"for the mean, {abs(mean_gap):.1e}"
    )
    print(
        "  median/min/max: over seeds, per stage. Medians need not sum to the "
        "median total;\n  each seed's values sum to that seed's gap, and the "
        "means sum to the mean gap."
    )


# --------------------------------------------------------------------------- #
# The verifier cross-check
# --------------------------------------------------------------------------- #
def _flags(op: object, panel: pl.DataFrame) -> bool | None:
    """True if ``assert_no_lookahead`` rejects ``op``; None if it could not run."""
    try:
        assert_no_lookahead(op, panel, entity=ENTITY, time=TIME, tol=1e-9)
    except AssertionError:
        return True
    except Exception:  # noqa: BLE001 - a verifier that cannot run is not a verdict
        return None
    return False


def verifier_column(steps: list[Step], seed: int) -> dict[str, tuple[bool | None, ...]]:
    """Run the future-perturbation verifier on both forms of every step."""
    panel = make_panel(40, 48, seed)
    out: dict[str, tuple[bool | None, ...]] = {}
    for step in steps:
        if step.kind == "expr":
            perm_exprs, pit_exprs = step.perm, step.pit
            assert not callable(perm_exprs) and pit_exprs is not None

            def _mk(exprs: Sequence[pl.Expr]) -> Callable[[pl.DataFrame], pl.DataFrame]:
                named = [e.alias(f"__chk{i}") for i, e in enumerate(exprs)]
                return lambda d: d.with_columns(named)

            out[step.key] = (
                _flags(_mk(perm_exprs), panel),  # type: ignore[arg-type]
                _flags(_mk(pit_exprs), panel),  # type: ignore[arg-type]
            )
        else:
            fn = step.perm
            assert callable(fn)

            def _mk_fit(fn: FitFn = fn) -> Callable[[pl.DataFrame], pl.DataFrame]:
                def op(d: pl.DataFrame) -> pl.DataFrame:
                    mat = fn(d, d)
                    return d.with_columns(
                        [pl.Series(f"__chk{i}", mat[:, i]) for i in range(mat.shape[1])]
                    )

                return op

            # The point-in-time mode of a fitted step has no whole-panel form --
            # it is defined relative to a fold -- so the verifier has nothing to
            # check and the cell is reported as n/a rather than as a pass.
            out[step.key] = (_flags(_mk_fit(), panel), None)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float) -> str:
    return f"{'n/a':>8}" if not math.isfinite(value) else f"{value:8.4f}"


def _verdict(flagged: bool | None) -> str:
    return {True: "FLAG", False: "clean", None: "n/a"}[flagged]


def run(
    n_entities: int,
    n_time: int,
    base_seed: int,
    n_seeds: int,
    alpha: float,
    only: str | None,
) -> None:
    steps = [s for s in _steps() if only is None or s.key == only]
    if not steps:
        raise SystemExit(f"no step named {only!r}")

    seeds = [base_seed + i for i in range(n_seeds)]
    perm: dict[str, list[float]] = {s.key: [] for s in steps}
    pit: dict[str, list[float]] = {s.key: [] for s in steps}

    replicates: list[tuple[int, pl.DataFrame, list[tuple[np.ndarray, ...]]]] = []
    for seed in seeds:
        df = make_panel(n_entities, n_time, seed)
        folds = _fold_positions(df, n_splits=5)
        for step in steps:
            p, q = score_step(df, folds, step, alpha)
            perm[step.key].append(p)
            pit[step.key].append(q)
        if only is None:
            replicates.append((seed, df, folds))

    checks = verifier_column(steps, base_seed)

    rows = n_entities * n_time
    print(
        f"\n### Panel: {n_entities:,} entities x {n_time:,} steps = {rows:,} rows "
        f"| seeds {seeds[0]}..{seeds[-1]} | PurgedKFold(5, horizon=1, embargo=2)"
    )
    head = (
        f"{'preprocessing step':<38}{'perm R2':>8}{'PIT R2':>8}{'gap':>8}"
        f"{'gap min':>8}{'gap max':>8}  {'perm?':<6}{'PIT?':<5}"
    )
    print(head)
    print("-" * len(head))

    ranked = []
    for step in steps:
        p = np.array(perm[step.key], dtype=np.float64)
        q = np.array(pit[step.key], dtype=np.float64)
        gaps = p - q
        med = float(np.nanmedian(gaps))
        ranked.append((med, step.label))
        flag_perm, flag_pit = checks[step.key]
        print(
            f"{step.label:<38}"
            f"{_fmt(float(np.nanmedian(p)))}"
            f"{_fmt(float(np.nanmedian(q)))}"
            f"{_fmt(med)}"
            f"{_fmt(float(np.nanmin(gaps)))}"
            f"{_fmt(float(np.nanmax(gaps)))}"
            f"  {_verdict(flag_perm):<6}{_verdict(flag_pit):<5}"
        )

    print("-" * len(head))
    print("perm/PIT: median pooled out-of-sample R^2 across seeds.")
    print("gap/min/max: the per-seed gap perm-PIT, summarised. It is a PAIRED")
    print("  statistic, so the median gap need not equal the difference of the two")
    print("  medians -- where they disagree, the step's effect is seed-dependent.")
    print("perm?/PIT?: does assert_no_lookahead reject that construction?")
    print("A large gap with a 'clean' verdict, or a FLAG with a zero gap, is the")
    print("interesting case: the two instruments measure different things.")
    print("The [control] row shares one construction on both sides; anything other")
    print("than an exactly zero gap there means the harness, not the step, leaks.")

    disagree = [
        (
            step.label,
            float(np.nanmedian(np.array(perm[step.key]) - np.array(pit[step.key]))),
        )
        for step in steps
        if checks[step.key][0] is not True and not step.key.startswith("control_")
    ]
    if disagree:
        print("\nPermissive forms the verifier did NOT flag:")
        for label, med in disagree:
            print(f"  {label:<38}gap {med:+.3f}")

    print("\nRanked by borrowed accuracy (median across seeds):")
    for med, label in sorted(ranked, reverse=True):
        bar = "#" * min(40, int(round(max(med, 0.0) * 200)))
        print(f"  {label:<38}{med:+.3f}  {bar}")

    if only is None:
        shapley_section(replicates, alpha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, default=20_000, help="approximate panel row count"
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument(
        "--seeds", type=int, default=5, help="how many consecutive seeds to average"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="ridge penalty on standardised features",
    )
    parser.add_argument("--only", default=None, help="score a single step by key")
    parser.add_argument(
        "--long",
        action="store_true",
        help="longer run: 150k rows and 9 seeds (overrides --rows/--seeds)",
    )
    args = parser.parse_args()

    rows, n_seeds = (150_000, 9) if args.long else (args.rows, args.seeds)
    n_time = 120
    n_entities = max(8, rows // n_time)

    print("Panelary borrowed-accuracy table")
    print(f"polars {pl.__version__} | numpy {np.__version__}")
    try:  # another agent owns panelary/leakage/_borrowed.py
        from panelary.leakage import borrowed_accuracy  # noqa: F401

        print(
            "panelary.leakage.borrowed_accuracy is importable; this table reports "
            "the same quantity at k=1 (one component, two-point gap), computed "
            "directly here so each row stays a single isolated step."
        )
    except Exception:  # noqa: BLE001 - optional at the time of writing
        print(
            "panelary.leakage.borrowed_accuracy not importable; the gap is "
            "computed directly as score(permissive) - score(point-in-time)."
        )

    run(n_entities, n_time, args.seed, n_seeds, args.alpha, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
