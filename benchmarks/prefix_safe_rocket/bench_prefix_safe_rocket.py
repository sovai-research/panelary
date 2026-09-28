"""Prefix-safe MiniRocket: the two leak channels, priced on a real panel.

MiniRocket-style features can borrow accuracy from the future through two
separate channels:

``pooling``
    PPV pooled over each entity's *whole* series (the way the transform is
    applied to a whole series) instead of over a trailing window ending at
    ``t``. Permissive: :class:`~panelary.embed._rocket.LeakyMiniRocketReference`.
    Point-in-time: :class:`~panelary.embed._rocket.CausalMiniRocket`.
``bias_fit``
    The per-feature biases (quantiles of convolution output) fitted once on
    the whole sample instead of on the training fold. Point-in-time fits on a
    copy of the panel whose input is nulled outside the training times, so the
    fit reads no test-fold value and no response straddles the gap.

Both are scored with the same purged k-fold CV, the same rows and the same
ridge head, and :func:`panelary.leakage.borrowed_accuracy` decomposes the total
gap into the two channels by exact Shapley value (``2 ** 2 = 4`` backtests per
replicate). Each replicate is an independent draw of the feature seed and of
the entity subsample; ``evaluate`` returns one score per replicate.

**Data.** ``data/sp500.parquet``: daily prices for 503 S&P 500 constituents,
2022-06-01 .. 2023-05-31 (251 trading days, balanced). One ticker, GEHC (spun
off in January 2023), carries ``price == 0.0`` placeholders before it listed;
those rows are treated as missing. The file documents neither its price
adjustment nor its constituent date, and GEHC's presence says the ticker list
is a mid-2023 snapshot -- so treat it as a convenient real panel, not a
survivorship-clean research sample.

**Features.** ``log(price)`` through :class:`CausalMiniRocket` (504 PPV
features, padding ``"none"``). The kernels sum to zero, so the convolution of
log prices is a weighted sum of log returns -- a local trend contrast -- and
one bias is meaningful across entities.

**Target and model.** Forward ``h``-day log return, cross-sectionally demeaned
per date by default (``--raw-target`` keeps it raw). ``PurgedKFold(5,
horizon=h, embargo=h)`` on the shared date axis. Ridge on standardised
features with the penalty chosen per fold by a purged inner holdout (last 20%
of the training dates), never zero. Scores: pooled out-of-sample R^2 (the
number the decomposition is run on) and the mean daily cross-sectional rank IC
of the same predictions (decomposed separately).

Run (keep BLAS and polars to two threads each on a shared machine)::

    OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 POLARS_MAX_THREADS=2 \\
        python benchmarks/prefix_safe_rocket/bench_prefix_safe_rocket.py
    python benchmarks/prefix_safe_rocket/bench_prefix_safe_rocket.py --suite \\
        --save benchmarks/prefix_safe_rocket/results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from panelary.core.model_selection import PurgedKFold
from panelary.core.panel_frame import PanelFrame
from panelary.embed._rocket import CausalMiniRocket, LeakyMiniRocketReference
from panelary.leakage import Component, borrowed_accuracy
from panelary.leakage._borrowed import resolve_modes
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "sp500.parquet"
ENTITY, TIME, PRICE = "ticker", "time", "price"
X, Y = "logp", "y"

#: Ridge penalty grid, as a multiple of the training row count (the features
#: are standardised, so the Gram diagonal is ~n). Zero is not on it.
LAMBDAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

#: The two channels. The callables are what `resolve_modes` hands `evaluate`.
COMPONENTS = (
    Component(
        "pooling",
        permissive=lambda: "series",
        point_in_time=lambda: "trailing",
        description="whole-series PPV vs trailing-window PPV",
    ),
    Component(
        "bias_fit",
        permissive=lambda: "all",
        point_in_time=lambda: "fold",
        description="biases fitted on every row vs on the training fold",
    ),
)


@dataclass(frozen=True)
class Config:
    """One benchmark configuration."""

    name: str
    horizon: int = 5
    window: int | None = 21
    min_periods: int | None = None
    max_dilation: int = 4
    pooling: tuple[str, ...] = ("ppv",)
    n_features: int = 504
    n_entities: int = 300
    replicates: int = 7
    n_splits: int = 5
    demean: bool = True


SUITE: tuple[Config, ...] = (
    Config("main: h=5, trailing 21d PPV", replicates=9),
    Config("h=21", horizon=21, replicates=5),
    Config("expanding PPV (min 21 obs)", window=None, min_periods=21, replicates=5),
    Config("PPV + MPV", pooling=("ppv", "mpv"), replicates=5),
    Config("raw (not demeaned) target", demean=False, replicates=5),
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_panel(horizon: int, demean: bool) -> pl.DataFrame:
    """Log prices and the forward ``horizon``-day log return, sorted.

    With ``demean`` the target is the forward return minus that date's
    cross-sectional mean (over the full 503-name panel): the market-relative
    return. That removes the common move no per-stock feature of this kind can
    forecast from one year of data, which otherwise dominates pooled R^2. It is
    a label transformation, computed per date; no feature sees it.
    """
    df = pl.read_parquet(DATA).sort(ENTITY, TIME)
    # GEHC (spun off 2023-01) carries price == 0.0 placeholders for its 137
    # pre-listing days: missing, not a price.
    logp = pl.when(pl.col(PRICE) > 0).then(pl.col(PRICE).log())
    df = df.with_columns(logp.alias(X)).with_columns(
        (pl.col(X).shift(-horizon).over(ENTITY) - pl.col(X)).alias(Y)
    )
    if demean:
        df = df.with_columns((pl.col(Y) - pl.col(Y).mean().over(TIME)).alias(Y))
    return df


def subsample(df: pl.DataFrame, n_entities: int, seed: int) -> pl.DataFrame:
    """A seeded random subset of entities (all of them if ``n_entities`` covers it)."""
    names = np.sort(df[ENTITY].unique().to_numpy())
    if n_entities >= names.size:
        return df
    keep = np.random.default_rng([seed, 99]).choice(names, n_entities, replace=False)
    return df.filter(pl.col(ENTITY).is_in(keep.tolist())).sort(ENTITY, TIME)


def fold_times(df: pl.DataFrame, cfg: Config) -> tuple[np.ndarray, list[tuple]]:
    """Sorted unique dates and ``(train_dates, test_dates)`` per purged fold."""
    times = np.sort(df[TIME].unique().to_numpy())
    cv = PurgedKFold(
        cfg.n_splits, horizon=cfg.horizon, embargo=cfg.horizon, return_indices=True
    )
    pf = PanelFrame(df, entity=ENTITY, time=TIME)
    return times, [(times[tr], times[te]) for tr, te in cv.split(pf)]


# --------------------------------------------------------------------------- #
# Ridge head (numpy; penalty chosen inside the training fold)
# --------------------------------------------------------------------------- #
def _standardise(x_fit: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mu = x_fit.mean(axis=0)
    sd = x_fit.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return tuple((a - mu) / sd for a in (x_fit, *others))


def _ridge_path(
    z: np.ndarray, y: np.ndarray, lambdas: tuple[float, ...]
) -> list[np.ndarray]:
    """Ridge weights for each ``lambda * n``, from one eigendecomposition."""
    gram = z.T @ z
    evals, evecs = np.linalg.eigh(gram)
    proj = evecs.T @ (z.T @ y)
    n = z.shape[0]
    return [evecs @ (proj / (evals + lam * n)) for lam in lambdas]


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / total


def ridge_predict(
    x: np.ndarray,
    y: np.ndarray,
    row_time: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, float]:
    """Fit on ``train`` rows (penalty by purged inner holdout), predict ``test``."""
    t_train = np.sort(np.unique(row_time[train]))
    split = t_train[int(0.8 * t_train.size)]
    purge_to = t_train[max(0, int(0.8 * t_train.size) - horizon)]
    inner_fit = train[row_time[train] < purge_to]
    inner_val = train[row_time[train] >= split]
    z_fit, z_val = _standardise(x[inner_fit], x[inner_val])
    y_bar = y[inner_fit].mean()
    path = _ridge_path(z_fit, y[inner_fit] - y_bar, LAMBDAS)
    scores = [_r2(y[inner_val], z_val @ w + y_bar) for w in path]
    lam = LAMBDAS[int(np.argmax(scores))]

    z_tr, z_te = _standardise(x[train], x[test])
    y_bar = y[train].mean()
    k = z_tr.shape[1]
    gram = z_tr.T @ z_tr + lam * z_tr.shape[0] * np.eye(k)
    rhs = z_tr.T @ (y[train] - y_bar)
    w = np.linalg.solve(gram, rhs[..., None])[..., 0]  # AGENTS.md invariant 4
    return z_te @ w + y_bar, lam


def mean_rank_ic(y: np.ndarray, pred: np.ndarray, row_time: np.ndarray) -> float:
    """Mean over dates of the cross-sectional Spearman correlation."""
    ics = []
    order = np.argsort(row_time, kind="stable")
    t_sorted = row_time[order]
    cuts = np.flatnonzero(t_sorted[1:] != t_sorted[:-1]) + 1
    for idx in np.split(order, cuts):
        if idx.size < 10:
            continue
        ry = np.argsort(np.argsort(y[idx])).astype(np.float64)
        rp = np.argsort(np.argsort(pred[idx])).astype(np.float64)
        ics.append(float(np.corrcoef(ry, rp)[0, 1]))
    return float(np.mean(ics))


# --------------------------------------------------------------------------- #
# One replicate: every coalition, every fold
# --------------------------------------------------------------------------- #
def _feature_matrix(est: CausalMiniRocket, panel: pl.DataFrame) -> np.ndarray:
    """Features as a float64 ``(rows, features)`` matrix, in ``panel``'s row order."""
    out = est.transform(panel).collect()
    return np.hstack(
        [out[f"{c}{est.suffix}"].to_numpy().astype(np.float64) for c in est.columns]
    )


def run_replicate(df_all: pl.DataFrame, cfg: Config, seed: int) -> dict[str, Any]:
    """Score all four coalitions (plus a diagnostic) for one seed."""
    df = subsample(df_all, cfg.n_entities, seed)
    _, folds = fold_times(df, cfg)
    y = df[Y].to_numpy()
    row_time = df[TIME].to_numpy()

    def make() -> CausalMiniRocket:
        return CausalMiniRocket(
            X,
            n_features=cfg.n_features,
            window=cfg.window,
            min_periods=cfg.min_periods,
            max_dilation=cfg.max_dilation,
            pooling=cfg.pooling,
            seed=seed,
            entity=ENTITY,
            time=TIME,
        )

    est_all = make().fit(df)
    feats_all = {
        "trailing": _feature_matrix(est_all, df),
        "series": _feature_matrix(LeakyMiniRocketReference.from_fitted(est_all), df),
    }
    # Rows every coalition can score: the trailing features' warm-up does not
    # depend on the biases, so one mask serves all four.
    usable = np.isfinite(feats_all["trailing"]).all(axis=1) & np.isfinite(y)

    coalitions = [frozenset(), frozenset({"pooling"}), frozenset({"bias_fit"})]
    coalitions.append(frozenset({"pooling", "bias_fit"}))
    preds: dict[frozenset[str] | str, list[np.ndarray]] = {c: [] for c in coalitions}
    preds["diag_series_train_only"] = []
    truth, times, lams = [], [], []
    for train_t, test_t in folds:
        train = np.flatnonzero(np.isin(row_time, train_t) & usable)
        test = np.flatnonzero(np.isin(row_time, test_t) & usable)
        in_train = pl.Series("in_train", np.isin(row_time, train_t))
        masked = df.with_columns(
            pl.when(in_train).then(pl.col(X)).otherwise(None).alias(X)
        )
        est_fold = make().fit(masked)
        feats_fold = {
            "trailing": _feature_matrix(est_fold, df),
            "series": _feature_matrix(
                LeakyMiniRocketReference.from_fitted(est_fold), df
            ),
        }
        for sel in coalitions:
            modes = resolve_modes(COMPONENTS, sel)
            source = feats_all if modes["bias_fit"]() == "all" else feats_fold
            x = source[modes["pooling"]()]
            p, lam = ridge_predict(x, y, row_time, train, test, cfg.horizon)
            preds[sel].append(p)
            lams.append(lam)
        # Diagnostic: a static per-entity PPV built from training rows only.
        # Same functional form as the leaky arm, no test-fold information.
        diag = _feature_matrix(LeakyMiniRocketReference.from_fitted(est_fold), masked)
        # An entity with too few training responses (GEHC, listed 2023-01) has
        # no train-only PPV: fill with the training rows' column means.
        fill = np.nanmean(diag[train], axis=0)
        diag = np.where(np.isfinite(diag), diag, fill[None, :])
        p, _ = ridge_predict(diag, y, row_time, train, test, cfg.horizon)
        preds["diag_series_train_only"].append(p)
        truth.append(y[test])
        times.append(row_time[test])

    y_te, t_te = np.concatenate(truth), np.concatenate(times)
    r2 = {k: _r2(y_te, np.concatenate(v)) for k, v in preds.items()}
    ic = {k: mean_rank_ic(y_te, np.concatenate(v), t_te) for k, v in preds.items()}
    return {
        "seed": seed,
        "n_entities": int(df[ENTITY].n_unique()),
        "n_scored_rows": int(y_te.size),
        "r2": r2,
        "ic": ic,
        "lambdas": lams,
    }


# --------------------------------------------------------------------------- #
# Decomposition and reporting
# --------------------------------------------------------------------------- #
def decompose(reps: list[dict[str, Any]], metric: str) -> Any:
    """Exact Shapley over the two channels, one score per replicate."""

    def evaluate(selection: frozenset[str]) -> np.ndarray:
        return np.array([rep[metric][selection] for rep in reps], dtype=np.float64)

    try:
        return borrowed_accuracy(COMPONENTS, evaluate)
    except (TypeError, ValueError):  # a scalar-only API: fall back per replicate
        return [
            borrowed_accuracy(COMPONENTS, lambda s, rep=rep: float(rep[metric][s]))
            for rep in reps
        ]


def _summ(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summarise(cfg: Config, reps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"config": asdict(cfg), "replicates": []}
    for rep in reps:
        out["replicates"].append(
            {
                "seed": rep["seed"],
                "n_entities": rep["n_entities"],
                "n_scored_rows": rep["n_scored_rows"],
                **{
                    f"{m}[{'+'.join(sorted(k)) if isinstance(k, frozenset) else k}]": v
                    for m in ("r2", "ic")
                    for k, v in rep[m].items()
                },
            }
        )
    for metric in ("r2", "ic"):
        report = decompose(reps, metric)
        if isinstance(report, list):
            totals = [r.total for r in report]
            phis = {c.name: [r.attribution[c.name] for r in report] for c in COMPONENTS}
        else:
            totals = list(report.replicate_totals) or [report.total]
            phis = {
                c.name: list(report.replicate_attribution.get(c.name, ()))
                or [report.attribution[c.name]]
                for c in COMPONENTS
            }
        effic = max(
            abs(sum(phis[c.name][i] for c in COMPONENTS) - totals[i])
            for i in range(len(totals))
        )
        coal = {
            "+".join(sorted(s)) or "point-in-time": _summ([r[metric][s] for r in reps])
            for s in reps[0][metric]
            if isinstance(s, frozenset)
        }
        coal["diag: series PPV, train rows only"] = _summ(
            [r[metric]["diag_series_train_only"] for r in reps]
        )
        out[metric] = {
            "coalitions": coal,
            "total": _summ(totals),
            "shapley": {k: _summ(v) for k, v in phis.items()},
            "shapley_mean_sums_to_total_mean": float(
                sum(np.mean(v) for v in phis.values()) - np.mean(totals)
            ),
            "max_efficiency_error": float(effic),
        }
    lams = [lam for rep in reps for lam in rep["lambdas"]]
    out["lambda_counts"] = {str(v): lams.count(v) for v in LAMBDAS}
    return out


def print_summary(summary: dict[str, Any]) -> None:
    cfg = summary["config"]
    reps = summary["replicates"]
    print(f"\n## {cfg['name']}")
    print(
        f"h={cfg['horizon']}  window={cfg['window']}  min_periods={cfg['min_periods']}"
        f"  max_dilation={cfg['max_dilation']}  pooling={cfg['pooling']}"
        f"  n_features={cfg['n_features']}  demeaned target={cfg['demean']}"
        f"  entities/replicate={cfg['n_entities']}"
        f"  replicates={len(reps)} (seeds {reps[0]['seed']}..{reps[-1]['seed']})"
        f"  scored rows/replicate={reps[0]['n_scored_rows']}"
    )
    for metric, label in (("r2", "pooled OOS R^2"), ("ic", "mean daily rank IC")):
        block = summary[metric]
        print(f"\n### {label}: median [min, max] over replicates")
        print("| arm | median | min | max |")
        print("|---|---:|---:|---:|")
        for name, s in block["coalitions"].items():
            print(
                f"| {name} | {s['median']:+.4f} | {s['min']:+.4f} | {s['max']:+.4f} |"
            )
        print("\n| Shapley | median | min | max | mean |")
        print("|---|---:|---:|---:|---:|")
        for name, s in block["shapley"].items():
            print(
                f"| {name} | {s['median']:+.4f} | {s['min']:+.4f} | {s['max']:+.4f}"
                f" | {s['mean']:+.4f} |"
            )
        t = block["total"]
        print(
            f"| **total** | {t['median']:+.4f} | {t['min']:+.4f} | {t['max']:+.4f}"
            f" | {t['mean']:+.4f} |"
        )
        print(
            f"max per-replicate |sum(phi) - total| = {block['max_efficiency_error']:.1e}"
        )
    print(
        f"\nridge lambda chosen (count over folds x arms): {summary['lambda_counts']}"
    )


# --------------------------------------------------------------------------- #
# The yes/no instrument, on the same constructions
# --------------------------------------------------------------------------- #
def verifier_crosscheck(
    df_all: pl.DataFrame, cfg: Config
) -> dict[str, dict[str, bool]]:
    """Does ``assert_no_lookahead`` / ``assert_prefix_invariant`` flag each arm?"""
    df = subsample(df_all, 12, 0).select(ENTITY, TIME, X)
    times = df[TIME].unique().sort().to_list()
    cut = times[len(times) // 2]

    def make() -> CausalMiniRocket:
        return CausalMiniRocket(
            X,
            n_features=cfg.n_features,
            window=cfg.window,
            min_periods=cfg.min_periods,
            max_dilation=cfg.max_dilation,
            pooling=cfg.pooling,
            entity=ENTITY,
            time=TIME,
        )

    fixed = make().fit(df.filter(pl.col(TIME) <= cut))
    arms: dict[str, Callable[[pl.DataFrame], Any]] = {
        "point-in-time (trailing, fixed biases)": fixed.transform,
        "pooling permissive (whole series)": LeakyMiniRocketReference.from_fitted(
            fixed
        ).transform,
        "bias_fit permissive (fit on the frame)": lambda d: make().fit(d).transform(d),
    }
    result: dict[str, dict[str, bool]] = {}
    for name, op in arms.items():
        flags = {}
        for label, check in (
            ("lookahead", assert_no_lookahead),
            ("prefix", assert_prefix_invariant),
        ):
            try:
                check(op, df, entity=ENTITY, time=TIME, cut=cut)
                flags[label] = False
            except AssertionError:
                flags[label] = True
        result[name] = flags
    return result


# --------------------------------------------------------------------------- #
def run(cfg: Config, df_cache: dict[tuple[int, bool], pl.DataFrame]) -> dict[str, Any]:
    key = (cfg.horizon, cfg.demean)
    if key not in df_cache:
        df_cache[key] = load_panel(cfg.horizon, cfg.demean)
    df_all = df_cache[key]
    t0 = time.perf_counter()
    reps = []
    for seed in range(cfg.replicates):
        reps.append(run_replicate(df_all, cfg, seed))
        print(
            f"  [{cfg.name}] replicate {seed} done ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    summary = summarise(cfg, reps)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["verifier"] = verifier_crosscheck(df_all, cfg)
    print_summary(summary)
    print("\nverifier (True = flags a leak):")
    for arm, flags in summary["verifier"].items():
        print(
            f"  {arm:<42} lookahead={flags['lookahead']!s:<5} prefix={flags['prefix']}"
        )
    print(f"({summary['seconds']} s)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true", help="run every SUITE config")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=21, help="0 = expanding")
    parser.add_argument("--min-periods", type=int, default=None)
    parser.add_argument("--max-dilation", type=int, default=4)
    parser.add_argument("--mpv", action="store_true", help="add MPV pooling")
    parser.add_argument("--n-features", type=int, default=504)
    parser.add_argument("--entities", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--raw-target", action="store_true", help="do not demean y")
    parser.add_argument("--save", type=Path, default=None, help="write JSON results")
    parser.add_argument(
        "--only",
        default=None,
        help="run the SUITE configs whose name starts with this; with --save on "
        "an existing file, replace just those entries",
    )
    args = parser.parse_args()

    if args.suite or args.only:
        configs = [
            c
            if args.replicates is None
            else Config(**{**asdict(c), "replicates": args.replicates})
            for c in SUITE
            if args.only is None or c.name.startswith(args.only)
        ]
    else:
        configs = [
            Config(
                "custom",
                horizon=args.horizon,
                window=args.window or None,
                min_periods=args.min_periods,
                max_dilation=args.max_dilation,
                pooling=("ppv", "mpv") if args.mpv else ("ppv",),
                n_features=args.n_features,
                n_entities=args.entities,
                replicates=args.replicates or 7,
                demean=not args.raw_target,
            )
        ]
    print("# Prefix-safe MiniRocket: borrowed accuracy by leak channel")
    print(
        f"polars {pl.__version__} | numpy {np.__version__} | Python "
        f"{platform.python_version()} | {platform.machine()}"
    )
    cache: dict[tuple[int, bool], pl.DataFrame] = {}
    results = [run(cfg, cache) for cfg in configs]
    if args.save is not None:
        if args.only is not None and args.save.exists():
            # Replace the re-run configs in place; keep every other entry.
            fresh = {r["config"]["name"]: r for r in results}
            kept = json.loads(args.save.read_text())["results"]
            results = [fresh.pop(r["config"]["name"], r) for r in kept]
            results.extend(fresh.values())
        args.save.write_text(
            json.dumps(
                {
                    "environment": {
                        "polars": pl.__version__,
                        "numpy": np.__version__,
                        "python": platform.python_version(),
                        "machine": platform.machine(),
                    },
                    "data": str(DATA.relative_to(ROOT)),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
