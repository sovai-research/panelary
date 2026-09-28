"""CrossROCKET vs handcrafted vs own-characteristic random features.

Cross-sectional forward-return prediction on ``data/sp500.parquet`` with a ridge
head under purged, embargoed K-fold CV. Three families of design matrix are
compared on *exactly the same rows and folds*:

* **H** -- handcrafted cross-sectional features: each characteristic's per-date
  centred rank (``.xs.rank(normalize="centered")``) and winsorised z-score
  (``.xs.standardize(winsor=0.01)``).
* **RFF** -- random Fourier features of each asset's *own* characteristics
  (inputs: the per-date centred ranks, scaled to unit variance), in the style of
  Kelly & Malamud's random-feature ridge. Multi-scale bandwidths {0.5, 1, 2}.
* **CR** -- :class:`panelary.embed._cross_rocket.CrossRocket` over the raw
  characteristics, with peers from trailing daily-return correlations.

plus ``H+RFF`` and ``H+CR`` (direct links), CrossROCKET ablations (no peer
family; ``activation="identity"`` instead of the default hinge),
unfitted mechanical baselines (short-term reversal, momentum), and a within-date
permuted-target null for the pipeline itself. Each design is scored twice on the
same test blocks: purged K-fold (the plan's protocol) and walk-forward (train on
earlier dates only). A fixed-lambda path is reported next to the CV-selected
lambda so a shrink-everything selection cannot hide a design's signal.

The same pipeline is then run on a **synthetic positive control** of identical
shape: 12 hidden clusters, and a planted reversal of each stock's
*idiosyncratic* 5-day return (its return net of its cluster). Only a
peer-relative feature can see that signal cleanly, so it checks that the peer
machinery works when there is something to find -- which the real-data null
alone cannot show. Two settings: cluster-factor daily vol 1.5% (equal to the
idiosyncratic vol) and 3% (cluster moves dominate).

Everything upstream of the ridge is stateless and causal (checked below with
``assert_no_lookahead`` on the characteristic builder), so the features are
built once on the whole panel and split afterwards. The ridge is the only
fitted object: features are standardised on the training fold, and ``lambda``
is chosen by an inner purged 3-fold split of the training dates (minimum MSE),
then refitted on the whole training fold -- never ridgeless.

Run (CPU-modest)::

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
    POLARS_MAX_THREADS=2 python benchmarks/cross_rocket/bench_cross_rocket.py

Writes ``benchmarks/cross_rocket/results.json`` and prints the tables used in
``docs/benchmarks/cross-rocket.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import panelary  # noqa: F401  (registers the .xs namespace)
from panelary.core.model_selection import PurgedKFold
from panelary.core.panel_frame import PanelFrame
from panelary.embed._cross_rocket import FAMILIES, CrossRocket
from panelary.testing import assert_no_lookahead

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "sp500.parquet"
OUT = Path(__file__).resolve().parent / "results.json"

E, T = "ticker", "date"
HORIZON = 5  # forward-return horizon, trading days
EMBARGO = 5
N_SPLITS = 5
INNER_SPLITS = 3
LAMBDAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
#: Fixed-lambda path reported without selection.
PATH = (1.0, 1e2, 1e4, 1e6)
CHARS = ["r1", "r5", "r21", "mom", "vol21", "maxr21", "hi60", "beta60"]


# --------------------------------------------------------------------------- #
# Data and characteristics (all trailing, per ticker)
# --------------------------------------------------------------------------- #
def characteristics(df: pl.DataFrame) -> pl.DataFrame:
    """Price-only characteristics at date t from data at dates <= t."""
    lp = pl.col("lp")
    r1 = pl.col("r1")
    out = (
        df.sort(E, T)
        .with_columns(lp=pl.col("price").log())
        .with_columns(r1=(lp - lp.shift(1)).over(E, order_by=T))
        .with_columns(mkt=r1.mean().over(T))  # same-date equal-weight market
        .with_columns(
            r5=(lp - lp.shift(5)).over(E, order_by=T),
            r21=(lp - lp.shift(21)).over(E, order_by=T),
            mom=(lp.shift(5) - lp.shift(60)).over(E, order_by=T),
            vol21=r1.rolling_std(21).over(E, order_by=T),
            maxr21=r1.rolling_max(21).over(E, order_by=T),
            hi60=(lp - lp.rolling_max(60)).over(E, order_by=T),
            _xy=(r1 * pl.col("mkt")).rolling_mean(60).over(E, order_by=T),
            _x=r1.rolling_mean(60).over(E, order_by=T),
            _y=pl.col("mkt").rolling_mean(60).over(E, order_by=T),
            _yy=(pl.col("mkt") ** 2).rolling_mean(60).over(E, order_by=T),
        )
        .with_columns(
            beta60=(pl.col("_xy") - pl.col("_x") * pl.col("_y"))
            / (pl.col("_yy") - pl.col("_y") ** 2)
        )
        .drop("_xy", "_x", "_y", "_yy", "lp", "mkt")
    )
    return out


def load_sp500() -> pl.DataFrame:
    raw = pl.read_parquet(DATA).rename({"time": T})
    # GEHC carries 0.0 prices before its Jan-2023 listing: treat as missing.
    return raw.with_columns(
        pl.when(pl.col("price") > 0).then(pl.col("price")).otherwise(None)
    )


def synthetic_prices(
    like: pl.DataFrame,
    factor_vol: float,
    seed: int = 2026,
    n_clusters: int = 12,
    kappa: float = 0.04,
) -> pl.DataFrame:
    """Same tickers and dates as ``like``; returns with a planted peer signal.

    ``r[i,t] = m[t] + f[c(i),t] + e[i,t] - kappa * sum(e[i,t-5:t])``: market,
    hidden-cluster factor and idiosyncratic shocks (daily vols 1%,
    ``factor_vol``, 1.5%),
    plus a reversal of the stock's own *idiosyncratic* 5-day shock. The cluster
    factor dominates a stock's raw 5-day return, so the signal is visible to a
    feature only once the cluster move is taken out -- i.e. relative to peers.
    """
    tickers = like[E].unique().sort().to_list()
    dates = like[T].unique().sort()
    rng = np.random.default_rng(seed)
    n, t = len(tickers), dates.len()
    cluster = rng.integers(n_clusters, size=n)
    m = 0.01 * rng.standard_normal(t)
    f = factor_vol * rng.standard_normal((t, n_clusters))
    e = 0.015 * rng.standard_normal((t, n))
    past5 = np.zeros((t, n))
    for lag in range(1, 6):
        past5[lag:] += e[:-lag]
    r = m[:, None] + f[:, cluster] + e - kappa * past5
    price = 100.0 * np.exp(np.cumsum(r, axis=0))
    return pl.DataFrame(
        {
            E: np.repeat(tickers, t),
            T: pl.concat([dates] * n),
            "price": price.T.reshape(-1),
            # hidden: used only to score an oracle, never given to a design
            "cluster": np.repeat(cluster, t),
        }
    )


def with_label(raw: pl.DataFrame) -> pl.DataFrame:
    feats = characteristics(raw)
    lp = pl.col("price").log()
    fwd = (lp.shift(-HORIZON) - lp).over(E, order_by=T).alias("fwd")
    return feats.with_columns(fwd)


def check_characteristics_causal(df: pl.DataFrame) -> None:
    """Run the characteristic builder through assert_no_lookahead (subsample)."""
    tickers = df[E].unique().sort().head(40)
    sub = df.filter(pl.col(E).is_in(tickers.implode())).select(E, T, "price")

    def op(frame: pl.DataFrame) -> pl.DataFrame:
        return characteristics(frame).select(E, T, *CHARS)

    assert_no_lookahead(op, sub, entity=E, time=T, tol=1e-9)


# --------------------------------------------------------------------------- #
# Design matrices
# --------------------------------------------------------------------------- #
def handcrafted(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs, names = [], []
    for c in CHARS:
        exprs.append(pl.col(c).xs.rank(normalize="centered").over(T).alias(f"h_rk_{c}"))
        exprs.append(pl.col(c).xs.standardize(winsor=0.01).over(T).alias(f"h_z_{c}"))
        names += [f"h_rk_{c}", f"h_z_{c}"]
    return df.select(E, T, *exprs), names


def rff(ranks: np.ndarray, n_features: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """Random Fourier features of each row's own (rank) characteristics."""
    rng = np.random.default_rng(seed)
    d = ranks.shape[1]
    half = n_features // 2
    gammas = np.array([0.5, 1.0, 2.0])[np.arange(half) % 3]
    omega = rng.standard_normal((d, half)) * gammas
    g = np.nan_to_num(ranks) * math.sqrt(12.0)  # uniform(-.5,.5) -> unit variance
    proj = g @ omega
    Z = np.concatenate([np.cos(proj), np.sin(proj)], axis=1) / math.sqrt(half)
    return Z, [f"rff_{i:03d}" for i in range(Z.shape[1])]


def cross_rocket(
    df: pl.DataFrame,
    seed: int,
    families: tuple[str, ...],
    n_ops: int,
    activation: str = "hinge",
) -> tuple[pl.DataFrame, list[str]]:
    xr = CrossRocket(
        columns=CHARS,
        n_operators=n_ops,
        families=families,
        activation=activation,
        peer_col="r1" if "peer_dev" in families else None,
        peer_windows=(20, 60),
        peer_sizes=(5, 10, 20),
        seed=seed,
        entity=E,
        time=T,
    )
    out = xr.fit_transform(df.select(E, T, *CHARS)).collect()
    return out, xr.output_names_


# --------------------------------------------------------------------------- #
# Ridge on per-date sufficient statistics
# --------------------------------------------------------------------------- #
class DateStats:
    """Per-date sums so any fold's standardised ridge is a sum and a solve."""

    def __init__(self, X: np.ndarray, y: np.ndarray, d_idx: np.ndarray, n_dates: int):
        p = X.shape[1]
        self.n = np.bincount(d_idx, minlength=n_dates).astype(np.float64)
        self.s1 = np.zeros((n_dates, p))
        self.s2 = np.zeros((n_dates, p, p))
        self.sxy = np.zeros((n_dates, p))
        self.sy = np.zeros(n_dates)
        order = np.argsort(d_idx, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(self.n.astype(np.int64))])
        for d in range(n_dates):
            rows = order[bounds[d] : bounds[d + 1]]
            Xd, yd = X[rows], y[rows]
            self.s1[d] = Xd.sum(0)
            self.s2[d] = Xd.T @ Xd
            self.sxy[d] = Xd.T @ yd
            self.sy[d] = yd.sum()

    def fit(self, dates: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        n = self.n[dates].sum()
        mu = self.s1[dates].sum(0) / n
        ybar = self.sy[dates].sum() / n
        C = self.s2[dates].sum(0) / n - np.outer(mu, mu)
        cxy = self.sxy[dates].sum(0) / n - mu * ybar
        sd = np.sqrt(np.clip(np.diag(C), 0.0, None))
        keep = sd > 1e-12
        sdk = np.where(keep, sd, 1.0)
        Cs = C / np.outer(sdk, sdk)
        cs = cxy / sdk
        Cs[~keep, :] = 0.0
        Cs[:, ~keep] = 0.0
        cs[~keep] = 0.0
        A = Cs + lam * np.eye(C.shape[0])
        beta_s = np.linalg.solve(A, cs[:, None])[:, 0]
        beta = np.where(keep, beta_s / sdk, 0.0)
        return beta, float(ybar - mu @ beta)


def inner_train(train: np.ndarray, val: np.ndarray) -> np.ndarray:
    lo, hi = val.min(), val.max()
    purged = (train >= lo - HORIZON) & (train <= hi + HORIZON + EMBARGO)
    return train[~purged]


def rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    rp = np.argsort(np.argsort(pred)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rp -= rp.mean()
    ry -= ry.mean()
    den = math.sqrt((rp @ rp) * (ry @ ry))
    return float(rp @ ry / den) if den > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int) -> float:
    x = x[~np.isnan(x)]
    n = x.size
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else float("nan")


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    d_idx: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_dates: int,
) -> dict[str, Any]:
    stats = DateStats(X, y, d_idx, n_dates)
    rows_of = [np.flatnonzero(d_idx == d) for d in range(n_dates)]
    date_ic = np.full(n_dates, np.nan)
    path_ic: dict[str, list[float]] = {f"{lam:g}": [] for lam in PATH}
    sse = sst = 0.0
    fold_ic, fold_r2, lams = [], [], []
    for train, test in folds:
        # inner purged split of the training dates to pick lambda (min MSE)
        chunks = np.array_split(np.sort(train), INNER_SPLITS)
        mse = np.zeros(len(LAMBDAS))
        for val in chunks:
            tr = inner_train(train, val)
            vrows = np.concatenate([rows_of[d] for d in val])
            for j, lam in enumerate(LAMBDAS):
                beta, b0 = stats.fit(tr, lam)
                resid = y[vrows] - (X[vrows] @ beta + b0)
                mse[j] += resid @ resid
        lam = LAMBDAS[int(np.argmin(mse))]
        lams.append(lam)
        for lam_fixed in PATH:
            b_f, c_f = stats.fit(train, lam_fixed)
            path_ic[f"{lam_fixed:g}"] += [
                rank_ic(X[rows_of[d]] @ b_f + c_f, fwd[rows_of[d]]) for d in test
            ]
        beta, b0 = stats.fit(train, lam)
        f_sse = f_sst = 0.0
        ics = []
        for d in test:
            r = rows_of[d]
            pred = X[r] @ beta + b0
            date_ic[d] = rank_ic(pred, fwd[r])
            ics.append(date_ic[d])
            pdm = pred - pred.mean()  # cross-sectional R^2: demean per date
            f_sse += float((y[r] - pdm) @ (y[r] - pdm))
            f_sst += float(y[r] @ y[r])
        sse += f_sse
        sst += f_sst
        fold_ic.append(float(np.nanmean(ics)))
        fold_r2.append(1.0 - f_sse / f_sst)
    return {
        "ic": float(np.nanmean(date_ic)),
        "ic_nw_t": newey_west_t(date_ic, HORIZON),
        "r2_pct": 100.0 * (1.0 - sse / sst),
        "fold_ic": fold_ic,
        "fold_r2_pct": [100.0 * v for v in fold_r2],
        "lambdas": lams,
        "path_ic": {k: float(np.nanmean(v)) for k, v in path_ic.items()},
    }


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def mech_result(
    signal: np.ndarray, fwd: np.ndarray, d_idx: np.ndarray, folds: list, n_dates: int
) -> dict[str, Any]:
    """Score an unfitted signal on the same test dates as the fitted designs."""
    date_ic = np.array(
        [rank_ic(signal[d_idx == d], fwd[d_idx == d]) for d in range(n_dates)]
    )
    test = np.concatenate([te for _, te in folds])
    return {
        "ic": float(np.nanmean(date_ic[test])),
        "ic_nw_t": newey_west_t(date_ic[test], HORIZON),
        "r2_pct": float("nan"),
        "fold_ic": [float(np.nanmean(date_ic[te])) for _, te in folds],
        "fold_r2_pct": [],
        "lambdas": [],
        "path_ic": {},
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ics = np.array([r["ic"] for r in runs])
    r2s = np.array([r["r2_pct"] for r in runs])
    fold_sd = [float(np.std(r["fold_ic"], ddof=1)) for r in runs]
    lam_counts = Counter(lam for r in runs for lam in r["lambdas"])
    return {
        "p": runs[0]["p"],
        "n_runs": len(runs),
        "ic_median": float(np.median(ics)),
        "ic_min": float(ics.min()),
        "ic_max": float(ics.max()),
        "ic_nw_t_median": float(np.median([r["ic_nw_t"] for r in runs])),
        "fold_ic_sd_median": float(np.median(fold_sd)),
        "r2_median": float(np.median(r2s)),
        "r2_min": float(r2s.min()),
        "r2_max": float(r2s.max()),
        "lambda_modal": lam_counts.most_common(1)[0][0] if lam_counts else None,
        "path_ic_median": {
            k: float(np.median([r["path_ic"][k] for r in runs]))
            for k in runs[0]["path_ic"]
        },
    }


def run_experiment(
    raw: pl.DataFrame, label: str, seeds: int, n_features: int
) -> dict[str, Any]:
    t_start = time.perf_counter()
    df = with_label(raw)
    hand, h_names = handcrafted(df)
    extra = []
    if "cluster" in df.columns:
        # Oracle for the synthetic control: the planted peer-relative reversal
        # scored with the *hidden* cluster labels -- the ceiling, not a design.
        r5 = pl.col("r5")
        extra = [(-(r5 - r5.mean().over(T, "cluster"))).alias("oracle")]
    base = df.select(E, T, *CHARS, "fwd", *extra).join(hand, on=[E, T])
    # The common evaluation sample: every characteristic and the label defined.
    sample = base.drop_nulls([*CHARS, *h_names, "fwd"])
    dates = sample[T].unique().sort()
    sample = sample.with_columns(
        d_idx=pl.col(T).rank("dense").cast(pl.Int64) - 1,
        # target: forward return, winsorised at 1/99% and demeaned per date
        y=pl.col("fwd").xs.winsorize(0.01).over(T),
    ).with_columns(y=pl.col("y") - pl.col("y").mean().over(T))
    sample = sample.sort(T, E)
    n_dates = dates.len()
    d_idx = sample["d_idx"].to_numpy()
    y = sample["y"].to_numpy()
    fwd = sample["fwd"].to_numpy()
    keys = sample.select(E, T)

    cv = PurgedKFold(
        n_splits=N_SPLITS, horizon=HORIZON, embargo=EMBARGO, return_indices=True
    )
    kfold = [
        (np.asarray(tr), np.asarray(te))
        for tr, te in cv.split(PanelFrame(keys, entity=E, time=T))
    ]
    # Walk-forward over the same test blocks: train only on dates before the
    # block, purged by the label horizon. Block 1 has no past, so it is skipped.
    walk = [
        (np.arange(0, int(te.min()) - HORIZON, dtype=np.int64), te)
        for _, te in kfold[1:]
    ]
    schemes = {"kfold": kfold, "walk": walk}
    meta = {
        "label": label,
        "rows": sample.height,
        "tickers": sample[E].n_unique(),
        "dates": n_dates,
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "horizon": HORIZON,
        "kfold": f"PurgedKFold({N_SPLITS}, horizon={HORIZON}, embargo={EMBARGO})",
        "walk": "same test blocks 2-5, train on earlier dates only (purged)",
        "fold_test_dates": [int(te.size) for _, te in kfold],
        "seeds": list(range(seeds)),
        "n_features": n_features,
        "lambdas": list(LAMBDAS),
        "characteristics": CHARS,
    }
    print(json.dumps(meta))

    def matrix(frame: pl.DataFrame, names: list[str]) -> np.ndarray:
        aligned = keys.join(
            frame.select(E, T, *names), on=[E, T], how="left", maintain_order="left"
        )
        return aligned.select(names).to_numpy().astype(np.float64)

    H = sample.select(h_names).to_numpy()
    results: dict[str, dict[str, list[dict[str, Any]]]] = {k: {} for k in schemes}

    def record(
        name: str,
        X: np.ndarray,
        seed: int | None,
        y_used: np.ndarray | None = None,
        fwd_used: np.ndarray | None = None,
    ) -> None:
        nan_frac = float(np.isnan(X).mean())
        X = np.nan_to_num(X)  # CR emits null where peers/cross-sections are thin
        yy = y if y_used is None else y_used
        ff = fwd if fwd_used is None else fwd_used
        for scheme, folds in schemes.items():
            res = evaluate(X, yy, ff, d_idx, folds, n_dates)
            res.update(seed=seed, p=X.shape[1], nan_frac=nan_frac)
            results[scheme].setdefault(name, []).append(res)
            print(
                f"[{label}] {scheme:<5} {name:<12} seed={seed!s:<4} "
                f"IC={res['ic']:+.4f} R2={res['r2_pct']:+.3f}% "
                f"path={res['path_ic']} nan={nan_frac:.3f}"
            )

    mechanical = [
        ("MECH reversal (-r5)", -sample["r5"].to_numpy()),
        ("MECH momentum (mom)", sample["mom"].to_numpy()),
    ]
    if "oracle" in sample.columns:
        mechanical.append(
            (
                "ORACLE peer-relative (-r5 vs hidden cluster)",
                sample["oracle"].to_numpy(),
            )
        )
    for name, sig in mechanical:
        for scheme, folds in schemes.items():
            res = mech_result(sig, fwd, d_idx, folds, n_dates)
            res.update(seed=None, p=1, nan_frac=0.0)
            results[scheme][name] = [res]

    record("H", H, None)
    ranks = sample.select(f"h_rk_{c}" for c in CHARS).to_numpy()
    for seed in range(seeds):
        Z, _ = rff(ranks, n_features, seed)
        record("RFF", Z, seed)
        record("H+RFF", np.hstack([H, Z]), seed)
        cr, names = cross_rocket(df, seed, FAMILIES, n_features)
        Xc = matrix(cr, names)
        record("CR", Xc, seed)
        record("H+CR", np.hstack([H, Xc]), seed)
        crn, names_n = cross_rocket(df, seed, FAMILIES[:3], n_features)
        record("CR no-peer", matrix(crn, names_n), seed)
        cri, names_i = cross_rocket(df, seed, FAMILIES, n_features, "identity")
        record("CR identity", matrix(cri, names_i), seed)
        if seed == 0:
            # Null: permute the target across entities within each date.
            # rank-IC is measured against fwd, so permute y and fwd together.
            rng = np.random.default_rng(12345)
            y_null, fwd_null = y.copy(), fwd.copy()
            for d in range(n_dates):
                r = np.flatnonzero(d_idx == d)
                src = rng.permutation(r)
                y_null[r], fwd_null[r] = y[src], fwd[src]
            record("NULL H", H, 0, y_null, fwd_null)
            record("NULL H+CR", np.hstack([H, Xc]), 0, y_null, fwd_null)

    summary = {
        scheme: {name: summarise(runs) for name, runs in res.items()}
        for scheme, res in results.items()
    }
    # Paired, per-seed differences in rank-IC (same seed, same folds).
    pairs = (
        ("CR - CR no-peer", "CR", "CR no-peer"),
        ("CR - RFF", "CR", "RFF"),
        ("H+CR - H", "H+CR", "H"),
        ("CR identity - CR", "CR identity", "CR"),
    )
    paired: dict[str, dict[str, Any]] = {}
    for scheme, res in results.items():
        paired[scheme] = {}
        for label_, a, b in pairs:
            ra, rb = res[a], res[b]
            if len(rb) == 1:
                rb = rb * len(ra)
            diffs = np.array([x["ic"] - y["ic"] for x, y in zip(ra, rb, strict=True)])
            paired[scheme][label_] = {
                "median": float(np.median(diffs)),
                "min": float(diffs.min()),
                "max": float(diffs.max()),
                "n_positive": int((diffs > 0).sum()),
                "n": int(diffs.size),
            }
    return {
        "meta": meta,
        "summary": summary,
        "paired": paired,
        "runs": results,
        "seconds": time.perf_counter() - t_start,
    }


def print_tables(exp: dict[str, Any]) -> None:
    label = exp["meta"]["label"]
    for scheme, table in exp["summary"].items():
        print(
            f"\n[{label} / {scheme}]\n| design | p | runs | rank-IC median "
            "[min, max] | NW t | fold-IC sd | R² % median [min, max] | modal λ "
            "| IC @ λ=1 / 1e2 / 1e4 / 1e6 |"
        )
        print("|---|---:|---:|---|---:|---:|---|---:|---|")
        for name, s in table.items():
            path = " / ".join(f"{v:+.3f}" for v in s["path_ic_median"].values())
            lam = "—" if s["lambda_modal"] is None else f"{s['lambda_modal']:g}"
            r2 = (
                "—"
                if math.isnan(s["r2_median"])
                else f"{s['r2_median']:+.3f} [{s['r2_min']:+.3f}, {s['r2_max']:+.3f}]"
            )
            print(
                f"| {name} | {s['p']} | {s['n_runs']} | {s['ic_median']:+.4f} "
                f"[{s['ic_min']:+.4f}, {s['ic_max']:+.4f}] | "
                f"{s['ic_nw_t_median']:+.2f} | {s['fold_ic_sd_median']:.4f} | "
                f"{r2} | {lam} | {path or '—'} |"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    raw = load_sp500()
    check_characteristics_causal(with_label(raw))
    print("characteristic builder: assert_no_lookahead passed (40-ticker subsample)")
    out = {"sp500": run_experiment(raw, "sp500", args.seeds, args.n_features)}
    if not args.skip_synthetic:
        for key, vol in (("synthetic_f1.5", 0.015), ("synthetic_f3", 0.03)):
            syn = synthetic_prices(raw, factor_vol=vol)
            out[key] = run_experiment(syn, key, args.seeds, args.n_features)
    out["versions"] = {
        "numpy": np.__version__,
        "polars": pl.__version__,
        "wall_seconds": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(out, indent=1))
    for key in ("sp500", "synthetic_f1.5", "synthetic_f3"):
        if key in out:
            print_tables(out[key])
            for scheme, table in out[key]["paired"].items():
                for name, d in table.items():
                    print(
                        f"[{key} / {scheme}] paired {name}: median {d['median']:+.4f} "
                        f"[{d['min']:+.4f}, {d['max']:+.4f}], "
                        f"{d['n_positive']}/{d['n']} seeds > 0"
                    )
    print(f"\nwall time {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
