"""``panelary.covariance.avg_correlation`` and ``common_idio_vol``.

Plan ``covariance-and-market-state`` M3 (the co-movement subset), sections 5.14,
6 and 7:

* **exact identities** -- the equal-weight average correlation equals the mean
  off-diagonal of ``np.corrcoef`` on a complete window, and the zero-after-demean
  estimator's own matrix when the window has holes;
* **Pollet-Wilson** -- the O(N) portfolio-variance path equals the kernel
  (``sum cov / sum sd sd``) on a constant universe, and dates where the
  universe changed are flagged and recomputed on the kernel;
* **common idiosyncratic volatility** against ``residualise`` called directly
  plus a numpy rolling sd;
* **group variants** against a manual loop, and trap **T12**: membership is the
  date-``t`` label, so relabelling the future cannot move the past, while the
  "latest classification" snapshot join is caught as a look-ahead;
* prefix invariance and no look-ahead **bitwise** (``tol=0.0``) on a panel whose
  names enter and leave, at every cut.

Traps T14 (a silent ``pinv`` of a singular sample covariance) and T15
(hyperparameters picked by a full-sample comparison) do not arise here: neither
function inverts a covariance matrix -- the average correlation is a quadratic
form in the window, and the residualiser's own ``pinv_sym`` of a ``k x k``
loading Gram is unchanged, reused code -- and neither selects anything from
data (the factor count, windows and refit cadence are arguments).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core.panel_frame import PanelFrame
from panelary.detect._panel import residualise
from panelary.registry import registry
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

cov = pn.covariance
KEYS = {"entity": "entity", "time": "time"}


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _dense_returns(n_ent: int, n_time: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A market factor, three sector factors, heterogeneous idiosyncratic scale."""
    rng = np.random.default_rng(seed)
    sector = rng.integers(0, 3, n_ent)
    market = rng.standard_normal(n_time)
    sectors = rng.standard_normal((n_time, 3))
    beta = 0.5 + rng.random(n_ent)
    scale = 0.5 + 2.0 * rng.random(n_ent)
    noise = rng.standard_t(5, (n_time, n_ent)) * scale
    r = 0.01 * (beta * market[:, None] + 0.7 * sectors[:, sector] + noise)
    return r, sector


def _long(
    r: np.ndarray,
    sector: np.ndarray | None = None,
    *,
    keep: np.ndarray | None = None,
    relabel_after: int | None = None,
) -> pl.DataFrame:
    """Long frame of the cells ``keep`` (default: all), NaN cells as null."""
    n_time, n_ent = r.shape
    keep = np.ones_like(r, dtype=bool) if keep is None else keep
    t_idx, e_idx = np.nonzero(keep)
    data: dict[str, object] = {
        "entity": [f"E{i:03d}" for i in e_idx],
        "time": t_idx.astype(np.int64),
        "ret": r[t_idx, e_idx],
    }
    if sector is not None:
        lab = sector[e_idx].copy()
        if relabel_after is not None:
            moved = (t_idx >= relabel_after) & (e_idx % 4 == 0)
            lab[moved] = (lab[moved] + 1) % 3
        data["sector"] = lab.astype(np.int64)
    return pl.DataFrame(data).with_columns(pl.col("ret").fill_nan(None))


def _churn_panel(seed: int = 0) -> pl.DataFrame:
    """Entry, exit, scattered missing rows and a mid-sample sector relabel."""
    r, sector = _dense_returns(36, 150, seed)
    rng = np.random.default_rng(seed + 1)
    keep = np.ones_like(r, dtype=bool)
    for i in range(r.shape[1]):
        if i % 3 == 0:
            keep[: rng.integers(10, 70), i] = False  # lists late
        if i % 5 == 1:
            keep[rng.integers(80, 140) :, i] = False  # delists
    keep &= rng.random(r.shape) > 0.02
    return _long(r, sector, keep=keep, relabel_after=90)


# --------------------------------------------------------------------------- #
# numpy references
# --------------------------------------------------------------------------- #
def _zad_window(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-after-demean window and per-name sd (``ddof`` = own count - 1)."""
    obs = np.isfinite(block)
    n = obs.sum(axis=0)
    mean = np.where(obs, block, 0.0).sum(axis=0) / n
    dev = np.where(obs, block - mean, 0.0)
    sd = np.sqrt((dev**2).sum(axis=0) / (n - 1))
    return dev, sd


def _ref_equal(block: np.ndarray) -> float:
    dev, sd = _zad_window(block)
    c = (dev / sd).T @ (dev / sd) / (block.shape[0] - 1)
    n = c.shape[0]
    return float((c.sum() - np.trace(c)) / (n * (n - 1)))


def _ref_pollet_wilson(block: np.ndarray) -> float:
    dev, sd = _zad_window(block)
    c = dev.T @ dev / (block.shape[0] - 1)
    off = c.sum() - np.trace(c)
    return float(off / (sd.sum() ** 2 - (sd**2).sum()))


def _universe(r: np.ndarray, t: int, window: int, need: int) -> np.ndarray:
    lo = max(0, t - window + 1)
    count = np.isfinite(r[lo : t + 1]).sum(axis=0)
    return np.flatnonzero(np.isfinite(r[t]) & (count >= need))


def _col(df: pl.DataFrame, name: str) -> np.ndarray:
    return df[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()


# --------------------------------------------------------------------------- #
# 1. Average correlation: exact identities
# --------------------------------------------------------------------------- #
def test_equal_weight_is_mean_offdiagonal_of_corrcoef() -> None:
    r, _ = _dense_returns(25, 90, seed=1)
    window = 30
    out = cov.avg_correlation(_long(r), returns="ret", window=window, **KEYS)
    got = _col(out, "avg_corr")
    # Dates before the panel start count as unobserved, so the first value is
    # where the partial window first reaches ceil(0.95 * 30) = 29 rows.
    first = math.ceil(0.95 * window) - 1
    assert np.isnan(got[:first]).all()
    for t in range(first, r.shape[0]):
        c = np.corrcoef(r[max(0, t - window + 1) : t + 1].T)
        want = (c.sum() - np.trace(c)) / (c.shape[0] * (c.shape[0] - 1))
        assert math.isclose(got[t], want, rel_tol=1e-12, abs_tol=1e-15), t
    assert (out["n_entities"][first:] == 25).all()


def test_window_with_holes_matches_the_zero_after_demean_matrix() -> None:
    r, _ = _dense_returns(20, 80, seed=2)
    rng = np.random.default_rng(9)
    r[rng.random(r.shape) < 0.04] = np.nan
    window, coverage = 25, 0.9
    need = math.ceil(coverage * window)
    out = cov.avg_correlation(
        _long(r), returns="ret", window=window, min_coverage=coverage, **KEYS
    )
    got = _col(out, "avg_corr")
    checked = 0
    for t in range(r.shape[0]):
        members = _universe(r, t, window, need)
        if members.size < 2:
            assert np.isnan(got[t])
            continue
        lo = max(0, t - window + 1)
        want = _ref_equal(r[lo : t + 1, members])
        assert math.isclose(got[t], want, rel_tol=1e-12, abs_tol=1e-15), t
        assert out["n_entities"][t] == members.size
        checked += 1
    assert checked > 40


def test_universe_is_as_of_the_date() -> None:
    """A late lister enters only once covered; a delisted name leaves at once;
    a name with zero variance in the window has no correlation."""
    r, _ = _dense_returns(12, 60, seed=3)
    r[:30, 0] = np.nan  # lists at t=30
    r[40:, 1] = np.nan  # last observed at t=39
    r[:, 2] = 0.01  # constant
    window = 10
    out = cov.avg_correlation(_long(r), returns="ret", window=window, **KEYS)
    n = out["n_entities"].to_numpy()
    assert n[35] == 10  # 12 minus the constant and the not-yet-covered lister
    assert n[39] == 11  # the lister is covered from t = 30 + window - 1
    assert n[40] == 10  # the delisted name is gone the next date


# --------------------------------------------------------------------------- #
# 2. Pollet-Wilson: fast path, flags, kernel fallback
# --------------------------------------------------------------------------- #
def test_pollet_wilson_fast_path_equals_kernel_on_constant_universe() -> None:
    r, _ = _dense_returns(30, 100, seed=4)
    window = 40
    out = cov.avg_correlation(
        _long(r), returns="ret", window=window, kind="pollet_wilson", **KEYS
    )
    stable = out["universe_stable"].to_numpy()
    assert not stable[: window - 1].any()
    assert stable[window - 1 :].all()
    got = _col(out, "avg_corr")
    equal = _col(
        cov.avg_correlation(_long(r), returns="ret", window=window, **KEYS), "avg_corr"
    )
    for t in range(window - 1, r.shape[0]):
        want = _ref_pollet_wilson(r[t - window + 1 : t + 1])
        assert math.isclose(got[t], want, rel_tol=1e-10), t
    # sigma-weighting is a different statistic from the plain mean
    assert np.nanmax(np.abs(got - equal)) > 1e-3


def test_pollet_wilson_flags_a_changing_universe_and_uses_the_kernel() -> None:
    r, _ = _dense_returns(20, 90, seed=5)
    r[:50, 3] = np.nan  # lists at t=50
    r[70:, 4] = np.nan  # delists after t=69
    r[60, 5] = np.nan  # one missing day
    window, coverage = 20, 0.9
    out = cov.avg_correlation(
        _long(r),
        returns="ret",
        window=window,
        kind="pollet_wilson",
        min_coverage=coverage,
        **KEYS,
    )
    stable = out["universe_stable"].to_numpy()
    got = _col(out, "avg_corr")
    need = math.ceil(coverage * window)
    changes = {50, 60, 61, 70}  # observed set differs from the previous date
    for t in range(window - 1, r.shape[0]):
        # unstable iff the window holds both sides of a change
        crosses = any(t - window + 1 < c <= t for c in changes)
        assert stable[t] == (not crosses), t
        members = _universe(r, t, window, need)
        lo = t - window + 1
        want = _ref_pollet_wilson(r[lo : t + 1, members])
        assert math.isclose(got[t], want, rel_tol=1e-10), t
    assert stable.sum() > 10 and (~stable[window - 1 :]).sum() > 10


# --------------------------------------------------------------------------- #
# 3. Group variants and trap T12
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["equal", "pollet_wilson"])
def test_group_variant_is_the_kernel_on_date_t_members(kind: str) -> None:
    r, sector = _dense_returns(45, 70, seed=6)
    df = _long(r, sector, relabel_after=40)
    window = 20
    out = cov.avg_correlation(
        df,
        returns="ret",
        window=window,
        kind=kind,
        group="sector",
        min_entities=4,
        **KEYS,
    )
    labels = df.pivot(on="entity", index="time", values="sector").sort("time")
    lab = labels.drop("time").to_numpy()
    ref = _ref_equal if kind == "equal" else _ref_pollet_wilson
    checked = 0
    for row in out.iter_rows(named=True):
        t, g = row["time"], row["sector"]
        members = np.flatnonzero(lab[t] == g)
        assert row["n_entities"] == (
            members.size if t >= window - 1 else row["n_entities"]
        )
        if t < window - 1:
            continue
        want = ref(r[t - window + 1 : t + 1, members])
        assert math.isclose(row["avg_corr"], want, rel_tol=1e-12, abs_tol=1e-15)
        checked += 1
    assert checked > 100


def test_t12_future_relabels_cannot_move_the_past() -> None:
    """The sector column is numeric, so the look-ahead verifier perturbs future
    labels too -- every future row lands in a group of its own."""
    pf = PanelFrame(_churn_panel(), **KEYS)

    def op(f):
        return cov.avg_correlation(
            f,
            returns="ret",
            window=20,
            group="sector",
            min_entities=3,
            broadcast=True,
            **KEYS,
        )

    assert op(pf.collect())["avg_corr"].drop_nulls().len() > 100
    assert_no_lookahead(op, pf, tol=0.0)
    assert_prefix_invariant(op, pf, tol=0.0)


def test_t12_latest_classification_snapshot_is_caught() -> None:
    """The trap itself: joining each name's *latest* label (a GICS snapshot)
    onto its history. The verifier must flag it, or the test above proves
    nothing."""
    pf = PanelFrame(_churn_panel(), **KEYS)

    def leaky(f):
        f = f.with_columns(pl.col("sector").last().over("entity"))
        return cov.avg_correlation(
            f,
            returns="ret",
            window=20,
            group="sector",
            min_entities=3,
            broadcast=True,
            **KEYS,
        )

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky, pf)


# --------------------------------------------------------------------------- #
# 4. Common idiosyncratic volatility
# --------------------------------------------------------------------------- #
def _rolling_sd(x: np.ndarray, window: int, need: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        block = x[max(0, t - window + 1) : t + 1]
        for i in range(x.shape[1]):
            v = block[:, i][np.isfinite(block[:, i])]
            if v.size >= need:
                out[t, i] = np.std(v, ddof=1)
    return out


def test_common_idio_vol_matches_residualise_plus_rolling_sd() -> None:
    r, sector = _dense_returns(15, 120, seed=7)
    window, fit_window, refit = 20, 40, 9
    out = cov.common_idio_vol(
        _long(r, sector),
        returns="ret",
        window=window,
        fit_window=fit_window,
        refit_every=refit,
        n_factors=2,
        **KEYS,
    )
    resid = residualise(
        r.T,
        n_factors=2,
        min_periods=fit_window,
        refit_every=refit,
        window=fit_window,
        min_obs=math.ceil(0.95 * fit_window),
    ).T
    idio = _rolling_sd(resid, window, math.ceil(0.95 * window))
    have = np.isfinite(idio).sum(1)
    want = np.full(idio.shape[0], np.nan)
    want[have >= 2] = np.nanmean(idio[have >= 2], axis=1)
    got = _col(out, "common_idio_vol")
    first = fit_window + math.ceil(0.95 * window) - 1
    assert np.isnan(got[:first]).all() and np.isfinite(got[first:]).all()
    np.testing.assert_allclose(got, want, rtol=1e-10, equal_nan=True)

    # the group variant is the same idio_vol averaged over date-t members
    grouped = cov.common_idio_vol(
        _long(r, sector),
        returns="ret",
        window=window,
        fit_window=fit_window,
        refit_every=refit,
        n_factors=2,
        group="sector",
        min_entities=2,
        **KEYS,
    )
    for row in grouped.iter_rows(named=True):
        members = np.flatnonzero(sector == row["sector"])
        vals = idio[row["time"], members]
        vals = vals[np.isfinite(vals)]
        assert row["n_entities"] == vals.size
        if vals.size >= 2:
            assert math.isclose(row["common_idio_vol"], vals.mean(), rel_tol=1e-10)
        else:
            assert row["common_idio_vol"] is None

    # broadcast carries each row's own idio_vol
    rows = cov.common_idio_vol(
        _long(r, sector),
        returns="ret",
        window=window,
        fit_window=fit_window,
        refit_every=refit,
        n_factors=2,
        broadcast=True,
        **KEYS,
    ).sort(["entity", "time"])
    np.testing.assert_allclose(
        _col(rows, "idio_vol"), idio.T.ravel(), rtol=1e-10, equal_nan=True
    )


def test_residuals_come_only_from_names_with_a_fitted_loading() -> None:
    """A name listing mid-segment has no residual until a refit gives it a
    loading; ``residualise``'s own-demeaned-return fallback is not used."""
    r, _ = _dense_returns(10, 100, seed=8)
    r[:55, 0] = np.nan  # lists at t=55
    out = cov.common_idio_vol(
        _long(r),
        returns="ret",
        window=5,
        fit_window=30,
        refit_every=10,
        min_coverage=1.0,
        broadcast=True,
        **KEYS,
    )
    e0 = out.filter(pl.col("entity") == "E000").sort("time")
    # The first refit whose 30-date block E000 fully covers is t = 90 (the grid
    # is 30, 40, ...); its idio_vol then needs 5 residuals, so it starts at 94.
    assert e0.filter(pl.col("time") < 94)["idio_vol"].null_count() == 94
    assert e0.filter(pl.col("time") >= 94)["idio_vol"].null_count() == 0
    # the other names are unaffected by E000's arrival in their loadings' past
    others = out.filter((pl.col("entity") != "E000") & (pl.col("time") >= 34))
    assert others["idio_vol"].null_count() == 0


# --------------------------------------------------------------------------- #
# 5. Leak safety, bitwise, on a churning panel
# --------------------------------------------------------------------------- #
_OPS = {
    "avg_corr_equal": lambda f: cov.avg_correlation(
        f, returns="ret", window=20, broadcast=True, **KEYS
    ),
    "avg_corr_pw": lambda f: cov.avg_correlation(
        f,
        returns="ret",
        window=20,
        kind="pollet_wilson",
        min_coverage=0.9,
        broadcast=True,
        **KEYS,
    ),
    "avg_corr_group": lambda f: cov.avg_correlation(
        f,
        returns="ret",
        window=20,
        group="sector",
        min_entities=3,
        broadcast=True,
        **KEYS,
    ),
    "civ": lambda f: cov.common_idio_vol(
        f,
        returns="ret",
        window=10,
        fit_window=30,
        refit_every=7,
        broadcast=True,
        **KEYS,
    ),
    "civ_group": lambda f: cov.common_idio_vol(
        f,
        returns="ret",
        window=10,
        fit_window=30,
        refit_every=7,
        group="sector",
        min_entities=2,
        broadcast=True,
        **KEYS,
    ),
}


@pytest.mark.parametrize("name", sorted(_OPS))
def test_prefix_invariant_and_causal_bitwise(name: str) -> None:
    df = _churn_panel()
    pf = PanelFrame(df, **KEYS)
    op = _OPS[name]
    feature = "avg_corr" if name.startswith("avg") else "common_idio_vol"
    assert op(df)[feature].drop_nulls().len() > 200, "vacuous"
    assert_no_lookahead(op, pf, tol=0.0)
    assert_no_lookahead(op, pf, tol=0.0, cut=89)  # the eve of the relabel
    # cuts on and off the refit grid (30, 37, ...), across the relabel at 90
    for cut in (40, 57, 90, 130):
        assert_prefix_invariant(op, pf, tol=0.0, cut=cut)


def test_pollet_wilson_fast_path_is_prefix_invariant() -> None:
    """The O(N) path (native rolling moments) on a stable universe, bitwise."""
    r, _ = _dense_returns(15, 80, seed=10)
    pf = PanelFrame(_long(r), **KEYS)

    def op(f):
        return cov.avg_correlation(
            f, returns="ret", window=15, kind="pollet_wilson", broadcast=True, **KEYS
        )

    assert op(pf.collect())["universe_stable"].sum() > 500
    for cut in (20, 40, 60):
        assert_prefix_invariant(op, pf, tol=0.0, cut=cut)
    assert_no_lookahead(op, pf, tol=0.0)


# --------------------------------------------------------------------------- #
# 6. Arguments, registry, import laziness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda df: cov.avg_correlation(df, returns="ret", kind="spearman", **KEYS),
            ValueError,
            "kind",
        ),
        (
            lambda df: cov.avg_correlation(df, returns="ret", window=1, **KEYS),
            ValueError,
            "window",
        ),
        (
            lambda df: cov.avg_correlation(df, returns="ret", window=5.0, **KEYS),
            TypeError,
            "window",
        ),
        (
            lambda df: cov.avg_correlation(df, returns="ret", min_coverage=0.0, **KEYS),
            ValueError,
            "min_coverage",
        ),
        (
            lambda df: cov.avg_correlation(df, returns="nope", **KEYS),
            ValueError,
            "not found",
        ),
        (
            lambda df: cov.avg_correlation(df, returns="ret", group="time", **KEYS),
            ValueError,
            "group",
        ),
        (
            lambda df: cov.common_idio_vol(df, returns="ret", n_factors=-1, **KEYS),
            ValueError,
            "n_factors",
        ),
        (
            lambda df: cov.avg_correlation(
                df.with_columns(avg_corr=pl.lit(1.0)),
                returns="ret",
                broadcast=True,
                **KEYS,
            ),
            ValueError,
            "overwrite",
        ),
        (
            lambda df: cov.common_idio_vol(
                df.with_columns(idio_vol=pl.lit(1.0)),
                returns="ret",
                broadcast=True,
                **KEYS,
            ),
            ValueError,
            "overwrite",
        ),
        (
            lambda df: cov.avg_correlation(
                pl.concat([df, df.head(1)]), returns="ret", **KEYS
            ),
            ValueError,
            "duplicated",
        ),
        (
            lambda df: cov.avg_correlation(df.to_pandas(), returns="ret", **KEYS),
            TypeError,
            "PanelFrame",
        ),
    ],
)
def test_bad_arguments_raise(call, error, match: str) -> None:
    r, sector = _dense_returns(5, 30, seed=11)
    with pytest.raises(error, match=match):
        call(_long(r, sector))


def test_panelframe_and_bare_frame_agree() -> None:
    df = _churn_panel()
    a = cov.avg_correlation(PanelFrame(df, **KEYS), returns="ret", window=15)
    b = cov.avg_correlation(df.lazy(), returns="ret", window=15, **KEYS)
    assert a.equals(b)


@pytest.mark.parametrize("name", ["avg_correlation", "common_idio_vol"])
def test_registered_as_rowwise_frame_ops(name: str) -> None:
    spec = registry.get(name)
    assert spec.namespace == "covariance"
    assert (spec.input_shape, spec.output_shape) == ("frame", "frame")
    assert spec.safe_scope == "rowwise" and spec.leakage_safe and not spec.panel_safe
    assert spec.backend_fn is getattr(cov, name)
    assert "T^2" not in (spec.cost_hint or "")


def test_covariance_is_lazy_and_light() -> None:
    """``import panelary`` does not load it; loading it pulls no heavy module."""
    probe = (
        "import json, sys\n"
        "import panelary\n"
        "before = 'panelary.covariance' in sys.modules\n"
        "panelary.covariance.avg_correlation\n"
        "tops = sorted({m.split('.')[0] for m in sys.modules})\n"
        "print(json.dumps({'before': before, 'tops': tops}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["before"] is False
    heavy = {"scipy", "sklearn", "pandas", "numba", "statsmodels"}
    assert not heavy & set(result["tops"])
