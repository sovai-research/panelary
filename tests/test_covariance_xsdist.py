"""Cross-date distribution features: Kelly-Jiang tail risk and tail betas, the
Wasserstein distance between consecutive cross-sections, average skewness.

Plan ``covariance-and-market-state`` M4 (sections 5.15, 6, 7.3 and 7.6):

* the Hill oracle -- ``kelly_jiang_tail`` is ``1 / hill_index`` of the pooled
  trailing window with ``k`` matched, to 1e-12 (both raw and residual modes);
* the W1 oracle -- ``xs_wasserstein`` equals ``scipy.stats.wasserstein_distance``
  to 1e-12 on cross-sections of unequal size, raw and standardised;
* ``kelly_jiang_beta`` against a two-pass numpy regression on the lagged tail;
* ``avg_skewness`` against per-name numpy skewness;
* trap **T10** -- pooling the calendar month that contains ``t`` (the paper's
  monthly design reused daily) is caught as a look-ahead, the trailing window
  is not;
* prefix invariance and no look-ahead **bitwise** (``tol=0.0``) on a panel whose
  names enter and leave.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core.panel_frame import PanelFrame
from panelary.detect._panel import residualise
from panelary.econ.features._common import rolling_beta
from panelary.econ.features._evt import hill_index
from panelary.registry import registry
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

cov = pn.covariance
KEYS = {"entity": "entity", "time": "time"}


def _returns(n_ent: int, n_time: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(n_time)
    beta = 0.5 + rng.random(n_ent)
    # a slowly varying tail: heavier-tailed noise in the second half
    df = np.where(np.arange(n_time) < n_time // 2, 8.0, 3.0)[:, None]
    noise = rng.standard_t(df, (n_time, n_ent))
    return 0.01 * (beta * market[:, None] + noise)


def _long(r: np.ndarray, keep: np.ndarray | None = None) -> pl.DataFrame:
    keep = np.isfinite(r) if keep is None else keep
    t_idx, e_idx = np.nonzero(keep)
    return pl.DataFrame(
        {
            "entity": [f"E{i:03d}" for i in e_idx],
            "time": t_idx.astype(np.int64),
            "ret": r[t_idx, e_idx],
        }
    ).with_columns(pl.col("ret").fill_nan(None))


def _churn(
    seed: int = 0, n_ent: int = 60, n_time: int = 120
) -> tuple[np.ndarray, pl.DataFrame]:
    r = _returns(n_ent, n_time, seed)
    rng = np.random.default_rng(seed + 1)
    keep = np.ones_like(r, dtype=bool)
    for i in range(n_ent):
        if i % 3 == 0:
            keep[: rng.integers(5, 60), i] = False
        if i % 5 == 1:
            keep[rng.integers(60, 110) :, i] = False
    keep &= rng.random(r.shape) > 0.03
    dense = np.where(keep, r, np.nan)
    return dense, _long(r, keep)


def _col(df: pl.DataFrame, name: str) -> np.ndarray:
    return df[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()


# --------------------------------------------------------------------------- #
# 1. Kelly-Jiang tail risk
# --------------------------------------------------------------------------- #
def _pooled(r: np.ndarray, t: int, window: int) -> np.ndarray:
    block = r[t - window + 1 : t + 1]
    return block[np.isfinite(block)]


def _check_hill(
    out: pl.DataFrame,
    r: np.ndarray,
    window: int,
    q: float,
    min_exc: int,
    rtol: float = 0.0,
) -> int:
    tail = _col(out, "kj_tail")
    thr = _col(out, "kj_threshold")
    assert np.isnan(tail[: window - 1]).all()
    checked = 0
    for t in range(window - 1, r.shape[0]):
        pooled = _pooled(r, t, window)
        k = math.floor(q * pooled.size)
        assert out["kj_n_obs"][t] == pooled.size
        assert out["kj_n_exceed"][t] == k
        u = np.sort(pooled)[k] if k < pooled.size else np.nan
        if k < min_exc or not u < 0:
            assert np.isnan(tail[t]), t
            continue
        # an order statistic, never an interpolated quantile
        assert math.isclose(thr[t], u, rel_tol=rtol, abs_tol=0.0), t
        want = 1.0 / hill_index(pooled, k=k, tail="lower")
        assert math.isclose(tail[t], want, rel_tol=max(rtol, 1e-12)), t
        checked += 1
    return checked


def test_kelly_jiang_is_hill_on_the_pooled_trailing_window() -> None:
    r, df = _churn()
    out = cov.kelly_jiang_tail(df, returns="ret", window=10, **KEYS)
    assert _check_hill(out, r, 10, 0.05, 10) > 60
    # heavier-tailed noise in the second half shows up as a larger lambda
    tail = _col(out, "kj_tail")
    assert np.nanmean(tail[70:]) > np.nanmean(tail[10:55])


def test_kelly_jiang_parameters() -> None:
    r, df = _churn(seed=3)
    out = cov.kelly_jiang_tail(
        df, returns="ret", window=5, q=0.1, min_exceedances=25, **KEYS
    )
    assert _check_hill(out, r, 5, 0.1, 25) > 20
    assert out["kj_tail"].null_count() > 4  # some dates are below 25 exceedances


def test_kelly_jiang_residual_mode_pools_factor_residuals() -> None:
    r = _returns(40, 150, seed=4)
    fit_window, refit = 40, 10
    out = cov.kelly_jiang_tail(
        _long(r),
        returns="ret",
        window=10,
        residual=True,
        n_factors=1,
        fit_window=fit_window,
        refit_every=refit,
        **KEYS,
    )
    resid = residualise(
        r.T,
        n_factors=1,
        min_periods=fit_window,
        refit_every=refit,
        window=fit_window,
        min_obs=math.ceil(0.95 * fit_window),
    ).T
    tail = _col(out, "kj_tail")
    assert np.isnan(tail[:fit_window]).all()  # residuals start at fit_window
    # residualise on the whole balanced panel vs one compacted segment at a
    # time: same arithmetic, different memory layout, so equal to rounding
    assert _check_hill(out, resid, 10, 0.05, 10, rtol=1e-12) > 90


def test_t10_calendar_month_threshold_is_caught() -> None:
    """Trap T10: Kelly and Jiang estimate one threshold per calendar month.
    Reused on daily data, the value on the 3rd of the month then pools the
    rest of that month. The verifier must catch it; the trailing window passes."""
    _dense, df = _churn(seed=5)
    pf = PanelFrame(df, **KEYS)

    def calendar_month(f):
        month = (pl.col("time") // 21).alias("__month")
        f = f.with_columns(month)
        pooled = (
            f.filter(pl.col("ret").is_not_null())
            .group_by("__month")
            .agg(pl.col("ret").sort().alias("__v"))
        )
        rows = []
        for m, vals in pooled.iter_rows():
            v = np.asarray(vals)
            k = math.floor(0.05 * v.size)
            rows.append((m, float(np.mean(np.log(v[:k] / v[k])))))
        lam = pl.DataFrame(rows, schema=["__month", "kj_tail"], orient="row")
        return f.join(lam, on="__month", how="left").drop("__month")

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(calendar_month, pf)

    def trailing(f):
        return cov.kelly_jiang_tail(f, returns="ret", window=21, broadcast=True, **KEYS)

    assert trailing(df)["kj_tail"].drop_nulls().len() > 2000
    assert_no_lookahead(trailing, pf, tol=0.0)


# --------------------------------------------------------------------------- #
# 2. Tail betas
# --------------------------------------------------------------------------- #
def test_kelly_jiang_beta_is_the_slope_on_lagged_tail_risk() -> None:
    r, df = _churn(seed=6)
    window = 30
    tail = cov.kelly_jiang_tail(df, returns="ret", window=5, q=0.1, **KEYS)
    lam = _col(tail, "kj_tail")
    out = cov.kelly_jiang_beta(
        df, returns="ret", window=window, tail_window=5, q=0.1, **KEYS
    )
    assert out.columns == [*df.columns, "kj_tail_lag", "kj_beta"]
    lagged = out.select("time", "kj_tail_lag").unique().sort("time")
    np.testing.assert_array_equal(
        _col(lagged, "kj_tail_lag")[1:], lam[lagged["time"].to_numpy()[1:] - 1]
    )
    checked = 0
    for (name,), part in out.group_by(["entity"], maintain_order=True):
        part = part.sort("time")
        y = _col(part, "ret")
        x = _col(part, "kj_tail_lag")
        beta = _col(part, "kj_beta")
        for j in range(part.height):
            yy, xx = (
                y[max(0, j - window + 1) : j + 1],
                x[max(0, j - window + 1) : j + 1],
            )
            ok = np.isfinite(yy) & np.isfinite(xx)
            if ok.sum() < window:
                assert np.isnan(beta[j]), (name, j)
                continue
            dx = xx[ok] - xx[ok].mean()
            want = (dx * (yy[ok] - yy[ok].mean())).sum() / (dx * dx).sum()
            assert math.isclose(beta[j], want, rel_tol=1e-9, abs_tol=1e-12), (name, j)
            checked += 1
    assert checked > 500


def test_kelly_jiang_beta_shift_only_changes_rounding() -> None:
    """``rolling_beta`` is reused unchanged; the constant shift of the regressor
    moves the result by rounding only, and makes it closer to the two-pass value."""
    _r, df = _churn(seed=7)
    ours = cov.kelly_jiang_beta(
        df, returns="ret", window=40, tail_window=5, q=0.1, **KEYS
    )
    raw = rolling_beta(
        ours.with_columns(pl.col("ret").fill_nan(None)),
        entity="entity",
        time="time",
        y="ret",
        x="kj_tail_lag",
        window=40,
        alias="raw",
    )
    a, b = _col(raw, "kj_beta"), _col(raw, "raw")
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 500
    np.testing.assert_allclose(a[both], b[both], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# 3. Wasserstein distance between consecutive cross-sections
# --------------------------------------------------------------------------- #
def test_w1_matches_scipy_on_unequal_cross_sections() -> None:
    stats = pytest.importorskip("scipy.stats")
    r, df = _churn(seed=8)
    out = cov.xs_wasserstein(df, value="ret", **KEYS)
    got = _col(out, "xs_w1")
    assert np.isnan(got[0])
    sizes = set()
    for t in range(1, r.shape[0]):
        a, b = r[t][np.isfinite(r[t])], r[t - 1][np.isfinite(r[t - 1])]
        sizes.add((a.size, b.size))
        want = stats.wasserstein_distance(a, b)
        assert math.isclose(got[t], want, rel_tol=1e-12, abs_tol=1e-16), t
    assert len({x for x in sizes if x[0] != x[1]}) > 20


def test_w1_standardized_matches_scipy_on_z_scores() -> None:
    stats = pytest.importorskip("scipy.stats")
    r, df = _churn(seed=9)
    out = cov.xs_wasserstein(df, value="ret", standardize=True, **KEYS)
    got = _col(out, "xs_w1")

    def z(v: np.ndarray) -> np.ndarray:
        v = v[np.isfinite(v)]
        return (v - v.mean()) / v.std()

    for t in range(1, r.shape[0]):
        want = stats.wasserstein_distance(z(r[t]), z(r[t - 1]))
        assert math.isclose(got[t], want, rel_tol=1e-10, abs_tol=1e-14), t


def test_w1_grid_is_the_midpoint_rule_and_converges() -> None:
    r, df = _churn(seed=10)
    exact = _col(cov.xs_wasserstein(df, value="ret", **KEYS), "xs_w1")

    def q(v: np.ndarray, grid: int) -> np.ndarray:
        v = np.sort(v[np.isfinite(v)])
        u = (np.arange(grid) + 0.5) / grid
        return v[np.ceil(u * v.size).astype(int) - 1]

    errors = []
    for grid in (16, 256, 4096):
        got = _col(cov.xs_wasserstein(df, value="ret", grid=grid, **KEYS), "xs_w1")
        for t in (1, 40, 119):
            want = np.abs(q(r[t], grid) - q(r[t - 1], grid)).mean()
            assert math.isclose(got[t], want, rel_tol=1e-12)
        errors.append(np.nanmax(np.abs(got - exact) / exact))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.02


def test_w1_edge_cases() -> None:
    df = pl.DataFrame(
        {
            "entity": ["a", "b", "a", "a", "b", "a", "b", "a", "b"],
            "time": [0, 0, 1, 2, 2, 3, 3, 4, 4],
            "ret": [1.0, 2.0, None, 5.0, 5.0, 0.0, 4.0, 1.0, 1.0],
        }
    )
    got = cov.xs_wasserstein(df, value="ret", **KEYS)["xs_w1"].to_list()
    # 0: no previous date; 1 and 2: date 1 has nothing observed
    assert got[:3] == [None, None, None]
    assert got[3] == pytest.approx(3.0)  # {0, 4} vs {5, 5}
    assert got[4] == pytest.approx(2.0)  # {1, 1} vs {0, 4}: 0.5 * 1 + 0.5 * 3
    z = cov.xs_wasserstein(df, value="ret", standardize=True, **KEYS)["xs_w1"].to_list()
    # dates 2 and 4 are constant: no z-scores, so every pair touching them is null
    assert z == [None, None, None, None, None]


# --------------------------------------------------------------------------- #
# 4. Average skewness
# --------------------------------------------------------------------------- #
def test_avg_skewness_is_the_mean_of_per_name_skewness() -> None:
    r, df = _churn(seed=11)
    window = 15
    out = cov.avg_skewness(df, returns="ret", window=window, min_periods=12, **KEYS)
    got = _col(out, "avg_skew")
    per_name: dict[int, list[float]] = {}
    for (_name,), part in df.group_by(["entity"], maintain_order=True):
        part = part.sort("time")
        v = _col(part, "ret")
        times = part["time"].to_numpy()
        for j in range(part.height):
            if not np.isfinite(v[j]):
                continue
            w = v[max(0, j - window + 1) : j + 1]
            w = w[np.isfinite(w)]
            if w.size < 12:
                continue
            d = w - w.mean()
            m2, m3 = (d**2).mean(), (d**3).mean()
            per_name.setdefault(int(times[j]), []).append(m3 / m2**1.5)
    checked = 0
    for t in range(r.shape[0]):
        vals = per_name.get(t, [])
        assert out["n_entities"][t] == len(vals)
        if vals:
            assert math.isclose(
                got[t], float(np.mean(vals)), rel_tol=1e-9, abs_tol=1e-12
            )
            checked += 1
        else:
            assert np.isnan(got[t])
    assert checked > 80


# --------------------------------------------------------------------------- #
# 5. Leak safety, bitwise, on a churning panel
# --------------------------------------------------------------------------- #
_OPS = {
    "kj": (
        lambda f: cov.kelly_jiang_tail(
            f, returns="ret", window=8, broadcast=True, **KEYS
        ),
        "kj_tail",
    ),
    "kj_residual": (
        lambda f: cov.kelly_jiang_tail(
            f,
            returns="ret",
            window=8,
            residual=True,
            fit_window=30,
            refit_every=7,
            broadcast=True,
            **KEYS,
        ),
        "kj_tail",
    ),
    "kj_beta": (
        lambda f: cov.kelly_jiang_beta(
            f, returns="ret", window=20, tail_window=5, q=0.1, **KEYS
        ),
        "kj_beta",
    ),
    "w1": (
        lambda f: cov.xs_wasserstein(f, value="ret", broadcast=True, **KEYS),
        "xs_w1",
    ),
    "w1_standardized": (
        lambda f: cov.xs_wasserstein(
            f, value="ret", standardize=True, broadcast=True, **KEYS
        ),
        "xs_w1",
    ),
    "w1_grid": (
        lambda f: cov.xs_wasserstein(f, value="ret", grid=64, broadcast=True, **KEYS),
        "xs_w1",
    ),
    "avg_skew": (
        lambda f: cov.avg_skewness(f, returns="ret", window=10, broadcast=True, **KEYS),
        "avg_skew",
    ),
}


@pytest.mark.parametrize("name", sorted(_OPS))
def test_prefix_invariant_and_causal_bitwise(name: str) -> None:
    _dense, df = _churn(seed=12)
    pf = PanelFrame(df, **KEYS)
    op, feature = _OPS[name]
    assert op(df)[feature].drop_nulls().len() > 1000, "vacuous"
    assert_no_lookahead(op, pf, tol=0.0)
    for cut in (20, 37, 59, 90):
        assert_prefix_invariant(op, pf, tol=0.0, cut=cut)
        assert_no_lookahead(op, pf, tol=0.0, cut=cut)


# --------------------------------------------------------------------------- #
# 6. Arguments and registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda df: cov.kelly_jiang_tail(df, returns="ret", q=0.0, **KEYS),
            ValueError,
            "q",
        ),
        (
            lambda df: cov.kelly_jiang_tail(df, returns="ret", window=0, **KEYS),
            ValueError,
            "window",
        ),
        (
            lambda df: cov.kelly_jiang_tail(
                df, returns="ret", min_exceedances=0, **KEYS
            ),
            ValueError,
            "min_exceedances",
        ),
        (
            lambda df: cov.kelly_jiang_beta(df, returns="ret", window=1, **KEYS),
            ValueError,
            "window",
        ),
        (
            lambda df: cov.kelly_jiang_beta(
                df.with_columns(kj_beta=pl.lit(0.0)), returns="ret", **KEYS
            ),
            ValueError,
            "overwrite",
        ),
        (
            lambda df: cov.xs_wasserstein(df, value="ret", grid=0, **KEYS),
            ValueError,
            "grid",
        ),
        (
            lambda df: cov.xs_wasserstein(df, value="nope", **KEYS),
            ValueError,
            "not found",
        ),
        (
            lambda df: cov.avg_skewness(df, returns="ret", window=2, **KEYS),
            ValueError,
            "window",
        ),
        (
            lambda df: cov.avg_skewness(
                df, returns="ret", window=5, min_periods=6, **KEYS
            ),
            ValueError,
            "min_periods",
        ),
        (
            lambda df: cov.avg_skewness(df, returns="nope", **KEYS),
            ValueError,
            "not found",
        ),
    ],
)
def test_bad_arguments_raise(call, error, match: str) -> None:
    _dense, df = _churn(seed=13, n_ent=6, n_time=30)
    with pytest.raises(error, match=match):
        call(df)


@pytest.mark.parametrize(
    "name", ["kelly_jiang_tail", "kelly_jiang_beta", "xs_wasserstein", "avg_skewness"]
)
def test_registered_as_rowwise_frame_ops(name: str) -> None:
    fn = getattr(cov, name)  # the first public access registers the catalogue
    spec = registry.get(name)
    assert spec.namespace == "covariance"
    assert (spec.input_shape, spec.output_shape) == ("frame", "frame")
    assert spec.safe_scope == "rowwise" and spec.leakage_safe and not spec.panel_safe
    assert spec.backend_fn is fn
    assert "T^2" not in (spec.cost_hint or "")
