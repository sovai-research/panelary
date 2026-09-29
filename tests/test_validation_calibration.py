"""CORP reliability diagrams, PAV and Murphy diagrams (``validation._calibration``).

PAV is checked against two independent oracles (scipy's
``optimize.isotonic_regression`` and scikit-learn's ``IsotonicRegression``,
both behind ``importorskip``) and against a brute-force min-max formula that
needs no library at all. The numba kernel must match its pure-Python twin
bitwise.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from panelary._internal._jit import assert_backend_parity, force_numpy
from panelary.validation import _calibration as cal
from panelary.validation._calibration import (
    CORPResult,
    _pav,
    _pav_rows,
    _pav_rows_kernel,
    _pav_rows_python,
    corp_reliability,
    murphy_diagram,
)


# --------------------------------------------------------------------------- #
# PAV
# --------------------------------------------------------------------------- #
def _pav_brute(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Isotonic fit by the min-max formula: max_{j<=i} min_{k>=i} avg(j..k)."""
    n = v.shape[0]
    cs = np.concatenate([[0.0], np.cumsum(v * w)])
    cw = np.concatenate([[0.0], np.cumsum(w)])
    out = np.empty(n)
    for i in range(n):
        best = -np.inf
        for j in range(i + 1):
            inner = min((cs[k + 1] - cs[j]) / (cw[k + 1] - cw[j]) for k in range(i, n))
            best = max(best, inner)
        out[i] = best
    return out


def test_pav_textbook_example() -> None:
    out = _pav(np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0]))
    np.testing.assert_array_equal(out, [1.0, 2.5, 2.5, 3.5, 3.5, 5.0])
    # already monotone -> unchanged (bitwise); decreasing -> the mean
    inc = np.array([0.1, 0.2, 0.2, 0.7])
    np.testing.assert_array_equal(_pav(inc), inc)
    np.testing.assert_allclose(_pav(np.array([3.0, 2.0, 1.0])), [2.0, 2.0, 2.0])


def test_pav_matches_brute_force_min_max_formula() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(1, 30))
        v = rng.standard_normal(n) + np.linspace(0, 1, n)
        w = rng.random(n) + 0.05
        np.testing.assert_allclose(_pav(v, w), _pav_brute(v, w), rtol=0, atol=1e-12)


def test_pav_matches_scipy_isotonic_regression() -> None:
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(4)
    v = rng.standard_normal(5000).cumsum() * 0.05 + rng.standard_normal(5000)
    w = rng.random(5000) + 0.1
    ref = opt.isotonic_regression(v, weights=w).x
    np.testing.assert_allclose(_pav(v, w), ref, rtol=0, atol=1e-12)


def test_pav_matches_sklearn_isotonic_regression() -> None:
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(5)
    x = np.sort(rng.random(3000))
    v = (rng.random(3000) < x**2).astype(float)
    w = rng.random(3000) + 0.1
    ref = isotonic.IsotonicRegression().fit(x, v, sample_weight=w).predict(x)
    np.testing.assert_allclose(_pav(v, w), ref, rtol=0, atol=1e-12)


def test_pav_twins_are_bitwise_identical() -> None:
    rng = np.random.default_rng(6)
    sums = rng.standard_normal((40, 257))
    weights = rng.random((40, 257))
    weights[weights < 0.15] = 0.0  # skipped entries come back NaN
    weights[3] = 0.0  # an all-empty row
    listed = np.empty_like(sums)
    _pav_rows_python(sums, weights, listed)
    arrayed = np.empty_like(sums)
    _pav_rows_kernel.py_func(sums, weights, arrayed)
    assert_backend_parity(arrayed, listed)
    assert np.isnan(listed[3]).all()
    assert np.array_equal(np.isnan(listed), ~(weights > 0))


def test_pav_numba_matches_python() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(7)
    sums = rng.standard_normal((25, 1000)).cumsum(axis=1)
    weights = rng.random((25, 1000)) + 0.01
    weights[:, ::17] = 0.0
    kernel = _pav_rows_kernel.compiled()
    assert kernel is not None
    fast = np.empty_like(sums)
    kernel(sums, weights, fast)
    twin = np.empty_like(sums)
    _pav_rows_python(sums, weights, twin)
    assert_backend_parity(fast, twin)
    with force_numpy():
        assert_backend_parity(_pav_rows(sums, weights), twin)


# --------------------------------------------------------------------------- #
# CORP
# --------------------------------------------------------------------------- #
def _calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    return p, (rng.random(n) < p).astype(float)


@pytest.mark.parametrize("score", ["brier", "log"])
def test_decomposition_identity_probability(score: str) -> None:
    rng = np.random.default_rng(8)
    p = np.clip(rng.beta(2, 3, 4000), 1e-6, 1 - 1e-6)
    y = (rng.random(4000) < p**1.5).astype(float)
    res = corp_reliability(p, y, score=score, band=None)
    assert res.mean_score == pytest.approx(res.mcb - res.dsc + res.unc, abs=1e-12)
    assert res.mcb > 0 and res.dsc > 0
    r = y.mean()
    if score == "brier":
        assert res.unc == pytest.approx(r * (1 - r), abs=1e-12)
        assert res.mean_score == pytest.approx(np.mean((p - y) ** 2), abs=1e-14)
    else:
        assert res.unc == pytest.approx(-(r * math.log(r) + (1 - r) * math.log(1 - r)))


def test_decomposition_identity_mean_functional() -> None:
    rng = np.random.default_rng(9)
    x = rng.standard_normal(3000)
    y = 0.6 * x + 0.3 + rng.standard_normal(3000)
    res = corp_reliability(x, y, functional="mean", band=None)
    assert res.score == "squared_error"
    assert res.mean_score == pytest.approx(res.mcb - res.dsc + res.unc, abs=1e-12)
    assert res.unc == pytest.approx(np.var(y), rel=1e-12)
    assert res.mcb > 0 and res.dsc > 0
    # the recalibrated curve is non-decreasing
    assert np.all(np.diff(res.recalibrated) >= 0)


def test_mcb_zero_for_recalibrated_forecasts() -> None:
    rng = np.random.default_rng(10)
    groups = rng.integers(0, 12, 5000)
    y = (rng.random(5000) < (groups + 1) / 14.0).astype(float)
    # forecast = the outcome mean of its group: isotonic-calibrated by design
    means = np.bincount(groups, weights=y) / np.bincount(groups)
    res = corp_reliability(means[groups], y, band=None)
    assert res.mcb == 0.0
    np.testing.assert_array_equal(res.recalibrated, res.x)


def test_dsc_zero_for_climatology() -> None:
    rng = np.random.default_rng(11)
    y = (rng.random(3000) < 0.3).astype(float)
    res = corp_reliability(np.full(3000, 0.3), y, band=None)
    assert res.dsc == 0.0
    assert res.x.shape == (1,)
    clim = corp_reliability(np.full(3000, y.mean()), y, band=None)
    assert clim.dsc == 0.0
    assert clim.mcb == pytest.approx(0.0, abs=1e-15)


def test_ties_are_pooled_before_pav() -> None:
    x = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.9, 0.9, 0.9])
    y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    res = corp_reliability(x, y, band=None)
    np.testing.assert_array_equal(res.x, [0.2, 0.5, 0.9])
    np.testing.assert_array_equal(res.weight, [3.0, 2.0, 3.0])
    # group means 1/3, 0, 2/3 -> pool the first two: 1/5, 1/5, 2/3
    np.testing.assert_allclose(res.recalibrated, [0.2, 0.2, 2 / 3], rtol=1e-15)


def test_log_score_infinite_is_reported_not_clipped() -> None:
    x = np.array([0.0, 0.3, 0.7, 1.0])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    res = corp_reliability(x, y, score="log", band=None)
    assert res.mean_score == math.inf and res.mcb == math.inf
    assert math.isfinite(res.dsc) and math.isfinite(res.unc)
    assert any("infinite" in w for w in res.warnings)
    assert res.to_dict()["mean_score"] is None  # JSON-safe


def test_nonfinite_pairs_dropped_with_warning() -> None:
    p, y = _calibrated(500, 12)
    p[[3, 7]] = np.nan
    y[11] = np.inf
    res = corp_reliability(p, y, band=None)
    assert res.n_obs == 497
    assert any("dropped 3" in w for w in res.warnings)


def test_input_validation() -> None:
    p, y = _calibrated(50, 13)
    with pytest.raises(ValueError, match="same length"):
        corp_reliability(p, y[:-1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        corp_reliability(p + 1.0, y)
    with pytest.raises(ValueError, match="binary"):
        corp_reliability(p, y + 0.5)
    with pytest.raises(ValueError, match="functional"):
        corp_reliability(p, y, functional="median")
    with pytest.raises(ValueError, match="score"):
        corp_reliability(p, y, functional="mean", score="log")
    with pytest.raises(ValueError, match="consistency"):
        corp_reliability(p, y, functional="mean", band="consistency")
    with pytest.raises(ValueError, match="level"):
        corp_reliability(p, y, level=1.5)
    with pytest.raises(ValueError, match="no pairs"):
        corp_reliability([np.nan], [1.0])


def test_no_fit_or_transform_surface() -> None:
    """Trap T4: the in-sample recalibration must not be reusable as a forecast."""
    for attr in ("fit", "predict", "transform", "fit_transform", "predict_proba"):
        assert not hasattr(CORPResult, attr)
    res = corp_reliability(*_calibrated(200, 14), band=None)
    assert not callable(res)
    with pytest.raises(AttributeError):
        res.mcb = 0.0  # type: ignore[misc]


def test_consistency_band_shape_order_and_determinism() -> None:
    p, y = _calibrated(2000, 15)
    a = corp_reliability(p, y, n_boot=200, seed=3)
    b = corp_reliability(p, y, n_boot=200, seed=3)
    c = corp_reliability(p, y, n_boot=200, seed=4)
    assert a.band == "consistency" and a.level == 0.9 and a.n_boot == 200
    assert a.band_low is not None and a.band_high is not None
    assert a.band_low.shape == a.x.shape
    assert np.all(a.band_low <= a.band_high)
    assert_backend_parity(a.band_low, b.band_low)  # same seed -> identical
    assert c.band_low is not None
    assert not np.array_equal(a.band_low, c.band_low)
    # under calibration the band is centred on the diagonal
    assert np.mean((a.band_low <= a.x) & (a.x <= a.band_high)) > 0.95


def test_band_with_repeated_forecast_levels_uses_binomial_counts() -> None:
    rng = np.random.default_rng(16)
    levels = np.linspace(0.05, 0.95, 10)
    p = levels[rng.integers(0, 10, 3000)]
    y = (rng.random(3000) < p).astype(float)
    res = corp_reliability(p, y, n_boot=300)
    assert res.band_low is not None and res.band_high is not None
    assert res.x.shape == (10,)
    assert np.all((res.band_low <= levels + 1e-12) & (levels - 1e-12 <= res.band_high))


def test_band_backends_agree_bitwise() -> None:
    pytest.importorskip("numba")
    p, y = _calibrated(1500, 17)
    fast = corp_reliability(p, y, n_boot=100)
    with force_numpy():
        twin = corp_reliability(p, y, n_boot=100)
    assert fast.band_low is not None and twin.band_low is not None
    assert_backend_parity(fast.band_low, twin.band_low)
    assert_backend_parity(fast.band_high, twin.band_high)
    assert_backend_parity(fast.recalibrated, twin.recalibrated)


def test_confidence_band_for_mean_functional() -> None:
    rng = np.random.default_rng(18)
    x = rng.standard_normal(800)
    y = x + rng.standard_normal(800)
    res = corp_reliability(x, y, functional="mean", n_boot=100)
    assert res.band == "confidence"
    assert res.band_low is not None and res.band_high is not None
    assert np.all(np.isfinite(res.band_low)) and np.all(res.band_low <= res.band_high)
    # a confidence band surrounds the fitted curve at most points
    inside = (res.band_low <= res.recalibrated) & (res.recalibrated <= res.band_high)
    assert inside.mean() > 0.8


def test_auto_band_skipped_with_warning_when_too_large(monkeypatch) -> None:
    monkeypatch.setattr(cal, "_BAND_AUTO_CELLS", 1000)
    res = corp_reliability(*_calibrated(500, 19), n_boot=100)
    assert res.band is None and res.band_low is None
    assert any("band skipped" in w for w in res.warnings)
    monkeypatch.setattr(cal, "_BAND_MAX_CELLS", 1000)
    with pytest.raises(ValueError, match="lower `n_boot`"):
        corp_reliability(*_calibrated(500, 19), n_boot=100, band="consistency")


def test_frame_and_json_payload() -> None:
    res = corp_reliability(*_calibrated(300, 20), n_boot=50)
    frame = res.to_frame()
    assert frame.columns == ["x", "recalibrated", "weight", "band_low", "band_high"]
    assert frame.height == res.x.shape[0]
    payload = json.loads(json.dumps(res.to_dict()))
    assert payload["band"] == "consistency" and payload["n_obs"] == 300
    no_band = corp_reliability(*_calibrated(300, 20), band=None).to_frame()
    assert no_band["band_low"].null_count() == no_band.height


@pytest.mark.slow
def test_consistency_band_pointwise_coverage_under_calibration() -> None:
    """A calibrated forecast's CORP curve lies inside the 90% band ~90% of the time."""
    inside = []
    for rep in range(100):
        p, y = _calibrated(1000, 1000 + rep)
        res = corp_reliability(p, y, n_boot=200, level=0.90, seed=rep)
        assert res.band_low is not None and res.band_high is not None
        ok = (res.band_low <= res.recalibrated) & (res.recalibrated <= res.band_high)
        inside.append(ok.mean())
    assert np.mean(inside) >= 0.88


def test_corp_is_stable_where_binned_ece_is_not() -> None:
    """The motivating example: binned ECE moves with the bin count, CORP does not."""
    rng = np.random.default_rng(21)
    p = rng.beta(0.7, 0.7, 2000)
    y = (rng.random(2000) < p).astype(float)

    def ece(bins: int) -> float:
        idx = np.minimum((p * bins).astype(int), bins - 1)
        n_b = np.bincount(idx, minlength=bins)
        gap = np.abs(np.bincount(idx, weights=p - y, minlength=bins))
        return float(gap[n_b > 0].sum() / p.size)

    eces = [ece(b) for b in (5, 10, 20, 50)]
    assert max(eces) / min(eces) > 2.0  # the verdict moves with the bins
    # CORP has no bins to move, and it does not depend on the row order either.
    res = corp_reliability(p, y, band=None)
    perm = rng.permutation(p.size)
    again = corp_reliability(p[perm], y[perm], band=None)
    for a, b in [(res.mcb, again.mcb), (res.dsc, again.dsc), (res.unc, again.unc)]:
        assert a == pytest.approx(b, rel=1e-13, abs=1e-16)
    assert_backend_parity(res.recalibrated, again.recalibrated)


# --------------------------------------------------------------------------- #
# Murphy diagrams
# --------------------------------------------------------------------------- #
def _elementary(
    x: np.ndarray, y: np.ndarray, theta: float, functional: str, level: float
) -> np.ndarray:
    if functional == "quantile":
        return ((y < x) - level) * ((theta < x).astype(float) - (theta < y))
    if functional == "probability":
        return theta * ((y == 0) & (x > theta)) + (1 - theta) * (
            (y == 1) & (x <= theta)
        )
    tau = 0.5 if functional == "mean" else level
    b = np.abs((y < x) - tau)
    return b * (
        np.maximum(y - theta, 0.0) - np.maximum(x - theta, 0.0) - (y - x) * (theta < x)
    )


@pytest.mark.parametrize(
    ("functional", "level"),
    [("quantile", 0.1), ("quantile", 0.5), ("expectile", 0.8), ("mean", 0.5)],
)
def test_murphy_exact_matches_brute_force(functional: str, level: float) -> None:
    rng = np.random.default_rng(22)
    y = rng.standard_normal(300) * 2 + 5
    fc = np.column_stack([y + rng.standard_normal(300), np.full(300, 5.0)])
    md = murphy_diagram(fc, y, functional=functional, level=level, names=["a", "b"])
    for j, name in enumerate(["a", "b"]):
        sub = md.filter(md["model"] == name)
        thetas = sub["theta"].to_numpy()
        got = sub["mean_elementary_score"].to_numpy()
        brute = np.array(
            [_elementary(fc[:, j], y, t, functional, level).mean() for t in thetas]
        )
        np.testing.assert_allclose(got, brute, rtol=0, atol=1e-12)
    # a user grid (including off-breakpoint points) is exact as well
    grid = np.linspace(-3, 13, 97)
    md2 = murphy_diagram(fc[:, 0], y, functional=functional, level=level, thetas=grid)
    brute = np.array(
        [_elementary(fc[:, 0], y, t, functional, level).mean() for t in grid]
    )
    np.testing.assert_allclose(
        md2["mean_elementary_score"].to_numpy(), brute, rtol=0, atol=1e-12
    )


def test_murphy_probability_matches_brute_force() -> None:
    p, y = _calibrated(400, 23)
    md = murphy_diagram(np.column_stack([p, p**2]), y, functional="probability")
    assert md["model"].unique().sort().to_list() == ["model_0", "model_1"]
    for j, name in enumerate(["model_0", "model_1"]):
        sub = md.filter(md["model"] == name)
        f = p if j == 0 else p**2
        brute = np.array(
            [_elementary(f, y, t, "probability", 0.5).mean() for t in sub["theta"]]
        )
        np.testing.assert_allclose(
            sub["mean_elementary_score"].to_numpy(), brute, rtol=0, atol=1e-12
        )


def _integral(x: np.ndarray, y: np.ndarray, functional: str, level: float) -> float:
    """Integrate the Murphy curve exactly: it is linear between breakpoints."""
    knots = np.unique(np.concatenate([x, y]))
    if functional == "probability":
        knots = np.unique(np.concatenate([[0.0, 1.0], knots]))
    mids = 0.5 * (knots[1:] + knots[:-1])
    md = murphy_diagram(x, y, functional=functional, level=level, thetas=mids)
    return float(np.sum(md["mean_elementary_score"].to_numpy() * np.diff(knots)))


def test_murphy_mixture_identities() -> None:
    """Ehm et al. (2016): integrating the elementary scores gives the score."""
    from panelary.validation import pinball_loss

    rng = np.random.default_rng(24)
    y = rng.standard_normal(500)
    x = 0.7 * y + rng.standard_normal(500) * 0.5 + 0.2
    for alpha in (0.05, 0.5, 0.9):
        assert _integral(x, y, "quantile", alpha) == pytest.approx(
            float(np.mean(pinball_loss(y, x, alpha))), abs=1e-10
        )
    assert _integral(x, y, "mean", 0.5) == pytest.approx(
        float(np.mean((x - y) ** 2)) / 4.0, abs=1e-10
    )
    tau = 0.7
    expectile_score = np.abs((y < x) - tau) * (x - y) ** 2
    assert _integral(x, y, "expectile", tau) == pytest.approx(
        float(np.mean(expectile_score)) / 2.0, abs=1e-10
    )
    p, yb = _calibrated(500, 25)
    assert _integral(p, yb, "probability", 0.5) == pytest.approx(
        float(np.mean((p - yb) ** 2)) / 2.0, abs=1e-10
    )


def test_murphy_perfect_forecast_is_zero_and_dominates() -> None:
    rng = np.random.default_rng(26)
    y = rng.standard_normal(200)
    md = murphy_diagram(np.column_stack([y, y + 0.3]), y, names=["perfect", "biased"])
    wide = md.pivot(on="model", index="theta", values="mean_elementary_score")
    assert np.all(wide["perfect"].to_numpy() == 0.0)
    assert np.all(wide["biased"].to_numpy() >= 0.0)


def test_murphy_nan_rows_are_dropped_per_model() -> None:
    rng = np.random.default_rng(27)
    y = rng.standard_normal(100)
    fc = np.column_stack([y + 0.1, y - 0.1])
    fc[:10, 0] = np.nan
    md = murphy_diagram(fc, y, thetas=[0.0, 0.5])
    ref = murphy_diagram(fc[10:, 0], y[10:], thetas=[0.0, 0.5])
    np.testing.assert_array_equal(
        md.filter(md["model"] == "model_0")["mean_elementary_score"].to_numpy(),
        ref["mean_elementary_score"].to_numpy(),
    )


def test_murphy_validation(monkeypatch) -> None:
    y = np.array([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="functional"):
        murphy_diagram(y, y, functional="median")
    with pytest.raises(ValueError, match="level"):
        murphy_diagram(y, y, functional="quantile", level=1.0)
    with pytest.raises(ValueError, match="rows"):
        murphy_diagram(y[:2], y)
    with pytest.raises(ValueError, match="names"):
        murphy_diagram(y, y, names=["a", "b"])
    with pytest.raises(ValueError, match="binary"):
        murphy_diagram(y, y + 0.5, functional="probability")
    monkeypatch.setattr(cal, "_MURPHY_MAX_ROWS", 2)
    with pytest.raises(ValueError, match="explicit `thetas`"):
        murphy_diagram(np.arange(5.0), np.arange(5.0) + 0.5)
