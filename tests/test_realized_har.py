"""Golden pins and HAR extensions for the realized-measures refactor (plan 5, M4).

Two promises are pinned **bitwise** against frozen copies of the pre-refactor
code, so a future edit cannot drift them silently:

* ``daily_realized_measures`` now runs on the per-session expression builders
  shared with ``intraday_realized_measures``; its output is unchanged.
* ``HARModel(spec="har")`` -- the default -- is unchanged by the HARQ / SHAR /
  HAR-CJ / insanity-filter extensions.

The extensions themselves are checked against independent ``numpy.linalg.lstsq``
fits, and their fitted constants (HARQ's centre, the insanity bounds) against
the train/test leak verifier.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from panelary.core import PanelFrame
from panelary.econ.features import (
    HARModel,
    daily_realized_measures,
    har_features,
    intraday_realized_measures,
)
from panelary.econ.features._common import entity_arrays, ols
from panelary.testing import assert_no_train_test_leak

_MU1_SQ_INV = np.pi / 2.0


# --------------------------------------------------------------------------- #
# Frozen pre-refactor implementations (copied verbatim from a3eb644)
# --------------------------------------------------------------------------- #
def _daily_realized_measures_reference(
    df: pl.DataFrame, *, entity: str, date: str, returns: str
) -> pl.DataFrame:
    frame = df
    absr = pl.col(returns).abs()
    grouped = (
        frame.group_by([entity, date], maintain_order=True)
        .agg(
            (pl.col(returns) ** 2).sum().alias("rv"),
            (absr * absr.shift(1)).sum().alias("_bp"),
            pl.len().alias("n_obs"),
        )
        .with_columns(
            pl.when(pl.col("n_obs") > 1)
            .then(
                _MU1_SQ_INV * (pl.col("n_obs") / (pl.col("n_obs") - 1)) * pl.col("_bp")
            )
            .otherwise(None)
            .alias("bv")
        )
        .drop("_bp")
    )
    return (
        grouped.with_columns(
            pl.max_horizontal(pl.col("rv") - pl.col("bv"), pl.lit(0.0)).alias("jump")
        )
        .with_columns(
            pl.when(pl.col("rv") > 0)
            .then(pl.col("jump") / pl.col("rv"))
            .otherwise(None)
            .alias("rel_jump")
        )
        .sort([entity, date])
    )


class _ReferenceHAR:
    """The pre-extension ``HARModel`` fit/transform, as plain functions."""

    def __init__(
        self,
        *,
        returns: str | None,
        rv: str | None,
        horizon: int,
        window: int,
        lags: tuple[int, ...],
        log_target: bool,
        min_train_rows: int,
    ) -> None:
        self.returns, self.rv = returns, rv
        self.horizon, self.window, self.lags = horizon, window, lags
        self.log_target, self.min_train_rows = log_target, min_train_rows
        self.coef_: dict[object, np.ndarray] = {}
        self.pooled_coef_: np.ndarray | None = None

    def _terms_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        return har_features(
            df.lazy(),
            entity="entity",
            time="time",
            returns=self.returns,
            rv=self.rv,
            window=self.window,
            lags=self.lags,
        )

    def _term_names(self) -> list[str]:
        names = ["har_d", "har_w", "har_m"]
        return [
            names[i] if i < len(names) else f"har_l{lag}"
            for i, lag in enumerate(self.lags)
        ]

    def _design(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        terms = self._term_names()
        rv_col = "rv" if self.returns is not None else self.rv
        assert rv_col is not None
        target = np.asarray(data[rv_col], dtype=float)
        X = np.column_stack([np.ones(target.shape[0])] + [data[name] for name in terms])
        y = np.full(target.shape[0], np.nan)
        if self.horizon < target.shape[0]:
            y[: -self.horizon] = target[self.horizon :]
        if self.log_target:
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(y > 0, np.log(y), np.nan)
                X[:, 1:] = np.where(X[:, 1:] > 0, np.log(X[:, 1:]), np.nan)
        return X, y

    def fit(self, df: pl.DataFrame) -> _ReferenceHAR:
        frame = self._terms_frame(df)
        rv_col = "rv" if self.returns is not None else self.rv
        cols = [rv_col, *self._term_names()]
        pooled_X: list[np.ndarray] = []
        pooled_y: list[np.ndarray] = []
        for key, _idx, data in entity_arrays(frame, "entity", cols):
            X, y = self._design(data)
            good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if good.sum() >= max(self.min_train_rows, X.shape[1] + 2):
                self.coef_[key] = ols(X[good], y[good]).beta
            if good.any():
                pooled_X.append(X[good])
                pooled_y.append(y[good])
        if pooled_X:
            Xp = np.vstack(pooled_X)
            yp = np.concatenate(pooled_y)
            if yp.shape[0] > Xp.shape[1]:
                self.pooled_coef_ = ols(Xp, yp).beta
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        frame = self._terms_frame(df)
        rv_col = "rv" if self.returns is not None else self.rv
        cols = [rv_col, *self._term_names()]
        preds = np.full(frame.height, np.nan)
        for key, idx, data in entity_arrays(frame, "entity", cols):
            X, _ = self._design(data)
            beta = self.coef_.get(key, self.pooled_coef_)
            if beta is None:
                continue
            yhat = X @ beta
            if self.log_target:
                yhat = np.exp(yhat)
            preds[idx] = yhat
        return frame.with_columns(pl.Series("har_forecast", preds))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _intraday_returns(seed: int, *, with_nulls: bool) -> pl.DataFrame:
    """Unsorted entity order, ragged days, single-observation days, nulls."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, int, float | None]] = []
    for ent in ("B", "A", "C"):
        for day in rng.permutation(6):
            m = int(rng.choice([1, 2, 3, 17, 60]))
            for _ in range(m):
                val: float | None = float(rng.standard_normal() * 0.002)
                if with_nulls and rng.random() < 0.05:
                    val = None
                rows.append((ent, int(day), val))
    return pl.DataFrame(
        rows,
        schema={"entity": pl.Utf8, "date": pl.Int64, "ret": pl.Float64},
        orient="row",
    )


def _daily_panel(
    n: int = 500, entities: tuple[str, ...] = ("A", "B", "C")
) -> pl.DataFrame:
    """Daily panel with rv, rq, rs_pos, rs_neg and a jump column."""
    frames = []
    for i, ent in enumerate(entities):
        rng = np.random.default_rng(40 + i)
        logv = np.zeros(n)
        for t in range(1, n):
            logv[t] = 0.97 * logv[t - 1] + 0.25 * rng.standard_normal()
        var = 1e-4 * np.exp(logv)
        rv = var * rng.chisquare(78, n) / 78
        share = rng.uniform(0.3, 0.7, n)
        jump = np.where(rng.random(n) < 0.1, rv * rng.uniform(0.1, 0.5, n), 0.0)
        frames.append(
            pl.DataFrame(
                {
                    "entity": [ent] * n,
                    "time": np.arange(n, dtype=np.int64),
                    "rv": rv,
                    "rq": rv**2 * rng.uniform(0.8, 1.6, n),
                    "rs_pos": rv * share,
                    "rs_neg": rv * (1.0 - share),
                    "jump": jump,
                }
            )
        )
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# Golden pins
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_daily_realized_measures_is_bitwise_unchanged(
    seed: int, with_nulls: bool
) -> None:
    df = _intraday_returns(seed, with_nulls=with_nulls)
    got = daily_realized_measures(df, entity="entity", date="date", returns="ret")
    want = _daily_realized_measures_reference(
        df, entity="entity", date="date", returns="ret"
    )
    assert_frame_equal(got, want, check_exact=True)
    # A LazyFrame input takes the same path.
    lazy = daily_realized_measures(
        df.lazy(), entity="entity", date="date", returns="ret"
    )
    assert_frame_equal(lazy, want, check_exact=True)


@pytest.mark.parametrize(
    ("mode", "log_target", "horizon"),
    [("rv", False, 1), ("rv", True, 1), ("returns", False, 1), ("rv", False, 5)],
)
def test_har_model_default_spec_is_bitwise_unchanged(
    mode: str, log_target: bool, horizon: int
) -> None:
    df = _daily_panel(400).with_columns(ret=pl.col("rv").sqrt() * 0.5)
    train = df.filter(pl.col("time") < 300)
    kwargs = (
        {"rv": "rv", "returns": None}
        if mode == "rv"
        else {"rv": None, "returns": "ret"}
    )
    ref = _ReferenceHAR(
        horizon=horizon,
        window=22,
        lags=(1, 5, 22),
        log_target=log_target,
        min_train_rows=100,
        **kwargs,
    ).fit(train.filter(pl.col("entity") != "C"))
    model = HARModel(
        horizon=horizon, log_target=log_target, min_train_rows=100, **kwargs
    ).fit(
        PanelFrame(train.filter(pl.col("entity") != "C"), entity="entity", time="time")
    )
    assert set(model.coef_) == set(ref.coef_)
    for key, beta in ref.coef_.items():
        assert np.array_equal(model.coef_[key], beta)
    assert ref.pooled_coef_ is not None and model.pooled_coef_ is not None
    assert np.array_equal(model.pooled_coef_, ref.pooled_coef_)
    # Entity "C" was never seen: it takes the pooled fallback in both.
    got = model.transform(PanelFrame(df, entity="entity", time="time")).collect()
    want = ref.transform(df)
    assert_frame_equal(got, want, check_exact=True)


def test_intraday_measures_are_a_superset_of_daily_measures() -> None:
    """On null-free input the shared measures agree bitwise."""
    df = _intraday_returns(7, with_nulls=False).with_columns(
        tick=pl.int_range(pl.len()).over("entity", "date")
    )
    daily = daily_realized_measures(df, entity="entity", date="date", returns="ret")
    intra = intraday_realized_measures(
        df,
        entity="entity",
        session="date",
        time="tick",
        returns="ret",
        measures=["rv", "bv", "jump", "rel_jump", "n_obs"],
        min_obs=3,
    )
    joined = daily.join(intra, on=["entity", "date"], suffix="_i")
    assert joined.height == daily.height
    assert (joined["n_obs"] == joined["n_obs_i"]).all()
    full = joined.filter(pl.col("n_obs") >= 3)
    for col in ("rv", "bv", "jump", "rel_jump"):
        a, b = full[col].to_numpy(), full[f"{col}_i"].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), col


# --------------------------------------------------------------------------- #
# HAR extensions vs independent least squares
# --------------------------------------------------------------------------- #
def _trailing(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    for t in range(w - 1, x.shape[0]):
        out[t] = x[t - w + 1 : t + 1].mean()
    return out


def _lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
    return beta, good


def _one_entity(df: pl.DataFrame, ent: str) -> dict[str, np.ndarray]:
    sub = df.filter(pl.col("entity") == ent).sort("time")
    return {c: sub[c].to_numpy() for c in sub.columns if c not in ("entity", "time")}


def test_harq_matches_an_independent_fit() -> None:
    df = _daily_panel(500)
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(rv="rv", spec="harq", rq="rq", min_train_rows=50).fit(panel)
    # The centre is the training mean of sqrt(RQ), pooled over the fold.
    assert model.rq_center_ == pytest.approx(float(np.sqrt(df["rq"].to_numpy()).mean()))
    assert model.feature_names_ == ["har_d", "har_w", "har_m", "harq_rq"]
    d = _one_entity(df, "B")
    rv = d["rv"]
    X = np.column_stack(
        [
            np.ones(rv.size),
            rv,
            _trailing(rv, 5),
            _trailing(rv, 22),
            (np.sqrt(d["rq"]) - model.rq_center_) * rv,
        ]
    )
    y = np.r_[rv[1:], np.nan]
    beta, good = _lstsq(X, y)
    np.testing.assert_allclose(model.coef_["B"], beta, rtol=1e-8, atol=1e-12)
    out = model.transform(panel).collect().filter(pl.col("entity") == "B").sort("time")
    np.testing.assert_allclose(
        out["har_forecast"].to_numpy()[good], (X @ beta)[good], rtol=1e-8
    )


def test_harq_centring_is_a_reparametrisation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different centre moves beta_d, not the forecasts (the regressors span
    the same space), so the frozen training centre only fixes interpretation."""
    from panelary.econ.features import _harrv

    df = _daily_panel(300)
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(rv="rv", spec="harq", rq="rq", min_train_rows=50).fit(panel)
    base = model.transform(panel).collect()["har_forecast"].to_numpy()
    monkeypatch.setattr(_harrv, "_rq_center", lambda frame, rq: 0.0)
    uncentred = HARModel(rv="rv", spec="harq", rq="rq", min_train_rows=50).fit(panel)
    assert uncentred.rq_center_ == 0.0
    assert not np.allclose(uncentred.coef_["A"][1], model.coef_["A"][1])
    other = uncentred.transform(panel).collect()["har_forecast"].to_numpy()
    np.testing.assert_allclose(other, base, rtol=1e-9, atol=1e-16)


def test_shar_matches_an_independent_fit() -> None:
    df = _daily_panel(500)
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(
        rv="rv", spec="shar", rs_pos="rs_pos", rs_neg="rs_neg", min_train_rows=50
    ).fit(panel)
    assert model.feature_names_ == ["shar_rs_pos", "shar_rs_neg", "har_w", "har_m"]
    d = _one_entity(df, "A")
    rv = d["rv"]
    X = np.column_stack(
        [
            np.ones(rv.size),
            d["rs_pos"],
            d["rs_neg"],
            _trailing(rv, 5),
            _trailing(rv, 22),
        ]
    )
    beta, _good = _lstsq(X, np.r_[rv[1:], np.nan])
    np.testing.assert_allclose(model.coef_["A"], beta, rtol=1e-8, atol=1e-12)


def test_har_cj_matches_an_independent_fit() -> None:
    df = _daily_panel(500)
    panel = PanelFrame(df, entity="entity", time="time")
    model = HARModel(rv="rv", spec="har_cj", jump="jump", min_train_rows=50).fit(panel)
    assert model.feature_names_ == [
        "cj_c_d",
        "cj_c_w",
        "cj_c_m",
        "cj_j_d",
        "cj_j_w",
        "cj_j_m",
    ]
    d = _one_entity(df, "C")
    c, j = d["rv"] - d["jump"], d["jump"]
    X = np.column_stack(
        [np.ones(c.size)]
        + [_trailing(c, w) for w in (1, 5, 22)]
        + [_trailing(j, w) for w in (1, 5, 22)]
    )
    beta, _good = _lstsq(X, np.r_[d["rv"][1:], np.nan])
    np.testing.assert_allclose(model.coef_["C"], beta, rtol=1e-7, atol=1e-12)


def test_insanity_filter_replaces_out_of_range_forecasts_with_the_training_mean() -> (
    None
):
    df = _daily_panel(300, entities=("A",))
    train = df.filter(pl.col("time") < 200)
    model = HARModel(rv="rv", min_train_rows=50, insanity_filter=True).fit(
        PanelFrame(train, entity="entity", time="time")
    )
    y = train["rv"].to_numpy()
    lo, hi, mean = model.bounds_["A"]
    # Bounds are the training *target* (RV_{t+1}) over the fitted rows.
    assert lo >= y.min() and hi <= y.max()
    assert mean == pytest.approx(y[22:].mean(), rel=1e-12)
    # A test day with an absurd RV pushes the raw forecast out of range.
    spiked = df.with_columns(
        pl.when(pl.col("time") == 250)
        .then(pl.col("rv") * 1e4)
        .otherwise(pl.col("rv"))
        .alias("rv")
    )
    panel = PanelFrame(spiked, entity="entity", time="time")
    filtered = model.transform(panel).collect().sort("time")["har_forecast"].to_numpy()
    raw_model = HARModel(rv="rv", min_train_rows=50).fit(
        PanelFrame(train, entity="entity", time="time")
    )
    raw = raw_model.transform(panel).collect().sort("time")["har_forecast"].to_numpy()
    outside = np.isfinite(raw) & ((raw < lo) | (raw > hi))
    assert outside[250]
    np.testing.assert_array_equal(filtered[outside], mean)
    np.testing.assert_array_equal(filtered[~outside], raw[~outside])


def test_fitted_constants_come_from_the_training_fold_only() -> None:
    """Trap 11: HARQ's centre and the insanity bounds are frozen at fit."""
    df = _daily_panel(300)
    panel = PanelFrame(df, entity="entity", time="time")
    times = df["time"].unique().sort()
    split = (times.filter(times < 200), times.filter(times >= 200))

    def harq(frame: PanelFrame) -> PanelFrame:
        train = frame.filter(pl.col("time") < 200)
        model = HARModel(
            rv="rv", spec="harq", rq="rq", min_train_rows=50, insanity_filter=True
        ).fit(train)
        return model.transform(frame)

    assert_no_train_test_leak(harq, panel, split)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rv": "rv", "spec": "nope"}, "spec"),
        ({"rv": "rv", "spec": "harq"}, "rq"),
        ({"rv": "rv", "spec": "shar", "rs_pos": "rs_pos"}, "rs_neg"),
        ({"rv": "rv", "spec": "har_cj"}, "jump"),
        ({"returns": "ret", "spec": "harq", "rq": "rq"}, "rv="),
        (
            {
                "rv": "rv",
                "spec": "shar",
                "rs_pos": "a",
                "rs_neg": "b",
                "log_target": True,
            },
            "log_target",
        ),
        ({"rv": "rv", "rq": "rq"}, "not used"),
    ],
)
def test_har_spec_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HARModel(**kwargs)  # type: ignore[arg-type]
