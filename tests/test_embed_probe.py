"""PreValidated Ridge readout (`panelary.embed.PreValidatedRidge`).

Contracts touched: the section-5 guardrails -- never ridgeless (``alpha <= 0``
raises), a hard warning when features outnumber dates (Nagel, 2025) -- and
``leakage_safe`` for the readout: every statistic (scaler, penalty choice,
coefficients) comes from the rows passed to ``fit``.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from panelary.core.model_selection import PurgedKFold
from panelary.embed import OverparameterisedReadoutWarning, PreValidatedRidge


def _data(n: int = 120, p: int = 8, noise: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * np.linspace(0.5, 3.0, p)
    beta = rng.standard_normal(p)
    y = X @ beta + 2.0 + noise * rng.standard_normal(n)
    return X, y


def test_closed_form_loo_equals_brute_force_for_every_alpha():
    X, y = _data(n=40, p=5)
    m = PreValidatedRidge(alphas=[0.01, 1.0, 50.0]).fit(X, y)
    Xs = (X - m.mean_) / m.scale_
    for i, a in enumerate(m.alphas_):
        errs = []
        for j in range(X.shape[0]):
            keep = np.arange(X.shape[0]) != j
            b, b0 = PreValidatedRidge._solve(Xs[keep], y[keep, None], a)
            errs.append((y[j] - (Xs[j] @ b + b0)[0]) ** 2)
        assert m.cv_error_[i] == pytest.approx(np.mean(errs), rel=1e-10)


def test_prevalidated_predictions_and_coefficients():
    X, y = _data()
    m = PreValidatedRidge().fit(X, y)
    assert m.alpha_ in m.alphas_
    assert m.prevalidated_.shape == (X.shape[0], 1)
    # Coefficients are in the original units.
    Xs = (X - m.mean_) / m.scale_
    b, b0 = PreValidatedRidge._solve(Xs, y[:, None], m.alpha_)
    np.testing.assert_allclose(m.predict(X), (Xs @ b + b0)[:, 0], rtol=1e-10)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.95


@pytest.mark.parametrize("alphas", [[0.0, 1.0], [-1.0], [np.inf], []])
def test_ridgeless_or_invalid_penalties_are_refused(alphas):
    with pytest.raises(ValueError, match="alpha|empty"):
        PreValidatedRidge(alphas=alphas)


def test_overparameterised_readout_warns_with_the_nagel_pointer():
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((30, 50)), rng.standard_normal(30)
    with pytest.warns(OverparameterisedReadoutWarning, match="Nagel"):
        PreValidatedRidge().fit(X, y)
    # T counts distinct dates, not rows: 60 rows over 10 dates, 20 features.
    X2, y2 = rng.standard_normal((60, 20)), rng.standard_normal(60)
    with pytest.warns(OverparameterisedReadoutWarning, match="T=10 dates"):
        PreValidatedRidge().fit(X2, y2, time=np.repeat(np.arange(10), 6))
    with warnings.catch_warnings():
        warnings.simplefilter("error", OverparameterisedReadoutWarning)
        PreValidatedRidge().fit(X2, y2)  # 20 features < 60 rows


def test_date_block_cv_and_purged_kfold():
    X, y = _data(n=160)
    t = np.repeat(np.arange(40), 4)
    blocks = PreValidatedRidge(cv=4, purge=1).fit(X, y, time=t)
    assert blocks.prevalidated_ is None and np.isfinite(blocks.cv_error_).all()
    by_idx = PreValidatedRidge(cv=PurgedKFold(4, embargo=1, return_indices=True)).fit(
        X, y, time=t
    )
    by_panel = PreValidatedRidge(cv=PurgedKFold(4, embargo=1)).fit(X, y, time=t)
    np.testing.assert_allclose(by_idx.cv_error_, by_panel.cv_error_)
    with pytest.raises(ValueError, match="time"):
        PreValidatedRidge(cv=4).fit(X, y)


def test_noise_pushes_the_penalty_up():
    X, y = _data(n=60, p=30, noise=0.1)
    _, y_noisy = _data(n=60, p=30, noise=20.0)
    clean = PreValidatedRidge(warn_overparameterised=False).fit(X, y)
    noisy = PreValidatedRidge(warn_overparameterised=False).fit(X, y_noisy)
    assert noisy.alpha_ > clean.alpha_


def test_classification_with_prevalidated_calibration():
    X, y = _data(n=200)
    labels = np.where(y > np.median(y), "up", "down")
    m = PreValidatedRidge(task="classification").fit(X, labels)
    assert set(m.classes_) == {"down", "up"}
    assert (m.predict(X) == labels).mean() > 0.9
    P = m.predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert m.temperature_ is not None and m.temperature_ > 0
    with pytest.raises(AttributeError):
        PreValidatedRidge().fit(X, y).predict_proba(X)


def test_scaler_and_penalty_come_from_the_training_rows_only():
    X, y = _data(n=100)
    train = np.arange(100) < 70
    fit_a = PreValidatedRidge().fit(X[train], y[train])
    Xb = X.copy()
    Xb[~train] += 1e6  # corrupt the "test" rows
    fit_b = PreValidatedRidge().fit(Xb[train], y[train])
    np.testing.assert_array_equal(fit_a.coef_, fit_b.coef_)
    # The leaky construction -- scaling statistics from all rows -- moves them.
    leaky = PreValidatedRidge()
    leaky.fit(np.vstack([Xb[train], Xb[~train]]), np.r_[y[train], y[~train]])
    assert not np.allclose(leaky.coef_, fit_a.coef_)


def test_frame_helpers_drop_and_propagate_missing_rows():
    X, y = _data(n=50, p=3)
    df = pl.DataFrame(
        {"emb": X[:, :2], "s": X[:, 2], "y": y, "t": np.arange(50)}
    ).with_columns(pl.col("emb").cast(pl.Array(pl.Float64, 2)))
    df = df.with_columns(
        pl.when(pl.col("t") == 5).then(None).otherwise(pl.col("s")).alias("s")
    )
    m = PreValidatedRidge().fit_frame(df, features=["emb", "s"], target="y", time="t")
    assert m.n_features_in_ == 3 and m.n_times_ == 49
    pred = m.predict_frame(df, features=["emb", "s"])
    assert np.isnan(pred[5]) and np.isfinite(np.delete(pred, 5)).all()


def test_bad_inputs():
    X, y = _data()
    with pytest.raises(ValueError, match="NaN"):
        PreValidatedRidge().fit(np.where(X > 2, np.nan, X), y)
    with pytest.raises(RuntimeError, match="not fitted"):
        PreValidatedRidge().predict(X)
    with pytest.raises(ValueError):
        PreValidatedRidge(cv=1)
    with pytest.raises(ValueError):
        PreValidatedRidge(task="ranking")
