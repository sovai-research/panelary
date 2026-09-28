"""Leakage regression suite for `panelary.embed` (models `tests/test_leakage.py`).

Contract touched: ``leakage_safe``. Every transform is run through the
future-perturbation instrument (:func:`panelary.testing.assert_no_lookahead`)
and, where it is fitted, the train/test instrument
(:func:`panelary.testing.assert_no_train_test_leak`). Each is paired with a
**deliberately leaky** construction of the same transform that must *fail*
the same instrument -- a pass is only evidence if the instrument can see the
leak. Length-dependence is covered separately in
``test_embed_prefix_invariance.py``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.embed import (
    CrossSectionalEmbedder,
    EmbeddingCompressor,
    EmbeddingWarmupWarning,
    HydraEmbedder,
    QuantEmbedder,
    RandIntC22,
    RandomFourierFeatures,
    TensorSketch,
)
from panelary.testing import assert_no_lookahead, assert_no_train_test_leak

E, T = "entity", "time"
W = 12

pytestmark = pytest.mark.filterwarnings("ignore::panelary.embed.EmbeddingWarmupWarning")


def _panel(n_ent: int = 4, n_time: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for e in range(n_ent):
        n = n_time - 4 * e
        parts.append(
            pl.DataFrame(
                {
                    E: np.full(n, f"e{e}"),
                    T: np.arange(n, dtype=np.int64) + 2 * e,
                    "x": rng.standard_normal(n).cumsum() * (1 + e),
                    "y": rng.standard_normal(n),
                }
            )
        )
    return pl.concat(parts)


def _collect(out: Any) -> pl.DataFrame:
    return out.collect() if isinstance(out, PanelFrame) else out


# --------------------------------------------------------------------------- #
# Leaky constructions (the instruments must see these)
# --------------------------------------------------------------------------- #
def _centred(
    op: Callable[[pl.DataFrame], Any],
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """The banned ``center=True``: window centred on t (reads W // 2 rows ahead)."""

    def run(frame: pl.DataFrame) -> pl.DataFrame:
        lead = frame.sort(E, T).with_columns(pl.col("x", "y").shift(-(W // 2)).over(E))
        return _collect(op(lead))

    return run


def _whole_series_scaled(
    op: Callable[[pl.DataFrame], Any],
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """A full-sample per-entity z-score applied before the (trailing) embedding."""

    def run(frame: pl.DataFrame) -> pl.DataFrame:
        z = frame.with_columns(
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).over(E).alias(c)
            for c in ("x", "y")
        )
        return _collect(op(z))

    return run


def _fit_on_everything(
    factory: Callable[[], Any],
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    def run(frame: pl.DataFrame) -> pl.DataFrame:
        return _collect(factory().fit(frame).transform(frame))

    return run


class _GlobalBandwidthRFF(RandomFourierFeatures):
    """Leaky: scaler and median heuristic from the *whole* fitted sample, used at every date."""

    leakage_safe = False

    def _fit(self, panel: PanelFrame) -> None:
        super()._fit(panel)
        from panelary.embed._contract import matrix_from_columns

        X = matrix_from_columns(panel.collect(), self.feature_names_in_)
        X = X[np.isfinite(X).all(axis=1)]
        mean, std = X.mean(axis=0), X.std(axis=0)
        Z = (X - mean) / std
        idx = np.random.default_rng(0).integers(0, Z.shape[0], size=(200, 2))
        sigma = float(np.median(np.linalg.norm(Z[idx[:, 0]] - Z[idx[:, 1]], axis=1)))
        K = self.schedule_time_.shape[0]
        self.schedule_mean_ = np.tile(mean, (K, 1))
        self.schedule_std_ = np.tile(std, (K, 1))
        self.schedule_sigma_ = np.full(K, sigma)


# --------------------------------------------------------------------------- #
# Temporal (trailing-window) embedders
# --------------------------------------------------------------------------- #
TEMPORAL: dict[str, Callable[[], Any]] = {
    "QuantEmbedder": lambda: QuantEmbedder(
        window=W, columns=["x", "y"], warmup="null", entity=E, time=T
    ),
    "RandIntC22": lambda: RandIntC22(
        window=W, n_intervals=3, min_interval_length=8, warmup="null", entity=E, time=T
    ),
    "QuantEmbedder[drop]": lambda: QuantEmbedder(
        window=W, columns="x", warmup="drop", entity=E, time=T
    ),
    "HydraEmbedder": lambda: HydraEmbedder(
        window=W, n_groups=2, n_kernels=3, warmup="null", entity=E, time=T
    ),
    "HydraEmbedder[normalise]": lambda: HydraEmbedder(
        window=W,
        n_groups=2,
        n_kernels=3,
        normalise=True,
        warmup="null",
        entity=E,
        time=T,
    ),
}


@pytest.mark.parametrize("name", sorted(TEMPORAL))
def test_temporal_embedders_never_look_ahead(name):
    df = _panel()
    emb = TEMPORAL[name]().fit(df)
    for cut in (25, 40, 55):
        assert_no_lookahead(emb.transform, df, entity=E, time=T, tol=0.0, cut=cut)


@pytest.mark.parametrize("name", sorted(TEMPORAL))
def test_temporal_embedders_centred_window_is_caught(name):
    df = _panel()
    emb = TEMPORAL[name]().fit(df)
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(_centred(emb.transform), df, entity=E, time=T)


@pytest.mark.parametrize("name", sorted(TEMPORAL))
def test_temporal_embedders_whole_series_scaling_is_caught(name):
    df = _panel()
    emb = TEMPORAL[name]().fit(df)
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(_whole_series_scaled(emb.transform), df, entity=E, time=T)


# --------------------------------------------------------------------------- #
# Fitted row-wise transforms
# --------------------------------------------------------------------------- #
def _rff() -> RandomFourierFeatures:
    return RandomFourierFeatures(
        n_components=20, columns=["x", "y"], min_periods=5, entity=E, time=T
    )


def test_rff_fit_on_everything_is_still_causal():
    """The expanding schedule: even a full-sample fit leaks nothing into row t."""
    df = _panel()
    for cut in (20, 40):
        assert_no_lookahead(
            _fit_on_everything(_rff), df, entity=E, time=T, tol=0.0, cut=cut
        )


def test_rff_global_bandwidth_is_caught():
    df = _panel()

    def leaky() -> _GlobalBandwidthRFF:
        return _GlobalBandwidthRFF(
            n_components=20, columns=["x", "y"], min_periods=5, entity=E, time=T
        )

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(_fit_on_everything(leaky), df, entity=E, time=T)


def test_rff_train_test_walk_forward():
    df = _panel()
    train, test = df.filter(pl.col(T) <= 35), df.filter(pl.col(T) > 35)
    rff = _rff().fit(train)
    assert_no_train_test_leak(
        rff.transform, df, (train, test), entity=E, time=T, tol=0.0
    )


def _embedded(df: pl.DataFrame) -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmbeddingWarmupWarning)
        q = QuantEmbedder(
            window=8, columns="x", warmup="null", keep_features=True, entity=E, time=T
        )
        return q.fit_transform(df).collect()


@pytest.mark.parametrize("method", ["pca", "svd"])
def test_fitted_compressor_fit_in_the_past_passes_fit_on_everything_fails(method):
    def make() -> EmbeddingCompressor:
        return EmbeddingCompressor(method, dim=3, columns=["x_quant"], entity=E, time=T)

    def honest(frame: pl.DataFrame) -> pl.DataFrame:
        emb = _embedded(frame)
        return _collect(make().fit(emb.filter(pl.col(T) <= 20)).transform(emb))

    def leaky(frame: pl.DataFrame) -> pl.DataFrame:
        emb = _embedded(frame)
        return _collect(make().fit(emb).transform(emb))

    df = _panel()
    assert_no_lookahead(honest, df, entity=E, time=T, tol=0.0, cut=30)
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky, df, entity=E, time=T, cut=30)


def test_srp_compressor_fit_on_everything_is_harmless():
    def op(frame: pl.DataFrame) -> pl.DataFrame:
        emb = _embedded(frame)
        return _collect(
            EmbeddingCompressor("srp", dim=4, columns=["x_quant"], entity=E, time=T)
            .fit(emb)
            .transform(emb)
        )

    assert_no_lookahead(op, _panel(), entity=E, time=T, tol=0.0, cut=30)


# --------------------------------------------------------------------------- #
# Cross-sectional mode
# --------------------------------------------------------------------------- #
def _xs_panel(n_ent: int = 30, n_time: int = 24, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_time, n_ent))
    return pl.DataFrame(
        {
            E: np.tile([f"s{i:02d}" for i in range(n_ent)], n_time),
            T: np.repeat(np.arange(n_time, dtype=np.int64), n_ent),
            "x": (f + 0.4 * rng.standard_normal((n_time, n_ent))).ravel(),
            "y": (0.7 * f + 0.4 * rng.standard_normal((n_time, n_ent))).ravel(),
            "z": rng.standard_normal(n_time * n_ent),
        }
    )


def _xs(cls: type = CrossSectionalEmbedder, **kw: Any) -> Any:
    return cls(
        align=kw.pop("align", "procrustes"),
        n_components=2,
        min_cross_section=10,
        entity=E,
        time=T,
        **kw,
    )


class _GlobalBasisXS(CrossSectionalEmbedder):
    """Leaky: one basis from every date at once, projected per date."""

    leakage_safe = False

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        from panelary.embed._xs import _standardise

        _, M, _ = self._per_date(panel)
        ok = np.isfinite(M).all(axis=1)
        Vt = np.linalg.svd(_standardise(M[ok], self.standardize), full_matrices=False)[
            2
        ]
        self._global = Vt[: self.n_components].T
        return super()._transform(panel)

    def _date_basis(self, M: np.ndarray) -> Any:
        got = super()._date_basis(M)
        return None if got is None else (got[0], got[1], self._global)


@pytest.mark.parametrize("align", ["procrustes", "link"])
def test_cross_sectional_alignment_never_looks_ahead(align):
    df = _xs_panel()
    xs = _xs(align=align).fit(df)
    for cut in (5, 12, 20):
        assert_no_lookahead(xs.transform, df, entity=E, time=T, tol=0.0, cut=cut)


def test_cross_sectional_global_basis_is_caught():
    df = _xs_panel()
    leaky = _xs(_GlobalBasisXS).fit(df)
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky.transform, df, entity=E, time=T)


# --------------------------------------------------------------------------- #
# TensorSketch (stateless, row-wise)
# --------------------------------------------------------------------------- #
def _ts() -> TensorSketch:
    return TensorSketch(
        n_components=16, degree=2, seed=1, columns=["x", "y"], entity=E, time=T
    )


def test_tensor_sketch_fit_on_everything_is_harmless():
    assert_no_lookahead(
        _fit_on_everything(_ts), _panel(), entity=E, time=T, tol=0.0, cut=30
    )


def test_tensor_sketch_of_pooled_standardised_inputs_is_caught():
    def leaky(frame: pl.DataFrame) -> pl.DataFrame:
        pooled = frame.with_columns(
            (pl.col(c) - pl.col(c).mean()) / pl.col(c).std() for c in ("x", "y")
        )
        return _collect(_ts().fit(pooled).transform(pooled))

    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(leaky, _panel(), entity=E, time=T)
