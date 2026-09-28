"""Prefix invariance for `panelary.embed`: ``f(x[:T])[t] == f(x[:T+k])[t]``.

Contract touched: ``AGENTS.md`` hard invariant 1 (no quantity may depend on the
series length). :func:`panelary.testing.assert_prefix_invariant` truncates the
panel at several cuts and requires every surviving output cell to be
unchanged. Each transform is paired with a **length-dependent** leaky variant
that must fail -- including one that the future-*value* perturbation
instrument cannot see (it scales by the entity's row count), which is why this
file exists separately from ``test_embed_leakage.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.embed import (
    CrossSectionalEmbedder,
    EmbeddingCompressor,
    HydraEmbedder,
    QuantEmbedder,
    RandIntC22,
    RandomFourierFeatures,
    TensorSketch,
)
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

E, T = "entity", "time"
W = 10

pytestmark = pytest.mark.filterwarnings("ignore::panelary.embed.EmbeddingWarmupWarning")


def _panel(n_ent: int = 4, n_time: int = 64, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for e in range(n_ent):
        n = n_time - 5 * e
        parts.append(
            pl.DataFrame(
                {
                    E: np.full(n, f"e{e}"),
                    T: np.arange(n, dtype=np.int64) + 3 * e,
                    "x": rng.standard_normal(n).cumsum(),
                    "y": rng.standard_normal(n) * (1 + e),
                }
            )
        )
    return pl.concat(parts)


def _collect(out: Any) -> pl.DataFrame:
    return out.collect() if isinstance(out, PanelFrame) else out


def _by_length(
    op: Callable[[pl.DataFrame], Any],
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Leaky: scale each entity by its row count -- invisible to value perturbation."""

    def run(frame: pl.DataFrame) -> pl.DataFrame:
        return _collect(op(frame.with_columns((pl.col("x", "y") / pl.len()).over(E))))

    return run


def _whole_series(
    op: Callable[[pl.DataFrame], Any],
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Leaky: per-entity z-score over the whole series.

    A whole-series *centring* alone would not do: Hydra's kernels sum to zero
    and its scale features are location-free, so it is exactly shift-invariant.
    """

    def run(frame: pl.DataFrame) -> pl.DataFrame:
        cols = pl.col("x", "y")
        return _collect(
            op(frame.with_columns(((cols - cols.mean()) / cols.std()).over(E)))
        )

    return run


STATELESS: dict[str, Callable[[], Any]] = {
    "QuantEmbedder": lambda: QuantEmbedder(
        window=W, columns=["x", "y"], warmup="null", entity=E, time=T
    ),
    "QuantEmbedder[drop,columns]": lambda: QuantEmbedder(
        window=W, columns="x", output="columns", dtype="float64", entity=E, time=T
    ),
    "RandIntC22": lambda: RandIntC22(
        window=W, n_intervals=3, min_interval_length=6, warmup="null", entity=E, time=T
    ),
    "HydraEmbedder": lambda: HydraEmbedder(
        window=W, n_groups=2, n_kernels=3, warmup="null", entity=E, time=T
    ),
}


@pytest.mark.parametrize("name", sorted(STATELESS))
def test_stateless_embedders_are_prefix_invariant(name):
    emb = STATELESS[name]().fit(_panel())
    assert_prefix_invariant(emb.transform, _panel(), entity=E, time=T, tol=0.0)


@pytest.mark.parametrize("name", sorted(STATELESS))
def test_length_dependent_variant_fails_prefix_but_passes_perturbation(name):
    emb = STATELESS[name]().fit(_panel())
    leaky = _by_length(emb.transform)
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky, _panel(), entity=E, time=T)
    # The value-perturbation instrument is blind to this one -- by design.
    assert_no_lookahead(leaky, _panel(), entity=E, time=T)


@pytest.mark.parametrize("name", sorted(STATELESS))
def test_whole_series_scaling_fails_prefix(name):
    emb = STATELESS[name]().fit(_panel())
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(
            _whole_series(emb.transform), _panel(), entity=E, time=T
        )


def test_rff_fit_and_transform_on_a_prefix_equals_the_full_run():
    def op(frame: pl.DataFrame) -> pl.DataFrame:
        rff = RandomFourierFeatures(
            n_components=20,
            columns=["x", "y"],
            min_periods=4,
            output="columns",
            dtype="float64",
            entity=E,
            time=T,
        )
        return _collect(rff.fit(frame).transform(frame))

    assert_prefix_invariant(op, _panel(), entity=E, time=T, tol=1e-12)


class _GlobalScalerRFF(RandomFourierFeatures):
    """Leaky: the last (full-sample) schedule entry is used at every date."""

    leakage_safe = False

    def _fit(self, panel: PanelFrame) -> None:
        super()._fit(panel)
        K = self.schedule_time_.shape[0]
        self.schedule_mean_ = np.tile(self.schedule_mean_[-1], (K, 1))
        self.schedule_std_ = np.tile(self.schedule_std_[-1], (K, 1))
        self.schedule_sigma_ = np.full(K, self.schedule_sigma_[-1])


def test_rff_global_scaler_fails_prefix():
    def op(frame: pl.DataFrame) -> pl.DataFrame:
        rff = _GlobalScalerRFF(
            n_components=20, columns=["x", "y"], min_periods=4, entity=E, time=T
        )
        return _collect(rff.fit(frame).transform(frame))

    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(op, _panel(), entity=E, time=T)


def test_srp_compression_of_an_embedding_is_prefix_invariant():
    def op(frame: pl.DataFrame) -> pl.DataFrame:
        emb = (
            QuantEmbedder(window=W, columns="x", warmup="null", entity=E, time=T)
            .fit_transform(frame)
            .collect()
        )
        return _collect(
            EmbeddingCompressor("srp", dim=6, columns=["x_quant"], entity=E, time=T)
            .fit(emb)
            .transform(emb)
        )

    assert_prefix_invariant(op, _panel(), entity=E, time=T, tol=0.0)


def _xs_panel(n_ent: int = 25, n_time: int = 20, seed: int = 4) -> pl.DataFrame:
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


@pytest.mark.parametrize("align", ["procrustes", "link"])
def test_cross_sectional_embedder_is_prefix_invariant(align):
    xs = CrossSectionalEmbedder(
        align=align, n_components=2, min_cross_section=10, entity=E, time=T
    ).fit(_xs_panel())
    assert_prefix_invariant(xs.transform, _xs_panel(), entity=E, time=T, tol=0.0)


class _GlobalBasisXS(CrossSectionalEmbedder):
    """Leaky: one basis from every date in the frame (length-dependent)."""

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


def test_cross_sectional_global_basis_fails_prefix():
    """A pooled z-score would *not* fail: the per-date z-score cancels it."""
    leaky = _GlobalBasisXS(
        align="procrustes", n_components=2, min_cross_section=10, entity=E, time=T
    ).fit(_xs_panel())
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky.transform, _xs_panel(), entity=E, time=T)


def test_tensor_sketch_is_prefix_invariant_and_pooled_scaling_is_not():
    def honest(frame: pl.DataFrame) -> pl.DataFrame:
        ts = TensorSketch(n_components=16, seed=2, columns=["x", "y"], entity=E, time=T)
        return _collect(ts.fit(frame).transform(frame))

    def leaky(frame: pl.DataFrame) -> pl.DataFrame:
        pooled = frame.with_columns(
            (pl.col(c) - pl.col(c).mean()) / pl.col(c).std() for c in ("x", "y")
        )
        return honest(pooled)

    assert_prefix_invariant(honest, _panel(), entity=E, time=T, tol=0.0)
    with pytest.raises(AssertionError, match="PREFIX-INVARIANCE"):
        assert_prefix_invariant(leaky, _panel(), entity=E, time=T)
