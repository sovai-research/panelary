"""The leak-safety contract of the shape algebra (plan section 9, item 2).

1. **Whole-series refuses across a boundary.** Every ``whole_series`` transform
   (``PAA`` / ``Spectral`` with ``flavour="whole_series"``, ``PartialTucker``)
   raises through ``_check_leakage`` -- directly and inside a ``Pipeline`` --
   and returns a frame keyed by entity alone, so joining it onto rows is a
   visible act.
2. **Entity axis refuses unaligned components.** ``axis="entity"``
   transforms refuse to emit per-date components as a panel column unless
   aligned; the aligned output is causal.
3. **Fitted feature-axis transforms learn from the fit panel only**, map rows
   independently, and compose in a walk-forward ``Pipeline``.
4. **The contract is enforced at class definition**, and can be narrowed per
   instance but never widened.
5. **The catalogue agrees with the code**: no registered shape spec is
   whole-series or leaky; the two excluded classes stay excluded.
6. **Import hygiene** (item 7): ``import panelary`` loads no shape transform,
   and no shape module pulls scipy / sklearn / tensorly.
7. **Budgets refuse before allocating.**

Modelled on ``tests/test_reduce_factor_leakage.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from panelary.core.panel_frame import PanelFrame
from panelary.core.pipeline import Pipeline
from panelary.registry import registry
from panelary.shape import (
    CUR,
    PAA,
    Axis,
    ColumnSubset,
    CrossSectionalRandomizedPCA,
    Delay,
    Flavour,
    FrequentDirections,
    Intent,
    PartialTucker,
    RandomizedPCA,
    ShapeBudgetError,
    ShapeSpec,
    ShapeTransform,
    Spectral,
    plan_chain,
)
from panelary.testing import assert_no_lookahead, assert_prefix_invariant


def _panel_df(n_ent: int = 6, n_t: int = 48, d: int = 6, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_ent * n_t
    latent = rng.standard_normal((n, 2))
    X = latent @ rng.standard_normal((2, d)) + 0.4 * rng.standard_normal((n, d))
    # Later dates drift, so anything that re-learns statistics is detectable.
    X = X + np.tile(np.linspace(0.0, 3.0, n_t), n_ent)[:, None]
    return pl.DataFrame(
        {
            "id": np.repeat([f"e{i}" for i in range(n_ent)], n_t),
            "t": np.tile(np.arange(n_t), n_ent),
            **{f"f{j}": X[:, j] for j in range(d)},
        }
    )


DF = _panel_df()
TRAIN = DF.filter(pl.col("t") < 32)
TEST = DF.filter(pl.col("t") >= 32)


def _pf(df: pl.DataFrame) -> PanelFrame:
    return PanelFrame(df, entity="id", time="t")


# --------------------------------------------------------------------------- #
# 1. whole-series transforms refuse across a train/test boundary
# --------------------------------------------------------------------------- #
WHOLE_SERIES: dict[str, Callable[[], ShapeTransform]] = {
    "paa": lambda: PAA(segments=4, flavour="whole_series"),
    "spectral": lambda: Spectral(k=3, flavour="whole_series"),
    "partial_tucker": lambda: PartialTucker({"time": 3, "feature": 2}),
}


@pytest.mark.parametrize("name", sorted(WHOLE_SERIES))
def test_whole_series_is_not_leakage_safe_and_refuses_across_a_boundary(
    name: str,
) -> None:
    t = WHOLE_SERIES[name]().fit(_pf(TRAIN))
    assert t.leakage_safe is False
    t._check_leakage(_pf(TRAIN))  # same panel: fine
    with pytest.raises(RuntimeError, match="leakage_safe = False"):
        t._check_leakage(_pf(TRAIN), _pf(TEST))


@pytest.mark.parametrize("name", sorted(WHOLE_SERIES))
def test_whole_series_trips_the_pipeline_gate(name: str) -> None:
    pipe = Pipeline([("shape", WHOLE_SERIES[name]())]).fit(_pf(TRAIN))
    with pytest.raises(RuntimeError, match="leakage_safe = False"):
        pipe.transform(_pf(TEST))


@pytest.mark.parametrize("name", sorted(WHOLE_SERIES))
def test_whole_series_output_is_keyed_by_entity_alone(name: str) -> None:
    out = WHOLE_SERIES[name]().fit_transform(_pf(TRAIN)).collect()
    assert out.height == TRAIN["id"].n_unique()
    assert out["t"].unique().to_list() == ["whole_series"]
    # Joining back onto rows on (entity, time) matches nothing -- a loud miss,
    # not a silent broadcast of a summary that saw the future.
    with pytest.raises(pl.exceptions.PolarsError):
        TRAIN.join(out, on=["id", "t"], how="inner")


def test_whole_series_refuses_keep_features() -> None:
    with pytest.raises(ValueError, match="look-ahead"):
        PAA(segments=4, flavour="whole_series", keep_features=True)
    with pytest.raises(ValueError, match="window"):
        PAA(segments=4)  # trailing needs a window: there is no implicit whole-series


def test_trailing_and_whole_series_flavours_resolve_their_specs() -> None:
    assert PAA(window=8, segments=4).shape_spec.flavour is Flavour.TRAILING
    ws = PAA(segments=4, flavour="whole_series")
    assert ws.shape_spec.flavour is Flavour.WHOLE_SERIES
    assert PAA.leakage_safe is True, "narrowing an instance must not touch the class"


# --------------------------------------------------------------------------- #
# 2. entity axis: refuse unaligned, aligned output is causal
# --------------------------------------------------------------------------- #
def test_entity_axis_refuses_unaligned_per_date_components() -> None:
    t = CrossSectionalRandomizedPCA(n_components=2)
    assert t.spec.axis is Axis.ENTITY and t.panel_safe is False and t.is_cross_sectional
    with pytest.raises(ValueError, match="rotation across dates"):
        t.fit_transform(_pf(DF))
    loadings = t.date_components(_pf(DF))
    assert set(loadings.columns) >= {"t", "component", "source_feature", "loading"}


def test_procrustes_aligned_components_are_causal() -> None:
    def op(frame: PanelFrame) -> PanelFrame:
        return CrossSectionalRandomizedPCA(
            n_components=2, align="procrustes"
        ).fit_transform(frame)

    pf = _pf(DF)
    assert_no_lookahead(op, pf, tol=0.0)
    assert_prefix_invariant(op, pf, tol=0.0)


def test_procrustes_alignment_gives_a_stable_basis_across_dates() -> None:
    """Isotropic 2-factor cross-sections: the per-date SVD basis is arbitrary
    within the subspace, the aligned one is not."""
    from panelary.shape._feature import fit_scaler

    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.standard_normal((5, 2)))[0]
    frames, mats = [], []
    for t in range(10):
        L = rng.standard_normal((200, 2)) @ Q.T + 0.01 * rng.standard_normal((200, 5))
        mats.append(L)
        frames.append(
            pl.DataFrame(
                {
                    "id": [f"e{i:03d}" for i in range(200)],
                    "t": [t] * 200,
                    **{f"f{j}": L[:, j] for j in range(5)},
                }
            )
        )
    df = pl.concat(frames)
    out = (
        CrossSectionalRandomizedPCA(n_components=2, align="procrustes")
        .fit_transform(_pf(df))
        .collect()
    )

    def implied(t: int) -> np.ndarray:
        M = mats[t]
        mean, scale = fit_scaler(M)
        scores = (
            out.filter(pl.col("t") == t)
            .sort("id")
            .select("xspc_1", "xspc_2")
            .to_numpy()
        )
        return np.linalg.lstsq((M - mean) / scale, scores, rcond=None)[0]

    aligned_steps = [np.linalg.norm(implied(t) - implied(t - 1)) for t in range(1, 10)]
    raw = CrossSectionalRandomizedPCA(n_components=2).date_components(_pf(df))

    def raw_v(t: int) -> np.ndarray:
        g = raw.filter(pl.col("t") == t).sort(["component", "source_feature"])
        return g["loading"].to_numpy().reshape(2, 5).T

    raw_steps = [np.linalg.norm(raw_v(t) - raw_v(t - 1)) for t in range(1, 10)]
    assert max(aligned_steps) < 0.25, aligned_steps
    assert max(raw_steps) > 0.5, raw_steps  # the unaligned basis does jump


# --------------------------------------------------------------------------- #
# 3. fitted feature-axis transforms: train-only, row-local, composable
# --------------------------------------------------------------------------- #
FITTED: dict[str, Callable[[], ShapeTransform]] = {
    "rsvd": lambda: RandomizedPCA(n_components=2),
    "frequent_directions": lambda: FrequentDirections(ell=4, n_components=2),
    "column_subset": lambda: ColumnSubset(k=2),
    "cur": lambda: CUR(k=2),
}


def _state_arrays(t: ShapeTransform) -> list[np.ndarray]:
    names = ("components_", "mean_", "scale_", "sketch_", "Z_", "U_")
    return [np.asarray(getattr(t, n)) for n in names if getattr(t, n, None) is not None]


@pytest.mark.parametrize("name", sorted(FITTED))
def test_fitted_state_comes_from_the_fit_panel_only(name: str) -> None:
    a = FITTED[name]().fit(_pf(TRAIN))
    b = FITTED[name]().fit(_pf(TRAIN))
    c = FITTED[name]().fit(_pf(DF))  # train + test
    sa, sb, sc = _state_arrays(a), _state_arrays(b), _state_arrays(c)
    assert all(np.array_equal(x, y) for x, y in zip(sa, sb, strict=True)), (
        "refit is not deterministic"
    )
    assert any(
        x.shape != y.shape or not np.array_equal(x, y)
        for x, y in zip(sa, sc, strict=True)
    ), (
        f"{name}: fitting on train+test gave identical state, so the test could not "
        "tell a train-only fit from a full-sample one"
    )
    # Transforming the test fold does not refit.
    before = [x.copy() for x in _state_arrays(a)]
    a.transform(_pf(TEST))
    assert all(
        np.array_equal(x, y) for x, y in zip(before, _state_arrays(a), strict=True)
    )


@pytest.mark.parametrize("name", sorted(FITTED))
def test_rows_are_mapped_independently(name: str) -> None:
    t = FITTED[name]().fit(_pf(TRAIN))
    full = t.transform(_pf(TEST)).collect().sort(["id", "t"])
    part = (
        t.transform(_pf(TEST.filter(pl.col("id") == "e3"))).collect().sort(["id", "t"])
    )
    joined = full.join(part, on=["id", "t"], suffix="_part")
    for c in part.columns:
        if c in ("id", "t"):
            continue
        assert np.array_equal(joined[c].to_numpy(), joined[f"{c}_part"].to_numpy())


def test_walk_forward_pipeline_of_trailing_lift_and_fitted_compress() -> None:
    """``Delay`` -> ``RandomizedPCA`` (SSA) survives a train/test boundary."""
    pipe = Pipeline(
        [
            ("delay", Delay(lags=4, columns=["f0", "f1"], dtype=pl.Float64)),
            ("rsvd", RandomizedPCA(n_components=2)),
        ]
    )
    train = TRAIN.filter(pl.col("t") >= 0)
    # Warm-up rows of the delay embedding are NaN: RandomizedPCA fits on
    # complete rows only and emits NaN for incomplete ones.
    pipe.fit(_pf(train))
    out = pipe.transform(_pf(TEST)).collect()
    assert out.height == TEST.height
    assert {"rpc_1", "rpc_2"} <= set(out.columns)


# --------------------------------------------------------------------------- #
# 4. the contract is enforced at class definition, narrowed never widened
# --------------------------------------------------------------------------- #
def _define(**attrs: object) -> type:
    body = {
        "panel_safe": True,
        "leakage_safe": True,
        "fit_is_empty": False,
        "is_cross_sectional": False,
        "_fit": lambda self, panel: None,
        "_transform": lambda self, panel: panel,
    }
    body.update(attrs)
    return type("_Probe", (ShapeTransform,), body)


def test_class_definition_refuses_contradictory_contracts() -> None:
    with pytest.raises(TypeError, match="whole_series"):
        _define(
            spec=ShapeSpec(
                intent=Intent.COMPRESS, axis=Axis.TIME, flavour=Flavour.WHOLE_SERIES
            )
        )
    with pytest.raises(TypeError, match="panel_safe"):
        _define(spec=ShapeSpec(intent=Intent.COMPRESS, axis=Axis.ENTITY))
    with pytest.raises(TypeError, match="is_cross_sectional"):
        _define(
            spec=ShapeSpec(intent=Intent.COMPRESS, axis=Axis.FEATURE),
            is_cross_sectional=True,
        )
    with pytest.raises(TypeError, match="spec"):
        _define()
    with pytest.raises(ValueError, match="flavour"):
        ShapeSpec(intent=Intent.COMPRESS, axis=Axis.TIME)
    with pytest.raises(ValueError):
        ShapeSpec(intent="expand", axis=Axis.FEATURE)
    ok = _define(spec=ShapeSpec(intent=Intent.COMPRESS, axis=Axis.FEATURE))
    assert ok.leakage_safe is True


def test_instances_narrow_but_never_widen() -> None:
    t = PartialTucker({"entity": 2, "time": 3})
    assert t.panel_safe is False and PartialTucker.panel_safe is True
    with pytest.raises(TypeError, match="cannot widen"):
        t._narrow_contract(leakage_safe=True)


# --------------------------------------------------------------------------- #
# 5. the catalogue agrees with the code
# --------------------------------------------------------------------------- #
def _shape_specs() -> list:
    return [s for s in registry.all() if s.namespace == "shape"]


def test_registered_shape_specs_are_leak_safe_and_match_their_classes() -> None:
    specs = _shape_specs()
    assert {s.name for s in specs} >= {
        "paa",
        "spectral",
        "delay",
        "rsvd",
        "sparse_rp",
        "srht",
        "count_sketch",
        "frequent_directions",
        "column_subset",
        "cur",
    }
    for s in specs:
        cls = s.backend_fn
        assert s.leakage_safe and cls.leakage_safe, s.name
        assert s.flavour != "whole_series", s.name
        assert s.panel_safe == cls.panel_safe and s.intent == str(cls.spec.intent)
        assert s.axis == str(cls.spec.axis) and s.safe_scope in {"rowwise", "window"}
        assert s.license == "Apache-2.0" and s.source
        # rowwise specs are exactly the stateless (fit_is_empty) ones: a fitted
        # map is only safe per fold, i.e. "window".
        assert (s.safe_scope == "rowwise") == bool(cls.fit_is_empty), s.name
    audit = registry.audit()
    for key in (
        "invalid_intent",
        "whole_series_leakage_safe",
        "quadratic_cost_in_core_tier",
    ):
        assert audit[key] == [], (key, audit[key])


def test_leaky_and_cross_sectional_classes_stay_out_of_the_catalogue() -> None:
    backends = {s.backend_fn for s in _shape_specs()}
    assert PartialTucker not in backends
    assert CrossSectionalRandomizedPCA not in backends


# --------------------------------------------------------------------------- #
# 6. import hygiene (plan section 9, item 7)
# --------------------------------------------------------------------------- #
_PROBE = r"""
import json, sys
import panelary
after_panelary = sorted(m for m in sys.modules if m.startswith("panelary.shape"))
import panelary.shape as shape
for name in shape.__all__:
    getattr(shape, name)
heavy = sorted({m.split(".")[0] for m in sys.modules} & {"scipy", "sklearn", "tensorly", "torch", "pandas"})
print(json.dumps({"after_panelary": after_panelary, "heavy": heavy}))
"""


def test_import_panelary_loads_no_shape_transform_and_shape_is_light() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    # The package initialiser and the materialization boundary are loaded
    # (panelary.cluster re-exports build_tensor); no transform module is.
    assert set(res["after_panelary"]) <= {"panelary.shape", "panelary.shape._tensor"}, (
        res
    )
    assert res["heavy"] == [], res


# --------------------------------------------------------------------------- #
# 7. budgets refuse before allocating
# --------------------------------------------------------------------------- #
def test_plan_refuses_before_allocating() -> None:
    t = PAA(window=8, segments=4, max_bytes=1_000)
    p = t.plan(_pf(DF))
    assert p.rows == DF.height and p.width == 6 * 4 and p.knob == "window"
    t.fit(_pf(DF))
    with pytest.raises(ShapeBudgetError, match="Lower `window`"):
        t.transform(_pf(DF))


def test_plan_chain_feeds_each_step_the_previous_output_shape() -> None:
    plans = plan_chain(
        [("delay", Delay(lags=16)), ("rsvd", RandomizedPCA(n_components=3))],
        _pf(DF),
        max_bytes=10**12,
    )
    assert plans[0].width == 6 * 16
    assert plans[1].in_width == 6 * 16 and plans[1].width == 3
    with pytest.raises(ShapeBudgetError, match="delay"):
        plan_chain([("delay", Delay(lags=16))], _pf(DF), max_bytes=100)
