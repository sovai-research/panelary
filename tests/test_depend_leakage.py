"""Leakage regression suite for :mod:`panelary.depend`.

Contract: a dependence *estimate* is fit-free; a dependence-based *selection*
is not, and neither is any bandwidth, quantile threshold or random-feature map
("a bandwidth is a fit"). Each test builds a deliberately leaky variant and
asserts that it is detected -- the out-of-fold score of the leaky version is
materially better than the honest one on pure noise, or its train-period
output moves when test-period data arrive.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary.depend as dp
from panelary.core.panel_frame import PanelFrame

N_ENT, T_LEN, N_FEAT, K = 5, 200, 300, 10


@pytest.fixture(scope="module")
def noise_panel() -> PanelFrame:
    """Target and every feature are independent noise: any out-of-fold
    'signal' in the selected features is leakage."""
    rng = np.random.default_rng(20260928)
    n = N_ENT * T_LEN
    data = {
        "e": np.repeat(np.arange(N_ENT), T_LEN),
        "t": np.tile(np.arange(T_LEN), N_ENT),
        "y": rng.standard_normal(n),
    }
    data.update({f"f{j:03d}": rng.standard_normal(n) for j in range(N_FEAT)})
    return PanelFrame(pl.DataFrame(data), entity="e", time="t")


class _LeakySelector(dp.ScreenSelector):
    """Screens the FULL panel regardless of the fold it is fitted on."""

    full_panel: PanelFrame

    def _fit(self, panel: PanelFrame) -> None:
        super()._fit(self.full_panel)


def _oof_score(selector: dp.ScreenSelector, panel: PanelFrame) -> float:
    """Walk-forward split at the median date; score = mean signed test-fold
    Spearman of the selected features, signed by the train-time estimate."""
    train = panel.filter(pl.col("t") < T_LEN // 2)
    test = panel.filter(pl.col("t") >= T_LEN // 2).collect()
    selector.fit(train)
    assert selector.screen_ is not None
    sign = dict(
        zip(
            selector.screen_["feature"],
            np.sign(selector.screen_["estimate"]),
            strict=True,
        )
    )
    y = test["y"].to_numpy()
    return float(
        np.mean(
            [sign[f] * dp.spearman(test[f].to_numpy(), y) for f in selector.selected_]
        )
    )


def test_leaky_screen_is_detected(noise_panel: PanelFrame) -> None:
    kw = {
        "features": [f"f{j:03d}" for j in range(N_FEAT)],
        "method": "spearman",
        "null": "asymptotic",
        "k": K,
        "alpha": 1.0,
        "by": "pooled",
    }
    honest = dp.ScreenSelector("y", **kw)
    leaky = _LeakySelector("y", **kw)
    leaky.full_panel = noise_panel
    s_honest = _oof_score(honest, noise_panel)
    s_leaky = _oof_score(leaky, noise_panel)
    # Pure noise: an honest selection has no out-of-fold edge ...
    assert abs(s_honest) < 0.03, s_honest
    # ... the leaky one "finds" one, because the test fold voted in the selection.
    assert s_leaky > 0.05, s_leaky
    assert s_leaky > s_honest + 0.04


def test_screen_selector_selection_depends_on_train_only(
    noise_panel: PanelFrame,
) -> None:
    """Corrupting the test fold must not change what an honest selector picks."""
    feats = [f"f{j:03d}" for j in range(40)]
    train_mask = pl.col("t") < T_LEN // 2
    sel_a = dp.ScreenSelector(
        "y", features=feats, method="spearman", null="asymptotic", k=5, alpha=1.0
    )
    sel_a.fit(noise_panel.filter(train_mask))
    corrupted = noise_panel.with_columns(
        pl.when(~train_mask)
        .then(pl.col("y") * -3.0 + 7.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    sel_b = dp.ScreenSelector(
        "y", features=feats, method="spearman", null="asymptotic", k=5, alpha=1.0
    )
    sel_b.fit(corrupted.filter(train_mask))
    assert sel_a.selected_ == sel_b.selected_


# --------------------------------------------------------------------------- #
# Rolling features: no look-ahead, no length dependence, per entity
# --------------------------------------------------------------------------- #
def _rolling_panel() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    frames = []
    for name, n in {"A": 90, "B": 90, "C": 75, "D": 60}.items():
        x = rng.standard_normal(n)
        z = np.sin(x) + 0.5 * rng.standard_normal(n)
        frames.append(
            pl.DataFrame({"entity": [name] * n, "time": np.arange(n), "x": x, "z": z})
        )
    return pl.concat(frames).sort(["time", "entity"])


_ROLLING = {
    "rolling_xi": lambda: pl.col("x").ts.rolling_xi("z", window=15),
    "rolling_xi_default": lambda: pl.col("x").ts.rolling_xi(),
    "rolling_dcor": lambda: pl.col("x").ts.rolling_dcor("z", window=15),
    "rolling_tail_dep": lambda: pl.col("x").ts.rolling_tail_dep("z", window=20, q=0.2),
    "rolling_gcmi": lambda: pl.col("x").ts.rolling_gcmi("z", window=15),
}


@pytest.mark.parametrize("name", sorted(_ROLLING))
def test_rolling_ops_have_no_lookahead(name: str) -> None:
    from panelary.testing import assert_no_lookahead, assert_prefix_invariant

    op = _ROLLING[name]().over("entity").alias("f")
    panel = _rolling_panel()
    assert_no_lookahead(op, panel, entity="entity", time="time")
    assert_prefix_invariant(op, panel, entity="entity", time="time")


def test_rolling_ops_stay_inside_their_entity() -> None:
    panel = _rolling_panel()
    op = pl.col("x").ts.rolling_xi("z", window=15).over("entity").alias("f")
    base = panel.with_columns(op)
    other = panel.with_columns(
        pl.when(pl.col("entity") == "B")
        .then(pl.col("z") * -5.0)
        .otherwise(pl.col("z"))
        .alias("z")
    ).with_columns(op)
    keep = pl.col("entity") != "B"
    assert base.filter(keep)["f"].equals(other.filter(keep)["f"])


def test_global_tail_quantile_is_detected_by_the_library_verifier() -> None:
    """A tail threshold fitted on the full series is a length-dependent leak;
    assert_prefix_invariant must catch it (and must not flag ours)."""
    from panelary.depend._rolling import window_statistic
    from panelary.testing import assert_prefix_invariant

    def leaky(s: pl.Series) -> pl.Series:
        a = s.struct.field("a").to_numpy()
        b = s.struct.field("b").to_numpy()
        ta, tb = np.nanquantile(a, 0.2), np.nanquantile(b, 0.2)

        def rows(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
            return ((wa <= ta) & (wb <= tb)).mean(axis=1) / 0.2

        return pl.Series(window_statistic(a, b, 20, rows), nan_to_null=True)

    leaky_op = (
        pl.struct(pl.col("z").alias("a"), pl.col("x").alias("b"))
        .map_batches(leaky, return_dtype=pl.Float64)
        .over("entity")
        .alias("f")
    )
    panel = _rolling_panel()
    with pytest.raises(AssertionError):
        assert_prefix_invariant(leaky_op, panel, entity="entity", time="time")
    ours = (
        pl.col("x").ts.rolling_tail_dep("z", window=20, q=0.2).over("entity").alias("f")
    )
    assert_prefix_invariant(ours, panel, entity="entity", time="time")


def test_rolling_specs_meet_the_registry_rules() -> None:
    """The rules existing suites apply to every `.ts` spec (see
    tests/test_extract_features.py and tests/test_registry_conformance.py)."""
    from panelary.namespaces.ts import ROLLING_SPECS
    from panelary.registry import VALID_SAFE_SCOPES, registry

    audit = registry.audit()
    for spec in ROLLING_SPECS:
        assert registry.get(spec.name) is spec
        assert spec.namespace == "ts" and spec.source == "Panelary"
        assert spec.panel_safe and spec.leakage_safe
        assert spec.safe_scope == "rowwise" and spec.safe_scope in VALID_SAFE_SCOPES
        assert spec.output_shape != "scalar"
        for bucket, names in audit.items():
            if isinstance(names, (list, tuple, set, frozenset)):
                assert spec.name not in names, (bucket, spec.name)
        # Every parameter defaulted: the conformance suite can build it bare.
        method = getattr(pl.col("x").ts, spec.name)
        import inspect

        required = [
            p.name
            for p in inspect.signature(method).parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        assert required == [], (spec.name, required)
