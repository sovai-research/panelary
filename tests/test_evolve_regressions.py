"""Regressions for `panelary.evolve` found by an external run of the search.

Each class pins one defect: the failing behaviour is reproduced on a panel with
a genuine, planted signal (``panelary.synth`` with ``signal_observed=True``),
and the test asserts the corrected behaviour.
"""

from __future__ import annotations

import functools

import numpy as np
import polars as pl
import pytest

evolve = pytest.importorskip("panelary.evolve")

from panelary.evolve._compile import compile_population  # noqa: E402
from panelary.evolve._fitness import PanelEvaluator  # noqa: E402
from panelary.evolve._honest import TrialLedger  # noqa: E402
from panelary.evolve._types import EvalContext, Gene, Genome  # noqa: E402
from panelary.factor import forward_return  # noqa: E402
from panelary.synth import SynthConfig, generate_panel  # noqa: E402

CTX = EvalContext(
    base_columns=("value", "signal"),
    base_units=("any", "any"),
    entity="entity",
    time="time",
)


@functools.lru_cache(maxsize=4)
def _planted(seed: int = 0, n_entities: int = 30, n_periods: int = 120) -> pl.DataFrame:
    """A synthetic panel whose ``fwd`` target the ``signal`` column predicts.

    ``fwd`` is ``value`` ``signal_lag`` steps ahead, built with
    :func:`panelary.factor.forward_return`, so the last ``signal_lag`` dates of
    every entity have a null target -- as any real forward label does.
    """
    data = generate_panel(
        SynthConfig(n_entities=n_entities, n_periods=n_periods, signal_observed=True),
        seed=seed,
    )
    lag = int(data.truth.signal_lag)
    frame = forward_return(
        data.panel, entity="entity", time="time", ret="value", horizon=lag, out="fwd"
    )
    return frame.select(["entity", "time", "value", "signal", "fwd"])


def _genome(op: str, arg: int, param_ix: int = 0) -> Genome:
    return Genome(genes=(Gene(op, (arg,), param_ix),), n_base=2, out_slot=-1)


def _no_pbo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the CSCV PBO, which costs ~0.2 s per recorded trial.

    Only for tests about *which* formulas the search scores highly; PBO is a
    property of the finished ledger and plays no part in the search itself.
    """
    monkeypatch.setattr(TrialLedger, "pbo", lambda self, **_: float("nan"))


class TestEvaluatorRowAlignment:
    """Bug: formulas were scored against a target in a different row order.

    The evaluator holds its targets in ``(time, entity)`` order while
    ``compile_population`` (default ``sort=True``) returns rows in
    ``(entity, time)`` order, and the two were paired by position. Every
    formula was therefore correlated with a shuffled target.
    """

    def test_genome_scores_match_the_feature_computed_directly(self) -> None:
        ev = PanelEvaluator(_planted().lazy(), ctx=CTX, target="fwd", seed=0)
        direct = ev._lf.select(pl.col("signal").rank().over("time")).collect()
        want = ev.score_matrix(direct.to_numpy())[0][0].score
        got = ev.evaluate([_genome("cs_rank", 1)])[0].score
        assert want > 0.2, "the planted signal should be strong on its own"
        assert got == pytest.approx(want, abs=1e-12)

    @pytest.mark.parametrize(
        "genome",
        [
            _genome("cs_rank", 1),
            _genome("ts_mean", 1, 0),  # ts_mean(signal, 5)
            _genome("ts_delta", 0, 0),  # ts_delta(value, 1)
        ],
        ids=["cs_rank", "ts_mean", "ts_delta"],
    )
    def test_score_does_not_depend_on_the_compiler_row_order(
        self, genome: Genome
    ) -> None:
        frame = _planted()
        default = PanelEvaluator(frame.lazy(), ctx=CTX, target="fwd", seed=0)
        unsorted = PanelEvaluator(
            frame.lazy(),
            ctx=CTX,
            target="fwd",
            seed=0,
            compile_fn=functools.partial(compile_population, sort=False),
        )
        shuffled = PanelEvaluator(
            frame.sample(fraction=1.0, shuffle=True, seed=11).lazy(),
            ctx=CTX,
            target="fwd",
            seed=0,
        )
        scores = [e.evaluate([genome])[0] for e in (default, unsorted, shuffled)]
        for other in scores[1:]:
            assert other.score == scores[0].score
            np.testing.assert_array_equal(other.per_case, scores[0].per_case)

    def test_search_finds_the_planted_signal_with_default_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_pbo(monkeypatch)
        frame = _planted()
        cfg = evolve.EvolveConfig(
            population=16, generations=2, n_islands=1, max_library=5, seed=0
        )
        control = frame.with_columns(
            pl.col("fwd").shuffle(seed=5).alias("fwd")
        )  # same target values, every relationship to the features destroyed

        def best(panel: pl.DataFrame) -> float:
            res = evolve.evolve_features(
                panel, target="fwd", entity="entity", time="time", config=cfg
            )
            return float(res.ledger["best_score"])

        found, null = best(frame), best(control)
        assert found > 0.25, f"planted signal not found: best IC {found:.3f}"
        assert found > 3.0 * max(null, 0.0), (found, null)

    def test_search_is_invariant_to_input_row_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_pbo(monkeypatch)
        frame = _planted()
        cfg = evolve.EvolveConfig(
            population=12, generations=2, n_islands=1, max_library=5, seed=2
        )
        runs = [
            evolve.evolve_features(
                panel, target="fwd", entity="entity", time="time", config=cfg
            )
            for panel in (frame, frame.sample(fraction=1.0, shuffle=True, seed=3))
        ]
        assert runs[0].to_frame().equals(runs[1].to_frame())
        assert runs[0].ledger.keys() == runs[1].ledger.keys()
        for key, value in runs[0].ledger.items():
            other = runs[1].ledger[key]
            if isinstance(value, float) and np.isnan(value):
                assert isinstance(other, float) and np.isnan(other), key
            else:
                assert other == value, key
