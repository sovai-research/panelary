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
from panelary.evolve._fitness import PanelEvaluator, rank_ic_series  # noqa: E402
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


class TestHoldoutCheck:
    """Bug: the held-out check reported failure because of a swallowed error.

    A forward label is null on its last ``horizon`` dates, and those dates are
    exactly the tail of the holdout window. ``rank_ic_series`` built its date
    groups *before* dropping the null targets, so trailing dates became empty
    groups and ``np.add.reduceat`` raised ``IndexError``. A bare
    ``except Exception`` turned that into a "HOLDOUT FAILED" verdict.
    """

    def test_rank_ic_series_skips_dates_without_a_target(self) -> None:
        rng = np.random.default_rng(0)
        n_dates, n_ent = 12, 20
        t = np.repeat(np.arange(n_dates), n_ent)
        x = rng.standard_normal(t.size)
        y = x + rng.standard_normal(t.size)
        y_gappy = y.copy()
        y_gappy[t >= n_dates - 2] = np.nan  # the forward label's null tail
        y_gappy[t == 4] = np.nan  # and a date in the middle with no label
        want = rank_ic_series(x, y, by_time=t)
        got = rank_ic_series(x, y_gappy, by_time=t)
        assert got.shape == (n_dates,)
        assert np.isnan(got[[4, n_dates - 2, n_dates - 1]]).all()
        keep = np.setdiff1d(np.arange(n_dates), [4, n_dates - 2, n_dates - 1])
        np.testing.assert_allclose(got[keep], want[keep], rtol=0, atol=1e-12)
        assert evolve.rank_ic(x, y_gappy, by_time=t) == pytest.approx(
            float(np.mean(want[keep])), abs=1e-12
        )

    def test_holdout_is_scored_when_the_label_has_a_null_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_pbo(monkeypatch)
        frame = _planted()
        assert frame["fwd"].null_count() > 0  # the realistic case
        cfg = evolve.EvolveConfig(
            population=16, generations=2, n_islands=1, max_library=5, seed=0
        )
        res = evolve.evolve_features(
            frame, target="fwd", entity="entity", time="time", config=cfg
        )
        diag = res.diagnostics
        assert "FAILED" not in diag["verdict"], diag["verdict"]
        assert "holdout_error" not in diag
        assert diag["n_holdout_candidates"] >= 5
        assert np.isfinite(diag["holdout_mean_ic"])

    def _search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[object, pl.LazyFrame, pl.LazyFrame | None]:
        from panelary.evolve._search import _time_split

        _no_pbo(monkeypatch)
        frame = _planted()
        cfg = evolve.EvolveConfig(
            population=16, generations=2, n_islands=1, max_library=5, seed=0
        )
        res = evolve.evolve_features(
            frame, target="fwd", entity="entity", time="time", config=cfg
        )
        search_lf, holdout_lf = _time_split(
            frame.lazy().sort(["entity", "time"]), "time", cfg.holdout_frac
        )
        return res, search_lf, holdout_lf

    def test_a_programming_error_in_the_holdout_check_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import panelary.evolve._fitness as fitness
        from panelary.evolve._search import _holdout_diagnostics

        res, search_lf, holdout_lf = self._search(monkeypatch)

        def broken(*_a: object, **_k: object) -> float:
            raise IndexError("index 659 out-of-bounds in add.reduceat [0, 659)")

        monkeypatch.setattr(fitness, "rank_ic_series", broken)
        monkeypatch.setattr(fitness, "rank_ic", broken)
        with pytest.raises(IndexError):
            _holdout_diagnostics(
                res.archive,
                holdout_lf,
                res.context,
                target="fwd",
                seed=0,
                search_lf=search_lf,
            )

    def test_a_polars_failure_on_the_holdout_is_recorded_and_warned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import panelary.evolve._compile as compiler
        from panelary.evolve._search import _holdout_diagnostics

        res, search_lf, holdout_lf = self._search(monkeypatch)

        def broken(*_a: object, **_k: object) -> None:
            raise pl.exceptions.ComputeError("formula blew up on the holdout")

        monkeypatch.setattr(compiler, "compile_population", broken)
        with pytest.warns(RuntimeWarning, match="holdout"):
            diag = _holdout_diagnostics(
                res.archive,
                holdout_lf,
                res.context,
                target="fwd",
                seed=0,
                search_lf=search_lf,
            )
        assert diag["verdict"].startswith("HOLDOUT FAILED")
        assert "ComputeError" in diag["holdout_error"]


class _Archive:
    """The one method `_holdout_diagnostics` reads from an archive."""

    def __init__(self, elites: list[tuple[Genome, float, tuple[float, ...]]]) -> None:
        self._elites = elites

    def elites(self) -> list[tuple[Genome, float, tuple[float, ...]]]:
        return list(self._elites)


class TestHoldoutOrientation:
    """Bug: the search diagnostic reported "NO SIGNAL" on a real signal.

    The evaluator orients every candidate on its training dates, so
    ``-signal`` earns the same search score as ``signal``. The holdout check
    scored the *raw* IC, putting every negatively signed elite on the wrong
    side of the in-sample/held-out scatter and destroying its correlation. It
    also compiled the elites on the holdout dates alone, so any formula with a
    window longer than the holdout (``ts_zscore(x, 60)`` on a 24-date holdout)
    was all-null there and silently dropped.
    """

    GENOMES = (
        Genome(genes=(Gene("cs_rank", (1,), 0),), n_base=2),  # rank(signal)
        Genome(genes=(Gene("neg", (1,), 0),), n_base=2),  # -signal
        Genome(  # -(value + signal)
            genes=(Gene("add_score", (0, 1), 0), Gene("neg", (2,), 0)), n_base=2
        ),
        Genome(genes=(Gene("ts_mean", (1,), 0),), n_base=2),  # ts_mean(signal, 5)
        Genome(genes=(Gene("neg", (0,), 0),), n_base=2),  # -value
        Genome(genes=(Gene("ts_zscore", (1,), 4),), n_base=2),  # ts_zscore(sig, 60)
        Genome(genes=(Gene("ts_delta", (1,), 0),), n_base=2),  # ts_delta(sig, 1)
    )

    def test_held_out_scores_are_oriented_like_the_search(self) -> None:
        from panelary.evolve._search import _holdout_diagnostics, _time_split

        frame = _planted()
        search_lf, holdout_lf = _time_split(
            frame.lazy().sort(["entity", "time"]), "time", 0.2
        )
        assert holdout_lf is not None
        ev = PanelEvaluator(search_lf, ctx=CTX, target="fwd", seed=0)
        scores = [r.score for r in ev.evaluate(list(self.GENOMES))]
        archive = _Archive(
            [(g, s, (0.0, 0.0, 0.0)) for g, s in zip(self.GENOMES, scores, strict=True)]
        )
        diag = _holdout_diagnostics(
            archive, holdout_lf, CTX, target="fwd", seed=0, search_lf=search_lf
        )
        assert diag["n_holdout_candidates"] == len(self.GENOMES)
        held = diag["scatter"][:, 1]
        # Every one of these formulas carries the planted signal, in one sign
        # or the other; oriented as the search oriented them, all generalise.
        assert (held > 0.05).all(), diag["scatter"]
        assert diag["verdict"].startswith("SIGNAL"), diag["verdict"]


class TestSummaryVerdictOnARealSignal:
    """Bug: ``TrialLedger.summary`` failed a genuine signal.

    ``expected_max_under_null`` was ``sqrt(V) * f(N_hat)`` with ``V`` the
    variance of the recorded scores *across trials*. That is the null variance
    only if no trial has skill. A search that finds a signal breeds variants of
    it, so ``V`` measured the gap between signal-carrying and signal-free
    formulas and the bar rose with the signal: the external run failed a
    planted signal scoring IC 0.53 against a bar of 0.68. The DSR had the
    converse defect: it deflated a per-case Sharpe against the variance of mean
    ICs, a smaller scale, so it was lenient. Both now use decoys scored through
    the identical path.
    """

    N_CASES = 120

    def _ledger(self, rng: np.random.Generator, *, skilled_every: int) -> TrialLedger:
        ledger = TrialLedger(seed=0)
        for i in range(300):
            skill = 0.4 if skilled_every and i % skilled_every == 0 else 0.0
            series = skill + 0.1 * rng.standard_normal(self.N_CASES)
            ledger.record(i, float(series.mean()), series)
        return ledger

    def test_the_null_is_not_inflated_by_signal_carrying_trials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TrialLedger, "pbo", lambda self, **_: 0.0)
        rng = np.random.default_rng(0)
        ledger = self._ledger(rng, skilled_every=3)
        decoys = 0.1 * rng.standard_normal((20, self.N_CASES))

        legacy = ledger.summary()  # no decoys: the cross-trial fallback
        assert legacy["verdict"] == "FAIL"
        assert "expected max under the null" in legacy["reason"]
        assert legacy["expected_max_under_null"] > 0.4  # the signal raised the bar

        fixed = ledger.summary(null_scores=decoys.mean(axis=1), null_series=decoys)
        assert fixed["null_source"] == "20 decoys"
        assert fixed["expected_max_under_null"] < 0.05
        assert fixed["deflated_sharpe"] > 0.95
        assert fixed["verdict"] == "PASS", fixed["reason"]

    def test_decoys_do_not_let_noise_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TrialLedger, "pbo", lambda self, **_: 0.0)
        rng = np.random.default_rng(1)
        ledger = self._ledger(rng, skilled_every=0)
        decoys = 0.1 * rng.standard_normal((20, self.N_CASES))
        out = ledger.summary(null_scores=decoys.mean(axis=1), null_series=decoys)
        assert out["verdict"] == "FAIL", out["reason"]
        assert out["deflated_sharpe"] < 0.95

    def test_search_passes_a_planted_signal_and_fails_its_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # PBO is pinned to a pass so that the verdict turns on the null alone:
        # the control must fail on E[max] / DSR, without PBO's help.
        monkeypatch.setattr(TrialLedger, "pbo", lambda self, **_: 0.0)
        frame = _planted()
        # 240 trials: enough that the search breeds variants of the signal,
        # which is what inflated the cross-trial variance.
        cfg = evolve.EvolveConfig(
            population=40, generations=3, n_islands=2, max_library=5, seed=0
        )
        control = frame.with_columns(pl.col("fwd").shuffle(seed=5).alias("fwd"))
        found, null = (
            evolve.evolve_features(
                panel, target="fwd", entity="entity", time="time", config=cfg
            ).ledger
            for panel in (frame, control)
        )
        assert found["verdict"] == "PASS", found["reason"]
        assert found["best_score"] > found["expected_max_under_null"]
        assert null["verdict"] == "FAIL", null["reason"]

    @pytest.mark.slow
    def test_search_verdicts_with_the_real_pbo(self) -> None:
        frame = _planted()
        cfg = evolve.EvolveConfig(
            population=20, generations=2, n_islands=1, max_library=5, seed=0
        )
        control = frame.with_columns(pl.col("fwd").shuffle(seed=5).alias("fwd"))
        found, null = (
            evolve.evolve_features(
                panel, target="fwd", entity="entity", time="time", config=cfg
            )
            for panel in (frame, control)
        )
        assert found.ledger["verdict"] == "PASS", found.ledger["reason"]
        assert null.ledger["verdict"] != "PASS", null.ledger["reason"]
        assert "null calibrated on" in found.summary()
