# `panelary/evolve/` — build contract

> **Status: implemented** (moved to `plans/done/` 2026-09-28). Every interface below exists in
> `panelary/evolve/` (`_ops`, `_compile`, `_select`, `_genome`, `_fitness`, `_honest`, plus the
> orchestrator's `_search`), covered by `tests/test_evolve.py` and `tests/test_evolve_leakage.py`;
> merged in `065a813` ("evolutionary alpha-factor mining").

Evolutionary / genetic **alpha-factor mining** for panel data. Issued to the
implementation agents 2026-09-09.

Pure **numpy + polars**. No scipy, sklearn, statsmodels, numba, Rust in the
import path. Optional accelerators (lightgbm) go behind
`panelary._internal._deps.require` and must have a pure-numpy fallback that is the
**default**.

Reuse, do not reinvent:
`panelary.core.model_selection` (`PurgedKFold`, `CombinatorialPurgedCV`,
`deflated_sharpe_ratio`, `probability_of_backtest_overfitting`, `_norm_cdf`,
`_norm_ppf`), `panelary.validation._selection_stats`
(`expected_maximum_sharpe`, `holm_bonferroni`, `benjamini_hochberg`,
`benjamini_yekutieli`, `romano_wolf`), `panelary.factor`
(`ic`, `ic_summary`, `forward_return`, `orthogonalize`, `portfolio_sort`),
`panelary.econ._common` (`ols`, `pinv_sym`, `norm_ppf`, `norm_cdf`).

## Why this module exists

Surveyed ~40 symbolic-regression / GP libraries and 9 alpha-mining repos:
**zero** implement purging, embargo, or any multiple-testing control, and
**zero** are panel-aware. Meanwhile the independent AlphaEval benchmark
(KDD 2026) finds the entire published field scores 0.017-0.041 predictive power
against a **random-formula baseline of 0.009**, and every published method is
*less diverse* than random formulas. So we do not compete on search cleverness.
We compete on the two things nobody has: **leak-safety by construction** and
**honest deflation**.

## Hard invariants (every function)

1. **Prefix invariance.** `f(x[:T])[t] == f(x[:T+k])[t]` for all `t <= T`. No
   quantity may depend on `len(x)` — not window sizes, not thresholds, not
   normalisation constants. This is also the definition of the causality test
   in `_honest.assert_causal`.
2. **Determinism.** No unseeded RNG. Any RNG takes an explicit `seed: int`;
   build a `np.random.default_rng(seed)` internally.
3. **float64 everywhere.** Upcast Float32 polars columns before accumulating.
4. `np.linalg.solve(A, b[..., None])[..., 0]` — never `solve(A, b)` (NumPy 2.0
   silently mis-solves when `p == batch size`). Relevant in `_fitness.AlphaPool`.
5. Never `rolling_map` (measured 249x penalty). Native rolling expressions only;
   `map_batches` per group is the escape hatch.
6. Every generated feature is leak-safe **by construction**: the grammar admits
   an operator only if its `FeatureSpec.leakage_safe` is True, and `.over()` is
   applied by the compiler, never hand-written into an operator body.

## Measured facts that dictate the design (verified on this repo, polars 1.44.1)

| Fact | Measurement | Consequence |
|---|---|---|
| Nesting `.over(time)` around `.over(entity)` returns **all nulls, silently** | `x.rolling_mean(2).over("sym").rank().over("date")` -> 0/8 non-null; staged -> 6/8 | Partition-scope switches **must** be separate materialisation stages. Forces the DAG-layered compiler. |
| Polars CSE does **not** dedupe shared subtrees across a population | 300 genomes sharing one subtree: inline 0.895 s vs hoisted 0.082 s | **10.9x**. Hand-rolled subexpression memoisation is mandatory. |
| Expression planning time is **quadratic in depth** | depth 1600 = 1.62 s before touching data (polars#16224, open) | Hard depth cap; emit shallow layers, never one deep expression. |
| Batched shallow expressions are linear and cheap | 2000 depth-3 exprs / 200k rows = 0.57 s (0.28 ms/expr) | Batch the whole population into one `select` per layer. |
| Cross-sectional ops cost ~20x a time-series op | `rank().over(time)` 18.4 ms/expr vs `rolling_mean().over(entity)` 0.85 ms/expr | Budget xs ops separately; `Op.cost` reflects it. |
| The registry has **no** ts series->series operators | 56 specs: ts=42 (all aggregators), xs=6, panel=3, ts series->series = **0** | `_ops.py` fills exactly this gap. |
| A 100,000-trial search on pure noise yields best in-sample Sharpe ~4.39 | `expected_maximum_sharpe`, verified against true E[max] of N normals | Reporting a raw in-sample score is a fabrication. `_honest.py` is not optional. |

## File ownership — touch ONLY your file(s)

| File | Owner | Status |
|---|---|---|
| `_types.py` | orchestrator | **done — do not modify** |
| `_ops.py` | Agent A | operator vocabulary |
| `_compile.py` | Agent B | DAG compiler (perf + correctness critical) |
| `_select.py` | Agent C | lexicase, NSGA-II, MAP-Elites archive |
| `_genome.py` | Agent D | representation + variation |
| `_fitness.py` | Agent E | leak-safe evaluation |
| `_honest.py` | Agent F | trial ledger, deflation, causality |
| `_search.py`, `__init__.py` | orchestrator | do not create |
| `tests/test_evolve_*.py` | orchestrator | do not create |

## Interfaces you may assume exist

### `_ops.py` (Agent A)
```python
OPS: dict[str, Op]                                  # name -> Op
def ops_by_kind(kind: OpKind) -> tuple[Op, ...]: ...
def ops_producing(unit: Unit) -> tuple[Op, ...]: ...
def default_grammar(units: Sequence[Unit]) -> tuple[Op, ...]: ...
# Op.build(*child_exprs, param) -> pl.Expr, and NEVER calls .over() itself.
```

### `_compile.py` (Agent B)
```python
def compile_population(genomes, ctx: EvalContext, lf: pl.LazyFrame
                       ) -> tuple[pl.LazyFrame, list[str]]: ...
    # one with_columns per (layer, scope); returns output col name per genome
def dag_stats(genomes, ctx) -> dict[str, int | float]: ...
def active_nodes(genome: Genome) -> frozenset[int]: ...
def depth(genome: Genome) -> int: ...
def validate(genome: Genome, ctx: EvalContext, *, max_depth: int) -> None: ...
def semantic_key(values: np.ndarray, *, seed: int = 0) -> int: ...
```

### `_select.py` (Agent C)
```python
def eps_lexicase(scores: np.ndarray, n_select: int, *, seed: int,
                 downsample: float = 0.1) -> np.ndarray: ...   # (n_pop, n_cases), higher better
def fast_non_dominated_sort(objectives: np.ndarray) -> list[np.ndarray]: ...
def crowding_distance(objectives: np.ndarray) -> np.ndarray: ...
class Archive:  # MAP-Elites, grid or CVT, annealed thresholds
    def add(self, genome, fitness: float, descriptors: Descriptors) -> bool: ...
    def elites(self) -> list[tuple[Genome, float, Descriptors]]: ...
```

### `_genome.py` (Agent D)
```python
def random_genome(ctx, grammar, *, n_genes: int, max_depth: int, seed: int) -> Genome: ...
def ramped_population(ctx, grammar, *, size: int, seed: int, **kw) -> list[Genome]: ...
def mutate(genome, grammar, ctx, *, seed: int, **rates) -> Genome: ...
def subgraph_crossover(a, b, grammar, ctx, *, seed: int) -> Genome: ...
def canonical_form(genome: Genome) -> Genome: ...
def structural_key(genome: Genome) -> int: ...
def complexity(genome: Genome) -> int: ...      # ACTIVE genes only
def to_infix(genome: Genome, ctx: EvalContext) -> str: ...
```

### `_fitness.py` (Agent E)
```python
def rank_ic(feature, fwd_ret, *, by_time) -> float: ...
def numerai_corr(pred: np.ndarray, target: np.ndarray) -> float: ...  # CORR20V2, MIT port
class AlphaPool:      # loss = w'Mw - 2w'c + 1 ; incremental O(K) row update
    def marginal_contribution(self, ic_vec: np.ndarray) -> float: ...
class PanelEvaluator:  # implements _types.Evaluator
    def evaluate(self, genomes) -> list[FitnessResult]: ...
def descriptors(...) -> Descriptors: ...
def null_threshold(real: np.ndarray, noise: np.ndarray, *, q: float = 1.0) -> float: ...
```

### `_honest.py` (Agent F)
```python
class TrialLedger:
    def record(self, key: int, score: float, series: np.ndarray | None = None) -> None: ...
    def implied_independent_trials(self) -> float: ...   # Bailey&LdP App.3 Eq.8-9
    def summary(self) -> dict[str, float | str]: ...     # M, N_hat, rho_bar, V, DSR, PBO, verdict
def minimum_backtest_length(n_trials: int, target_sharpe: float) -> float: ...
def assert_causal(expr: pl.Expr, df: pl.DataFrame, *, entity: str, time: str) -> bool: ...
def haircut_sharpe_ratio(sharpe: float, n_trials: int, *, rho: float = 0.2,
                         n_sim: int = 2000, seed: int = 0) -> dict[str, float]: ...
def cross_sectional_bootstrap(...) -> dict[str, float]: ...
def search_diagnostics(in_sample: np.ndarray, held_out: np.ndarray) -> dict: ...
```

## Licence discipline

Clean-room only. Definitions may be derived from **MIT/BSD/Apache** sources —
Qlib (MIT), wukan1986/polars_ta (MIT), expr_codegen (BSD-3), gplearn (BSD-3),
numerai-tools (MIT), AlphaEval (MIT) — and from published papers.

**Never copy from**: alphagen, AlphaForge, AlphaPROBE, QuantaAlpha, AlphaAgent,
alpha-gfn, Alpha2, gpquant, AlphaSAGE (1-byte empty LICENSE), yli188/Alpha101
— all have **no LICENSE file, i.e. all rights reserved**. Nor from GPL/AGPL:
FEAT, Brush, EvoGP, SRBench, pypbo, techfactor. Nor `vectorbt` (Commons Clause).
Record `source` and `license` on every `FeatureSpec`, per
`panelary/catch22.py`, which is the clean-room precedent.
