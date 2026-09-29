# `panelary/network/` — build contract: as-of graphs, network-lag features, spatial statistics, network autoregression, lead–lag networks

> **Cross-plan decisions:** [`00-cross-plan-coordination.md`](00-cross-plan-coordination.md) overrides this plan where they differ (shared refit schedule, dense pivot, numba helper, special functions, ownership of shared files and `_internal` modules).

> **Status (2026-09-29): not started — plan only.**
> Merge-gated by the AGENTS.md caller rule. See §0 and §15: no engine caller exists
> today, so only M1 has a proposed caller and nothing past M1 should be built until
> one is named.

Pure **numpy + polars**. The optional `fast` extra (numba) gives the **same bits**,
10–40× faster. No scipy, sklearn or networkx in the import path. Oracles (`networkx`,
PySAL `esda`/`libpysal`, `scipy.sparse.csgraph`) are **test-only** and sit behind
`pytest.importorskip`. This plan covers what the roadmap calls `pairwise/`
(`rolling_corr_network`, per-entity centrality, the fixed trailing `leadlag`; see
`docs/roadmap.md` "Pairwise & networks"). It lives in `panelary/network/` because spatial
statistics and as-of relationship graphs are not "pairwise".

---

## 0. The gate: read this before writing code

AGENTS.md says: *"No new public surface without a named caller in the engine, and a
test in the engine that exercises it."* I checked this on 2026-09-29. `truepoint/src`
does not import `panelary` anywhere. `truepoint/src/truepoint/_firewall.py` forbids
`diagnose/` from importing it (`FORBIDDEN_PRODUCT`). The caller directory that
AGENTS.md names, `truepoint/src/truepoint/quant/`, does not exist yet. So:

* **M1** (as-of `EdgeTable`, knowledge-time validation and resolution, the `network_lag`
  kernel) has a proposed caller (§15). It merges only once that truepoint test exists.
* **M2–M7** are fully designed so the work is not lost, but they are **parked**. Each one
  needs its own named caller before anyone builds it.

---

## 1. Why this exists

### 1.1 The published effect is large, persistent and entirely cross-entity

Returns of economically linked firms predict each other with a lag. No per-entity
feature library can express that.

* **Customer–supplier links.** Cohen & Frazzini (2008): lagged customer returns predict
  supplier returns. A long–short strategy earns "over 150 bp/month"†.
* **Supply-chain industries.** Menzly & Ozbas (2010): returns of supplier and customer
  industries cross-predict each other, which fits gradual information diffusion across
  segmented investor bases.
* **Technology links.** Lee, Sun, Wang & Zhang (2019): the returns of technology-linked
  firms (patent-space proximity) predict focal-firm returns.
* **Shared analyst coverage.** Ali & Hirshleifer (2020): connected-firm momentum through
  shared analysts subsumes most of the other momentum-spillover effects (industry,
  customer/supplier, geographic, technological)†.
* **Geography.** Parsons, Sabbatucci & Titman (2020): returns of firms headquartered in
  the same city lead–lag each other.

### 1.2 The caveat, and what our default does about it (measured)

Burt & Hrdlicka (2021, JFQA 56(8), 2634–2658, verified) show that this predictability
can arise when both firms have their own momentum and their returns are contemporaneously
correlated. Momentum commonality and news contribute roughly equally at the 1-month
horizon, and commonality alone explains the longer horizons. So a raw peer signal can
"predict" when **no information travels across the link at all**.

I simulated this on this machine (1,200 firms in 200 link groups, 5 peers each, 480
months). Expected returns are a persistent group component plus a persistent own
component, and ε is i.i.d. The "news" arm adds `0.10 × peers' last-month ε`. The
statistic is the mean per-date IC with the next-month return (t in parentheses):

| Peer feature at `t` | Commonality only (no transfer) | Genuine news diffusion |
|---|---|---|
| raw `W r_t` | +0.0061 (t = 4.5) | +0.0489 (t = 36.6) |
| residualised on own `r_t` | +0.0061 (4.6) | +0.0489 (36.5) |
| … and on own 12-month mean | +0.0059 (4.4) | +0.0487 (36.4) |
| residualised on **peers'** 12-month mean `W·mom` | **+0.0022 (1.6)** | **+0.0440 (33.0)** |

This result changes the design. Orthogonalising the peer signal on the focal firm's own
lagged signal does **not** remove the commonality channel. The peer average is simply a
better estimate of the shared expected return than the firm's own noisy return.
Residualising on the **peers' own trailing mean** removes about two-thirds of the
spurious IC, and it keeps 90% of a genuine diffusion effect. `network_lag` therefore
defaults to `orthogonalize_own=True` **with commonality controls**: own lag, own trailing
mean, and peer trailing mean, all over a calendar window, default `"1y"` (§7.2). The
simulation becomes the regression test `test_network_synth.py::test_commonality_only_is_suppressed`.
This is our DGP, not Burt–Hrdlicka's data. The docs will not claim that we reproduce
their decomposition.

### 1.3 Every naive implementation leaks, and it leaks through the graph, not the values

* **Backfilled relationship databases.** Link databases record a relationship's *start
  date*, not when it became knowable. Using `valid_from` as the knowledge time is look-ahead.
* **Classifications.** Current GICS or current headquarters applied to history.
* **Survivorship.** Weights row-standardised over neighbours that survive to the end of
  the sample.
* **Graphs estimated from the same window they are used in.** A correlation graph built
  from returns up to and including `t` and applied at `t`.
* **Contemporaneous spatial lags.** `W y_t` used to predict `y_t`.

`assert_no_lookahead` perturbs future *values*. It cannot see a graph that was known too
early, so this module adds a **knowledge-time invariance** instrument (§4, §9).

### 1.4 Measured facts that set the design

These are measurements, not estimates. Machine: Apple M5 Pro (15 cores). Software:
numpy 2.5.3 (Accelerate), polars 1.44.2, scipy 1.18.1, numba 0.67.0, networkx 3.6.1.
PySAL is not installed. Scripts are in the session scratchpad and will become
`benchmarks/bench_network.py`.

| Fact | Measurement | Consequence |
|---|---|---|
| 5,000 entities × 5,000 dates × 50 edges/node | 1.25 × 10⁹ edge-dates = **20 GB** at 16 B/edge | Never materialise per-date graphs. Resolve into **epochs** (piecewise-constant snapshots) and stream them. |
| Single-date SpMV, N = 5,000, E = 250k | `bincount` 1.14 ms · CSR `reduceat` 0.30 ms · numba sequential 0.12 ms · scipy CSR 0.09 ms | Per-date SpMV is cheap. The real design question is batching across dates. |
| Static graph × 5,000 dates (SpMM) | gather + `reduceat` **8.48 s** · ELL/slot row-gather **1.90 s** · dense GEMM 0.89 s (200 MB W) · scipy 0.24 s · numba parallel **0.07 s** | The numpy default is slot/ELL. numba is the fast path. |
| Heavy-tailed degrees (Zipf 1.8, max 2,000, E = 57k) × 5,000 dates | slot 0.61 s · scipy 0.06 s | Slot decomposition survives hubs because its Python-level loop is over slots, not rows. |
| Graph changes every date, batched 20 dates with global offsets | `bincount` **1.03 ms/date** (5.1 s / 5,000 dates) · `reduceat` 0.22 · numba 0.13 · numba parallel **0.024** · polars join + `group_by` 1.93 | For time-varying graphs: `bincount` in numpy, numba when available. |
| **Summation order** (weights spanning 10⁻⁸…10⁸) | slot, `bincount` and numba sequential are **bitwise equal** to a left-to-right loop. `reduceat` is off by up to 1.8 × 10⁻¹² relative, scipy by 1.2 × 10⁻¹¹. | One canonical order (invariant 3). Installing `fast` can never change a number. |
| **GEMM shape dependence** | `A @ X[:, :1]` ≠ `(A @ X)[:, :1]` bitwise. 3- and 700-column blocks match 1,000. | A date-blocked GEMM whose last block width depends on T breaks prefix invariance. Banned (invariant 4). |
| PageRank, N = 5,000, E = 250k | 13 iterations to L1 < 10⁻¹², 3.9 ms · networkx 92 ms | Power iteration, numpy. |
| Eigenvector centrality, dense N = 500 correlation | `eigh` 9.8 ms · power iteration 0.07 ms (10 iterations, agrees to 7.5 × 10⁻¹⁵) · batched over 64 dates 0.16 ms/date | Power iteration with an `eigh` fallback. |
| MST, N = 500 dense | numpy Prim O(N²) **1.1 ms** · scipy 17.9 ms · networkx 195 ms · weights agree to 1.1 × 10⁻¹³ | Prim, numpy. |
| TMFG, N = 500 | numpy O(N²) prototype, 29 ms, 1,494 = 3N − 6 edges | Ships. PMFG, at O(N³), does not. |
| Correlation matrix N = 500, W = 252 / Lévy-area matrix N = 500, W = 60 | 0.29 / 0.26 ms per date | Daily dense graphs at N = 500 are cheap. |
| Jacobi-PCG `(I + λL)z = x`, N = 5,000, E = 500k | 14 iterations to 10⁻¹⁰, 5.9 ms | Laplacian smoothing is cheap. |
| Global Moran's I, N = 5,000, k = 50 | 1.06 ms/date batched | Default diagnostic. |
| LISA conditional permutation, B = 999, N = 5,000, k = 50 | **0.32 s/date** = 27 min over 5,000 dates | Analytic moments are the default. Permutation is opt-in and strided. |
| Hermitian-RW embedding, N = 500 | complex `eigh` 89 ms · **real `eigh` of S̃ᵀS̃ 11.9 ms**, same projector to 4.6 × 10⁻¹⁵ · subspace iteration does not converge at eigen-ratio 0.983 | Real arithmetic, full `eigh` (§7.9). |
| `econ.pesaran_cd`, N = 500, T = 252, 5% missing | existing pair loop **1.78 s** · masked 4-GEMM **7.8 ms** · \|ΔCD\| = 1.0 × 10⁻¹¹ on CD = 1054.44 | Upgrade in place (§7.10). |

---

## 2. What already exists, and what this must NOT duplicate (verified)

| File : symbol | What it is | How this plan uses it |
|---|---|---|
| `panelary/core/asof.py:asof_join` (+ `_common_time_dtype`, `_time_family`) | Bitemporal vintage join. A vintage is eligible at `max(knowledge_time + lag, event_time)`. Fail-closed dtype rules. | **Reuse** for as-of node attributes (coordinates, characteristics, sector shares) and for `GroupGraph` membership. **Reuse** its dtype checks for edge time columns. Edge resolution is new because it returns the *set* of active edges, not the latest vintage, but it applies the same eligibility rule. |
| `panelary/core/_calendar.py:validate_duration`, `shift_forward`, `BusinessDays` | The single duration vocabulary. | `knowledge_lag`, `rebuild=` and `commonality_window` all take a `Duration`. Do **not** write a second parser. |
| `panelary/econ/_connectedness.py:connectedness`, `rolling_connectedness`, `ConnectednessFeatures` | Diebold–Yilmaz generalised-FEVD network. The emitted to/from/net values are weighted in/out strengths. | **Do not reimplement VAR/FEVD.** `EdgeTable.from_matrix(result.table, …)` ingests a DY table, so real centralities of the DY network come from §7.7. Flag for the econ owner: `ConnectednessFeatures.panel_safe = True` even though it mixes entities, and `econ/__init__.py` promises "per-entity centrality", but only to/from/net are emitted. |
| `panelary/econ/_panel.py:pesaran_cd` | CD test using an O(N²) **Python** loop over pairs (measured 1.78 s). | **Upgrade in place:** a masked-GEMM kernel plus `bias_corrected=` (CD*, §7.10). |
| `panelary/econ/_panel.py:CrossSectionalAverages`, `cce_mg`, `cips` | Per-date cross-sectional means (CCE factor proxies). | The complete-graph special case of `network_lag`. A test asserts agreement. Do not add a second CSA. Its semantics are the BHP stage-1 de-factoring. |
| `panelary/econ/_hdfe.py:_hdfe_vcov`, `_psd_clip`, `_VCOV_KINDS` | Classical, HC1, one- and two-way cluster, Driscoll–Kraay. | Add `vcov="conley"`. hdfe keeps the sandwich, dof and PSD clip. `network/_inference.py` supplies only the **meat**. |
| `panelary/econ/_famamacbeth.py:fama_macbeth` | Per-date OLS with Newey–West on λ. | SLX is a **recipe** (peer columns passed as `x`), not a new function. No Conley option (§7.10). |
| `panelary/econ/_common.py` | `ols`, `pinv_sym`, `norm_cdf`/`norm_ppf`, `factorize`, `group_sum`, `newey_west_lrv`, `auto_bandwidth` | Reuse everywhere. |
| `panelary/depend/_lag.py` | Lag convention `lag=k>0` pairs `x[t−k]` with `y[t]`. `adjust_pvalues`. `nonlinear_ccf` works on one *column* pair, aggregated over entities. | Adopt the same lag sign convention for entity-pair lead–lag. Reuse `adjust_pvalues`. `nonlinear_ccf` is **not** entity-pair lead–lag, so there is no overlap. |
| `panelary/depend/_matrix.py:dependence_matrix`, `to_distance`, `psd_repair` | p × p matrices over *columns*. Angular distance. | `to_distance(kind="angular")` for MST distances. `psd_repair` for correlation input. |
| `panelary/depend/_engine.py:rows_chunked`, `depend/_kernels.py:get_kernel`, `depend/_null.py:common_time_indices` | Row-batched dependence kernels and the panel null. | Nonlinear ccf for lead–lag (Kendall, dCor, ξ), plus optional edge significance. |
| `panelary/depend/_frame.py:extract` → `PanelArrays.dense` | Panel → (N, T) NaN layout with label-sorted entities. | **The** extraction path. Its codes are label-monotone (invariant 5). |
| `panelary/reduce/_common.py:top_eigenvectors`, `fix_signs` | `eigh` with symmetrisation, and a sign convention. | Centrality fallback and sign pinning. |
| `panelary/evolve/_select.py:_kmeans_pp_init`, `_cvt_centroids` | Private numpy k-means++ and Lloyd. | Hermitian-RW clustering. Proposal for the orchestrator: promote to `_internal/_kmeans.py`. Do not write a third k-means. |
| `panelary/namespaces/_neutralize_kernel.py:cross_section_residuals`, `factor/neutralize.py:orthogonalize` | Per-date OLS residuals and Gram–Schmidt. | **Parity oracles** for `orthogonalize_own`. The fast path is closed-form batched per-date OLS (§7.2). |
| `panelary/quality/*` (`CheckResult`, `ValidationReport`, `to_evidence`), `preprocessing/_base.py:LeakageWarning` | The data-integrity gate. | `EdgeTable.validate()` returns a `ValidationReport`. Leak-type findings raise `LeakageWarning`. |
| `panelary/testing.py:assert_no_lookahead`, `assert_prefix_invariant` | The two leak instruments. | Every op, plus the new knowledge-time harness (§9). |
| `panelary/synth/` | Factors, clusters and a planted lag, but **no network dial**. | The planted-spillover generator is a **test fixture** (`tests/_network_synth.py`), not new public surface (§9). |
| `panelary/validation/_selection_stats.py` | Holm, BH, BY. | BHP edge selection and LISA FDR. |
| `panelary/feature_extractors/_kernels.py:_get_cusum_numba` | The lazy, cached numba pattern behind `have("numba")`. | Copy the pattern for `network/_kernels_numba.py`. |

Nothing named `network`, `spatial`, `moran`, `pagerank`, `laplacian`, `leadlag` or
`conley` exists in `panelary/` (grep, 2026-09-29).

---

## 3. Scope and non-goals

**In scope.**
* The as-of graph primitive: `EdgeTable`, validation, resolution, epochs and CSR.
* Weight builders: kNN, distance band and kernel graphs on coordinates or characteristics;
  correlation graphs (threshold, top-k, MST, TMFG); BHP de-factored W; convex combinations.
* Network lag, peer gap, orthogonalisation, the diffusion bank, Laplacian smoothing and
  total variation.
* Moran, Geary, LISA and Getis–Ord.
* Centralities over time and Herskovic concentration/sparsity.
* Rolling NAR/GNAR.
* Lead–lag matrices, Hermitian-RW clustering and follower signals.
* CD*, the BKP α exponent and Conley HAC. The econ-side items live in econ files (§5).

**Not shipping. These are decisions, not omissions.**
* **PMFG.** It is O(N³) with planarity testing. TMFG has the same topology class at O(N²).
* **Betweenness and closeness.** Brandes' algorithm is O(N·E) per date: 1.25 × 10⁹
  operations per date at N = 5,000, k = 50. It is dead on a daily panel. The escape
  hatch is networkx on a handful of dates.
* **Community detection** (Louvain/Leiden). It is non-deterministic without a
  heavyweight dependency. Hermitian clustering covers the directed case we need.
* **GNN training.** The diffusion bank gives precompute-then-learn features for any model.
* **Graph learning beyond thresholding** (graphical lasso, NOTEARS). Shrinkage and
  precision matrices belong to sibling plan 3.
* **Spatial panel ML estimators** (SAR/SEM/SDM with FE, Lee–Yu). These are M7,
  optional, and reduced-form only (§7.11).
* **A scipy.sparse backend.** It is faster (0.24 s) but not canonical-order (1.2 × 10⁻¹¹),
  so installing scipy would change numbers. numba covers the speed and gives the same bits.
* **A `reduceat` backend** is excluded for the same reason (1.8 × 10⁻¹²). §14 Q3 revisits
  this with a benchmark.
* **A polars `join` + `group_by` aggregation backend.** It is 1.9 ms/date and its sum
  order is unspecified. Polars does the sorting, joining and resolution; numpy does the
  arithmetic.

---

## 4. Hard invariants (every function)

1. **Prefix invariance.** Outputs at dates `≤ t` are **bitwise** identical when rows after
   `t` are appended, via `assert_prefix_invariant`.
2. **Knowledge-time invariance.** Appending edge or attribute vintages whose
   `knowledge_time + lag > t` leaves every output at `≤ t` **bitwise** unchanged. Moving
   one such vintage's knowledge time to `≤ t` must change something (a sensitivity check,
   so the test can fail).
3. **Canonical summation order.** Every neighbour aggregation sums row `i`'s terms
   left-to-right in **ascending neighbour-label order**. `bincount` (time-varying), slot/ELL
   (static epochs) and numba are bitwise identical (measured). `reduceat`, scipy and any
   pairwise or SIMD reduction over edges are forbidden in kernels.
4. **No GEMM whose shape depends on T.** A GEMM with a fixed per-date shape (N_t × W) is
   allowed: N_t is fixed once `t` is. Date-blocked GEMM of `W @ X[:, block]` is not. The
   GEMV-vs-GEMM trap is measured.
5. **Node codes are label-monotone.** Codes come from sorted labels (`PanelArrays`). Never
   use appearance order, `unique()` order or Categorical physical codes. New entities
   appearing later may shift codes but never reorder existing ones, so row order and sums
   are preserved.
6. **Graph lag.** A data-built graph used at `t` comes from data through `t − graph_lag`,
   default `graph_lag=1`.
7. **As-of universe.** Weights are (re)normalised over neighbours that exist **and** are
   observed at the date used. Never over the whole-sample universe.
8. **Determinism.**
   * Iterative solvers start from fixed vectors (uniform, or `x0 = 0`) with fixed `tol` and
     `max_iter`.
   * Batched iterative solvers **freeze each column at its own convergence**, never "iterate
     all columns until the slowest converges".
   * Every RNG is `SeedSequence([seed, date_key, stream])`. `date_key` is the date's integer
     physical value (days since epoch, or the integer time value), **never its position**.
   * Ties are broken by `(value, label)`.
9. **float64 everywhere.** Use `np.linalg.solve(A, b[..., None])[..., 0]` (the NumPy 2.0
   batch pitfall, per `detect-build-contract.md`).
10. **No `rolling_map`, no per-window Python UDFs.** Python loops may run over epochs,
    slots, date-chunks or iterations, **never over entities or edges**.
11. **Strides are anchored to the calendar**, not to the panel's first row (§7.5).
12. **No contemporaneous own-target spatial lag.** Estimators refuse `W y_t` as a regressor
    for `y_t`, and `lag=0` is refused when `x` is the declared target.

---

## 5. Module placement and public API

```
panelary/network/
  __init__.py          # orchestrator; public names below; registers FeatureSpecs
  _edges.py            # EdgeTable, GroupGraph, validation, resolution -> epochs
  _graph.py            # Epoch, GraphSource protocol, transpose/symmetrise/normalise/combine
  _kernels.py          # canonical SpMV/SpMM: bincount (time-varying), slot/ELL (static)
  _kernels_numba.py    # optional, lazily compiled, same bits (fast extra)
  _build.py            # knn / band / kernel / correlation (threshold, topk, mst, tmfg) / bhp
  _lag.py              # network_lag, diffusion_bank, graph_smooth, graph_total_variation
  _spatial.py          # moran, geary, local_moran, getis_ord
  _centrality.py       # centrality, network_state (incl. Herskovic)
  _nar.py              # rolling_nar (+ GNAR r-stage)
  _leadlag.py          # leadlag_matrix, LeadLagResult, hermitian_clusters
  _inference.py        # ConleySpec, conley_meat (consumed by econ._hdfe)
panelary/econ/_panel.py   # pesaran_cd(..., bias_corrected=), cd_exponent  (econ-owned file)
panelary/econ/_hdfe.py    # vcov="conley", conley=ConleySpec                (econ-owned file)
```

| File | Owner | Milestone |
|---|---|---|
| `_edges.py`, `_graph.py`, `_kernels.py`, `_kernels_numba.py`, `_lag.py` (network_lag only) | Agent A | M1 |
| `_build.py` | Agent B | M2 |
| `_spatial.py` | Agent C | M3 |
| `_centrality.py`, rest of `_lag.py` | Agent D | M4 |
| `_leadlag.py` | Agent E | M5 |
| `_nar.py`, `_inference.py`, econ edits | Agent F | M6 |
| `tests/test_network_*.py`, `tests/_network_synth.py`, `tests/_network_harness.py` | Agent G | all |
| `__init__.py`, registry specs, docs, mkdocs nav, CHANGELOG, conformance-table rows | orchestrator | — |

### 5.1 Public API (milestone in brackets; nothing else becomes public)

```python
import panelary as pn
nw = pn.network

# [M1] the as-of graph primitive
edges = nw.EdgeTable(
    df, src="supplier", dst="customer", weight="sales_share",   # weight optional -> 1.0
    valid_from="start", valid_to="end",                          # valid_to optional (open)
    knowledge_time="filed",                                      # REQUIRED (see 6.3)
    knowledge_lag="1bd",                                         # Duration or {edge_type: Duration}
    edge_type="type", source="source", edge_id=None,             # default id: (src, dst, edge_type)
)
report = edges.validate(panel)            # -> panelary.quality.ValidationReport
g = edges.resolve(panel)                  # -> GraphSource (lazy epochs; nothing materialised)
g.epochs_frame()                          # epoch, start, stop, n_edges   (inspection)
g.at(date)                                # src, dst, weight as a polars frame (inspection)
g.transpose(); g.symmetrize(how="sum"); nw.combine([g1, g2], weights=[0.7, 0.3])
nw.EdgeTable.from_matrix(M, names, time=..., knowledge_time=...)  # DY table, lead-lag, ...
nw.GroupGraph(memberships, entity=, group=, event_time=, knowledge_time=)  # as-of peers

out = nw.network_lag(
    panel, x=["ret"], graph=g,
    lag=1,                          # steps on the panel's date axis; 0 allowed (see 7.2)
    direction="out",                # "out" (W), "in" (W^T), "both" (symmetrised)
    weighting="row",                # "row" | "raw" | "sym"
    hops=1,                         # iterated operator P^k
    missing="renormalize",          # "renormalize" | "zero" | "null"
    orthogonalize_own=True,         # per-date residualisation (7.2)
    commonality_window="1y",        # own + peer trailing means as controls; None = own lag only
    own_controls=None,              # extra own columns, e.g. ["mom_12_2"]
    peer_gap=False, prefix="net_", backend="auto",
)                                   # -> same frame type + net_ret, net_ret__n, net_ret__cov[, __gap]

# [M2] builders (each returns a GraphSource; graph_lag=1 by default)
nw.knn_graph(panel, cols=[...] | coords=("lat", "lon"), k=10, metric="euclidean" | "haversine",
             rebuild="1mo", graph_lag=1)
nw.band_graph(..., radius=...); nw.kernel_graph(..., kernel="gaussian", bandwidth=... | adaptive_k=10)
nw.correlation_graph(corr_source, method="mst" | "tmfg" | "topk" | "threshold", k=..., tau=...)
nw.defactored_graph(panel, value, window=252, factors="csa" | int, p=0.05, rebuild="1mo")  # BHP

# [M3] spatial statistics
nw.moran(panel, x, g) -> per-date frame; nw.geary(panel, x, g) -> per-date frame
nw.local_moran(panel, x, g, permutations=0, seed=0, rebuild=None) -> per-row
nw.getis_ord(panel, x, g, star=True) -> per-row

# [M4] centrality, diffusion, smoothing
nw.centrality(g, panel, measures=("strength", "eigenvector", "pagerank", "katz")) -> per-row
nw.network_state(g, panel, shares=None, eta=0.5) -> per-date (incl. Herskovic C/S)
nw.diffusion_bank(panel, x, g, hops=3, directions=("out", "in"), kind="power" | "ppr")
nw.graph_smooth(panel, x, g, lam=1.0, laplacian="combinatorial" | "normalized")
nw.graph_total_variation(panel, x, g) -> per-row TV + per-date Dirichlet energy

# [M5] lead-lag
ll = nw.leadlag_matrix(panel, "ret", window=60, max_lag=5, metric="ccf_auc" | "ccf_lag1" | "signature",
                       corr="pearson", rebuild=1, graph_lag=1)       # -> LeadLagResult
ll.leadingness()                    # per-row row-sum score (Huber 1962)
ll.graph(top_k=10)                  # GraphSource: follower <- leaders, weights S+
nw.hermitian_clusters(ll, k=4, rebuild="1mo", seed=0)   # per-row cluster, 0 = most leading
# follower signal = nw.network_lag(panel, "ret", ll.graph(top_k=10), lag=0, orthogonalize_own=False)

# [M6] estimators and inference
nw.rolling_nar(panel, y, g, window=252, controls=(), stages=1, rebuild=1)  # per-date coefs + per-row forecast
pn.econ.pesaran_cd(panel, value=..., bias_corrected=True, n_factors=1)
pn.econ.cd_exponent(panel, value=..., window=None)
pn.econ.hdfe(..., vcov="conley", conley=nw.ConleySpec(graph=g, kernel="bartlett", time_lags=0))
```

**Fixed result schemas** (so `pl.concat` across dates, ops and graphs just works):

* **Per-row ops.** The input frame plus `{prefix}{x}[__suffix]`, where suffixes are `__n`,
  `__cov`, `__gap`, `__orth`, `__z`, `__p` and `__quad`. The frame type is preserved:
  collected internally, as `factor.orthogonalize` does.
* **Per-date analyses.** Columns are `time, statistic, estimate, expected, variance_norm,
  variance_rand, z, p_value, null_method, n_nodes, n_edges, s0, s1, s2, converged, n_iter,
  seed, warnings`. Irrelevant fields are null.

---

## 6. Data model: `EdgeTable` → `GraphSource` → epochs

### 6.1 Schema

| Column | Dtype | Required | Meaning |
|---|---|---|---|
| `src`, `dst` | panel entity dtype | yes | Directed edge `src → dst`: `src`'s peer set contains `dst`. For supplier → customer, `network_lag(direction="out")` gives each supplier its customers' mean (Cohen–Frazzini). |
| `weight` | Float64 | no (1.0) | Finite. Negative weights allowed only when `signed=True`. |
| `valid_from` | time family | yes | When the relationship starts (economic time). |
| `valid_to` | time family | no (open) | Exclusive end. A closure learned later is a **new vintage** carrying a later `knowledge_time`. |
| `knowledge_time` | time family | **yes** | When the record became knowable. |
| `edge_type`, `source`, `edge_id` | Utf8 | no | Provenance. `edge_id` defaults to `(src, dst, edge_type)`. |

### 6.2 Resolution algorithm: O(V log V), polars sort plus numpy searchsorted

1. **Validate.** See §6.4. Time dtype families go through `asof._common_time_dtype`, so a
   Datetime filing time is never floored onto a Date panel.
2. **Order vintages.** Sort by `(edge_id, knowledge_time)`. Collapse exact duplicates.
3. **Believed interval.** `K_v = shift_forward(knowledge_time, knowledge_lag)`, and
   `K_next = K.shift(-1).over(edge_id)`. This `shift(-1)` runs over **knowledge order
   within one edge**, not over panel dates: vintage `v` stops being believed once its
   successor is known. It is audited and pinned by the knowledge-time suite.
4. **Active interval on the calendar.** `a_v = searchsorted(cal, max(K_v, valid_from), "left")`
   and `b_v = searchsorted(cal, min(K_next, valid_to), "left")`. Keep rows with `a_v < b_v`.
   The consequence: a later correction (for example "this link actually ended in March",
   filed in June) is invisible to every date before June, exactly like a restatement in
   `asof_join`.
5. **Codes.** Map `src`/`dst` to label-monotone codes through a sorted label table (a polars
   join). Edges whose labels never appear in the panel are dropped and reported
   (`edges:unknown_entity`).
6. **Epochs.** Boundaries are the sorted unique `{a_v, b_v}`. Each epoch's active set is
   `a ≤ s_e < b`. Chunked explosion `np.repeat(v, e_end − e_start)` produces `(epoch,
   vintage)` pairs **one chunk of epochs at a time**. Each epoch becomes a CSR (rows sorted
   by src code, dst ascending within each row): `indptr` int64, `dst` int32, `w` float64.
   Streaming memory is O(max epoch edges): 4 MB at E = 250k.
7. **The universe is not an epoch boundary.** Entities entering or leaving do not fragment
   epochs. The kernel sees `x = NaN` for absent entities, and row normalisation happens at
   compute time over observed neighbours (invariant 7). Daily universe churn therefore costs
   nothing.

### 6.3 Knowledge time: refuse to guess

* `knowledge_time` is required. `knowledge_time="valid_from"` is accepted **only** with
  `assume_known_at_valid_from=True`, and emits a `LeakageWarning` naming the trap.
* `knowledge_lag` accepts a `Duration`, or a dict by `edge_type`. There is **no default
  lag**, because a default would be a guess dressed as a fact.
* The docs list *suggested* conventions, each marked as a user choice:
  * 10-K customer disclosures: fiscal-year-end + 6 months, the Fama–French accounting-lag
    convention (as used by Cohen–Frazzini†).
  * I/B/E/S coverage: estimate date + 1bd.
  * Patents: publication date + 1bd.
  * HQ location: the **historical** 10-K header, filing date + 1bd (the Compustat header is
    current, hence backfilled).
  * GICS: historical classification effective date + 1bd.

### 6.4 Validation rules → `quality.ValidationReport` (each finding carries `locator` / `observed` / `expected`)

| Check id | Impact | Why |
|---|---|---|
| `edges:null_keys` | fail | Null `src`/`dst`/`valid_from`/`knowledge_time` cannot be placed in time. |
| `edges:time_dtype` | fail | Time families differ (asof rules). |
| `edges:interval` | fail | `valid_to ≤ valid_from`. |
| `edges:weight_finite` | fail | NaN or inf weight. |
| `edges:duplicate_vintage` | fail | Same `(edge_id, knowledge_time)` with different content. |
| `edges:negative_weight` | warn (fail if `signed=False` for an op that needs it) | PageRank, row-normalisation. |
| `edges:self_loop` | warn | Dropped by default (`self_loops="drop"`). |
| `edges:knowledge_equals_valid_from` | **warn, LeakageWarning** | More than 90% of rows have `knowledge_time == valid_from`: backfill suspected. |
| `edges:classification_single_vintage` | **warn, LeakageWarning** | A classification-like edge type has one knowledge date at or after most panel dates: "current GICS applied to history". |
| `edges:unknown_entity` | warn | A label never appears in the panel. |
| `edges:future_knowledge` | info | Knowledge time after the panel's end. Harmless and never used. |

### 6.5 `GroupGraph`: classification peers without O(Σ n_g²) edges

"Everyone in my industry" is a complete graph within each group. For a 500-member
industry that is 250k edges, but the peer mean is `(S_g − x_i)/(n_g − 1)`. Membership is
resolved with **`asof_join`** (an entity's latest-known group at `t`). Group sums use
`bincount` over `(date, group)` codes, which is canonical order. This costs O(N T), not
O(Σ n_g² T). The leave-one-out result differs from explicit-edge sums in the last bits, so
parity is 1e-12, documented. `GroupGraph.to_edges(max_group=...)` expands it when another
op needs explicit edges.

### 6.6 The `GraphSource` protocol

```python
@dataclass(frozen=True, slots=True)
class Epoch:
    start: int; stop: int                 # calendar index range [start, stop)
    indptr: np.ndarray; dst: np.ndarray; weight: np.ndarray   # canonical CSR over label codes
class GraphSource(Protocol):
    directed: bool
    def epochs(self, calendar: np.ndarray, labels: np.ndarray) -> Iterator[Epoch]: ...
```

Builders produce epochs **lazily**. Ops accept several `x` columns, so one pass over the
epochs serves all of them: the graph is built once per call. `materialize(max_bytes=)`
caches epochs for reuse across calls and refuses above the cap. It never silently spills.

---

## 7. Algorithms: candidates, choice, cost, accuracy

### 7.1 Aggregation kernels (the engine room)

`net_i = Σ_j w_ij x_j`, together with `den_i = Σ_j w_ij·1[x_j observed]` and
`n_i = Σ_j 1[observed]`.

| Candidate | Cost | Measured (N = 5,000, k = 50) | Canonical order? | Verdict |
|---|---|---|---|---|
| Dense GEMM `W @ X` | O(N² T) | 0.89 s (static); 200 MB W | no (shape-dependent) | Only N ≤ 2,000 **with fixed per-date shape**. Never date-blocked. |
| gather + `reduceat` | O(E T) | 8.48 s static · 0.22 ms/date time-varying | no | Rejected. |
| **slot/ELL row-gather**: for slot r, `acc[rows_r] += w_r[:, None] * X[dst_r, block]` | O(E T), contiguous row gathers | **1.90 s** static · 0.61 s heavy-tailed | **yes** | **Default for epochs of 8 or more dates.** Fixed 512-date column blocks; each column is independent, so block width cannot change bits. |
| **`bincount`** over global `(date, src)` codes, edges sorted `(date, src, dst)` | O(E) per date | **1.03 ms/date** | **yes** | **Default for time-varying graphs** (epochs of fewer than 8 dates, chunked 20 dates). |
| polars `join` + `group_by` | O(E log E) | 1.93 ms/date | no | Rejected. |
| scipy CSR | O(E T) | 0.24 s / 0.09 ms | no | Rejected (§3). |
| **numba sequential CSR** (`fast`) | O(E T) | **0.07 s** static (parallel over dates) · **0.024 ms/date** (parallel over rows) | **yes** | `backend="auto"` uses it when installed. Bitwise-parity test against numpy. `fastmath=False`, `error_model="numpy"`, compiled lazily with `cache=True`, following the `_get_cusum_numba` pattern. |

The static-vs-time-varying choice cannot change a result, because both paths are
canonical; a property test pins this. Memory for 5,000 × 5,000 per feature column: X at
200 MB, the observed mask at 25 MB, the output at 200 MB, and slot temporaries of 2 × 20
MB. **Peak is under 0.6 GB per column.** Columns are processed sequentially over one epoch
pass.

### 7.2 `network_lag`

`peer_{i,t} = Σ_j w_{ij,t} x_{j,t−ℓ}`.

* The graph is the one as of `t` (`graph_at="t"`, or `"t-lag"`). `ℓ` counts steps on the
  panel's shared date axis, following depend's convention.
* **`lag` default 1.** `lag=0` is causal for a predictor observed at `t`: a customer's return
  at `t` against a supplier's `r_{t+1}` target is the Cohen–Frazzini feature. It is refused
  when `x` equals the declared `target=`, and estimators refuse it (invariant 12).
* **Direction.** `"out"` uses the CSR as resolved. `"in"` uses the per-epoch transpose,
  re-sorted canonically. `"both"` merges `w_ij + w_ji` (or `max`), sort-merged per epoch.
* **Weighting.** `"row"` is `num/den`, the mean over observed peers. For signed graphs,
  `den = Σ|w|`. `"raw"` is `num`. `"sym"` is `D^{-1/2} A D^{-1/2}`, with the degrees
  computed per date over observed nodes.
* **Missing policy.** `"renormalize"` (default) is the as-of universe rule. `"zero"` keeps
  the denominator. `"null"` returns null if any neighbour is missing. `__cov = den/Σ_j w_ij`
  reports coverage.
* **Hops.** `P^k` is applied iteratively, with the missing policy at every hop. It is
  documented as the iterated operator, which differs from `(P^k)x` under renormalisation.
* **Orthogonalisation** (`orthogonalize_own=True`) runs per date, cross-sectionally:
  `peer ~ 1 + own_{t−ℓ} [+ own_mean_W + peer_mean_W] [+ own_controls]`, keeping the
  residual. The trailing means are `rolling_mean_by(time, commonality_window)` per entity.
  They are causal and prefix-invariant. The peer mean is one more SpMM.
  * **Algorithm.** Centre per date, form the per-date `p × p` moments with `bincount` over
    date codes, then run one batched `solve` over all dates. That is O(n p²) with no Python
    loop over dates. A date needs at least `p + 2` rows, otherwise the result is NaN.
  * **Parity.** Against `factor.orthogonalize` (the second Gram–Schmidt column) and
    `_neutralize_kernel.cross_section_residuals`, to 1e-12.
  * **Why the commonality controls:** §1.2. Replications of the raw literature
    (Cohen–Frazzini) set `orthogonalize_own=False`, and the docs say so.
* **Peer gap:** `x_{i,t−ℓ} − peer_{i,t}`.
* **Complete-graph identity.** `network_lag` over `GroupGraph(one group)` equals
  `CrossSectionalAverages` (leave-one-out corrected) to 1e-12.

### 7.3 Diffusion bank (SGC/SIGN-style)

`[x, Px, …, P^K x]` for `P ∈ {P_out, P_in}`.

* `operator` is `"rw"`, `"sym"` or `"rw_selfloop"`, where the last is Wu et al.'s
  `Ã = A + I`.
* `kind="ppr"` computes `Σ_{k≤K} α(1−α)^k P^k x`. K is the smallest value whose remainder
  bound `(1−α)^{K+1}` is at most `tol`. The formula is deterministic and has no iteration
  to converge.
* Cost is K × SpMM per direction. Features go to any model; panelary trains nothing.

### 7.4 Laplacian (Tikhonov) smoothing and total variation

* **Problem.** Solve `(M + λL) z = M x`, where M is the diagonal of observed nodes.
  `L = D − A_sym` (combinatorial) or `I − D^{-1/2} A D^{-1/2}`.
  * A component with no observed node gets ridge `εI` with `ε = 1e-8`, and is flagged.
  * This doubles as per-date harmonic interpolation, which is leak-safe.
* **Solver.** Jacobi-preconditioned CG: `x0 = 0`, stop at `‖r‖ ≤ 1e-10 ‖Mx‖`,
  `max_iter=500`. Measured at 14 iterations, 5.9 ms.
* **Candidates rejected.**
  * Dense Cholesky: O(N³) = 1.25 × 10¹¹ per date.
  * Unpreconditioned CG: needs √κ ≈ √(1 + 2λ d_max) more iterations.
  * Chebyshev semi-iteration: needs spectral bounds per date.
* **Batching.** Multi-RHS PCG over an epoch's dates. The per-column operator
  `m_col ∘ z + λ L z` shares L. Columns **freeze at their own convergence** (invariant 8).
* **λ is a parameter.** Choosing λ by CV is a fit, and belongs to the caller's fold.
* **Total variation.** Per row, `TV_i = Σ_j w_ij |x_i − x_j|` (one gather plus one
  `bincount`). Per date, the Dirichlet energy is `xᵀLx`.

### 7.5 Weight builders (M2)

**Rebuild schedule.** `rebuild=` takes an int `s` or a duration.
* An int `s` rebuilds on dates whose **calendar key** (`np.busday_count` from 1970-01-01
  for Date axes, or the integer value) satisfies `≡ 0 (mod s)`.
* A duration rebuilds on the first panel date of each `dt.truncate(period)`.
* The graph used at `t` is the latest rebuild at or before `t − graph_lag`.
* Appending never moves an earlier rebuild date. Left-truncation affects at most the first
  period, and that is documented.

**kNN, distance band and kernel graphs on characteristics or coordinates.**
* **Input.** Characteristics are as of `t − graph_lag`. They are standardised
  **cross-sectionally per rebuild date**; pooled scaling is a leak. Coordinates are
  converted to 3-D unit vectors, where chord distance is monotone in great-circle distance.
* **Algorithm.** Blocked GEMM `‖q‖² + ‖x‖² − 2qxᵀ` over query blocks of 512. The shape is
  set by N_t, which invariant 4 allows. `argpartition` keeps k + 8 candidates. Candidates
  get an **exact recompute** by direct subtraction (or exact haversine), then
  `lexsort(exact_d, label)`.
* **Guard.** If the GEMM gap between candidate k and candidate k + 8 is below 1e-9
  relative, the margin widens. Cancellation can never reorder true neighbours.
* **Cost.** O(N² d) per rebuild: about 20–50 ms at N = 5,000 and d = 10 (estimate; M2
  measures it). Daily rebuilds are about 2–4 min over 5,000 dates; `rebuild="1w"` brings
  that to under 1 min.
* **Kernels.** Gaussian (truncated at 3h) or bisquare. The bandwidth is either fixed, or
  adaptive per node `h_i = d_(k)(i)`, which is per date and therefore legal. **A global
  bandwidth estimated on the whole panel is a leak.**

**Correlation graphs**, consuming sibling 3's matrices.
* **Interface contract.** `CorrSource` is duck-typed to sibling 3's lazy series
  `cov.rolling(panel, returns=, window=, method=, schedule=cov.Schedule(...))`. It exposes
  `.dates`, `.universe(d)` and `.at(d).corr().to_dense()`. Each stamp's C is symmetric with
  a unit diagonal and uses returns at or before the stamp. **Their `Schedule` is the rebuild
  schedule:** the builder never re-schedules on top of it. It only shifts each stamp to be
  used from the next panel date (`graph_lag=1`).
* **Raw input.** A plain iterable of `(stamp_date, labels_sorted, C)` is also accepted.
* **Fallback.** If sibling 3 has not landed, use the sample Pearson correlation of the
  complete-case window, documented as unshrunk. The docs recommend shrinkage when N/W > 0.5.
* **`threshold`** keeps `|ρ| > τ`, signed or absolute.
* **`topk`** keeps the top k per row by `|ρ|`, with ties broken by label. Optional
  symmetrisation.
* **`mst`** (Mantegna 1999) runs **Prim, O(N²)** on `d = sqrt(2(1−ρ))`, i.e.
  `depend.to_distance(kind="angular")` × 2.
  * Kruskal is O(N² log N). Measured: scipy 17.9 ms, networkx 195 ms, Prim **1.1 ms**.
  * Start at the lowest label; `argmin` returns the lowest code among equal keys. This is
    deterministic, and label-monotone codes make it prefix-stable.
  * Oracle: total weight parity with scipy (1e-12) and exact edge sets when there are no ties.
* **`tmfg`** (Massara, Di Matteo & Aste 2017) runs in O(N²).
  * Seed with the tetrahedron of the 4 largest strengths over above-mean weights, ties by label.
  * Keep, for each face, the best remaining vertex and its gain.
  * Each step takes the argmax over faces, inserts the vertex (1 face becomes 3) and
    recomputes only the new and stale faces, each in O(N).
  * The result has 3N − 6 edges and is chordal.
  * Weight: `ρ²` by default†. Measured 29 ms prototype, budget 60 ms. PMFG is not shipped.

**BHP de-factored W** (Bailey, Holly & Pesaran 2016), one rebuild at a time, on the window
ending at `t − graph_lag`:
1. **De-factor.** Regress each series on the in-window cross-sectional average (CCE, same
   semantics as `CrossSectionalAverages`) or on m principal components, and keep the
   residuals `e_i`.
2. **Correlate.** Compute the residual correlations `ρ̂_ij` with one GEMM, O(N² W).
3. **Multiple testing** (Bailey, Pesaran & Smith 2019). Keep an edge when
   `√W |ρ̂_ij| > Φ^{-1}(1 − p / (2 f(N)))` with `f(N) = N(N−1)/2`†.
4. **Split.** Emit `W⁺` (significant positive) and `W⁻` (significant negative) separately,
   each row-standardised. Also report the residual α exponent (§7.10) as a check that the
   dependence left over is weak.

**`combine`**: `W(θ) = Σ θ_k P_k` over row-stochastic components, with θ on the simplex. The
result's epochs are the union of the components' epochs, merged by sort-merge. **θ is a
parameter**; estimating it is a fit.

### 7.6 Spatial autocorrelation (M3)

Everything is per date over observed nodes. `z = x − mean_t`; a global mean is a leak.

* **Global Moran's I.** `I = (N/S0)·zᵀWz / zᵀz`.
  * `S0 = Σw`.
  * `S1 = Σ_e (w_e + w_rev(e))² · (½ if rev(e) exists else 1)`. The reverse-edge index is
    built once per epoch by sort-merge.
  * `S2 = Σ_i (w_i· + w_·i)²`.
  * Moments under normality and randomisation follow Cliff & Ord (1981):
    `E[I] = −1/(N−1)`;
    `Var_N = (N²S1 − N S2 + 3S0²)/((N²−1)S0²) − E[I]²`;
    `Var_R = [N((N²−3N+3)S1 − N S2 + 3S0²) − b2((N²−N)S1 − 2N S2 + 6S0²)] / ((N−1)(N−2)(N−3)S0²) − E[I]²`,
    with `b2 = N Σz⁴ / (Σz²)²`.
  * The S-terms are recomputed per date because missingness changes W. That costs O(E).
    Measured 1.06 ms/date.
* **Geary's C.** `C = (N−1) Σ w_ij (x_i − x_j)² / (2 S0 Σz²)`, with `E[C] = 1` and
  Cliff–Ord variances.
* **Local Moran (LISA).** `I_i = (z_i/m2) Σ_j w_ij z_j`, with `m2 = Σz²/N`.
  * **Analytic by default.** `E[I_i] = −w_i·/(N−1)`. `Var_R[I_i]` follows Anselin (1995)
    as corrected by Sokal, Oden & Thomson (1998)†, built from `w_i(2) = Σ w_ij²` and
    `w_i(kh) = w_i·² − w_i(2)`.
  * **Conditional permutation, opt-in.** `permutations=B`, with `rebuild=` for striding.
    PySAL-style: one shared `(B × k_max)` random index matrix per date from
    `SeedSequence([seed, date_key, 1])`. Indices skip `i` by shifting. Neighbour weights are
    taken in slot order. `p = (1 + min(#≥, #≤))/(1 + B)`.
  * Cost is O(B·E) per date, 0.32 s measured. Quadrant codes are HH / LL / HL / LH. BH-FDR
    across `i` per date uses `validation._selection_stats.benjamini_hochberg`.
* **Getis–Ord.** The Ord & Getis (1995) z-scores†:
  `G*_i = (Σ_j w_ij x_j − x̄ W_i) / (s √((N S_1i − W_i²)/(N−1)))`. The self-weight is
  included for G*, and G uses N − 1 terms.
* **A numpy-only oracle that needs no PySAL.** For N ≤ 8, **enumerate all N! permutations**
  and check that the exact mean and variance of I equal `E[I]` and `Var_R` to 1e-12.
  Likewise enumerate the other N − 1 values for local Moran. This pins the † formulas
  independently of PySAL, and PySAL parity (`esda.Moran`, `Geary`, `Moran_Local`,
  `G_Local`) is added when installed.

### 7.7 Centrality over time (M4)

A graph is constant within an epoch, so **centrality is computed once per epoch and
broadcast**. Time-varying graphs are batched with global offsets.

* **Degree and strength.** In and out, via `bincount`.
* **Eigenvector.** Symmetrised, non-negative. Shifted power iteration `v ← (A + I)v/‖·‖`:
  the shift kills bipartite ±λ oscillation without changing the vector.
  * Start from a uniform vector. Stop when `‖Δv‖∞ < 1e-12`, or at `max_iter=1000`.
    Signs are pinned with `fix_signs`.
  * If it does not converge, fall back to `top_eigenvectors` (`eigh`) when N ≤ 2,000;
    otherwise return NaN with `converged=False`.
  * Measured 0.07 ms against 9.8 ms for `eigh` at N = 500. With several components the
    vector concentrates on the component with the largest λ, as documented; PageRank is
    the recommended alternative.
* **PageRank.** `p ← α Pᵀp + (α·dangling + 1 − α)/N`, with α = 0.85, a uniform start,
  L1 tolerance 1e-12 and `max_iter=1000`. Measured 13 iterations, 3.9 ms. Oracle:
  `nx.pagerank(tol=1e-14)` to 1e-10.
* **Katz.** `x ← κ/ρ(A) · Aᵀx + β`. κ = 0.5 is relative; ρ(A) comes from the power
  iteration above (per date, cross-sectional, legal). About 40 iterations reach 1e-12.
* **Asymmetry.** `(s_out − s_in)/(s_out + s_in)`.
* **`network_state`, per date.** Nodes, edges, density, mean and max degree, degree HHI,
  reciprocity (from the reverse-edge index), spectral radius.
* **Herskovic (2018).** Formulas verified from the author's 2015 handout:
  * `δ = (1−η)(I − ηWᵀ)^{-1} α`, by Neumann iteration `δ ← (1−η)α + ηWᵀδ`. W is
    row-stochastic and η < 1, so about 40 iterations reach 1e-12 at η = 0.5.
  * **Concentration** `N^C = Σ_i δ_i log δ_i`.
  * **Sparsity** `N^S = Σ_i δ_i Σ_j w_ij log w_ij`, with 0 log 0 = 0.
  * **Inputs.** The as-of share column α defaults to 1/N; η defaults to 0.5†.
  * **Scope note.** Herskovic defines these on **sector** input–output tables. At firm level,
    with disclosure truncated to customers above 10% of sales, the docs label them
    "IO-style" and do not claim equivalence.
  * **Rejected:** betweenness and closeness (§3).

### 7.8 NAR and GNAR, rolling (M6)

**NAR** (Zhu et al. 2017): `y_is = β0 + β1 (P y_{s−1})_i + β2 y_{i,s−1} + c_{i,s−1}ᵀγ + ε`.
* **Per-date sufficient statistics.** For each response date s: `M_s = Σ_i z zᵀ`,
  `v_s = Σ_i z y`, `Σy²` and `n`. These take `k(k+1)/2 + k + 2` `bincount`s over date codes.
* **Window sums, bitwise prefix-invariant.** Use **W shifted vectorised adds**:
  `acc += M[w : w + T − W + 1]` for `w = 0..W−1`. That is O(T W k²): 5,000 × 252 × 16 = 2 ×
  10⁷ operations.
  * Rejected: cumsum-difference. It is prefix-invariant, but cancellation reaches about
    10⁻¹³ on signed cross-moments at T = 5,000 and is amplified by `cond(M)`.
  * Rejected: `sliding_window_view(...).sum()`. Its reduction order is not guaranteed to be
    shape-independent.
* **Solve.** One batched `solve(A, b[..., None])`. Features are `β1_t`, `β2_t`, t-stats
  (window-homoskedastic or date-clustered), R², and a per-row forecast
  `ŷ_{i,t+1|t} = z_{i,t}ᵀβ_t`, with the coefficients coming from responses in `(t−W, t]`.
* **Flag.** Mark rows where `|β1| + |β2| ≥ 1`, Zhu et al.'s stationarity condition for
  row-normalised W. Stride with `rebuild=`.
* **GNAR** (Knight et al. 2020; `stages=R ≤ 2`).
  * r-stage neighbourhoods per epoch: `N^(2)(i) = N(N(i)) \ ({i} ∪ N(i))`, by expanding
    E·k candidate pairs and deduplicating by sort. Weights are `1/|N^(r)(i)|`.
  * The same sufficient-statistic machinery applies, with more columns.
* **Oracles.** `econ._common.ols` on the stacked window design, to 1e-10. Planted-DGP
  recovery (§9).
* **SLX in Fama–MacBeth.** `fama_macbeth(y="fwd_ret", x=["x", "net_x"])`. This is a
  documented recipe with a test. No new function.

### 7.9 Lead–lag (M5); the roadmap's `leadlag` done right

Everything is computed on the window ending at `t − graph_lag`, with lag sign as in depend
(`k > 0`: the row entity leads).

* **ccf-lag1 and ccf-auc**, following Bennett, Cucuringu & Reinert (2022), §3.1.1, checked
  against arXiv:2201.08283:
  * `I(i,j) = Σ_{l=1}^{L} |corr(y^i_{s−l}, y^j_s)|`. The absolute value is read from the
    paper's text; verify†.
  * `S_ij = sign(I(i,j) − I(j,i)) · max(I(i,j), I(j,i)) / (I(i,j) + I(j,i))`, skew-symmetric.
  * **Pearson fast path.** One GEMM per lag with **exact per-lag overlap moments**
    (`Y[a:b−l]ᵀ Y[a+l:b]` plus segment sums). The shape is fixed per (W, l), which
    invariant 4 allows. That is about 5 × 0.26 ms per date at N = 500.
  * **Nonlinear `corr`** (`"kendall" | "dcor" | "xi"`) uses depend's `get_kernel(m).rows`
    through `rows_chunked` over `N² · L` (pair, lag) rows. It is O(N² L W log W) per date,
    so `rebuild` of at least 21 is required at N = 500. This is documented.
* **Signature** (their eq. 3): `S = C_prevᵀ D − Dᵀ C_prev`, where D holds in-window log-price
  increments (standardised per series by default) and `C_prev` is the strictly-previous
  cumulative sum. It is one GEMM, 0.26 ms. It is positive when moves in `i` are followed by
  moves in `j`, and it cannot handle negative association (a caveat from the paper).
* **Leadingness.** The row-sum `Σ_j S_ij` (Huber 1962 row-sum ranking), emitted per row.
* **Hermitian-RW clustering** (Cucuringu et al. 2020; BCR §3.2.4).
  * `Ã = iS`. The RW-normalised Hermitian form is `i S̃` with `S̃ = D^{-1/2} S D^{-1/2}`
    and `d_i = Σ_j |S_ij|`.
  * Eigenvalues come in ±λ pairs with conjugate vectors, so **the pair-complete projector
    `P = Σ g gᴴ` is real and equals the projector onto the top-2m eigenvectors of the real
    symmetric `S̃ᵀS̃`.** Verified to 4.6 × 10⁻¹⁵. Real `eigh` costs 11.9 ms against 89 ms for
    complex `eigh`.
  * **Therefore no complex arithmetic,** and `l` is forced even (`l = 2⌈k/2⌉`). An odd `l`
    splits a ±λ pair, which is an arbitrary, non-deterministic tie. That trap is named and
    tested.
  * Map back with `U = D^{-1/2} Q`. Rows of `U Uᵀ` go to k-means++ and Lloyd (reused
    numpy, `n_init=10`, seeds `SeedSequence([seed, date_key, 2])`, best inertia).
  * **Labels are ranked by cluster leadingness** `L(c)` (BCR eq. 4), so 0 is the most
    leading cluster. That keeps labels meaningful across dates, with no label switching.
  * Subspace iteration was rejected: it had not converged after 500 iterations at
    eigen-ratio 0.983 (projector off by 8.6 × 10⁻⁶).
* **Follower signal = composition, not new code.** `ll.graph(top_k)` emits follower ←
  leader edges weighted by `S⁺`. Then `network_lag(ret, lag=0)` gives leaders' returns at
  `t` for followers' `r_{t+1}`. The cluster-level variant uses `GroupGraph(clusters)`.
* **The roadmap fix.** The SovAI original had a forward-looking shift. It is gone by
  construction: the only lags are `y[s−l]` with `l ≥ 1`, the window ends at `t − 1`, and
  §9 pins it with a leaky twin (window ending at `t + 1` / `shift(-1)`) that **must** fail
  `assert_prefix_invariant`, plus a planted-sign test.

### 7.10 Cross-sectional dependence and Conley inference (M6, econ-side)

* **CD kernel. Two paths, one owner** (this plan's M6 econ agent; sibling 3's §5.14
  hands this rewrite to "the econ owner").
  * **Balanced windows: O(NT).** Sibling 3's identity: standardise each series over the
    common window into `Z`, then `Σ_{i<j} ρ_ij = (‖Z1‖²/T − N)/2` and
    `CD = √(2T/(N(N−1))) Σ_{i<j} ρ_ij`.
  * **Unbalanced: pairwise-complete from 4 GEMMs** with the observed mask M. The √T_ij
    weights break the identity, so compute `n = MᵀM`, `Σx = X0ᵀM`, `Σx² = (X0²)ᵀM` and
    `Σxy = X0ᵀX0`, where X0 is shifted by the per-column observed mean to avoid
    cancellation. Then `CD = √(2/(N(N−1))) Σ_{i<j} √T_ij ρ_ij`.
  * Measured 7.8 ms against the existing loop's 1.78 s, matching to 1e-11. The balanced
    path must equal the GEMM path on complete data to 1e-12 (a test).
* **CD\*** (Pesaran & Xie, arXiv:2109.00408, eqs. 21–29, verified).
  * `CD* = (CD + √(T/2) θ̂)/(1 − θ̂)`, with `θ̂ = 1 − N^{-1} Σ_i â_i²`,
    `â_i = 1 − σ̂_i φ̂ᵀγ̂_i` and `φ̂ = N^{-1} Σ_i γ̂_i/σ̂_i`.
  * γ̂ and σ̂ come from PCA with `n_factors` on the balanced complete-case window, and the
    number of entities used is reported. The published venue is unconfirmed†.
* **BKP α** (Bailey, Kapetanios & Pesaran 2016, JAE 31(6), 929–960).
  * The core relation is `Var(x̄_t) = κ N^{2α−2} + c_N/N + …`. It gives
    `α̂ = 1 + ln σ̂²_x̄ /(2 ln N) − ln μ̂²_v /(2 ln N)`, with the bias correction
    `− ĉ_N /(2 N ln N σ̂²_x̄)`†. The exact μ̂²_v iteration and ĉ_N construction must be
    verified from the paper before coding.
  * Valid only for α > 1/2; otherwise return NaN plus a warning. It is exposed as
    `econ.cd_exponent` and as a rolling date-level diagnostic.
* **Conley (1999) spatial HAC for `hdfe`.**
  * `meat = Σ_t S_tᵀ(K_t S_t)` over a symmetric kernel graph K on economic distance
    (`kernel` is `"uniform"`, `"bartlett"`, or the graph weights), self-term included.
  * Optional `time_lags=L` adds Bartlett-weighted `Σ_t [S_tᵀ K S_{t−l} + ᵀ]`.
  * `K S_t` goes through the canonical kernel, one column per score. hdfe applies
    `_psd_clip`.
  * **Identity test:** Conley with the complete graph and uniform kernel equals
    `vcov="driscoll-kraay"` with `lags=0`, to 1e-12.
  * **No Fama–MacBeth option.** The λ̂_t time series already absorbs cross-sectional
    correlation (Petersen 2009†), and its serial correlation is covered by the existing
    Newey–West. Adding Conley there would double-count, and the docs say why.

### 7.11 SAR/SEM (M7, optional, only if a caller asks)

* **log|I − ρW|.** Use exact eigenvalues for N ≤ 2,000 (Ord 1975). For large sparse W, use
  Chebyshev with seeded Hutchinson traces (Pace & LeSage 2004†).
* **ρ̂.** Golden section on the concentrated likelihood over `(1/λ_min, 1/λ_max)`.
* **Leak rule.** A SAR model of `y_t` on `W y_t` is an *explanatory* model. Only reduced-form
  predictions `(I − ρ̂W)^{-1} X_t β̂` (with `X_t` known at t) may become features, fitted per
  fold (`PanelTransformer`, `leakage_safe=True` only because `fit` sees train rows alone).

---

## 8. Leak-safety design: every trap named

| # | Trap | How it leaks | Default that prevents it | Test that catches it |
|---|---|---|---|---|
| T1 | `knowledge_time := valid_from` (backfilled link DBs) | Links used before they were knowable | `knowledge_time` required; opt-in flag + LeakageWarning; `edges:knowledge_equals_valid_from` | knowledge-time suite; synthetic "late-disclosed links" panel where the leaky variant shows spurious IC |
| T2 | Current classification or HQ applied to history | Future membership | `GroupGraph` via `asof_join`; `edges:classification_single_vintage` | appending a later reclassification leaves earlier outputs unchanged |
| T3 | Retroactive correction applied from `valid_from` | A June correction changes March | vintage interval `[max(K_v, valid_from), min(K_next, valid_to))` | the restatement case in `test_network_edges.py` |
| T4 | Row-normalising over the whole-sample universe | Survivorship; length dependence | renormalise over observed neighbours at the date | prefix test with delisting entities |
| T5 | Graph from a window ending at `t`, used at `t` | Mechanical co-movement | `graph_lag=1` | leaky twin with `graph_lag=0` detected on a planted panel |
| T6 | Global bandwidth, threshold or θ estimated on the full panel | Fitted on the future | per-date adaptive bandwidth or fixed parameters | prefix test on builders |
| T7 | Pooled z-scores for characteristics or Moran | Future mean and sd | per-date centring and scaling | `assert_no_lookahead` |
| T8 | Contemporaneous `W y_t` for `y_t` | Simultaneity (the reflection problem, Manski 1993†) | `lag=1` default; estimators refuse; `lag=0` refused on the target | unit test |
| T9 | Date-blocked GEMM with a T-dependent last block | Bits change on append | invariant 4; slot/`bincount` kernels | GEMV-vs-GEMM regression test |
| T10 | Node ids from appearance or Categorical codes | Row order and bits change | label-monotone codes | append a new entity whose label sorts first |
| T11 | Stride anchored at the panel's first row | Rebuild dates move | calendar-keyed schedule | panels with different start dates share rebuild dates |
| T12 | RNG seeded by date position or one global stream | Left-truncation changes draws | `SeedSequence([seed, date_key, stream])` | truncation-from-the-left test |
| T13 | Batched CG or power iteration iterating to the slowest column | Bits of early dates depend on batch mates | per-column freezing | batch-composition invariance test |
| T14 | Odd `l` in Hermitian clustering | Arbitrary choice from a ±λ pair | `l = 2⌈k/2⌉` | determinism test across runs |
| T15 | Lead–lag via `shift(-l)` or a window ending at `t + 1` (the roadmap bug) | Future returns | only `y[s−l]` with `l ≥ 1`; window ends at `t − 1` | leaky twin fails `assert_prefix_invariant` |
| T16 | Entity-grouped CV with network features | Peers from the test fold enter train rows at the same dates (cross-fold contamination, not look-ahead) | documented; `network_lag(universe=train_entities)` option | test that the option restricts peers |
| T17 | NAR or θ fitted on the full sample, used as a feature | Fitted on the future | rolling fits only, stamped at `t` from responses `≤ t` | prefix test |

---

## 9. Tests

`--strict-markers` is on; no new markers are needed (`slow` and `benchmark` exist).

* **`tests/_network_synth.py`** (fixture, not public). A seeded, prefix-consistent planted
  network panel. Streams are keyed by step, as in `synth/_generate.py`. It provides:
  * the NAR DGP `y_t = β1 P y_{t−1} + β2 y_{t−1} + ε` with a known `EdgeTable`, whose
    knowledge times lag `valid_from` by a known amount;
  * a commonality-only DGP and a news-diffusion DGP (§1.2);
  * a leader/follower cluster DGP (followers load on leaders at lag 1);
  * a spatial-AR field with a known Moran sign.
* **`tests/_network_harness.py`.** `assert_knowledge_time_invariant(op, panel, edges, cut)`:
  append vintages known after `cut`, require bitwise-equal outputs at `≤ cut`, and show
  sensitivity when a vintage's knowledge time is pulled before `cut`. Promote to
  `panelary.testing` only if a caller asks (AGENTS rule).
* **`test_network_edges.py`.** Every §6.4 check fires, and does not fire on clean data. The
  bitemporal cases: restatement, closure learned late, correction invisible before its
  knowledge time, and open intervals. Dtype-family refusals (Datetime onto Date). Epochs
  equal a brute-force per-date resolution. `GroupGraph` equals `asof_join` membership.
* **`test_network_kernels.py`.**
  * slot, `bincount` and numba outputs are **bitwise equal** to a Python sequential
    reference: empty rows, hubs, 10^±8 weights, NaN masks.
  * The static-vs-time-varying path choice is bitwise neutral.
  * The GEMV/GEMM shape trap is reproduced, pinning why GEMM is banned.
  * The numba path skips cleanly without `fast`.
* **`test_network_prefix.py`.** `assert_no_lookahead` and `assert_prefix_invariant` over
  every per-row op and builder at several cuts. Also: rebuild-date anchoring, left-truncation
  seeds and batch-composition invariance (T11–T13).
* **`test_network_knowledge_time.py`.** The harness over every op, including builders fed
  by as-of attributes.
* **`test_network_oracle.py`** (each oracle behind `importorskip`):
  * networkx: PageRank, eigenvector, Katz, MST edges.
  * scipy `minimum_spanning_tree`: weight.
  * PySAL `esda`: Moran, Geary, Local Moran, G_Local.
  * **Exact-permutation enumeration** for Moran, Geary and LISA moments (numpy-only,
    always runs).
  * CG against `np.linalg.solve` (small dense). Herskovic δ against a direct solve.
  * NAR against `econ._common.ols`.
  * CD against the old loop implementation (copied into the test).
  * Conley against Driscoll–Kraay (the identity). `network_lag(GroupGraph)` against
    `CrossSectionalAverages`.
  * `orthogonalize_own` against `factor.orthogonalize` and `cross_section_residuals`.
* **`test_network_synth.py`** (partly `slow`):
  * NAR recovers β1 and β2 within 2 SE.
  * `test_commonality_only_is_suppressed`: raw peer IC is significant, and the default
    orthogonalised IC has |t| < 2.
  * The diffusion effect retains at least 80% of its IC.
  * Hermitian-RW ARI > 0.9 on the leader/follower DGP, with cluster 0 being the planted
    leaders.
  * Moran z sign is correct.
  * The late-disclosure leaky variant (T1) shows significant spurious IC; the honest one
    does not.
* **`test_network_leakage.py`.** Leaky twins for T1, T4, T5, T8, T12, T13 and T15. Each
  must be **detected**, following the `test_depend_leakage.py` pattern.
* **`test_econ_cd_star.py`** (`slow`). Mirrors depend's calibration style. With a strong
  factor and independent errors, **assert that CD over-rejects** (size > 0.3; the bug is
  real). Assert CD* size ∈ [0.03, 0.08] at 5%, and that α is recovered within 0.05 for
  planted α ∈ {0.6, 0.8, 1.0} at N = 500, T = 200.
* **`test_econ_hdfe_conley.py`.** Hand-computed 3-entity example, the DK identity and PSD clip.
* **Conformance.** New `network` FeatureSpecs are added to `_FRAME_OPS` in
  `tests/test_registry_conformance.py` with a probe `EdgeTable` over the probe panel's
  entities (staggered knowledge times). This is orchestrator-owned.
* **Standing guards.** `test_import_hygiene.py` (numba never imported at `import panelary`),
  `test_dependency_drift.py` (zero new mandatory dependencies), `test_wheel_guardrails.py`.

---

## 10. Benchmarks and performance budgets

`benchmarks/bench_network.py` is the script. `tests/test_network_perf.py` (marked
`benchmark`) asserts the budget, which is **2× the measured value** or 2× the M-milestone
measurement where marked "measure".

| Workload | Measured / expected | Budget |
|---|---|---|
| `network_lag`, 1 column, N = 5,000, T = 5,000, k = 50, static epochs, numpy | kernel 1.9 s × 2 (num + den) + orthogonalisation ≈ 5–6 s | ≤ 12 s; peak RSS ≤ 1 GB |
| same, numba | 0.07 s × 2 + orthogonalisation ≈ 1 s | ≤ 2 s |
| `network_lag`, daily-changing graph, 5,000 dates, numpy / numba | 1.03 ms / 0.024 ms per date × 2 | ≤ 21 s / ≤ 1 s (excluding the builder) |
| `EdgeTable.resolve`, 1M vintages, 240 epochs | measure (expected < 2 s) | 2× M1 measurement |
| kNN builder, N = 5,000, d = 10, per rebuild | measure (expected 20–50 ms) | 2× M2 measurement |
| MST / TMFG / correlation, N = 500, per date | 1.1 / 29 / 0.29 ms | 3 / 60 / 1 ms |
| eigenvector, dense N = 500, per date (batched) | 0.16 ms | 0.5 ms |
| PageRank, N = 5,000, E = 250k, per snapshot | 3.9 ms | 10 ms |
| Moran global, N = 5,000, k = 50 | 1.06 ms/date | 2.5 ms |
| LISA analytic / permutation B = 999 | measure (expected ≈ 2 ms) / 0.32 s per date | 2× / 0.7 s |
| PCG smoothing, N = 5,000, E = 500k | 5.9 ms | 15 ms |
| ccf-auc Pearson L = 5 / signature, N = 500, W = 60 | ≈ 1.5 / 0.26 ms per date | 4 / 1 ms |
| Hermitian-RW, N = 500, per rebuild | 11.9 ms `eigh` + k-means (measure) | 50 ms |
| `pesaran_cd`, N = 500, T = 252 | 7.8 ms (from 1.78 s) | 20 ms |

**Laptop headline** (5,000 entities × 5,000 dates, about 50 edges per node):
* A slowly-changing relationship graph: under 10 s per peer feature in pure numpy, and
  about 1 s with `fast`.
* Dense N = 500 correlation graphs (MST daily plus centrality): about 8 s end to end.
* TMFG daily: about 150 s. `rebuild="1w"` brings it to about 30 s.

---

## 11. Dependencies

* **Mandatory: none added** (`test_dependency_drift.py`).
* **Optional:** `numba` through the existing **`fast`** extra. Its `pyproject.toml` comment
  currently says it is "for the pure-Python CUSUM change-point kernel"; the orchestrator
  updates it to include the network kernels. No new extras.
* scipy, networkx and PySAL (`esda`, `libpysal`) are **test-only**, never in the import
  path, and are not added to `dev`: tests skip without them. The exact-enumeration oracles
  need nothing.
* **Polars floor: 1.35.** Use only `join`, `join_asof(by=...)`, `sort`, `over(order_by=...)`,
  `rolling_mean_by`, `search_sorted`, `dt.truncate` and `dt.offset_by`. All exist in 1.35.
  CI runs 1.35 on py3.10 and 1.42 on 3.11+.
* **mypy ratchet:** new modules must add zero errors. `ruff` as configured.

---

## 12. Milestones

| M | Contents | Exit criteria |
|---|---|---|
| **M1** | `EdgeTable` + validation → `ValidationReport`; resolution to epochs; `GraphSource`; canonical kernels (numpy + numba); `GroupGraph`; `network_lag` (direction, hops, missing policy, peer gap, `orthogonalize_own` with commonality controls); `_network_synth.py`; knowledge-time harness | Knowledge-time, prefix and kernel-parity suites green; commonality test green; perf budgets met; **truepoint caller test exists (§15)** |
| **M2** | Builders: kNN / band / kernel (coordinates and characteristics), correlation (threshold / topk / MST / TMFG via `CorrSource`), BHP de-factored W, `combine`, rebuild schedules | networkx and scipy parity; T5, T6, T11 twins detected. Needs a caller. |
| **M3** | Moran, Geary, LISA (analytic + permutation), Getis–Ord | exact-enumeration oracle; PySAL parity when installed |
| **M4** | Centrality, `network_state` (+ Herskovic), diffusion bank, `graph_smooth`, total variation | networkx parity; batch-freeze test |
| **M5** | Lead–lag matrices, leadingness, Hermitian-RW clustering, follower-signal recipe; roadmap `leadlag` closed | T14 and T15 twins detected; ARI > 0.9 on the planted DGP |
| **M6** | Rolling NAR/GNAR; SLX recipe; CD kernel upgrade + CD* + `cd_exponent`; `hdfe(vcov="conley")` | calibration (`slow`) green; DK identity |
| **M7** (optional) | SAR/SEM reduced form, with log-det by eigen / Chebyshev | only on explicit caller request |

M1 alone is shippable. Do not start M2 before M1's knowledge-time suite is green.

---

## 13. Boundaries with sibling plans (and existing code)

* **3. covariance-and-market-state.** It owns rolling and shrunk correlation and covariance
  matrices, market-state spectra (absorption ratio, top-eigenvalue share) and precision or
  LoGo matrices. **This plan consumes** its matrices through the `CorrSource` contract
  (§7.5), and emits node-level centralities and graph topologies only.
  * **Overlap risk 1: `pesaran_cd` rewrite.** Sibling 3 (§5.14 and its "not ours" list)
    proposes the balanced O(NT) identity and leaves the rewrite to the econ owner. This
    plan's M6 does it, using their identity for balanced windows plus the masked GEMM for
    unbalanced ones. **Only one of us edits `econ/_panel.py:pesaran_cd`.**
  * **Overlap risk 2: BKP α and CD\*** as "market state". This plan owns them because the
    brief assigns them here. Sibling 3's `avg_corr` is a feature, not a test.
  * **Overlap risk 3: PMFG.** Sibling 3's boundary table lists "graphs (TMFG, MST, PMFG)"
    as ours. **PMFG is deliberately not shipped** (§3), and their table should drop it.
  * **Agreed split** (their §12 table): graph eigenvector centrality is ours; the spectral
    market-mode loading and IPR of the correlation matrix are theirs.
* **10. panel-causal-inference.** Their plan (M7, boundary table) states "we never
  implement exposure mappings or Conley HAC". Exposure mappings are therefore
  `network_lag(treated, g, weighting="row" | "raw", lag=0, orthogonalize_own=False)` over
  an as-of `GraphSource`: the share or count of treated neighbours.
  * Our Conley is an `hdfe` vcov, so their TWFE or DiD regressions get it through `hdfe`.
  * They own the estimator; this plan owns the exposure operator and the Conley meat.
  * Because their M7 depends on our M1, **their M7 is a candidate second caller for M1**,
    but only once *they* have an engine caller.
* **9. tail-risk-and-self-excitation.** A multivariate-Hawkes excitation matrix, or a CoVaR
  network, enters through `EdgeTable.from_matrix`. They own the estimation.
* **8. multiscale-complexity-features.** They own path signatures of an entity's *own*
  channels. This plan owns the *cross-entity* pairwise Lévy-area matrix. There is no shared
  code beyond possibly a cumulative-sum helper.
* **2. drift-monitoring.** Graph-structure change monitoring, if wanted, consumes
  `network_state` series. It is not built here.
* **1, 4, 5, 7.** No overlap. Lead–lag via transfer entropy stays in `depend`
  (`transfer_entropy`).
* **Existing code** (§2): DY (`econ._connectedness`) is ingested, not rebuilt.
  `CrossSectionalAverages` is the complete-graph identity. `pesaran_cd` is upgraded in
  place: numbers change by up to 1e-11, and CHANGELOG notes it. `hdfe` gains one vcov
  branch. The private k-means in `evolve/_select.py` should be promoted rather than copied.

---

## 14. Risks and open questions (resolve with a benchmark, not an opinion)

1. **Caller.** This whole plan may stay parked (§0, §15). That is the correct outcome if no
   engine test needs it.
2. **`CorrSource` contract.** Sibling 3's plan fixes the shape of the API
   (`cov.rolling(...).at(d).corr().to_dense()`, `.universe(d)`, `Schedule`). Still
   unconfirmed: whether a stamp's window **includes** returns dated on the stamp.
   `graph_lag=1` is safe either way, but it wastes a date if their stamp already excludes
   `t`. Confirm this before M2.
3. **Canonical order versus speed.** `reduceat` is 4.6× faster than `bincount` for
   time-varying graphs but not canonical. If a real workload is `bincount`-bound without
   numba, decide from a benchmark whether a documented non-canonical `backend="reduceat"`
   is worth breaking cross-backend bit-parity.
4. **Formula checks.** The † items (LISA randomisation variance, Getis–Ord constants, TMFG
   weight convention, BPS `f(N)`, the BKP bias-correction details, Herskovic's η) must each
   be pinned by an oracle before merge. The exact-enumeration oracle covers the spatial
   ones; α needs Monte Carlo recovery.
5. **The commonality default.** `commonality_window="1y"` rests on one DGP. Validate on
   real linked-firm data (for example a customer–supplier panel) that the default does not
   remove genuine effects wholesale. If it does, demote it to opt-in and say why.
6. **Vendor reality.** Most relationship vendors do not ship knowledge times. The API
   refuses to guess, and adoption friction is the price. Document how to derive knowledge
   times from filing dates.
7. **Slot decomposition with extreme hubs.** A maximum degree around 10⁴ means 10⁴ Python
   iterations per block. Measure; the fallback is to force `bincount` per date for hub
   epochs, which is also canonical.
8. **Hermitian clustering at N ≥ 2,000.** `eigh` is O(N³), about 0.7 s. A randomised block
   Krylov method with a convergence certificate is future work; do not ship an uncertified
   subspace method.

---

## 15. Caller (AGENTS.md: "No new public surface without a named caller in the engine, and a test in the engine that exercises it")

**Verified state.** `truepoint/src` has zero `panelary` imports. `diagnose/` is
firewalled from panelary. The AGENTS-named `truepoint/src/truepoint/quant/` does not exist.

* **M1: proposed caller `truepoint/src/truepoint/quant/`** (to be created per AGENTS.md's
  "what to build first"). A customer's feature pipeline that uses relationship or peer data
  is audited by resolving its relationship table as-of. `EdgeTable.validate()` findings
  (`edges:knowledge_equals_valid_from`, `edges:classification_single_vintage`) and
  knowledge-time-invariance failures become `Evidence(kind=STATIC_ANALYSIS)` through
  `ValidationReport.to_evidence()`, next to `CompileResult.to_json()`. This is the graph
  analogue of the as-of join AGENTS.md already prioritises. **Required before merge:** a
  truepoint test that constructs a small `EdgeTable` with a backfilled knowledge time and
  asserts that the finding appears in the report.
* **M6 (CD\*, Conley): possible caller `truepoint/src/truepoint/inference/errorbars.py`,**
  only if its owner decides that error bars over items sharing linked entities need
  dependence-robust inference. It is deterministic and LLM-free, so it is compatible with
  `score/` importing it. Not assumed.
* **M2–M5 and M7: no caller identified. Do not build.**
* Nothing here may be used to generate, store or reveal sealed ground-truth items. Any use
  inside truepoint's `generate/` is a separate sealed-side decision outside this plan.
  Under the STRATEGY non-funnel rule, where these capabilities appear in a recommendation
  they are disclosed as affiliated and sit alongside non-affiliated alternatives (networkx,
  PySAL).

---

## 16. References († = not verified this session; check before citing in docs)

* **Economic links and predictability**
  * Cohen, L. & Frazzini, A. (2008). Economic links and predictable returns. *JF* 63(4)†.
  * Menzly, L. & Ozbas, O. (2010). Market segmentation and cross-predictability of returns. *JF* 65(4)†.
  * Lee, C. M. C., Sun, S. T., Wang, R. & Zhang, R. (2019). Technological links and predictable returns. *JFE* 132(3)†. (SSRN 3036241 confirmed.)
  * Ali, U. & Hirshleifer, D. (2020). Shared analyst coverage: Unifying momentum spillover effects. *JFE* 136(3)†.
  * Parsons, C. A., Sabbatucci, R. & Titman, S. (2020). Geographic lead-lag effects. *RFS* 33(10)†.
  * Burt, A. & Hrdlicka, C. (2021). Where does the predictability from sorting on returns of economically linked firms come from? *JFQA* 56(8), 2634–2658. Verified.
* **Lead–lag**
  * Bennett, S., Cucuringu, M. & Reinert, G. (2022). Lead–lag detection and network clustering for multivariate time series with an application to the US equity market. *Machine Learning* 111(12), 4497–4538; arXiv:2201.08283. Verified (§3.1–3.4).
  * Cucuringu, M., Li, H., Sun, H. & Zanetti, L. (2020). Hermitian matrices for clustering directed graphs. *AISTATS*†.
  * Levin, D., Lyons, T. & Ni, H. (2016). Learning from the past, predicting the statistics for the future, learning an evolving system. arXiv:1309.0260†.
  * Huber, P. J. (1962). Pairwise comparison and ranking. *Ann. Math. Stat.*†.
* **Cross-sectional dependence**
  * Pesaran, M. H. & Xie, Y. How to detect network dependence in latent factor models? A bias-corrected CD test. arXiv:2109.00408 (v8, 2026). Formula verified; journal†.
  * Pesaran, M. H. (2004/2015). General diagnostic tests for cross-sectional dependence in panels.
  * Juodis, A. & Reese, S. (2022). *JBES*†.
  * Bailey, N., Kapetanios, G. & Pesaran, M. H. (2016). Exponent of cross-sectional dependence: estimation and inference. *JAE* 31(6), 929–960. Verified; formula details†.
  * Bailey, N., Holly, S. & Pesaran, M. H. (2016). A two-stage approach to spatio-temporal analysis with strong and weak cross-sectional dependence. *JAE* 31(1)†. Title verified.
  * Bailey, N., Pesaran, M. H. & Smith, L. V. (2019). A multiple testing approach to the regularisation of large sample correlation matrices. *JoE* 208(2)†.
* **Networks, spatial statistics and estimators**
  * Zhu, X., Pan, R., Li, G., Liu, Y. & Wang, H. (2017). Network vector autoregression. *Ann. Statist.* 45(3)†.
  * Knight, M., Leeming, K., Nason, G. & Nunes, M. (2020). Generalized network autoregressive processes and the GNAR package. *JSS* 96(5)†.
  * Herskovic, B. (2018). Networks in production: Asset pricing implications. *JF* 73(4)†. Formulas verified from the author's Dec-2015 LSE handout.
  * Mantegna, R. N. (1999). Hierarchical structure in financial markets. *EPJ B* 11†.
  * Massara, G. P., Di Matteo, T. & Aste, T. (2017). Network filtering for big data: Triangulated Maximally Filtered Graph. *J. Complex Networks* 5(2)†.
  * Page, L., Brin, S., Motwani, R. & Winograd, T. (1999). PageRank†.
  * Katz, L. (1953). *Psychometrika* 18†.
  * Wu, F. et al. (2019). Simplifying graph convolutional networks. *ICML*†.
  * Frasca, F. et al. (2020). SIGN: Scalable inception graph neural networks†.
  * Moran, P. A. P. (1950). *Biometrika*†.
  * Geary, R. C. (1954)†.
  * Cliff, A. D. & Ord, J. K. (1981). *Spatial Processes*†.
  * Anselin, L. (1995). Local indicators of spatial association—LISA. *Geogr. Anal.* 27(2)†.
  * Sokal, R. R., Oden, N. L. & Thomson, B. A. (1998). Local spatial autocorrelation in a biological model. *Geogr. Anal.* 30(4)†.
  * Getis, A. & Ord, J. K. (1992), *Geogr. Anal.* 24(3); Ord, J. K. & Getis, A. (1995), *Geogr. Anal.* 27(4)†.
  * Ord, K. (1975). *JASA* 70†.
  * Pace, R. K. & LeSage, J. P. (2004). Chebyshev approximation of log-determinants of spatial weight matrices. *CSDA* 45(2)†.
* **Inference and identification**
  * Conley, T. G. (1999). GMM estimation with cross sectional dependence. *JoE* 92(1)†.
  * Driscoll, J. & Kraay, A. (1998). *ReStat* 80(4).
  * Petersen, M. (2009). Estimating standard errors in finance panel data sets. *RFS* 22(1)†.
  * Manski, C. (1993). Identification of endogenous social effects: The reflection problem. *ReStud* 60(3)†.
