"""The leak-safety acceptance suite for :mod:`panelary.detect`.

Five generic invariants, applied mechanically to **every** public entry point,
plus the two API-shape guards. These are the point of the module: a bubble
detector that is merely "roughly right" but peeks at the future is worse than
no detector at all, because it back-tests beautifully and trades catastrophically.

1. **Prefix invariance** -- ``f(y[:T])[t] == f(y[:T+k])[t]`` at every shared
   ``t``, over 15 expanding cut points. One test, three distinct leaks: a
   window rule of the form ``min_window = f(T)``, a critical value indexed by
   the caller's own sample size, and full-sample lag selection. With the window
   rule frozen the measured deviation is exactly ``0.0``; with the
   conventional ``floor(T*(0.01 + 1.8/sqrt(T)))`` rule it is mean 0.123 /
   max 2.515, revising 41.2% of the dates in the series.
2. **Future poison** -- overwrite everything after ``t`` with noise, then NaN,
   then a huge constant; the value at ``t`` must be bit-identical. Strictly
   stronger than (1): (1) can be satisfied by an implementation that reads a
   full-sample array and happens to be re-derived per prefix, (2) cannot.
3. **Determinism** -- twice in-process and once in a fresh subprocess, byte for
   byte. Catches unseeded generators and inherited worker state.
4. **Entity-shuffle invariance** -- permuting entities must permute the output
   and nothing else. Catches cross-sectional contamination.
5. **Level-shift invariance** -- adding up to 1e6 must not move the answer.
   Guards the ``y - y[0]`` anchoring in ``cumulative_moments``: raw prefix sums
   degrade to 4.4e-4 at an offset of 5e6, while anchored sums stay at 2.2e-11.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from tests import test_detect_support as S

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 15 expanding cut points; the first is comfortably past ``MIN_W`` so every
#: entry point has produced at least a handful of finite values.
CUTS = list(range(S.MIN_W + 25, S.N_LONG + 1, 10))[:15]


def _cases() -> list[str]:
    """Parametrisation ids; a sentinel keeps the test visible when nothing landed."""
    return S.case_names() or ["<no-entry-points-importable>"]


def _get(name: str):
    if name.startswith("<"):
        S.require("_bsadf", "_monitors")
        pytest.fail(f"no public entry points registered: {name}")
    return S.build_cases()[name]


# --------------------------------------------------------------------------- #
# 0. The registry itself
# --------------------------------------------------------------------------- #
def test_entry_point_registry_is_populated() -> None:
    """Every invariant below is parametrised over this registry; it must be full."""
    S.require("_bsadf", "_monitors")
    names = S.case_names()
    assert names, (
        "no `detect` entry points were registered -- the generic leak invariants "
        "would silently cover nothing."
    )
    assert any(n.startswith("bsadf_") for n in names)


# --------------------------------------------------------------------------- #
# 1. Prefix invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _cases())
def test_prefix_invariance(name: str) -> None:
    """Truncating the future must not move a single past value."""
    fn = _get(name)
    y = S.bubble_series(S.N_LONG, seed=7)

    outs = {c: np.asarray(fn(y[:c]), dtype=np.float64) for c in CUTS}

    worst = 0.0
    worst_at = None
    for i, c_short in enumerate(CUTS):
        short = outs[c_short]
        for c_long in CUTS[i + 1 :]:
            long = outs[c_long][: short.size]
            dev = S.max_abs_dev(short, long)
            if dev > worst:
                worst, worst_at = dev, (c_short, c_long)

    assert worst < 1e-10, (
        f"{name}: values computed on y[:{worst_at[0]}] differ from the same "
        f"positions computed on y[:{worst_at[1]}] by {worst:.6g}. The feature "
        "depends on len(y): look for `min_window = f(T)`, a critical value "
        "indexed by the caller's sample size, or full-sample lag selection."
    )


@pytest.mark.parametrize("name", _cases())
def test_prefix_invariance_is_exact_not_merely_close(name: str) -> None:
    """With the window rule frozen the deviation is *exactly* zero, not ~1e-12."""
    fn = _get(name)
    y = S.bubble_series(S.N_LONG, seed=7)
    short = np.asarray(fn(y[: CUTS[0]]), dtype=np.float64)
    long = np.asarray(fn(y), dtype=np.float64)[: short.size]
    assert S.bit_identical(short, long), (
        f"{name}: the prefix and the full sample agree only approximately. A "
        "leak-free implementation reads nothing but the prefix, so the two must "
        f"be bit-identical (max dev {S.max_abs_dev(short, long):.6g})."
    )


# --------------------------------------------------------------------------- #
# 2. Future poison
# --------------------------------------------------------------------------- #
def _poisoned(y: np.ndarray, t: int, kind: str) -> np.ndarray:
    out = y.copy()
    tail = out.size - (t + 1)
    if kind == "noise":
        out[t + 1 :] = np.random.default_rng(99).standard_normal(tail) * 50.0
    elif kind == "nan":
        out[t + 1 :] = np.nan
    elif kind == "huge":
        out[t + 1 :] = 1e12
    else:  # pragma: no cover - guarded by the parametrisation
        raise ValueError(kind)
    return out


@pytest.mark.parametrize("kind", ["noise", "nan", "huge"])
@pytest.mark.parametrize("name", _cases())
def test_future_poison(name: str, kind: str) -> None:
    """Corrupting the future must leave every past value bit-identical."""
    fn = _get(name)
    y = S.bubble_series(S.N_LONG, seed=7)
    clean = np.asarray(fn(y), dtype=np.float64)

    for t in (S.MIN_W + 20, S.MIN_W + 70, S.N_LONG - 40):
        got = np.asarray(fn(_poisoned(y, t, kind)), dtype=np.float64)
        n = min(t + 1, clean.size, got.size)
        assert S.bit_identical(clean[:n], got[:n]), (
            f"{name}: replacing y[{t + 1}:] with {kind!r} changed values at or "
            f"before t={t} (max dev {S.max_abs_dev(clean[:n], got[:n]):.6g}). "
            "Something reads a full-sample array."
        )


# --------------------------------------------------------------------------- #
# 3. Determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _cases())
def test_determinism_same_process(name: str) -> None:
    fn = _get(name)
    y = S.random_walk(S.N_LONG, seed=0)
    a = np.asarray(fn(y), dtype=np.float64)
    b = np.asarray(fn(y), dtype=np.float64)
    assert S.bit_identical(a, b), f"{name}: two calls in one process disagreed."


_SUBPROC = r"""
import hashlib, json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tests!r})
import numpy as np
import test_detect_support as S

out = {{}}
for name in S.case_names():
    arr = np.ascontiguousarray(S.run_case(name), dtype=np.float64)
    out[name] = hashlib.sha256(arr.tobytes()).hexdigest()
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def subprocess_hashes() -> dict[str, str]:
    """SHA-256 of every case, computed in a brand-new interpreter."""
    S.require("_bsadf", "_monitors")
    script = _SUBPROC.format(root=_REPO_ROOT, tests=os.path.join(_REPO_ROOT, "tests"))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"determinism probe subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", _cases())
def test_determinism_fresh_subprocess(name: str, subprocess_hashes) -> None:
    """A fresh interpreter must produce byte-identical output."""
    fn = _get(name)
    arr = np.ascontiguousarray(
        np.asarray(fn(S.random_walk(S.N_LONG, seed=0)), dtype=np.float64)
    )
    here = hashlib.sha256(arr.tobytes()).hexdigest()
    assert name in subprocess_hashes, f"{name} did not register in the subprocess"
    assert here == subprocess_hashes[name], (
        f"{name}: output differs between this process and a fresh one. Look for "
        "an unseeded RNG, hash-order iteration, or state inherited from a worker."
    )


# --------------------------------------------------------------------------- #
# 4. Entity-shuffle invariance
# --------------------------------------------------------------------------- #
def test_entity_shuffle_invariance_bsadf_panel() -> None:
    """Permuting entity rows permutes the output and changes nothing else."""
    fn = S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")
    mat = S.panel_matrix(n_entities=10, n_time=140, seed=11)
    base = np.asarray(fn(mat, min_window=S.MIN_W, lag=0), dtype=np.float64)
    assert base.shape == mat.shape, (
        f"bsadf_panel returned {base.shape}, expected {mat.shape}"
    )

    perm = np.random.default_rng(5).permutation(mat.shape[0])
    shuffled = np.asarray(fn(mat[perm], min_window=S.MIN_W, lag=0), dtype=np.float64)
    assert S.bit_identical(base[perm], shuffled), (
        "bsadf_panel is not equivariant to entity order: entity i's series "
        "produced a different answer once its neighbours moved. Something is "
        "pooling across the cross-section."
    )


def test_entity_shuffle_invariance_single_series_matches_panel() -> None:
    """Each panel row must equal the same entity run entirely on its own."""
    seq = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    pan = S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")
    mat = S.panel_matrix(n_entities=6, n_time=140, seed=12)
    got = np.asarray(pan(mat, min_window=S.MIN_W, lag=0), dtype=np.float64)
    for i in range(mat.shape[0]):
        alone = np.asarray(seq(mat[i], min_window=S.MIN_W, lag=0), dtype=np.float64)
        assert S.bit_identical(got[i], alone), (
            f"panel row {i} differs from the same series computed alone "
            f"(max dev {S.max_abs_dev(got[i], alone):.6g}) -- cross-sectional "
            "contamination."
        )


def test_row_shuffle_within_frame_is_irrelevant() -> None:
    """Row order inside the frame must not change the per-entity answer.

    The panel API is array-shaped, so the frame-level analogue is: build the
    same panel from rows presented in a scrambled order, sort it back into
    ``(entity, time)`` order, and require the same matrix.
    """
    fn = S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")
    mat = S.panel_matrix(n_entities=5, n_time=120, seed=13)
    n_ent, n_time = mat.shape
    df = pl.DataFrame(
        {
            "entity": np.repeat(np.arange(n_ent), n_time),
            "time": np.tile(np.arange(n_time), n_ent),
            "y": mat.reshape(-1),
        }
    )
    scrambled = df.sample(fraction=1.0, shuffle=True, seed=4)
    rebuilt = (
        scrambled.sort(["entity", "time"])
        .get_column("y")
        .to_numpy()
        .reshape(n_ent, n_time)
    )
    assert S.bit_identical(rebuilt, mat)
    a = np.asarray(fn(mat, min_window=S.MIN_W, lag=0), dtype=np.float64)
    b = np.asarray(fn(rebuilt, min_window=S.MIN_W, lag=0), dtype=np.float64)
    assert S.bit_identical(a, b)


# --------------------------------------------------------------------------- #
# 5. Level-shift invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("offset", [1e2, 1e4, 1e6])
@pytest.mark.parametrize(
    "name",
    [n for n in _cases() if n.startswith(S.LEVEL_INVARIANT_PREFIXES)] or ["<none>"],
)
def test_level_shift_invariance(name: str, offset: float) -> None:
    """An ADF with an intercept is level-free; drift here is lost precision."""
    fn = _get(name)
    y = S.bubble_series(S.N_LONG, seed=7)
    base = np.asarray(fn(y), dtype=np.float64)
    shifted = np.asarray(fn(y + offset), dtype=np.float64)
    dev = S.max_abs_dev(base, shifted)
    assert dev < 1e-9, (
        f"{name}: adding {offset:g} moved the statistic by {dev:.3g}. The "
        "prefix-sum accumulator is losing precision -- `cumulative_moments` "
        "must anchor `y -> y - y[0]` before accumulating."
    )


def test_level_shift_anchoring_survives_an_extreme_offset() -> None:
    """The measured failure mode: raw prefix sums hit 4.4e-4 at an offset of 5e6."""
    fn = S.require_attr(S.bsadf, "bsadf_sequence", "_bsadf")
    y = S.bubble_series(S.N_LONG, seed=7)
    base = np.asarray(fn(y, min_window=S.MIN_W, lag=0, grid=None), dtype=np.float64)
    far = np.asarray(
        fn(y + 5e6, min_window=S.MIN_W, lag=0, grid=None), dtype=np.float64
    )
    dev = S.max_abs_dev(base, far)
    assert dev < 1e-8, (
        f"offset 5e6 moved BSADF by {dev:.3g}; anchored sums measure 2.2e-11, "
        "un-anchored ones 4.4e-4."
    )


# --------------------------------------------------------------------------- #
# Refusals: things the public API must NOT expose
# --------------------------------------------------------------------------- #
#: A full-sample GSADF is a single number for the whole series -- broadcasting
#: it onto every row hands the model the answer. An episode *end* (and anything
#: derived from it, such as a duration) is only knowable after the episode has
#: finished, so it cannot be a feature at any row inside the episode.
FORBIDDEN_NAME_FRAGMENTS = (
    "episode_end",
    "episode_duration",
    "burst_end",
    "bubble_end",
    "collapse_date",
    "_duration",
    "end_idx",
    "end_date",
)

#: ``gsadf`` itself is *not* banned by name: it is a legitimate research
#: reduction. What is banned is a per-row GSADF, which
#: ``test_full_sample_gsadf_is_a_scalar_not_a_per_row_feature`` (in
#: ``test_detect_bsadf.py``) pins by asserting it returns a scalar.
NAME_EXEMPTIONS: frozenset[str] = frozenset()


def _forbidden(name: str) -> str | None:
    low = name.lower()
    if name in NAME_EXEMPTIONS:
        return None
    for frag in FORBIDDEN_NAME_FRAGMENTS:
        if frag in low:
            return frag
    return None


def test_no_public_name_advertises_an_episode_or_a_duration() -> None:
    """Scan every public name across the whole module, not just the package."""
    S.require("_bsadf", "_monitors", "_panel")
    offenders = []
    for modname in S.SUBMODULES:
        mod = S._MODULES[modname]
        if mod is None:
            continue
        names = getattr(mod, "__all__", None) or [
            n for n in dir(mod) if not n.startswith("_")
        ]
        offenders += [
            (f"{modname}.{n}", frag) for n in names if (frag := _forbidden(n))
        ]
    if S.detect_pkg is not None:
        pkg = S.detect_pkg
        names = getattr(pkg, "__all__", None) or [
            n for n in dir(pkg) if not n.startswith("_")
        ]
        offenders += [(n, frag) for n in names if (frag := _forbidden(n))]

    assert not offenders, (
        "panelary.detect exposes "
        + ", ".join(f"{n!r} (matched {f!r})" for n, f in offenders)
        + ". An episode end, duration or membership flag back-dates the "
        "episode's termination -- information from after t -- into row t."
    )


def test_the_tidy_feature_frame_has_no_forward_looking_column() -> None:
    """``panel_features`` is the shipped feature schema; scan it directly."""
    fn = S.require_attr(S.panel, "panel_features", "_panel")
    S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")

    mat = S.panel_matrix(n_entities=5, n_time=160, seed=18)
    stat = np.asarray(
        S.bsadf.bsadf_panel(mat, min_window=S.MIN_W, lag=0), dtype=np.float64
    )
    frame = fn(stat, 1.5)
    cols = list(frame.columns)
    bad = [(c, f) for c in cols if (f := _forbidden(c))]
    assert not bad, (
        "panel_features emitted "
        + ", ".join(f"{c!r} (matched {f!r})" for c, f in bad)
        + "; every column must be knowable at its own row's date."
    )
    assert any(c.endswith("_run") for c in cols), (
        f"panel_features has no `*_run` column: {cols}. The leak-safe stand-in "
        "for an episode duration is the length of the exceedance run *ending* "
        "at t, and without it callers will reach for the real duration."
    )


def test_no_per_row_feature_is_constant_across_the_whole_series() -> None:
    """A broadcast full-sample number is the signature of a look-ahead feature."""
    fn = S.require_attr(S.panel, "panel_features", "_panel")
    S.require_attr(S.bsadf, "bsadf_panel", "_bsadf")

    # A panel that actually contains bubbles, so every column has something to
    # vary: a constant column then genuinely means a broadcast full-sample value.
    mat = np.vstack([S.bubble_series(200, seed=40 + i) for i in range(5)])
    stat = np.asarray(
        S.bsadf.bsadf_panel(mat, min_window=S.MIN_W, lag=0), dtype=np.float64
    )
    # A genuinely time-varying critical value, so a constant column can only
    # mean a full-sample number broadcast across the rows.
    cv = 1.2 + 0.4 * np.sin(np.arange(stat.shape[1]) / 9.0)
    frame = fn(stat, cv)
    for name, dtype in frame.schema.items():
        if name in {"entity", "date"} or not dtype.is_numeric():
            continue
        col = frame.get_column(name).drop_nulls().drop_nans()
        if col.len() < 50:
            continue
        assert col.n_unique() > 1, (
            f"column {name!r} is a single value repeated across all "
            f"{col.len()} rows. Either it is a full-sample statistic broadcast "
            "onto every row, or it carries no information."
        )


def test_transformer_output_schemas_contain_no_forward_looking_columns() -> None:
    """Fit every zero-argument transformer and scan the columns it produces."""
    S.require("detect")
    from panelary.core import PanelFrame
    from panelary.core.protocol import PanelTransformer

    pkg = S.detect_pkg
    classes = [
        obj
        for name in (getattr(pkg, "__all__", None) or dir(pkg))
        if isinstance(obj := getattr(pkg, name, None), type)
        and issubclass(obj, PanelTransformer)
        and obj is not PanelTransformer
    ]
    if not classes:
        pytest.skip("panelary.detect exports no PanelTransformer subclasses yet")

    n_ent, n_time = 4, 160
    mat = S.panel_matrix(n_entities=n_ent, n_time=n_time, seed=17)
    frame = pl.DataFrame(
        {
            "entity": np.repeat([f"e{i}" for i in range(n_ent)], n_time),
            "time": np.tile(np.arange(n_time), n_ent),
            "y": mat.reshape(-1),
        }
    )
    pf = PanelFrame(frame, entity="entity", time="time")

    for cls in classes:
        assert cls.panel_safe is True, f"{cls.__name__}.panel_safe must be True"
        assert cls.leakage_safe is True, f"{cls.__name__}.leakage_safe must be True"
        try:
            obj = cls()
        except TypeError:
            continue  # needs constructor arguments; the name check above still ran
        try:
            out = obj.fit_transform(pf)
        except Exception as exc:  # pragma: no cover - reported, not swallowed
            pytest.fail(f"{cls.__name__}().fit_transform(panel) raised {exc!r}")
        cols = list(
            out.collect_schema().names()
            if hasattr(out, "collect_schema")
            else out.columns
        )
        bad = [(c, f) for c in cols if (f := _forbidden(c))]
        assert not bad, (
            f"{cls.__name__} produced forward-looking column(s) "
            + ", ".join(f"{c!r} (matched {f!r})" for c, f in bad)
        )


# --------------------------------------------------------------------------- #
# Import hygiene (mirrors tests/test_import_hygiene.py)
# --------------------------------------------------------------------------- #
DETECT_FORBIDDEN = ("scipy", "sklearn", "statsmodels", "numba", "matplotlib", "pandas")

_HYGIENE_PROBE = r"""
import json, sys
import importlib
for name in {mods!r}:
    importlib.import_module(name)
print(json.dumps(sorted({{n.split(".", 1)[0] for n in sys.modules}})))
"""


def test_detect_import_is_clean() -> None:
    """Importing ``detect`` -- and every submodule -- stays numpy + polars only."""
    S.require("_moments", "_bsadf", "_critvals", "_monitors", "_panel")
    mods = [f"panelary.detect.{m}" for m in S.SUBMODULES]
    if S.detect_pkg is not None:
        mods.insert(0, "panelary.detect")
    proc = subprocess.run(
        [sys.executable, "-c", _HYGIENE_PROBE.format(mods=mods)],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"import probe failed:\n{proc.stdout}\n{proc.stderr}"
    tops = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    leaked = sorted(tops & set(DETECT_FORBIDDEN))
    assert not leaked, (
        f"importing {mods} eagerly pulled in {leaked}. The detect module is "
        "contracted to pure numpy + polars."
    )
