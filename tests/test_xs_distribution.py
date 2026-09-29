"""The single-date cross-sectional distribution ops: ``.xs.dispersion``,
``.xs.tail_index``, ``.xs.up_share`` and ``.xs.entropy``.

Plan ``covariance-and-market-state`` sections 4.4, 5.15 and 7.6:

* every op against a numpy reference, date by date;
* ``.over([date, sector])`` against a manual group loop;
* the NaN / null policy (NaN is missing, exactly like null) and the degenerate
  cross-sections (empty, one name, all zero, constant);
* ``.xs.tail_index`` against the existing Hill oracle
  :func:`panelary.econ.features._evt.hill_index` with ``k`` matched, to 1e-12;
* prefix invariance and no look-ahead, **bitwise** (``tol=0.0``), under
  ``.over(time)`` and ``.over([time, group])``, on a panel whose names enter and
  leave;
* trap T11: evaluated without ``.over`` the ops pool every date, and the
  look-ahead verifier must catch that.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import panelary  # noqa: F401  -- registers the .xs namespace
from panelary.core.panel_frame import PanelFrame
from panelary.econ.features._evt import hill_index
from panelary.registry import registry
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

_MAD = 1.482602218505602
_RTOL = 1e-12


def _panel(
    *, n_dates: int = 10, n_names: int = 300, seed: int = 7, holes: bool = True
) -> pl.DataFrame:
    """Heavy-tailed returns with three sectors, names entering and leaving,
    a few NaNs and nulls, and a sector relabelling half-way through."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n_dates // 2, n_names)
    stop = rng.integers(n_dates // 2 + 1, n_dates + 1, n_names)
    stop[: n_names // 2] = n_dates  # half the names survive to the end
    sector = rng.integers(0, 3, n_names)
    ent, tim, ret, sec = [], [], [], []
    for i in range(n_names):
        for t in range(int(start[i]), int(stop[i])):
            ent.append(f"N{i:04d}")
            tim.append(t)
            ret.append(float(rng.standard_t(3) * 0.02))
            sec.append(f"s{(sector[i] + (t >= n_dates // 2) * (i % 4 == 0)) % 3}")
    df = pl.DataFrame({"entity": ent, "time": tim, "ret": ret, "sector": sec})
    if holes:
        idx = pl.int_range(pl.len())
        df = df.with_columns(
            pl.when(idx % 37 == 5)
            .then(float("nan"))
            .when(idx % 41 == 3)
            .then(None)
            .otherwise(pl.col("ret"))
            .alias("ret")
        )
    return df


def _finite(values: pl.Series) -> np.ndarray:
    arr = values.cast(pl.Float64).to_numpy()
    return arr[np.isfinite(arr)]


def _per_key(df: pl.DataFrame, expr: pl.Expr, keys: list[str]) -> dict:
    """``{key tuple: value}`` of an op evaluated ``.over(keys)``, one per group."""
    out = df.with_columns(expr.over(keys).alias("__o"))
    first = out.group_by(keys, maintain_order=True).agg(
        pl.col("__o").first(), pl.col("__o").n_unique().alias("__u")
    )
    assert (first["__u"] == 1).all(), "an .xs summary must be constant per group"
    return {
        tuple(row[k] for k in keys): row["__o"] for row in first.iter_rows(named=True)
    }


def _groups(df: pl.DataFrame, keys: list[str]):
    for key, part in df.group_by(keys, maintain_order=True):
        yield tuple(key), _finite(part["ret"])


def _close(got: float | None, want: float | None) -> bool:
    if want is None or (isinstance(want, float) and math.isnan(want)):
        return got is None
    return got is not None and math.isclose(got, want, rel_tol=_RTOL, abs_tol=1e-15)


# --------------------------------------------------------------------------- #
# numpy references
# --------------------------------------------------------------------------- #
def _ref_dispersion(v: np.ndarray, kind: str) -> float | None:
    if v.size == 0:
        return None
    if kind == "sd":
        return float(np.std(v, ddof=1)) if v.size >= 2 else None
    if kind == "mad":
        return float(_MAD * np.median(np.abs(v - np.median(v))))
    lo, hi = (0.25, 0.75) if kind == "iqr" else (0.1, 0.9)
    return float(np.quantile(v, hi) - np.quantile(v, lo))


def _ref_tail(v: np.ndarray, q: float, tail: str, min_exc: int) -> float | None:
    n = v.size
    k = math.floor(q * n)
    if k < min_exc:
        return None
    loss = -v if tail == "lower" else v
    u = np.sort(loss)[::-1][k]
    if not u > 0:
        return None
    return float(hill_index(v, k=k, tail=tail))


def _ref_up(v: np.ndarray) -> float | None:
    return float(np.mean(v > 0)) if v.size else None


def _ref_entropy_share(v: np.ndarray) -> float | None:
    a = np.abs(v)
    if v.size < 2 or a.sum() == 0:
        return None
    p = a / a.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(v.size))


def _ref_entropy_hist(v: np.ndarray) -> float | None:
    n = v.size
    if n == 0:
        return None
    width = 2.0 * (np.quantile(v, 0.75) - np.quantile(v, 0.25)) / n ** (1.0 / 3.0)
    if not width > 0:
        return None
    lo = v.min()
    n_bins = max(1.0, math.ceil((v.max() - lo) / width))
    if n_bins < 2:
        return None
    b = np.clip(np.floor((v - lo) / width), 0, n_bins - 1).astype(np.int64)
    p = np.bincount(b) / n
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n_bins))


_REFS = {
    "sd": (lambda: pl.col("ret").xs.dispersion(), lambda v: _ref_dispersion(v, "sd")),
    "mad": (
        lambda: pl.col("ret").xs.dispersion(kind="mad"),
        lambda v: _ref_dispersion(v, "mad"),
    ),
    "iqr": (
        lambda: pl.col("ret").xs.dispersion(kind="iqr"),
        lambda v: _ref_dispersion(v, "iqr"),
    ),
    "idr": (
        lambda: pl.col("ret").xs.dispersion(kind="idr"),
        lambda v: _ref_dispersion(v, "idr"),
    ),
    "tail_lower": (
        lambda: pl.col("ret").xs.tail_index(),
        lambda v: _ref_tail(v, 0.05, "lower", 10),
    ),
    "tail_upper": (
        lambda: pl.col("ret").xs.tail_index(q=0.08, tail="upper", min_exceedances=5),
        lambda v: _ref_tail(v, 0.08, "upper", 5),
    ),
    "up_share": (lambda: pl.col("ret").xs.up_share(), _ref_up),
    "entropy_share": (lambda: pl.col("ret").xs.entropy(), _ref_entropy_share),
    "entropy_hist": (
        lambda: pl.col("ret").xs.entropy(kind="hist"),
        _ref_entropy_hist,
    ),
}


# --------------------------------------------------------------------------- #
# 1. Against numpy, per date and per (date, group)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_REFS))
def test_matches_numpy_per_date(name: str) -> None:
    df = _panel()
    build, ref = _REFS[name]
    got = _per_key(df, build(), ["time"])
    checked = 0
    for key, v in _groups(df, ["time"]):
        want = ref(v)
        assert _close(got[key], want), (name, key, got[key], want)
        checked += want is not None
    assert checked >= 5, f"{name}: the reference is defined on too few dates"


@pytest.mark.parametrize("name", sorted(_REFS))
def test_group_variant_matches_manual_loop(name: str) -> None:
    """``.over([date, sector])`` is the op on each date-and-sector sub-sample."""
    df = _panel(n_names=900)
    build, ref = _REFS[name]
    keys = ["time", "sector"]
    got = _per_key(df, build(), keys)
    checked = 0
    for key, v in _groups(df, keys):
        want = ref(v)
        assert _close(got[key], want), (name, key, got[key], want)
        checked += want is not None
    assert checked >= 5


def test_tail_index_is_hill_with_matched_k_exactly() -> None:
    """The oracle row of plan section 7.3: ``hill_index`` with ``k`` matched."""
    rng = np.random.default_rng(3)
    for n in (200, 257, 1000, 3001):
        v = rng.standard_t(4, n)
        df = pl.DataFrame({"d": [0] * n, "r": v})
        for q in (0.05, 0.1):
            got = df.select(pl.col("r").xs.tail_index(q=q).over("d")).item(0, 0)
            k = math.floor(q * n)
            want = hill_index(v, k=k, tail="lower")
            assert math.isclose(got, want, rel_tol=1e-12), (n, q, got, want)


# --------------------------------------------------------------------------- #
# 2. NaN / null policy and degenerate cross-sections
# --------------------------------------------------------------------------- #
def _all_ops() -> dict[str, pl.Expr]:
    return {name: build() for name, (build, _ref) in _REFS.items()}


def test_nan_is_treated_exactly_like_null() -> None:
    df = _panel()
    as_null = df.with_columns(pl.col("ret").fill_nan(None))
    assert df["ret"].is_nan().sum() > 0
    for name, expr in _all_ops().items():
        a = df.select(expr.over("time").alias("o"))["o"]
        b = as_null.select(expr.over("time").alias("o"))["o"]
        assert a.equals(b), name


def test_degenerate_cross_sections_are_null_not_nan() -> None:
    df = pl.DataFrame(
        {
            "d": [0, 0, 1, 2, 2, 2, 3, 3, 3],
            "ret": [None, float("nan"), 0.5, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
        }
    )
    out = (
        df.with_columns([e.over("d").alias(n) for n, e in _all_ops().items()])
        .group_by("d", maintain_order=True)
        .first()
    )
    rows = {r["d"]: r for r in out.iter_rows(named=True)}
    # date 0: nothing observed -> every summary is null
    assert all(rows[0][n] is None for n in _REFS)
    # date 1: one name
    assert rows[1]["sd"] is None
    assert rows[1]["mad"] == 0.0
    assert rows[1]["up_share"] == 1.0
    assert rows[1]["entropy_share"] is None
    # date 2: all zero -> no move to share out, nobody up
    assert rows[2]["entropy_share"] is None
    assert rows[2]["up_share"] == 0.0
    assert rows[2]["entropy_hist"] is None  # zero IQR: no bin width
    # date 3: constant -> zero dispersion, still no bin width
    assert rows[3]["sd"] == 0.0
    assert rows[3]["iqr"] == 0.0
    assert rows[3]["entropy_hist"] is None
    for n in _REFS:
        assert not out[n].is_nan().any(), n


def test_integer_input_is_upcast_to_float64() -> None:
    df = pl.DataFrame({"d": [0, 0, 0, 0], "ret": [1, -2, 3, 4]})
    for name, expr in _all_ops().items():
        out = df.select(expr.over("d").alias("o"))
        assert out.schema["o"] == pl.Float64, name


def test_tail_index_threshold_must_be_a_loss() -> None:
    """With too few negative returns the threshold ``u`` is not a loss: null,
    rather than ``hill_index``'s documented shift-the-sample fallback."""
    v = np.linspace(0.01, 1.0, 400)  # no losses at all
    df = pl.DataFrame({"d": [0] * v.size, "r": v})
    assert df.select(pl.col("r").xs.tail_index().over("d")).item(0, 0) is None


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: pl.col("r").xs.dispersion(kind="var"), "kind"),
        (lambda: pl.col("r").xs.tail_index(q=0.0), "q"),
        (lambda: pl.col("r").xs.tail_index(q=1.0), "q"),
        (lambda: pl.col("r").xs.tail_index(tail="left"), "tail"),
        (lambda: pl.col("r").xs.tail_index(min_exceedances=0), "min_exceedances"),
        (lambda: pl.col("r").xs.tail_index(min_exceedances=2.5), "min_exceedances"),
        (lambda: pl.col("r").xs.entropy(kind="kde"), "kind"),
    ],
)
def test_bad_arguments_raise(build, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build()


# --------------------------------------------------------------------------- #
# 3. Leak safety: bitwise, grouped, and trap T11
# --------------------------------------------------------------------------- #
def _leak_expr(name: str) -> pl.Expr:
    """The op under test; the Hill tail gets a larger ``q`` so that the
    per-sector cross-sections (about 130 names) still produce values."""
    if name == "tail_lower":
        return pl.col("ret").xs.tail_index(q=0.2, min_exceedances=3)
    return _REFS[name][0]()


@pytest.mark.parametrize("over", [["time"], ["time", "sector"]], ids=["date", "group"])
@pytest.mark.parametrize("name", sorted(_REFS))
def test_prefix_invariant_and_causal_bitwise(name: str, over: list[str]) -> None:
    """Names enter and leave; ``tol=0.0`` at the default cuts and at every date."""
    df = _panel(n_dates=8, n_names=400)
    pf = PanelFrame(df, entity="entity", time="time")
    expr = _leak_expr(name).over(over).alias("o")
    assert df.select(expr)["o"].drop_nulls().len() > 0, "vacuous: all null"
    assert_prefix_invariant(expr, pf, tol=0.0)
    assert_no_lookahead(expr, pf, tol=0.0)
    for cut in range(0, 7):
        assert_prefix_invariant(expr, pf, tol=0.0, cut=cut)


@pytest.mark.parametrize("name", sorted(_REFS))
def test_bare_evaluation_is_a_lookahead_t11(name: str) -> None:
    """Trap T11: without ``.over(time)`` the "cross-section" is the whole panel,
    so every value depends on the future. The verifier must say so -- the
    instrument is sharp -- and the documented composition must pass."""
    df = _panel(n_dates=8, n_names=400)
    pf = PanelFrame(df, entity="entity", time="time")
    bare = _REFS[name][0]().alias("o")
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_no_lookahead(bare, pf)
    assert_no_lookahead(_REFS[name][0]().over("time").alias("o"), pf)


# --------------------------------------------------------------------------- #
# 4. Registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("spec_name", "method"),
    [
        ("xs_dispersion", "dispersion"),
        ("xs_tail_index", "tail_index"),
        ("xs_up_share", "up_share"),
        ("xs_entropy", "entropy"),
    ],
)
def test_registered_under_prefixed_names(spec_name: str, method: str) -> None:
    spec = registry.get(spec_name)
    assert spec.namespace == "xs"
    assert spec.safe_scope == "rowwise"
    assert spec.panel_safe is False and spec.leakage_safe is True
    assert spec.tier == "B"
    assert hasattr(pl.col("x").xs, method)
    # the bare names stay free for per-entity (time-series) siblings
    assert method not in {s.name for s in registry.all() if s.namespace == "xs"}
