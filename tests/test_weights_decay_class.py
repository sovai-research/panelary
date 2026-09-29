"""Return attribution, time decay, class weights and `attach` (AFML 4.10-4.11)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary.core._spans import _spans_from_t1
from panelary.weights._weights import _time_decay
from tests import _spans_reference as ref


def _labelled(seed: int = 0, n_ent: int = 3, n_t: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_ent):
        n = n_t - int(rng.integers(0, 15))
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        t1 = [
            int(t + rng.integers(0, 8)) if rng.random() > 0.1 else None
            for t in range(n)
        ]
        frames.append(
            pl.DataFrame(
                {
                    "id": [f"e{e}"] * n,
                    "t": np.arange(n),
                    "close": close,
                    "t1": t1,
                    "label": rng.choice([-1, 0, 1], n, p=[0.3, 0.5, 0.2]),
                },
                schema_overrides={"t1": pl.Int64},
            )
        )
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# Return attribution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("afml_compat", [False, True])
def test_return_attribution_matches_brute_force(seed: int, afml_compat: bool) -> None:
    df = _labelled(seed)
    out = pn.weights.return_attribution(
        df, t1="t1", price="close", afml_compat=afml_compat, entity="id", time="t"
    )
    sorted_df = df.sort(["id", "t"])
    spans = ref.ref_spans(sorted_df, entity="id", time="t", t1="t1")
    c = ref.ref_concurrency(spans, sorted_df.height)
    p = sorted_df.get_column("close").to_numpy()
    ids = sorted_df.get_column("id").to_list()
    r = [0.0] + [
        math.log(p[i] / p[i - 1]) if ids[i] == ids[i - 1] else 0.0
        for i in range(1, len(p))
    ]
    off = 0 if afml_compat else 1
    want = [
        abs(math.fsum(r[k] / c[k] for k in range(s + off, e + 1))) for s, e in spans
    ]
    got = out.get_column("w_ret").to_numpy()[[s for s, _ in spans]]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_return_attribution_nan_price_poisons_only_its_spans() -> None:
    df = pl.DataFrame(
        {
            "id": ["a"] * 8,
            "t": list(range(8)),
            "close": [1.0, 1.1, None, 1.2, 1.3, 1.2, 1.25, 1.3],
            "t1": [1, 2, 3, 4, 5, 6, 7, 7],
        }
    )
    w = pn.weights.return_attribution(df, t1="t1", price="close").get_column("w_ret")
    # returns at rows 2 and 3 are NaN (price missing): spans (t0, t1] touching
    # them are NaN; the others stay finite.
    assert np.isnan(w.to_numpy()[[1, 2]]).all()
    assert np.isfinite(w.to_numpy()[[0, 3, 4, 5, 6]]).all()


# --------------------------------------------------------------------------- #
# Time decay
# --------------------------------------------------------------------------- #
def _decay(df: pl.DataFrame, c: float) -> tuple[np.ndarray, np.ndarray]:
    _, table = _spans_from_t1(df, t1="t1", entity="id", time="t")
    u = pn.weights.average_uniqueness(df, t1="t1", entity="id", time="t")
    u_lab = u.get_column("uniqueness").to_numpy()[table.label_row]
    order = np.lexsort((table.entity_code, table.start_tpos))
    return _time_decay(u_lab, table, c)[order], np.cumsum(u_lab[order])


def test_time_decay_shapes() -> None:
    df = _labelled(1)
    d, x = _decay(df, 1.0)
    assert np.all(d == 1.0)
    d, x = _decay(df, 0.0)
    np.testing.assert_allclose(d, x / x[-1], rtol=0, atol=1e-15)
    d, x = _decay(df, -0.5)
    assert np.all(d[x <= 0.5 * x[-1]] == 0.0) and np.all(d[x > 0.5 * x[-1] + 1e-12] > 0)
    for c in (0.9, 0.3, 0.0, -0.3):
        d, _ = _decay(df, c)
        assert abs(d[-1] - 1.0) <= 1e-15
        assert np.all(np.diff(d) >= -1e-15)  # monotone in time


def test_time_decay_frame_and_validation() -> None:
    df = _labelled(2)
    out = pn.weights.time_decay(df, t1="t1", c=0.25, entity="id", time="t")
    assert out.schema["w_decay"] == pl.Float64
    dropped = pn.weights.spans(df, t1="t1", entity="id", time="t").n_dropped
    assert out.get_column("w_decay").null_count() == dropped
    with pytest.raises(ValueError, match=r"\(-1, 1\]"):
        pn.weights.time_decay(df, t1="t1", c=-1.0)
    with pytest.raises(ValueError, match=r"\(-1, 1\]"):
        pn.weights.FoldWeights(decay=1.5)


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #
def test_class_weights_match_sklearn_balanced() -> None:
    sk = pytest.importorskip("sklearn.utils.class_weight")
    rng = np.random.default_rng(0)
    y = rng.choice([-1, 0, 1, 3], 500, p=[0.1, 0.5, 0.3, 0.1])
    got = pn.weights.class_weights(y)
    classes = np.unique(y)
    want = sk.compute_class_weight("balanced", classes=classes, y=y)
    assert list(got) == classes.tolist()
    np.testing.assert_allclose(list(got.values()), want, rtol=1e-15)


def test_class_weights_effective_mass_identity() -> None:
    rng = np.random.default_rng(1)
    y = rng.choice(["a", "b", "c"], 300)
    u = rng.uniform(0.05, 1.0, 300)
    w = pn.weights.class_weights(pl.Series(y), effective=u)
    # every class carries the same weighted mass n / K
    mass = {k: w[k] * u[y == k].sum() for k in w}
    np.testing.assert_allclose(list(mass.values()), [u.sum() / 3] * 3, rtol=1e-12)
    assert pn.weights.class_weights([None, 1.0, np.nan, 2.0]) == {1.0: 1.0, 2.0: 1.0}
    assert pn.weights.class_weights([]) == {}


# --------------------------------------------------------------------------- #
# attach (global)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["uniqueness", "return", None])
def test_attach_normalises_and_zeroes_non_labels(kind: str | None) -> None:
    df = _labelled(3)
    out = pn.weights.attach(
        df,
        t1="t1",
        kind=kind,
        price="close" if kind == "return" else None,
        decay=0.5,
        balance="label",
        entity="id",
        time="t",
    )
    _, table = _spans_from_t1(df, t1="t1", entity="id", time="t")
    w = out.get_column("w").to_numpy()
    assert w[table.label_row].sum() == pytest.approx(len(table), rel=1e-12)
    others = np.setdiff1d(np.arange(out.height), table.label_row)
    assert np.all(w[others] == 0.0)


def test_attach_is_the_product_of_its_parts() -> None:
    df = _labelled(4)
    kw = {"t1": "t1", "entity": "id", "time": "t"}
    _, table = _spans_from_t1(df, **kw)
    rows = table.label_row
    u = (
        pn.weights.average_uniqueness(df, **kw)
        .get_column("uniqueness")
        .to_numpy()[rows]
    )
    d = pn.weights.time_decay(df, c=0.2, **kw).get_column("w_decay").to_numpy()[rows]
    labels = df.sort(["id", "t"]).get_column("label").to_numpy()[rows]
    cw = pn.weights.class_weights(labels)
    raw = u * d * np.array([cw[v] for v in labels])
    want = raw * len(raw) / raw.sum()
    got = (
        pn.weights.attach(df, decay=0.2, balance="label", **kw)
        .get_column("w")
        .to_numpy()
    )
    np.testing.assert_allclose(got[rows], want, rtol=1e-13)


def test_attach_argument_validation() -> None:
    df = _labelled(5)
    with pytest.raises(ValueError, match="price"):
        pn.weights.attach(df, t1="t1", kind="return")
    with pytest.raises(ValueError, match="kind"):
        pn.weights.attach(df, t1="t1", kind="both")
    with pytest.raises(ValueError, match="balance column"):
        pn.weights.attach(df, t1="t1", balance="nope")
