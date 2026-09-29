"""AFML symmetric CUSUM filter and the tick rule."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import panelary as pn
from panelary._internal._jit import assert_backend_parity, force_numpy, numba_available
from panelary.sample._kernels import _cusum_filter
from panelary.testing import assert_no_lookahead, assert_prefix_invariant


def _definition(y, h):
    """AFML snippet 2.4, from the definition (NaN y skipped)."""
    out, sp, sn = [], 0.0, 0.0
    for yt, ht in zip(y, h, strict=True):
        if np.isnan(yt):
            out.append(False)
            continue
        sp, sn = max(0.0, sp + yt), min(0.0, sn + yt)
        if sn < -ht:
            sn = 0.0
            out.append(True)
        elif sp > ht:
            sp = 0.0
            out.append(True)
        else:
            out.append(False)
    return np.array(out, dtype=bool)


@pytest.mark.parametrize("seed", range(5))
def test_matches_definition_and_backends_agree(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, 5000)
    y[rng.random(5000) < 0.05] = np.nan  # T22: NaN-laden input
    h = np.abs(rng.normal(2, 1, 5000))
    h[rng.random(5000) < 0.02] = np.nan
    with force_numpy():
        twin = _cusum_filter(y, h)
    assert np.array_equal(twin, _definition(y, h))
    if numba_available():
        assert_backend_parity(_cusum_filter(y, h), twin)


def test_only_the_triggered_side_resets() -> None:
    y = np.array([0.6, 0.6, -0.1, 0.6, -2.0, 0.1])
    got = _cusum_filter(y, np.full(6, 1.0))
    # S+ : .6, 1.2 (fire, reset), 0, .6, 0, .1 ; S- : 0, 0, -.1, 0, -2 (fire), 0
    assert got.tolist() == [False, True, False, False, True, False]


def _panel(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120
    return pl.DataFrame(
        {
            "id": np.repeat(["a", "b", "c"], n),
            "t": np.tile(np.arange(n), 3),
            "ret": rng.normal(0, 0.01, 3 * n),
        }
    ).with_columns(
        (pl.col("ret").rolling_std(20).over("id") * 2).alias("h_trailing"),
        (pl.col("ret").std().over("id") * 2).alias("h_full"),
    )


def test_expression_over_entities_scalar_and_column_thresholds() -> None:
    df = _panel()
    out = df.with_columns(
        pn.sample.cusum_filter(pl.col("ret"), threshold=0.02).over("id").alias("e1"),
        pn.sample.cusum_filter("ret", threshold="h_trailing").over("id").alias("e2"),
    )
    for name, sub in out.group_by("id", maintain_order=True):
        y = sub.get_column("ret").to_numpy()
        assert np.array_equal(
            sub.get_column("e1").to_numpy(), _definition(y, np.full(y.size, 0.02))
        )
        h = sub.get_column("h_trailing").fill_null(np.nan).to_numpy()
        assert np.array_equal(sub.get_column("e2").to_numpy(), _definition(y, h)), name
    assert out.schema["e1"] == pl.Boolean


def test_prefix_invariant_with_a_causal_threshold() -> None:
    df = _panel(1)
    expr = (
        pn.sample.cusum_filter("ret", threshold="h_trailing")
        .over("id")
        .cast(pl.Int8)
        .alias("ev")
    )
    assert_prefix_invariant(expr, df, entity="id", time="t", tol=0.0)
    assert_no_lookahead(expr, df, entity="id", time="t", tol=0.0)


def test_full_sample_threshold_is_caught() -> None:
    """T12: a threshold from the full-sample sigma is a look-ahead the helpers catch."""
    df = _panel(2)
    expr = (
        pn.sample.cusum_filter("ret", threshold="h_full")
        .over("id")
        .cast(pl.Int8)
        .alias("ev")
    )

    # the leak lives in the threshold column; recompute it inside the op
    def op(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            (pl.col("ret").std().over("id") * 2).alias("h_full")
        ).with_columns(expr)

    with pytest.raises(AssertionError):
        assert_prefix_invariant(
            op, df.drop("h_full", "h_trailing"), entity="id", time="t", tol=0.0
        )


def test_tick_rule() -> None:
    df = pl.DataFrame(
        {
            "sym": ["x"] * 7 + ["y"] * 4,
            "px": [10.0, 10.0, 10.1, 10.1, 10.0, 10.0, 10.2, 5.0, 4.9, 4.9, 5.1],
        }
    )
    side = df.select(pn.sample.tick_rule("px").over("sym")).to_series()
    assert side.dtype == pl.Int8
    assert side.to_list() == [None, None, 1, 1, -1, -1, 1, None, -1, -1, 1]


def test_threshold_type_is_checked() -> None:
    with pytest.raises(TypeError, match="threshold"):
        pn.sample.cusum_filter("ret", threshold=True)
