"""Prefix invariance of every ``flavour="trailing"`` shape transform (plan section 9, item 1).

``f(x[:T])[t] == f(x[:T+k])[t]`` **bit for bit** -- not ``allclose`` -- for a grid
of ``T``, ``k`` and every surviving ``t``, on hypothesis-generated ragged panels
(entities of different lengths, misaligned start dates, irregular time steps,
nulls and NaNs, shuffled row order). The computation reads identical inputs, so
anything short of bit-identity is a length dependence.

This is ``AGENTS.md`` invariant 1 and the one check that would catch a
regression from the trailing flavour to whole-series behaviour. A deliberately
leaky transform at the bottom proves the check can fail.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import panelary.shape as shape
from panelary.core.panel_frame import PanelFrame
from panelary.shape import PAA, Delay, Flavour, ShapeTransform, Spectral
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

ENT, TIME = "e", "t"

#: Every trailing transform, in the configurations that exercise distinct code
#: paths (each pool, each spectral mode, phase, a chunk size that splits
#: entities across FFT batches, dilation, array output).
TRAILING: dict[str, Callable[[], ShapeTransform]] = {
    "paa_mean": lambda: PAA(window=8, segments=4),
    "paa_max": lambda: PAA(window=6, segments=3, pool="max"),
    "paa_min": lambda: PAA(window=6, segments=2, pool="min"),
    "paa_std": lambda: PAA(window=8, segments=2, pool="std"),
    "paa_last": lambda: PAA(window=4, segments=4, pool="last"),
    "spectral_truncate": lambda: Spectral(window=8, k=3),
    "spectral_phase": lambda: Spectral(window=8, k=5, phase=True),
    "spectral_band": lambda: Spectral(window=16, k=3, mode="band"),
    "spectral_power": lambda: Spectral(window=8, mode="power"),
    "spectral_chunked": lambda: Spectral(window=6, k=2, chunk_rows=7),
    "delay": lambda: Delay(lags=4, dilation=2),
    "delay_array": lambda: Delay(lags=3, as_array=True),
}


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _panel(
    lengths: list[int],
    starts: list[int],
    seed: int,
    *,
    missing: float = 0.05,
    shuffle: bool = True,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ents: list[str] = []
    times: list[int] = []
    x: list[float | None] = []
    y: list[float] = []
    for i, (n, s) in enumerate(zip(lengths, starts, strict=True)):
        steps = rng.integers(1, 4, size=n)  # irregular spacing, strictly increasing
        t = s + np.cumsum(steps) - steps[0]
        xs = np.cumsum(rng.standard_normal(n))
        ys = rng.standard_normal(n) * 3.0
        for j in range(n):
            u = rng.random()
            if u < missing / 2:
                x.append(None)  # a null
            elif u < missing:
                x.append(float("nan"))  # a NaN
            else:
                x.append(float(xs[j]))
        ents += [f"e{i}"] * n
        times += [int(v) for v in t]
        y += [float(v) for v in ys]
    df = pl.DataFrame(
        {ENT: ents, TIME: times, "x": x, "y": y},
        schema={ENT: pl.Utf8, TIME: pl.Int64, "x": pl.Float64, "y": pl.Float64},
    )
    if shuffle:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed % (2**31))
    return df


@st.composite
def ragged_panels(draw: Any) -> pl.DataFrame:
    n_ent = draw(st.integers(1, 4))
    lengths = draw(st.lists(st.integers(1, 40), min_size=n_ent, max_size=n_ent))
    starts = draw(st.lists(st.integers(0, 12), min_size=n_ent, max_size=n_ent))
    seed = draw(st.integers(0, 2**32 - 1))
    return _panel(lengths, starts, seed)


# --------------------------------------------------------------------------- #
# Bitwise comparison
# --------------------------------------------------------------------------- #
def _run(factory: Callable[[], ShapeTransform], df: pl.DataFrame) -> pl.DataFrame:
    out = factory().fit_transform(df, entity=ENT, time=TIME).collect()
    arrays = [c for c, dt in out.schema.items() if isinstance(dt, pl.Array)]
    for c in arrays:  # compare array cells element by element
        width = out.schema[c].size  # type: ignore[union-attr]
        out = out.with_columns(
            [pl.col(c).arr.get(i).alias(f"{c}[{i}]") for i in range(width)]
        ).drop(c)
    return out.sort([ENT, TIME])


def _bits(s: pl.Series) -> np.ndarray:
    a = s.to_numpy()
    a = np.where(np.isnan(a), np.nan, a)  # one canonical NaN payload
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def _assert_bit_identical(
    longer: pl.DataFrame, shorter: pl.DataFrame, cut: int, label: str
) -> None:
    a = longer.filter(pl.col(TIME) <= cut)
    b = shorter.filter(pl.col(TIME) <= cut)
    assert a.select(ENT, TIME).equals(b.select(ENT, TIME)), f"{label}: keys differ"
    assert a.columns == b.columns
    for c in a.columns:
        if c in (ENT, TIME):
            continue
        ba, bb = _bits(a[c]), _bits(b[c])
        if not np.array_equal(ba, bb):
            i = int(np.flatnonzero(ba != bb)[0])
            raise AssertionError(
                f"{label}: column {c!r} at ({a[ENT][i]!r}, {a[TIME][i]}) is "
                f"{a[c][i]!r} with more history after it but {b[c][i]!r} when "
                f"the panel is cut at {TIME} <= {cut}: prefix invariance broken."
            )


def _check_prefix(
    factory: Callable[[], ShapeTransform], df: pl.DataFrame, label: str
) -> None:
    times = sorted(df[TIME].unique().to_list())
    if len(times) < 2:
        return
    full = _run(factory, df)
    # A grid of prefix lengths T and extensions k: every pair (T, T + k) of
    # cuts, plus the full panel as the largest extension.
    grid = sorted(
        {times[int(q * (len(times) - 1))] for q in (0.0, 0.2, 0.45, 0.7, 0.9)}
    )
    runs = {c: _run(factory, df.filter(pl.col(TIME) <= c)) for c in grid}
    for i, c in enumerate(grid):
        _assert_bit_identical(full, runs[c], c, f"{label} (T={c}, full)")
        for c2 in grid[i + 1 :]:
            _assert_bit_identical(runs[c2], runs[c], c, f"{label} (T={c}, T+k={c2})")


# --------------------------------------------------------------------------- #
# 0. Coverage: every trailing transform in the package is exercised
# --------------------------------------------------------------------------- #
def _public_transform_classes() -> list[type[ShapeTransform]]:
    out = []
    for name in shape.__all__:
        obj = getattr(shape, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, ShapeTransform)
            and obj is not ShapeTransform
            and not inspect.isabstract(obj)
        ):
            out.append(obj)
    return out


def test_every_trailing_transform_is_covered() -> None:
    trailing = {
        cls
        for cls in _public_transform_classes()
        if cls.spec.flavour is Flavour.TRAILING
    }
    covered = {type(f()) for f in TRAILING.values()}
    missing = sorted(c.__name__ for c in trailing - covered)
    assert not missing, (
        f"trailing transform(s) {missing} are not in TRAILING, so their prefix "
        "invariance is unverified. Add a factory for each."
    )


# --------------------------------------------------------------------------- #
# 1. Hypothesis: ragged panels, grid of cuts, bitwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(TRAILING))
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(df=ragged_panels())
def test_trailing_transform_is_bitwise_prefix_invariant(
    label: str, df: pl.DataFrame
) -> None:
    _check_prefix(TRAILING[label], df, label)


# --------------------------------------------------------------------------- #
# 2. The library's own verifiers, at zero tolerance, on a fixed ragged panel
# --------------------------------------------------------------------------- #
_FIXED = _panel([30, 24, 17, 9], [0, 3, 6, 1], seed=11)


@pytest.mark.parametrize("label", sorted(TRAILING))
def test_trailing_transform_passes_both_library_verifiers(label: str) -> None:
    factory = TRAILING[label]

    def op(frame: PanelFrame) -> pl.DataFrame:
        out = factory().fit_transform(frame).collect()
        return out.drop([c for c, dt in out.schema.items() if isinstance(dt, pl.Array)])

    pf = PanelFrame(_FIXED, entity=ENT, time=TIME)
    assert_prefix_invariant(op, pf, tol=0.0)
    assert_no_lookahead(op, pf, tol=0.0)


def test_input_row_order_does_not_matter() -> None:
    """Shuffling the input rows changes nothing (the spine sorts, then unsorts)."""
    a = _panel([20, 12], [0, 2], seed=3, shuffle=False)
    b = a.sample(fraction=1.0, shuffle=True, seed=5)
    for label, factory in TRAILING.items():
        ra, rb = _run(factory, a), _run(factory, b)
        _assert_bit_identical(ra, rb, int(a[TIME].max()), label)


# --------------------------------------------------------------------------- #
# 3. The check has teeth: a deliberately leaky trailing transform fails it
# --------------------------------------------------------------------------- #
class _LeakyPAA(PAA):
    """PAA minus the entity's *whole-series* mean: a look-ahead in trailing clothes."""

    def _trailing_kernel(self, pw: Any, sp: Any) -> np.ndarray:
        out = super()._trailing_kernel(pw, sp)
        ent_id = np.repeat(np.arange(sp.lengths.size), sp.lengths)
        vals = np.where(np.isfinite(sp.values), sp.values, 0.0)
        sums = np.zeros((sp.lengths.size, sp.values.shape[1]))
        np.add.at(sums, ent_id, vals)
        # kernels are column-major (k, m, n); the leak is a (n, k) whole-series mean
        return out - (sums / sp.lengths[:, None])[ent_id].T[:, None, :]


def test_the_check_catches_a_deliberately_leaky_transform() -> None:
    df = _panel([30, 24], [0, 3], seed=2, missing=0.0)
    with pytest.raises(AssertionError, match="prefix invariance broken"):
        _check_prefix(lambda: _LeakyPAA(window=4, segments=2), df, "leaky")
    pf = PanelFrame(df, entity=ENT, time=TIME)
    with pytest.raises(AssertionError):
        assert_prefix_invariant(
            lambda f: _LeakyPAA(window=4, segments=2).fit_transform(f), pf
        )
