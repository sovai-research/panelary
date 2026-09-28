"""Interval and dilation samplers (`panelary.embed._intervals`).

Contract touched: prefix invariance by construction -- every sampler is a
function of the window length and the parameters, never of data or of the
series length beyond the window.
"""

from __future__ import annotations

import numpy as np
import pytest

from panelary.embed import dilation_ladder, dyadic_intervals, random_dilated_intervals
from panelary.embed._intervals import IntervalSet, max_dilation_exponent


@pytest.mark.parametrize(
    ("length", "span", "expected"),
    [
        (64, 9, [1, 2, 4]),
        (9, 9, [1]),
        (17, 9, [1, 2]),
        (128, 9, [1, 2, 4, 8]),
        (10, 1, [1]),
    ],
)
def test_dilation_ladder(length, span, expected):
    assert dilation_ladder(length, span).tolist() == expected
    # The largest dilation fits; the next one does not.
    d = expected[-1]
    assert (span - 1) * d + 1 <= length
    if span > 1:
        assert (span - 1) * 2 * d + 1 > length


def test_max_dilation_exponent_refuses_a_pattern_longer_than_the_window():
    with pytest.raises(ValueError, match="does not fit"):
        max_dilation_exponent(5, 9)


def test_random_intervals_fit_inside_the_window_and_are_seeded():
    iv = random_dilated_intervals(64, 50, seed=3, min_length=20)
    assert len(iv) == 50
    last = iv.start + (iv.length - 1) * iv.dilation
    assert int(last.max()) <= 63 and int(iv.start.min()) >= 0
    assert int(iv.length.min()) >= 20
    assert (iv.dilation >= 1).all() and (iv.dilation > 1).any()
    again = random_dilated_intervals(64, 50, seed=3, min_length=20)
    for a, b in zip(iv.as_dict().values(), again.as_dict().values(), strict=True):
        np.testing.assert_array_equal(a, b)
    other = random_dilated_intervals(64, 50, seed=4, min_length=20)
    assert not np.array_equal(iv.start, other.start)


def test_random_intervals_min_length_caps_at_the_window():
    iv = random_dilated_intervals(12, 5, seed=0, min_length=20)
    assert (
        (iv.length == 12).all() and (iv.dilation == 1).all() and (iv.start == 0).all()
    )


def test_random_intervals_require_an_int_seed():
    with pytest.raises(TypeError, match="seed"):
        random_dilated_intervals(64, 3, seed=None)  # type: ignore[arg-type]


def test_interval_indices():
    iv = IntervalSet(
        window=10,
        start=np.array([1]),
        length=np.array([3]),
        dilation=np.array([4]),
    )
    assert iv.indices(0).tolist() == [1, 5, 9]
    with pytest.raises(ValueError, match="outside"):
        IntervalSet(
            window=9, start=np.array([1]), length=np.array([3]), dilation=np.array([4])
        )


def test_dyadic_intervals_structure():
    ivs = dyadic_intervals(8, 2)
    assert [(i.start, i.stop, i.depth, i.shifted) for i in ivs] == [
        (0, 8, 0, False),
        (0, 4, 1, False),
        (4, 8, 1, False),
        (2, 6, 1, True),
    ]
    # Depth is capped at floor(log2(n)) + 1, so no interval is empty.
    deep = dyadic_intervals(5, 10)
    assert max(i.depth for i in deep) == 2
    assert all(i.length >= 1 for i in deep)
    assert all(0 <= i.start < i.stop <= 5 for i in deep)
