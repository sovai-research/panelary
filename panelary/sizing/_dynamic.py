"""Dynamic position sizing and limit prices, sigmoid form (AFML ch. 10.4).

With ``x = forecast - price`` the divergence, the size is the sigmoid
``m(w, x) = x / sqrt(w + x^2)`` in ``(-1, 1)``; ``w`` is calibrated so that a
divergence ``x`` maps to a size ``m``: ``w = x^2 (m^-2 - 1)``.

* :func:`sigmoid_w`, :func:`sigmoid_size` -- calibrate and evaluate.
* :func:`target_position` -- ``trunc(m * max_pos)``.
* :func:`inverse_price` -- the price at which the size would be ``m``:
  ``forecast - m sqrt(w / (1 - m^2))``.
* :func:`limit_price` -- the breakeven limit price of an order from ``pos`` to
  ``target_pos``: the mean of ``inverse_price(forecast, w, j / max_pos)`` over
  the **signed** path ``j = pos + sgn, ..., target_pos``.

``sqrt(w)`` factors out of the limit price, so a per-``max_pos`` prefix table of
``g(j) = (j/Q) / sqrt(1 - (j/Q)^2)`` gives every row in O(1): O(rows + Q) in
total. AFML's printed loop, ``range(abs(pos + sgn), abs(target_pos + 1))``, is
right only for ``0 <= pos < target_pos``; the signed path agrees with it there
and is correct everywhere else. ``|j| = max_pos`` would need ``m = +-1``, an
infinite price, and raises.

Only the sigmoid form is provided; the power form is out of scope (plan 4
section 3.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "inverse_price",
    "limit_price",
    "sigmoid_size",
    "sigmoid_w",
    "target_position",
]


def _out(res: np.ndarray) -> Any:
    return res.item() if res.ndim == 0 else res


def sigmoid_w(divergence: float, size: float) -> float:
    """Calibrate ``w`` so that ``sigmoid_size(w, divergence) == size``.

    ``w = divergence^2 (size^-2 - 1)``; needs ``0 < |size| < 1`` and a non-zero
    divergence of the same sign as ``size``.
    """
    x, m = float(divergence), float(size)
    if not (0.0 < abs(m) < 1.0):
        raise ValueError(
            f"sigmoid_w: `size` must satisfy 0 < |size| < 1, got {size!r}."
        )
    if x == 0.0 or not np.isfinite(x) or (x > 0) != (m > 0):
        raise ValueError(
            "sigmoid_w: `divergence` must be finite, non-zero and of the same sign "
            f"as `size`, got divergence={divergence!r}, size={size!r}."
        )
    return x * x * (1.0 / (m * m) - 1.0)


def _check_w(w: Any) -> np.ndarray:
    arr = np.asarray(w, dtype=np.float64)
    if np.any(~(arr > 0.0)):
        raise ValueError("`w` must be > 0 (see sigmoid_w).")
    return arr


def sigmoid_size(w: Any, divergence: Any) -> Any:
    """``m = x / sqrt(w + x^2)``, the sigmoid bet size for divergence ``x``."""
    ww = _check_w(w)
    x = np.asarray(divergence, dtype=np.float64)
    return _out(x / np.sqrt(ww + x * x))


def target_position(w: Any, forecast: Any, price: Any, *, max_pos: int) -> Any:
    """``trunc(sigmoid_size(w, forecast - price) * max_pos)``, as int64."""
    q = _check_max_pos(max_pos)
    m = np.asarray(sigmoid_size(w, np.asarray(forecast, dtype=np.float64) - price))
    return _out(np.trunc(m * q).astype(np.int64))


def inverse_price(forecast: Any, w: Any, m: Any) -> Any:
    """Price at which the sigmoid size equals ``m``: ``f - m sqrt(w / (1 - m^2))``."""
    ww = _check_w(w)
    mm = np.asarray(m, dtype=np.float64)
    if np.any(np.abs(mm) >= 1.0):
        raise ValueError(
            "inverse_price: `m` must satisfy |m| < 1 (|m| = 1 is an infinite price)."
        )
    return _out(
        np.asarray(forecast, dtype=np.float64) - mm * np.sqrt(ww / (1.0 - mm * mm))
    )


def _check_max_pos(max_pos: Any) -> int:
    if (
        isinstance(max_pos, bool)
        or not isinstance(max_pos, (int, np.integer))
        or max_pos < 1
    ):
        raise ValueError(f"`max_pos` must be an integer >= 1, got {max_pos!r}.")
    return int(max_pos)


def _g_table(q: int) -> np.ndarray:
    """``H[j] = sum_{i=1..j} g(i)``, ``g(i) = (i/q) / sqrt(1 - (i/q)^2)``, j = 0..q-1."""
    i = np.arange(1, q, dtype=np.float64) / q
    return np.concatenate([[0.0], np.cumsum(i / np.sqrt(1.0 - i * i))])


def limit_price(
    target_pos: Any, pos: Any, forecast: Any, w: Any, *, max_pos: int
) -> Any:
    """Breakeven limit price for moving from ``pos`` to ``target_pos``.

    The mean of ``inverse_price(forecast, w, j / max_pos)`` over the signed path
    ``j = pos + sgn, pos + 2 sgn, ..., target_pos`` (``sgn`` the direction of
    the trade). NaN where ``target_pos == pos`` (no order). Vectorised over rows
    through a prefix table, O(rows + max_pos).

    Raises
    ------
    ValueError
        If any path position reaches ``|j| >= max_pos`` (``m = +-1`` has no
        finite inverse price).
    """
    q = _check_max_pos(max_pos)
    tgt = np.asarray(target_pos)
    cur = np.asarray(pos)
    ints = np.issubdtype(tgt.dtype, np.integer) and np.issubdtype(cur.dtype, np.integer)
    if not ints and not (np.all(np.mod(tgt, 1) == 0) and np.all(np.mod(cur, 1) == 0)):
        raise ValueError("limit_price: positions must be integers.")
    tgt = tgt.astype(np.int64)
    cur = cur.astype(np.int64)
    sgn = np.where(tgt >= cur, 1, -1)
    lo = np.minimum(cur + sgn, tgt)
    hi = np.maximum(cur + sgn, tgt)
    trade = tgt != cur
    if np.any(trade & ((np.abs(lo) >= q) | (np.abs(hi) >= q))):
        raise ValueError(
            "limit_price: the path from `pos` to `target_pos` reaches |j| >= max_pos, "
            "where the size would be +-1 and the inverse price infinite."
        )
    table = _g_table(q)

    def h(j: np.ndarray) -> np.ndarray:
        # H(j) for all integers |j| < q, using the odd symmetry H(j) = H(-j-1), j < 0.
        k = np.where(j >= 0, j, -j - 1)
        return table[np.clip(k, 0, q - 1)]

    count = np.abs(tgt - cur)
    path_sum = h(hi) - h(lo - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_g = np.where(trade, path_sum / np.maximum(count, 1), np.nan)
    ww = _check_w(w)
    res = np.asarray(forecast, dtype=np.float64) - np.sqrt(ww) * mean_g
    return _out(np.where(trade, res, np.nan))
