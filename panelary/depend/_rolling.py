"""Rolling dependence **features**: one value per row, trailing window, causal.

These emit columns, not tables. Each is a Polars expression -- a two-column
struct mapped per entity with ``map_batches`` (never ``rolling_map``, measured
249x slower) -- whose numpy kernel builds every trailing window with
:func:`numpy.lib.stride_tricks.sliding_window_view` and evaluates **all**
windows of an entity in one row-batched call.

Prefix invariance is the whole game here, and it is easy to break in three
ways, each of which ``tests/test_depend_prefix.py`` pins with a deliberately
leaky twin:

1. **The window is a fixed integer**, never ``n // k``.
2. **Ranks are computed inside the window**, never globally and then sliced.
3. **A tail quantile or kernel bandwidth comes from the window** (or is a
   constant): the tail threshold is ``floor(q * window)`` ranks *within* the
   window, and GCMI's copula transform ranks within the window.

A window containing a missing value yields a null, never a statistic computed
on a silently shortened sample. The first ``window - 1`` rows (``window`` when
the predictor is the series' own lag) are null.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import polars as pl

from panelary.depend._coef import _tail_rows, _xi_rows
from panelary.depend._energy import _dcor_rows
from panelary.depend._info import _gcmi_rows

__all__ = [
    "rolling_dcor",
    "rolling_gcmi",
    "rolling_tail_dep",
    "rolling_xi",
    "window_statistic",
]

RowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
_MAX_ELEMS = 2_000_000


def window_statistic(
    a: np.ndarray, b: np.ndarray, window: int, rows: RowFn
) -> np.ndarray:
    """Trailing-window statistic ``rows(a[t-w+1:t+1], b[t-w+1:t+1])`` for every ``t``.

    Parameters
    ----------
    a, b : numpy.ndarray
        Equal-length float64 series in time order (NaN = missing).
    window : int
        Fixed window length.
    rows : callable
        Row-batched statistic ``(R, w), (R, w) -> (R,)``; it sees one window
        per row, so every rank it takes is a within-window rank.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64, NaN where the window is incomplete or contains NaN.
    """
    n = a.size
    w = int(window)
    out = np.full(n, np.nan)
    if n < w:
        return out
    A = np.lib.stride_tricks.sliding_window_view(a, w)
    B = np.lib.stride_tricks.sliding_window_view(b, w)
    ok = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    idx = np.flatnonzero(ok)
    step = max(1, _MAX_ELEMS // w)
    res = out[w - 1 :]
    for s in range(0, idx.size, step):
        sel = idx[s : s + step]
        res[sel] = rows(A[sel], B[sel])
    return out


def _check_window(window: int, minimum: int, name: str) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError(
            f"{name}: `window` must be a fixed integer (never derived from the "
            f"series length), got {window!r}."
        )
    w = int(window)
    if w < minimum:
        raise ValueError(f"{name}: `window` must be >= {minimum}, got {w}.")
    return w


def _pair_expr(
    expr: pl.Expr, other: str | pl.Expr | None, window: int, rows: RowFn
) -> pl.Expr:
    pred = (
        expr.shift(1)
        if other is None
        else (pl.col(other) if isinstance(other, str) else other)
    )

    def run(s: pl.Series) -> pl.Series:
        a = s.struct.field("__a").cast(pl.Float64).to_numpy()
        b = s.struct.field("__b").cast(pl.Float64).to_numpy()
        vals = window_statistic(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            window,
            rows,
        )
        return pl.Series(s.name, vals, dtype=pl.Float64, nan_to_null=True)

    return pl.struct(
        pred.cast(pl.Float64).alias("__a"), expr.cast(pl.Float64).alias("__b")
    ).map_batches(run, return_dtype=pl.Float64)


def rolling_xi(
    expr: pl.Expr, other: str | pl.Expr | None = None, *, window: int = 20
) -> pl.Expr:
    """Trailing-window Chatterjee xi of this column **on** ``other``.

    ``xi(other -> self)`` over the last ``window`` rows: "how much is this
    column a (possibly non-monotone) function of ``other``". With
    ``other=None`` the predictor is the column's own lag, ``x[t-1] -> x[t]`` --
    a rolling nonlinear autocorrelation. Ranks are taken inside each window;
    ties in the predictor are broken by exact expectation (deterministic).

    Chatterjee, S. (2021). A new coefficient of correlation. *JASA* 116(536).
    Clean-room numpy implementation.
    """
    w = _check_window(window, 3, "rolling_xi")
    return _pair_expr(expr, other, w, _xi_rows)


def rolling_dcor(
    expr: pl.Expr, other: str | pl.Expr | None = None, *, window: int = 20
) -> pl.Expr:
    """Trailing-window bias-corrected distance correlation with ``other``.

    Symmetric; ``other=None`` pairs ``x[t-1]`` with ``x[t]``. Exact univariate
    ``O(w^1.5)`` path per window (no ``w x w`` matrix).

    Szekely, G. J. & Rizzo, M. L. (2014). Partial distance correlation with
    methods for dissimilarities. *Ann. Statist.* 42(6). Clean-room numpy.
    """
    w = _check_window(window, 4, "rolling_dcor")
    return _pair_expr(expr, other, w, _dcor_rows)


def rolling_tail_dep(
    expr: pl.Expr,
    other: str | pl.Expr | None = None,
    *,
    window: int = 60,
    q: float = 0.1,
    side: str = "lower",
) -> pl.Expr:
    """Trailing-window nonparametric tail dependence with ``other``.

    ``#{both in their q-tail within the window} / floor(q * window)``. The tail
    threshold is a within-window rank count -- it is **re-derived inside every
    window**, never a quantile of the full series. Under independence the value
    sits near ``q``, not 0.

    Parameters
    ----------
    window : int, default=60
        Fixed; ``floor(q * window) >= 1`` is required.
    q : float, default=0.1
        Tail probability in ``(0, 0.5)``.
    side : {"lower", "upper"}, default="lower"
    """
    if side not in {"lower", "upper"}:
        raise ValueError(
            f"rolling_tail_dep: `side` must be 'lower' or 'upper', got {side!r}."
        )
    if not 0.0 < q < 0.5:
        raise ValueError(f"rolling_tail_dep: `q` must be in (0, 0.5), got {q}.")
    w = _check_window(window, 2, "rolling_tail_dep")
    if int(np.floor(q * w)) < 1:
        raise ValueError(
            f"rolling_tail_dep: floor(q * window) = floor({q} * {w}) < 1; widen the "
            "window or raise q."
        )
    return _pair_expr(expr, other, w, partial(_tail_rows, q=float(q), side=side))


def rolling_gcmi(
    expr: pl.Expr, other: str | pl.Expr | None = None, *, window: int = 20
) -> pl.Expr:
    """Trailing-window Gaussian-copula mutual information (nats) with ``other``.

    Normal scores are taken inside each window. For one pair this is a
    monotone function of a rank correlation (see ``gcmi``): it tracks monotone
    dependence only.

    Ince, R. A. A. et al. (2017). *Human Brain Mapping* 38(3). Clean-room numpy.
    """
    w = _check_window(window, 5, "rolling_gcmi")
    return _pair_expr(expr, other, w, _gcmi_rows)
