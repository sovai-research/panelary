"""Input coercion and p-value helpers shared by the evaluation tests (private)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary._internal import _special

__all__ = [
    "ALTERNATIVES",
    "as_matrix",
    "as_vector",
    "check_alternative",
    "normal_pvalue",
    "resample_pvalue",
    "resolve_names",
]

ALTERNATIVES = ("two-sided", "greater", "less")


def as_matrix(x: Any, name: str) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Coerce ``(T,)`` / ``(T, M)`` input to a float64 ``(T, M)`` matrix.

    Accepts numpy arrays, nested sequences, :class:`polars.Series` and
    :class:`polars.DataFrame` (whose column names are returned). Nulls become
    ``nan``.
    """
    colnames: tuple[str, ...] | None = None
    if isinstance(x, pl.DataFrame):
        colnames = tuple(x.columns)
        arr = x.select(pl.all().cast(pl.Float64)).to_numpy()
    elif isinstance(x, pl.Series):
        colnames = (x.name,) if x.name else None
        arr = x.cast(pl.Float64).to_numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be (T,) or (T, M), got shape {arr.shape}.")
    return arr, colnames


def as_vector(x: Any, name: str, n: int | None = None) -> np.ndarray:
    """Coerce to a float64 ``(T,)`` vector (length-checked against ``n``)."""
    if isinstance(x, pl.Series):
        arr = x.cast(pl.Float64).to_numpy()
    elif isinstance(x, pl.DataFrame):
        if x.width != 1:
            raise ValueError(f"`{name}` must be a single column.")
        arr = x.to_series().cast(pl.Float64).to_numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be 1-D, got shape {arr.shape}.")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"`{name}` has {arr.shape[0]} rows; expected {n}.")
    return arr


def resolve_names(
    names: Sequence[str] | str | None, m: int, colnames: tuple[str, ...] | None
) -> tuple[str, ...]:
    """Model labels: explicit ``names``, else frame columns, else ``m0, m1, ...``."""
    if names is not None:
        out = (names,) if isinstance(names, str) else tuple(str(n) for n in names)
        if len(out) != m:
            raise ValueError(f"`names` has {len(out)} entries; expected {m}.")
        return out
    if colnames is not None and len(colnames) == m:
        return colnames
    return tuple(f"m{j}" for j in range(m))


def check_alternative(alternative: str) -> str:
    if alternative not in ALTERNATIVES:
        raise ValueError(
            f"unknown `alternative` {alternative!r}; expected one of {ALTERNATIVES}."
        )
    return alternative


def normal_pvalue(z: np.ndarray, alternative: str) -> np.ndarray:
    """N(0,1) p-values via ``erfc`` (accurate far into the tail); ``nan`` passes through."""
    zz = np.asarray(z, dtype=np.float64)
    if alternative == "greater":
        out = np.asarray(_special.norm_sf(zz), dtype=np.float64)
    elif alternative == "less":
        out = np.asarray(_special.norm_cdf(zz), dtype=np.float64)
    else:
        out = np.minimum(2.0 * np.asarray(_special.norm_sf(np.abs(zz)), float), 1.0)
    return np.where(np.isfinite(zz), out, np.nan)


def resample_pvalue(null: np.ndarray, stat: np.ndarray, alternative: str) -> np.ndarray:
    """``(1 + #{null >= stat}) / (1 + B)`` per column, in the direction of ``alternative``.

    ``null`` is ``(B, M)`` and already centred (e.g. ``(est* - est) / se*``).
    Non-finite null draws count as exceedances (conservative).
    """
    nb = null.shape[0]
    s = np.asarray(stat, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        if alternative == "greater":
            hit = null >= s
        elif alternative == "less":
            hit = null <= s
        else:
            hit = np.abs(null) >= np.abs(s)
    hit |= ~np.isfinite(null)
    out = (1.0 + hit.sum(axis=0)) / (1.0 + nb)
    return np.where(np.isfinite(s), out, np.nan)
