"""Leak-safe splitters on the time-position axis, and CPCV backtest paths.

This module is the *positional* face of Panelary's cross-validation machinery.
:mod:`panelary.core.model_selection` already implements purging, embargo,
:class:`~panelary.core.model_selection.PurgedKFold` and
:class:`~panelary.core.model_selection.CombinatorialPurgedCV` over
:class:`~panelary.core.panel_frame.PanelFrame` objects; those are
**re-exported unchanged** here. What is added is the small amount of surface the
rest of ``validation/`` needs and that did not exist yet:

* :func:`cpcv_splits` / :func:`walk_forward_splits` — purged + embargoed splits
  expressed as integer positions into the sorted unique-time index, so
  bootstraps, conformal calibration and Monte-Carlo studies can use them without
  materialising panels. The existing walk-forward splitters
  (:mod:`panelary.cross_validation`) do **not** purge; these do.
* :func:`purged_calibration_split` — the train/calibration split conformal
  prediction needs, with the boundary purged and embargoed.
* :func:`cpcv_backtest_paths` / :func:`walk_forward_backtest_path` — turn a
  ``fit_predict`` callback into the *distribution* of backtest paths (Lopez de
  Prado, Ch. 12) rather than a single trajectory; the matrix they return is the
  direct input to
  :func:`~panelary.validation.probability_of_backtest_overfitting`.
* :func:`fold_boundaries` — the segment cut points a block bootstrap must not
  straddle.

All purge/embargo arithmetic is delegated to
``core.model_selection._purge_embargo_positions`` so there is exactly one
implementation of the leakage rule in the library.

``horizon`` and ``embargo`` may be calendar durations (``"5d"``, ``"5bd"``,
:class:`~panelary.core.model_selection.BusinessDays`, ``datetime.timedelta``)
here too. Positions alone cannot say how much *time* separates them, so a
calendar spec needs the sorted unique time values, passed as ``times=``; the
calendar arithmetic is ``core.model_selection._train_positions``, the same
dispatcher the panel splitters use. Integer specs never need ``times`` and give
byte-identical splits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from panelary.core._calendar import (
    Duration,
    as_steps,
    is_steps,
    shift_forward,
    validate_duration,
)
from panelary.core.model_selection import (
    CombinatorialPurgedCV,
    CVReport,
    PurgedKFold,
    _purge_embargo_positions,
    _purge_embargo_positions_t1,
    _train_positions,
    cross_validate,
    expanding_window_split,
    sliding_window_split,
)

__all__ = [
    "CVReport",
    "CombinatorialPurgedCV",
    "IndexSplit",
    "PurgedKFold",
    "cpcv_backtest_paths",
    "cpcv_splits",
    "cross_validate",
    "expanding_window_split",
    "fold_boundaries",
    "purged_calibration_split",
    "sliding_window_split",
    "walk_forward_backtest_path",
    "walk_forward_splits",
]


@dataclass(frozen=True)
class IndexSplit:
    """One purged/embargoed split expressed as positions into the time index.

    Attributes
    ----------
    train : numpy.ndarray
        Sorted training positions.
    test : numpy.ndarray
        Sorted test positions.
    test_groups : tuple of int, optional
        For CPCV, the group ids used as the test set; ``None`` otherwise.
    """

    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...] | None = None


# --------------------------------------------------------------------------- #
# Calendar support for the positional splitters
# --------------------------------------------------------------------------- #
def _time_index(
    times: Any, n_times: int, horizon: object, embargo: object
) -> pl.Series | None:
    """Validate ``times`` against ``n_times``; ``None`` when nothing needs it.

    Calendar ``horizon`` / ``embargo`` specs are measured on time values, so
    they require ``times``: the sorted, strictly increasing, null-free unique
    time values, one per position.
    """
    horizon = validate_duration(horizon, name="horizon")
    embargo = validate_duration(embargo, name="embargo")
    calendar = not (is_steps(horizon) and is_steps(embargo))
    if times is None:
        if calendar:
            raise TypeError(
                "a calendar `horizon`/`embargo` is measured on time values, so "
                "the positional splitters need `times=`: the sorted unique time "
                f"values, one per position (length {n_times})."
            )
        return None
    series = times if isinstance(times, pl.Series) else pl.Series("time", times)
    if series.len() != n_times:
        raise ValueError(
            f"`times` has {series.len()} values but `n_times` is {n_times}; pass "
            "one time value per position of the unique-time index."
        )
    if series.null_count():
        raise ValueError("`times` must not contain nulls.")
    if not (series.is_sorted() and series.n_unique() == n_times):
        raise ValueError(
            "`times` must be strictly increasing (the sorted unique time index)."
        )
    dtype = series.dtype
    if calendar and not (dtype == pl.Date or isinstance(dtype, pl.Datetime)):
        raise TypeError(
            f"a calendar `horizon`/`embargo` needs Date or Datetime `times`, got "
            f"{dtype}; on an integer axis pass them as int numbers of steps."
        )
    return series


def _positional_t1(t1: Any, n_times: int) -> np.ndarray | None:
    """Validate a per-position ``t1`` array (label end times, one per position)."""
    if t1 is None:
        return None
    arr = t1.to_numpy() if isinstance(t1, pl.Series) else np.asarray(t1)
    if arr.ndim != 1 or arr.shape[0] != n_times:
        raise ValueError(
            f"`t1` must hold one label end time per position ({n_times}), got "
            f"shape {arr.shape}."
        )
    return arr


def _axis_values(index: pl.Series | None, n_times: int) -> np.ndarray:
    """The time values ``t1`` is compared with: ``times``, else the positions."""
    return index.to_numpy() if index is not None else np.arange(n_times)


def _calendar_gap(
    train: np.ndarray, start: int, embargo: Duration, times: pl.Series
) -> np.ndarray:
    """Keep train positions ``j`` with ``times[j] + embargo < times[start]``.

    The calendar form of the walk-forward gap ``j < start - embargo``: the
    embargo separates the *end of training* from the *start of the test
    block* by more than ``embargo`` of real time, not by ``embargo`` rows.
    """
    if not train.size:
        return train
    shifted = shift_forward(times.gather(train.tolist()), embargo)
    keep = (shifted < times[start]).to_numpy()
    return train[keep]


# --------------------------------------------------------------------------- #
# Positional splitters
# --------------------------------------------------------------------------- #
def cpcv_splits(
    n_times: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    horizon: int | Duration = 0,
    embargo: int | Duration = 0,
    times: Any = None,
    t1: Any = None,
) -> list[IndexSplit]:
    """Combinatorial Purged CV splits as integer time positions.

    Thin, allocation-free wrapper over
    :class:`~panelary.core.model_selection.CombinatorialPurgedCV`: the
    group partition, purge and embargo are that class's, not a second
    implementation.

    Parameters
    ----------
    n_times : int
        Number of unique time steps.
    n_groups : int, default=6
        Number ``N`` of contiguous time groups.
    n_test_groups : int, default=2
        Number ``k`` of groups per test set (``1 <= k < N``).
    horizon : int or duration, default=0
        Label horizon, used for purging: time steps, or a calendar duration
        (needs ``times``).
    embargo : int or duration, default=0
        Embargo applied after each contiguous test block: time steps, or a
        calendar duration such as ``"5bd"`` (needs ``times``).
    times : array-like or polars.Series, optional
        The sorted unique time values, one per position. Required only for a
        calendar ``horizon`` / ``embargo``.
    t1 : array-like, optional
        Per-position label end times (event-based purge; supersedes
        ``horizon``): values on the ``times`` axis, or positions when ``times``
        is omitted. The label at position ``j`` spans ``[times[j], t1[j]]``;
        a null never overlaps anything.

    Returns
    -------
    list of IndexSplit
        ``C(N, k)`` splits in lexicographic order of the test-group tuple.

    Examples
    --------
    >>> splits = cpcv_splits(12, n_groups=4, n_test_groups=2, embargo=1)
    >>> len(splits)
    6
    """
    cv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        horizon=horizon,
        embargo=embargo,
    )
    index = _time_index(times, n_times, horizon, embargo)
    t1_arr = _positional_t1(t1, n_times)
    if t1_arr is None:
        folds = cv._iter_folds(n_times, time_index=index)
    else:
        folds = cv._iter_folds(
            n_times, _axis_values(index, n_times), t1_arr, time_index=index
        )
    return [
        IndexSplit(f.train_positions, f.test_positions, f.test_groups) for f in folds
    ]


def walk_forward_splits(
    n_times: int,
    *,
    n_splits: int = 5,
    test_size: int | None = None,
    horizon: int | Duration = 0,
    embargo: int | Duration = 0,
    expanding: bool = True,
    window_size: int | None = None,
    times: Any = None,
    t1: Any = None,
) -> list[IndexSplit]:
    """Purged, embargoed walk-forward splits as integer time positions.

    Training uses only positions strictly *before* the test block, then the same
    purge/embargo rule as :class:`PurgedKFold` removes the boundary region whose
    labels overlap the test window.

    Parameters
    ----------
    n_times : int
        Number of unique time steps.
    n_splits : int, default=5
        Number of successive test blocks.
    test_size : int, optional
        Size of each test block. Defaults to ``n_times // (n_splits + 1)``.
    horizon : int or duration, default=0
        Label horizon used for purging the train/test boundary: time steps, or
        a calendar duration (needs ``times``).
    embargo : int or duration, default=0
        Additional embargo removed before the test block: positions, or a
        calendar duration (needs ``times``) -- training then ends more than
        ``embargo`` of real time before the test block starts.
    expanding : bool, default=True
        Expanding window (all history) vs a rolling window of ``window_size``.
    window_size : int, optional
        Rolling-window length when ``expanding=False``. Defaults to
        ``test_size * n_splits``.
    times : array-like or polars.Series, optional
        The sorted unique time values, one per position. Required only for a
        calendar ``horizon`` / ``embargo``.
    t1 : array-like, optional
        Per-position label end times (event-based purge; supersedes
        ``horizon``), as in :func:`cpcv_splits`. A training label is kept only
        if its span ends before the test block starts.

    Returns
    -------
    list of IndexSplit
        Splits in chronological order. Splits with an empty training set are
        dropped.

    Raises
    ------
    ValueError
        If the sample is too short for the requested geometry.
    """
    if n_splits < 1:
        raise ValueError(f"`n_splits` must be >= 1, got {n_splits}.")
    if test_size is None:
        test_size = n_times // (n_splits + 1)
    if test_size < 1:
        raise ValueError(
            f"`test_size` resolves to {test_size}; need at least "
            f"{n_splits + 1} time steps for n_splits={n_splits}."
        )
    if window_size is None:
        window_size = test_size * n_splits

    first_test = n_times - n_splits * test_size
    if first_test < 1:
        raise ValueError(
            f"n_splits={n_splits} x test_size={test_size} leaves no training "
            f"history in {n_times} time steps."
        )

    index = _time_index(times, n_times, horizon, embargo)
    horizon = validate_duration(horizon, name="horizon")
    embargo = validate_duration(embargo, name="embargo")
    integer = is_steps(horizon) and is_steps(embargo)
    t1_arr = _positional_t1(t1, n_times)
    axis = _axis_values(index, n_times) if t1_arr is not None else None

    out: list[IndexSplit] = []
    for s in range(n_splits):
        start = first_test + s * test_size
        stop = start + test_size
        test = np.arange(start, stop, dtype=np.int64)
        if t1_arr is not None and is_steps(embargo):
            assert axis is not None
            allowed = _purge_embargo_positions_t1(
                n_times, test, axis, t1_arr, as_steps(embargo)
            )
        elif integer and t1_arr is None:
            allowed = _purge_embargo_positions(
                n_times, test, as_steps(horizon), as_steps(embargo)
            )
        else:
            allowed = _train_positions(
                n_times,
                test,
                horizon=horizon,
                embargo=embargo,
                times=index,
                t1=t1_arr,
            )
        lo = 0 if expanding else max(0, start - window_size)
        train = allowed[(allowed < start) & (allowed >= lo)]
        # The embargo protects the *forward* side of a test block; for a strictly
        # backward-looking walk-forward fit it is applied as an extra gap.
        if is_steps(embargo):
            if as_steps(embargo) > 0 and train.size:
                train = train[train < start - as_steps(embargo)]
        else:
            assert index is not None
            train = _calendar_gap(train, start, embargo, index)
        if train.size:
            out.append(IndexSplit(train, test, None))
    return out


def purged_calibration_split(
    n_times: int,
    *,
    calibration_size: int | float = 0.25,
    horizon: int | Duration = 0,
    embargo: int | Duration = 0,
    times: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a time axis into ``(train, calibration)`` with a purged boundary.

    Conformal prediction is only honest if the calibration scores were produced
    by a model that never saw them. On a time series that requires more than a
    random split: the calibration block must come *after* training, and the
    boundary must be purged by the label horizon and embargoed for serial
    correlation.

    Parameters
    ----------
    n_times : int
        Number of unique time steps.
    calibration_size : int | float, default=0.25
        Absolute number of calibration steps, or a fraction of ``n_times``.
    horizon : int or duration, default=0
        Label horizon; training positions whose labels reach into the
        calibration block are purged. Time steps, or a calendar duration
        (needs ``times``).
    embargo : int or duration, default=0
        Extra positions dropped before the calibration block, or a calendar
        duration of real time (needs ``times``).
    times : array-like or polars.Series, optional
        The sorted unique time values, one per position. Required only for a
        calendar ``horizon`` / ``embargo``.

    Returns
    -------
    (train, calibration) : tuple of numpy.ndarray
        Sorted, disjoint position arrays. ``max(train) < min(calibration)``.

    Raises
    ------
    ValueError
        If the split leaves no training or no calibration positions.

    Examples
    --------
    >>> train, calib = purged_calibration_split(20, calibration_size=5, horizon=2)
    >>> int(train.max()), int(calib.min())
    (12, 15)
    """
    if n_times < 2:
        raise ValueError(f"`n_times` must be >= 2, got {n_times}.")
    if isinstance(calibration_size, float) and 0 < calibration_size < 1:
        n_calib = int(round(calibration_size * n_times))
    else:
        n_calib = int(calibration_size)
    if not (1 <= n_calib < n_times):
        raise ValueError(
            f"`calibration_size` resolves to {n_calib}, which must be in "
            f"[1, {n_times - 1}]."
        )
    start = n_times - n_calib
    calib = np.arange(start, n_times, dtype=np.int64)
    index = _time_index(times, n_times, horizon, embargo)
    horizon = validate_duration(horizon, name="horizon")
    embargo = validate_duration(embargo, name="embargo")
    if is_steps(horizon) and is_steps(embargo):
        allowed = _purge_embargo_positions(
            n_times, calib, as_steps(horizon), as_steps(embargo)
        )
    else:
        allowed = _train_positions(
            n_times, calib, horizon=horizon, embargo=embargo, times=index
        )
    train = allowed[allowed < start]
    if is_steps(embargo):
        if as_steps(embargo) > 0 and train.size:
            train = train[train < start - as_steps(embargo)]
    else:
        assert index is not None
        train = _calendar_gap(train, start, embargo, index)
    if train.size == 0:
        raise ValueError(
            "purging left no training positions; reduce `calibration_size`, "
            "`horizon` or `embargo`."
        )
    return train, calib


def fold_boundaries(splits: Sequence[IndexSplit]) -> list[int]:
    """Return the segment cut points implied by a list of splits.

    A block bootstrap must never resample across these positions, or a fold's
    observations would leak into another fold's resample (leak-safety guardrail
    #4). Feed the result to the ``boundaries`` argument of the samplers in
    :mod:`panelary.validation._bootstrap`.

    Parameters
    ----------
    splits : sequence of IndexSplit
        The splits whose test blocks define the segmentation.

    Returns
    -------
    list of int
        Sorted, unique start positions of each contiguous test block.
    """
    cuts: set[int] = set()
    for sp in splits:
        test = np.asarray(sp.test, dtype=np.int64)
        if test.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(test) != 1) + 1
        for pos in [0, *breaks.tolist()]:
            cuts.add(int(test[pos]))
        cuts.add(int(test[-1]) + 1)
    return sorted(cuts)


# --------------------------------------------------------------------------- #
# Backtest paths
# --------------------------------------------------------------------------- #
def cpcv_backtest_paths(
    n_times: int,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    horizon: int | Duration = 0,
    embargo: int | Duration = 0,
    times: Any = None,
) -> np.ndarray:
    """Run CPCV and stitch the folds into the full set of backtest paths.

    Lopez de Prado (Ch. 12): because each group is tested in many combinations,
    the folds recombine into ``C(N,k) * k / N`` **distinct full-length
    out-of-sample paths**. A single walk-forward backtest is one draw from this
    distribution; looking at the whole distribution is what makes the
    overfitting diagnostics (PBO, Deflated Sharpe) meaningful.

    Parameters
    ----------
    n_times : int
        Number of time steps.
    fit_predict : callable
        ``fit_predict(train_positions, test_positions) -> ndarray`` returning one
        value per test position (typically a per-period strategy return). It is
        called once per split, i.e. ``C(n_groups, n_test_groups)`` times.
    n_groups, n_test_groups, horizon, embargo, times
        See :func:`cpcv_splits`.

    Returns
    -------
    ndarray of shape (n_times, n_paths)
        Column ``p`` is backtest path ``p``, covering every time step exactly
        once.

    Raises
    ------
    ValueError
        If ``fit_predict`` returns the wrong number of values for a split.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.linspace(0, 1, 24)
    >>> paths = cpcv_backtest_paths(
    ...     24, lambda tr, te: r[te], n_groups=4, n_test_groups=2
    ... )
    >>> paths.shape
    (24, 3)
    """
    cv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        horizon=horizon,
        embargo=embargo,
    )
    groups = cv._group_positions(n_times)
    index = _time_index(times, n_times, horizon, embargo)
    folds = list(cv._iter_folds(n_times, time_index=index))

    # values[split][global position] for the split's test positions.
    per_split: list[dict[int, float]] = []
    for fold in folds:
        vals = np.asarray(
            fit_predict(fold.train_positions, fold.test_positions), dtype=float
        ).ravel()
        if vals.shape[0] != fold.test_positions.shape[0]:
            raise ValueError(
                f"`fit_predict` returned {vals.shape[0]} values for a test set "
                f"of size {fold.test_positions.shape[0]}."
            )
        per_split.append(
            dict(zip(fold.test_positions.tolist(), vals.tolist(), strict=True))
        )

    paths = cv.backtest_paths()
    out = np.full((n_times, len(paths)), np.nan, dtype=float)
    for p, path in enumerate(paths):
        for split_idx, g in path:
            table = per_split[split_idx]
            for pos in groups[g].tolist():
                out[pos, p] = table[pos]
    return out


def walk_forward_backtest_path(
    n_times: int,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_splits: int = 5,
    test_size: int | None = None,
    horizon: int | Duration = 0,
    embargo: int | Duration = 0,
    expanding: bool = True,
    times: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a purged walk-forward backtest, returning its single path.

    The counterpart to :func:`cpcv_backtest_paths`: one trajectory, covering only
    the positions that were ever out-of-sample.

    Parameters
    ----------
    n_times : int
        Number of time steps.
    fit_predict : callable
        ``fit_predict(train_positions, test_positions) -> ndarray``.
    n_splits, test_size, horizon, embargo, expanding, times
        See :func:`walk_forward_splits`.

    Returns
    -------
    (positions, values) : tuple of numpy.ndarray
        The out-of-sample positions in chronological order and the value
        produced for each.
    """
    splits = walk_forward_splits(
        n_times,
        n_splits=n_splits,
        test_size=test_size,
        horizon=horizon,
        embargo=embargo,
        expanding=expanding,
        times=times,
    )
    pos_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for sp in splits:
        vals = np.asarray(fit_predict(sp.train, sp.test), dtype=float).ravel()
        if vals.shape[0] != sp.test.shape[0]:
            raise ValueError(
                f"`fit_predict` returned {vals.shape[0]} values for a test set "
                f"of size {sp.test.shape[0]}."
            )
        pos_parts.append(sp.test)
        val_parts.append(vals)
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    return np.concatenate(pos_parts), np.concatenate(val_parts)
