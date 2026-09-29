"""Fold-local sample weights: recomputed from the training fold's labels only.

AFML computes concurrency on the whole sample and then cross-validates. Near a
fold boundary that concurrency counts purged and embargoed labels whose ``t1``
can depend on test-period prices, the normalisation and the time-decay anchor
include the test fold's labels, and class frequencies include its class mix --
so test-fold information shapes the training weights (trap T1). The fix is
also the conceptually correct definition: only *training* labels create
redundancy in the training set. :class:`FoldWeights` is a frozen recipe that
:func:`panelary.cross_validate` evaluates once per fold, on the fold's labels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core._spans import SpanTable, _spans_from_t1
from panelary.weights._weights import (
    _check_decay,
    _check_kind,
    _combine,
    _entity_first_rows,
    _labels_of,
    _log_returns,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["FoldWeights"]

_NORMALIZE = ("sum_n", None)


@dataclass(frozen=True)
class FoldWeights:
    """A fold-local sample-weight recipe, evaluated per cross-validation fold.

    Nothing is fitted on construction. For each fold,
    :meth:`compute` takes the labels whose start time lies in the fold's
    positions, recomputes concurrency (on the full row axis, counting only
    those labels), uniqueness or return attribution, the time-decay anchor, the
    class frequencies and the normalisation from them alone, and returns one
    weight per row of the fold. Pass it to
    ``cross_validate(..., sample_weight=fw)`` (training fold) and/or
    ``score_weight=fw`` (test fold).

    Parameters
    ----------
    t1 : str, default "t1"
        Label end-time column.
    kind : {"uniqueness", "return", None}, default "uniqueness"
        Base weight: average uniqueness, return attribution (needs ``price``;
        it already embeds concurrency) or 1.
    price : str, optional
        Price column for ``kind="return"``.
    decay : float, optional
        Time-decay parameter ``c`` in ``(-1, 1]`` (AFML 4.11), anchored to the
        fold's cumulative uniqueness. In CPCV with training on both sides of the
        test block the decay runs across the gap towards the most recent
        *training* label; a warning says so once per ``cross_validate`` call.
    balance : str, optional
        Label column for balanced class weights from the fold's class mix.
    normalize : {"sum_n", None}, default "sum_n"
        ``"sum_n"`` rescales the fold's label weights to sum to their number.
    censored : str or None, default "censored"
        Boolean column of unresolved labels to exclude (ignored if absent).
    afml_compat : bool, default False
        Return attribution over AFML's inclusive ``[t0, t1]``.

    Notes
    -----
    Rows of the fold that are not resolved labels (null ``t1``, censored, past
    the data) get weight ``0.0``. When ``test_positions`` are given,
    :meth:`compute` raises if any training label's span covers a test time: an
    under-purge (e.g. a splitter without ``t1=``, or a null ``t1`` at a test
    time) that would leak test prices into both the label and its weight.

    Examples
    --------
    >>> fw = FoldWeights(t1="t1", kind="return", price="close", decay=0.5)
    >>> report = pn.cross_validate(  # doctest: +SKIP
    ...     est, df, y="label", cv=pn.PurgedKFold(5, t1="t1", embargo=5),
    ...     sample_weight=fw, score_weight=fw,
    ... )
    >>> w_train = fw.compute(panel, train_positions)  # doctest: +SKIP
    """

    t1: str = "t1"
    kind: str | None = "uniqueness"
    price: str | None = None
    decay: float | None = None
    balance: str | None = None
    normalize: str | None = "sum_n"
    censored: str | None = "censored"
    afml_compat: bool = False

    def __post_init__(self) -> None:
        _check_kind(self.kind, self.price)
        if self.decay is not None:
            _check_decay(self.decay)
        if self.normalize not in _NORMALIZE:
            raise ValueError(
                f"`normalize` must be one of {_NORMALIZE}, got {self.normalize!r}."
            )

    def bind(
        self,
        panel: Any,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> BoundFoldWeights:
        """Build the span table of ``panel`` once, for many :meth:`compute` calls."""
        return BoundFoldWeights(self, panel, entity=entity, time=time)

    def compute(
        self,
        panel: Any,
        positions: ArrayLike,
        *,
        test_positions: ArrayLike | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> NDArray[np.float64]:
        """Fold-local weights for the rows at ``positions`` of the time index.

        Parameters
        ----------
        panel : PanelFrame, polars.DataFrame or polars.LazyFrame
            The **whole** labelled panel (the row axis spans must be read on).
        positions : array-like of int
            Positions in the panel's sorted unique-time index that form the fold
            (e.g. ``train`` from ``PurgedKFold(..., return_indices=True)``).
        test_positions : array-like of int, optional
            The fold's test positions; enables the under-purge guard.
        entity, time : str, optional
            Keys for a bare frame (default: a PanelFrame's keys, else columns
            0 and 1).

        Returns
        -------
        numpy.ndarray
            One float64 weight per panel row whose time is in ``positions``, in
            ``(entity, time)`` order -- the row order of
            ``panel.filter(time in fold)`` on the sorted panel.
        """
        return self.bind(panel, entity=entity, time=time).compute(
            positions, test_positions=test_positions
        )


class BoundFoldWeights:
    """A :class:`FoldWeights` recipe with the panel's span table built once."""

    def __init__(
        self,
        recipe: FoldWeights,
        panel: Any,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        frame, entity_col, time_col = _panel_frame(panel, entity, time)
        self.recipe = recipe
        self.entity_col = entity_col
        self.time_col = time_col
        self.frame, self.spans = _spans_from_t1(
            frame,
            t1=recipe.t1,
            entity=entity_col,
            time=time_col,
            censored=recipe.censored,
        )
        utimes = self.frame.get_column(time_col).unique().sort().to_numpy()
        self._grid_tpos = np.searchsorted(
            utimes, self.frame.get_column(time_col).to_numpy()
        ).astype(np.int64)
        self._log_ret = (
            _log_returns(
                self.frame,
                recipe.price,
                _entity_first_rows(self.frame, entity_col),
            )
            if recipe.kind == "return" and recipe.price is not None
            else None
        )
        self._labels = (
            _labels_of(self.frame, self.spans, recipe.balance)
            if recipe.balance is not None
            else None
        )
        self._warned_decay = False

    @property
    def n_times(self) -> int:
        return self.spans.n_times

    def fold_spans(self, positions: ArrayLike) -> tuple[SpanTable, NDArray[np.bool_]]:
        """The spans of the labels starting at ``positions`` and their mask."""
        pos = _positions(positions, self.n_times)
        in_fold = np.zeros(self.n_times, dtype=bool)
        in_fold[pos] = True
        mask = in_fold[self.spans.start_tpos]
        return self.spans.subset(mask), mask

    def compute(
        self, positions: ArrayLike, *, test_positions: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """See :meth:`FoldWeights.compute`."""
        recipe = self.recipe
        pos = _positions(positions, self.n_times)
        sub, mask = self.fold_spans(pos)
        if test_positions is not None:
            test = _positions(test_positions, self.n_times)
            self._guard(sub, test)
            if (
                recipe.decay is not None
                and not self._warned_decay
                and test.size
                and pos.size
                and pos.max() > test.min()
            ):
                self._warned_decay = True
                warnings.warn(
                    "FoldWeights(decay=...): this fold trains on both sides of its "
                    "test block, so the time decay runs across the gap towards the "
                    "most recent training label.",
                    UserWarning,
                    stacklevel=3,
                )
        w = _combine(
            sub,
            kind=recipe.kind,
            log_ret=self._log_ret,
            decay=recipe.decay,
            labels=None if self._labels is None else self._labels[mask],
            normalize=recipe.normalize,
            afml_compat=recipe.afml_compat,
        )
        full = np.zeros(self.spans.n_rows, dtype=np.float64)
        full[sub.label_row] = w
        rows_in_fold = np.zeros(self.n_times, dtype=bool)
        rows_in_fold[pos] = True
        return full[rows_in_fold[self._grid_tpos]]

    def _guard(self, sub: SpanTable, test: NDArray[np.int64]) -> None:
        """Raise if a fold label's span covers a test time (an under-purge)."""
        if test.size == 0 or len(sub) == 0:
            return
        bad = sub.covers(test)
        if not bool(bad.any()):
            return
        k = int(np.argmax(bad))
        row = int(sub.label_row[k])
        ent = self.frame.get_column(self.entity_col)[row]
        t0 = self.frame.get_column(self.time_col)[row]
        raise ValueError(
            f"{int(bad.sum())} training label(s) cover a test time, e.g. the "
            f"label of {ent!r} at {t0!r} (its span reaches time position "
            f"{int(sub.end_tpos[k])}). Their prices -- and so their labels and "
            "weights -- include the test period. Purge on the label spans: "
            f"construct the splitter with t1={self.recipe.t1!r}, and drop "
            "unresolved labels (null t1) before splitting."
        )


def _positions(positions: ArrayLike, n_times: int) -> NDArray[np.int64]:
    pos = np.unique(np.asarray(positions, dtype=np.int64))
    if pos.size and (pos[0] < 0 or pos[-1] >= n_times):
        raise ValueError(
            f"positions must lie in [0, {n_times - 1}] (the panel's unique-time "
            f"index), got [{pos[0]}, {pos[-1]}]."
        )
    return pos


def _panel_frame(
    panel: Any, entity: str | None, time: str | None
) -> tuple[pl.DataFrame, str, str]:
    """``(DataFrame, entity, time)`` from a PanelFrame or a bare polars frame."""
    from panelary.core.panel_frame import PanelFrame

    if isinstance(panel, PanelFrame):
        return (
            panel.collect(),
            entity if entity is not None else panel.entity_col,
            time if time is not None else panel.time_col,
        )
    frame = panel.collect() if isinstance(panel, pl.LazyFrame) else panel
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"expected a PanelFrame or a polars frame, got {type(panel).__name__}."
        )
    cols = frame.columns
    return (
        frame,
        entity if entity is not None else cols[0],
        time if time is not None else cols[1],
    )
