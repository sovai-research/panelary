"""Split-aware deduplication: no near-duplicate pair may straddle train/test.

A purged, embargoed splitter guarantees that no *label window* crosses the
train/test boundary. It says nothing about *content*: if row ``r`` in the test
fold is a near-copy of row ``s`` in the training fold, the model has seen the
answer. This module closes that hole without touching the splitters -- it uses
only their public ``split(panel)`` API.

* :func:`straddling_pairs` / :func:`assert_no_straddle` -- measure or assert
  the guarantee for one ``(train, test)`` pair of frames.
* :func:`straddling_clusters` -- the same check from a precomputed
  :func:`~panelary.clean.near_duplicate_clusters` table and the two folds'
  time sets (every Panelary splitter partitions the time axis).
* :func:`purge_near_duplicates` -- drop every training row that
  near-duplicates a test row (purging, in de Prado's sense, extended from
  label windows to content).
* :class:`SplitAwareCV` -- wrap any splitter so every fold it yields is
  purged that way; the similarity graph is computed once, panel-globally.

The simplest route to the same guarantee is to run
:class:`~panelary.clean.Deduplicator` on the whole panel **before** splitting:
its point-in-time rule leaves no near-duplicate pair anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import polars as pl

from panelary.clean._common import ROW, rewrap, to_panel, with_row_ids
from panelary.clean._dedup import _expand_pairs, _Graph, _graph_for
from panelary.core.panel_frame import PanelFrame

__all__ = [
    "SplitAwareCV",
    "assert_no_straddle",
    "purge_near_duplicates",
    "straddling_clusters",
    "straddling_pairs",
]

_SIDE = "__panelary_side"


def _stack(
    train: Any, test: Any, entity: str | None, time: str | None
) -> tuple[PanelFrame, pl.DataFrame, int]:
    """Concatenate train and test into one panel with a side marker."""
    tr = to_panel(train, entity, time)
    te = to_panel(test, tr.entity_col, tr.time_col)
    a = tr.collect()
    b = te.collect().select(a.columns)
    both = pl.concat([a, b], how="vertical_relaxed").with_columns(
        pl.Series(_SIDE, np.r_[np.zeros(a.height), np.ones(b.height)].astype(np.int8))
    )
    panel = PanelFrame(both, entity=tr.entity_col, time=tr.time_col, validate=False)
    return panel, a, a.height


def straddling_pairs(
    train: PanelFrame | pl.DataFrame | pl.LazyFrame,
    test: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    columns: Sequence[str] | str | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Near-duplicate pairs with one row in ``train`` and one in ``test``.

    Parameters
    ----------
    train, test : PanelFrame | polars.DataFrame | polars.LazyFrame
        The two folds (same columns). Any split works -- by time, by entity,
        random -- because membership is taken from the frames themselves.
    entity, time : str, optional
        Panel keys for bare frames.
    columns : str | sequence of str, optional
        Content columns (default: every non-key column).
    **kwargs
        Any :func:`~panelary.clean.near_duplicate_clusters` option
        (``method``, ``threshold``, ``num_perm``, ``seed``, ...).

    Returns
    -------
    polars.DataFrame
        ``train_row``, ``test_row`` (row positions within each fold) and
        ``similarity``; empty when the split is clean.
    """
    panel, _a, n_train = _stack(train, test, entity, time)
    cols = _content(panel, columns)
    _p, _f, graph = _graph_for(
        panel, entity=None, time=None, columns=cols, **_opts(kwargs)
    )
    pairs = _expand_pairs(graph)
    return pairs.filter(
        (pl.col("row_i") < n_train) & (pl.col("row_j") >= n_train)
    ).select(
        pl.col("row_i").alias("train_row"),
        (pl.col("row_j") - n_train).alias("test_row"),
        pl.col("similarity"),
    )


def assert_no_straddle(
    train: PanelFrame | pl.DataFrame | pl.LazyFrame,
    test: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    columns: Sequence[str] | str | None = None,
    **kwargs: Any,
) -> None:
    """Assert that no near-duplicate pair straddles the ``(train, test)`` split.

    Parameters
    ----------
    train, test, entity, time, columns, **kwargs
        See :func:`straddling_pairs`.

    Raises
    ------
    AssertionError
        If any pair straddles; the message reports how many and the first few.
    """
    bad = straddling_pairs(
        train, test, entity=entity, time=time, columns=columns, **kwargs
    )
    if bad.height:
        head = bad.head(5).rows()
        raise AssertionError(
            f"NEAR-DUPLICATE LEAK: {bad.height} near-duplicate pair(s) straddle "
            f"the train/test split, e.g. (train_row, test_row, similarity) = "
            f"{head}. Deduplicate the panel before splitting "
            "(panelary.clean.Deduplicator) or purge the training fold "
            "(panelary.clean.purge_near_duplicates)."
        )


def straddling_clusters(
    clusters: pl.DataFrame,
    train_times: Any,
    test_times: Any,
    *,
    time: str,
    cluster_col: str = "cluster_id",
) -> pl.DataFrame:
    """Clusters with members on both sides of a time-based split.

    Parameters
    ----------
    clusters : polars.DataFrame
        Output of :func:`~panelary.clean.near_duplicate_clusters` (needs the
        time column and ``cluster_col``).
    train_times, test_times : array-like | polars.Series | frame
        The time values of each fold. A frame / PanelFrame contributes the
        unique values of its ``time`` column.
    time : str
        Time column name in ``clusters``.
    cluster_col : str, default="cluster_id"
        Cluster-id column name.

    Returns
    -------
    polars.DataFrame
        ``cluster_col``, ``n_train``, ``n_test`` for each straddling cluster,
        sorted by cluster id; empty when the split is clean.
    """
    tr = _times(train_times, time)
    te = _times(test_times, time)
    side = (
        pl.when(pl.col(time).is_in(te.implode()))
        .then(pl.lit("test"))
        .when(pl.col(time).is_in(tr.implode()))
        .then(pl.lit("train"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    return (
        clusters.with_columns(side.alias("__side"))
        .group_by(cluster_col)
        .agg(
            (pl.col("__side") == "train").sum().cast(pl.Int64).alias("n_train"),
            (pl.col("__side") == "test").sum().cast(pl.Int64).alias("n_test"),
        )
        .filter((pl.col("n_train") > 0) & (pl.col("n_test") > 0))
        .sort(cluster_col)
    )


def _times(x: Any, time: str) -> pl.Series:
    if isinstance(x, PanelFrame):
        return x.collect().get_column(x.time_col).unique()
    if isinstance(x, pl.LazyFrame):
        return x.select(time).collect().to_series().unique()
    if isinstance(x, pl.DataFrame):
        return x.get_column(time).unique()
    if isinstance(x, pl.Series):
        return x.unique()
    return pl.Series(list(np.asarray(x).tolist())).unique()


def _content(panel: PanelFrame, columns: Sequence[str] | str | None) -> list[str]:
    if columns is not None:
        return [columns] if isinstance(columns, str) else list(columns)
    return [
        c
        for c in panel.columns
        if c not in (panel.entity_col, panel.time_col, _SIDE, ROW)
    ]


def _opts(kwargs: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "method": "minhash",
        "threshold": 0.8,
        "num_perm": 128,
        "seed": 0,
        "tokenizer": "auto",
        "ngram": 3,
        "verify": "exact",
        "scope": "global",
        "b_bits": None,
        "n_bits": 64,
    }
    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise TypeError(f"unexpected near-duplicate option(s): {sorted(unknown)}.")
    return {**defaults, **kwargs}


def _purge_mask(
    graph: _Graph, train_rows: np.ndarray, test_rows: np.ndarray
) -> np.ndarray:
    """Boolean mask over ``train_rows``: True where the row must be purged.

    A training row is purged iff it is a near-duplicate of some test row:
    same exact group, or its group is joined to a test row's group by a
    verified edge. Pairwise, not transitive -- it purges exactly the pairs.
    """
    rep = graph.rep
    bad = np.zeros(graph.n, dtype=bool)
    test_groups = np.unique(rep[test_rows])
    bad[test_groups] = True
    if graph.ei.size:
        hit_i = bad[graph.ei]
        hit_j = bad[graph.ej]
        bad[graph.ej[hit_i]] = True
        bad[graph.ei[hit_j]] = True
        bad[test_groups] = True
    return bad[rep[train_rows]]


def purge_near_duplicates(
    train: PanelFrame | pl.DataFrame | pl.LazyFrame,
    test: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    entity: str | None = None,
    time: str | None = None,
    columns: Sequence[str] | str | None = None,
    **kwargs: Any,
) -> Any:
    """Drop every training row that near-duplicates a test row.

    The test fold is never modified (it is what you evaluate on); the training
    fold loses exactly the rows that would let the model memorise a test row.
    Afterwards :func:`assert_no_straddle` holds.

    Parameters
    ----------
    train, test, entity, time, columns, **kwargs
        See :func:`straddling_pairs`.

    Returns
    -------
    PanelFrame | polars.DataFrame | polars.LazyFrame
        The purged training fold, in the container type of ``train``.
    """
    panel, a, n_train = _stack(train, test, entity, time)
    cols = _content(panel, columns)
    _p, _f, graph = _graph_for(
        panel, entity=None, time=None, columns=cols, **_opts(kwargs)
    )
    train_rows = np.arange(n_train, dtype=np.int64)
    test_rows = np.arange(n_train, graph.n, dtype=np.int64)
    drop = _purge_mask(graph, train_rows, test_rows)
    out = a.filter(pl.Series(~drop))
    tr = to_panel(train, entity, time)
    return rewrap(train, out, tr)


class SplitAwareCV:
    """Wrap a splitter so no near-duplicate pair straddles any fold it yields.

    The near-duplicate graph is computed **once**, over the whole panel
    (panel-global). Each ``(train, test)`` fold from the wrapped splitter is
    then purged: training rows that near-duplicate any test row are removed.
    Test folds pass through untouched. The wrapped splitter is used only via
    its public ``split(panel)`` API, so purging/embargo on the time axis are
    unchanged and compose with this content purge.

    Parameters
    ----------
    cv : object with ``split(panel)`` or callable
        E.g. :class:`~panelary.validation.PurgedKFold`,
        :class:`~panelary.validation.CombinatorialPurgedCV` (with the default
        ``return_indices=False``), or the callable returned by
        :func:`~panelary.validation.expanding_window_split`.
    entity, time : str, optional
        Panel keys for bare frames.
    columns : str | sequence of str, optional
        Content columns (default: every non-key column).
    **kwargs
        Any :func:`~panelary.clean.near_duplicate_clusters` option.

    Attributes
    ----------
    n_purged_ : list of int
        Training rows purged in each fold of the most recent :meth:`split`.

    Examples
    --------
    >>> from panelary.validation import PurgedKFold
    >>> cv = SplitAwareCV(PurgedKFold(n_splits=4), threshold=0.9)
    >>> for train, test in cv.split(panel):  # doctest: +SKIP
    ...     model.fit(train)
    """

    def __init__(
        self,
        cv: Any,
        *,
        entity: str | None = None,
        time: str | None = None,
        columns: Sequence[str] | str | None = None,
        **kwargs: Any,
    ) -> None:
        if getattr(cv, "return_indices", False):
            raise ValueError(
                "SplitAwareCV needs a splitter that yields frames; construct it "
                "with return_indices=False."
            )
        if not (callable(cv) or hasattr(cv, "split")):
            raise TypeError("`cv` must have a `split(panel)` method or be callable.")
        self.cv = cv
        self.entity = entity
        self.time = time
        self.columns = columns
        self.options = _opts(kwargs)
        self.n_purged_: list[int] = []

    def get_n_splits(self) -> int:
        """Number of folds of the wrapped splitter (when it can say)."""
        return int(self.cv.get_n_splits())

    def _folds(self, panel: PanelFrame) -> Any:
        split: Callable[[PanelFrame], Any] = (
            self.cv.split if hasattr(self.cv, "split") else self.cv
        )
        return split(panel)

    def split(
        self, panel: PanelFrame | pl.DataFrame | pl.LazyFrame
    ) -> Iterator[tuple[PanelFrame, PanelFrame]]:
        """Yield ``(train, test)`` PanelFrames with straddling pairs purged.

        Parameters
        ----------
        panel : PanelFrame | polars.DataFrame | polars.LazyFrame
            The full panel.

        Yields
        ------
        (train, test) : tuple of PanelFrame
        """
        pf = to_panel(panel, self.entity, self.time)
        base = pf.collect()
        frame = with_row_ids(base)
        cols = _content(pf, self.columns)
        tagged = PanelFrame(
            frame, entity=pf.entity_col, time=pf.time_col, validate=False
        )
        # Row ids of the graph and of `tagged` coincide: both number `base`.
        _p, _f, graph = _graph_for(
            PanelFrame(base, entity=pf.entity_col, time=pf.time_col, validate=False),
            entity=None,
            time=None,
            columns=cols,
            **self.options,
        )
        self.n_purged_ = []
        for train, test in self._folds(tagged):
            tr = to_panel(train, pf.entity_col, pf.time_col).collect()
            te = to_panel(test, pf.entity_col, pf.time_col).collect()
            train_rows = tr.get_column(ROW).to_numpy().astype(np.int64)
            test_rows = te.get_column(ROW).to_numpy().astype(np.int64)
            drop = _purge_mask(graph, train_rows, test_rows)
            self.n_purged_.append(int(drop.sum()))
            yield (
                PanelFrame(
                    tr.filter(pl.Series(~drop)).drop(ROW).lazy(),
                    entity=pf.entity_col,
                    time=pf.time_col,
                    validate=False,
                ),
                PanelFrame(
                    te.drop(ROW).lazy(),
                    entity=pf.entity_col,
                    time=pf.time_col,
                    validate=False,
                ),
            )
