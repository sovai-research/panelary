"""Survivorship: merge a cluster of duplicate records into one golden record.

Once rows are known to describe the same thing (a near-duplicate cluster, a
resolved entity), something has to decide which *value* of each column
survives. The rules here are the standard master-data-management ones,
expressed as Polars aggregations over rows sorted in a declared order:

``"first"`` / ``"last"``
    The value on the first / last row in order (nulls included).
``"first_non_null"`` / ``"last_non_null"``
    The first / last non-null value (recency or seniority with fallback).
``"most_frequent"``
    The modal non-null value; ties go to the value seen first.
``"longest"``
    The longest string (ties: first seen) -- e.g. the fullest company name.
``"max"`` / ``"min"``
    Column maximum / minimum.
``"most_complete"``
    The value from the row with the most non-null fields (ties: first).
``"source_priority"``
    The non-null value from the most trusted source, given ``source_col``
    and an ordered ``source_priority`` list (unknown sources rank last).

A callable ``rule(col) -> pl.Expr`` is accepted too and used verbatim as the
aggregation.

**Leakage note.** A golden record reads every row of its group -- including
later ones -- so merging is *not* point-in-time. Use it for reference data
(entity masters, security masters) or behind an explicit
``leakage_safe = False`` step, never silently inside a backtest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import polars as pl

from panelary.clean._common import check_choice

__all__ = ["Survivorship", "golden_records"]

Rule = Literal[
    "first",
    "last",
    "first_non_null",
    "last_non_null",
    "most_frequent",
    "longest",
    "max",
    "min",
    "most_complete",
    "source_priority",
]
RULES: tuple[str, ...] = (
    "first",
    "last",
    "first_non_null",
    "last_non_null",
    "most_frequent",
    "longest",
    "max",
    "min",
    "most_complete",
    "source_priority",
)
_NN = "__panelary_nonnull"
_PRIO = "__panelary_prio"


class Survivorship:
    """A set of per-column golden-record rules.

    Parameters
    ----------
    rules : mapping of str to rule, optional
        Column -> rule name (see module docs) or callable ``col -> pl.Expr``.
    default : rule, default="first_non_null"
        Rule for every column not named in ``rules``.
    source_col : str, optional
        Column naming each row's source; required by ``"source_priority"``.
    source_priority : sequence, optional
        Sources from most to least trusted.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "cluster": [1, 1, 1],
    ...         "t": [1, 2, 3],
    ...         "name": ["ACME", "Acme Corporation", None],
    ...         "px": [None, 10.0, 11.0],
    ...     }
    ... )
    >>> Survivorship({"name": "longest", "px": "last_non_null"}).apply(
    ...     df, by="cluster", order_by="t"
    ... ).row(0)
    (1, 1, 'Acme Corporation', 11.0)
    """

    def __init__(
        self,
        rules: Mapping[str, Rule | Callable[[str], pl.Expr]] | None = None,
        *,
        default: Rule = "first_non_null",
        source_col: str | None = None,
        source_priority: Sequence[object] | None = None,
    ) -> None:
        self.rules: dict[str, Rule | Callable[[str], pl.Expr]] = dict(rules or {})
        for col, rule in self.rules.items():
            if not callable(rule):
                check_choice(f"rules[{col!r}]", rule, RULES)
        check_choice("default", default, RULES)
        self.default = default
        self.source_col = source_col
        self.source_priority = list(source_priority) if source_priority else None
        uses_source = default == "source_priority" or any(
            r == "source_priority" for r in self.rules.values()
        )
        if uses_source and (source_col is None or not self.source_priority):
            raise ValueError(
                "rule 'source_priority' needs `source_col=` and a non-empty "
                "`source_priority=` list."
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Survivorship(rules={self.rules!r}, default={self.default!r})"

    # ------------------------------------------------------------------ #
    def _expr(self, col: str, rule: Rule | Callable[[str], pl.Expr]) -> pl.Expr:
        c = pl.col(col)
        if callable(rule):
            return rule(col).alias(col)
        if rule == "first":
            return c.first()
        if rule == "last":
            return c.last()
        if rule == "first_non_null":
            return c.drop_nulls().first()
        if rule == "last_non_null":
            return c.drop_nulls().last()
        if rule == "max":
            return c.max()
        if rule == "min":
            return c.min()
        if rule == "longest":
            length = c.cast(pl.String).str.len_chars().fill_null(-1)
            return c.sort_by(length, descending=True, maintain_order=True).first()
        if rule == "most_frequent":
            cnt = pl.col(f"__panelary_cnt_{col}")
            return c.sort_by(cnt, descending=True, maintain_order=True).first()
        if rule == "most_complete":
            return c.sort_by(_NN, descending=True, maintain_order=True).first()
        # source_priority: non-nulls first, then the most trusted source.
        return c.sort_by([c.is_null(), pl.col(_PRIO)], maintain_order=True).first()

    def apply(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        *,
        by: str | Sequence[str],
        order_by: str | Sequence[str] | None = None,
        descending: bool = False,
    ) -> pl.DataFrame:
        """Collapse each group of ``df`` into one golden record.

        Parameters
        ----------
        df : polars.DataFrame | polars.LazyFrame
            Records to merge.
        by : str | sequence of str
            Group (cluster) columns; one output row per group.
        order_by : str | sequence of str, optional
            The order that defines "first" / "last" / tie-breaks (e.g. the
            time column). Defaults to input order.
        descending : bool, default=False
            Sort ``order_by`` descending (most recent first).

        Returns
        -------
        polars.DataFrame
            One row per group, groups in order of first appearance, columns in
            input order.
        """
        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        keys = [by] if isinstance(by, str) else list(by)
        if order_by is not None:
            order = [order_by] if isinstance(order_by, str) else list(order_by)
            frame = frame.sort(order, descending=descending, maintain_order=True)
        value_cols = [c for c in frame.columns if c not in keys]
        chosen = {c: self.rules.get(c, self.default) for c in value_cols}
        helpers: list[pl.Expr] = []
        if any(r == "most_complete" for r in chosen.values()):
            helpers.append(
                pl.sum_horizontal(pl.col(value_cols).is_not_null()).alias(_NN)
            )
        if any(r == "source_priority" for r in chosen.values()):
            assert self.source_col is not None and self.source_priority is not None
            prio = {s: i for i, s in enumerate(self.source_priority)}
            helpers.append(
                pl.col(self.source_col)
                .replace_strict(prio, default=len(prio), return_dtype=pl.Int64)
                .alias(_PRIO)
            )
        for c, r in chosen.items():
            if r == "most_frequent":
                helpers.append(
                    pl.when(pl.col(c).is_null())
                    .then(pl.lit(-1, dtype=pl.Int64))
                    .otherwise(pl.len().over([*keys, c]).cast(pl.Int64))
                    .alias(f"__panelary_cnt_{c}")
                )
        if helpers:
            frame = frame.with_columns(helpers)
        out = frame.group_by(keys, maintain_order=True).agg(
            [self._expr(c, r) for c, r in chosen.items()]
        )
        return out.select(
            frame.select(pl.exclude(_NN, _PRIO, "^__panelary_cnt_.*$")).columns
        )

    def merge(
        self,
        frame: pl.DataFrame,
        *,
        cluster_col: str,
        entity: str,
        time: str,
        order_col: str,
        rank: np.ndarray,
    ) -> pl.DataFrame:
        """Golden record per cluster, keyed like the cluster's earliest row.

        Used by :class:`~panelary.clean.Deduplicator` with
        ``linkage="component"``: the entity and time of each output row come
        from the cluster's first row in ``rank`` order; every other column
        follows the configured rules in that order.
        """
        ranked = frame.with_columns(
            pl.Series("__panelary_rank", rank[frame.get_column(order_col).to_numpy()])
        ).sort("__panelary_rank")
        keyed = Survivorship(
            {**self.rules, entity: "first", time: "first", order_col: "first"},
            default=self.default,
            source_col=self.source_col,
            source_priority=self.source_priority,
        )
        out = keyed.apply(ranked.drop("__panelary_rank"), by=cluster_col)
        return out.sort(order_col)


def golden_records(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    by: str | Sequence[str],
    rules: Mapping[str, Rule | Callable[[str], pl.Expr]] | None = None,
    default: Rule = "first_non_null",
    order_by: str | Sequence[str] | None = None,
    descending: bool = False,
    source_col: str | None = None,
    source_priority: Sequence[object] | None = None,
) -> pl.DataFrame:
    """Functional form of :meth:`Survivorship.apply`.

    Parameters
    ----------
    df : polars.DataFrame | polars.LazyFrame
        Records to merge.
    by : str | sequence of str
        Group (cluster) columns.
    rules, default, source_col, source_priority
        See :class:`Survivorship`.
    order_by, descending
        See :meth:`Survivorship.apply`.

    Returns
    -------
    polars.DataFrame
        One golden record per group.
    """
    return Survivorship(
        rules, default=default, source_col=source_col, source_priority=source_priority
    ).apply(df, by=by, order_by=order_by, descending=descending)
