"""PanelFrame: a thin, typed, lazy view over a long-format panel.

A *panel* is a long-format table indexed by an **entity** (e.g. a security id,
a customer id, a country) and a **time** axis (e.g. a date or an integer step).
Every row is one observation of one entity at one time. Features live in the
remaining columns.

``PanelFrame`` is a *view*: it wraps a :class:`polars.LazyFrame` and remembers
which column is the entity and which is the time. It never copies data and stays
lazy until you explicitly call :meth:`PanelFrame.collect`. All panel-aware
operations downstream (windowing, lagging, cross-validation) build on the
``entity`` / ``time`` contract carried here.

Notes
-----
This module is part of the *correctness-by-construction* core. The whole point
of routing data through a ``PanelFrame`` is that the entity/time keys are
validated **once** and then trusted everywhere else, so that leakage-prone
operations (lags, rolling windows, splits) can be expressed safely with
``.over(entity_col)`` semantics.

**Row order is part of that contract, and it used to be unenforced.** The
``panel_safe`` rule reads "``.over(entity_col)`` on a *time-sorted* panel", but
``.over(entity)`` carries no notion of time: it serialises with
``order_by: null`` and simply takes each entity's rows in the order they
happen to sit in the frame. A ``shift``, a rolling window or an expanding sum
on a panel that arrived shuffled is therefore not an error -- it is a plausible
wrong number. :meth:`PanelFrame.is_sorted_per_entity` and
:meth:`PanelFrame.sort_panel` existed but were opt-in, and nothing called them.

What this module now does about it, in three additive layers:

1. **Track** what is known about the row order in an O(1) flag
   (:attr:`PanelFrame.sortedness`) that no operation has to collect to read.
   :meth:`PanelFrame.sort_panel` sets it and becomes a no-op when it is already
   known-sorted; order-preserving operations carry it forward.
2. **Verify** on request -- :meth:`PanelFrame.assert_sorted` (or
   ``PanelFrame(..., check_sorted=True)``) measures the real order once and
   caches the answer. This *must* stay opt-in: measuring materialises, and
   ``PanelFrame`` is lazy on purpose.
3. **Warn** -- the first time a within-entity operation runs on a panel whose
   order is not known, emit a :class:`PanelOrderWarning`. It warns rather than
   raises because silently reordering a caller's rows, or rejecting frames the
   library accepted yesterday, would both be breaking changes; the warning names
   the four ways to make it go away.

The fourth way is the real fix and needs no flag at all:
:meth:`PanelFrame.within_entity` pins ``order_by=time_col`` into the window
expression, so the result is correct whatever order the rows are in.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

__all__ = ["PanelFrame", "PanelOrderWarning"]

#: What is known about a panel's **row order** -- see :attr:`PanelFrame.sortedness`.
#:
#: ``"panel"``
#:     Rows are in ``(entity, time)`` ascending order: exactly what
#:     :meth:`PanelFrame.sort_panel` produces. Implies ``"time"``.
#: ``"time"``
#:     Within every entity ``time`` is non-decreasing (the ``panel_safe``
#:     precondition), but entities may be interleaved, so a real sort would
#:     still move rows around.
#: ``"unsorted"``
#:     Measured, and some entity's time axis goes backwards.
#: ``"unknown"``
#:     Nobody has said and nobody has looked. The default, and what every panel
#:     built before this existed is.
OrderState = Literal["unknown", "panel", "time", "unsorted"]

_UNKNOWN: OrderState = "unknown"
_PANEL: OrderState = "panel"
_TIME: OrderState = "time"
_UNSORTED: OrderState = "unsorted"

#: The states in which the ``panel_safe`` time-order precondition holds.
_ORDERED: frozenset[str] = frozenset({_PANEL, _TIME})


class PanelOrderWarning(UserWarning):
    """A within-entity operation ran on a panel of unknown row order.

    ``.over(entity_col)`` walks each entity's rows in *frame order*, not in time
    order -- there is no ``order_by`` in the expression to make it do otherwise.
    On a panel that is not time-sorted the result is wrong but well-formed: a
    lag that reaches sideways instead of backwards, a rolling mean over an
    arbitrary subset. Nothing raises, so the number reaches your model.

    This warning fires **once per frame**, on the first *order-dependent*
    within-entity operation performed through a :class:`PanelFrame` whose
    :attr:`PanelFrame.sortedness` is not known to be time-ordered. A shift, a
    rolling window, a cumulative sum, a forward fill; not a per-entity mean,
    which a shuffle cannot disturb. It is a warning and not an error because
    the alternatives are both breaking: raising would reject panels the library
    accepted yesterday, and sorting silently would reorder a caller's rows
    behind their back.

    Silence it truthfully with :meth:`PanelFrame.sort_panel`,
    :meth:`PanelFrame.assert_sorted`, :meth:`PanelFrame.mark_sorted`, or by
    writing the expression as :meth:`PanelFrame.within_entity`.
    """


_ORDER_WARNING = (
    "{op} does within-entity work on a panel whose row order is not known to be "
    "time-sorted. `.over({entity!r})` carries no notion of time -- it serialises "
    "with `order_by: null` and takes each entity's rows in frame order -- so a "
    "shift, a rolling window or an expanding aggregate over rows that are not in "
    "time order returns a plausible wrong number rather than an error. Fix it "
    "with `panel.sort_panel()`; verify an order you believe you already have "
    "with `panel.assert_sorted()` (or `PanelFrame(..., check_sorted=True)`), "
    "which materialises once; promise it in O(1) with `panel.mark_sorted()` (or "
    "`PanelFrame(..., assume_sorted=True)`); or write the expression as "
    "`panel.within_entity(expr)`, which pins `order_by={time!r}` into the window "
    "and is correct whatever the row order."
)


#: Rendered fragments of polars operations whose result depends on the order of
#: the rows *inside* the window. Deliberately over-inclusive at the edges -- an
#: opaque ``python_udf`` counts, because it could be doing anything -- and
#: deliberately not exhaustive. A miss costs a warning that does not fire; a
#: false alarm on an order-free aggregate (``mean().over(entity)``, the
#: fixed-effects idiom) is how a warning gets switched off wholesale.
_ORDER_SENSITIVE: tuple[str, ...] = (
    ".arg_",
    ".backward_fill(",
    ".bottom_k(",
    ".cum_",
    ".Cumulative_eval(",
    ".diff(",
    ".ewm_",
    ".fill_null_with_strategy(",
    ".first(",
    ".forward_fill(",
    ".gather(",
    ".index_of(",
    ".interpolate(",
    ".last(",
    ".pct_change(",
    ".peak_max(",
    ".peak_min(",
    ".python_udf(",
    ".reverse(",
    ".rle",
    ".rolling_",
    ".search_sorted(",
    ".shift(",
    ".shift_and_fill(",
    ".slice(",
    ".sort(",
    ".top_k(",
)


def _over_arguments(text: str) -> Iterator[str]:
    """Yield the argument text of every ``.over(...)`` in a rendered expression.

    Polars renders a window as ``.over([col("e")])`` and a time-ordered one as
    ``.over(partition_by: [col("e")], order_by: col("t"))``; this walks balanced
    parentheses so nested calls inside the argument do not truncate it.
    """
    token = ".over("
    start = 0
    while True:
        open_at = text.find(token, start)
        if open_at < 0:
            return
        cursor = open_at + len(token)
        depth = 1
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            cursor += 1
        yield text[open_at + len(token) : cursor - 1]
        start = cursor


def _is_unordered_over_entity(obj: Any, entity: str) -> bool:
    """Best-effort: is ``obj`` an *order-dependent* window over ``entity``?

    Three conditions, all read off the rendered expression: it windows over the
    entity column, that window carries no ``order_by``, and something in it
    actually depends on row order (:data:`_ORDER_SENSITIVE`). A per-entity mean
    is none the worse for a shuffled frame and must not be warned about.

    Deliberately a rendering-level heuristic: it drives a warning, never a
    result -- polars expressions expose no public predicate for "is this an
    order-dependent window over column X".

    Lists and tuples are recursed into element by element, because
    ``str([expr])`` renders each element with ``Expr.__repr__``, which truncates
    ("``col("x").shift([dyn int: 1]).o…``") and would hide the window.
    Generators are left alone: consuming one here would break the caller.
    """
    if isinstance(obj, (list, tuple)):
        return any(_is_unordered_over_entity(item, entity) for item in obj)
    try:
        text = str(obj)
    except Exception:  # pragma: no cover - defensive: repr must not break a call
        return False
    if ".over(" not in text:
        return False
    if not any(token in text for token in _ORDER_SENSITIVE):
        return False
    needle = f'col("{entity}")'
    return any(needle in arg and "order_by" not in arg for arg in _over_arguments(text))


def _output_names(obj: Any) -> tuple[str, ...] | None:
    """Columns an ``into_expr`` argument writes, or None if undeterminable.

    ``()`` means "writes nothing new" (a bare column reference passes a column
    through untouched). ``None`` means "cannot tell cheaply", which callers must
    read as "assume it writes anything".
    """
    if isinstance(obj, str):
        return ()
    if isinstance(obj, pl.Series):
        return (obj.name,)
    if isinstance(obj, pl.Expr):
        try:
            if obj.meta.is_column():
                return ()
            name = obj.meta.output_name(raise_if_undetermined=False)
        except Exception:  # pragma: no cover - polars version differences
            return None
        return None if name is None else (name,)
    if isinstance(obj, (list, tuple)):
        names: list[str] = []
        for item in obj:
            item_names = _output_names(item)
            if item_names is None:
                return None
            names.extend(item_names)
        return tuple(names)
    return None


def _to_lazyframe(data: Any) -> pl.LazyFrame:
    """Coerce supported inputs to a :class:`polars.LazyFrame` without copying eagerly.

    Accepts :class:`polars.LazyFrame`, :class:`polars.DataFrame`, and any object
    exposing a ``.lazy()`` method (e.g. a narwhals-wrapped frame). Anything else
    raises a :class:`TypeError` with an actionable message.
    """
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    # narwhals-ish / duck-typed frame: prefer a native polars handle.
    to_native = getattr(data, "to_native", None)
    if callable(to_native):
        native = to_native()
        if isinstance(native, pl.LazyFrame):
            return native
        if isinstance(native, pl.DataFrame):
            return native.lazy()
    lazy = getattr(data, "lazy", None)
    if callable(lazy):
        candidate = lazy()
        if isinstance(candidate, pl.LazyFrame):
            return candidate
    raise TypeError(
        "PanelFrame expects a polars DataFrame or LazyFrame "
        f"(or an object with a `.lazy()`/`.to_native()` returning one), "
        f"got {type(data).__name__!r}."
    )


class PanelFrame:
    """A typed, lazy view over a long-format panel keyed by ``(entity, time)``.

    The frame is never materialised on construction; only the schema (column
    names and dtypes) is inspected so that the entity/time contract can be
    validated cheaply. Use :meth:`collect` to evaluate.

    Parameters
    ----------
    data : polars.DataFrame | polars.LazyFrame | frame-like
        The underlying long-format panel. ``DataFrame`` inputs are wrapped with
        ``.lazy()`` (no eager work). Objects exposing ``.lazy()`` /
        ``.to_native()`` (e.g. narwhals frames) are accepted too.
    entity : str
        Name of the entity (panel id) column. Must exist in ``data``.
    time : str
        Name of the time column. Must exist in ``data`` and differ from
        ``entity``.
    validate : bool, default=True
        If True, run schema-level validation (column existence, distinctness,
        time dtype sanity). Set False only when re-wrapping a frame you already
        trust, e.g. inside :meth:`with_columns`.
    assume_sorted : bool, default=False
        If True, record -- in O(1), **without looking at the data** -- that the
        rows are already in ``(entity, time)`` ascending order, i.e. exactly
        what :meth:`sort_panel` would produce. A promise, not a check: it
        silences :class:`PanelOrderWarning` and lets :meth:`sort_panel` skip the
        sort, so only promise what is true.
    check_sorted : bool, default=False
        If True, **materialise once** and verify that ``time`` is non-decreasing
        within every entity, raising :class:`ValueError` if it is not. Opt-in
        because it collects; the default keeps construction lazy. Wins over
        ``assume_sorted`` when both are given -- a measurement beats a promise.

    Attributes
    ----------
    entity_col : str
        The validated entity column name.
    time_col : str
        The validated time column name.
    sortedness : OrderState
        What is known about the row order (``"unknown"`` / ``"time"`` /
        ``"panel"`` / ``"unsorted"``), readable in O(1).

    Raises
    ------
    TypeError
        If ``data`` cannot be coerced to a LazyFrame, or if ``entity`` / ``time``
        are not strings.
    ValueError
        If the entity or time column is missing, if they are the same column, if
        the time column has a dtype that cannot be ordered, or (with
        ``check_sorted=True``) if some entity's time axis goes backwards.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "A", "B", "B"],
    ...         "date": [1, 2, 1, 2],
    ...         "ret": [0.1, 0.2, -0.1, 0.0],
    ...     }
    ... )
    >>> panel = PanelFrame(df, entity="ticker", time="date")
    >>> panel.feature_cols
    ['ret']
    >>> panel.sort_panel().collect().shape
    (4, 3)

    Notes
    -----
    **Leakage contract.** A ``PanelFrame`` only guarantees the *keys* are valid;
    it does not by itself prevent look-ahead. Downstream transforms and
    splitters must express any forward-looking computation walk-forward via
    ``.over(entity_col)`` (use :meth:`over_entity`) and rely on
    :meth:`sort_panel` for deterministic ordering.

    **Order contract.** ``.over(entity_col)`` respects whatever order the rows
    are already in, so "time-sorted" is a genuine precondition of every
    within-entity operation. The frame carries what it knows about that order in
    :attr:`sortedness` (O(1) to read, never collected on construction) and warns
    -- once, with :class:`PanelOrderWarning` -- the first time within-entity work
    goes through a frame whose order is unknown. Nothing raises and nothing is
    reordered behind your back: the check is an affordance, not a gate.
    :meth:`within_entity` sidesteps the question entirely by pinning
    ``order_by=time_col`` into the expression.
    """

    __slots__ = ("_lf", "_entity", "_time", "_order", "_warned")

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | Any,
        entity: str,
        time: str,
        *,
        validate: bool = True,
        assume_sorted: bool = False,
        check_sorted: bool = False,
    ) -> None:
        if not isinstance(entity, str):
            raise TypeError(
                f"`entity` must be a column name (str), got {type(entity).__name__!r}."
            )
        if not isinstance(time, str):
            raise TypeError(
                f"`time` must be a column name (str), got {type(time).__name__!r}."
            )
        self._lf: pl.LazyFrame = _to_lazyframe(data)
        self._entity: str = entity
        self._time: str = time
        # O(1) on every path: a promise is recorded, never verified, unless the
        # caller explicitly asks for the collect that `check_sorted` implies.
        self._order: OrderState = _PANEL if assume_sorted else _UNKNOWN
        self._warned: bool = False
        if validate:
            self._validate_schema()
        if check_sorted:
            self._order = _UNKNOWN  # a measurement supersedes a promise
            self.assert_sorted()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _schema(self) -> pl.Schema:
        """Return the underlying lazy schema (cheap; does not collect data)."""
        return self._lf.collect_schema()

    def _validate_schema(self) -> None:
        """Validate the entity/time contract at the schema level.

        Checks, with Polars-quality error messages:

        * the entity column exists,
        * the time column exists,
        * entity and time are distinct columns,
        * the time column has an orderable dtype.
        """
        schema = self._schema()
        names = schema.names()

        if self._entity == self._time:
            raise ValueError(
                "`entity` and `time` must be different columns, "
                f"but both are {self._entity!r}."
            )
        if self._entity not in names:
            raise ValueError(
                f"entity column {self._entity!r} not found in panel. "
                f"Available columns: {names}."
            )
        if self._time not in names:
            raise ValueError(
                f"time column {self._time!r} not found in panel. "
                f"Available columns: {names}."
            )

        time_dtype = schema[self._time]
        if not (
            time_dtype.is_numeric()
            or time_dtype.is_temporal()
            or time_dtype == pl.Boolean
        ):
            raise ValueError(
                f"time column {self._time!r} has dtype {time_dtype!r}, which is "
                "not orderable as a time axis. Expected a numeric or temporal "
                "dtype (Int*, UInt*, Float*, Date, Datetime, Duration, Time). "
                "Cast it explicitly, e.g. "
                f"`df.with_columns(pl.col({self._time!r}).cast(pl.Int64))`."
            )

    def assert_unique_keys(self) -> Self:
        """Assert that every ``(entity, time)`` pair is unique. **Materialises.**

        This is the only validation that requires touching the data (a
        ``group_by`` count), so it is opt-in rather than run on construction.

        Returns
        -------
        PanelFrame
            ``self`` (for chaining), if all keys are unique.

        Raises
        ------
        ValueError
            If duplicate ``(entity, time)`` keys exist, including a small sample
            of the offending keys to aid debugging.
        """
        dupes = (
            self._lf.group_by(self._entity, self._time)
            .agg(pl.len().alias("__count__"))
            .filter(pl.col("__count__") > 1)
            .head(5)
            .collect()
        )
        if dupes.height > 0:
            total = (
                self._lf.group_by(self._entity, self._time)
                .agg(pl.len().alias("__count__"))
                .filter(pl.col("__count__") > 1)
                .select(pl.len())
                .collect()
                .item()
            )
            sample = dupes.select(self._entity, self._time, "__count__").rows()
            raise ValueError(
                f"panel has {total} duplicated ({self._entity}, {self._time}) "
                "key(s); each (entity, time) pair must be unique. "
                f"First offenders (entity, time, count): {sample}."
            )
        return self

    # ------------------------------------------------------------------ #
    # Row order: what is known, how to promise it, how to verify it
    # ------------------------------------------------------------------ #
    @property
    def sortedness(self) -> OrderState:
        """What is known about the panel's row order. **O(1)**; never collects.

        One of :data:`OrderState`: ``"unknown"`` (the default -- nobody has said
        and nobody has looked), ``"time"`` (verified: ``time`` is non-decreasing
        within every entity), ``"panel"`` (rows are in full ``(entity, time)``
        order -- what :meth:`sort_panel` produces or ``assume_sorted=True``
        promises), or ``"unsorted"`` (verified: some entity runs backwards).

        This is *knowledge about* the frame, not a property of it: ``"unknown"``
        says nothing about whether the rows happen to be sorted.
        """
        return self._order

    def mark_sorted(self) -> Self:
        """Promise, in O(1), that the rows are already ``(entity, time)`` sorted.

        The O(1) escape hatch from :class:`PanelOrderWarning` for a frame you
        built sorted -- a ``scan_parquet`` of a partitioned store, a join that
        preserved order, a frame a previous step already sorted. Nothing is
        checked: use :meth:`assert_sorted` if you want it verified.

        Returns
        -------
        PanelFrame
            A view with :attr:`sortedness` ``"panel"``; ``self`` if it already
            was. The data is untouched -- only what the frame knows changes.
        """
        if self._order == _PANEL:
            return self
        return self._rewrap(self._lf, order=_PANEL)

    def assert_sorted(self) -> Self:
        """Assert ``time`` is non-decreasing within every entity. **Materialises.**

        The order-shaped sibling of :meth:`assert_unique_keys`, and the same
        bargain: it is the one ordering check that has to touch the data, so it
        is opt-in rather than run on construction. The answer is cached in
        :attr:`sortedness`, so this is O(1) when the order is already known --
        after :meth:`sort_panel`, :meth:`mark_sorted`, or an earlier check.

        Returns
        -------
        PanelFrame
            ``self`` (for chaining), if the panel is in per-entity time order.

        Raises
        ------
        ValueError
            If any entity's time axis goes backwards, with a count and a small
            sample of the offending rows.
        """
        if self._order == _UNKNOWN:
            self._order = self._measure_order()
        if self._order != _UNSORTED:
            return self

        offenders = (
            self._lf.select(self._entity, self._time)
            .with_columns(
                pl.col(self._time).shift(1).over(self._entity).alias("__prev__")
            )
            .filter(pl.col(self._time) < pl.col("__prev__"))
        )
        sample = offenders.head(5).collect().rows()
        total = offenders.select(pl.len()).collect().item()
        raise ValueError(
            f"panel is not time-sorted within its entities: {total} row(s) go "
            f"backwards in time relative to the previous row of the same "
            f"{self._entity!r}. Within-entity operations (`.over("
            f"{self._entity!r})`) walk rows in frame order, so on this panel a "
            "shift or a rolling window would return a wrong number, not an "
            f"error. Call `.sort_panel()` first. First offenders ({self._entity}"
            f", {self._time}, previous {self._time}): {sample}."
        )

    def _measure_order(self) -> OrderState:
        """Measure the true row order in one pass. **Materialises.**

        Two questions, one scan: does any entity's time axis go backwards (the
        ``panel_safe`` precondition), and is the frame in full ``(entity, time)``
        order (what lets :meth:`sort_panel` skip the sort)?
        """
        keys = self._lf.select(self._entity, self._time)
        prev_time = pl.col(self._time).shift(1)
        prev_entity = pl.col(self._entity).shift(1)
        # Per-entity: `.over` gathers each group in frame order, which is
        # exactly the order a within-entity operation would see.
        backwards = (
            (pl.col(self._time) < prev_time)
            .over(self._entity)
            .fill_null(False)
            .any()
            .alias("__backwards__")
        )
        # Globally: rows ordered by entity, then by time within each entity.
        out_of_key_order = (
            (
                (pl.col(self._entity) < prev_entity)
                | (
                    (pl.col(self._entity) == prev_entity)
                    & (pl.col(self._time) < prev_time)
                )
            )
            .fill_null(False)
            .any()
            .alias("__unkeyed__")
        )
        try:
            row = keys.select(backwards, out_of_key_order).collect().row(0)
        except Exception:
            # Some entity dtypes (an unordered Categorical, a struct key) do not
            # support `<`. The per-entity question is the one that governs
            # correctness; answer it alone and decline to claim full key order.
            row = (keys.select(backwards).collect().item(), True)
        if bool(row[0]):
            return _UNSORTED
        return _TIME if bool(row[1]) else _PANEL

    def _warn_unordered(self, op: str) -> None:
        """Emit :class:`PanelOrderWarning` once, for a within-entity ``op``."""
        if self._order in _ORDERED or self._warned:
            return
        self._warned = True
        warnings.warn(
            _ORDER_WARNING.format(op=op, entity=self._entity, time=self._time),
            PanelOrderWarning,
            stacklevel=3,
        )

    def _unordered_within_entity(self, *args: Any) -> bool:
        """Would ``args`` do within-entity work this frame cannot vouch for?

        Short-circuits on the flag, so a panel whose order is known pays one
        set-membership test and never renders an expression.
        """
        if self._order in _ORDERED or self._warned:
            return False
        return any(_is_unordered_over_entity(a, self._entity) for a in args)

    def _order_after_projection(self, *args: Any, **named: Any) -> OrderState:
        """The order state that survives an order-preserving projection.

        ``with_columns`` / ``select`` keep row order, so what is known survives
        -- unless the projection rewrites a key column, in which case the frame
        knows nothing again. Anything undeterminable is treated as a rewrite.
        """
        if self._order == _UNKNOWN:
            return _UNKNOWN
        keys = (self._entity, self._time)
        if any(key in named for key in keys):
            return _UNKNOWN
        for obj in args:
            names = _output_names(obj)
            if names is None or any(name in keys for name in names):
                return _UNKNOWN
        return self._order

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def entity_col(self) -> str:
        """Name of the entity (panel id) column."""
        return self._entity

    @property
    def time_col(self) -> str:
        """Name of the time column."""
        return self._time

    @property
    def columns(self) -> list[str]:
        """All column names, in schema order."""
        return self._schema().names()

    @property
    def schema(self) -> pl.Schema:
        """The underlying lazy schema (column names -> dtypes)."""
        return self._schema()

    @property
    def feature_cols(self) -> list[str]:
        """Feature columns: every column that is neither entity nor time.

        Returns
        -------
        list of str
            Column names in schema order, excluding ``entity_col`` and
            ``time_col``.
        """
        keys = {self._entity, self._time}
        return [c for c in self._schema().names() if c not in keys]

    def entity(self) -> pl.Expr:
        """Return a Polars expression selecting the entity column."""
        return pl.col(self._entity)

    def time(self) -> pl.Expr:
        """Return a Polars expression selecting the time column."""
        return pl.col(self._time)

    # ------------------------------------------------------------------ #
    # Lazy / eager bridges
    # ------------------------------------------------------------------ #
    def lazy(self) -> pl.LazyFrame:
        """Return the underlying :class:`polars.LazyFrame` (no copy, no collect)."""
        return self._lf

    def collect(self, **kwargs: Any) -> pl.DataFrame:
        """Materialise the panel into a :class:`polars.DataFrame`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`polars.LazyFrame.collect`.

        Returns
        -------
        polars.DataFrame
        """
        return self._lf.collect(**kwargs)

    def to_frame(self) -> pl.LazyFrame:
        """Alias for :meth:`lazy`; returns the underlying LazyFrame."""
        return self._lf

    def to_native(self, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
        """Return the underlying native polars frame.

        Convenience for users who passed in a bare :class:`polars.DataFrame` /
        :class:`polars.LazyFrame` and want a native frame back after a
        transform, without keeping the :class:`PanelFrame` wrapper.

        Parameters
        ----------
        lazy : bool, default=True
            If True (default), return the underlying :class:`polars.LazyFrame`
            (no work). If False, :meth:`collect` it into a
            :class:`polars.DataFrame`.

        Returns
        -------
        polars.LazyFrame | polars.DataFrame
        """
        return self._lf if lazy else self._lf.collect()

    # ------------------------------------------------------------------ #
    # Panel-aware operations (return new PanelFrames; data stays lazy)
    # ------------------------------------------------------------------ #
    def _rewrap(
        self,
        lf: pl.LazyFrame,
        *,
        validate: bool = False,
        order: OrderState = _UNKNOWN,
    ) -> Self:
        """Wrap a derived LazyFrame in a new PanelFrame preserving the keys.

        ``order`` is what the *derived* frame knows about its row order; the
        default forgets, because most derivations may reorder rows.
        """
        out = type(self)(lf, entity=self._entity, time=self._time, validate=validate)
        out._order = order
        return out

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> Self:
        """Return a new :class:`PanelFrame` with added/replaced columns.

        Mirrors :meth:`polars.LazyFrame.with_columns`. The entity and time keys
        are preserved. Adding columns is cheap and stays lazy.

        Parameters
        ----------
        *exprs, **named_exprs
            Forwarded verbatim to :meth:`polars.LazyFrame.with_columns`.

        Returns
        -------
        PanelFrame
            A new view; ``self`` is unchanged.

        Warns
        -----
        PanelOrderWarning
            If an expression windows over the entity column without an
            ``order_by`` and this frame's order is not known to be time-sorted.
            See :class:`PanelOrderWarning`; the result is still computed.

        Raises
        ------
        ValueError
            If an expression attempts to drop or rename the entity/time column
            such that the contract would break. (Validation is re-run only if a
            key column is affected, to keep the common path cheap.)
        """
        if self._unordered_within_entity(*exprs, *named_exprs.values()):
            self._warn_unordered("with_columns")
        new_lf = self._lf.with_columns(*exprs, **named_exprs)
        new_names = new_lf.collect_schema().names()
        # Cheap guard: ensure keys survived.
        if self._entity not in new_names or self._time not in new_names:
            raise ValueError(
                "with_columns must not drop the panel keys "
                f"({self._entity!r}, {self._time!r}); resulting columns "
                f"were {new_names}."
            )
        # Row order is preserved, so what we knew about it still holds -- unless
        # a key column was itself rewritten.
        return self._rewrap(
            new_lf,
            validate=False,
            order=self._order_after_projection(*exprs, **named_exprs),
        )

    def select(self, *exprs: Any, **named_exprs: Any) -> Self:
        """Select columns, always keeping the entity and time keys.

        The entity and time columns are prepended to the selection if not
        already present, so the result is always a valid panel.

        Returns
        -------
        PanelFrame

        Warns
        -----
        PanelOrderWarning
            As :meth:`with_columns`, if a selected expression windows over the
            entity column without an ``order_by`` on a panel of unknown order.
        """
        if self._unordered_within_entity(*exprs, *named_exprs.values()):
            self._warn_unordered("select")
        new_lf = self._lf.select(*exprs, **named_exprs)
        names = new_lf.collect_schema().names()
        missing = [c for c in (self._entity, self._time) if c not in names]
        if missing:
            new_lf = self._lf.select(
                pl.col(self._entity), pl.col(self._time), *exprs, **named_exprs
            )
        return self._rewrap(
            new_lf,
            validate=False,
            order=self._order_after_projection(*exprs, **named_exprs),
        )

    def filter(self, *predicates: Any, **constraints: Any) -> Self:
        """Filter rows, preserving the panel keys. Mirrors :meth:`LazyFrame.filter`.

        Dropping rows cannot disturb the order of the rows that remain, so
        whatever was known about :attr:`sortedness` is carried through.
        """
        if self._unordered_within_entity(*predicates):
            self._warn_unordered("filter")
        return self._rewrap(
            self._lf.filter(*predicates, **constraints),
            validate=False,
            order=self._order,
        )

    def sort_panel(self, *, descending: bool = False) -> Self:
        """Return the panel sorted by ``(entity, time)`` ascending.

        Deterministic ordering is a precondition for correct lags, rolling
        windows and walk-forward splits, so most pipelines should call this
        once near the top.

        Calling it twice costs nothing: an ascending sort records
        :attr:`sortedness` ``"panel"``, and a second ascending
        :meth:`sort_panel` on a panel already known to be in that order returns
        ``self`` without adding a sort to the query plan.

        Parameters
        ----------
        descending : bool, default=False
            If True, sort the *time* axis descending within each entity. The
            result is deliberately *not* recorded as sorted: reverse time order
            is not the ``panel_safe`` precondition.

        Returns
        -------
        PanelFrame
            A new sorted view, or ``self`` when the sort would be a no-op.
        """
        if not descending and self._order == _PANEL:
            return self
        return self._rewrap(
            self._lf.sort([self._entity, self._time], descending=[False, descending]),
            validate=False,
            order=_UNKNOWN if descending else _PANEL,
        )

    def is_sorted_per_entity(self) -> bool:
        """Return True if ``time`` is non-decreasing within every entity.

        Evaluates against the panel's **current row order** (it does not sort
        first), so it answers "is this frame, as laid out, already in valid
        per-entity time order?". **Materialises** the first time, then caches
        the answer in :attr:`sortedness`; it is O(1) when the order is already
        known (after :meth:`sort_panel`, :meth:`mark_sorted`, or a previous
        check). Call :meth:`sort_panel` to enforce the ordering if this returns
        False.
        """
        if self._order == _UNKNOWN:
            self._order = self._measure_order()
        return self._order in _ORDERED

    # ------------------------------------------------------------------ #
    # Grouping helpers
    # ------------------------------------------------------------------ #
    def over_entity(self) -> str:
        """Return the entity column name for use with ``expr.over(...)``.

        Use this so feature code never hard-codes the key:

        >>> import polars as pl
        >>> panel = PanelFrame(
        ...     pl.DataFrame({"id": ["a", "a"], "t": [1, 2], "x": [1.0, 2.0]}),
        ...     entity="id", time="t",
        ... )
        >>> lag = pl.col("x").shift(1).over(panel.over_entity())

        Returns
        -------
        str
            The entity column name.

        Warns
        -----
        PanelOrderWarning
            Asking for the entity key is asking to window over it, and a bare
            ``.over(entity)`` inherits the frame's row order. Warned once if
            that order is not known to be time-sorted. Prefer
            :meth:`within_entity`, which cannot be used wrongly.
        """
        self._warn_unordered("over_entity()")
        return self._entity

    def within_entity(self, expr: pl.Expr) -> pl.Expr:
        """Evaluate ``expr`` within each entity, **in time order**, whatever the row order.

        The order-proof form of ``expr.over(entity_col)``: it pins
        ``order_by=time_col`` into the window, so polars sorts each entity's
        rows by time before evaluating and scatters the results back to their
        original positions. The frame does not need to be sorted, no warning
        fires, and nobody's rows move.

        >>> import polars as pl
        >>> shuffled = pl.DataFrame(
        ...     {"id": ["a", "a", "a"], "t": [3, 1, 2], "x": [30.0, 10.0, 20.0]}
        ... )
        >>> panel = PanelFrame(shuffled, entity="id", time="t")
        >>> lagged = panel.with_columns(
        ...     panel.within_entity(pl.col("x").shift(1)).alias("lag")
        ... )
        >>> lagged.collect()["lag"].to_list()
        [20.0, None, 10.0]

        Parameters
        ----------
        expr : polars.Expr
            The within-entity expression -- a shift, a rolling window, an
            expanding aggregate. Anything order-sensitive belongs here.

        Returns
        -------
        polars.Expr
            ``expr.over(entity_col, order_by=time_col)``.

        Raises
        ------
        NotImplementedError
            On polars versions whose ``Expr.over`` has no ``order_by``.
        """
        try:
            return expr.over(self._entity, order_by=self._time)
        except TypeError as exc:  # pragma: no cover - only on old polars
            raise NotImplementedError(
                "PanelFrame.within_entity needs `Expr.over(..., order_by=...)`, "
                f"which this polars ({pl.__version__}) does not have. Upgrade "
                "polars, or call `.sort_panel()` and use `.over(entity_col)`."
            ) from exc

    def group_by_entity(self, **kwargs: Any) -> pl.LazyGroupBy:
        """Return a lazy ``group_by`` over the entity column.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`polars.LazyFrame.group_by`.

        Returns
        -------
        polars.LazyGroupBy

        Warns
        -----
        PanelOrderWarning
            Once, if the row order is not known: aggregations that depend on it
            (``first``, ``last``, ``shift``, any cumulative) see each entity's
            rows in frame order.
        """
        self._warn_unordered("group_by_entity()")
        return self._lf.group_by(self._entity, **kwargs)

    def entities(self) -> pl.Series:
        """Return the sorted unique entity ids. **Materialises.**"""
        return (
            self._lf.select(pl.col(self._entity).unique().sort()).collect().to_series()
        )

    def n_entities(self) -> int:
        """Return the number of distinct entities. **Materialises.**"""
        return self._lf.select(pl.col(self._entity).n_unique()).collect().item()

    def time_index(self) -> pl.Series:
        """Return the sorted unique time values across all entities. **Materialises.**"""
        return self._lf.select(pl.col(self._time).unique().sort()).collect().to_series()

    # ------------------------------------------------------------------ #
    # Dunders
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        try:
            schema = self._schema()
            cols = schema.names()
            return (
                f"PanelFrame(entity={self._entity!r}, time={self._time!r}, "
                f"features={self.feature_cols!r}, columns={cols!r}, "
                f"sortedness={self._order!r})"
            )
        except Exception:  # pragma: no cover - repr must never raise
            return (
                f"PanelFrame(entity={self._entity!r}, time={self._time!r}, "
                "<schema unavailable>)"
            )

    def __contains__(self, col: object) -> bool:
        return isinstance(col, str) and col in self._schema().names()

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply ``func(self, *args, **kwargs)`` and return its result.

        Convenience for chaining custom panel-aware helpers.
        """
        return func(self, *args, **kwargs)


def as_panel(
    data: pl.DataFrame | pl.LazyFrame | PanelFrame | Any,
    entity: str | None = None,
    time: str | None = None,
    *,
    assume_sorted: bool = False,
    check_sorted: bool = False,
) -> PanelFrame:
    """Coerce ``data`` into a :class:`PanelFrame`.

    If ``data`` is already a :class:`PanelFrame` it is returned unchanged (its
    own keys win). Otherwise ``entity`` and ``time`` must be provided; if either
    is omitted the codebase convention is applied: **column 0 is the entity and
    column 1 is the time**.

    Parameters
    ----------
    data : polars frame, frame-like, or PanelFrame
    entity : str, optional
        Entity column. Defaults to the first column.
    time : str, optional
        Time column. Defaults to the second column.
    assume_sorted : bool, default=False
        Record in O(1) that the rows are already ``(entity, time)`` sorted; see
        :class:`PanelFrame`. Applied to an existing :class:`PanelFrame` too, via
        :meth:`PanelFrame.mark_sorted`.
    check_sorted : bool, default=False
        Verify the per-entity time order once, raising if it is violated.
        **Materialises.**

    Returns
    -------
    PanelFrame

    Raises
    ------
    ValueError
        If defaults are needed but the frame has fewer than two columns, or if
        ``check_sorted=True`` and the panel is not in per-entity time order.
    """
    if isinstance(data, PanelFrame):
        if check_sorted:
            return data.assert_sorted()
        return data.mark_sorted() if assume_sorted else data
    lf = _to_lazyframe(data)
    if entity is None or time is None:
        names = lf.collect_schema().names()
        if len(names) < 2:
            raise ValueError(
                "cannot infer panel keys: a panel needs at least an entity and a "
                f"time column, but the frame has columns {names}. Pass `entity=` "
                "and `time=` explicitly."
            )
        entity = entity if entity is not None else names[0]
        time = time if time is not None else names[1]
    return PanelFrame(
        lf,
        entity=entity,
        time=time,
        assume_sorted=assume_sorted,
        check_sorted=check_sorted,
    )
