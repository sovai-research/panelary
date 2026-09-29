"""Future-perturbation leak verifiers: the keystone of Panelary's guarantee.

A transform is *leak-free* if its output at time ``t`` depends only on data at
times ``<= t`` (within each entity). Panelary's whole "leak-safe" claim is only
worth as much as our ability to *check* it, and the cleanest, model-agnostic
check is a **future-perturbation experiment**:

1. Run the operation on a panel and record its output.
2. Corrupt every value strictly in the *future* (per entity / per the shared
   time axis), leaving the past untouched.
3. Run the operation again.
4. If the operation is leak-free, every output cell in the *past* must be
   **bit-identical** (within ``tol``) across the two runs. Any difference is a
   look-ahead: information from a perturbed future row flowed backwards.

This module exposes two assertions built on that mechanism:

* :func:`assert_no_lookahead` — split the shared time axis at a cut ``t`` and
  assert that perturbing ``time > t`` never changes any output at ``time <= t``,
  at every cut of a deterministic spread along the axis.
* :func:`assert_no_train_test_leak` — perturb a *test* fold and assert that the
  *train*-fold outputs are unchanged (the CV-boundary version of the same idea).

Perturbation is not the whole story. It corrupts future *values*, so it can only
see an operator whose output *depends on* those values. An operator whose output
depends on how many rows follow ``t`` — a window, threshold, lag order or
normalisation constant derived from ``len(x)``, or a whole-series aggregate
broadcast back over the entity — is bit-identical under any value perturbation
and sails through. That is the second, independent defect, and it gets its own
instrument:

* :func:`assert_prefix_invariant` — truncate the panel to ``time <= cut``, re-run
  the op, and require every output cell to equal the full-panel run restricted to
  the same region: hard invariant 1, ``f(x[:T])[t] == f(x[:T+k])[t]``.

Neither assertion subsumes the other. Perturbation catches value-dependence;
prefix invariance catches length-dependence. A correct operator needs both.

All three accept ``op`` as either a :class:`polars.Expr` (applied via
``with_columns``) or a callable ``frame -> frame`` (the callable may take and
return a :class:`~panelary.core.panel_frame.PanelFrame`,
:class:`polars.DataFrame`, or :class:`polars.LazyFrame`; the calling convention
is auto-detected). Failures name the first offending ``(column, entity, time)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame, as_panel

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "assert_no_lookahead",
    "assert_no_train_test_leak",
    "assert_prefix_invariant",
]


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
def _as_pf(panel: Any, entity: str | None, time: str | None) -> PanelFrame:
    """Coerce ``panel`` into a :class:`PanelFrame` (keys of an existing one win)."""
    if isinstance(panel, PanelFrame):
        return panel
    return as_panel(panel, entity=entity, time=time)


def _to_df(obj: Any, *, method: str) -> pl.DataFrame:
    """Coerce an op result into an eager :class:`polars.DataFrame`."""
    if isinstance(obj, PanelFrame):
        return obj.collect()
    if isinstance(obj, pl.LazyFrame):
        return obj.collect()
    if isinstance(obj, pl.DataFrame):
        return obj
    raise TypeError(
        f"{method}: `op` returned {type(obj).__name__!r}; a leak check needs a "
        "PanelFrame, polars.DataFrame, or polars.LazyFrame back from `op`."
    )


def _make_apply(
    op: Any,
    pf: PanelFrame,
    df0: pl.DataFrame,
    *,
    method: str = "assert_no_lookahead",
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Return a ``DataFrame -> DataFrame`` runner for ``op``.

    For a :class:`polars.Expr`, the runner is ``df.with_columns(op)``. For a
    callable, the input calling-convention (bare DataFrame, PanelFrame, or
    LazyFrame) is detected once against ``df0`` and then reused, so both the
    baseline and perturbed runs are driven identically.
    """
    entity, time = pf.entity_col, pf.time_col

    if isinstance(op, pl.Expr):
        return lambda df: df.with_columns(op)

    if not callable(op):
        raise TypeError(
            "`op` must be a polars.Expr or a callable frame->frame, got "
            f"{type(op).__name__!r}."
        )

    wrappers: list[Callable[[pl.DataFrame], Any]] = [
        lambda d: d,  # bare DataFrame
        lambda d: PanelFrame(d, entity=entity, time=time, validate=False),
        lambda d: d.lazy(),  # LazyFrame
    ]
    chosen: Callable[[pl.DataFrame], Any] | None = None
    last_exc: Exception | None = None
    for wrap in wrappers:
        try:
            _to_df(op(wrap(df0)), method=method)
        except Exception as exc:  # noqa: BLE001 - probing calling conventions
            last_exc = exc
            continue
        chosen = wrap
        break
    if chosen is None:
        raise TypeError(
            "could not call `op`: it did not accept a polars DataFrame, "
            "PanelFrame, or LazyFrame (or did not return a frame). Last error: "
            f"{last_exc!r}"
        )

    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return _to_df(op(chosen(df)), method=method)

    return apply


def _numeric_feature_cols(df: pl.DataFrame, entity: str, time: str) -> list[str]:
    """Numeric columns that are neither the entity nor the time key."""
    keys = {entity, time}
    return [
        name
        for name, dtype in df.schema.items()
        if name not in keys and dtype.is_numeric()
    ]


def _perturb(
    df: pl.DataFrame,
    cols: Sequence[str],
    mask: pl.Expr,
    *,
    seed: int = 0,
) -> pl.DataFrame:
    """Return a copy of ``df`` with a large perturbation added to ``cols``.

    Only rows where ``mask`` is True are altered; the entity/time keys and all
    unmasked rows are left byte-for-byte unchanged. The perturbation is a large
    (~1e6) random offset so any leak shows up far above ``tol``.
    """
    rng = np.random.default_rng(seed)
    out = df
    h = df.height
    for c in cols:
        noise = pl.Series(f"__noise_{c}__", rng.standard_normal(h) * 1.0e6 + 1.0e3)
        out = out.with_columns(noise).with_columns(
            pl.when(mask)
            .then(pl.col(c) + pl.col(f"__noise_{c}__"))
            .otherwise(pl.col(c))
            .alias(c)
        )
        out = out.drop(f"__noise_{c}__")
    return out


def _nan_mask(s: pl.Series) -> pl.Series:
    """Boolean mask, True exactly where ``s`` holds a floating-point NaN.

    Two Polars details make this worth a helper. ``Series.is_nan()`` returns
    *null* — not ``False`` — for null entries, so its raw result cannot be used
    in boolean algebra; and it raises outright for numeric dtypes that cannot
    represent NaN (``Decimal``). Both are normalised to ``False`` here.
    """
    if s.dtype.is_float():
        return s.is_nan().fill_null(value=False)
    return pl.repeat(False, s.len(), dtype=pl.Boolean, eager=True)


def _first_mismatch(
    base: pl.DataFrame,
    pert: pl.DataFrame,
    *,
    entity: str,
    time: str,
    tol: float,
) -> tuple[str, Any, Any] | None:
    """Return ``(column, entity_value, time_value)`` of the first differing cell.

    Both frames must already be filtered to the comparison region and sorted by
    ``(entity, time)``. Returns ``None`` if everything matches within ``tol``.
    """
    if base.height != pert.height:
        raise AssertionError(
            "leak check: the operation changed the number of rows in the "
            f"comparison region ({base.height} vs {pert.height}); the op must "
            "preserve the (entity, time) keys so past rows can be compared."
        )
    common = [c for c in base.columns if c in pert.columns]
    for c in common:
        bs = base[c]
        ps = pert[c]
        if bs.dtype.is_numeric() and ps.dtype.is_numeric():
            # In Polars, null and NaN are distinct, `fill_null` does not touch
            # NaN, and NaN orders *above* every float — so the naive
            # `(bs - ps).abs().fill_null(0.0) > tol` reports NaN-vs-NaN as a
            # difference. That is a false positive, and it flags legitimately
            # causal transforms: an expanding min-max scaler emits 0/0 = NaN at
            # each entity's first observation. Handle the four missing-value
            # cases explicitly instead:
            #     null vs null -> equal        NaN vs NaN    -> equal
            #     null vs NaN  -> mismatch     NaN vs number -> mismatch
            b_null, p_null = bs.is_null(), ps.is_null()
            b_nan, p_nan = _nan_mask(bs), _nan_mask(ps)
            missing = (b_null ^ p_null) | (b_nan ^ p_nan)
            # `tol` keeps its meaning, but only where both sides are ordinary
            # numbers; NaN and null rows are already decided above.
            ordinary = ~(b_null | p_null | b_nan | p_nan)
            big = ((bs - ps).abs() > tol).fill_null(value=False) & ordinary
            mism = missing | big
        else:
            mism = bs.ne_missing(ps)
        if bool(mism.any()):
            idx = int(mism.arg_true()[0])
            return c, base[entity][idx], base[time][idx]
    return None


def _require_keys(out: pl.DataFrame, entity: str, time: str, label: str) -> None:
    """Raise unless ``out`` still carries both panel keys, so rows can align."""
    cols = out.columns
    if entity not in cols or time not in cols:
        raise AssertionError(
            f"leak check: the {label} output dropped a key column "
            f"({entity!r}/{time!r}); the op must return a frame that still "
            "carries the (entity, time) keys so outputs can be aligned."
        )


#: One perturbation experiment: ``(perturb_mask, compare_mask, kept_desc,
#: changed_desc)``.
_Region = tuple[pl.Expr, pl.Expr, str, str]


def _assert_invariant(
    op: Any,
    pf: PanelFrame,
    *,
    regions: Sequence[_Region],
    tol: float,
) -> None:
    """Core mechanism shared by the perturbation assertions.

    For each region, perturb the rows selected by its ``perturb_mask``, re-run
    ``op``, and assert every output cell in its ``compare_mask`` region is
    unchanged within ``tol``. The unperturbed baseline is computed once and
    shared by every region, so each extra region costs one run of ``op``.
    """
    entity, time = pf.entity_col, pf.time_col
    df = pf.collect()
    num_cols = _numeric_feature_cols(df, entity, time)
    if not num_cols:
        raise ValueError(
            "leak check: the panel has no numeric feature columns to perturb; "
            "provide a panel whose features are numeric so the future can be "
            "corrupted meaningfully."
        )

    apply = _make_apply(op, pf, df)
    base_out = apply(df)
    _require_keys(base_out, entity, time, "baseline")

    for perturb_mask, compare_mask, kept_desc, changed_desc in regions:
        pert_out = apply(_perturb(df, num_cols, perturb_mask))
        _require_keys(pert_out, entity, time, "perturbed")

        base_cmp = base_out.filter(compare_mask).sort([entity, time])
        pert_cmp = pert_out.filter(compare_mask).sort([entity, time])

        hit = _first_mismatch(base_cmp, pert_cmp, entity=entity, time=time, tol=tol)
        if hit is not None:
            col, ent_val, time_val = hit
            raise AssertionError(
                "LOOK-AHEAD LEAK DETECTED: perturbing the future "
                f"({changed_desc}) changed output column {col!r} at "
                f"{entity}={ent_val!r}, {time}={time_val!r}, which lies in the "
                f"protected region ({kept_desc}). A leak-free operation's output "
                "at a given time may depend only on data at that time or earlier "
                "(within the entity). Express the feature walk-forward, e.g. "
                f"`expr.shift(k).over({entity!r})` with k >= 0, or a trailing "
                "rolling window."
            )


# --------------------------------------------------------------------------- #
# Where to cut
# --------------------------------------------------------------------------- #
#: Anchors of the default cut spread, as fractions of the distinct-time axis.
_CUT_ANCHORS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


def _default_cut_positions(n_times: int) -> list[int]:
    """Positions ``i`` of the default cuts ``times[i]`` on an ``n_times`` axis.

    A consecutive **pair** of cuts at each of four anchors spread along the
    axis. One cut is not enough, and neither is a handful of round fractions:
    a leak confined to a calendar period (a period mean broadcast back, a
    period-end value) is invisible at a cut on a period's last step, and a
    period length can divide ``n_times / 2`` or ``n_times / 4`` just as easily
    as it divides nothing. Of two consecutive cuts at most one can end a
    period longer than one step, so every pair has a cut strictly inside one.
    The last time is never a cut (nothing would lie beyond it).
    """
    positions: set[int] = set()
    for frac in _CUT_ANCHORS:
        i = int(frac * n_times) - 1
        positions.update((i, i + 1))
    return sorted(i for i in positions if 0 <= i <= n_times - 2)


def _default_cuts(times: Sequence[Any]) -> list[Any]:
    """The cut points the leak assertions test when none are given.

    Parameters
    ----------
    times : sequence
        The sorted distinct times of the panel.

    Returns
    -------
    list
        Up to eight time values: a consecutive pair at 20%, 40%, 60% and 80% of
        the axis (fewer on a short axis). Deterministic, so a failure
        reproduces.

    Examples
    --------
    >>> _default_cuts(list(range(10)))
    [1, 2, 3, 4, 5, 6, 7, 8]
    >>> _default_cuts(list(range(240)))
    [47, 48, 95, 96, 143, 144, 191, 192]
    """
    times = list(times)
    return [times[i] for i in _default_cut_positions(len(times))]


def _resolve_cuts(
    times: Sequence[Any], cut: Any, cuts: Sequence[Any] | None, *, method: str
) -> list[Any]:
    """``cut`` / ``cuts`` / the default spread, validated against the axis."""
    if cut is not None and cuts is not None:
        raise ValueError(f"{method}: pass `cut` or `cuts`, not both.")
    if cut is None and cuts is None:
        return _default_cuts(times)
    chosen = [cut] if cut is not None else list(cuts or [])
    if not chosen:
        raise ValueError(f"{method}: `cuts` must name at least one cut.")
    truncate = method == "assert_prefix_invariant"
    for c in chosen:
        if c >= times[-1]:
            what = (
                "truncates nothing" if truncate else "leaves no future rows to perturb"
            )
            raise ValueError(
                f"`cut={c!r}` {what} (max time is {times[-1]!r}); choose a smaller cut."
            )
        if c < times[0]:
            what = (
                "leaves no rows to compare"
                if truncate
                else "leaves no past rows to protect"
            )
            raise ValueError(
                f"`cut={c!r}` {what} (min time is {times[0]!r}); choose a larger cut."
            )
    return chosen


# --------------------------------------------------------------------------- #
# Public assertions
# --------------------------------------------------------------------------- #
def assert_no_lookahead(
    op: Any,
    panel: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    tol: float = 1e-9,
    cut: Any = None,
    cuts: Sequence[Any] | None = None,
) -> None:
    """Assert ``op`` never looks ahead on ``panel`` (future-perturbation test).

    Splits the shared, sorted unique-time axis at a cut and verifies that
    corrupting every value at ``time > cut`` leaves every output value at
    ``time <= cut`` bit-identical (within ``tol``) -- at **every** cut tested.
    By default that is a consecutive pair of cuts at
    20%, 40%, 60% and 80% of the axis. A single cut is not a test: a leak
    confined to a calendar period (a period mean broadcast back to its rows) is
    invisible at a cut on a period's last step. On 250 dates with 5-step
    periods, such a leak passed the old single median cut and failed at 200 of
    the other 249.

    Parameters
    ----------
    op : polars.Expr or callable
        The operation under test. A :class:`polars.Expr` is applied via
        ``frame.with_columns(op)``. A callable is invoked as ``op(frame)`` and
        may accept/return a :class:`PanelFrame`, :class:`polars.DataFrame`, or
        :class:`polars.LazyFrame` (auto-detected).
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        A long-format panel. Bare frames are wrapped via
        :func:`~panelary.core.panel_frame.as_panel`.
    entity, time : str, optional
        Panel keys, used only when ``panel`` is a bare frame.
    tol : float, default=1e-9
        Absolute tolerance for the "unchanged" comparison of numeric outputs.
    cut : optional
        A single time value to split at (``time <= cut`` is protected).
    cuts : sequence, optional
        Several time values to split at, each tested in turn. Pass at most one
        of ``cut`` and ``cuts``; with neither, the default spread above is used.
        Each cut costs one extra run of ``op``.

    Raises
    ------
    AssertionError
        If any protected (past) output cell changes when the future is
        perturbed — i.e. the operation leaks look-ahead information. The message
        names the cut and the first offending column, entity, and time.
    ValueError
        If the panel has fewer than two distinct times, or no numeric features,
        or a cut leaves no past or no future.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.core.panel_frame import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"e": ["a"] * 4, "t": [0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 4.0]}
    ... )
    >>> panel = PanelFrame(df, entity="e", time="t")
    >>> assert_no_lookahead(pl.col("x").shift(1).over("e").alias("lag"), panel)
    >>> assert_no_lookahead(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     pl.col("x").shift(-1).over("e").alias("lead"), panel
    ... )
    Traceback (most recent call last):
    AssertionError: LOOK-AHEAD LEAK DETECTED: ...
    """
    pf = _as_pf(panel, entity, time)
    times = pf.time_index().to_list()
    if len(times) < 2:
        raise ValueError(
            "assert_no_lookahead needs at least two distinct time steps to split "
            f"past from future, got {len(times)}."
        )
    chosen = _resolve_cuts(times, cut, cuts, method="assert_no_lookahead")

    tcol = pf.time_col
    _assert_invariant(
        op,
        pf,
        regions=[
            (
                pl.col(tcol) > c,
                pl.col(tcol) <= c,
                f"{tcol} <= {c!r}",
                f"{tcol} > {c!r}",
            )
            for c in chosen
        ],
        tol=tol,
    )


def _split_times(x: Any, pf: PanelFrame) -> list[Any]:
    """Extract the set of time values represented by one side of a split."""
    if isinstance(x, PanelFrame):
        return x.collect()[x.time_col].unique().to_list()
    if isinstance(x, pl.LazyFrame):
        return x.select(pf.time_col).collect()[pf.time_col].unique().to_list()
    if isinstance(x, pl.DataFrame):
        return x[pf.time_col].unique().to_list()
    if isinstance(x, pl.Series):
        return x.unique().to_list()
    # Fall back to an array-like of time values.
    return list(np.asarray(x).tolist())


def assert_no_train_test_leak(
    op: Any,
    panel: Any,
    split: tuple[Any, Any],
    *,
    entity: str | None = None,
    time: str | None = None,
    tol: float = 1e-9,
) -> None:
    """Assert perturbing the *test* fold leaves *train*-fold outputs unchanged.

    The cross-validation-boundary form of :func:`assert_no_lookahead`: given a
    ``(train, test)`` split, corrupt every value at the test-fold times and
    verify that no output at a train-fold time changes (within ``tol``). This
    catches transforms that let test-fold information bleed into train-fold
    features.

    Parameters
    ----------
    op : polars.Expr or callable
        The operation under test (see :func:`assert_no_lookahead`).
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        The full panel the op runs on.
    split : (train, test)
        A pair whose two members identify the train and test rows. Each member
        may be a :class:`PanelFrame`, a polars frame, or an array-like of time
        values; only its set of time values is used.
    entity, time : str, optional
        Panel keys, used only when ``panel`` is a bare frame.
    tol : float, default=1e-9
        Absolute tolerance for the "unchanged" comparison.

    Raises
    ------
    AssertionError
        If any train-fold output changes when the test fold is perturbed.
    ValueError
        If the split is malformed or the panel has no numeric features.

    Notes
    -----
    For an *interior* test block (train rows exist on both sides of it), even a
    correct backward-looking feature computed on a post-test train row will
    legitimately depend on test-period values; that is precisely why purging
    exists. Use this assertion with walk-forward splits (train entirely before
    test) or with the purged train set to check the guarantee you actually rely
    on.
    """
    if not (isinstance(split, (tuple, list)) and len(split) == 2):
        raise ValueError(
            "`split` must be a (train, test) pair, got "
            f"{type(split).__name__!r} of length "
            f"{len(split) if hasattr(split, '__len__') else '?'}."
        )
    pf = _as_pf(panel, entity, time)
    train_times = _split_times(split[0], pf)
    test_times = _split_times(split[1], pf)
    if not test_times:
        raise ValueError("`split` test fold is empty; nothing to perturb.")
    if not train_times:
        raise ValueError("`split` train fold is empty; nothing to protect.")

    tcol = pf.time_col
    test_lit = pl.Series(values=list(test_times)).implode()
    train_lit = pl.Series(values=list(train_times)).implode()
    _assert_invariant(
        op,
        pf,
        regions=[
            (
                pl.col(tcol).is_in(test_lit),
                pl.col(tcol).is_in(train_lit),
                "the train fold",
                "the test fold",
            )
        ],
        tol=tol,
    )


# --------------------------------------------------------------------------- #
# Prefix invariance: the length-dependence instrument
# --------------------------------------------------------------------------- #
def assert_prefix_invariant(
    op: Any,
    panel: Any,
    *,
    entity: str | None = None,
    time: str | None = None,
    tol: float = 1e-9,
    cut: Any = None,
    cuts: Sequence[Any] | None = None,
) -> None:
    """Assert ``op`` is prefix-invariant: its output never depends on later rows.

    Truncates the panel to ``time <= cut``, re-runs ``op`` on that prefix, and
    requires every output cell to equal the full-panel run restricted to the
    same region. That is hard invariant 1 stated operationally::

        f(x[:T])[t] == f(x[:T+k])[t]   for all t <= T

    **This is not the same check as** :func:`assert_no_lookahead`, **and neither
    subsumes the other.** ``assert_no_lookahead`` perturbs future *values*, so it
    catches *value*-dependence: an output at ``t`` that moves when a later
    observation changes. It is structurally blind to *length*-dependence, where
    the output at ``t`` moves because there are simply more rows after ``t`` —
    under a value perturbation the row count is untouched and the output is
    bit-identical, so the perturbation test passes. ``pl.col("x").count()`` is
    the canonical example: perturbation says clean, yet the value at row 0
    changes the moment the series grows. Conversely, this assertion is only
    evaluated at a handful of cuts, and it requires the op to run at all on a
    short frame — an op with a minimum-history requirement, or one whose
    truncated run legitimately errors, can only be interrogated by perturbation,
    which holds the panel's size fixed and names the offending *value* rather
    than merely the offending cell. Run **both**, and read a pass from either as
    evidence about one failure mode only.

    Parameters
    ----------
    op : polars.Expr or callable
        The operation under test, with exactly the calling conventions of
        :func:`assert_no_lookahead`: a :class:`polars.Expr` is applied via
        ``frame.with_columns(op)``; a callable is invoked as ``op(frame)`` and
        may accept/return a :class:`PanelFrame`, :class:`polars.DataFrame`, or
        :class:`polars.LazyFrame` (auto-detected).
    panel : PanelFrame | polars.DataFrame | polars.LazyFrame
        A long-format panel. Bare frames are wrapped via
        :func:`~panelary.core.panel_frame.as_panel`.
    entity, time : str, optional
        Panel keys, used only when ``panel`` is a bare frame.
    tol : float, default=1e-9
        Absolute tolerance for the "unchanged" comparison of numeric outputs.
        NaN compares equal to NaN and null to null; NaN-vs-number and
        null-vs-NaN are mismatches.
    cut : optional
        A single truncation point to test (the panel is cut to ``time <= cut``).
    cuts : sequence, optional
        Several truncation points, each tested in turn. Pass at most one of
        ``cut`` and ``cuts``. With neither, the default spread is used -- the
        same consecutive pairs of cuts as :func:`assert_no_lookahead`, since one
        prefix length can match the full panel by coincidence, and a cut on a
        period's last step hides a period-confined leak.

    Raises
    ------
    AssertionError
        If any output cell in the truncated region differs between the prefix
        run and the full run — i.e. the operation is length-dependent. The
        message names the first offending column, entity, and time.
    ValueError
        If the panel has fewer than two distinct times, or ``cut`` lies outside
        the time axis.

    See Also
    --------
    assert_no_lookahead : the value-perturbation half of the pair.
    assert_no_train_test_leak : the CV-boundary form of the perturbation test.

    Examples
    --------
    >>> import polars as pl
    >>> from panelary.core.panel_frame import PanelFrame
    >>> df = pl.DataFrame(
    ...     {"e": ["a"] * 4, "t": [0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 4.0]}
    ... )
    >>> panel = PanelFrame(df, entity="e", time="t")
    >>> assert_prefix_invariant(pl.col("x").cum_sum().over("e").alias("c"), panel)
    >>> assert_prefix_invariant(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     pl.col("x").count().over("e").alias("n"), panel
    ... )
    Traceback (most recent call last):
    AssertionError: PREFIX-INVARIANCE VIOLATION: ...
    """
    pf = _as_pf(panel, entity, time)
    ecol, tcol = pf.entity_col, pf.time_col
    times = pf.time_index().to_list()
    if len(times) < 2:
        raise ValueError(
            "assert_prefix_invariant needs at least two distinct time steps to "
            f"form a proper prefix, got {len(times)}."
        )
    chosen = _resolve_cuts(times, cut, cuts, method="assert_prefix_invariant")

    df = pf.collect()
    apply = _make_apply(op, pf, df, method="assert_prefix_invariant")
    full_out = apply(df)
    _require_keys(full_out, ecol, tcol, "full-panel")

    for c in chosen:
        prefix_out = apply(df.filter(pl.col(tcol) <= c))
        _require_keys(prefix_out, ecol, tcol, "truncated-panel")

        full_cmp = full_out.filter(pl.col(tcol) <= c).sort([ecol, tcol])
        pref_cmp = prefix_out.filter(pl.col(tcol) <= c).sort([ecol, tcol])

        hit = _first_mismatch(full_cmp, pref_cmp, entity=ecol, time=tcol, tol=tol)
        if hit is not None:
            col, ent_val, time_val = hit
            raise AssertionError(
                "PREFIX-INVARIANCE VIOLATION: truncating the panel to "
                f"{tcol} <= {c!r} changed output column {col!r} at "
                f"{ecol}={ent_val!r}, {tcol}={time_val!r}. The value at a given "
                "time therefore depends on how many rows come *after* it, so "
                "f(x[:T])[t] != f(x[:T+k])[t]: a window, threshold, lag order or "
                "normalisation constant derived from len(x), or a whole-series "
                "aggregate broadcast back over the entity. `assert_no_lookahead` "
                "cannot see this — it perturbs future values, and a "
                "length-dependent quantity is bit-identical under that "
                "perturbation — so the two assertions must both be run. Derive "
                "the quantity from the trailing history only, e.g. "
                f"`expr.cum_count().over({ecol!r})` rather than "
                f"`expr.count().over({ecol!r})`."
            )
