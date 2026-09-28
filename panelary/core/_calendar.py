"""Calendar durations on real time values: the one place a "5 days" is computed.

Two parts of the package need to move a time value forward by a *calendar*
amount rather than by a number of rows:

* **embargo and purge** in :mod:`panelary.core.model_selection` -- "embargo five
  business days after each test block", not "five rows", which on an irregular
  axis (gaps, weekends, holidays, a listing that starts late) is a different and
  usually wrong amount of time;
* **publication lag** in :mod:`panelary.core.asof` -- a figure filed on day
  ``k`` is usable from ``k + lag``.

Both use the same small vocabulary, :data:`Duration`, and the same function,
:func:`shift_forward`, so "a duration" means one thing across the library.

Accepted duration specs
-----------------------
``int`` (including NumPy integers)
    A number of *steps*. On an integer time axis :func:`shift_forward` adds it
    to the value; the splitters keep their historical meaning (positions in the
    sorted unique-time index), which is why an ``int`` is never reinterpreted
    as days.
``datetime.timedelta`` / ``numpy.timedelta64``
    An exact elapsed duration (36 hours is 36 hours, across a DST change too).
    On a ``Date`` axis it must be a whole number of days.
``str``
    A Polars duration string -- ``"5d"``, ``"2w"``, ``"1mo"``, ``"1q"``,
    ``"1y"``, ``"6h"``, ``"1d12h"`` -- applied with
    :meth:`polars.Expr.dt.offset_by`, so months and years are *calendar*
    months and years (Jan 31 + ``"1mo"`` is the last day of February) and a day
    is a calendar day in the column's time zone. ``"<n>bd"`` is shorthand for
    :class:`BusinessDays` ``(n)``. Sub-day units are refused on a ``Date``
    axis (Polars would silently floor them), and ``"i"`` (index) units are
    refused everywhere -- pass an ``int`` for steps.
:class:`BusinessDays`
    ``n`` business days via :func:`numpy.busday_offset`, with an optional
    holiday list and week mask. A value that falls on a non-business day is
    first rolled *forward* to the next business day, then advanced ``n`` --
    the conservative reading for both a lag and an embargo (it never makes
    information available, or training data eligible, earlier).

Every duration must be non-negative: a negative lag or embargo would move
information *backwards* in time, which is exactly what these exist to stop.

This module imports only :mod:`numpy`, :mod:`polars` and the standard library.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import polars as pl

__all__ = [
    "BusinessDays",
    "Duration",
    "as_steps",
    "is_steps",
    "shift_forward",
    "validate_duration",
]


@dataclass(frozen=True)
class BusinessDays:
    """A duration of ``n`` business days, for embargoes and publication lags.

    Parameters
    ----------
    n : int
        Number of business days, ``>= 0``.
    holidays : iterable of date-like, optional
        Non-business dates (``datetime.date``, ISO strings or
        ``numpy.datetime64``), passed to :func:`numpy.busday_offset`.
    weekmask : str, default="1111100"
        Which weekdays are business days, Monday first (NumPy's format).

    Notes
    -----
    A time on a non-business day is rolled *forward* to the next business day
    before advancing, so ``BusinessDays(0)`` maps a Saturday to the following
    Monday and ``BusinessDays(1)`` maps a Friday to the next Monday. On a
    ``Datetime`` axis the time of day is kept and the day offset is applied in
    the column's own time zone.

    Examples
    --------
    >>> from panelary.core._calendar import BusinessDays
    >>> BusinessDays(5, holidays=["2024-12-25"]).n
    5
    """

    n: int
    holidays: tuple[np.datetime64, ...] = field(default=())
    weekmask: str = "1111100"

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)):
            raise TypeError(
                f"BusinessDays(n) needs an integer, got {type(self.n).__name__!r}."
            )
        if self.n < 0:
            raise ValueError(f"BusinessDays(n) must be >= 0, got {self.n}.")
        hols = tuple(np.datetime64(h, "D") for h in _as_iterable(self.holidays))
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "holidays", tuple(sorted(set(hols))))
        # Let NumPy validate the week mask / holidays once, here, with its own
        # error, rather than on the first shift.
        np.busdaycalendar(weekmask=self.weekmask, holidays=list(self.holidays))

    def offset_dates(self, dates: np.ndarray) -> np.ndarray:
        """Roll ``datetime64[D]`` dates forward to a business day, then add ``n``."""
        return np.busday_offset(
            dates.astype("datetime64[D]"),
            self.n,
            roll="forward",
            weekmask=self.weekmask,
            holidays=list(self.holidays),
        )

    def __str__(self) -> str:
        extra = f", {len(self.holidays)} holidays" if self.holidays else ""
        mask = "" if self.weekmask == "1111100" else f", weekmask={self.weekmask!r}"
        return f"{self.n} business day(s){extra}{mask}"


#: What a lag / embargo / purge horizon may be; see the module docstring.
Duration = int | np.integer | _dt.timedelta | np.timedelta64 | str | BusinessDays

#: One ``<count><unit>`` term of a Polars duration string. Order matters: the
#: two-letter units must be tried before their one-letter prefixes.
_TERM = r"\d+(?:ns|us|ms|mo|bd|y|q|w|d|h|m|s|i)"
_DURATION_RE = re.compile(rf"^(?:{_TERM})+$")
_TERM_RE = re.compile(r"(\d+)(ns|us|ms|mo|bd|y|q|w|d|h|m|s|i)")
_SUB_DAY_UNITS = frozenset({"h", "m", "s", "ms", "us", "ns"})
_NS_PER_DAY = 86_400 * 10**9


def _as_iterable(obj: Any) -> Iterable[Any]:
    if obj is None:
        return ()
    if isinstance(obj, (str, bytes, _dt.date, np.datetime64)):
        return (obj,)
    return obj


def is_steps(spec: object) -> bool:
    """True for an integer (row/step) spec, the historical meaning of an ``int``.

    ``numpy.timedelta64`` subclasses ``numpy.signedinteger``, so it is excluded
    explicitly: a timedelta is a duration, never a count of steps.
    """
    return isinstance(spec, (int, np.integer)) and not isinstance(spec, np.timedelta64)


def as_steps(spec: object) -> int:
    """The ``int`` behind an integer (step) spec; ``TypeError`` for a duration."""
    if not is_steps(spec):
        raise TypeError(f"expected an integer number of steps, got {spec!r}.")
    return int(cast(int, spec))


def validate_duration(spec: object, *, name: str) -> Duration:
    """Validate a duration spec and return its normal form.

    ``"<n>bd"`` becomes :class:`BusinessDays`; everything else is returned as
    given once checked. Integers keep the exact error message the splitters
    have always raised, so validation is backward compatible.

    Raises
    ------
    TypeError
        If ``spec`` is not one of the accepted kinds.
    ValueError
        If it is negative, malformed, or uses ``"i"`` (index) units.
    """
    if isinstance(spec, (float, np.floating)) and float(spec).is_integer():
        # Tolerated for backward compatibility (``embargo=0.0`` used to work):
        # an integral float is a step count.
        spec = int(spec)
    if is_steps(spec):
        if spec < 0:  # type: ignore[operator]
            raise ValueError(f"`{name}` must be >= 0, got {spec}.")
        return spec  # type: ignore[return-value]
    if isinstance(spec, BusinessDays):
        return spec
    if isinstance(spec, _dt.timedelta):
        if spec < _dt.timedelta(0):
            raise ValueError(f"`{name}` must be a non-negative duration, got {spec}.")
        return spec
    if isinstance(spec, np.timedelta64):
        if np.isnat(spec):
            raise ValueError(f"`{name}` must not be NaT.")
        if np.datetime_data(spec.dtype)[0] in ("Y", "M"):
            raise ValueError(
                f"`{name}`={spec!r}: NumPy month/year timedeltas have no fixed "
                "length; pass a calendar string such as '1mo' or '1y' instead."
            )
        if spec < np.timedelta64(0, "ns"):
            raise ValueError(f"`{name}` must be a non-negative duration, got {spec}.")
        return spec
    if isinstance(spec, str):
        text = spec.strip()
        if not _DURATION_RE.match(text):
            raise ValueError(
                f"`{name}`={spec!r} is not a non-negative duration string. Use "
                "Polars duration syntax such as '5d', '2w', '1mo', '1q', '1y', "
                "'6h', or '<n>bd' for business days."
            )
        terms = _TERM_RE.findall(text)
        units = {u for _, u in terms}
        if "i" in units:
            raise ValueError(
                f"`{name}`={spec!r}: 'i' (index) units count rows, not time; "
                "pass an int for a number of steps."
            )
        if "bd" in units:
            if len(terms) != 1:
                raise ValueError(
                    f"`{name}`={spec!r}: business days ('bd') cannot be combined "
                    "with other units in one duration."
                )
            return BusinessDays(int(terms[0][0]))
        return text
    raise TypeError(
        f"`{name}` must be an int (steps) or a duration -- a datetime.timedelta, "
        "numpy.timedelta64, Polars duration string ('5d', '1mo') or "
        f"BusinessDays -- got {type(spec).__name__!r}."
    )


def _timedelta_to_str(td: _dt.timedelta | np.timedelta64, *, date_axis: bool) -> str:
    """An exact duration as a Polars duration string.

    On a ``Datetime`` axis it is exact *elapsed* time (``"<n>ns"``), so a
    timedelta of 36 hours is 36 hours even across a DST change -- use the
    string ``"1d12h"`` for calendar-day semantics. On a ``Date`` axis it must be
    a whole number of days and becomes ``"<n>d"``.
    """
    if isinstance(td, np.timedelta64):
        total_ns = int(td.astype("timedelta64[ns]").astype(np.int64))
    else:
        total_ns = td.days * _NS_PER_DAY + td.seconds * 10**9 + td.microseconds * 10**3
    if not date_axis:
        return f"{total_ns}ns"
    days, rem = divmod(total_ns, _NS_PER_DAY)
    if rem:
        raise ValueError(
            f"duration {td!r} is not a whole number of days, but the time axis "
            "is a Date (day resolution); round it to whole days explicitly."
        )
    return f"{days}d"


def shift_forward(values: pl.Series, spec: Duration) -> pl.Series:
    """Return ``values`` moved forward in time by ``spec``. Nulls stay null.

    Parameters
    ----------
    values : polars.Series
        Time values: an integer, ``Date`` or ``Datetime`` series.
    spec : Duration
        A spec already passed through :func:`validate_duration`, or a raw one.

    Returns
    -------
    polars.Series
        Same dtype as ``values``.

    Raises
    ------
    TypeError
        If the spec does not fit the axis: an ``int`` on a temporal axis, or a
        calendar duration on a non-temporal one.
    ValueError
        If a sub-day duration is applied to a ``Date`` axis.
    """
    spec = validate_duration(spec, name="duration")
    dtype = values.dtype
    temporal = dtype == pl.Date or isinstance(dtype, pl.Datetime)
    if is_steps(spec):
        if dtype.is_integer():
            return values + int(spec)  # type: ignore[arg-type]
        raise TypeError(
            f"an integer duration ({spec}) counts steps of an integer time axis, "
            f"but this axis is {dtype}; pass a calendar duration such as "
            f"'{spec}d', datetime.timedelta(days={spec}) or BusinessDays({spec})."
        )
    if not temporal:
        raise TypeError(
            f"calendar duration {spec!r} needs a Date or Datetime time axis, but "
            f"this axis is {dtype}; on an integer axis pass an int number of steps."
        )
    date_axis = dtype == pl.Date
    if isinstance(spec, BusinessDays):
        return _shift_business_days(values, spec)
    if isinstance(spec, (_dt.timedelta, np.timedelta64)):
        spec = _timedelta_to_str(spec, date_axis=date_axis)
    assert isinstance(spec, str)
    if date_axis and any(u in _SUB_DAY_UNITS for _, u in _TERM_RE.findall(spec)):
        raise ValueError(
            f"duration {spec!r} has a sub-day part, but the time axis is a Date "
            "(day resolution) and Polars would silently floor it; use whole days."
        )
    return values.dt.offset_by(spec)


def _shift_business_days(values: pl.Series, spec: BusinessDays) -> pl.Series:
    """Roll forward to a business day, then add ``spec.n`` business days."""
    if values.dtype == pl.Date:
        mask = values.is_not_null().to_numpy()
        out = np.full(values.len(), np.datetime64("NaT", "D"), dtype="datetime64[D]")
        if mask.any():
            out[mask] = spec.offset_dates(values.to_numpy()[mask])
        # NaT converts to a Polars null, so null inputs stay null.
        return pl.Series(values.name, out, dtype=pl.Date)
    # Datetime: offset by whole calendar days in the column's own time zone, so
    # the time of day is kept and a DST change does not move it by an hour.
    local_dates = values.dt.date()
    mask = local_dates.is_not_null().to_numpy()
    deltas = np.zeros(values.len(), dtype=np.int64)
    if mask.any():
        d = local_dates.to_numpy()[mask].astype("datetime64[D]")
        deltas[mask] = (spec.offset_dates(d) - d).astype(np.int64)
    by = pl.Series("__by__", [f"{k}d" for k in deltas.tolist()], dtype=pl.String)
    return (
        pl.DataFrame({"__v__": values, "__by__": by})
        .select(pl.col("__v__").dt.offset_by(pl.col("__by__")))
        .to_series()
        .alias(values.name)
    )
