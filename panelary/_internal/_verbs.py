"""The golden path: eight top-level verbs over Panelary's subpackages.

Panelary is deep -- ``select``, ``reduce``, ``cluster``, ``detect``, ``econ`` and
``feature_extractors`` each expose a dozen or more estimators and functions. This
module is the shallow end: one verb per task, in a single uniform shape, so that
the common case reads as one coherent API::

    import panelary as pn

    pn.impute(df)                       # fill gaps, point-in-time
    pn.features(df)                     # per-entity feature table
    pn.select(df, target="ret", k=10)   # keep the informative columns
    pn.reduce(df, n_components=3)       # compress them
    pn.cluster(df)                      # group entities
    pn.bubbles(df, min_window=50)       # explosive-regime statistic
    pn.regression(df, y="ret", x=["size"], absorb=["ticker", "date"])
    pn.causal(df, y="ret", treatment="flow")

Every verb has the same shape::

    verb(data, *, method=..., columns=..., <verb args>, entity=None, time=None,
         **kwargs)

* ``data`` is always first and positional: a :class:`~panelary.core.panel_frame.PanelFrame`
  or a bare ``polars`` frame in long ``(entity, time, *features)`` form.
* ``method`` is always the first keyword and always has a working default.
* ``columns`` always names the subset of feature columns to work on
  (``None`` = every numeric feature column).
* ``entity`` / ``time`` are always last and are only consulted for bare polars
  frames; a :class:`PanelFrame` carries its own keys.
* ``**kwargs`` always forwards, unchanged, to the underlying implementation --
  which is where the real documentation lives.

Nothing here implements an algorithm. Each verb resolves ``method`` to an
existing public callable and forwards to it; the docstrings name that callable
so the escape hatch is always one hop away.

Five verbs are plain functions. ``select``, ``cluster`` and ``reduce`` share a
name with a subpackage, so they are :class:`_SubpackageVerb` instances instead:
calling one runs the verb, and any attribute it lacks is resolved on the module
of the same name, which keeps ``pn.select.mrmr`` working. See that class for why
this is only done where there is an actual collision.

**Leak-safety.** Panelary's contract (see ``docs/leakage.md``) is that
within-entity work stays inside its entity and in time order, and that anything
with a ``fit`` is fit per fold on training rows only. A convenience verb cannot
enforce the second half for you: every verb below that fits something fits it on
exactly the rows you hand it, which is correct inside a fold and *wrong* on the
full sample when the result is later evaluated per fold. Each docstring has a
``Leak-safety`` section stating precisely which regime it is in. Two verbs are
strictly point-in-time regardless of how they are called (:func:`impute` with
its default, :func:`bubbles`); the rest are fit-on-what-you-pass.

This module imports nothing heavier than the standard library at module scope --
every ``polars`` / ``numpy`` / subpackage import happens inside a function body,
so ``import panelary`` stays inside its cold-import budget (see
``tests/test_import_hygiene.py``).
"""

from __future__ import annotations

import contextlib
import importlib
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence
    from types import ModuleType

    import polars as pl

    from panelary.core.panel_frame import PanelFrame
    from panelary.econ._dml import DMLResult, PDSResult
    from panelary.econ._famamacbeth import FamaMacBethResult
    from panelary.econ._hdfe import HDFEResult
    from panelary.econ._panel import PanelFitResult

__all__ = [
    "bubbles",
    "causal",
    "cluster",
    "features",
    "impute",
    "reduce",
    "regression",
    "select",
]

_P = ParamSpec("_P")
_R = TypeVar("_R")


# --------------------------------------------------------------------------- #
# Three verbs share a name with a subpackage
# --------------------------------------------------------------------------- #
class _SubpackageVerb(Generic[_P, _R]):
    """A verb that is also the door to the subpackage of the same name.

    ``select``, ``cluster`` and ``reduce`` are both verbs and subpackages, and
    only one of them can own the attribute ``panelary.select``. A plain function
    wins the name and takes the module's contents down with it: ``pn.select.mrmr``
    then raises ``AttributeError: 'function' object has no attribute 'mrmr'``,
    which names neither the cause nor the fix -- for a call the README itself
    documented. So the three colliding names are instances of this class instead:
    calling one runs the verb, and any attribute it does not have is resolved on
    the real module. All four spellings then work::

        pn.select(df, target="ret", k=10)   # the verb
        pn.select.mrmr                      # the module's symbol
        from panelary.select import mrmr    # untouched (goes via sys.modules)
        import panelary.select as s         # binds this object; s.mrmr works

    The module is imported through :func:`importlib.import_module` on first
    attribute miss, so wiring a verb costs no import time. The other five verbs
    collide with nothing and stay plain functions -- this machinery exists for a
    name conflict, not for decoration.

    Typing note: an instance with ``__call__`` is callable to a type checker,
    and the ``ParamSpec`` makes it callable *with the wrapped verb's signature*,
    so ``pn.select(df, k="oops")`` is still an error. The obvious alternative --
    making the module object itself callable by reassigning ``__class__`` --
    would have mypy report "Module not callable" at every user call site, and
    would hide the verb's docstring behind the module's.
    """

    def __init__(self, func: Callable[_P, _R], module: str) -> None:
        self._func = func
        self._module_name = module
        # Look like the verb to `help()`, `inspect.signature` and IDE hovers.
        # `__wrapped__` is what makes `inspect.signature` report the verb's
        # signature rather than `(*args, **kwargs)`.
        self.__wrapped__ = func
        self.__doc__ = func.__doc__
        self.__name__ = getattr(func, "__name__", module.rpartition(".")[2])
        self.__qualname__ = getattr(func, "__qualname__", self.__name__)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Run the verb."""
        return self._func(*args, **kwargs)

    @property
    def _module(self) -> ModuleType:
        """The subpackage this verb shares its name with (imported on demand)."""
        return importlib.import_module(self._module_name)

    def __getattr__(self, name: str) -> Any:
        """Resolve anything the verb does not have on the subpackage.

        Only called when normal lookup fails, and it reads ``__dict__``
        directly so that a half-initialised instance cannot recurse.
        """
        module_name = self.__dict__.get("_module_name")
        if module_name is None:  # pragma: no cover - only during construction
            raise AttributeError(name)
        try:
            return getattr(importlib.import_module(module_name), name)
        except AttributeError:
            raise AttributeError(
                f"{name!r} is neither an argument of the "
                f"pn.{self.__dict__.get('__name__', module_name)}() verb nor a "
                f"public name in {module_name!r}."
            ) from None

    def __dir__(self) -> list[str]:
        """Merge the verb's own attributes with the subpackage's, for completion."""
        names = set(object.__dir__(self))
        with contextlib.suppress(ImportError):  # pragma: no branch
            names |= set(dir(importlib.import_module(self._module_name)))
        return sorted(names)

    def __repr__(self) -> str:
        return (
            f"<panelary verb {self.__name__}(...), "
            f"also the {self._module_name} subpackage>"
        )


def _subpackage_verb(
    module: str,
) -> Callable[[Callable[_P, _R]], _SubpackageVerb[_P, _R]]:
    """Decorator: bind a verb to the subpackage whose name it takes over."""

    def decorate(func: Callable[_P, _R]) -> _SubpackageVerb[_P, _R]:
        return _SubpackageVerb(func, module)

    return decorate


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _unknown_method(verb: str, method: str, choices: Sequence[str]) -> ValueError:
    """Build the one error message every verb raises for a bad ``method``."""
    return ValueError(
        f"pn.{verb}: unknown method {method!r}. Choose one of {sorted(choices)}."
    )


def _as_list(columns: str | Sequence[str] | None) -> list[str] | None:
    """Normalise a ``columns`` argument to a list (or ``None``).

    A bare string is one column name, never a sequence of characters.
    """
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _resolve_columns(
    panel: PanelFrame, columns: str | Sequence[str] | None
) -> list[str]:
    """Resolve ``columns`` to a concrete list of feature columns.

    ``None`` means "every numeric column that is neither the entity nor the
    time key", matching the convention used by ``extract_features`` and the
    reducers. A single string is accepted as a one-element list.
    """
    explicit = _as_list(columns)
    if explicit is not None:
        return explicit
    schema = panel.schema
    resolved = [c for c in panel.feature_cols if schema[c].is_numeric()]
    if not resolved:
        raise ValueError(
            "no numeric feature column found "
            f"(entity={panel.entity_col!r}, time={panel.time_col!r}). "
            "Pass `columns=`."
        )
    return resolved


# --------------------------------------------------------------------------- #
# 1. impute
# --------------------------------------------------------------------------- #
def impute(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str | int | float = "ffill",
    columns: str | Sequence[str] | None = None,
    allow_leaky: bool = False,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> PanelFrame | pl.DataFrame | pl.LazyFrame:
    """Fill missing values in a panel, within each entity.

    Thin facade over :func:`panelary.preprocessing.impute` (the expression-based
    methods) and :func:`panelary.preprocessing.cafe_impute` (the model-based
    one). Only numeric feature columns are touched; the ``(entity, time)`` keys
    and any non-numeric columns pass through untouched and the input column
    order is preserved.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel. A bare frame is wrapped with
        :func:`~panelary.core.panel_frame.as_panel` using ``entity`` / ``time``
        (defaulting to column 0 and column 1).
    method : {"ffill", "cafe", "mean", "median", "fill", "bfill", "interpolate"} \
or int or float, default "ffill"
        The fill rule. A number fills with that constant. See *Leak-safety*
        below -- the methods differ sharply in what they are allowed to see.
    columns : str or sequence of str, optional
        Restrict imputation to these numeric columns. Supported by
        ``method="cafe"`` only; the expression-based methods always fill every
        numeric feature column.
    allow_leaky : bool, default False
        Acknowledge and silence the :class:`~panelary.preprocessing.LeakageWarning`
        raised by ``"bfill"`` / ``"interpolate"``.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to :func:`~panelary.preprocessing.cafe_impute` when
        ``method="cafe"`` (``engine``, ``add_uncertainty``, ``add_anomaly``,
        ``add_missingness``). Rejected for the other methods, which take no
        further options. Note that ``add_recoverability=True`` is accepted by
        the signature but raises ``NotImplementedError``: CAFE does not expose
        the prior variance needed to normalise the score.

    Returns
    -------
    PanelFrame | polars.DataFrame | polars.LazyFrame
        The imputed panel, in the same container the caller passed in
        (``PanelFrame`` in, ``PanelFrame`` out; eager in, eager out). Columns
        that CAFE *adds* (e.g. ``add_uncertainty=True``) are appended at the end.

    Raises
    ------
    ValueError
        If ``method`` is not a supported name/constant, or if ``columns`` is
        given for a method that cannot honour it.
    ImportError
        If ``method="cafe"`` and the optional ``cafe`` dependency is missing
        (``pip install 'panelary[cafe]'``).

    Warns
    -----
    panelary.preprocessing.LeakageWarning
        When ``method`` is ``"bfill"`` or ``"interpolate"`` and ``allow_leaky``
        is False.

    Leak-safety
    -----------
    Every method is ``panel_safe``: filling never crosses an entity boundary.
    They are **not** equally ``leakage_safe``, and the default is the safe one:

    * ``"ffill"`` (default) and ``"cafe"`` are strictly **point-in-time**: a
      filled cell at ``t`` uses only information available at ``t``. Safe
      anywhere, including on a test fold.
    * ``"mean"``, ``"median"``, ``"fill"`` use the entity's **whole-sample**
      statistic, so a value at ``t`` depends on observations after ``t``. Use
      them for exploration, or fit-per-fold via a transformer -- not on a full
      sample that is later cross-validated.
    * ``"bfill"``, ``"interpolate"`` read **future rows** directly. They warn,
      and the warning is the point.

    Examples
    --------
    >>> import polars as pl
    >>> import panelary as pn
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "A", "A", "B", "B", "B"],
    ...         "date": [1, 2, 3, 1, 2, 3],
    ...         "px": [1.0, None, 3.0, 10.0, None, 12.0],
    ...     }
    ... )
    >>> pn.impute(df)["px"].to_list()
    [1.0, 1.0, 3.0, 10.0, 10.0, 12.0]

    See Also
    --------
    panelary.preprocessing.impute : the expression-based implementation.
    panelary.preprocessing.cafe_impute : the point-in-time model-based one.
    panelary.imputation.CafeImputer : the same, as a pipeline step.
    """
    import polars as pl

    from panelary.core.panel_frame import PanelFrame, as_panel
    from panelary.preprocessing import cafe_impute
    from panelary.preprocessing import impute as _impute

    _EXPR_METHODS = ("mean", "median", "fill", "ffill", "bfill", "interpolate")
    is_constant = isinstance(method, (int, float)) and not isinstance(method, bool)
    if not is_constant and method not in (*_EXPR_METHODS, "cafe"):
        raise _unknown_method("impute", str(method), (*_EXPR_METHODS, "cafe"))

    panel = as_panel(data, entity, time)
    ent, tim = panel.entity_col, panel.time_col
    original = panel.columns

    # `preprocessing.impute` reads the entity/time keys off the first two
    # columns, so present them that way and restore the caller's order after.
    front = [ent, tim]
    lf = panel.lazy().select(front + [c for c in original if c not in front])

    if method == "cafe":
        out = cafe_impute(columns=_as_list(columns), **kwargs)(lf)
    else:
        if columns is not None:
            raise ValueError(
                f"pn.impute: `columns` is only supported by method='cafe'; "
                f"method={method!r} fills every numeric feature column."
            )
        if kwargs:
            raise ValueError(
                f"pn.impute: method={method!r} takes no extra options, got "
                f"{sorted(kwargs)}."
            )
        out = _impute(method, allow_leaky=allow_leaky)(lf)

    produced = out.collect_schema().names()
    out = out.select(original + [c for c in produced if c not in original])

    if isinstance(data, PanelFrame):
        return PanelFrame(out, entity=ent, time=tim, validate=False)
    if isinstance(data, pl.DataFrame):
        return out.collect()
    return out


# --------------------------------------------------------------------------- #
# 2. select
# --------------------------------------------------------------------------- #
@_subpackage_verb("panelary.select")
def select(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "mrmr",
    columns: str | Sequence[str] | None = None,
    target: str | None = None,
    k: int = 10,
    threshold: float = 0.95,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> list[str]:
    """Choose an informative, non-redundant subset of the feature columns.

    Thin facade over :mod:`panelary.select`: :func:`~panelary.select.mrmr`
    (supervised) and :func:`~panelary.select.pfa` /
    :func:`~panelary.select.variance` / :func:`~panelary.select.correlation`
    (unsupervised). All four return names only -- they never mutate ``data``.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel holding the candidate features (and ``target``).
    method : {"mrmr", "pfa", "variance", "correlation"}, default "mrmr"
        * ``"mrmr"`` -- minimum-Redundancy-Maximum-Relevance; needs ``target``.
        * ``"pfa"`` -- Principal Feature Analysis: cluster the PCA loading
          vectors, keep one representative per cluster. Needs scikit-learn.
        * ``"variance"`` -- keep the ``k`` highest-variance features.
        * ``"correlation"`` -- prune features correlated above ``threshold``.
    columns : str or sequence of str, optional
        Restrict the candidate pool (forwarded as ``features=``). Defaults to
        every numeric feature column that is not the target.
    target : str, optional
        Target column name. Required by ``method="mrmr"``, ignored by the
        unsupervised methods.
    k : int, default 10
        Number of features to keep. Used by ``"mrmr"``, ``"pfa"``, ``"variance"``.
    threshold : float, default 0.95
        Absolute-correlation cut-off; used by ``"correlation"`` only.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to the underlying function (e.g. ``n_components``,
        ``variance_threshold``, ``random_state`` for ``"pfa"``).

    Returns
    -------
    list of str
        The selected column names. For ``"mrmr"`` they are in selection order
        (most informative first).

    Raises
    ------
    ValueError
        If ``method`` is unknown, or ``method="mrmr"`` without a ``target``.
    ImportError
        If ``method="pfa"`` and scikit-learn is missing
        (``pip install 'panelary[ml]'``).

    Leak-safety
    -----------
    Every statistic is computed **only from the rows you pass in** -- there is no
    hidden global fit, and no method looks at a future row. That makes the call
    correct inside a fold; it does not make it correct to call once on the full
    sample and reuse the answer across folds. Under cross-validation, call
    ``pn.select`` on each training fold (or use
    :class:`~panelary.select.MRMRSelector` / :class:`~panelary.select.PFASelector`
    as a pipeline step, which refits per fold by construction).

    Examples
    --------
    >>> import polars as pl
    >>> import panelary as pn
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A"] * 6,
    ...         "date": list(range(6)),
    ...         "signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ...         "noise": [0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
    ...         "ret": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    ...     }
    ... )
    >>> pn.select(df, method="variance", k=1)
    ['ret']
    >>> pn.select(df, method="mrmr", target="ret", k=1)
    ['signal']

    See Also
    --------
    panelary.select.mda : permutation importance through a purged splitter.
    panelary.select.mdi : impurity importance from a fitted ensemble.
    """
    from panelary.select import correlation, mrmr, pfa, variance

    common: dict[str, Any] = {
        "features": _as_list(columns),
        "entity": entity,
        "time": time,
    }

    if method == "mrmr":
        if target is None:
            raise ValueError(
                "pn.select: method='mrmr' is supervised and needs a target "
                "column, e.g. pn.select(df, target='fwd_ret', k=10)."
            )
        return mrmr(data, target, k, **common, **kwargs)
    if method == "pfa":
        return pfa(data, k, **common, **kwargs)
    if method == "variance":
        return variance(data, k, **common, **kwargs)
    if method == "correlation":
        return correlation(data, threshold, **common, **kwargs)
    raise _unknown_method("select", method, ("mrmr", "pfa", "variance", "correlation"))


# --------------------------------------------------------------------------- #
# 3. features
# --------------------------------------------------------------------------- #
def features(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str | Sequence[str] = "all",
    columns: str | Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> pl.DataFrame | pl.LazyFrame | PanelFrame:
    """Compute the registered ``ts`` scalar features, one row per entity.

    Thin facade over :func:`panelary.feature_extractors.extract_features`: a
    tsfresh-style bulk extractor that runs a single lazy
    ``group_by(entity).agg(...)``.

    Two ``method`` values instead **lift** each value column into a strictly
    trailing basis, one output row per input row (see :mod:`panelary.shape`):
    ``"delay"`` (:class:`~panelary.shape.Delay`, a delay / Hankel embedding --
    ``lags=``, ``dilation=``) and ``"spectral"``
    (:class:`~panelary.shape.Spectral` -- ``window=``, ``k=``, ``mode=``,
    default ``mode="band"``).

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    method : str or sequence of str, default "all"
        Which registered ``ts`` features to compute. ``"all"`` is every
        registered scalar aggregation; otherwise pass an explicit list of
        feature names (e.g. ``["mean_abs_change", "absolute_energy"]``).
        Forwarded as ``features=``. ``"delay"`` / ``"spectral"`` select the
        trailing shape lifts described above instead.
    columns : str or sequence of str, optional
        Value column(s) to featurise (forwarded as ``column=``). Defaults to
        every numeric column that is neither the entity nor the time key. With
        one value column the outputs are named ``<feature>``; with several they
        are namespaced ``<column>__<feature>``.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to :func:`~panelary.feature_extractors.extract_features`,
        or to the :class:`~panelary.shape.Delay` /
        :class:`~panelary.shape.Spectral` constructor.

    Returns
    -------
    polars.DataFrame | polars.LazyFrame | PanelFrame
        One row per entity. Lazy in, lazy out; a ``PanelFrame`` yields a
        ``LazyFrame`` (it wraps one). For ``"delay"`` / ``"spectral"``: a
        :class:`~panelary.core.panel_frame.PanelFrame` with the keys plus the
        lifted columns, one row per input row.

    Raises
    ------
    ValueError
        If a requested feature is not a registered ``ts`` scalar aggregation,
        or if no value column can be resolved.

    Leak-safety
    -----------
    ``panel_safe`` by construction: every feature is a pure per-entity
    aggregation, so nothing crosses an entity boundary. It is **not**
    point-in-time -- each value summarises the *whole* window you pass, which is
    the point of a per-entity feature table but means the window must be a
    training window. For a per-row, strictly causal feature use the ``ts`` /
    ``panel`` expression namespaces (``pl.col("px").ts.<feature>()`` under
    ``.over(entity)``), not this verb.

    ``method="delay"`` and ``method="spectral"`` are the exception: they are
    point-in-time by construction -- row ``t`` reads only its own entity's
    trailing ``window`` (or ``(lags - 1) * dilation + 1``) observations, nothing
    is fitted, and the output is bit-identical under truncation of the panel.

    Examples
    --------
    >>> import polars as pl
    >>> import panelary as pn
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "A", "A", "B", "B", "B"],
    ...         "date": [1, 2, 3, 1, 2, 3],
    ...         "px": [1.0, 2.0, 3.0, 10.0, 8.0, 6.0],
    ...     }
    ... )
    >>> out = pn.features(df, method=["absolute_energy"])
    >>> sorted(out.columns)
    ['absolute_energy', 'ticker']

    See Also
    --------
    panelary.catch22 : the 22 canonical clean-room catch22 features.
    panelary.registry.registry : the machine-readable catalogue of operators.
    """
    from panelary.core.panel_frame import PanelFrame
    from panelary.feature_extractors import extract_features

    if isinstance(method, str) and method in ("delay", "spectral"):
        # The trailing LIFT direction of the shape algebra (panelary.shape).
        import panelary.shape as shape

        lift: Any
        if method == "delay":
            lift = shape.Delay(columns=_as_list(columns), **kwargs)
        else:
            kwargs.setdefault("mode", "band")
            lift = shape.Spectral(columns=_as_list(columns), **kwargs)
        return lift.fit_transform(data, entity=entity, time=time)

    if isinstance(data, PanelFrame):
        frame: pl.DataFrame | pl.LazyFrame = data.lazy()
        entity, time = data.entity_col, data.time_col
    else:
        frame = data

    return extract_features(
        frame,
        entity=entity,
        time=time,
        features=method,
        column=columns,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 4. bubbles
# --------------------------------------------------------------------------- #
def bubbles(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "bsadf",
    columns: str | Sequence[str] | None = None,
    min_window: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Score every row for an explosive (bubble) regime, per entity.

    Thin facade over :mod:`panelary.detect`: it runs
    :func:`~panelary.detect.bsadf_sequence` or
    :func:`~panelary.detect.hb_cusum` on each entity's series in time order and
    appends the resulting per-row statistic to the panel.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel. Each entity's series is scored independently.
    method : {"bsadf", "hb_cusum"}, default "bsadf"
        * ``"bsadf"`` -- backward sup-ADF (Phillips-Shi-Yu): at each endpoint,
          the supremum of the ADF t-statistic over all admissible start dates.
          Emits ``<column>__bsadf``.
        * ``"hb_cusum"`` -- the Homm-Breitung CUSUM monitor with its analytic
          Chu-Stinchcombe-White boundary. Emits ``<column>__hb_cusum`` and
          ``<column>__hb_cusum_boundary``; the regime is flagged where the
          statistic exceeds the boundary.
    columns : str or sequence of str, optional
        Level series to score (prices, not returns). Defaults to every numeric
        feature column.
    min_window : int
        Absolute number of observations, **required** -- there is deliberately
        no default. For ``"bsadf"`` it is the shortest backward window
        (``r_0 * T`` in the PSY notation, forwarded as ``min_window``); for
        ``"hb_cusum"`` it is the training sample of initial increments
        (forwarded as ``r0``). Both play the same role: the amount of history
        the detector needs before it will speak. See *Leak-safety*.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to the underlying detector (e.g. ``lag``, ``trend``, ``grid``
        for ``"bsadf"``; ``kappa`` for ``"hb_cusum"``).

    Returns
    -------
    polars.DataFrame
        The collected panel sorted by ``(entity, time)``, with the statistic
        column(s) appended. Eager, because the detectors are NumPy kernels.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if ``min_window`` is not given, or if no
        numeric column can be resolved.

    Leak-safety
    -----------
    Strictly point-in-time, and mechanically so: every statistic at ``t`` is a
    function of ``y[:t+1]`` only, so appending future data cannot change an
    already-published value (the prefix-invariance property asserted in
    ``tests/test_detect_leak_safety.py``). Entities never mix.

    This is exactly why ``min_window`` has no default. The conventional PSY rule
    ``floor(T(0.01 + 1.8/sqrt(T)))`` makes the window a function of how much
    data you happen to hold, which silently revises already-published dates as
    the sample grows -- a leak that no test catches unless you look for it. Pass
    an absolute number of observations, chosen once, and never a fraction of the
    sample.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import panelary as pn
    >>> rng = np.random.default_rng(0)
    >>> px = np.cumsum(rng.normal(size=80))
    >>> df = pl.DataFrame(
    ...     {"ticker": ["A"] * 80, "date": list(range(80)), "px": px}
    ... )
    >>> out = pn.bubbles(df, min_window=20)
    >>> "px__bsadf" in out.columns
    True

    See Also
    --------
    panelary.detect.bsadf_sequence : the single-series kernel.
    panelary.detect.breadth : cross-sectional aggregation of the statistic.
    panelary.detect.default_table : simulated (leak-free) critical values.
    """
    import numpy as np
    import polars as pl

    from panelary.core.panel_frame import as_panel
    from panelary.detect import bsadf_sequence, hb_cusum

    if method not in ("bsadf", "hb_cusum"):
        raise _unknown_method("bubbles", method, ("bsadf", "hb_cusum"))
    if min_window is None:
        raise ValueError(
            "pn.bubbles: `min_window` is required and has no default. It must be "
            "an absolute number of observations (e.g. min_window=50), never a "
            "fraction of the sample: a window that grows with the data revises "
            "statistics you have already published."
        )

    panel = as_panel(data, entity, time)
    ent, tim = panel.entity_col, panel.time_col
    cols = _resolve_columns(panel, columns)

    df = panel.collect().sort([ent, tim])
    out: dict[str, list[Any]] = {}
    for _key, sub in df.group_by(ent, maintain_order=True):
        for col in cols:
            y = sub[col].to_numpy().astype(np.float64)
            if method == "bsadf":
                pieces = {
                    f"{col}__bsadf": bsadf_sequence(y, min_window=min_window, **kwargs)
                }
            else:
                stat, boundary = hb_cusum(y, r0=min_window, **kwargs)
                pieces = {
                    f"{col}__hb_cusum": stat,
                    f"{col}__hb_cusum_boundary": boundary,
                }
            for name, values in pieces.items():
                out.setdefault(name, []).append(np.asarray(values, dtype=np.float64))

    return df.with_columns(
        [pl.Series(name, np.concatenate(chunks)) for name, chunks in out.items()]
    )


# --------------------------------------------------------------------------- #
# 5. cluster
# --------------------------------------------------------------------------- #
@_subpackage_verb("panelary.cluster")
def cluster(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "kshape",
    columns: str | Sequence[str] | None = None,
    n_clusters: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> PanelFrame:
    """Group entities, by time-series shape or by same-date feature vector.

    Thin facade over :mod:`panelary.cluster`: it builds the requested clusterer,
    calls ``fit_transform`` and returns the panel with the cluster features
    appended.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    method : {"kshape", "cross_sectional"}, default "kshape"
        * ``"kshape"`` -- :class:`~panelary.cluster.KShapeClusterer`: cluster
          each entity's whole *shape* and emit, per row, the Shape-Based
          Distance to each frozen centroid plus a hard label (``ksh_*``).
        * ``"cross_sectional"`` -- :class:`~panelary.cluster.CrossSectionalClusterer`:
          cluster entities by their same-date feature vectors with scikit-learn
          (``xcl_*``). Needs scikit-learn.
    columns : str or sequence of str, optional
        Feature column(s) to cluster on. Defaults to every numeric feature
        column. Forwarded as ``value=`` (k-Shape) or ``features=``
        (cross-sectional).
    n_clusters : int, optional
        Number of clusters. ``None`` means the underlying default: for k-Shape
        the :func:`~panelary.cluster.k_from_n_entities` heuristic on the
        training entity count, for the cross-sectional clusterer 8.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to the clusterer's constructor (e.g. ``window``, ``emit``,
        ``seed``, ``max_iter``, ``prefix``; ``algo``, ``scaler`` for the
        cross-sectional one).

    Returns
    -------
    PanelFrame
        The panel with the emitted cluster columns appended. Call ``.collect()``
        or ``.to_native()`` for a plain polars frame.

    Raises
    ------
    ValueError
        If ``method`` is unknown or no numeric column can be resolved.
    ImportError
        If ``method="cross_sectional"`` and scikit-learn is missing
        (``pip install 'panelary[ml]'``).

    Leak-safety
    -----------
    Both clusterers are ``panel_safe`` and ``leakage_safe`` *as estimators*:
    centroids are learned in ``fit`` and frozen, and k-Shape's default
    ``window="expanding"`` scores each row from that entity's history up to that
    row only.

    This verb, however, is a ``fit_transform`` on the rows you hand it -- correct
    on a training fold or a single exploratory frame, wrong if you run it once
    on the full sample and then cross-validate, because the centroids would have
    seen the test rows. Inside CV, construct the clusterer and use ``fit`` /
    ``transform`` explicitly, or put it in a
    :class:`~panelary.core.pipeline.Pipeline` that refits per fold.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import panelary as pn
    >>> rng = np.random.default_rng(0)
    >>> n = 30
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A"] * n + ["B"] * n,
    ...         "date": list(range(n)) * 2,
    ...         "px": np.concatenate([rng.normal(size=n), rng.normal(size=n) + 5]),
    ...     }
    ... )
    >>> out = pn.cluster(df, n_clusters=2, columns="px")
    >>> "ksh_label" in out.columns
    True

    See Also
    --------
    panelary.cluster.KShapeClusterer : the shape clusterer.
    panelary.cluster.CrossSectionalClusterer : the per-date clusterer.
    """
    from panelary.cluster import CrossSectionalClusterer, KShapeClusterer
    from panelary.core.panel_frame import as_panel

    if method not in ("kshape", "cross_sectional"):
        raise _unknown_method("cluster", method, ("kshape", "cross_sectional"))

    panel = as_panel(data, entity, time)
    cols = _resolve_columns(panel, columns)
    keys: dict[str, Any] = {"entity": panel.entity_col, "time": panel.time_col}

    estimator: KShapeClusterer | CrossSectionalClusterer
    if method == "kshape":
        estimator = KShapeClusterer(cols, n_clusters=n_clusters, **keys, **kwargs)
    else:
        # `CrossSectionalClusterer.n_clusters` is a plain int with its own
        # default; don't override it with None.
        count: dict[str, Any] = {} if n_clusters is None else {"n_clusters": n_clusters}
        estimator = CrossSectionalClusterer(cols, **count, **keys, **kwargs)
    return estimator.fit_transform(panel)


# --------------------------------------------------------------------------- #
# 6. regression
# --------------------------------------------------------------------------- #
def regression(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "hdfe",
    y: str,
    x: Sequence[str],
    absorb: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> HDFEResult | FamaMacBethResult | PanelFitResult:
    """Estimate a panel regression of ``y`` on ``x``, with honest standard errors.

    Thin facade over :mod:`panelary.econ`. These are *association* estimators:
    they answer "what is the conditional relationship between ``y`` and ``x`` in
    this panel, and how uncertain is it". For the effect of one treatment under
    confounding, use :func:`causal` instead.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel, one row per ``(entity, time)``.
    method : str, default "hdfe"
        * ``"hdfe"`` -- :func:`~panelary.econ.hdfe`, N-way fixed effects by
          alternating projections, with classical / robust / clustered /
          Driscoll-Kraay standard errors. Requires ``absorb``.
        * ``"fama_macbeth"`` -- :func:`~panelary.econ.fama_macbeth`, per-date
          cross-sectional regressions with Newey-West errors on the lambdas.
        * ``"mean_group"`` -- :func:`~panelary.econ.mean_group` (Pesaran-Smith).
        * ``"cce_mg"`` / ``"cce_pooled"`` -- :func:`~panelary.econ.cce_mg` /
          :func:`~panelary.econ.cce_pooled` (Pesaran CCE, robust to unobserved
          common factors).
        * ``"pmg"`` -- :func:`~panelary.econ.pmg`, Pooled Mean Group ARDL.
    y : str
        Dependent variable. For a predictive regression this is the *forward*
        return; lagging the regressors is the caller's responsibility.
    x : sequence of str
        Regressors.
    absorb : sequence of str, optional
        Fixed effects to absorb. Required by (and only used by) ``"hdfe"``,
        e.g. ``absorb=["ticker", "date"]``.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to the estimator (e.g. ``cluster``, ``vcov``, ``lags`` for
        ``"hdfe"``; ``winsorize_limit``, ``intercept`` for ``"fama_macbeth"``;
        ``min_obs`` for the heterogeneous-panel estimators).

    Returns
    -------
    HDFEResult | FamaMacBethResult | PanelFitResult
        A result object carrying at least ``params`` / coefficients, standard
        errors and t-statistics. The exact type depends on ``method``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or ``method="hdfe"`` without ``absorb``.

    Leak-safety
    -----------
    These are **in-sample** estimators: every one fits on exactly the rows you
    pass, and reports inference for that sample. That is the correct semantics
    for inference, and it is *not* a licence to fit on the full sample and then
    predict a held-out fold -- the coefficients would have seen it.

    Two internal properties are worth knowing. ``"fama_macbeth"`` winsorises and
    standardises **per date**, never globally, so a cross-sectional outlier in
    2019 cannot rescale 2015. The CCE estimators use same-date cross-sectional
    averages as factor proxies -- contemporaneous, not forward-looking. Nothing
    here reads a future row.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import panelary as pn
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> size = rng.normal(size=n)
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "B", "C", "D"] * (n // 4),
    ...         "date": np.repeat(np.arange(n // 4), 4),
    ...         "size": size,
    ...         "ret": 0.5 * size + rng.normal(size=n) * 0.1,
    ...     }
    ... )
    >>> res = pn.regression(df, y="ret", x=["size"], absorb=["ticker"])
    >>> round(float(res.params[0]), 1)
    0.5

    See Also
    --------
    panelary.econ.ivx : predictability inference under a near-unit-root regressor.
    panelary.econ.connectedness : Diebold-Yilmaz spillover networks.
    """
    from panelary.econ import (
        cce_mg,
        cce_pooled,
        fama_macbeth,
        hdfe,
        mean_group,
        pmg,
    )

    common: dict[str, Any] = {
        "y": y,
        "x": list(x),
        "entity": entity,
        "time": time,
    }
    if method == "hdfe":
        if absorb is None:
            raise ValueError(
                "pn.regression: method='hdfe' needs `absorb=` -- the columns "
                "whose fixed effects to absorb, e.g. absorb=['ticker', 'date']. "
                "For a regression without fixed effects use "
                "method='mean_group' or method='fama_macbeth'."
            )
        return hdfe(data, absorb=list(absorb), **common, **kwargs)
    if absorb is not None:
        raise ValueError(
            f"pn.regression: `absorb` is only used by method='hdfe', got "
            f"method={method!r}."
        )
    if method == "fama_macbeth":
        return fama_macbeth(data, **common, **kwargs)
    if method == "mean_group":
        return mean_group(data, **common, **kwargs)
    if method == "cce_mg":
        return cce_mg(data, **common, **kwargs)
    if method == "cce_pooled":
        return cce_pooled(data, **common, **kwargs)
    if method == "pmg":
        return pmg(data, **common, **kwargs)
    raise _unknown_method(
        "regression",
        method,
        ("hdfe", "fama_macbeth", "mean_group", "cce_mg", "cce_pooled", "pmg"),
    )


# --------------------------------------------------------------------------- #
# 7. causal
# --------------------------------------------------------------------------- #
def causal(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "dml",
    y: str,
    treatment: str,
    controls: Sequence[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> DMLResult | PDSResult:
    """Estimate the effect of one treatment on ``y``, controlling for the rest.

    Thin facade over the causal half of :mod:`panelary.econ`. Where
    :func:`regression` estimates a whole coefficient vector in one shot, this
    verb targets a **single** coefficient -- the treatment effect ``theta`` -- and
    uses machine learning for the nuisance functions in a way that keeps the
    confidence interval valid.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    method : {"dml", "pds"}, default "dml"
        * ``"dml"`` -- :func:`~panelary.econ.dml_partial_linear`: cross-fitted
          double machine learning for the partially linear model, with folds
          taken from Panelary's purged/embargoed splitters.
        * ``"pds"`` -- :func:`~panelary.econ.post_double_selection`:
          Belloni-Chernozhukov-Hansen post-double-selection LASSO with the
          rigorous plug-in penalty.
    y : str
        Outcome.
    treatment : str
        The treatment / feature of interest (forwarded as ``d``). Exactly one.
    controls : sequence of str, optional
        Confounders to adjust for. Defaults to every other numeric feature
        column.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to the estimator: ``n_folds``, ``horizon``, ``embargo``,
        ``learner``, ``ridge``, ``alpha`` for ``"dml"``; ``c``, ``gamma``,
        ``alpha`` for ``"pds"``.

    Returns
    -------
    DMLResult | PDSResult
        The point estimate ``theta``, its standard error and confidence
        interval, plus method-specific diagnostics (the selected control set for
        ``"pds"``, the fold structure for ``"dml"``).

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    Leak-safety
    -----------
    ``"dml"`` is the leak-aware one, and deliberately so: the nuisance functions
    are fit on ``n_folds - 1`` folds and applied to the held-out one, with the
    folds **purged and embargoed** by Panelary's splitters. If your outcome has
    a label horizon (e.g. a 5-day forward return), pass ``horizon=5`` so the
    purge covers it -- otherwise the training folds overlap the test fold's
    label window and the cross-fitting guarantee is void.

    ``"pds"`` selects controls and estimates on the same rows (that is the
    method), so it inherits the usual in-sample caveat: valid inference for the
    sample you passed, not a licence to reuse the fit on a later fold.

    Neither method makes a causal claim on your behalf: unconfoundedness given
    ``controls`` is an assumption you are asserting, not something either
    estimator can test.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import panelary as pn
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> z = rng.normal(size=n)
    >>> d = z + rng.normal(size=n) * 0.5
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "B"] * (n // 2),
    ...         "date": np.repeat(np.arange(n // 2), 2),
    ...         "flow": d,
    ...         "z": z,
    ...         "ret": 1.0 * d + z + rng.normal(size=n) * 0.1,
    ...     }
    ... )
    >>> res = pn.causal(df, y="ret", treatment="flow", controls=["z"])
    >>> round(float(res.theta), 1)
    1.0

    See Also
    --------
    panelary.econ.DoubleMLTransformer : the same estimator as a pipeline step.
    panelary.core.model_selection.PurgedKFold : the splitter DML cross-fits with.
    """
    from panelary.econ import dml_partial_linear, post_double_selection

    common: dict[str, Any] = {
        "y": y,
        "d": treatment,
        "controls": list(controls) if controls is not None else None,
        "entity": entity,
        "time": time,
    }
    if method == "dml":
        return dml_partial_linear(data, **common, **kwargs)
    if method == "pds":
        return post_double_selection(data, **common, **kwargs)
    raise _unknown_method("causal", method, ("dml", "pds"))


# --------------------------------------------------------------------------- #
# 8. reduce
# --------------------------------------------------------------------------- #
@_subpackage_verb("panelary.reduce")
def reduce(
    data: PanelFrame | pl.DataFrame | pl.LazyFrame,
    *,
    method: str = "pca",
    columns: str | Sequence[str] | None = None,
    n_components: int | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> PanelFrame:
    """Compress the feature columns into a few components or latent factors.

    Thin facade over :mod:`panelary.reduce`. Two families are reachable:
    the **row-wise reducers** (projection onto components, via
    :func:`~panelary.reduce.reduce_features`) and the **factor extractors**,
    which emit new ``factor_1 .. factor_r`` columns and can resolve the factor
    count themselves.

    Parameters
    ----------
    data : PanelFrame | polars.DataFrame | polars.LazyFrame
        Long-format panel.
    method : str, default "pca"
        Reducers: ``"pca"``, ``"svd"`` / ``"truncated_svd"``,
        ``"factor_analysis"``, ``"random_projection"`` /
        ``"gaussian_random_projection"``, ``"kernel_pca"``, ``"nmf"``,
        ``"umap"`` (needs ``umap-learn``).

        Factor extractors: ``"pca_factors"`` (baseline), ``"hfa"``
        (higher-order cumulant factors -- weak or masked **non-Gaussian**
        factors, ``order=3|4``), ``"ica"`` (maximally independent components,
        needs scikit-learn), ``"robust_pca"`` (heavy tails / contaminated rows).

        Shape-algebra compressors (:mod:`panelary.shape`, pure numpy, no
        scikit-learn): ``"rsvd"`` (randomized PCA), ``"sparse_rp"`` / ``"srht"``
        (seeded, data-independent projections), ``"id"`` (``ColumnSubset``:
        ``n_components`` of your *real* columns, by interpolative
        decomposition), ``"cur"`` (the same columns, plus CUR factors); and
        along time, ``"paa"`` (piecewise aggregate approximation, ``n_components``
        segments) and ``"spectral"`` (``n_components`` lowest DFT magnitudes),
        both of which need ``window=``.
    columns : str or sequence of str, optional
        Feature columns to reduce. Defaults to every numeric feature column.
        Forwarded as ``columns=`` (reducers) or ``features=`` (extractors).
    n_components : int, optional
        Number of components to keep. ``None`` lets the method choose: the
        reducers use their own default, the factor extractors resolve the count
        on the training rows (Bai-Ng ``IC_p2``, or the eigenvalue ratio of the
        cumulant spectrum for HFA). Forwarded as ``n_factors=`` to the
        extractors.
    entity, time : str, optional
        Panel keys; used only when ``data`` is a bare polars frame.
    **kwargs
        Forwarded to the underlying estimator's constructor (e.g.
        ``keep_features``, ``prefix``; ``order``, ``block_rows`` for ``"hfa"``).

    Returns
    -------
    PanelFrame
        The keys plus the component / factor columns (or the whole panel with
        them appended, depending on the estimator's ``keep`` / ``keep_features``
        option).

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    ImportError
        If the chosen method needs an extra that is missing (scikit-learn for
        ``"ica"`` and the sklearn-backed reducers, ``umap-learn`` for
        ``"umap"``).

    Leak-safety
    -----------
    Every reducer here is ``leakage_safe`` *as an estimator*: the scaler, the
    rotation and any automatic component count are learned in ``fit`` from the
    training rows only, and the components are sign-fixed deterministically so
    signs do not flip between refits. (This is the corrected port of an upstream
    that fit on the full sample and back-filled the future.)

    The verb itself is a ``fit_transform`` on the rows you pass -- right for a
    training fold or a one-off exploratory frame, wrong on a full sample that is
    later cross-validated, since the rotation would have seen the test rows.
    Inside CV, build the estimator (e.g.
    :class:`~panelary.reduce.PanelPCA`, :class:`~panelary.reduce.HFAFactors`)
    and let the :class:`~panelary.core.pipeline.Pipeline` refit it per fold.

    The time-axis methods (``"paa"``, ``"spectral"``) take ``window=`` and are
    strictly trailing: row ``t`` reads only its own entity's last ``window``
    observations and nothing is fitted, so they are point-in-time however they
    are called. Passing ``flavour="whole_series"`` moves the call into the
    fit-on-what-you-pass regime -- one row per entity, summarising its whole
    history, ``leakage_safe=False`` -- for describing a fixed sample only.
    ``"sparse_rp"`` / ``"srht"`` learn nothing from the data at all (the
    projection is a function of the width and ``seed=``).

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import panelary as pn
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> f = rng.normal(size=n)
    >>> df = pl.DataFrame(
    ...     {
    ...         "ticker": ["A", "B"] * (n // 2),
    ...         "date": np.repeat(np.arange(n // 2), 2),
    ...         "a": f + rng.normal(size=n) * 0.1,
    ...         "b": 2 * f + rng.normal(size=n) * 0.1,
    ...     }
    ... )
    >>> out = pn.reduce(df, n_components=1)
    >>> out.collect().height
    60

    See Also
    --------
    panelary.reduce.reduce_features : the reducer entry point this wraps.
    panelary.reduce.n_factors : the Bai-Ng / eigenvalue-ratio count selectors.
    panelary.reduce.CrossSectionalPCA : per-date cross-sectional reduction.
    """
    from panelary.reduce import (
        HFAFactors,
        ICAFactors,
        PCAFactors,
        RobustPCAFactors,
        reduce_features,
    )

    cols = _as_list(columns)

    extractors = {
        "pca_factors": PCAFactors,
        "hfa": HFAFactors,
        "ica": ICAFactors,
        "robust_pca": RobustPCAFactors,
    }
    if method in extractors:
        extractor = extractors[method](
            n_components,
            features=cols,
            entity=entity,
            time=time,
            **kwargs,
        )
        return extractor.fit_transform(data)

    # Shape-algebra compressors: method -> (class name, the width parameter
    # `n_components` maps onto). Imported lazily; `panelary.shape` loads no
    # transform module until one is used.
    shape_methods = {
        "rsvd": ("RandomizedPCA", "n_components"),
        "sparse_rp": ("SparseRandomProjection", "n_components"),
        "srht": ("SRHT", "n_components"),
        "id": ("ColumnSubset", "k"),
        "cur": ("CUR", "k"),
        "paa": ("PAA", "segments"),
        "spectral": ("Spectral", "k"),
    }
    if method in shape_methods:
        import panelary.shape as shape

        cls_name, width_param = shape_methods[method]
        if n_components is not None:
            kwargs[width_param] = n_components
        transform = getattr(shape, cls_name)(columns=cols, **kwargs)
        return transform.fit_transform(data, entity=entity, time=time)

    return reduce_features(
        data,
        method=method,
        n_components=n_components,
        columns=cols,
        entity=entity,
        time=time,
        **kwargs,
    )
