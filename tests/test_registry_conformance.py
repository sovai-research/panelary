"""Conformance suite for the operator registry: the safety claims, checked.

Every :class:`~panelary.registry.FeatureSpec` in the registry declares
``leakage_safe=True``. Until this file existed nothing verified a single one of
those declarations -- ``assert_no_lookahead`` appeared 28 times in ``tests/``,
every one of them inside a hand-written leakage suite, never once over the
catalogue. This suite parametrises over ``registry.all()`` so a newly registered
operator is covered the day it lands, not the day someone remembers it.

What is checked, and why each check is the shape it is
-----------------------------------------------------
1. **Every spec declares a scope.** ``leakage_safe`` alone is a boolean answer
   to a question with no boolean answer (*safe evaluated how?*), which is why
   :data:`~panelary.registry.VALID_SAFE_SCOPES` exists. ``"unspecified"`` is
   tolerated by the dataclass so third-party specs keep working; Panelary's own
   catalogue is held to the stricter standard here.
2. **Structural consistency**, proved from metadata with no data at all: a spec
   whose ``output_shape`` is ``"scalar"`` cannot be ``"rowwise"``. Reducing a
   series to one number and broadcasting it onto every row makes the value at
   *every* ``t`` a function of the entity's whole series, future included.
3. **``"rowwise"`` specs are verified per row with BOTH instruments.**
   :func:`~panelary.testing.assert_no_lookahead` perturbs future *values* and so
   catches value-dependence; :func:`~panelary.testing.assert_prefix_invariant`
   truncates the panel and so catches *length*-dependence. Neither subsumes the
   other, and a spec is only verified when it survives both.
4. **``"window"`` specs get the structural claim, plus evidence.** The claim
   itself ("safe only as the summary of an already-delimited window") is proved
   by shape, not by data -- but asserting *nothing* behavioural would make the
   label a rubber stamp, so each window spec is additionally required to be
   demonstrably NOT prefix-invariant when evaluated per row across the panel.
5. **The provenance/licence audit still passes**, via
   :meth:`~panelary.registry.FeatureRegistry.audit` rather than a second
   hand-rolled copy of it.

Why a passing ``assert_no_lookahead`` is never treated as evidence of safety
---------------------------------------------------------------------------
The perturbation verifier is **unsound for scalar aggregates**: perturbing the
future changes values but not the row count, so a statistic that depends only on
``len(x)`` sails through it (``pl.col("x").count()`` is the canonical case), and
a statistic that happens to be constant on the probe sample passes *vacuously*.
Measured on the probe panel below, ``ts.has_duplicate`` and
``ts.symmetry_looking`` pass both verifiers despite being series-to-scalar
reductions -- they are simply degenerate on this data. So the verifier is used
here only to catch failures, never to certify a spec as row-safe; the row-safe
side of the ledger is carried by the structural rule in (2).

The probe panel is built to make vacuous passes unlikely: six entities of four
different lengths, repeated values, an explicit exact duplicate, an exact tie
with the running maximum, sign changes, an interior null, and a second
correlated column (so frame-shaped operators have something to neutralise or
orthogonalise against). Seeded RNG, float64 throughout, no ``rolling_map``.

A note on ``.over``
-------------------
``.panel`` and ``.ts`` are within-entity and take ``.over(entity)``; ``.xs`` is
cross-sectional and takes ``.over(time)``. Getting that backwards does not raise
-- it silently produces a nonsense verdict -- so the mapping is table-driven, in
:data:`_OVER_KEY` for the expression namespaces and in :data:`_EVOLVE_OVER` for
the ``evolve`` vocabulary (whose partition scope is named by ``Op.kind``), rather
than inlined per test.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import panelary  # noqa: F401  -- registers the ts / xs / panel / factor specs
from panelary.core.panel_frame import PanelFrame
from panelary.factor import forward_return as _factor_forward_return
from panelary.factor import orthogonalize as _factor_orthogonalize
from panelary.registry import (
    VALID_SAFE_SCOPES,
    FeatureRegistry,
    FeatureSpec,
    register_feature,
    registry,
)
from panelary.testing import assert_no_lookahead, assert_prefix_invariant

# The `evolve` vocabulary registers 52 further specs, but only when
# `panelary.evolve` is imported -- which `tests/test_evolve.py` does at module
# scope, i.e. during collection, i.e. before this module is imported in a
# full-suite run but not when this file is run alone. Importing it here pins the
# parametrisation to the same set either way; a conformance suite whose coverage
# depends on module import order is not a conformance suite. A failure to import
# is tolerated (it is an optional subpackage) rather than skipping this file,
# which would take the other 56 specs down with it.
try:
    import panelary.evolve as _evolve
except Exception:  # noqa: BLE001 - an import failure just means fewer specs
    _evolve = None  # type: ignore[assignment]

# The shape algebra registers its catalogue the first time a public name is read
# from `panelary.shape` -- its package initialiser is lazy, so `import panelary`
# loads no transform module (plan `shape-build-contract.md` section 9, item 7).
# Load it here for the same reason as `evolve` above: the parametrisation must
# not depend on which test module happened to touch `panelary.shape` first.
try:
    import panelary.shape as _shape

    _shape.PAA  # noqa: B018 - the first access registers every shape FeatureSpec
except Exception:  # noqa: BLE001 - an import failure just means fewer specs
    _shape = None  # type: ignore[assignment]

# `panelary.covariance` is a lazy submodule with a lazy initialiser, exactly like
# `shape`: its frame operations register the first time a public name is read.
try:
    import panelary.covariance as _cov

    _cov.avg_correlation  # noqa: B018 - the first access registers the catalogue
except Exception:  # noqa: BLE001 - an import failure just means fewer specs
    _cov = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Probe panel
# --------------------------------------------------------------------------- #
ENTITY, TIME, VALUE, AUX = "entity", "time", "x", "z"

#: Four distinct lengths, so "this entity ran out of history" and "the panel ran
#: out of history" are different events and a length-dependent warm-up shows up.
_ENTITY_LENGTHS: dict[str, int] = {"A": 24, "B": 24, "C": 24, "D": 22, "E": 18, "F": 12}

_SEED = 20240917


def _probe_frame() -> pl.DataFrame:
    """Build the deterministic, deliberately non-degenerate probe panel.

    Rounding to one decimal manufactures ties and repeated values, which is what
    keeps ``has_duplicate`` / ``percent_reoccurring_*`` / ``ratio_n_unique_*``
    from being constant -- and therefore vacuous -- across the whole panel.
    """
    rng = np.random.default_rng(_SEED)
    entities: list[str] = []
    times: list[int] = []
    values: list[float] = []
    aux: list[float] = []
    for name, n in _ENTITY_LENGTHS.items():
        series = np.round(np.cumsum(rng.standard_normal(n)), 1)
        series[2] = series[1]  # an exact duplicate
        series[5] = -series[5]  # a sign change
        series[n - 3] = series[: n - 3].max()  # an exact tie with the running max
        entities += [name] * n
        times += list(range(n))
        values += [float(v) for v in series]
        aux += [float(v) for v in np.round(rng.standard_normal(n) * 2.0, 1)]

    return (
        pl.DataFrame(
            {ENTITY: entities, TIME: times, VALUE: values, AUX: aux},
            schema={
                ENTITY: pl.Utf8,
                TIME: pl.Int64,
                VALUE: pl.Float64,
                AUX: pl.Float64,
            },
        )
        .with_columns(
            # One interior null, mid-entity: enough to exercise the verifiers'
            # null handling without hollowing the panel out.
            pl.when((pl.col(ENTITY) == "C") & (pl.col(TIME) == 4))
            .then(None)
            .otherwise(pl.col(VALUE))
            .alias(VALUE)
        )
        .sort([TIME, ENTITY])
    )


_FRAME = _probe_frame()
_PANEL = PanelFrame(_FRAME, entity=ENTITY, time=TIME)


@pytest.fixture(scope="module")
def panel() -> PanelFrame:
    """The probe panel, built once per module."""
    return _PANEL


# --------------------------------------------------------------------------- #
# Building a runnable op out of a spec
# --------------------------------------------------------------------------- #
#: Namespaces that really are Polars expression namespaces, i.e. for which
#: ``pl.col(c).<ns>.<name>(...)`` resolves.
_EXPR_NAMESPACES = frozenset({"ts", "xs", "panel"})

#: Which key an operator must be grouped by. ``.xs`` is cross-sectional: it
#: partitions by *date*, not by entity.
_OVER_KEY: dict[str, str] = {"ts": ENTITY, "panel": ENTITY, "xs": TIME}

#: Registry names that differ from the namespace method name. The registry keys
#: specs by bare ``name``, so the cross-sectional z-score is registered as
#: ``cs_zscore`` while the method stays ``.xs.zscore``.
_METHOD_ALIASES: dict[str, str] = {
    "cs_zscore": "zscore",
    # Cross-sectional distribution summaries (plan covariance-and-market-state
    # section 4.4): registered with an ``xs_`` prefix, called without it.
    "xs_dispersion": "dispersion",
    "xs_tail_index": "tail_index",
    "xs_up_share": "up_share",
    "xs_entropy": "entropy",
}

#: Arguments synthesised for operators with required parameters. Anything not
#: listed here and not defaulted is skipped with a reason rather than guessed.
_SYNTHESISED_ARGS: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
    "frac_diff": ((0.4,), {}),
    "zscore": ((), {"window": 5}),
    "rs_vol": ((), {"window": 5}),
    "winsorize": ((0.1,), {}),
    "quantile_bin": ((), {"q": 3}),
    "neutralize": ((AUX,), {}),
    "rolling_xi": ((AUX,), {"window": 5}),
    "rolling_dcor": ((AUX,), {"window": 5}),
    "rolling_tail_dep": ((AUX,), {"window": 10, "q": 0.2}),
    "rolling_gcmi": ((AUX,), {"window": 6}),
    # Six names per date: q=0.5 gives k=3 exceedances, so the Hill value is
    # non-null on the probe panel instead of vacuously null (the default
    # q=0.05 needs 200 names per date).
    "xs_tail_index": ((), {"q": 0.5, "min_exceedances": 2}),
}


def _orthogonalize_op(frame: Any) -> Any:
    """``factor.orthogonalize`` bound to the probe panel's columns."""
    return _factor_orthogonalize(frame, [VALUE, AUX], time=TIME)


def _forward_return_op(frame: Any) -> Any:
    """``factor.forward_return`` bound to the probe panel's columns."""
    return _factor_forward_return(frame, entity=ENTITY, time=TIME, ret=VALUE, horizon=1)


#: Frame-shaped operators (``namespace="factor"``) that preserve the
#: ``(entity, time)`` keys, and so can still be driven through the verifiers as
#: ``frame -> frame`` callables.
_FRAME_OPS: dict[str, Callable[[Any], Any]] = {
    "orthogonalize": _orthogonalize_op,
    "forward_return": _forward_return_op,
}


def _shape_op(factory: Callable[[], Any]) -> Callable[[Any], Any]:
    """A ``panelary.shape`` transform as a ``frame -> frame`` callable.

    Every call builds a fresh instance and runs ``fit_transform`` on exactly the
    frame it is handed. For a stateless (``fit_is_empty``) transform the fit
    learns nothing, so any look-ahead or length dependence the verifiers find is
    the kernel's own. For a fitted one the fit sees the whole (possibly
    truncated) panel -- which is precisely the ``"window"`` claim that section 4
    below corroborates.
    """

    def op(frame: Any) -> Any:
        return factory().fit_transform(frame)

    return op


#: ``panelary.shape`` transforms, as ``name -> (rendering, factory)``, run on
#: the probe panel's two value columns. Windows are short enough that the
#: shortest entity (12 rows) still has full ones. The ``rowwise`` specs are
#: exactly the stateless transforms; ``rsvd`` and ``frequent_directions`` are
#: ``window`` specs, driven here for the evidence of section 4.
_SHAPE_FRAME_OPS: dict[str, tuple[str, Callable[[], Any]]] = (
    {}
    if _shape is None
    else {
        "paa": (
            "PAA(window=4, segments=2)",
            lambda: _shape.PAA(window=4, segments=2, columns=[VALUE, AUX]),
        ),
        "spectral": (
            "Spectral(window=4, k=2)",
            lambda: _shape.Spectral(window=4, k=2, columns=[VALUE, AUX]),
        ),
        "delay": (
            "Delay(lags=3, dilation=2)",
            lambda: _shape.Delay(lags=3, dilation=2, columns=[VALUE, AUX]),
        ),
        "sparse_rp": (
            "SparseRandomProjection(n_components=2, seed=0)",
            lambda: _shape.SparseRandomProjection(
                n_components=2, seed=0, columns=[VALUE, AUX]
            ),
        ),
        "srht": (
            "SRHT(n_components=2, seed=0)",
            lambda: _shape.SRHT(n_components=2, seed=0, columns=[VALUE, AUX]),
        ),
        "count_sketch": (
            "CountSketch(n_components=2, seed=0)",
            lambda: _shape.CountSketch(n_components=2, seed=0, columns=[VALUE, AUX]),
        ),
        "rsvd": (
            "RandomizedPCA(n_components=1)",
            lambda: _shape.RandomizedPCA(n_components=1, columns=[VALUE, AUX]),
        ),
        "frequent_directions": (
            "FrequentDirections(ell=2, n_components=1)",
            lambda: _shape.FrequentDirections(
                ell=2, n_components=1, columns=[VALUE, AUX]
            ),
        ),
    }
)
_FRAME_OPS.update(
    {name: _shape_op(factory) for name, (_r, factory) in _SHAPE_FRAME_OPS.items()}
)

#: ``panelary.covariance`` frame operations, as ``name -> (rendering, op)``.
#: Each returns one row per date (or panel rows, for ``kelly_jiang_beta``); the
#: per-date forms run with ``broadcast=True`` so the result is joined back onto
#: the probe panel's ``(entity, time)`` keys the verifiers compare on. Windows
#: are at most 5 dates (the shortest entity has 12 rows), and the tail
#: fractions are raised so six names a date still produce non-null values.
_COV_KEYS: dict[str, Any] = {"entity": ENTITY, "time": TIME}
_COVARIANCE_FRAME_OPS: dict[str, tuple[str, Callable[[Any], Any]]] = (
    {}
    if _cov is None
    else {
        "avg_correlation": (
            "avg_correlation(returns='x', window=5, broadcast=True)",
            lambda f: _cov.avg_correlation(
                f, returns=VALUE, window=5, broadcast=True, **_COV_KEYS
            ),
        ),
        "common_idio_vol": (
            "common_idio_vol(returns='x', window=4, fit_window=5, refit_every=3, "
            "broadcast=True)",
            lambda f: _cov.common_idio_vol(
                f,
                returns=VALUE,
                window=4,
                fit_window=5,
                refit_every=3,
                broadcast=True,
                **_COV_KEYS,
            ),
        ),
        "kelly_jiang_tail": (
            "kelly_jiang_tail(returns='x', window=3, q=0.25, min_exceedances=2, "
            "broadcast=True)",
            lambda f: _cov.kelly_jiang_tail(
                f,
                returns=VALUE,
                window=3,
                q=0.25,
                min_exceedances=2,
                broadcast=True,
                **_COV_KEYS,
            ),
        ),
        "kelly_jiang_beta": (
            "kelly_jiang_beta(returns='x', window=5, tail_window=3, q=0.25, "
            "min_exceedances=2)",
            lambda f: _cov.kelly_jiang_beta(
                f,
                returns=VALUE,
                window=5,
                tail_window=3,
                q=0.25,
                min_exceedances=2,
                **_COV_KEYS,
            ),
        ),
        "xs_wasserstein": (
            "xs_wasserstein(value='x', broadcast=True)",
            lambda f: _cov.xs_wasserstein(f, value=VALUE, broadcast=True, **_COV_KEYS),
        ),
        "avg_skewness": (
            "avg_skewness(returns='x', window=4, broadcast=True)",
            lambda f: _cov.avg_skewness(
                f, returns=VALUE, window=4, broadcast=True, **_COV_KEYS
            ),
        ),
    }
)
_FRAME_OPS.update({name: op for name, (_r, op) in _COVARIANCE_FRAME_OPS.items()})

#: Specs that cannot be driven through the verifiers at all, with the reason.
#: Both verifiers compare output cells keyed by ``(entity, time)``, so an
#: operator that does not hand back a panel cannot be checked by them.
_NOT_EXERCISABLE: dict[str, str] = {
    "ic": (
        "collapses the panel to one row per date, so the (entity, time) keys "
        "the verifiers compare on do not survive the call"
    ),
    "portfolio_sort": (
        "returns a SortResult summary object, not a panel frame, so there are "
        "no per-row outputs to compare"
    ),
}
if _shape is not None:
    _NOT_EXERCISABLE.update(
        {
            "column_subset": (
                "emits the kept input columns verbatim, so a per-row prefix check "
                "is degenerate by construction; that the column choice is learned "
                "from the fit panel only is tested in tests/test_shape_leak_safety.py"
            ),
            "cur": (
                "its transform is ColumnSubset's (the kept input columns "
                "verbatim) and its factors are not a panel; covered in "
                "tests/test_shape_leak_safety.py and tests/test_shape_roundtrip.py"
            ),
        }
    )

#: Failures that are already written up and waiting on an owner's decision. This
#: does NOT suppress anything -- the test still fails -- it only appends the
#: pointer to the message so the red is actionable rather than rediscovered.
_KNOWN_OPEN_DECISIONS: dict[str, str] = {}

#: Namespaces whose operators are reachable as neither an expression nor a frame
#: callable, with the reason and where they *are* covered instead. Empty today:
#: every namespace in the registry can be driven through the verifiers. The
#: mechanism is kept because delegating a whole namespace elsewhere is a decision
#: that must be written down rather than discovered from a skip count.
_NOT_EXERCISABLE_NAMESPACES: dict[str, str] = {}

#: Which key an ``evolve`` operator is partitioned by, read off its ``Op.kind``.
#: ``elem`` is element-wise, so the partition is immaterial -- entity is used to
#: match the scope the compiler applies.
_EVOLVE_OVER: dict[str, str] = {"ts": ENTITY, "xs": TIME, "elem": ENTITY}

#: Column arguments handed to a multi-ary evolve operator, in order.
_EVOLVE_ARGS = (VALUE, AUX, VALUE)


def _evolve_descriptor(spec: FeatureSpec) -> Any:
    """The ``Op`` behind an ``evolve_*`` spec, or ``None``."""
    if _evolve is None:
        return None
    return getattr(_evolve, "OPS", {}).get(spec.name.removeprefix("evolve_"))


def _build_evolve_op(spec: FeatureSpec) -> tuple[Any | None, str | None]:
    """Build an ``evolve`` operator's expression from its ``Op`` descriptor.

    The evolve vocabulary is not a Polars namespace -- the search compiler builds
    expressions from ``Op.build`` and applies the partition scope named by
    ``Op.kind``. Reconstructing that here is what lets these 52 specs be checked
    against the same two instruments as everything else, rather than delegated to
    ``tests/test_evolve_leakage.py`` (which covers only unary ``ts`` operators,
    only prefix invariance, and only with a NaN-masked ``allclose``).
    """
    if _evolve is None:  # pragma: no cover - evolve is a pure-Python subpackage
        return None, "panelary.evolve is not importable in this environment"
    op = _evolve_descriptor(spec)
    if op is None:
        return None, (
            f"no Op named {spec.name.removeprefix('evolve_')!r} in "
            "panelary.evolve.OPS, so the expression cannot be reconstructed"
        )
    over = _EVOLVE_OVER.get(op.kind)
    if over is None:
        return None, (
            f"Op.kind {op.kind!r} is not in _EVOLVE_OVER, so this suite does not "
            "know which key to partition by -- guessing would produce a "
            "confidently wrong verdict"
        )
    if op.arity > len(_EVOLVE_ARGS):
        return None, (
            f"arity {op.arity} needs more input columns than the probe panel "
            f"offers ({len(_EVOLVE_ARGS)})"
        )
    args = [pl.col(c) for c in _EVOLVE_ARGS[: op.arity]]
    param = op.params[0] if op.params else None
    expr = op.build(*args, param) if param is not None else op.build(*args)
    return expr.over(over).alias("__conformance__"), None


def _build_op(spec: FeatureSpec) -> tuple[Any | None, str | None]:
    """Return ``(op, None)`` for a runnable spec, or ``(None, reason)``.

    ``op`` is whatever the verifiers accept: a :class:`polars.Expr` already
    carrying its ``.over(...)``, or a ``frame -> frame`` callable.
    """
    if spec.name in _NOT_EXERCISABLE:
        return None, _NOT_EXERCISABLE[spec.name]
    if spec.name in _FRAME_OPS:
        return _FRAME_OPS[spec.name], None
    if spec.namespace in _NOT_EXERCISABLE_NAMESPACES:
        return None, _NOT_EXERCISABLE_NAMESPACES[spec.namespace]
    if spec.namespace == "evolve":
        return _build_evolve_op(spec)
    if spec.namespace not in _EXPR_NAMESPACES:
        return None, (
            f"namespace {spec.namespace!r} is not a Polars expression namespace "
            "and the operator has no frame-callable adapter registered here"
        )

    namespace = getattr(pl.col(VALUE), spec.namespace, None)
    if namespace is None:  # pragma: no cover - defended by test_namespaces.py
        return None, f"pl.Expr has no .{spec.namespace} namespace at runtime"

    method_name = _METHOD_ALIASES.get(spec.name, spec.name)
    method = getattr(namespace, method_name, None)
    if method is None:
        return None, f"no method pl.col(...).{spec.namespace}.{method_name}"

    required = [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required and spec.name not in _SYNTHESISED_ARGS:
        return None, (
            f"requires argument(s) {required} that cannot be synthesised; add an "
            "entry to _SYNTHESISED_ARGS to bring it under test"
        )

    args, kwargs = _SYNTHESISED_ARGS.get(spec.name, ((), {}))
    expr = method(*args, **kwargs)
    return expr.over(_OVER_KEY[spec.namespace]).alias("__conformance__"), None


def _call(spec: FeatureSpec) -> str:
    """A human-readable rendering of what this suite actually ran."""
    if spec.name in _SHAPE_FRAME_OPS:
        return f"panelary.shape.{_SHAPE_FRAME_OPS[spec.name][0]}.fit_transform(<probe panel>)"
    if spec.name in _COVARIANCE_FRAME_OPS:
        return f"panelary.covariance.{_COVARIANCE_FRAME_OPS[spec.name][0]}"
    if spec.name in _FRAME_OPS:
        return f"panelary.factor.{spec.name}(<probe panel>)"
    if spec.namespace == "evolve":
        op = _evolve_descriptor(spec)
        if op is None:  # pragma: no cover - only when the Op has gone missing
            return f"<no Op for {spec.qualified_name}>"
        cols = ", ".join(f"pl.col({c!r})" for c in _EVOLVE_ARGS[: op.arity])
        param = f", {op.params[0]!r}" if op.params else ""
        over = _EVOLVE_OVER.get(op.kind, "?")
        return f"OPS[{op.name!r}].build({cols}{param}).over({over!r})"
    args, kwargs = _SYNTHESISED_ARGS.get(spec.name, ((), {}))
    rendered = ", ".join(
        [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    )
    method = _METHOD_ALIASES.get(spec.name, spec.name)
    over = _OVER_KEY.get(spec.namespace, "?")
    return f"pl.col({VALUE!r}).{spec.namespace}.{method}({rendered}).over({over!r})"


# --------------------------------------------------------------------------- #
# Parametrisation
# --------------------------------------------------------------------------- #
_SPECS: list[FeatureSpec] = registry.all()
_IDS: list[str] = [s.qualified_name for s in _SPECS]


def _by_scope(scope: str) -> list[FeatureSpec]:
    return [s for s in _SPECS if s.safe_scope == scope]


_ROWWISE = _by_scope("rowwise")
_WINDOW = _by_scope("window")

#: Memo for ``assert_prefix_invariant`` verdicts, so the per-spec window check
#: and the window coverage tally do not each pay for the same 40-odd runs.
_PREFIX_VERDICTS: dict[str, bool] = {}


def _is_prefix_invariant(spec: FeatureSpec, op: Any) -> bool:
    """Run (once per spec) and cache: is ``op`` prefix-invariant on the panel?"""
    cached = _PREFIX_VERDICTS.get(spec.qualified_name)
    if cached is None:
        try:
            assert_prefix_invariant(op, _PANEL)
            cached = True
        except AssertionError:
            cached = False
        _PREFIX_VERDICTS[spec.qualified_name] = cached
    return cached


# --------------------------------------------------------------------------- #
# 0. The suite is not vacuous
# --------------------------------------------------------------------------- #
def test_registry_is_populated() -> None:
    """Every check below is parametrised over the registry; it must be full."""
    assert _SPECS, (
        "the registry is empty, so every parametrised check in this file "
        "expanded to zero cases and the suite proves nothing. Importing "
        "`panelary` must register the ts / xs / panel / factor catalogues."
    )
    missing = {"ts", "xs", "panel", "factor"} - {s.namespace for s in _SPECS}
    assert not missing, (
        f"namespace(s) {sorted(missing)} vanished from the registry. Either an "
        "operator catalogue stopped registering itself at import time, or a "
        "namespace was renamed without updating this suite."
    )


def test_no_exemption_has_gone_stale() -> None:
    """Every hand-written exemption must still name a live registered operator.

    :data:`_NOT_EXERCISABLE`, :data:`_SYNTHESISED_ARGS`, :data:`_FRAME_OPS` and
    :data:`_KNOWN_OPEN_DECISIONS` are the only places this suite lets itself off
    the hook. A stale entry -- a renamed or removed operator -- is how such a
    list rots into a blanket exemption that nobody re-reads.
    """
    names = {s.name for s in _SPECS}
    qualified = {s.qualified_name for s in _SPECS}
    for label, keys, universe in (
        ("_NOT_EXERCISABLE", set(_NOT_EXERCISABLE), names),
        ("_SYNTHESISED_ARGS", set(_SYNTHESISED_ARGS), names),
        ("_FRAME_OPS", set(_FRAME_OPS), names),
        ("_KNOWN_OPEN_DECISIONS", set(_KNOWN_OPEN_DECISIONS), qualified),
        ("_INTENTIONALLY_NOT_PANEL_SAFE", set(_INTENTIONALLY_NOT_PANEL_SAFE), names),
    ):
        stale = sorted(keys - universe)
        assert not stale, (
            f"{label} names {stale}, which is no longer in the registry. Remove "
            "the entry (or fix the name) so the exemption list keeps meaning "
            "what it says."
        )


def test_probe_panel_is_non_degenerate() -> None:
    """The probe data must be rich enough that a passing verifier means something.

    A verifier run on bland data returns a *vacuous* pass: the statistic never
    moved because the data never gave it anything to move about. These are the
    specific degeneracies that would hollow out the checks below.
    """
    df = _FRAME
    assert df[ENTITY].n_unique() >= 2, "need at least two entities"
    assert len(set(_ENTITY_LENGTHS.values())) >= 2, (
        "every entity has the same length, so a length-dependent warm-up would "
        "bite every entity identically and could hide inside the panel"
    )
    assert df[TIME].n_unique() >= 8, "need a time axis long enough to truncate"
    assert df[VALUE].null_count() > 0, "no nulls: null handling goes unexercised"

    per_entity = df.group_by(ENTITY).agg(
        pl.col(VALUE).n_unique().alias("n_unique"),
        pl.len().alias("n"),
        (pl.col(VALUE) > 0).any().alias("has_pos"),
        (pl.col(VALUE) < 0).any().alias("has_neg"),
    )
    assert (per_entity["n_unique"] < per_entity["n"]).all(), (
        "some entity has no repeated value, so duplicate- and reoccurrence-based "
        "operators are constant on it and pass vacuously"
    )
    assert per_entity["has_pos"].all() and per_entity["has_neg"].all(), (
        "some entity never changes sign, so sign- and streak-based operators are "
        "constant on it and pass vacuously"
    )


def test_this_suite_handles_every_scope_it_can_see() -> None:
    """A new ``safe_scope`` value must not fall through every check silently.

    ``rowwise`` is checked behaviourally, ``window`` structurally-plus-evidence,
    ``unspecified`` is rejected outright. A fourth value added to
    :data:`~panelary.registry.VALID_SAFE_SCOPES` would otherwise be covered by
    nothing at all, which is the failure mode this file exists to end.
    """
    unhandled = set(VALID_SAFE_SCOPES) - {"rowwise", "window", "unspecified"}
    assert not unhandled, (
        f"registry.VALID_SAFE_SCOPES gained {sorted(unhandled)}, which no check "
        "in tests/test_registry_conformance.py knows how to verify. Add a check "
        "for it before shipping specs that declare it."
    )


# --------------------------------------------------------------------------- #
# 1. Every spec declares a scope
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", _SPECS, ids=_IDS)
def test_spec_declares_a_safe_scope(spec: FeatureSpec) -> None:
    """No Panelary spec may be left ``"unspecified"``.

    The field defaults to ``"unspecified"`` so third-party specs keep working;
    the library's own catalogue does not get that dispensation. An operator that
    claims ``leakage_safe=True`` without saying *under which usage* is making a
    claim that cannot be checked, which is indistinguishable from not making one.
    """
    assert spec.safe_scope != "unspecified", (
        f"{spec.qualified_name} declares leakage_safe={spec.leakage_safe} but no "
        "safe_scope. Set safe_scope='rowwise' (the value at t uses only data at "
        "<= t, and this suite will verify that per row) or 'window' (causal only "
        "as the summary of an already-delimited window) on the FeatureSpec where "
        f"the {spec.namespace!r} catalogue registers it."
    )


# --------------------------------------------------------------------------- #
# 2. Structural consistency -- provable from metadata, no data required
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", _SPECS, ids=_IDS)
def test_scalar_output_is_never_rowwise(spec: FeatureSpec) -> None:
    """``output_shape="scalar"`` and ``safe_scope="rowwise"`` cannot both hold.

    A series-to-scalar reduction broadcast back over an entity puts the same
    number on every row, and that number is a function of the entity's *whole*
    series. The value at ``t = 0`` therefore depends on the observation at
    ``t = T``. This is a look-ahead by construction: it takes no data to
    establish, and no green test result should be trusted to refute it.
    """
    assert not (spec.output_shape == "scalar" and spec.safe_scope == "rowwise"), (
        f"{spec.qualified_name} declares output_shape='scalar' with "
        "safe_scope='rowwise'. Those contradict each other: reducing a series to "
        "one number and broadcasting it over the entity makes the value at every "
        "row a function of the entity's whole series, future included. The "
        "correct scope for a scalar aggregate is 'window'."
    )


# --------------------------------------------------------------------------- #
# 3. rowwise specs -- verified per row, with BOTH instruments
# --------------------------------------------------------------------------- #
def _op_or_skip(spec: FeatureSpec) -> Any:
    op, reason = _build_op(spec)
    if op is None:
        pytest.skip(f"{spec.qualified_name} not exercisable: {reason}")
    return op


def _triage(spec: FeatureSpec) -> str:
    """Append the open-decision pointer, if this failure is already written up."""
    note = _KNOWN_OPEN_DECISIONS.get(spec.qualified_name)
    return f"\n\nALREADY TRIAGED -- {note}" if note else ""


@pytest.mark.parametrize("spec", _ROWWISE, ids=[s.qualified_name for s in _ROWWISE])
def test_rowwise_spec_has_no_lookahead(spec: FeatureSpec, panel: PanelFrame) -> None:
    """Perturbing the future must not move a single past value.

    The value-dependence half of the pair. This catches an explicit
    ``shift(-1)``, a full-sample statistic, a threshold fitted on everything --
    anything whose output at ``t`` reacts to an observation after ``t``.
    """
    op = _op_or_skip(spec)
    try:
        assert_no_lookahead(op, panel)
    except AssertionError as exc:
        raise AssertionError(
            f"{spec.qualified_name} declares safe_scope='rowwise' but looks "
            f"ahead. Ran: {_call(spec)}\n\n{exc}{_triage(spec)}"
        ) from exc


#: Operators with a KNOWN, triaged prefix-invariance defect. An entry here is a
#: promise that the failure is understood and written up -- not permission to
#: ignore it. `strict=True` means the entry itself fails the build once the
#: defect is fixed, so a stale exemption cannot outlive the bug it names.
#:
#: A permanently-red gate is worse than no gate: AGENTS.md records that this
#: project's mypy job "had never once passed -- and a permanently-red required
#: check trains everyone to ignore CI, which is how two unrelated failures went
#: unnoticed." The same reasoning applies here.
_KNOWN_PREFIX_DEFECTS: dict[str, str] = {}


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            s,
            marks=pytest.mark.xfail(
                reason=_KNOWN_PREFIX_DEFECTS[s.qualified_name], strict=True
            ),
        )
        if s.qualified_name in _KNOWN_PREFIX_DEFECTS
        else s
        for s in _ROWWISE
    ],
    ids=[s.qualified_name for s in _ROWWISE],
)
def test_rowwise_spec_is_prefix_invariant(spec: FeatureSpec, panel: PanelFrame) -> None:
    """Truncating the panel must not move a single surviving value.

    The length-dependence half of the pair, and **not** implied by the one above:
    a value perturbation leaves the row count untouched, so a quantity that
    depends only on ``len(x)`` -- a warm-up capped at the series length, a window
    sized as ``f(T)``, a critical value indexed by the sample size -- sails
    through the perturbation test and fails here.
    """
    op = _op_or_skip(spec)
    try:
        assert_prefix_invariant(op, panel)
    except AssertionError as exc:
        raise AssertionError(
            f"{spec.qualified_name} declares safe_scope='rowwise' but its output "
            f"depends on how many rows follow it. Ran: {_call(spec)}\n\n{exc}"
            f"{_triage(spec)}"
        ) from exc


def test_rowwise_verification_coverage_is_accounted_for() -> None:
    """Every rowwise spec is either exercised, or skipped for a *registered* reason.

    A suite that silently skips most of what it claims to cover is worse than no
    suite, because the green tick gets read as verification. So the skips are not
    allowed to be incidental: an operator this file cannot build must be named in
    :data:`_NOT_EXERCISABLE` or :data:`_NOT_EXERCISABLE_NAMESPACES`, which forces
    a newly registered unbuildable operator to be looked at rather than absorbed.
    The failure message carries the full tally.
    """
    assert _ROWWISE, (
        "no spec declares safe_scope='rowwise', so both per-row verifications "
        "above expanded to zero cases. Either the scope migration has not landed "
        "(see test_spec_declares_a_safe_scope) or every operator in the "
        "catalogue was classified window-only."
    )

    exercised: list[str] = []
    accounted: list[str] = []
    unaccounted: list[str] = []
    for spec in _ROWWISE:
        op, reason = _build_op(spec)
        if op is not None:
            exercised.append(spec.qualified_name)
        elif (
            spec.name in _NOT_EXERCISABLE
            or spec.namespace in _NOT_EXERCISABLE_NAMESPACES
        ):
            accounted.append(f"{spec.qualified_name} ({reason})")
        else:
            unaccounted.append(f"{spec.qualified_name} ({reason})")

    tally = (
        f"rowwise specs: {len(_ROWWISE)} | verified with both instruments: "
        f"{len(exercised)} | skipped for a registered reason: {len(accounted)} | "
        f"skipped for no registered reason: {len(unaccounted)}"
    )
    assert not unaccounted, (
        f"{tally}\n\nThese rowwise specs could not be exercised and no reason is "
        "registered for them:\n  "
        + "\n  ".join(unaccounted)
        + "\n\nBring them under test (an entry in _SYNTHESISED_ARGS or "
        "_FRAME_OPS), or record why they cannot be, in _NOT_EXERCISABLE."
    )
    assert len(exercised) * 2 >= len(_ROWWISE), (
        f"{tally}\n\nFewer than half the rowwise specs are actually verified; the "
        "rest are skipped for registered but unresolved reasons:\n  "
        + "\n  ".join(accounted)
    )


# --------------------------------------------------------------------------- #
# 4. window specs -- the structural claim, plus evidence that it is real
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", _WINDOW, ids=[s.qualified_name for s in _WINDOW])
def test_window_spec_is_documented_window_only(spec: FeatureSpec) -> None:
    """A window-scope spec must be honest about what it is claiming.

    ``"window"`` says: causal as the summary of an already-delimited window --
    which is what :func:`panelary.extract_features` computes, one row per entity
    -- and a look-ahead the moment it is broadcast per row. That is a
    *qualification* of ``leakage_safe``, not a substitute for it, so the two must
    agree; and the scope is claimed against a shape that can actually hold it.
    """
    assert spec.leakage_safe, (
        f"{spec.qualified_name} declares safe_scope='window' but "
        "leakage_safe=False. safe_scope qualifies the leakage_safe claim; it "
        "does not stand in for one. Either the operator is causal at window "
        "scope (set leakage_safe=True) or it is not safe at all, and the scope "
        "label is meaningless."
    )
    assert spec.output_shape in {"scalar", "series", "frame"}, (
        f"{spec.qualified_name} declares an output_shape "
        f"({spec.output_shape!r}) this suite does not recognise, so the "
        "structural argument for its window scope cannot be checked."
    )


@pytest.mark.parametrize("spec", _WINDOW, ids=[s.qualified_name for s in _WINDOW])
def test_window_spec_is_not_prefix_invariant_per_row(
    spec: FeatureSpec, panel: PanelFrame
) -> None:
    """Evidence that ``"window"`` is a real constraint and not a rubber stamp.

    The structural argument in :func:`test_scalar_output_is_never_rowwise` is the
    proof; this is the corroboration. If a window-scope operator evaluated per
    row across the panel were *also* prefix-invariant, the classification would
    be costing users a feature for nothing -- so here a violation is the required
    outcome, and a clean run means the operator is degenerate on this data, not
    that the label is wrong.

    Note the asymmetry with the rowwise checks: there a violation is a bug; here
    a violation is the expected result.
    """
    op, reason = _build_op(spec)
    if op is None:
        pytest.skip(f"{spec.qualified_name} not exercisable: {reason}")

    if not _is_prefix_invariant(spec, op):
        return  # the expected outcome: the value moves as the series grows

    pytest.skip(
        f"{spec.qualified_name} is prefix-invariant on the probe panel, so the "
        "window classification is not cheaply demonstrable here. That is a "
        "statement about this data -- the statistic is degenerate on it -- and "
        "NOT evidence that the operator is row-safe; the structural rule in "
        f"test_scalar_output_is_never_rowwise still governs. Ran: {_call(spec)}"
    )


def test_window_evidence_coverage_is_accounted_for() -> None:
    """Most window specs must be demonstrably length-dependent, not just labelled.

    If nearly every window spec came back prefix-invariant on the probe panel,
    the honest conclusion would be that the probe data is too bland to say
    anything -- exactly the vacuous-pass failure mode this suite is built to
    avoid. The failure message carries the full tally.
    """
    assert _WINDOW, (
        "no spec declares safe_scope='window', which cannot be right while the "
        "registry holds series-to-scalar aggregates. Has the scope migration "
        "landed? (see test_spec_declares_a_safe_scope)"
    )

    demonstrated: list[str] = []
    degenerate: list[str] = []
    unexercisable: list[str] = []
    for spec in _WINDOW:
        op, reason = _build_op(spec)
        if op is None:
            unexercisable.append(f"{spec.qualified_name} ({reason})")
        elif _is_prefix_invariant(spec, op):
            degenerate.append(spec.qualified_name)
        else:
            demonstrated.append(spec.qualified_name)

    exercisable = len(demonstrated) + len(degenerate)
    tally = (
        f"window specs: {len(_WINDOW)} | demonstrably length-dependent: "
        f"{len(demonstrated)} | degenerate on the probe panel: {len(degenerate)} "
        f"| not exercisable: {len(unexercisable)}"
    )
    assert exercisable, f"{tally}\n\nno window spec could be exercised at all"
    assert len(demonstrated) * 10 >= exercisable * 8, (
        f"{tally}\n\nFewer than 80% of the exercisable window specs could be "
        "shown to be length-dependent, so the probe panel has gone too bland to "
        "corroborate the classification. These came back prefix-invariant:\n  "
        + "\n  ".join(degenerate)
    )


# --------------------------------------------------------------------------- #
# 5. Provenance / licence audit
# --------------------------------------------------------------------------- #
#: Operators that are *deliberately* not panel-safe: each mixes entities within
#: a date on purpose, which is the whole point of a cross-sectional operator.
#: Pinned rather than waved through, so a new entry has to be justified.
_INTENTIONALLY_NOT_PANEL_SAFE = frozenset(
    {
        "demean",
        "rank",
        "standardize",
        "ic",
        "orthogonalize",
        "portfolio_sort",
        # Cross-sectional distribution summaries: one value per date, from
        # every name on that date (plan covariance-and-market-state 4.4).
        "xs_dispersion",
        "xs_tail_index",
        "xs_up_share",
        "xs_entropy",
        # Covariance frame operations: per-date co-movement and distribution
        # features of the whole cross-section (plan covariance-and-market-state).
        "avg_correlation",
        "common_idio_vol",
        "kelly_jiang_tail",
        "kelly_jiang_beta",
        "xs_wasserstein",
        "avg_skewness",
    }
)


def test_registry_audit_is_clean() -> None:
    """The catalogue's provenance and licence guard still passes.

    :meth:`~panelary.registry.FeatureRegistry.audit` is the existing check, so
    this asserts its verdict rather than re-implementing it.
    ``missing_panel_safe`` is the one category not expected to be empty --
    cross-sectional operators mix entities by design -- so it is pinned to a
    named set instead of being asserted away.
    """
    report = registry.audit()

    assert report["missing_provenance"] == [], (
        f"operators with an empty `source`: {report['missing_provenance']}. "
        "Panelary's clean-room claim rests on provenance being recorded on "
        "every spec."
    )
    assert report["non_permissive_license"] == [], (
        "operators with a missing or non-permissive `license`: "
        f"{report['non_permissive_license']}. This is the GPL/copyleft "
        "tripwire; a copyleft upstream must not be redistributed here."
    )
    assert report["missing_leakage_safe"] == [], (
        f"operators declaring leakage_safe=False: {report['missing_leakage_safe']}"
        ". A registered operator that is not leak-free needs either a fix or a "
        "deliberate exclusion from the catalogue, not a quiet False."
    )

    unexpected = sorted(
        set(report["missing_panel_safe"]) - _INTENTIONALLY_NOT_PANEL_SAFE
    )
    assert not unexpected, (
        f"operators newly declaring panel_safe=False: {unexpected}. Not being "
        "panel-safe means the operator can bleed one entity's data into "
        "another's; that is defensible only for a deliberately cross-sectional "
        "operator, in which case add it to _INTENTIONALLY_NOT_PANEL_SAFE here."
    )


# --------------------------------------------------------------------------- #
# 6. The registration path itself must be able to satisfy rule 1
# --------------------------------------------------------------------------- #
def test_the_public_registration_decorator_can_declare_a_scope() -> None:
    """``register_feature`` must forward ``safe_scope`` to the spec it builds.

    ``AGENTS.md`` tells new operators to register via
    :func:`~panelary.registry.register_feature`, and
    :func:`test_spec_declares_a_safe_scope` requires every registered operator to
    declare a scope. If the decorator cannot pass one through, those two rules
    are in direct contradiction: every operator added the documented way is born
    ``"unspecified"`` and permanently in violation. Registered into a throwaway
    registry so the singleton is untouched.
    """
    params = inspect.signature(register_feature).parameters
    assert "safe_scope" in params, (
        "panelary.registry.register_feature takes no `safe_scope` argument, so "
        "an operator registered the documented way "
        '(AGENTS.md: "New operators should register a FeatureSpec via '
        'panelary.registry.register_feature") cannot declare one and is born '
        "'unspecified' -- permanently in violation of "
        'test_spec_declares_a_safe_scope. Add `safe_scope: str = "unspecified"` '
        "to the decorator and forward it to the FeatureSpec it builds. "
        f"Current parameters: {sorted(params)}"
    )

    scratch = FeatureRegistry()

    @register_feature(
        name="_conformance_probe",
        namespace="panel",
        input_shape="series",
        output_shape="series",
        tier="A",
        panel_safe=True,
        leakage_safe=True,
        safe_scope="rowwise",
        source="Panelary",
        license="Apache-2.0",
        target=scratch,
    )
    def _probe() -> None: ...

    assert scratch.get("_conformance_probe").safe_scope == "rowwise", (
        "register_feature accepted safe_scope but did not put it on the "
        "FeatureSpec, so every operator registered through the documented "
        "decorator is permanently 'unspecified'."
    )
