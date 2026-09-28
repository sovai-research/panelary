"""The embedding contract: serialisable state, the ``Embedder`` protocol, shared machinery.

Everything an ``embed/`` transform needs that is not the numeric kernel itself
lives here, so the kernels stay small and the contract has one definition:

* :class:`EmbeddingState` -- what is serialised: transform version, method,
  seed and parameters, input-column schema, and every fitted or drawn array
  (intervals, weights, scaler and bandwidth schedules, compression state).
  :meth:`EmbeddingState.fingerprint` hashes all of it, which is how
  ``tests/test_embed_stateless.py`` checks that ``fit_is_empty = True`` classes
  produce byte-identical state across disjoint fit sets.
* :class:`Embedder` -- the structural protocol every public embedder satisfies.
* :class:`_EmbedTransform` -- the base class (a
  :class:`~panelary.shape.ShapeTransform`, so ``fit_is_empty`` /
  ``is_cross_sectional`` are declared and enforced in exactly one place).
* :class:`_TemporalEmbedder` -- the trailing-window spine: sort once, frame
  each entity with :func:`panelary.shape._window.trailing_windows`, run the
  kernel once per chunk of rows, put rows back in input order.

Output layout (section 4 of the build contract)
-----------------------------------------------
Embeddings are emitted as **fixed-size** ``pl.Array`` columns (``Float32`` by
default; ``float16`` and ``float64`` on request). Thousands of top-level Polars
columns are never materialised unless the caller asks for ``output="columns"``
or calls :func:`panelary.shape.explode_embedding`. All arithmetic is float64;
only storage is narrowed. Measured storage error at ``float16`` is a median
relative error of ~1.7e-4 (QUANT features are bounded well inside float16's
range), which is the documented tolerance of the deterministic-inference
promise at that dtype.
"""

from __future__ import annotations

import abc
import hashlib
import json
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.shape._axes import InputShape, Plan, ShapeTransform
from panelary.shape._window import trailing_windows

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:  # pragma: no cover - typing_extensions ships with every type checker
        from typing_extensions import Self

    from numpy.typing import NDArray

__all__ = [
    "EMBED_STATE_VERSION",
    "Embedder",
    "EmbeddingState",
    "EmbeddingWarmupWarning",
]

#: Version of the serialised state layout. Bump on any incompatible change.
EMBED_STATE_VERSION: int = 1

_DTYPES: dict[str, tuple[Any, Any]] = {
    "float16": (np.float16, pl.Float16),
    "float32": (np.float32, pl.Float32),
    "float64": (np.float64, pl.Float64),
}


class EmbeddingWarmupWarning(UserWarning):
    """Rows without a full trailing window were dropped (``warmup="drop"``)."""


# --------------------------------------------------------------------------- #
# Serialisable state
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    """Normalise a parameter value to a canonical JSON-serialisable form."""
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"EmbeddingState: parameter value {value!r} ({type(value).__name__}) is "
        "not serialisable; embed parameters must be plain scalars or sequences."
    )


@dataclass(frozen=True)
class EmbeddingState:
    """Everything needed to reproduce an embedder's output exactly.

    Attributes
    ----------
    method : str
        The embedder's class name.
    version : int
        :data:`EMBED_STATE_VERSION` at the time of serialisation.
    params : dict
        Constructor parameters (seed, window, widths, policies, ...).
    columns : tuple of str
        Input-column schema the state was fitted for, in order.
    arrays : dict of str to numpy.ndarray
        Drawn or fitted arrays: intervals, random weights, scaler and bandwidth
        schedules, biases, compression components.

    Notes
    -----
    Deterministic inference is a compatibility promise: the same state and the
    same input give the same output. At ``dtype="float64"`` the promise is
    bit-level on one platform and BLAS; at ``float32`` / ``float16`` storage it
    holds to the storage precision.
    """

    method: str
    version: int
    params: dict[str, Any]
    columns: tuple[str, ...]
    arrays: dict[str, NDArray[Any]] = field(default_factory=dict)

    def header(self) -> dict[str, Any]:
        """The JSON-able part: method, version, params, columns, array index."""
        return {
            "method": self.method,
            "version": int(self.version),
            "params": _jsonable(self.params),
            "columns": list(self.columns),
            "arrays": {
                k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                for k, v in sorted(self.arrays.items())
            },
        }

    def fingerprint(self) -> str:
        """SHA-256 over the header and every array's raw bytes.

        Two states with the same fingerprint produce the same embedding. This
        is the byte-identity check behind ``fit_is_empty = True``.
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.header(), sort_keys=True).encode())
        for k in sorted(self.arrays):
            h.update(k.encode())
            h.update(np.array(self.arrays[k], order="C").tobytes())
        return h.hexdigest()

    def save(self, path: str | Path) -> Path:
        """Write the state to an ``.npz`` file (no pickle).

        Parameters
        ----------
        path : str or pathlib.Path
            Destination; ``.npz`` is appended by numpy if missing.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        path = Path(path)
        payload = {
            f"a__{k}": np.array(v, order="C", copy=True) for k, v in self.arrays.items()
        }
        payload["__header__"] = np.array(json.dumps(self.header(), sort_keys=True))
        np.savez(path, **payload)  # type: ignore[arg-type]
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingState:
        """Read a state written by :meth:`save` (``allow_pickle=False``).

        Raises
        ------
        ValueError
            If the file's state version is newer than this library's.
        """
        with np.load(Path(path), allow_pickle=False) as z:
            header = json.loads(str(z["__header__"]))
            arrays = {k[3:]: z[k].copy() for k in z.files if k.startswith("a__")}
        if int(header["version"]) > EMBED_STATE_VERSION:
            raise ValueError(
                f"EmbeddingState.load: state version {header['version']} is newer "
                f"than this library's ({EMBED_STATE_VERSION}); upgrade panelary."
            )
        return cls(
            method=header["method"],
            version=int(header["version"]),
            params=dict(header["params"]),
            columns=tuple(header["columns"]),
            arrays=arrays,
        )


@runtime_checkable
class Embedder(Protocol):
    """Structural protocol of every public ``embed/`` transform.

    ``fit_is_empty`` and ``is_cross_sectional`` are enforced, not documentary:
    :class:`~panelary.shape.ShapeTransform` hands a ``fit_is_empty`` class a
    zero-row frame in ``fit``, and refuses at class-definition time a
    cross-sectional class that claims ``panel_safe``.

    The transforms built in this package on ``_EmbedTransform`` additionally
    expose ``get_state() -> EmbeddingState`` and ``from_state``; the two rocket
    transforms keep their own introspection (``biases_``; ``state()``).
    """

    panel_safe: bool
    leakage_safe: bool
    fit_is_empty: bool
    is_cross_sectional: bool

    def fit(
        self, X: Any, *, entity: str | None = ..., time: str | None = ...
    ) -> Any: ...

    def transform(
        self, X: Any, *, entity: str | None = ..., time: str | None = ...
    ) -> PanelFrame: ...


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def check_seed(seed: object, owner: str) -> int:
    """Validate an explicit integer seed (``AGENTS.md`` invariant 2)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"{owner}: `seed` must be an explicit int, got {seed!r}.")
    return int(seed)


def check_pos_int(value: object, name: str, owner: str, *, minimum: int = 1) -> int:
    """Validate an integer ``>= minimum``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(
            f"{owner}: `{name}` must be an int >= {minimum}, got {value!r}."
        )
    return int(value)


def check_choice(value: object, name: str, allowed: Sequence[str], owner: str) -> str:
    """Validate a string option."""
    if value not in allowed:
        raise ValueError(
            f"{owner}: `{name}` must be one of {list(allowed)}, got {value!r}."
        )
    return str(value)


def matrix_from_columns(
    frame: pl.DataFrame, cols: Sequence[str]
) -> NDArray[np.float64]:
    """Read numeric and fixed-size ``pl.Array`` columns into one float64 matrix.

    Array columns contribute their full width, numeric columns one column
    each, concatenated in ``cols`` order. Nulls (a null array cell included)
    become NaN; nothing is imputed.
    """
    blocks: list[NDArray[np.float64]] = []
    n = frame.height
    for c in cols:
        s = frame[c]
        if isinstance(s.dtype, pl.Array):
            width = int(s.dtype.size)
            if n == 0:
                blocks.append(np.empty((0, width)))
                continue
            inner = s.cast(pl.Array(pl.Float64, width))
            null = inner.is_null().to_numpy()
            if null.any():
                inner = inner.fill_null(
                    pl.Series([[np.nan] * width], dtype=inner.dtype)
                )
            arr = np.array(inner.to_numpy(), dtype=np.float64, copy=True).reshape(
                n, width
            )
            if null.any():
                arr[null] = np.nan
            blocks.append(arr)
        else:
            blocks.append(
                s.cast(pl.Float64)
                .fill_null(np.nan)
                .to_numpy()
                .astype(np.float64)[:, None]
            )
    if not blocks:
        return np.empty((n, 0))
    return np.ascontiguousarray(np.hstack(blocks))


def resolve_matrix_columns(
    panel: PanelFrame, columns: Sequence[str] | None, owner: str
) -> list[str]:
    """Resolve input columns: numeric or ``pl.Array`` of numeric; default all of them."""
    schema = panel.schema

    def ok(dt: Any) -> bool:
        return bool(
            dt.is_numeric() or (isinstance(dt, pl.Array) and dt.inner.is_numeric())
        )

    if columns is not None:
        cols = list(columns)
        missing = [c for c in cols if c not in schema]
        if missing:
            raise ValueError(
                f"{owner}: column(s) {missing} not found. Available columns: {panel.columns}."
            )
        bad = [c for c in cols if not ok(schema[c])]
        if bad:
            raise ValueError(
                f"{owner}: column(s) {bad} are neither numeric nor a numeric pl.Array."
            )
    else:
        cols = [c for c in panel.feature_cols if ok(schema[c])]
    if not cols:
        raise ValueError(
            f"{owner}: no numeric (or numeric pl.Array) input columns; pass `columns=`."
        )
    return cols


def array_width(schema: Any, cols: Sequence[str]) -> int:
    """Total matrix width of ``cols`` (Array columns count their size)."""
    w = 0
    for c in cols:
        dt = schema[c]
        w += int(dt.size) if isinstance(dt, pl.Array) else 1
    return w


def emit_embedding(
    base: pl.DataFrame,
    name: str,
    values: NDArray[Any],
    valid: NDArray[np.bool_] | None,
    *,
    dtype: str,
    output: str,
    feature_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Attach an ``(n, D)`` float64 block to ``base`` as one Array column or D columns.

    Rows with ``valid == False`` become null (the whole array cell, or every
    column). ``values`` is cast to the storage ``dtype`` only here.
    """
    np_dt, pl_dt = _DTYPES[dtype]
    n, width = values.shape
    if output == "array":
        stored = np.ascontiguousarray(values.astype(np_dt, copy=False))
        if n == 0:
            s = pl.Series(name, [], dtype=pl.Array(pl_dt, width))
        else:
            s = pl.Series(name, stored).cast(pl.Array(pl_dt, width))
        out = base.with_columns(s)
        if valid is not None and not bool(np.all(valid)):
            out = out.with_columns(
                pl.when(pl.Series(values=valid, dtype=pl.Boolean))
                .then(pl.col(name))
                .alias(name)
            )
        return out
    names = (
        list(feature_names)
        if feature_names is not None
        else [str(i) for i in range(width)]
    )
    full_names = [f"{name}_{fname}" for fname in names]
    block = pl.from_numpy(
        np.ascontiguousarray(values, dtype=np.float64), schema=full_names, orient="row"
    )
    if valid is not None and not bool(np.all(valid)):
        keep = pl.Series("__pn_valid__", valid, dtype=pl.Boolean)
        block = block.with_columns(keep).select(
            pl.when(pl.col("__pn_valid__")).then(pl.col(c)).alias(c) for c in full_names
        )
    return base.with_columns(
        block.select(pl.col(c).cast(pl_dt) for c in full_names).get_columns()
    )


@dataclass(frozen=True)
class SortedRows:
    """A collected panel sorted ``(entity, time)`` plus the way back."""

    frame: pl.DataFrame
    order: NDArray[np.int64]
    lengths: NDArray[np.int64]
    entities: list[Any]
    times: pl.Series

    def values(self, cols: Sequence[str]) -> NDArray[np.float64]:
        """``cols`` as a float64 matrix in sorted order (nulls -> NaN)."""
        return matrix_from_columns(self.frame[self.order], cols)

    def unsort(self, block: NDArray[Any]) -> NDArray[Any]:
        """Scatter a sorted-order block back to input row order."""
        out = np.empty_like(block)
        out[self.order] = block
        return out


def sort_rows(panel: PanelFrame, owner: str) -> SortedRows:
    """Collect ``panel`` and compute its ``(entity, time)`` sort, refusing duplicate keys."""
    ent, tim = panel.entity_col, panel.time_col
    frame = panel.collect()
    keys = (
        frame.select(ent, tim)
        .with_row_index("__pn_row__")
        .sort([ent, tim], maintain_order=True)
    )
    if keys.height and keys.select(pl.struct(ent, tim).is_duplicated().any()).item():
        dup = (
            keys.filter(pl.struct(ent, tim).is_duplicated())
            .select(ent, tim)
            .head(3)
            .rows()
        )
        raise ValueError(
            f"{owner}: duplicate (entity, time) keys, e.g. {dup}; a trailing window "
            "needs each entity observed at most once per time."
        )
    order = keys["__pn_row__"].to_numpy().astype(np.int64)
    runs = keys.group_by(ent, maintain_order=True).len()
    return SortedRows(
        frame=frame,
        order=order,
        lengths=runs["len"].to_numpy().astype(np.int64),
        entities=runs[ent].to_list(),
        times=keys[tim],
    )


def iter_window_chunks(
    values: NDArray[np.float64],
    lengths: NDArray[np.int64],
    window: int,
    chunk_rows: int,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Yield ``(sorted_row_positions, windows)`` for every row with a full window.

    ``values`` is one column in ``(entity, time)`` order. Each entity is framed
    with :func:`panelary.shape._window.trailing_windows` (a view -- nothing is
    copied until a chunk is stacked), rows before the first full window are
    skipped, and views from several entities are stacked until ``chunk_rows``
    rows are collected, so the kernel runs over a batch, never per window and
    never per entity. A window never spans two entities.
    """
    pos_parts: list[NDArray[np.int64]] = []
    win_parts: list[NDArray[np.float64]] = []
    held = 0
    start = 0
    for n_e in lengths.tolist():
        if n_e >= window:
            view = trailing_windows(values[start : start + n_e], window)
            lo = window - 1
            while lo < n_e:
                take = min(n_e - lo, chunk_rows - held)
                pos_parts.append(
                    np.arange(start + lo, start + lo + take, dtype=np.int64)
                )
                win_parts.append(view[lo : lo + take])
                held += take
                lo += take
                if held >= chunk_rows:
                    yield (
                        np.concatenate(pos_parts),
                        np.ascontiguousarray(np.concatenate(win_parts)),
                    )
                    pos_parts, win_parts, held = [], [], 0
        start += n_e
    if held:
        yield np.concatenate(pos_parts), np.ascontiguousarray(np.concatenate(win_parts))


# --------------------------------------------------------------------------- #
# Base classes
# --------------------------------------------------------------------------- #
class _EmbedTransform(ShapeTransform):
    """Base of every ``embed/`` transform: state capture and restore.

    Subclasses list their constructor parameters in ``_param_names`` and their
    drawn/fitted array attributes in ``_state_arrays``; :meth:`get_state` and
    :meth:`from_state` are then generic.
    """

    _param_names: ClassVar[tuple[str, ...]] = ()
    _state_arrays: ClassVar[tuple[str, ...]] = ()

    def _state_params(self) -> dict[str, Any]:
        params = {name: getattr(self, name) for name in self._param_names}
        params["columns"] = self.columns
        params["entity"] = self._entity
        params["time"] = self._time
        return params

    def get_state(self) -> EmbeddingState:
        """The complete state (parameters, schema, drawn and fitted arrays).

        Returns
        -------
        EmbeddingState

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted("get_state")
        arrays: dict[str, NDArray[Any]] = {}
        for name in self._state_arrays:
            value = getattr(self, name)
            if value is not None:
                arrays[name] = np.array(value, copy=True)
        return EmbeddingState(
            method=type(self).__name__,
            version=EMBED_STATE_VERSION,
            params=self._state_params(),
            columns=tuple(self.feature_names_in_),
            arrays=arrays,
        )

    @classmethod
    def from_state(cls, state: EmbeddingState) -> Self:
        """Rebuild a fitted embedder from :meth:`get_state` output.

        Raises
        ------
        ValueError
            If ``state`` was produced by a different class.
        """
        if state.method != cls.__name__:
            raise ValueError(
                f"{cls.__name__}.from_state: state is for {state.method!r}."
            )
        params = dict(state.params)
        obj = cls(**params)
        obj.feature_names_in_ = list(state.columns)
        for name, value in state.arrays.items():
            setattr(obj, name, np.array(value, copy=True))
        obj._restore_hook()
        obj._fitted = True
        return obj

    def _restore_hook(self) -> None:
        """Rebuild derived (non-serialised) structures after :meth:`from_state`."""


class _TemporalEmbedder(_EmbedTransform):
    """Trailing-window embedder: one output row per ``(entity, time)`` row.

    Row ``t`` of an entity is embedded from ``x[t - window + 1 .. t]`` of that
    entity and nothing else. ``center=`` does not exist. Windows are
    fixed-length, so no quantity depends on how much history precedes or
    follows a row (prefix invariance, ``AGENTS.md`` invariant 1).

    NaN policy: a row whose window holds any missing value is emitted as null
    (never imputed). Rows before an entity's first full window (the warm-up)
    are dropped with an :class:`EmbeddingWarmupWarning` naming the affected
    entities (``warmup="drop"``, the default), or kept as null rows
    (``warmup="null"``).
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        *,
        window: int,
        columns: Sequence[str] | str | None = None,
        warmup: str = "drop",
        output: str = "array",
        dtype: str = "float32",
        suffix: str = "_emb",
        keep_features: bool = False,
        chunk_rows: int = 2048,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        owner = type(self).__name__
        self.window = check_pos_int(window, "window", owner, minimum=2)
        self.warmup = check_choice(warmup, "warmup", ("drop", "null"), owner)
        self.output = check_choice(output, "output", ("array", "columns"), owner)
        self.dtype = check_choice(dtype, "dtype", tuple(_DTYPES), owner)
        if not isinstance(suffix, str):
            raise TypeError(f"{owner}: `suffix` must be a str, got {suffix!r}.")
        self.suffix = suffix
        self.keep_features = bool(keep_features)
        self.chunk_rows = check_pos_int(chunk_rows, "chunk_rows", owner)

    _param_names: ClassVar[tuple[str, ...]] = (
        "window",
        "warmup",
        "output",
        "dtype",
        "suffix",
        "keep_features",
        "chunk_rows",
        "max_bytes",
    )

    # ------------------------------------------------------------------ #
    @property
    @abc.abstractmethod
    def feature_names_(self) -> list[str]:
        """Names of the per-column output features, in order."""

    @property
    def n_outputs(self) -> int:
        """Features emitted per input column."""
        return len(self.feature_names_)

    @abc.abstractmethod
    def _embed_windows(self, W: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(c, window)`` finite float64 windows -> ``(c, n_outputs)`` float64.

        Row ``i`` of the result must depend on ``W[i]`` alone.
        """

    def _out_width(self, in_width: int) -> int:
        return in_width * self.n_outputs

    def _plan(self, shape: InputShape) -> Plan:
        width = shape.width * self.n_outputs
        itemsize = np.dtype(_DTYPES[self.dtype][0]).itemsize
        scratch = 8.0 * self.chunk_rows * (self.window + 4 * self.n_outputs)
        scratch += 8.0 * shape.rows * (shape.width + self.n_outputs)
        scratch += float(itemsize) * shape.rows * width
        return self._make_plan(
            shape, rows=shape.rows, width=width, scratch=scratch, knob="window"
        )

    def _fit(self, panel: PanelFrame) -> None:
        # fit_is_empty: `panel` is a zero-row frame; only the schema is read.
        self.feature_names_in_ = self._resolve_columns(panel)

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        owner = type(self).__name__
        cols = list(self.feature_names_in_)
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise ValueError(f"{owner}.transform: column(s) {missing} not in panel.")
        sr = sort_rows(panel, owner)
        n = sr.frame.height
        self._enforce_budget(
            InputShape(rows=n, width=len(cols), entities=len(sr.entities))
        )
        X = sr.values(cols)
        full = sr.lengths >= self.window
        has_window = np.zeros(n, dtype=bool)
        starts = np.concatenate([[0], np.cumsum(sr.lengths)[:-1]]).astype(np.int64)
        for s0, n_e in zip(starts.tolist(), sr.lengths.tolist(), strict=True):
            if n_e >= self.window:
                has_window[s0 + self.window - 1 : s0 + n_e] = True
        # Results are written straight into input row order, in the storage
        # dtype: arithmetic is float64 per chunk, and no full-size float64 or
        # unsorted copy of the (n, D) block is ever held.
        np_dt = _DTYPES[self.dtype][0]
        warm = sr.unsort(~has_window)
        base = (
            sr.frame
            if self.keep_features
            else sr.frame.select(panel.entity_col, panel.time_col)
        )
        res = base
        for j, c in enumerate(cols):
            out = np.full((n, self.n_outputs), np.nan, dtype=np_dt)
            valid = ~warm  # per column: a gap in one column never nulls another
            for pos, W in iter_window_chunks(
                X[:, j], sr.lengths, self.window, self.chunk_rows
            ):
                rows = sr.order[pos]
                ok = np.isfinite(W).all(axis=1)
                if ok.any():
                    out[rows[ok]] = self._embed_windows(np.ascontiguousarray(W[ok]))
                valid[rows[~ok]] = False
            res = emit_embedding(
                res,
                f"{c}{self.suffix}",
                out,
                valid,
                dtype=self.dtype,
                output=self.output,
                feature_names=self.feature_names_,
            )
            del out
        if self.warmup == "drop" and warm.any():
            short = [
                e for e, ok in zip(sr.entities, full.tolist(), strict=True) if not ok
            ]
            msg = (
                f"{owner}: dropped {int(warm.sum())} warm-up row(s) without a full "
                f"{self.window}-row trailing window (the first {self.window - 1} rows "
                f"of each of {len(sr.entities)} entities)."
            )
            if short:
                shown = ", ".join(repr(e) for e in short[:10])
                more = f" and {len(short) - 10} more" if len(short) > 10 else ""
                msg += f" Entities with no full window at all (dropped entirely): {shown}{more}."
            msg += " Pass warmup='null' to keep them as null rows."
            warnings.warn(msg, EmbeddingWarmupWarning, stacklevel=3)
            res = res.filter(pl.Series(values=~warm, dtype=pl.Boolean))
        return PanelFrame(
            res, entity=panel.entity_col, time=panel.time_col, validate=False
        )
