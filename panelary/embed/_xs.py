"""Cross-sectional mode: fit on date t's cross-section, apply to date t's rows.

The per-date direction is the leak-safe one -- all of date ``t`` is observable
at ``t`` -- and it is what every replicated finance result does ("rank within
date, refit each period"). :class:`CrossSectionalEmbedder` reduces each date's
cross-section (optionally rank-transformed) to ``n_components`` principal
directions, fitted on that date alone.

The trap: per-date components are unidentified across dates
---------------------------------------------------------------
Each date's principal axes are defined only up to rotation (and sign, and
order) -- Gabaix et al., Appendix C. Emitting date-local scores as a time
series therefore produces a column whose meaning rotates arbitrarily from one
date to the next. This module **refuses** to do that: ``align`` has no default
and must be chosen explicitly:

``"procrustes"`` (recommended)
    Rotate date ``t``'s loadings onto date ``t-1``'s aligned loadings with the
    orthogonal Procrustes solution (:func:`procrustes_rotation`). The subspace
    is untouched; only its coordinate system is made continuous.
``"link"``
    The same Procrustes-aligned basis, plus the link penalty
    ``||z_t - z_{t-1}||^2``: an entity's score solves
    ``min_z ||x_t - V_t z||^2 + lambda ||z - z_{t-1}||^2``, i.e.
    ``z_t = (V_t' x_t + lambda z_{t-1}) / (1 + lambda)`` for entities scored on
    the previous date (``lambda = link_penalty``).

Both alignments run forward through the dates of the frame passed to
``transform``, reading only dates ``<= t``. The chain starts at the frame's
first usable date, so -- as for any recursive causal feature -- transform the
whole panel once and split afterwards.

:func:`procrustes_rotation` is also the aligner :mod:`panelary.shape`'s
``CrossSectionalRandomizedPCA`` was written to import.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.embed._contract import (
    _EmbedTransform,
    array_width,
    check_choice,
    check_pos_int,
    emit_embedding,
    matrix_from_columns,
    resolve_matrix_columns,
)
from panelary.shape._axes import Axis, InputShape, Intent, Plan, ShapeSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ALIGNMENTS", "CrossSectionalEmbedder", "procrustes_rotation"]

#: The two accepted cross-date alignments. There is deliberately no "none".
ALIGNMENTS: tuple[str, ...] = ("procrustes", "link")

_REFUSAL = (
    "per-date components are identified only up to rotation, sign and order across "
    "dates (Gabaix et al., App. C), so emitting them unaligned would turn arbitrarily "
    "rotating axes into a time series. Choose align='procrustes' (rotate onto the "
    "previous date) or align='link' (Procrustes plus the ||z_t - z_{t-1}||^2 link "
    "penalty); there is no unaligned option."
)


def procrustes_rotation(
    source: NDArray[Any], target: NDArray[Any]
) -> NDArray[np.float64]:
    """Orthogonal ``R`` minimising ``||source @ R - target||_F`` (Schonemann, 1966).

    Parameters
    ----------
    source, target : numpy.ndarray
        ``(d, k)`` loading matrices (same shape).

    Returns
    -------
    numpy.ndarray
        ``(k, k)`` orthogonal matrix ``U @ Vt`` where ``U S Vt = svd(source' target)``.
    """
    A = np.asarray(source, dtype=np.float64)
    B = np.asarray(target, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError(
            f"procrustes_rotation: shapes {A.shape} and {B.shape} must match (d, k)."
        )
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def _sign_fix(V: NDArray[np.float64]) -> NDArray[np.float64]:
    """Make each column's largest-magnitude loading positive (the ``reduce/`` rule)."""
    idx = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[idx, np.arange(V.shape[1])])
    return V * np.where(s == 0, 1.0, s)[None, :]


def _standardise(M: NDArray[np.float64], how: str | None) -> NDArray[np.float64]:
    if how == "rank":
        n = M.shape[0]
        r = np.argsort(
            np.argsort(M, axis=0, kind="stable"), axis=0, kind="stable"
        ).astype(np.float64)
        return r / max(n - 1, 1) - 0.5
    C = M - M.mean(axis=0)
    if how == "z":
        sd = M.std(axis=0)
        C = C / np.where(sd > 0, sd, 1.0)
    return C


class CrossSectionalEmbedder(_EmbedTransform):
    """Per-date principal-component embedding across entities, aligned across dates.

    Parameters
    ----------
    align : {"procrustes", "link"}
        **Required, no default** -- see the module docstring for why.
    n_components : int, default 3
    standardize : {"z", "rank", None}, default "z"
        Per-date column transform before the SVD: z-score, centred uniform
        rank (robust, the finance convention), or centring only.
    link_penalty : float, default 1.0
        ``lambda`` of ``align="link"`` (ignored otherwise); must be ``> 0``.
    min_cross_section : int, default 20
        Dates with fewer complete entities are emitted as null (the chain
        continues from the last usable date). Must exceed ``n_components``.
    columns : sequence of str, optional
        Numeric and/or numeric ``pl.Array`` columns (e.g. a temporal embedding).
    name : str, default "xs_embedding"
    output : {"array", "columns"}, default "array"
    dtype : {"float32", "float16", "float64"}, default "float32"
    keep_features : bool, default False
    max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.

    Notes
    -----
    ``is_cross_sectional = True``, ``panel_safe = False`` (entities are mixed
    within a date by design), ``leakage_safe = True`` (date ``t`` reads dates
    ``<= t`` only), ``fit_is_empty = True`` (``fit`` reads the schema; every
    per-date fit happens at transform time on that date's own rows).
    """

    panel_safe = False
    leakage_safe = True
    fit_is_empty = True
    is_cross_sectional = True
    spec = ShapeSpec(
        intent=Intent.COMPRESS,
        axis=Axis.ENTITY,
        width_rule="exact",
        invertible="none",
        streaming="batch",
        cost_hint="O(T E d k)",
    )
    _param_names: ClassVar[tuple[str, ...]] = (
        "align",
        "n_components",
        "standardize",
        "link_penalty",
        "min_cross_section",
        "name",
        "output",
        "dtype",
        "keep_features",
        "max_bytes",
    )

    def __init__(
        self,
        *,
        align: str,
        n_components: int = 3,
        standardize: str | None = "z",
        link_penalty: float = 1.0,
        min_cross_section: int = 20,
        columns: Sequence[str] | str | None = None,
        name: str = "xs_embedding",
        output: str = "array",
        dtype: str = "float32",
        keep_features: bool = False,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        owner = "CrossSectionalEmbedder"
        if align not in ALIGNMENTS:
            raise ValueError(f"{owner}: align={align!r} refused -- {_REFUSAL}")
        self.align = align
        self.n_components = check_pos_int(n_components, "n_components", owner)
        if standardize not in ("z", "rank", None):
            raise ValueError(
                f"{owner}: `standardize` must be 'z', 'rank' or None, got {standardize!r}."
            )
        self.standardize = standardize
        lp = float(link_penalty)
        if not np.isfinite(lp) or lp <= 0:
            raise ValueError(
                f"{owner}: `link_penalty` must be > 0, got {link_penalty!r}."
            )
        self.link_penalty = lp
        self.min_cross_section = check_pos_int(
            min_cross_section, "min_cross_section", owner, minimum=2
        )
        if self.min_cross_section <= self.n_components:
            raise ValueError(
                f"{owner}: `min_cross_section` must exceed `n_components`."
            )
        self.name = name
        self.output = check_choice(output, "output", ("array", "columns"), owner)
        self.dtype = check_choice(
            dtype, "dtype", ("float16", "float32", "float64"), owner
        )
        self.keep_features = bool(keep_features)
        self.in_width_: int = 0

    def _resolve_columns(self, panel: PanelFrame) -> list[str]:
        return resolve_matrix_columns(panel, self.columns, "CrossSectionalEmbedder")

    def _out_width(self, in_width: int) -> int:
        return self.n_components

    def _plan(self, shape: InputShape) -> Plan:
        k = self.n_components
        return self._make_plan(
            shape,
            rows=shape.rows,
            width=k,
            scratch=8.0 * shape.rows * (2 * shape.width + k),
            knob="n_components",
        )

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        d = array_width(panel.schema, cols)
        if d < self.n_components:
            raise ValueError(
                f"CrossSectionalEmbedder: input width {d} is below n_components={self.n_components}."
            )
        self.feature_names_in_ = cols
        self.in_width_ = d

    def _restore_hook(self) -> None:
        self.in_width_ = 0

    # ------------------------------------------------------------------ #
    def _per_date(
        self, panel: PanelFrame
    ) -> tuple[pl.DataFrame, NDArray[np.float64], list[tuple[int, int]]]:
        cols = list(self.feature_names_in_)
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"CrossSectionalEmbedder: column(s) {missing} not in panel."
            )
        ent, tim = panel.entity_col, panel.time_col
        frame = (
            panel.collect()
            .with_row_index("__pn_row__")
            .sort([tim, ent], maintain_order=True)
        )
        M = matrix_from_columns(frame, cols)
        t = frame[tim]
        starts = (
            np.flatnonzero(np.r_[True, (t[1:] != t[:-1]).to_numpy()])
            if frame.height
            else np.array([], dtype=np.int64)
        )
        bounds = list(
            zip(
                starts.tolist(),
                np.r_[starts[1:], frame.height].astype(np.int64).tolist(),
                strict=True,
            )
        )
        return frame, M, bounds

    def _date_basis(
        self, M: NDArray[np.float64]
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]] | None:
        ok = np.asarray(np.isfinite(M).all(axis=1), dtype=np.bool_)
        if int(ok.sum()) < self.min_cross_section:
            return None
        Ms = _standardise(M[ok], self.standardize)
        _, _, Vt = np.linalg.svd(Ms, full_matrices=False)
        return ok, Ms, Vt[: self.n_components].T

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        frame, M, bounds = self._per_date(panel)
        self._enforce_budget(
            InputShape(rows=frame.height, width=M.shape[1], entities=0)
        )
        k = self.n_components
        out = np.full((frame.height, k), np.nan)
        ent_codes = (
            frame[panel.entity_col].rank("dense").to_numpy().astype(np.int64) - 1
            if frame.height
            else np.zeros(0, np.int64)
        )
        n_ent = int(ent_codes.max()) + 1 if ent_codes.size else 0
        prev_score = np.full((n_ent, k), np.nan)
        prev_date = np.full(n_ent, -1, dtype=np.int64)
        V_prev: NDArray[np.float64] | None = None
        last_valid = -1
        for di, (lo, hi) in enumerate(bounds):
            got = self._date_basis(M[lo:hi])
            if got is None:
                continue
            ok, Ms, V = got
            V = _sign_fix(V) if V_prev is None else V @ procrustes_rotation(V, V_prev)
            Z = Ms @ V
            rows = np.arange(lo, hi)[ok]
            codes = ent_codes[rows]
            if self.align == "link" and last_valid >= 0:
                linked = prev_date[codes] == last_valid
                lam = self.link_penalty
                Z[linked] = (Z[linked] + lam * prev_score[codes[linked]]) / (1.0 + lam)
            out[rows] = Z
            prev_score[codes] = Z
            prev_date[codes] = di
            V_prev, last_valid = V, di
        restored = np.empty_like(out)
        order = frame["__pn_row__"].to_numpy()
        restored[order] = out
        base_frame = frame.sort("__pn_row__").drop("__pn_row__")
        base = (
            base_frame
            if self.keep_features
            else base_frame.select(panel.entity_col, panel.time_col)
        )
        valid = np.asarray(np.isfinite(restored).all(axis=1), dtype=np.bool_)
        res = emit_embedding(
            base,
            self.name,
            restored,
            valid,
            dtype=self.dtype,
            output=self.output,
            feature_names=[str(i) for i in range(k)],
        )
        return PanelFrame(
            res, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    def date_loadings(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        aligned: bool = True,
        entity: str | None = None,
        time: str | None = None,
    ) -> pl.DataFrame:
        """Per-date loadings: a *description* of each cross-section, not a feature.

        Parameters
        ----------
        X : PanelFrame | polars.DataFrame | polars.LazyFrame
        aligned : bool, default True
            Return the Procrustes-aligned loadings (what :meth:`transform`
            projects on); ``False`` returns the raw date-local ones.
        entity, time : str, optional

        Returns
        -------
        polars.DataFrame
            ``time, component, input_index, loading``.
        """
        self._check_fitted("date_loadings")
        panel = self._as_panel(X, method="date_loadings", entity=entity, time=time)
        frame, M, bounds = self._per_date(panel)
        tim = panel.time_col
        parts: list[pl.DataFrame] = []
        V_prev: NDArray[np.float64] | None = None
        for lo, hi in bounds:
            got = self._date_basis(M[lo:hi])
            if got is None:
                continue
            V = got[2]
            if aligned:
                V = (
                    _sign_fix(V)
                    if V_prev is None
                    else V @ procrustes_rotation(V, V_prev)
                )
                V_prev = V
            d, k = V.shape
            parts.append(
                pl.DataFrame(
                    {
                        "component": np.repeat(np.arange(k), d),
                        "input_index": np.tile(np.arange(d), k),
                        "loading": V.T.reshape(-1),
                    }
                ).with_columns(
                    pl.lit(frame[tim][lo], dtype=frame.schema[tim]).alias(tim)
                )
            )
        if not parts:
            return pl.DataFrame(
                schema={
                    tim: frame.schema[tim],
                    "component": pl.Int64,
                    "input_index": pl.Int64,
                    "loading": pl.Float64,
                }
            )
        return pl.concat(parts).select(tim, "component", "input_index", "loading")
