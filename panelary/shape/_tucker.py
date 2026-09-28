"""Partial Tucker (HOSVD-initialised HOOI) in numpy: compress tensor modes, keep entities.

The one call that justifies having a tensor representation at all::

    PartialTucker(rank={"time": 32, "feature": 8})   # X[E,T,F] -> core[E,32,8] + factors

Mode-unfold -> thin SVD -> mode-product, over ``np.moveaxis`` /
``np.tensordot``; the thin SVDs are :func:`~panelary.shape.randomized_svd`. No
TensorLy dependency: it is a test-only cross-check (``tests/test_shape_agreement.py``).

**Leak contract.** Compressing the ``time`` mode of a whole panel is a
whole-series operation: :class:`PartialTucker` is ``leakage_safe = False``, full
stop, and :meth:`~panelary.core.protocol.PanelTransformer._check_leakage`
refuses it across a train/test boundary. There is no trailing flavour: a
per-window Tucker is ``n_rows`` decompositions, which is not affordable and is
not pretended into existence. It describes a **fixed historical sample** --
regime analysis, factor structure over a training window, compressing a panel
for storage. For a row feature, use :class:`~panelary.shape.Delay` followed by
:class:`~panelary.shape.RandomizedPCA` (fit on training rows): that is SSA.

Entity is **never** compressed by default: compressing across entities mixes
one entity into another. ``rank={"entity": k}`` is permitted but narrows the
instance to ``panel_safe = False`` and is a research tool, not a feature path.

Reference: L. De Lathauwer, B. De Moor, J. Vandewalle (2000), "A multilinear
singular value decomposition" and "On the best rank-1 and rank-(R1,...,RN)
approximation of higher-order tensors", SIAM J. Matrix Anal. Appl. 21(4).
Clean-room implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import polars as pl

from panelary.core.panel_frame import PanelFrame
from panelary.reduce._base import _sign_of_max_abs
from panelary.shape._axes import (
    Axis,
    Flavour,
    InputShape,
    Intent,
    Plan,
    ShapeSpec,
    ShapeTransform,
    check_positive_int,
    check_seed,
)
from panelary.shape._rsvd import randomized_svd
from panelary.shape._tensor import Ragged, build_tensor, refuse_missing

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "MODES",
    "PartialTucker",
    "TuckerDecomposition",
    "fold",
    "hosvd",
    "mode_dot",
    "partial_tucker",
    "tucker_to_tensor",
    "unfold",
]

#: Mode names of a panel tensor, in axis order.
MODES: tuple[str, str, str] = ("entity", "time", "feature")


class TuckerDecomposition(NamedTuple):
    """Result of :func:`hosvd` / :func:`partial_tucker`.

    Attributes
    ----------
    core : numpy.ndarray
        The core tensor: the input with every compressed mode ``m`` replaced by
        ``ranks[m]`` components.
    factors : dict of int -> numpy.ndarray
        ``factors[m]`` is ``(I_m, r_m)`` with orthonormal, sign-fixed columns,
        for each compressed mode ``m``.
    n_iter : int
        HOOI sweeps run (0 for plain HOSVD).
    reconstruction_error : float
        ``||T - core x_m U_m||_F / ||T||_F``.
    """

    core: NDArray[np.float64]
    factors: dict[int, NDArray[np.float64]]
    n_iter: int
    reconstruction_error: float


def unfold(T: NDArray[Any], mode: int) -> NDArray[np.float64]:
    """Mode-``mode`` unfolding: ``(I_mode, prod(other dims))``, other axes in order."""
    A = np.asarray(T, dtype=np.float64)
    return np.moveaxis(A, mode, 0).reshape(A.shape[mode], -1)


def fold(M: NDArray[Any], mode: int, shape: Sequence[int]) -> NDArray[np.float64]:
    """Inverse of :func:`unfold`: rebuild a tensor of ``shape`` from its unfolding."""
    shape = tuple(int(s) for s in shape)
    rest = [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(
        np.asarray(M, dtype=np.float64).reshape(shape[mode], *rest), 0, mode
    )


def mode_dot(T: NDArray[Any], M: NDArray[Any], mode: int) -> NDArray[np.float64]:
    """Mode-``mode`` product ``T x_mode M``: ``M`` is ``(J, I_mode)``; axis ``mode`` becomes ``J``."""
    return np.moveaxis(
        np.tensordot(
            np.asarray(M, dtype=np.float64),
            np.asarray(T, dtype=np.float64),
            axes=(1, mode),
        ),
        0,
        mode,
    )


def tucker_to_tensor(
    core: NDArray[Any], factors: Mapping[int, NDArray[Any]]
) -> NDArray[np.float64]:
    """Reconstruct ``core x_m factors[m]`` over every factored mode."""
    out = np.asarray(core, dtype=np.float64)
    for m, U in factors.items():
        out = mode_dot(out, U, m)
    return out


def _resolve_ranks(
    T: NDArray[np.float64],
    ranks: Mapping[int | str, int] | Sequence[int],
    modes: Sequence[int | str] | None,
) -> dict[int, int]:
    def _mode(m: int | str) -> int:
        if isinstance(m, str):
            if m not in MODES:
                raise ValueError(f"unknown mode {m!r}; choose from {list(MODES)}.")
            return MODES.index(m)
        mi = int(m)
        if not 0 <= mi < T.ndim:
            raise ValueError(f"mode {m!r} outside a {T.ndim}-way tensor.")
        return mi

    if isinstance(ranks, Mapping):
        out = {_mode(m): int(r) for m, r in ranks.items()}
        if modes is not None and sorted(out) != sorted(_mode(m) for m in modes):
            raise ValueError("`modes` must name exactly the modes given in `ranks`.")
    else:
        if modes is None:
            raise ValueError("pass `modes=` when `ranks` is a sequence.")
        rs = list(ranks)
        ms = [_mode(m) for m in modes]
        if len(rs) != len(ms):
            raise ValueError("`ranks` and `modes` must have the same length.")
        out = dict(zip(ms, (int(r) for r in rs), strict=True))
    if not out:
        raise ValueError("at least one mode must be compressed.")
    for m, r in out.items():
        if r < 1 or r > T.shape[m]:
            raise ValueError(
                f"rank {r} for mode {m} is outside [1, {T.shape[m]}] (the mode's size)."
            )
    return dict(sorted(out.items()))


def _leading_left(M: NDArray[np.float64], r: int, seed: int) -> NDArray[np.float64]:
    """Leading ``r`` left singular vectors of ``M``, sign-fixed deterministically."""
    r = min(r, *M.shape)
    U, _, _ = randomized_svd(M, r, seed=seed)
    sign = np.array([_sign_of_max_abs(U[:, j]) for j in range(r)], dtype=np.float64)
    return U * sign[None, :]


def _project(
    T: NDArray[np.float64], factors: Mapping[int, NDArray[np.float64]]
) -> NDArray[np.float64]:
    out = T
    for m, U in factors.items():
        out = mode_dot(out, U.T, m)
    return out


def _rel_error(
    T: NDArray[np.float64],
    core: NDArray[np.float64],
    factors: Mapping[int, NDArray[np.float64]],
) -> float:
    """``||T - core x_m U_m||_F / ||T||_F``, computed directly.

    The shortcut ``sqrt(||T||^2 - ||core||^2)`` (valid for orthonormal
    factors) cancels catastrophically near an exact fit -- it reports ~1e-8
    for a reconstruction that is exact to 1e-15 -- so it is not used for the
    reported error.
    """
    norm_T = float(np.linalg.norm(T))
    if norm_T == 0.0:
        return 0.0
    return float(np.linalg.norm(T - tucker_to_tensor(core, factors)) / norm_T)


def _check_tensor(T: NDArray[Any], owner: str) -> NDArray[np.float64]:
    A = np.asarray(T, dtype=np.float64)
    if A.ndim < 2:
        raise ValueError(
            f"{owner}: expected a tensor of order >= 2, got shape {A.shape}."
        )
    if not np.isfinite(A).all():
        raise ValueError(
            f"{owner}: the tensor contains NaN/inf; decompositions cannot consume "
            "missing cells. Apply a Ragged policy (see panelary.shape.Ragged) first."
        )
    return A


def hosvd(
    T: NDArray[Any],
    ranks: Mapping[int | str, int] | Sequence[int],
    *,
    modes: Sequence[int | str] | None = None,
    seed: int = 0,
) -> TuckerDecomposition:
    """Truncated higher-order SVD over ``modes`` (the HOOI initialiser).

    Parameters
    ----------
    T : numpy.ndarray
        Finite tensor.
    ranks : mapping mode -> rank, or sequence aligned with ``modes``
        Modes are ints or the names ``"entity"``, ``"time"``, ``"feature"``.
    modes : sequence, optional
        The compressed modes; default the keys of ``ranks``.
    seed : int, default 0
        Seed of the randomized thin SVDs.

    Returns
    -------
    TuckerDecomposition
        With ``n_iter = 0``.
    """
    A = _check_tensor(T, "hosvd")
    seed = check_seed(seed, "hosvd")
    rk = _resolve_ranks(A, ranks, modes)
    factors = {m: _leading_left(unfold(A, m), r, seed) for m, r in rk.items()}
    core = _project(A, factors)
    return TuckerDecomposition(core, factors, 0, _rel_error(A, core, factors))


def partial_tucker(
    T: NDArray[Any],
    ranks: Mapping[int | str, int] | Sequence[int],
    *,
    modes: Sequence[int | str] | None = None,
    n_iter: int = 10,
    tol: float = 1e-10,
    seed: int = 0,
) -> TuckerDecomposition:
    """Partial Tucker by HOOI: compress ``modes`` only, leave the others intact.

    HOSVD initialisation, then alternating sweeps: for each mode ``m`` in
    ``modes``, project the other compressed modes onto their current factors
    and take the leading ``ranks[m]`` left singular vectors of the mode-``m``
    unfolding (:func:`~panelary.shape.randomized_svd`). Stops after ``n_iter``
    sweeps or when the relative error moves by less than ``tol``.

    Parameters
    ----------
    T : numpy.ndarray
        Finite tensor (NaN refused).
    ranks, modes
        As :func:`hosvd`.
    n_iter : int, default 10
        Maximum HOOI sweeps (0 = plain HOSVD).
    tol : float, default 1e-10
        Convergence tolerance on the relative reconstruction error.
    seed : int, default 0

    Returns
    -------
    TuckerDecomposition
    """
    A = _check_tensor(T, "partial_tucker")
    seed = check_seed(seed, "partial_tucker")
    if int(n_iter) < 0:
        raise ValueError(f"partial_tucker: n_iter must be >= 0, got {n_iter!r}.")
    rk = _resolve_ranks(A, ranks, modes)
    norm_T = float(np.linalg.norm(A)) or 1.0
    factors = {m: _leading_left(unfold(A, m), r, seed) for m, r in rk.items()}
    core = _project(A, factors)
    # With orthonormal factors the captured fraction ||core|| / ||T|| is
    # non-decreasing across HOOI sweeps; converge on its change.
    fit = float(np.linalg.norm(core)) / norm_T
    sweeps = 0
    for _ in range(int(n_iter)):
        if len(rk) == 1:
            break  # one factored mode: HOSVD is already optimal
        for m, r in rk.items():
            others = {m2: U for m2, U in factors.items() if m2 != m}
            factors[m] = _leading_left(unfold(_project(A, others), m), r, seed)
        core = _project(A, factors)
        new_fit = float(np.linalg.norm(core)) / norm_T
        sweeps += 1
        converged = abs(new_fit - fit) <= tol
        fit = new_fit
        if converged:
            break
    return TuckerDecomposition(core, factors, sweeps, _rel_error(A, core, factors))


class PartialTucker(ShapeTransform):
    """Partial Tucker of the ``(entity, time, feature)`` panel tensor -- whole-series.

    Builds the dense panel tensor (:func:`~panelary.shape.build_tensor`, no
    z-normalisation), applies an explicit :class:`~panelary.shape.Ragged`
    policy to missing cells, and runs :func:`partial_tucker` on the compressed
    modes. ``transform`` projects a panel on the **same dates** onto the fitted
    time/feature factors and returns **one row per entity** -- the flattened
    core -- keyed by ``entity`` alone (the time column carries the constant
    ``"whole_series"`` so a join onto rows fails loudly).

    ``leakage_safe = False``: compressing the time mode sees every date,
    future included. This is for describing a fixed historical sample; for a
    row feature use ``Delay`` + ``RandomizedPCA``.

    Parameters
    ----------
    rank : mapping of {"time", "feature", "entity"} -> int
        Components per compressed mode, e.g. ``{"time": 32, "feature": 8}``.
        Compressing ``"entity"`` narrows the instance to ``panel_safe = False``
        and disables :meth:`transform` (there is no per-entity output).
    n_iter : int, default 10
    tol : float, default 1e-10
    seed : int, default 0
    forward_fill : bool, default False
        Forward-fill missing cells within each entity (never backward) before
        the ragged policy is applied.
    ragged : {"refuse", "truncate"}, default "refuse"
        Missing cells on the union date grid are refused (naming the entities)
        or, with ``"truncate"`` and ``length=``, only the most recent
        ``length`` dates are kept.
    length : int, optional
    prefix : str, default "core"
        Output columns ``{prefix}_t{i}_f{j}``.
    columns, max_bytes, entity, time
        See :class:`~panelary.shape.ShapeTransform`.

    Attributes
    ----------
    core_ : numpy.ndarray
        ``(E, r_t, r_f)`` (uncompressed modes keep their full size).
    factors_ : dict of str -> numpy.ndarray
        ``factors_["time"]`` is ``(T, r_t)``, ``factors_["feature"]`` ``(F, r_f)``.
    entities_, times_ : polars.Series
        The fitted tensor's index.
    reconstruction_error_ : float
    n_iter_ : int
    """

    panel_safe = True
    leakage_safe = False
    fit_is_empty = False
    is_cross_sectional = False
    spec = ShapeSpec(
        intent=Intent.FACTORIZE,
        axis=Axis.TIME,
        flavour=Flavour.WHOLE_SERIES,
        width_rule="rank_dependent",
        invertible="from_factors",
        streaming="batch",
        cost_hint="O(E T F r n_iter)",
    )

    def __init__(
        self,
        rank: Mapping[str, int],
        *,
        n_iter: int = 10,
        tol: float = 1e-10,
        seed: int = 0,
        forward_fill: bool = False,
        ragged: Ragged | str = Ragged.REFUSE,
        length: int | None = None,
        prefix: str = "core",
        columns: Sequence[str] | str | None = None,
        max_bytes: int | None = None,
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(columns=columns, max_bytes=max_bytes, entity=entity, time=time)
        if not isinstance(rank, Mapping) or not rank:
            raise ValueError(
                "PartialTucker: `rank` must be a non-empty mapping such as "
                "{'time': 32, 'feature': 8}."
            )
        bad = [m for m in rank if m not in MODES]
        if bad:
            raise ValueError(
                f"PartialTucker: unknown mode(s) {bad}; choose from {list(MODES)}."
            )
        self.rank = {
            m: check_positive_int(r, f"rank[{m!r}]", "PartialTucker")
            for m, r in rank.items()
        }
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.seed = check_seed(seed, "PartialTucker")
        self.forward_fill = bool(forward_fill)
        self.ragged = Ragged(ragged)
        self.length = length
        self.prefix = prefix
        if "entity" in self.rank:
            self._narrow_contract(panel_safe=False)
        self.core_: NDArray[np.float64] | None = None
        self.factors_: dict[str, NDArray[np.float64]] = {}
        self.entities_: pl.Series | None = None
        self.times_: pl.Series | None = None
        self.reconstruction_error_: float | None = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------ #
    def _out_width(self, in_width: int) -> int | None:
        return None

    def _plan(self, shape: InputShape) -> Plan:
        E = max(shape.entities, 1)
        T = self.length if self.length is not None else max(1, shape.rows // E)
        F = shape.width
        r_t = min(self.rank.get("time", T), T)
        r_f = min(self.rank.get("feature", F), F)
        cells = float(E) * T * F
        return self._make_plan(
            shape,
            rows=E,
            width=r_t * r_f,
            scratch=8.0 * (4 * cells + E * r_t * r_f),
            knob="rank",
            output="entities",
        )

    def _tensor(self, panel: PanelFrame, cols: list[str]) -> Any:
        pt = build_tensor(
            panel, cols, forward_fill=self.forward_fill, z_normalize=False
        )
        return refuse_missing(
            pt, owner="PartialTucker", ragged=self.ragged, length=self.length
        )

    def _fit(self, panel: PanelFrame) -> None:
        cols = self._resolve_columns(panel)
        self.feature_names_in_ = cols
        pt = self._tensor(panel, cols)
        E, T, F = pt.tensor.shape
        self._enforce_budget(InputShape(rows=E * T, width=F, entities=E))
        res = partial_tucker(
            pt.tensor,
            {
                MODES.index(m): min(r, pt.tensor.shape[MODES.index(m)])
                for m, r in self.rank.items()
            },
            n_iter=self.n_iter,
            tol=self.tol,
            seed=self.seed,
        )
        self.core_ = res.core
        self.factors_ = {MODES[m]: U for m, U in res.factors.items()}
        self.entities_ = pt.entities
        self.times_ = pt.times
        self.reconstruction_error_ = res.reconstruction_error
        self.n_iter_ = res.n_iter

    def _names(self) -> list[str]:
        assert self.core_ is not None
        _, r_t, r_f = self.core_.shape
        return [f"{self.prefix}_t{i}_f{j}" for i in range(r_t) for j in range(r_f)]

    def _transform(self, panel: PanelFrame) -> PanelFrame:
        if "entity" in self.rank:
            raise ValueError(
                "PartialTucker: the entity mode is compressed, so there is no "
                "per-entity output to emit. Read `core_` / `factors_` directly."
            )
        assert self.times_ is not None
        pt = self._tensor(panel, self.feature_names_in_)
        if pt.times.len() != self.times_.len() or not pt.times.equals(self.times_):
            raise ValueError(
                "PartialTucker.transform: the panel's date grid differs from the "
                f"fitted one ({pt.times.len()} vs {self.times_.len()} dates). The "
                "time factor is indexed by the fitted dates, so projecting other "
                "dates onto it is meaningless."
            )
        E = pt.tensor.shape[0]
        self._enforce_budget(
            InputShape(
                rows=E * pt.tensor.shape[1], width=pt.tensor.shape[2], entities=E
            )
        )
        facs = {MODES.index(m): U for m, U in self.factors_.items()}
        core = _project(pt.tensor, facs)
        flat = core.reshape(E, -1)
        ent = panel.entity_col
        out = pt.entities.to_frame(ent).with_columns(
            pl.lit("whole_series").alias(panel.time_col),
            *[
                pl.Series(n, flat[:, i], dtype=pl.Float64)
                for i, n in enumerate(self._names())
            ],
        )
        return PanelFrame(out, entity=ent, time=panel.time_col, validate=False)

    def inverse_transform(
        self, Z: PanelFrame | pl.DataFrame | NDArray[Any]
    ) -> NDArray[np.float64]:
        """Reconstruct the ``(E, T, F)`` tensor from cores and the fitted factors.

        Parameters
        ----------
        Z : PanelFrame | polars.DataFrame | numpy.ndarray
            This transform's output (one row per entity) or an ``(E, r_t, r_f)``
            core array.

        Returns
        -------
        numpy.ndarray
            ``(E, T, F)`` on the fitted date grid.
        """
        self._check_fitted("inverse_transform")
        assert self.core_ is not None
        _, r_t, r_f = self.core_.shape
        if isinstance(Z, np.ndarray):
            core = np.asarray(Z, dtype=np.float64)
        else:
            frame = Z.collect() if isinstance(Z, PanelFrame) else Z
            core = (
                frame.select(self._names())
                .to_numpy()
                .astype(np.float64)
                .reshape(-1, r_t, r_f)
            )
        facs = {MODES.index(m): U for m, U in self.factors_.items() if m != "entity"}
        return tucker_to_tensor(core, facs)

    def _explain(self) -> pl.DataFrame:
        from panelary.shape._explain import explain_loadings

        frames = []
        if "feature" in self.factors_:
            U = self.factors_["feature"]
            frames.append(
                explain_loadings(
                    [f"feature_{j}" for j in range(U.shape[1])],
                    self.feature_names_in_,
                    U.T,
                    reconstruction_error=self.reconstruction_error_,
                )
            )
        if "time" in self.factors_ and self.times_ is not None:
            U = self.factors_["time"]
            frames.append(
                explain_loadings(
                    [f"time_{j}" for j in range(U.shape[1])],
                    [f"t={v}" for v in self.times_.to_list()],
                    U.T,
                    reconstruction_error=self.reconstruction_error_,
                )
            )
        if "entity" in self.factors_ and self.entities_ is not None:
            U = self.factors_["entity"]
            frames.append(
                explain_loadings(
                    [f"entity_{j}" for j in range(U.shape[1])],
                    [str(v) for v in self.entities_.to_list()],
                    U.T,
                    reconstruction_error=self.reconstruction_error_,
                )
            )
        return pl.concat(frames)
