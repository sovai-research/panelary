"""One result type for every forecast-evaluation and inference test.

Every test in the forecast-evaluation family (Sharpe inference, forecast
comparison, VaR/ES backtests, calibration, luck versus skill) returns an
:class:`EvaluationResult`, vectorised over ``M`` models. Results from different
tests ``pl.concat`` into one evidence table with the fixed
:data:`EVALUATION_SCHEMA`, which is what a report builder needs.

No test returns a bare p-value: ``reference`` always names the null
distribution that was actually used (``"t(T-1)"``, ``"chi2(2)"``,
``"binomial-exact"``, ``"cbb-studentized(b=6,B=4999)"``, ...).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl

__all__ = ["EVALUATION_SCHEMA", "EvaluationResult", "evaluation_table"]

#: Column order and dtypes of :meth:`EvaluationResult.to_frame`. Fields that do
#: not apply to a test are null.
EVALUATION_SCHEMA: dict[str, pl.DataType] = {
    "test": pl.String(),
    "model": pl.String(),
    "estimate": pl.Float64(),
    "statistic": pl.Float64(),
    "pvalue": pl.Float64(),
    "pvalue_adj": pl.Float64(),
    "adjustment": pl.String(),
    "reference": pl.String(),
    "alternative": pl.String(),
    "std_error": pl.Float64(),
    "ci_low": pl.Float64(),
    "ci_high": pl.Float64(),
    "n_obs": pl.Int64(),
    "horizon": pl.Int64(),
    "hac": pl.String(),
    "block_length": pl.Int64(),
    "n_resamples": pl.Int64(),
    "seed": pl.Int64(),
    "warnings": pl.List(pl.String()),
}

_PER_MODEL_FLOAT = ("estimate", "statistic", "pvalue")


def _as_float_vector(value: Any, m: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(m, float(arr))
    if arr.shape != (m,):
        raise ValueError(f"`{name}` must have shape ({m},), got {arr.shape}.")
    return arr


def _json_safe(value: Any) -> Any:
    """Convert numpy containers to JSON-safe Python; non-finite floats become None."""
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass(frozen=True, eq=False)
class EvaluationResult:
    """The outcome of one test over ``M`` models.

    Attributes
    ----------
    test : str
        Test identifier, e.g. ``"sharpe_difference"``, ``"clark_west"``, ``"kupiec"``.
    names : tuple of str
        ``(M,)`` model or strategy labels.
    estimate, statistic, pvalue : numpy.ndarray
        ``(M,)`` point estimate (ΔSR, R²_OS, hit rate, ...), test statistic and
        p-value.
    reference : str
        The null distribution actually used, e.g. ``"t(T-1)"``, ``"chi2(2)"``.
    alternative : str
        ``"two-sided"``, ``"greater"`` or ``"less"`` (or a test-specific label).
    n_obs : numpy.ndarray
        ``(M,)`` int64 pairwise-complete observation count actually used.
    std_error : numpy.ndarray, optional
        ``(M,)`` standard error of ``estimate``.
    ci : numpy.ndarray, optional
        ``(M, 2)`` confidence interval for ``estimate``.
    pvalue_adj : numpy.ndarray, optional
        ``(M,)`` multiplicity-adjusted p-values, with the method in ``adjustment``.
    adjustment : str, optional
        ``"romano-wolf"``, ``"holm"``, ``"bh(pi0=0.81)"``, ...
    horizon : int
        Forecast horizon the test assumes (1 unless stated).
    hac : str, optional
        Long-run variance used, e.g. ``"bartlett(L=4)"``, ``"qs-pw(S=3.17)"``.
    block_length, n_resamples, seed : int, optional
        Resampling settings, when the test resamples.
    details : Mapping[str, numpy.ndarray]
        Test-specific extras (per-model arrays have leading dimension ``M``).
    warnings : tuple of str
        Human-readable caveats raised while computing the test.
    """

    test: str
    names: tuple[str, ...]
    estimate: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    reference: str
    alternative: str
    n_obs: np.ndarray
    std_error: np.ndarray | None = None
    ci: np.ndarray | None = None
    pvalue_adj: np.ndarray | None = None
    adjustment: str | None = None
    horizon: int = 1
    hac: str | None = None
    block_length: int | None = None
    n_resamples: int | None = None
    seed: int | None = None
    details: Mapping[str, np.ndarray] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.names)
        m = len(names)
        if len(set(names)) != m:
            raise ValueError("`names` must be unique.")
        object.__setattr__(self, "names", names)
        for attr in _PER_MODEL_FLOAT:
            object.__setattr__(
                self, attr, _as_float_vector(getattr(self, attr), m, attr)
            )
        n_obs = np.asarray(self.n_obs, dtype=np.int64)
        if n_obs.ndim == 0:
            n_obs = np.full(m, int(n_obs), dtype=np.int64)
        if n_obs.shape != (m,):
            raise ValueError(f"`n_obs` must have shape ({m},), got {n_obs.shape}.")
        object.__setattr__(self, "n_obs", n_obs)
        for attr in ("std_error", "pvalue_adj"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, _as_float_vector(value, m, attr))
        if self.ci is not None:
            ci = np.asarray(self.ci, dtype=np.float64)
            if ci.shape != (m, 2):
                raise ValueError(f"`ci` must have shape ({m}, 2), got {ci.shape}.")
            object.__setattr__(self, "ci", ci)
        if self.pvalue_adj is not None and self.adjustment is None:
            raise ValueError("`pvalue_adj` requires `adjustment` to name the method.")
        object.__setattr__(self, "details", dict(self.details))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def n_models(self) -> int:
        """Number of models ``M``."""
        return len(self.names)

    def to_frame(self) -> pl.DataFrame:
        """One row per model, with the fixed :data:`EVALUATION_SCHEMA`."""
        m = self.n_models
        nulls_f: list[float | None] = [None] * m
        ci_low = self.ci[:, 0].tolist() if self.ci is not None else nulls_f
        ci_high = self.ci[:, 1].tolist() if self.ci is not None else nulls_f
        data: dict[str, Any] = {
            "test": [self.test] * m,
            "model": list(self.names),
            "estimate": self.estimate.tolist(),
            "statistic": self.statistic.tolist(),
            "pvalue": self.pvalue.tolist(),
            "pvalue_adj": (
                self.pvalue_adj.tolist() if self.pvalue_adj is not None else nulls_f
            ),
            "adjustment": [self.adjustment] * m,
            "reference": [self.reference] * m,
            "alternative": [self.alternative] * m,
            "std_error": (
                self.std_error.tolist() if self.std_error is not None else nulls_f
            ),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_obs": self.n_obs.tolist(),
            "horizon": [self.horizon] * m,
            "hac": [self.hac] * m,
            "block_length": [self.block_length] * m,
            "n_resamples": [self.n_resamples] * m,
            "seed": [self.seed] * m,
            "warnings": [list(self.warnings)] * m,
        }
        return pl.DataFrame(data, schema=EVALUATION_SCHEMA)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary (non-finite floats become ``None``)."""
        payload: dict[str, Any] = {
            "test": self.test,
            "names": list(self.names),
            "estimate": self.estimate,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "reference": self.reference,
            "alternative": self.alternative,
            "n_obs": self.n_obs,
            "std_error": self.std_error,
            "ci": self.ci,
            "pvalue_adj": self.pvalue_adj,
            "adjustment": self.adjustment,
            "horizon": self.horizon,
            "hac": self.hac,
            "block_length": self.block_length,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "details": dict(self.details),
            "warnings": list(self.warnings),
        }
        return {k: _json_safe(v) for k, v in payload.items()}

    def __getitem__(self, name: str) -> EvaluationResult:
        """The result restricted to one model.

        Per-model ``details`` entries (leading dimension ``M``) are sliced;
        other entries are kept whole. ``pvalue_adj`` keeps the value adjusted
        over the full family.
        """
        try:
            i = self.names.index(name)
        except ValueError:
            raise KeyError(name) from None
        m = self.n_models
        sel = slice(i, i + 1)

        def pick(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[sel]

        details = {
            k: (
                v[sel]
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == m
                else v
            )
            for k, v in self.details.items()
        }
        return replace(
            self,
            names=(self.names[i],),
            estimate=self.estimate[sel],
            statistic=self.statistic[sel],
            pvalue=self.pvalue[sel],
            n_obs=self.n_obs[sel],
            std_error=pick(self.std_error),
            ci=pick(self.ci),
            pvalue_adj=pick(self.pvalue_adj),
            details=details,
        )

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(test={self.test!r}, n_models={self.n_models}, "
            f"reference={self.reference!r}, alternative={self.alternative!r})"
        )


def evaluation_table(results: Iterable[EvaluationResult]) -> pl.DataFrame:
    """Stack several results into one evidence table (:data:`EVALUATION_SCHEMA`)."""
    frames = [r.to_frame() for r in results]
    if not frames:
        return pl.DataFrame(schema=EVALUATION_SCHEMA)
    return pl.concat(frames, how="vertical")
