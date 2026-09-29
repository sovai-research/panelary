"""Calibration diagnostics: CORP reliability diagrams and Murphy diagrams.

**CORP** (Dimitriadis, Gneiting & Jordan, 2021) replaces the binned
reliability diagram. Binned ECE depends on the number and placement of bins,
and the verdict moves when they move. CORP fits the recalibration curve with
the pool-adjacent-violators (PAV) algorithm instead. PAV is the isotonic
regression of the outcome on the forecast, so there is nothing to tune. The
same fit yields an exact decomposition of the mean score::

    mean_score = MCB - DSC + UNC

* ``MCB`` (miscalibration) is how much the score improves when the forecasts
  are replaced by their PAV-recalibrated values. It is ``>= 0``, and it is 0
  exactly when the forecasts are already isotonic-calibrated.
* ``DSC`` (discrimination) is how much the recalibrated forecasts improve on
  climatology (the constant forecast ``mean(y)``). It is ``>= 0``.
* ``UNC`` (uncertainty) is the score of climatology. It depends only on ``y``.

Both inequalities hold because PAV is simultaneously optimal for every Bregman
loss: the Brier and log scores for a binary outcome, and squared error for a
real-valued outcome (the ``functional="mean"`` case of Gneiting & Resin, 2023).

**The recalibrated values are a diagnostic, never a forecast.** PAV is fit
in-sample on the outcomes it is judged against. Reusing it as a recalibrated
forecast would be look-ahead. So :class:`CORPResult` has no ``predict`` or
``transform`` method.

**Murphy diagrams** (Ehm, Gneiting, Jordan & Krüger, 2016) plot the mean
*elementary* score against a threshold ``theta``. Every consistent scoring
function for a quantile, an expectile or a probability is a mixture of these
elementary scores. So one forecast dominates another for *every* consistent
score exactly when its curve is lower at every ``theta``.
:func:`murphy_diagram` evaluates the curve exactly, with sorted suffix sums and
``searchsorted``, in ``O((n + K) log n)`` per model.

numpy + polars only. PAV has an optional numba kernel (the ``fast`` extra)
that matches its pure-Python twin bitwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from panelary._internal._jit import lazy_njit
from panelary.validation._results import _json_safe

__all__ = ["CORPResult", "corp_reliability", "murphy_diagram"]

#: ``band="auto"`` computes the band only when the replicate matrix
#: (``n_boot`` x ``G``, float64) has at most this many cells (about 400 MB).
_BAND_AUTO_CELLS = 50_000_000
#: An explicitly requested band refuses to allocate more than this (about 2 GB).
_BAND_MAX_CELLS = 250_000_000
#: Rows (``B``) of the band replicate matrix drawn and pooled per chunk.
_BAND_CHUNK_CELLS = 4_000_000
#: :func:`murphy_diagram` refuses a default breakpoint grid larger than this
#: (``K * M`` output rows); pass ``thetas`` instead.
_MURPHY_MAX_ROWS = 50_000_000


# --------------------------------------------------------------------------- #
# Pool-adjacent-violators
# --------------------------------------------------------------------------- #
# Both implementations below perform the same floating-point operations in the
# same order (block sums are accumulated left to right, and a block value is
# always ``sum / weight``), so they agree bitwise. Entries with non-positive
# weight are skipped and come back as NaN; the confidence band relies on that.
@lazy_njit
def _pav_rows_kernel(
    sums: np.ndarray, weights: np.ndarray, out: np.ndarray
) -> None:  # pragma: no cover - exercised only when numba is installed
    n_rows, n = sums.shape
    bs = np.empty(n, dtype=np.float64)
    bw = np.empty(n, dtype=np.float64)
    bv = np.empty(n, dtype=np.float64)
    bst = np.empty(n, dtype=np.int64)
    for r in range(n_rows):
        k = -1
        for i in range(n):
            wi = weights[r, i]
            if not wi > 0.0:
                continue
            k += 1
            si = sums[r, i]
            bs[k] = si
            bw[k] = wi
            bv[k] = si / wi
            bst[k] = i
            while k > 0 and bv[k - 1] > bv[k]:
                bs[k - 1] = bs[k - 1] + bs[k]
                bw[k - 1] = bw[k - 1] + bw[k]
                bv[k - 1] = bs[k - 1] / bw[k - 1]
                k -= 1
        for i in range(n):
            out[r, i] = np.nan
        for j in range(k + 1):
            end = bst[j + 1] if j < k else n
            val = bv[j]
            for i in range(bst[j], end):
                if weights[r, i] > 0.0:
                    out[r, i] = val


def _pav_rows_python(sums: np.ndarray, weights: np.ndarray, out: np.ndarray) -> None:
    """Pure-Python twin of :func:`_pav_rows_kernel` (lists, bitwise identical)."""
    n_rows, n = sums.shape
    for r in range(n_rows):
        s_row = sums[r].tolist()
        w_row = weights[r].tolist()
        bs: list[float] = []
        bw: list[float] = []
        bv: list[float] = []
        bst: list[int] = []
        for i in range(n):
            wi = w_row[i]
            if not wi > 0.0:
                continue
            si = s_row[i]
            bs.append(si)
            bw.append(wi)
            bv.append(si / wi)
            bst.append(i)
            k = len(bv) - 1
            while k > 0 and bv[k - 1] > bv[k]:
                bs[k - 1] = bs[k - 1] + bs[k]
                bw[k - 1] = bw[k - 1] + bw[k]
                bv[k - 1] = bs[k - 1] / bw[k - 1]
                del bs[k], bw[k], bv[k], bst[k]
                k -= 1
        row = np.full(n, np.nan)
        if bst:
            starts = np.asarray(bst, dtype=np.int64)
            lengths = np.diff(np.append(starts, n))
            filled = np.repeat(np.asarray(bv, dtype=np.float64), lengths)
            row[starts[0] :] = filled
            row[~(weights[r] > 0.0)] = np.nan
        out[r] = row


def _pav_rows(sums: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted isotonic (non-decreasing) regression.

    Row ``r`` fits the block means ``sums[r, i] / weights[r, i]`` in the given
    (already sorted) order. Entries with ``weights <= 0`` are skipped and are
    NaN in the output. Uses the numba kernel when available.
    """
    s = np.ascontiguousarray(sums, dtype=np.float64)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    out = np.empty_like(s)
    kernel = _pav_rows_kernel.compiled()
    if kernel is not None:
        kernel(s, w, out)
    else:
        _pav_rows_python(s, w, out)
    return out


def _pav(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted isotonic regression of ``values`` in their given order."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, np.float64).ravel()
    return _pav_rows((v * w)[None, :], w[None, :])[0]


# --------------------------------------------------------------------------- #
# CORP reliability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class CORPResult:
    """A CORP reliability diagram and its exact score decomposition.

    ``mean_score == mcb - dsc + unc`` holds to rounding error.

    **The recalibrated values are an in-sample diagnostic.** They are the PAV
    fit of the outcomes on the forecasts, estimated on the very outcomes being
    evaluated. Using them as recalibrated forecasts is look-ahead. For that
    reason this class has no ``predict`` or ``transform`` method.

    Attributes
    ----------
    x : numpy.ndarray
        ``(G,)`` sorted unique forecast values.
    recalibrated : numpy.ndarray
        ``(G,)`` the PAV (isotonic) recalibration at each ``x``: the CORP
        reliability curve. A calibrated forecast has ``recalibrated ~ x``.
    weight : numpy.ndarray
        ``(G,)`` number of observations at each unique forecast value.
    band_low, band_high : numpy.ndarray or None
        ``(G,)`` pointwise band around the curve, or ``None`` when no band was
        computed. For ``band="consistency"`` the band shows where the CORP curve
        falls if the forecasts *were* calibrated: a curve outside the band is
        evidence of miscalibration at that ``x``. For ``band="confidence"`` it is
        a bootstrap confidence band for the recalibration curve itself.
    mean_score : float
        Mean score of the original forecasts. It is ``inf`` for the log score
        when a forecast of 0 or 1 meets the opposite outcome.
    mcb, dsc, unc : float
        Miscalibration, discrimination and uncertainty components.
    n_obs : int
        Number of (forecast, outcome) pairs used (non-finite pairs dropped).
    score : str
        ``"brier"``, ``"log"`` or ``"squared_error"``.
    functional : str
        ``"probability"`` or ``"mean"``.
    warnings : tuple of str
        Caveats raised while computing the diagnostic.
    band : str or None
        ``"consistency"``, ``"confidence"`` or ``None`` (no band).
    level : float or None
        Nominal pointwise coverage of the band.
    n_boot : int or None
        Resamples behind the band.
    seed : int or None
        Seed of the band resampling.
    """

    x: np.ndarray
    recalibrated: np.ndarray
    weight: np.ndarray
    band_low: np.ndarray | None
    band_high: np.ndarray | None
    mean_score: float
    mcb: float
    dsc: float
    unc: float
    n_obs: int
    score: str
    functional: str
    warnings: tuple[str, ...] = ()
    band: str | None = None
    level: float | None = None
    n_boot: int | None = None
    seed: int | None = None

    def to_frame(self) -> pl.DataFrame:
        """The reliability curve, one row per unique forecast value.

        Columns ``x``, ``recalibrated``, ``weight``, ``band_low`` and
        ``band_high`` (the band columns are null when no band was computed).
        """
        g = self.x.shape[0]
        nulls: list[float | None] = [None] * g
        return pl.DataFrame(
            {
                "x": self.x,
                "recalibrated": self.recalibrated,
                "weight": self.weight,
                "band_low": (
                    self.band_low.tolist() if self.band_low is not None else nulls
                ),
                "band_high": (
                    self.band_high.tolist() if self.band_high is not None else nulls
                ),
            },
            schema={
                "x": pl.Float64(),
                "recalibrated": pl.Float64(),
                "weight": pl.Float64(),
                "band_low": pl.Float64(),
                "band_high": pl.Float64(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary (non-finite floats become ``None``)."""
        payload: dict[str, Any] = {
            "functional": self.functional,
            "score": self.score,
            "n_obs": self.n_obs,
            "mean_score": self.mean_score,
            "mcb": self.mcb,
            "dsc": self.dsc,
            "unc": self.unc,
            "x": self.x,
            "recalibrated": self.recalibrated,
            "weight": self.weight,
            "band": self.band,
            "band_low": self.band_low,
            "band_high": self.band_high,
            "level": self.level,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "warnings": list(self.warnings),
        }
        return {k: _json_safe(v) for k, v in payload.items()}

    def __repr__(self) -> str:
        return (
            f"CORPResult(functional={self.functional!r}, score={self.score!r}, "
            f"n_obs={self.n_obs}, mean_score={self.mean_score:.6g}, "
            f"mcb={self.mcb:.6g}, dsc={self.dsc:.6g}, unc={self.unc:.6g})"
        )


def _mean_score(p: np.ndarray, y: np.ndarray, score: str) -> float:
    if score == "log":
        with np.errstate(divide="ignore"):
            loss = np.where(y == 1.0, -np.log(p), -np.log1p(-p))
        return float(np.mean(loss))
    d = p - y
    return float(np.mean(d * d))


def _check_level(level: float) -> float:
    lv = float(level)
    if not (0.0 < lv < 1.0):
        raise ValueError(f"`level` must be in (0, 1), got {level}.")
    return lv


def corp_reliability(
    forecast: np.ndarray | Sequence[float],
    outcome: np.ndarray | Sequence[float],
    *,
    functional: str = "probability",
    score: str = "brier",
    band: str | None = "auto",
    n_boot: int = 500,
    level: float = 0.90,
    seed: int = 0,
) -> CORPResult:
    """CORP reliability diagram with its exact MCB/DSC/UNC decomposition.

    The CORP approach of Dimitriadis, Gneiting & Jordan (2021): "consistent,
    optimally binned, reproducible, PAV-based". It is the stable replacement
    for a binned reliability diagram or a binned ECE. The recalibration curve is
    the isotonic regression of ``outcome`` on ``forecast`` (the
    pool-adjacent-violators algorithm, with ties in the forecast pooled first).
    There is no bin count or bin edge to choose, so the verdict cannot move
    with one.

    Parameters
    ----------
    forecast : array-like of shape (n,)
        Forecasts. For ``functional="probability"`` these are probabilities in
        ``[0, 1]``. For ``functional="mean"`` they are point forecasts of the
        mean.
    outcome : array-like of shape (n,)
        Realisations: ``{0, 1}`` for ``"probability"``, any real value for
        ``"mean"``. Pairs with a non-finite forecast or outcome are dropped.
    functional : {"probability", "mean"}, default "probability"
        ``"mean"`` is the Gneiting–Resin (2023) extension to real-valued
        outcomes under squared error.
    score : {"brier", "log", "squared_error"}, default "brier"
        The score to decompose. ``"log"`` needs ``functional="probability"``.
        With ``functional="mean"``, ``"brier"`` means squared error and is
        recorded as ``"squared_error"``.
    band : {"auto", "consistency", "confidence"} or None, default "auto"
        ``"consistency"`` (binary outcomes only) resamples outcomes under the
        hypothesis of calibration, ``y* ~ Bernoulli(x)``, and takes pointwise
        quantiles of the refitted curve. ``"confidence"`` resamples
        (forecast, outcome) pairs. ``"auto"`` picks ``"consistency"`` for
        probabilities and ``"confidence"`` for means. It computes the band only
        when the ``n_boot x G`` replicate matrix has at most 5e7 cells.
        Otherwise it returns no band and records a warning; the band is never
        silently subsampled. ``None`` skips the band.
    n_boot : int, default 500
        Resamples for the band.
    level : float, default 0.90
        Pointwise coverage of the band.
    seed : int, default 0
        Seed for the band resampling (one ``numpy.random.default_rng`` stream).

    Returns
    -------
    CORPResult
        The curve (``x``, ``recalibrated``, ``weight``), the optional band, and
        ``mean_score``, ``mcb``, ``dsc``, ``unc`` with
        ``mean_score == mcb - dsc + unc``.

    Raises
    ------
    ValueError
        On shape mismatch, no finite pairs, probabilities outside ``[0, 1]``,
        non-binary outcomes for ``"probability"``, or an unknown option.

    Notes
    -----
    The recalibrated values are an in-sample fit on the evaluated outcomes. They
    are a diagnostic, never a forecast (see :class:`CORPResult`).

    The log score of a forecast of exactly 0 or 1 that meets the opposite
    outcome is infinite. It is reported as ``inf`` with a warning, never
    clipped.

    References
    ----------
    Dimitriadis, T., Gneiting, T. & Jordan, A. I. (2021). Stable reliability
    diagrams for probabilistic classifiers. *PNAS* 118(8), e2016191118.

    Gneiting, T. & Resin, J. (2023). Regression diagnostics meets forecast
    evaluation. *Electron. J. Statist.* 17, 3226–3286.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = rng.random(2000)
    >>> y = (rng.random(2000) < p).astype(float)
    >>> res = corp_reliability(p, y, band=None)
    >>> bool(abs(res.mean_score - (res.mcb - res.dsc + res.unc)) < 1e-12)
    True
    """
    if functional not in ("probability", "mean"):
        raise ValueError(
            f"`functional` must be 'probability' or 'mean', got {functional!r}."
        )
    if functional == "probability":
        if score not in ("brier", "log"):
            raise ValueError(
                "`score` must be 'brier' or 'log' for functional='probability', "
                f"got {score!r}."
            )
        score_name = score
    else:
        if score not in ("brier", "squared_error"):
            raise ValueError(
                "`score` must be 'squared_error' (or its alias 'brier') for "
                f"functional='mean', got {score!r}."
            )
        score_name = "squared_error"
    if band not in ("auto", "consistency", "confidence", None):
        raise ValueError(
            f"`band` must be 'auto', 'consistency', 'confidence' or None, got {band!r}."
        )
    if band == "consistency" and functional != "probability":
        raise ValueError(
            "band='consistency' resamples y* ~ Bernoulli(x), so it needs "
            "functional='probability'; use band='confidence' for means."
        )
    lv = _check_level(level)
    n_boot = int(n_boot)
    if band is not None and n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1, got {n_boot}.")

    x_all = np.asarray(forecast, dtype=np.float64).ravel()
    y_all = np.asarray(outcome, dtype=np.float64).ravel()
    if x_all.shape != y_all.shape:
        raise ValueError(
            f"`forecast` and `outcome` must have the same length, got "
            f"{x_all.shape[0]} and {y_all.shape[0]}."
        )
    keep = np.isfinite(x_all) & np.isfinite(y_all)
    x = x_all[keep]
    y = y_all[keep]
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("no pairs with a finite forecast and outcome.")
    notes: list[str] = []
    dropped = int(x_all.shape[0] - n)
    if dropped:
        notes.append(f"dropped {dropped} pairs with a non-finite forecast or outcome")
    if functional == "probability":
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("probability forecasts must lie in [0, 1].")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError(
                "functional='probability' needs binary outcomes in {0, 1}."
            )

    ux, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    w = counts.astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=ux.shape[0])
    recal = _pav_rows(s[None, :], w[None, :])[0]

    r_bar = float(s.sum() / w.sum())
    s_x = _mean_score(x, y, score_name)
    s_c = _mean_score(recal[inv], y, score_name)
    s_r = _mean_score(np.full(n, r_bar), y, score_name)
    mcb = s_x - s_c
    dsc = s_r - s_c
    if not math.isfinite(s_x):
        notes.append(
            "log score is infinite: a forecast of 0 or 1 met the opposite outcome"
        )

    band_kind: str | None = None
    band_low: np.ndarray | None = None
    band_high: np.ndarray | None = None
    if band is not None:
        kind = band
        if band == "auto":
            kind = "consistency" if functional == "probability" else "confidence"
        g = ux.shape[0]
        cells = n_boot * (g if kind == "consistency" else max(g, n))
        if band == "auto" and cells > _BAND_AUTO_CELLS:
            notes.append(
                f"band skipped: n_boot x size = {cells:.3g} cells exceeds the "
                f"band='auto' limit of {_BAND_AUTO_CELLS:.0e}; pass "
                f"band={kind!r} explicitly (or a smaller n_boot) to compute it"
            )
        elif cells > _BAND_MAX_CELLS:
            raise ValueError(
                f"the {kind} band needs {cells:.3g} replicate cells (limit "
                f"{_BAND_MAX_CELLS:.0e}); lower `n_boot`."
            )
        else:
            rng = np.random.default_rng(seed)
            if kind == "consistency":
                reps = _consistency_replicates(ux, counts, n_boot, rng)
            else:
                reps = _confidence_replicates(ux, inv, y, n_boot, rng)
            q = np.quantile(reps, [(1.0 - lv) / 2.0, (1.0 + lv) / 2.0], axis=0)
            band_low, band_high = q[0], q[1]
            band_kind = kind

    return CORPResult(
        x=ux,
        recalibrated=recal,
        weight=w,
        band_low=band_low,
        band_high=band_high,
        mean_score=s_x,
        mcb=mcb,
        dsc=dsc,
        unc=s_r,
        n_obs=n,
        score=score_name,
        functional=functional,
        warnings=tuple(notes),
        band=band_kind,
        level=lv if band_kind is not None else None,
        n_boot=n_boot if band_kind is not None else None,
        seed=int(seed) if band_kind is not None else None,
    )


def _consistency_replicates(
    ux: np.ndarray, counts: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """PAV curves of ``y* ~ Bernoulli(x)``, drawn as per-group binomial counts.

    When every forecast value is unique the counts are single Bernoulli draws,
    taken as ``uniform < x`` (about 5x faster than ``Generator.binomial``).
    """
    g = ux.shape[0]
    out = np.empty((n_boot, g))
    w_row = counts.astype(np.float64)
    all_single = bool(np.all(counts == 1))
    step = max(1, _BAND_CHUNK_CELLS // max(g, 1))
    for start in range(0, n_boot, step):
        stop = min(n_boot, start + step)
        if all_single:
            k = (rng.random((stop - start, g)) < ux).astype(np.float64)
        else:
            k = rng.binomial(counts, ux, size=(stop - start, g)).astype(np.float64)
        weights = np.broadcast_to(w_row, k.shape)
        out[start:stop] = _pav_rows(k, weights)
    return out


def _confidence_replicates(
    ux: np.ndarray,
    inv: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """PAV curves of (forecast, outcome) pairs resampled with replacement.

    A unique forecast value absent from a resample is filled by linear
    interpolation between its present neighbours (constant beyond the ends).
    """
    g = ux.shape[0]
    n = y.shape[0]
    out = np.empty((n_boot, g))
    step = max(1, _BAND_CHUNK_CELLS // max(n, g, 1))
    for start in range(0, n_boot, step):
        stop = min(n_boot, start + step)
        rows = stop - start
        idx = rng.integers(0, n, size=(rows, n))
        flat = (np.arange(rows)[:, None] * g + inv[idx]).ravel()
        sums = np.bincount(flat, weights=y[idx].ravel(), minlength=rows * g)
        wts = np.bincount(flat, minlength=rows * g).astype(np.float64)
        fitted = _pav_rows(sums.reshape(rows, g), wts.reshape(rows, g))
        for r in range(rows):
            present = ~np.isnan(fitted[r])
            if not present.all():
                fitted[r] = np.interp(ux, ux[present], fitted[r][present])
        out[start:stop] = fitted
    return out


# --------------------------------------------------------------------------- #
# Murphy diagrams
# --------------------------------------------------------------------------- #
def _suffix(values: np.ndarray) -> np.ndarray:
    """``out[j] = sum(values[j:])``, with a trailing 0 (length ``n + 1``)."""
    out = np.zeros(values.shape[0] + 1)
    out[:-1] = np.cumsum(values[::-1])[::-1]
    return out


def _murphy_one(
    x: np.ndarray,
    y: np.ndarray,
    thetas: np.ndarray,
    functional: str,
    level: float,
    y_order: np.ndarray | None = None,
    y_center: float | None = None,
) -> np.ndarray:
    """One model's curve. ``y_order`` / ``y_center`` may be precomputed for ``y``."""
    n = x.shape[0]
    if functional == "probability":
        x0 = np.sort(x[y == 0.0])
        x1 = np.sort(x[y == 1.0])
        above0 = x0.shape[0] - np.searchsorted(x0, thetas, side="right")
        below1 = np.searchsorted(x1, thetas, side="right")
        return (thetas * above0 + (1.0 - thetas) * below1) / n
    if functional == "quantile":
        a = (y < x).astype(np.float64) - level
        ox = np.argsort(x, kind="stable")
        oy = np.argsort(y, kind="stable") if y_order is None else y_order
        ax = _suffix(a[ox])[np.searchsorted(x[ox], thetas, side="right")]
        ay = _suffix(a[oy])[np.searchsorted(y[oy], thetas, side="right")]
        return (ax - ay) / n
    # expectile (the mean is tau = 1/2). S = b * [(y - th)_+ - (y - th) 1{th < x}]
    # with b = |1{y < x} - tau|. The score is translation invariant, so shift by
    # the median outcome to keep the suffix sums small.
    # The comparisons use the raw values; only the sums are shifted.
    c = float(np.median(y)) if y_center is None else y_center
    tc = thetas - c
    b = np.abs((y < x).astype(np.float64) - level)
    by = b * (y - c)
    ox = np.argsort(x, kind="stable")
    oy = np.argsort(y, kind="stable") if y_order is None else y_order
    jx = np.searchsorted(x[ox], thetas, side="right")
    jy = np.searchsorted(y[oy], thetas, side="right")
    part_y = _suffix(by[oy])[jy] - tc * _suffix(b[oy])[jy]
    part_x = _suffix(by[ox])[jx] - tc * _suffix(b[ox])[jx]
    return (part_y - part_x) / n


def murphy_diagram(
    forecasts: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    *,
    functional: str = "mean",
    level: float = 0.5,
    thetas: np.ndarray | Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Murphy diagram: mean elementary scores over a threshold ``theta``.

    Ehm, Gneiting, Jordan & Krüger (2016) show that every consistent scoring
    function for a quantile, an expectile or an event probability is a mixture
    of elementary scores ``S_theta``. Model A dominates model B for **every**
    such score exactly when A's curve lies at or below B's at every ``theta``.
    The elementary scores used are:

    * quantile at level ``alpha``:
      ``(1{y < x} - alpha) (1{theta < x} - 1{theta < y})``. The curve is piecewise
      constant, and its integral over ``theta`` is the pinball loss.
    * expectile at level ``tau`` (the mean is ``tau = 1/2``):
      ``|1{y < x} - tau| [(y - theta)_+ - (x - theta)_+ - (y - x) 1{theta < x}]``.
      The curve is piecewise linear, and its integral is half the expectile
      score ``|1{y < x} - tau| (x - y)^2``. For the mean, that is a quarter of the
      squared error.
    * probability (binary ``y``):
      ``theta 1{y = 0, x > theta} + (1 - theta) 1{y = 1, x <= theta}``. The curve is
      piecewise linear on ``[0, 1]``, and its integral is half the Brier score.

    The curves are computed exactly at every ``theta``, from sorted suffix sums
    and ``searchsorted``, in ``O((n + K) log n)`` per model.

    Parameters
    ----------
    forecasts : array-like of shape (n,) or (n, M)
        Forecasts of the chosen functional, one column per model.
    y : array-like of shape (n,)
        Realisations (``{0, 1}`` for ``functional="probability"``).
    functional : {"mean", "expectile", "quantile", "probability"}, default "mean"
        The functional the forecasts target.
    level : float, default 0.5
        ``tau`` for ``"expectile"``, ``alpha`` for ``"quantile"``. It is ignored
        for ``"mean"`` (always 1/2) and for ``"probability"``.
    thetas : array-like, optional
        Evaluation grid. The default is the sorted union of every model's
        forecasts and the outcomes: the breakpoints of the curves, where they
        are evaluated exactly. A user grid is exact at its points too. The
        default grid is refused when ``K * M`` would exceed 5e7 rows; pass a
        grid in that case.
    names : sequence of str, optional
        Model labels (default ``model_0``, ``model_1``, ...).

    Returns
    -------
    polars.DataFrame
        Long frame with columns ``theta``, ``model`` and
        ``mean_elementary_score``. Each model uses the pairs where both its
        forecast and ``y`` are finite.

    Raises
    ------
    ValueError
        On shape mismatch, an unknown functional, ``level`` outside ``(0, 1)``,
        invalid probabilities or outcomes, or a default grid that is too large.

    References
    ----------
    Ehm, W., Gneiting, T., Jordan, A. & Krüger, F. (2016). Of quantiles and
    expectiles: consistent scoring functions, Choquet representations and
    forecast rankings. *JRSS-B* 78(3), 505–562.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(500)
    >>> f = np.column_stack([np.zeros(500), np.full(500, 0.5)])
    >>> md = murphy_diagram(f, y, thetas=[-1.0, 0.0, 1.0], names=["zero", "biased"])
    >>> md.columns
    ['theta', 'model', 'mean_elementary_score']
    """
    if functional not in ("mean", "expectile", "quantile", "probability"):
        raise ValueError(
            "`functional` must be 'mean', 'expectile', 'quantile' or "
            f"'probability', got {functional!r}."
        )
    lv = 0.5 if functional in ("mean", "probability") else _check_level(level)
    fc = np.asarray(forecasts, dtype=np.float64)
    if fc.ndim == 1:
        fc = fc[:, None]
    if fc.ndim != 2:
        raise ValueError(f"`forecasts` must be 1-D or 2-D, got {fc.ndim}-D.")
    yy = np.asarray(y, dtype=np.float64).ravel()
    if fc.shape[0] != yy.shape[0]:
        raise ValueError(
            f"`forecasts` has {fc.shape[0]} rows but `y` has {yy.shape[0]}."
        )
    m = fc.shape[1]
    labels = _model_names(names, m)
    if functional == "probability":
        fin_y = yy[np.isfinite(yy)]
        if np.any((fin_y != 0.0) & (fin_y != 1.0)):
            raise ValueError(
                "functional='probability' needs binary outcomes in {0, 1}."
            )
        fin_x = fc[np.isfinite(fc)]
        if np.any((fin_x < 0.0) | (fin_x > 1.0)):
            raise ValueError("probability forecasts must lie in [0, 1].")

    if thetas is None:
        pool = np.concatenate([yy[np.isfinite(yy)], fc[np.isfinite(fc)]])
        grid = np.unique(pool)
        if grid.shape[0] * m > _MURPHY_MAX_ROWS:
            raise ValueError(
                f"the default breakpoint grid has {grid.shape[0]} points for {m} "
                f"models ({grid.shape[0] * m:.3g} rows); pass an explicit `thetas` "
                "grid (the curves are exact at any theta)."
            )
    else:
        grid = np.asarray(thetas, dtype=np.float64).ravel()
        if not np.all(np.isfinite(grid)):
            raise ValueError("`thetas` must be finite.")

    scores = np.full((m, grid.shape[0]), np.nan)
    y_ok = np.isfinite(yy)
    y_fin = yy[y_ok]
    # Models with no missing forecasts share one sort of y (and its median).
    y_order = np.argsort(y_fin, kind="stable") if functional != "probability" else None
    y_center = float(np.median(y_fin)) if y_fin.size else None
    for j in range(m):
        keep = np.isfinite(fc[:, j]) & y_ok
        if not keep.any():
            continue
        if np.array_equal(keep, y_ok):
            scores[j] = _murphy_one(
                fc[keep, j], y_fin, grid, functional, lv, y_order, y_center
            )
        else:
            scores[j] = _murphy_one(fc[keep, j], yy[keep], grid, functional, lv)
    frames = [
        pl.DataFrame(
            {"theta": grid, "mean_elementary_score": scores[j]},
            schema={"theta": pl.Float64(), "mean_elementary_score": pl.Float64()},
        ).select(
            "theta",
            pl.lit(labels[j], dtype=pl.String).alias("model"),
            "mean_elementary_score",
        )
        for j in range(m)
    ]
    return pl.concat(frames, how="vertical")


def _model_names(names: Sequence[str] | None, m: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"model_{j}" for j in range(m))
    labels = tuple(str(n) for n in names)
    if len(labels) != m:
        raise ValueError(f"`names` has {len(labels)} entries for {m} models.")
    if len(set(labels)) != m:
        raise ValueError("`names` must be unique.")
    return labels
