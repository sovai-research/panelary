"""VaR / ES backtests: statistical evidence about the calibration of a risk forecast.

Every function here produces *statistical evidence about the calibration of a
risk forecast; not a regulatory determination*. No pass/fail zone ("traffic
light") is emitted. Exact tail probabilities are reported instead.

* :func:`exceedances` turns forecasts into hits in ``{0, 1, NaN}``. It serves
  VaR (a lower bound) and interval forecasts (``lower`` and ``upper``), because
  Christoffersen (1998) is an interval-forecast test.
* :func:`kupiec_test` tests unconditional coverage with an **exact binomial**
  p-value. The asymptotic chi-square LR over-rejects about 2x at T = 250 and
  VaR 1% (it is kept in ``details``).
* :func:`christoffersen_test` tests independence and conditional coverage. It
  uses Dufour's (2006) **exact Monte Carlo** test by default, because the
  asymptotic ``LR_ind`` is badly undersized (0.013 at a nominal 0.05).
* :func:`dynamic_quantile_test` is the Engle–Manganelli (2004) DQ regression
  test.
* :func:`acerbi_szekely_test` gives the Acerbi–Székely (2014) Z1/Z2 ES
  backtests. The p-values are simulated from a :class:`PredictiveSpec`.
* :func:`fz0_loss` (Patton, Ziegel & Chen, 2019) and :func:`qlike_loss`
  (Patton, 2011) are consistent losses for comparative backtesting. Feed them
  to Diebold–Mariano or the model confidence set.
* :func:`var_backtest` runs the standard battery and returns one evidence table
  (:data:`~panelary.validation._results.EVALUATION_SCHEMA`).

**Sign conventions.** Sign bugs are the most common VaR-backtest error, so
``var_convention`` is required wherever a VaR/ES level enters. Under
``"loss"`` VaR and ES are positive loss amounts, and a hit is ``y < -VaR``.
Under ``"quantile"`` they are the (negative) return quantile and tail mean,
and a hit is ``y < VaR``. Internally everything runs in the quantile
convention, where ``ES <= VaR < 0``.

**Alignment (trap T1).** The forecast for row ``t`` must be computed from data
up to ``t - h``. A VaR that saw ``y_t`` (a contemporaneous forecast) gives an
implausibly low hit rate: that is the symptom to look for. Build forecasts with
an explicit ``shift(h).over(entity)``.

**Panels.** Every test is column-wise, so ``N`` entities are ``N`` columns.
Per-entity p-values belong in
:func:`~panelary.validation.benjamini_yekutieli`, which is valid under
arbitrary cross-sectional dependence.
"""

from __future__ import annotations

import math
import warnings
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import polars as pl

from panelary._internal._special import norm_sf
from panelary.validation._results import EvaluationResult, evaluation_table

__all__ = [
    "PredictiveSpec",
    "acerbi_szekely_test",
    "christoffersen_test",
    "dynamic_quantile_test",
    "exceedances",
    "fz0_loss",
    "kupiec_test",
    "qlike_loss",
    "var_backtest",
]

#: Relative tolerance under which a simulated statistic ties the observed one.
#: Ties count against the model (conservative), and the tolerance absorbs
#: last-bit differences between the vectorised observed and null evaluations.
_TIE_RTOL = 1e-10
#: Cells per simulation chunk (about 32 MB of float64).
_SIM_CHUNK_CELLS = 4_000_000
#: Rows of the minlike rule: pmf(k) <= pmf(x) * (1 + 1e-7) is "as extreme".
_MINLIKE_RERR = 1.0 + 1e-7
#: Acerbi–Székely (2014) Z2 5% thresholds reported as stable across t3..t7 at
#: T = 250, alpha = 2.5% (published values, not re-derived here).
_Z2_THRESHOLD_5PCT = (-0.82, -0.70)

_DQ_ASYMPTOTIC_NOTE = (
    "asymptotic DQ is oversized when exceptions are rare (measured 0.093 at a "
    "nominal 0.05 for T=250, alpha=1%); use null='mc' for an exact-size test"
)
_ES_TESTS = ("z1", "z2")
_DEFAULT_TESTS = ("kupiec", "christoffersen", "dq", "z2")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _as_2d(a: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 1-D or 2-D, got {arr.ndim}-D.")
    return arr


def _broadcast_cols(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Broadcast ``(T, 1)`` / ``(T, M)`` arrays to one ``(T, M)`` shape."""
    shapes = {k: v.shape for k, v in arrays.items()}
    t = {s[0] for s in shapes.values()}
    if len(t) != 1:
        raise ValueError(f"row counts differ: {shapes}.")
    cols = {s[1] for s in shapes.values()} - {1}
    if len(cols) > 1:
        raise ValueError(f"column counts differ: {shapes}.")
    m = cols.pop() if cols else 1
    n = t.pop()
    return tuple(np.broadcast_to(v, (n, m)) for v in arrays.values())


def _model_names(names: Sequence[str] | None, m: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"model_{j}" for j in range(m))
    labels = tuple(str(n) for n in names)
    if len(labels) != m:
        raise ValueError(f"`names` has {len(labels)} entries for {m} models.")
    return labels


def _check_rate(level: float, *, tail: bool) -> float:
    a = float(level)
    if not (0.0 < a < 1.0):
        raise ValueError(
            f"`level` is the expected exception probability alpha and must lie "
            f"in (0, 1), got {level}."
        )
    if tail and a >= 0.5:
        raise ValueError(
            f"`level` is the tail probability alpha (0.01 for a 99% VaR), got "
            f"{level}; did you pass the confidence level?"
        )
    return a


def _hits_2d(hits: np.ndarray | Sequence[float]) -> np.ndarray:
    h = _as_2d(hits, "hits")
    fin = h[np.isfinite(h)]
    if np.any((fin != 0.0) & (fin != 1.0)):
        raise ValueError("`hits` must contain only 0, 1 or NaN (see `exceedances`).")
    if np.any(np.isinf(h)):
        raise ValueError("`hits` must not contain inf.")
    return h


def _rate_note(a: float) -> list[str]:
    if a > 0.5:
        return [
            f"level={a} > 0.5: `level` is the expected exception probability, "
            "not the confidence level"
        ]
    return []


def _chi2_sf(x: np.ndarray, df: int) -> np.ndarray:
    """Chi-square upper tail for integer ``df`` (closed forms for 1 and 2)."""
    xv = np.asarray(x, dtype=np.float64)
    if df == 1:
        return 2.0 * np.asarray(norm_sf(np.sqrt(np.maximum(xv, 0.0))), dtype=float)
    if df == 2:
        return np.exp(-0.5 * np.maximum(xv, 0.0))
    from panelary.econ._common import chi2_sf  # noqa: PLC0415  (no import cycle)

    return np.asarray(chi2_sf(xv, float(df)), dtype=np.float64)


def _mc_pvalue(
    null: np.ndarray, observed: np.ndarray, *, upper: bool = True
) -> np.ndarray:
    """``(1 + #{null as or more extreme}) / (1 + B)``; ties count as extreme."""
    srt = np.sort(null[np.isfinite(null)])
    b = srt.shape[0]
    obs = np.asarray(observed, dtype=np.float64)
    tol = _TIE_RTOL * np.maximum(1.0, np.abs(obs))
    if upper:
        count = b - np.searchsorted(srt, obs - tol, side="left")
    else:
        count = np.searchsorted(srt, obs + tol, side="right")
    p = (1.0 + count) / (1.0 + b)
    return np.where(np.isfinite(obs), p, np.nan)


def _model_rng(seed: int, name: str) -> np.random.Generator:
    """Per-model generator, keyed by name (not by position; trap T11)."""
    return np.random.default_rng([int(seed), zlib.crc32(name.encode("utf-8"))])


def _group_rng(seed: int, *keys: bytes) -> np.random.Generator:
    """Generator for a null shared by every model with the same key."""
    return np.random.default_rng([int(seed), *(zlib.crc32(k) for k in keys)])


def _xlogy_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a * log(a / (a + b))`` with ``0 log 0 = 0``."""
    tot = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), 1.0)
        return np.where(a > 0, a * np.log(np.where(a > 0, ratio, 1.0)), 0.0)


def _lr_pof(x: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    """Kupiec's proportion-of-failures likelihood ratio."""
    l1 = _xlogy_ratio(x, n - x) + _xlogy_ratio(n - x, x)
    l0 = x * math.log(alpha) + (n - x) * math.log1p(-alpha)
    return np.maximum(2.0 * (l1 - l0), 0.0)


def _lr_ind(
    n00: np.ndarray, n01: np.ndarray, n10: np.ndarray, n11: np.ndarray
) -> np.ndarray:
    """Christoffersen's first-order Markov independence likelihood ratio."""
    l1 = (
        _xlogy_ratio(n00, n01)
        + _xlogy_ratio(n01, n00)
        + _xlogy_ratio(n10, n11)
        + _xlogy_ratio(n11, n10)
    )
    c0 = n00 + n10
    c1 = n01 + n11
    l0 = _xlogy_ratio(c0, c1) + _xlogy_ratio(c1, c0)
    return np.maximum(2.0 * (l1 - l0), 0.0)


# --------------------------------------------------------------------------- #
# Hits
# --------------------------------------------------------------------------- #
def exceedances(
    y: np.ndarray | Sequence[float],
    *,
    lower: np.ndarray | Sequence[float] | None = None,
    upper: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Exceedance indicators of realisations against forecast bounds.

    A hit is ``y < lower`` or ``y > upper`` (strict). With only ``lower`` this
    is the VaR hit sequence in the quantile convention. Pass ``lower=-var`` for
    loss-convention VaR. With both bounds it is the miss sequence of an interval
    forecast, such as a conformal interval.

    Parameters
    ----------
    y : array-like of shape (T,) or (T, M)
        Realisations.
    lower, upper : array-like of shape (T,) or (T, M), optional
        Forecast bounds, known at ``t - h``. At least one is required.

    Returns
    -------
    numpy.ndarray of shape (T, M)
        ``1.0`` for a hit and ``0.0`` otherwise. ``NaN`` where ``y`` or a
        supplied bound is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> exceedances(np.array([-0.03, 0.01, np.nan]), lower=np.full(3, -0.02)).ravel()
    array([ 1.,  0., nan])
    """
    if lower is None and upper is None:
        raise ValueError("pass `lower`, `upper`, or both.")
    arrays: dict[str, np.ndarray] = {"y": _as_2d(y, "y")}
    if lower is not None:
        arrays["lower"] = _as_2d(lower, "lower")
    if upper is not None:
        arrays["upper"] = _as_2d(upper, "upper")
    b = dict(zip(arrays, _broadcast_cols(arrays), strict=True))
    yv = b["y"]
    valid = np.isfinite(yv)
    hit = np.zeros(yv.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        if "lower" in b:
            valid &= np.isfinite(b["lower"])
            hit |= yv < b["lower"]
        if "upper" in b:
            valid &= np.isfinite(b["upper"])
            hit |= yv > b["upper"]
    out = hit.astype(np.float64)
    out[~valid] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Kupiec
# --------------------------------------------------------------------------- #
def _binom_pmf(n: int, alpha: float) -> np.ndarray:
    """Binomial(n, alpha) pmf on ``0..n``, by the ratio recurrence from the mode.

    Relative error is a few ulps near the mode. That is two orders of magnitude
    better than ``exp(lgamma(...))``, whose ``lgamma(n + 1)`` term alone carries
    about ``n * 2e-15`` of error.
    """
    mode = min(int(math.floor((n + 1) * alpha)), n)
    rho = alpha / (1.0 - alpha)
    k = np.arange(n + 1, dtype=np.float64)
    r = np.empty(n + 1)
    r[mode] = 1.0
    if mode < n:
        j = k[mode:n]
        r[mode + 1 :] = np.cumprod((n - j) / (j + 1.0) * rho)
    if mode > 0:
        j = k[mode:0:-1]
        r[mode - 1 :: -1] = np.cumprod(j / ((n - j + 1.0) * rho))
    return r / r.sum()


def _kupiec_exact(
    x: np.ndarray, n: np.ndarray, alpha: float, alternative: str
) -> np.ndarray:
    """Exact binomial p-values, one pmf per distinct ``n`` shared across models."""
    p = np.full(x.shape, np.nan)
    for nn in np.unique(n[n > 0]):
        sel = n == nn
        pmf = _binom_pmf(int(nn), alpha)
        xi = x[sel].astype(np.int64)
        if alternative == "greater":
            tail = np.cumsum(pmf[::-1])[::-1]  # P(X >= k), summed from the far tail
            p[sel] = tail[xi]
        elif alternative == "less":
            p[sel] = np.cumsum(pmf)[xi]
        else:
            srt = np.sort(pmf)
            cs = np.cumsum(srt)
            idx = np.searchsorted(srt, pmf[xi] * _MINLIKE_RERR, side="right")
            p[sel] = cs[idx - 1]
    return np.minimum(p, 1.0)


def kupiec_test(
    hits: np.ndarray | Sequence[float],
    *,
    level: float,
    pvalue: Literal["exact", "asymptotic"] = "exact",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Kupiec (1995) unconditional-coverage test, exact binomial by default.

    Statistical evidence about the calibration of a risk forecast; not a
    regulatory determination.

    Under H0 the exception count of each model is Binomial(``T_eff``,
    ``alpha``). The two-sided exact p-value uses the "minlike" rule, which is
    also scipy's ``binomtest`` rule: it sums ``pmf(k)`` over every ``k`` with
    ``pmf(k) <= pmf(x) (1 + 1e-7)``. One pmf is computed per distinct
    ``T_eff`` and shared by all models. The exact test is conservative because
    the count is discrete: its measured size is 0.042 at a nominal 0.05, T = 250,
    VaR 1%. The asymptotic chi-square LR (size 0.097 in the same setting) is
    kept in ``details``.

    Parameters
    ----------
    hits : array-like of shape (T,) or (T, M)
        Exceedance indicators in ``{0, 1, NaN}`` (see :func:`exceedances`).
    level : float
        Expected exception probability ``alpha`` (0.01 for a 99% VaR).
    pvalue : {"exact", "asymptotic"}, default "exact"
        ``"asymptotic"`` refers the LR to chi-square(1). It is two-sided only.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        ``"greater"`` asks whether there are too many exceptions (risk
        underestimated). ``"less"`` asks whether there are too few.
    names : sequence of str, optional
        Model labels (default ``model_0``, ...).

    Returns
    -------
    EvaluationResult
        ``estimate`` is the hit rate. ``statistic`` is the exception count
        (exact) or the LR (asymptotic). ``details`` holds ``exceptions``,
        ``expected_exceptions``, ``lr_pof`` and ``pvalue_asymptotic``.

    References
    ----------
    Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement
    models. *J. Derivatives* 3(2), 73–84.
    """
    if pvalue not in ("exact", "asymptotic"):
        raise ValueError(f"`pvalue` must be 'exact' or 'asymptotic', got {pvalue!r}.")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            "`alternative` must be 'two-sided', 'greater' or 'less', "
            f"got {alternative!r}."
        )
    if pvalue == "asymptotic" and alternative != "two-sided":
        raise ValueError("the asymptotic LR test is two-sided only.")
    h = _hits_2d(hits)
    a = _check_rate(level, tail=False)
    labels = _model_names(names, h.shape[1])
    valid = np.isfinite(h)
    n = valid.sum(axis=0).astype(np.float64)
    x = np.where(valid, h, 0.0).sum(axis=0)
    notes = _rate_note(a)
    empty = n == 0
    if empty.any():
        notes.append(f"{int(empty.sum())} model(s) have no finite hits: NaN")
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(empty, np.nan, x / np.where(empty, 1.0, n))
    lr = np.where(empty, np.nan, _lr_pof(x, n, a))
    p_asym = _chi2_sf(lr, 1)
    if pvalue == "exact":
        p = _kupiec_exact(x, n, a, alternative)
        stat = np.where(empty, np.nan, x)
        reference = "binomial-exact"
    else:
        p = p_asym
        stat = lr
        reference = "chi2(1)"
    return EvaluationResult(
        test="kupiec",
        names=labels,
        estimate=rate,
        statistic=stat,
        pvalue=np.where(empty, np.nan, p),
        reference=reference,
        alternative=alternative,
        n_obs=n.astype(np.int64),
        details={
            "exceptions": x,
            "expected_exceptions": n * a,
            "lr_pof": lr,
            "pvalue_asymptotic": p_asym,
        },
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Christoffersen
# --------------------------------------------------------------------------- #
def _transition_counts(h: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(n00, n01, n10, n11)`` over consecutive rows where both are finite."""
    pair = valid[1:] & valid[:-1]
    prev = np.where(valid[:-1], h[:-1], 0.0) == 1.0
    cur = np.where(valid[1:], h[1:], 0.0) == 1.0
    n11 = (pair & prev & cur).sum(axis=0)
    n10 = (pair & prev & ~cur).sum(axis=0)
    n01 = (pair & ~prev & cur).sum(axis=0)
    n00 = pair.sum(axis=0) - n11 - n10 - n01
    return tuple(c.astype(np.float64) for c in (n00, n01, n10, n11))


def _christoffersen_null(
    mask: np.ndarray, alpha: float, kind: str, n_sims: int, seed: int
) -> np.ndarray:
    """Statistic under i.i.d. Bernoulli(alpha) hits on the finite rows of ``mask``."""
    pos = np.flatnonzero(mask)
    t_eff = pos.shape[0]
    pair_ok = np.diff(pos) == 1
    rng = _group_rng(
        seed,
        np.packbits(mask).tobytes() + np.int64(mask.size).tobytes(),
        np.float64(alpha).tobytes(),
    )
    out = np.empty(n_sims)
    step = max(1, _SIM_CHUNK_CELLS // max(t_eff, 1))
    for start in range(0, n_sims, step):
        stop = min(n_sims, start + step)
        s = rng.random((stop - start, t_eff)) < alpha
        prev, cur = s[:, :-1], s[:, 1:]
        n11 = (prev & cur & pair_ok).sum(axis=1).astype(np.float64)
        n10 = (prev & ~cur & pair_ok).sum(axis=1).astype(np.float64)
        n01 = (~prev & cur & pair_ok).sum(axis=1).astype(np.float64)
        n00 = pair_ok.sum() - n11 - n10 - n01
        stat = _lr_ind(n00, n01, n10, n11)
        if kind == "conditional_coverage":
            x = s.sum(axis=1).astype(np.float64)
            stat = stat + _lr_pof(x, np.full_like(x, float(t_eff)), alpha)
        else:
            # Condition on the event the observed model must satisfy to be
            # tested: at least one exception and one non-exception in the pairs.
            untestable = (n01 + n11 == 0) | (n00 + n10 == 0)
            stat = np.where(untestable, np.nan, stat)
        out[start:stop] = stat
    return out


def christoffersen_test(
    hits: np.ndarray | Sequence[float],
    *,
    level: float,
    kind: Literal["conditional_coverage", "independence"] = "conditional_coverage",
    null: Literal["mc", "asymptotic"] = "mc",
    n_sims: int = 9999,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Christoffersen (1998) independence / conditional-coverage test.

    Statistical evidence about the calibration of a risk forecast; not a
    regulatory determination.

    Transition counts come from consecutive rows where both hits are finite.
    ``LR_ind`` compares a first-order Markov chain with an i.i.d. sequence, and
    ``LR_cc = LR_uc + LR_ind`` also tests the exception rate against ``alpha``.

    With ``null="mc"`` (the default) this is Dufour's (2006) exact Monte Carlo
    test. H0 (i.i.d. Bernoulli(``alpha``) hits) is fully specified, so
    ``n_sims`` paths are simulated once per distinct missing-value pattern and
    shared by every model with that pattern. The p-value is
    ``(1 + #{null >= observed}) / (1 + B)``. For ``kind="independence"`` the
    null is conditioned on the event the observed series must satisfy to be
    testable: simulated paths with no exception (or nothing but exceptions) in
    their consecutive pairs are dropped, and ``B`` counts the rest. Ties in this
    discrete statistic count as ``>=``, which is conservative. The asymptotic chi-square
    p-value is kept in ``details``, but it is badly undersized: measured 0.013
    at a nominal 0.05 for ``LR_ind`` at T = 250, VaR 1%.

    Parameters
    ----------
    hits : array-like of shape (T,) or (T, M)
        Exceedance indicators in ``{0, 1, NaN}``.
    level : float
        Expected exception probability ``alpha``.
    kind : {"conditional_coverage", "independence"}, default "conditional_coverage"
        The statistic.
    null : {"mc", "asymptotic"}, default "mc"
        Exact Monte Carlo null or the chi-square(2 or 1) approximation.
    n_sims : int, default 9999
        Monte Carlo paths.
    seed : int, default 0
        The null of each missing-value pattern draws from
        ``default_rng([seed, crc32(pattern), crc32(alpha)])``. Adding or
        reordering models therefore never changes another model's p-value.
    names : sequence of str, optional
        Model labels.

    Returns
    -------
    EvaluationResult
        ``estimate`` is the hit rate (conditional coverage) or
        ``pi11 - pi01`` (independence). ``details`` holds ``lr_uc``, ``lr_ind``,
        ``pvalue_asymptotic``, the transition counts and ``pi01`` / ``pi11``.
        A model with no exceptions (or only exceptions) cannot be tested for
        independence: it gets NaN and a warning.

    References
    ----------
    Christoffersen, P. F. (1998). Evaluating interval forecasts. *IER* 39(4),
    841–862.

    Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters.
    *J. Econometrics* 133(2), 443–477.
    """
    if kind not in ("conditional_coverage", "independence"):
        raise ValueError(
            f"`kind` must be 'conditional_coverage' or 'independence', got {kind!r}."
        )
    if null not in ("mc", "asymptotic"):
        raise ValueError(f"`null` must be 'mc' or 'asymptotic', got {null!r}.")
    if null == "mc" and int(n_sims) < 1:
        raise ValueError(f"`n_sims` must be >= 1, got {n_sims}.")
    h = _hits_2d(hits)
    a = _check_rate(level, tail=False)
    labels = _model_names(names, h.shape[1])
    valid = np.isfinite(h)
    n = valid.sum(axis=0).astype(np.float64)
    x = np.where(valid, h, 0.0).sum(axis=0)
    n00, n01, n10, n11 = _transition_counts(h, valid)
    lr_uc = _lr_pof(x, n, a)
    lr_ind = _lr_ind(n00, n01, n10, n11)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        rate = x / n
    notes = _rate_note(a)
    too_short = n < 2
    if kind == "independence":
        stat = lr_ind
        df = 1
        estimate = pi11 - pi01
        bad = too_short | (n01 + n11 == 0) | (n00 + n10 == 0)
        if bad.any():
            notes.append(
                f"{int(bad.sum())} model(s) have no exceptions (or nothing but "
                "exceptions) in consecutive pairs: independence is untestable, NaN"
            )
    else:
        stat = lr_uc + lr_ind
        df = 2
        estimate = rate
        bad = too_short
        if bad.any():
            notes.append(f"{int(bad.sum())} model(s) have fewer than 2 hits: NaN")
    stat = np.where(bad, np.nan, stat)
    p_asym = _chi2_sf(stat, df)
    if null == "mc":
        p = np.full(stat.shape, np.nan)
        masks, inverse = np.unique(valid.T, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        for g in range(masks.shape[0]):
            sel = (inverse == g) & ~bad
            if not sel.any():
                continue
            null_stat = _christoffersen_null(masks[g], a, kind, int(n_sims), seed)
            p[sel] = _mc_pvalue(null_stat, stat[sel])
        reference = f"mc-exact(B={int(n_sims)})"
    else:
        p = p_asym
        reference = f"chi2({df})"
    return EvaluationResult(
        test=f"christoffersen_{'cc' if kind == 'conditional_coverage' else 'ind'}",
        names=labels,
        estimate=estimate,
        statistic=stat,
        pvalue=p,
        reference=reference,
        alternative="two-sided",
        n_obs=n.astype(np.int64),
        n_resamples=int(n_sims) if null == "mc" else None,
        seed=int(seed) if null == "mc" else None,
        details={
            "lr_uc": lr_uc,
            "lr_ind": lr_ind,
            "pvalue_asymptotic": p_asym,
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "pi01": pi01,
            "pi11": pi11,
        },
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Engle–Manganelli DQ
# --------------------------------------------------------------------------- #
def _projection_stat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched ``y' X (X'X)^+ X' y`` and ``rank(X)``.

    ``x`` is ``(B, k, n)`` (regressors in rows) and ``y`` is ``(B, n)``. Zero
    observations are ignored. The Gram matrix is column-equilibrated before its
    eigendecomposition, so a constant regressor (collinear with the intercept)
    or an all-zero lag column reduces the rank instead of breaking the solve.
    """
    gram = x @ np.swapaxes(x, 1, 2)
    xy = (x @ y[..., None])[..., 0]
    d = np.sqrt(np.einsum("bii->bi", gram))
    inv_d = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    gs = gram * inv_d[:, :, None] * inv_d[:, None, :]
    bs = xy * inv_d
    evals, evecs = np.linalg.eigh(gs)
    top = evals[:, -1:]
    keep = evals > np.maximum(top, 0.0) * 1e-10
    proj = (np.swapaxes(evecs, 1, 2) @ bs[..., None])[..., 0]
    safe = np.where(keep, evals, 1.0)
    q = np.where(keep, proj * proj / safe, 0.0).sum(axis=1)
    return q, keep.sum(axis=1)


def _dq_design(
    h: np.ndarray, v: np.ndarray, n_lags: int, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design ``(M, k, n)``, response ``(M, n)`` and row mask ``(M, n)``.

    ``h`` and ``v`` are ``(M, T)`` (one row per model, time along the row).
    Observations with any non-finite input are zeroed.
    """
    m, t = h.shape
    rows = t - n_lags
    k = n_lags + 2
    design = np.empty((m, k, rows))
    ok = np.isfinite(h[:, n_lags:]) & np.isfinite(v[:, n_lags:])
    design[:, 0, :] = 1.0
    for lag in range(1, n_lags + 1):
        lagged = h[:, n_lags - lag : t - lag]
        ok &= np.isfinite(lagged)
        design[:, lag, :] = lagged
    design[:, k - 1, :] = v[:, n_lags:]
    resp = h[:, n_lags:] - alpha
    design = np.where(ok[:, None, :], design, 0.0)
    resp = np.where(ok, resp, 0.0)
    return design, resp, ok


def dynamic_quantile_test(
    hits: np.ndarray | Sequence[float],
    var: np.ndarray | Sequence[float],
    *,
    level: float,
    n_lags: int = 4,
    null: Literal["asymptotic", "mc"] = "asymptotic",
    n_sims: int = 999,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Engle–Manganelli (2004) dynamic quantile (DQ) test.

    Statistical evidence about the calibration of a risk forecast; not a
    regulatory determination.

    ``Hit_t - alpha`` is regressed on ``X_t = [1, Hit_{t-1}, ..., Hit_{t-L},
    VaR_t]``, and ``DQ = b' X'X b / (alpha (1 - alpha))`` is referred to
    chi-square(``L + 2``). Under correct conditional coverage no regressor
    predicts the hit. The statistic is the squared norm of the projection of
    ``Hit - alpha`` onto the column space of ``X``, so it does not depend on the
    sign convention of ``var``. When the design is rank-deficient (a constant
    VaR, or no exceptions to lag), the projection uses the pseudo-inverse and the
    degrees of freedom drop to the rank. That is recorded in
    ``details["df"]`` and in a warning.

    **The chi-square reference is oversized when exceptions are rare.** The
    lagged-hit regressors are sparse 0/1 columns, so the statistic is far from
    its limit. Measured sizes at a nominal 0.05 (i.i.d. hits, 4,000
    replications): 0.068 at T = 250, alpha = 5%; 0.093 at T = 250, alpha = 1%;
    0.108 at T = 1000, alpha = 1%; 0.085 at T = 5000, alpha = 1%. ``null="mc"``
    holds its size (0.054 and 0.051 at T = 250 for alpha = 5% and 1%). Every
    asymptotic result carries this caveat as a warning.

    Parameters
    ----------
    hits : array-like of shape (T,) or (T, M)
        Exceedance indicators in ``{0, 1, NaN}``.
    var : array-like of shape (T,) or (T, M)
        The VaR forecasts behind the hits, in either sign convention.
    level : float
        Expected exception probability ``alpha``.
    n_lags : int, default 4
        Lagged hits in the design.
    null : {"asymptotic", "mc"}, default "asymptotic"
        ``"mc"`` simulates i.i.d. Bernoulli(``alpha``) hits with ``VaR_t`` held
        fixed. It runs per model and is opt-in: it costs ``n_sims`` small
        regressions per model.
    n_sims : int, default 999
        Monte Carlo replicates for ``null="mc"``.
    seed : int, default 0
        Model ``j`` draws from ``default_rng([seed, crc32(names[j])])``.
    names : sequence of str, optional
        Model labels.

    Returns
    -------
    EvaluationResult
        ``estimate`` is the hit rate over the regression rows, and
        ``statistic`` is DQ. ``details`` holds ``df`` and ``pvalue_asymptotic``.

    References
    ----------
    Engle, R. F. & Manganelli, S. (2004). CAViaR: conditional autoregressive
    value at risk by regression quantiles. *JBES* 22(4), 367–381.
    """
    if null not in ("asymptotic", "mc"):
        raise ValueError(f"`null` must be 'asymptotic' or 'mc', got {null!r}.")
    n_lags = int(n_lags)
    if n_lags < 0:
        raise ValueError(f"`n_lags` must be >= 0, got {n_lags}.")
    a = _check_rate(level, tail=False)
    hv, vv = _broadcast_cols({"hits": _hits_2d(hits), "var": _as_2d(var, "var")})
    h = np.array(hv, dtype=np.float64)
    v = np.array(vv, dtype=np.float64)
    t, m = h.shape
    labels = _model_names(names, m)
    k = n_lags + 2
    if t <= n_lags + k:
        raise ValueError(f"need more than {n_lags + k} rows for n_lags={n_lags}.")
    notes = _rate_note(a)
    q = np.empty(m)
    rank = np.empty(m, dtype=np.int64)
    n_rows = np.empty(m, dtype=np.int64)
    hit_sum = np.empty(m)
    step = max(1, _SIM_CHUNK_CELLS // ((t - n_lags) * k))
    h_t = np.ascontiguousarray(h.T)
    v_t = np.ascontiguousarray(v.T)
    for c0 in range(0, m, step):
        c1 = min(m, c0 + step)
        design, resp, okt = _dq_design(h_t[c0:c1], v_t[c0:c1], n_lags, a)
        q[c0:c1], rank[c0:c1] = _projection_stat(design, resp)
        n_rows[c0:c1] = okt.sum(axis=1)
        hit_sum[c0:c1] = np.where(okt, resp + a, 0.0).sum(axis=1)
    scale = a * (1.0 - a)
    stat = q / scale
    bad = n_rows <= k
    if bad.any():
        notes.append(f"{int(bad.sum())} model(s) have <= {k} usable rows: NaN")
    stat = np.where(bad, np.nan, stat)
    if np.any((rank < k) & ~bad):
        notes.append(
            "rank-deficient DQ design for some models (constant VaR or no "
            "exceptions to lag): degrees of freedom reduced to the rank "
            "(details['df'])"
        )
    p_asym = np.full(m, np.nan)
    for r in np.unique(rank[~bad]):
        sel = (rank == r) & ~bad
        p_asym[sel] = _chi2_sf(stat[sel], int(r))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = hit_sum / n_rows
    if null == "mc":
        p = np.full(m, np.nan)
        b_total = int(n_sims)
        if b_total < 1:
            raise ValueError(f"`n_sims` must be >= 1, got {n_sims}.")
        for j in np.flatnonzero(~bad):
            p[j] = _dq_mc_pvalue(
                h[:, j], v[:, j], n_lags, a, stat[j], b_total, seed, labels[j]
            )
        reference = f"mc-exact(B={b_total})"
    else:
        p = p_asym
        reference = f"chi2({k})"
        notes.append(_DQ_ASYMPTOTIC_NOTE)
    return EvaluationResult(
        test="dynamic_quantile",
        names=labels,
        estimate=rate,
        statistic=stat,
        pvalue=p,
        reference=reference,
        alternative="two-sided",
        n_obs=n_rows.astype(np.int64),
        n_resamples=int(n_sims) if null == "mc" else None,
        seed=int(seed) if null == "mc" else None,
        details={"df": rank.astype(np.float64), "pvalue_asymptotic": p_asym},
        warnings=tuple(notes),
    )


def _dq_mc_pvalue(
    h: np.ndarray,
    v: np.ndarray,
    n_lags: int,
    alpha: float,
    observed: float,
    n_sims: int,
    seed: int,
    name: str,
) -> float:
    rng = _model_rng(seed, name)
    finite = np.isfinite(h)
    t = h.shape[0]
    null = np.empty(n_sims)
    step = max(1, _SIM_CHUNK_CELLS // max(t * (n_lags + 2), 1))
    for start in range(0, n_sims, step):
        stop = min(n_sims, start + step)
        sims = (rng.random((stop - start, t)) < alpha).astype(np.float64)
        sims[:, ~finite] = np.nan
        vb = np.broadcast_to(v, (stop - start, t))
        design, resp, _ = _dq_design(sims, vb, n_lags, alpha)
        q, _ = _projection_stat(design, resp)
        null[start:stop] = q / (alpha * (1.0 - alpha))
    return float(_mc_pvalue(null, np.array([observed]))[0])


# --------------------------------------------------------------------------- #
# Predictive distributions and the Acerbi–Székely ES backtest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class PredictiveSpec:
    """The predictive distribution an ES backtest needs to simulate H0.

    Returns are ``loc + scale * eps`` with standardised innovations ``eps``,
    where ``loc`` and ``scale`` are known at ``t - h``.

    Attributes
    ----------
    family : {"normal", "student_t", "fhs"}
        ``"student_t"`` uses unit-variance Student-t innovations (``df > 2``).
        ``"fhs"`` is filtered historical simulation: innovations are resampled
        from ``residuals``.
    loc, scale : numpy.ndarray
        ``(T,)`` or ``(T, M)`` location and scale forecasts (``scale > 0``).
    df : float, optional
        Degrees of freedom for ``"student_t"``.
    residuals : numpy.ndarray, optional
        Standardised residual pool for ``"fhs"``: ``(R,)`` shared, or ``(R, M)``
        per model. NaNs are dropped.
    """

    family: Literal["normal", "student_t", "fhs"]
    loc: np.ndarray
    scale: np.ndarray
    df: float | None = None
    residuals: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        if self.family not in ("normal", "student_t", "fhs"):
            raise ValueError(
                f"`family` must be 'normal', 'student_t' or 'fhs', got {self.family!r}."
            )
        object.__setattr__(self, "loc", _as_2d(self.loc, "loc"))
        object.__setattr__(self, "scale", _as_2d(self.scale, "scale"))
        if self.loc.shape[0] != self.scale.shape[0]:
            raise ValueError("`loc` and `scale` must have the same number of rows.")
        fin = self.scale[np.isfinite(self.scale)]
        if np.any(fin <= 0):
            raise ValueError("`scale` must be strictly positive.")
        if self.family == "student_t" and (self.df is None or not float(self.df) > 2.0):
            raise ValueError("family='student_t' needs `df` > 2 (unit variance).")
        if self.family == "fhs":
            if self.residuals is None:
                raise ValueError("family='fhs' needs a `residuals` pool.")
            res = _as_2d(self.residuals, "residuals")
            if np.isfinite(res).sum(axis=0).min() < 2:
                raise ValueError("every `residuals` column needs >= 2 finite values.")
            object.__setattr__(self, "residuals", res)

    def _column(self, arr: np.ndarray, j: int) -> np.ndarray:
        return arr[:, 0] if arr.shape[1] == 1 else arr[:, j]

    def _draw(
        self, rng: np.random.Generator, j: int, rows: np.ndarray, n_draws: int
    ) -> np.ndarray:
        """``(n_draws, len(rows))`` simulated returns for model ``j``."""
        tv = rows.shape[0]
        if self.family == "normal":
            eps = rng.standard_normal((n_draws, tv))
        elif self.family == "student_t":
            nu = float(self.df)  # type: ignore[arg-type]
            eps = rng.standard_t(nu, (n_draws, tv)) * math.sqrt((nu - 2.0) / nu)
        else:
            assert self.residuals is not None
            pool = self._column(self.residuals, j)
            pool = pool[np.isfinite(pool)]
            eps = pool[rng.integers(0, pool.shape[0], size=(n_draws, tv))]
        loc = self._column(self.loc, j)[rows]
        scale = self._column(self.scale, j)[rows]
        return loc + scale * eps


def _to_quantile_convention(
    var: np.ndarray, es: np.ndarray | None, var_convention: str
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    if var_convention not in ("loss", "quantile"):
        raise ValueError(
            "`var_convention` must be 'loss' (VaR > 0, hit is y < -VaR) or "
            f"'quantile' (VaR < 0, hit is y < VaR), got {var_convention!r}."
        )
    sign = -1.0 if var_convention == "loss" else 1.0
    v = sign * var
    e = None if es is None else sign * es
    notes: list[str] = []
    fin = v[np.isfinite(v)]
    if fin.size:
        wrong = float(np.mean(fin >= 0.0))
        if wrong > 0.5:
            other = "quantile" if var_convention == "loss" else "loss"
            raise ValueError(
                f"{wrong:.0%} of the VaR values have the wrong sign for "
                f"var_convention={var_convention!r}; did you mean "
                f"var_convention={other!r}?"
            )
        if wrong > 0:
            notes.append(
                f"{wrong:.1%} of VaR values have the wrong sign for "
                f"var_convention={var_convention!r}"
            )
    if e is not None:
        with np.errstate(invalid="ignore"):
            worse = np.isfinite(e) & np.isfinite(v) & (e > v)
        if worse.any():
            notes.append(
                f"{int(worse.sum())} rows have ES less extreme than VaR "
                "(ES must lie at or beyond VaR)"
            )
    return v, e, notes


def _as_z(
    x: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float, kind: str
) -> np.ndarray:
    """Z1 or Z2 along the last axis (quantile convention, ``e < 0``)."""
    hit = x < v
    terms = np.where(hit, x / e, 0.0).sum(axis=-1)
    if kind == "z2":
        return 1.0 - terms / (x.shape[-1] * alpha)
    n_hit = hit.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_hit > 0, 1.0 - terms / n_hit, np.nan)


def acerbi_szekely_test(
    returns: np.ndarray | Sequence[float],
    var: np.ndarray | Sequence[float],
    es: np.ndarray | Sequence[float],
    *,
    level: float,
    var_convention: Literal["loss", "quantile"],
    kind: Literal["z1", "z2"] = "z2",
    predictive: PredictiveSpec | None = None,
    n_sims: int = 2000,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Acerbi–Székely (2014) expected-shortfall backtests Z1 and Z2.

    Statistical evidence about the calibration of a risk forecast; not a
    regulatory determination.

    In the quantile convention (``ES <= VaR < 0``) with ``I_t = 1{X_t < VaR_t}``:

    * ``Z2 = 1 - sum_t X_t I_t / (T alpha ES_t)``, which tests VaR and ES
      jointly;
    * ``Z1 = 1 - sum_t X_t I_t / (N_T ES_t)``, with ``N_T = sum_t I_t``, which
      tests ES given the exceptions.

    Both have expectation 0 under a correct forecast. Negative values mean risk
    was underestimated, so the test is one-sided (``alternative="less"``).

    The p-value needs the predictive distribution. With ``predictive`` the null
    is simulated: ``n_sims`` return paths are drawn from it per model, ``Z`` is
    recomputed with the same VaR/ES forecasts, and the p-value is
    ``(1 + #{Z* <= Z}) / (1 + n_sims)``. Without it the result carries only the
    statistic, no p-value, and the published "stable" Z2 5% threshold range in
    ``details`` (about -0.70 to -0.82 across t3..t7 at T = 250, alpha = 2.5%;
    Acerbi & Székely 2014, not re-derived here).

    Parameters
    ----------
    returns : array-like of shape (T,) or (T, M)
        Realised returns.
    var, es : array-like of shape (T,) or (T, M)
        VaR and ES forecasts known at ``t - h``, in ``var_convention``.
    level : float
        Tail probability ``alpha`` (0.025 for a 97.5% ES).
    var_convention : {"loss", "quantile"}
        Required. ``"loss"``: positive VaR/ES. ``"quantile"``: negative return
        quantile and tail mean. A majority of wrong-signed VaRs raises.
    kind : {"z2", "z1"}, default "z2"
        The statistic.
    predictive : PredictiveSpec, optional
        The forecast's predictive distribution, used to simulate H0.
    n_sims : int, default 2000
        Simulated paths per model.
    seed : int, default 0
        Model ``j`` draws from ``default_rng([seed, crc32(names[j])])``.
    names : sequence of str, optional
        Model labels.

    Returns
    -------
    EvaluationResult
        ``estimate`` and ``statistic`` are ``Z``. ``details`` holds
        ``exceptions``, ``expected_exceptions`` and, without ``predictive``,
        ``z2_threshold_5pct_low`` / ``_high``. For Z1, simulated paths with no
        exception are dropped from the null; ``details["n_null"]`` gives the
        count used.

    References
    ----------
    Acerbi, C. & Székely, B. (2014). Back-testing expected shortfall. *Risk*
    27(11), 76–81.
    """
    if kind not in _ES_TESTS:
        raise ValueError(f"`kind` must be 'z1' or 'z2', got {kind!r}.")
    a = _check_rate(level, tail=True)
    x, v_raw, e_raw = _broadcast_cols(
        {
            "returns": _as_2d(returns, "returns"),
            "var": _as_2d(var, "var"),
            "es": _as_2d(es, "es"),
        }
    )
    v, e, notes = _to_quantile_convention(
        np.asarray(v_raw), np.asarray(e_raw), var_convention
    )
    assert e is not None
    t, m = x.shape
    labels = _model_names(names, m)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(x) & np.isfinite(v) & np.isfinite(e) & (e < 0)
    if predictive is not None:
        if predictive.loc.shape[0] != t:
            raise ValueError(
                f"`predictive` has {predictive.loc.shape[0]} rows, returns have {t}."
            )
        for spec_arr in (predictive.loc, predictive.scale, predictive.residuals):
            if spec_arr is not None and spec_arr.shape[1] not in (1, m):
                raise ValueError(
                    "`predictive` arrays must have 1 column or one per model."
                )
        if int(n_sims) < 1:
            raise ValueError(f"`n_sims` must be >= 1, got {n_sims}.")
        valid &= np.isfinite(predictive.loc) & np.isfinite(predictive.scale)
    n_valid = valid.sum(axis=0)
    z = np.full(m, np.nan)
    n_hit = np.zeros(m)
    for j in range(m):
        rows = valid[:, j]
        if rows.any():
            z[j] = _as_z(x[rows, j], v[rows, j], e[rows, j], a, kind)
            n_hit[j] = float(np.sum(x[rows, j] < v[rows, j]))
    if kind == "z1" and np.any((n_hit == 0) & (n_valid > 0)):
        notes.append("Z1 is undefined without exceptions: NaN for those models")
    details: dict[str, np.ndarray] = {
        "exceptions": n_hit,
        "expected_exceptions": n_valid * a,
    }
    p = np.full(m, np.nan)
    if predictive is not None:
        n_null = np.zeros(m)
        for j in range(m):
            if not np.isfinite(z[j]):
                continue
            idx = np.flatnonzero(valid[:, j])
            null = _as_null(
                predictive,
                j,
                idx,
                v[idx, j],
                e[idx, j],
                a,
                kind,
                int(n_sims),
                seed,
                labels[j],
            )
            n_null[j] = np.isfinite(null).sum()
            p[j] = float(_mc_pvalue(null, np.array([z[j]]), upper=False)[0])
        details["n_null"] = n_null
        reference = f"mc-exact(B={int(n_sims)},{predictive.family})"
    else:
        notes.append(
            "no PredictiveSpec: no p-value; compare Z2 with the published 5% "
            "threshold range in details (Acerbi & Szekely 2014)"
        )
        details["z2_threshold_5pct_low"] = np.asarray(_Z2_THRESHOLD_5PCT[0])
        details["z2_threshold_5pct_high"] = np.asarray(_Z2_THRESHOLD_5PCT[1])
        reference = "none (statistic only)"
    return EvaluationResult(
        test=f"acerbi_szekely_{kind}",
        names=labels,
        estimate=z,
        statistic=z,
        pvalue=p,
        reference=reference,
        alternative="less",
        n_obs=n_valid.astype(np.int64),
        n_resamples=int(n_sims) if predictive is not None else None,
        seed=int(seed) if predictive is not None else None,
        details=details,
        warnings=tuple(notes),
    )


def _as_null(
    spec: PredictiveSpec,
    j: int,
    rows: np.ndarray,
    v: np.ndarray,
    e: np.ndarray,
    alpha: float,
    kind: str,
    n_sims: int,
    seed: int,
    name: str,
) -> np.ndarray:
    rng = _model_rng(seed, name)
    out = np.empty(n_sims)
    step = max(1, _SIM_CHUNK_CELLS // max(rows.shape[0], 1))
    for start in range(0, n_sims, step):
        stop = min(n_sims, start + step)
        sims = spec._draw(rng, j, rows, stop - start)
        out[start:stop] = _as_z(sims, v, e, alpha, kind)
    return out


# --------------------------------------------------------------------------- #
# Consistent losses for comparative backtesting
# --------------------------------------------------------------------------- #
def fz0_loss(
    returns: np.ndarray | Sequence[float],
    var: np.ndarray | Sequence[float],
    es: np.ndarray | Sequence[float],
    *,
    level: float,
) -> np.ndarray:
    """FZ0 joint VaR/ES loss of Patton, Ziegel & Chen (2019), eq. (6).

    ``L = -1/(alpha e) 1{Y <= v} (v - Y) + v / e + log(-e) - 1``, in the quantile
    convention (``e <= v < 0``). It is the unique Fissler–Ziegel loss whose
    differences are homogeneous of degree 0: rescaling returns and forecasts by
    ``c > 0`` shifts every loss by ``log c``, so loss differences between
    models are unit-free. Feed the per-period losses to Diebold–Mariano or the
    model confidence set for comparative backtesting (Nolde & Ziegel, 2017).

    Parameters
    ----------
    returns, var, es : array-like
        Realised returns and the VaR/ES forecasts (quantile convention),
        broadcastable to a common shape.
    level : float
        Tail probability ``alpha``.

    Returns
    -------
    numpy.ndarray
        Per-period loss (lower is better). It is NaN, with a ``UserWarning``,
        where ``v >= 0``, ``e >= 0`` or ``e > v``. It is NaN without a warning
        where an input is NaN.

    References
    ----------
    Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic semiparametric
    models for expected shortfall (and Value-at-Risk). *J. Econometrics*
    211(2), 388–413.

    Examples
    --------
    >>> round(float(fz0_loss(-1.0, -1.64, -2.06, level=0.05)), 6)
    0.518828
    """
    a = _check_rate(level, tail=True)
    y, v, e = np.broadcast_arrays(
        np.asarray(returns, dtype=np.float64),
        np.asarray(var, dtype=np.float64),
        np.asarray(es, dtype=np.float64),
    )
    with np.errstate(invalid="ignore"):
        bad = (v >= 0) | (e >= 0) | (e > v)
    if np.any(bad):
        warnings.warn(
            f"fz0_loss: {int(np.sum(bad))} rows violate e <= v < 0 (quantile "
            "convention); their loss is NaN",
            UserWarning,
            stacklevel=2,
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        ind = (y <= v).astype(np.float64)
        loss = -ind * (v - y) / (a * e) + v / e + np.log(-e) - 1.0
    return np.where(bad, np.nan, loss)


def qlike_loss(
    realized_variance: np.ndarray | Sequence[float],
    forecast_variance: np.ndarray | Sequence[float],
) -> np.ndarray:
    """QLIKE volatility loss of Patton (2011): ``RV/h - log(RV/h) - 1``.

    QLIKE is robust to noise in the volatility proxy: ranking forecasts by
    expected QLIKE against an unbiased noisy proxy (such as squared returns)
    gives the same ranking as against the true variance. Squared error on log
    variance does not have this property. The loss is 0 at ``h = RV``.

    Parameters
    ----------
    realized_variance : array-like
        Variance proxy (for example realised variance or squared returns),
        ``> 0``.
    forecast_variance : array-like
        Variance forecasts ``h > 0``, broadcastable to ``realized_variance``.

    Returns
    -------
    numpy.ndarray
        Per-period loss. It is NaN, with a ``UserWarning``, where either input is
        not strictly positive.

    References
    ----------
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
    volatility proxies. *J. Econometrics* 160(1), 246–256.
    """
    rv, h = np.broadcast_arrays(
        np.asarray(realized_variance, dtype=np.float64),
        np.asarray(forecast_variance, dtype=np.float64),
    )
    with np.errstate(invalid="ignore"):
        bad = (rv <= 0) | (h <= 0)
    if np.any(bad):
        warnings.warn(
            f"qlike_loss: {int(np.sum(bad))} rows have a non-positive variance; "
            "their loss is NaN",
            UserWarning,
            stacklevel=2,
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = rv / h
        loss = ratio - np.log(ratio) - 1.0
    return np.where(bad, np.nan, loss)


# --------------------------------------------------------------------------- #
# The battery
# --------------------------------------------------------------------------- #
def var_backtest(
    returns: np.ndarray | Sequence[float],
    var: np.ndarray | Sequence[float],
    es: np.ndarray | Sequence[float] | None = None,
    *,
    level: float,
    var_convention: Literal["loss", "quantile"],
    tests: Sequence[str] = _DEFAULT_TESTS,
    predictive: PredictiveSpec | None = None,
    seed: int = 0,
    names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Run a VaR/ES backtest battery and return one evidence table.

    Statistical evidence about the calibration of a risk forecast; not a
    regulatory determination. No pass/fail zone is emitted.

    Parameters
    ----------
    returns : array-like of shape (T,) or (T, M)
        Realised returns.
    var : array-like of shape (T,) or (T, M)
        VaR forecasts, each known at ``t - h``. Build them with an explicit
        ``shift(h).over(entity)``. A forecast that saw ``y_t`` shows up as an
        implausibly low hit rate (trap T1).
    es : array-like of shape (T,) or (T, M), optional
        ES forecasts. The ES tests (``"z1"``, ``"z2"``) run only when given.
    level : float
        Tail probability ``alpha`` (0.01 for a 99% VaR).
    var_convention : {"loss", "quantile"}
        **Required**, with no default: sign bugs are the most common VaR
        backtest error. ``"loss"``: VaR > 0 and a hit is ``y < -VaR``.
        ``"quantile"``: VaR < 0 and a hit is ``y < VaR``.
    tests : sequence of str, default ("kupiec", "christoffersen", "dq", "z2")
        Any of ``"kupiec"`` (exact two-sided), ``"christoffersen"``
        (conditional coverage, exact MC), ``"dq"`` (Monte Carlo null with 999
        paths per model, because the asymptotic DQ is oversized; for many
        models call :func:`dynamic_quantile_test` directly), ``"z1"`` and
        ``"z2"``. With the default and ``es=None``, ``"z2"`` is skipped. Asking
        for an ES test explicitly without ``es`` raises.
    predictive : PredictiveSpec, optional
        Enables Z1/Z2 p-values.
    seed : int, default 0
        Seed for the Monte Carlo nulls.
    names : sequence of str, optional
        Model labels.

    Returns
    -------
    polars.DataFrame
        The concatenated :data:`~panelary.validation._results.EVALUATION_SCHEMA`
        rows, one per (test, model).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = rng.standard_normal(500) * 0.01
    >>> var99 = np.full(500, 0.0233)          # loss convention: a positive number
    >>> table = var_backtest(r, var99, level=0.01, var_convention="loss")
    >>> table["test"].to_list()
    ['kupiec', 'christoffersen_cc', 'dynamic_quantile']
    """
    a = _check_rate(level, tail=True)
    requested = tuple(tests)
    unknown = set(requested) - {"kupiec", "christoffersen", "dq", *_ES_TESTS}
    if unknown:
        raise ValueError(f"unknown tests {sorted(unknown)}.")
    if es is None and any(tst in _ES_TESTS for tst in requested):
        if requested != _DEFAULT_TESTS:
            raise ValueError("the ES tests 'z1'/'z2' need `es`.")
        requested = tuple(tst for tst in requested if tst not in _ES_TESTS)
    y = _as_2d(returns, "returns")
    v_in = _as_2d(var, "var")
    v, _, notes = _to_quantile_convention(v_in, None, var_convention)
    hits = exceedances(y, lower=v)
    m = hits.shape[1]
    labels = _model_names(names, m)
    results: list[EvaluationResult] = []
    for tst in requested:
        if tst == "kupiec":
            res = kupiec_test(hits, level=a, names=labels)
        elif tst == "christoffersen":
            res = christoffersen_test(hits, level=a, seed=seed, names=labels)
        elif tst == "dq":
            vv = np.broadcast_to(v, hits.shape)
            res = dynamic_quantile_test(
                hits, vv, level=a, null="mc", seed=seed, names=labels
            )
        else:
            assert es is not None
            res = acerbi_szekely_test(
                y,
                v,
                np.asarray(es, dtype=np.float64)
                * (-1.0 if var_convention == "loss" else 1.0),
                level=a,
                var_convention="quantile",
                kind=tst,  # type: ignore[arg-type]
                predictive=predictive,
                seed=seed,
                names=labels,
            )
        if notes:
            res = _with_notes(res, notes)
        results.append(res)
    return evaluation_table(results)


def _with_notes(res: EvaluationResult, notes: list[str]) -> EvaluationResult:
    merged = tuple(dict.fromkeys((*notes, *res.warnings)))
    return replace(res, warnings=merged)
