"""Intraday -> daily realized measures: one row per ``(entity, session)``.

:func:`intraday_realized_measures` aggregates intraday prices (or returns) into
per-session variance, semivariance, quarticity, jump-robust and noise-robust
measures in one ``group_by(entity, session)``. Every value uses only that
session's own observations and is stamped at the session close, so the output
is contemporaneous: it may be used as a feature from the close onwards, and
never broadcast back onto the session's own intraday rows (that is a look-ahead
by construction, which is why the registry scope is ``"window"``).

The per-session expression builders at the top of this module are shared with
:func:`panelary.econ.features.daily_realized_measures`, which is their subset.

Measures (``M`` returns ``r_i`` in the session)
-----------------------------------------------
======================  =========================================================
``rv``                  ``sum r_i^2``
``rv_ss``               ``(1/K) sum_{i>=K} (p_i - p_{i-K})^2`` -- the mean of the
                        ``K`` offset-subsampled RVs (``K = subsample``)
``bv``                  ``(pi/2) (M/(M-1)) sum |r_i||r_{i-1}|``
``medrv``               ``pi/(6 - 4 sqrt 3 + pi) (M/(M-2)) sum med3(|r|)^2``
``minrv``               ``pi/(pi - 2) (M/(M-1)) sum min(|r_i|, |r_{i+1}|)^2``
``rs_pos`` / ``rs_neg``  ``sum r_i^2 1{r_i > 0}`` / ``1{r_i < 0}``
``sjv``                 ``rs_pos - rs_neg``
``rq``                  ``(M/3) sum r_i^4``
``tpq``                 ``M mu_{4/3}^{-3} (M/(M-2)) sum |r_i r_{i-1} r_{i-2}|^{4/3}``
``jump_z``              Huang-Tauchen ratio-max statistic
``n_obs``               ``M``
``jump`` / ``rel_jump``  ``max(rv - bv, 0)`` and ``jump / rv`` (opt-in)
``jump_sig`` / ``cont``  ``1{z > Phi^-1(alpha)} max(rv - bv, 0)`` and ``rv - jump_sig``
``medrq``               ADS (2012) median quarticity (opt-in)
``log_rv_var``          ``(2/3) sum r^4 / rv^2``: delta-method variance of ``log rv``
``log_rv_var_tpq``      ``2 tpq / (M bv^2)``: its jump-robust version
``rk``                  Parzen realized kernel, BNHLS bandwidth (opt-in)
``tsrv``                two-scales RV (opt-in)
``pav``                 pre-averaged RV (opt-in)
======================  =========================================================

Verified constants
------------------
Checked by quadrature against the order statistics of ``|Z|`` (2026-09-29):
``E[med3(|Z|)^2] = (6 - 4 sqrt 3 + pi)/pi``, ``E[min(|Z1|,|Z2|)^2] = (pi - 2)/pi``,
``E[med3(|Z|)^4] = (9 pi + 72 - 52 sqrt 3)/(3 pi)`` and
``mu_{4/3} = 2^{2/3} Gamma(7/6)/Gamma(1/2)``, each to 1e-15. The noise-robust
constants are documented in :mod:`panelary._internal._realized_kernel`.

References
----------
Barndorff-Nielsen & Shephard (2004, 2006); Huang & Tauchen (2005);
Barndorff-Nielsen, Kinnebrock & Shephard (2010); Patton & Sheppard (2015);
Andersen, Dobrev & Schaumburg (2012); Zhang, Mykland & Ait-Sahalia (2005);
Barndorff-Nielsen, Hansen, Lunde & Shephard (2008, 2009); Jacod, Li, Mykland,
Podolskij & Vetter (2009).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from panelary._internal._realized_kernel import (
    BNHLS_DENSE_RETURNS,
    BNHLS_SPARSE_RETURNS,
    bnhls_bandwidth_expr,
    parzen_expr,
    preaverage_psi_expr,
)
from panelary._internal._special import lgamma, norm_ppf

__all__ = ["intraday_realized_measures"]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: ``pi / 2 = 1 / mu_1^2``: scales bipower variation to integrated variance.
_MU1_SQ_INV = np.pi / 2.0
#: ``1 / E[med3(|Z|)^2]`` (MedRV, ADS 2012).
_MEDRV_C = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)
#: ``1 / E[min(|Z1|, |Z2|)^2]`` (MinRV, ADS 2012).
_MINRV_C = np.pi / (np.pi - 2.0)
#: ``1 / E[med3(|Z|)^4]`` (MedRQ, ADS 2012).
_MEDRQ_C = 3.0 * np.pi / (9.0 * np.pi + 72.0 - 52.0 * np.sqrt(3.0))
#: ``mu_{4/3} = E|Z|^{4/3}``.
_MU43 = 2.0 ** (2.0 / 3.0) * math.exp(float(lgamma(7.0 / 6.0)) - float(lgamma(0.5)))
#: ``mu_{4/3}^{-3}``: scales tri-power quarticity to integrated quarticity.
_TPQ_C = _MU43**-3
#: ``(pi/2)^2 + pi - 5``: asymptotic variance factor of ``(RV - BV)/RV``.
_JUMP_THETA = np.pi**2 / 4.0 + np.pi - 5.0

# --------------------------------------------------------------------------- #
# Measure catalogue
# --------------------------------------------------------------------------- #
_DEFAULT_MEASURES: tuple[str, ...] = (
    "rv",
    "rv_ss",
    "bv",
    "medrv",
    "minrv",
    "rs_pos",
    "rs_neg",
    "sjv",
    "rq",
    "tpq",
    "jump_z",
    "n_obs",
)
_OPT_IN_MEASURES: tuple[str, ...] = (
    "jump",
    "rel_jump",
    "jump_sig",
    "cont",
    "medrq",
    "log_rv_var",
    "log_rv_var_tpq",
    "rk",
    "tsrv",
    "pav",
)
_ALL_MEASURES: frozenset[str] = frozenset(_DEFAULT_MEASURES + _OPT_IN_MEASURES)

#: Diagnostic columns emitted right after their measure.
_DIAGNOSTICS: dict[str, tuple[str, ...]] = {
    "rk": ("rk_h", "rk_capped"),
    "tsrv": ("tsrv_k",),
    "pav": ("pav_k",),
}

#: Raw per-session aggregates each measure needs.
_NEEDS: dict[str, frozenset[str]] = {
    "rv": frozenset({"rv"}),
    "rv_ss": frozenset({"rvss"}),
    "bv": frozenset({"bp"}),
    "medrv": frozenset({"med2"}),
    "minrv": frozenset({"min2"}),
    "rs_pos": frozenset({"rsp"}),
    "rs_neg": frozenset({"rsn"}),
    "sjv": frozenset({"rsp", "rsn"}),
    "rq": frozenset({"r4"}),
    "tpq": frozenset({"tp"}),
    "jump_z": frozenset({"rv", "bp", "tp"}),
    "n_obs": frozenset(),
    "jump": frozenset({"rv", "bp"}),
    "rel_jump": frozenset({"rv", "bp"}),
    "jump_sig": frozenset({"rv", "bp", "tp"}),
    "cont": frozenset({"rv", "bp", "tp"}),
    "medrq": frozenset({"med4"}),
    "log_rv_var": frozenset({"rv", "r4"}),
    "log_rv_var_tpq": frozenset({"bp", "tp"}),
    "rk": frozenset({"gamma"}),
    "tsrv": frozenset({"rv"}),
    "pav": frozenset({"rv"}),
}

# Private column names (double underscore: never collide with user columns).
_R, _P, _X, _N = "__rm_r", "__rm_p", "__rm_x", "__rm_n"


# --------------------------------------------------------------------------- #
# Shared per-session expression builders (also used by daily_realized_measures)
# --------------------------------------------------------------------------- #
def _rv_expr(r: pl.Expr) -> pl.Expr:
    """``sum r^2`` over the group."""
    return (r**2).sum()


def _bipower_sum_expr(r: pl.Expr) -> pl.Expr:
    """``sum |r_i||r_{i-1}|`` over the group (unscaled bipower sum)."""
    absr = r.abs()
    return (absr * absr.shift(1)).sum()


def _bv_from_sum_expr(bp: pl.Expr, n: pl.Expr) -> pl.Expr:
    """Scale a bipower sum over ``n`` returns: ``(pi/2) (n/(n-1)) bp``."""
    return _MU1_SQ_INV * (n / (n - 1)) * bp


def _jump_expr(rv: pl.Expr, bv: pl.Expr) -> pl.Expr:
    """Non-negative jump ``max(rv - bv, 0)``."""
    return pl.max_horizontal(rv - bv, pl.lit(0.0))


def _rel_jump_expr(jump: pl.Expr, rv: pl.Expr) -> pl.Expr:
    """``jump / rv`` where ``rv > 0``, else null."""
    return pl.when(rv > 0).then(jump / rv).otherwise(None)


# --------------------------------------------------------------------------- #
# Stage builders
# --------------------------------------------------------------------------- #
def _median3(a: pl.Expr, b: pl.Expr, c: pl.Expr) -> pl.Expr:
    """Exact median of three: ``max(min(a, b), min(max(a, b), c))``.

    Used instead of ``a + b + c - max - min``, which is only equal up to
    rounding. The horizontal min/max skip nulls, so callers mask the edges.
    """
    return pl.max_horizontal(
        pl.min_horizontal(a, b), pl.min_horizontal(pl.max_horizontal(a, b), c)
    )


def _dense_step(n: pl.Expr) -> pl.Expr:
    """BNHLS dense-grid step: about 195 returns per session, at least 1."""
    return (
        (n.cast(pl.Float64) / BNHLS_DENSE_RETURNS)
        .round()
        .clip(lower_bound=1.0)
        .cast(pl.Int64)
    )


def _sparse_step(n: pl.Expr) -> pl.Expr:
    """BNHLS sparse-grid step: about 19.5 returns per session, at least 1."""
    return (
        (n.cast(pl.Float64) / BNHLS_SPARSE_RETURNS)
        .round()
        .clip(lower_bound=1.0)
        .cast(pl.Int64)
    )


def _k_step_increments(p: pl.Expr, k: pl.Expr | int) -> pl.Expr:
    """``p_i - p_{i-k}`` for ``i >= k`` within the group, with ``p_0 = 0``.

    ``p`` holds the log price relative to the session's first price at the
    return rows ``1..M``, so the increment ending at return ``k`` reaches back
    to the implicit ``p_0 = 0``. ``k`` may be a per-group expression.
    """
    return (p - p.shift(k).fill_null(0.0)).slice(k - 1)


def _gamma_aggs(kernel_max_lags: int) -> list[pl.Expr]:
    """Autocovariances ``gamma_h = sum_j x_j x_{j-h}``, ``h = 0..H_max``."""
    x = pl.col(_X)
    return [
        (x * x.shift(h)).sum().alias(f"__rm_g{h}") for h in range(kernel_max_lags + 1)
    ]


def _raw_aggs(
    needs: set[str],
    *,
    subsample: int,
    tsrv_fixed_k: int | None,
) -> list[pl.Expr]:
    """The per-session aggregates for one ``group_by(entity, session)``."""
    r = pl.col(_R)
    a = r.abs()
    aggs: list[pl.Expr] = [pl.len().alias(_N)]
    if "rv" in needs:
        aggs.append(_rv_expr(r).alias("__rm_rv"))
    if "bp" in needs:
        aggs.append(_bipower_sum_expr(r).alias("__rm_bp"))
    if "rvss" in needs:
        aggs.append((r.rolling_sum(subsample) ** 2).sum().alias("__rm_rvss"))
    if "rsp" in needs:
        aggs.append((r**2).filter(r > 0).sum().alias("__rm_rsp"))
    if "rsn" in needs:
        aggs.append((r**2).filter(r < 0).sum().alias("__rm_rsn"))
    if "r4" in needs:
        aggs.append((r**4).sum().alias("__rm_r4"))
    if "tp" in needs:
        a43 = a.pow(4.0 / 3.0)
        aggs.append((a43 * a43.shift(1) * a43.shift(2)).sum().alias("__rm_tp"))
    if needs & {"med2", "med4"}:
        prev, nxt = a.shift(1), a.shift(-1)
        med = pl.when(prev.is_not_null() & nxt.is_not_null()).then(
            _median3(prev, a, nxt)
        )
        if "med2" in needs:
            aggs.append((med**2).sum().alias("__rm_med2"))
        if "med4" in needs:
            aggs.append((med**4).sum().alias("__rm_med4"))
    if "min2" in needs:
        nxt = a.shift(-1)
        low = pl.when(nxt.is_not_null()).then(pl.min_horizontal(a, nxt))
        aggs.append((low**2).sum().alias("__rm_min2"))
    if "noise" in needs:
        p = pl.col(_P)
        q = _dense_step(pl.len())
        kiv = _sparse_step(pl.len())
        dq = _k_step_increments(p, q)
        ds = _k_step_increments(p, kiv)
        aggs.extend(
            [
                q.alias("__rm_q"),
                (dq * dq).sum().alias("__rm_dq2"),
                (dq != 0).sum().alias("__rm_nnzq"),
                kiv.alias("__rm_kiv"),
                (ds * ds).sum().alias("__rm_sp2"),
                (ds**4).sum().alias("__rm_sp4"),
            ]
        )
    if tsrv_fixed_k is not None:
        aggs.append((r.rolling_sum(tsrv_fixed_k) ** 2).sum().alias("__rm_tsk"))
    return aggs


def _noise_exprs(omega2: str) -> tuple[pl.Expr, pl.Expr]:
    """``(omega^2_hat, IV_hat)`` of BNHLS (2009) from the dense and sparse grids.

    ``IV_hat`` is the sparse-grid (about 20-minute) subsampled RV. ``"bnhls"``:
    ``omega^2 = RV_dense / (2 n_dense)`` with every dense offset pooled;
    ``"debiased"``: ``max(RV_dense - IV_hat, 0) / (2 n_dense)``, which removes
    the ``IV / (2 n_dense)`` term that dominates on low-noise bar data.
    """
    q = pl.col("__rm_q").cast(pl.Float64)
    nnz = pl.col("__rm_nnzq").cast(pl.Float64)
    iv = pl.col("__rm_sp2") / pl.col("__rm_kiv").cast(pl.Float64)
    rv_dense = pl.col("__rm_dq2") / q
    n_dense = nnz / q
    if omega2 == "bnhls":
        om2 = rv_dense / (2.0 * n_dense)
    else:
        om2 = pl.max_horizontal(rv_dense - iv, pl.lit(0.0)) / (2.0 * n_dense)
    ok = (nnz > 0) & (iv > 0)
    return pl.when(ok).then(om2), pl.when(ok).then(iv)


def _kernel_bandwidth_exprs(
    bandwidth: Literal["bnhls"] | int, kernel_max_lags: int, omega2: str
) -> tuple[pl.Expr, pl.Expr]:
    """``(H, capped)``: the per-session kernel bandwidth and its cap flag."""
    if bandwidth != "bnhls":
        return pl.lit(float(bandwidth)), pl.lit(value=False)
    n = pl.col(_N).cast(pl.Float64)
    om2, iv = _noise_exprs(omega2)
    h_star = bnhls_bandwidth_expr(om2 / iv, n - 2.0)
    h_raw = h_star.ceil()
    # min_horizontal skips nulls, so the validity mask is taken on `h_star`.
    h = pl.when(h_star.is_finite()).then(
        pl.min_horizontal(h_raw, pl.lit(float(kernel_max_lags)))
    )
    capped = pl.when(h_star.is_finite()).then(h_raw > kernel_max_lags)
    return h, capped


def _kernel_expr(kernel_max_lags: int) -> pl.Expr:
    """``gamma_0 + 2 sum_h k(h / (H + 1)) gamma_h`` from a materialised ``H``.

    ``H`` must already be a column: inlined, its expression would be repeated
    inside every one of the ``kernel_max_lags`` Parzen weights.
    """
    h = pl.col("__rm_h")
    terms = [
        parzen_expr(pl.lit(float(lag)) / (h + 1.0)) * pl.col(f"__rm_g{lag}")
        for lag in range(1, kernel_max_lags + 1)
    ]
    return pl.when(h.is_not_null()).then(
        pl.col("__rm_g0") + 2.0 * pl.sum_horizontal(terms)
    )


def _tsrv_value(rv_k: pl.Expr, k: pl.Expr) -> pl.Expr:
    """Small-sample-adjusted TSRV of ZMA (2005) from ``RV_ss(K)`` and ``RV``."""
    n = pl.col(_N).cast(pl.Float64)
    kf = k.cast(pl.Float64)
    ratio = ((n - kf + 1.0) / kf) / n
    value = (rv_k - ratio * pl.col("__rm_rv")) / (1.0 - ratio)
    return pl.when((kf >= 2.0) & (kf < n)).then(value)


def _tsrv_auto_k(omega2: str) -> pl.Expr:
    """ZMA (2005) ``K* = ceil(c* n^{2/3})``, ``c* = (12 omega^4 / IQ)^{1/3}``.

    ``omega^2`` as for the kernel bandwidth; ``IQ`` is the realized quarticity of
    the sparse (about 20-minute) grid, because on the dense grid ``sum r^4`` is
    dominated by the noise. Clipped to ``[2, max(2, n // 2)]``.
    """
    n = pl.col(_N).cast(pl.Float64)
    kiv = pl.col("__rm_kiv").cast(pl.Float64)
    om2, _iv = _noise_exprs(omega2)
    iq = n / (3.0 * kiv * kiv) * pl.col("__rm_sp4")
    c_star = (12.0 * om2 * om2 / iq).pow(1.0 / 3.0)
    k_raw = (c_star * n.pow(2.0 / 3.0)).ceil()
    upper = pl.max_horizontal(pl.lit(2.0), (n / 2.0).floor())
    # min/max_horizontal skip nulls, so the validity mask is taken on `k_raw`.
    k = pl.min_horizontal(pl.max_horizontal(k_raw, pl.lit(2.0)), upper)
    return pl.when((iq > 0) & k_raw.is_finite()).then(k).cast(pl.Int64)


def _pav_frame(clean: pl.DataFrame, keys: list[str], theta: float) -> pl.DataFrame:
    """``sum Ybar_i^2`` per session, with ``k = 2 ceil(theta sqrt(n) / 2)``.

    The pre-averaged return is a triangular filter of the returns, and a
    triangle is two box filters convolved: ``k Ybar = S_w(S_w(r))`` with
    ``S_w`` the trailing ``w = k/2`` sum. Two native rolling sums of the
    returns, so the cost is O(n) whatever ``k`` is. The window is a per-session
    constant, so sessions are processed in blocks of equal ``k`` (few distinct
    values in practice: ``k`` depends only on the session's return count).
    """
    n = pl.len().over(keys).cast(pl.Float64)
    k = (2.0 * (theta * n.sqrt() / 2.0).ceil()).cast(pl.Int64)
    tagged = clean.select([*keys, _R]).with_columns(k.alias("__rm_k"))
    blocks: list[pl.DataFrame] = []
    for part in tagged.partition_by("__rm_k", maintain_order=True):
        k_val = int(part.get_column("__rm_k")[0])
        half = max(1, k_val // 2)
        y = pl.col(_R).rolling_sum(half).rolling_sum(half)
        blocks.append(
            part.group_by(keys).agg(
                pl.col("__rm_k").first(), (y * y).sum().alias("__rm_yb2")
            )
        )
    if not blocks:
        return pl.DataFrame(
            schema={
                **dict(clean.select(keys).schema),
                "__rm_k": pl.Int64,
                "__rm_yb2": pl.Float64,
            }
        )
    return pl.concat(blocks)


def _pav_value() -> pl.Expr:
    """JLMPV (2009) pre-averaged realized variance with its noise correction.

    ``PAV = (n/(n-k+2)) (1/(k psi_2)) sum Ybar^2 - (psi_1 / (2 k^2 psi_2)) sum r^2``,
    from ``E[Ybar^2] = k psi_2 IV/n + (psi_1/k) omega^2`` and
    ``E[sum r^2] = IV + 2 n omega^2``.
    """
    n = pl.col(_N).cast(pl.Float64)
    k_col = pl.col("__rm_k")
    kf = k_col.cast(pl.Float64)
    psi1, psi2 = preaverage_psi_expr(k_col)
    ybar2 = pl.col("__rm_yb2") / (kf * kf)
    value = (n / (n - kf + 2.0)) / (kf * psi2) * ybar2 - psi1 / (
        2.0 * kf * kf * psi2
    ) * pl.col("__rm_rv")
    return pl.when(kf <= n).then(value)


def _stamp_expr(time: str, label: str, dtype: pl.DataType) -> pl.Expr:
    """The session-close timestamp: when the day's value becomes known.

    ``label="right"`` (bars stamped at their close): the last stamp.
    ``label="left"`` (bars stamped at their open): the last stamp plus the
    session's median bar spacing, i.e. the close of the last bar.
    """
    last = pl.col(time).max()
    if label == "right":
        return last
    stamp = last + pl.col(time).sort().diff().median()
    if dtype.is_integer():
        return stamp.round().cast(dtype)
    return stamp


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #
def intraday_realized_measures(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    entity: str,
    session: str,
    time: str,
    price: str | None = None,
    returns: str | None = None,
    measures: Sequence[str] = _DEFAULT_MEASURES,
    subsample: int = 5,
    jump_alpha: float = 0.999,
    kernel_bandwidth: Literal["bnhls"] | int = "bnhls",
    kernel_max_lags: int = 30,
    omega2: Literal["bnhls", "debiased"] = "bnhls",
    preaverage_theta: float = 1.0,
    tsrv_k: Literal["auto"] | int = "auto",
    min_obs: int = 10,
    label: Literal["right", "left"] = "right",
) -> pl.DataFrame:
    """Aggregate intraday prices or returns into per-session realized measures.

    One ``group_by(entity, session)`` computes every requested measure from that
    session's observations only. The first return of a session is taken from the
    session's first price, never from the previous session's close: overnight
    moves are a separate quantity. Each output row is stamped at the session
    close (column ``time``), the earliest moment its values are known.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        Long intraday frame: one row per tick or bar.
    entity, session, time : str
        Instrument key, session (trading-day) key, and intraday timestamp.
        Rows are ordered by ``time`` within each session (ties keep input order).
    price : str, optional
        Price column; returns are within-session log differences. Rows with a
        non-finite or non-positive price are dropped (the move across them is
        kept as one longer return). Exactly one of ``price`` / ``returns``.
    returns : str, optional
        Log-return column. Rows with a null or non-finite return are dropped,
        so their neighbours become adjacent.
    measures : sequence of str
        Which measures to emit, in this order. Defaults to the core battery
        ``("rv", "rv_ss", "bv", "medrv", "minrv", "rs_pos", "rs_neg", "sjv",
        "rq", "tpq", "jump_z", "n_obs")``. Opt-in: ``"jump"``, ``"rel_jump"``,
        ``"jump_sig"``, ``"cont"``, ``"medrq"``, ``"log_rv_var"``,
        ``"log_rv_var_tpq"``, ``"rk"`` (adds ``rk_h``, ``rk_capped``),
        ``"tsrv"`` (adds ``tsrv_k``) and ``"pav"`` (adds ``pav_k``).
    subsample : int, default 5
        ``K`` of ``rv_ss``. With 1-minute bars, ``K = 5`` averages the five
        offset 5-minute RVs, which Liu, Patton & Sheppard (2015) find hard to
        beat. It covers the ``M - K + 1`` complete ``K``-step returns, so its
        expectation is ``(M - K + 1)/M`` of the integrated variance.
    jump_alpha : float, default 0.999
        Significance level of the jump test behind ``jump_sig`` / ``cont``.
    kernel_bandwidth : "bnhls" or int, default "bnhls"
        Realized-kernel bandwidth ``H``. ``"bnhls"`` sets
        ``H = ceil(c* xi^{4/5} n^{3/5})`` per session from that session's own
        noise-to-signal ratio ``xi^2 = omega^2 / IV`` (BNHLS 2009); an int fixes
        ``H`` for every session.
    kernel_max_lags : int, default 30
        Autocovariances computed per session. A BNHLS bandwidth above it is
        capped and flagged in ``rk_capped``.
    omega2 : {"bnhls", "debiased"}, default "bnhls"
        Noise-variance estimate for the kernel bandwidth and the automatic TSRV
        ``K``. ``"bnhls"`` is ``RV_dense / (2 n_dense)`` on an about-2-minute
        grid; on low-noise bar data it is dominated by ``IV / (2 n_dense)``.
        ``"debiased"`` subtracts the sparse-grid IV first.
    preaverage_theta : float, default 1.0
        ``theta`` of the pre-averaging window ``k = 2 ceil(theta sqrt(M) / 2)``.
    tsrv_k : "auto" or int, default "auto"
        Slow-scale ``K`` of TSRV; ``"auto"`` uses ZMA's (2005) per-session optimum
        (see Notes).
    min_obs : int, default 10
        Sessions with fewer returns get null measures (``n_obs`` is still set).
    label : {"right", "left"}, default "right"
        Whether ``time`` stamps a bar's close (``"right"``) or its open
        (``"left"``, shifted to the close by the session's median bar spacing).

    Returns
    -------
    polars.DataFrame
        One row per ``(entity, session)`` present in ``df``, sorted by them,
        with ``time`` (the close stamp) and the requested measures. Undefined
        values are null.

    Notes
    -----
    **Leak safety.** Every value depends only on its own session. Use it at or
    after the session close (for example with an as-of join on ``time``); never
    join it back onto the same session's intraday rows.

    **Noise-robust measures.** ``rk`` is the non-flat-top Parzen kernel
    ``gamma_0 + 2 sum_{h=1}^H k(h/(H+1)) gamma_h`` on end-point-jittered
    returns (``m = 2``), which is non-negative. BNHLS (2009) estimate
    ``omega^2`` on an about-2-minute grid and ``IV`` on an about-20-minute grid;
    here those grids are about 195 and 19.5 returns per session (their values
    for a 390-minute session), so the rule needs no clock units. ``tsrv`` is
    ``(1 - nbar/n)^{-1} (RV_ss(K) - (nbar/n) RV)`` with ``nbar = (n - K + 1)/K``;
    negative values are kept. ``pav`` is JLMPV (2009) with
    ``g(x) = min(x, 1 - x)``.

    References
    ----------
    See the module docstring.
    """
    if (price is None) == (returns is None):
        raise ValueError("pass exactly one of `price` or `returns`.")
    requested = _validate_measures(measures)
    subsample = _positive_int(subsample, "subsample")
    kernel_max_lags = _positive_int(kernel_max_lags, "kernel_max_lags")
    min_obs = int(min_obs)
    if min_obs < 3:
        raise ValueError(f"`min_obs` must be >= 3, got {min_obs!r}.")
    if not 0.5 < float(jump_alpha) < 1.0:
        raise ValueError(f"`jump_alpha` must be in (0.5, 1), got {jump_alpha!r}.")
    if not float(preaverage_theta) > 0.0:
        raise ValueError(
            f"`preaverage_theta` must be positive, got {preaverage_theta!r}."
        )
    if label not in ("right", "left"):
        raise ValueError(f"`label` must be 'right' or 'left', got {label!r}.")
    if omega2 not in ("bnhls", "debiased"):
        raise ValueError(f"`omega2` must be 'bnhls' or 'debiased', got {omega2!r}.")
    bandwidth: Literal["bnhls"] | int
    if kernel_bandwidth == "bnhls":
        bandwidth = "bnhls"
    else:
        bandwidth = int(kernel_bandwidth)
        if not 0 <= bandwidth <= kernel_max_lags:
            raise ValueError(
                "an integer `kernel_bandwidth` must be in [0, kernel_max_lags] "
                f"= [0, {kernel_max_lags}], got {kernel_bandwidth!r}."
            )
    tsrv_fixed: int | None = None
    if tsrv_k != "auto":
        tsrv_fixed = int(tsrv_k)
        if tsrv_fixed < 2:
            raise ValueError(f"an integer `tsrv_k` must be >= 2, got {tsrv_k!r}.")

    lf = df.lazy()
    schema = lf.collect_schema()
    value_col = price if price is not None else returns
    assert value_col is not None  # for the type checker; guarded above
    for col in (entity, session, time, value_col):
        if col not in schema:
            raise ValueError(
                f"column {col!r} not found in frame; available: {list(schema)}."
            )
    if len({entity, session, time, value_col}) != 4:
        raise ValueError(
            "`entity`, `session`, `time` and the value column must differ."
        )
    time_dtype = schema[time]
    if not (time_dtype.is_numeric() or time_dtype.is_temporal()):
        raise ValueError(f"`time` must be numeric or temporal, got {time_dtype}.")
    keys = [entity, session]

    needs: set[str] = set()
    for name in requested:
        needs |= _NEEDS[name]
    use_rk = "rk" in requested
    use_tsrv = "tsrv" in requested
    use_pav = "pav" in requested
    if (use_rk and bandwidth == "bnhls") or (use_tsrv and tsrv_fixed is None):
        needs.add("noise")

    raw = lf.select(keys + [time, value_col])
    null_keys = (
        raw.select(pl.any_horizontal(pl.col(keys).is_null()).any()).collect().item()
    )
    if null_keys:
        raise ValueError("`entity` and `session` must not contain nulls.")
    sessions = raw.group_by(keys).agg(_stamp_expr(time, label, time_dtype).alias(time))

    clean = _clean_returns(
        raw,
        keys=keys,
        time=time,
        value=value_col,
        is_price=price is not None,
        with_levels="noise" in needs,
    )

    main = (
        clean.lazy()
        .group_by(keys)
        .agg(
            _raw_aggs(
                needs,
                subsample=subsample,
                tsrv_fixed_k=tsrv_fixed if use_tsrv else None,
            )
        )
    )
    out = sessions.join(main, on=keys, how="left").collect()

    if use_rk:
        gammas = (
            _jittered_returns(clean, keys)
            .group_by(keys)
            .agg(_gamma_aggs(kernel_max_lags))
        )
        # Sessions too short to jitter get gamma_h = 0 (an empty sum), and are
        # nulled by `min_obs` (>= 3) below; fill so the join keeps them defined.
        out = out.join(gammas, on=keys, how="left").with_columns(
            pl.col(f"__rm_g{h}").fill_null(0.0) for h in range(kernel_max_lags + 1)
        )
    if use_pav:
        pav = _pav_frame(clean, keys, float(preaverage_theta))
        out = out.join(pav, on=keys, how="left")
    if use_tsrv and tsrv_fixed is None:
        out = out.with_columns(_tsrv_auto_k(omega2).alias("__rm_tsrv_k"))
        with_k = out.select([*keys, "__rm_tsrv_k"]).filter(
            pl.col("__rm_tsrv_k").is_not_null()
        )
        k_expr = pl.col("__rm_tsrv_k").first()
        inc = _k_step_increments(pl.col(_P), k_expr)
        # `maintain_order="left"` keeps each session's rows in time order, which
        # the lagged increments below depend on.
        second = (
            clean.lazy()
            .join(with_k.lazy(), on=keys, how="inner", maintain_order="left")
            .group_by(keys)
            .agg((inc * inc).sum().alias("__rm_tsk"))
            .collect()
        )
        out = out.join(second, on=keys, how="left")
    elif use_tsrv:
        out = out.with_columns(pl.lit(tsrv_fixed, dtype=pl.Int64).alias("__rm_tsrv_k"))

    stages, columns = _final_columns(
        requested,
        bandwidth=bandwidth,
        kernel_max_lags=kernel_max_lags,
        omega2=omega2,
        subsample=subsample,
        jump_alpha=float(jump_alpha),
        min_obs=min_obs,
    )
    for stage in stages:
        out = out.with_columns(stage)
    return out.select([*keys, time, *columns]).sort(keys)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _positive_int(value: int, name: str) -> int:
    out = int(value)
    if out < 1:
        raise ValueError(f"`{name}` must be >= 1, got {value!r}.")
    return out


def _validate_measures(measures: Sequence[str]) -> list[str]:
    if isinstance(measures, str):
        raise TypeError("`measures` must be a sequence of names, not a string.")
    out = list(measures)
    unknown = sorted(set(out) - _ALL_MEASURES)
    if unknown:
        raise ValueError(
            f"unknown measure(s) {unknown}; choose from {sorted(_ALL_MEASURES)}."
        )
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate measure names in {out!r}.")
    if not out:
        raise ValueError("`measures` is empty.")
    return out


def _session_start(keys: list[str]) -> pl.Expr:
    """True on the first row of each session of a ``keys``-sorted frame.

    Row-local (a comparison with the previous row), so it costs no grouping and
    is exact; ``ne_missing`` makes the frame's first row a start.
    """
    return pl.any_horizontal([pl.col(k).ne_missing(pl.col(k).shift(1)) for k in keys])


def _session_end(keys: list[str]) -> pl.Expr:
    """True on the last row of each session of a ``keys``-sorted frame."""
    return pl.any_horizontal([pl.col(k).ne_missing(pl.col(k).shift(-1)) for k in keys])


def _clean_returns(
    raw: pl.LazyFrame,
    *,
    keys: list[str],
    time: str,
    value: str,
    is_price: bool,
    with_levels: bool,
) -> pl.DataFrame:
    """Sorted, null-free returns, and optionally the log-price levels.

    One row per return ``r_1 .. r_M`` of each session, in time order. ``_P`` is
    the log price relative to the session's first price at those rows, so the
    session's opening level ``p_0 = 0`` is implicit.
    """
    v = pl.col(value).cast(pl.Float64)
    if is_price:
        lp = pl.col("__rm_lp")
        start = _session_start(keys)
        frame = (
            raw.filter(v.is_finite() & (v > 0))
            .sort([*keys, time], maintain_order=True)
            .with_columns(v.log().alias("__rm_lp"))
            # The first price of a session has no return; its diff would reach
            # into the previous session, so it is nulled and dropped.
            .with_columns(pl.when(start).then(None).otherwise(lp.diff()).alias(_R))
        )
        if with_levels:
            opening = pl.when(start).then(lp).forward_fill()
            frame = frame.with_columns((lp - opening).alias(_P))
        frame = frame.filter(pl.col(_R).is_not_null()).drop("__rm_lp", value)
    else:
        frame = (
            raw.filter(v.is_finite())
            .sort([*keys, time], maintain_order=True)
            .with_columns(v.alias(_R))
            .drop(value)
        )
        if with_levels:
            frame = frame.with_columns(pl.col(_R).cum_sum().over(keys).alias(_P))
    return frame.collect()


def _jittered_returns(clean: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """End-point-jittered returns (BNHLS 2009, ``m = 2``), one row per value.

    From a session's returns ``r_1 .. r_M`` this keeps rows ``2 .. M-1`` with
    ``x = r_2 + r_1/2`` on the first kept row and ``x = r_{M-1} + r_M/2`` on the
    last, i.e. the returns between the averaged first two and last two prices.
    Null-free, so the autocovariance sums run on a dense column.
    """
    r = pl.col(_R)
    start, end = pl.col("__rm_start"), pl.col("__rm_end")
    # A row right after a start (before an end) is in the same session as the
    # start (end) row, so these plain shifts never cross a session boundary.
    after_start = start.shift(1).fill_null(value=False)
    before_end = end.shift(-1).fill_null(value=False)
    head = pl.when(after_start).then(0.5 * r.shift(1)).otherwise(0.0)
    tail = pl.when(before_end).then(0.5 * r.shift(-1)).otherwise(0.0)
    return (
        clean.lazy()
        .select([*keys, _R])
        .with_columns(
            _session_start(keys).alias("__rm_start"),
            _session_end(keys).alias("__rm_end"),
        )
        .with_columns((r + head + tail).alias(_X))
        .filter(~(start | end))
        .select([*keys, _X])
        .collect()
    )


def _final_columns(
    requested: list[str],
    *,
    bandwidth: Literal["bnhls"] | int,
    kernel_max_lags: int,
    omega2: str,
    subsample: int,
    jump_alpha: float,
    min_obs: int,
) -> tuple[list[list[pl.Expr]], list[pl.Expr]]:
    """Row-wise measures from the raw aggregates, masked by ``min_obs``.

    Returns ``(stages, columns)``: ``stages`` are applied in order with
    ``with_columns`` (quantities several measures share -- ``bv``, ``tpq``, the
    jump statistic, the kernel bandwidth -- are materialised once instead of
    being inlined into every expression that uses them), then ``columns`` are
    selected.
    """
    n = pl.col(_N)
    nf = n.cast(pl.Float64)
    rv = pl.col("__rm_rv")
    needs: set[str] = set()
    for name in requested:
        needs |= _NEEDS[name]

    first: list[pl.Expr] = []
    if "bp" in needs:
        first.append(
            pl.when(n > 1)
            .then(_bv_from_sum_expr(pl.col("__rm_bp"), n))
            .alias("__rm_bvv")
        )
    if "tp" in needs:
        first.append(
            pl.when(n > 2)
            .then(nf * _TPQ_C * (nf / (nf - 2.0)) * pl.col("__rm_tp"))
            .alias("__rm_tpqv")
        )
    if "rk" in requested:
        h, capped = _kernel_bandwidth_exprs(bandwidth, kernel_max_lags, omega2)
        first.extend([h.alias("__rm_h"), capped.alias("__rm_capped")])
    bv, tpq = pl.col("__rm_bvv"), pl.col("__rm_tpqv")
    second: list[pl.Expr] = []
    if {"bp", "tp", "rv"} <= needs:
        rel = (rv - bv) / rv
        scale = (
            _JUMP_THETA / nf * pl.max_horizontal(pl.lit(1.0), tpq / (bv * bv))
        ).sqrt()
        second.append(pl.when((rv > 0) & (bv > 0)).then(rel / scale).alias("__rm_z"))
    z = pl.col("__rm_z")
    z_crit = float(norm_ppf(jump_alpha))
    plain_jump = _jump_expr(rv, bv)
    jump_sig = (
        pl.when(z.is_null()).then(None).when(z > z_crit).then(plain_jump).otherwise(0.0)
    )

    exprs: dict[str, pl.Expr] = {
        "rv": rv,
        "rv_ss": pl.col("__rm_rvss") / float(subsample),
        "bv": bv,
        "medrv": pl.when(n > 2).then(
            _MEDRV_C * (nf / (nf - 2.0)) * pl.col("__rm_med2")
        ),
        "minrv": pl.when(n > 1).then(
            _MINRV_C * (nf / (nf - 1.0)) * pl.col("__rm_min2")
        ),
        "rs_pos": pl.col("__rm_rsp"),
        "rs_neg": pl.col("__rm_rsn"),
        "sjv": pl.col("__rm_rsp") - pl.col("__rm_rsn"),
        "rq": nf / 3.0 * pl.col("__rm_r4"),
        "tpq": tpq,
        "jump_z": z,
        "jump": plain_jump,
        "rel_jump": _rel_jump_expr(plain_jump, rv),
        "jump_sig": jump_sig,
        "cont": rv - jump_sig,
        "medrq": pl.when(n > 2).then(
            _MEDRQ_C * nf * (nf / (nf - 2.0)) * pl.col("__rm_med4")
        ),
        "log_rv_var": pl.when(rv > 0).then(2.0 / 3.0 * pl.col("__rm_r4") / (rv * rv)),
        "log_rv_var_tpq": pl.when(bv > 0).then(2.0 * tpq / (nf * bv * bv)),
    }
    diagnostics: dict[str, pl.Expr] = {}
    if "rk" in requested:
        exprs["rk"] = _kernel_expr(kernel_max_lags)
        diagnostics["rk_h"] = pl.col("__rm_h").cast(pl.Int64)
        diagnostics["rk_capped"] = pl.col("__rm_capped")
    if "tsrv" in requested:
        k = pl.col("__rm_tsrv_k")
        exprs["tsrv"] = _tsrv_value(pl.col("__rm_tsk") / k.cast(pl.Float64), k)
        diagnostics["tsrv_k"] = k
    if "pav" in requested:
        exprs["pav"] = _pav_value()
        diagnostics["pav_k"] = pl.col("__rm_k").cast(pl.Int64)

    enough = n >= min_obs
    out: list[pl.Expr] = []
    for name in requested:
        if name == "n_obs":
            out.append(n.fill_null(0).alias("n_obs"))
            continue
        out.append(pl.when(enough).then(exprs[name]).alias(name))
        for diag in _DIAGNOSTICS.get(name, ()):
            out.append(pl.when(enough).then(diagnostics[diag]).alias(diag))
    return [stage for stage in (first, second) if stage], out
