"""Frame-level information theory: mutual information and transfer entropy.

Both return the fixed result schema. Mutual information goes through the same
engine and null policy as :func:`~panelary.depend.dependence` (``"gcmi"`` with
its chi-square closed form, ``"ksg"`` with a permutation / block /
common-time null). Transfer entropy is computed per entity on the shared date
axis; its panel null permutes whole date columns of the **source** jointly
across entities, which breaks the source-to-target timing while keeping every
common shock.

Transfer entropy is **not** evidence of causation: under a common latent
driver that reaches two series at different delays, both directions inflate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from panelary.depend._engine import (
    _GCMI_NOTE,
    DEFAULT_RESAMPLES,
    _check_by,
    _sample_rows,
    panel_block_length,
    panel_serial,
    run_pair,
    stouffer,
)
from panelary.depend._frame import extract, result_frame
from panelary.depend._info import mi_ksg, transfer_entropy_array
from panelary.depend._kernels import Kernel, _fisher_var, get_kernel
from panelary.depend._null import (
    SERIAL_THRESHOLD,
    block_permutation_indices,
    circular_shift,
    common_time_indices,
    pvalue,
)

__all__ = ["ksg_kernel", "mutual_information", "transfer_entropy"]


def ksg_kernel(*, k: int = 5, max_n: int = 5000, seed: int = 0) -> Kernel:
    """Engine kernel for KSG mutual information (class D: one ``O(m^2)`` call
    per row, so resampling nulls cost ``B`` of them)."""

    def rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        return np.array(
            [mi_ksg(X[b], Y[b], k=k, max_n=max_n, seed=seed) for b in range(X.shape[0])]
        )

    return Kernel(
        "mi_ksg",
        rows,
        "greater",
        False,
        max(100, 10 * int(k) + 1),
        None,
        "mean",
        _fisher_var,
        estimator=f"ksg mutual information (k={int(k)}, nats)",
        policy="mi_ksg",
    )


def mutual_information(
    df: Any,
    x: str,
    y: str,
    *,
    estimator: str = "gcmi",
    by: str | None = "entity",
    null: str = "auto",
    entity: str | None = None,
    time: str | None = None,
    k: int = 5,
    max_n: int = 5000,
    how: str | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """Mutual information of ``x`` and ``y`` in nats, with an honest null.

    Parameters
    ----------
    estimator : {"gcmi", "ksg"}, default="gcmi"
        ``"gcmi"``: Gaussian-copula MI -- for one pair of columns a monotone
        function of a rank correlation (see the note it attaches). ``"ksg"``:
        Kraskov k-NN MI, genuinely nonlinear, ``O(n^2)`` per estimate and
        capped at ``max_n``.
    k : int, default=5
        Neighbours for ``"ksg"``.
    by, null, entity, time, how, n_resamples, block_length, seed, min_obs
        As in :func:`~panelary.depend.dependence`.

    Returns
    -------
    polars.DataFrame
        ``x``, ``y`` and the fixed schema.
    """
    if estimator == "gcmi":
        kernel = get_kernel("gcmi")
    elif estimator == "ksg":
        kernel = ksg_kernel(k=k, max_n=max_n, seed=seed)
    else:
        raise ValueError(
            f"unknown `estimator` {estimator!r}; expected 'gcmi' or 'ksg'."
        )
    pa = extract(df, [x, y], entity=entity, time=time)
    row = run_pair(
        pa,
        pa.dense(x),
        pa.dense(y),
        kernel,
        by=_check_by(by),
        null=null,
        how=how,
        demean="none",
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        min_obs=min_obs,
    )
    if estimator == "gcmi" and _GCMI_NOTE not in row["warnings"]:
        row["warnings"] = [*row["warnings"], _GCMI_NOTE]
    row.update(x=x, y=y, lag=0)
    return result_frame([row], keys={"x": pl.Utf8, "y": pl.Utf8})


def _te_rows(
    S: np.ndarray,
    Tg: np.ndarray,
    *,
    lag: int,
    history: int,
    estimator: str,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ent = S.shape[0]
    te = np.full(n_ent, np.nan)
    pv = np.full(n_ent, np.nan)
    nn = np.zeros(n_ent, dtype=np.int64)
    for i in range(n_ent):
        r = transfer_entropy_array(
            S[i], Tg[i], lag=lag, history=history, estimator=estimator, k=k, seed=seed
        )
        te[i], pv[i], nn[i] = r.te, r.p_value, r.n_obs
    return te, pv, nn


def transfer_entropy(
    df: Any,
    source: str,
    target: str,
    *,
    lag: int = 1,
    history: int = 1,
    estimator: str = "gaussian",
    null: str = "auto",
    entity: str | None = None,
    time: str | None = None,
    k: int = 5,
    n_resamples: int = DEFAULT_RESAMPLES,
    block_length: int | None = None,
    seed: int = 0,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Transfer entropy ``source -> target`` per entity, aggregated.

    ``TE = I(target_t ; source_{t-1..t-lag} | target_{t-1..t-history})`` (see
    :func:`~panelary.depend.transfer_entropy_array` for the estimators).

    Nulls: ``"asymptotic"`` -- chi-square(``lag``) per entity (Stouffer across
    entities), valid when ``history`` captures the target's own dynamics and
    entities are independent. ``"block"`` / ``"shift"`` -- resample the source
    series. ``"common-time"`` -- permute the source's date columns jointly
    across entities (blocks). ``"auto"``: common-time for a panel with a time
    column, else the closed form. Resampling nulls need the Gaussian or copula
    estimator.

    **Not evidence of causation** -- a common latent driver inflates both
    directions.

    Returns
    -------
    polars.DataFrame
        ``x`` (= source), ``y`` (= target) and the fixed schema; ``estimate``
        is the ``n``-weighted mean TE in nats.
    """
    pa = extract(df, [source, target], entity=entity, time=time)
    S, Tg = pa.dense(source), pa.dense(target)
    n_ent = S.shape[0]
    te, pv, nn = _te_rows(
        S, Tg, lag=lag, history=history, estimator=estimator, k=k, seed=seed
    )
    valid = np.isfinite(te) & (nn >= min_obs)
    notes = [
        "transfer entropy measures predictive information flow, not causation; a "
        "common driver at different delays inflates both directions."
    ]
    if (~valid & (nn > 0)).any():
        notes.append(
            f"{int((~valid & (nn > 0)).sum())} entities below min_obs={min_obs} excluded."
        )
    k_ok = int(valid.sum())
    est = (
        float(np.sum(te[valid] * nn[valid]) / nn[valid].sum()) if k_ok else float("nan")
    )
    row: dict[str, Any] = {
        "estimate": est,
        "method": "transfer_entropy",
        "estimator": f"{estimator} TE (lag={lag}, history={history}, nats)",
        "direction": "x->y",
        "lag": int(lag),
        "n_obs": int(nn[valid].sum()),
        "n_entities": k_ok,
        "coverage": k_ok / n_ent if n_ent else float("nan"),
        "transform": "none",
        "approximate": False,
    }
    panel_ok = n_ent >= 2 and pa.has_time
    scheme = null
    if null == "auto":
        scheme = "common-time" if panel_ok else "asymptotic"
    if scheme in {"asymptotic", "iid"}:
        if n_ent == 1 or k_ok == 1:
            p = float(pv[valid][0]) if k_ok else float("nan")
            label = "asymptotic" if estimator != "ksg" else "shift"
        else:
            p = stouffer(pv[valid], np.ones(k_ok), nn[valid], alternative="greater")
            label = "asymptotic+stouffer"
            notes.append(
                "Stouffer combination assumes cross-sectionally independent entities."
            )
        row.update(p_value=p, null_method=label)
    else:
        if estimator == "ksg":
            raise ValueError(
                "resampling nulls for transfer entropy need estimator='gaussian' or 'copula'."
            )
        if scheme == "common-time" and not pa.has_time:
            raise ValueError("null='common-time' needs a `time` column.")
        if scheme not in {"common-time", "block", "shift"}:
            raise ValueError(
                f"null={null!r} is not available for transfer entropy; use 'auto', "
                "'asymptotic', 'block', 'shift' or 'common-time'."
            )
        t_len = S.shape[1]
        b = int(n_resamples)
        rows_s = _sample_rows(n_ent)
        blen: int | None
        if block_length is not None:
            blen = int(block_length)
        elif panel_serial(S, Tg, rows_s) > SERIAL_THRESHOLD:
            blen = panel_block_length(S, Tg, rows_s, None)
        else:
            blen = 1
        if scheme == "common-time":
            idx = common_time_indices(
                np.arange(t_len), n_resamples=b, seed=seed, block=int(blen)
            )
        elif scheme == "block":
            idx = block_permutation_indices(
                t_len, block_length=int(blen), n_resamples=b, seed=seed
            )
        else:
            idx = circular_shift(t_len, n_resamples=b, seed=seed)
            blen = None
        draws = np.full(idx.shape[0], np.nan)
        for j in range(idx.shape[0]):
            te_b, _, nn_b = _te_rows(
                S[:, idx[j]],
                Tg,
                lag=lag,
                history=history,
                estimator=estimator,
                k=k,
                seed=seed,
            )
            ok = np.isfinite(te_b) & (nn_b >= min_obs)
            if ok.any():
                draws[j] = float(np.sum(te_b[ok] * nn_b[ok]) / nn_b[ok].sum())
        row.update(
            p_value=pvalue(est, draws),
            null_method=scheme,
            n_resamples=int(idx.shape[0]),
            block_length=blen,
            seed=int(seed),
        )
    row.update(x=source, y=target, warnings=notes)
    return result_frame([row], keys={"x": pl.Utf8, "y": pl.Utf8})
