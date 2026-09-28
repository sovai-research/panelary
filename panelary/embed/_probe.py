"""Layer 4: a ridge readout whose penalty is always cross-validated and never zero.

PreValidated Ridge (Dempster, Webb & Schmidt, *Machine Learning*, 2026) makes
a linear probe on a wide embedding cheap enough to run inside every fold: one
SVD of the training matrix gives the **exact leave-one-out ("prevalidated")
predictions for every penalty on a grid at once**, via the ridge hat matrix

    y_loo_i = y_i - (y_i - yhat_i) / (1 - H_ii),
    H = 11'/n + U diag(s^2 / (s^2 + alpha)) U'        (centred X = U S V')

so choosing ``alpha`` costs ``O(n p min(n, p))`` once plus ``O(n r)`` per grid
point -- no refits. For classification the prevalidated scores are also what
the probabilities are calibrated on (a single softmax temperature fit to the
leave-one-out scores), so no extra data is spent on calibration.

Guardrails (build contract section 5)
-------------------------------------
* **Never ridgeless.** ``alpha <= 0`` anywhere on the grid raises. With more
  random features than observations and no penalty, a regression degenerates
  mechanically into volatility-timed momentum and mis-learns on mean-reverting
  data (Nagel, 2025, NBER w34104).
* **Warn hard when P > T.** When features outnumber the distinct training
  dates (or rows, if no dates are given) an
  :class:`OverparameterisedReadoutWarning` is raised with that citation.
* **Leave-one-out is optimistic under dependence.** Panel rows are serially
  and cross-sectionally correlated, so plain LOO under-penalises. Pass
  ``cv=<int>`` (contiguous date blocks with a ``purge`` gap) or a
  :class:`~panelary.core.model_selection.PurgedKFold` to select ``alpha`` on
  time-respecting folds instead; the fold fits reuse one Gram matrix per fold.

Clean-room note: implemented from the closed-form leave-one-out identity for
penalised least squares, which predates the paper; the paper's contribution
this module follows is using the prevalidated predictions for both penalty
selection and calibration.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.embed._contract import matrix_from_columns

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

__all__ = ["DEFAULT_ALPHAS", "OverparameterisedReadoutWarning", "PreValidatedRidge"]

#: Default penalty grid: 1e-3 .. 1e4, three points per decade.
DEFAULT_ALPHAS: tuple[float, ...] = tuple(float(a) for a in np.logspace(-3, 4, 22))

_NAGEL = (
    "Nagel (2025, NBER w34104): with more features than observations, random-feature "
    "regressions degenerate mechanically into volatility-timed momentum and mis-learn "
    "on mean-reverting data. Run panelary.embed.reversal_check before trusting it."
)


class OverparameterisedReadoutWarning(UserWarning):
    """The readout has more features than training dates (``P > T``)."""


def _check_alphas(alphas: Sequence[float] | float) -> NDArray[np.float64]:
    a = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    if a.size == 0:
        raise ValueError("PreValidatedRidge: `alphas` must not be empty.")
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(
            "PreValidatedRidge: every alpha must be a finite, strictly positive "
            f"penalty; got {a.tolist()}. Ridgeless readouts are refused on purpose. "
            + _NAGEL
        )
    return np.sort(a)


def _golden_min(f: Any, lo: float, hi: float, iters: int = 60) -> float:
    """Minimise a unimodal scalar function on ``[lo, hi]`` (golden-section search)."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _softmax(S: NDArray[np.float64]) -> NDArray[np.float64]:
    Z = S - S.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class PreValidatedRidge:
    """Ridge readout with exact leave-one-out (or purged-fold) penalty selection.

    Parameters
    ----------
    alphas : sequence of float, default :data:`DEFAULT_ALPHAS`
        Penalty grid; every value must be ``> 0``.
    cv : "loo", int, or a splitter, default "loo"
        ``"loo"``: closed-form leave-one-out over rows (fast; optimistic under
        dependence). ``int k``: ``k`` contiguous blocks of distinct training
        dates, each held out with ``purge`` dates removed on either side
        (requires ``time=`` in :meth:`fit`). A splitter with ``split(panel)``
        (e.g. :class:`~panelary.core.model_selection.PurgedKFold`) is driven
        on the training dates.
    purge : int, default 0
        Dates dropped on each side of a held-out block (``cv=int`` only).
    task : {"regression", "classification"}, default "regression"
    standardize : bool, default True
        Scale features to unit training variance (statistics from the training
        rows only; features constant on the training rows are left unscaled).
    warn_overparameterised : bool, default True

    Attributes
    ----------
    alpha_ : float
        Selected penalty.
    alphas_ : numpy.ndarray
        The (sorted) grid.
    cv_error_ : numpy.ndarray
        Mean squared validation error per alpha (for classification with
        ``cv="loo"``, on the ``+-1`` one-vs-rest coding).
    coef_ : numpy.ndarray
        ``(p,)`` or ``(p, n_outputs)`` coefficients in the *original* feature units.
    intercept_ : float or numpy.ndarray
    prevalidated_ : numpy.ndarray or None
        Leave-one-out predictions at ``alpha_`` (``cv="loo"`` only).
    classes_ : numpy.ndarray or None
    temperature_ : float or None
        Softmax temperature fitted to the prevalidated scores (classification).
    n_features_in_, n_times_ : int
    """

    def __init__(
        self,
        alphas: Sequence[float] | float = DEFAULT_ALPHAS,
        *,
        cv: Any = "loo",
        purge: int = 0,
        task: str = "regression",
        standardize: bool = True,
        warn_overparameterised: bool = True,
    ) -> None:
        self.alphas_ = _check_alphas(alphas)
        if not (
            cv == "loo"
            or (isinstance(cv, int) and not isinstance(cv, bool))
            or hasattr(cv, "split")
        ):
            raise ValueError(
                f"PreValidatedRidge: `cv` must be 'loo', an int or a splitter, got {cv!r}."
            )
        if isinstance(cv, int) and not isinstance(cv, bool) and cv < 2:
            raise ValueError(f"PreValidatedRidge: `cv={cv}` must be >= 2 folds.")
        self.cv = cv
        if isinstance(purge, bool) or not isinstance(purge, int) or purge < 0:
            raise ValueError(
                f"PreValidatedRidge: `purge` must be an int >= 0, got {purge!r}."
            )
        self.purge = purge
        if task not in ("regression", "classification"):
            raise ValueError(
                f"PreValidatedRidge: `task` must be 'regression' or 'classification', got {task!r}."
            )
        self.task = task
        self.standardize = bool(standardize)
        self.warn_overparameterised = bool(warn_overparameterised)
        self.alpha_: float = float("nan")
        self.cv_error_: NDArray[np.float64] | None = None
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: Any = None
        self.prevalidated_: NDArray[np.float64] | None = None
        self.classes_: NDArray[Any] | None = None
        self.temperature_: float | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.n_features_in_: int = 0
        self.n_times_: int = 0

    # ------------------------------------------------------------------ #
    def _targets(self, y: NDArray[Any]) -> NDArray[np.float64]:
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if self.classes_.size < 2:
                raise ValueError(
                    "PreValidatedRidge: classification needs at least 2 classes."
                )
            return np.where(y[:, None] == self.classes_[None, :], 1.0, -1.0)
        Y = np.asarray(y, dtype=np.float64)
        return Y[:, None] if Y.ndim == 1 else Y

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        *,
        time: NDArray[Any] | Sequence[Any] | None = None,
    ) -> Self:
        """Select ``alpha`` by (pre)validation and fit on all rows.

        Parameters
        ----------
        X : numpy.ndarray
            ``(n, p)`` finite features.
        y : numpy.ndarray
            ``(n,)`` target (or ``(n, k)`` for multi-output regression; class
            labels for classification).
        time : array-like, optional
            Each row's date. Used for the ``P > T`` check (``T`` = distinct
            dates) and required by ``cv=int`` / splitter.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            On non-finite inputs, too few rows, or a CV that needs ``time``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"PreValidatedRidge.fit: X must be 2-D, got shape {X.shape}."
            )
        y_arr = np.asarray(y)
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError(
                "PreValidatedRidge.fit: X and y have different row counts."
            )
        if not np.isfinite(X).all():
            raise ValueError(
                "PreValidatedRidge.fit: X contains NaN/inf; drop or impute (in-fold) first."
            )
        n, p = X.shape
        if n < 3:
            raise ValueError("PreValidatedRidge.fit: need at least 3 rows.")
        Y = self._targets(y_arr)
        if not np.isfinite(Y).all():
            raise ValueError("PreValidatedRidge.fit: y contains NaN/inf.")
        t = None if time is None else np.asarray(time)
        self.n_times_ = int(np.unique(t).size) if t is not None else n
        self.n_features_in_ = p
        if self.warn_overparameterised and p > self.n_times_:
            warnings.warn(
                f"PreValidatedRidge: P={p} features > T={self.n_times_} "
                f"{'dates' if t is not None else 'rows'} in the readout. " + _NAGEL,
                OverparameterisedReadoutWarning,
                stacklevel=2,
            )
        mean = X.mean(axis=0)
        scale = X.std(axis=0) if self.standardize else np.ones(p)
        scale = np.where(scale > 0, scale, 1.0)
        Xs = (X - mean) / scale
        self.mean_, self.scale_ = mean, scale

        if self.cv == "loo":
            errors, loo = self._loo(Xs, Y)
        else:
            if t is None:
                raise ValueError(
                    "PreValidatedRidge.fit: fold CV needs `time=` (each row's date)."
                )
            errors, loo = self._folds(Xs, Y, t), None
        self.cv_error_ = errors
        best = int(np.argmin(errors))
        self.alpha_ = float(self.alphas_[best])
        beta, b0 = self._solve(Xs, Y, self.alpha_)
        coef = beta / scale[:, None]
        intercept = b0 - mean @ coef
        self.coef_ = (
            coef[:, 0] if coef.shape[1] == 1 and self.task == "regression" else coef
        )
        self.intercept_ = (
            float(intercept[0])
            if intercept.shape[0] == 1 and self.task == "regression"
            else intercept
        )
        self.prevalidated_ = None if loo is None else loo[best]
        if self.task == "classification" and loo is not None:
            codes = np.searchsorted(self.classes_, y_arr)  # type: ignore[arg-type]
            S = loo[best]

            def nll(log_tau: float) -> float:
                P = _softmax(S / np.exp(log_tau))
                return float(
                    -np.mean(np.log(np.maximum(P[np.arange(n), codes], 1e-300)))
                )

            self.temperature_ = float(np.exp(_golden_min(nll, -6.0, 6.0)))
        elif self.task == "classification":
            self.temperature_ = 1.0
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def _solve(
        Xs: NDArray[np.float64], Y: NDArray[np.float64], alpha: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Ridge with an unpenalised intercept on (already standardised) ``Xs``."""
        mx, my = Xs.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = Xs - mx, Y - my
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        beta = Vt.T @ ((s / (s * s + alpha))[:, None] * (U.T @ Yc))
        return beta, my - mx @ beta

    def _loo(
        self, Xs: NDArray[np.float64], Y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        n = Xs.shape[0]
        Xc = Xs - Xs.mean(axis=0)
        my = Y.mean(axis=0)
        Yc = Y - my
        U, s, _ = np.linalg.svd(Xc, full_matrices=False)
        UtY = U.T @ Yc
        s2 = s * s
        errors = np.empty(self.alphas_.size)
        loo: list[NDArray[np.float64]] = []
        for i, a in enumerate(self.alphas_):
            shrink = s2 / (s2 + a)
            fitted = my + U @ (shrink[:, None] * UtY)
            h = 1.0 / n + (U * U) @ shrink
            resid = (Y - fitted) / (1.0 - h)[:, None]
            errors[i] = float(np.mean(resid * resid))
            loo.append(Y - resid)
        return errors, loo

    def _fold_positions(
        self, t: NDArray[Any]
    ) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """``(train_date_positions, test_date_positions)`` into the sorted unique dates."""
        u = np.unique(t)
        if isinstance(self.cv, int):
            if u.size < self.cv:
                raise ValueError(
                    f"PreValidatedRidge: {u.size} dates cannot make {self.cv} folds."
                )
            out = []
            for block in np.array_split(np.arange(u.size), self.cv):
                lo, hi = int(block[0]) - self.purge, int(block[-1]) + self.purge
                train = np.flatnonzero(
                    (np.arange(u.size) < lo) | (np.arange(u.size) > hi)
                )
                out.append((train.astype(np.int64), block.astype(np.int64)))
            return out
        from panelary.core.panel_frame import PanelFrame

        frame = pl.DataFrame({"__pn_e__": np.arange(u.size), "__pn_t__": u})
        panel = PanelFrame(frame, entity="__pn_e__", time="__pn_t__", validate=False)
        out = []
        for tr, te in self.cv.split(panel):
            if isinstance(tr, np.ndarray):
                out.append((tr.astype(np.int64), te.astype(np.int64)))
            else:
                ttr = tr.collect()["__pn_t__"].to_numpy()
                tte = te.collect()["__pn_t__"].to_numpy()
                out.append(
                    (
                        np.searchsorted(u, ttr).astype(np.int64),
                        np.searchsorted(u, tte).astype(np.int64),
                    )
                )
        return out

    def _folds(
        self, Xs: NDArray[np.float64], Y: NDArray[np.float64], t: NDArray[Any]
    ) -> NDArray[np.float64]:
        u = np.unique(t)
        pos = np.searchsorted(u, t)
        sse = np.zeros(self.alphas_.size)
        count = 0
        for tr_pos, te_pos in self._fold_positions(t):
            tr = np.isin(pos, tr_pos)
            te = np.isin(pos, te_pos)
            if tr.sum() < 2 or te.sum() == 0:
                continue
            Xtr, Ytr = Xs[tr], Y[tr]
            mx, my = Xtr.mean(axis=0), Ytr.mean(axis=0)
            Xc = Xtr - mx
            G = Xc.T @ Xc
            evals, V = np.linalg.eigh(G)
            evals = np.maximum(evals, 0.0)
            VtXy = V.T @ (Xc.T @ (Ytr - my))
            Xte = (Xs[te] - mx) @ V
            for i, a in enumerate(self.alphas_):
                pred = my + Xte @ (VtXy / (evals + a)[:, None])
                sse[i] += float(np.sum((Y[te] - pred) ** 2))
            count += int(te.sum()) * Y.shape[1]
        if count == 0:
            raise ValueError(
                "PreValidatedRidge: no usable folds (every fold was empty after purging)."
            )
        return sse / count

    # ------------------------------------------------------------------ #
    def _check(self) -> None:
        if self.coef_ is None:
            raise RuntimeError(
                "this PreValidatedRidge instance is not fitted yet; call `fit` first."
            )

    def decision_function(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Linear scores ``X @ coef_ + intercept_``."""
        self._check()
        assert self.coef_ is not None
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predicted values (regression) or labels (classification)."""
        S = self.decision_function(X)
        if self.task == "classification":
            assert self.classes_ is not None
            return self.classes_[np.argmax(S, axis=1)]
        return S

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Class probabilities: softmax of the scores at the prevalidated temperature."""
        if self.task != "classification":
            raise AttributeError(
                "predict_proba is only defined for task='classification'."
            )
        S = self.decision_function(X)
        return _softmax(S / float(self.temperature_ or 1.0))

    # ------------------------------------------------------------------ #
    def fit_frame(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        features: Sequence[str],
        target: str,
        time: str | None = None,
    ) -> Self:
        """Fit on a polars frame (numeric and/or ``pl.Array`` feature columns).

        Rows with any missing feature or target are dropped (never imputed).
        """
        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        X = matrix_from_columns(df, list(features))
        y = df[target].to_numpy()
        ok = np.isfinite(X).all(axis=1) & (
            np.isfinite(y.astype(np.float64))
            if self.task == "regression"
            else df[target].is_not_null().to_numpy()
        )
        t = None if time is None else df[time].to_numpy()[ok]
        return self.fit(X[ok], y[ok], time=t)

    def predict_frame(
        self, frame: pl.DataFrame | pl.LazyFrame, *, features: Sequence[str]
    ) -> NDArray[Any]:
        """Predict for every row of a polars frame (NaN / None where a feature is missing)."""
        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        X = matrix_from_columns(df, list(features))
        ok = np.isfinite(X).all(axis=1)
        pred = self.predict(np.where(ok[:, None], X, 0.0))
        if self.task == "classification":
            out = pred.astype(object)
            out[~ok] = None
            return out
        pred = np.asarray(pred, dtype=np.float64)
        pred[~ok] = np.nan
        return pred
