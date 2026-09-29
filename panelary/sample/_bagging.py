"""Bagging on sequential-bootstrap samples (AFML ch. 6), as a panel estimator."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from panelary.core._spans import _spans_from_t1
from panelary.core.panel_frame import PanelFrame
from panelary.core.protocol import PanelEstimator
from panelary.sample._bootstrap import _bootstrap_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["SequentialBagging"]


def _clone(estimator: Any) -> Any:
    """An unfitted copy: ``sklearn.base.clone`` when it applies, else a deep copy."""
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:  # sklearn missing, or not an sklearn estimator
        return copy.deepcopy(estimator)


class SequentialBagging(PanelEstimator):
    """Bag an estimator over sequential-bootstrap samples of the labels.

    Each of the ``n_estimators`` members is fitted on a sample drawn by
    :func:`panelary.sample.sequential_bootstrap` from the **fit panel's** label
    spans, so inside :func:`panelary.cross_validate` the draws depend on the
    training fold only. Predictions are the average of the members' (for
    classifiers with ``predict_proba``: the class of the averaged
    probabilities).

    Parameters
    ----------
    estimator : object
        sklearn-shaped ``fit(X, y)`` / ``predict(X)`` estimator; cloned per
        member.
    n_estimators : int, default 100
        Number of members.
    max_samples : int, float or None, default None
        Draws per member: ``None`` = the number of resolved labels; a float in
        ``(0, 1]`` is a share of it.
    t1 : str, default "t1"
        Label end-time column of the fit panel.
    seed : int, default 0
        Members use ``SeedSequence(seed).spawn(n_estimators)``: deterministic,
        and independent of the number of threads.
    target : str, default "label"
        Target column.
    features : str or sequence of str, optional
        Predictor columns. ``None`` uses every numeric column except the keys,
        the target, ``t1`` and ``censored``. Other label outputs are
        forward-looking (e.g. ``triple_barrier``'s ``ret``): exclude them.
    method : {"exact", "uniqueness_iid"}, default "exact"
        Bootstrap method (see :func:`~panelary.sample.sequential_bootstrap`).
    stratify : {None, "entity"}, default None
        Pool labels, or bootstrap each entity independently.
    censored : str or None, default "censored"
        Unresolved-label column to exclude (ignored if absent).
    prediction_col : str, default "prediction"
        Output column.
    entity, time : str, optional
        Default panel keys for bare polars frames.

    Notes
    -----
    No out-of-bag score is offered: with overlapping labels an out-of-bag
    label is rarely independent of the in-bag labels that share its rows, so
    OOB accuracy is inflated (AFML ch. 6). Score with purged cross-validation.
    """

    panel_safe = True
    leakage_safe = True

    def __init__(
        self,
        estimator: Any,
        *,
        n_estimators: int = 100,
        max_samples: int | float | None = None,
        t1: str = "t1",
        seed: int = 0,
        target: str = "label",
        features: str | Sequence[str] | None = None,
        method: str = "exact",
        stratify: str | None = None,
        censored: str | None = "censored",
        prediction_col: str = "prediction",
        entity: str | None = None,
        time: str | None = None,
    ) -> None:
        super().__init__(entity=entity, time=time)
        if n_estimators < 1:
            raise ValueError(f"`n_estimators` must be >= 1, got {n_estimators}.")
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.t1 = t1
        self.seed = seed
        self.target = target
        self.features = [features] if isinstance(features, str) else features
        self.method = method
        self.stratify = stratify
        self.censored = censored
        self.prediction_col = prediction_col
        self.estimators_: list[Any] = []
        self.samples_: list[NDArray[np.int64]] = []
        self.features_: list[str] = []
        self.classes_: NDArray[Any] | None = None

    def _resolve_features(
        self, frame: pl.DataFrame, entity: str, time: str
    ) -> list[str]:
        if self.features is not None:
            missing = [c for c in self.features if c not in frame.columns]
            if missing:
                raise ValueError(f"feature column(s) {missing} not found.")
            return list(self.features)
        skip = {entity, time, self.target, self.t1, self.censored}
        feats = [
            c for c, dt in frame.schema.items() if c not in skip and dt.is_numeric()
        ]
        if not feats:
            raise ValueError("no numeric feature columns left; pass `features=`.")
        return feats

    def _fit(self, panel: PanelFrame) -> None:
        entity, time = panel.entity_col, panel.time_col
        frame, spans = _spans_from_t1(
            panel.collect(),
            t1=self.t1,
            entity=entity,
            time=time,
            censored=self.censored,
        )
        if self.target not in frame.columns:
            raise ValueError(f"target column {self.target!r} not found.")
        feats = self._resolve_features(frame, entity, time)
        X = frame.select(feats).to_numpy()
        y = frame.get_column(self.target).to_numpy()
        keys = frame.get_column(entity).unique(maintain_order=True).to_list()
        children = np.random.SeedSequence(self.seed).spawn(self.n_estimators)
        members: list[Any] = []
        samples: list[NDArray[np.int64]] = []
        for child in children:
            rows = _bootstrap_rows(
                spans,
                keys,
                n_draws=self.max_samples,
                ss=child,
                method=self.method,
                stratify=self.stratify,
            )
            est = _clone(self.estimator)
            est.fit(X[rows], y[rows])
            members.append(est)
            samples.append(rows)
        self.estimators_ = members
        self.samples_ = samples
        self.features_ = feats
        if all(hasattr(m, "predict_proba") for m in members):
            self.classes_ = np.unique(y[spans.label_row])
        else:
            self.classes_ = None

    def _mean_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        classes = self.classes_
        acc = np.zeros((X.shape[0], classes.shape[0]), dtype=np.float64)
        for est in self.estimators_:
            proba = np.asarray(est.predict_proba(X), dtype=np.float64)
            cols = np.searchsorted(classes, np.asarray(est.classes_))
            acc[:, cols] += proba
        return acc / len(self.estimators_)

    def _design(self, panel: PanelFrame) -> tuple[pl.DataFrame, np.ndarray]:
        keys = [panel.entity_col, panel.time_col]
        frame = panel.lazy().select([*keys, *self.features_]).collect()
        return frame.select(keys), frame.select(self.features_).to_numpy()

    def _predict(self, panel: PanelFrame) -> PanelFrame:
        keys, X = self._design(panel)
        if self.classes_ is not None:
            preds: np.ndarray = self.classes_[np.argmax(self._mean_proba(X), axis=1)]
        else:
            acc = np.zeros(X.shape[0], dtype=np.float64)
            for est in self.estimators_:
                acc += np.asarray(est.predict(X), dtype=np.float64).ravel()
            preds = acc / len(self.estimators_)
        out = keys.with_columns(pl.Series(self.prediction_col, preds))
        return PanelFrame(
            out, entity=panel.entity_col, time=panel.time_col, validate=False
        )

    def predict_proba(
        self,
        X: PanelFrame | pl.DataFrame | pl.LazyFrame,
        *,
        entity: str | None = None,
        time: str | None = None,
    ) -> PanelFrame:
        """Averaged class probabilities, one column per class seen in training."""
        panel = self._as_panel(X, method="predict_proba", entity=entity, time=time)
        self._check_fitted("predict_proba")
        if self.classes_ is None:
            raise AttributeError("the bagged estimator has no `predict_proba`.")
        keys, Xm = self._design(panel)
        proba = self._mean_proba(Xm)
        cols = [
            pl.Series(f"{self.prediction_col}_proba_{cls}", proba[:, i])
            for i, cls in enumerate(self.classes_.tolist())
        ]
        return PanelFrame(
            keys.with_columns(cols),
            entity=panel.entity_col,
            time=panel.time_col,
            validate=False,
        )
