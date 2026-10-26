"""Small custom estimators that must live in an importable module so joblib can load them."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier


class RidgeScoreClassifier(BaseEstimator, ClassifierMixin):
    """
    RidgeClassifier with a probability-like output.

    Ridge classification regresses on targets {-1, +1}, so the decision value d
    approximates E[2y-1]; ``predict_proba`` returns clip((d+1)/2, 0, 1).  These
    scores are NOT calibrated probabilities - they are only used so Ridge can be
    compared on ROC-AUC and the same confusion-matrix metrics as the other models.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self._ridge = RidgeClassifier(alpha=self.alpha).fit(X, y)
        self.classes_ = self._ridge.classes_
        return self

    def decision_function(self, X):
        return self._ridge.decision_function(X)

    def predict(self, X):
        return self._ridge.predict(X)

    def predict_proba(self, X):
        p_up = np.clip((self.decision_function(X) + 1.0) / 2.0, 0.0, 1.0)
        return np.column_stack([1.0 - p_up, p_up])
