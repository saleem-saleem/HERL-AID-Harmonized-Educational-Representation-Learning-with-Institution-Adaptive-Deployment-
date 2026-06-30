"""Standard baselines: Logistic Regression, Random Forest, XGBoost, DNN.

All use class weighting for imbalance and a uniform sklearn-style interface
``fit(X, y) / predict_proba(X)`` so the LOIO driver can treat them identically.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def get_standard_baselines(seed: int = 42) -> Dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=seed, n_jobs=-1),
        "dnn": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300,
                             random_state=seed),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                          subsample=0.9, eval_metric="logloss",
                                          random_state=seed, n_jobs=-1)
    return models


class SklearnProbaWrapper:
    """Uniform wrapper exposing predict_proba_pos(X)."""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, sample_weight=None):
        try:
            self.model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(X, y)
        return self

    def predict_proba_pos(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
