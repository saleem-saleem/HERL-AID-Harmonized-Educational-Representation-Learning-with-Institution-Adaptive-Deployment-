"""Leakage-safe preprocessing (Reviewer 3, Comment 1).

Every transformation is fit on the TRAINING partition only and then applied to the
held-out institution; class imbalance is handled by cost-sensitive class weights
(NOT pooled oversampling), so no synthetic instances cross the train/test boundary.
This module implements the Table P pipeline ordering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class LeakageSafePreprocessor:
    outlier_quantile: float = 0.95
    standardize: bool = True
    # fitted state (training only):
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    maha_threshold_: Optional[float] = None
    class_weights_: Dict[int, float] = field(default_factory=dict)

    # -- fit on TRAIN only --------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeakageSafePreprocessor":
        if self.standardize:
            self.mean_ = X.mean(0)
            self.std_ = X.std(0) + 1e-8
        Xs = self._standardize(X)
        # Mahalanobis (diagonal) outlier threshold from TRAIN statistics
        d2 = (Xs ** 2).sum(1)
        self.maha_threshold_ = float(np.quantile(d2, self.outlier_quantile))
        # cost-sensitive class weights from TRAIN frequencies
        classes, counts = np.unique(y, return_counts=True)
        n, k = len(y), len(classes)
        self.class_weights_ = {int(c): n / (k * cnt) for c, cnt in zip(classes, counts)}
        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize or self.mean_ is None:
            return X
        return (X - self.mean_) / self.std_

    # -- apply to TRAIN (with outlier removal) ------------------------------- #
    def transform_train(self, X: np.ndarray, y: np.ndarray):
        Xs = self._standardize(X)
        d2 = (Xs ** 2).sum(1)
        keep = d2 <= self.maha_threshold_
        return Xs[keep], y[keep]

    # -- apply to TEST (NO outlier removal; keep natural distribution) ------- #
    def transform_test(self, X: np.ndarray) -> np.ndarray:
        return self._standardize(X)

    def sample_weights(self, y: np.ndarray) -> np.ndarray:
        return np.array([self.class_weights_[int(t)] for t in y], dtype=np.float32)
