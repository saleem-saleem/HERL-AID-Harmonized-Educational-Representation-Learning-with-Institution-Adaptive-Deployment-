"""Stage 3 - CEISD: Contextual Educational Insight & Strategy Discovery.

Produces (i) calibrated risk probabilities via temperature scaling (fit on a
validation split, not the test institution), (ii) SHAP-based attributions to drive
personalized intervention, and (iii) a composite Educational Risk Index (ERI).
The SHAP step is the most memory-intensive; its background set is subsamplable.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize_scalar


# --------------------------------------------------------------------------- #
# Temperature scaling (calibration)
# --------------------------------------------------------------------------- #
class TemperatureScaler:
    def __init__(self):
        self.T_ = 1.0

    @staticmethod
    def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
        T = max(T, 1e-3)
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def fit(self, prob_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        logits = np.log(np.clip(prob_val, 1e-7, 1 - 1e-7) /
                        np.clip(1 - prob_val, 1e-7, 1 - 1e-7))
        res = minimize_scalar(self._nll, bounds=(0.05, 10.0), method="bounded",
                              args=(logits, y_val))
        self.T_ = float(res.x)
        return self

    def transform(self, prob: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(prob, 1e-7, 1 - 1e-7) /
                        np.clip(1 - prob, 1e-7, 1 - 1e-7))
        return 1.0 / (1.0 + np.exp(-logits / self.T_))


# --------------------------------------------------------------------------- #
# SHAP attribution + composite Educational Risk Index
# --------------------------------------------------------------------------- #
class CEISD:
    def __init__(self, cfg: Dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.scaler_ = TemperatureScaler()

    def calibrate(self, prob_val, y_val):
        self.scaler_.fit(prob_val, y_val)
        return self

    def calibrated_proba(self, prob):
        return self.scaler_.transform(prob)

    def shap_attributions(self, predict_fn, Z, background: Optional[np.ndarray] = None):
        """Return per-feature SHAP values (subsampled background for memory)."""
        try:
            import shap
        except ImportError:
            return None
        bg_n = self.cfg.get("shap_background", 200)
        rng = np.random.default_rng(self.seed)
        if background is None:
            background = Z[rng.choice(len(Z), size=min(bg_n, len(Z)), replace=False)]
        explainer = shap.KernelExplainer(predict_fn, background)
        sample = Z[rng.choice(len(Z), size=min(100, len(Z)), replace=False)]
        return explainer.shap_values(sample, silent=True)

    @staticmethod
    def educational_risk_index(prob: np.ndarray, deficits: Dict[str, np.ndarray]) -> np.ndarray:
        """Composite ERI: calibrated risk blended with normalized deficit signals."""
        eri = prob.copy()
        for _, d in deficits.items():
            dn = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
            eri = eri + 0.5 * dn
        return eri / (1 + 0.5 * len(deficits))
