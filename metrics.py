"""Evaluation metrics: predictive performance, fairness, and calibration.

All fairness quantities are defined relative to a binary protected attribute
``a in {0, 1}`` (gender) and the positive (at-risk) class ``y = 1``.
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score,
)


# --------------------------------------------------------------------------- #
# Predictive metrics
# --------------------------------------------------------------------------- #
def predictive_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # minority (at-risk, positive) class metrics
        "at_risk_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "at_risk_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    return out


# --------------------------------------------------------------------------- #
# Fairness metrics  (Demographic Parity and Equal Opportunity deviations)
# --------------------------------------------------------------------------- #
def demographic_parity_diff(y_pred: np.ndarray, a: np.ndarray) -> float:
    """|P(yhat=1|a=0) - P(yhat=1|a=1)|."""
    p0 = y_pred[a == 0].mean() if (a == 0).any() else 0.0
    p1 = y_pred[a == 1].mean() if (a == 1).any() else 0.0
    return abs(p0 - p1)


def equal_opportunity_diff(y_true: np.ndarray, y_pred: np.ndarray,
                           a: np.ndarray) -> float:
    """|TPR(a=0) - TPR(a=1)| on the positive (at-risk) class."""
    def tpr(mask):
        sel = mask & (y_true == 1)
        return y_pred[sel].mean() if sel.any() else 0.0
    return abs(tpr(a == 0) - tpr(a == 1))


def fairness_metrics(y_true: np.ndarray, y_prob: np.ndarray, a: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "delta_dp": demographic_parity_diff(y_pred, a),
        "delta_eo": equal_opportunity_diff(y_true, y_pred, a),
    }


# --------------------------------------------------------------------------- #
# Calibration  (Expected Calibration Error)
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Standard binned ECE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.any():
            conf = y_prob[mask].mean()
            acc = (y_true[mask] == 1).mean()
            ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


# --------------------------------------------------------------------------- #
# Cross-institution aggregation
# --------------------------------------------------------------------------- #
def cross_institution_variance(per_institution: Sequence[Dict[str, float]],
                               key: str) -> float:
    """Variance of a metric across held-out institutions (stability measure)."""
    vals = [d[key] for d in per_institution if not np.isnan(d.get(key, np.nan))]
    return float(np.var(vals)) if vals else float("nan")


def full_evaluation(y_true: np.ndarray, y_prob: np.ndarray, a: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute the complete metric suite for one held-out institution."""
    out: Dict[str, float] = {}
    out.update(predictive_metrics(y_true, y_prob, threshold))
    out.update(fairness_metrics(y_true, y_prob, a, threshold))
    out["ece"] = expected_calibration_error(y_true, y_prob)
    return out


def aggregate_folds(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean +/- std and cross-institution disparity variance over LOIO folds."""
    keys = fold_metrics[0].keys()
    agg: Dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))
    # variance of subgroup disparity across institutions (the IGFAE objective)
    agg["var_delta_dp"] = cross_institution_variance(fold_metrics, "delta_dp")
    agg["var_delta_eo"] = cross_institution_variance(fold_metrics, "delta_eo")
    return agg
