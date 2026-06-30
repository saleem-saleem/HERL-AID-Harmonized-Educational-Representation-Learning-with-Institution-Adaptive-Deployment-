"""Module ablation (Table M) - isolate the contribution of each component.

Variants:
  Full            : complete HERL-AID
  -MO-IIERL       : Stage-1 representation learning disabled (identity features)
  -IGFAE fairness : fairness weighting disabled (performance-only ensemble)
  -Calibration    : Stage-3 temperature scaling disabled

The diagnostic expectations (to be confirmed on real data): removing MO-IIERL
collapses cross-domain AUC; removing IGFAE lowers at-risk recall and worsens
fairness variance; removing calibration inflates ECE.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from .model import HERLAID
from .data_loader import loio_splits, PooledData
from .harmonize import ConstructEncoder, load_feature_dictionary
from .metrics import full_evaluation, aggregate_folds

VARIANTS = {
    "Full": dict(use_stage1=True, use_fairness=True, use_calibration=True),
    "-MO-IIERL": dict(use_stage1=False, use_fairness=True, use_calibration=True),
    "-IGFAE": dict(use_stage1=True, use_fairness=False, use_calibration=True),
    "-Calibration": dict(use_stage1=True, use_fairness=True, use_calibration=False),
}


def run_ablation(data: PooledData, cfg: Dict) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for vname, switches in VARIANTS.items():
        fold_metrics: List[Dict[str, float]] = []
        for k, tr, te in loio_splits(data):
            model = HERLAID(cfg, seed=cfg.get("seed", 42), **switches)
            model.fit(data.X[tr], data.y[tr], data.inst[tr], data.a[tr])
            p = model.predict_proba(data.X[te])
            fold_metrics.append(full_evaluation(data.y[te], p, data.a[te]))
        results[vname] = aggregate_folds(fold_metrics)
    return results
