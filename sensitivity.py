"""Hyperparameter-sensitivity sweeps (Reviewer 1, sensitivity comment).

Sweeps the Stage-1 distribution-alignment weight ``lambda_align`` and the
cross-institution stability weight ``alpha`` over logarithmic-ish grids, holding
all else at the defaults, under the same LOIO protocol. Produces the data behind
the two sensitivity tables/plots; the expected pattern is a broad stable plateau
around the defaults (lambda=0.5, alpha=0.3).
"""
from __future__ import annotations
import copy
from typing import Dict, List

import numpy as np

from .model import HERLAID
from .data_loader import loio_splits, PooledData
from .metrics import full_evaluation, aggregate_folds

LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
ALPHA_GRID = [0.0, 0.1, 0.3, 0.5, 1.0]


def _run_once(data: PooledData, cfg: Dict) -> Dict[str, float]:
    fold_metrics: List[Dict[str, float]] = []
    per_inst_auc = []
    for k, tr, te in loio_splits(data):
        model = HERLAID(cfg, seed=cfg.get("seed", 42))
        model.fit(data.X[tr], data.y[tr], data.inst[tr], data.a[tr])
        p = model.predict_proba(data.X[te])
        m = full_evaluation(data.y[te], p, data.a[te])
        fold_metrics.append(m)
        per_inst_auc.append(m["auc"])
    agg = aggregate_folds(fold_metrics)
    agg["auc_cross_std"] = float(np.nanstd(per_inst_auc))
    return agg


def sweep_lambda(data: PooledData, base_cfg: Dict) -> List[Dict]:
    rows = []
    for lam in LAMBDA_GRID:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("stage1_mo_iierl", {})["lambda_align"] = lam
        agg = _run_once(data, cfg)
        rows.append({"lambda": lam, "auc_cross_mean": agg["auc_mean"],
                     "var_delta_eo": agg["var_delta_eo"]})
    return rows


def sweep_alpha(data: PooledData, base_cfg: Dict) -> List[Dict]:
    rows = []
    for al in ALPHA_GRID:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("stage1_mo_iierl", {})["alpha"] = al
        agg = _run_once(data, cfg)
        rows.append({"alpha": al, "auc_mean": agg["auc_mean"],
                     "auc_cross_std": agg["auc_cross_std"],
                     "var_delta_eo": agg["var_delta_eo"]})
    return rows
