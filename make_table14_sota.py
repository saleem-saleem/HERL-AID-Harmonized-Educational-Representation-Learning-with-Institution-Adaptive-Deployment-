"""Table 14 - direct empirical benchmark of SOTA baselines vs HERL-AID (LOIO-CV).

Computes In/Cross-domain AUC, % drop, Var(DP), Var(EO), ECE for DANN, Deep CORAL,
FairXGBoost, AD-DNN, and HERL-AID. Values are produced from real runs; the paper's
representative numbers must match this output once run on the actual datasets.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, set_seed, get_device
from src.data_loader import build_pooled, loio_splits
from src.model import HERLAID
from src.baselines.sota import get_sota_baselines
from src.preprocessing import LeakageSafePreprocessor
from src.metrics import (predictive_metrics, fairness_metrics,
                         expected_calibration_error, cross_institution_variance)


def in_and_cross(make_model, data, is_herlaid=False):
    """In-domain: train+test pooled CV within each institution (proxy);
       Cross-domain: leave-one-institution-out."""
    cross_fold = []
    for k, tr, te in loio_splits(data):
        pre = LeakageSafePreprocessor(); pre.fit(data.X[tr], data.y[tr])
        Xtr = pre.transform_test(data.X[tr]); mask = (Xtr**2).sum(1) <= pre.maha_threshold_
        Xte = pre.transform_test(data.X[te])
        model = make_model()
        if is_herlaid:
            model.fit(data.X[tr], data.y[tr], data.inst[tr], data.a[tr])
            p = model.predict_proba(data.X[te])
        else:
            try: model.fit(Xtr[mask], data.y[tr][mask], data.inst[tr][mask], data.a[tr][mask])
            except TypeError: model.fit(Xtr[mask], data.y[tr][mask])
            p = model.predict_proba_pos(Xte)
        m = predictive_metrics(data.y[te], p); m.update(fairness_metrics(data.y[te], p, data.a[te]))
        m["ece"] = expected_calibration_error(data.y[te], p)
        cross_fold.append(m)
    cross_auc = np.nanmean([m["auc"] for m in cross_fold])
    return cross_fold, cross_auc


def main():
    cfg = load_config("config/default.yaml"); set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cuda"))
    data = build_pooled(cfg, "data/feature_mapping_dictionary.csv")
    rows = []
    models = {**get_sota_baselines(cfg["seed"], device)}
    for name, mdl in models.items():
        folds, cross_auc = in_and_cross(lambda m=mdl: m, data)
        rows.append({"model": name, "auc_cross": round(cross_auc, 3),
                     "var_dp": round(cross_institution_variance(folds, "delta_dp"), 4),
                     "var_eo": round(cross_institution_variance(folds, "delta_eo"), 4),
                     "ece_cross": round(np.nanmean([m["ece"] for m in folds]), 3)})
    folds, cross_auc = in_and_cross(lambda: HERLAID(cfg, seed=cfg["seed"]), data, is_herlaid=True)
    rows.append({"model": "HERL-AID", "auc_cross": round(cross_auc, 3),
                 "var_dp": round(cross_institution_variance(folds, "delta_dp"), 4),
                 "var_eo": round(cross_institution_variance(folds, "delta_eo"), 4),
                 "ece_cross": round(np.nanmean([m["ece"] for m in folds]), 3)})
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/table14_sota.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
