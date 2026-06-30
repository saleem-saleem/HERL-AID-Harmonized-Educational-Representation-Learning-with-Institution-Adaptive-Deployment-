"""Main experiment driver: leave-one-institution-out evaluation of HERL-AID and all
baselines across repeated seeds, with per-fold leakage-safe harmonization.

Usage:
    python scripts/run_experiments.py --config config/default.yaml --out results/
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, get_device, save_json, logger
from src.data_loader import build_pooled, loio_splits
from src.harmonize import ConstructEncoder, load_feature_dictionary
from src.model import HERLAID
from src.baselines.standard import get_standard_baselines, SklearnProbaWrapper
from src.baselines.sota import get_sota_baselines
from src.metrics import full_evaluation, aggregate_folds
from src.preprocessing import LeakageSafePreprocessor


def evaluate_model_loio(make_model, data, cfg, is_herlaid=False):
    fold_metrics: List[Dict] = []
    for k, tr, te in loio_splits(data):
        Xtr, ytr, ktr, atr = data.X[tr], data.y[tr], data.inst[tr], data.a[tr]
        Xte, yte, ate = data.X[te], data.y[te], data.a[te]
        if is_herlaid:
            model = make_model()
            model.fit(Xtr, ytr, ktr, atr)
            p = model.predict_proba(Xte)
        else:
            # leakage-safe preprocessing for baselines too
            pre = LeakageSafePreprocessor()
            pre.fit(Xtr, ytr)
            Xs = pre.transform_test(Xtr)
            mask = (Xs ** 2).sum(1) <= pre.maha_threshold_
            sw = pre.sample_weights(ytr[mask])
            model = make_model()
            try:
                model.fit(Xs[mask], ytr[mask], ktr[mask], atr[mask])
            except TypeError:
                try:
                    model.fit(Xs[mask], ytr[mask], sample_weight=sw)
                except TypeError:
                    model.fit(Xs[mask], ytr[mask])
            Xte_s = pre.transform_test(Xte)
            if hasattr(model, "predict_proba_pos"):
                p = model.predict_proba_pos(Xte_s)
            else:
                p = model.predict_proba(Xte_s)[:, 1]
        fold_metrics.append(full_evaluation(yte, p, ate))
    return aggregate_folds(fold_metrics), fold_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    device = get_device(cfg.get("device", "cuda"))
    logger.info("device=%s", device)

    n_seeds = cfg.get("evaluation", {}).get("n_seeds", 5)
    fdict_path = os.path.join("data", "feature_mapping_dictionary.csv")

    all_runs = {}
    for seed in range(cfg["seed"], cfg["seed"] + n_seeds):
        set_seed(seed)
        cfg_s = dict(cfg); cfg_s["seed"] = seed
        data = build_pooled(cfg_s, fdict_path)

        runs: Dict[str, Dict] = {}
        # HERL-AID
        agg, _ = evaluate_model_loio(lambda: HERLAID(cfg_s, seed=seed), data, cfg_s, is_herlaid=True)
        runs["HERL-AID"] = agg
        # standard baselines
        for name, model in get_standard_baselines(seed).items():
            agg, _ = evaluate_model_loio(lambda m=model: SklearnProbaWrapper(m), data, cfg_s)
            runs[name] = agg
        # SOTA baselines
        for name, model in get_sota_baselines(seed, device).items():
            agg, _ = evaluate_model_loio(lambda m=model: m, data, cfg_s)
            runs[name] = agg
        all_runs[f"seed_{seed}"] = runs
        logger.info("seed %d done", seed)

    os.makedirs(args.out, exist_ok=True)
    save_json(all_runs, os.path.join(args.out, "main_results.json"))

    # summary table: mean over seeds of the per-fold means
    rows = []
    models = list(next(iter(all_runs.values())).keys())
    for m in models:
        aucs = [all_runs[s][m]["auc_mean"] for s in all_runs]
        accs = [all_runs[s][m]["accuracy_mean"] for s in all_runs]
        f1s = [all_runs[s][m]["macro_f1_mean"] for s in all_runs]
        eos = [all_runs[s][m]["var_delta_eo"] for s in all_runs]
        eces = [all_runs[s][m]["ece_mean"] for s in all_runs]
        rows.append({"model": m,
                     "accuracy": f"{np.mean(accs):.3f}±{np.std(accs):.3f}",
                     "auc": f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}",
                     "macro_f1": f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}",
                     "var_delta_eo": f"{np.mean(eos):.4f}",
                     "ece": f"{np.mean(eces):.3f}"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary_table.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
