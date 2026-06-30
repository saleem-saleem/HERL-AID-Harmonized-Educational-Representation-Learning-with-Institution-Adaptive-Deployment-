"""Table M - module ablation. Writes results/table_ablation.csv."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, set_seed
from src.data_loader import build_pooled
from src.ablation import run_ablation


def main():
    cfg = load_config("config/default.yaml"); set_seed(cfg["seed"])
    data = build_pooled(cfg, "data/feature_mapping_dictionary.csv")
    res = run_ablation(data, cfg)
    rows = []
    for variant, agg in res.items():
        rows.append({"variant": variant,
                     "accuracy": round(agg["accuracy_mean"], 3),
                     "auc": round(agg["auc_mean"], 3),
                     "macro_f1": round(agg["macro_f1_mean"], 3),
                     "at_risk_recall": round(agg["at_risk_recall_mean"], 2),
                     "at_risk_f1": round(agg["at_risk_f1_mean"], 2),
                     "var_delta_dp": round(agg["var_delta_dp"], 4),
                     "var_delta_eo": round(agg["var_delta_eo"], 4),
                     "ece": round(agg["ece_mean"], 3)})
    df = pd.DataFrame(rows); os.makedirs("results", exist_ok=True)
    df.to_csv("results/table_ablation.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
