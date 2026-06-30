"""Lambda / alpha sensitivity sweeps. Writes results/sensitivity_*.csv."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, set_seed
from src.data_loader import build_pooled
from src.sensitivity import sweep_lambda, sweep_alpha


def main():
    cfg = load_config("config/default.yaml"); set_seed(cfg["seed"])
    data = build_pooled(cfg, "data/feature_mapping_dictionary.csv")
    os.makedirs("results", exist_ok=True)
    lam = pd.DataFrame(sweep_lambda(data, cfg)); lam.to_csv("results/sensitivity_lambda.csv", index=False)
    al = pd.DataFrame(sweep_alpha(data, cfg)); al.to_csv("results/sensitivity_alpha.csv", index=False)
    print("LAMBDA sweep:\n", lam.to_string(index=False))
    print("\nALPHA sweep:\n", al.to_string(index=False))


if __name__ == "__main__":
    main()
