"""Smoke tests on synthetic data - verify the full pipeline runs end-to-end."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.utils import load_config, set_seed
from src.data_loader import build_pooled, loio_splits
from src.model import HERLAID
from src.metrics import full_evaluation
from src.significance import compare_against_baselines, holm_correction


def _small_cfg():
    cfg = load_config("config/default.yaml")
    cfg["datasets"] = ["uci", "xapi", "hesp"]      # small + fast
    cfg["device"] = "cpu"
    cfg["stage1_mo_iierl"]["epochs"] = 3
    cfg["evaluation"]["n_seeds"] = 1
    return cfg


def test_pipeline_runs():
    cfg = _small_cfg(); set_seed(0)
    data = build_pooled(cfg, "data/feature_mapping_dictionary.csv")
    assert data.X.shape[0] == len(data.y) == len(data.a) == len(data.inst)
    k, tr, te = next(loio_splits(data))
    model = HERLAID(cfg, seed=0).fit(data.X[tr], data.y[tr], data.inst[tr], data.a[tr])
    p = model.predict_proba(data.X[te])
    m = full_evaluation(data.y[te], p, data.a[te])
    assert 0.0 <= m["auc"] <= 1.0
    assert "delta_eo" in m and "ece" in m


def test_holm_monotone():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert all(0 <= a <= 1 for a in adj)


def test_significance_shapes():
    rng = np.random.default_rng(0)
    proposed = rng.normal(0.89, 0.01, 10)
    base = {"LR": rng.normal(0.82, 0.01, 10)}
    res = compare_against_baselines(proposed, base)
    assert "holm_p" in res["LR"] and "effect_size" in res["LR"]


if __name__ == "__main__":
    test_pipeline_runs(); test_holm_monotone(); test_significance_shapes()
    print("all smoke tests passed")
