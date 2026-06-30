"""Data loading, label/attribute harmonization, and leave-one-institution-out splits.

Because raw datasets are not redistributed, this module (a) loads each raw dataset
from ``data/raw/<name>/`` when present, applying the label/attribute mappings, and
(b) provides a synthetic-data generator so the pipeline and tests run end-to-end
without the proprietary downloads. Replace ``_load_raw`` with real parsers once the
raw files are in place.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from .harmonize import ConstructEncoder, load_feature_dictionary

DATASETS = ["oulad", "uci", "xapi", "hesp", "woc2"]
# representative sizes (OULAD is the largest, 32,593 students)
_SYNTH_SIZES = {"oulad": 32593, "uci": 1044, "xapi": 480, "hesp": 145, "woc2": 600}


@dataclass
class PooledData:
    X: np.ndarray            # harmonized features  (N x D)
    y: np.ndarray            # binary at-risk target (N,)
    a: np.ndarray            # protected attribute   (N,)  gender in {0,1}
    inst: np.ndarray         # institution index     (N,)
    dataset_names: List[str]


def _load_raw(name: str, data_dir: str) -> pd.DataFrame | None:
    """Hook for real parsers. Return None to fall back to synthetic data."""
    # Implement per-dataset CSV parsing + label_mapping/sensitive_attribute_mapping
    # application here once data/raw/<name>/ is populated.
    return None


def _synthetic(name: str, seed: int) -> pd.DataFrame:
    """Generate schema-faithful synthetic data so the code runs without raw files.

    Columns follow the construct dictionary; a latent risk signal links engagement,
    prior academics, and a (mild, intentional) gender effect to motivate fairness.
    """
    rng = np.random.default_rng(abs(hash(name)) % (2**32) + seed)
    n = _SYNTH_SIZES[name]
    gender = rng.integers(0, 2, n)                       # 1=Female, 0=Male
    prior = rng.normal(0, 1, n)
    engagement = rng.normal(0, 1, n)
    assessment = rng.normal(0, 1, n)
    attendance = rng.normal(0, 1, n)
    support = rng.normal(0, 1, n)
    # latent at-risk logit; small unfair gender term the framework must neutralize
    logit = (-0.9 * prior - 0.8 * engagement - 0.5 * assessment
             - 0.3 * attendance - 0.2 * support + 0.25 * (gender == 1) + rng.normal(0, 0.5, n))
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({
        "gender": gender, "age": rng.integers(17, 40, n),
        "studied_credits": prior + rng.normal(0, 0.3, n),
        "G1": prior + rng.normal(0, 0.3, n), "G2": prior + rng.normal(0, 0.3, n),
        "sum_click_resource": engagement + rng.normal(0, 0.3, n),
        "raisedhands": engagement + rng.normal(0, 0.3, n),
        "assessment_score": assessment + rng.normal(0, 0.3, n),
        "coursework_marks": assessment + rng.normal(0, 0.3, n),
        "absences": -attendance + rng.normal(0, 0.3, n),
        "attendance": attendance + rng.normal(0, 0.3, n),
        "famsup": support + rng.normal(0, 0.3, n),
        "_at_risk": y,
    })
    return df


def build_pooled(config: dict, feature_dict_path: str, data_dir: str = "data") -> PooledData:
    fdict = load_feature_dictionary(feature_dict_path)
    common_dim = config.get("common_dim", 64)
    seed = config.get("seed", 42)
    Xs, ys, as_, insts, names = [], [], [], [], []
    for idx, name in enumerate(config.get("datasets", DATASETS)):
        df = _load_raw(name, data_dir)
        if df is None:
            df = _synthetic(name, seed)
        y = df["_at_risk"].to_numpy().astype(int)
        a = df["gender"].to_numpy().astype(int)
        enc = ConstructEncoder(common_dim=common_dim, seed=seed)
        X = enc.fit_transform(df, fdict, name)   # NOTE: re-fit per fold in training (see run_experiments)
        Xs.append(X); ys.append(y); as_.append(a)
        insts.append(np.full(len(df), idx)); names.append(name)
    return PooledData(
        X=np.concatenate(Xs), y=np.concatenate(ys), a=np.concatenate(as_),
        inst=np.concatenate(insts), dataset_names=names,
    )


def loio_splits(data: PooledData) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield (held_out_institution, train_idx, test_idx) for each institution."""
    for k in np.unique(data.inst):
        test_idx = np.where(data.inst == k)[0]
        train_idx = np.where(data.inst != k)[0]
        yield int(k), train_idx, test_idx
