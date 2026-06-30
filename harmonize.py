"""Cross-dataset feature harmonization (Reviewer Comment 2).

Heterogeneous raw feature spaces are mapped onto a shared, pedagogically grounded
schema of six construct groups (C1-C6) using ``feature_mapping_dictionary.csv``.
Within each dataset, the raw attributes assigned to a construct are standardized
and encoded into a fixed-length construct embedding; concatenating the six
embeddings yields a harmonized representation x~ in R^D of identical dimensionality
for every student. MMD / CORAL alignment then operate on this common space, never
on the raw inputs (which would be ill-posed across disjoint feature spaces).
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

CONSTRUCTS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def load_feature_dictionary(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _construct_columns(fdict: pd.DataFrame, construct: str, dataset: str) -> List[str]:
    row = fdict[fdict["construct_group"] == construct]
    if row.empty or dataset not in fdict.columns:
        return []
    cell = row.iloc[0][dataset]
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [c.strip() for c in str(cell).split(";") if c.strip()]


class ConstructEncoder:
    """Per-dataset linear encoder that maps each construct's raw features to a
    fixed-length embedding. Fit on TRAINING data only (leakage-safe)."""

    def __init__(self, common_dim: int = 64, seed: int = 42):
        self.common_dim = common_dim
        self.per_construct_dim = max(common_dim // len(CONSTRUCTS), 1)
        self.seed = seed
        self.projections_: Dict[str, np.ndarray] = {}
        self.means_: Dict[str, np.ndarray] = {}
        self.stds_: Dict[str, np.ndarray] = {}
        self.columns_: Dict[str, List[str]] = {}

    def _matrix(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        present = [c for c in cols if c in df.columns]
        if not present:
            return np.zeros((len(df), 0))
        return df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    def fit(self, df: pd.DataFrame, fdict: pd.DataFrame, dataset: str) -> "ConstructEncoder":
        rng = np.random.default_rng(self.seed)
        for c in CONSTRUCTS:
            cols = _construct_columns(fdict, c, dataset)
            self.columns_[c] = cols
            x = self._matrix(df, cols)
            if x.shape[1] == 0:
                self.projections_[c] = np.zeros((0, self.per_construct_dim))
                self.means_[c] = np.zeros(0)
                self.stds_[c] = np.ones(0)
                continue
            self.means_[c] = x.mean(0)
            self.stds_[c] = x.std(0) + 1e-8
            # random linear projection to a fixed per-construct dimension
            self.projections_[c] = rng.normal(
                0.0, 1.0 / np.sqrt(max(x.shape[1], 1)),
                size=(x.shape[1], self.per_construct_dim),
            )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        embeddings = []
        avail_flags = []
        for c in CONSTRUCTS:
            cols = self.columns_.get(c, [])
            x = self._matrix(df, cols)
            if x.shape[1] == 0 or self.projections_[c].shape[0] == 0:
                emb = np.zeros((len(df), self.per_construct_dim))
                avail_flags.append(np.zeros((len(df), 1)))
            else:
                xs = (x - self.means_[c]) / self.stds_[c]
                emb = xs @ self.projections_[c]
                avail_flags.append(np.ones((len(df), 1)))
            embeddings.append(emb)
        harmonized = np.concatenate(embeddings + avail_flags, axis=1)
        return harmonized.astype(np.float32)

    def fit_transform(self, df, fdict, dataset) -> np.ndarray:
        return self.fit(df, fdict, dataset).transform(df)
