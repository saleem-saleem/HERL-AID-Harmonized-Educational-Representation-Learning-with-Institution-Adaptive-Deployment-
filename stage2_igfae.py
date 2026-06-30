"""Stage 2 - IGFAE: Institution-Generalized Fairness-Aware Ensemble.

A heterogeneous ensemble of base learners is trained on the Stage-1 representation.
Base-learner weights combine validation performance with a fairness term: learners
whose predictions exhibit lower Equal-Opportunity deviation receive higher weight,
which *stabilizes* subgroup disparity (minimizes its cross-institution variance)
without imposing hard Demographic Parity. The fairness weight is ``kappa2``.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from .metrics import equal_opportunity_diff

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def _make_base_learners(names: List[str], seed: int) -> Dict[str, object]:
    pool = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed),
        "gbm": GradientBoostingClassifier(random_state=seed),
    }
    if _HAS_XGB:
        pool["xgboost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                        subsample=0.9, eval_metric="logloss",
                                        random_state=seed, n_jobs=-1)
    return {n: pool[n] for n in names if n in pool}


class IGFAE:
    def __init__(self, cfg: Dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.kappa2 = cfg.get("kappa2", 0.3)
        self.temperature = cfg.get("weight_temperature", 1.0)
        self.learners_: Dict[str, object] = {}
        self.weights_: Dict[str, float] = {}

    def fit(self, Z, y, a, sample_weight=None):
        names = self.cfg.get("base_learners", ["logreg", "random_forest", "xgboost", "mlp", "gbm"])
        self.learners_ = _make_base_learners(names, self.seed)
        # internal validation split (carved from TRAIN) for fairness-aware weighting
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, va = idx[:cut], idx[cut:]
        scores = {}
        for name, model in self.learners_.items():
            try:
                if sample_weight is not None and hasattr(model, "fit"):
                    try:
                        model.fit(Z[tr], y[tr], sample_weight=sample_weight[tr])
                    except TypeError:
                        model.fit(Z[tr], y[tr])
                else:
                    model.fit(Z[tr], y[tr])
                p = model.predict_proba(Z[va])[:, 1]
                auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else 0.5
                eo = equal_opportunity_diff(y[va], (p >= 0.5).astype(int), a[va])
                # higher AUC and lower EO deviation -> higher score
                scores[name] = auc - self.kappa2 * eo
                # refit on full training data
                try:
                    if sample_weight is not None:
                        model.fit(Z, y, sample_weight=sample_weight)
                    else:
                        model.fit(Z, y)
                except TypeError:
                    model.fit(Z, y)
            except Exception:
                scores[name] = -1e9
        # softmax over scores -> ensemble weights
        s = np.array(list(scores.values())) / max(self.temperature, 1e-6)
        s = s - s.max()
        w = np.exp(s); w = w / w.sum()
        self.weights_ = dict(zip(scores.keys(), w))
        return self

    def predict_proba(self, Z) -> np.ndarray:
        agg = np.zeros(len(Z))
        for name, model in self.learners_.items():
            agg += self.weights_[name] * model.predict_proba(Z)[:, 1]
        return agg
