"""HERL-AID - end-to-end orchestration of the three stages.

Fitting is leakage-safe: preprocessing, Stage-1 encoder, Stage-2 ensemble, and the
Stage-3 calibrator are all fit on the training institutions only; the held-out
institution is transformed and scored once.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np

from .preprocessing import LeakageSafePreprocessor
from .stage1_mo_iierl import MOIIERL
from .stage2_igfae import IGFAE
from .stage3_ceisd import CEISD
from .utils import get_device


class HERLAID:
    def __init__(self, cfg: Dict, seed: int = 42,
                 use_stage1: bool = True, use_fairness: bool = True,
                 use_calibration: bool = True):
        self.cfg = cfg
        self.seed = seed
        self.use_stage1 = use_stage1          # ablation switch: MO-IIERL
        self.use_fairness = use_fairness      # ablation switch: IGFAE fairness term
        self.use_calibration = use_calibration  # ablation switch: Stage-3 calibration
        self.device = get_device(cfg.get("device", "cuda"))
        self.pre_ = LeakageSafePreprocessor(
            outlier_quantile=cfg.get("preprocessing", {}).get("outlier_quantile", 0.95))
        self.stage1_: Optional[MOIIERL] = None
        self.igfae_: Optional[IGFAE] = None
        self.ceisd_: Optional[CEISD] = None

    def fit(self, X, y, inst, a):
        # 1) leakage-safe preprocessing (fit on TRAIN only)
        self.pre_.fit(X, y)
        Xtr, ytr = self.pre_.transform_train(X, y)
        # realign inst/a after outlier removal
        keep = np.isin(np.arange(len(y)), np.arange(len(y)))  # transform_train keeps order
        # recompute mask explicitly to subset inst/a
        Xs = self.pre_.transform_test(X)
        d2 = (Xs ** 2).sum(1)
        mask = d2 <= self.pre_.maha_threshold_
        Xtr, ytr, ktr, atr = Xs[mask], y[mask], inst[mask], a[mask]
        sw = self.pre_.sample_weights(ytr)

        # 2) Stage-1 representation (or identity if ablated)
        s1cfg = self.cfg.get("stage1_mo_iierl", {})
        if self.use_stage1:
            self.stage1_ = MOIIERL(Xtr.shape[1], s1cfg).fit(
                Xtr, ytr, ktr, atr, sample_weight=sw, device=self.device)
            Ztr = self.stage1_.embed(Xtr, self.device)
        else:
            Ztr = Xtr  # no representation learning

        # 3) Stage-2 fairness-aware ensemble
        s2cfg = dict(self.cfg.get("stage2_igfae", {}))
        if not self.use_fairness:
            s2cfg["kappa2"] = 0.0  # disable fairness weighting (performance-only)
        self.igfae_ = IGFAE(s2cfg, self.seed).fit(Ztr, ytr, atr, sample_weight=sw)

        # 4) Stage-3 calibration on an internal validation split
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(ytr)); cut = int(0.8 * len(ytr))
        va = idx[cut:]
        self.ceisd_ = CEISD(self.cfg.get("stage3_ceisd", {}), self.seed)
        if self.use_calibration and len(va) > 10:
            pv = self.igfae_.predict_proba(Ztr[va])
            self.ceisd_.calibrate(pv, ytr[va])
        return self

    def predict_proba(self, X):
        Xs = self.pre_.transform_test(X)
        Z = self.stage1_.embed(Xs, self.device) if self.use_stage1 else Xs
        p = self.igfae_.predict_proba(Z)
        if self.use_calibration:
            p = self.ceisd_.calibrated_proba(p)
        return p
