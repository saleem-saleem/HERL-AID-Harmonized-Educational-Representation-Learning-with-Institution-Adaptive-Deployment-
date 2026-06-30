"""State-of-the-art baselines for the direct empirical benchmark (Table 14).

  - DANN       : domain-adversarial NN (gradient reversal) -- domain adaptation
  - Deep CORAL : covariance alignment NN                    -- domain adaptation
  - FairXGBoost: fairness-reweighted gradient boosting      -- fairness-aware
  - AD-DNN     : adversarial-debiasing NN                   -- fairness-aware

Each exposes fit(X, y, inst, a) / predict_proba_pos(X). They are trained under the
SAME leakage-safe LOIO protocol and harmonized feature space as HERL-AID.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Gradient reversal (DANN)
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class _Backbone(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


def _to_t(a, device, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=device)


class DANN:
    def __init__(self, seed=42, epochs=80, lr=1e-3, batch_size=64, lambd=1.0, device="cpu"):
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.batch_size, self.lambd, self.device = batch_size, lambd, device

    def fit(self, X, y, inst, a=None):
        torch.manual_seed(self.seed)
        n_inst = int(np.max(inst)) + 1
        self.bb = _Backbone(X.shape[1]).to(self.device)
        self.cls = nn.Linear(self.bb.out_dim, 2).to(self.device)
        self.dom = nn.Linear(self.bb.out_dim, n_inst).to(self.device)
        opt = torch.optim.Adam(list(self.bb.parameters()) + list(self.cls.parameters())
                               + list(self.dom.parameters()), lr=self.lr)
        X, y, inst = _to_t(X, self.device), _to_t(y, self.device, torch.long), _to_t(inst, self.device, torch.long)
        n = X.size(0)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.batch_size):
                bi = perm[s:s + self.batch_size]
                f = self.bb(X[bi])
                cls_loss = F.cross_entropy(self.cls(f), y[bi])
                dom_loss = F.cross_entropy(self.dom(grad_reverse(f, self.lambd)), inst[bi])
                loss = cls_loss + dom_loss
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict_proba_pos(self, X):
        self.bb.eval(); self.cls.eval()
        X = _to_t(X, self.device)
        return F.softmax(self.cls(self.bb(X)), 1)[:, 1].cpu().numpy()


class DeepCORAL:
    def __init__(self, seed=42, epochs=80, lr=1e-3, batch_size=64, coral_w=1.0, device="cpu"):
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.batch_size, self.coral_w, self.device = batch_size, coral_w, device

    def fit(self, X, y, inst, a=None):
        from ..coral import multi_source_coral
        torch.manual_seed(self.seed)
        self.bb = _Backbone(X.shape[1]).to(self.device)
        self.cls = nn.Linear(self.bb.out_dim, 2).to(self.device)
        opt = torch.optim.Adam(list(self.bb.parameters()) + list(self.cls.parameters()), lr=self.lr)
        Xt, yt, kt = _to_t(X, self.device), _to_t(y, self.device, torch.long), _to_t(inst, self.device, torch.long)
        n = Xt.size(0)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.batch_size):
                bi = perm[s:s + self.batch_size]
                f = self.bb(Xt[bi])
                loss = F.cross_entropy(self.cls(f), yt[bi]) + self.coral_w * multi_source_coral(f, kt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict_proba_pos(self, X):
        self.bb.eval(); self.cls.eval()
        return F.softmax(self.cls(self.bb(_to_t(X, self.device))), 1)[:, 1].cpu().numpy()


class FairXGBoost:
    """Gradient boosting with fairness-aware sample reweighting on the protected group."""
    def __init__(self, seed=42, fairness_strength=1.0):
        self.seed, self.fairness_strength = seed, fairness_strength

    def fit(self, X, y, inst=None, a=None):
        from xgboost import XGBClassifier
        w = np.ones(len(y), dtype=float)
        if a is not None:
            # upweight the disadvantaged subgroup among positives to equalize TPR
            for grp in np.unique(a):
                m = (a == grp) & (y == 1)
                if m.any():
                    w[m] *= 1.0 + self.fairness_strength * (1.0 - m.mean())
        self.model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.9, eval_metric="logloss",
                                   random_state=self.seed, n_jobs=-1)
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict_proba_pos(self, X):
        return self.model.predict_proba(X)[:, 1]


class ADDNN:
    """Adversarial-debiasing DNN: predictor adversarially trained against an
    attribute discriminator so the representation is uninformative of A."""
    def __init__(self, seed=42, epochs=80, lr=1e-3, batch_size=64, adv_w=1.0, device="cpu"):
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.batch_size, self.adv_w, self.device = batch_size, adv_w, device

    def fit(self, X, y, inst=None, a=None):
        torch.manual_seed(self.seed)
        self.bb = _Backbone(X.shape[1]).to(self.device)
        self.cls = nn.Linear(self.bb.out_dim, 2).to(self.device)
        self.adv = nn.Linear(self.bb.out_dim, 2).to(self.device)
        opt = torch.optim.Adam(list(self.bb.parameters()) + list(self.cls.parameters())
                               + list(self.adv.parameters()), lr=self.lr)
        Xt = _to_t(X, self.device)
        yt = _to_t(y, self.device, torch.long)
        at = _to_t(a if a is not None else np.zeros(len(y)), self.device, torch.long)
        n = Xt.size(0)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.batch_size):
                bi = perm[s:s + self.batch_size]
                f = self.bb(Xt[bi])
                cls_loss = F.cross_entropy(self.cls(f), yt[bi])
                adv_loss = F.cross_entropy(self.adv(grad_reverse(f, self.adv_w)), at[bi])
                loss = cls_loss + adv_loss
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict_proba_pos(self, X):
        self.bb.eval(); self.cls.eval()
        return F.softmax(self.cls(self.bb(_to_t(X, self.device))), 1)[:, 1].cpu().numpy()


def get_sota_baselines(seed=42, device="cpu"):
    return {
        "DANN": DANN(seed=seed, device=device),
        "DeepCORAL": DeepCORAL(seed=seed, device=device),
        "FairXGBoost": FairXGBoost(seed=seed),
        "AD-DNN": ADDNN(seed=seed, device=device),
    }
