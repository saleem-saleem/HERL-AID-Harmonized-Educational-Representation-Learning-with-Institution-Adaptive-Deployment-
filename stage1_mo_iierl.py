"""Stage 1 - MO-IIERL: Multi-Objective Institution-Invariant Representation Learning.

Implements the encoder Z = g(x~) and the multi-objective Stage-1 objective (Eq. 40):

    L = -I(Z;Y) + beta * I(Z;K) + alpha * var_k I(Z;Y|k)
        + lambda * D_shift + eta * I(Z;A) + gamma * ||W||^2

operationalized as:
  -I(Z;Y)            -> supervised cross-entropy (class-weighted)
  beta * I(Z;K)      -> institution-adversarial / MMD penalty (institution-invariance)
  alpha * var_k ...  -> variance of per-institution loss (cross-institution stability)
  lambda * D_shift   -> mini-batch MMD + CORAL alignment
  eta * I(Z;A)       -> representation-attribute independence penalty (fairness)
  gamma * ||W||^2    -> weight decay

Mutual-information invariance scores rank features prior to training.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif

from .mmd import multi_source_mmd
from .coral import multi_source_coral


# --------------------------------------------------------------------------- #
# Invariance feature scoring (fit on TRAINING data only)
# --------------------------------------------------------------------------- #
def invariance_scores(X: np.ndarray, y: np.ndarray, inst: np.ndarray, a: np.ndarray,
                      beta: float, alpha: float, eta: float,
                      seed: int = 42) -> np.ndarray:
    """II_j = MI_j(Y) - beta*MI_j(K) - alpha*var_j - eta*MI_j(A)."""
    mi_y = mutual_info_classif(X, y, random_state=seed)
    mi_k = mutual_info_classif(X, inst, random_state=seed) if len(np.unique(inst)) > 1 else np.zeros(X.shape[1])
    mi_a = mutual_info_classif(X, a, random_state=seed) if len(np.unique(a)) > 1 else np.zeros(X.shape[1])
    var_j = X.var(0)
    var_j = var_j / (var_j.max() + 1e-8)
    return mi_y - beta * mi_k - alpha * var_j - eta * mi_a


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class MOIIERL(nn.Module):
    """Encoder + linear classifier head trained with the multi-objective loss."""

    def __init__(self, in_dim: int, cfg: Dict):
        super().__init__()
        self.encoder = Encoder(in_dim, tuple(cfg.get("encoder_hidden", (256, 128, 64))),
                               cfg.get("dropout", 0.2))
        self.classifier = nn.Linear(self.encoder.out_dim, 2)
        # head predicting the protected attribute, used to *penalize* its information
        self.attr_head = nn.Linear(self.encoder.out_dim, 2)
        self.cfg = cfg

    def forward(self, x):
        z = self.encoder(x)
        return z, self.classifier(z), self.attr_head(z)

    def fit(self, X, y, inst, a, sample_weight=None, device="cpu"):
        cfg = self.cfg
        self.to(device)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        inst = torch.tensor(inst, dtype=torch.long, device=device)
        a = torch.tensor(a, dtype=torch.long, device=device)
        sw = (torch.tensor(sample_weight, dtype=torch.float32, device=device)
              if sample_weight is not None else torch.ones_like(y, dtype=torch.float32))

        opt = torch.optim.Adam(self.parameters(), lr=cfg.get("lr", 1e-3),
                               weight_decay=cfg.get("gamma", 1e-4))
        B = cfg.get("batch_size", 64)
        bandwidths = cfg.get("mmd_bandwidths", (1, 2, 4, 8, 16))
        lam = cfg.get("lambda_align", 0.5)
        alpha = cfg.get("alpha", 0.3)
        beta = cfg.get("beta", 0.5)
        eta = cfg.get("eta", 0.2)
        coral_w = cfg.get("coral_weight", 0.5)
        best, patience, wait = float("inf"), cfg.get("early_stopping_patience", 10), 0
        n = X.size(0)

        for _ in range(cfg.get("epochs", 100)):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            for s in range(0, n, B):
                bi = perm[s:s + B]
                xb, yb, kb, ab, wb = X[bi], y[bi], inst[bi], a[bi], sw[bi]
                z, logits, attr_logits = self(xb)

                # -I(Z;Y): class-weighted CE
                ce = (F.cross_entropy(logits, yb, reduction="none") * wb).mean()

                # lambda * D_shift : MMD + CORAL across institutions in the batch
                d_shift = multi_source_mmd(z, kb, bandwidths) + coral_w * multi_source_coral(z, kb)

                # alpha * var_k : variance of per-institution loss (stability)
                inst_losses = []
                for k in torch.unique(kb):
                    m = kb == k
                    if m.sum() > 0:
                        inst_losses.append(F.cross_entropy(logits[m], yb[m]))
                var_k = torch.stack(inst_losses).var() if len(inst_losses) > 1 else z.new_zeros(())

                # eta * I(Z;A) : penalize ability to predict the protected attribute
                #   (minimize attribute predictability -> negative attr-CE as penalty)
                attr_ce = F.cross_entropy(attr_logits, ab)
                attr_penalty = -attr_ce  # minimizing L increases attr_ce => Z less informative of A

                loss = ce + lam * d_shift + alpha * var_k + eta * attr_penalty
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += float(loss.detach())
            if epoch_loss < best - 1e-4:
                best, wait = epoch_loss, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return self

    @torch.no_grad()
    def embed(self, X, device="cpu"):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        return self.encoder(X).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X, device="cpu"):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        return F.softmax(self.classifier(self.encoder(X)), dim=1).cpu().numpy()
