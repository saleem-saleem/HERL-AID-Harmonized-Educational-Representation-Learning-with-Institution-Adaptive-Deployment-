"""Deep CORAL: align second-order (covariance) statistics across institutions."""
from __future__ import annotations
import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance between the feature covariances of two batches."""
    d = source.size(1)

    def cov(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        xm = x - x.mean(0, keepdim=True)
        return (xm.t() @ xm) / max(n - 1, 1)

    cs, ct = cov(source), cov(target)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def multi_source_coral(features: torch.Tensor, institution: torch.Tensor) -> torch.Tensor:
    """Average pairwise CORAL loss across institutions in a mini-batch."""
    insts = torch.unique(institution)
    if insts.numel() < 2:
        return features.new_zeros(())
    total = features.new_zeros(())
    count = 0
    for i in range(len(insts)):
        for j in range(i + 1, len(insts)):
            fi = features[institution == insts[i]]
            fj = features[institution == insts[j]]
            if fi.size(0) > 1 and fj.size(0) > 1:
                total = total + coral_loss(fi, fj)
                count += 1
    return total / max(count, 1)
