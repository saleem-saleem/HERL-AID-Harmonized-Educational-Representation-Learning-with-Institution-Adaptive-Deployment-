"""Maximum Mean Discrepancy (MMD) with a multi-bandwidth Gaussian RBF kernel.

The naive estimator evaluates the full N x N Gram matrix at O(N^2) cost. Here MMD
is estimated **within mini-batches** of fixed size B, so each step costs O(B^2 d)
and the cumulative cost over an epoch is O((N/B) * B^2 d) = O(N B d) = O(N d),
i.e. *linear* in the number of students (Reviewer comment on complexity). GPU
memory is bounded by O(B^2), independent of N.
"""
from __future__ import annotations
from typing import Sequence

import torch


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sq = (x ** 2).sum(1, keepdim=True)
    y_sq = (y ** 2).sum(1, keepdim=True)
    return x_sq - 2.0 * x @ y.t() + y_sq.t()


def _multi_rbf(x: torch.Tensor, y: torch.Tensor,
               bandwidths: Sequence[float]) -> torch.Tensor:
    d2 = _pairwise_sq_dists(x, y)
    # median heuristic scale, combined with the supplied relative bandwidths
    with torch.no_grad():
        med = torch.median(d2.detach()) + 1e-8
    k = torch.zeros_like(d2)
    for b in bandwidths:
        k = k + torch.exp(-d2 / (2.0 * b * med))
    return k / len(bandwidths)


def mmd2(source: torch.Tensor, target: torch.Tensor,
         bandwidths: Sequence[float] = (1, 2, 4, 8, 16)) -> torch.Tensor:
    """Biased mini-batch MMD^2 estimate between two batches."""
    k_ss = _multi_rbf(source, source, bandwidths).mean()
    k_tt = _multi_rbf(target, target, bandwidths).mean()
    k_st = _multi_rbf(source, target, bandwidths).mean()
    return k_ss + k_tt - 2.0 * k_st


def multi_source_mmd(features: torch.Tensor, institution: torch.Tensor,
                     bandwidths: Sequence[float] = (1, 2, 4, 8, 16)) -> torch.Tensor:
    """Average pairwise MMD^2 across the institutions present in a mini-batch.

    Encourages institution-invariant representations by aligning the per-institution
    feature distributions within each batch.
    """
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
                total = total + mmd2(fi, fj, bandwidths)
                count += 1
    return total / max(count, 1)
