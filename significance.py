"""Statistical significance testing (Reviewer 3, transparency comment).

Compares HERL-AID against each baseline across repeated seeds/folds using the
paired Wilcoxon signed-rank test, with Holm correction for multiple comparisons
and a rank-biserial effect size.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size for the paired Wilcoxon test."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    pos = (d > 0).sum()
    return float(2 * pos / len(d) - 1)


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def compare_against_baselines(proposed: np.ndarray,
                              baselines: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """proposed / baselines: arrays of a metric over matched seeds*folds."""
    names = list(baselines.keys())
    raw_p, stats = [], {}
    for name in names:
        b = baselines[name]
        try:
            _, p = wilcoxon(proposed, b)
        except ValueError:
            p = 1.0
        raw_p.append(p)
        stats[name] = {
            "mean_proposed": float(np.mean(proposed)),
            "mean_baseline": float(np.mean(b)),
            "raw_p": float(p),
            "effect_size": rank_biserial(proposed, b),
        }
    adj = holm_correction(raw_p)
    for name, ap in zip(names, adj):
        stats[name]["holm_p"] = float(ap)
        stats[name]["significant_0.05"] = bool(ap < 0.05)
    return stats
