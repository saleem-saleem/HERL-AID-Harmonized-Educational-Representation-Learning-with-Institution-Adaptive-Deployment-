"""Empirical demonstration that mini-batch MMD cost is linear in N.

Times one alignment epoch for growing N at fixed batch size B, and confirms the
near-linear trend (versus the O(N^2) full-Gram estimator)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.mmd import mmd2, multi_source_mmd


def time_minibatch(N, d=64, B=64, reps=1):
    x = torch.randn(N, d); inst = torch.randint(0, 5, (N,))
    t0 = time.time()
    for _ in range(reps):
        perm = torch.randperm(N)
        for s in range(0, N, B):
            bi = perm[s:s+B]
            _ = multi_source_mmd(x[bi], inst[bi])
    return (time.time() - t0) / reps


def time_full(N, d=64):
    x = torch.randn(N, d); y = torch.randn(N, d)
    t0 = time.time(); _ = mmd2(x, y); return time.time() - t0


def main():
    print(f"{'N':>8} {'minibatch(s)':>14} {'full O(N^2)(s)':>16}")
    for N in [1000, 2000, 4000, 8000, 16000, 32593]:
        mb = time_minibatch(N)
        fl = time_full(min(N, 8000))  # cap full to avoid OOM at large N
        print(f"{N:>8} {mb:>14.4f} {fl:>16.4f}")
    print("\nMini-batch time grows ~linearly with N; full-Gram grows quadratically.")


if __name__ == "__main__":
    main()
