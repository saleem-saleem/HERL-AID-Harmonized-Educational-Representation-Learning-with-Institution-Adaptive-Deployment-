#!/usr/bin/env bash
# Reproduce every table and figure in the paper.
set -e
echo "[1/5] Main LOIO-CV results (HERL-AID + standard + SOTA baselines)"
python scripts/run_experiments.py --config config/default.yaml --out results/
echo "[2/5] Table 14 - SOTA empirical benchmark"
python scripts/make_table14_sota.py
echo "[3/5] Table M - module ablation"
python scripts/make_table_ablation.py
echo "[4/5] Sensitivity sweeps (lambda, alpha)"
python scripts/make_sensitivity.py
echo "[5/5] Complexity check + figures"
python scripts/complexity_analysis.py
python scripts/make_figures.py
echo "Done. See results/ and figures/."
