# HERL-AID-Harmonized-Educational-Representation-Learning-with-Institution-Adaptive-Deployment-
HERL-AID: A Fairness-Aware and Institution-Invariant Educational Risk Prediction Framework for Cross-Institution Generalization and Early Academic Intervention.

HERL-AID is a three-stage framework for fair, calibrated, cross-institution student
risk prediction:

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | **MO-IIERL** | Multi-objective institution-invariant representation learning (MMD + CORAL alignment, mutual-information invariance scoring) |
| 2 | **IGFAE** | Institution-generalized fairness-aware ensemble (Equal-Opportunity-variance stabilization across institutions) |
| 3 | **CEISD** | Contextual educational insight & strategy discovery (SHAP attribution + composite Educational Risk Index) |

The framework is evaluated with **leave-one-institution-out cross-validation (LOIO-CV)**
on five public datasets, against **standard** baselines (LR, RF, XGBoost, DNN) and
**state-of-the-art** baselines (DANN, Deep CORAL, FairXGBoost, AD-DNN).

This repository reproduces every table and figure in the paper.

---

## 1. Installation

```bash
git clone https://github.com/<organization>/herl-aid.git
cd herl-aid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or:  conda env create -f environment.yml && conda activate herl-aid
# or:  docker build -t herl-aid . && docker run --gpus all -it herl-aid
```

A GPU is optional — the entire pipeline runs CPU-only (see `--device cpu`).

## 2. Data

This study uses five **publicly available, anonymized** datasets. We do **not**
redistribute them. `data/README.md` lists the download links and SHA-256 checksums;
place the raw files in `data/raw/<dataset>/`. The harmonization, label, and
sensitive-attribute mappings are in:

- `data/feature_mapping_dictionary.csv` — raw attributes → 6 shared construct groups
- `data/label_mapping.csv` — native outcomes → common binary at-risk target `y ∈ {0,1}`
- `data/sensitive_attribute_mapping.csv` — protected attribute `A` (gender) per dataset

## 3. Reproduce everything

```bash
bash scripts/reproduce_all.sh          # runs all experiments, writes results/ + figures/
```

Or run individual pieces:

```bash
python scripts/run_experiments.py --config config/default.yaml   # main LOIO-CV results
python scripts/make_table14_sota.py        # Table 14  — SOTA empirical benchmark
python scripts/make_table_ablation.py      # Table M   — module ablation
python scripts/make_sensitivity.py         # λ / α sensitivity sweeps
python scripts/complexity_analysis.py      # mini-batch MMD scaling (linear in N)
python scripts/make_figures.py             # all figures (>=300 DPI, vector)
```

All randomness is seeded (`--seed`), and every transformation is fit on the training
partition only (leakage-safe; see `src/preprocessing.py`).

## 4. Mapping from reviewer responses to code

| Reviewer point | Code |
|----------------|------|
| Feature harmonization (common schema) | `src/harmonize.py`, `data/feature_mapping_dictionary.csv` |
| Outcome-label mapping (binary at-risk) | `src/data_loader.py`, `data/label_mapping.csv` |
| Sensitive-attribute definition (gender) | `data/sensitive_attribute_mapping.csv` |
| Data-leakage-safe pipeline (fit on train only) | `src/preprocessing.py` |
| SOTA baselines (DANN, Deep CORAL, FairXGBoost, AD-DNN) | `src/baselines/sota.py` |
| Module ablation (Table M) | `src/ablation.py` |
| Hyperparameter sensitivity (λ, α) | `src/sensitivity.py` |
| Statistical significance (Wilcoxon + Holm) | `src/significance.py` |
| Mini-batch MMD, linear complexity | `src/mmd.py`, `scripts/complexity_analysis.py` |
| Three-stage algorithms | `src/stage1_mo_iierl.py`, `src/stage2_igfae.py`, `src/stage3_ceisd.py` |

## 5. Repository layout

```
herl-aid/
├── config/         YAML configs for each stage + defaults
├── data/           mapping dictionaries + download instructions (no raw data)
├── src/            framework, baselines, metrics, ablation, sensitivity, significance
├── scripts/        reproduction / table / figure drivers
├── notebooks/      demo.ipynb
├── tests/          smoke tests on synthetic data
└── results/        generated outputs (csv / json / figures)
```

## 6. Citation

See `CITATION.cff`. If you use this code, please cite the paper.

## 7. License

Released under the MIT License (see `LICENSE`).
