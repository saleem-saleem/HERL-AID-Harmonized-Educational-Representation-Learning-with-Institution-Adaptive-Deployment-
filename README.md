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


## 4. Repository layout

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

## 5. Applications

HERL-AID (Harmonized Educational Risk Learning with Alignment, Invariance, and Debiasing) can be applied in a wide range of educational analytics and student-success initiatives:

### 🎓 Early Academic Risk Detection
- Identify students at risk of failure, withdrawal, or non-completion at an early stage.
- Enable timely academic interventions and personalized support.

### 🌐 Cross-Institution Student Performance Prediction
- Transfer predictive knowledge across institutions with different data distributions.
- Improve model generalization when deploying to unseen educational environments.

### ⚖️ Fair and Equitable Learning Analytics
- Reduce demographic disparities in predictive outcomes.
- Support fairness-aware decision-making across student subgroups.

### 📊 Learning Analytics and Educational Intelligence
- Analyze engagement, attendance, assessment behavior, and learning patterns.
- Generate actionable insights for educators and administrators.

### 🏫 Institutional Decision Support
- Assist academic advisors in identifying students requiring additional support.
- Support retention planning and student success initiatives.

### 🔄 Domain Generalization in Educational Data Mining
- Address distribution shifts across universities, colleges, and online learning platforms.
- Learn institution-invariant representations for robust deployment.

### 📈 Student Retention and Dropout Prevention
- Predict potential dropouts before academic disengagement becomes severe.
- Support evidence-based retention strategies.

### 🔍 Explainable Educational AI
- Provide interpretable predictions through SHAP-based explanations.
- Increase transparency and trust in AI-assisted educational decision systems.

### 🚀 Scalable Educational AI Deployment
- Operate on heterogeneous educational datasets with varying feature spaces.
- Deploy on commodity hardware with optional GPU acceleration.

## Potential Stakeholders

- Educational Institutions
- Universities and Colleges
- Academic Advisors
- Learning Analytics Researchers
- Educational Policymakers
- Online Learning Platforms
- Student Success Centers
- EdTech Companies
