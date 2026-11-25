# Subgroup-Calibrated Geometric Conformal Plausibility Agent
## Reproducibility Package

This package contains all code, data, and configurations needed to reproduce the results from the paper:

> Subgroup-Calibrated Geometric Conformal Plausibility Agent: An Explainable Agentic AI Framework for Fair and Distribution-Free Diabetes Risk Assessment

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (see Data Preparation below)
# 3. Run full pipeline
python -m src.run_all --config configs/config.yaml

# 4. Generate paper figures and tables
python -m src.generate_paper --config configs/config.yaml
```

### Directory Structure

```
reproducibility_package/
├── src/                    # Source code
├── data/
│   ├── nhanes/             # NHANES cohort & embeddings
│   ├── mimic_iv_demo/      # MIMIC-IV demo cohort
│   └── brfss_raw/          # BRFSS 2022 data
├── results/
│   ├── figures/            # Generated figures (PNG, 300 DPI)
│   ├── tables/             # LaTeX tables for paper
│   └── ablation/           # Ablation study results
├── configs/
│   └── config.yaml         # Pipeline configuration
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
├── config.yaml             # Pipeline configuration
├── run_all.sh              # Run entire pipeline
└── README.md               # This file
```

### Data Preparation

#### NHANES (Primary)
1. Download NHANES 2017-2018 XPT files from CDC
2. Place in `data/nhanes/`
3. Run: `python -m src.cohort --dataset nhanes`

#### MIMIC-IV Demo (Pipeline Testing)
1. Download from PhysioNet: https://physionet.org/files/mimic-iv-demo/2.2/
2. Place in `data/mimic_iv_demo/`
3. Run: `python -m src.cohort --dataset mimic_demo`

#### MIMIC-IV Full (Production)
1. Apply for PhysioNet access (CITI training + DUA)
2. Download MIMIC-IV v3.1 from PhysioNet
3. Use `mimic_iv_full_cohort.py` for full cohort construction

#### BRFSS (Supplementary)
- Already included in `data/brfss_raw/brfss_modeling_final.csv`

### Pipeline Stages

```bash
# Stage 1: Cohort Construction
python -m src.cohort --config configs/config.yaml

# Stage 2: Embedding
python -m src.embedding --config configs/config.yaml

# Stage 3: Conformal Calibration
python -m src.conformal --config configs/config.yaml

# Stage 4: Counterfactual Geometry
python -m src.counterfactual --config configs/config.yaml

# Stage 5: Attribution Significance
python -m src.attribution --config configs/config.yaml

# Stage 6: Auditor & Translator
python -m src.auditor --config configs/config.yaml
python -m src.translator --config configs/config.yaml

# Stage 7: Ablation Studies
python -m src.ablation --config configs/config.yaml

# Stage 8: Paper Generation
python -m src.generate_paper --config configs/config.yaml
```

Or run everything at once:
```bash
./run_all.sh
```

### Expected Results

| Metric | NHANES | MIMIC-IV Demo |
|--------|--------|---------------|
| Overall Coverage | 90.7% | 89.5% |
| Subgroup Coverage Range | 86-94% | 50-100% |
| Counterfactual Reach | 100% | 100% |
| Inside Ellipsoid | 100% | 100% |
| Abstention Rate | 0% | 100% |

### Configuration

All hyperparameters in `configs/config.yaml`:
- `conformal.alpha`: Significance level (default 0.1)
- `conformal.method`: `cv_plus`, `single_split`, `jackknife_plus`
- `conformal.hierarchical_shrinkage`: Enable hierarchical shrinkage
- `counterfactual.ellipsoid_multiplier`: Mahalanobis radius multiplier (default 2.0)
- `counterfactual.n_restarts`: Number of random restarts (default 5)
- `auditor.coverage_tolerance`: Max subgroup coverage gap (default 0.05)

### Reproducibility

All random seeds fixed (`random_state=42`).

### Paper Generation

```bash
python -m src.generate_paper --config configs/config.yaml
```

Generates:
- `results/tables/paper_tables.tex` - 3 LaTeX tables
- `results/figures/fig1_cohort_characteristics.png`
- `results/figures/fig2_conformal_results.png`
- `results/figures/fig3_counterfactuals.png`
- `results/figures/fig4_ablation_table.png`
- `results/figures/fig5_calibration_curves.png`
- `results/figures/fig6_feature_importance.png`
- `results/figures/fig7_mahalanobis.png`
- `results/figures/fig8_prediction_sets.png`
- `results/figures/fig9_coverage_heatmap.png`
- `paper_manuscript.tex` - Full LaTeX manuscript
- `paper_manuscript.md` - Markdown version
- `slides.md` - 15-slide presentation (Marp/reveal.js compatible)

### Citation

```bibtex
@article{conformal_plausibility_agent_2024,
  title={Subgroup-Calibrated Geometric Conformal Plausibility Agent: An Explainable Agentic AI Framework for Fair and Distribution-Free Diabetes Risk Assessment},
  author={Anonymous Authors},
  journal={Under Review},
  year={2024}
}
```

### License

MIT License

### Contact

Research Team: research@your-institution.edu
GitHub: github.com/your-repo/conformal-plausibility-agent
