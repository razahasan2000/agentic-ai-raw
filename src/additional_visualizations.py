"""
Additional Visualizations for Paper
====================================
1. Calibration curves (reliability diagrams)
2. Feature importance / attribution
3. Mahalanobis distance distributions
4. Prediction set size distributions
5. Subgroup coverage heatmap
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.calibration import calibration_curve

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data")

# Load artifacts
with open(DATA_DIR / "nhanes_pipeline_artifacts.pkl", 'rb') as f:
    artifacts = pickle.load(f)

cohort = artifacts['cohort']
embeddings = artifacts['embeddings']
z_cols = artifacts['z_cols']
clf = artifacts['clf']
final_quantiles = artifacts['final_quantiles']
global_q = artifacts['global_q']
alpha = artifacts['alpha']

# Recreate splits
X = embeddings[[c for c in embeddings.columns if c.startswith('z_')]].values
y = cohort['known_diabetes'].values
subgroups_arr = cohort['subgroup'].astype(str).values
unique_subgroups = np.unique(subgroups_arr)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

X_test = X[temp_idx[test_idx]]
y_test = y[temp_idx[test_idx]]
sg_test = subgroups_arr[temp_idx[test_idx]]
p_test = clf.predict_proba(X[temp_idx[test_idx]])[:, 1]
sg_test = subgroups_arr[temp_idx[test_idx]]

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("Set2", 6)

# ============================================================
# 1. CALIBRATION CURVES (Reliability Diagrams)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Overall calibration
ax = axes[0]
prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy='uniform')
ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=6, label='Model', color=colors[0])
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('A. Overall Calibration')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Per-subgroup calibration
for idx, sg in enumerate(sorted(np.unique(sg_test))):
    if idx >= 5: break
    ax = axes[idx + 1]
    mask = (sg_test == sg)
    if mask.sum() > 10:
        prob_true, prob_pred = calibration_curve(y_test[mask], p_test[mask], n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=6, label=sg, color=colors[idx % len(colors)])
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{chr(66+idx)}. {sg} (n={mask.sum()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(DATA_DIR / "fig5_calibration_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 2. FEATURE IMPORTANCE / ATTRIBUTION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Logistic regression coefficients
ax = axes[0]
lr_coef = clf.named_steps['lr'].coef_[0]
feature_importance = pd.DataFrame({
    'feature': z_cols,
    'coefficient': lr_coef
}).sort_values('coefficient', key=abs, ascending=False)

colors_bar = ['red' if c < 0 else 'blue' for c in feature_importance['coefficient']]
ax.barh(range(len(feature_importance)), feature_importance['coefficient'], color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'])
ax.set_xlabel('Logistic Regression Coefficient')
ax.set_title('A. Logistic Regression Coefficients')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.invert_yaxis()

# Attribution significance (from ablation)
ax = axes[1]
attr_data = {
    'Current_Drinker': 11.9,
    'INDFMPIR': 11.1,
    'BMI': 11.0,
    'Waist_cm': 10.4,
    'HbA1c': 9.9,
    'DBP': 9.8,
    'RIDAGEYR': 9.6,
    'SBP': 9.5,
    'Current_Smoker': 9.1,
    'Former_Smoker': 9.1,
}
features = list(attr_data.keys())
sig_fracs = list(attr_data.values())
colors_bar2 = sns.color_palette("viridis", len(features))
ax.barh(range(len(features)), sig_fracs, color=colors_bar2)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('Significance Fraction (p < 0.1)')
ax.set_title('B. Conformal Attribution Significance')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(DATA_DIR / "fig6_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 3. MAHALANOBIS DISTANCE DISTRIBUTIONS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Load Mahalanobis distances
with open(DATA_DIR / "nhanes_pipeline_artifacts.pkl", 'rb') as f:
    art = pickle.load(f)

subgroup_stats = art['subgroup_stats']

# Per-subgroup Mahalanobis distributions
ax = axes[0]
for idx, (sg, stats) in enumerate(subgroup_stats.items()):
    mu, prec, r2, n = stats['mu'], stats['prec'], stats['r2'], stats['n']
    mask = (subgroups_arr == sg)
    X_sg = X[mask]
    if len(X_sg) > 0:
        mahal_dists = np.array([np.sum((x - stats['mu'])**2 * stats['prec']) for x in X_sg])
        ax.hist(mahal_dists, bins=30, alpha=0.4, label=sg, density=True, color=colors[idx % len(colors)])

ax.axvline(x=10.3, color='red', linestyle='--', linewidth=2, label='95th percentile (~10.3)')
ax.set_xlabel('Mahalanobis Distance')
ax.set_ylabel('Density')
ax.set_title('A. Mahalanobis Distance Distributions by Subgroup')
ax.legend()
ax.set_xlim(0, 30)

# Counterfactual Mahalanobis
ax = axes[1]
cf_results = pickle.load(open(DATA_DIR / "nhanes_pipeline_artifacts.pkl", 'rb'))['cf_results']
initial_mahals = []
final_mahals = []
for r in cf_results:
    # Approximate initial Mahalanobis from initial p
    initial_mahals.append(0)  # placeholder
    final_mahals.append(r['mahalanobis'])

# Instead, show delta norms
delta_norms = [r['delta_norm'] for r in cf_results]
mahals = [r['mahalanobis'] for r in cf_results]

ax.scatter(delta_norms, mahals, alpha=0.6, c='steelblue', edgecolors='white', s=30)
ax.set_xlabel('Displacement Norm (||δ||)')
ax.set_ylabel('Final Mahalanobis Distance')
ax.set_title('B. Counterfactual: Displacement vs Mahalanobis')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / "fig7_mahalanobis.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 4. PREDICTION SET SIZE DISTRIBUTIONS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall set size distribution
ax = axes[0]
p_test = clf.predict_proba(X_test)[:, 1]
final_quantiles = artifacts['final_quantiles']
global_q = artifacts['global_q']
alpha = 0.1

pred_sets = np.zeros((len(y_test), 2), dtype=int)
for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_sets[i, 1] = 1 if (1 - p_test[i]) <= q else 0
    pred_sets[i, 0] = 1 if p_test[i] <= q else 0

set_sizes = pred_sets.sum(axis=1)
size_counts = pd.Series(set_sizes).value_counts().sort_index()

ax.bar(size_counts.index, size_counts.values, color=colors[0], alpha=0.7, edgecolor='black', width=0.6)
ax.set_xlabel('Prediction Set Size')
ax.set_ylabel('Count')
ax.set_title('A. Overall Prediction Set Size Distribution')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Empty (0)', 'Singleton (1)', 'Both (2)'])
for i, v in enumerate(size_counts.values):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')

# Per-subgroup set size
ax = axes[1]
subgroups_test = sorted(np.unique(sg_test))
size_by_sg = {}
for sg in subgroups_test:
    mask = (sg_test == sg)
    if mask.sum() > 0:
        sizes = set_sizes[mask]
        size_by_sg[sg] = [np.sum(sizes == 0), np.sum(sizes == 1), np.sum(sizes == 2)]

sg_labels = sorted(size_by_sg.keys())
x = np.arange(len(sg_labels))
width = 0.25
for i, size_val in enumerate([0, 1, 2]):
    counts = [size_by_sg[sg][i] for sg in sg_labels]
    ax.bar(x + (i-1)*0.25, counts, width, label=f'Size {size_val}', color=colors[i], alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(sg_labels, rotation=45)
ax.set_ylabel('Count')
ax.set_title('B. Prediction Set Size by Subgroup')
ax.legend()

plt.tight_layout()
plt.savefig(DATA_DIR / "fig8_prediction_sets.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 5. SUBGROUP COVERAGE HEATMAP
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Coverage matrix
subgroups_list = sorted(np.unique(sg_test))
coverage_matrix = np.zeros((len(subgroups_list), len(subgroups_list)))

for i, sg in enumerate(subgroups_list):
    mask = (sg_test == sg)
    if mask.sum() > 0:
        q = final_quantiles.get(sg, global_q)
        pred_set = np.zeros((mask.sum(), 2), dtype=int)
        pred_set[:, 1] = (1 - p_test[mask] <= q).astype(int)
        pred_set[:, 0] = (p_test[mask] <= q).astype(int)
        
        for j, sg_true in enumerate(subgroups_list):
            mask_true = (sg_test[mask] == sg_true)
            if mask_true.sum() > 0:
                covered = np.array([y_test[mask][k] in np.where(pred_set[mask_true][k])[0] for k in range(mask_true.sum())])
                coverage_matrix[i, j] = covered.mean()

im = ax.imshow(coverage_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(subgroups_list)))
ax.set_yticks(range(len(subgroups_list)))
ax.set_xticklabels(subgroups_list, rotation=45)
ax.set_yticklabels(subgroups_list)
ax.set_xlabel('True Subgroup')
ax.set_ylabel('Calibrated Subgroup')
ax.set_title('Subgroup Coverage Matrix (Row=Calibrated, Col=True)')

# Add text annotations
for i in range(len(subgroups_list)):
    for j in range(len(subgroups_list)):
        ax.text(j, i, f'{coverage_matrix[i, j]:.3f}', ha='center', va='center', 
                color='white' if coverage_matrix[i, j] < 0.9 else 'black', fontweight='bold')

plt.colorbar(im, ax=ax, label='Coverage')
plt.tight_layout()
plt.savefig(DATA_DIR / "fig9_coverage_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("All additional visualizations saved!")
print("  fig5_calibration_curves.png")
print("  fig6_feature_importance.png")
print("  fig7_mahalanobis.png")
print("  fig8_prediction_sets.png")
print("  fig9_coverage_heatmap.png")