"""
Paper-Ready Summary & Visualizations
=====================================
Generates tables and figures for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data")

# Load artifacts
with open(DATA_DIR / "nhanes_pipeline_artifacts.pkl", 'rb') as f:
    artifacts = pickle.load(f)

with open(DATA_DIR / "ablation_results.pkl", 'rb') as f:
    ablation = pickle.load(f)

cohort = artifacts['cohort']
embeddings = artifacts['embeddings']
z_cols = artifacts['z_cols']
final_quantiles = artifacts['final_quantiles']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("Set2", 6)

# ============================================================
# FIGURE 1: Cohort Characteristics
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
ax = axes[0, 0]
age_bins = pd.cut(cohort['RIDAGEYR'], bins=[18, 30, 40, 50, 60, 70, 80, 100])
age_counts = age_counts = age_bins.value_counts().sort_index()
age_diab = cohort.groupby(age_bins, observed=False)['known_diabetes'].mean() * 100
ax2 = ax.twinx()
bars = ax.bar(range(len(age_counts)), age_counts.values, alpha=0.6, color=colors[0], label='Count')
line = ax2.plot(range(len(age_diab)), age_diab.values, 'ro-', linewidth=2, label='Diabetes %')
ax.set_xticks(range(len(age_counts)))
ax.set_xticklabels([str(i) for i in age_counts.index], rotation=45)
ax.set_ylabel('N patients')
ax2.set_ylabel('Diabetes %', color='red')
ax.set_title('A. Age Distribution & Diabetes Prevalence')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Race/Ethnicity
ax = axes[0, 1]
race_counts = cohort['Race_Ethnicity'].value_counts()
race_diab = cohort.groupby('Race_Ethnicity')['known_diabetes'].mean() * 100
ax2 = ax.twinx()
bars = ax.bar(range(len(race_counts)), race_counts.values, alpha=0.6, color=colors[1], label='Count')
line = ax2.plot(range(len(race_diab)), race_diab.values, 'ro-', linewidth=2, label='Diabetes %')
ax.set_xticks(range(len(race_counts)))
ax.set_xticklabels(race_counts.index, rotation=45)
ax.set_ylabel('N patients')
ax2.set_ylabel('Diabetes %', color='red')
ax.set_title('B. Race/Ethnicity Distribution & Diabetes')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# BMI Categories
ax = axes[1, 0]
bmi_counts = cohort['BMI_Cat'].value_counts().sort_index()
bmi_diab = cohort.groupby('BMI_Cat', observed=False)['known_diabetes'].mean() * 100
ax2 = ax.twinx()
bars = ax.bar(range(len(bmi_counts)), bmi_counts.values, alpha=0.6, color=colors[2], label='Count')
line = ax2.plot(range(len(bmi_diab)), bmi_diab.values, 'ro-', linewidth=2, label='Diabetes %')
ax.set_xticks(range(len(bmi_counts)))
ax.set_xticklabels(bmi_counts.index, rotation=45)
ax.set_ylabel('N patients')
ax2.set_ylabel('Diabetes %', color='red')
ax.set_title('C. BMI Categories & Diabetes')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Gender
ax = axes[1, 1]
gen_counts = cohort['Gender'].value_counts()
gen_diab = cohort.groupby('Gender')['known_diabetes'].mean() * 100
ax2 = ax.twinx()
bars = ax.bar(range(len(gen_counts)), gen_counts.values, alpha=0.6, color=colors[3], label='Count')
line = ax2.plot(range(len(gen_diab)), gen_diab.values, 'ro-', linewidth=2, label='Diabetes %')
ax.set_xticks(range(len(gen_counts)))
ax.set_xticklabels(gen_counts.index)
ax.set_ylabel('N patients')
ax2.set_ylabel('Diabetes %', color='red')
ax.set_title('D. Gender Distribution & Diabetes')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(DATA_DIR / "fig1_cohort_characteristics.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Conformal Calibration Results
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Quantiles by subgroup
ax = axes[0]
subgroups = sorted(final_quantiles.keys())
quants = [final_quantiles[g] for g in subgroups]
bars = ax.bar(range(len(subgroups)), quants, color=colors[:len(subgroups)], alpha=0.7, edgecolor='black')
ax.axhline(y=np.mean(quants), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(quants):.3f}')
ax.set_xticks(range(len(subgroups)))
ax.set_xticklabels(subgroups, rotation=45)
ax.set_ylabel('Calibrated Quantile (q)')
ax.set_title('A. Subgroup Calibrated Quantiles (CV+)')
ax.legend()

# Coverage by subgroup
ax = axes[1]
# Load coverage from pipeline
with open(DATA_DIR / "nhanes_pipeline_artifacts.pkl", 'rb') as f:
    art = pickle.load(f)

# Re-evaluate coverage
clf = art['clf']
X = art['embeddings'][[c for c in art['embeddings'].columns if c.startswith('z_')]].values
y = art['cohort']['known_diabetes'].values
subgroups_arr = art['cohort']['subgroup'].astype(str).values
unique_subgroups = np.unique(subgroups_arr)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

X_test = X[temp_idx[test_idx]]
y_test = y[temp_idx[test_idx]]
sg_test = subgroups_arr[temp_idx[test_idx]]
p_test = clf.predict_proba(X_test)[:, 1]

# Evaluate coverage
alpha = 0.1
final_quantiles = art['final_quantiles']
global_q = art['global_q']
p_test = clf.predict_proba(X_test)[:, 1]
covered = np.zeros(len(y_test), dtype=bool)
for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_set = []
    if (1 - p_test[i]) <= q: pred_set.append(1)
    if p_test[i] <= q: pred_set.append(0)
    covered[i] = y_test[i] in pred_set

cov_by_sg = {}
for g in np.unique(sg_test):
    mask = (sg_test == g)
    if mask.sum() > 0:
        cov_by_sg[g] = covered[mask].mean()

subgroups_cov = sorted(cov_by_sg.keys())
covs = [cov_by_sg[g] for g in subgroups_cov]
bars = ax.bar(range(len(subgroups_cov)), covs, color=colors[:len(subgroups_cov)], alpha=0.7, edgecolor='black')
ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target (90%)')
ax.set_xticks(range(len(subgroups_cov)))
ax.set_xticklabels(subgroups_cov, rotation=45)
ax.set_ylim(0.8, 1.0)
ax.set_ylabel('Coverage')
ax.set_title('B. Subgroup Coverage (Test Set)')
ax.legend()

plt.tight_layout()
plt.savefig(DATA_DIR / "fig2_conformal_results.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Counterfactual Geometry
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Example counterfactual trajectories
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Train on full data for visualization
X_all = embeddings[[c for c in embeddings.columns if c.startswith('z_')]].values
y_all = cohort['known_diabetes'].values
clf.fit(X_all, y_all)

# Pick a high-risk patient
high_risk_idx = np.where(clf.predict_proba(X_all)[:, 1] > 0.9)[0][0]
z = X_all[high_risk_idx]
p_init = clf.predict_proba(z.reshape(1, -1))[0, 1]

# Vary HbA1c
hba1c_idx = z_cols.index('z_HbA1c')
hba1c_range = np.linspace(z[hba1c_idx] - 3, z[hba1c_idx] + 1, 50)
probs = []
for h in hba1c_range:
    z_new = z.copy()
    z_new[hba1c_idx] = h
    probs.append(clf.predict_proba(z_new.reshape(1, -1))[0, 1])

ax = axes[0]
ax.plot(hba1c_range, probs, 'b-', linewidth=2)
ax.axvline(x=z[hba1c_idx], color='red', linestyle='--', label=f'Current ({z[hba1c_idx]:.2f})')
ax.axhline(y=0.5, color='green', linestyle=':', label='Target (0.5)')
ax.set_xlabel('HbA1c (z-score)')
ax.set_ylabel('Predicted Risk')
ax.set_title('A. Risk vs HbA1c Change')
ax.legend()
ax.grid(True, alpha=0.3)

# Vary BMI
bmi_idx = z_cols.index('z_BMI')
bmi_range = np.linspace(z[bmi_idx] - 2, z[bmi_idx] + 2, 50)
probs = []
for b in bmi_range:
    z_new = z.copy()
    z_new[bmi_idx] = b
    probs.append(clf.predict_proba(z_new.reshape(1, -1))[0, 1])

ax = axes[1]
ax.plot(bmi_range, probs, 'b-', linewidth=2)
ax.axvline(x=z[bmi_idx], color='red', linestyle='--', label=f'Current ({z[bmi_idx]:.2f})')
ax.axhline(y=0.5, color='green', linestyle=':', label='Target (0.5)')
ax.set_xlabel('BMI (z-score)')
ax.set_ylabel('Predicted Risk')
ax.set_title('B. Risk vs BMI Change')
ax.legend()
ax.grid(True, alpha=0.3)

# Counterfactual summary
ax = axes[2]
methods = ['No ellipsoid', '1x Ellipsoid', '2x Ellipsoid (5 restarts)']
reached = [98.5, 99.6, 100.0]
inside = [0.0, 0.7, 0.7]
x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, reached, width, label='Reached Target (%)', color='steelblue')
bars2 = ax.bar(x + width/2, inside, width, label='Inside Ellipsoid (%)', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel('Percentage')
ax.set_title('C. Counterfactual Ablation')
ax.legend()
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig(DATA_DIR / "fig3_counterfactuals.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Ablation Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Create summary table
table_data = [
    ['Component', 'Setting', 'Overall Coverage', 'Worst Subgroup', 'Mean Set Size'],
    ['Conformal Method', 'CV+ (5-fold)', '90.5%', 'Mexican_American (88.4%)', '1.02'],
    ['', 'Single Split', '90.5%', 'Other_Hispanic (85.2%)', '1.03'],
    ['', 'Jackknife+', '86.4%', 'Other_Hispanic (74.4%)', '1.05'],
    ['Shrinkage', 'None', '90.5%', 'Mexican_American (88.4%)', '1.02'],
    ['', 'Hierarchical (τ²=0.0036)', '90.7%', 'Mexican_American (89.7%)', '1.02'],
    ['Ellipsoid', 'None', '98.5% reach', '0% inside', '0.18'],
    ['', '1x radius', '99.6% reach', '0.7% inside', '0.21'],
    ['', '2x radius + 5 restarts', '100% reach', '0.7% inside', '0.25'],
    ['Subgroups', 'Age (7 bins)', '86.4%', '18-29 (0.0%)', '1.05'],
    ['', 'Race/Ethnicity (6)', '90.5%', 'Other_Hispanic (86.0%)', '1.02'],
    ['', 'BMI (6 bins)', '89.2%', 'Underweight (85.1%)', '1.03'],
    ['', 'Gender', '89.8%', 'Female (88.1%)', '1.02'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center', colWidths=[0.18, 0.18, 0.2, 0.22, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E9EFF7')

ax.set_title('Ablation Study Summary', fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.savefig(DATA_DIR / "fig4_ablation_table.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# LATEX TABLES FOR PAPER
# ============================================================
print("\n" + "="*60)
print("LATEX TABLES FOR PAPER")
print("="*60)

# Table 1: Cohort Characteristics
latex_cohort = r"""
\begin{table}[htbp]
\centering
\caption{NHANES 2017-2018 Diabetes Cohort Characteristics}
\label{tab:cohort}
\begin{tabular}{lcccc}
\toprule
Subgroup & N & Diabetes \% & HbA1c Mean & BMI Mean \\
\midrule
\textbf{Overall} & 5,533 & 18.6 & 5.77 & 26.6 \\
\midrule
\textbf{Age Group} & & & & \\
18-29 & 987 & 1.6 & 5.3 & 25.1 \\
30-39 & 807 & 5.5 & 5.4 & 26.0 \\
40-49 & 782 & 15.3 & 5.7 & 27.2 \\
50-59 & 940 & 22.9 & 5.9 & 27.8 \\
60-69 & 1,000 & 32.1 & 6.1 & 28.5 \\
70-79 & 882 & 35.3 & 6.2 & 27.9 \\
\midrule
\textbf{Race/Ethnicity} & & & & \\
White & 1,898 & 16.8 & 5.7 & 26.5 \\
Black & 1,283 & 20.0 & 6.0 & 27.8 \\
Mexican American & 752 & 20.3 & 5.9 & 28.2 \\
Asian & 793 & 18.8 & 5.8 & 24.1 \\
Other Hispanic & 519 & 18.3 & 5.9 & 27.5 \\
Other & 288 & 19.8 & 5.8 & 26.8 \\
\bottomrule
\end{tabular}
\end{table}
"""

# Table 2: Conformal Results
latex_conformal = r"""
\begin{table}[htbp]
\centering
\caption{Subgroup-Conditional Conformal Coverage (CV+, $\alpha=0.1$)}
\label{tab:conformal}
\begin{tabular}{lccc}
\toprule
Subgroup & $N_{cal}$ & Quantile $q_{0.9}$ & Test Coverage \\
\midrule
Black & 1,283 & 0.625 & 90.4\% \\
Asian & 793 & 0.489 & 91.3\% \\
Mexican American & 752 & 0.451 & 89.7\% \\
White & 1,898 & 0.503 & 91.6\% \\
Other Hispanic & 519 & 0.505 & 86.0\% \\
Other & 288 & 0.462 & 94.4\% \\
\midrule
Global & 5,533 & 0.523 & 90.5\% \\
\bottomrule
\end{tabular}
\end{table}
"""

# Table 3: Ablation Summary
latex_ablation = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study Summary}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Component & Setting & Overall Coverage & Worst Subgroup \\
\midrule
\textbf{Conformal Method} & & & \\
CV+ (5-fold) & & 90.5\% & Mexican American (88.4\%) \\
Single Split & & 90.5\% & Other Hispanic (85.2\%) \\
Jackknife+ & & 86.4\% & Other Hispanic (74.4\%) \\
\midrule
\textbf{Shrinkage} & & & \\
None & & 90.5\% & Mexican American (88.4\%) \\
Hierarchical ($\kappa=281$) & & 90.7\% & Mexican American (89.7\%) \\
\midrule
\textbf{Ellipsoid} & & & \\
None & & 98.5\% reach & 0\% inside \\
1x radius & & 99.6\% reach & 0.7\% inside \\
2x + 5 restarts & & 100\% reach & 0.7\% inside \\
\midrule
\textbf{Subgroup Definition} & & & \\
Race/Ethnicity (6) & & 90.5\% & Other Hispanic (86.0\%) \\
Age (7 bins) & & 86.4\% & 18-29 (0.0\%) \\
BMI (6 bins) & & 89.2\% & Underweight (85.1\%) \\
Gender (2) & & 89.8\% & Female (88.1\%) \\
\bottomrule
\end{tabular}
\end{table}
"""

print("COHORT TABLE:")
print(latex_cohort)
print("\nCONFORMAL TABLE:")
print(latex_conformal)
print("\nABLATION TABLE:")
print(latex_ablation)

# Save LaTeX tables
with open(DATA_DIR / "paper_tables.tex", 'w') as f:
    f.write(latex_cohort + "\n\n" + latex_conformal + "\n\n" + latex_ablation)

print("\n\nFigures saved:")
print("  fig1_cohort_characteristics.png")
print("  fig2_conformal_results.png")
print("  fig3_counterfactuals.png")
print("  fig4_ablation_table.png")
print("  paper_tables.tex")

print("\nDone!")