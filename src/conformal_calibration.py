"""
Subgroup-Conditional Conformal Calibration with Hierarchical Shrinkage (Section 4.2)
====================================================================================
- Split conformal prediction with group-conditional quantiles
- Weighted empirical quantiles (importance weights)
- Hierarchical shrinkage: q_g = (n_g * q_g + kappa * q_global) / (n_g + kappa)
- kappa = 1 / tau^2 (between-stratum variance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")
cohort = pd.read_csv(DATA_DIR / "cohort_incident_t2d.csv")
embeddings = pd.read_csv(DATA_DIR / "embeddings.csv")

# ============================================================
# 1. PREPARE DATA FOR CONFORMAL PREDICTION
# ============================================================
# Merge embeddings with outcomes
data = cohort[['subject_id', 'incident_t2d']].merge(embeddings, on='subject_id')

# Features: all z_* columns (embedding space)
z_cols = [c for c in data.columns if c.startswith('z_')]
X = data[z_cols].values
y = data['incident_t2d'].values

# Subgroup indicator (use age_band as primary subgroup for demo)
subgroup = data['subgroup_age_band'].values
subgroups = np.unique(subgroup[~pd.isna(subgroup)])
print(f"Subgroups: {subgroups}")
print(f"Subgroup sizes: {pd.Series(subgroup).value_counts().to_dict()}")

# ============================================================
# 2. SPLIT: Train / Calibration / Test
# ============================================================
# Use 60% train, 20% calibration, 20% test
# Stratify by outcome and subgroup

# First split: 60% train, 40% temp
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))

X_train, y_train = X[train_idx], y[train_idx]
subgroup_train = subgroup[train_idx]

X_temp, y_temp = X[temp_idx], y[temp_idx]
subgroup_temp = subgroup[temp_idx]

# Second split: 50/50 of temp -> 20% cal, 20% test
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X_temp, y_temp))

X_cal, y_cal = X_temp[cal_idx], y_temp[cal_idx]
subgroup_cal = subgroup_temp[cal_idx]

X_test, y_test = X_temp[test_idx], y_temp[test_idx]
subgroup_test = subgroup_temp[test_idx]

print(f"\nSplit sizes: Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)}")
print(f"Train outcome rate: {y_train.mean():.3f}")
print(f"Cal outcome rate: {y_cal.mean():.3f}")
print(f"Test outcome rate: {y_test.mean():.3f}")

# ============================================================
# 3. FIT BASE PREDICTOR ON TRAIN SET
# ============================================================
print("\nFitting base predictor (Logistic Regression)...")
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
clf.fit(X_train, y_train)

# Get predicted probabilities
p_train = clf.predict_proba(X_train)[:, 1]
p_cal = clf.predict_proba(X_cal)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

print(f"Train AUC: {roc_auc_score(y_train, p_train):.3f}")
print(f"Cal AUC: {roc_auc_score(y_cal, p_cal):.3f}")
print(f"Test AUC: {roc_auc_score(y_test, p_test):.3f}")

# ============================================================
# 4. NONCONFORMITY SCORES
# ============================================================
# For binary classification, use: A = 1 - p(y_true)  (higher = more nonconforming)
# Or: A = -p(y_true)  (lower = more nonconforming)
# We use: A = 1 - p_true so that high scores = surprising

def nonconformity_score(p, y):
    """A = 1 - p(y) where p(y) is predicted prob of true class"""
    return 1 - p[np.arange(len(y)), y.astype(int)]

# For binary: p is probability of class 1
# A = 1 - p if y=1, else A = p if y=0
A_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
A_test = np.where(y_test == 1, 1 - p_test, p_test)

print(f"\nCal nonconformity scores: mean={A_cal.mean():.3f}, std={A_cal.std():.3f}")

# ============================================================
# 5. GLOBAL QUANTILE (Marginal Conformal)
# ============================================================
alpha = 0.1  # 90% coverage
n_cal = len(A_cal)
q_global = np.quantile(A_cal, 1 - alpha, method='higher')
print(f"\nGlobal quantile q_global({1-alpha:.0%}) = {q_global:.4f}")

# Marginal prediction sets on test
pred_sets_global = (A_test <= q_global).astype(int)
coverage_global = pred_sets_global.mean()
print(f"Global coverage on test: {coverage_global:.3f} (target: {1-alpha:.3f})")

# ============================================================
# 6. GROUP-CONDITIONAL QUANTILES (No Shrinkage)
# ============================================================
print("\n--- Group-Conditional Quantiles ---")
group_quantiles = {}
group_sizes = {}

for g in subgroups:
    mask = (subgroup_cal == g)
    n_g = mask.sum()
    group_sizes[g] = n_g
    
    if n_g > 0:
        A_g = A_cal[mask]
        # Weighted quantile (uniform weights for now)
        q_g = np.quantile(A_g, 1 - alpha, method='higher')
        group_quantiles[g] = q_g
        print(f"  {g}: n={n_g}, q_g={q_g:.4f}")
    else:
        group_quantiles[g] = q_global
        print(f"  {g}: n=0, using global q={q_global:.4f}")

# Group-conditional prediction sets
pred_sets_group = np.zeros(len(y_test), dtype=int)
for g in subgroups:
    mask = (subgroup_test == g)
    if mask.sum() > 0 and g in group_quantiles:
        pred_sets_group[mask] = (A_test[mask] <= group_quantiles[g]).astype(int)

# Coverage per group
print("\nGroup-conditional coverage:")
for g in subgroups:
    mask = (subgroup_test == g)
    if mask.sum() > 0:
        cov = pred_sets_group[mask].mean()
        print(f"  {g}: n={mask.sum()}, coverage={cov:.3f}")

# ============================================================
# 7. HIERARCHICAL SHRINKAGE
# ============================================================
print("\n--- Hierarchical Shrinkage ---")

# Estimate between-stratum variance tau^2
# Model: q_g ~ N(q_global, tau^2)
# Use method of moments or MLE on observed group quantiles
valid_groups = [g for g in subgroups if group_sizes[g] > 0]
if len(valid_groups) > 1:
    q_vals = np.array([group_quantiles[g] for g in valid_groups])
    n_vals = np.array([group_sizes[g] for g in valid_groups])
    
    # Precision-weighted mean (global quantile)
    q_global_weighted = np.average(q_vals, weights=n_vals)
    
    # Between-stratum variance (method of moments)
    # Var(q_g) = tau^2 + sigma_g^2 where sigma_g^2 = Var(A_g) / n_g
    within_var = np.array([np.var(A_cal[subgroup_cal == g]) / group_sizes[g] for g in valid_groups])
    total_var = np.var(q_vals, ddof=1)
    tau2 = max(0, total_var - np.mean(within_var))
    
    print(f"Between-stratum variance tau^2 = {tau2:.6f}")
    print(f"Global quantile (weighted) = {q_global_weighted:.4f}")
    
    # Shrinkage constant kappa = 1 / tau^2
    if tau2 > 0:
        kappa = 1.0 / tau2
    else:
        kappa = 1e6  # Large kappa = strong shrinkage to global
    
    print(f"Shrinkage constant kappa = {kappa:.2f}")
    
    # Shrunken quantiles: q_shrunk = (n_g * q_g + kappa * q_global) / (n_g + kappa)
    shrunken_quantiles = {}
    for g in subgroups:
        n_g = group_sizes[g]
        q_g = group_quantiles[g]
        q_shrunk = (n_g * q_g + kappa * q_global_weighted) / (n_g + kappa)
        shrunken_quantiles[g] = q_shrunk
        print(f"  {g}: n={n_g}, q_raw={q_g:.4f}, q_shrunk={q_shrunk:.4f}")
else:
    print("Too few groups for shrinkage estimation")
    shrunken_quantiles = group_quantiles.copy()
    kappa = 0

# ============================================================
# 8. WEIGHTED CALIBRATION (Importance Weights)
# ============================================================
print("\n--- Weighted Calibration ---")
# For demo: weight calibration samples by similarity to test subgroup distribution
# In practice: use sampling design weights or covariate shift weights

# Simple weighting: weight = 1 / (subgroup frequency in calibration)
subgroup_freq = pd.Series(subgroup_cal).value_counts(normalize=True)
weights = np.array([1.0 / subgroup_freq.get(g, 1.0) for g in subgroup_cal])
weights = weights / weights.sum() * len(weights)  # Normalize

# Weighted empirical quantile per group
weighted_quantiles = {}
for g in subgroups:
    mask = (subgroup_cal == g)
    if mask.sum() > 0:
        A_g = A_cal[mask]
        w_g = weights[mask]
        w_g = w_g / w_g.sum()
        
        # Weighted quantile: sort by A, cumsum weights
        sort_idx = np.argsort(A_g)
        A_sorted = A_g[sort_idx]
        w_sorted = w_g[sort_idx]
        cumsum_w = np.cumsum(w_sorted)
        q_idx = np.searchsorted(cumsum_w, 1 - alpha)
        q_idx = min(q_idx, len(A_sorted) - 1)
        q_weighted = A_sorted[q_idx]
        weighted_quantiles[g] = q_weighted
        print(f"  {g}: q_weighted={q_weighted:.4f} (unweighted={group_quantiles[g]:.4f})")
    else:
        weighted_quantiles[g] = group_quantiles[g]

# ============================================================
# 9. FINAL CALIBRATED QUANTILES (Shrunken + Weighted)
# ============================================================
# Combine: use shrunken as base, then apply weighting adjustment
# For simplicity, use shrunken quantiles as final
final_quantiles = shrunken_quantiles.copy()

print("\n--- Final Calibrated Quantiles ---")
for g in subgroups:
    print(f"  {g}: q_final={final_quantiles[g]:.4f}")

# ============================================================
# 10. FINAL PREDICTION SETS WITH COVERAGE GUARANTEES
# ============================================================
pred_sets_final = np.zeros(len(y_test), dtype=int)
for g in subgroups:
    mask = (subgroup_test == g)
    if mask.sum() > 0:
        pred_sets_final[mask] = (A_test[mask] <= final_quantiles[g]).astype(int)

print("\n=== Final Coverage Results ===")
print(f"Overall: {pred_sets_final.mean():.3f} (target: {1-alpha:.3f})")
for g in subgroups:
    mask = (subgroup_test == g)
    if mask.sum() > 0:
        cov = pred_sets_final[mask].mean()
        print(f"  {g}: n={mask.sum()}, coverage={cov:.3f}")

# ============================================================
# 11. PREDICTION SETS FOR EACH TEST POINT
# ============================================================
# For binary classification, prediction set is either {0}, {1}, or {0,1}
# A(x) <= q => include class
# We compute p-values for each class
p_values_test = np.zeros((len(y_test), 2))
for i in range(len(y_test)):
    g = subgroup_test[i]
    q = final_quantiles[g]
    # p-value for class 0: A = p_test[i] (since y=0 => A = p)
    p_values_test[i, 0] = (np.sum(A_cal[subgroup_cal == g] >= p_test[i]) + 1) / (np.sum(subgroup_cal == g) + 1)
    # p-value for class 1: A = 1 - p_test[i]
    p_values_test[i, 1] = (np.sum(A_cal[subgroup_cal == g] >= (1 - p_test[i])) + 1) / (np.sum(subgroup_cal == g) + 1)

# Prediction sets at level alpha
pred_sets = (p_values_test > alpha).astype(int)
set_sizes = pred_sets.sum(axis=1)

print(f"\nPrediction set sizes: mean={set_sizes.mean():.2f}, dist={pd.Series(set_sizes).value_counts().to_dict()}")
print(f"Empty sets: {(set_sizes == 0).sum()}")
print(f"Both classes: {(set_sizes == 2).sum()}")

# ============================================================
# 12. SAVE CALIBRATION ARTIFACTS
# ============================================================
cal_artifacts = {
    'alpha': alpha,
    'global_quantile': q_global,
    'group_quantiles': group_quantiles,
    'shrunken_quantiles': shrunken_quantiles,
    'weighted_quantiles': weighted_quantiles,
    'final_quantiles': final_quantiles,
    'kappa': kappa,
    'tau2': tau2 if 'tau2' in locals() else 0,
    'group_sizes': group_sizes,
    'clf': clf,
    'z_cols': z_cols,
    'subgroup_col': 'age_band'
}

import pickle
with open(DATA_DIR / "calibration_artifacts.pkl", 'wb') as f:
    pickle.dump(cal_artifacts, f)

print("\nCalibration artifacts saved.")

