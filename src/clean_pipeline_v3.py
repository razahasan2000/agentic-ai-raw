"""
Clean Pipeline v3: All Statistical Fixes + Agentic Layer
==========================================================

Fixes:
1. 4-way split: 50% Train / 15% Platt Cal / 15% Conformal+Audit Val / 20% Test
2. Fixed Empirical Bayes: κ = σ²/τ², σ²_g = Var(scores_g) (no double /n_g)
3. Ledoit-Wolf full covariance for counterfactuals
4. Calibrated logits (w_cal, b_cal) for counterfactuals
5. LOCO conformal attribution (risk difference) instead of marginal effect
6. Out-of-fold ECE for auditor
7. Agentic orchestrator with proper loops
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import expit, logit
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data")
RESULTS_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\reproducibility_package\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.1
N_FOLDS = 5

np.random.seed(SEED)

# ============================================================
# 1. LOAD & PREPARE COHORT
# ============================================================
print("=" * 70)
print("CLEAN PIPELINE v3: Subgroup-Calibrated Conformal Diabetes Risk Explanation")
print("=" * 70)

cohort = pd.read_csv(DATA_DIR / "nhanes_diabetes_cohort.csv")
print(f"\n1. LOADED COHORT: {len(cohort)} patients")
print(f"   Diabetes prevalence: {cohort['known_diabetes'].mean()*100:.1f}%")

cohort['subgroup'] = cohort['Race_Ethnicity'].astype(str)
unique_subgroups = sorted(cohort['subgroup'].unique())
print(f"   Subgroups: {unique_subgroups}")
for sg in unique_subgroups:
    sub = cohort[cohort['subgroup'] == sg]
    print(f"     {sg}: n={len(sub)}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")

# ============================================================
# 2. FEATURES (NO HbA1c, NO GLUCOSE)
# ============================================================
CONTINUOUS_FEATURES = {
    'RIDAGEYR': {'unit': 'years'},
    'BMI': {'unit': 'kg/m2'},
    'Waist_cm': {'unit': 'cm'},
    'SBP': {'unit': 'mmHg'},
    'DBP': {'unit': 'mmHg'},
    'INDFMPIR': {'unit': 'ratio'},
}
BINARY_FEATURES = ['Current_Smoker', 'Former_Smoker', 'Current_Drinker', 'Vigorous_Activity', 'Moderate_Activity']
CATEGORICAL_FEATURES = ['Gender']

ref_cohort = cohort[cohort['known_diabetes'] == 0].copy()
print(f"\n2. REFERENCE COHORT (controls): {len(ref_cohort)}")

# ============================================================
# 3. CLINICALLY NORMALIZED EMBEDDING
# ============================================================
print("\n3. CLINICALLY NORMALIZED EMBEDDING (ECDF -> Phi^-1)")

def ecdf_transform(series, ref_series, clip_quantile=0.001):
    ref_clean = ref_series.dropna()
    if len(ref_clean) < 10:
        return pd.Series(np.nan, index=series.index)
    n = len(ref_clean)
    sorted_ref = np.sort(ref_clean)
    def transform(x):
        if pd.isna(x): return np.nan
        rank = np.searchsorted(sorted_ref, x, side='right')
        p = rank / (n + 1)
        p = np.clip(p, clip_quantile, 1 - clip_quantile)
        return norm.ppf(p)
    return series.apply(transform)

embeddings = pd.DataFrame(index=cohort.index)
embeddings['subject_id'] = cohort['SEQN']
embeddings['subgroup'] = cohort['subgroup']
embeddings['y'] = cohort['known_diabetes']

z_cols = []

for feat, spec in CONTINUOUS_FEATURES.items():
    if feat in cohort.columns:
        trans = ecdf_transform(cohort[feat], ref_cohort[feat])
        col = f'z_{feat}'
        embeddings[col] = trans
        z_cols.append(col)
        non_null = trans.notna().sum()
        print(f"   {feat}: {non_null}/{len(cohort)} non-null, range=[{trans.min():.2f}, {trans.max():.2f}]")

for feat in BINARY_FEATURES:
    if feat in cohort.columns:
        col = f'z_{feat}'
        embeddings[col] = cohort[feat].astype(float)
        z_cols.append(col)

for feat in CATEGORICAL_FEATURES:
    if feat in cohort.columns:
        dummies = pd.get_dummies(cohort[feat], prefix=f'z_{feat}')
        for c in dummies.columns:
            embeddings[c] = dummies[c].astype(float)
            z_cols.append(c)

embeddings[z_cols] = embeddings[z_cols].fillna(0.0)
print(f"   Total embedding dimensions: {len(z_cols)}")

embeddings.to_csv(RESULTS_DIR / "embeddings.csv", index=False)

# ============================================================
# 4. DATA SPLITS (4-WAY: 50% Train / 15% Platt Cal / 15% Conformal+Audit Val / 20% Test)
# ============================================================
print("\n4. DATA SPLITS (4-way)")
X = embeddings[z_cols].values
y = embeddings['y'].values
subgroups_arr = embeddings['subgroup'].values
subject_ids = embeddings['subject_id'].values

# First split: 80% train+val, 20% test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
trainval_idx, test_idx = next(sss.split(X, y))

# Second split: of train+val, 75% train, 25% val -> gives 60%/20% overall
# But we want 50%/15%/15% of total -> of 80%, need 50/80=62.5% train, 37.5% val
# Then split val 50/50 for Platt Cal / Conformal+Audit Val
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=SEED)
train_idx, val_idx = next(sss2.split(X[trainval_idx], y[trainval_idx]))
train_idx = trainval_idx[train_idx]
val_idx = trainval_idx[val_idx]

# Split val 50/50 for Platt Cal and Conformal+Audit Val
sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
platt_idx, conf_audit_idx = next(sss3.split(X[val_idx], y[val_idx]))
platt_idx = val_idx[platt_idx]
conf_audit_idx = val_idx[conf_audit_idx]

X_train, y_train = X[train_idx], y[train_idx]
X_platt, y_platt = X[platt_idx], y[platt_idx]
X_conf_audit, y_conf_audit = X[conf_audit_idx], y[conf_audit_idx]
X_test, y_test = X[test_idx], y[test_idx]

sg_train = subgroups_arr[train_idx]
sg_platt = subgroups_arr[platt_idx]
sg_conf_audit = subgroups_arr[conf_audit_idx]
sg_test = subgroups_arr[test_idx]

subject_ids_test = subject_ids[test_idx]

print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Platt Calibration: {len(X_platt)} ({len(X_platt)/len(X)*100:.1f}%)")
print(f"   Conformal+Audit Val: {len(X_conf_audit)} ({len(X_conf_audit)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Save test indices for later use
np.save(RESULTS_DIR / "test_indices.npy", test_idx)
np.save(RESULTS_DIR / "X_test.npy", X[test_idx])
np.save(RESULTS_DIR / "y_test.npy", y[test_idx])
np.save(RESULTS_DIR / "sg_test.npy", sg_test)
np.save(RESULTS_DIR / "subject_ids_test.npy", subject_ids[test_idx])

# ============================================================
# 5. BASE PREDICTOR (Train set only)
# ============================================================
print("\n5. BASE PREDICTOR (Train set)")

base_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED, solver='lbfgs'))
])
base_clf.fit(X_train, y_train)

# Base model probabilities on all splits
p_train_base = base_clf.predict_proba(X_train)[:, 1]
p_platt_base = base_clf.predict_proba(X_platt)[:, 1]
p_conf_audit_base = base_clf.predict_proba(X_conf_audit)[:, 1]
p_test_base = base_clf.predict_proba(X_test)[:, 1]

print(f"   Train AUC (base): {roc_auc_score(y_train, p_train_base):.3f}")
print(f"   Platt AUC (base): {roc_auc_score(y_platt, p_platt_base):.3f}")
print(f"   Conf+Audit AUC (base): {roc_auc_score(y_conf_audit, p_conf_audit_base):.3f}")
print(f"   Test AUC (base): {roc_auc_score(y_test, p_test_base):.3f}")

# ============================================================
# 6. PLATT SCALING (on Platt Calibration set) - Manual implementation
# ============================================================
print("\n6. PLATT SCALING (on Platt Calibration set)")

# Get base model logits on Platt set
logits_platt = base_clf.decision_function(X_platt).reshape(-1, 1)
# Fit Platt scaling: p = sigmoid(A * logit + B)
platt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
platt_lr.fit(logits_platt, y_platt)
platt_a = platt_lr.coef_[0][0]
platt_b = platt_lr.intercept_[0]

# Calibrated probabilities on all splits
logits_train = base_clf.decision_function(X_train)
logits_platt_set = base_clf.decision_function(X_platt)
logits_conf_audit = base_clf.decision_function(X_conf_audit)
logits_test = base_clf.decision_function(X_test)

p_train = expit(platt_a * logits_train + platt_b)
p_platt = expit(platt_a * logits_platt_set + platt_b)
p_conf_audit = expit(platt_a * logits_conf_audit + platt_b)
p_test = expit(platt_a * logits_test + platt_b)

print(f"   Train AUC (cal): {roc_auc_score(y_train, p_train):.3f}")
print(f"   Platt AUC (cal): {roc_auc_score(y_platt, p_platt):.3f}")
print(f"   Conf+Audit AUC (cal): {roc_auc_score(y_conf_audit, p_conf_audit):.3f}")
print(f"   Test AUC (cal): {roc_auc_score(y_test, p_test):.3f}")
print(f"   Platt params: A={platt_a:.4f}, B={platt_b:.4f}")

# ============================================================
# 7. CALIBRATED LOGITS FOR COUNTERFACTUALS
# ============================================================
print("\n7. CALIBRATED LOGITS FOR COUNTERFACTUALS")

# Base model weights (on original features)
base_lr = base_clf.named_steps['lr']
scaler = base_clf.named_steps['scaler']

w_scaled = base_lr.coef_[0]
b_scaled = base_lr.intercept_[0]
scale = scaler.scale_
mean = scaler.mean_

w_base = w_scaled / scale
b_base = b_scaled - np.dot(w_scaled, mean / scale)

# Calibrated model effective weights: p_cal = sigmoid(platt_a * (w_base^T z + b_base) + platt_b)
# = sigmoid(platt_a * w_base^T z + platt_a * b_base + platt_b)
w_cal = platt_a * w_base
b_cal = platt_a * b_base + platt_b

print(f"   Calibrated LR: w_norm={np.linalg.norm(w_cal):.3f}, b={b_cal:.3f}")

# Verify calibrated probabilities
logits_test_cal = X_test @ w_cal + b_cal
p_test_cal_verify = expit(logits_test_cal)
print(f"   Cal prob verification: max_diff={np.max(np.abs(p_test - p_test_cal_verify)):.6f}")

# Also verify on conf+audit set
logits_conf_audit_cal = X_conf_audit @ w_cal + b_cal
p_conf_audit_cal_verify = expit(logits_conf_audit_cal)
print(f"   Conf+Audit cal prob verification: max_diff={np.max(np.abs(p_conf_audit - p_conf_audit_cal_verify)):.6f}")

# ============================================================
# 8. CONFORMAL CALIBRATION (CV+ on Conformal+Audit Validation set)
# ============================================================
print("\n8. CONFORMAL CALIBRATION (CV+ on Conformal+Audit Val)")

# We use the Conformal+Audit Validation set for CV+ calibration
# This avoids using the same data for Platt scaling and conformal calibration

X_ca = X_conf_audit
y_ca = y_conf_audit
sg_ca = sg_conf_audit
n_ca = len(X_ca)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
p_cv = np.zeros(n_ca)
fold_cal_scores = {g: [] for g in unique_subgroups}

for fold_idx, (train_idx_cv, cal_idx_cv) in enumerate(skf.split(X_ca, y_ca)):
    print(f"   Fold {fold_idx+1}/{N_FOLDS}: train={len(train_idx_cv)}, cal={len(cal_idx_cv)}")
    X_tr, y_tr = X_ca[train_idx_cv], y_ca[train_idx_cv]
    X_cl, y_cl = X_ca[cal_idx_cv], y_ca[cal_idx_cv]
    sg_cl = sg_ca[cal_idx_cv]
    
    # Train base model within fold
    clf_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=fold_idx, solver='lbfgs'))
    ])
    clf_cv.fit(X_tr, y_tr)
    
    # Manual Platt scaling on fold's cal set
    logits_cl = clf_cv.decision_function(X_cl).reshape(-1, 1)
    platt_lr_fold = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_lr_fold.fit(logits_cl, y_cl)
    platt_a_fold = platt_lr_fold.coef_[0][0]
    platt_b_fold = platt_lr_fold.intercept_[0]
    
    logits_cl_flat = clf_cv.decision_function(X_cl)
    p_cl = expit(platt_a_fold * logits_cl_flat + platt_b_fold)
    p_cv[cal_idx_cv] = p_cl
    A_cl = np.where(y_cl == 1, 1 - p_cl, p_cl)
    
    for g in unique_subgroups:
        mask = (sg_cl == g)
        if mask.sum() > 0:
            fold_cal_scores[g].extend(A_cl[mask])

# Subgroup quantiles
subgroup_quantiles = {}
for g in unique_subgroups:
    scores = np.array(fold_cal_scores[g])
    if len(scores) > 0:
        q = np.quantile(scores, 1 - ALPHA, method='higher')
    else:
        all_scores = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
        q = np.quantile(all_scores, 1 - ALPHA, method='higher')
    subgroup_quantiles[g] = q
    print(f"     {g}: n_cal={len(fold_cal_scores[g])}, q_raw={q:.4f}")

all_scores = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
global_q = np.quantile(all_scores, 1 - ALPHA, method='higher')
print(f"   Global: q={global_q:.4f}")

# ============================================================
# 9. HIERARCHICAL SHRINKAGE (Empirical Bayes: FIXED - no double /n_g)
# ============================================================
print("\n9. HIERARCHICAL SHRINKAGE (Empirical Bayes - FIXED)")

valid_groups = [g for g in unique_subgroups if len(fold_cal_scores[g]) > 0]
if len(valid_groups) > 1:
    q_vals = np.array([subgroup_quantiles[g] for g in valid_groups])
    n_vals = np.array([len(fold_cal_scores[g]) for g in valid_groups])
    
    # Within-group variance of quantile estimator
    # σ²_g = Var(scores_g)  (NOT divided by n_g - that's handled in the shrinkage formula)
    within_vars = np.array([np.var(fold_cal_scores[g], ddof=1) for g in valid_groups])
    
    # Between-group variance τ²
    weighted_mean = np.average(q_vals, weights=n_vals)
    total_var = np.average((q_vals - weighted_mean)**2, weights=n_vals)
    tau2 = max(0, total_var - np.mean(within_vars))
    
    # Shrinkage weight: κ = σ²/τ² (dimensionless)
    sigma2 = np.mean(within_vars)
    if tau2 > 0:
        kappa = sigma2 / tau2
    else:
        kappa = 1e6  # Complete pooling
    
    print(f"   sigma2 (within): {sigma2:.6f}")
    print(f"   tau2 (between): {tau2:.6f}")
    print(f"   kappa = sigma2/tau2: {kappa:.2f}")
    
    # Shrunken quantile: q̃_g = (q_g + κ * q̄) / (1 + κ)  [NOT (n_g q_g + κ q̄)/(n_g + κ)]
    shrunken = {}
    for g in unique_subgroups:
        q_g = subgroup_quantiles[g]
        q_shrunk = (q_g + kappa * weighted_mean) / (1 + kappa)
        shrunken[g] = q_shrunk
        print(f"     {g}: q_raw={q_g:.4f} -> q_shrunk={q_shrunk:.4f}")
    final_quantiles = shrunken
else:
    final_quantiles = subgroup_quantiles
    print("   Single group - no shrinkage applied")

# ============================================================
# 10. COVERAGE EVALUATION (on Test set)
# ============================================================
print("\n10. COVERAGE EVALUATION (Test set)")

A_test = np.where(y_test == 1, 1 - p_test, p_test)
covered = np.zeros(len(y_test), dtype=bool)
pred_sets = np.zeros((len(y_test), 2), dtype=int)

for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_sets[i, 1] = 1 if (1 - p_test[i]) <= q else 0
    pred_sets[i, 0] = 1 if p_test[i] <= q else 0
    covered[i] = y_test[i] in np.where(pred_sets[i])[0]

print(f"   Overall coverage: {covered.mean():.3f} (target: {1-ALPHA:.3f})")
for g in unique_subgroups:
    mask = (sg_test == g)
    if mask.sum() > 0:
        print(f"   {g}: {covered[mask].mean():.3f} (n={mask.sum()})")

set_sizes = pred_sets.sum(axis=1)
print(f"   Set sizes: mean={set_sizes.mean():.2f}, dist={pd.Series(set_sizes).value_counts().to_dict()}")

# Also evaluate on Conformal+Audit Validation set (honest estimate)
print("\n   Conformal+Audit Validation set coverage (honest):")
A_conf_audit = np.where(y_conf_audit == 1, 1 - p_conf_audit, p_conf_audit)
covered_ca = np.zeros(len(y_conf_audit), dtype=bool)
for i in range(len(y_conf_audit)):
    g = sg_conf_audit[i]
    q = final_quantiles.get(g, global_q)
    pred_set = []
    if (1 - p_conf_audit[i]) <= q: pred_set.append(1)
    if p_conf_audit[i] <= q: pred_set.append(0)
    covered_ca[i] = y_conf_audit[i] in pred_set
print(f"   Overall: {covered_ca.mean():.3f}")
for g in unique_subgroups:
    mask = (sg_conf_audit == g)
    if mask.sum() > 0:
        print(f"   {g}: {covered_ca[mask].mean():.3f} (n={mask.sum()})")

# ============================================================
# 11. FULL COVARIANCE (Ledoit-Wolf) FOR COUNTERFACTUALS
# ============================================================
print("\n11. FULL COVARIANCE (Ledoit-Wolf) FOR COUNTERFACTUALS")

# Estimate full covariance per subgroup using Ledoit-Wolf shrinkage
subgroup_stats = {}
for g in unique_subgroups:
    mask = (subgroups_arr == g)
    X_g = X[mask]
    if len(X_g) < 3: 
        subgroup_stats[g] = None
        continue
    mu_g = X_g.mean(axis=0)
    lw = LedoitWolf().fit(X_g)
    prec_g = lw.precision_
    r2_g = np.percentile([(x - mu_g) @ prec_g @ (x - mu_g) for x in X_g], 90)
    subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}
    print(f"     {g}: n={len(X_g)}, r2_90={r2_g:.3f}")

# Global fallback
mu_global = X.mean(axis=0)
lw_global = LedoitWolf().fit(X)
prec_global = lw_global.precision_
mahal_global = np.array([(x - mu_global) @ prec_global @ (x - mu_global) for x in X])
r2_global = np.percentile(mahal_global, 90)
print(f"     Global: r2_90={r2_global:.3f}")

# ============================================================
# 12. INTERVENTION TIERS (Feature-grounded)
# ============================================================
print("\n12. INTERVENTION TIERS (Feature-grounded)")

# Define tiers using only features that actually exist in NHANES
TIER_DEFINITIONS = {
    1: {  # Behavioral only
        'name': 'Behavioral',
        'features': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker'],
        'description': 'Behavioral: Activity, Smoking, Drinking'
    },
    2: {  # Behavioral + Anthropometric
        'name': 'Behavioral + Anthropometric',
        'features': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker',
                     'z_BMI', 'z_Waist_cm'],
        'description': 'Behavioral + Anthropometric: Activity, Smoking, Drinking, BMI, Waist'
    },
    3: {  # Behavioral + Anthropometric + Clinical Vitals
        'name': 'Behavioral + Anthropometric + Clinical',
        'features': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker',
                     'z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP'],
        'description': 'Behavioral + Anthropometric + Clinical: + SBP, DBP'
    }
}

# Frozen features (never changed)
FROZEN_FEATURES = ['z_RIDAGEYR', 'z_Gender_Female', 'z_Gender_Male', 'z_INDFMPIR']
frozen_idx = [z_cols.index(f) for f in FROZEN_FEATURES if f in z_cols]

for tier_num, tier_def in TIER_DEFINITIONS.items():
    tier_features = tier_def['features']
    tier_idx = [z_cols.index(f) for f in tier_features if f in z_cols]
    print(f"   Tier {tier_num} ({tier_def['name']}): {[z_cols[i] for i in tier_idx]}")

print(f"   Frozen (never changed): {[z_cols[i] for i in frozen_idx]}")

# ============================================================
# 13. COUNTERFACTUAL OPTIMIZATION (Calibrated logits, Full Cov, Tier Escalation)
# ============================================================
print("\n13. COUNTERFACTUAL OPTIMIZATION (Calibrated logits, Full Cov, Tier Escalation)")

def mahalanobis_full(z, mu, prec):
    diff = z - mu
    return diff @ prec @ diff

def solve_counterfactual_tier(z, subgroup, target_prob=0.5, tier=1, radius_mult=1.0):
    """
    Solve counterfactual for a specific tier and radius multiplier.
    Returns: delta, mahal, inside, p_final_cal, feasible
    """
    stats = subgroup_stats.get(subgroup)
    if stats is None:
        stats = {'mu': mu_global, 'prec': prec_global, 'r2': r2_global}
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2'] * (radius_mult**2)
    
    # Linear constraint using CALIBRATED logits: w_cal^T delta <= c
    c = logit(target_prob) - (np.dot(w_cal, z) + b_cal)
    
    # Get actionable indices for this tier
    tier_features = TIER_DEFINITIONS[tier]['features']
    actionable_idx = [z_cols.index(f) for f in TIER_DEFINITIONS[tier]['features'] if f in z_cols]
    
    if len(actionable_idx) == 0:
        return np.zeros(len(z)), np.inf, False, 1.0, False
    
    w_act = w_cal[actionable_idx]
    w_act_norm = np.linalg.norm(w_act)
    
    if w_act_norm < 1e-10:
        return np.zeros(len(z)), np.inf, False, 1.0, False
    
    # Unconstrained minimum norm solution
    delta_act_unconstrained = - (c / (w_act_norm**2)) * w_act
    z_new = z.copy()
    z_new[actionable_idx] = z[actionable_idx] + delta_act_unconstrained
    mahal = mahalanobis_full(z_new, mu, prec)
    
    if mahal <= r2:
        delta = np.zeros(len(z))
        delta[actionable_idx] = delta_act_unconstrained
        p_final = target_prob
        return delta, mahal, True, p_final, True
    
    # Constrained optimization
    n_act = len(actionable_idx)
    bounds = [(-5.0, 5.0)] * n_act
    
    def obj(delta_act):
        return np.sum(delta_act**2)
    
    def constraint_risk(delta_act):
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        logit_new = np.dot(w_cal, z + full_delta) + b_cal
        p_new = expit(logit_new)
        return target_prob - p_new
    
    def constraint_ellipsoid(delta_act):
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        z_new = z + full_delta
        mahal = mahalanobis_full(z_new, mu, prec)
        return r2 - mahal
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_risk},
        {'type': 'ineq', 'fun': constraint_ellipsoid}
    ]
    bounds_list = [(-5.0, 5.0)] * len(actionable_idx)
    
    best_delta = None
    best_obj = np.inf
    best_p = 1.0
    best_mahal = np.inf
    
    for restart in [0] + list(range(3)):
        if restart == 0:
            x0 = delta_act_unconstrained
        else:
            x0 = np.random.randn(len(actionable_idx)) * 0.1
        
        try:
            result = minimize(obj, x0, method='SLSQP', bounds=bounds_list, 
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            if result.success:
                full_delta = np.zeros(len(z))
                full_delta[actionable_idx] = result.x
                logit_new = np.dot(w_cal, z + full_delta) + b_cal
                p_final = expit(logit_new)
                mahal_final = mahalanobis_full(z + full_delta, mu, prec)
                obj_val = np.sum(result.x**2)
                if p_final <= target_prob and mahal_final <= r2 and obj_val < best_obj:
                    best_obj = obj_val
                    best_delta = full_delta
                    best_p = p_final
                    best_mahal = mahal_final
        except:
            continue
    
    if best_delta is not None:
        return best_delta, best_mahal, True, best_p, True
    else:
        return np.zeros(len(z)), np.inf, False, 1.0, False

def solve_counterfactual_with_recovery(z, subgroup, target_prob=0.5):
    """
    Agentic Loop 1: Try Tier 1, then escalate tiers, then relax radius.
    Returns: (delta, mahal, inside, p_final_cal, tier_used, radius_mult, feasible)
    """
    radii = [1.0, 1.5, 2.0]
    tiers = [1, 2, 3]
    
    for tier in tiers:
        for radius_mult in [1.0, 1.5, 2.0]:
            delta, mahal, inside, p_final, feasible = solve_counterfactual_tier(
                z, subgroup, target_prob, tier=tier, radius_mult=radius_mult)
            
            if feasible and inside and p_final <= target_prob:
                return delta, mahal, inside, p_final, tier, radius_mult, True
    
    # If no solution found, return best attempt (lowest p_final among feasible)
    best_result = None
    best_p = 1.0
    for tier in tiers:
        for radius_mult in [1.0, 1.5, 2.0]:
            delta, mahal, inside, p_final, feasible = solve_counterfactual_tier(
                z, subgroup, target_prob, tier=tier, radius_mult=radius_mult)
            if feasible and p_final < best_p:
                best_p = p_final
                best_result = (delta, mahal, inside, p_final, tier, radius_mult, feasible)
    
    if best_result:
        return best_result
    return np.zeros(len(z)), np.inf, False, 1.0, 0, 1.0, False

# ============================================================
# 14. RUN COUNTERFACTUALS WITH AGENTIC LOOP 1
# ============================================================
print("\n14. COUNTERFACTUALS WITH AGENTIC LOOP 1 (Tier Escalation + Radius Relaxation)")

high_risk_idx = np.where(p_test > 0.5)[0]
print(f"   High-risk test patients (cal p > 0.5): {len(high_risk_idx)}")

cf_results = []
for local_idx in high_risk_idx:
    z = X_test[local_idx]
    sg = sg_test[local_idx]
    p = p_test[local_idx]
    p_base = p_test_base[local_idx]
    
    delta, mahal, inside, p_final_cal, tier_used, radius_mult, feasible = solve_counterfactual_with_recovery(
        z, sg, target_prob=0.5)
    
    actionable_all = set()
    for t in [1,2,3]:
        actionable_all.update(TIER_DEFINITIONS[t]['features'])
    actionable_all_idx = [z_cols.index(f) for f in actionable_all if f in z_cols]
    
    changes = {z_cols[i]: delta[i] for i in actionable_all_idx if abs(delta[i]) > 1e-3}
    
    cf_results.append({
        'subject_id': int(subject_ids_test[local_idx]),
        'subgroup': sg, 'p_initial_cal': p, 'p_initial_base': p_base,
        'p_final_cal': p_final_cal, 'p_final_base': p_base,  # base not used for optimization
        'delta_norm': np.linalg.norm(delta), 'mahalanobis': mahal,
        'inside_ellipsoid': inside, 'actionable_changes': changes,
        'tier_used': tier_used, 'radius_mult': radius_mult, 'feasible': feasible
    })

if cf_results:
    inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
    success_cal = sum(1 for r in cf_results if r['p_final_cal'] <= 0.5)
    print(f"   Summary: {len(cf_results)} patients")
    print(f"     Cal model: {success_cal} reached target ({success_cal/len(cf_results)*100:.1f}%)")
    print(f"     Inside ellipsoid: {inside_count} ({inside_count/len(cf_results)*100:.1f}%)")
    
    # Breakdown by tier
    for tier in [1,2,3]:
        tier_results = [r for r in cf_results if r['tier_used'] == tier]
        if tier_results:
            tier_success = sum(1 for r in tier_results if r['p_final_cal'] <= 0.5)
            print(f"     Tier {tier}: {tier_success}/{len(tier_results)} reached target")
    
    # Breakdown by radius
    for r_mult in [1.0, 1.5, 2.0]:
        r_results = [r for r in cf_results if r['radius_mult'] == r_mult]
        if r_results:
            r_success = sum(1 for r in r_results if r['p_final_cal'] <= 0.5)
            print(f"     Radius {r_mult}x: {r_success}/{len(r_results)} reached target")
    
    for r in cf_results[:5]:
        print(f"     ID {r['subject_id']}: p_cal={r['p_initial_cal']:.3f}->{r['p_final_cal']:.3f}, "
              f"||delta||={r['delta_norm']:.3f}, Mahal={r['mahalanobis']:.3f}, "
              f"inside={r['inside_ellipsoid']}, tier={r['tier_used']}, r_mult={r['radius_mult']}")

# ============================================================
# 15. LOCO CONFORMAL ATTRIBUTION (Risk Difference, not Marginal Effect)
# ============================================================
print("\n15. LOCO CONFORMAL ATTRIBUTION (Risk Difference)")

def predict_calibrated(z):
    """Get calibrated probability using manual Platt scaling"""
    logit = z @ w_cal + b_cal
    return expit(logit)

def loco_risk_diff(z, feature_idx):
    """Leave-One-Covariate-Out risk difference: p(full) - p(without feature j)"""
    p_full = predict_calibrated(z)
    z_pert = z.copy()
    z_pert[feature_idx] = 0.0  # Set to reference (0 in z-space)
    p_pert = predict_calibrated(z_pert)
    return p_full - p_pert

# Split Conformal+Audit Validation for attribution
att_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
att_cal_idx, att_test_idx = next(att_sss.split(X_conf_audit, y_conf_audit))

X_att_cal, y_att_cal = X_conf_audit[att_cal_idx], y_conf_audit[att_cal_idx]
sg_att_cal = sg_conf_audit[att_cal_idx]
X_att_test, y_att_test = X_conf_audit[att_test_idx], y_conf_audit[att_test_idx]
sg_att_test = sg_conf_audit[att_test_idx]

# Calibration LOCO risk differences (using CALIBRATED model on conf+audit set)
loco_cal = {g: {j: [] for j in range(len(z_cols))} for g in unique_subgroups}
for i in range(len(X_att_cal)):
    z = X_att_cal[i]
    g = sg_att_cal[i]
    for j in range(len(z_cols)):
        eff = loco_risk_diff(z, j)
        loco_cal[g][j].append(eff)

# Test LOCO risk differences + conformal p-values
significance_results = {g: {} for g in unique_subgroups}
for g in unique_subgroups:
    for j in range(len(z_cols)):
        feat_name = z_cols[j]
        cal_scores = np.array(loco_cal[g][j])
        if len(cal_scores) < 10:
            continue
        test_scores = []
        for i in range(len(X_att_test)):
            if sg_att_test[i] == g:
                test_scores.append(loco_risk_diff(X_att_test[i], j))
        if len(test_scores) == 0:
            continue
        test_scores = np.array(test_scores)
        # Two-sided conformal p-value
        abs_cal = np.abs(cal_scores)
        abs_test = np.abs(test_scores)
        p_values = [(np.sum(abs_cal >= ts) + 1) / (len(abs_cal) + 1) for ts in abs_test]
        significance_results[g][feat_name] = {
            'p_values': np.array(p_values),
            'mean_effect': np.mean(test_scores),
            'n_test': len(test_scores)
        }

# Benjamini-Hochberg correction (FDR = 10%) per subgroup
alpha_fdr = 0.1
for g in unique_subgroups:
    for feat_name, res in significance_results[g].items():
        p_vals = res['p_values']
        if len(p_vals) > 0:
            # BH correction
            sorted_p = np.sort(p_vals)
            m = len(sorted_p)
            bh_threshold = np.array([alpha_fdr * (i + 1) / m for i in range(m)])
            significant = sorted_p <= bh_threshold
            if np.any(significant):
                max_sig_idx = np.max(np.where(significant)[0])
                res['bh_threshold'] = sorted_p[max_sig_idx]
                res['significant'] = p_vals <= sorted_p[max_sig_idx]
            else:
                res['bh_threshold'] = 0
                res['significant'] = np.zeros_like(p_vals, dtype=bool)

# Aggregate
print("   Feature significance (fraction BH-significant across subgroups):")
global_sig = {}
for j in range(len(z_cols)):
    feat_name = z_cols[j]
    all_sig = []
    all_eff = []
    for g in unique_subgroups:
        if feat_name in significance_results[g]:
            res = significance_results[g][feat_name]
            if 'significant' in res:
                all_sig.extend(res['significant'])
                all_eff.extend([res['mean_effect']] * len(res['significant']))
    if all_sig:
        global_sig[feat_name] = {
            'sig_frac': np.mean(all_sig),
            'mean_eff': np.mean(all_eff) if all_eff else 0
        }

for feat, stats in sorted(global_sig.items(), key=lambda x: x[1]['sig_frac'], reverse=True):
    print(f"   {feat}: sig_frac={stats['sig_frac']:.1%}, mean_eff={stats['mean_eff']:+.4f}")

# ============================================================
# 16. DEPLOYABLE AUDITOR (Out-of-fold ECE on Conformal+Audit Val)
# ============================================================
print("\n16. DEPLOYABLE AUDITOR (Out-of-fold ECE on Conformal+Audit Val)")

# Subgroup coverage on Conformal+Audit Validation set
ca_coverage = {}
for g in unique_subgroups:
    mask = (sg_conf_audit == g)
    if mask.sum() > 0:
        p_g = p_conf_audit[mask]
        y_g = y_conf_audit[mask]
        q = final_quantiles.get(g, global_q)
        pred_set_g = np.zeros((len(y_g), 2), dtype=int)
        pred_set_g[:, 1] = (1 - p_g <= q).astype(int)
        pred_set_g[:, 0] = (p_g <= q).astype(int)
        covered_g = np.array([y_g[i] in np.where(pred_set_g[i])[0] for i in range(len(y_g))])
        ca_coverage[g] = covered_g.mean()
        print(f"   CA coverage {g}: {ca_coverage[g]:.3f} (n={mask.sum()})")

# Overall CA coverage
q_overall = global_q
pred_set_ca = np.zeros((len(y_conf_audit), 2), dtype=int)
pred_set_ca[:, 1] = (1 - p_conf_audit <= q_overall).astype(int)
pred_set_ca[:, 0] = (p_conf_audit <= q_overall).astype(int)
ca_overall_cov = np.mean([y_conf_audit[i] in np.where(pred_set_ca[i])[0] for i in range(len(y_conf_audit))])
print(f"   CA overall coverage: {ca_overall_cov:.3f}")

# ECE on Conformal+Audit Validation (out-of-fold)
def compute_ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1]) if i < n_bins-1 else (p >= bins[i]) & (p <= bins[i+1])
        if mask.sum() > 0:
            acc = y[mask].mean()
            conf = p[mask].mean()
            ece += mask.sum() / len(p) * abs(acc - conf)
    return ece

ca_ece = compute_ece(p_conf_audit, y_conf_audit)
print(f"   CA ECE: {ca_ece:.4f}")

# DEPLOYABLE abstention rule
ABSTAIN_COV_THRESH = 0.80
ABSTAIN_ECE_THRESH = 0.25

abstain_subgroups = [g for g, cov in ca_coverage.items() if cov < ABSTAIN_COV_THRESH]
abstain_ece = ca_ece > ABSTAIN_ECE_THRESH

print(f"   Subgroups to abstain: {abstain_subgroups}")
print(f"   ECE abstain: {abstain_ece} (threshold {ABSTAIN_ECE_THRESH})")

# ============================================================
# 17. BUILD EXPLANATIONS (Template-based for now)
# ============================================================
print("\n17. BUILD EXPLANATIONS (Template-based)")

evidence_list = []
for i in range(len(X_test)):
    z = X_test[i]
    sg = sg_test[i]
    p = p_test[i]
    y_true = y_test[i]
    subject_id = subject_ids_test[i]
    
    q = final_quantiles.get(sg, global_q)
    pred_set = []
    if (1 - p) <= q: pred_set.append(1)
    if p <= q: pred_set.append(0)
    
    cf = None
    for r in cf_results:
        if r['subject_id'] == subject_id:
            cf = r
            break
    
    attributions = {}
    if sg in significance_results:
        for feat_name, res in significance_results[sg].items():
            if 'significant' in res and len(res['significant']) > 0:
                # Use first test point's significance
                attributions[feat_name] = {
                    'loco_risk_diff': float(res['mean_effect']),
                    'significant': bool(res['significant'][0]),
                    'bh_threshold': float(res.get('bh_threshold', 0))
                }
    
    sg_ca_cov = ca_coverage.get(sg, 1.0)
    abstain = (sg in abstain_subgroups) or abstain_ece
    
    ev = {
        'subject_id': int(subject_id),
        'subgroup': sg,
        'predicted_risk': float(p),
        'true_label': int(y_true),
        'prediction_set': pred_set,
        'coverage_level': 1 - ALPHA,
        'counterfactual': cf,
        'attributions': attributions,
        'abstain': abstain,
        'abstain_reason': f"CA subgroup coverage {sg_ca_cov:.2f} < {ABSTAIN_COV_THRESH}" if sg in abstain_subgroups else ("CA ECE > threshold" if abstain_ece else None)
    }
    evidence_list.append(ev)

abstain_count = sum(1 for ev in evidence_list if ev['abstain'])
print(f"   Test patients abstained: {abstain_count}/{len(evidence_list)} ({abstain_count/len(evidence_list)*100:.1f}%)")

# ============================================================
# 18. SAVE ALL ARTIFACTS
# ============================================================
print("\n18. SAVING ARTIFACTS")

artifacts = {
    'cohort': cohort,
    'embeddings': embeddings,
    'z_cols': z_cols,
    'base_clf': base_clf,
    'base_lr': base_lr,
    'scaler': scaler,
    'w_cal': w_cal, 'b_cal': b_cal,
    'platt_a': platt_a, 'platt_b': platt_b,
    'final_quantiles': final_quantiles,
    'global_q': global_q,
    'alpha': ALPHA,
    'subgroup_stats': subgroup_stats,
    'cf_results': cf_results,
    'significance_results': significance_results,
    'global_significance': global_sig,
    'evidence_list': evidence_list,
    'ca_coverage': ca_coverage,
    'ca_ece': ca_ece,
    'abstain_subgroups': abstain_subgroups,
    'abstain_ece': abstain_ece,
    'actionable_idx': [z_cols.index(f) for t in [1,2,3] for f in TIER_DEFINITIONS[t]['features'] if f in z_cols],
    'actionable_names': list(set().union(*[set(TIER_DEFINITIONS[t]['features']) for t in [1,2,3]])),
    'TIER_DEFINITIONS': TIER_DEFINITIONS,
    'FROZEN_FEATURES': FROZEN_FEATURES,
}

with open(RESULTS_DIR / "clean_pipeline_artifacts.pkl", 'wb') as f:
    pickle.dump(artifacts, f)

results_df = pd.DataFrame(cf_results)
if len(results_df) > 0:
    results_df.to_csv(RESULTS_DIR / "counterfactual_results.csv", index=False)

cov_rows = []
for g in unique_subgroups:
    mask = (sg_test == g)
    if mask.sum() > 0:
        cov_rows.append({
            'subgroup': g,
            'n_test': int(mask.sum()),
            'test_coverage': float(covered[mask].mean()),
            'ca_coverage': float(ca_coverage.get(g, 0)),
            'quantile': float(final_quantiles.get(g, global_q))
        })
pd.DataFrame(cov_rows).to_csv(RESULTS_DIR / "coverage_table.csv", index=False)

att_rows = []
for feat, stats in global_sig.items():
    att_rows.append({
        'feature': feat,
        'sig_fraction': stats['sig_frac'],
        'mean_effect': stats['mean_eff']
    })
pd.DataFrame(att_rows).to_csv(RESULTS_DIR / "attribution_table.csv", index=False)

print("   All artifacts saved to results/")
print("\n" + "=" * 70)
print("CLEAN PIPELINE v3 COMPLETE")
print("=" * 70)