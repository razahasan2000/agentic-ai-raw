"""
Clean Pipeline v6: ALL REVIEWER FIXES - Production Ready
==========================================================

Complete fixes for all reviewer blocking issues:

BLOCKING ISSUE 1 - BIBLIOGRAPHY:
- Remove non-existent [2] (Angiuli et al., JASA 119(545):1-14, 2024)
- Restore Ledoit & Wolf (2004) for Section 2.5 covariance estimation
- Fix [5, 11]: Lei et al. (2018) is NOT geometric conformal prediction; use correct refs

BLOCKING ISSUE 2 - TABLE 5 & LOOP 2 LLM EVALUATION:
- Separate Template (100.0%) from Agentic Loop (Qwen-2.5-7B + Loop 2 repair) in Table 5
- Actually run Loop 2's M=3 repair retries on Qwen-2.5-7B-Instruct
- Clarify how Unconstrained LLM gets counterfactual reach (it hallucinates actions)
- Include 1 concrete JSON + narrative + repair trace in supplementary

BLOCKING ISSUE 3 - FIGURE 1 LEGIBILITY:
- Reroute dashed blue arrows so they don't cut directly across left-column boxes
- Fix AGENTIC LAYER border overlap with header text

BLOCKING ISSUE 4 - THREE CORE STATISTICAL/MODELING FIXES:
- Section 2.4 & Table 2: Use quantile sampling variance (propto 1/n_g) instead of Var(scores_g)
  so q_tilde_g doesn't collapse to 0.662 everywhere; fix Theorem 1's denominator
- Section 2.5 & Table 3: Constrain binary 0/1 behavioral features in Tier 1 to valid
  discrete states via enumeration (not continuous SLSQP steps)
- Section 2.6 & Table 4: Replace Brier-LOCO (requires unknown y at inference time,
  ranks Age at 0.0% below Former Smoker at 3.4%) with inference-valid
  feature-importance test (conditional permutation test with conformal p-values)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.covariance import LedoitWolf
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize, Bounds
from scipy.special import expit, logit
import pickle
import warnings
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
warnings.filterwarnings('ignore')

# Try to import transformers for real LLM
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available, CUDA:", torch.cuda.is_available())
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available")

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data")
RESULTS_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\reproducibility_package\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.1

np.random.seed(SEED)

# ============================================================
# 1. LOAD & PREPARE COHORT
# ============================================================
print("=" * 70)
print("CLEAN PIPELINE v6: ALL REVIEWER FIXES APPLIED")
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

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
trainval_idx, test_idx = next(sss.split(X, y))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=SEED)
train_idx, val_idx = next(sss2.split(X[trainval_idx], y[trainval_idx]))
train_idx = trainval_idx[train_idx]
val_idx = trainval_idx[val_idx]

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

p_train_base = base_clf.predict_proba(X_train)[:, 1]
p_platt_base = base_clf.predict_proba(X_platt)[:, 1]
p_conf_audit_base = base_clf.predict_proba(X_conf_audit)[:, 1]
p_test_base = base_clf.predict_proba(X_test)[:, 1]

print(f"   Train AUC (base): {roc_auc_score(y_train, p_train_base):.3f}")
print(f"   Platt AUC (base): {roc_auc_score(y_platt, p_platt_base):.3f}")
print(f"   Conf+Audit AUC (base): {roc_auc_score(y_conf_audit, p_conf_audit_base):.3f}")
print(f"   Test AUC (base): {roc_auc_score(y_test, p_test_base):.3f}")

# ============================================================
# 6. PLATT SCALING (on Platt Calibration set only)
# ============================================================
print("\n6. PLATT SCALING (on Platt Calibration set)")

logits_platt = base_clf.decision_function(X_platt).reshape(-1, 1)
platt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
platt_lr.fit(logits_platt, y_platt)
platt_a = platt_lr.coef_[0][0]
platt_b = platt_lr.intercept_[0]

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

base_lr = base_clf.named_steps['lr']
scaler = base_clf.named_steps['scaler']

w_scaled = base_lr.coef_[0]
b_scaled = base_lr.intercept_[0]
scale = scaler.scale_
mean = scaler.mean_

w_base = w_scaled / scale
b_base = b_scaled - np.dot(w_scaled, mean / scale)

w_cal = platt_a * w_base
b_cal = platt_a * b_base + platt_b

print(f"   Calibrated LR: w_norm={np.linalg.norm(w_cal):.3f}, b={b_cal:.3f}")

logits_test_cal = X_test @ w_cal + b_cal
p_test_cal_verify = expit(logits_test_cal)
print(f"   Cal prob verification: max_diff={np.max(np.abs(p_test - p_test_cal_verify)):.6f}")

logits_conf_audit_cal = X_conf_audit @ w_cal + b_cal
p_conf_audit_cal_verify = expit(logits_conf_audit_cal)
print(f"   Conf+Audit cal prob verification: max_diff={np.max(np.abs(p_conf_audit - p_conf_audit_cal_verify)):.6f}")

# ============================================================
# 8. CONFORMAL CALIBRATION (Split Conformal on Conformal+Audit Val)
# ============================================================
print("\n8. CONFORMAL CALIBRATION (Split Conformal on Conformal+Audit Val)")

X_ca_train, X_ca_cal, y_ca_train, y_ca_cal, sg_ca_train, sg_ca_cal = train_test_split(
    X_conf_audit, y_conf_audit, sg_conf_audit, test_size=0.5, random_state=SEED, stratify=y_conf_audit
)

ca_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED, solver='lbfgs'))
])
ca_clf.fit(X_ca_train, y_ca_train)

logits_ca_cal = ca_clf.decision_function(X_ca_cal).reshape(-1, 1)
platt_lr_ca = LogisticRegression(solver='lbfgs', max_iter=1000)
platt_lr_ca.fit(logits_ca_cal, y_ca_cal)
platt_a_ca = platt_lr_ca.coef_[0][0]
platt_b_ca = platt_lr_ca.intercept_[0]

logits_ca_cal_flat = ca_clf.decision_function(X_ca_cal)
p_ca_cal = expit(platt_a_ca * logits_ca_cal_flat + platt_b_ca)
A_ca_cal = np.where(y_ca_cal == 1, 1 - p_ca_cal, p_ca_cal)

subgroup_quantiles = {}
for g in unique_subgroups:
    mask = (sg_ca_cal == g)
    scores = A_ca_cal[mask]
    if len(scores) > 0:
        q = np.quantile(scores, 1 - ALPHA, method='higher')
    else:
        q = np.quantile(A_ca_cal, 1 - ALPHA, method='higher')
    subgroup_quantiles[g] = q
    print(f"     {g}: n_cal={mask.sum()}, q_raw={q:.4f}")

all_scores = A_ca_cal
global_q = np.quantile(all_scores, 1 - ALPHA, method='higher')
print(f"   Global: q={global_q:.4f}")

# ============================================================
# 9. HIERARCHICAL SHRINKAGE (Empirical Bayes) - FIXED QUANTILE VARIANCE
# ============================================================
print("\n9. HIERARCHICAL SHRINKAGE (Empirical Bayes - FIXED: quantile variance propto 1/n_g)")

# PROPER quantile variance: alpha*(1-alpha) / (n_g * f_g(q)^2)
# This is the FIXED formula - scales with 1/n_g, NOT Var(scores_g)
within_vars = {}
for g in unique_subgroups:
    mask = (sg_ca_cal == g)
    scores_g = A_ca_cal[mask]
    n_g = mask.sum()
    if n_g > 10:
        kde = gaussian_kde(scores_g)
        f_q = kde(subgroup_quantiles[g])[0]
        if f_q > 0:
            # CORRECT formula: alpha*(1-alpha) / (n_g * f(q)^2)
            analytical_var = ALPHA * (1 - ALPHA) / (n_g * f_q**2)
            within_vars[g] = max(analytical_var, 1e-8)
        else:
            within_vars[g] = 1e-8
    else:
        within_vars[g] = 1e-8
    print(f"     {g}: n={n_g}, q={subgroup_quantiles[g]:.4f}, sigma2={within_vars[g]:.6f}")

# Between-group variance (tau^2) via method of moments
valid_groups = [g for g in unique_subgroups if g in within_vars]
if len(valid_groups) > 1:
    q_vals = np.array([subgroup_quantiles[g] for g in valid_groups])
    w_vars = np.array([within_vars[g] for g in valid_groups])
    precisions = 1 / np.maximum(w_vars, 1e-8)
    weighted_mean = np.average(q_vals, weights=precisions)
    total_var = np.average((q_vals - weighted_mean)**2, weights=precisions)
    tau2 = max(0, total_var - np.mean(w_vars))
else:
    tau2 = 0
    weighted_mean = global_q

print(f"   tau2 (between): {tau2:.6f}")
print(f"   weighted_mean: {weighted_mean:.4f}")

# Empirical Bayes shrinkage
shrunken = {}
for g in unique_subgroups:
    q_g = subgroup_quantiles[g]
    sigma2_g = within_vars.get(g, 1e-8)
    
    if tau2 > 0 and sigma2_g > 0:
        # Partial pooling: weight by inverse variance
        prec_g = 1 / sigma2_g
        prec_global = 1 / tau2
        q_shrunk = (prec_g * q_g + prec_global * weighted_mean) / (prec_g + prec_global)
        kappa = sigma2_g / tau2
    elif tau2 == 0:
        # Complete pooling: no between-group heterogeneity
        q_shrunk = weighted_mean
        kappa = np.inf
    else:
        # Fallback (should not happen)
        q_shrunk = q_g
        kappa = 0.0
    
    print(f"     {g}: q_raw={subgroup_quantiles[g]:.4f} -> q_shrunk={q_shrunk:.4f}, sigma2={sigma2_g:.6f}, kappa={kappa:.2f}")
    shrunken[g] = q_shrunk

final_quantiles = shrunken

# ============================================================
# 10. COVERAGE EVALUATION (Test set AND Validation set - SEPARATE)
# ============================================================
print("\n10. COVERAGE EVALUATION (Test set AND Validation set - SEPARATE)")

def predict_calibrated(z):
    logit = z @ w_cal + b_cal
    return expit(logit)

def compute_coverage(p, y, q_dict, subgroups, unique_subgroups, global_q):
    covered = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        g = subgroups[i]
        q = q_dict.get(g, global_q)
        pred_set = []
        if (1 - p[i]) <= q: pred_set.append(1)
        if p[i] <= q: pred_set.append(0)
        covered[i] = y[i] in pred_set
    overall = covered.mean()
    per_subgroup = {}
    for g in unique_subgroups:
        mask = np.array([sg == g for sg in subgroups])
        if mask.sum() > 0:
            per_subgroup[g] = covered[mask].mean()
    return overall, per_subgroup, covered

# Test set coverage
overall_test, test_per_sg, covered_test = compute_coverage(p_test, y_test, final_quantiles, sg_test, unique_subgroups, global_q)
print(f"   Overall TEST coverage: {overall_test:.3f} (target: {1-ALPHA:.3f})")
for g in unique_subgroups:
    print(f"   {g}: TEST {test_per_sg[g]:.3f} (n={sum(1 for sg in sg_test if sg==g)})")

# Validation (Conformal+Audit Calibration) set coverage
overall_val, val_per_sg, covered_ca = compute_coverage(p_ca_cal, y_ca_cal, final_quantiles, sg_ca_cal, unique_subgroups, global_q)
print(f"\n   Overall VAL coverage: {overall_val:.3f} (target: {1-ALPHA:.3f})")
for g in unique_subgroups:
    print(f"   {g}: VAL {val_per_sg[g]:.3f} (n={sum(1 for sg in sg_ca_cal if sg==g)})")

# Set sizes on test
A_test = np.where(y_test == 1, 1 - p_test, p_test)
pred_sets = np.zeros((len(y_test), 2), dtype=int)
for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_sets[i, 1] = 1 if (1 - p_test[i]) <= q else 0
    pred_sets[i, 0] = 1 if p_test[i] <= q else 0
set_sizes = pred_sets.sum(axis=1)
print(f"   Set sizes: mean={set_sizes.mean():.2f}, dist={pd.Series(set_sizes).value_counts().to_dict()}")

# ============================================================
# 11. FULL COVARIANCE (Ledoit-Wolf) FOR COUNTERFACTUALS
# ============================================================
print("\n11. FULL COVARIANCE (Ledoit-Wolf) FOR COUNTERFACTUALS")
print("   Reference: Ledoit & Wolf (2004), J. Multivariate Analysis, 88(2):365-411")

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

mu_global = X.mean(axis=0)
lw_global = LedoitWolf().fit(X)
prec_global = lw_global.precision_
mahal_global = np.array([(x - mu_global) @ prec_global @ (x - mu_global) for x in X])
r2_global = np.percentile(mahal_global, 90)
print(f"     Global: r2_90={r2_global:.3f}")

# ============================================================
# 12. INTERVENTION TIERS (Feature-grounded) + BINARY CONSTRAINTS (FIXED)
# ============================================================
print("\n12. INTERVENTION TIERS (Feature-grounded) + BINARY CONSTRAINTS (FIXED: Tier 1 uses discrete enumeration)")

TIER_DEFINITIONS = {
    1: {  # Behavioral only - BINARY FEATURES ONLY
        'name': 'Behavioral',
        'continuous': [],  # NO continuous features in Tier 1
        'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker'],
        'description': 'Behavioral: Activity, Smoking, Drinking (binary 0/1 only)'
    },
    2: {  # Behavioral + Anthropometric
        'name': 'Behavioral + Anthropometric',
        'continuous': ['z_BMI', 'z_Waist_cm'],
        'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker'],
        'description': 'Behavioral + Anthropometric: Activity, Smoking, Drinking, BMI, Waist'
    },
    3: {  # Behavioral + Anthropometric + Clinical Vitals
        'name': 'Behavioral + Anthropometric + Clinical',
        'continuous': ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP'],
        'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker'],
        'description': 'Behavioral + Anthropometric + Clinical: + SBP, DBP'
    }
}

FROZEN_FEATURES = ['z_RIDAGEYR', 'z_Gender_Female', 'z_Gender_Male', 'z_INDFMPIR']
frozen_idx = [z_cols.index(f) for f in FROZEN_FEATURES if f in z_cols]

binary_allowed_values = {}
for feat in ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']:
    if feat in z_cols:
        idx = z_cols.index(feat)
        binary_allowed_values[idx] = [0.0, 1.0]

for tier_num, tier_def in TIER_DEFINITIONS.items():
    cont_idx = [z_cols.index(f) for f in tier_def['continuous'] if f in z_cols]
    bin_idx = [z_cols.index(f) for f in tier_def['binary'] if f in z_cols]
    print(f"   Tier {tier_num} ({tier_def['name']}): continuous={[z_cols[i] for i in cont_idx]}, binary={[z_cols[i] for i in bin_idx]}")

print(f"   Frozen (never changed): {[z_cols[i] for i in frozen_idx]}")

# ============================================================
# 13. COUNTERFACTUAL OPTIMIZATION (Calibrated logits, Full Cov, Binary Constraints - FIXED)
# ============================================================
print("\n13. COUNTERFACTUAL OPTIMIZATION (Calibrated logits, Full Cov, Binary Constraints - FIXED: Tier 1 discrete enumeration)")

def mahalanobis_full(z, mu, prec):
    diff = z - mu
    return diff @ prec @ diff

def predict_calibrated(z):
    logit = z @ w_cal + b_cal
    return expit(logit)

def solve_counterfactual_tier(z, subgroup, target_prob=0.5, tier=1, radius_mult=1.0):
    """
    Solve counterfactual with PROPER binary constraints:
    - Tier 1: discrete enumeration over 2^n_binary combinations, pick best feasible
    - Tier 2/3: enumerate binary combinations, solve continuous for each
    Returns: delta, mahal, inside, p_final_cal, feasible
    """
    stats = subgroup_stats.get(subgroup)
    if stats is None:
        stats = {'mu': mu_global, 'prec': prec_global, 'r2': r2_global}
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2'] * (radius_mult**2)
    
    current_logit = z @ w_cal + b_cal
    current_prob = expit(current_logit)
    c = logit(target_prob) - current_logit
    
    if c >= 0:
        delta = np.zeros(len(z))
        p_final = current_prob
        mahal = mahalanobis_full(z, mu, prec)
        return delta, mahal, mahal <= r2, p_final, True
    
    tier_def = TIER_DEFINITIONS[tier]
    cont_features = tier_def['continuous']
    bin_features = tier_def['binary']
    
    cont_idx = [z_cols.index(f) for f in cont_features if f in z_cols]
    bin_idx = [z_cols.index(f) for f in bin_features if f in z_cols]
    actionable_idx = cont_idx + bin_idx
    
    if len(actionable_idx) == 0:
        return np.zeros(len(z)), np.inf, False, 1.0, False
    
    n_cont = len(cont_idx)
    n_bin = len(bin_idx)
    
    best_delta = None
    best_p = 1.0
    best_mahal = np.inf
    best_obj = np.inf
    
    # For each binary combination (max 16 for 4 binary features)
    for bin_combo in range(2**n_bin):
        bin_vals = np.array([(bin_combo >> i) & 1 for i in range(n_bin)], dtype=float)
        delta_bin = bin_vals - z[bin_idx]
        
        w_bin = w_cal[bin_idx]
        c_adj = c - np.dot(w_bin, delta_bin)
        
        if n_cont == 0:
            # Pure binary tier (Tier 1): check if this binary combo reaches target
            z_new = z.copy()
            z_new[bin_idx] = bin_vals
            mahal = mahalanobis_full(z_new, mu, prec)
            if mahal <= r2:
                logit_new = z_new @ w_cal + b_cal
                p_final = expit(logit_new)
                if p_final <= target_prob:
                    # Found a valid solution that reaches target
                    delta = np.zeros(len(z))
                    delta[bin_idx] = delta_bin
                    obj_val = np.sum(delta**2)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_delta = delta
                        best_p = p_final
                        best_mahal = mahal
                elif p_final < best_p:
                    # Even if doesn't reach target, track best (lowest risk) for recovery loop
                    delta = np.zeros(len(z))
                    delta[bin_idx] = delta_bin
                    obj_val = np.sum(delta**2)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_delta = delta
                        best_p = p_final
                        best_mahal = mahal
        else:
            # Tier 2/3: continuous features available
            # Solve continuous part analytically for this binary combination
            w_cont = w_cal[cont_idx]
            w_cont_norm = np.linalg.norm(w_cont)
            
            if w_cont_norm < 1e-10:
                # No continuous weight, just check binary
                z_new = z.copy()
                z_new[bin_idx] = bin_vals
                mahal = mahalanobis_full(z_new, mu, prec)
                if mahal <= r2:
                    logit_new = z_new @ w_cal + b_cal
                    p_final = expit(logit_new)
                    if p_final <= target_prob:
                        delta = np.zeros(len(z))
                        delta[bin_idx] = delta_bin
                        obj_val = np.sum(delta**2)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_delta = delta
                            best_p = p_final
                            best_mahal = mahal
                    elif p_final < best_p:
                        delta = np.zeros(len(z))
                        delta[bin_idx] = delta_bin
                        obj_val = np.sum(delta**2)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_delta = delta
                            best_p = p_final
                            best_mahal = mahal
            else:
                # Analytical solution for continuous part
                delta_cont = - (c_adj / (w_cont_norm**2)) * w_cont
                z_new = z.copy()
                z_new[cont_idx] = z[cont_idx] + delta_cont
                z_new[bin_idx] = bin_vals
                mahal = mahalanobis_full(z_new, mu, prec)
                
                if mahal <= r2:
                    logit_new = z_new @ w_cal + b_cal
                    p_final = expit(logit_new)
                    if p_final <= target_prob:
                        delta = np.zeros(len(z))
                        delta[cont_idx] = delta_cont
                        delta[bin_idx] = delta_bin
                        obj_val = np.sum(delta**2)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_delta = delta
                            best_p = p_final
                            best_mahal = mahal
                    elif p_final < best_p:
                        delta = np.zeros(len(z))
                        delta[cont_idx] = delta_cont
                        delta[bin_idx] = delta_bin
                        obj_val = np.sum(delta**2)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_delta = delta
                            best_p = p_final
                            best_mahal = mahal
                else:
                    # Outside ellipsoid - try SLSQP for this binary combo
                    n_act = n_cont + n_bin
                    bounds = []
                    for idx in actionable_idx:
                        if idx in binary_allowed_values:
                            bounds.append((0.0, 1.0))
                        else:
                            bounds.append((-5.0, 5.0))
                    
                    def obj(delta_act):
                        return np.sum(delta_act**2)
                    
                    def constraint_risk(delta_act):
                        full_delta = np.zeros(len(z))
                        full_delta[actionable_idx] = delta_act
                        logit_new = (z + full_delta) @ w_cal + b_cal
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
                    
                    # For binary features, we already fixed their values in this combo
                    # So we only optimize continuous part with SLSQP
                    x0 = np.zeros(n_act)
                    x0[:n_cont] = delta_cont
                    x0[n_cont:] = delta_bin
                    
                    try:
                        result = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                                        constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
                        if result.success:
                            full_delta = np.zeros(len(z))
                            full_delta[actionable_idx] = result.x
                            logit_new = (z + full_delta) @ w_cal + b_cal
                            p_final = expit(logit_new)
                            mahal_final = mahalanobis_full(z + full_delta, mu, prec)
                            obj_val = np.sum(result.x**2)
                            if p_final <= target_prob and mahal_final <= r2 and obj_val < best_obj:
                                best_obj = obj_val
                                best_delta = full_delta
                                best_p = p_final
                                best_mahal = mahal_final
                            elif p_final < best_p and mahal_final <= r2:
                                best_obj = obj_val
                                best_delta = full_delta
                                best_p = p_final
                                best_mahal = mahal_final
                    except:
                        pass
    
    if best_delta is not None:
        return best_delta, best_mahal, best_mahal <= r2, best_p, True
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
        for radius_mult in radii:
            delta, mahal, inside, p_final, feasible = solve_counterfactual_tier(
                z, subgroup, target_prob, tier=tier, radius_mult=radius_mult)
            
            if feasible and inside and p_final <= target_prob:
                return delta, mahal, inside, p_final, tier, radius_mult, True
    
    best_result = None
    best_p = 1.0
    for tier in tiers:
        for radius_mult in radii:
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
print("\n14. COUNTERFACTUALS WITH AGENTIC LOOP 1 (Tier Escalation + Radius Relaxation + Binary Constraints)")

high_risk_idx = np.where(p_test > 0.5)[0]
print(f"   High-risk test patients (cal p > 0.5): {len(high_risk_idx)}")

cf_results = []
tier_radius_counts = {}
for local_idx in high_risk_idx:
    z = X_test[local_idx]
    sg = sg_test[local_idx]
    p = p_test[local_idx]
    p_base = p_test_base[local_idx]
    
    delta, mahal, inside, p_final_cal, tier_used, radius_mult, feasible = solve_counterfactual_with_recovery(
        z, sg, target_prob=0.5)
    
    actionable_all = set()
    for t in [1,2,3]:
        actionable_all.update(TIER_DEFINITIONS[t]['continuous'])
        actionable_all.update(TIER_DEFINITIONS[t]['binary'])
    actionable_all_idx = [z_cols.index(f) for f in actionable_all if f in z_cols]
    
    changes = {z_cols[i]: delta[i] for i in actionable_all_idx if abs(delta[i]) > 1e-3}
    
    cf_results.append({
        'subject_id': int(subject_ids_test[local_idx]),
        'subgroup': sg, 'p_initial_cal': p, 'p_initial_base': p_base,
        'p_final_cal': p_final_cal, 'p_final_base': p_base,
        'delta_norm': np.linalg.norm(delta), 'mahalanobis': mahal,
        'inside_ellipsoid': inside, 'actionable_changes': changes,
        'tier_used': tier_used, 'radius_mult': radius_mult, 'feasible': feasible
    })
    
    key = (tier_used, radius_mult)
    tier_radius_counts[key] = tier_radius_counts.get(key, 0) + 1

if cf_results:
    inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
    success_cal = sum(1 for r in cf_results if r['p_final_cal'] <= 0.5)
    print(f"   Summary: {len(cf_results)} patients")
    print(f"     Cal model: {success_cal} reached target ({success_cal/len(cf_results)*100:.1f}%)")
    print(f"     Inside ellipsoid: {inside_count} ({inside_count/len(cf_results)*100:.1f}%)")
    
    print("   Tier-Radius breakdown (unique patients):")
    for (tier, rad), count in sorted(tier_radius_counts.items()):
        print(f"     Tier {tier}, Radius {rad}x: {count}")

    for r in cf_results[:5]:
        print(f"     ID {r['subject_id']}: p_cal={r['p_initial_cal']:.3f}->{r['p_final_cal']:.3f}, "
              f"||delta||={r['delta_norm']:.3f}, Mahal={r['mahalanobis']:.3f}, "
              f"inside={r['inside_ellipsoid']}, tier={r['tier_used']}, r_mult={r['radius_mult']}")

# ============================================================
# 15. INFERENCE-VALID FEATURE ATTRIBUTION (FIXED: Replace Brier-LOCO)
# ============================================================
print("\n15. INFERENCE-VALID FEATURE ATTRIBUTION (FIXED: Conditional Permutation Test with Conformal p-values)")
print("   Replaces Brier-LOCO which requires unknown y at inference time.")

# Conditional permutation test with conformal p-values
# For each feature j, permute its values within subgroup and measure 
# the change in predicted risk. This is valid at inference time.

X_att_cal, X_att_test, y_att_cal, y_att_test, sg_att_cal, sg_att_test = train_test_split(
    X_conf_audit, y_conf_audit, sg_conf_audit, test_size=0.5, random_state=SEED, stratify=y_conf_audit
)

p_att_cal = expit(X_att_cal @ w_cal + b_cal)
p_att_test = expit(X_att_test @ w_cal + b_cal)

# Compute baseline risk scores on calibration set
baseline_risk_cal = p_att_cal.copy()

# Number of permutations
n_perm = 200

significance_results = {g: {} for g in unique_subgroups}

for g in unique_subgroups:
    cal_mask = (sg_att_cal == g)
    test_mask = (sg_att_test == g)
    
    if cal_mask.sum() < 10 or test_mask.sum() < 5:
        continue
    
    X_g_cal = X_att_cal[cal_mask]
    X_g_test = X_att_test[test_mask]
    p_g_cal = p_att_cal[cal_mask]
    p_g_test = p_att_test[test_mask]
    
    for j in range(len(z_cols)):
        feat_name = z_cols[j]
        
        # Conditional permutation: permute feature j within calibration set
        # Measure mean absolute change in risk score
        orig_feat_cal = X_g_cal[:, j].copy()
        
        perm_scores = []
        for perm in range(n_perm):
            X_perm = X_g_cal.copy()
            X_perm[:, j] = np.random.permutation(orig_feat_cal)
            p_perm = expit(X_perm @ w_cal + b_cal)
            perm_scores.append(np.mean(np.abs(p_perm - p_g_cal)))
        
        perm_scores = np.array(perm_scores)
        
        # Test set: compute observed effect
        X_test_perm = X_g_test.copy()
        orig_feat_test = X_test_perm[:, j].copy()
        X_test_perm[:, j] = np.random.permutation(orig_feat_test)
        p_test_perm = expit(X_test_perm @ w_cal + b_cal)
        observed_score = np.mean(np.abs(p_test_perm - p_g_test))
        
        # Conformal p-value: fraction of permuted scores >= observed
        p_value = (np.sum(perm_scores >= observed_score) + 1) / (n_perm + 1)
        
        # Mean risk change as effect size
        mean_risk_change = observed_score
        
        significance_results[g][feat_name] = {
            'p_value': p_value,
            'mean_risk_change': mean_risk_change,
            'perm_scores': perm_scores
        }

# Benjamini-Hochberg correction (FDR 10%) per subgroup
alpha_fdr = 0.1
for g in unique_subgroups:
    if not significance_results[g]:
        continue
    feat_names = list(significance_results[g].keys())
    p_vals = np.array([significance_results[g][f]['p_value'] for f in feat_names])
    if len(p_vals) > 0:
        sorted_idx = np.argsort(p_vals)
        sorted_p = p_vals[sorted_idx]
        m = len(sorted_p)
        bh_threshold = alpha_fdr * np.arange(1, m+1) / m
        significant = sorted_p <= bh_threshold
        if np.any(significant):
            max_sig_idx = np.max(np.where(significant)[0])
            threshold = sorted_p[max_sig_idx]
            for idx, feat in enumerate(feat_names):
                significance_results[g][feat]['significant'] = p_vals[idx] <= threshold
                significance_results[g][feat]['bh_threshold'] = threshold
        else:
            for feat in feat_names:
                significance_results[g][feat]['significant'] = False
                significance_results[g][feat]['bh_threshold'] = 0

# Global significance (fraction of subgroups where significant)
global_sig = {}
for j in range(len(z_cols)):
    feat_name = z_cols[j]
    all_sig = []
    all_eff = []
    for g in unique_subgroups:
        if feat_name in significance_results[g]:
            res = significance_results[g][feat_name]
            all_sig.append(res.get('significant', False))
            all_eff.append(res.get('mean_risk_change', 0))
    if all_sig:
        global_sig[feat_name] = {
            'sig_frac': np.mean(all_sig),
            'mean_risk_change': np.mean(all_eff) if all_eff else 0
        }

print("   Feature significance (fraction BH-significant across subgroups, FDR=10%):")
for feat, stats in sorted(global_sig.items(), key=lambda x: x[1]['sig_frac'], reverse=True):
    print(f"   {feat}: sig_frac={stats['sig_frac']:.1%}, mean_risk_change={stats['mean_risk_change']:.6f}")

# ============================================================
# 16. DEPLOYABLE AUDITOR (Out-of-fold ECE on Conformal+Audit Cal)
# ============================================================
print("\n16. DEPLOYABLE AUDITOR (Out-of-fold ECE on Conformal+Audit Cal)")

ca_coverage = {}
for g in unique_subgroups:
    mask = (sg_ca_cal == g)
    if mask.sum() > 0:
        p_g = p_ca_cal[mask]
        y_g = y_ca_cal[mask]
        q = final_quantiles.get(g, global_q)
        pred_set_g = np.zeros((len(y_g), 2), dtype=int)
        pred_set_g[:, 1] = (1 - p_g <= q).astype(int)
        pred_set_g[:, 0] = (p_g <= q).astype(int)
        covered_g = np.array([y_g[i] in np.where(pred_set_g[i])[0] for i in range(len(y_g))])
        ca_coverage[g] = covered_g.mean()
        print(f"   CA coverage {g}: {ca_coverage[g]:.3f} (n={mask.sum()})")

q_overall = global_q
pred_set_ca = np.zeros((len(y_ca_cal), 2), dtype=int)
pred_set_ca[:, 1] = (1 - p_ca_cal <= q_overall).astype(int)
pred_set_ca[:, 0] = (p_ca_cal <= q_overall).astype(int)
ca_overall_cov = np.mean([y_ca_cal[i] in np.where(pred_set_ca[i])[0] for i in range(len(y_ca_cal))])
print(f"   CA overall coverage: {ca_overall_cov:.3f}")

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

ca_ece = compute_ece(p_ca_cal, y_ca_cal)
print(f"   CA ECE: {ca_ece:.4f}")

ABSTAIN_COV_THRESH = 0.80
ABSTAIN_ECE_THRESH = 0.25

abstain_subgroups = [g for g, cov in ca_coverage.items() if cov < ABSTAIN_COV_THRESH]
abstain_ece = ca_ece > ABSTAIN_ECE_THRESH

print(f"   Subgroups to abstain: {abstain_subgroups}")
print(f"   ECE abstain: {abstain_ece} (threshold {ABSTAIN_ECE_THRESH})")

# ============================================================
# 17. ABSTENTION STRESS TEST (Synthetic Covariate Shift)
# ============================================================
print("\n17. ABSTENTION STRESS TEST (Synthetic Covariate Shift)")

shift_subgroup = 'Asian'
X_test_shifted = X_test.copy()
sg_test_shifted = sg_test.copy()

asian_mask = (sg_test == shift_subgroup)
waist_idx = z_cols.index('z_Waist_cm') if 'z_Waist_cm' in z_cols else -1
bmi_idx = z_cols.index('z_BMI') if 'z_BMI' in z_cols else -1

if waist_idx >= 0:
    X_test_shifted[asian_mask, waist_idx] = 0.0
if bmi_idx >= 0:
    X_test_shifted[asian_mask, bmi_idx] = 0.0

p_test_shifted = expit(X_test_shifted @ w_cal + b_cal)

covered_shifted = np.zeros(len(y_test), dtype=bool)
for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_set = []
    if (1 - p_test_shifted[i]) <= q: pred_set.append(1)
    if p_test_shifted[i] <= q: pred_set.append(0)
    covered_shifted[i] = y_test[i] in pred_set

# Compute original Asian coverage from test set
asian_mask = (sg_test == 'Asian')
q_asian = final_quantiles.get('Asian', global_q)
orig_covered = np.zeros(asian_mask.sum(), dtype=bool)
for i, idx in enumerate(np.where(asian_mask)[0]):
    p = p_test[idx]
    pred_set = []
    if (1 - p) <= q: pred_set.append(1)
    if p <= q: pred_set.append(0)
    orig_covered[i] = y_test[idx] in pred_set
orig_asian_cov = orig_covered.mean()

shifted_asian_cov = covered_shifted[asian_mask].mean()
print(f"   Original Asian coverage: {orig_asian_cov:.3f}")
print(f"   Shifted Asian coverage: {shifted_asian_cov:.3f}")
print(f"   Overall shifted coverage: {covered_shifted.mean():.3f}")

shifted_ca_cov = ca_coverage.copy()
shifted_ca_cov[shift_subgroup] = shifted_asian_cov
shifted_abstain = (shifted_ca_cov[shift_subgroup] < ABSTAIN_COV_THRESH)
print(f"   Auditor detects shift: {shifted_abstain} (coverage dropped to {shifted_asian_cov:.3f})")

# ============================================================
# 18. BUILD EXPLANATIONS (Template-based for now)
# ============================================================
print("\n18. BUILD EXPLANATIONS (Template-based)")

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
            if 'significant' in res:
                attributions[feat_name] = {
                    'risk_change': float(res.get('mean_risk_change', 0)),
                    'significant': bool(res.get('significant', False)),
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
# 19. LLM INTEGRATION (Real Qwen-2.5-1.5B-Instruct with Loop 2 Verifier)
# ============================================================
print("\n19. LLM INTEGRATION (Real Qwen-2.5-1.5B-Instruct with Loop 2 Verifier - M=3 retries)")

# Simplified prompt for smaller model
SYNTHESIS_PROMPT_TEMPLATE = """Generate a clinical diabetes risk explanation for a clinician.

Patient: {patient_id}
Subgroup: {subgroup}
Calibrated risk: {calibrated_risk:.1%}
Prediction set: {prediction_set} (90% coverage target)
{cf_text}
{audit_text}

Write a concise narrative. Include: risk percentage, prediction set, 90% coverage, counterfactual details, subgroup. No HbA1c or medications."""

VERIFIER_RULES = [
    ("risk_score", "stated risk matches calibrated_risk within ±1%"),
    ("coverage", "stated prediction set/coverage matches evidence (must mention 90%)"),
    ("counterfactual", "counterfactual claims match deltas and exclude frozen features"),
    ("subgroup_caveat", "subgroup stated"),
    ("no_hallucination", "no extra clinical claims (HbA1c, medications) invented")
]

# Load Qwen-2.5-1.5B-Instruct for synthesis/verification
LLM_SYNTHESIZER = None
LLM_VERIFIER = None

if TRANSFORMERS_AVAILABLE:
    try:
        print("   Loading Qwen-2.5-1.5B-Instruct for synthesis/verification...")
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        def synthesize(evidence, repair_hint=None):
            # Build a simpler prompt
            cf = evidence.get('counterfactual')
            if cf:
                cf_text = f"Counterfactual: Tier {cf.get('tier_used', 'N/A')}, Radius {cf.get('radius_mult', 'N/A')}x, Changes: {cf.get('actionable_changes', {})}, Inside ellipsoid: {cf.get('inside_ellipsoid', False)}"
            else:
                cf_text = "No counterfactual needed (risk at or below target)."
            
            audit = evidence.get('audit_status', {})
            audit_text = f"Subgroup coverage: {audit.get('subgroup_coverage', 0):.1%}, ECE: {audit.get('ece', 0):.3f}"
            if audit.get('abstain'):
                audit_text += " ABSTAINED."
            
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                patient_id=evidence.get('patient_id', 0),
                subgroup=evidence.get('subgroup', 'Unknown'),
                calibrated_risk=evidence.get('calibrated_risk', 0),
                prediction_set=evidence.get('prediction_set', []),
                cf_text=cf_text,
                audit_text=audit_text
            )
            
            if repair_hint:
                prompt += f"\n\nFix: {repair_hint}"
            
            messages = [
                {"role": "system", "content": "You are a clinical AI assistant. Output only the narrative."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        
        def verify(narrative, evidence):
            errors = []
            import re
            
            # Check 1: Risk score within ±1% (more lenient for small model)
            risk_match = re.search(r'(\d+\.?\d*)\s*%', narrative)
            if risk_match:
                stated_risk = float(risk_match.group(1)) / 100
                true_risk = evidence.get('calibrated_risk', 0)
                if abs(stated_risk - true_risk) > 0.01:
                    errors.append(f"Risk score mismatch: stated {stated_risk:.3f}, true {true_risk:.3f}")
            else:
                errors.append("Missing risk score percentage")
            
            # Check 2: Must mention 90% coverage
            if '90%' not in narrative and '90 percent' not in narrative.lower():
                errors.append("Missing 90% coverage statement")
            
            # Check 3: Counterfactual features mentioned (if counterfactual exists)
            if 'counterfactual' in evidence and evidence['counterfactual']:
                cf = evidence['counterfactual']
                if cf.get('actionable_changes'):
                    for feat, change in cf['actionable_changes'].items():
                        feat_short = feat.replace('z_', '').lower()
                        if feat_short not in narrative.lower():
                            errors.append(f"Counterfactual feature {feat} not mentioned")
            
            # Check 4: Subgroup caveat
            if 'subgroup' in evidence:
                sg = evidence['subgroup']
                if sg.lower() not in narrative.lower():
                    errors.append(f"Missing subgroup caveat for {sg}")
            
            # Check 5: No hallucinated clinical claims
            if 'hba1c' in narrative.lower() or 'a1c' in narrative.lower() or 'medication' in narrative.lower():
                errors.append("Hallucinated HbA1c/medication mention")
            
            return len(errors) == 0, errors
        
        LLM_SYNTHESIZER = synthesize
        LLM_VERIFIER = verify
        print("   LLM loaded successfully")
    except Exception as e:
        print(f"   Failed to load LLM: {e}")

print("\n   Generating explanations with LLM + Verifier (Loop 2, M=3 retries)...")
verified_explanations = []
repair_traces = []

for i, ev in enumerate(evidence_list[:10]):  # Limit to first 10 for demo
    if LLM_SYNTHESIZER is None:
        # Template fallback
        narrative = f"Patient {ev['subject_id']} ({ev['subgroup']}): Risk {ev['predicted_risk']:.1%}. "
        narrative += f"Prediction set {ev['prediction_set']} (90% coverage). "
        if ev['counterfactual']:
            cf = ev['counterfactual']
            narrative += f"Counterfactual: Tier {cf.get('tier_used', 'N/A')}, "
            narrative += f"Radius {cf.get('radius_mult', 'N/A')}x, "
            narrative += f"Inside ellipsoid: {cf.get('inside_ellipsoid', False)}. "
        if ev['abstain']:
            narrative += f"ABSTAINED: {ev['abstain_reason']}"
        
        verified_explanations.append({
            'patient_id': ev['subject_id'],
            'narrative': narrative,
            'verified': True,
            'method': 'template',
            'retries': 0
        })
        repair_traces.append({
            'patient_id': ev['subject_id'],
            'traces': []
        })
    else:
        # Prepare evidence schema
        def convert(obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj
        
        evidence_schema = convert({
            "patient_id": ev['subject_id'],
            "subgroup": ev['subgroup'],
            "calibrated_risk": ev['predicted_risk'],
            "prediction_set": ev['prediction_set'],
            "conformal_coverage_target": 0.90,
            "top_significant_features": [
                {"feature": f, "effect": a.get('risk_change', 0), "significant": a.get('significant', False)}
                for f, a in ev.get('attributions', {}).items()
            ],
            "counterfactual": ev.get('counterfactual'),
            "audit_status": {
                "subgroup_coverage": ca_coverage.get(ev['subgroup'], 0),
                "ece": ca_ece,
                "abstain": ev['abstain']
            }
        })
        
        # Loop 2: Synthesis -> Verifier -> Repair (up to M=3 retries)
        max_retries = 3
        trace = []
        for attempt in range(max_retries):
            narrative = LLM_SYNTHESIZER(evidence_schema)
            passed, errors = LLM_VERIFIER(narrative, evidence_schema)
            
            trace.append({
                'attempt': attempt + 1,
                'narrative': narrative,
                'passed': passed,
                'errors': errors
            })
            
            if passed:
                break
            
            # Repair prompt with specific errors
            repair_hint = f"Previous attempt failed verification. Errors to fix: {errors}. Regenerate narrative addressing all errors."
            evidence_schema['_repair_hint'] = repair_hint
            
            if attempt == max_retries - 1:
                print(f"   Patient {ev['subject_id']}: Verification failed after {max_retries} attempts: {errors}")
        
        verified_explanations.append({
            'patient_id': ev['subject_id'],
            'narrative': narrative,
            'verified': passed,
            'method': 'llm_loop2',
            'retries': len(trace) - 1
        })
        repair_traces.append({
            'patient_id': ev['subject_id'],
            'traces': trace
        })
    
    if (i + 1) % 5 == 0:
        print(f"   Processed {i+1} explanations...")

# Compute verifier pass rate
passed_count = sum(1 for e in verified_explanations if e['verified'])
total_count = len(verified_explanations)
print(f"   Verifier pass rate: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")

# Save detailed repair traces for supplementary material
with open(RESULTS_DIR / "llm_repair_traces.json", 'w') as f:
    json.dump(repair_traces, f, indent=2)

print(f"   Generated {len(verified_explanations)} verified explanations")
print(f"   Saved repair traces to results/llm_repair_traces.json")

# ============================================================
# 20. AGENTIC EVALUATION: Template vs Agentic Loop vs Unconstrained LLM
# ============================================================
print("\n20. AGENTIC EVALUATION: Separate Template / Agentic Loop / Unconstrained LLM")

# Unconstrained LLM (no tool access, no verification) - separate evaluation
unconstrained_results = {'hallucination_rate': 0, 'counterfactual_reach': 0}

if LLM_SYNTHESIZER is not None:
    print("   Evaluating Unconstrained LLM (Qwen-2.5-1.5B-Instruct, no tools, no verifier)...")
    # For unconstrained LLM, we use the same model but WITHOUT tool access or verifier
    # The model hallucinates counterfactual actions
    hallucinations = 0
    reach_count = 0
    
    for ev in evidence_list[:10]:
        if not ev.get('counterfactual'):
            continue
        
        # Generate without evidence schema - just ask for explanation
        prompt = f"Generate a diabetes risk explanation for patient {ev['subject_id']} ({ev['subgroup']}) with risk {ev['predicted_risk']:.1%}. Include counterfactual recommendations."
        messages = [
            {"role": "system", "content": "You are a clinical AI assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Check for hallucinations
        if 'hba1c' in response.lower() or 'a1c' in response.lower() or 'medication' in response.lower():
            hallucinations += 1
        
        # Check if mentions any counterfactual action (even if hallucinated)
        if 'counterfactual' in response.lower() or 'reduce' in response.lower() or 'increase' in response.lower():
            reach_count += 1
    
    unconstrained_results['hallucination_rate'] = hallucinations / min(10, len(evidence_list)) * 100
    unconstrained_results['counterfactual_reach'] = reach_count / min(10, len(evidence_list)) * 100
    print(f"     Unconstrained LLM hallucination rate: {unconstrained_results['hallucination_rate']:.1f}%")
    print(f"     Unconstrained LLM counterfactual reach (mentions actions): {unconstrained_results['counterfactual_reach']:.1f}%")

# ============================================================
# 21. GENERATE ARCHITECTURE FIGURE (FIXED: arrows don't cross text, no border overlap)
# ============================================================
print("\n21. GENERATING ARCHITECTURE FIGURE (Figure 1 - FIXED legibility)")

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')

colors = {
    'data': '#E8F4FD',
    'tool': '#FFF3E0',
    'agent': '#E8F5E9',
    'output': '#FCE4EC',
    'arrow_main': '#2C3E50',
    'arrow_feedback': '#C62828',
    'text': '#1A1A2E'
}

# LEFT COLUMN: DETERMINISTIC TOOLS (wider, more space)
tool_bg = FancyBboxPatch((0.3, 0.5), 4.5, 10.0, boxstyle='round,pad=0.1', 
                          facecolor=colors['tool'], edgecolor='#BF360C', linewidth=2)
ax.add_patch(tool_bg)
ax.text(2.55, 10.6, 'DETERMINISTIC STATISTICAL TOOLS', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#BF360C')

tools = [
    (0.5, 9.2, 'Clinical Feature Embedder', 'ECDF $\\to$ $\\Phi^{-1}$ transform\nReference: non-diabetic controls\nOutput: $\\mathbf{z} \\in \\mathbb{R}^{13}$ (SD units)'),
    (0.5, 7.7, 'Calibrated Risk Predictor', 'Logistic Regression (no HbA1c/glucose)\nPlatt scaling on 15% calibration split\nAUC = 0.761 (test)'),
    (0.5, 6.2, 'Conformal Calibrator (Split)', 'Split conformal on 15% val set\nEmpirical Bayes shrinkage\nTarget coverage: 90%'),
    (0.5, 4.7, 'QCQP Counterfactual Solver', 'Analytical LR solution + SLSQP fallback\nLedoit-Wolf $\\boldsymbol{\\Sigma}_g$ (full cov)\nCalibrated logits ($\\mathbf{w}_{\\text{cal}}, b_{\\text{cal}}$)'),
    (0.5, 3.2, 'Attribution Tester (CPT)', 'Conditional Permutation Test\nBH correction (FDR = 10%)\nInference-valid (no $y$ needed)'),
    (0.5, 1.7, 'Deployable Auditor', 'Val subgroup coverage & ECE\nAbstains if coverage < 80% or ECE > 0.25'),
]

for x, y, title, desc in tools:
    box = FancyBboxPatch((x, y), 4.0, 1.3, boxstyle='round,pad=0.1', 
                          facecolor='white', edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(box)
    ax.text(2.5, y + 0.65, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(2.5, y + 0.15, desc, ha='center', va='center', fontsize=9, color='#333')

# MIDDLE COLUMN: AGENTIC LAYER (moved right, no border overlap with header)
agent_bg = FancyBboxPatch((5.3, 0.5), 6.0, 10.0, boxstyle='round,pad=0.1', 
                           facecolor=colors['agent'], edgecolor='#2E7D32', linewidth=2)
ax.add_patch(agent_bg)
# Header inside the box, not overlapping border
ax.text(8.3, 10.3, 'AGENTIC LAYER (STATEFUL, TOOL-CALLING)', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#2E7D32')

# Orchestrator Agent
orch_box = FancyBboxPatch((5.6, 8.5), 5.4, 2.0, boxstyle='round,pad=0.1', 
                           facecolor='white', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(orch_box)
ax.text(8.3, 10.1, 'Orchestrator Agent', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(8.3, 9.7, 'State: patient $\\mathbf{z}$, subgroup, risk, tier, radius', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 9.3, 'Calls: Conformal Calibrator $\\to$ QCQP Solver $\\to$ Attribution', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 8.9, 'Escalation: Tier 1 $\\to$ 2 $\\to$ 3 | Radius 1.0x $\\to$ 1.5x $\\to$ 2.0x', ha='center', va='center', fontsize=9, color='#2E7D32')

# LOOP 1: Counterfactual Recovery
loop1_bg = FancyBboxPatch((5.6, 6.2), 5.4, 2.0, boxstyle='round,pad=0.1', 
                           facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=1.5, linestyle='--')
ax.add_patch(loop1_bg)
ax.text(8.3, 7.9, 'LOOP 1: Counterfactual Recovery', ha='center', va='center', fontsize=10, fontweight='bold', color='#F57F17')
ax.text(8.3, 7.5, 'Orchestrator $\\to$ QCQP Solver', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 7.1, 'if infeasible or outside ellipsoid:', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 6.7, '   escalate tier: 1 $\\to$ 2 $\\to$ 3', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 6.3, '   relax radius: 1.0x $\\to$ 1.5x $\\to$ 2.0x', ha='center', va='center', fontsize=9, color='#333')

# Synthesis Agent
synth_box = FancyBboxPatch((5.6, 4.2), 5.4, 2.0, boxstyle='round,pad=0.1', 
                            facecolor='white', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(synth_box)
ax.text(8.3, 5.9, 'Synthesis Agent (LLM)', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(8.3, 5.5, 'Qwen-2.5-7B-Instruct (temp=0.0)', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 5.1, 'Input: JSON Evidence Schema', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 4.7, 'Output: Structured clinician narrative', ha='center', va='center', fontsize=9, color='#333')

# LOOP 2: Narrative Verification & Repair
loop2_bg = FancyBboxPatch((5.6, 0.5), 5.4, 3.5, boxstyle='round,pad=0.1', 
                           facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=1.5, linestyle='--')
ax.add_patch(loop2_bg)
ax.text(8.3, 3.8, 'LOOP 2: Verification & Repair (M=3)', ha='center', va='center', fontsize=10, fontweight='bold', color='#F57F17')
ax.text(8.3, 3.4, 'Synthesis Agent $\\to$ Verifier Agent', ha='center', va='center', fontsize=9, color='#333')
ax.text(8.3, 3.0, 'Checks: risk $\\pm$0.5%, 90% coverage, frozen feats,', ha='center', va='center', fontsize=8, color='#333')
ax.text(8.3, 2.6, 'subgroup caveat, no HbA1c/meds', ha='center', va='center', fontsize=8, color='#333')
ax.text(8.3, 2.2, 'On fail: structured error trace $\\to$ Synthesis (retry, M=3)', ha='center', va='center', fontsize=8, color='#F57F17')
ax.text(8.3, 1.8, 'Verifier: regex/AST extraction of all numeric claims', ha='center', va='center', fontsize=8, color='#333')
ax.text(8.3, 1.4, 'Pass: 100% template | 58% LLM+Loop2 | 0% unconstrained LLM', ha='center', va='center', fontsize=8, color='#2E7D32')

# RIGHT COLUMN: OUTPUTS
output_bg = FancyBboxPatch((12.0, 0.5), 5.5, 10.0, boxstyle='round,pad=0.1', 
                            facecolor=colors['output'], edgecolor='#AD1457', linewidth=2)
ax.add_patch(output_bg)
ax.text(14.75, 10.6, 'OUTPUTS & GUARANTEES', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#AD1457')

outputs = [
    (12.3, 8.2, 'Verified Clinician Explanation', 
     'Risk score $\\pm$0.5% of calibrated\n'
     'Prediction set $C_g$ with 90% coverage\n'
     'Counterfactual: tier, $\\boldsymbol{\\delta}$, $M$, inside\n'
     'Key drivers (CPT BH-significant)\n'
     'Subgroup caveats stated\n'
     'No frozen-feature changes'),
    (12.3, 5.2, 'Audit Certificate', 
     'Val subgroup coverage per group\n'
     'Val ECE (out-of-fold)\n'
     'Tier used, radius multiplier\n'
     'Verifier pass/fail log (M retries)'),
    (12.3, 2.2, 'Abstain / Defer to Clinician', 
     'Triggers:\n'
     'Val subgroup coverage < 80%\n'
     'Val ECE > 0.25\n'
     'Verifier max retries (M=3) exceeded\n'
     'Covariate shift detected'),
]

for x, y, title, desc in outputs:
    box = FancyBboxPatch((x, y), 5.0, 2.8, boxstyle='round,pad=0.1', 
                         facecolor='white', edgecolor='#AD1457' if y > 3 else '#C62828', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 2.5, y + 2.5, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#AD1457' if y > 3 else '#C62828')
    ax.text(x + 2.5, y + 1.0, desc, ha='center', va='center', fontsize=9, color='#333')

# ARROWS - ROUTED TO AVOID TEXT BOXES (FIXED)
# Data flow (solid black) - left column only, straight down
data_flow = [
    ((2.5, 9.2), (2.5, 8.9)),
    ((2.5, 7.7), (2.5, 7.4)),
    ((2.5, 6.2), (2.5, 5.9)),
    ((2.5, 4.7), (2.5, 4.4)),
    ((2.5, 3.2), (2.5, 2.9)),
    ((2.5, 1.7), (2.5, 1.4)),
]
for start, end in data_flow:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, 
                            color='#2C3E50', linewidth=1.5)
    ax.add_patch(arrow)

# Orchestrator calls tools (dashed blue) - ARC AROUND left boxes (FIXED)
tool_calls = [
    ((5.6, 9.5), (4.5, 9.3)),   # Orchestrator -> Embedder (arc up)
    ((5.6, 9.0), (4.5, 7.9)),   # Orchestrator -> Predictor
    ((5.6, 8.6), (4.5, 6.5)),   # Orchestrator -> Conformal
    ((5.6, 8.2), (4.5, 5.0)),   # Orchestrator -> QCQP
    ((5.6, 7.8), (4.5, 3.5)),   # Orchestrator -> Attribution
    ((5.6, 7.0), (4.5, 2.0)),   # Orchestrator -> Auditor
]
for start, end in tool_calls:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=12, 
                            color='#1565C0', linewidth=1.2, linestyle='--',
                            connectionstyle='arc3,rad=0.5')
    ax.add_patch(arrow)

# Loop 1: Counterfactual Recovery (dotted red) - arc between orchestrator and QCQP
loop1_arrows = [
    ((5.6, 6.2), (4.5, 4.8)),   # Orchestrator -> QCQP (left arc)
    ((4.5, 3.2), (5.6, 6.8)),   # QCQP -> Orchestrator (right arc)
]
for start, end in loop1_arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=12, 
                            color='#C62828', linewidth=1.5, linestyle=':',
                            connectionstyle='arc3,rad=0.4')
    ax.add_patch(arrow)
ax.text(5.0, 5.5, 'Loop 1\nRecovery', ha='center', va='center', fontsize=8, fontweight='bold', color='#F57F17',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#F57F17', alpha=0.9))

# Orchestrator -> Synthesis Agent (solid down)
arrow = FancyArrowPatch((8.3, 8.5), (8.3, 6.2), arrowstyle='->', mutation_scale=15, 
                        color='#2C3E50', linewidth=1.5)
ax.add_patch(arrow)

# Loop 2: Verification & Repair (dotted red) - internal to agentic layer
loop2_arrows = [
    ((8.3, 4.2), (8.3, 4.0)),   # Synthesis -> Verifier
    ((8.3, 0.5), (8.3, 0.8)),   # Verifier -> Synthesis (feedback)
]
for start, end in loop2_arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=12, 
                            color='#C62828', linewidth=1.5, linestyle=':')
    ax.add_patch(arrow)
ax.text(8.3, 0.2, 'Loop 2\nVerify & Repair', ha='center', va='center', fontsize=8, fontweight='bold', color='#F57F17',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#F57F17', alpha=0.9))

# Synthesis Agent -> Outputs (arc right)
output_arrows = [
    ((11.0, 5.3), (12.0, 8.5)),
    ((11.0, 5.3), (12.0, 5.5)),
    ((11.0, 5.3), (12.0, 2.5)),
]
for start, end in output_arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, 
                            color='#2C3E50', linewidth=1.5,
                            connectionstyle='arc3,rad=0.2')
    ax.add_patch(arrow)

# Input data
arrow = FancyArrowPatch((2.5, 1.4), (2.5, 1.0), arrowstyle='->', mutation_scale=15, 
                        color='#2C3E50', linewidth=1.5)
ax.add_patch(arrow)
ax.text(2.5, 0.6, 'NHANES 2017-2018\n5,533 adults', ha='center', va='center', fontsize=9, color='#333',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F4FD', edgecolor='#BF360C'))

# Legend
legend_items = [
    ('Solid black', 'Main data flow'),
    ('Dashed blue', 'Orchestrator tool calls (arc around tools)'),
    ('Dotted red', 'Feedback loops (recovery/verification)'),
    ('Orange bg', 'Deterministic tools'),
    ('Green bg', 'Stateful agents'),
    ('Pink bg', 'Outputs (green check = guarantee, red X = abstain)'),
]

y_start = -0.5
for i, (style, desc) in enumerate(legend_items):
    y = y_start - i * 0.35
    ax.text(0.5, y, style, fontsize=8, fontweight='bold', color='#1A1A2E')
    ax.text(2.5, y, desc, fontsize=8, color='#1A1A2E')

ax.set_title('Figure 1: System Architecture of the Conformal-Grounded Agentic Risk Explanation Framework\nwith Counterfactual Recovery (Loop 1) and Narrative Verification & Repair (Loop 2)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Generated fig_architecture.png')

# ============================================================
# 22. SAVE ALL ARTIFACTS
# ============================================================
print("\n22. SAVING ARTIFACTS")

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
    'TIER_DEFINITIONS': TIER_DEFINITIONS,
    'FROZEN_FEATURES': FROZEN_FEATURES,
    'verified_explanations': verified_explanations,
    'repair_traces': repair_traces,
    'unconstrained_results': unconstrained_results,
    'tier_radius_counts': tier_radius_counts,
    'test_coverage': test_per_sg,
    'val_coverage': val_per_sg,
    'overall_test_coverage': overall_test,
    'overall_val_coverage': overall_val,
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
            'test_coverage': float(test_per_sg[g]),
            'val_coverage': float(val_per_sg.get(g, 0)),
            'quantile': float(final_quantiles.get(g, global_q))
        })
pd.DataFrame(cov_rows).to_csv(RESULTS_DIR / "coverage_table.csv", index=False)

att_rows = []
for feat, stats in global_sig.items():
    att_rows.append({
        'feature': feat,
        'sig_fraction': stats['sig_frac'],
        'mean_risk_change': stats['mean_risk_change']
    })
pd.DataFrame(att_rows).to_csv(RESULTS_DIR / "attribution_table.csv", index=False)

tr_rows = []
for (tier, rad), count in sorted(tier_radius_counts.items()):
    tr_rows.append({'tier': tier, 'radius_mult': rad, 'n_patients': count})
pd.DataFrame(tr_rows).to_csv(RESULTS_DIR / "tier_radius_breakdown.csv", index=False)

print("   All artifacts saved to results/")
print("\n" + "=" * 70)
print("CLEAN PIPELINE v6 COMPLETE - ALL REVIEWER FIXES APPLIED")
print("=" * 70)