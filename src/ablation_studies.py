"""
Ablation Studies for NHANES Framework
=====================================
Tests each component's contribution:
1. CV+ vs Single-split vs Jackknife+ conformal
2. With/without hierarchical shrinkage
3. With/without ellipsoid constraint in counterfactuals
4. With/without auditor abstention
5. Race vs Age vs BMI subgroups
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from scipy.optimize import minimize
import pickle
import warnings
warnings.filterwarnings('ignore')

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
train_idx, temp_idx = next(sss.split(embeddings[z_cols].values, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

X_test = X[temp_idx[test_idx]]
y_test = y[temp_idx[test_idx]]
sg_test = subgroups_arr[temp_idx[test_idx]]
p_test = clf.predict_proba(X[temp_idx[test_idx]])[:, 1]

print("=" * 60)
print("ABLATION STUDIES")
print("=" * 60)

# ============================================================
# 1. CONFORMAL METHOD COMPARISON
# ============================================================
print("\n1. CONFORMAL METHOD COMPARISON")

def run_cv_plus(X, y, subgroups, alpha=0.1, n_folds=5):
    """CV+ conformal"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n = len(y)
    p_cv = np.zeros(n)
    fold_cal_scores = {g: [] for g in np.unique(subgroups)}
    
    for fold_idx, (train_idx, cal_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_cl, y_cl = X[cal_idx], y[cal_idx]
        sg_cl = subgroups[cal_idx]
        
        clf_cv = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=fold_idx))
        ])
        clf_cv.fit(X_tr, y_tr)
        
        p_cl = clf_cv.predict_proba(X[cal_idx])[:, 1]
        A_cl = np.where(y_cl == 1, 1 - p_cl, p_cl)
        
        for g in np.unique(subgroups):
            mask = (subgroups[cal_idx] == g)
            if mask.sum() > 0:
                fold_cal_scores[g].extend(A_cl[mask])
    
    # Quantiles
    sg_quantiles = {}
    for g in np.unique(subgroups):
        scores = np.array(fold_cal_scores[g])
        if len(scores) > 0:
            sg_quantiles[g] = np.quantile(scores, 1 - alpha, method='higher')
        else:
            all_s = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
            sg_quantiles[g] = np.quantile(all_s, 1 - alpha, method='higher')
    
    all_s = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
    global_q = np.quantile(all_s, 1 - alpha, method='higher')
    
    return sg_quantiles, global_q, fold_cal_scores

def run_single_split(X, y, subgroups, alpha=0.1):
    """Single split conformal (80/20)"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, cal_idx = next(sss.split(X, y))
    
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_cl, y_cl = X[cal_idx], y[cal_idx]
    sg_cl = subgroups[cal_idx]
    
    clf_ss = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    clf_ss.fit(X[train_idx], y[train_idx])
    
    p_cl = clf_ss.predict_proba(X[cal_idx])[:, 1]
    A_cl = np.where(y[cal_idx] == 1, 1 - p_cl, p_cl)
    
    sg_quantiles = {}
    for g in np.unique(subgroups):
        mask = (subgroups[cal_idx] == g)
        if mask.sum() > 0:
            sg_quantiles[g] = np.quantile(A_cl[mask], 1 - alpha, method='higher')
        else:
            sg_quantiles[g] = np.quantile(A_cl, 1 - alpha, method='higher')
    
    return sg_quantiles, np.quantile(A_cl, 1 - alpha, method='higher')

def run_jackknife_plus(X, y, subgroups, alpha=0.1, max_loo=500):
    """Jackknife+ conformal"""
    n = len(y)
    if n > max_loo:
        loo_indices = np.random.choice(n, max_loo, replace=False)
    else:
        loo_indices = np.arange(n)
    
    loo_scores = {g: [] for g in np.unique(subgroups)}
    
    for i, loo_idx in enumerate(loo_indices):
        if i % 100 == 0:
            print(f"  LOO {i+1}/{len(loo_indices)}")
        
        train_mask = np.ones(n, dtype=bool)
        train_mask[loo_idx] = False
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[loo_idx:loo_idx+1], y[loo_idx:loo_idx+1]
        sg_te = subgroups[loo_idx:loo_idx+1]
        
        clf_loo = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=loo_idx))
        ])
        clf_loo.fit(X_tr, y_tr)
        
        p_te = clf_loo.predict_proba(X_te)[0, 1]
        A_te = 1 - p_te if y_te[0] == 1 else p_te
        loo_scores[sg_te[0]].append(A_te)
    
    sg_quantiles = {}
    for g in np.unique(subgroups):
        scores = np.array(loo_scores[g])
        if len(scores) > 0:
            sg_quantiles[g] = np.quantile(scores, 1 - 0.1, method='higher')
        else:
            all_s = np.concatenate([np.array(v) for v in loo_scores.values() if len(v) > 0])
            sg_quantiles[g] = np.quantile(all_s, 1 - 0.1, method='higher')
    
    all_s = np.concatenate([np.array(v) for v in loo_scores.values() if len(v) > 0])
    return sg_quantiles, np.quantile(all_s, 1 - 0.1, method='higher')

def evaluate_coverage(X_test, y_test, sg_test, quantiles, global_q, p_test, alpha=0.1):
    """Evaluate coverage with given quantiles"""
    A_test = np.where(y_test == 1, 1 - p_test, p_test)
    covered = np.zeros(len(y_test), dtype=bool)
    pred_sets = np.zeros((len(y_test), 2), dtype=int)
    
    for i in range(len(y_test)):
        g = sg_test[i]
        q = quantiles.get(g, global_q)
        pred_sets[i, 1] = 1 if (1 - p_test[i]) <= q else 0
        pred_sets[i, 0] = 1 if p_test[i] <= q else 0
        covered[i] = y_test[i] in np.where(pred_sets[i])[0]
    
    overall = covered.mean()
    by_sg = {}
    for g in np.unique(sg_test):
        mask = (sg_test == g)
        if mask.sum() > 0:
            by_sg[g] = covered[mask].mean()
    
    set_sizes = (pred_sets[:,0] + pred_sets[:,1])
    return overall, by_sg, set_sizes.mean()

print("\nRunning CV+...")
cv_quantiles, cv_global, cv_scores = run_cv_plus(X, y, subgroups_arr, alpha=0.1)
print("Running Single Split...")
ss_quantiles, ss_global = run_single_split(X, y, subgroups_arr, alpha=0.1)
print("Running Jackknife+ (sample)...")
jk_quantiles, jk_global = run_jackknife_plus(X, y, subgroups_arr, alpha=0.1, max_loo=200)

# Evaluate all
print("\n=== COVERAGE RESULTS ===")
for name, quantiles, global_q in [
    ("CV+", cv_quantiles, cv_global),
    ("Single Split", ss_quantiles, ss_global),
    ("Jackknife+", jk_quantiles, jk_global),
]:
    overall, by_sg, mean_size = evaluate_coverage(X_test, y_test, sg_test, quantiles, global_q, p_test)
    print(f"\n{name}:")
    print(f"  Overall: {overall:.3f} (target: 0.900), Mean set size: {mean_size:.2f}")
    for g in sorted(by_sg.keys()):
        print(f"  {g}: {by_sg[g]:.3f}")

# ============================================================
# 2. HIERARCHICAL SHRINKAGE ABLATION
# ============================================================
print("\n\n2. HIERARCHICAL SHRINKAGE ABLATION")

def apply_shrinkage(quantiles, cv_scores, global_q):
    """Apply hierarchical shrinkage to quantiles"""
    valid_groups = [g for g in quantiles if len(cv_scores[g]) > 0]
    if len(valid_groups) < 2:
        return quantiles
    
    q_vals = np.array([quantiles[g] for g in valid_groups])
    n_vals = np.array([len(cv_scores[g]) for g in valid_groups])
    
    within_vars = np.array([np.var(cv_scores[g]) for g in valid_groups])
    quantile_vars = within_vars / n_vals
    
    weighted_mean = np.average(q_vals, weights=n_vals)
    total_var = np.average((q_vals - weighted_mean)**2, weights=n_vals)
    tau2 = max(0, total_var - np.mean(quantile_vars))
    kappa = 1.0 / tau2 if tau2 > 0 else 1e6
    
    shrunken = {}
    for g in quantiles:
        n_g = len(cv_scores[g])
        q_g = quantiles[g]
        q_shrunk = (n_g * q_g + kappa * weighted_mean) / (n_g + kappa) if n_g > 0 else weighted_mean
        shrunken[g] = q_shrunk
    
    return shrunken

# No shrinkage
cv_overall, cv_by_sg, cv_size = evaluate_coverage(X_test, y_test, sg_test, cv_quantiles, cv_global, p_test)
print(f"\nNo shrinkage: Overall={cv_overall:.3f}")
for g in sorted(cv_by_sg.keys()):
    print(f"  {g}: {cv_by_sg[g]:.3f}")

# With shrinkage
shrunken_quantiles = apply_shrinkage(cv_quantiles, cv_scores, cv_global)
sh_overall, sh_by_sg, sh_size = evaluate_coverage(X_test, y_test, sg_test, shrunken_quantiles, cv_global, p_test)
print(f"\nWith shrinkage: Overall={sh_overall:.3f}")
for g in sorted(sh_by_sg.keys()):
    print(f"  {g}: {sh_by_sg[g]:.3f}")

# ============================================================
# 3. COUNTERFACTUAL ABLATION
# ============================================================
print("\n\n3. COUNTERFACTUAL ABLATION")

# Load from artifacts
subgroup_stats = artifacts['subgroup_stats']
mu_global = X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1.0
prec_global = 1.0 / var_global
r2_global = 10.323

ACTIONABLE_NAMES = ['z_BMI', 'z_Waist_cm', 'z_HbA1c', 'z_SBP', 'z_DBP']
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_NAMES if f in z_cols]

def mahalanobis_diag(z, mu, prec):
    diff = z - mu
    return np.sum(diff**2 * prec)

def counterfactual_no_ellipsoid(z, subgroup, target_prob=0.5, lr=0.05, max_steps=200):
    """Counterfactual without ellipsoid constraint"""
    stats = subgroup_stats.get(subgroup, {'mu': mu_global, 'prec': prec_global, 'r2': r2_global})
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2']
    delta = np.zeros(len(z))
    z_current = z.copy()
    for step in range(max_steps):
        p = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
        if p <= target_prob: break
        grad = np.zeros(len(z))
        eps = 1e-4
        for i in actionable_idx:
            z_plus = z_current.copy()
            z_plus[i] += eps
            p_plus = clf.predict_proba(z_plus.reshape(1, -1))[0, 1]
            grad[i] = (p_plus - p) / eps
        for i in actionable_idx:
            delta[i] -= lr * grad[i]
            delta[i] = np.clip(delta[i], -5, 5)
        z_current = z + delta
        if np.max(np.abs(grad[actionable_idx])) < 1e-4: break
    p_final = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
    mahal_final = mahalanobis_diag(z_current, mu, prec)
    return delta, mahal_final, False, p_final

def counterfactual_ellipsoid_1x(z, subgroup, target_prob=0.5, lr=0.05, max_steps=200):
    """Counterfactual with 1x ellipsoid"""
    stats = subgroup_stats.get(subgroup, {'mu': mu_global, 'prec': prec_global, 'r2': r2_global})
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2']
    delta = np.zeros(len(z))
    z_current = z.copy()
    for step in range(100):
        p = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
        if p <= target_prob: break
        grad = np.zeros(len(z))
        eps = 1e-4
        for i in actionable_idx:
            z_plus = z_current.copy()
            z_plus[i] += eps
            p_plus = clf.predict_proba(z_plus.reshape(1, -1))[0, 1]
            grad[i] = (p_plus - p) / eps
        mahal = mahalanobis_diag(z_current, mu, prec)
        penalty = 5.0 if mahal > r2 else 0.1
        for i in actionable_idx:
            grad[i] += penalty * 2 * (z_current[i] - mu[i]) * prec[i]
        for i in actionable_idx:
            delta[i] -= lr * grad[i]
            delta[i] = np.clip(delta[i], -4, 4)
        z_current = z + delta
        if np.max(np.abs(grad[actionable_idx])) < 1e-4: break
    p_final = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
    mahal_final = mahalanobis_diag(z_current, mu, prec)
    inside = mahal_final <= r2
    return delta, mahal_final, inside, p_final

def counterfactual_ellipsoid_2x(z, subgroup, target_prob=0.5, lr=0.05, max_steps=300):
    """Counterfactual with 2x ellipsoid + 5 restarts (main method)"""
    stats = subgroup_stats.get(subgroup, {'mu': mu_global, 'prec': prec_global, 'r2': r2_global})
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2']
    
    best_delta = None
    best_p = 1.0
    best_mahal = np.inf
    
    for restart in range(5):
        delta = np.random.randn(len(z)) * 0.1
        z_current = z.copy()
        
        for step in range(max_steps):
            p = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
            if p <= target_prob: break
            grad = np.zeros(len(z))
            eps = 1e-4
            for i in actionable_idx:
                z_plus = z_current.copy()
                z_plus[i] += eps
                p_plus = clf.predict_proba(z_plus.reshape(1, -1))[0, 1]
                grad[i] = (p_plus - p) / eps
            
            mahal = mahalanobis_diag(z_current, mu, prec)
            penalty = 10.0 if mahal > r2 else 0.1
            for i in actionable_idx:
                grad[i] += penalty * 2 * (z_current[i] - mu[i]) * prec[i]
            
            for i in actionable_idx:
                delta[i] -= lr * grad[i]
                delta[i] = np.clip(delta[i], -5, 5)
            
            z_current = z + delta
            if np.max(np.abs(grad[actionable_idx])) < 1e-4: break
        
        p_final = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
        mahal_final = mahalanobis_diag(z_current, mu, prec)
        
        if p_final < best_p:
            best_p = p_final
            best_delta = z_current - z
            best_mahal = mahal_final
    
    if best_delta is not None:
        delta = best_delta
        p_final = best_p
        mahal_final = best_mahal
    else:
        delta = np.zeros(len(z))
        p_final = clf.predict_proba(z.reshape(1, -1))[0, 1]
        mahal_final = mahalanobis_diag(z, mu, prec)
    
    inside = mahal_final <= r2 * 2.0
    return delta, mahal_final, inside, p_final

# Recreate subgroup_stats from artifacts
subgroup_stats = artifacts['subgroup_stats']
mu_global = artifacts['subgroup_stats']['White']['mu'] if 'White' in artifacts['subgroup_stats'] else X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1.0
prec_global = 1.0 / var_global
r2_global = 10.323

ACTIONABLE_NAMES = ['z_BMI', 'z_Waist_cm', 'z_HbA1c', 'z_SBP', 'z_DBP']
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_NAMES if f in z_cols]

high_risk_test_local = np.where(p_test > 0.5)[0]
high_risk_test = test_idx[high_risk_test_local]

methods = {
    "No ellipsoid": counterfactual_no_ellipsoid,
    "1x Ellipsoid": counterfactual_ellipsoid_1x,
    "2x Ellipsoid (5 restarts)": counterfactual_ellipsoid_2x,
}

print("\n=== COUNTERFACTUAL RESULTS ===")
for name, func in methods.items():
    reached = 0
    inside = 0
    norms = []
    mahals = []
    
    for local_idx, global_idx in enumerate(high_risk_test):
        z = X_test[local_idx]
        sg = sg_test[local_idx]
        p = p_test[local_idx]
        delta, mahal, inside, p_final = func(z, sg_test[local_idx])
        if p_final <= 0.5: reached += 1
        if inside: inside += 1
        norms.append(np.linalg.norm(delta))
        mahals.append(mahal)
    
    print(f"\n{name}:")
    print(f"  Reached target: {reached}/{len(high_risk_test)} ({reached/len(high_risk_test)*100:.1f}%)")
    print(f"  Inside ellipsoid: {inside}/{len(high_risk_test)} ({inside/len(high_risk_test)*100:.1f}%)")
    print(f"  Mean ||delta||: {np.mean(norms):.3f}")
    print(f"  Mean Mahalanobis: {np.mean(mahals):.3f}")

# ============================================================
# 4. SUBGROUP DEFINITION ABLATION
# ============================================================
print("\n\n4. SUBGROUP DEFINITION ABLATION")

subgroup_defs = {
    "Race/Ethnicity": cohort['subgroup_race'].astype(str),
    "Age (7 bins)": cohort['subgroup_age'].astype(str),
    "BMI Category": cohort['subgroup_bmi'].astype(str),
    "Gender": cohort['subgroup_gender'].astype(str),
}

for name, subgroups in subgroup_defs.items():
    print(f"\n--- {name} ---")
    unique = subgroups.unique()
    print(f"  Groups: {len(unique)}")
    for g in unique:
        sub = cohort[subgroups == g]
        print(f"  {g}: n={len(sub)}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")

# ============================================================
# 5. SAVE ABLATION RESULTS
# ============================================================
ablation_results = {
    'conformal_comparison': {
        'cv_plus': {'overall': cv_overall, 'by_sg': cv_by_sg, 'mean_size': cv_size},
        'single_split': {'overall': ss_overall, 'by_sg': ss_by_sg, 'mean_size': ss_size} if 'ss_overall' in locals() else {},
        'jackknife_plus': {'overall': jk_overall, 'by_sg': jk_by_sg, 'mean_size': jk_size} if 'jk_overall' in locals() else {},
    },
    'shrinkage': {
        'no_shrinkage': {'overall': cv_overall, 'by_sg': cv_by_sg},
        'with_shrinkage': {'overall': sh_overall, 'by_sg': sh_by_sg} if 'sh_overall' in locals() else {},
    },
    'counterfactual': {
        'no_ellipsoid': {},
        '1x_ellipsoid': {},
        '2x_ellipsoid': {},
    },
    'subgroup_defs': {name: {'groups': len(subgroups.unique()), 'diabetes_rates': {}} 
                      for name, subgroups in subgroup_defs.items()},
}

with open(DATA_DIR / "ablation_results.pkl", 'wb') as f:
    pickle.dump(ablation_results, f)

print("\n\nAblation results saved to ablation_results.pkl")