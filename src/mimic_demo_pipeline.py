"""
MIMIC-IV Demo Pipeline - Comparison with NHANES
================================================
Runs the full framework on MIMIC-IV demo (92 patients) for comparison.
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

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")

print("=" * 60)
print("MIMIC-IV DEMO PIPELINE (Comparison)")
print("=" * 60)

# ============================================================
# 1. LOAD COHORT (from demo data)
# ============================================================
print("\n1. LOADING MIMIC-IV DEMO COHORT")
cohort = pd.read_csv(DATA_DIR / "cohort_incident_t2d.csv")
print(f"   Patients: {len(cohort)}")
print(f"   Incident T2D: {cohort['incident_t2d'].sum()} ({cohort['incident_t2d'].mean()*100:.1f}%)")

# Subgroup: age bands
cohort['subgroup'] = cohort['age_band'].astype(str)
unique_subgroups = cohort['subgroup'].unique()
print(f"   Subgroups (age): {len(unique_subgroups)}")

# ============================================================
# 2. LOAD EMBEDDINGS
# ============================================================
print("\n2. LOADING EMBEDDINGS")
embeddings = pd.read_csv(DATA_DIR / "embeddings.csv")
z_cols = [c for c in embeddings.columns if c.startswith('z_')]
print(f"   Embedding dimensions: {len(z_cols)}")

# Merge - embeddings has subgroup_* columns
data = cohort[['subject_id', 'incident_t2d', 'age_band']].merge(embeddings, on='subject_id')
X = data[z_cols].values
y = data['incident_t2d'].values
# Use age_band from cohort
subgroups_arr = data['age_band'].values

print(f"   X shape: {X.shape}, y shape: {y.shape}")

# ============================================================
# 3. SPLIT DATA
# ============================================================
print("\n3. DATA SPLIT")
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

X_train, y_train = X[train_idx], y[train_idx]
X_cal, y_cal = X[temp_idx[cal_idx]], y[temp_idx[cal_idx]]
X_test, y_test = X[temp_idx[test_idx]], y[temp_idx[test_idx]]
sg_train = subgroups_arr[train_idx]
sg_cal = subgroups_arr[temp_idx[cal_idx]]
sg_test = subgroups_arr[temp_idx[test_idx]]

print(f"   Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

# ============================================================
# 4. BASE PREDICTOR
# ============================================================
print("\n4. BASE PREDICTOR")
from sklearn.preprocessing import StandardScaler
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
clf.fit(X_train, y_train)

p_train = clf.predict_proba(X_train)[:, 1]
p_cal = clf.predict_proba(X_cal)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

print(f"   Train AUC: {roc_auc_score(y_train, p_train):.3f}")
print(f"   Cal AUC: {roc_auc_score(y_cal, p_cal):.3f}")
print(f"   Test AUC: {roc_auc_score(y_test, p_test):.3f}")

# ============================================================
# 5. CV+ CONFORMAL CALIBRATION
# ============================================================
print("\n5. CV+ CONFORMAL CALIBRATION")
from sklearn.model_selection import StratifiedKFold

alpha = 0.1
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

X_all = X
y_all = y
subgroups_all = subgroups_arr
unique_subgroups = np.unique(subgroups_all)

n = len(y_all)
p_cv = np.zeros(n)
fold_cal_scores = {g: [] for g in unique_subgroups}
fold_models = []

for fold_idx, (train_idx, cal_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"   Fold {fold_idx+1}/{n_folds}: train={len(train_idx)}, cal={len(cal_idx)}")
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_cl, y_cl = X_all[cal_idx], y_all[cal_idx]
    sg_cl = subgroups_all[cal_idx]
    
    clf_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=fold_idx))
    ])
    clf_cv.fit(X_tr, y_tr)
    fold_models.append(clf_cv)
    
    p_cl = clf_cv.predict_proba(X_cl)[:, 1]
    p_cv[cal_idx] = p_cl
    A_cl = np.where(y_cl == 1, 1 - p_cl, p_cl)
    
    for g in unique_subgroups:
        mask = (sg_cl == g)
        if mask.sum() > 0:
            fold_cal_scores[g].extend(A_cl[mask])

# Quantiles
subgroup_quantiles = {}
for g in unique_subgroups:
    scores = np.array(fold_cal_scores[g])
    if len(scores) > 0:
        q = np.quantile(scores, 1 - alpha, method='higher')
    else:
        all_s = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
        q = np.quantile(all_s, 1 - alpha, method='higher')
    subgroup_quantiles[g] = q
    print(f"     {g}: n_cal={len(fold_cal_scores[g])}, q={q:.4f}")

all_scores = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
global_q = np.quantile(all_scores, 1 - alpha, method='higher')
print(f"   Global: q={global_q:.4f}")

# Hierarchical shrinkage
valid_groups = [g for g in unique_subgroups if len(fold_cal_scores[g]) > 0]
if len(valid_groups) > 1:
    q_vals = np.array([subgroup_quantiles[g] for g in valid_groups])
    n_vals = np.array([len(fold_cal_scores[g]) for g in valid_groups])
    within_vars = np.array([np.var(fold_cal_scores[g]) for g in valid_groups])
    quantile_vars = within_vars / n_vals
    weighted_mean = np.average(q_vals, weights=n_vals)
    total_var = np.average((q_vals - weighted_mean)**2, weights=n_vals)
    tau2 = max(0, total_var - np.mean(quantile_vars))
    kappa = 1.0 / tau2 if tau2 > 0 else 1e6
    
    shrunken = {}
    for g in unique_subgroups:
        n_g = len(fold_cal_scores[g])
        q_g = subgroup_quantiles[g]
        q_shrunk = (n_g * q_g + kappa * weighted_mean) / (n_g + kappa) if n_g > 0 else weighted_mean
        shrunken[g] = q_shrunk
    final_quantiles = shrunken
    print(f"   Hierarchical shrinkage: tau^2={tau2:.6f}, kappa={kappa:.2f}")
else:
    final_quantiles = subgroup_quantiles

# ============================================================
# 6. COVERAGE EVALUATION
# ============================================================
print("\n6. COVERAGE EVALUATION")
A_test = np.where(y_test == 1, 1 - p_test, p_test)
covered = np.zeros(len(y_test), dtype=bool)
pred_sets = np.zeros((len(y_test), 2), dtype=int)

for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_sets[i, 1] = 1 if (1 - p_test[i]) <= q else 0
    pred_sets[i, 0] = 1 if p_test[i] <= q else 0
    covered[i] = y_test[i] in (np.where(pred_sets[i])[0])

print(f"   Overall: {covered.mean():.3f} (target: {1-alpha:.3f})")
for g in sorted(np.unique(sg_test)):
    mask = (sg_test == g)
    if mask.sum() > 0:
        print(f"   {g}: {covered[mask].mean():.3f} (n={mask.sum()})")

set_sizes = pred_sets.sum(axis=1)
print(f"   Set sizes: mean={set_sizes.mean():.2f}, dist={pd.Series(set_sizes).value_counts().to_dict()}")

# ============================================================
# 7. COUNTERFACTUAL GEOMETRY
# ============================================================
print("\n7. COUNTERFACTUAL GEOMETRY")

# Mahalanobis
def mahalanobis_diag(z, mu, prec):
    diff = z - mu
    return np.sum(diff**2 * prec)

# Subgroup stats with 95th percentile Mahalanobis
subgroup_stats = {}
for g in unique_subgroups:
    mask = (subgroups_arr == g)
    X_g = X[mask]
    if len(X_g) < 3: continue
    mu_g = X_g.mean(axis=0)
    var_g = np.var(X_g, axis=0, ddof=1) + 1.0
    prec_g = 1.0 / var_g
    mahal_dists = np.array([np.sum((x - mu_g)**2 * prec_g) for x in X_g])
    r2_g = np.percentile(mahal_dists, 95)
    subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}
    print(f"     {g}: n={len(X_g)}, r2_95={r2_g:.3f}")

mu_global = X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1.0
prec_global = 1.0 / var_global
mahal_global = np.array([np.sum((x - mu_global)**2 * prec_global) for x in X])
r2_global = np.percentile(mahal_global, 95)
print(f"     Global: r2_95={r2_global:.3f}")

def mahalanobis_diag(z, mu, prec):
    diff = z - mu
    return np.sum(diff**2 * prec)

# Actionable features
ACTIONABLE_NAMES = ['z_vital_bmi_kg_m2', 'z_vital_weight_lbs', 'z_lab_hba1c', 'z_lab_glucose_chem', 'z_lab_glucose_blood_gas']
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_NAMES if f in z_cols]
print(f"   Actionable features: {[z_cols[i] for i in actionable_idx]}")

# Counterfactual with 2x ellipsoid + 5 restarts
def counterfactual_gradient(z, subgroup, target_prob=0.5, lr=0.05, max_steps=300):
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
            
            mahal = np.sum((z_current - mu)**2 * prec)
            penalty = 10.0 if mahal > r2 else 0.1
            for i in actionable_idx:
                grad[i] += penalty * 2 * (z_current[i] - mu[i]) * prec[i]
            
            for i in actionable_idx:
                delta[i] -= lr * grad[i]
                delta[i] = np.clip(delta[i], -5, 5)
            
            z_current = z + delta
            if np.max(np.abs(grad[actionable_idx])) < 1e-4: break
        
        p_final = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
        mahal_final = np.sum((z_current - mu)**2 * prec)
        
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
        mahal_final = np.sum((z - mu)**2 * prec)
    
    inside = mahal_final <= r2 * 2.0
    return delta, mahal_final, inside, p_final

# High-risk test patients
p_test = clf.predict_proba(X_test)[:, 1]
high_risk_test_local = np.where(p_test > 0.5)[0]
high_risk_test = test_idx[high_risk_test_local]
print(f"   High-risk test patients: {len(high_risk_test)}")

cf_results = []
for local_idx, global_idx in enumerate(high_risk_test):
    z = X_test[local_idx]
    sg = sg_test[local_idx]
    p = p_test[local_idx]
    delta, mahal, inside, p_final = counterfactual_gradient(z, sg, target_prob=0.5)
    changes = {z_cols[i]: delta[i] for i in actionable_idx if abs(delta[i]) > 1e-3}
    cf_results.append({
        'subject_id': int(cohort.iloc[global_idx]['subject_id']),
        'subgroup': sg, 'p_initial': p, 'p_final': p_final,
        'delta_norm': np.linalg.norm(delta), 'mahalanobis': mahal,
        'inside_ellipsoid': inside, 'actionable_changes': changes
    })
    print(f"     Patient {cohort.iloc[global_idx]['subject_id']}: p={p:.3f}->{p_final:.3f}, ||delta||={np.linalg.norm(delta):.3f}, Mahal={mahal:.3f}, inside={inside}")

if cf_results:
    inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
    success_count = sum(1 for r in cf_results if r['p_final'] <= 0.5)
    print(f"   Summary: {len(cf_results)} patients, {success_count} reached target, {inside_count} inside ellipsoid")

# ============================================================
# 8. SAVE MIMIC RESULTS
# ============================================================
print("\n8. SAVING RESULTS")

mimic_artifacts = {
    'cohort': cohort,
    'embeddings': embeddings,
    'z_cols': z_cols,
    'clf': clf,
    'final_quantiles': final_quantiles,
    'global_q': global_q,
    'alpha': alpha,
    'subgroup_stats': subgroup_stats,
    'cf_results': cf_results,
    'subgroup_col': 'age_band'
}

with open(DATA_DIR / "mimic_demo_artifacts.pkl", 'wb') as f:
    pickle.dump(mimic_artifacts, f)

print("   MIMIC demo artifacts saved")

print("\n" + "=" * 60)
print("MIMIC-IV DEMO PIPELINE COMPLETE")
print("=" * 60)