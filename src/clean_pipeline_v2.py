"""
Clean Pipeline v2: Fixed Calibration, Analytical Counterfactuals, Proper Attribution
=====================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import norm
from scipy.optimize import minimize
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data")
RESULTS_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\reproducibility_package\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.1
N_FOLDS = 5

# ============================================================
# 1. LOAD & PREPARE COHORT
# ============================================================
print("=" * 70)
print("CLEAN PIPELINE v2: Subgroup-Calibrated Conformal Diabetes Risk Explanation")
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
# 4. DATA SPLITS
# ============================================================
print("\n4. DATA SPLITS")
X = embeddings[z_cols].values
y = embeddings['y'].values
subgroups_arr = embeddings['subgroup'].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
train_idx, temp_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

X_train, y_train = X[train_idx], y[train_idx]
X_cal, y_cal = X[temp_idx[cal_idx]], y[temp_idx[cal_idx]]
X_test, y_test = X[temp_idx[test_idx]], y[temp_idx[test_idx]]
sg_train = subgroups_arr[train_idx]
sg_cal = subgroups_arr[temp_idx[cal_idx]]
sg_test = subgroups_arr[temp_idx[test_idx]]

print(f"   Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

# ============================================================
# 5. BASE PREDICTOR + CALIBRATION (Platt scaling on calibration set)
# ============================================================
print("\n5. BASE PREDICTOR + CALIBRATION")

# Train base model on train set
base_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED, solver='lbfgs'))
])
base_clf.fit(X_train, y_train)

# Base model probabilities (for counterfactuals)
p_train_base = base_clf.predict_proba(X_train)[:, 1]
p_cal_base = base_clf.predict_proba(X_cal)[:, 1]
p_test_base = base_clf.predict_proba(X_test)[:, 1]

# Calibrate on calibration set
from sklearn.calibration import CalibratedClassifierCV
cal_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=2)
cal_clf.fit(X_cal, y_cal)

# Calibrated probabilities (for conformal calibration and predictions)
p_train = cal_clf.predict_proba(X_train)[:, 1]
p_cal = cal_clf.predict_proba(X_cal)[:, 1]
p_test = cal_clf.predict_proba(X_test)[:, 1]

print(f"   Train AUC (base): {roc_auc_score(y_train, p_train_base):.3f}, (cal): {roc_auc_score(y_train, p_train):.3f}")
print(f"   Cal AUC (base): {roc_auc_score(y_cal, p_cal_base):.3f}, (cal): {roc_auc_score(y_cal, p_cal):.3f}")
print(f"   Test AUC (base): {roc_auc_score(y_test, p_test_base):.3f}, (cal): {roc_auc_score(y_test, p_test):.3f}")

# ============================================================
# 6. CONFORMAL CALIBRATION (CV+)
# ============================================================
print("\n6. CONFORMAL CALIBRATION (CV+)")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
n = len(y)
p_cv = np.zeros(n)
fold_cal_scores = {g: [] for g in unique_subgroups}

for fold_idx, (train_idx_cv, cal_idx_cv) in enumerate(skf.split(X, y)):
    print(f"   Fold {fold_idx+1}/{N_FOLDS}: train={len(train_idx_cv)}, cal={len(cal_idx_cv)}")
    X_tr, y_tr = X[train_idx_cv], y[train_idx_cv]
    X_cl, y_cl = X[cal_idx_cv], y[cal_idx_cv]
    sg_cl = subgroups_arr[cal_idx_cv]
    
    clf_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=fold_idx, solver='lbfgs'))
    ])
    clf_cv.fit(X_tr, y_tr)
    
    # Calibrate this fold's model
    cal_clf_cv = CalibratedClassifierCV(clf_cv, method='sigmoid', cv=2)
    cal_clf_cv.fit(X_cl, y_cl)
    
    p_cl = cal_clf_cv.predict_proba(X_cl)[:, 1]
    p_cv[cal_idx_cv] = p_cl
    A_cl = np.where(y_cl == 1, 1 - p_cl, p_cl)
    
    for g in unique_subgroups:
        mask = (sg_cl == g)
        if mask.sum() > 0:
            fold_cal_scores[g].extend(A_cl[mask])

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
# 7. HIERARCHICAL SHRINKAGE (Empirical Bayes: sigma2/tau2)
# ============================================================
print("\n7. HIERARCHICAL SHRINKAGE (Empirical Bayes)")

valid_groups = [g for g in unique_subgroups if len(fold_cal_scores[g]) > 0]
if len(valid_groups) > 1:
    q_vals = np.array([subgroup_quantiles[g] for g in valid_groups])
    n_vals = np.array([len(fold_cal_scores[g]) for g in valid_groups])
    
    # Within-group variance of quantile estimator
    within_vars = np.array([np.var(fold_cal_scores[g], ddof=1) / max(len(fold_cal_scores[g]), 1) for g in valid_groups])
    
    weighted_mean = np.average(q_vals, weights=n_vals)
    total_var = np.average((q_vals - weighted_mean)**2, weights=n_vals)
    tau2 = max(0, total_var - np.mean(within_vars))
    
    sigma2 = np.mean(within_vars)
    if tau2 > 0:
        kappa = sigma2 / tau2
    else:
        kappa = 1e6
    
    print(f"   sigma2 (within): {sigma2:.6f}")
    print(f"   tau2 (between): {tau2:.6f}")
    print(f"   kappa = sigma2/tau2: {kappa:.2f}")
    
    shrunken = {}
    for g in unique_subgroups:
        n_g = len(fold_cal_scores[g])
        q_g = subgroup_quantiles[g]
        if n_g > 0:
            q_shrunk = (n_g * q_g + kappa * weighted_mean) / (n_g + kappa)
        else:
            q_shrunk = weighted_mean
        shrunken[g] = q_shrunk
        print(f"     {g}: q_raw={q_g:.4f} -> q_shrunk={q_shrunk:.4f}")
    final_quantiles = shrunken
else:
    final_quantiles = subgroup_quantiles
    print("   Single group - no shrinkage applied")

# ============================================================
# 8. COVERAGE EVALUATION
# ============================================================
print("\n8. COVERAGE EVALUATION")

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

# ============================================================
# 9. ANALYTICAL COUNTERFACTUAL FOR BASE LOGISTIC REGRESSION
# ============================================================
print("\n9. ANALYTICAL COUNTERFACTUAL OPTIMIZATION (Base LR)")

# Extract base logistic regression weights (uncalibrated, used for analytical counterfactuals)
base_lr = base_clf.named_steps['lr']
scaler = base_clf.named_steps['scaler']

w_scaled = base_lr.coef_[0]
b_scaled = base_lr.intercept_[0]
scale = scaler.scale_
mean = scaler.mean_

w = w_scaled / scale
b = b_scaled - np.dot(w_scaled, mean / scale)

print(f"   Base LR: w_norm={np.linalg.norm(w):.3f}, b={b:.3f}")

# Verify base model probabilities
logits_test = X_test @ w + b
p_test_base_verify = 1 / (1 + np.exp(-logits_test))
print(f"   Base prob verification: max_diff={np.max(np.abs(p_test_base - p_test_base_verify)):.6f}")

# Subgroup statistics for ellipsoid (using full data distribution)
subgroup_stats = {}
for g in unique_subgroups:
    mask = (subgroups_arr == g)
    X_g = X[mask]
    if len(X_g) < 3: continue
    mu_g = X_g.mean(axis=0)
    var_g = np.var(X_g, axis=0, ddof=1) + 1e-6
    prec_g = 1.0 / var_g
    diff = X_g - mu_g
    mahal_dists = np.sum(diff**2 * prec_g, axis=1)
    r2_g = np.percentile(mahal_dists, 90)
    subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}
    print(f"     {g}: n={len(X_g)}, r2_90={r2_g:.3f}")

mu_global = X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1e-6
prec_global = 1.0 / var_global
diff_global = X - mu_global
mahal_global = np.sum(diff_global**2 * prec_global, axis=1)
r2_global = np.percentile(mahal_global, 90)
print(f"     Global: r2_90={r2_global:.3f}")

ACTIONABLE_NAMES = ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP', 'z_INDFMPIR',
                    'z_Current_Smoker', 'z_Former_Smoker', 'z_Current_Drinker',
                    'z_Vigorous_Activity', 'z_Moderate_Activity']
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_NAMES if f in z_cols]
non_actionable_idx = [i for i in range(len(z_cols)) if i not in actionable_idx]
print(f"   Actionable: {[z_cols[i] for i in actionable_idx]}")
print(f"   Frozen: {[z_cols[i] for i in non_actionable_idx]}")

# Analytical solution for logistic regression counterfactual
# min ||delta||_2 s.t. w^T delta <= c, delta_A^c = 0, (z+delta-mu)^T Sigma^-1 (z+delta-mu) <= r^2
# where c = logit(tau) - (w^T z + b)

def solve_counterfactual_analytical(z, subgroup, target_prob=0.5):
    stats = subgroup_stats.get(subgroup, {'mu': mu_global, 'prec': prec_global, 'r2': r2_global})
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2']
    r = np.sqrt(r2)
    
    # Linear constraint: w^T delta <= c (using BASE model logits)
    c = np.log(target_prob / (1 - target_prob)) - (np.dot(w, z) + b)
    
    # Only actionable dimensions
    w_act = w[actionable_idx]
    z_act = z[actionable_idx]
    mu_act = mu[actionable_idx]
    prec_act = prec[actionable_idx]
    n_act = len(actionable_idx)
    
    # If already satisfies risk constraint (c >= 0), no action needed
    if c >= 0:
        delta = np.zeros(len(z))
        p_final = 1 / (1 + np.exp(-(np.dot(w, z) + b)))
        mahal = np.sum((z - mu)**2 * prec)
        return delta, mahal, mahal <= r2, p_final
    
    # Project w onto actionable subspace
    w_act_norm = np.linalg.norm(w_act)
    if w_act_norm < 1e-10:
        return np.zeros(len(z)), np.inf, False, 1.0
    
    # Unconstrained minimum norm solution: delta = - (c / ||w_act||^2) * w_act
    delta_act_unconstrained = - (c / (w_act_norm**2)) * w_act
    
    # Check ellipsoid constraint
    z_new_act = z_act + delta_act_unconstrained
    z_new = z.copy()
    z_new[actionable_idx] = z_new_act
    mahal = np.sum((z_new - mu)**2 * prec)
    
    if mahal <= r2:
        # Unconstrained solution is inside ellipsoid - optimal!
        delta = np.zeros(len(z))
        delta[actionable_idx] = delta_act_unconstrained
        p_final = target_prob  # By construction
        return delta, mahal, True, p_final
    
    # Constrained: minimize ||delta||^2 on intersection of half-space and ellipsoid
    # Use SLSQP with good initialization from unconstrained solution
    def obj(delta_act):
        return np.sum(delta_act**2)
    
    def constraint_risk(delta_act):
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        logit_new = np.dot(w, z + full_delta) + b
        p_new = 1 / (1 + np.exp(-logit_new))
        return target_prob - p_new
    
    def constraint_ellipsoid(delta_act):
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        z_new = z + full_delta
        mahal = np.sum((z_new - mu)**2 * prec)
        return r2 - mahal
    
    bounds = [(-5.0, 5.0)] * n_act
    constraints = [
        {'type': 'ineq', 'fun': constraint_risk},
        {'type': 'ineq', 'fun': constraint_ellipsoid}
    ]
    
    best_delta = None
    best_obj = np.inf
    best_p = 1.0
    best_mahal = np.inf
    
    for restart in [0] + list(range(5)):
        if restart == 0:
            x0 = delta_act_unconstrained
        else:
            x0 = np.random.randn(n_act) * 0.1
        
        try:
            result = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            if result.success:
                full_delta = np.zeros(len(z))
                full_delta[actionable_idx] = result.x
                p_final = 1 / (1 + np.exp(-(np.dot(w, z + full_delta) + b)))
                mahal_final = np.sum((z + full_delta - mu)**2 * prec)
                obj_val = np.sum(result.x**2)
                if p_final <= target_prob and mahal_final <= r2 and obj_val < best_obj:
                    best_obj = obj_val
                    best_delta = full_delta
                    best_p = p_final
                    best_mahal = mahal_final
        except:
            continue
    
    if best_delta is not None:
        delta = best_delta
        p_final = best_p
        mahal_final = best_mahal
    else:
        delta = np.zeros(len(z))
        p_final = 1 / (1 + np.exp(-(np.dot(w, z) + b)))
        mahal_final = np.sum((z - mu)**2 * prec)
    
    inside = mahal_final <= r2
    return delta, mahal_final, inside, p_final

# High-risk test patients (using CALIBRATED probabilities for selection)
high_risk_idx = np.where(p_test > 0.5)[0]
print(f"   High-risk test patients (cal p > 0.5): {len(high_risk_idx)}")

cf_results = []
for local_idx in high_risk_idx:
    z = X_test[local_idx]
    sg = sg_test[local_idx]
    p = p_test[local_idx]
    p_base = p_test_base[local_idx]
    delta, mahal, inside, p_final_base = solve_counterfactual_analytical(z, sg, target_prob=0.5)
    
    # Evaluate counterfactual on CALIBRATED model
    z_new = z + delta
    p_final_cal = cal_clf.predict_proba(z_new.reshape(1, -1))[0, 1]
    
    changes = {z_cols[i]: delta[i] for i in actionable_idx if abs(delta[i]) > 1e-3}
    cf_results.append({
        'subject_id': int(embeddings.iloc[test_idx[local_idx]]['subject_id']),
        'subgroup': sg, 'p_initial_cal': p, 'p_initial_base': p_base,
        'p_final_base': p_final_base, 'p_final_cal': p_final_cal,
        'delta_norm': np.linalg.norm(delta), 'mahalanobis': mahal,
        'inside_ellipsoid': inside, 'actionable_changes': changes
    })

if cf_results:
    inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
    success_base = sum(1 for r in cf_results if r['p_final_base'] <= 0.5)
    success_cal = sum(1 for r in cf_results if r['p_final_cal'] <= 0.5)
    print(f"   Summary: {len(cf_results)} patients")
    print(f"     Base model: {success_base} reached target ({success_base/len(cf_results)*100:.1f}%)")
    print(f"     Cal model: {success_cal} reached target ({success_cal/len(cf_results)*100:.1f}%)")
    print(f"     Inside ellipsoid: {inside_count} ({inside_count/len(cf_results)*100:.1f}%)")
    for r in cf_results[:5]:
        print(f"     ID {r['subject_id']}: p_cal={r['p_initial_cal']:.3f}->{r['p_final_cal']:.3f}, ||delta||={r['delta_norm']:.3f}, Mahal={r['mahalanobis']:.3f}, inside={r['inside_ellipsoid']}")

# ============================================================
# 10. CONFORMAL MARGINAL ATTRIBUTION (Split Conformal - Fixed Mapping)
# ============================================================
print("\n10. CONFORMAL MARGINAL ATTRIBUTION (Split Conformal)")

def marginal_effect(z, feature_idx, clf):
    p_orig = clf.predict_proba(z.reshape(1, -1))[0, 1]
    z_pert = z.copy()
    z_pert[feature_idx] = 0.0
    p_pert = clf.predict_proba(z_pert.reshape(1, -1))[0, 1]
    return p_orig - p_pert

# Split calibration data for attribution
att_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
att_cal_idx, att_test_idx = next(att_sss.split(X_cal, y_cal))

X_att_cal, y_att_cal = X_cal[att_cal_idx], y_cal[att_cal_idx]
sg_att_cal = sg_cal[att_cal_idx]
X_att_test, y_att_test = X_cal[att_test_idx], y_cal[att_test_idx]
sg_att_test = sg_cal[att_test_idx]

# Calibration marginal effects (using CALIBRATED model)
marginal_cal = {g: {j: [] for j in range(len(z_cols))} for g in unique_subgroups}
for i in range(len(X_att_cal)):
    z = X_att_cal[i]
    g = sg_att_cal[i]
    for j in range(len(z_cols)):
        eff = marginal_effect(z, j, cal_clf)
        marginal_cal[g][j].append(eff)

# Test marginal effects + conformal p-values (with proper indexing)
significance_results = {g: {} for g in unique_subgroups}
for g in unique_subgroups:
    for j in range(len(z_cols)):
        feat_name = z_cols[j]
        cal_scores = np.array(marginal_cal[g][j])
        if len(cal_scores) < 10:
            continue
        test_scores = []
        test_indices = []
        for i in range(len(X_att_test)):
            if sg_att_test[i] == g:
                test_scores.append(marginal_effect(X_att_test[i], j, cal_clf))
                test_indices.append(att_test_idx[i])
        if len(test_scores) == 0:
            continue
        test_scores = np.array(test_scores)
        abs_cal = np.abs(cal_scores)
        abs_test = np.abs(test_scores)
        p_values = [(np.sum(abs_cal >= ts) + 1) / (len(abs_cal) + 1) for ts in abs_test]
        significance_results[g][feat_name] = {
            'p_values': np.array(p_values),
            'mean_effect': np.mean(test_scores),
            'test_indices': test_indices,
            'n_test': len(test_scores)
        }

# Aggregate
print("   Feature significance (fraction p<0.1):")
global_sig = {}
for j in range(len(z_cols)):
    feat_name = z_cols[j]
    all_p = []
    all_eff = []
    for g in unique_subgroups:
        if feat_name in significance_results[g]:
            all_p.extend(significance_results[g][feat_name]['p_values'])
            all_eff.extend([significance_results[g][feat_name]['mean_effect']] * len(significance_results[g][feat_name]['p_values']))
    if all_p:
        global_sig[feat_name] = {
            'sig_frac': np.mean(np.array(all_p) < 0.1),
            'mean_p': np.mean(all_p),
            'mean_eff': np.mean(all_eff)
        }

for feat, stats in sorted(global_sig.items(), key=lambda x: x[1]['sig_frac'], reverse=True):
    print(f"   {feat}: sig_frac={stats['sig_frac']:.1%}, mean_p={stats['mean_p']:.3f}, mean_eff={stats['mean_eff']:+.4f}")

# ============================================================
# 11. DEPLOYABLE AUDITOR (Calibration-set statistics only)
# ============================================================
print("\n11. DEPLOYABLE AUDITOR (Calibration-set statistics)")

cal_coverage = {}
for g in unique_subgroups:
    mask = (sg_cal == g)
    if mask.sum() > 0:
        p_g = p_cal[mask]
        y_g = y_cal[mask]
        q = final_quantiles.get(g, global_q)
        pred_set_g = np.zeros((len(y_g), 2), dtype=int)
        pred_set_g[:, 1] = (1 - p_g <= q).astype(int)
        pred_set_g[:, 0] = (p_g <= q).astype(int)
        covered_g = np.array([y_g[i] in np.where(pred_set_g[i])[0] for i in range(len(y_g))])
        cal_coverage[g] = covered_g.mean()
        print(f"   Cal coverage {g}: {cal_coverage[g]:.3f} (n={mask.sum()})")

q_overall = global_q
pred_set_cal = np.zeros((len(y_cal), 2), dtype=int)
pred_set_cal[:, 1] = (1 - p_cal <= q_overall).astype(int)
pred_set_cal[:, 0] = (p_cal <= q_overall).astype(int)
cal_overall_cov = np.mean([y_cal[i] in np.where(pred_set_cal[i])[0] for i in range(len(y_cal))])
print(f"   Cal overall coverage: {cal_overall_cov:.3f}")

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

cal_ece = compute_ece(p_cal, y_cal)
print(f"   Cal ECE: {cal_ece:.4f}")

# REASONABLE abstention thresholds
ABSTAIN_COV_THRESH = 0.80
ABSTAIN_ECE_THRESH = 0.25  # More reasonable for logistic regression

abstain_subgroups = [g for g, cov in cal_coverage.items() if cov < ABSTAIN_COV_THRESH]
abstain_ece = cal_ece > ABSTAIN_ECE_THRESH

print(f"   Subgroups to abstain: {abstain_subgroups}")
print(f"   ECE abstain: {abstain_ece} (threshold {ABSTAIN_ECE_THRESH})")

# ============================================================
# 12. BUILD EXPLANATIONS
# ============================================================
print("\n12. BUILD EXPLANATIONS (Template-based)")

evidence_list = []
for i in range(len(X_test)):
    z = X_test[i]
    sg = sg_test[i]
    p = p_test[i]
    y_true = y_test[i]
    subject_id = embeddings.iloc[test_idx[i]]['subject_id']
    
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
            if len(res['p_values']) > 0:
                attributions[feat_name] = {
                    'marginal_effect': float(res['mean_effect']),
                    'p_value': float(res['p_values'][0]),
                    'significant': bool(res['p_values'][0] < 0.1)
                }
    
    sg_cal_cov = cal_coverage.get(sg, 1.0)
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
        'abstain_reason': f"Subgroup cal coverage {sg_cal_cov:.2f} < {ABSTAIN_COV_THRESH}" if sg in abstain_subgroups else ("Cal ECE > threshold" if abstain_ece else None)
    }
    evidence_list.append(ev)

abstain_count = sum(1 for ev in evidence_list if ev['abstain'])
print(f"   Test patients abstained: {abstain_count}/{len(evidence_list)} ({abstain_count/len(evidence_list)*100:.1f}%)")

print("\n   SAMPLE EXPLANATIONS:")
for ev in evidence_list[:3]:
    print(f"\n   Patient {ev['subject_id']} (Subgroup: {ev['subgroup']})")
    print(f"   Predicted risk: {ev['predicted_risk']:.1%}")
    print(f"   Prediction set: {ev['prediction_set']} (target coverage {1-ALPHA:.0%})")
    if ev['counterfactual']:
        cf = ev['counterfactual']
        print(f"   Counterfactual (base LR): {cf['p_initial_base']:.1%} -> {cf['p_final_base']:.1%}")
        print(f"   Counterfactual (calibrated): {cf['p_initial_cal']:.1%} -> {cf['p_final_cal']:.1%}")
        print(f"   ||delta||={cf['delta_norm']:.3f}, Mahal={cf['mahalanobis']:.3f}, inside={cf['inside_ellipsoid']}")
        for feat, change in cf['actionable_changes'].items():
            print(f"     {feat}: {change:+.2f} SD")
    print(f"   Top attributions:")
    sig_atts = [(f, a) for f, a in ev['attributions'].items() if a['significant']]
    for feat, att in sig_atts[:5]:
        print(f"     {feat}: effect={att['marginal_effect']:+.3f}, p={att['p_value']:.3f} *")
    if ev['abstain']:
        print(f"   *** ABSTAINED: {ev['abstain_reason']} ***")

# ============================================================
# 13. SAVE ALL ARTIFACTS
# ============================================================
print("\n13. SAVING ARTIFACTS")

artifacts = {
    'cohort': cohort,
    'embeddings': embeddings,
    'z_cols': z_cols,
    'base_clf': base_clf,
    'cal_clf': cal_clf,
    'base_lr': base_lr,
    'scaler': scaler,
    'w': w, 'b': b,
    'final_quantiles': final_quantiles,
    'global_q': global_q,
    'alpha': ALPHA,
    'subgroup_stats': subgroup_stats,
    'cf_results': cf_results,
    'significance_results': significance_results,
    'global_significance': global_sig,
    'evidence_list': evidence_list,
    'cal_coverage': cal_coverage,
    'cal_ece': cal_ece,
    'abstain_subgroups': abstain_subgroups,
    'abstain_ece': abstain_ece,
    'actionable_idx': actionable_idx,
    'actionable_names': ACTIONABLE_NAMES,
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
            'cal_coverage': float(cal_coverage.get(g, 0)),
            'quantile': float(final_quantiles.get(g, global_q))
        })
pd.DataFrame(cov_rows).to_csv(RESULTS_DIR / "coverage_table.csv", index=False)

att_rows = []
for feat, stats in global_sig.items():
    att_rows.append({
        'feature': feat,
        'sig_fraction': stats['sig_frac'],
        'mean_p': stats['mean_p'],
        'mean_effect': stats['mean_eff']
    })
pd.DataFrame(att_rows).to_csv(RESULTS_DIR / "attribution_table.csv", index=False)

print("   All artifacts saved to results/")
print("\n" + "=" * 70)
print("CLEAN PIPELINE v2 COMPLETE")
print("=" * 70)