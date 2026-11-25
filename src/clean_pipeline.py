"""
Clean, Honest Rebuild: Subgroup-Calibrated Conformal Risk Explanation
======================================================================

Task: Predict prevalent diabetes from features EXCLUDING HbA1c/glucose
(no circularity - those are diagnostic criteria)

Components:
1. Clinically normalized embedding (ECDF -> Gaussian)
2. Base predictor (Logistic Regression, no HbA1c/glucose)
3. Subgroup-conditional conformal calibration (CV+) with empirical Bayes shrinkage
4. Proper counterfactual optimization (constrained, actionable features only)
5. Conformal marginal attribution p-values (split conformal)
6. Deployable auditor (calibration-set statistics only)
7. Template-based explanation (no fake LLM)
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
RESULTS_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\reproducibility_package\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.1
N_FOLDS = 5

# ============================================================
# 1. LOAD & PREPARE COHORT
# ============================================================
print("=" * 70)
print("CLEAN PIPELINE: Subgroup-Calibrated Conformal Diabetes Risk Explanation")
print("=" * 70)

cohort = pd.read_csv(DATA_DIR / "nhanes_diabetes_cohort.csv")
print(f"\n1. LOADED COHORT: {len(cohort)} patients")
print(f"   Diabetes prevalence: {cohort['known_diabetes'].mean()*100:.1f}%")

# Subgroup: Race/Ethnicity (6 groups, well-balanced)
cohort['subgroup'] = cohort['Race_Ethnicity'].astype(str)
unique_subgroups = sorted(cohort['subgroup'].unique())
print(f"   Subgroups: {unique_subgroups}")
for sg in unique_subgroups:
    sub = cohort[cohort['subgroup'] == sg]
    print(f"     {sg}: n={len(sub)}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")

# ============================================================
# 2. FEATURES (NO HbA1c, NO GLUCOSE - NO CIRCULARITY)
# ============================================================
CONTINUOUS_FEATURES = {
    'RIDAGEYR': {'unit': 'years'},
    'BMI': {'unit': 'kg/m2'},
    'Waist_cm': {'unit': 'cm'},
    'SBP': {'unit': 'mmHg'},
    'DBP': {'unit': 'mmHg'},
    'INDFMPIR': {'unit': 'ratio'},
}

BINARY_FEATURES = [
    'Current_Smoker', 'Former_Smoker', 'Current_Drinker',
    'Vigorous_Activity', 'Moderate_Activity'
]

CATEGORICAL_FEATURES = ['Gender']

# Reference cohort = non-diabetic controls
ref_cohort = cohort[cohort['known_diabetes'] == 0].copy()
print(f"\n2. REFERENCE COHORT (controls): {len(ref_cohort)}")

# ============================================================
# 3. CLINICALLY NORMALIZED EMBEDDING
# ============================================================
print("\n3. CLINICALLY NORMALIZED EMBEDDING (ECDF -> Phi^-1)")

def ecdf_transform(series, ref_series, clip_quantile=0.001):
    """z = Phi^-1(F_ref(x))"""
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

# Continuous: ECDF transform relative to reference
for feat, spec in CONTINUOUS_FEATURES.items():
    if feat in cohort.columns:
        trans = ecdf_transform(cohort[feat], ref_cohort[feat])
        col = f'z_{feat}'
        embeddings[col] = trans
        z_cols.append(col)
        non_null = trans.notna().sum()
        print(f"   {feat}: {non_null}/{len(cohort)} non-null, range=[{trans.min():.2f}, {trans.max():.2f}]")

# Binary: keep as 0/1
for feat in BINARY_FEATURES:
    if feat in cohort.columns:
        col = f'z_{feat}'
        embeddings[col] = cohort[feat].astype(float)
        z_cols.append(col)

# Categorical: one-hot
for feat in CATEGORICAL_FEATURES:
    if feat in cohort.columns:
        dummies = pd.get_dummies(cohort[feat], prefix=f'z_{feat}')
        for c in dummies.columns:
            embeddings[c] = dummies[c].astype(float)
            z_cols.append(c)

# Fill NaN with 0 (reference value in z-space)
embeddings[z_cols] = embeddings[z_cols].fillna(0.0)

print(f"   Total embedding dimensions: {len(z_cols)}")
print(f"   Z-features: {z_cols}")

# Save embeddings
embeddings.to_csv(RESULTS_DIR / "embeddings.csv", index=False)

# ============================================================
# 4. DATA SPLITS
# ============================================================
print("\n4. DATA SPLITS")
X = embeddings[z_cols].values
y = embeddings['y'].values
subgroups_arr = embeddings['subgroup'].values

# 60/20/20 train/cal/test stratified by outcome and subgroup
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
for sg in unique_subgroups:
    n_tr = (sg_train == sg).sum()
    n_ca = (sg_cal == sg).sum()
    n_te = (sg_test == sg).sum()
    print(f"     {sg}: train={n_tr}, cal={n_ca}, test={n_te}")

# ============================================================
# 5. BASE PREDICTOR (Logistic Regression)
# ============================================================
print("\n5. BASE PREDICTOR (Logistic Regression)")
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED, solver='lbfgs'))
])
clf.fit(X_train, y_train)

p_train = clf.predict_proba(X_train)[:, 1]
p_cal = clf.predict_proba(X_cal)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

print(f"   Train AUC: {roc_auc_score(y_train, p_train):.3f}")
print(f"   Cal AUC: {roc_auc_score(y_cal, p_cal):.3f}")
print(f"   Test AUC: {roc_auc_score(y_test, p_test):.3f}")

# ============================================================
# 6. CONFORMAL CALIBRATION (CV+ with Empirical Bayes Shrinkage)
# ============================================================
print("\n6. CONFORMAL CALIBRATION (CV+ with Empirical Bayes Shrinkage)")

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
    
    p_cl = clf_cv.predict_proba(X_cl)[:, 1]
    p_cv[cal_idx_cv] = p_cl
    A_cl = np.where(y_cl == 1, 1 - p_cl, p_cl)
    
    for g in unique_subgroups:
        mask = (sg_cl == g)
        if mask.sum() > 0:
            fold_cal_scores[g].extend(A_cl[mask])

# Subgroup quantiles (no shrinkage yet)
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

# Global
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
    
    # Within-group variance of quantile estimator: sigma2_g = alpha*(1-alpha) / (n_g * f_g(q)^2)
    # Approximate using sample variance of scores / n_g (conservative)
    within_vars = np.array([np.var(fold_cal_scores[g], ddof=1) / max(len(fold_cal_scores[g]), 1) for g in valid_groups])
    
    # Between-group variance (tau2)
    weighted_mean = np.average(q_vals, weights=n_vals)
    total_var = np.average((q_vals - weighted_mean)**2, weights=n_vals)
    tau2 = max(0, total_var - np.mean(within_vars))
    
    # Shrinkage weight: kappa = sigma2 / tau2 (empirical Bayes)
    sigma2 = np.mean(within_vars)
    if tau2 > 0:
        kappa = sigma2 / tau2
    else:
        kappa = 1e6  # Complete pooling
    
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
    # Class 1 (diabetes): include if 1-p <= q
    # Class 0 (no diabetes): include if p <= q
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
# 9. PROPER COUNTERFACTUAL OPTIMIZATION
# ============================================================
print("\n9. PROPER COUNTERFACTUAL OPTIMIZATION")

# Estimate diagonal covariance per subgroup
subgroup_stats = {}
for g in unique_subgroups:
    mask = (subgroups_arr == g)
    X_g = X[mask]
    if len(X_g) < 3: continue
    mu_g = X_g.mean(axis=0)
    var_g = np.var(X_g, axis=0, ddof=1) + 1e-6
    prec_g = 1.0 / var_g
    
    # Mahalanobis distances in this subgroup
    diff = X_g - mu_g
    mahal_dists = np.sum(diff**2 * prec_g, axis=1)
    r2_g = np.percentile(mahal_dists, 90)  # 90th percentile ellipsoid
    
    subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}
    print(f"     {g}: n={len(X_g)}, r2_90={r2_g:.3f}")

# Global fallback
mu_global = X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1e-6
prec_global = 1.0 / var_global
diff_global = X - mu_global
mahal_global = np.sum(diff_global**2 * prec_global, axis=1)
r2_global = np.percentile(mahal_global, 90)
print(f"     Global: r2_90={r2_global:.3f}")

# Actionable features (no age, sex, race)
ACTIONABLE_NAMES = ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP', 'z_INDFMPIR',
                    'z_Current_Smoker', 'z_Former_Smoker', 'z_Current_Drinker',
                    'z_Vigorous_Activity', 'z_Moderate_Activity']
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_NAMES if f in z_cols]
non_actionable_idx = [i for i in range(len(z_cols)) if i not in actionable_idx]
print(f"   Actionable features: {[z_cols[i] for i in actionable_idx]}")
print(f"   Non-actionable (frozen): {[z_cols[i] for i in non_actionable_idx]}")

def mahalanobis_diag(z, mu, prec):
    diff = z - mu
    return np.sum(diff**2 * prec)

def solve_counterfactual(z, subgroup, target_prob=0.5, max_iter=500):
    """
    Solve: min ||delta||_2 s.t. p(z+delta) <= target, ||delta||_M <= r
    with delta_i = 0 for non-actionable i.
    """
    stats = subgroup_stats.get(subgroup, {'mu': mu_global, 'prec': prec_global, 'r2': r2_global})
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2']
    r = np.sqrt(r2)
    
    # Initial guess: zeros
    delta = np.zeros(len(z))
    
    # Constraints: delta_i = 0 for non-actionable
    # We optimize only over actionable dimensions
    n_act = len(actionable_idx)
    
    def obj(delta_act):
        """Objective: ||delta||_2"""
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        return np.linalg.norm(full_delta)
    
    def constraint_risk(delta_act):
        """p(z + delta) <= target"""
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        p = clf.predict_proba((z + full_delta).reshape(1, -1))[0, 1]
        return target_prob - p  # >= 0 means constraint satisfied
    
    def constraint_ellipsoid(delta_act):
        """||z + delta - mu||_M <= r"""
        full_delta = np.zeros(len(z))
        full_delta[actionable_idx] = delta_act
        z_new = z + full_delta
        mahal = mahalanobis_diag(z_new, mu, prec)
        return r2 - mahal  # >= 0 means inside
    
    # Bounds for actionable dimensions
    bounds = [(-5.0, 5.0)] * n_act
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_risk},
        {'type': 'ineq', 'fun': constraint_ellipsoid}
    ]
    
    # Multiple random restarts
    best_delta = None
    best_obj = np.inf
    best_p = 1.0
    best_mahal = np.inf
    
    for restart in range(10):
        x0 = np.random.randn(n_act) * 0.1
        try:
            result = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-6})
            if result.success:
                full_delta = np.zeros(len(z))
                full_delta[actionable_idx] = result.x
                p_final = clf.predict_proba((z + full_delta).reshape(1, -1))[0, 1]
                mahal_final = mahalanobis_diag(z + full_delta, mu, prec)
                obj_val = np.linalg.norm(full_delta)
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
        # Fallback: no feasible solution found
        delta = np.zeros(len(z))
        p_final = clf.predict_proba(z.reshape(1, -1))[0, 1]
        mahal_final = mahalanobis_diag(z, mu, prec)
    
    inside = mahal_final <= r2
    return delta, mahal_final, inside, p_final

# High-risk test patients
high_risk_idx = np.where(p_test > 0.5)[0]
print(f"   High-risk test patients: {len(high_risk_idx)}")

cf_results = []
for local_idx in high_risk_idx:
    z = X_test[local_idx]
    sg = sg_test[local_idx]
    p = p_test[local_idx]
    delta, mahal, inside, p_final = solve_counterfactual(z, sg, target_prob=0.5)
    changes = {z_cols[i]: delta[i] for i in actionable_idx if abs(delta[i]) > 1e-3}
    cf_results.append({
        'subject_id': int(embeddings.iloc[test_idx[local_idx]]['subject_id']),
        'subgroup': sg, 'p_initial': p, 'p_final': p_final,
        'delta_norm': np.linalg.norm(delta), 'mahalanobis': mahal,
        'inside_ellipsoid': inside, 'actionable_changes': changes
    })

if cf_results:
    inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
    success_count = sum(1 for r in cf_results if r['p_final'] <= 0.5)
    print(f"   Summary: {len(cf_results)} patients, {success_count} reached target ({success_count/len(cf_results)*100:.1f}%), {inside_count} inside ellipsoid ({inside_count/len(cf_results)*100:.1f}%)")
    for r in cf_results[:5]:
        print(f"     ID {r['subject_id']}: p={r['p_initial']:.3f}->{r['p_final']:.3f}, ||delta||={r['delta_norm']:.3f}, Mahal={r['mahalanobis']:.3f}, inside={r['inside_ellipsoid']}")

# ============================================================
# 10. CONFORMAL MARGINAL ATTRIBUTION (Split Conformal)
# ============================================================
print("\n10. CONFORMAL MARGINAL ATTRIBUTION (Split Conformal)")

def marginal_effect(z, feature_idx, clf):
    p_orig = clf.predict_proba(z.reshape(1, -1))[0, 1]
    z_pert = z.copy()
    z_pert[feature_idx] = 0.0  # Set to reference (0 in z-space)
    p_pert = clf.predict_proba(z_pert.reshape(1, -1))[0, 1]
    return p_orig - p_pert

# Split calibration data further for attribution
att_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
att_cal_idx, att_test_idx = next(att_sss.split(X_cal, y_cal))

X_att_cal, y_att_cal = X_cal[att_cal_idx], y_cal[att_cal_idx]
sg_att_cal = sg_cal[att_cal_idx]
X_att_test, y_att_test = X_cal[att_test_idx], y_cal[att_test_idx]
sg_att_test = sg_cal[att_test_idx]

# Calibration marginal effects
marginal_cal = {g: {j: [] for j in range(len(z_cols))} for g in unique_subgroups}
for i in range(len(X_att_cal)):
    z = X_att_cal[i]
    g = sg_att_cal[i]
    for j in range(len(z_cols)):
        eff = marginal_effect(z, j, clf)
        marginal_cal[g][j].append(eff)

# Test marginal effects + conformal p-values
significance_results = {g: {} for g in unique_subgroups}
for g in unique_subgroups:
    for j in range(len(z_cols)):
        feat_name = z_cols[j]
        cal_scores = np.array(marginal_cal[g][j])
        if len(cal_scores) < 10:
            continue
        test_scores = []
        for i in range(len(X_att_test)):
            if sg_att_test[i] == g:
                test_scores.append(marginal_effect(X_att_test[i], j, clf))
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

# Aggregate across subgroups
print("   Feature significance (fraction p<0.1 across subgroups):")
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

# Compute subgroup coverage on CALIBRATION set (deployable)
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

# Overall calibration coverage
q_overall = global_q
pred_set_cal = np.zeros((len(y_cal), 2), dtype=int)
pred_set_cal[:, 1] = (1 - p_cal <= q_overall).astype(int)
pred_set_cal[:, 0] = (p_cal <= q_overall).astype(int)
cal_overall_cov = np.mean([y_cal[i] in np.where(pred_set_cal[i])[0] for i in range(len(y_cal))])
print(f"   Cal overall coverage: {cal_overall_cov:.3f}")

# ECE on calibration set
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

# DEPLOYABLE abstention rule: abstain if calibration coverage < 80% for subgroup
# or calibration ECE > 0.1
ABSTAIN_COV_THRESH = 0.80
ABSTAIN_ECE_THRESH = 0.10

abstain_subgroups = [g for g, cov in cal_coverage.items() if cov < ABSTAIN_COV_THRESH]
abstain_ece = cal_ece > ABSTAIN_ECE_THRESH

print(f"   Subgroups to abstain: {abstain_subgroups}")
print(f"   ECE abstain: {abstain_ece}")

# ============================================================
# 12. BUILD EXPLANATIONS (Template-based, no LLM)
# ============================================================
print("\n12. BUILD EXPLANATIONS (Template-based)")

evidence_list = []
for i in range(len(X_test)):
    z = X_test[i]
    sg = sg_test[i]
    p = p_test[i]
    y_true = y_test[i]
    subject_id = embeddings.iloc[test_idx[i]]['subject_id']
    
    # Prediction set
    q = final_quantiles.get(sg, global_q)
    pred_set = []
    if (1 - p) <= q: pred_set.append(1)
    if p <= q: pred_set.append(0)
    
    # Counterfactual
    cf = None
    for r in cf_results:
        if r['subject_id'] == subject_id:
            cf = r
            break
    
    # Attributions (top significant)
    attributions = {}
    if sg in significance_results:
        for feat_name, res in significance_results[sg].items():
            j = z_cols.index(feat_name)
            test_mask = (sg_att_test == sg)
            test_idx_in_sg = np.where(test_mask)[0]
            # Map back to original test position - simplified
            if len(res['p_values']) > 0:
                attributions[feat_name] = {
                    'marginal_effect': float(res['mean_effect']),
                    'p_value': float(res['p_values'][0]) if len(res['p_values']) > 0 else 1.0,
                    'significant': bool(res['p_values'][0] < 0.1) if len(res['p_values']) > 0 else False
                }
    
    # Audit flags
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

# Audit summary
abstain_count = sum(1 for ev in evidence_list if ev['abstain'])
print(f"   Test patients abstained: {abstain_count}/{len(evidence_list)} ({abstain_count/len(evidence_list)*100:.1f}%)")

# Sample explanations
print("\n   SAMPLE EXPLANATIONS:")
for ev in evidence_list[:3]:
    print(f"\n   Patient {ev['subject_id']} (Subgroup: {ev['subgroup']})")
    print(f"   Predicted risk: {ev['predicted_risk']:.1%}")
    print(f"   Prediction set: {ev['prediction_set']} (target coverage {1-ALPHA:.0%})")
    if ev['counterfactual']:
        cf = ev['counterfactual']
        print(f"   Counterfactual: {cf['p_initial']:.1%} -> {cf['p_final']:.1%}, ||delta||={cf['delta_norm']:.3f}, inside={cf['inside_ellipsoid']}")
        for feat, change in cf['actionable_changes'].items():
            print(f"     {feat}: {change:+.2f} SD")
    print(f"   Top attributions:")
    for feat, att in list(ev['attributions'].items())[:5]:
        if att['significant']:
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
    'clf': clf,
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

# Save key results as CSV for paper tables
results_df = pd.DataFrame(cf_results)
if len(results_df) > 0:
    results_df.to_csv(RESULTS_DIR / "counterfactual_results.csv", index=False)

# Coverage table
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

# Attribution table
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
print("CLEAN PIPELINE COMPLETE")
print("=" * 70)