"""
Full Framework Pipeline on NHANES Diabetes Cohort
==================================================
Runs all components: Embedding -> Conformal Calibration -> Counterfactual Geometry 
-> Attribution Significance -> Auditor/Translator
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
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

# ============================================================
# 1. LOAD NHANES COHORT
# ============================================================
print("=" * 60)
print("NHANES FRAMEWORK PIPELINE")
print("=" * 60)

cohort = pd.read_csv(DATA_DIR / "nhanes_diabetes_cohort.csv")
print(f"\n1. LOADED COHORT: {len(cohort)} patients")
print(f"   Diabetes prevalence: {cohort['known_diabetes'].mean()*100:.1f}%")

# Subgroup definition - USE RACE/ETHNICITY (better balanced diabetes rates)
cohort['subgroup_age'] = pd.cut(cohort['RIDAGEYR'], 
                                 bins=[18, 30, 40, 50, 60, 70, 80, 100],
                                 labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])
cohort['subgroup_race'] = cohort['Race_Ethnicity']
cohort['subgroup_gender'] = cohort['Gender']
cohort['subgroup_bmi'] = pd.cut(cohort['BMI'],
                                 bins=[0, 18.5, 25, 30, 35, 40, 100],
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III'])

# Primary subgroup for calibration - RACE/ETHNICITY (better balanced)
# Merge small groups
cohort['subgroup'] = cohort['subgroup_race'].astype(str)
# Check diabetes rates by race
for sg in cohort['subgroup'].unique():
    sub = cohort[cohort['subgroup'] == sg]
    print(f"   {sg}: n={len(sub)}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")

unique_subgroups = cohort['subgroup'].unique()
print(f"   Subgroups (race): {len(unique_subgroups)}")

# ============================================================
# 2. CLINICALLY NORMALIZED EMBEDDING (Section 4.1)
# ============================================================
print("\n2. CLINICALLY NORMALIZED EMBEDDING")

# Features for embedding (numeric, clinical)
CONTINUOUS_FEATURES = {
    'RIDAGEYR': {'unit': 'years', 'ref_range': (18, 100)},
    'BMI': {'unit': 'kg/m2', 'ref_range': (10, 60)},
    'Waist_cm': {'unit': 'cm', 'ref_range': (40, 150)},
    'SBP': {'unit': 'mmHg', 'ref_range': (80, 200)},
    'DBP': {'unit': 'mmHg', 'ref_range': (40, 130)},
    'HbA1c': {'unit': '%', 'ref_range': (3.5, 15.0)},
    'INDFMPIR': {'unit': 'ratio', 'ref_range': (0, 10)},
}

# Reference cohort = non-diabetic
ref_cohort = cohort[cohort['known_diabetes'] == 0].copy()
print(f"   Reference cohort (controls): {len(ref_cohort)}")

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

print("   Transforming continuous features...")
embeddings = pd.DataFrame(index=cohort.index)
embeddings['subject_id'] = cohort['SEQN']

for feat, spec in CONTINUOUS_FEATURES.items():
    if feat in cohort.columns:
        trans = ecdf_transform(cohort[feat], ref_cohort[feat])
        embeddings[f'z_{feat}'] = trans
        non_null = trans.notna().sum()
        print(f"     {feat}: {non_null}/{len(cohort)} non-null, range=[{trans.min():.2f}, {trans.max():.2f}]")

# Binary features
for feat in ['Current_Smoker', 'Former_Smoker', 'Current_Drinker', 'Vigorous_Activity', 'Moderate_Activity']:
    if feat in cohort.columns:
        embeddings[f'z_{feat}'] = cohort[feat].astype(float)

# Subgroup indicators
for sg in ['subgroup_age', 'subgroup_race', 'subgroup_gender', 'subgroup_bmi']:
    if sg in cohort.columns:
        embeddings[f'subgroup_{sg}'] = cohort[sg].astype(str)

# Standardize z-features
z_cols = [c for c in embeddings.columns if c.startswith('z_')]
for col in z_cols:
    mean = embeddings[col].mean()
    std = embeddings[col].std()
    if std > 0:
        embeddings[col] = (embeddings[col] - mean) / std
    else:
        embeddings[col] = 0.0

embeddings[z_cols] = embeddings[z_cols].fillna(0.0)

print(f"   Embedding dimensions: {len(z_cols)}")
print(f"   Z-features: {z_cols}")

# Save embeddings
embeddings.to_csv(DATA_DIR / "nhanes_embeddings.csv", index=False)
print("   Embeddings saved")

# ============================================================
# 3. SPLIT DATA
# ============================================================
print("\n3. DATA SPLIT")
X = embeddings[z_cols].values
y = cohort['known_diabetes'].values
subgroups_arr = cohort['subgroup'].values

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
print("\n4. BASE PREDICTOR (Logistic Regression)")
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
# 5. CONFORMAL CALIBRATION (CV+)
# ============================================================
print("\n5. CONFORMAL CALIBRATION (CV+)")

alpha = 0.1
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

n = len(y)
p_cv = np.zeros(n)
fold_cal_scores = {g: [] for g in unique_subgroups}
fold_models = []

for fold_idx, (train_idx_cv, cal_idx_cv) in enumerate(skf.split(X, y)):
    print(f"   Fold {fold_idx+1}/{n_folds}: train={len(train_idx_cv)}, cal={len(cal_idx_cv)}")
    X_tr, y_tr = X[train_idx_cv], y[train_idx_cv]
    X_cl, y_cl = X[cal_idx_cv], y[cal_idx_cv]
    sg_cl = subgroups_arr[cal_idx_cv]
    
    clf_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=fold_idx))
    ])
    clf_cv.fit(X_tr, y_tr)
    fold_models.append(clf_cv)
    
    p_cl = clf_cv.predict_proba(X_cl)[:, 1]
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
        q = np.quantile(scores, 1 - alpha, method='higher')
    else:
        all_scores = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
        q = np.quantile(all_scores, 1 - alpha, method='higher')
    subgroup_quantiles[g] = q
    print(f"     {g}: n_cal={len(fold_cal_scores[g])}, q={q:.4f}")

# Global
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
# 6. EVALUATE COVERAGE
# ============================================================
print("\n6. COVERAGE EVALUATION")
A_test = np.where(y_test == 1, 1 - p_test, p_test)
covered = np.zeros(len(y_test), dtype=bool)
pred_sets = np.zeros((len(y_test), 2), dtype=int)

for i in range(len(y_test)):
    g = sg_test[i]
    q = final_quantiles.get(g, global_q)
    pred_sets[i, 0] = 1 if (p_test[i] <= q) else 0
    pred_sets[i, 1] = 1 if (1 - p_test[i] <= q) else 0
    covered[i] = y_test[i] in np.where(pred_sets[i])[0]

print(f"   Overall: {covered.mean():.3f} (target: {1-alpha:.3f})")
for g in unique_subgroups:
    mask = (sg_test == g)
    if mask.sum() > 0:
        print(f"   {g}: {covered[mask].mean():.3f} (n={mask.sum()})")

set_sizes = pred_sets.sum(axis=1)
def mahalanobis_diag(z, mu, prec):
    diff = z - mu
    return np.sum(diff**2 * prec)

print(f"   Set sizes: mean={set_sizes.mean():.2f}, dist={pd.Series(set_sizes).value_counts().to_dict()}")

# ============================================================
# 7. COUNTERFACTUAL GEOMETRY
# ============================================================
print("\n7. COUNTERFACTUAL GEOMETRY")

# Estimate diagonal covariance per subgroup + compute Mahalanobis radii
subgroup_stats = {}
for g in unique_subgroups:
    mask = (subgroups_arr == g)
    X_g = X[mask]
    if len(X_g) < 3: continue
    mu_g = X_g.mean(axis=0)
    var_g = np.var(X_g, axis=0, ddof=1) + 1.0
    prec_g = 1.0 / var_g
    
    # Compute Mahalanobis distances in this subgroup
    mahal_dists = np.array([mahalanobis_diag(x, mu_g, prec_g) for x in X_g])
    # Use 95th percentile as ellipsoid radius (more permissive than quantile)
    r2_g = np.percentile(mahal_dists, 95)
    
    subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}
    print(f"     {g}: n={len(X_g)}, r2_95={r2_g:.3f}")

# Global fallback
mu_global = X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1.0
prec_global = 1.0 / var_global
mahal_global = np.array([mahalanobis_diag(x, mu_global, prec_global) for x in X])
r2_global = np.percentile(mahal_global, 95)
print(f"     Global: r2_95={r2_global:.3f}")

# Actionable features (indices in embedding)
ACTIONABLE_NAMES = ['z_BMI', 'z_Waist_cm', 'z_HbA1c', 'z_SBP', 'z_DBP']
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_NAMES if f in z_cols]
print(f"   Actionable features: {[z_cols[i] for i in actionable_idx]}")

def counterfactual_gradient(z, subgroup, target_prob=0.5, lr=0.05, max_steps=300):
    stats = subgroup_stats.get(subgroup, {'mu': mu_global, 'prec': prec_global, 'r2': r2_global})
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2']
    
    best_delta = None
    best_p = 1.0
    best_mahal = np.inf
    
    # Multiple random restarts
    for restart in range(5):
        delta = np.random.randn(len(z)) * 0.1
        z_current = z.copy()
        
        for step in range(max_steps):
            p = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
            if p <= target_prob:
                break
            grad = np.zeros(len(z))
            eps = 1e-4
            for i in actionable_idx:
                z_plus = z_current.copy()
                z_plus[i] += eps
                p_plus = clf.predict_proba(z_plus.reshape(1, -1))[0, 1]
                grad[i] = (p_plus - p) / eps
            
            mahal = mahalanobis_diag(z_current, mu, prec)
            # Softer penalty: log barrier
            penalty_weight = 10.0 if mahal > r2 else 0.1
            for i in actionable_idx:
                grad[i] += penalty_weight * 2 * (z_current[i] - mu[i]) * prec[i]
            
            for i in actionable_idx:
                delta[i] -= lr * grad[i]
                delta[i] = np.clip(delta[i], -5, 5)
            
            z_current = z + delta
            if np.max(np.abs(grad[actionable_idx])) < 1e-4:
                break
        
        p_final = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
        mahal_final = mahalanobis_diag(z_current, mu, prec)
        
        # Track best
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
    
    inside = mahal_final <= r2 * 2.0  # 2x radius
    return delta, mahal_final, inside, p_final

# High-risk test patients (p_test is already test predictions)
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
        'subject_id': int(cohort.iloc[global_idx]['SEQN']),
        'subgroup': sg, 'p_initial': p, 'p_final': p_final,
        'delta_norm': np.linalg.norm(delta), 'mahalanobis': mahal,
        'inside_ellipsoid': inside, 'actionable_changes': changes
    })
    print(f"     Patient {cohort.iloc[global_idx]['SEQN']}: p={p:.3f}->{p_final:.3f}, ||delta||={np.linalg.norm(delta):.3f}, Mahal={mahal:.3f}, inside={inside}")

if cf_results:
    inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
    success_count = sum(1 for r in cf_results if r['p_final'] <= 0.5)
    print(f"   Summary: {len(cf_results)} patients, {success_count} reached target, {inside_count} inside ellipsoid")

# ============================================================
# 8. ATTRIBUTION SIGNIFICANCE
# ============================================================
print("\n8. ATTRIBUTION SIGNIFICANCE")

def marginal_effect(z, feature_idx, clf):
    p_orig = clf.predict_proba(z.reshape(1, -1))[0, 1]
    z_pert = z.copy()
    z_pert[feature_idx] = 0.0
    p_pert = clf.predict_proba(z_pert.reshape(1, -1))[0, 1]
    return p_orig - p_pert

# Calibration marginal effects
marginal_effects_cal = {}
for j in range(len(z_cols)):
    effects = []
    for i in range(len(X_cal)):
        effects.append(marginal_effect(X_cal[i], j, clf))
    marginal_effects_cal[j] = np.array(effects)

# Test marginal effects
marginal_effects_test = {}
for j in range(len(z_cols)):
    effects = []
    for i in range(len(X_test)):
        effects.append(marginal_effect(X_test[i], j, clf))
    marginal_effects_test[j] = np.array(effects)

# Subgroup-conditional conformal p-values
alpha_att = 0.1
significance_results = {}
for g in unique_subgroups:
    significance_results[g] = {}
    cal_mask = (sg_cal == g)
    test_mask = (sg_test == g)
    if cal_mask.sum() < 3: continue
    for j in range(len(z_cols)):
        feat_name = z_cols[j]
        cal_scores = np.abs(marginal_effects_cal[j][cal_mask])
        test_scores = np.abs(marginal_effects_test[j][test_mask])
        if len(cal_scores) == 0 or len(test_scores) == 0: continue
        p_values = [(np.sum(cal_scores >= ts) + 1) / (len(cal_scores) + 1) for ts in test_scores]
        significant = np.array(p_values) < alpha_att
        significance_results[g][feat_name] = {
            'p_values': np.array(p_values), 'significant': significant,
            'mean_effect': np.mean(marginal_effects_test[j][test_mask])
        }

# Global significance
print("   Global feature significance:")
global_sig = {}
for j in range(len(z_cols)):
    feat_name = z_cols[j]
    all_p = []
    all_eff = []
    for g in unique_subgroups:
        if g in significance_results and feat_name in significance_results[g]:
            all_p.extend(significance_results[g][feat_name]['p_values'])
            all_eff.extend([significance_results[g][feat_name]['mean_effect']] * len(significance_results[g][feat_name]['p_values']))
    if all_p:
        global_sig[feat_name] = {'sig_frac': np.mean(np.array(all_p) < alpha_att), 'mean_p': np.mean(all_p), 'mean_eff': np.mean(all_eff)}

for feat, stats in sorted(global_sig.items(), key=lambda x: x[1]['sig_frac'], reverse=True)[:10]:
    print(f"   {feat}: sig={stats['sig_frac']:.1%}, mean_p={stats['mean_p']:.3f}, mean_eff={stats['mean_eff']:+.4f}")

# ============================================================
# 9. AUDITOR & TRANSLATOR
# ============================================================
print("\n9. AUDITOR & TRANSLATOR")

# Build evidence schemas
evidence_list = []
for i in range(len(X_test)):
    z = X_test[i]
    sg = sg_test[i]
    p = p_test[i]
    y_true = y_test[i]
    subject_id = cohort.iloc[test_idx[i]]['SEQN']
    
    # Prediction set (matching step 6 logic)
    # Class 0 (no diabetes): A = p, include if p <= q
    # Class 1 (diabetes): A = 1-p, include if 1-p <= q
    q = final_quantiles.get(sg, global_q)
    pred_set = []
    if (1 - p) <= q: pred_set.append(1)  # Class 1: A = 1-p
    if p <= q: pred_set.append(0)  # Class 0: A = p
    
    # Counterfactual
    cf = None
    for r in cf_results:
        if r['subject_id'] == subject_id:
            cf = r
            break
    
    # Attributions
    attributions = {}
    if sg in significance_results:
        for feat_name, res in significance_results[sg].items():
            j = z_cols.index(feat_name)
            test_idx_in_sg = np.where(sg_test == sg)[0]
            patient_pos = np.where(test_idx_in_sg == i)[0]
            if len(patient_pos) > 0:
                pos = patient_pos[0]
                attributions[feat_name] = {
                    'marginal_effect': res['p_values'][pos],  # placeholder
                    'p_value': res['p_values'][pos],
                    'significant': res['significant'][pos]
                }
    
    ev = {
        'subject_id': int(subject_id),
        'subgroup': sg,
        'predicted_risk': float(p),
        'true_label': int(y_true),
        'prediction_set': pred_set,
        'coverage_level': 1 - alpha,
        'counterfactual': cf,
        'attributions': attributions
    }
    evidence_list.append(ev)

# Audit
coverage_by_sg = {}
for ev in evidence_list:
    sg = ev['subgroup']
    if sg not in coverage_by_sg:
        coverage_by_sg[sg] = {'covered': 0, 'total': 0}
    coverage_by_sg[sg]['total'] += 1
    if ev['true_label'] in ev['prediction_set']:
        coverage_by_sg[sg]['covered'] += 1

print("   Coverage by subgroup:")
for sg, stats in coverage_by_sg.items():
    cov = stats['covered'] / stats['total'] if stats['total'] > 0 else 0
    stats['coverage'] = cov
    print(f"     {sg}: {cov:.3f} (n={stats['total']})")

# Abstention: defer if subgroup coverage < 80%
abstain_count = 0
for ev in evidence_list:
    sg = ev['subgroup']
    sg_cov = coverage_by_sg.get(sg, {}).get('coverage', 1.0)
    if sg_cov < 0.8:
        abstain_count += 1

print(f"   Patients abstained: {abstain_count}/{len(evidence_list)}")

# Sample translation
print("\n   SAMPLE EXPLANATION:")
ev = evidence_list[0]
print(f"   Patient {ev['subject_id']} (Subgroup: {ev['subgroup']})")
print(f"   Predicted risk: {ev['predicted_risk']:.1%}")
print(f"   Prediction set: {ev['prediction_set']}")
if ev['counterfactual']:
    cf = ev['counterfactual']
    print(f"   Counterfactual: {cf['p_initial']:.1%} -> {cf['p_final']:.1%}, inside={cf['inside_ellipsoid']}")
    for feat, change in cf['actionable_changes'].items():
        print(f"     {feat}: {change:+.2f} SD")

# ============================================================
# 10. SAVE ALL ARTIFACTS
# ============================================================
print("\n10. SAVING ARTIFACTS")

artifacts = {
    'cohort': cohort,
    'embeddings': embeddings,
    'z_cols': z_cols,
    'clf': clf,
    'final_quantiles': final_quantiles,
    'global_q': global_q,
    'alpha': alpha,
    'subgroup_stats': subgroup_stats,
    'cf_results': cf_results,
    'significance_results': significance_results,
    'global_significance': global_sig,
    'evidence_list': evidence_list,
    'subgroup_col': 'subgroup_age'
}

with open(DATA_DIR / "nhanes_pipeline_artifacts.pkl", 'wb') as f:
    pickle.dump(artifacts, f)

print("   All artifacts saved to nhanes_pipeline_artifacts.pkl")

print("\n" + "=" * 60)
print("NHANES PIPELINE COMPLETE")
print("=" * 60)