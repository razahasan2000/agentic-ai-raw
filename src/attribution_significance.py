"""
Conformalized Marginal Attribution Significance (Section 4.4)
==============================================================
For each feature j:
- Perturb: reset z_j to reference value (0 in normalized space)
- Compute marginal effect: dp = p(z) - p(z with z_j=0)
- Subgroup-conditional conformal set for this marginal effect
- Conformal p-value: p_{g,j} = sup{α : dp ∈ C_g^α(z)}
- Small p-value = factor has significant marginal effect beyond calibration noise
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")
cohort = pd.read_csv(DATA_DIR / "cohort_incident_t2d.csv")
embeddings = pd.read_csv(DATA_DIR / "embeddings.csv")

with open(DATA_DIR / "calibration_artifacts.pkl", 'rb') as f:
    cal = pickle.load(f)

clf = cal['clf']
z_cols = cal['z_cols']
alpha = cal['alpha']
final_quantiles = cal['final_quantiles']
subgroup_col = cal['subgroup_col']

# ============================================================
# 1. PREPARE DATA
# ============================================================
data = cohort[['subject_id', 'incident_t2d']].merge(embeddings, on='subject_id')
X = data[z_cols].values
y = data['incident_t2d'].values
subgroups = data[f'subgroup_{subgroup_col}'].values
p_all = clf.predict_proba(X)[:, 1]

# Recreate calibration split
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

cal_indices = temp_idx[cal_idx]
test_indices = temp_idx[test_idx]

X_cal = X[cal_indices]
y_cal = y[cal_indices]
subgroups_cal = subgroups[cal_indices]
p_cal = p_all[cal_indices]

X_test = X[test_indices]
y_test = y[test_indices]
subgroups_test = subgroups[test_indices]
p_test = p_all[test_indices]

unique_subgroups = np.unique(subgroups[~pd.isna(subgroups)])

# ============================================================
# 2. MARGINAL EFFECTS FOR EACH FEATURE
# ============================================================
def marginal_effect(z, feature_idx, clf):
    """dp = p(z) - p(z with z_j = 0)"""
    p_orig = clf.predict_proba(z.reshape(1, -1))[0, 1]
    z_pert = z.copy()
    z_pert[feature_idx] = 0.0  # Reference value in normalized space
    p_pert = clf.predict_proba(z_pert.reshape(1, -1))[0, 1]
    return p_orig - p_pert

# Compute marginal effects for all features on calibration set
print("Computing marginal effects on calibration set...")
marginal_effects_cal = {}
for j in range(len(z_cols)):
    effects = []
    for i in range(len(X_cal)):
        eff = marginal_effect(X_cal[i], j, clf)
        effects.append(eff)
    marginal_effects_cal[j] = np.array(effects)

# On test set
marginal_effects_test = {}
for j in range(len(z_cols)):
    effects = []
    for i in range(len(X_test)):
        eff = marginal_effect(X_test[i], j, clf)
        effects.append(eff)
    marginal_effects_test[j] = np.array(effects)

# ============================================================
# 3. SUBGROUP-CONDITIONAL CONFORMAL SETS FOR MARGINAL EFFECTS
# ============================================================
# For each subgroup g and feature j, we have calibration marginal effects
# Nonconformity score for marginal effect: |dp - median_g| or similar
# We want: p-value = P(|dp| >= |observed|) under subgroup calibration

print("\n--- Conformal Marginal Significance ---")
alpha_att = 0.1  # Significance level for attributions

# For each subgroup and feature, compute nonconformity scores
# Score = |dp| (absolute marginal effect)
# Higher = more surprising

significance_results = {}

for g in unique_subgroups:
    significance_results[g] = {}
    cal_mask = (subgroups_cal == g)
    test_mask = (subgroups_test == g)
    
    if cal_mask.sum() < 3:
        continue
    
    for j in range(len(z_cols)):
        feat_name = z_cols[j]
        
        # Calibration nonconformity scores for this feature in this subgroup
        cal_effects = marginal_effects_cal[j][cal_mask]
        cal_scores = np.abs(cal_effects)
        
        # Test effects
        test_effects = marginal_effects_test[j][test_mask]
        test_scores = np.abs(test_effects)
        
        if len(cal_scores) == 0 or len(test_scores) == 0:
            continue
        
        # Conformal p-values for each test point
        p_values = []
        for ts in test_scores:
            # p-value = proportion of calibration scores ≥ test score
            p_val = (np.sum(cal_scores >= ts) + 1) / (len(cal_scores) + 1)
            p_values.append(p_val)
        
        # Significance: reject null (no marginal effect) if p < alpha
        significant = np.array(p_values) < alpha_att
        
        significance_results[g][feat_name] = {
            'cal_scores': cal_scores,
            'test_scores': test_scores,
            'test_effects': test_effects,
            'p_values': np.array(p_values),
            'significant': significant,
            'mean_effect': np.mean(test_effects),
            'mean_abs_effect': np.mean(test_scores)
        }

# ============================================================
# 4. SUMMARY: FEATURE SIGNIFICANCE BY SUBGROUP
# ============================================================
print("\n=== Feature Significance by Subgroup (alpha=0.1) ===")
for g in unique_subgroups:
    if g not in significance_results:
        continue
    print(f"\n  Subgroup {g}:")
    for feat_name, res in significance_results[g].items():
        sig_frac = res['significant'].mean() if len(res['significant']) > 0 else 0
        mean_eff = res['mean_effect']
        mean_abs = res['mean_abs_effect']
        print(f"    {feat_name}: sig={sig_frac:.1%}, mean dp={mean_eff:+.4f}, mean |dp|={mean_abs:.4f}")

# ============================================================
# 5. PER-PATIENT ATTRIBUTION WITH CONFORMAL P-VALUES
# ============================================================
print("\n=== Per-Patient Attribution (Test Set) ===")
for idx in range(min(5, len(X_test))):
    z = X_test[idx]
    sg = subgroups_test[idx]
    p = p_test[idx]
    y_true = y_test[idx]
    
    print(f"\n  Patient {data.iloc[test_indices[idx]]['subject_id']} ({sg}): p={p:.3f}, y={y_true}")
    
    if sg in significance_results:
        for feat_name, res in significance_results[sg].items():
            j = z_cols.index(feat_name)
            # Get this patient's p-value
            test_idx_in_subgroup = np.where(subgroups_test == sg)[0]
            patient_pos = np.where(test_idx_in_subgroup == idx)[0]
            if len(patient_pos) > 0:
                pv = res['p_values'][patient_pos[0]]
                eff = res['test_effects'][patient_pos[0]]
                sig = res['significant'][patient_pos[0]]
                if pv < 0.2:  # Show notable ones
                    print(f"    {feat_name}: dp={eff:+.4f}, p={pv:.3f}, sig={sig}")

# ============================================================
# 6. GLOBAL FEATURE IMPORTANCE (Aggregated)
# ============================================================
print("\n=== Global Attribution Significance (All Subgroups) ===")
global_significance = {}
for j in range(len(z_cols)):
    feat_name = z_cols[j]
    all_p = []
    all_eff = []
    for g in unique_subgroups:
        if g in significance_results and feat_name in significance_results[g]:
            all_p.extend(significance_results[g][feat_name]['p_values'])
            all_eff.extend(significance_results[g][feat_name]['test_effects'])
    
    if all_p:
        global_significance[feat_name] = {
            'mean_p': np.mean(all_p),
            'median_p': np.median(all_p),
            'sig_frac': np.mean(np.array(all_p) < alpha_att),
            'mean_effect': np.mean(all_eff),
            'mean_abs_effect': np.mean(np.abs(all_eff))
        }

# Sort by significance fraction
sorted_feats = sorted(global_significance.items(), key=lambda x: x[1]['sig_frac'], reverse=True)
for feat_name, stats in sorted_feats:
    print(f"  {feat_name}: sig_frac={stats['sig_frac']:.1%}, mean_p={stats['mean_p']:.3f}, "
          f"mean dp={stats['mean_effect']:+.4f}, mean |dp|={stats['mean_abs_effect']:.4f}")

# ============================================================
# 7. SAVE
# ============================================================
output = {
    'significance_results': significance_results,
    'global_significance': global_significance,
    'alpha': alpha_att,
    'z_cols': z_cols,
    'marginal_effects_cal': marginal_effects_cal,
    'marginal_effects_test': marginal_effects_test
}

with open(DATA_DIR / "attribution_significance.pkl", 'wb') as f:
    pickle.dump(output, f)

print("\nAttribution significance saved.")