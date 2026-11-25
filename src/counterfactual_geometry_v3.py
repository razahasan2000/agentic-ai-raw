"""
Conformalized Counterfactual Geometry - Full Embedding with Actionable Mask
============================================================================
- Keep full 13-dim embedding for classifier compatibility
- Only optimize actionable features
- Diagonal covariance on full space with strong regularization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.linalg import pinvh
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
# 1. FULL EMBEDDING
# ============================================================
data = cohort[['subject_id', 'incident_t2d']].merge(embeddings, on='subject_id')
X = data[z_cols].values
y = data['incident_t2d'].values
subgroups = data[f'subgroup_{subgroup_col}'].values
p_all = clf.predict_proba(X)[:, 1]

# Actionable feature indices in FULL embedding space
ACTIONABLE_FEATURES = [
    'z_vital_bmi_kg_m2',
    'z_vital_weight_lbs',
    'z_lab_hba1c',
    'z_lab_glucose_chem',
    'z_lab_glucose_blood_gas',
]
actionable_idx = [z_cols.index(f) for f in ACTIONABLE_FEATURES if f in z_cols]
non_actionable_idx = [i for i in range(len(z_cols)) if i not in actionable_idx]

print(f"Full dims: {len(z_cols)}")
print(f"Actionable idx: {actionable_idx} -> {[z_cols[i] for i in actionable_idx]}")

# ============================================================
# 2. DIAGONAL COVARIANCE ON FULL SPACE
# ============================================================
print("\n--- Subgroup Diagonal Covariance (Full Space) ---")
subgroup_stats = {}

for g in np.unique(subgroups[~pd.isna(subgroups)]):
    mask = (subgroups == g)
    X_g = X[mask]
    
    if len(X_g) < 3:
        continue
    
    mu_g = X_g.mean(axis=0)
    var_g = np.var(X_g, axis=0, ddof=1)
    reg = 1.0
    var_g_reg = var_g + reg
    prec_g = 1.0 / var_g_reg
    
    r2_g = final_quantiles.get(g, cal['global_quantile'])
    
    subgroup_stats[g] = {
        'mu': mu_g,
        'var': var_g_reg,
        'prec': prec_g,
        'r2': r2_g,
        'n': len(X_g)
    }
    print(f"  {g}: n={len(X_g)}, r^2={r2_g:.3f}")

# Global
mu_global = X.mean(axis=0)
var_global = np.var(X, axis=0, ddof=1) + 1.0
prec_global = 1.0 / var_global
r2_global = cal['global_quantile']

# ============================================================
# 3. MAHALANOBIS (DIAGONAL)
# ============================================================
def mahalanobis_diag(z, mu, prec):
    diff = z - mu
    return np.sum(diff**2 * prec)

# ============================================================
# 4. COUNTERFACTUAL VIA GRADIENT DESCENT (Full Space)
# ============================================================
def counterfactual_gradient(z, subgroup, target_prob=0.5, lr=0.05, max_steps=300):
    stats = subgroup_stats.get(subgroup, {
        'mu': mu_global, 'prec': prec_global, 'r2': r2_global
    })
    mu = stats['mu']
    prec = stats['prec']
    r2 = stats['r2']
    
    delta = np.zeros(len(z))
    z_current = z.copy()
    
    for step in range(max_steps):
        p = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
        
        if p <= target_prob:
            break
        
        # Gradient only on actionable features
        grad = np.zeros(len(z))
        eps = 1e-4
        for i in actionable_idx:
            z_plus = z_current.copy()
            z_plus[i] += eps
            p_plus = clf.predict_proba(z_plus.reshape(1, -1))[0, 1]
            grad[i] = (p_plus - p) / eps
        
        # Mahalanobis penalty gradient (all features, but only actionable updated)
        mahal = mahalanobis_diag(z_current, mu, prec)
        penalty_weight = 5.0 if mahal > r2 else 0.5
        for i in actionable_idx:
            grad[i] += penalty_weight * 2 * (z_current[i] - mu[i]) * prec[i]
        
        # Update
        for i in actionable_idx:
            delta[i] -= lr * grad[i]
            delta[i] = np.clip(delta[i], -4, 4)
        
        z_current = z + delta
        
        if np.max(np.abs(grad[actionable_idx])) < 1e-4:
            break
    
    p_final = clf.predict_proba(z_current.reshape(1, -1))[0, 1]
    mahal_final = mahalanobis_diag(z_current, mu, prec)
    inside = mahal_final <= r2 * 1.5
    
    return delta, mahal_final, inside, p_final

# ============================================================
# 5. TEST SET HIGH-RISK PATIENTS
# ============================================================
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
cal_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

test_indices = temp_idx[test_idx]
high_risk_test = test_indices[p_all[test_indices] > 0.5]

print(f"\n--- Counterfactuals for {len(high_risk_test)} High-Risk Test Patients ---")

results = []
for idx in high_risk_test:
    z = X[idx]
    sg = subgroups[idx]
    p = p_all[idx]
    
    delta, mahal, inside, p_final = counterfactual_gradient(z, sg, target_prob=0.5)
    
    actionable_changes = {}
    for i in actionable_idx:
        if abs(delta[i]) > 1e-3:
            actionable_changes[z_cols[i]] = delta[i]
    
    results.append({
        'subject_id': data.iloc[idx]['subject_id'],
        'subgroup': sg,
        'p_initial': p,
        'p_final': p_final,
        'delta_norm': np.linalg.norm(delta),
        'mahalanobis': mahal,
        'inside_ellipsoid': inside,
        'actionable_changes': actionable_changes,
    })
    
    print(f"\n  Patient {data.iloc[idx]['subject_id']} ({sg}):")
    print(f"    p: {p:.3f} -> {p_final:.3f}, ||delta||={np.linalg.norm(delta):.3f}, Mahal={mahal:.3f}, inside={inside}")
    for feat, change in actionable_changes.items():
        print(f"    {feat}: {change:+.3f} SD")

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n=== Counterfactual Summary ===")
if results:
    inside_count = sum(r['inside_ellipsoid'] for r in results)
    success_count = sum(1 for r in results if r['p_final'] <= 0.5)
    print(f"Total: {len(results)}")
    print(f"Reached target prob (<=0.5): {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Inside 1.5x ellipsoid: {inside_count} ({inside_count/len(results)*100:.1f}%)")
    print(f"Mean ||delta||: {np.mean([r['delta_norm'] for r in results]):.3f}")
    print(f"Mean Mahalanobis: {np.mean([r['mahalanobis'] for r in results]):.3f}")
    
    all_changes = {}
    for r in results:
        for feat, change in r['actionable_changes'].items():
            all_changes.setdefault(feat, []).append(change)
    
    print("\nActionable feature changes (mean SD):")
    for feat, changes in all_changes.items():
        print(f"  {feat}: {np.mean(changes):+.3f} ± {np.std(changes):.3f}")

# ============================================================
# 7. SAVE
# ============================================================
output = {
    'actionable_idx': actionable_idx,
    'actionable_features': [z_cols[i] for i in actionable_idx],
    'subgroup_stats': subgroup_stats,
    'global_stats': {'mu': mu_global, 'var': var_global, 'prec': prec_global, 'r2': r2_global},
    'counterfactuals': results,
    'z_cols': z_cols
}

with open(DATA_DIR / "counterfactual_results.pkl", 'wb') as f:
    pickle.dump(output, f)

print("\nResults saved.")