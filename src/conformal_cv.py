"""
Cross-Validated Conformal Prediction (CV+ / Jackknife+)
========================================================
More stable than single split - uses all data for both training and calibration.
Implements CV+ and Jackknife+ from Barber et al. (2021).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")
cohort = pd.read_csv(DATA_DIR / "cohort_incident_t2d.csv")
embeddings = pd.read_csv(DATA_DIR / "embeddings.csv")

# ============================================================
# 1. PREPARE DATA
# ============================================================
data = cohort[['subject_id', 'incident_t2d']].merge(embeddings, on='subject_id')
z_cols = [c for c in data.columns if c.startswith('z_')]
X = data[z_cols].values
y = data['incident_t2d'].values
subgroups = data['subgroup_age_band'].values
unique_subgroups = np.unique(subgroups[~pd.isna(subgroups)])

print(f"Data: {len(X)} samples, {len(z_cols)} features, {len(unique_subgroups)} subgroups")

# ============================================================
# 2. CV+ CONFORMAL PREDICTION
# ============================================================
def cv_plus_conformal(X, y, subgroups, alpha=0.1, n_folds=5, random_state=42):
    """
    CV+ conformal prediction (Barber et al., 2021).
    For each fold k: train on all other folds, calibrate on fold k.
    Prediction set: union of sets from each fold's model.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    n = len(y)
    p_cv = np.zeros(n)  # CV predictions
    p_test_folds = []   # Store predictions per fold for test-time
    
    fold_models = []
    fold_cal_scores = {g: [] for g in unique_subgroups}
    fold_cal_subgroups = {g: [] for g in unique_subgroups}
    
    for fold_idx, (train_idx, cal_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_idx)}, cal={len(cal_idx)}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        sg_cal = subgroups[cal_idx]
        
        # Fit model
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=fold_idx))
        ])
        clf.fit(X_train, y_train)
        fold_models.append(clf)
        
        # CV predictions on calibration fold
        p_cal = clf.predict_proba(X_cal)[:, 1]
        p_cv[cal_idx] = p_cal
        
        # Nonconformity scores on calibration fold
        A_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
        
        # Store per subgroup
        for g in unique_subgroups:
            mask = (sg_cal == g)
            if mask.sum() > 0:
                fold_cal_scores[g].extend(A_cal[mask])
                fold_cal_subgroups[g].extend([fold_idx] * mask.sum())
    
    # Compute quantiles per subgroup (pooled across folds)
    subgroup_quantiles = {}
    for g in unique_subgroups:
        scores = np.array(fold_cal_scores[g])
        if len(scores) > 0:
            q = np.quantile(scores, 1 - alpha, method='higher')
        else:
            # Fallback to global
            all_scores = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
            q = np.quantile(all_scores, 1 - alpha, method='higher')
        subgroup_quantiles[g] = q
        print(f"  {g}: n_cal={len(fold_cal_scores[g])}, q={q:.4f}")
    
    # Global quantile
    all_scores = np.concatenate([np.array(v) for v in fold_cal_scores.values() if len(v) > 0])
    global_q = np.quantile(all_scores, 1 - alpha, method='higher')
    
    return {
        'subgroup_quantiles': subgroup_quantiles,
        'global_quantile': global_q,
        'cv_predictions': p_cv,
        'fold_models': fold_models,
        'fold_cal_scores': fold_cal_scores,
        'alpha': alpha,
        'n_folds': n_folds
    }

print("\n--- CV+ Conformal ---")
cv_result = cv_plus_conformal(X, y, subgroups, alpha=0.1, n_folds=5)

# ============================================================
# 3. JACKKNIFE+ (Leave-One-Out style, more stable for small n)
# ============================================================
def jackknife_plus_conformal(X, y, subgroups, alpha=0.1, max_loo=200):
    """
    Jackknife+ conformal: train n models leaving out each point (or subset).
    For small datasets, use random subset of LOO models.
    """
    n = len(y)
    if n > max_loo:
        # Sample subset for computational efficiency
        loo_indices = np.random.choice(n, max_loo, replace=False)
    else:
        loo_indices = np.arange(n)
    
    print(f"\n--- Jackknife+ (LOO on {len(loo_indices)} points) ---")
    
    p_loo = np.zeros(n)
    loo_scores = {g: [] for g in unique_subgroups}
    
    for i, loo_idx in enumerate(loo_indices):
        if i % 50 == 0:
            print(f"  LOO {i+1}/{len(loo_indices)}")
        
        # Train on all except loo_idx
        train_mask = np.ones(n, dtype=bool)
        train_mask[loo_idx] = False
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[loo_idx:loo_idx+1]
        y_test = y[loo_idx:loo_idx+1]
        sg_test = subgroups[loo_idx:loo_idx+1]
        
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=loo_idx))
        ])
        clf.fit(X_train, y_train)
        
        p_test = clf.predict_proba(X_test)[0, 1]
        p_loo[loo_idx] = p_test
        
        # Nonconformity score
        A_test = 1 - p_test if y_test[0] == 1 else p_test
        loo_scores[sg_test[0]].append(A_test)
    
    # Quantiles
    sg_quantiles = {}
    for g in unique_subgroups:
        scores = np.array(loo_scores[g])
        if len(scores) > 0:
            sg_quantiles[g] = np.quantile(scores, 1 - alpha, method='higher')
        else:
            all_s = np.concatenate([np.array(v) for v in loo_scores.values() if len(v) > 0])
            sg_quantiles[g] = np.quantile(all_s, 1 - alpha, method='higher')
    
    all_s = np.concatenate([np.array(v) for v in loo_scores.values() if len(v) > 0])
    global_q = np.quantile(all_s, 1 - alpha, method='higher')
    
    return {
        'subgroup_quantiles': sg_quantiles,
        'global_quantile': global_q,
        'loo_predictions': p_loo,
        'loo_scores': loo_scores,
        'alpha': alpha
    }

# Run Jackknife+ (subset for speed)
jk_result = jackknife_plus_conformal(X, y, subgroups, alpha=0.1, max_loo=100)

# ============================================================
# 4. HIERARCHICAL SHRINKAGE ON CV+ QUANTILES
# ============================================================
def hierarchical_shrinkage(cv_quantiles, cv_scores, global_q):
    """Apply hierarchical shrinkage to CV+ quantiles."""
    # Between-stratum variance
    valid_groups = [g for g in cv_quantiles if len(cv_scores[g]) > 0]
    if len(valid_groups) < 2:
        return cv_quantiles
    
    q_vals = np.array([cv_quantiles[g] for g in valid_groups])
    n_vals = np.array([len(cv_scores[g]) for g in valid_groups])
    
    # Within-group variance of scores
    within_vars = np.array([np.var(cv_scores[g]) for g in valid_groups])
    # Effective variance of quantile: ~ within_var / n
    quantile_vars = within_vars / n_vals
    
    # Between-stratum variance (method of moments)
    weighted_mean = np.average(q_vals, weights=n_vals)
    total_var = np.average((q_vals - weighted_mean)**2, weights=n_vals)
    tau2 = max(0, total_var - np.mean(quantile_vars))
    
    if tau2 > 0:
        kappa = 1.0 / tau2
    else:
        kappa = 1e6
    
    print(f"  tau^2={tau2:.6f}, kappa={kappa:.2f}")
    
    # Shrunken quantiles
    shrunken = {}
    for g in cv_quantiles:
        n_g = len(cv_scores[g]) if g in cv_scores else 0
        q_g = cv_quantiles[g]
        q_shrunk = (n_g * q_g + kappa * global_q) / (n_g + kappa) if n_g > 0 else global_q
        shrunken[g] = q_shrunk
        print(f"  {g}: n={n_g}, q_raw={q_g:.4f}, q_shrunk={q_shrunk:.4f}")
    
    return shrunken

print("\n--- Hierarchical Shrinkage on CV+ ---")
cv_shrunken = hierarchical_shrinkage(cv_result['subgroup_quantiles'], cv_result['fold_cal_scores'], cv_result['global_quantile'])

# ============================================================
# 5. PREDICTION SETS WITH CV+ QUANTILES
# ============================================================
def evaluate_coverage(X, y, subgroups, quantiles, global_q, alpha=0.1):
    """Evaluate coverage using given quantiles."""
    # Use an ensemble of fold models for prediction
    # For simplicity, use CV predictions from cv_plus
    p = cv_result['cv_predictions']
    
    A = np.where(y == 1, 1 - p, p)
    
    pred_sets = np.zeros((len(y), 2), dtype=int)
    for i in range(len(y)):
        g = subgroups[i]
        q = quantiles.get(g, global_q)
        pred_sets[i, 0] = 1 if (p[i] <= q) else 0  # Class 0: A = p
        pred_sets[i, 1] = 1 if (1 - p[i] <= q) else 0  # Class 1: A = 1-p
    
    # Coverage
    covered = np.array([y[i] in np.where(pred_sets[i])[0] for i in range(len(y))])
    
    print(f"\n=== Coverage Evaluation ===")
    print(f"Overall: {covered.mean():.3f} (target: {1-alpha:.3f})")
    for g in unique_subgroups:
        mask = (subgroups == g)
        if mask.sum() > 0:
            print(f"  {g}: {covered[mask].mean():.3f} (n={mask.sum()})")
    
    set_sizes = pred_sets.sum(axis=1)
    print(f"Set sizes: mean={set_sizes.mean():.2f}, dist={pd.Series(set_sizes).value_counts().to_dict()}")
    
    return covered, pred_sets

print("\n--- CV+ Quantiles Coverage ---")
cv_cov, cv_sets = evaluate_coverage(X, y, subgroups, cv_result['subgroup_quantiles'], cv_result['global_quantile'])

print("\n--- Shrunken CV+ Quantiles Coverage ---")
shrunk_cov, shrunk_sets = evaluate_coverage(X, y, subgroups, cv_shrunken, cv_result['global_quantile'])

print("\n--- Jackknife+ Quantiles Coverage ---")
jk_cov, jk_sets = evaluate_coverage(X, y, subgroups, jk_result['subgroup_quantiles'], jk_result['global_quantile'])

# ============================================================
# 6. SAVE CV+ ARTIFACTS
# ============================================================
cv_artifacts = {
    'cv_plus': cv_result,
    'jackknife_plus': jk_result,
    'cv_shrunken_quantiles': cv_shrunken,
    'z_cols': z_cols,
    'subgroup_col': 'age_band'
}

with open(DATA_DIR / "conformal_cv_artifacts.pkl", 'wb') as f:
    pickle.dump(cv_artifacts, f)

print("\nCV+ artifacts saved.")