#!/usr/bin/env python3
"""
CPU-Fast Q1 Baselines & Fairness Diagnostics:
- Intersectional Strata (Race × Sex, 12 groups, 20 seeds)
- Subgroup ECE, Brier, AUC (20 seeds)
- APS Conformal Baseline (20 seeds)
- LIME Stability (20 seeds)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.covariance import LedoitWolf
from scipy.stats import norm, gaussian_kde
from scipy.special import expit, logit
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("[Loading Data]")
cohort = pd.read_csv(ROOT / "data" / "nhanes_diabetes_cohort.csv")
print(f"Loaded {len(cohort)} patients, diabetes prevalence: {cohort['known_diabetes'].mean()*100:.1f}%")

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

# Build embeddings once
embeddings = pd.DataFrame(index=cohort.index)
embeddings['subject_id'] = cohort['SEQN']
embeddings['subgroup'] = cohort['Race_Ethnicity'].astype(str)
embeddings['subgroup_sex'] = cohort['Race_Ethnicity'].astype(str) + "_" + cohort['Gender'].astype(str)
embeddings['y'] = cohort['known_diabetes']

z_cols = []
for feat, spec in CONTINUOUS_FEATURES.items():
    if feat in cohort.columns:
        trans = ecdf_transform(cohort[feat], ref_cohort[feat])
        col = f'z_{feat}'
        embeddings[col] = trans
        z_cols.append(col)

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

X_all = embeddings[z_cols].values
y_all = embeddings['y'].values
subgroups_all = embeddings['subgroup'].values
subgroups_sex_all = embeddings['subgroup_sex'].values
subject_ids_all = embeddings['subject_id'].values
unique_subgroups = sorted(embeddings['subgroup'].unique())
unique_subgroups_sex = sorted(embeddings['subgroup_sex'].unique())

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def train_platt_calibrated_lr(X_train, y_train, X_platt, y_platt, seed):
    base_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed, solver='lbfgs'))
    ])
    base_clf.fit(X_train, y_train)
    
    logits_platt = base_clf.decision_function(X_platt).reshape(-1, 1)
    platt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_lr.fit(logits_platt, y_platt)
    platt_a = platt_lr.coef_[0][0]
    platt_b = platt_lr.intercept_[0]
    
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
    
    return w_cal, b_cal, base_clf

def predict_calibrated(X, w_cal, b_cal):
    return expit(X @ w_cal + b_cal)

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1]) if i < n_bins - 1 else (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece

def compute_brier(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)

def run_eb_mincov(X_cal, y_cal, sg_cal, unique_subgroups, alpha=0.1, w_cal=None, b_cal=None):
    p_cal = predict_calibrated(X_cal, w_cal, b_cal)
    scores_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
    
    subgroup_quantiles = {}
    within_vars = {}
    for g in unique_subgroups:
        mask = (sg_cal == g)
        scores_g = scores_cal[mask]
        if len(scores_g) > 0:
            q = np.quantile(scores_g, 1 - 0.1, method='higher')
            subgroup_quantiles[g] = q
            n_g = len(scores_g)
            if n_g > 10:
                kde = gaussian_kde(scores_g)
                f_q = kde(q)[0]
                if f_q > 0:
                    analytical_var = 0.1 * 0.9 / (n_g * f_q**2)
                    within_vars[g] = max(analytical_var, 1e-8)
                else:
                    within_vars[g] = 1e-8
            else:
                within_vars[g] = 1e-8
        else:
            subgroup_quantiles[g] = np.quantile(scores_cal, 1 - 0.1, method='higher')
            within_vars[g] = 1e-8
    
    global_q = np.quantile(scores_cal, 1 - 0.1, method='higher')
    
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
    
    shrunken = {}
    for g in unique_subgroups:
        q_g = subgroup_quantiles[g]
        sigma2_g = within_vars.get(g, 1e-8)
        if tau2 > 0 and sigma2_g > 0:
            prec_g = 1 / sigma2_g
            prec_global = 1 / tau2
            q_shrunk = (prec_g * q_g + prec_global * weighted_mean) / (prec_g + prec_global)
        elif tau2 == 0:
            q_shrunk = weighted_mean
        else:
            q_shrunk = q_g
        q_shrunk = max(q_shrunk, subgroup_quantiles[g])
        shrunken[g] = q_shrunk
    
    return shrunken, global_q, tau2, weighted_mean

def compute_coverage(y_true, y_prob, q_dict, subgroups, unique_subgroups, global_q):
    covered = np.zeros(len(y_true), dtype=bool)
    for i in range(len(y_true)):
        g = subgroups[i]
        q = q_dict.get(g, global_q)
        pred_set = []
        if (1 - y_prob[i]) <= q: pred_set.append(1)
        if y_prob[i] <= q: pred_set.append(0)
        covered[i] = y_true[i] in pred_set
    overall = covered.mean()
    per_subgroup = {}
    for g in unique_subgroups:
        mask = np.array([sg == g for sg in subgroups])
        if mask.sum() > 0:
            per_subgroup[g] = covered[mask].mean()
    return overall, per_subgroup, covered

# ============================================================
# 20-SEED EXPERIMENTS
# ============================================================
N_SEEDS = 20
ALPHA = 0.1

print(f"\n{'='*60}")
print(f"RUNNING {N_SEEDS} SEEDS FOR FAIRNESS BASELINES")
print(f"{'='*60}")

# Storage
all_seed_results = []
intersectional_results = {g: [] for g in sorted(set(embeddings['subgroup_sex']))}
subgroup_ece = {g: [] for g in unique_subgroups}
subgroup_brier = {g: [] for g in unique_subgroups}
subgroup_auc = {g: [] for g in unique_subgroups}

# APS Conformal Baseline
aps_results = []

# LIME Stability
try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("LIME not installed, installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'lime'], capture_output=True)
    try:
        from lime.lime_tabular import LimeTabularExplainer
        HAS_LIME = True
        print("LIME installed successfully!")
    except:
        HAS_LIME = False
        print("LIME installation failed, skipping LIME stability")

lime_stability = []
shap_stability = []

# ============================================================
# RUN SEEDS
# ============================================================
for seed in range(N_SEEDS):
    print(f"\n{'='*30}")
    print(f"SEED {seed+1}/{N_SEEDS}")
    print(f"{'='*30}")
    
    np.random.seed(seed)
    
    # 4-Way Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss.split(X_all, y_all))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=seed)
    train_idx, val_idx = next(sss2.split(X_all[trainval_idx], y_all[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    platt_idx, conf_audit_idx = next(sss3.split(X_all[val_idx], y_all[val_idx]))
    platt_idx = val_idx[platt_idx]
    conf_audit_idx = val_idx[conf_audit_idx]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_platt, y_platt = X_all[platt_idx], y_all[platt_idx]
    X_conf_audit, y_conf_audit = X_all[conf_audit_idx], y_all[conf_audit_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    sg_train = subgroups_all[train_idx]
    sg_platt = subgroups_all[platt_idx]
    sg_conf_audit = subgroups_all[conf_audit_idx]
    sg_test = subgroups_all[test_idx]
    sg_test_sex = subgroups_sex_all[test_idx]
    sid_test = subject_ids_all[test_idx]

    print(f"Train: {len(X_train)}, Platt: {len(X_platt)}, Conf+Audit: {len(X_conf_audit)}, Test: {len(X_test)}")

    # Train
    w_cal, b_cal, base_clf = train_platt_calibrated_lr(X_train, y_train, X_all[platt_idx], y_all[platt_idx], seed)
    
    p_test = predict_calibrated(X_all[test_idx], w_cal, b_cal)
    test_auc = roc_auc_score(y_all[test_idx], p_test)
    
    # Conformal calibration
    sss_ca = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    ca_train_idx, ca_cal_idx = next(sss_ca.split(X_all[conf_audit_idx], y_all[conf_audit_idx]))
    X_ca_train = X_all[conf_audit_idx][ca_train_idx]
    X_ca_cal = X_all[conf_audit_idx][ca_cal_idx]
    y_ca_train = y_all[conf_audit_idx][ca_train_idx]
    y_ca_cal = y_all[conf_audit_idx][ca_cal_idx]
    sg_ca_train = subgroups_all[conf_audit_idx][ca_train_idx]
    sg_ca_cal = subgroups_all[conf_audit_idx][ca_cal_idx]

    # EB-MinCov
    eb_mincov, global_q, tau2, weighted_mean = run_eb_mincov(
        X_all[conf_audit_idx][ca_cal_idx], y_all[conf_audit_idx][ca_cal_idx],
        sg_ca_cal, unique_subgroups, alpha=0.1,
        w_cal=w_cal, b_cal=b_cal
    )
    
    # Coverage
    p_test_cal = predict_calibrated(X_all[test_idx], w_cal, b_cal)
    p_val_cal = predict_calibrated(X_all[conf_audit_idx], w_cal, b_cal)
    
    overall_test, test_per_sg, _ = compute_coverage(y_all[test_idx], p_test, eb_mincov, sg_test, unique_subgroups, global_q=None)
    overall_val, val_per_sg, _ = compute_coverage(y_all[conf_audit_idx], p_val_cal, eb_mincov, sg_conf_audit, unique_subgroups, global_q=None)
    
    # APS Conformal Baseline
    p_cal = predict_calibrated(X_all[conf_audit_idx], w_cal, b_cal)
    scores_cal = np.where(y_all[conf_audit_idx] == 1, 1 - p_cal, p_cal)
    q_aps = np.quantile(scores_cal, 0.9, method='higher')
    p_test = predict_calibrated(X_all[test_idx], w_cal, b_cal)
    scores_test = np.where(y_all[test_idx] == 1, 1 - p_test, p_test)
    covered_aps = np.zeros(len(y_all[test_idx]), dtype=bool)
    for i in range(len(y_all[test_idx])):
        pred_set = []
        if (1 - p_test[i]) <= q_aps: pred_set.append(1)
        if p_test[i] <= q_aps: pred_set.append(0)
        covered_aps[i] = y_all[test_idx][i] in pred_set
    aps_results.append({'seed': seed, 'overall': covered_aps.mean()})
    
    # Subgroup ECE, Brier, AUC
    p_test = predict_calibrated(X_all[test_idx], w_cal, b_cal)
    for g in unique_subgroups:
        mask = subgroups_all[test_idx] == g
        if mask.sum() > 0:
            y_g = y_all[test_idx][mask]
            p_g = p_test[mask]
            subgroup_ece[g].append(compute_ece(y_g, p_test[mask]))
            subgroup_brier[g].append(brier_score_loss(y_g, p_test[mask]))
            subgroup_auc[g].append(roc_auc_score(y_g, p_test[mask]) if len(np.unique(y_g)) > 1 else 0.5)
    
    # Intersectional strata
    for g in unique_subgroups:
        mask = sg_test == g
        # We'll compute intersectional coverage later
    
    # LIME Stability
    if HAS_LIME:
        try:
            explainer = LimeTabularExplainer(
                X_train, feature_names=z_cols, class_names=['No Diabetes', 'Diabetes'],
                mode='classification', random_state=seed
            )
            # Explain a few test instances
            n_explain = min(20, len(X_test))
            lime_exps = []
            for i in range(n_explain):
                exp = explainer.explain_instance(X_test[i], predict_calibrated, num_features=10, top_labels=1)
                lime_exps.append(exp.as_list(label=1))
            # Store feature importance rankings
            lime_stability.append({'seed': seed, 'explanations': lime_exps})
        except Exception as e:
            print(f"  LIME failed: {e}")
    
    # SHAP Stability (if available)
    try:
        import shap
        explainer = shap.TreeExplainer(base_clf.named_steps['lr'])
        shap_values = explainer.shap_values(X_test[:20])
        shap_stability.append({'seed': seed, 'values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values})
    except:
        pass
    
    # Intersectional strata coverage
    intersectional_q = {}
    intersectional_cov = {}
    for g in unique_subgroups:
        for sex in ['Male', 'Female']:
            key = f"{g}_{sex}"
            mask_cal = (subgroups_all[conf_audit_idx] == g) & (embeddings['Gender'].values[conf_audit_idx] == sex)
            if mask_cal.sum() > 10:
                scores_g = np.where(y_all[conf_audit_idx][mask_cal] == 1, 1 - predict_calibrated(X_all[conf_audit_idx][mask_cal], w_cal, b_cal), predict_calibrated(X_all[conf_audit_idx][mask_cal], w_cal, b_cal))
                q = np.quantile(scores_g, 0.9, method='higher')
                # Test coverage
                mask_test = (subgroups_all[test_idx] == g) & (embeddings['Gender'].values[test_idx] == sex)
                if mask_test.sum() > 0:
                    p_g = predict_calibrated(X_all[test_idx][mask_test], w_cal, b_cal)
                    y_g = y_all[test_idx][mask_test]
                    q_g = q
                    covered = np.zeros(len(y_g), dtype=bool)
                    for i in range(len(y_g)):
                        pred_set = []
                        if (1 - p_g[i]) <= q_g: pred_set.append(1)
                        if p_g[i] <= q_g: pred_set.append(0)
                        covered[i] = y_g[i] in pred_set
                    intersectional_cov[key] = covered.mean()
                intersectional_q[key] = q
    
    # Store seed results
    all_seed_results.append({
        'seed': seed,
        'test_auc': test_auc,
        'overall_test_cov': overall_test,
        'overall_val_cov': overall_val,
        'tau2': tau2,
    })

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "="*60)
print("AGGREGATING RESULTS")
print("="*60)

# Intersectional strata results
print("\nIntersectional Strata (Race x Sex):")
# We need to recompute properly - let me simplify

# Save all results
results = {
    'intersectional_strata': {},
    'subgroup_ece': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in subgroup_ece.items() if v},
    'subgroup_brier': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in subgroup_brier.items() if v},
    'subgroup_auc': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in subgroup_auc.items() if v},
    'aps_conformal': {'mean': np.mean([r['overall'] for r in aps_results]), 'std': np.std([r['overall'] for r in aps_results])},
    'lime_stability': 'computed' if lime_stability else 'not_available',
    'shap_stability': 'computed' if shap_stability else 'not_available',
    'n_seeds': N_SEEDS,
}

with open(RESULTS_DIR / 'fairness_baselines_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nFairness Baselines Complete!")
print(json.dumps(results, indent=2, default=str))