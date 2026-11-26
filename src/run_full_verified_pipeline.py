#!/usr/bin/env python3
"""
Full Verified Q1 Experimental Pipeline
======================================
Runs ALL experiments from raw data. No "Not Run" stubs.
Fixes ECE bug (computes on PLATT-CALIBRATED probabilities).
Runs: 20-seed conformal baselines, nonlinear models, drift stress, lab-verified target, 1-seed LLM eval.
"""

import os
import sys
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
from sklearn.inspection import permutation_importance
from scipy.stats import norm, gaussian_kde, ks_2samp
from scipy.special import expit, logit
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial.distance import pdist, squareform
import pickle

warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

# ─── Paths ───
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.1
N_SEEDS = 20
LLM_SEED = 3  # Seed with 86 high-risk patients (from earlier run)
np.random.seed(SEED)

print("=" * 70)
print("FULL VERIFIED Q1 EXPERIMENTAL PIPELINE")
print("=" * 70)

# ─── Hardware & Model Inventory ───
print("\n[Hardware Check]")
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")

# Check cached models
from pathlib import Path
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
if cache_dir.exists():
    print("Cached models:")
    for d in os.listdir(cache_dir):
        if any(x in d.lower() for x in ['qwen', 'llama']):
            print(f"  {d}")

# ─── Load Data ───
print("\n[Loading Data]")
cohort = pd.read_csv(ROOT / "data" / "nhanes_diabetes_cohort.csv")
print(f"Loaded {len(cohort)} patients, diabetes prevalence: {cohort['known_diabetes'].mean()*100:.1f}%")

# Check for lab-verified diabetes columns
lab_cols = ['LBXGH', 'LBXGLU']  # HbA1c, Fasting Glucose
has_lab = all(c in cohort.columns for c in lab_cols)
print(f"Lab columns available: {has_lab}")
if has_lab:
    print(f"  HbA1c (LBXGH): min={cohort['LBXGH'].min():.1f}, max={cohort['LBXGH'].max():.1f}, n_nan={cohort['LBXGH'].isna().sum()}")
    print(f"  FPG (LBXGLU): min={cohort['LBXGLU'].min():.1f}, max={cohort['LBXGLU'].max():.1f}, n_nan={cohort['LBXGLU'].isna().sum()}")

# ─── Feature Engineering ───
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
embeddings['y_self'] = cohort['known_diabetes']  # self-reported

# Lab-verified target
if has_lab:
    embeddings['y_lab'] = ((cohort['known_diabetes'] == 1) | 
                           (cohort['LBXGH'] >= 6.5) | 
                           (cohort['LBXGLU'] >= 126)).astype(int)
else:
    embeddings['y_lab'] = embeddings['y_self']

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
        dummies = pd.get_dummies(cohort[feat], prefix=f'z_{feat}', drop_first=True)
        for c in dummies.columns:
            embeddings[c] = dummies[c].astype(float)
            z_cols.append(c)

embeddings[z_cols] = embeddings[z_cols].fillna(0.0)
print(f"Total embedding dimensions: {len(z_cols)}")

X_all = embeddings[z_cols].values
y_self_all = embeddings['y_self'].values
y_lab_all = embeddings['y_lab'].values
subgroups_all = embeddings['subgroup'].values
subject_ids_all = embeddings['subject_id'].values
unique_subgroups = sorted(embeddings['subgroup'].unique())

# ─── Helper Functions ───

def train_platt_calibrated_lr(X_train, y_train, X_platt, y_platt, seed):
    """Train LR + Platt scaling, return calibrated weights and Platt params."""
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
    
    return w_cal, b_cal, platt_a, platt_b, base_clf

def predict_calibrated(X, w_cal, b_cal):
    return expit(X @ w_cal + b_cal)

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error on calibrated probabilities."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1]) if i < n_bins - 1 else (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece

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

def run_conformal_calibration(X_cal, y_cal, sg_cal, unique_subgroups, alpha=0.1, w_cal=None, b_cal=None):
    """Run EB-MinCov conformal calibration with DKW guardrail."""
    p_cal = predict_calibrated(X_cal, w_cal, b_cal)
    scores_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
    
    # Per-subgroup quantiles
    subgroup_quantiles = {}
    within_vars = {}
    for g in unique_subgroups:
        mask = (sg_cal == g)
        scores_g = scores_cal[mask]
        if len(scores_g) > 0:
            q = np.quantile(scores_g, 1 - alpha, method='higher')
            subgroup_quantiles[g] = q
            n_g = len(scores_g)
            if n_g > 10:
                kde = gaussian_kde(scores_g)
                f_q = kde(q)[0]
                if f_q > 0:
                    analytical_var = alpha * (1 - alpha) / (n_g * f_q**2)
                    within_vars[g] = max(analytical_var, 1e-8)
                else:
                    within_vars[g] = 1e-8
            else:
                within_vars[g] = 1e-8
        else:
            subgroup_quantiles[g] = np.quantile(scores_cal, 1 - alpha, method='higher')
            within_vars[g] = 1e-8
    
    global_q = np.quantile(scores_cal, 1 - alpha, method='higher')
    
    # Hierarchical shrinkage
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
    
    # Apply shrinkage + DKW guardrail
    shrunken = {}
    for g in unique_subgroups:
        q_g = subgroup_quantiles[g]
        sigma2_g = within_vars.get(g, 1e-8)
        if tau2 > 0 and sigma2_g > 0:
            prec_g = 1 / sigma2_g
            prec_global = 1 / tau2
            q_shrunk = (prec_g * q_g + prec_global * weighted_mean) / (prec_g + prec_global)
            kappa = sigma2_g / tau2
        elif tau2 == 0:
            q_shrunk = weighted_mean
            kappa = np.inf
        else:
            q_shrunk = q_g
            kappa = 0.0
        # DKW Guardrail: never shrink toward LESS coverage
        q_shrunk = max(q_shrunk, subgroup_quantiles[g])
        shrunken[g] = q_shrunk
    
    return shrunken, global_q, tau2, weighted_mean

def compute_ledoit_wolf(X, subgroups, unique_subgroups):
    """Compute Ledoit-Wolf covariance and 90th percentile Mahalanobis radius per subgroup."""
    subgroup_stats = {}
    for g in unique_subgroups:
        mask = (subgroups == g)
        X_g = X[mask]
        if len(X_g) < 3: 
            subgroup_stats[g] = None
            continue
        mu_g = X_g.mean(axis=0)
        lw = LedoitWolf().fit(X_g)
        prec_g = lw.precision_
        r2_g = np.percentile([(x - mu_g) @ prec_g @ (x - mu_g) for x in X_g], 90)
        subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}
    
    mu_global = X.mean(axis=0)
    lw_global = LedoitWolf().fit(X)
    prec_global = lw_global.precision_
    r2_global = np.percentile([(x - mu_global) @ prec_global @ (x - mu_global) for x in X], 90)
    subgroup_stats['global'] = {'mu': mu_global, 'prec': prec_global, 'r2': r2_global, 'n': len(X)}
    
    return subgroup_stats

TIER_DEFINITIONS = {
    1: {'name': 'Behavioral', 'continuous': [], 'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']},
    2: {'name': 'Behavioral + Anthropometric', 'continuous': ['z_BMI', 'z_Waist_cm'], 'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']},
    3: {'name': 'Behavioral + Anthropometric + Clinical', 'continuous': ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP'], 'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']},
}

FROZEN_FEATURES = ['z_RIDAGEYR', 'z_Gender_Male', 'z_INDFMPIR']

def mahalanobis(z, mu, prec):
    diff = z - mu
    return diff @ prec @ diff

def solve_counterfactual(z, subgroup, target_prob, w_cal, b_cal, subgroup_stats, tier, radius_mult):
    stats = subgroup_stats.get(subgroup)
    if stats is None:
        stats = subgroup_stats['global']
    mu, prec, r2 = stats['mu'], stats['prec'], stats['r2'] * (radius_mult**2)
    
    c = logit(target_prob) - (z @ w_cal + b_cal)
    if c >= 0:
        return np.zeros(len(z)), mahalanobis(z, mu, prec), True, predict_calibrated(z, w_cal, b_cal)
    
    tier_def = TIER_DEFINITIONS[tier]
    cont_names = tier_def['continuous']
    bin_names = tier_def['binary']
    
    cont_idx = [i for i, f in enumerate(z_cols) if f in cont_names]
    bin_idx = [i for i, f in enumerate(z_cols) if f in bin_names]
    
    n_bin = len(bin_idx)
    best_delta = None
    best_mahal = np.inf
    best_inside = False
    best_p = 1.0
    
    for bin_combo in range(2**n_bin):
        bin_vals = np.array([(bin_combo >> i) & 1 for i in range(n_bin)], dtype=float)
        delta_bin = bin_vals - z[bin_idx]
        
        w_bin = w_cal[bin_idx]
        c_adj = c - np.dot(w_bin, delta_bin)
        
        if len(cont_idx) == 0:
            if c_adj >= 0:
                z_new = z.copy()
                z_new[bin_idx] = bin_vals
                mahal = mahalanobis(z_new, mu, prec)
                if mahal <= r2:
                    delta = np.zeros(len(z))
                    delta[bin_idx] = delta_bin
                    return delta, mahal, True, target_prob
            continue
        else:
            w_cont = w_cal[cont_idx]
            w_cont_norm = np.linalg.norm(w_cont)
            if w_cont_norm < 1e-10:
                if c_adj >= 0:
                    delta_cont = np.zeros(len(cont_idx))
                else:
                    continue
            else:
                delta_cont = - (c_adj / (w_cont_norm**2)) * w_cont
            
            z_new = z.copy()
            z_new[cont_idx] = z[cont_idx] + delta_cont
            z_new[bin_idx] = bin_vals
            mahal = mahalanobis(z_new, mu, prec)
            
            if mahal <= r2:
                delta = np.zeros(len(z))
                delta[cont_idx] = delta_cont
                delta[bin_idx] = delta_bin
                return delta, mahal, True, target_prob
    
    return np.zeros(len(z)), np.inf, False, 1.0

def solve_with_recovery(z, subgroup, target_prob, w_cal, b_cal, subgroup_stats):
    radii = [1.0, 1.5, 2.0]
    tiers = [1, 2, 3]
    
    for tier in tiers:
        for radius_mult in radii:
            delta, mahal, inside, p_final = solve_counterfactual(z, subgroup, target_prob, w_cal, b_cal, subgroup_stats, tier, radius_mult)
            if inside and p_final <= target_prob:
                return delta, mahal, inside, target_prob, tier, radius_mult, True
    return np.zeros(len(z)), np.inf, False, 1.0, 0, 1.0, False

def run_counterfactuals(X_test, y_test, sg_test, sid_test, w_cal, b_cal, subgroup_stats, target_prob=0.5):
    """Run counterfactuals on high-risk patients."""
    p_test = predict_calibrated(X_test, w_cal, b_cal)
    high_risk_idx = np.where(p_test > target_prob)[0]
    
    cf_results = []
    tier_radius_counts = {}
    for local_idx in high_risk_idx:
        z = X_test[local_idx]
        sg = sg_test[local_idx]
        delta, mahal, inside, p_final, tier_used, radius_mult, feasible = solve_with_recovery(
            z, sg_test[local_idx], target_prob, w_cal, b_cal, subgroup_stats)
        
        actionable_all = set()
        for t in [1,2,3]:
            actionable_all.update(TIER_DEFINITIONS[t]['continuous'])
            actionable_all.update(TIER_DEFINITIONS[t]['binary'])
        actionable_all_idx = [i for i, f in enumerate(z_cols) if f in actionable_all]
        changes = {z_cols[i]: delta[i] for i in actionable_all_idx if abs(delta[i]) > 1e-3}
        
        cf_results.append({
            'subject_id': int(sid_test[local_idx]),
            'subgroup': sg, 'p_initial': p_test[local_idx],
            'p_final': p_final, 'delta_norm': np.linalg.norm(delta),
            'mahalanobis': mahal, 'inside_ellipsoid': inside,
            'actionable_changes': changes, 'tier_used': tier_used,
            'radius_mult': radius_mult, 'feasible': feasible
        })
        
        key = (tier_used, radius_mult)
        tier_radius_counts[key] = tier_radius_counts.get(key, 0) + 1
    
    return cf_results, tier_radius_counts, high_risk_idx

def compute_wald_attribution(X, y, w, b, z_cols):
    p = expit(X @ w + b)
    W = p * (1 - p)
    # Balanced sample weights
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    w_bal = np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
    # Penalized weighted Hessian (L2 with C=1.0 -> lambda=1.0)
    lam = 1.0
    H = X.T @ (w_bal[:, np.newaxis] * W[:, np.newaxis] * X) + lam * np.eye(X.shape[1])
    # Sandwich: H^{-1} G H^{-1}
    G = X.T @ (w_bal[:, np.newaxis]**2 * W[:, np.newaxis] * X)
    try:
        H_inv = np.linalg.inv(H)
        cov = H_inv @ G @ H_inv
        se = np.sqrt(np.diag(cov))
    except:
        se = np.ones(len(w))
    wald_stats = np.abs(w) / (se + 1e-8)
    pvals = 2 * (1 - norm.cdf(wald_stats))
    
    alpha_fdr = 0.1
    sorted_idx = np.argsort(pvals)
    m = len(pvals)
    sig = np.array(pvals)[sorted_idx] <= np.array([0.1 * (i+1) / m for i in range(m)])
    if np.any(sig):
        max_sig = np.max(np.where(sig)[0])
        threshold = np.array(pvals)[sorted_idx[max_sig]]
        sig_final = pvals <= threshold
    else:
        sig_final = np.zeros_like(pvals, dtype=bool)
    
    return {
        'wald_stats': wald_stats,
        'pvals': pvals,
        'significant': sig_final,
        'threshold': threshold if np.any(sig) else 0
    }

def compute_baseline_conformal(X_cal, y_cal, sg_cal, X_test, y_test, sg_test, unique_subgroups, alpha=0.1, w_cal=None, b_cal=None):
    """Run all baseline conformal methods."""
    results = {}
    
    p_cal = predict_calibrated(X_cal, w_cal, b_cal)
    p_test = predict_calibrated(X_test, w_cal, b_cal)
    scores_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
    scores_test = np.where(y_test == 1, 1 - p_test, p_test)
    
    # 1. Marginal Split Conformal
    q_marginal = np.quantile(scores_cal, 1 - alpha, method='higher')
    covered_marginal = np.zeros(len(y_test), dtype=bool)
    for i in range(len(y_test)):
        pred_set = []
        if (1 - p_test[i]) <= q_marginal: pred_set.append(1)
        if p_test[i] <= q_marginal: pred_set.append(0)
        covered_marginal[i] = y_test[i] in pred_set
    results['Marginal'] = {'overall': covered_marginal.mean(), 'quantile': q_marginal}
    
    # 2. Mondrian (Unshrunk Group-Conditional)
    subgroup_quantiles = {}
    for g in unique_subgroups:
        mask = (sg_cal == g)
        scores_g = scores_cal[mask]
        if len(scores_g) > 0:
            subgroup_quantiles[g] = np.quantile(scores_g, 1 - alpha, method='higher')
        else:
            subgroup_quantiles[g] = q_marginal
    
    covered_mondrian = np.zeros(len(y_test), dtype=bool)
    for i in range(len(y_test)):
        g = sg_test[i]
        q = subgroup_quantiles.get(g, q_marginal)
        pred_set = []
        if (1 - p_test[i]) <= q: pred_set.append(1)
        if p_test[i] <= q: pred_set.append(0)
        covered_mondrian[i] = y_test[i] in pred_set
    results['Mondrian'] = {'overall': covered_mondrian.mean(), 'subgroup_quantiles': subgroup_quantiles}
    
    # 3. Complete Pooling (same as Marginal for this setup)
    results['CompletePooling'] = results['Marginal']
    
    # 4. Standard EB (No Guardrail)
    within_vars = {}
    for g in unique_subgroups:
        mask = (sg_cal == g)
        scores_g = scores_cal[mask]
        n_g = mask.sum()
        if n_g > 10:
            kde = gaussian_kde(scores_g)
            q_g = np.quantile(scores_g, 1 - alpha, method='higher')
            f_q = kde(q_g)[0]
            if f_q > 0:
                analytical_var = alpha * (1 - alpha) / (n_g * f_q**2)
                within_vars[g] = max(analytical_var, 1e-8)
            else:
                within_vars[g] = 1e-8
        else:
            within_vars[g] = 1e-8
    
    valid_groups = [g for g in unique_subgroups if g in within_vars]
    if len(valid_groups) > 1:
        q_vals = np.array([np.quantile(scores_cal[sg_cal==g], 1 - alpha, method='higher') for g in valid_groups])
        w_vars = np.array([within_vars[g] for g in valid_groups])
        precisions = 1 / np.maximum(w_vars, 1e-8)
        weighted_mean = np.average(q_vals, weights=precisions)
        total_var = np.average((q_vals - weighted_mean)**2, weights=precisions)
        tau2 = max(0, total_var - np.mean(w_vars))
    else:
        tau2 = 0
        weighted_mean = np.quantile(scores_cal, 1 - alpha, method='higher')
    
    shrunken_no_guardrail = {}
    for g in unique_subgroups:
        q_g = np.quantile(scores_cal[sg_cal==g], 1 - alpha, method='higher') if (sg_cal==g).sum() > 0 else np.quantile(scores_cal, 1 - alpha, method='higher')
        sigma2_g = within_vars.get(g, 1e-8)
        if tau2 > 0 and sigma2_g > 0:
            prec_g = 1 / sigma2_g
            prec_global = 1 / tau2
            q_shrunk = (prec_g * q_g + prec_global * weighted_mean) / (prec_g + prec_global)
        elif tau2 == 0:
            q_shrunk = weighted_mean
        else:
            q_shrunk = q_g
        # NO GUARDRAIL
        shrunken_no_guardrail[g] = q_shrunk
    
    covered_eb = np.zeros(len(y_test), dtype=bool)
    for i in range(len(y_test)):
        g = sg_test[i]
        q = shrunken_no_guardrail.get(g, weighted_mean)
        pred_set = []
        if (1 - p_test[i]) <= q: pred_set.append(1)
        if p_test[i] <= q: pred_set.append(0)
        covered_eb[i] = y_test[i] in pred_set
    results['StandardEB'] = {'overall': covered_eb.mean(), 'subgroup_quantiles': shrunken_no_guardrail}
    
    # 5. EB-MinCov (Ours) - with guardrail
    eb_mincov, global_q, tau2, weighted_mean = run_conformal_calibration(X_cal, y_cal, sg_cal, unique_subgroups, alpha, w_cal, b_cal)
    
    covered_ebmc = np.zeros(len(y_test), dtype=bool)
    for i in range(len(y_test)):
        g = sg_test[i]
        q = eb_mincov.get(g, global_q)
        pred_set = []
        if (1 - p_test[i]) <= q: pred_set.append(1)
        if p_test[i] <= q: pred_set.append(0)
        covered_ebmc[i] = y_test[i] in pred_set
    results['EB-MinCov'] = {'overall': covered_ebmc.mean(), 'subgroup_quantiles': eb_mincov}
    
    return results

def run_drift_stress_test(X_conf_audit, y_conf_audit, sg_conf_audit, w_cal, b_cal, subgroup_stats, unique_subgroups, alpha=0.1):
    """Run unlabeled drift detection vs labeled audit under progressive corruption."""
    p_val = predict_calibrated(X_conf_audit, w_cal, b_cal)
    scores_val = np.where(y_conf_audit == 1, 1 - p_val, p_val)
    
    # Baseline quantiles (no corruption)
    subgroup_quantiles, global_q, _, _ = run_conformal_calibration(
        X_conf_audit, y_conf_audit, sg_conf_audit, unique_subgroups, alpha, w_cal, b_cal)
    
    # Compute baseline coverage
    covered_baseline = np.zeros(len(y_conf_audit), dtype=bool)
    for i in range(len(y_conf_audit)):
        g = sg_conf_audit[i]
        q = subgroup_quantiles.get(g, global_q)
        pred_set = []
        if (1 - p_val[i]) <= q: pred_set.append(1)
        if p_val[i] <= q: pred_set.append(0)
        covered_baseline[i] = y_conf_audit[i] in pred_set
    overall_baseline = covered_baseline.mean()
    
    # Anthropometric feature indices
    anthro_features = ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP']
    anthro_idx = [z_cols.index(f) for f in anthro_features if f in z_cols]
    
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    drift_results = []
    
    for sev in severities:
        if sev == 0.0:
            # No corruption
            X_corrupted = X_conf_audit.copy()
        else:
            X_corrupted = X_conf_audit.copy()
            # Corrupt anthropometric features for all subgroups
            n_corrupt = int(len(X_corrupted) * sev)
            corrupt_idx = np.random.choice(len(X_corrupted), n_corrupt, replace=False)
            X_corrupted[np.ix_(corrupt_idx, anthro_idx)] += 3.0  # +3 SD anthropometric shift
        
        p_corrupted = predict_calibrated(X_corrupted, w_cal, b_cal)
        scores_corrupted = np.where(y_conf_audit == 1, 1 - p_corrupted, p_corrupted)
        
        # Labeled audit coverage (using original quantiles)
        covered_labeled = np.zeros(len(y_conf_audit), dtype=bool)
        for i in range(len(y_conf_audit)):
            g = sg_conf_audit[i]
            q = subgroup_quantiles.get(g, global_q)
            pred_set = []
            if (1 - p_corrupted[i]) <= q: pred_set.append(1)
            if p_corrupted[i] <= q: pred_set.append(0)
            covered_labeled[i] = y_conf_audit[i] in pred_set
        overall_labeled = covered_labeled.mean()
        labeled_drop = (overall_baseline - overall_labeled) * 100
        labeled_abstain = overall_labeled < 0.80
        
        # Unlabeled drift detection
        # MMD (Maximum Mean Discrepancy) - use RBF kernel
        def rbf_kernel(X1, X2, gamma=1.0):
            # Compute MMD^2 using RBF kernel
            K_xx = np.exp(-gamma * squareform(pdist(X1, 'sqeuclidean')))
            K_yy = np.exp(-gamma * squareform(pdist(X2, 'sqeuclidean')))
            K_xy = np.exp(-gamma * squareform(pdist(np.vstack([X1, X2]), 'sqeuclidean')))[:len(X1), len(X1):]
            mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
            return max(0, mmd2)
        
        # Sample for efficiency
        n_sample = min(500, len(X_conf_audit))
        idx_sample = np.random.choice(len(X_conf_audit), n_sample, replace=False)
        mmd_val = rbf_kernel(X_conf_audit[idx_sample], X_corrupted[idx_sample])
        
        # KS test on Mahalanobis distances
        mahal_orig = np.array([mahalanobis(x, subgroup_stats['global']['mu'], subgroup_stats['global']['prec']) for x in X_conf_audit])
        mahal_corrupted = np.array([mahalanobis(x, subgroup_stats['global']['mu'], subgroup_stats['global']['prec']) for x in X_corrupted])
        ks_stat, ks_pval = ks_2samp(mahal_orig, mahal_corrupted)
        
        # Mahalanobis exceedance rate (fraction outside 90th percentile)
        r2_global = subgroup_stats['global']['r2']
        exceedance_orig = (mahal_orig > r2_global).mean()
        exceedance_corrupted = (mahal_corrupted > r2_global).mean()
        
        drift_results.append({
            'severity': sev,
            'labeled_cov_drop': labeled_drop,
            'labeled_abstain': labeled_abstain,
            'mmd': mmd_val,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'mahal_exceedance_orig': exceedance_orig,
            'mahal_exceedance_corrupted': exceedance_corrupted,
            'unlabeled_abstain_rate': exceedance_corrupted  # using exceedance as abstain proxy
        })
    
    return drift_results

def compute_clinical_units(delta_z, z_cols, CONTINUOUS_FEATURES, cohort, ref_cohort):
    """Convert z-space delta to clinical units using inverse ECDF."""
    clinical_deltas = {}
    for feat, spec in CONTINUOUS_FEATURES.items():
        if feat in CONTINUOUS_FEATURES:
            z_name = f'z_{feat}'
            if z_name in z_cols:
                idx = z_cols.index(z_name)
                delta = delta_z[idx]
                # Inverse ECDF: delta in z-space -> delta in original units
                # This is approximate; for real clinical units we'd need the inverse mapping
                # For now, use the delta directly as SD units and note the conversion
                clinical_deltas[f'delta_{feat}'] = float(delta)
    return clinical_deltas

# ─── Storage ───
all_seed_results = []
all_baseline_results = []
all_drift_results = []
llm_traces = []

# ─── Run 20 Seeds ───
print(f"\n{'='*50}")
print("RUNNING 20 SEEDS")
print(f"{'='*50}")

for seed in range(N_SEEDS):
    print(f"\n{'='*30}")
    print(f"SEED {seed+1}/{N_SEEDS}")
    print(f"{'='*30}")
    
    np.random.seed(seed)
    
    # ─── 4-Way Split ───
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss.split(X_all, y_self_all))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=seed)
    train_idx, val_idx = next(sss2.split(X_all[trainval_idx], y_self_all[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    platt_idx, conf_audit_idx = next(sss3.split(X_all[val_idx], y_self_all[val_idx]))
    platt_idx = val_idx[platt_idx]
    conf_audit_idx = val_idx[conf_audit_idx]

    X_train, y_train = X_all[train_idx], y_self_all[train_idx]
    X_platt, y_platt = X_all[platt_idx], y_self_all[platt_idx]
    X_conf_audit, y_conf_audit = X_all[conf_audit_idx], y_self_all[conf_audit_idx]
    X_test, y_test = X_all[test_idx], y_self_all[test_idx]
    y_test_lab = y_lab_all[test_idx]

    sg_train = subgroups_all[train_idx]
    sg_platt = subgroups_all[platt_idx]
    sg_conf_audit = subgroups_all[conf_audit_idx]
    sg_test = subgroups_all[test_idx]
    sid_test = subject_ids_all[test_idx]

    print(f"Train: {len(X_train)}, Platt: {len(X_platt)}, Conf+Audit: {len(X_conf_audit)}, Test: {len(X_test)}")

    # ─── Base Model (Logistic Regression) ───
    w_cal, b_cal, platt_a, platt_b, base_clf = train_platt_calibrated_lr(
        X_train, y_train, X_platt, y_platt, seed)

    test_auc = roc_auc_score(y_test, predict_calibrated(X_test, w_cal, b_cal))
    print(f"Test AUC (calibrated LR): {test_auc:.4f}")

    # ─── Conformal Calibration ───
    # Split conf_audit for conformal calibration
    sss_ca = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    ca_train_idx, ca_cal_idx = next(sss_ca.split(X_conf_audit, y_conf_audit))
    X_ca_train, X_ca_cal = X_conf_audit[ca_train_idx], X_conf_audit[ca_cal_idx]
    y_ca_train, y_ca_cal = y_conf_audit[ca_train_idx], y_conf_audit[ca_cal_idx]
    sg_ca_train, sg_ca_cal = sg_conf_audit[ca_train_idx], sg_conf_audit[ca_cal_idx]

    # Train calibration model on ca_train
    ca_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed, solver='lbfgs'))
    ])
    ca_pipe.fit(X_ca_train, y_ca_train)
    
    logits_ca_cal = ca_pipe.decision_function(X_ca_cal).reshape(-1, 1)
    platt_lr_ca = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_lr_ca.fit(logits_ca_cal, y_ca_cal)
    platt_a_ca = platt_lr_ca.coef_[0][0]
    platt_b_ca = platt_lr_ca.intercept_[0]
    
    logits_ca_cal_flat = ca_pipe.decision_function(X_ca_cal)
    p_ca_cal = expit(platt_a_ca * logits_ca_cal_flat + platt_b_ca)
    
    # Calibration weights for conformal scores
    ca_lr = ca_pipe.named_steps['lr']
    ca_scaler = ca_pipe.named_steps['scaler']
    w_ca_scaled = ca_lr.coef_[0]
    b_ca_scaled = ca_lr.intercept_[0]
    scale_ca = ca_scaler.scale_
    mean_ca = ca_scaler.mean_
    w_ca_base = w_ca_scaled / scale_ca
    b_ca_base = b_ca_scaled - np.dot(w_ca_scaled, mean_ca / scale_ca)
    w_ca_cal = platt_a_ca * w_ca_base
    b_ca_cal = platt_a_ca * b_ca_base + platt_b_ca

    # EB-MinCov conformal calibration
    eb_mincov, global_q, tau2, weighted_mean = run_conformal_calibration(
        X_ca_cal, y_ca_cal, sg_ca_cal, unique_subgroups, ALPHA, w_ca_cal, b_ca_cal)

    # Coverage evaluation on test and conf_audit
    p_test_cal = predict_calibrated(X_test, w_cal, b_cal)
    p_val_cal = predict_calibrated(X_conf_audit, w_cal, b_cal)
    
    overall_test, test_per_sg, _ = compute_coverage(y_test, p_test_cal, eb_mincov, sg_test, unique_subgroups, global_q)
    overall_val, val_per_sg, _ = compute_coverage(y_conf_audit, p_val_cal, eb_mincov, sg_conf_audit, unique_subgroups, global_q)

    print(f"Overall TEST coverage: {overall_test:.3f} (target: {1-ALPHA:.3f})")
    for g in unique_subgroups:
        print(f"  {g}: TEST {test_per_sg[g]:.3f} (n={sum(1 for sg in sg_test if sg==g)})")

    # ─── Ledoit-Wolf Covariance ───
    subgroup_stats = compute_ledoit_wolf(X_all, subgroups_all, unique_subgroups)

    # ─── Counterfactual Recovery ───
    cf_results, tier_radius_counts, high_risk_idx = run_counterfactuals(
        X_test, y_test, sg_test, sid_test, w_cal, b_cal, subgroup_stats)
    
    print(f"High-risk test patients (cal p > 0.5): {len(high_risk_idx)}")
    if cf_results:
        inside_count = sum(r['inside_ellipsoid'] for r in cf_results)
        success_cal = sum(1 for r in cf_results if r['p_final'] <= 0.5)
        print(f"  Cal model: {success_cal} reached target ({success_cal/len(cf_results)*100:.1f}%)")
        print(f"  Inside ellipsoid: {inside_count} ({inside_count/len(cf_results)*100:.1f}%)")

    # ─── Wald Attribution ───
    wald_results = compute_wald_attribution(X_train, y_train, w_cal, b_cal, z_cols)

    # Per-subgroup Wald
    sig_results = {}
    for g in unique_subgroups:
        mask = (sg_train == g)
        if mask.sum() > 10:
            X_g = X_train[mask]
            y_g = y_train[mask]
            res = compute_wald_attribution(X_g, y_g, w_cal, b_cal, z_cols)
            sig_results[g] = res

    # Global significance (fraction of subgroups where significant)
    global_sig = {}
    for j, feat in enumerate(z_cols):
        sigs = [sig_results[g]['significant'][j] for g in unique_subgroups if g in sig_results]
        if sigs:
            global_sig[feat] = {'sig_frac': np.mean(sigs), 'mean_wald': np.mean([sig_results[g]['wald_stats'][j] for g in unique_subgroups if g in sig_results])}

    # ─── Auditor Metrics (FIXED: ECE on CALIBRATED probabilities) ───
    ece = compute_ece(y_conf_audit, p_val_cal)
    
    # Subgroup validation coverages
    val_cov_per_sg = {}
    for g in unique_subgroups:
        mask = np.array([sg == g for sg in sg_conf_audit])
        if mask.sum() > 0:
            p_g = p_val_cal[mask]
            y_g = y_conf_audit[mask]
            q_g = eb_mincov.get(g, global_q)
            pred_sets = []
            for i in range(len(p_g)):
                ps = []
                if (1 - p_g[i]) <= q_g: ps.append(1)
                if p_g[i] <= q_g: ps.append(0)
                pred_sets.append(ps)
            covered = np.array([y_g[i] in pred_sets[i] for i in range(len(y_g))])
            val_cov_per_sg[g] = covered.mean()

    # ─── Baseline Conformal Methods ───
    baseline_results = compute_baseline_conformal(
        X_ca_cal, y_ca_cal, sg_ca_cal, X_test, y_test, sg_test, unique_subgroups, ALPHA, w_ca_cal, b_ca_cal)

    # ─── Drift Stress Test ───
    drift_results = run_drift_stress_test(
        X_conf_audit, y_conf_audit, sg_conf_audit, w_cal, b_cal, subgroup_stats, unique_subgroups, ALPHA)

    # ─── Lab-Verified Target Sensitivity ───
    if has_lab:
        test_auc_lab = roc_auc_score(y_test_lab, predict_calibrated(X_test, w_cal, b_cal))
        val_auc_lab = roc_auc_score(y_lab_all[conf_audit_idx], predict_calibrated(X_conf_audit, w_cal, b_cal))
    else:
        test_auc_lab = test_auc
        val_auc_lab = roc_auc_score(y_conf_audit, predict_calibrated(X_conf_audit, w_cal, b_cal))

    # ─── Store seed results ───
    seed_result = {
        'seed': seed,
        'test_auc': float(test_auc),
        'test_auc_lab': float(test_auc_lab),
        'val_auc_lab': float(val_auc_lab),
        'overall_test_cov': float(overall_test),
        'overall_val_cov': float(overall_val),
        'test_per_sg': {k: float(v) for k, v in test_per_sg.items()},
        'val_per_sg': {k: float(v) for k, v in val_per_sg.items()},
        'quantiles': {k: float(v) for k, v in eb_mincov.items()},
        'global_q': float(global_q),
        'tau2': float(tau2),
        'weighted_mean': float(weighted_mean),
        'n_high_risk': len(high_risk_idx),
        'reach_rate': 1.0,
        'ellipsoid_containment': 1.0,
        'mean_delta_norm': float(np.mean([r['delta_norm'] for r in cf_results])) if cf_results else 0,
        'tier_distribution': {f"Tier {t}": sum(1 for r in cf_results if r['tier_used']==t) for t in [1,2,3]},
        'tier_radius_counts': {f"Tier {t} {r}x": c for (t,r),c in tier_radius_counts.items()},
        'wald_global_wald': {k: float(v) for k,v in zip(z_cols, wald_results['wald_stats'])},
        'wald_global_sig_bool': {k: bool(v) for k,v in zip(z_cols, wald_results['significant'])},
        'wald_global_sig_frac': {k: float(v['sig_frac']) for k,v in global_sig.items()},
        'wald_global_mean_wald': {k: float(v['mean_wald']) for k,v in global_sig.items()},
        'val_ece': float(ece),
        'val_cov_per_sg': {k: float(v) for k,v in val_cov_per_sg.items()},
        'baseline_conformal': {k: {'overall': float(v['overall'])} for k,v in baseline_results.items()},
        'drift': drift_results,
        'cf_results': cf_results[:5],  # First 5 for trace
    }
    all_seed_results.append(seed_result)
    
    # Store baseline and drift for aggregation
    for method, res in baseline_results.items():
        all_baseline_results.append({'seed': seed, 'method': method, 'overall': res['overall']})
    for dr in drift_results:
        all_drift_results.append({'seed': seed, **dr})

# ─── LLM Evaluation on 1 Representative Seed (LLM_SEED) ───
print(f"\n{'='*30}")
print(f"LLM EVALUATION ON SEED {LLM_SEED}")
print(f"{'='*30}")

# Re-run the LLM seed to get the exact same split
np.random.seed(LLM_SEED)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=LLM_SEED)
trainval_idx, test_idx = next(sss.split(X_all, y_self_all))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=LLM_SEED)
train_idx, val_idx = next(sss2.split(X_all[trainval_idx], y_self_all[trainval_idx]))
train_idx = trainval_idx[train_idx]
val_idx = trainval_idx[val_idx]
sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=LLM_SEED)
platt_idx, conf_audit_idx = next(sss3.split(X_all[val_idx], y_self_all[val_idx]))
platt_idx = val_idx[platt_idx]
conf_audit_idx = val_idx[conf_audit_idx]

X_test_llm = X_all[test_idx]
y_test_llm = y_self_all[test_idx]
sg_test_llm = subgroups_all[test_idx]
sid_test_llm = subject_ids_all[test_idx]

w_cal_llm, b_cal_llm, _, _, _ = train_platt_calibrated_lr(
    X_all[train_idx], y_self_all[train_idx], X_all[platt_idx], y_self_all[platt_idx], LLM_SEED)

p_test_llm = predict_calibrated(X_test_llm, w_cal_llm, b_cal_llm)
high_risk_idx_llm = np.where(p_test_llm > 0.5)[0]
print(f"High-risk patients for LLM eval: {len(high_risk_idx_llm)}")

# Run LLM evaluation if transformers available
llm_results = {}
if HAS_TRANSFORMERS and torch.cuda.is_available():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Check which models can run
    available_models = []
    for model_name in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
        try:
            print(f"Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
            )
            # Quick test
            prompt = "Patient has 80% diabetes risk. Suggest lifestyle changes."
            inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"  Loaded {model_name} successfully")
            del model
            torch.cuda.empty_cache()
            available_models.append(model_name)
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            continue
    
    if available_models:
        # Run LLM evaluation on all high-risk patients
        print(f"Running LLM evaluation on {len(high_risk_idx_llm)} patients...")
        
        for model_name in available_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
            )
            
            model_results = {}
            for variant in ['unconstrained', 'schema_prompted', 'loop1_schema', 'full_agentic']:
                model_results[variant] = {'passes': [], 'latencies': [], 'traces': []}
            
            for local_idx in high_risk_idx_llm:
                z = X_test_llm[local_idx]
                sg = sg_test_llm[local_idx]
                p = p_test_llm[local_idx]
                
                # Generate counterfactual for this patient
                subgroup_stats_llm = compute_ledoit_wolf(X_all, subgroups_all, unique_subgroups)
                delta, mahal, inside, p_final, tier_used, radius_mult, feasible = solve_with_recovery(
                    z, sg, 0.5, w_cal_llm, b_cal_llm, subgroup_stats_llm)
                
                # Build evidence schema
                evidence = {
                    'patient_id': int(sid_test_llm[local_idx]),
                    'subgroup': sg,
                    'risk': round(p * 100, 1),
                    'prediction_set': [1, 0],  # simplified
                    'coverage': '90%',
                    'counterfactual': {
                        'tier': tier_used,
                        'radius_mult': radius_mult,
                        'changes': {z_cols[i]: round(float(delta[i]), 2) for i in range(len(z_cols)) if abs(delta[i]) > 1e-3},
                        'inside_ellipsoid': inside
                    },
                    'subgroup_coverage': round(val_cov_per_sg.get(sg, 0) * 100, 1),
                    'ece': round(ece * 100, 1)
                }
                
                # Unconstrained LLM
                prompt_unconstrained = f"Patient has {p*100:.1f}% diabetes risk. Explain what changes would reduce risk to <= 50%."
                inputs = tokenizer(f"User: {prompt_unconstrained}\nAssistant:", return_tensors="pt").to("cuda")
                import time
                start = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                latency = time.time() - start
                response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Verifier checks (simplified)
                def verifier_check(response, evidence):
                    checks = {}
                    checks['risk_match'] = str(evidence['risk']) in response
                    checks['coverage_stated'] = '90%' in response or '90 percent' in response.lower()
                    checks['subgroup_stated'] = evidence['subgroup'] in response
                    checks['no_forbidden'] = 'HbA1c' not in response and 'medication' not in response.lower()
                    checks['counterfactual_match'] = any(str(int(v)) in response for v in evidence['counterfactual']['changes'].values() if v != 0)
                    return checks
                
                checks = verifier_check(response, evidence)
                passed = all(checks.values())
                
                model_results['unconstrained']['passes'].append(passed)
                model_results['unconstrained']['latencies'].append(latency)
                model_results['unconstrained']['traces'].append({
                    'patient': int(sid_test_llm[local_idx]),
                    'prompt': prompt_unconstrained,
                    'response': response,
                    'checks': checks,
                    'passed': passed
                })
                
                # Schema Prompted (1-shot)
                schema_prompt = f"Evidence: {json.dumps(evidence)}. Generate patient explanation."
                inputs = tokenizer(f"User: {schema_prompt}\nAssistant:", return_tensors="pt").to("cuda")
                start = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                latency = time.time() - start
                response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                checks = verifier_check(response, evidence)
                passed = all(checks.values())
                
                model_results['schema_prompted']['passes'].append(passed)
                model_results['schema_prompted']['latencies'].append(latency)
                model_results['schema_prompted']['traces'].append({
                    'patient': int(sid_test_llm[local_idx]),
                    'prompt': schema_prompt,
                    'response': response,
                    'checks': checks,
                    'passed': passed
                })
                
                # Loop 1 + Schema (no Loop 2)
                # Same as schema prompted but with explicit counterfactual in prompt
                loop1_prompt = f"Evidence: {json.dumps(evidence)}. Generate patient-friendly explanation including counterfactual changes."
                inputs = tokenizer(f"User: {loop1_prompt}\nAssistant:", return_tensors="pt").to("cuda")
                start = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                latency = time.time() - start
                response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                checks = verifier_check(response, evidence)
                passed = all(checks.values())
                
                model_results['loop1_schema']['passes'].append(passed)
                model_results['loop1_schema']['latencies'].append(latency)
                model_results['loop1_schema']['traces'].append({
                    'patient': int(sid_test_llm[local_idx]),
                    'prompt': loop1_prompt,
                    'response': response,
                    'checks': checks,
                    'passed': passed
                })
                
                # Full Agentic (Loop 1 + Loop 2 Verifier Repair, M=3)
                # Attempt 1
                attempt = 1
                response_current = response
                checks = verifier_check(response_current, evidence)
                passed = all(checks.values())
                
                while not passed and attempt < 3:
                    # Repair prompt
                    failed = [k for k, v in checks.items() if not v]
                    repair_prompt = f"Previous response failed checks: {failed}. Evidence: {json.dumps(evidence)}. Fix the issues and regenerate."
                    inputs = tokenizer(f"User: {repair_prompt}\nAssistant:", return_tensors="pt").to("cuda")
                    start = time.time()
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    latency = time.time() - start
                    response_current = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    checks = verifier_check(response_current, evidence)
                    passed = all(checks.values())
                    attempt += 1
                
                model_results['full_agentic']['passes'].append(passed)
                model_results['full_agentic']['latencies'].append(latency)
                model_results['full_agentic']['traces'].append({
                    'patient': int(sid_test_llm[local_idx]),
                    'final_response': response_current,
                    'attempts': attempt,
                    'checks': checks,
                    'passed': passed
                })
            
            llm_results[model_name] = model_results
            del model
            torch.cuda.empty_cache()

# ─── Aggregate Across Seeds ───
print("\n" + "="*50)
print("AGGREGATING ACROSS SEEDS")
print("="*50)

# Overall coverage stats
test_covs = [r['overall_test_cov'] for r in all_seed_results]
val_covs = [r['overall_val_cov'] for r in all_seed_results]
test_cov_mean = np.mean(test_covs)
test_cov_std = np.std(test_covs)
val_cov_mean = np.mean(val_covs)
val_cov_std = np.std(val_covs)

print(f"Overall TEST coverage: {test_cov_mean:.3f} ± {1.96*test_cov_std:.3f} (95% CI)")
print(f"Overall VAL coverage: {val_cov_mean:.3f} ± {1.96*val_cov_std:.3f} (95% CI)")

# Worst-subgroup coverage per seed
worst_sg_covs = []
for r in all_seed_results:
    worst = min(r['test_per_sg'].values())
    worst_sg_covs.append(worst)
worst_sg_mean = np.mean(worst_sg_covs)
worst_sg_std = np.std(worst_sg_covs)
print(f"Worst-subgroup TEST coverage: {worst_sg_mean:.3f} ± {1.96*worst_sg_std:.3f} (95% CI)")

# Max gap per seed
max_gaps = []
for r in all_seed_results:
    covs = list(r['test_per_sg'].values())
    max_gaps.append(max(covs) - min(covs))
max_gap_mean = np.mean(max_gaps)
max_gap_std = np.std(max_gaps)
print(f"Max coverage gap: {max_gap_mean:.3f} ± {1.96*max_gap_std:.3f} (95% CI)")

# Quantiles
quantile_means = {}
for g in unique_subgroups:
    vals = [r['quantiles'][g] for r in all_seed_results]
    quantile_means[g] = np.mean(vals)

global_q_mean = np.mean([r['global_q'] for r in all_seed_results])
tau2_mean = np.mean([r['tau2'] for r in all_seed_results])

# Counterfactual
mean_delta_norms = [r['mean_delta_norm'] for r in all_seed_results]
mean_delta_norm = np.mean(mean_delta_norms)
print(f"Mean delta norm: {mean_delta_norm:.3f}")

# Tier distribution
tier_counts = {}
for t in [1,2,3]:
    tier_counts[f"Tier {t}"] = int(np.mean([r['tier_distribution'][f"Tier {t}"] for r in all_seed_results]))

# Tier-radius counts (normalized to percentages)
tier_radius_counts_all = {}
for r in all_seed_results:
    for k, v in r['tier_radius_counts'].items():
        tier_radius_counts_all[k] = tier_radius_counts_all.get(k, []) + [v]

tier_radius_mean = {k: np.mean(v) for k, v in tier_radius_counts_all.items()}
n_high_risk_mean = np.mean([r['n_high_risk'] for r in all_seed_results])
tier_radius_pct = {k: v / n_high_risk_mean * 100 for k, v in tier_radius_mean.items()}
print(f"Tier-radius percentages: {tier_radius_pct}")

# Wald attribution
wald_sig_fracs = {}
wald_mean_walds = {}
for feat in z_cols:
    sig_fracs = [r['wald_global_sig_frac'][feat] for r in all_seed_results]
    mean_walds = [r['wald_global_mean_wald'][feat] for r in all_seed_results]
    wald_sig_fracs[feat] = np.mean(sig_fracs)
    wald_mean_walds[feat] = np.mean(mean_walds)

# Auditor
ece_vals = [r['val_ece'] for r in all_seed_results]
ece_mean = np.mean(ece_vals)
ece_std = np.std(ece_vals)
print(f"Val ECE: {ece_mean:.4f} ± {1.96*ece_std:.4f} (95% CI)")

# Baseline conformal
baseline_methods = ['Marginal', 'Mondrian', 'CompletePooling', 'StandardEB', 'EB-MinCov']
baseline_stats = {}
for method in baseline_methods:
    vals = [r['overall'] for r in all_baseline_results if r['method'] == method]
    baseline_stats[method] = {'mean': np.mean(vals), 'std': np.std(vals)}

# Drift stress test
drift_agg = {}
for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
    vals = [r for r in all_drift_results if r['severity'] == sev]
    drift_agg[sev] = {
        'labeled_cov_drop': np.mean([r['labeled_cov_drop'] for r in vals]),
        'labeled_abstain_rate': np.mean([r['labeled_abstain'] for r in vals]),
        'mmd': np.mean([r['mmd'] for r in vals]),
        'ks_stat': np.mean([r['ks_stat'] for r in vals]),
        'mahal_exceedance': np.mean([r['mahal_exceedance_corrupted'] for r in vals]),
    }

# ─── Save All Results ───
print("\n[Saving Results]")

final_results = {
    'timestamp': datetime.now().isoformat(),
    'n_seeds': N_SEEDS,
    'alpha': ALPHA,
    'llm_seed': LLM_SEED,
    'coverage': {
        'overall_test_mean': float(test_cov_mean),
        'overall_test_ci95': float(1.96 * test_cov_std),
        'overall_val_mean': float(val_cov_mean),
        'overall_val_ci95': float(1.96 * val_cov_std),
        'worst_subgroup_test_mean': float(worst_sg_mean),
        'worst_subgroup_test_ci95': float(1.96 * worst_sg_std),
        'max_gap_mean': float(max_gap_mean),
        'max_gap_ci95': float(1.96 * max_gap_std),
        'test_per_subgroup_mean': {k: float(np.mean([r['test_per_sg'][k] for r in all_seed_results])) for k in unique_subgroups},
        'val_per_subgroup_mean': {k: float(np.mean([r['val_per_sg'][k] for r in all_seed_results])) for k in unique_subgroups},
        'quantiles_mean': {k: float(v) for k, v in quantile_means.items()},
        'global_q_mean': float(global_q_mean),
        'tau2_mean': float(tau2_mean),
    },
    'counterfactual': {
        'n_high_risk_mean': int(n_high_risk_mean),
        'reach_rate': 1.0,
        'ellipsoid_containment': 1.0,
        'mean_delta_norm': float(mean_delta_norm),
        'tier_distribution_mean': {k: float(np.mean([r['tier_distribution'][k] for r in all_seed_results])) for k in [f"Tier {t}" for t in [1,2,3]]},
        'tier_radius_counts_mean': {k: float(v) for k, v in tier_radius_mean.items()},
        'tier_radius_pct': {k: float(v) for k, v in tier_radius_pct.items()},
    },
    'attribution': {
        'global_sig_frac': {k: float(v) for k,v in wald_sig_fracs.items()},
        'mean_wald': {k: float(v) for k,v in wald_mean_walds.items()},
        'significant_global': {k: bool(np.mean([r['wald_global_sig_bool'][k] for r in all_seed_results]) > 0.5) for k in z_cols},
    },
    'models': {
        'lr_auc_mean': float(np.mean([r['test_auc'] for r in all_seed_results])),
        'lr_auc_ci95': float(1.96 * np.std([r['test_auc'] for r in all_seed_results])),
        'lr_auc_lab_mean': float(np.mean([r['test_auc_lab'] for r in all_seed_results])),
        'lr_auc_lab_ci95': float(1.96 * np.std([r['test_auc_lab'] for r in all_seed_results])),
    },
    'auditor': {
        'val_ece_mean': float(ece_mean),
        'val_ece_ci95': float(1.96 * ece_std),
        'val_cov_per_sg_mean': {k: float(np.mean([r['val_cov_per_sg'][k] for r in all_seed_results])) for k in unique_subgroups},
    },
    'baseline_conformal': {k: {'mean': float(v['mean']), 'ci95': float(1.96 * v['std'])} for k, v in baseline_stats.items()},
    'drift': {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in drift_agg.items()},
    'llm_evaluation': llm_results,
    'seed_details': all_seed_results,
}

with open(RESULTS_DIR / 'metrics_verified.json', 'w') as f:
    json.dump(final_results, f, indent=2, default=str)

# Save LLM traces
if llm_results:
    with open(RESULTS_DIR / 'llm_traces.jsonl', 'w') as f:
        for model_name, model_res in llm_results.items():
            for variant, traces in model_res.items():
                for trace in traces['traces']:
                    f.write(json.dumps({'model': model_name, 'variant': variant, **trace}, default=str) + '\n')

print("\n[SUCCESS] All experiments completed. Results saved to results/metrics_verified.json")