#!/usr/bin/env python3
"""
7B LLM Benchmark - 4-bit NF4, GPU-0 only, resumable
===================================================
Evaluates Qwen2.5-7B-Instruct on the same seed-3 / 86 high-risk patients and
the same four variants, prompts, verifier and token budgets as the 1.5B run
(run_llm_eval_fixed.py), so Table 8 is apples-to-apples.

Memory design (the original OOM fix):
  * genuine 4-bit NF4 weights with double quantisation  (~4.5-6.5 GB, not 15.2 GB fp16)
  * device_map pinned to GPU 0, never "auto"           (no silent CPU offload)
  * SDPA attention + use_cache=True                     (KV cache, no eager attention)
  * one sequence at a time; OOM retry clears cache, then halves the token
    budget *transparently* (recorded as oom_retries / reduced max_new_tokens)

Durability:
  * every trace is appended + fsync'd immediately
  * a torn final line from a kill is repaired on the next start
  * completed (patient, variant) pairs are skipped; duplicates are removed
  * metrics are written atomically via a .tmp + os.replace
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from scipy.stats import norm, gaussian_kde
from scipy.special import expit, logit
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="4-bit Qwen2.5-7B agentic benchmark (resumable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate only the first N high-risk patients (0 = all 86).")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Verifier repair attempts for full_agentic (3 = full protocol).")
    p.add_argument("--variants", default="unconstrained,schema_prompted,loop1_schema,full_agentic",
                   help="Comma-separated subset of variants.")
    p.add_argument("--traces-file", default="results/llm_traces_7b.jsonl")
    p.add_argument("--metrics-file", default="results/metrics_verified.json")
    p.add_argument("--mock", action="store_true",
                   help="Run the pipeline with a deterministic stub model (no GPU).")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore an existing traces file and start from scratch.")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args(argv)

    if args.seed != 3:
        p.error("this benchmark is pinned to seed 3 to match the 1.5B evaluation")
    if args.limit < 0:
        p.error("--limit must be >= 0")
    if args.max_attempts < 1:
        p.error("--max-attempts must be >= 1")
    return args


args = parse_args()

# ------------------------------------------------------------------
# Original data / model / conformal setup (unchanged from the 1.5B run)
# ------------------------------------------------------------------
# Load data
print("[Loading Data]")
cohort = pd.read_csv(ROOT / "data" / "nhanes_diabetes_cohort.csv")
print(f"Loaded {len(cohort)} patients, diabetes prevalence: {cohort['known_diabetes'].mean()*100:.1f}%")

# Feature engineering (same as before)
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

embeddings = pd.DataFrame(index=cohort.index)
embeddings['subject_id'] = cohort['SEQN']
embeddings['subgroup'] = cohort['Race_Ethnicity'].astype(str)
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
subject_ids_all = embeddings['subject_id'].values
unique_subgroups = sorted(embeddings['subgroup'].unique())

# Re-create seed 3 split
LLM_SEED = 3
np.random.seed(LLM_SEED)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=LLM_SEED)
trainval_idx, test_idx = next(sss.split(X_all, y_all))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=LLM_SEED)
train_idx, val_idx = next(sss2.split(X_all[trainval_idx], y_all[trainval_idx]))
train_idx = trainval_idx[train_idx]
val_idx = trainval_idx[val_idx]

sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=LLM_SEED)
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
sid_test = subject_ids_all[test_idx]

print(f"Train: {len(X_train)}, Platt: {len(X_platt)}, Conf+Audit: {len(X_conf_audit)}, Test: {len(X_test)}")

# Train LR + Platt
base_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=LLM_SEED, solver='lbfgs'))
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

def predict_calibrated(X, w_cal, b_cal):
    return expit(X @ w_cal + b_cal)

# Conformal calibration
ca_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=LLM_SEED, solver='lbfgs'))
])

sss_ca = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=LLM_SEED)
ca_train_idx, ca_cal_idx = next(sss_ca.split(X_conf_audit, y_conf_audit))
X_ca_train, X_ca_cal = X_conf_audit[ca_train_idx], X_conf_audit[ca_cal_idx]
y_ca_train, y_ca_cal = y_conf_audit[ca_train_idx], y_conf_audit[ca_cal_idx]
sg_ca_train, sg_ca_cal = sg_conf_audit[ca_train_idx], sg_conf_audit[ca_cal_idx]

ca_pipe.fit(X_ca_train, y_ca_train)
logits_ca_cal = ca_pipe.decision_function(X_ca_cal).reshape(-1, 1)
platt_lr_ca = LogisticRegression(solver='lbfgs', max_iter=1000)
platt_lr_ca.fit(logits_ca_cal, y_ca_cal)
platt_a_ca = platt_lr_ca.coef_[0][0]
platt_b_ca = platt_lr_ca.intercept_[0]

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

# EB-MinCov conformal
ALPHA = 0.1
logits_ca_cal_flat = ca_pipe.decision_function(X_ca_cal)
p_ca_cal = expit(platt_a_ca * logits_ca_cal_flat + platt_b_ca)
scores_ca_cal = np.where(y_ca_cal == 1, 1 - p_ca_cal, p_ca_cal)

subgroup_quantiles = {}
within_vars = {}
for g in unique_subgroups:
    mask = (sg_ca_cal == g)
    scores_g = scores_ca_cal[mask]
    if len(scores_g) > 0:
        q = np.quantile(scores_g, 1 - ALPHA, method='higher')
        subgroup_quantiles[g] = q
        n_g = len(scores_g)
        if n_g > 10:
            kde = gaussian_kde(scores_g)
            f_q = kde(q)[0]
            if f_q > 0:
                analytical_var = ALPHA * (1 - ALPHA) / (n_g * f_q**2)
                within_vars[g] = max(analytical_var, 1e-8)
            else:
                within_vars[g] = 1e-8
        else:
            within_vars[g] = 1e-8
    else:
        subgroup_quantiles[g] = np.quantile(scores_ca_cal, 1 - ALPHA, method='higher')
        within_vars[g] = 1e-8

global_q = np.quantile(scores_ca_cal, 1 - ALPHA, method='higher')
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

eb_mincov = shrunken

# Calibration weights
def predict_calibrated(X, w_cal, b_cal):
    return expit(X @ w_cal + b_cal)

p_val_cal = predict_calibrated(X_conf_audit, w_cal, b_cal)

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

ece = np.mean(np.abs(p_val_cal - y_conf_audit))

# Ledoit-Wolf
subgroup_stats = {}
for g in unique_subgroups:
    mask = (subgroups_all == g)
    X_g = X_all[mask]
    if len(X_g) < 3: 
        subgroup_stats[g] = None
        continue
    mu_g = X_g.mean(axis=0)
    lw = LedoitWolf().fit(X_g)
    prec_g = lw.precision_
    r2_g = np.percentile([(x - mu_g) @ prec_g @ (x - mu_g) for x in X_g], 90)
    subgroup_stats[g] = {'mu': mu_g, 'prec': prec_g, 'r2': r2_g, 'n': len(X_g)}

mu_global = X_all.mean(axis=0)
lw_global = LedoitWolf().fit(X_all)
prec_global = lw_global.precision_
r2_global = np.percentile([(x - mu_global) @ prec_global @ (x - mu_global) for x in X_all], 90)
subgroup_stats['global'] = {'mu': mu_global, 'prec': prec_global, 'r2': r2_global, 'n': len(X_all)}

# Counterfactual solver
TIER_DEFINITIONS = {
    1: {'name': 'Behavioral', 'continuous': [], 'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']},
    2: {'name': 'Behavioral + Anthropometric', 'continuous': ['z_BMI', 'z_Waist_cm'], 'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']},
    3: {'name': 'Behavioral + Anthropometric + Clinical', 'continuous': ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP'], 'binary': ['z_Moderate_Activity', 'z_Vigorous_Activity', 'z_Current_Smoker', 'z_Current_Drinker']},
}

FROZEN_FEATURES = ['z_RIDAGEYR', 'z_Gender_Female', 'z_Gender_Male', 'z_INDFMPIR']

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

# Get high-risk patients
p_test_cal = predict_calibrated(X_test, w_cal, b_cal)
high_risk_idx = np.where(p_test_cal > 0.5)[0]
print(f"High-risk patients for 7B LLM eval: {len(high_risk_idx)}")

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
VARIANT_ORDER = ['unconstrained', 'schema_prompted', 'loop1_schema', 'full_agentic']
TOKEN_BUDGETS = {
    'unconstrained': 100,
    'schema_prompted': 150,
    'loop1_schema': 150,
    'full_agentic': 150,
}

def verifier_check(response, evidence):
    """Verifier identical to run_llm_eval_fixed.py (1.5B run)."""
    checks = {}
    risk_str = f"{evidence['risk']}"
    checks['risk_match'] = risk_str in response or f"{evidence['risk']:.1f}" in response
    checks['coverage_stated'] = '90%' in response or '90 percent' in response.lower()
    checks['subgroup_stated'] = evidence['subgroup'] in response
    checks['no_forbidden'] = ('HbA1c' not in response
                              and 'medication' not in response.lower()
                              and 'medicine' not in response.lower())
    checks['counterfactual_match'] = True  # simplified, same as 1.5B run
    return checks


SELECTED_VARIANTS = [v.strip() for v in args.variants.split(',') if v.strip()]
for _v in SELECTED_VARIANTS:
    if _v not in VARIANT_ORDER:
        raise SystemExit(f"Unknown variant '{_v}'. Choose from {VARIANT_ORDER}")

TRACES_PATH = Path(args.traces_file)
METRICS_PATH = Path(args.metrics_file)
if not TRACES_PATH.is_absolute():
    TRACES_PATH = ROOT / TRACES_PATH
if not METRICS_PATH.is_absolute():
    METRICS_PATH = ROOT / METRICS_PATH
TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = "MOCK-Qwen2.5-7B-Instruct" if args.mock else args.model
QUANT_LABEL = "none (mock)" if args.mock else "4-bit NF4, double-quant, fp16 compute"

if args.limit and args.limit > 0:
    eval_idx = high_risk_idx[:args.limit]
else:
    eval_idx = high_risk_idx

print(f"[Scope] patients={len(eval_idx)}/{len(high_risk_idx)} variants={SELECTED_VARIANTS} "
      f"max_attempts={args.max_attempts} mock={args.mock}")


def log(msg):
    print(msg, flush=True)


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def vram_stats():
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_GB": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_GB": round(torch.cuda.memory_reserved() / 1e9, 3),
        "peak_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "free_GB": round(free / 1e9, 3),
        "total_GB": round(total / 1e9, 3),
    }


# ============================================================
# RESUME: load completed (patient, variant) pairs, repair torn tail
# ============================================================
def repair_trailing_partial_line(path):
    """A killed process can leave a half-written last line. Truncate it."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        data = f.read()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    keep = data[:cut + 1] if cut >= 0 else b""
    with open(path, "wb") as f:
        f.write(keep)
        f.flush()
        os.fsync(f.fileno())
    log(f"[Resume] truncated torn final line ({len(data) - len(keep)} bytes discarded)")


existing = {}
duplicates_seen = 0
if TRACES_PATH.exists() and not args.no_resume:
    repair_trailing_partial_line(TRACES_PATH)
    with open(TRACES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("patient"), rec.get("variant"))
            if key in existing:
                duplicates_seen += 1
            existing[key] = rec  # last write wins

if existing:
    models_seen = {r.get("model") for r in existing.values()}
    if models_seen != {MODEL_LABEL}:
        raise SystemExit(
            f"[Resume] traces file contains records from {sorted(map(str, models_seen))}, "
            f"but this run uses {MODEL_LABEL!r}. Use a fresh --traces-file or --no-resume."
        )
    if duplicates_seen:
        # Rewrite compacted so the canonical file has exactly one record per key.
        backup = TRACES_PATH.with_suffix(TRACES_PATH.suffix + ".dup.bak")
        os.replace(TRACES_PATH, backup)
        with open(TRACES_PATH, "w", encoding="utf-8") as f:
            for rec in existing.values():
                f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        log(f"[Resume] deduplicated {duplicates_seen} duplicate record(s); backup={backup.name}")

done = set(existing.keys())
todo_pairs = [(int(pid), v) for pid in eval_idx for v in SELECTED_VARIANTS
              if (int(sid_test[pid]), v) not in done]
log(f"[Resume] completed={len(done)} pending={len(todo_pairs)}")


# ============================================================
# MODEL LOADING (4-bit NF4, GPU 0 only)
# ============================================================
tokenizer = None
model = None
load_seconds = None
load_vram = {}
vram_before = {}

if not args.mock:
    from transformers import BitsAndBytesConfig

    log(f"[Load] {args.model} with 4-bit NF4 on GPU 0")
    vram_before = vram_stats()
    log(f"[VRAM] before load: {vram_before}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    load_seconds = time.perf_counter() - t0
    load_vram = vram_stats()

    param_devices = {str(p.device) for p in model.parameters()}
    log(f"[Load] done in {load_seconds:.1f}s | param devices={sorted(param_devices)}")
    log(f"[VRAM] after load: {load_vram}")
    log(f"[Load] hf_device_map={getattr(model, 'hf_device_map', None)}")
    log(f"[Load] attn_implementation={getattr(model.config, '_attn_implementation', None)}")

    if any(d in ("cpu", "meta") for d in param_devices):
        raise RuntimeError(
            f"Model was offloaded to {sorted(param_devices)}; refusing to benchmark "
            "a CPU/meta-offloaded model. Check VRAM and quantisation config."
        )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


# ============================================================
# GENERATION (OOM-aware, transparent budget reduction)
# ============================================================
def _mock_generate(prompt, max_new_tokens, evidence, fail=False):
    """Deterministic stand-in used only for --mock pipeline smoke tests."""
    risk = evidence["risk"]
    if fail:
        body = f"This patient has a {risk}% risk of diabetes. Focus on diet and walking."
    else:
        body = (f"This patient has a {risk}% risk of diabetes. The {evidence['subgroup']} "
                f"subgroup has {evidence['subgroup_coverage']}% coverage at the 90% level. "
                f"Suggested changes: {evidence['counterfactual']['changes']}.")
    return body, 0.0, max_new_tokens, 0


def generate_text(prompt, max_new_tokens, evidence, mock_fail=False):
    """Return (response, latency_s, effective_max_new_tokens, oom_retries).

    Latency covers only tokenisation->generation->decode for this call.
    On repeated OOM we halve the token budget and record it in the trace.
    """
    if args.mock:
        return _mock_generate(prompt, max_new_tokens, evidence, fail=mock_fail)

    inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to(model.device)
    effective = int(max_new_tokens)
    oom_retries = 0

    for attempt in range(3):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=effective,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            response = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            del out
            return response, latency, effective, oom_retries
        except torch.cuda.OutOfMemoryError:
            oom_retries += 1
            torch.cuda.empty_cache()
            if attempt == 0:
                log("  [OOM] cache cleared; retrying same token budget")
                continue
            new_effective = max(32, effective // 2)
            if new_effective == effective:
                break
            effective = new_effective
            log(f"  [OOM] retrying with reduced max_new_tokens={effective} "
                f"(recorded in trace as oom_token_reduction)")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            oom_retries += 1
            torch.cuda.empty_cache()
            if attempt == 0:
                continue
            effective = max(32, effective // 2)
            log(f"  [OOM] RuntimeError OOM; reduced max_new_tokens={effective}")

    raise RuntimeError("generation failed after repeated CUDA OOM")


# ============================================================
# EVALUATION LOOP
# ============================================================
vram_events = []
records_this_run = 0
t_run_start = time.perf_counter()

with open(TRACES_PATH, "a", encoding="utf-8") as trace_file:

    def emit(rec):
        global records_this_run
        trace_file.write(json.dumps(rec, default=str) + "\n")
        trace_file.flush()
        os.fsync(trace_file.fileno())
        records_this_run += 1

    for pos, local_idx in enumerate(eval_idx):
        z = X_test[local_idx]
        sg = sg_test[local_idx]
        p = p_test_cal[local_idx]
        patient_id = int(sid_test[local_idx])

        pending = [v for v in SELECTED_VARIANTS if (patient_id, v) not in done]
        if not pending:
            continue

        delta, mahal, inside, p_final, tier_used, radius_mult, feasible = solve_with_recovery(
            z, sg, 0.5, w_cal, b_cal, subgroup_stats)

        cf_changes = {z_cols[i]: round(delta[i], 2)
                      for i in range(len(delta)) if abs(delta[i]) > 1e-3}

        evidence = {
            'patient_id': patient_id,
            'subgroup': sg,
            'risk': round(p * 100, 1),
            'prediction_set': [1, 0],
            'coverage': '90%',
            'counterfactual': {
                'tier': tier_used,
                'radius_mult': radius_mult,
                'changes': cf_changes,
                'inside_ellipsoid': bool(inside),
            },
            'subgroup_coverage': round(val_cov_per_sg.get(sg, 0) * 100, 1),
            'ece': round(ece * 100, 1),
        }

        responses = {}  # variant -> dict(response, latency, checks, passed, ...)

        # (a) Unconstrained
        if 'unconstrained' in pending:
            prompt = (f"Patient has {p*100:.1f}% diabetes risk. "
                      f"Explain what lifestyle changes would reduce risk to <= 50%.")
            resp, lat, used, oom = generate_text(prompt, TOKEN_BUDGETS['unconstrained'], evidence)
            checks = verifier_check(resp, evidence)
            passed = all(checks.values())
            responses['unconstrained'] = dict(response=resp, latency=lat, checks=checks,
                                              passed=passed, prompt=prompt,
                                              max_new_tokens=used, oom_retries=oom)
            emit({'model': MODEL_LABEL, 'variant': 'unconstrained', 'mock': bool(args.mock),
                  'quantization': QUANT_LABEL, 'patient': patient_id, 'subgroup': sg,
                  'risk': evidence['risk'], 'prompt': prompt, 'response': resp,
                  'checks': checks, 'passed': passed, 'latency_s': round(lat, 3),
                  'max_new_tokens': used, 'oom_retries': oom})

        # (b) Schema prompted
        if 'schema_prompted' in pending:
            prompt = f"Evidence: {json.dumps(evidence)}. Generate patient-friendly explanation."
            resp, lat, used, oom = generate_text(prompt, TOKEN_BUDGETS['schema_prompted'], evidence)
            checks = verifier_check(resp, evidence)
            passed = all(checks.values())
            responses['schema_prompted'] = dict(response=resp, latency=lat, checks=checks,
                                                passed=passed, prompt=prompt,
                                                max_new_tokens=used, oom_retries=oom)
            emit({'model': MODEL_LABEL, 'variant': 'schema_prompted', 'mock': bool(args.mock),
                  'quantization': QUANT_LABEL, 'patient': patient_id, 'subgroup': sg,
                  'risk': evidence['risk'], 'prompt': prompt, 'response': resp,
                  'checks': checks, 'passed': passed, 'latency_s': round(lat, 3),
                  'max_new_tokens': used, 'oom_retries': oom})

        # (c) Loop 1 + schema (no repair)
        if 'loop1_schema' in pending:
            prompt = (f"Evidence: {json.dumps(evidence)}. Generate patient-friendly "
                      f"explanation including counterfactual changes.")
            resp, lat, used, oom = generate_text(prompt, TOKEN_BUDGETS['loop1_schema'], evidence)
            checks = verifier_check(resp, evidence)
            passed = all(checks.values())
            responses['loop1_schema'] = dict(response=resp, latency=lat, checks=checks,
                                             passed=passed, prompt=prompt,
                                             max_new_tokens=used, oom_retries=oom)
            emit({'model': MODEL_LABEL, 'variant': 'loop1_schema', 'mock': bool(args.mock),
                  'quantization': QUANT_LABEL, 'patient': patient_id, 'subgroup': sg,
                  'risk': evidence['risk'], 'prompt': prompt, 'response': resp,
                  'checks': checks, 'passed': passed, 'latency_s': round(lat, 3),
                  'max_new_tokens': used, 'oom_retries': oom})

        # (d) Full agentic: Loop 1 response + verifier-driven repairs
        if 'full_agentic' in pending:
            if 'loop1_schema' in responses:
                base = responses['loop1_schema']
            else:
                prompt = (f"Evidence: {json.dumps(evidence)}. Generate patient-friendly "
                          f"explanation including counterfactual changes.")
                resp, lat, used, oom = generate_text(prompt, TOKEN_BUDGETS['full_agentic'], evidence)
                base = dict(response=resp, latency=lat, checks=verifier_check(resp, evidence),
                            prompt=prompt, max_new_tokens=used, oom_retries=oom)
                base['passed'] = all(base['checks'].values())

            current_response = base['response']
            current_checks = base['checks']
            passed = all(current_checks.values())
            attempts = 1
            total_latency = base['latency']
            attempt_latencies = [round(base['latency'], 3)]
            attempt_checks = [current_checks]
            total_oom = base.get('oom_retries', 0)
            used_budget = base.get('max_new_tokens', TOKEN_BUDGETS['full_agentic'])

            while not passed and attempts < args.max_attempts:
                failed = [k for k, v in current_checks.items() if not v]
                prompt = (f"Previous response failed checks: {failed}. "
                          f"Evidence: {json.dumps(evidence)}. Fix the issues and regenerate.")
                current_response, lat, used_budget, oom = generate_text(
                    prompt, TOKEN_BUDGETS['full_agentic'], evidence)
                current_checks = verifier_check(current_response, evidence)
                passed = all(current_checks.values())
                attempts += 1
                total_latency += lat
                total_oom += oom
                attempt_latencies.append(round(lat, 3))
                attempt_checks.append(current_checks)

            emit({'model': MODEL_LABEL, 'variant': 'full_agentic', 'mock': bool(args.mock),
                  'quantization': QUANT_LABEL, 'patient': patient_id, 'subgroup': sg,
                  'risk': evidence['risk'], 'prompt': base['prompt'],
                  'final_response': current_response, 'response': current_response,
                  'attempts': attempts, 'attempt_checks': attempt_checks,
                  'checks': current_checks, 'passed': passed,
                  'latency_s': round(total_latency, 3),
                  'attempt_latencies': attempt_latencies,
                  'max_new_tokens': used_budget, 'oom_retries': total_oom})

        done.update((patient_id, v) for v in pending)

        if pos % 5 == 0 or pos == len(eval_idx) - 1:
            elapsed = time.perf_counter() - t_run_start
            stats = vram_stats()
            vram_events.append({"patient": patient_id, **stats})
            log(f"[{pos+1}/{len(eval_idx)}] patient={patient_id} sg={sg} "
                f"risk={evidence['risk']:.1f} | emitted={records_this_run} | "
                f"elapsed={elapsed/60:.1f}min | VRAM peak={stats.get('peak_allocated_GB')}GB "
                f"free={stats.get('free_GB')}GB")


# ============================================================
# AGGREGATE (from the canonical trace file, deduplicated)
# ============================================================
def load_traces(path):
    """Return {(patient,variant): rec}, last record wins."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(rec.get("patient"), rec.get("variant"))] = rec
    return out


canonical = load_traces(TRACES_PATH)
scope_keys = {(int(sid_test[pid]), v) for pid in eval_idx for v in SELECTED_VARIANTS}
scoped = {k: rec for k, rec in canonical.items() if k in scope_keys}

per_variant = {}
for variant in SELECTED_VARIANTS:
    recs = [rec for (pid, v), rec in scoped.items() if v == variant]
    n = len(recs)
    if n == 0:
        per_variant[variant] = {"n": 0, "completed": False}
        continue
    passed = sum(1 for r in recs if r.get("passed"))
    lats = sorted(float(r.get("latency_s", 0.0)) for r in recs)
    per_variant[variant] = {
        "n": n,
        "completed": n == len(eval_idx),
        "passed": passed,
        "final_pass_pct": round(100.0 * passed / n, 2),
        "latency_mean_s": round(float(np.mean(lats)), 3),
        "latency_median_s": round(float(np.median(lats)), 3),
        "latency_p95_s": round(float(np.percentile(lats, 95)), 3),
        "tokens_per_s_mean": round(float(np.mean([
            r.get("max_new_tokens", TOKEN_BUDGETS[variant]) / max(float(r.get("latency_s", 0.0)), 1e-6)
            for r in recs])), 2),
        "oom_retries_total": int(sum(int(r.get("oom_retries", 0)) for r in recs)),
    }

full = [rec for (pid, v), rec in scoped.items() if v == 'full_agentic']
attempt_progression = {}
fallback = None
fallback_pct = None
if full:
    counts = {}
    for r in full:
        a = int(r.get("attempts", 1))
        counts[a] = counts.get(a, 0) + 1
    n = len(full)
    attempt_progression = {
        "attempt_1_pass": counts.get(1, 0),
        "attempt_2_pass": counts.get(2, 0),
        "attempt_3_pass": counts.get(3, 0),
        "attempt_1_pct": round(100.0 * counts.get(1, 0) / n, 2),
        "attempt_2_pct": round(100.0 * counts.get(2, 0) / n, 2),
        "attempt_3_pct": round(100.0 * counts.get(3, 0) / n, 2),
        "attempt_1_cum_pct": round(100.0 * counts.get(1, 0) / n, 2),
        "attempt_2_cum_pct": round(100.0 * (counts.get(1, 0) + counts.get(2, 0)) / n, 2),
        "attempt_3_cum_pct": round(100.0 * (counts.get(1, 0) + counts.get(2, 0)
                                             + counts.get(3, 0)) / n, 2),
    }
    passed_full = sum(1 for r in full if r.get("passed"))
    fallback = n - passed_full
    fallback_pct = round(100.0 * fallback / n, 2)

complete = all(per_variant.get(v, {}).get("completed") for v in SELECTED_VARIANTS)
all_variants = set(SELECTED_VARIANTS) == set(VARIANT_ORDER)
elapsed_total = time.perf_counter() - t_run_start

summary = {
    "model": MODEL_LABEL,
    "requested_model": args.model,
    "mock": bool(args.mock),
    "quantization": QUANT_LABEL,
    "seed": args.seed,
    "timestamp": datetime.now().isoformat(),
    "protocol": {
        "n_high_risk_available": int(len(high_risk_idx)),
        "n_patients_evaluated": int(len(eval_idx)),
        "variants": SELECTED_VARIANTS,
        "max_attempts": int(args.max_attempts),
        "token_budgets": TOKEN_BUDGETS,
        "decoding": "greedy (do_sample=False)",
        "prompt_templates": "identical to run_llm_eval_fixed.py (1.5B)",
    },
    "complete_full_protocol": bool(complete and all_variants and args.max_attempts >= 3
                                   and len(eval_idx) == len(high_risk_idx)),
    "variants": per_variant,
    "attempt_progression": attempt_progression,
    "fallback": fallback,
    "fallback_pct": fallback_pct,
    "runtime": {
        "model_load_s": round(load_seconds, 2) if load_seconds else None,
        "evaluation_s": round(elapsed_total, 2),
        "evaluation_min": round(elapsed_total / 60, 2),
        "records_written_this_run": records_this_run,
        "duplicates_removed_on_resume": duplicates_seen,
    },
    "vram": {
        "before_load": vram_before,
        "after_load": load_vram,
        "peak_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if torch.cuda.is_available() else None,
        "samples": vram_events[-5:],
    },
}

if args.mock and "metrics_verified" in METRICS_PATH.name:
    raise SystemExit("Refusing to write MOCK results into the canonical metrics file.")

# Merge under llm_evaluation_7b without touching other keys.
metrics = {}
if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError:
        metrics = {}
metrics["llm_evaluation_7b"] = summary
atomic_write_json(METRICS_PATH, metrics)

log("\n[7B Results]")
for variant in SELECTED_VARIANTS:
    d = per_variant.get(variant, {})
    if d.get("n"):
        log(f"  {variant}: Final Pass = {d['final_pass_pct']:.1f}% "
            f"({d['passed']}/{d['n']}), mean latency = {d['latency_mean_s']:.2f}s, "
            f"p95 = {d['latency_p95_s']:.2f}s")
if attempt_progression:
    log(f"  attempts: 1={attempt_progression['attempt_1_pass']} "
        f"2={attempt_progression['attempt_2_pass']} 3={attempt_progression['attempt_3_pass']} "
        f"| fallback={fallback} ({fallback_pct}%)")
log(f"[Done] traces={TRACES_PATH} metrics={METRICS_PATH} "
    f"elapsed={elapsed_total/60:.1f}min complete={summary['complete_full_protocol']}")
