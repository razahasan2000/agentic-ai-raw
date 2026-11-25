#!/usr/bin/env python3
"""
Multi-Cycle NHANES Temporal Validation & Lab-Verified Target Sensitivity
CPU-only experiment - attempts to download NHANES XPT files from CDC
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import urllib.request
import io
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
import time

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# NHANES DATA DOWNLOAD HELPERS
# ============================================================
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes/"

# File mapping for cycles
CYCLE_FILES = {
    '2013-2014': {
        'demo': 'DEMO_H.XPT',
        'bmx': 'BMX_H.XPT',
        'bp': 'BPX_H.XPT',
        'smq': 'SMQ_H.XPT',
        'alq': 'ALQ_H.XPT',
        'paq': 'PAQ_H.XPT',
        'diq': 'DIQ_H.XPT',
        'ghb': 'GHB_H.XPT',  # HbA1c
        'glu': 'GLU_H.XPT',  # Fasting glucose
    },
    '2015-2016': {
        'demo': 'DEMO_I.XPT',
        'bmx': 'BMX_I.XPT',
        'bp': 'BPX_I.XPT',
        'smq': 'SMQ_I.XPT',
        'alq': 'ALQ_I.XPT',
        'paq': 'PAQ_I.XPT',
        'diq': 'DIQ_I.XPT',
        'ghb': 'GHB_I.XPT',
        'glu': 'GLU_I.XPT',
    },
    '2017-2018': {
        'demo': 'DEMO_J.XPT',
        'bmx': 'BMX_J.XPT',
        'bp': 'BPX_J.XPT',
        'smq': 'SMQ_J.XPT',
        'alq': 'ALQ_J.XPT',
        'paq': 'PAQ_J.XPT',
        'diq': 'DIQ_J.XPT',
        'ghb': 'GHB_J.XPT',
        'glu': 'GLU_J.XPT',
    },
    '2021-2023': {
        'demo': 'DEMO_L.XPT',
        'bmx': 'BMX_L.XPT',
        'bp': 'BPX_L.XPT',
        'smq': 'SMQ_L.XPT',
        'alq': 'ALQ_L.XPT',
        'paq': 'PAQ_L.XPT',
        'diq': 'DIQ_L.XPT',
        'ghb': 'GHB_L.XPT',
        'glu': 'GLU_L.XPT',
    },
}

def download_xpt(cycle, file_key):
    """Download XPT file from CDC NHANES."""
    cycle_suffix = CYCLE_FILES[cycle][file_key]
    url = f"{NHANES_BASE}{cycle}/{cycle_suffix}"
    try:
        print(f"  Downloading {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        df = pd.read_sas(io.BytesIO(data), format='xport')
        print(f"  Downloaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return None

def merge_cycle_data(cycle):
    """Merge all NHANES files for a cycle."""
    print(f"\n=== Processing {cycle} ===")
    files = CYCLE_FILES[cycle]
    dfs = {}
    
    for key, filename in files.items():
        df = download_xpt(cycle, key)
        if df is not None:
            dfs[key] = df
    
    if not dfs:
        return None
    
    # Merge on SEQN
    merged = dfs['demo'][['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDHHIN2', 'WTINT2YR', 'SDMVPSU', 'SDMVSTRA']].copy()
    
    # Demographics
    merged['Gender'] = merged['RIAGENDR'].map({1: 'Male', 2: 'Female'})
    merged['Race_Ethnicity'] = merged['RIDRETH3'].map({
        1: 'Mexican_American', 2: 'Other_Hispanic', 3: 'White', 
        4: 'Black', 6: 'Asian', 7: 'Other'
    })
    
    # BMI, Waist
    if 'bmx' in dfs:
        bmx = dfs['bmx'][['SEQN', 'BMXBMI', 'BMXWAIST']].copy()
        bmx.columns = ['SEQN', 'BMI', 'Waist_cm']
        merged = merged.merge(bmx, on='SEQN', how='left')
    
    # Blood Pressure
    if 'bp' in dfs:
        bp = dfs['bp'][['SEQN', 'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2', 'BPXSY3', 'BPXDI3']].copy()
        bp['SBP'] = bp[['BPXSY1', 'BPXSY2', 'BPXSY3']].mean(axis=1)
        bp['DBP'] = bp[['BPXDI1', 'BPXDI2', 'BPXDI3']].mean(axis=1)
        bp = bp[['SEQN', 'SBP', 'DBP']]
        merged = merged.merge(bp, on='SEQN', how='left')
    
    # Smoking
    if 'smq' in dfs:
        smq = dfs['smq'][['SEQN', 'SMQ020', 'SMQ040']].copy()
        smq['Current_Smoker'] = (smq['SMQ020'] == 1).astype(int)
        smq['Former_Smoker'] = ((smq['SMQ020'] == 2) & (smq['SMQ040'] == 1)).astype(int)
        smq = smq[['SEQN', 'Current_Smoker', 'Former_Smoker']]
        merged = merged.merge(smq, on='SEQN', how='left')
    
    # Alcohol
    if 'alq' in dfs:
        alq = dfs['alq'][['SEQN', 'ALQ120Q']].copy()
        alq['Current_Drinker'] = (alq['ALQ120Q'] >= 1).astype(int)
        alq = alq[['SEQN', 'Current_Drinker']]
        merged = merged.merge(alq, on='SEQN', how='left')
    
    # Physical Activity
    if 'paq' in dfs:
        paq = dfs['paq'][['SEQN', 'PAD630', 'PAD660']].copy()
        paq['Moderate_Activity'] = (paq['PAD630'] >= 1).astype(int)
        paq['Vigorous_Activity'] = (paq['PAD660'] >= 1).astype(int)
        paq = paq[['SEQN', 'Moderate_Activity', 'Vigorous_Activity']]
        merged = merged.merge(paq, on='SEQN', how='left')
    
    # Diabetes
    if 'diq' in dfs:
        diq = dfs['diq'][['SEQN', 'DIQ010']].copy()
        diq.columns = ['SEQN', 'known_diabetes']
        diq['known_diabetes'] = diq['known_diabetes'].map({1: 1, 2: 0, 3: 0})
        merged = merged.merge(diq, on='SEQN', how='left')
    
    # Lab values (HbA1c, Fasting Glucose)
    if 'ghb' in dfs:
        ghb = dfs['ghb'][['SEQN', 'LBXGH']].copy()
        ghb.columns = ['SEQN', 'HbA1c']
        merged = merged.merge(ghb, on='SEQN', how='left')
    
    if 'glu' in dfs:
        glu = dfs['glu'][['SEQN', 'LBXGLU']].copy()
        glu.columns = ['SEQN', 'FPG']
        merged = merged.merge(glu, on='SEQN', how='left')
    
    # Lab-verified diabetes
    if 'HbA1c' in merged.columns and 'FPG' in merged.columns:
        merged['lab_diabetes'] = ((merged['known_diabetes'] == 1) | 
                                   (merged['HbA1c'] >= 6.5) | 
                                   (merged['FPG'] >= 126)).astype(int)
    else:
        merged['lab_diabetes'] = merged['known_diabetes']
    
    # Income-to-poverty ratio
    if 'INDHHIN2' in merged.columns:
        merged['INDFMPIR'] = merged['INDHHIN2'] / 5.0  # approximate
    
    print(f"  Merged {len(merged)} participants")
    return merged

# ============================================================
# MAIN EXECUTION
# ============================================================
print("=" * 60)
print("MULTI-CYCLE NHANES TEMPORAL VALIDATION")
print("=" * 60)

all_cycles_data = {}
cycles_to_try = ['2013-2014', '2015-2016', '2017-2018', '2021-2023']

for cycle in cycles_to_try:
    try:
        df = merge_cycle_data(cycle)
        if df is not None and len(df) > 100:
            all_cycles_data[cycle] = df
            print(f"Successfully loaded {cycle}: {len(df)} participants")
        else:
            print(f"Failed to load {cycle}")
    except Exception as e:
        print(f"Error processing {cycle}: {e}")

# Save raw merged data
for cycle, df in all_cycles_data.items():
    df.to_pickle(ROOT / "results" / f"nhanes_{cycle.replace('-', '_')}.pkl")

print(f"\nLoaded cycles: {list(all_cycles_data.keys())}")

# ============================================================
# TEMPORAL VALIDATION: Train 2013-2016 -> Cal 2017-2018 -> Test 2021-2023
# ============================================================
print("\n" + "=" * 60)
print("TEMPORAL VALIDATION")
print("=" * 60)

train_cycles = ['2013-2014', '2015-2016']
cal_cycle = '2017-2018'
test_cycle = '2021-2023'

if all(c in all_cycles_data for c in train_cycles + [cal_cycle, test_cycle]):
    print("All required cycles available!")
    
    # Combine training cycles
    train_dfs = [all_cycles_data[c] for c in train_cycles]
    train_df = pd.concat(train_dfs, ignore_index=True)
    cal_df = all_cycles_data[cal_cycle]
    test_df = all_cycles_data[test_cycle]
    
    print(f"Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}")
    
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
    
    # Reference cohort (non-diabetic from training)
    ref_cohort = train_df[train_df['known_diabetes'] == 0].copy()
    
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
    
    def build_embeddings(df, ref_cohort):
        embeddings = pd.DataFrame(index=df.index)
        embeddings['subject_id'] = df['SEQN']
        embeddings['subgroup'] = df['Race_Ethnicity'].astype(str)
        embeddings['y_self'] = df['known_diabetes']
        embeddings['y_lab'] = df['lab_diabetes']
        
        z_cols = []
        for feat, spec in CONTINUOUS_FEATURES.items():
            if feat in df.columns:
                trans = ecdf_transform(df[feat], ref_cohort[feat])
                col = f'z_{feat}'
                embeddings[col] = trans
                z_cols.append(col)
        
        for feat in BINARY_FEATURES:
            if feat in df.columns:
                col = f'z_{feat}'
                embeddings[col] = df[feat].astype(float)
                z_cols.append(col)
        
        for feat in CATEGORICAL_FEATURES:
            if feat in df.columns:
                dummies = pd.get_dummies(df[feat], prefix=f'z_{feat}')
                for c in dummies.columns:
                    embeddings[c] = dummies[c].astype(float)
                    z_cols.append(c)
        
        embeddings[z_cols] = embeddings[z_cols].fillna(0.0)
        return embeddings, z_cols
    
    train_emb, z_cols = build_embeddings(train_df, ref_cohort)
    cal_emb, _ = build_embeddings(cal_df, ref_cohort)
    test_emb, _ = build_embeddings(test_df, ref_cohort)
    
    X_train, y_train = train_emb[z_cols].values, train_emb['y_self'].values
    X_cal, y_cal = cal_emb[z_cols].values, cal_emb['y_self'].values
    X_test, y_test_self = test_emb[z_cols].values, test_emb['y_self'].values
    y_test_lab = test_emb['y_lab'].values
    sg_test = test_emb['subgroup'].values
    
    # Train model
    base_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs'))
    ])
    base_clf.fit(X_train, y_train)
    
    # Platt
    logits_platt = base_clf.decision_function(cal_emb[z_cols].values).reshape(-1, 1)
    platt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_lr.fit(logits_platt, y_cal)
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
    
    # Evaluate on test
    p_test = predict_calibrated(test_emb[z_cols].values, w_cal, b_cal)
    
    auc_self = roc_auc_score(y_test_self, p_test)
    auc_lab = roc_auc_score(y_test_lab, p_test)
    prev_self = y_test_self.mean()
    prev_lab = y_test_lab.mean()
    
    print(f"\nTemporal Validation Results:")
    print(f"  Self-reported AUC: {auc_self:.4f}, Prevalence: {prev_self:.1%}")
    print(f"  Lab-verified AUC:  {auc_lab:.4f}, Prevalence: {prev_lab:.1%}")
    
    temporal_results = {
        'train_n': len(train_df),
        'cal_n': len(cal_df),
        'test_n': len(test_df),
        'self_reported_auc': float(auc_self),
        'lab_verified_auc': float(auc_lab),
        'prevalence_self': float(prev_self),
        'prevalence_lab': float(prev_lab),
    }
else:
    print("Required cycles not all available - temporal validation skipped")
    temporal_results = {'error': 'Required cycles not available'}

# ============================================================
# LAB-VERIFIED TARGET SENSITIVITY (on 2017-2018)
# ============================================================
print("\n" + "=" * 60)
print("LAB-VERIFIED TARGET SENSITIVITY (2017-2018)")
print("=" * 60)

if '2017-2018' in all_cycles_data:
    df = all_cycles_data['2017-2018'].copy()
    df = df.dropna(subset=['known_diabetes', 'lab_diabetes', 'HbA1c', 'FPG'] + 
                    ['BMI', 'Waist_cm', 'SBP', 'DBP', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3'])
    
    print(f"Complete cases: {len(df)}")
    print(f"Self-reported prevalence: {df['known_diabetes'].mean():.1%}")
    print(f"Lab-verified prevalence: {df['lab_diabetes'].mean():.1%}")
    
    # Train on self-reported vs lab-verified
    ref_cohort = df[df['known_diabetes'] == 0].copy()
    
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
    
    def build_embeddings(df, ref_cohort):
        embeddings = pd.DataFrame(index=df.index)
        embeddings['subject_id'] = df['SEQN']
        embeddings['subgroup'] = df['Race_Ethnicity'].astype(str)
        embeddings['y_self'] = df['known_diabetes']
        embeddings['y_lab'] = df['lab_diabetes']
        
        z_cols = []
        for feat, spec in CONTINUOUS_FEATURES.items():
            if feat in df.columns:
                trans = ecdf_transform(df[feat], ref_cohort[feat])
                col = f'z_{feat}'
                embeddings[col] = trans
                z_cols.append(col)
        
        for feat in BINARY_FEATURES:
            if feat in df.columns:
                col = f'z_{feat}'
                embeddings[col] = df[feat].astype(float)
                z_cols.append(col)
        
        for feat in CATEGORICAL_FEATURES:
            if feat in df.columns:
                dummies = pd.get_dummies(df[feat], prefix=f'z_{feat}')
                for c in dummies.columns:
                    embeddings[c] = dummies[c].astype(float)
                    z_cols.append(c)
        
        embeddings[z_cols] = embeddings[z_cols].fillna(0.0)
        return embeddings, z_cols
    
    # 4-way split
    embeddings, z_cols = build_embeddings(df, ref_cohort)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(sss.split(embeddings[z_cols].values, embeddings['y_self'].values))
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.375, random_state=42)
    train_idx, val_idx = next(sss2.split(embeddings[z_cols].values[trainval_idx], embeddings['y_self'].values[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]
    
    sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    platt_idx, conf_audit_idx = next(sss3.split(embeddings[z_cols].values[val_idx], embeddings['y_self'].values[val_idx]))
    platt_idx = val_idx[platt_idx]
    conf_audit_idx = val_idx[conf_audit_idx]
    
    X_train = embeddings[z_cols].values[train_idx]
    y_train_self = embeddings['y_self'].values[train_idx]
    y_train_lab = embeddings['y_lab'].values[train_idx]
    
    X_platt = embeddings[z_cols].values[platt_idx]
    y_platt_self = embeddings['y_self'].values[platt_idx]
    y_platt_lab = embeddings['y_lab'].values[platt_idx]
    
    X_test = embeddings[z_cols].values[test_idx]
    y_test_self = embeddings['y_self'].values[test_idx]
    y_test_lab = embeddings['y_lab'].values[test_idx]
    
    # Train on self-reported
    base_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs'))
    ])
    base_clf.fit(X_train, y_train_self)
    
    # Platt
    logits_platt = base_clf.decision_function(X_platt).reshape(-1, 1)
    platt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_lr.fit(logits_platt, y_platt_self)
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
    
    p_test = predict_calibrated(embeddings[z_cols].values[test_idx], w_cal, b_cal)
    
    auc_self = roc_auc_score(embeddings['y_self'].values[test_idx], p_test)
    auc_lab = roc_auc_score(embeddings['y_lab'].values[test_idx], p_test)
    
    print(f"\nLab-Verified Target Sensitivity:")
    print(f"  Self-reported AUC: {auc_self:.4f}")
    print(f"  Lab-verified AUC:  {auc_lab:.4f}")
    
    lab_results = {
        'self_reported_auc': float(auc_self),
        'lab_verified_auc': float(auc_lab),
    }
else:
    lab_results = {'error': '2017-2018 data not available'}

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'temporal_validation': temporal_results,
    'lab_verified_sensitivity': lab_results,
    'available_cycles': list(all_cycles_data.keys()),
}

with open(RESULTS_DIR / 'multi_cycle_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("MULTI-CYCLE VALIDATION COMPLETE")
print("=" * 60)
print(json.dumps(results, indent=2, default=str))