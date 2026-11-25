"""
Clinically Normalized Risk-Factor Embedding (Section 4.1)
==========================================================
Maps raw risk factors into a space where distances are commensurable:
- Continuous: z_j = Phi^-1(F_j^ref(x_j))  [standard normal quantile of reference CDF]
- Categorical/Ordinal: learned monotone encoding with clinically meaningful units
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")
cohort = pd.read_csv(DATA_DIR / "cohort_incident_t2d.csv")

# ============================================================
# 1. DEFINE FEATURE SPECIFICATIONS
# ============================================================
# Continuous features with clinical reference ranges
CONTINUOUS_FEATURES = {
    'age': {'type': 'continuous', 'unit': 'years', 'ref_range': (18, 100)},
    'lab_hba1c': {'type': 'continuous', 'unit': '%', 'ref_range': (4.0, 14.0)},
    'lab_glucose_chem': {'type': 'continuous', 'unit': 'mg/dL', 'ref_range': (50, 400)},
    'lab_glucose_blood_gas': {'type': 'continuous', 'unit': 'mg/dL', 'ref_range': (50, 400)},
    'lab_fructosamine': {'type': 'continuous', 'unit': 'umol/L', 'ref_range': (100, 600)},
    'lab_glycated_hb': {'type': 'continuous', 'unit': '%', 'ref_range': (4.0, 14.0)},
    'vital_bmi_kg_m2': {'type': 'continuous', 'unit': 'kg/m2', 'ref_range': (10, 60)},
    'vital_weight_lbs': {'type': 'continuous', 'unit': 'lbs', 'ref_range': (50, 500)},
    'vital_height_inches': {'type': 'continuous', 'unit': 'inches', 'ref_range': (48, 84)},
    'comorb_count': {'type': 'continuous', 'unit': 'count', 'ref_range': (0, 10)},
}

# Ordinal/Categorical features with clinical ordering
ORDINAL_FEATURES = {
    'age_band': {
        'type': 'ordinal',
        'categories': ['18-39', '40-49', '50-59', '60-69', '70-79', '80+'],
        'clinical_unit': 'decade'  # 1 unit = 1 decade
    },
    'comorb_band': {
        'type': 'ordinal',
        'categories': ['0', '1', '2', '3+'],
        'clinical_unit': 'comorbidity'
    },
    'gender': {
        'type': 'binary',
        'categories': ['F', 'M'],
        'clinical_unit': 'sex'
    },
    'race_group': {
        'type': 'categorical',
        'categories': None,  # will infer
        'clinical_unit': 'race'
    },
    'insurance_group': {
        'type': 'categorical',
        'categories': None,
        'clinical_unit': 'insurance'
    },
}

# Features to exclude from embedding (outcomes, IDs, etc.)
EXCLUDE = {'subject_id', 'hadm_id', 'incident_t2d', 'time_to_t2d', 't2d_time', 
           'followup_days', 'race', 'insurance', 'admission_type', 'admission_location'}

# ============================================================
# 2. REFERENCE COHORT FOR CDF ESTIMATION
# ============================================================
# Use the training cohort (non-cases) as reference, or full cohort
# For demo, use full cohort; in practice use separate reference population
ref_cohort = cohort.copy()  # In practice: cohort[~cohort['incident_t2d']]

print(f"Reference cohort size: {len(ref_cohort)}")

# ============================================================
# 3. CONTINUOUS: EMPIRICAL CDF + INVERSE NORMAL TRANSFORM
# ============================================================
def ecdf_transform(series, ref_series, clip_quantile=0.001):
    """
    z = Phi^-1(F_ref(x))
    where F_ref is empirical CDF of reference cohort
    """
    # Clip extreme values to avoid infinities
    ref_clean = ref_series.dropna()
    if len(ref_clean) < 10:
        return pd.Series(np.nan, index=series.index)
    
    # Empirical CDF using rank / (n+1) to avoid 0/1
    n = len(ref_clean)
    sorted_ref = np.sort(ref_clean)
    
    def transform(x):
        if pd.isna(x):
            return np.nan
        # Rank in reference distribution
        rank = np.searchsorted(sorted_ref, x, side='right')
        p = rank / (n + 1)
        # Clip to avoid +/- inf
        p = np.clip(p, clip_quantile, 1 - clip_quantile)
        return norm.ppf(p)
    
    return series.apply(transform)

print("\nTransforming continuous features...")
embeddings = pd.DataFrame(index=cohort.index)
embeddings['subject_id'] = cohort['subject_id']

for feat, spec in CONTINUOUS_FEATURES.items():
    if feat in cohort.columns:
        ref_vals = ref_cohort[feat]
        trans = ecdf_transform(cohort[feat], ref_vals)
        embeddings[f'z_{feat}'] = trans
        # Report coverage
        non_null = trans.notna().sum()
        print(f"  {feat}: {non_null}/{len(cohort)} non-null, range=[{trans.min():.2f}, {trans.max():.2f}]")

# ============================================================
# 4. ORDINAL: MONOTONE ENCODING WITH CLINICAL UNITS
# ============================================================
def ordinal_encode(series, categories, unit_scale=1.0):
    """
    Map ordered categories to equally spaced values.
    Unit increment = 1 clinical unit (e.g., 1 decade, 1 comorbidity)
    """
    cat_to_val = {cat: i * unit_scale for i, cat in enumerate(categories)}
    return series.map(cat_to_val)

print("\nTransforming ordinal/categorical features...")
for feat, spec in ORDINAL_FEATURES.items():
    if feat in cohort.columns:
        if spec['type'] == 'ordinal' and spec['categories']:
            encoded = ordinal_encode(cohort[feat], spec['categories'], unit_scale=1.0)
            embeddings[f'z_{feat}'] = encoded
            print(f"  {feat}: {encoded.notna().sum()}/{len(cohort)} non-null")
        elif spec['type'] == 'binary':
            # Binary: 0/1 encoding
            cats = spec['categories']
            encoded = cohort[feat].map({cats[0]: 0.0, cats[1]: 1.0})
            embeddings[f'z_{feat}'] = encoded
            print(f"  {feat}: {encoded.notna().sum()}/{len(cohort)} non-null")
        elif spec['type'] == 'categorical':
            # For categorical, use one-hot or target encoding
            # Here: leave for later (subgroup indicator, not embedding)
            pass

# ============================================================
# 5. SUBGROUP INDICATORS (not embedded, used for conditioning)
# ============================================================
subgroup_cols = ['age_band', 'gender', 'comorb_band', 'race_group', 'insurance_group']
for sg in subgroup_cols:
    if sg in cohort.columns:
        embeddings[f'subgroup_{sg}'] = cohort[sg].astype(str)

# ============================================================
# 6. STANDARDIZE EMBEDDING (zero mean, unit variance per feature)
# ============================================================
# Only standardize continuous embeddings (z_* columns)
z_cols = [c for c in embeddings.columns if c.startswith('z_')]
for col in z_cols:
    mean = embeddings[col].mean()
    std = embeddings[col].std()
    if std > 0:
        embeddings[col] = (embeddings[col] - mean) / std
    else:
        embeddings[col] = 0.0

# ============================================================
# 7. HANDLE MISSING VALUES (impute with 0 = reference median)
# ============================================================
# In the normalized space, 0 = median of reference cohort
embeddings[z_cols] = embeddings[z_cols].fillna(0.0)

# ============================================================
# 8. SAVE EMBEDDING
# ============================================================
output_path = DATA_DIR / "embeddings.csv"
embeddings.to_csv(output_path, index=False)
print(f"\nEmbeddings saved to {output_path}")
print(f"Shape: {embeddings.shape}")
print(f"Embedding columns ({len(z_cols)}): {z_cols}")
print(f"Subgroup columns: {[c for c in embeddings.columns if c.startswith('subgroup_')]}")

# Quick stats
print("\nEmbedding statistics:")
print(embeddings[z_cols].describe().T[['mean', 'std', 'min', 'max']].to_string())