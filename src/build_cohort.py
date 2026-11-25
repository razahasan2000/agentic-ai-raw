"""
Cohort Construction for Incident T2D Prediction (MIMIC-IV Demo)
================================================================
Target: Predict incident T2D within 365 days of index hospital admission
Cohort: Adults (≥18) with no prior T2D diagnosis/medication/HbA1c≥6.5%
Index: First eligible hospital admission
Outcome: T2D diagnosis (E11.*) OR diabetes medication OR HbA1c≥6.5% within 365 days post-discharge
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")

# ============================================================
# 1. LOAD CORE TABLES
# ============================================================
print("Loading core tables...")
patients = pd.read_csv(DATA_DIR / "patients.csv")
admissions = pd.read_csv(DATA_DIR / "admissions.csv")
diagnoses = pd.read_csv(DATA_DIR / "diagnoses_icd.csv")
labevents = pd.read_csv(DATA_DIR / "labevents.csv")
prescriptions = pd.read_csv(DATA_DIR / "prescriptions.csv")
omr = pd.read_csv(DATA_DIR / "omr.csv")
d_labitems = pd.read_csv(DATA_DIR / "d_labitems.csv")

# Parse dates
for df, cols in [(admissions, ['admittime', 'dischtime', 'deathtime']),
                  (labevents, ['charttime', 'storetime']),
                  (prescriptions, ['starttime', 'stoptime']),
                  (omr, ['chartdate'])]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

# ============================================================
# 2. DEFINE T2D PHENOTYPE (eMERGE-inspired)
# ============================================================
# ICD-10: E11.* = Type 2 Diabetes Mellitus
T2D_ICD10_PREFIX = 'E11'

# Diabetes medications (generic names)
DM_DRUGS = {
    'metformin', 'insulin', 'glipizide', 'glyburide', 'glimepiride',
    'pioglitazone', 'rosiglitazone', 'sitagliptin', 'saxagliptin',
    'linagliptin', 'alogliptin', 'exenatide', 'liraglutide',
    'dulaglutide', 'semaglutide', 'canagliflozin', 'dapagliflozin',
    'empagliflozin', 'ertugliflozin', 'repaglinide', 'nateglinide',
    'acarbose', 'miglitol', 'chlorpropamide', 'tolazamide', 'tolbutamide'
}

# Key lab item IDs
LAB_ITEMS = {
    'hba1c': [50852],           # % Hemoglobin A1c
    'glucose_blood_gas': [50809, 52027],  # Glucose (blood gas)
    'glucose_chem': [50931, 52569, 273, 302],  # Glucose (chemistry)
    'fructosamine': [51628, 51629],
    'glycated_hb': [51631],
}

# HbA1c threshold for diabetes (≥6.5% = 48 mmol/mol)
HBA1C_THRESHOLD = 6.5  # %

# ============================================================
# 3. IDENTIFY PATIENTS WITH PRIOR T2D (before each admission)
# ============================================================
print("Identifying prior T2D...")

# T2D diagnoses
t2d_dx = diagnoses[diagnoses['icd_code'].str.startswith(T2D_ICD10_PREFIX, na=False)].copy()
t2d_dx = t2d_dx.merge(admissions[['hadm_id', 'admittime']], on='hadm_id', how='left')
t2d_dx = t2d_dx.rename(columns={'admittime': 'dx_time'})

# Diabetes medications
dm_rx = prescriptions[prescriptions['drug'].str.lower().isin(DM_DRUGS)].copy()
dm_rx = dm_rx.rename(columns={'starttime': 'rx_time'})

# HbA1c ≥ 6.5%
hba1c_high = labevents[labevents['itemid'].isin(LAB_ITEMS['hba1c']) & 
                        (labevents['valuenum'] >= HBA1C_THRESHOLD)].copy()
hba1c_high = hba1c_high.rename(columns={'charttime': 'lab_time'})

# Combine all T2D evidence
def get_prior_t2d_cutoff(subject_id, index_time, t2d_dx, dm_rx, hba1c_high):
    """Return earliest T2D evidence time for a patient before index_time"""
    times = []
    for df, col in [(t2d_dx, 'dx_time'), (dm_rx, 'rx_time'), (hba1c_high, 'lab_time')]:
        pt_times = df[(df['subject_id'] == subject_id) & (df[col] < index_time)][col]
        if len(pt_times) > 0:
            times.append(pt_times.min())
    return min(times) if times else pd.NaT

# ============================================================
# 4. BUILD COHORT: First eligible admission per patient
# ============================================================
print("Building cohort...")

# Adult patients only (anchor_age ≥ 18 at admission)
admissions = admissions.merge(patients[['subject_id', 'anchor_age', 'gender', 'anchor_year']], on='subject_id')
admissions['age_at_admit'] = admissions['anchor_age'] + (admissions['admittime'].dt.year - admissions['anchor_year'])
adult_adm = admissions[admissions['age_at_admit'] >= 18].copy()

# Sort by subject_id, admittime
adult_adm = adult_adm.sort_values(['subject_id', 'admittime']).reset_index(drop=True)

# For each admission, check if patient had prior T2D
print("Checking prior T2D for each admission...")
prior_t2d = []
for _, row in adult_adm.iterrows():
    cutoff = get_prior_t2d_cutoff(row['subject_id'], row['admittime'], t2d_dx, dm_rx, hba1c_high)
    prior_t2d.append(cutoff)

adult_adm['prior_t2d_time'] = prior_t2d
adult_adm['has_prior_t2d'] = adult_adm['prior_t2d_time'].notna()

# First eligible admission (no prior T2D)
eligible = adult_adm[~adult_adm['has_prior_t2d']].copy()
first_adm = eligible.drop_duplicates(subset='subject_id', keep='first').copy()
first_adm = first_adm.rename(columns={'hadm_id': 'index_hadm_id', 'admittime': 'index_admittime', 
                                       'dischtime': 'index_dischtime', 'age_at_admit': 'index_age'})

print(f"Eligible patients (no prior T2D): {len(first_adm)}")

# ============================================================
# 5. DEFINE OUTCOME: Incident T2D within 365 days post-discharge
# ============================================================
print("Defining outcomes...")

# Outcome window: (dischtime, dischtime + 365 days]
first_adm['outcome_start'] = first_adm['index_dischtime']
first_adm['outcome_end'] = first_adm['index_dischtime'] + pd.Timedelta(days=365)

# Check for T2D diagnosis in outcome window
outcome_dx = t2d_dx.merge(first_adm[['subject_id', 'outcome_start', 'outcome_end']], on='subject_id')
outcome_dx = outcome_dx[(outcome_dx['dx_time'] > outcome_dx['outcome_start']) & 
                         (outcome_dx['dx_time'] <= outcome_dx['outcome_end'])]
dx_outcome = outcome_dx.groupby('subject_id')['dx_time'].min().reset_index()
dx_outcome.columns = ['subject_id', 't2d_dx_time']

# Check for diabetes medication in outcome window
outcome_rx = dm_rx.merge(first_adm[['subject_id', 'outcome_start', 'outcome_end']], on='subject_id')
outcome_rx = outcome_rx[(outcome_rx['rx_time'] > outcome_rx['outcome_start']) & 
                         (outcome_rx['rx_time'] <= outcome_rx['outcome_end'])]
rx_outcome = outcome_rx.groupby('subject_id')['rx_time'].min().reset_index()
rx_outcome.columns = ['subject_id', 't2d_rx_time']

# Check for HbA1c ≥ 6.5% in outcome window
outcome_hba1c = hba1c_high.merge(first_adm[['subject_id', 'outcome_start', 'outcome_end']], on='subject_id')
outcome_hba1c = outcome_hba1c[(outcome_hba1c['lab_time'] > outcome_hba1c['outcome_start']) & 
                               (outcome_hba1c['lab_time'] <= outcome_hba1c['outcome_end'])]
hba1c_outcome = outcome_hba1c.groupby('subject_id')['lab_time'].min().reset_index()
hba1c_outcome.columns = ['subject_id', 't2d_hba1c_time']

# Combine outcomes
outcomes = first_adm[['subject_id', 'index_hadm_id', 'index_admittime', 'index_dischtime', 
                       'index_age', 'gender', 'outcome_start', 'outcome_end']].copy()
outcomes = outcomes.merge(dx_outcome, on='subject_id', how='left')
outcomes = outcomes.merge(rx_outcome, on='subject_id', how='left')
outcomes = outcomes.merge(hba1c_outcome, on='subject_id', how='left')

# Earliest T2D evidence
outcomes['t2d_time'] = outcomes[['t2d_dx_time', 't2d_rx_time', 't2d_hba1c_time']].min(axis=1)
outcomes['incident_t2d'] = outcomes['t2d_time'].notna()
outcomes['time_to_t2d'] = (outcomes['t2d_time'] - outcomes['outcome_start']).dt.days

# Censoring: death or end of follow-up
outcomes['censor_time'] = outcomes[['outcome_end', 'deathtime']].min(axis=1) if 'deathtime' in outcomes.columns else outcomes['outcome_end']
outcomes['followup_days'] = (outcomes['censor_time'] - outcomes['outcome_start']).dt.days

print(f"\nCohort Summary:")
print(f"  Total patients: {len(outcomes)}")
print(f"  Incident T2D: {outcomes['incident_t2d'].sum()} ({outcomes['incident_t2d'].mean()*100:.1f}%)")
print(f"  Median followup: {outcomes['followup_days'].median():.0f} days")

# ============================================================
# 6. EXTRACT FEATURES AT INDEX ADMISSION (and prior 1 year)
# ============================================================
print("\nExtracting features...")

# Lookback window: 1 year before index admission
first_adm['lookback_start'] = first_adm['index_admittime'] - pd.Timedelta(days=365)

# --- Demographics ---
features = first_adm[['subject_id', 'index_hadm_id', 'index_age', 'gender']].copy()
features = features.rename(columns={'index_age': 'age', 'index_hadm_id': 'hadm_id'})

# --- Comorbidities (Charlson/Deyo) from diagnoses at/before index ---
# Simplified: key comorbidities relevant to diabetes
COMORBIDITY_ICD10 = {
    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15'],
    'ckd': ['N18', 'N19'],
    'cvd': ['I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I50', 'I63', 'I64', 'I65', 'I66', 'I67'],
    'obesity': ['E66'],
    'dyslipidemia': ['E78'],
    'gestational_dm': ['O24'],  # gestational diabetes (exclusion)
}

# Need charttime in diagnoses - merge with admissions
dx_with_time = diagnoses.merge(admissions[['hadm_id', 'admittime']], on='hadm_id', how='left')
dx_with_time = dx_with_time.rename(columns={'admittime': 'charttime'})

for name, prefixes in COMORBIDITY_ICD10.items():
    mask = dx_with_time['icd_code'].str.startswith(tuple(prefixes), na=False)
    com = dx_with_time[mask].merge(first_adm[['subject_id', 'lookback_start', 'index_admittime']], on='subject_id')
    com = com[(com['charttime'] >= com['lookback_start']) & (com['charttime'] <= com['index_admittime'])]
    features[f'comorb_{name}'] = features['subject_id'].isin(com['subject_id'].unique()).astype(int)

# --- Labs: most recent value in lookback window ---
# Pre-compute lookback_start and index_admittime per subject
lookback_map = first_adm.set_index('subject_id')[['lookback_start', 'index_admittime']]

def get_latest_lab(subject_id, itemids, lookback_start, index_time, labevents):
    """Get most recent lab value in lookback window"""
    labs = labevents[(labevents['subject_id'] == subject_id) & 
                      (labevents['itemid'].isin(itemids)) &
                      (labevents['charttime'] >= lookback_start) &
                      (labevents['charttime'] <= index_time) &
                      (labevents['valuenum'].notna())]
    if len(labs) == 0:
        return np.nan
    # Use most recent
    labs = labs.sort_values('charttime')
    return labs.iloc[-1]['valuenum']

# For each lab category, get latest value
for lab_name, itemids in LAB_ITEMS.items():
    features[f'lab_{lab_name}'] = features.apply(
        lambda r: get_latest_lab(r['subject_id'], itemids, 
                                  lookback_map.loc[r['subject_id'], 'lookback_start'],
                                  lookback_map.loc[r['subject_id'], 'index_admittime'],
                                  labevents), axis=1)

# --- Vitals from OMR (BMI, Weight, Height, BP) ---
omr_clean = omr.copy()
omr_clean['chartdate'] = pd.to_datetime(omr_clean['chartdate'], errors='coerce')
omr_clean['result_value'] = pd.to_numeric(omr_clean['result_value'], errors='coerce')

for vital_name in ['BMI (kg/m2)', 'Weight (Lbs)', 'Height (Inches)', 'Blood Pressure']:
    vit = omr_clean[omr_clean['result_name'] == vital_name].copy()
    vit = vit.merge(first_adm[['subject_id', 'lookback_start', 'index_admittime']], on='subject_id')
    vit = vit[(vit['chartdate'] >= vit['lookback_start']) & (vit['chartdate'] <= vit['index_admittime'])]
    vit = vit.sort_values('chartdate').drop_duplicates('subject_id', keep='last')
    vital_col = vital_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    features = features.merge(vit[['subject_id', 'result_value']].rename(columns={'result_value': f'vital_{vital_col}'}), 
                              on='subject_id', how='left')

# --- Medications: diabetes drug exposure in lookback ---
dm_rx_lookback = dm_rx.merge(first_adm[['subject_id', 'lookback_start', 'index_admittime']], on='subject_id')
dm_rx_lookback = dm_rx_lookback[(dm_rx_lookback['rx_time'] >= dm_rx_lookback['lookback_start']) & 
                                 (dm_rx_lookback['rx_time'] <= dm_rx_lookback['index_admittime'])]
features['prior_dm_drug'] = features['subject_id'].isin(dm_rx_lookback['subject_id'].unique()).astype(int)

# --- Admission characteristics ---
adm_chars = first_adm[['subject_id', 'admission_type', 'admission_location', 'insurance', 'race']].copy()
features = features.merge(adm_chars, on='subject_id', how='left')

# ============================================================
# 7. SUBGROUP DEFINITIONS (for fairness auditing)
# ============================================================
print("Defining subgroups...")

# Age bands
features['age_band'] = pd.cut(features['age'], bins=[18, 40, 50, 60, 70, 80, 100], 
                               labels=['18-39', '40-49', '50-59', '60-69', '70-79', '80+'])

# Charlson comorbidity index (simplified)
comorb_cols = [c for c in features.columns if c.startswith('comorb_')]
features['comorb_count'] = features[comorb_cols].sum(axis=1)
features['comorb_band'] = pd.cut(features['comorb_count'], bins=[-1, 0, 1, 2, 10], 
                                  labels=['0', '1', '2', '3+'])

# Race/ethnicity (simplified)
features['race_group'] = features['race'].fillna('Unknown')

# Insurance type
features['insurance_group'] = features['insurance'].fillna('Unknown')

# Combine with outcomes
cohort = features.merge(outcomes[['subject_id', 'incident_t2d', 'time_to_t2d', 'followup_days', 't2d_time']], on='subject_id')

# ============================================================
# 8. SAVE COHORT
# ============================================================
output_path = DATA_DIR / "cohort_incident_t2d.csv"
cohort.to_csv(output_path, index=False)
print(f"\nCohort saved to {output_path}")
print(f"Shape: {cohort.shape}")
print(f"\nFeature columns ({len(cohort.columns)}):")
for c in sorted(cohort.columns):
    print(f"  {c}")

# Subgroup distribution
print("\n=== Subgroup Distributions ===")
for sg in ['age_band', 'gender', 'comorb_band', 'race_group', 'insurance_group']:
    if sg in cohort.columns:
        print(f"\n{sg}:")
        print(cohort[sg].value_counts(dropna=False))

# Missing data summary
print("\n=== Missing Data (% missing) ===")
missing = cohort.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].to_string())

print("\nDone!")