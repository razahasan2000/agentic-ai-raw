"""
NHANES 2017-2018 Diabetes Cohort Construction
=============================================
Builds incident T2D prediction cohort from NHANES 2017-2018 data.
"""

import pyreadstat
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\nhanes_2017_2020")

def load_nhanes_table(name):
    """Load NHANES XPT table"""
    df, _ = pyreadstat.read_xport(DATA_DIR / f"{name}.XPT")
    return df

def main():
    print("=" * 60)
    print("NHANES 2017-2018 DIABETES COHORT CONSTRUCTION")
    print("=" * 60)
    
    # 1. Load core tables
    print("\n1. Loading tables...")
    demo = load_nhanes_table("DEMO_J")
    diq = load_nhanes_table("DIQ_J")
    ghb = load_nhanes_table("GHB_J")
    glu = load_nhanes_table("GLU_J")
    bmx = load_nhanes_table("BMX_J")
    bpx = load_nhanes_table("BPX_J")
    inq = load_nhanes_table("INQ_J")
    smq = load_nhanes_table("SMQ_J")
    alq = load_nhanes_table("ALQ_J")
    paq = load_nhanes_table("PAQ_J")
    
    # 2. Define diabetes status
    # DIQ010: 1=Yes, 2=No, 3=Borderline, 9=Don't know
    diq['has_diabetes'] = (diq['DIQ010'] == 1).astype(int)
    diq['diabetes_borderline'] = (diq['DIQ010'] == 3).astype(int)
    
    # Age at diagnosis
    diq['age_diagnosis'] = diq['DID040']
    
    # 3. Merge all tables on SEQN
    print("\n2. Merging tables...")
    cohort = demo.merge(diq[['SEQN', 'has_diabetes', 'diabetes_borderline', 'age_diagnosis', 
                              'DIQ160', 'DIQ170', 'DIQ180', 'DIQ230', 'DIQ240', 'DIQ275']], 
                        on='SEQN', how='left')
    
    # Add HbA1c
    cohort = cohort.merge(ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')
    cohort = cohort.rename(columns={'LBXGH': 'HbA1c'})
    
    # Add Glucose (fasting)
    glu_fasting = glu[glu['LBDGLUSI'] == 1][['SEQN', 'LBXGLU']].rename(columns={'LBXGLU': 'Glucose'})
    cohort = cohort.merge(glu_fasting, on='SEQN', how='left')
    
    # Add Body measures
    cohort = cohort.merge(bmx[['SEQN', 'BMXBMI', 'BMXWT', 'BMXHT', 'BMXWAIST']], on='SEQN', how='left')
    cohort = cohort.rename(columns={'BMXBMI': 'BMI', 'BMXWT': 'Weight_kg', 'BMXHT': 'Height_cm', 'BMXWAIST': 'Waist_cm'})
    
    # Add Blood pressure (average of 2 readings)
    bpx['SBP'] = (bpx['BPXSY1'] + bpx['BPXSY2']) / 2
    bpx['DBP'] = (bpx['BPXDI1'] + bpx['BPXDI2']) / 2
    cohort = cohort.merge(bpx[['SEQN', 'SBP', 'DBP']], on='SEQN', how='left')
    
    # Income
    cohort = cohort.merge(inq[['SEQN', 'INDFMMPC']], on='SEQN', how='left')
    cohort = cohort.rename(columns={'INDFMMPC': 'Income_Poverty_Ratio'})
    
    # Smoking
    smq['Current_Smoker'] = (smq['SMQ020'] == 1).astype(int)
    smq['Former_Smoker'] = (smq['SMQ020'] == 2).astype(int)
    cohort = cohort.merge(smq[['SEQN', 'Current_Smoker', 'Former_Smoker']], on='SEQN', how='left')
    
    # Alcohol (ALQ111: 1=Yes, 2=No)
    alq['Current_Drinker'] = (alq['ALQ111'] == 1).astype(int)
    cohort = cohort.merge(alq[['SEQN', 'Current_Drinker']], on='SEQN', how='left')
    
    # Physical activity
    paq['Vigorous_Activity'] = (paq['PAQ605'] == 1).astype(int)
    paq['Moderate_Activity'] = (paq['PAQ620'] == 1).astype(int)
    cohort = cohort.merge(paq[['SEQN', 'Vigorous_Activity', 'Moderate_Activity']], on='SEQN', how='left')
    
    # 4. Filter: Adults (>=18), examined (RIDSTATR==2)
    print("\n3. Filtering cohort...")
    cohort = cohort[(cohort['RIDAGEYR'] >= 18) & (cohort['RIDSTATR'] == 2)].copy()
    print(f"   Adults examined: {len(cohort):,}")
    
    # 5. Define diabetes status
    # Known diabetes: DIQ010==1 OR HbA1c>=6.5% OR Glucose>=126
    cohort['known_diabetes'] = (
        (cohort['has_diabetes'] == 1) | 
        (cohort['HbA1c'] >= 6.5) | 
        (cohort['Glucose'] >= 126)
    ).astype(int)
    
    # Prediabetes: HbA1c 5.7-6.4% OR Glucose 100-125 (no known diabetes)
    cohort['prediabetes'] = (
        (cohort['known_diabetes'] == 0) & 
        (((cohort['HbA1c'] >= 5.7) & (cohort['HbA1c'] < 6.5)) | 
         ((cohort['Glucose'] >= 100) & (cohort['Glucose'] < 126)))
    ).astype(int)
    
    # Normal: neither
    cohort['normal_glucose'] = ((cohort['known_diabetes'] == 0) & (cohort['prediabetes'] == 0)).astype(int)
    
    # 6. Define outcome: Incident T2D (for those without baseline diabetes)
    # In cross-sectional NHANES, we can't do true incident prediction
    # But we can use prevalent diabetes as outcome for model development
    cohort['outcome_diabetes'] = cohort['known_diabetes']
    cohort['outcome_prediabetes'] = cohort['prediabetes']
    
    # 7. Feature engineering
    print("\n4. Feature engineering...")
    
    # Age groups
    cohort['Age_Group'] = pd.cut(cohort['RIDAGEYR'], 
                                  bins=[18, 30, 40, 50, 60, 70, 80, 100],
                                  labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])
    
    # BMI categories
    cohort['BMI_Cat'] = pd.cut(cohort['BMI'],
                                bins=[0, 18.5, 25, 30, 35, 40, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III'])
    
    # Race/Ethnicity labels
    race_map = {1: 'Mexican_American', 2: 'Other_Hispanic', 3: 'White', 4: 'Black', 6: 'Asian', 7: 'Other'}
    cohort['Race_Ethnicity'] = cohort['RIDRETH3'].map(race_map)
    
    # Gender labels
    cohort['Gender'] = cohort['RIAGENDR'].map({1: 'Male', 2: 'Female'})
    
    # Income categories
    cohort['Income_Cat'] = pd.cut(cohort['Income_Poverty_Ratio'],
                                   bins=[-1, 1, 2, 3, 4, 5, np.inf],
                                   labels=['<100%', '100-199%', '200-299%', '300-399%', '400-499%', '500%+'])
    
    # Smoking status
    cohort['Smoking_Status'] = 'Never'
    cohort.loc[cohort['Current_Smoker'] == 1, 'Smoking_Status'] = 'Current'
    cohort.loc[cohort['Former_Smoker'] == 1, 'Smoking_Status'] = 'Former'
    
    # 8. Select features for modeling
    feature_cols = [
        'RIDAGEYR', 'Gender', 'Race_Ethnicity', 'INDFMPIR', 'Income_Cat',
        'BMI', 'Waist_cm', 'SBP', 'DBP', 'HbA1c', 'Glucose',
        'Current_Smoker', 'Former_Smoker', 'Smoking_Status',
        'Current_Drinker', 'Vigorous_Activity', 'Moderate_Activity',
        'Age_Group', 'BMI_Cat', 'Race_Ethnicity', 'Income_Cat', 'Smoking_Status'
    ]
    
    # Target
    target_col = 'known_diabetes'
    
    # 9. Summary statistics
    print("\n5. Cohort Summary:")
    print(f"   Total adults examined: {len(cohort):,}")
    print(f"   Known diabetes: {cohort['known_diabetes'].sum():,} ({cohort['known_diabetes'].mean()*100:.1f}%)")
    print(f"   Prediabetes: {cohort['prediabetes'].sum():,} ({cohort['prediabetes'].mean()*100:.1f}%)")
    print(f"   Normal glucose: {cohort['normal_glucose'].sum():,} ({cohort['normal_glucose'].mean()*100:.1f}%)")
    
    print("\n   By Age Group:")
    for ag, sub in cohort.groupby('Age_Group', observed=False):
        if len(sub) > 0:
            print(f"     {ag}: n={len(sub):,}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")
    
    print("\n   By Race/Ethnicity:")
    for re, sub in cohort.groupby('Race_Ethnicity'):
        if len(sub) > 0:
            print(f"     {re}: n={len(sub):,}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")
    
    print("\n   By Gender:")
    for g, sub in cohort.groupby('Gender'):
        if len(sub) > 0:
            print(f"     {g}: n={len(sub):,}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")
    
    print("\n   By BMI Category:")
    for bmi, sub in cohort.groupby('BMI_Cat', observed=False):
        if len(sub) > 0:
            print(f"     {bmi}: n={len(sub):,}, diabetes={sub['known_diabetes'].mean()*100:.1f}%")
    
    # 10. Missing data
    print("\n6. Missing Data (%):")
    key_cols = ['HbA1c', 'Glucose', 'BMI', 'Waist_cm', 'SBP', 'DBP', 'INDFMPIR']
    for c in key_cols:
        if c in cohort.columns:
            miss = cohort[c].isnull().mean() * 100
            print(f"   {c}: {miss:.1f}%")
    
    # 11. Save cohort
    output_path = DATA_DIR.parent / "nhanes_diabetes_cohort.csv"
    cohort.to_csv(output_path, index=False)
    print(f"\n7. Cohort saved to: {output_path}")
    print(f"   Shape: {cohort.shape}")
    print(f"   Features: {len([c for c in cohort.columns if c not in ['SEQN', 'known_diabetes', 'prediabetes', 'normal_glucose', 'outcome_diabetes', 'outcome_prediabetes']])}")
    
    # Save feature/target matrices for modeling
    # Use only numeric features for modeling (avoid get_dummies complexity)
    numeric_features = [
        'RIDAGEYR', 'BMI', 'Waist_cm', 'SBP', 'DBP', 'HbA1c', 'Glucose',
        'Current_Smoker', 'Former_Smoker', 'Current_Drinker', 
        'Vigorous_Activity', 'Moderate_Activity', 'INDFMPIR'
    ]
    # Only use features that exist
    numeric_features = [c for c in numeric_features if c in cohort.columns]
    
    feature_data = cohort[numeric_features].copy()
    target_data = cohort[target_col].copy()
    
    # Impute missing
    for c in feature_data.columns:
        feature_data[c] = pd.to_numeric(feature_data[c], errors='coerce')
        if feature_data[c].isnull().any():
            feature_data[c] = feature_data[c].fillna(feature_data[c].median())
    
    X_path = DATA_DIR.parent / "nhanes_X.csv"
    y_path = DATA_DIR.parent / "nhanes_y.csv"
    feature_data.to_csv(X_path, index=False)
    target_data.to_csv(y_path, index=False)
    print(f"   X shape: {feature_data.shape} -> {X_path}")
    print(f"   y shape: {target_data.shape} -> {y_path}")
    
    return cohort, feature_data, target_data

if __name__ == '__main__':
    cohort, X, y = main()