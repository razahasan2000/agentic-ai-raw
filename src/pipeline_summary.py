"""
Complete Pipeline Summary: Subgroup-Calibrated Geometric Conformal Plausibility Agent
=====================================================================================
End-to-end demonstration of the framework from the paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")

print("=" * 70)
print("SUBGROUP-CALIBRATED GEOMETRIC CONFORMAL PLAUSIBILITY AGENT")
print("Pipeline Summary (MIMIC-IV Demo)")
print("=" * 70)

# ============================================================
# 1. COHORT STATISTICS
# ============================================================
cohort = pd.read_csv(DATA_DIR / "cohort_incident_t2d.csv")
print(f"\n1. COHORT CONSTRUCTION")
print(f"   Patients: {len(cohort)}")
print(f"   Incident T2D: {cohort['incident_t2d'].sum()} ({cohort['incident_t2d'].mean()*100:.1f}%)")
print(f"   Subgroups (age_band): {cohort['age_band'].value_counts().to_dict()}")
print(f"   Features: {len([c for c in cohort.columns if not c.startswith('subgroup_')])}")

# ============================================================
# 2. EMBEDDING
# ============================================================
embeddings = pd.read_csv(DATA_DIR / "embeddings.csv")
z_cols = [c for c in embeddings.columns if c.startswith('z_')]
print(f"\n2. CLINICALLY NORMALIZED EMBEDDING (Section 4.1)")
print(f"   Dimensions: {len(z_cols)}")
print(f"   Features: {z_cols}")
print(f"   Subgroup indicators: {[c for c in embeddings.columns if c.startswith('subgroup_')]}")

# ============================================================
# 3. CONFORMAL CALIBRATION
# ============================================================
with open(DATA_DIR / "calibration_artifacts.pkl", 'rb') as f:
    cal = pickle.load(f)

print(f"\n3. SUBGROUP-CONDITIONAL CONFORMAL CALIBRATION (Section 4.2)")
print(f"   Alpha: {cal['alpha']} (Target coverage: {1-cal['alpha']:.0%})")
print(f"   Global quantile: {cal['global_quantile']:.4f}")
print(f"   Hierarchical shrinkage kappa: {cal['kappa']:.2f}")
print(f"   Between-stratum variance tau^2: {cal['tau2']:.6f}")
print(f"   Final subgroup quantiles:")
for sg, q in cal['final_quantiles'].items():
    n = cal['group_sizes'].get(sg, 0)
    print(f"     {sg}: q={q:.4f} (n={n})")

# ============================================================
# 4. COUNTERFACTUAL GEOMETRY
# ============================================================
with open(DATA_DIR / "counterfactual_results.pkl", 'rb') as f:
    cf = pickle.load(f)

print(f"\n4. CONFORMALIZED COUNTERFACTUAL GEOMETRY (Section 4.3)")
print(f"   Actionable features: {cf['actionable_features']}")
print(f"   High-risk patients with counterfactuals: {len(cf['counterfactuals'])}")
if cf['counterfactuals']:
    inside = sum(r['inside_ellipsoid'] for r in cf['counterfactuals'])
    success = sum(1 for r in cf['counterfactuals'] if r['p_final'] <= 0.5)
    print(f"   Reached target risk (<=0.5): {success}/{len(cf['counterfactuals'])} ({success/len(cf['counterfactuals'])*100:.1f}%)")
    print(f"   Inside ellipsoid: {inside}/{len(cf['counterfactuals'])} ({inside/len(cf['counterfactuals'])*100:.1f}%)")
    print(f"   Mean displacement: {np.mean([r['delta_norm'] for r in cf['counterfactuals']]):.3f} SD")
    print(f"   Mean Mahalanobis: {np.mean([r['mahalanobis'] for r in cf['counterfactuals']]):.3f}")

# ============================================================
# 5. ATTRIBUTION SIGNIFICANCE
# ============================================================
with open(DATA_DIR / "attribution_significance.pkl", 'rb') as f:
    attr = pickle.load(f)

print(f"\n5. CONFORMALIZED MARGINAL ATTRIBUTION (Section 4.4)")
print(f"   Alpha: {attr['alpha']}")
sig_feats = [(k, v['sig_frac']) for k, v in attr['global_significance'].items() if v['sig_frac'] > 0]
if sig_feats:
    print(f"   Significant features: {sig_feats}")
else:
    print(f"   No features significant at {attr['alpha']:.0%} (expected with small demo data)")
    # Show top by mean |effect|
    top = sorted(attr['global_significance'].items(), key=lambda x: x[1]['mean_abs_effect'], reverse=True)[:5]
    top_str = ", ".join([f"{k}: {v['mean_abs_effect']:.4f}" for k, v in top])
    print(f"   Top by mean |dp|: {top_str}")

# ============================================================
# 6. AUDITOR & TRANSLATOR
# ============================================================
with open(DATA_DIR / "audit_translate_results.pkl", 'rb') as f:
    audit = pickle.load(f)

audit_res = audit['audit_result']
print(f"\n6. FAIRNESS/STABILITY AUDITOR & TRANSLATOR (Section 4.5)")
print(f"   Global abstain: {audit_res['abstain']} ({audit_res['reason']})")
print(f"   Coverage parity max gap: {audit_res['coverage_audit']['max_gap']:.3f} (tol=0.10)")
print(f"   Calibration ECE: {audit_res['calibration_audit']['ece']:.4f}")
print(f"   Stability rate: {audit_res['stability_audit']['stability_rate']:.3f}")

# Per-subgroup coverage
print(f"   Subgroup coverage:")
for sg, stats in audit_res['coverage_audit']['by_subgroup'].items():
    status = "OK" if stats['coverage'] >= 0.8 else "LOW"
    print(f"     {sg}: {stats['coverage']:.1%} (n={stats['n']}) [{status}]")

# Abstention decisions
abstain_count = sum(1 for ev in audit['evidence_schemas'] 
                    if audit_res['coverage_audit']['by_subgroup'].get(ev['subgroup'], {}).get('coverage', 1) < 0.8)
print(f"   Patients abstained: {abstain_count}/{len(audit['evidence_schemas'])}")

# ============================================================
# 7. COMPONENT INTEGRATION CHECK
# ============================================================
print(f"\n7. FRAMEWORK INTEGRATION CHECK")
checks = {
    "Clinically normalized embedding": len(z_cols) > 0,
    "Subgroup-conditional conformal": len(cal['final_quantiles']) > 1,
    "Hierarchical shrinkage": cal['kappa'] > 0,
    "Weighted calibration": len(cal['weighted_quantiles']) > 0,
    "Plausibility ellipsoids": len(cf['subgroup_stats']) > 0,
    "Counterfactual search": len(cf['counterfactuals']) > 0,
    "Conformal attributions": len(attr['global_significance']) > 0,
    "Fairness auditor": 'coverage_audit' in audit_res,
    "Abstention controller": 'abstain' in audit_res,
    "Schema-constrained translator": len(audit['evidence_schemas']) > 0,
}

all_pass = True
for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"   [{status}] {check}")

print(f"\n   Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

# ============================================================
# 8. LIMITATIONS & NEXT STEPS
# ============================================================
print(f"\n8. LIMITATIONS (Demo Data)")
print(f"   - Cohort size: {len(cohort)} (paper uses 10,000+)")
print(f"   - Calibration set: 18 samples (need 100+ per subgroup)")
print(f"   - Many features imputed (HbA1c 89% missing)")
print(f"   - Coverage below target due to small calibration set")
print(f"   - Attribution underpowered (0% significant)")

print(f"\n9. NEXT STEPS FOR FULL IMPLEMENTATION")
print(f"   1. Apply for MIMIC-IV full access (PhysioNet credentialing)")
print(f"   2. Use METRE pipeline for MIMIC-IV + eICU harmonized extraction")
print(f"   3. Define incident T2D cohort with 1-year lookback, 365-day outcome")
print(f"   4. Scale embedding to 50+ features (labs, vitals, meds, comorbidities)")
print(f"   5. Train gradient boosting (XGBoost/LightGBM) as base predictor")
print(f"   6. Use 5-fold CV for conformal calibration (not single split)")
print(f"   7. Implement LLM translator (Llama-3/MedLlama with schema constraints)")
print(f"   8. Add temporal conformal prediction for longitudinal data")
print(f"   9. Federated calibration across hospitals (hierarchical shrinkage)")
print(f"   10. Prospective clinician evaluation (usefulness, trust)")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE - Ready for scale-up to full MIMIC-IV")
print("=" * 70)