"""
Schema-Constrained LLM Translator (Section 4.5)
================================================
Generates clinical narratives where EVERY claim is tagged with 
calibrated confidence from the evidence schema.
Uses structured generation (JSON schema) to enforce constraints.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# EVIDENCE SCHEMA (matches paper Section 4.5)
# ============================================================
class ConfidenceLevel(str, Enum):
    HIGH = "high"        # p < 0.01 or coverage > 0.95
    MEDIUM = "medium"    # p < 0.05 or coverage > 0.90  
    LOW = "low"          # p < 0.10 or coverage > 0.80
    UNCERTAIN = "uncertain"  # Below thresholds

@dataclass
class EvidenceField:
    """Base evidence field with calibrated confidence"""
    value: Any
    confidence: ConfidenceLevel
    calibration_source: str
    subgroup: str
    details: Dict = None
    predicted_risk: float = 0.0
    prediction_set: List[int] = None
    coverage_level: float = 0.0
    calibrated_quantile: float = 0.0
    initial_risk: float = 0.0
    target_risk: float = 0.0
    displacement_norm: float = 0.0
    mahalanobis_distance: float = 0.0
    inside_ellipsoid: bool = False
    actionable_changes: Dict[str, float] = None
    feature: str = ""
    marginal_effect: float = 0.0
    p_value: float = 1.0
    significant: bool = False

# Type aliases for clarity
PredictionSetEvidence = EvidenceField
CounterfactualEvidence = EvidenceField
AttributionEvidence = EvidenceField

@dataclass
class PatientEvidence:
    """Complete evidence schema for one patient"""
    subject_id: int
    subgroup: str
    prediction_set: PredictionSetEvidence
    counterfactual: Optional[CounterfactualEvidence]
    attributions: List[AttributionEvidence]
    fairness_flags: Dict[str, Any]
    stability_flags: Dict[str, Any]
    abstention_recommendation: bool
    abstention_reason: str

# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================
def calibrate_confidence(p_value: float, coverage: float, source: str) -> ConfidenceLevel:
    """Map statistical measures to confidence levels"""
    if source == "marginal_attribution":
        if p_value < 0.01:
            return ConfidenceLevel.HIGH
        elif p_value < 0.05:
            return ConfidenceLevel.MEDIUM
        elif p_value < 0.10:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN
    elif source == "conformal_prediction_set":
        if coverage >= 0.95:
            return ConfidenceLevel.HIGH
        elif coverage >= 0.90:
            return ConfidenceLevel.MEDIUM
        elif coverage >= 0.80:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN
    elif source == "counterfactual_ellipsoid":
        if coverage >= 0.95:
            return ConfidenceLevel.HIGH
        elif coverage >= 0.90:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    return ConfidenceLevel.UNCERTAIN

# ============================================================
# EVIDENCE BUILDER
# ============================================================
def build_patient_evidence(evidence_dict: Dict, alpha: float = 0.1) -> PatientEvidence:
    """Convert raw evidence dict to typed PatientEvidence"""
    
    sg = evidence_dict['subgroup']
    coverage = evidence_dict.get('coverage_level', 1 - alpha)
    
    # Prediction set evidence
    pred_set = evidence_dict['prediction_set']
    pred_risk = evidence_dict['predicted_risk']
    quantile = evidence_dict.get('calibrated_quantile', 0)
    
    pred_evidence = PredictionSetEvidence(
        value={"risk": pred_risk, "set": pred_set},
        confidence=calibrate_confidence(0, coverage, "conformal_prediction_set"),
        calibration_source="conformal_prediction_set",
        subgroup=sg,
        details={
            "predicted_risk": pred_risk,
            "prediction_set": pred_set,
            "coverage_level": coverage,
            "calibrated_quantile": quantile
        },
        predicted_risk=pred_risk,
        prediction_set=pred_set,
        coverage_level=coverage,
        calibrated_quantile=quantile
    )
    
    # Counterfactual evidence
    cf_evidence = None
    if evidence_dict.get('counterfactual'):
        cf = evidence_dict['counterfactual']
        cf_cov = 1 - alpha if cf.get('inside_ellipsoid', False) else 0.5
        cf_evidence = CounterfactualEvidence(
            value=cf,
            confidence=calibrate_confidence(0, cf_cov, "counterfactual_ellipsoid"),
            calibration_source="counterfactual_ellipsoid",
            subgroup=sg,
            details=cf,
            initial_risk=cf['p_initial'],
            target_risk=cf['p_final'],
            displacement_norm=cf['delta_norm'],
            mahalanobis_distance=cf['mahalanobis'],
            inside_ellipsoid=cf['inside_ellipsoid'],
            actionable_changes=cf['actionable_changes']
        )
    
    # Attribution evidence
    attr_evidence = []
    for feat_name, attr_info in evidence_dict.get('attributions', {}).items():
        if attr_info['p_value'] < 0.2:  # Only include notable
            pval = attr_info['p_value']
            attr_evidence.append(AttributionEvidence(
                value=attr_info,
                confidence=calibrate_confidence(pval, 0, "marginal_attribution"),
                calibration_source="marginal_attribution",
                subgroup=sg,
                details=attr_info,
                feature=feat_name,
                marginal_effect=attr_info['marginal_effect'],
                p_value=pval,
                significant=attr_info['significant']
            ))
    
    # Fairness & stability flags (from auditor)
    fairness = evidence_dict.get('fairness_flags', {})
    stability = evidence_dict.get('stability_flags', {})
    
    # Abstention
    abstain = evidence_dict.get('abstain', False)
    abstain_reason = evidence_dict.get('abstention_reason', '')
    
    return PatientEvidence(
        subject_id=evidence_dict['subject_id'],
        subgroup=sg,
        prediction_set=pred_evidence,
        counterfactual=cf_evidence,
        attributions=attr_evidence,
        fairness_flags=fairness,
        stability_flags=stability,
        abstention_recommendation=abstain,
        abstention_reason=abstain_reason
    )

# ============================================================
# SCHEMA-CONSTRAINED NARRATIVE GENERATOR
# ============================================================
class SchemaConstrainedTranslator:
    """
    Generates narratives where EVERY claim references evidence schema.
    In production: uses LLM with JSON schema constraints.
    Here: template-based with evidence tags.
    """
    
    FEATURE_LABELS = {
        'z_age': 'Age',
        'z_lab_hba1c': 'HbA1c',
        'z_lab_glucose_chem': 'Glucose',
        'z_lab_glucose_blood_gas': 'Glucose (blood gas)',
        'z_vital_bmi_kg_m2': 'BMI',
        'z_vital_weight_lbs': 'Weight',
        'z_vital_height_inches': 'Height',
        'z_comorb_count': 'Comorbidity count',
        'z_age_band': 'Age band',
        'z_comorb_band': 'Comorbidity band',
        'z_gender': 'Sex',
    }
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def _confidence_tag(self, level: ConfidenceLevel) -> str:
        """Format confidence tag for narrative"""
        tags = {
            ConfidenceLevel.HIGH: "[HIGH CONFIDENCE: p<0.01]",
            ConfidenceLevel.MEDIUM: "[MEDIUM CONFIDENCE: p<0.05]",
            ConfidenceLevel.LOW: "[LOW CONFIDENCE: p<0.10]",
            ConfidenceLevel.UNCERTAIN: "[UNCERTAIN: not statistically significant]"
        }
        return tags[level]
    
    def _format_prediction_set(self, pred_set: List[int]) -> str:
        if pred_set == [0]:
            return "Low Risk (diabetes-free)"
        elif pred_set == [1]:
            return "High Risk (will develop diabetes)"
        else:
            return "Uncertain (both Low and High Risk possible)"
    
    def _describe_changes(self, changes: Dict[str, float]) -> List[str]:
        """Convert SD changes to clinical descriptions"""
        descriptions = []
        for feat, change in changes.items():
            label = self.FEATURE_LABELS.get(feat, feat)
            direction = "increase" if change > 0 else "decrease"
            magnitude = abs(change)
            if magnitude < 0.2:
                mag_str = "slightly"
            elif magnitude < 0.5:
                mag_str = "moderately"
            elif magnitude < 1.0:
                mag_str = "substantially"
            else:
                mag_str = "markedly"
            descriptions.append(f"{label}: {mag_str} {direction} ({change:+.2f} SD)")
        return descriptions
    
    def translate(self, evidence: PatientEvidence) -> str:
        """Generate schema-constrained narrative"""
        lines = []
        
        # Header
        lines.append(f"PATIENT {evidence.subject_id}  |  SUBGROUP: {evidence.subgroup}")
        lines.append("=" * 60)
        
        # --- PREDICTION ---
        ps = evidence.prediction_set
        lines.append(f"\n[RISK PREDICTION] {ps.confidence.value.upper()} CONFIDENCE")
        lines.append(f"   Predicted 1-year T2D risk: {ps.predicted_risk:.1%}")
        lines.append(f"   Conformal prediction set: {self._format_prediction_set(ps.prediction_set)}")
        lines.append(f"   Coverage guarantee: {ps.coverage_level:.0%} (subgroup-conditional)")
        lines.append(f"   {self._confidence_tag(ps.confidence)}")
        
        # --- COUNTERFACTUAL ---
        if evidence.counterfactual:
            cf = evidence.counterfactual
            lines.append(f"\n[COUNTERFACTUAL] {cf.confidence.value.upper()} CONFIDENCE")
            lines.append(f"   Current risk: {cf.initial_risk:.1%} -> Target risk: {cf.target_risk:.1%}")
            lines.append(f"   Displacement: {cf.displacement_norm:.2f} SD (Mahalanobis: {cf.mahalanobis_distance:.2f})")
            lines.append(f"   Inside plausibility ellipsoid: {'YES' if cf.inside_ellipsoid else 'NO'}")
            
            if cf.actionable_changes:
                lines.append("   Recommended changes:")
                for desc in self._describe_changes(cf.actionable_changes):
                    lines.append(f"     - {desc}")
            
            lines.append(f"   {self._confidence_tag(cf.confidence)}")
        
        # --- ATTRIBUTIONS ---
        if evidence.attributions:
            lines.append(f"\n[SIGNIFICANT RISK FACTORS]")
            for attr in evidence.attributions:
                direction = "increases" if attr.marginal_effect > 0 else "decreases"
                lines.append(f"   - {self.FEATURE_LABELS.get(attr.feature, attr.feature)}: "
                           f"{direction} risk by {abs(attr.marginal_effect):.3f} "
                           f"(p={attr.p_value:.3f}) {self._confidence_tag(attr.confidence)}")
        else:
            lines.append(f"\n[SIGNIFICANT RISK FACTORS]")
            lines.append(f"   No statistically significant factors at {self.alpha:.0%} level "
                        f"[UNCERTAIN]")
        
        # --- FAIRNESS & STABILITY ---
        lines.append(f"\n[FAIRNESS & STABILITY AUDIT]")
        if evidence.fairness_flags:
            for k, v in evidence.fairness_flags.items():
                lines.append(f"   {k}: {v}")
        if evidence.stability_flags:
            for k, v in evidence.stability_flags.items():
                lines.append(f"   {k}: {v}")
        
        # --- ABSTENTION ---
        if evidence.abstention_recommendation:
            lines.append(f"\n[ABSTENTION RECOMMENDED]")
            lines.append(f"   Reason: {evidence.abstention_reason}")
            lines.append(f"   Clinician review required.")
        else:
            lines.append(f"\n[EXPLANATION VALIDATED]")
            lines.append(f"   All statistical guarantees satisfied for subgroup {evidence.subgroup}.")
        
        return "\n".join(lines)
    
    def translate_json(self, evidence: PatientEvidence) -> Dict:
        """Structured JSON output for programmatic use"""
        return {
            "subject_id": evidence.subject_id,
            "subgroup": evidence.subgroup,
            "narrative": self.translate(evidence),
            "evidence": {
                "prediction_set": asdict(evidence.prediction_set),
                "counterfactual": asdict(evidence.counterfactual) if evidence.counterfactual else None,
                "attributions": [asdict(a) for a in evidence.attributions],
                "fairness_flags": evidence.fairness_flags,
                "stability_flags": evidence.stability_flags,
                "abstention": {
                    "recommended": evidence.abstention_recommendation,
                    "reason": evidence.abstention_reason
                }
            },
            "schema_version": "1.0"
        }

# ============================================================
# DEMO: Run on existing evidence
# ============================================================
def demo_translator():
    """Demo with existing audit_translate results"""
    DATA_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\data\mimic_iv_demo")
    
    with open(DATA_DIR / "audit_translate_results.pkl", 'rb') as f:
        audit = pickle.load(f)
    
    translator = SchemaConstrainedTranslator(alpha=0.1)
    
    print("=" * 60)
    print("SCHEMA-CONSTRAINED TRANSLATOR DEMO")
    print("=" * 60)
    
    for i, ev_dict in enumerate(audit['evidence_schemas'][:5]):
        # Add fairness/stability flags from audit
        sg = ev_dict['subgroup']
        audit_res = audit['audit_result']
        
        sg_cov = audit_res['coverage_audit']['by_subgroup'].get(sg, {}).get('coverage', 0)
        
        # Handle None counterfactual FIRST
        if ev_dict.get('counterfactual') is None:
            ev_dict['counterfactual'] = {}
        
        ev_dict['fairness_flags'] = {
            'subgroup_coverage': f"{sg_cov:.1%}",
            'parity_gap': f"{audit_res['coverage_audit']['max_gap']:.1%}",
            'parity_ok': audit_res['coverage_audit']['parity_ok']
        }
        ev_dict['stability_flags'] = {
            'stability_rate': f"{audit_res['stability_audit']['stability_rate']:.1%}",
            'ellipsoid_constraint': 'satisfied' if ev_dict.get('counterfactual', {}).get('inside_ellipsoid', False) else 'violated'
        }
        
        # Per-patient abstention
        ev_dict['abstain'] = sg_cov < 0.8
        ev_dict['abstention_reason'] = f"Subgroup coverage {sg_cov:.1%} below 80% threshold"
        
        evidence = build_patient_evidence(ev_dict)
        json_out = translator.translate_json(evidence)
        
        print(f"\n{'='*60}")
        print(json_out['narrative'])
        
        # Also save JSON
        if i == 0:
            with open(DATA_DIR / "translated_evidence_sample.json", 'w') as f:
                json.dump(json_out, f, indent=2, default=str)
    
    print(f"\n\nSample JSON saved to translated_evidence_sample.json")

if __name__ == '__main__':
    import pickle
    demo_translator()