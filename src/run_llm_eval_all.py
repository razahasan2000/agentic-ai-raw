#!/usr/bin/env python3
"""
Run LLM evaluation on ALL 86 high-risk patients with Qwen-2.5-7B-Instruct
"""

import pickle
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Load artifacts
with open('../results/clean_pipeline_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

evidence_list = artifacts['evidence_list']
ca_coverage = artifacts['ca_coverage']
ca_ece = artifacts['ca_ece']
ca_coverage = artifacts['ca_coverage']

# Filter high-risk patients with counterfactuals
high_risk_evidence = [e for e in evidence_list if e['predicted_risk'] > 0.5 and e.get('counterfactual')]
print(f"High-risk patients with counterfactuals: {len(high_risk_evidence)}")

# Load Qwen-2.5-7B-Instruct
print("Loading Qwen-2.5-7B-Instruct...")
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("LLM loaded successfully")

# Synthesis and Verifier functions
SYNTHESIS_PROMPT_TEMPLATE = """Generate a clinical diabetes risk explanation for a clinician.

Patient: {patient_id}
Subgroup: {subgroup}
Calibrated risk: {calibrated_risk:.1%}
Prediction set: {prediction_set} (90% coverage target)
{cf_text}
{audit_text}

Write a concise narrative. Include: risk percentage, prediction set, 90% coverage, counterfactual details, subgroup. No HbA1c or medications."""

VERIFIER_RULES = [
    ("risk_score", "stated risk matches calibrated_risk within ±1%"),
    ("coverage", "stated prediction set/coverage matches evidence (must mention 90%)"),
    ("counterfactual", "counterfactual claims match deltas and exclude frozen features"),
    ("subgroup_caveat", "subgroup stated"),
    ("no_hallucination", "no extra clinical claims (HbA1c, medications) invented")
]

def synthesize(evidence, repair_hint=None):
    cf = evidence.get('counterfactual')
    if cf:
        cf_text = f"Counterfactual: Tier {cf.get('tier_used', 'N/A')}, Radius {cf.get('radius_mult', 'N/A')}x, Changes: {cf.get('actionable_changes', {})}, Inside ellipsoid: {cf.get('inside_ellipsoid', False)}"
    else:
        cf_text = "No counterfactual needed (risk at or below target)."
    
    audit = evidence.get('audit_status', {})
    audit_text = f"Subgroup coverage: {audit.get('subgroup_coverage', 0):.1%}, ECE: {audit.get('ece', 0):.3f}"
    if audit.get('abstain'):
        audit_text += " ABSTAINED."
    
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        patient_id=evidence.get('patient_id', 0),
        subgroup=evidence.get('subgroup', 'Unknown'),
        calibrated_risk=evidence.get('calibrated_risk', 0),
        prediction_set=evidence.get('prediction_set', []),
        cf_text=cf_text,
        audit_text=audit_text
    )
    
    if repair_hint:
        prompt += f"\n\nFix: {repair_hint}"
    
    messages = [
        {"role": "system", "content": "You are a clinical AI assistant. Output only the narrative."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def verify(narrative, evidence):
    errors = []
    
    # Check 1: Risk score within ±1%
    risk_match = re.search(r'(\d+\.?\d*)\s*%', narrative)
    if risk_match:
        stated_risk = float(risk_match.group(1)) / 100
        true_risk = evidence.get('calibrated_risk', 0)
        if abs(stated_risk - true_risk) > 0.01:
            errors.append(f"Risk score mismatch: stated {stated_risk:.3f}, true {true_risk:.3f}")
    else:
        errors.append("Missing risk score percentage")
    
    # Check 2: Must mention 90% coverage
    if '90%' not in narrative and '90 percent' not in narrative.lower():
        errors.append("Missing 90% coverage statement")
    
    # Check 3: Counterfactual features mentioned
    if 'counterfactual' in evidence and evidence['counterfactual']:
        cf = evidence['counterfactual']
        if cf.get('actionable_changes'):
            for feat, change in cf['actionable_changes'].items():
                feat_name = feat.replace('z_', '').lower()
                if feat_name not in narrative.lower():
                    errors.append(f"Counterfactual feature {feat} not mentioned")
    
    # Check 4: Subgroup caveat
    if 'subgroup' in evidence:
        sg = evidence['subgroup']
        if sg.lower() not in narrative.lower():
            errors.append(f"Missing subgroup caveat for {sg}")
    
    # Check 5: No hallucinated clinical claims
    if 'hba1c' in narrative.lower() or 'a1c' in narrative.lower() or 'medication' in narrative.lower():
        errors.append("Hallucinated HbA1c/medication mention")
    
    return len(errors) == 0, errors

# Run evaluation on all high-risk patients
high_risk_evidence = [e for e in evidence_list if e['predicted_risk'] > 0.5 and e.get('counterfactual')]
print(f"Evaluating {len(high_risk_evidence)} high-risk patients...")

verified_explanations = []
repair_traces = []

for i, ev in enumerate(high_risk_evidence):
    # Prepare evidence schema
    def convert(obj):
        if isinstance(obj, (bool, int, float)):
            return obj
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        return obj
    
    cf = ev.get('counterfactual')
    if cf:
        cf_clean = {
            'tier_used': cf.get('tier_used'),
            'radius_mult': cf.get('radius_mult'),
            'actionable_changes': cf.get('actionable_changes', {}),
            'mahalanobis': cf.get('mahalanobis'),
            'ellipsoid_radius': cf.get('ellipsoid_radius'),
            'inside': cf.get('inside_ellipsoid')
        }
    else:
        cf_clean = None
    
    evidence_schema = {
        "patient_id": ev['subject_id'],
        "subgroup": ev['subgroup'],
        "calibrated_risk": ev['predicted_risk'],
        "prediction_set": ev['prediction_set'],
        "conformal_coverage_target": 0.90,
        "top_significant_features": [
            {"feature": f, "effect": a.get('wald_stat', 0), "significant": a.get('significant', False)}
            for f, a in ev.get('attributions', {}).items()
        ],
        "counterfactual": cf_clean,
        "audit_status": {
            "subgroup_coverage": ca_coverage.get(ev['subgroup'], 0),
            "ece": ca_ece,
            "abstain": ev['abstain']
        }
    }
    
    # Loop 2: Synthesis -> Verifier -> Repair (up to 3 retries)
    max_retries = 3
    trace = []
    for attempt in range(max_retries):
        narrative = synthesize(evidence_schema)
        passed, errors = verify(narrative, evidence_schema)
        
        trace.append({
            'attempt': attempt + 1,
            'narrative': narrative,
            'passed': passed,
            'errors': errors
        })
        
        if passed:
            break
        
        repair_hint = f"Previous attempt failed verification. Errors to fix: {errors}. Regenerate narrative addressing all errors."
        if attempt == max_retries - 1:
            print(f"   Patient {ev['subject_id']}: Verification failed after {max_retries} attempts: {errors}")
    
    verified_explanations.append({
        'patient_id': ev['subject_id'],
        'narrative': trace[-1]['narrative'],
        'verified': trace[-1]['passed'],
        'method': 'llm_loop2',
        'retries': len(trace) - 1,
        'attempt_1_pass': trace[0]['passed'],
        'final_pass': trace[-1]['passed']
    })
    repair_traces.append({
        'patient_id': ev['subject_id'],
        'traces': trace
    })
    
    if (i + 1) % 10 == 0:
        print(f"   Processed {i+1}/{len(high_risk_evidence)} patients...")

# Compute verifier pass rates
attempt_1_passed = sum(1 for e in verified_explanations if e.get('attempt_1_pass', False))
final_passed = sum(1 for e in verified_explanations if e.get('final_pass', False))
total_count = len(verified_explanations)
print(f"\nVerifier pass rate (Attempt 1): {attempt_1_passed}/{total_count} ({attempt_1_passed/total_count*100:.1f}%)")
print(f"Verifier pass rate (Final, M=3): {final_passed}/{total_count} ({final_passed/total_count*100:.1f}%)")

# Save results
with open('../results/llm_repair_traces_all.json', 'w') as f:
    json.dump(repair_traces, f, indent=2)

with open('../results/verified_explanations_all.json', 'w') as f:
    json.dump(verified_explanations, f, indent=2)

print("\nResults saved to results/llm_repair_traces_all.json and results/verified_explanations_all.json")