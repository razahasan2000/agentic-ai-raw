#!/usr/bin/env python3
"""
Generate Loop 2 repair trace table from llm_traces.jsonl
"""

import json

with open('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/llm_traces.jsonl', 'r') as f:
    traces = [json.loads(line) for line in f]

# Find a case where full_agentic: Attempt 1 failed, Attempt 2 or 3 passed
full_agentic = [t for t in traces if t.get('variant') == 'full_agentic']
failed_then_passed = []
for t in full_agentic:
    attempts = t.get('attempts', 1)
    passed = t.get('passed', False)
    if attempts > 1 and passed:
        failed_then_passed.append(t)

print(f"Total full_agentic: {len(full_agentic)}")
print(f"Failed then passed: {len(failed_then_passed)}")

# Pick the first one
if failed_then_passed:
    trace = failed_then_passed[0]
    print(f"\nSelected trace:")
    print(f"  Patient: {trace['patient']}")
    print(f"  Subgroup: {trace['subgroup']}")
    print(f"  Risk: {trace['risk']}%")
    print(f"  Attempts: {trace['attempts']}")
    print(f"  Passed: {trace['passed']}")
    print(f"  Final response: {trace['final_response'][:200]}...")
    
    # We need the attempt 1 response and verifier error
    # The trace only has final_response. We need to check if there are intermediate traces.
    # The llm_traces.jsonl should have separate entries for each attempt.
    # Let me check if there are separate entries for each attempt.
    
    # Look for all traces for this patient
    patient_traces = [t for t in traces if t.get('patient') == trace['patient'] and t.get('variant') == 'full_agentic']
    print(f"\nAll traces for patient {trace['patient']}:")
    for t in patient_traces:
        print(f"  Attempt: {t.get('attempts', 1)}, Passed: {t.get('passed')}, Response: {t.get('final_response', t.get('response', ''))[:100]}...")

# Also find a case for the table that shows the evidence schema
print("\n\n=== EVIDENCE SCHEMA EXAMPLE ===")
# Get the first full_agentic trace with full details
for t in full_agentic:
    if 'final_response' in t:
        print(f"Patient: {t['patient']}")
        print(f"Risk: {t['risk']}%")
        print(f"Subgroup: {t['subgroup']}")
        print(f"Attempts: {t.get('attempts', 1)}")
        print(f"Passed: {t['passed']}")
        print(f"Response: {t['final_response'][:300]}")
        break