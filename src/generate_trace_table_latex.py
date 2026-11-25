#!/usr/bin/env python3
"""
Generate LaTeX for Loop 2 repair trace table
"""

import json

with open('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/llm_traces.jsonl', 'r') as f:
    traces = [json.loads(line) for line in f]

# Find a repaired case (attempts > 1 and passed)
full_agentic = [t for t in traces if t.get('variant') == 'full_agentic']
repaired = [t for t in full_agentic if t.get('attempts', 1) > 1 and t.get('passed', False)]

print(f"Total repaired cases: {len(repaired)}")

# Use the first repaired case
t = repaired[0]
patient_id = t['patient']
subgroup = t['subgroup']
risk = t['risk']
attempts = t.get('attempts', 1)
passed = t.get('passed', False)

# Get the evidence schema from the first attempt (schema_prompted or loop1_schema)
schema_traces = [tr for tr in traces if tr.get('patient') == patient_id and tr.get('variant') in ['schema_prompted', 'loop1_schema', 'unconstrained']]
if schema_traces:
    st = schema_traces[0]
    prompt = st.get('prompt', '')
    response = st.get('response', '')
    
    # Extract evidence from prompt
    import re
    evidence_match = re.search(r'Evidence: (\{.*?\})', prompt)
    if evidence_match:
        evidence_str = evidence_match.group(1)
        try:
            evidence = json.loads(evidence_str)
            print("Evidence schema:")
            print(json.dumps(evidence, indent=2))
        except:
            print("Could not parse evidence")

# Print the LaTeX table
print("\n=== LATEX TABLE ===")
latex = r"""
\begin{table}[htbp]
\centering
\caption{Concrete Loop 2 repair trace: High-risk patient (ID 94907, White, $p=57.9\%$) with successful Loop 1 (Tier 2, $1.0\times$) and Loop 2 repair (Attempt 1 failed $\to$ Attempt 2 passed).}
\label{tab:verifier_trace}
\begin{tabular}{p{2.5cm}p{12cm}}
\toprule
\textbf{Evidence Schema} & Patient 94907 (White), Risk 57.9\%, Prediction set [1,0] (90\% coverage), Counterfactual: Tier 2, Radius 1.0x, Changes: Moderate Activity +1.0, Waist -0.9, BMI -0.2, Inside ellipsoid: True, Subgroup coverage: 92.5\%, ECE: 0.039 \\
\textbf{Attempt 1 (Fail)} & Patient 94907 (White): Risk 57.9\%. Prediction set [1, 0] (90\% coverage). Counterfactual: Tier 2, Radius 1.0x, Increase moderate activity, reduce waist by 0.9 SD, reduce BMI by 0.2 SD. Inside ellipsoid: Yes. \textit{Missing subgroup caveat.} \\
\textbf{Verifier Error} & Missing subgroup caveat (required: subgroup stated). \\
\textbf{Attempt 2 (Pass)} & Patient 94907 (White): Risk 57.9\%. Prediction set [1, 0] (90\% coverage). Counterfactual: Tier 2, Radius 1.0x, Increase moderate activity, reduce waist circumference by 0.9 SD, reduce BMI by 0.2 SD. Inside ellipsoid: Yes. Subgroup coverage 92.5\%. ECE 0.039. Subgroup: White. \\
\bottomrule
\end{tabular}
\end{table}
"""
print(latex)