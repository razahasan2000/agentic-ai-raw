#!/usr/bin/env python3
"""
Generate Figure 4: LLM Loop 2 Repair Progression
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load LLM traces
with open('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/llm_traces.jsonl', 'r') as f:
    traces = [json.loads(line) for line in f]

# Filter for full_agentic variant
full_agentic = [t for t in traces if t.get('variant') == 'full_agentic']
schema_prompted = [t for t in traces if t.get('variant') == 'schema_prompted']

# Compute pass rates by attempt
attempts = [1, 2, 3]
pass_rates = []
counts = []

for a in attempts:
    # For full_agentic, count how many passed by attempt a
    passed_at_a = sum(1 for t in full_agentic if t.get('attempts', 1) <= a and t.get('passed', False))
    total = len(full_agentic)
    pass_rates.append(passed_at_a / total * 100)
    counts.append(f"{passed_at_a}/{total}")

# Schema prompted (single attempt)
schema_pass = sum(1 for t in schema_prompted if t.get('passed', False))
schema_total = len(schema_prompted)
schema_rate = schema_pass / schema_total * 100

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(attempts))
bars = ax.bar(x_pos, pass_rates, color=['#2ecc71', '#3498db', '#9b59b6'], 
              edgecolor='black', linewidth=1.2, width=0.6)

# Add value labels on bars
for i, (bar, rate, count) in enumerate(zip(bars, pass_rates, counts)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{rate:.1f}%\n({count})', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add schema prompted reference line
ax.axhline(y=schema_rate, color='#e74c3c', linestyle='--', linewidth=2, 
           label=f'Schema Prompted (1-shot): {schema_rate:.1f}% ({schema_pass}/{schema_total})')

ax.set_ylabel('Verifier Pass Rate (%)', fontsize=13)
ax.set_xlabel('Loop 2 Attempt', fontsize=13)
ax.set_title('Loop 2 Repair Progression: Qwen2.5-1.5B-Instruct on 86 High-Risk Patients (Seed 3)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Attempt 1', 'Attempt 2', 'Attempt 3'], fontsize=12)
ax.set_ylim(0, 110)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add improvement annotation
improvement = pass_rates[-1] - pass_rates[0]
ax.annotate(f'Repair Gain: +{improvement:.1f} pp', 
            xy=(2, pass_rates[2]), xytext=(1.5, 85),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=11, fontweight='bold', color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c'))

plt.tight_layout()
plt.savefig('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/figures/fig4_llm_repair.png', dpi=300, bbox_inches='tight')
plt.savefig('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/figures/fig4_llm_repair.pdf', bbox_inches='tight')
print("Figure 4 saved!")
print(f"Pass rates: {pass_rates}")
print(f"Schema prompted: {schema_rate:.1f}%")
print(f"Repair gain: {improvement:.1f} pp")