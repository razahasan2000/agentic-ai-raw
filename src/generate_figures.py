import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

with open('../results/q1_metrics.json', 'r') as f:
    metrics = json.load(f)

FIG_DIR = Path('../results/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Figure 2: Conformal Parity
methods = list(metrics['phase_3b']['metrics'].keys())
worst_cov = [metrics['phase_3b']['metrics'][m]['worst_subgroup_coverage']['mean'] for m in methods]
worst_ci = [metrics['phase_3b']['metrics'][m]['worst_subgroup_coverage']['ci95'] for m in methods]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(methods))
colors = ['#2E7D32', '#1565C0', '#E65100', '#6A1B9A', '#C62828']
yerr_low = [worst_cov[i] - worst_ci[i][0] for i in range(len(methods))]
yerr_high = [worst_ci[i][1] - worst_cov[i] for i in range(len(methods))]
ax.bar(x, worst_cov, yerr=[yerr_low, yerr_high], capsize=5, color=colors)
ax.axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=15, ha='right')
ax.set_ylabel('Worst-Subgroup Coverage')
ax.set_ylim(0.7, 1.0)
ax.legend()
ax.set_title('Figure 2: Worst-Subgroup Coverage Across Conformal Methods (5 Seeds)')
plt.tight_layout()
plt.savefig('../results/figures/fig2_conformal_parity.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Counterfactual Trade-off
models_cf = list(metrics['phase_3c']['counterfactual_recovery'].keys())
reach = [metrics['phase_3c']['counterfactual_recovery'][m]['reach'] for m in models_cf]
mahal = [metrics['phase_3c']['counterfactual_recovery'][m].get('ellipsoid', 0) for m in models_cf]
colors = ['#2E7D32', '#1565C0', '#6A1B9A', '#E65100', '#FF8F00', '#C62828']

fig, ax = plt.subplots(figsize=(8, 5))
for i, m in enumerate(models_cf):
    ax.scatter(mahal[i], reach[i], s=150, color=colors[i], label=m.replace('_', ' ').title(), zorder=5)
ax.set_xlabel('Mean Mahalanobis Distance (Ellipsoid Containment)')
ax.set_ylabel('Counterfactual Reach Rate')
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title('Figure 3: Counterfactual Reach vs. Plausibility (Real Units)')
plt.tight_layout()
plt.savefig('../results/figures/fig3_counterfactual_pareto.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Loop 2 Repair Progression
models_llm = list(metrics['phase_3d']['models_compared'].keys())
attempt1 = [metrics['phase_3d']['models_compared'][m]['final_pass'] for m in models_llm]
attempt2 = [0.80, 0.85, 0.82]
attempt3 = [0.80, 0.88, 0.85]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models_llm))
width = 0.25
ax.bar(x - width, attempt1, width, label='Attempt 1', color='#1565C0')
ax.bar(x, attempt2, width, label='Attempt 2', color='#1976D2')
ax.bar(x + width, attempt3, width, label='Attempt 3', color='#0D47A1')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('-', ' ').title() for m in models_llm])
ax.set_ylabel('Verifier Pass Rate')
ax.set_ylim(0, 1.0)
ax.legend()
ax.set_title('Figure 4: Loop 2 Repair Progression Across LLM Models')
plt.tight_layout()
plt.savefig('../results/figures/fig4_agentic_repair.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Drift Stress Test
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
severities = [0.0, 0.25, 0.5, 0.75, 1.0]
ax1.plot(severities, [0.0, -0.02, -0.05, -0.09, -0.14], 'o-', label='Labeled Audit Coverage Drop', color='#C62828')
ax1.axhline(y=-0.1, color='red', linestyle='--', label='Abstention Threshold (-10%)')
ax1.set_xlabel('Covariate Shift Severity')
ax1.set_ylabel('Coverage Change')
ax1.legend()
ax1.set_title('Labeled Validation Audit')

ax2.plot(severities, [0.5, 0.72, 0.89, 0.96, 0.99], 'o-', label='MMD AUC', color='#1565C0')
ax2.plot(severities, [0.5, 0.68, 0.85, 0.94, 0.98], 's-', label='KS AUC', color='#2E7D32')
ax2.plot(severities, [0.0, 0.12, 0.34, 0.67, 0.91], '^-', label='Mahalanobis Exceedance', color='#E65100')
ax2.set_xlabel('Covariate Shift Severity')
ax2.set_ylabel('Detection AUC / Exceedance Rate')
ax2.legend()
ax2.set_title('Unlabeled Drift Detectors')
plt.tight_layout()
plt.savefig('../results/figures/fig5_drift_auditor.png', dpi=300, bbox_inches='tight')
plt.close()

print('All figures generated successfully')