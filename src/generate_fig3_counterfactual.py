#!/usr/bin/env python3
"""
Generate Figure 3: Counterfactual Pareto - Reach Rate vs Mean ||δ||_2
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/metrics_verified.json', 'r') as f:
    m = json.load(f)

# Counterfactual baseline data from Table 5 (now Table 6 in paper)
# Loop 1 (LR): Reach=100%, Containment=100%, Mean ||δ||=1.38
# Loop 1 (XGBoost): Reach=100%, Containment=100%, Mean ||δ||=1.42
# Loop 1 (LightGBM): Reach=100%, Containment=100%, Mean ||δ||=1.40
# Wachter Unconstrained: Reach=96.8%, Containment=41.2%, Mean ||δ||=0.87
# Static Tier 1 (1.0x): Reach=35.2%, Containment=35.2%, Mean ||δ||=0.52
# Static Tier 2 (1.0x): Reach=58.7%, Containment=58.7%, Mean ||δ||=1.12
# Static Tier 3 (2.0x): Reach=82.1%, Containment=74.3%, Mean ||δ||=2.15

methods = ['Loop 1\n(LR)', 'Loop 1\n(XGBoost)', 'Loop 1\n(LightGBM)', 'Wachter\nUnconstrained', 'Static\nTier 1 (1.0x)', 'Static\nTier 2 (1.0x)', 'Static\nTier 3 (2.0x)']
reach = [100.0, 100.0, 100.0, 96.8, 35.2, 58.7, 82.1]
containment = [100.0, 100.0, 100.0, 41.2, 35.2, 58.7, 74.3]
delta_norm = [1.38, 1.42, 1.40, 0.87, 0.52, 1.12, 2.15]

# Colors and markers for each method
colors = ['#2ecc71', '#27ae60', '#1e8c49', '#e74c3c', '#3498db', '#2980b9', '#8e44ad']
markers = ['o', '^', 's', 'D', 'o', '^', 's']
sizes = [120, 120, 120, 120, 100, 100, 100]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (method, r, c, d) in enumerate(zip(methods, reach, containment, delta_norm)):
    # X-axis: Mean Mahalanobis distance (approximated by delta_norm)
    # Y-axis: Counterfactual Reach Rate
    # Marker size proportional to containment
    size = 50 + c * 1.5  # Scale containment to marker size
    ax.scatter(d, r, s=size, c=colors[i], marker=markers[i], label=method, 
               edgecolors='black', linewidth=1.5, alpha=0.8, zorder=3)
    
    # Add method label with staggered offsets for Loop 1 labels
    if i < 3:
        offsets = [(5, 8), (5, -12), (5, 20)]
        ax.annotate(method, (d, r), xytext=offsets[i], textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors[i])
    else:
        ax.annotate(method, (d, r), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors[i])

# Add diagonal reference line for perfect containment
ax.plot([0, 2.5], [0, 100], 'k--', alpha=0.3, label='Perfect containment (y=x)')

ax.set_xlabel('Mean ||δ||₂ (SD units) / Mean Mahalanobis Distance', fontsize=13)
ax.set_ylabel('Counterfactual Reach Rate (%)', fontsize=13)
ax.set_title('Counterfactual Reach vs. Plausibility Across Methods (20 Seeds)', fontsize=14, fontweight='bold')
ax.set_xlim(-0.1, 2.5)
ax.set_ylim(-5, 115)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/figures/fig3_counterfactual_pareto.png', dpi=300, bbox_inches='tight')
plt.savefig('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/figures/fig3_counterfactual_pareto.pdf', bbox_inches='tight')
print("Figure 3 saved!")