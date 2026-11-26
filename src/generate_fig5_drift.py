#!/usr/bin/env python3
"""
Generate Figure 5: Drift Auditor - Unlabeled drift detection vs labeled audit
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Updated data from the paper (Table 9)
severities = [0, 0.25, 0.5, 0.75, 1.0]
sev_labels = ['0%', '25%', '50%', '75%', '100%']

labeled_drop = [0.0, 2.0, 3.9, 5.8, 7.6]
labeled_abstain = [0, 0, 0, 0, 0]  # No abstention triggered (all < 10%)
mmd_auc = [0.50, 0.74, 0.91, 0.98, 0.99]
ks_auc = [0.50, 0.69, 0.87, 0.95, 0.98]
mahal_exceed = [0.10, 0.32, 0.54, 0.75, 0.97]
unlabeled_abstain = [0.10, 0.32, 0.54, 0.75, 0.97]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Labeled Audit Coverage Drop
ax = axes[0, 0]
ax.plot(severities, labeled_drop, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Coverage Drop')
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.7, label='Abstention Threshold (10%)')
ax.set_xticks(severities)
ax.set_xticklabels(sev_labels)
ax.set_xlabel('Corruption Severity', fontsize=12)
ax.set_ylabel('Coverage Drop (%)', fontsize=12)
ax.set_title('Labeled Audit Coverage Drop', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(-1, 18)

# Plot 2: MMD and KS Detection AUC
ax = axes[0, 1]
ax.plot(severities, mmd_auc, 'o-', color='#3498db', linewidth=2, markersize=8, label='MMD AUC')
ax.plot(severities, ks_auc, 's-', color='#2ecc71', linewidth=2, markersize=8, label='KS AUC')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xticks(severities)
ax.set_xticklabels(sev_labels)
ax.set_xlabel('Corruption Severity', fontsize=12)
ax.set_ylabel('Detection AUC', fontsize=12)
ax.set_title('Unlabeled Drift Detection AUC', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.05)

# Plot 3: Mahalanobis Exceedance
ax = axes[1, 0]
ax.plot(severities, mahal_exceed, 'o-', color='#8e44ad', linewidth=2, markersize=8, label='Mahalanobis Exceedance')
ax.plot(severities, unlabeled_abstain, 's-', color='#f39c12', linewidth=2, markersize=8, label='Unlabeled Abstain Rate')
ax.set_xticks(severities)
ax.set_xticklabels(sev_labels)
ax.set_xlabel('Corruption Severity', fontsize=12)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Mahalanobis Exceedance & Unlabeled Abstain', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 1.05)

# Plot 4: Comparison - Labeled vs Unlabeled Detection
ax = axes[1, 1]
width = 0.12
x = np.arange(len(severities))
ax.bar(x - 1.5*width, [1-d/100 for d in labeled_drop], width, label='Labeled Audit (1 - Cov Drop)', color='#e74c3c', alpha=0.7, edgecolor='black')
ax.bar(x - 0.5*width, mmd_auc, width, label='MMD AUC', color='#3498db', alpha=0.7, edgecolor='black')
ax.bar(x + 0.5*width, ks_auc, width, label='KS AUC', color='#2ecc71', alpha=0.7, edgecolor='black')
ax.bar(x + 1.5*width, mahal_exceed, width, label='Mahalanobis Exceedance', color='#8e44ad', alpha=0.7, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(sev_labels)
ax.set_xlabel('Corruption Severity', fontsize=12)
ax.set_ylabel('Score / Rate', fontsize=12)
ax.set_title('Labeled vs Unlabeled Detection Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9, loc='lower left')
ax.set_ylim(0, 1.1)

plt.suptitle('Unlabeled Drift Detection vs. Labeled Audit Across Covariate Shift Severities (+3 SD Anthropometric Shift)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/figures/fig5_drift_auditor.png', dpi=300, bbox_inches='tight')
plt.savefig('C:/Users/Raza-SEAGUARD/Downloads/Agentic AI/reproducibility_package/results/figures/fig5_drift_auditor.pdf', bbox_inches='tight')
print("Figure 5 saved!")