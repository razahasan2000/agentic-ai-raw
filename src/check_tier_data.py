#!/usr/bin/env python3
import json
import numpy as np

with open('../results/metrics_verified.json') as f:
    m = json.load(f)

cf = m['counterfactual']
print('=== TIER DISTRIBUTION ===')
print('tier_distribution_mean:', cf['tier_distribution_mean'])
print('tier_radius_counts_mean:', cf['tier_radius_counts_mean'])
print('tier_radius_pct:', cf['tier_radius_pct'])

# Compute exact percentages from per-seed counts
tier1_total = 0
tier2_total = 0
tier3_total = 0
n_high_risk_total = 0

tier_1_0x = 0
tier_1_15x = 0
tier_2_10x = 0
tier_2_15x = 0
tier_3_20x = 0  # Tier 3 is 2.0x

for seed in m['seed_details']:
    n_hr = seed['n_high_risk']
    n_high_risk_total += n_hr
    td = seed['tier_distribution']
    tr = seed['tier_radius_counts']
    
    tier1_total += td.get('Tier 1', 0)
    tier2_total += td.get('Tier 2', 0)
    tier3_total += td.get('Tier 3', 0)
    
    tier_1_0x += tr.get('Tier 1 1.0x', 0)
    tier_1_15x += tr.get('Tier 1 1.5x', 0)
    tier_2_10x += tr.get('Tier 2 1.0x', 0)
    tier_2_15x += tr.get('Tier 2 1.5x', 0)
    # Tier 3 could be at different radii
    for k, v in tr.items():
        if 'Tier 3' in k:
            tier_3_20x += v

print(f'\nTotal high-risk across 20 seeds: {n_high_risk_total}')
print(f'Tier 1 total: {tier1_total} ({tier1_total/n_high_risk_total*100:.1f}%)')
print(f'Tier 2 total: {tier2_total} ({tier2_total/n_high_risk_total*100:.1f}%)')
print(f'Tier 3 total: {tier3_total} ({tier3_total/n_high_risk_total*100:.1f}%)')
print(f'Sum: {(tier1_total+tier2_total+tier3_total)/n_high_risk_total*100:.1f}%')

print(f'\nTier 1 1.0x: {tier_1_0x} ({tier_1_0x/n_high_risk_total*100:.1f}%)')
print(f'Tier 1 1.5x: {tier_1_15x} ({tier_1_15x/n_high_risk_total*100:.1f}%)')
print(f'Tier 2 1.0x: {tier_2_10x} ({tier_2_10x/n_high_risk_total*100:.1f}%)')
print(f'Tier 2 1.5x: {tier_2_15x} ({tier_2_15x/n_high_risk_total*100:.1f}%)')
print(f'Tier 3 (all radii): {tier_3_20x} ({tier_3_20x/n_high_risk_total*100:.1f}%)')
print(f'Sum: {(tier_1_0x+tier_1_15x+tier_2_10x+tier_2_15x+tier_3_20x)/n_high_risk_total*100:.1f}%')