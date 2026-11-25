#!/usr/bin/env python3
import json

with open('../results/metrics_verified.json') as f:
    m = json.load(f)

# Compute exact totals from per-seed data
tier_1_0x = 0
tier_1_15x = 0
tier_2_10x = 0
tier_2_15x = 0
tier_1_20x = 0
tier_2_20x = 0
tier_3_any = 0

for seed in m['seed_details']:
    tr = seed['tier_radius_counts']
    tier_1_0x += tr.get('Tier 1 1.0x', 0)
    tier_1_15x += tr.get('Tier 1 1.5x', 0)
    tier_2_10x += tr.get('Tier 2 1.0x', 0)
    tier_2_15x += tr.get('Tier 2 1.5x', 0)
    tier_1_20x += tr.get('Tier 1 2.0x', 0)
    tier_2_20x += tr.get('Tier 2 2.0x', 0)
    for k, v in tr.items():
        if k.startswith('Tier 3'):
            tier_3_any += v

n_total = sum(s['n_high_risk'] for s in m['seed_details'])

print(f'Total high-risk: {n_total}')
print(f'Tier 1 1.0x: {tier_1_0x} ({tier_1_0x/n_total*100:.1f}%)')
print(f'Tier 1 1.5x: {tier_1_15x} ({tier_1_15x/n_total*100:.1f}%)')
print(f'Tier 2 1.0x: {tier_2_10x} ({tier_2_10x/n_total*100:.1f}%)')
print(f'Tier 2 1.5x: {tier_2_15x} ({tier_2_15x/n_total*100:.1f}%)')
print(f'Tier 1 2.0x: {tier_1_20x} ({tier_1_20x/n_total*100:.1f}%)')
print(f'Tier 2 2.0x: {tier_2_20x} ({tier_2_20x/n_total*100:.1f}%)')
print(f'Tier 3 (any): {tier_3_any} ({tier_3_any/n_total*100:.1f}%)')
print(f'Sum: {(tier_1_0x+tier_1_15x+tier_2_10x+tier_2_15x+tier_1_20x+tier_2_20x+tier_3_any)/n_total*100:.1f}%')

# Tier distribution
t1 = sum(s['tier_distribution'].get('Tier 1', 0) for s in m['seed_details'])
t2 = sum(s['tier_distribution'].get('Tier 2', 0) for s in m['seed_details'])
t3 = sum(s['tier_distribution'].get('Tier 3', 0) for s in m['seed_details'])
print(f'\nTier 1: {t1} ({t1/n_total*100:.1f}%)')
print(f'Tier 2: {t2} ({t2/n_total*100:.1f}%)')
print(f'Tier 3: {t3} ({t3/n_total*100:.1f}%)')
print(f'Sum: {(t1+t2+t3)/n_total*100:.1f}%')