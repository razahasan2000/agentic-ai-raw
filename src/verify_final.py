#!/usr/bin/env python3
import json

with open('../results/metrics_verified.json') as f:
    m = json.load(f)

print('=== FINAL VERIFICATION ===')
print()

# 1. Abstract numbers
print('1. ABSTRACT CHECKS:')
print('   LR AUC:', round(m['models']['lr_auc_mean'], 3), '[CI:', round(m['models']['lr_auc_mean']-m['models']['lr_auc_ci95'], 3), '-', round(m['models']['lr_auc_mean']+m['models']['lr_auc_ci95'], 3), ']')
print('   Overall TEST Coverage:', round(m['coverage']['overall_test_mean']*100, 1), '% [CI:', round((m['coverage']['overall_test_mean']-m['coverage']['overall_test_ci95'])*100, 1), '-', round((m['coverage']['overall_test_mean']+m['coverage']['overall_test_ci95'])*100, 1), ']%')
print('   Worst Subgroup:', round(m['coverage']['worst_subgroup_test_mean']*100, 1), '%')
print('   Mondrian Worst:', round(m['baseline_conformal']['Mondrian']['mean']*100, 1), '% (NOT 89.4%)')
print('   Standard EB Worst:', round(m['baseline_conformal']['StandardEB']['mean']*100, 1), '% (NOT 89.3%)')
print('   ECE:', round(m['auditor']['val_ece_mean']*100, 1), '% [CI:', round((m['auditor']['val_ece_mean']-m['auditor']['val_ece_ci95'])*100, 1), '-', round((m['auditor']['val_ece_mean']+m['auditor']['val_ece_ci95'])*100, 1), ']%')
print('   LLM Loop 2: 94.2% final (67.4% -> 91.9% -> 94.2%)')
print()

# 2. Counterfactual Tier 3
print('2. TIER 3 CHECKS:')
cf = m['counterfactual']
print('   Tier 3 patients:', cf['tier_distribution_mean'].get('Tier 3', 0), '(', cf['tier_radius_pct'].get('Tier 3 2.0x', 0), '%)')
print('   Tier-radius:', cf['tier_radius_pct'])
print()

# 3. ECE
print('3. ECE CHECK:')
print('   Val ECE:', round(m['auditor']['val_ece_mean']*100, 1), '% (should be ~3.9%, NOT 26.2%)')
print()

# 4. Conformal baselines
print('4. CONFORMAL BASELINES (20 seeds):')
for k, v in m['baseline_conformal'].items():
    print('   ', k, ': Worst=', round(v['mean']*100, 1), '%')
print()

# 5. LLM
print('5. LLM LOOP 2 (86 patients, seed 3):')
llm = m['llm_evaluation']['Qwen/Qwen2.5-1.5B-Instruct']
for v in ['unconstrained', 'schema_prompted', 'loop1_schema', 'full_agentic']:
    p = sum(llm[v]['passes'])/len(llm[v]['passes'])*100
    print('   ', v, ':', round(p, 1), '%')
print()

# 6. Table 10 vs Figure 5
print('6. DRIFT (Table 10):')
for sev in ['0.0', '0.25', '0.5', '0.75', '1.0']:
    d = m['drift'][sev]
    print('   ', sev, ': Labeled Drop=', round(d['labeled_cov_drop'], 1), '%, Mahal=', round(d['mahal_exceedance'], 2))

print()
print('=== ALL CHECKS COMPLETE ===')