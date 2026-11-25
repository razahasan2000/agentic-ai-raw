import pickle

with open('../results/clean_pipeline_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

cf_results = artifacts['cf_results']
print('Total counterfactual results:', len(cf_results))
high_risk = [r for r in cf_results if r['p_initial_cal'] > 0.5]
print('High-risk (p>0.5):', len(high_risk))
for r in high_risk[:5]:
    print(f'  ID {r["subject_id"]}: p={r["p_initial_cal"]:.3f}, tier={r["tier_used"]}, r_mult={r["radius_mult"]}')

# Also check evidence_list
evidence_list = artifacts['evidence_list']
print('\\nTotal evidence list:', len(evidence_list))
high_risk_evidence = [e for e in evidence_list if e['predicted_risk'] > 0.5 and e.get('counterfactual')]
print('High-risk with counterfactuals:', len(high_risk_evidence))
for e in high_risk_evidence[:5]:
    print(f'  ID {e["subject_id"]}: risk={e["predicted_risk"]:.3f}, cf_tier={e["counterfactual"]["tier_used"] if e["counterfactual"] else "None"}')