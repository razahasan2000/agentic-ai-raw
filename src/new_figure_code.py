# LEFT COLUMN: DETERMINISTIC TOOLS
tool_bg = FancyBboxPatch((0.3, 0.5), 4.2, 9.5, boxstyle='round,pad=0.1', 
                          facecolor=colors['tool'], edgecolor='#BF360C', linewidth=2)
ax.add_patch(tool_bg)
ax.text(2.4, 10.0, 'DETERMINISTIC STATISTICAL TOOLS', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#BF360C')

tools = [
    (0.5, 8.5, 'Clinical Feature Embedder', 'ECDF -> Phi^-1 transform\nReference: non-diabetic controls\nOutput: z in R^13 (SD units)'),
    (0.5, 7.0, 'Calibrated Risk Predictor', 'Logistic Regression (no HbA1c/glucose)\nPlatt scaling on 15% calibration split\nAUC = 0.761 (test)'),
    (0.5, 5.5, 'Conformal Calibrator (Split)', 'Split conformal on 15% val set\nEmpirical Bayes shrinkage\nTarget coverage: 90%'),
    (0.5, 4.0, 'QCQP Counterfactual Solver', 'Analytical LR solution + SLSQP fallback\nLedoit-Wolf Sigma_g (full covariance)\nCalibrated logits (w_cal, b_cal)'),
    (0.5, 2.5, 'Attribution Tester (LOCO)', 'Leave-One-Covariate-Out Brier diff\nBH correction (FDR = 10%)\nSignificant: Age, Waist, BMI, SBP'),
    (0.5, 1.0, 'Deployable Auditor', 'Val subgroup coverage & ECE\nAbstains if coverage < 80% or ECE > 0.25'),
]

for x, y, title, desc in tools:
    box = FancyBboxPatch((x, y), 3.8, 1.3, boxstyle='round,pad=0.1', 
                          facecolor='white', edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(box)
    ax.text(2.4, y + 0.65, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(2.4, y + 0.2, desc, ha='center', va='center', fontsize=9, color='#333')

# MIDDLE COLUMN: AGENTIC LAYER
agent_bg = FancyBboxPatch((5.0, 0.5), 5.5, 9.5, boxstyle='round,pad=0.1', 
                           facecolor=colors['agent'], edgecolor='#2E7D32', linewidth=2)
ax.add_patch(agent_bg)
ax.text(7.75, 10.0, 'AGENTIC LAYER (STATEFUL, TOOL-CALLING)', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#2E7D32')

# Orchestrator Agent (moved up to avoid header overlap)
orch_box = FancyBboxPatch((5.3, 7.8), 4.9, 2.0, boxstyle='round,pad=0.1', 
                           facecolor='white', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(orch_box)
ax.text(7.75, 9.5, 'Orchestrator Agent', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(7.75, 9.1, 'State: patient z, subgroup, risk, tier, radius', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 8.7, 'Calls: Conformal Calibrator -> QCQP Solver -> Attribution', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 8.3, 'Escalation: Tier 1 -> 2 -> 3 | Radius 1.0x -> 1.5x -> 2.0x', ha='center', va='center', fontsize=9, color='#2E7D32')

# LOOP 1: Counterfactual Recovery
loop1_bg = FancyBboxPatch((5.3, 5.5), 4.9, 2.0, boxstyle='round,pad=0.1', 
                           facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=1.5, linestyle='--')
ax.add_patch(loop1_bg)
ax.text(7.75, 7.2, 'LOOP 1: Counterfactual Recovery', ha='center', va='center', fontsize=10, fontweight='bold', color='#F57F17')
ax.text(7.75, 6.8, 'Orchestrator -> QCQP Solver', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 6.4, 'if infeasible or M > r^2:', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 6.0, '   escalate tier: 1 -> 2 -> 3', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 5.6, '   relax radius: 1.0x -> 1.5x -> 2.0x', ha='center', va='center', fontsize=9, color='#333')

# Synthesis Agent
synth_box = FancyBboxPatch((5.3, 3.5), 4.9, 2.0, boxstyle='round,pad=0.1', 
                            facecolor='white', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(synth_box)
ax.text(7.75, 4.6, 'Synthesis Agent (LLM)', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(7.75, 4.2, 'Qwen-2.5-7B-Instruct (temp=0.0)', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 3.9, 'Input: JSON Evidence Schema', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 3.6, 'Output: Structured clinician narrative', ha='center', va='center', fontsize=9, color='#333')

# LOOP 2: Narrative Verification & Repair
loop2_bg = FancyBboxPatch((5.3, 0.5), 4.9, 2.8, boxstyle='round,pad=0.1', 
                           facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=1.5, linestyle='--')
ax.add_patch(loop2_bg)
ax.text(7.75, 3.0, 'LOOP 2: Verification & Repair (M=3)', ha='center', va='center', fontsize=10, fontweight='bold', color='#F57F17')
ax.text(7.75, 2.6, 'Synthesis -> Verifier Agent', ha='center', va='center', fontsize=9, color='#333')
ax.text(7.75, 2.2, 'Checks: risk +/-0.5%, coverage, frozen feats, subgroup caveat', ha='center', va='center', fontsize=8, color='#333')
ax.text(7.75, 1.8, 'On fail: structured error trace -> Synthesis (retry, M=3)', ha='center', va='center', fontsize=8, color='#F57F17')
ax.text(7.75, 1.4, 'Verifier: regex/AST extraction of all numeric claims', ha='center', va='center', fontsize=8, color='#333')

# RIGHT COLUMN: OUTPUTS
output_bg = FancyBboxPatch((11.2, 0.5), 5.0, 9.5, boxstyle='round,pad=0.1', 
                           facecolor=colors['output'], edgecolor='#AD1457', linewidth=2)
ax.add_patch(output_bg)
ax.text(13.7, 10.0, 'OUTPUTS & GUARANTEES', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#AD1457')

outputs = [
    (11.5, 7.5, 'Verified Clinician Explanation', 
     'Risk score +/-0.5% of calibrated\n'
     'Prediction set Cg with 90% coverage\n'
     'Counterfactual: tier, delta, M, inside\n'
     'Key drivers (LOCO BH-significant)\n'
     'Subgroup caveats stated\n'
     'No frozen-feature changes'),
    (11.5, 5.0, 'Audit Certificate', 
     'Val subgroup coverage per group\n'
     'Val ECE (out-of-fold)\n'
     'Tier used, radius multiplier\n'
     'Verifier pass/fail log (M retries)'),
    (11.5, 2.5, 'Abstain / Defer to Clinician', 
     'Triggers:\n'
     'Val subgroup coverage < 80%\n'
     'Val ECE > 0.25\n'
     'Verifier max retries (M=3) exceeded\n'
     'Covariate shift detected'),
]

for x, y, title, desc in outputs:
    box = FancyBboxPatch((x, y), 4.5, 2.3, boxstyle='round,pad=0.1', 
                         facecolor='white', edgecolor='#AD1457' if y > 3 else '#C62828', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 2.25, y + 2.0, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#AD1457' if y > 3 else '#C62828')
    ax.text(x + 2.25, y + 0.8, desc, ha='center', va='center', fontsize=9, color='#333')

# ARROWS - routed to avoid text boxes
# Data flow (solid black) - left column only
data_flow = [
    ((2.4, 8.5), (2.4, 8.2)),
    ((2.4, 7.0), (2.4, 6.7)),
    ((2.4, 5.5), (2.4, 5.2)),
    ((2.4, 4.0), (2.4, 3.7)),
    ((2.4, 2.5), (2.4, 2.2)),
]
for start, end in data_flow:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, 
                            color='#2C3E50', linewidth=1.5)
    ax.add_patch(arrow)

# Orchestrator calls tools (dashed blue) - routed to avoid text
tool_calls = [
    ((4.7, 8.8), (0.5, 8.8)),   # Orchestrator -> Embedder
    ((4.7, 7.7), (0.5, 7.3)),   # Orchestrator -> Predictor
    ((4.7, 7.0), (0.5, 5.8)),   # Orchestrator -> Conformal
    ((4.7, 6.5), (0.5, 4.3)),   # Orchestrator -> QCQP
    ((4.7, 6.0), (0.5, 3.0)),   # Orchestrator -> Attribution
    ((4.7, 2.7), (0.5, 1.5)),   # Orchestrator -> Auditor
]
for start, end in tool_calls:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=12, 
                            color='#1565C0', linewidth=1.2, linestyle='--',
                            connectionstyle='arc3,rad=-0.3')
    ax.add_patch(arrow)

# Loop 1: Counterfactual Recovery (dotted red) - routed around boxes
loop1_arrows = [
    ((7.75, 5.5), (2.4, 4.3)),   # Orchestrator -> QCQP (left arc)
    ((2.4, 2.5), (7.75, 6.8)),   # QCQP -> Orchestrator (right arc)
]
for start, end in loop1_arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=12, 
                            color='#C62828', linewidth=1.5, linestyle=':',
                            connectionstyle='arc3,rad=0.4')
    ax.add_patch(arrow)
ax.text(4.5, 4.5, 'Loop 1\nRecovery', ha='center', va='center', fontsize=8, fontweight='bold', color='#F57F17',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#F57F17', alpha=0.9))

# Orchestrator -> Synthesis Agent
arrow = FancyArrowPatch((7.75, 7.8), (7.75, 5.3), arrowstyle='->', mutation_scale=15, 
                        color='#2C3E50', linewidth=1.5)
ax.add_patch(arrow)

# Loop 2: Verification & Repair (dotted red)
loop2_arrows = [
    ((7.75, 3.5), (7.75, 3.3)),  # Synthesis -> Verifier
    ((7.75, 0.5), (7.75, 0.8)),  # Verifier -> Synthesis (feedback)
]
for start, end in loop2_arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=12, 
                            color='#C62828', linewidth=1.5, linestyle=':')
    ax.add_patch(arrow)
ax.text(7.75, 0.2, 'Loop 2\nVerify & Repair', ha='center', va='center', fontsize=8, fontweight='bold', color='#F57F17',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#F57F17', alpha=0.9))

# Synthesis Agent -> Outputs
output_arrows = [
    ((10.3, 4.8), (11.2, 7.8)),
    ((10.3, 4.8), (11.2, 5.0)),
    ((10.3, 4.8), (11.2, 2.5)),
]
for start, end in output_arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, 
                            color='#2C3E50', linewidth=1.5,
                            connectionstyle='arc3,rad=0.2')
    ax.add_patch(arrow)

# Input data
arrow = FancyArrowPatch((2.4, 1.0), (2.4, 0.6), arrowstyle='->', mutation_scale=15, 
                        color='#2C3E50', linewidth=1.5)
ax.add_patch(arrow)
ax.text(2.4, 0.3, 'NHANES 2017-2018\n5,533 adults', ha='center', va='center', fontsize=9, color='#333',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F4FD', edgecolor='#BF360C'))

# Legend
legend_items = [
    ('Solid black', 'Main data flow'),
    ('Dashed blue', 'Orchestrator tool calls'),
    ('Dotted red', 'Feedback loops (recovery/verification)'),
    ('Orange bg', 'Deterministic tools'),
    ('Green bg', 'Stateful agents'),
    ('Pink bg', 'Outputs (green check = guarantee, red X = abstain)'),
]

y_start = -0.5
for i, (style, desc) in enumerate(legend_items):
    y = y_start - i * 0.35
    ax.text(0.5, y, style, fontsize=8, fontweight='bold', color='#1A1A2E')
    ax.text(2.5, y, desc, fontsize=8, color='#1A1A2E')

ax.set_title('Figure 1: System Architecture of the Conformal-Grounded Agentic Risk Explanation Framework\nwith Counterfactual Recovery (Loop 1) and Narrative Verification & Repair (Loop 2)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print('   Generated fig_architecture.png')