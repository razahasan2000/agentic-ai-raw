"""
Genuine Agentic AI Framework for Conformal Risk Explanation
============================================================

Components:
1. Tool-Calling Orchestrator with iterative recovery
2. LLM Explanation Agent with structured JSON schema + Verifier Agent
3. Empirical evaluation: Agentic vs Non-Agentic baselines
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Try to import LLM libraries (will work if available)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

RESULTS_DIR = Path(r"C:\Users\Raza-SEAGUARD\Downloads\Agentic AI\reproducibility_package\results")

# ============================================================
# TOOLS: Concrete implementations the agents can call
# ============================================================

class ToolResult:
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error

class ConformalTool:
    """Tool for conformal prediction set computation"""
    
    def __init__(self, artifacts: Dict):
        self.artifacts = artifacts
        self.z_cols = artifacts['z_cols']
        self.final_quantiles = artifacts['final_quantiles']
        self.global_q = artifacts['global_q']
        self.alpha = artifacts['alpha']
        self.cal_clf = artifacts['cal_clf']
    
    def compute_prediction_set(self, z: np.ndarray, subgroup: str) -> ToolResult:
        """Compute conformal prediction set for a patient"""
        try:
            p = self.cal_clf.predict_proba(z.reshape(1, -1))[0, 1]
            q = self.final_quantiles.get(subgroup, self.global_q)
            pred_set = []
            if (1 - p) <= q: pred_set.append(1)
            if p <= q: pred_set.append(0)
            return ToolResult(True, {
                'predicted_risk': float(p),
                'prediction_set': pred_set,
                'quantile_used': float(q),
                'coverage_target': 1 - self.alpha
            })
        except Exception as e:
            return ToolResult(False, error=str(e))

class CounterfactualTool:
    """Tool for counterfactual optimization with multiple strategies"""
    
    def __init__(self, artifacts: Dict):
        self.artifacts = artifacts
        self.z_cols = artifacts['z_cols']
        self.actionable_idx = artifacts['actionable_idx']
        self.subgroup_stats = artifacts['subgroup_stats']
        self.w = artifacts['w']
        self.b = artifacts['b']
        self.cal_clf = artifacts['cal_clf']
        self.base_clf = artifacts['base_clf']
        
        self.mu_global = artifacts['subgroup_stats'].get('global_mu', 
            np.mean([s['mu'] for s in artifacts['subgroup_stats'].values() if isinstance(s, dict)], axis=0))
        self.prec_global = artifacts['subgroup_stats'].get('global_prec',
            1.0 / (np.var([s['mu'] for s in artifacts['subgroup_stats'].values() if isinstance(s, dict)], axis=0) + 1e-6))
        self.r2_global = artifacts['subgroup_stats'].get('global_r2', 19.276)
    
    def solve(self, z: np.ndarray, subgroup: str, 
              target_prob: float = 0.5,
              intervention_tier: str = 'lifestyle',
              relax_ellipsoid: float = 1.0) -> ToolResult:
        """
        Solve counterfactual with configurable intervention tier and ellipsoid relaxation.
        
        intervention_tier: 'lifestyle' (BMI, waist, activity, diet) 
                          'clinical' (+ BP) 
                          'pharmacological' (+ all actionable)
        relax_ellipsoid: multiplier on r2 (1.0 = strict, 2.0 = permissive)
        """
        try:
            stats = self.subgroup_stats.get(subgroup, {
                'mu': self.mu_global, 'prec': self.prec_global, 'r2': self.r2_global
            })
            mu, prec, r2 = stats['mu'], stats['prec'], stats['r2'] * relax_ellipsoid
            
            # Define actionable features by tier
            tier_features = {
                'lifestyle': ['z_BMI', 'z_Waist_cm', 'z_Vigorous_Activity', 'z_Moderate_Activity', 'z_INDFMPIR'],
                'clinical': ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP', 'z_Vigorous_Activity', 'z_Moderate_Activity', 'z_INDFMPIR'],
                'pharmacological': ['z_BMI', 'z_Waist_cm', 'z_SBP', 'z_DBP', 'z_INDFMPIR', 
                                   'z_Current_Smoker', 'z_Former_Smoker', 'z_Current_Drinker',
                                   'z_Vigorous_Activity', 'z_Moderate_Activity']
            }
            
            actionable_names = tier_features.get(intervention_tier, tier_features['lifestyle'])
            actionable_idx = [self.z_cols.index(f) for f in actionable_names if f in self.z_cols]
            
            if len(actionable_idx) == 0:
                return ToolResult(False, error="No actionable features for this tier")
            
            # Linear constraint: w^T delta <= c (base LR logits)
            c = np.log(target_prob / (1 - target_prob)) - (np.dot(self.w, z) + self.b)
            
            w_act = self.w[actionable_idx]
            w_act_norm = np.linalg.norm(w_act)
            
            if w_act_norm < 1e-10:
                return ToolResult(False, error="No gradient in actionable subspace")
            
            # Unconstrained solution
            delta_act_unconstrained = - (c / (w_act_norm**2)) * w_act
            z_new = z.copy()
            z_new[actionable_idx] = z[actionable_idx] + delta_act_unconstrained
            mahal = np.sum((z_new - mu)**2 * prec)
            
            if mahal <= r2:
                delta = np.zeros(len(z))
                delta[actionable_idx] = delta_act_unconstrained
                p_final_cal = self.cal_clf.predict_proba(z_new.reshape(1, -1))[0, 1]
                p_final_base = self.base_clf.predict_proba(z_new.reshape(1, -1))[0, 1]
                p_initial_cal = self.cal_clf.predict_proba(z.reshape(1, -1))[0, 1]
                return ToolResult(True, {
                    'delta': delta.tolist(),
                    'delta_norm': float(np.linalg.norm(delta)),
                    'p_final_calibrated': float(p_final_cal),
                    'p_final_base': float(p_final_base),
                    'p_initial_calibrated': float(p_initial_cal),
                    'mahalanobis': float(mahal),
                    'inside_ellipsoid': True,
                    'intervention_tier': intervention_tier,
                    'relax_ellipsoid': relax_ellipsoid,
                    'actionable_changes': {self.z_cols[i]: float(delta[i]) for i in actionable_idx if abs(delta[i]) > 1e-3}
                })
            
            # Constrained optimization via SLSQP
            from scipy.optimize import minimize
            
            n_act = len(actionable_idx)
            bounds = [(-5.0, 5.0)] * n_act
            
            def obj(delta_act):
                return np.sum(delta_act**2)
            
            def constraint_risk(delta_act):
                full_delta = np.zeros(len(z))
                full_delta[actionable_idx] = delta_act
                p_new = self.cal_clf.predict_proba((z + full_delta).reshape(1, -1))[0, 1]
                return target_prob - p_new
            
            def constraint_ellipsoid(delta_act):
                full_delta = np.zeros(len(z))
                full_delta[actionable_idx] = delta_act
                z_new = z + full_delta
                mahal = np.sum((z_new - mu)**2 * prec)
                return r2 - mahal
            
            constraints = [
                {'type': 'ineq', 'fun': constraint_risk},
                {'type': 'ineq', 'fun': constraint_ellipsoid}
            ]
            
            best_delta = None
            best_obj = np.inf
            best_p = 1.0
            best_mahal = np.inf
            
            for restart in [0] + list(range(5)):
                x0 = delta_act_unconstrained if restart == 0 else np.random.randn(n_act) * 0.1
                try:
                    result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                                    constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
                    if result.success:
                        full_delta = np.zeros(len(z))
                        full_delta[actionable_idx] = result.x
                        p_final = self.cal_clf.predict_proba((z + full_delta).reshape(1, -1))[0, 1]
                        mahal_final = np.sum((z + full_delta - mu)**2 * prec)
                        obj_val = np.sum(result.x**2)
                        if p_final <= target_prob and mahal_final <= r2 and obj_val < best_obj:
                            best_obj = obj_val
                            best_delta = full_delta
                            best_p = p_final
                            best_mahal = mahal_final
                except:
                    continue
            
            if best_delta is not None:
                delta = best_delta
                p_final_cal = best_p
                p_final_base = self.base_clf.predict_proba((z + delta).reshape(1, -1))[0, 1]
                p_initial_cal = self.cal_clf.predict_proba(z.reshape(1, -1))[0, 1]
                mahal_final = best_mahal
            else:
                delta = np.zeros(len(z))
                p_final_cal = self.cal_clf.predict_proba(z.reshape(1, -1))[0, 1]
                p_final_base = self.base_clf.predict_proba(z.reshape(1, -1))[0, 1]
                p_initial_cal = self.cal_clf.predict_proba(z.reshape(1, -1))[0, 1]
                mahal_final = np.sum((z - mu)**2 * prec)
            
            return ToolResult(True, {
                'delta': delta.tolist(),
                'delta_norm': float(np.linalg.norm(delta)),
                'p_final_calibrated': float(p_final_cal),
                'p_final_base': float(p_final_base),
                'p_initial_calibrated': float(p_initial_cal),
                'mahalanobis': float(mahal_final),
                'inside_ellipsoid': mahal_final <= r2,
                'intervention_tier': intervention_tier,
                'relax_ellipsoid': relax_ellipsoid,
                'actionable_changes': {self.z_cols[i]: float(delta[i]) for i in actionable_idx if abs(delta[i]) > 1e-3}
            })
            
        except Exception as e:
            return ToolResult(False, error=str(e))

class AttributionTool:
    """Tool for conformal marginal attribution"""
    
    def __init__(self, artifacts: Dict):
        self.artifacts = artifacts
        self.z_cols = artifacts['z_cols']
        self.significance_results = artifacts['significance_results']
        self.global_significance = artifacts['global_significance']
        self.cal_clf = artifacts['cal_clf']
    
    def get_significant_features(self, subgroup: str, alpha: float = 0.1) -> ToolResult:
        try:
            if subgroup not in self.significance_results:
                return ToolResult(True, {'significant_features': [], 'all_features': {}})
            
            sig = []
            all_feat = {}
            for feat_name, res in self.significance_results[subgroup].items():
                if len(res['p_values']) > 0:
                    p_val = float(res['p_values'][0])
                    eff = float(res['mean_effect'])
                    all_feat[feat_name] = {'p_value': p_val, 'effect': eff, 'significant': p_val < alpha}
                    if p_val < alpha:
                        sig.append(feat_name)
            
            return ToolResult(True, {
                'significant_features': sig,
                'all_features': all_feat,
                'alpha': alpha
            })
        except Exception as e:
            return ToolResult(False, error=str(e))

class AuditorTool:
    """Tool for deployable auditing"""
    
    def __init__(self, artifacts: Dict):
        self.artifacts = artifacts
        self.cal_coverage = artifacts['cal_coverage']
        self.cal_ece = artifacts['cal_ece']
        self.abstain_subgroups = artifacts['abstain_subgroups']
        self.abstain_ece = artifacts['abstain_ece']
    
    def check_abstention(self, subgroup: str) -> ToolResult:
        sg_cov = self.cal_coverage.get(subgroup, 1.0)
        abstain = (subgroup in self.abstain_subgroups) or self.abstain_ece
        reason = None
        if subgroup in self.abstain_subgroups:
            reason = f"Subgroup cal coverage {sg_cov:.2f} < 0.80"
        elif self.abstain_ece:
            reason = f"Cal ECE {self.cal_ece:.4f} > 0.25"
        return ToolResult(True, {
            'abstain': abstain,
            'reason': reason,
            'subgroup_cal_coverage': sg_cov,
            'cal_ece': self.cal_ece
        })

# ============================================================
# AGENT STATE & MEMORY
# ============================================================

@dataclass
class AgentState:
    """Persistent state for the agentic loop"""
    patient_id: str
    subgroup: str
    z_vector: np.ndarray
    predicted_risk: float
    prediction_set: List[int]
    iteration: int = 0
    max_iterations: int = 5
    
    # Counterfactual trajectory
    counterfactual_history: List[Dict] = field(default_factory=list)
    current_intervention_tier: str = 'lifestyle'
    current_relax_ellipsoid: float = 1.0
    
    # Attribution
    significant_features: List[str] = field(default_factory=list)
    
    # Audit
    abstain: bool = False
    abstain_reason: str = None
    
    # Explanation generation
    explanation_drafts: List[str] = field(default_factory=list)
    verifier_feedback: List[str] = field(default_factory=list)
    final_explanation: str = None
    
    def to_dict(self):
        return {
            'patient_id': self.patient_id,
            'subgroup': self.subgroup,
            'predicted_risk': self.predicted_risk,
            'prediction_set': self.prediction_set,
            'iteration': self.iteration,
            'counterfactual_history': self.counterfactual_history,
            'current_intervention_tier': self.current_intervention_tier,
            'current_relax_ellipsoid': self.current_relax_ellipsoid,
            'significant_features': self.significant_features,
            'abstain': self.abstain,
            'abstain_reason': self.abstain_reason,
            'final_explanation': self.final_explanation
        }

# ============================================================
# BASE AGENT CLASS
# ============================================================

class Agent(ABC):
    def __init__(self, name: str, tools: Dict[str, Any]):
        self.name = name
        self.tools = tools
    
    @abstractmethod
    def act(self, state: AgentState) -> AgentState:
        pass

# ============================================================
# EXPLANATION AGENT (LLM-based with tool calling)
# ============================================================

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_summary": {"type": "string"},
        "risk_assessment": {"type": "string"},
        "conformal_guarantee": {"type": "string"},
        "counterfactual_recommendation": {"type": "string"},
        "key_drivers": {"type": "array", "items": {"type": "string"}},
        "abstention_note": {"type": "string"},
        "confidence_bounds": {
            "type": "object",
            "properties": {
                "risk_score": {"type": "number"},
                "coverage_level": {"type": "number"},
                "counterfactual_feasibility": {"type": "string"}
            },
            "required": ["risk_score", "coverage_level", "counterfactual_feasibility"]
        }
    },
    "required": ["patient_summary", "risk_assessment", "conformal_guarantee", 
                  "counterfactual_recommendation", "key_drivers", "confidence_bounds"]
}

VERIFIER_PROMPT = """You are a Verifier Agent. Check the LLM explanation against the ground truth statistical outputs.
Reject if ANY of the following:
1. Risk score differs from ground truth by > 0.5%
2. Coverage level differs from ground truth
3. Counterfactual claims risk reduction not supported by optimization output
4. Recommends changing immutable features (age, sex, race)
5. Omits subgroup-specific caveats
6. States confidence not backed by conformal guarantee
7. Hallucinates any numeric value

Return JSON: {"pass": true/false, "errors": ["list of specific errors"], "corrected_explanation": "optional corrected text"}"""

class LLMExplanationAgent(Agent):
    """LLM-based explanation agent with structured output"""
    
    def __init__(self, tools: Dict, model: str = "llama3", use_ollama: bool = True):
        super().__init__("ExplanationAgent", tools)
        self.model = model
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.use_openai = not self.use_ollama and OPENAI_AVAILABLE
        
        if self.use_ollama:
            self.client = ollama
        elif self.use_openai:
            self.client = OpenAI()
        else:
            self.client = None
    
    def _build_prompt(self, state: AgentState) -> str:
        cf = state.counterfactual_history[-1] if state.counterfactual_history else {}
        
        # Format numbers safely
        p_final = cf.get('p_final_calibrated', 'N/A')
        p_initial = cf.get('p_initial_calibrated', 'N/A')
        delta_norm = cf.get('delta_norm', 'N/A')
        
        p_final_str = f"{p_final:.1%}" if isinstance(p_final, (int, float)) else str(p_final)
        p_initial_str = f"{p_initial:.1%}" if isinstance(p_initial, (int, float)) else str(p_initial)
        delta_norm_str = f"{delta_norm:.3f}" if isinstance(delta_norm, (int, float)) else str(delta_norm)
        
        return f"""Generate a clinician-facing explanation for diabetes risk assessment.

GROUND TRUTH (must match exactly):
- Patient ID: {state.patient_id}
- Subgroup: {state.subgroup}
- Predicted risk: {state.predicted_risk:.1%}
- Prediction set: {state.prediction_set} (target coverage 90%)
- Conformal guarantee: 90% coverage conditional on subgroup
- Counterfactual: {p_final_str} (from {p_initial_str}) 
  ||delta||={delta_norm_str} SD, inside_ellipsoid={cf.get('inside_ellipsoid', False)}
  Tier: {cf.get('intervention_tier', 'lifestyle')}, ellipsoid_relax={cf.get('relax_ellipsoid', 1.0)}
  Changes: {cf.get('actionable_changes', {})}
- Significant features: {state.significant_features}
- Abstention: {state.abstain} ({state.abstain_reason if state.abstain else 'none'})

Output JSON matching schema exactly. Every numeric claim must match ground truth."""
    
    def _call_llm(self, prompt: str) -> str:
        if self.use_ollama:
            response = self.client.chat(model=self.model, messages=[
                {"role": "system", "content": "You are a medical AI explanation generator. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ], format=EXPLANATION_SCHEMA, options={"temperature": 0.1})
            return response['message']['content']
        elif self.use_openai:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical AI explanation generator. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return response.choices[0].message.content
        else:
            # Fallback: template-based
            return self._template_fallback(prompt)
    
    def _template_fallback(self, prompt: str) -> str:
        """Deterministic template when no LLM available"""
        import re
        # Extract ground truth from prompt
        risk_match = re.search(r'Predicted risk: ([\d.]+)%', prompt)
        risk = float(risk_match.group(1))/100 if risk_match else 0.5
        
        return json.dumps({
            "patient_summary": f"Patient {re.search(r'Patient ID: (\d+)', prompt).group(1)} in {re.search(r'Subgroup: (\w+)', prompt).group(1)} subgroup",
            "risk_assessment": f"Predicted prevalent diabetes risk: {risk:.1%}.",
            "conformal_guarantee": "90% coverage conditional on subgroup (distribution-free).",
            "counterfactual_recommendation": "Lifestyle modifications (BMI, waist, activity) recommended to reduce risk.",
            "key_drivers": ["BMI", "Waist circumference", "Physical activity"],
            "abstention_note": "None.",
            "confidence_bounds": {
                "risk_score": risk,
                "coverage_level": 0.90,
                "counterfactual_feasibility": "Inside plausibility ellipsoid"
            }
        })
    
    def act(self, state: AgentState) -> AgentState:
        if state.abstain:
            state.final_explanation = f"ABSTAINED: {state.abstain_reason}. Cannot provide reliable explanation for subgroup {state.subgroup}."
            return state
        
        prompt = self._build_prompt(state)
        raw_output = self._call_llm(prompt)
        
        try:
            explanation_json = json.loads(raw_output)
            state.explanation_drafts.append(raw_output)
        except json.JSONDecodeError:
            state.explanation_drafts.append(raw_output)
            explanation_json = self._template_fallback(prompt)
        
        # Store for verifier
        state.final_explanation = json.dumps(explanation_json, indent=2)
        return state

class VerifierAgent(Agent):
    """Deterministic verifier that checks LLM output against ground truth"""
    
    def __init__(self, tools: Dict):
        super().__init__("VerifierAgent", tools)
    
    def act(self, state: AgentState) -> AgentState:
        if state.abstain or not state.final_explanation:
            return state
        
        try:
            expl = json.loads(state.final_explanation)
        except:
            state.verifier_feedback.append({"pass": False, "errors": ["Invalid JSON output"]})
            return state
        
        errors = []
        
        # Check risk score
        if 'confidence_bounds' in expl:
            claimed_risk = expl['confidence_bounds'].get('risk_score', 0)
            if abs(claimed_risk - state.predicted_risk) > 0.005:
                errors.append(f"Risk score mismatch: claimed {claimed_risk:.3f}, actual {state.predicted_risk:.3f}")
        
        # Check coverage
        if 'confidence_bounds' in expl:
            claimed_cov = expl['confidence_bounds'].get('coverage_level', 0)
            if abs(claimed_cov - 0.90) > 0.001:
                errors.append(f"Coverage mismatch: claimed {claimed_cov:.3f}, actual 0.900")
        
        # Check counterfactual
        if state.counterfactual_history:
            cf = state.counterfactual_history[-1]
            if 'counterfactual_recommendation' in expl:
                # Check if recommendation matches feasibility
                if not cf.get('inside_ellipsoid', False) and 'feasible' in expl['counterfactual_recommendation'].lower():
                    errors.append("Counterfactual claimed feasible but optimization returned out-of-ellipsoid")
        
        # Check immutable features - only flag if explicitly recommends changing them
        # Look for patterns like "reduce age", "change sex", "modify race"
        immutable_patterns = [
            ('age', ['reduce age', 'lower age', 'decrease age', 'change age']),
            ('sex', ['change sex', 'modify sex', 'alter sex']),
            ('race', ['change race', 'modify race', 'alter race']),
            ('gender', ['change gender', 'modify gender', 'alter gender'])
        ]
        expl_lower = state.final_explanation.lower()
        for feat, patterns in immutable_patterns:
            for pattern in patterns:
                if pattern in expl_lower:
                    errors.append(f"Recommends changing immutable feature: {feat} ('{pattern}')")
                    break
        
        # Check subgroup caveat
        if state.subgroup and state.subgroup.lower() not in state.final_explanation.lower():
            errors.append(f"Missing subgroup-specific caveat for {state.subgroup}")
        
        state.verifier_feedback.append({"pass": len(errors) == 0, "errors": errors})
        
        return state

# ============================================================
# ORCHESTRATOR: Tool-Calling with Iterative Recovery
# ============================================================

class OrchestratorAgent:
    """Main orchestrator running the agentic loop"""
    
    def __init__(self, artifacts: Dict):
        # Initialize tools
        self.conformal_tool = ConformalTool(artifacts)
        self.counterfactual_tool = CounterfactualTool(artifacts)
        self.attribution_tool = AttributionTool(artifacts)
        self.auditor_tool = AuditorTool(artifacts)
        
        self.tools = {
            'conformal': self.conformal_tool,
            'counterfactual': self.counterfactual_tool,
            'attribution': self.attribution_tool,
            'auditor': self.auditor_tool
        }
        
        # Sub-agents
        self.explanation_agent = LLMExplanationAgent(self.tools)
        self.verifier_agent = VerifierAgent(self.tools)
        
        # Intervention tiers for recovery
        self.intervention_tiers = ['lifestyle', 'clinical', 'pharmacological']
        self.ellipsoid_relax_factors = [1.0, 1.5, 2.0, 3.0]
    
    def run(self, patient_id: str, z: np.ndarray, subgroup: str) -> AgentState:
        """Execute full agentic loop for one patient"""
        
        # Initialize state
        state = AgentState(
            patient_id=patient_id,
            subgroup=subgroup,
            z_vector=z,
            predicted_risk=0.0,
            prediction_set=[]
        )
        
        # Step 1: Conformal prediction set
        cf_result = self.conformal_tool.compute_prediction_set(z, subgroup)
        if not cf_result.success:
            state.abstain = True
            state.abstain_reason = f"Conformal tool failed: {cf_result.error}"
            return state
        
        state.predicted_risk = cf_result.data['predicted_risk']
        state.prediction_set = cf_result.data['prediction_set']
        
        # Step 2: Audit check
        audit_result = self.auditor_tool.check_abstention(subgroup)
        if audit_result.success:
            state.abstain = audit_result.data['abstain']
            state.abstain_reason = audit_result.data['reason']
            if state.abstain:
                return state
        
        # Step 3: Attribution
        attr_result = self.attribution_tool.get_significant_features(subgroup)
        if attr_result.success:
            state.significant_features = attr_result.data['significant_features']
        
        # Step 4: Counterfactual with iterative recovery (only for high-risk)
        if state.predicted_risk > 0.5:
            state = self._run_counterfactual_loop(state)
        
        # Step 5: Explanation generation with verification
        state = self.explanation_agent.act(state)
        state = self.verifier_agent.act(state)
        
        return state
    
    def _run_counterfactual_loop(self, state: AgentState) -> AgentState:
        """Iterative counterfactual optimization with recovery strategies"""
        
        for tier_idx, tier in enumerate(self.intervention_tiers):
            for relax_idx, relax in enumerate(self.ellipsoid_relax_factors):
                if state.iteration >= state.max_iterations:
                    break
                
                state.current_intervention_tier = tier
                state.current_relax_ellipsoid = relax
                
                cf_result = self.counterfactual_tool.solve(
                    state.z_vector, 
                    state.subgroup,
                    target_prob=0.5,
                    intervention_tier=tier,
                    relax_ellipsoid=relax
                )
                
                state.iteration += 1
                
                if cf_result.success:
                    cf_data = cf_result.data
                    state.counterfactual_history.append(cf_data)
                    
                    # Success criteria: reached target AND inside ellipsoid
                    if cf_data['p_final_calibrated'] <= 0.5 and cf_data['inside_ellipsoid']:
                        # Found valid counterfactual
                        return state
                    
                    # Partial success: inside ellipsoid but didn't reach target
                    if cf_data['inside_ellipsoid'] and cf_data['p_final_calibrated'] < cf_data.get('p_initial_calibrated', 1.0):
                        # Continue searching for better
                        continue
                    
                    # Failed: outside ellipsoid or no improvement
                    # Try next relaxation/tier
                    continue
                else:
                    # Tool failed, try next
                    continue
        
        # If we exhausted all tiers/relaxations, keep best attempt
        if state.counterfactual_history:
            # Pick the one with lowest p_final_calibrated among inside_ellipsoid
            inside_cfs = [cf for cf in state.counterfactual_history if cf.get('inside_ellipsoid', False)]
            if inside_cfs:
                best = min(inside_cfs, key=lambda x: x['p_final_calibrated'])
                state.counterfactual_history = [best]
        
        return state


# ============================================================
# BASELINE AGENTS FOR EVALUATION
# ============================================================

class StaticScriptBaseline:
    """Non-agentic baseline: fixed pipeline, no recovery"""
    
    def __init__(self, artifacts: Dict):
        self.conformal_tool = ConformalTool(artifacts)
        self.counterfactual_tool = CounterfactualTool(artifacts)
        self.attribution_tool = AttributionTool(artifacts)
        self.auditor_tool = AuditorTool(artifacts)
    
    def run(self, patient_id: str, z: np.ndarray, subgroup: str) -> Dict:
        # Single shot, no iteration
        cf_result = self.conformal_tool.compute_prediction_set(z, subgroup)
        if not cf_result.success:
            return {'success': False, 'error': cf_result.error}
        
        audit_result = self.auditor_tool.check_abstention(subgroup)
        if audit_result.data['abstain']:
            return {'success': True, 'abstain': True, 'reason': audit_result.data['reason']}
        
        attr_result = self.attribution_tool.get_significant_features(subgroup)
        
        # Single counterfactual attempt (lifestyle, strict ellipsoid)
        cf_result = None
        if cf_result.data['predicted_risk'] > 0.5:
            cf_result = self.counterfactual_tool.solve(
                z, subgroup, target_prob=0.5,
                intervention_tier='lifestyle', relax_ellipsoid=1.0
            )
        
        # Template explanation
        explanation = self._template_explanation(cf_result.data if cf_result else None,
                                                 cf_result.data if cf_result else None,
                                                 attr_result.data if attr_result else None,
                                                 subgroup)
        
        return {
            'success': True,
            'predicted_risk': cf_result.data['predicted_risk'],
            'prediction_set': cf_result.data['prediction_set'],
            'counterfactual': cf_result.data if cf_result else None,
            'significant_features': attr_result.data['significant_features'] if attr_result.success else [],
            'explanation': explanation,
            'iterations': 1,
            'abstain': False
        }
    
    def _template_explanation(self, conformal_data, cf_data, attr_data, subgroup):
        return f"Patient risk: {conformal_data['predicted_risk']:.1%}. Prediction set: {conformal_data['prediction_set']}. " + \
               (f"Counterfactual: {cf_data['p_final_calibrated']:.1%} (delta={cf_data['delta_norm']:.3f}). " if cf_data else "No counterfactual. ") + \
               f"Significant features: {attr_data['significant_features'] if attr_data else []}. Subgroup: {subgroup}."


class UnconstrainedLLMBaseline:
    """LLM-only baseline without tools or verification"""
    
    def __init__(self, artifacts: Dict):
        self.artifacts = artifacts
        self.llm_agent = LLMExplanationAgent({}, model="llama3")
    
    def run(self, patient_id: str, z: np.ndarray, subgroup: str) -> Dict:
        # Just ask LLM without any tool outputs
        # This will hallucinate
        prompt = f"Explain diabetes risk for patient {patient_id} in {subgroup} subgroup with features {z[:5]}."
        raw = self.llm_agent._call_llm(prompt)
        
        return {
            'success': True,
            'explanation': raw,
            'has_ground_truth': False,
            'iterations': 1,
            'abstain': False
        }


# ============================================================
# EVALUATION FRAMEWORK
# ============================================================

def evaluate_agentic_vs_baselines(artifacts: Dict, test_indices: np.ndarray, 
                                   z_test: np.ndarray, sg_test: np.ndarray,
                                   patient_ids: np.ndarray) -> Dict:
    """Compare agentic loop vs baselines on test set"""
    
    orchestrator = OrchestratorAgent(artifacts)
    static_baseline = StaticScriptBaseline(artifacts)
    llm_baseline = UnconstrainedLLMBaseline(artifacts)
    
    results = {
        'agentic': [],
        'static_script': [],
        'unconstrained_llm': []
    }
    
    n_eval = min(50, len(test_indices))  # Limit for speed
    
    for i in range(n_eval):
        local_idx = test_indices[i]
        z = z_test[local_idx]
        sg = sg_test[local_idx]
        pid = patient_ids[local_idx]
        
        print(f"  Evaluating patient {pid} ({i+1}/{n_eval})...")
        
        # Agentic
        try:
            state = orchestrator.run(str(pid), z, sg)
            results['agentic'].append({
                'patient_id': pid,
                'predicted_risk': state.predicted_risk,
                'abstain': state.abstain,
                'cf_reached_target': any(cf.get('p_final_calibrated', 1.0) <= 0.5 
                                         for cf in state.counterfactual_history),
                'cf_inside_ellipsoid': any(cf.get('inside_ellipsoid', False) 
                                           for cf in state.counterfactual_history),
                'cf_delta_norm': min((cf.get('delta_norm', float('inf')) 
                                     for cf in state.counterfactual_history), default=0),
                'iterations': state.iteration,
                'verifier_pass': all(fb.get('pass', False) 
                                     for fb in state.verifier_feedback if isinstance(fb, dict)),
                'explanation_length': len(state.final_explanation) if state.final_explanation else 0
            })
        except Exception as e:
            results['agentic'].append({'patient_id': pid, 'error': str(e)})
        
        # Static script
        try:
            out = static_baseline.run(str(pid), z, sg)
            results['static_script'].append({
                'patient_id': pid,
                'predicted_risk': out.get('predicted_risk', 0),
                'abstain': out.get('abstain', False),
                'cf_reached_target': out.get('counterfactual', {}).get('p_final_calibrated', 1.0) <= 0.5,
                'cf_inside_ellipsoid': out.get('counterfactual', {}).get('inside_ellipsoid', False),
                'iterations': out.get('iterations', 1)
            })
        except Exception as e:
            results['static_script'].append({'patient_id': pid, 'error': str(e)})
        
        # Unconstrained LLM
        try:
            out = llm_baseline.run(str(pid), z, sg)
            results['unconstrained_llm'].append({
                'patient_id': pid,
                'has_ground_truth': out.get('has_ground_truth', False),
                'explanation_length': len(out.get('explanation', ''))
            })
        except Exception as e:
            results['unconstrained_llm'].append({'patient_id': pid, 'error': str(e)})
    
    # Aggregate metrics
    def agg(res_list, key, default=0):
        vals = [r.get(key, default) for r in res_list if 'error' not in r]
        return np.mean(vals) if vals else default
    
    summary = {
        'agentic': {
            'n_evaluated': len([r for r in results['agentic'] if 'error' not in r]),
            'cf_reach_rate': agg(results['agentic'], 'cf_reached_target'),
            'cf_ellipsoid_rate': agg(results['agentic'], 'cf_inside_ellipsoid'),
            'mean_delta_norm': agg(results['agentic'], 'cf_delta_norm'),
            'mean_iterations': agg(results['agentic'], 'iterations'),
            'verifier_pass_rate': agg(results['agentic'], 'verifier_pass'),
            'abstain_rate': agg(results['agentic'], 'abstain')
        },
        'static_script': {
            'n_evaluated': len([r for r in results['static_script'] if 'error' not in r]),
            'cf_reach_rate': agg(results['static_script'], 'cf_reached_target'),
            'cf_ellipsoid_rate': agg(results['static_script'], 'cf_inside_ellipsoid'),
            'mean_iterations': agg(results['static_script'], 'iterations'),
            'abstain_rate': agg(results['static_script'], 'abstain')
        },
        'unconstrained_llm': {
            'n_evaluated': len([r for r in results['unconstrained_llm'] if 'error' not in r]),
            'ground_truth_rate': agg(results['unconstrained_llm'], 'has_ground_truth'),
            'mean_explanation_length': agg(results['unconstrained_llm'], 'explanation_length')
        }
    }
    
    return {'detailed': results, 'summary': summary}


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGENTIC AI FRAMEWORK - EVALUATION")
    print("=" * 70)
    
    # Load artifacts
    with open(RESULTS_DIR / "clean_pipeline_artifacts.pkl", 'rb') as f:
        artifacts = pickle.load(f)
    
    # Load pre-saved test arrays (exact test set from pipeline)
    X_test = np.load(RESULTS_DIR / "X_test.npy")
    sg_test = np.load(RESULTS_DIR / "sg_test.npy", allow_pickle=True)
    patient_ids_test = np.load(RESULTS_DIR / "patient_ids_test.npy", allow_pickle=True)
    
    # Get high-risk patients from evidence_list (which has correct risk values)
    evidence_list = artifacts['evidence_list']
    high_risk_evidence = [ev for ev in evidence_list if ev['predicted_risk'] > 0.5]
    print(f"Found {len(high_risk_evidence)} high-risk patients from evidence_list")
    
    # Map to test set indices using patient IDs
    eval_data = []
    for ev in high_risk_evidence:
        idx = np.where(patient_ids_test == ev['subject_id'])[0]
        if len(idx) > 0:
            local_idx = idx[0]
            eval_data.append({
                'patient_id': ev['subject_id'],
                'z': X_test[local_idx],
                'subgroup': sg_test[local_idx],
                'ground_truth_risk': ev['predicted_risk'],
                'ground_truth_cf': ev['counterfactual']
            })
    
    print(f"Evaluating on {len(eval_data)} high-risk patients from test set...")
    
    orchestrator = OrchestratorAgent(artifacts)
    
    agentic_results = []
    for d in eval_data:
        state = orchestrator.run(str(d['patient_id']), d['z'], d['subgroup'])
        agentic_results.append({
            'patient_id': d['patient_id'],
            'subgroup': d['subgroup'],
            'risk': state.predicted_risk,
            'ground_truth_risk': d['ground_truth_risk'],
            'abstain': state.abstain,
            'cf_history': state.counterfactual_history,
            'verifier_feedback': state.verifier_feedback,
            'final_explanation': state.final_explanation
        })
    
    # Save results
    with open(RESULTS_DIR / "agentic_evaluation.json", 'w') as f:
        json.dump({
            'agentic_results': agentic_results,
            'n_patients': len(eval_data)
        }, f, indent=2, default=str)
    
    print("\nAgentic Results Summary:")
    for r in agentic_results:
        cf = r['cf_history'][-1] if r['cf_history'] else {}
        p_final = cf.get('p_final_calibrated', 'N/A')
        p_final_str = f"{p_final:.1%}" if isinstance(p_final, (int, float)) else str(p_final)
        inside = cf.get('inside_ellipsoid', 'N/A')
        inside_str = str(inside)
        vf = r['verifier_feedback'][-1] if r['verifier_feedback'] else 'N/A'
        vf_str = str(vf)
        print(f"  Patient {r['patient_id']} ({r['subgroup']}): risk={r['risk']:.1%} (GT={r['ground_truth_risk']:.1%}), "
              f"cf_reached={p_final_str}, "
              f"inside={inside_str}, "
              f"verifier={vf_str}")
    
    print("\nEvaluation saved to results/agentic_evaluation.json")
    print("=" * 70)