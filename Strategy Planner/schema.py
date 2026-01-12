from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class FallacySignal:
    fallacy_type: str
    confidence: float
    topk: Optional[List[Dict[str, float]]] = None  # e.g. [{"false dilemma":0.7}, {"slippery slope":0.2}]


@dataclass
class StrategyRequest:
    text: str  # 对方论证文本（句/段/整段都行）
    fallacy: FallacySignal
    context: Optional[str] = None  # 可选：对话上下文
    user_goal: str = "win_debate"  # win_debate | clarify_truth | reduce_conflict
    constraints: Dict[str, Any] = field(default_factory=dict)  # e.g. {"no_personal_attack": True}


@dataclass
class StrategyPlan:
    primary_strategy: str
    attack_steps: List[Dict[str, Any]]  # 每步：goal, move, example, evidence_needed
    risk_assessment: Dict[str, Any]     # risk_level, risks, mitigation
    success_probability: float          # 0-1
    timing_advice: List[str]
    evidence_needs: List[str]
    notes: Optional[str] = None


@dataclass
class CandidatePlan:
    plan: StrategyPlan
    scores: Optional[Dict[str, float]] = None  # critic scores
    rationale: Optional[str] = None
