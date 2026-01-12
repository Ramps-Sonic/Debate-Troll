import json
from typing import List, Dict, Any

from config import StrategyPlannerConfig
from schema import StrategyRequest, StrategyPlan, CandidatePlan
from prompts import build_generator_prompt, build_critic_prompt
from llm_client import LLMClient


def _safe_json_loads(s: str) -> Any:
    """
    Attempt to parse JSON even if the model wrapped it with code fences.
    """
    s = s.strip()
    if s.startswith("```"):
        # strip code fences
        s = s.strip("`")
        # sometimes includes "json\n"
        s = s.replace("json\n", "", 1).strip()
    return json.loads(s)


class StrategyPlanner:
    def __init__(self, cfg: StrategyPlannerConfig, client: LLMClient):
        self.cfg = cfg
        self.client = client

    def plan(self, req: StrategyRequest) -> CandidatePlan:
        payload = {
            "text": req.text,
            "context": req.context,
            "fallacy_type": req.fallacy.fallacy_type,
            "confidence": float(req.fallacy.confidence),
            "style": self.cfg.debate_style,
            "audience": self.cfg.audience,
            "user_goal": req.user_goal,
            "constraints": req.constraints,
        }

        # 1) generate candidates
        candidates: List[Dict[str, Any]] = []
        for i in range(self.cfg.n_candidates):
            msgs = build_generator_prompt(payload)
            raw = self.client.chat(
                msgs,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            plan_dict = _safe_json_loads(raw)
            candidates.append(plan_dict)

        # 2) pick best (optional)
        if self.cfg.use_critic and len(candidates) > 1:
            critic_msgs = build_critic_prompt(
                text=req.text,
                candidates_json=candidates,
                fallacy_type=req.fallacy.fallacy_type,
                confidence=float(req.fallacy.confidence),
            )
            raw_crit = self.client.chat(
                critic_msgs,
                temperature=0.2,
                max_tokens=700,
            )
            crit = _safe_json_loads(raw_crit)
            best_idx = int(crit["best_index"])
            scores_list = crit.get("scores", [])
            rationale = crit.get("reason", "")

            best_plan = StrategyPlan(**candidates[best_idx])
            # attach best candidate score if exists
            best_score = None
            for s in scores_list:
                if int(s.get("index", -1)) == best_idx:
                    best_score = {
                        "targeting": float(s.get("targeting", 0)),
                        "safety": float(s.get("safety", 0)),
                        "clarity": float(s.get("clarity", 0)),
                        "persuasion": float(s.get("persuasion", 0)),
                        "overall": float(s.get("overall", 0)),
                    }
                    break

            return CandidatePlan(plan=best_plan, scores=best_score, rationale=rationale)

        # no critic: return first
        best_plan = StrategyPlan(**candidates[0])
        return CandidatePlan(plan=best_plan)
