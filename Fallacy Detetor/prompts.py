import json
from typing import Dict, Any, List
from templates import FALLACY_TYPE_ZH, STRATEGY_TEMPLATES


def build_generator_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    payload contains:
      - text, fallacy_type, confidence, style, audience, user_goal, constraints
      - templates: from STRATEGY_TEMPLATES
    """
    fallacy_type = payload["fallacy_type"]
    zh = FALLACY_TYPE_ZH.get(fallacy_type, fallacy_type)
    tmpl = STRATEGY_TEMPLATES.get(fallacy_type, {"core": [], "moves": []})

    system = (
        "你是一个辩论反驳策略规划专家（StrategyPlanner）。"
        "你的任务不是写完整反驳稿，而是输出结构化的反驳计划（JSON），"
        "包含主要策略、攻击路径（2-4步）、风险评估、成功概率、时机建议、证据需求。"
        "输出必须是严格 JSON，不能包含多余文本。"
    )

    user = {
        "task": "Generate one rebuttal strategy plan",
        "input_text": payload["text"],
        "fallacy": {
            "type": fallacy_type,
            "type_zh": zh,
            "confidence": payload["confidence"],
        },
        "context": payload.get("context", None),
        "goal": payload.get("user_goal", "win_debate"),
        "style": payload.get("style", "rational"),
        "audience": payload.get("audience", "neutral_judge"),
        "constraints": payload.get("constraints", {}),
        "strategy_space": {
            "core_principles": tmpl.get("core", []),
            "common_moves": tmpl.get("moves", []),
        },
        "output_schema": {
            "primary_strategy": "string",
            "attack_steps": [
                {
                    "step": "int (1..4)",
                    "goal": "string",
                    "move": "string (concrete action)",
                    "example": "string (one example sentence/question)",
                    "evidence_needed": "string or null",
                }
            ],
            "risk_assessment": {
                "risk_level": "low|medium|high",
                "risks": ["string"],
                "mitigation": ["string"],
            },
            "success_probability": "float 0..1",
            "timing_advice": ["string"],
            "evidence_needs": ["string"],
            "notes": "string (optional)",
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_critic_prompt(text: str, candidates_json: List[Dict[str, Any]], fallacy_type: str, confidence: float) -> List[Dict[str, str]]:
    system = (
        "你是一个辩论策略评审（Critic）。"
        "你会评估多个策略计划，选择最优的一个。"
        "只输出严格 JSON：包含每个候选的评分，以及最终选择的索引与理由。"
        "评分维度：targeting(针对性), safety(风险低), clarity(可执行性), persuasion(说服力)。"
    )
    user = {
        "input_text": text,
        "fallacy": {"type": fallacy_type, "confidence": confidence},
        "candidates": candidates_json,
        "output_schema": {
            "scores": [
                {
                    "index": "int",
                    "targeting": "0..10",
                    "safety": "0..10",
                    "clarity": "0..10",
                    "persuasion": "0..10",
                    "overall": "0..10",
                    "short_comment": "string",
                }
            ],
            "best_index": "int",
            "reason": "string",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
