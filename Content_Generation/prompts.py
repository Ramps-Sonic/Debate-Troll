from __future__ import annotations

import json
from typing import Any, Dict, List


def build_rebuttal_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate a rebuttal draft based on a strategy plan.

    Output MUST be strict JSON only.
    """

    constraints = payload.get("constraints", {}) or {}
    must_cite = bool(constraints.get("must_cite_evidence", False))
    no_new_facts = bool(constraints.get("no_new_facts", False))

    strategy_plan = payload.get("strategy_plan") or {}
    evidence_snippets = payload.get("evidence_snippets") or strategy_plan.get("retrieved_evidence") or []

    system = (
        "你是 ContentGenerator（反驳内容生成智能体）。"
        "你要基于输入文本、谬误类型、策略计划以及检索证据，生成可直接用于辩论的反驳稿。"
        "要求：不做人身攻击；语言风格服从 style；强度服从 intensity；受众服从 audience。"
        " 如果输入是英文，请务必用英文输出。"
        + (" 事实约束：不得引入新的事实性主张；只能复述或推导自【检索证据】与用户输入。" if no_new_facts else "")
        + (" 引用约束：正文中涉及证据性表述时，必须在句末用 [E1]/[E2] 形式标注来源（仅使用提供的证据编号）。" if must_cite else "")
        + " 如果提供了检索证据（retrieved_evidence），你需要优先用它们支撑关键句；若证据不足，必须明确写‘需要进一步核查/需要数据支持’，不要编造。"
        + " 只输出严格 JSON，不能输出多余文本。"
    )

    user = {
        "task": "Generate rebuttal draft",
        "input_text": payload["text"],
        "fallacy": {
            "type": payload["fallacy_type"],
            "confidence": payload["confidence"],
        },
        "strategy_plan": strategy_plan,
        "retrieved_evidence": evidence_snippets,
        "style": payload.get("style", "学术理性 (Academic)"),
        "intensity": int(payload.get("intensity", 5)),
        "audience": payload.get("audience", "专业评委"),
        "context": payload.get("context"),
        "constraints": payload.get("constraints", {}),
        "output_schema": {
            "title": "string",
            "short": "string (<= 120 chars)",
            "full": "string (2-5 paragraphs)",
            "key_points": ["string"],
            "questions": ["string"],
            "evidence_needs": ["string"],
            "risk_notes": ["string"],
            "citations": ["string (e.g., E1, E2)"]
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_satire_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Rewrite an existing rebuttal into satire style.

    Key constraints:
      - do NOT add new factual claims
      - keep core logic unchanged
      - use rhetorical questions / exaggeration / irony
      - no hate, no slurs, no harassment
      - must be FULL LENGTH (not summarized), very spicy and sarcastic
    """

    system = (
        "你是 SatireBot（讽刺化表达生成）。"
        "你的任务是将输入的反驳内容改写为一种极度辛辣、阴阳怪气、充满讽刺意味的风格。"
        "要求：\n"
        "1. 保持原有的逻辑反驳点，不要改变核心论据。\n"
        "2. 语气要刻薄、犀利，多用反问、夸张、比喻和幽默的嘲讽。\n"
        "3. 必须输出完整长度的文本（2-5段），不要简写，尽情发挥嘲讽的艺术。\n"
        "4. 严禁人身攻击、脏话或仇恨言论，要做到‘骂人不带脏字’的高级黑。\n"
        "5. 你的目标是让对方的观点显得极其荒谬可笑。\n"
        "只输出严格 JSON。"
    )

    user = {
        "task": "Rewrite rebuttal into satire style (Full Length & Spicy)",
        "original": {
            "title": payload["title"],
            "short": payload["short"],
            "full": payload["full"],
            "key_points": payload.get("key_points", []),
        },
        "intensity": 10,  # Max intensity for satire
        "constraints": payload.get("constraints", {}),
        "output_schema": {
            "title": "string",
            "short": "string (<= 140 Chinese chars)",
            "full": "string (FULL LENGTH, 500+ chars, extremely sarcastic)",
            "key_points": ["string"],
            "questions": ["string"],
            "evidence_needs": ["string"],
            "risk_notes": ["string"],
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
