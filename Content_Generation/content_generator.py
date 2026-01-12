from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional

from llm_client import LLMClient
from prompts import build_rebuttal_prompt, build_satire_prompt
from schema import ContentGenConfig, ContentGenRequest, ContentGenOutput, RebuttalDraft
# Optional: import retriever type for type hinting (use string or conditional import to avoid circular dependency issues if any)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retrieval.colbertv2_retriever import ColBERTv2Retriever


def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()
    return json.loads(s)


def _rule_based_satire(draft: RebuttalDraft, intensity: int) -> RebuttalDraft:
    """Fallback satire (no LLM). Keeps logic, only tweaks tone."""

    intens = max(0, min(10, int(intensity)))
    prefix = "（讽刺版）"
    title = prefix + draft.title

    def add_irony(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        if intens >= 7:
            return s + "——听起来真是‘无懈可击’呢？"
        if intens >= 4:
            return s + "——这逻辑也太‘丝滑’了吧？"
        return s + "——我们先别急着下结论，好吗？"

    short = add_irony(draft.short)
    full = draft.full
    # very light touch: add rhetorical questions without changing claims
    full = full.replace("。", "。难道不是吗？", 1) if "。" in full else full

    questions = list(draft.questions)
    if not questions:
        questions = ["这一步的必然性证据在哪里？", "中间环节为什么不能被约束/阻断？"]

    return RebuttalDraft(
        title=title,
        short=short,
        full=full,
        key_points=draft.key_points,
        questions=questions,
        evidence_needs=draft.evidence_needs,
        risk_notes=(draft.risk_notes + ["讽刺强度过高可能引发反感，建议按场合降级。"])
        if intens >= 6
        else draft.risk_notes,
        citations=list(draft.citations),
    )


class ContentGenerator:
    def __init__(self, cfg: ContentGenConfig, client: Optional[LLMClient] = None, retriever: Optional['ColBERTv2Retriever'] = None):
        self.cfg = cfg
        self.client = client
        self.retriever = retriever

    def _generate_query_from_plan(self, text: str, plan: Dict[str, Any]) -> str:
        """
        Extract query keywords from strategy plan.
        Priority: evidence_needs -> key_points -> original text keywords
        """
        # 1. Try explicit evidence needs from StrategyPlanner
        needs = plan.get("evidence_needs", [])
        if needs and isinstance(needs, list):
            # Combine top 2 needs
            return " ".join(needs[:2])
        
        # 2. Try attack steps goals
        steps = plan.get("attack_steps", [])
        if steps and isinstance(steps, list):
            goals = [s.get("evidence_needed", "") for s in steps if s.get("evidence_needed")]
            if goals:
                 return " ".join(goals[:2])

        # 3. Fallback: Use the controversy text itself (truncated)
        # Simple stop-word removal could happen here, but LLM usually handles natural language queries well enough with ColBERT
        return text[:100]

    def generate(self, req: ContentGenRequest, enable_satire: bool = False) -> ContentGenOutput:
        if self.client is None:
            raise RuntimeError("ContentGenerator requires an LLMClient (set env vars and pass client).")

        evidence_snippets = []
        
        # === RAG Logic: Auto-Retrieve if retriever is present ===
        if self.retriever:
            # Generate query from the plan
            query = self._generate_query_from_plan(req.text, req.strategy_plan if isinstance(req.strategy_plan, dict) else {})
            print(f"[ContentGenerator] Generated RAG Query: {query}")
            
            # Search
            try:
                results = self.retriever.search(query, k=5)
                # Format into strings
                evidence_snippets = [f"[{e.eid}] {e.text}" for e in results]
                print(f"[ContentGenerator] Retrieved {len(evidence_snippets)} snippets.")
            except Exception as e:
                print(f"[ContentGenerator] Retrieval failed: {e}")
        # ========================================================
        
        # Allow manual override if evidence was passed in request (though less likely in full flow)
        if not evidence_snippets and isinstance(req.strategy_plan, dict):
            maybe = req.strategy_plan.get("retrieved_evidence")
            if isinstance(maybe, list):
                evidence_snippets = [str(x) for x in maybe if str(x).strip()]

        payload = {
            "text": req.text,
            "fallacy_type": req.fallacy_type,
            "confidence": float(req.confidence),
            "strategy_plan": req.strategy_plan,
            "evidence_snippets": evidence_snippets,
            "style": req.style,
            "intensity": int(req.intensity),
            "audience": req.audience,
            "context": req.context,
            "constraints": req.constraints,
        }

        raw = self.client.chat(
            build_rebuttal_prompt(payload),
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        d = _safe_json_loads(raw)

        rebuttal = RebuttalDraft(
            title=str(d.get("title", "反驳稿")),
            short=str(d.get("short", "")),
            full=str(d.get("full", "")),
            key_points=list(d.get("key_points", []) or []),
            questions=list(d.get("questions", []) or []),
            evidence_needs=list(d.get("evidence_needs", []) or []),
            risk_notes=list(d.get("risk_notes", []) or []),
            citations=list(d.get("citations", []) or []),
        )

        satire: Optional[RebuttalDraft] = None
        if enable_satire:
            try:
                sraw = self.client.chat(
                    build_satire_prompt(
                        {
                            "title": rebuttal.title,
                            "short": rebuttal.short,
                            "full": rebuttal.full,
                            "key_points": rebuttal.key_points,
                            "intensity": int(req.intensity),
                            "constraints": req.constraints,
                        }
                    ),
                    temperature=0.7,
                    max_tokens=min(1200, self.cfg.max_tokens),
                )
                sd = _safe_json_loads(sraw)
                satire = RebuttalDraft(
                    title=str(sd.get("title", "（讽刺版）" + rebuttal.title)),
                    short=str(sd.get("short", "")),
                    full=str(sd.get("full", "")),
                    key_points=list(sd.get("key_points", []) or []),
                    questions=list(sd.get("questions", []) or []),
                    evidence_needs=list(sd.get("evidence_needs", []) or []),
                    risk_notes=list(sd.get("risk_notes", []) or []),
                    citations=list(sd.get("citations", []) or []),
                )
            except Exception:
                satire = _rule_based_satire(rebuttal, intensity=int(req.intensity))

        return ContentGenOutput(rebuttal=rebuttal, satire=satire)

    @staticmethod
    def to_dict(out: ContentGenOutput) -> Dict[str, Any]:
        return asdict(out)
