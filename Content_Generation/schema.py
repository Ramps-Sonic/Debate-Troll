from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContentGenConfig:
    # LLM
    model_name: str = "qwen-max"

    # generation
    temperature: float = 0.5
    max_tokens: int = 1200

    # safety / control
    refuse_personal_attack: bool = True
    refuse_hate: bool = True

    # output preferences
    default_language: str = "zh"  # "zh" | "en"


@dataclass
class ContentGenRequest:
    text: str
    fallacy_type: str
    confidence: float
    strategy_plan: Dict[str, Any]

    style: str = "学术理性 (Academic)"
    intensity: int = 5
    audience: str = "专业评委"

    context: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RebuttalDraft:
    title: str
    short: str
    full: str
    key_points: List[str]
    questions: List[str]
    evidence_needs: List[str]
    risk_notes: List[str]
    citations: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


@dataclass
class ContentGenOutput:
    rebuttal: RebuttalDraft
    # Bonus A
    satire: Optional[RebuttalDraft] = None
