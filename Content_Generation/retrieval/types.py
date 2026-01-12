from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EvidenceSnippet:
    eid: str
    text: str
    score: float
    meta: Dict[str, Any]

    def format_for_prompt(self) -> str:
        title = str(self.meta.get("title", ""))
        source = str(self.meta.get("source", ""))
        extra = []
        if title:
            extra.append(f"title={title}")
        if source:
            extra.append(f"source={source}")
        extra_s = (" | " + ", ".join(extra)) if extra else ""
        return f"[{self.eid}] score={self.score:.4f}{extra_s}\n{self.text.strip()}"
