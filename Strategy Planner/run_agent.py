#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from config import StrategyPlannerConfig  # :contentReference[oaicite:3]{index=3}
from schema import StrategyRequest, FallacySignal  # :contentReference[oaicite:4]{index=4}
from llm_client import LLMClient  # :contentReference[oaicite:5]{index=5}
from strategy_planner import StrategyPlanner  # :contentReference[oaicite:6]{index=6}


def _jsonable(obj: Any) -> Any:
    """Convert common python objects into JSON-serializable values."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump())
    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return _jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj))
        except Exception:
            pass
    return {"_repr": repr(obj)}


def _read_stdin_json() -> Optional[Dict[str, Any]]:
    """
    If stdin is piped, try to parse JSON from it.
    Expected schema (minimal):
      {
        "text": "...",
        "fallacy": {"fallacy_type": "...", "confidence": 0.0},
        "context": null,
        "user_goal": "win_debate",
        "constraints": {"no_personal_attack": true}
      }
    """
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return None
        raw = sys.stdin.read()
        if not raw.strip():
            return None
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse stdin as JSON: {e}")


def _build_request(payload: Dict[str, Any]) -> StrategyRequest:
    if "text" not in payload:
        raise ValueError("Missing required field: text")
    if "fallacy" not in payload or not isinstance(payload["fallacy"], dict):
        raise ValueError("Missing required field: fallacy (object)")

    f = payload["fallacy"]
    if "fallacy_type" not in f or "confidence" not in f:
        raise ValueError("fallacy must contain: fallacy_type, confidence")

    fallacy = FallacySignal(
        fallacy_type=str(f["fallacy_type"]),
        confidence=float(f["confidence"]),
        topk=f.get("topk", None),
    )

    return StrategyRequest(
        text=str(payload["text"]),
        fallacy=fallacy,
        context=payload.get("context", None),
        user_goal=str(payload.get("user_goal", "win_debate")),
        constraints=payload.get("constraints", {}) or {},
    )


def main():
    parser = argparse.ArgumentParser(description="StrategyPlanner Agent runner (single-file entry).")
    parser.add_argument("--text", type=str, default=None, help="Opponent argument text.")
    parser.add_argument("--fallacy_type", type=str, default=None, help="Detected fallacy type.")
    parser.add_argument("--confidence", type=float, default=None, help="Fallacy confidence 0..1.")
    parser.add_argument("--context", type=str, default=None, help="Optional dialogue context.")
    parser.add_argument("--user_goal", type=str, default="win_debate", help="win_debate|clarify_truth|reduce_conflict")
    parser.add_argument("--constraints", type=str, default="{}", help="JSON string of constraints dict.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    args = parser.parse_args()

    # -------------------------
    # 1) Config (WRITE-ONCE HERE)
    # -------------------------
    # IMPORTANT: You said you want no env vars. So hardcode here.
    cfg = StrategyPlannerConfig(
        backend="qwen",  # kept for consistency; StrategyPlanner itself only needs client+style fields :contentReference[oaicite:7]{index=7}
        api_key="sk-e7aeb269e8d14136bb608e15b433b6b5",
        api_base="https://dashscope.aliyuncs.com/compatible-mode",  # no trailing /v1 needed :contentReference[oaicite:8]{index=8}
        model_name="qwen-plus",
        temperature=0.4,
        max_tokens=900,
        n_candidates=3,
        use_critic=True,
        debate_style="rational",
        audience="neutral_judge",
        verbose=False,
    )

    # -------------------------
    # 2) Build request (stdin JSON OR CLI args)
    # -------------------------
    payload = _read_stdin_json()
    if payload is not None:
        req = _build_request(payload)
    else:
        # CLI mode requires these three at minimum
        if args.text is None or args.fallacy_type is None or args.confidence is None:
            raise RuntimeError(
                "No stdin JSON detected. For CLI mode you must provide: --text --fallacy_type --confidence\n"
                "Or pipe JSON into stdin."
            )
        try:
            constraints = json.loads(args.constraints)
            if not isinstance(constraints, dict):
                raise ValueError("constraints must be a JSON object (dict)")
        except Exception as e:
            raise RuntimeError(f"Failed to parse --constraints as JSON object: {e}")

        req = StrategyRequest(
            text=args.text,
            fallacy=FallacySignal(fallacy_type=args.fallacy_type, confidence=float(args.confidence)),
            context=args.context,
            user_goal=args.user_goal,
            constraints=constraints,
        )

    # -------------------------
    # 3) Run planner
    # -------------------------
    client = LLMClient(api_key=cfg.api_key, api_base=cfg.api_base, model=cfg.model_name)  # :contentReference[oaicite:9]{index=9}
    planner = StrategyPlanner(cfg, client)  # :contentReference[oaicite:10]{index=10}
    out = planner.plan(req)

    result = {
        "request": _jsonable(req),
        "result": _jsonable(out),
        "best_plan": _jsonable(getattr(out, "plan", None)),
        "scores": _jsonable(getattr(out, "scores", None)),
        "critic_rationale": _jsonable(getattr(out, "rationale", None)),
    }

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
