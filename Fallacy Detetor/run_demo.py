import os
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from config import StrategyPlannerConfig
from schema import StrategyRequest, FallacySignal
from llm_client import build_client_from_env
from strategy_planner import StrategyPlanner


def _jsonable(obj: Any):
    """
    Best-effort conversion to JSON-serializable objects.
    Supports: dataclass, pydantic (v1/v2), dict/list/primitive.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    # dataclass
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

    # fallback: try vars
    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj))
        except Exception:
            pass

    # last resort: repr
    return {"_repr": repr(obj)}


def main():
    # 1) config
    cfg = StrategyPlannerConfig(
        backend="qwen",
        model_name=os.environ.get("LLM_MODEL", "qwen-max"),
        verbose=True,
        n_candidates=3,
        use_critic=True,
    )

    # 2) client
    client = build_client_from_env(default_model=cfg.model_name)

    # 3) planner
    planner = StrategyPlanner(cfg, client)

    # 4) demo input (你可以替换成 FallacyDetector 的输出)
    text = "如果我们允许学生用AI写作业，那么下一步学生就会完全不学习，最后教育体系必然崩溃。"
    fallacy = FallacySignal(fallacy_type="slippery_slope", confidence=0.87)

    req = StrategyRequest(
        text=text,
        fallacy=fallacy,
        context=None,
        user_goal="win_debate",
        constraints={"no_personal_attack": True, "be_concise": True},
    )

    out = planner.plan(req)

    # ---- 原样终端输出 ----
    print("\n=== Best Plan ===")
    print(out.plan)
    if getattr(out, "scores", None):
        print("\n=== Scores ===")
        print(out.scores)
    if getattr(out, "rationale", None):
        print("\n=== Critic Rationale ===")
        print(out.rationale)

    # ---- 写 JSON 文件 ----
    payload = {
        "request": _jsonable(req),
        "result": _jsonable(out),
        # 你终端最关心的几块也单独拎出来，方便下游 agent 直接取：
        "best_plan": _jsonable(getattr(out, "plan", None)),
        "scores": _jsonable(getattr(out, "scores", None)),
        "critic_rationale": _jsonable(getattr(out, "rationale", None)),
    }

    out_path = "output_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] wrote -> {out_path}")


if __name__ == "__main__":
    main()

