from dataclasses import dataclass


@dataclass
class StrategyPlannerConfig:
    # ---- LLM backend ----
    # backend: "openai" | "qwen" | "local" (你先用 qwen/openai 最省事)
    backend: str = "qwen"

    # for API backends
    api_key: str = "sk-e7aeb269e8d14136bb608e15b433b6b5"  # 建议用环境变量传入（更安全）
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode"  # 有些平台需要，例如 OpenAI-compatible 的 base URL
    model_name: str = "qwen-plus"  # 或 "gpt-4.1-mini" 等

    # ---- generation ----
    temperature: float = 0.4
    max_tokens: int = 900

    # ---- planning ----
    n_candidates: int = 3  # 生成几个候选策略，再用 critic 选一个
    use_critic: bool = True

    # ---- thresholds ----
    high_conf: float = 0.80
    mid_conf: float = 0.55

    # ---- safety / style ----
    debate_style: str = "rational"  # "rational" | "aggressive" | "friendly"
    audience: str = "neutral_judge"  # "neutral_judge" | "opponent" | "public"

    # ---- logging ----
    verbose: bool = True
