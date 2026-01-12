from dataclasses import dataclass


@dataclass
class TrainConfig:
    # -------------------------
    # Paths
    # -------------------------
    data_dir: str = "cocolofa"
    output_dir: str = "outputs/fallacy_detector"

    # -------------------------
    # Model
    # -------------------------
    model_name: str = "mistralai/Mistral-7B-v0.1"
    max_length: int = 512
    use_article: bool = False
    drop_unknown_label: bool = False

    # -------------------------
    # QLoRA / LoRA
    # -------------------------
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # -------------------------
    # Training
    # -------------------------
    seed: int = 42
    epochs: int = 3
    lr: float = 2e-4
    wd: float = 0.0

    train_bs: int = 1
    eval_bs: int = 1
    grad_accum: int = 16

    warmup_ratio: float = 0.03
    logging_steps: int = 50

    eval_strategy: str = "epoch"   # "no" | "steps" | "epoch"
    save_strategy: str = "epoch"   # "no" | "steps" | "epoch"
    save_total_limit: int = 2

    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True

    # Only used if *_strategy == "steps"
    eval_steps: int | None = None
    save_steps: int | None = None


# ------------------------------------------------------------
# Presets (optional): easy experiment switching
# ------------------------------------------------------------

def preset_baseline() -> TrainConfig:
    """
    A lightweight baseline without 4-bit.
    Useful for sanity-checking pipeline.
    """
    cfg = TrainConfig()
    cfg.use_4bit = False
    cfg.train_bs = 2
    cfg.eval_bs = 2
    cfg.grad_accum = 1
    cfg.bf16 = True
    cfg.gradient_checkpointing = False
    return cfg


def preset_qlora_fast() -> TrainConfig:
    """
    QLoRA preset for 7B models under limited GPU memory.
    """
    cfg = TrainConfig()
    cfg.use_4bit = True
    cfg.max_length = 256
    cfg.train_bs = 1
    cfg.eval_bs = 1
    cfg.grad_accum = 16
    cfg.bf16 = True
    cfg.gradient_checkpointing = True
    return cfg


def preset_with_article() -> TrainConfig:
    """
    Same as qlora_fast, but prepend article content for each comment.
    """
    cfg = preset_qlora_fast()
    cfg.use_article = True
    return cfg
