import os
import math
import argparse
from typing import Dict, Optional

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset import CoCoLoFaDataset
from model import create_fallacy_model

from config import (
    TrainConfig,
    preset_baseline,
    preset_qlora_fast,
    preset_with_article,
)


# ============================================================
# Helpers
# ============================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = float((preds == labels).mean())

    try:
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(labels, preds, average="macro"))
    except Exception:
        macro_f1 = float("nan")

    return {"acc": acc, "macro_f1": macro_f1}


def try_load_split(
    cfg: TrainConfig,
    split: str,
    tokenizer,
) -> Optional[CoCoLoFaDataset]:
    """
    Tries to load <cfg.data_dir>/<split>.json. If not found, returns None.
    """
    path = os.path.join(cfg.data_dir, f"{split}.json")
    if not os.path.isfile(path):
        return None
    return CoCoLoFaDataset(
        path=cfg.data_dir,
        split=split,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        use_article=cfg.use_article,
        drop_unknown_label=cfg.drop_unknown_label,
    )


def load_cfg(preset: str) -> TrainConfig:
    if preset == "baseline":
        return preset_baseline()
    if preset == "qlora_fast":
        return preset_qlora_fast()
    if preset == "with_article":
        return preset_with_article()
    raise ValueError(f"Unknown preset: {preset}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Minimal CLI
    parser.add_argument(
        "--preset",
        type=str,
        default="qlora_fast",
        choices=["baseline", "qlora_fast", "with_article"],
        help="Training preset defined in config.py",
    )

    # Optional overrides
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    # A few common overrides (optional)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train_bs", type=int, default=None)
    parser.add_argument("--eval_bs", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    args = parser.parse_args()

    # Load preset config
    cfg = load_cfg(args.preset)

    # Apply overrides
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.model_name is not None:
        cfg.model_name = args.model_name

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.train_bs is not None:
        cfg.train_bs = args.train_bs
    if args.eval_bs is not None:
        cfg.eval_bs = args.eval_bs
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    if args.max_length is not None:
        cfg.max_length = args.max_length

    # Seed + output dir
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("========== TrainConfig ==========")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")
    print("=================================")

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # Some LLMs don't have pad_token; use eos as padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    train_dataset = try_load_split(cfg, "train", tokenizer)
    if train_dataset is None:
        raise FileNotFoundError(f"Missing train split: {os.path.join(cfg.data_dir, 'train.json')}")

    dev_dataset = try_load_split(cfg, "dev", tokenizer)
    if dev_dataset is None:
        dev_dataset = try_load_split(cfg, "val", tokenizer)

    test_dataset = try_load_split(cfg, "test", tokenizer)

    print(f"[Data] train: {len(train_dataset)}")
    if dev_dataset is not None:
        print(f"[Data] dev/val: {len(dev_dataset)}")
    if test_dataset is not None:
        print(f"[Data] test: {len(test_dataset)}")

    # ------------------------------------------------------------
    # Model (LoRA / QLoRA)
    # ------------------------------------------------------------
    model = create_fallacy_model(
        model_name=cfg.model_name,
        use_4bit=cfg.use_4bit,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=None,
        device_map="auto",
    )

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------
    eval_steps = cfg.eval_steps
    save_steps = cfg.save_steps

    if cfg.eval_strategy == "steps" and eval_steps is None:
        # heuristic: evaluate ~4 times per epoch
        # Note: steps_per_epoch depends on effective batch size
        world_size = max(1, torch.cuda.device_count())
        steps_per_epoch = math.ceil(len(train_dataset) / (cfg.train_bs * world_size))
        eval_steps = max(50, steps_per_epoch // 4)

    if cfg.save_strategy == "steps" and save_steps is None:
        save_steps = eval_steps

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        weight_decay=cfg.wd,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_grad_norm=1.0,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=(dev_dataset is not None and cfg.eval_strategy != "no"),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to="none",
        remove_unused_columns=False,  # important: keep tokenized fields + labels
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if dev_dataset is not None else None,
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    trainer.train()
    trainer.save_state()

    # Save last
    last_dir = os.path.join(cfg.output_dir, "last")
    os.makedirs(last_dir, exist_ok=True)
    trainer.save_model(last_dir)
    tokenizer.save_pretrained(last_dir)
    print(f"[Save] last model -> {last_dir}")

    # Save best
    if dev_dataset is not None and training_args.load_best_model_at_end:
        best_dir = os.path.join(cfg.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"[Save] best model -> {best_dir}")

    # Optional test evaluation
    if test_dataset is not None:
        print("[Eval] running on test split...")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        print("[Test metrics]", metrics)

    print("Done.")


if __name__ == "__main__":
    main()
