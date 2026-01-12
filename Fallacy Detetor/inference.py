# inference.py
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel, PeftConfig
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


# ----------------------------
# Label normalization (长期方案：输出统一用下划线key)
# ----------------------------
LABEL_ALIASES = {
    "appeal to authority": "appeal_to_authority",
    "appeal_to_authority": "appeal_to_authority",
    "appeal to majority": "appeal_to_majority",
    "appeal_to_majority": "appeal_to_majority",
    "appeal to nature": "appeal_to_nature",
    "appeal_to_nature": "appeal_to_nature",
    "appeal to tradition": "appeal_to_tradition",
    "appeal_to_tradition": "appeal_to_tradition",
    "appeal to worse problems": "appeal_to_worse_problems",
    "appeal_to_worse_problems": "appeal_to_worse_problems",
    "false dilemma": "false_dilemma",
    "false_dilemma": "false_dilemma",
    "hasty generalization": "hasty_generalization",
    "hasty_generalization": "hasty_generalization",
    "slippery slope": "slippery_slope",
    "slippery_slope": "slippery_slope",
    "whataboutism": "appeal_to_worse_problems",
}


def normalize_label(s: str) -> str:
    if not s:
        return s
    x = s.strip().lower()
    x = x.replace("-", " ")
    x = " ".join(x.split())
    if x in LABEL_ALIASES:
        return LABEL_ALIASES[x]
    # heuristic: spaces -> underscores
    if " " in x:
        return x.replace(" ", "_")
    return x


# ----------------------------
# Config
# ----------------------------
@dataclass
class InferConfig:
    model_dir: str = "outputs/fallacy_detector/best"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    dtype: str = "auto"   # "auto" | "bf16" | "fp16" | "fp32"

    max_length: int = 256     # should match training max_length
    stride: int = 128         # overlap tokens (window shift = max_length-2 - stride)
    batch_size: int = 8

    topk: int = 3
    min_text_chars: int = 1


def pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    # auto
    if torch.cuda.is_available():
        # bf16 preferred on A100/H100 etc., fp16 otherwise
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# ----------------------------
# Model loading (supports full checkpoint or PEFT adapter)
# ----------------------------
def load_model_and_tokenizer(model_dir: str, device: torch.device, torch_dtype: torch.dtype):
    """
    Load tokenizer + sequence classification model from a local directory.

    Supports:
      - Full model saved in model_dir (config + weights present)
      - PEFT adapter saved in model_dir (adapter_config.json present)
        -> loads base model from peft_cfg.base_model_name_or_path, then applies adapter.

    Notes:
      - We force `local_files_only=True` when reading from model_dir to avoid HF Hub validation.
      - For PEFT base model, `base_model_name_or_path` might be a Hub repo (e.g., mistralai/Mistral-7B-v0.1).
        If your environment cannot access HF Hub reliably, pre-download the base model to a local path and
        set base_model_name_or_path to that local path when training/saving, or manually edit it in adapter config.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"model_dir is not a local directory: {model_dir}")

    # --- Tokenizer: treat model_dir as local folder ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,  # ✅ critical: prevent treating path as HF repo_id
    )

    # Case 1: full model saved in model_dir
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,  # ✅ critical
        )
        model.to(device)
        model.eval()
        return model, tokenizer

    except Exception as e_full:
        # Case 2: PEFT adapter directory
        if not _PEFT_AVAILABLE:
            raise RuntimeError(
                f"Failed to load full model from local dir: {model_dir}\n"
                f"and peft is not available.\n"
                f"Original error: {repr(e_full)}"
            )

        # PeftConfig expects adapter_config.json in model_dir
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base_name = peft_cfg.base_model_name_or_path

        # Base model might be a HF repo_id OR a local path.
        # If it's a local path, we can force local_files_only=True.
        base_is_local = isinstance(base_name, str) and os.path.isdir(base_name)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True if base_is_local else False,  # ✅ local if possible
        )

        model = PeftModel.from_pretrained(base_model, model_dir)
        model.to(device)
        model.eval()
        return model, tokenizer


# ----------------------------
# Sliding window tokenization with offsets (best-effort)
# ----------------------------
def build_windows(
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
) -> Tuple[List[Dict[str, Any]], Optional[List[Tuple[int, int]]]]:
    """
    Returns:
      windows: list of {input_ids, attention_mask, window_meta}
      offsets: optional list of (start_char, end_char) for each token in the FULL tokenization
               Only available when tokenizer supports return_offsets_mapping
    """
    # tokenization without truncation to get full ids + offsets
    # Offsets mapping exists only for fast tokenizers.
    try:
        enc_full = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc_full.get("offset_mapping", None)
    except Exception:
        enc_full = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
        )
        offsets = None

    input_ids_full = enc_full["input_ids"]
    # if tokenizer returns a list already (single example), good.
    # Some tokenizers may return nested list - normalize to list[int]
    if len(input_ids_full) > 0 and isinstance(input_ids_full[0], list):
        input_ids_full = input_ids_full[0]
        if offsets is not None and len(offsets) > 0 and isinstance(offsets[0], list):
            offsets = offsets[0]

    # Window size for content tokens (we will add special tokens later)
    content_len = max_length - 2  # reserve for BOS/EOS (or similar)
    if content_len <= 0:
        raise ValueError("max_length too small.")

    # step size (how many tokens to move each window)
    step = max(1, content_len - stride)

    windows: List[Dict[str, Any]] = []
    n = len(input_ids_full)
    if n == 0:
        return windows, offsets

    start = 0
    widx = 0
    while start < n:
        end = min(n, start + content_len)
        chunk_ids = input_ids_full[start:end]

        # Add special tokens
        # tokenizer.build_inputs_with_special_tokens handles BOS/EOS properly
        ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        attn = [1] * len(ids)

        # best-effort char span
        span = None
        if offsets is not None and len(offsets) == n:
            # offsets align to chunk_ids indices [start:end]
            span_start = offsets[start][0]
            span_end = offsets[end - 1][1]
            span = (int(span_start), int(span_end))

        windows.append({
            "input_ids": ids,
            "attention_mask": attn,
            "meta": {
                "window_index": widx,
                "token_start": start,
                "token_end": end,
                "char_span": span,  # may be None
            }
        })

        widx += 1
        if end == n:
            break
        start += step

    return windows, offsets


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def run_inference_on_windows(
    model,
    tokenizer,
    windows: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    results = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        for x in batch:
            ids = x["input_ids"]
            attn = x["attention_mask"]
            pad_len = max_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(attn)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, C]
        probs = F.softmax(logits, dim=-1)

        probs_cpu = probs.float().cpu().tolist()
        logits_cpu = logits.float().cpu().tolist()

        for j, x in enumerate(batch):
            results.append({
                "meta": x["meta"],
                "probs": probs_cpu[j],
                "logits": logits_cpu[j],
            })

    return results


def topk_from_probs(id2label: Dict[int, str], probs: List[float], k: int) -> List[Dict[str, float]]:
    pairs = [(i, float(p)) for i, p in enumerate(probs)]
    pairs.sort(key=lambda t: t[1], reverse=True)
    out = []
    for i, p in pairs[:k]:
        lab = normalize_label(id2label.get(i, str(i)))
        out.append({lab: p})
    return out


def select_most_severe_window(
    id2label: Dict[int, str],
    window_results: List[Dict[str, Any]],
    topk: int,
) -> Dict[str, Any]:
    """
    Severity definition (MVP, simple and effective):
      severity_score = max_class_probability in that window.
    Choose the window with the largest severity_score.
    """
    best = None
    for r in window_results:
        probs = r["probs"]
        max_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        max_p = float(probs[max_idx])
        if best is None or max_p > best["severity_score"]:
            best = {
                "severity_score": max_p,
                "pred_id": max_idx,
                "pred_label": normalize_label(id2label.get(max_idx, str(max_idx))),
                "confidence": max_p,
                "topk": topk_from_probs(id2label, probs, topk),
                "meta": r["meta"],
                "probs": probs,
            }
    return best if best is not None else {}


def build_structured_output(
    text: str,
    best_window: Dict[str, Any],
    id2label: Dict[int, str],
    tokenizer,
) -> Dict[str, Any]:
    if not best_window:
        return {
            "text_length": len(text),
            "num_windows": 0,
            "best_span": None,
            "fallacy_type": None,
            "confidence": 0.0,
            "topk": [],
        }

    char_span = best_window["meta"].get("char_span", None)
    token_start = best_window["meta"].get("token_start")
    token_end = best_window["meta"].get("token_end")

    snippet = None
    if char_span is not None:
        s, e = char_span
        snippet = text[s:e]
    else:
        # fallback: decode this window's tokens (approx)
        # best effort: rebuild ids from token_start/token_end without special tokens is hard here,
        # so we use window index decode from stored meta is not possible. Return None snippet.
        snippet = None

    return {
        "text_length": len(text),
        "num_windows": None,  # fill later by caller
        "best_span": {
            "window_index": best_window["meta"]["window_index"],
            "char_start": char_span[0] if char_span else None,
            "char_end": char_span[1] if char_span else None,
            "token_start": token_start,
            "token_end": token_end,
            "snippet": snippet,
        },
        "fallacy_type": best_window["pred_label"],
        "confidence": best_window["confidence"],
        "topk": best_window["topk"],
        # extra for debugging / downstream use
        "severity_score": best_window["severity_score"],
    }


def infer_text(
    model,
    tokenizer,
    text: str,
    cfg: InferConfig,
    device: torch.device,
) -> Dict[str, Any]:
    text = text.strip()
    if len(text) < cfg.min_text_chars:
        return {
            "text_length": len(text),
            "num_windows": 0,
            "best_span": None,
            "fallacy_type": None,
            "confidence": 0.0,
            "topk": [],
        }

    windows, _offsets = build_windows(
        tokenizer=tokenizer,
        text=text,
        max_length=cfg.max_length,
        stride=cfg.stride,
    )

    if len(windows) == 0:
        return {
            "text_length": len(text),
            "num_windows": 0,
            "best_span": None,
            "fallacy_type": None,
            "confidence": 0.0,
            "topk": [],
        }

    window_results = run_inference_on_windows(
        model=model,
        tokenizer=tokenizer,
        windows=windows,
        device=device,
        batch_size=cfg.batch_size,
    )

    # id2label from config (preferred)
    if hasattr(model, "config") and getattr(model.config, "id2label", None):
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        # fallback: numeric labels
        num_labels = len(window_results[0]["probs"])
        id2label = {i: str(i) for i in range(num_labels)}

    best = select_most_severe_window(
        id2label=id2label,
        window_results=window_results,
        topk=cfg.topk,
    )
    out = build_structured_output(
        text=text,
        best_window=best,
        id2label=id2label,
        tokenizer=tokenizer,
    )
    out["num_windows"] = len(windows)
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="outputs/fallacy_detector/best")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)

    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--text_file", type=str, default=None)
    ap.add_argument("--jsonl_in", type=str, default=None)
    ap.add_argument("--jsonl_out", type=str, default=None)
    return ap.parse_args()


def read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()
    cfg = InferConfig(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        topk=args.topk,
    )

    device = pick_device(cfg.device)
    torch_dtype = pick_dtype(cfg.dtype)

    model, tokenizer = load_model_and_tokenizer(cfg.model_dir, device=device, torch_dtype=torch_dtype)

    # Single text
    if args.text is not None or args.text_file is not None:
        text = args.text if args.text is not None else read_text_from_file(args.text_file)
        out = infer_text(model, tokenizer, text, cfg, device=device)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # Batch jsonl: each line must contain {"text": "..."} (and can contain extra fields)
    if args.jsonl_in and args.jsonl_out:
        with open(args.jsonl_in, "r", encoding="utf-8") as fin, open(args.jsonl_out, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                pred = infer_text(model, tokenizer, text, cfg, device=device)
                obj["fallacy_signal"] = pred
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote -> {args.jsonl_out}")
        return

    raise SystemExit("Provide --text/--text_file or (--jsonl_in and --jsonl_out).")


if __name__ == "__main__":
    main()
