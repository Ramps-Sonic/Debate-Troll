# run_agent.py
import json
import argparse
from typing import Any, Dict, Optional

import torch

from inference import (
    InferConfig,
    pick_device,
    pick_dtype,
    load_model_and_tokenizer,
    infer_text,
)

# ----------------------------
# Public API for AG2 integration
# ----------------------------
def predict_fallacy(
    text: str,
    model_dir: str = "outputs/fallacy_detector/best",
    device: str = "auto",
    dtype: str = "auto",
    max_length: int = 256,
    stride: int = 128,
    batch_size: int = 8,
    topk: int = 3,
    min_text_chars: int = 1,
) -> Dict[str, Any]:
    """
    A thin wrapper around inference.py for easy integration.

    Returns a dict:
      {
        "text_length": ...,
        "num_windows": ...,
        "best_span": {...},
        "fallacy_type": "appeal_to_authority" | ... | None,
        "confidence": float,
        "topk": [{label: prob}, ...],
        ...
      }
    """
    cfg = InferConfig(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        topk=topk,
        min_text_chars=min_text_chars,
    )

    dev = pick_device(cfg.device)
    torch_dtype = pick_dtype(cfg.dtype)

    model, tokenizer = load_model_and_tokenizer(cfg.model_dir, device=dev, torch_dtype=torch_dtype)
    return infer_text(model, tokenizer, text, cfg, device=dev)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Fallacy Detection Agent (packaged entrypoint)")
    ap.add_argument("--model_dir", type=str, default="outputs/fallacy_detector/best")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min_text_chars", type=int, default=1)

    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--text_file", type=str, default=None)
    ap.add_argument("--jsonl_in", type=str, default=None)
    ap.add_argument("--jsonl_out", type=str, default=None)
    return ap.parse_args()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()

    # build cfg
    cfg = InferConfig(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        topk=args.topk,
        min_text_chars=args.min_text_chars,
    )

    device = pick_device(cfg.device)
    torch_dtype = pick_dtype(cfg.dtype)
    model, tokenizer = load_model_and_tokenizer(cfg.model_dir, device=device, torch_dtype=torch_dtype)

    # single
    if args.text is not None or args.text_file is not None:
        text = args.text if args.text is not None else read_text(args.text_file)
        out = infer_text(model, tokenizer, text, cfg, device=device)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # jsonl batch
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
