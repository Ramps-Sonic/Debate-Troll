# FallacyDetector/merge_lora.py
import os
import argparse
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

try:
    from safetensors.torch import load_file as safe_load_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


def infer_num_labels_from_adapter(adapter_dir: str) -> int:
    """
    Try to infer num_labels from adapter_model.safetensors.
    We look for a key that ends with 'score.modules_to_save.default.weight'
    or contains '.score.' weight.
    """
    if not _HAS_SAFETENSORS:
        raise RuntimeError("safetensors is required. Please `pip install safetensors`.")

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = safe_load_file(path, device="cpu")

    # most common key from your error:
    # base_model.model.score.modules_to_save.default.weight
    cand_keys = [k for k in sd.keys() if k.endswith("score.modules_to_save.default.weight")]
    if not cand_keys:
        # fallback: any score weight
        cand_keys = [k for k in sd.keys() if k.endswith("score.weight")]

    if not cand_keys:
        raise RuntimeError("Cannot find score weight in adapter_model.safetensors to infer num_labels.")

    w = sd[cand_keys[0]]
    return int(w.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--force_num_labels", type=int, default=None, help="override inferred num_labels")
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    peft_cfg = PeftConfig.from_pretrained(args.adapter_dir, local_files_only=True)
    base_name = peft_cfg.base_model_name_or_path
    print("[Base repo_id]", base_name)

    # infer num_labels from adapter weights
    inferred = infer_num_labels_from_adapter(args.adapter_dir)
    num_labels = args.force_num_labels if args.force_num_labels is not None else inferred
    print(f"[num_labels] inferred={inferred} use={num_labels}")

    # Load base model OFFLINE (from cache only)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=num_labels,            # ✅ critical fix
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,            # ✅ no network
    )

    # Load tokenizer OFFLINE (your adapter dir has tokenizer files, so this is safe)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        use_fast=True,
        local_files_only=True,
    )

    # Attach adapter
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        local_files_only=True,            # ✅ no network
    )

    # Merge and save
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)

    print("[OK] merged model saved to:", args.out_dir)


if __name__ == "__main__":
    main()

